"""
RTXpy - Ray tracing using NVIDIA OptiX, accessible from Python.

This module provides GPU-accelerated ray-triangle intersection using
NVIDIA's OptiX ray tracing engine via the otk-pyoptix Python bindings.
"""

import os
import atexit
import struct

# CRITICAL: cupy must be imported before optix for proper CUDA context sharing
import cupy
has_cupy = True

import optix

import numpy as np


# -----------------------------------------------------------------------------
# Singleton state management
# -----------------------------------------------------------------------------

class _OptixState:
    """
    Manages the global OptiX state including device context, module, pipeline,
    shader binding table, and acceleration structure cache.
    """

    def __init__(self):
        self.context = None
        self.module = None
        self.pipeline = None
        self.raygen_pg = None
        self.miss_pg = None
        self.hit_pg = None
        self.sbt = None

        # Acceleration structure cache
        self.gas_handle = 0
        self.gas_buffer = None
        self.current_hash = 0xFFFFFFFFFFFFFFFF  # uint64(-1)

        # Device memory for params
        self.d_params = None

        # Device buffers for CPU->GPU transfers
        self.d_rays = None
        self.d_rays_size = 0
        self.d_hits = None
        self.d_hits_size = 0

        self.initialized = False

    def cleanup(self):
        """Release all OptiX and CUDA resources."""
        # Free device buffers
        self.d_params = None
        self.d_rays = None
        self.d_hits = None
        self.d_rays_size = 0
        self.d_hits_size = 0

        # Free acceleration structure
        self.gas_buffer = None
        self.gas_handle = 0
        self.current_hash = 0xFFFFFFFFFFFFFFFF

        # OptiX objects are automatically cleaned up by Python GC
        self.sbt = None
        self.pipeline = None
        self.hit_pg = None
        self.miss_pg = None
        self.raygen_pg = None
        self.module = None
        self.context = None

        self.initialized = False


_state = _OptixState()


def _cleanup_at_exit():
    """Cleanup function registered with atexit."""
    global _state
    if _state:
        _state.cleanup()


# -----------------------------------------------------------------------------
# PTX loading
# -----------------------------------------------------------------------------

def _load_ptx_file(filename: str) -> str:
    """Load PTX file from the package directory."""
    # Try the directory where this module is located
    module_dir = os.path.dirname(os.path.realpath(__file__))

    path = os.path.join(module_dir, filename)
    if os.path.exists(path):
        with open(path, 'rb') as f:
            return f.read()

    # Try data subdirectory
    path = os.path.join(module_dir, 'data', filename)
    if os.path.exists(path):
        with open(path, 'rb') as f:
            return f.read()

    raise RuntimeError(f"Failed to load {filename}")


# -----------------------------------------------------------------------------
# OptiX initialization
# -----------------------------------------------------------------------------

def _log_callback(level, tag, message):
    """OptiX log callback for debugging."""
    print(f"[OPTIX][{level}][{tag}]: {message}")


def _init_optix():
    """Initialize OptiX context, module, pipeline, and SBT."""
    global _state

    if _state.initialized:
        return

    # Create OptiX device context (uses cupy's CUDA context)
    _state.context = optix.deviceContextCreate(
        cupy.cuda.get_current_stream().ptr,
        optix.DeviceContextOptions(
            logCallbackLevel=4,
        )
    )

    # Load PTX and create module
    ptx_data = _load_ptx_file("kernel.ptx")

    module_options = optix.ModuleCompileOptions(
        maxRegisterCount=optix.COMPILE_DEFAULT_MAX_REGISTER_COUNT,
        optLevel=optix.COMPILE_OPTIMIZATION_DEFAULT,
        debugLevel=optix.COMPILE_DEBUG_LEVEL_MINIMAL,
    )

    pipeline_options = optix.PipelineCompileOptions(
        usesMotionBlur=False,
        traversableGraphFlags=optix.TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS,
        numPayloadValues=4,
        numAttributeValues=2,
        exceptionFlags=optix.EXCEPTION_FLAG_NONE,
        pipelineLaunchParamsVariableName="params",
    )

    _state.module, log = _state.context.moduleCreate(
        module_options,
        pipeline_options,
        ptx_data,
    )

    # Create program groups
    pg_options = optix.ProgramGroupOptions()

    # Raygen program group
    raygen_desc = optix.ProgramGroupDesc()
    raygen_desc.raygenModule = _state.module
    raygen_desc.raygenEntryFunctionName = "__raygen__main"
    _state.raygen_pg, log = _state.context.programGroupCreate(
        [raygen_desc],
        pg_options,
    )
    _state.raygen_pg = _state.raygen_pg[0]

    # Miss program group
    miss_desc = optix.ProgramGroupDesc()
    miss_desc.missModule = _state.module
    miss_desc.missEntryFunctionName = "__miss__miss"
    _state.miss_pg, log = _state.context.programGroupCreate(
        [miss_desc],
        pg_options,
    )
    _state.miss_pg = _state.miss_pg[0]

    # Hit group (closest hit only)
    hit_desc = optix.ProgramGroupDesc()
    hit_desc.hitgroupModuleCH = _state.module
    hit_desc.hitgroupEntryFunctionNameCH = "__closesthit__chit"
    _state.hit_pg, log = _state.context.programGroupCreate(
        [hit_desc],
        pg_options,
    )
    _state.hit_pg = _state.hit_pg[0]

    # Create pipeline
    link_options = optix.PipelineLinkOptions(
        maxTraceDepth=1,
    )

    program_groups = [_state.raygen_pg, _state.miss_pg, _state.hit_pg]
    _state.pipeline = _state.context.pipelineCreate(
        pipeline_options,
        link_options,
        program_groups,
        "",  # log
    )

    # Configure stack sizes
    stack_sizes = optix.StackSizes()
    for pg in program_groups:
        optix.util.accumulateStackSizes(pg, stack_sizes, _state.pipeline)

    (dc_from_traversal, dc_from_state, continuation) = optix.util.computeStackSizes(
        stack_sizes,
        1,  # maxTraceDepth
        0,  # maxCCDepth
        0,  # maxDCDepth
    )

    _state.pipeline.setStackSize(
        dc_from_traversal,
        dc_from_state,
        continuation,
        1,  # maxTraversableDepth
    )

    # Create shader binding table
    _create_sbt()

    # Allocate params buffer (24 bytes: handle(8) + rays_ptr(8) + hits_ptr(8))
    _state.d_params = cupy.zeros(24, dtype=cupy.uint8)

    _state.initialized = True
    atexit.register(_cleanup_at_exit)


def _create_sbt():
    """Create the shader binding table."""
    global _state

    # SBT record header size is 32 bytes (OPTIX_SBT_RECORD_HEADER_SIZE)
    # We use empty data records, so total size is just the header
    header_size = optix.SBT_RECORD_HEADER_SIZE

    # Pack raygen record
    raygen_record = bytearray(header_size)
    optix.sbtRecordPackHeader(_state.raygen_pg, raygen_record)
    d_raygen = cupy.array(np.frombuffer(raygen_record, dtype=np.uint8))

    # Pack miss record
    miss_record = bytearray(header_size)
    optix.sbtRecordPackHeader(_state.miss_pg, miss_record)
    d_miss = cupy.array(np.frombuffer(miss_record, dtype=np.uint8))

    # Pack hit group record
    hit_record = bytearray(header_size)
    optix.sbtRecordPackHeader(_state.hit_pg, hit_record)
    d_hit = cupy.array(np.frombuffer(hit_record, dtype=np.uint8))

    _state.sbt = optix.ShaderBindingTable(
        raygenRecord=d_raygen.data.ptr,
        missRecordBase=d_miss.data.ptr,
        missRecordStrideInBytes=header_size,
        missRecordCount=1,
        hitgroupRecordBase=d_hit.data.ptr,
        hitgroupRecordStrideInBytes=header_size,
        hitgroupRecordCount=1,
    )

    # Keep references to prevent garbage collection
    _state._sbt_raygen = d_raygen
    _state._sbt_miss = d_miss
    _state._sbt_hit = d_hit


# -----------------------------------------------------------------------------
# Acceleration structure building
# -----------------------------------------------------------------------------

def _build_accel(hash_value: int, vertices, indices) -> int:
    """
    Build an OptiX acceleration structure for the given triangle mesh.

    Args:
        hash_value: Hash to identify this geometry (for caching)
        vertices: Vertex buffer (Nx3 float32, flattened)
        indices: Index buffer (Mx3 int32, flattened)

    Returns:
        0 on success, non-zero on error
    """
    global _state

    if not _state.initialized:
        _init_optix()

    # Check if we already have this acceleration structure cached
    if _state.current_hash == hash_value:
        return 0

    # Reset hash until successful build
    _state.current_hash = 0xFFFFFFFFFFFFFFFF

    # Ensure data is on GPU as cupy arrays
    if isinstance(vertices, cupy.ndarray):
        d_vertices = vertices
    else:
        d_vertices = cupy.asarray(vertices, dtype=cupy.float32)

    if isinstance(indices, cupy.ndarray):
        d_indices = indices
    else:
        d_indices = cupy.asarray(indices, dtype=cupy.int32)

    # Calculate counts
    num_vertices = d_vertices.size // 3
    num_triangles = d_indices.size // 3

    if num_vertices == 0 or num_triangles == 0:
        return -1

    # Build input
    build_input = optix.BuildInputTriangleArray(
        vertexBuffers_=[d_vertices.data.ptr],
        vertexFormat=optix.VERTEX_FORMAT_FLOAT3,
        vertexStrideInBytes=12,  # 3 * sizeof(float)
        indexBuffer=d_indices.data.ptr,
        numIndexTriplets=num_triangles,
        indexFormat=optix.INDICES_FORMAT_UNSIGNED_INT3,
        indexStrideInBytes=12,  # 3 * sizeof(int)
        flags_=[optix.GEOMETRY_FLAG_DISABLE_ANYHIT],
        numSbtRecords=1,
    )
    build_input.numVertices = num_vertices

    # Acceleration structure options
    accel_options = optix.AccelBuildOptions(
        buildFlags=optix.BUILD_FLAG_ALLOW_RANDOM_VERTEX_ACCESS,
        operation=optix.BUILD_OPERATION_BUILD,
    )

    # Compute memory requirements
    buffer_sizes = _state.context.accelComputeMemoryUsage(
        [accel_options],
        [build_input],
    )

    # Allocate buffers
    d_temp = cupy.zeros(buffer_sizes.tempSizeInBytes, dtype=cupy.uint8)
    _state.gas_buffer = cupy.zeros(buffer_sizes.outputSizeInBytes, dtype=cupy.uint8)

    # Build acceleration structure
    _state.gas_handle = _state.context.accelBuild(
        0,  # stream
        [accel_options],
        [build_input],
        d_temp.data.ptr,
        buffer_sizes.tempSizeInBytes,
        _state.gas_buffer.data.ptr,
        buffer_sizes.outputSizeInBytes,
        [],  # emitted properties
    )

    _state.current_hash = hash_value
    return 0


# -----------------------------------------------------------------------------
# Ray tracing
# -----------------------------------------------------------------------------

def _trace_rays(rays, hits, num_rays: int) -> int:
    """
    Trace rays against the current acceleration structure.

    Args:
        rays: Ray buffer (Nx8 float32: ox,oy,oz,tmin,dx,dy,dz,tmax)
        hits: Hit buffer (Nx4 float32: t,nx,ny,nz)
        num_rays: Number of rays to trace

    Returns:
        0 on success, non-zero on error
    """
    global _state

    if not _state.initialized:
        return -1

    if _state.gas_handle == 0:
        return -1

    # Size check
    if rays.size != num_rays * 8 or hits.size != num_rays * 4:
        return -1

    # Ensure rays are on GPU
    if isinstance(rays, cupy.ndarray):
        d_rays = rays
        rays_on_host = False
    else:
        # Allocate/resize device buffer if needed
        rays_size = num_rays * 8 * 4  # 8 floats * 4 bytes
        if _state.d_rays_size != rays_size:
            _state.d_rays = cupy.zeros(num_rays * 8, dtype=cupy.float32)
            _state.d_rays_size = rays_size
        _state.d_rays[:] = cupy.asarray(rays, dtype=cupy.float32)
        d_rays = _state.d_rays
        rays_on_host = True

    # Ensure hits buffer is on GPU
    if isinstance(hits, cupy.ndarray):
        d_hits = hits
        hits_on_host = False
    else:
        # Allocate/resize device buffer if needed
        hits_size = num_rays * 4 * 4  # 4 floats * 4 bytes
        if _state.d_hits_size != hits_size:
            _state.d_hits = cupy.zeros(num_rays * 4, dtype=cupy.float32)
            _state.d_hits_size = hits_size
        d_hits = _state.d_hits
        hits_on_host = True

    # Pack params: handle(8 bytes) + rays_ptr(8 bytes) + hits_ptr(8 bytes)
    params_data = struct.pack(
        'QQQ',
        _state.gas_handle,
        d_rays.data.ptr,
        d_hits.data.ptr,
    )
    _state.d_params[:] = cupy.frombuffer(np.frombuffer(params_data, dtype=np.uint8), dtype=cupy.uint8)

    # Launch
    optix.launch(
        _state.pipeline,
        0,  # stream
        _state.d_params.data.ptr,
        24,  # sizeof(Params)
        _state.sbt,
        num_rays,  # width
        1,  # height
        1,  # depth
    )

    # Copy results back if hits was on host
    if hits_on_host:
        cupy.cuda.Stream.null.synchronize()
        hits[:] = d_hits.get()

    return 0


# -----------------------------------------------------------------------------
# Public API (backwards compatible)
# -----------------------------------------------------------------------------

class RTX:
    """
    RTX ray tracing interface.

    This class provides GPU-accelerated ray-triangle intersection using
    NVIDIA's OptiX ray tracing engine.
    """

    def __init__(self):
        """Initialize the RTX context."""
        _init_optix()

    def build(self, hashValue: int, vertexBuffer, indexBuffer) -> int:
        """
        Build an acceleration structure for the given triangle mesh.

        Args:
            hashValue: A hash value to uniquely identify the geometry (for caching)
            vertexBuffer: Vertex buffer (flattened float32 array, 3 floats per vertex)
            indexBuffer: Index buffer (flattened int32 array, 3 ints per triangle)

        Returns:
            0 on success, non-zero on error
        """
        return _build_accel(hashValue, vertexBuffer, indexBuffer)

    def getHash(self) -> int:
        """
        Get the hash of the current acceleration structure.

        Returns:
            The hash value, or uint64(-1) if no structure is present
        """
        return _state.current_hash

    def trace(self, rays, hits, numRays: int) -> int:
        """
        Trace rays against the current acceleration structure.

        Args:
            rays: Ray buffer (8 float32 per ray: ox,oy,oz,tmin,dx,dy,dz,tmax)
            hits: Hit buffer (4 float32 per hit: t,nx,ny,nz)
                  t=-1 indicates a miss
            numRays: Number of rays to trace

        Returns:
            0 on success, non-zero on error
        """
        return _trace_rays(rays, hits, numRays)
