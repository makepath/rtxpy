"""
RTXpy - Ray tracing using NVIDIA OptiX, accessible from Python.

This module provides GPU-accelerated ray-triangle intersection using
NVIDIA's OptiX ray tracing engine via the otk-pyoptix Python bindings.
"""

import os
import atexit
import struct
from dataclasses import dataclass, field
from typing import Dict, List, Optional

# CRITICAL: cupy must be imported before optix for proper CUDA context sharing
import cupy
has_cupy = True

import optix

import numpy as np


# -----------------------------------------------------------------------------
# Data structures for multi-GAS support
# -----------------------------------------------------------------------------

@dataclass
class _GASEntry:
    """Storage for a single Geometry Acceleration Structure."""
    gas_id: str
    gas_handle: int
    gas_buffer: cupy.ndarray  # Must keep reference to prevent GC
    vertices_hash: int
    transform: List[float] = field(default_factory=lambda: [
        1.0, 0.0, 0.0, 0.0,  # Row 0: [Xx, Xy, Xz, Tx]
        0.0, 1.0, 0.0, 0.0,  # Row 1: [Yx, Yy, Yz, Ty]
        0.0, 0.0, 1.0, 0.0,  # Row 2: [Zx, Zy, Zz, Tz]
    ])  # 12 floats (3x4 row-major affine transform)


# -----------------------------------------------------------------------------
# Singleton state management
# -----------------------------------------------------------------------------

class _OptixState:
    """
    Manages the global OptiX state including device context, module, pipeline,
    shader binding table, and acceleration structure cache.
    """

    def __init__(self):
        self.device_id = None  # CUDA device ID used for this context
        self.context = None
        self.module = None
        self.pipeline = None
        self.raygen_pg = None
        self.miss_pg = None
        self.hit_pg = None
        self.sbt = None

        # Single-GAS mode acceleration structure cache
        self.gas_handle = 0
        self.gas_buffer = None
        self.current_hash = 0xFFFFFFFFFFFFFFFF  # uint64(-1)

        # Multi-GAS mode state
        self.gas_entries: Dict[str, _GASEntry] = {}  # Dict[str, _GASEntry]
        self.ias_handle = 0
        self.ias_buffer = None
        self.ias_dirty = True
        self.instances_buffer = None
        self.single_gas_mode = True  # False when multi-GAS active

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
        # Reset device tracking
        self.device_id = None

        # Free device buffers
        self.d_params = None
        self.d_rays = None
        self.d_hits = None
        self.d_rays_size = 0
        self.d_hits_size = 0

        # Free single-GAS mode acceleration structure
        self.gas_buffer = None
        self.gas_handle = 0
        self.current_hash = 0xFFFFFFFFFFFFFFFF

        # Free multi-GAS mode resources
        self.gas_entries = {}
        self.ias_handle = 0
        self.ias_buffer = None
        self.ias_dirty = True
        self.instances_buffer = None
        self.single_gas_mode = True

        # OptiX objects are automatically cleaned up by Python GC
        self.sbt = None
        self.pipeline = None
        self.hit_pg = None
        self.miss_pg = None
        self.raygen_pg = None
        self.module = None
        self.context = None

        self.initialized = False

    def reset_device(self):
        """Reset device tracking (called during cleanup)."""
        self.device_id = None


_state = _OptixState()


def _cleanup_at_exit():
    """Cleanup function registered with atexit."""
    global _state
    if _state:
        _state.cleanup()


# -----------------------------------------------------------------------------
# Device utilities
# -----------------------------------------------------------------------------

def get_device_count() -> int:
    """
    Get the number of available CUDA devices.

    Returns:
        Number of CUDA-capable GPUs available.

    Example:
        >>> import rtxpy
        >>> rtxpy.get_device_count()
        2
    """
    return cupy.cuda.runtime.getDeviceCount()


def get_device_properties(device: int = 0) -> dict:
    """
    Get properties of a CUDA device.

    Args:
        device: Device ID (0, 1, 2, ...). Defaults to device 0.

    Returns:
        Dictionary containing device properties including:
        - name: Device name (e.g., "NVIDIA GeForce RTX 3090")
        - compute_capability: Tuple of (major, minor) compute capability
        - total_memory: Total device memory in bytes
        - multiprocessor_count: Number of streaming multiprocessors

    Raises:
        ValueError: If device ID is invalid.

    Example:
        >>> import rtxpy
        >>> props = rtxpy.get_device_properties(0)
        >>> print(props['name'])
        NVIDIA GeForce RTX 3090
    """
    device_count = cupy.cuda.runtime.getDeviceCount()
    if device < 0 or device >= device_count:
        raise ValueError(
            f"Invalid device ID {device}. "
            f"Available devices: 0-{device_count - 1}"
        )

    with cupy.cuda.Device(device):
        props = cupy.cuda.runtime.getDeviceProperties(device)

    return {
        'name': props['name'].decode('utf-8') if isinstance(props['name'], bytes) else props['name'],
        'compute_capability': (props['major'], props['minor']),
        'total_memory': props['totalGlobalMem'],
        'multiprocessor_count': props['multiProcessorCount'],
    }


def list_devices() -> list:
    """
    List all available CUDA devices with their properties.

    Returns:
        List of dictionaries, each containing device properties.
        Each dict includes 'id' (device index) plus all properties
        from get_device_properties().

    Example:
        >>> import rtxpy
        >>> for dev in rtxpy.list_devices():
        ...     print(f"GPU {dev['id']}: {dev['name']}")
        GPU 0: NVIDIA GeForce RTX 3090
        GPU 1: NVIDIA GeForce RTX 2080
    """
    devices = []
    for i in range(get_device_count()):
        props = get_device_properties(i)
        props['id'] = i
        devices.append(props)
    return devices


def get_current_device() -> Optional[int]:
    """
    Get the CUDA device ID that RTX is currently using.

    Returns:
        Device ID if RTX has been initialized, None otherwise.

    Example:
        >>> import rtxpy
        >>> rtx = rtxpy.RTX(device=1)
        >>> rtxpy.get_current_device()
        1
    """
    return _state.device_id if _state.initialized else None


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


def _init_optix(device: Optional[int] = None):
    """
    Initialize OptiX context, module, pipeline, and SBT.

    Args:
        device: CUDA device ID to use. If None, uses the current CuPy device.
                If already initialized, this parameter is ignored (a warning
                would be appropriate if it differs from the active device).
    """
    global _state

    if _state.initialized:
        # Already initialized - check if user requested a different device
        if device is not None and _state.device_id != device:
            import warnings
            warnings.warn(
                f"RTX already initialized on device {_state.device_id}. "
                f"Ignoring request for device {device}. "
                "Create a new Python process to use a different device.",
                RuntimeWarning
            )
        return

    # Select the CUDA device if specified
    if device is not None:
        device_count = cupy.cuda.runtime.getDeviceCount()
        if device < 0 or device >= device_count:
            raise ValueError(
                f"Invalid device ID {device}. "
                f"Available devices: 0-{device_count - 1}"
            )
        cupy.cuda.Device(device).use()
        _state.device_id = device
    else:
        # Use current device
        _state.device_id = cupy.cuda.Device().id

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
        traversableGraphFlags=optix.TRAVERSABLE_GRAPH_FLAG_ALLOW_ANY,
        numPayloadValues=6,  # t, nx, ny, nz, primitive_id, instance_id
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
        2,  # maxTraversableDepth (IAS -> GAS = 2 levels)
    )

    # Create shader binding table
    _create_sbt()

    # Allocate params buffer (24 bytes: handle(8) + rays_ptr(8) + hits_ptr(8))
    _state.d_params = cupy.zeros(40, dtype=cupy.uint8)  # 5 pointers * 8 bytes

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

def _build_gas_for_geometry(vertices, indices):
    """
    Build a single GAS (Geometry Acceleration Structure) for the given mesh.

    Args:
        vertices: Vertex buffer (Nx3 float32, flattened)
        indices: Index buffer (Mx3 int32, flattened)

    Returns:
        Tuple of (gas_handle, gas_buffer) or (0, None) on error
    """
    global _state

    if not _state.initialized:
        _init_optix()

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
        return 0, None

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

    # Acceleration structure options - enable compaction for memory savings
    accel_options = optix.AccelBuildOptions(
        buildFlags=optix.BUILD_FLAG_ALLOW_RANDOM_VERTEX_ACCESS | optix.BUILD_FLAG_ALLOW_COMPACTION,
        operation=optix.BUILD_OPERATION_BUILD,
    )

    # Compute memory requirements
    buffer_sizes = _state.context.accelComputeMemoryUsage(
        [accel_options],
        [build_input],
    )

    # Allocate buffers
    d_temp = cupy.zeros(buffer_sizes.tempSizeInBytes, dtype=cupy.uint8)
    gas_buffer = cupy.zeros(buffer_sizes.outputSizeInBytes, dtype=cupy.uint8)

    # Allocate buffer to receive compacted size
    compacted_size_buffer = cupy.zeros(1, dtype=cupy.uint64)

    # Build acceleration structure with compacted size emission
    gas_handle = _state.context.accelBuild(
        0,  # stream
        [accel_options],
        [build_input],
        d_temp.data.ptr,
        buffer_sizes.tempSizeInBytes,
        gas_buffer.data.ptr,
        buffer_sizes.outputSizeInBytes,
        [optix.AccelEmitDesc(compacted_size_buffer.data.ptr, optix.PROPERTY_TYPE_COMPACTED_SIZE)],
    )

    # Synchronize to ensure compacted size is available
    cupy.cuda.Stream.null.synchronize()

    # Compact if it saves memory
    compacted_size = int(compacted_size_buffer[0])
    if compacted_size < gas_buffer.nbytes:
        compacted_buffer = cupy.zeros(compacted_size, dtype=cupy.uint8)
        gas_handle = _state.context.accelCompact(
            0,  # stream
            gas_handle,
            compacted_buffer.data.ptr,
            compacted_size,
        )
        gas_buffer = compacted_buffer

    return gas_handle, gas_buffer


def _build_ias():
    """
    Build an Instance Acceleration Structure (IAS) from all GAS entries.

    This creates a top-level acceleration structure that references all
    geometry acceleration structures with their transforms.
    """
    global _state

    if not _state.initialized:
        _init_optix()

    if not _state.gas_entries:
        _state.ias_handle = 0
        _state.ias_buffer = None
        _state.ias_dirty = False
        return

    num_instances = len(_state.gas_entries)

    # OptixInstance structure is 80 bytes:
    # - transform: float[12] (3x4 row-major) = 48 bytes
    # - instanceId: uint32 = 4 bytes
    # - sbtOffset: uint32 = 4 bytes
    # - visibilityMask: uint32 = 4 bytes
    # - flags: uint32 = 4 bytes
    # - traversableHandle: uint64 = 8 bytes
    # - pad: uint32[2] = 8 bytes
    # Total = 80 bytes

    INSTANCE_SIZE = 80
    instances_data = bytearray(num_instances * INSTANCE_SIZE)

    for i, (gas_id, entry) in enumerate(_state.gas_entries.items()):
        offset = i * INSTANCE_SIZE

        # Pack transform (12 floats, 48 bytes)
        transform_bytes = struct.pack('12f', *entry.transform)
        instances_data[offset:offset + 48] = transform_bytes

        # Pack instanceId (4 bytes)
        struct.pack_into('I', instances_data, offset + 48, i)

        # Pack sbtOffset (4 bytes) - all use same hit group (SBT index 0)
        struct.pack_into('I', instances_data, offset + 52, 0)

        # Pack visibilityMask (4 bytes) - 0xFF = visible to all rays
        struct.pack_into('I', instances_data, offset + 56, 0xFF)

        # Pack flags (4 bytes) - OPTIX_INSTANCE_FLAG_NONE = 0
        struct.pack_into('I', instances_data, offset + 60, 0)

        # Pack traversableHandle (8 bytes)
        struct.pack_into('Q', instances_data, offset + 64, entry.gas_handle)

        # Padding (8 bytes) - already zeros

    # Copy instances to GPU
    _state.instances_buffer = cupy.array(
        np.frombuffer(instances_data, dtype=np.uint8)
    )

    # Build input for IAS
    build_input = optix.BuildInputInstanceArray(
        instances=_state.instances_buffer.data.ptr,
        numInstances=num_instances,
    )

    # Acceleration structure options
    accel_options = optix.AccelBuildOptions(
        buildFlags=optix.BUILD_FLAG_ALLOW_UPDATE,
        operation=optix.BUILD_OPERATION_BUILD,
    )

    # Compute memory requirements
    buffer_sizes = _state.context.accelComputeMemoryUsage(
        [accel_options],
        [build_input],
    )

    # Allocate buffers
    d_temp = cupy.zeros(buffer_sizes.tempSizeInBytes, dtype=cupy.uint8)
    _state.ias_buffer = cupy.zeros(buffer_sizes.outputSizeInBytes, dtype=cupy.uint8)

    # Build IAS
    _state.ias_handle = _state.context.accelBuild(
        0,  # stream
        [accel_options],
        [build_input],
        d_temp.data.ptr,
        buffer_sizes.tempSizeInBytes,
        _state.ias_buffer.data.ptr,
        buffer_sizes.outputSizeInBytes,
        [],  # emitted properties
    )

    _state.ias_dirty = False


def _build_accel(hash_value: int, vertices, indices) -> int:
    """
    Build an OptiX acceleration structure for the given triangle mesh.

    This enables single-GAS mode and clears any multi-GAS state.

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

    # Clear multi-GAS state when switching to single-GAS mode
    if not _state.single_gas_mode:
        _state.gas_entries = {}
        _state.ias_handle = 0
        _state.ias_buffer = None
        _state.ias_dirty = True
        _state.instances_buffer = None
        _state.single_gas_mode = True

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

    # Acceleration structure options - enable compaction for memory savings
    accel_options = optix.AccelBuildOptions(
        buildFlags=optix.BUILD_FLAG_ALLOW_RANDOM_VERTEX_ACCESS | optix.BUILD_FLAG_ALLOW_COMPACTION,
        operation=optix.BUILD_OPERATION_BUILD,
    )

    # Compute memory requirements
    buffer_sizes = _state.context.accelComputeMemoryUsage(
        [accel_options],
        [build_input],
    )

    # Allocate buffers
    d_temp = cupy.zeros(buffer_sizes.tempSizeInBytes, dtype=cupy.uint8)
    gas_buffer = cupy.zeros(buffer_sizes.outputSizeInBytes, dtype=cupy.uint8)

    # Allocate buffer to receive compacted size
    compacted_size_buffer = cupy.zeros(1, dtype=cupy.uint64)

    # Build acceleration structure with compacted size emission
    _state.gas_handle = _state.context.accelBuild(
        0,  # stream
        [accel_options],
        [build_input],
        d_temp.data.ptr,
        buffer_sizes.tempSizeInBytes,
        gas_buffer.data.ptr,
        buffer_sizes.outputSizeInBytes,
        [optix.AccelEmitDesc(compacted_size_buffer.data.ptr, optix.PROPERTY_TYPE_COMPACTED_SIZE)],
    )

    # Synchronize to ensure compacted size is available
    cupy.cuda.Stream.null.synchronize()

    # Compact if it saves memory
    compacted_size = int(compacted_size_buffer[0])
    if compacted_size < gas_buffer.nbytes:
        compacted_buffer = cupy.zeros(compacted_size, dtype=cupy.uint8)
        _state.gas_handle = _state.context.accelCompact(
            0,  # stream
            _state.gas_handle,
            compacted_buffer.data.ptr,
            compacted_size,
        )
        _state.gas_buffer = compacted_buffer
    else:
        _state.gas_buffer = gas_buffer

    _state.current_hash = hash_value
    return 0


# -----------------------------------------------------------------------------
# Ray tracing
# -----------------------------------------------------------------------------

def _trace_rays(rays, hits, num_rays: int, primitive_ids=None, instance_ids=None) -> int:
    """
    Trace rays against the current acceleration structure.

    Supports both single-GAS mode (using gas_handle) and multi-GAS mode
    (using IAS that references multiple GAS).

    Args:
        rays: Ray buffer (Nx8 float32: ox,oy,oz,tmin,dx,dy,dz,tmax)
        hits: Hit buffer (Nx4 float32: t,nx,ny,nz)
        num_rays: Number of rays to trace
        primitive_ids: Optional output buffer (Nx1 int32) for triangle indices.
                       -1 indicates a miss.
        instance_ids: Optional output buffer (Nx1 int32) for geometry/instance indices.
                      -1 indicates a miss. Useful in multi-GAS mode to identify
                      which geometry was hit.

    Returns:
        0 on success, non-zero on error
    """
    global _state

    if not _state.initialized:
        return -1

    # Determine which traversable handle to use
    if _state.single_gas_mode:
        if _state.gas_handle == 0:
            return -1
        trace_handle = _state.gas_handle
    else:
        # Multi-GAS mode: rebuild IAS if dirty
        if _state.ias_dirty:
            _build_ias()
        if _state.ias_handle == 0:
            return -1
        trace_handle = _state.ias_handle

    # Size check
    if rays.size != num_rays * 8 or hits.size != num_rays * 4:
        return -1

    # Validate optional buffers
    if primitive_ids is not None and primitive_ids.size != num_rays:
        return -1
    if instance_ids is not None and instance_ids.size != num_rays:
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

    # Handle optional primitive_ids buffer
    d_prim_ids_ptr = 0
    prim_ids_on_host = False
    if primitive_ids is not None:
        if isinstance(primitive_ids, cupy.ndarray):
            d_prim_ids = primitive_ids
            prim_ids_on_host = False
        else:
            d_prim_ids = cupy.zeros(num_rays, dtype=cupy.int32)
            prim_ids_on_host = True
        d_prim_ids_ptr = d_prim_ids.data.ptr

    # Handle optional instance_ids buffer
    d_inst_ids_ptr = 0
    inst_ids_on_host = False
    if instance_ids is not None:
        if isinstance(instance_ids, cupy.ndarray):
            d_inst_ids = instance_ids
            inst_ids_on_host = False
        else:
            d_inst_ids = cupy.zeros(num_rays, dtype=cupy.int32)
            inst_ids_on_host = True
        d_inst_ids_ptr = d_inst_ids.data.ptr

    # Pack params: handle(8) + rays_ptr(8) + hits_ptr(8) + prim_ids_ptr(8) + inst_ids_ptr(8)
    params_data = struct.pack(
        'QQQQQ',
        trace_handle,
        d_rays.data.ptr,
        d_hits.data.ptr,
        d_prim_ids_ptr,
        d_inst_ids_ptr,
    )
    _state.d_params[:] = cupy.frombuffer(np.frombuffer(params_data, dtype=np.uint8), dtype=cupy.uint8)

    # Launch
    optix.launch(
        _state.pipeline,
        0,  # stream
        _state.d_params.data.ptr,
        40,  # sizeof(Params): 5 pointers * 8 bytes
        _state.sbt,
        num_rays,  # width
        1,  # height
        1,  # depth
    )

    # Copy results back if buffers were on host
    if hits_on_host or prim_ids_on_host or inst_ids_on_host:
        cupy.cuda.Stream.null.synchronize()

    if hits_on_host:
        hits[:] = d_hits.get()
    if prim_ids_on_host:
        primitive_ids[:] = d_prim_ids.get()
    if inst_ids_on_host:
        instance_ids[:] = d_inst_ids.get()

    return 0


# -----------------------------------------------------------------------------
# Public API (backwards compatible)
# -----------------------------------------------------------------------------

class RTX:
    """
    RTX ray tracing interface.

    This class provides GPU-accelerated ray-triangle intersection using
    NVIDIA's OptiX ray tracing engine.

    Args:
        device: CUDA device ID to use (0, 1, 2, ...). If None (default),
                uses the currently active CuPy device. Use get_device_count()
                to see available devices.

    Example:
        # Use default device (device 0 or current CuPy device)
        rtx = RTX()

        # Use specific GPU
        rtx = RTX(device=1)

    Note:
        The RTX context is a singleton - all RTX instances share the same
        underlying OptiX context. The device can only be set on first
        initialization. Subsequent RTX() calls with a different device
        will emit a warning.
    """

    def __init__(self, device: Optional[int] = None):
        """
        Initialize the RTX context.

        Args:
            device: CUDA device ID to use. If None, uses the current device.
        """
        _init_optix(device)

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

    @property
    def device(self) -> Optional[int]:
        """
        The CUDA device ID this RTX instance is using.

        Returns:
            Device ID (0, 1, 2, ...) or None if not initialized.
        """
        return _state.device_id

    def getHash(self) -> int:
        """
        Get the hash of the current acceleration structure.

        Returns:
            The hash value, or uint64(-1) if no structure is present
        """
        return _state.current_hash

    def trace(self, rays, hits, numRays: int, primitive_ids=None, instance_ids=None) -> int:
        """
        Trace rays against the current acceleration structure.

        Works with both single-GAS mode (after build()) and multi-GAS mode
        (after add_geometry()).

        Args:
            rays: Ray buffer (8 float32 per ray: ox,oy,oz,tmin,dx,dy,dz,tmax)
            hits: Hit buffer (4 float32 per hit: t,nx,ny,nz)
                  t=-1 indicates a miss
            numRays: Number of rays to trace
            primitive_ids: Optional output buffer (numRays x int32) for triangle indices.
                           Will contain the index of the hit triangle within its geometry,
                           or -1 for rays that missed.
            instance_ids: Optional output buffer (numRays x int32) for geometry/instance indices.
                          Will contain the instance ID of the hit geometry, or -1 for misses.
                          Useful in multi-GAS mode to identify which geometry was hit.

        Returns:
            0 on success, non-zero on error
        """
        return _trace_rays(rays, hits, numRays, primitive_ids, instance_ids)

    # -------------------------------------------------------------------------
    # Multi-GAS API
    # -------------------------------------------------------------------------

    def add_geometry(self, geometry_id: str, vertices, indices,
                     transform: Optional[List[float]] = None) -> int:
        """
        Add a geometry (GAS) to the scene with an optional transform.

        This enables multi-GAS mode. If called after build(), the single-GAS
        state is cleared. Adding a geometry with an existing ID replaces it.

        Args:
            geometry_id: Unique identifier for this geometry
            vertices: Vertex buffer (flattened float32 array, 3 floats per vertex)
            indices: Index buffer (flattened int32 array, 3 ints per triangle)
            transform: Optional 12-float list representing a 3x4 row-major
                      affine transform matrix. Defaults to identity.
                      Format: [Xx, Xy, Xz, Tx, Yx, Yy, Yz, Ty, Zx, Zy, Zz, Tz]

        Returns:
            0 on success, non-zero on error
        """
        global _state

        if not _state.initialized:
            _init_optix()

        # Switch to multi-GAS mode if currently in single-GAS mode
        if _state.single_gas_mode:
            _state.gas_handle = 0
            _state.gas_buffer = None
            _state.current_hash = 0xFFFFFFFFFFFFFFFF
            _state.single_gas_mode = False

        # Build the GAS for this geometry
        gas_handle, gas_buffer = _build_gas_for_geometry(vertices, indices)
        if gas_handle == 0:
            return -1

        # Compute a hash for caching purposes
        if isinstance(vertices, cupy.ndarray):
            vertices_for_hash = vertices.get()
        else:
            vertices_for_hash = np.asarray(vertices)
        vertices_hash = hash(vertices_for_hash.tobytes())

        # Set transform (identity if not provided)
        if transform is None:
            transform = [
                1.0, 0.0, 0.0, 0.0,
                0.0, 1.0, 0.0, 0.0,
                0.0, 0.0, 1.0, 0.0,
            ]
        else:
            transform = list(transform)
            if len(transform) != 12:
                return -1

        # Create or update the GAS entry
        _state.gas_entries[geometry_id] = _GASEntry(
            gas_id=geometry_id,
            gas_handle=gas_handle,
            gas_buffer=gas_buffer,
            vertices_hash=vertices_hash,
            transform=transform,
        )

        # Mark IAS as needing rebuild
        _state.ias_dirty = True

        return 0

    def remove_geometry(self, geometry_id: str) -> int:
        """
        Remove a geometry from the scene.

        Args:
            geometry_id: The ID of the geometry to remove

        Returns:
            0 on success, -1 if geometry not found
        """
        global _state

        if geometry_id not in _state.gas_entries:
            return -1

        del _state.gas_entries[geometry_id]
        _state.ias_dirty = True

        return 0

    def update_transform(self, geometry_id: str,
                        transform: List[float]) -> int:
        """
        Update the transform of an existing geometry.

        Args:
            geometry_id: The ID of the geometry to update
            transform: 12-float list representing a 3x4 row-major affine
                      transform matrix.
                      Format: [Xx, Xy, Xz, Tx, Yx, Yy, Yz, Ty, Zx, Zy, Zz, Tz]

        Returns:
            0 on success, -1 if geometry not found or invalid transform
        """
        global _state

        if geometry_id not in _state.gas_entries:
            return -1

        transform = list(transform)
        if len(transform) != 12:
            return -1

        _state.gas_entries[geometry_id].transform = transform
        _state.ias_dirty = True

        return 0

    def list_geometries(self) -> List[str]:
        """
        Get a list of all geometry IDs in the scene.

        Returns:
            List of geometry ID strings
        """
        return list(_state.gas_entries.keys())

    def get_geometry_count(self) -> int:
        """
        Get the number of geometries in the scene.

        Returns:
            Number of geometries (0 in single-GAS mode)
        """
        return len(_state.gas_entries)

    def has_geometry(self, geometry_id: str) -> bool:
        """
        Check if a geometry with the given ID exists.

        Args:
            geometry_id: The ID of the geometry to check.

        Returns:
            True if the geometry exists, False otherwise.
        """
        return geometry_id in _state.gas_entries

    def clear_scene(self) -> None:
        """
        Remove all geometries and reset to single-GAS mode.

        After calling this, you can use either build() for single-GAS mode
        or add_geometry() for multi-GAS mode.
        """
        global _state

        # Clear multi-GAS state
        _state.gas_entries = {}
        _state.ias_handle = 0
        _state.ias_buffer = None
        _state.ias_dirty = True
        _state.instances_buffer = None

        # Clear single-GAS state
        _state.gas_handle = 0
        _state.gas_buffer = None
        _state.current_hash = 0xFFFFFFFFFFFFFFFF

        # Reset to single-GAS mode
        _state.single_gas_mode = True
