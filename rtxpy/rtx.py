"""
RTXpy - Ray tracing using NVIDIA OptiX, accessible from Python.

This module provides GPU-accelerated ray-triangle intersection using
NVIDIA's OptiX ray tracing engine via the pyoptix-contrib Python bindings.
"""

import os
import atexit
import struct
from dataclasses import dataclass, field
from typing import Dict, List, Optional

# CRITICAL: cupy must be imported before optix for proper CUDA context sharing
import cupy
has_cupy = True
cupy.cuda.set_pinned_memory_allocator(cupy.cuda.PinnedMemoryPool().malloc)

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
    visible: bool = True
    num_vertices: int = 0
    num_triangles: int = 0
    is_curve: bool = False  # True for round curve tube GAS
    is_heightfield: bool = False  # True for heightfield custom primitive GAS
    is_sphere: bool = False  # True for sphere primitive GAS (point clouds)
    d_normals: Optional[cupy.ndarray] = None  # GPU per-vertex normals (N*3 float32)
    d_indices: Optional[cupy.ndarray] = None  # GPU index buffer (M*3 int32) for normal lookup


# -----------------------------------------------------------------------------
# Singleton state management (shared OptiX resources only)
# -----------------------------------------------------------------------------

class _OptixState:
    """
    Manages the global OptiX state including device context, module, pipeline,
    and shader binding table. These resources are shared across all RTX instances.

    Geometry state (acceleration structures) is managed per-RTX-instance to
    provide isolation between different DataArrays/users.
    """

    def __init__(self):
        self.device_id = None  # CUDA device ID used for this context
        self.context = None
        self.module = None
        self.pipeline = None
        self.raygen_pg = None
        self.miss_pg = None
        self.hit_pg = None
        self.curve_hit_pg = None  # Hit group for round curve tubes
        self.curve_module = None  # Built-in IS module for curves
        self.heightfield_hit_pg = None  # Hit group for heightfield custom primitives
        self.sbt = None

        # Device memory for params (shared, overwritten before each trace)
        self.d_params = None

        # Capability / version info (populated during _init_optix)
        self.capabilities = None

        # Denoiser state
        self.denoiser = None
        self.d_denoiser_state = None
        self.d_denoiser_scratch = None
        self.d_denoiser_normals = None
        self.d_denoiser_output = None
        self.d_denoiser_albedo = None
        self.d_denoiser_flow = None
        self._denoiser_temporal = False
        self.denoiser_width = 0
        self.denoiser_height = 0
        self._denoiser_failed = False

        self.initialized = False

    def cleanup(self):
        """Release all OptiX and CUDA resources."""
        # Destroy denoiser
        if self.denoiser is not None:
            self.denoiser.destroy()
        self.denoiser = None
        self.d_denoiser_state = None
        self.d_denoiser_scratch = None
        self.d_denoiser_normals = None
        self.d_denoiser_output = None
        self.denoiser_width = 0
        self.denoiser_height = 0
        self._denoiser_failed = False

        # Reset device tracking
        self.device_id = None

        # Free device buffers
        self.d_params = None

        # OptiX objects are automatically cleaned up by Python GC
        self.sbt = None
        self.pipeline = None
        self.heightfield_hit_pg = None
        self.curve_hit_pg = None
        self.curve_module = None
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


# -----------------------------------------------------------------------------
# Per-instance geometry state
# -----------------------------------------------------------------------------

class _GeometryState:
    """
    Per-RTX-instance geometry state. Each RTX instance has its own isolated
    geometry state, allowing multiple DataArrays to maintain separate scenes.
    """

    def __init__(self):
        # Single-GAS mode acceleration structure cache
        self.gas_handle = 0
        self.gas_buffer = None
        self.current_hash = 0xFFFFFFFFFFFFFFFF  # uint64(-1)

        # Multi-GAS mode state
        self.gas_entries: Dict[str, _GASEntry] = {}
        self.ias_handle = 0
        self.ias_buffer = None
        self.ias_dirty = True
        self.instances_buffer = None
        self.single_gas_mode = True  # False when multi-GAS active

        # Heightfield state
        self.heightfield_data = None      # GPU buffer (cupy) for elevation array
        self.hf_width = 0
        self.hf_height = 0
        self.hf_spacing_x = 0.0
        self.hf_spacing_y = 0.0
        self.hf_ve = 1.0
        self.hf_tile_size = 32
        self.hf_num_tiles_x = 0

        # Point cloud state — per-GAS colors keyed by geometry_id
        self.point_colors_per_gas = {}  # {geometry_id: np.ndarray (N*4,)}
        self.point_colors = None  # concatenated GPU buffer (built on demand)
        self.point_color_offsets = None  # GPU int32 per-instance offsets

        # Smooth normal table — GPU uint64 array [2*N], built in _build_ias
        self.d_smooth_normal_table = None

        # Device buffers for CPU->GPU transfers (per-instance)
        self.d_rays = None
        self.d_rays_size = 0
        self.d_hits = None
        self.d_hits_size = 0

    def clear(self):
        """Clear all geometry state."""
        # Clear multi-GAS state
        self.gas_entries = {}
        self.ias_handle = 0
        self.ias_buffer = None
        self.ias_dirty = True
        self.instances_buffer = None

        # Clear single-GAS state
        self.gas_handle = 0
        self.gas_buffer = None
        self.current_hash = 0xFFFFFFFFFFFFFFFF

        # Clear heightfield state
        self.heightfield_data = None
        self.hf_width = 0
        self.hf_height = 0
        self.hf_spacing_x = 0.0
        self.hf_spacing_y = 0.0
        self.hf_ve = 1.0
        self.hf_tile_size = 32
        self.hf_num_tiles_x = 0

        # Clear point cloud state
        self.point_colors_per_gas = {}
        self.point_colors = None
        self.point_color_offsets = None

        # Clear smooth normal table
        self.d_smooth_normal_table = None

        # Reset to single-GAS mode
        self.single_gas_mode = True


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


def _detect_capabilities(context) -> dict:
    """Detect OptiX and hardware capabilities after context creation."""
    optix_version = optix.version()  # (major, minor, micro)

    rtcore_version = context.getProperty(
        optix.DEVICE_PROPERTY_RTCORE_VERSION
    )

    # CUDA driver version (e.g. 12080 → 12.8)
    driver_version_int = cupy.cuda.runtime.driverGetVersion()
    cuda_major = driver_version_int // 1000
    cuda_minor = (driver_version_int % 1000) // 10

    # GPU compute capability
    dev = cupy.cuda.Device()
    props = cupy.cuda.runtime.getDeviceProperties(dev.id)
    cc_major = props['major']
    cc_minor = props['minor']

    # NVIDIA driver version from nvidia-smi (best-effort)
    nvidia_driver = 'unknown'
    try:
        import subprocess
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=driver_version',
             '--format=csv,noheader', '-i', str(dev.id)],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            nvidia_driver = result.stdout.strip().split('\n')[0]
    except Exception:
        pass

    # Feature flags — prefer runtime device property queries (OptiX 9.1+)
    optix_major = optix_version[0] if isinstance(optix_version, tuple) else 0
    has_optix9 = optix_major >= 9
    is_blackwell = cc_major >= 10  # sm_100+ = Blackwell

    # Query runtime cluster/coopvec support when OptiX 9+
    has_clusters = False
    has_cooperative_vectors = False
    cluster_limits = {}
    if has_optix9 and hasattr(optix, 'DEVICE_PROPERTY_CLUSTER_ACCEL'):
        try:
            cluster_flags = context.getProperty(
                optix.DEVICE_PROPERTY_CLUSTER_ACCEL)
            has_clusters = bool(
                cluster_flags
                & int(optix.DEVICE_PROPERTY_CLUSTER_ACCEL_FLAG_STANDARD))
            if has_clusters:
                cluster_limits = {
                    'max_cluster_vertices': context.getProperty(
                        optix.DEVICE_PROPERTY_LIMIT_MAX_CLUSTER_VERTICES),
                    'max_cluster_triangles': context.getProperty(
                        optix.DEVICE_PROPERTY_LIMIT_MAX_CLUSTER_TRIANGLES),
                    'max_clusters_per_gas': context.getProperty(
                        optix.DEVICE_PROPERTY_LIMIT_MAX_CLUSTERS_PER_GAS),
                }
        except Exception:
            pass
    if has_optix9 and hasattr(optix, 'DEVICE_PROPERTY_COOP_VEC'):
        try:
            coop_flags = context.getProperty(optix.DEVICE_PROPERTY_COOP_VEC)
            has_cooperative_vectors = bool(
                coop_flags
                & int(optix.DEVICE_PROPERTY_COOP_VEC_FLAG_STANDARD))
        except Exception:
            pass

    return {
        'optix_version': optix_version,
        'optix_version_str': '.'.join(str(x) for x in optix_version)
            if isinstance(optix_version, tuple) else str(optix_version),
        'rtcore_version': rtcore_version,
        'cuda_driver': f'{cuda_major}.{cuda_minor}',
        'nvidia_driver': nvidia_driver,
        'compute_capability': (cc_major, cc_minor),
        'gpu_name': (props['name'].decode('utf-8')
                     if isinstance(props['name'], bytes) else props['name']),
        # Feature flags (runtime-detected)
        'has_clusters': has_clusters,
        'has_cooperative_vectors': has_cooperative_vectors,
        'has_hw_linear_curves': is_blackwell,
        'has_rocaps_curves': has_optix9,
        'has_round_quadratic_bspline': True,  # always (OptiX 7.4+)
        # Cluster limits (only populated when has_clusters=True)
        **cluster_limits,
    }


def get_capabilities() -> Optional[dict]:
    """
    Get OptiX and hardware capability information.

    Returns None if RTX has not been initialized yet. Otherwise returns
    a dict with keys:

    - ``optix_version``: tuple (major, minor, micro)
    - ``optix_version_str``: e.g. ``'7.7.0'``
    - ``rtcore_version``: RT Core generation (e.g. 20 = 2nd gen)
    - ``cuda_driver``: CUDA runtime driver version string
    - ``nvidia_driver``: NVIDIA display driver version string
    - ``compute_capability``: tuple (major, minor)
    - ``gpu_name``: GPU device name string
    - ``has_clusters``: OptiX 9+ cluster/mega-geometry BVH
    - ``has_cooperative_vectors``: OptiX 9+ Tensor Core access (Blackwell)
    - ``has_hw_linear_curves``: hardware-accelerated linear curves (Blackwell)
    - ``has_rocaps_curves``: software Rocaps curve intersector (OptiX 9+)
    - ``has_round_quadratic_bspline``: round curve tubes (always True)
    """
    if not _state.initialized or _state.capabilities is None:
        return None
    return dict(_state.capabilities)


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

    # Detect capabilities now that context exists
    _state.capabilities = _detect_capabilities(_state.context)
    caps = _state.capabilities
    print(f"OptiX {caps['optix_version_str']} | "
          f"RT Core {caps['rtcore_version']} | "
          f"{caps['gpu_name']} (sm_{caps['compute_capability'][0]}"
          f"{caps['compute_capability'][1]}) | "
          f"Driver {caps['nvidia_driver']}")

    # Load PTX and create module
    ptx_data = _load_ptx_file("kernel.ptx")

    # Payload semantics: raygen reads after trace, CH+MS write, AH/IS unused
    _sem = (int(optix.PAYLOAD_SEMANTICS_TRACE_CALLER_READ)
            | int(optix.PAYLOAD_SEMANTICS_CH_WRITE)
            | int(optix.PAYLOAD_SEMANTICS_MS_WRITE))
    payload_type = optix.PayloadType(payloadSemantics=[_sem] * 6)

    module_options = optix.ModuleCompileOptions(
        maxRegisterCount=optix.COMPILE_DEFAULT_MAX_REGISTER_COUNT,
        optLevel=optix.COMPILE_OPTIMIZATION_DEFAULT,
        debugLevel=optix.COMPILE_DEBUG_LEVEL_MINIMAL,
        payloadTypes=[payload_type],
    )

    _pco_kwargs = dict(
        usesMotionBlur=False,
        traversableGraphFlags=optix.TRAVERSABLE_GRAPH_FLAG_ALLOW_ANY,
        numPayloadValues=6,  # t, nx, ny, nz, primitive_id, instance_id
        numAttributeValues=2,
        exceptionFlags=optix.EXCEPTION_FLAG_NONE,
        pipelineLaunchParamsVariableName="params",
        usesPrimitiveTypeFlags=(
            optix.PRIMITIVE_TYPE_FLAGS_TRIANGLE
            | optix.PRIMITIVE_TYPE_FLAGS_CUSTOM
            | optix.PRIMITIVE_TYPE_FLAGS_SPHERE
            | (optix.PRIMITIVE_TYPE_FLAGS_ROUND_QUADRATIC_BSPLINE_ROCAPS
               if _state.capabilities.get('has_rocaps_curves')
               else optix.PRIMITIVE_TYPE_FLAGS_ROUND_QUADRATIC_BSPLINE)
        ),
    )
    if _state.capabilities.get('has_clusters'):
        _pco_kwargs['allowClusteredGeometry'] = 1
    pipeline_options = optix.PipelineCompileOptions(**_pco_kwargs)

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

    # Hit group (closest hit only — triangles)
    hit_desc = optix.ProgramGroupDesc()
    hit_desc.hitgroupModuleCH = _state.module
    hit_desc.hitgroupEntryFunctionNameCH = "__closesthit__chit"
    _state.hit_pg, log = _state.context.programGroupCreate(
        [hit_desc],
        pg_options,
    )
    _state.hit_pg = _state.hit_pg[0]

    # Built-in IS module for curves (Rocaps if available, else standard)
    _curve_prim_type = (
        optix.PRIMITIVE_TYPE_ROUND_QUADRATIC_BSPLINE_ROCAPS
        if _state.capabilities.get('has_rocaps_curves')
        else optix.PRIMITIVE_TYPE_ROUND_QUADRATIC_BSPLINE
    )
    _curve_is_options = optix.BuiltinISOptions(
        builtinISModuleType=_curve_prim_type,
        usesMotionBlur=False,
    )
    _state.curve_module = _state.context.builtinISModuleGet(
        module_options,
        pipeline_options,
        _curve_is_options,
    )

    # Hit group for curves (same closest-hit, built-in IS for intersection)
    curve_hit_desc = optix.ProgramGroupDesc()
    curve_hit_desc.hitgroupModuleCH = _state.module
    curve_hit_desc.hitgroupEntryFunctionNameCH = "__closesthit__chit"
    curve_hit_desc.hitgroupModuleIS = _state.curve_module
    _state.curve_hit_pg, log = _state.context.programGroupCreate(
        [curve_hit_desc],
        pg_options,
    )
    _state.curve_hit_pg = _state.curve_hit_pg[0]

    # Hit group for heightfield custom primitives (custom IS + dedicated CH)
    hf_hit_desc = optix.ProgramGroupDesc()
    hf_hit_desc.hitgroupModuleCH = _state.module
    hf_hit_desc.hitgroupEntryFunctionNameCH = "__closesthit__heightfield"
    hf_hit_desc.hitgroupModuleIS = _state.module
    hf_hit_desc.hitgroupEntryFunctionNameIS = "__intersection__heightfield"
    _state.heightfield_hit_pg, log = _state.context.programGroupCreate(
        [hf_hit_desc],
        pg_options,
    )
    _state.heightfield_hit_pg = _state.heightfield_hit_pg[0]

    # Built-in IS module for spheres (point cloud rendering)
    _sphere_is_options = optix.BuiltinISOptions(
        builtinISModuleType=optix.PRIMITIVE_TYPE_SPHERE,
        usesMotionBlur=False,
    )
    _state.sphere_module = _state.context.builtinISModuleGet(
        module_options,
        pipeline_options,
        _sphere_is_options,
    )

    # Hit group for spheres (built-in IS for intersection, dedicated CH for normals)
    sphere_hit_desc = optix.ProgramGroupDesc()
    sphere_hit_desc.hitgroupModuleCH = _state.module
    sphere_hit_desc.hitgroupEntryFunctionNameCH = "__closesthit__sphere"
    sphere_hit_desc.hitgroupModuleIS = _state.sphere_module
    _state.sphere_hit_pg, log = _state.context.programGroupCreate(
        [sphere_hit_desc],
        pg_options,
    )
    _state.sphere_hit_pg = _state.sphere_hit_pg[0]

    # Create pipeline
    link_options = optix.PipelineLinkOptions(
        maxTraceDepth=1,
    )

    program_groups = [_state.raygen_pg, _state.miss_pg, _state.hit_pg,
                      _state.curve_hit_pg, _state.heightfield_hit_pg,
                      _state.sphere_hit_pg]
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

    # Allocate params buffer: 48 + 40 (heightfield) + 8 (point_colors) + 8 (smooth_normal_table) = 104
    _state.d_params = cupy.zeros(104, dtype=cupy.uint8)

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

    # Pack hit group records: [0] = triangles, [1] = curves, [2] = heightfield, [3] = spheres
    hit_record = bytearray(header_size)
    optix.sbtRecordPackHeader(_state.hit_pg, hit_record)

    curve_hit_record = bytearray(header_size)
    optix.sbtRecordPackHeader(_state.curve_hit_pg, curve_hit_record)

    hf_hit_record = bytearray(header_size)
    optix.sbtRecordPackHeader(_state.heightfield_hit_pg, hf_hit_record)

    sphere_hit_record = bytearray(header_size)
    optix.sbtRecordPackHeader(_state.sphere_hit_pg, sphere_hit_record)

    # Concatenate all hit records into a single buffer
    hit_all = (bytearray(hit_record) + bytearray(curve_hit_record) +
               bytearray(hf_hit_record) + bytearray(sphere_hit_record))
    d_hit = cupy.array(np.frombuffer(hit_all, dtype=np.uint8))

    _state.sbt = optix.ShaderBindingTable(
        raygenRecord=d_raygen.data.ptr,
        missRecordBase=d_miss.data.ptr,
        missRecordStrideInBytes=header_size,
        missRecordCount=1,
        hitgroupRecordBase=d_hit.data.ptr,
        hitgroupRecordStrideInBytes=header_size,
        hitgroupRecordCount=4,
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


def _build_gas_clustered(vertices, indices, grid_H, grid_W):
    """
    Build a GAS via OptiX 9 Cluster Acceleration Structures (CLAS).

    The terrain grid is partitioned into spatial blocks of up to BLOCK×BLOCK
    cells.  Each block becomes one CLAS (cluster).  All clusters are then
    assembled into a single GAS.

    Args:
        vertices: Vertex buffer (Nx3 float32, flattened) — already on GPU or host.
        indices:  Index buffer (Mx3 int32, flattened).
        grid_H:   Number of vertex rows in the terrain grid.
        grid_W:   Number of vertex columns in the terrain grid.

    Returns:
        Tuple of (gas_handle, gas_buffer) or (0, None) on error.
    """
    global _state

    if not _state.initialized:
        _init_optix()

    d_vertices = (vertices if isinstance(vertices, cupy.ndarray)
                  else cupy.asarray(vertices, dtype=cupy.float32))
    d_indices = (indices if isinstance(indices, cupy.ndarray)
                 else cupy.asarray(indices, dtype=cupy.int32))

    num_vertices = d_vertices.size // 3
    num_triangles = d_indices.size // 3
    if num_vertices == 0 or num_triangles == 0:
        return 0, None

    # -- Partition grid into spatial blocks --------------------------------
    max_cluster_v = min(
        _state.capabilities.get('max_cluster_vertices', 256), 256)
    max_cluster_t = min(
        _state.capabilities.get('max_cluster_triangles', 256), 256)

    # Largest BLOCK so that (BLOCK+1)^2 ≤ max_verts AND 2*BLOCK^2 ≤ max_tris
    import math
    block_v = int(math.isqrt(max_cluster_v)) - 1       # vertices side
    block_t = int(math.isqrt(max_cluster_t // 2))       # triangles side
    BLOCK = max(1, min(block_v, block_t))                # cells per side

    cell_rows = grid_H - 1
    cell_cols = grid_W - 1
    blocks_r = (cell_rows + BLOCK - 1) // BLOCK
    blocks_c = (cell_cols + BLOCK - 1) // BLOCK
    num_clusters = blocks_r * blocks_c

    if num_clusters == 0:
        return _build_gas_for_geometry(vertices, indices)

    # -- Build per-cluster Args structs on host, then upload ---------------
    # TrianglesArgs layout (72 bytes — see optix_types.h):
    #   0: clusterId          u32
    #   4: clusterFlags       u32
    #   8: packed bitfield    u32  (triCount:9|vertCount:9|truncBits:6|idxFmt:4|ommFmt:4)
    #  12: basePrimitiveInfo  u32  (sbtIndex:24|reserved:5|primFlags:3)
    #  16: indexStride         u16
    #  18: vertexStride        u16
    #  20: primInfoStride      u16
    #  22: ommIdxStride        u16
    #  24: indexBuffer         u64
    #  32: vertexBuffer        u64
    #  40: primitiveInfoBuffer u64
    #  48: opacityMicromapArray u64
    #  56: opacityMicromapIdxBuf u64
    #  64: instBBoxLimit       u64
    ARGS_SIZE = 72

    # We'll build small index sub-buffers per cluster (re-indexed)
    # and collect them all into one large GPU buffer.
    args_host = np.zeros(num_clusters * ARGS_SIZE, dtype=np.uint8)
    h_indices = (d_indices.get() if isinstance(d_indices, cupy.ndarray)
                 else np.asarray(d_indices, dtype=np.int32))
    h_indices = h_indices.reshape(-1, 3)

    # Per-cluster re-indexed index arrays (host)
    cluster_index_arrays = []
    max_tri_per_cluster = 0
    max_vert_per_cluster = 0

    for br in range(blocks_r):
        for bc in range(blocks_c):
            cid = br * blocks_c + bc
            r0 = br * BLOCK
            c0 = bc * BLOCK
            r1 = min(r0 + BLOCK, cell_rows)
            c1 = min(c0 + BLOCK, cell_cols)
            bH = r1 - r0   # cells in this block
            bW = c1 - c0

            # Vertex range for this block: rows [r0..r1] × cols [c0..c1]
            v_min = r0 * grid_W + c0
            v_rows = bH + 1
            v_cols = bW + 1
            v_count = v_rows * v_cols

            # Gather triangle indices for this block
            tri_list = []
            for lr in range(bH):
                for lc in range(bW):
                    gr = r0 + lr
                    gc = c0 + lc
                    tri_idx = (gr * cell_cols + gc) * 2
                    tri_list.append(tri_idx)
                    tri_list.append(tri_idx + 1)
            tri_count = len(tri_list)

            # Re-index: map global vertex ids → local [0..v_count)
            local_indices = np.empty(tri_count * 3, dtype=np.int32)
            for i, ti in enumerate(tri_list):
                for k in range(3):
                    gv = h_indices[ti, k]
                    # Map from global (row*W+col) to local block coords
                    gv_row = gv // grid_W - r0
                    gv_col = gv % grid_W - c0
                    local_indices[i * 3 + k] = gv_row * v_cols + gv_col

            cluster_index_arrays.append(local_indices)
            max_tri_per_cluster = max(max_tri_per_cluster, tri_count)
            max_vert_per_cluster = max(max_vert_per_cluster, v_count)

            # Pack the Args struct for this cluster
            off = cid * ARGS_SIZE
            # clusterId
            struct.pack_into('<I', args_host, off + 0, cid)
            # clusterFlags = NONE
            struct.pack_into('<I', args_host, off + 4, 0)
            # packed bitfield
            idx_fmt = 4  # OPTIX_CLUSTER_ACCEL_INDICES_FORMAT_32BIT
            packed = (tri_count & 0x1FF) | ((v_count & 0x1FF) << 9) | (idx_fmt << 24)
            struct.pack_into('<I', args_host, off + 8, packed)
            # basePrimitiveInfo: sbtIndex=0, primFlags=DISABLE_ANYHIT(bit 2)=4
            prim_info = (0 & 0xFFFFFF) | (4 << 29)  # flags in top 3 bits
            struct.pack_into('<I', args_host, off + 12, prim_info)
            # strides (0 = natural)
            struct.pack_into('<HHHH', args_host, off + 16, 0, 0, 0, 0)
            # indexBuffer, vertexBuffer — filled after GPU upload
            # others = 0 (no prim info, no OMM, no bbox)

    # Upload all cluster index arrays to one contiguous GPU buffer
    # and compute the per-cluster vertex buffer offsets.
    max_idx_size = max(len(a) for a in cluster_index_arrays)
    idx_padded = np.zeros(num_clusters * max_idx_size, dtype=np.int32)
    for i, arr in enumerate(cluster_index_arrays):
        idx_padded[i * max_idx_size: i * max_idx_size + len(arr)] = arr
    d_cluster_indices = cupy.asarray(idx_padded)

    # Now fill in the GPU pointers in each Args struct
    for br in range(blocks_r):
        for bc in range(blocks_c):
            cid = br * blocks_c + bc
            off = cid * ARGS_SIZE
            r0 = br * BLOCK
            c0 = bc * BLOCK

            # Index buffer pointer
            idx_ptr = d_cluster_indices.data.ptr + cid * max_idx_size * 4
            struct.pack_into('<Q', args_host, off + 24, idx_ptr)

            # Vertex buffer pointer: point to first vertex of this block
            # Vertices for this block span rows [r0..r0+bH] × cols [c0..c0+bW]
            # But the global vertex buffer is flat: vertex (r,c) = d_vertices[r*W+c]
            # Clusters require contiguous vertex buffer, so we need sub-buffers.
            # For now, we'll use row-stride trick: set vertex stride to grid_W*12
            # and point at the first vertex in the block.
            v_min = r0 * grid_W + c0
            vert_ptr = d_vertices.data.ptr + v_min * 12  # 3 floats * 4 bytes
            struct.pack_into('<Q', args_host, off + 32, vert_ptr)

            # Override vertexStride to grid_W * 12 (bytes per row of the grid)
            # Wait — vertexStride means distance between consecutive vertex
            # indices, not consecutive grid rows.  Since the local index maps
            # (local_row * v_cols + local_col) → vertex at global offset
            # (r0 + local_row)*grid_W + (c0 + local_col), the layout IS NOT
            # contiguous. We need a contiguous sub-buffer.

    # The non-contiguous vertex layout means we need to extract and flatten
    # sub-buffers per cluster. Let's build a single packed vertex buffer.
    verts_per_cluster = []
    for br in range(blocks_r):
        for bc in range(blocks_c):
            r0 = br * BLOCK
            c0 = bc * BLOCK
            r1 = min(r0 + BLOCK, cell_rows) + 1  # +1 for vertex count
            c1 = min(c0 + BLOCK, cell_cols) + 1
            # Extract sub-grid vertices and flatten
            # d_vertices is flat: vertex (r,c) = d_vertices[(r*grid_W+c)*3 : ...]
            rows = np.arange(r0, r1)
            cols = np.arange(c0, c1)
            # Global flat indices for each vertex in the block
            global_ids = (rows[:, None] * grid_W + cols[None, :]).ravel()
            verts_per_cluster.append(global_ids)

    max_v_per_cluster = max(len(v) for v in verts_per_cluster)
    # Build a flat packed vertex buffer: num_clusters × max_v × 3 floats
    h_vertices = d_vertices.get().reshape(-1, 3)
    packed_verts = np.zeros(num_clusters * max_v_per_cluster * 3,
                            dtype=np.float32)
    for i, gids in enumerate(verts_per_cluster):
        base = i * max_v_per_cluster * 3
        for j, gid in enumerate(gids):
            packed_verts[base + j*3: base + j*3 + 3] = h_vertices[gid]
    d_packed_verts = cupy.asarray(packed_verts)

    # Update args with correct vertex pointers
    for cid in range(num_clusters):
        off = cid * ARGS_SIZE
        vert_ptr = d_packed_verts.data.ptr + cid * max_v_per_cluster * 12
        struct.pack_into('<Q', args_host, off + 32, vert_ptr)

    # Upload args to GPU
    d_args = cupy.asarray(np.frombuffer(args_host, dtype=np.uint8))

    # Compute actual totals across all clusters (boundary vertices are
    # duplicated in per-cluster sub-buffers, so the sum exceeds num_vertices).
    total_cluster_verts = sum(len(v) for v in verts_per_cluster)
    total_cluster_tris = sum(len(a) // 3 for a in cluster_index_arrays)

    # -- Phase 1: Build clusters from triangles ----------------------------
    cluster_build_input = {
        'type': int(optix.CLUSTER_ACCEL_BUILD_TYPE_CLUSTERS_FROM_TRIANGLES),
        'triangles': {
            'flags': int(optix.CLUSTER_ACCEL_BUILD_FLAG_PREFER_FAST_TRACE),
            'maxArgCount': num_clusters,
            'vertexFormat': int(optix.VERTEX_FORMAT_FLOAT3),
            'maxSbtIndexValue': 0,
            'maxUniqueSbtIndexCountPerArg': 1,
            'maxTriangleCountPerArg': max_tri_per_cluster,
            'maxVertexCountPerArg': max_vert_per_cluster,
            'maxTotalTriangleCount': total_cluster_tris,
            'maxTotalVertexCount': total_cluster_verts,
            'minPositionTruncateBitCount': 0,
        },
    }

    # Compute memory for cluster build
    clas_sizes = optix.clusterAccelComputeMemoryUsage(
        _state.context,
        optix.CLUSTER_ACCEL_BUILD_MODE_IMPLICIT_DESTINATIONS,
        cluster_build_input,
    )

    d_clas_temp = cupy.zeros(max(clas_sizes.tempSizeInBytes, 1),
                             dtype=cupy.uint8)
    # Output buffer alignment: 128 bytes
    out_size = clas_sizes.outputSizeInBytes
    out_size = ((out_size + 127) // 128) * 128
    d_clas_output = cupy.zeros(max(out_size, 128), dtype=cupy.uint8)
    # Handles buffer: one uint64 per cluster
    d_clas_handles = cupy.zeros(num_clusters, dtype=cupy.uint64)
    # Sizes buffer: one uint32 per cluster
    d_clas_sizes = cupy.zeros(num_clusters, dtype=cupy.uint32)

    optix.clusterAccelBuild(
        _state.context,
        0,  # stream
        optix.CLUSTER_ACCEL_BUILD_MODE_IMPLICIT_DESTINATIONS,
        cluster_build_input,
        d_clas_output.data.ptr,
        d_clas_output.nbytes,
        d_clas_temp.data.ptr,
        d_clas_temp.nbytes,
        d_clas_handles.data.ptr,
        0,  # handles stride (natural = 8)
        d_clas_sizes.data.ptr,
        0,  # sizes stride (natural = 4)
        d_args.data.ptr,
        0,  # argsCount (null → use maxArgCount)
        ARGS_SIZE,
    )
    cupy.cuda.Stream.null.synchronize()

    # Free temp
    del d_clas_temp

    # -- Phase 2: Build GAS from clusters ----------------------------------
    # Pack ClustersArgs struct (16 bytes) in device memory
    clusters_args = np.zeros(16, dtype=np.uint8)
    struct.pack_into('<I', clusters_args, 0, num_clusters)
    struct.pack_into('<I', clusters_args, 4, 0)  # stride (natural = 8)
    struct.pack_into('<Q', clusters_args, 8, d_clas_handles.data.ptr)
    d_clusters_args = cupy.asarray(clusters_args)

    gas_build_input = {
        'type': int(optix.CLUSTER_ACCEL_BUILD_TYPE_GASES_FROM_CLUSTERS),
        'clusters': {
            'flags': int(optix.CLUSTER_ACCEL_BUILD_FLAG_PREFER_FAST_TRACE),
            'maxArgCount': 1,
            'maxTotalClusterCount': num_clusters,
            'maxClusterCountPerArg': num_clusters,
        },
    }

    gas_sizes = optix.clusterAccelComputeMemoryUsage(
        _state.context,
        optix.CLUSTER_ACCEL_BUILD_MODE_IMPLICIT_DESTINATIONS,
        gas_build_input,
    )

    d_gas_temp = cupy.zeros(max(gas_sizes.tempSizeInBytes, 1),
                            dtype=cupy.uint8)
    gas_out_size = ((gas_sizes.outputSizeInBytes + 127) // 128) * 128
    d_gas_output = cupy.zeros(max(gas_out_size, 128), dtype=cupy.uint8)
    d_gas_handle = cupy.zeros(1, dtype=cupy.uint64)

    optix.clusterAccelBuild(
        _state.context,
        0,  # stream
        optix.CLUSTER_ACCEL_BUILD_MODE_IMPLICIT_DESTINATIONS,
        gas_build_input,
        d_gas_output.data.ptr,
        d_gas_output.nbytes,
        d_gas_temp.data.ptr,
        d_gas_temp.nbytes,
        d_gas_handle.data.ptr,
        0,  # handles stride
        0,  # sizes buffer (not needed for GAS)
        0,  # sizes stride
        d_clusters_args.data.ptr,
        0,  # argsCount
        0,  # argsStride (natural = 16)
    )
    cupy.cuda.Stream.null.synchronize()

    gas_handle = int(d_gas_handle[0])

    # Keep references to prevent GC of all backing buffers
    gas_buffer = (d_gas_output, d_clas_output, d_clas_handles,
                  d_packed_verts, d_cluster_indices)

    return gas_handle, gas_buffer


def _build_gas_for_curves(vertices, widths, indices, num_segments):
    """
    Build a GAS for round quadratic B-spline curve tubes.

    Args:
        vertices: Control point positions (N*3 float32, flattened)
        widths: Per-control-point radii (N float32)
        indices: Segment start indices (num_segments int32)
        num_segments: Number of curve segments

    Returns:
        Tuple of (gas_handle, gas_buffer) or (0, None) on error
    """
    global _state

    if not _state.initialized:
        _init_optix()

    d_vertices = cupy.asarray(vertices, dtype=cupy.float32)
    d_widths = cupy.asarray(widths, dtype=cupy.float32)
    d_indices = cupy.asarray(indices, dtype=cupy.int32)
    num_vertices = d_vertices.size // 3

    if num_vertices < 3 or num_segments == 0:
        return 0, None

    build_input = optix.BuildInputCurveArray()
    build_input.curveType = (
        optix.PRIMITIVE_TYPE_ROUND_QUADRATIC_BSPLINE_ROCAPS
        if _state.capabilities and _state.capabilities.get('has_rocaps_curves')
        else optix.PRIMITIVE_TYPE_ROUND_QUADRATIC_BSPLINE
    )
    build_input.numPrimitives = num_segments
    build_input.numVertices = num_vertices
    build_input.vertexBuffers = [d_vertices.data.ptr]
    build_input.vertexStrideInBytes = 12  # 3 * sizeof(float)
    build_input.widthBuffers = [d_widths.data.ptr]
    build_input.widthStrideInBytes = 4   # sizeof(float)
    build_input.indexBuffer = d_indices.data.ptr
    build_input.indexStrideInBytes = 4   # sizeof(int)
    build_input.flag = optix.GEOMETRY_FLAG_DISABLE_ANYHIT

    accel_options = optix.AccelBuildOptions(
        buildFlags=(optix.BUILD_FLAG_ALLOW_RANDOM_VERTEX_ACCESS
                    | optix.BUILD_FLAG_ALLOW_COMPACTION),
        operation=optix.BUILD_OPERATION_BUILD,
    )

    buffer_sizes = _state.context.accelComputeMemoryUsage(
        [accel_options],
        [build_input],
    )

    d_temp = cupy.zeros(buffer_sizes.tempSizeInBytes, dtype=cupy.uint8)
    gas_buffer = cupy.zeros(buffer_sizes.outputSizeInBytes, dtype=cupy.uint8)
    compacted_size_buffer = cupy.zeros(1, dtype=cupy.uint64)

    gas_handle = _state.context.accelBuild(
        0,  # stream
        [accel_options],
        [build_input],
        d_temp.data.ptr,
        buffer_sizes.tempSizeInBytes,
        gas_buffer.data.ptr,
        buffer_sizes.outputSizeInBytes,
        [optix.AccelEmitDesc(compacted_size_buffer.data.ptr,
                             optix.PROPERTY_TYPE_COMPACTED_SIZE)],
    )

    cupy.cuda.Stream.null.synchronize()

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


def _build_gas_for_heightfield(elevation_data, H, W, spacing_x, spacing_y, ve, tile_size, active_mask=None):
    """
    Build a GAS for heightfield terrain using custom AABB primitives.

    Each AABB covers a tile_size x tile_size region of the heightfield grid.
    The intersection program ray-marches through cells within each tile.

    Args:
        elevation_data: numpy float32 array of shape (H, W) with elevation values
        H: Number of rows
        W: Number of columns
        spacing_x: World-space pixel spacing in X
        spacing_y: World-space pixel spacing in Y
        ve: Vertical exaggeration factor
        tile_size: Tile dimension (e.g. 32)
        active_mask: Optional numpy bool array of length num_tiles.
            When provided, inactive tiles get zero-volume AABBs so only
            a subset of the heightfield grid participates in ray tracing.

    Returns:
        Tuple of (gas_handle, gas_buffer, d_elevation, num_tiles_x, num_tiles_y)
    """
    global _state

    if not _state.initialized:
        _init_optix()

    import math

    num_tiles_x = math.ceil((W - 1) / tile_size)
    num_tiles_y = math.ceil((H - 1) / tile_size)
    num_tiles = num_tiles_x * num_tiles_y

    # Upload elevation data to GPU
    elev_np = np.asarray(elevation_data, dtype=np.float32)
    d_elevation = cupy.asarray(elev_np)

    # Build AABB for each tile
    aabbs = np.zeros(num_tiles * 6, dtype=np.float32)
    eps = 0.01  # Small padding for Z bounds

    for ty in range(num_tiles_y):
        for tx in range(num_tiles_x):
            tile_idx = ty * num_tiles_x + tx

            # Skip inactive tiles (LOD-managed heightfield mode)
            if active_mask is not None and not active_mask[tile_idx]:
                base = tile_idx * 6
                aabbs[base:base + 6] = 0.0
                continue

            # Cell range for this tile
            c0 = tx * tile_size
            r0 = ty * tile_size
            c1 = min(c0 + tile_size, W - 1)
            r1 = min(r0 + tile_size, H - 1)

            if c0 >= c1 or r0 >= r1:
                # Degenerate tile — set a zero-volume AABB
                base = tile_idx * 6
                aabbs[base:base + 6] = 0.0
                continue

            # Extract elevation tile (include +1 for cell corners)
            tile_elev = elev_np[r0:r1 + 1, c0:c1 + 1]
            valid = tile_elev[~np.isnan(tile_elev)]
            if valid.size == 0:
                z_min = 0.0
                z_max = 0.0
            else:
                z_min = float(valid.min())
                z_max = float(valid.max())

            z_min *= ve
            z_max *= ve

            base = tile_idx * 6
            aabbs[base + 0] = c0 * spacing_x           # min_x
            aabbs[base + 1] = r0 * spacing_y           # min_y
            aabbs[base + 2] = z_min - eps              # min_z
            aabbs[base + 3] = c1 * spacing_x           # max_x
            aabbs[base + 4] = r1 * spacing_y           # max_y
            aabbs[base + 5] = z_max + eps              # max_z

    d_aabbs = cupy.asarray(aabbs)

    # Build custom primitive GAS
    build_input = optix.BuildInputCustomPrimitiveArray()
    build_input.aabbBuffers = [d_aabbs.data.ptr]
    build_input.numPrimitives = num_tiles
    build_input.strideInBytes = 24  # 6 floats
    build_input.flags = [optix.GEOMETRY_FLAG_DISABLE_ANYHIT]
    build_input.numSbtRecords = 1

    accel_options = optix.AccelBuildOptions(
        buildFlags=optix.BUILD_FLAG_ALLOW_COMPACTION,
        operation=optix.BUILD_OPERATION_BUILD,
    )

    buffer_sizes = _state.context.accelComputeMemoryUsage(
        [accel_options],
        [build_input],
    )

    d_temp = cupy.zeros(buffer_sizes.tempSizeInBytes, dtype=cupy.uint8)
    gas_buffer = cupy.zeros(buffer_sizes.outputSizeInBytes, dtype=cupy.uint8)
    compacted_size_buffer = cupy.zeros(1, dtype=cupy.uint64)

    gas_handle = _state.context.accelBuild(
        0,
        [accel_options],
        [build_input],
        d_temp.data.ptr,
        buffer_sizes.tempSizeInBytes,
        gas_buffer.data.ptr,
        buffer_sizes.outputSizeInBytes,
        [optix.AccelEmitDesc(compacted_size_buffer.data.ptr,
                             optix.PROPERTY_TYPE_COMPACTED_SIZE)],
    )

    cupy.cuda.Stream.null.synchronize()

    compacted_size = int(compacted_size_buffer[0])
    if compacted_size < gas_buffer.nbytes:
        compacted_buffer = cupy.zeros(compacted_size, dtype=cupy.uint8)
        gas_handle = _state.context.accelCompact(
            0,
            gas_handle,
            compacted_buffer.data.ptr,
            compacted_size,
        )
        gas_buffer = compacted_buffer

    return gas_handle, gas_buffer, d_elevation, num_tiles_x, num_tiles_y


def _build_gas_for_spheres(centers, radii, single_radius=False):
    """
    Build a GAS for sphere primitives (point cloud rendering).

    Args:
        centers: Sphere center positions (N*3 float32, flattened)
        radii: Per-sphere radii (N float32) or single float if single_radius=True
        single_radius: If True, radii contains a single value for all spheres

    Returns:
        Tuple of (gas_handle, gas_buffer) or (0, None) on error
    """
    global _state

    if not _state.initialized:
        _init_optix()

    d_centers = cupy.asarray(centers, dtype=cupy.float32)
    num_vertices = d_centers.size // 3

    if num_vertices == 0:
        return 0, None

    if single_radius:
        d_radii = cupy.asarray([radii] if np.isscalar(radii) else radii,
                               dtype=cupy.float32)
    else:
        d_radii = cupy.asarray(radii, dtype=cupy.float32)

    build_input = optix.BuildInputSphereArray()
    build_input.vertexBuffers = [d_centers.data.ptr]
    build_input.numVertices = num_vertices
    build_input.vertexStrideInBytes = 12  # 3 * sizeof(float)
    build_input.radiusBuffers = [d_radii.data.ptr]
    build_input.radiusStrideInBytes = 4 if not single_radius else 0
    build_input.singleRadius = 1 if single_radius else 0
    build_input.flags = [optix.GEOMETRY_FLAG_DISABLE_ANYHIT]
    build_input.numSbtRecords = 1

    accel_options = optix.AccelBuildOptions(
        buildFlags=optix.BUILD_FLAG_ALLOW_COMPACTION,
        operation=optix.BUILD_OPERATION_BUILD,
    )

    buffer_sizes = _state.context.accelComputeMemoryUsage(
        [accel_options],
        [build_input],
    )

    d_temp = cupy.zeros(buffer_sizes.tempSizeInBytes, dtype=cupy.uint8)
    gas_buffer = cupy.zeros(buffer_sizes.outputSizeInBytes, dtype=cupy.uint8)
    compacted_size_buffer = cupy.zeros(1, dtype=cupy.uint64)

    gas_handle = _state.context.accelBuild(
        0,  # stream
        [accel_options],
        [build_input],
        d_temp.data.ptr,
        buffer_sizes.tempSizeInBytes,
        gas_buffer.data.ptr,
        buffer_sizes.outputSizeInBytes,
        [optix.AccelEmitDesc(compacted_size_buffer.data.ptr,
                             optix.PROPERTY_TYPE_COMPACTED_SIZE)],
    )

    cupy.cuda.Stream.null.synchronize()

    compacted_size = int(compacted_size_buffer[0])
    if compacted_size < gas_buffer.nbytes:
        compacted_buffer = cupy.zeros(compacted_size, dtype=cupy.uint8)
        gas_handle = _state.context.accelCompact(
            0,
            gas_handle,
            compacted_buffer.data.ptr,
            compacted_size,
        )
        gas_buffer = compacted_buffer

    return gas_handle, gas_buffer


def _build_ias(geom_state: _GeometryState):
    """
    Build an Instance Acceleration Structure (IAS) from all GAS entries.

    This creates a top-level acceleration structure that references all
    geometry acceleration structures with their transforms.

    Args:
        geom_state: The geometry state containing GAS entries to build IAS from.
    """
    global _state

    if not _state.initialized:
        _init_optix()

    if not geom_state.gas_entries:
        geom_state.ias_handle = 0
        geom_state.ias_buffer = None
        geom_state.ias_dirty = False
        return

    num_instances = len(geom_state.gas_entries)

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

    for i, (gas_id, entry) in enumerate(geom_state.gas_entries.items()):
        offset = i * INSTANCE_SIZE

        # Pack transform (12 floats, 48 bytes)
        transform_bytes = struct.pack('12f', *entry.transform)
        instances_data[offset:offset + 48] = transform_bytes

        # Pack instanceId (4 bytes)
        struct.pack_into('I', instances_data, offset + 48, i)

        # Pack sbtOffset (4 bytes) - 0=triangles, 1=curves, 2=heightfield, 3=spheres
        if entry.is_sphere:
            sbt_offset = 3
        elif entry.is_heightfield:
            sbt_offset = 2
        elif entry.is_curve:
            sbt_offset = 1
        else:
            sbt_offset = 0
        struct.pack_into('I', instances_data, offset + 52, sbt_offset)

        # Pack visibilityMask (4 bytes) - 0xFF = visible, 0x00 = hidden
        mask = 0xFF if entry.visible else 0x00
        struct.pack_into('I', instances_data, offset + 56, mask)

        # Pack flags (4 bytes) - OPTIX_INSTANCE_FLAG_NONE = 0
        struct.pack_into('I', instances_data, offset + 60, 0)

        # Pack traversableHandle (8 bytes)
        struct.pack_into('Q', instances_data, offset + 64, entry.gas_handle)

        # Padding (8 bytes) - already zeros

    # Copy instances to GPU
    geom_state.instances_buffer = cupy.array(
        np.frombuffer(instances_data, dtype=np.uint8)
    )

    # Build input for IAS
    build_input = optix.BuildInputInstanceArray(
        instances=geom_state.instances_buffer.data.ptr,
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
    geom_state.ias_buffer = cupy.zeros(buffer_sizes.outputSizeInBytes, dtype=cupy.uint8)

    # Build IAS
    geom_state.ias_handle = _state.context.accelBuild(
        0,  # stream
        [accel_options],
        [build_input],
        d_temp.data.ptr,
        buffer_sizes.tempSizeInBytes,
        geom_state.ias_buffer.data.ptr,
        buffer_sizes.outputSizeInBytes,
        [],  # emitted properties
    )

    geom_state.ias_dirty = False

    # Build smooth normal lookup table: [2*i]=normals_ptr, [2*i+1]=indices_ptr
    has_any_normals = any(
        e.d_normals is not None for e in geom_state.gas_entries.values())
    if has_any_normals:
        table = np.zeros(2 * num_instances, dtype=np.uint64)
        for i, (gid, entry) in enumerate(geom_state.gas_entries.items()):
            if entry.d_normals is not None and entry.d_indices is not None:
                table[2 * i] = entry.d_normals.data.ptr
                table[2 * i + 1] = entry.d_indices.data.ptr
        geom_state.d_smooth_normal_table = cupy.asarray(table)
    else:
        geom_state.d_smooth_normal_table = None


def _build_accel(geom_state: _GeometryState, hash_value: int, vertices, indices) -> int:
    """
    Build an OptiX acceleration structure for the given triangle mesh.

    This enables single-GAS mode and clears any multi-GAS state.

    Args:
        geom_state: The geometry state to store the acceleration structure in.
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
    if not geom_state.single_gas_mode:
        geom_state.gas_entries = {}
        geom_state.ias_handle = 0
        geom_state.ias_buffer = None
        geom_state.ias_dirty = True
        geom_state.instances_buffer = None
        geom_state.single_gas_mode = True

    # Check if we already have this acceleration structure cached
    if geom_state.current_hash == hash_value:
        return 0

    # Reset hash until successful build
    geom_state.current_hash = 0xFFFFFFFFFFFFFFFF

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
    geom_state.gas_handle = _state.context.accelBuild(
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
        geom_state.gas_handle = _state.context.accelCompact(
            0,  # stream
            geom_state.gas_handle,
            compacted_buffer.data.ptr,
            compacted_size,
        )
        geom_state.gas_buffer = compacted_buffer
    else:
        geom_state.gas_buffer = gas_buffer

    geom_state.current_hash = hash_value
    return 0


# -----------------------------------------------------------------------------
# Ray tracing
# -----------------------------------------------------------------------------

def _trace_rays(geom_state: _GeometryState, rays, hits, num_rays: int,
                primitive_ids=None, instance_ids=None, ray_flags=None) -> int:
    """
    Trace rays against the acceleration structure in the given geometry state.

    Supports both single-GAS mode (using gas_handle) and multi-GAS mode
    (using IAS that references multiple GAS).

    Args:
        geom_state: The geometry state containing the acceleration structure.
        rays: Ray buffer (Nx8 float32: ox,oy,oz,tmin,dx,dy,dz,tmax)
        hits: Hit buffer (Nx4 float32: t,nx,ny,nz)
        num_rays: Number of rays to trace
        primitive_ids: Optional output buffer (Nx1 int32) for triangle indices.
                       -1 indicates a miss.
        instance_ids: Optional output buffer (Nx1 int32) for geometry/instance indices.
                      -1 indicates a miss. Useful in multi-GAS mode to identify
                      which geometry was hit.
        ray_flags: Optional OptiX ray flags (unsigned int). Default is
                   OPTIX_RAY_FLAG_CULL_BACK_FACING_TRIANGLES (0x10).
                   Use RTX.RAY_FLAG_OCCLUSION for shadow/AO queries.

    Returns:
        0 on success, non-zero on error
    """
    global _state

    if not _state.initialized:
        return -1

    # Determine which traversable handle to use
    if geom_state.single_gas_mode:
        if geom_state.gas_handle == 0:
            return -1
        trace_handle = geom_state.gas_handle
    else:
        # Multi-GAS mode: rebuild IAS if dirty
        if geom_state.ias_dirty:
            _build_ias(geom_state)
        if geom_state.ias_handle == 0:
            return -1
        trace_handle = geom_state.ias_handle

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
        # Allocate/resize device buffer if needed (per-instance)
        rays_size = num_rays * 8 * 4  # 8 floats * 4 bytes
        if geom_state.d_rays_size != rays_size:
            geom_state.d_rays = cupy.zeros(num_rays * 8, dtype=cupy.float32)
            geom_state.d_rays_size = rays_size
        geom_state.d_rays[:] = cupy.asarray(rays, dtype=cupy.float32)
        d_rays = geom_state.d_rays
        rays_on_host = True

    # Ensure hits buffer is on GPU
    if isinstance(hits, cupy.ndarray):
        d_hits = hits
        hits_on_host = False
    else:
        # Allocate/resize device buffer if needed (per-instance)
        hits_size = num_rays * 4 * 4  # 4 floats * 4 bytes
        if geom_state.d_hits_size != hits_size:
            geom_state.d_hits = cupy.zeros(num_rays * 4, dtype=cupy.float32)
            geom_state.d_hits_size = hits_size
        d_hits = geom_state.d_hits
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

    # Default ray flags: cull back-facing triangles
    if ray_flags is None:
        ray_flags = 0x10  # OPTIX_RAY_FLAG_CULL_BACK_FACING_TRIANGLES

    # Pack params: 48 bytes (existing) + 48 bytes (heightfield) = 96 bytes
    # Heightfield fields: ptr(8) + width(4) + height(4) + sx(4) + sy(4) + ve(4) + tile(4) + ntx(4) + pad(4) = 40+8 pad?
    # Actually: Q(8) + ii(8) + ff(8) + f(4) + ii(8) + i_pad(4) = 40 → need pad to 48
    hf_data_ptr = 0
    hf_w = 0
    hf_h = 0
    hf_sx = 0.0
    hf_sy = 0.0
    hf_ve = 1.0
    hf_tile = 0
    hf_ntx = 0

    if geom_state.heightfield_data is not None:
        hf_data_ptr = geom_state.heightfield_data.data.ptr
        hf_w = geom_state.hf_width
        hf_h = geom_state.hf_height
        hf_sx = geom_state.hf_spacing_x
        hf_sy = geom_state.hf_spacing_y
        hf_ve = geom_state.hf_ve
        hf_tile = geom_state.hf_tile_size
        hf_ntx = geom_state.hf_num_tiles_x

    # Point cloud colors pointer
    pc_colors_ptr = 0
    if geom_state.point_colors is not None:
        pc_colors_ptr = geom_state.point_colors.data.ptr

    # Smooth normal table pointer
    sn_table_ptr = 0
    if geom_state.d_smooth_normal_table is not None:
        sn_table_ptr = geom_state.d_smooth_normal_table.data.ptr

    params_data = struct.pack(
        'QQQQQIIQiifffiiIQQ',
        trace_handle,           # 8
        d_rays.data.ptr,        # 8
        d_hits.data.ptr,        # 8
        d_prim_ids_ptr,         # 8
        d_inst_ids_ptr,         # 8
        ray_flags,              # 4
        0,                      # 4 padding
        hf_data_ptr,            # 8
        hf_w,                   # 4
        hf_h,                   # 4
        hf_sx,                  # 4
        hf_sy,                  # 4
        hf_ve,                  # 4
        hf_tile,                # 4
        hf_ntx,                 # 4
        0,                      # 4 padding for alignment
        pc_colors_ptr,          # 8
        sn_table_ptr,           # 8
    )
    _state.d_params[:] = cupy.frombuffer(np.frombuffer(params_data, dtype=np.uint8), dtype=cupy.uint8)

    # Launch
    optix.launch(
        _state.pipeline,
        0,  # stream
        _state.d_params.data.ptr,
        104,  # sizeof(Params)
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
    NVIDIA's OptiX ray tracing engine. Each RTX instance maintains its own
    isolated geometry state, allowing multiple instances to manage separate
    scenes without interference.

    Args:
        device: CUDA device ID to use (0, 1, 2, ...). If None (default),
                uses the currently active CuPy device. Use get_device_count()
                to see available devices.

    Example:
        # Use default device (device 0 or current CuPy device)
        rtx = RTX()

        # Use specific GPU
        rtx = RTX(device=1)

        # Multiple independent instances
        rtx1 = RTX()
        rtx2 = RTX()
        rtx1.add_geometry("mesh1", v1, i1)  # Only in rtx1's scene
        rtx2.add_geometry("mesh2", v2, i2)  # Only in rtx2's scene

    Note:
        The underlying OptiX context, pipeline, and shader binding table are
        shared across all RTX instances (singleton). However, geometry state
        (acceleration structures, meshes) is per-instance, providing isolation
        between different users/DataArrays.
    """

    def __init__(self, device: Optional[int] = None):
        """
        Initialize the RTX context.

        Args:
            device: CUDA device ID to use. If None, uses the current device.
        """
        _init_optix(device)
        # Each RTX instance has its own isolated geometry state
        self._geom_state = _GeometryState()

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
        return _build_accel(self._geom_state, hashValue, vertexBuffer, indexBuffer)

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
        return self._geom_state.current_hash

    # OptiX ray flag constants
    RAY_FLAG_NONE = 0x00
    RAY_FLAG_CULL_BACK_FACING = 0x10   # OPTIX_RAY_FLAG_CULL_BACK_FACING_TRIANGLES
    RAY_FLAG_TERMINATE_ON_FIRST_HIT = 0x04  # OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT
    # Combined flag for shadow/AO occlusion queries (early out + backface cull)
    RAY_FLAG_OCCLUSION = 0x10 | 0x04

    def trace(self, rays, hits, numRays: int, primitive_ids=None, instance_ids=None,
              ray_flags=None) -> int:
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
            ray_flags: Optional OptiX ray flags (unsigned int). Default is
                       RAY_FLAG_CULL_BACK_FACING. Use RAY_FLAG_OCCLUSION for
                       shadow/AO queries to enable early termination.

        Returns:
            0 on success, non-zero on error
        """
        return _trace_rays(self._geom_state, rays, hits, numRays, primitive_ids, instance_ids,
                           ray_flags=ray_flags)

    def pick(self, origin, direction) -> dict:
        """Fire a single ray and return hit info.

        Parameters
        ----------
        origin : array-like
            Ray origin (x, y, z).
        direction : array-like
            Ray direction (dx, dy, dz), will be normalized.

        Returns
        -------
        dict
            Keys: 'hit' (bool), 'geometry_id' (str or None),
            't' (float), 'normal' (tuple), 'position' (tuple),
            'primitive_id' (int), 'instance_id' (int).
        """
        o = np.asarray(origin, dtype=np.float32)
        d = np.asarray(direction, dtype=np.float32)
        d = d / (np.linalg.norm(d) + 1e-30)

        rays = cupy.array([o[0], o[1], o[2], 0.001,
                           d[0], d[1], d[2], 1e10], dtype=cupy.float32)
        hits = cupy.zeros(4, dtype=cupy.float32)
        prim_ids = cupy.full(1, -1, dtype=cupy.int32)
        inst_ids = cupy.full(1, -1, dtype=cupy.int32)

        self.trace(rays, hits, 1, primitive_ids=prim_ids, instance_ids=inst_ids)

        t = float(hits[0])
        if t > 0:
            iid = int(inst_ids[0])
            geom_list = self.list_geometries()
            geom_id = geom_list[iid] if 0 <= iid < len(geom_list) else None
            pos = o + d * t
            return {
                'hit': True,
                'geometry_id': geom_id,
                't': t,
                'normal': (float(hits[1]), float(hits[2]), float(hits[3])),
                'position': (float(pos[0]), float(pos[1]), float(pos[2])),
                'primitive_id': int(prim_ids[0]),
                'instance_id': iid,
            }
        return {
            'hit': False,
            'geometry_id': None,
            't': -1.0,
            'normal': (0.0, 0.0, 0.0),
            'position': (0.0, 0.0, 0.0),
            'primitive_id': -1,
            'instance_id': -1,
        }

    # -------------------------------------------------------------------------
    # Multi-GAS API
    # -------------------------------------------------------------------------

    def add_geometry(self, geometry_id: str, vertices, indices,
                     transform: Optional[List[float]] = None,
                     grid_dims: Optional[tuple] = None,
                     normals=None) -> int:
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
            grid_dims: Optional (H, W) grid dimensions for cluster-accelerated
                      builds.  When provided and OptiX 9+ clusters are
                      available, uses the CLAS pipeline for faster BVH builds.
            normals: Optional per-vertex normal buffer (flattened float32 array,
                    3 floats per vertex).  When provided, the closest-hit
                    shader interpolates smooth normals using barycentrics
                    instead of computing flat face normals.

        Returns:
            0 on success, non-zero on error
        """
        global _state

        if not _state.initialized:
            _init_optix()

        # Switch to multi-GAS mode if currently in single-GAS mode
        if self._geom_state.single_gas_mode:
            self._geom_state.gas_handle = 0
            self._geom_state.gas_buffer = None
            self._geom_state.current_hash = 0xFFFFFFFFFFFFFFFF
            self._geom_state.single_gas_mode = False

        # Compute hash to skip GAS rebuild when vertices haven't changed
        if isinstance(vertices, cupy.ndarray):
            vertices_for_hash = vertices.get()
        else:
            vertices_for_hash = np.asarray(vertices)
        vertices_hash = hash(vertices_for_hash.tobytes())

        existing = self._geom_state.gas_entries.get(geometry_id)
        if existing is not None and existing.vertices_hash == vertices_hash:
            # GAS already built for identical vertices — update transform/normals only
            if transform is not None:
                existing.transform = list(transform)
                self._geom_state.ias_dirty = True
            if normals is not None:
                existing.d_normals = cupy.asarray(
                    np.asarray(normals, dtype=np.float32))
                if existing.d_indices is None:
                    existing.d_indices = cupy.asarray(
                        np.asarray(indices, dtype=np.int32))
                self._geom_state.ias_dirty = True
            return 0

        # Build the GAS for this geometry
        use_clusters = (
            grid_dims is not None
            and _state.capabilities
            and _state.capabilities.get('has_clusters')
        )
        if use_clusters:
            gas_handle, gas_buffer = _build_gas_clustered(
                vertices, indices, grid_dims[0], grid_dims[1])
        else:
            gas_handle, gas_buffer = _build_gas_for_geometry(vertices, indices)
        if gas_handle == 0:
            return -1

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

        # Compute vertex/triangle counts from input arrays
        num_vertices = len(vertices_for_hash.ravel()) // 3
        indices_np = indices.get() if isinstance(indices, cupy.ndarray) else np.asarray(indices)
        num_triangles = len(indices_np.ravel()) // 3

        # Upload smooth normals and index buffer if provided
        d_normals_gpu = None
        d_indices_gpu = None
        if normals is not None:
            d_normals_gpu = cupy.asarray(
                np.asarray(normals, dtype=np.float32))
            d_indices_gpu = cupy.asarray(indices_np)

        # Create or update the GAS entry
        self._geom_state.gas_entries[geometry_id] = _GASEntry(
            gas_id=geometry_id,
            gas_handle=gas_handle,
            gas_buffer=gas_buffer,
            vertices_hash=vertices_hash,
            transform=transform,
            num_vertices=num_vertices,
            num_triangles=num_triangles,
            d_normals=d_normals_gpu,
            d_indices=d_indices_gpu,
        )

        # Mark IAS as needing rebuild
        self._geom_state.ias_dirty = True

        return 0

    def add_curve_geometry(self, geometry_id: str, vertices, widths,
                           indices,
                           transform: Optional[List[float]] = None) -> int:
        """
        Add round quadratic B-spline curve tubes to the scene.

        This enables multi-GAS mode. Curve GAS entries use a separate
        hit group with the built-in curve IS module.

        Args:
            geometry_id: Unique identifier for this geometry
            vertices: Control point positions (N*3 float32, flattened)
            widths: Per-control-point radii (N float32)
            indices: Segment start indices (M int32, one per segment)
            transform: Optional 12-float 3x4 row-major affine transform.

        Returns:
            0 on success, non-zero on error
        """
        global _state

        if not _state.initialized:
            _init_optix()

        # Switch to multi-GAS mode if currently in single-GAS mode
        if self._geom_state.single_gas_mode:
            self._geom_state.gas_handle = 0
            self._geom_state.gas_buffer = None
            self._geom_state.current_hash = 0xFFFFFFFFFFFFFFFF
            self._geom_state.single_gas_mode = False

        # Compute hash to skip GAS rebuild when vertices haven't changed
        if isinstance(vertices, cupy.ndarray):
            vertices_for_hash = vertices.get()
        else:
            vertices_for_hash = np.asarray(vertices)
        vertices_hash = hash(vertices_for_hash.tobytes())

        existing = self._geom_state.gas_entries.get(geometry_id)
        if existing is not None and existing.vertices_hash == vertices_hash:
            if transform is not None:
                existing.transform = list(transform)
                self._geom_state.ias_dirty = True
            return 0

        # Compute segment count from indices
        indices_np = indices.get() if isinstance(indices, cupy.ndarray) else np.asarray(indices)
        num_segments = len(indices_np)

        gas_handle, gas_buffer = _build_gas_for_curves(
            vertices, widths, indices, num_segments)
        if gas_handle == 0:
            return -1

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

        num_vertices = len(vertices_for_hash.ravel()) // 3

        self._geom_state.gas_entries[geometry_id] = _GASEntry(
            gas_id=geometry_id,
            gas_handle=gas_handle,
            gas_buffer=gas_buffer,
            vertices_hash=vertices_hash,
            transform=transform,
            num_vertices=num_vertices,
            num_triangles=0,
            is_curve=True,
        )

        self._geom_state.ias_dirty = True
        return 0

    def add_heightfield_geometry(self, geometry_id: str, elevation,
                                 H: int, W: int,
                                 spacing_x: float, spacing_y: float,
                                 ve: float = 1.0,
                                 tile_size: int = 32,
                                 active_mask=None,
                                 transform=None) -> int:
        """
        Add a heightfield terrain as a custom-primitive GAS.

        The terrain is represented as a set of tiled AABBs. A custom
        intersection program ray-marches through the grid at trace time,
        never materializing an explicit triangle mesh. This dramatically
        reduces GPU memory for large terrains and provides smooth bilinear
        normals.

        Args:
            geometry_id: Unique identifier (typically 'terrain').
            elevation: 2-D array (H, W) of float32 elevation values (numpy or cupy).
            H: Number of rows.
            W: Number of columns.
            spacing_x: World-space pixel spacing in X.
            spacing_y: World-space pixel spacing in Y.
            ve: Vertical exaggeration. Default 1.0.
            tile_size: Tile dimension for AABB grouping. Default 32.
            active_mask: Optional bool array (one per AABB tile). When
                provided, inactive tiles get zero-volume AABBs.
            transform: Optional 12-float affine transform (3x4 row-major).
                Defaults to identity.

        Returns:
            0 on success, non-zero on error.
        """
        global _state

        if not _state.initialized:
            _init_optix()

        # Switch to multi-GAS mode
        if self._geom_state.single_gas_mode:
            self._geom_state.gas_handle = 0
            self._geom_state.gas_buffer = None
            self._geom_state.current_hash = 0xFFFFFFFFFFFFFFFF
            self._geom_state.single_gas_mode = False

        # Get elevation as numpy
        if hasattr(elevation, 'get'):
            elev_np = elevation.get()
        else:
            elev_np = np.asarray(elevation, dtype=np.float32)

        gas_handle, gas_buffer, d_elevation, num_tiles_x, num_tiles_y = \
            _build_gas_for_heightfield(elev_np, H, W, spacing_x, spacing_y, ve, tile_size, active_mask)

        if gas_handle == 0:
            return -1

        # Store heightfield metadata on geometry state for params packing
        self._geom_state.heightfield_data = d_elevation
        self._geom_state.hf_width = W
        self._geom_state.hf_height = H
        self._geom_state.hf_spacing_x = spacing_x
        self._geom_state.hf_spacing_y = spacing_y
        self._geom_state.hf_ve = ve
        self._geom_state.hf_tile_size = tile_size
        self._geom_state.hf_num_tiles_x = num_tiles_x

        if transform is None:
            transform = [
                1.0, 0.0, 0.0, 0.0,
                0.0, 1.0, 0.0, 0.0,
                0.0, 0.0, 1.0, 0.0,
            ]

        # Compute hash for cache invalidation
        vertices_hash = hash(elev_np.tobytes())

        self._geom_state.gas_entries[geometry_id] = _GASEntry(
            gas_id=geometry_id,
            gas_handle=gas_handle,
            gas_buffer=gas_buffer,
            vertices_hash=vertices_hash,
            transform=transform,
            num_vertices=0,
            num_triangles=0,
            is_heightfield=True,
        )

        self._geom_state.ias_dirty = True
        return 0

    def add_sphere_geometry(self, geometry_id: str, centers, radii,
                            colors=None,
                            transform: Optional[List[float]] = None) -> int:
        """
        Add sphere primitives to the scene (for point cloud rendering).

        Uses OptiX built-in sphere intersection for hardware-accelerated
        ray-sphere tests. Each sphere is defined by a center point and radius.

        Args:
            geometry_id: Unique identifier for this geometry.
            centers: Sphere center positions (N*3 float32, flattened or Nx3).
            radii: Per-sphere radii (N float32), or a single float for uniform radius.
            colors: Optional per-point RGBA colors (N*4 float32). If provided,
                    stored on geometry state for use by the shade kernel.
            transform: Optional 12-float 3x4 row-major affine transform.

        Returns:
            0 on success, non-zero on error.
        """
        global _state

        if not _state.initialized:
            _init_optix()

        # Switch to multi-GAS mode if currently in single-GAS mode
        if self._geom_state.single_gas_mode:
            self._geom_state.gas_handle = 0
            self._geom_state.gas_buffer = None
            self._geom_state.current_hash = 0xFFFFFFFFFFFFFFFF
            self._geom_state.single_gas_mode = False

        # Prepare centers
        if isinstance(centers, cupy.ndarray):
            centers_for_hash = centers.get()
        else:
            centers_for_hash = np.asarray(centers, dtype=np.float32)
        vertices_hash = hash(centers_for_hash.tobytes())

        existing = self._geom_state.gas_entries.get(geometry_id)
        if existing is not None and existing.vertices_hash == vertices_hash:
            if transform is not None:
                existing.transform = list(transform)
                self._geom_state.ias_dirty = True
            return 0

        # Determine if single radius
        single_radius = np.isscalar(radii)

        gas_handle, gas_buffer = _build_gas_for_spheres(
            centers, radii, single_radius=single_radius)
        if gas_handle == 0:
            return -1

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

        num_vertices = len(centers_for_hash.ravel()) // 3

        self._geom_state.gas_entries[geometry_id] = _GASEntry(
            gas_id=geometry_id,
            gas_handle=gas_handle,
            gas_buffer=gas_buffer,
            vertices_hash=vertices_hash,
            transform=transform,
            num_vertices=num_vertices,
            num_triangles=0,
            is_sphere=True,
        )

        # Store per-point colors for this GAS
        if colors is not None:
            self._geom_state.point_colors_per_gas[geometry_id] = np.asarray(
                colors, dtype=np.float32)
            # Invalidate concatenated buffer so it gets rebuilt
            self._geom_state.point_colors = None
            self._geom_state.point_color_offsets = None

        self._geom_state.ias_dirty = True
        return 0

    def remove_geometry(self, geometry_id: str) -> int:
        """
        Remove a geometry from the scene.

        Args:
            geometry_id: The ID of the geometry to remove

        Returns:
            0 on success, -1 if geometry not found
        """
        if geometry_id not in self._geom_state.gas_entries:
            return -1

        del self._geom_state.gas_entries[geometry_id]
        self._geom_state.ias_dirty = True

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
        if geometry_id not in self._geom_state.gas_entries:
            return -1

        transform = list(transform)
        if len(transform) != 12:
            return -1

        self._geom_state.gas_entries[geometry_id].transform = transform
        self._geom_state.ias_dirty = True

        return 0

    def list_geometries(self) -> List[str]:
        """
        Get a list of all geometry IDs in the scene.

        Returns:
            List of geometry ID strings
        """
        return list(self._geom_state.gas_entries.keys())

    def get_geometry_count(self) -> int:
        """
        Get the number of geometries in the scene.

        Returns:
            Number of geometries (0 in single-GAS mode)
        """
        return len(self._geom_state.gas_entries)

    def build_point_colors_gpu(self):
        """Build concatenated per-point color buffer and per-instance offsets.

        Returns (point_colors, point_color_offsets) as cupy arrays, or
        (None, None) if no sphere geometries have per-point colors.

        point_colors: (total_points * 4,) float32 — RGBA per point
        point_color_offsets: (num_instances,) int32 — offset into
            point_colors for each IAS instance (-1 = no per-point colors)
        """
        gs = self._geom_state
        if not gs.point_colors_per_gas:
            return None, None

        # Return cached if still valid
        if gs.point_colors is not None and gs.point_color_offsets is not None:
            return gs.point_colors, gs.point_color_offsets

        geom_ids = list(gs.gas_entries.keys())
        n_instances = len(geom_ids)
        offsets = np.full(n_instances, -1, dtype=np.int32)
        parts = []
        cumulative = 0

        for i, gid in enumerate(geom_ids):
            if gid in gs.point_colors_per_gas:
                colors_flat = gs.point_colors_per_gas[gid]
                n_points = len(colors_flat) // 4
                offsets[i] = cumulative
                parts.append(colors_flat)
                cumulative += n_points

        if not parts:
            return None, None

        all_colors = np.concatenate(parts)
        gs.point_colors = cupy.asarray(all_colors, dtype=cupy.float32)
        gs.point_color_offsets = cupy.asarray(offsets, dtype=cupy.int32)
        return gs.point_colors, gs.point_color_offsets

    def has_geometry(self, geometry_id: str) -> bool:
        """
        Check if a geometry with the given ID exists.

        Args:
            geometry_id: The ID of the geometry to check.

        Returns:
            True if the geometry exists, False otherwise.
        """
        return geometry_id in self._geom_state.gas_entries

    def get_geometry_transform(self, geometry_id: str) -> Optional[List[float]]:
        """
        Get the transform of a geometry.

        Args:
            geometry_id: The ID of the geometry.

        Returns:
            12-float list representing the 3x4 transform matrix, or None if not found.
            Format: [Xx, Xy, Xz, Tx, Yx, Yy, Yz, Ty, Zx, Zy, Zz, Tz]
            The translation (position) is at indices 3, 7, 11 (Tx, Ty, Tz).
        """
        if geometry_id not in self._geom_state.gas_entries:
            return None
        return self._geom_state.gas_entries[geometry_id].transform.copy()

    def set_geometry_visible(self, geometry_id: str, visible: bool) -> int:
        """
        Set whether a geometry is visible to rays.

        Uses the OptiX visibility mask to hide/show geometries without
        removing them from the scene.

        Args:
            geometry_id: The ID of the geometry to show/hide.
            visible: True to make visible, False to hide.

        Returns:
            0 on success, -1 if geometry not found.
        """
        if geometry_id not in self._geom_state.gas_entries:
            return -1
        self._geom_state.gas_entries[geometry_id].visible = visible
        self._geom_state.ias_dirty = True
        return 0

    def clear_scene(self) -> None:
        """
        Remove all geometries and reset to single-GAS mode.

        After calling this, you can use either build() for single-GAS mode
        or add_geometry() for multi-GAS mode.
        """
        self._geom_state.clear()

    def memory_usage(self) -> dict:
        """Return a breakdown of GPU memory used by the scene.

        Returns
        -------
        dict
            Keys: mode, geometries (list of per-geometry dicts),
            ias_bytes, instances_bytes, ray_buffers_bytes, total_bytes.
        """
        gs = self._geom_state

        if gs.gas_entries:
            mode = 'multi-gas'
        elif gs.gas_buffer is not None:
            mode = 'single-gas'
        else:
            mode = 'empty'

        geometries = []
        total_gas = 0

        if mode == 'multi-gas':
            for gid, entry in gs.gas_entries.items():
                nbytes = entry.gas_buffer.nbytes if entry.gas_buffer is not None else 0
                total_gas += nbytes
                geometries.append({
                    'id': gid,
                    'gas_bytes': nbytes,
                    'num_vertices': entry.num_vertices,
                    'num_triangles': entry.num_triangles,
                    'visible': entry.visible,
                })
        elif mode == 'single-gas' and gs.gas_buffer is not None:
            total_gas = gs.gas_buffer.nbytes
            geometries.append({
                'id': 'single-gas',
                'gas_bytes': total_gas,
                'num_vertices': 0,
                'num_triangles': 0,
                'visible': True,
            })

        ias_bytes = gs.ias_buffer.nbytes if gs.ias_buffer is not None else 0
        instances_bytes = gs.instances_buffer.nbytes if gs.instances_buffer is not None else 0
        ray_buffers_bytes = gs.d_rays_size + gs.d_hits_size

        total_bytes = total_gas + ias_bytes + instances_bytes + ray_buffers_bytes

        return {
            'mode': mode,
            'geometries': geometries,
            'ias_bytes': ias_bytes,
            'instances_bytes': instances_bytes,
            'ray_buffers_bytes': ray_buffers_bytes,
            'total_bytes': total_bytes,
        }


# -----------------------------------------------------------------------------
# OptiX AI Denoiser
# -----------------------------------------------------------------------------

def _ensure_denoiser(width, height, temporal=False):
    """Create or reconfigure the OptiX AI denoiser for the given dimensions.

    Parameters
    ----------
    temporal : bool
        If True, use DENOISER_MODEL_KIND_TEMPORAL (requires flow vectors).
        If False, use DENOISER_MODEL_KIND_HDR (spatial only).

    Returns True if the denoiser is ready, False if unavailable.
    """
    global _state

    if _state._denoiser_failed:
        return False

    if not _state.initialized:
        _init_optix()

    # Recreate denoiser if mode changed or not yet created
    need_create = _state.denoiser is None
    if not need_create and _state._denoiser_temporal != temporal:
        _state.denoiser = None
        _state.denoiser_width = 0
        _state.denoiser_height = 0
        need_create = True

    if need_create:
        opts = optix.DenoiserOptions()
        opts.guideNormal = 1
        opts.guideAlbedo = 1
        model = (optix.DENOISER_MODEL_KIND_TEMPORAL if temporal
                 else optix.DENOISER_MODEL_KIND_HDR)
        try:
            _state.denoiser = _state.context.denoiserCreate(model, opts)
        except RuntimeError:
            import warnings
            warnings.warn(
                "OptiX AI Denoiser unavailable (missing nvoptix.bin "
                "weights file). Denoising will be skipped.",
                RuntimeWarning)
            _state._denoiser_failed = True
            return False
        _state._denoiser_temporal = temporal

    if _state.denoiser_width != width or _state.denoiser_height != height:
        sizes = _state.denoiser.computeMemoryResources(width, height)
        _state.d_denoiser_state = cupy.empty(
            sizes.stateSizeInBytes, dtype=cupy.uint8)
        _state.d_denoiser_scratch = cupy.empty(
            sizes.withoutOverlapScratchSizeInBytes, dtype=cupy.uint8)
        _state.denoiser.setup(
            0,  # stream
            width, height,
            _state.d_denoiser_state.data.ptr, sizes.stateSizeInBytes,
            _state.d_denoiser_scratch.data.ptr,
            sizes.withoutOverlapScratchSizeInBytes)
        _state.d_denoiser_normals = cupy.empty(
            (height, width, 3), dtype=cupy.float32)
        _state.d_denoiser_output = cupy.empty(
            (height, width, 3), dtype=cupy.float32)
        _state.d_denoiser_albedo = cupy.empty(
            (height, width, 3), dtype=cupy.float32)
        if temporal:
            _state.d_denoiser_flow = cupy.zeros(
                (height, width, 3), dtype=cupy.float32)
        _state.denoiser_width = width
        _state.denoiser_height = height

    return True


def denoise(d_color, d_normals, width, height, cam_right, cam_up, cam_forward,
            albedo=None, flow=None):
    """Apply the OptiX AI Denoiser to a noisy HDR image.

    Parameters
    ----------
    d_color : cupy.ndarray
        (height, width, 3) float32 HDR color buffer. Modified in-place
        with denoised result.
    d_normals : cupy.ndarray
        (height, width, 3) float32 world-space hit normals.
    width, height : int
        Image dimensions.
    cam_right, cam_up, cam_forward : array-like
        Camera basis vectors (3,) for transforming normals to camera space.
    albedo : cupy.ndarray, optional
        (height, width, 3) float32 albedo guide (material color before lighting).
    flow : cupy.ndarray, optional
        (height, width, 2) float32 screen-space motion vectors (pixels).
        If provided, temporal denoising is used.
    """
    global _state
    temporal = flow is not None
    if not _ensure_denoiser(width, height, temporal=temporal):
        return

    # Transform world-space normals to camera space via matrix multiply.
    # Column matrix: columns = right, up, forward.
    d_basis = cupy.asarray(
        np.stack([np.asarray(cam_right, dtype=np.float32),
                  np.asarray(cam_up, dtype=np.float32),
                  np.asarray(cam_forward, dtype=np.float32)], axis=1),
        dtype=cupy.float32)  # (3, 3)
    flat_normals = d_normals.reshape(-1, 3)
    _state.d_denoiser_normals.reshape(-1, 3)[:] = flat_normals @ d_basis

    row_stride_3 = width * 3 * 4  # 3 float32 × 4 bytes
    pixel_stride_3 = 3 * 4

    color_image = optix.Image2D()
    color_image.data = d_color.data.ptr
    color_image.width = width
    color_image.height = height
    color_image.rowStrideInBytes = row_stride_3
    color_image.pixelStrideInBytes = pixel_stride_3
    color_image.format = optix.PIXEL_FORMAT_FLOAT3

    output_image = optix.Image2D()
    output_image.data = _state.d_denoiser_output.data.ptr
    output_image.width = width
    output_image.height = height
    output_image.rowStrideInBytes = row_stride_3
    output_image.pixelStrideInBytes = pixel_stride_3
    output_image.format = optix.PIXEL_FORMAT_FLOAT3

    normal_image = optix.Image2D()
    normal_image.data = _state.d_denoiser_normals.data.ptr
    normal_image.width = width
    normal_image.height = height
    normal_image.rowStrideInBytes = row_stride_3
    normal_image.pixelStrideInBytes = pixel_stride_3
    normal_image.format = optix.PIXEL_FORMAT_FLOAT3

    layer = optix.DenoiserLayer()
    layer.input = color_image
    layer.output = output_image

    guide = optix.DenoiserGuideLayer()
    guide.normal = normal_image

    # Albedo guide
    if albedo is not None:
        _state.d_denoiser_albedo[:] = albedo
        albedo_image = optix.Image2D()
        albedo_image.data = _state.d_denoiser_albedo.data.ptr
        albedo_image.width = width
        albedo_image.height = height
        albedo_image.rowStrideInBytes = row_stride_3
        albedo_image.pixelStrideInBytes = pixel_stride_3
        albedo_image.format = optix.PIXEL_FORMAT_FLOAT3
        guide.albedo = albedo_image

    # Flow guide (temporal denoising)
    if temporal:
        # Flow is (H, W, 2) — copy into padded (H, W, 3) buffer for FLOAT3 format
        _state.d_denoiser_flow[:, :, :2] = flow
        flow_image = optix.Image2D()
        flow_image.data = _state.d_denoiser_flow.data.ptr
        flow_image.width = width
        flow_image.height = height
        flow_image.rowStrideInBytes = row_stride_3
        flow_image.pixelStrideInBytes = pixel_stride_3
        flow_image.format = optix.PIXEL_FORMAT_FLOAT3
        guide.flow = flow_image

    params = optix.DenoiserParams()
    params.blendFactor = 0.0

    _state.denoiser.invoke(
        0,  # stream
        params,
        _state.d_denoiser_state.data.ptr,
        _state.d_denoiser_state.nbytes,
        guide,
        layer,
        1,     # numLayers
        0, 0,  # inputOffsetX, inputOffsetY
        _state.d_denoiser_scratch.data.ptr,
        _state.d_denoiser_scratch.nbytes,
    )

    # Copy denoised result back into the input buffer
    d_color[:] = _state.d_denoiser_output
