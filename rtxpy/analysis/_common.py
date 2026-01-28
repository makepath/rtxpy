"""Common utilities shared between analysis modules.

Internal module providing shared ray generation kernels and mesh preparation.
"""

from numba import cuda
import numpy as np

from .._cuda_utils import calc_dims
from ..mesh import triangulate_terrain
from ..rtx import RTX, has_cupy

if has_cupy:
    import cupy


@cuda.jit
def _generate_primary_rays_kernel(data, x_coords, y_coords, H, W):
    """GPU kernel for generating orthographic camera rays looking straight down.

    Generates parallel rays from a virtual camera at height 10000 pointing
    downward (-Z direction). Each ray corresponds to one pixel in the output.
    """
    i, j = cuda.grid(2)
    if i >= 0 and i < H and j >= 0 and j < W:
        # Handle edge cases to avoid self-intersection at mesh boundaries
        if j == W - 1:
            data[i, j, 0] = j - 1e-3
        else:
            data[i, j, 0] = j + 1e-3

        if i == H - 1:
            data[i, j, 1] = i - 1e-3
        else:
            data[i, j, 1] = i + 1e-3

        data[i, j, 2] = 10000  # Camera height
        data[i, j, 3] = 1e-3   # t_min
        data[i, j, 4] = 0      # direction x
        data[i, j, 5] = 0      # direction y
        data[i, j, 6] = -1     # direction z (pointing down)
        data[i, j, 7] = np.inf  # t_max


def generate_primary_rays(rays, x_coords, y_coords, H, W):
    """Generate orthographic camera rays for terrain intersection.

    Parameters
    ----------
    rays : cupy.ndarray
        Output array of shape (H, W, 8) for ray data.
    x_coords : array-like
        X coordinates (unused in current implementation, for API compatibility).
    y_coords : array-like
        Y coordinates (unused in current implementation, for API compatibility).
    H : int
        Height of the raster.
    W : int
        Width of the raster.

    Returns
    -------
    int
        0 on success.
    """
    griddim, blockdim = calc_dims((H, W))
    # Ensure coordinate arrays are contiguous for numba transfer
    # (even though they're unused in the kernel, numba still transfers them)
    if hasattr(x_coords, 'get'):
        x_coords = cupy.ascontiguousarray(x_coords)
    else:
        x_coords = np.ascontiguousarray(x_coords)
    if hasattr(y_coords, 'get'):
        y_coords = cupy.ascontiguousarray(y_coords)
    else:
        y_coords = np.ascontiguousarray(y_coords)
    _generate_primary_rays_kernel[griddim, blockdim](rays, x_coords, y_coords, H, W)
    return 0


def prepare_mesh(raster, rtx=None):
    """Prepare a triangle mesh from raster data and build the RTX acceleration structure.

    This function handles the common pattern of:
    1. Creating or reusing an RTX instance
    2. Checking if the mesh needs rebuilding (via hash comparison)
    3. Triangulating the terrain
    4. Building the GAS (Geometry Acceleration Structure)

    Parameters
    ----------
    raster : xarray.DataArray
        Raster terrain data with coordinates.
    rtx : RTX, optional
        Existing RTX instance to reuse. If None, a new instance is created.

    Returns
    -------
    RTX
        The RTX instance with the built acceleration structure.

    Raises
    ------
    ValueError
        If mesh generation or GAS building fails.
    """
    if rtx is None:
        rtx = RTX()

    H, W = raster.shape

    # Check if we need to rebuild the mesh
    datahash = np.uint64(hash(str(raster.data.get())) % (1 << 64))
    optixhash = np.uint64(rtx.getHash())

    if optixhash != datahash:
        numTris = (H - 1) * (W - 1) * 2
        verts = cupy.empty(H * W * 3, np.float32)
        triangles = cupy.empty(numTris * 3, np.int32)

        # Generate mesh from terrain
        res = triangulate_terrain(verts, triangles, raster)
        if res:
            raise ValueError(f"Failed to generate mesh from terrain. Error code: {res}")

        res = rtx.build(datahash, verts, triangles)
        if res:
            raise ValueError(f"OptiX failed to build GAS with error code: {res}")

        # Clear GPU memory
        del verts
        del triangles
        cupy.get_default_memory_pool().free_all_blocks()

    return rtx
