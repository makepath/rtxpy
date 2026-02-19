"""Common utilities shared between analysis modules.

Internal module providing shared ray generation kernels and mesh preparation.
"""

from numba import cuda
import numpy as np

from .._cuda_utils import calc_dims
from ..mesh import triangulate_terrain, voxelate_terrain
from ..rtx import RTX, has_cupy

if has_cupy:
    import cupy


def _compute_pixel_spacing(da):
    """Derive real-world pixel spacing from a DataArray's x/y coordinates.

    Parameters
    ----------
    da : xarray.DataArray
        Raster with 'x' and 'y' coordinate arrays.

    Returns
    -------
    tuple of float
        (pixel_spacing_x, pixel_spacing_y) in the CRS's linear units.
        Falls back to (1.0, 1.0) when coordinates are missing, too short,
        or the CRS uses geographic (degree) units.
    """
    try:
        x = da.coords['x'].values
        y = da.coords['y'].values
    except (KeyError, AttributeError):
        return (1.0, 1.0)

    if len(x) < 2 or len(y) < 2:
        return (1.0, 1.0)

    # Guard against geographic CRS (degrees) — spacing in degrees is not
    # meaningful as a metric distance, so fall back to pixel coords.
    try:
        crs = da.rio.crs
        if crs is not None and crs.is_geographic:
            return (1.0, 1.0)
    except Exception:
        pass  # rioxarray not available or no CRS set — proceed with diffs

    psx = float(abs(x[1] - x[0]))
    psy = float(abs(y[1] - y[0]))

    if psx == 0 or psy == 0:
        return (1.0, 1.0)

    return (psx, psy)


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


def generate_primary_rays(rays, x_coords, y_coords, H, W,
                          pixel_spacing_x=1.0, pixel_spacing_y=1.0):
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
    pixel_spacing_x : float, optional
        World-space spacing per pixel in X. Default 1.0.
    pixel_spacing_y : float, optional
        World-space spacing per pixel in Y. Default 1.0.

    Returns
    -------
    int
        0 on success.
    """
    griddim, blockdim = calc_dims((H, W))
    # Ensure coordinate arrays are contiguous and writable for numba transfer
    # (even though they're unused in the kernel, numba still transfers them
    # and requires writable buffers for the copy-back mechanism)
    if hasattr(x_coords, 'get'):
        x_coords = cupy.ascontiguousarray(x_coords)
    else:
        x_coords = np.ascontiguousarray(x_coords)
        if not x_coords.flags.writeable:
            x_coords = x_coords.copy()
    if hasattr(y_coords, 'get'):
        y_coords = cupy.ascontiguousarray(y_coords)
    else:
        y_coords = np.ascontiguousarray(y_coords)
        if not y_coords.flags.writeable:
            y_coords = y_coords.copy()
    _generate_primary_rays_kernel[griddim, blockdim](rays, x_coords, y_coords, H, W)

    # Scale ray origins from pixel space to world space
    if pixel_spacing_x != 1.0 or pixel_spacing_y != 1.0:
        rays[:, :, 0] *= pixel_spacing_x
        rays[:, :, 1] *= pixel_spacing_y

    return 0


def prepare_mesh(raster, rtx=None, mesh_type='heightfield',
                 pixel_spacing_x=1.0, pixel_spacing_y=1.0):
    """Prepare a triangle mesh from raster data and build the RTX acceleration structure.

    This function handles the common pattern of:
    1. Creating or reusing an RTX instance
    2. Checking if the mesh needs rebuilding (via hash comparison)
    3. Triangulating or voxelating the terrain
    4. Scaling X/Y to world coordinates using pixel_spacing
    5. Building the GAS (Geometry Acceleration Structure)

    Parameters
    ----------
    raster : xarray.DataArray
        Raster terrain data with coordinates.
    rtx : RTX, optional
        Existing RTX instance to reuse. If None, a new instance is created.
    mesh_type : str, optional
        Mesh generation method: 'tin' or 'voxel'. Default is 'tin'.
    pixel_spacing_x : float, optional
        World-space spacing per pixel in X. Default 1.0.
    pixel_spacing_y : float, optional
        World-space spacing per pixel in Y. Default 1.0.

    Returns
    -------
    RTX
        The RTX instance with the built acceleration structure.

    Raises
    ------
    ValueError
        If mesh generation or GAS building fails.
    """
    valid_types = ('tin', 'voxel', 'heightfield')
    if mesh_type not in valid_types:
        raise ValueError(
            f"Invalid mesh_type '{mesh_type}'. Must be one of: {valid_types}"
        )

    if rtx is None:
        rtx = RTX()

    H, W = raster.shape

    if mesh_type == 'heightfield':
        # Heightfield path: upload raw elevation grid, no triangle mesh
        terrain_data = raster.data
        if hasattr(terrain_data, 'get'):
            elev_np = terrain_data.get().astype(np.float32)
        else:
            elev_np = np.asarray(terrain_data, dtype=np.float32)

        res = rtx.add_heightfield_geometry(
            'terrain', elev_np, H, W,
            spacing_x=pixel_spacing_x,
            spacing_y=pixel_spacing_y,
        )
        if res:
            raise ValueError(f"Failed to build heightfield GAS. Error code: {res}")
        return rtx

    # Include mesh_type and pixel_spacing in hash so changes trigger rebuild
    hash_str = str(raster.data.get()) + mesh_type + f'{pixel_spacing_x},{pixel_spacing_y}'
    datahash = np.uint64(hash(hash_str) % (1 << 64))
    optixhash = np.uint64(rtx.getHash())

    if optixhash != datahash:
        if mesh_type == 'voxel':
            numVerts = H * W * 8
            numTris = H * W * 12
            verts = cupy.empty(numVerts * 3, np.float32)
            triangles = cupy.empty(numTris * 3, np.int32)

            # Use terrain minimum as base elevation
            base_elevation = float(cupy.nanmin(raster.data))
            res = voxelate_terrain(verts, triangles, raster,
                                   base_elevation=base_elevation)
        else:
            numTris = (H - 1) * (W - 1) * 2
            verts = cupy.empty(H * W * 3, np.float32)
            triangles = cupy.empty(numTris * 3, np.int32)
            res = triangulate_terrain(verts, triangles, raster)

        if res:
            raise ValueError(f"Failed to generate mesh from terrain. Error code: {res}")

        # Scale vertex X/Y from pixel indices to world coordinates
        if pixel_spacing_x != 1.0 or pixel_spacing_y != 1.0:
            verts[0::3] *= pixel_spacing_x
            verts[1::3] *= pixel_spacing_y

        res = rtx.build(datahash, verts, triangles)
        if res:
            raise ValueError(f"OptiX failed to build GAS with error code: {res}")

        # Clear GPU memory
        del verts
        del triangles
        cupy.get_default_memory_pool().free_all_blocks()

    return rtx
