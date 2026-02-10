"""Viewshed analysis using GPU-accelerated ray tracing.

This module computes visibility from a given observer point across a terrain,
determining which areas are visible and at what angle.
"""

from numba import cuda
import numpy as np
import math

from typing import Union

from .._cuda_utils import calc_dims, add, diff, mul, dot, float3, make_float3, invert
from ._common import generate_primary_rays, prepare_mesh
from ..rtx import RTX, has_cupy

if has_cupy:
    import cupy

# Default viewshed parameters
_OBS_ELEV = 0
_TARGET_ELEV = 0

# Value for invisible cells
INVISIBLE = -1


def _lazy_import_xarray():
    """Lazily import xarray with helpful error message."""
    try:
        import xarray as xr
        return xr
    except ImportError:
        raise ImportError(
            "xarray is required for viewshed analysis. "
            "Install it with: pip install xarray "
            "or: pip install rtxpy[analysis]"
        )


@cuda.jit
def _generate_viewshed_rays_kernel(camRays, hits, vsrays, visibility_grid, H, W, vp):
    """GPU kernel for generating rays from surface points to the viewshed origin."""
    i, j = cuda.grid(2)
    if i >= 0 and i < H and j >= 0 and j < W:
        elevationOffset = vp[2]
        targetElevation = vp[3]
        dist = hits[i, j, 0]  # distance to surface from camera
        norm = make_float3(hits[i, j], 1)  # normal vector at intersection

        if norm[2] < 0:  # if back hit, face forward
            norm = invert(norm)

        cameraRay = camRays[i, j]
        rayOrigin = make_float3(cameraRay, 0)
        rayDir = make_float3(cameraRay, 4)
        hitP = add(rayOrigin, mul(rayDir, dist))
        newOrigin = add(hitP, mul(norm, 1e-3))  # offset to avoid self-intersection
        newOrigin = add(newOrigin, float3(0, 0, targetElevation))

        w = int(vp[0])
        h = int(vp[1])
        viewshedRay = camRays[h, w]
        dist = hits[h, w, 0]
        rayOrigin = make_float3(viewshedRay, 0)
        rayDir = make_float3(viewshedRay, 4)
        hitP = add(rayOrigin, mul(rayDir, dist))
        viewshedPoint = add(hitP, float3(0, 0, elevationOffset))

        newDir = diff(viewshedPoint, newOrigin)
        length = math.sqrt(dot(newDir, newDir))
        newDir = mul(newDir, 1 / length)

        cosine = dot(norm, newDir)
        theta = math.acos(cosine)
        theta = (180 * theta) / math.pi

        # Prepare viewshed ray
        vsray = vsrays[i, j]
        vsray[0] = newOrigin[0]
        vsray[1] = newOrigin[1]
        vsray[2] = newOrigin[2]
        vsray[3] = 0
        vsray[4] = newDir[0]
        vsray[5] = newDir[1]
        vsray[6] = newDir[2]
        vsray[7] = length

        visibility_grid[i, j] = theta


def _generate_viewshed_rays(rays, hits, vsrays, visibility_grid, H, W, vp):
    """Generate rays from each surface point toward the viewshed origin."""
    griddim, blockdim = calc_dims((H, W))
    _generate_viewshed_rays_kernel[griddim, blockdim](
        rays, hits, vsrays, visibility_grid, H, W, vp
    )
    return 0


@cuda.jit
def _calc_viewshed_kernel(hits, visibility_grid, H, W):
    """GPU kernel to determine visibility based on ray intersection results."""
    i, j = cuda.grid(2)
    if i >= 0 and i < H and j >= 0 and j < W:
        dist = hits[i, j, 0]
        # If dist >= 0, something was hit along the ray path,
        # meaning the target is not visible from the viewpoint
        if dist >= 0:
            visibility_grid[i, j] = INVISIBLE


def _calc_viewshed(hits, visibility_grid, H, W):
    """Compute final visibility values from viewshed ray hits."""
    griddim, blockdim = calc_dims((H, W))
    _calc_viewshed_kernel[griddim, blockdim](hits, visibility_grid, H, W)
    return 0


def _viewshed_rt(raster, optix, x, y, observer_elev, target_elev,
                  pixel_spacing_x=1.0, pixel_spacing_y=1.0,
                  between_traces_cb=None):
    """Internal function to perform viewshed ray tracing.

    Parameters
    ----------
    pixel_spacing_x, pixel_spacing_y : float
        World-space spacing per pixel.  When the terrain mesh has been built
        with pixel_spacing applied (as in the interactive viewer), these must
        match so the orthographic primary rays land on the mesh.
    between_traces_cb : callable, optional
        Called after the first trace (terrain surface detection) and before
        the second trace (occlusion check).  Use this to change geometry
        visibility so that the first trace hits only terrain while the
        second trace includes buildings / structures.
    """
    xr = _lazy_import_xarray()

    H, W = raster.shape

    y_coords = raster.indexes.get('y').values
    x_coords = raster.indexes.get('x').values

    # Validate x argument
    if x < x_coords.min() or x > x_coords.max():
        raise ValueError("x argument outside of raster x_range")

    # Validate y argument
    if y < y_coords.min() or y > y_coords.max():
        raise ValueError("y argument outside of raster y_range")

    selection = raster.sel(x=[x], y=[y], method='nearest')
    x = selection.x.values[0]
    y = selection.y.values[0]

    y_view = np.where(y_coords == y)[0][0]
    x_view = np.where(x_coords == x)[0][0]

    # Device buffers
    d_rays = cupy.empty((H, W, 8), np.float32)
    d_hits = cupy.empty((H, W, 4), np.float32)
    d_visgrid = cupy.empty((H, W), np.float32)
    d_vsrays = cupy.empty((H, W, 8), np.float32)

    generate_primary_rays(d_rays, x_coords, y_coords, H, W)

    # Scale ray origins from pixel space to world space so they intersect
    # the terrain mesh when it has been built with pixel_spacing applied.
    if pixel_spacing_x != 1.0 or pixel_spacing_y != 1.0:
        d_rays[:, :, 0] *= pixel_spacing_x
        d_rays[:, :, 1] *= pixel_spacing_y

    device = cupy.cuda.Device(0)
    device.synchronize()
    optix.trace(d_rays, d_hits, W * H)

    # Allow caller to change visibility between the two traces so
    # buildings/structures occlude the viewshed rays.
    if between_traces_cb is not None:
        between_traces_cb()

    vp = (x_view, y_view, observer_elev, target_elev)
    _generate_viewshed_rays(d_rays, d_hits, d_vsrays, d_visgrid, H, W, vp)
    device.synchronize()
    optix.trace(d_vsrays, d_hits, W * H)

    _calc_viewshed(d_hits, d_visgrid, H, W)

    if isinstance(raster.data, np.ndarray):
        visgrid = cupy.asnumpy(d_visgrid)
    else:
        visgrid = d_visgrid

    view = xr.DataArray(
        visgrid,
        name="viewshed",
        coords=raster.coords,
        dims=raster.dims,
        attrs=raster.attrs
    )
    return view


def viewshed(raster,
             x: Union[int, float],
             y: Union[int, float],
             observer_elev: float = _OBS_ELEV,
             target_elev: float = _TARGET_ELEV,
             rtx: RTX = None):
    """Compute viewshed from an observer point on the terrain.

    The viewshed analysis determines which areas of the terrain are visible
    from a given observer location, using GPU-accelerated ray tracing.

    Parameters
    ----------
    raster : xarray.DataArray
        2D raster terrain data with 'x' and 'y' coordinates.
        Data must be a cupy array on the GPU.
    x : int or float
        X coordinate of the observer location (in raster coordinate units).
    y : int or float
        Y coordinate of the observer location (in raster coordinate units).
    observer_elev : float, optional
        Height offset above the terrain surface for the observer. Default is 0.
    target_elev : float, optional
        Height offset above the terrain surface for target points. Default is 0.
    rtx : RTX, optional
        Existing RTX instance to reuse. If None, a new instance is created
        and the terrain mesh is built automatically.

    Returns
    -------
    xarray.DataArray
        Visibility raster with the same coordinates as the input.
        Values indicate the viewing angle in degrees for visible cells,
        or -1 (INVISIBLE) for cells not visible from the observer.

    Raises
    ------
    ValueError
        If x or y coordinates are outside the raster extent.
        If raster.data is not a cupy array.
        If mesh generation or ray tracing fails.

    Examples
    --------
    >>> import xarray as xr
    >>> import cupy
    >>> # Load terrain data
    >>> terrain = xr.open_dataarray('dem.tif')
    >>> terrain = terrain.assign({'data': cupy.array(terrain.data)})
    >>> # Compute viewshed from point
    >>> vis = rtxpy.viewshed(terrain, x=500000, y=4500000, observer_elev=2)
    """
    xr = _lazy_import_xarray()

    if not has_cupy:
        raise ImportError(
            "cupy is required for viewshed analysis. "
            "Install it with: conda install -c conda-forge cupy"
        )

    if not isinstance(raster.data, cupy.ndarray):
        raise ValueError("raster.data must be a cupy array")

    # If an RTX with existing geometries is provided (multi-GAS scene),
    # use it directly so viewshed rays are occluded by all scene geometry.
    # Only build a terrain-only mesh when no RTX is given.
    if rtx is not None and rtx.get_geometry_count() > 0:
        optix = rtx
    else:
        optix = prepare_mesh(raster, rtx)
    return _viewshed_rt(raster, optix, x, y, observer_elev, target_elev)
