"""Hillshade analysis using GPU-accelerated ray tracing.

This module computes terrain shading based on sun position, with optional
shadow casting using ray tracing.
"""

from numba import cuda
import numpy as np

from typing import Optional

from .._cuda_utils import calc_dims, add, mul, dot, float3, make_float3, invert
from ._common import generate_primary_rays, prepare_mesh
from ..rtx import RTX, has_cupy

if has_cupy:
    import cupy


def _lazy_import_xarray():
    """Lazily import xarray with helpful error message."""
    try:
        import xarray as xr
        return xr
    except ImportError:
        raise ImportError(
            "xarray is required for hillshade analysis. "
            "Install it with: pip install xarray "
            "or: pip install rtxpy[analysis]"
        )


def _lazy_import_scipy_rotation():
    """Lazily import scipy.spatial.transform.Rotation with helpful error message."""
    try:
        from scipy.spatial.transform import Rotation as R
        return R
    except ImportError:
        raise ImportError(
            "scipy is required for hillshade analysis (for sun direction calculation). "
            "Install it with: pip install scipy "
            "or: pip install rtxpy[all]"
        )


def get_sun_dir(angle_altitude, azimuth):
    """Calculate the sun direction vector based on altitude angle and azimuth.

    Parameters
    ----------
    angle_altitude : float
        Sun altitude angle in degrees (0 = horizon, 90 = directly overhead).
    azimuth : float
        Sun azimuth angle in degrees, measured clockwise from north.
        Common values: 0 = North, 90 = East, 180 = South, 270 = West.

    Returns
    -------
    numpy.ndarray
        Unit vector pointing toward the sun (x, y, z).

    Examples
    --------
    >>> sun = get_sun_dir(angle_altitude=45, azimuth=180)  # Sun from the south
    >>> print(sun)  # Approximately [0, -0.707, 0.707]
    """
    R = _lazy_import_scipy_rotation()

    north = (0, 1, 0)
    rx = R.from_euler('x', angle_altitude, degrees=True)
    rz = R.from_euler('z', azimuth + 180, degrees=True)
    sunDir = rx.apply(north)
    sunDir = rz.apply(sunDir)
    return sunDir


@cuda.jit
def _generate_shadow_rays_kernel(rays, hits, normals, H, W, sunDir):
    """GPU kernel to generate shadow rays from surface points toward the sun."""
    i, j = cuda.grid(2)
    if i >= 0 and i < H and j >= 0 and j < W:
        dist = hits[i, j, 0]
        norm = make_float3(hits[i, j], 1)
        if norm[2] < 0:
            norm = invert(norm)

        ray = rays[i, j]
        rayOrigin = make_float3(ray, 0)
        rayDir = make_float3(ray, 4)
        p = add(rayOrigin, mul(rayDir, dist))

        newOrigin = add(p, mul(norm, 1e-3))
        ray[0] = newOrigin[0]
        ray[1] = newOrigin[1]
        ray[2] = newOrigin[2]
        ray[3] = 1e-3
        ray[4] = sunDir[0]
        ray[5] = sunDir[1]
        ray[6] = sunDir[2]
        ray[7] = np.inf if dist > 0 else 0

        normals[i, j, 0] = norm[0]
        normals[i, j, 1] = norm[1]
        normals[i, j, 2] = norm[2]


def _generate_shadow_rays(rays, hits, normals, H, W, sunDir):
    """Generate shadow rays from each surface point toward the sun."""
    griddim, blockdim = calc_dims((H, W))
    _generate_shadow_rays_kernel[griddim, blockdim](rays, hits, normals, H, W, sunDir)
    return 0


@cuda.jit
def _shade_lambert_kernel(hits, normals, output, H, W, sunDir, castShadows):
    """GPU kernel for Lambertian shading with optional shadows."""
    i, j = cuda.grid(2)
    if i >= 0 and i < H and j >= 0 and j < W:
        norm = make_float3(normals[i, j], 0)
        light_dir = make_float3(sunDir, 0)
        cos_theta = dot(light_dir, norm)

        temp = (cos_theta + 1) / 2

        if castShadows and hits[i, j, 0] >= 0:
            temp = temp / 2

        if temp > 1:
            temp = 1
        elif temp < 0:
            temp = 0

        output[i, j] = temp


def _shade_lambert(hits, normals, output, H, W, sunDir, castShadows):
    """Apply Lambertian shading based on surface normals and sun direction."""
    griddim, blockdim = calc_dims((H, W))
    _shade_lambert_kernel[griddim, blockdim](hits, normals, output, H, W, sunDir, castShadows)
    return 0


def _hillshade_rt(raster, optix, shadows, azimuth, angle_altitude, name):
    """Internal function to perform hillshade ray tracing."""
    xr = _lazy_import_xarray()

    H, W = raster.shape
    sunDir = cupy.array(get_sun_dir(angle_altitude, azimuth))

    # Device buffers
    d_rays = cupy.empty((H, W, 8), np.float32)
    d_hits = cupy.empty((H, W, 4), np.float32)
    d_aux = cupy.empty((H, W, 3), np.float32)
    d_output = cupy.empty((H, W), np.float32)

    y_coords = cupy.array(raster.indexes.get('y').values)
    x_coords = cupy.array(raster.indexes.get('x').values)

    generate_primary_rays(d_rays, x_coords, y_coords, H, W)
    device = cupy.cuda.Device(0)
    device.synchronize()
    optix.trace(d_rays, d_hits, W * H)

    _generate_shadow_rays(d_rays, d_hits, d_aux, H, W, sunDir)
    if shadows:
        device.synchronize()
        optix.trace(d_rays, d_hits, W * H)

    _shade_lambert(d_hits, d_aux, d_output, H, W, sunDir, shadows)

    if isinstance(raster.data, np.ndarray):
        output = cupy.asnumpy(d_output[:, :])
        nanValue = np.nan
    else:
        output = d_output[:, :]
        nanValue = cupy.nan

    # Set edge pixels to NaN
    output[0, :] = nanValue
    output[-1, :] = nanValue
    output[:, 0] = nanValue
    output[:, -1] = nanValue

    hill = xr.DataArray(
        output,
        name=name,
        coords=raster.coords,
        dims=raster.dims,
        attrs=raster.attrs
    )
    return hill


def hillshade(raster,
              shadows: bool = False,
              azimuth: int = 225,
              angle_altitude: int = 25,
              rtx: RTX = None,
              name: Optional[str] = 'hillshade'):
    """Compute hillshade illumination for terrain visualization.

    Hillshade creates a shaded relief effect by simulating how the terrain
    would look when illuminated by the sun from a given direction.

    Parameters
    ----------
    raster : xarray.DataArray
        2D raster terrain data with 'x' and 'y' coordinates.
        Data should be a cupy array on the GPU for best performance.
    shadows : bool, optional
        If True, cast shadow rays to determine which areas are in shadow.
        Shadows are rendered at half brightness. Default is False.
    azimuth : int, optional
        Sun azimuth angle in degrees, measured clockwise from north.
        Default is 225 (southwest).
    angle_altitude : int, optional
        Sun altitude angle in degrees above the horizon.
        Default is 25.
    rtx : RTX, optional
        Existing RTX instance to reuse. If None, a new instance is created
        and the terrain mesh is built automatically.
    name : str, optional
        Name attribute for the output DataArray. Default is 'hillshade'.

    Returns
    -------
    xarray.DataArray
        Hillshade raster with values from 0 (dark) to 1 (bright).
        Edge pixels are set to NaN.

    Notes
    -----
    The hillshade is computed using Lambertian shading, where brightness
    depends on the cosine of the angle between the surface normal and
    the sun direction.

    If raster.data is a numpy array, a warning will be printed about
    performance overhead from CPU-GPU transfers.

    Examples
    --------
    >>> import xarray as xr
    >>> import cupy
    >>> # Load terrain data
    >>> terrain = xr.open_dataarray('dem.tif')
    >>> terrain = terrain.assign({'data': cupy.array(terrain.data)})
    >>> # Compute hillshade with morning sun from the east
    >>> shade = rtxpy.hillshade(terrain, azimuth=90, angle_altitude=30)
    >>> # Compute hillshade with shadows
    >>> shade_shadow = rtxpy.hillshade(terrain, shadows=True)
    """
    xr = _lazy_import_xarray()

    if not has_cupy:
        raise ImportError(
            "cupy is required for hillshade analysis. "
            "Install it with: conda install -c conda-forge cupy"
        )

    if not isinstance(raster.data, cupy.ndarray):
        import warnings
        warnings.warn(
            "raster.data is not a cupy array. "
            "Additional overhead will be incurred from CPU-GPU transfers."
        )

    optix = prepare_mesh(raster, rtx)
    return _hillshade_rt(
        raster, optix, shadows=shadows, azimuth=azimuth,
        angle_altitude=angle_altitude, name=name
    )
