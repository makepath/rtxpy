"""Slope and aspect analysis using GPU-accelerated ray tracing.

Slope measures terrain steepness (0° flat, 90° vertical).
Aspect measures downhill compass bearing (0° north, clockwise, -1 flat).
Both derive from surface normals returned by a single primary ray trace.
"""

from numba import cuda
import math
import numpy as np

from .._cuda_utils import calc_dims
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
            "xarray is required for slope/aspect analysis. "
            "Install it with: pip install xarray "
            "or: pip install rtxpy[analysis]"
        )


@cuda.jit
def _calc_slope_kernel(hits, output, H, W):
    """GPU kernel to compute slope in degrees from surface normals.

    slope = acos(nz) — angle between normal and vertical.
    0° = flat, 90° = vertical cliff.
    """
    i, j = cuda.grid(2)
    if i >= 0 and i < H and j >= 0 and j < W:
        nz = hits[i, j, 3]
        if hits[i, j, 0] <= 0:
            # Ray miss
            output[i, j] = np.float32(np.nan)
        else:
            # Ensure nz is in [-1, 1] for acos
            if nz > 1.0:
                nz = np.float32(1.0)
            elif nz < -1.0:
                nz = np.float32(-1.0)
            # Normals may point downward; use absolute value
            if nz < 0:
                nz = -nz
            output[i, j] = np.float32(math.acos(nz) * (180.0 / math.pi))


@cuda.jit
def _calc_aspect_kernel(hits, output, H, W):
    """GPU kernel to compute aspect (compass bearing) from surface normals.

    Convention (ESRI/QGIS standard):
      0° = North, 90° = East, 180° = South, 270° = West
      -1 = flat (horizontal normal magnitude below threshold)

    Formula: atan2(nx, -ny) normalised to [0, 360).
    In mesh coords X = col (east), Y = row (south for north-up rasters).
    """
    i, j = cuda.grid(2)
    if i >= 0 and i < H and j >= 0 and j < W:
        if hits[i, j, 0] <= 0:
            # Ray miss
            output[i, j] = np.float32(np.nan)
        else:
            nx = hits[i, j, 1]
            ny = hits[i, j, 2]
            nz = hits[i, j, 3]

            # Ensure normal points upward
            if nz < 0:
                nx = -nx
                ny = -ny

            horiz_mag = math.sqrt(nx * nx + ny * ny)
            if horiz_mag < 1e-6:
                # Flat — no meaningful aspect
                output[i, j] = np.float32(-1.0)
            else:
                angle = math.atan2(nx, -ny) * (180.0 / math.pi)
                if angle < 0:
                    angle += 360.0
                output[i, j] = np.float32(angle)


def _slope_rt(raster, optix):
    """Internal: trace primary rays and compute slope."""
    xr = _lazy_import_xarray()

    H, W = raster.shape

    d_rays = cupy.empty((H, W, 8), np.float32)
    d_hits = cupy.empty((H, W, 4), np.float32)
    d_output = cupy.empty((H, W), np.float32)

    y_coords = cupy.array(raster.indexes.get('y').values)
    x_coords = cupy.array(raster.indexes.get('x').values)

    generate_primary_rays(d_rays, x_coords, y_coords, H, W)
    cupy.cuda.Device(0).synchronize()
    optix.trace(d_rays, d_hits, W * H)

    griddim, blockdim = calc_dims((H, W))
    _calc_slope_kernel[griddim, blockdim](d_hits, d_output, H, W)

    if isinstance(raster.data, np.ndarray):
        output = cupy.asnumpy(d_output)
        nanValue = np.nan
    else:
        output = d_output[:, :]
        nanValue = cupy.nan

    # Set edge pixels to NaN (mesh boundary artefacts)
    output[0, :] = nanValue
    output[-1, :] = nanValue
    output[:, 0] = nanValue
    output[:, -1] = nanValue

    return xr.DataArray(
        output,
        name='slope',
        coords=raster.coords,
        dims=raster.dims,
        attrs=raster.attrs,
    )


def _aspect_rt(raster, optix):
    """Internal: trace primary rays and compute aspect."""
    xr = _lazy_import_xarray()

    H, W = raster.shape

    d_rays = cupy.empty((H, W, 8), np.float32)
    d_hits = cupy.empty((H, W, 4), np.float32)
    d_output = cupy.empty((H, W), np.float32)

    y_coords = cupy.array(raster.indexes.get('y').values)
    x_coords = cupy.array(raster.indexes.get('x').values)

    generate_primary_rays(d_rays, x_coords, y_coords, H, W)
    cupy.cuda.Device(0).synchronize()
    optix.trace(d_rays, d_hits, W * H)

    griddim, blockdim = calc_dims((H, W))
    _calc_aspect_kernel[griddim, blockdim](d_hits, d_output, H, W)

    if isinstance(raster.data, np.ndarray):
        output = cupy.asnumpy(d_output)
        nanValue = np.nan
    else:
        output = d_output[:, :]
        nanValue = cupy.nan

    # Set edge pixels to NaN (mesh boundary artefacts)
    output[0, :] = nanValue
    output[-1, :] = nanValue
    output[:, 0] = nanValue
    output[:, -1] = nanValue

    return xr.DataArray(
        output,
        name='aspect',
        coords=raster.coords,
        dims=raster.dims,
        attrs=raster.attrs,
    )


def slope(raster, rtx: RTX = None):
    """Compute terrain slope in degrees from surface normals.

    Parameters
    ----------
    raster : xarray.DataArray
        2D raster terrain data with 'x' and 'y' coordinates.
    rtx : RTX, optional
        Existing RTX instance to reuse. If None, a new instance is created.

    Returns
    -------
    xarray.DataArray
        Slope in degrees (0° flat, 90° vertical). Edge pixels are NaN.
    """
    if not has_cupy:
        raise ImportError(
            "cupy is required for slope analysis. "
            "Install it with: conda install -c conda-forge cupy"
        )

    if not isinstance(raster.data, cupy.ndarray):
        import warnings
        warnings.warn(
            "raster.data is not a cupy array. "
            "Additional overhead will be incurred from CPU-GPU transfers."
        )

    optix = prepare_mesh(raster, rtx)
    return _slope_rt(raster, optix)


def aspect(raster, rtx: RTX = None):
    """Compute terrain aspect (compass bearing of downhill direction).

    Convention (ESRI/QGIS standard):
      0° = North, 90° = East, 180° = South, 270° = West.
      -1 = flat terrain (no meaningful downhill direction).

    Parameters
    ----------
    raster : xarray.DataArray
        2D raster terrain data with 'x' and 'y' coordinates.
    rtx : RTX, optional
        Existing RTX instance to reuse. If None, a new instance is created.

    Returns
    -------
    xarray.DataArray
        Aspect in degrees [0, 360) or -1 for flat. Edge pixels are NaN.
    """
    if not has_cupy:
        raise ImportError(
            "cupy is required for aspect analysis. "
            "Install it with: conda install -c conda-forge cupy"
        )

    if not isinstance(raster.data, cupy.ndarray):
        import warnings
        warnings.warn(
            "raster.data is not a cupy array. "
            "Additional overhead will be incurred from CPU-GPU transfers."
        )

    optix = prepare_mesh(raster, rtx)
    return _aspect_rt(raster, optix)
