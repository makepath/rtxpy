"""
GPU-accelerated hillshade rendering using ray tracing.

This module provides functions for computing hillshade (terrain shading)
on raster data using NVIDIA OptiX ray tracing for optional shadow casting.

Functions
---------
hillshade
    Compute hillshade with optional ray-traced shadows.
hillshade_with_context
    Compute hillshade using an existing RTX context.
"""

import numpy as np
from numba import cuda
from typing import Optional

import cupy
import xarray as xr
from scipy.spatial.transform import Rotation as R

from .rtx import RTX
from ._cuda_utils import calc_dims, make_float3, add, mul, dot, invert, float3
from . import mesh


@cuda.jit
def _generatePrimaryRays(data, x_coords, y_coords, H, W):
    """
    Generate orthographic camera rays looking straight down at the terrain.
    """
    i, j = cuda.grid(2)
    if i >= 0 and i < H and j >= 0 and j < W:
        if (j == W-1):
            data[i, j, 0] = j - 1e-3
        else:
            data[i, j, 0] = j + 1e-3

        if (i == H-1):
            data[i, j, 1] = i - 1e-3
        else:
            data[i, j, 1] = i + 1e-3

        data[i, j, 2] = 10000  # Camera height
        data[i, j, 3] = 1e-3
        data[i, j, 4] = 0
        data[i, j, 5] = 0
        data[i, j, 6] = -1
        data[i, j, 7] = np.inf


def _generatePrimaryRaysWrapper(rays, x_coords, y_coords, H, W):
    griddim, blockdim = calc_dims((H, W))
    _generatePrimaryRays[griddim, blockdim](rays, x_coords, y_coords, H, W)
    return 0


@cuda.jit
def _generateShadowRays(rays, hits, normals, H, W, sunDir):
    """
    Generate shadow rays from surface intersection points toward the sun.
    """
    i, j = cuda.grid(2)
    if i >= 0 and i < H and j >= 0 and j < W:
        dist = hits[i, j, 0]
        norm = make_float3(hits[i, j], 1)
        if (norm[2] < 0):
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


def _generateShadowRaysWrapper(rays, hits, normals, H, W, sunDir):
    griddim, blockdim = calc_dims((H, W))
    _generateShadowRays[griddim, blockdim](rays, hits, normals, H, W, sunDir)
    return 0


@cuda.jit
def _shadeLambert(hits, normals, output, H, W, sunDir, castShadows):
    """
    Apply Lambertian shading with optional shadow darkening.
    """
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


def _shadeLambertWrapper(hits, normals, output, H, W, sunDir, castShadows):
    griddim, blockdim = calc_dims((H, W))
    _shadeLambert[griddim, blockdim](hits, normals, output, H, W, sunDir, castShadows)
    return 0


def _getSunDir(angle_altitude, azimuth):
    """
    Calculate the vector towards the sun based on altitude angle and azimuth.
    """
    north = (0, 1, 0)
    rx = R.from_euler('x', angle_altitude, degrees=True)
    rz = R.from_euler('z', azimuth+180, degrees=True)
    sunDir = rx.apply(north)
    sunDir = rz.apply(sunDir)
    return sunDir


def hillshade_with_context(raster: xr.DataArray,
                           rtx: RTX,
                           shadows: bool = False,
                           azimuth: int = 225,
                           angle_altitude: int = 25,
                           name: Optional[str] = 'hillshade') -> xr.DataArray:
    """
    Compute hillshade using an existing RTX context.

    This function performs hillshade rendering with optional ray-traced shadows
    using an existing RTX context that already has terrain geometry built.

    Parameters
    ----------
    raster : xr.DataArray
        Input terrain raster with x and y coordinates.
    rtx : RTX
        Pre-configured RTX instance with terrain geometry already built.
    shadows : bool, optional
        Whether to compute ray-traced shadows. Default is False.
    azimuth : int, optional
        Sun azimuth angle in degrees (0=North, 90=East). Default is 225 (SW).
    angle_altitude : int, optional
        Sun altitude angle in degrees above horizon. Default is 25.
    name : str, optional
        Name for the output DataArray. Default is 'hillshade'.

    Returns
    -------
    xr.DataArray
        Hillshade result with values from 0 (dark) to 1 (bright).
    """
    H, W = raster.shape
    sunDir = cupy.array(_getSunDir(angle_altitude, azimuth))

    # Device buffers
    d_rays = cupy.empty((H, W, 8), np.float32)
    d_hits = cupy.empty((H, W, 4), np.float32)
    d_aux = cupy.empty((H, W, 3), np.float32)
    d_output = cupy.empty((H, W), np.float32)

    y_coords = cupy.array(raster.indexes.get('y').values)
    x_coords = cupy.array(raster.indexes.get('x').values)

    _generatePrimaryRaysWrapper(d_rays, x_coords, y_coords, H, W)
    device = cupy.cuda.Device(0)
    device.synchronize()
    rtx.trace(d_rays, d_hits, W*H)

    _generateShadowRaysWrapper(d_rays, d_hits, d_aux, H, W, sunDir)
    if shadows:
        device.synchronize()
        rtx.trace(d_rays, d_hits, W*H)

    _shadeLambertWrapper(d_hits, d_aux, d_output, H, W, sunDir, shadows)

    if isinstance(raster.data, np.ndarray):
        output = cupy.asnumpy(d_output[:, :])
        nanValue = np.nan
    else:
        output = d_output[:, :]
        nanValue = cupy.nan

    output[0, :] = nanValue
    output[-1, :] = nanValue
    output[:, 0] = nanValue
    output[:, -1] = nanValue

    hill = xr.DataArray(output,
                        name=name,
                        coords=raster.coords,
                        dims=raster.dims,
                        attrs=raster.attrs)
    return hill


def hillshade(raster: xr.DataArray,
              shadows: bool = False,
              azimuth: int = 225,
              angle_altitude: int = 25,
              name: Optional[str] = 'hillshade') -> xr.DataArray:
    """
    Compute hillshade with optional ray-traced shadows.

    This function performs hillshade (terrain shading) rendering on a raster
    DEM. It automatically builds the necessary ray tracing acceleration
    structure if needed.

    Parameters
    ----------
    raster : xr.DataArray
        Input terrain raster as an xarray DataArray with x and y coordinates.
        Should be a cupy array for best performance.
    shadows : bool, optional
        Whether to compute ray-traced shadows. When True, areas occluded from
        the sun are darkened. Default is False.
    azimuth : int, optional
        Sun azimuth angle in degrees (0=North, 90=East, 180=South, 270=West).
        Default is 225 (Southwest).
    angle_altitude : int, optional
        Sun altitude angle in degrees above the horizon. Default is 25.
    name : str, optional
        Name for the output DataArray. Default is 'hillshade'.

    Returns
    -------
    xr.DataArray
        Hillshade result with values from 0 (dark) to 1 (bright).
        Edge pixels are set to NaN.

    Raises
    ------
    ValueError
        If mesh generation or OptiX build fails.

    Examples
    --------
    >>> import rtxpy
    >>> result = rtxpy.hillshade.hillshade(terrain_raster, shadows=True, azimuth=315)
    """
    if not isinstance(raster.data, cupy.ndarray):
        print("WARNING: raster.data is not a cupy array. Additional overhead will be incurred")
    H, W = raster.data.squeeze().shape
    rtx = RTX()

    datahash = np.uint64(hash(str(raster.data.get())) % (1 << 64))
    optixhash = np.uint64(rtx.getHash())
    if (optixhash != datahash):
        numTris = (H - 1) * (W - 1) * 2
        verts = cupy.empty(H * W * 3, np.float32)
        triangles = cupy.empty(numTris * 3, np.int32)

        # Generate mesh from terrain
        res = mesh.triangulate_terrain(verts, triangles, raster)
        if res:
            raise ValueError("Failed to generate mesh from terrain. Error code:{}".format(res))
        res = rtx.build(datahash, verts, triangles)
        if res:
            raise ValueError("OptiX failed to build GAS with error code:{}".format(res))
        # Clear GPU memory no longer needed
        verts = None
        triangles = None
        cupy.get_default_memory_pool().free_all_blocks()

    hill = hillshade_with_context(raster, rtx, azimuth=azimuth,
                                  angle_altitude=angle_altitude,
                                  shadows=shadows, name=name)
    return hill


# Backwards compatibility aliases
hillshade_gpu = hillshade
hillshade_rt = hillshade_with_context
