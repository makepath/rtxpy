"""
GPU-accelerated viewshed analysis using ray tracing.

This module provides functions for computing viewshed (visibility) analysis
on terrain data using NVIDIA OptiX ray tracing.

Functions
---------
viewshed
    Compute viewshed from a point on a raster terrain.
viewshed_with_context
    Compute viewshed using an existing RTX context.
"""

from numba import cuda
import numpy as np
import math
from typing import Union

import cupy
import xarray as xr

from .rtx import RTX
from ._cuda_utils import calc_dims, make_float3, add, mul, diff, dot, invert, float3
from . import mesh

# view options default values
OBS_ELEV = 0
TARGET_ELEV = 0

# if a cell is invisible, its value is set to -1
INVISIBLE = -1


@cuda.jit
def _generatePrimaryRays(data, x_coords, y_coords, H, W):
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

        data[i, j, 2] = 10000  # Location of the camera (height)
        data[i, j, 3] = 1e-3
        data[i, j, 4] = 0
        data[i, j, 5] = 0
        data[i, j, 6] = -1
        data[i, j, 7] = np.inf


def _generatePrimaryRaysWrapper(rays, x_coords, y_coords, H, W):
    griddim, blockdim = calc_dims((H, W))
    d_y_coords = cupy.array(y_coords)
    d_x_coords = cupy.array(x_coords)
    _generatePrimaryRays[griddim, blockdim](rays, d_x_coords, d_y_coords, H, W)
    return 0


@cuda.jit
def _generateViewshedRays(camRays, hits, vsrays, visibility_grid, H, W, vp):
    i, j = cuda.grid(2)
    if i >= 0 and i < H and j >= 0 and j < W:
        elevationOffset = vp[2]
        targetElevation = vp[3]
        dist = hits[i, j, 0]  # distance to surface from camera
        norm = make_float3(hits[i, j], 1)  # normal vector at intersection with surface
        if (norm[2] < 0):  # if back hit, face forward
            norm = invert(norm)
        cameraRay = camRays[i, j]
        rayOrigin = make_float3(cameraRay, 0)  # get the camera ray origin
        rayDir = make_float3(cameraRay, 4)  # get the camera ray direction
        hitP = add(rayOrigin, mul(rayDir, dist))  # calculate intersection point
        newOrigin = add(hitP, mul(norm, 1e-3))  # generate new ray origin with offset
        newOrigin = add(newOrigin, float3(0, 0, targetElevation))  # add target elevation

        w = int(vp[0])
        h = int(vp[1])
        viewshedRay = camRays[h, w]  # get camera ray for viewshed origin
        dist = hits[h, w, 0]  # get distance from camera to viewshed point
        rayOrigin = make_float3(viewshedRay, 0)
        rayDir = make_float3(viewshedRay, 4)
        hitP = add(rayOrigin, mul(rayDir, dist))
        viewshedPoint = add(hitP, float3(0, 0, elevationOffset))

        newDir = diff(viewshedPoint, newOrigin)  # vector from SurfaceHit to VP
        length = math.sqrt(dot(newDir, newDir))  # distance from surface to VP
        newDir = mul(newDir, 1/length)  # normalize direction

        cosine = dot(norm, newDir)  # cosine of angle between n and v
        theta = math.acos(cosine)  # angle in radians
        theta = (180*theta)/math.pi  # angle in degrees

        # prepare viewshed ray for visibility test
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


def _generateViewshedRaysWrapper(rays, hits, vsrays, visibility_grid, H, W, vp):
    griddim, blockdim = calc_dims((H, W))
    _generateViewshedRays[griddim, blockdim](rays, hits, vsrays, visibility_grid, H, W, vp)
    return 0


@cuda.jit
def _calcViewshed(hits, visibility_grid, H, W):
    i, j = cuda.grid(2)
    if i >= 0 and i < H and j >= 0 and j < W:
        dist = hits[i, j, 0]
        # If dist > 0, intersection occurred along ray length
        # meaning pixel is not directly visible from view point
        if dist >= 0:
            visibility_grid[i, j] = INVISIBLE


def _calcViewshedWrapper(hits, visibility_grid, H, W):
    griddim, blockdim = calc_dims((H, W))
    _calcViewshed[griddim, blockdim](hits, visibility_grid, H, W)
    return 0


def viewshed_with_context(raster: xr.DataArray,
                          rtx: RTX,
                          x: Union[int, float],
                          y: Union[int, float],
                          observer_elev: float = OBS_ELEV,
                          target_elev: float = TARGET_ELEV) -> xr.DataArray:
    """
    Compute viewshed analysis using an existing RTX context.

    This function performs viewshed (visibility) analysis from a specified
    observer point on the terrain. It uses an existing RTX context that
    already has the terrain geometry built, avoiding rebuild overhead.

    Parameters
    ----------
    raster : xr.DataArray
        Input terrain raster with x and y coordinates.
    rtx : RTX
        Pre-configured RTX instance with terrain geometry already built.
    x : int or float
        X coordinate of the observer position.
    y : int or float
        Y coordinate of the observer position.
    observer_elev : float, optional
        Height of the observer above the terrain surface. Default is 0.
    target_elev : float, optional
        Height of target points above the terrain surface. Default is 0.

    Returns
    -------
    xr.DataArray
        Viewshed result where visible cells contain the viewing angle in
        degrees, and invisible cells contain -1.

    Raises
    ------
    ValueError
        If x or y coordinates are outside the raster extent.
    """
    H, W = raster.shape

    y_coords = raster.indexes.get('y').values
    x_coords = raster.indexes.get('x').values

    # validate x arg
    if x < x_coords.min():
        raise ValueError("x argument outside of raster x_range")
    elif x > x_coords.max():
        raise ValueError("x argument outside of raster x_range")

    # validate y arg
    if y < y_coords.min():
        raise ValueError("y argument outside of raster y_range")
    elif y > y_coords.max():
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

    _generatePrimaryRaysWrapper(d_rays, x_coords, y_coords, H, W)
    device = cupy.cuda.Device(0)
    device.synchronize()
    rtx.trace(d_rays, d_hits, W*H)

    _generateViewshedRaysWrapper(d_rays, d_hits, d_vsrays, d_visgrid, H, W,
                                 (x_view, y_view, observer_elev, target_elev))
    device.synchronize()
    rtx.trace(d_vsrays, d_hits, W*H)

    _calcViewshedWrapper(d_hits, d_visgrid, H, W)

    if isinstance(raster.data, np.ndarray):
        visgrid = cupy.asnumpy(d_visgrid)
    else:
        visgrid = d_visgrid

    view = xr.DataArray(visgrid,
                        name="viewshed",
                        coords=raster.coords,
                        dims=raster.dims,
                        attrs=raster.attrs)
    return view


def viewshed(raster: xr.DataArray,
             x: Union[int, float],
             y: Union[int, float],
             observer_elev: float = OBS_ELEV,
             target_elev: float = TARGET_ELEV) -> xr.DataArray:
    """
    Compute viewshed analysis from a point on a raster terrain.

    This function performs viewshed (visibility) analysis from a specified
    observer point on the terrain. It automatically builds the necessary
    ray tracing acceleration structure if needed.

    Parameters
    ----------
    raster : xr.DataArray
        Input terrain raster as an xarray DataArray with x and y coordinates.
        The data must be a cupy array on GPU.
    x : int or float
        X coordinate of the observer position.
    y : int or float
        Y coordinate of the observer position.
    observer_elev : float, optional
        Height of the observer above the terrain surface. Default is 0.
    target_elev : float, optional
        Height of target points above the terrain surface. Default is 0.

    Returns
    -------
    xr.DataArray
        Viewshed result where visible cells contain the viewing angle in
        degrees, and invisible cells contain -1.

    Raises
    ------
    ValueError
        If raster.data is not a cupy array, or if the mesh generation
        or OptiX build fails.

    Examples
    --------
    >>> import rtxpy
    >>> result = rtxpy.viewshed.viewshed(terrain_raster, x=1000, y=2000, observer_elev=2)
    """
    if not isinstance(raster.data, cupy.ndarray):
        raise ValueError("raster.data must be a cupy array")

    H, W = raster.shape
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

    view = viewshed_with_context(raster, rtx, x, y, observer_elev, target_elev)
    return view


# Backwards compatibility aliases
viewshed_gpu = viewshed
viewshed_rt = viewshed_with_context
