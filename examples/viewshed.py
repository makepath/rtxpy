from numba import cuda
import numpy as np
import numba as nb

import cupy
import xarray as xr
from rtxpy import RTX
import math

from typing import Union

from scipy.spatial.transform import Rotation as R

from cuda_utils import *
import mesh_utils

# view options default values
OBS_ELEV = 0
TARGET_ELEV = 0

# if a cell is invisible, its value is set to -1
INVISIBLE = -1

@cuda.jit
def _generatePrimaryRays(data, x_coords, y_coords, H, W):
    i, j = cuda.grid(2)
    if i>=0 and i < H and j>=0 and j < W:
#        if (j == W-1):
#            data[i,j,0] = y_coords[j] - 1e-6
#        else:
#            data[i,j,0] = y_coords[j] + 1e-6
#
#        if (i == H-1):
#            data[i,j,1] = x_coords[i] - 1e-6
#        else:
#            data[i,j,1] = x_coords[i] + 1e-6

        if (j == W-1):
            data[i,j,0] = j - 1e-3
        else:
            data[i,j,0] = j + 1e-3

        if (i == H-1):
            data[i,j,1] = i - 1e-3
        else:
            data[i,j,1] = i + 1e-3

        data[i,j,2] = 10000 # Location of the camera (height)
        data[i,j,3] = 1e-3
        data[i,j,4] = 0
        data[i,j,5] = 0
        data[i,j,6] = -1
        data[i,j,7] = np.inf

def generatePrimaryRays(rays, x_coords, y_coords, H, W):
    griddim, blockdim = calc_dims((H, W))
    d_y_coords = cupy.array(y_coords)
    d_x_coords = cupy.array(x_coords)
    _generatePrimaryRays[griddim, blockdim](rays, d_x_coords, d_y_coords, H, W)
    return 0

@cuda.jit
def _generateViewshedRays(camRays, hits, vsrays, visibility_grid, H, W, vp):
    i, j = cuda.grid(2)
    if i>=0 and i < H and j>=0 and j < W:
        elevationOffset = vp[2]
        targetElevation = vp[3]
        dist = hits[i,j,0]  # distance to surface from camera
        norm = make_float3(hits[i,j], 1)  # normal vector at intersection with surface
        #inorm = norm
        if (norm[2] < 0):  # if back hit, face forward
            norm = invert(norm)
        cameraRay = camRays[i,j]
        rayOrigin = make_float3(cameraRay, 0)  # get the camera ray origin
        rayDir    = make_float3(cameraRay, 4)  # get the camera ray direction
        hitP = add(rayOrigin, mul(rayDir,dist))  # calculate intersection point
        newOrigin = add(hitP, mul(norm, 1e-3))  # generate new ray origin, and a little offset to avoid self-intersection
        newOrigin = add(newOrigin, float3(0, 0, targetElevation))  # move the new origin up by the selected by user targetElevation factor

        w = int(vp[0])
        h = int(vp[1])
        viewshedRay = camRays[h,w]  # get the camera ray that was cast for the location of the viewshed origin
        dist = hits[h,w,0]  # get the distance from the camera to the viewshed point
        rayOrigin = make_float3(viewshedRay, 0)  # get the origin on the camera of the ray towards VP point
        rayDir    = make_float3(viewshedRay, 4)  # get the direction from camera to VP point
        hitP = add(rayOrigin, mul(rayDir,dist))  # calculate distance from camera to VP
        viewshedPoint = add(hitP, float3(0, 0, elevationOffset))  # calculate the VP location on the surface and add the VP offset

        newDir = diff(viewshedPoint, newOrigin)  # calculate vector from SurfaceHit to VP
        length = math.sqrt(dot(newDir, newDir))  # calculate distance from surface to VP
        newDir = mul(newDir, 1/length)  # normalize the direction (vector v)

        cosine = dot(norm, newDir)  # cosine of the angle between n and v
        theta = math.acos(cosine)  # Cosine angle in radians
        theta = (180*theta)/math.pi  # Cosine angle in degrees

        # prepare a viewshed ray to cast to determine visibility
        vsray = vsrays[i,j]
        vsray[0] = newOrigin[0]
        vsray[1] = newOrigin[1]
        vsray[2] = newOrigin[2]
        vsray[3] = 0
        vsray[4] = newDir[0]
        vsray[5] = newDir[1]
        vsray[6] = newDir[2]
        vsray[7] = length

        visibility_grid[i,j] = theta

def generateViewshedRays(rays, hits, vsrays, visibility_grid, H, W, vp):
    griddim, blockdim = calc_dims((H, W))
    _generateViewshedRays[griddim, blockdim](rays, hits, vsrays, visibility_grid, H, W, vp)
    return 0

@cuda.jit
def _calcViewshed(hits, visibility_grid, H, W):
    i, j = cuda.grid(2)
    if i>=0 and i < H and j>=0 and j < W:
        dist = hits[i,j,0]
        # We traced the viewshed rays and now hits contains the intersection data
        # if dist > 0, then we were able to hit something along the length of the ray
        # which means that the pixel we targeted is not directly visible from the view point
        if dist>=0:
            visibility_grid[i,j] = INVISIBLE

def calcViewshed(hits, visibility_grid, H, W):
    griddim, blockdim = calc_dims((H, W))
    _calcViewshed[griddim, blockdim](hits, visibility_grid, H, W)
    return 0

def viewshed_rt(raster: xr.DataArray,
             optix: RTX,
             x: Union[int, float],
             y: Union[int, float],
             observer_elev: float = OBS_ELEV,
             target_elev: float = TARGET_ELEV) -> xr.DataArray:

    H,W = raster.shape

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
    d_rays = cupy.empty((H,W,8), np.float32)
    d_hits = cupy.empty((H,W,4), np.float32)
    d_visgrid = cupy.empty((H,W), np.float32)
    d_vsrays  = cupy.empty((H,W,8), np.float32)

    generatePrimaryRays(d_rays, x_coords, y_coords, H, W)
    device = cupy.cuda.Device(0)
    device.synchronize()
    res = optix.trace(d_rays, d_hits, W*H)

    generateViewshedRays(d_rays, d_hits, d_vsrays, d_visgrid, H, W, (x_view, y_view, observer_elev, target_elev))
    device.synchronize()
    res = optix.trace(d_vsrays, d_hits, W*H)

    calcViewshed(d_hits, d_visgrid, H, W)

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

def viewshed_gpu(raster: xr.DataArray,
             x: Union[int, float],
             y: Union[int, float],
             observer_elev: float = OBS_ELEV,
             target_elev: float = TARGET_ELEV) -> xr.DataArray:
    if not isinstance(raster.data, cupy.ndarray):
        raise ValueError("raster.data must be a cupy array")

    H,W = raster.shape
    optix = RTX()

    datahash = np.uint64(hash(str(raster.data.get())) % (1 << 64))
    optixhash = np.uint64(optix.getHash())
    if (optixhash != datahash):
        numTris = (H - 1) * (W - 1) * 2
        verts = cupy.empty(H * W * 3, np.float32)
        triangles = cupy.empty(numTris * 3, np.int32)

        # Generate a mesh from the terrain (buffers are on the GPU, so generation happens also on GPU)
        res = mesh_utils.triangulateTerrain(verts, triangles, raster)
        if res:
            raise ValueError("Failed to generate mesh from terrain. Error code:{}".format(res))
        res = optix.build(datahash, verts, triangles)
        if res:
            raise ValueError("OptiX failed to build GAS with error code:{}".format(res))
        #Clear some GPU memory that we no longer need
        verts = None
        triangles = None
        cupy.get_default_memory_pool().free_all_blocks()

    view = viewshed_rt(raster, optix, x, y, observer_elev, target_elev)
    return view
