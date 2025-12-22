import numpy as np
import numba as nb

import cupy
import xarray as xr
from rtxpy import RTX

from typing import Optional

from scipy.spatial.transform import Rotation as R

from raytrace.cuda_utils import *
from raytrace import mesh_utils

@nb.cuda.jit
def _generatePrimaryRays(data, x_coords, y_coords, H, W):
    """
    A GPU kernel that given a set of x and y discrete coordinates on a raster terrain
    generates in @data a list of parallel rays that represent camera rays generated from an ortographic camera
    that is looking straight down at the surface from an origin height 10000
    """
    i, j = nb.cuda.grid(2)
    if i>=0 and i < H and j>=0 and j < W:
        #data[i,j,0] = j + 1e-6 # x_coords[j] + 1e-6
        #data[i,j,1] = i + 1e-6 # y_coords[i] + 1e-6

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
    _generatePrimaryRays[griddim, blockdim](rays, x_coords, y_coords, H, W)
    return 0


@nb.cuda.jit
def _generateShadowRays(rays, hits, normals, H, W, sunDir):
    """
    A GPU kernel that given a set rays and their respective intersection points,
    generates in rays (overwriting the original content) a new set of rays (shadow rays)
    That have their origins at the point of intersection of their parent ray and direction - the direction towards the sun
    The normals vectors at the point of intersection of the original rays are cached in @normals
    Thus we can later use them to do lambertian shading, after the shadow rays have been traced
    """
    i, j = nb.cuda.grid(2)
    if i>=0 and i < H and j>=0 and j < W:
        dist = hits[i,j,0]
        norm = make_float3(hits[i,j], 1)
        if (norm[2] < 0):
            norm = invert(norm)
        ray = rays[i,j]
        rayOrigin = make_float3(ray, 0)
        rayDir    = make_float3(ray, 4)
        p = add(rayOrigin, mul(rayDir,dist))

        newOrigin = add(p, mul(norm, 1e-3))
        ray[0] = newOrigin[0]
        ray[1] = newOrigin[1]
        ray[2] = newOrigin[2]
        ray[3] = 1e-3
        ray[4] = sunDir[0]
        ray[5] = sunDir[1]
        ray[6] = sunDir[2]
        ray[7] = np.inf if dist > 0 else 0

        normals[i,j,0] = norm[0]
        normals[i,j,1] = norm[1]
        normals[i,j,2] = norm[2]

def generateShadowRays(rays, hits, normals, H, W, sunDir):
    griddim, blockdim = calc_dims((H, W))
    _generateShadowRays[griddim, blockdim](rays, hits, normals, H, W, sunDir)
    return 0


@nb.cuda.jit
def _shadeLambert(hits, normals, output, H, W, sunDir, castShadows):
    """
    This kernel does a simple Lambertian shading
    The hits array contains the results of tracing the shadow rays through the scene.
    If the value in hits[x,y,0] is > 0, then a valid intersection occurred and that means that the point
    at location x,y is in shadow.
    The normals array stores the normal at the intersecion point of each camera ray
    We then use the information for light visibility and normal to apply Lambert's cosine law
    The final result is stored in output which is an RGB array
    """
    i, j = nb.cuda.grid(2)
    if i>=0 and i < H and j>=0 and j < W:
        # Normal at the intersection of camera ray (i,j) with the scene
        norm = make_float3(normals[i,j], 0)

        # Below is same as existing algorithm without shadows and is OK with shadows.
        # Could be improved with a bit of antialiasing at edges of shadow???

        light_dir = make_float3(sunDir, 0)  # Might have to make it zero if back cull.
        cos_theta = dot(light_dir, norm)  # light_dir and norm are already normalised.

        temp = (cos_theta + 1) / 2

        if castShadows and hits[i, j, 0] >= 0:
            temp = temp / 2

        if temp > 1:
            temp = 1
        elif temp < 0:
            temp = 0

        output[i, j] = temp



def shadeLambert(hits, normals, output, H, W, sunDir, castShadows):
    griddim, blockdim = calc_dims((H, W))
    _shadeLambert[griddim, blockdim](hits, normals, output, H, W, sunDir, castShadows)
    return 0


def getSunDir(angle_altitude, azimuth):
    """
    Calculate the vector towards the sun based on sun altitude angle and azimuth
    """
    north = (0,1,0)
    rx = R.from_euler('x', angle_altitude, degrees=True)
    rz = R.from_euler('z', azimuth+180, degrees=True)
    sunDir = rx.apply(north)
    sunDir = rz.apply(sunDir)
    return sunDir


def hillshade_rt(raster: xr.DataArray,
              optix: RTX,
              shadows: bool = False,
              azimuth: int = 225,
              angle_altitude: int = 25,
              name: Optional[str] = 'hillshade') -> xr.DataArray:

    H,W = raster.shape
    sunDir = cupy.array(getSunDir(angle_altitude, azimuth))

    #output = np.zeros((H,W,3), np.float32)

    # Device buffers
    d_rays   = cupy.empty((H,W,8), np.float32)
    d_hits   = cupy.empty((H,W,4), np.float32)
    d_aux    = cupy.empty((H,W,3), np.float32)
    d_output = cupy.empty((H,W), np.float32)

    y_coords = cupy.array(raster.indexes.get('y').values)
    x_coords = cupy.array(raster.indexes.get('x').values)

    generatePrimaryRays(d_rays, x_coords, y_coords, H, W)
    device = cupy.cuda.Device(0)
    device.synchronize()
    res = optix.trace(d_rays, d_hits, W*H)

    generateShadowRays(d_rays, d_hits, d_aux, H, W, sunDir)
    if shadows:
        device.synchronize()
        res = optix.trace(d_rays, d_hits, W*H)

    shadeLambert(d_hits, d_aux, d_output, H, W, sunDir, shadows)

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

def hillshade_gpu(raster: xr.DataArray,
              shadows: bool = False,
              azimuth: int = 225,
              angle_altitude: int = 25,
              name: Optional[str] = 'hillshade') -> xr.DataArray:
    # Move the terrain to GPU for testing the GPU path
    if not isinstance(raster.data, cupy.ndarray):
        print("WARNING: raster.data is not a cupy array. Additional overhead will be incurred")
    H,W = raster.shape
    optix = RTX()

    datahash = np.uint64(hash(str(raster.data.get())))
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

    hill = hillshade_rt(raster, optix, azimuth=azimuth, angle_altitude=angle_altitude, shadows=shadows, name=name)
    return hill
