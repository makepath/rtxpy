"""
Mesh utilities for rtxpy terrain processing.

This module provides functions for triangulating terrain data and exporting
meshes to STL format.

Functions
---------
triangulate_terrain
    Convert a 2D terrain array into a triangulated mesh.
write_stl
    Export a triangulated mesh to STL file format.
"""

import numba as nb
from numba import cuda
import numpy as np
import cupy


@cuda.jit
def _triangulateTerrain(verts, triangles, data, H, W, scale, stride):
    globalId = stride + cuda.grid(1)
    if globalId < W*H:
        h = globalId // W
        w = globalId % W
        meshMapIndex = h * W + w

        val = data[h, w]

        offset = 3*meshMapIndex
        verts[offset] = w  # x_coords[w] # w
        verts[offset+1] = h  # y_coords[h] # h
        verts[offset+2] = val * scale

        if w != W - 1 and h != H - 1:
            offset = 6*(h * (W-1) + w)
            triangles[offset+0] = np.int32(meshMapIndex + W)
            triangles[offset+1] = np.int32(meshMapIndex + W + 1)
            triangles[offset+2] = np.int32(meshMapIndex)
            triangles[offset+3] = np.int32(meshMapIndex + W + 1)
            triangles[offset+4] = np.int32(meshMapIndex + 1)
            triangles[offset+5] = np.int32(meshMapIndex)


@nb.njit(parallel=True)
def _triangulateCPU(verts, triangles, data, H, W, scale):
    for h in nb.prange(H):
        for w in range(W):
            meshMapIndex = h * W + w

            val = data[h, w]

            offset = 3*meshMapIndex
            verts[offset] = w  # x_coords[w] # w
            verts[offset+1] = h  # y_coords[h] # h
            verts[offset+2] = val * scale

            if w != W - 1 and h != H - 1:
                offset = 6*(h * (W-1) + w)
                triangles[offset+0] = np.int32(meshMapIndex + W)
                triangles[offset+1] = np.int32(meshMapIndex + W + 1)
                triangles[offset+2] = np.int32(meshMapIndex)
                triangles[offset+3] = np.int32(meshMapIndex + W+1)
                triangles[offset+4] = np.int32(meshMapIndex + 1)
                triangles[offset+5] = np.int32(meshMapIndex)


def triangulate_terrain(verts, triangles, terrain, scale=1):
    """
    Convert a 2D terrain array into a triangulated mesh.

    This function populates pre-allocated vertex and triangle buffers with
    mesh data generated from the terrain heightmap. The mesh is created by
    generating two triangles for each cell in the terrain grid.

    Parameters
    ----------
    verts : array-like
        Pre-allocated buffer for vertex data. Should have shape (H * W * 3,)
        where H and W are the terrain dimensions. Each vertex consists of
        3 float32 values (x, y, z).
    triangles : array-like
        Pre-allocated buffer for triangle indices. Should have shape
        ((H-1) * (W-1) * 2 * 3,) for the index buffer.
    terrain : array-like
        2D terrain heightmap with shape (H, W). Can be a numpy array,
        cupy array, or xarray DataArray.
    scale : float, optional
        Scale factor for elevation values. Default is 1.

    Returns
    -------
    int
        0 on success.

    Notes
    -----
    - If terrain.data is a numpy array, uses parallel CPU processing via numba.
    - If terrain.data is a cupy array, uses GPU processing via CUDA kernels.
    """
    H, W = terrain.shape
    if isinstance(terrain.data, np.ndarray):
        _triangulateCPU(verts, triangles, terrain.data, H, W, scale)
    if isinstance(terrain.data, cupy.ndarray):
        jobSize = H*W
        blockdim = 1024
        griddim = (jobSize + blockdim - 1) // 1024
        d = 100
        offset = 0
        while (jobSize > 0):
            batch = min(d, griddim)
            _triangulateTerrain[batch, blockdim](verts, triangles,
                                                 terrain.data, H, W,
                                                 scale, offset)
            offset += batch*blockdim
            jobSize -= batch*blockdim
    return 0


@nb.jit(nopython=True)
def _fillContents(content, verts, triangles, numTris):
    v = np.empty(12, np.float32)
    pad = np.zeros(2, np.int8)
    offset = 0
    for i in range(numTris):
        t0 = triangles[3*i+0]
        t1 = triangles[3*i+1]
        t2 = triangles[3*i+2]
        v[3*0+0] = 0
        v[3*0+1] = 0
        v[3*0+2] = 0
        v[3*1+0] = verts[3*t0+0]
        v[3*1+1] = verts[3*t0+1]
        v[3*1+2] = verts[3*t0+2]
        v[3*2+0] = verts[3*t1+0]
        v[3*2+1] = verts[3*t1+1]
        v[3*2+2] = verts[3*t1+2]
        v[3*3+0] = verts[3*t2+0]
        v[3*3+1] = verts[3*t2+1]
        v[3*3+2] = verts[3*t2+2]

        offset = 50*i
        content[offset:offset+48] = v.view(np.uint8)
        content[offset+48:offset+50] = pad


def write_stl(name, verts, triangles):
    """
    Save a triangulated mesh to a standard STL file.

    STL is a widely supported 3D mesh format. Windows has a default STL viewer
    and most 3D applications support it natively due to its simplicity.

    Parameters
    ----------
    name : str
        The name of the mesh file to save. Should end in .stl
    verts : array-like
        Vertex buffer containing all mesh vertices. Format is 3 float32 values
        per vertex (x, y, z coordinates). Can be numpy or cupy array.
    triangles : array-like
        Index buffer containing all mesh triangles. Format is 3 int32 values
        per triangle (vertex indices). Can be numpy or cupy array.

    Notes
    -----
    If input arrays are cupy arrays, they will be automatically converted
    to numpy arrays for file writing.
    """
    ib = triangles
    vb = verts
    if isinstance(ib, cupy.ndarray):
        ib = cupy.asnumpy(ib)
    if isinstance(vb, cupy.ndarray):
        vb = cupy.asnumpy(vb)

    header = np.zeros(80, np.uint8)
    nf = np.empty(1, np.uint32)
    numTris = triangles.shape[0] // 3
    nf[0] = numTris
    f = open(name, 'wb')
    f.write(header)
    f.write(nf)

    # size of 1 triangle in STL is 50 bytes
    # 12 floats (each 4 bytes) for a total of 48
    # And additional 2 bytes for padding
    content = np.empty(numTris*(50), np.uint8)
    _fillContents(content, vb, ib, numTris)
    f.write(content)
    f.close()


# Backwards compatibility aliases
triangulateTerrain = triangulate_terrain
write = write_stl
