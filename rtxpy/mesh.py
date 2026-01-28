"""Mesh utilities for terrain triangulation and STL export.

This module provides functions for converting raster terrain data into
triangle meshes suitable for ray tracing, and for exporting meshes to STL format.
"""

import numba as nb
from numba import cuda
import numpy as np

from .rtx import has_cupy

if has_cupy:
    import cupy


@cuda.jit
def _triangulate_terrain_gpu(verts, triangles, data, H, W, scale, stride):
    """GPU kernel for terrain triangulation."""
    globalId = stride + cuda.grid(1)
    if globalId < W * H:
        h = globalId // W
        w = globalId % W
        meshMapIndex = h * W + w

        val = data[h, w]

        offset = 3 * meshMapIndex
        verts[offset] = w
        verts[offset + 1] = h
        verts[offset + 2] = val * scale

        if w != W - 1 and h != H - 1:
            offset = 6 * (h * (W - 1) + w)
            triangles[offset + 0] = np.int32(meshMapIndex + W)
            triangles[offset + 1] = np.int32(meshMapIndex + W + 1)
            triangles[offset + 2] = np.int32(meshMapIndex)
            triangles[offset + 3] = np.int32(meshMapIndex + W + 1)
            triangles[offset + 4] = np.int32(meshMapIndex + 1)
            triangles[offset + 5] = np.int32(meshMapIndex)


@nb.njit(parallel=True)
def _triangulate_terrain_cpu(verts, triangles, data, H, W, scale):
    """CPU implementation of terrain triangulation using numba."""
    for h in nb.prange(H):
        for w in range(W):
            meshMapIndex = h * W + w

            val = data[h, w]

            offset = 3 * meshMapIndex
            verts[offset] = w
            verts[offset + 1] = h
            verts[offset + 2] = val * scale

            if w != W - 1 and h != H - 1:
                offset = 6 * (h * (W - 1) + w)
                triangles[offset + 0] = np.int32(meshMapIndex + W)
                triangles[offset + 1] = np.int32(meshMapIndex + W + 1)
                triangles[offset + 2] = np.int32(meshMapIndex)
                triangles[offset + 3] = np.int32(meshMapIndex + W + 1)
                triangles[offset + 4] = np.int32(meshMapIndex + 1)
                triangles[offset + 5] = np.int32(meshMapIndex)


def triangulate_terrain(verts, triangles, terrain, scale=1.0):
    """Convert a 2D terrain array into a triangle mesh.

    This function populates the provided vertex and triangle buffers with
    mesh data representing the terrain surface. Each cell in the terrain
    becomes a vertex, and adjacent cells are connected with two triangles.

    Parameters
    ----------
    verts : array-like
        Output vertex buffer of shape (H * W * 3,) as float32.
        Will be populated with x, y, z coordinates for each vertex.
    triangles : array-like
        Output triangle index buffer of shape ((H-1) * (W-1) * 2 * 3,) as int32.
        Will be populated with vertex indices for each triangle.
    terrain : array-like
        2D array of elevation values with shape (H, W). Can be a numpy array,
        cupy array, or xarray DataArray. If an xarray DataArray is passed,
        its underlying data array will be used.
    scale : float, optional
        Scale factor applied to elevation values. Default is 1.0.

    Returns
    -------
    int
        0 on success.

    Notes
    -----
    The function automatically selects CPU or GPU execution based on the
    input array type. If terrain is a cupy array, GPU execution is used;
    otherwise, CPU execution with parallel numba is used.
    """
    # Handle xarray DataArray by extracting underlying data
    # Check for xarray DataArray specifically (has 'dims' attribute)
    if hasattr(terrain, 'dims') and hasattr(terrain, 'coords'):
        data = terrain.data  # xarray stores cupy/numpy array in .data
    else:
        data = terrain
    H, W = terrain.shape

    if isinstance(data, np.ndarray):
        _triangulate_terrain_cpu(verts, triangles, data, H, W, scale)
    elif has_cupy and isinstance(data, cupy.ndarray):
        jobSize = H * W
        blockdim = 1024
        griddim = (jobSize + blockdim - 1) // 1024
        d = 100
        offset = 0
        while jobSize > 0:
            batch = min(d, griddim)
            _triangulate_terrain_gpu[batch, blockdim](
                verts, triangles, data, H, W, scale, offset
            )
            offset += batch * blockdim
            jobSize -= batch * blockdim
    else:
        raise TypeError(
            f"Unsupported terrain data type: {type(data)}. "
            "Expected numpy.ndarray or cupy.ndarray."
        )

    return 0


@nb.jit(nopython=True)
def _fill_stl_contents(content, verts, triangles, numTris):
    """Fill STL binary content from mesh data."""
    v = np.empty(12, np.float32)
    pad = np.zeros(2, np.int8)
    for i in range(numTris):
        t0 = triangles[3 * i + 0]
        t1 = triangles[3 * i + 1]
        t2 = triangles[3 * i + 2]
        # Normal (set to zero, viewers recalculate)
        v[3 * 0 + 0] = 0
        v[3 * 0 + 1] = 0
        v[3 * 0 + 2] = 0
        # Vertex 0
        v[3 * 1 + 0] = verts[3 * t0 + 0]
        v[3 * 1 + 1] = verts[3 * t0 + 1]
        v[3 * 1 + 2] = verts[3 * t0 + 2]
        # Vertex 1
        v[3 * 2 + 0] = verts[3 * t1 + 0]
        v[3 * 2 + 1] = verts[3 * t1 + 1]
        v[3 * 2 + 2] = verts[3 * t1 + 2]
        # Vertex 2
        v[3 * 3 + 0] = verts[3 * t2 + 0]
        v[3 * 3 + 1] = verts[3 * t2 + 1]
        v[3 * 3 + 2] = verts[3 * t2 + 2]

        offset = 50 * i
        content[offset:offset + 48] = v.view(np.uint8)
        content[offset + 48:offset + 50] = pad


def write_stl(filename, verts, triangles):
    """Save a triangle mesh to a binary STL file.

    STL is a simple, widely-supported 3D format. Windows has a built-in
    STL viewer, and most 3D applications can open STL files.

    Parameters
    ----------
    filename : str
        Output file path. Should end with '.stl'.
    verts : array-like
        Vertex buffer as float32 with 3 values per vertex (x, y, z).
    triangles : array-like
        Triangle index buffer as int32 with 3 indices per triangle.

    Notes
    -----
    If cupy arrays are provided, they will be automatically converted
    to numpy arrays for writing.
    """
    ib = triangles
    vb = verts

    # Convert cupy arrays to numpy if needed
    if has_cupy:
        if isinstance(ib, cupy.ndarray):
            ib = cupy.asnumpy(ib)
        if isinstance(vb, cupy.ndarray):
            vb = cupy.asnumpy(vb)

    header = np.zeros(80, np.uint8)
    nf = np.empty(1, np.uint32)
    numTris = triangles.shape[0] // 3
    nf[0] = numTris

    with open(filename, 'wb') as f:
        f.write(header)
        f.write(nf)

        # Size of 1 triangle in STL is 50 bytes:
        # 12 floats (each 4 bytes) = 48 bytes + 2 bytes padding
        content = np.empty(numTris * 50, np.uint8)
        _fill_stl_contents(content, vb, ib, numTris)
        f.write(content)
