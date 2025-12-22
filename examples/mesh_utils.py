
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
def triangulateCPU(verts, triangles, data, H, W, scale):
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


def triangulateTerrain(verts, triangles, terrain, scale=1):
    H, W = terrain.shape
    if isinstance(terrain.data, np.ndarray):
        triangulateCPU(verts, triangles, terrain.data, H, W, scale)
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
def fillContents(content, verts, triangles, numTris):
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


def write(name, verts, triangles):
    """
    Save a triangulated raster to a standard STL file.
    Windows has a default STL viewer and probably all 3D viewers have native support for it
    because of its simplicity. Can be used to verify the correctness of the algorithm or
    to visualize the mesh to get a notion of the size/complexity etc.
    @param name - The name of the mesh file we're going to save. Should end in .stl
    @param verts - A numpy array containing all the vertices of the mesh. Format is 3 float32 per vertex (vertex buffer)
    @param triangles - A numpy array containing all the triangles of the mesh. Format is 3 int32 per triangle (index buffer)
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
    fillContents(content, vb, ib, numTris)
    f.write(content)
    f.close()
