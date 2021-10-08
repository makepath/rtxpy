import ctypes
import sys, os

try:
    import cupy
    has_cupy = True
except ModuleNotFoundError:
    has_cupy = False

import atexit
#Handle to the OptiX library.
c_lib = None

def free_optix_resources():
    global c_lib
    if c_lib:
        c_lib.cleanRTX()
    c_lib = None

class RTX():
    def __init__(self):
        global c_lib
        if c_lib != None:
            return

        dir_path = os.path.dirname(os.path.realpath(__file__))
        # Load the shared library into c types.
        if sys.platform.startswith("win"):
            dir_path = dir_path + "\\rtxpy.dll"
        elif sys.platform == "darwin":
            dir_path = dir_path + "/librtxpy.dylib"
        else:
            dir_path = dir_path + "/librtxpy.so"

        try:
            c_lib = ctypes.CDLL(dir_path)
            c_lib.initRTX.restype = ctypes.c_int
            c_lib.buildRTX.restype = ctypes.c_int
            c_lib.traceRTX.restype = ctypes.c_int
            c_lib.cleanRTX.restype = ctypes.c_int
            c_lib.getHashRTX.restype = ctypes.c_uint64
        except:
            raise RuntimeError("Failed to load RTX library")

        if c_lib.initRTX():
            free_optix_resources()
            raise RuntimeError("Failed to initialize RTX library")
        else:
            atexit.register(free_optix_resources)

    def build(self, hashValue, vertexBuffer, indexBuffer):
        if has_cupy and isinstance(vertexBuffer, cupy.ndarray):
            vb = ctypes.c_ulonglong(vertexBuffer.data.ptr)
        else:
            vb = vertexBuffer.ctypes.data_as(ctypes.POINTER(ctypes.c_void_p))
        if has_cupy and isinstance(indexBuffer, cupy.ndarray):
            ib = ctypes.c_ulonglong(indexBuffer.data.ptr)
        else:
            ib = indexBuffer.ctypes.data_as(ctypes.POINTER(ctypes.c_void_p))

        if c_lib:
            res = c_lib.buildRTX(
                ctypes.c_uint64(hashValue),
                vb,
                vertexBuffer.size*4, #sizeof(float) is 4
                ib,
                indexBuffer.size*4 #sizeof(int) is 4
            )
        else:
            raise RuntimeError("Cannot communicate with OptiX")

        return res

    def getHash(self):
        if c_lib:
            return c_lib.getHashRTX()
        else:
            raise RuntimeError("Cannot communicate with OptiX")

    def trace(self, rays, hits, numRays):
        if (rays.size != hits.size*2):
            return -1
        if has_cupy and isinstance(rays, cupy.ndarray):
            rays = ctypes.c_ulonglong(rays.data.ptr)
        else:
            rays = rays.ctypes.data_as(ctypes.POINTER(ctypes.c_void_p))
        if has_cupy and isinstance(hits, cupy.ndarray):
            hits = ctypes.c_ulonglong(hits.data.ptr)
        else:
            hits = hits.ctypes.data_as(ctypes.POINTER(ctypes.c_void_p))

        if c_lib:
            res = c_lib.traceRTX(
                rays,
                hits,
                numRays
            )
        else:
            raise Exception("Cannot communicate with OptiX")
        return res
