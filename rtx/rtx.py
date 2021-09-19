import ctypes
import sys, os
import cupy

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
            dir_path = dir_path + "\\rtx.dll"
        else:
            dir_path = dir_path + "/librtx.so"
        
        try:
            c_lib = ctypes.CDLL(dir_path)
            c_lib.initRTX.restype = ctypes.c_int
            c_lib.buildRTX.restype = ctypes.c_int
            c_lib.traceRTX.restype = ctypes.c_int
            c_lib.cleanRTX.restype = ctypes.c_int
            c_lib.getHashRTX.restype = ctypes.c_uint64
            if c_lib.initRTX():
                free_optix_resources()
                raise "Failed to initialize RTX library"
            else:
                atexit.register(free_optix_resources)
        except:
            raise "Failed not load RTX library"
    
    def build(self, hashValue, vertexBuffer, indexBuffer):
        if isinstance(vertexBuffer, cupy.ndarray):
            vb = ctypes.c_ulonglong(vertexBuffer.data.ptr)
        else:
            vb = vertexBuffer.ctypes.data_as(ctypes.POINTER(ctypes.c_void_p))
        if isinstance(indexBuffer, cupy.ndarray):
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
            raise "Cannot communicate with OptiX"

        return res
    def getHash(self):
        if c_lib:
            return c_lib.getHashRTX()
        else:
            raise "Cannot communicate with OptiX"

    def trace(self, rays, hits, numRays):
        if (rays.size != hits.size*2):
            return -1
        if isinstance(rays, cupy.ndarray):
            rays = ctypes.c_ulonglong(rays.data.ptr)
        else:
            rays = rays.ctypes.data_as(ctypes.POINTER(ctypes.c_void_p))
        if isinstance(hits, cupy.ndarray):
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
            raise "Cannot communicate with OptiX"
        return res
