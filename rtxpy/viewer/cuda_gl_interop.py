"""CUDA-GL interop for zero-copy display via PBO.

Registers an OpenGL Pixel Buffer Object (PBO) with CUDA so the GPU can
write directly into GL-visible memory.  The PBO is then uploaded to the
display texture via a GPU-internal ``glTexSubImage2D`` (no CPU involvement).

Falls back gracefully on WSL2 / headless / driver mismatch.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import logging
import os
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    import cupy as cp
    import moderngl

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# GL constants
# ---------------------------------------------------------------------------
GL_PIXEL_UNPACK_BUFFER = 0x88EC
GL_FLOAT = 0x1406
GL_RGB = 0x1906
GL_TEXTURE_2D = 0x0DE1

# ---------------------------------------------------------------------------
# CUDA driver API constants
# ---------------------------------------------------------------------------
CU_GRAPHICS_REGISTER_FLAGS_NONE = 0
CU_GRAPHICS_REGISTER_FLAGS_WRITE_DISCARD = 2

# CUDA error codes
CUDA_SUCCESS = 0
CUDA_ERROR_NOT_SUPPORTED = 801

# ---------------------------------------------------------------------------
# Lazy-loaded shared library handles
# ---------------------------------------------------------------------------
_libcuda: Optional[ctypes.CDLL] = None
_libgl: Optional[ctypes.CDLL] = None


def _get_libcuda() -> ctypes.CDLL:
    global _libcuda
    if _libcuda is None:
        path = ctypes.util.find_library("cuda")
        if path is None:
            # Common fallback paths
            for candidate in ("libcuda.so.1", "libcuda.so"):
                try:
                    _libcuda = ctypes.CDLL(candidate)
                    break
                except OSError:
                    continue
            if _libcuda is None:
                raise OSError("Cannot find libcuda.so")
        else:
            _libcuda = ctypes.CDLL(path)
    return _libcuda


def _get_libgl() -> ctypes.CDLL:
    global _libgl
    if _libgl is None:
        path = ctypes.util.find_library("GL")
        if path is None:
            for candidate in ("libGL.so.1", "libGL.so"):
                try:
                    _libgl = ctypes.CDLL(candidate)
                    break
                except OSError:
                    continue
            if _libgl is None:
                raise OSError("Cannot find libGL.so")
        else:
            _libgl = ctypes.CDLL(path)
    return _libgl


# ---------------------------------------------------------------------------
# ctypes signatures for CUDA driver API
# ---------------------------------------------------------------------------
def _setup_cuda_bindings(lib: ctypes.CDLL):
    """Set up argtypes/restypes for the CUDA driver API functions we need."""
    # cuGraphicsGLRegisterBuffer(CUgraphicsResource *pResource, GLuint buffer, unsigned int flags)
    lib.cuGraphicsGLRegisterBuffer.argtypes = [
        ctypes.POINTER(ctypes.c_void_p),  # pResource
        ctypes.c_uint,                     # buffer (GLuint)
        ctypes.c_uint,                     # flags
    ]
    lib.cuGraphicsGLRegisterBuffer.restype = ctypes.c_int

    # cuGraphicsUnregisterResource(CUgraphicsResource resource)
    lib.cuGraphicsUnregisterResource.argtypes = [ctypes.c_void_p]
    lib.cuGraphicsUnregisterResource.restype = ctypes.c_int

    # cuGraphicsMapResources(unsigned int count, CUgraphicsResource *resources, CUstream stream)
    lib.cuGraphicsMapResources.argtypes = [
        ctypes.c_uint,
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.c_void_p,  # CUstream (0 = default)
    ]
    lib.cuGraphicsMapResources.restype = ctypes.c_int

    # cuGraphicsUnmapResources(unsigned int count, CUgraphicsResource *resources, CUstream stream)
    lib.cuGraphicsUnmapResources.argtypes = [
        ctypes.c_uint,
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.c_void_p,
    ]
    lib.cuGraphicsUnmapResources.restype = ctypes.c_int

    # cuGraphicsResourceGetMappedPointer(CUdeviceptr *pDevPtr, size_t *pSize,
    #                                     CUgraphicsResource resource)
    lib.cuGraphicsResourceGetMappedPointer_v2.argtypes = [
        ctypes.POINTER(ctypes.c_size_t),  # pDevPtr
        ctypes.POINTER(ctypes.c_size_t),  # pSize
        ctypes.c_void_p,                  # resource
    ]
    lib.cuGraphicsResourceGetMappedPointer_v2.restype = ctypes.c_int


def _setup_gl_bindings(lib: ctypes.CDLL):
    """Set up argtypes/restypes for the raw GL calls we need."""
    # glBindBuffer(GLenum target, GLuint buffer)
    lib.glBindBuffer.argtypes = [ctypes.c_uint, ctypes.c_uint]
    lib.glBindBuffer.restype = None

    # glTexSubImage2D(GLenum target, GLint level, GLint xoffset, GLint yoffset,
    #                 GLsizei width, GLsizei height, GLenum format, GLenum type,
    #                 const void *pixels)
    lib.glTexSubImage2D.argtypes = [
        ctypes.c_uint,   # target
        ctypes.c_int,    # level
        ctypes.c_int,    # xoffset
        ctypes.c_int,    # yoffset
        ctypes.c_int,    # width
        ctypes.c_int,    # height
        ctypes.c_uint,   # format
        ctypes.c_uint,   # type
        ctypes.c_void_p, # pixels (NULL for PBO)
    ]
    lib.glTexSubImage2D.restype = None

    # glBindTexture(GLenum target, GLuint texture)
    lib.glBindTexture.argtypes = [ctypes.c_uint, ctypes.c_uint]
    lib.glBindTexture.restype = None


# ---------------------------------------------------------------------------
# CudaGLFrameBuffer
# ---------------------------------------------------------------------------
class CudaGLFrameBuffer:
    """Manages a PBO registered with CUDA for zero-copy GPU→GL display.

    Parameters
    ----------
    width, height : int
        Frame dimensions.
    ctx : moderngl.Context
        ModernGL context (must be current).
    """

    def __init__(self, width: int, height: int, ctx: "moderngl.Context"):
        self._width = width
        self._height = height
        self._ctx = ctx
        self._mapped = False

        # Load and set up ctypes bindings
        self._cuda = _get_libcuda()
        self._gl = _get_libgl()
        _setup_cuda_bindings(self._cuda)
        _setup_gl_bindings(self._gl)

        # Create GL buffer (PBO) via ModernGL
        nbytes = width * height * 3 * 4  # RGB float32
        self._gl_buf = ctx.buffer(reserve=nbytes)
        self._glo = self._gl_buf.glo  # raw GLuint handle

        # Register with CUDA
        self._cuda_resource = ctypes.c_void_p(0)
        err = self._cuda.cuGraphicsGLRegisterBuffer(
            ctypes.byref(self._cuda_resource),
            ctypes.c_uint(self._glo),
            CU_GRAPHICS_REGISTER_FLAGS_WRITE_DISCARD,
        )
        if err != CUDA_SUCCESS:
            self._gl_buf.release()
            raise RuntimeError(
                f"cuGraphicsGLRegisterBuffer failed with error {err}"
            )

    @property
    def width(self) -> int:
        return self._width

    @property
    def height(self) -> int:
        return self._height

    def map(self) -> "cp.ndarray":
        """Map the PBO for CUDA writes and return a cupy array.

        Returns a cupy ndarray of shape ``(H, W, 3)`` with dtype float32
        wrapping the PBO's device memory.  The array is valid until
        :meth:`unmap` is called.
        """
        import cupy as cp

        if self._mapped:
            raise RuntimeError("CudaGLFrameBuffer is already mapped")

        res_arr = (ctypes.c_void_p * 1)(self._cuda_resource)
        err = self._cuda.cuGraphicsMapResources(1, res_arr, None)
        if err != CUDA_SUCCESS:
            raise RuntimeError(f"cuGraphicsMapResources failed: {err}")
        self._mapped = True

        dev_ptr = ctypes.c_size_t(0)
        buf_size = ctypes.c_size_t(0)
        err = self._cuda.cuGraphicsResourceGetMappedPointer_v2(
            ctypes.byref(dev_ptr), ctypes.byref(buf_size), self._cuda_resource
        )
        if err != CUDA_SUCCESS:
            self.unmap()
            raise RuntimeError(
                f"cuGraphicsResourceGetMappedPointer_v2 failed: {err}"
            )

        # Wrap the device pointer in a cupy array (no copy)
        mem = cp.cuda.UnownedMemory(dev_ptr.value, buf_size.value, owner=self)
        memptr = cp.cuda.MemoryPointer(mem, 0)
        arr = cp.ndarray(
            (self._height, self._width, 3), dtype=cp.float32, memptr=memptr
        )
        return arr

    def unmap(self):
        """Release the PBO back to GL after CUDA writes are done."""
        if not self._mapped:
            return
        res_arr = (ctypes.c_void_p * 1)(self._cuda_resource)
        self._cuda.cuGraphicsUnmapResources(1, res_arr, None)
        self._mapped = False

    def upload_to_texture(self, tex: "moderngl.Texture"):
        """GPU-internal PBO→texture upload (no CPU involvement).

        Binds the PBO as ``GL_PIXEL_UNPACK_BUFFER`` and calls
        ``glTexSubImage2D`` with ``data=NULL`` to trigger a GPU-side copy
        from the PBO into the texture.
        """
        if self._mapped:
            raise RuntimeError(
                "Cannot upload_to_texture while PBO is mapped for CUDA"
            )

        # Bind PBO as pixel unpack source
        self._gl.glBindBuffer(GL_PIXEL_UNPACK_BUFFER, self._glo)

        # Bind the target texture
        self._gl.glBindTexture(GL_TEXTURE_2D, tex.glo)

        # GPU-internal copy: data=NULL means read from bound PBO
        self._gl.glTexSubImage2D(
            GL_TEXTURE_2D,
            0,       # level
            0, 0,    # xoffset, yoffset
            self._width,
            self._height,
            GL_RGB,
            GL_FLOAT,
            None,    # NULL → read from PBO
        )

        # Unbind PBO
        self._gl.glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0)

    def resize(self, width: int, height: int):
        """Resize the PBO (unregister → recreate → re-register)."""
        if self._mapped:
            self.unmap()

        # Unregister old resource
        self._cuda.cuGraphicsUnregisterResource(self._cuda_resource)

        # Release old GL buffer
        self._gl_buf.release()

        # Create new GL buffer
        self._width = width
        self._height = height
        nbytes = width * height * 3 * 4
        self._gl_buf = self._ctx.buffer(reserve=nbytes)
        self._glo = self._gl_buf.glo

        # Re-register
        self._cuda_resource = ctypes.c_void_p(0)
        err = self._cuda.cuGraphicsGLRegisterBuffer(
            ctypes.byref(self._cuda_resource),
            ctypes.c_uint(self._glo),
            CU_GRAPHICS_REGISTER_FLAGS_WRITE_DISCARD,
        )
        if err != CUDA_SUCCESS:
            raise RuntimeError(
                f"cuGraphicsGLRegisterBuffer failed on resize: {err}"
            )

    def release(self):
        """Unregister and release all resources."""
        if self._mapped:
            self.unmap()
        if self._cuda_resource:
            self._cuda.cuGraphicsUnregisterResource(self._cuda_resource)
            self._cuda_resource = ctypes.c_void_p(0)
        if self._gl_buf is not None:
            self._gl_buf.release()
            self._gl_buf = None

    @staticmethod
    def is_available(ctx: "moderngl.Context") -> bool:
        """Test whether CUDA-GL interop works in the current environment.

        Returns False on WSL2, headless, or when the driver doesn't support
        CUDA-GL interop.  Also respects ``RTXPY_NO_GL_INTEROP=1``.
        """
        # Manual override
        if os.environ.get("RTXPY_NO_GL_INTEROP", "0") == "1":
            logger.info("CUDA-GL interop disabled via RTXPY_NO_GL_INTEROP=1")
            return False

        # WSL2 detection — software GL, interop won't work
        try:
            release = os.uname().release.lower()
            if "microsoft" in release:
                logger.info(
                    "CUDA-GL interop not available (WSL2 detected)"
                )
                return False
        except Exception:
            pass

        # Try to create a tiny buffer and register it
        try:
            buf = ctx.buffer(reserve=4 * 3 * 4)  # 2×2 RGB f32
            cuda = _get_libcuda()
            _setup_cuda_bindings(cuda)
            resource = ctypes.c_void_p(0)
            err = cuda.cuGraphicsGLRegisterBuffer(
                ctypes.byref(resource),
                ctypes.c_uint(buf.glo),
                CU_GRAPHICS_REGISTER_FLAGS_WRITE_DISCARD,
            )
            if err != CUDA_SUCCESS:
                buf.release()
                logger.info(
                    "CUDA-GL interop not available (register failed: %d)", err
                )
                return False
            # Success — clean up the test buffer
            cuda.cuGraphicsUnregisterResource(resource)
            buf.release()
            return True
        except Exception as exc:
            logger.info("CUDA-GL interop not available: %s", exc)
            return False

    def __del__(self):
        try:
            self.release()
        except Exception:
            pass
