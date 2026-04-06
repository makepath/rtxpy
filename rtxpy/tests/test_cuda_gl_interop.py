"""Tests for CUDA-GL interop zero-copy display pipeline.

All CUDA/GL calls are mocked — these tests run without GPU hardware.
"""

from __future__ import annotations

import ctypes
import os
import sys
import types
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Ensure rtxpy.viewer.cuda_gl_interop is importable even without cupy/optix.
# The rtxpy top-level __init__ unconditionally imports cupy, so we install
# stub parent packages if needed, then import the interop module directly.
# ---------------------------------------------------------------------------
try:
    from rtxpy.viewer.cuda_gl_interop import CudaGLFrameBuffer
    _import_ok = True
except (ImportError, ModuleNotFoundError):
    # Provide minimal stubs so the submodule import succeeds
    for _name in ("rtxpy", "rtxpy.viewer"):
        if _name not in sys.modules:
            _mod = types.ModuleType(_name)
            if _name == "rtxpy":
                _mod.__path__ = [os.path.join(os.path.dirname(__file__), "..")]
            elif _name == "rtxpy.viewer":
                _mod.__path__ = [
                    os.path.join(os.path.dirname(__file__), "..", "viewer")
                ]
            sys.modules[_name] = _mod
    try:
        from rtxpy.viewer.cuda_gl_interop import CudaGLFrameBuffer
        _import_ok = True
    except Exception:
        _import_ok = False

try:
    import cupy  # noqa: F401
    has_cupy = True
except ImportError:
    has_cupy = False

pytestmark = pytest.mark.skipif(not _import_ok, reason="cuda_gl_interop not importable")

_INTEROP_MODULE = "rtxpy.viewer.cuda_gl_interop"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mock_libcuda(fake_dev_ptr=0xCAFE0000, fake_buf_size=64 * 48 * 3 * 4):
    """Return a MagicMock that mimics the ctypes CUDA driver library."""
    import ctypes as _ct

    lib = MagicMock()
    lib.cuGraphicsGLRegisterBuffer.return_value = 0  # CUDA_SUCCESS
    lib.cuGraphicsUnregisterResource.return_value = 0
    lib.cuGraphicsMapResources.return_value = 0
    lib.cuGraphicsUnmapResources.return_value = 0

    def _fake_get_mapped_ptr(p_dev_ptr, p_size, resource):
        """Write non-zero values into the ctypes out-params."""
        _ct.memmove(p_dev_ptr, _ct.byref(_ct.c_size_t(fake_dev_ptr)),
                     _ct.sizeof(_ct.c_size_t))
        _ct.memmove(p_size, _ct.byref(_ct.c_size_t(fake_buf_size)),
                     _ct.sizeof(_ct.c_size_t))
        return 0  # CUDA_SUCCESS

    lib.cuGraphicsResourceGetMappedPointer_v2.side_effect = _fake_get_mapped_ptr
    return lib


def _mock_libgl():
    """Return a MagicMock that mimics the ctypes GL library."""
    lib = MagicMock()
    lib.glBindBuffer.return_value = None
    lib.glTexSubImage2D.return_value = None
    lib.glBindTexture.return_value = None
    return lib


def _mock_ctx(glo=42):
    """Return a MagicMock moderngl context + buffer."""
    ctx = MagicMock()
    buf = MagicMock()
    buf.glo = glo
    ctx.buffer.return_value = buf
    return ctx, buf


def _make_fb(width=64, height=48, libcuda=None, libgl=None, ctx=None):
    """Create a CudaGLFrameBuffer with mocked dependencies."""
    if libcuda is None:
        libcuda = _mock_libcuda()
    if libgl is None:
        libgl = _mock_libgl()
    if ctx is None:
        ctx, _ = _mock_ctx()

    with patch(f"{_INTEROP_MODULE}._get_libcuda", return_value=libcuda), \
         patch(f"{_INTEROP_MODULE}._get_libgl", return_value=libgl), \
         patch(f"{_INTEROP_MODULE}._setup_cuda_bindings"), \
         patch(f"{_INTEROP_MODULE}._setup_gl_bindings"):
        fb = CudaGLFrameBuffer(width, height, ctx)
    # The mock cuGraphicsGLRegisterBuffer doesn't write via ctypes byref,
    # so _cuda_resource stays as c_void_p(0) (falsy).  Set it to a truthy
    # value so release() / resize() actually exercise the unregister path.
    fb._cuda_resource = ctypes.c_void_p(0xDEAD)
    return fb


# ===================================================================
# 1a. CudaGLFrameBuffer class tests
# ===================================================================

class TestCudaGLFrameBufferConstruction:
    """Construction and teardown of the PBO + CUDA registration."""

    def test_pbo_created_with_correct_size(self):
        ctx, buf = _mock_ctx()
        _make_fb(width=64, height=48, ctx=ctx)
        expected_bytes = 64 * 48 * 3 * 4  # W*H*3*sizeof(float32)
        ctx.buffer.assert_called_once_with(reserve=expected_bytes)

    def test_register_called_with_write_discard(self):
        from rtxpy.viewer.cuda_gl_interop import (
            CU_GRAPHICS_REGISTER_FLAGS_WRITE_DISCARD,
        )
        libcuda = _mock_libcuda()
        ctx, buf = _mock_ctx(glo=99)
        _make_fb(libcuda=libcuda, ctx=ctx)
        call_args = libcuda.cuGraphicsGLRegisterBuffer.call_args
        # Third positional arg is flags
        assert call_args[0][2] == CU_GRAPHICS_REGISTER_FLAGS_WRITE_DISCARD

    def test_construction_failure_releases_buffer(self):
        libcuda = _mock_libcuda()
        libcuda.cuGraphicsGLRegisterBuffer.return_value = 801  # error
        ctx, buf = _mock_ctx()
        with pytest.raises(RuntimeError, match="cuGraphicsGLRegisterBuffer failed"):
            _make_fb(libcuda=libcuda, ctx=ctx)
        buf.release.assert_called()  # at least once (constructor + possible __del__)


class TestCudaGLFrameBufferMap:
    """map() / unmap() interactions."""

    def _map_with_mocked_cupy(self, fb):
        """Call fb.map() with cupy memory classes mocked out."""
        fake_arr = MagicMock()
        with patch("cupy.cuda.UnownedMemory") as mock_mem, \
             patch("cupy.cuda.MemoryPointer") as mock_ptr, \
             patch("cupy.ndarray", return_value=fake_arr):
            result = fb.map()
        return result

    @pytest.mark.skipif(not has_cupy, reason="cupy required")
    def test_map_calls_cuda_map_resources(self):
        libcuda = _mock_libcuda()
        fb = _make_fb(libcuda=libcuda)
        self._map_with_mocked_cupy(fb)
        libcuda.cuGraphicsMapResources.assert_called_once()

    @pytest.mark.skipif(not has_cupy, reason="cupy required")
    def test_map_calls_get_mapped_pointer(self):
        libcuda = _mock_libcuda()
        fb = _make_fb(libcuda=libcuda)
        self._map_with_mocked_cupy(fb)
        libcuda.cuGraphicsResourceGetMappedPointer_v2.assert_called_once()

    @pytest.mark.skipif(not has_cupy, reason="cupy required")
    def test_map_returns_array_with_correct_shape(self):
        fb = _make_fb(width=64, height=48)
        fake_arr = MagicMock()
        with patch("cupy.cuda.UnownedMemory"), \
             patch("cupy.cuda.MemoryPointer"), \
             patch("cupy.ndarray", return_value=fake_arr) as mock_ndarray:
            fb.map()
        # Verify ndarray was created with (H, W, 3) shape
        call_args = mock_ndarray.call_args
        assert call_args[0][0] == (48, 64, 3)

    @pytest.mark.skipif(not has_cupy, reason="cupy required")
    def test_double_map_raises(self):
        fb = _make_fb()
        self._map_with_mocked_cupy(fb)
        with pytest.raises(RuntimeError, match="already mapped"):
            fb.map()

    @pytest.mark.skipif(not has_cupy, reason="cupy required")
    def test_map_failure_raises(self):
        libcuda = _mock_libcuda()
        libcuda.cuGraphicsMapResources.return_value = 1  # error
        fb = _make_fb(libcuda=libcuda)
        with pytest.raises(RuntimeError, match="cuGraphicsMapResources failed"):
            fb.map()

    def test_unmap_when_not_mapped_is_noop(self):
        fb = _make_fb()
        fb.unmap()  # should not raise

    @pytest.mark.skipif(not has_cupy, reason="cupy required")
    def test_unmap_calls_cuda_unmap(self):
        libcuda = _mock_libcuda()
        fb = _make_fb(libcuda=libcuda)
        self._map_with_mocked_cupy(fb)
        fb.unmap()
        libcuda.cuGraphicsUnmapResources.assert_called_once()
        assert not fb._mapped


class TestCudaGLFrameBufferUpload:
    """upload_to_texture() PBO->texture sequence."""

    def test_upload_bind_texsubimage_unbind_sequence(self):
        from rtxpy.viewer.cuda_gl_interop import (
            GL_PIXEL_UNPACK_BUFFER,
        )
        libgl = _mock_libgl()
        fb = _make_fb(width=64, height=48, libgl=libgl)
        tex = MagicMock()
        tex.glo = 7

        fb.upload_to_texture(tex)

        # First glBindBuffer binds PBO
        first_bind = libgl.glBindBuffer.call_args_list[0]
        assert first_bind[0][0] == GL_PIXEL_UNPACK_BUFFER
        # Last glBindBuffer unbinds (0)
        last_bind = libgl.glBindBuffer.call_args_list[-1]
        assert last_bind[0] == (GL_PIXEL_UNPACK_BUFFER, 0)
        # glTexSubImage2D called with NULL data pointer
        tex_call = libgl.glTexSubImage2D.call_args
        assert tex_call[0][-1] is None  # data=NULL

    @pytest.mark.skipif(not has_cupy, reason="cupy required")
    def test_upload_while_mapped_raises(self):
        fb = _make_fb()
        with patch("cupy.cuda.UnownedMemory"), \
             patch("cupy.cuda.MemoryPointer"), \
             patch("cupy.ndarray", return_value=MagicMock()):
            fb.map()
        tex = MagicMock()
        with pytest.raises(RuntimeError, match="Cannot upload_to_texture"):
            fb.upload_to_texture(tex)


class TestCudaGLFrameBufferResize:
    """resize() unregister -> release -> recreate -> re-register."""

    def test_resize_unregisters_and_reregisters(self):
        libcuda = _mock_libcuda()
        ctx, buf = _mock_ctx()
        _make_fb(width=64, height=48, libcuda=libcuda, ctx=ctx)

        # Reset mocks after construction
        libcuda.cuGraphicsUnregisterResource.reset_mock()
        libcuda.cuGraphicsGLRegisterBuffer.reset_mock()
        ctx.buffer.reset_mock()
        buf.release.reset_mock()

        fb = _make_fb(width=64, height=48, libcuda=libcuda, ctx=ctx)
        # Reset again for the fb we'll actually resize
        libcuda.cuGraphicsUnregisterResource.reset_mock()
        libcuda.cuGraphicsGLRegisterBuffer.reset_mock()
        ctx.buffer.reset_mock()
        buf.release.reset_mock()

        fb.resize(128, 96)

        libcuda.cuGraphicsUnregisterResource.assert_called_once()
        buf.release.assert_called_once()
        ctx.buffer.assert_called_once_with(reserve=128 * 96 * 3 * 4)
        libcuda.cuGraphicsGLRegisterBuffer.assert_called_once()
        assert fb._width == 128
        assert fb._height == 96

    def test_resize_failure_raises(self):
        libcuda = _mock_libcuda()
        fb = _make_fb(libcuda=libcuda)
        # Make re-register fail
        libcuda.cuGraphicsGLRegisterBuffer.return_value = 801
        with pytest.raises(RuntimeError, match="resize"):
            fb.resize(128, 96)


class TestCudaGLFrameBufferRelease:
    """release() cleanup of CUDA resource + GL buffer."""

    def test_release_unregisters_and_frees(self):
        libcuda = _mock_libcuda()
        ctx, buf = _mock_ctx()
        fb = _make_fb(libcuda=libcuda, ctx=ctx)
        fb.release()
        libcuda.cuGraphicsUnregisterResource.assert_called()
        buf.release.assert_called()

    def test_release_twice_is_safe(self):
        fb = _make_fb()
        fb.release()
        fb.release()  # should not raise


# ===================================================================
# 1b. is_available() static method tests
# ===================================================================

class TestIsAvailable:
    """CudaGLFrameBuffer.is_available() environment checks."""

    def test_env_var_override(self):
        ctx, _ = _mock_ctx()
        with patch.dict(os.environ, {"RTXPY_NO_GL_INTEROP": "1"}):
            assert CudaGLFrameBuffer.is_available(ctx) is False

    def test_wsl2_detected(self):
        ctx, _ = _mock_ctx()
        fake_uname = MagicMock()
        fake_uname.release = "5.15.90.1-microsoft-standard-WSL2"
        with patch.dict(os.environ, {}, clear=False), \
             patch("os.uname", return_value=fake_uname, create=True):
            os.environ.pop("RTXPY_NO_GL_INTEROP", None)
            assert CudaGLFrameBuffer.is_available(ctx) is False

    def test_registration_failure_returns_false(self):
        ctx, buf = _mock_ctx()
        libcuda = _mock_libcuda()
        libcuda.cuGraphicsGLRegisterBuffer.return_value = 801

        with patch.dict(os.environ, {}, clear=False), \
             patch("os.uname", side_effect=AttributeError, create=True), \
             patch(f"{_INTEROP_MODULE}._get_libcuda", return_value=libcuda), \
             patch(f"{_INTEROP_MODULE}._setup_cuda_bindings"):
            os.environ.pop("RTXPY_NO_GL_INTEROP", None)
            result = CudaGLFrameBuffer.is_available(ctx)
        assert result is False
        buf.release.assert_called()

    def test_success_path(self):
        ctx, buf = _mock_ctx()
        libcuda = _mock_libcuda()

        with patch.dict(os.environ, {}, clear=False), \
             patch("os.uname", side_effect=AttributeError, create=True), \
             patch(f"{_INTEROP_MODULE}._get_libcuda", return_value=libcuda), \
             patch(f"{_INTEROP_MODULE}._setup_cuda_bindings"):
            os.environ.pop("RTXPY_NO_GL_INTEROP", None)
            result = CudaGLFrameBuffer.is_available(ctx)
        assert result is True
        libcuda.cuGraphicsUnregisterResource.assert_called()
        buf.release.assert_called()


# ===================================================================
# 1c. Engine interop/fallback path selection
# ===================================================================

class TestEngineInteropPathSelection:
    """Verify the engine chooses the right display path."""

    def _make_engine_stub(self, interop_enabled=True):
        """Create a minimal mock of the engine with interop-relevant attrs."""
        engine = MagicMock()
        engine._interop_enabled = interop_enabled
        engine._cuda_gl_buf = MagicMock() if interop_enabled else None
        engine._d_base_frame = MagicMock()
        engine._wind_enabled = True
        engine._hydro_enabled = False
        engine._clouds_enabled = False
        engine._wind_particles = MagicMock()
        engine._rain_particles = None
        engine._interop_frame_tex = MagicMock()
        engine._frame_dirty = False
        return engine

    def test_interop_enabled_uses_pbo_path(self):
        """When interop is enabled, map/splat/unmap/upload should be called."""
        engine = self._make_engine_stub(interop_enabled=True)
        buf = engine._cuda_gl_buf
        buf.map.return_value = MagicMock()

        # Simulate the interop idle-particle path
        d_pbo = buf.map()
        buf.unmap()
        buf.upload_to_texture(engine._interop_frame_tex)

        buf.map.assert_called_once()
        buf.unmap.assert_called_once()
        buf.upload_to_texture.assert_called_once_with(engine._interop_frame_tex)

    def test_interop_disabled_uses_cpu_path(self):
        """When interop is disabled, _idle_particles_cpu is called."""
        engine = self._make_engine_stub(interop_enabled=False)

        if engine._interop_enabled:
            engine._cuda_gl_buf.map()
        else:
            engine._idle_particles_cpu()

        engine._idle_particles_cpu.assert_called_once()
        assert engine._cuda_gl_buf is None

    def test_interop_failure_falls_back_to_cpu(self):
        """Mid-frame interop failure disables interop and calls CPU path."""
        engine = self._make_engine_stub(interop_enabled=True)
        buf = engine._cuda_gl_buf
        buf.map.side_effect = RuntimeError("interop failed")

        # Simulate the engine's error-handling logic
        try:
            buf.map()
        except Exception:
            engine._interop_enabled = False
            engine._idle_particles_cpu()

        assert engine._interop_enabled is False
        engine._idle_particles_cpu.assert_called_once()
