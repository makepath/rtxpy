"""Main run loop for the interactive viewer."""

import os
import threading
import time
from typing import Optional, Tuple

import numpy as np

from .constants import _QUAD_VERT, _QUAD_FRAG
from .helpers import _glfw_to_key
from ..rtx import has_cupy

if has_cupy:
    import cupy as cp


class RunManager:
    """The run() method: GLFW window creation, main loop, and cleanup.

    Accesses viewer state via ``self.v`` (back-reference to InteractiveViewer).
    """

    def __init__(self, viewer):
        self.v = viewer

    def run(self, start_position: Optional[Tuple[float, float, float]] = None,
            look_at: Optional[Tuple[float, float, float]] = None):
        """
        Run the interactive viewer.

        Parameters
        ----------
        start_position : tuple, optional
            Starting camera position (x, y, z). If None, positions
            camera at the south edge of the terrain looking north.
        look_at : tuple, optional
            Initial look-at point. If None, looks toward terrain center.
        """
        v = self.v
        import glfw
        import moderngl

        H, W = v.terrain_shape

        # World-coordinate extents (accounts for pixel_spacing)
        world_W = W * v.pixel_spacing_x
        world_H = H * v.pixel_spacing_y
        world_diag = np.sqrt(world_W**2 + world_H**2)
        v._scene_diagonal = world_diag

        # Set initial move speed based on terrain extent (~1% of diagonal per keystroke)
        if v.move_speed is None:
            v.move_speed = world_diag * 0.01

        # Default: south bottom middle, looking north toward terrain center
        if start_position is None:
            start_position = (
                world_W / 2,
                world_H * 1.05,
                v.elev_max + world_diag * 0.08,
            )

        v.position = np.array(start_position, dtype=np.float32)

        # Calculate initial yaw/pitch from look_at
        if look_at is not None:
            direction = np.array(look_at) - v.position
            direction = direction / (np.linalg.norm(direction) + 1e-8)
            v.yaw = np.degrees(np.arctan2(direction[1], direction[0]))
            v.pitch = np.degrees(np.arcsin(np.clip(direction[2], -1, 1)))
        else:
            # Look toward terrain center
            center = np.array([world_W / 2, world_H / 2, v.elev_mean])
            direction = center - v.position
            direction = direction / (np.linalg.norm(direction) + 1e-8)
            v.yaw = np.degrees(np.arctan2(direction[1], direction[0]))
            v.pitch = np.degrees(np.arcsin(np.clip(direction[2], -1, 1)))

        # Save terminal state before GLFW (it can alter termios)
        import sys
        _saved_termios = None
        try:
            import termios
            _saved_termios = termios.tcgetattr(sys.stdin.fileno())
        except (ImportError, OSError, ValueError):
            pass
        except Exception:
            # termios.error doesn't inherit from OSError on all platforms
            pass

        # --- GLFW window creation ---
        # Force X11 on GLFW 3.4+ to get window decorations (title bar with
        # min/max/close).  WSLg's Wayland compositor doesn't provide them.
        # Skip on native Windows where X11 is not available.
        if os.name != 'nt' and hasattr(glfw, 'PLATFORM'):
            glfw.init_hint(glfw.PLATFORM, glfw.PLATFORM_X11)
        if not glfw.init():
            raise RuntimeError("Failed to initialise GLFW")

        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
        glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, True)

        window = glfw.create_window(
            v.width, v.height,
            f'rtxpy \u2014 {v._title}', None, None,
        )
        if not window:
            glfw.terminate()
            raise RuntimeError("Failed to create GLFW window")

        glfw.make_context_current(window)
        glfw.swap_interval(0)  # No VSync — render as fast as GPU allows

        v._glfw_window = window

        # --- ModernGL context + fullscreen quad ---
        ctx = moderngl.create_context()
        v._mgl_ctx = ctx  # stored for _update_frame interop texture resize
        prog = ctx.program(vertex_shader=_QUAD_VERT, fragment_shader=_QUAD_FRAG)

        # Fullscreen quad: position (x, y) + UV (u, v)
        # V is flipped: v=1 at bottom of screen maps to row 0 (top of image),
        # because OpenGL textures start at the bottom but numpy row 0 is top.
        quad_data = np.array([
            # x,    y,   u,  v
            -1.0, -1.0, 0.0, 1.0,  # bottom-left  -> top of image
             1.0, -1.0, 1.0, 1.0,  # bottom-right -> top of image
            -1.0,  1.0, 0.0, 0.0,  # top-left     -> bottom of image
             1.0,  1.0, 1.0, 0.0,  # top-right    -> bottom of image
        ], dtype='f4')
        vbo = ctx.buffer(quad_data.tobytes())
        vao = ctx.simple_vertex_array(prog, vbo, 'in_pos', 'in_uv')

        # Bind texture units: frame on unit 0, overlay on unit 1
        prog['frame'].value = 0
        prog['overlay'].value = 1
        prog['has_overlay'].value = 0

        # Frame texture — sized to render resolution, updated every frame
        frame_tex = ctx.texture(
            (v.render_width, v.render_height), 3, dtype='f4',
        )
        frame_tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
        v._interop_frame_tex = frame_tex  # for _update_frame interop path

        # Overlay texture — RGBA for alpha-blended CPU overlays
        v._overlay_tex = ctx.texture(
            (v.render_width, v.render_height), 4, dtype='f4',
        )
        v._overlay_tex.filter = (moderngl.LINEAR, moderngl.LINEAR)

        # --- CUDA-GL interop ---
        v._interop_enabled = False
        v._cuda_gl_buf = None
        v._overlay_gen = 0      # bumped when overlay content changes
        v._overlay_uploaded = -1  # last generation uploaded to GPU
        if has_cupy:
            try:
                from ..viewer.cuda_gl_interop import CudaGLFrameBuffer
                if CudaGLFrameBuffer.is_available(ctx):
                    v._cuda_gl_buf = CudaGLFrameBuffer(
                        v.render_width, v.render_height, ctx,
                    )
                    v._interop_enabled = True
                    print("CUDA-GL interop enabled (zero-copy display)")
            except Exception as exc:
                print(f"CUDA-GL interop not available: {exc}")
        if not v._interop_enabled:
            print("Using CPU display path (GPU->CPU->GPU roundtrip)")

        # --- Pre-render help text overlay ---
        v._render_title_overlay()
        v._render_legend_overlay()
        v._render_help_text()

        # --- Initialize minimap ---
        v._minimap_bg_extent = None
        v._compute_minimap_background()

        # --- GLFW callbacks ---
        viewer = v  # closure reference

        def _key_cb(_win, glfw_key, _scancode, action, mods):
            raw_key, key_lower = _glfw_to_key(glfw_key, mods)
            if not raw_key:
                return
            if action == glfw.PRESS:
                viewer._handle_key_press(raw_key, key_lower)
            elif action == glfw.RELEASE:
                viewer._handle_key_release(key_lower)

        def _scroll_cb(_win, _xoffset, yoffset):
            viewer._handle_scroll(yoffset)

        def _mouse_btn_cb(_win, button, action, _mods):
            xpos, ypos = glfw.get_cursor_pos(_win)
            if action == glfw.PRESS:
                viewer._handle_mouse_press(button, xpos, ypos)
            elif action == glfw.RELEASE:
                viewer._handle_mouse_release(button)

        def _cursor_cb(_win, xpos, ypos):
            viewer._handle_mouse_motion(xpos, ypos)

        def _framebuffer_size_cb(_win, fb_width, fb_height):
            if fb_width <= 0 or fb_height <= 0:
                return  # minimised
            viewer.width = fb_width
            viewer.height = fb_height
            viewer.render_width = int(fb_width * viewer.render_scale)
            viewer.render_height = int(fb_height * viewer.render_scale)
            ctx.viewport = (0, 0, fb_width, fb_height)
            # Invalidate pinned buffer so it's re-allocated at new size
            viewer._pinned_frame = None
            viewer._pinned_mem = None
            # Resize CUDA-GL PBO if interop is active
            if viewer._interop_enabled and viewer._cuda_gl_buf is not None:
                try:
                    viewer._cuda_gl_buf.resize(
                        viewer.render_width, viewer.render_height)
                    # Resize the GL texture to match the new PBO dimensions
                    viewer._interop_frame_tex.release()
                    viewer._interop_frame_tex = ctx.texture(
                        (viewer.render_width, viewer.render_height),
                        3, dtype='f4')
                    viewer._interop_frame_tex.filter = (
                        moderngl.LINEAR, moderngl.LINEAR)
                    # Resize overlay texture to match
                    viewer._overlay_tex.release()
                    viewer._overlay_tex = ctx.texture(
                        (viewer.render_width, viewer.render_height),
                        4, dtype='f4')
                    viewer._overlay_tex.filter = (
                        moderngl.LINEAR, moderngl.LINEAR)
                    viewer._overlay_uploaded = -1
                except Exception:
                    viewer._interop_enabled = False
            viewer._render_needed = True

        glfw.set_key_callback(window, _key_cb)
        glfw.set_scroll_callback(window, _scroll_cb)
        glfw.set_mouse_button_callback(window, _mouse_btn_cb)
        glfw.set_cursor_pos_callback(window, _cursor_cb)
        glfw.set_framebuffer_size_callback(window, _framebuffer_size_cb)

        print(f"\nInteractive Viewer Started")
        print(f"  Window: {v.width}x{v.height}")
        print(f"  Render: {v.render_width}x{v.render_height} ({v.render_scale:.0%})")
        print(f"  Terrain: {W}x{H}, elevation {v.elev_min:.0f}m - {v.elev_max:.0f}m")
        print(f"\nPress H for controls, X or Esc to exit\n")

        v.running = True
        v._display_frame = None
        v._frame_dirty = False
        v._render_needed = True  # Ensure first frame renders
        v._overlay_rgba = None   # RGBA overlay for interop path
        v._fps_counter = 0
        v._fps_last_time = time.monotonic()
        v._last_tick_time = time.monotonic()

        # Render the initial frame so the window isn't blank
        v._tick()

        # --- REPL thread ---
        if v._repl:
            from .viewer_proxy import ViewerProxy
            proxy = ViewerProxy(v)

            def _run_repl():
                # Auto-play tour if one was provided
                if getattr(v, '_tour', None) is not None:
                    import time as _time
                    _time.sleep(0.5)  # let first frames render
                    try:
                        proxy.tour(v._tour)
                    except Exception as exc:
                        print(f"Tour error: {exc}")

                banner = (
                    "\nrtxpy interactive REPL\n"
                    "Use `v` (the viewer proxy) to interact with the scene.\n"
                    "Examples:\n"
                    "  v.hillshade(shadows=True)\n"
                    "  v.viewshed(x=500, y=300)\n"
                    "  v.add_layer('slope', slope(v.raster).data)\n"
                    "  v.set_colormap('viridis')\n"
                    "  v.shadows = False\n"
                    "Type exit() or close the window to quit.\n"
                )
                ns = {
                    'v': proxy,
                    'viewer': proxy,
                    'np': np,
                }
                try:
                    import xarray
                    ns['xr'] = xarray
                except ImportError:
                    pass
                try:
                    from IPython.terminal.embed import InteractiveShellEmbed
                    shell = InteractiveShellEmbed(
                        banner1=banner, user_ns=ns, exit_msg='')
                    shell()
                except ImportError:
                    import code
                    code.interact(banner=banner, local=ns)
                # When REPL exits, close the viewer window
                v.running = False

            repl_thread = threading.Thread(
                target=_run_repl, daemon=True, name='rtxpy-repl')
            repl_thread.start()

        # --- Main loop ---
        try:
            while not glfw.window_should_close(window) and v.running:
                # Poll input before tick to avoid one-frame input lag
                glfw.poll_events()

                v._drain_command_queue()
                v._tick()
                v._present_if_dirty(frame_tex, prog, ctx,
                                       vao, glfw, window, moderngl)

                # Idle: yield CPU when nothing is happening (no movement,
                # no pending render).  Keeps input polling responsive at
                # ~120 Hz while avoiding a busy-wait spin.
                if not v._held_keys and not v._mouse_dragging:
                    time.sleep(0.008)
        finally:
            # --- Cleanup ---
            if v._cuda_gl_buf is not None:
                try:
                    v._cuda_gl_buf.release()
                except Exception:
                    pass
                v._cuda_gl_buf = None
            v._interop_enabled = False
            v._interop_frame_tex = None
            v._mgl_ctx = None
            v._overlay_tex.release()
            frame_tex.release()
            vbo.release()
            vao.release()
            prog.release()
            ctx.release()
            glfw.destroy_window(window)
            glfw.terminate()
            v._glfw_window = None
            # Restore terminal state (GLFW can disable echo / alter termios)
            if _saved_termios is not None:
                try:
                    import termios
                    termios.tcsetattr(sys.stdin.fileno(), termios.TCSANOW, _saved_termios)
                except (ImportError, OSError, ValueError):
                    pass
            sys.stdout.write('\033[?25h')  # show cursor
            sys.stdout.flush()

        # Clean up LOD thread pool
        if v._terrain_lod_manager is not None:
            v._terrain_lod_manager.shutdown()

        # Clean up terrain reload thread pool
        pool = v.terrain._terrain_reload_pool
        if pool is not None:
            pool.shutdown(wait=False)
            v.terrain._terrain_reload_pool = None

        # Clean up tile service
        if v._tile_service is not None:
            v._tile_service.shutdown()

        # Clean up GTFS-RT thread
        v._cleanup_gtfs_rt()

        print(f"Viewer closed after {v.frame_count} frames")
