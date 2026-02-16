"""Interactive terrain viewer using GLFW + ModernGL for display.

This module provides a simple game-engine-like render loop for
exploring terrain interactively with keyboard controls.
Uses GLFW for windowing/input and ModernGL for GPU texture display.
"""

import os
import queue
import threading
import time
import numpy as np

# On WSL2 the hardware GLX drivers often segfault. Force Mesa software
# rendering for the display path (CUDA still handles the ray tracing).
if 'microsoft' in os.uname().release.lower():
    os.environ.setdefault('LIBGL_ALWAYS_SOFTWARE', '1')
from typing import Optional, Tuple

from .rtx import RTX, has_cupy

if has_cupy:
    import cupy as cp


# ---------------------------------------------------------------------------
# OpenGL shaders for fullscreen textured quad
# ---------------------------------------------------------------------------
_QUAD_VERT = """
#version 330
in vec2 in_pos;
in vec2 in_uv;
out vec2 v_uv;
void main() {
    gl_Position = vec4(in_pos, 0.0, 1.0);
    v_uv = in_uv;
}
"""

_QUAD_FRAG = """
#version 330
uniform sampler2D frame;
in vec2 v_uv;
out vec4 fragColor;
void main() {
    fragColor = vec4(texture(frame, v_uv).rgb, 1.0);
}
"""


def _glfw_to_key(glfw_key, mods):
    """Translate a GLFW key code + modifiers to the string format used by
    _handle_key_press / _handle_key_release.

    Returns (raw_key, key_lower) matching the old matplotlib convention:
    - raw_key preserves case (uppercase if SHIFT held for letters)
    - key_lower is always lowercase
    """
    import glfw

    _SPECIAL = {
        glfw.KEY_UP: 'up', glfw.KEY_DOWN: 'down',
        glfw.KEY_LEFT: 'left', glfw.KEY_RIGHT: 'right',
        glfw.KEY_PAGE_UP: 'pageup', glfw.KEY_PAGE_DOWN: 'pagedown',
        glfw.KEY_ESCAPE: 'escape',
        glfw.KEY_EQUAL: '=', glfw.KEY_MINUS: '-',
        glfw.KEY_COMMA: ',', glfw.KEY_PERIOD: '.',
        glfw.KEY_LEFT_BRACKET: '[', glfw.KEY_RIGHT_BRACKET: ']',
        glfw.KEY_SEMICOLON: ';', glfw.KEY_APOSTROPHE: "'",
    }

    if glfw_key in _SPECIAL:
        raw = _SPECIAL[glfw_key]
        # SHIFT variants for special keys
        if mods & glfw.MOD_SHIFT:
            if raw == '=':
                raw = '+'
            elif raw == '-':
                raw = '_'  # unlikely to be used, keep '-' behaviour
            elif raw == ';':
                raw = ':'
            elif raw == "'":
                raw = '"'
        return raw, raw.lower()

    # Letter keys A-Z
    if glfw.KEY_A <= glfw_key <= glfw.KEY_Z:
        lower = chr(glfw_key - glfw.KEY_A + ord('a'))
        if mods & glfw.MOD_SHIFT:
            return lower.upper(), lower
        return lower, lower

    # Digit keys 0-9
    if glfw.KEY_0 <= glfw_key <= glfw.KEY_9:
        digit = chr(glfw_key - glfw.KEY_0 + ord('0'))
        return digit, digit

    return '', ''


# ---------------------------------------------------------------------------
# Multi-observer system — up to 8 independent observers with drone/tour
# ---------------------------------------------------------------------------

OBSERVER_COLORS = [
    (1.0, 0.2, 0.2),   # 1: red
    (0.2, 0.6, 1.0),   # 2: blue
    (0.2, 1.0, 0.3),   # 3: green
    (1.0, 0.8, 0.1),   # 4: yellow
    (1.0, 0.4, 0.0),   # 5: orange
    (0.8, 0.2, 1.0),   # 6: purple
    (0.0, 1.0, 0.9),   # 7: cyan
    (1.0, 0.5, 0.7),   # 8: pink
]


class Observer:
    """State for a single observer slot (1-8)."""

    __slots__ = (
        'slot', 'position', 'observer_elev', 'drone_mode', 'drone_placed',
        'yaw', 'pitch', 'saved_camera', 'tour_thread', 'tour_stop',
        'viewshed_enabled', 'viewshed_cache',
    )

    def __init__(self, slot, position, observer_elev=0.05):
        self.slot = slot
        self.position = position          # (x, y) world coords
        self.observer_elev = observer_elev
        self.drone_mode = 'off'           # 'off' | '3rd' | 'fpv'
        self.drone_placed = False
        self.yaw = 0.0
        self.pitch = 0.0
        self.saved_camera = None          # (position, yaw, pitch)
        self.tour_thread = None
        self.tour_stop = threading.Event()
        self.viewshed_enabled = False
        self.viewshed_cache = None

    @property
    def color(self):
        return OBSERVER_COLORS[(self.slot - 1) % len(OBSERVER_COLORS)]

    def geometry_id(self, part_idx):
        """Unique geometry ID for a drone sub-mesh, e.g. '_observer3_2'."""
        return f'_observer{self.slot}_{part_idx}'

    def is_touring(self):
        return (self.tour_thread is not None and self.tour_thread.is_alive())

    def stop_tour(self):
        self.tour_stop.set()
        if self.tour_thread is not None:
            self.tour_thread.join(timeout=2.0)
            self.tour_thread = None
        self.tour_stop.clear()


def _bilinear_terrain_z(terrain, vx, vy, psx, psy):
    """Sample terrain Z at world positions using bilinear interpolation.

    This matches the interpolation used by the triangle mesh surface,
    preventing Z mismatches between placed meshes and the rendered terrain.

    Supports both numpy and cupy arrays — the array module is chosen
    automatically based on the type of ``terrain``.
    """
    if has_cupy and isinstance(terrain, cp.ndarray):
        xp = cp
    else:
        xp = np
    H, W = terrain.shape
    cols = vx / psx
    rows = vy / psy
    cols = xp.clip(cols, 0, W - 1)
    rows = xp.clip(rows, 0, H - 1)
    x0 = xp.clip(xp.floor(cols).astype(xp.int32), 0, max(W - 2, 0))
    y0 = xp.clip(xp.floor(rows).astype(xp.int32), 0, max(H - 2, 0))
    fx = cols - x0
    fy = rows - y0
    z00 = terrain[y0, x0].astype(xp.float32)
    z10 = terrain[y0, xp.minimum(x0 + 1, W - 1)].astype(xp.float32)
    z01 = terrain[xp.minimum(y0 + 1, H - 1), x0].astype(xp.float32)
    z11 = terrain[xp.minimum(y0 + 1, H - 1),
                  xp.minimum(x0 + 1, W - 1)].astype(xp.float32)
    z00 = xp.where(xp.isnan(z00), 0.0, z00)
    z10 = xp.where(xp.isnan(z10), 0.0, z10)
    z01 = xp.where(xp.isnan(z01), 0.0, z01)
    z11 = xp.where(xp.isnan(z11), 0.0, z11)
    return (z00 * (1 - fx) * (1 - fy) +
            z10 * fx * (1 - fy) +
            z01 * (1 - fx) * fy +
            z11 * fx * fy)


class _MeshChunkManager:
    """Dynamically loads/unloads mesh chunks based on camera position.

    Manages chunk lifecycle: reads per-chunk mesh data from a zarr store,
    caches it in memory, and merges visible chunks per geometry ID into
    the RTX scene.  Only nearby chunks (within ``radius`` of the camera)
    are kept in the scene; the rest are removed.
    """

    def __init__(self, zarr_path, psx, psy):
        import zarr as _zarr
        store = _zarr.open(str(zarr_path), mode='r', use_consolidated=False)
        mg = store['meshes']

        self._elev_shape = tuple(mg.attrs['elevation_shape'])
        self._elev_chunks = tuple(mg.attrs['elevation_chunks'])
        self._chunk_h, self._chunk_w = self._elev_chunks
        self._psx = psx
        self._psy = psy
        self._n_chunk_rows = (self._elev_shape[0] + self._chunk_h - 1) // self._chunk_h
        self._n_chunk_cols = (self._elev_shape[1] + self._chunk_w - 1) // self._chunk_w

        # Per-gid colors from zarr attrs
        self._colors = {}
        self._gids = []
        for gid in mg:
            gg = mg[gid]
            if hasattr(gg, 'attrs'):
                self._colors[gid] = tuple(gg.attrs.get('color', (0.6, 0.6, 0.6)))
                self._gids.append(gid)

        # Cache: (cr, cc) -> {gid: (verts, indices)} or None if empty
        self._cache = {}
        self._visible = set()
        self._active_gids = set()  # gids currently in the RTX scene
        self.radius = 2
        self._zarr_path = zarr_path

    def _load_chunk(self, cr, cc):
        """Load a single chunk from zarr into cache."""
        if (cr, cc) in self._cache:
            return
        from .mesh_store import load_meshes_from_zarr
        meshes, _, _, curves = load_meshes_from_zarr(
            self._zarr_path, chunks=[(cr, cc)])
        # Merge curves into the same dict with a marker
        combined = {}
        for gid, data in meshes.items():
            combined[gid] = data  # (verts, indices)
        for gid, data in curves.items():
            combined[gid] = data  # (verts, widths, indices)
        self._cache[(cr, cc)] = combined

    def update(self, cam_x, cam_y, viewer):
        """Called per tick. Returns True if meshes changed."""
        # Camera world pos -> chunk coord
        cc_cam = int(cam_x / self._psx) // self._chunk_w
        cr_cam = int(cam_y / self._psy) // self._chunk_h

        # Compute visible ring clamped to grid
        cr0 = max(cr_cam - self.radius, 0)
        cr1 = min(cr_cam + self.radius, self._n_chunk_rows - 1)
        cc0 = max(cc_cam - self.radius, 0)
        cc1 = min(cc_cam + self.radius, self._n_chunk_cols - 1)

        new_visible = set()
        for cr in range(cr0, cr1 + 1):
            for cc in range(cc0, cc1 + 1):
                new_visible.add((cr, cc))

        if new_visible == self._visible:
            return False

        self._visible = new_visible

        # Load any uncached chunks
        for cr, cc in new_visible:
            self._load_chunk(cr, cc)

        # Merge visible chunks per gid
        merged = {}
        for gid in self._gids:
            all_verts = []
            all_widths = []
            all_indices = []
            vert_offset = 0
            is_curve = False
            for cr, cc in sorted(new_visible):
                chunk_data = self._cache.get((cr, cc), {})
                if gid not in chunk_data:
                    continue
                data = chunk_data[gid]
                if len(data) == 3:
                    # Curve geometry: (verts, widths, indices)
                    verts, widths, indices = data
                    is_curve = True
                    if len(indices) == 0:
                        continue
                    all_widths.append(widths)
                else:
                    verts, indices = data
                    if len(indices) == 0:
                        continue
                all_indices.append(indices + vert_offset)
                all_verts.append(verts)
                vert_offset += len(verts) // 3
            if all_verts:
                if is_curve:
                    merged[gid] = (np.concatenate(all_verts),
                                   np.concatenate(all_widths),
                                   np.concatenate(all_indices))
                else:
                    merged[gid] = (np.concatenate(all_verts),
                                   np.concatenate(all_indices))

        # Remove gids no longer present
        rtx = viewer.rtx
        accessor = viewer._accessor
        for gid in list(self._active_gids):
            if gid not in merged:
                rtx.remove_geometry(gid)
                if accessor is not None:
                    accessor._baked_meshes.pop(gid, None)
                    accessor._geometry_colors.pop(gid, None)
                self._active_gids.discard(gid)

        # Get current (possibly subsampled) terrain data
        terrain_np = viewer.raster.data
        if hasattr(terrain_np, 'get'):
            terrain_np = terrain_np.get()
        else:
            terrain_np = np.asarray(terrain_np)
        H, W = terrain_np.shape
        ve = viewer.vertical_exaggeration

        # Get full-res terrain for computing original base_z
        base_terrain = viewer._base_raster.data
        if hasattr(base_terrain, 'get'):
            base_terrain_np = base_terrain.get()
        else:
            base_terrain_np = np.asarray(base_terrain)
        base_psx = viewer._base_pixel_spacing_x
        base_psy = viewer._base_pixel_spacing_y

        # Upload terrain to GPU once (use cached if available)
        gpu_terrain = None
        gpu_base_terrain = None
        if has_cupy:
            if viewer._gpu_terrain is None:
                viewer._gpu_terrain = cp.asarray(terrain_np)
            gpu_terrain = viewer._gpu_terrain
            if viewer._gpu_base_terrain is None:
                viewer._gpu_base_terrain = cp.asarray(base_terrain_np)
            gpu_base_terrain = viewer._gpu_base_terrain

        # Add/update merged gids
        for gid, data in merged.items():
            is_curve = len(data) == 3
            if is_curve:
                verts, widths, indices = data
            else:
                verts, indices = data

            # Re-snap Z coordinates to current terrain surface + VE.
            # Meshes from zarr have Z computed from the full-res terrain.
            # When terrain is subsampled, the rendered surface differs from
            # the full-res values, so we re-anchor each vertex's height
            # offset onto the current terrain using bilinear interpolation.
            n_verts = len(verts) // 3
            use_gpu = (gpu_terrain is not None
                       and gpu_base_terrain is not None
                       and n_verts > 1000)

            if use_gpu:
                vx = cp.asarray(verts[0::3])
                vy = cp.asarray(verts[1::3])
                vz_stored = cp.asarray(verts[2::3])

                orig_base_z_gpu = _bilinear_terrain_z(
                    gpu_base_terrain, vx, vy, base_psx, base_psy)
                z_offset = vz_stored - orig_base_z_gpu

                new_base_z = _bilinear_terrain_z(
                    gpu_terrain, vx, vy,
                    viewer.pixel_spacing_x, viewer.pixel_spacing_y)

                updated_verts_gpu = cp.asarray(verts.copy())
                updated_verts_gpu[2::3] = (new_base_z + z_offset) * ve

                if is_curve:
                    rtx.add_curve_geometry(
                        gid, updated_verts_gpu,
                        cp.asarray(widths), cp.asarray(indices))
                else:
                    rtx.add_geometry(gid, updated_verts_gpu, cp.asarray(indices))
                self._active_gids.add(gid)

                if accessor is not None:
                    accessor._geometry_colors[gid] = self._colors.get(gid, (0.6, 0.6, 0.6))
                    orig_base_z_np = orig_base_z_gpu.get()
                    if is_curve:
                        accessor._baked_meshes[gid] = (
                            verts.copy(), widths.copy(), indices.copy(), orig_base_z_np)
                    else:
                        accessor._baked_meshes[gid] = (verts.copy(), indices.copy(), orig_base_z_np)
            else:
                vx = verts[0::3]
                vy = verts[1::3]
                vz_stored = verts[2::3].copy()

                orig_base_z = _bilinear_terrain_z(
                    base_terrain_np, vx, vy, base_psx, base_psy)
                z_offset = vz_stored - orig_base_z

                new_base_z = _bilinear_terrain_z(
                    terrain_np, vx, vy,
                    viewer.pixel_spacing_x, viewer.pixel_spacing_y)

                updated_verts = verts.copy()
                updated_verts[2::3] = (new_base_z + z_offset) * ve

                if is_curve:
                    rtx.add_curve_geometry(gid, updated_verts, widths, indices)
                else:
                    rtx.add_geometry(gid, updated_verts, indices)
                self._active_gids.add(gid)

                if accessor is not None:
                    accessor._geometry_colors[gid] = self._colors.get(gid, (0.6, 0.6, 0.6))
                    if is_curve:
                        accessor._baked_meshes[gid] = (
                            verts.copy(), widths.copy(), indices.copy(), orig_base_z)
                    else:
                        accessor._baked_meshes[gid] = (verts.copy(), indices.copy(), orig_base_z)

        if accessor is not None:
            accessor._geometry_colors_dirty = True

        # Refresh viewer geometry tracking (same pattern as FIRMS toggle)
        viewer._all_geometries = rtx.list_geometries()
        groups = set()
        for g in viewer._all_geometries:
            parts = g.rsplit('_', 1)
            if len(parts) == 2 and parts[1].isdigit():
                base = parts[0]
            else:
                base = g
            if base != 'terrain':
                groups.add(base)
        viewer._geometry_layer_order = ['none', 'all'] + sorted(groups)

        # Apply current visibility mode
        layer_idx = viewer._geometry_layer_idx
        if layer_idx < len(viewer._geometry_layer_order):
            layer_name = viewer._geometry_layer_order[layer_idx]
        else:
            layer_name = 'none'
            viewer._geometry_layer_idx = 0

        for geom_id in viewer._all_geometries:
            if geom_id == 'terrain':
                continue
            if layer_name == 'none':
                rtx.set_geometry_visible(geom_id, False)
            elif layer_name == 'all':
                rtx.set_geometry_visible(geom_id, True)
            else:
                parts = geom_id.rsplit('_', 1)
                base_name = parts[0] if len(parts) == 2 and parts[1].isdigit() else geom_id
                visible = (base_name == layer_name or geom_id == layer_name)
                rtx.set_geometry_visible(geom_id, visible)

        n_tris = 0
        n_segs = 0
        for g in merged:
            if len(merged[g]) == 3:
                n_segs += len(merged[g][2])
            else:
                n_tris += len(merged[g][1]) // 3
        parts = []
        if n_tris > 0:
            parts.append(f"{n_tris:,} triangles")
        if n_segs > 0:
            parts.append(f"{n_segs:,} curve segments")
        print(f"Mesh chunks: loaded {len(new_visible)} chunks, "
              f"{len(merged)} geometries ({', '.join(parts)})")
        return True


class ViewerProxy:
    """Thread-safe handle to the running InteractiveViewer.

    Exposed as ``v`` (and ``viewer``) in the REPL started by
    ``explore(repl=True)``.  Methods push callables onto a queue that
    the main GLFW thread drains each tick, so OptiX calls always
    happen on the correct thread.
    """

    def __init__(self, viewer: 'InteractiveViewer'):
        self._viewer = viewer
        self._q = viewer._command_queue

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _submit(self, fn):
        """Push *fn* to the render thread and block until it completes.

        ``fn(viewer)`` is called on the main thread.  Returns whatever
        ``fn`` returns.
        """
        result = [None]
        error = [None]
        event = threading.Event()

        def _wrapped(viewer):
            try:
                result[0] = fn(viewer)
            except Exception as exc:
                error[0] = exc
            finally:
                event.set()

        self._q.put(_wrapped)
        event.wait(timeout=30)
        if error[0] is not None:
            raise error[0]
        return result[0]

    def _submit_fire_and_forget(self, fn):
        """Push *fn* without blocking the caller."""
        self._q.put(fn)

    # ------------------------------------------------------------------
    # Read-only state (safe to access from any thread)
    # ------------------------------------------------------------------

    @property
    def raster(self):
        """The current terrain DataArray (may be subsampled)."""
        return self._viewer.raster

    @property
    def base_raster(self):
        """The full-resolution terrain DataArray."""
        return self._viewer._base_raster

    @property
    def position(self):
        """Current camera position ``(x, y, z)`` as a numpy array copy."""
        return self._viewer.position.copy()

    @property
    def yaw(self):
        return self._viewer.yaw

    @property
    def pitch(self):
        return self._viewer.pitch

    @property
    def shadows(self):
        return self._viewer.shadows

    @shadows.setter
    def shadows(self, value):
        def fn(v):
            v.shadows = bool(value)
            v._update_frame()
        self._submit(fn)

    @property
    def colormap(self):
        return self._viewer.colormap

    @colormap.setter
    def colormap(self, value):
        self.set_colormap(value)

    @property
    def vertical_exaggeration(self):
        return self._viewer.vertical_exaggeration

    @property
    def overlay_names(self):
        """List of available overlay layer names."""
        return list(self._viewer._overlay_layers.keys())

    # ------------------------------------------------------------------
    # Terrain analysis  (run on main thread → display result)
    # ------------------------------------------------------------------

    def hillshade(self, **kwargs):
        """Compute hillshade on the current terrain and show it."""
        def fn(v):
            acc = v._accessor
            if acc is None:
                print("No accessor — cannot compute hillshade")
                return
            data = acc.hillshade(**kwargs)
            _add_overlay(v, 'hillshade', data.data)
        self._submit(fn)

    def viewshed(self, x, y, observer_elev=2, **kwargs):
        """Compute viewshed and show it as an overlay."""
        def fn(v):
            acc = v._accessor
            if acc is None:
                print("No accessor — cannot compute viewshed")
                return
            data = acc.viewshed(x=x, y=y, observer_elev=observer_elev,
                                **kwargs)
            _add_overlay(v, 'viewshed', data.data)
        self._submit(fn)

    def slope(self, **kwargs):
        """Compute slope and show it as an overlay."""
        def fn(v):
            acc = v._accessor
            if acc is None:
                print("No accessor — cannot compute slope")
                return
            data = acc.slope(**kwargs)
            _add_overlay(v, 'slope', data.data)
        self._submit(fn)

    def aspect(self, **kwargs):
        """Compute aspect and show it as an overlay."""
        def fn(v):
            acc = v._accessor
            if acc is None:
                print("No accessor — cannot compute aspect")
                return
            data = acc.aspect(**kwargs)
            _add_overlay(v, 'aspect', data.data)
        self._submit(fn)

    # ------------------------------------------------------------------
    # Layer management
    # ------------------------------------------------------------------

    def add_layer(self, name, data):
        """Add (or replace) a named overlay layer and switch to it.

        Parameters
        ----------
        name : str
            Layer name shown when cycling with G.
        data : array-like
            2-D array (numpy or cupy) matching the terrain shape.
        """
        def fn(v):
            _add_overlay(v, name, data)
        self._submit(fn)

    def remove_layer(self, name):
        """Remove an overlay layer by name."""
        def fn(v):
            if name in v._overlay_layers:
                del v._overlay_layers[name]
                if name in v._base_overlay_layers:
                    del v._base_overlay_layers[name]
                v._overlay_names = list(v._overlay_layers.keys())
                v._terrain_layer_order = (
                    ['elevation'] + list(v._overlay_names))
                # Reset to elevation if we removed the active layer
                if v._terrain_layer_idx >= len(v._terrain_layer_order):
                    v._terrain_layer_idx = 0
                    v._active_overlay_data = None
                v._update_frame()
                print(f"Removed layer: {name}")
        self._submit(fn)

    def show_layer(self, name):
        """Switch the terrain coloring to a named layer (or 'elevation')."""
        def fn(v):
            if name == 'elevation':
                v._active_color_data = None
                v._active_overlay_data = None
                v._terrain_layer_idx = 0
                v._update_frame()
                print("Terrain: elevation")
                return
            if name not in v._overlay_layers:
                print(f"Unknown layer: {name}. "
                      f"Available: {list(v._overlay_layers.keys())}")
                return
            idx = v._terrain_layer_order.index(name)
            v._terrain_layer_idx = idx
            v._active_color_data = None
            v._active_overlay_data = v._overlay_layers[name]
            v._update_frame()
            print(f"Terrain: {name}")
        self._submit(fn)

    # ------------------------------------------------------------------
    # Display settings
    # ------------------------------------------------------------------

    def set_colormap(self, cmap):
        """Change the active colormap by name (e.g. 'terrain', 'viridis')."""
        def fn(v):
            v.colormap = cmap
            v._update_frame()
            print(f"Colormap: {cmap}")
        self._submit(fn)

    def set_color_stretch(self, stretch):
        """Set the color stretch ('linear', 'sqrt', 'cbrt', 'log')."""
        def fn(v):
            if stretch in v._color_stretches:
                v.color_stretch = stretch
                v._color_stretch_idx = v._color_stretches.index(stretch)
                v._update_frame()
                print(f"Color stretch: {stretch}")
            else:
                print(f"Unknown stretch: {stretch}. "
                      f"Options: {v._color_stretches}")
        self._submit(fn)

    # ------------------------------------------------------------------
    # Misc
    # ------------------------------------------------------------------

    def screenshot(self):
        """Save a screenshot.  Uses the viewer's built-in logic."""
        def fn(v):
            v._save_screenshot()
        self._submit(fn)

    def run(self, fn):
        """Execute an arbitrary callable on the render thread.

        ``fn(viewer)`` receives the ``InteractiveViewer`` instance.
        Use this for anything not covered by a convenience method.
        """
        return self._submit(fn)

    # ------------------------------------------------------------------
    # Tour / scripting
    # ------------------------------------------------------------------

    def mark(self):
        """Capture current camera state as a keyframe dict.

        Fly to desired positions manually, call ``v.mark()`` to record
        them, then assemble the results into a tour keyframe list.
        """
        from .tour import mark_camera
        kf = mark_camera(self)
        import pprint
        pprint.pprint(kf)
        return kf

    def tour(self, keyframes, fps=30, record=False, output_dir='.',
             loop=False):
        """Play a scripted camera tour.

        Parameters
        ----------
        keyframes : list of dict or str
            List of keyframe dicts, or path to a ``.py`` file that
            defines a ``tour`` variable containing the keyframe list.
        fps : int
            Target playback framerate.
        record : bool
            Save each frame as a PNG for video assembly.
        output_dir : str or Path
            Directory for recorded frames.
        loop : bool
            Repeat the tour indefinitely until the viewer closes.
        """
        from .tour import play_tour
        if isinstance(keyframes, str):
            ns = {}
            with open(keyframes) as f:
                exec(f.read(), ns)
            keyframes = ns['tour']
            if 'loop' in ns:
                loop = ns['loop']
        play_tour(self, keyframes, fps=fps, record=record,
                  output_dir=output_dir, loop=loop)

    def show_geometry(self, name):
        """Show only a specific geometry group (or ``'all'`` / ``'none'``).

        Parameters
        ----------
        name : str
            Geometry group name, ``'all'`` to show everything, or
            ``'none'`` to hide all non-terrain geometries.
        """
        def fn(v):
            if name == 'all':
                for gid in v._all_geometries:
                    v.rtx.set_geometry_visible(gid, True)
                print("Geometry: all")
            elif name == 'none':
                for gid in v._all_geometries:
                    if gid != 'terrain':
                        v.rtx.set_geometry_visible(gid, False)
                print("Geometry: none")
            else:
                visible_count = 0
                for gid in v._all_geometries:
                    parts = gid.rsplit('_', 1)
                    base = parts[0] if len(parts) == 2 and parts[1].isdigit() else gid
                    visible = (base == name or gid == name or gid == 'terrain')
                    v.rtx.set_geometry_visible(gid, visible)
                    if visible:
                        visible_count += 1
                print(f"Geometry: {name} ({visible_count} visible)")
            v._update_frame()
        self._submit(fn)

    # ------------------------------------------------------------------
    # Multi-observer API
    # ------------------------------------------------------------------

    def place_observer(self, slot, x=None, y=None):
        """Create or move an observer in *slot* (1-8).

        Defaults to the current camera position if *x*/*y* are omitted.
        """
        def fn(v):
            # Observer is defined at module level in this file
            if slot not in v._observers:
                obs = Observer(slot, position=None,
                               observer_elev=v.viewshed_observer_elev)
                v._observers[slot] = obs
            obs = v._observers[slot]
            v._place_observer_at(obs, x=x, y=y)
            v._active_observer = slot
        self._submit(fn)

    def remove_observer(self, slot):
        """Remove an observer from *slot*."""
        def fn(v):
            v._clear_observer_slot(slot)
        self._submit(fn)

    def remove_all_observers(self):
        """Kill all observers — stop tours, exit drone modes, remove all."""
        def fn(v):
            v._clear_all_observers()
        self._submit(fn)

    def select_observer(self, slot):
        """Select an observer slot for keyboard control."""
        def fn(v):
            if slot in v._observers:
                v._active_observer = slot
                print(f"Observer {slot}: selected")
            else:
                print(f"Observer {slot} does not exist")
        self._submit(fn)

    def observer_tour(self, slot, keyframes, fps=30, loop=False):
        """Run a tour on an observer's drone.

        Parameters
        ----------
        slot : int
            Observer slot (1-8). Auto-created at first keyframe if needed.
        keyframes : list of dict or str
            Keyframe list, or path to a ``.py`` file containing a ``tour``
            variable.
        fps : int
            Target playback framerate.
        loop : bool
            Repeat indefinitely until stopped.
        """
        import threading as _threading
        from .tour import play_observer_tour

        if isinstance(keyframes, str):
            ns = {}
            with open(keyframes) as f:
                exec(f.read(), ns)
            keyframes = ns['tour']
            if 'loop' in ns:
                loop = ns['loop']

        # Auto-create observer at first keyframe position if needed
        first_pos = None
        for kf in keyframes:
            if 'position' in kf:
                first_pos = kf['position']
                break

        def _setup(v):
            # Observer is defined at module level in this file
            if slot in v._observers:
                obs = v._observers[slot]
                obs.stop_tour()
            else:
                obs = Observer(slot, position=None,
                               observer_elev=v.viewshed_observer_elev)
                v._observers[slot] = obs
            if obs.position is None and first_pos is not None:
                v._place_observer_at(obs, x=first_pos[0], y=first_pos[1])
            elif obs.position is None:
                v._place_observer_at(obs)

        self._submit(_setup)

        def _tour_thread():
            play_observer_tour(self, slot, keyframes, fps=fps, loop=loop)

        obs = self._viewer._observers.get(slot)
        if obs is not None:
            obs.tour_stop.clear()
            t = _threading.Thread(target=_tour_thread, daemon=True)
            obs.tour_thread = t
            t.start()

    def stop_observer_tour(self, slot):
        """Stop a running tour on observer *slot*."""
        obs = self._viewer._observers.get(slot)
        if obs is not None:
            obs.stop_tour()
            print(f"Observer {slot} tour stopped")

    def observer_position(self, slot):
        """Get an observer's current (x, y, z) position."""
        obs = self._viewer._observers.get(slot)
        if obs is None:
            return None
        ox, oy = obs.position
        tz = self._viewer._get_terrain_z(ox, oy)
        return (ox, oy, tz + obs.observer_elev)

    def __repr__(self):
        v = self._viewer
        layers = ', '.join(v._overlay_layers.keys()) or '(none)'
        obs_info = ''
        if v._observers:
            obs_info = f", observers={list(v._observers.keys())}"
        return (f"ViewerProxy(layers=[{layers}], "
                f"colormap={v.colormap!r}, "
                f"shadows={v.shadows}{obs_info})")


def _add_overlay(viewer, name, data):
    """Add or replace an overlay layer on *viewer* and switch to it.

    Must be called on the main (render) thread.
    """
    viewer._overlay_layers[name] = data
    viewer._base_overlay_layers[name] = data
    viewer._overlay_names = list(viewer._overlay_layers.keys())
    viewer._terrain_layer_order = (
        ['elevation'] + list(viewer._overlay_names))
    idx = viewer._terrain_layer_order.index(name)
    viewer._terrain_layer_idx = idx
    viewer._active_color_data = None
    viewer._active_overlay_data = data
    viewer._update_frame()
    print(f"Terrain: {name}")


class InteractiveViewer:
    """
    Interactive terrain viewer using GLFW + ModernGL.

    Provides keyboard-controlled camera for exploring ray-traced terrain.
    Uses GLFW for windowing/input and ModernGL for GPU texture display.

    Controls
    --------
    - W/Up: Move forward
    - S/Down: Move backward
    - A/Left: Strafe left
    - D/Right: Strafe right
    - Q/Page Up: Move up
    - E/Page Down: Move down
    - I/J/K/L: Look up/left/down/right
    - Click+Drag: Pan (slippy-map style)
    - Scroll wheel: Zoom in/out (FOV)
    - +/=: Increase speed
    - -: Decrease speed
    - G: Cycle terrain color (elevation → overlays)
    - U: Cycle basemap (none → satellite → osm)
    - N: Cycle geometry layer (none → all → groups)
    - P: Jump to previous geometry in current group
    - ,/.: Decrease/increase overlay alpha (transparency)
    - O: Place observer (for viewshed) at look-at point
    - Shift+O: Cycle drone mode (off → 3rd person → FPV → off)
    - V: Toggle viewshed overlay (teal glow shows visible terrain)
    - [/]: Decrease/increase observer height
    - R: Decrease terrain resolution (coarser, up to 8x subsample)
    - Shift+R: Increase terrain resolution (finer, down to 1x)
    - Z: Decrease vertical exaggeration
    - Shift+Z: Increase vertical exaggeration
    - B: Toggle mesh type (TIN / voxel)
    - Y: Cycle color stretch (linear, sqrt, cbrt, log)
    - T: Toggle shadows
    - 0: Toggle ambient occlusion (progressive)
    - Shift+G: Cycle GI bounces (1→2→3→1)
    - Shift+D: Toggle OptiX AI Denoiser
    - C: Cycle colormap
    - Shift+F: Fetch/toggle FIRMS fire layer (7d LANDSAT 30m)
    - Shift+W: Toggle wind particle animation
    - Shift+B: Toggle GTFS-RT realtime vehicle overlay
    - F: Save screenshot
    - M: Toggle minimap overlay
    - H: Toggle help overlay
    - X: Exit

    Examples
    --------
    >>> viewer = InteractiveViewer(dem)
    >>> viewer.run()
    """

    def __init__(self, raster, width: int = 800, height: int = 600,
                 render_scale: float = 0.5, key_repeat_interval: float = 0.05,
                 rtx: 'RTX' = None,
                 pixel_spacing_x: float = 1.0, pixel_spacing_y: float = 1.0,
                 mesh_type: str = 'heightfield',
                 overlay_layers: dict = None,
                 title: str = None,
                 subsample: int = 1):
        """
        Initialize the interactive viewer.

        Parameters
        ----------
        raster : xarray.DataArray
            Terrain raster data with cupy array.
        width : int
            Display width in pixels.
        height : int
            Display height in pixels.
        render_scale : float
            Render at this fraction of display size (0.25-1.0).
            Lower values = higher FPS but lower quality.
        key_repeat_interval : float
            Minimum seconds between key repeat events (default 0.05 = 20 FPS max).
            Lower values = more responsive but more GPU load.
        rtx : RTX, optional
            Existing RTX instance with geometries (e.g., from place_mesh).
            If provided, renders the full scene including placed meshes.
        pixel_spacing_x : float, optional
            X spacing between pixels in world units (e.g., 30.0 for 30m/pixel).
            Must match the spacing used when triangulating terrain. Default 1.0.
        pixel_spacing_y : float, optional
            Y spacing between pixels in world units. Default 1.0.
        mesh_type : str, optional
            Mesh generation method: 'tin' or 'voxel'. Default is 'tin'.
        """
        if not has_cupy:
            raise ImportError(
                "cupy is required for the interactive viewer. "
                "Install with: conda install -c conda-forge cupy"
            )

        self.raster = raster
        self.rtx = rtx
        self.width = width
        self.height = height
        self.render_scale = np.clip(render_scale, 0.25, 1.0)
        self.render_width = int(width * self.render_scale)
        self.render_height = int(height * self.render_scale)

        # Pixel spacing for coordinate conversion (world coords -> pixel indices)
        self.pixel_spacing_x = pixel_spacing_x
        self.pixel_spacing_y = pixel_spacing_y
        self.mesh_type = mesh_type

        # Dynamic resolution state — preserve originals for subsampling
        self._base_raster = raster
        self._base_pixel_spacing_x = pixel_spacing_x
        self._base_pixel_spacing_y = pixel_spacing_y
        self._base_overlay_layers = overlay_layers.copy() if overlay_layers else {}
        self.subsample_factor = max(1, int(subsample))
        self._terrain_mesh_cache = {}  # (factor, mesh_type) -> (verts_base, indices, terrain_np)
        self._baked_mesh_cache = {}   # (factor, geom_id) -> (scaled_v, orig_idx)
        self._chunk_manager = None    # set by explore() when scene_zarr provided

        # GPU terrain cache for accelerated mesh Z re-snapping
        self._gpu_terrain = None       # CuPy array of current (subsampled) terrain
        self._gpu_base_terrain = None  # CuPy array of full-res terrain (stable)

        # Async readback: non-blocking stream + pinned host buffer
        self._readback_stream = cp.cuda.Stream(non_blocking=True)
        self._pinned_mem = None
        self._pinned_frame = None

        # Apply initial subsample to the working raster
        if self.subsample_factor > 1:
            f = self.subsample_factor
            raster = raster.isel(
                {raster.dims[0]: slice(None, None, f),
                 raster.dims[1]: slice(None, None, f)}
            )
            self.raster = raster
            self.pixel_spacing_x = pixel_spacing_x * f
            self.pixel_spacing_y = pixel_spacing_y * f
            if overlay_layers:
                self._overlay_layers = {
                    name: data[::f, ::f] for name, data in overlay_layers.items()
                }
                self._overlay_names = list(self._overlay_layers.keys())

        # Color stretch cycling (Y key)
        self._color_stretches = ['linear', 'sqrt', 'cbrt', 'log']
        self._color_stretch_idx = 0

        # Vertical exaggeration (Z / Shift+Z)
        self.vertical_exaggeration = 1.0

        # Overlay layers for Dataset variable cycling (G key)
        # Dict of {name: 2D cupy/numpy array} — colormap data alternatives
        self._overlay_layers = overlay_layers or {}
        self._overlay_names = list(self._overlay_layers.keys())
        self._active_color_data = None  # None = use elevation_data
        self._active_overlay_data = None  # Transparent overlay on top of base
        self._overlay_alpha = 0.7  # Overlay blending alpha (0=base only, 1=overlay only)

        # Independent terrain color cycling (G key): elevation + overlay names
        self._terrain_layer_order = ['elevation'] + list(self._overlay_names)
        self._terrain_layer_idx = 0

        # Independent basemap cycling (U key)
        self._basemap_options = ['none', 'satellite', 'osm']
        self._basemap_idx = 0

        # Title / name for display
        if title:
            self._title = title
        elif hasattr(raster, 'name') and raster.name:
            self._title = str(raster.name)
        else:
            self._title = 'rtxpy'

        # GAS layer visibility tracking
        self._all_geometries = []
        self._layer_positions = {}  # layer_name -> [(x, y, z, geometry_id), ...]
        self._current_geom_idx = 0  # Current geometry index within active layer

        # Independent geometry cycling (N key): none → all → sorted groups
        self._geometry_layer_order = ['none', 'all']

        if rtx is not None:
            self._all_geometries = rtx.list_geometries()
            # Group geometries by prefix (e.g., 'tower_0', 'tower_1' -> 'tower')
            groups = set()
            layer_geoms = {}  # layer_name -> [geometry_ids]

            for g in self._all_geometries:
                # Extract base name (before _N suffix if present)
                parts = g.rsplit('_', 1)
                if len(parts) == 2 and parts[1].isdigit():
                    base_name = parts[0]
                else:
                    base_name = g
                if base_name != 'terrain':
                    groups.add(base_name)

                if base_name not in layer_geoms:
                    layer_geoms[base_name] = []
                layer_geoms[base_name].append(g)

            self._geometry_layer_order.extend(sorted(groups))

            # Extract positions from transforms for each layer
            for layer_name, geom_ids in layer_geoms.items():
                positions = []
                for geom_id in sorted(geom_ids):  # Sort for consistent ordering
                    transform = rtx.get_geometry_transform(geom_id)
                    if transform:
                        # Position is at indices 3, 7, 11 (Tx, Ty, Tz)
                        x, y, z = transform[3], transform[7], transform[11]
                        positions.append((x, y, z, geom_id))
                self._layer_positions[layer_name] = positions

        # Camera state
        self.position = None
        self.yaw = 90.0      # Degrees, 0 = +X, 90 = +Y
        self.pitch = -15.0   # Degrees, negative = looking down
        self.move_speed = None  # Set in run() based on terrain extent
        self.look_speed = 5.0

        # Rendering settings
        self.fov = 60.0
        self._time_presets = [
            ('Morning',     135.0, 25.0),
            ('Midday',      180.0, 65.0),
            ('Afternoon',   225.0, 35.0),
            ('Golden Hour', 270.0, 12.0),
            ('Sunset',      280.0,  3.0),
        ]
        self._time_preset_idx = 2  # Afternoon (default)
        self.sun_azimuth = 225.0
        self.sun_altitude = 35.0
        self.shadows = True
        self.ambient = 0.2
        self.colormap = 'gray'
        self.colormaps = ['gray', 'terrain', 'viridis', 'plasma', 'cividis']
        self.colormap_idx = 0
        self.color_stretch = 'linear'

        # Ambient occlusion state
        self.ao_enabled = False
        self.ao_radius = None  # auto-computed from scene extent
        self.gi_intensity = 2.0  # GI bounce intensity multiplier
        self.gi_bounces = 1  # Number of GI bounces (1=single, 2-3=multi)
        self._ao_samples_per_frame = 4  # AO rays per pixel per frame
        self._ao_max_frames = 32  # stop accumulating after this many frames
        self._ao_frame_count = 0
        self._d_ao_accum = None  # GPU accumulation buffer (H, W, 3) float32
        self._prev_cam_state = None  # (position_tuple, yaw, pitch, fov) for dirty detection

        # Denoiser state
        self.denoise_enabled = False
        self._prev_cam_for_flow = None  # (pos, forward, right, up, aspect, fov_scale) from prev frame
        self._d_flow = None  # (H, W, 2) float32 motion vectors

        # Depth of field state
        self.dof_enabled = False
        self._dof_aperture = 20.0  # lens radius in scene units
        self._dof_focal_distance = 1000.0  # focal plane distance (= look_at distance)

        # Tile overlay settings
        self._tile_service = None
        self._tiles_enabled = False
        self._geometry_layer_idx = 0  # Start at 'none'

        # Viewshed settings
        self.viewshed_enabled = False
        self.viewshed_observer_elev = 0.05  # Default ~2m at 0.025× scale
        self.viewshed_target_elev = 0.0
        self.viewshed_opacity = 0.35
        self._viewshed_cache = None  # Cached viewshed result
        self._viewshed_coverage = 0.0  # Percentage of terrain visible
        self._viewshed_recalc_interval = 0.4  # Seconds between dynamic recalcs
        self._last_viewshed_time = 0.0  # Timestamp of last viewshed calc
        # Multi-observer system (up to 8 independent observers)
        self._observers = {}              # dict[int, Observer] — slot 1-8
        self._active_observer = None      # int (slot 1-8) or None
        self._shared_drone_parts = None   # loaded once from drone.glb, shared by all

        # State
        self.running = False
        self.show_help = True
        self.show_minimap = True
        self.frame_count = 0
        self._last_title = None
        self._last_subtitle = None

        # Minimap state (initialized in run() via _compute_minimap_background)
        self._minimap_background = None
        self._minimap_scale_x = 1.0
        self._minimap_scale_y = 1.0
        self._minimap_has_tiles = False
        self._minimap_rect = None  # (x0, y0, w, h) in frame coords
        self._drone_glow = False

        # Help text cache (pre-rendered RGBA numpy array via PIL)
        self._help_text_rgba = None

        # FIRMS fire layer state
        self._accessor = None         # RTX accessor for place_geojson
        self._firms_loaded = False    # Whether fire data has been fetched
        self._firms_visible = False   # Current visibility state

        # Wind particle state
        self._wind_data = None        # Raw wind dict from fetch_wind()
        self._wind_enabled = False
        self._wind_u_px = None        # (H, W) U component in pixels/tick
        self._wind_v_px = None        # (H, W) V component in pixels/tick
        self._wind_particles = None   # (N, 2) particle positions in pixel coords (row, col)
        self._wind_ages = None        # (N,) age in ticks
        self._wind_max_age = 80       # Max lifetime before respawn
        self._wind_n_particles = 10000
        self._wind_trail_len = 20     # Number of trail positions to keep
        self._wind_trails = None      # (N, trail_len, 2) ring buffer of past positions
        self._wind_speed_mult = 250.0  # Velocity exaggeration for visibility
        self._wind_min_depth = 0.0    # Min camera distance to render (set in _init_wind)
        self._wind_dot_radius = 2     # Radius of each particle dot in screen pixels
        self._wind_alpha = 0.055      # Per-pixel alpha for particle dots
        self._wind_min_visible_age = 6  # Ticks before particle becomes visible (builds trail first)
        self._wind_terrain_np = None  # Cached CPU terrain for wind Z lookup

        # GTFS-RT realtime vehicle overlay state
        self._gtfs_rt_url = None
        self._gtfs_rt_enabled = False
        self._gtfs_rt_vehicles = None       # (positions, bearings, colors) tuple
        self._gtfs_rt_poll_interval = 15.0
        self._gtfs_rt_thread = None         # daemon Thread
        self._gtfs_rt_stop = threading.Event()
        self._gtfs_rt_lock = threading.Lock()
        self._gtfs_rt_route_colors = {}     # {route_id: (r,g,b)}
        self._gtfs_rt_dot_radius = 4        # Screen pixels per vehicle dot
        self._gtfs_rt_alpha = 0.85          # Dot alpha

        # Held keys tracking for smooth simultaneous input
        self._held_keys = set()

        # GLFW window handle (set in run())
        self._glfw_window = None
        self._display_frame = None
        self._render_needed = True  # Flag: something changed, need to re-render

        # REPL command queue — background REPL thread pushes callables,
        # main loop drains and executes them on the render thread.
        self._command_queue = queue.Queue()
        self._repl = False

        # FPS tracking
        self._fps_counter = 0
        self._fps_last_time = 0.0
        self._fps_display = 0.0

        # Delta-time for frame-rate-independent movement
        self._last_tick_time = 0.0  # set in run()
        self._dt_scale = 1.0  # multiplier: actual_dt / reference_dt(0.05)

        # Mouse drag state for slippy-map panning
        self._mouse_dragging = False
        self._mouse_last_x = None
        self._mouse_last_y = None

        # Dynamic terrain loading (zarr streaming)
        self._terrain_loader = None          # callback: (lon, lat) → xr.DataArray
        self._coord_origin_x = 0.0           # lon of pixel (0,0) in current window
        self._coord_origin_y = 0.0           # lat of pixel (0,0)
        self._coord_step_x = 1.0             # lon step per pixel
        self._coord_step_y = -1.0            # lat step per pixel (negative = southward)
        self._reload_cooldown = 2.0          # min seconds between reloads
        self._last_reload_time = 0.0

        # Derive coordinate metadata from raster coords if available
        if hasattr(raster, 'x') and hasattr(raster, 'y') and len(raster.x) > 1:
            self._coord_origin_x = float(raster.x.values[0])
            self._coord_origin_y = float(raster.y.values[0])
            self._coord_step_x = float(raster.x.values[1] - raster.x.values[0])
            self._coord_step_y = float(raster.y.values[1] - raster.y.values[0])

        # Build water mask from *full-resolution* base raster (not subsampled)
        # so it can be applied to full-resolution overlay layers.
        base_data = self._base_raster.data
        if hasattr(base_data, 'get'):
            base_np = base_data.get()
        else:
            base_np = np.asarray(base_data)

        # Detect ocean-fill: global DEMs (Copernicus, SRTM) fill ocean with
        # exactly 0.0 instead of NaN/nodata.  Replace with NaN so the render
        # kernel ocean water shader activates over true ocean areas.
        ocean_fill = (base_np == 0.0) & ~np.isnan(base_np)
        n_ocean_fill = int(ocean_fill.sum())
        if n_ocean_fill > base_np.size * 0.01:
            base_np[ocean_fill] = np.nan  # local copy for water_mask below
            # Create a copy of the raster data with NaN-marked ocean
            if hasattr(self._base_raster.data, 'get'):  # cupy
                new_data = self._base_raster.data.copy()
                new_data[cp.asarray(ocean_fill)] = cp.nan
            else:
                new_data = self._base_raster.data.copy()
                new_data[ocean_fill] = np.nan
            self._base_raster = self._base_raster.copy(data=new_data)
            # Re-derive working raster from updated base
            if self.subsample_factor > 1:
                f = self.subsample_factor
                self.raster = self._base_raster.isel({
                    self._base_raster.dims[0]: slice(None, None, f),
                    self._base_raster.dims[1]: slice(None, None, f)
                })
            else:
                self.raster = self._base_raster

        floor_val = float(np.nanmin(base_np))
        floor_max = float(np.nanmax(base_np))
        eps = (floor_max - floor_val) * 1e-4 if floor_max > floor_val else 1e-6
        self._water_mask = (base_np <= floor_val + eps) | np.isnan(base_np)

        # Get terrain info (after ocean-fill → NaN replacement)
        H, W = self.raster.shape
        terrain_data = self.raster.data
        if hasattr(terrain_data, 'get'):
            terrain_np = terrain_data.get()
        else:
            terrain_np = np.asarray(terrain_data)

        self.terrain_shape = (H, W)
        self.elev_min = float(np.nanmin(terrain_np))
        self.elev_max = float(np.nanmax(terrain_np))
        self.elev_mean = float(np.nanmean(terrain_np))

        # Compute land-only elevation range for coloring (excludes water)
        land_pixels = base_np[~self._water_mask]
        if land_pixels.size > 0:
            self._land_color_range = (float(np.nanmin(land_pixels)),
                                      float(np.nanmax(land_pixels)))
        else:
            self._land_color_range = None

        # Apply water mask to overlay layers (set water pixels to NaN so
        # nanmin/nanmax in the render pipeline ignores them for color range)
        if self._water_mask.any():
            for name in list(self._base_overlay_layers.keys()):
                data = self._base_overlay_layers[name]
                if hasattr(data, 'get'):
                    # cupy array — upload mask, apply on GPU
                    mask_gpu = cp.asarray(self._water_mask)
                    data = data.copy()
                    data[mask_gpu] = cp.nan
                else:
                    data = np.array(data, dtype=np.float32)
                    data[self._water_mask] = np.nan
                self._base_overlay_layers[name] = data
            # Rebuild working overlays from masked base
            if self.subsample_factor > 1:
                f = self.subsample_factor
                self._overlay_layers = {
                    name: data[::f, ::f]
                    for name, data in self._base_overlay_layers.items()
                }
            else:
                self._overlay_layers = dict(self._base_overlay_layers)

        # Build terrain geometry if RTX exists but has no terrain.
        # Without this, render() falls into the auto-VE / prepare_mesh path
        # which computes vertical_exaggeration from pixel dimensions (not world
        # units), producing wrong results when pixel_spacing != 1.
        if rtx is not None and not rtx.has_geometry('terrain'):
            from . import mesh as mesh_mod
            if mesh_type == 'heightfield':
                rtx.add_heightfield_geometry(
                    'terrain', terrain_np, H, W,
                    spacing_x=self.pixel_spacing_x,
                    spacing_y=self.pixel_spacing_y,
                    ve=1.0,
                )
                cache_key = (self.subsample_factor, mesh_type)
                self._terrain_mesh_cache[cache_key] = (
                    None, None, terrain_np.copy(),
                )
            else:
                if mesh_type == 'voxel':
                    nv = H * W * 8
                    nt = H * W * 12
                    verts = np.zeros(nv * 3, dtype=np.float32)
                    idxs = np.zeros(nt * 3, dtype=np.int32)
                    base_elev = float(np.nanmin(terrain_np))
                    mesh_mod.voxelate_terrain(verts, idxs, raster, scale=1.0,
                                              base_elevation=base_elev)
                else:
                    nv = H * W
                    nt = (H - 1) * (W - 1) * 2
                    verts = np.zeros(nv * 3, dtype=np.float32)
                    idxs = np.zeros(nt * 3, dtype=np.int32)
                    mesh_mod.triangulate_terrain(verts, idxs, raster, scale=1.0)

                if self.pixel_spacing_x != 1.0 or self.pixel_spacing_y != 1.0:
                    verts[0::3] *= self.pixel_spacing_x
                    verts[1::3] *= self.pixel_spacing_y

                cache_key = (self.subsample_factor, mesh_type)
                self._terrain_mesh_cache[cache_key] = (
                    verts.copy(), idxs.copy(), terrain_np.copy(),
                )

                rtx.add_geometry('terrain', verts, idxs)

    def _get_front(self):
        """Get the forward direction vector."""
        yaw_rad = np.radians(self.yaw)
        pitch_rad = np.radians(self.pitch)
        return np.array([
            np.cos(yaw_rad) * np.cos(pitch_rad),
            np.sin(yaw_rad) * np.cos(pitch_rad),
            np.sin(pitch_rad)
        ], dtype=np.float32)

    def _get_right(self):
        """Get the right direction vector."""
        front = self._get_front()
        world_up = np.array([0, 0, 1], dtype=np.float32)
        right = np.cross(world_up, front)
        return right / (np.linalg.norm(right) + 1e-8)

    def _get_look_at(self):
        """Get the current look-at point."""
        return self.position + self._get_front() * 1000.0

    def _build_title(self):
        """Build a rich status title string for the viewer window."""
        H, W = self.terrain_shape
        parts = [self._title]

        # Resolution
        res = f"{W}\u00d7{H}"
        if self.subsample_factor > 1:
            res += f" ({self.subsample_factor}\u00d7 sub)"
        parts.append(res)

        # Mesh type
        parts.append(self.mesh_type.upper())

        # Terrain color layer
        terrain_name = self._terrain_layer_order[self._terrain_layer_idx]
        if terrain_name != 'elevation' and terrain_name in self._overlay_layers:
            alpha_pct = int(self._overlay_alpha * 100)
            parts.append(f"{terrain_name} ({alpha_pct}%)")
        else:
            parts.append('elevation')

        # Basemap
        basemap = self._basemap_options[self._basemap_idx]
        if basemap != 'none':
            parts.append(f"tiles:{basemap}")

        # Geometry layer
        geom_layer = self._geometry_layer_order[self._geometry_layer_idx]
        if geom_layer != 'none':
            parts.append(geom_layer)

        # Colormap + stretch
        cmap_str = self.colormap
        if self.color_stretch != 'linear':
            cmap_str += f" ({self.color_stretch})"
        parts.append(cmap_str)

        # Vertical exaggeration (only if != 1.0)
        if abs(self.vertical_exaggeration - 1.0) > 0.01:
            parts.append(f"VE {self.vertical_exaggeration:.1f}\u00d7")

        # Shadows
        if not self.shadows:
            parts.append('no shadows')

        # Ambient occlusion
        if self.ao_enabled:
            total = self._ao_frame_count * self._ao_samples_per_frame
            cap = self._ao_max_frames * self._ao_samples_per_frame
            parts.append(f'AO {total}/{cap}')

        # Denoiser
        if self.denoise_enabled:
            parts.append('DENOISE')

        # Wind
        if self._wind_enabled:
            parts.append('wind')

        # Active observer drone mode
        active_obs = (self._observers.get(self._active_observer)
                      if self._active_observer else None)
        if active_obs is not None:
            if active_obs.drone_mode == 'fpv':
                parts.append(f'OBS{active_obs.slot} FPV')
            elif active_obs.drone_mode == '3rd':
                parts.append(f'OBS{active_obs.slot} 3RD')

        return '  \u2502  '.join(parts)

    def _compute_minimap_background(self):
        """Compute a stylised RGBA minimap image.

        Downsamples terrain to max 200px, computes hillshade for land,
        masks water/NaN as dark ocean, and applies a warm-toned smoky
        colour scheme so the minimap pops against the dark viewer chrome.
        """
        H, W = self.terrain_shape
        terrain_data = self.raster.data
        if hasattr(terrain_data, 'get'):
            terrain_np = terrain_data.get()
        else:
            terrain_np = np.asarray(terrain_data)

        # Downsample to max 200px on longest side
        max_dim = 200
        longest = max(H, W)
        if longest > max_dim:
            scale = max_dim / longest
            new_h = max(1, int(H * scale))
            new_w = max(1, int(W * scale))
            y_idx = np.linspace(0, H - 1, new_h).astype(int)
            x_idx = np.linspace(0, W - 1, new_w).astype(int)
            terrain_small = terrain_np[np.ix_(y_idx, x_idx)]
        else:
            terrain_small = terrain_np.copy()
            new_h, new_w = H, W

        # Water mask: NaN or <= 0
        water = np.isnan(terrain_small) | (terrain_small <= 0)

        # Fill NaNs for gradient computation
        if water.any():
            med = np.nanmedian(terrain_small)
            terrain_small = terrain_small.copy()
            terrain_small[water] = med if np.isfinite(med) else 0.0

        # Hillshade (sun from upper-left)
        dy, dx = np.gradient(terrain_small)
        az_rad = np.radians(315)
        alt_rad = np.radians(45)
        slp = np.sqrt(dx**2 + dy**2)
        asp = np.arctan2(-dy, dx)
        shaded = (np.sin(alt_rad) * np.cos(np.arctan(slp)) +
                  np.cos(alt_rad) * np.sin(np.arctan(slp)) *
                  np.cos(az_rad - asp))
        shaded = np.clip(shaded, 0, 1)

        # Elevation tint: normalise to [0,1] for colour ramp
        emin = np.nanmin(terrain_small[~water]) if (~water).any() else 0
        emax = np.nanmax(terrain_small[~water]) if (~water).any() else 1
        erng = emax - emin if emax > emin else 1.0
        elev_norm = np.clip((terrain_small - emin) / erng, 0, 1)

        # Build RGBA image
        rgba = np.zeros((new_h, new_w, 4), dtype=np.float32)

        # Land: smoky warm tones — blend hillshade with elevation tint
        # Low elevation → dark olive/brown, high → pale sand/cream
        lo = np.array([0.18, 0.20, 0.14])  # dark olive
        hi = np.array([0.85, 0.80, 0.70])  # warm cream
        for c in range(3):
            tint = lo[c] + (hi[c] - lo[c]) * elev_norm
            # Mix 60 % hillshade + 40 % elevation tint for a smoky look
            rgba[:, :, c] = shaded * 0.6 * tint + tint * 0.4
        rgba[:, :, 3] = 1.0  # fully opaque land

        # Water: dark blue-black, semi-transparent
        rgba[water, 0] = 0.08
        rgba[water, 1] = 0.10
        rgba[water, 2] = 0.18
        rgba[water, 3] = 0.7

        rgba[:, :, :3] = np.clip(rgba[:, :, :3], 0, 1)

        # Blend satellite imagery if tile service has fetched tiles
        if (self._tile_service is not None
                and getattr(self._tile_service, '_fetched', None)):
            cpu_tex = getattr(self._tile_service, '_rgb_texture', None)
            if cpu_tex is not None and cpu_tex.shape[0] == H and cpu_tex.shape[1] == W:
                y_idx_t = np.linspace(0, H - 1, new_h).astype(int)
                x_idx_t = np.linspace(0, W - 1, new_w).astype(int)
                sat_small = cpu_tex[np.ix_(y_idx_t, x_idx_t)]  # (new_h, new_w, 3)
                # Only blend where satellite has actual data (not all-black)
                has_coverage = sat_small.max(axis=2) > 0.01
                blended = np.zeros_like(rgba[:, :, :3])
                for c in range(3):
                    blended[:, :, c] = sat_small[:, :, c] * 0.7 + rgba[:, :, c] * 0.3
                for c in range(3):
                    rgba[:, :, c] = np.where(has_coverage, blended[:, :, c], rgba[:, :, c])
                self._minimap_has_tiles = True

        self._minimap_background = rgba
        self._minimap_scale_x = new_w / W
        self._minimap_scale_y = new_h / H

    def _rebuild_at_resolution(self, factor):
        """Rebuild terrain mesh at a different subsample factor.

        Subsamples the original raster by ``factor`` (1 = full res, 2 = half,
        etc.), rebuilds the terrain geometry, re-snaps any placed meshes to the
        new surface, and refreshes the minimap.

        Parameters
        ----------
        factor : int
            Subsample factor (1, 2, 4, or 8).
        """
        from . import mesh as mesh_mod

        self.subsample_factor = factor
        base = self._base_raster

        # 1. Subsample the raster
        if factor > 1:
            sub = base.isel(
                {base.dims[0]: slice(None, None, factor),
                 base.dims[1]: slice(None, None, factor)}
            )
        else:
            sub = base

        self.raster = sub
        self._wind_terrain_np = None  # invalidate cached terrain
        H, W = sub.shape
        self.terrain_shape = (H, W)

        # 2. Update pixel spacing
        self.pixel_spacing_x = self._base_pixel_spacing_x * factor
        self.pixel_spacing_y = self._base_pixel_spacing_y * factor

        # 3. Build or retrieve cached terrain mesh
        ve = self.vertical_exaggeration
        cache_key = (factor, self.mesh_type)

        if self.mesh_type == 'heightfield':
            # Heightfield path: no triangle mesh needed
            if cache_key in self._terrain_mesh_cache:
                _, _, terrain_np = self._terrain_mesh_cache[cache_key]
            else:
                terrain_data = sub.data
                if hasattr(terrain_data, 'get'):
                    terrain_np = terrain_data.get()
                else:
                    terrain_np = np.asarray(terrain_data)
                self._terrain_mesh_cache[cache_key] = (
                    None, None, terrain_np.copy(),
                )

            if self.rtx is not None:
                self.rtx.add_heightfield_geometry(
                    'terrain', terrain_np, H, W,
                    spacing_x=self.pixel_spacing_x,
                    spacing_y=self.pixel_spacing_y,
                    ve=ve,
                )
        else:
            if cache_key in self._terrain_mesh_cache:
                # Cache hit — reuse pre-built mesh (stored at scale=1.0)
                verts_base, indices, terrain_np = self._terrain_mesh_cache[cache_key]
                vertices = verts_base.copy()
                if ve != 1.0:
                    vertices[2::3] *= ve
            else:
                # Cache miss — build mesh at scale=1.0 and cache it
                terrain_data = sub.data
                if hasattr(terrain_data, 'get'):
                    terrain_np = terrain_data.get()
                else:
                    terrain_np = np.asarray(terrain_data)

                if self.mesh_type == 'voxel':
                    num_verts = H * W * 8
                    num_tris = H * W * 12
                    vertices = np.zeros(num_verts * 3, dtype=np.float32)
                    indices = np.zeros(num_tris * 3, dtype=np.int32)
                    base_elev = float(np.nanmin(terrain_np))
                    mesh_mod.voxelate_terrain(vertices, indices, sub, scale=1.0,
                                              base_elevation=base_elev)
                else:
                    num_verts = H * W
                    num_tris = (H - 1) * (W - 1) * 2
                    vertices = np.zeros(num_verts * 3, dtype=np.float32)
                    indices = np.zeros(num_tris * 3, dtype=np.int32)
                    mesh_mod.triangulate_terrain(vertices, indices, sub, scale=1.0)

                # Scale x,y to world units
                if self.pixel_spacing_x != 1.0 or self.pixel_spacing_y != 1.0:
                    vertices[0::3] *= self.pixel_spacing_x
                    vertices[1::3] *= self.pixel_spacing_y

                # Store in cache (scale=1.0, x/y already scaled)
                self._terrain_mesh_cache[cache_key] = (
                    vertices.copy(), indices.copy(), terrain_np.copy()
                )

                # Apply VE to this copy
                if ve != 1.0:
                    vertices[2::3] *= ve

            # 4. Replace terrain geometry (add_geometry overwrites existing key
            #    in-place, preserving dict insertion order and instance IDs)
            if self.rtx is not None:
                self.rtx.add_geometry('terrain', vertices, indices)

        self.elev_min = float(np.nanmin(terrain_np)) * ve
        self.elev_max = float(np.nanmax(terrain_np)) * ve
        self.elev_mean = float(np.nanmean(terrain_np)) * ve

        # Update land-only color range with VE
        f = self.subsample_factor
        wm = self._water_mask[::f, ::f] if f > 1 else self._water_mask
        land_pixels = terrain_np[~wm[:terrain_np.shape[0], :terrain_np.shape[1]]]
        if land_pixels.size > 0:
            self._land_color_range = (float(np.nanmin(land_pixels)) * ve,
                                      float(np.nanmax(land_pixels)) * ve)

        # 5. Subsample overlay layers
        if self._base_overlay_layers:
            self._overlay_layers = {}
            for name, data in self._base_overlay_layers.items():
                if factor > 1:
                    self._overlay_layers[name] = data[::factor, ::factor]
                else:
                    self._overlay_layers[name] = data
            self._overlay_names = list(self._overlay_layers.keys())
            # Rebuild terrain layer order with new overlay names
            self._terrain_layer_order = ['elevation'] + list(self._overlay_names)
            if self._terrain_layer_idx >= len(self._terrain_layer_order):
                self._terrain_layer_idx = 0
            # Reset active overlay data if an overlay is selected
            terrain_name = self._terrain_layer_order[self._terrain_layer_idx]
            if terrain_name != 'elevation' and terrain_name in self._overlay_layers:
                self._active_overlay_data = self._overlay_layers[terrain_name]

        # 6. Invalidate chunk manager cache (meshes need new Z coords)
        if self._chunk_manager is not None:
            # Clear chunk cache and baked mesh entries for chunk-loaded geometries
            for gid in list(self._chunk_manager._active_gids):
                if hasattr(self, '_baked_meshes'):
                    self._baked_meshes.pop(gid, None)
                if self._accessor is not None:
                    self._accessor._baked_meshes.pop(gid, None)
            self._chunk_manager._cache.clear()
            self._chunk_manager._visible.clear()
            self._chunk_manager._active_gids.clear()
            # Force immediate reload at new resolution
            if hasattr(self, 'position'):
                self._chunk_manager.update(self.position[0], self.position[1], self)

        # 7. Re-snap placed meshes to new terrain surface
        # Invalidate GPU terrain cache (terrain changed) and upload once
        self._gpu_terrain = None
        if self.rtx is not None:
            gpu_terrain = None
            if has_cupy:
                gpu_terrain = cp.asarray(terrain_np)
                self._gpu_terrain = gpu_terrain
            for geom_id in self.rtx.list_geometries():
                if geom_id == 'terrain':
                    continue
                # Baked meshes — re-snap Z to new terrain surface + VE
                if hasattr(self, '_baked_meshes') and geom_id in self._baked_meshes:
                    baked = self._baked_meshes[geom_id]
                    is_curve = (len(baked) == 4)
                    baked_key = (factor, geom_id)
                    if baked_key in self._baked_mesh_cache:
                        cached = self._baked_mesh_cache[baked_key]
                        if is_curve:
                            scaled_v, orig_w, orig_idx = cached
                            self.rtx.add_curve_geometry(
                                geom_id, scaled_v, orig_w, orig_idx)
                        else:
                            scaled_v, orig_idx = cached
                            self.rtx.add_geometry(geom_id, scaled_v, orig_idx)
                    else:
                        if is_curve:
                            orig_v, orig_w, orig_idx, orig_base_z = baked
                        elif len(baked) == 3:
                            orig_v, orig_idx, orig_base_z = baked
                        else:
                            orig_v, orig_idx = baked
                            orig_base_z = None

                        n_verts = len(orig_v) // 3
                        use_gpu = (gpu_terrain is not None
                                   and orig_base_z is not None
                                   and n_verts > 1000)

                        if use_gpu:
                            vx = cp.asarray(orig_v[0::3])
                            vy = cp.asarray(orig_v[1::3])
                            new_base_z = _bilinear_terrain_z(
                                gpu_terrain, vx, vy,
                                self.pixel_spacing_x, self.pixel_spacing_y)
                            z_offset = cp.asarray(orig_v[2::3]) - cp.asarray(orig_base_z)
                            new_z = (new_base_z + z_offset) * ve
                            scaled_v_gpu = cp.asarray(orig_v.copy())
                            scaled_v_gpu[2::3] = new_z
                            if is_curve:
                                self._baked_mesh_cache[baked_key] = (
                                    scaled_v_gpu.get().copy(), orig_w, orig_idx)
                                self.rtx.add_curve_geometry(
                                    geom_id, scaled_v_gpu,
                                    cp.asarray(orig_w),
                                    cp.asarray(orig_idx))
                            else:
                                self._baked_mesh_cache[baked_key] = (
                                    scaled_v_gpu.get().copy(), orig_idx)
                                self.rtx.add_geometry(geom_id, scaled_v_gpu,
                                                      cp.asarray(orig_idx))
                        else:
                            scaled_v = orig_v.copy()
                            if orig_base_z is not None:
                                vx = orig_v[0::3]
                                vy = orig_v[1::3]
                                new_base_z = _bilinear_terrain_z(
                                    terrain_np, vx, vy,
                                    self.pixel_spacing_x, self.pixel_spacing_y)
                                z_offset = orig_v[2::3] - orig_base_z
                                scaled_v[2::3] = (new_base_z + z_offset) * ve
                            else:
                                scaled_v[2::3] *= ve
                            if is_curve:
                                self._baked_mesh_cache[baked_key] = (
                                    scaled_v.copy(), orig_w, orig_idx)
                                self.rtx.add_curve_geometry(
                                    geom_id, scaled_v, orig_w, orig_idx)
                            else:
                                self._baked_mesh_cache[baked_key] = (
                                    scaled_v.copy(), orig_idx)
                                self.rtx.add_geometry(geom_id, scaled_v, orig_idx)
                    continue
                # Instanced meshes — update transform Z from terrain
                transform = self.rtx.get_geometry_transform(geom_id)
                if transform is None:
                    continue
                wx, wy = transform[3], transform[7]
                z = float(_bilinear_terrain_z(
                    terrain_np,
                    np.array([wx], dtype=np.float32),
                    np.array([wy], dtype=np.float32),
                    self.pixel_spacing_x, self.pixel_spacing_y)[0]) * ve
                transform[11] = z
                self.rtx.update_transform(geom_id, transform)

        # 8. Re-snap all observer drones to new terrain
        for obs in self._observers.values():
            if obs.drone_placed and obs.position is not None:
                self._update_observer_drone_for(obs)

        # 9. Recompute minimap
        self._compute_minimap_background()

        # 10. Clear viewshed cache (no longer matches terrain)
        self._viewshed_cache = None
        for obs in self._observers.values():
            obs.viewshed_cache = None
            if obs.viewshed_enabled:
                obs.viewshed_enabled = False
        if self.viewshed_enabled:
            self.viewshed_enabled = False
            print("  Viewshed disabled (terrain changed). Press V to recalculate.")

        print(f"Resolution: {W}x{H} (subsample {factor}x)")
        self._update_frame()

    def _rebuild_vertical_exaggeration(self, ve):
        """Rebuild terrain mesh with a new vertical exaggeration factor.

        Parameters
        ----------
        ve : float
            Vertical exaggeration multiplier applied to elevation values.
        """
        from . import mesh as mesh_mod

        self.vertical_exaggeration = ve
        H, W = self.terrain_shape

        # Use cached mesh if available, otherwise build and cache
        cache_key = (self.subsample_factor, self.mesh_type)

        if self.mesh_type == 'heightfield':
            # Heightfield path: rebuild GAS with new VE
            if cache_key in self._terrain_mesh_cache:
                _, _, terrain_np = self._terrain_mesh_cache[cache_key]
            else:
                terrain_data = self.raster.data
                if hasattr(terrain_data, 'get'):
                    terrain_np = terrain_data.get()
                else:
                    terrain_np = np.asarray(terrain_data)
                self._terrain_mesh_cache[cache_key] = (
                    None, None, terrain_np.copy(),
                )

            if self.rtx is not None:
                self.rtx.add_heightfield_geometry(
                    'terrain', terrain_np, H, W,
                    spacing_x=self.pixel_spacing_x,
                    spacing_y=self.pixel_spacing_y,
                    ve=ve,
                )
        else:
            if cache_key in self._terrain_mesh_cache:
                verts_base, indices, terrain_np = self._terrain_mesh_cache[cache_key]
                vertices = verts_base.copy()
                if ve != 1.0:
                    vertices[2::3] *= ve
            else:
                terrain_data = self.raster.data
                if hasattr(terrain_data, 'get'):
                    terrain_np = terrain_data.get()
                else:
                    terrain_np = np.asarray(terrain_data)

                if self.mesh_type == 'voxel':
                    nv = H * W * 8
                    nt = H * W * 12
                    vertices = np.zeros(nv * 3, dtype=np.float32)
                    indices = np.zeros(nt * 3, dtype=np.int32)
                    base_elev = float(np.nanmin(terrain_np))
                    mesh_mod.voxelate_terrain(vertices, indices, self.raster,
                                              scale=1.0, base_elevation=base_elev)
                else:
                    nv = H * W
                    nt = (H - 1) * (W - 1) * 2
                    vertices = np.zeros(nv * 3, dtype=np.float32)
                    indices = np.zeros(nt * 3, dtype=np.int32)
                    mesh_mod.triangulate_terrain(vertices, indices, self.raster,
                                                 scale=1.0)

                if self.pixel_spacing_x != 1.0 or self.pixel_spacing_y != 1.0:
                    vertices[0::3] *= self.pixel_spacing_x
                    vertices[1::3] *= self.pixel_spacing_y

                self._terrain_mesh_cache[cache_key] = (
                    vertices.copy(), indices.copy(), terrain_np.copy()
                )

                if ve != 1.0:
                    vertices[2::3] *= ve

            # Replace terrain geometry (preserves dict insertion order)
            if self.rtx is not None:
                self.rtx.add_geometry('terrain', vertices, indices)

        # Update elevation stats (scaled)
        self.elev_min = float(np.nanmin(terrain_np)) * ve
        self.elev_max = float(np.nanmax(terrain_np)) * ve
        self.elev_mean = float(np.nanmean(terrain_np)) * ve

        # Update land-only color range with VE
        f = self.subsample_factor
        wm = self._water_mask[::f, ::f] if f > 1 else self._water_mask
        land_pixels = terrain_np[~wm[:terrain_np.shape[0], :terrain_np.shape[1]]]
        if land_pixels.size > 0:
            self._land_color_range = (float(np.nanmin(land_pixels)) * ve,
                                      float(np.nanmax(land_pixels)) * ve)

        # Re-snap placed meshes to scaled terrain
        # Invalidate GPU terrain cache (VE changed terrain Z) and upload once
        self._gpu_terrain = None
        if self.rtx is not None:
            gpu_terrain = None
            if has_cupy:
                gpu_terrain = cp.asarray(terrain_np)
                self._gpu_terrain = gpu_terrain
            for geom_id in self.rtx.list_geometries():
                if geom_id == 'terrain':
                    continue
                # Baked meshes (merged buildings/curves) — re-snap Z to terrain + VE
                if hasattr(self, '_baked_meshes') and geom_id in self._baked_meshes:
                    baked = self._baked_meshes[geom_id]
                    is_curve = (len(baked) == 4)
                    if is_curve:
                        orig_v, orig_w, orig_idx, orig_base_z = baked
                    elif len(baked) == 3:
                        orig_v, orig_idx, orig_base_z = baked
                    else:
                        orig_v, orig_idx = baked
                        orig_base_z = None

                    n_verts = len(orig_v) // 3
                    use_gpu = (gpu_terrain is not None
                               and orig_base_z is not None
                               and n_verts > 1000)

                    if use_gpu:
                        vx = cp.asarray(orig_v[0::3])
                        vy = cp.asarray(orig_v[1::3])
                        cur_base_z = _bilinear_terrain_z(
                            gpu_terrain, vx, vy,
                            self.pixel_spacing_x, self.pixel_spacing_y)
                        z_offset = cp.asarray(orig_v[2::3]) - cp.asarray(orig_base_z)
                        new_z = (cur_base_z + z_offset) * ve
                        scaled_v_gpu = cp.asarray(orig_v.copy())
                        scaled_v_gpu[2::3] = new_z
                        if is_curve:
                            self.rtx.add_curve_geometry(
                                geom_id, scaled_v_gpu,
                                cp.asarray(orig_w),
                                cp.asarray(orig_idx))
                        else:
                            self.rtx.add_geometry(geom_id, scaled_v_gpu,
                                                  cp.asarray(orig_idx))
                    else:
                        scaled_v = orig_v.copy()
                        if orig_base_z is not None:
                            vx = orig_v[0::3]
                            vy = orig_v[1::3]
                            cur_base_z = _bilinear_terrain_z(
                                terrain_np, vx, vy,
                                self.pixel_spacing_x, self.pixel_spacing_y)
                            z_offset = orig_v[2::3] - orig_base_z
                            scaled_v[2::3] = (cur_base_z + z_offset) * ve
                        else:
                            scaled_v[2::3] *= ve
                        if is_curve:
                            self.rtx.add_curve_geometry(
                                geom_id, scaled_v, orig_w, orig_idx)
                        else:
                            self.rtx.add_geometry(geom_id, scaled_v, orig_idx)
                    continue
                # Instanced meshes — update transform Z from terrain
                transform = self.rtx.get_geometry_transform(geom_id)
                if transform is None:
                    continue
                wx, wy = transform[3], transform[7]
                z = float(_bilinear_terrain_z(
                    terrain_np,
                    np.array([wx], dtype=np.float32),
                    np.array([wy], dtype=np.float32),
                    self.pixel_spacing_x, self.pixel_spacing_y)[0]) * ve
                transform[11] = z
                self.rtx.update_transform(geom_id, transform)

        # Re-snap all observer drones to updated terrain
        for obs in self._observers.values():
            if obs.drone_placed and obs.position is not None:
                self._update_observer_drone_for(obs)

        # Clear viewshed cache
        self._viewshed_cache = None
        for obs in self._observers.values():
            obs.viewshed_cache = None
            if obs.viewshed_enabled:
                obs.viewshed_enabled = False
        if self.viewshed_enabled:
            self.viewshed_enabled = False
            print("  Viewshed disabled (terrain changed). Press V to recalculate.")

        print(f"Vertical exaggeration: {ve:.2f}x")
        self._update_frame()

    def _project_corner_to_terrain(self, nx, ny, cam_pos, forward, right,
                                     up_cam, fov_scale, aspect, terrain_z):
        """Project an NDC screen corner onto the terrain z-plane.

        Parameters
        ----------
        nx, ny : float
            Normalised device coords (-1..1), where (-1,-1) = bottom-left.
        cam_pos : ndarray (3,)
            Camera position in world space (already VE-scaled Z).
        forward, right, up_cam : ndarray (3,)
            Camera basis vectors.
        fov_scale : float
            tan(fov/2).
        aspect : float
            Width / height.
        terrain_z : float
            Z plane to intersect (mean_elev * VE).

        Returns
        -------
        (world_x, world_y) or None if ray doesn't hit ground.
        """
        ray_dir = forward + nx * fov_scale * aspect * right + ny * fov_scale * up_cam
        norm = np.linalg.norm(ray_dir)
        if norm < 1e-8:
            return None
        ray_dir /= norm

        # Intersect with z = terrain_z plane
        if abs(ray_dir[2]) < 1e-8:
            # Ray parallel to ground — project far forward
            t = 1e5
        else:
            t = (terrain_z - cam_pos[2]) / ray_dir[2]

        if t < 0:
            # Looking up past horizon — project far forward along horizontal
            horiz = np.array([ray_dir[0], ray_dir[1], 0.0])
            hn = np.linalg.norm(horiz)
            if hn < 1e-8:
                return None
            horiz /= hn
            far_dist = max(self.terrain_shape[0] * self.pixel_spacing_y,
                           self.terrain_shape[1] * self.pixel_spacing_x)
            return (cam_pos[0] + horiz[0] * far_dist,
                    cam_pos[1] + horiz[1] * far_dist)

        hit = cam_pos + ray_dir * t
        return (float(hit[0]), float(hit[1]))

    def _blit_minimap_on_frame(self, img):
        """Composite minimap overlay onto the rendered frame (numpy blit).

        Draws the minimap background with rounded corners and drop shadow,
        terrain footprint quad, camera dot, direction line, and observer dots
        directly onto the frame array in the bottom-right.

        Parameters
        ----------
        img : ndarray, shape (H, W, 3), float32 0-1
            Rendered frame to composite onto. Modified in-place.
        """
        if self._minimap_background is None or not self.show_minimap:
            return

        # Lazy re-check: pick up satellite tiles once they arrive
        if (not self._minimap_has_tiles
                and self._tile_service is not None
                and getattr(self._tile_service, '_fetched', None)):
            self._compute_minimap_background()

        mm_bg = self._minimap_background  # (mm_h, mm_w, 4) RGBA float32
        mm_h, mm_w = mm_bg.shape[:2]
        fh, fw = img.shape[:2]

        # Size the minimap to ~20% of frame width
        target_w = max(40, int(fw * 0.2))
        scale = target_w / mm_w
        target_h = max(20, int(mm_h * scale))
        target_w = min(target_w, fw - 8)
        target_h = min(target_h, fh - 8)

        # Nearest-neighbour resize
        y_idx = np.linspace(0, mm_h - 1, target_h).astype(int)
        x_idx = np.linspace(0, mm_w - 1, target_w).astype(int)
        bg_resized = mm_bg[np.ix_(y_idx, x_idx)].copy()  # (th, tw, 4)

        # --- Rounded corner mask ---
        corner_radius = min(8, target_h // 4, target_w // 4)
        if corner_radius > 1:
            mask = np.ones((target_h, target_w), dtype=np.float32)
            yy = np.arange(target_h)[:, None]
            xx = np.arange(target_w)[None, :]
            # Four corners: (cy, cx) of the inscribed circle center
            corners = [
                (corner_radius, corner_radius),                        # top-left
                (corner_radius, target_w - 1 - corner_radius),        # top-right
                (target_h - 1 - corner_radius, corner_radius),        # bottom-left
                (target_h - 1 - corner_radius, target_w - 1 - corner_radius),  # bottom-right
            ]
            for cy, cx in corners:
                # Select the corner quadrant
                if cy <= corner_radius:
                    row_sel = yy < corner_radius
                else:
                    row_sel = yy > target_h - 1 - corner_radius
                if cx <= corner_radius:
                    col_sel = xx < corner_radius
                else:
                    col_sel = xx > target_w - 1 - corner_radius
                in_corner = row_sel & col_sel
                dist_sq = (yy - cy) ** 2 + (xx - cx) ** 2
                outside_circle = dist_sq > corner_radius ** 2
                mask = np.where(in_corner & outside_circle, 0.0, mask)
            bg_resized[:, :, 3] *= mask

        # Placement: bottom-right with 6px margin
        margin = 6
        y0 = fh - target_h - margin
        x0 = fw - target_w - margin

        # --- Drop shadow (dark rounded rect offset by 2px) ---
        shadow_off = 2
        sy0 = y0 + shadow_off
        sx0 = x0 + shadow_off
        sy1 = min(sy0 + target_h, fh)
        sx1 = min(sx0 + target_w, fw)
        sh = sy1 - sy0
        sw = sx1 - sx0
        if sh > 0 and sw > 0:
            shadow_alpha = 0.35
            if corner_radius > 1:
                shadow_mask = mask[:sh, :sw] * shadow_alpha
            else:
                shadow_mask = np.full((sh, sw), shadow_alpha, dtype=np.float32)
            shadow_region = img[sy0:sy1, sx0:sx1]
            shadow_region[:] = shadow_region * (1 - shadow_mask[:, :, None])

        # Alpha-composite background onto frame
        alpha = bg_resized[:, :, 3:4]
        rgb = bg_resized[:, :, :3]
        region = img[y0:y0+target_h, x0:x0+target_w]
        region[:] = region * (1 - alpha) + rgb * alpha

        # Store minimap rect for click-to-teleport
        self._minimap_rect = (x0, y0, target_w, target_h)

        # --- Terrain footprint (visible area quad) ---
        H, W = self.terrain_shape
        cam_col = self.position[0] / self.pixel_spacing_x
        cam_row = self.position[1] / self.pixel_spacing_y
        # Minimap local coords
        lx = cam_col / W * target_w
        ly = cam_row / H * target_h

        ve = self.vertical_exaggeration
        terrain_z = self._get_terrain_z(self.position[0], self.position[1]) * ve

        # Camera basis in VE-scaled space
        pos_ve = np.array([self.position[0], self.position[1],
                           self.position[2] * ve], dtype=np.float32)
        look_ve = np.array([self.position[0] + self._get_front()[0] * 1000,
                            self.position[1] + self._get_front()[1] * 1000,
                            (self.position[2] + self._get_front()[2] * 1000) * ve],
                           dtype=np.float32)
        # Simple basis from yaw/pitch
        yaw_rad = np.radians(self.yaw)
        pitch_rad = np.radians(self.pitch)
        forward = np.array([
            np.cos(yaw_rad) * np.cos(pitch_rad),
            np.sin(yaw_rad) * np.cos(pitch_rad),
            np.sin(pitch_rad),
        ], dtype=np.float32)
        world_up = np.array([0, 0, 1], dtype=np.float32)
        right = np.cross(world_up, forward)
        rn = np.linalg.norm(right)
        if rn > 1e-8:
            right /= rn
        else:
            right = np.array([1, 0, 0], dtype=np.float32)
        up_cam = np.cross(forward, right)

        fov_scale = np.tan(np.radians(self.fov) / 2.0)
        aspect = self.render_width / max(1, self.render_height)

        # Project 4 screen corners onto terrain z-plane
        ndc_corners = [(-1, -1), (1, -1), (1, 1), (-1, 1)]  # BL, BR, TR, TL
        mm_corners = []
        for nx, ny in ndc_corners:
            hit = self._project_corner_to_terrain(
                nx, ny, pos_ve, forward, right, up_cam, fov_scale, aspect, terrain_z)
            if hit is None:
                mm_corners = []
                break
            # Convert world XY to minimap-local coords
            mcol = hit[0] / self.pixel_spacing_x / W * target_w
            mrow = hit[1] / self.pixel_spacing_y / H * target_h
            mm_corners.append((mcol, mrow))

        if len(mm_corners) == 4:
            pts = np.array(mm_corners)  # (4, 2)
            # Fill as two triangles (BL-BR-TR and BL-TR-TL)
            tri1 = pts[[0, 1, 2]]
            tri2 = pts[[0, 2, 3]]
            fill_color = np.array([0.9, 0.9, 0.9])
            self._fill_triangle(img, tri1, x0, y0, target_w, target_h,
                                color=fill_color, alpha_val=0.12)
            self._fill_triangle(img, tri2, x0, y0, target_w, target_h,
                                color=fill_color, alpha_val=0.12)
            # Outline edges
            edge_color = np.array([0.8, 0.8, 0.8])
            for i in range(4):
                j = (i + 1) % 4
                self._draw_line(img, pts[i, 0], pts[i, 1],
                                pts[j, 0], pts[j, 1],
                                x0, y0, target_w, target_h,
                                color=edge_color, thickness=1)

        # Direction line (2px wide red)
        line_len = max(target_h, target_w) * 0.12
        ex = lx + np.cos(yaw_rad) * line_len
        ey = ly + np.sin(yaw_rad) * line_len
        self._draw_line(img, lx, ly, ex, ey, x0, y0, target_w, target_h,
                        color=np.array([1.0, 0.27, 0.27]), thickness=2)

        # Camera dot (red circle, r=3)
        self._draw_dot(img, lx, ly, x0, y0, target_w, target_h,
                       color=np.array([1.0, 0.0, 0.0]), radius=3)

        # Observer dots — colored per-slot, active gets larger radius
        for slot, obs in self._observers.items():
            if obs.position is None:
                continue
            obs_x, obs_y = obs.position
            obs_lx = (obs_x / self.pixel_spacing_x) / W * target_w
            obs_ly = (obs_y / self.pixel_spacing_y) / H * target_h
            r = 4 if slot == self._active_observer else 2
            self._draw_dot(img, obs_lx, obs_ly, x0, y0, target_w, target_h,
                           color=np.array(obs.color), radius=r)

    @staticmethod
    def _draw_dot(img, lx, ly, x0, y0, tw, th, color, radius=3):
        """Draw a filled circle at minimap-local (lx, ly) onto frame."""
        fh, fw = img.shape[:2]
        cx = int(round(lx)) + x0
        cy = int(round(ly)) + y0
        # Clip to minimap rect intersected with frame
        clip_x0, clip_y0 = max(0, x0), max(0, y0)
        clip_x1, clip_y1 = min(fw, x0 + tw), min(fh, y0 + th)
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                if dx*dx + dy*dy <= radius*radius:
                    px, py = cx + dx, cy + dy
                    if clip_x0 <= px < clip_x1 and clip_y0 <= py < clip_y1:
                        img[py, px, :] = color

    @staticmethod
    def _draw_line(img, x1, y1, x2, y2, x0, y0, tw, th, color, thickness=1):
        """Draw a line from (x1,y1) to (x2,y2) in minimap-local coords."""
        fh, fw = img.shape[:2]
        # Clip to minimap rect intersected with frame
        clip_x0, clip_y0 = max(0, x0), max(0, y0)
        clip_x1, clip_y1 = min(fw, x0 + tw), min(fh, y0 + th)
        steps = max(2, int(np.sqrt((x2-x1)**2 + (y2-y1)**2) * 2))
        for i in range(steps + 1):
            t = i / steps
            px = int(round(x1 + (x2-x1)*t)) + x0
            py = int(round(y1 + (y2-y1)*t)) + y0
            for d in range(-(thickness//2), thickness//2 + 1):
                for e in range(-(thickness//2), thickness//2 + 1):
                    ppx, ppy = px + d, py + e
                    if clip_x0 <= ppx < clip_x1 and clip_y0 <= ppy < clip_y1:
                        img[ppy, ppx, :] = color

    @staticmethod
    def _fill_triangle(img, tri, x0, y0, tw, th, color, alpha_val=0.25):
        """Rasterize a filled triangle onto the frame with alpha blending."""
        fh, fw = img.shape[:2]
        # Bounding box in frame coords, clipped to minimap rect
        pts_x = tri[:, 0] + x0
        pts_y = tri[:, 1] + y0
        clip_x0, clip_y0 = max(0, x0), max(0, y0)
        clip_x1, clip_y1 = min(fw - 1, x0 + tw - 1), min(fh - 1, y0 + th - 1)
        min_x = max(clip_x0, int(np.floor(pts_x.min())))
        max_x = min(clip_x1, int(np.ceil(pts_x.max())))
        min_y = max(clip_y0, int(np.floor(pts_y.min())))
        max_y = min(clip_y1, int(np.ceil(pts_y.max())))

        # Vectorised point-in-triangle using barycentric coords
        v0 = tri[2] - tri[0]
        v1 = tri[1] - tri[0]
        d00 = v0[0]*v0[0] + v0[1]*v0[1]
        d01 = v0[0]*v1[0] + v0[1]*v1[1]
        d11 = v1[0]*v1[0] + v1[1]*v1[1]
        denom = d00*d11 - d01*d01
        if abs(denom) < 1e-12:
            return

        ys = np.arange(min_y, max_y + 1)
        xs = np.arange(min_x, max_x + 1)
        if len(ys) == 0 or len(xs) == 0:
            return
        gx, gy = np.meshgrid(xs, ys)
        v2x = gx - (tri[0, 0] + x0)
        v2y = gy - (tri[0, 1] + y0)
        d20 = v2x*v0[0] + v2y*v0[1]
        d21 = v2x*v1[0] + v2y*v1[1]
        u = (d11*d20 - d01*d21) / denom
        v = (d00*d21 - d01*d20) / denom
        inside = (u >= 0) & (v >= 0) & (u + v <= 1)

        if inside.any():
            iy = gy[inside]
            ix = gx[inside]
            img[iy, ix, :] = img[iy, ix, :] * (1 - alpha_val) + color * alpha_val

    # ------------------------------------------------------------------
    # Wind particle animation
    # ------------------------------------------------------------------

    def _toggle_wind(self):
        """Toggle wind particle animation on/off."""
        if self._wind_data is None:
            print("No wind data loaded. Pass wind_data to explore().")
            return
        self._wind_enabled = not self._wind_enabled
        print(f"Wind particles: {'ON' if self._wind_enabled else 'OFF'}")
        self._update_frame()

    def _toggle_firms(self):
        """Fetch and toggle NASA FIRMS LANDSAT fire footprints (Shift+F)."""
        if self._accessor is None:
            print("No accessor available for FIRMS fire layer.")
            return

        if not self._firms_loaded:
            # First press: fetch + place
            print("Fetching FIRMS fire data (7d LANDSAT)...")
            try:
                from .remote_data import fetch_firms
                from .tiles import _build_latlon_grids
                import warnings

                # Get WGS84 bounds from the raster
                lats, lons = _build_latlon_grids(self._base_raster)
                bounds = (
                    float(lons.min()), float(lats.min()),
                    float(lons.max()), float(lats.max()),
                )

                # Detect CRS for reprojection
                crs = None
                try:
                    raster_crs = self._base_raster.rio.crs
                    if raster_crs is not None and not raster_crs.is_geographic:
                        crs = str(raster_crs)
                except (AttributeError, ImportError):
                    pass

                fire_data = fetch_firms(bounds, date_span='7d', crs=crs)

                n_fires = len(fire_data.get('features', []))
                if n_fires == 0:
                    print("No fire detections in the last 7 days.")
                    self._firms_loaded = True
                    return

                with warnings.catch_warnings():
                    warnings.filterwarnings(
                        "ignore", message="place_geojson called before")
                    self._accessor.place_geojson(
                        fire_data,
                        height=max(self.pixel_spacing_x,
                                   self.pixel_spacing_y) * 0.5,
                        geometry_id='fire',
                        color=(1.0, 0.25, 0.0, 3.0),
                        extrude=True,
                        merge=True,
                    )

                self._firms_loaded = True
                self._firms_visible = True

                # Ensure geometry color builder is active
                if (self._geometry_colors_builder is None
                        and self._accessor._geometry_colors):
                    self._geometry_colors_builder = (
                        self._accessor._build_geometry_colors_gpu)

                # Refresh geometry layer tracking
                if self.rtx is not None:
                    self._all_geometries = self.rtx.list_geometries()
                    groups = set()
                    for g in self._all_geometries:
                        parts = g.rsplit('_', 1)
                        if len(parts) == 2 and parts[1].isdigit():
                            base = parts[0]
                        else:
                            base = g
                        if base != 'terrain':
                            groups.add(base)
                    self._geometry_layer_order = (
                        ['none', 'all'] + sorted(groups))

                print(f"FIRMS fire: ON  ({n_fires} detections)")
                self._update_frame()

            except Exception as e:
                print(f"FIRMS fire fetch failed: {e}")
            return

        # Subsequent presses: toggle visibility
        self._firms_visible = not self._firms_visible
        if self.rtx is not None:
            for geom_id in self.rtx.list_geometries():
                if geom_id.startswith('fire'):
                    self.rtx.set_geometry_visible(
                        geom_id, self._firms_visible)
        print(f"FIRMS fire: {'ON' if self._firms_visible else 'OFF'}")
        self._update_frame()

    def _init_wind(self, wind_data):
        """Interpolate wind U/V from lat/lon grid onto the terrain pixel grid.

        Converts wind from m/s in geographic space to pixels/tick in raster
        pixel space so particles can be advected directly in pixel coords.
        """
        self._wind_data = wind_data
        if wind_data is None:
            return

        # Allow wind_data dict to carry optional tuning overrides
        if 'n_particles' in wind_data:
            self._wind_n_particles = int(wind_data['n_particles'])
        if 'max_age' in wind_data:
            self._wind_max_age = int(wind_data['max_age'])
        if 'speed_mult' in wind_data:
            self._wind_speed_mult = float(wind_data['speed_mult'])
        if 'trail_len' in wind_data:
            self._wind_trail_len = int(wind_data['trail_len'])
        if 'dot_radius' in wind_data:
            self._wind_dot_radius = int(wind_data['dot_radius'])
        if 'alpha' in wind_data:
            self._wind_alpha = float(wind_data['alpha'])
        if 'min_visible_age' in wind_data:
            self._wind_min_visible_age = int(wind_data['min_visible_age'])

        from .tiles import _build_latlon_grids
        raster = self._base_raster
        H, W = raster.shape

        # Build per-pixel lat/lon grids for the terrain
        lats_grid, lons_grid = _build_latlon_grids(raster)

        # Wind data grid
        w_lats = wind_data['lats']  # (ny,)
        w_lons = wind_data['lons']  # (nx,)
        w_u = wind_data['u']        # (ny, nx) m/s eastward
        w_v = wind_data['v']        # (ny, nx) m/s northward

        # For each terrain pixel, bilinear-interpolate wind U/V from the
        # wind lat/lon grid.
        from scipy.interpolate import RegularGridInterpolator
        interp_u = RegularGridInterpolator(
            (w_lats, w_lons), w_u,
            method='linear', bounds_error=False, fill_value=0.0,
        )
        interp_v = RegularGridInterpolator(
            (w_lats, w_lons), w_v,
            method='linear', bounds_error=False, fill_value=0.0,
        )

        points = np.stack([lats_grid.ravel(), lons_grid.ravel()], axis=-1)
        u_ms = interp_u(points).reshape(H, W).astype(np.float32)
        v_ms = interp_v(points).reshape(H, W).astype(np.float32)

        # Convert m/s to pixels/tick.
        # pixel_spacing is in metres, so 1 pixel = pixel_spacing metres.
        # At ~20 ticks/sec, scale = dt / pixel_spacing.
        # Multiply by speed_mult for dramatic visual effect.
        dt = 0.05  # seconds per tick (matches key_repeat_interval)
        sm = self._wind_speed_mult
        self._wind_u_px = u_ms * dt * sm / self._base_pixel_spacing_x   # east = +col
        self._wind_v_px = -(v_ms * dt * sm / self._base_pixel_spacing_y)  # north = -row (row 0 is north)

        # Precompute terrain slope gradients (pixels/tick contribution).
        # dz/dcol and dz/drow tell us the downslope direction in pixel space.
        # Particles get pushed downhill and deflected around steep terrain.
        terrain_data = self._base_raster.data
        if hasattr(terrain_data, 'get'):
            terrain_np = terrain_data.get()
        else:
            terrain_np = np.asarray(terrain_data)
        # Gradient in row/col directions (units: elevation per pixel)
        # NaN-fill so ocean/water pixels have zero slope influence and
        # particles flow purely by wind over water.
        grad_row, grad_col = np.gradient(np.nan_to_num(terrain_np.astype(np.float32), nan=0.0))
        # Downslope force = -gradient (pushes particles toward lower elevation)
        # Scale relative to wind speed so slope matters but doesn't dominate
        slope_scale = dt * sm * 0.15
        self._wind_slope_col = (-grad_col * slope_scale).astype(np.float32)
        self._wind_slope_row = (-grad_row * slope_scale).astype(np.float32)

        # Spawn initial particles with jittered lifetimes for staggered deaths
        self._wind_particles = np.column_stack([
            np.random.uniform(0, H, self._wind_n_particles),
            np.random.uniform(0, W, self._wind_n_particles),
        ]).astype(np.float32)
        self._wind_lifetimes = np.random.randint(
            self._wind_max_age // 2, self._wind_max_age, self._wind_n_particles)
        self._wind_ages = np.random.randint(0, self._wind_max_age, self._wind_n_particles)
        self._wind_trails = np.zeros(
            (self._wind_n_particles, self._wind_trail_len, 2), dtype=np.float32,
        )
        # Initialize trails to current position
        for t in range(self._wind_trail_len):
            self._wind_trails[:, t, :] = self._wind_particles

        # Min render distance — skip particles near the camera so they
        # don't appear as distracting blobs in the foreground
        world_diag = np.sqrt((W * self._base_pixel_spacing_x)**2 +
                             (H * self._base_pixel_spacing_y)**2)
        self._wind_min_depth = world_diag * 0.02

        print(f"  Wind field interpolated onto {H}x{W} terrain grid")

    def _update_wind_particles(self):
        """Advect wind particles one tick using bilinear-sampled wind field."""
        if self._wind_u_px is None or self._wind_particles is None:
            return

        H, W = self._wind_u_px.shape
        pts = self._wind_particles  # (N, 2) — (row, col)

        # Shift trail buffer (drop oldest, prepend current position)
        self._wind_trails[:, 1:, :] = self._wind_trails[:, :-1, :]
        self._wind_trails[:, 0, :] = pts

        # Bilinear sample wind at particle positions
        rows = pts[:, 0]
        cols = pts[:, 1]
        # Replace NaN with 0 before int cast to avoid RuntimeWarning
        r0 = np.clip(np.floor(np.nan_to_num(rows, nan=0.0)).astype(int), 0, H - 2)
        c0 = np.clip(np.floor(np.nan_to_num(cols, nan=0.0)).astype(int), 0, W - 2)
        fr = rows - r0
        fc = cols - c0

        # Subsample factor: wind grids are at base resolution
        f = self.subsample_factor

        # Sample U (col velocity)
        u00 = self._wind_u_px[r0, c0]
        u10 = self._wind_u_px[r0, c0 + 1]
        u01 = self._wind_u_px[r0 + 1, c0]
        u11 = self._wind_u_px[r0 + 1, c0 + 1]
        u_val = u00 * (1 - fr) * (1 - fc) + u10 * (1 - fr) * fc + u01 * fr * (1 - fc) + u11 * fr * fc

        # Sample V (row velocity)
        v00 = self._wind_v_px[r0, c0]
        v10 = self._wind_v_px[r0, c0 + 1]
        v01 = self._wind_v_px[r0 + 1, c0]
        v11 = self._wind_v_px[r0 + 1, c0 + 1]
        v_val = v00 * (1 - fr) * (1 - fc) + v10 * (1 - fr) * fc + v01 * fr * (1 - fc) + v11 * fr * fc

        # Add terrain slope influence — particles flow downhill.
        # If already headed downhill (wind aligns with downslope),
        # dampen the slope contribution so it doesn't pile on.
        if self._wind_slope_col is not None:
            sc00 = self._wind_slope_col[r0, c0]
            sc10 = self._wind_slope_col[r0, c0 + 1]
            sc01 = self._wind_slope_col[r0 + 1, c0]
            sc11 = self._wind_slope_col[r0 + 1, c0 + 1]
            slope_u = sc00 * (1 - fr) * (1 - fc) + sc10 * (1 - fr) * fc + sc01 * fr * (1 - fc) + sc11 * fr * fc

            sr00 = self._wind_slope_row[r0, c0]
            sr10 = self._wind_slope_row[r0, c0 + 1]
            sr01 = self._wind_slope_row[r0 + 1, c0]
            sr11 = self._wind_slope_row[r0 + 1, c0 + 1]
            slope_v = sr00 * (1 - fr) * (1 - fc) + sr10 * (1 - fr) * fc + sr01 * fr * (1 - fc) + sr11 * fr * fc

            # Dot of wind velocity with downslope direction:
            # positive = already headed downhill → reduce slope push
            slope_mag = np.sqrt(slope_u**2 + slope_v**2) + 1e-8
            wind_mag = np.sqrt(u_val**2 + v_val**2) + 1e-8
            alignment = (u_val * slope_u + v_val * slope_v) / (wind_mag * slope_mag)
            # alignment in [-1, 1]: +1 = fully downhill, -1 = fully uphill
            # dampen: full slope when uphill (alignment<0), reduced when downhill
            dampen = np.clip(1.0 - alignment, 0.2, 1.0)

            u_val += slope_u * dampen
            v_val += slope_v * dampen

        # Advect (scale by dt so wind speed is frame-rate independent)
        s = self._dt_scale
        pts[:, 0] += v_val * s  # row
        pts[:, 1] += u_val * s  # col

        # Age particles
        self._wind_ages += 1

        # Respawn out-of-bounds, NaN, or aged-out particles
        nan_pos = np.isnan(pts[:, 0]) | np.isnan(pts[:, 1])
        oob = nan_pos | (pts[:, 0] < 0) | (pts[:, 0] >= H) | (pts[:, 1] < 0) | (pts[:, 1] >= W)
        old = self._wind_ages >= self._wind_lifetimes
        respawn = oob | old

        n_respawn = int(respawn.sum())
        if n_respawn > 0:
            pts[respawn, 0] = np.random.uniform(0, H, n_respawn)
            pts[respawn, 1] = np.random.uniform(0, W, n_respawn)
            self._wind_ages[respawn] = 0
            # Jitter per-particle lifetime so they don't all expire in sync
            self._wind_lifetimes[respawn] = np.random.randint(
                self._wind_max_age // 2, self._wind_max_age, n_respawn)
            # Reset trails for respawned particles
            for t in range(self._wind_trail_len):
                self._wind_trails[respawn, t, :] = pts[respawn]

    def _draw_wind_on_frame(self, img):
        """Project wind particles to screen space and draw on rendered frame.

        Fully vectorised: projects all trail positions in one batch, then
        splats them with a single ``np.add.at`` call per stamp offset.

        Parameters
        ----------
        img : ndarray, shape (H_screen, W_screen, 3)
            Rendered frame (float32 0-1) to draw on. Modified in-place.
        """
        if self._wind_particles is None:
            return

        from .analysis.render import _compute_camera_basis
        import math

        sh, sw = img.shape[:2]
        N = self._wind_particles.shape[0]
        trail_len = self._wind_trail_len

        # Camera basis matching the ray tracer
        cam_pos = self.position
        look_at = self._get_look_at()
        forward, right, cam_up = _compute_camera_basis(
            tuple(cam_pos), tuple(look_at), (0, 0, 1),
        )
        fov_scale = math.tan(math.radians(self.fov) / 2.0)
        aspect_ratio = sw / sh

        # Cached CPU terrain — avoid GPU→CPU copy every frame
        if self._wind_terrain_np is None:
            terrain_data = self.raster.data
            if hasattr(terrain_data, 'get'):
                self._wind_terrain_np = terrain_data.get()
            else:
                self._wind_terrain_np = np.asarray(terrain_data)
        terrain_np = self._wind_terrain_np
        tH, tW = terrain_np.shape

        f = self.subsample_factor
        psx = self._base_pixel_spacing_x
        psy = self._base_pixel_spacing_y
        ve = self.vertical_exaggeration
        min_depth = self._wind_min_depth

        # --- Batch all trail positions into one flat array ---
        # trails shape: (N, trail_len, 2) → (N * trail_len, 2)
        all_pts = self._wind_trails.reshape(-1, 2)  # (N*T, 2)
        rows_all = all_pts[:, 0]
        cols_all = all_pts[:, 1]

        # --- Single batched projection ---
        sr = np.clip(np.nan_to_num(rows_all / f, nan=0.0).astype(np.int32), 0, tH - 1)
        sc = np.clip(np.nan_to_num(cols_all / f, nan=0.0).astype(np.int32), 0, tW - 1)
        z_vals = np.nan_to_num(terrain_np[sr, sc], nan=0.0) * ve + 3.0

        wx = cols_all * psx
        wy = rows_all * psy

        dx = wx - cam_pos[0]
        dy = wy - cam_pos[1]
        dz = z_vals - cam_pos[2]

        depth = dx * forward[0] + dy * forward[1] + dz * forward[2]
        valid = depth > min_depth

        inv_depth = np.where(valid, 1.0 / (depth + 1e-10), 0.0)
        u_cam = dx * right[0] + dy * right[1] + dz * right[2]
        v_cam = dx * cam_up[0] + dy * cam_up[1] + dz * cam_up[2]
        u_ndc = u_cam * inv_depth / (fov_scale * aspect_ratio)
        v_ndc = v_cam * inv_depth / fov_scale

        sx_all = np.nan_to_num(((u_ndc + 1.0) * 0.5 * sw), nan=-1.0).astype(np.int32)
        sy_all = np.nan_to_num(((1.0 - v_ndc) * 0.5 * sh), nan=-1.0).astype(np.int32)

        on_screen = valid & (sx_all >= 0) & (sx_all < sw) & (sy_all >= 0) & (sy_all < sh)

        # --- Build per-point alpha (fade-in, fade-out, trail decay) ---
        # tile ages/lifetimes to match (N*T,) layout
        ages = self._wind_ages  # (N,)
        lifetimes = self._wind_lifetimes  # (N,)

        # Trail index for each point: 0=head, 1=prev, ...
        trail_idx = np.tile(np.arange(trail_len, dtype=np.float32), N)  # (N*T,)
        # Particle must be at least trail_idx ticks old
        ages_rep = np.repeat(ages, trail_len)  # (N*T,)
        lifetimes_rep = np.repeat(lifetimes, trail_len)  # (N*T,)
        age_ok = ages_rep > trail_idx

        # Fade in/out over particle lifetime
        # Dead zone: invisible for first _wind_min_visible_age ticks while
        # the particle silently builds a trail, then fade in over 10 ticks.
        # This eliminates the "twinkle" of a single dot appearing.
        mva = self._wind_min_visible_age
        fade_in = np.clip((ages_rep - mva) / 10.0, 0, 1)
        fade_out = np.clip((lifetimes_rep - ages_rep) / 20.0, 0, 1)
        # Trail decay: head=1.0, tail→0.0
        trail_fade = 1.0 - (trail_idx / trail_len)

        alpha = self._wind_alpha * fade_in * fade_out * trail_fade

        # Final mask: on screen, old enough, positive alpha
        mask = on_screen & age_ok & (alpha > 1e-6)
        if not mask.any():
            return img

        sx_m = sx_all[mask]
        sy_m = sy_all[mask]
        alpha_m = alpha[mask].astype(np.float32)

        # --- Splat with stamp offsets using np.add.at ---
        color = np.array([0.3, 0.9, 0.8], dtype=np.float32)
        r = self._wind_dot_radius
        for offy in range(-r, r + 1):
            for offx in range(-r, r + 1):
                dist_sq = offx * offx + offy * offy
                if dist_sq > r * r:
                    continue
                falloff = 1.0 - (dist_sq / (r * r)) ** 0.5

                px = sx_m + offx
                py = sy_m + offy
                ok = (px >= 0) & (px < sw) & (py >= 0) & (py < sh)
                if not ok.any():
                    continue

                contribution = alpha_m[ok] * falloff
                for c in range(3):
                    np.add.at(img[:, :, c], (py[ok], px[ok]), contribution * color[c])

        np.clip(img, 0, 1, out=img)
        return img

    # ------------------------------------------------------------------
    # GTFS-RT realtime vehicle overlay
    # ------------------------------------------------------------------

    def _init_gtfs_rt(self, realtime_url, route_colors=None):
        """Initialize GTFS-RT realtime vehicle polling.

        Parameters
        ----------
        realtime_url : str
            URL to a GTFS-Realtime VehiclePositions protobuf feed.
        route_colors : dict, optional
            ``{route_id: (r, g, b)}`` mapping.  If not provided, all
            vehicles render in white.
        """
        self._gtfs_rt_url = realtime_url
        if route_colors:
            self._gtfs_rt_route_colors = route_colors
        print(f"GTFS-RT feed configured: {realtime_url}")
        print("  Press Shift+B to toggle realtime vehicle overlay.")

    def _toggle_gtfs_rt(self):
        """Toggle GTFS-RT realtime vehicle overlay on/off."""
        if self._gtfs_rt_url is None:
            print("No GTFS-RT feed configured. Pass realtime_url in gtfs_data metadata.")
            return
        self._gtfs_rt_enabled = not self._gtfs_rt_enabled
        if self._gtfs_rt_enabled:
            if self._gtfs_rt_thread is None or not self._gtfs_rt_thread.is_alive():
                self._gtfs_rt_stop.clear()
                self._gtfs_rt_thread = threading.Thread(
                    target=self._gtfs_rt_poll_loop, daemon=True)
                self._gtfs_rt_thread.start()
            print("GTFS-RT vehicles: ON")
        else:
            self._gtfs_rt_stop.set()
            print("GTFS-RT vehicles: OFF")
        self._update_frame()

    def _gtfs_rt_poll_loop(self):
        """Background thread: poll GTFS-RT feed at regular intervals."""
        import requests

        while not self._gtfs_rt_stop.is_set():
            try:
                resp = requests.get(self._gtfs_rt_url, timeout=30)
                resp.raise_for_status()
                self._parse_gtfs_rt_response(resp.content)
                self._render_needed = True
            except Exception as e:
                print(f"GTFS-RT poll error: {e}")

            self._gtfs_rt_stop.wait(self._gtfs_rt_poll_interval)

    def _parse_gtfs_rt_response(self, data):
        """Parse GTFS-RT protobuf VehiclePositions into numpy arrays."""
        try:
            from google.transit import gtfs_realtime_pb2
        except ImportError:
            print("gtfs-realtime-bindings required for GTFS-RT. "
                  "Install with: pip install gtfs-realtime-bindings")
            self._gtfs_rt_stop.set()
            self._gtfs_rt_enabled = False
            return

        feed = gtfs_realtime_pb2.FeedMessage()
        feed.ParseFromString(data)

        positions = []
        bearings = []
        colors = []

        for entity in feed.entity:
            if not entity.HasField('vehicle'):
                continue
            vp = entity.vehicle
            if not vp.HasField('position'):
                continue
            pos = vp.position
            lat = pos.latitude
            lon = pos.longitude
            bearing = pos.bearing if pos.bearing else 0.0

            # Determine color from route
            route_id = vp.trip.route_id if vp.HasField('trip') else ''
            color = self._gtfs_rt_route_colors.get(route_id, (1.0, 1.0, 1.0))

            positions.append((lon, lat))
            bearings.append(bearing)
            colors.append(color)

        if positions:
            with self._gtfs_rt_lock:
                self._gtfs_rt_vehicles = (
                    np.array(positions, dtype=np.float64),
                    np.array(bearings, dtype=np.float32),
                    np.array(colors, dtype=np.float32),
                )

    def _draw_gtfs_rt_on_frame(self, img):
        """Draw GTFS-RT vehicle positions as colored dots on the frame."""
        with self._gtfs_rt_lock:
            if self._gtfs_rt_vehicles is None:
                return
            positions, bearings, colors = self._gtfs_rt_vehicles

        if len(positions) == 0:
            return

        # Convert lon/lat to world coordinates (pixel space)
        da = self.raster
        y_coords = da.coords[da.dims[-2]].values
        x_coords = da.coords[da.dims[-1]].values

        # lon/lat → pixel coords
        px_x = (positions[:, 0] - x_coords[0]) / (x_coords[-1] - x_coords[0]) * (len(x_coords) - 1)
        px_y = (positions[:, 1] - y_coords[0]) / (y_coords[-1] - y_coords[0]) * (len(y_coords) - 1)

        # World coords (match terrain mesh coordinate system)
        wx = px_x * abs(self.pixel_spacing_x)
        wy = px_y * abs(self.pixel_spacing_y)

        # Sample terrain Z for each vehicle (nearest neighbor)
        H, W = da.shape[-2:]
        ix = np.clip(np.round(px_x).astype(int), 0, W - 1)
        iy = np.clip(np.round(px_y).astype(int), 0, H - 1)

        terrain_np = self._wind_terrain_np
        if terrain_np is None:
            try:
                import cupy
                terrain_np = cupy.asnumpy(da.values)
            except Exception:
                terrain_np = np.asarray(da.values)
            self._wind_terrain_np = terrain_np

        wz = terrain_np[iy, ix].astype(np.float64) * self.vertical_exaggeration
        # Replace NaN with 0
        wz = np.where(np.isfinite(wz), wz, 0.0)

        # Project to screen space
        world = np.stack([wx, wy, wz], axis=-1)  # (N, 3)
        cam_pos = np.array(self.position, dtype=np.float64)
        cam_fwd = np.array(self._camera_forward(), dtype=np.float64)
        cam_right = np.array(self._camera_right(), dtype=np.float64)
        cam_up = np.array(self._camera_up(), dtype=np.float64)

        rel = world - cam_pos  # (N, 3)
        depth = rel @ cam_fwd
        behind = depth <= 0.1
        depth[behind] = 1.0  # avoid division by zero

        fov_rad = np.radians(self.fov)
        sh, sw = img.shape[:2]
        f = sw / (2.0 * np.tan(fov_rad / 2.0))

        sx = (rel @ cam_right) * f / depth + sw / 2.0
        sy = (rel @ cam_up) * f / depth + sh / 2.0
        # Flip Y (screen Y is top-down)
        sy = sh - 1 - sy

        # Filter to on-screen, not behind camera
        valid = (~behind) & (sx >= -10) & (sx < sw + 10) & (sy >= -10) & (sy < sh + 10)
        if not valid.any():
            return

        sx = sx[valid].astype(np.int32)
        sy = sy[valid].astype(np.int32)
        vc = colors[valid]
        r = self._gtfs_rt_dot_radius
        alpha = self._gtfs_rt_alpha

        # Splat colored dots
        for i in range(len(sx)):
            x0 = max(0, sx[i] - r)
            x1 = min(sw, sx[i] + r + 1)
            y0 = max(0, sy[i] - r)
            y1 = min(sh, sy[i] + r + 1)
            if x0 >= x1 or y0 >= y1:
                continue
            # Circular mask
            yy, xx = np.mgrid[y0:y1, x0:x1]
            dist_sq = (xx - sx[i]) ** 2 + (yy - sy[i]) ** 2
            mask = dist_sq <= r * r
            falloff = np.where(mask, 1.0 - np.sqrt(dist_sq[mask].astype(float)) / r, 0.0)
            c = vc[i]
            for ch in range(3):
                patch = img[y0:y1, x0:x1, ch]
                patch[mask] = patch[mask] * (1.0 - alpha * falloff) + c[ch] * alpha * falloff

        return img

    def _cleanup_gtfs_rt(self):
        """Stop the GTFS-RT poll thread."""
        if self._gtfs_rt_thread is not None:
            self._gtfs_rt_stop.set()
            self._gtfs_rt_thread.join(timeout=2.0)
            self._gtfs_rt_thread = None

    def _handle_key_press(self, raw_key, key):
        """Handle key press - add to held keys or handle instant actions.

        Parameters
        ----------
        raw_key : str
            Key with original case (uppercase if SHIFT held).
        key : str
            Lowercase version of the key.
        """

        # Drone mode cycle: Shift+O (before other keys)
        if raw_key == 'O':
            obs = self._observers.get(self._active_observer) if self._active_observer else None
            if obs is None:
                print("No observer selected. Press 1-8 first.")
            else:
                self._cycle_drone_mode_for(obs)
            return

        # Snap camera to active observer: Shift+V
        if raw_key == 'V':
            obs = self._observers.get(self._active_observer) if self._active_observer else None
            if obs is None:
                print("No observer selected. Press 1-8 first.")
            else:
                self._snap_to_observer(obs)
            return

        # Kill all observers: Shift+K
        if raw_key == 'K':
            self._clear_all_observers()
            return

        # FIRMS fire layer: Shift+F (before 'f' screenshot)
        if raw_key == 'F':
            self._toggle_firms()
            return

        # Wind toggle: Shift+W (before movement keys capture 'w')
        if raw_key == 'W':
            self._toggle_wind()
            return

        # GTFS-RT realtime vehicle toggle: Shift+B
        if raw_key == 'B':
            self._toggle_gtfs_rt()
            return

        # Denoiser toggle: Shift+D (before movement keys capture 'd')
        if raw_key == 'D':
            self.denoise_enabled = not self.denoise_enabled
            self._d_ao_accum = None
            self._ao_frame_count = 0
            self._prev_cam_state = None
            self._prev_cam_for_flow = None
            print(f"Denoiser: {'ON' if self.denoise_enabled else 'OFF'}")
            self._update_frame()
            return

        # GI bounces cycle: Shift+G (1 → 2 → 3 → 1)
        if raw_key == 'G':
            self.gi_bounces = self.gi_bounces % 3 + 1
            self._d_ao_accum = None
            self._ao_frame_count = 0
            self._prev_cam_state = None
            print(f"GI bounces: {self.gi_bounces}")
            self._update_frame()
            return

        # Drone glow toggle: Shift+L
        if raw_key == 'L':
            self._drone_glow = not self._drone_glow
            self._apply_drone_glow()
            print(f"Drone glow: {'ON' if self._drone_glow else 'OFF'}")
            return

        # Time-of-day cycle: Shift+T (before 't' shadows toggle)
        if raw_key == 'T':
            self._time_preset_idx = (self._time_preset_idx + 1) % len(self._time_presets)
            name, az, alt = self._time_presets[self._time_preset_idx]
            self.sun_azimuth = az
            self.sun_altitude = alt
            self._d_ao_accum = None
            self._ao_frame_count = 0
            self._prev_cam_state = None
            print(f"Time of day: {name} (az={az:.0f}, alt={alt:.0f})")
            self._update_frame()
            return

        # Movement/look keys are tracked as held
        movement_keys = {'w', 's', 'a', 'd', 'up', 'down', 'left', 'right',
                         'q', 'e', 'pageup', 'pagedown', 'i', 'j', 'k', 'l'}

        if key in movement_keys:
            self._held_keys.add(key)
            return

        # Instant actions (not held)
        # Speed (limits scale with terrain size in world units)
        if key in ('+', '='):
            H, W = self.terrain_shape
            world_diag = np.sqrt((W * self.pixel_spacing_x)**2 + (H * self.pixel_spacing_y)**2)
            max_speed = world_diag * 0.1  # Max 10% of terrain per keystroke
            self.move_speed = min(max_speed, self.move_speed * 1.2)
            print(f"Speed: {self.move_speed:.3f}")
        elif key == '-':
            H, W = self.terrain_shape
            world_diag = np.sqrt((W * self.pixel_spacing_x)**2 + (H * self.pixel_spacing_y)**2)
            min_speed = 0.001
            self.move_speed = max(min_speed, self.move_speed / 1.2)
            print(f"Speed: {self.move_speed:.3f}")

        # Toggles
        elif key == 't':
            self.shadows = not self.shadows
            print(f"Shadows: {'ON' if self.shadows else 'OFF'}")
            self._update_frame()
        elif key == 'c':
            self.colormap_idx = (self.colormap_idx + 1) % len(self.colormaps)
            self.colormap = self.colormaps[self.colormap_idx]
            print(f"Colormap: {self.colormap}")
            self._update_frame()
        elif key == 'g':
            self._cycle_terrain_layer()
        elif key == 'n':
            self._cycle_geometry_layer()
        elif key == 'p':
            self._jump_to_geometry(-1)  # Previous geometry in current group
        elif key == 'h':
            self.show_help = not self.show_help
            self._update_frame()
        elif key == 'm':
            self.show_minimap = not self.show_minimap
            self._update_frame()

        # Observer slot selection: 1-8
        elif key in ('1', '2', '3', '4', '5', '6', '7', '8'):
            self._select_or_create_observer(int(key))

        # Move active observer to camera position
        elif key == 'o':
            obs = self._observers.get(self._active_observer) if self._active_observer else None
            if obs is None:
                print("No observer selected. Press 1-8 to create one.")
            elif obs.drone_mode == 'off':
                self._place_observer_at(obs)
        elif key == 'v':
            self._toggle_viewshed()
        elif key == '[':
            self._adjust_observer_elevation(-0.01)
        elif key == ']':
            self._adjust_observer_elevation(0.01)

        # Screenshot
        elif key == 'f':
            self._save_screenshot()

        # Terrain resolution: R = coarser, Shift+R = finer
        elif key == 'r':
            if raw_key == 'R':
                # Shift+R → finer (halve factor, min 1)
                new_factor = max(1, self.subsample_factor // 2)
            else:
                # r → coarser (double factor, max 8)
                new_factor = min(8, self.subsample_factor * 2)
            if new_factor != self.subsample_factor:
                self._rebuild_at_resolution(new_factor)

        # Color stretch cycling
        elif key == 'y':
            self._color_stretch_idx = (self._color_stretch_idx + 1) % len(self._color_stretches)
            self.color_stretch = self._color_stretches[self._color_stretch_idx]
            print(f"Color stretch: {self.color_stretch}")
            self._update_frame()

        # Cycle mesh type (tin → voxel → heightfield → tin)
        elif key == 'b':
            cycle = {'tin': 'voxel', 'voxel': 'heightfield', 'heightfield': 'tin'}
            self.mesh_type = cycle.get(self.mesh_type, 'tin')
            self._rebuild_vertical_exaggeration(self.vertical_exaggeration)
            print(f"Mesh type: {self.mesh_type}")

        # Basemap cycling: U = cycle none → satellite → osm → none
        elif key == 'u':
            self._cycle_basemap()

        # Overlay alpha: , = decrease, . = increase
        elif key == ',':
            self._overlay_alpha = max(0.0, round(self._overlay_alpha - 0.1, 1))
            print(f"Overlay alpha: {int(self._overlay_alpha * 100)}%")
            self._update_frame()
        elif key == '.':
            self._overlay_alpha = min(1.0, round(self._overlay_alpha + 0.1, 1))
            print(f"Overlay alpha: {int(self._overlay_alpha * 100)}%")
            self._update_frame()

        # Vertical exaggeration: Z = decrease, Shift+Z = increase (0.1 steps)
        elif key == 'z':
            if raw_key == 'Z':
                new_ve = round(self.vertical_exaggeration + 0.1, 1)
                new_ve = min(10.0, new_ve)
            else:
                new_ve = round(self.vertical_exaggeration - 0.1, 1)
                new_ve = max(0.1, new_ve)
            if new_ve != self.vertical_exaggeration:
                self._rebuild_vertical_exaggeration(new_ve)

        # Ambient occlusion toggle
        elif key == '0':
            self.ao_enabled = not self.ao_enabled
            self._d_ao_accum = None
            self._ao_frame_count = 0
            self._prev_cam_state = None
            print(f"Ambient Occlusion: {'ON' if self.ao_enabled else 'OFF'}")
            self._update_frame()

        # Depth of field toggle
        elif key == '9':
            self.dof_enabled = not self.dof_enabled
            # Reset accumulation so DOF takes effect immediately
            self._d_ao_accum = None
            self._ao_frame_count = 0
            self._prev_cam_state = None
            print(f"Depth of Field: {'ON' if self.dof_enabled else 'OFF'}")
            self._update_frame()

        # DOF aperture: ; = decrease, ' = increase
        elif key == ';':
            self._dof_aperture = max(1.0, self._dof_aperture * 0.7)
            self._d_ao_accum = None
            self._ao_frame_count = 0
            self._prev_cam_state = None
            print(f"DOF aperture: {self._dof_aperture:.1f}")
            self._update_frame()
        elif key == "'":
            self._dof_aperture = min(200.0, self._dof_aperture * 1.4)
            self._d_ao_accum = None
            self._ao_frame_count = 0
            self._prev_cam_state = None
            print(f"DOF aperture: {self._dof_aperture:.1f}")
            self._update_frame()

        # DOF focal distance: : = decrease, " = increase
        elif key == ':':
            self._dof_focal_distance = max(10.0, self._dof_focal_distance * 0.7)
            self._d_ao_accum = None
            self._ao_frame_count = 0
            self._prev_cam_state = None
            print(f"DOF focal distance: {self._dof_focal_distance:.0f}")
            self._update_frame()
        elif key == '"':
            self._dof_focal_distance = min(10000.0, self._dof_focal_distance * 1.4)
            self._d_ao_accum = None
            self._ao_frame_count = 0
            self._prev_cam_state = None
            print(f"DOF focal distance: {self._dof_focal_distance:.0f}")
            self._update_frame()

        # Exit
        elif key in ('escape', 'x'):
            self.running = False

    def _handle_key_release(self, key):
        """Handle key release - remove from held keys.

        Parameters
        ----------
        key : str
            Lowercase key name.
        """
        self._held_keys.discard(key)

    @staticmethod
    def _get_drone_front_for(obs):
        """Get forward direction for drone flight (uses observer yaw/pitch)."""
        yaw_rad = np.radians(obs.yaw)
        pitch_rad = np.radians(obs.pitch)
        return np.array([
            np.cos(yaw_rad) * np.cos(pitch_rad),
            np.sin(yaw_rad) * np.cos(pitch_rad),
            np.sin(pitch_rad)
        ], dtype=np.float32)

    @staticmethod
    def _get_drone_right_for(obs):
        """Get right direction for drone flight."""
        yaw_rad = np.radians(obs.yaw)
        pitch_rad = np.radians(obs.pitch)
        front = np.array([
            np.cos(yaw_rad) * np.cos(pitch_rad),
            np.sin(yaw_rad) * np.cos(pitch_rad),
            np.sin(pitch_rad)
        ], dtype=np.float32)
        world_up = np.array([0, 0, 1], dtype=np.float32)
        right = np.cross(world_up, front)
        return right / (np.linalg.norm(right) + 1e-8)

    def _clamp_drone_pos(self, pos):
        """Clamp drone position to stay within terrain extent and above surface."""
        H, W = self.terrain_shape
        x_max = (W - 1) * self.pixel_spacing_x
        y_max = (H - 1) * self.pixel_spacing_y
        pos[0] = np.clip(pos[0], 0, x_max)
        pos[1] = np.clip(pos[1], 0, y_max)
        terrain_z = self._get_terrain_z(pos[0], pos[1])
        if pos[2] < terrain_z:
            pos[2] = terrain_z
        return pos

    def _sync_drone_from_pos_for(self, obs, pos):
        """Update an observer's position and drone mesh from a 3D position."""
        pos = self._clamp_drone_pos(pos)
        obs.position = (float(pos[0]), float(pos[1]))
        obs.observer_elev = float(pos[2]) - self._get_terrain_z(
            pos[0], pos[1])
        if obs.observer_elev < 0:
            obs.observer_elev = 0.0
        self._update_observer_drone_for(obs)

        # Dynamically recalculate viewshed as the drone moves (throttled)
        if obs.viewshed_enabled:
            now = time.monotonic()
            if now - self._last_viewshed_time >= self._viewshed_recalc_interval:
                self._last_viewshed_time = now
                obs.viewshed_cache = None
                self._calculate_viewshed(quiet=True)

    def _check_terrain_reload(self):
        """Check if camera is near terrain edge and reload a new window if needed."""
        if self._terrain_loader is None:
            return

        now = time.time()
        if now - self._last_reload_time < self._reload_cooldown:
            return

        if self.position is None:
            return

        H, W = self.terrain_shape
        cam_col = self.position[0] / self.pixel_spacing_x
        cam_row = self.position[1] / self.pixel_spacing_y

        # Check if camera is within 20% of any edge
        margin_x = W * 0.2
        margin_y = H * 0.2
        near_edge = (cam_col < margin_x or cam_col > W - margin_x or
                     cam_row < margin_y or cam_row > H - margin_y)
        if not near_edge:
            return

        # Compute camera lon/lat from world position
        cam_lon = self._coord_origin_x + cam_col * self._coord_step_x
        cam_lat = self._coord_origin_y + cam_row * self._coord_step_y

        # Call the terrain loader
        new_raster = self._terrain_loader(cam_lon, cam_lat)
        if new_raster is None:
            self._last_reload_time = now
            return

        cam_z = self.position[2]

        # Extract coordinate metadata from new raster
        new_origin_x = float(new_raster.x.values[0])
        new_origin_y = float(new_raster.y.values[0])
        new_step_x = float(new_raster.x.values[1] - new_raster.x.values[0])
        new_step_y = float(new_raster.y.values[1] - new_raster.y.values[0])

        # Compute camera position in new window's pixel space
        new_col = (cam_lon - new_origin_x) / new_step_x
        new_row = (cam_lat - new_origin_y) / new_step_y

        # Replace rasters
        self._base_raster = new_raster
        self.raster = new_raster
        self._wind_terrain_np = None  # invalidate cached terrain

        # Update coordinate tracking
        self._coord_origin_x = new_origin_x
        self._coord_origin_y = new_origin_y
        self._coord_step_x = new_step_x
        self._coord_step_y = new_step_y

        # Recompute terrain stats
        new_H, new_W = new_raster.shape
        self.terrain_shape = (new_H, new_W)

        terrain_data = new_raster.data
        if hasattr(terrain_data, 'get'):
            terrain_np = terrain_data.get()
        else:
            terrain_np = np.asarray(terrain_data)

        # Detect ocean-fill (0-valued pixels) and replace with NaN
        ocean_fill = (terrain_np == 0.0) & ~np.isnan(terrain_np)
        if ocean_fill.sum() > terrain_np.size * 0.01:
            terrain_np[ocean_fill] = np.nan
            if hasattr(new_raster.data, 'get'):
                new_data = new_raster.data.copy()
                new_data[cp.asarray(ocean_fill)] = cp.nan
            else:
                new_data = new_raster.data.copy()
                new_data[ocean_fill] = np.nan
            self._base_raster = new_raster.copy(data=new_data)
            self.raster = self._base_raster

        ve = self.vertical_exaggeration
        self.elev_min = float(np.nanmin(terrain_np)) * ve
        self.elev_max = float(np.nanmax(terrain_np)) * ve
        self.elev_mean = float(np.nanmean(terrain_np)) * ve

        # Rebuild water mask
        floor_val = float(np.nanmin(terrain_np))
        floor_max = float(np.nanmax(terrain_np))
        eps = (floor_max - floor_val) * 1e-4 if floor_max > floor_val else 1e-6
        self._water_mask = (terrain_np <= floor_val + eps) | np.isnan(terrain_np)

        land_pixels = terrain_np[~self._water_mask]
        if land_pixels.size > 0:
            self._land_color_range = (float(np.nanmin(land_pixels)) * ve,
                                      float(np.nanmax(land_pixels)) * ve)

        # Clear terrain mesh cache (old window geometry is stale)
        self._terrain_mesh_cache.clear()
        self._baked_mesh_cache.clear()

        # Rebuild terrain mesh
        from . import mesh as mesh_mod

        H, W = new_H, new_W
        cache_key = (self.subsample_factor, self.mesh_type)

        if self.mesh_type == 'heightfield':
            if self.rtx is not None:
                self.rtx.add_heightfield_geometry(
                    'terrain', terrain_np, H, W,
                    spacing_x=self.pixel_spacing_x,
                    spacing_y=self.pixel_spacing_y,
                    ve=ve,
                )
            self._terrain_mesh_cache[cache_key] = (None, None, terrain_np.copy())
        else:
            if self.mesh_type == 'voxel':
                num_verts = H * W * 8
                num_tris = H * W * 12
                vertices = np.zeros(num_verts * 3, dtype=np.float32)
                indices = np.zeros(num_tris * 3, dtype=np.int32)
                base_elev = float(np.nanmin(terrain_np))
                mesh_mod.voxelate_terrain(vertices, indices, new_raster, scale=1.0,
                                          base_elevation=base_elev)
            else:
                num_verts = H * W
                num_tris = (H - 1) * (W - 1) * 2
                vertices = np.zeros(num_verts * 3, dtype=np.float32)
                indices = np.zeros(num_tris * 3, dtype=np.int32)
                mesh_mod.triangulate_terrain(vertices, indices, new_raster, scale=1.0)

            # Scale x,y to world units
            if self.pixel_spacing_x != 1.0 or self.pixel_spacing_y != 1.0:
                vertices[0::3] *= self.pixel_spacing_x
                vertices[1::3] *= self.pixel_spacing_y

            # Apply vertical exaggeration
            if ve != 1.0:
                vertices[2::3] *= ve

            # Cache the new mesh
            base_verts = vertices.copy()
            if ve != 1.0:
                base_verts[2::3] /= ve
            self._terrain_mesh_cache[cache_key] = (base_verts, indices.copy(), terrain_np.copy())

            # Replace terrain geometry
            if self.rtx is not None:
                self.rtx.add_geometry('terrain', vertices, indices)

        # Reposition camera in new window
        self.position = np.array([
            new_col * self.pixel_spacing_x,
            new_row * self.pixel_spacing_y,
            cam_z
        ], dtype=float)

        # Refresh minimap
        self._compute_minimap_background()

        self._last_reload_time = time.time()
        self._render_needed = True
        print(f"Terrain reloaded: center ({cam_lon:.4f}, {cam_lat:.4f}), "
              f"window {new_W}x{new_H}")

    def _tick(self):
        """Continuous render loop — process held keys and redraw (called by timer)."""
        if not self.running:
            return

        # Delta-time: scale movement relative to the old 20 Hz reference rate
        now = time.monotonic()
        dt = now - self._last_tick_time
        self._last_tick_time = now
        # Clamp to avoid huge jumps (e.g. after a stall or first frame)
        dt = min(dt, 0.1)
        dt_scale = dt / 0.05  # 0.05 = 1/20 Hz reference

        # Process held movement / look keys
        if self._held_keys:
            speed = self.move_speed * dt_scale
            look = self.look_speed * dt_scale

            # Get active observer (if any)
            active_obs = (self._observers.get(self._active_observer)
                          if self._active_observer else None)

            if (active_obs is not None and active_obs.drone_mode == '3rd'
                    and active_obs.drone_placed):
                # --- 3rd-person: WASD/IJKL fly the drone, camera stays ---
                front = self._get_drone_front_for(active_obs)
                right = self._get_drone_right_for(active_obs)

                obs_x, obs_y = active_obs.position
                terrain_z = self._get_terrain_z(obs_x, obs_y)
                drone_pos = np.array([obs_x, obs_y,
                                      terrain_z + active_obs.observer_elev],
                                     dtype=float)

                if 'w' in self._held_keys or 'up' in self._held_keys:
                    drone_pos += front * speed
                if 's' in self._held_keys or 'down' in self._held_keys:
                    drone_pos -= front * speed
                if 'a' in self._held_keys or 'left' in self._held_keys:
                    drone_pos -= right * speed
                if 'd' in self._held_keys or 'right' in self._held_keys:
                    drone_pos += right * speed
                if 'q' in self._held_keys or 'pageup' in self._held_keys:
                    drone_pos[2] += speed
                if 'e' in self._held_keys or 'pagedown' in self._held_keys:
                    drone_pos[2] -= speed

                if 'i' in self._held_keys:
                    active_obs.pitch = min(89, active_obs.pitch + look)
                if 'k' in self._held_keys:
                    active_obs.pitch = max(-89, active_obs.pitch - look)
                if 'j' in self._held_keys:
                    active_obs.yaw -= look
                if 'l' in self._held_keys:
                    active_obs.yaw += look

                self._sync_drone_from_pos_for(active_obs, drone_pos)

            else:
                # --- Normal / FPV: WASD moves camera ---
                front = self._get_front()
                right = self._get_right()

                if 'w' in self._held_keys or 'up' in self._held_keys:
                    self.position += front * speed
                if 's' in self._held_keys or 'down' in self._held_keys:
                    self.position -= front * speed
                if 'a' in self._held_keys or 'left' in self._held_keys:
                    self.position -= right * speed
                if 'd' in self._held_keys or 'right' in self._held_keys:
                    self.position += right * speed
                if 'q' in self._held_keys or 'pageup' in self._held_keys:
                    cam_up = np.cross(front, right)
                    cam_up /= (np.linalg.norm(cam_up) + 1e-8)
                    self.position += cam_up * speed
                if 'e' in self._held_keys or 'pagedown' in self._held_keys:
                    cam_up = np.cross(front, right)
                    cam_up /= (np.linalg.norm(cam_up) + 1e-8)
                    self.position -= cam_up * speed

                if 'i' in self._held_keys:
                    self.pitch = min(89, self.pitch + look)
                if 'k' in self._held_keys:
                    self.pitch = max(-89, self.pitch - look)
                if 'j' in self._held_keys:
                    self.yaw -= look
                if 'l' in self._held_keys:
                    self.yaw += look

                # In FPV, sync drone to camera
                if (active_obs is not None and active_obs.drone_mode == 'fpv'
                        and active_obs.drone_placed):
                    self._sync_drone_from_pos_for(active_obs, self.position)

            self._render_needed = True
        self._dt_scale = dt_scale

        self._check_terrain_reload()
        if self._chunk_manager is not None:
            if self._chunk_manager.update(self.position[0], self.position[1], self):
                self._geometry_colors_builder = self._accessor._build_geometry_colors_gpu
                self._render_needed = True
        # AO: keep accumulating samples when camera is stationary
        if (self.ao_enabled and not self._held_keys
                and not self._mouse_dragging
                and self._ao_frame_count < self._ao_max_frames):
            self._render_needed = True

        if self._render_needed:
            self._update_frame()
            self._render_needed = False
        elif self._wind_enabled and self._wind_particles is not None and self._pinned_frame is not None:
            # Wind is on but camera didn't move — skip the expensive ray
            # trace and just re-advect particles + re-composite overlays.
            self._update_wind_particles()
            self._composite_overlays()

    def _cycle_terrain_layer(self):
        """Cycle terrain color: elevation → overlay1 → overlay2 → ... → elevation.

        Only affects terrain coloring. Does NOT touch basemap or geometry.
        """
        if not self._terrain_layer_order:
            print("No terrain layers available")
            return

        self._terrain_layer_idx = (self._terrain_layer_idx + 1) % len(self._terrain_layer_order)
        layer_name = self._terrain_layer_order[self._terrain_layer_idx]

        if layer_name == 'elevation':
            self._active_color_data = None
            self._active_overlay_data = None
            print(f"Terrain: elevation")
        else:
            self._active_color_data = None
            self._active_overlay_data = self._overlay_layers[layer_name]
            alpha_pct = int(self._overlay_alpha * 100)
            print(f"Terrain: {layer_name} (alpha {alpha_pct}%, ,/. to adjust)")

        self._update_frame()

    def _cycle_basemap(self):
        """Cycle basemap: none → satellite → osm → none.

        Auto-creates XYZTileService on-the-fly if needed.
        """
        self._basemap_idx = (self._basemap_idx + 1) % len(self._basemap_options)
        provider = self._basemap_options[self._basemap_idx]

        if provider == 'none':
            self._tiles_enabled = False
            print("Basemap: none")
        else:
            from .tiles import XYZTileService
            # Create or switch tile service
            if self._tile_service is not None:
                if self._tile_service.provider_name != provider:
                    self._tile_service.shutdown()
                    self._tile_service = XYZTileService(
                        url_template=provider, raster=self._base_raster,
                    )
                    self._tile_service.fetch_visible_tiles()
            else:
                self._tile_service = XYZTileService(
                    url_template=provider, raster=self._base_raster,
                )
                self._tile_service.fetch_visible_tiles()
            self._tiles_enabled = True
            print(f"Basemap: {provider}")

        self._update_frame()

    def _cycle_geometry_layer(self):
        """Cycle geometry visibility: none → all → group1 → group2 → ... → none.

        Uses rtx.set_geometry_visible() to show/hide geometry groups.
        """
        if self.rtx is None or len(self._geometry_layer_order) <= 2:
            # Only 'none' and 'all' with no actual groups
            if self.rtx is None:
                print("No geometries in scene")
                return

        self._geometry_layer_idx = (self._geometry_layer_idx + 1) % len(self._geometry_layer_order)
        layer_name = self._geometry_layer_order[self._geometry_layer_idx]

        if layer_name == 'none':
            # Hide all non-terrain geometries
            for geom_id in self._all_geometries:
                if geom_id != 'terrain':
                    self.rtx.set_geometry_visible(geom_id, False)
            print("Geometry: none")

        elif layer_name == 'all':
            # Show all geometries
            for geom_id in self._all_geometries:
                self.rtx.set_geometry_visible(geom_id, True)
            print("Geometry: all")

        else:
            # Show only this geometry group + terrain
            visible_count = 0
            for geom_id in self._all_geometries:
                parts = geom_id.rsplit('_', 1)
                base_name = parts[0] if len(parts) == 2 and parts[1].isdigit() else geom_id
                if base_name == layer_name or geom_id == layer_name or geom_id == 'terrain':
                    self.rtx.set_geometry_visible(geom_id, True)
                    visible_count += 1
                else:
                    self.rtx.set_geometry_visible(geom_id, False)
            print(f"Geometry: {layer_name} ({visible_count} visible)")

        self._current_geom_idx = 0
        self._update_frame()

    def _jump_to_geometry(self, direction):
        """Jump camera to next/previous geometry in current layer.

        Parameters
        ----------
        direction : int
            1 for next, -1 for previous.
        """
        if self.rtx is None:
            print("No geometries in scene")
            return

        # Get current geometry layer name
        mode = self._geometry_layer_order[self._geometry_layer_idx]

        if mode == 'none':
            print("No geometry layer selected. Press N to select one.")
            return

        if mode == 'all':
            # Cycle through all geometry positions across all groups
            all_positions = []
            for layer_name, positions in sorted(self._layer_positions.items()):
                all_positions.extend(positions)
            if not all_positions:
                print("No geometry positions available")
                return
            self._current_geom_idx = (self._current_geom_idx + direction) % len(all_positions)
            x, y, z, geom_id = all_positions[self._current_geom_idx]
            yaw_rad = np.radians(self.yaw)
            forward_level = np.array([np.cos(yaw_rad), np.sin(yaw_rad), 0], dtype=np.float32)
            self.position = np.array([
                x - forward_level[0] * 100,
                y - forward_level[1] * 100,
                z + 50
            ], dtype=np.float32)
            self.pitch = -15.0
            print(f"Jumped to {geom_id} ({self._current_geom_idx + 1}/{len(all_positions)})")
            print(f"  Position: ({x:.0f}, {y:.0f}, {z:.0f})")
            self._update_frame()
            return

        # Get positions for current layer
        if mode not in self._layer_positions:
            print(f"No positions for layer: {mode}")
            return

        positions = self._layer_positions[mode]
        if not positions:
            print(f"No geometries in layer: {mode}")
            return

        # Cycle through geometries
        self._current_geom_idx = (self._current_geom_idx + direction) % len(positions)
        x, y, z, geom_id = positions[self._current_geom_idx]

        # Position camera at geometry location, slightly above and behind
        # Calculate offset based on current viewing direction
        height_offset = 50  # Height above geometry
        distance_back = 100  # Distance behind geometry

        # Get current forward direction (but level, no pitch)
        yaw_rad = np.radians(self.yaw)
        forward_level = np.array([np.cos(yaw_rad), np.sin(yaw_rad), 0], dtype=np.float32)

        # Position camera behind and above the geometry
        self.position = np.array([
            x - forward_level[0] * distance_back,
            y - forward_level[1] * distance_back,
            z + height_offset
        ], dtype=np.float32)

        # Look at the geometry
        self.pitch = -15.0  # Look slightly down

        print(f"Jumped to {geom_id} ({self._current_geom_idx + 1}/{len(positions)})")
        print(f"  Position: ({x:.0f}, {y:.0f}, {z:.0f})")
        self._update_frame()

    def _get_terrain_z(self, world_x, world_y):
        """Sample terrain elevation at a world-coordinate position."""
        H, W = self.terrain_shape
        col = int(np.clip(world_x / self.pixel_spacing_x, 0, W - 1))
        row = int(np.clip(world_y / self.pixel_spacing_y, 0, H - 1))
        terrain_data = self.raster.data
        if hasattr(terrain_data, 'get'):
            z = float(terrain_data[row, col].get())
        else:
            z = float(terrain_data[row, col])
        if np.isnan(z):
            z = 0.0
        return z

    def _load_drone_parts(self):
        """Load drone GLB split by material, returning per-part geometry + color."""
        import os
        drone_path = os.path.join(
            os.path.dirname(__file__), '..', 'examples', 'models', 'drone.glb'
        )
        if not os.path.exists(drone_path):
            print(f"  Drone model not found at {drone_path}")
            return []

        try:
            import trimesh
        except ImportError:
            print("  trimesh required for drone model")
            return []

        scene = trimesh.load(drone_path)
        if not isinstance(scene, trimesh.Scene):
            return []

        # First pass: collect sub-meshes and compute shared bounding box
        raw_parts = []
        all_verts = []
        for name, geom in scene.geometry.items():
            if not isinstance(geom, trimesh.Trimesh):
                continue
            verts = geom.vertices.copy().astype(np.float32)
            faces = geom.faces.copy().astype(np.int32)
            # Swap Y/Z (Y-up → Z-up)
            verts[:, [1, 2]] = verts[:, [2, 1]]
            # Extract material base color
            color = (0.6, 0.6, 0.6)  # fallback grey
            if hasattr(geom, 'visual') and hasattr(geom.visual, 'material'):
                mat = geom.visual.material
                if hasattr(mat, 'baseColorFactor') and mat.baseColorFactor is not None:
                    c = mat.baseColorFactor
                    color = (c[0] / 255.0, c[1] / 255.0, c[2] / 255.0)
            raw_parts.append((verts, faces, color))
            all_verts.append(verts)

        if not raw_parts:
            return []

        # Compute shared centre and Z-base across all parts
        combined = np.concatenate(all_verts, axis=0)
        center_xy = (combined[:, :2].min(axis=0) + combined[:, :2].max(axis=0)) / 2
        z_min = combined[:, 2].min()

        # Second pass: apply shared transform
        parts = []
        for verts, faces, color in raw_parts:
            verts[:, 0] -= center_xy[0]
            verts[:, 1] -= center_xy[1]
            verts[:, 2] -= z_min
            parts.append((verts.flatten(), faces.flatten(), color))
        return parts

    def _update_observer_drone_for(self, obs):
        """Place or update the drone mesh at an observer's position."""
        if obs.position is None or self.rtx is None:
            return

        from .mesh import make_transform

        # Lazy-load drone parts once (shared across all observers)
        if self._shared_drone_parts is None:
            self._shared_drone_parts = self._load_drone_parts()
            if not self._shared_drone_parts:
                return

        obs_x, obs_y = obs.position
        terrain_z = self._get_terrain_z(obs_x, obs_y)
        obs_z = terrain_z + obs.observer_elev

        # Scale drone to ~0.05× pixel_spacing so it's visible but not huge
        drone_scale = 0.0125 * max(self.pixel_spacing_x, self.pixel_spacing_y)

        transform = make_transform(x=obs_x, y=obs_y, z=obs_z, scale=drone_scale)

        # Tint base colors toward observer slot color
        slot_color = obs.color
        for i, (verts, idxs, base_color) in enumerate(self._shared_drone_parts):
            gid = obs.geometry_id(i)
            # Mix: 50% base + 50% slot tint
            tinted = tuple(0.5 * base_color[c] + 0.5 * slot_color[c]
                           for c in range(3))
            if obs.drone_placed:
                self.rtx.update_transform(gid, transform)
            else:
                self.rtx.add_geometry(gid, verts, idxs, transform=transform)

        # Set geometry colors (needs the accessor's color dict)
        if not obs.drone_placed:
            builder = getattr(self, '_geometry_colors_builder', None)
            if builder is not None:
                acc = getattr(builder, '__self__', None)
                if acc is not None and hasattr(acc, '_geometry_colors'):
                    for i, (_, _, base_color) in enumerate(self._shared_drone_parts):
                        if self._drone_glow:
                            color = (*slot_color, 1.8)
                        else:
                            color = tuple(0.5 * base_color[c] + 0.5 * slot_color[c]
                                          for c in range(3))
                        acc._geometry_colors[obs.geometry_id(i)] = color
                    acc._geometry_colors_dirty = True

        obs.drone_placed = True

    def _apply_drone_glow(self):
        """Toggle emissive glow on/off for all placed drone geometries."""
        builder = getattr(self, '_geometry_colors_builder', None)
        if builder is None:
            return
        acc = getattr(builder, '__self__', None)
        if acc is None or not hasattr(acc, '_geometry_colors'):
            return
        parts = self._shared_drone_parts
        if not parts:
            return
        changed = False
        for obs in self._observers.values():
            if not obs.drone_placed:
                continue
            slot_color = obs.color
            for i, (_, _, base_color) in enumerate(parts):
                gid = obs.geometry_id(i)
                if self._drone_glow:
                    acc._geometry_colors[gid] = (*slot_color, 1.8)
                else:
                    acc._geometry_colors[gid] = tuple(
                        0.5 * base_color[c] + 0.5 * slot_color[c]
                        for c in range(3))
            changed = True
        if changed:
            acc._geometry_colors_dirty = True
            self._update_frame()

    def _set_drone_visibility_for(self, obs, visible):
        """Show or hide all drone sub-mesh geometries for an observer."""
        if obs.drone_placed and self.rtx is not None:
            for i in range(len(self._shared_drone_parts or [])):
                self.rtx.set_geometry_visible(obs.geometry_id(i), visible)

    def _cycle_drone_mode_for(self, obs):
        """Cycle drone mode for observer: off → 3rd person → FPV → off."""
        if obs.position is None:
            print(f"Observer {obs.slot} has no position.")
            return

        if obs.drone_mode == 'off':
            # --- Enter 3rd person ---
            obs.saved_camera = (
                self.position.copy(),
                float(self.yaw),
                float(self.pitch),
            )
            obs.yaw = float(self.yaw)
            obs.pitch = 0.0
            obs.drone_mode = '3rd'
            print(f"Observer {obs.slot} DRONE 3RD PERSON: ON")

        elif obs.drone_mode == '3rd':
            # --- 3rd person → FPV ---
            obs_x, obs_y = obs.position
            terrain_z = self._get_terrain_z(obs_x, obs_y)
            obs_z = terrain_z + obs.observer_elev
            self.position = np.array([obs_x, obs_y, obs_z], dtype=float)
            self.yaw = obs.yaw
            self.pitch = obs.pitch
            self._set_drone_visibility_for(obs, False)
            obs.drone_mode = 'fpv'
            print(f"Observer {obs.slot} DRONE FPV: ON")

        else:
            # --- FPV → off ---
            self._sync_drone_from_pos_for(obs, self.position)
            self._set_drone_visibility_for(obs, True)
            if obs.saved_camera is not None:
                self.position = obs.saved_camera[0]
                self.yaw = obs.saved_camera[1]
                self.pitch = obs.saved_camera[2]
                obs.saved_camera = None
            obs.drone_mode = 'off'
            print(f"Observer {obs.slot} DRONE: OFF")

        self._update_frame()

    def _snap_to_observer(self, obs):
        """Snap external camera to look at an observer's drone from nearby."""
        if obs.position is None:
            print(f"Observer {obs.slot} has no position.")
            return
        if obs.drone_mode == 'fpv':
            return

        obs_x, obs_y = obs.position
        terrain_z = self._get_terrain_z(obs_x, obs_y)
        obs_z = terrain_z + obs.observer_elev

        spacing = max(self.pixel_spacing_x, self.pixel_spacing_y)
        offset = spacing * 8.0
        dx = self.position[0] - obs_x
        dy = self.position[1] - obs_y
        dist_xy = np.sqrt(dx * dx + dy * dy)
        if dist_xy > 1e-6:
            dx /= dist_xy
            dy /= dist_xy
        else:
            dx, dy = 1.0, 0.0

        self.position = np.array([
            obs_x + dx * offset,
            obs_y + dy * offset,
            obs_z + spacing * 3.0,
        ], dtype=float)

        to_drone = np.array([obs_x - self.position[0],
                             obs_y - self.position[1],
                             obs_z - self.position[2]])
        to_drone /= (np.linalg.norm(to_drone) + 1e-8)
        self.yaw = float(np.degrees(np.arctan2(to_drone[1], to_drone[0])))
        self.pitch = float(np.degrees(np.arcsin(np.clip(to_drone[2], -1, 1))))

        print(f"Snapped to observer {obs.slot} at ({obs_x:.0f}, {obs_y:.0f})")
        self._update_frame()

    def _place_observer_at(self, obs, x=None, y=None):
        """Move an observer to a position (defaults to camera XY).

        Parameters
        ----------
        obs : Observer
            The observer to position.
        x, y : float, optional
            World coordinates. If None, use current camera position.
        """
        H, W = self.terrain_shape
        cam_x = x if x is not None else self.position[0]
        cam_y = y if y is not None else self.position[1]

        max_x = (W - 1) * self.pixel_spacing_x
        max_y = (H - 1) * self.pixel_spacing_y

        obs_x = float(np.clip(cam_x, 0, max_x))
        obs_y = float(np.clip(cam_y, 0, max_y))

        obs.position = (obs_x, obs_y)
        self._update_observer_drone_for(obs)

        print(f"Observer {obs.slot} placed at ({obs_x:.0f}, {obs_y:.0f})")

        if obs.viewshed_enabled:
            self._calculate_viewshed(quiet=True)

        self._update_frame()

    def _select_or_create_observer(self, slot):
        """Handle number key 1-8: select/create/deselect observer slot."""
        if self._active_observer == slot:
            # Deselect — exit FPV first if active
            obs = self._observers.get(slot)
            if obs is not None and obs.drone_mode == 'fpv':
                self._exit_fpv_for(obs)
            self._active_observer = None
            self.viewshed_enabled = False
            self._viewshed_cache = None
            print(f"Observer {slot}: deselected")
            self._update_frame()
            return

        # If switching away from an FPV observer, exit FPV first
        if self._active_observer is not None:
            prev_obs = self._observers.get(self._active_observer)
            if prev_obs is not None and prev_obs.drone_mode == 'fpv':
                self._exit_fpv_for(prev_obs)

        if slot in self._observers:
            # Select existing — auto-enter FPV
            self._active_observer = slot
            obs = self._observers[slot]
            # Sync viewer-level viewshed from this observer
            self.viewshed_enabled = obs.viewshed_enabled
            self._viewshed_cache = obs.viewshed_cache
            # Enter FPV: save camera, snap to observer, hide drone
            obs.saved_camera = (
                self.position.copy(),
                float(self.yaw),
                float(self.pitch),
            )
            obs_x, obs_y = obs.position
            terrain_z = self._get_terrain_z(obs_x, obs_y)
            obs_z = terrain_z + obs.observer_elev
            self.position = np.array([obs_x, obs_y, obs_z], dtype=float)
            self.yaw = obs.yaw
            self.pitch = obs.pitch
            self._set_drone_visibility_for(obs, False)
            obs.drone_mode = 'fpv'
            print(f"Observer {slot}: FPV at ({obs.position[0]:.0f}, {obs.position[1]:.0f})")
        else:
            # Create new just in front of camera, matching altitude and angle
            front = self._get_front()
            spacing = max(self.pixel_spacing_x, self.pixel_spacing_y)
            offset = spacing * 3  # A few pixels in front
            obs_x = self.position[0] + front[0] * offset
            obs_y = self.position[1] + front[1] * offset
            # Clamp to terrain bounds
            H, W = self.terrain_shape
            obs_x = float(np.clip(obs_x, 0, (W - 1) * self.pixel_spacing_x))
            obs_y = float(np.clip(obs_y, 0, (H - 1) * self.pixel_spacing_y))
            terrain_z = self._get_terrain_z(obs_x, obs_y)
            cam_elev = max(0.0, self.position[2] - terrain_z)
            obs = Observer(slot, position=(obs_x, obs_y),
                           observer_elev=cam_elev)
            obs.yaw = self.yaw
            obs.pitch = self.pitch
            self._observers[slot] = obs
            self._active_observer = slot
            self._update_observer_drone_for(obs)
            print(f"Observer {slot} placed at ({obs_x:.0f}, {obs_y:.0f}), "
                  f"h={cam_elev:.3f}, yaw={self.yaw:.0f}, pitch={self.pitch:.0f}")
            if obs.viewshed_enabled:
                self._calculate_viewshed(quiet=True)
            self._update_frame()
            return

        self._update_frame()

    def _exit_fpv_for(self, obs):
        """Exit FPV mode for an observer, restoring camera."""
        if obs.drone_mode != 'fpv':
            return
        self._sync_drone_from_pos_for(obs, self.position)
        self._set_drone_visibility_for(obs, True)
        if obs.saved_camera is not None:
            self.position = obs.saved_camera[0]
            self.yaw = obs.saved_camera[1]
            self.pitch = obs.saved_camera[2]
            obs.saved_camera = None
        obs.drone_mode = 'off'

    def _clear_observer_slot(self, slot):
        """Remove a single observer and its geometry."""
        obs = self._observers.get(slot)
        if obs is None:
            return

        # Stop tour if running
        obs.stop_tour()

        # Exit drone mode (restore camera if FPV)
        if obs.drone_mode != 'off':
            if obs.drone_mode == 'fpv':
                self._set_drone_visibility_for(obs, True)
            if obs.saved_camera is not None:
                self.position = obs.saved_camera[0]
                self.yaw = obs.saved_camera[1]
                self.pitch = obs.saved_camera[2]
                obs.saved_camera = None
            obs.drone_mode = 'off'

        # Remove drone geometry
        if obs.drone_placed and self.rtx is not None:
            n = len(self._shared_drone_parts) if self._shared_drone_parts else 0
            builder = getattr(self, '_geometry_colors_builder', None)
            acc = getattr(builder, '__self__', None) if builder else None
            for i in range(n):
                gid = obs.geometry_id(i)
                self.rtx.remove_geometry(gid)
                if acc is not None and hasattr(acc, '_geometry_colors'):
                    acc._geometry_colors.pop(gid, None)
            if acc is not None and hasattr(acc, '_geometry_colors_dirty'):
                acc._geometry_colors_dirty = True
            obs.drone_placed = False

        del self._observers[slot]
        if self._active_observer == slot:
            self._active_observer = None

        print(f"Observer {slot} removed")
        self._update_frame()

    def _clear_all_observers(self):
        """Kill all observers — stop tours, exit drone modes, remove geometry."""
        # Find if any observer is in FPV and restore camera
        for obs in self._observers.values():
            if obs.drone_mode == 'fpv' and obs.saved_camera is not None:
                self.position = obs.saved_camera[0]
                self.yaw = obs.saved_camera[1]
                self.pitch = obs.saved_camera[2]
                break  # Only one can be in FPV at a time

        for slot in list(self._observers.keys()):
            obs = self._observers[slot]
            obs.stop_tour()
            # Remove drone geometry
            if obs.drone_placed and self.rtx is not None:
                n = len(self._shared_drone_parts) if self._shared_drone_parts else 0
                builder = getattr(self, '_geometry_colors_builder', None)
                acc = getattr(builder, '__self__', None) if builder else None
                for i in range(n):
                    gid = obs.geometry_id(i)
                    self.rtx.remove_geometry(gid)
                    if acc is not None and hasattr(acc, '_geometry_colors'):
                        acc._geometry_colors.pop(gid, None)
                if acc is not None and hasattr(acc, '_geometry_colors_dirty'):
                    acc._geometry_colors_dirty = True

        self._observers.clear()
        self._active_observer = None
        self.viewshed_enabled = False
        self._viewshed_cache = None
        print("All observers removed")
        self._update_frame()

    def _calculate_viewshed(self, quiet=False):
        """Calculate viewshed from the placed observer position.

        Uses GPU ray tracing to compute visibility from the fixed observer.
        Observer position is in world coordinates; this method converts to
        pixel indices for the viewshed calculation.

        Parameters
        ----------
        quiet : bool
            If True, suppress verbose output (used during dynamic updates).
        """
        from .analysis.viewshed import _viewshed_rt

        # Get observer position: from _calculate_viewshed_for compat bridge,
        # or from the active observer
        obs_pos = getattr(self, '_observer_position_compat', None)
        if obs_pos is None:
            # Try active observer
            obs = self._observers.get(self._active_observer) if self._active_observer else None
            if obs is not None:
                obs_pos = obs.position
        if obs_pos is None:
            if not quiet:
                print("No observer placed. Press 1-8 to create one.")
            return None

        world_x, world_y = obs_pos
        H, W = self.terrain_shape

        # Convert world coords to pixel indices
        px_x = world_x / self.pixel_spacing_x
        px_y = world_y / self.pixel_spacing_y

        # Validate coordinates are within terrain bounds (in pixel space)
        if px_x < 0 or px_x >= W or px_y < 0 or px_y >= H:
            if not quiet:
                print(f"Observer position pixel ({px_x:.1f}, {px_y:.1f}) outside terrain bounds")
            return None

        if not quiet:
            print(f"Computing viewshed... (observer height: {self.viewshed_observer_elev:.3f})")
            print(f"  Raster shape: {self.raster.shape}, pixel_spacing: ({self.pixel_spacing_x:.1f}, {self.pixel_spacing_y:.1f})")

        try:
            # Use the scene's existing RTX which includes all geometries
            # (terrain, buildings, etc.) so viewshed rays are occluded by them.
            rtx = self.rtx
            if not quiet:
                print(f"  Using scene RTX ({rtx.get_geometry_count()} geometries)")

            # Two-phase visibility for realistic viewshed:
            #  Phase 1 (primary rays): only terrain visible — rays hit ground,
            #           not building rooftops.
            #  Phase 2 (occlusion rays): terrain + structures visible — buildings
            #           block line-of-sight from ground to observer.
            #
            # The between_traces_cb callback switches from phase 1 → 2.
            saved_visibility = {}
            non_terrain_ids = []
            for geom_id in self._all_geometries:
                entry = rtx._geom_state.gas_entries.get(geom_id)
                if entry is not None:
                    saved_visibility[geom_id] = entry.visible
                    if geom_id != 'terrain':
                        non_terrain_ids.append(geom_id)
                        # Phase 1: hide non-terrain so primary rays hit ground
                        rtx.set_geometry_visible(geom_id, False)
                    elif not entry.visible:
                        rtx.set_geometry_visible(geom_id, True)

            # Hide all observer drones so they don't block viewshed
            for obs in self._observers.values():
                if obs.drone_placed and self._shared_drone_parts:
                    for i in range(len(self._shared_drone_parts)):
                        gid = obs.geometry_id(i)
                        saved_visibility[gid] = True
                        rtx.set_geometry_visible(gid, False)

            def _enable_structures():
                """Callback: make structures visible for occlusion trace."""
                for gid in non_terrain_ids:
                    rtx.set_geometry_visible(gid, True)

            # Convert pixel indices to raster coords
            y_coords = self.raster.indexes.get('y').values
            x_coords = self.raster.indexes.get('x').values

            # Clamp to valid range and get actual coord values
            x_idx = int(np.clip(px_x, 0, W - 1))
            y_idx = int(np.clip(px_y, 0, H - 1))
            x_coord = x_coords[x_idx] if x_idx < len(x_coords) else x_coords[-1]
            y_coord = y_coords[y_idx] if y_idx < len(y_coords) else y_coords[-1]

            if not quiet:
                print(f"  Observer at raster coords: ({x_coord:.1f}, {y_coord:.1f})")

            viewshed = _viewshed_rt(
                self.raster, rtx,
                x_coord, y_coord,
                self.viewshed_observer_elev,
                self.viewshed_target_elev,
                pixel_spacing_x=self.pixel_spacing_x,
                pixel_spacing_y=self.pixel_spacing_y,
                between_traces_cb=_enable_structures,
            )

            # Restore original visibility state
            for geom_id, was_visible in saved_visibility.items():
                rtx.set_geometry_visible(geom_id, was_visible)

            # Calculate coverage percentage
            vis_data = viewshed.data
            if hasattr(vis_data, 'get'):
                vis_np = vis_data.get()
            else:
                vis_np = vis_data
            visible_cells = np.sum(vis_np >= 0)
            total_cells = vis_np.size
            self._viewshed_coverage = 100.0 * visible_cells / total_cells

            # Cache result
            self._viewshed_cache = viewshed
            self._last_viewshed_time = time.monotonic()

            if not quiet:
                print(f"  Coverage: {self._viewshed_coverage:.1f}% terrain visible")
            return viewshed

        except Exception as e:
            if not quiet:
                import traceback
                print(f"Viewshed calculation failed: {e}")
                traceback.print_exc()
            return None

    def _apply_viewshed_overlay(self, img):
        """Apply viewshed overlay to rendered image.

        Visible areas get a teal glow, invisible areas remain unchanged.

        Parameters
        ----------
        img : ndarray
            RGB image array (H, W, 3) with values 0-255.

        Returns
        -------
        ndarray
            Image with viewshed overlay applied.
        """
        if self._viewshed_cache is None:
            return img

        vis_data = self._viewshed_cache.data
        if hasattr(vis_data, 'get'):
            vis_np = vis_data.get()
        else:
            vis_np = np.asarray(vis_data)

        # Resize viewshed to match render resolution
        scale_y = img.shape[0] / vis_np.shape[0]
        scale_x = img.shape[1] / vis_np.shape[1]
        if scale_y != 1.0 or scale_x != 1.0:
            try:
                from scipy.ndimage import zoom
                vis_resized = zoom(vis_np, (scale_y, scale_x), order=0)
            except ImportError:
                # Fallback: use cv2 for resizing
                try:
                    import cv2
                    vis_resized = cv2.resize(vis_np, (img.shape[1], img.shape[0]),
                                             interpolation=cv2.INTER_NEAREST)
                except ImportError:
                    # Last resort: nearest neighbor with numpy
                    y_idx = np.linspace(0, vis_np.shape[0]-1, img.shape[0]).astype(int)
                    x_idx = np.linspace(0, vis_np.shape[1]-1, img.shape[1]).astype(int)
                    vis_resized = vis_np[np.ix_(y_idx, x_idx)]
        else:
            vis_resized = vis_np

        # Create result image
        img_float = img.astype(np.float32)
        result = img_float.copy()

        # Visible areas: apply teal glow
        # Teal color: RGB(0, 200, 200) - cyan/teal
        visible_mask = vis_resized >= 0

        # Intensity based on viewing angle (0-90 degrees)
        # Lower angle = more direct view = brighter glow
        vis_angles = np.clip(vis_resized, 0, 90)
        glow_intensity = 1.0 - (vis_angles / 90.0)  # 1.0 at 0°, 0.0 at 90°
        glow_intensity = np.clip(glow_intensity, 0.4, 1.0)  # Min glow level

        # Teal glow color
        teal_r, teal_g, teal_b = 0, 220, 210  # Bright teal/cyan

        # Apply glow only to visible areas using additive blending
        alpha = self.viewshed_opacity
        for c, teal_val in enumerate([teal_r, teal_g, teal_b]):
            channel = result[:, :, c]
            glow = glow_intensity * teal_val * alpha
            channel[visible_mask] = np.clip(
                channel[visible_mask] * (1 - alpha * 0.3) + glow[visible_mask],
                0, 255
            )

        return result.astype(np.uint8)

    def _toggle_viewshed(self):
        """Toggle viewshed overlay on/off for the active observer."""
        obs = self._observers.get(self._active_observer) if self._active_observer else None
        if obs is None:
            print("No observer selected. Press 1-8 to select/create one.")
            return

        obs.viewshed_enabled = not obs.viewshed_enabled

        if obs.viewshed_enabled:
            print("Calculating viewshed...")
            # Temporarily set position/elev for _calculate_viewshed
            viewshed = self._calculate_viewshed_for(obs)
            if viewshed is None:
                obs.viewshed_enabled = False
                print("Viewshed: OFF (calculation failed)")
            else:
                self.viewshed_enabled = True
                self._viewshed_cache = obs.viewshed_cache
                print(f"Viewshed: ON ({self._viewshed_coverage:.1f}% coverage)")
        else:
            print("Viewshed: OFF")
            self.viewshed_enabled = False
            self._viewshed_cache = None

        self._update_frame()

    def _calculate_viewshed_for(self, obs, quiet=False):
        """Calculate viewshed using an observer's position/elevation."""
        # Temporarily bridge to existing _calculate_viewshed by setting compat state
        old_pos = getattr(self, '_observer_position_compat', None)
        old_elev = self.viewshed_observer_elev
        self._observer_position_compat = obs.position
        self.viewshed_observer_elev = obs.observer_elev
        result = self._calculate_viewshed(quiet=quiet)
        obs.viewshed_cache = self._viewshed_cache
        self._observer_position_compat = old_pos
        self.viewshed_observer_elev = old_elev
        return result

    def _adjust_observer_elevation(self, delta):
        """Adjust active observer's elevation."""
        obs = self._observers.get(self._active_observer) if self._active_observer else None
        if obs is None:
            print("No observer selected. Press 1-8 first.")
            return

        obs.observer_elev = max(0, obs.observer_elev + delta)
        print(f"Observer {obs.slot} height: {obs.observer_elev:.3f}")

        self._update_observer_drone_for(obs)

        if obs.viewshed_enabled:
            obs.viewshed_cache = None
            self._calculate_viewshed_for(obs)
            self._viewshed_cache = obs.viewshed_cache
            self._update_frame()

    def _save_screenshot(self):
        """Save current view as PNG image.

        When AO is enabled, renders multiple accumulated frames for
        high-quality output with smooth AA, soft shadows, AO, and DOF.
        """
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"rtxpy_screenshot_{timestamp}.png"

        # Pass viewshed data directly to render if enabled
        viewshed_data = None
        observer_pos = None
        active_obs = (self._observers.get(self._active_observer)
                      if self._active_observer else None)
        if active_obs is not None and active_obs.position is not None:
            observer_pos = active_obs.position
        if self.viewshed_enabled and self._viewshed_cache is not None:
            viewshed_data = self._viewshed_cache

        # Get tile texture for screenshot if enabled
        rgb_texture = None
        if self._tiles_enabled and self._tile_service is not None:
            rgb_texture = self._tile_service.get_gpu_texture()
            if rgb_texture is not None and self.subsample_factor > 1:
                f = self.subsample_factor
                rgb_texture = rgb_texture[::f, ::f, :]

        # Build geometry colors GPU LUT if a builder is available
        geometry_colors = None
        builder = getattr(self, '_geometry_colors_builder', None)
        if builder is not None:
            geometry_colors = builder()

        from .analysis import render as render_func

        # Common render kwargs
        render_kwargs = dict(
            camera_position=tuple(self.position),
            look_at=tuple(self._get_look_at()),
            fov=self.fov,
            width=self.width,
            height=self.height,
            sun_azimuth=self.sun_azimuth,
            sun_altitude=self.sun_altitude,
            shadows=self.shadows,
            ambient=self.ambient,
            colormap=self.colormap,
            rtx=self.rtx,
            viewshed_data=viewshed_data,
            viewshed_opacity=self.viewshed_opacity,
            observer_position=observer_pos,
            pixel_spacing_x=self.pixel_spacing_x,
            pixel_spacing_y=self.pixel_spacing_y,
            color_stretch=self.color_stretch,
            color_range=self._land_color_range,
            rgb_texture=rgb_texture,
            overlay_data=self._active_overlay_data,
            overlay_alpha=self._overlay_alpha,
            geometry_colors=geometry_colors,
        )

        # Accumulated multi-frame screenshot when AO is enabled
        num_frames = 64 if self.ao_enabled else 1

        if num_frames > 1:
            import cupy
            from .analysis.render import _bloom, _tone_map_aces, _render_buffers
            print(f"Rendering {num_frames} accumulated frames...", end='', flush=True)

            # DOF params
            if self.dof_enabled:
                dof_aperture = self._dof_aperture
                dof_focal = self._dof_focal_distance
            else:
                dof_aperture = 0.0
                dof_focal = 0.0

            d_accum = None
            for i in range(num_frames):
                frame_seed = i + 1
                d_frame = render_func(
                    self.raster,
                    **render_kwargs,
                    ao_samples=self._ao_samples_per_frame,
                    ao_radius=self.ao_radius,
                    ao_seed=i,
                    gi_intensity=self.gi_intensity,
                    gi_bounces=self.gi_bounces,
                    frame_seed=frame_seed,
                    sun_angle=1.5,
                    aperture=dof_aperture,
                    focal_distance=dof_focal,
                    bloom=False,
                    tone_map=False,
                    _return_gpu=True,
                )
                if d_accum is None:
                    d_accum = d_frame.astype(cupy.float32)
                else:
                    d_accum += d_frame
            d_accum /= num_frames

            # Apply bloom and tone mapping once to the averaged result
            bufs = _render_buffers
            if bufs.bloom_temp is not None:
                _bloom(d_accum, bufs.bloom_temp, bufs.bloom_scratch)
            _tone_map_aces(d_accum)

            img = cupy.asnumpy(d_accum)
            print(" done")
        else:
            img = render_func(self.raster, **render_kwargs)

        # Convert from float [0-1] to uint8 [0-255]
        img_uint8 = (np.clip(img, 0, 1) * 255).astype(np.uint8)

        from PIL import Image
        Image.fromarray(img_uint8).save(filename)

        print(f"Screenshot saved: {filename}")

    def _render_frame(self):
        """Render a frame using rtxpy."""
        from .analysis import render

        # Always show observer orb when placed; viewshed overlay only when enabled
        viewshed_data = None
        observer_pos = None
        active_obs = (self._observers.get(self._active_observer)
                      if self._active_observer else None)
        if active_obs is not None and active_obs.position is not None:
            observer_pos = active_obs.position
        if self.viewshed_enabled:
            if self._viewshed_cache is not None:
                viewshed_data = self._viewshed_cache
            else:
                # Debug: viewshed enabled but no cache
                if self.frame_count % 100 == 0:  # Only print occasionally
                    print(f"[DEBUG] Viewshed enabled but cache is None")

        # Get GPU texture from tile service if enabled
        rgb_texture = None
        if self._tiles_enabled and self._tile_service is not None:
            rgb_texture = self._tile_service.get_gpu_texture()
            # Tile texture is always at base resolution — stride-subsample
            # to match the current (possibly subsampled) raster
            if rgb_texture is not None and self.subsample_factor > 1:
                f = self.subsample_factor
                rgb_texture = rgb_texture[::f, ::f, :]

        # Build geometry colors GPU LUT if a builder is available
        geometry_colors = None
        builder = getattr(self, '_geometry_colors_builder', None)
        if builder is not None:
            geometry_colors = builder()

        # AO parameters: multiple samples per frame for smooth early results,
        # with progressive accumulation across frames for further refinement
        ao_samples = self._ao_samples_per_frame if self.ao_enabled else 0
        ao_seed = self._ao_frame_count if self.ao_enabled else 0

        # When progressive accumulation is active, pass frame seed for AA + soft shadows + DOF
        frame_seed = self._ao_frame_count + 1 if self.ao_enabled else 0

        # Depth of field (requires progressive accumulation via AO)
        if self.dof_enabled and self.ao_enabled:
            dof_aperture = self._dof_aperture
            dof_focal = self._dof_focal_distance
        else:
            dof_aperture = 0.0
            dof_focal = 0.0

        # When progressive AO accumulation or denoising is active, defer
        # bloom and tone mapping until after averaging / denoising.  Both
        # are non-linear operations that must act on the clean signal.
        defer_post = self.ao_enabled or self.denoise_enabled

        d_output = render(
            self.raster,
            camera_position=tuple(self.position),
            look_at=tuple(self._get_look_at()),
            fov=self.fov,
            width=self.render_width,
            height=self.render_height,
            sun_azimuth=self.sun_azimuth,
            sun_altitude=self.sun_altitude,
            shadows=self.shadows,
            ambient=self.ambient,
            colormap=self.colormap,
            rtx=self.rtx,
            viewshed_data=viewshed_data,
            viewshed_opacity=self.viewshed_opacity,
            observer_position=observer_pos,
            pixel_spacing_x=self.pixel_spacing_x,
            pixel_spacing_y=self.pixel_spacing_y,
            mesh_type=self.mesh_type,
            color_data=self._active_color_data,
            color_stretch=self.color_stretch,
            color_range=self._land_color_range,
            rgb_texture=rgb_texture,
            overlay_data=self._active_overlay_data,
            overlay_alpha=self._overlay_alpha,
            geometry_colors=geometry_colors,
            ao_samples=ao_samples,
            ao_radius=self.ao_radius,
            ao_seed=ao_seed,
            gi_intensity=self.gi_intensity,
            gi_bounces=self.gi_bounces,
            frame_seed=frame_seed,
            sun_angle=1.5,
            aperture=dof_aperture,
            focal_distance=dof_focal,
            bloom=not defer_post,
            tone_map=not defer_post,
            _return_gpu=True,
        )

        return d_output

    def _update_frame(self):
        """Full render: GPU ray trace → D2H copy → overlays → display."""
        # Sync previous frame's async D2H copy (no-op on first frame)
        self._readback_stream.synchronize()

        # GPU render — returns cupy array (no D2H copy)
        d_output = self._render_frame()
        self.frame_count += 1

        # Progressive AO accumulation
        if self.ao_enabled:
            from .analysis.render import _bloom, _tone_map_aces, _render_buffers

            # Check if camera moved — compare current state to previous
            cam_state = (tuple(self.position), self.yaw, self.pitch, self.fov)
            if self._prev_cam_state != cam_state:
                # Camera moved: reset accumulation
                self._d_ao_accum = None
                self._ao_frame_count = 0
                self._prev_cam_state = cam_state

            # Accumulate
            if self._d_ao_accum is None or self._d_ao_accum.shape != d_output.shape:
                self._d_ao_accum = d_output.copy()
            else:
                self._d_ao_accum += d_output
            self._ao_frame_count += 1

            # Average the accumulated frames
            d_display = self._d_ao_accum / self._ao_frame_count
        else:
            d_display = d_output

        # Deferred post-processing: denoise → bloom → tone map.
        # These are deferred when AO or denoiser is active so they
        # operate on the clean / averaged signal.
        defer_post = self.ao_enabled or self.denoise_enabled
        if defer_post:
            if not self.ao_enabled:
                from .analysis.render import _bloom, _tone_map_aces, _render_buffers

            if self.denoise_enabled:
                from .rtx import denoise as _denoise
                from .analysis.render import (
                    _compute_camera_basis, _render_buffers as _bufs,
                    compute_flow,
                )
                h, w = self.render_height, self.render_width
                d_normals = _bufs.primary_hits.reshape(h, w, 4)[:, :, 1:4].copy()
                ve = self.vertical_exaggeration
                pos = self.position
                look = self._get_look_at()
                scaled_pos = (pos[0], pos[1], pos[2] * ve)
                scaled_look = (look[0], look[1], look[2] * ve)
                forward, right, cam_up = _compute_camera_basis(
                    scaled_pos, scaled_look, (0, 0, 1))

                # Compute flow vectors for temporal denoising
                d_flow = None
                aspect = w / h
                fov_scale = np.tan(np.radians(self.fov) / 2.0)
                if self._prev_cam_for_flow is not None:
                    prev_pos, prev_fwd, prev_right, prev_up, prev_aspect, prev_fov_scale = self._prev_cam_for_flow
                    # Allocate / resize flow buffer
                    if self._d_flow is None or self._d_flow.shape != (h, w, 2):
                        self._d_flow = cp.zeros((h, w, 2), dtype=cp.float32)
                    d_prev_pos = cp.asarray(np.array(prev_pos, dtype=np.float32))
                    d_prev_fwd = cp.asarray(np.array(prev_fwd, dtype=np.float32))
                    d_prev_right = cp.asarray(np.array(prev_right, dtype=np.float32))
                    d_prev_up = cp.asarray(np.array(prev_up, dtype=np.float32))
                    compute_flow(
                        self._d_flow, _bufs.primary_rays, _bufs.primary_hits,
                        w, h,
                        d_prev_pos, d_prev_fwd, d_prev_right, d_prev_up,
                        prev_aspect, prev_fov_scale,
                    )
                    d_flow = self._d_flow

                self._prev_cam_for_flow = (
                    scaled_pos, tuple(forward), tuple(right), tuple(cam_up),
                    aspect, fov_scale,
                )

                _denoise(d_display, d_normals, w, h, right, cam_up, forward,
                         albedo=_bufs.albedo, flow=d_flow)

            bufs = _render_buffers
            if bufs.bloom_temp is not None:
                _bloom(d_display, bufs.bloom_temp, bufs.bloom_scratch)
            _tone_map_aces(d_display)

        # Allocate pinned host buffer lazily (or on shape change)
        if self._pinned_frame is None or self._pinned_frame.shape != d_display.shape:
            self._pinned_mem = cp.cuda.alloc_pinned_memory(d_display.nbytes)
            self._pinned_frame = np.frombuffer(
                self._pinned_mem, dtype=np.float32, count=d_display.size
            ).reshape(d_display.shape)

        # Start async D2H copy on non-blocking stream
        d_display.get(out=self._pinned_frame, stream=self._readback_stream)

        # CPU work while DMA runs
        if self._wind_enabled and self._wind_particles is not None:
            self._update_wind_particles()

        # Wait for DMA to complete
        self._readback_stream.synchronize()

        # Composite overlays on top of the ray-traced base frame
        self._composite_overlays()

    def _composite_overlays(self):
        """Composite CPU overlays (wind, minimap, help) onto the base frame.

        Can be called without re-ray-tracing to animate wind cheaply.
        """
        # FPS tracking
        self._fps_counter += 1
        now = time.monotonic()
        elapsed = now - self._fps_last_time
        if elapsed >= 1.0:
            self._fps_display = self._fps_counter / elapsed
            self._fps_counter = 0
            self._fps_last_time = now

        # Build window title
        title = self._build_title()
        pos = self.position
        fps = self._fps_display
        sub = f"{fps:.0f} FPS  Pos: ({pos[0]:.0f}, {pos[1]:.0f}, {pos[2]:.0f})  Speed: {self.move_speed:.0f}"
        if self._observers:
            obs_parts = []
            for slot in sorted(self._observers):
                obs = self._observers[slot]
                marker = '*' if slot == self._active_observer else ''
                mode = ''
                if obs.drone_mode != 'off':
                    mode = f' {obs.drone_mode.upper()}'
                if obs.is_touring():
                    mode += ' TOUR'
                obs_parts.append(f"{slot}{marker}{mode}")
            sub += f"  \u2502  Obs: [{' '.join(obs_parts)}]"
            active_obs = (self._observers.get(self._active_observer)
                          if self._active_observer else None)
            if active_obs is not None:
                sub += f"  h={active_obs.observer_elev:.3f}"
            if self.viewshed_enabled:
                sub += f"  Coverage: {self._viewshed_coverage:.1f}%"

        combined = f"{title}  |  {sub}"
        if combined != self._last_title:
            self._last_title = combined
            if self._glfw_window is not None:
                import glfw
                glfw.set_window_title(self._glfw_window, combined)

        # Build display frame (copy if we need overlays, else use pinned directly)
        needs_overlay = (
            (self._wind_enabled and self._wind_particles is not None)
            or (self._gtfs_rt_enabled and self._gtfs_rt_vehicles is not None)
            or self.show_minimap
            or self.show_help
        )
        if needs_overlay:
            img = self._pinned_frame.copy()
        else:
            img = self._pinned_frame

        # Wind overlay
        if self._wind_enabled and self._wind_particles is not None:
            self._draw_wind_on_frame(img)

        # GTFS-RT vehicle overlay
        if self._gtfs_rt_enabled and self._gtfs_rt_vehicles is not None:
            self._draw_gtfs_rt_on_frame(img)

        # Minimap overlay
        self._blit_minimap_on_frame(img)

        # Help text overlay
        if self.show_help and self._help_text_rgba is not None:
            self._blit_help_on_frame(img)

        self._display_frame = img
        self._frame_dirty = True

    def _handle_scroll(self, yoffset):
        """Handle mouse scroll wheel for zoom.

        Parameters
        ----------
        yoffset : float
            Scroll amount (positive = scroll up = zoom in).
        """
        if yoffset > 0:
            self.fov = max(20, self.fov - 3)
        else:
            self.fov = min(120, self.fov + 3)
        print(f"FOV: {self.fov:.0f}")
        self._update_frame()

    def _handle_mouse_press(self, button, xpos, ypos):
        """Start drag on left-click, or teleport if click is on minimap.

        Parameters
        ----------
        button : int
            Mouse button (0 = left, 1 = right, 2 = middle).
        xpos, ypos : float
            Cursor position in window pixels.
        """
        if button == 0:  # left click
            # Check for minimap click-to-teleport
            if self._minimap_rect is not None and self.show_minimap:
                mx0, my0, mw, mh = self._minimap_rect
                # Convert window coords to frame (render) coords
                frame_x = xpos * self.render_width / max(1, self.width)
                frame_y = ypos * self.render_height / max(1, self.height)
                if (mx0 <= frame_x < mx0 + mw and my0 <= frame_y < my0 + mh):
                    # Convert minimap-local → terrain pixel → world XY
                    local_x = frame_x - mx0
                    local_y = frame_y - my0
                    H, W = self.terrain_shape
                    terrain_col = local_x / mw * W
                    terrain_row = local_y / mh * H
                    world_x = terrain_col * self.pixel_spacing_x
                    world_y = terrain_row * self.pixel_spacing_y
                    self.position[0] = world_x
                    self.position[1] = world_y
                    self._update_frame()
                    return

            self._mouse_dragging = True
            self._mouse_last_x = xpos
            self._mouse_last_y = ypos

    def _handle_mouse_release(self, button):
        """End drag on button release."""
        self._mouse_dragging = False

    def _handle_mouse_motion(self, xpos, ypos):
        """Pan camera on mouse drag (slippy-map style).

        Parameters
        ----------
        xpos, ypos : float
            Cursor position in screen pixels.
        """
        if not self._mouse_dragging or self._mouse_last_x is None:
            return

        dx = xpos - self._mouse_last_x
        # GLFW Y is top-down; invert so dragging up → positive dy
        dy = -(ypos - self._mouse_last_y)
        self._mouse_last_x = xpos
        self._mouse_last_y = ypos

        if dx == 0 and dy == 0:
            return

        H, W = self.terrain_shape
        world_diag = np.sqrt(
            (W * self.pixel_spacing_x) ** 2
            + (H * self.pixel_spacing_y) ** 2
        )
        sensitivity = world_diag * 0.20 / self.width

        right = self._get_right()
        front = self._get_front()
        front_horiz = np.array([front[0], front[1], 0], dtype=np.float32)
        norm = np.linalg.norm(front_horiz)
        if norm > 1e-8:
            front_horiz /= norm
        else:
            front_horiz = np.array([0, 1, 0], dtype=np.float32)

        # Scene follows cursor: drag right → camera left
        self.position -= right * dx * sensitivity
        self.position -= front_horiz * dy * sensitivity

        self._update_frame()

    def _render_help_text(self):
        """Pre-render help text to an RGBA numpy array using PIL.

        Called once at startup; the result is cached in self._help_text_rgba.
        Two-column layout with styled section headers and key highlighting.
        """
        # Two columns of (section_title, [(key, description), ...])
        col_left = [
            ("MOVEMENT", [
                ("W/S/A/D", "Move fwd / back / left / right"),
                ("Arrows", "Move fwd / back / left / right"),
                ("Q / E", "Move up / down"),
                ("I/J/K/L", "Look up / left / down / right"),
                ("Drag", "Pan (slippy-map)"),
                ("Scroll", "Zoom (FOV)"),
                ("+ / -", "Speed up / down"),
            ]),
            ("TERRAIN", [
                ("G", "Cycle terrain layer"),
                ("U", "Cycle basemap"),
                ("C", "Cycle colormap"),
                ("Y", "Cycle color stretch"),
                (", / .", "Overlay alpha"),
                ("R / Shift+R", "Resolution down / up"),
                ("Z / Shift+Z", "Vert. exag. down / up"),
                ("B", "Toggle TIN / Voxel"),
                ("T", "Toggle shadows"),
                ("Shift+T", "Cycle time of day"),
            ]),
            ("DATA LAYERS", [
                ("Shift+F", "FIRMS fire (7d)"),
                ("Shift+W", "Toggle wind"),
            ]),
        ]
        col_right = [
            ("RENDERING", [
                ("0", "Toggle ambient occlusion"),
                ("Shift+G", "Cycle GI bounces (1-3)"),
                ("Shift+D", "Toggle AI denoiser"),
                ("9", "Toggle depth of field"),
                ("; / '", "DOF aperture down / up"),
                (": / \"", "DOF focal dist. down / up"),
            ]),
            ("GEOMETRY", [
                ("N", "Cycle geometry layer"),
                ("P", "Prev geometry in group"),
            ]),
            ("OBSERVERS", [
                ("1-8", "Select / create observer"),
                ("O", "Move observer to camera"),
                ("Shift+O", "Drone mode (3rd / FPV)"),
                ("Shift+V", "Snap camera to observer"),
                ("Shift+K", "Kill all observers"),
                ("V", "Toggle viewshed"),
                ("[ / ]", "Observer height down / up"),
            ]),
            ("OTHER", [
                ("F", "Screenshot"),
                ("M", "Toggle minimap"),
                ("H", "Toggle this help"),
                ("X / Esc", "Exit"),
            ]),
        ]

        try:
            from PIL import Image, ImageDraw, ImageFont

            font_size = 12
            header_size = 13
            mono_path = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"
            bold_path = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf"
            sans_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
            sans_bold_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
            try:
                font = ImageFont.truetype(sans_path, font_size)
                font_key = ImageFont.truetype(mono_path, font_size)
                font_header = ImageFont.truetype(sans_bold_path, header_size)
            except (OSError, IOError):
                font = ImageFont.load_default()
                font_key = font
                font_header = font

            line_h = font_size + 5
            header_h = header_size + 8
            section_gap = 6
            key_col_w = 105  # width reserved for keys
            desc_col_w = 195  # width for descriptions
            col_w = key_col_w + desc_col_w
            col_gap = 20
            pad_x = 14
            pad_y = 12
            corner_r = 10

            # Colors
            bg_color = (15, 18, 24, 210)          # dark blue-black, 82% opaque
            header_color = (180, 210, 255, 255)    # light blue
            key_color = (255, 200, 100, 245)       # warm amber
            desc_color = (210, 215, 225, 220)      # soft white
            separator_color = (80, 90, 110, 120)   # subtle line
            accent_color = (90, 140, 220, 180)     # blue accent for header underline

            def _column_height(sections):
                h = 0
                for i, (title, items) in enumerate(sections):
                    if i > 0:
                        h += section_gap
                    h += header_h + 3  # header + underline space
                    h += len(items) * line_h
                return h

            left_h = _column_height(col_left)
            right_h = _column_height(col_right)
            content_h = max(left_h, right_h)
            footer_h = header_h + section_gap  # space for "Press H to close"
            img_w = pad_x * 2 + col_w * 2 + col_gap
            img_h = pad_y * 2 + content_h + footer_h

            # Create with transparent background, then draw rounded rect
            img = Image.new('RGBA', (img_w, img_h), (0, 0, 0, 0))
            draw = ImageDraw.Draw(img)

            # Rounded rectangle background
            draw.rounded_rectangle(
                [0, 0, img_w - 1, img_h - 1],
                radius=corner_r, fill=bg_color,
            )

            # Subtle border
            draw.rounded_rectangle(
                [0, 0, img_w - 1, img_h - 1],
                radius=corner_r, outline=(60, 70, 90, 140), width=1,
            )

            def _draw_column(sections, x_start, y_start):
                y = y_start
                for si, (title, items) in enumerate(sections):
                    if si > 0:
                        y += section_gap
                    # Section header
                    draw.text((x_start, y), title, fill=header_color,
                              font=font_header)
                    # Accent underline
                    underline_y = y + header_size + 2
                    draw.line(
                        [(x_start, underline_y),
                         (x_start + col_w - 10, underline_y)],
                        fill=accent_color, width=1)
                    y = underline_y + 3

                    # Key-description rows
                    for key_text, desc_text in items:
                        draw.text((x_start, y), key_text,
                                  fill=key_color, font=font_key)
                        draw.text((x_start + key_col_w, y), desc_text,
                                  fill=desc_color, font=font)
                        y += line_h

            _draw_column(col_left, pad_x, pad_y)
            _draw_column(col_right, pad_x + col_w + col_gap, pad_y)

            # Vertical separator between columns
            sep_x = pad_x + col_w + col_gap // 2
            draw.line(
                [(sep_x, pad_y + 4), (sep_x, pad_y + content_h - 4)],
                fill=separator_color, width=1)

            # Bold "Press H to close" footer, centered
            footer_text = "Press H to close"
            bbox = font_header.getbbox(footer_text)
            fw = bbox[2] - bbox[0]
            footer_x = (img_w - fw) // 2
            footer_y = pad_y + content_h + section_gap
            draw.text((footer_x, footer_y), footer_text,
                      fill=header_color, font=font_header)

            self._help_text_rgba = np.array(img, dtype=np.float32) / 255.0
        except ImportError:
            self._help_text_rgba = None

    def _blit_help_on_frame(self, img):
        """Alpha-composite cached help text onto the rendered frame.

        Parameters
        ----------
        img : ndarray, shape (H, W, 3), float32 0-1
            Rendered frame. Modified in-place.
        """
        if self._help_text_rgba is None:
            return
        ht = self._help_text_rgba
        hh, hw = ht.shape[:2]
        fh, fw = img.shape[:2]
        # Top-left with small margin
        margin = 8
        # Clamp to frame size
        bh = min(hh, fh - margin)
        bw = min(hw, fw - margin)
        if bh <= 0 or bw <= 0:
            return
        alpha = ht[:bh, :bw, 3:4]
        rgb = ht[:bh, :bw, :3]
        region = img[margin:margin+bh, margin:margin+bw]
        region[:] = region * (1 - alpha) + rgb * alpha

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
        import glfw
        import moderngl

        H, W = self.terrain_shape

        # World-coordinate extents (accounts for pixel_spacing)
        world_W = W * self.pixel_spacing_x
        world_H = H * self.pixel_spacing_y
        world_diag = np.sqrt(world_W**2 + world_H**2)

        # Set initial move speed based on terrain extent (~1% of diagonal per keystroke)
        if self.move_speed is None:
            self.move_speed = world_diag * 0.01

        # Default: south bottom middle, looking north toward terrain center
        if start_position is None:
            start_position = (
                world_W / 2,
                world_H * 1.05,
                self.elev_max + world_diag * 0.08,
            )

        self.position = np.array(start_position, dtype=np.float32)

        # Calculate initial yaw/pitch from look_at
        if look_at is not None:
            direction = np.array(look_at) - self.position
            direction = direction / (np.linalg.norm(direction) + 1e-8)
            self.yaw = np.degrees(np.arctan2(direction[1], direction[0]))
            self.pitch = np.degrees(np.arcsin(np.clip(direction[2], -1, 1)))
        else:
            # Look toward terrain center
            center = np.array([world_W / 2, world_H / 2, self.elev_mean])
            direction = center - self.position
            direction = direction / (np.linalg.norm(direction) + 1e-8)
            self.yaw = np.degrees(np.arctan2(direction[1], direction[0]))
            self.pitch = np.degrees(np.arcsin(np.clip(direction[2], -1, 1)))

        # --- GLFW window creation ---
        if not glfw.init():
            raise RuntimeError("Failed to initialise GLFW")

        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
        glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, True)

        window = glfw.create_window(
            self.width, self.height,
            f'rtxpy \u2014 {self._title}', None, None,
        )
        if not window:
            glfw.terminate()
            raise RuntimeError("Failed to create GLFW window")

        glfw.make_context_current(window)
        glfw.swap_interval(0)  # No VSync — render as fast as GPU allows

        self._glfw_window = window

        # --- ModernGL context + fullscreen quad ---
        ctx = moderngl.create_context()
        prog = ctx.program(vertex_shader=_QUAD_VERT, fragment_shader=_QUAD_FRAG)

        # Fullscreen quad: position (x, y) + UV (u, v)
        # V is flipped: v=1 at bottom of screen maps to row 0 (top of image),
        # because OpenGL textures start at the bottom but numpy row 0 is top.
        quad_data = np.array([
            # x,    y,   u,  v
            -1.0, -1.0, 0.0, 1.0,  # bottom-left  → top of image
             1.0, -1.0, 1.0, 1.0,  # bottom-right → top of image
            -1.0,  1.0, 0.0, 0.0,  # top-left     → bottom of image
             1.0,  1.0, 1.0, 0.0,  # top-right    → bottom of image
        ], dtype='f4')
        vbo = ctx.buffer(quad_data.tobytes())
        vao = ctx.simple_vertex_array(prog, vbo, 'in_pos', 'in_uv')

        # Frame texture — sized to render resolution, updated every frame
        frame_tex = ctx.texture(
            (self.render_width, self.render_height), 3, dtype='f4',
        )
        frame_tex.filter = (moderngl.LINEAR, moderngl.LINEAR)

        # --- Pre-render help text overlay ---
        self._render_help_text()

        # --- Initialize minimap ---
        self._compute_minimap_background()

        # --- GLFW callbacks ---
        viewer = self  # closure reference

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
            viewer._render_needed = True

        glfw.set_key_callback(window, _key_cb)
        glfw.set_scroll_callback(window, _scroll_cb)
        glfw.set_mouse_button_callback(window, _mouse_btn_cb)
        glfw.set_cursor_pos_callback(window, _cursor_cb)
        glfw.set_framebuffer_size_callback(window, _framebuffer_size_cb)

        print(f"\nInteractive Viewer Started")
        print(f"  Window: {self.width}x{self.height}")
        print(f"  Render: {self.render_width}x{self.render_height} ({self.render_scale:.0%})")
        print(f"  Terrain: {W}x{H}, elevation {self.elev_min:.0f}m - {self.elev_max:.0f}m")
        print(f"\nPress H for controls, X or Esc to exit\n")

        self.running = True
        self._display_frame = None
        self._frame_dirty = False
        self._render_needed = True  # Ensure first frame renders
        self._fps_counter = 0
        self._fps_last_time = time.monotonic()
        self._last_tick_time = time.monotonic()

        # Render the initial frame so the window isn't blank
        self._tick()

        # --- REPL thread ---
        if self._repl:
            proxy = ViewerProxy(self)

            def _run_repl():
                # Auto-play tour if one was provided
                if getattr(self, '_tour', None) is not None:
                    import time as _time
                    _time.sleep(0.5)  # let first frames render
                    try:
                        proxy.tour(self._tour)
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
                self.running = False

            repl_thread = threading.Thread(
                target=_run_repl, daemon=True, name='rtxpy-repl')
            repl_thread.start()

        # --- Main loop ---
        try:
            while not glfw.window_should_close(window) and self.running:
                self._tick()

                # Drain REPL command queue (thread-safe)
                while True:
                    try:
                        cmd = self._command_queue.get_nowait()
                    except queue.Empty:
                        break
                    try:
                        cmd(self)
                    except Exception as exc:
                        import traceback
                        traceback.print_exc()
                    self._render_needed = True

                # Upload frame to texture and render only when dirty
                if self._frame_dirty and self._display_frame is not None:
                    tex_w, tex_h = frame_tex.size
                    fh, fw = self._display_frame.shape[:2]
                    if fw != tex_w or fh != tex_h:
                        frame_tex.release()
                        frame_tex = ctx.texture((fw, fh), 3, dtype='f4')
                        frame_tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
                    frame_tex.write(self._display_frame)
                    frame_tex.use()
                    ctx.clear()
                    vao.render(moderngl.TRIANGLE_STRIP)
                    glfw.swap_buffers(window)
                    self._frame_dirty = False

                glfw.poll_events()

                # Idle: yield CPU when nothing is happening (no movement,
                # no pending render).  Keeps input polling responsive at
                # ~120 Hz while avoiding a busy-wait spin.
                if not self._held_keys and not self._mouse_dragging:
                    time.sleep(0.008)
        finally:
            # --- Cleanup ---
            frame_tex.release()
            vbo.release()
            vao.release()
            prog.release()
            ctx.release()
            glfw.destroy_window(window)
            glfw.terminate()
            self._glfw_window = None
            # Reset terminal state (GLFW can hide cursor / alter termios)
            import sys
            sys.stdout.write('\033[?25h')  # show cursor
            sys.stdout.flush()

        # Clean up tile service
        if self._tile_service is not None:
            self._tile_service.shutdown()

        # Clean up GTFS-RT thread
        self._cleanup_gtfs_rt()

        print(f"Viewer closed after {self.frame_count} frames")


def explore(raster, width: int = 800, height: int = 600,
            render_scale: float = 0.5,
            start_position: Optional[Tuple[float, float, float]] = None,
            look_at: Optional[Tuple[float, float, float]] = None,
            key_repeat_interval: float = 0.05,
            rtx: 'RTX' = None,
            pixel_spacing_x: float = 1.0, pixel_spacing_y: float = 1.0,
            mesh_type: str = 'heightfield',
            overlay_layers: dict = None,
            color_stretch: str = 'linear',
            title: str = None,
            tile_service=None,
            geometry_colors_builder=None,
            baked_meshes=None,
            subsample: int = 1,
            wind_data=None,
            gtfs_data=None,
            accessor=None,
            terrain_loader=None,
            scene_zarr=None,
            ao_samples: int = 0,
            gi_bounces: int = 1,
            denoise: bool = False,
            repl: bool = False,
            tour=None):
    """
    Launch an interactive terrain viewer.

    Uses GLFW + ModernGL for display.
    Keyboard controls allow flying through the terrain.

    Parameters
    ----------
    raster : xarray.DataArray
        Terrain raster data with cupy array.
    width : int
        Display width in pixels. Default is 800.
    height : int
        Display height in pixels. Default is 600.
    render_scale : float
        Render at this fraction of display size (0.25-1.0).
        Lower values give higher FPS. Default is 0.5.
    start_position : tuple, optional
        Starting camera position (x, y, z). If None, starts at the
        south edge looking north.
    look_at : tuple, optional
        Initial look-at point.
    key_repeat_interval : float
        Minimum seconds between key repeat events (default 0.05 = 20 FPS max).
        Lower values = more responsive but more GPU load.
    rtx : RTX, optional
        Existing RTX instance with geometries (e.g., from place_mesh).
        If provided, renders the full scene including placed meshes.
    pixel_spacing_x : float, optional
        X spacing between pixels in world units (e.g., 30.0 for 30m/pixel).
        Must match the spacing used when triangulating terrain. Default 1.0.
    pixel_spacing_y : float, optional
        Y spacing between pixels in world units. Default 1.0.
    mesh_type : str, optional
        Mesh generation method: 'tin' or 'voxel'. Default is 'tin'.
    scene_zarr : str or Path, optional
        Path to a zarr store with a ``meshes/`` group. When provided,
        mesh chunks are loaded dynamically based on camera position
        instead of loading the full scene upfront.
    accessor : RTXAccessor, optional
        RTX accessor instance for on-demand data fetching (e.g. FIRMS fire
        layer via Shift+F).
    wind_data : dict, optional
        Wind data from ``fetch_wind()``. If provided, Shift+W toggles
        wind particle animation.
    gtfs_data : dict, optional
        GTFS data from ``fetch_gtfs()``. If the metadata contains a
        ``realtime_url``, Shift+B toggles realtime vehicle positions.
    ao_samples : int, optional
        If > 0, enable ambient occlusion on launch with progressive
        accumulation (1 sample per frame). Press 0 to toggle at runtime.
        Default is 0 (disabled).
    denoise : bool, optional
        If True, enable the OptiX AI Denoiser on launch. Press Shift+D
        to toggle at runtime. Default is False.
    tour : list of dict or str, optional
        If provided, automatically play a camera tour after the viewer
        launches.  Can be a list of keyframe dicts or a path to a
        ``.py`` file that defines a ``tour`` variable.  Implies
        ``repl=True`` — the REPL starts after the tour finishes.

    Controls
    --------
    - W/Up: Move forward
    - S/Down: Move backward
    - A/Left: Strafe left
    - D/Right: Strafe right
    - Q/Page Up: Move up
    - E/Page Down: Move down
    - I/J/K/L: Look up/left/down/right
    - Click+Drag: Pan (slippy-map style)
    - Scroll wheel: Zoom in/out (FOV)
    - +/=: Increase speed
    - -: Decrease speed
    - G: Cycle terrain color (elevation → overlays)
    - U: Cycle basemap (none → satellite → osm)
    - N: Cycle geometry layer (none → all → groups)
    - P: Jump to previous geometry in current group
    - ,/.: Decrease/increase overlay alpha (transparency)
    - O: Place observer (for viewshed) at look-at point
    - Shift+O: Cycle drone mode (off → 3rd person → FPV → off)
    - V: Toggle viewshed overlay (teal glow shows visible terrain)
    - [/]: Decrease/increase observer height
    - R: Decrease terrain resolution (coarser, up to 8x subsample)
    - Shift+R: Increase terrain resolution (finer, down to 1x)
    - Z: Decrease vertical exaggeration
    - Shift+Z: Increase vertical exaggeration
    - B: Toggle mesh type (TIN / voxel)
    - Y: Cycle color stretch (linear, sqrt, cbrt, log)
    - T: Toggle shadows
    - 0: Toggle ambient occlusion (progressive)
    - Shift+G: Cycle GI bounces (1→2→3→1)
    - Shift+D: Toggle OptiX AI Denoiser
    - C: Cycle colormap
    - Shift+F: Fetch/toggle FIRMS fire layer (7d LANDSAT 30m)
    - Shift+W: Toggle wind particle animation
    - Shift+B: Toggle GTFS-RT realtime vehicle overlay
    - F: Save screenshot
    - M: Toggle minimap overlay
    - H: Toggle help overlay
    - X: Exit

    Examples
    --------
    >>> import rtxpy
    >>> dem = xr.open_dataarray('terrain.tif')
    >>> dem = dem.copy(data=cupy.asarray(dem.data))
    >>> rtxpy.explore(dem)

    >>> # Or via accessor
    >>> dem.rtx.explore()
    """
    # Auto-detect Jupyter and use widget-based viewer
    from .notebook import _detect_jupyter
    if _detect_jupyter():
        from .notebook import JupyterViewer
        ViewerClass = JupyterViewer
    else:
        ViewerClass = InteractiveViewer

    viewer = ViewerClass(
        raster,
        width=width,
        height=height,
        render_scale=render_scale,
        key_repeat_interval=key_repeat_interval,
        rtx=rtx,
        pixel_spacing_x=pixel_spacing_x,
        pixel_spacing_y=pixel_spacing_y,
        mesh_type=mesh_type,
        overlay_layers=overlay_layers,
        title=title,
        subsample=subsample,
    )
    viewer._geometry_colors_builder = geometry_colors_builder
    viewer._baked_meshes = baked_meshes or {}
    viewer._accessor = accessor
    viewer._terrain_loader = terrain_loader
    if scene_zarr is not None:
        viewer._chunk_manager = _MeshChunkManager(
            scene_zarr, pixel_spacing_x, pixel_spacing_y)
    viewer.color_stretch = color_stretch
    if color_stretch in viewer._color_stretches:
        viewer._color_stretch_idx = viewer._color_stretches.index(color_stretch)
    if tile_service is not None:
        viewer._tile_service = tile_service
        # Sync basemap index to match the active tile service (but start OFF)
        pname = tile_service.provider_name
        if pname in viewer._basemap_options:
            viewer._basemap_idx = viewer._basemap_options.index(pname)
        else:
            viewer._basemap_idx = 0  # 'none'

    # Wind data initialization
    if wind_data is not None:
        viewer._init_wind(wind_data)

    # GTFS-RT initialization
    if gtfs_data is not None:
        rt_url = (gtfs_data.get('metadata') or {}).get('realtime_url')
        if rt_url:
            # Build route_id -> (r,g,b) colour map from route features
            rc_map = {}
            for f in gtfs_data.get('routes', {}).get('features', []):
                props = f.get('properties') or {}
                rc = (props.get('route_color') or '').strip().lstrip('#')
                rname = props.get('route_short_name', '')
                if len(rc) == 6:
                    try:
                        rgb = (int(rc[0:2], 16) / 255.0,
                               int(rc[2:4], 16) / 255.0,
                               int(rc[4:6], 16) / 255.0)
                        # Key by route_short_name since GTFS-RT uses route_id
                        # which may differ; store both for best chance of matching
                        if rname:
                            rc_map[rname] = rgb
                        rid = props.get('route_id', '')
                        if rid:
                            rc_map[rid] = rgb
                    except ValueError:
                        pass
            viewer._init_gtfs_rt(rt_url, route_colors=rc_map)

    # Ambient occlusion initialization
    if ao_samples > 0:
        viewer.ao_enabled = True
    viewer.gi_bounces = gi_bounces

    # Denoiser initialization
    if denoise:
        viewer.denoise_enabled = True

    # Initial state: everything off except elevation
    viewer._tiles_enabled = False
    viewer._basemap_idx = 0  # 'none'
    viewer._geometry_layer_idx = 0  # 'none'
    if rtx is not None:
        for geom_id in viewer._all_geometries:
            if geom_id != 'terrain':
                rtx.set_geometry_visible(geom_id, False)
    if tour is not None:
        repl = True
    viewer._repl = repl
    viewer._tour = tour
    return viewer.run(start_position=start_position, look_at=look_at)
