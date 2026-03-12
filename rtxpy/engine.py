"""Interactive terrain viewer using GLFW + ModernGL for display.

This module provides a simple game-engine-like render loop for
exploring terrain interactively with keyboard controls.
Uses GLFW for windowing/input and ModernGL for GPU texture display.
"""

import math
import os
import queue
import threading
import time
import numpy as np

# On WSL2 the hardware GLX drivers often segfault. Force Mesa software
# rendering for the display path (CUDA still handles the ray tracing).
if os.name != 'nt' and 'microsoft' in os.uname().release.lower():
    os.environ.setdefault('LIBGL_ALWAYS_SOFTWARE', '1')
from typing import Optional, Tuple

from .rtx import RTX, has_cupy
from .viewer.input_state import InputState
from .viewer.camera import CameraState
from .viewer.render_settings import RenderSettings
from .viewer.overlays import OverlayManager
from .viewer.geometry_layers import GeometryLayerManager
from .viewer.terrain import TerrainState
from .viewer.observers import ObserverManager, Observer, OBSERVER_COLORS
from .viewer.wind import WindState
from .viewer.cloud import CloudState
from .viewer.hydro import HydroState
from .viewer.hydro_manager import HydroManager
from .viewer.hud import HUDState
from .viewer.keybindings import MOVEMENT_KEYS, SHIFT_BINDINGS, KEY_BINDINGS, SPECIAL_BINDINGS

if has_cupy:
    import cupy as cp
    from numba import cuda


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


# ---------------------------------------------------------------------------
# Numba CUDA kernel for wind particle splatting
# ---------------------------------------------------------------------------
if has_cupy:
    @cuda.jit
    def _wind_splat_kernel(
        trails,       # (N*T, 2) float32 — (row, col) per trail point
        alphas,       # (N*T,) float32 — pre-computed alpha per trail point
        terrain,      # (tH, tW) float32 — terrain elevation
        output,       # (sh, sw, 3) float32 — frame buffer (atomic add)
        # Camera basis — scalar args to avoid tiny GPU allocations
        cam_x, cam_y, cam_z,
        fwd_x, fwd_y, fwd_z,
        rgt_x, rgt_y, rgt_z,
        up_x, up_y, up_z,
        # Projection params
        fov_scale, aspect_ratio,
        # Terrain/world params
        psx, psy, ve, subsample_f, min_depth,
        # Splat params
        dot_radius,
        # Wind color
        color_r, color_g, color_b,
    ):
        idx = cuda.grid(1)
        if idx >= trails.shape[0]:
            return

        a = alphas[idx]
        if a < 1e-6:
            return

        row = trails[idx, 0]
        col = trails[idx, 1]

        # Terrain Z lookup (nearest-neighbor, clamped)
        tH = terrain.shape[0]
        tW = terrain.shape[1]
        sr = int(row / subsample_f)
        sc = int(col / subsample_f)
        if sr < 0:
            sr = 0
        elif sr >= tH:
            sr = tH - 1
        if sc < 0:
            sc = 0
        elif sc >= tW:
            sc = tW - 1
        z_raw = terrain[sr, sc]
        # Ocean/water pixels are NaN — map to 0 (matches CPU path)
        if z_raw != z_raw:  # NaN check (works under fast math)
            z_raw = 0.0
        z_val = z_raw * ve + 3.0

        # World position
        wx = col * psx
        wy = row * psy

        # Camera-relative
        dx = wx - cam_x
        dy = wy - cam_y
        dz = z_val - cam_z

        # Depth along forward axis
        depth = dx * fwd_x + dy * fwd_y + dz * fwd_z
        if depth <= min_depth:
            return

        inv_depth = 1.0 / (depth + 1e-10)
        u_cam = dx * rgt_x + dy * rgt_y + dz * rgt_z
        v_cam = dx * up_x + dy * up_y + dz * up_z
        u_ndc = u_cam * inv_depth / (fov_scale * aspect_ratio)
        v_ndc = v_cam * inv_depth / fov_scale

        sh = output.shape[0]
        sw = output.shape[1]
        sx = int((u_ndc + 1.0) * 0.5 * sw)
        sy = int((1.0 - v_ndc) * 0.5 * sh)

        if sx < 0 or sx >= sw or sy < 0 or sy >= sh:
            return

        # Circular stamp splat
        r = dot_radius
        for offy in range(-r, r + 1):
            for offx in range(-r, r + 1):
                dist_sq = offx * offx + offy * offy
                if dist_sq > r * r:
                    continue
                falloff = 1.0 - math.sqrt(dist_sq) / r
                px = sx + offx
                py = sy + offy
                if px < 0 or px >= sw or py < 0 or py >= sh:
                    continue
                contrib = a * falloff
                cuda.atomic.add(output, (py, px, 0), contrib * color_r)
                cuda.atomic.add(output, (py, px, 1), contrib * color_g)
                cuda.atomic.add(output, (py, px, 2), contrib * color_b)


    @cuda.jit
    def _rain_splat_kernel(
        pts,          # (N, 2) float32 — (row, col) per rain particle
        z_frac,       # (N,) float32 — altitude fraction (0=ground, 1=cloud)
        alphas,       # (N,) float32 — pre-computed alpha
        streak_lens,  # (N,) int32 — vertical streak length in pixels
        terrain,      # (tH, tW) float32 — terrain elevation
        output,       # (sh, sw, 3) float32 — frame buffer (atomic add)
        # Camera basis
        cam_x, cam_y, cam_z,
        fwd_x, fwd_y, fwd_z,
        rgt_x, rgt_y, rgt_z,
        up_x, up_y, up_z,
        # Projection
        fov_scale, aspect_ratio,
        # World params
        psx, psy, ve, subsample_f, cloud_z, min_depth,
        # Color
        color_r, color_g, color_b,
    ):
        idx = cuda.grid(1)
        if idx >= pts.shape[0]:
            return

        a = alphas[idx]
        if a < 0.002:
            return

        row = pts[idx, 0]
        col = pts[idx, 1]

        # Terrain Z lookup
        tH = terrain.shape[0]
        tW = terrain.shape[1]
        sr = int(row / subsample_f)
        sc = int(col / subsample_f)
        if sr < 0:
            sr = 0
        elif sr >= tH:
            sr = tH - 1
        if sc < 0:
            sc = 0
        elif sc >= tW:
            sc = tW - 1
        z_raw = terrain[sr, sc]
        if z_raw != z_raw:
            z_raw = 0.0
        terrain_z = z_raw * ve
        rain_z = terrain_z + z_frac[idx] * (cloud_z - terrain_z)

        wx = col * psx
        wy = row * psy

        dx = wx - cam_x
        dy = wy - cam_y
        dz = rain_z - cam_z

        depth = dx * fwd_x + dy * fwd_y + dz * fwd_z
        if depth <= min_depth:
            return

        inv_depth = 1.0 / (depth + 1e-10)
        u_cam = dx * rgt_x + dy * rgt_y + dz * rgt_z
        v_cam = dx * up_x + dy * up_y + dz * up_z
        u_ndc = u_cam * inv_depth / (fov_scale * aspect_ratio)
        v_ndc = v_cam * inv_depth / fov_scale

        sh = output.shape[0]
        sw = output.shape[1]
        sx = int((u_ndc + 1.0) * 0.5 * sw)
        sy = int((1.0 - v_ndc) * 0.5 * sh)

        if sx < 0 or sx >= sw or sy < 0 or sy >= sh:
            return

        # Vertical streak
        sl = streak_lens[idx]
        for dy_off in range(sl):
            py = sy + dy_off
            if py < 0 or py >= sh:
                continue
            t = float(dy_off) / float(sl) if sl > 0 else 0.0
            streak_a = a * (1.0 - t * 0.6)
            cuda.atomic.add(output, (py, sx, 0), streak_a * color_r)
            cuda.atomic.add(output, (py, sx, 1), streak_a * color_g)
            cuda.atomic.add(output, (py, sx, 2), streak_a * color_b)


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


# Observer and OBSERVER_COLORS imported from viewer.observers


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

        # Distance-aware loading parameters
        self._chunk_world_w = self._chunk_w * psx
        self._chunk_world_h = self._chunk_h * psy
        self.max_distance = None  # None = use radius-based fallback
        self.per_tick_load_limit = 2  # max new zarr reads per tick
        self.max_chunks = 25  # max visible chunks

        # LOD-aware loading state
        self._lod_distances = None  # set from LOD manager
        self._tile_lods = None  # {(tr,tc): lod_level} from aligned LOD manager
        self._last_cam_pos = None  # for movement detection
        self._cam_moving = False

        # Mesh simplification for placed geometry at higher LOD levels.
        # LOD 0 = full detail (ratio 1.0), LOD 1 = 50%, LOD 2 = 25%, LOD 3+ = 10%.
        self._simplify_ratios = (1.0, 0.5, 0.25, 0.1)
        # Cache: (cr, cc, gid, lod) -> (simplified_verts, simplified_indices)
        self._simplify_cache = {}

    def _load_chunk(self, cr, cc):
        """Load a single chunk from zarr into cache."""
        if (cr, cc) in self._cache:
            return
        from .mesh_store import load_meshes_from_zarr
        meshes, _, _, curves, _spheres = load_meshes_from_zarr(
            self._zarr_path, chunks=[(cr, cc)])
        # Merge curves into the same dict with a marker
        combined = {}
        for gid, data in meshes.items():
            combined[gid] = data  # (verts, indices)
        for gid, data in curves.items():
            combined[gid] = data  # (verts, widths, indices)
        self._cache[(cr, cc)] = combined

    def _chunk_center(self, cr, cc):
        """World-coordinate center of chunk (cr, cc)."""
        cx = (cc * self._chunk_w + self._chunk_w * 0.5) * self._psx
        cy = (cr * self._chunk_h + self._chunk_h * 0.5) * self._psy
        return cx, cy

    def _get_simplified(self, cr, cc, gid, lod, verts, indices):
        """Return (possibly simplified) mesh for a chunk at a given LOD.

        LOD 0 returns the original mesh.  LOD 1+ applies quadric
        decimation with ratio from ``_simplify_ratios``, caching the
        result for reuse across frames.
        """
        ratio_idx = min(lod, len(self._simplify_ratios) - 1)
        ratio = self._simplify_ratios[ratio_idx]
        if ratio >= 1.0:
            return verts, indices
        key = (cr, cc, gid, lod)
        cached = self._simplify_cache.get(key)
        if cached is not None:
            return cached
        from .lod import simplify_mesh
        sv, si = simplify_mesh(verts, indices, ratio)
        self._simplify_cache[key] = (sv, si)
        return sv, si

    def update(self, cam_x, cam_y, viewer):
        """Called per tick. Returns True if meshes changed."""
        import math
        from .lod import compute_lod_level

        # Detect camera movement for LOD-aware load deferral
        move_thresh = self._chunk_world_w * 0.1
        if self._last_cam_pos is not None:
            dx = cam_x - self._last_cam_pos[0]
            dy = cam_y - self._last_cam_pos[1]
            self._cam_moving = (dx * dx + dy * dy) > move_thresh * move_thresh
        self._last_cam_pos = (cam_x, cam_y)

        max_dist = self.max_distance
        lod_dists = self._lod_distances
        tile_lods = self._tile_lods  # {(tr,tc): lod} when grids aligned
        chunk_dists = {}
        chunk_lods = {}  # per-chunk LOD level

        if tile_lods is not None:
            # Grids aligned: reuse LOD manager's tile assignments directly.
            # tile_lods keys are (tile_row, tile_col) which map 1:1 to
            # chunk (cr, cc) when tile_size == chunk_size.
            new_visible = set()
            max_lod = len(lod_dists) if lod_dists else 999
            for (cr, cc), lod in tile_lods.items():
                if cr >= self._n_chunk_rows or cc >= self._n_chunk_cols:
                    continue
                if lod > max_lod:
                    continue
                chunk_lods[(cr, cc)] = lod
                cx, cy = self._chunk_center(cr, cc)
                chunk_dists[(cr, cc)] = math.sqrt(
                    (cam_x - cx) ** 2 + (cam_y - cy) ** 2)
                new_visible.add((cr, cc))
            # Cap at max_chunks, keeping closest
            if len(new_visible) > self.max_chunks:
                by_dist = sorted(new_visible, key=lambda c: chunk_dists[c])
                new_visible = set(by_dist[:self.max_chunks])
        elif max_dist is not None:
            # Distance-aware: compute visible chunks from world-rect
            from .mesh_store import chunks_for_world_rect
            x0 = cam_x - max_dist
            y0 = cam_y - max_dist
            x1 = cam_x + max_dist
            y1 = cam_y + max_dist
            candidates = chunks_for_world_rect(
                x0, y0, x1, y1,
                self._psx, self._psy,
                self._chunk_h, self._chunk_w,
                self._elev_shape)
            for cr, cc in candidates:
                cx, cy = self._chunk_center(cr, cc)
                chunk_dists[(cr, cc)] = math.sqrt(
                    (cam_x - cx) ** 2 + (cam_y - cy) ** 2)
            if lod_dists:
                max_lod = len(lod_dists)
                candidates = [
                    c for c in candidates
                    if compute_lod_level(chunk_dists[c], lod_dists) <= max_lod
                ]
            candidates.sort(key=lambda c: chunk_dists[c])
            new_visible = set(candidates[:self.max_chunks])
            # Compute per-chunk LOD for mesh simplification
            if lod_dists:
                for cr, cc in new_visible:
                    chunk_lods[(cr, cc)] = compute_lod_level(
                        chunk_dists[(cr, cc)], lod_dists)
        else:
            # Legacy radius-based ring
            cc_cam = int(cam_x / self._psx) // self._chunk_w
            cr_cam = int(cam_y / self._psy) // self._chunk_h
            cr0 = max(cr_cam - self.radius, 0)
            cr1 = min(cr_cam + self.radius, self._n_chunk_rows - 1)
            cc0 = max(cc_cam - self.radius, 0)
            cc1 = min(cc_cam + self.radius, self._n_chunk_cols - 1)
            new_visible = set()
            for cr in range(cr0, cr1 + 1):
                for cc in range(cc0, cc1 + 1):
                    new_visible.add((cr, cc))
                    cx, cy = self._chunk_center(cr, cc)
                    chunk_dists[(cr, cc)] = math.sqrt(
                        (cam_x - cx) ** 2 + (cam_y - cy) ** 2)

        # Check if any visible chunks are uncached (deferred from prior tick)
        has_deferred = any((cr, cc) not in self._cache for cr, cc in new_visible)

        if new_visible == self._visible and not has_deferred:
            return False

        # Evict simplification cache entries for chunks leaving visible set
        departed = self._visible - new_visible
        if departed and self._simplify_cache:
            for k in [k for k in self._simplify_cache
                      if (k[0], k[1]) in departed]:
                del self._simplify_cache[k]

        self._visible = new_visible

        # Load uncached chunks, prioritized by distance (closest first).
        # Limited to per_tick_load_limit new zarr reads per tick.
        uncached = [(cr, cc) for cr, cc in new_visible
                    if (cr, cc) not in self._cache]
        if uncached and chunk_dists:
            uncached.sort(key=lambda c: chunk_dists.get(c, 0))
        loads = 0
        for cr, cc in uncached:
            if loads >= self.per_tick_load_limit:
                break
            # When moving, defer distant (LOD 1+) chunks
            lod = chunk_lods.get((cr, cc))
            if lod is None and lod_dists and (cr, cc) in chunk_dists:
                lod = compute_lod_level(chunk_dists[(cr, cc)], lod_dists)
            if self._cam_moving and lod is not None and lod > 0:
                continue
            self._load_chunk(cr, cc)
            loads += 1

        # Merge visible chunks per gid.  Iterate chunks first so we only
        # touch gids that actually have data (skips empty lookups).
        # Per-gid accumulators: {gid: (all_verts, all_widths, all_indices,
        #                              vert_offset, is_curve)}
        merge_acc = {}
        for cr, cc in sorted(new_visible):
            chunk_data = self._cache.get((cr, cc))
            if not chunk_data:
                continue
            clod = chunk_lods.get((cr, cc), 0)
            for gid, data in chunk_data.items():
                if len(data) == 3:
                    verts, widths, indices = data
                    if len(indices) == 0:
                        continue
                    acc = merge_acc.get(gid)
                    if acc is None:
                        acc = ([], [], [], [0], True)
                        merge_acc[gid] = acc
                    acc[0].append(verts)
                    acc[1].append(widths)
                    acc[2].append(indices + acc[3][0])
                    acc[3][0] += len(verts) // 3
                else:
                    verts, indices = data
                    if len(indices) == 0:
                        continue
                    if clod > 0:
                        verts, indices = self._get_simplified(
                            cr, cc, gid, clod, verts, indices)
                    acc = merge_acc.get(gid)
                    if acc is None:
                        acc = ([], [], [], [0], False)
                        merge_acc[gid] = acc
                    acc[0].append(verts)
                    acc[2].append(indices + acc[3][0])
                    acc[3][0] += len(verts) // 3

        merged = {}
        for gid, (all_verts, all_widths, all_indices, _, is_curve) in merge_acc.items():
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

            # Apply VE to Z coordinates and cache base_z for VE rescaling.
            # orig_base_z and new_base_z both sample the same full-res terrain
            # at the same XY positions, so (new_base_z + z_offset) == stored_z.
            # The only transformation needed is: final_z = stored_z * ve.
            # We still compute base_z once for the baked mesh cache (used by
            # _rebuild_vertical_exaggeration to rescale without re-reading zarr).
            n_verts = len(verts) // 3
            use_gpu = (gpu_terrain is not None
                       and gpu_base_terrain is not None
                       and n_verts > 10000)

            if use_gpu:
                # cp.asarray copies H→D (verts is numpy), so we can
                # mutate it in-place without an extra GPU copy.
                updated_verts_gpu = cp.asarray(verts)
                if ve != 1.0:
                    updated_verts_gpu[2::3] *= ve

                if is_curve:
                    rtx.add_curve_geometry(
                        gid, updated_verts_gpu,
                        cp.asarray(widths), cp.asarray(indices))
                else:
                    rtx.add_geometry(gid, updated_verts_gpu, cp.asarray(indices))
                self._active_gids.add(gid)

                if accessor is not None:
                    accessor._geometry_colors[gid] = self._colors.get(gid, (0.6, 0.6, 0.6))
                    vx = cp.asarray(verts[0::3])
                    vy = cp.asarray(verts[1::3])
                    orig_base_z_np = _bilinear_terrain_z(
                        gpu_base_terrain, vx, vy,
                        base_psx, base_psy).get()
                    if is_curve:
                        accessor._baked_meshes[gid] = (
                            verts.copy(), widths.copy(), indices.copy(), orig_base_z_np)
                    else:
                        accessor._baked_meshes[gid] = (verts.copy(), indices.copy(), orig_base_z_np)
            else:
                updated_verts = verts.copy()
                if ve != 1.0:
                    updated_verts[2::3] *= ve

                if is_curve:
                    rtx.add_curve_geometry(gid, updated_verts, widths, indices)
                else:
                    rtx.add_geometry(gid, updated_verts, indices)
                self._active_gids.add(gid)

                if accessor is not None:
                    accessor._geometry_colors[gid] = self._colors.get(gid, (0.6, 0.6, 0.6))
                    orig_base_z = _bilinear_terrain_z(
                        base_terrain_np, verts[0::3], verts[1::3],
                        base_psx, base_psy)
                    if is_curve:
                        accessor._baked_meshes[gid] = (
                            verts.copy(), widths.copy(), indices.copy(), orig_base_z)
                    else:
                        accessor._baked_meshes[gid] = (verts.copy(), indices.copy(), orig_base_z)

        if accessor is not None:
            accessor._geometry_colors_dirty = True

        # Refresh viewer geometry tracking (same pattern as FIRMS toggle)
        from .viewer.terrain_lod import is_terrain_lod_gid
        viewer._all_geometries = rtx.list_geometries()
        groups = set()
        for g in viewer._all_geometries:
            if is_terrain_lod_gid(g):
                continue
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
    # Picking
    # ------------------------------------------------------------------

    def pick(self, screen_x, screen_y):
        """Pick geometry at screen coordinates. Returns hit info dict."""
        def fn(v):
            origin, direction = v._screen_to_ray(screen_x, screen_y)
            return v.rtx.pick(origin, direction)
        return self._submit(fn)

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

    def add_hydro(self, flow_accum, **kwargs):
        """Add hydrological flow particle visualization.

        Uses MFD (Multiple Flow Direction) to compute flow vectors from
        terrain elevation so particles follow natural drainage paths
        distributed across all downhill neighbors.

        Parameters
        ----------
        flow_accum : array-like, shape (H, W)
            Flow accumulation grid (cell counts or area).  Compute with
            ``xrspatial.flow_accumulation()``.
        **kwargs
            Optional overrides: n_particles, max_age, trail_len, speed,
            accum_threshold, color, alpha, dot_radius, min_visible_age,
            flow_dir_mfd (xrspatial MFD fractions),
            elevation (conditioned DEM for manual MFD fallback).
        """
        stream_order = kwargs.get('stream_order')
        stream_link = kwargs.get('stream_link')
        def fn(v):
            v._init_hydro(flow_accum, **kwargs)
            v._hydro_enabled = True
            # Add stream link overlay with palette-matched colors
            if stream_link is not None:
                sl_np = stream_link.get() if hasattr(stream_link, 'get') else np.asarray(stream_link)
                sl_np = np.asarray(sl_np, dtype=np.float32)
                # Use stream order values for coloring, NaN where no stream
                if stream_order is not None:
                    so_for_sl = stream_order.get() if hasattr(stream_order, 'get') else np.asarray(stream_order)
                    so_for_sl = np.asarray(so_for_sl, dtype=np.float32)
                    max_order = int(np.nanmax(so_for_sl)) if np.any(~np.isnan(so_for_sl)) else 5
                    palette_lut = InteractiveViewer._build_stream_palette_lut(
                        max_order)
                    sl_color = np.where(
                        np.isnan(sl_np) | (sl_np <= 0) | np.isnan(so_for_sl) | (so_for_sl <= 0),
                        np.float32(np.nan), so_for_sl)
                    _add_overlay(v, 'stream_link', sl_color,
                                 color_lut=palette_lut)
                else:
                    sl_np = np.where(np.isnan(sl_np) | (sl_np <= 0),
                                     np.float32(np.nan), sl_np)
                    _add_overlay(v, 'stream_link', sl_np)
            v._update_frame()
        self._submit(fn)

    def remove_layer(self, name):
        """Remove an overlay layer by name."""
        def fn(v):
            if name in v._overlay_layers:
                del v._overlay_layers[name]
                if name in v._base_overlay_layers:
                    del v._base_overlay_layers[name]
                if name in v._overlay_color_luts:
                    del v._overlay_color_luts[name]
                v._overlay_names = list(v._overlay_layers.keys())
                v._terrain_layer_order = (
                    ['elevation'] + list(v._overlay_names))
                # Reset to elevation if we removed the active layer
                if v._terrain_layer_idx >= len(v._terrain_layer_order):
                    v._terrain_layer_idx = 0
                    v._active_overlay_data = None
                    v._overlay_as_water = False
                    v._active_overlay_color_lut = None
                v._update_frame()
                print(f"Removed layer: {name}")
        self._submit(fn)

    def show_layer(self, name):
        """Switch the terrain coloring to a named layer (or 'elevation')."""
        def fn(v):
            if name == 'elevation':
                v._active_color_data = None
                v._active_overlay_data = None
                v._overlay_as_water = False
                v._active_overlay_color_lut = None
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
            v._overlay_as_water = (
                name.startswith('flood_')
                or (name == 'stream_link' and v._hydro_enabled))
            v._active_overlay_color_lut = v._overlay_color_luts.get(name)
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


def _add_overlay(viewer, name, data, color_lut=None):
    """Add or replace an overlay layer on *viewer* and switch to it.

    Must be called on the main (render) thread.

    Parameters
    ----------
    color_lut : np.ndarray, optional
        (256, 3) float32 categorical palette LUT.  When provided, the
        overlay bypasses the terrain colormap and color stretch.
    """
    viewer._overlay_layers[name] = data
    viewer._base_overlay_layers[name] = data
    if color_lut is not None:
        viewer._overlay_color_luts[name] = color_lut
    viewer._overlay_names = list(viewer._overlay_layers.keys())
    viewer._terrain_layer_order = (
        ['elevation'] + list(viewer._overlay_names))
    idx = viewer._terrain_layer_order.index(name)
    viewer._terrain_layer_idx = idx
    viewer._active_color_data = None
    viewer._active_overlay_data = data
    viewer._overlay_as_water = name.startswith('flood_')
    viewer._active_overlay_color_lut = color_lut
    # Skip render if camera not yet initialized (overlays added before run())
    if viewer.position is not None:
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
    - U/Shift+U: Cycle basemap forward/backward (none → satellite → osm)
    - N: Cycle geometry layer (none → all → groups)
    - P: Jump to previous geometry in current group
    - Shift+C: Cycle point cloud color mode (elevation/intensity/classification/rgb)
    - ,/.: Decrease/increase overlay alpha (transparency)
    - O: Place observer (for viewshed) at look-at point
    - Shift+O: Cycle drone mode (off → 3rd person → FPV → off)
    - V: Toggle viewshed overlay (teal glow shows visible terrain)
    - [/]: Decrease/increase observer height
    - R: Decrease terrain resolution (coarser, up to 8x subsample)
    - Shift+R: Increase terrain resolution (finer, down to 1x)
    - Z: Decrease vertical exaggeration
    - Shift+Z: Increase vertical exaggeration
    - Y: Cycle color stretch (linear, sqrt, cbrt, log)
    - T: Toggle shadows
    - 0: Toggle ambient occlusion (progressive)
    - Shift+G: Cycle GI bounces (1→2→3→1)
    - Shift+D: Toggle OptiX AI Denoiser
    - C: Cycle colormap
    - Shift+F: Fetch/toggle FIRMS fire layer (7d LANDSAT 30m)
    - Shift+W: Toggle wind particle animation
    - Shift+Y: Toggle hydro flow particle animation
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
                 overlay_layers: dict = None,
                 title: str = None,
                 subtitle: str = None,
                 legend: dict = None,
                 subsample: int = 1,
                 skirt: bool = True):
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
        """
        if not has_cupy:
            raise ImportError(
                "cupy is required for the interactive viewer. "
                "Install with: conda install -c conda-forge cupy"
            )

        # Terrain state (raster, spacing, elevation stats, mesh caches)
        self.terrain = TerrainState(
            raster, pixel_spacing_x=pixel_spacing_x,
            pixel_spacing_y=pixel_spacing_y,
            subsample=subsample, skirt=skirt,
        )

        self.rtx = rtx
        self._scene_diagonal = 1.0  # updated in run() with actual terrain extent
        self.width = width
        self.height = height
        self.render_scale = np.clip(render_scale, 0.25, 1.0)
        self.render_width = int(width * self.render_scale)
        self.render_height = int(height * self.render_scale)

        # Async readback: non-blocking stream + pinned host buffer
        self._readback_stream = cp.cuda.Stream(non_blocking=True)
        self._pinned_mem = None
        self._pinned_frame = None

        # Overlay layers and basemap state (must be created before subsample code)
        _base_ovl = overlay_layers.copy() if overlay_layers else {}
        self.overlays = OverlayManager(
            overlay_layers=overlay_layers,
            base_overlay_layers=_base_ovl,
        )

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

        # Geometry layer visibility tracking
        self.geometry_layers = GeometryLayerManager()

        if rtx is not None:
            self._all_geometries = rtx.list_geometries()
            groups = set()
            layer_geoms = {}

            for g in self._all_geometries:
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

            for layer_name, geom_ids in layer_geoms.items():
                positions = []
                for geom_id in sorted(geom_ids):
                    transform = rtx.get_geometry_transform(geom_id)
                    if transform:
                        x, y, z = transform[3], transform[7], transform[11]
                        positions.append((x, y, z, geom_id))
                self._layer_positions[layer_name] = positions

        # Camera state (position, orientation, FOV, speed, time presets)
        self.camera = CameraState()

        # Rendering settings (lighting, colormap, AO, denoiser, DOF)
        self.render_settings = RenderSettings()

        # Observer system (viewshed, drones, multi-observer)
        self.observer_mgr = ObserverManager()

        # HUD state (title, subtitle, legend, help, minimap)
        _hud_title = title
        if not _hud_title:
            if hasattr(raster, 'name') and raster.name:
                _hud_title = str(raster.name)
            else:
                _hud_title = 'rtxpy'
        self.hud = HUDState(title=_hud_title, subtitle=subtitle, legend=legend)

        # State
        self.running = False
        self.frame_count = 0

        # FIRMS fire layer state
        self._accessor = None         # RTX accessor for place_geojson
        self._firms_loaded = False    # Whether fire data has been fetched
        self._firms_visible = False   # Current visibility state

        # Wind particle state
        self.wind = WindState()

        # Cloud particle state
        self.clouds = CloudState()

        # Hydro flow particle state
        self.hydro = HydroState()
        self.hydro_mgr = HydroManager(self.hydro)

        # Per-tile overlay compositing (created when LOD is enabled)
        self._overlay_tile_mgr = None
        # Per-tile basemap texture compositing (created when LOD is enabled)
        self._texture_tile_mgr = None

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

        # Input state (held keys, mouse drag)
        self.input = InputState()

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

        # Mouse drag state lives in self.input (InputState)

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
                new_data = base_np.copy()
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

        # Enable LOD terrain immediately — no single-GAS terrain is built.
        # LOD tiles are created lazily during the first update() call.
        if rtx is not None:
            self._enable_terrain_lod()

    # ------------------------------------------------------------------
    # Delegation properties — InputState
    # ------------------------------------------------------------------

    @property
    def _held_keys(self):
        return self.input.held_keys

    @_held_keys.setter
    def _held_keys(self, value):
        self.input.held_keys = value

    @property
    def _mouse_dragging(self):
        return self.input.mouse_dragging

    @_mouse_dragging.setter
    def _mouse_dragging(self, value):
        self.input.mouse_dragging = value

    @property
    def _mouse_last_x(self):
        return self.input.mouse_last_x

    @_mouse_last_x.setter
    def _mouse_last_x(self, value):
        self.input.mouse_last_x = value

    @property
    def _mouse_last_y(self):
        return self.input.mouse_last_y

    @_mouse_last_y.setter
    def _mouse_last_y(self, value):
        self.input.mouse_last_y = value

    # ------------------------------------------------------------------
    # Delegation properties — CameraState
    # ------------------------------------------------------------------

    @property
    def position(self):
        return self.camera.position

    @position.setter
    def position(self, value):
        self.camera.position = value

    @property
    def yaw(self):
        return self.camera.yaw

    @yaw.setter
    def yaw(self, value):
        self.camera.yaw = value

    @property
    def pitch(self):
        return self.camera.pitch

    @pitch.setter
    def pitch(self, value):
        self.camera.pitch = value

    @property
    def fov(self):
        return self.camera.fov

    @fov.setter
    def fov(self, value):
        self.camera.fov = value

    @property
    def move_speed(self):
        return self.camera.move_speed

    @move_speed.setter
    def move_speed(self, value):
        self.camera.move_speed = value

    @property
    def look_speed(self):
        return self.camera.look_speed

    @look_speed.setter
    def look_speed(self, value):
        self.camera.look_speed = value

    @property
    def _time_presets(self):
        return self.camera._time_presets

    @property
    def _time_preset_idx(self):
        return self.camera._time_preset_idx

    @_time_preset_idx.setter
    def _time_preset_idx(self, value):
        self.camera._time_preset_idx = value

    # ------------------------------------------------------------------
    # Delegation properties — RenderSettings
    # ------------------------------------------------------------------

    @property
    def shadows(self):
        return self.render_settings.shadows

    @shadows.setter
    def shadows(self, value):
        self.render_settings.shadows = value

    @property
    def ambient(self):
        return self.render_settings.ambient

    @ambient.setter
    def ambient(self, value):
        self.render_settings.ambient = value

    @property
    def sun_azimuth(self):
        return self.render_settings.sun_azimuth

    @sun_azimuth.setter
    def sun_azimuth(self, value):
        self.render_settings.sun_azimuth = value

    @property
    def sun_altitude(self):
        return self.render_settings.sun_altitude

    @sun_altitude.setter
    def sun_altitude(self, value):
        self.render_settings.sun_altitude = value

    @property
    def fog_density(self):
        return self.render_settings.fog_density

    @fog_density.setter
    def fog_density(self, value):
        self.render_settings.fog_density = value

    @property
    def fog_color(self):
        return self.render_settings.fog_color

    @fog_color.setter
    def fog_color(self, value):
        self.render_settings.fog_color = value

    @property
    def colormap(self):
        return self.render_settings.colormap

    @colormap.setter
    def colormap(self, value):
        self.render_settings.colormap = value

    @property
    def colormaps(self):
        return self.render_settings.colormaps

    @property
    def colormap_idx(self):
        return self.render_settings.colormap_idx

    @colormap_idx.setter
    def colormap_idx(self, value):
        self.render_settings.colormap_idx = value

    @property
    def color_stretch(self):
        return self.render_settings.color_stretch

    @color_stretch.setter
    def color_stretch(self, value):
        self.render_settings.color_stretch = value

    @property
    def _color_stretches(self):
        return self.render_settings._color_stretches

    @property
    def _color_stretch_idx(self):
        return self.render_settings._color_stretch_idx

    @_color_stretch_idx.setter
    def _color_stretch_idx(self, value):
        self.render_settings._color_stretch_idx = value

    @property
    def ao_enabled(self):
        return self.render_settings.ao_enabled

    @ao_enabled.setter
    def ao_enabled(self, value):
        self.render_settings.ao_enabled = value

    @property
    def ao_radius(self):
        return self.render_settings.ao_radius

    @ao_radius.setter
    def ao_radius(self, value):
        self.render_settings.ao_radius = value

    @property
    def gi_intensity(self):
        return self.render_settings.gi_intensity

    @gi_intensity.setter
    def gi_intensity(self, value):
        self.render_settings.gi_intensity = value

    @property
    def gi_bounces(self):
        return self.render_settings.gi_bounces

    @gi_bounces.setter
    def gi_bounces(self, value):
        self.render_settings.gi_bounces = value

    @property
    def _ao_samples_per_frame(self):
        return self.render_settings._ao_samples_per_frame

    @property
    def _ao_max_frames(self):
        return self.render_settings._ao_max_frames

    @property
    def _ao_frame_count(self):
        return self.render_settings._ao_frame_count

    @_ao_frame_count.setter
    def _ao_frame_count(self, value):
        self.render_settings._ao_frame_count = value

    @property
    def _d_ao_accum(self):
        return self.render_settings._d_ao_accum

    @_d_ao_accum.setter
    def _d_ao_accum(self, value):
        self.render_settings._d_ao_accum = value

    @property
    def _prev_cam_state(self):
        return self.render_settings._prev_cam_state

    @_prev_cam_state.setter
    def _prev_cam_state(self, value):
        self.render_settings._prev_cam_state = value

    @property
    def edl_enabled(self):
        return self.render_settings.edl_enabled

    @edl_enabled.setter
    def edl_enabled(self, value):
        self.render_settings.edl_enabled = value

    @property
    def denoise_enabled(self):
        return self.render_settings.denoise_enabled

    @denoise_enabled.setter
    def denoise_enabled(self, value):
        self.render_settings.denoise_enabled = value

    @property
    def _prev_cam_for_flow(self):
        return self.render_settings._prev_cam_for_flow

    @_prev_cam_for_flow.setter
    def _prev_cam_for_flow(self, value):
        self.render_settings._prev_cam_for_flow = value

    @property
    def _d_flow(self):
        return self.render_settings._d_flow

    @_d_flow.setter
    def _d_flow(self, value):
        self.render_settings._d_flow = value

    @property
    def dof_enabled(self):
        return self.render_settings.dof_enabled

    @dof_enabled.setter
    def dof_enabled(self, value):
        self.render_settings.dof_enabled = value

    @property
    def _dof_aperture(self):
        return self.render_settings._dof_aperture

    @_dof_aperture.setter
    def _dof_aperture(self, value):
        self.render_settings._dof_aperture = value

    @property
    def _dof_focal_distance(self):
        return self.render_settings._dof_focal_distance

    @_dof_focal_distance.setter
    def _dof_focal_distance(self, value):
        self.render_settings._dof_focal_distance = value

    # ------------------------------------------------------------------
    # Delegation properties — OverlayManager
    # ------------------------------------------------------------------

    @property
    def _overlay_layers(self):
        return self.overlays.overlay_layers

    @_overlay_layers.setter
    def _overlay_layers(self, value):
        self.overlays.overlay_layers = value

    @property
    def _overlay_names(self):
        return self.overlays.overlay_names

    @_overlay_names.setter
    def _overlay_names(self, value):
        self.overlays.overlay_names = value

    @property
    def _active_color_data(self):
        return self.overlays.active_color_data

    @_active_color_data.setter
    def _active_color_data(self, value):
        self.overlays.active_color_data = value

    @property
    def _active_overlay_data(self):
        return self.overlays.active_overlay_data

    @_active_overlay_data.setter
    def _active_overlay_data(self, value):
        self.overlays.active_overlay_data = value

    @property
    def _overlay_alpha(self):
        return self.overlays.overlay_alpha

    @_overlay_alpha.setter
    def _overlay_alpha(self, value):
        self.overlays.overlay_alpha = value

    @property
    def _overlay_as_water(self):
        return self.overlays.overlay_as_water

    @_overlay_as_water.setter
    def _overlay_as_water(self, value):
        self.overlays.overlay_as_water = value

    @property
    def _active_overlay_color_lut(self):
        return self.overlays.active_overlay_color_lut

    @_active_overlay_color_lut.setter
    def _active_overlay_color_lut(self, value):
        self.overlays.active_overlay_color_lut = value

    @property
    def _overlay_color_luts(self):
        return self.overlays.overlay_color_luts

    @_overlay_color_luts.setter
    def _overlay_color_luts(self, value):
        self.overlays.overlay_color_luts = value

    @property
    def _terrain_layer_order(self):
        return self.overlays.terrain_layer_order

    @_terrain_layer_order.setter
    def _terrain_layer_order(self, value):
        self.overlays.terrain_layer_order = value

    @property
    def _terrain_layer_idx(self):
        return self.overlays.terrain_layer_idx

    @_terrain_layer_idx.setter
    def _terrain_layer_idx(self, value):
        self.overlays.terrain_layer_idx = value

    @property
    def _base_overlay_layers(self):
        return self.overlays.base_overlay_layers

    @_base_overlay_layers.setter
    def _base_overlay_layers(self, value):
        self.overlays.base_overlay_layers = value

    @property
    def _tile_service(self):
        return self.overlays.tile_service

    @_tile_service.setter
    def _tile_service(self, value):
        self.overlays.tile_service = value

    @property
    def _tiles_enabled(self):
        return self.overlays.tiles_enabled

    @_tiles_enabled.setter
    def _tiles_enabled(self, value):
        self.overlays.tiles_enabled = value

    @property
    def _basemap_options(self):
        return self.overlays.basemap_options

    @property
    def _basemap_idx(self):
        return self.overlays.basemap_idx

    @_basemap_idx.setter
    def _basemap_idx(self, value):
        self.overlays.basemap_idx = value

    # ------------------------------------------------------------------
    # Delegation properties — HUDState
    # ------------------------------------------------------------------

    @property
    def _title(self):
        return self.hud.title

    @_title.setter
    def _title(self, value):
        self.hud.title = value

    @property
    def _subtitle(self):
        return self.hud.subtitle

    @_subtitle.setter
    def _subtitle(self, value):
        self.hud.subtitle = value

    @property
    def _legend_config(self):
        return self.hud.legend_config

    @_legend_config.setter
    def _legend_config(self, value):
        self.hud.legend_config = value

    @property
    def _info_text(self):
        return self.hud.info_text

    @_info_text.setter
    def _info_text(self, value):
        self.hud.info_text = value

    @property
    def _title_overlay_rgba(self):
        return self.hud.title_overlay_rgba

    @_title_overlay_rgba.setter
    def _title_overlay_rgba(self, value):
        self.hud.title_overlay_rgba = value

    @property
    def _legend_rgba(self):
        return self.hud.legend_rgba

    @_legend_rgba.setter
    def _legend_rgba(self, value):
        self.hud.legend_rgba = value

    @property
    def _help_page_idx(self):
        return self.hud.help_page_idx

    @_help_page_idx.setter
    def _help_page_idx(self, value):
        self.hud.help_page_idx = value

    @property
    def _help_pages(self):
        return self.hud.help_pages

    @_help_pages.setter
    def _help_pages(self, value):
        self.hud.help_pages = value

    @property
    def show_minimap(self):
        return self.hud.show_minimap

    @show_minimap.setter
    def show_minimap(self, value):
        self.hud.show_minimap = value

    @property
    def _last_title(self):
        return self.hud.last_title

    @_last_title.setter
    def _last_title(self, value):
        self.hud.last_title = value

    @property
    def _last_subtitle(self):
        return self.hud.last_subtitle

    @_last_subtitle.setter
    def _last_subtitle(self, value):
        self.hud.last_subtitle = value

    @property
    def _minimap_background(self):
        return self.hud.minimap_background

    @_minimap_background.setter
    def _minimap_background(self, value):
        self.hud.minimap_background = value

    @property
    def _minimap_scale_x(self):
        return self.hud.minimap_scale_x

    @_minimap_scale_x.setter
    def _minimap_scale_x(self, value):
        self.hud.minimap_scale_x = value

    @property
    def _minimap_scale_y(self):
        return self.hud.minimap_scale_y

    @_minimap_scale_y.setter
    def _minimap_scale_y(self, value):
        self.hud.minimap_scale_y = value

    @property
    def _minimap_has_tiles(self):
        return self.hud.minimap_has_tiles

    @_minimap_has_tiles.setter
    def _minimap_has_tiles(self, value):
        self.hud.minimap_has_tiles = value

    @property
    def _minimap_rect(self):
        return self.hud.minimap_rect

    @_minimap_rect.setter
    def _minimap_rect(self, value):
        self.hud.minimap_rect = value

    @property
    def _minimap_world_extent(self):
        return self.hud.minimap_world_extent

    @_minimap_world_extent.setter
    def _minimap_world_extent(self, value):
        self.hud.minimap_world_extent = value

    @property
    def _minimap_style(self):
        return self.hud.minimap_style

    @_minimap_style.setter
    def _minimap_style(self, value):
        self.hud.minimap_style = value

    @property
    def _minimap_layer(self):
        return self.hud.minimap_layer

    @_minimap_layer.setter
    def _minimap_layer(self, value):
        self.hud.minimap_layer = value

    @property
    def _minimap_colors(self):
        return self.hud.minimap_colors

    @_minimap_colors.setter
    def _minimap_colors(self, value):
        self.hud.minimap_colors = value

    @property
    def _minimap_bg_extent(self):
        return self.hud.minimap_bg_extent

    @_minimap_bg_extent.setter
    def _minimap_bg_extent(self, value):
        self.hud.minimap_bg_extent = value

    @property
    def _minimap_last_stream_time(self):
        return self.hud.minimap_last_stream_time

    @_minimap_last_stream_time.setter
    def _minimap_last_stream_time(self, value):
        self.hud.minimap_last_stream_time = value

    # ------------------------------------------------------------------
    # Delegation properties — WindState
    # ------------------------------------------------------------------

    @property
    def _wind_data(self):
        return self.wind.wind_data

    @_wind_data.setter
    def _wind_data(self, value):
        self.wind.wind_data = value

    @property
    def _wind_enabled(self):
        return self.wind.wind_enabled

    @_wind_enabled.setter
    def _wind_enabled(self, value):
        self.wind.wind_enabled = value

    @property
    def _wind_u_px(self):
        return self.wind.wind_u_px

    @_wind_u_px.setter
    def _wind_u_px(self, value):
        self.wind.wind_u_px = value

    @property
    def _wind_v_px(self):
        return self.wind.wind_v_px

    @_wind_v_px.setter
    def _wind_v_px(self, value):
        self.wind.wind_v_px = value

    @property
    def _wind_particles(self):
        return self.wind.wind_particles

    @_wind_particles.setter
    def _wind_particles(self, value):
        self.wind.wind_particles = value

    @property
    def _wind_ages(self):
        return self.wind.wind_ages

    @_wind_ages.setter
    def _wind_ages(self, value):
        self.wind.wind_ages = value

    @property
    def _wind_max_age(self):
        return self.wind.wind_max_age

    @_wind_max_age.setter
    def _wind_max_age(self, value):
        self.wind.wind_max_age = value

    @property
    def _wind_n_particles(self):
        return self.wind.wind_n_particles

    @_wind_n_particles.setter
    def _wind_n_particles(self, value):
        self.wind.wind_n_particles = value

    @property
    def _wind_trail_len(self):
        return self.wind.wind_trail_len

    @_wind_trail_len.setter
    def _wind_trail_len(self, value):
        self.wind.wind_trail_len = value

    @property
    def _wind_trails(self):
        return self.wind.wind_trails

    @_wind_trails.setter
    def _wind_trails(self, value):
        self.wind.wind_trails = value

    @property
    def _wind_speed_mult(self):
        return self.wind.wind_speed_mult

    @_wind_speed_mult.setter
    def _wind_speed_mult(self, value):
        self.wind.wind_speed_mult = value

    @property
    def _wind_min_depth(self):
        return self.wind.wind_min_depth

    @_wind_min_depth.setter
    def _wind_min_depth(self, value):
        self.wind.wind_min_depth = value

    @property
    def _wind_dot_radius(self):
        return self.wind.wind_dot_radius

    @_wind_dot_radius.setter
    def _wind_dot_radius(self, value):
        self.wind.wind_dot_radius = value

    @property
    def _wind_alpha(self):
        return self.wind.wind_alpha

    @_wind_alpha.setter
    def _wind_alpha(self, value):
        self.wind.wind_alpha = value

    @property
    def _wind_min_visible_age(self):
        return self.wind.wind_min_visible_age

    @_wind_min_visible_age.setter
    def _wind_min_visible_age(self, value):
        self.wind.wind_min_visible_age = value

    @property
    def _wind_terrain_np(self):
        return self.wind.wind_terrain_np

    @_wind_terrain_np.setter
    def _wind_terrain_np(self, value):
        self.wind.wind_terrain_np = value

    @property
    def _d_wind_trails(self):
        return self.wind.d_wind_trails

    @_d_wind_trails.setter
    def _d_wind_trails(self, value):
        self.wind.d_wind_trails = value

    @property
    def _d_wind_alpha(self):
        return self.wind.d_wind_alpha

    @_d_wind_alpha.setter
    def _d_wind_alpha(self, value):
        self.wind.d_wind_alpha = value

    @property
    def _d_base_frame(self):
        return self.wind.d_base_frame

    @_d_base_frame.setter
    def _d_base_frame(self, value):
        self.wind.d_base_frame = value

    @property
    def _d_wind_scratch(self):
        return self.wind.d_wind_scratch

    @_d_wind_scratch.setter
    def _d_wind_scratch(self, value):
        self.wind.d_wind_scratch = value

    @property
    def _wind_done_event(self):
        return self.wind.wind_done_event

    @_wind_done_event.setter
    def _wind_done_event(self, value):
        self.wind.wind_done_event = value

    # ------------------------------------------------------------------
    # Delegation properties — CloudState
    # ------------------------------------------------------------------

    @property
    def _clouds_enabled(self):
        return self.clouds.clouds_enabled

    @_clouds_enabled.setter
    def _clouds_enabled(self, value):
        self.clouds.clouds_enabled = value

    @property
    def _cloud_cover_grid(self):
        return self.clouds.cloud_cover_grid

    @_cloud_cover_grid.setter
    def _cloud_cover_grid(self, value):
        self.clouds.cloud_cover_grid = value

    @property
    def _cloud_particles(self):
        return self.clouds.cloud_particles

    @_cloud_particles.setter
    def _cloud_particles(self, value):
        self.clouds.cloud_particles = value

    @property
    def _cloud_sizes(self):
        return self.clouds.cloud_sizes

    @_cloud_sizes.setter
    def _cloud_sizes(self, value):
        self.clouds.cloud_sizes = value

    @property
    def _cloud_alphas(self):
        return self.clouds.cloud_alphas

    @_cloud_alphas.setter
    def _cloud_alphas(self, value):
        self.clouds.cloud_alphas = value

    @property
    def _cloud_ages(self):
        return self.clouds.cloud_ages

    @_cloud_ages.setter
    def _cloud_ages(self, value):
        self.clouds.cloud_ages = value

    @property
    def _cloud_lifetimes(self):
        return self.clouds.cloud_lifetimes

    @_cloud_lifetimes.setter
    def _cloud_lifetimes(self, value):
        self.clouds.cloud_lifetimes = value

    @property
    def _cloud_n_particles(self):
        return self.clouds.cloud_n_particles

    @_cloud_n_particles.setter
    def _cloud_n_particles(self, value):
        self.clouds.cloud_n_particles = value

    @property
    def _cloud_max_age(self):
        return self.clouds.cloud_max_age

    @_cloud_max_age.setter
    def _cloud_max_age(self, value):
        self.clouds.cloud_max_age = value

    @property
    def _cloud_altitude(self):
        return self.clouds.cloud_altitude

    @_cloud_altitude.setter
    def _cloud_altitude(self, value):
        self.clouds.cloud_altitude = value

    @property
    def _cloud_min_depth(self):
        return self.clouds.cloud_min_depth

    @_cloud_min_depth.setter
    def _cloud_min_depth(self, value):
        self.clouds.cloud_min_depth = value

    @property
    def _cloud_terrain_np(self):
        return self.clouds.cloud_terrain_np

    @_cloud_terrain_np.setter
    def _cloud_terrain_np(self, value):
        self.clouds.cloud_terrain_np = value

    @property
    def _volumetric_clouds_enabled(self):
        return self.clouds.volumetric_clouds_enabled

    @_volumetric_clouds_enabled.setter
    def _volumetric_clouds_enabled(self, value):
        self.clouds.volumetric_clouds_enabled = value

    @property
    def _cloud_thickness(self):
        return self.clouds.cloud_thickness

    @_cloud_thickness.setter
    def _cloud_thickness(self, value):
        self.clouds.cloud_thickness = value

    @property
    def _cloud_time(self):
        return self.clouds.cloud_time

    @_cloud_time.setter
    def _cloud_time(self, value):
        self.clouds.cloud_time = value

    # ------------------------------------------------------------------
    # Delegation properties — HydroState
    # ------------------------------------------------------------------

    @property
    def _hydro_data(self):
        return self.hydro.hydro_data

    @_hydro_data.setter
    def _hydro_data(self, value):
        self.hydro.hydro_data = value

    @property
    def _hydro_lazy(self):
        return self.hydro.hydro_lazy

    @_hydro_lazy.setter
    def _hydro_lazy(self, value):
        self.hydro.hydro_lazy = value

    @property
    def _hydro_enabled(self):
        return self.hydro.hydro_enabled

    @_hydro_enabled.setter
    def _hydro_enabled(self, value):
        self.hydro.hydro_enabled = value

    @property
    def _hydro_flow_u_px(self):
        return self.hydro.hydro_flow_u_px

    @_hydro_flow_u_px.setter
    def _hydro_flow_u_px(self, value):
        self.hydro.hydro_flow_u_px = value

    @property
    def _hydro_flow_v_px(self):
        return self.hydro.hydro_flow_v_px

    @_hydro_flow_v_px.setter
    def _hydro_flow_v_px(self, value):
        self.hydro.hydro_flow_v_px = value

    @property
    def _hydro_flow_accum_norm(self):
        return self.hydro.hydro_flow_accum_norm

    @_hydro_flow_accum_norm.setter
    def _hydro_flow_accum_norm(self, value):
        self.hydro.hydro_flow_accum_norm = value

    @property
    def _hydro_stream_order(self):
        return self.hydro.hydro_stream_order

    @_hydro_stream_order.setter
    def _hydro_stream_order(self, value):
        self.hydro.hydro_stream_order = value

    @property
    def _hydro_stream_order_raw(self):
        return self.hydro.hydro_stream_order_raw

    @_hydro_stream_order_raw.setter
    def _hydro_stream_order_raw(self, value):
        self.hydro.hydro_stream_order_raw = value

    @property
    def _hydro_stream_link(self):
        return self.hydro.hydro_stream_link

    @_hydro_stream_link.setter
    def _hydro_stream_link(self, value):
        self.hydro.hydro_stream_link = value

    @property
    def _hydro_particle_raw_order(self):
        return self.hydro.hydro_particle_raw_order

    @_hydro_particle_raw_order.setter
    def _hydro_particle_raw_order(self, value):
        self.hydro.hydro_particle_raw_order = value

    @property
    def _hydro_particles(self):
        return self.hydro.hydro_particles

    @_hydro_particles.setter
    def _hydro_particles(self, value):
        self.hydro.hydro_particles = value

    @property
    def _hydro_ages(self):
        return self.hydro.hydro_ages

    @_hydro_ages.setter
    def _hydro_ages(self, value):
        self.hydro.hydro_ages = value

    @property
    def _hydro_lifetimes(self):
        return self.hydro.hydro_lifetimes

    @_hydro_lifetimes.setter
    def _hydro_lifetimes(self, value):
        self.hydro.hydro_lifetimes = value

    @property
    def _hydro_max_age(self):
        return self.hydro.hydro_max_age

    @property
    def _hydro_n_particles(self):
        return self.hydro.hydro_n_particles

    @_hydro_n_particles.setter
    def _hydro_n_particles(self, value):
        self.hydro.hydro_n_particles = value

    @property
    def _hydro_trail_len(self):
        return self.hydro.hydro_trail_len

    @_hydro_trail_len.setter
    def _hydro_trail_len(self, value):
        self.hydro.hydro_trail_len = value

    @property
    def _hydro_trails(self):
        return self.hydro.hydro_trails

    @_hydro_trails.setter
    def _hydro_trails(self, value):
        self.hydro.hydro_trails = value

    @property
    def _hydro_speed(self):
        return self.hydro.hydro_speed

    @_hydro_speed.setter
    def _hydro_speed(self, value):
        self.hydro.hydro_speed = value

    @property
    def _hydro_min_depth(self):
        return self.hydro.hydro_min_depth

    @_hydro_min_depth.setter
    def _hydro_min_depth(self, value):
        self.hydro.hydro_min_depth = value

    @property
    def _hydro_max_depth(self):
        return self.hydro.hydro_max_depth

    @_hydro_max_depth.setter
    def _hydro_max_depth(self, value):
        self.hydro.hydro_max_depth = value

    @property
    def _hydro_ref_depth(self):
        return self.hydro.hydro_ref_depth

    @_hydro_ref_depth.setter
    def _hydro_ref_depth(self, value):
        self.hydro.hydro_ref_depth = value

    @property
    def _hydro_dot_radius(self):
        return self.hydro.hydro_dot_radius

    @_hydro_dot_radius.setter
    def _hydro_dot_radius(self, value):
        self.hydro.hydro_dot_radius = value

    @property
    def _hydro_alpha(self):
        return self.hydro.hydro_alpha

    @_hydro_alpha.setter
    def _hydro_alpha(self, value):
        self.hydro.hydro_alpha = value

    @property
    def _hydro_min_visible_age(self):
        return self.hydro.hydro_min_visible_age

    @property
    def _hydro_accum_threshold(self):
        return self.hydro.hydro_accum_threshold

    @_hydro_accum_threshold.setter
    def _hydro_accum_threshold(self, value):
        self.hydro.hydro_accum_threshold = value

    @property
    def _hydro_color(self):
        return self.hydro.hydro_color

    @_hydro_color.setter
    def _hydro_color(self, value):
        self.hydro.hydro_color = value

    @property
    def _hydro_terrain_np(self):
        return self.hydro.hydro_terrain_np

    @_hydro_terrain_np.setter
    def _hydro_terrain_np(self, value):
        self.hydro.hydro_terrain_np = value

    @property
    def _hydro_spawn_probs(self):
        return self.hydro.hydro_spawn_probs

    @_hydro_spawn_probs.setter
    def _hydro_spawn_probs(self, value):
        self.hydro.hydro_spawn_probs = value

    @property
    def _hydro_spawn_indices(self):
        return self.hydro.hydro_spawn_indices

    @_hydro_spawn_indices.setter
    def _hydro_spawn_indices(self, value):
        self.hydro.hydro_spawn_indices = value

    @property
    def _hydro_spawn_valid_probs(self):
        return self.hydro.hydro_spawn_valid_probs

    @_hydro_spawn_valid_probs.setter
    def _hydro_spawn_valid_probs(self, value):
        self.hydro.hydro_spawn_valid_probs = value

    @property
    def _d_hydro_trails(self):
        return self.hydro.d_hydro_trails

    @_d_hydro_trails.setter
    def _d_hydro_trails(self, value):
        self.hydro.d_hydro_trails = value

    @property
    def _d_hydro_ages(self):
        return self.hydro.d_hydro_ages

    @_d_hydro_ages.setter
    def _d_hydro_ages(self, value):
        self.hydro.d_hydro_ages = value

    @property
    def _d_hydro_lifetimes(self):
        return self.hydro.d_hydro_lifetimes

    @_d_hydro_lifetimes.setter
    def _d_hydro_lifetimes(self, value):
        self.hydro.d_hydro_lifetimes = value

    @property
    def _hydro_done_event(self):
        return self.hydro.hydro_done_event

    @_hydro_done_event.setter
    def _hydro_done_event(self, value):
        self.hydro.hydro_done_event = value

    @property
    def _hydro_slope_mag(self):
        return self.hydro.hydro_slope_mag

    @_hydro_slope_mag.setter
    def _hydro_slope_mag(self, value):
        self.hydro.hydro_slope_mag = value

    @property
    def _hydro_particle_accum(self):
        return self.hydro.hydro_particle_accum

    @_hydro_particle_accum.setter
    def _hydro_particle_accum(self, value):
        self.hydro.hydro_particle_accum = value

    @property
    def _d_hydro_colors(self):
        return self.hydro.d_hydro_colors

    @_d_hydro_colors.setter
    def _d_hydro_colors(self, value):
        self.hydro.d_hydro_colors = value

    @property
    def _d_hydro_radii(self):
        return self.hydro.d_hydro_radii

    @_d_hydro_radii.setter
    def _d_hydro_radii(self, value):
        self.hydro.d_hydro_radii = value

    @property
    def _d_hydro_particles(self):
        return self.hydro.d_hydro_particles

    @_d_hydro_particles.setter
    def _d_hydro_particles(self, value):
        self.hydro.d_hydro_particles = value

    @property
    def _d_hydro_particle_accum(self):
        return self.hydro.d_hydro_particle_accum

    @_d_hydro_particle_accum.setter
    def _d_hydro_particle_accum(self, value):
        self.hydro.d_hydro_particle_accum = value

    @property
    def _d_hydro_particle_raw_order(self):
        return self.hydro.d_hydro_particle_raw_order

    @_d_hydro_particle_raw_order.setter
    def _d_hydro_particle_raw_order(self, value):
        self.hydro.d_hydro_particle_raw_order = value

    @property
    def _d_hydro_flow_u(self):
        return self.hydro.d_hydro_flow_u

    @_d_hydro_flow_u.setter
    def _d_hydro_flow_u(self, value):
        self.hydro.d_hydro_flow_u = value

    @property
    def _d_hydro_flow_v(self):
        return self.hydro.d_hydro_flow_v

    @_d_hydro_flow_v.setter
    def _d_hydro_flow_v(self, value):
        self.hydro.d_hydro_flow_v = value

    @property
    def _d_hydro_slope_mag(self):
        return self.hydro.d_hydro_slope_mag

    @_d_hydro_slope_mag.setter
    def _d_hydro_slope_mag(self, value):
        self.hydro.d_hydro_slope_mag = value

    @property
    def _d_hydro_stream_order(self):
        return self.hydro.d_hydro_stream_order

    @_d_hydro_stream_order.setter
    def _d_hydro_stream_order(self, value):
        self.hydro.d_hydro_stream_order = value

    @property
    def _d_hydro_stream_order_raw(self):
        return self.hydro.d_hydro_stream_order_raw

    @_d_hydro_stream_order_raw.setter
    def _d_hydro_stream_order_raw(self, value):
        self.hydro.d_hydro_stream_order_raw = value

    @property
    def _d_hydro_accum_norm(self):
        return self.hydro.d_hydro_accum_norm

    @_d_hydro_accum_norm.setter
    def _d_hydro_accum_norm(self, value):
        self.hydro.d_hydro_accum_norm = value

    @property
    def _d_hydro_palette(self):
        return self.hydro.d_hydro_palette

    @_d_hydro_palette.setter
    def _d_hydro_palette(self, value):
        self.hydro.d_hydro_palette = value

    @property
    def _d_hydro_respawn_flags(self):
        return self.hydro.d_hydro_respawn_flags

    @_d_hydro_respawn_flags.setter
    def _d_hydro_respawn_flags(self, value):
        self.hydro.d_hydro_respawn_flags = value

    @property
    def _hydro_particle_colors(self):
        return self.hydro.hydro_particle_colors

    @_hydro_particle_colors.setter
    def _hydro_particle_colors(self, value):
        self.hydro.hydro_particle_colors = value

    @property
    def _hydro_particle_radii(self):
        return self.hydro.hydro_particle_radii

    @_hydro_particle_radii.setter
    def _hydro_particle_radii(self, value):
        self.hydro.hydro_particle_radii = value

    # ------------------------------------------------------------------
    # Delegation properties — ObserverManager
    # ------------------------------------------------------------------

    @property
    def _observers(self):
        return self.observer_mgr.observers

    @_observers.setter
    def _observers(self, value):
        self.observer_mgr.observers = value

    @property
    def _active_observer(self):
        return self.observer_mgr.active_observer

    @_active_observer.setter
    def _active_observer(self, value):
        self.observer_mgr.active_observer = value

    @property
    def viewshed_enabled(self):
        return self.observer_mgr.viewshed_enabled

    @viewshed_enabled.setter
    def viewshed_enabled(self, value):
        self.observer_mgr.viewshed_enabled = value

    @property
    def viewshed_observer_elev(self):
        return self.observer_mgr.viewshed_observer_elev

    @viewshed_observer_elev.setter
    def viewshed_observer_elev(self, value):
        self.observer_mgr.viewshed_observer_elev = value

    @property
    def viewshed_target_elev(self):
        return self.observer_mgr.viewshed_target_elev

    @viewshed_target_elev.setter
    def viewshed_target_elev(self, value):
        self.observer_mgr.viewshed_target_elev = value

    @property
    def viewshed_opacity(self):
        return self.observer_mgr.viewshed_opacity

    @viewshed_opacity.setter
    def viewshed_opacity(self, value):
        self.observer_mgr.viewshed_opacity = value

    @property
    def _viewshed_cache(self):
        return self.observer_mgr.viewshed_cache

    @_viewshed_cache.setter
    def _viewshed_cache(self, value):
        self.observer_mgr.viewshed_cache = value

    @property
    def _viewshed_coverage(self):
        return self.observer_mgr.viewshed_coverage

    @_viewshed_coverage.setter
    def _viewshed_coverage(self, value):
        self.observer_mgr.viewshed_coverage = value

    @property
    def _viewshed_recalc_interval(self):
        return self.observer_mgr.viewshed_recalc_interval

    @property
    def _last_viewshed_time(self):
        return self.observer_mgr.last_viewshed_time

    @_last_viewshed_time.setter
    def _last_viewshed_time(self, value):
        self.observer_mgr.last_viewshed_time = value

    @property
    def _shared_drone_parts(self):
        return self.observer_mgr.shared_drone_parts

    @_shared_drone_parts.setter
    def _shared_drone_parts(self, value):
        self.observer_mgr.shared_drone_parts = value

    @property
    def _drone_glow(self):
        return self.observer_mgr.drone_glow

    @_drone_glow.setter
    def _drone_glow(self, value):
        self.observer_mgr.drone_glow = value

    # ------------------------------------------------------------------
    # Delegation properties — TerrainState
    # ------------------------------------------------------------------

    @property
    def raster(self):
        return self.terrain.raster

    @raster.setter
    def raster(self, value):
        self.terrain.raster = value

    @property
    def _base_raster(self):
        return self.terrain._base_raster

    @_base_raster.setter
    def _base_raster(self, value):
        self.terrain._base_raster = value

    @property
    def terrain_shape(self):
        return self.terrain.terrain_shape

    @terrain_shape.setter
    def terrain_shape(self, value):
        self.terrain.terrain_shape = value

    @property
    def elev_min(self):
        return self.terrain.elev_min

    @elev_min.setter
    def elev_min(self, value):
        self.terrain.elev_min = value

    @property
    def elev_max(self):
        return self.terrain.elev_max

    @elev_max.setter
    def elev_max(self, value):
        self.terrain.elev_max = value

    @property
    def elev_mean(self):
        return self.terrain.elev_mean

    @elev_mean.setter
    def elev_mean(self, value):
        self.terrain.elev_mean = value

    @property
    def pixel_spacing_x(self):
        return self.terrain.pixel_spacing_x

    @pixel_spacing_x.setter
    def pixel_spacing_x(self, value):
        self.terrain.pixel_spacing_x = value

    @property
    def pixel_spacing_y(self):
        return self.terrain.pixel_spacing_y

    @pixel_spacing_y.setter
    def pixel_spacing_y(self, value):
        self.terrain.pixel_spacing_y = value

    @property
    def _base_pixel_spacing_x(self):
        return self.terrain._base_pixel_spacing_x

    @_base_pixel_spacing_x.setter
    def _base_pixel_spacing_x(self, value):
        self.terrain._base_pixel_spacing_x = value

    @property
    def _base_pixel_spacing_y(self):
        return self.terrain._base_pixel_spacing_y

    @_base_pixel_spacing_y.setter
    def _base_pixel_spacing_y(self, value):
        self.terrain._base_pixel_spacing_y = value

    @property
    def subsample_factor(self):
        return self.terrain.subsample_factor

    @subsample_factor.setter
    def subsample_factor(self, value):
        self.terrain.subsample_factor = value

    @property
    def _baked_mesh_cache(self):
        return self.terrain._baked_mesh_cache

    @_baked_mesh_cache.setter
    def _baked_mesh_cache(self, value):
        self.terrain._baked_mesh_cache = value

    @property
    def _gpu_terrain(self):
        return self.terrain._gpu_terrain

    @_gpu_terrain.setter
    def _gpu_terrain(self, value):
        self.terrain._gpu_terrain = value

    @property
    def _gpu_base_terrain(self):
        return self.terrain._gpu_base_terrain

    @_gpu_base_terrain.setter
    def _gpu_base_terrain(self, value):
        self.terrain._gpu_base_terrain = value


    @property
    def terrain_skirt(self):
        return self.terrain.terrain_skirt

    @terrain_skirt.setter
    def terrain_skirt(self, value):
        self.terrain.terrain_skirt = value

    @property
    def _water_mask(self):
        return self.terrain._water_mask

    @_water_mask.setter
    def _water_mask(self, value):
        self.terrain._water_mask = value

    @property
    def vertical_exaggeration(self):
        return self.terrain.vertical_exaggeration

    @vertical_exaggeration.setter
    def vertical_exaggeration(self, value):
        self.terrain.vertical_exaggeration = value

    @property
    def _land_color_range(self):
        return self.terrain._land_color_range

    @_land_color_range.setter
    def _land_color_range(self, value):
        self.terrain._land_color_range = value

    @property
    def _terrain_loader(self):
        return self.terrain._terrain_loader

    @_terrain_loader.setter
    def _terrain_loader(self, value):
        self.terrain._terrain_loader = value

    @property
    def _coord_origin_x(self):
        return self.terrain._coord_origin_x

    @_coord_origin_x.setter
    def _coord_origin_x(self, value):
        self.terrain._coord_origin_x = value

    @property
    def _coord_origin_y(self):
        return self.terrain._coord_origin_y

    @_coord_origin_y.setter
    def _coord_origin_y(self, value):
        self.terrain._coord_origin_y = value

    @property
    def _coord_step_x(self):
        return self.terrain._coord_step_x

    @_coord_step_x.setter
    def _coord_step_x(self, value):
        self.terrain._coord_step_x = value

    @property
    def _coord_step_y(self):
        return self.terrain._coord_step_y

    @_coord_step_y.setter
    def _coord_step_y(self, value):
        self.terrain._coord_step_y = value

    @property
    def _reload_cooldown(self):
        return self.terrain._reload_cooldown

    @property
    def _last_reload_time(self):
        return self.terrain._last_reload_time

    @_last_reload_time.setter
    def _last_reload_time(self, value):
        self.terrain._last_reload_time = value

    @property
    def lod_enabled(self):
        return self.terrain.lod_enabled

    @lod_enabled.setter
    def lod_enabled(self, value):
        self.terrain.lod_enabled = value

    @property
    def _terrain_lod_manager(self):
        return self.terrain._terrain_lod_manager

    @_terrain_lod_manager.setter
    def _terrain_lod_manager(self, value):
        self.terrain._terrain_lod_manager = value

    # ------------------------------------------------------------------
    # Delegation properties — GeometryLayerManager
    # ------------------------------------------------------------------

    @property
    def _all_geometries(self):
        return self.geometry_layers.all_geometries

    @_all_geometries.setter
    def _all_geometries(self, value):
        self.geometry_layers.all_geometries = value

    @property
    def _geometry_layer_order(self):
        return self.geometry_layers.geometry_layer_order

    @_geometry_layer_order.setter
    def _geometry_layer_order(self, value):
        self.geometry_layers.geometry_layer_order = value

    @property
    def _geometry_layer_idx(self):
        return self.geometry_layers.geometry_layer_idx

    @_geometry_layer_idx.setter
    def _geometry_layer_idx(self, value):
        self.geometry_layers.geometry_layer_idx = value

    @property
    def _layer_positions(self):
        return self.geometry_layers.layer_positions

    @_layer_positions.setter
    def _layer_positions(self, value):
        self.geometry_layers.layer_positions = value

    @property
    def _current_geom_idx(self):
        return self.geometry_layers.current_geom_idx

    @_current_geom_idx.setter
    def _current_geom_idx(self, value):
        self.geometry_layers.current_geom_idx = value

    @property
    def _pc_color_modes(self):
        return self.geometry_layers.pc_color_modes

    @property
    def _pc_color_mode_idx(self):
        return self.geometry_layers.pc_color_mode_idx

    @_pc_color_mode_idx.setter
    def _pc_color_mode_idx(self, value):
        self.geometry_layers.pc_color_mode_idx = value

    @property
    def _chunk_manager(self):
        return self.geometry_layers.chunk_manager

    @_chunk_manager.setter
    def _chunk_manager(self, value):
        self.geometry_layers.chunk_manager = value

    @property
    def _baked_meshes(self):
        return self.geometry_layers.baked_meshes

    @_baked_meshes.setter
    def _baked_meshes(self, value):
        self.geometry_layers.baked_meshes = value

    @property
    def _geometry_colors_builder(self):
        return self.geometry_layers.geometry_colors_builder

    @_geometry_colors_builder.setter
    def _geometry_colors_builder(self, value):
        self.geometry_layers.geometry_colors_builder = value

    # ------------------------------------------------------------------
    # Thin wrappers — camera methods
    # ------------------------------------------------------------------

    def _get_front(self):
        """Get the forward direction vector."""
        return self.camera.get_front()

    def _get_right(self):
        """Get the right direction vector."""
        return self.camera.get_right()

    def _screen_to_ray(self, screen_x, screen_y):
        """Convert screen pixel coordinates to a world-space ray."""
        return self.camera.screen_to_ray(
            screen_x, screen_y,
            self.render_width, self.render_height,
            self.width, self.height,
        )

    def _get_look_at(self):
        """Get the current look-at point."""
        return self.camera.get_look_at()

    def _build_title(self):
        """Build a rich status title string for the viewer window."""
        H, W = self.terrain_shape
        parts = [self._title]

        # Resolution
        res = f"{W}\u00d7{H}"
        if self.subsample_factor > 1:
            res += f" ({self.subsample_factor}\u00d7 sub)"
        parts.append(res)

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

        # Hydro
        if self._hydro_enabled:
            parts.append('hydro')

        # Active observer drone mode
        active_obs = (self._observers.get(self._active_observer)
                      if self._active_observer else None)
        if active_obs is not None:
            if active_obs.drone_mode == 'fpv':
                parts.append(f'OBS{active_obs.slot} FPV')
            elif active_obs.drone_mode == '3rd':
                parts.append(f'OBS{active_obs.slot} 3RD')

        return '  \u2502  '.join(parts)

    @staticmethod
    def _elevation_to_minimap_rgba(terrain_small):
        """Convert a 2D elevation array to an RGBA minimap image.

        Parameters
        ----------
        terrain_small : ndarray, shape (H, W), float32
            Elevation data (may contain NaN for water/nodata).

        Returns
        -------
        rgba : ndarray, shape (H, W, 4), float32
            RGBA minimap image with hillshade and water coloring.
        water : ndarray, shape (H, W), bool
            Water mask (NaN pixels).
        """
        new_h, new_w = terrain_small.shape

        # Water mask: only NaN (not <= 0, which catches valid terrain)
        water = np.isnan(terrain_small)

        # Fill NaNs for gradient computation — use edge extrapolation
        # to avoid step-change gradient artifacts at the data boundary
        if water.any():
            terrain_small = terrain_small.copy()
            # Simple iterative nearest-neighbor fill: propagate valid
            # values into NaN regions to avoid gradient discontinuities.
            # Cap iterations — minimap is small (~200px), so 50 is plenty.
            filled = terrain_small
            for _ in range(min(50, max(new_h, new_w))):
                still_nan = np.isnan(filled)
                if not still_nan.any():
                    break
                # Shift in 4 directions and average available neighbors
                padded = np.pad(filled, 1, mode='edge')
                neighbors = np.stack([
                    padded[:-2, 1:-1],  # up
                    padded[2:, 1:-1],   # down
                    padded[1:-1, :-2],  # left
                    padded[1:-1, 2:],   # right
                ], axis=0)
                with np.errstate(all='ignore'):
                    fill_vals = np.nanmean(neighbors, axis=0)
                filled = np.where(still_nan & np.isfinite(fill_vals),
                                  fill_vals, filled)
            terrain_small = filled

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
        land = ~water
        emin = np.nanmin(terrain_small[land]) if land.any() else 0
        emax = np.nanmax(terrain_small[land]) if land.any() else 1
        erng = emax - emin if emax > emin else 1.0
        elev_norm = np.clip((terrain_small - emin) / erng, 0, 1)

        # Build RGBA image
        rgba = np.zeros((new_h, new_w, 4), dtype=np.float32)

        # Grayscale hillshade base for all land
        grey = shaded * 0.5 + elev_norm * 0.3 + 0.1
        grey = np.clip(grey, 0, 1)
        for c in range(3):
            rgba[:, :, c] = grey
        rgba[:, :, 3] = 1.0

        # Water (NaN): dark blue-black
        rgba[water, 0] = 0.08
        rgba[water, 1] = 0.10
        rgba[water, 2] = 0.18
        rgba[water, 3] = 0.7

        rgba[:, :, :3] = np.clip(rgba[:, :, :3], 0, 1)

        return rgba, water

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

        rgba, water = self._elevation_to_minimap_rgba(terrain_small)

        # Check for categorical overlay layer coloring
        _layer_data = None
        if (self._minimap_layer and self._minimap_colors
                and self._minimap_layer in self._base_overlay_layers):
            ld = self._base_overlay_layers[self._minimap_layer]
            if hasattr(ld, 'get'):
                ld = ld.get()
            ld = np.asarray(ld, dtype=np.float64)
            if longest > max_dim:
                _layer_data = ld[np.ix_(y_idx, x_idx)]
            else:
                _layer_data = ld.copy()

        if _layer_data is not None:
            # Recompute hillshade for categorical overlay blending
            ts_filled = terrain_small.copy()
            if water.any():
                med = np.nanmedian(ts_filled)
                ts_filled[water] = med if np.isfinite(med) else 0.0
            dy, dx = np.gradient(ts_filled)
            az_rad = np.radians(315)
            alt_rad = np.radians(45)
            slp = np.sqrt(dx**2 + dy**2)
            asp = np.arctan2(-dy, dx)
            shaded = np.clip(
                np.sin(alt_rad) * np.cos(np.arctan(slp)) +
                np.cos(alt_rad) * np.sin(np.arctan(slp)) *
                np.cos(az_rad - asp), 0, 1)
            # Overlay risk colours on matched pixels; unmatched stays grey
            for val, (r, g, b) in self._minimap_colors.items():
                mask = np.isclose(_layer_data, float(val), atol=0.1)
                for c, cv in enumerate((r, g, b)):
                    rgba[:, :, c] = np.where(
                        mask, cv * (shaded * 0.5 + 0.5), rgba[:, :, c])

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

        # Apply minimap style filter
        if self._minimap_style == 'cyberpunk':
            rgba = self._apply_cyberpunk_minimap(rgba, water)

        self._minimap_background = rgba
        self._minimap_scale_x = new_w / W
        self._minimap_scale_y = new_h / H

    def _compute_streaming_minimap(self, wx_min, wy_min, wx_max, wy_max):
        """Fetch elevation for a world extent and build a minimap image.

        Uses ``_tile_data_fn`` to fetch elevation covering the full
        minimap extent (initial terrain + streaming area), then runs
        the standard hillshade pipeline on it.

        Parameters
        ----------
        wx_min, wy_min, wx_max, wy_max : float
            World-space extent to cover in the minimap.
        """
        tile_data_fn = getattr(self, '_tile_data_fn', None)
        crs_tf = getattr(self, '_minimap_crs_transform', None)
        if tile_data_fn is None or crs_tf is None:
            return

        crs_x0, crs_y0, crs_dx, crs_dy = crs_tf
        psx, psy = self.pixel_spacing_x, self.pixel_spacing_y

        # Convert world coords to CRS coordinates
        # world coord: wx = col * psx + offset_x
        # col = (wx - offset_x) / psx
        # crs_x = crs_x0 + col * crs_dx
        lod_mgr = getattr(self.terrain, '_terrain_lod_manager', None)
        ox = lod_mgr._offset_x if lod_mgr is not None else 0.0
        oy = lod_mgr._offset_y if lod_mgr is not None else 0.0

        col0 = (wx_min - ox) / psx
        col1 = (wx_max - ox) / psx
        row0 = (wy_min - oy) / psy
        row1 = (wy_max - oy) / psy

        cx0 = crs_x0 + col0 * crs_dx
        cx1 = crs_x0 + col1 * crs_dx
        cy0 = crs_y0 + row0 * crs_dy
        cy1 = crs_y0 + row1 * crs_dy

        # tile_data_fn expects (x_min, y_min, x_max, y_max, target_samples)
        x_min, x_max = min(cx0, cx1), max(cx0, cx1)
        y_min, y_max = min(cy0, cy1), max(cy0, cy1)

        try:
            elev = tile_data_fn(x_min, y_min, x_max, y_max, 200)
        except Exception:
            return
        if elev is None:
            return

        elev = np.asarray(elev, dtype=np.float32)
        if elev.ndim != 2 or elev.size == 0:
            return

        rgba, water = self._elevation_to_minimap_rgba(elev)

        if self._minimap_style == 'cyberpunk':
            rgba = self._apply_cyberpunk_minimap(rgba, water)

        self._minimap_background = rgba
        self._minimap_bg_extent = (wx_min, wy_min, wx_max, wy_max)
        # Scale factors map the full bg image to the world extent
        self._minimap_scale_x = rgba.shape[1] / max(1, wx_max - wx_min)
        self._minimap_scale_y = rgba.shape[0] / max(1, wy_max - wy_min)
        self._minimap_last_stream_time = time.monotonic()

    def _apply_cyberpunk_minimap(self, rgba, water):
        """Apply a neon-edge cyberpunk filter to the minimap RGBA image.

        Detects edges via Sobel, colours them with a cyan/magenta neon
        palette, darkens the base, and adds faint scan-lines.
        """
        h, w = rgba.shape[:2]

        # Convert to luminance for edge detection
        lum = rgba[:, :, 0] * 0.299 + rgba[:, :, 1] * 0.587 + rgba[:, :, 2] * 0.114

        # Sobel edge detection
        # Horizontal kernel [-1 0 1; -2 0 2; -1 0 1]
        sx = np.zeros_like(lum)
        sy = np.zeros_like(lum)
        if h > 2 and w > 2:
            sx[1:-1, 1:-1] = (
                -lum[:-2, :-2] + lum[:-2, 2:]
                - 2 * lum[1:-1, :-2] + 2 * lum[1:-1, 2:]
                - lum[2:, :-2] + lum[2:, 2:]
            )
            sy[1:-1, 1:-1] = (
                -lum[:-2, :-2] - 2 * lum[:-2, 1:-1] - lum[:-2, 2:]
                + lum[2:, :-2] + 2 * lum[2:, 1:-1] + lum[2:, 2:]
            )

        edges = np.sqrt(sx ** 2 + sy ** 2)
        edges = np.clip(edges / (edges.max() + 1e-8), 0, 1)

        # Boost contrast — power curve to sharpen edges
        edges = edges ** 0.6

        # Neon colour: cyan for terrain edges, magenta for strong edges
        cyan = np.array([0.0, 1.0, 1.0])
        magenta = np.array([1.0, 0.0, 0.8])
        # Blend cyan→magenta based on edge intensity
        neon = np.zeros((h, w, 3), dtype=np.float32)
        for c in range(3):
            neon[:, :, c] = cyan[c] + (magenta[c] - cyan[c]) * edges

        # Dark base: heavily darken the original image
        dark = np.zeros_like(rgba)
        for c in range(3):
            dark[:, :, c] = rgba[:, :, c] * 0.12
        dark[:, :, 3] = rgba[:, :, 3]

        # Water gets a deep dark blue-purple
        dark[water, 0] = 0.03
        dark[water, 1] = 0.02
        dark[water, 2] = 0.08
        dark[water, 3] = 0.85

        # Composite neon edges onto dark base (additive blend)
        edge_alpha = edges * 0.9  # edge glow strength
        result = dark.copy()
        for c in range(3):
            result[:, :, c] = np.clip(
                dark[:, :, c] + neon[:, :, c] * edge_alpha, 0, 1)

        # Add faint simulated glow: dilate edges slightly via max-filter
        if h > 4 and w > 4:
            glow = edges.copy()
            # Simple 3x3 max pooling for glow spread
            padded = np.pad(glow, 1, mode='edge')
            glow = np.maximum.reduce([
                padded[:-2, :-2], padded[:-2, 1:-1], padded[:-2, 2:],
                padded[1:-1, :-2], padded[1:-1, 1:-1], padded[1:-1, 2:],
                padded[2:, :-2], padded[2:, 1:-1], padded[2:, 2:],
            ])
            glow_extra = np.clip(glow - edges, 0, 1) * 0.3
            for c in range(3):
                result[:, :, c] = np.clip(
                    result[:, :, c] + neon[:, :, c] * glow_extra, 0, 1)

        # Scan-lines: darken every other row slightly
        scanline = np.ones(h, dtype=np.float32)
        scanline[1::2] = 0.82
        for c in range(3):
            result[:, :, c] *= scanline[:, np.newaxis]

        result[:, :, :3] = np.clip(result[:, :, :3], 0, 1)
        return result

    def _enable_terrain_lod(self):
        """Set up per-tile LOD terrain rendering.

        Creates a :class:`TerrainLODManager` that renders each tile at
        a distance-appropriate resolution.  Called once during __init__.
        """
        from .viewer.terrain_lod import TerrainLODManager

        # Get full-res terrain as numpy
        base = self._base_raster
        terrain_data = base.data
        if hasattr(terrain_data, 'get'):
            terrain_np = terrain_data.get()
        else:
            terrain_np = np.asarray(terrain_data)

        # Fill NaN at raster edges (common from UTM reprojection) so both
        # the LOD tile builder and the render kernel see clean elevation.
        # Without this, NaN pixels render as blue ocean water.
        if np.any(np.isnan(terrain_np)):
            terrain_np = terrain_np.copy()
            for _ in range(20):
                still_nan = np.isnan(terrain_np)
                if not still_nan.any():
                    break
                padded = np.pad(terrain_np, 1, mode='edge')
                neighbors = np.stack([
                    padded[:-2, 1:-1], padded[2:, 1:-1],
                    padded[1:-1, :-2], padded[1:-1, 2:],
                ], axis=0)
                with np.errstate(all='ignore'):
                    fill_vals = np.nanmean(neighbors, axis=0)
                terrain_np = np.where(
                    still_nan & np.isfinite(fill_vals),
                    fill_vals, terrain_np)
            # Update raster so render kernel sees clean data too
            is_cupy = hasattr(base.data, 'get')
            if is_cupy:
                import cupy
                self.raster = self.raster.copy(
                    data=cupy.asarray(terrain_np))
            else:
                self.raster = self.raster.copy(data=terrain_np)

        # Remove the single terrain geometry
        if self.rtx.has_geometry('terrain'):
            self.rtx.remove_geometry('terrain')
        if self.rtx.has_geometry('terrain_skirt'):
            self.rtx.remove_geometry('terrain_skirt')

        # Choose tile size.  When a zarr chunk manager is active, align
        # to the zarr elevation chunk size so terrain tiles and mesh chunks
        # share the same spatial grid — one distance lookup drives both.
        H, W = terrain_np.shape
        if (self._chunk_manager is not None
                and self._chunk_manager._chunk_h == self._chunk_manager._chunk_w):
            tile_size = self._chunk_manager._chunk_h
            print(f"LOD tile size {tile_size} (aligned to zarr chunk grid)")
        else:
            tile_size = max(32, min(256, max(H, W) // 8))
            print(f"LOD tile size {tile_size}")

        mgr = TerrainLODManager(
            terrain_np,
            tile_size=tile_size,
            pixel_spacing_x=self._base_pixel_spacing_x,
            pixel_spacing_y=self._base_pixel_spacing_y,
            max_lod=3,
            base_subsample=self.subsample_factor,
        )
        # Carry forward any world offset from a previous terrain reload
        ox = self.terrain._world_offset_x
        oy = self.terrain._world_offset_y
        if ox != 0.0 or oy != 0.0:
            mgr.set_offset(ox, oy)
        # Enable tile streaming if a data callback was provided
        tile_data_fn = getattr(self, '_tile_data_fn', None)
        if tile_data_fn is not None:
            mgr.set_tile_data_fn(tile_data_fn)
            # Pass CRS coordinate transform so tile_data_fn receives
            # actual CRS coordinates (e.g. UTM) instead of viewer
            # world-space coords (pixel * abs(spacing) + offset).
            try:
                x = base.coords['x'].values
                y = base.coords['y'].values
                if len(x) >= 2 and len(y) >= 2:
                    crs_dx = float(x[1] - x[0])
                    crs_dy = float(y[1] - y[0])
                    mgr.set_crs_transform(float(x[0]), float(y[0]),
                                          crs_dx, crs_dy)
                    self._minimap_crs_transform = (
                        float(x[0]), float(y[0]), crs_dx, crs_dy)
            except (KeyError, AttributeError):
                pass
        self._terrain_lod_manager = mgr
        self.lod_enabled = True

        # Create per-tile overlay and texture managers for LOD-aware compositing
        from .viewer.overlay_tiles import OverlayTileManager, TextureTileManager
        self._overlay_tile_mgr = OverlayTileManager(tile_size)
        self._texture_tile_mgr = TextureTileManager(tile_size)
        # Register tile lifecycle callbacks so both managers stay
        # in sync with the LOD tile set automatically.
        otm = self._overlay_tile_mgr
        ttm = self._texture_tile_mgr

        # With LOD active, basemap goes through per-tile lazy fetch — stop
        # any monolithic XYZ tile fetch that was started before LOD enable.
        if self._tile_service is not None:
            self._tile_service._generation += 1  # cancel in-flight fetches

        # Lazy per-tile basemap fetching.  Each tile's CRS bounds are
        # converted to WGS84 and XYZ map tiles are fetched in background
        # threads, then stored in the TextureTileManager.
        _crs_origin = mgr._crs_origin      # (crs_x0, crs_y0) or None
        _crs_spacing = mgr._crs_spacing    # (crs_dx, crs_dy) or None
        _ts = tile_size
        _viewer = self
        from concurrent.futures import ThreadPoolExecutor as _TPE
        _basemap_executor = _TPE(max_workers=4)
        _basemap_pending = set()  # tiles currently being fetched

        def _on_tile_added(tr, tc, elev):
            # Don't call otm/ttm.invalidate() here — set_tile() already
            # marks dirty when actual data arrives.  Invalidating on
            # every terrain LOD change forces composite rebuild + GPU
            # upload every frame during camera movement.

            # Lazy-fetch basemap for this tile in background.
            # Use elev.shape to get actual tile dimensions (edge tiles
            # may be smaller than tile_size).
            if (_crs_origin is not None
                    and _crs_spacing is not None
                    and _viewer._tiles_enabled
                    and _viewer._tile_service is not None
                    and (tr, tc) not in _basemap_pending
                    and not ttm.has_tile(tr, tc)):
                th = elev.shape[0] if elev is not None else _ts
                tw = elev.shape[1] if elev is not None else _ts
                _basemap_pending.add((tr, tc))
                _basemap_executor.submit(
                    _fetch_tile_basemap, tr, tc, th, tw)

        def _fetch_tile_basemap(tr, tc, th, tw):
            """Fetch basemap RGB for a single LOD tile and store it."""
            try:
                crs_x0, crs_y0 = _crs_origin
                crs_dx, crs_dy = _crs_spacing
                c0 = tc * _ts
                r0 = tr * _ts
                # CRS bounds: pixel-center to pixel-center so linspace
                # produces exact pixel coordinates.  Using c0+tw would
                # overshoot by one pixel and stretch the basemap.
                cx0 = crs_x0 + c0 * crs_dx
                cy0 = crs_y0 + r0 * crs_dy
                cx1 = crs_x0 + (c0 + tw - 1) * crs_dx
                cy1 = crs_y0 + (r0 + th - 1) * crs_dy
                x_min, x_max = min(cx0, cx1), max(cx0, cx1)
                y_min, y_max = min(cy0, cy1), max(cy0, cy1)
                rgb = _viewer._tile_service.fetch_rgb_for_bounds(
                    x_min, y_min, x_max, y_max, th, tw)
                if rgb is not None and not np.all(rgb == 0):
                    ttm.set_tile(tr, tc, rgb)
            except Exception:
                pass
            finally:
                _basemap_pending.discard((tr, tc))

        def _on_tile_removed(tr, tc):
            # Only remove basemap texture — it's re-fetched lazily
            # via _on_tile_added when the tile returns.  Overlay data
            # is bulk-populated from populate_from_array() and won't
            # be re-created on re-add, so we must keep it.
            ttm.remove_tile(tr, tc)

        mgr.set_tile_callbacks(
            on_added=_on_tile_added,
            on_removed=_on_tile_removed,
        )
        # If an overlay already exists, slice it into per-tile chunks
        if self._active_overlay_data is not None:
            n_tr = (H + tile_size - 1) // tile_size
            n_tc = (W + tile_size - 1) // tile_size
            active_name = None
            if self._terrain_layer_idx < len(self._terrain_layer_order):
                active_name = self._terrain_layer_order[
                    self._terrain_layer_idx]
            for name, data in self._overlay_layers.items():
                if name == active_name:
                    self._overlay_tile_mgr.populate_from_array(
                        data, tile_size, n_tr, n_tc)
                    lut = self._overlay_color_luts.get(name)
                    if lut is not None:
                        self._overlay_tile_mgr.set_color_lut(lut)

        # Basemap tiles are fetched lazily via _on_tile_added callback —
        # no monolithic texture slicing needed.

        # Wire LOD manager into HydroManager for streaming support
        self.hydro_mgr.set_lod_manager(mgr)
        if tile_data_fn is not None:
            self.hydro_mgr.set_tile_data_fn(tile_data_fn)
        try:
            x = base.coords['x'].values
            y = base.coords['y'].values
            if len(x) >= 2 and len(y) >= 2:
                self.hydro_mgr.set_crs_transform(
                    float(x[0]), float(y[0]),
                    float(x[1] - x[0]), float(y[1] - y[0]))
        except (KeyError, AttributeError):
            pass

        # When streaming is active, use TIN meshes for ALL tiles (including
        # LOD 0) so initial-extent and streaming tiles render identically.
        # Without streaming, heightfield gives better quality for LOD 0.
        if not mgr._streaming:
            mgr.enable_heightfield_lod0()

        # Force initial tile build — no build limit so all in-bounds
        # tiles appear on the first frame (no progressive pop-in on
        # enable).  Streaming tiles build progressively after launch.
        # Use terrain center as fallback if camera position isn't set yet
        # (called from __init__ before run() sets the start position).
        cam_pos = self.position
        if cam_pos is None:
            H, W = self.terrain_shape
            cx = W * self.pixel_spacing_x * 0.5
            cy = H * self.pixel_spacing_y * 0.5
            cam_pos = np.array([cx, cy, 0.0])
        saved_limit = mgr.per_tick_build_limit
        saved_streaming = mgr._streaming
        mgr._streaming = False  # only in-bounds tiles on initial build
        mgr.per_tick_build_limit = 10000
        mgr.update(cam_pos, self.rtx,
                    ve=self.vertical_exaggeration, force=True,
                    camera_front=self._get_front(), fov=self.camera.fov)
        mgr.per_tick_build_limit = saved_limit
        mgr._streaming = saved_streaming
        # Force one more update so streaming tiles begin building
        if saved_streaming:
            mgr._last_update_pos = None
        # Enable threaded mesh building for subsequent ticks
        mgr.enable_threaded_building()
        # Batch same-LOD tiles into single GAS entries to reduce IAS count
        mgr.enable_batched_upload()
        # Only render if camera position has been initialised (run() sets it).
        # During __init__ the position is still None.
        if self.position is not None:
            self._update_frame()

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

        # Update LOD manager's base subsample and force tile rebuild
        if self._terrain_lod_manager is not None:
            self._terrain_lod_manager.set_base_subsample(factor)
            self._terrain_lod_manager.update(
                self.position, self.rtx,
                ve=self.vertical_exaggeration, force=True,
                camera_front=self._get_front(), fov=self.camera.fov)

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
        self._hydro_terrain_np = None
        self._d_base_frame = None     # invalidate GPU wind/hydro buffers
        self._d_wind_scratch = None
        self._d_cloud_fog_map = None  # invalidate cached cloud fog map
        H, W = sub.shape
        self.terrain_shape = (H, W)

        # 2. Update pixel spacing
        self.pixel_spacing_x = self._base_pixel_spacing_x * factor
        self.pixel_spacing_y = self._base_pixel_spacing_y * factor

        # 3. Get terrain_np for elevation stats
        ve = self.vertical_exaggeration
        terrain_data = sub.data
        if hasattr(terrain_data, 'get'):
            terrain_np = terrain_data.get()
        else:
            terrain_np = np.asarray(terrain_data)

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
                self._overlay_as_water = (
                    terrain_name.startswith('flood_')
                    or (terrain_name == 'stream_link'
                        and self._hydro_enabled))
                self._active_overlay_color_lut = self._overlay_color_luts.get(
                    terrain_name)

        # 6. Invalidate chunk manager scene state (re-snap Z at new resolution).
        #    Raw zarr data in _cache is resolution-independent (Phase 1 ensures
        #    Z re-snap always uses full-res terrain), so we keep it and only
        #    clear the baked/active/visible state so update() re-merges cheaply.
        if self._chunk_manager is not None:
            for gid in list(self._chunk_manager._active_gids):
                if hasattr(self, '_baked_meshes'):
                    self._baked_meshes.pop(gid, None)
                if self._accessor is not None:
                    self._accessor._baked_meshes.pop(gid, None)
                # Remove stale geometry from RTX scene
                if self.rtx is not None and self.rtx.has_geometry(gid):
                    self.rtx.remove_geometry(gid)
            self._chunk_manager._visible.clear()
            self._chunk_manager._active_gids.clear()
            # Force immediate re-merge from cache at new resolution
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
            from .viewer.terrain_lod import is_terrain_lod_gid
            for geom_id in self.rtx.list_geometries():
                if is_terrain_lod_gid(geom_id):
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
        self._minimap_bg_extent = None
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

        # Force re-upload of all LOD tiles with new VE
        if self._terrain_lod_manager is not None:
            self._terrain_lod_manager._tile_lods.clear()
            self._terrain_lod_manager.update(
                self.position, self.rtx, ve=ve, force=True,
                camera_front=self._get_front(), fov=self.camera.fov)

        terrain_data = self.raster.data
        if hasattr(terrain_data, 'get'):
            terrain_np = terrain_data.get()
        else:
            terrain_np = np.asarray(terrain_data)

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
            from .viewer.terrain_lod import is_terrain_lod_gid
            for geom_id in self.rtx.list_geometries():
                if is_terrain_lod_gid(geom_id):
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

        # Size minimap: match legend height if available, else ~20% of frame.
        # Use initial terrain aspect ratio (not background image shape)
        # so dimensions stay fixed even when streaming changes the bg.
        H_t, W_t = self.terrain_shape
        terrain_aspect = W_t / max(1, H_t)
        if self._legend_rgba is not None:
            target_h = self._legend_rgba.shape[0]
        else:
            target_w = max(40, int(fw * 0.2))
            target_h = max(20, int(target_w / terrain_aspect))
        target_w = max(20, int(target_h * terrain_aspect))
        target_w = min(target_w, fw)
        target_h = min(target_h, fh)

        # --- World extent for coordinate mapping ---
        # When LOD streaming is active, extend beyond initial terrain
        # to include camera position so the dot stays visible.
        H, W = self.terrain_shape
        psx, psy = self.pixel_spacing_x, self.pixel_spacing_y
        terrain_wx = W * psx
        terrain_wy = H * psy

        lod_mgr = getattr(self.terrain, '_terrain_lod_manager', None)
        streaming = (lod_mgr is not None
                     and getattr(lod_mgr, '_streaming', False))

        if streaming:
            cam_x, cam_y = self.position[0], self.position[1]
            # Desired extent: union of terrain bounds and camera + margin
            margin = max(terrain_wx, terrain_wy) * 0.5
            wx_min = min(0.0, cam_x - margin)
            wy_min = min(0.0, cam_y - margin)
            wx_max = max(terrain_wx, cam_x + margin)
            wy_max = max(terrain_wy, cam_y + margin)

            # Check if streaming minimap recompute is needed:
            # - No extent yet (first time)
            # - Camera outside inner 60% of current extent AND 2s throttle
            bg_ext = self._minimap_bg_extent
            now = time.monotonic()
            throttle_ok = (now - self._minimap_last_stream_time >= 2.0)
            need_recompute = (bg_ext is None)
            if not need_recompute and throttle_ok:
                # Recompute if camera is outside inner 60% of extent
                bw = bg_ext[2] - bg_ext[0]
                bh = bg_ext[3] - bg_ext[1]
                inner_margin_x = bw * 0.2
                inner_margin_y = bh * 0.2
                need_recompute = (
                    cam_x < bg_ext[0] + inner_margin_x
                    or cam_x > bg_ext[2] - inner_margin_x
                    or cam_y < bg_ext[1] + inner_margin_y
                    or cam_y > bg_ext[3] - inner_margin_y)

            if need_recompute:
                self._compute_streaming_minimap(
                    wx_min, wy_min, wx_max, wy_max)
                # Re-read background after streaming recompute
                mm_bg = self._minimap_background
                mm_h, mm_w = mm_bg.shape[:2]

            # Use streaming extent for coordinate mapping
            if self._minimap_bg_extent is not None:
                wx_min, wy_min, wx_max, wy_max = self._minimap_bg_extent
        else:
            wx_min, wy_min = 0.0, 0.0
            wx_max, wy_max = terrain_wx, terrain_wy

        wx_range = wx_max - wx_min
        wy_range = wy_max - wy_min

        # Resize background to target dimensions (fixed size, no aspect
        # ratio adjustment — keeps minimap dimensions stable)
        y_idx = np.linspace(0, mm_h - 1, target_h).astype(int)
        x_idx = np.linspace(0, mm_w - 1, target_w).astype(int)
        bg_resized = mm_bg[np.ix_(y_idx, x_idx)].copy()

        # Placement: flush bottom-right
        y0 = fh - target_h
        x0 = fw - target_w

        # Alpha-composite background onto frame
        alpha = bg_resized[:, :, 3:4]
        rgb = bg_resized[:, :, :3]
        region = img[y0:y0+target_h, x0:x0+target_w]
        region[:] = region * (1 - alpha) + rgb * alpha

        # Store minimap rect and world extent for click-to-teleport
        self._minimap_rect = (x0, y0, target_w, target_h)
        self._minimap_world_extent = (wx_min, wy_min, wx_max, wy_max)

        # --- Camera position in minimap-local coords ---
        lx = (self.position[0] - wx_min) / wx_range * target_w
        ly = (self.position[1] - wy_min) / wy_range * target_h

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
            mcol = (hit[0] - wx_min) / wx_range * target_w
            mrow = (hit[1] - wy_min) / wy_range * target_h
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
            obs_lx = (obs_x - wx_min) / wx_range * target_w
            obs_ly = (obs_y - wy_min) / wy_range * target_h
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
                    from .viewer.terrain_lod import is_terrain_lod_gid
                    self._all_geometries = self.rtx.list_geometries()
                    groups = set()
                    for g in self._all_geometries:
                        if is_terrain_lod_gid(g):
                            continue
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

    def _init_weather(self, weather_data):
        """Interpolate weather variables from lat/lon grid onto terrain pixels.

        Registers each variable (temperature, precipitation, cloud_cover,
        pressure) as an overlay layer accessible via G-key cycling.
        """
        from .tiles import _build_latlon_grids
        from scipy.interpolate import RegularGridInterpolator

        raster = self._base_raster
        H, W = raster.shape

        lats_grid, lons_grid = _build_latlon_grids(raster)
        points = np.stack([lats_grid.ravel(), lons_grid.ravel()], axis=-1)

        w_lats = weather_data['lats']   # (ny,)
        w_lons = weather_data['lons']   # (nx,)

        variables = ['temperature', 'precipitation', 'cloud_cover', 'pressure']
        units = {'temperature': '°C', 'precipitation': 'mm',
                 'cloud_cover': '%', 'pressure': 'hPa'}
        added = []

        for var in variables:
            if var not in weather_data:
                continue
            grid = weather_data[var]  # (ny, nx)
            interp = RegularGridInterpolator(
                (w_lats, w_lons), grid,
                method='linear', bounds_error=False, fill_value=np.nan,
            )
            layer = interp(points).reshape(H, W).astype(np.float32)
            _add_overlay(self, var, layer)
            mean_val = np.nanmean(layer)
            added.append(f"{var} {mean_val:.1f}{units.get(var, '')}")

        if added:
            print(f"  Weather: {len(added)} layers added ({', '.join(added)})")

        # Auto-initialize cloud + rain particles from weather data
        self._init_clouds(weather_data)

    # ------------------------------------------------------------------
    # Cloud + rain particle system
    # ------------------------------------------------------------------

    def _init_clouds(self, weather_data):
        """Initialize cloud and rain particles from weather data.

        Spawns cloud puffs proportional to cloud_cover and rain streaks
        proportional to precipitation.  Both drift with the wind field
        if available.
        """
        if weather_data is None:
            return

        cloud_cover = weather_data.get('cloud_cover')
        precipitation = weather_data.get('precipitation')
        if cloud_cover is None and precipitation is None:
            return

        from .tiles import _build_latlon_grids
        from scipy.interpolate import RegularGridInterpolator

        raster = self._base_raster
        H, W = raster.shape
        w_lats = weather_data['lats']
        w_lons = weather_data['lons']

        lats_grid, lons_grid = _build_latlon_grids(raster)
        points = np.stack([lats_grid.ravel(), lons_grid.ravel()], axis=-1)

        # Interpolate cloud_cover (0-100%) to terrain grid
        if cloud_cover is not None:
            interp = RegularGridInterpolator(
                (w_lats, w_lons), cloud_cover.astype(np.float32),
                method='linear', bounds_error=False, fill_value=0.0,
            )
            cc = interp(points).reshape(H, W).astype(np.float32)
            self._cloud_cover_grid = np.clip(cc / 100.0, 0, 1)
        else:
            self._cloud_cover_grid = np.zeros((H, W), dtype=np.float32)

        # Interpolate precipitation (mm) to terrain grid
        if precipitation is not None:
            interp_p = RegularGridInterpolator(
                (w_lats, w_lons), precipitation.astype(np.float32),
                method='linear', bounds_error=False, fill_value=0.0,
            )
            self._rain_grid = interp_p(points).reshape(H, W).astype(np.float32)
        else:
            self._rain_grid = np.zeros((H, W), dtype=np.float32)

        # Cloud altitude: above terrain max
        terrain_data = raster.data
        if hasattr(terrain_data, 'get'):
            terrain_np = terrain_data.get()
        else:
            terrain_np = np.asarray(terrain_data)
        psx = self._base_pixel_spacing_x
        psy = self._base_pixel_spacing_y
        z_max = float(np.nanmax(terrain_np))
        z_range = z_max - float(np.nanmin(terrain_np))
        diag = np.sqrt((W * psx) ** 2 + (H * psy) ** 2)
        self._cloud_altitude = z_max + max(z_range * 0.5, diag * 0.04)
        self._cloud_min_depth = diag * 0.01
        self._cloud_thickness = max(z_range * 0.3, diag * 0.02)
        self._volumetric_clouds_enabled = True

        # Spawn cloud particles weighted by cloud_cover density
        N = self._cloud_n_particles
        cc_flat = self._cloud_cover_grid.ravel()
        cc_sum = cc_flat.sum()
        if cc_sum > 0:
            prob = cc_flat / cc_sum
        else:
            prob = np.ones_like(cc_flat) / cc_flat.size

        chosen = np.random.choice(cc_flat.size, size=N, p=prob)
        rows = (chosen // W).astype(np.float32)
        cols = (chosen % W).astype(np.float32)
        # Jitter within cell
        rows += np.random.uniform(-0.5, 0.5, N).astype(np.float32)
        cols += np.random.uniform(-0.5, 0.5, N).astype(np.float32)
        self._cloud_particles = np.column_stack([rows, cols]).astype(np.float32)

        # Per-particle size (world-space radius)
        base_size = max(psx, psy) * 8
        self._cloud_sizes = (
            np.random.uniform(0.6, 1.8, N).astype(np.float32) * base_size
        )

        # Per-particle alpha from local cloud_cover
        r_idx = np.clip(rows.astype(int), 0, H - 1)
        c_idx = np.clip(cols.astype(int), 0, W - 1)
        local_cc = self._cloud_cover_grid[r_idx, c_idx]
        self._cloud_alphas = (local_cc * 0.35).astype(np.float32)

        # Ages / lifetimes
        self._cloud_max_age = 300
        self._cloud_lifetimes = np.random.randint(
            self._cloud_max_age // 2, self._cloud_max_age, N)
        self._cloud_ages = np.random.randint(0, self._cloud_max_age, N)

        # Rain particles — fewer, driven by precipitation
        rain_N = 8000
        rain_flat = self._rain_grid.ravel()
        rain_sum = rain_flat.sum()
        if rain_sum > 0:
            rain_prob = rain_flat / rain_sum
            chosen_r = np.random.choice(rain_flat.size, size=rain_N, p=rain_prob)
        else:
            chosen_r = np.random.randint(0, rain_flat.size, rain_N)
        r_rows = (chosen_r // W).astype(np.float32)
        r_cols = (chosen_r % W).astype(np.float32)
        r_rows += np.random.uniform(-0.5, 0.5, rain_N).astype(np.float32)
        r_cols += np.random.uniform(-0.5, 0.5, rain_N).astype(np.float32)
        self._rain_particles = np.column_stack([r_rows, r_cols]).astype(np.float32)
        # Rain z: random altitude between terrain and cloud layer
        self._rain_z_frac = np.random.uniform(0.0, 1.0, rain_N).astype(np.float32)
        self._rain_ages = np.random.randint(0, 40, rain_N)
        self._rain_lifetimes = np.random.randint(20, 40, rain_N)
        self._rain_max_age = 40

        mean_cc = float(np.mean(self._cloud_cover_grid) * 100)
        mean_precip = float(np.mean(self._rain_grid))
        print(f"  Clouds: {N} particles (mean cover {mean_cc:.0f}%)"
              f"  Rain: {rain_N} particles (mean precip {mean_precip:.1f}mm)")

    def _toggle_clouds(self):
        """Toggle cloud fog + rain on/off."""
        if self._cloud_cover_grid is None:
            print("No weather data loaded. Pass weather_data to explore().")
            return
        self._clouds_enabled = not self._clouds_enabled
        if self._clouds_enabled and self._cloud_cover_grid is not None:
            cc = self._cloud_cover_grid
            print(f"Clouds: ON  (cover min={cc.min():.2f} max={cc.max():.2f} mean={cc.mean():.2f})")
        else:
            print(f"Clouds: OFF")
        self._render_needed = True

    def _action_toggle_clouds(self):
        self._toggle_clouds()

    def _update_rain_particles(self):
        """Animate rain streaks: fall downward and respawn."""
        if not hasattr(self, '_rain_particles') or self._rain_particles is None:
            return

        pts = self._rain_particles
        H, W = self._cloud_cover_grid.shape

        # Rain falls: decrease z_frac (toward terrain)
        self._rain_z_frac -= 0.06
        self._rain_ages += 1

        # Slight wind drift
        if self._wind_u_px is not None:
            r0 = np.clip(pts[:, 0].astype(int), 0, H - 1)
            c0 = np.clip(pts[:, 1].astype(int), 0, W - 1)
            s = getattr(self, '_dt_scale', 1.0)
            pts[:, 1] += self._wind_u_px[r0, c0] * 0.15 * s
            pts[:, 0] += self._wind_v_px[r0, c0] * 0.15 * s

        # Respawn when hitting ground or aged out
        respawn = (
            (self._rain_z_frac <= 0) |
            (self._rain_ages >= self._rain_lifetimes) |
            (pts[:, 0] < 0) | (pts[:, 0] >= H) |
            (pts[:, 1] < 0) | (pts[:, 1] >= W)
        )
        n_respawn = int(respawn.sum())
        if n_respawn > 0:
            rain_flat = self._rain_grid.ravel()
            rain_sum = rain_flat.sum()
            if rain_sum > 0:
                prob = rain_flat / rain_sum
            else:
                prob = np.ones_like(rain_flat) / rain_flat.size
            chosen = np.random.choice(rain_flat.size, size=n_respawn, p=prob)
            pts[respawn, 0] = (chosen // W).astype(np.float32) + np.random.uniform(-0.5, 0.5, n_respawn)
            pts[respawn, 1] = (chosen % W).astype(np.float32) + np.random.uniform(-0.5, 0.5, n_respawn)
            self._rain_z_frac[respawn] = np.random.uniform(0.7, 1.0, n_respawn).astype(np.float32)
            self._rain_ages[respawn] = 0
            self._rain_lifetimes[respawn] = np.random.randint(20, 40, n_respawn)

    def _draw_rain_on_frame(self, img, forward, right, cam_up,
                            fov_scale, aspect, min_depth):
        """Render rain streaks on the frame."""
        sh, sw = img.shape[:2]
        psx = self._base_pixel_spacing_x
        psy = self._base_pixel_spacing_y
        ve = self.vertical_exaggeration
        cam_pos = self.position

        # Cached terrain for z lookup
        if self._cloud_terrain_np is None:
            terrain_data = self.raster.data
            if hasattr(terrain_data, 'get'):
                self._cloud_terrain_np = terrain_data.get()
            else:
                self._cloud_terrain_np = np.asarray(terrain_data)
        terrain_np = self._cloud_terrain_np
        tH, tW = terrain_np.shape
        f = self.subsample_factor
        cloud_z = self._cloud_altitude * ve

        pts = self._rain_particles
        N = pts.shape[0]

        # Rain z: interpolate between terrain and cloud altitude
        sr = np.clip((pts[:, 0] / f).astype(int), 0, tH - 1)
        sc = np.clip((pts[:, 1] / f).astype(int), 0, tW - 1)
        terrain_z = np.nan_to_num(terrain_np[sr, sc], nan=0.0) * ve
        rain_z = terrain_z + self._rain_z_frac * (cloud_z - terrain_z)

        wx = pts[:, 1] * psx
        wy = pts[:, 0] * psy

        dx = wx - cam_pos[0]
        dy = wy - cam_pos[1]
        dz = rain_z - cam_pos[2]

        depth = dx * forward[0] + dy * forward[1] + dz * forward[2]
        valid = depth > min_depth

        inv_depth = np.where(valid, 1.0 / (depth + 1e-10), 0.0)
        u_cam = dx * right[0] + dy * right[1] + dz * right[2]
        v_cam = dx * cam_up[0] + dy * cam_up[1] + dz * cam_up[2]
        u_ndc = u_cam * inv_depth / (fov_scale * aspect)
        v_ndc = v_cam * inv_depth / fov_scale

        sx = ((u_ndc + 1.0) * 0.5 * sw).astype(np.int32)
        sy = ((1.0 - v_ndc) * 0.5 * sh).astype(np.int32)

        # Rain streak length (vertical on screen, 3-8 pixels)
        streak_len = np.clip((5.0 * inv_depth * sh * fov_scale * 0.003), 2, 8).astype(np.int32)

        # Alpha based on precipitation intensity
        rain_r = np.clip((pts[:, 0] / f).astype(int), 0, tH - 1)
        rain_c = np.clip((pts[:, 1] / f).astype(int), 0, tW - 1)
        local_precip = self._rain_grid[
            np.clip(pts[:, 0].astype(int), 0, self._rain_grid.shape[0] - 1),
            np.clip(pts[:, 1].astype(int), 0, self._rain_grid.shape[1] - 1),
        ]
        base_alpha = np.clip(local_precip / 5.0, 0.05, 0.5).astype(np.float32)

        # Fade
        ages = self._rain_ages.astype(np.float32)
        lifetimes = self._rain_lifetimes.astype(np.float32)
        fade = np.clip(ages / 3.0, 0, 1) * np.clip((lifetimes - ages) / 5.0, 0, 1)
        alpha = base_alpha * fade * 0.15

        on_screen = (
            valid &
            (sx >= 0) & (sx < sw) &
            (sy >= 0) & (sy < sh + 8) &
            (alpha > 0.002)
        )
        if not on_screen.any():
            return

        sx_v = sx[on_screen]
        sy_v = sy[on_screen]
        al_v = alpha[on_screen]
        sl_v = streak_len[on_screen]

        # Draw vertical streaks (light blue-gray)
        color = np.array([0.7, 0.75, 0.85], dtype=np.float32)
        max_sl = int(sl_v.max()) if sl_v.size > 0 else 3
        for dy_off in range(max_sl):
            py = sy_v + dy_off
            streak_ok = (dy_off < sl_v) & (py >= 0) & (py < sh)
            if not streak_ok.any():
                continue
            t = dy_off / (sl_v[streak_ok].astype(np.float32))
            streak_alpha = al_v[streak_ok] * (1.0 - t * 0.6)
            for c in range(3):
                np.add.at(img[:, :, c], (py[streak_ok], sx_v[streak_ok]),
                         streak_alpha * color[c])

        np.clip(img, 0, 1, out=img)

    def _splat_rain_gpu(self, d_frame):
        """Project and splat rain particles as vertical streaks on GPU."""
        if not hasattr(self, '_rain_particles') or self._rain_particles is None:
            return
        rain_grid = getattr(self, '_rain_grid', None)
        if rain_grid is None or rain_grid.sum() < 0.01:
            return

        from .analysis.render import _compute_camera_basis

        sh, sw = d_frame.shape[:2]
        psx = float(self._base_pixel_spacing_x)
        psy = float(self._base_pixel_spacing_y)
        ve = float(self.vertical_exaggeration)
        cloud_z = float(self._cloud_altitude * ve)
        min_depth = float(self._cloud_min_depth)

        cam_pos = self.position
        look_at = self._get_look_at()
        forward, right, cam_up = _compute_camera_basis(
            tuple(cam_pos), tuple(look_at), (0, 0, 1),
        )
        fov_scale = float(math.tan(math.radians(self.fov) / 2.0))
        aspect = float(sw / sh)

        rain_pts = self._rain_particles
        rain_N = rain_pts.shape[0]
        f = float(self.subsample_factor)

        # GPU terrain for z lookup
        terrain_data = self.raster.data
        if not isinstance(terrain_data, cp.ndarray):
            terrain_data = cp.asarray(terrain_data)

        # Pre-compute rain alpha + streak length on CPU
        local_precip = self._rain_grid[
            np.clip(rain_pts[:, 0].astype(int), 0, self._rain_grid.shape[0] - 1),
            np.clip(rain_pts[:, 1].astype(int), 0, self._rain_grid.shape[1] - 1),
        ]
        base_alpha = np.clip(local_precip / 5.0, 0.05, 0.5).astype(np.float32)
        rain_ages = self._rain_ages.astype(np.float32)
        rain_lifetimes = self._rain_lifetimes.astype(np.float32)
        fade = (np.clip(rain_ages / 3.0, 0, 1)
                * np.clip((rain_lifetimes - rain_ages) / 5.0, 0, 1))
        rain_alpha = (base_alpha * fade * 0.15).astype(np.float32)

        # Streak length: compute from depth
        r_wx = rain_pts[:, 1] * psx
        r_wy = rain_pts[:, 0] * psy
        r_depth = ((r_wx - cam_pos[0]) * forward[0]
                   + (r_wy - cam_pos[1]) * forward[1]
                   + (cloud_z * 0.5 - cam_pos[2]) * forward[2])
        r_inv = np.where(r_depth > min_depth, 1.0 / (r_depth + 1e-10), 0.0)
        streak = np.clip((5.0 * r_inv * sh * fov_scale * 0.003), 2, 8).astype(np.int32)

        # Upload
        _dr = getattr(self, '_d_rain_pts', None)
        if _dr is None or _dr.shape[0] != rain_N:
            self._d_rain_pts = cp.empty((rain_N, 2), dtype=cp.float32)
            self._d_rain_zfrac = cp.empty(rain_N, dtype=cp.float32)
            self._d_rain_alpha = cp.empty(rain_N, dtype=cp.float32)
            self._d_rain_streak = cp.empty(rain_N, dtype=cp.int32)
        self._d_rain_pts.set(rain_pts)
        self._d_rain_zfrac.set(self._rain_z_frac)
        self._d_rain_alpha.set(rain_alpha)
        self._d_rain_streak.set(streak)

        tpb = 256
        bpg_r = (rain_N + tpb - 1) // tpb
        _rain_splat_kernel[bpg_r, tpb](
            self._d_rain_pts,
            self._d_rain_zfrac,
            self._d_rain_alpha,
            self._d_rain_streak,
            terrain_data,
            d_frame,
            float(cam_pos[0]), float(cam_pos[1]), float(cam_pos[2]),
            float(forward[0]), float(forward[1]), float(forward[2]),
            float(right[0]), float(right[1]), float(right[2]),
            float(cam_up[0]), float(cam_up[1]), float(cam_up[2]),
            fov_scale, aspect,
            psx, psy, ve, f, cloud_z, min_depth,
            0.7, 0.75, 0.85,
        )

        cp.clip(d_frame, 0, 1, out=d_frame)

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

    def _splat_wind_gpu(self, d_frame):
        """Project and splat wind particles on GPU via Numba CUDA kernel.

        Parameters
        ----------
        d_frame : cupy.ndarray, shape (H, W, 3)
            GPU frame buffer (float32 0-1). Modified in-place via atomic add.
        """
        if self._wind_particles is None or self._wind_trails is None:
            return

        from .analysis.render import _compute_camera_basis

        sh, sw = d_frame.shape[:2]
        N = self._wind_particles.shape[0]
        trail_len = self._wind_trail_len

        # Camera basis
        cam_pos = self.position
        look_at = self._get_look_at()
        forward, right, cam_up = _compute_camera_basis(
            tuple(cam_pos), tuple(look_at), (0, 0, 1),
        )
        fov_scale = math.tan(math.radians(self.fov) / 2.0)
        aspect_ratio = sw / sh

        # Pre-compute alpha on CPU (vectorized — same logic as _draw_wind_on_frame)
        ages = self._wind_ages
        lifetimes = self._wind_lifetimes
        trail_idx = np.tile(np.arange(trail_len, dtype=np.float32), N)
        ages_rep = np.repeat(ages, trail_len)
        lifetimes_rep = np.repeat(lifetimes, trail_len)
        age_ok = ages_rep > trail_idx

        mva = self._wind_min_visible_age
        fade_in = np.clip((ages_rep - mva) / 10.0, 0, 1)
        fade_out = np.clip((lifetimes_rep - ages_rep) / 20.0, 0, 1)
        trail_fade = 1.0 - (trail_idx / trail_len)
        alpha = self._wind_alpha * fade_in * fade_out * trail_fade
        alpha[~age_ok] = 0.0

        # Flatten trails: (N, T, 2) → (N*T, 2)
        all_pts = self._wind_trails.reshape(-1, 2).astype(np.float32)
        alpha = alpha.astype(np.float32)
        total = N * trail_len

        # Upload to reusable GPU buffers
        if self._d_wind_trails is None or self._d_wind_trails.shape[0] != total:
            self._d_wind_trails = cp.empty((total, 2), dtype=cp.float32)
            self._d_wind_alpha = cp.empty(total, dtype=cp.float32)
        self._d_wind_trails.set(all_pts)
        self._d_wind_alpha.set(alpha)

        # GPU terrain — use raster.data directly
        terrain_data = self.raster.data
        if not isinstance(terrain_data, cp.ndarray):
            terrain_data = cp.asarray(terrain_data)

        # Kernel launch
        threadsperblock = 256
        blockspergrid = (total + threadsperblock - 1) // threadsperblock

        f = float(self.subsample_factor)
        psx = float(self._base_pixel_spacing_x)
        psy = float(self._base_pixel_spacing_y)
        ve = float(self.vertical_exaggeration)
        min_depth = float(self._wind_min_depth)
        r = int(self._wind_dot_radius)

        _wind_splat_kernel[blockspergrid, threadsperblock](
            self._d_wind_trails,
            self._d_wind_alpha,
            terrain_data,
            d_frame,
            # Camera position
            float(cam_pos[0]), float(cam_pos[1]), float(cam_pos[2]),
            # Forward
            float(forward[0]), float(forward[1]), float(forward[2]),
            # Right
            float(right[0]), float(right[1]), float(right[2]),
            # Up
            float(cam_up[0]), float(cam_up[1]), float(cam_up[2]),
            # Projection
            float(fov_scale), float(aspect_ratio),
            # Terrain/world
            psx, psy, ve, f, min_depth,
            # Splat
            r,
            # Color (teal)
            0.3, 0.9, 0.8,
        )

        # Clamp output
        cp.clip(d_frame, 0, 1, out=d_frame)

    # ------------------------------------------------------------------
    # Hydrological flow particles
    # ------------------------------------------------------------------

    def _toggle_hydro(self):
        """Toggle hydro flow particles + stream_link water overlay together."""
        if self._hydro_data is None:
            # Lazy mode: compute hydro from terrain on first enable
            if self._hydro_lazy:
                print("Computing hydro from terrain (first enable)...")
                if not self._compute_hydro_from_terrain():
                    return
            else:
                print("No hydro data. Use v.add_hydro(flow_accum).")
                return
        self._hydro_enabled = not self._hydro_enabled

        if self._hydro_enabled:
            # Save current overlay state for restoration on OFF
            self._hydro_prev_layer_idx = self._terrain_layer_idx
            # Switch to stream_link overlay with water reflection shader
            if 'stream_link' in self._overlay_layers:
                idx = self._terrain_layer_order.index('stream_link')
                self._terrain_layer_idx = idx
                self._active_overlay_data = self._overlay_layers['stream_link']
                self._overlay_as_water = True
                self._active_overlay_color_lut = self._overlay_color_luts.get(
                    'stream_link')
                print("Hydro flow: ON  (stream overlay + water shader)")
            else:
                print("Hydro flow: ON")
        else:
            # Restore previous overlay state
            prev = getattr(self, '_hydro_prev_layer_idx', 0)
            if prev >= len(self._terrain_layer_order):
                prev = 0
            self._terrain_layer_idx = prev
            layer = self._terrain_layer_order[prev]
            if layer == 'elevation':
                self._active_overlay_data = None
                self._overlay_as_water = False
                self._active_overlay_color_lut = None
            else:
                self._active_overlay_data = self._overlay_layers[layer]
                self._overlay_as_water = layer.startswith('flood_')
                self._active_overlay_color_lut = (
                    self._overlay_color_luts.get(layer))
            print("Hydro flow: OFF")

        self._update_frame()

    def _action_toggle_hydro(self):
        self._toggle_hydro()

    # Delegate palette/helpers to hydro_manager module
    from .viewer.hydro_manager import (
        STREAM_ORDER_PALETTE as _STREAM_ORDER_PALETTE,
        build_stream_palette_lut as _build_stream_palette_lut_fn,
        color_from_order as _hydro_color_from_order_fn,
        radius_from_order as _hydro_radius_from_order_fn,
    )

    _build_stream_palette_lut = staticmethod(_build_stream_palette_lut_fn)
    _hydro_color_from_order = staticmethod(_hydro_color_from_order_fn)
    _hydro_radius_from_order = staticmethod(_hydro_radius_from_order_fn)

    def _init_hydro(self, flow_accum, **kwargs):
        """Initialize hydro flow particles using MFD flow direction.

        Delegates to ``self.hydro_mgr.init_from_flow()``.

        Parameters
        ----------
        flow_accum : array-like, shape (H, W)
            Flow accumulation grid (cell counts or area).
        **kwargs
            Optional overrides: n_particles, max_age, trail_len, speed,
            accum_threshold, color, alpha, dot_radius, min_visible_age,
            stream_order (array), flow_dir_mfd (xrspatial MFD fractions,
            shape (8,H,W)), elevation (conditioned DEM for manual MFD).
        """
        terrain_data = self.raster.data
        self.hydro_mgr.init_from_flow(
            flow_accum, terrain_data,
            self._base_pixel_spacing_x, self._base_pixel_spacing_y,
            **kwargs)

    def _compute_hydro_from_terrain(self):
        """Compute hydrological flow from current terrain on GPU.

        Delegates to ``self.hydro_mgr.compute_from_terrain()``.
        Returns True on success, False on failure.
        """
        self.hydro_mgr.set_terrain_ref(
            None, self._base_pixel_spacing_x, self._base_pixel_spacing_y)
        try:
            result = self.hydro_mgr.compute_from_terrain(self.raster)
        except Exception as exc:
            print(f"Hydro computation failed: {exc}")
            return False
        if result is None:
            return False

        # Register stream_link overlay with palette coloring
        if 'stream_link_overlay' in result:
            overlay = result['stream_link_overlay']
            _add_overlay(self, 'stream_link', overlay,
                         color_lut=result['palette_lut'])
            # Populate per-tile overlay manager for LOD rendering
            if self._overlay_tile_mgr is not None:
                mgr = self._terrain_lod_manager
                ts = mgr._tile_size if mgr is not None else 128
                H, W = overlay.shape
                n_tr = (H + ts - 1) // ts
                n_tc = (W + ts - 1) // ts
                self._overlay_tile_mgr.populate_from_array(
                    overlay, ts, n_tr, n_tc)
                self._overlay_tile_mgr.set_color_lut(
                    result['palette_lut'])
                n_tiles = len(self._overlay_tile_mgr._tile_overlays)
                print(f"  Overlay tiles: {n_tiles}/{n_tr*n_tc} "
                      f"(ts={ts}, grid={n_tr}x{n_tc}, "
                      f"overlay={H}x{W})")
        else:
            print("  Warning: no stream_link_overlay in hydro result")
        return True

    def _transfer_streaming_overlay(self):
        """Transfer pending streaming stream overlay to the overlay tile mgr.

        Called after hydro_mgr.check_streaming_result().  Only sets tiles
        that are outside the initial terrain grid (streaming tiles) to
        avoid overwriting the more accurate xrspatial-computed overlay.
        """
        otm = self._overlay_tile_mgr
        if otm is None:
            return
        # Only transfer when overlay is the active layer
        if self._active_overlay_data is None:
            return
        overlay, win_r0, win_c0 = self.hydro_mgr.pop_streaming_overlay()
        if overlay is None:
            return

        mgr = self._terrain_lod_manager
        if mgr is None:
            return

        ts = mgr._tile_size
        n_tr_initial = mgr._n_tile_rows
        n_tc_initial = mgr._n_tile_cols
        ov_h, ov_w = overlay.shape

        # Slice overlay into per-tile chunks, only for streaming tiles
        count = 0
        # Tile range covered by this window
        tr_start = win_r0 // ts
        tc_start = win_c0 // ts
        tr_end = (win_r0 + ov_h + ts - 1) // ts
        tc_end = (win_c0 + ov_w + ts - 1) // ts

        for tr in range(tr_start, tr_end):
            for tc in range(tc_start, tc_end):
                # Skip initial-grid tiles — they have accurate overlay
                if 0 <= tr < n_tr_initial and 0 <= tc < n_tc_initial:
                    continue
                # Extract tile region from overlay
                r0 = tr * ts - win_r0
                c0 = tc * ts - win_c0
                r1 = min(r0 + ts, ov_h)
                c1 = min(c0 + ts, ov_w)
                if r0 < 0 or c0 < 0 or r1 <= r0 or c1 <= c0:
                    continue
                tile_data = overlay[r0:r1, c0:c1]
                if np.all(np.isnan(tile_data)):
                    continue
                otm.set_tile(tr, tc, tile_data.copy())
                count += 1

        if count > 0:
            print(f"  Streaming overlay: {count} tiles added")

    def _update_hydro_particles(self):
        """Advect hydro particles one tick on GPU.  Delegates to HydroManager."""
        dt_scale = float(getattr(self, '_dt_scale', 1.0))
        self.hydro_mgr.update_particles(dt_scale=dt_scale)

    def _draw_hydro_on_frame(self, img):
        """Project hydro particles to screen space and draw on rendered frame.

        CPU fallback with per-particle accumulation-scaled color and radius.

        Parameters
        ----------
        img : ndarray, shape (H_screen, W_screen, 3)
            Rendered frame (float32 0-1). Modified in-place.
        """
        if self._hydro_particles is None:
            return

        from .analysis.render import _compute_camera_basis

        sh, sw = img.shape[:2]
        N = self._hydro_particles.shape[0]
        trail_len = self._hydro_trail_len

        cam_pos = self.position
        look_at = self._get_look_at()
        forward, right, cam_up = _compute_camera_basis(
            tuple(cam_pos), tuple(look_at), (0, 0, 1),
        )
        fov_scale = math.tan(math.radians(self.fov) / 2.0)
        aspect_ratio = sw / sh

        if self._hydro_terrain_np is None:
            terrain_data = self.raster.data
            if hasattr(terrain_data, 'get'):
                self._hydro_terrain_np = terrain_data.get()
            else:
                self._hydro_terrain_np = np.asarray(terrain_data)
        terrain_np = self._hydro_terrain_np
        tH, tW = terrain_np.shape

        f = self.subsample_factor
        psx = self._base_pixel_spacing_x
        psy = self._base_pixel_spacing_y
        ve = self.vertical_exaggeration
        min_depth = self._hydro_min_depth

        all_pts = self._hydro_trails.reshape(-1, 2)
        rows_all = all_pts[:, 0]
        cols_all = all_pts[:, 1]

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

        # Depth test against ray-traced terrain
        depth_t_np = None
        if hasattr(self, '_d_depth_t') and self._d_depth_t is not None and self._d_depth_t.shape[0] > 0:
            depth_t_np = cp.asnumpy(self._d_depth_t) if hasattr(self._d_depth_t, 'get') else np.asarray(self._d_depth_t)
        if depth_t_np is not None:
            sx_c = np.clip(sx_all, 0, sw - 1)
            sy_c = np.clip(sy_all, 0, sh - 1)
            t_vals = depth_t_np[sy_c, sx_c]
            u_px = (2.0 * sx_c / sw - 1.0) * fov_scale * aspect_ratio
            v_px = (1.0 - 2.0 * sy_c / sh) * fov_scale
            inv_cos = np.sqrt(1.0 + u_px**2 + v_px**2)
            terrain_fwd = t_vals / inv_cos
            occluded = (t_vals > 0) & (t_vals < 1e20) & (depth > terrain_fwd)
            on_screen = on_screen & ~occluded

        ages = self._hydro_ages
        lifetimes = self._hydro_lifetimes

        trail_idx = np.tile(np.arange(trail_len, dtype=np.float32), N)
        ages_rep = np.repeat(ages, trail_len)
        lifetimes_rep = np.repeat(lifetimes, trail_len)
        age_ok = ages_rep > trail_idx

        mva = self._hydro_min_visible_age
        fade_in = np.clip((ages_rep - mva) / 10.0, 0, 1)
        fade_out = np.clip((lifetimes_rep - ages_rep) / 20.0, 0, 1)
        # Quadratic trail decay — comet-tail effect (matches GPU kernel)
        trail_fade = (1.0 - trail_idx / trail_len) ** 2

        ref_d = self._hydro_ref_depth
        depth_scale = ref_d / (depth + ref_d)
        alpha = self._hydro_alpha * fade_in * fade_out * trail_fade * depth_scale

        # Head glow: bright spark at particle position (tidx == 0)
        is_head = trail_idx == 0
        alpha = np.where(is_head, alpha * 1.5, alpha)

        mask = on_screen & age_ok & (alpha > 1e-6)
        if not mask.any():
            return img

        sx_m = sx_all[mask]
        sy_m = sy_all[mask]
        alpha_m = alpha[mask].astype(np.float32)

        # Per-particle color/radius → per-trail-point via particle index
        particle_idx = np.arange(N * trail_len) // trail_len
        pidx_m = particle_idx[mask]
        color_m = self._hydro_particle_colors[pidx_m]  # (M, 3)
        radii_m = self._hydro_particle_radii[pidx_m]    # (M,)

        # Head glow: +1px radius at particle position
        is_head_m = trail_idx[mask] == 0
        radii_m = np.where(is_head_m, radii_m + 1, radii_m)

        for r_val in range(1, 7):
            r_mask = radii_m == r_val
            if not r_mask.any():
                continue
            sxr = sx_m[r_mask]
            syr = sy_m[r_mask]
            ar = alpha_m[r_mask]
            cr = color_m[r_mask]

            for offy in range(-r_val, r_val + 1):
                for offx in range(-r_val, r_val + 1):
                    dist_sq = offx * offx + offy * offy
                    if dist_sq > r_val * r_val:
                        continue
                    falloff = 1.0 - (dist_sq / (r_val * r_val)) ** 0.5

                    px = sxr + offx
                    py = syr + offy
                    ok = (px >= 0) & (px < sw) & (py >= 0) & (py < sh)
                    if not ok.any():
                        continue

                    contribution = ar[ok] * falloff
                    for c in range(3):
                        np.add.at(img[:, :, c], (py[ok], px[ok]),
                                  contribution * cr[ok, c])

        np.clip(img, 0, 1, out=img)
        return img

    def _splat_hydro_gpu(self, d_frame):
        """Project and splat hydro particles on GPU.  Delegates to HydroManager."""
        terrain_data = self.raster.data
        if has_cupy and not isinstance(terrain_data, cp.ndarray):
            terrain_data = cp.asarray(terrain_data)

        depth_t = getattr(self, '_d_depth_t', None)

        self.hydro_mgr.splat_gpu(
            d_frame,
            camera_pos=self.position,
            look_at=self._get_look_at(),
            fov=self.fov,
            ve=self.vertical_exaggeration,
            subsample_factor=self.subsample_factor,
            terrain_gpu=terrain_data,
            depth_t=depth_t,
        )

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

    # ------------------------------------------------------------------
    # Thin action methods for key-binding dispatch
    # ------------------------------------------------------------------

    def _action_shift_o(self):
        obs = self._observers.get(self._active_observer) if self._active_observer else None
        if obs is None:
            print("No observer selected. Press 1-8 first.")
        else:
            self._cycle_drone_mode_for(obs)

    def _action_shift_v(self):
        obs = self._observers.get(self._active_observer) if self._active_observer else None
        if obs is None:
            print("No observer selected. Press 1-8 first.")
        else:
            self._snap_to_observer(obs)

    def _action_clear_observers(self):
        self._clear_all_observers()

    def _action_toggle_firms(self):
        self._toggle_firms()

    def _action_toggle_wind(self):
        self._toggle_wind()

    def _action_toggle_terrain_vis(self):
        from .viewer.terrain_lod import is_terrain_lod_gid
        # Toggle all LOD terrain tiles together
        vis = None
        for gid in self.rtx.list_geometries():
            if is_terrain_lod_gid(gid):
                if vis is None:
                    e = self.rtx._geom_state.gas_entries.get(gid)
                    vis = not e.visible if e is not None else True
                self.rtx.set_geometry_visible(gid, vis)
        if vis is not None:
            print(f"Terrain {'shown' if vis else 'hidden'}")
            self._needs_render = True

    def _action_toggle_gtfs_rt(self):
        self._toggle_gtfs_rt()

    def _action_cycle_pc_colors(self):
        self._cycle_pointcloud_colors()

    def _action_toggle_denoiser(self):
        self.denoise_enabled = not self.denoise_enabled
        self.render_settings.reset_accumulation()
        self._prev_cam_for_flow = None
        print(f"Denoiser: {'ON' if self.denoise_enabled else 'OFF'}")
        self._update_frame()

    def _action_cycle_gi(self):
        self.gi_bounces = self.gi_bounces % 3 + 1
        self.render_settings.reset_accumulation()
        print(f"GI bounces: {self.gi_bounces}")
        self._update_frame()

    def _action_prev_help_page(self):
        if self._help_pages:
            self._help_page_idx -= 1
            if self._help_page_idx < -1:
                self._help_page_idx = len(self._help_pages) - 1
        self._update_frame()

    def _action_toggle_drone_glow(self):
        self._drone_glow = not self._drone_glow
        self._apply_drone_glow()
        print(f"Drone glow: {'ON' if self._drone_glow else 'OFF'}")

    def _action_cycle_time(self):
        self._time_preset_idx = (self._time_preset_idx + 1) % len(self._time_presets)
        name, az, alt = self._time_presets[self._time_preset_idx]
        self.sun_azimuth = az
        self.sun_altitude = alt
        self.render_settings.reset_accumulation()
        print(f"Time of day: {name} (az={az:.0f}, alt={alt:.0f})")
        self._update_frame()

    def _action_toggle_shadows(self):
        self.shadows = not self.shadows
        print(f"Shadows: {'ON' if self.shadows else 'OFF'}")
        self._update_frame()

    def _action_cycle_colormap(self):
        self.colormap_idx = (self.colormap_idx + 1) % len(self.colormaps)
        self.colormap = self.colormaps[self.colormap_idx]
        print(f"Colormap: {self.colormap}")
        self._update_frame()

    def _action_jump_prev_geom(self):
        self._jump_to_geometry(-1)

    def _action_next_help_page(self):
        if self._help_pages:
            self._help_page_idx += 1
            if self._help_page_idx >= len(self._help_pages):
                self._help_page_idx = -1
        else:
            self._help_page_idx = -1
        self._update_frame()

    def _action_toggle_minimap(self):
        self.show_minimap = not self.show_minimap
        self._update_frame()

    def _action_place_observer(self):
        obs = self._observers.get(self._active_observer) if self._active_observer else None
        if obs is None:
            print("No observer selected. Press 1-8 to create one.")
        elif obs.drone_mode == 'off':
            self._place_observer_at(obs)

    def _action_observer_elev_down(self):
        self._adjust_observer_elevation(-0.01)

    def _action_observer_elev_up(self):
        self._adjust_observer_elevation(0.01)

    def _action_cycle_color_stretch(self):
        self._color_stretch_idx = (self._color_stretch_idx + 1) % len(self._color_stretches)
        self.color_stretch = self._color_stretches[self._color_stretch_idx]
        print(f"Color stretch: {self.color_stretch}")
        self._update_frame()

    def _action_cycle_basemap_fwd(self):
        self._cycle_basemap()

    def _action_cycle_basemap_rev(self):
        self._cycle_basemap(reverse=True)

    def _action_overlay_alpha_down(self):
        self._overlay_alpha = max(0.0, round(self._overlay_alpha - 0.1, 1))
        print(f"Overlay alpha: {int(self._overlay_alpha * 100)}%")
        self._update_frame()

    def _action_overlay_alpha_up(self):
        self._overlay_alpha = min(1.0, round(self._overlay_alpha + 0.1, 1))
        print(f"Overlay alpha: {int(self._overlay_alpha * 100)}%")
        self._update_frame()

    def _action_speed_up(self):
        H, W = self.terrain_shape
        world_diag = np.sqrt((W * self.pixel_spacing_x)**2 + (H * self.pixel_spacing_y)**2)
        max_speed = world_diag * 0.1
        self.move_speed = min(max_speed, self.move_speed * 1.2)
        print(f"Speed: {self.move_speed:.3f}")

    def _action_speed_down(self):
        self.move_speed = max(0.001, self.move_speed / 1.2)
        print(f"Speed: {self.move_speed:.3f}")

    def _action_resolution_coarser(self):
        new_factor = min(8, self.subsample_factor * 2)
        if new_factor != self.subsample_factor:
            self._rebuild_at_resolution(new_factor)

    def _action_resolution_finer(self):
        new_factor = max(1, self.subsample_factor // 2)
        if new_factor != self.subsample_factor:
            self._rebuild_at_resolution(new_factor)

    def _action_ve_down(self):
        new_ve = max(0.1, round(self.vertical_exaggeration - 0.1, 1))
        if new_ve != self.vertical_exaggeration:
            self._rebuild_vertical_exaggeration(new_ve)

    def _action_ve_up(self):
        new_ve = min(10.0, round(self.vertical_exaggeration + 0.1, 1))
        if new_ve != self.vertical_exaggeration:
            self._rebuild_vertical_exaggeration(new_ve)

    def _action_toggle_ao(self):
        self.ao_enabled = not self.ao_enabled
        self.render_settings.reset_accumulation()
        print(f"Ambient Occlusion: {'ON' if self.ao_enabled else 'OFF'}")
        self._update_frame()

    def _action_toggle_dof(self):
        self.dof_enabled = not self.dof_enabled
        self.render_settings.reset_accumulation()
        print(f"Depth of Field: {'ON' if self.dof_enabled else 'OFF'}")
        self._update_frame()

    def _action_dof_aperture_down(self):
        self._dof_aperture = max(1.0, self._dof_aperture * 0.7)
        self.render_settings.reset_accumulation()
        print(f"DOF aperture: {self._dof_aperture:.1f}")
        self._update_frame()

    def _action_dof_aperture_up(self):
        self._dof_aperture = min(200.0, self._dof_aperture * 1.4)
        self.render_settings.reset_accumulation()
        print(f"DOF aperture: {self._dof_aperture:.1f}")
        self._update_frame()

    def _action_dof_focal_down(self):
        self._dof_focal_distance = max(10.0, self._dof_focal_distance * 0.7)
        self.render_settings.reset_accumulation()
        print(f"DOF focal distance: {self._dof_focal_distance:.0f}")
        self._update_frame()

    def _action_dof_focal_up(self):
        self._dof_focal_distance = min(10000.0, self._dof_focal_distance * 1.4)
        self.render_settings.reset_accumulation()
        print(f"DOF focal distance: {self._dof_focal_distance:.0f}")
        self._update_frame()

    def _action_exit(self):
        self.running = False

    # ------------------------------------------------------------------
    # Key dispatch
    # ------------------------------------------------------------------

    def _handle_key_press(self, raw_key, key):
        """Handle key press — table-driven dispatch.

        Parameters
        ----------
        raw_key : str
            Key with original case (uppercase if SHIFT held).
        key : str
            Lowercase version of the key.
        """
        # 1. Shift bindings (uppercase raw_key)
        if raw_key in SHIFT_BINDINGS:
            getattr(self, SHIFT_BINDINGS[raw_key])()
            return

        # 2. Movement keys → add to held set
        if key in MOVEMENT_KEYS:
            self._held_keys.add(key)
            return

        # 3. Observer slots 1-8
        if key in ('1', '2', '3', '4', '5', '6', '7', '8'):
            self._select_or_create_observer(int(key))
            return

        # 4. Special bindings (need both raw_key and key)
        pair = (raw_key, key)
        if pair in SPECIAL_BINDINGS:
            getattr(self, SPECIAL_BINDINGS[pair])()
            return

        # 5. Regular key bindings (lowercase key)
        if key in KEY_BINDINGS:
            getattr(self, KEY_BINDINGS[key])()
            return

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
        """Check if camera is near terrain edge and prefetch the next window.

        The terrain loader runs in a background thread so it doesn't block
        the render loop (erosion/hydro can take many seconds).  Each tick
        we either (a) submit a new loader job if near-edge, or (b) poll
        for a completed result and swap in the new terrain.

        Prefetch strategy: triggers at 40% from any edge (not 20%) so the
        load starts well before the camera reaches the boundary.  The load
        center is offset in the camera's direction of travel so the new
        terrain extends further ahead.
        """
        if self._terrain_loader is None:
            return
        # Streaming LOD handles edge loading — skip terrain replacement
        if (getattr(self, '_tile_data_fn', None) is not None
                and self.lod_enabled):
            return

        # --- Phase 2: check for completed background load ---
        future = self.terrain._terrain_reload_future
        if future is not None:
            if not future.done():
                return  # still computing — keep rendering
            # Harvest result
            self.terrain._terrain_reload_future = None
            try:
                result, cam_lon, cam_lat = future.result()
            except Exception as e:
                print(f"Terrain loader error: {e}")
                self._last_reload_time = time.time()
                return
            if result is None:
                self._last_reload_time = time.time()
                return
            self._apply_terrain_reload(result, cam_lon, cam_lat)
            return

        # --- Phase 1: decide whether to submit a new load ---
        now = time.time()
        if now - self._last_reload_time < self._reload_cooldown:
            return

        if self.position is None:
            return

        H, W = self.terrain_shape
        # Camera position relative to the terrain grid (accounting for
        # any world offset from previous reloads).
        ox = self.terrain._world_offset_x
        oy = self.terrain._world_offset_y
        cam_col = (self.position[0] - ox) / self.pixel_spacing_x
        cam_row = (self.position[1] - oy) / self.pixel_spacing_y

        # Prefetch at 40% from any edge — starts loading well before
        # the camera reaches the boundary.
        margin_x = W * 0.4
        margin_y = H * 0.4
        near_edge = (cam_col < margin_x or cam_col > W - margin_x or
                     cam_row < margin_y or cam_row > H - margin_y)
        if not near_edge:
            return

        # Compute camera lon/lat from world position
        cam_lon = self._coord_origin_x + cam_col * self._coord_step_x
        cam_lat = self._coord_origin_y + cam_row * self._coord_step_y

        # Offset load center in the direction of camera travel so the
        # new terrain extends further ahead of the camera.
        front = self._get_front()
        fx, fy = float(front[0]), float(front[1])
        flen = np.sqrt(fx * fx + fy * fy)
        if flen > 0.01:
            # Offset by 25% of the window in the camera's forward direction,
            # converted from pixel space to geographic coordinates.
            offset_px_x = (W * 0.25) * (fx / flen)
            offset_px_y = (H * 0.25) * (fy / flen)
            cam_lon += offset_px_x * self._coord_step_x
            cam_lat += offset_px_y * self._coord_step_y

        # Submit loader to background thread
        from concurrent.futures import ThreadPoolExecutor
        pool = self.terrain._terrain_reload_pool
        if pool is None:
            pool = ThreadPoolExecutor(max_workers=1)
            self.terrain._terrain_reload_pool = pool

        loader = self._terrain_loader

        def _bg_load(lon, lat):
            return loader(lon, lat), lon, lat

        self.terrain._terrain_reload_future = pool.submit(_bg_load, cam_lon, cam_lat)
        # Prevent re-submission while the load is in progress
        self._last_reload_time = now + 999999

    def _apply_terrain_reload(self, result, cam_lon, cam_lat):
        """Apply a completed terrain reload result (runs on main thread).

        The camera position is kept stable — instead of teleporting the
        camera to its new-grid coordinates, we offset the terrain vertices
        so the same geographic point maps to the same world-space position.
        This eliminates the jarring jump that would otherwise occur.
        """
        new_hydro = None
        if isinstance(result, tuple):
            new_raster, new_hydro = result
        else:
            new_raster = result

        # --- Compute world offset to keep camera stable ---
        old_pos_x = self.position[0]
        old_pos_y = self.position[1]
        cam_z = self.position[2]

        # Extract coordinate metadata from new raster
        new_origin_x = float(new_raster.x.values[0])
        new_origin_y = float(new_raster.y.values[0])
        new_step_x = float(new_raster.x.values[1] - new_raster.x.values[0])
        new_step_y = float(new_raster.y.values[1] - new_raster.y.values[0])

        # Where the camera would land in the new grid (pixel coords)
        new_col = (cam_lon - new_origin_x) / new_step_x
        new_row = (cam_lat - new_origin_y) / new_step_y

        # Offset = current world position minus where new grid would
        # place the camera.  Adding this to all vertices keeps the camera
        # at (old_pos_x, old_pos_y) without moving it.
        psx = self.pixel_spacing_x
        psy = self.pixel_spacing_y
        offset_x = old_pos_x - new_col * psx
        offset_y = old_pos_y - new_row * psy
        self.terrain._world_offset_x = offset_x
        self.terrain._world_offset_y = offset_y

        # Update coordinate mapping so world-to-geo still works:
        # lon = coord_origin_x + (pos_x / psx) * coord_step_x
        # We need this to produce cam_lon when pos_x = old_pos_x.
        self._coord_origin_x = cam_lon - (old_pos_x / psx) * new_step_x
        self._coord_origin_y = cam_lat - (old_pos_y / psy) * new_step_y
        self._coord_step_x = new_step_x
        self._coord_step_y = new_step_y

        # Replace rasters
        self._base_raster = new_raster
        self.raster = new_raster
        self._wind_terrain_np = None  # invalidate cached terrain
        self._hydro_terrain_np = None
        self._d_base_frame = None     # invalidate GPU wind/hydro buffers
        self._d_wind_scratch = None
        self._d_depth_t = None        # invalidate depth buffer

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

        # Clear all terrain caches (old window geometry is stale)
        self.terrain.clear_all_caches()

        # --- Rebuild terrain mesh via LOD manager ---
        if self._terrain_lod_manager is not None:
            mgr = self._terrain_lod_manager
            mgr.set_terrain(terrain_np, offset_x=offset_x, offset_y=offset_y)
            # Force immediate full tile rebuild
            saved_limit = mgr.per_tick_build_limit
            mgr.per_tick_build_limit = 10000
            mgr.update(self.position, self.rtx,
                        ve=ve, force=True,
                        camera_front=self._get_front(), fov=self.camera.fov)
            mgr.per_tick_build_limit = saved_limit

        # Reinitialize hydro for new terrain
        if new_hydro is not None and self._hydro_data is not None:
            was_enabled = self._hydro_enabled
            flow_accum = new_hydro['flow_accum']
            hydro_opts = {k: v for k, v in new_hydro.items()
                          if k not in ('flow_accum', 'enabled')}
            self._init_hydro(flow_accum, **hydro_opts)
            self._hydro_enabled = was_enabled
        elif self._hydro_lazy and self._hydro_data is not None:
            was_enabled = self._hydro_enabled
            self._compute_hydro_from_terrain()
            self._hydro_enabled = was_enabled

        # Camera stays at its current position — no jump.
        # Only update Z if the terrain height changed significantly
        # under the camera (keeps altitude above ground consistent).

        # Refresh minimap
        self._minimap_bg_extent = None
        self._compute_minimap_background()

        self._last_reload_time = time.time()
        self._render_needed = True
        print(f"Terrain reloaded: center ({cam_lon:.4f}, {cam_lat:.4f}), "
              f"window {new_W}x{new_H}")

    def _tick(self):
        """Per-frame tick: delta time → input/movement → simulation → render."""
        if not self.running:
            return

        # --- Delta time ---
        now = time.monotonic()
        dt = min(now - self._last_tick_time, 0.1)
        self._last_tick_time = now
        dt_scale = dt / 0.05  # 0.05 = 1/20 Hz reference

        # --- Input / movement ---
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

        # --- Simulation (terrain reload, chunk loading, AO accumulation) ---
        self._check_terrain_reload()
        # Terrain LOD runs first so tile_lods are fresh for chunk manager
        if self.lod_enabled and self._terrain_lod_manager is not None:
            if self._terrain_lod_manager.update(
                    self.position, self.rtx,
                    ve=self.vertical_exaggeration,
                    camera_front=self._get_front(), fov=self.camera.fov):
                self._render_needed = True
        if self._chunk_manager is not None:
            # When LOD is active, sync distance parameters from LOD manager
            if (self.lod_enabled and self._terrain_lod_manager is not None
                    and self._terrain_lod_manager._lod_distances):
                self._chunk_manager.max_distance = (
                    self._terrain_lod_manager._lod_distances[-1])
                self._chunk_manager._lod_distances = (
                    self._terrain_lod_manager._lod_distances)
                # When grids are aligned, pass tile LOD assignments directly
                # so the chunk manager skips its own distance computation.
                self._chunk_manager._tile_lods = (
                    self._terrain_lod_manager._tile_lods)
            else:
                self._chunk_manager._lod_distances = None
                self._chunk_manager._tile_lods = None
            if self._chunk_manager.update(self.position[0], self.position[1], self):
                self._geometry_colors_builder = self._accessor._build_geometry_colors_gpu
                self._render_needed = True
        # AO/DOF: keep accumulating samples when camera is stationary
        if ((self.ao_enabled or self.dof_enabled) and not self._held_keys
                and not self._mouse_dragging
                and self._ao_frame_count < self._ao_max_frames):
            self._render_needed = True

        # --- Render ---
        if self._render_needed:
            self._update_frame()
            self._render_needed = False
        elif ((self._wind_enabled or self._hydro_enabled
               or (self._clouds_enabled and self._rain_particles is not None))
              and self._d_base_frame is not None):
            # Particles active but camera didn't move — skip the expensive ray
            # trace and just re-advect particles + GPU splat on fresh copy.
            cp.copyto(self._d_wind_scratch, self._d_base_frame)
            if self._wind_enabled and self._wind_particles is not None:
                self._update_wind_particles()
                self._splat_wind_gpu(self._d_wind_scratch)
            if self._hydro_enabled and self._hydro_particles is not None:
                self.hydro_mgr.check_streaming_result()
                self._transfer_streaming_overlay()
                cam_r = self.position[1] / self._base_pixel_spacing_y
                cam_c = self.position[0] / self._base_pixel_spacing_x
                self.hydro_mgr.update_streaming_window(cam_r, cam_c)
                self._update_hydro_particles()
                self._splat_hydro_gpu(self._d_wind_scratch)
            if self._clouds_enabled and self._rain_particles is not None:
                self._update_rain_particles()
                self._splat_rain_gpu(self._d_wind_scratch)
            self._d_wind_scratch.get(out=self._pinned_frame)
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
            self._overlay_as_water = False
            self._active_overlay_color_lut = None
            print(f"Terrain: elevation")
        else:
            self._active_color_data = None
            self._active_overlay_data = self._overlay_layers[layer_name]
            self._overlay_as_water = (
                layer_name.startswith('flood_')
                or (layer_name == 'stream_link' and self._hydro_enabled))
            self._active_overlay_color_lut = self._overlay_color_luts.get(
                layer_name)
            if self._overlay_as_water:
                print(f"Terrain: {layer_name} (water)")
            else:
                alpha_pct = int(self._overlay_alpha * 100)
                print(f"Terrain: {layer_name} (alpha {alpha_pct}%, ,/. to adjust)")

        self._update_frame()

    def _cycle_basemap(self, reverse=False):
        """Cycle basemap: none → satellite → osm → none.

        Auto-creates XYZTileService on-the-fly if needed.
        """
        step = -1 if reverse else 1
        self._basemap_idx = (self._basemap_idx + step) % len(self._basemap_options)
        provider = self._basemap_options[self._basemap_idx]

        if provider == 'none':
            self._tiles_enabled = False
            if self._texture_tile_mgr is not None:
                self._texture_tile_mgr.clear()
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
            else:
                self._tile_service = XYZTileService(
                    url_template=provider, raster=self._base_raster,
                )
            self._tiles_enabled = True
            print(f"Basemap: {provider}")
            # When LOD active, clear and re-fetch basemap lazily per tile.
            # When LOD not active, use monolithic fetch.
            if (self._texture_tile_mgr is not None
                    and self.lod_enabled
                    and self._terrain_lod_manager is not None):
                self._texture_tile_mgr.clear()
                # Re-fire tile callbacks for all visible tiles so basemap
                # gets fetched for them in background threads.
                mgr = self._terrain_lod_manager
                if mgr._on_tile_added is not None:
                    for (tr, tc) in list(mgr._tile_lods.keys()):
                        mgr._fire_tile_added(tr, tc)
            else:
                self._tile_service.fetch_visible_tiles()

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
                if base_name == layer_name or geom_id == layer_name:
                    self.rtx.set_geometry_visible(geom_id, True)
                    visible_count += 1
                else:
                    self.rtx.set_geometry_visible(geom_id, False)
            print(f"Geometry: {layer_name} ({visible_count} visible)")

        self._current_geom_idx = 0
        self._update_frame()

    def _cycle_pointcloud_colors(self):
        """Cycle point cloud color mode: elevation → intensity → classification → rgb."""
        acc = self._accessor
        if acc is None or not hasattr(acc, '_pc_attributes') or not acc._pc_attributes:
            print("No point cloud geometries in scene")
            return

        from .pointcloud import build_colors

        self._pc_color_mode_idx = (self._pc_color_mode_idx + 1) % len(self._pc_color_modes)
        mode = self._pc_color_modes[self._pc_color_mode_idx]

        updated = 0
        for gid, (centers, attributes) in acc._pc_attributes.items():
            # Check the geometry still exists
            if self.rtx is None or not self.rtx.has_geometry(gid):
                continue

            # Build new colors
            new_colors = build_colors(centers, attributes, color_mode=mode)
            colors_flat = new_colors.ravel().astype(np.float32)

            # Update per-point colors on the RTX geometry state
            gs = self.rtx._geom_state
            gs.point_colors_per_gas[gid] = colors_flat
            # Invalidate concatenated GPU buffer so it gets rebuilt
            gs.point_colors = None
            gs.point_color_offsets = None

            # Update baked mesh colors (3rd element of 4-tuple)
            if gid in acc._baked_meshes:
                baked = acc._baked_meshes[gid]
                acc._baked_meshes[gid] = (baked[0], baked[1], colors_flat.copy(), baked[3])

            updated += 1

        if updated > 0:
            print(f"Point cloud color: {mode}")
            self._d_ao_accum = None
            self._ao_frame_count = 0
            self._prev_cam_state = None
            self._update_frame()
        else:
            print(f"No active point cloud geometries to recolor")

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

        # Basemap texture for screenshot
        rgb_texture = None
        _tex_off_y = 0
        _tex_off_x = 0
        if (self._texture_tile_mgr is not None
                and self.lod_enabled
                and self._terrain_lod_manager is not None):
            visible = set(self._terrain_lod_manager._tile_lods.keys())
            d_tex, tex_r, tex_c = self._texture_tile_mgr.get_composite(
                visible)
            if d_tex is not None:
                rgb_texture = d_tex
                _tex_off_y = tex_r
                _tex_off_x = tex_c
        elif self._tiles_enabled and self._tile_service is not None:
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
        # fog_density is scene-relative; convert to absolute for the kernel
        _fog = self.fog_density / self._scene_diagonal if self.fog_density > 0 else 0.0

        # Cloud fog map for screenshot — reuse cached GPU array
        _cloud_fog_map = None
        _cloud_fog_density = 0.0
        if self._clouds_enabled and self._cloud_cover_grid is not None:
            _d = getattr(self, '_d_cloud_fog_map', None)
            f = self.subsample_factor
            if _d is None or getattr(self, '_cloud_fog_subsample', 0) != f:
                src = self._cloud_cover_grid
                if f > 1:
                    src = src[::f, ::f]
                self._d_cloud_fog_map = cp.asarray(src, dtype=cp.float32)
                self._cloud_fog_subsample = f
            _cloud_fog_map = self._d_cloud_fog_map
            _cloud_fog_density = 12.0 / self._scene_diagonal

        # Resolve overlay: per-tile composite when LOD active,
        # else monolithic array.  Used by both screenshot and live paths.
        _ov_data = self._active_overlay_data
        _ov_lut = self._active_overlay_color_lut
        _ov_off_y = 0
        _ov_off_x = 0
        if (self._overlay_tile_mgr is not None
                and self._active_overlay_data is not None
                and self.lod_enabled
                and self._terrain_lod_manager is not None):
            visible = set(self._terrain_lod_manager._tile_lods.keys())
            d_comp, off_r, off_c = self._overlay_tile_mgr.get_composite(
                visible)
            if d_comp is not None:
                _ov_data = d_comp
                _ov_off_y = off_r
                _ov_off_x = off_c
                lut = self._overlay_tile_mgr.color_lut
                if lut is not None:
                    _ov_lut = lut

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
            fog_density=_fog,
            fog_color=self.fog_color,
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
            rgb_texture_offset_y=_tex_off_y,
            rgb_texture_offset_x=_tex_off_x,
            overlay_data=_ov_data,
            overlay_alpha=self._overlay_alpha,
            overlay_as_water=self._overlay_as_water,
            overlay_color_lut=_ov_lut,
            overlay_offset_y=_ov_off_y,
            overlay_offset_x=_ov_off_x,
            geometry_colors=geometry_colors,
            cloud_fog_map=_cloud_fog_map,
            cloud_fog_density=_cloud_fog_density,
            volumetric_clouds=self._volumetric_clouds_enabled and self._clouds_enabled,
            cloud_base_z=self._cloud_altitude,
            cloud_top_z=self._cloud_altitude + self._cloud_thickness,
            cloud_time=self._cloud_time,
        )

        # Accumulated multi-frame screenshot when AO or DOF is enabled
        num_frames = 64 if (self.ao_enabled or self.dof_enabled) else 1

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

        # Basemap texture — per-tile composite when LOD active,
        # else monolithic GPU texture from tile service.
        rgb_texture = None
        _tex_off_y = 0
        _tex_off_x = 0
        if (self._texture_tile_mgr is not None
                and self.lod_enabled
                and self._terrain_lod_manager is not None):
            visible = set(self._terrain_lod_manager._tile_lods.keys())
            d_tex, tex_r, tex_c = self._texture_tile_mgr.get_composite(
                visible)
            if d_tex is not None:
                rgb_texture = d_tex
                _tex_off_y = tex_r
                _tex_off_x = tex_c
        elif self._tiles_enabled and self._tile_service is not None:
            rgb_texture = self._tile_service.get_gpu_texture()
            if rgb_texture is not None and self.subsample_factor > 1:
                f = self.subsample_factor
                rgb_texture = rgb_texture[::f, ::f, :]

        # Build geometry colors GPU LUT if a builder is available
        geometry_colors = None
        builder = getattr(self, '_geometry_colors_builder', None)
        if builder is not None:
            geometry_colors = builder()

        # Progressive accumulation is needed for AO convergence and/or
        # DOF (thin-lens jitter needs multi-frame averaging to converge).
        needs_accum = self.ao_enabled or self.dof_enabled

        # AO parameters: multiple samples per frame for smooth early results,
        # with progressive accumulation across frames for further refinement
        ao_samples = self._ao_samples_per_frame if self.ao_enabled else 0
        ao_seed = self._ao_frame_count if self.ao_enabled else 0

        # When progressive accumulation is active, pass frame seed for AA + soft shadows + DOF
        frame_seed = self._ao_frame_count + 1 if needs_accum else 0

        # Depth of field
        if self.dof_enabled:
            dof_aperture = self._dof_aperture
            dof_focal = self._dof_focal_distance
        else:
            dof_aperture = 0.0
            dof_focal = 0.0

        # When progressive AO accumulation or denoising is active, defer
        # bloom and tone mapping until after averaging / denoising.  Both
        # are non-linear operations that must act on the clean signal.
        defer_post = self.ao_enabled or self.denoise_enabled

        # fog_density is scene-relative; convert to absolute for the kernel
        _fog = self.fog_density / self._scene_diagonal if self.fog_density > 0 else 0.0

        # Cloud fog map: pass cloud_cover_grid for cloud shadow modulation
        _cloud_fog_map = None
        _cloud_fog_density = 0.0
        if self._clouds_enabled and self._cloud_cover_grid is not None:
            # Cache GPU array to avoid per-frame upload
            _d = getattr(self, '_d_cloud_fog_map', None)
            f = self.subsample_factor
            if _d is None or getattr(self, '_cloud_fog_subsample', 0) != f:
                src = self._cloud_cover_grid
                if f > 1:
                    src = src[::f, ::f]
                self._d_cloud_fog_map = cp.asarray(src, dtype=cp.float32)
                self._cloud_fog_subsample = f
            _cloud_fog_map = self._d_cloud_fog_map
            # Spatial frequency: ~12 cloud cells across scene
            _cloud_fog_density = 12.0 / self._scene_diagonal

        # Resolve overlay: per-tile composite when LOD active,
        # else monolithic array.
        _ov_data = self._active_overlay_data
        _ov_lut = self._active_overlay_color_lut
        _ov_off_y = 0
        _ov_off_x = 0
        if (self._overlay_tile_mgr is not None
                and self._active_overlay_data is not None
                and self.lod_enabled
                and self._terrain_lod_manager is not None):
            visible = set(self._terrain_lod_manager._tile_lods.keys())
            d_comp, off_r, off_c = self._overlay_tile_mgr.get_composite(
                visible)
            if d_comp is not None:
                _ov_data = d_comp
                _ov_off_y = off_r
                _ov_off_x = off_c
                lut = self._overlay_tile_mgr.color_lut
                if lut is not None:
                    _ov_lut = lut

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
            fog_density=_fog,
            fog_color=self.fog_color,
            colormap=self.colormap,
            rtx=self.rtx,
            viewshed_data=viewshed_data,
            viewshed_opacity=self.viewshed_opacity,
            observer_position=observer_pos,
            pixel_spacing_x=self.pixel_spacing_x,
            pixel_spacing_y=self.pixel_spacing_y,
            mesh_type='heightfield',
            color_data=self._active_color_data,
            color_stretch=self.color_stretch,
            color_range=self._land_color_range,
            rgb_texture=rgb_texture,
            rgb_texture_offset_y=_tex_off_y,
            rgb_texture_offset_x=_tex_off_x,
            overlay_data=_ov_data,
            overlay_alpha=self._overlay_alpha,
            overlay_as_water=self._overlay_as_water,
            overlay_color_lut=_ov_lut,
            overlay_offset_y=_ov_off_y,
            overlay_offset_x=_ov_off_x,
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
            edge_strength=0.2,
            edge_color=(0.15, 0.13, 0.10),
            edl=self.edl_enabled,
            cloud_fog_map=_cloud_fog_map,
            cloud_fog_density=_cloud_fog_density,
            volumetric_clouds=self._volumetric_clouds_enabled and self._clouds_enabled,
            cloud_base_z=self._cloud_altitude,
            cloud_top_z=self._cloud_altitude + self._cloud_thickness,
            cloud_time=self._cloud_time,
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

        # Extract depth buffer from primary hits for hydro occlusion culling.
        # primary_hits[:, 0] holds ray t-values; persists until next render.
        if self._hydro_enabled:
            from .analysis.render import _render_buffers as _depth_bufs
            if _depth_bufs.primary_hits is not None:
                h, w = self.render_height, self.render_width
                _ddt = getattr(self, '_d_depth_t', None)
                if _ddt is None or _ddt.shape != (h, w):
                    self._d_depth_t = cp.empty((h, w), dtype=cp.float32)
                t_src = _depth_bufs.primary_hits.reshape(h * w, 4)[:, 0]
                self._d_depth_t[:] = t_src.reshape(h, w)

        # Progressive accumulation (needed for AO convergence and/or DOF)
        needs_accum = self.ao_enabled or self.dof_enabled
        if needs_accum:
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
        # These are deferred when AO, DOF, or denoiser is active so they
        # operate on the clean / averaged signal.
        defer_post = needs_accum or self.denoise_enabled
        if defer_post:
            if not needs_accum:
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

        # Save clean post-processed frame for idle wind/hydro/rain replay
        _any_particles = (self._wind_enabled or self._hydro_enabled
                          or (self._clouds_enabled and self._rain_particles is not None))
        if _any_particles:
            if self._d_base_frame is None or self._d_base_frame.shape != d_display.shape:
                self._d_base_frame = cp.empty_like(d_display)
                self._d_wind_scratch = cp.empty_like(d_display)
            cp.copyto(self._d_base_frame, d_display)

        # GPU wind: advect on CPU, splat on GPU, then readback
        if self._wind_enabled and self._wind_particles is not None:
            self._update_wind_particles()
            self._splat_wind_gpu(d_display)

        # GPU hydro: advect + splat on GPU
        if self._hydro_enabled and self._hydro_particles is not None:
            self.hydro_mgr.check_streaming_result()
            self._transfer_streaming_overlay()
            cam_r = self.position[1] / self._base_pixel_spacing_y
            cam_c = self.position[0] / self._base_pixel_spacing_x
            self.hydro_mgr.update_streaming_window(cam_r, cam_c)
            self._update_hydro_particles()
            self._splat_hydro_gpu(d_display)

        # GPU rain: advect on CPU, splat on GPU (cloud fog is baked into ray trace)
        if self._clouds_enabled and self._rain_particles is not None:
            self._update_rain_particles()
            self._splat_rain_gpu(d_display)

        # Advance volumetric cloud animation time
        if self._clouds_enabled and self._volumetric_clouds_enabled:
            self._cloud_time += 0.05

        # Sync: splat kernels run on stream 0, readback on non-blocking stream
        if _any_particles:
            sync_event = self._wind_done_event or self._hydro_done_event
            if sync_event is None:
                sync_event = cp.cuda.Event()
                if self._wind_enabled:
                    self._wind_done_event = sync_event
                else:
                    self._hydro_done_event = sync_event
            sync_event.record()
            self._readback_stream.wait_event(sync_event)

        # Async D2H copy on non-blocking stream
        d_display.get(out=self._pinned_frame, stream=self._readback_stream)

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
        _help_visible = (self._help_page_idx >= 0 and self._help_pages)
        needs_overlay = (
            (self._gtfs_rt_enabled and self._gtfs_rt_vehicles is not None)
            or self.show_minimap
            or self._title_overlay_rgba is not None
            or self._legend_rgba is not None
            or _help_visible
        )
        if needs_overlay:
            img = self._pinned_frame.copy()
        else:
            img = self._pinned_frame

        # GTFS-RT vehicle overlay
        if self._gtfs_rt_enabled and self._gtfs_rt_vehicles is not None:
            self._draw_gtfs_rt_on_frame(img)

        # Minimap overlay
        self._blit_minimap_on_frame(img)

        # Title overlay
        self._blit_title_on_frame(img)

        # Legend overlay (always visible)
        self._blit_legend_on_frame(img)

        # Help page overlay
        if _help_visible:
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
                    # Convert minimap-local → world XY
                    local_x = frame_x - mx0
                    local_y = frame_y - my0
                    ext = self._minimap_world_extent
                    if ext is not None:
                        wx_min, wy_min, wx_max, wy_max = ext
                        world_x = wx_min + local_x / mw * (wx_max - wx_min)
                        world_y = wy_min + local_y / mh * (wy_max - wy_min)
                    else:
                        H, W = self.terrain_shape
                        world_x = local_x / mw * W * self.pixel_spacing_x
                        world_y = local_y / mh * H * self.pixel_spacing_y
                    self.position[0] = world_x
                    self.position[1] = world_y
                    self._update_frame()
                    return

            self._mouse_dragging = True
            self._mouse_last_x = xpos
            self._mouse_last_y = ypos

        elif button == 1:  # right click — object picking
            origin, direction = self._screen_to_ray(xpos, ypos)
            result = self.rtx.pick(origin, direction)
            if result['hit']:
                gid = result['geometry_id'] or '?'
                px, py, pz = result['position']
                print(f"Pick: geometry='{gid}'  pos=({px:.1f}, {py:.1f}, {pz:.1f})  "
                      f"t={result['t']:.1f}  prim={result['primitive_id']}  "
                      f"instance={result['instance_id']}")
            else:
                print("Pick: no geometry hit")

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

    # ------------------------------------------------------------------
    # Title & legend overlays
    # ------------------------------------------------------------------

    def _render_title_overlay(self):
        """Pre-render title + subtitle with drop-shadow to an RGBA array.

        Rendered at 2x resolution for antialiased text, then downsampled.
        No background — text floats over the scene with a strong shadow.
        """
        if not self._title or self._title == 'rtxpy':
            return
        try:
            from PIL import Image, ImageDraw, ImageFont, ImageFilter

            bold_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
            sans_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"

            # Render at 2x for antialiasing via supersample
            ss = 2
            try:
                font_title = ImageFont.truetype(bold_path, 22 * ss)
                font_sub = ImageFont.truetype(sans_path, 11 * ss)
            except (OSError, IOError):
                font_title = ImageFont.load_default()
                font_sub = font_title
                ss = 1  # fall back to 1x with default font

            pad_x, pad_y = 14 * ss, 10 * ss
            shadow_spread = 4 * ss  # extra margin for shadow blur

            # Measure title
            title_bbox = font_title.getbbox(self._title)
            title_w = title_bbox[2] - title_bbox[0]
            title_h = title_bbox[3] - title_bbox[1]

            # Measure subtitle
            sub_h = 0
            sub_w = 0
            if self._subtitle:
                sub_bbox = font_sub.getbbox(self._subtitle)
                sub_w = sub_bbox[2] - sub_bbox[0]
                sub_h = sub_bbox[3] - sub_bbox[1] + 4 * ss

            img_w = pad_x * 2 + max(title_w, sub_w) + shadow_spread * 2
            img_h = pad_y * 2 + title_h + sub_h + shadow_spread * 2

            # --- Shadow layer: black text, then blur ---
            shadow_img = Image.new('RGBA', (img_w, img_h), (0, 0, 0, 0))
            shadow_draw = ImageDraw.Draw(shadow_img)
            sx = pad_x + shadow_spread
            sy = pad_y + shadow_spread

            # Draw shadow text with slight offset
            shadow_off = 2 * ss
            shadow_draw.text((sx + shadow_off, sy + shadow_off),
                             self._title, fill=(0, 0, 0, 255),
                             font=font_title)
            if self._subtitle:
                shadow_draw.text(
                    (sx + shadow_off, sy + title_h + 4 * ss + shadow_off),
                    self._subtitle, fill=(0, 0, 0, 255), font=font_sub)

            # Multi-pass blur for a deep, soft shadow
            for _ in range(3):
                shadow_img = shadow_img.filter(
                    ImageFilter.GaussianBlur(radius=3 * ss))

            # Boost shadow opacity
            shadow_arr = np.array(shadow_img, dtype=np.float32)
            shadow_arr[:, :, 3] = np.clip(shadow_arr[:, :, 3] * 2.5, 0, 255)
            shadow_img = Image.fromarray(shadow_arr.astype(np.uint8))

            # --- Foreground layer: crisp white text ---
            fg_img = Image.new('RGBA', (img_w, img_h), (0, 0, 0, 0))
            fg_draw = ImageDraw.Draw(fg_img)
            fg_draw.text((sx, sy), self._title,
                         fill=(255, 255, 255, 255), font=font_title)
            if self._subtitle:
                fg_draw.text((sx, sy + title_h + 4 * ss), self._subtitle,
                             fill=(200, 210, 225, 230), font=font_sub)

            # Composite: shadow behind, text on top
            result = Image.alpha_composite(shadow_img, fg_img)

            # Downsample from 2x to 1x with antialiasing
            if ss > 1:
                final_w = img_w // ss
                final_h = img_h // ss
                result = result.resize((final_w, final_h), Image.LANCZOS)

            self._title_overlay_rgba = (
                np.array(result, dtype=np.float32) / 255.0)
        except ImportError:
            pass

    def _render_legend_overlay(self):
        """Pre-render legend to an RGBA numpy array using PIL.

        Called once at startup; cached in ``self._legend_rgba``.
        Expects ``self._legend_config`` to be a dict with 'title' and
        'entries' keys.
        """
        if not self._legend_config:
            return
        entries = self._legend_config.get('entries', [])
        if not entries:
            return
        try:
            from PIL import Image, ImageDraw, ImageFont

            bold_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
            sans_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
            try:
                font_header = ImageFont.truetype(bold_path, 13)
                font_label = ImageFont.truetype(sans_path, 12)
            except (OSError, IOError):
                font_header = ImageFont.load_default()
                font_label = font_header

            pad_x, pad_y = 14, 10
            swatch_size = 12
            swatch_gap = 8   # gap between swatch and label
            line_h = 18
            bg_color = (15, 18, 24, 200)
            header_color = (180, 210, 255, 255)
            label_color = (210, 215, 225, 220)

            legend_title = self._legend_config.get('title', '')

            # Measure widths
            max_label_w = 0
            for label, _color in entries:
                bbox = font_label.getbbox(label)
                max_label_w = max(max_label_w, bbox[2] - bbox[0])

            content_w = swatch_size + swatch_gap + max_label_w
            if legend_title:
                hdr_bbox = font_header.getbbox(legend_title)
                content_w = max(content_w, hdr_bbox[2] - hdr_bbox[0])

            img_w = pad_x * 2 + content_w
            header_h = 0
            if legend_title:
                header_h = 13 + 8  # font size + gap below header
            img_h = pad_y * 2 + header_h + len(entries) * line_h

            img = Image.new('RGBA', (img_w, img_h), (0, 0, 0, 0))
            draw = ImageDraw.Draw(img)
            draw.rectangle(
                [0, 0, img_w - 1, img_h - 1],
                fill=bg_color,
            )
            draw.rectangle(
                [0, 0, img_w - 1, img_h - 1],
                outline=(60, 70, 90, 140), width=1,
            )

            y = pad_y
            if legend_title:
                draw.text((pad_x, y), legend_title,
                          fill=header_color, font=font_header)
                y += header_h

            for label, color_rgb in entries:
                # Colour swatch
                r8 = int(min(1.0, color_rgb[0]) * 255)
                g8 = int(min(1.0, color_rgb[1]) * 255)
                b8 = int(min(1.0, color_rgb[2]) * 255)
                sx = pad_x
                sy = y + (line_h - swatch_size) // 2
                draw.rectangle(
                    [sx, sy, sx + swatch_size - 1, sy + swatch_size - 1],
                    fill=(r8, g8, b8, 255),
                )
                # Label
                draw.text((pad_x + swatch_size + swatch_gap, y + 1),
                          label, fill=label_color, font=font_label)
                y += line_h

            self._legend_rgba = np.array(img, dtype=np.float32) / 255.0
        except ImportError:
            pass

    def _blit_title_on_frame(self, img):
        """Alpha-composite cached title overlay onto the rendered frame.

        Placed flush in the top-left corner (no margin).

        Parameters
        ----------
        img : ndarray, shape (H, W, 3), float32 0-1
            Rendered frame. Modified in-place.
        """
        if self._title_overlay_rgba is None:
            return
        ov = self._title_overlay_rgba
        oh, ow = ov.shape[:2]
        fh, fw = img.shape[:2]
        bh = min(oh, fh)
        bw = min(ow, fw)
        if bh <= 0 or bw <= 0:
            return
        alpha = ov[:bh, :bw, 3:4]
        rgb = ov[:bh, :bw, :3]
        region = img[:bh, :bw]
        region[:] = region * (1 - alpha) + rgb * alpha

    def _blit_legend_on_frame(self, img):
        """Alpha-composite cached legend overlay onto the rendered frame.

        Placed flush in the bottom-left corner (no margin).

        Parameters
        ----------
        img : ndarray, shape (H, W, 3), float32 0-1
            Rendered frame. Modified in-place.
        """
        if self._legend_rgba is None:
            return
        ov = self._legend_rgba
        oh, ow = ov.shape[:2]
        fh, fw = img.shape[:2]
        y0 = fh - oh
        if y0 < 0:
            y0 = 0
        bh = min(oh, fh - y0)
        bw = min(ow, fw)
        if bh <= 0 or bw <= 0:
            return
        alpha = ov[:bh, :bw, 3:4]
        rgb = ov[:bh, :bw, :3]
        region = img[y0:y0+bh, :bw]
        region[:] = region * (1 - alpha) + rgb * alpha

    def _render_help_text(self):
        """Pre-render paginated help overlays cached in ``_help_pages``.

        Each page matches the legend/minimap height and is positioned
        at the bottom of the frame.  Press H to cycle pages, then off.
        """
        all_sections = [
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
                ("U / Shift+U", "Cycle basemap fwd / back"),
                ("C", "Cycle colormap"),
                ("Y", "Cycle color stretch"),
                (", / .", "Overlay alpha"),
                ("R / Shift+R", "Resolution down / up"),
                ("Z / Shift+Z", "Vert. exag. down / up"),
                ("B", "Toggle TIN / Voxel"),
                ("T", "Toggle shadows"),
                ("Shift+T", "Cycle time of day"),
                ("Shift+E", "Toggle terrain"),
            ]),
            ("DATA LAYERS", [
                ("Shift+F", "FIRMS fire (7d)"),
                ("Shift+W", "Toggle wind"),
                ("Shift+N", "Toggle clouds + rain"),
                ("Shift+Y", "Toggle hydro flow"),
            ]),
            ("RENDERING", [
                ("0", "Toggle ambient occlusion"),
                ("Shift+H", "Prev help page"),
                ("Shift+G", "Cycle GI bounces (1-3)"),
                ("Shift+D", "Toggle AI denoiser"),
                ("9", "Toggle depth of field"),
                ("; / '", "DOF aperture down / up"),
                (": / \"", "DOF focal dist. down / up"),
            ]),
            ("GEOMETRY", [
                ("N", "Cycle geometry layer"),
                ("P", "Prev geometry in group"),
                ("Shift+C", "Cycle point cloud colors"),
                ("Right-Click", "Pick geometry"),
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
                ("H", "Cycle help pages"),
                ("X / Esc", "Exit"),
            ]),
        ]

        # Append info text as a section if provided
        if self._info_text:
            info_items = []
            for line in self._info_text.strip().splitlines():
                line = line.strip()
                if line:
                    info_items.append(("", line))
            if info_items:
                all_sections.append(("MODEL INFO", info_items))

        try:
            from PIL import Image, ImageDraw, ImageFont

            font_size = 12
            header_size = 13
            mono_path = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"
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
            hdr_h = header_size + 8
            section_gap = 6
            key_col_w = 105
            desc_col_w = 195
            col_w = key_col_w + desc_col_w
            pad_x = 14
            pad_y = 10

            # Colors
            bg_color = (15, 18, 24, 210)
            header_color = (180, 210, 255, 255)
            key_color = (255, 200, 100, 245)
            desc_color = (210, 215, 225, 220)
            accent_color = (90, 140, 220, 180)
            page_color = (120, 140, 170, 180)

            # Target height: match legend if available
            if self._legend_rgba is not None:
                target_h = self._legend_rgba.shape[0]
            else:
                target_h = 200
            usable_h = target_h - pad_y * 2

            def _section_h(section):
                _, items = section
                return hdr_h + 3 + len(items) * line_h

            # Pack sections into pages greedily
            pages = []
            current_page = []
            current_h = 0
            for sec in all_sections:
                sh = _section_h(sec)
                needed = sh + (section_gap if current_page else 0)
                if current_page and current_h + needed > usable_h:
                    pages.append(current_page)
                    current_page = [sec]
                    current_h = sh
                else:
                    current_h += needed
                    current_page.append(sec)
            if current_page:
                pages.append(current_page)

            n_pages = len(pages)
            img_w = pad_x * 2 + col_w

            self._help_pages = []
            for pi, page_sections in enumerate(pages):
                img = Image.new('RGBA', (img_w, target_h), (0, 0, 0, 0))
                draw = ImageDraw.Draw(img)
                draw.rectangle(
                    [0, 0, img_w - 1, target_h - 1], fill=bg_color)
                draw.rectangle(
                    [0, 0, img_w - 1, target_h - 1],
                    outline=(60, 70, 90, 140), width=1)

                y = pad_y
                for si, (title, items) in enumerate(page_sections):
                    if si > 0:
                        y += section_gap
                    draw.text((pad_x, y), title, fill=header_color,
                              font=font_header)
                    underline_y = y + header_size + 2
                    draw.line(
                        [(pad_x, underline_y),
                         (pad_x + col_w - 10, underline_y)],
                        fill=accent_color, width=1)
                    y = underline_y + 3
                    for key_text, desc_text in items:
                        if key_text:
                            draw.text((pad_x, y), key_text,
                                      fill=key_color, font=font_key)
                            draw.text((pad_x + key_col_w, y), desc_text,
                                      fill=desc_color, font=font)
                        else:
                            draw.text((pad_x, y), desc_text,
                                      fill=desc_color, font=font)
                        y += line_h

                # Page indicator (top-right)
                page_text = f"{pi + 1}/{n_pages}"
                bbox = font_header.getbbox(page_text)
                ptw = bbox[2] - bbox[0]
                draw.text((img_w - pad_x - ptw, pad_y), page_text,
                          fill=page_color, font=font_header)

                self._help_pages.append(
                    np.array(img, dtype=np.float32) / 255.0)
        except ImportError:
            self._help_pages = []

    def _blit_help_on_frame(self, img):
        """Alpha-composite the current help page onto the rendered frame.

        Positioned flush at the bottom, right after the legend.

        Parameters
        ----------
        img : ndarray, shape (H, W, 3), float32 0-1
            Rendered frame. Modified in-place.
        """
        idx = self._help_page_idx
        if idx < 0 or idx >= len(self._help_pages):
            return
        ht = self._help_pages[idx]
        hh, hw = ht.shape[:2]
        fh, fw = img.shape[:2]

        # X: right after legend, or flush left
        if self._legend_rgba is not None:
            x0 = self._legend_rgba.shape[1]
        else:
            x0 = 0

        # Flush bottom
        y0 = fh - hh
        if y0 < 0:
            y0 = 0

        bh = min(hh, fh - y0)
        bw = min(hw, fw - x0)
        if bh <= 0 or bw <= 0:
            return
        alpha = ht[:bh, :bw, 3:4]
        rgb = ht[:bh, :bw, :3]
        region = img[y0:y0+bh, x0:x0+bw]
        region[:] = region * (1 - alpha) + rgb * alpha

    def _drain_command_queue(self):
        """Drain REPL command queue — execute pending callables (thread-safe)."""
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

    def _present_if_dirty(self, frame_tex, ctx, vao, glfw, window, moderngl):
        """Upload frame to GL texture and present if the frame changed."""
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
        self._scene_diagonal = world_diag

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
        self._render_title_overlay()
        self._render_legend_overlay()
        self._render_help_text()

        # --- Initialize minimap ---
        self._minimap_bg_extent = None
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
                # Poll input before tick to avoid one-frame input lag
                glfw.poll_events()

                self._drain_command_queue()
                self._tick()
                self._present_if_dirty(frame_tex, ctx, vao, glfw, window,
                                       moderngl)

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
        if self._terrain_lod_manager is not None:
            self._terrain_lod_manager.shutdown()

        # Clean up terrain reload thread pool
        pool = self.terrain._terrain_reload_pool
        if pool is not None:
            pool.shutdown(wait=False)
            self.terrain._terrain_reload_pool = None

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
            overlay_layers: dict = None,
            color_stretch: str = 'linear',
            title: str = None,
            subtitle: str = None,
            legend: dict = None,
            tile_service=None,
            geometry_colors_builder=None,
            baked_meshes=None,
            subsample: int = 1,
            wind_data=None,
            weather_data=None,
            hydro_data=None,
            gtfs_data=None,
            accessor=None,
            terrain_loader=None,
            tile_data_fn=None,
            scene_zarr=None,
            ao_samples: int = 0,
            gi_bounces: int = 1,
            denoise: bool = False,
            fog_density: float = 0.0,
            fog_color: tuple = (0.7, 0.8, 0.9),
            colormap: str = None,
            sun_azimuth: float = None,
            sun_altitude: float = None,
            shadows: bool = None,
            ambient: float = None,
            minimap_style: str = None,
            minimap_layer: str = None,
            minimap_colors: dict = None,
            info_text: str = None,
            skirt: bool = True,
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
    hydro_data : dict or True, optional
        Hydrological flow data.  Pass ``True`` or ``{'enabled': False}``
        for lazy mode: MFD flow analysis is computed on GPU from the
        current terrain when Shift+Y is first pressed.  Or pass a dict
        with key ``'flow_accum'`` for pre-computed data.  Optional keys:
        ``'flow_dir_mfd'`` (xrspatial MFD fractions, shape (8,H,W)),
        ``'n_particles'``, ``'max_age'``, ``'trail_len'``, ``'speed'``,
        ``'accum_threshold'``, ``'color'``, ``'alpha'``, ``'dot_radius'``.
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
    - U/Shift+U: Cycle basemap forward/backward (none → satellite → osm)
    - N: Cycle geometry layer (none → all → groups)
    - P: Jump to previous geometry in current group
    - Shift+C: Cycle point cloud color mode (elevation/intensity/classification/rgb)
    - ,/.: Decrease/increase overlay alpha (transparency)
    - O: Place observer (for viewshed) at look-at point
    - Shift+O: Cycle drone mode (off → 3rd person → FPV → off)
    - V: Toggle viewshed overlay (teal glow shows visible terrain)
    - [/]: Decrease/increase observer height
    - R: Decrease terrain resolution (coarser, up to 8x subsample)
    - Shift+R: Increase terrain resolution (finer, down to 1x)
    - Z: Decrease vertical exaggeration
    - Shift+Z: Increase vertical exaggeration
    - Y: Cycle color stretch (linear, sqrt, cbrt, log)
    - T: Toggle shadows
    - 0: Toggle ambient occlusion (progressive)
    - Shift+G: Cycle GI bounces (1→2→3→1)
    - Shift+D: Toggle OptiX AI Denoiser
    - C: Cycle colormap
    - Shift+F: Fetch/toggle FIRMS fire layer (7d LANDSAT 30m)
    - Shift+W: Toggle wind particle animation
    - Shift+Y: Toggle hydro flow particle animation
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
        overlay_layers=overlay_layers,
        title=title,
        subtitle=subtitle,
        legend=legend,
        subsample=subsample,
        skirt=skirt,
    )
    viewer._geometry_colors_builder = geometry_colors_builder
    viewer._baked_meshes = baked_meshes or {}
    viewer._minimap_style = minimap_style
    viewer._minimap_layer = minimap_layer
    viewer._minimap_colors = minimap_colors
    viewer._info_text = info_text
    viewer._accessor = accessor
    viewer._terrain_loader = terrain_loader
    viewer._tile_data_fn = tile_data_fn
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

    # Weather overlay initialization
    if weather_data is not None:
        viewer._init_weather(weather_data)

    # Hydro flow initialization
    if hydro_data is not None:
        if hydro_data is True or 'flow_accum' not in hydro_data:
            # Lazy mode: compute MFD hydro from terrain on first enable
            viewer._hydro_lazy = True
            hydro_start_enabled = (
                hydro_data.get('enabled', False)
                if isinstance(hydro_data, dict) else False)
            if hydro_start_enabled:
                viewer._compute_hydro_from_terrain()
                viewer._hydro_enabled = True
                if 'stream_link' in viewer._overlay_layers:
                    viewer._overlay_as_water = True
        else:
            # Pre-computed hydro data provided
            hydro_start_enabled = hydro_data.get('enabled', True)
            flow_accum = hydro_data['flow_accum']
            hydro_opts = {k: v for k, v in hydro_data.items()
                          if k not in ('flow_accum', 'enabled')}
            viewer._init_hydro(flow_accum, **hydro_opts)
            # Re-register stream_link overlay with NaN + palette coloring
            if (viewer._hydro_stream_order_raw is not None
                    and 'stream_link' in viewer._overlay_layers):
                max_order = int(viewer._hydro_stream_order_raw.max())
                palette_lut = InteractiveViewer._build_stream_palette_lut(
                    max_order)
                sl_data = viewer._base_overlay_layers['stream_link']
                if hasattr(sl_data, 'get'):
                    sl_data = sl_data.get()
                sl_data = np.asarray(sl_data, dtype=np.float32)
                so_raw = viewer._hydro_stream_order_raw.astype(np.float32)
                sl_color = np.where(
                    (sl_data <= 0) | (so_raw <= 0),
                    np.float32(np.nan), so_raw)
                _add_overlay(viewer, 'stream_link', sl_color,
                             color_lut=palette_lut)
            if hydro_start_enabled:
                viewer._hydro_enabled = True
                if 'stream_link' in viewer._overlay_layers:
                    viewer._overlay_as_water = True
            else:
                viewer._hydro_enabled = False
                viewer._terrain_layer_idx = 0
                viewer._active_overlay_data = None
                viewer._overlay_as_water = False
                viewer._active_overlay_color_lut = None

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

    # Shared render params (fog, lighting, colormap)
    if fog_density > 0:
        viewer.fog_density = fog_density
    if fog_color != (0.7, 0.8, 0.9):
        viewer.fog_color = fog_color
    if colormap is not None:
        viewer.colormap = colormap
        if colormap in viewer.colormaps:
            viewer.colormap_idx = viewer.colormaps.index(colormap)
    if sun_azimuth is not None:
        viewer.sun_azimuth = sun_azimuth
    if sun_altitude is not None:
        viewer.sun_altitude = sun_altitude
    if shadows is not None:
        viewer.shadows = shadows
    if ambient is not None:
        viewer.ambient = ambient

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
