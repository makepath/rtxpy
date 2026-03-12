"""HydroManager: manages hydrological flow particle simulation.

Encapsulates flow field computation, particle lifecycle (spawn, advect,
respawn), GPU splatting, and streaming tile integration.  Replaces the
~850 lines of hydro code previously embedded in engine.py.

Streaming support
-----------------
When a ``TerrainLODManager`` is attached (via ``set_lod_manager``), the
flow field covers a *window* of tiles centred on the camera.  Particles
are stored in **global pixel coordinates** so their positions stay valid
across window shifts.  The advection kernel receives ``(win_r0, win_c0)``
offsets to translate global coords → local flow field indices.

The window is recomputed asynchronously (``ThreadPoolExecutor(1)``) when
the camera moves more than ~2 tiles from the window centre.
"""

from __future__ import annotations

import math
import time
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .terrain_lod import TerrainLODManager

try:
    import cupy as cp
    has_cupy = True
except ImportError:
    cp = None
    has_cupy = False


# Stream-order colour palette (shared with engine.py for overlay LUT)
STREAM_ORDER_PALETTE = np.array([
    [0.0,  0.0,  0.0 ],  # 0: unused
    [0.50, 0.80, 1.00],  # 1: pale sky blue (headwaters)
    [0.38, 0.68, 0.98],  # 2: light blue
    [0.28, 0.55, 0.95],  # 3: sky blue
    [0.18, 0.42, 0.90],  # 4: medium blue
    [0.10, 0.30, 0.85],  # 5: royal blue
    [0.06, 0.20, 0.78],  # 6: deep blue
    [0.03, 0.12, 0.70],  # 7: dark blue
    [0.01, 0.06, 0.60],  # 8: navy (major rivers)
], dtype=np.float32)


def build_stream_palette_lut(max_order):
    """Build a 256-entry color LUT for stream order overlay rendering."""
    lut = np.zeros((256, 3), dtype=np.float32)
    denom = max(max_order - 1, 1)
    for i in range(256):
        order = int(round(1 + (i / 255.0) * denom))
        order = max(1, min(8, order))
        lut[i] = STREAM_ORDER_PALETTE[order]
    return lut


def color_from_order(order_norm, raw_order=None):
    """Map stream order → (R, G, B) per particle."""
    if raw_order is not None:
        idx = np.clip(raw_order, 1, 8).astype(int)
        colors = STREAM_ORDER_PALETTE[idx].copy()
    else:
        colors = np.empty((len(order_norm), 3), dtype=np.float32)
        colors[:, 0] = 0.02 + order_norm * 0.43
        colors[:, 1] = 0.10 + order_norm * 0.65
        colors[:, 2] = 0.55 + order_norm * 0.40
    return np.clip(colors, 0.0, 1.0)


def radius_from_order(order_norm, raw_order=None):
    """Map stream order → radius (2–5) per particle."""
    if raw_order is not None:
        return np.clip(raw_order + 1, 2, 5).astype(np.int32)
    return np.clip(2 + (order_norm * 3).astype(np.int32),
                   2, 5).astype(np.int32)


class HydroManager:
    """Manages hydrological flow particle simulation and rendering.

    Parameters
    ----------
    hydro_state : HydroState
        The viewer's HydroState object (owns CPU-side particle arrays
        and GPU buffer references).
    """

    __slots__ = (
        '_state',
        # LOD integration
        '_lod_manager',
        '_tile_data_fn',
        '_crs_transform',
        # Flow field window (streaming)
        '_win_r0', '_win_c0',
        '_win_h', '_win_w',
        '_window_center_tr', '_window_center_tc',
        '_window_radius',
        '_window_future',
        '_window_executor',
        '_last_window_time',
        # Streaming stream_link overlay (pending for engine pickup)
        '_pending_stream_overlay',
        '_pending_overlay_bounds',
        # Terrain reference
        '_terrain_np',
        '_psx', '_psy',
    )

    def __init__(self, hydro_state):
        self._state = hydro_state
        self._lod_manager = None
        self._tile_data_fn = None
        self._crs_transform = None

        # Streaming window state
        self._win_r0 = 0.0
        self._win_c0 = 0.0
        self._win_h = 0
        self._win_w = 0
        self._window_center_tr = None
        self._window_center_tc = None
        self._window_radius = 5  # tiles in each direction
        self._window_future = None
        self._window_executor = None
        self._last_window_time = 0.0

        # Streaming stream_link overlay (set by _compute_windowed_flow)
        self._pending_stream_overlay = None
        self._pending_overlay_bounds = None  # (win_r0, win_c0, win_h, win_w)

        # Terrain ref (set during init)
        self._terrain_np = None
        self._psx = 1.0
        self._psy = 1.0

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------

    def set_lod_manager(self, mgr: 'TerrainLODManager'):
        """Attach LOD manager for streaming tile integration."""
        self._lod_manager = mgr

    def set_tile_data_fn(self, fn):
        """Set the tile data callback for streaming elevation fetches."""
        self._tile_data_fn = fn

    def set_crs_transform(self, x0, y0, dx, dy):
        """Store CRS origin and pixel spacing for coord conversion."""
        self._crs_transform = (float(x0), float(y0), float(dx), float(dy))

    def set_terrain_ref(self, terrain_np, psx, psy):
        """Store terrain array and pixel spacing for Z lookups."""
        self._terrain_np = terrain_np
        self._psx = psx
        self._psy = psy

    @property
    def streaming(self):
        """True if streaming tile mode is active."""
        return self._lod_manager is not None and self._tile_data_fn is not None

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def init_from_flow(self, flow_accum, terrain_data, psx, psy, **kwargs):
        """Initialize hydro particles from flow accumulation grid.

        This is the main entry point — equivalent to the old
        ``_init_hydro`` on InteractiveViewer.

        Parameters
        ----------
        flow_accum : array-like, shape (H, W)
            Flow accumulation grid.
        terrain_data : array-like, shape (H, W)
            Terrain elevation for slope computation.
        psx, psy : float
            Pixel spacing (world units per pixel).
        **kwargs
            Optional overrides: n_particles, max_age, trail_len, speed,
            accum_threshold, color, alpha, dot_radius, min_visible_age,
            stream_order, flow_dir_mfd, elevation.
        """
        st = self._state
        self._psx = psx
        self._psy = psy

        if hasattr(flow_accum, 'get'):
            flow_accum = flow_accum.get()
        flow_accum = np.asarray(flow_accum, dtype=np.float64)
        H, W = flow_accum.shape

        # Stream order grid (optional but strongly recommended)
        stream_order = kwargs.pop('stream_order', None)
        if stream_order is not None:
            if hasattr(stream_order, 'get'):
                stream_order = stream_order.get()
            stream_order = np.asarray(stream_order, dtype=np.float64)
            stream_order = np.nan_to_num(stream_order, nan=0.0)
        has_stream_order = stream_order is not None and (stream_order > 0).any()

        st.hydro_data = True

        # Apply optional overrides
        for key, attr, conv in [
            ('n_particles', 'hydro_n_particles', int),
            ('max_age', 'hydro_max_age', int),
            ('trail_len', 'hydro_trail_len', int),
            ('speed', 'hydro_speed', float),
            ('accum_threshold', 'hydro_accum_threshold', int),
            ('color', 'hydro_color', tuple),
            ('alpha', 'hydro_alpha', float),
            ('dot_radius', 'hydro_dot_radius', int),
            ('min_visible_age', 'hydro_min_visible_age', int),
        ]:
            if key in kwargs:
                val = conv(kwargs[key])
                setattr(st, attr, val)

        # MFD flow vectors
        flow_dir_mfd = kwargs.pop('flow_dir_mfd', None)

        sqrt2_inv = 1.0 / np.sqrt(2.0)
        _dir_dr = np.array([0.0, sqrt2_inv, 1.0, sqrt2_inv,
                            0.0, -sqrt2_inv, -1.0, -sqrt2_inv])
        _dir_dc = np.array([1.0, sqrt2_inv, 0.0, -sqrt2_inv,
                            -1.0, -sqrt2_inv, 0.0, sqrt2_inv])

        if flow_dir_mfd is not None:
            if hasattr(flow_dir_mfd, 'get'):
                flow_dir_mfd = flow_dir_mfd.get()
            frac = np.asarray(flow_dir_mfd, dtype=np.float64)
            frac = np.nan_to_num(frac, nan=0.0)
            flow_v = np.tensordot(_dir_dr, frac, axes=([0], [0]))
            flow_u = np.tensordot(_dir_dc, frac, axes=([0], [0]))
            valid_flow = np.any(frac > 0, axis=0)
            del frac
        else:
            elevation = kwargs.pop('elevation', None)
            if elevation is not None:
                if hasattr(elevation, 'get'):
                    elevation = elevation.get()
                elevation = np.asarray(elevation, dtype=np.float64)
            else:
                if hasattr(terrain_data, 'get'):
                    elevation = terrain_data.get()
                else:
                    elevation = np.asarray(terrain_data, dtype=np.float64)
                elevation = elevation.astype(np.float64)

            nan_mask_elev = np.isnan(elevation)
            elev_clean = np.where(nan_mask_elev, 1e10, elevation)

            sqrt2 = np.sqrt(2.0)
            mfd_p = 1.1
            flow_u = np.zeros((H, W), dtype=np.float64)
            flow_v = np.zeros((H, W), dtype=np.float64)

            _nb_offsets = [
                (-1, -1, sqrt2), (-1,  0, 1.0), (-1,  1, sqrt2),
                ( 0, -1, 1.0),                  ( 0,  1, 1.0),
                ( 1, -1, sqrt2), ( 1,  0, 1.0), ( 1,  1, sqrt2),
            ]
            _nb_dr = np.array([-sqrt2_inv, -1.0, -sqrt2_inv,
                                0.0, 0.0,
                                sqrt2_inv, 1.0, sqrt2_inv])
            _nb_dc = np.array([-sqrt2_inv, 0.0, sqrt2_inv,
                               -1.0, 1.0,
                               -sqrt2_inv, 0.0, sqrt2_inv])

            for k, (dr, dc, dist) in enumerate(_nb_offsets):
                cr = slice(max(0, -dr), H - max(0, dr))
                cc = slice(max(0, -dc), W - max(0, dc))
                nr = slice(max(0, -dr) + dr, H - max(0, dr) + dr)
                nc = slice(max(0, -dc) + dc, W - max(0, dc) + dc)
                drop = elev_clean[cr, cc] - elev_clean[nr, nc]
                slope = np.maximum(drop / dist, 0.0)
                weight = slope ** mfd_p
                flow_v[cr, cc] += weight * _nb_dr[k]
                flow_u[cr, cc] += weight * _nb_dc[k]

            flow_u[nan_mask_elev] = 0.0
            flow_v[nan_mask_elev] = 0.0

        # Normalize to unit vectors
        mag = np.sqrt(flow_u**2 + flow_v**2)
        valid_flow = mag > 0
        flow_u[valid_flow] /= mag[valid_flow]
        flow_v[valid_flow] /= mag[valid_flow]

        st.hydro_flow_u_px = flow_u.astype(np.float32)
        st.hydro_flow_v_px = flow_v.astype(np.float32)

        # Normalize accumulation
        fa_clipped = np.clip(flow_accum, 1, None)
        log_fa = np.log10(fa_clipped)
        threshold = np.log10(max(st.hydro_accum_threshold, 1))
        log_max = log_fa.max()
        if log_max > threshold:
            accum_norm = np.clip(
                (log_fa - threshold) / (log_max - threshold), 0, 1)
        else:
            accum_norm = np.zeros_like(log_fa)
        st.hydro_flow_accum_norm = accum_norm.astype(np.float32)

        # Store stream order grids
        if has_stream_order:
            max_order = stream_order.max()
            so_norm = (stream_order / max(max_order, 1)).astype(np.float32)
            st.hydro_stream_order = so_norm
            st.hydro_stream_order_raw = stream_order.astype(np.int32)
            print(f"  Stream order: max {int(max_order)}, "
                  f"{int((stream_order > 0).sum())} stream cells")
        else:
            st.hydro_stream_order = None
            st.hydro_stream_order_raw = None

        # Stream link grid
        stream_link_grid = kwargs.pop('stream_link', None)
        if stream_link_grid is not None:
            if hasattr(stream_link_grid, 'get'):
                stream_link_grid = stream_link_grid.get()
            st.hydro_stream_link = np.nan_to_num(
                np.asarray(stream_link_grid, dtype=np.float64), nan=0.0
            ).astype(np.int32)
        else:
            st.hydro_stream_link = None

        # Build spawn probabilities
        if has_stream_order:
            spawn_weights = np.where(stream_order > 0,
                                     np.sqrt(stream_order), 0.0)
            spawn_weights[~valid_flow] = 0.0
        else:
            spawn_weights = accum_norm.copy()
            spawn_weights[~valid_flow] = 0.0

        flat_weights = spawn_weights.ravel()
        valid_mask = flat_weights > 0
        valid_indices = np.nonzero(valid_mask)[0]
        if len(valid_indices) > 0:
            valid_probs = flat_weights[valid_indices].astype(np.float64)
            valid_probs /= valid_probs.sum()
        else:
            valid_flow_flat = valid_flow.ravel()
            valid_indices = np.nonzero(valid_flow_flat)[0]
            if len(valid_indices) > 0:
                valid_probs = np.ones(len(valid_indices), dtype=np.float64)
                valid_probs /= valid_probs.sum()
            else:
                valid_indices = np.arange(H * W)
                valid_probs = np.ones(H * W, dtype=np.float64) / (H * W)
        st.hydro_spawn_indices = valid_indices
        st.hydro_spawn_valid_probs = valid_probs

        # Spawn initial particles
        N = st.hydro_n_particles
        chosen = np.random.choice(len(valid_indices), N, p=valid_probs)
        indices = valid_indices[chosen]
        rows = (indices // W).astype(np.float32) + \
            np.random.uniform(-0.5, 0.5, N).astype(np.float32)
        cols = (indices % W).astype(np.float32) + \
            np.random.uniform(-0.5, 0.5, N).astype(np.float32)
        rows = np.clip(rows, 0, H - 1)
        cols = np.clip(cols, 0, W - 1)

        st.hydro_particles = np.column_stack([rows, cols]).astype(np.float32)
        st.hydro_ages = np.random.randint(0, st.hydro_max_age, N).astype(np.int32)
        st.hydro_lifetimes = np.random.randint(
            st.hydro_max_age // 2, st.hydro_max_age, N).astype(np.int32)
        st.hydro_trails = np.zeros(
            (N, st.hydro_trail_len, 2), dtype=np.float32)
        for t in range(st.hydro_trail_len):
            st.hydro_trails[:, t, :] = st.hydro_particles

        # Compute terrain slope magnitude
        if hasattr(terrain_data, 'get'):
            elev = terrain_data.get().astype(np.float64)
        else:
            elev = np.asarray(terrain_data, dtype=np.float64)
        grad_row, grad_col = np.gradient(np.nan_to_num(elev, nan=0.0))
        slope_mag = np.sqrt(grad_row**2 + grad_col**2).astype(np.float32)
        p95 = np.percentile(slope_mag[slope_mag > 0], 95) \
            if (slope_mag > 0).any() else 1.0
        slope_norm = np.clip(
            slope_mag / max(p95, 1e-6), 0, 1).astype(np.float32)
        st.hydro_slope_mag = slope_norm

        # Per-particle visual properties
        r_idx = np.clip(np.floor(rows).astype(int), 0, H - 1)
        c_idx = np.clip(np.floor(cols).astype(int), 0, W - 1)
        if has_stream_order:
            order_val = so_norm[r_idx, c_idx].astype(np.float32)
            raw_order = st.hydro_stream_order_raw[r_idx, c_idx]
        else:
            order_val = accum_norm[r_idx, c_idx].astype(np.float32)
            raw_order = None
        st.hydro_particle_accum = order_val
        st.hydro_particle_raw_order = raw_order
        st.hydro_particle_colors = color_from_order(
            order_val, raw_order=raw_order)
        st.hydro_particle_radii = radius_from_order(
            order_val, raw_order=raw_order)

        # Min render distance and depth-scaled alpha reference
        world_diag = np.sqrt((W * psx)**2 + (H * psy)**2)
        st.hydro_min_depth = 1.0
        st.hydro_max_depth = world_diag * 0.35
        st.hydro_ref_depth = world_diag * 0.15

        # Set streaming window to cover full initial terrain
        self._win_r0 = 0.0
        self._win_c0 = 0.0
        self._win_h = H
        self._win_w = W

        # Upload to GPU
        self._upload_to_gpu(N)

        print(f"  Hydro flow initialized on {H}x{W} grid "
              f"({N} particles, threshold={st.hydro_accum_threshold})")

    def compute_from_terrain(self, raster):
        """Compute hydrological flow from terrain elevation on GPU.

        Uses xrspatial MFD functions.  Called lazily on first hydro
        enable or after terrain reload.

        Parameters
        ----------
        raster : xarray.DataArray
            Terrain elevation (may be CuPy-backed).

        Returns
        -------
        dict or None
            ``{'stream_order_raw': ..., 'stream_link': ...}`` on success,
            or None on failure.  Caller uses these to register overlays.
        """
        try:
            from xrspatial import fill as _fill
            from xrspatial import flow_direction_mfd as _fd_mfd
            from xrspatial import flow_accumulation_mfd as _fa_mfd
            from xrspatial import stream_order_mfd as _so_mfd
            from xrspatial import stream_link_mfd as _sl_mfd
        except ImportError:
            print("Hydro requires xrspatial: pip install xrspatial")
            return None
        try:
            from scipy.ndimage import uniform_filter
        except ImportError:
            print("Hydro requires scipy: pip install scipy")
            return None

        print("Computing hydrological flow on GPU...")
        data = raster.data
        is_cupy = hasattr(data, 'get')

        # Condition DEM
        elev_np = data.get() if is_cupy else np.array(data)
        elev_np = elev_np.astype(np.float32)
        ocean = (elev_np == 0.0) | np.isnan(elev_np)
        elev_np[ocean] = -100.0

        smoothed = uniform_filter(elev_np, size=15, mode='nearest')
        smoothed[ocean] = -100.0
        del elev_np

        if is_cupy:
            sm = cp.asarray(smoothed)
        else:
            sm = smoothed
        del smoothed

        filled = _fill(raster.copy(data=sm))
        fill_depth = filled.data - sm
        resolved = filled.data + fill_depth * 0.01
        del filled, fill_depth, sm

        if is_cupy:
            cp.random.seed(0)
            resolved += cp.random.uniform(
                0, 0.0001, resolved.shape, dtype=cp.float32)
            resolved[cp.asarray(ocean)] = -100.0
        else:
            np.random.seed(0)
            resolved += np.random.uniform(
                0, 0.0001, resolved.shape).astype(np.float32)
            resolved[ocean] = -100.0

        resolved_da = raster.copy(data=resolved)
        fd_mfd = _fd_mfd(resolved_da, boundary='nearest')
        fa_mfd = _fa_mfd(fd_mfd)
        del resolved_da, resolved

        so = _so_mfd(fd_mfd, fa_mfd, threshold=50)
        sl = _sl_mfd(fd_mfd, fa_mfd, threshold=50)

        fa_out = fa_mfd.data
        fd_out = fd_mfd.data
        so_out = so.data
        sl_out = sl.data
        xp = cp if is_cupy else np
        if is_cupy:
            ocean_gpu = cp.asarray(ocean)
            fa_out[ocean_gpu] = cp.nan
            fd_out[:, ocean_gpu] = cp.nan
            so_out[ocean_gpu] = cp.nan
            sl_out[ocean_gpu] = cp.nan
        else:
            fa_out[ocean] = np.nan
            fd_out[:, ocean] = np.nan
            so_out[ocean] = np.nan
            sl_out[ocean] = np.nan

        sl_clean = xp.nan_to_num(sl_out, nan=0.0).astype(xp.float32)

        terrain_data = raster.data
        self.init_from_flow(
            fa_out,
            terrain_data,
            self._psx, self._psy,
            flow_dir_mfd=fd_out,
            stream_order=so_out,
            stream_link=sl_clean,
        )

        H, W = raster.shape
        print(f"  Hydro flow computed on GPU ({H}x{W} grid, MFD)")

        # Return overlay data for the engine to register
        result = {}
        st = self._state
        if st.hydro_stream_order_raw is not None:
            result['stream_order_raw'] = st.hydro_stream_order_raw
            max_order = int(st.hydro_stream_order_raw.max())
            result['palette_lut'] = build_stream_palette_lut(max_order)

            sl_np = sl_clean.get() if is_cupy else np.asarray(sl_clean)
            so_raw = st.hydro_stream_order_raw.astype(np.float32)
            sl_color = np.where(
                (sl_np <= 0) | (so_raw <= 0),
                np.float32(np.nan), so_raw)
            result['stream_link_overlay'] = sl_color
            result['palette_lut'] = build_stream_palette_lut(max_order)

        return result

    def _upload_to_gpu(self, N):
        """Upload all particle + grid arrays to GPU."""
        if not has_cupy:
            return
        st = self._state
        st.d_hydro_particles = cp.asarray(st.hydro_particles)
        st.d_hydro_ages = cp.asarray(st.hydro_ages)
        st.d_hydro_lifetimes = cp.asarray(st.hydro_lifetimes)
        st.d_hydro_trails = cp.asarray(st.hydro_trails)
        st.d_hydro_colors = cp.asarray(st.hydro_particle_colors)
        st.d_hydro_radii = cp.asarray(st.hydro_particle_radii)
        st.d_hydro_particle_accum = cp.asarray(st.hydro_particle_accum)
        if st.hydro_particle_raw_order is not None:
            st.d_hydro_particle_raw_order = cp.asarray(
                st.hydro_particle_raw_order)
        else:
            st.d_hydro_particle_raw_order = cp.zeros(N, dtype=cp.int32)
        st.d_hydro_flow_u = cp.asarray(st.hydro_flow_u_px)
        st.d_hydro_flow_v = cp.asarray(st.hydro_flow_v_px)
        if st.hydro_slope_mag is not None:
            st.d_hydro_slope_mag = cp.asarray(st.hydro_slope_mag)
        else:
            st.d_hydro_slope_mag = cp.empty((0, 0), dtype=cp.float32)
        if st.hydro_stream_order is not None:
            st.d_hydro_stream_order = cp.asarray(st.hydro_stream_order)
        else:
            st.d_hydro_stream_order = cp.empty((0, 0), dtype=cp.float32)
        if st.hydro_stream_order_raw is not None:
            st.d_hydro_stream_order_raw = cp.asarray(
                st.hydro_stream_order_raw)
        else:
            st.d_hydro_stream_order_raw = cp.empty((0, 0), dtype=cp.int32)
        st.d_hydro_accum_norm = cp.asarray(st.hydro_flow_accum_norm)
        st.d_hydro_palette = cp.asarray(STREAM_ORDER_PALETTE)
        st.d_hydro_respawn_flags = cp.zeros(N, dtype=cp.int32)

    # ------------------------------------------------------------------
    # Per-tick update
    # ------------------------------------------------------------------

    def update_particles(self, dt_scale=1.0):
        """Advect hydro particles one tick on GPU.

        Two-pass: GPU advection kernel, then CPU respawn batch.
        """
        if not has_cupy:
            return
        st = self._state
        if st.d_hydro_flow_u is None or st.d_hydro_particles is None:
            return

        from ._hydro_kernels import hydro_advect_kernel, hydro_respawn_kernel

        N = st.d_hydro_particles.shape[0]

        has_so = 1 if (st.hydro_stream_order is not None) else 0
        has_slope = 1 if (st.hydro_slope_mag is not None) else 0
        has_raw = 1 if (st.hydro_stream_order_raw is not None) else 0

        speed = float(st.hydro_speed)
        trail_len = int(st.hydro_trail_len)
        rng_base = np.random.randint(0, 2**62)

        threadsperblock = 256
        blockspergrid = (N + threadsperblock - 1) // threadsperblock

        hydro_advect_kernel[blockspergrid, threadsperblock](
            st.d_hydro_particles,
            st.d_hydro_ages,
            st.d_hydro_lifetimes,
            st.d_hydro_trails,
            st.d_hydro_particle_accum,
            st.d_hydro_particle_raw_order,
            st.d_hydro_colors,
            st.d_hydro_radii,
            st.d_hydro_flow_u,
            st.d_hydro_flow_v,
            st.d_hydro_slope_mag,
            st.d_hydro_stream_order,
            st.d_hydro_stream_order_raw,
            st.d_hydro_accum_norm,
            st.d_hydro_palette,
            st.d_hydro_respawn_flags,
            speed, float(dt_scale), trail_len,
            has_so, has_slope, has_raw,
            float(self._win_r0), float(self._win_c0),
            rng_base,
        )

        # Read back respawn flags and handle respawns on CPU
        respawn_flags = st.d_hydro_respawn_flags.get()
        respawn_idx = np.nonzero(respawn_flags)[0]
        n_respawn = len(respawn_idx)

        if n_respawn > 0:
            H, W = st.d_hydro_flow_u.shape
            chosen = np.random.choice(
                len(st.hydro_spawn_indices), n_respawn,
                p=st.hydro_spawn_valid_probs)
            flat_indices = st.hydro_spawn_indices[chosen]
            # Spawn positions are in flow-field-local coords;
            # convert to global by adding window offset
            spawn_rows = (flat_indices // W).astype(np.float32) + \
                np.random.uniform(-0.5, 0.5, n_respawn).astype(np.float32) + \
                float(self._win_r0)
            spawn_cols = (flat_indices % W).astype(np.float32) + \
                np.random.uniform(-0.5, 0.5, n_respawn).astype(np.float32) + \
                float(self._win_c0)
            spawn_rows = np.clip(spawn_rows,
                                 self._win_r0, self._win_r0 + H - 1)
            spawn_cols = np.clip(spawn_cols,
                                 self._win_c0, self._win_c0 + W - 1)
            new_lifetimes = np.random.randint(
                st.hydro_max_age // 2, st.hydro_max_age,
                n_respawn).astype(np.int32)

            d_respawn_idx = cp.asarray(respawn_idx.astype(np.int32))
            d_spawn_rows = cp.asarray(spawn_rows)
            d_spawn_cols = cp.asarray(spawn_cols)
            d_new_lifetimes = cp.asarray(new_lifetimes)

            blocks_r = (n_respawn + threadsperblock - 1) // threadsperblock
            hydro_respawn_kernel[blocks_r, threadsperblock](
                st.d_hydro_particles,
                st.d_hydro_ages,
                st.d_hydro_lifetimes,
                st.d_hydro_trails,
                st.d_hydro_particle_accum,
                st.d_hydro_particle_raw_order,
                st.d_hydro_colors,
                st.d_hydro_radii,
                d_respawn_idx,
                d_spawn_rows,
                d_spawn_cols,
                d_new_lifetimes,
                st.d_hydro_stream_order,
                st.d_hydro_stream_order_raw,
                st.d_hydro_accum_norm,
                st.d_hydro_palette,
                trail_len, has_so, has_raw,
                float(self._win_r0), float(self._win_c0),
            )

    # ------------------------------------------------------------------
    # GPU splatting
    # ------------------------------------------------------------------

    def splat_gpu(self, d_frame, camera_pos, look_at, fov, ve,
                  subsample_factor, terrain_gpu, depth_t=None):
        """Project and splat hydro particles on GPU.

        Parameters
        ----------
        d_frame : cupy.ndarray, shape (H, W, 3)
            GPU frame buffer (float32 0-1). Modified in-place.
        camera_pos : tuple of 3 floats
        look_at : tuple of 3 floats
        fov : float
            Vertical field of view in degrees.
        ve : float
            Vertical exaggeration.
        subsample_factor : float
        terrain_gpu : cupy.ndarray, shape (tH, tW)
            GPU terrain for Z lookup.
        depth_t : cupy.ndarray or None
            Depth buffer for occlusion culling.
        """
        st = self._state
        if st.d_hydro_particles is None or st.d_hydro_trails is None:
            return
        if not has_cupy:
            return

        from ._hydro_kernels import hydro_splat_kernel
        from ..analysis.render import _compute_camera_basis

        N = st.d_hydro_particles.shape[0]
        trail_len = st.hydro_trail_len
        total = N * trail_len

        forward, right, cam_up = _compute_camera_basis(
            tuple(camera_pos), tuple(look_at), (0, 0, 1),
        )
        fov_scale = math.tan(math.radians(fov) / 2.0)
        aspect_ratio = d_frame.shape[1] / d_frame.shape[0]

        d_trails_flat = st.d_hydro_trails.reshape(-1, 2)

        if depth_t is None:
            depth_t = cp.empty((0, 0), dtype=cp.float32)

        threadsperblock = 256
        blockspergrid = (total + threadsperblock - 1) // threadsperblock

        hydro_splat_kernel[blockspergrid, threadsperblock](
            d_trails_flat,
            st.d_hydro_ages,
            st.d_hydro_lifetimes,
            st.d_hydro_colors,
            st.d_hydro_radii,
            trail_len,
            float(st.hydro_alpha),
            int(st.hydro_min_visible_age),
            float(st.hydro_ref_depth),
            terrain_gpu,
            depth_t,
            d_frame,
            float(camera_pos[0]), float(camera_pos[1]),
            float(camera_pos[2]),
            float(forward[0]), float(forward[1]), float(forward[2]),
            float(right[0]), float(right[1]), float(right[2]),
            float(cam_up[0]), float(cam_up[1]), float(cam_up[2]),
            float(fov_scale), float(aspect_ratio),
            float(self._psx),
            float(self._psy),
            float(ve),
            float(subsample_factor),
            float(st.hydro_min_depth),
            float(st.hydro_max_depth),
        )

        cp.clip(d_frame, 0, 1, out=d_frame)

    # ------------------------------------------------------------------
    # Streaming window management
    # ------------------------------------------------------------------

    def update_streaming_window(self, camera_row, camera_col):
        """Check if the flow field window needs to shift for streaming.

        Called each tick when streaming is active.  If the camera has
        moved more than 2 tiles from the window centre and enough time
        has passed, triggers an async recompute of the flow field over
        a new tile window.

        Parameters
        ----------
        camera_row, camera_col : float
            Camera position in pixel (row, col) coordinates.
        """
        if not self.streaming:
            return

        mgr = self._lod_manager
        ts = mgr._tile_size

        cam_tr = int(camera_row / ts)
        cam_tc = int(camera_col / ts)

        # Check if we need to shift
        if self._window_center_tr is not None:
            dr = abs(cam_tr - self._window_center_tr)
            dc = abs(cam_tc - self._window_center_tc)
            if dr <= 2 and dc <= 2:
                return  # still within comfort zone

        # Don't start a new compute if one is in flight
        if self._window_future is not None and not self._window_future.done():
            return

        # Throttle: at most every 3 seconds
        now = time.monotonic()
        if now - self._last_window_time < 3.0:
            return

        # Collect the result of any previous future
        if self._window_future is not None and self._window_future.done():
            self._apply_window_result(self._window_future.result())
            self._window_future = None

        # Submit async recompute
        self._last_window_time = now
        radius = self._window_radius
        r0_tile = cam_tr - radius
        c0_tile = cam_tc - radius
        r1_tile = cam_tr + radius + 1
        c1_tile = cam_tc + radius + 1

        # Pixel bounds of the window
        win_r0 = r0_tile * ts
        win_c0 = c0_tile * ts
        win_r1 = r1_tile * ts
        win_c1 = c1_tile * ts

        if self._window_executor is None:
            self._window_executor = ThreadPoolExecutor(max_workers=1)

        self._window_future = self._window_executor.submit(
            self._compute_windowed_flow,
            win_r0, win_c0, win_r1, win_c1,
            cam_tr, cam_tc,
        )

    def check_streaming_result(self):
        """Poll for completed async window recompute.  Call each tick."""
        if self._window_future is not None and self._window_future.done():
            try:
                result = self._window_future.result()
                if result is not None:
                    self._apply_window_result(result)
            except Exception as e:
                print(f"Hydro streaming window error: {e}")
            self._window_future = None

    def pop_streaming_overlay(self):
        """Return and clear pending streaming stream overlay.

        Returns ``(overlay, win_r0, win_c0)`` or ``(None, 0, 0)``.
        """
        ov = self._pending_stream_overlay
        bounds = self._pending_overlay_bounds
        self._pending_stream_overlay = None
        self._pending_overlay_bounds = None
        if ov is None or bounds is None:
            return None, 0, 0
        return ov, bounds[0], bounds[1]

    @staticmethod
    def _compute_stream_overlay(elevation, threshold=50):
        """Compute stream overlay via D8 flow accumulation (CPU).

        Subsamples large grids for performance, then upsamples result.
        Returns ``(H, W)`` float32 array: stream order values (1-8)
        where streams exist, NaN elsewhere.
        """
        H, W = elevation.shape

        # Subsample to keep computation under ~1M cells
        max_cells = 1024
        subsample = max(1, max(H, W) // max_cells)
        if subsample > 1:
            elev_s = elevation[::subsample, ::subsample].copy()
        else:
            elev_s = elevation
        sH, sW = elev_s.shape
        N = sH * sW

        elev_clean = np.nan_to_num(elev_s, nan=1e10).astype(np.float64)
        flat_elev = elev_clean.ravel()
        valid_cell = flat_elev < 1e9

        # --- D8 flow direction (vectorized) ---
        idx_grid = np.arange(N, dtype=np.int64).reshape(sH, sW)
        target = np.full(N, -1, dtype=np.int64)
        max_slope = np.full((sH, sW), 0.0, dtype=np.float64)

        sqrt2 = np.sqrt(2.0)
        nb_offsets = [
            (-1, -1, sqrt2), (-1, 0, 1.0), (-1, 1, sqrt2),
            (0, -1, 1.0),                   (0, 1, 1.0),
            (1, -1, sqrt2),  (1, 0, 1.0),  (1, 1, sqrt2),
        ]
        for dr, dc, dist in nb_offsets:
            rs = slice(max(0, -dr), sH - max(0, dr))
            cs = slice(max(0, -dc), sW - max(0, dc))
            rn = slice(max(0, -dr) + dr, sH - max(0, dr) + dr)
            cn = slice(max(0, -dc) + dc, sW - max(0, dc) + dc)
            drop = elev_clean[rs, cs] - elev_clean[rn, cn]
            slope = drop / dist
            better = slope > max_slope[rs, cs]
            if better.any():
                src = idx_grid[rs, cs][better].ravel()
                dst = idx_grid[rn, cn][better].ravel()
                target[src] = dst
                max_slope[rs, cs] = np.maximum(max_slope[rs, cs], slope)

        # --- Flow accumulation: propagate high→low ---
        order = np.argsort(-flat_elev)
        accum = np.ones(N, dtype=np.float64)
        for idx in order:
            if not valid_cell[idx]:
                continue
            tgt = target[idx]
            if tgt >= 0:
                accum[tgt] += accum[idx]

        accum_2d = accum.reshape(sH, sW)

        # --- Detect streams and map to stream order ---
        is_stream = accum_2d > threshold
        if not is_stream.any():
            return np.full((H, W), np.nan, dtype=np.float32)

        stream_vals = accum_2d[is_stream]
        log_vals = np.log10(np.maximum(stream_vals, 1.0))
        log_min = np.log10(max(threshold, 1.0))
        log_max = log_vals.max()
        log_range = max(log_max - log_min, 1.0)
        order_vals = 1.0 + (log_vals - log_min) / log_range * 7.0
        order_vals = np.clip(order_vals, 1.0, 8.0).astype(np.float32)

        overlay_s = np.full((sH, sW), np.nan, dtype=np.float32)
        overlay_s[is_stream] = order_vals

        # Upsample back to full resolution
        if subsample > 1:
            from PIL import Image
            # Nearest-neighbor for categorical data
            img = Image.fromarray(overlay_s)
            overlay = np.array(
                img.resize((W, H), Image.NEAREST), dtype=np.float32)
        else:
            overlay = overlay_s

        return overlay

    def _compute_windowed_flow(self, win_r0, win_c0, win_r1, win_c1,
                               center_tr, center_tc):
        """Build flow field for the given pixel window (runs off main thread).

        Assembles elevation from:
        1. Initial terrain (for in-bounds pixels)
        2. LOD tile cache (for cached streaming tiles)
        3. tile_data_fn (for uncached areas)

        Then computes MFD flow vectors and spawn probabilities.

        Returns a dict with all the arrays needed to update GPU state,
        or None on failure.
        """
        mgr = self._lod_manager
        if mgr is None:
            return None

        win_h = win_r1 - win_r0
        win_c = win_c1 - win_c0
        if win_h <= 0 or win_c <= 0:
            return None

        # Assemble elevation array for the window
        elevation = np.full((win_h, win_c), np.nan, dtype=np.float32)

        # 1. Fill from initial terrain where available
        terrain_np = mgr._terrain_np
        t_H, t_W = terrain_np.shape
        # Overlap region between window and initial terrain
        src_r0 = max(0, -win_r0)
        src_c0 = max(0, -win_c0)
        src_r1 = min(win_h, t_H - win_r0)
        src_c1 = min(win_c, t_W - win_c0)
        ter_r0 = max(0, win_r0)
        ter_c0 = max(0, win_c0)
        if src_r1 > src_r0 and src_c1 > src_c0:
            ter_r1 = ter_r0 + (src_r1 - src_r0)
            ter_c1 = ter_c0 + (src_c1 - src_c0)
            elevation[src_r0:src_r1, src_c0:src_c1] = \
                terrain_np[ter_r0:ter_r1, ter_c0:ter_c1]

        # 2. Fill remaining NaN areas from tile_data_fn
        if self._tile_data_fn is not None and np.isnan(elevation).any():
            ts = mgr._tile_size
            crs_tf = self._crs_transform
            if crs_tf is not None:
                crs_x0, crs_y0, crs_dx, crs_dy = crs_tf
                psx = abs(mgr._psx)
                psy = abs(mgr._psy)

                # Iterate tile-sized blocks in the window
                for br in range(0, win_h, ts):
                    for bc in range(0, win_c, ts):
                        br1 = min(br + ts, win_h)
                        bc1 = min(bc + ts, win_c)
                        block = elevation[br:br1, bc:bc1]
                        if not np.isnan(block).any():
                            continue

                        # Global pixel coords of this block
                        gr0 = win_r0 + br
                        gc0 = win_c0 + bc
                        gr1 = win_r0 + br1
                        gc1 = win_c0 + bc1

                        # Convert to CRS
                        x_min = crs_x0 + gc0 * crs_dx
                        x_max = crs_x0 + gc1 * crs_dx
                        y_min = crs_y0 + gr0 * crs_dy
                        y_max = crs_y0 + gr1 * crs_dy

                        if x_min > x_max:
                            x_min, x_max = x_max, x_min
                        if y_min > y_max:
                            y_min, y_max = y_max, y_min

                        try:
                            tile_data = self._tile_data_fn(
                                x_min, y_min, x_max, y_max,
                                max(br1 - br, bc1 - bc))
                        except Exception:
                            continue

                        if tile_data is not None:
                            td = np.asarray(tile_data, dtype=np.float32)
                            # Resize to match block dimensions
                            if td.shape != (br1 - br, bc1 - bc):
                                from PIL import Image
                                img = Image.fromarray(td)
                                img = img.resize(
                                    (bc1 - bc, br1 - br),
                                    Image.BILINEAR)
                                td = np.array(img, dtype=np.float32)
                            # Only fill NaN cells
                            nan_mask = np.isnan(elevation[br:br1, bc:bc1])
                            elevation[br:br1, bc:bc1] = np.where(
                                nan_mask, td, elevation[br:br1, bc:bc1])

        # If still mostly NaN, bail
        valid_frac = np.isfinite(elevation).mean()
        if valid_frac < 0.1:
            return None

        # Fill remaining NaN with nearest-neighbor
        for _ in range(min(50, max(win_h, win_c))):
            still_nan = np.isnan(elevation)
            if not still_nan.any():
                break
            padded = np.pad(elevation, 1, mode='edge')
            neighbors = np.stack([
                padded[:-2, 1:-1], padded[2:, 1:-1],
                padded[1:-1, :-2], padded[1:-1, 2:],
            ], axis=0)
            with np.errstate(all='ignore'):
                fill_vals = np.nanmean(neighbors, axis=0)
            elevation = np.where(
                still_nan & np.isfinite(fill_vals), fill_vals, elevation)

        # Compute stream overlay for streaming tiles
        try:
            stream_overlay = self._compute_stream_overlay(elevation)
        except Exception:
            stream_overlay = None

        # Compute MFD flow from elevation
        nan_mask = np.isnan(elevation)
        elev_clean = np.where(nan_mask, 1e10, elevation).astype(np.float64)

        sqrt2 = np.sqrt(2.0)
        sqrt2_inv = 1.0 / sqrt2
        mfd_p = 1.1
        flow_u = np.zeros((win_h, win_c), dtype=np.float64)
        flow_v = np.zeros((win_h, win_c), dtype=np.float64)

        _nb_offsets = [
            (-1, -1, sqrt2), (-1,  0, 1.0), (-1,  1, sqrt2),
            ( 0, -1, 1.0),                  ( 0,  1, 1.0),
            ( 1, -1, sqrt2), ( 1,  0, 1.0), ( 1,  1, sqrt2),
        ]
        _nb_dr = np.array([-sqrt2_inv, -1.0, -sqrt2_inv,
                            0.0, 0.0,
                            sqrt2_inv, 1.0, sqrt2_inv])
        _nb_dc = np.array([-sqrt2_inv, 0.0, sqrt2_inv,
                           -1.0, 1.0,
                           -sqrt2_inv, 0.0, sqrt2_inv])

        for k, (dr, dc, dist) in enumerate(_nb_offsets):
            cr = slice(max(0, -dr), win_h - max(0, dr))
            cc = slice(max(0, -dc), win_c - max(0, dc))
            nr = slice(max(0, -dr) + dr, win_h - max(0, dr) + dr)
            nc = slice(max(0, -dc) + dc, win_c - max(0, dc) + dc)
            drop = elev_clean[cr, cc] - elev_clean[nr, nc]
            slope = np.maximum(drop / dist, 0.0)
            weight = slope ** mfd_p
            flow_v[cr, cc] += weight * _nb_dr[k]
            flow_u[cr, cc] += weight * _nb_dc[k]

        flow_u[nan_mask] = 0.0
        flow_v[nan_mask] = 0.0

        # Normalize
        mag = np.sqrt(flow_u**2 + flow_v**2)
        valid_flow = mag > 0
        flow_u[valid_flow] /= mag[valid_flow]
        flow_v[valid_flow] /= mag[valid_flow]

        flow_u = flow_u.astype(np.float32)
        flow_v = flow_v.astype(np.float32)

        # Slope magnitude
        grad_row, grad_col = np.gradient(
            np.nan_to_num(elevation, nan=0.0).astype(np.float64))
        slope_mag = np.sqrt(grad_row**2 + grad_col**2).astype(np.float32)
        p95 = np.percentile(slope_mag[slope_mag > 0], 95) \
            if (slope_mag > 0).any() else 1.0
        slope_norm = np.clip(
            slope_mag / max(p95, 1e-6), 0, 1).astype(np.float32)

        # Accumulation-based spawn weights (simple: use flow magnitude)
        spawn_weights = np.where(valid_flow, mag.astype(np.float32), 0.0)
        flat_weights = spawn_weights.ravel()
        valid_mask = flat_weights > 0
        valid_indices = np.nonzero(valid_mask)[0]
        if len(valid_indices) > 0:
            valid_probs = flat_weights[valid_indices].astype(np.float64)
            valid_probs /= valid_probs.sum()
        else:
            valid_indices = np.arange(win_h * win_c)
            valid_probs = np.ones(win_h * win_c, dtype=np.float64) / \
                (win_h * win_c)

        # Accumulation norm for particle colouring
        accum_norm = np.clip(mag.astype(np.float32) /
                             max(mag.max(), 1e-6), 0, 1)

        return {
            'win_r0': win_r0, 'win_c0': win_c0,
            'win_h': win_h, 'win_w': win_c,
            'center_tr': center_tr, 'center_tc': center_tc,
            'flow_u': flow_u, 'flow_v': flow_v,
            'slope_mag': slope_norm,
            'accum_norm': accum_norm.astype(np.float32),
            'spawn_indices': valid_indices,
            'spawn_probs': valid_probs,
            'stream_overlay': stream_overlay,
        }

    def _apply_window_result(self, result):
        """Apply a completed windowed flow computation to GPU state."""
        if result is None:
            return

        st = self._state
        self._win_r0 = float(result['win_r0'])
        self._win_c0 = float(result['win_c0'])
        self._win_h = result['win_h']
        self._win_w = result['win_w']
        self._window_center_tr = result['center_tr']
        self._window_center_tc = result['center_tc']

        # Update CPU-side grids
        st.hydro_flow_u_px = result['flow_u']
        st.hydro_flow_v_px = result['flow_v']
        st.hydro_slope_mag = result['slope_mag']
        st.hydro_flow_accum_norm = result['accum_norm']
        st.hydro_spawn_indices = result['spawn_indices']
        st.hydro_spawn_valid_probs = result['spawn_probs']

        # Stream order not available in windowed mode (would need
        # full xrspatial which is too slow for async recompute)
        st.hydro_stream_order = None
        st.hydro_stream_order_raw = None

        # Upload new grids to GPU
        if has_cupy:
            st.d_hydro_flow_u = cp.asarray(result['flow_u'])
            st.d_hydro_flow_v = cp.asarray(result['flow_v'])
            st.d_hydro_slope_mag = cp.asarray(result['slope_mag'])
            st.d_hydro_accum_norm = cp.asarray(result['accum_norm'])
            st.d_hydro_stream_order = cp.empty((0, 0), dtype=cp.float32)
            st.d_hydro_stream_order_raw = cp.empty((0, 0), dtype=cp.int32)

        # Store stream overlay for engine to pick up
        stream_ov = result.get('stream_overlay')
        if stream_ov is not None:
            self._pending_stream_overlay = stream_ov
            self._pending_overlay_bounds = (
                result['win_r0'], result['win_c0'],
                result['win_h'], result['win_w'],
            )

        print(f"  Hydro window shifted to "
              f"({result['win_r0']}, {result['win_c0']}) "
              f"size {result['win_h']}x{result['win_w']}")

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def shutdown(self):
        """Clean up thread pool."""
        if self._window_executor is not None:
            self._window_executor.shutdown(wait=False)
            self._window_executor = None
