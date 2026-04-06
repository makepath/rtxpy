"""Hydro-flow and GTFS-RT vehicle overlay subsystem for InteractiveViewer."""

import threading

import numpy as np

from ..rtx import has_cupy

if has_cupy:
    import cupy as cp

from .helpers import _add_overlay


class _HydroSystemView:
    """Adapter exposing hydro particle state in the ParticleSystem interface.

    Hydro particles are GPU-managed by HydroManager, so we can't use
    ParticleSystem directly.  This provides read-only access to the
    arrays that render_hydro_cpu() needs.
    """

    def __init__(self, viewer):
        self.positions = viewer._hydro_particles
        self.ages = viewer._hydro_ages
        self.lifetimes = viewer._hydro_lifetimes
        self.trails = viewer._hydro_trails
        self.n = (viewer._hydro_particles.shape[0]
                  if viewer._hydro_particles is not None else 0)
        self.trail_len = viewer._hydro_trail_len


class HydroController:
    """Hydrological flow particles and GTFS-RT realtime vehicle overlay.

    Accesses viewer state via ``self.v`` (back-reference to InteractiveViewer).
    """

    def __init__(self, viewer):
        self.v = viewer

    # ------------------------------------------------------------------
    # Hydrological flow particles
    # ------------------------------------------------------------------

    def _toggle_hydro(self):
        """Toggle hydro flow particles + stream_link water overlay together."""
        v = self.v
        if v._hydro_data is None:
            # Lazy mode: compute hydro from terrain on first enable
            if v._hydro_lazy:
                print("Computing hydro from terrain (first enable)...")
                if not self._compute_hydro_from_terrain():
                    return
            else:
                print("No hydro data. Use v.add_hydro(flow_accum).")
                return
        v._hydro_enabled = not v._hydro_enabled

        if v._hydro_enabled:
            # Save current overlay state for restoration on OFF
            v._hydro_prev_layer_idx = v._terrain_layer_idx
            # Switch to stream_link overlay with water reflection shader
            if 'stream_link' in v._overlay_layers:
                idx = v._terrain_layer_order.index('stream_link')
                v._terrain_layer_idx = idx
                v._active_overlay_data = v._overlay_layers['stream_link']
                v._overlay_as_water = True
                v._active_overlay_color_lut = v._overlay_color_luts.get(
                    'stream_link')
                print("Hydro flow: ON  (stream overlay + water shader)")
            else:
                print("Hydro flow: ON")
        else:
            # Restore previous overlay state
            prev = getattr(v, '_hydro_prev_layer_idx', 0)
            if prev >= len(v._terrain_layer_order):
                prev = 0
            v._terrain_layer_idx = prev
            layer = v._terrain_layer_order[prev]
            if layer == 'elevation':
                v._active_overlay_data = None
                v._overlay_as_water = False
                v._active_overlay_color_lut = None
            else:
                v._active_overlay_data = v._overlay_layers[layer]
                v._overlay_as_water = layer.startswith('flood_')
                v._active_overlay_color_lut = (
                    v._overlay_color_luts.get(layer))
            print("Hydro flow: OFF")

        v._update_frame()

    def _action_toggle_hydro(self):
        self._toggle_hydro()

    # Delegate palette/helpers to hydro_manager module
    from ..viewer.hydro_manager import (
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

        Delegates to ``v.hydro_mgr.init_from_flow()``.

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
        v = self.v
        terrain_data = v.raster.data
        v.hydro_mgr.init_from_flow(
            flow_accum, terrain_data,
            v._base_pixel_spacing_x, v._base_pixel_spacing_y,
            **kwargs)

    def _compute_hydro_from_terrain(self):
        """Compute hydrological flow from current terrain on GPU.

        Delegates to ``v.hydro_mgr.compute_from_terrain()``.
        Returns True on success, False on failure.
        """
        v = self.v
        v.hydro_mgr.set_terrain_ref(
            None, v._base_pixel_spacing_x, v._base_pixel_spacing_y)
        try:
            result = v.hydro_mgr.compute_from_terrain(v.raster)
        except Exception as exc:
            print(f"Hydro computation failed: {exc}")
            return False
        if result is None:
            return False

        # Register stream_link overlay with palette coloring
        if 'stream_link_overlay' in result:
            overlay = result['stream_link_overlay']
            _add_overlay(v, 'stream_link', overlay,
                         color_lut=result['palette_lut'])
            # Populate per-tile overlay manager for LOD rendering
            if v._overlay_tile_mgr is not None:
                mgr = v._terrain_lod_manager
                ts = mgr._tile_size if mgr is not None else 128
                H, W = overlay.shape
                n_tr = (H + ts - 1) // ts
                n_tc = (W + ts - 1) // ts
                v._overlay_tile_mgr.populate_from_array(
                    overlay, ts, n_tr, n_tc)
                v._overlay_tile_mgr.set_color_lut(
                    result['palette_lut'])
                n_tiles = len(v._overlay_tile_mgr._tile_overlays)
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
        v = self.v
        otm = v._overlay_tile_mgr
        if otm is None:
            return
        # Only transfer when overlay is the active layer
        if v._active_overlay_data is None:
            return
        overlay, win_r0, win_c0 = v.hydro_mgr.pop_streaming_overlay()
        if overlay is None:
            return

        mgr = v._terrain_lod_manager
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
        v = self.v
        dt_scale = float(getattr(v, '_dt_scale', 1.0))
        v.hydro_mgr.update_particles(dt_scale=dt_scale)

    def _draw_hydro_on_frame(self, img):
        """Project hydro particles to screen space and draw on rendered frame.

        CPU fallback — delegates to ``particles.hydro.render_hydro_cpu()``.

        Parameters
        ----------
        img : ndarray, shape (H_screen, W_screen, 3)
            Rendered frame (float32 0-1). Modified in-place.
        """
        v = self.v
        if v._hydro_particles is None:
            return

        from .particles.hydro import render_hydro_cpu
        from .particles.project import get_camera_basis

        system = _HydroSystemView(v)

        cam = get_camera_basis(v.position, v._get_look_at(), v.fov,
                               img.shape[1], img.shape[0])

        if v._hydro_terrain_np is None:
            td = v.raster.data
            v._hydro_terrain_np = td.get() if hasattr(td, 'get') else np.asarray(td)

        depth_buffer = getattr(v, '_d_depth_t', None)

        render_hydro_cpu(system, img, cam, v._hydro_terrain_np,
                         v._base_pixel_spacing_x, v._base_pixel_spacing_y,
                         v.vertical_exaggeration, v.subsample_factor,
                         v._hydro_min_depth,
                         base_alpha=v._hydro_alpha,
                         min_visible_age=v._hydro_min_visible_age,
                         ref_depth=v._hydro_ref_depth,
                         head_glow=1.5,
                         particle_colors=v._hydro_particle_colors,
                         particle_radii=v._hydro_particle_radii,
                         depth_buffer=depth_buffer)

    def _splat_hydro_gpu(self, d_frame):
        """Project and splat hydro particles on GPU.  Delegates to HydroManager."""
        v = self.v
        terrain_data = v.raster.data
        if has_cupy and not isinstance(terrain_data, cp.ndarray):
            terrain_data = cp.asarray(terrain_data)

        depth_t = getattr(v, '_d_depth_t', None)

        v.hydro_mgr.splat_gpu(
            d_frame,
            camera_pos=v.position,
            look_at=v._get_look_at(),
            fov=v.fov,
            ve=v.vertical_exaggeration,
            subsample_factor=v.subsample_factor,
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
        v = self.v
        v._gtfs_rt_url = realtime_url
        if route_colors:
            v._gtfs_rt_route_colors = route_colors
        print(f"GTFS-RT feed configured: {realtime_url}")
        print("  Press Shift+B to toggle realtime vehicle overlay.")

    def _toggle_gtfs_rt(self):
        """Toggle GTFS-RT realtime vehicle overlay on/off."""
        v = self.v
        if v._gtfs_rt_url is None:
            print("No GTFS-RT feed configured. Pass realtime_url in gtfs_data metadata.")
            return
        v._gtfs_rt_enabled = not v._gtfs_rt_enabled
        if v._gtfs_rt_enabled:
            if v._gtfs_rt_thread is None or not v._gtfs_rt_thread.is_alive():
                v._gtfs_rt_stop.clear()
                v._gtfs_rt_thread = threading.Thread(
                    target=self._gtfs_rt_poll_loop, daemon=True)
                v._gtfs_rt_thread.start()
            print("GTFS-RT vehicles: ON")
        else:
            v._gtfs_rt_stop.set()
            print("GTFS-RT vehicles: OFF")
        v._update_frame()

    def _gtfs_rt_poll_loop(self):
        """Background thread: poll GTFS-RT feed at regular intervals."""
        v = self.v
        import requests

        while not v._gtfs_rt_stop.is_set():
            try:
                resp = requests.get(v._gtfs_rt_url, timeout=30)
                resp.raise_for_status()
                self._parse_gtfs_rt_response(resp.content)
                v._render_needed = True
            except Exception as e:
                print(f"GTFS-RT poll error: {e}")

            v._gtfs_rt_stop.wait(v._gtfs_rt_poll_interval)

    def _parse_gtfs_rt_response(self, data):
        """Parse GTFS-RT protobuf VehiclePositions into numpy arrays."""
        v = self.v
        try:
            from google.transit import gtfs_realtime_pb2
        except ImportError:
            print("gtfs-realtime-bindings required for GTFS-RT. "
                  "Install with: pip install gtfs-realtime-bindings")
            v._gtfs_rt_stop.set()
            v._gtfs_rt_enabled = False
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
            color = v._gtfs_rt_route_colors.get(route_id, (1.0, 1.0, 1.0))

            positions.append((lon, lat))
            bearings.append(bearing)
            colors.append(color)

        if positions:
            with v._gtfs_rt_lock:
                v._gtfs_rt_vehicles = (
                    np.array(positions, dtype=np.float64),
                    np.array(bearings, dtype=np.float32),
                    np.array(colors, dtype=np.float32),
                )

    def _draw_gtfs_rt_on_frame(self, img):
        """Draw GTFS-RT vehicle positions as colored dots on the frame."""
        v = self.v
        with v._gtfs_rt_lock:
            if v._gtfs_rt_vehicles is None:
                return
            positions, bearings, colors = v._gtfs_rt_vehicles

        if len(positions) == 0:
            return

        # Convert lon/lat to world coordinates (pixel space)
        da = v.raster
        y_coords = da.coords[da.dims[-2]].values
        x_coords = da.coords[da.dims[-1]].values

        # lon/lat → pixel coords
        px_x = (positions[:, 0] - x_coords[0]) / (x_coords[-1] - x_coords[0]) * (len(x_coords) - 1)
        px_y = (positions[:, 1] - y_coords[0]) / (y_coords[-1] - y_coords[0]) * (len(y_coords) - 1)

        # World coords (match terrain mesh coordinate system)
        wx = px_x * abs(v.pixel_spacing_x)
        wy = px_y * abs(v.pixel_spacing_y)

        # Sample terrain Z for each vehicle (nearest neighbor)
        H, W = da.shape[-2:]
        ix = np.clip(np.round(px_x).astype(int), 0, W - 1)
        iy = np.clip(np.round(px_y).astype(int), 0, H - 1)

        terrain_np = v._wind_terrain_np
        if terrain_np is None:
            try:
                import cupy
                terrain_np = cupy.asnumpy(da.values)
            except Exception:
                terrain_np = np.asarray(da.values)
            v._wind_terrain_np = terrain_np

        wz = terrain_np[iy, ix].astype(np.float64) * v.vertical_exaggeration
        # Replace NaN with 0
        wz = np.where(np.isfinite(wz), wz, 0.0)

        # Project to screen space
        world = np.stack([wx, wy, wz], axis=-1)  # (N, 3)
        cam_pos = np.array(v.position, dtype=np.float64)
        cam_fwd = np.array(v._camera_forward(), dtype=np.float64)
        cam_right = np.array(v._camera_right(), dtype=np.float64)
        cam_up = np.array(v._camera_up(), dtype=np.float64)

        rel = world - cam_pos  # (N, 3)
        depth = rel @ cam_fwd
        behind = depth <= 0.1
        depth[behind] = 1.0  # avoid division by zero

        fov_rad = np.radians(v.fov)
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
        r = v._gtfs_rt_dot_radius
        alpha = v._gtfs_rt_alpha

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
        v = self.v
        if v._gtfs_rt_thread is not None:
            v._gtfs_rt_stop.set()
            v._gtfs_rt_thread.join(timeout=2.0)
            v._gtfs_rt_thread = None
