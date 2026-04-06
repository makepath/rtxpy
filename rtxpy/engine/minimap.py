"""Minimap rendering subsystem for InteractiveViewer."""

import time

import numpy as np

from ..rtx import has_cupy


class MinimapRenderer:
    """Minimap background, projection, and blitting methods.

    Composition subsystem -- holds a reference to the full
    ``InteractiveViewer`` as ``self.v``.
    """

    def __init__(self, viewer):
        self.v = viewer

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
        v = self.v
        H, W = v.terrain_shape
        terrain_data = v.raster.data
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
        if (v._minimap_layer and v._minimap_colors
                and v._minimap_layer in v._base_overlay_layers):
            ld = v._base_overlay_layers[v._minimap_layer]
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
            for val, (r, g, b) in v._minimap_colors.items():
                mask = np.isclose(_layer_data, float(val), atol=0.1)
                for c, cv in enumerate((r, g, b)):
                    rgba[:, :, c] = np.where(
                        mask, cv * (shaded * 0.5 + 0.5), rgba[:, :, c])

        # Blend satellite imagery if tile service has fetched tiles
        if (v._tile_service is not None
                and getattr(v._tile_service, '_fetched', None)):
            cpu_tex = getattr(v._tile_service, '_rgb_texture', None)
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
                v._minimap_has_tiles = True

        # Apply minimap style filter
        if v._minimap_style == 'cyberpunk':
            rgba = self._apply_cyberpunk_minimap(rgba, water)

        v._minimap_background = rgba
        v._minimap_scale_x = new_w / W
        v._minimap_scale_y = new_h / H

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
        v = self.v
        tile_data_fn = getattr(v, '_tile_data_fn', None)
        crs_tf = getattr(v, '_minimap_crs_transform', None)
        if tile_data_fn is None or crs_tf is None:
            return

        crs_x0, crs_y0, crs_dx, crs_dy = crs_tf
        psx, psy = v.pixel_spacing_x, v.pixel_spacing_y

        # Convert world coords to CRS coordinates
        # world coord: wx = col * psx + offset_x
        # col = (wx - offset_x) / psx
        # crs_x = crs_x0 + col * crs_dx
        lod_mgr = getattr(v.terrain, '_terrain_lod_manager', None)
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

        if v._minimap_style == 'cyberpunk':
            rgba = self._apply_cyberpunk_minimap(rgba, water)

        v._minimap_background = rgba
        v._minimap_bg_extent = (wx_min, wy_min, wx_max, wy_max)
        # Scale factors map the full bg image to the world extent
        v._minimap_scale_x = rgba.shape[1] / max(1, wx_max - wx_min)
        v._minimap_scale_y = rgba.shape[0] / max(1, wy_max - wy_min)
        v._minimap_last_stream_time = time.monotonic()

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
        # Blend cyan->magenta based on edge intensity
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
        v = self.v
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
            far_dist = max(v.terrain_shape[0] * v.pixel_spacing_y,
                           v.terrain_shape[1] * v.pixel_spacing_x)
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
        v = self.v
        if v._minimap_background is None or not v.show_minimap:
            return

        # Lazy re-check: pick up satellite tiles once they arrive
        if (not v._minimap_has_tiles
                and v._tile_service is not None
                and getattr(v._tile_service, '_fetched', None)):
            self._compute_minimap_background()

        mm_bg = v._minimap_background  # (mm_h, mm_w, 4) RGBA float32
        mm_h, mm_w = mm_bg.shape[:2]
        fh, fw = img.shape[:2]

        # Size minimap: match legend height if available, else ~20% of frame.
        # Use initial terrain aspect ratio (not background image shape)
        # so dimensions stay fixed even when streaming changes the bg.
        H_t, W_t = v.terrain_shape
        terrain_aspect = W_t / max(1, H_t)
        if v._legend_rgba is not None:
            target_h = v._legend_rgba.shape[0]
        else:
            target_w = max(40, int(fw * 0.2))
            target_h = max(20, int(target_w / terrain_aspect))
        target_w = max(20, int(target_h * terrain_aspect))
        target_w = min(target_w, fw)
        target_h = min(target_h, fh)

        # --- World extent for coordinate mapping ---
        # When LOD streaming is active, extend beyond initial terrain
        # to include camera position so the dot stays visible.
        H, W = v.terrain_shape
        psx, psy = v.pixel_spacing_x, v.pixel_spacing_y
        terrain_wx = W * psx
        terrain_wy = H * psy

        lod_mgr = getattr(v.terrain, '_terrain_lod_manager', None)
        streaming = (lod_mgr is not None
                     and getattr(lod_mgr, '_streaming', False))

        if streaming:
            cam_x, cam_y = v.position[0], v.position[1]
            # Desired extent: union of terrain bounds and camera + margin
            margin = max(terrain_wx, terrain_wy) * 0.5
            wx_min = min(0.0, cam_x - margin)
            wy_min = min(0.0, cam_y - margin)
            wx_max = max(terrain_wx, cam_x + margin)
            wy_max = max(terrain_wy, cam_y + margin)

            # Check if streaming minimap recompute is needed:
            # - No extent yet (first time)
            # - Camera outside inner 60% of current extent AND 2s throttle
            bg_ext = v._minimap_bg_extent
            now = time.monotonic()
            throttle_ok = (now - v._minimap_last_stream_time >= 2.0)
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
                mm_bg = v._minimap_background
                mm_h, mm_w = mm_bg.shape[:2]

            # Use streaming extent for coordinate mapping
            if v._minimap_bg_extent is not None:
                wx_min, wy_min, wx_max, wy_max = v._minimap_bg_extent
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
        v._minimap_rect = (x0, y0, target_w, target_h)
        v._minimap_world_extent = (wx_min, wy_min, wx_max, wy_max)

        # --- Camera position in minimap-local coords ---
        lx = (v.position[0] - wx_min) / wx_range * target_w
        ly = (v.position[1] - wy_min) / wy_range * target_h

        ve = v.vertical_exaggeration
        terrain_z = v._get_terrain_z(v.position[0], v.position[1]) * ve

        # Camera basis in VE-scaled space
        pos_ve = np.array([v.position[0], v.position[1],
                           v.position[2] * ve], dtype=np.float32)
        look_ve = np.array([v.position[0] + v._get_front()[0] * 1000,
                            v.position[1] + v._get_front()[1] * 1000,
                            (v.position[2] + v._get_front()[2] * 1000) * ve],
                           dtype=np.float32)
        # Simple basis from yaw/pitch
        yaw_rad = np.radians(v.yaw)
        pitch_rad = np.radians(v.pitch)
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

        fov_scale = np.tan(np.radians(v.fov) / 2.0)
        aspect = v.render_width / max(1, v.render_height)

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
        for slot, obs in v._observers.items():
            if obs.position is None:
                continue
            obs_x, obs_y = obs.position
            obs_lx = (obs_x - wx_min) / wx_range * target_w
            obs_ly = (obs_y - wy_min) / wy_range * target_h
            r = 4 if slot == v._active_observer else 2
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
        if np.isnan(x1) or np.isnan(y1) or np.isnan(x2) or np.isnan(y2):
            return
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
        if np.any(np.isnan(pts_x)) or np.any(np.isnan(pts_y)):
            return
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
