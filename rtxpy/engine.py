"""Interactive terrain viewer using matplotlib for display.

This module provides a simple game-engine-like render loop for
exploring terrain interactively with keyboard controls.
Uses matplotlib for display (no additional dependencies).
"""

import time
import numpy as np
from typing import Optional, Tuple

from .rtx import RTX, has_cupy

if has_cupy:
    import cupy as cp


class InteractiveViewer:
    """
    Interactive terrain viewer using matplotlib.

    Provides keyboard-controlled camera for exploring ray-traced terrain.
    Uses matplotlib's event system for input handling.

    Controls
    --------
    - W/Up: Move forward
    - S/Down: Move backward
    - A/Left: Strafe left
    - D/Right: Strafe right
    - Q/Page Up: Move up
    - E/Page Down: Move down
    - I/J/K/L: Look up/left/down/right
    - Scroll wheel: Zoom in/out (FOV)
    - +/=: Increase speed
    - -: Decrease speed
    - G: Cycle geometry layers (when meshes are placed)
    - N: Jump to next geometry in current layer
    - P: Jump to previous geometry in current layer
    - O: Place observer (for viewshed) at look-at point
    - V: Toggle viewshed overlay (teal glow shows visible terrain)
    - [/]: Decrease/increase observer height
    - R: Decrease terrain resolution (coarser, up to 8x subsample)
    - Shift+R: Increase terrain resolution (finer, down to 1x)
    - Z: Decrease vertical exaggeration
    - Shift+Z: Increase vertical exaggeration
    - B: Toggle mesh type (TIN / voxel)
    - Y: Cycle color stretch (linear, sqrt, cbrt, log)
    - T: Toggle shadows
    - C: Cycle colormap
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
                 mesh_type: str = 'tin',
                 overlay_layers: dict = None):
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
        self.subsample_factor = 1

        # Color stretch cycling (Y key)
        self._color_stretches = ['linear', 'sqrt', 'cbrt', 'log']
        self._color_stretch_idx = 0

        # Vertical exaggeration (Z / Shift+Z)
        self.vertical_exaggeration = 1.0

        # Overlay layers for Dataset variable cycling (G key)
        # Dict of {name: 2D cupy/numpy array} — colormap data alternatives
        self._overlay_layers = overlay_layers or {}
        self._overlay_names = list(self._overlay_layers.keys())
        self._overlay_idx = -1  # -1 = elevation (default), 0+ = overlay index
        self._active_color_data = None  # None = use elevation_data

        # GAS layer visibility tracking
        self._all_geometries = []
        self._layer_mode = 0  # 0=all, then cycle through geometry groups
        self._layer_modes = ['all']  # Will be populated with geometry groups
        self._layer_positions = {}  # layer_name -> [(x, y, z, geometry_id), ...]
        self._current_geom_idx = 0  # Current geometry index within active layer

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
                groups.add(base_name)

                if base_name not in layer_geoms:
                    layer_geoms[base_name] = []
                layer_geoms[base_name].append(g)

            self._layer_modes.extend(sorted(groups))

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
        self.sun_azimuth = 225.0
        self.sun_altitude = 35.0
        self.shadows = True
        self.ambient = 0.2
        self.colormap = 'terrain'
        self.colormaps = ['terrain', 'viridis', 'plasma', 'cividis', 'gray']
        self.colormap_idx = 0
        self.color_stretch = 'linear'

        # Viewshed settings
        self.viewshed_enabled = False
        self.viewshed_observer_elev = 10.0  # Default 10m above surface (tower height)
        self.viewshed_target_elev = 0.0
        self.viewshed_opacity = 0.6
        self._viewshed_cache = None  # Cached viewshed result
        self._viewshed_coverage = 0.0  # Percentage of terrain visible
        self._observer_position = None  # Fixed observer position (x, y) in terrain coords

        # State
        self.running = False
        self.show_help = True
        self.show_minimap = True
        self.frame_count = 0

        # Minimap state (initialized in run() via _compute_minimap_background/_create_minimap)
        self._minimap_ax = None
        self._minimap_im = None
        self._minimap_camera_dot = None
        self._minimap_direction_line = None
        self._minimap_fov_wedge = None
        self._minimap_observer_dot = None
        self._minimap_background = None
        self._minimap_scale_x = 1.0
        self._minimap_scale_y = 1.0

        # Held keys tracking for smooth simultaneous input
        self._held_keys = set()
        self._tick_interval = int(key_repeat_interval * 1000)  # Convert to ms for timer
        self._timer = None

        # Get terrain info
        H, W = raster.shape
        terrain_data = raster.data
        if hasattr(terrain_data, 'get'):
            terrain_np = terrain_data.get()
        else:
            terrain_np = np.asarray(terrain_data)

        self.terrain_shape = (H, W)
        self.elev_min = float(np.nanmin(terrain_np))
        self.elev_max = float(np.nanmax(terrain_np))
        self.elev_mean = float(np.nanmean(terrain_np))

        # Build terrain geometry if RTX exists but has no terrain.
        # Without this, render() falls into the auto-VE / prepare_mesh path
        # which computes vertical_exaggeration from pixel dimensions (not world
        # units), producing wrong results when pixel_spacing != 1.
        if rtx is not None and not rtx.has_geometry('terrain'):
            from . import mesh as mesh_mod
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

            if pixel_spacing_x != 1.0 or pixel_spacing_y != 1.0:
                verts[0::3] *= pixel_spacing_x
                verts[1::3] *= pixel_spacing_y

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

    def _compute_minimap_background(self):
        """Compute a CPU hillshade image for the minimap background.

        Downsamples terrain to max 200px on longest side, then uses
        numpy gradient-based hillshade. Runs once at startup.
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
            # Simple nearest-neighbor downsample via strided indexing
            y_idx = np.linspace(0, H - 1, new_h).astype(int)
            x_idx = np.linspace(0, W - 1, new_w).astype(int)
            terrain_small = terrain_np[np.ix_(y_idx, x_idx)]
        else:
            terrain_small = terrain_np.copy()
            new_h, new_w = H, W

        # Replace NaNs with median for gradient computation
        mask = np.isnan(terrain_small)
        if mask.any():
            terrain_small[mask] = np.nanmedian(terrain_small)

        # Compute hillshade using gradient
        dy, dx = np.gradient(terrain_small)
        # Sun from upper-left (azimuth=315, altitude=45)
        az_rad = np.radians(315)
        alt_rad = np.radians(45)
        slope = np.sqrt(dx**2 + dy**2)
        aspect = np.arctan2(-dy, dx)
        shaded = (np.sin(alt_rad) * np.cos(np.arctan(slope)) +
                  np.cos(alt_rad) * np.sin(np.arctan(slope)) *
                  np.cos(az_rad - aspect))
        shaded = np.clip(shaded, 0, 1)

        self._minimap_background = shaded
        self._minimap_scale_x = new_w / W  # minimap pixels per terrain pixel
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
        H, W = sub.shape
        self.terrain_shape = (H, W)

        # 2. Update pixel spacing
        self.pixel_spacing_x = self._base_pixel_spacing_x * factor
        self.pixel_spacing_y = self._base_pixel_spacing_y * factor

        # 3. Update elevation stats
        terrain_data = sub.data
        if hasattr(terrain_data, 'get'):
            terrain_np = terrain_data.get()
        else:
            terrain_np = np.asarray(terrain_data)
        self.elev_min = float(np.nanmin(terrain_np))
        self.elev_max = float(np.nanmax(terrain_np))
        self.elev_mean = float(np.nanmean(terrain_np))

        # 4. Remove old terrain geometry and rebuild
        if self.rtx is not None and self.rtx.has_geometry('terrain'):
            self.rtx.remove_geometry('terrain')

        if self.rtx is not None:
            if self.mesh_type == 'voxel':
                # Voxel mesh: 8 verts/cell, 12 tris/cell
                num_verts = H * W * 8
                num_tris = H * W * 12
                vertices = np.zeros(num_verts * 3, dtype=np.float32)
                indices = np.zeros(num_tris * 3, dtype=np.int32)
                base_elev = float(np.nanmin(terrain_np))
                mesh_mod.voxelate_terrain(vertices, indices, sub, scale=1.0,
                                          base_elevation=base_elev)
            else:
                # TIN mesh: 1 vert/cell, 2 tris/quad
                num_verts = H * W
                num_tris = (H - 1) * (W - 1) * 2
                vertices = np.zeros(num_verts * 3, dtype=np.float32)
                indices = np.zeros(num_tris * 3, dtype=np.int32)
                mesh_mod.triangulate_terrain(vertices, indices, sub, scale=1.0)

            # Scale x,y to world units
            if self.pixel_spacing_x != 1.0 or self.pixel_spacing_y != 1.0:
                vertices[0::3] *= self.pixel_spacing_x
                vertices[1::3] *= self.pixel_spacing_y

            self.rtx.add_geometry('terrain', vertices, indices)

        # 5. Subsample overlay layers
        if self._base_overlay_layers:
            self._overlay_layers = {}
            for name, data in self._base_overlay_layers.items():
                if factor > 1:
                    self._overlay_layers[name] = data[::factor, ::factor]
                else:
                    self._overlay_layers[name] = data
            self._overlay_names = list(self._overlay_layers.keys())
            # Reset active color data if an overlay is selected
            if self._overlay_idx >= 0 and self._overlay_idx < len(self._overlay_names):
                oname = self._overlay_names[self._overlay_idx]
                self._active_color_data = self._overlay_layers[oname]

        # 6. Re-snap placed meshes to new terrain surface
        if self.rtx is not None:
            for geom_id in self.rtx.list_geometries():
                if geom_id == 'terrain':
                    continue
                transform = self.rtx.get_geometry_transform(geom_id)
                if transform is None:
                    continue
                # World position
                wx, wy = transform[3], transform[7]
                # Convert to pixel coords in the new subsampled grid
                px = wx / self.pixel_spacing_x
                py = wy / self.pixel_spacing_y
                # Clamp and sample Z
                ix = int(np.clip(px, 0, W - 1))
                iy = int(np.clip(py, 0, H - 1))
                z = float(terrain_np[iy, ix])
                transform[11] = z
                self.rtx.update_transform(geom_id, transform)

        # 7. Recompute minimap
        self._compute_minimap_background()
        if self._minimap_im is not None:
            self._minimap_im.set_data(self._minimap_background)
            # Update minimap axes limits for new background size
            mm_h, mm_w = self._minimap_background.shape
            self._minimap_im.set_extent([-0.5, mm_w - 0.5, mm_h - 0.5, -0.5])

        # 8. Clear viewshed cache (no longer matches terrain)
        self._viewshed_cache = None
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

        terrain_data = self.raster.data
        if hasattr(terrain_data, 'get'):
            terrain_np = terrain_data.get()
        else:
            terrain_np = np.asarray(terrain_data)

        # Remove old terrain and rebuild with new scale
        if self.rtx is not None and self.rtx.has_geometry('terrain'):
            self.rtx.remove_geometry('terrain')

        if self.rtx is not None:
            if self.mesh_type == 'voxel':
                nv = H * W * 8
                nt = H * W * 12
                vertices = np.zeros(nv * 3, dtype=np.float32)
                indices = np.zeros(nt * 3, dtype=np.int32)
                base_elev = float(np.nanmin(terrain_np)) * ve
                mesh_mod.voxelate_terrain(vertices, indices, self.raster,
                                          scale=ve, base_elevation=base_elev)
            else:
                nv = H * W
                nt = (H - 1) * (W - 1) * 2
                vertices = np.zeros(nv * 3, dtype=np.float32)
                indices = np.zeros(nt * 3, dtype=np.int32)
                mesh_mod.triangulate_terrain(vertices, indices, self.raster,
                                             scale=ve)

            if self.pixel_spacing_x != 1.0 or self.pixel_spacing_y != 1.0:
                vertices[0::3] *= self.pixel_spacing_x
                vertices[1::3] *= self.pixel_spacing_y

            self.rtx.add_geometry('terrain', vertices, indices)

        # Update elevation stats (scaled)
        self.elev_min = float(np.nanmin(terrain_np)) * ve
        self.elev_max = float(np.nanmax(terrain_np)) * ve
        self.elev_mean = float(np.nanmean(terrain_np)) * ve

        # Re-snap placed meshes to scaled terrain
        if self.rtx is not None:
            for geom_id in self.rtx.list_geometries():
                if geom_id == 'terrain':
                    continue
                transform = self.rtx.get_geometry_transform(geom_id)
                if transform is None:
                    continue
                wx, wy = transform[3], transform[7]
                px = wx / self.pixel_spacing_x
                py = wy / self.pixel_spacing_y
                ix = int(np.clip(px, 0, W - 1))
                iy = int(np.clip(py, 0, H - 1))
                z = float(terrain_np[iy, ix]) * ve
                transform[11] = z
                self.rtx.update_transform(geom_id, transform)

        # Clear viewshed cache
        self._viewshed_cache = None
        if self.viewshed_enabled:
            self.viewshed_enabled = False
            print("  Viewshed disabled (terrain changed). Press V to recalculate.")

        print(f"Vertical exaggeration: {ve:.2f}x")
        self._update_frame()

    def _create_minimap(self):
        """Create the minimap inset axes and persistent artists."""
        if self._minimap_background is None:
            return

        # Create inset axes in bottom-right corner (~20% of figure width)
        mm_h, mm_w = self._minimap_background.shape
        aspect = mm_h / mm_w
        ax_width = 0.2
        ax_height = ax_width * aspect * (self.width / self.height)
        # Clamp height so it doesn't get too tall
        ax_height = min(ax_height, 0.35)
        margin = 0.02
        self._minimap_ax = self.fig.add_axes(
            [1 - ax_width - margin, margin, ax_width, ax_height]
        )
        self._minimap_ax.set_xticks([])
        self._minimap_ax.set_yticks([])
        for spine in self._minimap_ax.spines.values():
            spine.set_edgecolor('white')
            spine.set_linewidth(0.8)

        # Display hillshade background (origin='upper' so row 0 is top = +Y)
        self._minimap_im = self._minimap_ax.imshow(
            self._minimap_background, cmap='gray', vmin=0, vmax=1,
            aspect='auto', origin='upper'
        )

        # FOV wedge (filled semi-transparent cone showing visible area)
        from matplotlib.patches import Polygon
        self._minimap_fov_wedge = Polygon(
            [[0, 0]], closed=True, facecolor='red', alpha=0.25,
            edgecolor='red', linewidth=0.8, zorder=3
        )
        self._minimap_ax.add_patch(self._minimap_fov_wedge)

        # Direction line (bright, with arrowhead effect via thicker line)
        self._minimap_direction_line, = self._minimap_ax.plot(
            [], [], color='#ff4444', linewidth=2.0, solid_capstyle='round', zorder=4
        )

        # Camera position dot (white-outlined red)
        self._minimap_camera_dot = self._minimap_ax.scatter(
            [], [], c='red', s=30, zorder=5, edgecolors='white', linewidths=0.8
        )

        # Observer dot (magenta star)
        self._minimap_observer_dot = self._minimap_ax.scatter(
            [], [], c='magenta', s=50, marker='*', zorder=6,
            edgecolors='white', linewidths=0.3
        )

        self._minimap_ax.set_visible(self.show_minimap)

    def _update_minimap(self):
        """Update minimap artists with current camera/observer state."""
        if self._minimap_ax is None:
            return

        self._minimap_ax.set_visible(self.show_minimap)
        if not self.show_minimap:
            return

        H, W = self.terrain_shape

        # Convert camera world position to minimap pixel coords
        # World coords: x = col * pixel_spacing_x, y = row * pixel_spacing_y
        # Pixel indices: col = x / pixel_spacing_x, row = y / pixel_spacing_y
        # Minimap coords: mx = col * scale_x, my = row * scale_y
        cam_col = self.position[0] / self.pixel_spacing_x
        cam_row = self.position[1] / self.pixel_spacing_y

        mx = cam_col * self._minimap_scale_x
        # Flip Y: minimap origin='upper', so row 0 is displayed at top
        # In world coords, +Y is increasing row. With origin='upper',
        # imshow row 0 is at top, so minimap y = row * scale_y directly.
        my = cam_row * self._minimap_scale_y

        # Update camera dot
        self._minimap_camera_dot.set_offsets([[mx, my]])

        # Direction line length in minimap pixels
        mm_h, mm_w = self._minimap_background.shape
        line_len = max(mm_h, mm_w) * 0.12

        # Yaw: 0 = +X (right on minimap), 90 = +Y (down on minimap with origin='upper')
        yaw_rad = np.radians(self.yaw)
        dx = np.cos(yaw_rad) * line_len
        dy = np.sin(yaw_rad) * line_len  # +Y in world = +row = down in minimap

        self._minimap_direction_line.set_data([mx, mx + dx], [my, my + dy])

        # FOV wedge (filled triangle from camera through left/right edges)
        half_fov = np.radians(self.fov / 2)
        fov_len = line_len * 0.8

        left_angle = yaw_rad - half_fov
        right_angle = yaw_rad + half_fov

        lx = np.cos(left_angle) * fov_len
        ly = np.sin(left_angle) * fov_len
        rx = np.cos(right_angle) * fov_len
        ry = np.sin(right_angle) * fov_len

        self._minimap_fov_wedge.set_xy([
            [mx, my],
            [mx + lx, my + ly],
            [mx + rx, my + ry],
        ])

        # Observer dot
        if self._observer_position is not None:
            obs_x, obs_y = self._observer_position
            obs_col = obs_x / self.pixel_spacing_x
            obs_row = obs_y / self.pixel_spacing_y
            omx = obs_col * self._minimap_scale_x
            omy = obs_row * self._minimap_scale_y
            self._minimap_observer_dot.set_offsets([[omx, omy]])
            self._minimap_observer_dot.set_visible(True)
        else:
            self._minimap_observer_dot.set_visible(False)

    def _handle_key_press(self, event):
        """Handle key press - add to held keys or handle instant actions."""
        raw_key = event.key if event.key else ''
        key = raw_key.lower()

        # Movement/look keys are tracked as held
        movement_keys = {'w', 's', 'a', 'd', 'up', 'down', 'left', 'right',
                         'q', 'e', 'pageup', 'pagedown', 'i', 'j', 'k', 'l'}

        if key in movement_keys:
            self._held_keys.add(key)
            return

        # Instant actions (not held)
        # Speed (limits scale with terrain size)
        if key in ('+', '='):
            terrain_diagonal = np.sqrt(self.terrain_shape[0]**2 + self.terrain_shape[1]**2)
            max_speed = terrain_diagonal * 0.1  # Max 10% of terrain per keystroke
            self.move_speed = min(max_speed, self.move_speed * 1.2)
            print(f"Speed: {self.move_speed:.1f}")
        elif key == '-':
            terrain_diagonal = np.sqrt(self.terrain_shape[0]**2 + self.terrain_shape[1]**2)
            min_speed = terrain_diagonal * 0.001  # Min 0.1% of terrain per keystroke
            self.move_speed = max(min_speed, self.move_speed / 1.2)
            print(f"Speed: {self.move_speed:.1f}")

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
            self._cycle_layer()
        elif key == 'n':
            self._jump_to_geometry(1)  # Next geometry
        elif key == 'p':
            self._jump_to_geometry(-1)  # Previous geometry
        elif key == 'h':
            self.show_help = not self.show_help
            self._update_frame()
        elif key == 'm':
            self.show_minimap = not self.show_minimap
            self._update_frame()

        # Viewshed controls
        elif key == 'o':
            self._place_observer()
        elif key == 'v':
            self._toggle_viewshed()
        elif key == '[':
            self._adjust_observer_elevation(-5.0)
        elif key == ']':
            self._adjust_observer_elevation(5.0)

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

        # Toggle mesh type (tin ↔ voxel)
        elif key == 'b':
            self.mesh_type = 'voxel' if self.mesh_type == 'tin' else 'tin'
            self._rebuild_vertical_exaggeration(self.vertical_exaggeration)
            print(f"Mesh type: {self.mesh_type}")

        # Vertical exaggeration: Z = decrease, Shift+Z = increase
        elif key == 'z':
            if raw_key == 'Z':
                new_ve = min(10.0, self.vertical_exaggeration * 1.5)
            else:
                new_ve = max(0.25, self.vertical_exaggeration / 1.5)
            if new_ve != self.vertical_exaggeration:
                self._rebuild_vertical_exaggeration(new_ve)

        # Exit
        elif key in ('escape', 'x'):
            self.running = False
            if self._timer is not None:
                self._timer.stop()
            import matplotlib.pyplot as plt
            plt.close(self.fig)

    def _handle_key_release(self, event):
        """Handle key release - remove from held keys."""
        key = event.key.lower() if event.key else ''
        self._held_keys.discard(key)

    def _tick(self):
        """Process all held keys and update frame (called by timer)."""
        if not self.running or not self._held_keys:
            return

        # Get direction vectors (computed once per tick)
        front = self._get_front()
        right = self._get_right()

        # Process all held movement keys
        if 'w' in self._held_keys or 'up' in self._held_keys:
            self.position += front * self.move_speed
        if 's' in self._held_keys or 'down' in self._held_keys:
            self.position -= front * self.move_speed
        if 'a' in self._held_keys or 'left' in self._held_keys:
            self.position -= right * self.move_speed
        if 'd' in self._held_keys or 'right' in self._held_keys:
            self.position += right * self.move_speed
        if 'q' in self._held_keys or 'pageup' in self._held_keys:
            self.position[2] += self.move_speed
        if 'e' in self._held_keys or 'pagedown' in self._held_keys:
            self.position[2] -= self.move_speed

        # Process all held look keys
        if 'i' in self._held_keys:
            self.pitch = min(89, self.pitch + self.look_speed)
        if 'k' in self._held_keys:
            self.pitch = max(-89, self.pitch - self.look_speed)
        if 'j' in self._held_keys:
            self.yaw -= self.look_speed
        if 'l' in self._held_keys:
            self.yaw += self.look_speed

        # Trigger redraw
        self._update_frame()

    def _cycle_layer(self):
        """Cycle through layer modes.

        When overlay layers are present (Dataset mode): cycles the colormap
        data source (elevation → overlay1 → overlay2 → … → elevation).

        When geometry groups are present (placed-mesh mode): uses RTX
        visibility masks to hide/show geometry groups.
        """
        # Dataset overlay mode: cycle color data source
        if self._overlay_names:
            # -1 = elevation, 0..N-1 = overlay layers
            self._overlay_idx += 1
            if self._overlay_idx >= len(self._overlay_names):
                self._overlay_idx = -1

            if self._overlay_idx == -1:
                self._active_color_data = None
                print(f"Layer: elevation (terrain coloring)")
            else:
                name = self._overlay_names[self._overlay_idx]
                self._active_color_data = self._overlay_layers[name]
                print(f"Layer: {name}")
            self._update_frame()
            return

        # Placed-mesh geometry mode: toggle visibility
        if not self._layer_modes or self.rtx is None:
            print("No geometries in scene")
            return

        self._layer_mode = (self._layer_mode + 1) % len(self._layer_modes)
        mode = self._layer_modes[self._layer_mode]

        if mode == 'all':
            for geom_id in self._all_geometries:
                self.rtx.set_geometry_visible(geom_id, True)
            print(f"Layer: ALL ({len(self._all_geometries)} geometries)")
        else:
            visible_count = 0
            for geom_id in self._all_geometries:
                parts = geom_id.rsplit('_', 1)
                base_name = parts[0] if len(parts) == 2 and parts[1].isdigit() else geom_id

                if base_name == mode or geom_id == mode or geom_id == 'terrain':
                    self.rtx.set_geometry_visible(geom_id, True)
                    visible_count += 1
                else:
                    self.rtx.set_geometry_visible(geom_id, False)

            print(f"Layer: {mode} ({visible_count} visible)")

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

        # Get current layer name
        mode = self._layer_modes[self._layer_mode]

        if mode == 'all':
            print("Select a specific layer with G first (e.g., 'tower', 'house')")
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

    def _place_observer(self):
        """Place a viewshed observer at the current camera position on terrain.

        The observer is placed at the camera's x,y location, projected onto
        the terrain surface. This becomes the fixed point for viewshed analysis.
        Observer position is stored in world coordinates (same as camera).
        """
        H, W = self.terrain_shape

        # Use camera position (in world coordinates if pixel_spacing != 1.0)
        cam_x = self.position[0]
        cam_y = self.position[1]

        # Compute terrain bounds in world coordinates
        max_x = (W - 1) * self.pixel_spacing_x
        max_y = (H - 1) * self.pixel_spacing_y

        # Clamp to terrain bounds (in world coordinates)
        obs_x = float(np.clip(cam_x, 0, max_x))
        obs_y = float(np.clip(cam_y, 0, max_y))

        # Store in world coordinates (for orb rendering)
        self._observer_position = (obs_x, obs_y)

        # Also compute pixel indices for display
        px_x = int(obs_x / self.pixel_spacing_x)
        px_y = int(obs_y / self.pixel_spacing_y)

        print(f"Observer placed at world ({obs_x:.0f}, {obs_y:.0f}), pixel ({px_x}, {px_y})")
        print(f"  Height: {self.viewshed_observer_elev:.0f}m above terrain")
        print(f"  Press V to toggle viewshed, [/] to adjust height")

        # If viewshed is already enabled, recalculate
        if self.viewshed_enabled:
            self._calculate_viewshed()

        self._update_frame()

    def _calculate_viewshed(self):
        """Calculate viewshed from the placed observer position.

        Uses GPU ray tracing to compute visibility from the fixed observer.
        Observer position is in world coordinates; this method converts to
        pixel indices for the viewshed calculation.
        """
        from .analysis.viewshed import _viewshed_rt
        from .analysis._common import prepare_mesh

        if self._observer_position is None:
            print("No observer placed. Press O to place an observer first.")
            return None

        # Observer position is in world coordinates
        world_x, world_y = self._observer_position
        H, W = self.terrain_shape

        # Convert world coords to pixel indices
        px_x = world_x / self.pixel_spacing_x
        px_y = world_y / self.pixel_spacing_y

        # Validate coordinates are within terrain bounds (in pixel space)
        if px_x < 0 or px_x >= W or px_y < 0 or px_y >= H:
            print(f"Observer position pixel ({px_x:.1f}, {px_y:.1f}) outside terrain bounds")
            return None

        print(f"Computing viewshed... (observer height: {self.viewshed_observer_elev:.0f}m)")
        print(f"  Raster shape: {self.raster.shape}, pixel_spacing: ({self.pixel_spacing_x:.1f}, {self.pixel_spacing_y:.1f})")

        try:
            # Always use a fresh mesh for viewshed calculation
            # (self.rtx might have placed meshes that interfere with viewshed rays)
            print("  Building fresh terrain mesh...")
            rtx = prepare_mesh(self.raster, rtx=None)
            print("  Mesh built successfully")

            # Convert pixel indices to raster coords
            y_coords = self.raster.indexes.get('y').values
            x_coords = self.raster.indexes.get('x').values

            # Clamp to valid range and get actual coord values
            x_idx = int(np.clip(px_x, 0, W - 1))
            y_idx = int(np.clip(px_y, 0, H - 1))
            x_coord = x_coords[x_idx] if x_idx < len(x_coords) else x_coords[-1]
            y_coord = y_coords[y_idx] if y_idx < len(y_coords) else y_coords[-1]

            print(f"  Observer at raster coords: ({x_coord:.1f}, {y_coord:.1f})")

            viewshed = _viewshed_rt(
                self.raster, rtx,
                x_coord, y_coord,
                self.viewshed_observer_elev,
                self.viewshed_target_elev
            )

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

            print(f"  Coverage: {self._viewshed_coverage:.1f}% terrain visible")
            return viewshed

        except Exception as e:
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
        """Toggle viewshed overlay on/off."""
        if self._observer_position is None:
            print("No observer placed. Press O to place an observer first.")
            return

        self.viewshed_enabled = not self.viewshed_enabled

        if self.viewshed_enabled:
            print("Calculating viewshed...")
            viewshed = self._calculate_viewshed()
            if viewshed is None:
                self.viewshed_enabled = False
                print("Viewshed: OFF (calculation failed)")
            else:
                print(f"Viewshed: ON ({self._viewshed_coverage:.1f}% coverage)")
                # Debug: verify viewshed cache
                if self._viewshed_cache is not None:
                    print(f"  Viewshed cache shape: {self._viewshed_cache.shape}")
                else:
                    print("  WARNING: Viewshed cache is None!")
        else:
            print("Viewshed: OFF")

        self._update_frame()

    def _clear_observer(self):
        """Clear the placed observer and viewshed."""
        self._observer_position = None
        self._viewshed_cache = None
        self.viewshed_enabled = False
        print("Observer cleared")
        self._update_frame()

    def _adjust_observer_elevation(self, delta):
        """Adjust observer elevation for viewshed calculation."""
        self.viewshed_observer_elev = max(0, self.viewshed_observer_elev + delta)
        print(f"Observer height: {self.viewshed_observer_elev:.0f}m")

        # Clear cache and recalculate viewshed if enabled
        if self.viewshed_enabled and self._observer_position is not None:
            self._viewshed_cache = None  # Clear cache to force recalculation
            self._calculate_viewshed()
            self._update_frame()

    def _save_screenshot(self):
        """Save current view as PNG image."""
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"rtxpy_screenshot_{timestamp}.png"

        # Pass viewshed data directly to render if enabled
        viewshed_data = None
        observer_pos = None
        if self.viewshed_enabled and self._viewshed_cache is not None:
            viewshed_data = self._viewshed_cache
            if self._observer_position is not None:
                observer_pos = self._observer_position

        # Render at full resolution for screenshot
        from .analysis import render as render_func
        img = render_func(
            self.raster,
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
        )

        # Convert from float [0-1] to uint8 [0-255]
        img_uint8 = (np.clip(img, 0, 1) * 255).astype(np.uint8)

        # Save using PIL or matplotlib
        try:
            from PIL import Image
            Image.fromarray(img_uint8).save(filename)
        except ImportError:
            import matplotlib.pyplot as plt
            plt.imsave(filename, img)

        print(f"Screenshot saved: {filename}")

    def _render_frame(self):
        """Render a frame using rtxpy."""
        from .analysis import render

        # Pass viewshed data directly to render if enabled
        viewshed_data = None
        observer_pos = None
        if self.viewshed_enabled:
            if self._viewshed_cache is not None:
                viewshed_data = self._viewshed_cache
                if self._observer_position is not None:
                    observer_pos = self._observer_position
            else:
                # Debug: viewshed enabled but no cache
                if self.frame_count % 100 == 0:  # Only print occasionally
                    print(f"[DEBUG] Viewshed enabled but cache is None")

        img = render(
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
        )

        return img

    def _update_frame(self):
        """Render and display a new frame."""
        img = self._render_frame()
        self.frame_count += 1

        # Update the image
        self.im.set_data(img)

        # Update title with position info
        pos = self.position
        title = f"Pos: ({pos[0]:.0f}, {pos[1]:.0f}, {pos[2]:.0f}) | Speed: {self.move_speed:.0f}"
        if self._observer_position is not None:
            obs_x, obs_y = self._observer_position
            title += f" | Observer: ({obs_x:.0f}, {obs_y:.0f}) @ {self.viewshed_observer_elev:.0f}m"
            if self.viewshed_enabled:
                title += f" | Coverage: {self._viewshed_coverage:.1f}%"
        self.ax.set_title(title, fontsize=10, color='white')

        # Update help text
        if self.show_help:
            self.help_text.set_visible(True)
        else:
            self.help_text.set_visible(False)

        self._update_minimap()

        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()

    def _handle_scroll(self, event):
        """Handle mouse scroll wheel for zoom."""
        if event.step > 0:
            self.fov = max(20, self.fov - 3)
        else:
            self.fov = min(120, self.fov + 3)
        print(f"FOV: {self.fov:.0f}")
        self._update_frame()

    def run(self, start_position: Optional[Tuple[float, float, float]] = None,
            look_at: Optional[Tuple[float, float, float]] = None):
        """
        Run the interactive viewer.

        Parameters
        ----------
        start_position : tuple, optional
            Starting camera position (x, y, z). If None, positions
            camera above terrain center.
        look_at : tuple, optional
            Initial look-at point. If None, looks toward terrain center.
        """
        import matplotlib
        import matplotlib.pyplot as plt

        # Check if we're in a Jupyter notebook and need to switch backends
        current_backend = matplotlib.get_backend().lower()
        in_notebook = False
        try:
            from IPython import get_ipython
            ipy = get_ipython()
            if ipy is not None and 'IPKernelApp' in ipy.config:
                in_notebook = True
        except (ImportError, AttributeError):
            pass

        # Warn if using a non-interactive backend
        non_interactive_backends = ['agg', 'module://matplotlib_inline.backend_inline', 'inline']
        if any(nb in current_backend for nb in non_interactive_backends):
            if in_notebook:
                print("\n" + "="*70)
                print("WARNING: Matplotlib is using a non-interactive backend.")
                print("Keyboard controls will NOT work with the inline backend.")
                print("\nTo fix, run this BEFORE calling explore():")
                print("    %matplotlib qt")
                print("  OR")
                print("    %matplotlib tk")
                print("  OR (if ipympl is installed):")
                print("    %matplotlib widget")
                print("\nThen restart the kernel and run the notebook again.")
                print("="*70 + "\n")
            else:
                print("WARNING: Non-interactive matplotlib backend detected.")
                print("Keyboard controls may not work.")

        # Disable default matplotlib keybindings that conflict with our controls
        for key in ['s', 'q', 'l', 'k', 'a', 'w', 'e', 'c', 'h', 't']:
            if key in plt.rcParams.get('keymap.save', []):
                plt.rcParams['keymap.save'].remove(key)
            if key in plt.rcParams.get('keymap.quit', []):
                plt.rcParams['keymap.quit'].remove(key)
            if key in plt.rcParams.get('keymap.xscale', []):
                plt.rcParams['keymap.xscale'].remove(key)
            if key in plt.rcParams.get('keymap.yscale', []):
                plt.rcParams['keymap.yscale'].remove(key)
        # Clear all default keymaps to avoid conflicts
        for param in list(plt.rcParams.keys()):
            if param.startswith('keymap.'):
                plt.rcParams[param] = []

        H, W = self.terrain_shape

        # Set initial move speed based on terrain extent (~1% of diagonal per keystroke)
        terrain_diagonal = np.sqrt(W**2 + H**2)
        if self.move_speed is None:
            self.move_speed = terrain_diagonal * 0.01

        # Default start position
        if start_position is None:
            start_position = (
                W / 2,
                -H * 0.3,
                self.elev_max + terrain_diagonal * 0.1
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
            center = np.array([W / 2, H / 2, self.elev_mean])
            direction = center - self.position
            direction = direction / (np.linalg.norm(direction) + 1e-8)
            self.yaw = np.degrees(np.arctan2(direction[1], direction[0]))
            self.pitch = np.degrees(np.arcsin(np.clip(direction[2], -1, 1)))

        # Create figure
        plt.ion()  # Interactive mode
        self.fig, self.ax = plt.subplots(1, 1, figsize=(self.width/100, self.height/100), dpi=100)
        self.fig.patch.set_facecolor('black')
        self.ax.set_facecolor('black')
        self.ax.axis('off')
        self.fig.subplots_adjust(left=0, right=1, top=0.95, bottom=0)

        # Render initial frame
        img = self._render_frame()
        self.im = self.ax.imshow(img, aspect='auto')

        # Help text
        help_str = (
            "WASD/Arrows: Move | Q/E: Up/Down | IJKL: Look | Scroll: Zoom | +/-: Speed\n"
            "G: Layers | N/P: Geometry | O: Place Observer | V: Toggle Viewshed | [/]: Height\n"
            "R/Shift+R: Resolution | Z/Shift+Z: Vert. Exag. | B: TIN/Voxel | Y: Stretch\n"
            "T: Shadows | C: Colormap | F: Screenshot | M: Minimap | H: Help | X: Exit"
        )
        self.help_text = self.ax.text(
            0.01, 0.02, help_str,
            transform=self.ax.transAxes,
            fontsize=8,
            color='white',
            alpha=0.8,
            verticalalignment='bottom',
            fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='black', alpha=0.5)
        )

        # Initialize minimap
        self._compute_minimap_background()
        self._create_minimap()

        # Connect event handlers
        self.fig.canvas.mpl_connect('key_press_event', self._handle_key_press)
        self.fig.canvas.mpl_connect('key_release_event', self._handle_key_release)
        self.fig.canvas.mpl_connect('scroll_event', self._handle_scroll)

        # Set up timer for smooth key repeat
        self._timer = self.fig.canvas.new_timer(interval=self._tick_interval)
        self._timer.add_callback(self._tick)
        self._timer.start()

        # Window title
        self.fig.canvas.manager.set_window_title('rtxpy Interactive Viewer')

        print(f"\nInteractive Viewer Started")
        print(f"  Window: {self.width}x{self.height}")
        print(f"  Render: {self.render_width}x{self.render_height} ({self.render_scale:.0%})")
        print(f"  Terrain: {W}x{H}, elevation {self.elev_min:.0f}m - {self.elev_max:.0f}m")
        print(f"\nPress H for controls, X or Esc to exit\n")

        self.running = True
        self._update_frame()

        # Keep window open until closed
        plt.show(block=True)

        # Clean up timer
        if self._timer is not None:
            self._timer.stop()
            self._timer = None

        print(f"Viewer closed after {self.frame_count} frames")


def explore(raster, width: int = 800, height: int = 600,
            render_scale: float = 0.5,
            start_position: Optional[Tuple[float, float, float]] = None,
            look_at: Optional[Tuple[float, float, float]] = None,
            key_repeat_interval: float = 0.05,
            rtx: 'RTX' = None,
            pixel_spacing_x: float = 1.0, pixel_spacing_y: float = 1.0,
            mesh_type: str = 'tin',
            overlay_layers: dict = None,
            color_stretch: str = 'linear'):
    """
    Launch an interactive terrain viewer.

    Uses matplotlib for display - no additional dependencies required.
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
        Starting camera position (x, y, z).
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

    Controls
    --------
    - W/Up: Move forward
    - S/Down: Move backward
    - A/Left: Strafe left
    - D/Right: Strafe right
    - Q/Page Up: Move up
    - E/Page Down: Move down
    - I/J/K/L: Look up/left/down/right
    - Scroll wheel: Zoom in/out (FOV)
    - +/=: Increase speed
    - -: Decrease speed
    - G: Cycle geometry layers (when meshes are placed)
    - N: Jump to next geometry in current layer
    - P: Jump to previous geometry in current layer
    - O: Place observer (for viewshed) at look-at point
    - V: Toggle viewshed overlay (teal glow shows visible terrain)
    - [/]: Decrease/increase observer height
    - R: Decrease terrain resolution (coarser, up to 8x subsample)
    - Shift+R: Increase terrain resolution (finer, down to 1x)
    - Z: Decrease vertical exaggeration
    - Shift+Z: Increase vertical exaggeration
    - B: Toggle mesh type (TIN / voxel)
    - Y: Cycle color stretch (linear, sqrt, cbrt, log)
    - T: Toggle shadows
    - C: Cycle colormap
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
    viewer = InteractiveViewer(
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
    )
    viewer.color_stretch = color_stretch
    if color_stretch in viewer._color_stretches:
        viewer._color_stretch_idx = viewer._color_stretches.index(color_stretch)
    viewer.run(start_position=start_position, look_at=look_at)
