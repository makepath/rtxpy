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
    - Click+Drag: Pan (slippy-map style)
    - Scroll wheel: Zoom in/out (FOV)
    - +/=: Increase speed
    - -: Decrease speed
    - G: Cycle terrain color (elevation → overlays)
    - U: Cycle basemap (none → satellite → osm → topo)
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
    - C: Cycle colormap
    - Shift+F: Fetch/toggle FIRMS fire layer (7d LANDSAT 30m)
    - Shift+W: Toggle wind particle animation
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
        self._basemap_options = ['none', 'satellite', 'osm', 'topo']
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
        self.sun_azimuth = 225.0
        self.sun_altitude = 35.0
        self.shadows = True
        self.ambient = 0.2
        self.colormap = 'gray'
        self.colormaps = ['gray', 'terrain', 'viridis', 'plasma', 'cividis']
        self.colormap_idx = 0
        self.color_stretch = 'linear'

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
        self._observer_position = None  # Fixed observer position (x, y) in terrain coords
        self._observer_drone_parts = None  # List of (verts, idxs, (r,g,b)) per sub-mesh
        self._observer_drone_placed = False  # Whether drone geometry is in the scene
        self._drone_mode = 'off'  # 'off' | '3rd' | 'fpv'
        self._saved_camera = None  # (position, yaw, pitch) before entering drone mode
        self._drone_yaw = 0.0     # Drone heading (for 3rd-person flight)
        self._drone_pitch = 0.0   # Drone pitch (for 3rd-person flight)

        # State
        self.running = False
        self.show_help = True
        self.show_minimap = True
        self.frame_count = 0
        self._last_title = None
        self._last_subtitle = None

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
        self._wind_max_age = 120      # Max lifetime before respawn
        self._wind_n_particles = 6000
        self._wind_trail_len = 20     # Number of trail positions to keep
        self._wind_trails = None      # (N, trail_len, 2) ring buffer of past positions
        self._wind_speed_mult = 50.0  # Velocity exaggeration for visibility
        self._wind_min_depth = 0.0    # Min camera distance to render (set in _init_wind)
        self._wind_dot_radius = 3     # Radius of each particle dot in screen pixels
        self._wind_alpha = 0.035      # Per-pixel alpha for particle dots

        # Held keys tracking for smooth simultaneous input
        self._held_keys = set()
        self._tick_interval = int(key_repeat_interval * 1000)  # Convert to ms for timer
        self._timer = None

        # Mouse drag state for slippy-map panning
        self._mouse_dragging = False
        self._mouse_last_x = None
        self._mouse_last_y = None

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

        # Build water mask from *full-resolution* base raster (not subsampled)
        # so it can be applied to full-resolution overlay layers.
        base_data = self._base_raster.data
        if hasattr(base_data, 'get'):
            base_np = base_data.get()
        else:
            base_np = np.asarray(base_data)
        floor_val = float(np.nanmin(base_np))
        floor_max = float(np.nanmax(base_np))
        eps = (floor_max - floor_val) * 1e-4 if floor_max > floor_val else 1e-6
        self._water_mask = (base_np <= floor_val + eps) | np.isnan(base_np)

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

            # Cache the initial terrain mesh at scale=1.0 (before VE)
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

        # Wind
        if self._wind_enabled:
            parts.append('wind')

        # Drone mode
        if self._drone_mode == 'fpv':
            parts.append('DRONE FPV')
        elif self._drone_mode == '3rd':
            parts.append('DRONE 3RD')

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
        H, W = sub.shape
        self.terrain_shape = (H, W)

        # 2. Update pixel spacing
        self.pixel_spacing_x = self._base_pixel_spacing_x * factor
        self.pixel_spacing_y = self._base_pixel_spacing_y * factor

        # 3. Build or retrieve cached terrain mesh
        ve = self.vertical_exaggeration
        cache_key = (factor, self.mesh_type)

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

        # 4. Replace terrain geometry (add_geometry overwrites existing key
        #    in-place, preserving dict insertion order and instance IDs)
        if self.rtx is not None:
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
            # Rebuild terrain layer order with new overlay names
            self._terrain_layer_order = ['elevation'] + list(self._overlay_names)
            if self._terrain_layer_idx >= len(self._terrain_layer_order):
                self._terrain_layer_idx = 0
            # Reset active overlay data if an overlay is selected
            terrain_name = self._terrain_layer_order[self._terrain_layer_idx]
            if terrain_name != 'elevation' and terrain_name in self._overlay_layers:
                self._active_overlay_data = self._overlay_layers[terrain_name]

        # 6. Re-snap placed meshes to new terrain surface
        if self.rtx is not None:
            for geom_id in self.rtx.list_geometries():
                if geom_id == 'terrain':
                    continue
                # Baked meshes — re-snap Z to new terrain surface + VE
                if hasattr(self, '_baked_meshes') and geom_id in self._baked_meshes:
                    baked_key = (factor, geom_id)
                    if baked_key in self._baked_mesh_cache:
                        scaled_v, orig_idx = self._baked_mesh_cache[baked_key]
                    else:
                        baked = self._baked_meshes[geom_id]
                        if len(baked) == 3:
                            orig_v, orig_idx, orig_base_z = baked
                        else:
                            orig_v, orig_idx = baked
                            orig_base_z = None
                        scaled_v = orig_v.copy()
                        if orig_base_z is not None:
                            # Sample new terrain Z at each vertex position
                            vx = orig_v[0::3]
                            vy = orig_v[1::3]
                            px = vx / self.pixel_spacing_x
                            py = vy / self.pixel_spacing_y
                            ix = np.clip(np.round(px).astype(int), 0, W - 1)
                            iy = np.clip(np.round(py).astype(int), 0, H - 1)
                            new_base_z = terrain_np[iy, ix].astype(np.float32)
                            new_base_z = np.where(np.isnan(new_base_z), 0.0, new_base_z)
                            z_offset = orig_v[2::3] - orig_base_z
                            scaled_v[2::3] = (new_base_z + z_offset) * ve
                        else:
                            scaled_v[2::3] *= ve
                        self._baked_mesh_cache[baked_key] = (scaled_v.copy(), orig_idx)
                    self.rtx.add_geometry(geom_id, scaled_v, orig_idx)
                    continue
                # Instanced meshes — update transform Z from terrain
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

        # 7. Recompute minimap
        self._compute_minimap_background()
        if self._minimap_im is not None:
            self._minimap_im.set_data(self._minimap_background)
            # Update minimap axes limits for new background size
            mm_h, mm_w = self._minimap_background.shape[:2]
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

        # Use cached mesh if available, otherwise build and cache
        cache_key = (self.subsample_factor, self.mesh_type)

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
        if self.rtx is not None:
            for geom_id in self.rtx.list_geometries():
                if geom_id == 'terrain':
                    continue
                # Baked meshes (merged buildings) — re-snap Z to terrain + VE
                if hasattr(self, '_baked_meshes') and geom_id in self._baked_meshes:
                    baked = self._baked_meshes[geom_id]
                    if len(baked) == 3:
                        orig_v, orig_idx, orig_base_z = baked
                    else:
                        orig_v, orig_idx = baked
                        orig_base_z = None
                    scaled_v = orig_v.copy()
                    if orig_base_z is not None:
                        # Sample current terrain Z at each vertex position
                        vx = orig_v[0::3]
                        vy = orig_v[1::3]
                        px = vx / self.pixel_spacing_x
                        py = vy / self.pixel_spacing_y
                        ix = np.clip(np.round(px).astype(int), 0, W - 1)
                        iy = np.clip(np.round(py).astype(int), 0, H - 1)
                        cur_base_z = terrain_np[iy, ix].astype(np.float32)
                        cur_base_z = np.where(np.isnan(cur_base_z), 0.0, cur_base_z)
                        z_offset = orig_v[2::3] - orig_base_z
                        scaled_v[2::3] = (cur_base_z + z_offset) * ve
                    else:
                        scaled_v[2::3] *= ve
                    self.rtx.add_geometry(geom_id, scaled_v, orig_idx)
                    continue
                # Instanced meshes — update transform Z from terrain
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
        mm_h, mm_w = self._minimap_background.shape[:2]
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
            spine.set_edgecolor('#555555')
            spine.set_linewidth(0.6)

        # Display RGBA background (origin='upper' so row 0 is top = +Y)
        self._minimap_ax.set_facecolor('#0a0c14')
        self._minimap_im = self._minimap_ax.imshow(
            self._minimap_background,
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
        mm_h, mm_w = self._minimap_background.shape[:2]
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
        grad_row, grad_col = np.gradient(terrain_np.astype(np.float32))
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
        r0 = np.clip(np.floor(rows).astype(int), 0, H - 2)
        c0 = np.clip(np.floor(cols).astype(int), 0, W - 2)
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

        # Advect
        pts[:, 0] += v_val  # row
        pts[:, 1] += u_val  # col

        # Age particles
        self._wind_ages += 1

        # Respawn out-of-bounds or aged-out particles
        oob = (pts[:, 0] < 0) | (pts[:, 0] >= H) | (pts[:, 1] < 0) | (pts[:, 1] >= W)
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

        Uses the same pinhole camera model as the ray tracer to project
        3D particle world positions onto 2D screen pixels, then draws
        semi-transparent white dots with short trails.

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
        pts = self._wind_particles  # (N, 2) — (row, col) in base-raster pixel coords

        # Camera basis matching the ray tracer
        cam_pos = self.position
        look_at = self._get_look_at()
        forward, right, cam_up = _compute_camera_basis(
            tuple(cam_pos), tuple(look_at), (0, 0, 1),
        )
        fov_scale = math.tan(math.radians(self.fov) / 2.0)
        aspect = sw / sh

        # Get terrain elevation for Z coordinate
        terrain_data = self.raster.data
        if hasattr(terrain_data, 'get'):
            terrain_np = terrain_data.get()
        else:
            terrain_np = np.asarray(terrain_data)
        tH, tW = terrain_np.shape

        # Convert particle pixel coords to world coords
        # Account for subsample factor: particles are in base raster coords
        f = self.subsample_factor
        psx = self._base_pixel_spacing_x
        psy = self._base_pixel_spacing_y

        def _project_points(rows, cols):
            """Project (row, col) in base-raster space to screen (sx, sy)."""
            # Sample terrain Z at subsampled resolution
            sr = np.clip((rows / f).astype(int), 0, tH - 1)
            sc = np.clip((cols / f).astype(int), 0, tW - 1)
            z_vals = terrain_np[sr, sc] * self.vertical_exaggeration
            # Hover particles well above terrain so they're clearly visible
            z_vals = z_vals + 3.0

            # World coordinates
            wx = cols * psx
            wy = rows * psy
            wz = z_vals

            # Camera-relative position
            dx = wx - cam_pos[0]
            dy = wy - cam_pos[1]
            dz = wz - cam_pos[2]

            # Project onto camera basis
            depth = dx * forward[0] + dy * forward[1] + dz * forward[2]
            u_cam = dx * right[0] + dy * right[1] + dz * right[2]
            v_cam = dx * cam_up[0] + dy * cam_up[1] + dz * cam_up[2]

            # Skip particles behind camera or too close
            valid = depth > self._wind_min_depth

            # NDC to screen
            u_ndc = np.where(valid, u_cam / (depth * fov_scale * aspect + 1e-10), -2)
            v_ndc = np.where(valid, v_cam / (depth * fov_scale + 1e-10), -2)

            sx = ((u_ndc + 1.0) * 0.5 * sw).astype(int)
            sy = (((1.0 - v_ndc)) * 0.5 * sh).astype(int)

            # Clip to screen bounds
            on_screen = valid & (sx >= 0) & (sx < sw) & (sy >= 0) & (sy < sh)
            return sx, sy, on_screen, depth

        # Precompute circular stamp offsets for fat dots
        r = self._wind_dot_radius
        offsets = []
        for dy in range(-r, r + 1):
            for dx in range(-r, r + 1):
                dist_sq = dx * dx + dy * dy
                if dist_sq <= r * r:
                    # Smooth circular falloff: 1 at centre, 0 at edge
                    falloff = 1.0 - (dist_sq / (r * r)) ** 0.5
                    offsets.append((dx, dy, falloff))

        # Draw trails — transparent lime, very subtle fat blobs
        color = np.array([0.3, 0.9, 0.8], dtype=np.float32)

        ages = self._wind_ages  # (N,)

        for t in range(self._wind_trail_len - 1, -1, -1):
            # Don't draw trail points that haven't had time to separate
            # (particle must be at least t ticks old for trail slot t)
            trail_pts = self._wind_trails[:, t, :]
            age_ok = ages > t
            sx, sy, on_screen, depth = _project_points(trail_pts[:, 0], trail_pts[:, 1])

            mask = on_screen & age_ok
            if not mask.any():
                continue

            # Smooth fade-in over the first 15 ticks, fade-out over last 30
            masked_ages = ages[mask]
            masked_lifetimes = self._wind_lifetimes[mask]
            fade_in = np.clip(masked_ages / 15.0, 0, 1)
            fade_out = np.clip((masked_lifetimes - masked_ages) / 30.0, 0, 1)
            fade_in = fade_in * fade_out

            sx_m = sx[mask]
            sy_m = sy[mask]

            for dx, dy, falloff in offsets:
                px = sx_m + dx
                py = sy_m + dy
                valid = (px >= 0) & (px < sw) & (py >= 0) & (py < sh)
                if not valid.any():
                    continue
                pxv = px[valid]
                pyv = py[valid]
                pixel_alpha = self._wind_alpha * falloff * fade_in[valid]
                img[pyv, pxv, :] = np.clip(
                    img[pyv, pxv, :] + np.expand_dims(pixel_alpha, 1) * color,
                    0, 1,
                )

        return img

    def _handle_key_press(self, event):
        """Handle key press - add to held keys or handle instant actions."""
        raw_key = event.key if event.key else ''
        key = raw_key.lower()

        # Drone mode cycle: Shift+O (before other keys)
        if raw_key == 'O':
            self._cycle_drone_mode()
            return

        # Snap camera to drone: Shift+V
        if raw_key == 'V':
            self._snap_to_drone()
            return

        # FIRMS fire layer: Shift+F (before 'f' screenshot)
        if raw_key == 'F':
            self._toggle_firms()
            return

        # Wind toggle: Shift+W (before movement keys capture 'w')
        if raw_key == 'W':
            self._toggle_wind()
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

        # Viewshed controls
        elif key == 'o':
            if self._drone_mode == 'off':
                self._place_observer()
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

        # Toggle mesh type (tin ↔ voxel)
        elif key == 'b':
            self.mesh_type = 'voxel' if self.mesh_type == 'tin' else 'tin'
            self._rebuild_vertical_exaggeration(self.vertical_exaggeration)
            print(f"Mesh type: {self.mesh_type}")

        # Basemap cycling: U = cycle none → satellite → osm → topo → none
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

    def _get_drone_front(self):
        """Get forward direction for drone flight (uses drone yaw/pitch)."""
        yaw_rad = np.radians(self._drone_yaw)
        pitch_rad = np.radians(self._drone_pitch)
        return np.array([
            np.cos(yaw_rad) * np.cos(pitch_rad),
            np.sin(yaw_rad) * np.cos(pitch_rad),
            np.sin(pitch_rad)
        ], dtype=np.float32)

    def _get_drone_right(self):
        """Get right direction for drone flight."""
        front = self._get_drone_front()
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

    def _sync_drone_from_pos(self, pos):
        """Update observer position and drone mesh from a 3D position."""
        pos = self._clamp_drone_pos(pos)
        self._observer_position = (float(pos[0]), float(pos[1]))
        self.viewshed_observer_elev = float(pos[2]) - self._get_terrain_z(
            pos[0], pos[1])
        if self.viewshed_observer_elev < 0:
            self.viewshed_observer_elev = 0.0
        self._update_observer_drone()

        # Dynamically recalculate viewshed as the drone moves (throttled)
        if self.viewshed_enabled:
            now = time.monotonic()
            if now - self._last_viewshed_time >= self._viewshed_recalc_interval:
                self._last_viewshed_time = now
                self._viewshed_cache = None
                self._calculate_viewshed(quiet=True)

    def _tick(self):
        """Continuous render loop — process held keys and redraw (called by timer)."""
        if not self.running:
            return

        # Process held movement / look keys
        if self._held_keys:
            if self._drone_mode == '3rd' and self._observer_drone_placed:
                # --- 3rd-person: WASD/IJKL fly the drone, camera stays ---
                front = self._get_drone_front()
                right = self._get_drone_right()

                # Current drone 3D position
                obs_x, obs_y = self._observer_position
                terrain_z = self._get_terrain_z(obs_x, obs_y)
                drone_pos = np.array([obs_x, obs_y,
                                      terrain_z + self.viewshed_observer_elev],
                                     dtype=float)

                if 'w' in self._held_keys or 'up' in self._held_keys:
                    drone_pos += front * self.move_speed
                if 's' in self._held_keys or 'down' in self._held_keys:
                    drone_pos -= front * self.move_speed
                if 'a' in self._held_keys or 'left' in self._held_keys:
                    drone_pos -= right * self.move_speed
                if 'd' in self._held_keys or 'right' in self._held_keys:
                    drone_pos += right * self.move_speed
                if 'q' in self._held_keys or 'pageup' in self._held_keys:
                    drone_pos[2] += self.move_speed
                if 'e' in self._held_keys or 'pagedown' in self._held_keys:
                    drone_pos[2] -= self.move_speed

                if 'i' in self._held_keys:
                    self._drone_pitch = min(89, self._drone_pitch + self.look_speed)
                if 'k' in self._held_keys:
                    self._drone_pitch = max(-89, self._drone_pitch - self.look_speed)
                if 'j' in self._held_keys:
                    self._drone_yaw -= self.look_speed
                if 'l' in self._held_keys:
                    self._drone_yaw += self.look_speed

                self._sync_drone_from_pos(drone_pos)

            else:
                # --- Normal / FPV: WASD moves camera ---
                front = self._get_front()
                right = self._get_right()

                if 'w' in self._held_keys or 'up' in self._held_keys:
                    self.position += front * self.move_speed
                if 's' in self._held_keys or 'down' in self._held_keys:
                    self.position -= front * self.move_speed
                if 'a' in self._held_keys or 'left' in self._held_keys:
                    self.position -= right * self.move_speed
                if 'd' in self._held_keys or 'right' in self._held_keys:
                    self.position += right * self.move_speed
                if 'q' in self._held_keys or 'pageup' in self._held_keys:
                    cam_up = np.cross(front, right)
                    cam_up /= (np.linalg.norm(cam_up) + 1e-8)
                    self.position += cam_up * self.move_speed
                if 'e' in self._held_keys or 'pagedown' in self._held_keys:
                    cam_up = np.cross(front, right)
                    cam_up /= (np.linalg.norm(cam_up) + 1e-8)
                    self.position -= cam_up * self.move_speed

                if 'i' in self._held_keys:
                    self.pitch = min(89, self.pitch + self.look_speed)
                if 'k' in self._held_keys:
                    self.pitch = max(-89, self.pitch - self.look_speed)
                if 'j' in self._held_keys:
                    self.yaw -= self.look_speed
                if 'l' in self._held_keys:
                    self.yaw += self.look_speed

                # In FPV, sync drone to camera
                if self._drone_mode == 'fpv' and self._observer_drone_placed:
                    self._sync_drone_from_pos(self.position)

        self._update_frame()

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
        """Cycle basemap: none → satellite → osm → topo → none.

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

    def _update_observer_drone(self):
        """Place or update the drone mesh at the observer position."""
        if self._observer_position is None or self.rtx is None:
            return

        from .mesh import make_transform

        # Lazy-load drone parts once
        if self._observer_drone_parts is None:
            self._observer_drone_parts = self._load_drone_parts()
            if not self._observer_drone_parts:
                return

        obs_x, obs_y = self._observer_position
        terrain_z = self._get_terrain_z(obs_x, obs_y)
        obs_z = terrain_z + self.viewshed_observer_elev

        # Scale drone to ~0.05× pixel_spacing so it's visible but not huge
        drone_scale = 0.0125 * max(self.pixel_spacing_x, self.pixel_spacing_y)

        transform = make_transform(x=obs_x, y=obs_y, z=obs_z, scale=drone_scale)

        for i, (verts, idxs, color) in enumerate(self._observer_drone_parts):
            gid = f'_observer_{i}'
            if self._observer_drone_placed:
                self.rtx.update_transform(gid, transform)
            else:
                self.rtx.add_geometry(gid, verts, idxs, transform=transform)

        # Set geometry colors (needs the accessor's color dict)
        if not self._observer_drone_placed:
            builder = getattr(self, '_geometry_colors_builder', None)
            if builder is not None:
                # Access the accessor's _geometry_colors dict via the builder's __self__
                acc = getattr(builder, '__self__', None)
                if acc is not None and hasattr(acc, '_geometry_colors'):
                    for i, (_, _, color) in enumerate(self._observer_drone_parts):
                        acc._geometry_colors[f'_observer_{i}'] = color
                    acc._geometry_colors_dirty = True

        self._observer_drone_placed = True

    def _set_drone_visibility(self, visible):
        """Show or hide all drone sub-mesh geometries."""
        if self._observer_drone_placed and self.rtx is not None:
            for i in range(len(self._observer_drone_parts or [])):
                self.rtx.set_geometry_visible(f'_observer_{i}', visible)

    def _cycle_drone_mode(self):
        """Cycle drone control mode: off → 3rd person → FPV → off (Shift+O).

        off:
          Normal camera control. Drone is visible at observer position.
        3rd person:
          Camera stays fixed. WASD/IJKL fly the drone. Watch it move.
        FPV:
          Camera = drone. WASD flies both. Drone mesh hidden.
        """
        if self._observer_position is None:
            print("No observer placed. Press O first.")
            return

        if self._drone_mode == 'off':
            # --- Enter 3rd person ---
            # Save camera so we can restore on full exit
            self._saved_camera = (
                self.position.copy(),
                float(self.yaw),
                float(self.pitch),
            )
            # Initialise drone heading from camera yaw
            self._drone_yaw = float(self.yaw)
            self._drone_pitch = 0.0
            self._drone_mode = '3rd'
            print("Drone 3RD PERSON: ON  (WASD flies drone, Shift+O → FPV)")

        elif self._drone_mode == '3rd':
            # --- 3rd person → FPV ---
            obs_x, obs_y = self._observer_position
            terrain_z = self._get_terrain_z(obs_x, obs_y)
            obs_z = terrain_z + self.viewshed_observer_elev
            self.position = np.array([obs_x, obs_y, obs_z], dtype=float)
            self.yaw = self._drone_yaw
            self.pitch = self._drone_pitch
            # Hide drone mesh (you are the drone)
            self._set_drone_visibility(False)
            self._drone_mode = 'fpv'
            print("Drone FPV: ON  (WASD flies camera+drone, Shift+O → exit)")

        else:
            # --- FPV → off ---
            # Sync final drone position from camera
            self._sync_drone_from_pos(self.position)
            # Show drone mesh again
            self._set_drone_visibility(True)
            # Restore saved external camera
            if self._saved_camera is not None:
                self.position = self._saved_camera[0]
                self.yaw = self._saved_camera[1]
                self.pitch = self._saved_camera[2]
                self._saved_camera = None
            self._drone_mode = 'off'
            print("Drone mode: OFF")

        self._update_frame()

    def _snap_to_drone(self):
        """Snap external camera to look at the drone from nearby (Shift+V)."""
        if self._observer_position is None:
            print("No observer placed. Press O first.")
            return
        if self._drone_mode == 'fpv':
            # Already in FPV — nothing to snap to
            return

        obs_x, obs_y = self._observer_position
        terrain_z = self._get_terrain_z(obs_x, obs_y)
        obs_z = terrain_z + self.viewshed_observer_elev

        # Place camera a short distance behind and above the drone
        spacing = max(self.pixel_spacing_x, self.pixel_spacing_y)
        offset = spacing * 8.0  # 8 pixels back
        # Direction from drone to current camera (keep the viewing angle)
        dx = self.position[0] - obs_x
        dy = self.position[1] - obs_y
        dist_xy = np.sqrt(dx * dx + dy * dy)
        if dist_xy > 1e-6:
            dx /= dist_xy
            dy /= dist_xy
        else:
            # Camera is right on top of drone — pick an arbitrary direction
            dx, dy = 1.0, 0.0

        self.position = np.array([
            obs_x + dx * offset,
            obs_y + dy * offset,
            obs_z + spacing * 3.0,  # a bit above
        ], dtype=float)

        # Point camera at the drone
        to_drone = np.array([obs_x - self.position[0],
                             obs_y - self.position[1],
                             obs_z - self.position[2]])
        to_drone /= (np.linalg.norm(to_drone) + 1e-8)
        self.yaw = float(np.degrees(np.arctan2(to_drone[1], to_drone[0])))
        self.pitch = float(np.degrees(np.arcsin(np.clip(to_drone[2], -1, 1))))

        print(f"Snapped to drone at ({obs_x:.0f}, {obs_y:.0f})")
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

        # Store in world coordinates
        self._observer_position = (obs_x, obs_y)

        # Place/update drone mesh at observer position
        self._update_observer_drone()

        # Also compute pixel indices for display
        px_x = int(obs_x / self.pixel_spacing_x)
        px_y = int(obs_y / self.pixel_spacing_y)

        print(f"Observer placed at world ({obs_x:.0f}, {obs_y:.0f}), pixel ({px_x}, {px_y})")
        print(f"  Height: {self.viewshed_observer_elev:.3f} above terrain")
        print(f"  Press V to toggle viewshed, [/] to adjust height")

        # If viewshed is already enabled, recalculate
        if self.viewshed_enabled:
            self._calculate_viewshed()

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

        if self._observer_position is None:
            if not quiet:
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

            # Always hide the drone so it doesn't block its own viewshed
            if self._observer_drone_placed and self._observer_drone_parts:
                for i in range(len(self._observer_drone_parts)):
                    gid = f'_observer_{i}'
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
        # Exit drone mode if active (restore external camera)
        if self._drone_mode != 'off':
            if self._drone_mode == 'fpv':
                self._set_drone_visibility(True)
            if self._saved_camera is not None:
                self.position = self._saved_camera[0]
                self.yaw = self._saved_camera[1]
                self.pitch = self._saved_camera[2]
                self._saved_camera = None
            self._drone_mode = 'off'

        self._observer_position = None
        self._viewshed_cache = None
        self.viewshed_enabled = False

        # Remove all drone sub-mesh geometries from scene
        if self._observer_drone_placed and self.rtx is not None:
            n = len(self._observer_drone_parts) if self._observer_drone_parts else 0
            builder = getattr(self, '_geometry_colors_builder', None)
            acc = getattr(builder, '__self__', None) if builder else None
            for i in range(n):
                gid = f'_observer_{i}'
                self.rtx.remove_geometry(gid)
                if acc is not None and hasattr(acc, '_geometry_colors'):
                    acc._geometry_colors.pop(gid, None)
            if acc is not None and hasattr(acc, '_geometry_colors_dirty'):
                acc._geometry_colors_dirty = True
            self._observer_drone_placed = False

        print("Observer cleared")
        self._update_frame()

    def _adjust_observer_elevation(self, delta):
        """Adjust observer elevation for viewshed calculation."""
        self.viewshed_observer_elev = max(0, self.viewshed_observer_elev + delta)
        print(f"Observer height: {self.viewshed_observer_elev:.3f}")

        # Update drone position to match new elevation
        self._update_observer_drone()

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
        if self._observer_position is not None:
            observer_pos = self._observer_position
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
            color_range=self._land_color_range,
            rgb_texture=rgb_texture,
            overlay_data=self._active_overlay_data,
            overlay_alpha=self._overlay_alpha,
            geometry_colors=geometry_colors,
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

        # Always show observer orb when placed; viewshed overlay only when enabled
        viewshed_data = None
        observer_pos = None
        if self._observer_position is not None:
            observer_pos = self._observer_position
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
            color_range=self._land_color_range,
            rgb_texture=rgb_texture,
            overlay_data=self._active_overlay_data,
            overlay_alpha=self._overlay_alpha,
            geometry_colors=geometry_colors,
        )

        return img

    def _update_frame(self):
        """Render and display a new frame."""
        img = self._render_frame()
        self.frame_count += 1

        # Wind particle overlay
        if self._wind_enabled and self._wind_particles is not None:
            self._update_wind_particles()
            img = img.copy()  # Don't modify cached render
            self._draw_wind_on_frame(img)

        # Update the image
        self.im.set_data(img)

        # Update suptitle with map info (only when changed)
        title = self._build_title()
        if title != self._last_title:
            self._suptitle.set_text(title)
            self._last_title = title

        # Update subtitle with camera / observer info (only when changed)
        pos = self.position
        sub = f"Pos: ({pos[0]:.0f}, {pos[1]:.0f}, {pos[2]:.0f})  Speed: {self.move_speed:.0f}"
        if self._observer_position is not None:
            obs_x, obs_y = self._observer_position
            sub += f"  \u2502  Observer: ({obs_x:.0f}, {obs_y:.0f}) h={self.viewshed_observer_elev:.3f}"
            if self.viewshed_enabled:
                sub += f"  Coverage: {self._viewshed_coverage:.1f}%"
        if sub != self._last_subtitle:
            self.ax.set_title(sub, fontsize=8, color='#aaaaaa', pad=2)
            self._last_subtitle = sub

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

    def _handle_mouse_press(self, event):
        """Start drag on left-click inside the main axes."""
        if event.inaxes != self.ax:
            return
        if event.button == 1:
            self._mouse_dragging = True
            self._mouse_last_x = event.x
            self._mouse_last_y = event.y

    def _handle_mouse_release(self, event):
        """End drag on any button release."""
        self._mouse_dragging = False

    def _handle_mouse_motion(self, event):
        """Pan camera on mouse drag (slippy-map style)."""
        if not self._mouse_dragging or self._mouse_last_x is None:
            return
        if event.x is None or event.y is None:
            self._mouse_dragging = False
            return
        # Cancel drag if mouse left the axes (missed release event)
        if event.inaxes != self.ax:
            self._mouse_dragging = False
            return
        # Cancel drag if no button is held (missed release event)
        if hasattr(event, 'button') and event.button is None:
            self._mouse_dragging = False
            return

        dx = event.x - self._mouse_last_x
        dy = event.y - self._mouse_last_y
        self._mouse_last_x = event.x
        self._mouse_last_y = event.y

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
        # Drag forward (mouse moves up, dy > 0) → camera moves forward
        self.position -= right * dx * sensitivity
        self.position -= front_horiz * dy * sensitivity

        self._update_frame()

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

        # Create figure (suppress the matplotlib navigation toolbar)
        old_toolbar = plt.rcParams.get('toolbar', 'toolbar2')
        plt.rcParams['toolbar'] = 'None'
        plt.ion()  # Interactive mode
        self.fig, self.ax = plt.subplots(1, 1, figsize=(self.width/100, self.height/100), dpi=100)
        plt.rcParams['toolbar'] = old_toolbar
        self.fig.patch.set_facecolor('black')
        self.ax.set_facecolor('black')
        self.ax.axis('off')
        self.fig.subplots_adjust(left=0, right=1, top=0.92, bottom=0)

        # Main title (map info, updated each frame)
        self._suptitle = self.fig.suptitle(
            self._build_title(), fontsize=11, color='white',
            fontweight='bold', y=0.98,
        )

        # Render initial frame
        img = self._render_frame()
        self.im = self.ax.imshow(img, aspect='auto')

        # Help text — vertical list down the left side
        help_str = (
            "MOVEMENT\n"
            "  W/S/A/D  Move forward/back/left/right\n"
            "  Arrows   Move forward/back/left/right\n"
            "  Q / E    Move up / down\n"
            "  I/J/K/L  Look up/left/down/right\n"
            "  Drag     Pan (slippy-map)\n"
            "  Scroll   Zoom (FOV)\n"
            "  + / -    Speed up / down\n"
            "\n"
            "TERRAIN\n"
            "  G        Cycle terrain layer\n"
            "  U        Cycle basemap\n"
            "  C        Cycle colormap\n"
            "  Y        Cycle color stretch\n"
            "  , / .    Overlay alpha down / up\n"
            "  R        Decrease resolution\n"
            "  Shift+R  Increase resolution\n"
            "  Z        Decrease vert. exag.\n"
            "  Shift+Z  Increase vert. exag.\n"
            "  B        Toggle TIN / Voxel\n"
            "  T        Toggle shadows\n"
            "\n"
            "GEOMETRY\n"
            "  N        Cycle geometry layer\n"
            "  P        Prev geometry in group\n"
            "\n"
            "DRONE / VIEWSHED\n"
            "  O        Place observer\n"
            "  Shift+O  Drone mode (3rd/FPV)\n"
            "  Shift+V  Snap camera to drone\n"
            "  V        Toggle viewshed\n"
            "  [ / ]    Observer height down / up\n"
            "\n"
            "DATA LAYERS\n"
            "  Shift+F  FIRMS fire (7d)\n"
            "  Shift+W  Toggle wind\n"
            "\n"
            "OTHER\n"
            "  F        Screenshot\n"
            "  M        Toggle minimap\n"
            "  H        Toggle this help\n"
            "  X / Esc  Exit"
        )
        self.help_text = self.ax.text(
            0.01, 0.98, help_str,
            transform=self.ax.transAxes,
            fontsize=11,
            color='white',
            alpha=0.9,
            verticalalignment='top',
            fontfamily='monospace',
            bbox=dict(boxstyle='round,pad=0.6', facecolor='black', alpha=0.6)
        )

        # Initialize minimap
        self._compute_minimap_background()
        self._create_minimap()

        # Connect event handlers
        self.fig.canvas.mpl_connect('key_press_event', self._handle_key_press)
        self.fig.canvas.mpl_connect('key_release_event', self._handle_key_release)
        self.fig.canvas.mpl_connect('scroll_event', self._handle_scroll)
        self.fig.canvas.mpl_connect('button_press_event', self._handle_mouse_press)
        self.fig.canvas.mpl_connect('button_release_event', self._handle_mouse_release)
        self.fig.canvas.mpl_connect('motion_notify_event', self._handle_mouse_motion)

        # Set up timer for smooth key repeat
        self._timer = self.fig.canvas.new_timer(interval=self._tick_interval)
        self._timer.add_callback(self._tick)
        self._timer.start()

        # Window title bar
        self.fig.canvas.manager.set_window_title(f'rtxpy \u2014 {self._title}')

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

        # Clean up tile service
        if self._tile_service is not None:
            self._tile_service.shutdown()

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
            color_stretch: str = 'linear',
            title: str = None,
            tile_service=None,
            geometry_colors_builder=None,
            baked_meshes=None,
            subsample: int = 1,
            wind_data=None,
            accessor=None):
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
    accessor : RTXAccessor, optional
        RTX accessor instance for on-demand data fetching (e.g. FIRMS fire
        layer via Shift+F).
    wind_data : dict, optional
        Wind data from ``fetch_wind()``. If provided, Shift+W toggles
        wind particle animation.

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
    - U: Cycle basemap (none → satellite → osm → topo)
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
    - C: Cycle colormap
    - Shift+F: Fetch/toggle FIRMS fire layer (7d LANDSAT 30m)
    - Shift+W: Toggle wind particle animation
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
        title=title,
        subsample=subsample,
    )
    viewer._geometry_colors_builder = geometry_colors_builder
    viewer._baked_meshes = baked_meshes or {}
    viewer._accessor = accessor
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

    # Initial state: everything off except elevation
    viewer._tiles_enabled = False
    viewer._basemap_idx = 0  # 'none'
    viewer._geometry_layer_idx = 0  # 'none'
    if rtx is not None:
        for geom_id in viewer._all_geometries:
            if geom_id != 'terrain':
                rtx.set_geometry_visible(geom_id, False)
    viewer.run(start_position=start_position, look_at=look_at)
