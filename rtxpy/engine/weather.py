"""Wind, cloud, rain, and FIRMS weather subsystem for InteractiveViewer."""

import math

import numpy as np

from ..rtx import has_cupy
from .helpers import _add_overlay

if has_cupy:
    import cupy as cp
    from numba import cuda
    from .constants import _wind_splat_kernel, _rain_splat_kernel


class WeatherManager:
    """Wind particles, cloud/rain systems, FIRMS fire layer, and weather
    overlay methods.

    Accesses viewer state via ``self.v`` (back-reference to InteractiveViewer).
    """

    def __init__(self, viewer):
        self.v = viewer

    # ------------------------------------------------------------------
    # Wind particle animation
    # ------------------------------------------------------------------

    def _toggle_wind(self):
        """Toggle wind particle animation on/off."""
        v = self.v
        if v._wind_data is None:
            print("No wind data loaded. Pass wind_data to explore().")
            return
        v._wind_enabled = not v._wind_enabled
        print(f"Wind particles: {'ON' if v._wind_enabled else 'OFF'}")
        v._update_frame()

    def _toggle_firms(self):
        """Fetch and toggle NASA FIRMS LANDSAT fire footprints (Shift+F)."""
        v = self.v
        if v._accessor is None:
            print("No accessor available for FIRMS fire layer.")
            return

        if not v._firms_loaded:
            # First press: fetch + place
            print("Fetching FIRMS fire data (7d LANDSAT)...")
            try:
                from ..remote_data import fetch_firms
                from ..tiles import _build_latlon_grids
                import warnings

                # Get WGS84 bounds from the raster
                lats, lons = _build_latlon_grids(v._base_raster)
                bounds = (
                    float(lons.min()), float(lats.min()),
                    float(lons.max()), float(lats.max()),
                )

                # Detect CRS for reprojection
                crs = None
                try:
                    raster_crs = v._base_raster.rio.crs
                    if raster_crs is not None and not raster_crs.is_geographic:
                        crs = str(raster_crs)
                except (AttributeError, ImportError):
                    pass

                fire_data = fetch_firms(bounds, date_span='7d', crs=crs)

                n_fires = len(fire_data.get('features', []))
                if n_fires == 0:
                    print("No fire detections in the last 7 days.")
                    v._firms_loaded = True
                    return

                with warnings.catch_warnings():
                    warnings.filterwarnings(
                        "ignore", message="place_geojson called before")
                    v._accessor.place_geojson(
                        fire_data,
                        height=max(v.pixel_spacing_x,
                                   v.pixel_spacing_y) * 0.5,
                        geometry_id='fire',
                        color=(1.0, 0.25, 0.0, 3.0),
                        extrude=True,
                        merge=True,
                    )

                v._firms_loaded = True
                v._firms_visible = True

                # Ensure geometry color builder is active
                if (v._geometry_colors_builder is None
                        and v._accessor._geometry_colors):
                    v._geometry_colors_builder = (
                        v._accessor._build_geometry_colors_gpu)

                # Refresh geometry layer tracking
                if v.rtx is not None:
                    from ..viewer.terrain_lod import is_terrain_lod_gid
                    v._all_geometries = v.rtx.list_geometries()
                    groups = set()
                    for g in v._all_geometries:
                        if is_terrain_lod_gid(g):
                            continue
                        parts = g.rsplit('_', 1)
                        if len(parts) == 2 and parts[1].isdigit():
                            base = parts[0]
                        else:
                            base = g
                        if base != 'terrain':
                            groups.add(base)
                    v._geometry_layer_order = (
                        ['none', 'all'] + sorted(groups))

                print(f"FIRMS fire: ON  ({n_fires} detections)")
                v._update_frame()

            except Exception as e:
                print(f"FIRMS fire fetch failed: {e}")
            return

        # Subsequent presses: toggle visibility
        v._firms_visible = not v._firms_visible
        if v.rtx is not None:
            for geom_id in v.rtx.list_geometries():
                if geom_id.startswith('fire'):
                    v.rtx.set_geometry_visible(
                        geom_id, v._firms_visible)
        print(f"FIRMS fire: {'ON' if v._firms_visible else 'OFF'}")
        v._update_frame()

    def _init_weather(self, weather_data):
        """Interpolate weather variables from lat/lon grid onto terrain pixels.

        Registers each variable (temperature, precipitation, cloud_cover,
        pressure) as an overlay layer accessible via G-key cycling.
        """
        v = self.v
        from ..tiles import _build_latlon_grids
        from scipy.interpolate import RegularGridInterpolator

        raster = v._base_raster
        H, W = raster.shape

        lats_grid, lons_grid = _build_latlon_grids(raster)
        points = np.stack([lats_grid.ravel(), lons_grid.ravel()], axis=-1)

        w_lats = weather_data['lats']   # (ny,)
        w_lons = weather_data['lons']   # (nx,)

        variables = ['temperature', 'precipitation', 'cloud_cover', 'pressure']
        units = {'temperature': '\u00b0C', 'precipitation': 'mm',
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
            _add_overlay(v, var, layer)
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
        v = self.v
        if weather_data is None:
            return

        cloud_cover = weather_data.get('cloud_cover')
        precipitation = weather_data.get('precipitation')
        if cloud_cover is None and precipitation is None:
            return

        from ..tiles import _build_latlon_grids
        from scipy.interpolate import RegularGridInterpolator

        raster = v._base_raster
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
            v._cloud_cover_grid = np.clip(cc / 100.0, 0, 1)
        else:
            v._cloud_cover_grid = np.zeros((H, W), dtype=np.float32)

        # Interpolate precipitation (mm) to terrain grid
        if precipitation is not None:
            interp_p = RegularGridInterpolator(
                (w_lats, w_lons), precipitation.astype(np.float32),
                method='linear', bounds_error=False, fill_value=0.0,
            )
            v._rain_grid = interp_p(points).reshape(H, W).astype(np.float32)
        else:
            v._rain_grid = np.zeros((H, W), dtype=np.float32)

        # Cloud altitude: above terrain max
        terrain_data = raster.data
        if hasattr(terrain_data, 'get'):
            terrain_np = terrain_data.get()
        else:
            terrain_np = np.asarray(terrain_data)
        psx = v._base_pixel_spacing_x
        psy = v._base_pixel_spacing_y
        z_max = float(np.nanmax(terrain_np))
        z_range = z_max - float(np.nanmin(terrain_np))
        diag = np.sqrt((W * psx) ** 2 + (H * psy) ** 2)
        v._cloud_altitude = z_max + max(z_range * 0.5, diag * 0.04)
        v._cloud_min_depth = diag * 0.01
        v._cloud_thickness = max(z_range * 0.3, diag * 0.02)
        v._volumetric_clouds_enabled = True

        # Spawn cloud particles weighted by cloud_cover density
        N = v._cloud_n_particles
        cc_flat = v._cloud_cover_grid.ravel()
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
        v._cloud_particles = np.column_stack([rows, cols]).astype(np.float32)

        # Per-particle size (world-space radius)
        base_size = max(psx, psy) * 8
        v._cloud_sizes = (
            np.random.uniform(0.6, 1.8, N).astype(np.float32) * base_size
        )

        # Per-particle alpha from local cloud_cover
        r_idx = np.clip(rows.astype(int), 0, H - 1)
        c_idx = np.clip(cols.astype(int), 0, W - 1)
        local_cc = v._cloud_cover_grid[r_idx, c_idx]
        v._cloud_alphas = (local_cc * 0.35).astype(np.float32)

        # Ages / lifetimes
        v._cloud_max_age = 300
        v._cloud_lifetimes = np.random.randint(
            v._cloud_max_age // 2, v._cloud_max_age, N)
        v._cloud_ages = np.random.randint(0, v._cloud_max_age, N)

        # Rain particles — fewer, driven by precipitation
        rain_N = 8000
        from .particles.system import ParticleSystem
        v._rain_system = ParticleSystem(
            n=rain_N, grid_shape=(H, W), max_age=40,
            trail_len=0, weight_grid=v._rain_grid,
        )
        v._rain_z_frac = np.random.uniform(0.7, 1.0, rain_N).astype(np.float32)
        # Compatibility aliases
        v._rain_particles = v._rain_system.positions
        v._rain_ages = v._rain_system.ages
        v._rain_lifetimes = v._rain_system.lifetimes
        v._rain_max_age = 40

        mean_cc = float(np.mean(v._cloud_cover_grid) * 100)
        mean_precip = float(np.mean(v._rain_grid))
        print(f"  Clouds: {N} particles (mean cover {mean_cc:.0f}%)"
              f"  Rain: {rain_N} particles (mean precip {mean_precip:.1f}mm)")

    def _toggle_clouds(self):
        """Toggle cloud fog + rain on/off."""
        v = self.v
        if v._cloud_cover_grid is None:
            print("No weather data loaded. Pass weather_data to explore().")
            return
        v._clouds_enabled = not v._clouds_enabled
        if v._clouds_enabled and v._cloud_cover_grid is not None:
            cc = v._cloud_cover_grid
            print(f"Clouds: ON  (cover min={cc.min():.2f} max={cc.max():.2f} mean={cc.mean():.2f})")
        else:
            print(f"Clouds: OFF")
        v._render_needed = True

    def _action_toggle_clouds(self):
        self._toggle_clouds()

    def _update_rain_particles(self):
        """Animate rain streaks: fall downward and respawn."""
        v = self.v
        system = getattr(v, '_rain_system', None)
        if system is None:
            return
        from .particles.rain import advect_rain, respawn_rain
        v._rain_z_frac = advect_rain(
            system, v._rain_z_frac,
            wind_u=v._wind_u_px, wind_v=v._wind_v_px,
            dt_scale=getattr(v, '_dt_scale', 1.0),
        )
        v._rain_z_frac = respawn_rain(system, v._rain_z_frac, v._rain_grid)

    def _draw_rain_on_frame(self, img, forward, right, cam_up,
                            fov_scale, aspect, min_depth):
        """Render rain streaks on the frame."""
        v = self.v
        system = getattr(v, '_rain_system', None)
        if system is None:
            return
        from .particles.rain import render_rain_cpu
        from .particles.project import get_camera_basis
        cam = get_camera_basis(v.position, v._get_look_at(), v.fov,
                               img.shape[1], img.shape[0])
        # Cached terrain for z lookup
        if v._cloud_terrain_np is None:
            td = v.raster.data
            v._cloud_terrain_np = td.get() if hasattr(td, 'get') else np.asarray(td)
        render_rain_cpu(system, v._rain_z_frac, img, cam,
                        v._cloud_terrain_np,
                        v._base_pixel_spacing_x, v._base_pixel_spacing_y,
                        v.vertical_exaggeration, v.subsample_factor,
                        v._cloud_altitude, v._cloud_min_depth,
                        rain_grid=v._rain_grid)

    def _splat_rain_gpu(self, d_frame):
        """Project and splat rain particles as vertical streaks on GPU."""
        v = self.v
        system = getattr(v, '_rain_system', None)
        if system is None:
            return
        rain_grid = getattr(v, '_rain_grid', None)
        if rain_grid is None or rain_grid.sum() < 0.01:
            return

        from ..analysis.render import _compute_camera_basis
        from .particles.rain import prepare_rain_gpu_buffers

        sh, sw = d_frame.shape[:2]
        psx = float(v._base_pixel_spacing_x)
        psy = float(v._base_pixel_spacing_y)
        ve = float(v.vertical_exaggeration)
        cloud_z = float(v._cloud_altitude * ve)
        min_depth = float(v._cloud_min_depth)

        cam_pos = v.position
        look_at = v._get_look_at()
        forward, right, cam_up = _compute_camera_basis(
            tuple(cam_pos), tuple(look_at), (0, 0, 1),
        )
        fov_scale = float(math.tan(math.radians(v.fov) / 2.0))
        aspect = float(sw / sh)

        # Pre-compute alpha + streak length via helper
        cam = dict(
            forward=forward, right=right, up=cam_up,
            fov_scale=fov_scale, aspect=aspect, cam_pos=cam_pos,
            screen_h=sh,
        )
        rain_pts, z_frac, rain_alpha, streak = prepare_rain_gpu_buffers(
            system, v._rain_z_frac, cam, psx, psy, ve,
            v._cloud_altitude, min_depth, rain_grid=rain_grid,
        )
        rain_N = system.n
        f = float(v.subsample_factor)

        # GPU terrain for z lookup
        terrain_data = v.raster.data
        if not isinstance(terrain_data, cp.ndarray):
            terrain_data = cp.asarray(terrain_data)

        # Upload
        _dr = getattr(v, '_d_rain_pts', None)
        if _dr is None or _dr.shape[0] != rain_N:
            v._d_rain_pts = cp.empty((rain_N, 2), dtype=cp.float32)
            v._d_rain_zfrac = cp.empty(rain_N, dtype=cp.float32)
            v._d_rain_alpha = cp.empty(rain_N, dtype=cp.float32)
            v._d_rain_streak = cp.empty(rain_N, dtype=cp.int32)
        v._d_rain_pts.set(rain_pts)
        v._d_rain_zfrac.set(z_frac)
        v._d_rain_alpha.set(rain_alpha)
        v._d_rain_streak.set(streak)

        tpb = 256
        bpg_r = (rain_N + tpb - 1) // tpb
        _rain_splat_kernel[bpg_r, tpb](
            v._d_rain_pts,
            v._d_rain_zfrac,
            v._d_rain_alpha,
            v._d_rain_streak,
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
        v = self.v
        v._wind_data = wind_data
        if wind_data is None:
            return

        # Allow wind_data dict to carry optional tuning overrides
        if 'n_particles' in wind_data:
            v._wind_n_particles = int(wind_data['n_particles'])
        if 'max_age' in wind_data:
            v._wind_max_age = int(wind_data['max_age'])
        if 'speed_mult' in wind_data:
            v._wind_speed_mult = float(wind_data['speed_mult'])
        if 'trail_len' in wind_data:
            v._wind_trail_len = int(wind_data['trail_len'])
        if 'dot_radius' in wind_data:
            v._wind_dot_radius = int(wind_data['dot_radius'])
        if 'alpha' in wind_data:
            v._wind_alpha = float(wind_data['alpha'])
        if 'min_visible_age' in wind_data:
            v._wind_min_visible_age = int(wind_data['min_visible_age'])

        from ..tiles import _build_latlon_grids
        raster = v._base_raster
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
        sm = v._wind_speed_mult
        v._wind_u_px = u_ms * dt * sm / v._base_pixel_spacing_x   # east = +col
        v._wind_v_px = -(v_ms * dt * sm / v._base_pixel_spacing_y)  # north = -row (row 0 is north)

        # Precompute terrain slope gradients (pixels/tick contribution).
        # dz/dcol and dz/drow tell us the downslope direction in pixel space.
        # Particles get pushed downhill and deflected around steep terrain.
        terrain_data = v._base_raster.data
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
        v._wind_slope_col = (-grad_col * slope_scale).astype(np.float32)
        v._wind_slope_row = (-grad_row * slope_scale).astype(np.float32)

        # Create particle system for wind
        from .particles.system import ParticleSystem
        v._wind_system = ParticleSystem(
            n=v._wind_n_particles,
            grid_shape=(H, W),
            max_age=v._wind_max_age,
            trail_len=v._wind_trail_len,
        )
        # Compatibility aliases so delegation properties still work
        v._wind_particles = v._wind_system.positions
        v._wind_ages = v._wind_system.ages
        v._wind_lifetimes = v._wind_system.lifetimes
        v._wind_trails = v._wind_system.trails

        # Min render distance — skip particles near the camera so they
        # don't appear as distracting blobs in the foreground
        world_diag = np.sqrt((W * v._base_pixel_spacing_x)**2 +
                             (H * v._base_pixel_spacing_y)**2)
        v._wind_min_depth = world_diag * 0.02

        print(f"  Wind field interpolated onto {H}x{W} terrain grid")

    def _update_wind_particles(self):
        """Advect wind particles one tick using bilinear-sampled wind field."""
        v = self.v
        system = getattr(v, '_wind_system', None)
        if system is None or v._wind_u_px is None:
            return
        from .particles.wind import advect_wind
        system.push_trail()
        advect_wind(system, v._wind_u_px, v._wind_v_px, v._dt_scale,
                    slope_col=v._wind_slope_col, slope_row=v._wind_slope_row)
        mask = system.tick_age()
        system.respawn(mask)

    def _draw_wind_on_frame(self, img):
        """Project wind particles to screen space and draw on rendered frame.

        Parameters
        ----------
        img : ndarray, shape (H_screen, W_screen, 3)
            Rendered frame (float32 0-1) to draw on. Modified in-place.
        """
        v = self.v
        system = getattr(v, '_wind_system', None)
        if system is None:
            return
        from .particles.wind import render_wind_cpu
        from .particles.project import get_camera_basis
        cam = get_camera_basis(v.position, v._get_look_at(), v.fov,
                               img.shape[1], img.shape[0])
        # Cached CPU terrain
        if v._wind_terrain_np is None:
            td = v.raster.data
            v._wind_terrain_np = td.get() if hasattr(td, 'get') else np.asarray(td)
        render_wind_cpu(system, img, cam, v._wind_terrain_np,
                        v._base_pixel_spacing_x, v._base_pixel_spacing_y,
                        v.vertical_exaggeration, v.subsample_factor,
                        v._wind_min_depth,
                        base_alpha=v._wind_alpha,
                        min_visible_age=v._wind_min_visible_age,
                        dot_radius=v._wind_dot_radius)

    def _splat_wind_gpu(self, d_frame):
        """Project and splat wind particles on GPU via Numba CUDA kernel.

        Parameters
        ----------
        d_frame : cupy.ndarray, shape (H, W, 3)
            GPU frame buffer (float32 0-1). Modified in-place via atomic add.
        """
        v = self.v
        system = getattr(v, '_wind_system', None)
        if system is None or system.trails is None:
            return

        from ..analysis.render import _compute_camera_basis
        from .particles.wind import prepare_wind_gpu_buffers

        sh, sw = d_frame.shape[:2]
        N = system.n
        trail_len = system.trail_len
        total = N * trail_len

        # Camera basis
        cam_pos = v.position
        look_at = v._get_look_at()
        forward, right, cam_up = _compute_camera_basis(
            tuple(cam_pos), tuple(look_at), (0, 0, 1),
        )
        fov_scale = math.tan(math.radians(v.fov) / 2.0)
        aspect_ratio = sw / sh

        # Pre-compute alpha and flatten trails via helper
        all_pts, alpha = prepare_wind_gpu_buffers(
            system, v._wind_alpha, v._wind_min_visible_age)

        # Upload to reusable GPU buffers
        if v._d_wind_trails is None or v._d_wind_trails.shape[0] != total:
            v._d_wind_trails = cp.empty((total, 2), dtype=cp.float32)
            v._d_wind_alpha = cp.empty(total, dtype=cp.float32)
        v._d_wind_trails.set(all_pts)
        v._d_wind_alpha.set(alpha)

        # GPU terrain — use raster.data directly
        terrain_data = v.raster.data
        if not isinstance(terrain_data, cp.ndarray):
            terrain_data = cp.asarray(terrain_data)

        # Kernel launch
        threadsperblock = 256
        blockspergrid = (total + threadsperblock - 1) // threadsperblock

        f = float(v.subsample_factor)
        psx = float(v._base_pixel_spacing_x)
        psy = float(v._base_pixel_spacing_y)
        ve = float(v.vertical_exaggeration)
        min_depth = float(v._wind_min_depth)
        r = int(v._wind_dot_radius)

        _wind_splat_kernel[blockspergrid, threadsperblock](
            v._d_wind_trails,
            v._d_wind_alpha,
            terrain_data,
            d_frame,
            float(cam_pos[0]), float(cam_pos[1]), float(cam_pos[2]),
            float(forward[0]), float(forward[1]), float(forward[2]),
            float(right[0]), float(right[1]), float(right[2]),
            float(cam_up[0]), float(cam_up[1]), float(cam_up[2]),
            float(fov_scale), float(aspect_ratio),
            psx, psy, ve, f, min_depth,
            r,
            0.3, 0.9, 0.8,
        )

        cp.clip(d_frame, 0, 1, out=d_frame)
