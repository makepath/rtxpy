"""InteractiveViewer class — composed from subsystem objects."""

import queue
import threading
import time

import numpy as np

from ..rtx import RTX, has_cupy
from ..viewer.input_state import InputState
from ..viewer.camera import CameraState
from ..viewer.render_settings import RenderSettings
from ..viewer.overlays import OverlayManager
from ..viewer.geometry_layers import GeometryLayerManager
from ..viewer.terrain import TerrainState
from ..viewer.observers import ObserverManager
from ..viewer.wind import WindState
from ..viewer.cloud import CloudState
from ..viewer.hydro import HydroState
from ..viewer.hydro_manager import HydroManager
from ..viewer.hud import HUDState

from .delegate import apply_delegations, apply_camera_wrappers, apply_forwarding
from .minimap import MinimapRenderer
from .terrain_ops import TerrainOps
from .weather import WeatherManager
from .hydro import HydroController
from .observers import ObserverController
from .renderer import FrameRenderer
from .input import InputHandler
from .hud import HUDRenderer
from .run import RunManager

if has_cupy:
    import cupy as cp


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

        # Subsystem objects — each holds a back-reference to this viewer as self.v
        self._minimap_renderer = MinimapRenderer(self)
        self._terrain_ops = TerrainOps(self)
        self._weather_mgr = WeatherManager(self)
        self._hydro_ctrl = HydroController(self)
        self._observer_ctrl = ObserverController(self)
        self._renderer = FrameRenderer(self)
        self._input_handler = InputHandler(self)
        self._hud_renderer = HUDRenderer(self)
        self._run_mgr = RunManager(self)

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
    # Core tick loop
    # ------------------------------------------------------------------

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
        # Scene mesh updates: prefer LOD-manager-driven SceneMeshManager,
        # fall back to legacy _MeshChunkManager for backward compat.
        smm = (self._terrain_lod_manager.scene_mesh_manager
               if self._terrain_lod_manager is not None else None)
        if smm is not None and smm.is_dirty:
            if self._update_scene_meshes(smm):
                self._render_needed = True
        elif self._chunk_manager is not None:
            # Legacy path: sync distance params from LOD manager
            if (self.lod_enabled and self._terrain_lod_manager is not None
                    and self._terrain_lod_manager._lod_distances):
                self._chunk_manager.max_distance = (
                    self._terrain_lod_manager._lod_distances[-1])
                self._chunk_manager._lod_distances = (
                    self._terrain_lod_manager._lod_distances)
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
            if self._interop_enabled:
                # Interop: splat into PBO directly, no CPU involvement
                try:
                    d_pbo = self._cuda_gl_buf.map()
                    try:
                        cp.copyto(d_pbo, self._d_base_frame)
                        if self._wind_enabled and self._wind_particles is not None:
                            self._update_wind_particles()
                            self._splat_wind_gpu(d_pbo)
                        if self._hydro_enabled and self._hydro_particles is not None:
                            self.hydro_mgr.check_streaming_result()
                            self._transfer_streaming_overlay()
                            cam_r = self.position[1] / self._base_pixel_spacing_y
                            cam_c = self.position[0] / self._base_pixel_spacing_x
                            self.hydro_mgr.update_streaming_window(cam_r, cam_c)
                            self._update_hydro_particles()
                            self._splat_hydro_gpu(d_pbo)
                        if self._clouds_enabled and self._rain_particles is not None:
                            self._update_rain_particles()
                            self._splat_rain_gpu(d_pbo)
                    finally:
                        self._cuda_gl_buf.unmap()
                    self._cuda_gl_buf.upload_to_texture(self._interop_frame_tex)
                    self._update_window_title()
                    self._frame_dirty = True
                except Exception:
                    import traceback
                    traceback.print_exc()
                    self._interop_enabled = False
                    # Fall through to CPU path below
                    self._idle_particles_cpu()
            else:
                self._idle_particles_cpu()

    def _idle_particles_cpu(self):
        """CPU fallback for idle particle replay (no ray trace needed)."""
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

    # ------------------------------------------------------------------
    # Terrain reload (camera near edge → prefetch next window)
    # ------------------------------------------------------------------

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

        # Camera stays at its current position ��� no jump.
        # Only update Z if the terrain height changed significantly
        # under the camera (keeps altitude above ground consistent).

        # Refresh minimap
        self._minimap_bg_extent = None
        self._compute_minimap_background()

        self._last_reload_time = time.time()
        self._render_needed = True
        print(f"Terrain reloaded: center ({cam_lon:.4f}, {cam_lat:.4f}), "
              f"window {new_W}x{new_H}")


# Install delegation properties, camera wrappers, and subsystem forwarding
# onto InteractiveViewer after the class body is defined.
apply_delegations(InteractiveViewer)
apply_camera_wrappers(InteractiveViewer)
apply_forwarding(InteractiveViewer)
