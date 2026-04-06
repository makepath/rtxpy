"""Compact delegation properties for InteractiveViewer.

Replaces ~1700 lines of boilerplate @property getter/setter pairs with
declarative mapping tables.  Each entry maps a property name on
InteractiveViewer to (sub_object_name, attribute_name) on the sub-object.
"""

import numpy as np


def _make_property(sub_obj, attr_name):
    """Create a read/write property delegating to *sub_obj*.*attr_name*."""
    return property(
        lambda self, _s=sub_obj, _a=attr_name: getattr(getattr(self, _s), _a),
        lambda self, value, _s=sub_obj, _a=attr_name: setattr(getattr(self, _s), _a, value),
    )


# ---- Mapping tables: {property_name: (sub_object, attribute)} ----
# When property_name == attribute_name, a set entry is used and
# expanded below.

_SIMPLE_DELEGATIONS = {
    # -- InputState --
    'input': {
        '_held_keys': 'held_keys',
        '_mouse_dragging': 'mouse_dragging',
        '_mouse_last_x': 'mouse_last_x',
        '_mouse_last_y': 'mouse_last_y',
    },
    # -- CameraState --
    'camera': {
        'position': 'position',
        'yaw': 'yaw',
        'pitch': 'pitch',
        'fov': 'fov',
        'move_speed': 'move_speed',
        'look_speed': 'look_speed',
        '_time_presets': '_time_presets',
        '_time_preset_idx': '_time_preset_idx',
    },
    # -- RenderSettings --
    'render_settings': {
        'shadows': 'shadows',
        'ambient': 'ambient',
        'sun_azimuth': 'sun_azimuth',
        'sun_altitude': 'sun_altitude',
        'fog_density': 'fog_density',
        'fog_color': 'fog_color',
        'colormap': 'colormap',
        'colormaps': 'colormaps',
        'colormap_idx': 'colormap_idx',
        'color_stretch': 'color_stretch',
        '_color_stretches': '_color_stretches',
        '_color_stretch_idx': '_color_stretch_idx',
        'ao_enabled': 'ao_enabled',
        'ao_radius': 'ao_radius',
        'gi_intensity': 'gi_intensity',
        'gi_bounces': 'gi_bounces',
        '_ao_samples_per_frame': '_ao_samples_per_frame',
        '_ao_max_frames': '_ao_max_frames',
        '_ao_frame_count': '_ao_frame_count',
        '_d_ao_accum': '_d_ao_accum',
        '_prev_cam_state': '_prev_cam_state',
        'edl_enabled': 'edl_enabled',
        'denoise_enabled': 'denoise_enabled',
        '_prev_cam_for_flow': '_prev_cam_for_flow',
        '_d_flow': '_d_flow',
        'dof_enabled': 'dof_enabled',
        '_dof_aperture': '_dof_aperture',
        '_dof_focal_distance': '_dof_focal_distance',
    },
    # -- OverlayManager --
    'overlays': {
        '_overlay_layers': 'overlay_layers',
        '_overlay_names': 'overlay_names',
        '_active_color_data': 'active_color_data',
        '_active_overlay_data': 'active_overlay_data',
        '_overlay_alpha': 'overlay_alpha',
        '_overlay_as_water': 'overlay_as_water',
        '_active_overlay_color_lut': 'active_overlay_color_lut',
        '_overlay_color_luts': 'overlay_color_luts',
        '_terrain_layer_order': 'terrain_layer_order',
        '_terrain_layer_idx': 'terrain_layer_idx',
        '_base_overlay_layers': 'base_overlay_layers',
        '_tile_service': 'tile_service',
        '_tiles_enabled': 'tiles_enabled',
        '_basemap_options': 'basemap_options',
        '_basemap_idx': 'basemap_idx',
    },
    # -- GeometryLayerManager --
    'geometry_layers': {
        '_all_geometries': 'all_geometries',
        '_geometry_layer_order': 'geometry_layer_order',
        '_geometry_layer_idx': 'geometry_layer_idx',
        '_layer_positions': 'layer_positions',
        '_current_geom_idx': 'current_geom_idx',
        '_pc_color_modes': 'pc_color_modes',
        '_pc_color_mode_idx': 'pc_color_mode_idx',
        '_chunk_manager': 'chunk_manager',
        '_baked_meshes': 'baked_meshes',
        '_geometry_colors_builder': 'geometry_colors_builder',
    },
    # -- HUDState --
    'hud': {
        '_title': 'title',
        '_subtitle': 'subtitle',
        '_legend_config': 'legend_config',
        '_info_text': 'info_text',
        '_title_overlay_rgba': 'title_overlay_rgba',
        '_legend_rgba': 'legend_rgba',
        '_help_page_idx': 'help_page_idx',
        '_help_pages': 'help_pages',
        'show_minimap': 'show_minimap',
        '_last_title': 'last_title',
        '_last_subtitle': 'last_subtitle',
        '_minimap_background': 'minimap_background',
        '_minimap_scale_x': 'minimap_scale_x',
        '_minimap_scale_y': 'minimap_scale_y',
        '_minimap_has_tiles': 'minimap_has_tiles',
        '_minimap_rect': 'minimap_rect',
        '_minimap_world_extent': 'minimap_world_extent',
        '_minimap_style': 'minimap_style',
        '_minimap_layer': 'minimap_layer',
        '_minimap_colors': 'minimap_colors',
        '_minimap_bg_extent': 'minimap_bg_extent',
        '_minimap_last_stream_time': 'minimap_last_stream_time',
    },
    # -- WindState --
    'wind': {
        '_wind_data': 'wind_data',
        '_wind_enabled': 'wind_enabled',
        '_wind_u_px': 'wind_u_px',
        '_wind_v_px': 'wind_v_px',
        '_wind_particles': 'wind_particles',
        '_wind_ages': 'wind_ages',
        '_wind_max_age': 'wind_max_age',
        '_wind_n_particles': 'wind_n_particles',
        '_wind_trail_len': 'wind_trail_len',
        '_wind_trails': 'wind_trails',
        '_wind_speed_mult': 'wind_speed_mult',
        '_wind_min_depth': 'wind_min_depth',
        '_wind_dot_radius': 'wind_dot_radius',
        '_wind_alpha': 'wind_alpha',
        '_wind_min_visible_age': 'wind_min_visible_age',
        '_wind_terrain_np': 'wind_terrain_np',
        '_d_wind_trails': 'd_wind_trails',
        '_d_wind_alpha': 'd_wind_alpha',
        '_d_base_frame': 'd_base_frame',
        '_d_wind_scratch': 'd_wind_scratch',
        '_wind_done_event': 'wind_done_event',
    },
    # -- CloudState --
    'clouds': {
        '_clouds_enabled': 'clouds_enabled',
        '_cloud_cover_grid': 'cloud_cover_grid',
        '_cloud_particles': 'cloud_particles',
        '_cloud_sizes': 'cloud_sizes',
        '_cloud_alphas': 'cloud_alphas',
        '_cloud_ages': 'cloud_ages',
        '_cloud_lifetimes': 'cloud_lifetimes',
        '_cloud_n_particles': 'cloud_n_particles',
        '_cloud_max_age': 'cloud_max_age',
        '_cloud_altitude': 'cloud_altitude',
        '_cloud_min_depth': 'cloud_min_depth',
        '_cloud_terrain_np': 'cloud_terrain_np',
        '_volumetric_clouds_enabled': 'volumetric_clouds_enabled',
        '_cloud_thickness': 'cloud_thickness',
        '_cloud_time': 'cloud_time',
    },
    # -- ObserverManager --
    'observer_mgr': {
        '_observers': 'observers',
        '_active_observer': 'active_observer',
        'viewshed_enabled': 'viewshed_enabled',
        'viewshed_observer_elev': 'viewshed_observer_elev',
        'viewshed_target_elev': 'viewshed_target_elev',
        'viewshed_opacity': 'viewshed_opacity',
        '_viewshed_cache': 'viewshed_cache',
        '_viewshed_coverage': 'viewshed_coverage',
        '_viewshed_recalc_interval': 'viewshed_recalc_interval',
        '_last_viewshed_time': 'last_viewshed_time',
        '_shared_drone_parts': 'shared_drone_parts',
        '_drone_glow': 'drone_glow',
    },
    # -- TerrainState --
    'terrain': {
        'raster': 'raster',
        '_base_raster': '_base_raster',
        'terrain_shape': 'terrain_shape',
        'elev_min': 'elev_min',
        'elev_max': 'elev_max',
        'elev_mean': 'elev_mean',
        'pixel_spacing_x': 'pixel_spacing_x',
        'pixel_spacing_y': 'pixel_spacing_y',
        '_base_pixel_spacing_x': '_base_pixel_spacing_x',
        '_base_pixel_spacing_y': '_base_pixel_spacing_y',
        'subsample_factor': 'subsample_factor',
        '_baked_mesh_cache': '_baked_mesh_cache',
        '_gpu_terrain': '_gpu_terrain',
        '_gpu_base_terrain': '_gpu_base_terrain',
        'terrain_skirt': 'terrain_skirt',
        '_water_mask': '_water_mask',
        'vertical_exaggeration': 'vertical_exaggeration',
        '_land_color_range': '_land_color_range',
        '_terrain_loader': '_terrain_loader',
        '_coord_origin_x': '_coord_origin_x',
        '_coord_origin_y': '_coord_origin_y',
        '_coord_step_x': '_coord_step_x',
        '_coord_step_y': '_coord_step_y',
        '_reload_cooldown': '_reload_cooldown',
        '_last_reload_time': '_last_reload_time',
        'lod_enabled': 'lod_enabled',
        '_terrain_lod_manager': '_terrain_lod_manager',
    },
    # -- HydroState --
    'hydro': {
        '_hydro_data': 'hydro_data',
        '_hydro_lazy': 'hydro_lazy',
        '_hydro_enabled': 'hydro_enabled',
        '_hydro_flow_u_px': 'hydro_flow_u_px',
        '_hydro_flow_v_px': 'hydro_flow_v_px',
        '_hydro_flow_accum_norm': 'hydro_flow_accum_norm',
        '_hydro_stream_order': 'hydro_stream_order',
        '_hydro_stream_order_raw': 'hydro_stream_order_raw',
        '_hydro_stream_link': 'hydro_stream_link',
        '_hydro_particle_raw_order': 'hydro_particle_raw_order',
        '_hydro_particles': 'hydro_particles',
        '_hydro_ages': 'hydro_ages',
        '_hydro_lifetimes': 'hydro_lifetimes',
        '_hydro_max_age': 'hydro_max_age',
        '_hydro_n_particles': 'hydro_n_particles',
        '_hydro_trail_len': 'hydro_trail_len',
        '_hydro_trails': 'hydro_trails',
        '_hydro_speed': 'hydro_speed',
        '_hydro_min_depth': 'hydro_min_depth',
        '_hydro_max_depth': 'hydro_max_depth',
        '_hydro_ref_depth': 'hydro_ref_depth',
        '_hydro_dot_radius': 'hydro_dot_radius',
        '_hydro_alpha': 'hydro_alpha',
        '_hydro_min_visible_age': 'hydro_min_visible_age',
        '_hydro_accum_threshold': 'hydro_accum_threshold',
        '_hydro_color': 'hydro_color',
        '_hydro_terrain_np': 'hydro_terrain_np',
        '_hydro_spawn_probs': 'hydro_spawn_probs',
        '_hydro_spawn_indices': 'hydro_spawn_indices',
        '_hydro_spawn_valid_probs': 'hydro_spawn_valid_probs',
        '_d_hydro_trails': 'd_hydro_trails',
        '_d_hydro_ages': 'd_hydro_ages',
        '_d_hydro_lifetimes': 'd_hydro_lifetimes',
        '_hydro_done_event': 'hydro_done_event',
        '_hydro_slope_mag': 'hydro_slope_mag',
        '_hydro_particle_accum': 'hydro_particle_accum',
        '_d_hydro_colors': 'd_hydro_colors',
        '_d_hydro_radii': 'd_hydro_radii',
        '_d_hydro_particles': 'd_hydro_particles',
        '_d_hydro_particle_accum': 'd_hydro_particle_accum',
        '_d_hydro_particle_raw_order': 'd_hydro_particle_raw_order',
        '_d_hydro_flow_u': 'd_hydro_flow_u',
        '_d_hydro_flow_v': 'd_hydro_flow_v',
        '_d_hydro_slope_mag': 'd_hydro_slope_mag',
        '_d_hydro_stream_order': 'd_hydro_stream_order',
        '_d_hydro_stream_order_raw': 'd_hydro_stream_order_raw',
        '_d_hydro_accum_norm': 'd_hydro_accum_norm',
        '_d_hydro_palette': 'd_hydro_palette',
        '_d_hydro_respawn_flags': 'd_hydro_respawn_flags',
        '_hydro_particle_colors': 'hydro_particle_colors',
        '_hydro_particle_radii': 'hydro_particle_radii',
    },
}


def apply_delegations(cls):
    """Install all delegation properties onto *cls*."""
    for sub_obj, mapping in _SIMPLE_DELEGATIONS.items():
        for prop_name, attr_name in mapping.items():
            setattr(cls, prop_name, _make_property(sub_obj, attr_name))


# ------------------------------------------------------------------
# Camera thin wrappers (kept as regular methods, not properties)
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


def apply_camera_wrappers(cls):
    """Install camera wrapper methods onto *cls*."""
    cls._get_front = _get_front
    cls._get_right = _get_right
    cls._screen_to_ray = _screen_to_ray
    cls._get_look_at = _get_look_at
    cls._build_title = _build_title


# ------------------------------------------------------------------
# Subsystem forwarding methods
# ------------------------------------------------------------------
# Each entry: (viewer_method_name, subsystem_attr, subsystem_method)
# All use *args/**kwargs forwarding so signatures don't need repeating here.
# Methods with default arguments (e.g. _cycle_basemap(reverse=False)) are
# defined explicitly in apply_forwarding() below.

_FORWARDING = [
    # MinimapRenderer
    ('_compute_minimap_background', '_minimap_renderer', '_compute_minimap_background'),
    ('_blit_minimap_on_frame', '_minimap_renderer', '_blit_minimap_on_frame'),
    ('_compute_streaming_minimap', '_minimap_renderer', '_compute_streaming_minimap'),
    # TerrainOps
    ('_update_scene_meshes', '_terrain_ops', '_update_scene_meshes'),
    ('_enable_terrain_lod', '_terrain_ops', '_enable_terrain_lod'),
    ('_rebuild_at_resolution', '_terrain_ops', '_rebuild_at_resolution'),
    ('_rebuild_vertical_exaggeration', '_terrain_ops', '_rebuild_vertical_exaggeration'),
    # WeatherManager
    ('_toggle_wind', '_weather_mgr', '_toggle_wind'),
    ('_toggle_firms', '_weather_mgr', '_toggle_firms'),
    ('_init_weather', '_weather_mgr', '_init_weather'),
    ('_init_wind', '_weather_mgr', '_init_wind'),
    ('_init_clouds', '_weather_mgr', '_init_clouds'),
    ('_toggle_clouds', '_weather_mgr', '_toggle_clouds'),
    ('_action_toggle_clouds', '_weather_mgr', '_action_toggle_clouds'),
    ('_update_wind_particles', '_weather_mgr', '_update_wind_particles'),
    ('_draw_wind_on_frame', '_weather_mgr', '_draw_wind_on_frame'),
    ('_splat_wind_gpu', '_weather_mgr', '_splat_wind_gpu'),
    ('_update_rain_particles', '_weather_mgr', '_update_rain_particles'),
    ('_draw_rain_on_frame', '_weather_mgr', '_draw_rain_on_frame'),
    ('_splat_rain_gpu', '_weather_mgr', '_splat_rain_gpu'),
    # HydroController
    ('_toggle_hydro', '_hydro_ctrl', '_toggle_hydro'),
    ('_action_toggle_hydro', '_hydro_ctrl', '_action_toggle_hydro'),
    ('_transfer_streaming_overlay', '_hydro_ctrl', '_transfer_streaming_overlay'),
    ('_update_hydro_particles', '_hydro_ctrl', '_update_hydro_particles'),
    ('_draw_hydro_on_frame', '_hydro_ctrl', '_draw_hydro_on_frame'),
    ('_splat_hydro_gpu', '_hydro_ctrl', '_splat_hydro_gpu'),
    ('_init_gtfs_rt', '_hydro_ctrl', '_init_gtfs_rt'),
    ('_toggle_gtfs_rt', '_hydro_ctrl', '_toggle_gtfs_rt'),
    ('_draw_gtfs_rt_on_frame', '_hydro_ctrl', '_draw_gtfs_rt_on_frame'),
    ('_cleanup_gtfs_rt', '_hydro_ctrl', '_cleanup_gtfs_rt'),
    # ObserverController
    ('_load_drone_parts', '_observer_ctrl', '_load_drone_parts'),
    ('_update_observer_drone_for', '_observer_ctrl', '_update_observer_drone_for'),
    ('_apply_drone_glow', '_observer_ctrl', '_apply_drone_glow'),
    ('_set_drone_visibility_for', '_observer_ctrl', '_set_drone_visibility_for'),
    ('_cycle_drone_mode_for', '_observer_ctrl', '_cycle_drone_mode_for'),
    ('_snap_to_observer', '_observer_ctrl', '_snap_to_observer'),
    ('_place_observer_at', '_observer_ctrl', '_place_observer_at'),
    ('_select_or_create_observer', '_observer_ctrl', '_select_or_create_observer'),
    ('_exit_fpv_for', '_observer_ctrl', '_exit_fpv_for'),
    ('_clear_observer_slot', '_observer_ctrl', '_clear_observer_slot'),
    ('_clear_all_observers', '_observer_ctrl', '_clear_all_observers'),
    ('_toggle_viewshed', '_observer_ctrl', '_toggle_viewshed'),
    ('_apply_viewshed_overlay', '_observer_ctrl', '_apply_viewshed_overlay'),
    ('_calculate_viewshed_for', '_observer_ctrl', '_calculate_viewshed_for'),
    ('_adjust_observer_elevation', '_observer_ctrl', '_adjust_observer_elevation'),
    ('_sync_drone_from_pos_for', '_observer_ctrl', '_sync_drone_from_pos_for'),
    # FrameRenderer
    ('_render_frame', '_renderer', '_render_frame'),
    ('_update_frame', '_renderer', '_update_frame'),
    ('_update_frame_cpu', '_renderer', '_update_frame_cpu'),
    ('_composite_overlays', '_renderer', '_composite_overlays'),
    ('_update_window_title', '_renderer', '_update_window_title'),
    ('_build_overlay_rgba', '_renderer', '_build_overlay_rgba'),
    ('_build_overlay_rgba_cached', '_renderer', '_build_overlay_rgba_cached'),
    ('_overlay_input_signature', '_renderer', '_overlay_input_signature'),
    ('_save_screenshot', '_renderer', '_save_screenshot'),
    # InputHandler
    ('_handle_key_press', '_input_handler', '_handle_key_press'),
    ('_handle_key_release', '_input_handler', '_handle_key_release'),
    ('_handle_scroll', '_input_handler', '_handle_scroll'),
    ('_handle_mouse_press', '_input_handler', '_handle_mouse_press'),
    ('_handle_mouse_release', '_input_handler', '_handle_mouse_release'),
    ('_handle_mouse_motion', '_input_handler', '_handle_mouse_motion'),
    ('_cycle_terrain_layer', '_input_handler', '_cycle_terrain_layer'),
    ('_cycle_geometry_layer', '_input_handler', '_cycle_geometry_layer'),
    ('_cycle_pointcloud_colors', '_input_handler', '_cycle_pointcloud_colors'),
    ('_get_terrain_z', '_input_handler', '_get_terrain_z'),
    # HUDRenderer
    ('_render_title_overlay', '_hud_renderer', '_render_title_overlay'),
    ('_blit_title_on_frame', '_hud_renderer', '_blit_title_on_frame'),
    ('_render_legend_overlay', '_hud_renderer', '_render_legend_overlay'),
    ('_blit_legend_on_frame', '_hud_renderer', '_blit_legend_on_frame'),
    ('_render_help_text', '_hud_renderer', '_render_help_text'),
    ('_blit_help_on_frame', '_hud_renderer', '_blit_help_on_frame'),
    ('_drain_command_queue', '_hud_renderer', '_drain_command_queue'),
]


def _make_forwarder(sub_attr, sub_method):
    """Create a method that forwards to *sub_attr*.*sub_method* on self."""
    def forwarder(self, *args, **kwargs):
        return getattr(getattr(self, sub_attr), sub_method)(*args, **kwargs)
    return forwarder


def apply_forwarding(cls):
    """Install subsystem forwarding methods onto *cls*."""
    for method_name, sub_attr, sub_method in _FORWARDING:
        setattr(cls, method_name, _make_forwarder(sub_attr, sub_method))

    # Methods with special signatures
    def _init_hydro(self, flow_accum, **kwargs):
        return self._hydro_ctrl._init_hydro(flow_accum, **kwargs)
    cls._init_hydro = _init_hydro

    def _compute_hydro_from_terrain(self):
        return self._hydro_ctrl._compute_hydro_from_terrain()
    cls._compute_hydro_from_terrain = _compute_hydro_from_terrain

    def _calculate_viewshed(self, quiet=False):
        self._observer_ctrl._calculate_viewshed(quiet=quiet)
    cls._calculate_viewshed = _calculate_viewshed

    def _present_if_dirty(self, *args):
        self._renderer._present_if_dirty(*args)
    cls._present_if_dirty = _present_if_dirty

    def _cycle_basemap(self, reverse=False):
        self._input_handler._cycle_basemap(reverse)
    cls._cycle_basemap = _cycle_basemap

    def _jump_to_geometry(self, direction=-1):
        self._input_handler._jump_to_geometry(direction)
    cls._jump_to_geometry = _jump_to_geometry

    def run(self, **kwargs):
        return self._run_mgr.run(**kwargs)
    cls.run = run

    # Static methods
    from .observers import ObserverController
    from .minimap import MinimapRenderer
    cls._get_drone_front_for = staticmethod(ObserverController._get_drone_front_for)
    cls._get_drone_right_for = staticmethod(ObserverController._get_drone_right_for)
    cls._elevation_to_minimap_rgba = staticmethod(MinimapRenderer._elevation_to_minimap_rgba)

    # Hydro palette class-level delegation
    from .hydro import HydroController
    cls._STREAM_ORDER_PALETTE = HydroController._STREAM_ORDER_PALETTE
    cls._build_stream_palette_lut = staticmethod(HydroController._build_stream_palette_lut_fn)
    cls._hydro_color_from_order = staticmethod(HydroController._hydro_color_from_order_fn)
    cls._hydro_radius_from_order = staticmethod(HydroController._hydro_radius_from_order_fn)
