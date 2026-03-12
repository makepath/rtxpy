"""Hydrological flow particle state for the interactive viewer."""


class HydroState:
    """Holds all hydro flow particle simulation and GPU splatting state.

    Groups the hydro-related variables and GPU buffers into a
    single object for cleaner viewer initialization.
    """

    __slots__ = (
        'hydro_data', 'hydro_enabled', 'hydro_lazy',
        'hydro_flow_u_px', 'hydro_flow_v_px',
        'hydro_flow_accum_norm',
        'hydro_stream_order',
        'hydro_stream_order_raw',
        'hydro_stream_link',
        'hydro_particles', 'hydro_ages', 'hydro_lifetimes',
        'hydro_max_age', 'hydro_n_particles',
        'hydro_trail_len', 'hydro_trails',
        'hydro_speed', 'hydro_min_depth', 'hydro_max_depth', 'hydro_ref_depth',
        'hydro_dot_radius', 'hydro_alpha',
        'hydro_min_visible_age',
        'hydro_accum_threshold',
        'hydro_color',
        'hydro_terrain_np',
        'hydro_spawn_probs',
        'hydro_spawn_indices', 'hydro_spawn_valid_probs',
        # Accumulation-scaled rendering
        'hydro_slope_mag',
        'hydro_particle_accum',
        'hydro_particle_colors',
        'hydro_particle_radii',
        'hydro_particle_raw_order',
        # GPU buffers (rendering)
        'd_hydro_trails',
        'd_hydro_ages', 'd_hydro_lifetimes',
        'd_hydro_colors', 'd_hydro_radii',
        'hydro_done_event',
        # GPU-resident advection state
        'd_hydro_particles',
        'd_hydro_particle_accum',
        'd_hydro_particle_raw_order',
        'd_hydro_flow_u', 'd_hydro_flow_v',
        'd_hydro_slope_mag',
        'd_hydro_stream_order', 'd_hydro_stream_order_raw',
        'd_hydro_accum_norm',
        'd_hydro_palette',
        'd_hydro_respawn_flags',
    )

    def __init__(self):
        self.hydro_data = None
        self.hydro_enabled = False
        self.hydro_lazy = False
        self.hydro_flow_u_px = None
        self.hydro_flow_v_px = None
        self.hydro_flow_accum_norm = None
        self.hydro_stream_order = None
        self.hydro_stream_order_raw = None
        self.hydro_stream_link = None
        self.hydro_particles = None
        self.hydro_ages = None
        self.hydro_lifetimes = None
        self.hydro_max_age = 200
        self.hydro_n_particles = 12000
        self.hydro_trail_len = 20
        self.hydro_trails = None
        self.hydro_speed = 0.75
        self.hydro_min_depth = 0.0
        self.hydro_max_depth = 0.0  # 0 = unlimited
        self.hydro_ref_depth = 1.0
        self.hydro_dot_radius = 2
        self.hydro_alpha = 0.5
        self.hydro_min_visible_age = 2
        self.hydro_accum_threshold = 50
        self.hydro_color = (0.2, 0.5, 1.0)
        self.hydro_terrain_np = None
        self.hydro_spawn_probs = None
        self.hydro_spawn_indices = None
        self.hydro_spawn_valid_probs = None

        # Accumulation-scaled rendering
        self.hydro_slope_mag = None
        self.hydro_particle_accum = None
        self.hydro_particle_colors = None
        self.hydro_particle_radii = None
        self.hydro_particle_raw_order = None

        # GPU buffers (rendering)
        self.d_hydro_trails = None
        self.d_hydro_ages = None
        self.d_hydro_lifetimes = None
        self.d_hydro_colors = None
        self.d_hydro_radii = None
        self.hydro_done_event = None
        # GPU-resident advection state
        self.d_hydro_particles = None
        self.d_hydro_particle_accum = None
        self.d_hydro_particle_raw_order = None
        self.d_hydro_flow_u = None
        self.d_hydro_flow_v = None
        self.d_hydro_slope_mag = None
        self.d_hydro_stream_order = None
        self.d_hydro_stream_order_raw = None
        self.d_hydro_accum_norm = None
        self.d_hydro_palette = None
        self.d_hydro_respawn_flags = None
