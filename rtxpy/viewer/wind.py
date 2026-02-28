"""Wind particle state for the interactive viewer."""


class WindState:
    """Holds all wind particle simulation and GPU splatting state.

    Groups the ~18 wind-related variables and GPU buffers into a
    single object for cleaner viewer initialization.
    """

    __slots__ = (
        'wind_data', 'wind_enabled',
        'wind_u_px', 'wind_v_px',
        'wind_particles', 'wind_ages',
        'wind_max_age', 'wind_n_particles',
        'wind_trail_len', 'wind_trails',
        'wind_speed_mult', 'wind_min_depth',
        'wind_dot_radius', 'wind_alpha',
        'wind_min_visible_age', 'wind_terrain_np',
        # GPU buffers
        'd_wind_trails', 'd_wind_alpha',
        'd_base_frame', 'd_wind_scratch',
        'wind_done_event',
    )

    def __init__(self):
        self.wind_data = None
        self.wind_enabled = False
        self.wind_u_px = None
        self.wind_v_px = None
        self.wind_particles = None
        self.wind_ages = None
        self.wind_max_age = 80
        self.wind_n_particles = 10000
        self.wind_trail_len = 20
        self.wind_trails = None
        self.wind_speed_mult = 250.0
        self.wind_min_depth = 0.0
        self.wind_dot_radius = 2
        self.wind_alpha = 0.055
        self.wind_min_visible_age = 6
        self.wind_terrain_np = None

        # GPU buffers
        self.d_wind_trails = None
        self.d_wind_alpha = None
        self.d_base_frame = None
        self.d_wind_scratch = None
        self.wind_done_event = None
