"""Cloud particle state for the interactive viewer."""


class CloudState:
    """Holds cloud particle simulation state.

    Cloud particles float above the terrain at a fixed altitude,
    drift with the wind field, and render as soft white puffs with
    density proportional to the interpolated cloud_cover field.
    """

    __slots__ = (
        'clouds_enabled',
        'cloud_cover_grid',     # (H, W) interpolated 0-1 cloud fraction
        'cloud_particles',      # (N, 2) — row, col in full-res pixel coords
        'cloud_sizes',          # (N,) — world-space radius per particle
        'cloud_alphas',         # (N,) — base alpha per particle
        'cloud_ages',           # (N,)
        'cloud_lifetimes',      # (N,)
        'cloud_n_particles',
        'cloud_max_age',
        'cloud_altitude',       # world-space Z for the cloud layer
        'cloud_min_depth',      # min camera distance for rendering
        'cloud_terrain_np',     # cached CPU terrain for projection
        'volumetric_clouds_enabled',  # bool: volumetric ray-marched clouds
        'cloud_thickness',      # world-space thickness of cloud slab
        'cloud_time',           # animation time counter for wind drift
    )

    def __init__(self):
        self.clouds_enabled = False
        self.cloud_cover_grid = None
        self.cloud_particles = None
        self.cloud_sizes = None
        self.cloud_alphas = None
        self.cloud_ages = None
        self.cloud_lifetimes = None
        self.cloud_n_particles = 4000
        self.cloud_max_age = 300
        self.cloud_altitude = 0.0
        self.cloud_min_depth = 0.0
        self.cloud_terrain_np = None
        self.volumetric_clouds_enabled = False
        self.cloud_thickness = 0.0
        self.cloud_time = 0.0
