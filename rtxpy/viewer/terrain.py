"""Terrain state for the interactive viewer."""


class TerrainState:
    """Holds terrain raster data, spacing, elevation stats, and mesh caches.

    Groups all terrain-specific state so the viewer can reason about
    terrain independently from rendering, camera, or UI concerns.
    """

    __slots__ = (
        'raster', '_base_raster',
        'terrain_shape', 'elev_min', 'elev_max', 'elev_mean',
        'pixel_spacing_x', 'pixel_spacing_y',
        '_base_pixel_spacing_x', '_base_pixel_spacing_y',
        'subsample_factor',
        '_terrain_mesh_cache', '_baked_mesh_cache',
        '_gpu_terrain', '_gpu_base_terrain',
        'mesh_type', '_water_mask',
        'vertical_exaggeration', '_land_color_range',
        'terrain_skirt',
        '_terrain_loader',
        '_coord_origin_x', '_coord_origin_y',
        '_coord_step_x', '_coord_step_y',
        '_reload_cooldown', '_last_reload_time',
        '_terrain_reload_future', '_terrain_reload_pool',
        # LOD state
        'lod_enabled', '_terrain_lod_manager',
    )

    def __init__(self, raster, pixel_spacing_x=1.0, pixel_spacing_y=1.0,
                 mesh_type='heightfield', subsample=1, skirt=True):
        self.raster = raster
        self._base_raster = raster
        self.pixel_spacing_x = pixel_spacing_x
        self.pixel_spacing_y = pixel_spacing_y
        self._base_pixel_spacing_x = pixel_spacing_x
        self._base_pixel_spacing_y = pixel_spacing_y
        self.mesh_type = mesh_type
        self.subsample_factor = max(1, int(subsample))
        self.vertical_exaggeration = 1.0
        self.terrain_skirt = skirt

        # Elevation stats (set by viewer __init__ after ocean-fill)
        self.terrain_shape = (0, 0)
        self.elev_min = 0.0
        self.elev_max = 0.0
        self.elev_mean = 0.0
        self._land_color_range = None
        self._water_mask = None

        # Mesh caches
        self._terrain_mesh_cache = {}
        self._baked_mesh_cache = {}
        self._gpu_terrain = None
        self._gpu_base_terrain = None

        # Dynamic terrain loading
        self._terrain_loader = None
        self._coord_origin_x = 0.0
        self._coord_origin_y = 0.0
        self._coord_step_x = 1.0
        self._coord_step_y = -1.0
        self._reload_cooldown = 2.0
        self._last_reload_time = 0.0
        self._terrain_reload_future = None
        self._terrain_reload_pool = None

        # LOD
        self.lod_enabled = False
        self._terrain_lod_manager = None
