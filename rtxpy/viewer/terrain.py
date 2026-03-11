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
        '_baked_mesh_cache',
        '_gpu_terrain', '_gpu_base_terrain',
        '_water_mask',
        'vertical_exaggeration', '_land_color_range',
        'terrain_skirt',
        '_terrain_loader',
        '_coord_origin_x', '_coord_origin_y',
        '_coord_step_x', '_coord_step_y',
        '_reload_cooldown', '_last_reload_time',
        '_terrain_reload_future', '_terrain_reload_pool',
        # World-space offset: keeps camera stable across terrain reloads
        '_world_offset_x', '_world_offset_y',
        # LOD state
        'lod_enabled', '_terrain_lod_manager',
    )

    def __init__(self, raster, pixel_spacing_x=1.0, pixel_spacing_y=1.0,
                 subsample=1, skirt=True):
        self.raster = raster
        self._base_raster = raster
        self.pixel_spacing_x = pixel_spacing_x
        self.pixel_spacing_y = pixel_spacing_y
        self._base_pixel_spacing_x = pixel_spacing_x
        self._base_pixel_spacing_y = pixel_spacing_y
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

        # World-space offset for stable camera across terrain reloads
        self._world_offset_x = 0.0
        self._world_offset_y = 0.0

        # LOD
        self.lod_enabled = False
        self._terrain_lod_manager = None

    def clear_all_caches(self):
        """Clear all terrain mesh caches (baked meshes and LOD tiles).

        Call this when terrain data or resolution changes to ensure no
        stale geometry survives across the different caching layers.
        """
        self._baked_mesh_cache.clear()
        if self._terrain_lod_manager is not None:
            self._terrain_lod_manager._tile_cache.clear()
            self._terrain_lod_manager._tile_lods.clear()
            self._terrain_lod_manager._pyramid_cache.clear()
