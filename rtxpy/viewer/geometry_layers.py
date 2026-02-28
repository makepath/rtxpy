"""Geometry layer visibility management for the interactive viewer."""


class GeometryLayerManager:
    """Tracks GAS geometry groups and layer cycling state.

    Manages which geometry groups are visible, the current cycling
    index, per-layer positions for jump-to-geometry, and point-cloud
    color modes.
    """

    __slots__ = (
        'all_geometries', 'geometry_layer_order', 'geometry_layer_idx',
        'layer_positions', 'current_geom_idx',
        'pc_color_modes', 'pc_color_mode_idx',
        'chunk_manager', 'baked_meshes', 'geometry_colors_builder',
    )

    def __init__(self):
        self.all_geometries = []
        self.geometry_layer_order = ['none', 'all']
        self.geometry_layer_idx = 1  # Start at 'all'
        self.layer_positions = {}   # layer_name -> [(x, y, z, geometry_id), ...]
        self.current_geom_idx = 0
        self.pc_color_modes = ['elevation', 'intensity', 'classification', 'rgb']
        self.pc_color_mode_idx = 0
        self.chunk_manager = None   # set by explore() when scene_zarr provided
        self.baked_meshes = {}
        self.geometry_colors_builder = None
