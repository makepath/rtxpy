"""Overlay layer management for the interactive viewer."""


class OverlayManager:
    """Manages terrain overlay layers and basemap tile settings.

    Tracks which overlay is active, the alpha blending value, basemap
    cycling state, and tile service reference.
    """

    __slots__ = (
        'overlay_layers', 'overlay_names',
        'active_color_data', 'active_overlay_data',
        'active_overlay_color_lut',
        'overlay_color_luts',
        'overlay_alpha', 'overlay_as_water',
        'terrain_layer_order', 'terrain_layer_idx',
        'base_overlay_layers',
        'tile_service', 'tiles_enabled',
        'basemap_options', 'basemap_idx',
    )

    def __init__(self, overlay_layers=None, base_overlay_layers=None):
        self.overlay_layers = overlay_layers or {}
        self.overlay_names = list(self.overlay_layers.keys())
        self.active_color_data = None
        self.active_overlay_data = None
        self.active_overlay_color_lut = None
        self.overlay_color_luts = {}
        self.overlay_alpha = 0.7
        self.overlay_as_water = False

        self.terrain_layer_order = ['elevation'] + list(self.overlay_names)
        self.terrain_layer_idx = 0

        self.base_overlay_layers = base_overlay_layers or {}

        self.tile_service = None
        self.tiles_enabled = False
        self.basemap_options = ['none', 'satellite', 'osm']
        self.basemap_idx = 0
