"""HUD (heads-up display) state for the interactive viewer."""


class HUDState:
    """Holds title, subtitle, legend, help pages, and minimap state.

    Groups all on-screen overlay / HUD variables into a single object.
    """

    __slots__ = (
        'title', 'subtitle', 'legend_config', 'info_text',
        'title_overlay_rgba', 'legend_rgba',
        'help_page_idx', 'help_pages',
        'show_minimap',
        'last_title', 'last_subtitle',
        'minimap_background', 'minimap_scale_x', 'minimap_scale_y',
        'minimap_has_tiles', 'minimap_rect',
        'minimap_world_extent',
        'minimap_style', 'minimap_layer', 'minimap_colors',
        'minimap_bg_extent', 'minimap_last_stream_time',
    )

    def __init__(self, title='rtxpy', subtitle=None, legend=None):
        self.title = title
        self.subtitle = subtitle
        self.legend_config = legend
        self.info_text = None
        self.title_overlay_rgba = None
        self.legend_rgba = None
        self.help_page_idx = 0   # -1 = off, 0..N-1 = page index
        self.help_pages = []
        self.show_minimap = True
        self.last_title = None
        self.last_subtitle = None

        # Minimap state (initialized in run() via _compute_minimap_background)
        self.minimap_background = None
        self.minimap_scale_x = 1.0
        self.minimap_scale_y = 1.0
        self.minimap_has_tiles = False
        self.minimap_rect = None
        self.minimap_world_extent = None  # (wx_min, wy_min, wx_max, wy_max)
        self.minimap_style = None
        self.minimap_layer = None
        self.minimap_colors = None
        self.minimap_bg_extent = None  # (wx_min, wy_min, wx_max, wy_max)
        self.minimap_last_stream_time = 0.0
