"""Declarative key-binding tables for the interactive viewer.

Three dispatch tables map key events to method names on InteractiveViewer:

- ``SHIFT_BINDINGS``: Checked first.  Maps uppercase ``raw_key`` (Shift
  held) to a method name.  E.g. ``'O'`` → ``'_action_shift_o'``.
- ``KEY_BINDINGS``: Checked second.  Maps lowercase ``key`` to a method
  name.  E.g. ``'t'`` → ``'_action_toggle_shadows'``.
- ``SPECIAL_BINDINGS``: Checked last.  Maps ``(raw_key, key)`` tuples
  for keys that need both values (e.g. ``r`` vs ``R``).

Movement keys (WASD, arrows, IJKL, Q/E, PageUp/Down) are tracked in
``MOVEMENT_KEYS`` and handled separately by adding to ``_held_keys``.

Observer slots 1-8 are handled separately in ``_handle_key_press``.
"""

# Keys that get added to _held_keys for continuous movement/look
MOVEMENT_KEYS = frozenset({
    'w', 's', 'a', 'd',
    'up', 'down', 'left', 'right',
    'q', 'e', 'pageup', 'pagedown',
    'i', 'j', 'k', 'l',
})

# Shift+<key> bindings — checked first (raw_key is uppercase)
SHIFT_BINDINGS = {
    'O': '_action_shift_o',         # Cycle drone mode
    'V': '_action_shift_v',         # Snap to observer
    'K': '_action_clear_observers', # Kill all observers
    'F': '_action_toggle_firms',    # FIRMS fire layer
    'W': '_action_toggle_wind',     # Wind particles
    'E': '_action_toggle_terrain_vis',  # Toggle terrain visibility
    'B': '_action_toggle_gtfs_rt',  # GTFS-RT vehicles
    'C': '_action_cycle_pc_colors', # Point cloud color mode
    'D': '_action_toggle_denoiser', # Denoiser
    'G': '_action_cycle_gi',        # GI bounces
    'H': '_action_prev_help_page',  # Previous help page
    'L': '_action_toggle_drone_glow',  # Drone glow
    'T': '_action_cycle_time',      # Time-of-day
    'Y': '_action_toggle_hydro',    # Hydro flow particles
    'N': '_action_toggle_clouds',   # Cloud layer
}

# Lowercase key bindings — checked after shift bindings
KEY_BINDINGS = {
    't': '_action_toggle_shadows',
    'c': '_action_cycle_colormap',
    'g': '_cycle_terrain_layer',
    'n': '_cycle_geometry_layer',
    'p': '_action_jump_prev_geom',
    'h': '_action_next_help_page',
    'm': '_action_toggle_minimap',
    'o': '_action_place_observer',
    'v': '_toggle_viewshed',
    '[': '_action_observer_elev_down',
    ']': '_action_observer_elev_up',
    'f': '_save_screenshot',
    'y': '_action_cycle_color_stretch',
    'b': '_action_cycle_mesh_type',
    'u': '_action_cycle_basemap_fwd',
    ',': '_action_overlay_alpha_down',
    '.': '_action_overlay_alpha_up',
    '0': '_action_toggle_ao',
    '9': '_action_toggle_dof',
    ';': '_action_dof_aperture_down',
    "'": '_action_dof_aperture_up',
    'escape': '_action_exit',
    'x': '_action_exit',
}

# Keys that need both raw_key and key for dispatch
# (raw_key, key) → method name
SPECIAL_BINDINGS = {
    # Speed
    ('+', '+'): '_action_speed_up',
    ('=', '='): '_action_speed_up',
    ('-', '-'): '_action_speed_down',
    # Resolution: r = coarser, R = finer
    ('r', 'r'): '_action_resolution_coarser',
    ('R', 'r'): '_action_resolution_finer',
    # Vertical exaggeration: z = decrease, Z = increase
    ('z', 'z'): '_action_ve_down',
    ('Z', 'z'): '_action_ve_up',
    # Basemap: U = reverse
    ('U', 'u'): '_action_cycle_basemap_rev',
    # DOF focal distance: : = decrease, " = increase
    (':', ':'): '_action_dof_focal_down',
    ('"', '"'): '_action_dof_focal_up',
}
