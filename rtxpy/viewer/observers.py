"""Observer management for the interactive viewer."""

import threading


OBSERVER_COLORS = [
    (1.0, 0.2, 0.2),   # 1: red
    (0.2, 0.6, 1.0),   # 2: blue
    (0.2, 1.0, 0.3),   # 3: green
    (1.0, 0.8, 0.1),   # 4: yellow
    (1.0, 0.4, 0.0),   # 5: orange
    (0.8, 0.2, 1.0),   # 6: purple
    (0.0, 1.0, 0.9),   # 7: cyan
    (1.0, 0.5, 0.7),   # 8: pink
]


class Observer:
    """State for a single observer slot (1-8)."""

    __slots__ = (
        'slot', 'position', 'observer_elev', 'drone_mode', 'drone_placed',
        'yaw', 'pitch', 'saved_camera', 'tour_thread', 'tour_stop',
        'viewshed_enabled', 'viewshed_cache',
    )

    def __init__(self, slot, position, observer_elev=0.05):
        self.slot = slot
        self.position = position
        self.observer_elev = observer_elev
        self.drone_mode = 'off'
        self.drone_placed = False
        self.yaw = 0.0
        self.pitch = 0.0
        self.saved_camera = None
        self.tour_thread = None
        self.tour_stop = threading.Event()
        self.viewshed_enabled = False
        self.viewshed_cache = None

    @property
    def color(self):
        return OBSERVER_COLORS[(self.slot - 1) % len(OBSERVER_COLORS)]

    def geometry_id(self, part_idx):
        """Unique geometry ID for a drone sub-mesh."""
        return f'_observer{self.slot}_{part_idx}'

    def is_touring(self):
        return self.tour_thread is not None and self.tour_thread.is_alive()

    def stop_tour(self):
        self.tour_stop.set()
        if self.tour_thread is not None:
            self.tour_thread.join(timeout=2.0)
            self.tour_thread = None
        self.tour_stop.clear()


class ObserverManager:
    """Manages multi-observer system (up to 8 independent observers).

    Holds observer instances, viewshed settings, and drone part state.
    """

    __slots__ = (
        'observers', 'active_observer',
        'viewshed_enabled', 'viewshed_observer_elev',
        'viewshed_target_elev', 'viewshed_opacity',
        'viewshed_cache', 'viewshed_coverage',
        'viewshed_recalc_interval', 'last_viewshed_time',
        'shared_drone_parts', 'drone_glow',
    )

    def __init__(self):
        self.observers = {}             # dict[int, Observer] — slot 1-8
        self.active_observer = None     # int (slot 1-8) or None
        self.viewshed_enabled = False
        self.viewshed_observer_elev = 0.05
        self.viewshed_target_elev = 0.0
        self.viewshed_opacity = 0.35
        self.viewshed_cache = None
        self.viewshed_coverage = 0.0
        self.viewshed_recalc_interval = 0.4
        self.last_viewshed_time = 0.0
        self.shared_drone_parts = None
        self.drone_glow = False
