"""Input state tracking for the interactive viewer."""


class InputState:
    """Tracks keyboard and mouse input state.

    Holds the set of currently-pressed movement keys and mouse drag
    state, decoupled from the viewer's rendering logic.
    """

    __slots__ = ('held_keys', 'mouse_dragging', 'mouse_last_x', 'mouse_last_y')

    def __init__(self):
        self.held_keys = set()
        self.mouse_dragging = False
        self.mouse_last_x = None
        self.mouse_last_y = None
