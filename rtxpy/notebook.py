"""Jupyter notebook integration for the interactive terrain viewer.

Provides ``JupyterViewer``, a subclass of ``InteractiveViewer`` that
renders frames into an ``ipywidgets.Image`` widget with mouse/keyboard
input via ``ipyevents``.  All GPU rendering, camera logic, overlays,
and keyboard shortcuts are inherited unchanged.
"""

import io
import queue
import threading
import time
from typing import Optional, Tuple

import numpy as np

from .engine import InteractiveViewer


# ---------------------------------------------------------------------------
# Jupyter environment detection
# ---------------------------------------------------------------------------

def _detect_jupyter() -> bool:
    """Return True if running inside a Jupyter kernel."""
    try:
        from IPython import get_ipython
        ip = get_ipython()
        if ip is None:
            return False
        return ip.__class__.__module__.startswith('ipykernel')
    except ImportError:
        return False


# ---------------------------------------------------------------------------
# Browser key → rtxpy key mapping
# ---------------------------------------------------------------------------

_BROWSER_SPECIAL_KEYS = {
    'ArrowUp': 'up',
    'ArrowDown': 'down',
    'ArrowLeft': 'left',
    'ArrowRight': 'right',
    'PageUp': 'pageup',
    'PageDown': 'pagedown',
    'Escape': 'escape',
    'Equal': '=',
    'Minus': '-',
    'Comma': ',',
    'Period': '.',
    'BracketLeft': '[',
    'BracketRight': ']',
    'Semicolon': ';',
    'Quote': "'",
}

# Shift variants for special keys (mirrors _glfw_to_key)
_SHIFT_SPECIAL = {
    '=': '+',
    '-': '_',
    ';': ':',
    "'": '"',
}


def _map_browser_key(event: dict) -> Tuple[str, str]:
    """Convert a browser keyboard event to (raw_key, key_lower).

    Returns ('', '') for unmapped keys.
    """
    key = event.get('key', '')
    code = event.get('code', '')
    shift = event.get('shiftKey', False)

    # Special keys (arrows, page up/down, etc.)
    if code in _BROWSER_SPECIAL_KEYS:
        raw = _BROWSER_SPECIAL_KEYS[code]
        if shift and raw in _SHIFT_SPECIAL:
            raw = _SHIFT_SPECIAL[raw]
        return raw, raw.lower()

    # Letter keys
    if len(key) == 1 and key.isalpha():
        lower = key.lower()
        if shift:
            return lower.upper(), lower
        return lower, lower

    # Digit keys
    if len(key) == 1 and key.isdigit():
        return key, key

    # Direct single-char keys (+, -, etc.)
    if len(key) == 1:
        return key, key.lower()

    return '', ''


# ---------------------------------------------------------------------------
# JavaScript for keyboard/scroll isolation
# ---------------------------------------------------------------------------

# Injected into the notebook output to prevent keyboard events from
# reaching the notebook's own shortcut handlers (cell navigation,
# command-mode shortcuts like A/B/C/D/H/M/X, arrow scrolling, etc.).
# Also prevents scroll-wheel from scrolling the notebook page.
#
# When the widget is focused (blue border), all keyboard and wheel
# events are captured exclusively by the viewer.
_KEYBOARD_CAPTURE_JS = """
<script>
(function() {
    function setup() {
        var el = document.querySelector('.rtxpy-viewer');
        if (!el) { setTimeout(setup, 300); return; }

        el.setAttribute('tabindex', '0');
        el.style.outline = 'none';
        el.style.cursor = 'crosshair';

        // Tell Lumino (JupyterLab) to skip shortcut processing for
        // keyboard events originating from this element.  ipyevents
        // already captures keyboard via its own document-level listener,
        // so we just need Lumino to not interfere.
        el.setAttribute('data-lm-suppress-shortcuts', 'true');

        // Visual focus indicator
        el.addEventListener('focus', function() {
            el.style.outline = '2px solid #4a9eff';
            // Classic Notebook: disable its keyboard manager
            if (window.IPython && IPython.keyboard_manager) {
                IPython.keyboard_manager.disable();
            }
        });
        el.addEventListener('blur', function() {
            el.style.outline = 'none';
            if (window.IPython && IPython.keyboard_manager) {
                IPython.keyboard_manager.enable();
            }
        });

        // Stop wheel from scrolling the notebook page
        el.addEventListener('wheel', function(e) {
            e.stopPropagation();
            e.preventDefault();
        }, {passive: false});

        // Auto-focus on click
        el.addEventListener('mousedown', function() {
            el.focus();
        });
    }
    setup();
})();
</script>
"""


# ---------------------------------------------------------------------------
# JupyterViewer
# ---------------------------------------------------------------------------

class JupyterViewer(InteractiveViewer):
    """Interactive terrain viewer for Jupyter notebooks.

    Inherits all rendering, camera, and input logic from
    ``InteractiveViewer``.  Overrides ``run()`` to display frames in
    an ``ipywidgets.Image`` widget instead of a GLFW window.
    """

    def _render_help_text(self):
        """Render help text, scaling to fit the render resolution."""
        super()._render_help_text()
        if self._help_text_rgba is None:
            return
        ht = self._help_text_rgba
        hh, hw = ht.shape[:2]
        max_w = self.render_width - 16
        max_h = self.render_height - 16
        if hw > max_w or hh > max_h:
            scale = min(max_w / hw, max_h / hh)
            new_w = max(1, int(hw * scale))
            new_h = max(1, int(hh * scale))
            from PIL import Image
            img = Image.fromarray((ht * 255).astype(np.uint8), 'RGBA')
            img = img.resize((new_w, new_h), Image.LANCZOS)
            self._help_text_rgba = np.array(img, dtype=np.float32) / 255.0

    def _handle_key_press(self, raw_key, key):
        """Override to suppress exit keys in Jupyter.

        In Jupyter, 'x' and 'escape' shouldn't kill the viewer — too easy
        to hit accidentally.  Use ``widget.stop()`` instead.
        """
        if key in ('escape', 'x'):
            return
        super()._handle_key_press(raw_key, key)

    def run(self, start_position: Optional[Tuple[float, float, float]] = None,
            look_at: Optional[Tuple[float, float, float]] = None):
        """Start the viewer and return an interactive widget.

        Parameters
        ----------
        start_position : tuple, optional
            Starting camera position (x, y, z).
        look_at : tuple, optional
            Initial look-at point.

        Returns
        -------
        ipywidgets.Image
            The widget displaying rendered frames.  The widget has a
            ``_viewer`` attribute pointing back to this viewer and a
            ``stop()`` method to shut down the render thread.
        """
        import ipywidgets as widgets
        from ipyevents import Event
        from IPython.display import display, HTML

        H, W = self.terrain_shape
        world_W = W * self.pixel_spacing_x
        world_H = H * self.pixel_spacing_y
        world_diag = np.sqrt(world_W**2 + world_H**2)

        if self.move_speed is None:
            self.move_speed = world_diag * 0.01

        if start_position is None:
            start_position = (
                world_W / 2,
                world_H * 1.05,
                self.elev_max + world_diag * 0.08,
            )

        self.position = np.array(start_position, dtype=np.float32)

        if look_at is not None:
            direction = np.array(look_at) - self.position
            direction = direction / (np.linalg.norm(direction) + 1e-8)
            self.yaw = np.degrees(np.arctan2(direction[1], direction[0]))
            self.pitch = np.degrees(np.arcsin(np.clip(direction[2], -1, 1)))
        else:
            center = np.array([world_W / 2, world_H / 2, self.elev_mean])
            direction = center - self.position
            direction = direction / (np.linalg.norm(direction) + 1e-8)
            self.yaw = np.degrees(np.arctan2(direction[1], direction[0]))
            self.pitch = np.degrees(np.arcsin(np.clip(direction[2], -1, 1)))

        # Pre-render overlays (help text scales to fit render resolution)
        self._render_help_text()
        self._compute_minimap_background()

        # --- Widget setup ---
        self._widget = widgets.Image(
            format='jpeg',
            width=self.width,
            height=self.height,
        )
        self._widget.add_class('rtxpy-viewer')
        self._widget._viewer = self
        self._widget.stop = self.stop

        # --- Input event handling ---
        self._input_queue = queue.Queue(maxsize=200)

        event_handler = Event(
            source=self._widget,
            watched_events=[
                'keydown', 'keyup',
                'mousedown', 'mouseup', 'mousemove',
                'wheel',
            ],
            prevent_default_action=True,
            wait=0,
        )
        event_handler.on_dom_event(self._handle_dom_event)
        self._event_handler = event_handler

        # --- State ---
        self.running = True
        self._display_frame = None
        self._frame_dirty = False
        self._render_needed = True
        self._fps_counter = 0
        self._fps_last_time = time.monotonic()
        self._last_tick_time = time.monotonic()

        # Render first frame synchronously so widget isn't blank
        self._tick()
        self._push_frame()

        # --- Background render thread ---
        self._render_thread = threading.Thread(
            target=self._jupyter_render_loop,
            daemon=True,
            name='rtxpy-jupyter-render',
        )
        self._render_thread.start()

        print(f"rtxpy viewer ({self.width}x{self.height}) — click image to focus, H for help, widget.stop() to exit")

        # Display widget + keyboard isolation JavaScript
        display(self._widget)
        display(HTML(_KEYBOARD_CAPTURE_JS))
        return self._widget

    def stop(self):
        """Stop the render thread and release resources."""
        self.running = False
        if hasattr(self, '_render_thread') and self._render_thread.is_alive():
            self._render_thread.join(timeout=2.0)
        if self._tile_service is not None:
            self._tile_service.shutdown()

    # --- DOM event handler (runs in Jupyter comm thread) ---

    def _handle_dom_event(self, event):
        """Queue a browser DOM event for processing by the render thread."""
        try:
            self._input_queue.put_nowait(event)
        except queue.Full:
            pass  # drop event if queue is full

    # --- Render loop (background thread) ---

    def _jupyter_render_loop(self):
        """Background thread: process input, tick, push frames."""
        target_period = 1.0 / 15  # ~15 FPS display rate

        while self.running:
            loop_start = time.monotonic()

            # Drain input queue
            while True:
                try:
                    event = self._input_queue.get_nowait()
                except queue.Empty:
                    break
                self._dispatch_dom_event(event)

            # Process REPL command queue
            while True:
                try:
                    cmd = self._command_queue.get_nowait()
                except queue.Empty:
                    break
                try:
                    cmd(self)
                except Exception:
                    import traceback
                    traceback.print_exc()
                self._render_needed = True

            # Tick (movement, rendering)
            self._tick()

            # Push frame to widget if dirty
            if self._frame_dirty and self._display_frame is not None:
                self._push_frame()
                self._frame_dirty = False

            # Sleep to maintain target frame rate
            elapsed = time.monotonic() - loop_start
            sleep_time = target_period - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)
            elif not self._held_keys and not self._mouse_dragging:
                time.sleep(0.008)

    def _push_frame(self):
        """Encode the current display frame as JPEG and update the widget."""
        frame = self._display_frame
        if frame is None:
            return

        from PIL import Image

        img_uint8 = (np.clip(frame, 0, 1) * 255).astype(np.uint8)
        buf = io.BytesIO()
        Image.fromarray(img_uint8).save(buf, format='JPEG', quality=85)
        self._widget.value = buf.getvalue()

    # --- DOM event dispatch ---

    def _dispatch_dom_event(self, event):
        """Route a browser DOM event to the appropriate handler."""
        etype = event.get('type', '')

        if etype == 'keydown':
            raw_key, key_lower = _map_browser_key(event)
            if raw_key:
                self._handle_key_press(raw_key, key_lower)

        elif etype == 'keyup':
            raw_key, key_lower = _map_browser_key(event)
            if key_lower:
                self._handle_key_release(key_lower)

        elif etype == 'mousedown':
            button = event.get('button', 0)
            x = event.get('offsetX', 0)
            y = event.get('offsetY', 0)
            self._handle_mouse_press(button, x, y)

        elif etype == 'mouseup':
            button = event.get('button', 0)
            self._handle_mouse_release(button)

        elif etype == 'mousemove':
            x = event.get('offsetX', 0)
            y = event.get('offsetY', 0)
            self._handle_mouse_motion(x, y)

        elif etype == 'wheel':
            dy = event.get('deltaY', 0)
            # Browser deltaY: positive = scroll down
            if dy != 0:
                self._handle_scroll(-1 if dy > 0 else 1)
