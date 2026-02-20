# Enhancement Proposal: Jupyter Notebook Integration for explore()

## Context
The interactive terrain viewer (`explore()`) currently requires GLFW+ModernGL — a native window. This doesn't work in Jupyter notebooks. The goal is to stream ray-traced frames into a Jupyter widget with interactive mouse/keyboard controls, reusing the existing GPU rendering pipeline.

**Key architectural insight**: `InteractiveViewer` already separates rendering from display. `_render_frame()`, `_update_frame()`, `_composite_overlays()`, all input handlers, and all camera/scene state are GLFW-independent. Only `run()` (line 4973) touches GLFW.

## Approach
Create `JupyterViewer(InteractiveViewer)` that overrides `run()` to use `ipywidgets.Image` + `ipyevents` instead of GLFW+ModernGL. A background thread renders frames and pushes JPEG bytes to the widget. Auto-detect Jupyter in the `explore()` function.

## Files

### 1. NEW: `rtxpy/notebook.py` (~350 lines)

**`JupyterViewer(InteractiveViewer)`** — inherits everything, overrides only `run()`.

```python
class JupyterViewer(InteractiveViewer):
    def run(self, start_position=None, look_at=None):
        """Non-blocking. Returns ipywidgets.Image for display."""
        # 1. Camera init (same 20 lines as GLFW run(), lines 5001-5022)
        # 2. Create ipywidgets.Image widget
        # 3. Attach ipyevents for mouse/keyboard capture
        # 4. Start background render thread
        # 5. Return widget
```

**Render thread** (`_jupyter_render_loop`):
```
loop:
    drain input queue → route to _handle_key_press / _handle_mouse_motion / etc
    _tick()            → updates camera from held keys, triggers _update_frame()
    if _frame_dirty:
        encode _display_frame → JPEG bytes → widget.value
    sleep 8ms if idle
```

**Frame encoding** (`_jupyter_push_frame`):
```python
img_uint8 = (np.clip(self._display_frame, 0, 1) * 255).astype(np.uint8)
buf = io.BytesIO()
Image.fromarray(img_uint8).save(buf, format='JPEG', quality=85)
self._widget.value = buf.getvalue()
```
~3ms per frame at 1024×768. JPEG chosen over PNG for 5-10x faster encoding.

**Input mapping** (`_map_browser_key`):
Browser key events → rtxpy `(raw_key, key_lower)` format.
- `'ArrowUp'` → `('up', 'up')`
- `'a'` → `('a', 'a')`, `'A'` (shift) → `('A', 'a')`
- Special keys: `[`, `]`, `+`, `-`, etc. via lookup table

**`stop()`**: Sets `running = False`, joins render thread.

**`_detect_jupyter()`**: Module-level helper — checks `IPython.get_ipython()` for `'IPKernelApp'`.

### 2. MODIFY: `rtxpy/engine.py`

**`explore()` function (line 5376)** — add Jupyter auto-detection before viewer creation:

```python
if _detect_jupyter():
    from .notebook import JupyterViewer
    ViewerClass = JupyterViewer
else:
    ViewerClass = InteractiveViewer
viewer = ViewerClass(raster, width=width, ...)
```
Then at line 5434: `return viewer.run(...)` — returns widget in Jupyter, None in GLFW.

### 3. MODIFY: `rtxpy/accessor.py`

Both `RTXAccessor.explore()` (line 2024) and `DatasetRTXAccessor.explore()` (line 2472) call `engine.explore()` which handles the auto-detection. The accessor `return` value from `explore()` will be the widget in Jupyter (callers can `display()` it or let Jupyter auto-display).

No `explore_notebook()` method needed — auto-detection handles it transparently.

### 4. MODIFY: `pyproject.toml`

Add optional dependency group:
```toml
notebook = ["ipywidgets>=8.0", "ipyevents>=2.0", "Pillow"]
```
Update `all` extra to include notebook deps.

### 5. MODIFY: `rtxpy/__init__.py`

Add conditional export:
```python
try:
    from .notebook import JupyterViewer
except ImportError:
    pass
```

## Key Design Details

### Thread Safety
- Input queue: `queue.Queue(maxsize=100)` — browser events → render thread
- `_display_frame`: Written only by render thread, read by frame push (same thread)
- `widget.value = bytes`: Thread-safe in ipywidgets

### Frame Rate
- Target 15 FPS display (67ms period)
- Render on demand (only when camera changes)
- Continuous mode for wind/AO accumulation at same 15fps
- 8ms idle sleep → ~120Hz input polling

### ipyevents Usage
```python
from ipyevents import Event
event = Event(source=widget, watched_events=[
    'keydown', 'keyup', 'mousedown', 'mouseup', 'mousemove', 'wheel'
])
event.on_dom_event(self._handle_dom_event)
```
Each event dispatches to existing `_handle_key_press()`, `_handle_mouse_motion()`, etc.

### Existing Methods Reused (no changes needed)
- `_render_frame()` (engine.py:4393) — GPU ray trace
- `_update_frame()` (engine.py:4491) — full pipeline with AO, denoise, bloom
- `_composite_overlays()` (engine.py:4602) — wind, minimap, help
- `_handle_key_press()` (engine.py:2823) — all keyboard shortcuts
- `_handle_mouse_press/release/motion()` (engine.py:4687-4767)
- `_handle_scroll()` (engine.py:4672) — FOV zoom
- `_tick()` (engine.py:3298) — continuous camera movement
- `_render_help_text()`, `_compute_minimap_background()`

## Usage

```python
# In a Jupyter notebook cell:
widget = dem.rtx.explore(width=1024, height=768)
# Widget auto-displays. Click to focus, then use keyboard/mouse.

# To stop:
widget._viewer.stop()
```

## Verification
1. `pip install ipywidgets ipyevents` in the notebook environment
2. Open a notebook, load a DEM, call `.explore()`
3. Widget should display rendered terrain in the cell output
4. Mouse drag → pan, scroll → zoom
5. Keyboard: W/S/A/D movement, C cycle colormap, T toggle shadows, G cycle layers
6. Wind particles (Shift+W) should animate
7. `.stop()` cleanly exits render thread

## Dependencies
- `ipywidgets >= 8.0` (widget framework)
- `ipyevents >= 2.0` (keyboard/mouse event capture on widgets)
- `Pillow` (JPEG encoding — already required by viewer)
- All optional — GLFW path still works when not in Jupyter
