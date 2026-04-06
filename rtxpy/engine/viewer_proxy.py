"""Thread-safe proxy handle to the running InteractiveViewer."""

from __future__ import annotations

import threading

import numpy as np

from .helpers import _add_overlay, _bilinear_terrain_z
from ..viewer.observers import Observer


class ViewerProxy:
    """Thread-safe handle to the running InteractiveViewer.

    Exposed as ``v`` (and ``viewer``) in the REPL started by
    ``explore(repl=True)``.  Methods push callables onto a queue that
    the main GLFW thread drains each tick, so OptiX calls always
    happen on the correct thread.
    """

    def __init__(self, viewer: 'InteractiveViewer'):
        self._viewer = viewer
        self._q = viewer._command_queue

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _submit(self, fn):
        """Push *fn* to the render thread and block until it completes.

        ``fn(viewer)`` is called on the main thread.  Returns whatever
        ``fn`` returns.
        """
        result = [None]
        error = [None]
        event = threading.Event()

        def _wrapped(viewer):
            try:
                result[0] = fn(viewer)
            except Exception as exc:
                error[0] = exc
            finally:
                event.set()

        self._q.put(_wrapped)
        event.wait(timeout=30)
        if error[0] is not None:
            raise error[0]
        return result[0]

    def _submit_fire_and_forget(self, fn):
        """Push *fn* without blocking the caller."""
        self._q.put(fn)

    # ------------------------------------------------------------------
    # Read-only state (safe to access from any thread)
    # ------------------------------------------------------------------

    @property
    def raster(self):
        """The current terrain DataArray (may be subsampled)."""
        return self._viewer.raster

    @property
    def base_raster(self):
        """The full-resolution terrain DataArray."""
        return self._viewer._base_raster

    @property
    def position(self):
        """Current camera position ``(x, y, z)`` as a numpy array copy."""
        return self._viewer.position.copy()

    @property
    def yaw(self):
        return self._viewer.yaw

    @property
    def pitch(self):
        return self._viewer.pitch

    @property
    def shadows(self):
        return self._viewer.shadows

    @shadows.setter
    def shadows(self, value):
        def fn(v):
            v.shadows = bool(value)
            v._update_frame()
        self._submit(fn)

    @property
    def colormap(self):
        return self._viewer.colormap

    @colormap.setter
    def colormap(self, value):
        self.set_colormap(value)

    @property
    def vertical_exaggeration(self):
        return self._viewer.vertical_exaggeration

    @property
    def overlay_names(self):
        """List of available overlay layer names."""
        return list(self._viewer._overlay_layers.keys())

    # ------------------------------------------------------------------
    # Terrain analysis  (run on main thread → display result)
    # ------------------------------------------------------------------

    def hillshade(self, **kwargs):
        """Compute hillshade on the current terrain and show it."""
        def fn(v):
            acc = v._accessor
            if acc is None:
                print("No accessor — cannot compute hillshade")
                return
            data = acc.hillshade(**kwargs)
            _add_overlay(v, 'hillshade', data.data)
        self._submit(fn)

    def viewshed(self, x, y, observer_elev=2, **kwargs):
        """Compute viewshed and show it as an overlay."""
        def fn(v):
            acc = v._accessor
            if acc is None:
                print("No accessor — cannot compute viewshed")
                return
            data = acc.viewshed(x=x, y=y, observer_elev=observer_elev,
                                **kwargs)
            _add_overlay(v, 'viewshed', data.data)
        self._submit(fn)

    def slope(self, **kwargs):
        """Compute slope and show it as an overlay."""
        def fn(v):
            acc = v._accessor
            if acc is None:
                print("No accessor — cannot compute slope")
                return
            data = acc.slope(**kwargs)
            _add_overlay(v, 'slope', data.data)
        self._submit(fn)

    def aspect(self, **kwargs):
        """Compute aspect and show it as an overlay."""
        def fn(v):
            acc = v._accessor
            if acc is None:
                print("No accessor — cannot compute aspect")
                return
            data = acc.aspect(**kwargs)
            _add_overlay(v, 'aspect', data.data)
        self._submit(fn)

    # ------------------------------------------------------------------
    # Picking
    # ------------------------------------------------------------------

    def pick(self, screen_x, screen_y):
        """Pick geometry at screen coordinates. Returns hit info dict."""
        def fn(v):
            origin, direction = v._screen_to_ray(screen_x, screen_y)
            return v.rtx.pick(origin, direction)
        return self._submit(fn)

    # ------------------------------------------------------------------
    # Layer management
    # ------------------------------------------------------------------

    def add_layer(self, name, data):
        """Add (or replace) a named overlay layer and switch to it.

        Parameters
        ----------
        name : str
            Layer name shown when cycling with G.
        data : array-like
            2-D array (numpy or cupy) matching the terrain shape.
        """
        def fn(v):
            _add_overlay(v, name, data)
        self._submit(fn)

    def add_hydro(self, flow_accum, **kwargs):
        """Add hydrological flow particle visualization.

        Uses MFD (Multiple Flow Direction) to compute flow vectors from
        terrain elevation so particles follow natural drainage paths
        distributed across all downhill neighbors.

        Parameters
        ----------
        flow_accum : array-like, shape (H, W)
            Flow accumulation grid (cell counts or area).  Compute with
            ``xrspatial.flow_accumulation()``.
        **kwargs
            Optional overrides: n_particles, max_age, trail_len, speed,
            accum_threshold, color, alpha, dot_radius, min_visible_age,
            flow_dir_mfd (xrspatial MFD fractions),
            elevation (conditioned DEM for manual MFD fallback).
        """
        stream_order = kwargs.get('stream_order')
        stream_link = kwargs.get('stream_link')
        def fn(v):
            from .core import InteractiveViewer
            v._init_hydro(flow_accum, **kwargs)
            v._hydro_enabled = True
            # Add stream link overlay with palette-matched colors
            if stream_link is not None:
                sl_np = stream_link.get() if hasattr(stream_link, 'get') else np.asarray(stream_link)
                sl_np = np.asarray(sl_np, dtype=np.float32)
                # Use stream order values for coloring, NaN where no stream
                if stream_order is not None:
                    so_for_sl = stream_order.get() if hasattr(stream_order, 'get') else np.asarray(stream_order)
                    so_for_sl = np.asarray(so_for_sl, dtype=np.float32)
                    max_order = int(np.nanmax(so_for_sl)) if np.any(~np.isnan(so_for_sl)) else 5
                    palette_lut = InteractiveViewer._build_stream_palette_lut(
                        max_order)
                    sl_color = np.where(
                        np.isnan(sl_np) | (sl_np <= 0) | np.isnan(so_for_sl) | (so_for_sl <= 0),
                        np.float32(np.nan), so_for_sl)
                    _add_overlay(v, 'stream_link', sl_color,
                                 color_lut=palette_lut)
                else:
                    sl_np = np.where(np.isnan(sl_np) | (sl_np <= 0),
                                     np.float32(np.nan), sl_np)
                    _add_overlay(v, 'stream_link', sl_np)
            v._update_frame()
        self._submit(fn)

    def remove_layer(self, name):
        """Remove an overlay layer by name."""
        def fn(v):
            if name in v._overlay_layers:
                del v._overlay_layers[name]
                if name in v._base_overlay_layers:
                    del v._base_overlay_layers[name]
                if name in v._overlay_color_luts:
                    del v._overlay_color_luts[name]
                v._overlay_names = list(v._overlay_layers.keys())
                v._terrain_layer_order = (
                    ['elevation'] + list(v._overlay_names))
                # Reset to elevation if we removed the active layer
                if v._terrain_layer_idx >= len(v._terrain_layer_order):
                    v._terrain_layer_idx = 0
                    v._active_overlay_data = None
                    v._overlay_as_water = False
                    v._active_overlay_color_lut = None
                v._update_frame()
                print(f"Removed layer: {name}")
        self._submit(fn)

    def show_layer(self, name):
        """Switch the terrain coloring to a named layer (or 'elevation')."""
        def fn(v):
            if name == 'elevation':
                v._active_color_data = None
                v._active_overlay_data = None
                v._overlay_as_water = False
                v._active_overlay_color_lut = None
                v._terrain_layer_idx = 0
                v._update_frame()
                print("Terrain: elevation")
                return
            if name not in v._overlay_layers:
                print(f"Unknown layer: {name}. "
                      f"Available: {list(v._overlay_layers.keys())}")
                return
            idx = v._terrain_layer_order.index(name)
            v._terrain_layer_idx = idx
            v._active_color_data = None
            v._active_overlay_data = v._overlay_layers[name]
            v._overlay_as_water = (
                name.startswith('flood_')
                or (name == 'stream_link' and v._hydro_enabled))
            v._active_overlay_color_lut = v._overlay_color_luts.get(name)
            v._update_frame()
            print(f"Terrain: {name}")
        self._submit(fn)

    # ------------------------------------------------------------------
    # Display settings
    # ------------------------------------------------------------------

    def set_colormap(self, cmap):
        """Change the active colormap by name (e.g. 'terrain', 'viridis')."""
        def fn(v):
            v.colormap = cmap
            v._update_frame()
            print(f"Colormap: {cmap}")
        self._submit(fn)

    def set_color_stretch(self, stretch):
        """Set the color stretch ('linear', 'sqrt', 'cbrt', 'log')."""
        def fn(v):
            if stretch in v._color_stretches:
                v.color_stretch = stretch
                v._color_stretch_idx = v._color_stretches.index(stretch)
                v._update_frame()
                print(f"Color stretch: {stretch}")
            else:
                print(f"Unknown stretch: {stretch}. "
                      f"Options: {v._color_stretches}")
        self._submit(fn)

    # ------------------------------------------------------------------
    # Misc
    # ------------------------------------------------------------------

    def screenshot(self):
        """Save a screenshot.  Uses the viewer's built-in logic."""
        def fn(v):
            v._save_screenshot()
        self._submit(fn)

    def run(self, fn):
        """Execute an arbitrary callable on the render thread.

        ``fn(viewer)`` receives the ``InteractiveViewer`` instance.
        Use this for anything not covered by a convenience method.
        """
        return self._submit(fn)

    # ------------------------------------------------------------------
    # Tour / scripting
    # ------------------------------------------------------------------

    def mark(self):
        """Capture current camera state as a keyframe dict.

        Fly to desired positions manually, call ``v.mark()`` to record
        them, then assemble the results into a tour keyframe list.
        """
        from ..tour import mark_camera
        kf = mark_camera(self)
        import pprint
        pprint.pprint(kf)
        return kf

    def tour(self, keyframes, fps=30, record=False, output_dir='.',
             loop=False):
        """Play a scripted camera tour.

        Parameters
        ----------
        keyframes : list of dict or str
            List of keyframe dicts, or path to a ``.py`` file that
            defines a ``tour`` variable containing the keyframe list.
        fps : int
            Target playback framerate.
        record : bool
            Save each frame as a PNG for video assembly.
        output_dir : str or Path
            Directory for recorded frames.
        loop : bool
            Repeat the tour indefinitely until the viewer closes.
        """
        from ..tour import play_tour
        if isinstance(keyframes, str):
            ns = {}
            with open(keyframes) as f:
                exec(f.read(), ns)
            keyframes = ns['tour']
            if 'loop' in ns:
                loop = ns['loop']
        play_tour(self, keyframes, fps=fps, record=record,
                  output_dir=output_dir, loop=loop)

    def show_geometry(self, name):
        """Show only a specific geometry group (or ``'all'`` / ``'none'``).

        Parameters
        ----------
        name : str
            Geometry group name, ``'all'`` to show everything, or
            ``'none'`` to hide all non-terrain geometries.
        """
        def fn(v):
            if name == 'all':
                for gid in v._all_geometries:
                    v.rtx.set_geometry_visible(gid, True)
                print("Geometry: all")
            elif name == 'none':
                for gid in v._all_geometries:
                    if gid != 'terrain':
                        v.rtx.set_geometry_visible(gid, False)
                print("Geometry: none")
            else:
                visible_count = 0
                for gid in v._all_geometries:
                    parts = gid.rsplit('_', 1)
                    base = parts[0] if len(parts) == 2 and parts[1].isdigit() else gid
                    visible = (base == name or gid == name or gid == 'terrain')
                    v.rtx.set_geometry_visible(gid, visible)
                    if visible:
                        visible_count += 1
                print(f"Geometry: {name} ({visible_count} visible)")
            v._update_frame()
        self._submit(fn)

    # ------------------------------------------------------------------
    # Multi-observer API
    # ------------------------------------------------------------------

    def place_observer(self, slot, x=None, y=None):
        """Create or move an observer in *slot* (1-8).

        Defaults to the current camera position if *x*/*y* are omitted.
        """
        def fn(v):
            # Observer is defined at module level in this file
            if slot not in v._observers:
                obs = Observer(slot, position=None,
                               observer_elev=v.viewshed_observer_elev)
                v._observers[slot] = obs
            obs = v._observers[slot]
            v._place_observer_at(obs, x=x, y=y)
            v._active_observer = slot
        self._submit(fn)

    def remove_observer(self, slot):
        """Remove an observer from *slot*."""
        def fn(v):
            v._clear_observer_slot(slot)
        self._submit(fn)

    def remove_all_observers(self):
        """Kill all observers — stop tours, exit drone modes, remove all."""
        def fn(v):
            v._clear_all_observers()
        self._submit(fn)

    def select_observer(self, slot):
        """Select an observer slot for keyboard control."""
        def fn(v):
            if slot in v._observers:
                v._active_observer = slot
                print(f"Observer {slot}: selected")
            else:
                print(f"Observer {slot} does not exist")
        self._submit(fn)

    def observer_tour(self, slot, keyframes, fps=30, loop=False):
        """Run a tour on an observer's drone.

        Parameters
        ----------
        slot : int
            Observer slot (1-8). Auto-created at first keyframe if needed.
        keyframes : list of dict or str
            Keyframe list, or path to a ``.py`` file containing a ``tour``
            variable.
        fps : int
            Target playback framerate.
        loop : bool
            Repeat indefinitely until stopped.
        """
        import threading as _threading
        from ..tour import play_observer_tour

        if isinstance(keyframes, str):
            ns = {}
            with open(keyframes) as f:
                exec(f.read(), ns)
            keyframes = ns['tour']
            if 'loop' in ns:
                loop = ns['loop']

        # Auto-create observer at first keyframe position if needed
        first_pos = None
        for kf in keyframes:
            if 'position' in kf:
                first_pos = kf['position']
                break

        def _setup(v):
            # Observer is defined at module level in this file
            if slot in v._observers:
                obs = v._observers[slot]
                obs.stop_tour()
            else:
                obs = Observer(slot, position=None,
                               observer_elev=v.viewshed_observer_elev)
                v._observers[slot] = obs
            if obs.position is None and first_pos is not None:
                v._place_observer_at(obs, x=first_pos[0], y=first_pos[1])
            elif obs.position is None:
                v._place_observer_at(obs)

        self._submit(_setup)

        def _tour_thread():
            play_observer_tour(self, slot, keyframes, fps=fps, loop=loop)

        obs = self._viewer._observers.get(slot)
        if obs is not None:
            obs.tour_stop.clear()
            t = _threading.Thread(target=_tour_thread, daemon=True)
            obs.tour_thread = t
            t.start()

    def stop_observer_tour(self, slot):
        """Stop a running tour on observer *slot*."""
        obs = self._viewer._observers.get(slot)
        if obs is not None:
            obs.stop_tour()
            print(f"Observer {slot} tour stopped")

    def observer_position(self, slot):
        """Get an observer's current (x, y, z) position."""
        obs = self._viewer._observers.get(slot)
        if obs is None:
            return None
        ox, oy = obs.position
        tz = self._viewer._get_terrain_z(ox, oy)
        return (ox, oy, tz + obs.observer_elev)

    def __repr__(self):
        v = self._viewer
        layers = ', '.join(v._overlay_layers.keys()) or '(none)'
        obs_info = ''
        if v._observers:
            obs_info = f", observers={list(v._observers.keys())}"
        return (f"ViewerProxy(layers=[{layers}], "
                f"colormap={v.colormap!r}, "
                f"shadows={v.shadows}{obs_info})")
