"""Interactive terrain viewer using GLFW + ModernGL for display.

This package provides a game-engine-like render loop for exploring terrain
interactively with keyboard controls.

Architecture
------------
``InteractiveViewer`` (in ``core.py``) is a single class with no multiple
inheritance.  Its behaviour is split across **subsystem objects** that each
hold a back-reference to the viewer as ``self.v``:

    minimap.MinimapRenderer      — minimap background and blitting
    terrain_ops.TerrainOps       — LOD setup, resolution/VE rebuild
    weather.WeatherManager       — wind, clouds, rain, FIRMS
    hydro.HydroController        — hydro flow, GTFS-RT vehicles
    observers.ObserverController — drones, observers, viewshed
    renderer.FrameRenderer       — ray-trace rendering, screenshots
    input.InputHandler           — key/mouse dispatch, layer cycling
    hud.HUDRenderer              — title, legend, help overlays
    run.RunManager               — GLFW window + main loop

``delegate.py`` installs delegation properties (forwarding
``viewer.position`` to ``viewer.camera.position``, etc.) and thin forwarding
methods (routing ``viewer._update_frame()`` to the appropriate subsystem)
onto the class after its body is defined.
"""

from .core import InteractiveViewer
from .explore import explore

__all__ = ['InteractiveViewer', 'explore']
