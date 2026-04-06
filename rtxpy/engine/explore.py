"""Public explore() function — convenience launcher for InteractiveViewer."""

from typing import Optional, Tuple

import numpy as np

from .helpers import _add_overlay


def explore(raster, width: int = 800, height: int = 600,
            render_scale: float = 0.5,
            start_position: Optional[Tuple[float, float, float]] = None,
            look_at: Optional[Tuple[float, float, float]] = None,
            key_repeat_interval: float = 0.05,
            rtx: 'RTX' = None,
            pixel_spacing_x: float = 1.0, pixel_spacing_y: float = 1.0,
            overlay_layers: dict = None,
            color_stretch: str = 'linear',
            title: str = None,
            subtitle: str = None,
            legend: dict = None,
            tile_service=None,
            geometry_colors_builder=None,
            baked_meshes=None,
            subsample: int = 1,
            wind_data=None,
            weather_data=None,
            hydro_data=None,
            gtfs_data=None,
            accessor=None,
            terrain_loader=None,
            tile_data_fn=None,
            scene_zarr=None,
            terrain_source=None,
            ao_samples: int = 0,
            gi_bounces: int = 1,
            denoise: bool = False,
            fog_density: float = 0.0,
            fog_color: tuple = (0.7, 0.8, 0.9),
            colormap: str = None,
            sun_azimuth: float = None,
            sun_altitude: float = None,
            shadows: bool = None,
            ambient: float = None,
            minimap_style: str = None,
            minimap_layer: str = None,
            minimap_colors: dict = None,
            info_text: str = None,
            skirt: bool = True,
            repl: bool = False,
            tour=None):
    """
    Launch an interactive terrain viewer.

    Uses GLFW + ModernGL for display.
    Keyboard controls allow flying through the terrain.

    Parameters
    ----------
    raster : xarray.DataArray
        Terrain raster data with cupy array.
    width : int
        Display width in pixels. Default is 800.
    height : int
        Display height in pixels. Default is 600.
    render_scale : float
        Render at this fraction of display size (0.25-1.0).
        Lower values give higher FPS. Default is 0.5.
    start_position : tuple, optional
        Starting camera position (x, y, z). If None, starts at the
        south edge looking north.
    look_at : tuple, optional
        Initial look-at point.
    key_repeat_interval : float
        Minimum seconds between key repeat events (default 0.05 = 20 FPS max).
        Lower values = more responsive but more GPU load.
    rtx : RTX, optional
        Existing RTX instance with geometries (e.g., from place_mesh).
        If provided, renders the full scene including placed meshes.
    pixel_spacing_x : float, optional
        X spacing between pixels in world units (e.g., 30.0 for 30m/pixel).
        Must match the spacing used when triangulating terrain. Default 1.0.
    pixel_spacing_y : float, optional
        Y spacing between pixels in world units. Default 1.0.
    scene_zarr : str or Path, optional
        Path to a zarr store with a ``meshes/`` group. When provided,
        mesh chunks are loaded dynamically based on camera position
        instead of loading the full scene upfront.
    accessor : RTXAccessor, optional
        RTX accessor instance for on-demand data fetching (e.g. FIRMS fire
        layer via Shift+F).
    wind_data : dict, optional
        Wind data from ``fetch_wind()``. If provided, Shift+W toggles
        wind particle animation.
    hydro_data : dict or True, optional
        Hydrological flow data.  Pass ``True`` or ``{'enabled': False}``
        for lazy mode: MFD flow analysis is computed on GPU from the
        current terrain when Shift+Y is first pressed.  Or pass a dict
        with key ``'flow_accum'`` for pre-computed data.  Optional keys:
        ``'flow_dir_mfd'`` (xrspatial MFD fractions, shape (8,H,W)),
        ``'n_particles'``, ``'max_age'``, ``'trail_len'``, ``'speed'``,
        ``'accum_threshold'``, ``'color'``, ``'alpha'``, ``'dot_radius'``.
    gtfs_data : dict, optional
        GTFS data from ``fetch_gtfs()``. If the metadata contains a
        ``realtime_url``, Shift+B toggles realtime vehicle positions.
    ao_samples : int, optional
        If > 0, enable ambient occlusion on launch with progressive
        accumulation (1 sample per frame). Press 0 to toggle at runtime.
        Default is 0 (disabled).
    denoise : bool, optional
        If True, enable the OptiX AI Denoiser on launch. Press Shift+D
        to toggle at runtime. Default is False.
    tour : list of dict or str, optional
        If provided, automatically play a camera tour after the viewer
        launches.  Can be a list of keyframe dicts or a path to a
        ``.py`` file that defines a ``tour`` variable.  Implies
        ``repl=True`` — the REPL starts after the tour finishes.

    Controls
    --------
    - W/Up: Move forward
    - S/Down: Move backward
    - A/Left: Strafe left
    - D/Right: Strafe right
    - Q/Page Up: Move up
    - E/Page Down: Move down
    - I/J/K/L: Look up/left/down/right
    - Click+Drag: Pan (slippy-map style)
    - Scroll wheel: Zoom in/out (FOV)
    - +/=: Increase speed
    - -: Decrease speed
    - G: Cycle terrain color (elevation → overlays)
    - U/Shift+U: Cycle basemap forward/backward (none → satellite → osm)
    - N: Cycle geometry layer (none → all → groups)
    - P: Jump to previous geometry in current group
    - Shift+C: Cycle point cloud color mode (elevation/intensity/classification/rgb)
    - ,/.: Decrease/increase overlay alpha (transparency)
    - O: Place observer (for viewshed) at look-at point
    - Shift+O: Cycle drone mode (off → 3rd person → FPV → off)
    - V: Toggle viewshed overlay (teal glow shows visible terrain)
    - [/]: Decrease/increase observer height
    - R: Decrease terrain resolution (coarser, up to 8x subsample)
    - Shift+R: Increase terrain resolution (finer, down to 1x)
    - Z: Decrease vertical exaggeration
    - Shift+Z: Increase vertical exaggeration
    - Y: Cycle color stretch (linear, sqrt, cbrt, log)
    - T: Toggle shadows
    - 0: Toggle ambient occlusion (progressive)
    - Shift+G: Cycle GI bounces (1→2→3→1)
    - Shift+D: Toggle OptiX AI Denoiser
    - C: Cycle colormap
    - Shift+F: Fetch/toggle FIRMS fire layer (7d LANDSAT 30m)
    - Shift+W: Toggle wind particle animation
    - Shift+Y: Toggle hydro flow particle animation
    - Shift+B: Toggle GTFS-RT realtime vehicle overlay
    - F: Save screenshot
    - M: Toggle minimap overlay
    - H: Toggle help overlay
    - X: Exit

    Examples
    --------
    >>> import rtxpy
    >>> dem = xr.open_dataarray('terrain.tif')
    >>> dem = dem.copy(data=cupy.asarray(dem.data))
    >>> rtxpy.explore(dem)

    >>> # Or via accessor
    >>> dem.rtx.explore()
    """
    from .core import InteractiveViewer
    from .mesh_chunk_manager import _MeshChunkManager

    # Auto-detect Jupyter and use widget-based viewer
    from ..notebook import _detect_jupyter
    if _detect_jupyter():
        from ..notebook import JupyterViewer
        ViewerClass = JupyterViewer
    else:
        ViewerClass = InteractiveViewer

    # Pre-set terrain_source and scene_zarr on the class so
    # _enable_terrain_lod() (called during __init__) can see them.
    # These are consumed during LOD setup, before post-init runs.
    ViewerClass._pre_terrain_source = terrain_source
    ViewerClass._pre_scene_zarr = scene_zarr
    ViewerClass._pre_tile_data_fn = tile_data_fn

    viewer = ViewerClass(
        raster,
        width=width,
        height=height,
        render_scale=render_scale,
        key_repeat_interval=key_repeat_interval,
        rtx=rtx,
        pixel_spacing_x=pixel_spacing_x,
        pixel_spacing_y=pixel_spacing_y,
        overlay_layers=overlay_layers,
        title=title,
        subtitle=subtitle,
        legend=legend,
        subsample=subsample,
        skirt=skirt,
    )

    # Clean up class-level pre-sets
    ViewerClass._pre_terrain_source = None
    ViewerClass._pre_scene_zarr = None
    ViewerClass._pre_tile_data_fn = None

    viewer._geometry_colors_builder = geometry_colors_builder
    viewer._baked_meshes = baked_meshes or {}
    viewer._minimap_style = minimap_style
    viewer._minimap_layer = minimap_layer
    viewer._minimap_colors = minimap_colors
    viewer._info_text = info_text
    viewer._accessor = accessor
    viewer._terrain_loader = terrain_loader
    viewer._tile_data_fn = tile_data_fn
    viewer._terrain_source = terrain_source
    viewer._scene_zarr = scene_zarr
    # Only create legacy _MeshChunkManager when the new chunk-source
    # path is NOT active.  When terrain_source is provided, the LOD
    # manager's SceneMeshManager handles placed geometry instead.
    if scene_zarr is not None and terrain_source is None:
        viewer._chunk_manager = _MeshChunkManager(
            scene_zarr, pixel_spacing_x, pixel_spacing_y)
    viewer.color_stretch = color_stretch
    if color_stretch in viewer._color_stretches:
        viewer._color_stretch_idx = viewer._color_stretches.index(color_stretch)
    if tile_service is not None:
        viewer._tile_service = tile_service
        # Sync basemap index to match the active tile service (but start OFF)
        pname = tile_service.provider_name
        if pname in viewer._basemap_options:
            viewer._basemap_idx = viewer._basemap_options.index(pname)
        else:
            viewer._basemap_idx = 0  # 'none'

    # Wind data initialization
    if wind_data is not None:
        viewer._init_wind(wind_data)

    # Weather overlay initialization
    if weather_data is not None:
        viewer._init_weather(weather_data)

    # Hydro flow initialization
    if hydro_data is not None:
        if hydro_data is True or 'flow_accum' not in hydro_data:
            # Lazy mode: compute MFD hydro from terrain on first enable
            viewer._hydro_lazy = True
            hydro_start_enabled = (
                hydro_data.get('enabled', False)
                if isinstance(hydro_data, dict) else False)
            if hydro_start_enabled:
                viewer._compute_hydro_from_terrain()
                viewer._hydro_enabled = True
                if 'stream_link' in viewer._overlay_layers:
                    viewer._overlay_as_water = True
        else:
            # Pre-computed hydro data provided
            hydro_start_enabled = hydro_data.get('enabled', True)
            flow_accum = hydro_data['flow_accum']
            hydro_opts = {k: v for k, v in hydro_data.items()
                          if k not in ('flow_accum', 'enabled')}
            viewer._init_hydro(flow_accum, **hydro_opts)
            # Re-register stream_link overlay with NaN + palette coloring
            if (viewer._hydro_stream_order_raw is not None
                    and 'stream_link' in viewer._overlay_layers):
                max_order = int(viewer._hydro_stream_order_raw.max())
                palette_lut = InteractiveViewer._build_stream_palette_lut(
                    max_order)
                sl_data = viewer._base_overlay_layers['stream_link']
                if hasattr(sl_data, 'get'):
                    sl_data = sl_data.get()
                sl_data = np.asarray(sl_data, dtype=np.float32)
                so_raw = viewer._hydro_stream_order_raw.astype(np.float32)
                sl_color = np.where(
                    (sl_data <= 0) | (so_raw <= 0),
                    np.float32(np.nan), so_raw)
                _add_overlay(viewer, 'stream_link', sl_color,
                             color_lut=palette_lut)
            if hydro_start_enabled:
                viewer._hydro_enabled = True
                if 'stream_link' in viewer._overlay_layers:
                    viewer._overlay_as_water = True
            else:
                viewer._hydro_enabled = False
                viewer._terrain_layer_idx = 0
                viewer._active_overlay_data = None
                viewer._overlay_as_water = False
                viewer._active_overlay_color_lut = None

    # GTFS-RT initialization
    if gtfs_data is not None:
        rt_url = (gtfs_data.get('metadata') or {}).get('realtime_url')
        if rt_url:
            # Build route_id -> (r,g,b) colour map from route features
            rc_map = {}
            for f in gtfs_data.get('routes', {}).get('features', []):
                props = f.get('properties') or {}
                rc = (props.get('route_color') or '').strip().lstrip('#')
                rname = props.get('route_short_name', '')
                if len(rc) == 6:
                    try:
                        rgb = (int(rc[0:2], 16) / 255.0,
                               int(rc[2:4], 16) / 255.0,
                               int(rc[4:6], 16) / 255.0)
                        # Key by route_short_name since GTFS-RT uses route_id
                        # which may differ; store both for best chance of matching
                        if rname:
                            rc_map[rname] = rgb
                        rid = props.get('route_id', '')
                        if rid:
                            rc_map[rid] = rgb
                    except ValueError:
                        pass
            viewer._init_gtfs_rt(rt_url, route_colors=rc_map)

    # Ambient occlusion initialization
    if ao_samples > 0:
        viewer.ao_enabled = True
    else:
        viewer.ao_enabled = False
    viewer.gi_bounces = gi_bounces

    # Denoiser initialization
    if denoise:
        viewer.denoise_enabled = True

    # Shared render params (fog, lighting, colormap)
    if fog_density > 0:
        viewer.fog_density = fog_density
    if fog_color != (0.7, 0.8, 0.9):
        viewer.fog_color = fog_color
    if colormap is not None:
        viewer.colormap = colormap
        if colormap in viewer.colormaps:
            viewer.colormap_idx = viewer.colormaps.index(colormap)
    if sun_azimuth is not None:
        viewer.sun_azimuth = sun_azimuth
    if sun_altitude is not None:
        viewer.sun_altitude = sun_altitude
    if shadows is not None:
        viewer.shadows = shadows
    if ambient is not None:
        viewer.ambient = ambient

    # Initial state: everything off except elevation
    viewer._tiles_enabled = False
    viewer._basemap_idx = 0  # 'none'
    viewer._geometry_layer_idx = 0  # 'none'
    if rtx is not None:
        for geom_id in viewer._all_geometries:
            if geom_id != 'terrain':
                rtx.set_geometry_visible(geom_id, False)
    if tour is not None:
        repl = True
    viewer._repl = repl
    viewer._tour = tour
    return viewer.run(start_position=start_position, look_at=look_at)
