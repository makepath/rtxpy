"""xarray accessor for rtxpy GPU-accelerated terrain analysis."""

import numpy as np
import xarray as xr
from .rtx import RTX, has_cupy


@xr.register_dataarray_accessor("rtx")
class RTXAccessor:
    """xarray DataArray accessor for rtxpy GPU-accelerated terrain analysis.

    This accessor provides convenient access to rtxpy analysis functions
    directly on xarray DataArrays.

    Examples
    --------
    >>> import rtxpy
    >>> import xarray as xr
    >>> import cupy
    >>> dem = xr.open_dataarray('dem.tif')
    >>> dem = dem.copy(data=cupy.asarray(dem.data))
    >>> # Use accessor methods
    >>> hs = dem.rtx.hillshade(shadows=True)
    >>> vs = dem.rtx.viewshed(x=500000, y=4500000)
    """

    def __init__(self, xarray_obj):
        self._obj = xarray_obj
        self._rtx_instance = None
        # Track pixel spacing for coordinate conversion (set by triangulate/place_mesh)
        self._pixel_spacing_x = 1.0
        self._pixel_spacing_y = 1.0
        # Per-geometry solid color overrides: {geometry_id: (r, g, b)}
        self._geometry_colors = {}
        self._geometry_colors_dirty = True
        self._geometry_colors_gpu = None
        # Baked merged meshes for VE rescaling: {geometry_id: (vertices, indices)}
        self._baked_meshes = {}

    @property
    def _rtx(self):
        """Lazily create and cache an RTX instance."""
        if self._rtx_instance is None:
            self._rtx_instance = RTX()
        return self._rtx_instance

    def to_cupy(self):
        """Convert DataArray data to cupy array on GPU.

        Returns a new DataArray with data on the GPU. If the data is already
        a cupy array, returns the original DataArray unchanged.

        Returns
        -------
        xarray.DataArray
            DataArray with cupy array data.

        Raises
        ------
        ImportError
            If cupy is not available.

        Examples
        --------
        >>> dem_gpu = dem.rtx.to_cupy()
        >>> hs = dem_gpu.rtx.hillshade()
        """
        if not has_cupy:
            raise ImportError(
                "cupy is required for GPU operations. "
                "Install with: conda install -c conda-forge cupy"
            )
        import cupy
        if isinstance(self._obj.data, cupy.ndarray):
            return self._obj
        return self._obj.copy(data=cupy.asarray(self._obj.data))

    def viewshed(self, x, y, observer_elev=0, target_elev=0, rtx=None):
        """Compute viewshed from observer point.

        Determines which areas of the terrain are visible from a given
        observer location using GPU-accelerated ray tracing.

        Parameters
        ----------
        x : int or float
            X coordinate of the observer location (in raster coordinate units).
        y : int or float
            Y coordinate of the observer location (in raster coordinate units).
        observer_elev : float, optional
            Height offset above the terrain surface for the observer. Default is 0.
        target_elev : float, optional
            Height offset above the terrain surface for target points. Default is 0.
        rtx : RTX, optional
            Existing RTX instance to reuse. If None, uses the accessor's
            cached RTX instance.

        Returns
        -------
        xarray.DataArray
            Visibility raster with the same coordinates as the input.
            Values indicate the viewing angle in degrees for visible cells,
            or -1 for cells not visible from the observer.

        Raises
        ------
        ValueError
            If x or y coordinates are outside the raster extent.
            If raster.data is not a cupy array.

        Examples
        --------
        >>> vis = dem.rtx.viewshed(x=500000, y=4500000, observer_elev=2)
        """
        from .analysis import viewshed as _viewshed
        return _viewshed(
            self._obj, x=x, y=y,
            observer_elev=observer_elev,
            target_elev=target_elev,
            rtx=rtx
        )

    def hillshade(self, shadows=False, azimuth=225, angle_altitude=25,
                  name='hillshade', rtx=None):
        """Compute hillshade illumination.

        Creates a shaded relief effect by simulating how the terrain
        would look when illuminated by the sun from a given direction.

        Parameters
        ----------
        shadows : bool, optional
            If True, cast shadow rays to determine which areas are in shadow.
            Shadows are rendered at half brightness. Default is False.
        azimuth : int, optional
            Sun azimuth angle in degrees, measured clockwise from north.
            Default is 225 (southwest).
        angle_altitude : int, optional
            Sun altitude angle in degrees above the horizon. Default is 25.
        name : str, optional
            Name attribute for the output DataArray. Default is 'hillshade'.
        rtx : RTX, optional
            Existing RTX instance to reuse. If None, uses the accessor's
            cached RTX instance.

        Returns
        -------
        xarray.DataArray
            Hillshade raster with values from 0 (dark) to 1 (bright).
            Edge pixels are set to NaN.

        Examples
        --------
        >>> shade = dem.rtx.hillshade(shadows=True, azimuth=90)
        """
        from .analysis import hillshade as _hillshade
        return _hillshade(
            self._obj, shadows=shadows, azimuth=azimuth,
            angle_altitude=angle_altitude, name=name,
            rtx=rtx
        )

    def slope(self, rtx=None):
        """Compute terrain slope in degrees.

        Parameters
        ----------
        rtx : RTX, optional
            Existing RTX instance to reuse. If None, uses the accessor's
            cached RTX instance.

        Returns
        -------
        xarray.DataArray
            Slope in degrees (0° flat, 90° vertical). Edge pixels are NaN.

        Examples
        --------
        >>> s = dem.rtx.slope()
        """
        from .analysis import slope as _slope
        return _slope(self._obj, rtx=rtx)

    def aspect(self, rtx=None):
        """Compute terrain aspect (compass bearing of downhill direction).

        Convention: 0° = North, 90° = East, 180° = South, 270° = West.
        -1 = flat terrain.

        Parameters
        ----------
        rtx : RTX, optional
            Existing RTX instance to reuse. If None, uses the accessor's
            cached RTX instance.

        Returns
        -------
        xarray.DataArray
            Aspect in degrees [0, 360) or -1 for flat. Edge pixels are NaN.

        Examples
        --------
        >>> a = dem.rtx.aspect()
        """
        from .analysis import aspect as _aspect
        return _aspect(self._obj, rtx=rtx)

    def clear(self):
        """Remove all geometries and reset to single-GAS mode.

        After calling this, the RTX instance can use either build() for
        single-GAS mode or add_geometry() for multi-GAS mode.

        Examples
        --------
        >>> dem.rtx.clear()
        """
        self._rtx.clear_scene()

    def add_geometry(self, geometry_id, vertices, indices, transform=None):
        """Add a geometry to the scene with an optional transform.

        This enables multi-GAS mode, allowing multiple geometries to be
        traced together. Adding a geometry with an existing ID replaces it.

        Parameters
        ----------
        geometry_id : str
            Unique identifier for this geometry.
        vertices : array-like
            Vertex buffer (flattened float32 array, 3 floats per vertex).
        indices : array-like
            Index buffer (flattened int32 array, 3 ints per triangle).
        transform : list of float, optional
            12-float list representing a 3x4 row-major affine transform matrix.
            Defaults to identity. Format: [Xx, Xy, Xz, Tx, Yx, Yy, Yz, Ty, Zx, Zy, Zz, Tz]

        Returns
        -------
        int
            0 on success, non-zero on error.

        Examples
        --------
        >>> dem.rtx.add_geometry('terrain', vertices, indices)
        >>> dem.rtx.add_geometry('building', bldg_verts, bldg_idx, transform=[...])
        """
        return self._rtx.add_geometry(geometry_id, vertices, indices, transform)

    def remove_geometry(self, geometry_id):
        """Remove a geometry from the scene.

        Parameters
        ----------
        geometry_id : str
            The ID of the geometry to remove.

        Returns
        -------
        int
            0 on success, -1 if geometry not found.

        Examples
        --------
        >>> dem.rtx.remove_geometry('building')
        """
        return self._rtx.remove_geometry(geometry_id)

    def list_geometries(self):
        """Get a list of all geometry IDs in the scene.

        Returns
        -------
        list of str
            List of geometry ID strings.

        Examples
        --------
        >>> dem.rtx.list_geometries()
        ['terrain', 'building1', 'building2']
        """
        return self._rtx.list_geometries()

    def get_geometry_count(self):
        """Get the number of geometries in the scene.

        Returns
        -------
        int
            Number of geometries (0 in single-GAS mode).

        Examples
        --------
        >>> dem.rtx.get_geometry_count()
        3
        """
        return self._rtx.get_geometry_count()

    def has_geometry(self, geometry_id):
        """Check if a geometry with the given ID exists.

        Parameters
        ----------
        geometry_id : str
            The ID of the geometry to check.

        Returns
        -------
        bool
            True if the geometry exists, False otherwise.

        Examples
        --------
        >>> dem.rtx.has_geometry('terrain')
        True
        """
        return self._rtx.has_geometry(geometry_id)

    def set_geometry_color(self, geometry_id, color):
        """Set a solid color override for a geometry.

        When rendering, this geometry will be drawn with the given flat color
        instead of the elevation colormap.

        Parameters
        ----------
        geometry_id : str
            The ID of the geometry to color.
        color : tuple of float
            RGB color as (r, g, b) with values [0-1].

        Examples
        --------
        >>> dem.rtx.set_geometry_color('building_0', (0.9, 0.2, 0.2))
        """
        self._geometry_colors[geometry_id] = tuple(color[:3])
        self._geometry_colors_dirty = True

    def _build_geometry_colors_gpu(self):
        """Build a GPU array mapping instance IDs to solid colors.

        Returns
        -------
        cupy.ndarray or None
            Array of shape (num_geometries, 4) with [r, g, b, active] per
            instance, or None if no geometry colors are set.
        """
        if not self._geometry_colors:
            return None

        if not self._geometry_colors_dirty and self._geometry_colors_gpu is not None:
            return self._geometry_colors_gpu

        import cupy
        geom_ids = self._rtx.list_geometries()
        if not geom_ids:
            return None

        n = len(geom_ids)
        colors = np.zeros((n, 4), dtype=np.float32)

        for i, gid in enumerate(geom_ids):
            if gid in self._geometry_colors:
                c = self._geometry_colors[gid]
                if len(c) == 4:
                    colors[i] = [c[0], c[1], c[2], c[3]]
                else:
                    colors[i] = [c[0], c[1], c[2], 1.0]

        self._geometry_colors_gpu = cupy.asarray(colors)
        self._geometry_colors_dirty = False
        return self._geometry_colors_gpu

    def trace(self, rays, hits, num_rays, primitive_ids=None, instance_ids=None):
        """Trace rays against the current acceleration structure.

        Works with both single-GAS mode (after build()) and multi-GAS mode
        (after add_geometry()).

        Parameters
        ----------
        rays : array-like
            Ray buffer (8 float32 per ray: ox, oy, oz, tmin, dx, dy, dz, tmax).
        hits : array-like
            Hit buffer (4 float32 per hit: t, nx, ny, nz). t=-1 indicates a miss.
        num_rays : int
            Number of rays to trace.
        primitive_ids : array-like, optional
            Output buffer (num_rays x int32) for triangle indices.
            Will contain the index of the hit triangle within its geometry,
            or -1 for rays that missed.
        instance_ids : array-like, optional
            Output buffer (num_rays x int32) for geometry/instance indices.
            Will contain the instance ID of the hit geometry, or -1 for misses.
            Useful in multi-GAS mode to identify which geometry was hit.

        Returns
        -------
        int
            0 on success, non-zero on error.

        Examples
        --------
        >>> rays = np.array([ox, oy, oz, 0.0, dx, dy, dz, 1000.0], dtype=np.float32)
        >>> hits = np.zeros(4, dtype=np.float32)
        >>> dem.rtx.trace(rays, hits, 1)
        0
        """
        return self._rtx.trace(rays, hits, num_rays, primitive_ids, instance_ids)

    def render(self, camera_position, look_at, fov=60.0, up=(0, 0, 1),
               width=1920, height=1080, sun_azimuth=225, sun_altitude=45,
               shadows=True, ambient=0.15, fog_density=0.0,
               fog_color=(0.7, 0.8, 0.9), colormap='terrain',
               color_range=None, output_path=None, alpha=False,
               vertical_exaggeration=None, rtx=None):
        """Render terrain with a perspective camera for movie-quality visualization.

        Uses OptiX ray tracing to render terrain with realistic lighting, shadows,
        atmospheric effects, and elevation-based coloring.

        Parameters
        ----------
        camera_position : tuple of float
            Camera position in world coordinates (x, y, z). x and y are in pixel
            coordinates (0 to width-1, 0 to height-1). z is in the same units as
            elevation data (typically meters).
        look_at : tuple of float
            Target point the camera looks at (x, y, z).
        fov : float, optional
            Vertical field of view in degrees. Default is 60.
        up : tuple of float, optional
            World up vector. Default is (0, 0, 1).
        width : int, optional
            Output image width in pixels. Default is 1920.
        height : int, optional
            Output image height in pixels. Default is 1080.
        sun_azimuth : float, optional
            Sun azimuth angle in degrees, measured clockwise from north.
            Default is 225 (southwest).
        sun_altitude : float, optional
            Sun altitude angle in degrees above the horizon. Default is 45.
        shadows : bool, optional
            If True, cast shadow rays for realistic shadows. Default is True.
        ambient : float, optional
            Ambient light intensity [0-1]. Default is 0.15.
        fog_density : float, optional
            Exponential fog density. 0 disables fog. Default is 0.
        fog_color : tuple of float, optional
            Fog color as (r, g, b) values [0-1]. Default is (0.7, 0.8, 0.9).
        colormap : str, optional
            Matplotlib colormap name or 'hillshade' for grayscale shading.
            Default is 'terrain'.
        color_range : tuple of float, optional
            Elevation range (min, max) for colormap. If None, uses data range.
        output_path : str, optional
            If provided, saves the rendered image to this path (PNG, TIFF, etc.).
        alpha : bool, optional
            If True, output has 4 channels (RGBA) with alpha=0 for sky.
            Default is False.
        vertical_exaggeration : float, optional
            Scale factor for elevation values. Values < 1 reduce vertical
            exaggeration (useful when elevation units don't match pixel units).
            If None, auto-computes a value to make relief proportional to
            terrain extent. Use 1.0 for no scaling.
        rtx : RTX, optional
            Existing RTX instance to reuse. If None, uses the accessor's
            cached RTX instance.

        Returns
        -------
        numpy.ndarray
            Rendered image of shape (height, width, 3) or (height, width, 4)
            as float32 with values [0-1].

        Examples
        --------
        >>> img = dem.rtx.render(
        ...     camera_position=(W/2, -50, elev_max + 200),
        ...     look_at=(W/2, H/2, elev_mean),
        ...     shadows=True,
        ...     output_path='terrain_render.png'
        ... )
        """
        from .analysis import render as _render
        return _render(
            self._obj,
            camera_position=camera_position,
            look_at=look_at,
            fov=fov,
            up=up,
            width=width,
            height=height,
            sun_azimuth=sun_azimuth,
            sun_altitude=sun_altitude,
            shadows=shadows,
            ambient=ambient,
            fog_density=fog_density,
            fog_color=fog_color,
            colormap=colormap,
            color_range=color_range,
            output_path=output_path,
            alpha=alpha,
            vertical_exaggeration=vertical_exaggeration,
            rtx=rtx,  # User can override, but default None creates fresh instance
        )

    def place_mesh(self, mesh_source, positions, geometry_id=None, scale=1.0,
                   rotation_z=0.0, swap_yz=False, center_xy=True, base_at_zero=True,
                   pixel_spacing_x=1.0, pixel_spacing_y=1.0):
        """Load a mesh and place instances on the terrain at specified positions.

        This is a convenience method that combines load_mesh() and
        make_transforms_on_terrain() to place 3D models on the terrain surface.

        Parameters
        ----------
        mesh_source : str, Path, or callable
            Either a path to a mesh file (GLB, OBJ, STL, etc.) or a callable
            that returns ``(vertices, indices)`` as flat numpy arrays
            (float32 vertices, int32 indices). When a callable is provided,
            ``scale``, ``swap_yz``, ``center_xy``, and ``base_at_zero`` are
            ignored since the callable is responsible for producing
            ready-to-use mesh data.
        positions : list of tuple, or callable
            List of (x, y) positions in pixel coordinates where instances
            should be placed, or a callable that returns such a list.
            The callable receives the terrain as a 2D numpy array and
            should return a list of ``(x, y)`` pixel-coordinate tuples.
            The Z coordinate is automatically sampled from the terrain.
        geometry_id : str, optional
            Base ID for the geometries. If placing multiple instances,
            they will be named "{geometry_id}_{i}". If None, uses the
            filename stem (filepath) or the callable's ``__name__``.
        scale : float, optional
            Scale factor for the mesh. Only used when ``mesh_source`` is a
            filepath. Default is 1.0.
        rotation_z : float or 'random', optional
            Rotation around Z axis in radians, or 'random' for random
            rotations per instance. Default is 0.0.
        swap_yz : bool, optional
            If True, swap Y and Z coordinates (for Y-up models). Only used
            when ``mesh_source`` is a filepath. Default is False.
        center_xy : bool, optional
            If True, center the mesh at the XY origin. Only used when
            ``mesh_source`` is a filepath. Default is True.
        base_at_zero : bool, optional
            If True, place the mesh base at Z=0. Only used when
            ``mesh_source`` is a filepath. Default is True.
        pixel_spacing_x : float, optional
            X spacing between pixels in world units. Default is 1.0 (pixel coords).
        pixel_spacing_y : float, optional
            Y spacing between pixels in world units. Default is 1.0 (pixel coords).

        Returns
        -------
        tuple
            (vertices, indices, transforms) - The loaded mesh data and transforms.

        Examples
        --------
        >>> # Place towers at hilltop positions (pixel coordinates)
        >>> tower_positions = [(100, 50), (200, 150), (300, 100)]
        >>> verts, indices, transforms = dem.rtx.place_mesh(
        ...     'tower.glb',
        ...     tower_positions,
        ...     geometry_id='tower',
        ...     scale=0.1
        ... )

        >>> # Place with callables for both mesh and positions
        >>> def make_tower():
        ...     return load_mesh('tower.glb', scale=2.7, center_xy=True, base_at_zero=True)
        >>> def pick_hilltops(terrain):
        ...     # return [(x, y), ...] pixel coords chosen from terrain
        ...     ...
        >>> verts, indices, transforms = dem.rtx.place_mesh(
        ...     make_tower,
        ...     pick_hilltops,
        ...     pixel_spacing_x=25.0,
        ...     pixel_spacing_y=25.0
        ... )
        """
        from pathlib import Path
        from .mesh import load_mesh, make_transform
        import numpy as np

        if callable(mesh_source):
            vertices, indices = mesh_source()
            if geometry_id is None:
                geometry_id = getattr(mesh_source, '__name__', 'mesh')
        else:
            filepath = Path(mesh_source)
            if geometry_id is None:
                geometry_id = filepath.stem

            # Load the mesh
            vertices, indices = load_mesh(
                filepath,
                scale=scale,
                swap_yz=swap_yz,
                center_xy=center_xy,
                base_at_zero=base_at_zero
            )

        # Get terrain data as numpy array
        terrain_data = self._obj.data
        if hasattr(terrain_data, 'get'):  # cupy array
            terrain_data = terrain_data.get()
        else:
            terrain_data = np.asarray(terrain_data)

        H, W = terrain_data.shape

        # Resolve positions callable
        if callable(positions):
            positions = positions(terrain_data)

        # Create transforms for each position with pixel spacing
        transforms = []
        for i, (px, py) in enumerate(positions):
            # Sample terrain elevation at pixel position
            ix = int(np.clip(px, 0, W - 1))
            iy = int(np.clip(py, 0, H - 1))
            z = float(terrain_data[iy, ix])

            # Convert pixel coords to world coords
            world_x = px * pixel_spacing_x
            world_y = py * pixel_spacing_y

            # Determine rotation
            if rotation_z == 'random':
                rot = np.random.uniform(0, 2 * np.pi)
            else:
                rot = float(rotation_z)

            transform = make_transform(x=world_x, y=world_y, z=z, scale=1.0, rotation_z=rot)
            transforms.append(transform)

        # Add each instance to the scene
        for i, transform in enumerate(transforms):
            instance_id = f"{geometry_id}_{i}" if len(transforms) > 1 else geometry_id
            self._rtx.add_geometry(instance_id, vertices, indices, transform=transform)

        return vertices, indices, transforms

    _GEOJSON_PALETTE = [
        (0.90, 0.22, 0.20),  # red
        (0.20, 0.56, 0.90),  # blue
        (0.95, 0.75, 0.10),  # yellow
        (0.20, 0.78, 0.40),  # green
        (0.75, 0.30, 0.75),  # purple
        (1.00, 0.50, 0.00),  # orange
        (0.00, 0.80, 0.80),  # cyan
        (1.00, 0.40, 0.60),  # pink
    ]

    def place_geojson(self, geojson, height=10.0,
                      label_field=None, height_field=None,
                      fill_mesh=None, fill_spacing=None, fill_scale=1.0,
                      models=None, model_field=None,
                      geometry_id='geojson',
                      densify=True,
                      merge=False,
                      extrude=False,
                      mesh_cache=None,
                      color=None,
                      width=None):
        """Load GeoJSON and place features as 3D meshes on terrain.

        Points become glowing orbs hovering above the terrain (or
        fill_mesh / model instances), LineStrings become flat ribbon
        meshes hugging the terrain surface, and Polygons become
        outline ribbons with optional scattered fill_mesh instances
        inside.

        When ``models`` and ``model_field`` are provided, features whose
        ``model_field`` property matches a key in ``models`` use the
        associated mesh instead.  For Points this replaces the marker cube;
        for Polygons the wall is suppressed and the matched model is
        scattered to fill the area.

        Parameters
        ----------
        geojson : str, Path, or dict
            GeoJSON file path or dict (FeatureCollection, Feature, or Geometry).
        height : float
            Default extrusion height for walls/markers (world units).
        label_field : str, optional
            Property field for grouping/naming geometries.
        height_field : str, optional
            Property field for per-feature extrusion height (overrides height).
        fill_mesh : str, Path, or callable, optional
            For Polygon/MultiPolygon: mesh to scatter inside polygon areas.
        fill_spacing : float, optional
            Spacing between fill instances in pixel units. Required with
            fill_mesh or models.
        fill_scale : float
            Scale factor for fill_mesh and model instances.
        models : dict, optional
            Maps GeoJSON property values to mesh sources. Values can be file
            paths (str/Path) or callables returning ``(vertices, indices)``.
            Example: ``models={'forest': 'tree.glb', 'urban': make_building}``.
        model_field : str, optional
            GeoJSON property name to look up in ``models``.
            Example: ``model_field='landcover'``.
        geometry_id : str
            Base prefix for geometry IDs (default 'geojson').
        densify : bool or float
            Controls vertex densification on LineStrings and Polygon
            outlines. ``True`` (default) densifies at 1-pixel steps so
            geometry closely follows terrain. ``False`` disables
            densification. A float sets the step size in pixels (e.g.
            ``densify=0.5`` for half-pixel resolution, ``densify=5.0``
            for sparser sampling).
        merge : bool
            When ``True``, all non-instanced geometry (LineStrings and
            Polygon outlines) is concatenated into a single GAS instead
            of one GAS per feature.  Much faster for large feature counts
            (e.g. thousands of building footprints) at the cost of
            per-feature visibility control.  Default ``False``.
        extrude : bool
            When ``True``, Polygons are extruded into solid 3D geometry
            (vertical walls + roof cap) instead of tube outlines.
            Height comes from ``height_field`` per feature or falls back
            to the ``height`` parameter.  Ideal for building footprints.
            Default ``False``.
        mesh_cache : str or Path, optional
            Path to a ``.npz`` file for caching the merged mesh.
            Requires ``merge=True``.  On first run the merged
            vertex/index arrays are saved; on subsequent runs they are
            loaded directly, skipping GeoJSON parsing, coordinate
            projection, and triangulation entirely.
        color : tuple, optional
            Solid ``(r, g, b)`` colour override for all features.
            When set, bypasses the cycling palette.
        width : float, optional
            Ribbon half-width when ``height <= 0``.  Defaults to
            ``pixel_spacing * 0.5`` (roughly one pixel wide).

        Returns
        -------
        dict
            ``{'features': int, 'geometries': int, 'geometry_ids': list}``
        """
        import warnings
        from pathlib import Path as _Path
        from .mesh import load_mesh, make_transform
        from .geojson import (
            _load_geojson, _flatten_multi, _sanitize_label,
            _geojson_to_world_coords, _build_transformer,
            _make_marker_cube, _make_marker_orb, _densify_on_terrain,
            _linestring_to_tube_mesh, _linestring_to_ribbon_mesh,
            _polygon_to_tube_mesh, _polygon_to_ribbon_mesh,
            _extrude_polygon, _scatter_in_polygon,
        )

        # Flat ribbon for LineStrings and Polygon outlines
        use_ribbon = not extrude
        if width is None:
            width = (self._pixel_spacing_x + self._pixel_spacing_y) * 0.08
        # Small hover above terrain; bilinear Z sampling keeps ribbons close
        ribbon_hover = (self._pixel_spacing_x + self._pixel_spacing_y) * 0.01

        # Fast path: load pre-computed merged mesh from cache
        if mesh_cache is not None and merge:
            cache_p = _Path(mesh_cache)
            if cache_p.exists():
                print(f"Loading cached mesh: {cache_p.name}")
                npz = np.load(cache_p)
                merged_v = npz['vertices']
                merged_idx = npz['indices']
                self._rtx.add_geometry(geometry_id, merged_v, merged_idx)
                if color is not None:
                    cache_color = color
                elif extrude:
                    cache_color = (0.6, 0.6, 0.6)
                else:
                    cache_color = self._GEOJSON_PALETTE[0]
                self._geometry_colors[geometry_id] = cache_color
                self._geometry_colors_dirty = True
                # Compute per-vertex terrain Z for re-snapping on resolution change
                terrain_data_np = self._obj.data
                if hasattr(terrain_data_np, 'get'):
                    terrain_data_np = terrain_data_np.get()
                else:
                    terrain_data_np = np.asarray(terrain_data_np)
                _H, _W = terrain_data_np.shape
                _psx = self._pixel_spacing_x
                _psy = self._pixel_spacing_y
                _vx = merged_v[0::3]
                _vy = merged_v[1::3]
                _ix = np.clip(np.round(_vx / _psx).astype(int), 0, _W - 1)
                _iy = np.clip(np.round(_vy / _psy).astype(int), 0, _H - 1)
                _base_z = terrain_data_np[_iy, _ix].astype(np.float32)
                _base_z = np.where(np.isnan(_base_z), 0.0, _base_z)
                self._baked_meshes[geometry_id] = (merged_v.copy(),
                                                   merged_idx.copy(),
                                                   _base_z)
                return {
                    'features': -1,
                    'geometries': 1,
                    'geometry_ids': [geometry_id],
                }

        # Resolve pixel spacing
        psx = self._pixel_spacing_x
        psy = self._pixel_spacing_y

        if psx == 1.0 and psy == 1.0:
            warnings.warn(
                "place_geojson called before triangulate()/voxelate() — "
                "pixel spacing defaults to 1.0, which may produce wrong "
                "world coordinates for CRS-referenced rasters.",
                stacklevel=2,
            )

        # Get terrain data as numpy
        terrain_data = self._obj.data
        if hasattr(terrain_data, 'get'):  # cupy
            terrain_data = terrain_data.get()
        else:
            terrain_data = np.asarray(terrain_data)
        H, W = terrain_data.shape

        # Build CRS transformer once
        transformer = _build_transformer(self._obj)

        # Parse GeoJSON
        features = _load_geojson(geojson)
        if not features:
            return {'features': 0, 'geometries': 0, 'geometry_ids': []}

        # Load fill mesh if provided
        fill_verts = fill_indices = None
        if fill_mesh is not None:
            if fill_spacing is None:
                raise ValueError("fill_spacing is required when fill_mesh is set")
            if callable(fill_mesh):
                fill_verts, fill_indices = fill_mesh()
            else:
                fill_verts, fill_indices = load_mesh(
                    _Path(fill_mesh), scale=fill_scale,
                    swap_yz=True, center_xy=True, base_at_zero=True,
                )

        # Load per-feature models
        model_cache = {}
        if models is not None:
            if model_field is None:
                raise ValueError("model_field is required when models is set")
            if fill_spacing is None:
                raise ValueError("fill_spacing is required when models is set")
            for key, msrc in models.items():
                if callable(msrc):
                    model_cache[key] = msrc()
                else:
                    model_cache[key] = load_mesh(
                        _Path(msrc), scale=fill_scale,
                        swap_yz=True, center_xy=True, base_at_zero=True,
                    )

        geometry_ids = []
        geom_counter = 0
        palette = self._GEOJSON_PALETTE
        oob_counter = [0]  # mutable counter for out-of-bounds coords

        # Merge accumulator: collect verts/indices and flush once at the end
        _merge_verts = []
        _merge_indices = []
        _merge_vert_offset = 0
        _merge_color = None

        for feat_i, (geom, props) in enumerate(features):
            if geom is None or not geom.get("coordinates"):
                continue

            # Per-feature color from cycling palette
            feat_color = palette[feat_i % len(palette)]

            # Per-feature height
            feat_height = height
            if height_field and height_field in props:
                try:
                    feat_height = float(props[height_field])
                except (ValueError, TypeError):
                    pass

            # Label for geometry ID
            label = None
            if label_field and label_field in props:
                label = _sanitize_label(props[label_field])

            # Per-feature model lookup
            feat_model_verts = feat_model_indices = None
            if model_cache and model_field and model_field in props:
                prop_val = props[model_field]
                if prop_val in model_cache:
                    feat_model_verts, feat_model_indices = model_cache[prop_val]

            # Flatten Multi* types
            primitives = _flatten_multi(geom)

            for prim in primitives:
                ptype = prim.get("type", "")
                coords = prim.get("coordinates")
                if coords is None:
                    continue

                # Build geometry ID
                if label:
                    gid = f"{geometry_id}_{label}_{geom_counter}"
                else:
                    gid = f"{geometry_id}_{geom_counter}"

                if ptype == "Point":
                    # Single point — place per-feature model, fill_mesh, or marker
                    wc = _geojson_to_world_coords(
                        [coords], self._obj, terrain_data, psx, psy,
                        transformer=transformer, oob_counter=oob_counter,
                    )
                    if len(wc) == 0:
                        continue
                    pt = wc[0]

                    if feat_model_verts is not None:
                        v, idx = feat_model_verts, feat_model_indices
                        sc = fill_scale
                        hover = 0.0
                        pt_color = feat_color
                    elif fill_verts is not None:
                        v, idx = fill_verts, fill_indices
                        sc = fill_scale
                        hover = 0.0
                        pt_color = feat_color
                    else:
                        orb_radius = feat_height * 0.12
                        v, idx = _make_marker_orb(radius=orb_radius)
                        sc = 1.0
                        hover = feat_height * 0.5
                        # Emissive glow: alpha > 1.0 signals min lighting
                        pt_color = (feat_color[0], feat_color[1],
                                    feat_color[2], 1.6)

                    t = make_transform(
                        x=float(pt[0]), y=float(pt[1]),
                        z=float(pt[2]) + hover,
                        scale=sc,
                    )
                    self._rtx.add_geometry(gid, v, idx, transform=t)
                    geometry_ids.append(gid)
                    self._geometry_colors[gid] = color if color is not None else pt_color
                    geom_counter += 1

                elif ptype == "LineString":
                    wc = _geojson_to_world_coords(
                        coords, self._obj, terrain_data, psx, psy,
                        transformer=transformer, oob_counter=oob_counter,
                    )
                    if len(wc) < 2:
                        continue
                    if densify is not False:
                        step = 1.0 if densify is True else float(densify)
                        wc = _densify_on_terrain(
                            wc, terrain_data, psx, psy, step=step)
                    if use_ribbon:
                        v, idx = _linestring_to_ribbon_mesh(
                            wc, width=width, hover=ribbon_hover,
                        )
                    else:
                        v, idx = _linestring_to_tube_mesh(
                            wc, radius=feat_height * 0.1,
                            hover=feat_height * 0.15,
                        )
                    if len(v) == 0:
                        continue
                    use_color = color if color is not None else feat_color
                    if merge:
                        _merge_indices.append(idx + _merge_vert_offset)
                        _merge_verts.append(v)
                        _merge_vert_offset += len(v) // 3
                        if _merge_color is None:
                            _merge_color = use_color
                    else:
                        self._rtx.add_geometry(gid, v, idx)
                        geometry_ids.append(gid)
                        self._geometry_colors[gid] = use_color
                    geom_counter += 1

                elif ptype == "Polygon":
                    # coords is [exterior_ring, *hole_rings]
                    # Determine scatter mesh: per-feature model > fill_mesh > none
                    scatter_verts = scatter_indices = None
                    if feat_model_verts is not None:
                        scatter_verts, scatter_indices = feat_model_verts, feat_model_indices
                    elif fill_verts is not None:
                        scatter_verts, scatter_indices = fill_verts, fill_indices

                    rings_world = []
                    ring_pixel_coords = None  # for fill/scatter
                    for ri, ring in enumerate(coords):
                        need_pixels = (ri == 0 and scatter_verts is not None)
                        result = _geojson_to_world_coords(
                            ring, self._obj, terrain_data, psx, psy,
                            transformer=transformer,
                            return_pixel_coords=need_pixels,
                            oob_counter=oob_counter,
                        )
                        if need_pixels:
                            wc, ring_pixel_coords = result
                        else:
                            wc = result
                        rings_world.append(wc)

                    # Build polygon mesh (only when no per-feature model)
                    if feat_model_verts is None:
                        if extrude:
                            v, idx = _extrude_polygon(
                                rings_world, feat_height)
                        else:
                            if densify is not False:
                                step = 1.0 if densify is True else float(densify)
                                dense_rings = [
                                    _densify_on_terrain(
                                        r, terrain_data, psx, psy, step=step)
                                    for r in rings_world
                                ]
                            else:
                                dense_rings = rings_world
                            if use_ribbon:
                                v, idx = _polygon_to_ribbon_mesh(
                                    dense_rings, width=width,
                                    hover=ribbon_hover,
                                )
                            else:
                                v, idx = _polygon_to_tube_mesh(
                                    dense_rings, radius=feat_height * 0.1,
                                    hover=feat_height * 0.15,
                                )
                        if len(v) > 0:
                            use_color = color if color is not None else feat_color
                            if merge:
                                _merge_indices.append(idx + _merge_vert_offset)
                                _merge_verts.append(v)
                                _merge_vert_offset += len(v) // 3
                                if _merge_color is None:
                                    _merge_color = use_color
                            else:
                                self._rtx.add_geometry(gid, v, idx)
                                geometry_ids.append(gid)
                                self._geometry_colors[gid] = use_color
                            geom_counter += 1

                    # Scatter mesh inside polygon
                    if scatter_verts is not None and ring_pixel_coords is not None:
                        positions = _scatter_in_polygon(
                            ring_pixel_coords, fill_spacing, H, W
                        )
                        for fi, (px, py) in enumerate(positions):
                            ix = int(np.clip(px, 0, W - 1))
                            iy = int(np.clip(py, 0, H - 1))
                            z = float(terrain_data[iy, ix])
                            if np.isnan(z):
                                z = 0.0
                            wx = px * psx
                            wy = py * psy
                            fill_gid = f"{gid}_fill_{fi}"
                            t = make_transform(
                                x=wx, y=wy, z=z, scale=fill_scale,
                            )
                            self._rtx.add_geometry(
                                fill_gid, scatter_verts, scatter_indices, transform=t
                            )
                            geometry_ids.append(fill_gid)
                            self._geometry_colors[fill_gid] = feat_color
                            geom_counter += 1

        # Flush merged geometry as a single GAS
        if merge and _merge_verts:
            merged_v = np.concatenate(_merge_verts)
            merged_idx = np.concatenate(_merge_indices).astype(np.int32)
            self._rtx.add_geometry(geometry_id, merged_v, merged_idx)
            geometry_ids.append(geometry_id)
            # Color: explicit > merge accumulator > defaults
            if color is not None:
                self._geometry_colors[geometry_id] = color
            elif extrude:
                self._geometry_colors[geometry_id] = (0.6, 0.6, 0.6)
            elif _merge_color is not None:
                self._geometry_colors[geometry_id] = _merge_color
            # Store for VE rescaling / resolution re-snapping in explore()
            _vx = merged_v[0::3]
            _vy = merged_v[1::3]
            _ix = np.clip(np.round(_vx / psx).astype(int), 0, W - 1)
            _iy = np.clip(np.round(_vy / psy).astype(int), 0, H - 1)
            _base_z = terrain_data[_iy, _ix].astype(np.float32)
            _base_z = np.where(np.isnan(_base_z), 0.0, _base_z)
            self._baked_meshes[geometry_id] = (merged_v.copy(),
                                               merged_idx.copy(),
                                               _base_z)
            # Save merged mesh to cache
            if mesh_cache is not None:
                cache_p = _Path(mesh_cache)
                cache_p.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(cache_p, vertices=merged_v,
                                    indices=merged_idx)
                print(f"Saved mesh cache: {cache_p.name} "
                      f"({len(merged_v)//3} verts, "
                      f"{len(merged_idx)//3} tris)")

        if oob_counter[0] > 0:
            warnings.warn(
                f"{oob_counter[0]} GeoJSON coordinate(s) outside raster "
                "extent — clipped to bounds.",
                stacklevel=2,
            )

        if geom_counter > 0:
            self._geometry_colors_dirty = True

        return {
            'features': len(features),
            'geometries': geom_counter,
            'geometry_ids': geometry_ids,
        }

    def triangulate(self, geometry_id='terrain', scale=1.0,
                     pixel_spacing_x=1.0, pixel_spacing_y=1.0):
        """Triangulate the terrain and add it to the scene.

        Creates a triangle mesh from the raster elevation data and adds it
        to the RTX scene for ray tracing operations.

        Parameters
        ----------
        geometry_id : str, optional
            ID for the terrain geometry. Default is 'terrain'.
        scale : float, optional
            Scale factor for elevation values. Default is 1.0.
        pixel_spacing_x : float, optional
            X spacing between pixels in world units. Default is 1.0 (pixel coords).
        pixel_spacing_y : float, optional
            Y spacing between pixels in world units. Default is 1.0 (pixel coords).

        Returns
        -------
        tuple
            (vertices, indices) - The terrain mesh data as numpy arrays.
            Vertices are scaled by pixel_spacing (x=col*spacing_x, y=row*spacing_y).

        Examples
        --------
        >>> # Triangulate terrain in pixel coordinates
        >>> verts, indices = dem.rtx.triangulate()

        >>> # Triangulate with real-world spacing (e.g., 25m per pixel)
        >>> verts, indices = dem.rtx.triangulate(pixel_spacing_x=25.0, pixel_spacing_y=25.0)
        """
        from .mesh import triangulate_terrain
        import numpy as np

        H, W = self._obj.shape

        # Allocate buffers
        num_vertices = H * W
        num_triangles = (H - 1) * (W - 1) * 2
        vertices = np.zeros(num_vertices * 3, dtype=np.float32)
        indices = np.zeros(num_triangles * 3, dtype=np.int32)

        # Triangulate the terrain (creates vertices in pixel coordinates)
        triangulate_terrain(vertices, indices, self._obj, scale=scale)

        # Scale x,y coordinates to world units if pixel spacing != 1.0
        if pixel_spacing_x != 1.0 or pixel_spacing_y != 1.0:
            # Vertices are stored as [x0,y0,z0, x1,y1,z1, ...]
            vertices[0::3] *= pixel_spacing_x  # Scale all x coordinates
            vertices[1::3] *= pixel_spacing_y  # Scale all y coordinates

        # Store pixel spacing and mesh type for use in explore/viewshed
        self._pixel_spacing_x = pixel_spacing_x
        self._pixel_spacing_y = pixel_spacing_y
        self._terrain_mesh_type = 'tin'

        # Add to scene
        self._rtx.add_geometry(geometry_id, vertices, indices)

        return vertices, indices

    def voxelate(self, geometry_id='terrain', scale=1.0, base_elevation=None,
                 pixel_spacing_x=1.0, pixel_spacing_y=1.0):
        """Voxelate the terrain into box-columns and add to the scene.

        Creates a voxelized mesh where each raster cell becomes a rectangular
        column extending from base_elevation up to the cell's elevation.

        Parameters
        ----------
        geometry_id : str, optional
            ID for the terrain geometry. Default is 'terrain'.
        scale : float, optional
            Scale factor for elevation values. Default is 1.0.
        base_elevation : float, optional
            Z coordinate for the bottom of all columns. If None, uses
            min(terrain) * scale so all columns have visible height.
        pixel_spacing_x : float, optional
            X spacing between pixels in world units. Default is 1.0.
        pixel_spacing_y : float, optional
            Y spacing between pixels in world units. Default is 1.0.

        Returns
        -------
        tuple
            (vertices, indices) - The voxelized mesh data as numpy arrays.
        """
        from .mesh import voxelate_terrain
        import numpy as np

        H, W = self._obj.shape

        # Auto-compute base elevation from terrain minimum
        if base_elevation is None:
            terrain_data = self._obj.data if hasattr(self._obj, 'data') else self._obj
            if has_cupy:
                import cupy
                if isinstance(terrain_data, cupy.ndarray):
                    base_elevation = float(cupy.nanmin(terrain_data).get()) * scale
                else:
                    base_elevation = float(np.nanmin(terrain_data)) * scale
            else:
                base_elevation = float(np.nanmin(terrain_data)) * scale

        # Allocate buffers: 8 verts per cell, 12 tris per cell
        num_vertices = H * W * 8
        num_triangles = H * W * 12
        vertices = np.zeros(num_vertices * 3, dtype=np.float32)
        indices = np.zeros(num_triangles * 3, dtype=np.int32)

        # Voxelate the terrain
        voxelate_terrain(vertices, indices, self._obj, scale=scale,
                         base_elevation=base_elevation)

        # Scale x,y coordinates to world units if pixel spacing != 1.0
        if pixel_spacing_x != 1.0 or pixel_spacing_y != 1.0:
            vertices[0::3] *= pixel_spacing_x
            vertices[1::3] *= pixel_spacing_y

        # Store pixel spacing and mesh type for use in explore/viewshed
        self._pixel_spacing_x = pixel_spacing_x
        self._pixel_spacing_y = pixel_spacing_y
        self._terrain_mesh_type = 'voxel'

        # Add to scene
        self._rtx.add_geometry(geometry_id, vertices, indices)

        return vertices, indices

    def flyover(self, output_path, duration=30.0, fps=10.0, orbit_scale=0.6,
                altitude_offset=500.0, fov=60.0, fov_range=None,
                width=1280, height=720, sun_azimuth=225, sun_altitude=35,
                shadows=True, ambient=0.2, colormap='terrain',
                vertical_exaggeration=None, rtx=None):
        """Create a flyover animation orbiting around the terrain.

        Generates a smooth orbital camera path around the terrain center,
        rendering each frame and saving as an animated GIF.

        Parameters
        ----------
        output_path : str
            Path to save the output GIF animation.
        duration : float, optional
            Animation duration in seconds. Default is 30.
        fps : float, optional
            Frames per second. Default is 10.
        orbit_scale : float, optional
            Orbit radius as fraction of terrain dimensions. Default is 0.6.
        altitude_offset : float, optional
            Camera altitude above maximum terrain elevation. Default is 500.
        fov : float, optional
            Base field of view in degrees. Default is 60.
        fov_range : tuple of float, optional
            (min_fov, max_fov) for dynamic zoom. If None, uses constant FOV.
        width : int, optional
            Output image width in pixels. Default is 1280.
        height : int, optional
            Output image height in pixels. Default is 720.
        sun_azimuth : float, optional
            Sun azimuth angle in degrees. Default is 225 (southwest).
        sun_altitude : float, optional
            Sun altitude angle in degrees. Default is 35.
        shadows : bool, optional
            If True, cast shadow rays. Default is True.
        ambient : float, optional
            Ambient light intensity [0-1]. Default is 0.2.
        colormap : str, optional
            Matplotlib colormap name. Default is 'terrain'.
        vertical_exaggeration : float, optional
            Scale factor for elevation values. If None, auto-computed.
        rtx : RTX, optional
            Existing RTX instance to reuse. If None, uses the accessor's
            cached RTX instance.

        Returns
        -------
        str
            Path to the saved GIF file.

        Examples
        --------
        >>> dem.rtx.flyover('flyover.gif')
        >>> dem.rtx.flyover('flyover.gif', duration=60, fps=15)
        >>> dem.rtx.flyover('flyover.gif', fov_range=(30, 70))  # Dynamic zoom
        """
        from .analysis import flyover as _flyover
        # Note: Always pass rtx=None to let the analysis function create its own
        # RTX instance. This prevents the analysis function from clearing the
        # multi-GAS scene that may have been built with triangulate/place_mesh.
        return _flyover(
            self._obj,
            output_path=output_path,
            duration=duration,
            fps=fps,
            orbit_scale=orbit_scale,
            altitude_offset=altitude_offset,
            fov=fov,
            fov_range=fov_range,
            width=width,
            height=height,
            sun_azimuth=sun_azimuth,
            sun_altitude=sun_altitude,
            shadows=shadows,
            ambient=ambient,
            colormap=colormap,
            vertical_exaggeration=vertical_exaggeration,
            rtx=rtx,  # User can override, but default None creates fresh instance
        )

    def view(self, x, y, z, output_path, duration=10.0, fps=12.0,
             look_distance=1000.0, look_down_angle=10.0, fov=70.0,
             width=1280, height=720, sun_azimuth=225, sun_altitude=35,
             shadows=True, ambient=0.2, colormap='terrain',
             vertical_exaggeration=None, rtx=None):
        """Create a 360° panoramic view animation from a specific point.

        Generates a rotating view from a fixed camera position, looking outward
        in all directions to create a panoramic effect.

        Parameters
        ----------
        x : float
            X coordinate of the viewpoint (in pixel coordinates).
        y : float
            Y coordinate of the viewpoint (in pixel coordinates).
        z : float
            Z coordinate (elevation) of the viewpoint.
        output_path : str
            Path to save the output GIF animation.
        duration : float, optional
            Animation duration in seconds. Default is 10.
        fps : float, optional
            Frames per second. Default is 12.
        look_distance : float, optional
            Distance to the look-at point from the camera. Default is 1000.
        look_down_angle : float, optional
            Angle in degrees to look down from horizontal. Default is 10.
        fov : float, optional
            Field of view in degrees. Default is 70 (wide for panoramic feel).
        width : int, optional
            Output image width in pixels. Default is 1280.
        height : int, optional
            Output image height in pixels. Default is 720.
        sun_azimuth : float, optional
            Sun azimuth angle in degrees. Default is 225 (southwest).
        sun_altitude : float, optional
            Sun altitude angle in degrees. Default is 35.
        shadows : bool, optional
            If True, cast shadow rays. Default is True.
        ambient : float, optional
            Ambient light intensity [0-1]. Default is 0.2.
        colormap : str, optional
            Matplotlib colormap name. Default is 'terrain'.
        vertical_exaggeration : float, optional
            Scale factor for elevation values. If None, auto-computed.
        rtx : RTX, optional
            Existing RTX instance to reuse. If None, uses the accessor's
            cached RTX instance.

        Returns
        -------
        str
            Path to the saved GIF file.

        Examples
        --------
        >>> # View from a specific coordinate
        >>> dem.rtx.view(x=500, y=300, z=2500, output_path='hilltop_view.gif')

        >>> # View from terrain surface + height offset
        >>> elev = dem.values[int(y), int(x)]
        >>> dem.rtx.view(x=100, y=200, z=elev + 50, output_path='view.gif')
        """
        from .analysis import view as _view
        return _view(
            self._obj,
            x=x,
            y=y,
            z=z,
            output_path=output_path,
            duration=duration,
            fps=fps,
            look_distance=look_distance,
            look_down_angle=look_down_angle,
            fov=fov,
            width=width,
            height=height,
            sun_azimuth=sun_azimuth,
            sun_altitude=sun_altitude,
            shadows=shadows,
            ambient=ambient,
            colormap=colormap,
            vertical_exaggeration=vertical_exaggeration,
            rtx=rtx,  # User can override, but default None creates fresh instance
        )

    def place_tiles(self, url='osm', zoom=None):
        """Drape XYZ map tiles on terrain. Call before explore().

        Downloads map tiles in the background and composites them into an
        RGB texture matching the terrain grid. When explore() is called
        afterwards, the tiles will be displayed on the terrain surface
        with lighting and shadows applied on top.

        Parameters
        ----------
        url : str, optional
            Provider name or custom URL template with {z}/{x}/{y} placeholders.
            Built-in providers: 'osm', 'satellite', 'topo'.
            Default is 'osm' (OpenStreetMap).
        zoom : int, optional
            Tile zoom level (0–19). If ``None``, defaults to 13.

        Returns
        -------
        XYZTileService
            The tile service instance (tiles begin fetching immediately).
        """
        from .tiles import XYZTileService
        self._tile_service = XYZTileService(
            url_template=url, raster=self._obj, zoom=zoom,
        )
        self._tile_service.fetch_visible_tiles()
        return self._tile_service

    def explore(self, width=800, height=600, render_scale=0.5,
                start_position=None, look_at=None, key_repeat_interval=0.05,
                pixel_spacing_x=None, pixel_spacing_y=None,
                mesh_type='tin', color_stretch='linear', title=None,
                subsample=1, wind_data=None):
        """Launch an interactive terrain viewer with keyboard controls.

        Opens a matplotlib window for terrain exploration with keyboard
        controls. Uses matplotlib's event system - no additional dependencies.

        Any meshes placed with place_mesh() will be visible in the scene.
        Use the G key to cycle through geometry layer information.

        Parameters
        ----------
        width : int, optional
            Window width in pixels. Default is 800.
        height : int, optional
            Window height in pixels. Default is 600.
        render_scale : float, optional
            Render at this fraction of window size (0.25-1.0).
            Lower values give faster response. Default is 0.5.
        start_position : tuple of float, optional
            Starting camera position (x, y, z). If None, starts at the
            south edge looking north.
        look_at : tuple of float, optional
            Initial look-at point. If None, looks toward terrain center.
        key_repeat_interval : float, optional
            Minimum seconds between key repeat events (default 0.05 = 20 FPS max).
            Lower values = more responsive but more GPU load.
        pixel_spacing_x : float, optional
            X spacing between pixels in world units. If None, uses the value
            from the last triangulate() call (default 1.0).
        pixel_spacing_y : float, optional
            Y spacing between pixels in world units. If None, uses the value
            from the last triangulate() call (default 1.0).
        mesh_type : str, optional
            Mesh generation method: 'tin' or 'voxel'.
            Default is 'tin'.
        subsample : int, optional
            Initial terrain subsample factor (1, 2, 4, 8).  Full-resolution
            data is preserved; press Shift+R / R to change at runtime.
            Default is 1 (full resolution).

        Controls
        --------
        - W/Up: Move forward
        - S/Down: Move backward
        - A/Left: Strafe left
        - D/Right: Strafe right
        - Q/Page Up: Move up
        - E/Page Down: Move down
        - I/J/K/L: Look up/left/down/right
        - Scroll wheel: Zoom in/out (FOV)
        - +/-: Adjust movement speed
        - G: Cycle geometry layers (shows info about placed meshes)
        - N: Jump to next geometry in current layer
        - P: Jump to previous geometry in current layer
        - O: Place observer (for viewshed) at camera position
        - V: Toggle viewshed overlay (teal glow shows visible terrain)
        - [/]: Decrease/increase observer height
        - R: Decrease terrain resolution (coarser, up to 8x subsample)
        - Shift+R: Increase terrain resolution (finer, down to 1x)
        - Z: Decrease vertical exaggeration
        - Shift+Z: Increase vertical exaggeration
        - B: Toggle mesh type (TIN / voxel)
        - Y: Cycle color stretch (linear, sqrt, cbrt, log)
        - T: Toggle shadows
        - C: Cycle colormap
        - U: Toggle tile overlay (if place_tiles() was called)
        - F: Save screenshot
        - M: Toggle minimap overlay
        - H: Toggle help overlay
        - X: Exit

        Examples
        --------
        >>> dem.rtx.explore()
        >>> dem.rtx.explore(width=1024, height=768, render_scale=0.25)
        >>> dem.rtx.explore(start_position=(500, -200, 3000))
        >>> # With satellite tiles
        >>> dem.rtx.place_tiles('satellite')
        >>> dem.rtx.explore()
        """
        from .engine import explore as _explore

        # Use stored pixel spacing if not explicitly provided
        spacing_x = pixel_spacing_x if pixel_spacing_x is not None else self._pixel_spacing_x
        spacing_y = pixel_spacing_y if pixel_spacing_y is not None else self._pixel_spacing_y

        # Rebuild terrain geometry if mesh_type doesn't match current state
        current_mesh_type = getattr(self, '_terrain_mesh_type', 'tin')
        if mesh_type != current_mesh_type and 'terrain' in (self._rtx.list_geometries() or []):
            self._rtx.remove_geometry('terrain')
            if mesh_type == 'voxel':
                self.voxelate(geometry_id='terrain',
                              pixel_spacing_x=spacing_x,
                              pixel_spacing_y=spacing_y)
            else:
                self.triangulate(geometry_id='terrain',
                                 pixel_spacing_x=spacing_x,
                                 pixel_spacing_y=spacing_y)

        # Pass geometry color builder if any colors are set
        geometry_colors_builder = None
        if self._geometry_colors:
            geometry_colors_builder = self._build_geometry_colors_gpu

        _explore(
            self._obj,
            width=width,
            height=height,
            render_scale=render_scale,
            start_position=start_position,
            look_at=look_at,
            key_repeat_interval=key_repeat_interval,
            rtx=self._rtx,
            pixel_spacing_x=spacing_x,
            pixel_spacing_y=spacing_y,
            mesh_type=mesh_type,
            color_stretch=color_stretch,
            title=title,
            tile_service=getattr(self, '_tile_service', None),
            geometry_colors_builder=geometry_colors_builder,
            baked_meshes=self._baked_meshes if self._baked_meshes else None,
            subsample=subsample,
            wind_data=wind_data,
        )

    def memory_usage(self):
        """Print and return a breakdown of memory used by the scene.

        Shows terrain raster, per-geometry acceleration structures,
        tile overlay textures, and totals.

        Returns
        -------
        dict
            Raw byte counts for programmatic use.
        """
        lines = []
        lines.append("Memory Usage")
        lines.append("\u2500" * 44)

        total_gpu = 0
        total_cpu = 0

        # -- Terrain raster --
        data = self._obj.data
        raster_bytes = data.nbytes
        shape = self._obj.shape
        dtype = data.dtype
        on_gpu = hasattr(data, 'device')  # cupy arrays have .device
        if on_gpu:
            total_gpu += raster_bytes
        else:
            total_cpu += raster_bytes
        loc = "GPU" if on_gpu else "CPU"
        lines.append(
            f"  Terrain raster     {_fmt_bytes(raster_bytes):>10s}"
            f"   {shape[0]}\u00d7{shape[1]} {dtype} ({loc})"
        )

        # -- Scene geometries --
        scene = self._rtx.memory_usage()
        num_geom = len(scene['geometries'])
        lines.append("")
        lines.append(
            f"  Scene: {scene['mode']}, {num_geom} geometries"
        )

        # Group geometries by prefix (text before last _)
        groups = {}
        for g in scene['geometries']:
            gid = g['id']
            parts = gid.rsplit('_', 1)
            if len(parts) == 2 and parts[1].isdigit():
                prefix = parts[0]
            else:
                prefix = gid
            if prefix not in groups:
                groups[prefix] = {
                    'gas_bytes': 0, 'num_vertices': 0,
                    'num_triangles': 0, 'count': 0,
                }
            grp = groups[prefix]
            grp['gas_bytes'] += g['gas_bytes']
            grp['num_vertices'] += g['num_vertices']
            grp['num_triangles'] += g['num_triangles']
            grp['count'] += 1

        for prefix, grp in groups.items():
            label = prefix
            if grp['count'] > 1:
                label = f"{prefix} ({grp['count']})"
            verts = f"{grp['num_vertices']:,} verts"
            tris = f"{grp['num_triangles']:,} tris"
            lines.append(
                f"    {label:<20s}{_fmt_bytes(grp['gas_bytes']):>10s}"
                f"  {verts:>14s}  {tris:>14s}"
            )
            total_gpu += grp['gas_bytes']

        ias_total = scene['ias_bytes'] + scene['instances_bytes']
        if ias_total > 0:
            lines.append(
                f"    {'IAS + instances':<20s}{_fmt_bytes(ias_total):>10s}"
            )
            total_gpu += ias_total

        if scene['ray_buffers_bytes'] > 0:
            lines.append(
                f"    {'Ray buffers':<20s}"
                f"{_fmt_bytes(scene['ray_buffers_bytes']):>10s}"
            )
            total_gpu += scene['ray_buffers_bytes']

        # -- Tile overlay --
        tile_svc = getattr(self, '_tile_service', None)
        if tile_svc is not None:
            lines.append("")
            lines.append(
                f"  Tile overlay"
                f"       {tile_svc.provider_name}"
            )
            cpu_tex = getattr(tile_svc, '_rgb_texture', None)
            if cpu_tex is not None:
                cpu_bytes = cpu_tex.nbytes
                total_cpu += cpu_bytes
                lines.append(
                    f"    CPU texture       {_fmt_bytes(cpu_bytes):>10s}"
                )
            gpu_tex = getattr(tile_svc, '_gpu_texture', None)
            if gpu_tex is not None:
                gpu_bytes = gpu_tex.nbytes
                total_gpu += gpu_bytes
                lines.append(
                    f"    GPU texture       {_fmt_bytes(gpu_bytes):>10s}"
                )
            tile_cache = getattr(tile_svc, '_tile_cache', None)
            if tile_cache:
                cache_bytes = sum(t.nbytes for t in tile_cache.values())
                num_tiles = len(tile_cache)
                total_cpu += cache_bytes
                lines.append(
                    f"    Tile cache        {_fmt_bytes(cache_bytes):>10s}"
                    f"  ({num_tiles} tiles)"
                )

        # -- Geometry colors --
        if self._geometry_colors:
            color_bytes = len(self._geometry_colors) * 16  # 4 floats
            total_gpu += color_bytes

        # -- Totals --
        lines.append("")
        lines.append(f"  Total GPU          {_fmt_bytes(total_gpu):>10s}")
        lines.append(f"  Total CPU          {_fmt_bytes(total_cpu):>10s}")
        lines.append("\u2500" * 44)

        print("\n".join(lines))

        return {
            'raster_bytes': raster_bytes,
            'raster_on_gpu': on_gpu,
            'scene': scene,
            'total_gpu': total_gpu,
            'total_cpu': total_cpu,
        }


def _fmt_bytes(n):
    """Format a byte count as a human-readable string."""
    if n < 1024:
        return f"{n} B"
    elif n < 1024 * 1024:
        return f"{n / 1024:.1f} KB"
    elif n < 1024 * 1024 * 1024:
        return f"{n / (1024 * 1024):.2f} MB"
    else:
        return f"{n / (1024 * 1024 * 1024):.2f} GB"


@xr.register_dataset_accessor("rtx")
class RTXDatasetAccessor:
    """xarray Dataset accessor for rtxpy GPU-accelerated terrain analysis.

    Allows exploring a Dataset with multiple 2D variables as separate
    geometry layers, toggled with the G key.

    Examples
    --------
    >>> ds = xr.Dataset({'elevation': dem, 'slope': slope_arr})
    >>> ds.rtx.explore(z='elevation')
    """

    def __init__(self, xarray_obj):
        self._obj = xarray_obj
        self._rtx_instance = None
        self._z_var = None
        self._terrain_da_cache = None
        self._pixel_spacing_x = 1.0
        self._pixel_spacing_y = 1.0

    @property
    def _rtx(self):
        """Lazily create and cache an RTX instance."""
        if self._rtx_instance is None:
            self._rtx_instance = RTX()
        return self._rtx_instance

    def _get_terrain_da(self, z):
        """Return a cached DataArray for the terrain variable *z*.

        xarray's ``Dataset.__getitem__`` returns a new DataArray object
        each time, so the accessor (and its RTX instance) would differ
        between calls.  Caching ensures ``place_geojson`` and ``explore``
        share the same RTX scene.
        """
        if self._terrain_da_cache is None or self._z_var != z:
            if z not in self._obj:
                raise ValueError(
                    f"Variable '{z}' not found in Dataset. "
                    f"Available: {list(self._obj.data_vars)}"
                )
            self._z_var = z
            self._terrain_da_cache = self._obj[z]
        return self._terrain_da_cache

    def set_geometry_color(self, geometry_id, color, z=None):
        """Set a solid color override for a geometry.

        Delegates to the cached terrain DataArray's accessor.

        Parameters
        ----------
        geometry_id : str
            The ID of the geometry to color.
        color : tuple of float
            RGB color as (r, g, b) with values [0-1].
        z : str, optional
            Dataset variable to use as terrain. If None, uses the
            variable set by a prior call.
        """
        if z is None:
            z = self._z_var
        if z is None:
            raise ValueError("z must be specified (no prior terrain variable set)")
        terrain_da = self._get_terrain_da(z)
        terrain_da.rtx.set_geometry_color(geometry_id, color)

    def place_geojson(self, geojson, z=None, **kwargs):
        """Load GeoJSON and place features as 3D meshes on terrain.

        Delegates to the DataArray accessor's ``place_geojson`` on the
        terrain variable specified by *z*.

        Parameters
        ----------
        geojson : str, Path, or dict
            GeoJSON file path or dict.
        z : str, optional
            Dataset variable to use as the terrain surface.  If None,
            uses the variable set by a prior ``explore()`` or
            ``place_geojson()`` call.
        **kwargs
            Forwarded to :meth:`RTXAccessor.place_geojson`.

        Returns
        -------
        dict
            ``{'features': int, 'geometries': int, 'geometry_ids': list}``
        """
        if z is None:
            z = self._z_var
        if z is None:
            raise ValueError("z must be specified (no prior terrain variable set)")
        terrain_da = self._get_terrain_da(z)
        return terrain_da.rtx.place_geojson(geojson, **kwargs)

    def explore(self, z, width=800, height=600, render_scale=0.5,
                start_position=None, look_at=None, key_repeat_interval=0.05,
                pixel_spacing_x=None, pixel_spacing_y=None,
                mesh_type='tin', color_stretch='linear', title=None,
                subsample=1, wind_data=None):
        """Launch an interactive terrain viewer with Dataset variables as
        overlay layers cycled with the G key.

        The variable named by ``z`` provides the 3D terrain surface.
        Other 2D variables with matching spatial dimensions become
        colormap overlays: press **G** to cycle which variable drives
        the terrain coloring (elevation → landcover → slope → …).

        Parameters
        ----------
        z : str
            Name of the Dataset variable to use as the primary terrain
            elevation surface.
        width : int, optional
            Window width in pixels. Default is 800.
        height : int, optional
            Window height in pixels. Default is 600.
        render_scale : float, optional
            Render at this fraction of window size (0.25-1.0). Default is 0.5.
        start_position : tuple of float, optional
            Starting camera position (x, y, z). If None, starts at the
            south edge looking north.
        look_at : tuple of float, optional
            Initial look-at point.
        key_repeat_interval : float, optional
            Minimum seconds between key repeat events. Default is 0.05.
        pixel_spacing_x : float, optional
            X spacing between pixels in world units. Default is 1.0.
        pixel_spacing_y : float, optional
            Y spacing between pixels in world units. Default is 1.0.
        mesh_type : str, optional
            Mesh generation method: 'tin' or 'voxel'. Default is 'tin'.

        Examples
        --------
        >>> ds.rtx.explore(z='elevation')
        """
        from .engine import explore as _explore

        terrain_da = self._get_terrain_da(z)

        spacing_x = pixel_spacing_x if pixel_spacing_x is not None else self._pixel_spacing_x
        spacing_y = pixel_spacing_y if pixel_spacing_y is not None else self._pixel_spacing_y

        # NOTE: terrain mesh is built by the engine at the correct
        # (possibly subsampled) resolution — not here at full res.

        # Collect other compatible 2D variables as overlay layers.
        # These share the same terrain mesh but change the colormap data.
        overlay_layers = {}
        terrain_dims = terrain_da.dims
        terrain_shape = terrain_da.shape
        for var_name in self._obj.data_vars:
            if var_name == z:
                continue
            var = self._obj[var_name]
            if var.dims == terrain_dims and var.shape == terrain_shape:
                overlay_layers[var_name] = var.data

        if overlay_layers:
            names = ', '.join(overlay_layers.keys())
            print(f"Overlay layers: {names}  (press G to cycle)")

        # Pass geometry color builder if any colors are set on the terrain DA
        geometry_colors_builder = None
        if terrain_da.rtx._geometry_colors:
            geometry_colors_builder = terrain_da.rtx._build_geometry_colors_gpu

        _explore(
            terrain_da,
            width=width,
            height=height,
            render_scale=render_scale,
            start_position=start_position,
            look_at=look_at,
            key_repeat_interval=key_repeat_interval,
            rtx=terrain_da.rtx._rtx,
            pixel_spacing_x=spacing_x,
            pixel_spacing_y=spacing_y,
            mesh_type=mesh_type,
            overlay_layers=overlay_layers,
            color_stretch=color_stretch,
            title=title,
            tile_service=getattr(self, '_tile_service', None),
            geometry_colors_builder=geometry_colors_builder,
            baked_meshes=terrain_da.rtx._baked_meshes if terrain_da.rtx._baked_meshes else None,
            subsample=subsample,
            wind_data=wind_data,
        )

    def place_tiles(self, url='osm', z=None, zoom=None):
        """Drape XYZ map tiles on terrain. Call before explore().

        Downloads map tiles in the background and composites them into an
        RGB texture matching the terrain grid. When explore() is called
        afterwards, the tiles will be displayed on the terrain surface
        with lighting and shadows applied on top.

        Parameters
        ----------
        url : str, optional
            Provider name or custom URL template with {z}/{x}/{y} placeholders.
            Built-in providers: 'osm', 'satellite', 'topo'.
            Default is 'osm' (OpenStreetMap).
        z : str, optional
            Name of the Dataset variable to use as the spatial reference
            for tile reprojection (CRS, bounds, shape). If None, uses the
            first data variable.
        zoom : int, optional
            Tile zoom level (0–19). If ``None``, defaults to 13.

        Returns
        -------
        XYZTileService
            The tile service instance (tiles begin fetching immediately).
        """
        from .tiles import XYZTileService
        if z is None:
            z = list(self._obj.data_vars)[0]
        raster = self._obj[z]
        self._tile_service = XYZTileService(
            url_template=url, raster=raster, zoom=zoom,
        )
        self._tile_service.fetch_visible_tiles()
        return self._tile_service

    def memory_usage(self, z=None):
        """Print and return a breakdown of memory used by the scene.

        Delegates to the DataArray accessor on the terrain variable.

        Parameters
        ----------
        z : str, optional
            Dataset variable to use as terrain. If None, uses the
            variable set by a prior call, or the first data variable.

        Returns
        -------
        dict
            Raw byte counts for programmatic use.
        """
        if z is None:
            z = self._z_var
        if z is None:
            z = list(self._obj.data_vars)[0]
        terrain_da = self._get_terrain_da(z)
        return terrain_da.rtx.memory_usage()
