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

    def explore(self, width=800, height=600, render_scale=0.5,
                start_position=None, look_at=None, key_repeat_interval=0.05,
                pixel_spacing_x=None, pixel_spacing_y=None,
                mesh_type='tin', color_stretch='linear'):
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
            Starting camera position (x, y, z). If None, auto-positions
            above the terrain center.
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
        - F: Save screenshot
        - M: Toggle minimap overlay
        - H: Toggle help overlay
        - X: Exit

        Examples
        --------
        >>> dem.rtx.explore()
        >>> dem.rtx.explore(width=1024, height=768, render_scale=0.25)
        >>> dem.rtx.explore(start_position=(500, -200, 3000))
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
        )


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
        self._pixel_spacing_x = 1.0
        self._pixel_spacing_y = 1.0

    @property
    def _rtx(self):
        """Lazily create and cache an RTX instance."""
        if self._rtx_instance is None:
            self._rtx_instance = RTX()
        return self._rtx_instance

    def explore(self, z, width=800, height=600, render_scale=0.5,
                start_position=None, look_at=None, key_repeat_interval=0.05,
                pixel_spacing_x=None, pixel_spacing_y=None,
                mesh_type='tin', color_stretch='linear'):
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
            Starting camera position (x, y, z).
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

        if z not in self._obj:
            raise ValueError(
                f"Variable '{z}' not found in Dataset. "
                f"Available: {list(self._obj.data_vars)}"
            )

        self._z_var = z
        terrain_da = self._obj[z]

        spacing_x = pixel_spacing_x if pixel_spacing_x is not None else self._pixel_spacing_x
        spacing_y = pixel_spacing_y if pixel_spacing_y is not None else self._pixel_spacing_y

        # Build the terrain surface using the DataArray accessor
        if mesh_type == 'voxel':
            terrain_da.rtx.voxelate(
                geometry_id='terrain',
                pixel_spacing_x=spacing_x,
                pixel_spacing_y=spacing_y,
            )
        else:
            terrain_da.rtx.triangulate(
                geometry_id='terrain',
                pixel_spacing_x=spacing_x,
                pixel_spacing_y=spacing_y,
            )

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
        )
