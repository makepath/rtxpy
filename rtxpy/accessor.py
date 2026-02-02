"""xarray accessor for rtxpy GPU-accelerated terrain analysis."""

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
            rtx=rtx if rtx is not None else self._rtx
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
            rtx=rtx if rtx is not None else self._rtx
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
