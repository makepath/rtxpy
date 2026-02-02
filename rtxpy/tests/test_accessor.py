"""Tests for the xarray accessor."""

import numpy as np
import pytest

from rtxpy.rtx import has_cupy

# Skip all tests if cupy is not available
pytestmark = pytest.mark.skipif(not has_cupy, reason="cupy required for accessor tests")


def has_xarray():
    """Check if xarray is available."""
    try:
        import xarray  # noqa: F401
        return True
    except ImportError:
        return False


def has_scipy():
    """Check if scipy is available."""
    try:
        import scipy  # noqa: F401
        return True
    except ImportError:
        return False


@pytest.fixture
def sample_terrain():
    """Create a sample terrain DataArray for testing."""
    if not has_xarray():
        pytest.skip("xarray not available")

    import xarray as xr
    import cupy

    # Create a simple bowl-shaped terrain
    H, W = 50, 50
    y = np.linspace(0, 49, H)
    x = np.linspace(0, 49, W)
    yy, xx = np.meshgrid(y, x, indexing='ij')
    elevation = 100 - 0.1 * ((xx - 25) ** 2 + (yy - 25) ** 2)
    elevation = elevation.astype(np.float32)

    # Create DataArray with cupy data
    da = xr.DataArray(
        cupy.array(elevation),
        dims=['y', 'x'],
        coords={'y': y, 'x': x},
        name='elevation'
    )
    return da


@pytest.fixture
def sample_terrain_numpy():
    """Create a sample terrain DataArray with numpy data."""
    if not has_xarray():
        pytest.skip("xarray not available")

    import xarray as xr

    H, W = 50, 50
    y = np.linspace(0, 49, H)
    x = np.linspace(0, 49, W)
    yy, xx = np.meshgrid(y, x, indexing='ij')
    elevation = 100 - 0.1 * ((xx - 25) ** 2 + (yy - 25) ** 2)
    elevation = elevation.astype(np.float32)

    da = xr.DataArray(
        elevation,
        dims=['y', 'x'],
        coords={'y': y, 'x': x},
        name='elevation'
    )
    return da


class TestAccessorRegistration:
    """Tests for accessor registration and basic functionality."""

    @pytest.mark.skipif(not has_xarray(), reason="xarray not available")
    def test_accessor_available(self, sample_terrain):
        """Test that the accessor is registered and available."""
        assert hasattr(sample_terrain, 'rtx')

    @pytest.mark.skipif(not has_xarray(), reason="xarray not available")
    def test_rtx_instance_caching(self, sample_terrain):
        """Test that RTX instance is lazily created and cached."""
        accessor = sample_terrain.rtx
        # First access should create instance
        rtx1 = accessor._rtx
        # Second access should return same instance
        rtx2 = accessor._rtx
        assert rtx1 is rtx2


class TestToCupy:
    """Tests for the to_cupy() method."""

    @pytest.mark.skipif(not has_xarray(), reason="xarray not available")
    def test_to_cupy_from_numpy(self, sample_terrain_numpy):
        """Test converting numpy data to cupy."""
        import cupy

        result = sample_terrain_numpy.rtx.to_cupy()

        assert isinstance(result.data, cupy.ndarray)
        # Check data values are preserved
        np.testing.assert_array_almost_equal(
            cupy.asnumpy(result.data),
            sample_terrain_numpy.data
        )

    @pytest.mark.skipif(not has_xarray(), reason="xarray not available")
    def test_to_cupy_idempotent(self, sample_terrain):
        """Test that to_cupy on cupy data returns the same DataArray."""
        result = sample_terrain.rtx.to_cupy()
        # Should return the same object since data is already cupy
        assert result is sample_terrain

    @pytest.mark.skipif(not has_xarray(), reason="xarray not available")
    def test_to_cupy_preserves_coords(self, sample_terrain_numpy):
        """Test that to_cupy preserves coordinates and dims."""
        result = sample_terrain_numpy.rtx.to_cupy()

        np.testing.assert_array_equal(
            result.coords['x'].values,
            sample_terrain_numpy.coords['x'].values
        )
        np.testing.assert_array_equal(
            result.coords['y'].values,
            sample_terrain_numpy.coords['y'].values
        )
        assert result.dims == sample_terrain_numpy.dims


class TestViewshedAccessor:
    """Tests for viewshed() via accessor."""

    @pytest.mark.skipif(not has_xarray(), reason="xarray not available")
    def test_viewshed_via_accessor(self, sample_terrain):
        """Test basic viewshed computation via accessor."""
        result = sample_terrain.rtx.viewshed(x=25, y=25, observer_elev=10)

        assert result is not None
        assert result.shape == sample_terrain.shape
        assert result.name == "viewshed"

    @pytest.mark.skipif(not has_xarray(), reason="xarray not available")
    def test_viewshed_with_custom_rtx(self, sample_terrain):
        """Test viewshed with a custom RTX instance."""
        from rtxpy import RTX

        custom_rtx = RTX()
        result = sample_terrain.rtx.viewshed(x=25, y=25, rtx=custom_rtx)

        assert result is not None
        assert result.shape == sample_terrain.shape

    @pytest.mark.skipif(not has_xarray(), reason="xarray not available")
    def test_viewshed_rtx_reuse(self, sample_terrain):
        """Test that multiple viewshed calls reuse the cached RTX instance."""
        accessor = sample_terrain.rtx

        # First call creates and caches RTX
        result1 = accessor.viewshed(x=25, y=25)

        # Get the cached RTX instance
        rtx_instance = accessor._rtx

        # Second call should use the same RTX
        result2 = accessor.viewshed(x=30, y=30)

        # RTX instance should be the same
        assert accessor._rtx is rtx_instance
        assert result1 is not None
        assert result2 is not None


class TestHillshadeAccessor:
    """Tests for hillshade() via accessor."""

    @pytest.mark.skipif(not has_xarray(), reason="xarray not available")
    @pytest.mark.skipif(not has_scipy(), reason="scipy not available")
    def test_hillshade_via_accessor(self, sample_terrain):
        """Test basic hillshade computation via accessor."""
        result = sample_terrain.rtx.hillshade()

        assert result is not None
        assert result.shape == sample_terrain.shape
        assert result.name == "hillshade"

    @pytest.mark.skipif(not has_xarray(), reason="xarray not available")
    @pytest.mark.skipif(not has_scipy(), reason="scipy not available")
    def test_hillshade_with_shadows(self, sample_terrain):
        """Test hillshade with shadow casting via accessor."""
        result = sample_terrain.rtx.hillshade(shadows=True)

        assert result is not None
        assert result.shape == sample_terrain.shape

    @pytest.mark.skipif(not has_xarray(), reason="xarray not available")
    @pytest.mark.skipif(not has_scipy(), reason="scipy not available")
    def test_hillshade_custom_params(self, sample_terrain):
        """Test hillshade with custom parameters via accessor."""
        result = sample_terrain.rtx.hillshade(
            azimuth=90,
            angle_altitude=45,
            name="custom_shade"
        )

        assert result is not None
        assert result.name == "custom_shade"

    @pytest.mark.skipif(not has_xarray(), reason="xarray not available")
    @pytest.mark.skipif(not has_scipy(), reason="scipy not available")
    def test_hillshade_with_custom_rtx(self, sample_terrain):
        """Test hillshade with a custom RTX instance."""
        from rtxpy import RTX

        custom_rtx = RTX()
        result = sample_terrain.rtx.hillshade(rtx=custom_rtx)

        assert result is not None
        assert result.shape == sample_terrain.shape


class TestMethodChaining:
    """Tests for method chaining with the accessor."""

    @pytest.mark.skipif(not has_xarray(), reason="xarray not available")
    @pytest.mark.skipif(not has_scipy(), reason="scipy not available")
    def test_to_cupy_then_hillshade(self, sample_terrain_numpy):
        """Test chaining to_cupy() with hillshade()."""
        result = sample_terrain_numpy.rtx.to_cupy().rtx.hillshade()

        assert result is not None
        assert result.shape == sample_terrain_numpy.shape
        assert result.name == "hillshade"

    @pytest.mark.skipif(not has_xarray(), reason="xarray not available")
    def test_to_cupy_then_viewshed(self, sample_terrain_numpy):
        """Test chaining to_cupy() with viewshed()."""
        result = sample_terrain_numpy.rtx.to_cupy().rtx.viewshed(x=25, y=25)

        assert result is not None
        assert result.shape == sample_terrain_numpy.shape
        assert result.name == "viewshed"


class TestGeometryManagement:
    """Tests for geometry management methods via accessor."""

    @pytest.fixture
    def simple_mesh(self):
        """Create simple mesh data for testing."""
        import cupy
        vertices = cupy.float32([0, 0, 0, 1, 0, 0, 0.5, 1, 0])
        indices = cupy.int32([0, 1, 2])
        return vertices, indices

    @pytest.mark.skipif(not has_xarray(), reason="xarray not available")
    def test_clear(self, sample_terrain, simple_mesh):
        """Test clear() removes all geometries."""
        vertices, indices = simple_mesh

        accessor = sample_terrain.rtx
        accessor.add_geometry("mesh1", vertices, indices)
        accessor.add_geometry("mesh2", vertices, indices)
        assert accessor.get_geometry_count() == 2

        accessor.clear()
        assert accessor.get_geometry_count() == 0
        assert accessor.list_geometries() == []

    @pytest.mark.skipif(not has_xarray(), reason="xarray not available")
    def test_add_geometry(self, sample_terrain, simple_mesh):
        """Test add_geometry() adds geometry to scene."""
        vertices, indices = simple_mesh

        accessor = sample_terrain.rtx
        accessor.clear()

        result = accessor.add_geometry("test_mesh", vertices, indices)
        assert result == 0
        assert accessor.get_geometry_count() == 1
        assert "test_mesh" in accessor.list_geometries()

    @pytest.mark.skipif(not has_xarray(), reason="xarray not available")
    def test_add_geometry_with_transform(self, sample_terrain, simple_mesh):
        """Test add_geometry() with a transform."""
        vertices, indices = simple_mesh

        accessor = sample_terrain.rtx
        accessor.clear()

        transform = [1, 0, 0, 10, 0, 1, 0, 0, 0, 0, 1, 0]  # Translate x by 10
        result = accessor.add_geometry("translated", vertices, indices, transform=transform)
        assert result == 0
        assert accessor.has_geometry("translated")

    @pytest.mark.skipif(not has_xarray(), reason="xarray not available")
    def test_remove_geometry(self, sample_terrain, simple_mesh):
        """Test remove_geometry() removes geometry from scene."""
        vertices, indices = simple_mesh

        accessor = sample_terrain.rtx
        accessor.clear()

        accessor.add_geometry("to_remove", vertices, indices)
        assert accessor.has_geometry("to_remove")

        result = accessor.remove_geometry("to_remove")
        assert result == 0
        assert not accessor.has_geometry("to_remove")

    @pytest.mark.skipif(not has_xarray(), reason="xarray not available")
    def test_remove_geometry_not_found(self, sample_terrain):
        """Test remove_geometry() returns -1 for non-existent geometry."""
        accessor = sample_terrain.rtx
        accessor.clear()

        result = accessor.remove_geometry("nonexistent")
        assert result == -1

    @pytest.mark.skipif(not has_xarray(), reason="xarray not available")
    def test_list_geometries(self, sample_terrain, simple_mesh):
        """Test list_geometries() returns all geometry IDs."""
        vertices, indices = simple_mesh

        accessor = sample_terrain.rtx
        accessor.clear()

        accessor.add_geometry("alpha", vertices, indices)
        accessor.add_geometry("beta", vertices, indices)
        accessor.add_geometry("gamma", vertices, indices)

        geoms = accessor.list_geometries()
        assert len(geoms) == 3
        assert "alpha" in geoms
        assert "beta" in geoms
        assert "gamma" in geoms

    @pytest.mark.skipif(not has_xarray(), reason="xarray not available")
    def test_get_geometry_count(self, sample_terrain, simple_mesh):
        """Test get_geometry_count() returns correct count."""
        vertices, indices = simple_mesh

        accessor = sample_terrain.rtx
        accessor.clear()

        assert accessor.get_geometry_count() == 0

        accessor.add_geometry("mesh1", vertices, indices)
        assert accessor.get_geometry_count() == 1

        accessor.add_geometry("mesh2", vertices, indices)
        assert accessor.get_geometry_count() == 2

        accessor.remove_geometry("mesh1")
        assert accessor.get_geometry_count() == 1

    @pytest.mark.skipif(not has_xarray(), reason="xarray not available")
    def test_has_geometry(self, sample_terrain, simple_mesh):
        """Test has_geometry() returns correct boolean."""
        vertices, indices = simple_mesh

        accessor = sample_terrain.rtx
        accessor.clear()

        assert not accessor.has_geometry("test")

        accessor.add_geometry("test", vertices, indices)
        assert accessor.has_geometry("test")
        assert not accessor.has_geometry("other")

        accessor.remove_geometry("test")
        assert not accessor.has_geometry("test")

    @pytest.mark.skipif(not has_xarray(), reason="xarray not available")
    def test_replace_geometry_same_id(self, sample_terrain, simple_mesh):
        """Test that adding geometry with same ID replaces it."""
        vertices, indices = simple_mesh

        accessor = sample_terrain.rtx
        accessor.clear()

        accessor.add_geometry("mesh", vertices, indices)
        assert accessor.get_geometry_count() == 1

        # Add again with same ID
        accessor.add_geometry("mesh", vertices, indices)
        assert accessor.get_geometry_count() == 1  # Still just one

    @pytest.mark.skipif(not has_xarray(), reason="xarray not available")
    def test_geometry_methods_use_cached_rtx(self, sample_terrain, simple_mesh):
        """Test that geometry methods use the cached RTX instance."""
        vertices, indices = simple_mesh

        accessor = sample_terrain.rtx
        accessor.clear()

        # Force RTX creation
        rtx_instance = accessor._rtx

        accessor.add_geometry("mesh", vertices, indices)

        # Should still be using the same RTX instance
        assert accessor._rtx is rtx_instance


class TestMultipleDataArrays:
    """Tests for behavior when multiple DataArrays use the accessor.

    Each RTX instance (and thus each accessor) has isolated geometry state.
    The underlying OptiX context is shared for efficiency, but geometry
    (acceleration structures, meshes) is per-instance.
    """

    @pytest.fixture
    def terrain_a(self):
        """Create first terrain DataArray."""
        if not has_xarray():
            pytest.skip("xarray not available")

        import xarray as xr
        import cupy

        H, W = 30, 30
        y = np.linspace(0, 29, H)
        x = np.linspace(0, 29, W)
        yy, xx = np.meshgrid(y, x, indexing='ij')
        # Bowl shape
        elevation = 100 - 0.1 * ((xx - 15) ** 2 + (yy - 15) ** 2)

        return xr.DataArray(
            cupy.array(elevation.astype(np.float32)),
            dims=['y', 'x'],
            coords={'y': y, 'x': x},
            name='terrain_a'
        )

    @pytest.fixture
    def terrain_b(self):
        """Create second terrain DataArray with different data."""
        if not has_xarray():
            pytest.skip("xarray not available")

        import xarray as xr
        import cupy

        H, W = 30, 30
        y = np.linspace(0, 29, H)
        x = np.linspace(0, 29, W)
        yy, xx = np.meshgrid(y, x, indexing='ij')
        # Peak shape (different from terrain_a)
        elevation = 50 + 0.2 * ((xx - 15) ** 2 + (yy - 15) ** 2)

        return xr.DataArray(
            cupy.array(elevation.astype(np.float32)),
            dims=['y', 'x'],
            coords={'y': y, 'x': x},
            name='terrain_b'
        )

    @pytest.mark.skipif(not has_xarray(), reason="xarray not available")
    def test_different_dataarrays_have_different_accessors(self, terrain_a, terrain_b):
        """Test that each DataArray gets its own accessor instance."""
        accessor_a = terrain_a.rtx
        accessor_b = terrain_b.rtx

        # Each DataArray should have its own accessor instance
        assert accessor_a is not accessor_b
        assert accessor_a._obj is terrain_a
        assert accessor_b._obj is terrain_b

    @pytest.mark.skipif(not has_xarray(), reason="xarray not available")
    def test_accessors_have_isolated_geometry_state(self, terrain_a, terrain_b):
        """Test that different accessors have isolated geometry state.

        Each accessor's RTX instance has its own geometry state, so adding
        geometry via one accessor does not affect others.
        """
        import cupy
        vertices = cupy.float32([0, 0, 0, 1, 0, 0, 0.5, 1, 0])
        indices = cupy.int32([0, 1, 2])

        # Clear both to start fresh
        terrain_a.rtx.clear()
        terrain_b.rtx.clear()

        # Add geometry via terrain_a's accessor
        terrain_a.rtx.add_geometry("from_terrain_a", vertices, indices)

        # terrain_a should see the geometry
        assert terrain_a.rtx.has_geometry("from_terrain_a")
        assert terrain_a.rtx.get_geometry_count() == 1

        # terrain_b should NOT see the geometry (isolated state)
        assert not terrain_b.rtx.has_geometry("from_terrain_a")
        assert terrain_b.rtx.get_geometry_count() == 0

    @pytest.mark.skipif(not has_xarray(), reason="xarray not available")
    def test_clear_from_one_accessor_does_not_affect_others(self, terrain_a, terrain_b):
        """Test that clear() from one accessor only clears its own geometry."""
        import cupy
        vertices = cupy.float32([0, 0, 0, 1, 0, 0, 0.5, 1, 0])
        indices = cupy.int32([0, 1, 2])

        terrain_a.rtx.clear()
        terrain_b.rtx.clear()

        # Add geometry via both accessors
        terrain_a.rtx.add_geometry("mesh_a", vertices, indices)
        terrain_b.rtx.add_geometry("mesh_b", vertices, indices)

        assert terrain_a.rtx.get_geometry_count() == 1
        assert terrain_b.rtx.get_geometry_count() == 1

        # Clear from terrain_b
        terrain_b.rtx.clear()

        # terrain_a should still have its geometry
        assert terrain_a.rtx.get_geometry_count() == 1
        assert terrain_a.rtx.has_geometry("mesh_a")

        # terrain_b should be empty
        assert terrain_b.rtx.get_geometry_count() == 0

    @pytest.mark.skipif(not has_xarray(), reason="xarray not available")
    def test_viewshed_uses_correct_terrain(self, terrain_a, terrain_b):
        """Test that viewshed uses the correct terrain for each DataArray."""
        # Run viewshed on terrain_a
        result_a = terrain_a.rtx.viewshed(x=15, y=15, observer_elev=10)
        assert result_a is not None
        assert result_a.shape == terrain_a.shape

        # Run viewshed on terrain_b
        result_b = terrain_b.rtx.viewshed(x=15, y=15, observer_elev=10)
        assert result_b is not None
        assert result_b.shape == terrain_b.shape

        # Results should be different because terrains are different
        import cupy
        data_a = cupy.asnumpy(result_a.data)
        data_b = cupy.asnumpy(result_b.data)

        # The viewsheds should not be identical (different terrain shapes)
        assert not np.allclose(data_a, data_b, equal_nan=True)

    @pytest.mark.skipif(not has_xarray(), reason="xarray not available")
    def test_alternating_viewshed_calls_work_correctly(self, terrain_a, terrain_b):
        """Test that alternating between DataArrays produces correct results.

        With isolated state, each accessor caches its own terrain mesh,
        so alternating calls should be efficient and correct.
        """
        # Get reference results
        ref_a = terrain_a.rtx.viewshed(x=15, y=15, observer_elev=5)
        ref_b = terrain_b.rtx.viewshed(x=15, y=15, observer_elev=5)

        # Alternate multiple times
        for _ in range(3):
            result_a = terrain_a.rtx.viewshed(x=15, y=15, observer_elev=5)
            result_b = terrain_b.rtx.viewshed(x=15, y=15, observer_elev=5)

            import cupy
            # Results should match references
            np.testing.assert_array_almost_equal(
                cupy.asnumpy(result_a.data),
                cupy.asnumpy(ref_a.data)
            )
            np.testing.assert_array_almost_equal(
                cupy.asnumpy(result_b.data),
                cupy.asnumpy(ref_b.data)
            )

    @pytest.mark.skipif(not has_xarray(), reason="xarray not available")
    @pytest.mark.skipif(not has_scipy(), reason="scipy not available")
    def test_hillshade_uses_correct_terrain(self, terrain_a, terrain_b):
        """Test that hillshade uses the correct terrain for each DataArray."""
        # Run hillshade on both terrains
        result_a = terrain_a.rtx.hillshade(shadows=True)
        result_b = terrain_b.rtx.hillshade(shadows=True)

        import cupy
        data_a = cupy.asnumpy(result_a.data)
        data_b = cupy.asnumpy(result_b.data)

        # Results should differ (different terrain shapes)
        # Compare interior (edges are NaN)
        interior_a = data_a[1:-1, 1:-1]
        interior_b = data_b[1:-1, 1:-1]
        assert not np.allclose(interior_a, interior_b)

    @pytest.mark.skipif(not has_xarray(), reason="xarray not available")
    def test_same_geometry_id_in_different_accessors_is_independent(self, terrain_a, terrain_b):
        """Test that same geometry ID in different accessors are independent.

        Each accessor has isolated state, so using the same geometry ID
        in different accessors creates separate geometries.
        """
        import cupy

        terrain_a.rtx.clear()
        terrain_b.rtx.clear()

        # Add geometry with ID "shared" from terrain_a (at z=0)
        verts_a = cupy.float32([0, 0, 0, 1, 0, 0, 0.5, 1, 0])
        indices = cupy.int32([0, 1, 2])
        terrain_a.rtx.add_geometry("shared", verts_a, indices)

        # Add geometry with same ID from terrain_b (at z=5)
        verts_b = cupy.float32([0, 0, 5, 1, 0, 5, 0.5, 1, 5])
        terrain_b.rtx.add_geometry("shared", verts_b, indices)

        # Each accessor should have exactly one geometry
        assert terrain_a.rtx.get_geometry_count() == 1
        assert terrain_b.rtx.get_geometry_count() == 1

        # Both have "shared" but they are independent
        assert terrain_a.rtx.has_geometry("shared")
        assert terrain_b.rtx.has_geometry("shared")

        # Verify they trace differently (different z heights)
        rays_a = cupy.float32([0.5, 0.33, 10, 0, 0, 0, -1, 1000])
        hits_a = cupy.float32([0, 0, 0, 0])
        terrain_a.rtx._rtx.trace(rays_a, hits_a, 1)

        rays_b = cupy.float32([0.5, 0.33, 10, 0, 0, 0, -1, 1000])
        hits_b = cupy.float32([0, 0, 0, 0])
        terrain_b.rtx._rtx.trace(rays_b, hits_b, 1)

        # terrain_a's mesh is at z=0, so distance should be ~10
        # terrain_b's mesh is at z=5, so distance should be ~5
        np.testing.assert_almost_equal(float(hits_a[0]), 10.0, decimal=1)
        np.testing.assert_almost_equal(float(hits_b[0]), 5.0, decimal=1)
