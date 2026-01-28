"""Tests for analysis functions (viewshed and hillshade)."""

import numpy as np
import pytest

from rtxpy.rtx import has_cupy

# Skip all tests if cupy is not available (required for analysis)
pytestmark = pytest.mark.skipif(not has_cupy, reason="cupy required for analysis functions")


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


class TestViewshed:
    """Tests for viewshed function."""

    @pytest.mark.skipif(not has_xarray(), reason="xarray not available")
    def test_viewshed_basic(self, sample_terrain):
        """Test basic viewshed computation."""
        from rtxpy import viewshed

        result = viewshed(sample_terrain, x=25, y=25, observer_elev=10)

        assert result is not None
        assert result.shape == sample_terrain.shape
        assert result.name == "viewshed"

    @pytest.mark.skipif(not has_xarray(), reason="xarray not available")
    def test_viewshed_coords_preserved(self, sample_terrain):
        """Test that output coordinates match input."""
        from rtxpy import viewshed

        result = viewshed(sample_terrain, x=25, y=25)

        np.testing.assert_array_equal(result.coords['x'].values, sample_terrain.coords['x'].values)
        np.testing.assert_array_equal(result.coords['y'].values, sample_terrain.coords['y'].values)

    @pytest.mark.skipif(not has_xarray(), reason="xarray not available")
    def test_viewshed_invalid_x(self, sample_terrain):
        """Test that invalid x coordinate raises ValueError."""
        from rtxpy import viewshed

        with pytest.raises(ValueError, match="x argument outside"):
            viewshed(sample_terrain, x=-100, y=25)

        with pytest.raises(ValueError, match="x argument outside"):
            viewshed(sample_terrain, x=1000, y=25)

    @pytest.mark.skipif(not has_xarray(), reason="xarray not available")
    def test_viewshed_invalid_y(self, sample_terrain):
        """Test that invalid y coordinate raises ValueError."""
        from rtxpy import viewshed

        with pytest.raises(ValueError, match="y argument outside"):
            viewshed(sample_terrain, x=25, y=-100)

        with pytest.raises(ValueError, match="y argument outside"):
            viewshed(sample_terrain, x=25, y=1000)

    @pytest.mark.skipif(not has_xarray(), reason="xarray not available")
    def test_viewshed_with_target_elev(self, sample_terrain):
        """Test viewshed with target elevation offset."""
        from rtxpy import viewshed

        result = viewshed(sample_terrain, x=25, y=25, observer_elev=5, target_elev=2)

        assert result is not None
        assert result.shape == sample_terrain.shape

    def test_viewshed_requires_cupy_data(self):
        """Test that numpy data raises ValueError."""
        if not has_xarray():
            pytest.skip("xarray not available")

        import xarray as xr
        from rtxpy import viewshed

        # Create DataArray with numpy data (not cupy)
        da = xr.DataArray(
            np.zeros((10, 10), dtype=np.float32),
            dims=['y', 'x'],
            coords={'y': range(10), 'x': range(10)}
        )

        with pytest.raises(ValueError, match="cupy array"):
            viewshed(da, x=5, y=5)


class TestHillshade:
    """Tests for hillshade function."""

    @pytest.mark.skipif(not has_xarray(), reason="xarray not available")
    @pytest.mark.skipif(not has_scipy(), reason="scipy not available")
    def test_hillshade_basic(self, sample_terrain):
        """Test basic hillshade computation."""
        from rtxpy import hillshade

        result = hillshade(sample_terrain)

        assert result is not None
        assert result.shape == sample_terrain.shape
        assert result.name == "hillshade"

    @pytest.mark.skipif(not has_xarray(), reason="xarray not available")
    @pytest.mark.skipif(not has_scipy(), reason="scipy not available")
    def test_hillshade_with_shadows(self, sample_terrain):
        """Test hillshade with shadow casting."""
        from rtxpy import hillshade

        result = hillshade(sample_terrain, shadows=True)

        assert result is not None
        assert result.shape == sample_terrain.shape

    @pytest.mark.skipif(not has_xarray(), reason="xarray not available")
    @pytest.mark.skipif(not has_scipy(), reason="scipy not available")
    def test_hillshade_custom_sun_position(self, sample_terrain):
        """Test hillshade with custom sun position."""
        from rtxpy import hillshade

        result = hillshade(sample_terrain, azimuth=90, angle_altitude=45)

        assert result is not None
        assert result.shape == sample_terrain.shape

    @pytest.mark.skipif(not has_xarray(), reason="xarray not available")
    @pytest.mark.skipif(not has_scipy(), reason="scipy not available")
    def test_hillshade_edges_nan(self, sample_terrain):
        """Test that hillshade output has NaN edges."""
        import cupy
        from rtxpy import hillshade

        result = hillshade(sample_terrain)

        # Convert to numpy for checking
        if isinstance(result.data, cupy.ndarray):
            data = cupy.asnumpy(result.data)
        else:
            data = result.data

        # Check edges are NaN
        assert np.all(np.isnan(data[0, :]))
        assert np.all(np.isnan(data[-1, :]))
        assert np.all(np.isnan(data[:, 0]))
        assert np.all(np.isnan(data[:, -1]))

    @pytest.mark.skipif(not has_xarray(), reason="xarray not available")
    @pytest.mark.skipif(not has_scipy(), reason="scipy not available")
    def test_hillshade_values_in_range(self, sample_terrain):
        """Test that hillshade values are between 0 and 1."""
        import cupy
        from rtxpy import hillshade

        result = hillshade(sample_terrain)

        if isinstance(result.data, cupy.ndarray):
            data = cupy.asnumpy(result.data)
        else:
            data = result.data

        # Ignore NaN values at edges
        valid_data = data[1:-1, 1:-1]
        assert np.all(valid_data >= 0)
        assert np.all(valid_data <= 1)

    @pytest.mark.skipif(not has_xarray(), reason="xarray not available")
    @pytest.mark.skipif(not has_scipy(), reason="scipy not available")
    def test_hillshade_custom_name(self, sample_terrain):
        """Test hillshade with custom output name."""
        from rtxpy import hillshade

        result = hillshade(sample_terrain, name="my_hillshade")

        assert result.name == "my_hillshade"

    @pytest.mark.skipif(not has_xarray(), reason="xarray not available")
    @pytest.mark.skipif(not has_scipy(), reason="scipy not available")
    def test_hillshade_coords_preserved(self, sample_terrain):
        """Test that output coordinates match input."""
        from rtxpy import hillshade

        result = hillshade(sample_terrain)

        np.testing.assert_array_equal(result.coords['x'].values, sample_terrain.coords['x'].values)
        np.testing.assert_array_equal(result.coords['y'].values, sample_terrain.coords['y'].values)


class TestGetSunDir:
    """Tests for get_sun_dir utility function."""

    @pytest.mark.skipif(not has_scipy(), reason="scipy not available")
    def test_sun_overhead(self):
        """Test sun direction when overhead (altitude=90)."""
        from rtxpy.analysis import get_sun_dir

        sun_dir = get_sun_dir(angle_altitude=90, azimuth=0)

        # Should point straight up
        np.testing.assert_almost_equal(sun_dir[0], 0, decimal=5)
        np.testing.assert_almost_equal(sun_dir[1], 0, decimal=5)
        np.testing.assert_almost_equal(sun_dir[2], 1, decimal=5)

    @pytest.mark.skipif(not has_scipy(), reason="scipy not available")
    def test_sun_from_south(self):
        """Test sun direction from the south."""
        from rtxpy.analysis import get_sun_dir

        sun_dir = get_sun_dir(angle_altitude=45, azimuth=180)

        # Should have positive z (above horizon)
        assert sun_dir[2] > 0  # Above horizon
        # The function adds 180 to azimuth internally, so azimuth=180
        # means sun from north (y positive direction toward sun)

    @pytest.mark.skipif(not has_scipy(), reason="scipy not available")
    def test_sun_returns_unit_vector(self):
        """Test that sun direction is a unit vector."""
        from rtxpy.analysis import get_sun_dir

        for azimuth in [0, 90, 180, 270]:
            for altitude in [15, 30, 45, 60]:
                sun_dir = get_sun_dir(angle_altitude=altitude, azimuth=azimuth)
                length = np.sqrt(np.sum(sun_dir ** 2))
                np.testing.assert_almost_equal(length, 1.0, decimal=5)
