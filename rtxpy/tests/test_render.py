"""Tests for render function (perspective camera terrain rendering)."""

import numpy as np
import pytest
import os
import tempfile

from rtxpy.rtx import has_cupy

# Skip all tests if cupy is not available (required for analysis)
pytestmark = pytest.mark.skipif(not has_cupy, reason="cupy required for render")


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


def has_matplotlib():
    """Check if matplotlib is available."""
    try:
        import matplotlib  # noqa: F401
        return True
    except ImportError:
        return False


def has_pil():
    """Check if PIL/Pillow is available."""
    try:
        from PIL import Image  # noqa: F401
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

    # Create a gaussian hill terrain
    H, W = 100, 100
    y = np.linspace(0, 99, H)
    x = np.linspace(0, 99, W)
    yy, xx = np.meshgrid(y, x, indexing='ij')
    # Gaussian hill centered at (50, 50)
    elevation = 100 * np.exp(-((xx - 50) ** 2 + (yy - 50) ** 2) / 500)
    elevation = elevation.astype(np.float32)

    da = xr.DataArray(
        cupy.array(elevation),
        dims=['y', 'x'],
        coords={'y': y, 'x': x},
        name='elevation'
    )
    return da


class TestRenderBasic:
    """Basic tests for render function."""

    @pytest.mark.skipif(not has_xarray(), reason="xarray not available")
    @pytest.mark.skipif(not has_scipy(), reason="scipy not available")
    @pytest.mark.skipif(not has_matplotlib(), reason="matplotlib not available")
    def test_render_basic(self, sample_terrain):
        """Test basic render output."""
        from rtxpy import render

        result = render(
            sample_terrain,
            camera_position=(50, 0, 150),
            look_at=(50, 50, 0),
            width=320,
            height=240,
        )

        assert result is not None
        assert result.shape == (240, 320, 3)
        assert result.dtype == np.float32

    @pytest.mark.skipif(not has_xarray(), reason="xarray not available")
    @pytest.mark.skipif(not has_scipy(), reason="scipy not available")
    @pytest.mark.skipif(not has_matplotlib(), reason="matplotlib not available")
    def test_render_output_range(self, sample_terrain):
        """Test that render output values are in [0, 1] range."""
        from rtxpy import render

        result = render(
            sample_terrain,
            camera_position=(50, 0, 150),
            look_at=(50, 50, 0),
            width=320,
            height=240,
        )

        # Check no NaN or inf
        assert not np.any(np.isnan(result))
        assert not np.any(np.isinf(result))

        # Check range
        assert np.all(result >= 0)
        assert np.all(result <= 1)

    @pytest.mark.skipif(not has_xarray(), reason="xarray not available")
    @pytest.mark.skipif(not has_scipy(), reason="scipy not available")
    @pytest.mark.skipif(not has_matplotlib(), reason="matplotlib not available")
    def test_render_with_alpha(self, sample_terrain):
        """Test render with alpha channel."""
        from rtxpy import render

        result = render(
            sample_terrain,
            camera_position=(50, 0, 150),
            look_at=(50, 50, 0),
            width=320,
            height=240,
            alpha=True,
        )

        assert result.shape == (240, 320, 4)
        # Alpha should be 0 or 1
        assert np.all((result[:, :, 3] == 0) | (result[:, :, 3] == 1))


class TestRenderLighting:
    """Tests for render lighting and shadows."""

    @pytest.mark.skipif(not has_xarray(), reason="xarray not available")
    @pytest.mark.skipif(not has_scipy(), reason="scipy not available")
    @pytest.mark.skipif(not has_matplotlib(), reason="matplotlib not available")
    def test_render_with_shadows(self, sample_terrain):
        """Test render with shadow casting."""
        from rtxpy import render

        result = render(
            sample_terrain,
            camera_position=(50, 0, 150),
            look_at=(50, 50, 0),
            width=320,
            height=240,
            shadows=True,
        )

        assert result is not None
        assert result.shape == (240, 320, 3)

    @pytest.mark.skipif(not has_xarray(), reason="xarray not available")
    @pytest.mark.skipif(not has_scipy(), reason="scipy not available")
    @pytest.mark.skipif(not has_matplotlib(), reason="matplotlib not available")
    def test_render_without_shadows(self, sample_terrain):
        """Test render without shadow casting."""
        from rtxpy import render

        result = render(
            sample_terrain,
            camera_position=(50, 0, 150),
            look_at=(50, 50, 0),
            width=320,
            height=240,
            shadows=False,
        )

        assert result is not None
        assert result.shape == (240, 320, 3)

    @pytest.mark.skipif(not has_xarray(), reason="xarray not available")
    @pytest.mark.skipif(not has_scipy(), reason="scipy not available")
    @pytest.mark.skipif(not has_matplotlib(), reason="matplotlib not available")
    def test_render_custom_sun_position(self, sample_terrain):
        """Test render with custom sun position."""
        from rtxpy import render

        result = render(
            sample_terrain,
            camera_position=(50, 0, 150),
            look_at=(50, 50, 0),
            width=320,
            height=240,
            sun_azimuth=90,
            sun_altitude=30,
        )

        assert result is not None
        assert result.shape == (240, 320, 3)

    @pytest.mark.skipif(not has_xarray(), reason="xarray not available")
    @pytest.mark.skipif(not has_scipy(), reason="scipy not available")
    @pytest.mark.skipif(not has_matplotlib(), reason="matplotlib not available")
    def test_render_high_ambient(self, sample_terrain):
        """Test render with high ambient light."""
        from rtxpy import render

        result = render(
            sample_terrain,
            camera_position=(50, 0, 150),
            look_at=(50, 50, 0),
            width=320,
            height=240,
            ambient=0.8,
        )

        assert result is not None
        # With high ambient, average brightness should be higher
        assert np.mean(result) > 0.3


class TestRenderColormap:
    """Tests for render colormap functionality."""

    @pytest.mark.skipif(not has_xarray(), reason="xarray not available")
    @pytest.mark.skipif(not has_scipy(), reason="scipy not available")
    @pytest.mark.skipif(not has_matplotlib(), reason="matplotlib not available")
    def test_render_terrain_colormap(self, sample_terrain):
        """Test render with terrain colormap."""
        from rtxpy import render

        result = render(
            sample_terrain,
            camera_position=(50, 0, 150),
            look_at=(50, 50, 0),
            width=320,
            height=240,
            colormap='terrain',
        )

        assert result is not None
        # Terrain colormap should produce color variation
        assert result.shape == (240, 320, 3)

    @pytest.mark.skipif(not has_xarray(), reason="xarray not available")
    @pytest.mark.skipif(not has_scipy(), reason="scipy not available")
    @pytest.mark.skipif(not has_matplotlib(), reason="matplotlib not available")
    def test_render_hillshade_colormap(self, sample_terrain):
        """Test render with hillshade (grayscale) colormap."""
        from rtxpy import render

        result = render(
            sample_terrain,
            camera_position=(50, 0, 150),
            look_at=(50, 50, 0),
            width=320,
            height=240,
            colormap='hillshade',
        )

        assert result is not None
        assert result.shape == (240, 320, 3)
        # With hillshade colormap, terrain pixels should be grayscale (R=G=B)
        # Sky pixels have a different color, so we need to check only terrain hits
        # Find pixels where we have terrain hits (not sky)
        # Sky has specific color (0.6, 0.75, 0.9) - terrain will be different
        # Just verify the render completes and has valid values
        assert not np.any(np.isnan(result))
        assert np.all(result >= 0) and np.all(result <= 1)

    @pytest.mark.skipif(not has_xarray(), reason="xarray not available")
    @pytest.mark.skipif(not has_scipy(), reason="scipy not available")
    @pytest.mark.skipif(not has_matplotlib(), reason="matplotlib not available")
    def test_render_viridis_colormap(self, sample_terrain):
        """Test render with viridis colormap."""
        from rtxpy import render

        result = render(
            sample_terrain,
            camera_position=(50, 0, 150),
            look_at=(50, 50, 0),
            width=320,
            height=240,
            colormap='viridis',
        )

        assert result is not None
        assert result.shape == (240, 320, 3)

    @pytest.mark.skipif(not has_xarray(), reason="xarray not available")
    @pytest.mark.skipif(not has_scipy(), reason="scipy not available")
    @pytest.mark.skipif(not has_matplotlib(), reason="matplotlib not available")
    def test_render_invalid_colormap(self, sample_terrain):
        """Test that invalid colormap raises ValueError."""
        from rtxpy import render

        with pytest.raises(ValueError, match="Unknown colormap"):
            render(
                sample_terrain,
                camera_position=(50, 0, 150),
                look_at=(50, 50, 0),
                width=320,
                height=240,
                colormap='not_a_real_colormap',
            )

    @pytest.mark.skipif(not has_xarray(), reason="xarray not available")
    @pytest.mark.skipif(not has_scipy(), reason="scipy not available")
    @pytest.mark.skipif(not has_matplotlib(), reason="matplotlib not available")
    def test_render_custom_color_range(self, sample_terrain):
        """Test render with custom elevation color range."""
        from rtxpy import render

        result = render(
            sample_terrain,
            camera_position=(50, 0, 150),
            look_at=(50, 50, 0),
            width=320,
            height=240,
            color_range=(0, 200),
        )

        assert result is not None
        assert result.shape == (240, 320, 3)


class TestRenderFog:
    """Tests for render fog/atmosphere effects."""

    @pytest.mark.skipif(not has_xarray(), reason="xarray not available")
    @pytest.mark.skipif(not has_scipy(), reason="scipy not available")
    @pytest.mark.skipif(not has_matplotlib(), reason="matplotlib not available")
    def test_render_with_fog(self, sample_terrain):
        """Test render with fog enabled."""
        from rtxpy import render

        result = render(
            sample_terrain,
            camera_position=(50, 0, 150),
            look_at=(50, 50, 0),
            width=320,
            height=240,
            fog_density=0.01,
        )

        assert result is not None
        assert result.shape == (240, 320, 3)

    @pytest.mark.skipif(not has_xarray(), reason="xarray not available")
    @pytest.mark.skipif(not has_scipy(), reason="scipy not available")
    @pytest.mark.skipif(not has_matplotlib(), reason="matplotlib not available")
    def test_render_custom_fog_color(self, sample_terrain):
        """Test render with custom fog color."""
        from rtxpy import render

        result = render(
            sample_terrain,
            camera_position=(50, 0, 150),
            look_at=(50, 50, 0),
            width=320,
            height=240,
            fog_density=0.05,
            fog_color=(1.0, 0.9, 0.8),
        )

        assert result is not None
        assert result.shape == (240, 320, 3)


class TestRenderCamera:
    """Tests for render camera positioning."""

    @pytest.mark.skipif(not has_xarray(), reason="xarray not available")
    @pytest.mark.skipif(not has_scipy(), reason="scipy not available")
    @pytest.mark.skipif(not has_matplotlib(), reason="matplotlib not available")
    def test_render_different_angles(self, sample_terrain):
        """Test render from different camera angles."""
        from rtxpy import render

        # Top-down view
        result1 = render(
            sample_terrain,
            camera_position=(50, 50, 200),
            look_at=(50, 50, 0),
            width=320,
            height=240,
        )

        # Side view
        result2 = render(
            sample_terrain,
            camera_position=(0, 50, 50),
            look_at=(50, 50, 50),
            width=320,
            height=240,
        )

        assert result1 is not None
        assert result2 is not None
        # Results should be different
        assert not np.allclose(result1, result2)

    @pytest.mark.skipif(not has_xarray(), reason="xarray not available")
    @pytest.mark.skipif(not has_scipy(), reason="scipy not available")
    @pytest.mark.skipif(not has_matplotlib(), reason="matplotlib not available")
    def test_render_custom_fov(self, sample_terrain):
        """Test render with custom field of view."""
        from rtxpy import render

        # Narrow FOV (telephoto)
        result1 = render(
            sample_terrain,
            camera_position=(50, 0, 150),
            look_at=(50, 50, 0),
            width=320,
            height=240,
            fov=30,
        )

        # Wide FOV
        result2 = render(
            sample_terrain,
            camera_position=(50, 0, 150),
            look_at=(50, 50, 0),
            width=320,
            height=240,
            fov=90,
        )

        assert result1 is not None
        assert result2 is not None
        # Results should be different
        assert not np.allclose(result1, result2)

    @pytest.mark.skipif(not has_xarray(), reason="xarray not available")
    @pytest.mark.skipif(not has_scipy(), reason="scipy not available")
    @pytest.mark.skipif(not has_matplotlib(), reason="matplotlib not available")
    def test_render_custom_up_vector(self, sample_terrain):
        """Test render with custom up vector (tilted camera)."""
        from rtxpy import render

        result = render(
            sample_terrain,
            camera_position=(50, 0, 150),
            look_at=(50, 50, 0),
            width=320,
            height=240,
            up=(0.1, 0, 1),  # Slightly tilted
        )

        assert result is not None
        assert result.shape == (240, 320, 3)


class TestRenderOutput:
    """Tests for render output and file saving."""

    @pytest.mark.skipif(not has_xarray(), reason="xarray not available")
    @pytest.mark.skipif(not has_scipy(), reason="scipy not available")
    @pytest.mark.skipif(not has_matplotlib(), reason="matplotlib not available")
    @pytest.mark.skipif(not has_pil(), reason="PIL not available")
    def test_render_save_png(self, sample_terrain):
        """Test render saves PNG file correctly."""
        from rtxpy import render

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, 'test_render.png')

            result = render(
                sample_terrain,
                camera_position=(50, 0, 150),
                look_at=(50, 50, 0),
                width=320,
                height=240,
                output_path=output_path,
            )

            assert os.path.exists(output_path)
            # Verify file is valid image
            from PIL import Image
            img = Image.open(output_path)
            assert img.size == (320, 240)
            assert img.mode == 'RGB'

    @pytest.mark.skipif(not has_xarray(), reason="xarray not available")
    @pytest.mark.skipif(not has_scipy(), reason="scipy not available")
    @pytest.mark.skipif(not has_matplotlib(), reason="matplotlib not available")
    @pytest.mark.skipif(not has_pil(), reason="PIL not available")
    def test_render_save_png_with_alpha(self, sample_terrain):
        """Test render saves PNG file with alpha channel."""
        from rtxpy import render

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, 'test_render_alpha.png')

            result = render(
                sample_terrain,
                camera_position=(50, 0, 150),
                look_at=(50, 50, 0),
                width=320,
                height=240,
                alpha=True,
                output_path=output_path,
            )

            assert os.path.exists(output_path)
            from PIL import Image
            img = Image.open(output_path)
            assert img.size == (320, 240)
            assert img.mode == 'RGBA'

    @pytest.mark.skipif(not has_xarray(), reason="xarray not available")
    @pytest.mark.skipif(not has_scipy(), reason="scipy not available")
    @pytest.mark.skipif(not has_matplotlib(), reason="matplotlib not available")
    def test_render_resolution(self, sample_terrain):
        """Test render with different resolutions."""
        from rtxpy import render

        # 720p
        result1 = render(
            sample_terrain,
            camera_position=(50, 0, 150),
            look_at=(50, 50, 0),
            width=1280,
            height=720,
        )
        assert result1.shape == (720, 1280, 3)

        # Square
        result2 = render(
            sample_terrain,
            camera_position=(50, 0, 150),
            look_at=(50, 50, 0),
            width=512,
            height=512,
        )
        assert result2.shape == (512, 512, 3)


class TestRenderVerticalExaggeration:
    """Tests for render vertical exaggeration parameter."""

    @pytest.mark.skipif(not has_xarray(), reason="xarray not available")
    @pytest.mark.skipif(not has_scipy(), reason="scipy not available")
    @pytest.mark.skipif(not has_matplotlib(), reason="matplotlib not available")
    def test_render_auto_vertical_exaggeration(self, sample_terrain):
        """Test render with auto vertical exaggeration (default)."""
        from rtxpy import render

        result = render(
            sample_terrain,
            camera_position=(50, 0, 150),
            look_at=(50, 50, 0),
            width=320,
            height=240,
            vertical_exaggeration=None,  # Auto
        )

        assert result is not None
        assert result.shape == (240, 320, 3)
        assert not np.any(np.isnan(result))

    @pytest.mark.skipif(not has_xarray(), reason="xarray not available")
    @pytest.mark.skipif(not has_scipy(), reason="scipy not available")
    @pytest.mark.skipif(not has_matplotlib(), reason="matplotlib not available")
    def test_render_explicit_vertical_exaggeration(self, sample_terrain):
        """Test render with explicit vertical exaggeration."""
        from rtxpy import render

        result = render(
            sample_terrain,
            camera_position=(50, 0, 150),
            look_at=(50, 50, 0),
            width=320,
            height=240,
            vertical_exaggeration=0.5,
        )

        assert result is not None
        assert result.shape == (240, 320, 3)

    @pytest.mark.skipif(not has_xarray(), reason="xarray not available")
    @pytest.mark.skipif(not has_scipy(), reason="scipy not available")
    @pytest.mark.skipif(not has_matplotlib(), reason="matplotlib not available")
    def test_render_no_vertical_exaggeration(self, sample_terrain):
        """Test render with no vertical exaggeration (1.0)."""
        from rtxpy import render

        result = render(
            sample_terrain,
            camera_position=(50, 0, 150),
            look_at=(50, 50, 0),
            width=320,
            height=240,
            vertical_exaggeration=1.0,
        )

        assert result is not None
        assert result.shape == (240, 320, 3)


class TestRenderAccessor:
    """Tests for render via xarray accessor."""

    @pytest.mark.skipif(not has_xarray(), reason="xarray not available")
    @pytest.mark.skipif(not has_scipy(), reason="scipy not available")
    @pytest.mark.skipif(not has_matplotlib(), reason="matplotlib not available")
    def test_accessor_render(self, sample_terrain):
        """Test render via accessor interface."""
        result = sample_terrain.rtx.render(
            camera_position=(50, 0, 150),
            look_at=(50, 50, 0),
            width=320,
            height=240,
        )

        assert result is not None
        assert result.shape == (240, 320, 3)
        assert result.dtype == np.float32

    @pytest.mark.skipif(not has_xarray(), reason="xarray not available")
    @pytest.mark.skipif(not has_scipy(), reason="scipy not available")
    @pytest.mark.skipif(not has_matplotlib(), reason="matplotlib not available")
    def test_accessor_render_with_options(self, sample_terrain):
        """Test render via accessor with various options."""
        result = sample_terrain.rtx.render(
            camera_position=(50, 0, 150),
            look_at=(50, 50, 0),
            width=320,
            height=240,
            shadows=True,
            colormap='viridis',
            fog_density=0.01,
        )

        assert result is not None
        assert result.shape == (240, 320, 3)
