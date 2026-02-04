"""Tests for animation functions (flyover and view)."""

import numpy as np
import pytest
import os
import tempfile

from rtxpy.rtx import has_cupy

# Skip all tests if cupy is not available (required for analysis)
pytestmark = pytest.mark.skipif(not has_cupy, reason="cupy required for animation")


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


def has_imageio():
    """Check if imageio is available."""
    try:
        import imageio  # noqa: F401
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


class TestFlyoverBasic:
    """Basic tests for flyover function."""

    @pytest.mark.skipif(not has_xarray(), reason="xarray not available")
    @pytest.mark.skipif(not has_scipy(), reason="scipy not available")
    @pytest.mark.skipif(not has_matplotlib(), reason="matplotlib not available")
    @pytest.mark.skipif(not has_imageio(), reason="imageio not available")
    def test_flyover_basic(self, sample_terrain):
        """Test basic flyover animation creation."""
        from rtxpy import flyover

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, 'flyover.gif')

            result = flyover(
                sample_terrain,
                output_path=output_path,
                duration=2.0,  # Short for testing
                fps=5.0,
                width=160,
                height=120,
            )

            assert result == output_path
            assert os.path.exists(output_path)
            # Verify file size is non-zero
            assert os.path.getsize(output_path) > 0

    @pytest.mark.skipif(not has_xarray(), reason="xarray not available")
    @pytest.mark.skipif(not has_scipy(), reason="scipy not available")
    @pytest.mark.skipif(not has_matplotlib(), reason="matplotlib not available")
    @pytest.mark.skipif(not has_imageio(), reason="imageio not available")
    def test_flyover_with_fov_range(self, sample_terrain):
        """Test flyover with dynamic zoom (FOV range)."""
        from rtxpy import flyover

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, 'flyover_zoom.gif')

            result = flyover(
                sample_terrain,
                output_path=output_path,
                duration=2.0,
                fps=5.0,
                width=160,
                height=120,
                fov_range=(40, 80),  # Dynamic zoom
            )

            assert os.path.exists(output_path)

    @pytest.mark.skipif(not has_xarray(), reason="xarray not available")
    @pytest.mark.skipif(not has_scipy(), reason="scipy not available")
    @pytest.mark.skipif(not has_matplotlib(), reason="matplotlib not available")
    @pytest.mark.skipif(not has_imageio(), reason="imageio not available")
    def test_flyover_custom_orbit(self, sample_terrain):
        """Test flyover with custom orbit parameters."""
        from rtxpy import flyover

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, 'flyover_orbit.gif')

            result = flyover(
                sample_terrain,
                output_path=output_path,
                duration=2.0,
                fps=5.0,
                width=160,
                height=120,
                orbit_scale=0.8,
                altitude_offset=200.0,
            )

            assert os.path.exists(output_path)


class TestFlyoverAccessor:
    """Tests for flyover via xarray accessor."""

    @pytest.mark.skipif(not has_xarray(), reason="xarray not available")
    @pytest.mark.skipif(not has_scipy(), reason="scipy not available")
    @pytest.mark.skipif(not has_matplotlib(), reason="matplotlib not available")
    @pytest.mark.skipif(not has_imageio(), reason="imageio not available")
    def test_accessor_flyover(self, sample_terrain):
        """Test flyover via accessor interface."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, 'flyover_accessor.gif')

            result = sample_terrain.rtx.flyover(
                output_path,
                duration=2.0,
                fps=5.0,
                width=160,
                height=120,
            )

            assert result == output_path
            assert os.path.exists(output_path)

    @pytest.mark.skipif(not has_xarray(), reason="xarray not available")
    @pytest.mark.skipif(not has_scipy(), reason="scipy not available")
    @pytest.mark.skipif(not has_matplotlib(), reason="matplotlib not available")
    @pytest.mark.skipif(not has_imageio(), reason="imageio not available")
    def test_accessor_flyover_with_options(self, sample_terrain):
        """Test flyover via accessor with various options."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, 'flyover_options.gif')

            result = sample_terrain.rtx.flyover(
                output_path,
                duration=2.0,
                fps=5.0,
                width=160,
                height=120,
                shadows=True,
                colormap='viridis',
                fov_range=(30, 70),
            )

            assert os.path.exists(output_path)


class TestViewBasic:
    """Basic tests for view function."""

    @pytest.mark.skipif(not has_xarray(), reason="xarray not available")
    @pytest.mark.skipif(not has_scipy(), reason="scipy not available")
    @pytest.mark.skipif(not has_matplotlib(), reason="matplotlib not available")
    @pytest.mark.skipif(not has_imageio(), reason="imageio not available")
    def test_view_basic(self, sample_terrain):
        """Test basic view animation creation."""
        from rtxpy import view

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, 'view.gif')

            result = view(
                sample_terrain,
                x=50,
                y=50,
                z=150,
                output_path=output_path,
                duration=2.0,
                fps=6.0,
                width=160,
                height=120,
            )

            assert result == output_path
            assert os.path.exists(output_path)
            assert os.path.getsize(output_path) > 0

    @pytest.mark.skipif(not has_xarray(), reason="xarray not available")
    @pytest.mark.skipif(not has_scipy(), reason="scipy not available")
    @pytest.mark.skipif(not has_matplotlib(), reason="matplotlib not available")
    @pytest.mark.skipif(not has_imageio(), reason="imageio not available")
    def test_view_custom_look_params(self, sample_terrain):
        """Test view with custom look distance and angle."""
        from rtxpy import view

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, 'view_look.gif')

            result = view(
                sample_terrain,
                x=50,
                y=50,
                z=150,
                output_path=output_path,
                duration=2.0,
                fps=6.0,
                width=160,
                height=120,
                look_distance=500.0,
                look_down_angle=20.0,
            )

            assert os.path.exists(output_path)

    @pytest.mark.skipif(not has_xarray(), reason="xarray not available")
    @pytest.mark.skipif(not has_scipy(), reason="scipy not available")
    @pytest.mark.skipif(not has_matplotlib(), reason="matplotlib not available")
    @pytest.mark.skipif(not has_imageio(), reason="imageio not available")
    def test_view_from_terrain_surface(self, sample_terrain):
        """Test view from terrain surface with height offset."""
        from rtxpy import view

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, 'view_surface.gif')

            # Get terrain elevation at viewpoint
            terrain_data = sample_terrain.data
            if hasattr(terrain_data, 'get'):  # cupy array
                terrain_data = terrain_data.get()
            else:
                terrain_data = np.asarray(terrain_data)

            x, y = 50, 50
            terrain_elev = terrain_data[int(y), int(x)]
            observer_height = 10  # 10 units above terrain

            result = view(
                sample_terrain,
                x=x,
                y=y,
                z=terrain_elev + observer_height,
                output_path=output_path,
                duration=2.0,
                fps=6.0,
                width=160,
                height=120,
            )

            assert os.path.exists(output_path)


class TestViewAccessor:
    """Tests for view via xarray accessor."""

    @pytest.mark.skipif(not has_xarray(), reason="xarray not available")
    @pytest.mark.skipif(not has_scipy(), reason="scipy not available")
    @pytest.mark.skipif(not has_matplotlib(), reason="matplotlib not available")
    @pytest.mark.skipif(not has_imageio(), reason="imageio not available")
    def test_accessor_view(self, sample_terrain):
        """Test view via accessor interface."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, 'view_accessor.gif')

            result = sample_terrain.rtx.view(
                x=50,
                y=50,
                z=150,
                output_path=output_path,
                duration=2.0,
                fps=6.0,
                width=160,
                height=120,
            )

            assert result == output_path
            assert os.path.exists(output_path)

    @pytest.mark.skipif(not has_xarray(), reason="xarray not available")
    @pytest.mark.skipif(not has_scipy(), reason="scipy not available")
    @pytest.mark.skipif(not has_matplotlib(), reason="matplotlib not available")
    @pytest.mark.skipif(not has_imageio(), reason="imageio not available")
    def test_accessor_view_with_options(self, sample_terrain):
        """Test view via accessor with various options."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, 'view_options.gif')

            result = sample_terrain.rtx.view(
                x=50,
                y=50,
                z=150,
                output_path=output_path,
                duration=2.0,
                fps=6.0,
                width=160,
                height=120,
                shadows=True,
                colormap='terrain',
                fov=60,
            )

            assert os.path.exists(output_path)


class TestAnimationGifValidity:
    """Tests to verify generated GIFs are valid."""

    @pytest.mark.skipif(not has_xarray(), reason="xarray not available")
    @pytest.mark.skipif(not has_scipy(), reason="scipy not available")
    @pytest.mark.skipif(not has_matplotlib(), reason="matplotlib not available")
    @pytest.mark.skipif(not has_imageio(), reason="imageio not available")
    def test_flyover_gif_readable(self, sample_terrain):
        """Test that flyover GIF can be read back."""
        import imageio

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, 'flyover.gif')

            sample_terrain.rtx.flyover(
                output_path,
                duration=2.0,
                fps=5.0,
                width=160,
                height=120,
            )

            # Read back the GIF
            reader = imageio.get_reader(output_path)
            frames = list(reader)
            reader.close()

            # Should have frames
            assert len(frames) >= 2
            # Each frame should be correct size
            for frame in frames:
                assert frame.shape[0] == 120  # height
                assert frame.shape[1] == 160  # width
                assert frame.shape[2] in [3, 4]  # RGB or RGBA

    @pytest.mark.skipif(not has_xarray(), reason="xarray not available")
    @pytest.mark.skipif(not has_scipy(), reason="scipy not available")
    @pytest.mark.skipif(not has_matplotlib(), reason="matplotlib not available")
    @pytest.mark.skipif(not has_imageio(), reason="imageio not available")
    def test_view_gif_readable(self, sample_terrain):
        """Test that view GIF can be read back."""
        import imageio

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, 'view.gif')

            sample_terrain.rtx.view(
                x=50,
                y=50,
                z=150,
                output_path=output_path,
                duration=2.0,
                fps=6.0,
                width=160,
                height=120,
            )

            # Read back the GIF
            reader = imageio.get_reader(output_path)
            frames = list(reader)
            reader.close()

            # Should have frames
            assert len(frames) >= 2
            # Each frame should be correct size
            for frame in frames:
                assert frame.shape[0] == 120
                assert frame.shape[1] == 160


class TestAnimationFrameCount:
    """Tests to verify animation frame counts."""

    @pytest.mark.skipif(not has_xarray(), reason="xarray not available")
    @pytest.mark.skipif(not has_scipy(), reason="scipy not available")
    @pytest.mark.skipif(not has_matplotlib(), reason="matplotlib not available")
    @pytest.mark.skipif(not has_imageio(), reason="imageio not available")
    def test_flyover_frame_count(self, sample_terrain):
        """Test flyover produces correct number of frames."""
        import imageio

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, 'flyover.gif')

            duration = 3.0
            fps = 5.0
            expected_frames = int(duration * fps)

            sample_terrain.rtx.flyover(
                output_path,
                duration=duration,
                fps=fps,
                width=80,
                height=60,
            )

            reader = imageio.get_reader(output_path)
            frames = list(reader)
            reader.close()

            assert len(frames) == expected_frames

    @pytest.mark.skipif(not has_xarray(), reason="xarray not available")
    @pytest.mark.skipif(not has_scipy(), reason="scipy not available")
    @pytest.mark.skipif(not has_matplotlib(), reason="matplotlib not available")
    @pytest.mark.skipif(not has_imageio(), reason="imageio not available")
    def test_view_frame_count(self, sample_terrain):
        """Test view produces correct number of frames."""
        import imageio

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, 'view.gif')

            duration = 2.0
            fps = 8.0
            expected_frames = int(duration * fps)

            sample_terrain.rtx.view(
                x=50,
                y=50,
                z=150,
                output_path=output_path,
                duration=duration,
                fps=fps,
                width=80,
                height=60,
            )

            reader = imageio.get_reader(output_path)
            frames = list(reader)
            reader.close()

            assert len(frames) == expected_frames
