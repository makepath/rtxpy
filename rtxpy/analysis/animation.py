"""Animation functions for terrain flyover and point-of-view rendering.

This module provides GPU-accelerated ray tracing for creating animated
visualizations of terrain data.
"""

import numpy as np
from pathlib import Path
from typing import Optional, Tuple, Union

from ..rtx import RTX, has_cupy
from .render import render

if has_cupy:
    import cupy


def _lazy_import_imageio():
    """Lazily import imageio with helpful error message."""
    try:
        import imageio
        return imageio
    except ImportError:
        raise ImportError(
            "imageio is required for animation export. "
            "Install it with: pip install imageio "
            "or: pip install rtxpy[all]"
        )


def _lazy_import_scipy():
    """Lazily import scipy.ndimage with helpful error message."""
    try:
        from scipy.ndimage import gaussian_filter1d
        return gaussian_filter1d
    except ImportError:
        raise ImportError(
            "scipy is required for smooth animations. "
            "Install it with: pip install scipy "
            "or: pip install rtxpy[all]"
        )


def flyover(
    raster,
    output_path: str,
    duration: float = 30.0,
    fps: float = 10.0,
    orbit_scale: float = 0.6,
    altitude_offset: float = 500.0,
    fov: float = 60.0,
    fov_range: Optional[Tuple[float, float]] = None,
    width: int = 1280,
    height: int = 720,
    sun_azimuth: float = 225,
    sun_altitude: float = 35,
    shadows: bool = True,
    ambient: float = 0.2,
    colormap: str = 'terrain',
    vertical_exaggeration: Optional[float] = None,
    rtx: RTX = None,
) -> str:
    """Create a flyover animation orbiting around the terrain.

    Generates a smooth orbital camera path around the terrain center,
    with optional dynamic zoom (FOV variation) based on terrain features.

    Parameters
    ----------
    raster : xarray.DataArray
        2D raster terrain data with 'x' and 'y' coordinates.
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
        Existing RTX instance to reuse.

    Returns
    -------
    str
        Path to the saved GIF file.

    Examples
    --------
    >>> dem.rtx.flyover('flyover.gif', duration=60, fps=15)
    >>> dem.rtx.flyover('flyover.gif', fov_range=(30, 70))  # Dynamic zoom
    """
    imageio = _lazy_import_imageio()
    gaussian_filter1d = _lazy_import_scipy()

    if not has_cupy:
        raise ImportError(
            "cupy is required for flyover animation. "
            "Install with: conda install -c conda-forge cupy"
        )

    # Get terrain dimensions
    H, W = raster.shape
    terrain_data = raster.data
    if hasattr(terrain_data, 'get'):  # cupy array
        terrain_data = terrain_data.get()
    else:
        terrain_data = np.asarray(terrain_data)

    elev_min = float(np.nanmin(terrain_data))
    elev_max = float(np.nanmax(terrain_data))
    elev_mean = float(np.nanmean(terrain_data))

    # Scene center
    center_x = W / 2
    center_y = H / 2
    center_z = elev_mean

    # Orbit parameters
    orbit_radius_x = W * orbit_scale
    orbit_radius_y = H * orbit_scale
    cruise_altitude = elev_max + altitude_offset

    # Frame count
    num_frames = int(duration * fps)
    if num_frames < 2:
        num_frames = 2

    # Generate orbital path (one complete orbit)
    angles = np.linspace(0, 2 * np.pi, num_frames, endpoint=False)

    # Camera positions
    cam_x = center_x + orbit_radius_x * np.sin(angles)
    cam_y = center_y - orbit_radius_y * np.cos(angles)
    cam_z = np.full(num_frames, cruise_altitude)

    # Look-at: always center
    look_x = np.full(num_frames, center_x)
    look_y = np.full(num_frames, center_y)
    look_z = np.full(num_frames, center_z)

    # FOV values (constant or dynamic)
    if fov_range is not None:
        min_fov, max_fov = fov_range
        # Vary FOV sinusoidally for smooth zoom in/out
        fov_values = (min_fov + max_fov) / 2 + (max_fov - min_fov) / 2 * np.sin(angles * 2)
        fov_values = gaussian_filter1d(fov_values, sigma=5)
    else:
        fov_values = np.full(num_frames, fov)

    # Render frames
    frames = []
    for i in range(num_frames):
        camera_pos = (cam_x[i], cam_y[i], cam_z[i])
        look_at = (look_x[i], look_y[i], look_z[i])

        # Sun follows camera for consistent lighting
        current_sun_azimuth = sun_azimuth + np.degrees(angles[i])

        img = render(
            raster,
            camera_position=camera_pos,
            look_at=look_at,
            fov=fov_values[i],
            width=width,
            height=height,
            sun_azimuth=current_sun_azimuth,
            sun_altitude=sun_altitude,
            shadows=shadows,
            ambient=ambient,
            colormap=colormap,
            vertical_exaggeration=vertical_exaggeration,
            rtx=rtx,
        )

        # Convert to uint8
        img_uint8 = (np.clip(img, 0, 1) * 255).astype(np.uint8)
        frames.append(img_uint8)

    # Save GIF
    output_path = str(output_path)
    # Use duration (ms per frame) instead of deprecated fps parameter
    duration_ms = 1000.0 / fps
    imageio.mimsave(output_path, frames, duration=duration_ms, loop=0)

    return output_path


def view(
    raster,
    x: float,
    y: float,
    z: float,
    output_path: str,
    duration: float = 10.0,
    fps: float = 12.0,
    look_distance: float = 1000.0,
    look_down_angle: float = 10.0,
    fov: float = 70.0,
    width: int = 1280,
    height: int = 720,
    sun_azimuth: float = 225,
    sun_altitude: float = 35,
    shadows: bool = True,
    ambient: float = 0.2,
    colormap: str = 'terrain',
    vertical_exaggeration: Optional[float] = None,
    rtx: RTX = None,
) -> str:
    """Create a 360° panoramic view animation from a specific point.

    Generates a rotating view from a fixed camera position, looking outward
    in all directions to create a panoramic effect.

    Parameters
    ----------
    raster : xarray.DataArray
        2D raster terrain data with 'x' and 'y' coordinates.
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
        Existing RTX instance to reuse.

    Returns
    -------
    str
        Path to the saved GIF file.

    Examples
    --------
    >>> # View from a hilltop
    >>> dem.rtx.view(x=500, y=300, z=2500, output_path='hilltop_view.gif')

    >>> # View from tower position with terrain-sampled elevation
    >>> tower_elev = dem.values[int(y), int(x)] + 45  # 45m tower
    >>> dem.rtx.view(x=100, y=200, z=tower_elev, output_path='tower_view.gif')
    """
    imageio = _lazy_import_imageio()

    if not has_cupy:
        raise ImportError(
            "cupy is required for view animation. "
            "Install with: conda install -c conda-forge cupy"
        )

    # Frame count
    num_frames = int(duration * fps)
    if num_frames < 2:
        num_frames = 2

    # Generate rotation angles (one complete 360° rotation)
    angles = np.linspace(0, 2 * np.pi, num_frames, endpoint=False)

    # Convert look_down_angle to radians
    look_down_rad = np.radians(look_down_angle)

    # Camera position is fixed
    camera_pos = (float(x), float(y), float(z))

    # Render frames
    frames = []
    for i in range(num_frames):
        angle = angles[i]

        # Look-at point rotates around the camera
        # Horizontal component
        look_x = x + look_distance * np.sin(angle) * np.cos(look_down_rad)
        look_y = y + look_distance * np.cos(angle) * np.cos(look_down_rad)
        # Vertical component (looking slightly down)
        look_z = z - look_distance * np.sin(look_down_rad)

        look_at = (look_x, look_y, look_z)

        img = render(
            raster,
            camera_position=camera_pos,
            look_at=look_at,
            fov=fov,
            width=width,
            height=height,
            sun_azimuth=sun_azimuth,
            sun_altitude=sun_altitude,
            shadows=shadows,
            ambient=ambient,
            colormap=colormap,
            vertical_exaggeration=vertical_exaggeration,
            rtx=rtx,
        )

        # Convert to uint8
        img_uint8 = (np.clip(img, 0, 1) * 255).astype(np.uint8)
        frames.append(img_uint8)

    # Save GIF
    output_path = str(output_path)
    # Use duration (ms per frame) instead of deprecated fps parameter
    duration_ms = 1000.0 / fps
    imageio.mimsave(output_path, frames, duration=duration_ms, loop=0)

    return output_path
