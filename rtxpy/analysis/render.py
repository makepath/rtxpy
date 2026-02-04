"""Perspective camera rendering for movie-quality terrain visualization.

This module provides GPU-accelerated ray tracing for terrain rendering with
perspective cameras, shadows, atmospheric effects, and colormap-based shading.
"""

from numba import cuda
import numpy as np
import math

from typing import Optional, Tuple

from .._cuda_utils import calc_dims, add, diff, mul, dot, float3, make_float3, invert
from ._common import prepare_mesh
from .hillshade import get_sun_dir
from ..rtx import RTX, has_cupy

if has_cupy:
    import cupy


def _lazy_import_xarray():
    """Lazily import xarray with helpful error message."""
    try:
        import xarray as xr
        return xr
    except ImportError:
        raise ImportError(
            "xarray is required for render. "
            "Install it with: pip install xarray "
            "or: pip install rtxpy[analysis]"
        )


def _lazy_import_matplotlib():
    """Lazily import matplotlib with helpful error message."""
    try:
        import matplotlib.pyplot as plt
        import matplotlib.cm as cm
        return plt, cm
    except ImportError:
        raise ImportError(
            "matplotlib is required for colormap rendering. "
            "Install it with: pip install matplotlib "
            "or: pip install rtxpy[all]"
        )


def _lazy_import_pil():
    """Lazily import PIL with helpful error message."""
    try:
        from PIL import Image
        return Image
    except ImportError:
        raise ImportError(
            "Pillow is required for saving images. "
            "Install it with: pip install Pillow "
            "or: pip install rtxpy[all]"
        )


def _compute_camera_basis(camera_position, look_at, up):
    """Compute camera basis vectors (forward, right, up) from position and target.

    Parameters
    ----------
    camera_position : tuple of float
        Camera position (x, y, z).
    look_at : tuple of float
        Target point to look at (x, y, z).
    up : tuple of float
        World up vector.

    Returns
    -------
    tuple of np.ndarray
        (forward, right, up) unit vectors.
    """
    camera_pos = np.array(camera_position, dtype=np.float32)
    target = np.array(look_at, dtype=np.float32)
    world_up = np.array(up, dtype=np.float32)

    forward = target - camera_pos
    forward = forward / np.linalg.norm(forward)

    right = np.cross(forward, world_up)
    right_norm = np.linalg.norm(right)

    # Handle case where forward is parallel to up vector
    if right_norm < 1e-6:
        # Use a different up vector to compute right
        alt_up = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        right = np.cross(forward, alt_up)
        right_norm = np.linalg.norm(right)
        if right_norm < 1e-6:
            alt_up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
            right = np.cross(forward, alt_up)
            right_norm = np.linalg.norm(right)

    right = right / right_norm

    cam_up = np.cross(right, forward)
    cam_up = cam_up / np.linalg.norm(cam_up)

    return forward, right, cam_up


def _get_colormap_lut(colormap, num_entries=256):
    """Generate a color lookup table from a matplotlib colormap.

    Parameters
    ----------
    colormap : str
        Name of matplotlib colormap or 'hillshade'.
    num_entries : int
        Number of entries in the LUT.

    Returns
    -------
    np.ndarray
        Color lookup table of shape (num_entries, 3) with float32 values [0-1].
    """
    if colormap == 'hillshade':
        # Grayscale LUT for hillshade mode
        lut = np.zeros((num_entries, 3), dtype=np.float32)
        for i in range(num_entries):
            v = i / (num_entries - 1)
            lut[i] = [v, v, v]
        return lut

    plt, cm = _lazy_import_matplotlib()

    try:
        cmap = plt.get_cmap(colormap)
    except ValueError:
        raise ValueError(f"Unknown colormap: {colormap}")

    lut = np.zeros((num_entries, 3), dtype=np.float32)
    for i in range(num_entries):
        rgba = cmap(i / (num_entries - 1))
        lut[i] = [rgba[0], rgba[1], rgba[2]]

    return lut


@cuda.jit(device=True)
def _normalize(v):
    """Normalize a float3 vector."""
    length = math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])
    if length > 0:
        return float3(v[0] / length, v[1] / length, v[2] / length)
    return v


@cuda.jit
def _generate_perspective_rays_kernel(rays, width, height, camera_pos, forward, right, up, fov_scale):
    """GPU kernel to generate perspective camera rays.

    Uses pinhole camera model: ray_dir = forward + u*right + v*up
    where u and v are in normalized device coordinates scaled by FOV.
    """
    px, py = cuda.grid(2)
    if px < width and py < height:
        # Convert pixel to normalized device coordinates (-1 to 1)
        aspect = width / height
        u = (2.0 * (px + 0.5) / width - 1.0) * aspect * fov_scale
        v = (1.0 - 2.0 * (py + 0.5) / height) * fov_scale

        # Compute ray direction
        dir_x = forward[0] + u * right[0] + v * up[0]
        dir_y = forward[1] + u * right[1] + v * up[1]
        dir_z = forward[2] + u * right[2] + v * up[2]

        # Normalize direction
        length = math.sqrt(dir_x * dir_x + dir_y * dir_y + dir_z * dir_z)
        dir_x /= length
        dir_y /= length
        dir_z /= length

        # Store ray (origin + direction)
        idx = py * width + px
        rays[idx, 0] = camera_pos[0]
        rays[idx, 1] = camera_pos[1]
        rays[idx, 2] = camera_pos[2]
        rays[idx, 3] = 1e-3  # t_min
        rays[idx, 4] = dir_x
        rays[idx, 5] = dir_y
        rays[idx, 6] = dir_z
        rays[idx, 7] = np.inf  # t_max


def _generate_perspective_rays(rays, width, height, camera_pos, forward, right, up, fov):
    """Generate perspective camera rays.

    Parameters
    ----------
    rays : cupy.ndarray
        Output array of shape (width*height, 8) for ray data.
    width : int
        Output image width.
    height : int
        Output image height.
    camera_pos : cupy.ndarray
        Camera position (3,).
    forward : cupy.ndarray
        Camera forward vector (3,).
    right : cupy.ndarray
        Camera right vector (3,).
    up : cupy.ndarray
        Camera up vector (3,).
    fov : float
        Vertical field of view in degrees.
    """
    fov_scale = math.tan(math.radians(fov) / 2.0)

    threadsperblock = (16, 16)
    blockspergrid_x = (width + threadsperblock[0] - 1) // threadsperblock[0]
    blockspergrid_y = (height + threadsperblock[1] - 1) // threadsperblock[1]
    blockspergrid = (blockspergrid_x, blockspergrid_y)

    _generate_perspective_rays_kernel[blockspergrid, threadsperblock](
        rays, width, height, camera_pos, forward, right, up, fov_scale
    )


@cuda.jit
def _generate_shadow_rays_from_hits_kernel(shadow_rays, primary_rays, hits, num_rays, sun_dir):
    """GPU kernel to generate shadow rays from primary hit points toward the sun."""
    idx = cuda.grid(1)
    if idx < num_rays:
        t = hits[idx, 0]

        if t > 0:
            # Get normal at hit point
            nx = hits[idx, 1]
            ny = hits[idx, 2]
            nz = hits[idx, 3]

            # Flip normal if facing away from ray
            ray_dx = primary_rays[idx, 4]
            ray_dy = primary_rays[idx, 5]
            ray_dz = primary_rays[idx, 6]

            dot_nd = nx * ray_dx + ny * ray_dy + nz * ray_dz
            if dot_nd > 0:
                nx = -nx
                ny = -ny
                nz = -nz

            # Compute hit point
            ox = primary_rays[idx, 0]
            oy = primary_rays[idx, 1]
            oz = primary_rays[idx, 2]

            hit_x = ox + t * ray_dx
            hit_y = oy + t * ray_dy
            hit_z = oz + t * ray_dz

            # Offset along normal to avoid self-intersection
            offset = 1e-3
            origin_x = hit_x + nx * offset
            origin_y = hit_y + ny * offset
            origin_z = hit_z + nz * offset

            shadow_rays[idx, 0] = origin_x
            shadow_rays[idx, 1] = origin_y
            shadow_rays[idx, 2] = origin_z
            shadow_rays[idx, 3] = 1e-3  # t_min
            shadow_rays[idx, 4] = sun_dir[0]
            shadow_rays[idx, 5] = sun_dir[1]
            shadow_rays[idx, 6] = sun_dir[2]
            shadow_rays[idx, 7] = np.inf  # t_max
        else:
            # No hit - shadow ray should not trace
            shadow_rays[idx, 0] = 0
            shadow_rays[idx, 1] = 0
            shadow_rays[idx, 2] = 0
            shadow_rays[idx, 3] = 0
            shadow_rays[idx, 4] = 0
            shadow_rays[idx, 5] = 0
            shadow_rays[idx, 6] = 1
            shadow_rays[idx, 7] = 0  # t_max = 0 means no trace


def _generate_shadow_rays_from_hits(shadow_rays, primary_rays, hits, num_rays, sun_dir):
    """Generate shadow rays from primary ray hit points toward the sun."""
    threadsperblock = 256
    blockspergrid = (num_rays + threadsperblock - 1) // threadsperblock

    _generate_shadow_rays_from_hits_kernel[blockspergrid, threadsperblock](
        shadow_rays, primary_rays, hits, num_rays, sun_dir
    )


@cuda.jit
def _shade_terrain_kernel(
    output, primary_rays, primary_hits, shadow_hits,
    elevation_data, color_lut, num_rays, width, height,
    sun_dir, ambient, cast_shadows,
    fog_density, fog_color_r, fog_color_g, fog_color_b,
    sky_color_r, sky_color_g, sky_color_b,
    elev_min, elev_range, alpha_channel,
    viewshed_data, viewshed_enabled, viewshed_opacity,
    observer_x, observer_y,
    pixel_spacing_x, pixel_spacing_y
):
    """GPU kernel for terrain shading with lighting, shadows, fog, colormapping, and viewshed."""
    idx = cuda.grid(1)
    if idx < num_rays:
        t = primary_hits[idx, 0]

        px = idx % width
        py = idx // width

        if t > 0:
            # Get normal
            nx = primary_hits[idx, 1]
            ny = primary_hits[idx, 2]
            nz = primary_hits[idx, 3]

            # Flip normal if back-facing
            ray_dx = primary_rays[idx, 4]
            ray_dy = primary_rays[idx, 5]
            ray_dz = primary_rays[idx, 6]

            dot_nd = nx * ray_dx + ny * ray_dy + nz * ray_dz
            if dot_nd > 0:
                nx = -nx
                ny = -ny
                nz = -nz

            # Compute hit point
            ox = primary_rays[idx, 0]
            oy = primary_rays[idx, 1]
            oz = primary_rays[idx, 2]

            hit_x = ox + t * ray_dx
            hit_y = oy + t * ray_dy
            hit_z = oz + t * ray_dz

            # Get elevation at hit point for color lookup
            # Convert world coords to pixel indices using pixel spacing
            elev_y = int(hit_y / pixel_spacing_y + 0.5)
            elev_x = int(hit_x / pixel_spacing_x + 0.5)

            elev_h = elevation_data.shape[0]
            elev_w = elevation_data.shape[1]

            if elev_y >= 0 and elev_y < elev_h and elev_x >= 0 and elev_x < elev_w:
                elevation = elevation_data[elev_y, elev_x]
            else:
                elevation = hit_z

            # Normalize elevation to [0, 1] for colormap lookup
            if elev_range > 0:
                elev_norm = (elevation - elev_min) / elev_range
            else:
                elev_norm = 0.5

            if elev_norm < 0:
                elev_norm = 0.0
            elif elev_norm > 1:
                elev_norm = 1.0

            # Color lookup
            lut_idx = int(elev_norm * 255)
            if lut_idx > 255:
                lut_idx = 255
            if lut_idx < 0:
                lut_idx = 0

            base_r = color_lut[lut_idx, 0]
            base_g = color_lut[lut_idx, 1]
            base_b = color_lut[lut_idx, 2]

            # Lambertian shading
            cos_theta = nx * sun_dir[0] + ny * sun_dir[1] + nz * sun_dir[2]
            if cos_theta < 0:
                cos_theta = 0.0

            # Shadow factor
            shadow_factor = 1.0
            if cast_shadows:
                shadow_t = shadow_hits[idx, 0]
                if shadow_t > 0:
                    shadow_factor = 0.5

            # Final lighting
            diffuse = cos_theta * shadow_factor
            lighting = ambient + (1.0 - ambient) * diffuse

            color_r = base_r * lighting
            color_g = base_g * lighting
            color_b = base_b * lighting

            # Apply viewshed overlay (teal glow on VISIBLE areas only)
            if viewshed_enabled:
                vs_h = viewshed_data.shape[0]
                vs_w = viewshed_data.shape[1]

                # Draw bright magenta orb at observer location (in world coords)
                dist_to_obs = math.sqrt((hit_x - observer_x) * (hit_x - observer_x) +
                                        (hit_y - observer_y) * (hit_y - observer_y))
                # Scale orb radius with pixel spacing for consistent visual size
                orb_radius = 2.0 * max(pixel_spacing_x, pixel_spacing_y)
                if dist_to_obs < orb_radius:
                    # Bright glowing magenta orb at observer
                    orb_intensity = 1.0 - (dist_to_obs / orb_radius)
                    orb_intensity = orb_intensity * orb_intensity  # Squared for sharper glow
                    color_r = color_r * (1.0 - orb_intensity) + 1.0 * orb_intensity
                    color_g = color_g * (1.0 - orb_intensity) + 0.0 * orb_intensity
                    color_b = color_b * (1.0 - orb_intensity) + 1.0 * orb_intensity
                elif elev_y >= 0 and elev_y < vs_h and elev_x >= 0 and elev_x < vs_w:
                    vis_val = viewshed_data[elev_y, elev_x]

                    # Apply teal glow only to VISIBLE areas
                    if not math.isnan(vis_val) and vis_val >= 0.0:
                        # Visible - teal/cyan glow
                        alpha = viewshed_opacity
                        color_r = color_r * (1.0 - alpha)
                        color_g = color_g * (1.0 - alpha) + 0.9 * alpha
                        color_b = color_b * (1.0 - alpha) + 0.85 * alpha
                    # else: invisible or NaN - keep original terrain color

            # Clamp
            if color_r > 1.0:
                color_r = 1.0
            if color_g > 1.0:
                color_g = 1.0
            if color_b > 1.0:
                color_b = 1.0

            # Fog
            if fog_density > 0:
                fog_amount = 1.0 - math.exp(-fog_density * t)
                color_r = color_r * (1 - fog_amount) + fog_color_r * fog_amount
                color_g = color_g * (1 - fog_amount) + fog_color_g * fog_amount
                color_b = color_b * (1 - fog_amount) + fog_color_b * fog_amount

            output[py, px, 0] = color_r
            output[py, px, 1] = color_g
            output[py, px, 2] = color_b
            if alpha_channel:
                output[py, px, 3] = 1.0
        else:
            # Miss - sky color
            output[py, px, 0] = sky_color_r
            output[py, px, 1] = sky_color_g
            output[py, px, 2] = sky_color_b
            if alpha_channel:
                output[py, px, 3] = 0.0


def _shade_terrain(
    output, primary_rays, primary_hits, shadow_hits,
    elevation_data, color_lut, num_rays, width, height,
    sun_dir, ambient, cast_shadows,
    fog_density, fog_color,
    elev_min, elev_range, alpha,
    viewshed_data=None, viewshed_opacity=0.6,
    observer_x=0.0, observer_y=0.0,
    pixel_spacing_x=1.0, pixel_spacing_y=1.0
):
    """Apply terrain shading with all effects."""
    threadsperblock = 256
    blockspergrid = (num_rays + threadsperblock - 1) // threadsperblock

    # Sky color (light blue)
    sky_color = (0.6, 0.75, 0.9)

    # Handle viewshed - need a placeholder if not provided
    viewshed_enabled = viewshed_data is not None
    if not viewshed_enabled:
        # Create dummy 1x1 array (won't be used)
        viewshed_data = cupy.zeros((1, 1), dtype=np.float32)

    _shade_terrain_kernel[blockspergrid, threadsperblock](
        output, primary_rays, primary_hits, shadow_hits,
        elevation_data, color_lut, num_rays, width, height,
        sun_dir, ambient, cast_shadows,
        fog_density, fog_color[0], fog_color[1], fog_color[2],
        sky_color[0], sky_color[1], sky_color[2],
        elev_min, elev_range, alpha,
        viewshed_data, viewshed_enabled, viewshed_opacity,
        observer_x, observer_y,
        pixel_spacing_x, pixel_spacing_y
    )


def _save_image(output, output_path):
    """Save the rendered image to a file.

    Parameters
    ----------
    output : np.ndarray
        Image array of shape (H, W, 3) or (H, W, 4) with values [0-1].
    output_path : str
        Path to save the image (supports PNG, TIFF, JPEG, etc.).
    """
    Image = _lazy_import_pil()

    # Convert to uint8
    img_data = (np.clip(output, 0, 1) * 255).astype(np.uint8)

    if output.shape[2] == 4:
        img = Image.fromarray(img_data, mode='RGBA')
    else:
        img = Image.fromarray(img_data, mode='RGB')

    img.save(output_path)


def render(
    raster,
    camera_position: Tuple[float, float, float],
    look_at: Tuple[float, float, float],
    fov: float = 60.0,
    up: Tuple[float, float, float] = (0, 0, 1),
    width: int = 1920,
    height: int = 1080,
    sun_azimuth: float = 225,
    sun_altitude: float = 45,
    shadows: bool = True,
    ambient: float = 0.15,
    fog_density: float = 0.0,
    fog_color: Tuple[float, float, float] = (0.7, 0.8, 0.9),
    colormap: str = 'terrain',
    color_range: Optional[Tuple[float, float]] = None,
    output_path: Optional[str] = None,
    alpha: bool = False,
    vertical_exaggeration: Optional[float] = None,
    rtx: RTX = None,
    viewshed_data=None,
    viewshed_opacity: float = 0.6,
    observer_position: Optional[Tuple[float, float]] = None,
    pixel_spacing_x: float = 1.0,
    pixel_spacing_y: float = 1.0,
) -> np.ndarray:
    """Render terrain with a perspective camera for movie-quality visualization.

    Uses OptiX ray tracing to render terrain with realistic lighting, shadows,
    atmospheric effects, and elevation-based coloring.

    Parameters
    ----------
    raster : xarray.DataArray
        2D raster terrain data with 'x' and 'y' coordinates.
        Data should be a cupy array on the GPU for best performance.
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
        Existing RTX instance to reuse. If None, a new instance is created.

    Returns
    -------
    np.ndarray
        Rendered image of shape (height, width, 3) or (height, width, 4)
        as float32 with values [0-1].

    Examples
    --------
    >>> import rtxpy
    >>> import xarray as xr
    >>> dem = xr.open_dataarray('dem.tif')
    >>> dem = dem.rtx.to_cupy()
    >>> img = dem.rtx.render(
    ...     camera_position=(W/2, -50, elev_max + 200),
    ...     look_at=(W/2, H/2, elev_mean),
    ...     shadows=True,
    ...     output_path='terrain_render.png'
    ... )
    """
    xr = _lazy_import_xarray()

    if not has_cupy:
        raise ImportError(
            "cupy is required for render. "
            "Install it with: conda install -c conda-forge cupy"
        )

    if not isinstance(raster.data, cupy.ndarray):
        import warnings
        warnings.warn(
            "raster.data is not a cupy array. "
            "Additional overhead will be incurred from CPU-GPU transfers."
        )
        elevation_data = cupy.asarray(raster.data)
    else:
        elevation_data = raster.data

    H, W = raster.shape

    # Compute vertical exaggeration if not specified
    # Goal: make the terrain relief roughly proportional to the horizontal extent
    elev_min_orig = float(cupy.nanmin(elevation_data))
    elev_max_orig = float(cupy.nanmax(elevation_data))
    elev_range_orig = elev_max_orig - elev_min_orig

    if vertical_exaggeration is None:
        # Auto-compute: scale so relief is ~20% of horizontal extent
        horizontal_extent = max(H, W)
        if elev_range_orig > 0:
            vertical_exaggeration = (horizontal_extent * 0.2) / elev_range_orig
        else:
            vertical_exaggeration = 1.0

    # If RTX has multi-GAS content (meshes placed via add_geometry),
    # use it directly without calling prepare_mesh which would rebuild as single-GAS.
    # The meshes were already placed with correct coordinates, so we use them as-is.
    # Also disable vertical exaggeration since the scene is already built.
    if rtx is not None and rtx.get_geometry_count() > 0:
        optix = rtx
        scaled_raster = raster
        vertical_exaggeration = 1.0  # Don't scale camera for pre-built scenes
    elif vertical_exaggeration != 1.0:
        # Scale elevation data for mesh building
        scaled_elevation = elevation_data * vertical_exaggeration
        # Create a temporary raster with scaled elevations
        scaled_raster = raster.copy(data=scaled_elevation)
        # Don't reuse rtx when scaling - need fresh mesh
        optix = prepare_mesh(scaled_raster, rtx=None)
    else:
        scaled_raster = raster
        optix = prepare_mesh(raster, rtx)

    # Scale camera position and look_at z coordinates
    scaled_camera_position = (
        camera_position[0],
        camera_position[1],
        camera_position[2] * vertical_exaggeration
    )
    scaled_look_at = (
        look_at[0],
        look_at[1],
        look_at[2] * vertical_exaggeration
    )

    num_rays = width * height

    # Compute camera basis vectors using scaled positions
    forward, right, cam_up = _compute_camera_basis(scaled_camera_position, scaled_look_at, up)

    # Upload camera vectors to GPU
    d_camera_pos = cupy.array(scaled_camera_position, dtype=np.float32)
    d_forward = cupy.array(forward, dtype=np.float32)
    d_right = cupy.array(right, dtype=np.float32)
    d_up = cupy.array(cam_up, dtype=np.float32)

    # Sun direction
    sun_dir = get_sun_dir(sun_altitude, sun_azimuth)
    d_sun_dir = cupy.array(sun_dir, dtype=np.float32)

    # Color lookup table
    color_lut = _get_colormap_lut(colormap)
    d_color_lut = cupy.array(color_lut, dtype=np.float32)

    # Elevation range for colormap
    if color_range is not None:
        elev_min, elev_max = color_range
    else:
        elev_min = float(cupy.nanmin(elevation_data))
        elev_max = float(cupy.nanmax(elevation_data))
    elev_range = elev_max - elev_min

    # Allocate buffers
    d_primary_rays = cupy.empty((num_rays, 8), dtype=np.float32)
    d_primary_hits = cupy.empty((num_rays, 4), dtype=np.float32)
    d_shadow_rays = cupy.empty((num_rays, 8), dtype=np.float32)
    d_shadow_hits = cupy.empty((num_rays, 4), dtype=np.float32)

    num_channels = 4 if alpha else 3
    d_output = cupy.zeros((height, width, num_channels), dtype=np.float32)

    device = cupy.cuda.Device(0)

    # Step 1: Generate perspective rays
    _generate_perspective_rays(
        d_primary_rays, width, height,
        d_camera_pos, d_forward, d_right, d_up, fov
    )
    device.synchronize()

    # Step 2: Trace primary rays
    optix.trace(d_primary_rays, d_primary_hits, num_rays)

    # Step 3: Generate and trace shadow rays (if enabled)
    if shadows:
        _generate_shadow_rays_from_hits(
            d_shadow_rays, d_primary_rays, d_primary_hits, num_rays, d_sun_dir
        )
        device.synchronize()
        optix.trace(d_shadow_rays, d_shadow_hits, num_rays)
    else:
        # Fill shadow hits with -1 (no shadow)
        d_shadow_hits.fill(-1)

    # Prepare viewshed data if provided
    d_viewshed = None
    if viewshed_data is not None:
        if hasattr(viewshed_data, 'data'):
            # It's an xarray DataArray
            vs_data = viewshed_data.data
        else:
            vs_data = viewshed_data
        if not isinstance(vs_data, cupy.ndarray):
            d_viewshed = cupy.asarray(vs_data, dtype=np.float32)
        else:
            d_viewshed = vs_data.astype(np.float32)


    # Get observer position for viewshed marker
    obs_x = float(observer_position[0]) if observer_position else 0.0
    obs_y = float(observer_position[1]) if observer_position else 0.0

    # Step 4: Shade terrain
    _shade_terrain(
        d_output, d_primary_rays, d_primary_hits, d_shadow_hits,
        elevation_data, d_color_lut, num_rays, width, height,
        d_sun_dir, ambient, shadows,
        fog_density, fog_color,
        elev_min, elev_range, alpha,
        d_viewshed, viewshed_opacity,
        obs_x, obs_y,
        pixel_spacing_x, pixel_spacing_y
    )
    device.synchronize()

    # Transfer to CPU
    output = cupy.asnumpy(d_output)

    # Save image if requested
    if output_path is not None:
        _save_image(output, output_path)

    # Clean up
    del d_primary_rays, d_primary_hits, d_shadow_rays, d_shadow_hits
    del d_output, d_camera_pos, d_forward, d_right, d_up, d_sun_dir, d_color_lut
    cupy.get_default_memory_pool().free_all_blocks()

    return output
