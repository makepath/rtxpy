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

    right = np.cross(world_up, forward)
    right_norm = np.linalg.norm(right)

    # Handle case where forward is parallel to up vector
    if right_norm < 1e-6:
        # Use a different up vector to compute right
        alt_up = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        right = np.cross(alt_up, forward)
        right_norm = np.linalg.norm(right)
        if right_norm < 1e-6:
            alt_up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
            right = np.cross(alt_up, forward)
            right_norm = np.linalg.norm(right)

    right = right / right_norm

    cam_up = np.cross(forward, right)
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


@cuda.jit(device=True)
def _compute_physical_sky(ray_dx, ray_dy, ray_dz, sun_dir):
    """Compute physical sky color from a ray direction and sun position.

    Returns (r, g, b) tuple in linear HDR space (may exceed 1.0 near sun).
    """
    # Elevation angle of ray (0 = horizon, 1 = zenith, negative = below)
    ray_elev = ray_dz  # z component of unit direction = sin(elevation)
    if ray_elev < 0.0:
        ray_elev = 0.0

    # Zenith -> horizon interpolation with quadratic falloff
    horizon_blend = 1.0 - ray_elev
    horizon_blend = horizon_blend * horizon_blend

    # Sun glow near the sun direction
    sun_dot = ray_dx * sun_dir[0] + ray_dy * sun_dir[1] + ray_dz * sun_dir[2]
    if sun_dot < 0.0:
        sun_dot = 0.0

    # Broad halo around sun
    sun_glow = sun_dot * sun_dot
    sun_glow = sun_glow * sun_glow  # ^4
    sun_glow = sun_glow * 0.4

    # Sun altitude affects brightness and warmth
    sun_elev = sun_dir[2]
    if sun_elev < 0.0:
        sun_elev = 0.0
    brightness = 0.5 + 0.5 * sun_elev

    # Zenith: deep blue
    zen_r = 0.15 * brightness
    zen_g = 0.25 * brightness
    zen_b = 0.55 * brightness

    # Horizon: pale warm white (warmer at low sun angles)
    warmth = 1.0 - sun_elev
    hor_r = (0.70 + 0.20 * warmth) * brightness
    hor_g = (0.75 + 0.05 * warmth) * brightness
    hor_b = (0.85 - 0.15 * warmth) * brightness

    # Blend zenith -> horizon + sun glow
    sr = zen_r * (1.0 - horizon_blend) + hor_r * horizon_blend + sun_glow * 1.0
    sg = zen_g * (1.0 - horizon_blend) + hor_g * horizon_blend + sun_glow * 0.9
    sb = zen_b * (1.0 - horizon_blend) + hor_b * horizon_blend + sun_glow * 0.6

    return sr, sg, sb


@cuda.jit
def _generate_perspective_rays_kernel(rays, width, height, camera_pos, forward, right, up,
                                       fov_scale, jitter_seed, aperture, focal_distance):
    """GPU kernel to generate perspective camera rays.

    Uses pinhole camera model: ray_dir = forward + u*right + v*up
    where u and v are in normalized device coordinates scaled by FOV.
    When jitter_seed > 0, adds sub-pixel random offset for anti-aliasing.
    When aperture > 0, applies thin-lens depth-of-field (requires jitter_seed > 0).
    """
    px, py = cuda.grid(2)
    if px < width and py < height:
        idx = py * width + px
        aspect = width / height

        if jitter_seed > 0:
            # Hash-based RNG (same pattern as AO kernel)
            h = np.uint32(idx * np.uint32(1337) + jitter_seed)
            h = (h ^ (h >> np.uint32(16))) * np.uint32(2654435761)
            h = (h ^ (h >> np.uint32(16))) * np.uint32(2246822519)
            h = h ^ (h >> np.uint32(16))
            jx = float(h & np.uint32(0xFFFF)) / 65535.0 - 0.5  # [-0.5, 0.5]
            h = (h * np.uint32(1103515245) + np.uint32(12345))
            h = h ^ (h >> np.uint32(16))
            jy = float(h & np.uint32(0xFFFF)) / 65535.0 - 0.5
        else:
            jx = 0.0
            jy = 0.0

        u = (2.0 * (px + 0.5 + jx) / width - 1.0) * aspect * fov_scale
        v = (1.0 - 2.0 * (py + 0.5 + jy) / height) * fov_scale

        # Compute ray direction (unnormalized)
        dir_x = forward[0] + u * right[0] + v * up[0]
        dir_y = forward[1] + u * right[1] + v * up[1]
        dir_z = forward[2] + u * right[2] + v * up[2]

        # Origin defaults to camera position
        ox = camera_pos[0]
        oy = camera_pos[1]
        oz = camera_pos[2]

        # Thin-lens depth of field
        if aperture > 0.0 and jitter_seed > 0:
            # Focal point on the focal plane (perpendicular to forward)
            fp_x = camera_pos[0] + focal_distance * dir_x
            fp_y = camera_pos[1] + focal_distance * dir_y
            fp_z = camera_pos[2] + focal_distance * dir_z

            # Two more random numbers for lens disk sampling
            h = (h * np.uint32(1103515245) + np.uint32(12345))
            h = h ^ (h >> np.uint32(16))
            lr1 = float(h & np.uint32(0xFFFF)) / 65535.0
            h = (h * np.uint32(1103515245) + np.uint32(12345))
            h = h ^ (h >> np.uint32(16))
            lr2 = float(h & np.uint32(0xFFFF)) / 65535.0

            # Uniform disk sampling
            lens_r = aperture * math.sqrt(lr1)
            lens_phi = 2.0 * math.pi * lr2
            lens_dx = lens_r * math.cos(lens_phi)
            lens_dy = lens_r * math.sin(lens_phi)

            # Offset origin on lens disk (in camera right/up plane)
            ox += lens_dx * right[0] + lens_dy * up[0]
            oy += lens_dx * right[1] + lens_dy * up[1]
            oz += lens_dx * right[2] + lens_dy * up[2]

            # New direction: from offset origin to focal point
            dir_x = fp_x - ox
            dir_y = fp_y - oy
            dir_z = fp_z - oz

        # Normalize direction
        length = math.sqrt(dir_x * dir_x + dir_y * dir_y + dir_z * dir_z)
        dir_x /= length
        dir_y /= length
        dir_z /= length

        # Store ray (origin + direction)
        rays[idx, 0] = ox
        rays[idx, 1] = oy
        rays[idx, 2] = oz
        rays[idx, 3] = 1e-3  # t_min
        rays[idx, 4] = dir_x
        rays[idx, 5] = dir_y
        rays[idx, 6] = dir_z
        rays[idx, 7] = np.inf  # t_max


def _generate_perspective_rays(rays, width, height, camera_pos, forward, right, up, fov,
                               jitter_seed=np.uint32(0), aperture=0.0, focal_distance=0.0):
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
    jitter_seed : np.uint32, optional
        When > 0, adds sub-pixel jitter for anti-aliasing. Default is 0 (no jitter).
    aperture : float, optional
        Lens aperture radius for depth of field. 0 disables DOF. Default is 0.0.
    focal_distance : float, optional
        Distance to the focal plane. Objects at this distance are sharp.
        Default is 0.0 (no DOF).
    """
    fov_scale = math.tan(math.radians(fov) / 2.0)

    threadsperblock = (16, 16)
    blockspergrid_x = (width + threadsperblock[0] - 1) // threadsperblock[0]
    blockspergrid_y = (height + threadsperblock[1] - 1) // threadsperblock[1]
    blockspergrid = (blockspergrid_x, blockspergrid_y)

    _generate_perspective_rays_kernel[blockspergrid, threadsperblock](
        rays, width, height, camera_pos, forward, right, up, fov_scale,
        jitter_seed, np.float32(aperture), np.float32(focal_distance)
    )


@cuda.jit
def _generate_shadow_rays_from_hits_kernel(shadow_rays, primary_rays, hits, num_rays, sun_dir,
                                            sun_angle_rad, shadow_seed):
    """GPU kernel to generate shadow rays from primary hit points toward the sun.

    When shadow_seed > 0 and sun_angle_rad > 0, jitters the shadow ray direction
    within a cone around sun_dir for soft shadows (finite-size light source).
    """
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

            # Compute shadow direction (possibly jittered for soft shadows)
            if shadow_seed > 0 and sun_angle_rad > 0.0:
                # Hash-based RNG (same pattern as AO kernel)
                h = np.uint32(idx * np.uint32(2719) + shadow_seed)
                h = (h ^ (h >> np.uint32(16))) * np.uint32(2654435761)
                h = (h ^ (h >> np.uint32(16))) * np.uint32(2246822519)
                h = h ^ (h >> np.uint32(16))
                r1 = float(h & np.uint32(0xFFFF)) / 65535.0
                h = (h * np.uint32(1103515245) + np.uint32(12345))
                h = h ^ (h >> np.uint32(16))
                r2 = float(h & np.uint32(0xFFFF)) / 65535.0

                # Uniform disk -> cone deflection
                cone_r = sun_angle_rad * math.sqrt(r1)
                cone_phi = 2.0 * math.pi * r2
                dx_local = cone_r * math.cos(cone_phi)
                dy_local = cone_r * math.sin(cone_phi)

                # Build tangent frame from sun_dir
                sx = sun_dir[0]
                sy = sun_dir[1]
                sz = sun_dir[2]
                if abs(sx) < 0.9:
                    tx = 0.0
                    ty = -sz
                    tz = sy
                else:
                    tx = sz
                    ty = 0.0
                    tz = -sx
                t_len = math.sqrt(tx * tx + ty * ty + tz * tz)
                if t_len > 1e-8:
                    tx /= t_len
                    ty /= t_len
                    tz /= t_len
                bx = sy * tz - sz * ty
                by = sz * tx - sx * tz
                bz = sx * ty - sy * tx

                # Perturbed direction
                sdx = sx + dx_local * tx + dy_local * bx
                sdy = sy + dx_local * ty + dy_local * by
                sdz = sz + dx_local * tz + dy_local * bz
                s_len = math.sqrt(sdx * sdx + sdy * sdy + sdz * sdz)
                sdx /= s_len
                sdy /= s_len
                sdz /= s_len
            else:
                sdx = sun_dir[0]
                sdy = sun_dir[1]
                sdz = sun_dir[2]

            shadow_rays[idx, 0] = origin_x
            shadow_rays[idx, 1] = origin_y
            shadow_rays[idx, 2] = origin_z
            shadow_rays[idx, 3] = 1e-3  # t_min
            shadow_rays[idx, 4] = sdx
            shadow_rays[idx, 5] = sdy
            shadow_rays[idx, 6] = sdz
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


def _generate_shadow_rays_from_hits(shadow_rays, primary_rays, hits, num_rays, sun_dir,
                                     sun_angle_rad=0.0, shadow_seed=np.uint32(0)):
    """Generate shadow rays from primary ray hit points toward the sun."""
    threadsperblock = 256
    blockspergrid = (num_rays + threadsperblock - 1) // threadsperblock

    _generate_shadow_rays_from_hits_kernel[blockspergrid, threadsperblock](
        shadow_rays, primary_rays, hits, num_rays, sun_dir, sun_angle_rad, shadow_seed
    )


@cuda.jit
def _generate_ao_rays_kernel(ao_rays, primary_rays, hits, num_rays, ao_radius, seed):
    """GPU kernel to generate ambient occlusion rays from primary hit points.

    For each pixel with a hit, generates a cosine-weighted random direction
    on the hemisphere around the surface normal, with tmax limited to ao_radius.
    Uses a simple hash-based RNG seeded per pixel.
    """
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

            # Hash-based RNG: two uniform randoms from pixel index + seed
            h = np.uint32(idx * np.uint32(1337) + seed)
            h = (h ^ (h >> np.uint32(16))) * np.uint32(2654435761)
            h = (h ^ (h >> np.uint32(16))) * np.uint32(2246822519)
            h = h ^ (h >> np.uint32(16))
            r1 = float(h & np.uint32(0xFFFF)) / 65535.0

            h = (h * np.uint32(1103515245) + np.uint32(12345))
            h = h ^ (h >> np.uint32(16))
            r2 = float(h & np.uint32(0xFFFF)) / 65535.0

            # Cosine-weighted hemisphere sample in local coords
            # r1 = cos^2(theta), so cos_theta = sqrt(r1)
            cos_theta = math.sqrt(r1)
            sin_theta = math.sqrt(1.0 - r1)
            phi = 2.0 * math.pi * r2
            local_x = sin_theta * math.cos(phi)
            local_y = sin_theta * math.sin(phi)
            local_z = cos_theta

            # Build tangent frame from normal
            # Choose a vector not parallel to normal
            if abs(nx) < 0.9:
                tx = 0.0
                ty = -nz
                tz = ny
            else:
                tx = nz
                ty = 0.0
                tz = -nx
            # Normalize tangent
            t_len = math.sqrt(tx * tx + ty * ty + tz * tz)
            if t_len > 1e-8:
                tx /= t_len
                ty /= t_len
                tz /= t_len

            # Bitangent = normal x tangent
            bx = ny * tz - nz * ty
            by = nz * tx - nx * tz
            bz = nx * ty - ny * tx

            # Transform local -> world
            dir_x = local_x * tx + local_y * bx + local_z * nx
            dir_y = local_x * ty + local_y * by + local_z * ny
            dir_z = local_x * tz + local_y * bz + local_z * nz

            # Normalize direction (should be unit already but be safe)
            d_len = math.sqrt(dir_x * dir_x + dir_y * dir_y + dir_z * dir_z)
            if d_len > 1e-8:
                dir_x /= d_len
                dir_y /= d_len
                dir_z /= d_len

            ao_rays[idx, 0] = origin_x
            ao_rays[idx, 1] = origin_y
            ao_rays[idx, 2] = origin_z
            ao_rays[idx, 3] = 1e-3  # t_min
            ao_rays[idx, 4] = dir_x
            ao_rays[idx, 5] = dir_y
            ao_rays[idx, 6] = dir_z
            ao_rays[idx, 7] = ao_radius  # t_max
        else:
            # No hit - AO ray should not trace
            ao_rays[idx, 0] = 0
            ao_rays[idx, 1] = 0
            ao_rays[idx, 2] = 0
            ao_rays[idx, 3] = 0
            ao_rays[idx, 4] = 0
            ao_rays[idx, 5] = 0
            ao_rays[idx, 6] = 1
            ao_rays[idx, 7] = 0  # t_max = 0 means no trace


def _generate_ao_rays(ao_rays, primary_rays, hits, num_rays, ao_radius, seed):
    """Generate ambient occlusion rays from primary ray hit points."""
    threadsperblock = 256
    blockspergrid = (num_rays + threadsperblock - 1) // threadsperblock

    _generate_ao_rays_kernel[blockspergrid, threadsperblock](
        ao_rays, primary_rays, hits, num_rays, ao_radius, np.uint32(seed)
    )


@cuda.jit
def _accumulate_ao_kernel(ao_factor, ao_hits, num_rays, ao_samples):
    """GPU kernel to accumulate AO results: subtract 1/ao_samples for each hit."""
    idx = cuda.grid(1)
    if idx < num_rays:
        t = ao_hits[idx, 0]
        if t > 0:
            ao_factor[idx] -= 1.0 / ao_samples


def _accumulate_ao(ao_factor, ao_hits, num_rays, ao_samples):
    """Accumulate AO hit results into the ao_factor buffer."""
    threadsperblock = 256
    blockspergrid = (num_rays + threadsperblock - 1) // threadsperblock

    _accumulate_ao_kernel[blockspergrid, threadsperblock](
        ao_factor, ao_hits, num_rays, ao_samples
    )


@cuda.jit
def _accumulate_gi_kernel(gi_color, ao_rays, ao_hits, num_rays, ao_samples,
                          vertical_exaggeration, elev_min, elev_range,
                          color_lut, color_stretch, gi_throughput):
    """GPU kernel to accumulate multi-bounce diffuse GI from AO hit points.

    For each AO ray that hit a surface, looks up the surface color at the hit
    point (via elevation -> colormap LUT) and accumulates it into gi_color,
    weighted by the current path throughput. Then updates throughput by
    multiplying with the hit surface albedo (Lambertian BRDF).

    Bounce 0 with throughput=(1,1,1) is identical to single-bounce GI.
    Subsequent bounces are naturally attenuated by accumulated albedo product.
    """
    idx = cuda.grid(1)
    if idx < num_rays:
        t = ao_hits[idx, 0]
        if t > 0:
            # Hit position Z -> elevation
            hit_z = ao_rays[idx, 2] + t * ao_rays[idx, 6]
            elevation = hit_z / vertical_exaggeration

            # Normalize elevation to [0, 1] for colormap lookup
            if elev_range > 0:
                elev_norm = (elevation - elev_min) / elev_range
            else:
                elev_norm = 0.5

            if elev_norm < 0.0:
                elev_norm = 0.0
            elif elev_norm > 1.0:
                elev_norm = 1.0

            # Apply nonlinear stretch: 0=linear, 1=cbrt, 2=log, 3=sqrt
            if color_stretch == 1:
                elev_norm = math.pow(elev_norm, 1.0 / 3.0)
            elif color_stretch == 2:
                elev_norm = math.log(1.0 + elev_norm * 9.0) / math.log(10.0)
            elif color_stretch == 3:
                elev_norm = math.sqrt(elev_norm)

            # Color lookup
            lut_idx = int(elev_norm * 255)
            if lut_idx > 255:
                lut_idx = 255
            if lut_idx < 0:
                lut_idx = 0

            hit_r = color_lut[lut_idx, 0]
            hit_g = color_lut[lut_idx, 1]
            hit_b = color_lut[lut_idx, 2]

            # Accumulate weighted by path throughput
            gi_color[idx, 0] += gi_throughput[idx, 0] * hit_r / ao_samples
            gi_color[idx, 1] += gi_throughput[idx, 1] * hit_g / ao_samples
            gi_color[idx, 2] += gi_throughput[idx, 2] * hit_b / ao_samples

            # Update throughput: Lambertian BRDF = albedo/pi,
            # cosine-weighted pdf = cos/pi -> cancel to albedo
            gi_throughput[idx, 0] *= hit_r
            gi_throughput[idx, 1] *= hit_g
            gi_throughput[idx, 2] *= hit_b


def _accumulate_gi(gi_color, ao_rays, ao_hits, num_rays, ao_samples,
                   vertical_exaggeration, elev_min, elev_range,
                   color_lut, color_stretch, gi_throughput):
    """Accumulate multi-bounce diffuse GI from AO hit points."""
    threadsperblock = 256
    blockspergrid = (num_rays + threadsperblock - 1) // threadsperblock

    _accumulate_gi_kernel[blockspergrid, threadsperblock](
        gi_color, ao_rays, ao_hits, num_rays, ao_samples,
        np.float32(vertical_exaggeration), np.float32(elev_min),
        np.float32(elev_range), color_lut, np.int32(color_stretch),
        gi_throughput
    )


@cuda.jit
def _generate_reflection_rays_kernel(reflection_rays, primary_rays, primary_hits,
                                      instance_ids, geometry_colors, num_rays,
                                      elevation_data, pixel_spacing_x, pixel_spacing_y):
    """GPU kernel to generate reflection rays for water surfaces.

    For water pixels (geometry_colors alpha >= 2.0) and NaN ocean terrain,
    computes mirror reflection direction R = D - 2(D·N)N from the primary hit
    point. Non-water pixels get t_max = 0 (no trace).
    """
    idx = cuda.grid(1)
    if idx < num_rays:
        t = primary_hits[idx, 0]

        # Default: no trace
        is_water = False
        is_ocean = False
        if t > 0:
            inst_id = instance_ids[idx]
            if inst_id >= 0 and inst_id < geometry_colors.shape[0]:
                gc_alpha = geometry_colors[inst_id, 3]
                if gc_alpha >= 2.0 and gc_alpha < 5.0:
                    is_water = True

            # Check if terrain hit is NaN ocean
            if not is_water:
                ray_dx = primary_rays[idx, 4]
                ray_dy = primary_rays[idx, 5]
                ray_dz = primary_rays[idx, 6]
                ox = primary_rays[idx, 0]
                oy = primary_rays[idx, 1]
                hit_x = ox + t * ray_dx
                hit_y = oy + t * ray_dy
                ey = int(hit_y / pixel_spacing_y + 0.5)
                ex = int(hit_x / pixel_spacing_x + 0.5)
                if 0 <= ey < elevation_data.shape[0] and 0 <= ex < elevation_data.shape[1]:
                    if math.isnan(elevation_data[ey, ex]):
                        is_water = True
                        is_ocean = True

        if is_water:
            # Get normal at hit point
            nx = primary_hits[idx, 1]
            ny = primary_hits[idx, 2]
            nz = primary_hits[idx, 3]

            # Ocean: force flat water normal
            if is_ocean:
                nx = 0.0
                ny = 0.0
                nz = 1.0

            # Flip normal if facing away from ray
            ray_dx = primary_rays[idx, 4]
            ray_dy = primary_rays[idx, 5]
            ray_dz = primary_rays[idx, 6]

            dot_nd = nx * ray_dx + ny * ray_dy + nz * ray_dz
            if dot_nd > 0:
                nx = -nx
                ny = -ny
                nz = -nz
                dot_nd = -dot_nd

            # Compute hit point
            ox = primary_rays[idx, 0]
            oy = primary_rays[idx, 1]
            oz = primary_rays[idx, 2]

            hit_x = ox + t * ray_dx
            hit_y = oy + t * ray_dy
            hit_z = oz + t * ray_dz

            # Reflection direction: R = D - 2(D·N)N
            ref_dx = ray_dx - 2.0 * dot_nd * nx
            ref_dy = ray_dy - 2.0 * dot_nd * ny
            ref_dz = ray_dz - 2.0 * dot_nd * nz

            # Normalize
            r_len = math.sqrt(ref_dx * ref_dx + ref_dy * ref_dy + ref_dz * ref_dz)
            if r_len > 1e-8:
                ref_dx /= r_len
                ref_dy /= r_len
                ref_dz /= r_len

            # Offset origin along normal to avoid self-intersection
            offset = 1e-2
            reflection_rays[idx, 0] = hit_x + nx * offset
            reflection_rays[idx, 1] = hit_y + ny * offset
            reflection_rays[idx, 2] = hit_z + nz * offset
            reflection_rays[idx, 3] = 1e-3  # t_min
            reflection_rays[idx, 4] = ref_dx
            reflection_rays[idx, 5] = ref_dy
            reflection_rays[idx, 6] = ref_dz
            reflection_rays[idx, 7] = np.inf  # t_max
        else:
            # Not water — no trace needed
            reflection_rays[idx, 0] = 0.0
            reflection_rays[idx, 1] = 0.0
            reflection_rays[idx, 2] = 0.0
            reflection_rays[idx, 3] = 0.0
            reflection_rays[idx, 4] = 0.0
            reflection_rays[idx, 5] = 0.0
            reflection_rays[idx, 6] = 1.0
            reflection_rays[idx, 7] = 0.0  # t_max = 0 -> no trace


def _generate_reflection_rays(reflection_rays, primary_rays, primary_hits,
                               instance_ids, geometry_colors, num_rays,
                               elevation_data, pixel_spacing_x, pixel_spacing_y):
    """Generate reflection rays for water surfaces and NaN ocean terrain."""
    threadsperblock = 256
    blockspergrid = (num_rays + threadsperblock - 1) // threadsperblock
    _generate_reflection_rays_kernel[blockspergrid, threadsperblock](
        reflection_rays, primary_rays, primary_hits,
        instance_ids, geometry_colors, num_rays,
        elevation_data, np.float32(pixel_spacing_x), np.float32(pixel_spacing_y)
    )


@cuda.jit(device=True, inline='always')
def _cloud_hash_i(n):
    """Integer hash for procedural noise (works under Numba CUDA)."""
    n = (n << 13) ^ n
    return (n * (n * n * 15731 + 789221) + 1376312589) & 0x7fffffff


@cuda.jit(device=True, inline='always')
def _cloud_hash_3(a, b, c):
    """Hash 3 integer coords to a float in [0, 1)."""
    return _cloud_hash_i(a + _cloud_hash_i(b + _cloud_hash_i(c))) / 2147483647.0


@cuda.jit(device=True, inline='always')
def _cloud_noise_3d(x, y, z):
    """3D value noise via integer hashing + trilinear interpolation."""
    ix = int(math.floor(x))
    iy = int(math.floor(y))
    iz = int(math.floor(z))
    fx = x - ix
    fy = y - iy
    fz = z - iz
    # Smoothstep
    ux = fx * fx * (3.0 - 2.0 * fx)
    uy = fy * fy * (3.0 - 2.0 * fy)
    uz = fz * fz * (3.0 - 2.0 * fz)
    # Hash 8 corners
    n000 = _cloud_hash_3(ix, iy, iz)
    n100 = _cloud_hash_3(ix + 1, iy, iz)
    n010 = _cloud_hash_3(ix, iy + 1, iz)
    n110 = _cloud_hash_3(ix + 1, iy + 1, iz)
    n001 = _cloud_hash_3(ix, iy, iz + 1)
    n101 = _cloud_hash_3(ix + 1, iy, iz + 1)
    n011 = _cloud_hash_3(ix, iy + 1, iz + 1)
    n111 = _cloud_hash_3(ix + 1, iy + 1, iz + 1)
    # Trilinear
    nx00 = n000 * (1.0 - ux) + n100 * ux
    nx10 = n010 * (1.0 - ux) + n110 * ux
    nx01 = n001 * (1.0 - ux) + n101 * ux
    nx11 = n011 * (1.0 - ux) + n111 * ux
    nxy0 = nx00 * (1.0 - uy) + nx10 * uy
    nxy1 = nx01 * (1.0 - uy) + nx11 * uy
    return nxy0 * (1.0 - uz) + nxy1 * uz


@cuda.jit(device=True, inline='always')
def _cloud_fbm_3d(x, y, z, octaves):
    """Fractal Brownian motion: sum of noise at increasing frequencies."""
    value = 0.0
    amplitude = 0.5
    freq = 1.0
    for _ in range(octaves):
        value += amplitude * _cloud_noise_3d(x * freq, y * freq, z * freq)
        freq *= 2.0
        amplitude *= 0.5
    return value


@cuda.jit
def _volumetric_cloud_kernel(
    output, primary_rays, primary_hits,
    cloud_cover_grid,
    sun_dir,
    width, height,
    cloud_base_z, cloud_top_z,
    pixel_spacing_x, pixel_spacing_y,
    grid_rows, grid_cols,
    time_offset,
    extinction,
    terrain_rows, terrain_cols,
):
    """Ray-march volumetric clouds and composite over shaded frame."""
    idx = cuda.grid(1)
    if idx >= width * height:
        return

    py = idx // width
    px = idx % width

    # Read ray origin and direction
    ox = primary_rays[idx, 0]
    oy = primary_rays[idx, 1]
    oz = primary_rays[idx, 2]
    dx = primary_rays[idx, 4]
    dy = primary_rays[idx, 5]
    dz = primary_rays[idx, 6]

    # Hit distance (terrain/geometry); 0 or NaN means miss (sky)
    hit_t = primary_hits[idx, 0]
    # NaN-safe: treat NaN as miss (no terrain to occlude clouds)
    if not (hit_t > 0.0):
        hit_t = 0.0

    # Ray-slab intersection: cloud_base_z to cloud_top_z (horizontal slab)
    if abs(dz) < 1e-8:
        return  # Ray parallel to slab — skip

    t_base = (cloud_base_z - oz) / dz
    t_top = (cloud_top_z - oz) / dz
    entry_t = min(t_base, t_top)
    exit_t = max(t_base, t_top)

    # Clamp to valid range
    if entry_t < 0.0:
        entry_t = 0.0
    # Clouds behind terrain are occluded
    if hit_t > 0.0 and exit_t > hit_t:
        exit_t = hit_t
    if entry_t >= exit_t:
        return

    slab_thickness = cloud_top_z - cloud_base_z
    inv_slab = 1.0 / slab_thickness

    # Pre-loop: sample coverage + noise ONCE at ray midpoint.
    # This eliminates all hash/noise from the inner loop.
    mid_t = (entry_t + exit_t) * 0.5
    mid_x = ox + mid_t * dx
    mid_y = oy + mid_t * dy
    mid_gx = mid_x / pixel_spacing_x * grid_cols / terrain_cols
    mid_gy = mid_y / pixel_spacing_y * grid_rows / terrain_rows
    mid_ix = int(mid_gx)
    mid_iy = int(mid_gy)
    if mid_ix < 0 or mid_ix >= grid_cols or mid_iy < 0 or mid_iy >= grid_rows:
        return
    coverage = cloud_cover_grid[mid_iy, mid_ix]
    if coverage < 0.05:
        return

    # Single noise sample for cloud shape (2D at slab midpoint)
    noise_scale = 8.0 * inv_slab
    nx = mid_x * noise_scale + time_offset * 0.3
    ny = mid_y * noise_scale + time_offset * 0.1
    noise_val = _cloud_noise_3d(nx, ny, 0.0)

    # Pre-compute base shape: coverage * noise, then threshold
    base_shape = coverage * noise_val
    if base_shape < 0.05:
        return

    # March through slab in 8 steps — inner loop is now trivially cheap
    num_steps = 8
    step_size = (exit_t - entry_t) / num_steps

    # Sun direction for lighting
    sun_x = sun_dir[0]
    sun_y = sun_dir[1]
    sun_z = sun_dir[2]

    # Henyey-Greenstein phase function (cos angle between view and sun)
    cos_theta = -(dx * sun_x + dy * sun_y + dz * sun_z)
    g = 0.6
    g2 = g * g
    phase = 0.25 / 3.14159 * (1.0 - g2) / ((1.0 + g2 - 2.0 * g * cos_theta) ** 1.5)
    g_back = -0.3
    g_back2 = g_back * g_back
    phase_back = 0.25 / 3.14159 * (1.0 - g_back2) / ((1.0 + g_back2 - 2.0 * g_back * cos_theta) ** 1.5)
    phase_total = 0.7 * phase + 0.3 * phase_back

    transmittance = 1.0
    cloud_r = 0.0
    cloud_g = 0.0
    cloud_b = 0.0

    # Sun color: warm white
    sun_cr = 1.0
    sun_cg = 0.95
    sun_cb = 0.85
    # Ambient sky light
    amb_cr = 0.55
    amb_cg = 0.65
    amb_cb = 0.80

    for step in range(num_steps):
        if transmittance < 0.01:
            break

        t = entry_t + (step + 0.5) * step_size
        sz = oz + t * dz

        # Height fraction within slab [0,1]
        h_frac = (sz - cloud_base_z) * inv_slab
        if h_frac < 0.0 or h_frac > 1.0:
            continue

        # Height gradient: cumulus profile
        if h_frac < 0.3:
            h_grad = h_frac / 0.3
        elif h_frac < 0.6:
            h_grad = 1.0
        else:
            h_grad = 1.0 - (h_frac - 0.6) * 2.5
        if h_grad <= 0.0:
            continue

        # Density from pre-computed shape * height gradient
        density = (base_shape * h_grad - 0.12) * 3.5
        if density <= 0.0:
            continue
        if density > 1.0:
            density = 1.0

        # Beer-Lambert + powder approximation
        optical_step = density * step_size * extinction
        beer = math.exp(-optical_step)
        powder = 1.0 - math.exp(-2.0 * optical_step)
        energy = 2.0 * beer * powder

        # Light energy: phase-function sun + ambient sky
        light_r = energy * phase_total * sun_cr + energy * 0.35 * amb_cr
        light_g = energy * phase_total * sun_cg + energy * 0.35 * amb_cg
        light_b = energy * phase_total * sun_cb + energy * 0.35 * amb_cb

        # Accumulate
        cloud_r += transmittance * light_r
        cloud_g += transmittance * light_g
        cloud_b += transmittance * light_b
        transmittance *= beer

    # If no cloud contribution, skip
    if transmittance > 0.999:
        return

    # Composite: final = cloud_accumulated + transmittance * existing_pixel
    existing_r = output[py, px, 0]
    existing_g = output[py, px, 1]
    existing_b = output[py, px, 2]

    output[py, px, 0] = cloud_r + transmittance * existing_r
    output[py, px, 1] = cloud_g + transmittance * existing_g
    output[py, px, 2] = cloud_b + transmittance * existing_b


_DUMMY_CLOUD_1x1 = None


def _apply_volumetric_clouds(d_output, d_primary_rays, d_primary_hits,
                             cloud_cover_grid, d_sun_dir,
                             width, height,
                             cloud_base_z, cloud_top_z,
                             pixel_spacing_x, pixel_spacing_y,
                             terrain_rows, terrain_cols,
                             cloud_time, extinction=0.8):
    """Launch volumetric cloud ray march kernel."""
    global _DUMMY_CLOUD_1x1
    if cloud_cover_grid is None:
        if _DUMMY_CLOUD_1x1 is None:
            _DUMMY_CLOUD_1x1 = cupy.zeros((1, 1), dtype=np.float32)
        cloud_cover_grid = _DUMMY_CLOUD_1x1
    grid_rows, grid_cols = cloud_cover_grid.shape
    if grid_rows <= 1:
        return  # No real cloud data
    # Normalize extinction by slab thickness so visual density is
    # scale-independent.  Target ~3 optical depths at full density
    # across the entire slab → translucent-to-opaque clouds.
    slab_thickness = cloud_top_z - cloud_base_z
    if slab_thickness > 0:
        extinction = 3.0 / slab_thickness

    num_pixels = width * height
    threadsperblock = 256
    blockspergrid = (num_pixels + threadsperblock - 1) // threadsperblock
    _volumetric_cloud_kernel[blockspergrid, threadsperblock](
        d_output, d_primary_rays, d_primary_hits,
        cloud_cover_grid,
        d_sun_dir,
        np.int32(width), np.int32(height),
        np.float32(cloud_base_z), np.float32(cloud_top_z),
        np.float32(pixel_spacing_x), np.float32(pixel_spacing_y),
        np.int32(grid_rows), np.int32(grid_cols),
        np.float32(cloud_time),
        np.float32(extinction),
        np.int32(terrain_rows), np.int32(terrain_cols),
    )


@cuda.jit
def _shade_terrain_kernel(
    output, albedo_out, primary_rays, primary_hits, shadow_hits,
    elevation_data, color_lut, num_rays, width, height,
    sun_dir, ambient, cast_shadows,
    fog_density, fog_color_r, fog_color_g, fog_color_b,
    sky_color_r, sky_color_g, sky_color_b,
    elev_min, elev_range, alpha_channel,
    viewshed_data, viewshed_enabled, viewshed_opacity,
    observer_x, observer_y,
    pixel_spacing_x, pixel_spacing_y,
    color_stretch,
    rgb_texture,
    rgb_texture_offset_y, rgb_texture_offset_x,
    overlay_data, overlay_alpha, overlay_min, overlay_range,
    overlay_as_water, overlay_color_lut,
    overlay_offset_y, overlay_offset_x,
    instance_ids, geometry_colors,
    primitive_ids, point_colors, point_color_offsets,
    ao_factor, gi_color, gi_intensity,
    reflection_hits, reflection_rays,
    cloud_fog_map, cloud_fog_density
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

            # Convert world coords to pixel indices (needed for colormap and viewshed)
            elev_y = int(hit_y / pixel_spacing_y + 0.5)
            elev_x = int(hit_x / pixel_spacing_x + 0.5)

            # Check for per-geometry solid color override
            # Alpha encoding: 0 = no override, (0,1] = normal shaded,
            #                  (1,2) = emissive glow (alpha-1 = min lighting floor),
            #                  >=2   = water shader  (alpha-2 = specular strength)
            #                  >=5   = per-point color (sphere/point cloud)
            inst_id = instance_ids[idx]
            has_color_override = False
            emissive = 0.0
            is_water = False
            water_specular = 0.0
            if inst_id >= 0 and inst_id < geometry_colors.shape[0]:
                gc_alpha = geometry_colors[inst_id, 3]
                if gc_alpha >= 5.0:
                    # Per-point color lookup for sphere geometries
                    prim_id = primitive_ids[idx]
                    if (inst_id < point_color_offsets.shape[0]
                            and point_color_offsets[inst_id] >= 0
                            and prim_id >= 0):
                        pc_idx = (point_color_offsets[inst_id] + prim_id) * 4
                        if pc_idx + 3 < point_colors.shape[0]:
                            base_r = point_colors[pc_idx]
                            base_g = point_colors[pc_idx + 1]
                            base_b = point_colors[pc_idx + 2]
                            has_color_override = True
                    if not has_color_override:
                        # Fallback to geometry color RGB
                        base_r = geometry_colors[inst_id, 0]
                        base_g = geometry_colors[inst_id, 1]
                        base_b = geometry_colors[inst_id, 2]
                        has_color_override = True
                elif gc_alpha > 0.0:
                    base_r = geometry_colors[inst_id, 0]
                    base_g = geometry_colors[inst_id, 1]
                    base_b = geometry_colors[inst_id, 2]
                    has_color_override = True
                    if gc_alpha >= 2.0:
                        is_water = True
                        water_specular = gc_alpha - 2.0
                    elif gc_alpha > 1.0:
                        emissive = gc_alpha - 1.0

            if not has_color_override:

                elev_h = elevation_data.shape[0]
                elev_w = elevation_data.shape[1]

                # Check for NaN ocean terrain
                is_nan_ocean = False
                if 0 <= elev_y < elev_h and 0 <= elev_x < elev_w:
                    if math.isnan(elevation_data[elev_y, elev_x]):
                        is_nan_ocean = True

                if is_nan_ocean:
                    # Ocean water — deep blue base, flat normal, water shader
                    is_water = True
                    water_specular = 0.12
                    base_r = 0.06
                    base_g = 0.12
                    base_b = 0.22
                    nx = 0.0
                    ny = 0.0
                    nz = 1.0
                else:
                    # RGB texture mode: real texture has shape > 1, dummy is (1,1,3)
                    tex_h = rgb_texture.shape[0]
                    tex_w = rgb_texture.shape[1]

                    if tex_h > 1:
                        # Sample RGB directly from tile texture
                        tex_y = elev_y - rgb_texture_offset_y
                        tex_x = elev_x - rgb_texture_offset_x
                        if tex_y >= 0 and tex_y < tex_h and tex_x >= 0 and tex_x < tex_w:
                            base_r = rgb_texture[tex_y, tex_x, 0]
                            base_g = rgb_texture[tex_y, tex_x, 1]
                            base_b = rgb_texture[tex_y, tex_x, 2]
                        else:
                            base_r = 0.3
                            base_g = 0.3
                            base_b = 0.3
                    else:
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

                        # Apply nonlinear stretch: 0=linear, 1=cbrt, 2=log, 3=sqrt
                        if color_stretch == 1:
                            elev_norm = math.pow(elev_norm, 1.0 / 3.0)
                        elif color_stretch == 2:
                            elev_norm = math.log(1.0 + elev_norm * 9.0) / math.log(10.0)
                        elif color_stretch == 3:
                            elev_norm = math.sqrt(elev_norm)

                        # Color lookup
                        lut_idx = int(elev_norm * 255)
                        if lut_idx > 255:
                            lut_idx = 255
                        if lut_idx < 0:
                            lut_idx = 0

                        base_r = color_lut[lut_idx, 0]
                        base_g = color_lut[lut_idx, 1]
                        base_b = color_lut[lut_idx, 2]

                    # Overlay blending: transparent scalar layer on top of base
                    ov_h = overlay_data.shape[0]
                    ov_w = overlay_data.shape[1]
                    if ov_h > 1 and overlay_alpha > 0.0:
                        ov_y = elev_y - overlay_offset_y
                        ov_x = elev_x - overlay_offset_x
                        if ov_y >= 0 and ov_y < ov_h and ov_x >= 0 and ov_x < ov_w:
                            ov_val = overlay_data[ov_y, ov_x]
                            if not math.isnan(ov_val):
                                if overlay_as_water and ov_val > 0.5:
                                    # Flood water shader — same look as ocean
                                    is_water = True
                                    water_specular = 0.12
                                    base_r = 0.06
                                    base_g = 0.12
                                    base_b = 0.22
                                    nx = 0.0
                                    ny = 0.0
                                    nz = 1.0
                                else:
                                    if overlay_range > 0:
                                        ov_norm = (ov_val - overlay_min) / overlay_range
                                    else:
                                        ov_norm = 0.5
                                    if ov_norm < 0:
                                        ov_norm = 0.0
                                    elif ov_norm > 1:
                                        ov_norm = 1.0
                                    _use_ov_lut = overlay_color_lut.shape[0] > 1
                                    if not _use_ov_lut:
                                        # Apply color stretch only with terrain colormap
                                        if color_stretch == 1:
                                            ov_norm = math.pow(ov_norm, 1.0 / 3.0)
                                        elif color_stretch == 2:
                                            ov_norm = math.log(1.0 + ov_norm * 9.0) / math.log(10.0)
                                        elif color_stretch == 3:
                                            ov_norm = math.sqrt(ov_norm)
                                    ov_idx = int(ov_norm * 255)
                                    if ov_idx > 255:
                                        ov_idx = 255
                                    if ov_idx < 0:
                                        ov_idx = 0
                                    if _use_ov_lut:
                                        ov_r = overlay_color_lut[ov_idx, 0]
                                        ov_g = overlay_color_lut[ov_idx, 1]
                                        ov_b = overlay_color_lut[ov_idx, 2]
                                    else:
                                        ov_r = color_lut[ov_idx, 0]
                                        ov_g = color_lut[ov_idx, 1]
                                        ov_b = color_lut[ov_idx, 2]
                                    a = overlay_alpha
                                    base_r = base_r * (1.0 - a) + ov_r * a
                                    base_g = base_g * (1.0 - a) + ov_g * a
                                    base_b = base_b * (1.0 - a) + ov_b * a

            # Write albedo (material color before lighting) for denoiser guide
            if albedo_out.shape[0] > 1:
                albedo_out[py, px, 0] = base_r
                albedo_out[py, px, 1] = base_g
                albedo_out[py, px, 2] = base_b

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

            # Cloud shadow — patchy darkening where cloud_cover is high
            cfm_h = cloud_fog_map.shape[0]
            cfm_w = cloud_fog_map.shape[1]
            cloud_shadow = 0.0
            if cfm_h > 1:
                cy = elev_y
                if cy < 0:
                    cy = 0
                elif cy >= cfm_h:
                    cy = cfm_h - 1
                cx = elev_x
                if cx < 0:
                    cx = 0
                elif cx >= cfm_w:
                    cx = cfm_w - 1
                cc = cloud_fog_map[cy, cx]
                if cc > 0.01:
                    # Procedural noise for patchy cloud shapes
                    wx = hit_x * cloud_fog_density
                    wy = hit_y * cloud_fog_density
                    n = (math.sin(wx * 1.2 + wy * 0.7) * 0.45
                         + math.sin(wx * 2.7 - wy * 1.8 + 3.1) * 0.30
                         + math.sin(wx * 0.5 + wy * 3.4 - 1.7) * 0.25)
                    # n in ~[-1,1] → [0,1], squared for sharper edges
                    cloud_mask = n * 0.5 + 0.5
                    cloud_mask = cloud_mask * cloud_mask
                    cloud_shadow = cc * cloud_mask * 0.7
                    if cloud_shadow > 0.7:
                        cloud_shadow = 0.7
                    shadow_factor *= (1.0 - cloud_shadow)

            # Final lighting
            diffuse = cos_theta * shadow_factor
            lighting = ambient + (1.0 - ambient) * diffuse
            # Apply ambient occlusion
            lighting *= ao_factor[idx]
            # Emissive glow: raise the lighting floor
            if emissive > 0.0:
                if lighting < emissive:
                    lighting = emissive

            color_r = base_r * lighting + base_r * gi_color[idx, 0] * gi_intensity
            color_g = base_g * lighting + base_g * gi_color[idx, 1] * gi_intensity
            color_b = base_b * lighting + base_b * gi_color[idx, 2] * gi_intensity

            # Water shader: reflections + specular highlight + Fresnel
            if is_water:
                # Procedural wave normals for shimmer
                wx = hit_x * 0.5
                wy = hit_y * 0.5
                h1 = (math.sin(wx * 1.1 + wy * 0.7) * 0.4
                      + math.sin(wx * 2.3 - wy * 1.9) * 0.3)
                h2 = (math.sin(wx * 0.8 - wy * 1.3) * 0.4
                      + math.sin(wx * 1.7 + wy * 2.1) * 0.3)
                wave_strength = 0.015
                nx += h1 * wave_strength
                ny += h2 * wave_strength
                n_len = math.sqrt(nx * nx + ny * ny + nz * nz)
                nx /= n_len
                ny /= n_len
                nz /= n_len

                # View direction
                vx = -ray_dx
                vy = -ray_dy
                vz = -ray_dz

                # Fresnel: more reflective at grazing angles
                n_dot_v = abs(nx * vx + ny * vy + nz * vz)
                fresnel = 0.3 + 0.7 * (1.0 - n_dot_v)

                # Compute reflection color from traced reflection rays
                refl_t = reflection_hits[idx, 0]
                if refl_t > 0:
                    # Reflection hit terrain — shade with simple colormap + diffuse
                    refl_hx = reflection_rays[idx, 0] + refl_t * reflection_rays[idx, 4]
                    refl_hy = reflection_rays[idx, 1] + refl_t * reflection_rays[idx, 5]
                    refl_hz = reflection_rays[idx, 2] + refl_t * reflection_rays[idx, 6]

                    # Look up elevation at reflected hit point
                    refl_ey = int(refl_hy / pixel_spacing_y + 0.5)
                    refl_ex = int(refl_hx / pixel_spacing_x + 0.5)
                    elev_h = elevation_data.shape[0]
                    elev_w = elevation_data.shape[1]

                    # Check for RGB texture first
                    tex_h = rgb_texture.shape[0]
                    tex_w = rgb_texture.shape[1]

                    refl_tex_y = refl_ey - rgb_texture_offset_y
                    refl_tex_x = refl_ex - rgb_texture_offset_x
                    if tex_h > 1 and refl_tex_y >= 0 and refl_tex_y < tex_h and refl_tex_x >= 0 and refl_tex_x < tex_w:
                        refl_r = rgb_texture[refl_tex_y, refl_tex_x, 0]
                        refl_g = rgb_texture[refl_tex_y, refl_tex_x, 1]
                        refl_b = rgb_texture[refl_tex_y, refl_tex_x, 2]
                    elif refl_ey >= 0 and refl_ey < elev_h and refl_ex >= 0 and refl_ex < elev_w:
                        refl_elev = elevation_data[refl_ey, refl_ex]
                        if elev_range > 0:
                            refl_norm = (refl_elev - elev_min) / elev_range
                        else:
                            refl_norm = 0.5
                        if refl_norm < 0.0:
                            refl_norm = 0.0
                        elif refl_norm > 1.0:
                            refl_norm = 1.0
                        refl_lut = int(refl_norm * 255)
                        if refl_lut > 255:
                            refl_lut = 255
                        if refl_lut < 0:
                            refl_lut = 0
                        refl_r = color_lut[refl_lut, 0]
                        refl_g = color_lut[refl_lut, 1]
                        refl_b = color_lut[refl_lut, 2]
                    else:
                        refl_r = 0.3
                        refl_g = 0.3
                        refl_b = 0.3

                    # Simple diffuse lighting on reflected surface
                    refl_nx = reflection_hits[idx, 1]
                    refl_ny = reflection_hits[idx, 2]
                    refl_nz = reflection_hits[idx, 3]
                    refl_cos = refl_nx * sun_dir[0] + refl_ny * sun_dir[1] + refl_nz * sun_dir[2]
                    if refl_cos < 0.0:
                        refl_cos = -refl_cos
                    refl_light = ambient + (1.0 - ambient) * refl_cos
                    refl_r *= refl_light
                    refl_g *= refl_light
                    refl_b *= refl_light
                else:
                    # Reflection miss -> sky
                    ref_dx = reflection_rays[idx, 4]
                    ref_dy = reflection_rays[idx, 5]
                    ref_dz = reflection_rays[idx, 6]
                    if sky_color_r < 0:
                        refl_r, refl_g, refl_b = _compute_physical_sky(ref_dx, ref_dy, ref_dz, sun_dir)
                    else:
                        refl_r = sky_color_r
                        refl_g = sky_color_g
                        refl_b = sky_color_b

                # Blend base water color with reflection using Fresnel
                color_r = color_r * (1.0 - fresnel) + refl_r * fresnel
                color_g = color_g * (1.0 - fresnel) + refl_g * fresnel
                color_b = color_b * (1.0 - fresnel) + refl_b * fresnel

                # Blinn-Phong specular: H = normalize(L + V)
                hx = sun_dir[0] + vx
                hy = sun_dir[1] + vy
                hz = sun_dir[2] + vz
                h_len = math.sqrt(hx * hx + hy * hy + hz * hz)
                if h_len > 1e-6:
                    hx /= h_len
                    hy /= h_len
                    hz /= h_len
                n_dot_h = nx * hx + ny * hy + nz * hz
                if n_dot_h < 0.0:
                    n_dot_h = 0.0
                # Sharp specular exponent for water glints
                spec = n_dot_h * n_dot_h
                spec = spec * spec     # ^4
                spec = spec * spec     # ^8
                spec = spec * spec     # ^16
                spec = spec * spec     # ^32
                spec = spec * spec     # ^64
                spec *= water_specular * shadow_factor

                # Add specular on top
                color_r += spec
                color_g += spec
                color_b += spec * 0.9

            # Observer marker removed — drone mesh is placed as scene geometry

            # Viewshed overlay:
            #  - Terrain: teal glow on visible areas
            #  - Buildings/geometry: light green tint if in visible area
            if viewshed_enabled:
                vs_h = viewshed_data.shape[0]
                vs_w = viewshed_data.shape[1]
                if elev_y >= 0 and elev_y < vs_h and elev_x >= 0 and elev_x < vs_w:
                    vis_val = viewshed_data[elev_y, elev_x]
                    if not math.isnan(vis_val) and vis_val >= 0.0:
                        alpha = viewshed_opacity
                        if has_color_override:
                            # Light green for buildings in viewshed
                            color_r = color_r * (1.0 - alpha) + 0.4 * alpha
                            color_g = color_g * (1.0 - alpha) + 0.95 * alpha
                            color_b = color_b * (1.0 - alpha) + 0.3 * alpha
                        else:
                            # Teal glow for terrain
                            color_r = color_r * (1.0 - alpha)
                            color_g = color_g * (1.0 - alpha) + 0.9 * alpha
                            color_b = color_b * (1.0 - alpha) + 0.85 * alpha

            # Height-attenuated atmospheric fog
            # Fog density decays exponentially with altitude:
            #   rho(z) = fog_density * exp(-b * (z - elev_min))
            # The optical depth along the ray is the analytic integral
            # of rho over the camera-to-hit path, so valleys fill with
            # haze while ridgelines stay crisp.
            if fog_density > 0:
                b = 2.5 / (elev_range + 1e-6)
                dz = hit_z - oz
                if abs(dz) > 0.001:
                    exp_cam = math.exp(-b * (oz - elev_min))
                    exp_hit = math.exp(-b * (hit_z - elev_min))
                    optical_depth = fog_density * t * (exp_cam - exp_hit) / (b * dz)
                else:
                    z_mid = (oz + hit_z) * 0.5
                    optical_depth = fog_density * t * math.exp(-b * (z_mid - elev_min))
                if optical_depth > 0.0:
                    fog_amount = 1.0 - math.exp(-optical_depth)
                    color_r = color_r * (1 - fog_amount) + fog_color_r * fog_amount
                    color_g = color_g * (1 - fog_amount) + fog_color_g * fog_amount
                    color_b = color_b * (1 - fog_amount) + fog_color_b * fog_amount

            # Cloud desaturation — overcast areas lose color contrast
            if cloud_shadow > 0.15:
                gray = color_r * 0.3 + color_g * 0.59 + color_b * 0.11
                desat = (cloud_shadow - 0.15) * 0.6
                if desat > 0.35:
                    desat = 0.35
                color_r = color_r * (1.0 - desat) + gray * desat
                color_g = color_g * (1.0 - desat) + gray * desat
                color_b = color_b * (1.0 - desat) + gray * desat

            output[py, px, 0] = color_r
            output[py, px, 1] = color_g
            output[py, px, 2] = color_b
            if alpha_channel:
                output[py, px, 3] = 1.0
        else:
            # Miss - black albedo (sky has no material)
            if albedo_out.shape[0] > 1:
                albedo_out[py, px, 0] = 0.0
                albedo_out[py, px, 1] = 0.0
                albedo_out[py, px, 2] = 0.0
            # Miss - sky color
            if sky_color_r < 0:
                # Physical sky via shared device function
                ray_dx = primary_rays[idx, 4]
                ray_dy = primary_rays[idx, 5]
                ray_dz = primary_rays[idx, 6]

                sr, sg, sb = _compute_physical_sky(ray_dx, ray_dy, ray_dz, sun_dir)
                if sr > 1.0:
                    sr = 1.0
                if sg > 1.0:
                    sg = 1.0
                if sb > 1.0:
                    sb = 1.0

                output[py, px, 0] = sr
                output[py, px, 1] = sg
                output[py, px, 2] = sb
            else:
                output[py, px, 0] = sky_color_r
                output[py, px, 1] = sky_color_g
                output[py, px, 2] = sky_color_b
            if alpha_channel:
                output[py, px, 3] = 0.0


@cuda.jit
def _tone_map_aces_kernel(output, height, width, num_channels):
    """Apply ACES filmic tone mapping in-place (Stephen Hill approximation)."""
    idx = cuda.grid(1)
    if idx < height * width:
        py = idx // width
        px = idx % width
        for c in range(num_channels):
            x = output[py, px, c]
            # ACES filmic: (x*(2.51*x+0.03)) / (x*(2.43*x+0.59)+0.14)
            output[py, px, c] = (x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14)


def _tone_map_aces(output):
    """Apply ACES filmic tone mapping to GPU output buffer in-place."""
    height, width, num_channels = output.shape
    num_pixels = height * width
    threadsperblock = 256
    blockspergrid = (num_pixels + threadsperblock - 1) // threadsperblock
    _tone_map_aces_kernel[blockspergrid, threadsperblock](output, height, width, num_channels)


@cuda.jit
def _edge_outline_kernel(output, instance_ids, height, width,
                         edge_strength, edge_r, edge_g, edge_b):
    """Darken pixels at boundaries between different instance_ids."""
    idx = cuda.grid(1)
    if idx >= height * width:
        return
    my_id = instance_ids[idx]
    if my_id < 0:
        return
    py = idx // width
    px = idx % width
    is_edge = False
    # Check 4 cardinal neighbors
    if py > 0 and instance_ids[idx - width] != my_id:
        is_edge = True
    elif py < height - 1 and instance_ids[idx + width] != my_id:
        is_edge = True
    elif px > 0 and instance_ids[idx - 1] != my_id:
        is_edge = True
    elif px < width - 1 and instance_ids[idx + 1] != my_id:
        is_edge = True
    if is_edge:
        inv = 1.0 - edge_strength
        output[py, px, 0] = output[py, px, 0] * inv + edge_r * edge_strength
        output[py, px, 1] = output[py, px, 1] * inv + edge_g * edge_strength
        output[py, px, 2] = output[py, px, 2] * inv + edge_b * edge_strength


def _edge_outline(output, instance_ids, edge_strength=0.6,
                  edge_color=(0.05, 0.05, 0.05)):
    """Apply screen-space edge detection on instance_id boundaries."""
    height, width, _ = output.shape
    num_pixels = height * width
    threadsperblock = 256
    blockspergrid = (num_pixels + threadsperblock - 1) // threadsperblock
    _edge_outline_kernel[blockspergrid, threadsperblock](
        output, instance_ids, height, width,
        edge_strength, *edge_color)


@cuda.jit
def _edl_kernel(output, depth, height, width, radius, strength):
    """Eye Dome Lighting: darken pixels at depth discontinuities."""
    idx = cuda.grid(1)
    if idx >= height * width:
        return
    py = idx // width
    px = idx % width
    center_d = depth[idx]
    if center_d <= 0.0:
        return
    log_center = math.log2(center_d)
    response = 0.0
    # Sample 8 directions (pi/4 apart)
    for i in range(8):
        angle = i * (math.pi / 4.0)
        # math.floor(x + 0.5) for correct rounding of negative offsets
        nx = px + int(math.floor(radius * math.cos(angle) + 0.5))
        ny = py + int(math.floor(radius * math.sin(angle) + 0.5))
        if 0 <= nx < width and 0 <= ny < height:
            nd = depth[ny * width + nx]
            if nd > 0.0:
                dd = log_center - math.log2(nd)
                if dd > 0.0:
                    response += dd
            else:
                # Empty/sky neighbor — silhouette edge, add fixed contribution
                response += 0.5
        else:
            # Out-of-bounds — treat as silhouette edge
            response += 0.5
    shade = math.exp(-response * strength)
    output[py, px, 0] *= shade
    output[py, px, 1] *= shade
    output[py, px, 2] *= shade


def _edl(output, depth, width, height, radius=2.0, strength=0.7):
    """Apply Eye Dome Lighting post-process for depth-edge enhancement."""
    num_pixels = height * width
    threadsperblock = 256
    blockspergrid = (num_pixels + threadsperblock - 1) // threadsperblock
    _edl_kernel[blockspergrid, threadsperblock](
        output, depth, height, width, radius, strength)


@cuda.jit
def _compute_flow_kernel(flow_out, primary_rays, primary_hits,
                         width, height,
                         prev_pos, prev_forward, prev_right, prev_up,
                         aspect, fov_scale):
    """Compute per-pixel screen-space motion vectors by reprojecting hits
    through the previous frame's camera."""
    idx = cuda.grid(1)
    if idx >= width * height:
        return
    py = idx // width
    px = idx % width
    t = primary_hits[idx, 0]
    if t <= 0:
        flow_out[py, px, 0] = 0.0
        flow_out[py, px, 1] = 0.0
        return
    # 3D hit point from current ray
    hx = primary_rays[idx, 0] + t * primary_rays[idx, 4]
    hy = primary_rays[idx, 1] + t * primary_rays[idx, 5]
    hz = primary_rays[idx, 2] + t * primary_rays[idx, 6]
    # Reproject through previous camera
    ox = hx - prev_pos[0]
    oy = hy - prev_pos[1]
    oz = hz - prev_pos[2]
    depth = ox * prev_forward[0] + oy * prev_forward[1] + oz * prev_forward[2]
    if depth <= 1e-6:
        flow_out[py, px, 0] = 0.0
        flow_out[py, px, 1] = 0.0
        return
    u = (ox * prev_right[0] + oy * prev_right[1] + oz * prev_right[2]) / (depth * aspect * fov_scale)
    v = (ox * prev_up[0] + oy * prev_up[1] + oz * prev_up[2]) / (depth * fov_scale)
    prev_px = (u + 1.0) * width / 2.0 - 0.5
    prev_py = (1.0 - v) * height / 2.0 - 0.5
    flow_out[py, px, 0] = prev_px - px
    flow_out[py, px, 1] = prev_py - py


def compute_flow(flow_out, primary_rays, primary_hits, width, height,
                 prev_pos, prev_forward, prev_right, prev_up,
                 aspect, fov_scale):
    """Compute screen-space flow vectors for temporal denoising.

    Parameters
    ----------
    flow_out : cupy.ndarray
        (height, width, 2) float32 output — per-pixel (dx, dy) in pixels.
    primary_rays, primary_hits : cupy.ndarray
        Ray buffers from current frame.
    prev_pos, prev_forward, prev_right, prev_up : cupy.ndarray
        Previous frame camera basis vectors (device arrays, shape (3,)).
    aspect : float
        Aspect ratio (width / height).
    fov_scale : float
        tan(fov_radians / 2).
    """
    num_rays = width * height
    threadsperblock = 256
    blockspergrid = (num_rays + threadsperblock - 1) // threadsperblock
    _compute_flow_kernel[blockspergrid, threadsperblock](
        flow_out, primary_rays, primary_hits, width, height,
        prev_pos, prev_forward, prev_right, prev_up,
        aspect, fov_scale
    )


@cuda.jit
def _bloom_threshold_kernel(bright, output, height, width, threshold):
    """Extract pixels brighter than threshold into a separate buffer."""
    idx = cuda.grid(1)
    if idx < height * width:
        py = idx // width
        px = idx % width
        r = output[py, px, 0]
        g = output[py, px, 1]
        b = output[py, px, 2]
        lum = 0.2126 * r + 0.7152 * g + 0.0722 * b
        if lum > threshold:
            scale = (lum - threshold) / lum
            bright[py, px, 0] = r * scale
            bright[py, px, 1] = g * scale
            bright[py, px, 2] = b * scale
        else:
            bright[py, px, 0] = 0.0
            bright[py, px, 1] = 0.0
            bright[py, px, 2] = 0.0


@cuda.jit
def _bloom_blur_kernel(dst, src, height, width, radius, horizontal):
    """Separable Gaussian blur (approximate with linear weights)."""
    idx = cuda.grid(1)
    if idx < height * width:
        py = idx // width
        px = idx % width

        acc_r = 0.0
        acc_g = 0.0
        acc_b = 0.0
        weight_sum = 0.0

        for i in range(-radius, radius + 1):
            if horizontal:
                sx = px + i
                sy = py
            else:
                sx = px
                sy = py + i

            if sx >= 0 and sx < width and sy >= 0 and sy < height:
                # Gaussian weight: exp(-0.5 * (i/sigma)^2), sigma ≈ radius/2.5
                sigma = float(radius) / 2.5
                w = math.exp(-0.5 * (float(i) / sigma) * (float(i) / sigma))
                acc_r += src[sy, sx, 0] * w
                acc_g += src[sy, sx, 1] * w
                acc_b += src[sy, sx, 2] * w
                weight_sum += w

        if weight_sum > 0:
            dst[py, px, 0] = acc_r / weight_sum
            dst[py, px, 1] = acc_g / weight_sum
            dst[py, px, 2] = acc_b / weight_sum


@cuda.jit
def _bloom_composite_kernel(output, bloom, height, width, intensity):
    """Additively blend bloom buffer into output."""
    idx = cuda.grid(1)
    if idx < height * width:
        py = idx // width
        px = idx % width
        output[py, px, 0] += bloom[py, px, 0] * intensity
        output[py, px, 1] += bloom[py, px, 1] * intensity
        output[py, px, 2] += bloom[py, px, 2] * intensity


def _bloom(output, temp, scratch, threshold=0.7, radius=12, intensity=0.35):
    """Apply bloom post-process: threshold -> blur -> composite."""
    height, width = output.shape[0], output.shape[1]
    num_pixels = height * width
    threadsperblock = 256
    blockspergrid = (num_pixels + threadsperblock - 1) // threadsperblock

    # Extract bright pixels into temp
    _bloom_threshold_kernel[blockspergrid, threadsperblock](
        temp, output, height, width, np.float32(threshold)
    )

    # Horizontal blur: temp -> scratch
    _bloom_blur_kernel[blockspergrid, threadsperblock](
        scratch, temp, height, width, np.int32(radius), True
    )

    # Vertical blur: scratch -> temp
    _bloom_blur_kernel[blockspergrid, threadsperblock](
        temp, scratch, height, width, np.int32(radius), False
    )

    # Composite: add bloom back into output
    _bloom_composite_kernel[blockspergrid, threadsperblock](
        output, temp, height, width, np.float32(intensity)
    )


# Lazy singletons for dummy GPU arrays (avoid per-frame allocations)
_DUMMY_1x1 = None
_DUMMY_1x1x3 = None
_DUMMY_1x4 = None
_DUMMY_AO_ONES = None  # (num_rays,) all-ones buffer for disabled AO
_DUMMY_AO_SIZE = 0
_DUMMY_GI_COLOR = None  # (num_rays, 3) all-zero for no GI
_DUMMY_GI_SIZE = 0
_DUMMY_REFL_HITS = None  # (num_rays, 4) all-zero for no reflections
_DUMMY_REFL_RAYS = None
_DUMMY_REFL_SIZE = 0
_DUMMY_ALBEDO = None  # (1, 1, 3) placeholder when albedo not captured


def _shade_terrain(
    output, primary_rays, primary_hits, shadow_hits,
    elevation_data, color_lut, num_rays, width, height,
    sun_dir, ambient, cast_shadows,
    fog_density, fog_color,
    elev_min, elev_range, alpha,
    viewshed_data=None, viewshed_opacity=0.6,
    observer_x=-1e30, observer_y=-1e30,
    pixel_spacing_x=1.0, pixel_spacing_y=1.0,
    color_stretch=0,
    sky_color=(-1.0, 0.0, 0.0),
    rgb_texture=None,
    rgb_texture_offset_y=0, rgb_texture_offset_x=0,
    overlay_data=None, overlay_alpha=0.5,
    overlay_min=0.0, overlay_range=1.0,
    overlay_as_water=False,
    overlay_color_lut=None,
    overlay_offset_y=0, overlay_offset_x=0,
    instance_ids=None, geometry_colors=None,
    primitive_ids=None, point_colors=None, point_color_offsets=None,
    ao_factor=None, gi_color=None, gi_intensity=2.0,
    reflection_hits=None, reflection_rays=None,
    albedo_out=None,
    cloud_fog_map=None, cloud_fog_density=0.0,
):
    """Apply terrain shading with all effects."""
    threadsperblock = 256
    blockspergrid = (num_rays + threadsperblock - 1) // threadsperblock

    global _DUMMY_1x1, _DUMMY_1x1x3, _DUMMY_1x4

    # Handle viewshed - need a placeholder if not provided
    viewshed_enabled = viewshed_data is not None
    if not viewshed_enabled:
        if _DUMMY_1x1 is None:
            _DUMMY_1x1 = cupy.zeros((1, 1), dtype=np.float32)
        viewshed_data = _DUMMY_1x1

    # Handle RGB texture - need a placeholder if not provided
    # Dummy is (1,1,3); kernel checks shape[0] > 1 to decide whether to use it
    if rgb_texture is None:
        if _DUMMY_1x1x3 is None:
            _DUMMY_1x1x3 = cupy.zeros((1, 1, 3), dtype=np.float32)
        rgb_texture = _DUMMY_1x1x3

    # Handle overlay data - dummy (1,1) when not provided
    # Kernel checks shape[0] > 1 to decide whether to blend
    if overlay_data is None:
        if _DUMMY_1x1 is None:
            _DUMMY_1x1 = cupy.zeros((1, 1), dtype=np.float32)
        overlay_data = _DUMMY_1x1

    # Handle overlay color LUT — custom palette for categorical overlays
    # Dummy is (1,3); kernel checks shape[0] > 1 to decide whether to use
    if overlay_color_lut is None:
        if _DUMMY_1x1x3 is None:
            _DUMMY_1x1x3 = cupy.zeros((1, 1, 3), dtype=np.float32)
        overlay_color_lut = _DUMMY_1x1x3[:, 0, :]  # (1, 3)
    elif not isinstance(overlay_color_lut, cupy.ndarray):
        overlay_color_lut = cupy.asarray(overlay_color_lut, dtype=np.float32)

    # Handle geometry_colors for per-geometry solid coloring
    if geometry_colors is None:
        if _DUMMY_1x4 is None:
            _DUMMY_1x4 = cupy.zeros((1, 4), dtype=np.float32)
        geometry_colors = _DUMMY_1x4

    # Handle per-point colors for sphere geometries
    global _DUMMY_PC_PRIMS, _DUMMY_PC_PRIMS_SIZE
    global _DUMMY_PC_COLORS, _DUMMY_PC_OFFSETS
    if primitive_ids is None:
        if not hasattr(_shade_terrain, '_dpc_prims') or \
                _shade_terrain._dpc_prims_size != num_rays:
            _shade_terrain._dpc_prims = cupy.full(num_rays, -1, dtype=cupy.int32)
            _shade_terrain._dpc_prims_size = num_rays
        primitive_ids = _shade_terrain._dpc_prims
    if point_colors is None:
        if not hasattr(_shade_terrain, '_dpc_colors'):
            _shade_terrain._dpc_colors = cupy.zeros(4, dtype=cupy.float32)
        point_colors = _shade_terrain._dpc_colors
    if point_color_offsets is None:
        if not hasattr(_shade_terrain, '_dpc_offsets'):
            _shade_terrain._dpc_offsets = cupy.full(1, -1, dtype=cupy.int32)
        point_color_offsets = _shade_terrain._dpc_offsets

    # Handle AO factor - cached all-ones when disabled
    global _DUMMY_AO_ONES, _DUMMY_AO_SIZE
    if ao_factor is None:
        if _DUMMY_AO_ONES is None or _DUMMY_AO_SIZE != num_rays:
            _DUMMY_AO_ONES = cupy.ones(num_rays, dtype=np.float32)
            _DUMMY_AO_SIZE = num_rays
        ao_factor = _DUMMY_AO_ONES

    # Handle GI color - cached all-zeros when disabled
    global _DUMMY_GI_COLOR, _DUMMY_GI_SIZE
    if gi_color is None:
        if _DUMMY_GI_COLOR is None or _DUMMY_GI_SIZE != num_rays:
            _DUMMY_GI_COLOR = cupy.zeros((num_rays, 3), dtype=np.float32)
            _DUMMY_GI_SIZE = num_rays
        gi_color = _DUMMY_GI_COLOR

    # Handle reflection buffers - dummy zero arrays when no reflections
    global _DUMMY_REFL_HITS, _DUMMY_REFL_RAYS, _DUMMY_REFL_SIZE
    if reflection_hits is None:
        if _DUMMY_REFL_HITS is None or _DUMMY_REFL_SIZE != num_rays:
            _DUMMY_REFL_HITS = cupy.zeros((num_rays, 4), dtype=np.float32)
            _DUMMY_REFL_RAYS = cupy.zeros((num_rays, 8), dtype=np.float32)
            _DUMMY_REFL_SIZE = num_rays
        reflection_hits = _DUMMY_REFL_HITS
        reflection_rays = _DUMMY_REFL_RAYS

    # Handle albedo output - dummy (1,1,3) when not capturing
    global _DUMMY_ALBEDO
    if albedo_out is None:
        if _DUMMY_ALBEDO is None:
            _DUMMY_ALBEDO = cupy.zeros((1, 1, 3), dtype=np.float32)
        albedo_out = _DUMMY_ALBEDO

    # Handle cloud fog map - dummy (1,1) when not provided
    # Kernel checks shape[0] > 1 to decide whether to apply
    if cloud_fog_map is None:
        if _DUMMY_1x1 is None:
            _DUMMY_1x1 = cupy.zeros((1, 1), dtype=np.float32)
        cloud_fog_map = _DUMMY_1x1

    _shade_terrain_kernel[blockspergrid, threadsperblock](
        output, albedo_out, primary_rays, primary_hits, shadow_hits,
        elevation_data, color_lut, num_rays, width, height,
        sun_dir, ambient, cast_shadows,
        fog_density, fog_color[0], fog_color[1], fog_color[2],
        sky_color[0], sky_color[1], sky_color[2],
        elev_min, elev_range, alpha,
        viewshed_data, viewshed_enabled, viewshed_opacity,
        observer_x, observer_y,
        pixel_spacing_x, pixel_spacing_y,
        color_stretch,
        rgb_texture,
        np.int32(rgb_texture_offset_y), np.int32(rgb_texture_offset_x),
        overlay_data, overlay_alpha, overlay_min, overlay_range,
        overlay_as_water, overlay_color_lut,
        np.int32(overlay_offset_y), np.int32(overlay_offset_x),
        instance_ids, geometry_colors,
        primitive_ids, point_colors, point_color_offsets,
        ao_factor, gi_color, np.float32(gi_intensity),
        reflection_hits, reflection_rays,
        cloud_fog_map, np.float32(cloud_fog_density)
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


class _RenderBuffers:
    """Reusable GPU buffer pool for the render pipeline."""

    def __init__(self):
        self._key = None
        self.primary_rays = None
        self.primary_hits = None
        self.shadow_rays = None
        self.shadow_hits = None
        self.output = None
        self.albedo = None
        self.instance_ids = None
        self.primitive_ids = None
        self.ao_rays = None
        self.ao_hits = None
        self.gi_color = None
        self.gi_throughput = None
        self.reflection_rays = None
        self.reflection_hits = None
        self.bloom_temp = None
        self.bloom_scratch = None

    def get(self, width, height, shadows, alpha, need_instance_ids, ao=False):
        num_rays = width * height
        num_channels = 4 if alpha else 3
        key = (width, height, shadows, alpha, ao)
        if key != self._key:
            self.primary_rays = cupy.empty((num_rays, 8), dtype=np.float32)
            self.primary_hits = cupy.empty((num_rays, 4), dtype=np.float32)
            self.shadow_rays = cupy.empty((num_rays, 8), dtype=np.float32)
            self.shadow_hits = cupy.empty((num_rays, 4), dtype=np.float32)
            self.output = cupy.zeros((height, width, num_channels), dtype=np.float32)
            self.albedo = cupy.zeros((height, width, 3), dtype=np.float32)
            self.instance_ids = cupy.full(num_rays, -1, dtype=cupy.int32)
            self.primitive_ids = cupy.full(num_rays, -1, dtype=cupy.int32)
            if ao:
                self.ao_rays = cupy.empty((num_rays, 8), dtype=np.float32)
                self.ao_hits = cupy.empty((num_rays, 4), dtype=np.float32)
                self.gi_color = cupy.zeros((num_rays, 3), dtype=np.float32)
                self.gi_throughput = cupy.ones((num_rays, 3), dtype=np.float32)
            else:
                self.ao_rays = None
                self.ao_hits = None
                self.gi_color = None
                self.gi_throughput = None
            if need_instance_ids:
                self.reflection_rays = cupy.empty((num_rays, 8), dtype=np.float32)
                self.reflection_hits = cupy.empty((num_rays, 4), dtype=np.float32)
            else:
                self.reflection_rays = None
                self.reflection_hits = None
            self.bloom_temp = cupy.zeros((height, width, 3), dtype=np.float32)
            self.bloom_scratch = cupy.zeros((height, width, 3), dtype=np.float32)
            self._key = key
        else:
            self.output.fill(0)
            self.albedo.fill(0)
            if need_instance_ids:
                self.instance_ids.fill(-1)
            if self.gi_color is not None:
                self.gi_color.fill(0)
        return self


_render_buffers = _RenderBuffers()

_colormap_lut_cache = {}  # {colormap_name: cupy.ndarray on GPU}


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
    mesh_type: str = 'heightfield',
    color_data=None,
    color_stretch: str = 'linear',
    sky_color: Optional[Tuple[float, float, float]] = None,
    rgb_texture=None,
    rgb_texture_offset_y: int = 0,
    rgb_texture_offset_x: int = 0,
    overlay_data=None,
    overlay_alpha: float = 0.5,
    overlay_as_water: bool = False,
    overlay_color_lut=None,
    overlay_offset_y: int = 0,
    overlay_offset_x: int = 0,
    geometry_colors=None,
    ao_samples: int = 0,
    ao_radius: Optional[float] = None,
    ao_seed: int = 0,
    gi_intensity: float = 2.0,
    gi_bounces: int = 1,
    frame_seed: int = 0,
    sun_angle: float = 0.0,
    aperture: float = 0.0,
    focal_distance: float = 0.0,
    tone_map: bool = True,
    bloom: bool = True,
    denoise: bool = False,
    edge_lines: bool = True,
    edge_strength: float = 0.6,
    edge_color: Tuple[float, float, float] = (0.05, 0.05, 0.05),
    edl: bool = True,
    edl_strength: float = 0.7,
    edl_radius: float = 2.0,
    cloud_fog_map=None,
    cloud_fog_density: float = 0.0,
    volumetric_clouds: bool = False,
    cloud_base_z: float = 0.0,
    cloud_top_z: float = 0.0,
    cloud_time: float = 0.0,
    _return_gpu: bool = False,
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
        # Auto-compute: scale so relief is ~20% of horizontal extent (in world units)
        horizontal_extent = max(H * pixel_spacing_y, W * pixel_spacing_x)
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
        optix = prepare_mesh(scaled_raster, rtx=None, mesh_type=mesh_type,
                             pixel_spacing_x=pixel_spacing_x,
                             pixel_spacing_y=pixel_spacing_y)
    else:
        scaled_raster = raster
        optix = prepare_mesh(raster, rtx, mesh_type=mesh_type,
                             pixel_spacing_x=pixel_spacing_x,
                             pixel_spacing_y=pixel_spacing_y)

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

    # Color lookup table (cached on GPU)
    if colormap not in _colormap_lut_cache:
        color_lut = _get_colormap_lut(colormap)
        _colormap_lut_cache[colormap] = cupy.array(color_lut, dtype=np.float32)
    d_color_lut = _colormap_lut_cache[colormap]

    # Determine which data drives the colormap lookup.
    # color_data overrides elevation_data for coloring (e.g. landcover on terrain).
    if color_data is not None:
        if not isinstance(color_data, cupy.ndarray):
            colormap_data = cupy.asarray(color_data, dtype=cupy.float32)
        else:
            colormap_data = color_data.astype(cupy.float32)
    else:
        colormap_data = elevation_data

    # Elevation range for colormap
    if color_range is not None:
        elev_min, elev_max = color_range
    else:
        elev_min = float(cupy.nanmin(colormap_data))
        elev_max = float(cupy.nanmax(colormap_data))
    elev_range = elev_max - elev_min

    # Color stretch mode: string -> int for CUDA kernel (needed early for GI in AO loop)
    _stretch_modes = {'linear': 0, 'cbrt': 1, 'log': 2, 'sqrt': 3}
    stretch_int = _stretch_modes.get(color_stretch, 0)

    # Detect NaN ocean terrain (needs reflection buffers even without geometry)
    has_nan_ocean = bool(cupy.any(cupy.isnan(elevation_data)))

    # Allocate (or reuse) buffers
    bufs = _render_buffers.get(width, height, shadows, alpha,
                               geometry_colors is not None or has_nan_ocean,
                               ao=ao_samples > 0)
    d_primary_rays = bufs.primary_rays
    d_primary_hits = bufs.primary_hits
    d_shadow_rays = bufs.shadow_rays
    d_shadow_hits = bufs.shadow_hits
    d_output = bufs.output

    # Compute derived seeds for AA and soft shadows from frame_seed
    jitter_seed = np.uint32(frame_seed * 3 + 1) if frame_seed > 0 else np.uint32(0)
    shadow_seed = np.uint32(frame_seed * 3 + 2) if frame_seed > 0 else np.uint32(0)
    sun_angle_rad = math.radians(sun_angle) if sun_angle > 0 else 0.0

    # Auto-compute focal distance from camera-to-lookat if not specified
    if aperture > 0 and focal_distance <= 0:
        dx = scaled_look_at[0] - scaled_camera_position[0]
        dy = scaled_look_at[1] - scaled_camera_position[1]
        dz = scaled_look_at[2] - scaled_camera_position[2]
        focal_distance = math.sqrt(dx * dx + dy * dy + dz * dz)

    # Step 1: Generate perspective rays
    _generate_perspective_rays(
        d_primary_rays, width, height,
        d_camera_pos, d_forward, d_right, d_up, fov,
        jitter_seed=jitter_seed, aperture=aperture, focal_distance=focal_distance
    )

    # Step 2: Trace primary rays (with instance_ids if geometry_colors provided)
    d_instance_ids = bufs.instance_ids
    d_primitive_ids = bufs.primitive_ids
    if geometry_colors is not None:
        optix.trace(d_primary_rays, d_primary_hits, num_rays,
                    instance_ids=d_instance_ids, primitive_ids=d_primitive_ids)
    else:
        optix.trace(d_primary_rays, d_primary_hits, num_rays)

    # Step 3: Generate and trace shadow rays (if enabled)
    if shadows:
        _generate_shadow_rays_from_hits(
            d_shadow_rays, d_primary_rays, d_primary_hits, num_rays, d_sun_dir,
            sun_angle_rad=sun_angle_rad, shadow_seed=shadow_seed
        )
        optix.trace(d_shadow_rays, d_shadow_hits, num_rays,
                    ray_flags=RTX.RAY_FLAG_OCCLUSION)
    else:
        # Fill shadow hits with -1 (no shadow)
        d_shadow_hits.fill(-1)

    # Step 3b: Ambient occlusion pass
    d_ao_factor = None
    if ao_samples > 0:
        d_ao_rays = bufs.ao_rays
        d_ao_hits = bufs.ao_hits

        # Auto-compute AO radius from scene extent if not specified
        if ao_radius is None:
            H_raster, W_raster = raster.shape
            diagonal = math.sqrt((H_raster * pixel_spacing_y) ** 2 +
                                 (W_raster * pixel_spacing_x) ** 2)
            ao_radius = diagonal * 0.05

        d_ao_factor = cupy.ones(num_rays, dtype=np.float32)
        d_gi_color = bufs.gi_color
        d_gi_throughput = bufs.gi_throughput

        for s in range(ao_samples):
            sample_seed = ao_seed * ao_samples + s
            d_gi_throughput.fill(1.0)  # reset per-sample path throughput

            # Bounce 0 (existing flow)
            _generate_ao_rays(d_ao_rays, d_primary_rays, d_primary_hits,
                              num_rays, ao_radius, sample_seed)
            optix.trace(d_ao_rays, d_ao_hits, num_rays,
                        ray_flags=RTX.RAY_FLAG_OCCLUSION)
            _accumulate_ao(d_ao_factor, d_ao_hits, num_rays, ao_samples)
            _accumulate_gi(d_gi_color, d_ao_rays, d_ao_hits, num_rays,
                           ao_samples, vertical_exaggeration,
                           elev_min, elev_range, d_color_lut, stretch_int,
                           d_gi_throughput)

            # Additional bounces
            for bounce in range(1, gi_bounces):
                bounce_seed = sample_seed * 7919 + bounce * 6271
                # In-place: new AO rays from previous hit points
                _generate_ao_rays(d_ao_rays, d_ao_rays, d_ao_hits,
                                  num_rays, ao_radius, bounce_seed)
                optix.trace(d_ao_rays, d_ao_hits, num_rays,
                            ray_flags=RTX.RAY_FLAG_OCCLUSION)
                _accumulate_gi(d_gi_color, d_ao_rays, d_ao_hits, num_rays,
                               ao_samples, vertical_exaggeration,
                               elev_min, elev_range, d_color_lut, stretch_int,
                               d_gi_throughput)

    # Step 3c: Reflection rays for water surfaces and NaN ocean
    d_reflection_hits = None
    d_reflection_rays = None
    if bufs.reflection_rays is not None:
        d_reflection_rays = bufs.reflection_rays
        d_reflection_hits = bufs.reflection_hits
        # Use real geometry_colors or dummy for the kernel
        gc = geometry_colors
        if gc is None:
            global _DUMMY_1x4
            if _DUMMY_1x4 is None:
                _DUMMY_1x4 = cupy.zeros((1, 4), dtype=np.float32)
            gc = _DUMMY_1x4
        _generate_reflection_rays(
            d_reflection_rays, d_primary_rays, d_primary_hits,
            d_instance_ids, gc, num_rays,
            colormap_data, pixel_spacing_x, pixel_spacing_y
        )
        optix.trace(d_reflection_rays, d_reflection_hits, num_rays)

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


    # Get observer position for marker orb (sentinel = no observer placed)
    obs_x = float(observer_position[0]) if observer_position else -1e30
    obs_y = float(observer_position[1]) if observer_position else -1e30

    # Prepare overlay data for transparent blending
    d_overlay = None
    ov_min = 0.0
    ov_range = 1.0
    if overlay_data is not None:
        if not isinstance(overlay_data, cupy.ndarray):
            d_overlay = cupy.asarray(overlay_data, dtype=cupy.float32)
        else:
            d_overlay = overlay_data if overlay_data.dtype == cupy.float32 else overlay_data.astype(cupy.float32)
        ov_min = float(cupy.nanmin(d_overlay))
        ov_max = float(cupy.nanmax(d_overlay))
        ov_range = ov_max - ov_min

    # Prepare cloud fog map for GPU
    d_cloud_fog_map = None
    if cloud_fog_map is not None:
        if not isinstance(cloud_fog_map, cupy.ndarray):
            d_cloud_fog_map = cupy.asarray(cloud_fog_map, dtype=cupy.float32)
        else:
            d_cloud_fog_map = cloud_fog_map if cloud_fog_map.dtype == cupy.float32 else cloud_fog_map.astype(cupy.float32)

    # Build per-point color buffers for sphere geometries
    d_point_colors = None
    d_point_color_offsets = None
    if optix is not None:
        d_point_colors, d_point_color_offsets = optix.build_point_colors_gpu()

    # Step 4: Shade terrain
    _shade_terrain(
        d_output, d_primary_rays, d_primary_hits, d_shadow_hits,
        colormap_data, d_color_lut, num_rays, width, height,
        d_sun_dir, ambient, shadows,
        fog_density, fog_color,
        elev_min, elev_range, alpha,
        d_viewshed, viewshed_opacity,
        obs_x, obs_y,
        pixel_spacing_x, pixel_spacing_y,
        stretch_int,
        sky_color=(-1.0, 0.0, 0.0) if sky_color is None else sky_color,
        rgb_texture=rgb_texture,
        rgb_texture_offset_y=rgb_texture_offset_y,
        rgb_texture_offset_x=rgb_texture_offset_x,
        overlay_data=d_overlay, overlay_alpha=overlay_alpha,
        overlay_min=ov_min, overlay_range=ov_range,
        overlay_as_water=overlay_as_water,
        overlay_color_lut=overlay_color_lut,
        overlay_offset_y=overlay_offset_y,
        overlay_offset_x=overlay_offset_x,
        instance_ids=d_instance_ids, geometry_colors=geometry_colors,
        primitive_ids=d_primitive_ids,
        point_colors=d_point_colors,
        point_color_offsets=d_point_color_offsets,
        ao_factor=d_ao_factor,
        gi_color=bufs.gi_color if ao_samples > 0 else None,
        gi_intensity=gi_intensity,
        reflection_hits=d_reflection_hits, reflection_rays=d_reflection_rays,
        albedo_out=bufs.albedo,
        cloud_fog_map=None if volumetric_clouds else d_cloud_fog_map,
        cloud_fog_density=cloud_fog_density,
    )

    # Volumetric clouds (ray-marched, composited over shaded frame)
    if volumetric_clouds and cloud_top_z > cloud_base_z:
        d_primary_rays_flat = bufs.primary_rays
        d_primary_hits_flat = bufs.primary_hits
        _apply_volumetric_clouds(
            d_output, d_primary_rays_flat, d_primary_hits_flat,
            d_cloud_fog_map, d_sun_dir,
            width, height,
            cloud_base_z, cloud_top_z,
            pixel_spacing_x, pixel_spacing_y,
            H, W,
            cloud_time,
        )

    # AI denoiser (after shading, before bloom/tone mapping)
    if denoise:
        from ..rtx import denoise as _denoise
        d_normals = d_primary_hits.reshape(height, width, 4)[:, :, 1:4].copy()
        _denoise(d_output, d_normals, width, height, right, cam_up, forward,
                 albedo=bufs.albedo)

    # Edge outlines on placed geometry (after denoise, before bloom)
    if edge_lines and geometry_colors is not None:
        _edge_outline(d_output, d_instance_ids, edge_strength, edge_color)

    # Eye Dome Lighting (depth-edge enhancement, especially for point clouds)
    if edl:
        d_depth_1d = d_primary_hits.reshape(num_rays, 4)[:, 0].copy()
        _edl(d_output, d_depth_1d, width, height,
             radius=edl_radius, strength=edl_strength)

    # Bloom post-process (before tone mapping so ACES compresses bloom gracefully)
    if bloom:
        _bloom(d_output, bufs.bloom_temp, bufs.bloom_scratch)

    # Tone mapping (ACES filmic curve)
    if tone_map:
        _tone_map_aces(d_output)

    cupy.cuda.Stream.null.synchronize()

    if _return_gpu:
        return d_output

    # Transfer to CPU
    output = cupy.asnumpy(d_output)

    # Save image if requested
    if output_path is not None:
        _save_image(output, output_path)

    return output
