"""Shared projection and splatting primitives for all particle types.

These are the ~200 lines of camera math and CPU splatting that were
duplicated identically across wind, rain, and hydro renderers.
"""

import math

import numpy as np


def get_camera_basis(position, look_at, fov, screen_w, screen_h):
    """Compute camera basis vectors and projection scalars.

    Returns
    -------
    dict with keys: forward, right, up, fov_scale, aspect, cam_pos
    """
    from ...analysis.render import _compute_camera_basis
    forward, right, cam_up = _compute_camera_basis(
        tuple(position), tuple(look_at), (0, 0, 1),
    )
    fov_scale = math.tan(math.radians(fov) / 2.0)
    aspect = screen_w / screen_h
    return dict(
        forward=forward, right=right, up=cam_up,
        fov_scale=fov_scale, aspect=aspect, cam_pos=position,
    )


def project_to_screen(world_xy, z_vals, cam, screen_h, screen_w):
    """Project world-space points to screen-space pixel coordinates.

    Parameters
    ----------
    world_xy : ndarray, shape (M, 2)
        World-space (x, y) positions.
    z_vals : ndarray, shape (M,)
        World-space Z (elevation) per point.
    cam : dict
        Camera basis from ``get_camera_basis()``.
    screen_h, screen_w : int
        Screen dimensions in pixels.

    Returns
    -------
    sx, sy : ndarray (M,) int32
        Screen pixel coordinates.
    depth : ndarray (M,) float32
        Depth along camera forward axis.
    valid : ndarray (M,) bool
        True where depth > 0 and point is on screen.
    """
    forward = cam['forward']
    right = cam['right']
    up = cam['up']
    cam_pos = cam['cam_pos']
    fov_scale = cam['fov_scale']
    aspect = cam['aspect']

    dx = world_xy[:, 0] - cam_pos[0]
    dy = world_xy[:, 1] - cam_pos[1]
    dz = z_vals - cam_pos[2]

    depth = dx * forward[0] + dy * forward[1] + dz * forward[2]
    positive = depth > 0
    inv_depth = np.where(positive, 1.0 / (depth + 1e-10), 0.0)

    u_cam = dx * right[0] + dy * right[1] + dz * right[2]
    v_cam = dx * up[0] + dy * up[1] + dz * up[2]
    u_ndc = u_cam * inv_depth / (fov_scale * aspect)
    v_ndc = v_cam * inv_depth / fov_scale

    sx = np.nan_to_num(((u_ndc + 1.0) * 0.5 * screen_w), nan=-1.0).astype(np.int32)
    sy = np.nan_to_num(((1.0 - v_ndc) * 0.5 * screen_h), nan=-1.0).astype(np.int32)

    on_screen = (positive
                 & (sx >= 0) & (sx < screen_w)
                 & (sy >= 0) & (sy < screen_h))

    return sx, sy, depth, on_screen


def terrain_z_lookup(rows, cols, terrain_np, subsample, ve, z_offset=0.0):
    """Sample terrain Z at pixel positions.

    Parameters
    ----------
    rows, cols : ndarray (M,)
        Particle positions in full-res pixel coords.
    terrain_np : ndarray (tH, tW)
        CPU terrain elevation array (subsampled resolution).
    subsample : int
        Subsample factor (rows/cols are divided by this).
    ve : float
        Vertical exaggeration.
    z_offset : float
        Constant added after VE scaling (e.g. +3.0 for wind).
    """
    tH, tW = terrain_np.shape
    sr = np.clip(np.nan_to_num(rows / subsample, nan=0.0).astype(np.int32), 0, tH - 1)
    sc = np.clip(np.nan_to_num(cols / subsample, nan=0.0).astype(np.int32), 0, tW - 1)
    return np.nan_to_num(terrain_np[sr, sc], nan=0.0) * ve + z_offset


def compute_trail_alpha(ages, lifetimes, trail_len, n_particles,
                        base_alpha, min_visible_age,
                        fade_mode='linear', head_glow=1.0,
                        depth=None, ref_depth=None):
    """Compute per-trail-point alpha for N particles with T trail points.

    Returns
    -------
    alpha : ndarray, shape (N * trail_len,)
    age_ok : ndarray, shape (N * trail_len,), bool
    """
    T = trail_len
    N = n_particles

    trail_idx = np.tile(np.arange(T, dtype=np.float32), N)
    ages_rep = np.repeat(ages, T)
    lifetimes_rep = np.repeat(lifetimes, T)

    # Particle must be old enough for this trail slot to be valid
    age_ok = ages_rep > trail_idx

    # Fade in/out
    fade_in = np.clip((ages_rep - min_visible_age) / 10.0, 0, 1)
    fade_out = np.clip((lifetimes_rep - ages_rep) / 20.0, 0, 1)

    # Trail decay
    if fade_mode == 'quadratic':
        trail_fade = (1.0 - trail_idx / T) ** 2
    else:
        trail_fade = 1.0 - trail_idx / T

    alpha = base_alpha * fade_in * fade_out * trail_fade

    # Optional depth-based attenuation
    if depth is not None and ref_depth is not None:
        depth_rep = np.repeat(depth, T) if len(depth) == N else depth
        alpha *= ref_depth / (depth_rep + ref_depth)

    # Head glow
    if head_glow != 1.0:
        is_head = trail_idx == 0
        alpha = np.where(is_head, alpha * head_glow, alpha)

    alpha[~age_ok] = 0.0
    return alpha, age_ok


def splat_dots_cpu(img, sx, sy, alphas, color, radius):
    """Splat circular dots onto a CPU frame buffer.

    Parameters
    ----------
    img : ndarray (H, W, 3), float32
        Frame buffer, modified in-place.
    sx, sy : ndarray (M,), int32
        Screen positions of visible points.
    alphas : ndarray (M,), float32
        Per-point alpha.
    color : ndarray (3,) or (M, 3), float32
        Constant or per-point RGB color.
    radius : int
        Splat radius in pixels.
    """
    sh, sw = img.shape[:2]
    per_particle_color = color.ndim == 2

    if per_particle_color:
        # Group by radius to avoid per-particle inner loop
        _splat_dots_colored(img, sx, sy, alphas, color, radius, sh, sw)
    else:
        for offy in range(-radius, radius + 1):
            for offx in range(-radius, radius + 1):
                dist_sq = offx * offx + offy * offy
                if dist_sq > radius * radius:
                    continue
                falloff = 1.0 - (dist_sq / (radius * radius)) ** 0.5

                px = sx + offx
                py = sy + offy
                ok = (px >= 0) & (px < sw) & (py >= 0) & (py < sh)
                if not ok.any():
                    continue

                contribution = alphas[ok] * falloff
                for c in range(3):
                    np.add.at(img[:, :, c], (py[ok], px[ok]),
                              contribution * color[c])


def _splat_dots_colored(img, sx, sy, alphas, colors, radius, sh, sw):
    """Splat with per-point color (used by hydro)."""
    for offy in range(-radius, radius + 1):
        for offx in range(-radius, radius + 1):
            dist_sq = offx * offx + offy * offy
            if dist_sq > radius * radius:
                continue
            falloff = 1.0 - (dist_sq / (radius * radius)) ** 0.5

            px = sx + offx
            py = sy + offy
            ok = (px >= 0) & (px < sw) & (py >= 0) & (py < sh)
            if not ok.any():
                continue

            contribution = alphas[ok] * falloff
            for c in range(3):
                np.add.at(img[:, :, c], (py[ok], px[ok]),
                          contribution * colors[ok, c])


def splat_streaks_cpu(img, sx, sy, alphas, streak_lens, color):
    """Splat vertical streaks onto a CPU frame buffer (for rain).

    Parameters
    ----------
    img : ndarray (H, W, 3), float32
    sx, sy : ndarray (M,), int32
    alphas : ndarray (M,), float32
    streak_lens : ndarray (M,), int32
    color : ndarray (3,), float32
    """
    sh, sw = img.shape[:2]
    max_sl = int(streak_lens.max()) if streak_lens.size > 0 else 3
    for dy_off in range(max_sl):
        py = sy + dy_off
        streak_ok = (dy_off < streak_lens) & (py >= 0) & (py < sh)
        if not streak_ok.any():
            continue
        t = dy_off / (streak_lens[streak_ok].astype(np.float32))
        streak_alpha = alphas[streak_ok] * (1.0 - t * 0.6)
        for c in range(3):
            np.add.at(img[:, :, c], (py[streak_ok], sx[streak_ok]),
                      streak_alpha * color[c])
