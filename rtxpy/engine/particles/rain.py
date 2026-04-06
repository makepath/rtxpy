"""Rain particle advection and rendering.

Rain particles fall vertically (z_frac decreasing from ~1.0 to 0.0)
and render as vertical streaks rather than dot trails.  They use the
shared ``ParticleSystem`` for lifecycle but carry extra ``z_frac``
state for the altitude interpolation.
"""

import numpy as np

from . import project


# ------------------------------------------------------------------
# Advection
# ------------------------------------------------------------------

def advect_rain(system, z_frac, wind_u=None, wind_v=None, dt_scale=1.0):
    """Descend rain particles and apply light wind drift.

    Parameters
    ----------
    system : ParticleSystem
    z_frac : ndarray (N,) float32
        Altitude fraction: 1.0 = cloud top, 0.0 = terrain surface.
        Modified in-place.
    wind_u, wind_v : ndarray (H, W), optional
        Wind velocity fields for horizontal drift (15% strength).
    """
    z_frac -= 0.06
    system.ages += 1

    pts = system.positions
    if wind_u is not None and wind_v is not None:
        H, W = wind_u.shape
        r0 = np.clip(pts[:, 0].astype(int), 0, H - 1)
        c0 = np.clip(pts[:, 1].astype(int), 0, W - 1)
        pts[:, 1] += wind_u[r0, c0] * 0.15 * dt_scale
        pts[:, 0] += wind_v[r0, c0] * 0.15 * dt_scale

    return z_frac


def respawn_rain(system, z_frac, rain_grid):
    """Respawn rain particles that hit the ground or aged out.

    Parameters
    ----------
    system : ParticleSystem
    z_frac : ndarray (N,) — modified in-place
    rain_grid : ndarray (H, W) — precipitation weights for spawn

    Returns
    -------
    z_frac : ndarray (N,) — updated
    """
    H, W = system.grid_shape
    pts = system.positions
    respawn = (
        (z_frac <= 0)
        | (system.ages >= system.lifetimes)
        | (pts[:, 0] < 0) | (pts[:, 0] >= H)
        | (pts[:, 1] < 0) | (pts[:, 1] >= W)
    )
    n_respawn = int(respawn.sum())
    if n_respawn == 0:
        return z_frac

    rain_flat = rain_grid.ravel()
    rain_sum = rain_flat.sum()
    if rain_sum > 0:
        prob = rain_flat / rain_sum
    else:
        prob = np.ones_like(rain_flat) / rain_flat.size
    chosen = np.random.choice(rain_flat.size, size=n_respawn, p=prob)
    pts[respawn, 0] = (chosen // W).astype(np.float32) + \
        np.random.uniform(-0.5, 0.5, n_respawn).astype(np.float32)
    pts[respawn, 1] = (chosen % W).astype(np.float32) + \
        np.random.uniform(-0.5, 0.5, n_respawn).astype(np.float32)
    z_frac[respawn] = np.random.uniform(0.7, 1.0, n_respawn).astype(np.float32)
    system.ages[respawn] = 0
    system.lifetimes[respawn] = np.random.randint(20, 40, n_respawn).astype(np.int32)

    return z_frac


# ------------------------------------------------------------------
# CPU Rendering
# ------------------------------------------------------------------

RAIN_COLOR = np.array([0.7, 0.75, 0.85], dtype=np.float32)


def render_rain_cpu(system, z_frac, img, cam, terrain_np,
                    pixel_spacing_x, pixel_spacing_y,
                    ve, subsample, cloud_altitude, min_depth,
                    rain_grid=None):
    """Project rain particles and render as vertical streaks.

    Parameters
    ----------
    system : ParticleSystem
    z_frac : ndarray (N,) — altitude fractions
    img : ndarray (H, W, 3), float32 — modified in-place
    cam : dict from ``project.get_camera_basis()``
    terrain_np : ndarray (tH, tW)
    rain_grid : ndarray, optional — for intensity-based alpha
    """
    sh, sw = img.shape[:2]
    psx, psy = pixel_spacing_x, pixel_spacing_y
    pts = system.positions
    N = pts.shape[0]
    cloud_z = cloud_altitude * ve
    f = subsample

    # Terrain Z and rain altitude
    tH, tW = terrain_np.shape
    sr = np.clip((pts[:, 0] / f).astype(int), 0, tH - 1)
    sc = np.clip((pts[:, 1] / f).astype(int), 0, tW - 1)
    terrain_z = np.nan_to_num(terrain_np[sr, sc], nan=0.0) * ve
    rain_z = terrain_z + z_frac * (cloud_z - terrain_z)

    # Project
    world_xy = np.column_stack([pts[:, 1] * psx, pts[:, 0] * psy])
    sx, sy, depth, on_screen = project.project_to_screen(
        world_xy, rain_z, cam, sh, sw + 8)  # allow slight off-screen for streaks
    on_screen &= depth > min_depth

    # Streak length based on depth
    inv_depth = np.where(depth > min_depth, 1.0 / (depth + 1e-10), 0.0)
    streak_len = np.clip(
        (5.0 * inv_depth * sh * cam['fov_scale'] * 0.003), 2, 8
    ).astype(np.int32)

    # Alpha from precipitation intensity
    if rain_grid is not None:
        local_precip = rain_grid[
            np.clip(pts[:, 0].astype(int), 0, rain_grid.shape[0] - 1),
            np.clip(pts[:, 1].astype(int), 0, rain_grid.shape[1] - 1),
        ]
        base_alpha = np.clip(local_precip / 5.0, 0.05, 0.5).astype(np.float32)
    else:
        base_alpha = np.full(N, 0.15, dtype=np.float32)

    ages = system.ages.astype(np.float32)
    lifetimes = system.lifetimes.astype(np.float32)
    fade = np.clip(ages / 3.0, 0, 1) * np.clip((lifetimes - ages) / 5.0, 0, 1)
    alpha = base_alpha * fade * 0.15

    # Clamp screen coords for splatting
    on_screen &= (sx >= 0) & (sx < sw) & (alpha > 0.002)
    if not on_screen.any():
        return

    project.splat_streaks_cpu(img, sx[on_screen], sy[on_screen],
                              alpha[on_screen], streak_len[on_screen],
                              RAIN_COLOR)
    np.clip(img, 0, 1, out=img)


# ------------------------------------------------------------------
# GPU helpers
# ------------------------------------------------------------------

def prepare_rain_gpu_buffers(system, z_frac, cam, pixel_spacing_x,
                             pixel_spacing_y, ve, cloud_altitude,
                             min_depth, rain_grid=None):
    """Pre-compute alpha and streak lengths for GPU kernel upload.

    Returns (rain_pts, z_frac, alpha, streak_lens) as numpy arrays.
    """
    pts = system.positions
    N = pts.shape[0]
    psx, psy = pixel_spacing_x, pixel_spacing_y
    cloud_z = float(cloud_altitude * ve)

    if rain_grid is not None:
        local_precip = rain_grid[
            np.clip(pts[:, 0].astype(int), 0, rain_grid.shape[0] - 1),
            np.clip(pts[:, 1].astype(int), 0, rain_grid.shape[1] - 1),
        ]
        base_alpha = np.clip(local_precip / 5.0, 0.05, 0.5).astype(np.float32)
    else:
        base_alpha = np.full(N, 0.15, dtype=np.float32)

    ages = system.ages.astype(np.float32)
    lifetimes = system.lifetimes.astype(np.float32)
    fade = np.clip(ages / 3.0, 0, 1) * np.clip((lifetimes - ages) / 5.0, 0, 1)
    alpha = (base_alpha * fade * 0.15).astype(np.float32)

    # Streak length from approximate depth
    forward = cam['forward']
    cam_pos = cam['cam_pos']
    sh = cam.get('screen_h', 300)
    fov_scale = cam['fov_scale']
    r_wx = pts[:, 1] * psx
    r_wy = pts[:, 0] * psy
    r_depth = ((r_wx - cam_pos[0]) * forward[0]
               + (r_wy - cam_pos[1]) * forward[1]
               + (cloud_z * 0.5 - cam_pos[2]) * forward[2])
    r_inv = np.where(r_depth > min_depth, 1.0 / (r_depth + 1e-10), 0.0)
    streak = np.clip((5.0 * r_inv * sh * fov_scale * 0.003), 2, 8).astype(np.int32)

    return pts.copy(), z_frac.copy(), alpha, streak
