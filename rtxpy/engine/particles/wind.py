"""Wind particle advection and rendering.

Free functions operating on a ``ParticleSystem`` — no viewer references.
"""

import numpy as np

from . import project


# ------------------------------------------------------------------
# Advection
# ------------------------------------------------------------------

def advect_wind(system, wind_u, wind_v, dt_scale,
                slope_col=None, slope_row=None):
    """Advect wind particles one tick via bilinear field sampling.

    Parameters
    ----------
    system : ParticleSystem
    wind_u : ndarray (H, W)  — eastward velocity in pixels/tick
    wind_v : ndarray (H, W)  — northward velocity in pixels/tick (row direction)
    dt_scale : float          — frame-rate-independent scaling factor
    slope_col, slope_row : ndarray (H, W), optional
        Terrain downslope push fields.
    """
    pts = system.positions
    H, W = wind_u.shape

    # Bilinear sample wind at particle positions
    rows = pts[:, 0]
    cols = pts[:, 1]
    r0 = np.clip(np.floor(np.nan_to_num(rows, nan=0.0)).astype(int), 0, H - 2)
    c0 = np.clip(np.floor(np.nan_to_num(cols, nan=0.0)).astype(int), 0, W - 2)
    fr = rows - r0
    fc = cols - c0

    u_val = _bilerp(wind_u, r0, c0, fr, fc)
    v_val = _bilerp(wind_v, r0, c0, fr, fc)

    # Terrain slope influence
    if slope_col is not None and slope_row is not None:
        slope_u = _bilerp(slope_col, r0, c0, fr, fc)
        slope_v = _bilerp(slope_row, r0, c0, fr, fc)

        slope_mag = np.sqrt(slope_u**2 + slope_v**2) + 1e-8
        wind_mag = np.sqrt(u_val**2 + v_val**2) + 1e-8
        alignment = (u_val * slope_u + v_val * slope_v) / (wind_mag * slope_mag)
        dampen = np.clip(1.0 - alignment, 0.2, 1.0)

        u_val += slope_u * dampen
        v_val += slope_v * dampen

    pts[:, 0] += v_val * dt_scale
    pts[:, 1] += u_val * dt_scale


def _bilerp(grid, r0, c0, fr, fc):
    """Bilinear interpolation at integer corners + fractional offsets."""
    return (grid[r0, c0] * (1 - fr) * (1 - fc)
            + grid[r0, c0 + 1] * (1 - fr) * fc
            + grid[r0 + 1, c0] * fr * (1 - fc)
            + grid[r0 + 1, c0 + 1] * fr * fc)


# ------------------------------------------------------------------
# CPU Rendering
# ------------------------------------------------------------------

WIND_COLOR = np.array([0.3, 0.9, 0.8], dtype=np.float32)


def render_wind_cpu(system, img, cam, terrain_np,
                    pixel_spacing_x, pixel_spacing_y,
                    ve, subsample, min_depth,
                    base_alpha=0.6, min_visible_age=5,
                    dot_radius=2):
    """Project wind trails to screen and splat as teal dots.

    Parameters
    ----------
    system : ParticleSystem (must have trails)
    img : ndarray (H, W, 3), float32 — modified in-place
    cam : dict from ``project.get_camera_basis()``
    terrain_np : ndarray (tH, tW) — CPU terrain
    """
    if system.positions is None or system.trails is None:
        return

    sh, sw = img.shape[:2]
    N = system.n
    T = system.trail_len
    psx, psy = pixel_spacing_x, pixel_spacing_y

    # Flatten trails: (N, T, 2) -> (N*T, 2)
    all_pts = system.trails.reshape(-1, 2)
    rows_all = all_pts[:, 0]
    cols_all = all_pts[:, 1]

    # Terrain Z lookup + world projection
    z_vals = project.terrain_z_lookup(rows_all, cols_all, terrain_np,
                                      subsample, ve, z_offset=3.0)
    world_xy = np.column_stack([cols_all * psx, rows_all * psy])
    sx, sy, depth, on_screen = project.project_to_screen(
        world_xy, z_vals, cam, sh, sw)

    # Filter by min depth
    on_screen &= depth > min_depth

    # Alpha
    alpha, age_ok = project.compute_trail_alpha(
        system.ages, system.lifetimes, T, N,
        base_alpha, min_visible_age, fade_mode='linear')

    mask = on_screen & age_ok & (alpha > 1e-6)
    if not mask.any():
        return

    project.splat_dots_cpu(img, sx[mask], sy[mask], alpha[mask].astype(np.float32),
                           WIND_COLOR, dot_radius)
    np.clip(img, 0, 1, out=img)


# ------------------------------------------------------------------
# GPU Rendering
# ------------------------------------------------------------------

def prepare_wind_gpu_buffers(system, base_alpha, min_visible_age):
    """Pre-compute alpha and flatten trails for GPU kernel upload.

    Returns (all_pts, alpha) as numpy arrays ready for GPU upload.
    """
    N = system.n
    T = system.trail_len

    alpha, _age_ok = project.compute_trail_alpha(
        system.ages, system.lifetimes, T, N,
        base_alpha, min_visible_age, fade_mode='linear')

    all_pts = system.trails.reshape(-1, 2).astype(np.float32)
    return all_pts, alpha.astype(np.float32)
