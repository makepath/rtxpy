"""Hydro particle rendering helpers.

Hydro advection is GPU-first and handled by ``HydroManager`` in
``rtxpy/viewer/hydro_manager.py`` via CUDA kernels.  This module
provides the CPU rendering path and shared alpha computation that
were previously embedded in the engine hydro controller.
"""

import numpy as np

from . import project

from ...rtx import has_cupy
if has_cupy:
    import cupy as cp


# ------------------------------------------------------------------
# CPU Rendering
# ------------------------------------------------------------------

def render_hydro_cpu(system, img, cam, terrain_np,
                     pixel_spacing_x, pixel_spacing_y,
                     ve, subsample, min_depth,
                     base_alpha=0.6, min_visible_age=5,
                     ref_depth=500.0, head_glow=1.5,
                     particle_colors=None, particle_radii=None,
                     depth_buffer=None):
    """Project hydro trails to screen and splat with per-particle color.

    Parameters
    ----------
    system : ParticleSystem (must have trails)
    img : ndarray (H, W, 3), float32 — modified in-place
    cam : dict from ``project.get_camera_basis()``
    terrain_np : ndarray (tH, tW)
    particle_colors : ndarray (N, 3), optional — per-particle RGB
    particle_radii : ndarray (N,), optional — per-particle radius
    depth_buffer : ndarray (sH, sW), optional — ray-traced depth for occlusion
    """
    if system.positions is None or system.trails is None:
        return

    sh, sw = img.shape[:2]
    N = system.n
    T = system.trail_len
    psx, psy = pixel_spacing_x, pixel_spacing_y

    all_pts = system.trails.reshape(-1, 2)
    rows_all = all_pts[:, 0]
    cols_all = all_pts[:, 1]

    z_vals = project.terrain_z_lookup(rows_all, cols_all, terrain_np,
                                      subsample, ve, z_offset=3.0)
    world_xy = np.column_stack([cols_all * psx, rows_all * psy])
    sx, sy, depth, on_screen = project.project_to_screen(
        world_xy, z_vals, cam, sh, sw)
    on_screen &= depth > min_depth

    # Depth test against ray-traced terrain
    if depth_buffer is not None:
        depth_t_np = depth_buffer
        if has_cupy and hasattr(depth_t_np, 'get'):
            depth_t_np = depth_t_np.get()
        fov_scale = cam['fov_scale']
        aspect = cam['aspect']
        sx_c = np.clip(sx, 0, sw - 1)
        sy_c = np.clip(sy, 0, sh - 1)
        t_vals = depth_t_np[sy_c, sx_c]
        u_px = (2.0 * sx_c / sw - 1.0) * fov_scale * aspect
        v_px = (1.0 - 2.0 * sy_c / sh) * fov_scale
        inv_cos = np.sqrt(1.0 + u_px**2 + v_px**2)
        terrain_fwd = t_vals / inv_cos
        occluded = (t_vals > 0) & (t_vals < 1e20) & (depth > terrain_fwd)
        on_screen = on_screen & ~occluded

    # Per-particle depth for alpha computation
    particle_depth = depth.reshape(N, T)[:, 0]

    alpha, age_ok = project.compute_trail_alpha(
        system.ages, system.lifetimes, T, N,
        base_alpha, min_visible_age,
        fade_mode='quadratic', head_glow=head_glow,
        depth=particle_depth, ref_depth=ref_depth)

    mask = on_screen & age_ok & (alpha > 1e-6)
    if not mask.any():
        return

    sx_m = sx[mask]
    sy_m = sy[mask]
    alpha_m = alpha[mask].astype(np.float32)

    # Default color/radius if not provided
    if particle_colors is None:
        particle_colors = np.full((N, 3), [0.2, 0.6, 1.0], dtype=np.float32)
    if particle_radii is None:
        particle_radii = np.full(N, 3, dtype=np.int32)

    # Expand per-particle -> per-trail-point
    particle_idx = np.arange(N * T) // T
    pidx_m = particle_idx[mask]
    color_m = particle_colors[pidx_m]
    radii_m = particle_radii[pidx_m]

    # Head glow: +1px at trail head
    trail_idx = np.tile(np.arange(T, dtype=np.float32), N)
    is_head_m = trail_idx[mask] == 0
    radii_m = np.where(is_head_m, radii_m + 1, radii_m)

    # Splat by radius value to batch the inner loop
    for r_val in range(1, 7):
        r_mask = radii_m == r_val
        if not r_mask.any():
            continue
        project.splat_dots_cpu(img,
                               sx_m[r_mask], sy_m[r_mask],
                               alpha_m[r_mask], color_m[r_mask],
                               r_val)

    np.clip(img, 0, 1, out=img)
