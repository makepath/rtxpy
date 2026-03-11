"""CUDA kernels for hydrological flow particle simulation.

Three Numba CUDA kernels:
- ``hydro_splat_kernel``: project particles to screen space and splat
- ``hydro_advect_kernel``: bilinear flow lookup, advection, respawn detection
- ``hydro_respawn_kernel``: reset respawned particles

Separated from engine.py for cleaner organisation.  The kernels are
compiled once on first use by Numba's JIT.
"""

import math

from numba import cuda


@cuda.jit
def hydro_splat_kernel(
    trails,       # (N*T, 2) float32 — (row, col) per trail point
    ages,         # (N,) int32 — per-particle age
    lifetimes,    # (N,) int32 — per-particle lifetime
    colors,       # (N, 3) float32 — per-particle (r, g, b)
    radii,        # (N,) int32 — per-particle splat radius
    trail_len,    # int32 scalar — trail points per particle
    base_alpha,   # float32 scalar — base alpha intensity
    min_vis_age,  # int32 scalar — minimum visible age
    ref_depth,    # float32 scalar — depth-scaling reference distance
    terrain,      # (tH, tW) float32 — terrain elevation
    depth_t,      # (sh, sw) float32 — ray-trace t-values for occlusion
    output,       # (sh, sw, 3) float32 — frame buffer (atomic add)
    # Camera basis — scalar args to avoid tiny GPU allocations
    cam_x, cam_y, cam_z,
    fwd_x, fwd_y, fwd_z,
    rgt_x, rgt_y, rgt_z,
    up_x, up_y, up_z,
    # Projection params
    fov_scale, aspect_ratio,
    # Terrain/world params
    psx, psy, ve, subsample_f, min_depth, max_depth,
):
    idx = cuda.grid(1)
    if idx >= trails.shape[0]:
        return

    # Compute alpha on-GPU from per-particle ages/lifetimes
    pidx = idx // trail_len
    tidx = idx % trail_len
    age = ages[pidx]
    lifetime = lifetimes[pidx]

    # Trail point not yet laid down
    if age <= tidx:
        return

    # Fade in / fade out / trail decay
    fade_in = (age - min_vis_age) * 0.1
    if fade_in < 0.0:
        fade_in = 0.0
    elif fade_in > 1.0:
        fade_in = 1.0
    fade_out = (lifetime - age) * 0.05
    if fade_out < 0.0:
        fade_out = 0.0
    elif fade_out > 1.0:
        fade_out = 1.0
    # Quadratic trail decay — comet-tail effect
    t = float(tidx) / float(trail_len)
    trail_fade = (1.0 - t) * (1.0 - t)
    a = base_alpha * fade_in * fade_out * trail_fade

    # Head glow: bright spark at particle position
    if tidx == 0:
        a = a * 1.5

    if a < 1e-6:
        return

    row = trails[idx, 0]
    col = trails[idx, 1]

    # Terrain Z lookup (nearest-neighbor, clamped)
    tH = terrain.shape[0]
    tW = terrain.shape[1]
    sr = int(row / subsample_f)
    sc = int(col / subsample_f)
    if sr < 0:
        sr = 0
    elif sr >= tH:
        sr = tH - 1
    if sc < 0:
        sc = 0
    elif sc >= tW:
        sc = tW - 1
    z_raw = terrain[sr, sc]
    if z_raw != z_raw:  # NaN check
        z_raw = 0.0
    z_val = z_raw * ve + 3.0

    # World position
    wx = col * psx
    wy = row * psy

    # Camera-relative
    dx = wx - cam_x
    dy = wy - cam_y
    dz = z_val - cam_z

    # Depth along forward axis
    depth = dx * fwd_x + dy * fwd_y + dz * fwd_z
    if depth <= min_depth:
        return
    if max_depth > 0.0 and depth > max_depth:
        return

    # Depth-scaled alpha: closer = brighter, farther = fainter.
    # Prevents zoomed-out over-saturation from dense overlapping particles.
    depth_scale = ref_depth / (depth + ref_depth)
    a = a * depth_scale

    if a < 1e-6:
        return

    inv_depth = 1.0 / (depth + 1e-10)
    u_cam = dx * rgt_x + dy * rgt_y + dz * rgt_z
    v_cam = dx * up_x + dy * up_y + dz * up_z
    u_ndc = u_cam * inv_depth / (fov_scale * aspect_ratio)
    v_ndc = v_cam * inv_depth / fov_scale

    sh = output.shape[0]
    sw = output.shape[1]
    sx = int((u_ndc + 1.0) * 0.5 * sw)
    sy = int((1.0 - v_ndc) * 0.5 * sh)

    if sx < 0 or sx >= sw or sy < 0 or sy >= sh:
        return

    # Depth test: cull particles occluded by terrain.
    # Convert ray t-value at this pixel to forward depth, then compare
    # to the particle's forward depth (already computed as `depth`).
    if depth_t.shape[0] > 0:
        t_val = depth_t[sy, sx]
        if t_val > 0.0 and t_val < 1.0e20:
            # Forward depth = t / sqrt(1 + u_cam^2 + v_cam^2)
            u_px = (2.0 * float(sx) / float(sw) - 1.0) * fov_scale * aspect_ratio
            v_px = (1.0 - 2.0 * float(sy) / float(sh)) * fov_scale
            inv_cos = math.sqrt(1.0 + u_px * u_px + v_px * v_px)
            terrain_fwd = t_val / inv_cos
            if depth > terrain_fwd:
                return

    # Per-particle color and radius
    color_r = colors[pidx, 0]
    color_g = colors[pidx, 1]
    color_b = colors[pidx, 2]
    r = radii[pidx]
    if r < 1:
        r = 1
    # Head glow: +1px radius halo at particle position
    if tidx == 0:
        r = r + 1

    # Circular stamp splat
    for offy in range(-r, r + 1):
        for offx in range(-r, r + 1):
            dist_sq = offx * offx + offy * offy
            if dist_sq > r * r:
                continue
            falloff = 1.0 - math.sqrt(dist_sq) / r
            px = sx + offx
            py = sy + offy
            if px < 0 or px >= sw or py < 0 or py >= sh:
                continue
            contrib = a * falloff
            cuda.atomic.add(output, (py, px, 0), contrib * color_r)
            cuda.atomic.add(output, (py, px, 1), contrib * color_g)
            cuda.atomic.add(output, (py, px, 2), contrib * color_b)


@cuda.jit
def hydro_advect_kernel(
    # Particle state (GPU-resident, modified in-place)
    particles,      # (N, 2) float32 — (row, col) positions
    ages,           # (N,) int32
    lifetimes,      # (N,) int32
    trails,         # (N, T, 2) float32 — trail history
    particle_accum, # (N,) float32 — max-tracked stream weight
    particle_raw_order,  # (N,) int32 — max-tracked raw Strahler order
    colors,         # (N, 3) float32 — per-particle RGB
    radii,          # (N,) int32 — per-particle splat radius
    # Grid textures (GPU-resident, read-only)
    flow_u,         # (H, W) float32 — MFD flow col-component
    flow_v,         # (H, W) float32 — MFD flow row-component
    slope_mag,      # (H, W) float32 — normalized slope
    stream_order,   # (H, W) float32 — normalized stream order (or empty)
    stream_order_raw,  # (H, W) int32 — raw Strahler order (or empty)
    accum_norm,     # (H, W) float32 — normalized flow accumulation
    # Palette for color lookup (9, 3) float32
    palette,        # (9, 3) float32 — stream order color palette
    # Output: respawn flags
    respawn_flags,  # (N,) int32 — 1 if particle needs respawn
    # Scalar params
    speed, dt_scale, trail_len,
    has_so,         # int32: 1 if stream_order is valid
    has_slope,      # int32: 1 if slope_mag is valid
    has_raw_order,  # int32: 1 if stream_order_raw / particle_raw_order valid
    # Window offset for streaming: particle coords are global,
    # flow field covers a window starting at (win_r0, win_c0)
    win_r0, win_c0,
    # RNG seed
    rng_base,       # int64 — base seed for per-particle RNG
):
    """Advect one hydro particle: bilinear flow lookup, trail shift, respawn detection."""
    i = cuda.grid(1)
    N = particles.shape[0]
    if i >= N:
        return

    H = flow_u.shape[0]
    W = flow_u.shape[1]

    row = particles[i, 0]
    col = particles[i, 1]

    # Shift trail buffer: slot 0 = current pos (before advection)
    t = trail_len - 1
    while t > 0:
        trails[i, t, 0] = trails[i, t - 1, 0]
        trails[i, t, 1] = trails[i, t - 1, 1]
        t -= 1
    trails[i, 0, 0] = row
    trails[i, 0, 1] = col

    # Map global particle position to local flow field coordinates
    local_r = row - win_r0
    local_c = col - win_c0

    # Bilinear interpolation of MFD flow vectors
    r_clean = local_r
    c_clean = local_c
    if r_clean != r_clean:
        r_clean = 0.0
    if c_clean != c_clean:
        c_clean = 0.0
    if r_clean < 0.0:
        r_clean = 0.0
    elif r_clean > H - 1.0:
        r_clean = H - 1.0
    if c_clean < 0.0:
        c_clean = 0.0
    elif c_clean > W - 1.0:
        c_clean = W - 1.0

    r0 = int(r_clean)
    c0 = int(c_clean)
    if r0 > H - 2:
        r0 = H - 2
    if c0 > W - 2:
        c0 = W - 2
    if r0 < 0:
        r0 = 0
    if c0 < 0:
        c0 = 0
    r1 = r0 + 1
    c1 = c0 + 1

    dr = r_clean - float(r0)
    dc = c_clean - float(c0)
    w00 = (1.0 - dr) * (1.0 - dc)
    w01 = (1.0 - dr) * dc
    w10 = dr * (1.0 - dc)
    w11 = dr * dc

    u_val = (flow_u[r0, c0] * w00 + flow_u[r0, c1] * w01 +
             flow_u[r1, c0] * w10 + flow_u[r1, c1] * w11)
    v_val = (flow_v[r0, c0] * w00 + flow_v[r0, c1] * w01 +
             flow_v[r1, c0] * w10 + flow_v[r1, c1] * w11)

    # Integer indices for grid lookups
    ri = int(r_clean)
    ci = int(c_clean)
    if ri > H - 1:
        ri = H - 1
    if ci > W - 1:
        ci = W - 1
    if ri < 0:
        ri = 0
    if ci < 0:
        ci = 0

    # Max-track stream order / accumulation
    if has_so:
        cur_val = stream_order[ri, ci]
    else:
        cur_val = accum_norm[ri, ci]
    old_val = particle_accum[i]
    if cur_val > old_val:
        particle_accum[i] = cur_val
        # Update raw order
        if has_raw_order:
            cur_raw = stream_order_raw[ri, ci]
            if cur_raw > particle_raw_order[i]:
                particle_raw_order[i] = cur_raw
        # Recompute color + radius from new weight
        if has_raw_order:
            raw_o = particle_raw_order[i]
            idx = raw_o
            if idx < 1:
                idx = 1
            if idx > 8:
                idx = 8
            colors[i, 0] = palette[idx, 0]
            colors[i, 1] = palette[idx, 1]
            colors[i, 2] = palette[idx, 2]
            rad = raw_o + 1
            if rad < 2:
                rad = 2
            if rad > 5:
                rad = 5
            radii[i] = rad
        else:
            a_val = particle_accum[i]
            colors[i, 0] = 0.02 + a_val * 0.43
            colors[i, 1] = 0.10 + a_val * 0.65
            colors[i, 2] = 0.55 + a_val * 0.40
            rad = 2 + int(a_val * 3.0)
            if rad < 2:
                rad = 2
            if rad > 5:
                rad = 5
            radii[i] = rad

    # Simple xorshift64 RNG seeded per-particle per-frame
    s = rng_base * 2654435761 + i * 1442695040888963407
    s = s ^ (s >> 17)
    s = s * 6364136223846793005
    s = s ^ (s >> 31)
    # Two uniform floats in [-0.1, 0.1] for jitter
    jitter_r = ((s & 0xFFFF) / 65535.0 - 0.5) * 0.2
    s = s * 6364136223846793005 + 1
    s = s ^ (s >> 31)
    jitter_c = ((s & 0xFFFF) / 65535.0 - 0.5) * 0.2

    # Slope-based speed
    slope_f = 1.0
    if has_slope:
        slope_f = 0.3 + 0.7 * slope_mag[ri, ci]

    # Advect (in global coordinates — flow vectors are direction-only)
    particles[i, 0] = row + (v_val + jitter_r) * speed * dt_scale * slope_f
    particles[i, 1] = col + (u_val + jitter_c) * speed * dt_scale * slope_f

    # Age
    ages[i] = ages[i] + 1

    # Respawn detection: OOB (relative to flow window), aged-out, stuck
    new_r = particles[i, 0]
    new_c = particles[i, 1]
    new_lr = new_r - win_r0
    new_lc = new_c - win_c0
    is_nan = (new_r != new_r) or (new_c != new_c)
    is_oob = is_nan or new_lr < 0.0 or new_lr >= H or new_lc < 0.0 or new_lc >= W
    is_old = ages[i] >= lifetimes[i]
    is_stuck = (u_val * u_val + v_val * v_val) < 1e-6
    if is_oob or is_old or is_stuck:
        respawn_flags[i] = 1
    else:
        respawn_flags[i] = 0


@cuda.jit
def hydro_respawn_kernel(
    # Particle state (GPU-resident, modified in-place)
    particles,      # (N, 2) float32
    ages,           # (N,) int32
    lifetimes,      # (N,) int32
    trails,         # (N, T, 2) float32
    particle_accum, # (N,) float32
    particle_raw_order,  # (N,) int32
    colors,         # (N, 3) float32
    radii,          # (N,) int32
    # Respawn data (uploaded from CPU)
    respawn_indices,  # (M,) int32 — which particles to respawn
    spawn_rows,       # (M,) float32 — new row positions (global coords)
    spawn_cols,       # (M,) float32 — new col positions (global coords)
    new_lifetimes,    # (M,) int32
    # Grid lookups
    stream_order,     # (H, W) float32
    stream_order_raw, # (H, W) int32
    accum_norm,       # (H, W) float32
    palette,          # (9, 3) float32
    # Scalars
    trail_len, has_so, has_raw_order,
    # Window offset
    win_r0, win_c0,
):
    """Apply respawn: reset position, age, trails, color/radius for respawned particles."""
    m = cuda.grid(1)
    if m >= respawn_indices.shape[0]:
        return

    i = respawn_indices[m]
    new_r = spawn_rows[m]
    new_c = spawn_cols[m]
    H = stream_order.shape[0] if has_so else accum_norm.shape[0]
    W = stream_order.shape[1] if has_so else accum_norm.shape[1]

    particles[i, 0] = new_r
    particles[i, 1] = new_c
    ages[i] = 0
    lifetimes[i] = new_lifetimes[m]

    # Reset trails to new position
    for t in range(trail_len):
        trails[i, t, 0] = new_r
        trails[i, t, 1] = new_c

    # Look up stream weight at spawn point (local coords)
    ri = int(new_r - win_r0)
    ci = int(new_c - win_c0)
    if ri < 0:
        ri = 0
    if ri >= H:
        ri = H - 1
    if ci < 0:
        ci = 0
    if ci >= W:
        ci = W - 1

    if has_so:
        val = stream_order[ri, ci]
    else:
        val = accum_norm[ri, ci]
    particle_accum[i] = val

    if has_raw_order:
        raw_o = stream_order_raw[ri, ci]
        particle_raw_order[i] = raw_o
        idx = raw_o
        if idx < 1:
            idx = 1
        if idx > 8:
            idx = 8
        colors[i, 0] = palette[idx, 0]
        colors[i, 1] = palette[idx, 1]
        colors[i, 2] = palette[idx, 2]
        rad = raw_o + 1
        if rad < 2:
            rad = 2
        if rad > 5:
            rad = 5
        radii[i] = rad
    else:
        colors[i, 0] = 0.02 + val * 0.43
        colors[i, 1] = 0.10 + val * 0.65
        colors[i, 2] = 0.55 + val * 0.40
        rad = 2 + int(val * 3.0)
        if rad < 2:
            rad = 2
        if rad > 5:
            rad = 5
        radii[i] = rad
