"""OpenGL shaders and CUDA kernels for the interactive viewer."""

import math

from ..rtx import has_cupy

if has_cupy:
    from numba import cuda


# ---------------------------------------------------------------------------
# OpenGL shaders for fullscreen textured quad
# ---------------------------------------------------------------------------
_QUAD_VERT = """
#version 330
in vec2 in_pos;
in vec2 in_uv;
out vec2 v_uv;
void main() {
    gl_Position = vec4(in_pos, 0.0, 1.0);
    v_uv = in_uv;
}
"""

_QUAD_FRAG = """
#version 330
uniform sampler2D frame;
uniform sampler2D overlay;
uniform int has_overlay;
in vec2 v_uv;
out vec4 fragColor;
void main() {
    vec3 base = texture(frame, v_uv).rgb;
    if (has_overlay == 1) {
        vec4 ov = texture(overlay, v_uv);
        base = mix(base, ov.rgb, ov.a);
    }
    fragColor = vec4(base, 1.0);
}
"""


# ---------------------------------------------------------------------------
# Numba CUDA kernel for wind particle splatting
# ---------------------------------------------------------------------------
if has_cupy:
    @cuda.jit
    def _wind_splat_kernel(
        trails,       # (N*T, 2) float32 — (row, col) per trail point
        alphas,       # (N*T,) float32 — pre-computed alpha per trail point
        terrain,      # (tH, tW) float32 — terrain elevation
        output,       # (sh, sw, 3) float32 — frame buffer (atomic add)
        # Camera basis — scalar args to avoid tiny GPU allocations
        cam_x, cam_y, cam_z,
        fwd_x, fwd_y, fwd_z,
        rgt_x, rgt_y, rgt_z,
        up_x, up_y, up_z,
        # Projection params
        fov_scale, aspect_ratio,
        # Terrain/world params
        psx, psy, ve, subsample_f, min_depth,
        # Splat params
        dot_radius,
        # Wind color
        color_r, color_g, color_b,
    ):
        idx = cuda.grid(1)
        if idx >= trails.shape[0]:
            return

        a = alphas[idx]
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
        # Ocean/water pixels are NaN — map to 0 (matches CPU path)
        if z_raw != z_raw:  # NaN check (works under fast math)
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

        # Circular stamp splat
        r = dot_radius
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
    def _rain_splat_kernel(
        pts,          # (N, 2) float32 — (row, col) per rain particle
        z_frac,       # (N,) float32 — altitude fraction (0=ground, 1=cloud)
        alphas,       # (N,) float32 — pre-computed alpha
        streak_lens,  # (N,) int32 — vertical streak length in pixels
        terrain,      # (tH, tW) float32 — terrain elevation
        output,       # (sh, sw, 3) float32 — frame buffer (atomic add)
        # Camera basis
        cam_x, cam_y, cam_z,
        fwd_x, fwd_y, fwd_z,
        rgt_x, rgt_y, rgt_z,
        up_x, up_y, up_z,
        # Projection
        fov_scale, aspect_ratio,
        # World params
        psx, psy, ve, subsample_f, cloud_z, min_depth,
        # Color
        color_r, color_g, color_b,
    ):
        idx = cuda.grid(1)
        if idx >= pts.shape[0]:
            return

        a = alphas[idx]
        if a < 0.002:
            return

        row = pts[idx, 0]
        col = pts[idx, 1]

        # Terrain Z lookup
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
        if z_raw != z_raw:
            z_raw = 0.0
        terrain_z = z_raw * ve
        rain_z = terrain_z + z_frac[idx] * (cloud_z - terrain_z)

        wx = col * psx
        wy = row * psy

        dx = wx - cam_x
        dy = wy - cam_y
        dz = rain_z - cam_z

        depth = dx * fwd_x + dy * fwd_y + dz * fwd_z
        if depth <= min_depth:
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

        # Vertical streak
        sl = streak_lens[idx]
        for dy_off in range(sl):
            py = sy + dy_off
            if py < 0 or py >= sh:
                continue
            t = float(dy_off) / float(sl) if sl > 0 else 0.0
            streak_a = a * (1.0 - t * 0.6)
            cuda.atomic.add(output, (py, sx, 0), streak_a * color_r)
            cuda.atomic.add(output, (py, sx, 1), streak_a * color_g)
            cuda.atomic.add(output, (py, sx, 2), streak_a * color_b)
