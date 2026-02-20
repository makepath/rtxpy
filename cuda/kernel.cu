#include <optix.h>
#include "common.h"

// ---- Compatibility helpers (older sample code used these names) ----
#ifndef int_as_float
#define int_as_float __int_as_float
#endif

#ifndef float_as_int
#define float_as_int __float_as_int
#endif
// -------------------------------------------------------------------

typedef unsigned long long uint64_t;

extern "C" {
__constant__ Params params;
}

extern "C" __global__ void __raygen__main()
{
    const uint3        idx        = optixGetLaunchIndex();
    const uint3        dim        = optixGetLaunchDimensions();
    const uint64_t linear_idx = idx.z * dim.y * dim.x + idx.y * dim.x + idx.x;

    unsigned int t, nx, ny, nz, prim_id, inst_id;
    Ray ray = params.rays[linear_idx];
    optixTrace(
        OPTIX_PAYLOAD_TYPE_ID_0,
        params.handle,
        ray.origin,
        ray.dir,
        ray.tmin,
        ray.tmax,
        0.0f,
        OptixVisibilityMask( 1 ),
        params.ray_flags,
        RAY_TYPE_RADIANCE,
        RAY_TYPE_COUNT,
        RAY_TYPE_RADIANCE,
        t,
        nx,
        ny,
        nz,
        prim_id,
        inst_id
    );

    Hit hit;
    hit.t                   = int_as_float( t );
    hit.geom_normal.x       = int_as_float( nx );
    hit.geom_normal.y       = int_as_float( ny );
    hit.geom_normal.z       = int_as_float( nz );
    params.hits[linear_idx] = hit;

    // Write optional primitive and instance IDs
    if (params.primitive_ids != nullptr) {
        params.primitive_ids[linear_idx] = static_cast<int>(prim_id);
    }
    if (params.instance_ids != nullptr) {
        params.instance_ids[linear_idx] = static_cast<int>(inst_id);
    }
}


extern "C" __global__ void __miss__miss()
{
    optixSetPayload_0( float_as_int( -1.0f ) );
    optixSetPayload_1( float_as_int( 1.0f ) );
    optixSetPayload_2( float_as_int( 0.0f ) );
    optixSetPayload_3( float_as_int( 0.0f ) );
    optixSetPayload_4( static_cast<unsigned int>(-1) );  // primitive_id = -1 for miss
    optixSetPayload_5( static_cast<unsigned int>(-1) );  // instance_id = -1 for miss
}


__device__ float3 operator-(const float3& a, const float3& b) {
    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}
__device__ float dot(const float3& a, const float3& b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}
__device__ float3 normalize(const float3& a) {
    const float lenSqr = dot(a, a);
    const float factor = 1.0f / sqrtf(lenSqr);
    return make_float3(a.x * factor, a.y * factor, a.z * factor);
}
__device__ float3 cross(const float3& a, const float3& b) {
    const float x = a.y * b.z - a.z * b.y;
    const float y = a.x * b.z - a.z * b.x;
    const float z = a.x * b.y - a.y * b.x;
    return make_float3(x, -y, z);
}

extern "C" __global__ void __closesthit__chit()
{
    const unsigned int t = optixGetRayTmax();
    unsigned int primIdx = optixGetPrimitiveIndex();

    float3 n;
    if (optixIsTriangleHit()) {
        float3 data[3];
        // Always use the 4-parameter overload for backward compatibility.
        // The parameterless overload (OptiX 9.1+) requires ABI version 99,
        // which needs driver 570+.  The 4-param form works on all versions.
        OptixTraversableHandle gas = optixGetGASTraversableHandle();
        unsigned int sbtIdx = optixGetSbtGASIndex();
        float time = optixGetRayTime();
        optixGetTriangleVertexData(gas, primIdx, sbtIdx, time, data);
        float3 AB = data[1] - data[0];
        float3 AC = data[2] - data[0];
        n = normalize(cross(AB, AC));
    } else {
        // Round curve tube: use face-up normal for terrain roads/rivers
        n = make_float3(0.0f, 0.0f, 1.0f);
    }

    // Set the hit data
    optixSetPayload_0(float_as_int(t));
    optixSetPayload_1(float_as_int(n.x));
    optixSetPayload_2(float_as_int(n.y));
    optixSetPayload_3(float_as_int(n.z));
    optixSetPayload_4(primIdx);                     // primitive/triangle index
    optixSetPayload_5(optixGetInstanceId());        // instance/geometry index
}


// ---------------------------------------------------------------------------
// Heightfield custom intersection program
// ---------------------------------------------------------------------------

// Helper: Moller-Trumbore ray-triangle intersection (single-sided)
__device__ bool ray_triangle(
    const float3& orig, const float3& dir,
    const float3& v0, const float3& v1, const float3& v2,
    float& t_out)
{
    const float3 e1 = v1 - v0;
    const float3 e2 = v2 - v0;
    const float3 pvec = cross(dir, e2);
    const float det = dot(e1, pvec);
    if (det < 1e-8f) return false;  // back-face or parallel
    const float inv_det = 1.0f / det;
    const float3 tvec = orig - v0;
    const float u = dot(tvec, pvec) * inv_det;
    if (u < 0.0f || u > 1.0f) return false;
    const float3 qvec = cross(tvec, e1);
    const float v = dot(dir, qvec) * inv_det;
    if (v < 0.0f || u + v > 1.0f) return false;
    t_out = dot(e2, qvec) * inv_det;
    return t_out > 0.0f;
}

// Bit-level NaN check — survives --use_fast_math (isnan gets optimized away)
__device__ __forceinline__ bool is_nan_safe(float f)
{
    unsigned int bits = __float_as_uint(f);
    return ((bits & 0x7F800000u) == 0x7F800000u) && ((bits & 0x007FFFFFu) != 0u);
}

// Fetch heightfield elevation with VE applied, NaN → 0.
// NaN cells become flat z=0 terrain; the render kernel detects them by
// checking the original elevation array and applies an ocean water shader.
__device__ __forceinline__ float hf_z(int r, int c)
{
    float z = params.heightfield_data[r * params.hf_width + c];
    if (is_nan_safe(z)) z = 0.0f;
    return z * params.hf_ve;
}

extern "C" __global__ void __intersection__heightfield()
{
    const int prim_idx = optixGetPrimitiveIndex();
    const float3 ray_o = optixGetObjectRayOrigin();
    const float3 ray_d = optixGetObjectRayDirection();
    const float  ray_tmin = optixGetRayTmin();
    const float  ray_tmax = optixGetRayTmax();

    const float sx = params.hf_spacing_x;
    const float sy = params.hf_spacing_y;
    const int tile_size = params.hf_tile_size;
    const int W = params.hf_width;
    const int H = params.hf_height;

    // Tile grid coordinates from primitive index
    const int tile_col = prim_idx % params.hf_num_tiles_x;
    const int tile_row = prim_idx / params.hf_num_tiles_x;

    // Cell range for this tile (in grid cells, clamped to DEM extent)
    const int cell_col0 = tile_col * tile_size;
    const int cell_row0 = tile_row * tile_size;
    const int cell_col1 = min(cell_col0 + tile_size, W - 1);
    const int cell_row1 = min(cell_row0 + tile_size, H - 1);

    if (cell_col0 >= cell_col1 || cell_row0 >= cell_row1)
        return;

    // Tile AABB in world space
    const float tile_x0 = cell_col0 * sx;
    const float tile_y0 = cell_row0 * sy;
    const float tile_x1 = cell_col1 * sx;
    const float tile_y1 = cell_row1 * sy;

    // Compute ray entry into tile AABB (XY only, Z handled per cell)
    float t_enter = ray_tmin;
    float t_exit = ray_tmax;

    // Clamp ray to tile XY bounds
    if (fabsf(ray_d.x) > 1e-8f) {
        float t0 = (tile_x0 - ray_o.x) / ray_d.x;
        float t1 = (tile_x1 - ray_o.x) / ray_d.x;
        if (t0 > t1) { float tmp = t0; t0 = t1; t1 = tmp; }
        t_enter = fmaxf(t_enter, t0);
        t_exit  = fminf(t_exit, t1);
    } else {
        if (ray_o.x < tile_x0 || ray_o.x > tile_x1) return;
    }
    if (fabsf(ray_d.y) > 1e-8f) {
        float t0 = (tile_y0 - ray_o.y) / ray_d.y;
        float t1 = (tile_y1 - ray_o.y) / ray_d.y;
        if (t0 > t1) { float tmp = t0; t0 = t1; t1 = tmp; }
        t_enter = fmaxf(t_enter, t0);
        t_exit  = fminf(t_exit, t1);
    } else {
        if (ray_o.y < tile_y0 || ray_o.y > tile_y1) return;
    }

    if (t_enter > t_exit) return;

    // Entry point in world space
    float3 p = make_float3(
        ray_o.x + ray_d.x * t_enter,
        ray_o.y + ray_d.y * t_enter,
        ray_o.z + ray_d.z * t_enter);

    // Convert to grid coordinates (fractional)
    float gx = p.x / sx;
    float gy = p.y / sy;

    // Current cell
    int cx = (int)floorf(gx);
    int cy = (int)floorf(gy);
    cx = max(cx, cell_col0);
    cx = min(cx, cell_col1 - 1);
    cy = max(cy, cell_row0);
    cy = min(cy, cell_row1 - 1);

    // DDA step direction
    int step_x = (ray_d.x >= 0.0f) ? 1 : -1;
    int step_y = (ray_d.y >= 0.0f) ? 1 : -1;

    // DDA t-deltas (world-space t per grid cell)
    float dt_x = (fabsf(ray_d.x) > 1e-8f) ? fabsf(sx / ray_d.x) : 1e30f;
    float dt_y = (fabsf(ray_d.y) > 1e-8f) ? fabsf(sy / ray_d.y) : 1e30f;

    // Next cell boundary t values
    float next_t_x, next_t_y;
    if (fabsf(ray_d.x) > 1e-8f) {
        float boundary_x = (ray_d.x >= 0.0f) ? (cx + 1) * sx : cx * sx;
        next_t_x = (boundary_x - ray_o.x) / ray_d.x;
    } else {
        next_t_x = 1e30f;
    }
    if (fabsf(ray_d.y) > 1e-8f) {
        float boundary_y = (ray_d.y >= 0.0f) ? (cy + 1) * sy : cy * sy;
        next_t_y = (boundary_y - ray_o.y) / ray_d.y;
    } else {
        next_t_y = 1e30f;
    }

    // DDA loop through cells within this tile
    float best_t = ray_tmax;
    float best_nx = 0.0f, best_ny = 0.0f;
    bool found = false;

    for (int iter = 0; iter < tile_size * tile_size * 2 + 4; iter++) {
        if (cx < cell_col0 || cx >= cell_col1 ||
            cy < cell_row0 || cy >= cell_row1)
            break;

        // Grid cell corners (row=cy, col=cx)
        // v00 = (cx, cy), v10 = (cx+1, cy), v01 = (cx, cy+1), v11 = (cx+1, cy+1)
        const float z00 = hf_z(cy,     cx    );
        const float z10 = hf_z(cy,     cx + 1);
        const float z01 = hf_z(cy + 1, cx    );
        const float z11 = hf_z(cy + 1, cx + 1);

        const float3 v00 = make_float3(cx       * sx, cy       * sy, z00);
        const float3 v10 = make_float3((cx + 1) * sx, cy       * sy, z10);
        const float3 v01 = make_float3(cx       * sx, (cy + 1) * sy, z01);
        const float3 v11 = make_float3((cx + 1) * sx, (cy + 1) * sy, z11);

        // Test two triangles per cell (same winding as triangulate_terrain):
        // Triangle 0: v00, v10, v01  (lower-left)
        // Triangle 1: v10, v11, v01  (upper-right)
        float t_hit;
        if (ray_triangle(ray_o, ray_d, v00, v10, v01, t_hit)) {
            if (t_hit >= ray_tmin && t_hit < best_t) {
                best_t = t_hit;
                found = true;
                // Compute bilinear normal at hit point
                float3 hp = make_float3(
                    ray_o.x + ray_d.x * t_hit,
                    ray_o.y + ray_d.y * t_hit,
                    ray_o.z + ray_d.z * t_hit);
                float u = (hp.x / sx) - cx;
                float v = (hp.y / sy) - cy;
                u = fmaxf(0.0f, fminf(1.0f, u));
                v = fmaxf(0.0f, fminf(1.0f, v));
                float dz_dx = ((1.0f - v) * (z10 - z00) + v * (z11 - z01)) / sx;
                float dz_dy = ((1.0f - u) * (z01 - z00) + u * (z11 - z10)) / sy;
                float3 n = normalize(make_float3(-dz_dx, -dz_dy, 1.0f));
                best_nx = n.x;
                best_ny = n.y;
            }
        }
        if (ray_triangle(ray_o, ray_d, v10, v11, v01, t_hit)) {
            if (t_hit >= ray_tmin && t_hit < best_t) {
                best_t = t_hit;
                found = true;
                float3 hp = make_float3(
                    ray_o.x + ray_d.x * t_hit,
                    ray_o.y + ray_d.y * t_hit,
                    ray_o.z + ray_d.z * t_hit);
                float u = (hp.x / sx) - cx;
                float v = (hp.y / sy) - cy;
                u = fmaxf(0.0f, fminf(1.0f, u));
                v = fmaxf(0.0f, fminf(1.0f, v));
                float dz_dx = ((1.0f - v) * (z10 - z00) + v * (z11 - z01)) / sx;
                float dz_dy = ((1.0f - u) * (z01 - z00) + u * (z11 - z10)) / sy;
                float3 n = normalize(make_float3(-dz_dx, -dz_dy, 1.0f));
                best_nx = n.x;
                best_ny = n.y;
            }
        }

        // Early exit: if we found a hit in this cell, the DDA guarantees
        // we won't find a closer one in later cells (front-to-back order)
        if (found) break;

        // Step to next cell
        if (next_t_x < next_t_y) {
            cx += step_x;
            next_t_x += dt_x;
        } else {
            cy += step_y;
            next_t_y += dt_y;
        }
    }

    if (found) {
        // Pack normal components as attributes
        unsigned int a0 = float_as_int(best_nx);
        unsigned int a1 = float_as_int(best_ny);
        optixReportIntersection(best_t, 0, a0, a1);
    }
}


extern "C" __global__ void __closesthit__heightfield()
{
    const float t = optixGetRayTmax();

    // Reconstruct normal from attributes packed by IS program
    const float nx = int_as_float(optixGetAttribute_0());
    const float ny = int_as_float(optixGetAttribute_1());
    float nz_sq = 1.0f - nx * nx - ny * ny;
    if (nz_sq < 0.0f) nz_sq = 0.0f;
    const float nz = sqrtf(nz_sq);

    optixSetPayload_0(float_as_int(t));
    optixSetPayload_1(float_as_int(nx));
    optixSetPayload_2(float_as_int(ny));
    optixSetPayload_3(float_as_int(nz));
    optixSetPayload_4(optixGetPrimitiveIndex());
    optixSetPayload_5(optixGetInstanceId());
}
