#include <optix.h>
#include "common.h"

typedef unsigned long long uint64_t;

extern "C" {
__constant__ Params params;
}

extern "C" __global__ void __raygen__main()
{
    const uint3        idx        = optixGetLaunchIndex();
    const uint3        dim        = optixGetLaunchDimensions();
    const uint64_t linear_idx = idx.z * dim.y * dim.x + idx.y * dim.x + idx.x;

    unsigned int t, nx, ny, nz;
    Ray ray = params.rays[linear_idx];
    optixTrace(
        params.handle, 
        ray.origin, 
        ray.dir, 
        ray.tmin, 
        ray.tmax, 
        0.0f, 
        OptixVisibilityMask( 1 ),
        OPTIX_RAY_FLAG_NONE, 
        RAY_TYPE_RADIANCE, 
        RAY_TYPE_COUNT, 
        RAY_TYPE_RADIANCE, 
        t, 
        nx, 
        ny, 
        nz
    );

    Hit hit;
    hit.t                   = int_as_float( t );
    hit.geom_normal.x       = int_as_float( nx );
    hit.geom_normal.y       = int_as_float( ny );
    hit.geom_normal.z       = int_as_float( nz );
    params.hits[linear_idx] = hit;
}


extern "C" __global__ void __miss__miss()
{
    optixSetPayload_0( float_as_int( -1.0f ) );
    optixSetPayload_1( float_as_int( 1.0f ) );
    optixSetPayload_2( float_as_int( 0.0f ) );
    optixSetPayload_3( float_as_int( 0.0f ) );
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

    OptixTraversableHandle gas = optixGetGASTraversableHandle();
    unsigned int primIdx = optixGetPrimitiveIndex();
    unsigned int sbtIdx = optixGetSbtGASIndex();
    float time = optixGetRayTime();

    float3 data[3];
    optixGetTriangleVertexData(gas, primIdx, sbtIdx, time, data);
    float3 AB = data[1] - data[0];
    float3 AC = data[2] - data[0];
    float3 n = normalize(cross(AB, AC));

    // Set the hit data
    optixSetPayload_0(float_as_int(t));
    optixSetPayload_1(float_as_int(n.x));
    optixSetPayload_2(float_as_int(n.y));
    optixSetPayload_3(float_as_int(n.z));
}
