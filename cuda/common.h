//
// Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//  * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//  * Neither the name of NVIDIA CORPORATION nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
// OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//

#pragma once

enum RayType
{
    RAY_TYPE_RADIANCE = 0,
    RAY_TYPE_COUNT
};

struct Ray
{
    float3 origin;
    float  tmin;
    float3 dir;
    float  tmax;
};

struct Hit
{
    float  t;
    float3 geom_normal;
};

struct Params
{
    OptixTraversableHandle handle;
    Ray*                   rays;
    Hit*                   hits;
    int*                   primitive_ids;  // Optional: triangle index per ray (-1 for miss)
    int*                   instance_ids;   // Optional: geometry/instance index per ray (-1 for miss)
    unsigned int           ray_flags;      // OptixRayFlags (e.g. CULL_BACK_FACING, TERMINATE_ON_FIRST_HIT)
    // --- heightfield fields (offset 48) ---
    float*                 heightfield_data;  // device pointer to H×W float32 elevation array
    int                    hf_width;          // W (columns)
    int                    hf_height;         // H (rows)
    float                  hf_spacing_x;      // world-space pixel spacing X
    float                  hf_spacing_y;      // world-space pixel spacing Y
    float                  hf_ve;             // vertical exaggeration
    int                    hf_tile_size;       // tile dimension (e.g. 32)
    int                    hf_num_tiles_x;     // number of tiles in X direction
    int                    _pad0;              // padding for pointer alignment
    // --- point cloud fields (offset 88) ---
    float*                 point_colors;       // per-point RGBA (4 floats per point, indexed by primitive_id)
};
