#include "cuew/cuew.h"          // this pulls in cuda.h safely (no prototypes)

#include <optix.h>
#include <optix_function_table_definition.h>
#include <optix_stack_size.h>
#include <optix_stubs.h>

// now your other includes:
#include <assert.h>
#include <climits>
#include <sstream>
#include <vector>
#include <fstream>
#include <string>

struct float3 { float x,y,z; };
struct int3   { int i[3]; };

#include "common.h"
#include "rtx.h"
#include "internal.h"


// Optional: returns CUDA driver version as int like 12040 == 12.4
static int getCudaDriverVersion()
{
    int v = 0;
    cuDriverGetVersion(&v);
    return v;
}

#define OPTIX_CHECK_RTX(call)                                                     \
    do {                                                                          \
        OptixResult _res = (call);                                                \
        if (_res != OPTIX_SUCCESS) {                                              \
            std::ostringstream _oss;                                              \
            _oss << "OptiX call failed: " << #call                                \
                 << " at " << __FILE__ << ":" << __LINE__                         \
                 << " (" << __FUNCTION__ << ")\n"                                 \
                 << "  OptixResult: " << optixResultToString(_res)                \
                 << " (" << static_cast<int>(_res) << ")\n"                       \
                 << "  CUDA driver version: " << getCudaDriverVersion() << "\n"  \
                 << "  OPTIX_ABI_VERSION (compiled): " << OPTIX_ABI_VERSION       \
                 << "\n";                                                         \
            if (_res == OPTIX_ERROR_UNSUPPORTED_ABI_VERSION) {                    \
                _oss << "  Hint: driver/runtime ABI mismatch. Your NVIDIA "       \
                        "driver is likely too old for this OptiX SDK.\n";         \
            }                                                                     \
            setLastErrorRTX(_oss.str().c_str());                                  \
            return static_cast<int>(_res);                                        \
        }                                                                         \
    } while (0)

State global_state;

/// Read the contents of a text file @fileName and return them in a string
std::string getTextFileContents(const char* fileName) {
    FILE* f = fopen(fileName, "rt");
    std::string res;

    if (f) {
        fseek(f, 0, SEEK_END);
        size_t size = ftell(f);
        fseek(f, 0, SEEK_SET);

        if (size > 0) {
            res.resize(size + 1);
            size_t bytes_read = fread(&res[0], 1, size, f);
            (void)bytes_read;
	    res[size] = 0;
        }
        fclose(f);

    }
    return res;
}


static thread_local std::string g_last_error;

extern "C" void setLastErrorRTX(const char* msg)
{
    g_last_error = (msg ? msg : "");
}

extern "C" const char* getLastErrorRTX()
{
    return g_last_error.empty() ? nullptr : g_last_error.c_str();
}


static const char* optixResultToString(OptixResult r)
{
    switch (r)
    {
        case OPTIX_SUCCESS: return "OPTIX_SUCCESS";
        case OPTIX_ERROR_INVALID_VALUE: return "OPTIX_ERROR_INVALID_VALUE";
        case OPTIX_ERROR_HOST_OUT_OF_MEMORY: return "OPTIX_ERROR_HOST_OUT_OF_MEMORY";
        case OPTIX_ERROR_INVALID_OPERATION: return "OPTIX_ERROR_INVALID_OPERATION";
        case OPTIX_ERROR_FILE_IO_ERROR: return "OPTIX_ERROR_FILE_IO_ERROR";
        case OPTIX_ERROR_INVALID_FILE_FORMAT: return "OPTIX_ERROR_INVALID_FILE_FORMAT";
        case OPTIX_ERROR_DISK_CACHE_INVALID_PATH: return "OPTIX_ERROR_DISK_CACHE_INVALID_PATH";
        case OPTIX_ERROR_UNSUPPORTED_ABI_VERSION: return "OPTIX_ERROR_UNSUPPORTED_ABI_VERSION";
        case OPTIX_ERROR_FUNCTION_TABLE_SIZE_MISMATCH: return "OPTIX_ERROR_FUNCTION_TABLE_SIZE_MISMATCH";
        case OPTIX_ERROR_INVALID_DEVICE_CONTEXT: return "OPTIX_ERROR_INVALID_DEVICE_CONTEXT";
        default: return "OPTIX_ERROR_UNKNOWN";
    }
}

int createModule(State& state)
{
    char   log[16384];
    size_t logSize = sizeof(log);

    OptixModuleCompileOptions module_compile_options = {};
    module_compile_options.maxRegisterCount = OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT;
    module_compile_options.optLevel         = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
    module_compile_options.debugLevel       = OPTIX_COMPILE_DEBUG_LEVEL_MINIMAL;

    state.pipeline_compile_options.usesMotionBlur        = false;
    state.pipeline_compile_options.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
    state.pipeline_compile_options.numPayloadValues      = 4;
    state.pipeline_compile_options.numAttributeValues    = 2;
    state.pipeline_compile_options.exceptionFlags        = OPTIX_EXCEPTION_FLAG_NONE;
    state.pipeline_compile_options.pipelineLaunchParamsVariableName = "params";

    std::string ptx = load_ptx_file("kernel.ptx");
    if (ptx.empty()) {
        fprintf(stderr, "Failed to load kernel.ptx\n");
        return -1;
    }

    const char*  input     = ptx.data();
    const size_t inputSize = ptx.size();

    OPTIX_CHECK_LOG(
        optixModuleCreate(
            state.context,
            &module_compile_options,
            &state.pipeline_compile_options,
            input,
            inputSize,
            log,
            &logSize,
            &state.ptx_module
        )
    );

    return 0;
}



int createProgramGroups(State& state)
{
    char   log[2048];
    size_t logSize = sizeof( log );

    OptixProgramGroupOptions program_group_options = {};

    OptixProgramGroupDesc raygen_prog_group_desc    = {};
    raygen_prog_group_desc.kind                     = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
    raygen_prog_group_desc.raygen.module            = state.ptx_module;
    raygen_prog_group_desc.raygen.entryFunctionName = "__raygen__main";

    OPTIX_CHECK_LOG(
        optixProgramGroupCreate(
            state.context,
            &raygen_prog_group_desc,
            1,  // num program groups
            &program_group_options,
            log, 
            &logSize,
            &state.raygen
        )
    );

    OptixProgramGroupDesc miss_prog_group_desc  = {};
    miss_prog_group_desc.kind                   = OPTIX_PROGRAM_GROUP_KIND_MISS;
    miss_prog_group_desc.miss.module            = state.ptx_module;
    miss_prog_group_desc.miss.entryFunctionName = "__miss__miss";
    OPTIX_CHECK_LOG(
        optixProgramGroupCreate(
            state.context, 
            &miss_prog_group_desc,
            1,  // num program groups
            &program_group_options, 
            log, 
            &logSize,
            &state.miss
        )
    );

    OptixProgramGroupDesc hit_prog_group_desc = {};
    hit_prog_group_desc.kind                  = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
    hit_prog_group_desc.hitgroup.moduleCH     = state.ptx_module;
    hit_prog_group_desc.hitgroup.entryFunctionNameCH = "__closesthit__chit";
    OPTIX_CHECK_LOG(
        optixProgramGroupCreate(
            state.context, 
            &hit_prog_group_desc,
            1,  // num program groups
            &program_group_options, 
            log, 
            &logSize, 
            &state.hit
        ) 
    );
    return 0;
}

int createPipelines(State& state)
{
    OptixResult res = OPTIX_SUCCESS;
    char   log[2048];
    size_t sizeof_log = sizeof(log);

    const uint32_t    max_trace_depth   = 1;
    OptixProgramGroup program_groups[3] = { state.raygen, state.miss, state.hit };

    OptixPipelineLinkOptions pipeline_link_options = {};
    pipeline_link_options.maxTraceDepth = max_trace_depth;

    res = optixPipelineCreate(
        state.context,
        &state.pipeline_compile_options,
        &pipeline_link_options,
        program_groups,
        sizeof(program_groups) / sizeof(program_groups[0]),
        log,
        &sizeof_log,
        &state.pipeline
    );
    if (res != OPTIX_SUCCESS) {
        fprintf(stderr, "Failed to create OptiX Pipeline.\nLog:\n%s\n", log);
        return -1;
    }

    OptixStackSizes stack_sizes = {};
    for (auto& prog_group : program_groups) {
        OPTIX_CHECK(optixUtilAccumulateStackSizes(prog_group, &stack_sizes, state.pipeline));
    }

    uint32_t direct_callable_stack_size_from_traversal = 0;
    uint32_t direct_callable_stack_size_from_state     = 0;
    uint32_t continuation_stack_size                   = 0;

    OPTIX_CHECK(
        optixUtilComputeStackSizes(
            &stack_sizes,
            max_trace_depth,
            0, // maxCCDepth
            0, // maxDCDepth
            &direct_callable_stack_size_from_traversal,
            &direct_callable_stack_size_from_state,
            &continuation_stack_size
        )
    );

    OPTIX_CHECK(
        optixPipelineSetStackSize(
            state.pipeline,
            direct_callable_stack_size_from_traversal,
            direct_callable_stack_size_from_state,
            continuation_stack_size,
            1 // maxTraversableDepth
        )
    );

    return 0;
}


int createSBT(State& state)
{
    CUresult err = CUDA_SUCCESS;
    // raygen
    RayGenSbtRecord rgSBT;
    OPTIX_CHECK(optixSbtRecordPackHeader(state.raygen, &rgSBT));
    const int raygenRecordSize = sizeof(RayGenSbtRecord);

    CUdeviceptr  d_raygen_record = 0;
    err = cuMemAlloc(&d_raygen_record, raygenRecordSize);
    assert(err == CUDA_SUCCESS);
    err = cuMemcpyHtoD(d_raygen_record, &rgSBT, raygenRecordSize);
    assert(err == CUDA_SUCCESS);

    // miss
    MissSbtRecord msSBT;
    OPTIX_CHECK(optixSbtRecordPackHeader(state.miss, &msSBT));
    const int missSbtRecordSize = sizeof(MissSbtRecord);

    CUdeviceptr  d_miss_record    = 0;
    err = cuMemAlloc(&d_miss_record, missSbtRecordSize);
    assert(err == CUDA_SUCCESS);
    err = cuMemcpyHtoD(d_miss_record, &msSBT, missSbtRecordSize);
    assert(err == CUDA_SUCCESS);

    // hit group
    HitGroupSbtRecord hgSBT;
    OPTIX_CHECK(optixSbtRecordPackHeader(state.hit, &hgSBT));
    const int  hitgroupSbtRecordSize = sizeof(MissSbtRecord);

    CUdeviceptr  d_chit_record = 0;
    err = cuMemAlloc(&d_chit_record, hitgroupSbtRecordSize);
    assert(err == CUDA_SUCCESS);
    err = cuMemcpyHtoD(d_chit_record, &hgSBT, hitgroupSbtRecordSize);
    assert(err == CUDA_SUCCESS);

    OptixShaderBindingTable &shaderbt = state.sbt;
    memset(&shaderbt, 0, sizeof(OptixShaderBindingTable));
    shaderbt.raygenRecord = d_raygen_record;
    shaderbt.missRecordBase = d_miss_record;
    shaderbt.missRecordStrideInBytes = sizeof(MissSbtRecord);
    shaderbt.missRecordCount = 1;
    shaderbt.hitgroupRecordBase = d_chit_record;
    shaderbt.hitgroupRecordStrideInBytes = sizeof(HitGroupSbtRecord);
    shaderbt.hitgroupRecordCount = 1;
    return err;
}

int cleanup_internal(State& state);
int buildRTX_internal(State& state, uint64_t hash, float3* verts, int64_t numVerts, int3* triangles, int numTriangles);
int initRTX_internal(State& state);
int traceRTX_internal(State& state, Ray* rays, Hit* hits, int size);
int initBuffers_internal(State& state, int numRays);

int buildRTX_internal(State& state, uint64_t hash, float3* verts, int64_t numVerts, int3* triangles, int numTriangles) {
	assert(numVerts<INT_MAX);
    if (!state.valid) {
        fprintf(stderr, "State is invalid!");
        return -2;
    }

    if (state.scene.hash == hash) {
#if DEBUG_PRINTS
        fprintf(stderr, "Hash matches currently stored acceleration stucture. Reusing...");
#endif        
        return 0;
    } else {
        // Reset the hash until we successfully build a GAS.
        state.scene.hash = uint64_t(-1);
    }

    CUresult err = CUDA_SUCCESS;

    size_t d_verts_size = sizeof(float3) * numVerts;
    size_t d_tris_size = sizeof(int3) * numTriangles;

    CUpointer_attribute attr = CU_POINTER_ATTRIBUTE_MEMORY_TYPE;
    unsigned int data = 0;
    CUdeviceptr d_verts = (CUdeviceptr)verts;
    err = cuPointerGetAttribute(&data, attr, d_verts);
    if (err == CUDA_ERROR_INVALID_VALUE || data == CU_MEMORYTYPE_HOST) {
        err = cuMemAlloc(&d_verts, d_verts_size);
        CHECK_CUDA_LOG(err, "Failed to allocate vertex buffer");
        err = cuMemcpyHtoD(d_verts, verts, d_verts_size);
        CHECK_CUDA_LOG(err, "Failed to transfer vertex buffer to device");
    }

    CUdeviceptr  d_tris = (CUdeviceptr)triangles;
    err = cuPointerGetAttribute(&data, attr, d_tris);
    if (err == CUDA_ERROR_INVALID_VALUE || data == CU_MEMORYTYPE_HOST) {
        err = cuMemAlloc(&d_tris, d_tris_size);
        CHECK_CUDA_LOG(err, "Failed to allocate index buffer");
        err = cuMemcpyHtoD(d_tris, triangles, d_tris_size);
        CHECK_CUDA_LOG(err, "Failed to transfer index buffer");
    }

    CUdeviceptr vbuff = d_verts;

    OptixBuildInput geometry;
    memset(&geometry, 0, sizeof(OptixBuildInput));
    geometry.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
    geometry.triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
    geometry.triangleArray.vertexStrideInBytes = sizeof(float3);
    geometry.triangleArray.numVertices = int(numVerts);
    geometry.triangleArray.vertexBuffers = &vbuff;

    geometry.triangleArray.indexBuffer = d_tris;
    geometry.triangleArray.indexStrideInBytes = sizeof(int3);
    geometry.triangleArray.numIndexTriplets = numTriangles;
    geometry.triangleArray.indexFormat = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;

    uint32_t flags[] = { OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT };
    geometry.triangleArray.flags = flags;
    geometry.triangleArray.numSbtRecords = 1;

    OptixAccelBuildOptions accelOptions = {};
    accelOptions.buildFlags = OPTIX_BUILD_FLAG_ALLOW_RANDOM_VERTEX_ACCESS;
    accelOptions.motionOptions.numKeys = 1;
    accelOptions.operation = OPTIX_BUILD_OPERATION_BUILD;

    OptixAccelBufferSizes blasBuffSizes;
    OPTIX_CHECK(optixAccelComputeMemoryUsage(state.context, &accelOptions, &geometry, 1, &blasBuffSizes));

    CUdeviceptr  d_tmp = 0;
    err = cuMemAlloc(&d_tmp, blasBuffSizes.tempSizeInBytes);
    CHECK_CUDA_LOG(err, "Failed to allocate temp buffer for acceleration structure");

    err = cuMemFree(state.scene.memory);
    CHECK_CUDA_LOG(err, "Failed to delete old acceleration stucture");

    err = cuMemAlloc(&state.scene.memory, blasBuffSizes.outputSizeInBytes);
    CHECK_CUDA_LOG(err, "Failed to allocate main buffer for acceleration structure");

    OPTIX_CHECK(
        optixAccelBuild(
            state.context,
            0,
            &accelOptions,
            &geometry,
            1,
            d_tmp,
            blasBuffSizes.tempSizeInBytes,
            state.scene.memory,
            blasBuffSizes.outputSizeInBytes,
            &state.scene.accelerationStructureHandle,
            nullptr,
            0
        )
    );

    err = cuMemFree(d_tmp);
    CHECK_CUDA_LOG(err, "Failed to delete temp buffer for acceleration stucture");
    
    if (d_verts != (CUdeviceptr)verts) {
        err = cuMemFree(d_verts);
        CHECK_CUDA_LOG(err, "Failed to delete vertex buffer for acceleration stucture");
    }
    if (d_tris != (CUdeviceptr)triangles) {
        err = cuMemFree(d_tris);
        CHECK_CUDA_LOG(err, "Failed to delete index buffer for acceleration stucture");
    }
    state.scene.hash = hash;
    return err;
}

int cleanup_internal(State& state)
{
    // If CUDA context was never created, nothing to do.
    if (state.cuda.context == 0) {
        state.valid = false;
        return 0;
    }

    state.valid = false;

    // Free scene memory safely
    state.scene.freeMem();
    state.scene.hash = uint64_t(-1);

    // OptiX objects (ONLY destroy if non-zero)
    if (state.pipeline) {
        optixPipelineDestroy(state.pipeline);
        state.pipeline = 0;
    }
    if (state.raygen) {
        optixProgramGroupDestroy(state.raygen);
        state.raygen = 0;
    }
    if (state.miss) {
        optixProgramGroupDestroy(state.miss);
        state.miss = 0;
    }
    if (state.hit) {
        optixProgramGroupDestroy(state.hit);
        state.hit = 0;
    }
    if (state.ptx_module) {
        optixModuleDestroy(state.ptx_module);
        state.ptx_module = 0;
    }
    if (state.context) {
        optixDeviceContextDestroy(state.context);
        state.context = 0;
    }

    // Free SBT buffers (ONLY if non-zero)
    if (state.sbt.raygenRecord) {
        cuMemFree(state.sbt.raygenRecord);
    }
    if (state.sbt.missRecordBase) {
        cuMemFree(state.sbt.missRecordBase);
    }
    if (state.sbt.hitgroupRecordBase) {
        cuMemFree(state.sbt.hitgroupRecordBase);
    }
    memset(&state.sbt, 0, sizeof(state.sbt));

    // Free params/rays/hits (ONLY if non-zero)
    if (state.d_params) cuMemFree(state.d_params);
    if (state.d_rays)   cuMemFree(state.d_rays);
    if (state.d_hits)   cuMemFree(state.d_hits);

    state.d_params = 0;
    state.d_rays   = 0;
    state.d_hits   = 0;
    state.d_rays_size = 0;
    state.d_hits_size = 0;

    // Stream/context teardown
    if (state.cuda.stream) {
        cuStreamDestroy(state.cuda.stream);
        state.cuda.stream = 0;
    }

    if (state.cuda.device) {
        cuDevicePrimaryCtxRelease(state.cuda.device);
        state.cuda.device = 0;
    }

    state.cuda.context = 0;
    return 0;
}


int traceRTX_internal(State& state, Ray* rays, Hit* hits, int size) {
    if (!state.valid) {
        fprintf(stderr, "State is invalid!");
        return -2;
    }

    CUresult err = CUDA_SUCCESS;
    size_t d_rays_size = sizeof(Ray) * size;
    size_t d_hits_size = sizeof(Hit) * size;

    CUpointer_attribute attr = CU_POINTER_ATTRIBUTE_MEMORY_TYPE;
    unsigned int data=0;
    CUdeviceptr rays_ptr = (CUdeviceptr)rays;
    err = cuPointerGetAttribute(&data, attr, rays_ptr);
    if (err == CUDA_ERROR_INVALID_VALUE || data == CU_MEMORYTYPE_HOST) {
        initBuffers_internal(state, size);
        err = cuMemcpyHtoDAsync(state.d_rays, rays, d_rays_size, state.cuda.stream);
        CHECK_CUDA_LOG(err, "Failed to transfer rays buffer to device");
        rays_ptr = state.d_rays;
    }

    data = 0;
    CUdeviceptr hits_ptr = (CUdeviceptr)hits;
    err = cuPointerGetAttribute(&data, attr, hits_ptr);
    if (err == CUDA_ERROR_INVALID_VALUE || data == CU_MEMORYTYPE_HOST) {
        hits_ptr = state.d_hits;
    }

    Params params;
    params.handle = state.scene.accelerationStructureHandle;
    params.rays = (Ray*)rays_ptr;
    params.hits = (Hit*)hits_ptr;

    err = cuMemcpyHtoDAsync(state.d_params, &params, sizeof(Params), state.cuda.stream);
    CHECK_CUDA_LOG(err, "Failed to transfer params buffer to device");

    OPTIX_CHECK(optixLaunch(state.pipeline, state.cuda.stream, state.d_params, sizeof(Params), &state.sbt, size, 1, 1));

    if (hits != params.hits) {
        err = cuMemcpyDtoHAsync(hits, (CUdeviceptr)params.hits, d_hits_size, state.cuda.stream);
        CHECK_CUDA_LOG(err, "Failed to transfer hits buffer from device");
    }
    err = cuStreamSynchronize(state.cuda.stream);
    CHECK_CUDA_LOG(err, "Failed to synchronize device");
    return 0;
}

void contextLogCallback(unsigned int level, const char* tag, const char* message, void* cbdata) {
	fprintf(stderr, "[OPTIX CB][%d][%s] %s\n", level, tag, message);
}

int initRTX_internal(State& state) {
    if (state.valid) {
        //fprintf(stderr, "State already initialized\n");
        return 0;
    }
    int deviceIndex = 0;
    int err = cuewInit(CUEW_INIT_CUDA);
    if (err != CUEW_SUCCESS) {
        fprintf(stderr, "Error[%d] at %s[%d]: Failed to find CUDA\n", err, __FUNCTION__, __LINE__);
        return err;
    }
    err = cuInit(0);
    CHECK_CUDA_LOG(err, "Failed to initialize CUDA");

    CUdevice device = 0;
    err = cuDeviceGet(&device, deviceIndex);
    CHECK_CUDA_LOG(err, "Failed to obtain a handle to device 0");

    CUcontext cudaContext = 0;
    err = cuDevicePrimaryCtxRetain(&cudaContext, device);
    CHECK_CUDA_LOG(err, "Failed to create a CUDA context");

    err = cuCtxSetCurrent(cudaContext);
    CHECK_CUDA_LOG(err, "Failed to push CUDA context");

    CUstream stream = 0;
    err = cuStreamCreate(&stream, CU_STREAM_NON_BLOCKING);
    CHECK_CUDA_LOG(err, "Failed to create a CUDA stream");

    state.cuda.stream = stream;
    state.cuda.context = cudaContext;
    state.cuda.device = device;

    err = cuMemAlloc(&state.d_params, sizeof(Params));
    CHECK_CUDA_LOG(err, "Failed to allocate internal state buffer");

  // Make the optixInit failure readable in Python:
    OPTIX_CHECK_RTX(optixInit());

    // Always zero-init options:
    OptixDeviceContextOptions options = {};
    options.logCallbackFunction = contextLogCallback;
    options.logCallbackData     = nullptr;
    options.logCallbackLevel    = 4; // max verbosity

#if OPTIX_VERSION >= 70300
#if DEBUG_PRINTS
    options.validationMode = OPTIX_DEVICE_CONTEXT_VALIDATION_MODE_ALL;
#endif
#endif

    OPTIX_CHECK_RTX(optixDeviceContextCreate(state.cuda.context, &options, &state.context));

    if (!state.context) {
        setLastErrorRTX("optixDeviceContextCreate returned success but state.context is null");
        return -1;
    }

    // For your other helpers, also setLastErrorRTX before returning -1
    if (createModule(state))         { /* setLastErrorRTX inside createModule */ return -1; }
    if (createProgramGroups(state))  { /* ... */ return -1; }
    if (createPipelines(state))      { /* ... */ return -1; }
    if (createSBT(state))            { /* ... */ return -1; }

    state.valid = true;
    return 0;
}

int initBuffers_internal(State& state, int numRays) {
    if (!state.valid) {
        fprintf(stderr, "State is invalid!");
        return -2;
    }

    CUresult err = CUDA_SUCCESS;
    size_t newRaysSize = sizeof(Ray) * (size_t)numRays;
    size_t newHitsSize = sizeof(Hit) * (size_t)numRays;

    if (newRaysSize != state.d_rays_size || newHitsSize != state.d_hits_size) {

        // Only free if we actually have something allocated
        if (state.d_rays) {
            err = cuMemFree(state.d_rays);
            CHECK_CUDA_LOG(err, "Failed to deallocate old input data buffer");
            state.d_rays = 0;
        }
        if (state.d_hits) {
            err = cuMemFree(state.d_hits);
            CHECK_CUDA_LOG(err, "Failed to deallocate old output data buffer");
            state.d_hits = 0;
        }

        // Allocate new buffers
        state.d_rays_size = newRaysSize;
        err = cuMemAlloc(&state.d_rays, state.d_rays_size);
        CHECK_CUDA_LOG(err, "Failed to allocate input data buffer");

        state.d_hits_size = newHitsSize;
        err = cuMemAlloc(&state.d_hits, state.d_hits_size);
        CHECK_CUDA_LOG(err, "Failed to allocate output data buffer");
    }

    return err;
}

////////////////////////////////////////////////////////
// DLL exposed functionality methods start here
////////////////////////////////////////////////////////

DLL_API int buildRTX(uint64_t hash, void* pverts, int64_t vBytes, void* ptriangles, int tBytes) {

    const int numVerts = int(vBytes / (sizeof(float) * 3));
    const int numTriangles = int(tBytes / (sizeof(int) * 3));
    int3* triangles = reinterpret_cast<int3*>(ptriangles);
    float3* verts = reinterpret_cast<float3*>(pverts);

    if (!verts || !triangles || numVerts == 0 || numTriangles == 0) {
#if DEBUG_PRINTS
        fprintf(stderr, "[ERROR] Empty index/vertex buffer detected!\n");
#endif
        return -1;
    }
#if DEBUG_PRINTS
    fprintf(stderr, "%s[%d]: %s\n", __FILE__, __LINE__, __FUNCTION__);
    fprintf(stderr, "%s[%d]: verts = %p numVerts = %d | tris = %p, numtris = %d\n", __FILE__, __LINE__,
        verts, numVerts, triangles, numTriangles
    );
#endif

    int err = buildRTX_internal(global_state, hash, verts, numVerts, triangles, numTriangles);
    return err;
}

DLL_API int initRTX()
{
#if DEBUG_PRINTS
    fprintf(stderr, "%s[%d]: %s\n", __FILE__, __LINE__, __FUNCTION__);
#endif
    int err = initRTX_internal(global_state);
    return err;
}

DLL_API int traceRTX(void* rrays, void* hhits, int size) {
#if DEBUG_PRINTS
    fprintf(stderr, "%s[%d]: %s\n", __FILE__, __LINE__, __FUNCTION__);
#endif
    if (!rrays || !hhits || size == 0) {
        fprintf(stderr, "Invalid call to trace with empty buffers\n");
        return -1;
    }
    Ray* rays = reinterpret_cast<Ray*>(rrays);
    Hit* hits = reinterpret_cast<Hit*>(hhits);
    int err = traceRTX_internal(global_state, rays, hits, size);
    return err;
}

DLL_API int cleanRTX() {
#if DEBUG_PRINTS
    fprintf(stderr, "%s[%d]: %s\n", __FILE__, __LINE__, __FUNCTION__);
#endif
    cleanup_internal(global_state);
    return 0;
}

DLL_API uint64_t getHashRTX() {
#if DEBUG_PRINTS
    fprintf(stderr, "%s[%d]: %s\n", __FILE__, __LINE__, __FUNCTION__);
#endif
    if (!global_state.valid) {
        fprintf(stderr, "State is invalid!");
        return uint64_t(-1);
    }
    return global_state.scene.hash;
}
