#include <assert.h>
#include "cuew.h"

#define OPTIX_DONT_INCLUDE_CUDA
#include <optix.h>
#include <optix_function_table_definition.h>
#include <optix_stack_size.h>
#include <optix_stubs.h>

#include <sstream>

#include <vector>
#include <fstream>

struct float3 { float x,y,z; };
struct int3   {   int i[3]; };
#include "common.h"

#include "rtx.h"
#include "internal.h"

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


int createModule(State& state)
{
    char   log[2048];  // For error reporting from OptiX creation functions
    size_t logSize = sizeof( log );

    OptixModuleCompileOptions module_compile_options = {};
    module_compile_options.maxRegisterCount          = OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT;

    module_compile_options.optLevel   = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
    module_compile_options.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_LINEINFO;

    state.pipeline_compile_options.usesMotionBlur        = false;
    state.pipeline_compile_options.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
    state.pipeline_compile_options.numPayloadValues      = 4;
    state.pipeline_compile_options.numAttributeValues    = 2;
    state.pipeline_compile_options.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;  // should be OPTIX_EXCEPTION_FLAG_STACK_OVERFLOW;
    state.pipeline_compile_options.pipelineLaunchParamsVariableName = "params";

#if 0
    std::string input2 = getTextFileContents("kernel.ptx");
    size_t      inputSize2 = input2.length();
    input2 += "\0\0\0";
    unsigned* asd = reinterpret_cast<unsigned*>(&input2[0]);
    for (int i = 0; i < inputSize2/4; i++) {
        fprintf(stderr, "0x%x, ", asd[i]);
        if (i % 20 == 0 && i > 0) {
            fprintf(stderr, "\n");
        }
    }
#endif

    std::string input(reinterpret_cast<const char*>(buff));
    size_t      inputSize = input.length();

    OPTIX_CHECK_LOG(
        optixModuleCreateFromPTX(
            state.context, 
            &module_compile_options, 
            &state.pipeline_compile_options,
            &input[0], 
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
    size_t sizeof_log = sizeof( log );

    const uint32_t    max_trace_depth   = 1;
    OptixProgramGroup program_groups[3] = {state.raygen, state.miss, state.hit};

    OptixPipelineLinkOptions pipeline_link_options = {};
    pipeline_link_options.maxTraceDepth            = max_trace_depth;
    pipeline_link_options.debugLevel               = OPTIX_COMPILE_DEBUG_LEVEL_FULL;

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
        fprintf(stderr, "Failed to create OptiX Pipeline.");
        return -1;
    }
    OptixStackSizes stack_sizes = {};
    for(auto& prog_group : program_groups) {
        OPTIX_CHECK(optixUtilAccumulateStackSizes(prog_group, &stack_sizes));
    }

    uint32_t direct_callable_stack_size_from_traversal;
    uint32_t direct_callable_stack_size_from_state;
    uint32_t continuation_stack_size;
    OPTIX_CHECK(
        optixUtilComputeStackSizes(
            &stack_sizes, 
            max_trace_depth,
            0,  // maxCCDepth
            0,  // maxDCDEpth
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
            1  // maxTraversableDepth
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
    //In case cleanup has already been called, don't call it again
    if (state.cuda.context == 0) {
        return 0;
    }
    state.valid = false;
    state.scene.freeMem();
    
    OPTIX_CHECK(optixPipelineDestroy(state.pipeline));
    OPTIX_CHECK(optixProgramGroupDestroy(state.raygen));
    OPTIX_CHECK(optixProgramGroupDestroy(state.miss));
    OPTIX_CHECK(optixProgramGroupDestroy(state.hit));
    state.pipeline = 0;
    state.raygen = 0;
    state.miss = 0;
    state.hit = 0;
    
    OPTIX_CHECK(optixModuleDestroy(state.ptx_module));
    OPTIX_CHECK(optixDeviceContextDestroy(state.context));
    state.ptx_module = 0;
    state.context = 0;

    CUresult err = CUDA_SUCCESS;
    err = cuMemFree(state.sbt.raygenRecord);
    assert(err == CUDA_SUCCESS);
    err = cuMemFree(state.sbt.missRecordBase);
    assert(err == CUDA_SUCCESS);
    err = cuMemFree(state.sbt.hitgroupRecordBase);
    assert(err == CUDA_SUCCESS);
    memset(&state.sbt, 0, sizeof(state.sbt));

    err = cuMemFree(state.d_params);
    assert(err == CUDA_SUCCESS);
    err = cuMemFree(state.d_rays);
    assert(err == CUDA_SUCCESS);
    err = cuMemFree(state.d_hits);
    assert(err == CUDA_SUCCESS);
    state.d_params = 0;
    state.d_hits = 0;
    state.d_rays = 0;
    state.d_hits_size = 0;
    state.d_rays_size = 0;

    err = cuStreamDestroy(state.cuda.stream);
    assert(err == CUDA_SUCCESS);

    err = cuDevicePrimaryCtxRelease(state.cuda.device);
    assert(err == CUDA_SUCCESS);

    state.cuda.stream = 0;
    state.cuda.device = 0;
    state.cuda.context = 0;

    return err;
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

    OPTIX_CHECK(optixInit());

	OptixDeviceContextOptions options;
	options.logCallbackLevel = 0; //MAX verbosity 4
	options.logCallbackFunction = contextLogCallback;
	options.logCallbackData = nullptr;
#if OPTIX_VERSION >= 70300
#if DEBUG_PRINTS
	options.validationMode = OPTIX_DEVICE_CONTEXT_VALIDATION_MODE_ALL;
#endif //DEBUG_PRINTS
#endif //OPTIX_VERSION >= 70300
    OPTIX_CHECK(optixDeviceContextCreate(state.cuda.context, &options, &state.context));
    if (!state.context) {
        return -1;
    }
    if (createModule(state)) {
        return -1;
    }
    if (createProgramGroups(state)) {
        return -1;
    }
    if (createPipelines(state)) {
        return -1;
    }
    if (createSBT(state)) {
        return -1;
    }

    state.valid = true;
    return err;
}

int initBuffers_internal(State& state, int numRays) {
    if (!state.valid) {
        fprintf(stderr, "State is invalid!");
        return -2;
    }
    CUresult err = CUDA_SUCCESS;
    size_t newRaysSize = sizeof(Ray) * numRays;
    size_t newHitsSize = sizeof(Hit) * numRays;
    if (newRaysSize != state.d_rays_size || newHitsSize != state.d_hits_size) {
        err = cuMemFree(state.d_rays);
        CHECK_CUDA_LOG(err, "Failed to deallocate old input data buffer");
        err = cuMemFree(state.d_hits);
        CHECK_CUDA_LOG(err, "Failed to deallocate old output data buffer");

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
