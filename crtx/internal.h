#pragma once

#include <stdint.h>
#include <stddef.h>
#include <stdio.h>
#include <string>
#include <fstream>
#include <sstream>

// NOTE: dllmain.cpp must include <cuda.h> and <optix.h> BEFORE including this file.
#ifdef WIN32
#define __align__(X) __declspec(align(X))
#else
#define __align__(X) __attribute__((aligned(X)))
#endif

template <typename T>
struct Record {
    __align__(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
    T data;
};

struct Empty {};
typedef Record<Empty>   RayGenSbtRecord;
typedef Record<Empty>   MissSbtRecord;
typedef Record<Empty>   HitGroupSbtRecord;

#define OPTIX_CHECK( call )                                                    \
    do                                                                         \
    {                                                                          \
        OptixResult res = call;                                                \
        if( res != OPTIX_SUCCESS )                                             \
        {                                                                      \
            std::stringstream ss;                                              \
            ss << "Optix call '" << #call << "' failed with code("             \
               << res << "): " __FILE__ ":"                                    \
               << __LINE__ << ")\n";                                           \
            fprintf(stderr, "[OptiX Error] %s\n", ss.str().c_str());           \
            return res;                                                        \
        }                                                                      \
    } while( 0 )

#define OPTIX_CHECK_LOG( call )                                                \
    do                                                                         \
    {                                                                          \
        OptixResult res = call;                                                \
        const size_t sizeof_log_returned = logSize;                            \
        logSize = sizeof( log );                                               \
        if( res != OPTIX_SUCCESS )                                             \
        {                                                                      \
            std::stringstream ss;                                              \
            ss << "Optix call '" << #call << "' failed with code("             \
               << res << "): " __FILE__ ":"                                    \
               << __LINE__ << ")\nLog:\n" << log                               \
               << ( sizeof_log_returned > sizeof( log ) ? "<TRUNCATED>" : "" ) \
               << "\n";                                                        \
            fprintf(stderr, "[OptiX Error] %s\n", ss.str().c_str());           \
            return res;                                                        \
        }                                                                      \
    } while( 0 )

inline void checkCuda(int err, const char* msg, const char* fn, int line) {
    if (!msg || msg[0] == '\0') {
        fprintf(stderr, "CUDA Error[%d] at %s[%d]\n", err, fn, line);
    } else {
        fprintf(stderr, "CUDA Error[%d] at %s[%d] : %s\n", err, fn, line, msg);
    }
}

#define CHECK_CUDA_LOG(err, msg) \
    do { \
        if ((err) != CUDA_SUCCESS) { \
            checkCuda(int(err), msg, __FUNCTION__, __LINE__); \
            return err; \
        } \
    } while (false)

#define CHECK_CUDA(err) CHECK_CUDA_LOG(err, "")

struct Scene {
    Scene() {
        accelerationStructureHandle = 0;
        memory = 0;
        hash = uint64_t(-1);
    }
    ~Scene() { freeMem(); }

    void freeMem() {
        if (memory) {
            cuMemFree(memory);
            memory = 0;
        }
    }

    uint64_t hash;
    CUdeviceptr memory;
    OptixTraversableHandle accelerationStructureHandle;
};

struct State {
    OptixDeviceContext          context = 0;
    Scene                       scene;

    OptixPipelineCompileOptions pipeline_compile_options = {};
    OptixModule                 ptx_module = 0;
    OptixPipeline               pipeline = 0;

    OptixProgramGroup raygen = 0;
    OptixProgramGroup miss   = 0;
    OptixProgramGroup hit    = 0;

    OptixShaderBindingTable sbt = {};

    CUdeviceptr d_params = 0;

    CUdeviceptr d_rays = 0;
    size_t      d_rays_size = 0;
    CUdeviceptr d_hits = 0;
    size_t      d_hits_size = 0;

    struct {
        CUdevice  device  = 0;
        CUstream  stream  = 0;
        CUcontext context = 0;
    } cuda;

    bool valid = false;
};

// Read a PTX file into a std::string (throws on failure)
inline std::string load_ptx_file(const char* path)
{
    std::ifstream f(path, std::ios::in | std::ios::binary);
    if (!f) {
        throw std::runtime_error(std::string("Could not open PTX file: ") + path);
    }

    f.seekg(0, std::ios::end);
    std::streamoff size = f.tellg();
    f.seekg(0, std::ios::beg);

    if (size <= 0) {
        throw std::runtime_error(std::string("PTX file is empty: ") + path);
    }

    std::string ptx;
    ptx.resize(static_cast<size_t>(size));
    f.read(&ptx[0], size);
    if (!f) {
        throw std::runtime_error(std::string("Failed to read PTX file: ") + path);
    }
    return ptx;
}
