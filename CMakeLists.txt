cmake_minimum_required(VERSION 3.10)
project(rtxpy)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_definitions(-DRTX_EXPORTS)

if (WIN32)
  add_definitions(-D_CRT_SECURE_NO_WARNINGS)
endif()

set(SOURCE_DIR "crtx")

set(HEADERS
  ${SOURCE_DIR}/common.h
  ${SOURCE_DIR}/internal.h
  ${SOURCE_DIR}/rtx.h
)

set(SOURCES
  ${SOURCE_DIR}/dllmain.cpp
  ${SOURCE_DIR}/cuew/cuew.c
)

add_library(${PROJECT_NAME} SHARED ${HEADERS} ${SOURCES})
target_compile_definitions(${PROJECT_NAME} PRIVATE CUDA_NO_PROTOTYPES OPTIX_DONT_INCLUDE_CUDA)

# ---- CUDA toolkit path (adjust if yours differs) ----
set(CUDA_TOOLKIT_ROOT_DIR "/usr/local/cuda")
set(CUDA_INCLUDE_DIR "${CUDA_TOOLKIT_ROOT_DIR}/include")
set(CUDA_LIB_DIR "${CUDA_TOOLKIT_ROOT_DIR}/lib64")

target_include_directories(${PROJECT_NAME} PRIVATE
  ${SOURCE_DIR}
  ${SOURCE_DIR}/optix_9.1/include
  ${SOURCE_DIR}/optix_9.1
  ${SOURCE_DIR}/cuew
  ${CUDA_INCLUDE_DIR}
)

# Link search paths:
# - CUDA toolkit libs (cudart, etc.) live here
# - WSL provides the NVIDIA driver libcuda.so here
target_link_directories(${PROJECT_NAME} PRIVATE
  ${CUDA_LIB_DIR}
  /usr/lib/wsl/lib
)

target_link_libraries(${PROJECT_NAME} PRIVATE
  cuda      # libcuda.so (driver API)
  dl
  pthread
)

# Ensure runtime can find libcuda.so on WSL
target_link_options(${PROJECT_NAME} PRIVATE
  "-Wl,-rpath,/usr/lib/wsl/lib"
)
