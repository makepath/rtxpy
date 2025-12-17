#!/bin/bash
set -euo pipefail

unameOut="$(uname -s)"
case "${unameOut}" in
    Linux*)     machine=Linux;;
    Darwin*)    machine=Mac;;
    CYGWIN*)    machine=Cygwin;;
    MINGW*)     machine=MinGw;;
    *)          machine="UNKNOWN:${unameOut}";;
esac

mkdir -p external/shaders

OPTIX_VERSION=9.1.0

if [ "${machine}" == "Linux" ]
then
    echo "Setting up variables for Linux"

    NVCC="/usr/local/cuda/bin/nvcc"
    COMPILER="g++"

    INCLUDES=(
        -I"./optix_9.1"                         # <-- OptiX 9.1 headers vendored in this repo
        -I"../include"
        -I"/usr/local/cuda/samples/common/inc"  # For helper_math.h / math_helper.h (CUDA samples)
    )

elif [ "${machine}" == "MinGw" ]
then
    echo "Setting up variables for Windows (Git Bash)"

    CUDA_VERSION=11.4
    INCLUDES=(
        -I"./optix_7.1"  # <-- also use vendored headers on Windows
        -I"../include"
        -I"/c/ProgramData/NVIDIA Corporation/CUDA Samples/v${CUDA_VERSION}/common/inc"
    )

    NVCC="/c/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v${CUDA_VERSION}/bin/nvcc"
    COMPILER="/c/Program Files (x86)/Microsoft Visual Studio/2019/BuildTools/VC/Tools/MSVC/14.29.30037/bin/Hostx64/x64"
else
    echo "Unsupported OS : ${machine}"
    exit 1
fi

echo "Compiling for OptiX ${OPTIX_VERSION}"
echo "NVCC compiler currently set: ${NVCC}"
echo "C++ compiler currently set: ${COMPILER}"

NVCC_FLAGS=(
    -m64
    --std=c++11
    --use_fast_math
    -cudart=static
    -arch=sm_86
    -Xptxas -v
)

rm -f kernel.ptx

exec "${NVCC}" "${NVCC_FLAGS[@]}" -ccbin "${COMPILER}" "${INCLUDES[@]}" -ptx -o kernel.ptx kernel.cu \
    >> cudaoutput.txt | tee
