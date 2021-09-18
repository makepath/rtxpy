#!/bin/bash

unameOut="$(uname -s)"
case "${unameOut}" in
    Linux*)     machine=Linux;;
    Darwin*)    machine=Mac;;
    CYGWIN*)    machine=Cygwin;;
    MINGW*)     machine=MinGw;;
    *)          machine="UNKNOWN:${unameOut}"
esac

if [ ! -d "external/shaders" ]
then
	mkdir external/shaders
fi

if [ "${machine}" == "Linux" ]
then
	echo "Setting up variables for Linux"
	export OPTIX_VERSION=7.1.0
	export INCLUDES="-I'/<PATH_TO>/NVIDIA-OptiX-SDK-${OPTIX_VERSION}-linux64-x86_64/include'"
	export INCLUDES="$INCLUDES -I'../include'"
	export INCLUDES="$INCLUDES -I'/usr/local/cuda/samples/common/inc'" #For math_helper.h
	export NVCC="/usr/local/cuda/bin/nvcc"
	export COMPILER="g++"
else 
	if [ "${machine}" == "MinGw" ]
	then
		echo "Setting up variables for Windows (Git Bash)"

		export OPTIX_VERSION=7.1.0
		export CUDA_VERSION=11.4
		export INCLUDES=(-I"/c/ProgramData/NVIDIA Corporation/OptiX SDK $OPTIX_VERSION/include" -I"../include" -I"/c/ProgramData/NVIDIA Corporation/CUDA Samples/v${CUDA_VERSION}/common/inc")
		export NVCC="/c/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v${CUDA_VERSION}/bin/nvcc"
		# You may need to update the path to a valid compiler. This points to MSVS 2019 compiler
		export COMPILER="/c/Program Files (x86)/Microsoft Visual Studio/2019/BuildTools/VC/Tools/MSVC/14.29.30037/bin/Hostx64/x64"
	else
		echo "Unsupported OS : ${machine}"
	fi
fi

echo "Compiling for OptiX $OPTIX_VERSION"
echo "NVCC compiler currently set: $NVCC"
echo "C++ compiler currently set: $COMPILER"

export NVCC_FLAGS="-m64 --std c++11 --use_fast_math -cudart static -arch sm_50 -Xptxas -v"

if [ -f "kernel.ptx" ]
then
	rm kernel.ptx
fi

exec "$NVCC" $NVCC_FLAGS -ccbin "$COMPILER" "${INCLUDES[@]}" -ptx -o kernel.ptx  kernel.cu >> cudaoutput.txt | tee
