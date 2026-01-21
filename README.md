# RTXpy

Ray tracing using CUDA, accessible from Python.

## Hardware requirements

  * Nvidia Maxwell GPU or newer
  * Nvidia driver version:
    * 456.71 or newer for Windows
    * 455.28 or newer for Linux

## Installation

    pip install rtxpy

## Installation from source

Requires CMake 3.10 or higher to build.

To install RTXpy from source use

    pip install -ve .

`cupy` is an optional runtime dependency. If you know the version of the CUDA
toolkit you have installed, which can be obtained by running `nvcc --version`,
you can install the appropriate `cupy` wheel. For example, for CUDA toolkit
11.2 use

    pip install cupy-cuda112

To run tests

    pip install -ve .[tests]
    pytest -v rtxpy/tests


## Building from source:

### Building kernel.ptx
```bash
cd crtx
bash compileOptiX.sh
cp kernel.ptx ../rtxpy
```

### Building `librtxpy.so`
```bash
bash clean_build.sh
cp build/librtxpy.so ./rtxpy
```

### Building on WSL2:
To get the build working on WSL, I followed the post below:
https://forums.developer.nvidia.com/t/problem-running-optix-7-6-in-wsl/239355/8

---------------------

Welcome @chris.schwindt,

I believe we’re not yet packaging OptiX into the WSL2 driver. I believe this is hung up on a redesign of the driver packaging and delivery process, which is why it’s taking such a long time.

I have heard rumors that people have been able to get OptiX to work in WSL2 via manual install. This is unofficial and subject to change, so your mileage may vary, but here are some steps that may work for you:

Running OptiX Applications on WSL 2
Install WSL 2 and enable CUDA
Follow the canonical methods for installing WSL, display driver, and CUDA Toolkit within WSL

As mentioned in the docs, do not install a Linux Display driver in WSL, this will break the mapping of libcuda.
There are CUDA Toolkit downloads specifically for WSL that will not attempt to install a driver, only the toolkit.
You can also deselect the driver in a normal version of the toolkit.
Obtain OptiX / RTCore libraries for Linux
Download and extract libraries from the linux display driver.
You can run the driver installer in WSL using ./[driver filename].run -x which will unpack the driver but not install it.
Copy libnvoptix.so.XXX.00, libnvidia-rtcore.so.XXX.00, and libnvidia-ptxjitcompiler.so.XXX.00 into C:/Windows/System32/lxss/lib where XXX is the driver version.
Rename libnvoptix.so.XX.00 to libnvoptix.so.1
Rename libnvidia-ptxjitcompiler.so.XXX.00 to libnvidia-ptxjitcompiler.so.1
Do not rename libnvidia-rtcore.so.XXX.00
Be aware that future drivers may need additional libraries that will need to be copied.
Building an OptiX Application
You may need to add /usr/local/cuda/bin to your PATH to access NVCC, but do NOT add /usr/local/cuda/lib64 to LD_LIBRARY_PATH as you normally would when installing the CUDA toolkit. libcuda and other libraries are passed through from C:/Windows/System32/lxss/lib where you placed the OptiX and RTCore libs.
Instead, add /usr/lib/wsl/lib to your LD_LIBRARY_PATH to pick up CUDA, OptiX, etc.
Running an OptiX Application
With LD_LIBRARY_PATH set per the previous step, you should be able to run an OptiX executable.
You may need to rebuild the WSL cache. You can do so by quitting any WSL sessions and running wsl --shutdown from Powershell, then starting a new WSL session. Failing to reset the cache may lead to strange load paths.
You may verify paths are correct using strace, e.g., strace -o trace ./bin/optixHello
–
David.

---------------------

I ended up downloading: https://uk.download.nvidia.com/XFree86/Linux-x86_64/590.44.01/NVIDIA-Linux-x86_64-590.44.01.run
Nvidia Driver: 591.44

I then extract files and followed instructions above


I then extracted 
```bash
bash 
```
