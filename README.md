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
