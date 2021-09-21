# RTX

Ray tracing using CUDA accessible from Python

Requires CMake 3.10 or higher, either system CMake or can

    pip install cmake

To install

    pip install -ve .

Optional runtime dependencies

    cupy

If you know the version of the CUDA toolkit that you have installed, which can
be obtained by running `nvcc --version`, you can install the appropriate `cupy`
wheel using

    pip install cupy-cuda101

which is for CUDA toolkit 10.1

To run tests

    pip install -ve .[tests]
    pytest -v rtx/tests
