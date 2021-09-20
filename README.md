# Numba RTX

Requires CMake 3.10 or higher, either system CMake or can

    pip install cmake

To install

    pip install -ve .

Optional runtime dependencies

    cupy

To run tests

    pip install -ve .[tests]
    pytest -v rtx/tests
