"""Internal CUDA utility functions for GPU kernels.

This module provides low-level vector operations for use in CUDA kernels.
Not part of the public API.
"""

from numba import cuda
import numpy as np


def calc_dims(shape):
    """Calculate CUDA grid and block dimensions for a 2D shape."""
    threadsperblock = (32, 32)
    blockspergrid = (
        (shape[0] + (threadsperblock[0] - 1)) // threadsperblock[0],
        (shape[1] + (threadsperblock[1] - 1)) // threadsperblock[1]
    )
    return blockspergrid, threadsperblock


@cuda.jit(device=True)
def float3(a, b, c):
    """Create a float3 tuple."""
    return (np.float32(a), np.float32(b), np.float32(c))


@cuda.jit(device=True)
def add(a, b):
    """Add two float3 vectors."""
    return float3(a[0] + b[0], a[1] + b[1], a[2] + b[2])


@cuda.jit(device=True)
def diff(a, b):
    """Subtract two float3 vectors (a - b)."""
    return float3(a[0] - b[0], a[1] - b[1], a[2] - b[2])


@cuda.jit(device=True)
def mul(a, b):
    """Multiply a float3 vector by a scalar."""
    return float3(a[0] * b, a[1] * b, a[2] * b)


@cuda.jit(device=True)
def dot(a, b):
    """Compute dot product of two float3 vectors."""
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


@cuda.jit(device=True)
def make_float3(a, offset):
    """Create a float3 from an array at the given offset."""
    return float3(a[offset], a[offset + 1], a[offset + 2])


@cuda.jit(device=True)
def invert(a):
    """Negate a float3 vector."""
    return float3(-a[0], -a[1], -a[2])
