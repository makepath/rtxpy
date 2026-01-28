"""
Internal CUDA utility functions for rtxpy terrain analysis.

This module provides low-level CUDA device functions for vector math operations
used by viewshed and hillshade computations.
"""

from numba import cuda
import numpy as np


def calc_dims(shape):
    """Calculate CUDA grid and block dimensions for a 2D array."""
    threadsperblock = (32, 32)
    blockspergrid = (
        (shape[0] + (threadsperblock[0] - 1)) // threadsperblock[0],
        (shape[1] + (threadsperblock[1] - 1)) // threadsperblock[1]
    )
    return blockspergrid, threadsperblock


@cuda.jit(device=True)
def add(a, b):
    return float3(a[0]+b[0], a[1]+b[1], a[2]+b[2])


@cuda.jit(device=True)
def diff(a, b):
    return float3(a[0]-b[0], a[1]-b[1], a[2]-b[2])


@cuda.jit(device=True)
def mul(a, b):
    return float3(a[0]*b, a[1]*b, a[2]*b)


@cuda.jit(device=True)
def multColor(a, b):
    return float3(a[0]*b[0], a[1]*b[1], a[2]*b[2])


@cuda.jit(device=True)
def dot(a, b):
    return a[0]*b[0] + a[1]*b[1] + a[2]*b[2]


@cuda.jit(device=True)
def mix(a, b, k):
    return add(mul(a, k), mul(b, 1-k))


@cuda.jit(device=True)
def make_float3(a, offset):
    return float3(a[offset], a[offset+1], a[offset+2])


@cuda.jit(device=True)
def invert(a):
    return float3(-a[0], -a[1], -a[2])


@cuda.jit(device=True)
def float3(a, b, c):
    return (np.float32(a), np.float32(b), np.float32(c))
