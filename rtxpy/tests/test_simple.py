import numpy as np
import pytest
from rtxpy import RTX, has_cupy


@pytest.mark.parametrize("test_cupy", [False, True])
def test_simple(test_cupy):
    if test_cupy:
        if not has_cupy:
            pytest.skip("cupy not available")

        import cupy
        backend = cupy
    else:
        import numpy
        backend = numpy

    verts = backend.float32([0,0,0, 1,0,0, 0,1,0, 1,1,0])
    triangles = backend.int32([0,1,2, 2,1,3])

    rays = backend.float32([0.33,0.33,100,0,0,0,-1,1000])
    hits =backend.float32([0,0,0,0])

    try:
        optix = RTX()
    except RuntimeError as e:
        # RTX fails to initialize if CUDA not available.
        if str(e) == "Failed to initialize RTX library":
            pytest.xfail("CUDA not available")
        raise

    res = optix.build(0, verts, triangles)
    assert res == 0

    res = optix.trace(rays,  hits, 1)
    assert res == 0
    np.testing.assert_almost_equal(hits, [100.0, 0.0, 0.0, 1.0])
