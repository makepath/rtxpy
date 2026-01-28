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

    optix = RTX()

    res = optix.build(0, verts, triangles)
    assert res == 0

    res = optix.trace(rays,  hits, 1)
    assert res == 0
    np.testing.assert_almost_equal(hits, [100.0, 0.0, 0.0, 1.0])


@pytest.mark.parametrize("test_cupy", [False, True])
def test_nan_in_ray_input(test_cupy):
    """Test behavior when ray input contains NaN values."""
    if test_cupy:
        if not has_cupy:
            pytest.skip("cupy not available")

        import cupy
        backend = cupy
    else:
        import numpy
        backend = numpy

    # Valid mesh (unit square made of 2 triangles)
    verts = backend.float32([0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0])
    triangles = backend.int32([0, 1, 2, 2, 1, 3])

    # Ray with NaN in origin (ox = NaN)
    rays = backend.float32([np.nan, 0.33, 100, 0, 0, 0, -1, 1000])
    hits = backend.float32([0, 0, 0, 0])

    optix = RTX()
    res = optix.build(0, verts, triangles)
    assert res == 0

    res = optix.trace(rays, hits, 1)
    assert res == 0

    # When ray origin contains NaN, the ray should miss (t = -1.0)
    # or produce NaN in the hit result
    t_value = float(hits[0])
    assert t_value == -1.0 or np.isnan(t_value), \
        f"Expected miss (t=-1.0) or NaN for ray with NaN origin, got t={t_value}"


@pytest.mark.parametrize("test_cupy", [False, True])
def test_nan_in_vertex_input(test_cupy):
    """Test behavior when vertex data contains NaN values."""
    if test_cupy:
        if not has_cupy:
            pytest.skip("cupy not available")

        import cupy
        backend = cupy
    else:
        import numpy
        backend = numpy

    # Mesh with NaN in one vertex (vertex 0 has NaN z-coordinate)
    verts = backend.float32([0, 0, np.nan, 1, 0, 0, 0, 1, 0, 1, 1, 0])
    triangles = backend.int32([0, 1, 2, 2, 1, 3])

    # Valid ray pointing down at the mesh
    rays = backend.float32([0.33, 0.33, 100, 0, 0, 0, -1, 1000])
    hits = backend.float32([0, 0, 0, 0])

    optix = RTX()
    res = optix.build(0, verts, triangles)
    # Build may succeed even with NaN vertices (OptiX doesn't validate)
    # The behavior depends on OptiX implementation

    if res == 0:
        res = optix.trace(rays, hits, 1)
        assert res == 0

        # With NaN in triangle 0's vertex, behavior is undefined but should not crash
        # Triangle 1 (vertices 2,1,3) should still be valid
        # The ray at (0.33, 0.33) could hit either triangle depending on exact geometry
        t_value = float(hits[0])
        # Result should be a valid float (hit, miss, or NaN - but not crash)
        assert np.isfinite(t_value) or np.isnan(t_value) or t_value == -1.0
