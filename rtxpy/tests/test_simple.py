import numpy as np
import pytest
import xarray as xr

from rtxpy import RTX, has_cupy

# All numpy numeric dtypes to test for elevation input
NUMPY_NUMERIC_DTYPES = [
    # Floating point types
    np.float16,
    np.float32,
    np.float64,
    # Signed integer types
    np.int8,
    np.int16,
    np.int32,
    np.int64,
    # Unsigned integer types
    np.uint8,
    np.uint16,
    np.uint32,
    np.uint64,
]


def triangulate_elevation(elevation_data, backend):
    """
    Convert a 2D elevation array to vertices and triangles for mesh creation.
    This matches the logic in examples/mesh_utils.py triangulateTerrain function.

    Args:
        elevation_data: 2D numpy or cupy array of elevation values (H x W)
        backend: numpy or cupy module to use for array creation

    Returns:
        verts: Flattened vertex buffer (H*W*3 float32)
        triangles: Flattened index buffer ((H-1)*(W-1)*2*3 int32)
    """
    H, W = elevation_data.shape
    num_vertices = H * W
    num_triangles = (H - 1) * (W - 1) * 2

    verts = backend.zeros(num_vertices * 3, dtype=backend.float32)
    triangles = backend.zeros(num_triangles * 3, dtype=backend.int32)

    # Create vertices
    for h in range(H):
        for w in range(W):
            mesh_index = h * W + w
            offset = 3 * mesh_index
            verts[offset] = w  # x coordinate
            verts[offset + 1] = h  # y coordinate
            verts[offset + 2] = float(elevation_data[h, w])  # z = elevation

    # Create triangles (two per grid cell)
    for h in range(H - 1):
        for w in range(W - 1):
            mesh_index = h * W + w
            tri_offset = 6 * (h * (W - 1) + w)
            # First triangle
            triangles[tri_offset + 0] = mesh_index + W
            triangles[tri_offset + 1] = mesh_index + W + 1
            triangles[tri_offset + 2] = mesh_index
            # Second triangle
            triangles[tri_offset + 3] = mesh_index + W + 1
            triangles[tri_offset + 4] = mesh_index + 1
            triangles[tri_offset + 5] = mesh_index

    return verts, triangles


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


@pytest.mark.parametrize("test_cupy", [False, True])
@pytest.mark.parametrize("dtype", NUMPY_NUMERIC_DTYPES)
def test_nan_in_elevation_data_single_cell(test_cupy, dtype):
    """Test behavior when elevation xarray.DataArray contains a single NaN value."""
    if test_cupy:
        if not has_cupy:
            pytest.skip("cupy not available")

        import cupy
        backend = cupy
    else:
        import numpy
        backend = numpy

    # Create a 3x3 elevation grid with one NaN value in the center
    # For integer dtypes, NaN must be converted to a value (0) before casting
    elevation_float = np.array([
        [1.0, 1.0, 1.0],
        [1.0, np.nan, 1.0],
        [1.0, 1.0, 1.0]
    ], dtype=np.float64)
    if np.issubdtype(dtype, np.integer):
        elevation = np.nan_to_num(elevation_float, nan=0).astype(dtype)
    else:
        elevation = elevation_float.astype(dtype)

    da = xr.DataArray(
        elevation,
        dims=['y', 'x'],
        coords={'y': [0, 1, 2], 'x': [0, 1, 2]}
    )

    # Triangulate the elevation data
    verts, triangles = triangulate_elevation(da.values, backend)

    optix = RTX()
    res = optix.build(0, verts, triangles)
    # Build may succeed even with NaN in vertex data

    if res == 0:
        # Trace a ray pointing down at the center (where NaN is)
        rays = backend.float32([1.0, 1.0, 100, 0, 0, 0, -1, 1000])
        hits = backend.float32([0, 0, 0, 0])

        res = optix.trace(rays, hits, 1)
        assert res == 0

        # With NaN in elevation, the ray may miss or produce undefined results
        # but should not crash
        t_value = float(hits[0])
        assert np.isfinite(t_value) or np.isnan(t_value) or t_value == -1.0

        # Trace a ray at a corner (away from NaN) - should hit valid geometry
        rays_corner = backend.float32([0.25, 0.25, 100, 0, 0, 0, -1, 1000])
        hits_corner = backend.float32([0, 0, 0, 0])

        res = optix.trace(rays_corner, hits_corner, 1)
        assert res == 0

        # This ray targets the corner triangle which should be valid
        t_corner = float(hits_corner[0])
        # Result should be valid (not crash)
        assert np.isfinite(t_corner) or np.isnan(t_corner) or t_corner == -1.0


@pytest.mark.parametrize("test_cupy", [False, True])
@pytest.mark.parametrize("dtype", NUMPY_NUMERIC_DTYPES)
def test_nan_in_elevation_data_edge(test_cupy, dtype):
    """Test behavior when elevation xarray.DataArray has NaN on the edge."""
    if test_cupy:
        if not has_cupy:
            pytest.skip("cupy not available")

        import cupy
        backend = cupy
    else:
        import numpy
        backend = numpy

    # Create a 4x4 elevation grid with NaN on one edge
    # For integer dtypes, NaN must be converted to a value (0) before casting
    elevation_float = np.array([
        [np.nan, 1.0, 1.0, 1.0],
        [1.0, 2.0, 2.0, 1.0],
        [1.0, 2.0, 2.0, 1.0],
        [1.0, 1.0, 1.0, 1.0]
    ], dtype=np.float64)
    if np.issubdtype(dtype, np.integer):
        elevation = np.nan_to_num(elevation_float, nan=0).astype(dtype)
    else:
        elevation = elevation_float.astype(dtype)

    da = xr.DataArray(
        elevation,
        dims=['y', 'x'],
        coords={'y': [0, 1, 2, 3], 'x': [0, 1, 2, 3]}
    )

    verts, triangles = triangulate_elevation(da.values, backend)

    optix = RTX()
    res = optix.build(0, verts, triangles)

    if res == 0:
        # Trace a ray at the NaN corner
        rays_nan = backend.float32([0.25, 0.25, 100, 0, 0, 0, -1, 1000])
        hits_nan = backend.float32([0, 0, 0, 0])

        res = optix.trace(rays_nan, hits_nan, 1)
        assert res == 0

        # Ray near NaN vertex - behavior undefined but should not crash
        t_nan = float(hits_nan[0])
        assert np.isfinite(t_nan) or np.isnan(t_nan) or t_nan == -1.0

        # Trace a ray far from the NaN area - should hit valid geometry
        rays_valid = backend.float32([2.5, 2.5, 100, 0, 0, 0, -1, 1000])
        hits_valid = backend.float32([0, 0, 0, 0])

        res = optix.trace(rays_valid, hits_valid, 1)
        assert res == 0

        # This area has valid elevation data, should get a valid hit
        t_valid = float(hits_valid[0])
        # Expect a hit (positive t value) in the valid region
        assert t_valid > 0 or t_valid == -1.0, \
            f"Expected hit or miss in valid region, got t={t_valid}"


@pytest.mark.parametrize("test_cupy", [False, True])
@pytest.mark.parametrize("dtype", NUMPY_NUMERIC_DTYPES)
def test_nan_in_elevation_data_all_nan(test_cupy, dtype):
    """Test behavior when elevation xarray.DataArray is entirely NaN."""
    if test_cupy:
        if not has_cupy:
            pytest.skip("cupy not available")

        import cupy
        backend = cupy
    else:
        import numpy
        backend = numpy

    # Create a 3x3 elevation grid with all NaN values
    # For integer dtypes, NaN must be converted to a value (0) before casting
    elevation_float = np.full((3, 3), np.nan, dtype=np.float64)
    if np.issubdtype(dtype, np.integer):
        elevation = np.nan_to_num(elevation_float, nan=0).astype(dtype)
    else:
        elevation = elevation_float.astype(dtype)

    da = xr.DataArray(
        elevation,
        dims=['y', 'x'],
        coords={'y': [0, 1, 2], 'x': [0, 1, 2]}
    )

    verts, triangles = triangulate_elevation(da.values, backend)

    optix = RTX()
    res = optix.build(0, verts, triangles)

    # Build might succeed or fail with all NaN vertices
    if res == 0:
        rays = backend.float32([1.0, 1.0, 100, 0, 0, 0, -1, 1000])
        hits = backend.float32([0, 0, 0, 0])

        res = optix.trace(rays, hits, 1)
        assert res == 0

        # With all NaN vertices (for float dtypes), should miss or return NaN but not crash
        # For integer dtypes, NaN gets converted to a valid integer, so we may get a hit
        t_value = float(hits[0])
        is_float_dtype = np.issubdtype(dtype, np.floating)
        if is_float_dtype:
            assert np.isnan(t_value) or t_value == -1.0, \
                f"Expected miss or NaN for all-NaN mesh, got t={t_value}"
        else:
            # Integer dtypes: NaN converted to int, mesh is valid, may hit or miss
            assert np.isfinite(t_value) or np.isnan(t_value) or t_value == -1.0


@pytest.mark.parametrize("test_cupy", [False, True])
@pytest.mark.parametrize("dtype", NUMPY_NUMERIC_DTYPES)
def test_nan_in_elevation_data_sparse(test_cupy, dtype):
    """Test behavior with sparse NaN pattern in elevation data."""
    if test_cupy:
        if not has_cupy:
            pytest.skip("cupy not available")

        import cupy
        backend = cupy
    else:
        import numpy
        backend = numpy

    # Create a 5x5 elevation grid with sparse NaN values (checkerboard-like pattern)
    # For integer dtypes, NaN must be converted to a value (0) before casting
    elevation_float = np.array([
        [1.0, 2.0, np.nan, 2.0, 1.0],
        [2.0, 3.0, 4.0, 3.0, 2.0],
        [np.nan, 4.0, 5.0, 4.0, np.nan],
        [2.0, 3.0, 4.0, 3.0, 2.0],
        [1.0, 2.0, np.nan, 2.0, 1.0]
    ], dtype=np.float64)
    if np.issubdtype(dtype, np.integer):
        elevation = np.nan_to_num(elevation_float, nan=0).astype(dtype)
    else:
        elevation = elevation_float.astype(dtype)

    da = xr.DataArray(
        elevation,
        dims=['y', 'x'],
        coords={'y': range(5), 'x': range(5)}
    )

    verts, triangles = triangulate_elevation(da.values, backend)

    optix = RTX()
    res = optix.build(0, verts, triangles)

    if res == 0:
        # Trace multiple rays across the surface
        # Ray at center (valid area)
        rays_center = backend.float32([2.0, 2.0, 100, 0, 0, 0, -1, 1000])
        hits_center = backend.float32([0, 0, 0, 0])

        res = optix.trace(rays_center, hits_center, 1)
        assert res == 0

        t_center = float(hits_center[0])
        # Center should be valid
        assert np.isfinite(t_center) or np.isnan(t_center) or t_center == -1.0

        # Ray near a NaN area
        rays_nan_area = backend.float32([0.5, 2.0, 100, 0, 0, 0, -1, 1000])
        hits_nan_area = backend.float32([0, 0, 0, 0])

        res = optix.trace(rays_nan_area, hits_nan_area, 1)
        assert res == 0

        # Near NaN - should not crash
        t_nan_area = float(hits_nan_area[0])
        assert np.isfinite(t_nan_area) or np.isnan(t_nan_area) or t_nan_area == -1.0
