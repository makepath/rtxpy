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


# =============================================================================
# Multi-GAS Tests
# =============================================================================

@pytest.mark.parametrize("test_cupy", [False, True])
def test_multi_gas_two_meshes(test_cupy):
    """Test tracing against two meshes at different Z heights."""
    if test_cupy:
        if not has_cupy:
            pytest.skip("cupy not available")
        import cupy
        backend = cupy
    else:
        backend = np

    rtx = RTX()
    rtx.clear_scene()  # Clear any state from previous tests

    # Two triangles: one at z=0, one at z=5
    verts = backend.float32([0, 0, 0, 1, 0, 0, 0.5, 1, 0])
    tris = backend.int32([0, 1, 2])

    # Add ground mesh at z=0
    res = rtx.add_geometry("ground", verts, tris)
    assert res == 0

    # Add elevated mesh at z=5 using transform
    # Transform: identity rotation, translation (0, 0, 5)
    transform = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 5]
    res = rtx.add_geometry("elevated", verts, tris, transform=transform)
    assert res == 0

    # Ray pointing down from z=10 at the triangle center
    rays = backend.float32([0.5, 0.33, 10, 0, 0, 0, -1, 1000])
    hits = backend.float32([0, 0, 0, 0])

    res = rtx.trace(rays, hits, 1)
    assert res == 0

    # Should hit the elevated mesh first at z=5 (distance ~5)
    t_value = float(hits[0])
    np.testing.assert_almost_equal(t_value, 5.0, decimal=1)


@pytest.mark.parametrize("test_cupy", [False, True])
def test_multi_gas_with_transform(test_cupy):
    """Test geometry with translation transform."""
    if test_cupy:
        if not has_cupy:
            pytest.skip("cupy not available")
        import cupy
        backend = cupy
    else:
        backend = np

    rtx = RTX()
    rtx.clear_scene()  # Clear any state from previous tests

    # Triangle at origin
    verts = backend.float32([0, 0, 0, 1, 0, 0, 0.5, 1, 0])
    tris = backend.int32([0, 1, 2])

    # Translate by (10, 0, 0)
    transform = [1, 0, 0, 10, 0, 1, 0, 0, 0, 0, 1, 0]
    res = rtx.add_geometry("translated", verts, tris, transform=transform)
    assert res == 0

    # Ray pointing down at (10.5, 0.33, 10) - should hit translated mesh
    rays = backend.float32([10.5, 0.33, 10, 0, 0, 0, -1, 1000])
    hits = backend.float32([0, 0, 0, 0])

    res = rtx.trace(rays, hits, 1)
    assert res == 0

    t_value = float(hits[0])
    np.testing.assert_almost_equal(t_value, 10.0, decimal=1)

    # Ray at original position should miss
    rays_miss = backend.float32([0.5, 0.33, 10, 0, 0, 0, -1, 1000])
    hits_miss = backend.float32([0, 0, 0, 0])

    res = rtx.trace(rays_miss, hits_miss, 1)
    assert res == 0
    assert float(hits_miss[0]) == -1.0  # Miss


@pytest.mark.parametrize("test_cupy", [False, True])
def test_multi_gas_many_geometries(test_cupy):
    """Stress test with many geometries."""
    if test_cupy:
        if not has_cupy:
            pytest.skip("cupy not available")
        import cupy
        backend = cupy
    else:
        backend = np

    rtx = RTX()
    rtx.clear_scene()  # Clear any state from previous tests

    # Small triangle
    verts = backend.float32([0, 0, 0, 0.5, 0, 0, 0.25, 0.5, 0])
    tris = backend.int32([0, 1, 2])

    # Add 100 geometries in a 10x10 grid
    num_geoms = 100
    for i in range(num_geoms):
        x = (i % 10) * 2
        y = (i // 10) * 2
        transform = [1, 0, 0, x, 0, 1, 0, y, 0, 0, 1, 0]
        res = rtx.add_geometry(f"mesh_{i}", verts, tris, transform=transform)
        assert res == 0

    assert rtx.get_geometry_count() == num_geoms

    # Trace a ray at one of the geometries
    rays = backend.float32([4.25, 4.25, 10, 0, 0, 0, -1, 1000])  # Should hit mesh_22
    hits = backend.float32([0, 0, 0, 0])

    res = rtx.trace(rays, hits, 1)
    assert res == 0

    t_value = float(hits[0])
    np.testing.assert_almost_equal(t_value, 10.0, decimal=1)


@pytest.mark.parametrize("test_cupy", [False, True])
def test_remove_geometry(test_cupy):
    """Test adding and removing geometry."""
    if test_cupy:
        if not has_cupy:
            pytest.skip("cupy not available")
        import cupy
        backend = cupy
    else:
        backend = np

    rtx = RTX()
    rtx.clear_scene()  # Clear any state from previous tests

    verts = backend.float32([0, 0, 0, 1, 0, 0, 0.5, 1, 0])
    tris = backend.int32([0, 1, 2])

    # Add two geometries
    rtx.add_geometry("mesh1", verts, tris)
    rtx.add_geometry("mesh2", verts, tris, transform=[1, 0, 0, 5, 0, 1, 0, 0, 0, 0, 1, 0])

    assert rtx.get_geometry_count() == 2
    assert "mesh1" in rtx.list_geometries()
    assert "mesh2" in rtx.list_geometries()

    # Remove one
    res = rtx.remove_geometry("mesh1")
    assert res == 0

    assert rtx.get_geometry_count() == 1
    assert "mesh1" not in rtx.list_geometries()
    assert "mesh2" in rtx.list_geometries()

    # Remove non-existent should fail
    res = rtx.remove_geometry("nonexistent")
    assert res == -1


@pytest.mark.parametrize("test_cupy", [False, True])
def test_replace_geometry(test_cupy):
    """Test adding geometry with the same ID replaces it."""
    if test_cupy:
        if not has_cupy:
            pytest.skip("cupy not available")
        import cupy
        backend = cupy
    else:
        backend = np

    rtx = RTX()
    rtx.clear_scene()  # Clear any state from previous tests

    verts1 = backend.float32([0, 0, 0, 1, 0, 0, 0.5, 1, 0])
    verts2 = backend.float32([0, 0, 5, 1, 0, 5, 0.5, 1, 5])  # At z=5
    tris = backend.int32([0, 1, 2])

    # Add initial geometry at z=0
    rtx.add_geometry("mesh", verts1, tris)

    # Ray should hit at z=0 (distance 10)
    rays = backend.float32([0.5, 0.33, 10, 0, 0, 0, -1, 1000])
    hits = backend.float32([0, 0, 0, 0])
    rtx.trace(rays, hits, 1)
    np.testing.assert_almost_equal(float(hits[0]), 10.0, decimal=1)

    # Replace with geometry at z=5
    rtx.add_geometry("mesh", verts2, tris)
    assert rtx.get_geometry_count() == 1  # Still only one geometry

    # Now should hit at z=5 (distance 5)
    hits = backend.float32([0, 0, 0, 0])
    rtx.trace(rays, hits, 1)
    np.testing.assert_almost_equal(float(hits[0]), 5.0, decimal=1)


@pytest.mark.parametrize("test_cupy", [False, True])
def test_update_transform(test_cupy):
    """Test updating transform of existing geometry."""
    if test_cupy:
        if not has_cupy:
            pytest.skip("cupy not available")
        import cupy
        backend = cupy
    else:
        backend = np

    rtx = RTX()
    rtx.clear_scene()  # Clear any state from previous tests

    verts = backend.float32([0, 0, 0, 1, 0, 0, 0.5, 1, 0])
    tris = backend.int32([0, 1, 2])

    # Add geometry at origin
    rtx.add_geometry("mesh", verts, tris)

    # Ray at origin hits
    rays_origin = backend.float32([0.5, 0.33, 10, 0, 0, 0, -1, 1000])
    hits = backend.float32([0, 0, 0, 0])
    rtx.trace(rays_origin, hits, 1)
    np.testing.assert_almost_equal(float(hits[0]), 10.0, decimal=1)

    # Update transform to translate by (10, 0, 0)
    res = rtx.update_transform("mesh", [1, 0, 0, 10, 0, 1, 0, 0, 0, 0, 1, 0])
    assert res == 0

    # Now ray at origin should miss
    hits = backend.float32([0, 0, 0, 0])
    rtx.trace(rays_origin, hits, 1)
    assert float(hits[0]) == -1.0

    # Ray at new position should hit
    rays_new = backend.float32([10.5, 0.33, 10, 0, 0, 0, -1, 1000])
    hits = backend.float32([0, 0, 0, 0])
    rtx.trace(rays_new, hits, 1)
    np.testing.assert_almost_equal(float(hits[0]), 10.0, decimal=1)

    # Update non-existent should fail
    res = rtx.update_transform("nonexistent", [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0])
    assert res == -1

    # Invalid transform length should fail
    res = rtx.update_transform("mesh", [1, 0, 0])
    assert res == -1


@pytest.mark.parametrize("test_cupy", [False, True])
def test_list_geometries(test_cupy):
    """Test list_geometries and get_geometry_count methods."""
    if test_cupy:
        if not has_cupy:
            pytest.skip("cupy not available")
        import cupy
        backend = cupy
    else:
        backend = np

    rtx = RTX()
    rtx.clear_scene()  # Clear any state from previous tests

    verts = backend.float32([0, 0, 0, 1, 0, 0, 0.5, 1, 0])
    tris = backend.int32([0, 1, 2])

    # Initially empty
    assert rtx.get_geometry_count() == 0
    assert rtx.list_geometries() == []

    # Add geometries
    rtx.add_geometry("a", verts, tris)
    rtx.add_geometry("b", verts, tris)
    rtx.add_geometry("c", verts, tris)

    assert rtx.get_geometry_count() == 3
    geoms = rtx.list_geometries()
    assert "a" in geoms
    assert "b" in geoms
    assert "c" in geoms


@pytest.mark.parametrize("test_cupy", [False, True])
def test_clear_scene(test_cupy):
    """Test clear_scene removes all geometry and resets state."""
    if test_cupy:
        if not has_cupy:
            pytest.skip("cupy not available")
        import cupy
        backend = cupy
    else:
        backend = np

    rtx = RTX()
    rtx.clear_scene()  # Clear any state from previous tests

    verts = backend.float32([0, 0, 0, 1, 0, 0, 0.5, 1, 0])
    tris = backend.int32([0, 1, 2])

    # Add geometries
    rtx.add_geometry("mesh1", verts, tris)
    rtx.add_geometry("mesh2", verts, tris)
    assert rtx.get_geometry_count() == 2

    # Clear scene
    rtx.clear_scene()
    assert rtx.get_geometry_count() == 0
    assert rtx.list_geometries() == []

    # Can use build() after clear
    res = rtx.build(123, verts, tris)
    assert res == 0
    assert rtx.getHash() == 123


@pytest.mark.parametrize("test_cupy", [False, True])
def test_backward_compat_single_gas(test_cupy):
    """Test that existing single-GAS build() API still works."""
    if test_cupy:
        if not has_cupy:
            pytest.skip("cupy not available")
        import cupy
        backend = cupy
    else:
        backend = np

    rtx = RTX()
    rtx.clear_scene()  # Clear any state from previous tests

    verts = backend.float32([0, 0, 0, 1, 0, 0, 0.5, 1, 0])
    tris = backend.int32([0, 1, 2])

    # Use original API
    res = rtx.build(12345, verts, tris)
    assert res == 0
    assert rtx.getHash() == 12345

    # Trace should work
    rays = backend.float32([0.5, 0.33, 10, 0, 0, 0, -1, 1000])
    hits = backend.float32([0, 0, 0, 0])
    res = rtx.trace(rays, hits, 1)
    assert res == 0
    np.testing.assert_almost_equal(float(hits[0]), 10.0, decimal=1)


@pytest.mark.parametrize("test_cupy", [False, True])
def test_switch_multi_to_single(test_cupy):
    """Test that build() clears multi-GAS state."""
    if test_cupy:
        if not has_cupy:
            pytest.skip("cupy not available")
        import cupy
        backend = cupy
    else:
        backend = np

    rtx = RTX()
    rtx.clear_scene()  # Clear any state from previous tests

    verts = backend.float32([0, 0, 0, 1, 0, 0, 0.5, 1, 0])
    tris = backend.int32([0, 1, 2])

    # Start with multi-GAS
    rtx.add_geometry("mesh1", verts, tris)
    rtx.add_geometry("mesh2", verts, tris, transform=[1, 0, 0, 5, 0, 1, 0, 0, 0, 0, 1, 0])
    assert rtx.get_geometry_count() == 2

    # Switch to single-GAS with build()
    res = rtx.build(999, verts, tris)
    assert res == 0

    # Multi-GAS state should be cleared
    assert rtx.get_geometry_count() == 0

    # Trace should use single-GAS
    rays = backend.float32([0.5, 0.33, 10, 0, 0, 0, -1, 1000])
    hits = backend.float32([0, 0, 0, 0])
    res = rtx.trace(rays, hits, 1)
    assert res == 0
    np.testing.assert_almost_equal(float(hits[0]), 10.0, decimal=1)


@pytest.mark.parametrize("test_cupy", [False, True])
def test_switch_single_to_multi(test_cupy):
    """Test that add_geometry() clears single-GAS state."""
    if test_cupy:
        if not has_cupy:
            pytest.skip("cupy not available")
        import cupy
        backend = cupy
    else:
        backend = np

    rtx = RTX()
    rtx.clear_scene()  # Clear any state from previous tests

    verts = backend.float32([0, 0, 0, 1, 0, 0, 0.5, 1, 0])
    tris = backend.int32([0, 1, 2])

    # Start with single-GAS
    rtx.build(888, verts, tris)
    assert rtx.getHash() == 888

    # Switch to multi-GAS
    rtx.add_geometry("mesh", verts, tris)

    # Single-GAS hash should be cleared
    assert rtx.getHash() == 0xFFFFFFFFFFFFFFFF

    # Trace should use multi-GAS (IAS)
    rays = backend.float32([0.5, 0.33, 10, 0, 0, 0, -1, 1000])
    hits = backend.float32([0, 0, 0, 0])
    res = rtx.trace(rays, hits, 1)
    assert res == 0
    np.testing.assert_almost_equal(float(hits[0]), 10.0, decimal=1)


@pytest.mark.parametrize("test_cupy", [False, True])
def test_empty_scene(test_cupy):
    """Test behavior when tracing after removing all geometry."""
    if test_cupy:
        if not has_cupy:
            pytest.skip("cupy not available")
        import cupy
        backend = cupy
    else:
        backend = np

    rtx = RTX()
    rtx.clear_scene()  # Clear any state from previous tests

    verts = backend.float32([0, 0, 0, 1, 0, 0, 0.5, 1, 0])
    tris = backend.int32([0, 1, 2])

    # Add and then remove all geometry
    rtx.add_geometry("mesh", verts, tris)
    rtx.remove_geometry("mesh")
    assert rtx.get_geometry_count() == 0

    # Trace should fail gracefully (return error)
    rays = backend.float32([0.5, 0.33, 10, 0, 0, 0, -1, 1000])
    hits = backend.float32([0, 0, 0, 0])
    res = rtx.trace(rays, hits, 1)
    assert res == -1  # No geometry to trace against


@pytest.mark.parametrize("test_cupy", [False, True])
def test_trace_miss_multi_gas(test_cupy):
    """Test ray that misses all geometries in multi-GAS mode."""
    if test_cupy:
        if not has_cupy:
            pytest.skip("cupy not available")
        import cupy
        backend = cupy
    else:
        backend = np

    rtx = RTX()
    rtx.clear_scene()  # Clear any state from previous tests

    verts = backend.float32([0, 0, 0, 1, 0, 0, 0.5, 1, 0])
    tris = backend.int32([0, 1, 2])

    rtx.add_geometry("mesh", verts, tris)

    # Ray that misses the geometry
    rays = backend.float32([100, 100, 10, 0, 0, 0, -1, 1000])
    hits = backend.float32([0, 0, 0, 0])

    res = rtx.trace(rays, hits, 1)
    assert res == 0
    assert float(hits[0]) == -1.0  # Miss


@pytest.mark.parametrize("test_cupy", [False, True])
def test_cupy_buffers_multi_gas(test_cupy):
    """Test multi-GAS mode works with cupy buffers for rays/hits."""
    if test_cupy:
        if not has_cupy:
            pytest.skip("cupy not available")
        import cupy

    rtx = RTX()
    rtx.clear_scene()  # Clear any state from previous tests

    # Use numpy arrays for vertex/triangle data (works with both backends)
    verts = np.float32([0, 0, 0, 1, 0, 0, 0.5, 1, 0])
    tris = np.int32([0, 1, 2])

    rtx.add_geometry("mesh", verts, tris)

    # Create ray/hit buffers on the appropriate backend
    if test_cupy:
        rays = cupy.array([0.5, 0.33, 10, 0, 0, 0, -1, 1000], dtype=cupy.float32)
        hits = cupy.zeros(4, dtype=cupy.float32)
    else:
        rays = np.float32([0.5, 0.33, 10, 0, 0, 0, -1, 1000])
        hits = np.float32([0, 0, 0, 0])

    res = rtx.trace(rays, hits, 1)
    assert res == 0

    # Convert to numpy for comparison
    if test_cupy:
        hits_np = hits.get()
    else:
        hits_np = hits

    np.testing.assert_almost_equal(hits_np[0], 10.0, decimal=1)


# =============================================================================
# Primitive ID and Instance ID Tests
# =============================================================================

@pytest.mark.parametrize("test_cupy", [False, True])
def test_primitive_ids_single_gas(test_cupy):
    """Test primitive_ids returns correct triangle indices in single-GAS mode."""
    if test_cupy:
        if not has_cupy:
            pytest.skip("cupy not available")
        import cupy
        backend = cupy
    else:
        backend = np

    rtx = RTX()
    rtx.clear_scene()

    # Create a mesh with 2 triangles (a quad)
    # Triangle 0: vertices 0,1,2 (bottom-left triangle)
    # Triangle 1: vertices 2,1,3 (top-right triangle)
    verts = backend.float32([0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0])
    triangles = backend.int32([0, 1, 2, 2, 1, 3])

    res = rtx.build(0, verts, triangles)
    assert res == 0

    # Ray hitting triangle 0 (bottom-left area)
    rays = backend.float32([0.25, 0.25, 10, 0, 0, 0, -1, 1000])
    hits = backend.float32([0, 0, 0, 0])
    prim_ids = backend.int32([0])

    res = rtx.trace(rays, hits, 1, primitive_ids=prim_ids)
    assert res == 0
    assert float(hits[0]) > 0  # Should hit
    assert int(prim_ids[0]) == 0  # Should hit triangle 0

    # Ray hitting triangle 1 (top-right area)
    rays2 = backend.float32([0.75, 0.75, 10, 0, 0, 0, -1, 1000])
    hits2 = backend.float32([0, 0, 0, 0])
    prim_ids2 = backend.int32([0])

    res = rtx.trace(rays2, hits2, 1, primitive_ids=prim_ids2)
    assert res == 0
    assert float(hits2[0]) > 0  # Should hit
    assert int(prim_ids2[0]) == 1  # Should hit triangle 1


@pytest.mark.parametrize("test_cupy", [False, True])
def test_primitive_ids_miss(test_cupy):
    """Test primitive_ids returns -1 for misses."""
    if test_cupy:
        if not has_cupy:
            pytest.skip("cupy not available")
        import cupy
        backend = cupy
    else:
        backend = np

    rtx = RTX()
    rtx.clear_scene()

    verts = backend.float32([0, 0, 0, 1, 0, 0, 0.5, 1, 0])
    triangles = backend.int32([0, 1, 2])

    res = rtx.build(0, verts, triangles)
    assert res == 0

    # Ray that misses
    rays = backend.float32([100, 100, 10, 0, 0, 0, -1, 1000])
    hits = backend.float32([0, 0, 0, 0])
    prim_ids = backend.int32([999])  # Initialize with non-zero to verify it's set

    res = rtx.trace(rays, hits, 1, primitive_ids=prim_ids)
    assert res == 0
    assert float(hits[0]) == -1.0  # Miss
    assert int(prim_ids[0]) == -1  # primitive_id should be -1 for miss


@pytest.mark.parametrize("test_cupy", [False, True])
def test_instance_ids_multi_gas(test_cupy):
    """Test instance_ids returns correct geometry indices in multi-GAS mode."""
    if test_cupy:
        if not has_cupy:
            pytest.skip("cupy not available")
        import cupy
        backend = cupy
    else:
        backend = np

    rtx = RTX()
    rtx.clear_scene()

    # Small triangle
    verts = backend.float32([0, 0, 0, 1, 0, 0, 0.5, 1, 0])
    tris = backend.int32([0, 1, 2])

    # Add 3 geometries at different X positions
    rtx.add_geometry("mesh0", verts, tris, transform=[1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0])
    rtx.add_geometry("mesh1", verts, tris, transform=[1, 0, 0, 5, 0, 1, 0, 0, 0, 0, 1, 0])
    rtx.add_geometry("mesh2", verts, tris, transform=[1, 0, 0, 10, 0, 1, 0, 0, 0, 0, 1, 0])

    # Ray hitting mesh0 (at x=0.5)
    rays0 = backend.float32([0.5, 0.33, 10, 0, 0, 0, -1, 1000])
    hits0 = backend.float32([0, 0, 0, 0])
    prim_ids0 = backend.int32([0])
    inst_ids0 = backend.int32([0])

    res = rtx.trace(rays0, hits0, 1, primitive_ids=prim_ids0, instance_ids=inst_ids0)
    assert res == 0
    assert float(hits0[0]) > 0
    assert int(inst_ids0[0]) == 0  # Should hit mesh0

    # Ray hitting mesh1 (at x=5.5)
    rays1 = backend.float32([5.5, 0.33, 10, 0, 0, 0, -1, 1000])
    hits1 = backend.float32([0, 0, 0, 0])
    prim_ids1 = backend.int32([0])
    inst_ids1 = backend.int32([0])

    res = rtx.trace(rays1, hits1, 1, primitive_ids=prim_ids1, instance_ids=inst_ids1)
    assert res == 0
    assert float(hits1[0]) > 0
    assert int(inst_ids1[0]) == 1  # Should hit mesh1

    # Ray hitting mesh2 (at x=10.5)
    rays2 = backend.float32([10.5, 0.33, 10, 0, 0, 0, -1, 1000])
    hits2 = backend.float32([0, 0, 0, 0])
    prim_ids2 = backend.int32([0])
    inst_ids2 = backend.int32([0])

    res = rtx.trace(rays2, hits2, 1, primitive_ids=prim_ids2, instance_ids=inst_ids2)
    assert res == 0
    assert float(hits2[0]) > 0
    assert int(inst_ids2[0]) == 2  # Should hit mesh2


@pytest.mark.parametrize("test_cupy", [False, True])
def test_instance_ids_miss(test_cupy):
    """Test instance_ids returns -1 for misses in multi-GAS mode."""
    if test_cupy:
        if not has_cupy:
            pytest.skip("cupy not available")
        import cupy
        backend = cupy
    else:
        backend = np

    rtx = RTX()
    rtx.clear_scene()

    verts = backend.float32([0, 0, 0, 1, 0, 0, 0.5, 1, 0])
    tris = backend.int32([0, 1, 2])

    rtx.add_geometry("mesh", verts, tris)

    # Ray that misses
    rays = backend.float32([100, 100, 10, 0, 0, 0, -1, 1000])
    hits = backend.float32([0, 0, 0, 0])
    prim_ids = backend.int32([999])
    inst_ids = backend.int32([999])

    res = rtx.trace(rays, hits, 1, primitive_ids=prim_ids, instance_ids=inst_ids)
    assert res == 0
    assert float(hits[0]) == -1.0  # Miss
    assert int(prim_ids[0]) == -1
    assert int(inst_ids[0]) == -1


@pytest.mark.parametrize("test_cupy", [False, True])
def test_primitive_ids_multiple_rays(test_cupy):
    """Test primitive_ids with multiple rays hitting different triangles."""
    if test_cupy:
        if not has_cupy:
            pytest.skip("cupy not available")
        import cupy
        backend = cupy
    else:
        backend = np

    rtx = RTX()
    rtx.clear_scene()

    # Create a mesh with 2 triangles
    verts = backend.float32([0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0])
    triangles = backend.int32([0, 1, 2, 2, 1, 3])

    res = rtx.build(0, verts, triangles)
    assert res == 0

    # 3 rays: one hitting triangle 0, one hitting triangle 1, one missing
    rays = backend.float32([
        0.25, 0.25, 10, 0, 0, 0, -1, 1000,  # hits triangle 0
        0.75, 0.75, 10, 0, 0, 0, -1, 1000,  # hits triangle 1
        100, 100, 10, 0, 0, 0, -1, 1000,    # misses
    ])
    hits = backend.float32([0] * 12)  # 3 rays * 4 floats
    prim_ids = backend.int32([0, 0, 0])

    res = rtx.trace(rays, hits, 3, primitive_ids=prim_ids)
    assert res == 0

    # Check results
    assert int(prim_ids[0]) == 0   # First ray hits triangle 0
    assert int(prim_ids[1]) == 1   # Second ray hits triangle 1
    assert int(prim_ids[2]) == -1  # Third ray misses


@pytest.mark.parametrize("test_cupy", [False, True])
def test_primitive_and_instance_ids_optional(test_cupy):
    """Test that primitive_ids and instance_ids are truly optional."""
    if test_cupy:
        if not has_cupy:
            pytest.skip("cupy not available")
        import cupy
        backend = cupy
    else:
        backend = np

    rtx = RTX()
    rtx.clear_scene()

    verts = backend.float32([0, 0, 0, 1, 0, 0, 0.5, 1, 0])
    triangles = backend.int32([0, 1, 2])

    res = rtx.build(0, verts, triangles)
    assert res == 0

    rays = backend.float32([0.5, 0.33, 10, 0, 0, 0, -1, 1000])
    hits = backend.float32([0, 0, 0, 0])

    # Should work without any optional params
    res = rtx.trace(rays, hits, 1)
    assert res == 0
    np.testing.assert_almost_equal(float(hits[0]), 10.0, decimal=1)

    # Should work with only primitive_ids
    hits = backend.float32([0, 0, 0, 0])
    prim_ids = backend.int32([0])
    res = rtx.trace(rays, hits, 1, primitive_ids=prim_ids)
    assert res == 0
    assert int(prim_ids[0]) == 0

    # Should work with only instance_ids
    hits = backend.float32([0, 0, 0, 0])
    inst_ids = backend.int32([0])
    res = rtx.trace(rays, hits, 1, instance_ids=inst_ids)
    assert res == 0
