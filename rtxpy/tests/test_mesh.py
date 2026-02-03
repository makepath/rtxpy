"""Tests for mesh utilities."""

import numpy as np
import pytest

from rtxpy import (
    triangulate_terrain,
    write_stl,
    load_obj,
    make_transform,
    make_transforms_on_terrain,
)
from rtxpy.rtx import has_cupy


class TestTriangulateTerrain:
    """Tests for triangulate_terrain function."""

    def test_simple_terrain_cpu(self):
        """Test triangulation of a simple 3x3 terrain on CPU."""
        H, W = 3, 3
        terrain = np.array([
            [0.0, 1.0, 0.0],
            [1.0, 2.0, 1.0],
            [0.0, 1.0, 0.0],
        ], dtype=np.float32)

        num_verts = H * W
        num_tris = (H - 1) * (W - 1) * 2
        verts = np.zeros(num_verts * 3, dtype=np.float32)
        triangles = np.zeros(num_tris * 3, dtype=np.int32)

        result = triangulate_terrain(verts, triangles, terrain)
        assert result == 0

        # Check vertex count
        assert len(verts) == 27  # 9 vertices * 3 components

        # Check that z values match elevation
        for h in range(H):
            for w in range(W):
                idx = (h * W + w) * 3
                assert verts[idx] == w  # x
                assert verts[idx + 1] == h  # y
                assert verts[idx + 2] == terrain[h, w]  # z

        # Check triangle count
        assert len(triangles) == 24  # 8 triangles * 3 indices

    def test_terrain_with_scale(self):
        """Test that scale parameter affects z values."""
        H, W = 2, 2
        terrain = np.array([
            [1.0, 1.0],
            [1.0, 1.0],
        ], dtype=np.float32)

        num_verts = H * W
        num_tris = (H - 1) * (W - 1) * 2
        verts = np.zeros(num_verts * 3, dtype=np.float32)
        triangles = np.zeros(num_tris * 3, dtype=np.int32)

        scale = 10.0
        triangulate_terrain(verts, triangles, terrain, scale=scale)

        # All z values should be scaled
        for i in range(num_verts):
            assert verts[i * 3 + 2] == 10.0  # 1.0 * 10.0

    def test_flat_terrain(self):
        """Test triangulation of flat terrain."""
        H, W = 4, 4
        terrain = np.zeros((H, W), dtype=np.float32)

        num_verts = H * W
        num_tris = (H - 1) * (W - 1) * 2
        verts = np.zeros(num_verts * 3, dtype=np.float32)
        triangles = np.zeros(num_tris * 3, dtype=np.int32)

        result = triangulate_terrain(verts, triangles, terrain)
        assert result == 0

        # All z values should be 0
        for i in range(num_verts):
            assert verts[i * 3 + 2] == 0.0

    @pytest.mark.skipif(not has_cupy, reason="cupy not available")
    def test_simple_terrain_gpu(self):
        """Test triangulation on GPU with cupy arrays."""
        import cupy

        H, W = 3, 3
        terrain = cupy.array([
            [0.0, 1.0, 0.0],
            [1.0, 2.0, 1.0],
            [0.0, 1.0, 0.0],
        ], dtype=cupy.float32)

        num_verts = H * W
        num_tris = (H - 1) * (W - 1) * 2
        verts = cupy.zeros(num_verts * 3, dtype=cupy.float32)
        triangles = cupy.zeros(num_tris * 3, dtype=cupy.int32)

        result = triangulate_terrain(verts, triangles, terrain)
        assert result == 0

        # Convert to numpy for checking
        verts_np = cupy.asnumpy(verts)
        terrain_np = cupy.asnumpy(terrain)

        for h in range(H):
            for w in range(W):
                idx = (h * W + w) * 3
                assert verts_np[idx] == w
                assert verts_np[idx + 1] == h
                assert verts_np[idx + 2] == terrain_np[h, w]


class TestWriteStl:
    """Tests for write_stl function."""

    def test_write_simple_mesh(self, tmp_path):
        """Test writing a simple mesh to STL."""
        # Simple two-triangle mesh (unit square)
        verts = np.array([
            0, 0, 0,  # vertex 0
            1, 0, 0,  # vertex 1
            0, 1, 0,  # vertex 2
            1, 1, 0,  # vertex 3
        ], dtype=np.float32)
        triangles = np.array([
            0, 1, 2,  # triangle 0
            2, 1, 3,  # triangle 1
        ], dtype=np.int32)

        filepath = tmp_path / "test.stl"
        write_stl(str(filepath), verts, triangles)

        # Check file exists and has correct size
        assert filepath.exists()
        # STL header: 80 bytes
        # Triangle count: 4 bytes
        # 2 triangles * 50 bytes each: 100 bytes
        expected_size = 80 + 4 + 2 * 50
        assert filepath.stat().st_size == expected_size

    def test_write_triangulated_terrain(self, tmp_path):
        """Test writing triangulated terrain to STL."""
        H, W = 3, 3
        terrain = np.array([
            [0.0, 1.0, 0.0],
            [1.0, 2.0, 1.0],
            [0.0, 1.0, 0.0],
        ], dtype=np.float32)

        num_verts = H * W
        num_tris = (H - 1) * (W - 1) * 2
        verts = np.zeros(num_verts * 3, dtype=np.float32)
        triangles = np.zeros(num_tris * 3, dtype=np.int32)

        triangulate_terrain(verts, triangles, terrain)

        filepath = tmp_path / "terrain.stl"
        write_stl(str(filepath), verts, triangles)

        assert filepath.exists()
        # 8 triangles * 50 bytes + 84 bytes header
        expected_size = 80 + 4 + 8 * 50
        assert filepath.stat().st_size == expected_size

    @pytest.mark.skipif(not has_cupy, reason="cupy not available")
    def test_write_from_cupy_arrays(self, tmp_path):
        """Test that cupy arrays are converted automatically."""
        import cupy

        verts = cupy.array([
            0, 0, 0,
            1, 0, 0,
            0.5, 1, 0,
        ], dtype=cupy.float32)
        triangles = cupy.array([0, 1, 2], dtype=cupy.int32)

        filepath = tmp_path / "cupy_mesh.stl"
        write_stl(str(filepath), verts, triangles)

        assert filepath.exists()
        expected_size = 80 + 4 + 1 * 50
        assert filepath.stat().st_size == expected_size


class TestLoadObj:
    """Tests for load_obj function."""

    def test_load_simple_triangle(self, tmp_path):
        """Test loading OBJ with a single triangle."""
        obj_content = """# Simple triangle
v 0.0 0.0 0.0
v 1.0 0.0 0.0
v 0.5 1.0 0.0
f 1 2 3
"""
        filepath = tmp_path / "triangle.obj"
        filepath.write_text(obj_content)

        verts, indices = load_obj(filepath)

        # 3 vertices * 3 components = 9
        assert len(verts) == 9
        assert verts.dtype == np.float32
        # 1 triangle * 3 indices = 3
        assert len(indices) == 3
        assert indices.dtype == np.int32

        # Check vertex values
        np.testing.assert_array_almost_equal(
            verts, [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.5, 1.0, 0.0]
        )
        # Check indices (0-based)
        np.testing.assert_array_equal(indices, [0, 1, 2])

    def test_load_quad_triangulation(self, tmp_path):
        """Test that quads are automatically triangulated."""
        obj_content = """# Quad (unit square)
v 0.0 0.0 0.0
v 1.0 0.0 0.0
v 1.0 1.0 0.0
v 0.0 1.0 0.0
f 1 2 3 4
"""
        filepath = tmp_path / "quad.obj"
        filepath.write_text(obj_content)

        verts, indices = load_obj(filepath)

        # 4 vertices
        assert len(verts) == 12
        # Quad becomes 2 triangles: 6 indices
        assert len(indices) == 6

        # Fan triangulation: [0,1,2] and [0,2,3]
        np.testing.assert_array_equal(indices, [0, 1, 2, 0, 2, 3])

    def test_load_with_scale(self, tmp_path):
        """Test that scale parameter is applied to vertices."""
        obj_content = """v 1.0 2.0 3.0
v 4.0 5.0 6.0
v 7.0 8.0 9.0
f 1 2 3
"""
        filepath = tmp_path / "scaled.obj"
        filepath.write_text(obj_content)

        verts, _ = load_obj(filepath, scale=0.1)

        np.testing.assert_array_almost_equal(
            verts, [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        )

    def test_load_with_swap_yz(self, tmp_path):
        """Test that swap_yz swaps Y and Z coordinates."""
        obj_content = """v 1.0 2.0 3.0
v 4.0 5.0 6.0
v 7.0 8.0 9.0
f 1 2 3
"""
        filepath = tmp_path / "swapped.obj"
        filepath.write_text(obj_content)

        verts, _ = load_obj(filepath, swap_yz=True)

        # Y and Z should be swapped: (x, y, z) -> (x, z, y)
        np.testing.assert_array_almost_equal(
            verts, [1.0, 3.0, 2.0, 4.0, 6.0, 5.0, 7.0, 9.0, 8.0]
        )

    def test_load_face_with_texture_and_normal_indices(self, tmp_path):
        """Test parsing faces with v/vt/vn format."""
        obj_content = """v 0.0 0.0 0.0
v 1.0 0.0 0.0
v 0.5 1.0 0.0
vt 0.0 0.0
vt 1.0 0.0
vt 0.5 1.0
vn 0.0 0.0 1.0
f 1/1/1 2/2/1 3/3/1
"""
        filepath = tmp_path / "with_texnorm.obj"
        filepath.write_text(obj_content)

        verts, indices = load_obj(filepath)

        # Should correctly parse vertex indices, ignoring vt and vn
        assert len(verts) == 9
        np.testing.assert_array_equal(indices, [0, 1, 2])

    def test_load_face_with_vertex_texture_only(self, tmp_path):
        """Test parsing faces with v/vt format."""
        obj_content = """v 0.0 0.0 0.0
v 1.0 0.0 0.0
v 0.5 1.0 0.0
vt 0.0 0.0
vt 1.0 0.0
vt 0.5 1.0
f 1/1 2/2 3/3
"""
        filepath = tmp_path / "with_tex.obj"
        filepath.write_text(obj_content)

        verts, indices = load_obj(filepath)

        assert len(verts) == 9
        np.testing.assert_array_equal(indices, [0, 1, 2])

    def test_load_face_with_vertex_normal_only(self, tmp_path):
        """Test parsing faces with v//vn format."""
        obj_content = """v 0.0 0.0 0.0
v 1.0 0.0 0.0
v 0.5 1.0 0.0
vn 0.0 0.0 1.0
f 1//1 2//1 3//1
"""
        filepath = tmp_path / "with_norm.obj"
        filepath.write_text(obj_content)

        verts, indices = load_obj(filepath)

        assert len(verts) == 9
        np.testing.assert_array_equal(indices, [0, 1, 2])

    def test_load_negative_indices(self, tmp_path):
        """Test parsing faces with negative (relative) indices."""
        obj_content = """v 0.0 0.0 0.0
v 1.0 0.0 0.0
v 0.5 1.0 0.0
f -3 -2 -1
"""
        filepath = tmp_path / "negative.obj"
        filepath.write_text(obj_content)

        verts, indices = load_obj(filepath)

        # -3, -2, -1 should resolve to 0, 1, 2
        np.testing.assert_array_equal(indices, [0, 1, 2])

    def test_load_multiple_faces(self, tmp_path):
        """Test loading OBJ with multiple faces."""
        obj_content = """v 0.0 0.0 0.0
v 1.0 0.0 0.0
v 1.0 1.0 0.0
v 0.0 1.0 0.0
f 1 2 3
f 1 3 4
"""
        filepath = tmp_path / "two_triangles.obj"
        filepath.write_text(obj_content)

        verts, indices = load_obj(filepath)

        assert len(verts) == 12
        assert len(indices) == 6
        np.testing.assert_array_equal(indices, [0, 1, 2, 0, 2, 3])

    def test_load_with_comments_and_blank_lines(self, tmp_path):
        """Test that comments and blank lines are ignored."""
        obj_content = """# This is a comment
# Another comment

v 0.0 0.0 0.0

v 1.0 0.0 0.0
# Comment between vertices
v 0.5 1.0 0.0

f 1 2 3
# Final comment
"""
        filepath = tmp_path / "with_comments.obj"
        filepath.write_text(obj_content)

        verts, indices = load_obj(filepath)

        assert len(verts) == 9
        assert len(indices) == 3

    def test_load_file_not_found(self):
        """Test that FileNotFoundError is raised for missing files."""
        with pytest.raises(FileNotFoundError):
            load_obj("/nonexistent/path/to/file.obj")

    def test_load_no_vertices(self, tmp_path):
        """Test that ValueError is raised when file has no vertices."""
        obj_content = """# No vertices
f 1 2 3
"""
        filepath = tmp_path / "no_verts.obj"
        filepath.write_text(obj_content)

        with pytest.raises(ValueError, match="No vertices found"):
            load_obj(filepath)

    def test_load_no_faces(self, tmp_path):
        """Test that ValueError is raised when file has no faces."""
        obj_content = """v 0.0 0.0 0.0
v 1.0 0.0 0.0
v 0.5 1.0 0.0
"""
        filepath = tmp_path / "no_faces.obj"
        filepath.write_text(obj_content)

        with pytest.raises(ValueError, match="No faces found"):
            load_obj(filepath)

    def test_load_pentagon_triangulation(self, tmp_path):
        """Test that polygons with more than 4 vertices are triangulated."""
        obj_content = """# Pentagon
v 0.0 1.0 0.0
v 0.951 0.309 0.0
v 0.588 -0.809 0.0
v -0.588 -0.809 0.0
v -0.951 0.309 0.0
f 1 2 3 4 5
"""
        filepath = tmp_path / "pentagon.obj"
        filepath.write_text(obj_content)

        verts, indices = load_obj(filepath)

        # 5 vertices
        assert len(verts) == 15
        # Pentagon becomes 3 triangles: [0,1,2], [0,2,3], [0,3,4]
        assert len(indices) == 9
        np.testing.assert_array_equal(indices, [0, 1, 2, 0, 2, 3, 0, 3, 4])


class TestMakeTransform:
    """Tests for make_transform function."""

    def test_identity_transform(self):
        """Test that default parameters give identity transform."""
        t = make_transform()
        expected = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0]
        np.testing.assert_array_almost_equal(t, expected)

    def test_translation_only(self):
        """Test translation without rotation or scale."""
        t = make_transform(x=10, y=20, z=30)
        expected = [1, 0, 0, 10, 0, 1, 0, 20, 0, 0, 1, 30]
        np.testing.assert_array_almost_equal(t, expected)

    def test_scale_only(self):
        """Test uniform scaling."""
        t = make_transform(scale=2.0)
        expected = [2, 0, 0, 0, 0, 2, 0, 0, 0, 0, 2, 0]
        np.testing.assert_array_almost_equal(t, expected)

    def test_rotation_90_degrees(self):
        """Test 90 degree rotation around Z axis."""
        import math
        t = make_transform(rotation_z=math.pi / 2)
        # cos(90) = 0, sin(90) = 1
        # Rotation matrix: [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
        expected = [0, -1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0]
        np.testing.assert_array_almost_equal(t, expected, decimal=10)

    def test_combined_transform(self):
        """Test scale + rotation + translation."""
        import math
        t = make_transform(x=5, y=10, z=15, scale=2.0, rotation_z=math.pi / 2)
        # Scale 2 * Rotation 90deg
        expected = [0, -2, 0, 5, 2, 0, 0, 10, 0, 0, 2, 15]
        np.testing.assert_array_almost_equal(t, expected, decimal=10)

    def test_returns_list(self):
        """Test that result is a list of 12 floats."""
        t = make_transform(x=1, y=2, z=3)
        assert isinstance(t, list)
        assert len(t) == 12


class TestMakeTransformsOnTerrain:
    """Tests for make_transforms_on_terrain function."""

    def test_single_position(self):
        """Test placing one object on terrain."""
        terrain = np.array([
            [0, 0, 0],
            [0, 5, 0],
            [0, 0, 0],
        ], dtype=np.float32)

        positions = [(1, 1)]  # Center of terrain, elevation = 5
        transforms = make_transforms_on_terrain(positions, terrain)

        assert len(transforms) == 1
        t = transforms[0]
        assert len(t) == 12
        # Check translation components (indices 3, 7, 11)
        assert t[3] == 1.0  # x
        assert t[7] == 1.0  # y
        assert t[11] == 5.0  # z (sampled from terrain)

    def test_multiple_positions(self):
        """Test placing multiple objects on terrain."""
        terrain = np.array([
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9],
        ], dtype=np.float32)

        positions = [(0, 0), (1, 1), (2, 2)]
        transforms = make_transforms_on_terrain(positions, terrain)

        assert len(transforms) == 3
        # Check z values match terrain
        assert transforms[0][11] == 1.0  # terrain[0, 0]
        assert transforms[1][11] == 5.0  # terrain[1, 1]
        assert transforms[2][11] == 9.0  # terrain[2, 2]

    def test_with_scale(self):
        """Test scale parameter is applied."""
        terrain = np.zeros((3, 3), dtype=np.float32)
        positions = [(1, 1)]
        transforms = make_transforms_on_terrain(positions, terrain, scale=0.5)

        t = transforms[0]
        # Scale should appear in diagonal elements
        assert t[0] == 0.5  # Xx
        assert t[5] == 0.5  # Yy
        assert t[10] == 0.5  # Zz

    def test_with_scalar_rotation(self):
        """Test single rotation value applied to all."""
        import math
        terrain = np.zeros((3, 3), dtype=np.float32)
        positions = [(0, 0), (1, 1)]
        transforms = make_transforms_on_terrain(positions, terrain,
                                                 rotation_z=math.pi / 2)

        # Both should have same rotation
        for t in transforms:
            np.testing.assert_almost_equal(t[0], 0.0)  # cos(90) = 0
            np.testing.assert_almost_equal(t[4], 1.0)  # sin(90) = 1

    def test_with_array_rotation(self):
        """Test different rotation for each position."""
        import math
        terrain = np.zeros((3, 3), dtype=np.float32)
        positions = [(0, 0), (1, 1)]
        rotations = [0.0, math.pi / 2]
        transforms = make_transforms_on_terrain(positions, terrain,
                                                 rotation_z=rotations)

        # First: no rotation
        np.testing.assert_almost_equal(transforms[0][0], 1.0)  # cos(0)
        np.testing.assert_almost_equal(transforms[0][4], 0.0)  # sin(0)
        # Second: 90 degree rotation
        np.testing.assert_almost_equal(transforms[1][0], 0.0)  # cos(90)
        np.testing.assert_almost_equal(transforms[1][4], 1.0)  # sin(90)

    def test_position_clipping(self):
        """Test that out-of-bounds positions are clipped."""
        terrain = np.array([
            [1, 2],
            [3, 4],
        ], dtype=np.float32)

        # Position outside terrain bounds
        positions = [(10, 10)]  # Should be clipped to (1, 1)
        transforms = make_transforms_on_terrain(positions, terrain)

        # Should sample terrain[1, 1] = 4
        assert transforms[0][11] == 4.0

    def test_rotation_length_mismatch(self):
        """Test error when rotation array length doesn't match positions."""
        terrain = np.zeros((3, 3), dtype=np.float32)
        positions = [(0, 0), (1, 1), (2, 2)]
        rotations = [0.0, 0.0]  # Only 2, but 3 positions

        with pytest.raises(ValueError, match="rotation_z length"):
            make_transforms_on_terrain(positions, terrain, rotation_z=rotations)

    def test_numpy_array_positions(self):
        """Test that numpy array positions work."""
        terrain = np.array([
            [1, 2],
            [3, 4],
        ], dtype=np.float32)

        positions = np.array([[0, 0], [1, 1]])
        transforms = make_transforms_on_terrain(positions, terrain)

        assert len(transforms) == 2
        assert transforms[0][11] == 1.0  # terrain[0, 0]
        assert transforms[1][11] == 4.0  # terrain[1, 1]
