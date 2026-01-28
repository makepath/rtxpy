"""Tests for mesh utilities."""

import numpy as np
import pytest

from rtxpy import triangulate_terrain, write_stl
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
