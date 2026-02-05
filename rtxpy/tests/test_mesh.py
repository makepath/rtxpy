"""Tests for mesh utilities."""

import numpy as np
import pytest

from rtxpy import triangulate_terrain, voxelate_terrain, write_stl
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


class TestVoxelateTerrain:
    """Tests for voxelate_terrain function."""

    def _alloc(self, H, W):
        """Allocate voxelation buffers for H x W terrain."""
        num_verts = H * W * 8
        num_tris = H * W * 12
        verts = np.zeros(num_verts * 3, dtype=np.float32)
        triangles = np.zeros(num_tris * 3, dtype=np.int32)
        return verts, triangles

    def _get_cell_verts(self, verts, cell_idx):
        """Extract 8 vertices (as 8x3 array) for a given cell index."""
        vo = cell_idx * 24
        return verts[vo:vo + 24].reshape(8, 3)

    def test_simple_terrain_cpu(self):
        """Test voxelation of a 2x2 terrain — verify vertex positions."""
        H, W = 2, 2
        terrain = np.array([
            [1.0, 2.0],
            [3.0, 4.0],
        ], dtype=np.float32)

        verts, triangles = self._alloc(H, W)
        result = voxelate_terrain(verts, triangles, terrain, base_elevation=0.0)
        assert result == 0

        # Check cell (0,0): elevation=1.0, at position (w=0,h=0)
        cv = self._get_cell_verts(verts, 0)
        # Bottom verts z=0
        assert cv[0, 2] == 0.0
        assert cv[1, 2] == 0.0
        assert cv[2, 2] == 0.0
        assert cv[3, 2] == 0.0
        # Top verts z=1.0
        assert cv[4, 2] == 1.0
        assert cv[5, 2] == 1.0
        assert cv[6, 2] == 1.0
        assert cv[7, 2] == 1.0
        # Check x,y of bottom-left and top-right
        np.testing.assert_array_equal(cv[0, :2], [0, 0])
        np.testing.assert_array_equal(cv[6, :2], [1, 1])

        # Check cell (0,1): elevation=2.0, at position (w=1,h=0)
        cv1 = self._get_cell_verts(verts, 1)
        assert cv1[4, 2] == 2.0  # top z
        np.testing.assert_array_equal(cv1[0, :2], [1, 0])  # starts at w=1

        # Check cell (1,0): elevation=3.0, at position (w=0,h=1)
        cv2 = self._get_cell_verts(verts, 2)
        assert cv2[4, 2] == 3.0
        np.testing.assert_array_equal(cv2[0, :2], [0, 1])

        # Check buffer sizes
        assert len(verts) == H * W * 8 * 3
        assert len(triangles) == H * W * 12 * 3

    def test_terrain_with_scale(self):
        """Test that scale affects z values."""
        H, W = 2, 2
        terrain = np.array([
            [1.0, 1.0],
            [1.0, 1.0],
        ], dtype=np.float32)

        verts, triangles = self._alloc(H, W)
        voxelate_terrain(verts, triangles, terrain, scale=10.0, base_elevation=0.0)

        # Top z should be 1.0 * 10.0 = 10.0 for all cells
        for cell in range(H * W):
            cv = self._get_cell_verts(verts, cell)
            assert cv[4, 2] == 10.0

    def test_base_elevation(self):
        """Test that base_elevation controls column bottom."""
        H, W = 2, 2
        terrain = np.array([
            [5.0, 5.0],
            [5.0, 5.0],
        ], dtype=np.float32)

        verts, triangles = self._alloc(H, W)
        voxelate_terrain(verts, triangles, terrain, base_elevation=2.0)

        for cell in range(H * W):
            cv = self._get_cell_verts(verts, cell)
            # Bottom at base_elevation
            assert cv[0, 2] == 2.0
            assert cv[1, 2] == 2.0
            # Top at elevation
            assert cv[4, 2] == 5.0

    def test_triangle_winding_produces_outward_normals(self):
        """Cross product check on top face — normal should point +Z."""
        H, W = 1, 1
        terrain = np.array([[5.0]], dtype=np.float32)

        verts, triangles = self._alloc(H, W)
        voxelate_terrain(verts, triangles, terrain, base_elevation=0.0)

        # Top face is first two triangles: indices 0-2 and 3-5
        v = verts.reshape(-1, 3)
        i0, i1, i2 = triangles[0], triangles[1], triangles[2]
        p0, p1, p2 = v[i0], v[i1], v[i2]
        normal = np.cross(p1 - p0, p2 - p0)
        # Top face normal should point in +Z direction
        assert normal[2] > 0

    def test_flat_terrain(self):
        """Test voxelation of 3x3 uniform terrain."""
        H, W = 3, 3
        terrain = np.full((H, W), 10.0, dtype=np.float32)

        verts, triangles = self._alloc(H, W)
        result = voxelate_terrain(verts, triangles, terrain, base_elevation=0.0)
        assert result == 0

        # All top verts at z=10, all bottom at z=0
        for cell in range(H * W):
            cv = self._get_cell_verts(verts, cell)
            for i in range(4):
                assert cv[i, 2] == 0.0
                assert cv[i + 4, 2] == 10.0

    @pytest.mark.skipif(not has_cupy, reason="cupy not available")
    def test_simple_terrain_gpu(self):
        """Test voxelation on GPU with cupy arrays."""
        import cupy

        H, W = 2, 2
        terrain = cupy.array([
            [1.0, 2.0],
            [3.0, 4.0],
        ], dtype=cupy.float32)

        num_verts = H * W * 8
        num_tris = H * W * 12
        verts = cupy.zeros(num_verts * 3, dtype=cupy.float32)
        triangles = cupy.zeros(num_tris * 3, dtype=cupy.int32)

        result = voxelate_terrain(verts, triangles, terrain, base_elevation=0.0)
        assert result == 0

        verts_np = cupy.asnumpy(verts).reshape(-1, 3)
        # Cell 0: top z = 1.0
        assert verts_np[4, 2] == 1.0
        # Cell 3 (h=1,w=1): top z = 4.0
        assert verts_np[3 * 8 + 4, 2] == 4.0

    def test_write_voxelated_terrain_to_stl(self, tmp_path):
        """Test STL export of voxelated terrain."""
        H, W = 2, 2
        terrain = np.array([
            [1.0, 2.0],
            [3.0, 4.0],
        ], dtype=np.float32)

        verts, triangles = self._alloc(H, W)
        voxelate_terrain(verts, triangles, terrain, base_elevation=0.0)

        filepath = tmp_path / "voxelated.stl"
        write_stl(str(filepath), verts, triangles)

        assert filepath.exists()
        num_tris = H * W * 12
        expected_size = 80 + 4 + num_tris * 50
        assert filepath.stat().st_size == expected_size
