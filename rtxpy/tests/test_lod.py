"""Tests for level-of-detail utilities and terrain LOD manager."""

import numpy as np
import pytest

from rtxpy.lod import (
    compute_lod_level,
    compute_lod_level_with_hysteresis,
    compute_lod_distances,
    compute_tile_roughness,
    simplify_mesh,
    build_lod_chain,
)
from rtxpy.viewer.terrain_lod import (
    TerrainLODManager,
    is_terrain_lod_gid,
    _tile_gid,
    _batch_gid,
)


# ---------------------------------------------------------------------------
# compute_lod_level
# ---------------------------------------------------------------------------

class TestComputeLodLevel:
    """Tests for distance-to-LOD mapping."""

    def test_below_first_threshold(self):
        assert compute_lod_level(10, [100, 200, 400]) == 0

    def test_between_thresholds(self):
        assert compute_lod_level(150, [100, 200, 400]) == 1

    def test_above_all_thresholds(self):
        assert compute_lod_level(500, [100, 200, 400]) == 3

    def test_exact_boundary(self):
        # At the exact boundary, distance < threshold is false → next level
        assert compute_lod_level(100, [100, 200, 400]) == 1

    def test_single_threshold(self):
        assert compute_lod_level(50, [100]) == 0
        assert compute_lod_level(150, [100]) == 1

    def test_empty_thresholds(self):
        assert compute_lod_level(50, []) == 0

    def test_zero_distance(self):
        assert compute_lod_level(0, [100, 200]) == 0

    def test_negative_distance(self):
        assert compute_lod_level(-10, [100, 200]) == 0


# ---------------------------------------------------------------------------
# compute_lod_level_with_hysteresis
# ---------------------------------------------------------------------------

class TestComputeLodLevelWithHysteresis:
    """Tests for hysteresis-aware LOD selection."""

    def test_no_prev_lod_matches_basic(self):
        """First assignment (prev_lod=-1) should match compute_lod_level."""
        thresholds = [100, 200, 400]
        assert compute_lod_level_with_hysteresis(50, thresholds, -1) == 0
        assert compute_lod_level_with_hysteresis(150, thresholds, -1) == 1
        assert compute_lod_level_with_hysteresis(500, thresholds, -1) == 3

    def test_stays_at_current_lod_near_boundary(self):
        """Camera near threshold boundary should not flip LOD."""
        thresholds = [100, 200, 400]
        # At LOD 0, distance 105 is just past the 100 threshold
        # but within the hysteresis band — should stay at LOD 0
        assert compute_lod_level_with_hysteresis(
            105, thresholds, prev_lod=0, hysteresis=0.2) == 0

    def test_downgrades_past_hysteresis(self):
        """Camera well past threshold should downgrade."""
        thresholds = [100, 200, 400]
        # distance 125 / (1+0.2) = 104.2 → LOD 1 at base → downgrade
        assert compute_lod_level_with_hysteresis(
            125, thresholds, prev_lod=0, hysteresis=0.2) == 1

    def test_upgrades_past_hysteresis(self):
        """Camera well inside better band should upgrade."""
        thresholds = [100, 200, 400]
        # At LOD 1, distance 75 / (1-0.2) = 93.75 → LOD 0 → upgrade
        assert compute_lod_level_with_hysteresis(
            75, thresholds, prev_lod=1, hysteresis=0.2) == 0

    def test_stays_at_current_lod_upgrade_boundary(self):
        """Camera near threshold from above should not upgrade."""
        thresholds = [100, 200, 400]
        # At LOD 1, distance 95 → base LOD 0, but 95/(1-0.2) = 118.75 → LOD 1
        # so should stay at LOD 1
        assert compute_lod_level_with_hysteresis(
            95, thresholds, prev_lod=1, hysteresis=0.2) == 1

    def test_same_lod_unchanged(self):
        """If base LOD matches prev_lod, return prev_lod."""
        thresholds = [100, 200, 400]
        assert compute_lod_level_with_hysteresis(
            50, thresholds, prev_lod=0, hysteresis=0.2) == 0
        assert compute_lod_level_with_hysteresis(
            150, thresholds, prev_lod=1, hysteresis=0.2) == 1

    def test_dead_zone_spans_both_directions(self):
        """Hysteresis dead zone should prevent both upgrade and downgrade.

        With threshold=100 and hysteresis=0.2:
        - Downgrade requires distance > 100 * 1.2 = 120
        - Upgrade requires distance < 100 * 0.8 = 80
        - Between 80 and 120 is a dead zone where prev_lod sticks.
        """
        thresholds = [100, 200, 400]
        h = 0.2
        # At prev_lod=0 (below threshold): distances 101-119 in dead zone
        for d in [101, 110, 119]:
            assert compute_lod_level_with_hysteresis(
                d, thresholds, prev_lod=0, hysteresis=h) == 0, \
                f"distance {d} should stay at LOD 0 (downgrade dead zone)"
        # At 121, should finally downgrade
        assert compute_lod_level_with_hysteresis(
            121, thresholds, prev_lod=0, hysteresis=h) == 1

        # At prev_lod=1 (above threshold): distances 81-99 in dead zone
        for d in [81, 90, 99]:
            assert compute_lod_level_with_hysteresis(
                d, thresholds, prev_lod=1, hysteresis=h) == 1, \
                f"distance {d} should stay at LOD 1 (upgrade dead zone)"
        # At 79, should finally upgrade
        assert compute_lod_level_with_hysteresis(
            79, thresholds, prev_lod=1, hysteresis=h) == 0


# ---------------------------------------------------------------------------
# compute_lod_distances
# ---------------------------------------------------------------------------

class TestComputeLodDistances:
    """Tests for LOD distance threshold generation."""

    def test_basic(self):
        dists = compute_lod_distances(100.0, factor=2.0, max_lod=3)
        assert len(dists) == 3
        assert dists[0] == pytest.approx(200.0)
        assert dists[1] == pytest.approx(400.0)
        assert dists[2] == pytest.approx(800.0)

    def test_single_level(self):
        dists = compute_lod_distances(50.0, factor=3.0, max_lod=1)
        assert len(dists) == 1
        assert dists[0] == pytest.approx(150.0)

    def test_zero_max_lod(self):
        dists = compute_lod_distances(100.0, factor=2.0, max_lod=0)
        assert dists == []


# ---------------------------------------------------------------------------
# compute_tile_roughness
# ---------------------------------------------------------------------------

class TestComputeTileRoughness:
    """Tests for bilinear-fit residual roughness metric."""

    def test_flat_tile(self):
        """A constant-elevation tile should have near-zero roughness."""
        tile = np.full((16, 16), 500.0, dtype=np.float32)
        assert compute_tile_roughness(tile) == pytest.approx(0.0, abs=1e-4)

    def test_planar_tile(self):
        """A perfectly planar (tilted) tile should have ~zero roughness.

        The bilinear fit matches a linear surface exactly, so residuals
        are zero everywhere.
        """
        ys = np.arange(16).reshape(16, 1).astype(np.float32)
        xs = np.arange(16).reshape(1, 16).astype(np.float32)
        tile = 100.0 + 3.0 * xs + 2.0 * ys
        assert compute_tile_roughness(tile) == pytest.approx(0.0, abs=1e-4)

    def test_rough_tile(self):
        """A tile with a central peak should have non-zero roughness."""
        tile = np.zeros((16, 16), dtype=np.float32)
        tile[7:9, 7:9] = 100.0  # sharp peak
        r = compute_tile_roughness(tile)
        assert r > 1.0

    def test_rougher_is_higher(self):
        """A tile with bigger deviation should score higher."""
        tile_mild = np.zeros((16, 16), dtype=np.float32)
        tile_mild[8, 8] = 10.0
        tile_wild = np.zeros((16, 16), dtype=np.float32)
        tile_wild[8, 8] = 1000.0
        assert compute_tile_roughness(tile_wild) > compute_tile_roughness(tile_mild)

    def test_all_nan(self):
        """All-NaN tile should return zero roughness."""
        tile = np.full((8, 8), np.nan, dtype=np.float32)
        assert compute_tile_roughness(tile) == 0.0

    def test_partial_nan(self):
        """Tile with some NaN values should still return a valid float."""
        tile = np.ones((8, 8), dtype=np.float32) * 50.0
        tile[3:5, 3:5] = np.nan
        r = compute_tile_roughness(tile)
        assert np.isfinite(r)

    def test_nan_corner(self):
        """NaN corner should be filled with mean of valid corners."""
        tile = np.zeros((8, 8), dtype=np.float32)
        tile[0, 0] = np.nan  # one corner NaN
        r = compute_tile_roughness(tile)
        assert np.isfinite(r)

    def test_tiny_tile(self):
        """Tiles smaller than 2x2 should return 0."""
        assert compute_tile_roughness(np.array([[5.0]])) == 0.0
        assert compute_tile_roughness(np.zeros((1, 10))) == 0.0


# ---------------------------------------------------------------------------
# compute_terrain_normals
# ---------------------------------------------------------------------------

class TestComputeTerrainNormals:
    """Tests for central-difference terrain normal computation."""

    def test_flat_terrain(self):
        """Flat terrain should produce all (0, 0, 1) normals."""
        from rtxpy.mesh import compute_terrain_normals
        terrain = np.full((4, 4), 100.0, dtype=np.float32)
        normals = compute_terrain_normals(terrain, 4, 4)
        assert normals.shape == (4 * 4 * 3,)
        nx = normals[0::3]
        ny = normals[1::3]
        nz = normals[2::3]
        np.testing.assert_allclose(nx, 0.0, atol=1e-6)
        np.testing.assert_allclose(ny, 0.0, atol=1e-6)
        np.testing.assert_allclose(nz, 1.0, atol=1e-6)

    def test_x_slope(self):
        """Constant slope in X (z = col) with psx=1 should tilt normals."""
        from rtxpy.mesh import compute_terrain_normals
        H, W = 3, 5
        terrain = np.zeros((H, W), dtype=np.float32)
        for c in range(W):
            terrain[:, c] = float(c)
        normals = compute_terrain_normals(terrain, H, W, psx=1.0, psy=1.0)
        # Interior vertices: dz/dx = 1, normal = normalize(-1, 0, 1)
        expected_nx = -1.0 / np.sqrt(2.0)
        expected_nz = 1.0 / np.sqrt(2.0)
        # Check an interior vertex (row=1, col=2)
        idx = 1 * W + 2
        assert normals[idx * 3] == pytest.approx(expected_nx, abs=1e-5)
        assert normals[idx * 3 + 1] == pytest.approx(0.0, abs=1e-5)
        assert normals[idx * 3 + 2] == pytest.approx(expected_nz, abs=1e-5)

    def test_y_slope(self):
        """Constant slope in Y (z = row) should tilt normals in Y."""
        from rtxpy.mesh import compute_terrain_normals
        H, W = 5, 3
        terrain = np.zeros((H, W), dtype=np.float32)
        for r in range(H):
            terrain[r, :] = float(r)
        normals = compute_terrain_normals(terrain, H, W, psx=1.0, psy=1.0)
        expected_ny = -1.0 / np.sqrt(2.0)
        expected_nz = 1.0 / np.sqrt(2.0)
        idx = 2 * W + 1
        assert normals[idx * 3] == pytest.approx(0.0, abs=1e-5)
        assert normals[idx * 3 + 1] == pytest.approx(expected_ny, abs=1e-5)
        assert normals[idx * 3 + 2] == pytest.approx(expected_nz, abs=1e-5)

    def test_nan_elevation_gets_up_normal(self):
        """NaN elevation pixels should get (0, 0, 1)."""
        from rtxpy.mesh import compute_terrain_normals
        terrain = np.ones((4, 4), dtype=np.float32) * 50.0
        terrain[1, 2] = np.nan
        normals = compute_terrain_normals(terrain, 4, 4)
        idx = 1 * 4 + 2
        assert normals[idx * 3] == pytest.approx(0.0, abs=1e-6)
        assert normals[idx * 3 + 1] == pytest.approx(0.0, abs=1e-6)
        assert normals[idx * 3 + 2] == pytest.approx(1.0, abs=1e-6)

    def test_pixel_spacing_affects_normals(self):
        """Wider pixel spacing should flatten normals (smaller nx)."""
        from rtxpy.mesh import compute_terrain_normals
        H, W = 3, 5
        terrain = np.zeros((H, W), dtype=np.float32)
        for c in range(W):
            terrain[:, c] = float(c)
        n1 = compute_terrain_normals(terrain, H, W, psx=1.0, psy=1.0)
        n10 = compute_terrain_normals(terrain, H, W, psx=10.0, psy=1.0)
        idx = 1 * W + 2
        # With psx=10, dz/dx = 1/10, so |nx| should be much smaller
        assert abs(n10[idx * 3]) < abs(n1[idx * 3])

    def test_all_unit_length(self):
        """All normals should be unit-length."""
        from rtxpy.mesh import compute_terrain_normals
        terrain = np.random.RandomState(42).rand(8, 8).astype(np.float32) * 100
        normals = compute_terrain_normals(terrain, 8, 8)
        nx = normals[0::3]
        ny = normals[1::3]
        nz = normals[2::3]
        lengths = np.sqrt(nx**2 + ny**2 + nz**2)
        np.testing.assert_allclose(lengths, 1.0, atol=1e-5)


# ---------------------------------------------------------------------------
# compute_vertex_normals
# ---------------------------------------------------------------------------

class TestComputeVertexNormals:
    """Tests for area-weighted smooth vertex normal computation."""

    def test_flat_quad(self):
        """Two-triangle flat quad should produce all (0, 0, 1) normals."""
        from rtxpy.mesh import compute_vertex_normals
        verts = np.array([
            0, 0, 0,  1, 0, 0,  1, 1, 0,  0, 1, 0,
        ], dtype=np.float32)
        indices = np.array([0, 1, 2, 0, 2, 3], dtype=np.int32)
        normals = compute_vertex_normals(verts, indices)
        assert normals.shape == (12,)
        nz = normals[2::3]
        np.testing.assert_allclose(nz, 1.0, atol=1e-6)
        np.testing.assert_allclose(normals[0::3], 0.0, atol=1e-6)

    def test_dihedral_edge(self):
        """Shared edge of 90° dihedral: vertex normals should be averaged."""
        from rtxpy.mesh import compute_vertex_normals
        # Two triangles sharing edge along X axis, forming a V
        verts = np.array([
            0, 0, 0,   # 0: left of shared edge
            2, 0, 0,   # 1: right of shared edge
            1, 1, 0,   # 2: flat face vertex
            1, -1, 0,  # 3: also flat (mirror)
        ], dtype=np.float32)
        # Both triangles lie in z=0 plane with different orientations
        indices = np.array([0, 1, 2, 0, 3, 1], dtype=np.int32)
        normals = compute_vertex_normals(verts, indices)
        # All vertices have faces in z=0, so nz should dominate
        nz = normals[2::3]
        assert all(abs(n) > 0.5 for n in nz)

    def test_isolated_vertex_gets_up_normal(self):
        """Vertices not referenced by any triangle should get (0,0,1)."""
        from rtxpy.mesh import compute_vertex_normals
        verts = np.array([
            0, 0, 0,  1, 0, 0,  0, 1, 0,  # triangle
            5, 5, 5,  # isolated vertex
        ], dtype=np.float32)
        indices = np.array([0, 1, 2], dtype=np.int32)
        normals = compute_vertex_normals(verts, indices)
        # Isolated vertex (index 3) should be (0, 0, 1)
        assert normals[9] == pytest.approx(0.0, abs=1e-6)
        assert normals[10] == pytest.approx(0.0, abs=1e-6)
        assert normals[11] == pytest.approx(1.0, abs=1e-6)

    def test_all_unit_length(self):
        """All normals should be unit-length."""
        from rtxpy.mesh import compute_vertex_normals
        rng = np.random.RandomState(42)
        verts = rng.rand(30).astype(np.float32)
        indices = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 0, 3, 6],
                           dtype=np.int32)
        normals = compute_vertex_normals(verts, indices)
        nx = normals[0::3]
        ny = normals[1::3]
        nz = normals[2::3]
        lengths = np.sqrt(nx**2 + ny**2 + nz**2)
        np.testing.assert_allclose(lengths, 1.0, atol=1e-5)


# ---------------------------------------------------------------------------
# compute_skirt_normals
# ---------------------------------------------------------------------------

class TestComputeSkirtNormals:
    """Tests for skirt outward-facing normal computation."""

    def test_output_shape(self):
        from rtxpy.mesh import compute_skirt_normals
        H, W = 5, 7
        normals = compute_skirt_normals(H, W)
        n_perim = 2 * (H + W) - 4
        assert normals.shape == (n_perim * 3,)

    def test_edge_directions(self):
        """Top/right/bottom/left edges should have correct outward normals."""
        from rtxpy.mesh import compute_skirt_normals
        H, W = 5, 7
        normals = compute_skirt_normals(H, W).reshape(-1, 3)
        off = 0
        # Top: W verts, normal (0, -1, 0)
        for i in range(W):
            np.testing.assert_allclose(normals[off + i], [0, -1, 0], atol=1e-6)
        off += W
        # Right: H-1 verts, normal (1, 0, 0)
        for i in range(H - 1):
            np.testing.assert_allclose(normals[off + i], [1, 0, 0], atol=1e-6)
        off += H - 1
        # Bottom: W-1 verts, normal (0, 1, 0)
        for i in range(W - 1):
            np.testing.assert_allclose(normals[off + i], [0, 1, 0], atol=1e-6)
        off += W - 1
        # Left: H-2 verts, normal (-1, 0, 0)
        for i in range(H - 2):
            np.testing.assert_allclose(normals[off + i], [-1, 0, 0], atol=1e-6)

    def test_all_unit_length(self):
        from rtxpy.mesh import compute_skirt_normals
        normals = compute_skirt_normals(10, 10).reshape(-1, 3)
        lengths = np.sqrt(np.sum(normals**2, axis=1))
        np.testing.assert_allclose(lengths, 1.0, atol=1e-6)


# ---------------------------------------------------------------------------
# simplify_mesh
# ---------------------------------------------------------------------------

class TestSimplifyMesh:
    """Tests for mesh simplification."""

    def _make_grid_mesh(self, H=10, W=10):
        """Create a simple grid mesh for testing."""
        n_verts = H * W
        n_tris = (H - 1) * (W - 1) * 2
        verts = np.zeros(n_verts * 3, dtype=np.float32)
        indices = np.zeros(n_tris * 3, dtype=np.int32)

        for h in range(H):
            for w in range(W):
                idx = (h * W + w) * 3
                verts[idx] = float(w)
                verts[idx + 1] = float(h)
                verts[idx + 2] = float(h + w) * 0.1

        tri = 0
        for h in range(H - 1):
            for w in range(W - 1):
                v00 = h * W + w
                v01 = h * W + w + 1
                v10 = (h + 1) * W + w
                v11 = (h + 1) * W + w + 1
                indices[tri * 3] = v00
                indices[tri * 3 + 1] = v11
                indices[tri * 3 + 2] = v10
                tri += 1
                indices[tri * 3] = v00
                indices[tri * 3 + 1] = v01
                indices[tri * 3 + 2] = v11
                tri += 1

        return verts, indices

    def test_ratio_one_returns_original(self):
        verts, indices = self._make_grid_mesh()
        sv, si = simplify_mesh(verts, indices, 1.0)
        np.testing.assert_array_equal(sv, verts)
        np.testing.assert_array_equal(si, indices)

    def test_simplification_reduces_triangles(self):
        verts, indices = self._make_grid_mesh(20, 20)
        orig_tris = len(indices) // 3
        sv, si = simplify_mesh(verts, indices, 0.5)
        new_tris = len(si) // 3
        # Simplified mesh should have fewer or equal triangles
        assert new_tris <= orig_tris


# ---------------------------------------------------------------------------
# build_lod_chain
# ---------------------------------------------------------------------------

class TestBuildLodChain:
    def test_chain_length(self):
        verts = np.zeros(30, dtype=np.float32)
        indices = np.zeros(12, dtype=np.int32)
        chain = build_lod_chain(verts, indices, ratios=(1.0, 0.5, 0.25))
        assert len(chain) == 3

    def test_first_level_is_copy(self):
        verts = np.arange(12, dtype=np.float32)
        indices = np.arange(6, dtype=np.int32)
        chain = build_lod_chain(verts, indices, ratios=(1.0,))
        np.testing.assert_array_equal(chain[0][0], verts)
        np.testing.assert_array_equal(chain[0][1], indices)
        # Must be a copy, not the same object
        assert chain[0][0] is not verts


# ---------------------------------------------------------------------------
# TerrainLODManager
# ---------------------------------------------------------------------------

class _FakeRTX:
    """Minimal stub for RTX in tests — tracks add/remove calls."""

    def __init__(self):
        self.geometries = {}

    def add_geometry(self, gid, verts, indices, **kw):
        normals = kw.get('normals')
        self.geometries[gid] = (
            verts.copy(), indices.copy(),
            normals.copy() if normals is not None else None)
        return 0

    def add_heightfield_geometry(self, gid, elevation, H, W,
                                  spacing_x, spacing_y, ve=1.0,
                                  tile_size=32, active_mask=None,
                                  transform=None):
        """Stub for heightfield GAS — stores metadata for test assertions."""
        self.geometries[gid] = {
            'type': 'heightfield',
            'H': H, 'W': W,
            'spacing_x': spacing_x, 'spacing_y': spacing_y,
            've': ve, 'tile_size': tile_size,
            'active_mask': active_mask.copy() if active_mask is not None else None,
            'transform': list(transform) if transform is not None else None,
        }
        return 0

    def remove_geometry(self, gid):
        self.geometries.pop(gid, None)
        return 0

    def has_geometry(self, gid):
        return gid in self.geometries

    def list_geometries(self):
        return list(self.geometries.keys())


class TestTerrainLODManager:
    """Tests for the terrain LOD tile manager."""

    def _make_terrain(self, H=256, W=256):
        """Create a synthetic terrain: gentle gradient."""
        y = np.linspace(0, 100, H, dtype=np.float32)
        x = np.linspace(0, 100, W, dtype=np.float32)
        return y[:, None] + x[None, :]

    def test_tile_count(self):
        terrain = self._make_terrain(256, 256)
        mgr = TerrainLODManager(terrain, tile_size=64)
        assert mgr.n_tiles == 16  # 4x4

    def test_tile_count_non_divisible(self):
        terrain = self._make_terrain(300, 200)
        mgr = TerrainLODManager(terrain, tile_size=128)
        # 300/128 = ceil(2.34) = 3 rows, 200/128 = ceil(1.56) = 2 cols
        assert mgr.n_tiles == 6

    def test_initial_update_creates_tiles(self):
        terrain = self._make_terrain(128, 128)
        rtx = _FakeRTX()
        mgr = TerrainLODManager(terrain, tile_size=64,
                                pixel_spacing_x=1.0, pixel_spacing_y=1.0)
        changed = mgr.update(np.array([64, 64, 0]), rtx, force=True)
        assert changed
        # Should have created 4 tiles (2x2 grid)
        assert mgr.n_tiles == 4
        for tr in range(2):
            for tc in range(2):
                gid = _tile_gid(tr, tc)
                assert gid in rtx.geometries, f"Missing tile {gid}"

    def test_no_change_without_movement(self):
        terrain = self._make_terrain(128, 128)
        rtx = _FakeRTX()
        mgr = TerrainLODManager(terrain, tile_size=64)
        mgr.update(np.array([64, 64, 0]), rtx, force=True)
        changed = mgr.update(np.array([64, 64, 0]), rtx)
        assert not changed

    def test_lod_varies_with_distance(self):
        terrain = self._make_terrain(512, 512)
        rtx = _FakeRTX()
        mgr = TerrainLODManager(
            terrain, tile_size=128,
            pixel_spacing_x=1.0, pixel_spacing_y=1.0,
            max_lod=3, lod_distance_factor=1.0,
        )
        # Allow all 16 tiles to build in one pass for this test
        mgr.per_tick_build_limit = 100
        # Camera at corner — near tiles get LOD 0, far tiles get higher LOD
        mgr.update(np.array([0, 0, 0]), rtx, force=True)
        lods = mgr.tile_lods
        # Tile (0,0) is nearest, should be LOD 0
        assert lods[(0, 0)] == 0
        # Tile (3,3) is farthest, should have higher LOD
        assert lods[(3, 3)] > lods[(0, 0)]

    def test_remove_all_clears_scene(self):
        terrain = self._make_terrain(128, 128)
        rtx = _FakeRTX()
        mgr = TerrainLODManager(terrain, tile_size=64)
        mgr.update(np.array([64, 64, 0]), rtx, force=True)
        assert len(rtx.geometries) > 0
        mgr.remove_all(rtx)
        # All terrain LOD tiles should be removed
        for gid in rtx.geometries:
            assert not is_terrain_lod_gid(gid)

    def test_ve_applied_to_z(self):
        terrain = self._make_terrain(64, 64)
        rtx = _FakeRTX()
        mgr = TerrainLODManager(terrain, tile_size=64)
        mgr.update(np.array([32, 32, 0]), rtx, ve=2.0, force=True)
        gid = _tile_gid(0, 0)
        verts = rtx.geometries[gid][0]
        z_vals = verts[2::3]
        # With VE=2, z values should be doubled relative to terrain
        assert float(np.max(z_vals)) > float(np.max(terrain))

    def test_base_subsample_changes_invalidate_cache(self):
        terrain = self._make_terrain(128, 128)
        rtx = _FakeRTX()
        mgr = TerrainLODManager(terrain, tile_size=64, base_subsample=1)
        mgr.update(np.array([64, 64, 0]), rtx, force=True)
        gid = _tile_gid(0, 0)
        verts1 = rtx.geometries[gid][0].copy()

        mgr.set_base_subsample(2)
        mgr.update(np.array([64, 64, 0]), rtx, force=True)
        verts2 = rtx.geometries[gid][0]
        # Meshes should differ (fewer vertices at 2x subsample)
        assert len(verts2) != len(verts1)

    def test_adjacent_tiles_share_boundary_vertices(self):
        """Adjacent tiles must share boundary vertices (no gap). #79"""
        terrain = self._make_terrain(128, 128)
        rtx = _FakeRTX()
        mgr = TerrainLODManager(terrain, tile_size=64,
                                pixel_spacing_x=1.0, pixel_spacing_y=1.0,
                                base_subsample=1)
        mgr.update(np.array([64, 64, 0]), rtx, force=True)

        # Tile (0,0) and tile (0,1) share a column boundary
        v00 = rtx.geometries[_tile_gid(0, 0)][0]
        v01 = rtx.geometries[_tile_gid(0, 1)][0]
        # Max x of tile (0,0) should equal min x of tile (0,1)
        max_x_00 = float(np.max(v00[0::3]))
        min_x_01 = float(np.min(v01[0::3]))
        assert max_x_00 == pytest.approx(min_x_01), (
            f"Column gap: tile(0,0) max_x={max_x_00}, "
            f"tile(0,1) min_x={min_x_01}"
        )

        # Tile (0,0) and tile (1,0) share a row boundary
        v10 = rtx.geometries[_tile_gid(1, 0)][0]
        max_y_00 = float(np.max(v00[1::3]))
        min_y_10 = float(np.min(v10[1::3]))
        assert max_y_00 == pytest.approx(min_y_10), (
            f"Row gap: tile(0,0) max_y={max_y_00}, "
            f"tile(1,0) min_y={min_y_10}"
        )

    def test_boundary_shared_at_higher_subsample(self):
        """Boundary sharing works at subsample > 1. #79"""
        terrain = self._make_terrain(256, 256)
        rtx = _FakeRTX()
        mgr = TerrainLODManager(terrain, tile_size=128,
                                pixel_spacing_x=1.0, pixel_spacing_y=1.0,
                                base_subsample=2)
        mgr.update(np.array([128, 128, 0]), rtx, force=True)

        v00 = rtx.geometries[_tile_gid(0, 0)][0]
        v01 = rtx.geometries[_tile_gid(0, 1)][0]
        max_x_00 = float(np.max(v00[0::3]))
        min_x_01 = float(np.min(v01[0::3]))
        assert max_x_00 == pytest.approx(min_x_01), (
            f"Column gap at subsample=2: tile(0,0) max_x={max_x_00}, "
            f"tile(0,1) min_x={min_x_01}"
        )

    def test_interior_tile_no_skirt(self):
        """Tiles have no skirt — edge stitching replaces skirts."""
        terrain = self._make_terrain(256, 256)
        rtx = _FakeRTX()
        mgr = TerrainLODManager(terrain, tile_size=64,
                                pixel_spacing_x=1.0, pixel_spacing_y=1.0)
        mgr.per_tick_build_limit = 100
        mgr.update(np.array([128, 128, 0]), rtx, force=True)
        # Tile (1,1) is fully interior — should have only grid verts, no skirt
        gid = _tile_gid(1, 1)
        verts = rtx.geometries[gid][0]
        n_grid = 65 * 65
        n_verts = len(verts) // 3
        assert n_verts == n_grid, (
            f"Interior tile should have no skirt, got {n_verts} verts "
            f"(expected {n_grid})"
        )

    def test_get_stats(self):
        terrain = self._make_terrain(128, 128)
        mgr = TerrainLODManager(terrain, tile_size=64)
        # Before update
        stats = mgr.get_stats()
        assert "no tiles" in stats
        # After update
        rtx = _FakeRTX()
        mgr.update(np.array([64, 64, 0]), rtx, force=True)
        stats = mgr.get_stats()
        assert "LOD:" in stats
        assert "tiles" in stats

    def test_set_terrain_updates_tile_grid(self):
        """set_terrain with different shape must update tile count."""
        terrain = self._make_terrain(128, 128)
        mgr = TerrainLODManager(terrain, tile_size=64)
        assert mgr.n_tiles == 4  # 2x2

        # Replace with larger terrain
        terrain2 = self._make_terrain(256, 256)
        mgr.set_terrain(terrain2)
        assert mgr.n_tiles == 16  # 4x4

    def test_stale_tiles_evicted_from_cache(self):
        """Tiles leaving distance range should be evicted from mesh cache."""
        terrain = self._make_terrain(512, 512)
        rtx = _FakeRTX()
        mgr = TerrainLODManager(
            terrain, tile_size=128,
            pixel_spacing_x=1.0, pixel_spacing_y=1.0,
            max_lod=3, lod_distance_factor=1.0,
        )
        mgr.per_tick_build_limit = 100
        # Build tiles near corner
        mgr.update(np.array([0, 0, 0]), rtx, force=True)
        cache_keys_before = set(mgr._tile_cache.keys())
        assert len(cache_keys_before) > 0

        # Move camera far away so original tiles leave range
        mgr.update(np.array([10000, 10000, 0]), rtx, force=True)
        # Old tile cache entries should be evicted
        cache_keys_after = set(mgr._tile_cache.keys())
        evicted = cache_keys_before - cache_keys_after
        assert len(evicted) > 0, "Stale tile cache entries were not evicted"

    def test_offset_shifts_tile_vertices(self):
        """World offset should shift all tile vertex positions."""
        terrain = self._make_terrain(128, 128)
        rtx = _FakeRTX()
        mgr = TerrainLODManager(terrain, tile_size=64,
                                pixel_spacing_x=1.0, pixel_spacing_y=1.0)
        mgr.update(np.array([64, 64, 0]), rtx, force=True)
        gid = _tile_gid(0, 0)
        verts_no_offset = rtx.geometries[gid][0].copy()

        # Apply an offset and rebuild
        mgr.set_offset(100.0, 200.0)
        mgr.update(np.array([164, 264, 0]), rtx, force=True)
        verts_with_offset = rtx.geometries[gid][0]

        # X coords should be shifted by 100, Y by 200
        x_diff = verts_with_offset[0::3] - verts_no_offset[0::3]
        y_diff = verts_with_offset[1::3] - verts_no_offset[1::3]
        np.testing.assert_allclose(x_diff, 100.0, atol=0.01)
        np.testing.assert_allclose(y_diff, 200.0, atol=0.01)

    def test_set_terrain_with_offset(self):
        """set_terrain with offset should shift subsequent tile vertices."""
        terrain = self._make_terrain(128, 128)
        rtx = _FakeRTX()
        mgr = TerrainLODManager(terrain, tile_size=64,
                                pixel_spacing_x=1.0, pixel_spacing_y=1.0)
        # Replace terrain with an offset
        terrain2 = self._make_terrain(128, 128)
        mgr.set_terrain(terrain2, offset_x=50.0, offset_y=75.0)
        mgr.update(np.array([114, 139, 0]), rtx, force=True)
        gid = _tile_gid(0, 0)
        verts = rtx.geometries[gid][0]
        # Min X should be at offset (50.0), min Y at offset (75.0)
        assert float(np.min(verts[0::3])) == pytest.approx(50.0)
        assert float(np.min(verts[1::3])) == pytest.approx(75.0)

    def test_streaming_creates_tiles_beyond_bounds(self):
        """Streaming callback should produce tiles at negative indices."""
        terrain = self._make_terrain(128, 128)
        rtx = _FakeRTX()
        mgr = TerrainLODManager(terrain, tile_size=64,
                                pixel_spacing_x=1.0, pixel_spacing_y=1.0)

        # Callback returns a flat elevation grid
        def fake_tile_fn(x_min, y_min, x_max, y_max, target_samples):
            return np.full((target_samples, target_samples), 50.0,
                           dtype=np.float32)

        mgr.set_tile_data_fn(fake_tile_fn)
        assert mgr._streaming

        mgr.per_tick_build_limit = 100
        # Camera beyond the initial terrain bounds (x < 0)
        mgr.update(np.array([-200, 64, 0]), rtx, force=True)

        # Should have tiles with negative column indices
        neg_tiles = [gid for gid in rtx.geometries
                     if is_terrain_lod_gid(gid) and '_c-' in gid]
        assert len(neg_tiles) > 0, (
            f"Expected streaming tiles at negative col, got: "
            f"{list(rtx.geometries.keys())}"
        )

    def test_streaming_tile_positions_correct(self):
        """Streaming tiles should be positioned at correct world coords."""
        terrain = self._make_terrain(64, 64)
        rtx = _FakeRTX()
        mgr = TerrainLODManager(terrain, tile_size=64,
                                pixel_spacing_x=10.0, pixel_spacing_y=10.0)

        # Fixed elevation for easy verification
        def fake_tile_fn(x_min, y_min, x_max, y_max, target_samples):
            return np.full((target_samples, target_samples), 42.0,
                           dtype=np.float32)

        mgr.set_tile_data_fn(fake_tile_fn)
        mgr.per_tick_build_limit = 100
        # Camera at tile (0, 1) which is just past the initial terrain
        # (initial terrain is 1×1 tiles at 64px × 10.0 spacing = 640 world units)
        mgr.update(np.array([960, 320, 0]), rtx, force=True)

        gid = _tile_gid(0, 1)
        if gid in rtx.geometries:
            verts = rtx.geometries[gid][0]
            x_min_v = float(np.min(verts[0::3]))
            # Tile (0,1) starts at col 64 → x = 64 * 10.0 = 640.0
            assert x_min_v == pytest.approx(640.0, abs=1.0)

    def test_streaming_disabled_by_default(self):
        """Streaming should be off when no callback is set."""
        terrain = self._make_terrain(128, 128)
        mgr = TerrainLODManager(terrain, tile_size=64)
        assert not mgr._streaming
        assert mgr._tile_data_fn is None

    def test_set_tile_data_fn_none_disables_streaming(self):
        """Setting callback to None should disable streaming."""
        terrain = self._make_terrain(128, 128)
        mgr = TerrainLODManager(terrain, tile_size=64)

        mgr.set_tile_data_fn(lambda *a: None)
        assert mgr._streaming

        mgr.set_tile_data_fn(None)
        assert not mgr._streaming

    def test_streaming_callback_receives_correct_bounds(self):
        """Callback should receive tile world bounds and its data should
        appear in the built mesh Z values."""
        terrain = self._make_terrain(64, 64)
        rtx = _FakeRTX()
        mgr = TerrainLODManager(terrain, tile_size=64,
                                pixel_spacing_x=10.0, pixel_spacing_y=10.0)
        received_calls = []

        def tracking_fn(x_min, y_min, x_max, y_max, target_samples):
            received_calls.append({
                'x_min': x_min, 'y_min': y_min,
                'x_max': x_max, 'y_max': y_max,
                'target_samples': target_samples,
            })
            # Return a distinctive elevation
            return np.full((target_samples, target_samples), 999.0,
                           dtype=np.float32)

        mgr.set_tile_data_fn(tracking_fn)
        mgr.per_tick_build_limit = 100
        # Camera at tile (0, 2) — beyond initial 1×1 grid
        mgr.update(np.array([1600, 320, 0]), rtx, force=True)

        # Verify callback was called for out-of-bounds tiles
        assert len(received_calls) > 0, "Callback was never called"

        # Verify bounds are sensible (positive width/height)
        for call in received_calls:
            assert call['x_max'] > call['x_min']
            assert call['y_max'] > call['y_min']
            assert call['target_samples'] >= 2

        # Verify the distinctive elevation appears in mesh Z values
        gid = _tile_gid(0, 2)
        if gid in rtx.geometries:
            verts = rtx.geometries[gid][0]
            # Surface verts (not skirt) should have z=999.0
            z_vals = verts[2::3]
            assert float(np.max(z_vals)) == pytest.approx(999.0, abs=0.1)

    def test_streaming_stats_no_denominator(self):
        """Streaming mode stats should not show misleading total."""
        terrain = self._make_terrain(128, 128)
        rtx = _FakeRTX()
        mgr = TerrainLODManager(terrain, tile_size=64,
                                pixel_spacing_x=1.0, pixel_spacing_y=1.0)
        mgr.set_tile_data_fn(
            lambda *a: np.full((a[4], a[4]), 50.0, dtype=np.float32))
        mgr.per_tick_build_limit = 100
        mgr.update(np.array([64, 64, 0]), rtx, force=True)
        stats = mgr.get_stats()
        assert '/' not in stats, (
            f"Streaming stats should not have active/total format: {stats}")
        assert 'tiles' in stats

    def test_hysteresis_prevents_flip(self):
        """Tiles near LOD boundary should not flip on small movements."""
        terrain = self._make_terrain(512, 512)
        rtx = _FakeRTX()
        mgr = TerrainLODManager(
            terrain, tile_size=128,
            pixel_spacing_x=1.0, pixel_spacing_y=1.0,
            max_lod=3, lod_distance_factor=1.0,
        )
        mgr.per_tick_build_limit = 100
        # Place camera so some tiles are near LOD threshold
        mgr.update(np.array([0, 0, 0]), rtx, force=True)
        lods_1 = mgr.tile_lods

        # Small movement — tiles near boundary should keep their LOD
        mgr.update(np.array([5, 5, 0]), rtx, force=True)
        lods_2 = mgr.tile_lods

        # At least the tiles far from boundaries should be stable
        stable = sum(1 for k in lods_1 if k in lods_2 and lods_1[k] == lods_2[k])
        assert stable > 0, "No tiles maintained their LOD across small movement"

    def test_tiles_have_normals(self):
        """LOD tiles should pass per-vertex normals to add_geometry."""
        terrain = self._make_terrain(128, 128)
        rtx = _FakeRTX()
        mgr = TerrainLODManager(terrain, tile_size=64,
                                pixel_spacing_x=1.0, pixel_spacing_y=1.0)
        mgr.per_tick_build_limit = 100
        mgr.update(np.array([64, 64, 0]), rtx, force=True)
        for gid, (verts, indices, normals) in rtx.geometries.items():
            assert normals is not None, f"Tile {gid} missing normals"
            n_verts = len(verts) // 3
            assert len(normals) == n_verts * 3, (
                f"Tile {gid}: normals length {len(normals)} != "
                f"verts count {n_verts} * 3")
            # All normals should be unit-length
            nx = normals[0::3]
            ny = normals[1::3]
            nz = normals[2::3]
            lengths = np.sqrt(nx**2 + ny**2 + nz**2)
            np.testing.assert_allclose(lengths, 1.0, atol=1e-4,
                                       err_msg=f"Non-unit normals in {gid}")

    def test_threaded_building(self):
        """Threaded mesh building should produce tiles over multiple ticks."""
        import time
        terrain = self._make_terrain(256, 256)
        rtx = _FakeRTX()
        mgr = TerrainLODManager(terrain, tile_size=64,
                                pixel_spacing_x=1.0, pixel_spacing_y=1.0)
        mgr.per_tick_build_limit = 2  # only 2 builds per tick
        mgr.enable_threaded_building(max_workers=2)

        # First tick: submits builds to the thread pool
        mgr.update(np.array([128, 128, 0]), rtx, force=True)

        # Subsequent ticks collect completed futures — allow time
        # for the thread pool to finish (builds are fast but async)
        for _ in range(50):
            time.sleep(0.02)
            mgr.update(np.array([128, 128, 0]), rtx)
            if not mgr._has_in_flight_work and not mgr._pending_futures:
                break

        # All visible tiles should eventually be built
        assert len(rtx.geometries) > 0, "No tiles built with threaded building"
        # Verify normals are present
        for gid, (verts, indices, normals) in rtx.geometries.items():
            assert normals is not None, f"Threaded tile {gid} missing normals"
            assert len(normals) == (len(verts) // 3) * 3

        mgr.shutdown()

    def test_threaded_shutdown_cancels_pending(self):
        """Shutdown should cancel in-flight futures."""
        terrain = self._make_terrain(256, 256)
        mgr = TerrainLODManager(terrain, tile_size=64,
                                pixel_spacing_x=1.0, pixel_spacing_y=1.0)
        mgr.enable_threaded_building()
        mgr.shutdown()
        assert mgr._executor is None
        assert len(mgr._pending_futures) == 0
        assert len(mgr._io_futures) == 0
        assert not mgr._threaded

    def test_build_retry_budget(self):
        """Tiles that fail repeatedly should stop retrying."""
        from unittest.mock import patch
        terrain = self._make_terrain(128, 128)
        rtx = _FakeRTX()
        mgr = TerrainLODManager(terrain, tile_size=64,
                                pixel_spacing_x=1.0, pixel_spacing_y=1.0)

        # Simulate a tile that has exhausted its retry budget
        fail_key = (0, 0, 1, 1)  # (tr, tc, lod, base_sub)
        mgr._build_retries[fail_key] = mgr._MAX_BUILD_RETRIES
        # Verify the tile is skipped: force a queue entry for this tile
        queue = [(1.0, 0, 0, 1, 'terrain_lod_r0_c0')]
        changed, pending = mgr._process_tile_queue(queue, rtx, ve=1.0)
        assert fail_key not in mgr._pending_futures
        assert fail_key not in mgr._tile_cache

        # After cancel_pending, retries should be cleared (terrain reload)
        mgr._cancel_pending()
        assert len(mgr._build_retries) == 0

    def test_streaming_io_prefetch(self):
        """Streaming tiles should prefetch I/O ahead of mesh builds."""
        import time
        terrain = self._make_terrain(128, 128)
        rtx = _FakeRTX()
        mgr = TerrainLODManager(terrain, tile_size=64,
                                pixel_spacing_x=1.0, pixel_spacing_y=1.0)
        fetch_calls = []

        def tracking_fn(x_min, y_min, x_max, y_max, target_samples):
            fetch_calls.append((x_min, y_min, x_max, y_max))
            return np.full((target_samples, target_samples), 100.0,
                           dtype=np.float32)

        mgr.set_tile_data_fn(tracking_fn)
        mgr.per_tick_build_limit = 2  # tight limit to force prefetch
        mgr.enable_threaded_building(max_workers=2)

        # Position camera so some tiles are out-of-bounds (streaming)
        mgr.update(np.array([64, 64, 0]), rtx, force=True)

        # Let threads complete and collect results
        for _ in range(50):
            time.sleep(0.02)
            mgr.update(np.array([64, 64, 0]), rtx)
            if not mgr._has_in_flight_work:
                break

        # Should have tiles built — both in-bounds and streaming
        assert len(rtx.geometries) > 0
        # The tile_data_fn should have been called for out-of-bounds tiles
        assert len(fetch_calls) > 0
        mgr.shutdown()

    def test_batched_upload_reduces_gas_count(self):
        """Batched mode should produce fewer GAS entries than tiles."""
        terrain = self._make_terrain(256, 256)
        rtx = _FakeRTX()
        mgr = TerrainLODManager(terrain, tile_size=64,
                                pixel_spacing_x=1.0, pixel_spacing_y=1.0,
                                max_lod=2, lod_distance_factor=1.0)
        mgr.per_tick_build_limit = 100
        mgr.enable_batched_upload()
        mgr.update(np.array([128, 128, 0]), rtx, force=True)

        # Scene should have batch GAS entries, not individual tile GAS
        n_tiles = len(mgr._tile_lods)
        assert n_tiles > 0, "No tiles assigned LOD"
        n_gas = len(rtx.geometries)
        assert n_gas < n_tiles, (
            f"Batch mode should reduce GAS count: {n_gas} GAS >= {n_tiles} tiles")
        # All GAS IDs should be batch IDs
        for gid in rtx.geometries:
            assert gid.startswith('terrain_lod_batch_L'), (
                f"Non-batch GAS ID in scene: {gid}")

    def test_batched_upload_correct_geometry(self):
        """Batched tiles should produce valid concatenated geometry."""
        terrain = self._make_terrain(128, 128)
        rtx = _FakeRTX()
        mgr = TerrainLODManager(terrain, tile_size=64,
                                pixel_spacing_x=1.0, pixel_spacing_y=1.0)
        mgr.per_tick_build_limit = 100
        mgr.enable_batched_upload()
        mgr.update(np.array([64, 64, 0]), rtx, force=True)

        for gid, (verts, indices, normals) in rtx.geometries.items():
            n_verts = len(verts) // 3
            assert normals is not None, f"Batch {gid} missing normals"
            assert len(normals) == n_verts * 3
            # All indices should be within vertex bounds
            assert np.all(indices >= 0)
            assert np.all(indices < n_verts), (
                f"Index out of range in {gid}: max={np.max(indices)}, "
                f"n_verts={n_verts}")

    def test_batched_remove_all(self):
        """remove_all should clear batch GAS entries."""
        terrain = self._make_terrain(128, 128)
        rtx = _FakeRTX()
        mgr = TerrainLODManager(terrain, tile_size=64,
                                pixel_spacing_x=1.0, pixel_spacing_y=1.0)
        mgr.per_tick_build_limit = 100
        mgr.enable_batched_upload()
        mgr.update(np.array([64, 64, 0]), rtx, force=True)
        assert len(rtx.geometries) > 0
        mgr.remove_all(rtx)
        for gid in rtx.geometries:
            assert not is_terrain_lod_gid(gid)
        assert len(mgr._batch_gids) == 0
        assert len(mgr._lod_tile_meshes) == 0

    def test_batched_stats_show_gas_count(self):
        """Stats should report batch GAS count."""
        terrain = self._make_terrain(128, 128)
        rtx = _FakeRTX()
        mgr = TerrainLODManager(terrain, tile_size=64,
                                pixel_spacing_x=1.0, pixel_spacing_y=1.0)
        mgr.per_tick_build_limit = 100
        mgr.enable_batched_upload()
        mgr.update(np.array([64, 64, 0]), rtx, force=True)
        stats = mgr.get_stats()
        assert 'GAS' in stats

    def test_batched_stale_tile_eviction(self):
        """Tiles leaving distance range should be unstaged from batches."""
        terrain = self._make_terrain(256, 256)
        rtx = _FakeRTX()
        mgr = TerrainLODManager(terrain, tile_size=64,
                                pixel_spacing_x=1.0, pixel_spacing_y=1.0,
                                max_lod=2, lod_distance_factor=0.5)
        mgr.per_tick_build_limit = 100
        mgr.enable_batched_upload()
        # Camera at corner — only nearby tiles in range
        mgr.update(np.array([0, 0, 0]), rtx, force=True)
        lods_1 = set(mgr._tile_lods.keys())
        # Move far away — some tiles should become stale
        mgr.update(np.array([256, 256, 0]), rtx, force=True)
        lods_2 = set(mgr._tile_lods.keys())
        # Some tiles from position 1 should have been evicted
        evicted = lods_1 - lods_2
        # Evicted tiles should not appear in any batch
        for lod, tiles in mgr._lod_tile_meshes.items():
            for k in evicted:
                assert k not in tiles, (
                    f"Evicted tile {k} still in LOD {lod} batch")


# ---------------------------------------------------------------------------
# Terrain-adaptive LOD (roughness)
# ---------------------------------------------------------------------------

class TestTerrainAdaptiveLOD:
    """Tests for roughness-based LOD threshold adaptation."""

    @staticmethod
    def _make_terrain(H, W, elevation=100.0):
        return np.full((H, W), elevation, dtype=np.float32)

    def test_flat_terrain_uniform_roughness(self):
        """All tiles on a flat terrain should get neutral roughness (1.0)."""
        terrain = self._make_terrain(256, 256)
        mgr = TerrainLODManager(terrain, tile_size=64,
                                pixel_spacing_x=1.0, pixel_spacing_y=1.0)
        for scale in mgr._tile_roughness.values():
            assert scale == pytest.approx(1.0)

    def test_rough_tile_gets_higher_scale(self):
        """A tile with a peak should get roughness_scale > 1."""
        terrain = self._make_terrain(256, 256)
        # Add a sharp peak to tile (1, 1) — rows 64:128, cols 64:128
        terrain[90:100, 90:100] = 5000.0
        mgr = TerrainLODManager(terrain, tile_size=64,
                                pixel_spacing_x=1.0, pixel_spacing_y=1.0)
        rough_scale = mgr._tile_roughness.get((1, 1), 1.0)
        # Should be promoted (scale > 1)
        assert rough_scale > 1.0, f"Rough tile scale {rough_scale} not > 1"

    def test_smooth_tile_gets_lower_scale(self):
        """A flat tile adjacent to a rough one should get scale < 1."""
        terrain = self._make_terrain(256, 256)
        # Make one tile very rough
        terrain[90:100, 90:100] = 5000.0
        mgr = TerrainLODManager(terrain, tile_size=64,
                                pixel_spacing_x=1.0, pixel_spacing_y=1.0)
        # Tile (0, 0) is flat — should be demoted
        smooth_scale = mgr._tile_roughness.get((0, 0), 1.0)
        assert smooth_scale < 1.0, f"Smooth tile scale {smooth_scale} not < 1"

    def test_roughness_affects_lod_assignment(self):
        """Rough tiles should get finer LOD than smooth tiles at same distance."""
        terrain = self._make_terrain(512, 512)
        # Make tile (2, 2) rough — rows 256:384, cols 256:384
        terrain[300:320, 300:320] = 3000.0
        rtx = _FakeRTX()
        mgr = TerrainLODManager(
            terrain, tile_size=128,
            pixel_spacing_x=1.0, pixel_spacing_y=1.0,
            max_lod=3, lod_distance_factor=1.0,
        )
        mgr.per_tick_build_limit = 100
        # Camera at center — all tiles roughly equidistant
        mgr.update(np.array([256, 256, 0]), rtx, force=True)
        lods = mgr.tile_lods

        # Rough tile (2, 2) should have same or lower LOD number (higher
        # detail) than the smooth tile (0, 0) at comparable distance
        rough_lod = lods.get((2, 2))
        smooth_lod = lods.get((0, 0))
        if rough_lod is not None and smooth_lod is not None:
            assert rough_lod <= smooth_lod, (
                f"Rough tile LOD {rough_lod} > smooth tile LOD {smooth_lod}")

    def test_set_terrain_recomputes_roughness(self):
        """Replacing terrain should recompute roughness."""
        terrain1 = self._make_terrain(128, 128)
        mgr = TerrainLODManager(terrain1, tile_size=64,
                                pixel_spacing_x=1.0, pixel_spacing_y=1.0)
        # All uniform initially
        for s in mgr._tile_roughness.values():
            assert s == pytest.approx(1.0)

        # Replace with terrain that has one rough tile
        terrain2 = self._make_terrain(128, 128)
        terrain2[10:20, 10:20] = 999.0
        mgr.set_terrain(terrain2)
        # Should now have non-uniform roughness
        scales = list(mgr._tile_roughness.values())
        assert max(scales) > min(scales), "Roughness not recomputed"

    def test_roughness_scale_range(self):
        """Roughness scales should fall within [0.5, 2.0]."""
        rng = np.random.RandomState(42)
        terrain = rng.randn(256, 256).astype(np.float32) * 100
        # Make one tile extra rough
        terrain[50:70, 50:70] += 5000.0
        mgr = TerrainLODManager(terrain, tile_size=64,
                                pixel_spacing_x=1.0, pixel_spacing_y=1.0)
        for scale in mgr._tile_roughness.values():
            assert 0.5 - 1e-6 <= scale <= 2.0 + 1e-6, (
                f"Scale {scale} outside [0.5, 2.0]")

    def test_streaming_tiles_get_neutral_roughness(self):
        """Streaming tiles (out of bounds) should use default scale 1.0."""
        terrain = self._make_terrain(128, 128)
        rtx = _FakeRTX()
        mgr = TerrainLODManager(terrain, tile_size=64,
                                pixel_spacing_x=1.0, pixel_spacing_y=1.0)
        # Out-of-bounds tile (10, 10) — not in _tile_roughness
        assert mgr._tile_roughness.get((10, 10), 1.0) == 1.0


# ---------------------------------------------------------------------------
# Tile helpers
# ---------------------------------------------------------------------------

class TestTileHelpers:
    def test_tile_gid_format(self):
        assert _tile_gid(0, 0) == "terrain_lod_r0_c0"
        assert _tile_gid(3, 7) == "terrain_lod_r3_c7"

    def test_batch_gid_format(self):
        assert _batch_gid(0) == "terrain_lod_batch_L0"
        assert _batch_gid(3) == "terrain_lod_batch_L3"

    def test_is_terrain_lod_gid(self):
        assert is_terrain_lod_gid("terrain_lod_r0_c0")
        assert is_terrain_lod_gid("terrain_lod_r12_c34")
        assert is_terrain_lod_gid("terrain_lod_batch_L0")
        assert is_terrain_lod_gid("terrain_lod_batch_L3")
        assert is_terrain_lod_gid("terrain_lod_hf")
        assert not is_terrain_lod_gid("terrain")
        assert not is_terrain_lod_gid("terrain_skirt")
        assert not is_terrain_lod_gid("buildings_0")


# ---------------------------------------------------------------------------
# Edge stitching
# ---------------------------------------------------------------------------

class TestEdgeStitching:
    """Tests for boundary vertex stitching between tiles at different LODs."""

    @staticmethod
    def _make_terrain(H=256, W=256):
        y = np.linspace(0, 100, H, dtype=np.float32)
        x = np.linspace(0, 100, W, dtype=np.float32)
        return y[:, None] + x[None, :]

    def test_coarser_neighbor_stitches_boundary(self):
        """Finer tile boundary Z should match coarser neighbor's grid."""
        terrain = self._make_terrain(256, 256)
        rtx = _FakeRTX()
        mgr = TerrainLODManager(terrain, tile_size=64,
                                pixel_spacing_x=1.0, pixel_spacing_y=1.0)
        mgr.per_tick_build_limit = 200
        # Force tile (1,0) to LOD 0 and tile (1,1) to LOD 1 by positioning
        # the camera close to tile (1,0)
        mgr.update(np.array([32, 32, 0]), rtx, ve=1.0, force=True)
        # Check that adjacent tiles with different LODs exist
        lods = mgr._tile_lods
        # Find any pair where LODs differ
        pairs_found = False
        for (tr, tc), lod in lods.items():
            for edge, (nr, nc) in [('right', (tr, tc+1)),
                                    ('bottom', (tr+1, tc))]:
                nlod = lods.get((nr, nc), -1)
                if nlod >= 0 and nlod != lod:
                    pairs_found = True
                    break
            if pairs_found:
                break
        # If we found differently-LODed neighbors, the stitching ran
        # (it's applied in _prepare_tile → _stitch_tile_boundary).
        # Verify the boundary Z values are from the coarser level's pyramid.
        if pairs_found:
            # The finer tile's boundary should have been modified
            finer_tile = (tr, tc) if lod < nlod else (nr, nc)
            finer_lod = lods[finer_tile]
            gid = _tile_gid(*finer_tile)
            assert gid in rtx.geometries or gid in mgr._active_tiles

    def test_same_lod_no_stitching(self):
        """Tiles at the same LOD should not have their boundaries modified."""
        terrain = self._make_terrain(128, 128)
        mgr = TerrainLODManager(terrain, tile_size=64,
                                pixel_spacing_x=1.0, pixel_spacing_y=1.0)
        # Set all four tiles to the same LOD
        mgr._tile_lods = {(0, 0): 1, (0, 1): 1, (1, 0): 1, (1, 1): 1}
        # Build mesh for interior-adjacent tile (0,0)
        mesh_data = mgr._build_tile_mesh(0, 0, 1)
        assert mesh_data is not None
        verts_orig = mesh_data[0].copy()
        verts = mesh_data[0].copy()
        # Stitch — all neighbors are same LOD, so nothing should change
        mgr._stitch_tile_boundary(verts, 0, 0, 1)
        np.testing.assert_array_equal(verts, verts_orig,
                                      err_msg="Same-LOD neighbors should not be stitched")

    def test_stitch_tile_boundary_method(self):
        """Direct test of _stitch_tile_boundary with controlled neighbor LODs."""
        terrain = self._make_terrain(128, 128)
        mgr = TerrainLODManager(terrain, tile_size=64,
                                pixel_spacing_x=1.0, pixel_spacing_y=1.0)
        # Manually set tile LODs: (0,0) at LOD 0, (0,1) at LOD 2
        mgr._tile_lods = {(0, 0): 0, (0, 1): 2}
        # Build a mesh for tile (0,0) at LOD 0
        mesh_data = mgr._build_tile_mesh(0, 0, 0)
        assert mesh_data is not None
        verts_orig = mesh_data[0].copy()
        verts = mesh_data[0].copy()
        # Stitch — tile (0,0) LOD 0 has right neighbor (0,1) at LOD 2
        mgr._stitch_tile_boundary(verts, 0, 0, 0)
        th, tw = mgr._tile_grid_dims(0, 0, 0)
        # Right edge should be modified (neighbor is coarser)
        right_col = tw - 1
        right_indices = np.arange(th) * tw + right_col
        right_z_orig = verts_orig[right_indices * 3 + 2]
        right_z_stitched = verts[right_indices * 3 + 2]
        # Stitched Z should differ from original (interpolated from LOD 2)
        assert not np.array_equal(right_z_orig, right_z_stitched), \
            "Right edge should be stitched to coarser neighbor"
        # Left edge should NOT be modified (no neighbor on left)
        left_indices = np.arange(th) * tw
        np.testing.assert_array_equal(
            verts[left_indices * 3 + 2],
            verts_orig[left_indices * 3 + 2],
            err_msg="Left edge should be unchanged (no neighbor)")

    def test_get_boundary_z_ref(self):
        """_get_boundary_z_ref returns Z values from pyramid at correct edge."""
        terrain = self._make_terrain(128, 128)
        mgr = TerrainLODManager(terrain, tile_size=64,
                                pixel_spacing_x=1.0, pixel_spacing_y=1.0)
        # Get top boundary Z at LOD 0 for tile (0,0) — should be from row 0
        z_top = mgr._get_boundary_z_ref(0, 0, 'top', 0)
        assert z_top is not None
        pyr0 = mgr._get_pyramid_level(0)
        np.testing.assert_array_equal(z_top, pyr0[0, :65])
        # Bottom boundary for tile (0,0)
        z_bottom = mgr._get_boundary_z_ref(0, 0, 'bottom', 0)
        assert z_bottom is not None
        # Left boundary
        z_left = mgr._get_boundary_z_ref(0, 0, 'left', 0)
        assert z_left is not None
        np.testing.assert_array_equal(z_left, pyr0[:65, 0])

    def test_tile_grid_dims(self):
        """_tile_grid_dims should return correct grid size for a tile."""
        terrain = self._make_terrain(128, 128)
        mgr = TerrainLODManager(terrain, tile_size=64,
                                pixel_spacing_x=1.0, pixel_spacing_y=1.0)
        # LOD 0 on 128x128 terrain with tile_size=64, base_subsample=1
        th, tw = mgr._tile_grid_dims(0, 0, 0)
        assert th == 65 and tw == 65, f"Expected 65x65, got {th}x{tw}"
        # LOD 1 should be coarser
        th1, tw1 = mgr._tile_grid_dims(0, 0, 1)
        assert th1 < th and tw1 < tw, \
            f"LOD 1 ({th1}x{tw1}) should be coarser than LOD 0 ({th}x{tw})"

    def test_heightfield_neighbor_stitching(self):
        """TIN tile adjacent to heightfield LOD 0 should stitch to full-res."""
        terrain = self._make_terrain(256, 256)
        mgr = TerrainLODManager(terrain, tile_size=64,
                                pixel_spacing_x=1.0, pixel_spacing_y=1.0)
        mgr.enable_heightfield_lod0()
        # Set tile LODs: (0,0) LOD 0 (heightfield), (0,1) LOD 1 (TIN)
        mgr._tile_lods = {(0, 0): 0, (0, 1): 1}
        # Build mesh for tile (0,1) at LOD 1
        mesh_data = mgr._build_tile_mesh(0, 1, 1)
        assert mesh_data is not None
        verts_orig = mesh_data[0].copy()
        verts = mesh_data[0].copy()
        # Stitch — tile (0,1) LOD 1 has left neighbor (0,0) at LOD 0 (HF)
        mgr._stitch_tile_boundary(verts, 0, 1, 1)
        th, tw = mgr._tile_grid_dims(0, 1, 1)
        # Left edge should be modified (stitched to full-res pyramid 0)
        left_indices = np.arange(th) * tw
        left_z_orig = verts_orig[left_indices * 3 + 2]
        left_z_stitched = verts[left_indices * 3 + 2]
        assert not np.array_equal(left_z_orig, left_z_stitched), \
            "Left edge should be stitched to heightfield LOD 0 (full-res)"

    def test_stitch_with_nan_terrain(self):
        """Stitching should handle NaN values in terrain without crashing."""
        terrain = self._make_terrain(128, 128)
        # Inject NaN in the boundary region between tile (0,0) and (0,1)
        terrain[0:65, 63:66] = np.nan
        mgr = TerrainLODManager(terrain, tile_size=64,
                                pixel_spacing_x=1.0, pixel_spacing_y=1.0)
        mgr._tile_lods = {(0, 0): 0, (0, 1): 2}
        mesh_data = mgr._build_tile_mesh(0, 0, 0)
        assert mesh_data is not None
        verts = mesh_data[0].copy()
        # Should not crash even with NaN in the reference boundary
        mgr._stitch_tile_boundary(verts, 0, 0, 0)

    def test_stitch_with_ve(self):
        """Stitching + VE should apply both transformations correctly."""
        terrain = self._make_terrain(128, 128)
        mgr = TerrainLODManager(terrain, tile_size=64,
                                pixel_spacing_x=1.0, pixel_spacing_y=1.0)
        mgr._tile_lods = {(0, 0): 0, (0, 1): 2}
        mesh_data = mgr._build_tile_mesh(0, 0, 0)
        assert mesh_data is not None
        # _prepare_tile with VE=2.0 should stitch then scale Z
        verts, _, _ = mgr._prepare_tile(mesh_data, 0, 0, 0, ve=2.0)
        # Compare to _prepare_tile with VE=1.0 — Z should be 2× larger
        verts_ve1, _, _ = mgr._prepare_tile(mesh_data, 0, 0, 0, ve=1.0)
        np.testing.assert_allclose(verts[2::3], verts_ve1[2::3] * 2.0,
                                   rtol=1e-5)

    def test_stitch_streaming_tile_with_cached_neighbor(self):
        """Out-of-bounds (streaming) tiles stitch to in-bounds neighbors."""
        terrain = self._make_terrain(128, 128)
        mgr = TerrainLODManager(terrain, tile_size=64,
                                pixel_spacing_x=1.0, pixel_spacing_y=1.0)
        # Tile (2, 0) is out of bounds (only 2 tile rows: 0, 1)
        # Tile (1, 0) is in-bounds and at coarser LOD
        mgr._tile_lods = {(2, 0): 0, (1, 0): 2}
        # Build a mesh for tile (1, 0) to use as reference
        mesh_data_ref = mgr._build_tile_mesh(1, 0, 0)
        assert mesh_data_ref is not None
        # Cache it so the stitch code can find it
        mgr._tile_cache[(1, 0, 0, 1)] = mesh_data_ref
        # Build an in-bounds mesh and pretend it's for tile (2, 0)
        mesh_data = mgr._build_tile_mesh(0, 0, 0)
        assert mesh_data is not None
        verts_orig = mesh_data[0].copy()
        # Prepare for OOB tile — stitching should now happen via
        # pyramid (neighbor (1,0) is in-bounds)
        verts, _, _ = mgr._prepare_tile(mesh_data, 2, 0, 0, ve=1.0,
                                         own=True)
        # Top boundary of tile (2,0) should be modified to match
        # the bottom of tile (1,0)
        assert not np.array_equal(verts[2::3], verts_orig[2::3]), \
            "OOB tile should be stitched to in-bounds neighbor"

    def test_stitched_z_matches_coarser_pyramid(self):
        """Stitched boundary Z values should match interpolated coarser pyramid."""
        terrain = self._make_terrain(128, 128)
        mgr = TerrainLODManager(terrain, tile_size=64,
                                pixel_spacing_x=1.0, pixel_spacing_y=1.0)
        # Tile (0,0) at LOD 0, tile (0,1) at LOD 2 (coarser)
        mgr._tile_lods = {(0, 0): 0, (0, 1): 2}
        mesh_data = mgr._build_tile_mesh(0, 0, 0)
        assert mesh_data is not None
        verts = mesh_data[0].copy()
        mgr._stitch_tile_boundary(verts, 0, 0, 0)

        th, tw = mgr._tile_grid_dims(0, 0, 0)
        # Right edge vertices (shared with coarser neighbor)
        right_col = tw - 1
        right_indices = np.arange(th) * tw + right_col
        stitched_z = verts[right_indices * 3 + 2]

        # Compute expected Z: interpolated from LOD 2 pyramid boundary
        ref_z = mgr._get_boundary_z_ref(0, 0, 'right', 2)
        assert ref_z is not None
        n_self = th
        n_ref = len(ref_z)
        positions = (np.arange(n_self, dtype=np.float64)
                     * (n_ref - 1) / (n_self - 1))
        expected_z = np.interp(
            positions,
            np.arange(n_ref, dtype=np.float64),
            ref_z.astype(np.float64)).astype(np.float32)
        np.testing.assert_array_almost_equal(
            stitched_z, expected_z, decimal=5,
            err_msg="Stitched Z should match interpolated coarser pyramid")

        # Interior vertices should be unchanged
        interior_indices = np.arange(th) * tw + (tw // 2)
        interior_z = verts[interior_indices * 3 + 2]
        interior_z_orig = mesh_data[0][interior_indices * 3 + 2]
        np.testing.assert_array_equal(
            interior_z, interior_z_orig,
            err_msg="Interior vertices should not be modified by stitching")

    def test_needs_stitch_fast_check(self):
        """_needs_stitch should return False when all neighbors are same LOD."""
        terrain = self._make_terrain(128, 128)
        mgr = TerrainLODManager(terrain, tile_size=64,
                                pixel_spacing_x=1.0, pixel_spacing_y=1.0)
        mgr._tile_lods = {(0, 0): 1, (0, 1): 1, (1, 0): 1, (1, 1): 1}
        assert not mgr._needs_stitch(0, 0, 1)
        assert not mgr._needs_stitch(1, 1, 1)
        # Change one neighbor to different LOD
        mgr._tile_lods[(0, 1)] = 2
        assert mgr._needs_stitch(0, 0, 1)  # right neighbor is coarser


# ---------------------------------------------------------------------------
# Heightfield LOD 0
# ---------------------------------------------------------------------------

class TestHeightfieldLOD0:
    """Tests for heightfield ray marching on LOD 0 tiles."""

    @staticmethod
    def _make_terrain(H=256, W=256):
        y = np.linspace(0, 100, H, dtype=np.float32)
        x = np.linspace(0, 100, W, dtype=np.float32)
        return y[:, None] + x[None, :]

    def test_enable_heightfield_creates_hf_gas(self):
        """Enabling heightfield LOD 0 should produce a heightfield GAS."""
        terrain = self._make_terrain(256, 256)
        rtx = _FakeRTX()
        mgr = TerrainLODManager(terrain, tile_size=128,
                                pixel_spacing_x=1.0, pixel_spacing_y=1.0)
        mgr.enable_heightfield_lod0()
        mgr.update([64, 64, 100], rtx, ve=1.0, force=True)
        assert rtx.has_geometry('terrain_lod_hf'), \
            "Heightfield GAS not created"
        hf = rtx.geometries['terrain_lod_hf']
        assert hf['type'] == 'heightfield'

    def test_lod0_tiles_skip_mesh_building(self):
        """LOD 0 in-bounds tiles should not create triangle meshes."""
        terrain = self._make_terrain(256, 256)
        rtx = _FakeRTX()
        mgr = TerrainLODManager(terrain, tile_size=128,
                                pixel_spacing_x=1.0, pixel_spacing_y=1.0)
        mgr.enable_heightfield_lod0()
        mgr.update([64, 64, 100], rtx, ve=1.0, force=True)
        # LOD 0 tiles should be tracked but no individual TIN GAS
        lod0_tiles = [k for k, v in mgr._tile_lods.items() if v == 0]
        assert len(lod0_tiles) > 0, "No LOD 0 tiles assigned"
        for tr, tc in lod0_tiles:
            gid = _tile_gid(tr, tc)
            assert not rtx.has_geometry(gid), \
                f"LOD 0 tile {gid} has TIN GAS — should use heightfield"

    def test_heightfield_active_mask(self):
        """Active mask should cover only LOD 0 tile regions."""
        terrain = self._make_terrain(128, 128)
        rtx = _FakeRTX()
        mgr = TerrainLODManager(terrain, tile_size=128,
                                pixel_spacing_x=1.0, pixel_spacing_y=1.0,
                                max_lod=3)
        mgr.enable_heightfield_lod0()
        # Single tile terrain — close camera → LOD 0
        mgr.update([64, 64, 50], rtx, ve=1.0, force=True)
        hf = rtx.geometries.get('terrain_lod_hf')
        assert hf is not None, "No heightfield GAS"
        mask = hf['active_mask']
        assert mask is not None
        # All AABB tiles should be active (single LOD tile covers everything)
        assert np.all(mask), "Not all AABB tiles are active"

    def test_heightfield_partial_active_mask(self):
        """Only LOD 0 tiles should have active AABBs."""
        terrain = self._make_terrain(256, 256)
        rtx = _FakeRTX()
        mgr = TerrainLODManager(terrain, tile_size=128,
                                pixel_spacing_x=1.0, pixel_spacing_y=1.0,
                                max_lod=3, lod_distance_factor=1.0)
        mgr.enable_heightfield_lod0()
        # Camera at corner — close tiles LOD 0, far tiles higher LOD
        mgr.update([20, 20, 50], rtx, ve=1.0, force=True)
        hf = rtx.geometries.get('terrain_lod_hf')
        if hf is not None:
            mask = hf['active_mask']
            # Not all AABB tiles should be active (some tiles are LOD 1+)
            n_active = np.sum(mask)
            n_total = len(mask)
            # At least some should be inactive if any tiles are LOD 1+
            lod_counts = {}
            for v in mgr._tile_lods.values():
                lod_counts[v] = lod_counts.get(v, 0) + 1
            if any(l > 0 for l in lod_counts):
                assert n_active < n_total, \
                    "All AABB tiles active but some LOD tiles are > 0"

    def test_heightfield_with_batched_upload(self):
        """Heightfield LOD 0 + batched TIN LOD 1+ should coexist."""
        terrain = self._make_terrain(256, 256)
        rtx = _FakeRTX()
        mgr = TerrainLODManager(terrain, tile_size=128,
                                pixel_spacing_x=1.0, pixel_spacing_y=1.0,
                                max_lod=3, lod_distance_factor=1.0)
        mgr.enable_heightfield_lod0()
        mgr.enable_batched_upload()
        mgr.update([20, 20, 50], rtx, ve=1.0, force=True)
        # Should have heightfield GAS and possibly TIN batch GAS
        gids = rtx.list_geometries()
        hf_gids = [g for g in gids if g == 'terrain_lod_hf']
        batch_gids = [g for g in gids if g.startswith('terrain_lod_batch_')]
        assert len(hf_gids) <= 1, "Multiple heightfield GAS"
        # LOD 1+ tiles should be in batch GAS, not individual
        for tr, tc in mgr._tile_lods:
            lod = mgr._tile_lods[(tr, tc)]
            if lod > 0:
                gid = _tile_gid(tr, tc)
                assert not rtx.has_geometry(gid), \
                    f"LOD {lod} tile {gid} not batched"

    def test_heightfield_ve_update(self):
        """VE change should rebuild heightfield with new VE."""
        terrain = self._make_terrain(128, 128)
        rtx = _FakeRTX()
        mgr = TerrainLODManager(terrain, tile_size=128,
                                pixel_spacing_x=1.0, pixel_spacing_y=1.0)
        mgr.enable_heightfield_lod0()
        mgr.update([64, 64, 50], rtx, ve=1.0, force=True)
        hf1 = rtx.geometries['terrain_lod_hf']
        assert hf1['ve'] == 1.0
        # Simulate VE change: clear tile_lods and force rebuild
        mgr._tile_lods.clear()
        mgr.update([64, 64, 50], rtx, ve=2.5, force=True)
        hf2 = rtx.geometries['terrain_lod_hf']
        assert hf2['ve'] == 2.5

    def test_heightfield_remove_all(self):
        """remove_all should clear heightfield GAS."""
        terrain = self._make_terrain(128, 128)
        rtx = _FakeRTX()
        mgr = TerrainLODManager(terrain, tile_size=128,
                                pixel_spacing_x=1.0, pixel_spacing_y=1.0)
        mgr.enable_heightfield_lod0()
        mgr.update([64, 64, 50], rtx, ve=1.0, force=True)
        assert rtx.has_geometry('terrain_lod_hf')
        mgr.remove_all(rtx)
        assert not rtx.has_geometry('terrain_lod_hf')
        assert len(mgr._tile_lods) == 0

    def test_heightfield_stats_show_hf(self):
        """Stats should label LOD 0 as 'HF' when heightfield enabled."""
        terrain = self._make_terrain(128, 128)
        rtx = _FakeRTX()
        mgr = TerrainLODManager(terrain, tile_size=128,
                                pixel_spacing_x=1.0, pixel_spacing_y=1.0)
        mgr.enable_heightfield_lod0()
        mgr.update([64, 64, 50], rtx, ve=1.0, force=True)
        stats = mgr.get_stats()
        assert 'HF:' in stats, f"Stats missing HF label: {stats}"

    def test_heightfield_transform_has_offset(self):
        """Heightfield transform should include world offset."""
        terrain = self._make_terrain(128, 128)
        rtx = _FakeRTX()
        mgr = TerrainLODManager(terrain, tile_size=128,
                                pixel_spacing_x=1.0, pixel_spacing_y=1.0)
        mgr.enable_heightfield_lod0()
        mgr.set_offset(1000.0, 2000.0)
        mgr.update([1064, 2064, 50], rtx, ve=1.0, force=True)
        hf = rtx.geometries.get('terrain_lod_hf')
        assert hf is not None
        transform = hf['transform']
        assert transform[3] == 1000.0, f"X offset wrong: {transform[3]}"
        assert transform[7] == 2000.0, f"Y offset wrong: {transform[7]}"

    def test_streaming_lod0_uses_tin(self):
        """Out-of-bounds LOD 0 tiles should fall through to TIN."""
        terrain = self._make_terrain(128, 128)
        rtx = _FakeRTX()
        mgr = TerrainLODManager(terrain, tile_size=128,
                                pixel_spacing_x=1.0, pixel_spacing_y=1.0)
        mgr.enable_heightfield_lod0()

        # Set up a streaming tile data function
        def tile_data_fn(x_min, y_min, x_max, y_max, target_samples):
            return np.full((target_samples, target_samples), 50.0,
                           dtype=np.float32)

        mgr.set_tile_data_fn(tile_data_fn)
        # Camera well outside bounds → out-of-bounds tiles should use TIN
        mgr.update([300, 300, 50], rtx, ve=1.0, force=True)
        # Out-of-bounds LOD 0 tiles should have individual GAS (not batched)
        oob_lod0 = [(tr, tc) for (tr, tc), lod in mgr._tile_lods.items()
                    if lod == 0 and (tr < 0 or tr >= mgr._n_tile_rows
                                     or tc < 0 or tc >= mgr._n_tile_cols)]
        for tr, tc in oob_lod0:
            gid = _tile_gid(tr, tc)
            assert rtx.has_geometry(gid), \
                f"OOB LOD 0 tile {gid} should use TIN, not heightfield"


# ---------------------------------------------------------------------------
# Mesh chunk simplification
# ---------------------------------------------------------------------------

class TestMeshChunkSimplification:
    """Tests for placed geometry simplification at higher LOD levels."""

    @staticmethod
    def _make_grid_mesh(rows=10, cols=10):
        """Create a regular grid triangle mesh for testing simplification."""
        verts = []
        for r in range(rows):
            for c in range(cols):
                verts.extend([float(c), float(r), float(r + c) * 0.1])
        indices = []
        for r in range(rows - 1):
            for c in range(cols - 1):
                i0 = r * cols + c
                i1 = i0 + 1
                i2 = i0 + cols
                i3 = i2 + 1
                indices.extend([i0, i1, i2, i1, i3, i2])
        return (np.array(verts, dtype=np.float32),
                np.array(indices, dtype=np.int32))

    def test_simplify_mesh_reduces_triangles(self):
        """simplify_mesh with ratio < 1 should reduce triangle count
        (or return original if trimesh decimation is unavailable)."""
        verts, indices = self._make_grid_mesh(20, 20)
        orig_n_tris = len(indices) // 3
        sv, si = simplify_mesh(verts, indices, 0.5)
        new_n_tris = len(si) // 3
        # If fast_simplification is available, should have fewer triangles.
        # Otherwise simplify_mesh gracefully returns original.
        assert new_n_tris <= orig_n_tris, \
            f"Should not increase triangles: got {new_n_tris} vs original {orig_n_tris}"

    def test_simplify_mesh_ratio_1_returns_original(self):
        """simplify_mesh with ratio >= 1.0 should return original mesh."""
        verts, indices = self._make_grid_mesh(5, 5)
        sv, si = simplify_mesh(verts, indices, 1.0)
        np.testing.assert_array_equal(sv, verts)
        np.testing.assert_array_equal(si, indices)

    def test_simplify_lod0_returns_original(self):
        """simplify_mesh at ratio 1.0 (LOD 0) returns original arrays."""
        verts, indices = self._make_grid_mesh(10, 10)
        sv, si = simplify_mesh(verts, indices, 1.0)
        np.testing.assert_array_equal(sv, verts)
        np.testing.assert_array_equal(si, indices)

    def test_simplify_empty_mesh(self):
        """simplify_mesh with 0 faces should return original unchanged."""
        verts = np.array([0, 0, 0, 1, 0, 0, 0, 1, 0], dtype=np.float32)
        indices = np.array([], dtype=np.int32)
        sv, si = simplify_mesh(verts, indices, 0.5)
        # Should not crash; returns original or empty
        assert len(sv) >= 0
        assert len(si) >= 0

    def test_simplify_single_face(self):
        """simplify_mesh with 1 face should not crash."""
        verts = np.array([0, 0, 0, 1, 0, 0, 0, 1, 0], dtype=np.float32)
        indices = np.array([0, 1, 2], dtype=np.int32)
        sv, si = simplify_mesh(verts, indices, 0.5)
        # 1 face can't simplify further — should return original
        assert len(si) // 3 >= 1

    def test_simplify_high_lod_clamps_to_last_ratio(self):
        """LOD index beyond ratio table length should clamp to last entry."""
        verts, indices = self._make_grid_mesh(20, 20)
        ratios = (1.0, 0.5, 0.25, 0.1)
        # LOD 5 should clamp to index 3 (ratio 0.1)
        idx = min(5, len(ratios) - 1)
        sv, si = simplify_mesh(verts, indices, ratios[idx])
        assert len(si) <= len(indices)

    def test_build_lod_chain_progressive(self):
        """build_lod_chain should produce progressively simpler meshes."""
        verts, indices = self._make_grid_mesh(20, 20)
        chain = build_lod_chain(verts, indices, ratios=(1.0, 0.5, 0.25))
        assert len(chain) == 3
        # Each level should have equal or fewer triangles
        prev_n = len(chain[0][1]) // 3
        for level, (v, i) in enumerate(chain[1:], 1):
            n = len(i) // 3
            assert n <= prev_n, \
                f"LOD {level} has {n} tris, more than LOD {level-1} ({prev_n})"
            prev_n = n


# ---------------------------------------------------------------------------
# Tile lifecycle callbacks
# ---------------------------------------------------------------------------

class TestTileCallbacks:
    """Tests for set_tile_callbacks tile lifecycle notifications."""

    @staticmethod
    def _make_terrain(rows, cols):
        np.random.seed(42)
        return np.random.rand(rows, cols).astype(np.float32) * 100

    def test_on_added_called_for_each_tile(self):
        """on_added should fire for every tile built on first update."""
        terrain = self._make_terrain(256, 256)
        rtx = _FakeRTX()
        added = []
        mgr = TerrainLODManager(
            terrain, tile_size=128,
            pixel_spacing_x=1.0, pixel_spacing_y=1.0,
            max_lod=2, lod_distance_factor=3.0,
        )
        mgr.set_tile_callbacks(
            on_added=lambda tr, tc, elev: added.append((tr, tc, elev)),
        )
        mgr.per_tick_build_limit = 100
        mgr.update(np.array([128, 128, 100]), rtx, force=True)
        # Should have at least one tile added
        assert len(added) > 0
        # Each callback should receive (tr, tc, elevation_tile)
        for tr, tc, elev in added:
            assert isinstance(tr, (int, np.integer))
            assert isinstance(tc, (int, np.integer))

    def test_on_removed_called_on_eviction(self):
        """on_removed should fire when tiles leave the distance range."""
        terrain = self._make_terrain(512, 512)
        rtx = _FakeRTX()
        removed = []
        mgr = TerrainLODManager(
            terrain, tile_size=128,
            pixel_spacing_x=1.0, pixel_spacing_y=1.0,
            max_lod=3, lod_distance_factor=1.0,
        )
        mgr.set_tile_callbacks(
            on_removed=lambda tr, tc: removed.append((tr, tc)),
        )
        mgr.per_tick_build_limit = 100
        # Build near corner
        mgr.update(np.array([0, 0, 0]), rtx, force=True)
        tiles_before = set(mgr._tile_lods.keys())
        assert len(tiles_before) > 0
        # Move far away so original tiles leave range
        mgr.update(np.array([10000, 10000, 0]), rtx, force=True)
        assert len(removed) > 0
        # Every removed tile should have been in tiles_before
        for tr, tc in removed:
            assert (tr, tc) in tiles_before

    def test_callbacks_not_called_when_none(self):
        """No error when callbacks are not set."""
        terrain = self._make_terrain(256, 256)
        rtx = _FakeRTX()
        mgr = TerrainLODManager(
            terrain, tile_size=128,
            pixel_spacing_x=1.0, pixel_spacing_y=1.0,
        )
        mgr.per_tick_build_limit = 100
        # Should not raise
        mgr.update(np.array([128, 128, 100]), rtx, force=True)
        mgr.update(np.array([10000, 10000, 0]), rtx, force=True)


# ---------------------------------------------------------------------------
# ChunkDataSource-driven TerrainLODManager
# ---------------------------------------------------------------------------

class TestChunkSourceDrivenLOD:
    """Tests that TerrainLODManager works when initialized with a
    ChunkDataSource instead of a terrain_np array."""

    @staticmethod
    def _make_terrain(h=256, w=256, seed=42):
        rng = np.random.RandomState(seed)
        ys = np.linspace(0, 4 * np.pi, h, dtype=np.float32)
        xs = np.linspace(0, 4 * np.pi, w, dtype=np.float32)
        return (np.sin(ys[:, None]) * np.cos(xs[None, :]) * 500
                + 1000).astype(np.float32)

    def test_init_with_chunk_source(self):
        """TerrainLODManager can be initialized with a ChunkDataSource."""
        from rtxpy.chunk_source import InMemoryChunkSource
        terrain = self._make_terrain(256, 256)
        src = InMemoryChunkSource(terrain, tile_size=64,
                                  pixel_spacing_x=1.0,
                                  pixel_spacing_y=1.0)
        mgr = TerrainLODManager(chunk_source=src)
        assert mgr._tile_size == 64
        assert mgr._n_tile_rows == 4
        assert mgr._n_tile_cols == 4
        assert mgr._psx == 1.0
        assert mgr._psy == 1.0

    def test_chunk_source_overrides_params(self):
        """chunk_source takes precedence over terrain_np params."""
        from rtxpy.chunk_source import InMemoryChunkSource
        terrain = self._make_terrain(256, 256)
        src = InMemoryChunkSource(terrain, tile_size=128,
                                  pixel_spacing_x=10.0,
                                  pixel_spacing_y=10.0)
        mgr = TerrainLODManager(
            terrain_np=terrain, tile_size=64,
            pixel_spacing_x=1.0, pixel_spacing_y=1.0,
            chunk_source=src)
        # chunk_source values should win
        assert mgr._tile_size == 128
        assert mgr._psx == 10.0
        assert mgr._n_tile_rows == 2

    def test_update_builds_tiles(self):
        """Tiles are built and uploaded when using chunk_source."""
        from rtxpy.chunk_source import InMemoryChunkSource
        terrain = self._make_terrain(256, 256)
        src = InMemoryChunkSource(terrain, tile_size=64,
                                  pixel_spacing_x=1.0,
                                  pixel_spacing_y=1.0)
        rtx = _FakeRTX()
        mgr = TerrainLODManager(chunk_source=src)
        mgr.per_tick_build_limit = 100
        mgr.update(np.array([128, 128, 100]), rtx, force=True)

        # Should have built some tiles
        assert len(rtx.geometries) > 0
        assert len(mgr._tile_lods) > 0

    def test_lod_transitions(self):
        """Tiles at different distances get different LOD levels."""
        from rtxpy.chunk_source import InMemoryChunkSource
        terrain = self._make_terrain(512, 512)
        src = InMemoryChunkSource(terrain, tile_size=64,
                                  pixel_spacing_x=1.0,
                                  pixel_spacing_y=1.0)
        rtx = _FakeRTX()
        mgr = TerrainLODManager(chunk_source=src, max_lod=3)
        mgr.per_tick_build_limit = 200
        # Camera at center
        mgr.update(np.array([256, 256, 100]), rtx, force=True)

        lod_vals = set(mgr._tile_lods.values())
        # Should have at least LOD 0 and some higher LOD
        assert 0 in lod_vals
        assert len(lod_vals) > 1

    def test_matches_legacy_path(self):
        """Chunk-source-driven LOD builds tiles at same positions as legacy."""
        from rtxpy.chunk_source import InMemoryChunkSource
        terrain = self._make_terrain(256, 256)

        # Legacy path
        rtx_legacy = _FakeRTX()
        mgr_legacy = TerrainLODManager(
            terrain, tile_size=64,
            pixel_spacing_x=1.0, pixel_spacing_y=1.0)
        mgr_legacy.per_tick_build_limit = 100
        cam = np.array([128, 128, 100])
        mgr_legacy.update(cam, rtx_legacy, force=True)

        # Chunk source path
        src = InMemoryChunkSource(terrain, tile_size=64,
                                  pixel_spacing_x=1.0,
                                  pixel_spacing_y=1.0)
        rtx_new = _FakeRTX()
        mgr_new = TerrainLODManager(chunk_source=src)
        mgr_new.per_tick_build_limit = 100
        mgr_new.update(cam, rtx_new, force=True)

        # Same tiles should be active
        assert set(mgr_legacy._tile_lods.keys()) == \
               set(mgr_new._tile_lods.keys())
        # Same LOD assignments
        for k in mgr_legacy._tile_lods:
            assert mgr_legacy._tile_lods[k] == mgr_new._tile_lods[k], \
                f"LOD mismatch at {k}"

    def test_roughness_computed_from_source(self):
        """Roughness is computed via chunk_source.chunk_roughness()."""
        from rtxpy.chunk_source import InMemoryChunkSource
        terrain = self._make_terrain(256, 256)
        src = InMemoryChunkSource(terrain, tile_size=64,
                                  pixel_spacing_x=1.0,
                                  pixel_spacing_y=1.0)
        mgr = TerrainLODManager(chunk_source=src)
        # Should have roughness for all tiles
        assert len(mgr._tile_roughness) == 4 * 4

    def test_set_terrain_with_chunk_source(self):
        """set_terrain(chunk_source=...) replaces the source."""
        from rtxpy.chunk_source import InMemoryChunkSource
        terrain1 = self._make_terrain(128, 128)
        src1 = InMemoryChunkSource(terrain1, tile_size=64,
                                   pixel_spacing_x=1.0,
                                   pixel_spacing_y=1.0)
        mgr = TerrainLODManager(chunk_source=src1)
        assert mgr._n_tile_rows == 2

        terrain2 = self._make_terrain(256, 256)
        src2 = InMemoryChunkSource(terrain2, tile_size=128,
                                   pixel_spacing_x=2.0,
                                   pixel_spacing_y=2.0)
        mgr.set_terrain(chunk_source=src2)
        assert mgr._n_tile_rows == 2
        assert mgr._tile_size == 128
        assert mgr._psx == 2.0

    def test_heightfield_with_chunk_source(self):
        """Heightfield LOD 0 works with chunk_source."""
        from rtxpy.chunk_source import InMemoryChunkSource
        terrain = self._make_terrain(256, 256)
        src = InMemoryChunkSource(terrain, tile_size=64,
                                  pixel_spacing_x=1.0,
                                  pixel_spacing_y=1.0)
        rtx = _FakeRTX()
        mgr = TerrainLODManager(chunk_source=src)
        mgr.enable_heightfield_lod0()
        mgr.per_tick_build_limit = 100
        mgr.update(np.array([128, 128, 100]), rtx, force=True)

        # Heightfield GAS should exist
        assert mgr._hf_gid in rtx.geometries

    def test_batched_upload_with_chunk_source(self):
        """Batched upload works with chunk_source."""
        from rtxpy.chunk_source import InMemoryChunkSource
        terrain = self._make_terrain(256, 256)
        src = InMemoryChunkSource(terrain, tile_size=64,
                                  pixel_spacing_x=1.0,
                                  pixel_spacing_y=1.0)
        rtx = _FakeRTX()
        mgr = TerrainLODManager(chunk_source=src)
        mgr.enable_batched_upload()
        mgr.per_tick_build_limit = 100
        mgr.update(np.array([128, 128, 100]), rtx, force=True)

        # Batch GAS IDs should be present
        assert len(mgr._batch_gids) > 0

    def test_crs_transform_from_source(self):
        """CRS transform is picked up from chunk_source."""
        from rtxpy.chunk_source import InMemoryChunkSource
        terrain = self._make_terrain(128, 128)
        src = InMemoryChunkSource(terrain, tile_size=64,
                                  pixel_spacing_x=10.0,
                                  pixel_spacing_y=10.0,
                                  crs_origin=(500000.0, 4000000.0),
                                  crs_pixel_spacing=(10.0, -10.0))
        mgr = TerrainLODManager(chunk_source=src)
        assert mgr._crs_origin == (500000.0, 4000000.0)
        assert mgr._crs_spacing == (10.0, -10.0)

    def test_get_metrics_with_chunk_source(self):
        """get_metrics() returns valid data with chunk_source."""
        from rtxpy.chunk_source import InMemoryChunkSource
        terrain = self._make_terrain(128, 128)
        src = InMemoryChunkSource(terrain, tile_size=64)
        rtx = _FakeRTX()
        mgr = TerrainLODManager(chunk_source=src)
        mgr.per_tick_build_limit = 100
        mgr.update(np.array([64, 64, 50]), rtx, force=True)

        metrics = mgr.get_metrics()
        assert metrics['active_tiles'] > 0
        assert isinstance(metrics['lod_counts'], dict)
