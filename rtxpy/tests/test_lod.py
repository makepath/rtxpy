"""Tests for level-of-detail utilities and terrain LOD manager."""

import numpy as np
import pytest

from rtxpy.lod import (
    compute_lod_level,
    compute_lod_distances,
    simplify_mesh,
    build_lod_chain,
)
from rtxpy.viewer.terrain_lod import (
    TerrainLODManager,
    is_terrain_lod_gid,
    _add_tile_skirt,
    _tile_gid,
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
        self.geometries[gid] = (verts.copy(), indices.copy())
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
        assert "LOD tiles:" in stats


# ---------------------------------------------------------------------------
# Tile helpers
# ---------------------------------------------------------------------------

class TestTileHelpers:
    def test_tile_gid_format(self):
        assert _tile_gid(0, 0) == "terrain_lod_r0_c0"
        assert _tile_gid(3, 7) == "terrain_lod_r3_c7"

    def test_is_terrain_lod_gid(self):
        assert is_terrain_lod_gid("terrain_lod_r0_c0")
        assert is_terrain_lod_gid("terrain_lod_r12_c34")
        assert not is_terrain_lod_gid("terrain")
        assert not is_terrain_lod_gid("terrain_skirt")
        assert not is_terrain_lod_gid("buildings_0")


class TestAddTileSkirt:
    def test_adds_skirt_vertices(self):
        """Skirt should add perimeter + wall vertices."""
        H, W = 4, 4
        n_verts = H * W
        n_tris = (H - 1) * (W - 1) * 2

        verts = np.zeros(n_verts * 3, dtype=np.float32)
        indices = np.zeros(n_tris * 3, dtype=np.int32)
        for h in range(H):
            for w in range(W):
                idx = (h * W + w) * 3
                verts[idx] = float(w)
                verts[idx + 1] = float(h)
                verts[idx + 2] = float(h + w)

        new_v, new_i = _add_tile_skirt(verts, indices, H, W)
        # Perimeter of 4x4 grid: 4+3+3+2 = 12 vertices added
        n_perim = 2 * (H + W) - 4
        assert len(new_v) == (n_verts + n_perim) * 3
        assert len(new_i) > len(indices)

    def test_skirt_z_below_min(self):
        H, W = 3, 3
        verts = np.zeros(9 * 3, dtype=np.float32)
        for i in range(9):
            verts[i * 3 + 2] = 10.0  # all z = 10
        indices = np.zeros(8 * 3, dtype=np.int32)

        new_v, _ = _add_tile_skirt(verts, indices, H, W)
        skirt_z = new_v[9 * 3 + 2::3]  # z of skirt vertices
        assert np.all(skirt_z < 10.0)
