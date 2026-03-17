"""Tests for SceneMeshManager and its integration with TerrainLODManager."""

import os
import tempfile

import numpy as np
import pytest

from rtxpy.viewer.scene_meshes import SceneMeshManager


def _make_scene_zarr(chunk_size=64, terrain_shape=(256, 256)):
    """Create a minimal scene zarr with terrain + placed geometry."""
    import zarr

    tmpdir = tempfile.mkdtemp()
    zarr_path = os.path.join(tmpdir, 'scene.zarr')
    root = zarr.open_group(zarr_path, mode='w')

    # Elevation array
    H, W = terrain_shape
    terrain = np.random.RandomState(42).rand(H, W).astype(np.float32) * 100
    root.create_array(
        'elevation',
        data=terrain,
        chunks=(chunk_size, chunk_size),
        overwrite=True,
    )

    # Meshes group
    mg = root.create_group('meshes', overwrite=True)
    mg.attrs['pixel_spacing'] = [1.0, 1.0]
    mg.attrs['elevation_shape'] = [H, W]
    mg.attrs['elevation_chunks'] = [chunk_size, chunk_size]

    # Add a 'building' geometry with data in chunk (0,0) and (1,1)
    bg = mg.create_group('building')
    bg.attrs['color'] = [0.8, 0.7, 0.6]

    # Chunk (0, 0): a simple box
    verts_00 = np.array([
        10, 10, 0,   20, 10, 0,   20, 20, 0,
        10, 10, 5,   20, 10, 5,   20, 20, 5,
    ], dtype=np.float32)
    indices_00 = np.array([0, 1, 2, 3, 4, 5], dtype=np.int32)
    cg00 = bg.create_group('0_0')
    cg00.create_array('vertices', data=verts_00, chunks=(len(verts_00),))
    cg00.create_array('indices', data=indices_00, chunks=(len(indices_00),))

    # Chunk (1, 1): another box
    verts_11 = np.array([
        80, 80, 0,   90, 80, 0,   90, 90, 0,
        80, 80, 8,   90, 80, 8,   90, 90, 8,
    ], dtype=np.float32)
    indices_11 = np.array([0, 1, 2, 3, 4, 5], dtype=np.int32)
    cg11 = bg.create_group('1_1')
    cg11.create_array('vertices', data=verts_11, chunks=(len(verts_11),))
    cg11.create_array('indices', data=indices_11, chunks=(len(indices_11),))

    # Add a 'road' curve geometry in chunk (0, 1)
    rg = mg.create_group('road')
    rg.attrs['color'] = [0.3, 0.3, 0.3]
    rg.attrs['type'] = 'curve'

    verts_01 = np.array([
        64, 10, 0,   70, 30, 0,   80, 50, 0,
    ], dtype=np.float32)
    widths_01 = np.array([2.0, 2.0, 2.0], dtype=np.float32)
    indices_01 = np.array([0], dtype=np.int32)  # one segment
    cg01 = rg.create_group('0_1')
    cg01.create_array('vertices', data=verts_01, chunks=(len(verts_01),))
    cg01.create_array('widths', data=widths_01, chunks=(len(widths_01),))
    cg01.create_array('indices', data=indices_01, chunks=(len(indices_01),))

    return zarr_path, terrain


class TestSceneMeshManager:
    def test_init_reads_metadata(self):
        zarr_path, _ = _make_scene_zarr()
        mgr = SceneMeshManager(zarr_path)
        assert 'building' in mgr.geometry_ids
        assert 'road' in mgr.geometry_ids
        assert mgr.colors['building'] == (0.8, 0.7, 0.6)
        assert mgr.colors['road'] == (0.3, 0.3, 0.3)

    def test_update_visibility_loads_chunks(self):
        zarr_path, _ = _make_scene_zarr()
        mgr = SceneMeshManager(zarr_path)

        # Make chunk (0,0) visible at LOD 0
        changed = mgr.update_visibility({(0, 0): 0})
        assert changed
        merged = mgr.get_merged()
        assert 'building' in merged
        # Should have the verts from chunk (0,0)
        verts = merged['building'][0]
        assert len(verts) > 0

    def test_multiple_chunks_merge(self):
        zarr_path, _ = _make_scene_zarr()
        mgr = SceneMeshManager(zarr_path)
        mgr.per_tick_load_limit = 10

        # Make both building chunks visible
        changed = mgr.update_visibility({(0, 0): 0, (1, 1): 0})
        assert changed
        merged = mgr.get_merged()
        assert 'building' in merged
        # Should have verts from both chunks merged
        verts = merged['building'][0]
        assert len(verts) == 36  # 18 floats per chunk * 2

    def test_curve_geometry(self):
        zarr_path, _ = _make_scene_zarr()
        mgr = SceneMeshManager(zarr_path)

        changed = mgr.update_visibility({(0, 1): 0})
        assert changed
        merged = mgr.get_merged()
        assert 'road' in merged
        data = merged['road']
        assert len(data) == 3  # (verts, widths, indices)

    def test_chunk_departure_clears(self):
        zarr_path, _ = _make_scene_zarr()
        mgr = SceneMeshManager(zarr_path)

        mgr.update_visibility({(0, 0): 0, (1, 1): 0})
        mgr.get_merged()
        assert 'building' in mgr.active_gids

        # Remove chunk (1,1)
        changed = mgr.update_visibility({(0, 0): 0})
        assert changed
        merged = mgr.get_merged()
        assert 'building' in merged
        # Should only have chunk (0,0) verts now
        assert len(merged['building'][0]) == 18

    def test_no_change_returns_false(self):
        zarr_path, _ = _make_scene_zarr()
        mgr = SceneMeshManager(zarr_path)
        mgr.per_tick_load_limit = 10

        mgr.update_visibility({(0, 0): 0})
        mgr.get_merged()

        # Same visibility — no change
        changed = mgr.update_visibility({(0, 0): 0})
        assert not changed

    def test_rate_limited_loading(self):
        zarr_path, _ = _make_scene_zarr()
        mgr = SceneMeshManager(zarr_path)
        mgr.per_tick_load_limit = 1

        # Request 3 chunks but limit is 1 per tick
        mgr.update_visibility({(0, 0): 0, (0, 1): 0, (1, 1): 0})
        # Some chunks may not be loaded yet
        assert len(mgr._chunk_cache) <= 2  # at most 1 new + 0 existing

    def test_distance_sorted_loading(self):
        zarr_path, _ = _make_scene_zarr()
        mgr = SceneMeshManager(zarr_path)
        mgr.per_tick_load_limit = 1

        # Chunk (1,1) is closer
        dists = {(0, 0): 1000.0, (1, 1): 10.0}
        mgr.update_visibility({(0, 0): 0, (1, 1): 0},
                              chunk_distances=dists)
        # Should load (1,1) first (closer)
        assert (1, 1) in mgr._chunk_cache

    def test_get_removed_gids(self):
        zarr_path, _ = _make_scene_zarr()
        mgr = SceneMeshManager(zarr_path)
        mgr.per_tick_load_limit = 10

        mgr.update_visibility({(0, 0): 0, (0, 1): 0})
        prev = set(mgr.active_gids)

        mgr.update_visibility({})
        removed = mgr.get_removed_gids(prev)
        assert 'building' in removed
        assert 'road' in removed

    def test_clear_caches(self):
        zarr_path, _ = _make_scene_zarr()
        mgr = SceneMeshManager(zarr_path)
        mgr.update_visibility({(0, 0): 0})
        assert len(mgr._chunk_cache) > 0
        mgr.clear_caches()
        assert len(mgr._chunk_cache) == 0
        assert len(mgr._visible_set) == 0

    def test_clear_active(self):
        zarr_path, _ = _make_scene_zarr()
        mgr = SceneMeshManager(zarr_path)
        mgr.update_visibility({(0, 0): 0})
        assert len(mgr._chunk_cache) > 0
        mgr.clear_active()
        # Cache preserved but active state cleared
        assert len(mgr._chunk_cache) > 0
        assert len(mgr._visible_set) == 0

    def test_get_stats(self):
        zarr_path, _ = _make_scene_zarr()
        mgr = SceneMeshManager(zarr_path)
        mgr.update_visibility({(0, 0): 0})
        stats = mgr.get_stats()
        assert 'Meshes:' in stats
        assert '1 chunks' in stats


class _FakeRTX:
    """Minimal RTX stub for tests."""
    def __init__(self):
        self.geometries = {}

    def add_geometry(self, gid, verts, indices, **kw):
        normals = kw.get('normals')
        self.geometries[gid] = (
            verts.copy(), indices.copy(),
            normals.copy() if normals is not None else None)
        return 0

    def add_heightfield_geometry(self, gid, elevation, H, W, **kw):
        self.geometries[gid] = {'type': 'heightfield'}
        return 0

    def remove_geometry(self, gid):
        self.geometries.pop(gid, None)
        return 0

    def has_geometry(self, gid):
        return gid in self.geometries


class TestLODManagerMeshIntegration:
    """Tests that TerrainLODManager drives SceneMeshManager."""

    @staticmethod
    def _make_terrain(h=256, w=256):
        rng = np.random.RandomState(42)
        return (rng.rand(h, w) * 100).astype(np.float32)

    def test_set_scene_zarr(self):
        from rtxpy.viewer.terrain_lod import TerrainLODManager
        zarr_path, _ = _make_scene_zarr()
        terrain = self._make_terrain()
        mgr = TerrainLODManager(terrain, tile_size=64)
        mgr.set_scene_zarr(zarr_path)
        assert mgr.scene_mesh_manager is not None
        assert 'building' in mgr.scene_mesh_manager.geometry_ids

    def test_detach_scene_zarr(self):
        from rtxpy.viewer.terrain_lod import TerrainLODManager
        zarr_path, _ = _make_scene_zarr()
        terrain = self._make_terrain()
        mgr = TerrainLODManager(terrain, tile_size=64)
        mgr.set_scene_zarr(zarr_path)
        mgr.set_scene_zarr(None)
        assert mgr.scene_mesh_manager is None

    def test_update_drives_mesh_visibility(self):
        from rtxpy.viewer.terrain_lod import TerrainLODManager

        zarr_path, _ = _make_scene_zarr()
        terrain = self._make_terrain()

        mgr = TerrainLODManager(terrain, tile_size=64)
        mgr.set_scene_zarr(zarr_path)
        mgr.per_tick_build_limit = 100
        mgr.scene_mesh_manager.per_tick_load_limit = 100

        rtx = _FakeRTX()
        cam = np.array([32, 32, 100])
        mgr.update(cam, rtx, force=True)

        # Mesh manager should have been updated with tile_lods
        smm = mgr.scene_mesh_manager
        assert len(smm._visible_set) > 0
        # Building chunk (0,0) should be visible (camera is near it)
        merged = smm.get_merged()
        assert 'building' in merged

    def test_with_chunk_source(self):
        from rtxpy.chunk_source import InMemoryChunkSource
        from rtxpy.viewer.terrain_lod import TerrainLODManager

        zarr_path, _ = _make_scene_zarr()
        terrain = self._make_terrain()
        src = InMemoryChunkSource(terrain, tile_size=64)

        mgr = TerrainLODManager(chunk_source=src)
        mgr.set_scene_zarr(zarr_path)
        mgr.per_tick_build_limit = 100
        mgr.scene_mesh_manager.per_tick_load_limit = 100

        rtx = _FakeRTX()
        cam = np.array([128, 128, 100])
        mgr.update(cam, rtx, force=True)

        smm = mgr.scene_mesh_manager
        assert len(smm._visible_set) > 0
