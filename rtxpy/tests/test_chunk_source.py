"""Tests for the ChunkDataSource abstraction."""

import math
import os
import tempfile

import numpy as np
import pytest

from rtxpy.chunk_source import (
    ChunkDataSource,
    InMemoryChunkSource,
    ZarrChunkSource,
    chunk_source_from_raster,
    chunk_source_from_zarr,
)
from rtxpy.lod import box_downsample_2x as _box_downsample_2x


# ======================================================================
# Helpers
# ======================================================================

def _make_terrain(h=256, w=256, seed=42):
    """Generate a synthetic terrain with hills and a few NaN pixels."""
    rng = np.random.RandomState(seed)
    ys = np.linspace(0, 4 * np.pi, h, dtype=np.float32)
    xs = np.linspace(0, 4 * np.pi, w, dtype=np.float32)
    terrain = (np.sin(ys[:, None]) * np.cos(xs[None, :]) * 500
               + 1000).astype(np.float32)
    # Sprinkle a few NaN pixels (simulating ocean / no-data)
    terrain[0, 0] = np.nan
    terrain[-1, -1] = np.nan
    return terrain


def _make_zarr_store(terrain, chunk_size=64, cf_encode=False):
    """Write terrain to a temporary zarr store and return the path."""
    import zarr

    tmpdir = tempfile.mkdtemp()
    zarr_path = os.path.join(tmpdir, 'test.zarr')
    root = zarr.open_group(zarr_path, mode='w')

    if cf_encode:
        # Store as int32 centimeters with CF encoding
        scale = 0.01
        offset = 0.0
        fill_value = -999999
        encoded = terrain.copy()
        nan_mask = np.isnan(encoded)
        encoded[nan_mask] = 0
        encoded = (encoded / scale).astype(np.int32)
        encoded[nan_mask] = fill_value
        arr = root.create_array(
            'elevation',
            data=encoded,
            chunks=(chunk_size, chunk_size),
            fill_value=fill_value,
            overwrite=True,
        )
        arr.attrs['scale_factor'] = scale
        arr.attrs['add_offset'] = offset
    else:
        root.create_array(
            'elevation',
            data=terrain,
            chunks=(chunk_size, chunk_size),
            overwrite=True,
        )

    return zarr_path


# ======================================================================
# _box_downsample_2x
# ======================================================================

class TestBoxDownsample:
    def test_basic_shape(self):
        arr = np.ones((10, 8), dtype=np.float32)
        out = _box_downsample_2x(arr)
        assert out.shape == (5, 4)

    def test_odd_dimensions(self):
        arr = np.ones((11, 9), dtype=np.float32)
        out = _box_downsample_2x(arr)
        # Trims to even: (10, 8) -> (5, 4)
        assert out.shape == (5, 4)

    def test_values(self):
        arr = np.array([[1, 2, 3, 4],
                        [5, 6, 7, 8]], dtype=np.float32)
        out = _box_downsample_2x(arr)
        assert out.shape == (1, 2)
        np.testing.assert_allclose(out[0, 0], 3.5)   # mean(1,2,5,6)
        np.testing.assert_allclose(out[0, 1], 5.5)   # mean(3,4,7,8)

    def test_nan_aware(self):
        arr = np.array([[np.nan, 2, 3, 4],
                        [5, 6, 7, 8]], dtype=np.float32)
        out = _box_downsample_2x(arr)
        # mean of (nan, 2, 5, 6) = mean(2, 5, 6) = 4.333...
        np.testing.assert_allclose(out[0, 0], (2 + 5 + 6) / 3, rtol=1e-5)

    def test_tiny_array(self):
        arr = np.ones((1, 4), dtype=np.float32)
        out = _box_downsample_2x(arr)
        # Too small to downsample, returns copy
        assert out.shape == (1, 4)


# ======================================================================
# InMemoryChunkSource
# ======================================================================

class TestInMemoryChunkSource:
    def test_single_chunk_default(self):
        terrain = _make_terrain(100, 100)
        src = InMemoryChunkSource(terrain)
        assert src.shape == (100, 100)
        assert src.chunk_shape == (100, 100)
        assert src.n_chunk_rows == 1
        assert src.n_chunk_cols == 1

    def test_grid_dimensions(self):
        terrain = _make_terrain(256, 256)
        src = InMemoryChunkSource(terrain, tile_size=64)
        assert src.n_chunk_rows == 4
        assert src.n_chunk_cols == 4

    def test_grid_dimensions_non_divisible(self):
        terrain = _make_terrain(200, 150)
        src = InMemoryChunkSource(terrain, tile_size=64)
        # ceil(200/64)=4, ceil(150/64)=3
        assert src.n_chunk_rows == 4
        assert src.n_chunk_cols == 3

    def test_pixel_spacing(self):
        terrain = _make_terrain(100, 100)
        src = InMemoryChunkSource(terrain, pixel_spacing_x=10.0,
                                  pixel_spacing_y=10.0)
        assert src.pixel_spacing == (10.0, 10.0)

    def test_read_chunk_lod0(self):
        terrain = _make_terrain(256, 256)
        src = InMemoryChunkSource(terrain, tile_size=64)

        tile = src.read_chunk(0, 0, lod=0)
        assert tile is not None
        assert tile.dtype == np.float32
        # LOD 0, tile_size=64 -> should be ~64x64 (+ overlap for shared edge)
        assert tile.shape[0] >= 2
        assert tile.shape[1] >= 2

    def test_read_chunk_lod1_smaller(self):
        terrain = _make_terrain(256, 256)
        src = InMemoryChunkSource(terrain, tile_size=64)

        lod0 = src.read_chunk(0, 0, lod=0)
        lod1 = src.read_chunk(0, 0, lod=1)
        assert lod1 is not None
        # LOD 1 should be roughly half the size of LOD 0
        assert lod1.shape[0] < lod0.shape[0]
        assert lod1.shape[1] < lod0.shape[1]

    def test_read_chunk_lod2_even_smaller(self):
        terrain = _make_terrain(256, 256)
        src = InMemoryChunkSource(terrain, tile_size=128)

        lod0 = src.read_chunk(0, 0, lod=0)
        lod2 = src.read_chunk(0, 0, lod=2)
        assert lod2 is not None
        assert lod2.shape[0] < lod0.shape[0]

    def test_read_chunk_out_of_bounds(self):
        terrain = _make_terrain(100, 100)
        src = InMemoryChunkSource(terrain, tile_size=64)
        assert src.read_chunk(10, 10, lod=0) is None

    def test_read_full_res_chunk(self):
        terrain = _make_terrain(128, 128)
        src = InMemoryChunkSource(terrain, tile_size=64)

        full = src.read_full_res_chunk(0, 0)
        assert full is not None
        assert full.shape == (64, 64)
        np.testing.assert_array_equal(full, terrain[:64, :64])

    def test_read_full_res_edge_chunk(self):
        terrain = _make_terrain(100, 100)
        src = InMemoryChunkSource(terrain, tile_size=64)

        # Last chunk is partial: 100-64 = 36 pixels
        full = src.read_full_res_chunk(1, 1)
        assert full is not None
        assert full.shape == (36, 36)

    def test_is_in_bounds(self):
        terrain = _make_terrain(128, 128)
        src = InMemoryChunkSource(terrain, tile_size=64)
        assert src.is_in_bounds(0, 0)
        assert src.is_in_bounds(1, 1)
        assert not src.is_in_bounds(2, 0)
        assert not src.is_in_bounds(-1, 0)

    def test_supports_streaming_false(self):
        terrain = _make_terrain(100, 100)
        src = InMemoryChunkSource(terrain)
        assert not src.supports_streaming

    def test_elevation_stats(self):
        terrain = _make_terrain(100, 100)
        src = InMemoryChunkSource(terrain)
        stats = src.elevation_stats()
        assert stats is not None
        emin, emax, emean = stats
        assert emin == pytest.approx(float(np.nanmin(terrain)), abs=0.1)
        assert emax == pytest.approx(float(np.nanmax(terrain)), abs=0.1)

    def test_chunk_roughness(self):
        terrain = _make_terrain(128, 128)
        src = InMemoryChunkSource(terrain, tile_size=64)
        r = src.chunk_roughness(0, 0)
        assert isinstance(r, float)
        assert r >= 0.0

    def test_roughness_flat_terrain(self):
        flat = np.ones((100, 100), dtype=np.float32) * 500.0
        src = InMemoryChunkSource(flat, tile_size=64)
        r = src.chunk_roughness(0, 0)
        assert r == pytest.approx(0.0, abs=1e-4)

    def test_crs_transform(self):
        terrain = _make_terrain(100, 100)
        src = InMemoryChunkSource(terrain, crs_origin=(500000.0, 4000000.0),
                                  crs_pixel_spacing=(10.0, -10.0))
        ct = src.crs_transform
        assert ct == (500000.0, 4000000.0, 10.0, -10.0)

    def test_crs_transform_none(self):
        terrain = _make_terrain(100, 100)
        src = InMemoryChunkSource(terrain)
        assert src.crs_transform is None

    def test_clear_caches(self):
        terrain = _make_terrain(128, 128)
        src = InMemoryChunkSource(terrain, tile_size=64)
        # Read to populate pyramid cache
        src.read_chunk(0, 0, lod=1)
        assert len(src._pyramid_cache) > 0
        src.clear_caches()
        assert len(src._pyramid_cache) == 0

    def test_set_terrain(self):
        terrain = _make_terrain(128, 128)
        src = InMemoryChunkSource(terrain, tile_size=64)
        assert src.n_chunk_rows == 2
        assert src.n_chunk_cols == 2

        new_terrain = _make_terrain(256, 256)
        src.set_terrain(new_terrain)
        assert src.shape == (256, 256)
        assert src.n_chunk_rows == 4
        assert src.n_chunk_cols == 4

    def test_base_subsample(self):
        terrain = _make_terrain(256, 256)
        src = InMemoryChunkSource(terrain, tile_size=128)

        lod0_bs1 = src.read_chunk(0, 0, lod=0, base_subsample=1)
        lod0_bs2 = src.read_chunk(0, 0, lod=0, base_subsample=2)
        assert lod0_bs1 is not None
        assert lod0_bs2 is not None
        # base_subsample=2 should produce roughly half the samples
        assert lod0_bs2.shape[0] < lod0_bs1.shape[0]

    def test_factory_function(self):
        terrain = _make_terrain(100, 100)
        src = chunk_source_from_raster(terrain, tile_size=50,
                                       pixel_spacing_x=10.0,
                                       pixel_spacing_y=10.0)
        assert isinstance(src, InMemoryChunkSource)
        assert src.pixel_spacing == (10.0, 10.0)


# ======================================================================
# ZarrChunkSource
# ======================================================================

class TestZarrChunkSource:
    @pytest.fixture
    def terrain_and_zarr(self, tmp_path):
        """Create a terrain and matching zarr store."""
        terrain = _make_terrain(256, 256)
        zarr_path = _make_zarr_store(terrain, chunk_size=64)
        return terrain, zarr_path

    @pytest.fixture
    def cf_encoded_zarr(self, tmp_path):
        """Create a CF-encoded zarr store."""
        terrain = _make_terrain(256, 256)
        zarr_path = _make_zarr_store(terrain, chunk_size=64, cf_encode=True)
        return terrain, zarr_path

    def test_grid_dimensions(self, terrain_and_zarr):
        terrain, zarr_path = terrain_and_zarr
        src = ZarrChunkSource(zarr_path, 'elevation')
        assert src.shape == (256, 256)
        assert src.chunk_shape == (64, 64)
        assert src.n_chunk_rows == 4
        assert src.n_chunk_cols == 4

    def test_read_chunk_lod0(self, terrain_and_zarr):
        terrain, zarr_path = terrain_and_zarr
        src = ZarrChunkSource(zarr_path, 'elevation')

        tile = src.read_chunk(0, 0, lod=0)
        assert tile is not None
        assert tile.dtype == np.float32
        # Should match the raw terrain data
        expected = terrain[:64, :64]
        np.testing.assert_allclose(tile, expected, rtol=1e-5)

    def test_read_chunk_lod1(self, terrain_and_zarr):
        terrain, zarr_path = terrain_and_zarr
        src = ZarrChunkSource(zarr_path, 'elevation')

        lod0 = src.read_chunk(0, 0, lod=0)
        lod1 = src.read_chunk(0, 0, lod=1)
        assert lod1 is not None
        assert lod1.shape[0] < lod0.shape[0]

    def test_read_full_res(self, terrain_and_zarr):
        terrain, zarr_path = terrain_and_zarr
        src = ZarrChunkSource(zarr_path, 'elevation')

        full = src.read_full_res_chunk(1, 2)
        assert full is not None
        expected = terrain[64:128, 128:192]
        np.testing.assert_allclose(full, expected, rtol=1e-5)

    def test_out_of_bounds(self, terrain_and_zarr):
        _, zarr_path = terrain_and_zarr
        src = ZarrChunkSource(zarr_path, 'elevation')
        assert src.read_chunk(10, 10, lod=0) is None

    def test_supports_streaming(self, terrain_and_zarr):
        _, zarr_path = terrain_and_zarr
        src = ZarrChunkSource(zarr_path, 'elevation')
        assert src.supports_streaming

    def test_cf_decode(self, cf_encoded_zarr):
        terrain, zarr_path = cf_encoded_zarr
        src = ZarrChunkSource(zarr_path, 'elevation')

        tile = src.read_chunk(1, 1, lod=0)
        assert tile is not None
        expected = terrain[64:128, 64:128]
        # CF decode introduces small rounding (int32 centimeters)
        np.testing.assert_allclose(tile, expected, atol=0.01)

    def test_elevation_stats(self, terrain_and_zarr):
        terrain, zarr_path = terrain_and_zarr
        src = ZarrChunkSource(zarr_path, 'elevation')

        stats = src.elevation_stats()
        assert stats is not None
        emin, emax, emean = stats
        # Stats from strided read should be approximately correct
        assert emin < emax
        assert emin == pytest.approx(float(np.nanmin(terrain)), rel=0.1)
        assert emax == pytest.approx(float(np.nanmax(terrain)), rel=0.1)

    def test_crs_transform(self, terrain_and_zarr):
        _, zarr_path = terrain_and_zarr
        src = ZarrChunkSource(zarr_path, 'elevation',
                              crs_origin=(500000.0, 4000000.0),
                              crs_pixel_spacing=(10.0, -10.0))
        ct = src.crs_transform
        assert ct == (500000.0, 4000000.0, 10.0, -10.0)

    def test_cache_and_clear(self, terrain_and_zarr):
        _, zarr_path = terrain_and_zarr
        src = ZarrChunkSource(zarr_path, 'elevation')

        src.read_chunk(0, 0, lod=0)
        assert len(src._tile_cache) > 0
        src.clear_caches()
        assert len(src._tile_cache) == 0

    def test_chunk_roughness(self, terrain_and_zarr):
        _, zarr_path = terrain_and_zarr
        src = ZarrChunkSource(zarr_path, 'elevation')
        r = src.chunk_roughness(0, 0)
        assert isinstance(r, float)
        assert r >= 0.0

    def test_factory_function(self, terrain_and_zarr):
        _, zarr_path = terrain_and_zarr
        src = chunk_source_from_zarr(zarr_path, 'elevation',
                                     pixel_spacing_x=10.0,
                                     pixel_spacing_y=10.0)
        assert isinstance(src, ZarrChunkSource)
        assert src.pixel_spacing == (10.0, 10.0)


# ======================================================================
# Cross-source consistency
# ======================================================================

class TestCrossSourceConsistency:
    """Verify that InMemoryChunkSource and ZarrChunkSource produce
    equivalent results from the same underlying data."""

    def test_lod0_match(self):
        terrain = _make_terrain(128, 128)
        zarr_path = _make_zarr_store(terrain, chunk_size=64)

        mem_src = InMemoryChunkSource(terrain, tile_size=64)
        zarr_src = ZarrChunkSource(zarr_path, 'elevation')

        for cr in range(2):
            for cc in range(2):
                mem_tile = mem_src.read_chunk(cr, cc, lod=0)
                zarr_tile = zarr_src.read_chunk(cr, cc, lod=0)
                assert mem_tile is not None
                assert zarr_tile is not None
                np.testing.assert_allclose(
                    mem_tile, zarr_tile, rtol=1e-5,
                    err_msg=f'LOD 0 mismatch at chunk ({cr}, {cc})')

    def test_full_res_match(self):
        terrain = _make_terrain(128, 128)
        zarr_path = _make_zarr_store(terrain, chunk_size=64)

        mem_src = InMemoryChunkSource(terrain, tile_size=64)
        zarr_src = ZarrChunkSource(zarr_path, 'elevation')

        for cr in range(2):
            for cc in range(2):
                mem_full = mem_src.read_full_res_chunk(cr, cc)
                zarr_full = zarr_src.read_full_res_chunk(cr, cc)
                assert mem_full is not None
                assert zarr_full is not None
                np.testing.assert_allclose(
                    mem_full, zarr_full, rtol=1e-5,
                    err_msg=f'Full-res mismatch at chunk ({cr}, {cc})')

    def test_roughness_match(self):
        terrain = _make_terrain(128, 128)
        zarr_path = _make_zarr_store(terrain, chunk_size=64)

        mem_src = InMemoryChunkSource(terrain, tile_size=64)
        zarr_src = ZarrChunkSource(zarr_path, 'elevation')

        for cr in range(2):
            for cc in range(2):
                mem_r = mem_src.chunk_roughness(cr, cc)
                zarr_r = zarr_src.chunk_roughness(cr, cc)
                assert mem_r == pytest.approx(zarr_r, abs=1e-4), \
                    f'Roughness mismatch at ({cr}, {cc})'
