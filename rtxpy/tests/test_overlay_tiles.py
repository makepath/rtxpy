"""Tests for per-tile overlay compositing."""

import numpy as np
import pytest

from rtxpy.viewer.overlay_tiles import OverlayTileManager


class TestOverlayTileManager:
    """Tests for OverlayTileManager composite generation."""

    def test_empty_manager_returns_none(self):
        mgr = OverlayTileManager(tile_size=64)
        d, r, c = mgr.get_composite({(0, 0), (0, 1)})
        assert d is None
        assert r == 0
        assert c == 0

    def test_single_tile_composite(self):
        mgr = OverlayTileManager(tile_size=64)
        data = np.ones((64, 64), dtype=np.float32) * 5.0
        mgr.set_tile(0, 0, data)
        # get_composite returns None for GPU (no cupy in test), but
        # the internal _composite should be built
        mgr.get_composite({(0, 0)})
        assert mgr._composite is not None
        assert mgr._composite.shape == (64, 64)
        np.testing.assert_allclose(mgr._composite, 5.0)
        assert mgr._origin_row == 0
        assert mgr._origin_col == 0

    def test_multi_tile_composite_offsets(self):
        mgr = OverlayTileManager(tile_size=64)
        mgr.set_tile(1, 2, np.ones((64, 64), dtype=np.float32) * 1.0)
        mgr.set_tile(2, 3, np.ones((64, 64), dtype=np.float32) * 2.0)
        mgr.get_composite({(1, 2), (2, 3)})
        # Bounding box: rows 1-2, cols 2-3 → composite 128×128
        assert mgr._composite.shape == (128, 128)
        assert mgr._origin_row == 1 * 64
        assert mgr._origin_col == 2 * 64
        # Tile (1,2) at local (0,0), tile (2,3) at local (64,64)
        np.testing.assert_allclose(mgr._composite[0:64, 0:64], 1.0)
        np.testing.assert_allclose(mgr._composite[64:128, 64:128], 2.0)
        # Gaps should be NaN
        assert np.all(np.isnan(mgr._composite[0:64, 64:128]))

    def test_populate_from_array(self):
        mgr = OverlayTileManager(tile_size=64)
        overlay = np.arange(128 * 128, dtype=np.float32).reshape(128, 128)
        mgr.populate_from_array(overlay, 64, 2, 2)
        assert mgr.has_tile(0, 0)
        assert mgr.has_tile(0, 1)
        assert mgr.has_tile(1, 0)
        assert mgr.has_tile(1, 1)

    def test_populate_skips_all_nan_tiles(self):
        mgr = OverlayTileManager(tile_size=64)
        overlay = np.full((128, 128), np.nan, dtype=np.float32)
        # Only top-left has data
        overlay[0:32, 0:32] = 1.0
        mgr.populate_from_array(overlay, 64, 2, 2)
        assert mgr.has_tile(0, 0)
        assert not mgr.has_tile(0, 1)
        assert not mgr.has_tile(1, 0)
        assert not mgr.has_tile(1, 1)

    def test_remove_tile(self):
        mgr = OverlayTileManager(tile_size=64)
        mgr.set_tile(0, 0, np.ones((64, 64), dtype=np.float32))
        assert mgr.has_tile(0, 0)
        mgr.remove_tile(0, 0)
        assert not mgr.has_tile(0, 0)

    def test_invalidate_forces_recomposite(self):
        mgr = OverlayTileManager(tile_size=64)
        mgr.set_tile(0, 0, np.ones((64, 64), dtype=np.float32))
        mgr.get_composite({(0, 0)})
        comp1 = mgr._composite
        # Without invalidate, same tile set → cached
        mgr.get_composite({(0, 0)})
        # With invalidate, should rebuild
        mgr.invalidate()
        mgr.set_tile(0, 0, np.ones((64, 64), dtype=np.float32) * 99)
        # Reset throttle so rebuild happens immediately in tests
        mgr._last_rebuild = 0.0
        mgr.get_composite({(0, 0)})
        np.testing.assert_allclose(mgr._composite, 99.0)

    def test_visible_subset_only(self):
        """Only visible tiles with data are composited."""
        mgr = OverlayTileManager(tile_size=64)
        mgr.set_tile(0, 0, np.ones((64, 64), dtype=np.float32))
        mgr.set_tile(5, 5, np.ones((64, 64), dtype=np.float32) * 2)
        # Only request tile (0,0) as visible
        mgr.get_composite({(0, 0)})
        assert mgr._composite.shape == (64, 64)
        assert mgr._origin_row == 0

    def test_color_lut(self):
        mgr = OverlayTileManager(tile_size=64)
        lut = np.zeros((256, 3), dtype=np.float32)
        mgr.set_color_lut(lut)
        assert mgr.color_lut is lut


class TestTextureTileManager:
    """Tests for TextureTileManager RGB composite generation."""

    def test_empty_returns_none(self):
        from rtxpy.viewer.overlay_tiles import TextureTileManager
        mgr = TextureTileManager(tile_size=64)
        d, r, c = mgr.get_composite({(0, 0)})
        assert d is None

    def test_single_tile_composite(self):
        from rtxpy.viewer.overlay_tiles import TextureTileManager
        mgr = TextureTileManager(tile_size=64)
        data = np.ones((64, 64, 3), dtype=np.float32) * 0.5
        mgr.set_tile(0, 0, data)
        mgr.get_composite({(0, 0)})
        assert mgr._composite is not None
        assert mgr._composite.shape == (64, 64, 3)
        np.testing.assert_allclose(mgr._composite, 0.5)

    def test_multi_tile_offsets(self):
        from rtxpy.viewer.overlay_tiles import TextureTileManager
        mgr = TextureTileManager(tile_size=64)
        mgr.set_tile(1, 2, np.ones((64, 64, 3), dtype=np.float32) * 0.3)
        mgr.set_tile(2, 3, np.ones((64, 64, 3), dtype=np.float32) * 0.7)
        mgr.get_composite({(1, 2), (2, 3)})
        assert mgr._composite.shape == (128, 128, 3)
        assert mgr._origin_row == 64
        assert mgr._origin_col == 128
        np.testing.assert_allclose(mgr._composite[0:64, 0:64], 0.3)
        np.testing.assert_allclose(mgr._composite[64:128, 64:128], 0.7)
        # Gaps should be zero
        np.testing.assert_allclose(mgr._composite[0:64, 64:128], 0.0)

    def test_populate_from_array(self):
        from rtxpy.viewer.overlay_tiles import TextureTileManager
        mgr = TextureTileManager(tile_size=64)
        tex = np.ones((128, 128, 3), dtype=np.float32) * 0.5
        mgr.populate_from_array(tex, 64, 2, 2)
        assert mgr.has_tile(0, 0)
        assert mgr.has_tile(1, 1)

    def test_populate_skips_zero_tiles(self):
        from rtxpy.viewer.overlay_tiles import TextureTileManager
        mgr = TextureTileManager(tile_size=64)
        tex = np.zeros((128, 128, 3), dtype=np.float32)
        tex[0:32, 0:32] = 0.5  # Only top-left has data
        mgr.populate_from_array(tex, 64, 2, 2)
        assert mgr.has_tile(0, 0)
        assert not mgr.has_tile(0, 1)
        assert not mgr.has_tile(1, 0)
        assert not mgr.has_tile(1, 1)

    def test_remove_tile(self):
        from rtxpy.viewer.overlay_tiles import TextureTileManager
        mgr = TextureTileManager(tile_size=64)
        mgr.set_tile(0, 0, np.ones((64, 64, 3), dtype=np.float32))
        assert mgr.has_tile(0, 0)
        mgr.remove_tile(0, 0)
        assert not mgr.has_tile(0, 0)
