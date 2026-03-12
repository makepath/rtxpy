"""Per-tile overlay compositing for LOD terrain.

Manages overlay data (hydro stream_link, etc.) on a per-tile basis and
composites visible tiles into a single contiguous GPU array with pixel
offsets.  The render kernel indexes the composite using::

    ov_y = elev_y - overlay_offset_y
    ov_x = elev_x - overlay_offset_x

This keeps the Numba CUDA kernel simple (single 2D array + two int
offsets) while supporting unbounded tiled terrain.
"""

import threading
import time

import numpy as np


class OverlayTileManager:
    """Composites per-tile overlay arrays into a GPU-ready buffer.

    Parameters
    ----------
    tile_size : int
        Tile size in pixels (must match TerrainLODManager).
    """

    __slots__ = (
        '_tile_size',
        '_tile_overlays',   # {(tr, tc): np.ndarray}
        '_color_lut',       # np.ndarray (256, 3) float32 or None
        '_composite',       # np.ndarray (H, W) float32 or None
        '_d_composite',     # cupy ndarray or None
        '_origin_row',      # int: pixel row of composite[0,0]
        '_origin_col',      # int: pixel col of composite[0,0]
        '_dirty',           # bool
        '_composited_tiles', # frozenset of (tr, tc)
        '_last_rebuild',    # float: monotonic time of last rebuild
        '_lock',
    )

    _REBUILD_INTERVAL = 0.25

    def __init__(self, tile_size):
        self._tile_size = tile_size
        self._tile_overlays = {}
        self._color_lut = None
        self._composite = None
        self._d_composite = None
        self._origin_row = 0
        self._origin_col = 0
        self._dirty = True
        self._composited_tiles = frozenset()
        self._last_rebuild = 0.0
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Tile data management
    # ------------------------------------------------------------------

    def set_color_lut(self, lut):
        """Set the palette LUT for categorical overlay coloring."""
        self._color_lut = lut

    @property
    def color_lut(self):
        return self._color_lut

    def set_tile(self, tr, tc, data):
        """Store overlay data for tile (tr, tc).

        Parameters
        ----------
        data : np.ndarray, shape (th, tw)
            Overlay values (float32, NaN = transparent).
        """
        with self._lock:
            self._tile_overlays[(tr, tc)] = np.asarray(data, dtype=np.float32)
            self._dirty = True

    def remove_tile(self, tr, tc):
        """Remove overlay for tile (tr, tc)."""
        with self._lock:
            if (tr, tc) in self._tile_overlays:
                del self._tile_overlays[(tr, tc)]
                self._dirty = True

    def has_tile(self, tr, tc):
        return (tr, tc) in self._tile_overlays

    def clear(self):
        """Remove all tile overlays."""
        with self._lock:
            self._tile_overlays.clear()
            self._composite = None
            self._d_composite = None
            self._dirty = True

    def populate_from_array(self, overlay, tile_size, n_tile_rows, n_tile_cols):
        """Slice a monolithic overlay array into per-tile chunks.

        Parameters
        ----------
        overlay : np.ndarray, shape (H, W)
            Full-terrain overlay (e.g. from compute_from_terrain).
        tile_size : int
            Tile size in pixels.
        n_tile_rows, n_tile_cols : int
            Number of tile rows/columns in the initial terrain grid.
        """
        H, W = overlay.shape
        with self._lock:
            for tr in range(n_tile_rows):
                for tc in range(n_tile_cols):
                    r0 = tr * tile_size
                    c0 = tc * tile_size
                    r1 = min(r0 + tile_size, H)
                    c1 = min(c0 + tile_size, W)
                    tile_data = overlay[r0:r1, c0:c1]
                    # Skip all-NaN tiles to save memory
                    if np.all(np.isnan(tile_data)):
                        continue
                    self._tile_overlays[(tr, tc)] = tile_data.copy()
            self._dirty = True

    # ------------------------------------------------------------------
    # Composite for rendering
    # ------------------------------------------------------------------

    def get_composite(self, visible_tiles):
        """Return (gpu_array, offset_row, offset_col) for the render kernel.

        Parameters
        ----------
        visible_tiles : set of (tr, tc)
            Currently visible tile coordinates.

        Returns
        -------
        (d_composite, offset_row, offset_col) or (None, 0, 0)
            GPU overlay array and pixel offsets into elev_y/elev_x space.
        """
        # Filter to tiles that actually have overlay data
        with self._lock:
            tiles_with_data = visible_tiles & set(self._tile_overlays.keys())

        if not tiles_with_data:
            return None, 0, 0

        tiles_frozen = frozenset(tiles_with_data)

        # Rebuild composite only when tile set changed or data is dirty
        if not self._dirty and tiles_frozen == self._composited_tiles:
            if self._d_composite is not None:
                return self._d_composite, self._origin_row, self._origin_col

        # Superset: visible set contracted but data unchanged
        if (not self._dirty
                and tiles_frozen <= self._composited_tiles
                and self._d_composite is not None):
            return self._d_composite, self._origin_row, self._origin_col

        # Throttle rebuilds to avoid per-frame GPU uploads
        now = time.monotonic()
        if (self._dirty
                and self._d_composite is not None
                and (now - self._last_rebuild) < self._REBUILD_INTERVAL):
            return self._d_composite, self._origin_row, self._origin_col

        self._dirty = False
        self._composited_tiles = tiles_frozen
        self._last_rebuild = now

        ts = self._tile_size

        # Compute bounding box in tile coordinates
        min_tr = min(tr for tr, tc in tiles_with_data)
        max_tr = max(tr for tr, tc in tiles_with_data)
        min_tc = min(tc for tr, tc in tiles_with_data)
        max_tc = max(tc for tr, tc in tiles_with_data)

        # Pixel origin of the composite
        origin_row = min_tr * ts
        origin_col = min_tc * ts

        # Composite dimensions
        comp_h = (max_tr - min_tr + 1) * ts
        comp_w = (max_tc - min_tc + 1) * ts

        # Build composite (NaN = no data / transparent)
        composite = np.full((comp_h, comp_w), np.nan, dtype=np.float32)

        with self._lock:
            for tr, tc in tiles_with_data:
                tile_data = self._tile_overlays.get((tr, tc))
                if tile_data is None:
                    continue
                r0 = (tr - min_tr) * ts
                c0 = (tc - min_tc) * ts
                th, tw = tile_data.shape
                composite[r0:r0 + th, c0:c0 + tw] = tile_data

        self._composite = composite
        self._origin_row = origin_row
        self._origin_col = origin_col

        # Upload to GPU
        try:
            import cupy
            self._d_composite = cupy.asarray(composite)
        except ImportError:
            self._d_composite = None
            return None, 0, 0

        return self._d_composite, origin_row, origin_col

    def invalidate(self):
        """Force recomposite on next get_composite call."""
        self._dirty = True


class TextureTileManager:
    """Composites per-tile RGB textures into a GPU-ready buffer.

    Same composite-with-offset pattern as :class:`OverlayTileManager`
    but for ``(H, W, 3)`` float32 RGB data (basemap imagery).

    Parameters
    ----------
    tile_size : int
        Tile size in pixels (must match TerrainLODManager).
    """

    __slots__ = (
        '_tile_size',
        '_tile_textures',   # {(tr, tc): np.ndarray (th, tw, 3)}
        '_composite',       # np.ndarray (H, W, 3) float32 or None
        '_d_composite',     # cupy ndarray or None
        '_origin_row',      # int
        '_origin_col',      # int
        '_dirty',
        '_composited_tiles',
        '_last_rebuild',    # float: monotonic time of last rebuild
        '_lock',
    )

    # Minimum seconds between composite rebuilds.  Background fetch
    # threads calling set_tile() each mark dirty; without throttling
    # we'd rebuild+upload every frame while fetches stream in.
    _REBUILD_INTERVAL = 0.25

    def __init__(self, tile_size):
        self._tile_size = tile_size
        self._tile_textures = {}
        self._composite = None
        self._d_composite = None
        self._origin_row = 0
        self._origin_col = 0
        self._dirty = True
        self._composited_tiles = frozenset()
        self._last_rebuild = 0.0
        self._lock = threading.Lock()

    def set_tile(self, tr, tc, data):
        """Store RGB texture data for tile (tr, tc).

        Parameters
        ----------
        data : np.ndarray, shape (th, tw, 3)
            RGB values as float32 [0-1].
        """
        with self._lock:
            self._tile_textures[(tr, tc)] = np.asarray(data, dtype=np.float32)
            self._dirty = True

    def remove_tile(self, tr, tc):
        """Remove texture for tile (tr, tc)."""
        with self._lock:
            if (tr, tc) in self._tile_textures:
                del self._tile_textures[(tr, tc)]
                self._dirty = True

    def has_tile(self, tr, tc):
        return (tr, tc) in self._tile_textures

    def clear(self):
        """Remove all tile textures."""
        with self._lock:
            self._tile_textures.clear()
            self._composite = None
            self._d_composite = None
            self._dirty = True

    def populate_from_array(self, rgb_texture, tile_size, n_tile_rows, n_tile_cols):
        """Slice a monolithic RGB texture into per-tile chunks.

        Parameters
        ----------
        rgb_texture : np.ndarray, shape (H, W, 3)
            Full-terrain RGB texture (float32 [0-1]).
        tile_size : int
            Tile size in pixels.
        n_tile_rows, n_tile_cols : int
            Number of tile rows/columns.
        """
        H, W = rgb_texture.shape[:2]
        with self._lock:
            for tr in range(n_tile_rows):
                for tc in range(n_tile_cols):
                    r0 = tr * tile_size
                    c0 = tc * tile_size
                    r1 = min(r0 + tile_size, H)
                    c1 = min(c0 + tile_size, W)
                    tile_data = rgb_texture[r0:r1, c0:c1]
                    # Skip all-zero tiles (no texture data)
                    if np.all(tile_data == 0):
                        continue
                    self._tile_textures[(tr, tc)] = tile_data.copy()
            self._dirty = True

    def get_composite(self, visible_tiles):
        """Return (gpu_array, offset_row, offset_col) for the render kernel.

        Parameters
        ----------
        visible_tiles : set of (tr, tc)
            Currently visible tile coordinates.

        Returns
        -------
        (d_composite, offset_row, offset_col) or (None, 0, 0)
            GPU RGB array ``(H, W, 3)`` and pixel offsets.
        """
        with self._lock:
            tiles_with_data = visible_tiles & set(self._tile_textures.keys())

        if not tiles_with_data:
            return None, 0, 0

        tiles_frozen = frozenset(tiles_with_data)

        # Fast path: nothing changed at all.
        if not self._dirty and tiles_frozen == self._composited_tiles:
            if self._d_composite is not None:
                return self._d_composite, self._origin_row, self._origin_col

        # Superset path: visible set contracted but data unchanged —
        # existing composite already covers all needed tiles.
        if (not self._dirty
                and tiles_frozen <= self._composited_tiles
                and self._d_composite is not None):
            return self._d_composite, self._origin_row, self._origin_col

        # Throttle: don't rebuild more than once per _REBUILD_INTERVAL.
        # Background fetch threads mark dirty in bursts; without this
        # we'd rebuild+upload every frame while fetches stream in.
        now = time.monotonic()
        if (self._dirty
                and self._d_composite is not None
                and (now - self._last_rebuild) < self._REBUILD_INTERVAL):
            return self._d_composite, self._origin_row, self._origin_col

        self._dirty = False
        self._composited_tiles = tiles_frozen
        self._last_rebuild = now

        ts = self._tile_size

        min_tr = min(tr for tr, tc in tiles_with_data)
        max_tr = max(tr for tr, tc in tiles_with_data)
        min_tc = min(tc for tr, tc in tiles_with_data)
        max_tc = max(tc for tr, tc in tiles_with_data)

        origin_row = min_tr * ts
        origin_col = min_tc * ts

        comp_h = (max_tr - min_tr + 1) * ts
        comp_w = (max_tc - min_tc + 1) * ts

        # Build composite (zeros = no texture)
        composite = np.zeros((comp_h, comp_w, 3), dtype=np.float32)

        with self._lock:
            for tr, tc in tiles_with_data:
                tile_data = self._tile_textures.get((tr, tc))
                if tile_data is None:
                    continue
                r0 = (tr - min_tr) * ts
                c0 = (tc - min_tc) * ts
                th, tw = tile_data.shape[:2]
                composite[r0:r0 + th, c0:c0 + tw] = tile_data

        self._composite = composite
        self._origin_row = origin_row
        self._origin_col = origin_col

        try:
            import cupy
            self._d_composite = cupy.asarray(composite)
        except ImportError:
            self._d_composite = None
            return None, 0, 0

        return self._d_composite, origin_row, origin_col

    def invalidate(self):
        """Force recomposite on next get_composite call."""
        self._dirty = True
