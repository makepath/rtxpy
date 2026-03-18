"""Chunk data source abstraction for terrain elevation.

Provides a uniform interface for reading tiled terrain elevation data,
whether the backing store is an in-memory numpy array or a zarr store
on disk (or remote).  The :class:`TerrainLODManager` uses this interface
instead of accessing terrain arrays directly, enabling zarr-chunk-driven
visualization without loading the full raster into memory.

Two concrete implementations are provided:

- :class:`InMemoryChunkSource` — wraps a numpy ``(H, W)`` array, slicing
  it into tiles and building a lazy box-filter pyramid for LOD levels.
  This is the degenerate "single raster" case used when the user calls
  ``dem.rtx.explore()`` with a plain DataArray.

- :class:`ZarrChunkSource` — reads elevation from a zarr store, using the
  zarr array's own chunk grid as the tile grid.  Supports pre-built LOD
  arrays (``{name}_lod2``, ``{name}_lod4``, …) for fast coarse reads, and
  falls back to reading full-resolution data and box-filtering in memory
  when pre-built arrays are absent.
"""

import logging
import math
from abc import ABC, abstractmethod

import numpy as np

from .lod import box_downsample_2x as _box_downsample_2x

logger = logging.getLogger(__name__)


# ======================================================================
# Abstract base
# ======================================================================

class ChunkDataSource(ABC):
    """Abstract interface for tiled terrain elevation data.

    A chunk source divides a terrain raster into a regular grid of
    rectangular chunks (tiles).  The LOD manager requests elevation data
    per chunk at a given LOD level, and the source handles all I/O and
    downsampling internally.

    Terminology
    -----------
    - *chunk* and *tile* are used interchangeably — one chunk = one LOD
      tile.
    - ``(cr, cc)`` is the chunk row/column index (0-based).
    - ``lod`` is the LOD level (0 = full resolution, *k* = 2^k
      subsampling relative to LOD 0).
    """

    # ------------------------------------------------------------------
    # Grid geometry (must be implemented)
    # ------------------------------------------------------------------

    @property
    @abstractmethod
    def shape(self):
        """Full-resolution terrain shape ``(H, W)`` in pixels."""

    @property
    @abstractmethod
    def chunk_shape(self):
        """Chunk (tile) shape ``(chunk_h, chunk_w)`` in full-res pixels."""

    @property
    @abstractmethod
    def pixel_spacing(self):
        """World-space pixel spacing ``(psx, psy)``."""

    @property
    def n_chunk_rows(self):
        """Number of chunk rows in the grid."""
        H, _ = self.shape
        ch, _ = self.chunk_shape
        return (H + ch - 1) // ch

    @property
    def n_chunk_cols(self):
        """Number of chunk columns in the grid."""
        _, W = self.shape
        _, cw = self.chunk_shape
        return (W + cw - 1) // cw

    # ------------------------------------------------------------------
    # CRS metadata (optional)
    # ------------------------------------------------------------------

    @property
    def crs_transform(self):
        """CRS origin and signed pixel spacing.

        Returns ``(origin_x, origin_y, dx, dy)`` or ``None``.
        When set, the LOD manager uses these to convert pixel indices to
        CRS coordinates (e.g. UTM easting/northing) for streaming tile
        callbacks.
        """
        return None

    # ------------------------------------------------------------------
    # Data access (must be implemented)
    # ------------------------------------------------------------------

    @abstractmethod
    def read_chunk(self, cr, cc, lod, base_subsample=1):
        """Read elevation for chunk ``(cr, cc)`` at LOD level ``lod``.

        Parameters
        ----------
        cr, cc : int
            Chunk row and column (0-based).
        lod : int
            LOD level.  0 = full resolution (after ``base_subsample``),
            *k* = additional 2^k subsampling.
        base_subsample : int
            Global base subsample factor (from R/Shift+R in viewer).
            LOD 0 returns data at this stride; LOD *k* returns data at
            ``base_subsample * 2^k`` stride.

        Returns
        -------
        np.ndarray or None
            Float32 2D elevation array, or ``None`` if the chunk is
            empty or outside the data extent.
        """

    def read_full_res_chunk(self, cr, cc):
        """Read full-resolution elevation for chunk ``(cr, cc)``.

        Used for roughness computation and edge stitching where the
        finest available data is needed regardless of the current LOD.
        Default implementation delegates to ``read_chunk(cr, cc, 0, 1)``.
        """
        return self.read_chunk(cr, cc, lod=0, base_subsample=1)

    # ------------------------------------------------------------------
    # Streaming / unbounded grid
    # ------------------------------------------------------------------

    def can_stream(self, cr, cc):
        """Return ``True`` if chunk ``(cr, cc)`` can be fetched on demand.

        For in-memory sources, returns ``True`` only for in-bounds chunks.
        For zarr sources with streaming capability, returns ``True`` for
        any coordinate within the data extent.
        """
        return self.is_in_bounds(cr, cc)

    def is_in_bounds(self, cr, cc):
        """Return ``True`` if ``(cr, cc)`` is within the initial grid."""
        return 0 <= cr < self.n_chunk_rows and 0 <= cc < self.n_chunk_cols

    @property
    def supports_streaming(self):
        """Whether this source can provide data beyond the initial grid.

        When ``True``, the LOD manager treats the tile grid as unbounded
        and calls :meth:`read_chunk` for tiles outside
        ``n_chunk_rows × n_chunk_cols``.
        """
        return False

    # ------------------------------------------------------------------
    # Roughness (optional override)
    # ------------------------------------------------------------------

    def chunk_roughness(self, cr, cc):
        """Compute terrain roughness for chunk ``(cr, cc)``.

        Returns the standard deviation of elevation residuals from a
        bilinear corner fit (see :func:`~rtxpy.lod.compute_tile_roughness`).
        Default implementation reads the full-res chunk and computes.
        """
        from .lod import compute_tile_roughness

        tile = self.read_full_res_chunk(cr, cc)
        if tile is None:
            return 0.0
        return compute_tile_roughness(tile)

    # ------------------------------------------------------------------
    # Elevation stats (optional override)
    # ------------------------------------------------------------------

    def elevation_stats(self):
        """Return ``(elev_min, elev_max, elev_mean)`` for the full terrain.

        Default implementation returns ``None`` — the caller must compute
        stats from an alternative source.  Subclasses should override
        when stats can be cheaply obtained (e.g. from zarr metadata or
        a coarse global read).
        """
        return None

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def close(self):
        """Release any resources held by the source."""

    def clear_caches(self):
        """Clear any internal caches (e.g. pyramid levels)."""


# ======================================================================
# In-memory implementation
# ======================================================================

class InMemoryChunkSource(ChunkDataSource):
    """Chunk source backed by an in-memory numpy array.

    Tiles are slices into the array.  LOD levels are produced by a lazy
    box-filter pyramid — each level halves the resolution of the previous
    one via 2×2 NaN-aware averaging.  No data is copied until a specific
    LOD level is first requested.

    Parameters
    ----------
    terrain_np : np.ndarray
        Full-resolution elevation array, shape ``(H, W)``, float32.
    tile_size : int
        Tile edge length in full-resolution pixels.  If ``None``, the
        entire array is treated as a single chunk.
    pixel_spacing_x, pixel_spacing_y : float
        World-space size of one raster pixel.
    max_lod : int
        Maximum number of pyramid levels to cache.
    crs_origin : tuple of (float, float) or None
        ``(origin_x, origin_y)`` in CRS coordinates.
    crs_pixel_spacing : tuple of (float, float) or None
        ``(dx, dy)`` signed CRS spacing per pixel.
    """

    __slots__ = (
        '_terrain_np', '_tile_size', '_psx', '_psy',
        '_max_lod', '_pyramid_cache',
        '_H', '_W', '_n_cr', '_n_cc',
        '_crs_origin', '_crs_pixel_spacing',
    )

    def __init__(self, terrain_np, tile_size=None,
                 pixel_spacing_x=1.0, pixel_spacing_y=1.0,
                 max_lod=3,
                 crs_origin=None, crs_pixel_spacing=None):
        H, W = terrain_np.shape
        self._terrain_np = terrain_np
        self._H = H
        self._W = W

        # Default: single chunk covering the whole array
        if tile_size is None:
            tile_size = max(H, W)
        self._tile_size = tile_size

        self._psx = pixel_spacing_x
        self._psy = pixel_spacing_y
        self._max_lod = max_lod

        self._n_cr = (H + tile_size - 1) // tile_size
        self._n_cc = (W + tile_size - 1) // tile_size

        # Lazy pyramid: level -> downsampled array
        self._pyramid_cache = {}

        # CRS metadata
        self._crs_origin = crs_origin
        self._crs_pixel_spacing = crs_pixel_spacing

    # -- Grid geometry --------------------------------------------------

    @property
    def shape(self):
        return (self._H, self._W)

    @property
    def chunk_shape(self):
        return (self._tile_size, self._tile_size)

    @property
    def pixel_spacing(self):
        return (self._psx, self._psy)

    @property
    def n_chunk_rows(self):
        return self._n_cr

    @property
    def n_chunk_cols(self):
        return self._n_cc

    @property
    def crs_transform(self):
        if self._crs_origin is not None and self._crs_pixel_spacing is not None:
            ox, oy = self._crs_origin
            dx, dy = self._crs_pixel_spacing
            return (ox, oy, dx, dy)
        return None

    # -- Pyramid --------------------------------------------------------

    def _get_pyramid_level(self, level):
        """Return the downsampled terrain for *level*, building lazily.

        Level 0 is the full-resolution terrain (or base-subsampled).
        Each subsequent level halves via 2×2 NaN-aware box averaging.
        """
        cached = self._pyramid_cache.get(level)
        if cached is not None:
            return cached

        if level == 0:
            arr = self._terrain_np
        else:
            prev = self._get_pyramid_level(level - 1)
            arr = _box_downsample_2x(prev)

        self._pyramid_cache[level] = arr

        # Evict levels beyond max_lod to bound memory
        for stale in [k for k in self._pyramid_cache if k > self._max_lod]:
            del self._pyramid_cache[stale]

        return arr

    def _get_pyramid_level_subsampled(self, level, base_subsample):
        """Get pyramid data accounting for base_subsample.

        When base_subsample > 1, level 0 is the strided terrain and
        subsequent levels downsample from there.
        """
        if base_subsample <= 1:
            return self._get_pyramid_level(level)

        # For base_subsample > 1, we need a separate cache key approach.
        # Build from strided base on the fly — this is cheap since the
        # terrain is already in memory.
        key = (level, base_subsample)
        cached = self._pyramid_cache.get(key)
        if cached is not None:
            return cached

        if level == 0:
            arr = self._terrain_np[::base_subsample, ::base_subsample].copy()
        else:
            prev = self._get_pyramid_level_subsampled(level - 1, base_subsample)
            arr = _box_downsample_2x(prev)

        self._pyramid_cache[key] = arr
        return arr

    # -- Data access ----------------------------------------------------

    def read_chunk(self, cr, cc, lod, base_subsample=1):
        if not self.is_in_bounds(cr, cc):
            return None

        ts = self._tile_size
        pyr_level = min(lod, self._max_lod)

        if base_subsample > 1:
            pyr_arr = self._get_pyramid_level_subsampled(pyr_level,
                                                         base_subsample)
            pyr_sub = base_subsample * (2 ** pyr_level)
        else:
            pyr_arr = self._get_pyramid_level(pyr_level)
            pyr_sub = 2 ** pyr_level

        pH, pW = pyr_arr.shape

        # Map full-res tile pixel coords to pyramid pixel coords.
        r0_full = cr * ts
        c0_full = cc * ts
        r1_full = min(r0_full + ts, self._H)
        c1_full = min(c0_full + ts, self._W)

        r0_pyr = r0_full // pyr_sub
        c0_pyr = c0_full // pyr_sub
        r1_pyr = min((r1_full + pyr_sub - 1) // pyr_sub, pH)
        c1_pyr = min((c1_full + pyr_sub - 1) // pyr_sub, pW)

        # Extra subsampling if lod > max precomputed level
        subsample = base_subsample * (2 ** lod) if base_subsample > 1 else 2 ** lod
        extra_sub = max(1, subsample // pyr_sub)

        tile = pyr_arr[r0_pyr:r1_pyr:extra_sub, c0_pyr:c1_pyr:extra_sub]
        if tile.shape[0] < 2 or tile.shape[1] < 2:
            return None

        return np.asarray(tile, dtype=np.float32)

    def read_full_res_chunk(self, cr, cc):
        if not self.is_in_bounds(cr, cc):
            return None
        ts = self._tile_size
        r0 = cr * ts
        c0 = cc * ts
        r1 = min(r0 + ts, self._H)
        c1 = min(c0 + ts, self._W)
        return np.asarray(self._terrain_np[r0:r1, c0:c1], dtype=np.float32)

    # -- Stats ----------------------------------------------------------

    def elevation_stats(self):
        arr = self._terrain_np
        emin = float(np.nanmin(arr))
        emax = float(np.nanmax(arr))
        emean = float(np.nanmean(arr))
        return (emin, emax, emean)

    # -- Cache management -----------------------------------------------

    def clear_caches(self):
        self._pyramid_cache.clear()

    def set_terrain(self, terrain_np, tile_size=None):
        """Replace the backing terrain array.

        Clears all caches and recomputes grid dimensions.
        """
        H, W = terrain_np.shape
        self._terrain_np = terrain_np
        self._H = H
        self._W = W
        if tile_size is not None:
            self._tile_size = tile_size
        ts = self._tile_size
        self._n_cr = (H + ts - 1) // ts
        self._n_cc = (W + ts - 1) // ts
        self._pyramid_cache.clear()


# ======================================================================
# Zarr implementation
# ======================================================================

class ZarrChunkSource(ChunkDataSource):
    """Chunk source backed by a zarr store.

    Uses the zarr array's chunk grid as the tile grid.  Supports
    pre-built LOD arrays (``{array_name}_lod{factor}``) for fast coarse
    reads; falls back to reading the full-resolution chunk and
    box-filtering in memory.

    Parameters
    ----------
    zarr_path : str
        Path to the zarr store (directory or zip).
    array_name : str
        Name of the elevation array within the zarr group.
    pixel_spacing_x, pixel_spacing_y : float
        World-space size of one raster pixel.
    max_lod : int
        Maximum LOD level.
    crs_origin : tuple of (float, float) or None
        ``(origin_x, origin_y)`` in CRS coordinates.
    crs_pixel_spacing : tuple of (float, float) or None
        ``(dx, dy)`` signed CRS spacing per pixel.
    cf_decode : tuple of (float, float, fill_value) or None
        ``(scale_factor, add_offset, fill_value)`` for CF decoding
        raw integer data to float elevation.  If ``None``, the array
        is assumed to already contain float elevation values.
    """

    __slots__ = (
        '_zarr_path', '_array_name', '_root', '_arr',
        '_psx', '_psy', '_max_lod',
        '_H', '_W', '_chunk_h', '_chunk_w', '_n_cr', '_n_cc',
        '_crs_origin', '_crs_pixel_spacing',
        '_cf_scale', '_cf_offset', '_cf_fill',
        '_lod_arrays',  # {factor: zarr.Array}
        '_tile_cache',  # {(cr, cc, lod, base_sub): np.ndarray}
    )

    def __init__(self, zarr_path, array_name,
                 pixel_spacing_x=1.0, pixel_spacing_y=1.0,
                 max_lod=3,
                 crs_origin=None, crs_pixel_spacing=None,
                 cf_decode=None):
        import zarr

        self._zarr_path = zarr_path
        self._array_name = array_name

        self._root = zarr.open_group(zarr_path, mode='r')
        self._arr = self._root[array_name]

        self._H, self._W = self._arr.shape

        # Read chunk shape from zarr metadata
        chunks = self._arr.chunks
        if isinstance(chunks, (list, tuple)):
            self._chunk_h = int(chunks[0])
            self._chunk_w = int(chunks[1])
        else:
            self._chunk_h = int(chunks)
            self._chunk_w = int(chunks)

        self._n_cr = (self._H + self._chunk_h - 1) // self._chunk_h
        self._n_cc = (self._W + self._chunk_w - 1) // self._chunk_w

        self._psx = pixel_spacing_x
        self._psy = pixel_spacing_y
        self._max_lod = max_lod

        # CRS metadata
        self._crs_origin = crs_origin
        self._crs_pixel_spacing = crs_pixel_spacing

        # CF decode parameters
        if cf_decode is not None:
            self._cf_scale, self._cf_offset, self._cf_fill = cf_decode
        else:
            # Try to read from zarr attrs.  CF convention stores the fill
            # value as a `_FillValue` attribute; the zarr metadata
            # `fill_value` is a different concept (chunk fill, often 0).
            attrs = self._arr.attrs
            self._cf_scale = float(attrs.get('scale_factor', 1.0))
            self._cf_offset = float(attrs.get('add_offset', 0.0))
            cf_fv = attrs.get('_FillValue', None)
            if cf_fv is not None:
                self._cf_fill = cf_fv
            else:
                self._cf_fill = self._arr.fill_value
            # If scale is 1.0 and offset is 0.0 and fill is NaN,
            # no CF decode needed (data is already float elevation)
            if (self._cf_scale == 1.0 and self._cf_offset == 0.0
                    and (self._cf_fill is None
                         or (isinstance(self._cf_fill, float)
                             and math.isnan(self._cf_fill)))):
                self._cf_fill = None  # sentinel: no CF decode

        # Discover pre-built LOD arrays
        self._lod_arrays = {}
        factor = 2
        while factor <= 2 ** max_lod:
            lod_name = f'{array_name}_lod{factor}'
            if lod_name in self._root:
                self._lod_arrays[factor] = self._root[lod_name]
                logger.debug('Found LOD array: %s', lod_name)
            factor *= 2

        # Per-chunk read cache (bounded by max_tiles in LOD manager)
        self._tile_cache = {}

    # -- Grid geometry --------------------------------------------------

    @property
    def shape(self):
        return (self._H, self._W)

    @property
    def chunk_shape(self):
        return (self._chunk_h, self._chunk_w)

    @property
    def pixel_spacing(self):
        return (self._psx, self._psy)

    @property
    def n_chunk_rows(self):
        return self._n_cr

    @property
    def n_chunk_cols(self):
        return self._n_cc

    @property
    def crs_transform(self):
        if self._crs_origin is not None and self._crs_pixel_spacing is not None:
            ox, oy = self._crs_origin
            dx, dy = self._crs_pixel_spacing
            return (ox, oy, dx, dy)
        return None

    @property
    def supports_streaming(self):
        # Zarr sources can always serve any in-bounds chunk on demand
        return True

    def can_stream(self, cr, cc):
        # Can serve any chunk within the zarr array extent
        return self.is_in_bounds(cr, cc)

    # -- Data access ----------------------------------------------------

    def _cf_decode_array(self, data):
        """Apply CF decoding to raw integer data."""
        if self._cf_fill is None:
            # No CF decode needed — already float elevation
            return np.asarray(data, dtype=np.float32)

        data = np.asarray(data, dtype=np.float32)
        data[data == self._cf_fill] = np.nan
        if self._cf_scale != 1.0 or self._cf_offset != 0.0:
            data = data * self._cf_scale + self._cf_offset
        return data

    def _read_raw_chunk(self, cr, cc):
        """Read a single chunk from the zarr array and CF-decode it."""
        r0 = cr * self._chunk_h
        c0 = cc * self._chunk_w
        r1 = min(r0 + self._chunk_h, self._H)
        c1 = min(c0 + self._chunk_w, self._W)

        try:
            raw = self._arr[r0:r1, c0:c1]
        except Exception:
            logger.warning('Failed to read zarr chunk (%d, %d)', cr, cc)
            return None

        # Handle CuPy arrays from GPU-direct zarr reads
        if not isinstance(raw, np.ndarray):
            try:
                import cupy as cp
                if isinstance(raw, cp.ndarray):
                    raw = cp.asnumpy(raw)
            except ImportError:
                raw = np.asarray(raw)

        return self._cf_decode_array(raw)

    def _read_lod_chunk(self, cr, cc, lod_factor):
        """Read a chunk from a pre-built LOD array."""
        lod_arr = self._lod_arrays.get(lod_factor)
        if lod_arr is None:
            return None

        lH, lW = lod_arr.shape

        # Map full-res chunk coords to LOD array coords
        r0 = (cr * self._chunk_h) // lod_factor
        c0 = (cc * self._chunk_w) // lod_factor
        r1 = min(((cr + 1) * self._chunk_h) // lod_factor + 1, lH)
        c1 = min(((cc + 1) * self._chunk_w) // lod_factor + 1, lW)

        if r1 <= r0 or c1 <= c0:
            return None

        try:
            raw = lod_arr[r0:r1, c0:c1]
        except Exception:
            return None

        if not isinstance(raw, np.ndarray):
            try:
                import cupy as cp
                if isinstance(raw, cp.ndarray):
                    raw = cp.asnumpy(raw)
            except ImportError:
                raw = np.asarray(raw)

        # LOD arrays are pre-decoded (float32, NaN for fill) when built
        # by _build_lod_arrays.  Source-level LOD arrays may still need
        # CF decode if they were built from the raw integer source.
        return np.asarray(raw, dtype=np.float32)

    def read_chunk(self, cr, cc, lod, base_subsample=1):
        if not self.is_in_bounds(cr, cc):
            return None

        cache_key = (cr, cc, lod, base_subsample)
        cached = self._tile_cache.get(cache_key)
        if cached is not None:
            return cached

        total_subsample = base_subsample * (2 ** lod)

        # Try pre-built LOD array first
        if total_subsample > 1:
            # Find the best available LOD factor <= total_subsample
            best_factor = 1
            for factor in sorted(self._lod_arrays.keys()):
                if factor <= total_subsample:
                    best_factor = factor
                else:
                    break

            if best_factor > 1:
                tile = self._read_lod_chunk(cr, cc, best_factor)
                if tile is not None:
                    # Apply remaining stride if needed
                    remaining = total_subsample // best_factor
                    if remaining > 1:
                        tile = tile[::remaining, ::remaining]
                    if tile.shape[0] >= 2 and tile.shape[1] >= 2:
                        self._tile_cache[cache_key] = tile
                        return tile

        # Fall back: read full-res chunk and box-filter
        full = self._read_raw_chunk(cr, cc)
        if full is None:
            return None

        if total_subsample > 1:
            # Apply base_subsample stride first, then box-filter
            if base_subsample > 1:
                full = full[::base_subsample, ::base_subsample]

            # Box-filter downsample for LOD levels
            tile = full
            for _ in range(lod):
                tile = _box_downsample_2x(tile)
        else:
            tile = full

        if tile.shape[0] < 2 or tile.shape[1] < 2:
            return None

        tile = np.asarray(tile, dtype=np.float32)
        self._tile_cache[cache_key] = tile
        return tile

    def read_full_res_chunk(self, cr, cc):
        if not self.is_in_bounds(cr, cc):
            return None
        return self._read_raw_chunk(cr, cc)

    # -- Stats ----------------------------------------------------------

    def elevation_stats(self):
        """Compute elevation stats from a coarse global read.

        Reads the coarsest available LOD array (or strides the full-res
        array) to compute approximate min/max/mean without loading the
        full dataset.
        """
        # Use the coarsest pre-built LOD array if available
        if self._lod_arrays:
            coarsest_factor = max(self._lod_arrays.keys())
            arr = self._lod_arrays[coarsest_factor]
            try:
                data = np.asarray(arr[:], dtype=np.float32)
                if not isinstance(data, np.ndarray):
                    try:
                        import cupy as cp
                        data = cp.asnumpy(data)
                    except ImportError:
                        data = np.asarray(data)
                emin = float(np.nanmin(data))
                emax = float(np.nanmax(data))
                emean = float(np.nanmean(data))
                return (emin, emax, emean)
            except Exception:
                pass

        # Fall back: stride the full-res array heavily
        stride = max(1, max(self._H, self._W) // 1024)
        try:
            data = self._arr[::stride, ::stride]
            if not isinstance(data, np.ndarray):
                try:
                    import cupy as cp
                    data = cp.asnumpy(data)
                except ImportError:
                    data = np.asarray(data)
            data = self._cf_decode_array(data)
            emin = float(np.nanmin(data))
            emax = float(np.nanmax(data))
            emean = float(np.nanmean(data))
            return (emin, emax, emean)
        except Exception:
            return None

    # -- Cache management -----------------------------------------------

    def clear_caches(self):
        self._tile_cache.clear()

    def close(self):
        self._tile_cache.clear()
        self._lod_arrays.clear()


# ======================================================================
# Factory functions
# ======================================================================

def chunk_source_from_raster(terrain_np, tile_size=None,
                             pixel_spacing_x=1.0, pixel_spacing_y=1.0,
                             max_lod=3,
                             crs_origin=None, crs_pixel_spacing=None):
    """Create an :class:`InMemoryChunkSource` from a numpy array.

    Parameters
    ----------
    terrain_np : np.ndarray
        Full-resolution elevation, shape ``(H, W)``.
    tile_size : int or None
        Tile edge length.  ``None`` → single chunk covering the array.
    pixel_spacing_x, pixel_spacing_y : float
        World-space pixel spacing.
    max_lod : int
        Maximum pyramid levels.
    crs_origin : tuple or None
        ``(origin_x, origin_y)`` in CRS coordinates.
    crs_pixel_spacing : tuple or None
        ``(dx, dy)`` signed pixel spacing.

    Returns
    -------
    InMemoryChunkSource
    """
    return InMemoryChunkSource(
        terrain_np, tile_size=tile_size,
        pixel_spacing_x=pixel_spacing_x,
        pixel_spacing_y=pixel_spacing_y,
        max_lod=max_lod,
        crs_origin=crs_origin,
        crs_pixel_spacing=crs_pixel_spacing,
    )


def chunk_source_from_zarr(zarr_path, array_name,
                           pixel_spacing_x=1.0, pixel_spacing_y=1.0,
                           max_lod=3,
                           crs_origin=None, crs_pixel_spacing=None,
                           cf_decode=None):
    """Create a :class:`ZarrChunkSource` from a zarr store.

    Parameters
    ----------
    zarr_path : str
        Path to the zarr store.
    array_name : str
        Name of the elevation array in the zarr group.
    pixel_spacing_x, pixel_spacing_y : float
        World-space pixel spacing.
    max_lod : int
        Maximum LOD level.
    crs_origin : tuple or None
        ``(origin_x, origin_y)`` in CRS coordinates.
    crs_pixel_spacing : tuple or None
        ``(dx, dy)`` signed pixel spacing.
    cf_decode : tuple or None
        ``(scale_factor, add_offset, fill_value)`` for CF decoding.

    Returns
    -------
    ZarrChunkSource
    """
    return ZarrChunkSource(
        zarr_path, array_name,
        pixel_spacing_x=pixel_spacing_x,
        pixel_spacing_y=pixel_spacing_y,
        max_lod=max_lod,
        crs_origin=crs_origin,
        crs_pixel_spacing=crs_pixel_spacing,
        cf_decode=cf_decode,
    )


