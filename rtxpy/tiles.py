"""XYZ map tile fetching and compositing service.

Downloads map tiles (satellite, street map, topo, etc.) from XYZ tile servers
and composites them into an RGB texture that matches the terrain grid dimensions.
Each tile is reprojected to the raster's native CRS so the imagery aligns with
the terrain regardless of projection.  Tiles stream in the background via a
thread pool and appear progressively.

Usage::

    from rtxpy.tiles import XYZTileService
    svc = XYZTileService('https://tile.openstreetmap.org/{z}/{x}/{y}.png', raster)
    svc.fetch_visible_tiles()
    # ... each frame:
    gpu_tex = svc.get_gpu_texture()  # (H, W, 3) cupy float32 or None
"""

import math
import os
import threading
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.request import urlopen, Request

import numpy as np


TILE_PROVIDERS = {
    'osm': 'https://tile.openstreetmap.org/{z}/{x}/{y}.png',
    'satellite': 'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
    'topo': 'https://tile.opentopomap.org/{z}/{x}/{y}.png',
}


# ---------------------------------------------------------------------------
# Tile math
# ---------------------------------------------------------------------------

def lat_lon_to_tile(lat, lon, zoom):
    """Convert lat/lon (WGS84 degrees) to tile x, y at given zoom.

    Returns
    -------
    (tile_x, tile_y) : tuple of int
    """
    n = 2 ** zoom
    lat_rad = math.radians(lat)
    x = int((lon + 180.0) / 360.0 * n)
    y = int((1.0 - math.log(math.tan(lat_rad) + 1.0 / math.cos(lat_rad)) / math.pi) / 2.0 * n)
    x = max(0, min(n - 1, x))
    y = max(0, min(n - 1, y))
    return x, y


def tile_to_lat_lon(x, y, zoom):
    """Return the NW corner lat/lon of tile (x, y, zoom).

    Returns
    -------
    (lat, lon) : tuple of float
        Latitude and longitude in degrees.
    """
    n = 2 ** zoom
    lon = x / n * 360.0 - 180.0
    lat = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * y / n))))
    return lat, lon


def compute_zoom_from_resolution(pixel_spacing_meters):
    """Choose zoom level whose ground resolution best matches *pixel_spacing_meters*.

    Uses equatorial resolution as baseline:
    resolution(z) ~ 156543 / 2^z  meters/pixel (for 256-px tiles).
    """
    if pixel_spacing_meters <= 0:
        return 18
    zoom = math.log2(156543.0 / pixel_spacing_meters)
    zoom = max(0, min(19, int(round(zoom))))
    return zoom


def tiles_for_bounds(lat_min, lon_min, lat_max, lon_max, zoom):
    """Return list of (x, y, z) tile coordinates covering the given WGS84 bbox.

    Parameters
    ----------
    lat_min, lon_min, lat_max, lon_max : float
        Bounding box in WGS84 degrees.
    zoom : int
        Tile zoom level.

    Returns
    -------
    list of (x, y, z)
    """
    # NW corner -> tile with smallest y
    x_min, y_min = lat_lon_to_tile(lat_max, lon_min, zoom)
    # SE corner -> tile with largest y
    x_max, y_max = lat_lon_to_tile(lat_min, lon_max, zoom)

    result = []
    for ty in range(y_min, y_max + 1):
        for tx in range(x_min, x_max + 1):
            result.append((tx, ty, zoom))
    return result


# ---------------------------------------------------------------------------
# Per-pixel lat/lon grid construction
# ---------------------------------------------------------------------------

def _build_latlon_grids(raster):
    """Build (H, W) arrays of WGS84 latitude and longitude for every pixel.

    Uses the raster's coordinate arrays and CRS metadata (via rioxarray /
    pyproj) to transform each pixel centre to WGS84.  Falls back to
    treating coordinates as lon/lat when pyproj or CRS metadata is
    unavailable.

    Returns
    -------
    lats : ndarray, shape (H, W)
    lons : ndarray, shape (H, W)
    """
    H, W = raster.shape

    # Get coordinate arrays from xarray
    try:
        x_coords = raster.coords['x'].values
        y_coords = raster.coords['y'].values
    except KeyError:
        # No named x/y coords — uniform grid on [0, 1]
        lats = np.linspace(1, 0, H).reshape(-1, 1) * np.ones(W)
        lons = np.linspace(0, 1, W).reshape(1, -1) * np.ones((H, 1))
        return lats.astype(np.float64), lons.astype(np.float64)

    # Meshgrid in the raster's native CRS (ensure numpy for pyproj)
    xx, yy = np.meshgrid(
        np.asarray(x_coords, dtype=np.float64),
        np.asarray(y_coords, dtype=np.float64),
    )

    # Try CRS-aware reprojection
    try:
        import pyproj
        crs = raster.rio.crs
    except (ImportError, AttributeError):
        crs = None

    if crs is not None:
        if crs.is_geographic:
            # Already lon/lat
            return yy, xx
        # Use proj4 string to avoid PROJ database lookup for WGS84
        try:
            transformer = pyproj.Transformer.from_crs(
                crs, "+proj=longlat +datum=WGS84 +no_defs",
                always_xy=True,
            )
        except pyproj.exceptions.CRSError:
            # Fallback: try EPSG code (needs PROJ database)
            transformer = pyproj.Transformer.from_crs(
                crs, "EPSG:4326", always_xy=True,
            )
        lons, lats = transformer.transform(xx, yy)
        return lats, lons

    # Fallback: assume x = lon, y = lat
    return yy, xx


def _compute_pixel_spacing_meters(raster):
    """Estimate the ground-truth pixel spacing in metres.

    For projected CRS the coordinate deltas are already metric.
    For geographic CRS we convert degrees to approximate metres.
    """
    try:
        x_coords = raster.coords['x'].values
        y_coords = raster.coords['y'].values
        dx = abs(float(x_coords[1] - x_coords[0])) if len(x_coords) > 1 else 1.0
        dy = abs(float(y_coords[1] - y_coords[0])) if len(y_coords) > 1 else 1.0
    except (KeyError, IndexError):
        return 30.0  # sensible default

    try:
        crs = raster.rio.crs
        if crs is not None and crs.is_geographic:
            avg_lat = float(y_coords.mean())
            m_per_deg = 111_320 * math.cos(math.radians(avg_lat))
            return ((dx + dy) / 2.0) * m_per_deg
    except (ImportError, AttributeError):
        pass

    return (dx + dy) / 2.0


# ---------------------------------------------------------------------------
# XYZTileService
# ---------------------------------------------------------------------------

class XYZTileService:
    """Background tile fetcher that composites XYZ map tiles into an RGB texture.

    The service builds per-pixel WGS84 lat/lon arrays from the raster's
    coordinate system (using pyproj when available), so tiles are correctly
    reprojected onto any CRS.

    Parameters
    ----------
    url_template : str
        Provider name (e.g. ``'osm'``, ``'satellite'``) or a full URL
        template containing ``{z}``, ``{x}``, ``{y}`` placeholders.
    raster : xarray.DataArray
        Terrain raster — used to determine bounds, shape, and CRS.
    zoom : int, optional
        Tile zoom level (0–19). If ``None``, defaults to 13.
    """

    def __init__(self, url_template, raster, zoom=None):
        # Resolve provider name -> URL template
        if url_template in TILE_PROVIDERS:
            self.provider_name = url_template
            self._url_template = TILE_PROVIDERS[url_template]
        else:
            self.provider_name = 'custom'
            self._url_template = url_template

        # Disk cache: ~/.cache/rtxpy/tiles/<provider>/
        self._disk_cache_dir = (
            Path.home() / '.cache' / 'rtxpy' / 'tiles' / self.provider_name
        )
        self._disk_cache_dir.mkdir(parents=True, exist_ok=True)

        self._raster = raster
        H, W = raster.shape

        # Per-pixel WGS84 coordinate grids — shape (H, W) each
        self._lats, self._lons = _build_latlon_grids(raster)

        # Terrain WGS84 bounds (from the grids)
        self._lat_min = float(np.nanmin(self._lats))
        self._lat_max = float(np.nanmax(self._lats))
        self._lon_min = float(np.nanmin(self._lons))
        self._lon_max = float(np.nanmax(self._lons))

        self._zoom = zoom if zoom is not None else 13

        print(f"  Tile service: zoom {self._zoom}, "
              f"bounds ({self._lat_min:.4f}, {self._lon_min:.4f}) - "
              f"({self._lat_max:.4f}, {self._lon_max:.4f})")

        # RGB texture (CPU, float32 [0-1], H x W x 3)
        self._rgb_texture = np.zeros((H, W, 3), dtype=np.float32)
        self._texture_dirty = True
        self._gpu_texture = None

        # Thread safety
        self._lock = threading.Lock()

        # Tile cache: (x, y, z) -> (256, 256, 3) float32
        self._tile_cache = OrderedDict()
        self._cache_limit = 500

        # Background fetching
        self._executor = ThreadPoolExecutor(max_workers=4)
        self._pending = set()   # tile coords currently being fetched
        self._fetched = set()   # tile coords already composited
        self._generation = 0    # incremented on shutdown / reset

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fetch_visible_tiles(self):
        """Compute tile list for terrain bounds and submit unfetched tiles."""
        tile_list = tiles_for_bounds(
            self._lat_min, self._lon_min,
            self._lat_max, self._lon_max,
            self._zoom,
        )
        print(f"  Fetching {len(tile_list)} tiles at zoom {self._zoom}...")
        gen = self._generation
        for coord in tile_list:
            if coord not in self._fetched and coord not in self._pending:
                self._pending.add(coord)
                self._executor.submit(self._fetch_tile, coord, gen)

    def get_gpu_texture(self):
        """Return (H, W, 3) cupy float32 array, or None if no tiles yet.

        Uploads to GPU only when the CPU texture has been updated.
        """
        with self._lock:
            if not self._texture_dirty and self._gpu_texture is not None:
                return self._gpu_texture
            if not self._fetched:
                return None
            try:
                import cupy
                self._gpu_texture = cupy.asarray(self._rgb_texture)
            except ImportError:
                return None
            self._texture_dirty = False
            return self._gpu_texture

    def reinit_for_raster(self, raster, pixel_spacing_x=None, pixel_spacing_y=None):
        """Re-initialize texture for a new raster (e.g. after resolution change).

        Recomputes per-pixel coordinate grids and re-composites cached tiles.
        """
        self._raster = raster
        H, W = raster.shape

        # Rebuild coordinate grids
        self._lats, self._lons = _build_latlon_grids(raster)
        self._lat_min = float(np.nanmin(self._lats))
        self._lat_max = float(np.nanmax(self._lats))
        self._lon_min = float(np.nanmin(self._lons))
        self._lon_max = float(np.nanmax(self._lons))

        with self._lock:
            self._rgb_texture = np.zeros((H, W, 3), dtype=np.float32)
            self._texture_dirty = True
            self._gpu_texture = None

        # Re-composite any cached tiles onto the new grid
        old_fetched = list(self._fetched)
        self._fetched.clear()
        for coord in old_fetched:
            if coord in self._tile_cache:
                self._composite_tile(coord, self._tile_cache[coord])

        # Fetch any new tiles needed at this zoom
        self.fetch_visible_tiles()

    def shutdown(self):
        """Cancel pending fetches and shut down the executor."""
        self._generation += 1
        self._executor.shutdown(wait=False)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _disk_path(self, coord):
        """Return the disk cache path for a tile coordinate."""
        tx, ty, tz = coord
        return self._disk_cache_dir / f"{tz}" / f"{tx}_{ty}.png"

    def _fetch_tile(self, coord, generation):
        """Download a single tile and composite it (runs in background thread)."""
        if generation != self._generation:
            return  # stale request

        tx, ty, tz = coord

        # 1. In-memory cache
        if coord in self._tile_cache:
            tile_array = self._tile_cache[coord]
        else:
            tile_array = None
            disk_path = self._disk_path(coord)

            # 2. Disk cache
            if disk_path.exists():
                try:
                    from PIL import Image
                    img = Image.open(disk_path).convert('RGB')
                    tile_array = np.asarray(img, dtype=np.float32) / 255.0
                except Exception:
                    pass  # corrupt file — re-download

            # 3. Download
            if tile_array is None:
                url = self._url_template.format(x=tx, y=ty, z=tz)
                try:
                    req = Request(url, headers={'User-Agent': 'rtxpy/0.1'})
                    with urlopen(req, timeout=15) as resp:
                        data = resp.read()
                except Exception:
                    self._pending.discard(coord)
                    return

                try:
                    from PIL import Image
                    import io
                    img = Image.open(io.BytesIO(data)).convert('RGB')
                    tile_array = np.asarray(img, dtype=np.float32) / 255.0
                except Exception:
                    self._pending.discard(coord)
                    return

                # Save to disk cache
                try:
                    disk_path.parent.mkdir(parents=True, exist_ok=True)
                    with open(disk_path, 'wb') as f:
                        f.write(data)
                except Exception:
                    pass  # non-fatal

            # Store in memory LRU cache
            self._tile_cache[coord] = tile_array
            if len(self._tile_cache) > self._cache_limit:
                self._tile_cache.popitem(last=False)

        if generation != self._generation:
            return

        self._composite_tile(coord, tile_array)
        self._pending.discard(coord)

    def _composite_tile(self, coord, tile_array):
        """Write tile pixels into the RGB texture (thread-safe).

        Uses the per-pixel lat/lon grids so the tile is correctly
        reprojected regardless of the raster's CRS.
        """
        tx, ty, tz = coord
        tile_h, tile_w = tile_array.shape[:2]

        # Tile NW and SE corners in WGS84
        nw_lat, nw_lon = tile_to_lat_lon(tx, ty, tz)
        se_lat, se_lon = tile_to_lat_lon(tx + 1, ty + 1, tz)

        lat_span = nw_lat - se_lat
        lon_span = se_lon - nw_lon
        if lat_span == 0 or lon_span == 0:
            self._fetched.add(coord)
            return

        # Find all raster pixels whose lat/lon falls inside this tile
        mask = ((self._lats >= se_lat) & (self._lats <= nw_lat) &
                (self._lons >= nw_lon) & (self._lons <= se_lon))

        rows, cols = np.where(mask)
        if len(rows) == 0:
            self._fetched.add(coord)
            return

        # For each matching raster pixel, compute its position within the tile
        pixel_lats = self._lats[rows, cols]
        pixel_lons = self._lons[rows, cols]

        row_fracs = (nw_lat - pixel_lats) / lat_span   # 0 = top (north), 1 = bottom (south)
        col_fracs = (pixel_lons - nw_lon) / lon_span    # 0 = left (west),  1 = right (east)

        tile_rows = np.clip((row_fracs * tile_h).astype(int), 0, tile_h - 1)
        tile_cols = np.clip((col_fracs * tile_w).astype(int), 0, tile_w - 1)

        # Sample tile and write into texture
        pixels = tile_array[tile_rows, tile_cols]  # (N, 3)

        with self._lock:
            self._rgb_texture[rows, cols] = pixels
            self._texture_dirty = True

        self._fetched.add(coord)
