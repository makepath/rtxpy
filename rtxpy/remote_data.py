"""Download remote geospatial data: DEM tiles, OSM features, buildings, roads, water, and fire.

Supports Copernicus GLO-30, USGS SRTM 1-arc-second, USGS 3DEP
1/3-arc-second (10 m), and USGS 3DEP 1-meter DEM sources,
OpenStreetMap vector features via osmnx,
Microsoft Global ML Building Footprints, convenience wrappers
for roads and water features, and NASA FIRMS fire detection footprints.

All network dependencies (``requests``, ``rioxarray``, ``osmnx``,
``pandas``, ``geopandas``, ``shapely``) are optional and imported
lazily — a clear ``ImportError`` is raised at call time if a required
package is missing.
"""

import json
import math
import re
from pathlib import Path

# ---------------------------------------------------------------------------
# Overture Maps Foundation (optional, requires duckdb)
# ---------------------------------------------------------------------------

_OVERTURE_RELEASE = '2026-01-21.0'
_OVERTURE_S3 = f's3://overturemaps-us-west-2/release/{_OVERTURE_RELEASE}'

_OVERTURE_MAJOR_CLASSES = {'motorway', 'trunk', 'primary', 'secondary'}
_OVERTURE_MINOR_CLASSES = {'tertiary', 'residential', 'living_street',
                           'unclassified', 'service'}


def _query_overture(bounds, theme, type_name, columns, release=None):
    """Query Overture Maps GeoParquet on S3 via DuckDB.

    Parameters
    ----------
    bounds : tuple of float
        (west, south, east, north) in WGS84 degrees.
    theme : str
        Overture theme (e.g. ``'buildings'``, ``'transportation'``).
    type_name : str
        Overture type within theme (e.g. ``'building'``, ``'segment'``).
    columns : list of str
        Columns to select from the parquet dataset.
    release : str, optional
        Overture release version.  Defaults to ``_OVERTURE_RELEASE``.

    Returns
    -------
    pandas.DataFrame
        One row per feature with requested columns plus ``geometry_json``.
    """
    try:
        import duckdb
    except ImportError:
        raise ImportError(
            "duckdb is required for Overture Maps data. "
            "Install with: pip install duckdb"
        )

    conn = duckdb.connect()
    conn.execute("INSTALL spatial; LOAD spatial;")
    conn.execute("INSTALL httpfs; LOAD httpfs;")
    conn.execute("SET s3_region='us-west-2';")

    release = release or _OVERTURE_RELEASE
    s3_path = (f's3://overturemaps-us-west-2/release/{release}'
               f'/theme={theme}/type={type_name}/*')

    west, south, east, north = bounds
    col_str = ', '.join(columns)

    query = f"""
        SELECT {col_str}, ST_AsGeoJSON(geometry) AS geometry_json
        FROM read_parquet('{s3_path}', filename=true, hive_partitioning=1)
        WHERE bbox.xmin > {west} AND bbox.xmax < {east}
          AND bbox.ymin > {south} AND bbox.ymax < {north}
    """

    result = conn.execute(query).fetchdf()
    conn.close()
    return result


def _compute_srtm_tiles(bounds):
    """Return list of (tile_name, url) for USGS SRTM 1-arc-second tiles.

    SRTM tiles are named by their *northern* latitude boundary:
    ``n43w122`` covers lat [42, 43], lon [-122, -121].
    """
    west, south, east, north = bounds
    base_url = (
        "https://prd-tnm.s3.amazonaws.com"
        "/StagedProducts/Elevation/1/TIFF/current"
    )

    lat_min = math.ceil(south)
    lat_max = math.ceil(north)
    lon_min = math.floor(west)
    lon_max = math.floor(east)

    tiles = []
    for lat in range(lat_min, lat_max + 1):
        for lon in range(lon_min, lon_max + 1):
            ns = "n" if lat >= 0 else "s"
            ew = "w" if lon < 0 else "e"
            tile_name = f"{ns}{abs(lat):02d}{ew}{abs(lon):03d}"
            url = f"{base_url}/{tile_name}/USGS_1_{tile_name}.tif"
            tiles.append((tile_name, url))
    return tiles


def _compute_usgs_10m_tiles(bounds):
    """Return list of (tile_name, url) for USGS 3DEP 1/3-arc-second tiles.

    Same grid naming as SRTM (``n43w122`` covers lat [42, 43],
    lon [-122, -121]) but hosted under the ``/13/`` path prefix
    with filename prefix ``USGS_13_``.
    """
    west, south, east, north = bounds
    base_url = (
        "https://prd-tnm.s3.amazonaws.com"
        "/StagedProducts/Elevation/13/TIFF/current"
    )

    lat_min = math.ceil(south)
    lat_max = math.ceil(north)
    lon_min = math.floor(west)
    lon_max = math.floor(east)

    tiles = []
    for lat in range(lat_min, lat_max + 1):
        for lon in range(lon_min, lon_max + 1):
            ns = "n" if lat >= 0 else "s"
            ew = "w" if lon < 0 else "e"
            tile_name = f"{ns}{abs(lat):02d}{ew}{abs(lon):03d}"
            url = f"{base_url}/{tile_name}/USGS_13_{tile_name}.tif"
            tiles.append((tile_name, url))
    return tiles


def _compute_copernicus_tiles(bounds):
    """Return list of (tile_name, url) for Copernicus GLO-30 tiles.

    Copernicus tiles are named by their *SW corner*:
    ``Copernicus_DSM_COG_10_N10_00_W061_00_DEM`` covers lat [10, 11],
    lon [-61, -60].
    """
    west, south, east, north = bounds
    base_url = "https://copernicus-dem-30m.s3.amazonaws.com"

    lat_min = math.floor(south)
    lat_max = math.floor(north)
    lon_min = math.floor(west)
    lon_max = math.floor(east)

    tiles = []
    for lat in range(lat_min, lat_max + 1):
        for lon in range(lon_min, lon_max + 1):
            ns = "N" if lat >= 0 else "S"
            ew = "E" if lon >= 0 else "W"
            tile_name = (
                f"Copernicus_DSM_COG_10_{ns}{abs(lat):02d}_00"
                f"_{ew}{abs(lon):03d}_00_DEM"
            )
            url = f"{base_url}/{tile_name}/{tile_name}.tif"
            tiles.append((tile_name, url))
    return tiles


_TNM_API_URL = "https://tnmaccess.nationalmap.gov/api/v1/products"


def _query_usgs_1m_tiles(bounds):
    """Query USGS TNM API for 1-meter DEM tiles covering a bounding box.

    The National Map API discovers available 1 m DEM tiles from the
    USGS 3DEP program across all lidar projects.  When multiple
    projects cover the same grid cell the newest publication is kept.

    Parameters
    ----------
    bounds : tuple of float
        (west, south, east, north) in WGS84 degrees.

    Returns
    -------
    list of (tile_name, url)
    """
    try:
        import requests
    except ImportError:
        raise ImportError(
            "requests is required for fetch_dem(). "
            "Install it with: pip install requests"
        )

    west, south, east, north = bounds
    all_items = []
    offset = 0

    while True:
        params = {
            "datasets": "Digital Elevation Model (DEM) 1 meter",
            "bbox": f"{west},{south},{east},{north}",
            "prodFormats": "GeoTIFF",
            "max": 100,
            "offset": offset,
        }
        resp = requests.get(_TNM_API_URL, params=params, timeout=60)
        resp.raise_for_status()
        data = resp.json()

        items = data.get("items", [])
        all_items.extend(items)
        if len(items) < 100:
            break
        offset += 100

    # Prefer newer publications when tiles overlap
    all_items.sort(
        key=lambda item: item.get("publicationDate", ""),
        reverse=True,
    )

    seen_coords = set()
    tiles = []
    for item in all_items:
        url = item.get("downloadURL", "")
        if not url or not url.endswith(".tif"):
            continue

        m = re.search(r"x(\d+)y(\d+)", url)
        if not m:
            continue

        coord = (int(m.group(1)), int(m.group(2)))
        if coord in seen_coords:
            continue
        seen_coords.add(coord)

        tile_name = url.split("/")[-1].replace(".tif", "")
        tiles.append((tile_name, url))

    return tiles


def _download_tile(url, tile_path):
    """Download a single tile with streaming and caching."""
    try:
        import requests
    except ImportError:
        raise ImportError(
            "requests is required for fetch_dem(). "
            "Install it with: pip install requests"
        )

    resp = requests.get(url, timeout=180, stream=True)
    resp.raise_for_status()
    with open(tile_path, "wb") as f:
        for chunk in resp.iter_content(chunk_size=1 << 20):
            f.write(chunk)


def _save_zarr(da, output_path):
    """Save a DataArray to a CF-encoded zarr store with int16 compression."""
    import numpy as np
    import xarray as xr
    from zarr.codecs import BloscCodec, BloscShuffle

    H, W = da.shape[-2], da.shape[-1]

    # Build dataset with elevation + spatial_ref
    ds = da.to_dataset(name='elevation')

    # Drop scalar coord variables left by rioxarray (e.g. 'band')
    for coord in list(ds.coords):
        if coord not in ('x', 'y') and coord not in ds.dims:
            ds = ds.drop_vars(coord)

    # Add spatial_ref variable with CRS metadata
    crs_wkt = da.rio.crs.to_wkt() if da.rio.crs else ""
    transform = da.rio.transform()
    geo_transform = (f"{transform.c} {transform.a} {transform.b} "
                     f"{transform.f} {transform.d} {transform.e}")
    ds['spatial_ref'] = xr.DataArray(
        np.int32(0),
        attrs={
            'crs_wkt': crs_wkt,
            'GeoTransform': geo_transform,
        },
    )

    # Drop attrs that conflict with our CF encoding or spatial_ref variable
    for attr in ('grid_mapping', 'spatial_ref',
                 'scale_factor', 'add_offset', '_FillValue'):
        ds['elevation'].attrs.pop(attr, None)

    encoding = {
        'elevation': {
            'dtype': 'int16',
            'scale_factor': np.float64(0.1),
            'add_offset': np.float64(0.0),
            '_FillValue': np.int16(-9999),
            'compressors': BloscCodec(cname='zstd', clevel=6,
                                      shuffle=BloscShuffle.bitshuffle),
            'chunks': (min(2048, H), min(2048, W)),
        },
    }

    ds.to_zarr(str(output_path), mode='w', encoding=encoding)


def _load_zarr(output_path):
    """Load a CF-encoded zarr store and return a float DataArray with CRS."""
    import xarray as xr
    import rioxarray  # noqa: F401 — needed for .rio accessor

    ds = xr.open_zarr(str(output_path))
    da = ds['elevation']

    # Attach CRS from spatial_ref variable
    if 'spatial_ref' in ds:
        crs_wkt = ds['spatial_ref'].attrs.get('crs_wkt', '')
        if crs_wkt:
            da = da.rio.write_crs(crs_wkt)

    return da


def _merge_clip_reproject(tile_paths, bounds, crs, output_path):
    """Merge tile arrays, clip to bounds, optionally reproject, and save."""
    try:
        import rioxarray as rxr
    except ImportError:
        raise ImportError(
            "rioxarray is required for fetch_dem(). "
            "Install it with: pip install rioxarray"
        )

    tiles = [rxr.open_rasterio(str(p), masked=True).squeeze() for p in tile_paths]

    if len(tiles) == 1:
        merged = tiles[0]
    else:
        # Reproject to a common CRS when tiles span multiple zones
        base_crs = tiles[0].rio.crs
        reprojected = []
        for t in tiles:
            if t.rio.crs != base_crs:
                t = t.rio.reproject(base_crs)
            reprojected.append(t)

        from rioxarray.merge import merge_arrays
        merged = merge_arrays(reprojected)

    west, south, east, north = bounds
    # bounds are always WGS84; pass crs so clip works for projected rasters
    merged = merged.rio.clip_box(
        minx=west, miny=south, maxx=east, maxy=north,
        crs="EPSG:4326",
    )

    if crs is not None:
        merged = merged.rio.reproject(crs)

    # Dispatch by output format
    output_path = Path(output_path)
    if output_path.suffix == '.zarr':
        _save_zarr(merged, output_path)
    else:
        merged.rio.to_raster(str(output_path))

    return merged


def fetch_dem(bounds, output_path, source="copernicus", crs=None, cache_dir=None):
    """Download, merge, and clip DEM tiles for a bounding box.

    Parameters
    ----------
    bounds : tuple of float
        (west, south, east, north) in WGS84 degrees.
    output_path : str or Path
        Where to save the final merged/clipped/reprojected DEM.
        Use ``.zarr`` for a chunked, CF-encoded zarr store (int16 +
        scale_factor=0.1, Blosc zstd compression) or ``.tif`` for
        GeoTIFF.  If the path already exists, loads and returns it
        directly.
    source : str
        ``'copernicus'`` for Copernicus GLO-30 (30 m), ``'srtm'`` for
        USGS 1-arc-second (~30 m), ``'usgs_10m'`` for USGS 3DEP
        1/3-arc-second (~10 m, US coverage), or ``'usgs_1m'`` for
        USGS 3DEP 1-meter lidar DEM (US coverage only, ~30 MB per
        10 km tile).
    crs : str, optional
        Target CRS for reprojection (e.g. ``'EPSG:32620'``).
        ``None`` keeps the native CRS.
    cache_dir : str or Path, optional
        Directory for caching individual tiles.  Defaults to
        *output_path*'s parent directory.

    Returns
    -------
    xarray.DataArray
    """
    try:
        import rioxarray as rxr
    except ImportError:
        raise ImportError(
            "rioxarray is required for fetch_dem(). "
            "Install it with: pip install rioxarray"
        )

    output_path = Path(output_path)

    # Cache hit: zarr stores are directories, tif files are regular files
    if output_path.suffix == '.zarr' and output_path.is_dir():
        print(f"Using cached DEM: {output_path.name}")
        return _load_zarr(output_path)
    elif output_path.suffix != '.zarr' and output_path.exists():
        print(f"Using cached DEM: {output_path.name}")
        return rxr.open_rasterio(str(output_path), masked=True).squeeze()

    if cache_dir is None:
        cache_dir = output_path.parent
    else:
        cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    source = source.lower()
    if source == "srtm":
        tiles = _compute_srtm_tiles(bounds)
        ext_prefix = "USGS_1_"
    elif source == "copernicus":
        tiles = _compute_copernicus_tiles(bounds)
        ext_prefix = ""
    elif source == "usgs_10m":
        tiles = _compute_usgs_10m_tiles(bounds)
        ext_prefix = "USGS_13_"
    elif source == "usgs_1m":
        tiles = _query_usgs_1m_tiles(bounds)
        ext_prefix = ""
    else:
        raise ValueError(
            f"Unknown source {source!r}; use 'copernicus', 'srtm', 'usgs_10m', or 'usgs_1m'"
        )

    print(f"Downloading {len(tiles)} {source} tile(s)...")

    tile_paths = []
    for tile_name, url in tiles:
        tile_path = cache_dir / f"{ext_prefix}{tile_name}.tif"

        if not tile_path.exists():
            print(f"  Downloading {tile_name}...")
            try:
                _download_tile(url, tile_path)
            except Exception as e:
                print(f"  Warning: Failed to download {tile_name}: {e}")
                continue
        else:
            print(f"  Using cached {tile_name}")

        tile_paths.append(tile_path)

    if not tile_paths:
        raise RuntimeError("Failed to download any elevation tiles")

    print(f"  Merging {len(tile_paths)} tile(s)...")
    merged = _merge_clip_reproject(tile_paths, bounds, crs, output_path)
    print(f"  Saved DEM to {output_path}")

    return merged


def _query_usgs_lidar_tiles(bounds):
    """Query USGS TNM API for LiDAR Point Cloud (LPC) LAZ tiles.

    Parameters
    ----------
    bounds : tuple of float
        (west, south, east, north) in WGS84 degrees.

    Returns
    -------
    list of (tile_name, url, size_bytes)
        Sorted by newest ``publicationDate`` first, deduplicated by tile name.
    """
    try:
        import requests
    except ImportError:
        raise ImportError(
            "requests is required for fetch_lidar(). "
            "Install it with: pip install requests"
        )

    west, south, east, north = bounds
    all_items = []
    offset = 0

    while True:
        params = {
            "datasets": "Lidar Point Cloud (LPC)",
            "bbox": f"{west},{south},{east},{north}",
            "prodFormats": "LAZ",
            "max": 100,
            "offset": offset,
        }
        resp = requests.get(_TNM_API_URL, params=params, timeout=60)
        resp.raise_for_status()
        data = resp.json()

        items = data.get("items", [])
        all_items.extend(items)
        if len(items) < 100:
            break
        offset += 100

    # Prefer newer publications when tiles overlap
    all_items.sort(
        key=lambda item: item.get("publicationDate", ""),
        reverse=True,
    )

    seen_names = set()
    tiles = []
    for item in all_items:
        url = item.get("downloadURL", "")
        if not url:
            continue

        tile_name = item.get("title", url.split("/")[-1])
        # Sanitize tile name for use as filename
        tile_name = re.sub(r'[^\w\-.]', '_', tile_name)

        if tile_name in seen_names:
            continue
        seen_names.add(tile_name)

        size_bytes = item.get("sizeInBytes", 0)
        tiles.append((tile_name, url, size_bytes))

    return tiles


def fetch_lidar(bounds, cache_dir=None, max_tiles=None):
    """Download USGS 3DEP LiDAR LAZ tiles for a bounding box.

    Queries the USGS TNM API for Lidar Point Cloud (LPC) tiles in LAZ
    format, downloads them with caching, and returns paths to the local
    files.  The LAZ files can be passed directly to
    ``place_pointcloud()`` which handles coordinate conversion.

    Parameters
    ----------
    bounds : tuple of float
        (west, south, east, north) in WGS84 degrees.
    cache_dir : str or Path, optional
        Directory for cached LAZ files.  Defaults to
        ``~/.cache/rtxpy/lidar/``.
    max_tiles : int, optional
        Limit the number of tiles downloaded.  Tiles are sorted by
        newest publication date first, so the most recent data is
        preferred.  ``None`` downloads all available tiles.

    Returns
    -------
    list of Path
        Paths to downloaded LAZ files.
    """
    try:
        import requests  # noqa: F401 — validate dependency early
    except ImportError:
        raise ImportError(
            "requests is required for fetch_lidar(). "
            "Install it with: pip install requests"
        )

    if cache_dir is None:
        cache_dir = Path.home() / ".cache" / "rtxpy" / "lidar"
    else:
        cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    tiles = _query_usgs_lidar_tiles(bounds)
    if not tiles:
        print("No LiDAR tiles found for the given bounds.")
        return []

    total_mb = sum(s for _, _, s in tiles) / (1 << 20)
    print(f"Found {len(tiles)} LiDAR tile(s) ({total_mb:.0f} MB total)")

    if max_tiles is not None and max_tiles < len(tiles):
        print(f"  Limiting to {max_tiles} of {len(tiles)} tile(s)")
        tiles = tiles[:max_tiles]

    laz_paths = []
    for tile_name, url, size_bytes in tiles:
        # Ensure .laz extension
        fname = tile_name if tile_name.endswith(".laz") else f"{tile_name}.laz"
        tile_path = cache_dir / fname

        if tile_path.exists() and tile_path.stat().st_size > 0:
            print(f"  Using cached {tile_name}")
        else:
            size_mb = size_bytes / (1 << 20)
            print(f"  Downloading {tile_name} ({size_mb:.1f} MB)...")
            try:
                _download_tile(url, tile_path)
            except Exception as e:
                print(f"  Warning: Failed to download {tile_name}: {e}")
                continue

        laz_paths.append(tile_path)

    if not laz_paths:
        raise RuntimeError("Failed to download any LiDAR tiles")

    print(f"  {len(laz_paths)} LAZ file(s) ready")
    return laz_paths


def fetch_osm(bounds, tags=None, crs=None, cache_path=None):
    """Download OpenStreetMap features for a bounding box.

    Returns a GeoJSON FeatureCollection dict that can be passed directly
    to ``place_geojson()``.

    Parameters
    ----------
    bounds : tuple of float
        (west, south, east, north) in WGS84 degrees.
    tags : dict, optional
        OSM tags to query.  Keys are tag names, values are ``True``
        (any value), a string, or a list of strings.
        Default: ``{'highway': True, 'building': True}``.
    crs : str, optional
        Target CRS for reprojection (e.g. ``'EPSG:5070'``).  When set,
        the geometry is reprojected to match the terrain so that
        ``place_geojson()`` can place features without an additional
        CRS transform.  ``None`` keeps the native WGS84 coordinates.
    cache_path : str or Path, optional
        Path to cache the result as GeoJSON.  If the file already
        exists, loads and returns it directly.  The cache stores the
        *final* (possibly reprojected) result.

    Returns
    -------
    dict
        GeoJSON FeatureCollection.

    Examples
    --------
    >>> from rtxpy import fetch_osm
    >>> roads = fetch_osm((-122.3, 42.8, -121.9, 43.0),
    ...                   tags={'highway': True}, crs='EPSG:5070')
    >>> dem.rtx.place_geojson(roads, height=5.0, label_field='name')
    """
    try:
        import osmnx as ox
    except ModuleNotFoundError:
        raise ImportError(
            "osmnx is required for fetch_osm(). "
            "Install it with: pip install osmnx"
        )
    except ImportError as exc:
        # osmnx installed but broken (e.g. shapely version mismatch)
        raise ImportError(
            f"osmnx failed to import: {exc}\n"
            "Try upgrading: pip install osmnx --upgrade"
        ) from exc
    import json

    if cache_path is not None:
        cache_path = Path(cache_path)
        if cache_path.exists():
            print(f"Using cached OSM data: {cache_path.name}")
            with open(cache_path) as f:
                return json.load(f)

    if tags is None:
        tags = {"highway": True, "building": True}

    print(f"Downloading OSM features ({', '.join(tags)})...")
    gdf = ox.features.features_from_bbox(bounds, tags)

    if gdf.empty:
        print("  No features found")
        geojson = {"type": "FeatureCollection", "features": []}
    else:
        print(f"  Downloaded {len(gdf)} features")

        # Reproject to target CRS if requested
        if crs is not None:
            gdf = gdf.to_crs(crs)
            print(f"  Reprojected to {crs}")

        # Reset the MultiIndex (element_type, osmid) to regular columns,
        # drop columns that are entirely empty, and convert to GeoJSON.
        gdf = gdf.reset_index()
        gdf = gdf.dropna(axis=1, how="all")
        geojson = json.loads(gdf.to_json())

    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "w") as f:
            json.dump(geojson, f)
        print(f"  Cached to {cache_path}")

    return geojson


# ---------------------------------------------------------------------------
# Microsoft Global ML Building Footprints
# ---------------------------------------------------------------------------

_BUILDINGS_LINKS_URL = (
    "https://minedbuildings.z5.web.core.windows.net"
    "/global-buildings/dataset-links.csv"
)


def _lat_lon_to_tile(lat, lon, zoom):
    """Convert WGS84 lat/lon to tile x, y at *zoom*."""
    n = 2 ** zoom
    lat_rad = math.radians(lat)
    x = int((lon + 180.0) / 360.0 * n)
    y = int(
        (1.0 - math.log(math.tan(lat_rad) + 1.0 / math.cos(lat_rad))
         / math.pi) / 2.0 * n
    )
    return max(0, min(n - 1, x)), max(0, min(n - 1, y))


def _tile_to_quadkey(tx, ty, level):
    """Convert tile x, y at *level* to a quadkey string."""
    qk = []
    for i in range(level, 0, -1):
        digit = 0
        mask = 1 << (i - 1)
        if tx & mask:
            digit += 1
        if ty & mask:
            digit += 2
        qk.append(str(digit))
    return "".join(qk)


def _quadkeys_for_bounds(bounds, level=9):
    """Return the set of quadkeys at *level* that cover *bounds*."""
    west, south, east, north = bounds
    x_min, y_min = _lat_lon_to_tile(north, west, level)   # NW corner
    x_max, y_max = _lat_lon_to_tile(south, east, level)   # SE corner
    keys = set()
    for ty in range(y_min, y_max + 1):
        for tx in range(x_min, x_max + 1):
            keys.add(_tile_to_quadkey(tx, ty, level))
    return keys


def _feature_in_bounds(feature, west, south, east, north):
    """Quick check: is any vertex of the feature inside the bbox?"""
    geom = feature.get("geometry", {})
    coords = geom.get("coordinates", [])
    # Polygon → coords is [ring, ...], each ring is [[lon, lat], ...]
    for ring in coords:
        for pt in ring:
            lon, lat = pt[0], pt[1]
            if west <= lon <= east and south <= lat <= north:
                return True
    return False


def _fetch_buildings_overture(bounds, cache_path=None, crs=None):
    """Fetch building footprints from Overture Maps via DuckDB."""
    print("Querying Overture Maps buildings...")
    df = _query_overture(bounds, 'buildings', 'building',
                         ['height', 'num_floors', 'names'])

    import pandas as pd

    features = []
    for _, row in df.iterrows():
        geom = json.loads(row['geometry_json'])

        # Height: prefer explicit height, fall back to num_floors * 3m
        height = row.get('height')
        if pd.isna(height):
            nf = row.get('num_floors')
            if not pd.isna(nf):
                height = float(nf) * 3.0
            else:
                height = -1.0

        features.append({
            "type": "Feature",
            "geometry": geom,
            "properties": {
                "height": float(height),
                "confidence": 1.0,
            },
        })

    print(f"  Found {len(features)} buildings from Overture Maps")

    # Reproject if requested
    if crs is not None and features:
        try:
            import geopandas as gpd
        except ImportError:
            raise ImportError(
                "geopandas is required for CRS reprojection. "
                "Install with: pip install geopandas"
            )
        gdf = gpd.GeoDataFrame.from_features(features, crs="EPSG:4326")
        gdf = gdf.to_crs(crs)
        print(f"  Reprojected to {crs}")
        geojson = json.loads(gdf.to_json())
    else:
        geojson = {"type": "FeatureCollection", "features": features}

    # Cache result
    if cache_path is not None:
        cache_path = Path(cache_path)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "w") as f:
            json.dump(geojson, f)
        print(f"  Cached to {cache_path}")

    return geojson


def fetch_buildings(bounds, cache_path=None, crs=None, cache_dir=None,
                     source='microsoft'):
    """Download building footprints for a bounding box.

    Supports two data sources:

    - ``'microsoft'`` (default) — Microsoft Global ML Building Footprints.
      Uses the `dataset-links.csv
      <https://github.com/microsoft/GlobalMLBuildingFootprints>`_ index to
      find level-9 quadkey partitions that overlap *bounds*, downloads the
      compressed GeoJSONL files, filters features to the bounding box, and
      returns a standard GeoJSON FeatureCollection.
    - ``'overture'`` — Overture Maps Foundation buildings.  Queries the
      Overture GeoParquet dataset on S3 via DuckDB (requires ``duckdb``).
      Provides deduplicated footprints with richer attributes (height,
      number of floors) aggregated from OSM, Microsoft, Meta, and Esri.

    Parameters
    ----------
    bounds : tuple of float
        (west, south, east, north) in WGS84 degrees.
    cache_path : str or Path, optional
        Path to cache the final GeoJSON result.  If the file already
        exists, loads and returns it directly.
    crs : str, optional
        Target CRS for reprojection (e.g. ``'EPSG:32620'``).  When set,
        geometries are reprojected so ``place_geojson()`` can place them
        without an additional CRS transform.  ``None`` keeps WGS84.
    cache_dir : str or Path, optional
        Directory for caching the dataset-links.csv index and downloaded
        partition files.  Defaults to ``~/.cache/rtxpy/buildings``.
        Only used with ``source='microsoft'``.
    source : str
        Data source: ``'microsoft'`` (default) or ``'overture'``.

    Returns
    -------
    dict
        GeoJSON FeatureCollection.  Each feature has ``properties``
        containing ``height`` (metres, ``-1`` if unknown) and
        ``confidence`` (0–1).

    Examples
    --------
    >>> from rtxpy import fetch_buildings
    >>> bldgs = fetch_buildings((-61.5, 10.6, -61.4, 10.7),
    ...                         crs='EPSG:32620')
    >>> dem.rtx.place_geojson(bldgs, height=8.0)
    """
    if cache_path is not None:
        cache_path = Path(cache_path)
        if cache_path.exists():
            print(f"Using cached buildings: {cache_path.name}")
            with open(cache_path) as f:
                return json.load(f)

    source = source.lower()
    if source == 'overture':
        return _fetch_buildings_overture(bounds, cache_path=cache_path,
                                         crs=crs)
    elif source != 'microsoft':
        raise ValueError(
            f"Unknown source {source!r}; use 'microsoft' or 'overture'"
        )

    if cache_dir is None:
        cache_dir = Path.home() / ".cache" / "rtxpy" / "buildings"
    else:
        cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    # 1. Download / cache the dataset index -----------------------------------
    links_path = cache_dir / "dataset-links.csv"
    if not links_path.exists():
        print("Downloading Microsoft Buildings dataset index...")
        _download_tile(_BUILDINGS_LINKS_URL, links_path)

    # 2. Find matching quadkeys -----------------------------------------------
    quadkeys = _quadkeys_for_bounds(bounds)
    print(f"  Bounding box covers {len(quadkeys)} quadkey(s) at level 9")

    import csv
    matching_urls = []
    with open(links_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            qk = row["QuadKey"].strip()
            if qk in quadkeys:
                loc = row.get("Location", "").strip()
                matching_urls.append((qk, loc, row["Url"].strip()))

    if not matching_urls:
        print("  No building footprint partitions found for this area")
        geojson = {"type": "FeatureCollection", "features": []}
        if cache_path is not None:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            with open(cache_path, "w") as f:
                json.dump(geojson, f)
        return geojson

    print(f"  Downloading {len(matching_urls)} partition(s)...")

    # 3. Download partitions and extract features -----------------------------
    west, south, east, north = bounds
    features = []

    try:
        import pandas as pd
    except ImportError:
        raise ImportError(
            "pandas is required for fetch_buildings(). "
            "Install it with: pip install pandas"
        )

    for qk, loc, url in matching_urls:
        # Cache by location+quadkey to avoid collisions when multiple
        # country sources share the same quadkey.
        cache_name = f"{loc}_{qk}.csv.gz" if loc else f"{qk}.csv.gz"
        part_path = cache_dir / cache_name
        if not part_path.exists():
            print(f"    Downloading {loc}/{qk}...")
            try:
                _download_tile(url, part_path)
            except Exception as e:
                print(f"    Warning: Failed to download {loc}/{qk}: {e}")
                continue
        else:
            print(f"    Using cached {loc}/{qk}")

        # Read GeoJSONL (each line is a JSON feature despite .csv.gz ext)
        try:
            df = pd.read_json(part_path, lines=True)
        except Exception as e:
            print(f"    Warning: Failed to parse {loc}/{qk}: {e}")
            continue

        for _, row in df.iterrows():
            feature = {
                "type": "Feature",
                "geometry": row["geometry"],
                "properties": row.get("properties", {}),
            }
            if _feature_in_bounds(feature, west, south, east, north):
                features.append(feature)

    print(f"  Found {len(features)} buildings in bounding box")

    # 4. Reproject if requested -----------------------------------------------
    if crs is not None and features:
        try:
            import geopandas as gpd
            from shapely.geometry import shape
        except ImportError:
            raise ImportError(
                "geopandas and shapely are required for CRS reprojection. "
                "Install with: pip install geopandas shapely"
            )

        gdf = gpd.GeoDataFrame.from_features(features, crs="EPSG:4326")
        gdf = gdf.to_crs(crs)
        print(f"  Reprojected to {crs}")
        geojson = json.loads(gdf.to_json())
    else:
        geojson = {"type": "FeatureCollection", "features": features}

    # 5. Cache and return -----------------------------------------------------
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "w") as f:
            json.dump(geojson, f)
        print(f"  Cached to {cache_path}")

    return geojson


# ---------------------------------------------------------------------------
# Convenience wrappers: roads and water
# ---------------------------------------------------------------------------

# OSM highway values grouped by importance
_MAJOR_ROAD_VALUES = [
    "motorway", "trunk", "primary", "secondary",
    "motorway_link", "trunk_link", "primary_link", "secondary_link",
]
_MINOR_ROAD_VALUES = [
    "tertiary", "residential", "unclassified", "service",
    "living_street", "tertiary_link",
]


def _fetch_roads_overture(bounds, road_type="all", cache_path=None, crs=None):
    """Fetch road data from Overture Maps via DuckDB."""
    print("Querying Overture Maps roads...")
    df = _query_overture(bounds, 'transportation', 'segment',
                         ['class', 'subtype', 'names', 'road_surface'])

    # Filter to road subtypes only (exclude rail, water)
    if 'subtype' in df.columns:
        df = df[df['subtype'] == 'road']

    # Filter by road_type using Overture class values
    road_type = road_type.lower()
    if road_type == 'major':
        allowed = _OVERTURE_MAJOR_CLASSES
    elif road_type == 'minor':
        allowed = _OVERTURE_MINOR_CLASSES
    elif road_type == 'all':
        allowed = _OVERTURE_MAJOR_CLASSES | _OVERTURE_MINOR_CLASSES
    else:
        raise ValueError(
            f"Unknown road_type {road_type!r}; use 'major', 'minor', or 'all'"
        )

    if 'class' in df.columns:
        df = df[df['class'].isin(allowed)]

    import pandas as pd

    features = []
    for _, row in df.iterrows():
        geom = json.loads(row['geometry_json'])

        # Extract name from Overture names struct (may be dict or None)
        names = row.get('names')
        name = None
        if isinstance(names, dict):
            name = names.get('primary', None)

        # Map Overture class to OSM-style highway tag for compatibility
        road_class = row.get('class', '')
        if pd.isna(road_class):
            road_class = None

        features.append({
            "type": "Feature",
            "geometry": geom,
            "properties": {
                "name": name,
                "highway": road_class if road_class else None,
            },
        })

    print(f"  Found {len(features)} road segments from Overture Maps")

    # Reproject if requested
    if crs is not None and features:
        try:
            import geopandas as gpd
        except ImportError:
            raise ImportError(
                "geopandas is required for CRS reprojection. "
                "Install with: pip install geopandas"
            )
        gdf = gpd.GeoDataFrame.from_features(features, crs="EPSG:4326")
        gdf = gdf.to_crs(crs)
        print(f"  Reprojected to {crs}")
        geojson = json.loads(gdf.to_json())
    else:
        geojson = {"type": "FeatureCollection", "features": features}

    # Cache result
    if cache_path is not None:
        cache_path = Path(cache_path)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "w") as f:
            json.dump(geojson, f)
        print(f"  Cached to {cache_path}")

    return geojson


def fetch_roads(bounds, road_type="all", cache_path=None, crs=None,
                source='osm'):
    """Download road data for a bounding box.

    Supports two data sources:

    - ``'osm'`` (default) — OpenStreetMap via osmnx.
    - ``'overture'`` — Overture Maps Foundation transportation data.
      Queries the Overture GeoParquet dataset on S3 via DuckDB (requires
      ``duckdb``).  Provides deduplicated road segments with
      classification and surface attributes.

    Parameters
    ----------
    bounds : tuple of float
        (west, south, east, north) in WGS84 degrees.
    road_type : str
        Which roads to include:

        - ``'major'`` — motorways, trunks, primary, secondary (and links)
        - ``'minor'`` — tertiary, residential, unclassified, service, etc.
        - ``'all'`` (default) — both major and minor roads
    cache_path : str or Path, optional
        Path to cache the result as GeoJSON.  If the file already
        exists, loads and returns it directly.
    crs : str, optional
        Target CRS for reprojection (e.g. ``'EPSG:32620'``).
        ``None`` keeps WGS84.
    source : str
        Data source: ``'osm'`` (default) or ``'overture'``.

    Returns
    -------
    dict
        GeoJSON FeatureCollection.

    Examples
    --------
    >>> from rtxpy import fetch_roads
    >>> roads = fetch_roads((-122.3, 42.8, -121.9, 43.0),
    ...                     road_type='major', crs='EPSG:5070')
    >>> dem.rtx.place_geojson(roads, height=3.0, label_field='name')
    """
    source = source.lower()
    if source == 'overture':
        return _fetch_roads_overture(bounds, road_type=road_type,
                                     cache_path=cache_path, crs=crs)
    elif source != 'osm':
        raise ValueError(
            f"Unknown source {source!r}; use 'osm' or 'overture'"
        )

    road_type = road_type.lower()
    if road_type == "major":
        values = _MAJOR_ROAD_VALUES
    elif road_type == "minor":
        values = _MINOR_ROAD_VALUES
    elif road_type == "all":
        values = _MAJOR_ROAD_VALUES + _MINOR_ROAD_VALUES
    else:
        raise ValueError(
            f"Unknown road_type {road_type!r}; use 'major', 'minor', or 'all'"
        )

    tags = {"highway": values}
    return fetch_osm(bounds, tags=tags, crs=crs, cache_path=cache_path)


# OSM tags for water features
_WATERWAY_VALUES = ["river", "stream", "canal", "drain", "ditch"]

# Overture water subtype groupings
_OVERTURE_WATERWAY_SUBTYPES = {'river', 'canal', 'stream'}
_OVERTURE_WATERWAY_CLASSES = {'drain', 'ditch'}
_OVERTURE_WATERBODY_SUBTYPES = {'lake', 'pond', 'reservoir', 'ocean'}


def _fetch_water_overture(bounds, water_type="all", cache_path=None, crs=None):
    """Fetch water features from Overture Maps via DuckDB."""
    print("Querying Overture Maps water...")
    df = _query_overture(bounds, 'base', 'water',
                         ['subtype', 'class', 'names'])

    import pandas as pd

    water_type = water_type.lower()

    # Filter by water_type
    if water_type == 'waterway':
        mask = (df['subtype'].isin(_OVERTURE_WATERWAY_SUBTYPES) |
                df['class'].isin(_OVERTURE_WATERWAY_CLASSES))
        df = df[mask]
    elif water_type == 'waterbody':
        df = df[df['subtype'].isin(_OVERTURE_WATERBODY_SUBTYPES)]
    elif water_type != 'all':
        raise ValueError(
            f"Unknown water_type {water_type!r}; "
            "use 'waterway', 'waterbody', or 'all'"
        )

    features = []
    for _, row in df.iterrows():
        geom = json.loads(row['geometry_json'])

        # Extract name from Overture names struct
        names = row.get('names')
        name = None
        if isinstance(names, dict):
            name = names.get('primary', None)

        # Map Overture subtype/class → OSM-style properties for place_water()
        subtype = row.get('subtype', '')
        cls = row.get('class', '')
        if pd.isna(subtype):
            subtype = ''
        if pd.isna(cls):
            cls = ''

        props = {"name": name}
        if subtype in ('river', 'canal'):
            props['waterway'] = subtype
        elif subtype == 'stream' or cls in ('drain', 'ditch'):
            props['waterway'] = cls if cls in ('drain', 'ditch') else 'stream'
        elif subtype in ('lake', 'pond', 'reservoir', 'ocean'):
            props['natural'] = 'water'
        else:
            # Fallback: treat as minor waterway
            props['waterway'] = 'stream'

        features.append({
            "type": "Feature",
            "geometry": geom,
            "properties": props,
        })

    print(f"  Found {len(features)} water features from Overture Maps")

    # Reproject if requested
    if crs is not None and features:
        try:
            import geopandas as gpd
        except ImportError:
            raise ImportError(
                "geopandas is required for CRS reprojection. "
                "Install with: pip install geopandas"
            )
        gdf = gpd.GeoDataFrame.from_features(features, crs="EPSG:4326")
        gdf = gdf.to_crs(crs)
        print(f"  Reprojected to {crs}")
        geojson = json.loads(gdf.to_json())
    else:
        geojson = {"type": "FeatureCollection", "features": features}

    # Cache result
    if cache_path is not None:
        cache_path = Path(cache_path)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "w") as f:
            json.dump(geojson, f)
        print(f"  Cached to {cache_path}")

    return geojson


def fetch_water(bounds, water_type="all", cache_path=None, crs=None,
                source='osm'):
    """Download water / waterway features.

    Supports two data sources:

    - ``'osm'`` (default) — OpenStreetMap via osmnx.
    - ``'overture'`` — Overture Maps Foundation water data.
      Queries the Overture GeoParquet dataset on S3 via DuckDB (requires
      ``duckdb``).

    Parameters
    ----------
    bounds : tuple of float
        (west, south, east, north) in WGS84 degrees.
    water_type : str
        Which features to include:

        - ``'waterway'`` — linear features: rivers, streams, canals, etc.
        - ``'waterbody'`` — area features: lakes, reservoirs, ponds.
        - ``'all'`` (default) — both waterways and waterbodies.
    cache_path : str or Path, optional
        Path to cache the result as GeoJSON.  If the file already
        exists, loads and returns it directly.
    crs : str, optional
        Target CRS for reprojection (e.g. ``'EPSG:32620'``).
        ``None`` keeps WGS84.
    source : str
        Data source: ``'osm'`` (default) or ``'overture'``.

    Returns
    -------
    dict
        GeoJSON FeatureCollection.

    Examples
    --------
    >>> from rtxpy import fetch_water
    >>> rivers = fetch_water((-61.6, 10.4, -61.2, 10.7),
    ...                      water_type='waterway', crs='EPSG:32620')
    >>> dem.rtx.place_geojson(rivers, height=2.0, label_field='name')
    """
    if cache_path is not None:
        cache_path = Path(cache_path)
        if cache_path.exists():
            print(f"Using cached water data: {cache_path.name}")
            with open(cache_path) as f:
                return json.load(f)

    source = source.lower()
    if source == 'overture':
        return _fetch_water_overture(bounds, water_type=water_type,
                                     cache_path=cache_path, crs=crs)
    elif source != 'osm':
        raise ValueError(
            f"Unknown source {source!r}; use 'osm' or 'overture'"
        )

    water_type = water_type.lower()
    if water_type == "waterway":
        tags = {"waterway": _WATERWAY_VALUES}
    elif water_type == "waterbody":
        tags = {"natural": "water"}
    elif water_type == "all":
        tags = {"waterway": _WATERWAY_VALUES, "natural": "water"}
    else:
        raise ValueError(
            f"Unknown water_type {water_type!r}; "
            "use 'waterway', 'waterbody', or 'all'"
        )

    return fetch_osm(bounds, tags=tags, crs=crs, cache_path=cache_path)


# ---------------------------------------------------------------------------
# Overture Maps: places, infrastructure, land use
# ---------------------------------------------------------------------------


def fetch_places(bounds, category=None, cache_path=None, crs=None):
    """Download point-of-interest data from Overture Maps.

    Returns Point geometries from the Overture ``places/place`` dataset.

    Parameters
    ----------
    bounds : tuple of float
        (west, south, east, north) in WGS84 degrees.
    category : str or list of str, optional
        Filter on Overture ``categories.primary`` (e.g. ``'eat_and_drink'``,
        ``'education'``, ``['hospital', 'school']``).  ``None`` returns all.
    cache_path : str or Path, optional
        Path to cache the result as GeoJSON.  If the file already
        exists, loads and returns it directly.
    crs : str, optional
        Target CRS for reprojection.  ``None`` keeps WGS84.

    Returns
    -------
    dict
        GeoJSON FeatureCollection with Point geometries.  Each feature
        has ``name``, ``category``, and ``confidence`` properties.

    Examples
    --------
    >>> from rtxpy import fetch_places
    >>> restaurants = fetch_places((-61.55, 10.62, -61.48, 10.69),
    ...                            category='eat_and_drink')
    """
    if cache_path is not None:
        cache_path = Path(cache_path)
        if cache_path.exists():
            print(f"Using cached places data: {cache_path.name}")
            with open(cache_path) as f:
                return json.load(f)

    print("Querying Overture Maps places...")
    df = _query_overture(bounds, 'places', 'place',
                         ['names', 'categories', 'confidence'])

    import pandas as pd

    # Filter by category
    if category is not None:
        if isinstance(category, str):
            category = [category]
        category_set = set(category)

        def _matches_category(cats):
            if isinstance(cats, dict):
                return cats.get('primary', '') in category_set
            return False

        df = df[df['categories'].apply(_matches_category)]

    features = []
    for _, row in df.iterrows():
        geom = json.loads(row['geometry_json'])

        names = row.get('names')
        name = None
        if isinstance(names, dict):
            name = names.get('primary', None)

        cats = row.get('categories')
        cat = None
        if isinstance(cats, dict):
            cat = cats.get('primary', None)

        conf = row.get('confidence')
        if pd.isna(conf):
            conf = -1.0

        features.append({
            "type": "Feature",
            "geometry": geom,
            "properties": {
                "name": name,
                "category": cat,
                "confidence": float(conf),
            },
        })

    print(f"  Found {len(features)} places from Overture Maps")

    # Reproject if requested
    if crs is not None and features:
        try:
            import geopandas as gpd
        except ImportError:
            raise ImportError(
                "geopandas is required for CRS reprojection. "
                "Install with: pip install geopandas"
            )
        gdf = gpd.GeoDataFrame.from_features(features, crs="EPSG:4326")
        gdf = gdf.to_crs(crs)
        print(f"  Reprojected to {crs}")
        geojson = json.loads(gdf.to_json())
    else:
        geojson = {"type": "FeatureCollection", "features": features}

    # Cache result
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "w") as f:
            json.dump(geojson, f)
        print(f"  Cached to {cache_path}")

    return geojson


def fetch_infrastructure(bounds, infra_type='all', cache_path=None, crs=None):
    """Download infrastructure features from Overture Maps.

    Returns features from the Overture ``base/infrastructure`` dataset.

    Parameters
    ----------
    bounds : tuple of float
        (west, south, east, north) in WGS84 degrees.
    infra_type : str
        Filter on Overture ``subtype``:

        - ``'communication'`` — cell towers, antennas
        - ``'power'`` — power lines, substations
        - ``'bridge'`` — bridges
        - ``'tower'`` — towers
        - ``'transit'`` — transit stations
        - ``'airport'`` — airports, runways, helipads
        - ``'all'`` (default) — everything
    cache_path : str or Path, optional
        Path to cache the result as GeoJSON.
    crs : str, optional
        Target CRS for reprojection.  ``None`` keeps WGS84.

    Returns
    -------
    dict
        GeoJSON FeatureCollection.  Each feature has ``name``,
        ``subtype``, ``class``, and ``height`` (metres, -1 if unknown)
        properties.

    Examples
    --------
    >>> from rtxpy import fetch_infrastructure
    >>> towers = fetch_infrastructure((-61.55, 10.62, -61.48, 10.69),
    ...                               infra_type='communication')
    """
    if cache_path is not None:
        cache_path = Path(cache_path)
        if cache_path.exists():
            print(f"Using cached infrastructure data: {cache_path.name}")
            with open(cache_path) as f:
                return json.load(f)

    print("Querying Overture Maps infrastructure...")
    df = _query_overture(bounds, 'base', 'infrastructure',
                         ['subtype', 'class', 'names', 'height'])

    import pandas as pd

    infra_type = infra_type.lower()
    valid_types = {'communication', 'power', 'bridge', 'tower',
                   'transit', 'airport'}
    if infra_type != 'all':
        if infra_type not in valid_types:
            raise ValueError(
                f"Unknown infra_type {infra_type!r}; use one of "
                f"{sorted(valid_types)} or 'all'"
            )
        if 'subtype' in df.columns:
            df = df[df['subtype'] == infra_type]

    features = []
    for _, row in df.iterrows():
        geom = json.loads(row['geometry_json'])

        names = row.get('names')
        name = None
        if isinstance(names, dict):
            name = names.get('primary', None)

        subtype = row.get('subtype', None)
        if pd.isna(subtype):
            subtype = None
        cls = row.get('class', None)
        if pd.isna(cls):
            cls = None

        height = row.get('height')
        if pd.isna(height):
            height = -1.0

        features.append({
            "type": "Feature",
            "geometry": geom,
            "properties": {
                "name": name,
                "subtype": subtype,
                "class": cls,
                "height": float(height),
            },
        })

    print(f"  Found {len(features)} infrastructure features from Overture Maps")

    # Reproject if requested
    if crs is not None and features:
        try:
            import geopandas as gpd
        except ImportError:
            raise ImportError(
                "geopandas is required for CRS reprojection. "
                "Install with: pip install geopandas"
            )
        gdf = gpd.GeoDataFrame.from_features(features, crs="EPSG:4326")
        gdf = gdf.to_crs(crs)
        print(f"  Reprojected to {crs}")
        geojson = json.loads(gdf.to_json())
    else:
        geojson = {"type": "FeatureCollection", "features": features}

    # Cache result
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "w") as f:
            json.dump(geojson, f)
        print(f"  Cached to {cache_path}")

    return geojson


def fetch_land_use(bounds, land_type='all', cache_path=None, crs=None):
    """Download land use/land cover features from Overture Maps.

    Returns features from the Overture ``base/land_use`` dataset.

    Parameters
    ----------
    bounds : tuple of float
        (west, south, east, north) in WGS84 degrees.
    land_type : str
        Filter on Overture ``subtype``:

        - ``'residential'``, ``'park'``, ``'agriculture'``,
          ``'education'``, ``'military'``, ``'protected'``,
          ``'developed'``, ``'recreation'``
        - ``'all'`` (default) — everything
    cache_path : str or Path, optional
        Path to cache the result as GeoJSON.
    crs : str, optional
        Target CRS for reprojection.  ``None`` keeps WGS84.

    Returns
    -------
    dict
        GeoJSON FeatureCollection.  Each feature has ``name``,
        ``subtype``, and ``class`` properties.

    Examples
    --------
    >>> from rtxpy import fetch_land_use
    >>> parks = fetch_land_use((-61.55, 10.62, -61.48, 10.69),
    ...                        land_type='park')
    """
    if cache_path is not None:
        cache_path = Path(cache_path)
        if cache_path.exists():
            print(f"Using cached land use data: {cache_path.name}")
            with open(cache_path) as f:
                return json.load(f)

    print("Querying Overture Maps land use...")
    df = _query_overture(bounds, 'base', 'land_use',
                         ['subtype', 'class', 'names'])

    import pandas as pd

    land_type = land_type.lower()
    valid_types = {'residential', 'park', 'agriculture', 'education',
                   'military', 'protected', 'developed', 'recreation'}
    if land_type != 'all':
        if land_type not in valid_types:
            raise ValueError(
                f"Unknown land_type {land_type!r}; use one of "
                f"{sorted(valid_types)} or 'all'"
            )
        if 'subtype' in df.columns:
            df = df[df['subtype'] == land_type]

    features = []
    for _, row in df.iterrows():
        geom = json.loads(row['geometry_json'])

        names = row.get('names')
        name = None
        if isinstance(names, dict):
            name = names.get('primary', None)

        subtype = row.get('subtype', None)
        if pd.isna(subtype):
            subtype = None
        cls = row.get('class', None)
        if pd.isna(cls):
            cls = None

        features.append({
            "type": "Feature",
            "geometry": geom,
            "properties": {
                "name": name,
                "subtype": subtype,
                "class": cls,
            },
        })

    print(f"  Found {len(features)} land use features from Overture Maps")

    # Reproject if requested
    if crs is not None and features:
        try:
            import geopandas as gpd
        except ImportError:
            raise ImportError(
                "geopandas is required for CRS reprojection. "
                "Install with: pip install geopandas"
            )
        gdf = gpd.GeoDataFrame.from_features(features, crs="EPSG:4326")
        gdf = gdf.to_crs(crs)
        print(f"  Reprojected to {crs}")
        geojson = json.loads(gdf.to_json())
    else:
        geojson = {"type": "FeatureCollection", "features": features}

    # Cache result
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "w") as f:
            json.dump(geojson, f)
        print(f"  Cached to {cache_path}")

    return geojson


# ---------------------------------------------------------------------------
# NYC Open Data: restaurant inspection grades
# ---------------------------------------------------------------------------

_NYC_RESTAURANT_URL = "https://data.cityofnewyork.us/resource/43nn-pn8j.json"


def fetch_restaurant_grades(bounds, cache_path=None, crs=None):
    """Download NYC restaurant health inspection grades.

    Queries the DOHMH Restaurant Inspection Results dataset on NYC Open
    Data and returns one Point per restaurant with its most recent
    letter grade.

    Parameters
    ----------
    bounds : tuple of float
        (west, south, east, north) in WGS84 degrees.
    cache_path : str or Path, optional
        Path to cache the result as GeoJSON.  If the file already
        exists, loads and returns it directly.
    crs : str, optional
        Target CRS for reprojection.  ``None`` keeps WGS84.

    Returns
    -------
    dict
        GeoJSON FeatureCollection with Point geometries.  Each feature
        has ``name``, ``cuisine``, ``grade`` (A/B/C), and ``score``
        (lower is better) properties.

    Examples
    --------
    >>> from rtxpy import fetch_restaurant_grades
    >>> grades = fetch_restaurant_grades((-74.02, 40.70, -73.97, 40.75))
    >>> len(grades['features'])
    1234
    """
    try:
        import requests
    except ImportError:
        raise ImportError(
            "requests is required for fetch_restaurant_grades(). "
            "Install with: pip install requests"
        )

    if cache_path is not None:
        cache_path = Path(cache_path)
        if cache_path.exists():
            print(f"Using cached restaurant grades: {cache_path.name}")
            with open(cache_path) as f:
                return json.load(f)

    west, south, east, north = bounds
    print("Fetching NYC restaurant inspection grades...")

    # Paginated fetch — SODA API caps at 50000 rows per request
    all_rows = []
    offset = 0
    page_size = 50000
    while True:
        params = {
            "$select": ("camis, dba, cuisine_description, "
                        "latitude, longitude, score, grade, grade_date"),
            "$where": (f"latitude between {south} and {north} "
                       f"AND longitude between {west} and {east} "
                       f"AND grade IS NOT NULL AND latitude > 0"),
            "$order": "camis, grade_date DESC",
            "$limit": page_size,
            "$offset": offset,
        }
        resp = requests.get(_NYC_RESTAURANT_URL, params=params, timeout=60)
        resp.raise_for_status()
        rows = resp.json()
        all_rows.extend(rows)
        if len(rows) < page_size:
            break
        offset += page_size

    print(f"  Received {len(all_rows)} inspection rows")

    # Deduplicate: keep first row per camis (latest grade_date due to ordering)
    seen = set()
    features = []
    for row in all_rows:
        camis = row.get('camis')
        if camis in seen:
            continue
        seen.add(camis)

        lat = float(row.get('latitude', 0))
        lon = float(row.get('longitude', 0))
        if lat == 0 or lon == 0:
            continue

        grade = row.get('grade', '')
        if grade not in ('A', 'B', 'C'):
            continue

        try:
            score = int(row.get('score', -1))
        except (ValueError, TypeError):
            score = -1

        features.append({
            "type": "Feature",
            "geometry": {
                "type": "Point",
                "coordinates": [lon, lat],
            },
            "properties": {
                "name": row.get('dba', ''),
                "cuisine": row.get('cuisine_description', ''),
                "grade": grade,
                "score": score,
            },
        })

    print(f"  {len(features)} unique restaurants with grades A/B/C")

    # Reproject if requested
    if crs is not None and features:
        try:
            import geopandas as gpd
        except ImportError:
            raise ImportError(
                "geopandas is required for CRS reprojection. "
                "Install with: pip install geopandas"
            )
        gdf = gpd.GeoDataFrame.from_features(features, crs="EPSG:4326")
        gdf = gdf.to_crs(crs)
        print(f"  Reprojected to {crs}")
        geojson = json.loads(gdf.to_json())
    else:
        geojson = {"type": "FeatureCollection", "features": features}

    # Cache result
    if cache_path is not None:
        cache_path = Path(cache_path)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "w") as f:
            json.dump(geojson, f)
        print(f"  Cached to {cache_path}")

    return geojson


# ---------------------------------------------------------------------------
# Wind data (Open-Meteo)
# ---------------------------------------------------------------------------

_OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"


def fetch_wind(bounds, grid_size=20):
    """Fetch current wind data from Open-Meteo for a bounding box.

    Queries the Open-Meteo forecast API for 10 m wind speed and
    direction on a regular lat/lon grid, then decomposes into U/V
    components suitable for particle animation.

    Parameters
    ----------
    bounds : tuple of float
        (west, south, east, north) in WGS84 degrees.
    grid_size : int
        Number of grid points along each axis (default 20).
        Total API points = grid_size².  Open-Meteo allows up to
        ~1 000 points per request.

    Returns
    -------
    dict
        ``'u'`` : ndarray (ny, nx) — east–west wind component (m/s).
        ``'v'`` : ndarray (ny, nx) — north–south wind component (m/s).
        ``'speed'`` : ndarray (ny, nx) — wind speed (m/s).
        ``'direction'`` : ndarray (ny, nx) — meteorological direction
        (degrees, where wind is coming *from*).
        ``'lats'`` : ndarray (ny,) — latitude values.
        ``'lons'`` : ndarray (nx,) — longitude values.

    Examples
    --------
    >>> from rtxpy import fetch_wind
    >>> wind = fetch_wind((-43.42, -23.08, -43.10, -22.84))
    >>> wind['u'].shape
    (20, 20)
    """
    try:
        import requests
    except ImportError:
        raise ImportError(
            "requests is required for fetch_wind(). "
            "Install it with: pip install requests"
        )
    import numpy as np

    west, south, east, north = bounds

    lons = np.linspace(west, east, grid_size)
    lats = np.linspace(south, north, grid_size)
    grid_lons, grid_lats = np.meshgrid(lons, lats)

    lat_str = ",".join(f"{v:.4f}" for v in grid_lats.ravel())
    lon_str = ",".join(f"{v:.4f}" for v in grid_lons.ravel())

    print(f"Fetching wind data ({grid_size}x{grid_size} grid)...")
    resp = requests.get(
        _OPEN_METEO_URL,
        params={
            "latitude": lat_str,
            "longitude": lon_str,
            "current": "wind_speed_10m,wind_direction_10m",
            "wind_speed_unit": "ms",
        },
        timeout=30,
    )
    resp.raise_for_status()
    data = resp.json()

    ny, nx = grid_size, grid_size
    speed = np.empty((ny, nx), dtype=np.float32)
    direction = np.empty((ny, nx), dtype=np.float32)

    for i, point in enumerate(data):
        row = i // nx
        col = i % nx
        current = point.get("current", point)
        speed[row, col] = current["wind_speed_10m"]
        direction[row, col] = current["wind_direction_10m"]

    # Decompose speed + direction into U/V components.
    # Meteorological convention: direction is where wind comes FROM,
    # so a 90 deg wind blows from east to west (negative U).
    dir_rad = np.deg2rad(direction)
    u = -speed * np.sin(dir_rad)
    v = -speed * np.cos(dir_rad)

    mean_speed = float(np.mean(speed))
    print(f"  Mean wind speed: {mean_speed:.1f} m/s")

    return {
        "u": u,
        "v": v,
        "speed": speed,
        "direction": direction,
        "lats": lats,
        "lons": lons,
    }


# ---------------------------------------------------------------------------
# NASA FIRMS fire detection footprints
# ---------------------------------------------------------------------------

_FIRMS_BASE_URL = "https://firms.modaps.eosdis.nasa.gov/api/kml_fire_footprints"

# Approximate WGS84 bounding boxes for each FIRMS region
_FIRMS_REGION_BOUNDS = {
    "canada":                     (-141, 41, -52, 84),
    "alaska":                     (-180, 51, -130, 72),
    "usa_contiguous_and_hawaii":  (-180, 18, -66, 50),
    "central_america":            (-118, 7, -60, 33),
    "south_america":              (-82, -56, -34, 13),
    "europe":                     (-25, 35, 45, 72),
    "northern_and_central_africa":(-18, -5, 52, 38),
    "southern_africa":            (8, -35, 52, 5),
    "russia_asia":                (26, 35, 180, 82),
    "south_asia":                 (60, 5, 100, 40),
    "southeast_asia":             (92, -11, 162, 28),
    "australia_newzealand":       (112, -48, 180, -10),
}


def _firms_regions_for_bounds(bounds):
    """Return FIRMS region names whose bounding box overlaps *bounds*."""
    west, south, east, north = bounds
    regions = []
    for name, (rw, rs, re, rn) in _FIRMS_REGION_BOUNDS.items():
        if west <= re and east >= rw and south <= rn and north >= rs:
            regions.append(name)
    return regions


def _parse_kml_fire_footprints(kml_bytes, bounds=None):
    """Parse fire footprint Placemarks from KML bytes into GeoJSON features.

    Each Placemark may contain a Polygon (footprint) or Point (centroid).
    Only Polygon placemarks are returned.  If *bounds* is given, features
    are filtered to the bounding box.
    """
    import xml.etree.ElementTree as ET

    root = ET.fromstring(kml_bytes)
    # KML namespace
    ns = ""
    if root.tag.startswith("{"):
        ns = root.tag.split("}")[0] + "}"

    features = []
    for pm in root.iter(f"{ns}Placemark"):
        # Look for Polygon geometry
        poly_el = pm.find(f".//{ns}Polygon")
        if poly_el is None:
            continue

        coords_el = poly_el.find(
            f".//{ns}outerBoundaryIs/{ns}LinearRing/{ns}coordinates"
        )
        if coords_el is None or not coords_el.text:
            continue

        # Parse "lon,lat,alt lon,lat,alt ..." into coordinate ring
        ring = []
        for triplet in coords_el.text.strip().split():
            parts = triplet.split(",")
            lon, lat = float(parts[0]), float(parts[1])
            ring.append([lon, lat])

        if not ring:
            continue

        # Filter by bounding box if provided
        if bounds is not None:
            west, south, east, north = bounds
            if not any(west <= c[0] <= east and south <= c[1] <= north
                       for c in ring):
                continue

        # Extract name / description as properties
        props = {}
        name_el = pm.find(f"{ns}name")
        if name_el is not None and name_el.text:
            props["name"] = name_el.text.strip()
        desc_el = pm.find(f"{ns}description")
        if desc_el is not None and desc_el.text:
            desc = desc_el.text.strip()
            props["description"] = desc
            # Parse structured HTML fields like "<b>Key: </b> Value<br/>"
            for m in re.finditer(
                r"<b>\s*([^<:]+?)\s*:\s*</b>\s*([^<]+)", desc
            ):
                key = m.group(1).strip().lower().replace(" ", "_")
                val = m.group(2).strip()
                props[key] = val

        features.append({
            "type": "Feature",
            "geometry": {
                "type": "Polygon",
                "coordinates": [ring],
            },
            "properties": props,
        })

    return features


def fetch_firms(bounds, date_span="24h", region=None, cache_path=None,
                crs=None):
    """Download NASA FIRMS LANDSAT 30 m fire detection footprints.

    Fetches fire footprint polygons from the `FIRMS KML fire footprints
    API <https://firms.modaps.eosdis.nasa.gov/api/kml_fire_footprints/>`_
    using the LANDSAT sensor (30 m resolution).

    Parameters
    ----------
    bounds : tuple of float
        (west, south, east, north) in WGS84 degrees.
    date_span : str
        Time window: ``'24h'``, ``'48h'``, ``'72h'``, or ``'7d'``.
        Default ``'24h'``.
    region : str, optional
        FIRMS region name (e.g. ``'usa_contiguous_and_hawaii'``).
        If ``None``, the region is auto-detected from *bounds*.
        When bounds span multiple regions, all matching regions are
        queried.
    cache_path : str or Path, optional
        Path to cache the final GeoJSON result.  If the file already
        exists, loads and returns it directly.
    crs : str, optional
        Target CRS for reprojection (e.g. ``'EPSG:32611'``).
        ``None`` keeps WGS84.

    Returns
    -------
    dict
        GeoJSON FeatureCollection.  Each feature is a 30 m fire
        detection polygon with ``name`` and ``description`` properties
        from the FIRMS KML.

    Examples
    --------
    >>> from rtxpy import fetch_firms
    >>> fires = fetch_firms((-118.6, 34.0, -118.1, 34.3), date_span='7d')
    >>> dem.rtx.place_geojson(fires, height=50.0, color='red')
    """
    import json
    import zipfile
    import io

    try:
        import requests
    except ImportError:
        raise ImportError(
            "requests is required for fetch_firms(). "
            "Install it with: pip install requests"
        )

    if cache_path is not None:
        cache_path = Path(cache_path)
        if cache_path.exists():
            print(f"Using cached FIRMS data: {cache_path.name}")
            with open(cache_path) as f:
                return json.load(f)

    valid_spans = ("24h", "48h", "72h", "7d")
    if date_span not in valid_spans:
        raise ValueError(
            f"Unknown date_span {date_span!r}; use one of {valid_spans}"
        )

    # Determine which FIRMS region(s) to query
    if region is not None:
        if region not in _FIRMS_REGION_BOUNDS:
            raise ValueError(
                f"Unknown FIRMS region {region!r}; choose from: "
                + ", ".join(sorted(_FIRMS_REGION_BOUNDS))
            )
        regions = [region]
    else:
        regions = _firms_regions_for_bounds(bounds)
        if not regions:
            print("No FIRMS region covers the given bounding box")
            return {"type": "FeatureCollection", "features": []}

    print(f"Fetching FIRMS LANDSAT fire footprints ({date_span}) "
          f"for {len(regions)} region(s)...")

    all_features = []
    for rgn in regions:
        url = (f"{_FIRMS_BASE_URL}/?region={rgn}"
               f"&date_span={date_span}&sensor=landsat")
        print(f"  Downloading {rgn}...")
        try:
            resp = requests.get(url, timeout=120)
            resp.raise_for_status()
        except Exception as e:
            print(f"  Warning: Failed to download {rgn}: {e}")
            continue

        # KMZ is a zip archive containing a .kml file
        try:
            with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
                kml_names = [n for n in zf.namelist()
                             if n.lower().endswith(".kml")]
                if not kml_names:
                    print(f"  Warning: No KML found in {rgn} KMZ")
                    continue
                kml_bytes = zf.read(kml_names[0])
        except zipfile.BadZipFile:
            # Response might be raw KML (not zipped)
            kml_bytes = resp.content

        features = _parse_kml_fire_footprints(kml_bytes, bounds=bounds)
        all_features.extend(features)
        print(f"    {len(features)} fire footprints in bounding box")

    print(f"  Total: {len(all_features)} fire footprints")

    # Reproject if requested
    if crs is not None and all_features:
        try:
            import geopandas as gpd
        except ImportError:
            raise ImportError(
                "geopandas is required for CRS reprojection. "
                "Install with: pip install geopandas"
            )
        gdf = gpd.GeoDataFrame.from_features(all_features, crs="EPSG:4326")
        gdf = gdf.to_crs(crs)
        print(f"  Reprojected to {crs}")
        geojson = json.loads(gdf.to_json())
    else:
        geojson = {"type": "FeatureCollection", "features": all_features}

    # Cache result
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "w") as f:
            json.dump(geojson, f)
        print(f"  Cached to {cache_path}")

    return geojson


# ---------------------------------------------------------------------------
# GTFS Transit Feeds
# ---------------------------------------------------------------------------

_GTFS_ROUTE_TYPE_NAMES = {
    0: 'tram', 1: 'subway', 2: 'rail', 3: 'bus', 4: 'ferry',
    5: 'tram', 6: 'gondola', 7: 'funicular',
    # Extended route types (hundreds)
    100: 'rail', 200: 'rail', 400: 'subway', 700: 'bus', 900: 'tram',
    1000: 'ferry', 1100: 'bus', 1300: 'gondola', 1400: 'funicular',
}


def _gtfs_route_type_name(route_type):
    """Map GTFS route_type int to a human-readable category name."""
    rt = int(route_type)
    if rt in _GTFS_ROUTE_TYPE_NAMES:
        return _GTFS_ROUTE_TYPE_NAMES[rt]
    # Extended types (>= 100): use hundreds group
    if rt >= 100:
        hundreds = (rt // 100) * 100
        return _GTFS_ROUTE_TYPE_NAMES.get(hundreds, 'other')
    return 'other'


def _discover_gtfs_feeds(bounds, cache_dir):
    """Query the Mobility Database CSV catalog to find GTFS feeds overlapping bounds.

    Returns list of dicts with keys: feed_id, provider, feed_url, bbox.
    """
    import requests
    import csv
    import io

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    catalog_path = cache_dir / 'mobility_database_catalogs.csv'

    # Download catalog if not cached (refresh weekly)
    import time
    if not catalog_path.exists() or (time.time() - catalog_path.stat().st_mtime > 7 * 86400):
        print("  Downloading Mobility Database catalog...")
        url = "https://bit.ly/catalogs-csv"
        resp = requests.get(url, timeout=60, allow_redirects=True)
        resp.raise_for_status()
        catalog_path.write_bytes(resp.content)
        print(f"  Catalog cached to {catalog_path}")

    # Parse catalog and find overlapping feeds
    west, south, east, north = bounds
    matches = []
    with open(catalog_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Only GTFS (not GTFS-RT)
            data_type = row.get('data_type', '')
            if data_type != 'gtfs':
                continue
            status = row.get('status', '')
            if status not in ('', 'active'):
                continue
            # Check bounding box overlap
            try:
                feed_west = float(row.get('location.bounding_box.minimum_longitude', ''))
                feed_south = float(row.get('location.bounding_box.minimum_latitude', ''))
                feed_east = float(row.get('location.bounding_box.maximum_longitude', ''))
                feed_north = float(row.get('location.bounding_box.maximum_latitude', ''))
            except (ValueError, TypeError):
                continue
            # Check overlap
            if feed_east < west or feed_west > east or feed_north < south or feed_south > north:
                continue
            feed_url = row.get('urls.latest', '') or row.get('urls.direct_download', '')
            if not feed_url:
                continue
            # Compute IoU (intersection over union) for ranking
            # This prefers feeds whose bbox tightly matches the query
            ow = max(0, min(east, feed_east) - max(west, feed_west))
            oh = max(0, min(north, feed_north) - max(south, feed_south))
            intersection = ow * oh
            query_area = (east - west) * (north - south)
            feed_area = max(1e-12, (feed_east - feed_west) * (feed_north - feed_south))
            union = query_area + feed_area - intersection
            iou = intersection / union if union > 0 else 0
            provider = row.get('provider', 'Unknown')
            feed_id = row.get('mdb_source_id', '')
            matches.append({
                'feed_id': feed_id,
                'provider': provider,
                'feed_url': feed_url,
                'bbox': (feed_west, feed_south, feed_east, feed_north),
                'overlap': intersection,
                'iou': iou,
            })

    # Sort by IoU (best spatial match first)
    matches.sort(key=lambda m: m['iou'], reverse=True)
    return matches


def _parse_gtfs_zip(zip_path_or_bytes, bounds, route_types=None,
                    include_stops=True):
    """Parse a GTFS ZIP file and return routes/stops as GeoJSON.

    Parameters
    ----------
    zip_path_or_bytes : str, Path, or bytes
        Path to GTFS ZIP or raw bytes.
    bounds : tuple
        (west, south, east, north) for spatial filtering.
    route_types : list of int, optional
        Filter to these GTFS route_type values.
    include_stops : bool
        Whether to also extract stops.

    Returns
    -------
    dict
        ``{'routes': FeatureCollection, 'stops': FeatureCollection,
           'metadata': {...}}``
    """
    import zipfile
    import io

    try:
        import pandas as pd
    except ImportError:
        raise ImportError(
            "pandas is required for GTFS parsing. "
            "Install with: pip install pandas"
        )

    west, south, east, north = bounds

    # Open ZIP
    if isinstance(zip_path_or_bytes, (str, Path)):
        zf = zipfile.ZipFile(zip_path_or_bytes, 'r')
    else:
        zf = zipfile.ZipFile(io.BytesIO(zip_path_or_bytes), 'r')

    zip_names = set(zf.namelist())

    def _read_csv(name):
        if name not in zip_names:
            return None
        with zf.open(name) as f:
            return pd.read_csv(f, dtype=str, keep_default_na=False)

    routes_df = _read_csv('routes.txt')
    trips_df = _read_csv('trips.txt')
    shapes_df = _read_csv('shapes.txt')
    stops_df = _read_csv('stops.txt')
    stop_times_df = _read_csv('stop_times.txt')

    zf.close()

    if routes_df is None:
        raise ValueError("GTFS ZIP missing routes.txt")

    # Normalise route_type to int
    if 'route_type' in routes_df.columns:
        routes_df['route_type'] = pd.to_numeric(routes_df['route_type'],
                                                 errors='coerce').fillna(3).astype(int)

    # Filter route types if requested
    if route_types is not None:
        routes_df = routes_df[routes_df['route_type'].isin(route_types)]

    # Build route info lookup
    route_info = {}
    for _, r in routes_df.iterrows():
        rid = r.get('route_id', '')
        route_info[rid] = {
            'route_type': int(r.get('route_type', 3)),
            'route_color': r.get('route_color', ''),
            'route_short_name': r.get('route_short_name', ''),
            'route_long_name': r.get('route_long_name', ''),
        }

    # --- Build route LineStrings ---
    route_features = []

    if shapes_df is not None and len(shapes_df) > 0:
        # Join shapes -> trips -> routes
        shapes_df['shape_pt_lat'] = pd.to_numeric(shapes_df['shape_pt_lat'],
                                                    errors='coerce')
        shapes_df['shape_pt_lon'] = pd.to_numeric(shapes_df['shape_pt_lon'],
                                                    errors='coerce')
        shapes_df['shape_pt_sequence'] = pd.to_numeric(
            shapes_df['shape_pt_sequence'], errors='coerce')
        shapes_df = shapes_df.dropna(subset=['shape_pt_lat', 'shape_pt_lon',
                                              'shape_pt_sequence'])

        # Get shape_id -> route_id mapping via trips
        shape_route = {}
        if trips_df is not None:
            for _, t in trips_df.iterrows():
                sid = t.get('shape_id', '')
                rid = t.get('route_id', '')
                if sid and rid and rid in route_info:
                    if sid not in shape_route:
                        shape_route[sid] = rid

        # Build LineStrings from shapes
        for shape_id, group in shapes_df.groupby('shape_id'):
            group = group.sort_values('shape_pt_sequence')
            coords = list(zip(group['shape_pt_lon'].values,
                               group['shape_pt_lat'].values))
            if len(coords) < 2:
                continue

            # Spatial filter: check if any point in bounds
            lons = group['shape_pt_lon'].values
            lats = group['shape_pt_lat'].values
            if lons.max() < west or lons.min() > east:
                continue
            if lats.max() < south or lats.min() > north:
                continue

            rid = shape_route.get(shape_id, '')
            props = dict(route_info.get(rid, {}))
            props['route_id'] = rid
            props['shape_id'] = shape_id
            props.setdefault('route_type', 3)

            route_features.append({
                'type': 'Feature',
                'geometry': {'type': 'LineString', 'coordinates': coords},
                'properties': props,
            })

    elif stop_times_df is not None and stops_df is not None:
        # Fallback: build shapes from stop sequences
        print("  No shapes.txt -- building routes from stop_times.txt")
        stops_df['stop_lat'] = pd.to_numeric(stops_df['stop_lat'],
                                              errors='coerce')
        stops_df['stop_lon'] = pd.to_numeric(stops_df['stop_lon'],
                                              errors='coerce')
        stop_coords = {}
        for _, s in stops_df.iterrows():
            sid = s.get('stop_id', '')
            lat = s['stop_lat']
            lon = s['stop_lon']
            if pd.notna(lat) and pd.notna(lon):
                stop_coords[sid] = (lon, lat)

        if trips_df is not None:
            trip_route = {}
            for _, t in trips_df.iterrows():
                trip_route[t.get('trip_id', '')] = t.get('route_id', '')

            stop_times_df['stop_sequence'] = pd.to_numeric(
                stop_times_df['stop_sequence'], errors='coerce')

            # Group by trip, build linestring per trip, deduplicate by route
            seen_routes = set()
            for trip_id, group in stop_times_df.groupby('trip_id'):
                rid = trip_route.get(trip_id, '')
                if rid in seen_routes or rid not in route_info:
                    continue
                seen_routes.add(rid)
                group = group.sort_values('stop_sequence')
                coords = []
                for _, st in group.iterrows():
                    c = stop_coords.get(st.get('stop_id', ''))
                    if c:
                        coords.append(c)
                if len(coords) < 2:
                    continue
                # Spatial filter
                lons = [c[0] for c in coords]
                lats = [c[1] for c in coords]
                if max(lons) < west or min(lons) > east:
                    continue
                if max(lats) < south or min(lats) > north:
                    continue
                props = dict(route_info.get(rid, {}))
                props['shape_id'] = f'trip_{trip_id}'
                route_features.append({
                    'type': 'Feature',
                    'geometry': {'type': 'LineString', 'coordinates': coords},
                    'properties': props,
                })

    # --- Build stop Points ---
    stop_features = []
    if include_stops and stops_df is not None:
        stops_df['stop_lat'] = pd.to_numeric(
            stops_df.get('stop_lat', pd.Series(dtype=float)), errors='coerce')
        stops_df['stop_lon'] = pd.to_numeric(
            stops_df.get('stop_lon', pd.Series(dtype=float)), errors='coerce')
        stops_df = stops_df.dropna(subset=['stop_lat', 'stop_lon'])

        # Determine which route types and colours serve each stop
        stop_route_types = {}
        stop_route_colors = {}  # stop_id -> set of route_color hex strings
        if stop_times_df is not None and trips_df is not None:
            trip_route_info = {}  # trip_id -> (route_type, route_color)
            for _, t in trips_df.iterrows():
                rid = t.get('route_id', '')
                info = route_info.get(rid)
                if info:
                    trip_route_info[t.get('trip_id', '')] = (
                        info['route_type'], info.get('route_color', ''))
            for _, st in stop_times_df.iterrows():
                sid = st.get('stop_id', '')
                tid = st.get('trip_id', '')
                ri = trip_route_info.get(tid)
                if ri is not None:
                    stop_route_types.setdefault(sid, set()).add(ri[0])
                    rc = ri[1].strip().lstrip('#')
                    if len(rc) == 6:
                        stop_route_colors.setdefault(sid, set()).add(rc.upper())

        for _, s in stops_df.iterrows():
            lat, lon = s['stop_lat'], s['stop_lon']
            if lon < west or lon > east or lat < south or lat > north:
                continue
            sid = s.get('stop_id', '')
            rts = sorted(stop_route_types.get(sid, set()))
            # Filter stops to only those serving requested route types
            if route_types is not None and rts:
                if not any(rt in route_types for rt in rts):
                    continue
            rcs = sorted(stop_route_colors.get(sid, set()))
            stop_features.append({
                'type': 'Feature',
                'geometry': {'type': 'Point', 'coordinates': [lon, lat]},
                'properties': {
                    'stop_name': s.get('stop_name', ''),
                    'stop_id': sid,
                    'route_types': rts,
                    'route_colors': rcs,
                },
            })

    # Metadata
    rt_counts = {}
    for f in route_features:
        rt = f['properties'].get('route_type', 3)
        name = _gtfs_route_type_name(rt)
        rt_counts[name] = rt_counts.get(name, 0) + 1

    metadata = {
        'n_routes': len(route_features),
        'n_stops': len(stop_features),
        'route_type_counts': rt_counts,
    }

    return {
        'routes': {'type': 'FeatureCollection', 'features': route_features},
        'stops': {'type': 'FeatureCollection', 'features': stop_features},
        'metadata': metadata,
    }


def fetch_gtfs(bounds, source='auto', feed_url=None, gtfs_path=None,
               route_types=None, cache_path=None, crs=None,
               include_stops=True, realtime_url=None):
    """Download and parse GTFS transit data for a bounding box.

    Discovers a GTFS feed from the Mobility Database, or uses a
    user-provided feed URL or local ZIP file.  Returns route shapes
    as LineStrings and stops as Points.

    Parameters
    ----------
    bounds : tuple of float
        (west, south, east, north) in WGS84 degrees.
    source : str
        ``'auto'`` (default) discovers a feed via the Mobility Database
        catalog.  Ignored when ``feed_url`` or ``gtfs_path`` is given.
    feed_url : str, optional
        Direct URL to a GTFS ZIP file.
    gtfs_path : str or Path, optional
        Path to a local GTFS ZIP file.
    route_types : list of int, optional
        Filter to specific GTFS route types (0=tram, 1=subway, 2=rail,
        3=bus, 4=ferry, etc.).  ``None`` returns all.
    cache_path : str or Path, optional
        Path to cache the parsed result as JSON.  If the file already
        exists, loads and returns it directly.
    crs : str, optional
        Target CRS for reprojection (e.g. ``'EPSG:32618'``).
        ``None`` keeps WGS84.
    include_stops : bool
        Whether to include stop points.  Default ``True``.
    realtime_url : str, optional
        URL to a GTFS-Realtime VehiclePositions protobuf feed.
        Stored in the returned metadata for use by ``explore()``.

    Returns
    -------
    dict
        ``{'routes': FeatureCollection, 'stops': FeatureCollection,
           'metadata': {...}}``.

    Examples
    --------
    >>> from rtxpy import fetch_gtfs
    >>> gtfs = fetch_gtfs((-74.05, 40.68, -73.90, 40.82))
    >>> dem.rtx.place_gtfs(gtfs)
    >>> dem.rtx.explore()
    """
    try:
        import requests
    except ImportError:
        raise ImportError(
            "requests is required for fetch_gtfs(). "
            "Install with: pip install requests"
        )

    # Check cache
    if cache_path is not None:
        cache_path = Path(cache_path)
        if cache_path.exists():
            print(f"Using cached GTFS data: {cache_path.name}")
            with open(cache_path) as f:
                return json.load(f)

    cache_dir = Path.home() / '.cache' / 'rtxpy' / 'gtfs'
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Determine ZIP source
    zip_bytes = None
    feed_name = None

    if gtfs_path is not None:
        # Local file
        gtfs_path = Path(gtfs_path)
        print(f"Loading GTFS from {gtfs_path}")
        feed_name = gtfs_path.stem
        result = _parse_gtfs_zip(gtfs_path, bounds,
                                  route_types=route_types,
                                  include_stops=include_stops)

    elif feed_url is not None:
        # Direct URL
        print(f"Downloading GTFS feed: {feed_url}")
        resp = requests.get(feed_url, timeout=120)
        resp.raise_for_status()
        zip_bytes = resp.content
        feed_name = feed_url.split('/')[-1].replace('.zip', '')
        # Cache the ZIP
        zip_cache = cache_dir / f'{feed_name}.zip'
        zip_cache.write_bytes(zip_bytes)
        result = _parse_gtfs_zip(zip_bytes, bounds,
                                  route_types=route_types,
                                  include_stops=include_stops)

    else:
        # Auto-discover from Mobility Database
        print(f"Discovering GTFS feeds for bounds {bounds}...")
        feeds = _discover_gtfs_feeds(bounds, cache_dir)
        if not feeds:
            print("  No GTFS feeds found for this area.")
            return {
                'routes': {'type': 'FeatureCollection', 'features': []},
                'stops': {'type': 'FeatureCollection', 'features': []},
                'metadata': {'n_routes': 0, 'n_stops': 0,
                             'route_type_counts': {}},
            }

        best = feeds[0]
        feed_name = best['provider']
        print(f"  Best match: {best['provider']} (id={best['feed_id']})")
        print(f"  Downloading {best['feed_url']}...")

        resp = requests.get(best['feed_url'], timeout=120)
        resp.raise_for_status()
        zip_bytes = resp.content

        # Cache the ZIP
        safe_name = re.sub(r'[^\w\-.]', '_', best['provider'])
        zip_cache = cache_dir / f'{safe_name}_{best["feed_id"]}.zip'
        zip_cache.write_bytes(zip_bytes)
        print(f"  ZIP cached to {zip_cache}")

        result = _parse_gtfs_zip(zip_bytes, bounds,
                                  route_types=route_types,
                                  include_stops=include_stops)

    # Add feed name and realtime URL to metadata
    result['metadata']['feed_name'] = feed_name or ''
    if realtime_url:
        result['metadata']['realtime_url'] = realtime_url

    meta = result['metadata']
    print(f"  {meta['n_routes']} route shapes, {meta['n_stops']} stops")
    if meta['route_type_counts']:
        parts = [f"{v} {k}" for k, v in sorted(meta['route_type_counts'].items())]
        print(f"  Types: {', '.join(parts)}")

    # Reproject if requested
    if crs is not None:
        try:
            import geopandas as gpd
        except ImportError:
            raise ImportError(
                "geopandas is required for CRS reprojection. "
                "Install with: pip install geopandas"
            )
        for key in ('routes', 'stops'):
            fc = result[key]
            if fc['features']:
                gdf = gpd.GeoDataFrame.from_features(fc['features'],
                                                      crs="EPSG:4326")
                gdf = gdf.to_crs(crs)
                result[key] = json.loads(gdf.to_json())
        print(f"  Reprojected to {crs}")

    # Cache result
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, 'w') as f:
            json.dump(result, f)
        print(f"  Cached to {cache_path}")

    return result
