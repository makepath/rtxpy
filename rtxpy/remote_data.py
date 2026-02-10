"""Download remote geospatial data: DEM tiles, OSM features, buildings, roads, water, and fire.

Supports Copernicus GLO-30, USGS SRTM 1-arc-second, and USGS 3DEP
1-meter DEM sources, OpenStreetMap vector features via osmnx,
Microsoft Global ML Building Footprints, convenience wrappers
for roads and water features, and NASA FIRMS fire detection footprints.

All network dependencies (``requests``, ``rioxarray``, ``osmnx``,
``pandas``, ``geopandas``, ``shapely``) are optional and imported
lazily — a clear ``ImportError`` is raised at call time if a required
package is missing.
"""

import math
import re
from pathlib import Path


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

    merged.rio.to_raster(str(output_path))
    return merged


def fetch_dem(bounds, output_path, source="copernicus", crs=None, cache_dir=None):
    """Download, merge, and clip DEM tiles for a bounding box.

    Parameters
    ----------
    bounds : tuple of float
        (west, south, east, north) in WGS84 degrees.
    output_path : str or Path
        Where to save the final merged/clipped/reprojected GeoTIFF.
        If the file already exists, loads and returns it directly.
    source : str
        ``'copernicus'`` for Copernicus GLO-30 (30 m), ``'srtm'`` for
        USGS 1-arc-second (~30 m), or ``'usgs_1m'`` for USGS 3DEP
        1-meter lidar DEM (US coverage only, ~30 MB per 10 km tile).
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

    if output_path.exists():
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
    elif source == "usgs_1m":
        tiles = _query_usgs_1m_tiles(bounds)
        ext_prefix = ""
    else:
        raise ValueError(
            f"Unknown source {source!r}; use 'copernicus', 'srtm', or 'usgs_1m'"
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


def fetch_buildings(bounds, cache_path=None, crs=None, cache_dir=None):
    """Download Microsoft Global Building Footprints for a bounding box.

    Uses the `dataset-links.csv
    <https://github.com/microsoft/GlobalMLBuildingFootprints>`_ index to
    find level-9 quadkey partitions that overlap *bounds*, downloads the
    compressed GeoJSONL files, filters features to the bounding box, and
    returns a standard GeoJSON FeatureCollection.

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
    import json

    if cache_path is not None:
        cache_path = Path(cache_path)
        if cache_path.exists():
            print(f"Using cached buildings: {cache_path.name}")
            with open(cache_path) as f:
                return json.load(f)

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


def fetch_roads(bounds, road_type="all", cache_path=None, crs=None):
    """Download road data from OpenStreetMap for a bounding box.

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


def fetch_water(bounds, water_type="all", cache_path=None, crs=None):
    """Download water / waterway features from OpenStreetMap.

    Parameters
    ----------
    bounds : tuple of float
        (west, south, east, north) in WGS84 degrees.
    water_type : str
        Which features to include:

        - ``'waterway'`` — linear features: rivers, streams, canals, etc.
        - ``'waterbody'`` — area features: lakes, reservoirs, ponds
          (``natural=water``).
        - ``'all'`` (default) — both waterways and waterbodies.
    cache_path : str or Path, optional
        Path to cache the result as GeoJSON.  If the file already
        exists, loads and returns it directly.
    crs : str, optional
        Target CRS for reprojection (e.g. ``'EPSG:32620'``).
        ``None`` keeps WGS84.

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
