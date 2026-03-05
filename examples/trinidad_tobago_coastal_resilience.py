"""Trinidad & Tobago Coastal Resilience — Storm Surge Impact Analysis.

Model storm-surge and sea-level-rise scenarios across Trinidad & Tobago.
Copernicus 30 m elevation data is used to identify which OSM buildings and
roads would be inundated at progressively higher water levels — highlighting
low-water crossings and coastal infrastructure most at risk.

Storm-surge scenarios: 1 m, 2 m, 3 m, 5 m, 10 m above present sea level.
Infrastructure is colour-coded by the lowest surge level that would flood it:

    dark purple → 1 m  (extreme – low-water crossings, immediate flood zone)
    vermillion → 2 m  (very high)
    orange     → 3 m  (high)
    sky blue   → 5 m  (moderate)
    bluish grn → 10 m (low)
    grey       → safe (above 10 m)

Dataset layers (press G to cycle):
    elevation  – terrain height (scaled)
    surge_risk – graduated: lowest surge level to flood each pixel
    flood_1m … flood_10m – water rendering per scenario (rising flood)

Flood model: simple bathtub inundation — each pixel with Copernicus DEM
elevation <= surge level is marked as flooded. No hydrodynamic connectivity
or wave run-up; assumes static uniform water surface. Conservative for open
coast (ignores wave setup) but optimistic for inland areas (ignores drainage
blockage). Buildings classified by centroid elevation, roads by their lowest
vertex (low-water crossing analysis).

Press U for satellite overlay, Shift+W for wind animation, O/V for viewshed.

Requirements:
    pip install rtxpy[all] matplotlib xarray rioxarray requests pyproj Pillow
"""

import warnings

import numpy as np
import xarray as xr
from pathlib import Path
from pyproj import Transformer

from rtxpy import fetch_dem, fetch_buildings, fetch_roads, fetch_water

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
BOUNDS = (-61.95, 10.04, -60.44, 11.40)    # both islands (WGS 84)
CRS = 'EPSG:32620'                          # UTM zone 20N
CACHE = Path(__file__).parent
ZARR = CACHE / "tt_coastal_dem.zarr"
SURGE_LEVELS = [1, 2, 3, 5, 10]             # metres above sea level

RISK_COLORS = {
    1:  (0.50, 0.00, 0.50),    # dark purple  – extreme
    2:  (0.84, 0.19, 0.12),    # vermillion   – very high
    3:  (0.90, 0.60, 0.00),    # orange        – high
    5:  (0.34, 0.71, 0.91),    # sky blue      – moderate
    10: (0.00, 0.62, 0.45),    # bluish green  – low
}

BLDG_HEIGHT = 8   # default 8 m building height


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _sample_dem(raw_elev, lons, lats):
    """Return ground elevation (m) at each (lon, lat) via nearest-pixel lookup."""
    transformer = Transformer.from_crs("EPSG:4326", CRS, always_xy=True)
    pxs, pys = transformer.transform(
        np.asarray(lons, dtype=np.float64),
        np.asarray(lats, dtype=np.float64),
    )
    pxs, pys = np.asarray(pxs), np.asarray(pys)

    xs = raw_elev.x.values.copy()
    ys = raw_elev.y.values.copy()
    elev = raw_elev.values

    # Ensure ascending order for searchsorted
    if xs[-1] < xs[0]:
        xs = xs[::-1]
        elev = elev[:, ::-1]
    if ys[-1] < ys[0]:
        ys = ys[::-1]
        elev = elev[::-1, :]

    def _nearest(arr, vals):
        idx = np.searchsorted(arr, vals).clip(0, len(arr) - 1)
        left = np.maximum(idx - 1, 0)
        return np.where(np.abs(arr[left] - vals) < np.abs(arr[idx] - vals), left, idx)

    return elev[_nearest(ys, pys), _nearest(xs, pxs)]


def _fc(features):
    """Wrap features in a GeoJSON FeatureCollection."""
    return {"type": "FeatureCollection", "features": features}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_terrain():
    """Fetch Copernicus DEM; return (scaled GPU DataArray, raw CPU DataArray)."""
    raw = fetch_dem(
        bounds=BOUNDS,
        output_path=ZARR,
        source='copernicus',
        crs=CRS,
    )
    raw_elev = raw.copy(deep=True)

    terrain = raw.copy(deep=True)
    terrain.data = np.ascontiguousarray(terrain.data)
    emin, emax = float(np.nanmin(raw_elev.data)), float(np.nanmax(raw_elev.data))
    print(f"Terrain: {terrain.shape}, elevation {emin:.1f} m to {emax:.1f} m")

    terrain = terrain.rtx.to_cupy()
    return terrain, raw_elev


# ---------------------------------------------------------------------------
# Surge analysis — raster layers
# ---------------------------------------------------------------------------
def build_surge_layers(raw_elev):
    """Create raster flood-zone layers from unscaled elevation.

    Returns a dict of DataArrays ready for the Dataset.
    """
    elev = raw_elev.values
    land = ~np.isnan(elev)          # True where we have elevation data
    layers = {}

    # Graduated layer: lowest surge level that floods each pixel
    risk = np.full_like(elev, np.nan)
    for level in reversed(SURGE_LEVELS):
        risk = np.where(land & (elev <= level), float(level), risk)
    layers['surge_risk'] = xr.DataArray(
        np.ascontiguousarray(risk), coords=raw_elev.coords, dims=raw_elev.dims,
    )

    # Per-scenario flood masks (1 = flooded, NaN = everything else)
    pixel_km2 = 30 * 30 / 1e6
    print("\nStorm-surge inundation area:")
    for level in SURGE_LEVELS:
        flooded = land & (elev <= level)
        area = float(flooded.sum()) * pixel_km2
        print(f"  {level:>2d} m surge: {area:>8.1f} km² inundated")
        mask = np.where(flooded, 1.0, np.nan)
        layers[f'flood_{level}m'] = xr.DataArray(
            np.ascontiguousarray(mask), coords=raw_elev.coords, dims=raw_elev.dims,
        )

    return layers


# ---------------------------------------------------------------------------
# Infrastructure classification
# ---------------------------------------------------------------------------
def classify_buildings(bldg_data, raw_elev):
    """Split buildings into risk bins by centroid ground elevation."""
    features = bldg_data.get('features', [])
    if not features:
        return {}

    # Compute centroids from exterior ring
    lons, lats = [], []
    for f in features:
        coords = f['geometry']['coordinates']
        ring = coords[0][0] if f['geometry']['type'] == 'MultiPolygon' else coords[0]
        lons.append(sum(c[0] for c in ring) / len(ring))
        lats.append(sum(c[1] for c in ring) / len(ring))

    elevs = _sample_dem(raw_elev, lons, lats)

    bins = {lv: [] for lv in SURGE_LEVELS}
    bins['safe'] = []
    for i, f in enumerate(features):
        e = elevs[i]
        if np.isnan(e):
            bins['safe'].append(f)
            continue
        placed = False
        for lv in SURGE_LEVELS:
            if e <= lv:
                bins[lv].append(f)
                placed = True
                break
        if not placed:
            bins['safe'].append(f)

    print("\nBuildings by storm-surge risk:")
    for lv in SURGE_LEVELS:
        n = len(bins[lv])
        if n:
            print(f"  ≤ {lv:>2d} m: {n:>7,d} buildings")
    print(f"   safe:  {len(bins['safe']):>7,d} buildings")
    return bins


def classify_roads(road_data, raw_elev):
    """Split roads into risk bins by minimum vertex elevation (low-water crossings)."""
    features = road_data.get('features', [])
    if not features:
        return {}

    # Gather every vertex with its parent feature index
    all_lons, all_lats, feat_idx = [], [], []
    for i, f in enumerate(features):
        geom = f['geometry']
        if geom['type'] == 'LineString':
            lines = [geom['coordinates']]
        elif geom['type'] == 'MultiLineString':
            lines = geom['coordinates']
        else:
            continue  # skip Point, Polygon, etc.
        for line in lines:
            for coord in line:
                all_lons.append(coord[0])
                all_lats.append(coord[1])
                feat_idx.append(i)

    if not all_lons:
        return {}

    all_elevs = _sample_dem(raw_elev, all_lons, all_lats)
    feat_idx = np.asarray(feat_idx)

    # Minimum elevation per road segment (the low-water crossing point)
    min_elev = np.full(len(features), np.inf)
    valid = ~np.isnan(all_elevs)
    np.minimum.at(min_elev, feat_idx[valid], all_elevs[valid])

    bins = {lv: [] for lv in SURGE_LEVELS}
    bins['safe'] = []
    for i, f in enumerate(features):
        e = min_elev[i]
        if np.isinf(e):
            bins['safe'].append(f)
            continue
        placed = False
        for lv in SURGE_LEVELS:
            if e <= lv:
                bins[lv].append(f)
                placed = True
                break
        if not placed:
            bins['safe'].append(f)

    print("\nRoads by storm-surge risk (low-water crossing analysis):")
    for lv in SURGE_LEVELS:
        n = len(bins[lv])
        if n:
            print(f"  ≤ {lv:>2d} m: {n:>7,d} road segments")
    print(f"   safe:  {len(bins['safe']):>7,d} road segments")
    return bins


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    terrain, raw_elev = load_terrain()

    # --- Derived raster layers ---------------------------------------------
    print("Computing terrain analysis layers...")
    surge_layers = build_surge_layers(raw_elev)

    ds = xr.Dataset({
        'elevation': terrain.rename(None),
        **{k: v.rtx.to_cupy() for k, v in surge_layers.items()},
    })
    print(ds)

    # --- Satellite tiles ---------------------------------------------------
    print("\nLoading satellite tiles...")
    ds.rtx.place_tiles('satellite', z='elevation')

    # --- Load meshes from zarr cache, or build from GeoJSON -----------------
    import zarr as _zarr
    _has_mesh_cache = False
    try:
        _store = _zarr.open(str(ZARR), mode='r', use_consolidated=False)
        _has_mesh_cache = 'meshes' in _store and len(list(_store['meshes'])) > 0
        _store = None
    except Exception:
        pass

    if _has_mesh_cache:
        ds.rtx.load_meshes(ZARR)
    else:
        # --- Buildings colour-coded by risk --------------------------------
        try:
            bldg_data = fetch_buildings(
                bounds=BOUNDS, source='overture',
                cache_path=CACHE / "tt_coastal_buildings.geojson",
            )
            bldg_bins = classify_buildings(bldg_data, raw_elev)

            for lv in SURGE_LEVELS:
                if bldg_bins.get(lv):
                    with warnings.catch_warnings():
                        warnings.filterwarnings("ignore", message="place_geojson called before")
                        info = ds.rtx.place_geojson(
                            _fc(bldg_bins[lv]), z='elevation',
                            height=BLDG_HEIGHT, extrude=True, merge=True,
                            geometry_id=f'bldg_{lv}m',
                            color=RISK_COLORS[lv],
                        )
                        print(f"  Placed {info['geometries']} buildings at ≤{lv} m risk")

            if bldg_bins.get('safe'):
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", message="place_geojson called before")
                    info = ds.rtx.place_geojson(
                        _fc(bldg_bins['safe']), z='elevation',
                        height=BLDG_HEIGHT, extrude=True, merge=True,
                        geometry_id='bldg_safe',
                        color=(0.65, 0.65, 0.65),
                    )
                    print(f"  Placed {info['geometries']} safe buildings")
        except Exception as e:
            print(f"Skipping buildings: {e}")

        # --- Roads colour-coded by risk ------------------------------------
        try:
            road_data = fetch_roads(
                bounds=BOUNDS, road_type='all', source='overture',
                cache_path=CACHE / "tt_coastal_roads.geojson",
            )
            road_bins = classify_roads(road_data, raw_elev)

            for lv in SURGE_LEVELS:
                if road_bins.get(lv):
                    with warnings.catch_warnings():
                        warnings.filterwarnings("ignore", message="place_geojson called before")
                        info = ds.rtx.place_roads(
                            _fc(road_bins[lv]), z='elevation',
                            geometry_id=f'road_{lv}m',
                            color=RISK_COLORS[lv],
                        )
                        print(f"  Placed {info['geometries']} road segments at ≤{lv} m risk")

            if road_bins.get('safe'):
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", message="place_geojson called before")
                    info = ds.rtx.place_roads(
                        _fc(road_bins['safe']), z='elevation',
                        geometry_id='road_safe',
                        color=(0.30, 0.30, 0.30),
                    )
                    print(f"  Placed {info['geometries']} safe road segments")
        except Exception as e:
            print(f"Skipping roads: {e}")

        # --- Water features (coastline/rivers for context) -----------------
        try:
            water_data = fetch_water(
                bounds=BOUNDS, water_type='all',
                cache_path=CACHE / "tt_coastal_water.geojson",
            )
            results = ds.rtx.place_water(water_data, z='elevation')
            for cat, info in results.items():
                print(f"  Placed {info['geometries']} {cat} water features")
        except Exception as e:
            print(f"Skipping water: {e}")

        # --- Save all meshes to zarr for next run --------------------------
        try:
            ds.rtx.save_meshes(ZARR)
        except Exception as e:
            print(f"Could not save mesh cache: {e}")

    # --- Wind (storm context) ----------------------------------------------
    wind = None
    try:
        from rtxpy import fetch_wind
        wind = fetch_wind(BOUNDS, grid_size=15)
    except Exception as e:
        print(f"Skipping wind: {e}")

    # --- Weather (clouds + rain via Shift+N) -------------------------------
    weather = None
    try:
        from rtxpy import fetch_weather
        weather = fetch_weather(BOUNDS, grid_size=15)
    except Exception as e:
        print(f"Skipping weather: {e}")

    # --- Launch explorer ---------------------------------------------------
    print("\n" + "=" * 60)
    print("COASTAL RESILIENCE EXPLORER")
    print("=" * 60)
    print("  G        cycle layers (elevation → slope → surge_risk → flood maps)")
    print("  U        toggle satellite overlay")
    print("  Shift+W  toggle wind particles (storm simulation)")
    print("  Shift+N  toggle clouds + rain")
    print("  O / V    set observer / toggle viewshed")
    print("  M        minimap")
    print("  H        full help overlay")
    print("=" * 60)

    ds.rtx.explore(
        z='elevation',
        scene_zarr=ZARR,
        title='Trinidad & Tobago: Storm Surge',
        subtitle='Copernicus 30m DEM  \u00b7  Overture Maps  \u00b7  OSM',
        legend={
            'entries': [
                ('Extreme (\u2264 1 m)',    (0.50, 0.00, 0.50)),
                ('Very high (\u2264 2 m)',  (0.84, 0.19, 0.12)),
                ('High (\u2264 3 m)',       (0.90, 0.60, 0.00)),
                ('Moderate (\u2264 5 m)',   (0.34, 0.71, 0.91)),
                ('Low (\u2264 10 m)',       (0.00, 0.62, 0.45)),
                ('Safe',                     (0.65, 0.65, 0.65)),
            ],
        },
        width=2048,
        height=1600,
        render_scale=0.5,
        color_stretch='cbrt',
        subsample=1,
        wind_data=wind,
        weather_data=weather,
        minimap_style='cyberpunk',
        minimap_layer='surge_risk',
        minimap_colors=RISK_COLORS,
        info_text=(
            "Bathtub inundation model\n"
            "Copernicus 30m DEM \u2264 surge level = flooded\n"
            "No hydrodynamic connectivity or wave run-up\n"
            "Buildings classified by centroid elevation\n"
            "Roads by lowest vertex (low-water crossing)"
        ),
        repl=True,
    )

    print("Done")
