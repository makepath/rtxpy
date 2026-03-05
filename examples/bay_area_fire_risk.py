"""Bay Area Wildfire Risk Analysis — GPU ML with cuML + rtxpy.

Combines GPU-accelerated terrain analysis with RAPIDS cuML machine learning
to produce wildfire risk and urban analysis layers for the San Francisco
Bay Area.  The entire pipeline stays on GPU: DEM → cupy → cuML → explore().

Wind data from Open-Meteo is interpolated onto the DEM grid and used as
features in the fire risk model: wind speed amplifies fire spread, and
wind-aligned slopes (wind blowing uphill) are especially dangerous.

Analysis layers (press G to cycle):
    elevation      – raw terrain height
    slope          – steepness in degrees
    aspect         – downhill compass bearing
    terrain_class  – K-Means landform classification (6 classes)
    wind_speed     – interpolated 10 m wind speed (m/s)
    wind_exposure  – wind-slope alignment (positive = wind blowing uphill)
    fire_density   – KDE heatmap of recent FIRMS fire detections
    fire_risk      – Random Forest fire probability (terrain + wind + fires)
    urban_cluster  – HDBSCAN neighbourhood clustering from building centroids

Buildings are coloured by their HDBSCAN cluster (outliers rendered grey).
FIRMS fire detections are placed as glowing orange markers for ground truth.

Requirements:
    pip install rtxpy[all] cuml-cu12 xarray rioxarray requests pyproj Pillow
"""

import warnings

import numpy as np
import cupy as cp
import xarray as xr
from pathlib import Path
from pyproj import Transformer

from xrspatial import slope, aspect, quantile

from rtxpy import fetch_dem, fetch_buildings, fetch_roads, fetch_water, fetch_firms
import rtxpy

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
# Greater Bay Area — Marin Headlands to southern San Jose
BOUNDS = (-122.60, 37.20, -121.80, 37.90)
CRS = 'EPSG:32610'  # UTM zone 10N
CACHE = Path(__file__).parent
ZARR = CACHE / "bay_area_dem.zarr"

# K-Means
N_TERRAIN_CLASSES = 6

# KDE
KDE_BANDWIDTH = 1500.0

# Random Forest
RF_N_TREES = 100
RF_MAX_DEPTH = 10

# HDBSCAN building clustering
HDBSCAN_MIN_CLUSTER = 80
HDBSCAN_MIN_SAMPLES = 15

# 12 distinct cluster colours (colour-blind-friendly qualitative palette)
CLUSTER_COLORS = [
    (0.90, 0.25, 0.20),   # red
    (0.20, 0.60, 0.85),   # blue
    (0.30, 0.75, 0.40),   # green
    (0.95, 0.60, 0.15),   # orange
    (0.60, 0.35, 0.70),   # purple
    (0.95, 0.85, 0.20),   # yellow
    (0.65, 0.45, 0.25),   # brown
    (0.85, 0.45, 0.65),   # pink
    (0.45, 0.75, 0.75),   # teal
    (0.75, 0.75, 0.75),   # light grey
    (0.55, 0.55, 0.20),   # olive
    (0.40, 0.20, 0.55),   # dark purple
]
OUTLIER_COLOR = (0.35, 0.35, 0.35)  # dark grey for noise


# ---------------------------------------------------------------------------
# Terrain loading
# ---------------------------------------------------------------------------
def load_terrain():
    """Fetch USGS 10 m DEM, return GPU DataArray + CPU copy."""
    raw = fetch_dem(
        bounds=BOUNDS,
        output_path=ZARR,
        source='usgs_10m',
        crs=CRS,
    )
    raw_cpu = raw.copy(deep=True)

    raw.data = np.ascontiguousarray(raw.data)
    emin = float(np.nanmin(raw.data))
    emax = float(np.nanmax(raw.data))
    print(f"Terrain: {raw.shape}, elevation {emin:.0f} m to {emax:.0f} m")

    terrain = raw.rtx.to_cupy()
    return terrain, raw_cpu


# ---------------------------------------------------------------------------
# Wind interpolation onto DEM grid
# ---------------------------------------------------------------------------
def interpolate_wind(wind_data, terrain_da):
    """Interpolate coarse wind grid onto the full-resolution DEM grid.

    Uses scipy RegularGridInterpolator to bilinear-interpolate wind speed
    and direction from the Open-Meteo grid (~15x15) onto every DEM pixel.
    Returns cupy arrays: (wind_speed, wind_u, wind_v) on the DEM grid.
    """
    from scipy.interpolate import RegularGridInterpolator

    if wind_data is None:
        shape = terrain_da.shape
        z = cp.zeros(shape, dtype=cp.float32)
        return z, z, z

    # Wind grid is in WGS84 lat/lon — reproject DEM pixel coords to WGS84
    transformer = Transformer.from_crs(CRS, "EPSG:4326", always_xy=True)
    xs_crs = terrain_da.x.values
    ys_crs = terrain_da.y.values
    grid_x_crs, grid_y_crs = np.meshgrid(xs_crs, ys_crs)
    lons_dem, lats_dem = transformer.transform(
        grid_x_crs.ravel().astype(np.float64),
        grid_y_crs.ravel().astype(np.float64),
    )
    lons_dem = lons_dem.reshape(terrain_da.shape)
    lats_dem = lats_dem.reshape(terrain_da.shape)

    w_lats = wind_data['lats']
    w_lons = wind_data['lons']

    interp_speed = RegularGridInterpolator(
        (w_lats, w_lons), wind_data['speed'],
        method='linear', bounds_error=False, fill_value=None,
    )
    interp_u = RegularGridInterpolator(
        (w_lats, w_lons), wind_data['u'],
        method='linear', bounds_error=False, fill_value=None,
    )
    interp_v = RegularGridInterpolator(
        (w_lats, w_lons), wind_data['v'],
        method='linear', bounds_error=False, fill_value=None,
    )

    pts = np.column_stack([lats_dem.ravel(), lons_dem.ravel()])
    speed = interp_speed(pts).reshape(terrain_da.shape).astype(np.float32)
    u = interp_u(pts).reshape(terrain_da.shape).astype(np.float32)
    v = interp_v(pts).reshape(terrain_da.shape).astype(np.float32)

    avg_speed = float(np.nanmean(speed))
    print(f"  Wind interpolated to DEM grid: mean speed {avg_speed:.1f} m/s")

    return cp.asarray(speed), cp.asarray(u), cp.asarray(v)


def compute_wind_exposure(aspect_gpu, wind_u, wind_v):
    """Compute wind-slope alignment: positive when wind blows uphill.

    The aspect gives the downhill direction.  Wind blowing opposite to the
    downhill direction (i.e. uphill) drives fire spread.  We compute:
        exposure = -dot(wind_unit, downhill_unit)
    Ranges from -1 (wind blowing downhill) to +1 (wind blowing uphill).
    """
    asp_rad = cp.deg2rad(aspect_gpu)
    down_x = cp.sin(asp_rad)
    down_y = cp.cos(asp_rad)

    wind_mag = cp.sqrt(wind_u**2 + wind_v**2)
    wind_mag = cp.where(wind_mag == 0, 1.0, wind_mag)
    wu = wind_u / wind_mag
    wv = wind_v / wind_mag

    exposure = -(wu * down_x + wv * down_y)

    speed_norm = cp.sqrt(wind_u**2 + wind_v**2)
    speed_max = speed_norm.max()
    if speed_max > 0:
        exposure = exposure * (speed_norm / speed_max)

    return exposure


# ---------------------------------------------------------------------------
# cuML: K-Means terrain classification
# ---------------------------------------------------------------------------
def terrain_classification(ds):
    """Cluster pixels by (elevation, slope, aspect) into landform classes."""
    from cuml import KMeans

    elev = ds['elevation'].data
    slp  = ds['slope'].data
    asp  = ds['aspect'].data
    shape = elev.shape

    e_flat = elev.ravel()
    s_flat = slp.ravel()
    a_flat = asp.ravel()

    valid = ~(cp.isnan(e_flat) | cp.isnan(s_flat) | cp.isnan(a_flat))
    n_valid = int(valid.sum())
    print(f"  K-Means: {n_valid:,} valid pixels, {N_TERRAIN_CLASSES} classes")

    features = cp.stack([e_flat[valid], s_flat[valid], a_flat[valid]], axis=1)

    fmin = features.min(axis=0)
    fmax = features.max(axis=0)
    frange = fmax - fmin
    frange[frange == 0] = 1.0
    features = (features - fmin) / frange

    km = KMeans(n_clusters=N_TERRAIN_CLASSES, max_iter=300, random_state=42)
    labels = km.fit_predict(features)

    result = cp.full(e_flat.shape[0], cp.nan, dtype=cp.float32)
    result[valid] = labels.astype(cp.float32)
    result = result.reshape(shape)

    print(f"  K-Means done — cluster sizes: "
          f"{', '.join(str(int((labels == i).sum())) for i in range(N_TERRAIN_CLASSES))}")
    return result


# ---------------------------------------------------------------------------
# cuML: KDE fire density heatmap
# ---------------------------------------------------------------------------
def fire_density_layer(fire_data, terrain_da):
    """Build a fire density heatmap via cuml.neighbors.KernelDensity."""
    from cuml.neighbors import KernelDensity

    features = fire_data.get('features', [])
    if not features:
        print("  KDE: no fire detections — returning zeros")
        return cp.zeros(terrain_da.shape, dtype=cp.float32)

    centroids = []
    for f in features:
        geom = f['geometry']
        if geom['type'] == 'Point':
            centroids.append(geom['coordinates'][:2])
        elif geom['type'] in ('Polygon', 'MultiPolygon'):
            coords = geom['coordinates']
            ring = coords[0][0] if geom['type'] == 'MultiPolygon' else coords[0]
            cx = sum(c[0] for c in ring) / len(ring)
            cy = sum(c[1] for c in ring) / len(ring)
            centroids.append([cx, cy])
    centroids = np.array(centroids, dtype=np.float64)

    if fire_data.get('crs') is None or 'EPSG:4326' in str(fire_data.get('crs', '')):
        transformer = Transformer.from_crs("EPSG:4326", CRS, always_xy=True)
        xs, ys = transformer.transform(centroids[:, 0], centroids[:, 1])
        centroids = np.column_stack([xs, ys])

    print(f"  KDE: fitting on {len(centroids)} fire centroids, "
          f"bandwidth={KDE_BANDWIDTH:.0f} m")

    centroids_gpu = cp.asarray(centroids, dtype=cp.float32)
    kde = KernelDensity(bandwidth=KDE_BANDWIDTH, kernel='gaussian')
    kde.fit(centroids_gpu)

    xs = cp.asarray(terrain_da.x.values, dtype=cp.float32)
    ys = cp.asarray(terrain_da.y.values, dtype=cp.float32)
    grid_x, grid_y = cp.meshgrid(xs, ys)
    query = cp.stack([grid_x.ravel(), grid_y.ravel()], axis=1)

    log_density = kde.score_samples(query)
    density = cp.exp(log_density).reshape(terrain_da.shape)

    dmin, dmax = density.min(), density.max()
    if dmax > dmin:
        density = (density - dmin) / (dmax - dmin)

    elev = terrain_da.data if hasattr(terrain_da.data, '__cuda_array_interface__') \
        else cp.asarray(terrain_da.data)
    density = cp.where(cp.isnan(elev), cp.nan, density)

    print(f"  KDE done — heatmap shape {density.shape}")
    return density


# ---------------------------------------------------------------------------
# cuML: Random Forest fire risk prediction
# ---------------------------------------------------------------------------
def fire_risk_layer(ds, fire_data, terrain_da, raw_cpu,
                    wind_speed_gpu=None, wind_exposure_gpu=None):
    """Train RF to predict fire probability from terrain + wind features."""
    from cuml.ensemble import RandomForestClassifier

    features_list = fire_data.get('features', [])
    if not features_list:
        print("  RF: no fire detections — returning zeros")
        return cp.zeros(terrain_da.shape, dtype=cp.float32)

    shape = terrain_da.shape
    elev_cpu = raw_cpu.values
    xs_cpu = raw_cpu.x.values
    ys_cpu = raw_cpu.y.values

    centroids = []
    for f in features_list:
        geom = f['geometry']
        if geom['type'] == 'Point':
            centroids.append(geom['coordinates'][:2])
        elif geom['type'] in ('Polygon', 'MultiPolygon'):
            coords = geom['coordinates']
            ring = coords[0][0] if geom['type'] == 'MultiPolygon' else coords[0]
            cx = sum(c[0] for c in ring) / len(ring)
            cy = sum(c[1] for c in ring) / len(ring)
            centroids.append([cx, cy])
    centroids = np.array(centroids, dtype=np.float64)

    if fire_data.get('crs') is None or 'EPSG:4326' in str(fire_data.get('crs', '')):
        transformer = Transformer.from_crs("EPSG:4326", CRS, always_xy=True)
        cxs, cys = transformer.transform(centroids[:, 0], centroids[:, 1])
        centroids = np.column_stack([cxs, cys])

    x_sorted = xs_cpu if xs_cpu[-1] > xs_cpu[0] else xs_cpu[::-1]
    y_sorted = ys_cpu if ys_cpu[-1] > ys_cpu[0] else ys_cpu[::-1]
    x_flip = xs_cpu[-1] < xs_cpu[0]
    y_flip = ys_cpu[-1] < ys_cpu[0]

    col_idx = np.searchsorted(x_sorted, centroids[:, 0]).clip(0, len(xs_cpu) - 1)
    row_idx = np.searchsorted(y_sorted, centroids[:, 1]).clip(0, len(ys_cpu) - 1)
    if x_flip:
        col_idx = len(xs_cpu) - 1 - col_idx
    if y_flip:
        row_idx = len(ys_cpu) - 1 - row_idx

    fire_mask = np.zeros(shape, dtype=bool)
    for r, c in zip(row_idx, col_idx):
        r0, r1 = max(0, r - 3), min(shape[0], r + 4)
        c0, c1 = max(0, c - 3), min(shape[1], c + 4)
        fire_mask[r0:r1, c0:c1] = True

    elev_gpu = ds['elevation'].data
    slp_gpu  = ds['slope'].data
    asp_gpu  = ds['aspect'].data
    south_score = cp.cos(cp.deg2rad(asp_gpu - 180.0))

    feature_arrays = [
        elev_gpu.ravel(),
        slp_gpu.ravel(),
        asp_gpu.ravel(),
        south_score.ravel(),
    ]
    feature_names = ['elevation', 'slope', 'aspect', 'south_score']

    if wind_speed_gpu is not None:
        feature_arrays.append(wind_speed_gpu.ravel())
        feature_names.append('wind_speed')
    if wind_exposure_gpu is not None:
        feature_arrays.append(wind_exposure_gpu.ravel())
        feature_names.append('wind_exposure')

    feat_stack = cp.stack(feature_arrays, axis=1)

    valid_mask = ~cp.isnan(feat_stack).any(axis=1)
    valid_np = cp.asnumpy(valid_mask.reshape(shape))

    pos_mask = fire_mask & valid_np
    neg_mask = ~fire_mask & valid_np & ~np.isnan(elev_cpu)

    pos_idx = np.flatnonzero(pos_mask.ravel())
    neg_idx = np.flatnonzero(neg_mask.ravel())

    n_pos = len(pos_idx)
    if n_pos == 0:
        print("  RF: no valid fire pixels after mapping — returning zeros")
        return cp.zeros(shape, dtype=cp.float32)

    n_neg = min(len(neg_idx), n_pos * 3)
    rng = np.random.default_rng(42)
    neg_sample = rng.choice(neg_idx, size=n_neg, replace=False)

    train_idx = cp.asarray(np.concatenate([pos_idx, neg_sample]))
    labels = cp.concatenate([cp.ones(n_pos, dtype=cp.int32),
                             cp.zeros(n_neg, dtype=cp.int32)])

    X_train = feat_stack[train_idx]

    print(f"  RF: {n_pos} fire pixels, {n_neg} non-fire pixels, "
          f"{RF_N_TREES} trees, max_depth={RF_MAX_DEPTH}")
    print(f"  RF features: {', '.join(feature_names)}")

    rf = RandomForestClassifier(
        n_estimators=RF_N_TREES,
        max_depth=RF_MAX_DEPTH,
        random_state=42,
    )
    rf.fit(X_train, labels)

    valid_features = feat_stack[valid_mask]
    proba = rf.predict_proba(valid_features)
    fire_prob_col = proba[:, 1]

    result = cp.full(feat_stack.shape[0], cp.nan, dtype=cp.float32)
    result[valid_mask] = fire_prob_col.astype(cp.float32)
    result = result.reshape(shape)

    pct = cp.nanmean(result)
    print(f"  RF done — mean fire probability: {float(pct):.3f}")
    return result


# ---------------------------------------------------------------------------
# cuML: HDBSCAN building cluster detection
# ---------------------------------------------------------------------------
def cluster_buildings(bldg_data, terrain_da):
    """Discover neighbourhood clusters from building centroids using HDBSCAN.

    Returns (labels_per_building, cluster_raster).
    - labels_per_building: numpy int array, -1 = noise
    - cluster_raster: cupy 2D float array painted with cluster IDs for the
      explore() overlay (NaN where no buildings).
    """
    from cuml.cluster import HDBSCAN

    features = bldg_data.get('features', [])
    if not features:
        return np.array([]), cp.full(terrain_da.shape, cp.nan, dtype=cp.float32)

    # Extract building centroids (WGS84 → CRS)
    lons, lats = [], []
    for f in features:
        coords = f['geometry']['coordinates']
        ring = coords[0][0] if f['geometry']['type'] == 'MultiPolygon' else coords[0]
        lons.append(sum(c[0] for c in ring) / len(ring))
        lats.append(sum(c[1] for c in ring) / len(ring))

    transformer = Transformer.from_crs("EPSG:4326", CRS, always_xy=True)
    pxs, pys = transformer.transform(
        np.asarray(lons, dtype=np.float64),
        np.asarray(lats, dtype=np.float64),
    )
    centroids = np.column_stack([pxs, pys]).astype(np.float32)

    print(f"  HDBSCAN: {len(centroids):,} buildings, "
          f"min_cluster_size={HDBSCAN_MIN_CLUSTER}, "
          f"min_samples={HDBSCAN_MIN_SAMPLES}")

    centroids_gpu = cp.asarray(centroids)
    model = HDBSCAN(
        min_cluster_size=HDBSCAN_MIN_CLUSTER,
        min_samples=HDBSCAN_MIN_SAMPLES,
    )
    labels_gpu = model.fit_predict(centroids_gpu)
    labels = cp.asnumpy(labels_gpu)

    n_clusters = int(labels.max()) + 1
    n_noise = int((labels == -1).sum())
    print(f"  HDBSCAN done — {n_clusters} clusters, {n_noise} outliers")

    # Build a raster layer: paint each DEM pixel with the cluster ID of the
    # nearest building (within 100 m), so the layer is visible in explore().
    xs = terrain_da.x.values
    ys = terrain_da.y.values

    # For performance, use a simple rasterisation: map each building centroid
    # to its nearest pixel and paint it.
    x_sorted = xs if xs[-1] > xs[0] else xs[::-1]
    y_sorted = ys if ys[-1] > ys[0] else ys[::-1]
    x_flip = xs[-1] < xs[0]
    y_flip = ys[-1] < ys[0]

    cols = np.searchsorted(x_sorted, centroids[:, 0]).clip(0, len(xs) - 1)
    rows = np.searchsorted(y_sorted, centroids[:, 1]).clip(0, len(ys) - 1)
    if x_flip:
        cols = len(xs) - 1 - cols
    if y_flip:
        rows = len(ys) - 1 - rows

    raster = np.full(terrain_da.shape, np.nan, dtype=np.float32)
    for i in range(len(labels)):
        if labels[i] >= 0:
            raster[rows[i], cols[i]] = float(labels[i])

    return labels, cp.asarray(raster)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    terrain, raw_cpu = load_terrain()

    # --- Standard terrain layers ----------------------------------------------
    print("Computing terrain analysis layers...")
    ds = xr.Dataset({
        'elevation': terrain.rename(None),
        'slope': slope(terrain),
        'aspect': aspect(terrain),
        'quantile': quantile(terrain),
    })
    print(ds)

    # === cuML: K-Means terrain classification =================================
    print("\n--- cuML: K-Means Terrain Classification ---")
    tc = terrain_classification(ds)
    ds['terrain_class'] = xr.DataArray(
        tc, coords=ds['elevation'].coords, dims=ds['elevation'].dims,
    )

    # === Fetch wind data (used for ML features + particle animation) ==========
    print("\n--- Fetching wind data ---")
    wind = None
    try:
        from rtxpy import fetch_wind
        wind = fetch_wind(BOUNDS, grid_size=15)
    except Exception as e:
        print(f"  Skipping wind: {e}")

    # === Fetch weather data (clouds + rain via Shift+N) =======================
    weather = None
    try:
        from rtxpy import fetch_weather
        weather = fetch_weather(BOUNDS, grid_size=15)
    except Exception as e:
        print(f"  Skipping weather: {e}")

    print("\n--- Interpolating wind onto DEM grid ---")
    wind_speed_gpu, wind_u_gpu, wind_v_gpu = interpolate_wind(wind, terrain)
    ds['wind_speed'] = xr.DataArray(
        wind_speed_gpu,
        coords=ds['elevation'].coords, dims=ds['elevation'].dims,
    )

    wind_exposure_gpu = compute_wind_exposure(
        ds['aspect'].data, wind_u_gpu, wind_v_gpu,
    )
    ds['wind_exposure'] = xr.DataArray(
        wind_exposure_gpu,
        coords=ds['elevation'].coords, dims=ds['elevation'].dims,
    )

    # === Fetch fire data ======================================================
    print("\n--- Fetching FIRMS fire detections ---")
    fire_data = {'features': []}
    try:
        fire_data = fetch_firms(
            bounds=BOUNDS, date_span='7d',
            cache_path=CACHE / "bay_area_fires.geojson",
            crs=CRS,
        )
        n_fires = len(fire_data.get('features', []))
        print(f"  {n_fires} fire detections in the last 7 days")
    except Exception as e:
        print(f"  Skipping FIRMS: {e}")

    # === cuML: KDE fire density heatmap =======================================
    print("\n--- cuML: KDE Fire Density Heatmap ---")
    fd = fire_density_layer(fire_data, terrain)
    ds['fire_density'] = xr.DataArray(
        fd, coords=ds['elevation'].coords, dims=ds['elevation'].dims,
    )

    # === cuML: Random Forest fire risk prediction =============================
    print("\n--- cuML: Random Forest Fire Risk Prediction ---")
    fr = fire_risk_layer(
        ds, fire_data, terrain, raw_cpu,
        wind_speed_gpu=wind_speed_gpu,
        wind_exposure_gpu=wind_exposure_gpu,
    )
    ds['fire_risk'] = xr.DataArray(
        fr, coords=ds['elevation'].coords, dims=ds['elevation'].dims,
    )

    # === Fetch buildings for HDBSCAN ==========================================
    print("\n--- Fetching buildings ---")
    bldg_data = {'features': []}
    try:
        bldg_data = fetch_buildings(
            bounds=BOUNDS, source='overture',
            cache_path=CACHE / "bay_area_buildings.geojson",
        )
        print(f"  {len(bldg_data.get('features', []))} buildings")
    except Exception as e:
        print(f"  Skipping buildings: {e}")

    # === cuML: HDBSCAN building cluster detection =============================
    print("\n--- cuML: HDBSCAN Neighbourhood Clustering ---")
    bldg_labels, cluster_raster = cluster_buildings(bldg_data, terrain)
    ds['urban_cluster'] = xr.DataArray(
        cluster_raster,
        coords=ds['elevation'].coords, dims=ds['elevation'].dims,
    )

    print(f"\nDataset layers: {list(ds.data_vars)}")
    print(ds)

    # --- Satellite tiles ------------------------------------------------------
    print("\nLoading satellite tiles...")
    ds.rtx.place_tiles('satellite', z='elevation', zoom=13)

    # --- Load or build scene meshes -------------------------------------------
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
        # --- Buildings coloured by HDBSCAN cluster ----------------------------
        if bldg_data.get('features') and len(bldg_labels) > 0:
            # Group buildings by cluster label
            unique_labels = sorted(set(bldg_labels))
            for label in unique_labels:
                mask = bldg_labels == label
                feats = [f for f, m in zip(bldg_data['features'], mask) if m]
                if not feats:
                    continue

                if label == -1:
                    gid = 'bldg_outlier'
                    color = OUTLIER_COLOR
                    tag = 'outlier'
                else:
                    gid = f'bldg_c{label}'
                    color = CLUSTER_COLORS[label % len(CLUSTER_COLORS)]
                    tag = f'cluster {label}'

                try:
                    with warnings.catch_warnings():
                        warnings.filterwarnings("ignore",
                                                message="place_geojson called before")
                        info = ds.rtx.place_geojson(
                            {"type": "FeatureCollection", "features": feats},
                            z='elevation', height=8, extrude=True, merge=True,
                            geometry_id=gid, color=color,
                        )
                        print(f"  Placed {info['geometries']} {tag} buildings")
                except Exception as e:
                    print(f"  Skipping {tag}: {e}")

        # --- Roads ------------------------------------------------------------
        try:
            for rt, gid, clr in [('major', 'road_major', (0.10, 0.10, 0.10)),
                                  ('minor', 'road_minor', (0.55, 0.55, 0.55))]:
                data = fetch_roads(bounds=BOUNDS, road_type=rt, source='overture',
                                   cache_path=CACHE / f"bay_area_roads_{rt}.geojson")
                info = ds.rtx.place_roads(data, z='elevation',
                                          geometry_id=gid, color=clr)
                print(f"  Placed {info['geometries']} Overture {rt} road geometries")
        except ImportError:
            print("Skipping Overture roads (pip install duckdb)")
        except Exception as e:
            print(f"Skipping roads: {e}")

        # --- Water features ---------------------------------------------------
        try:
            water_data = fetch_water(bounds=BOUNDS, water_type='all',
                                     cache_path=CACHE / "bay_area_water.geojson")
            results = ds.rtx.place_water(water_data, z='elevation')
            for cat, info in results.items():
                print(f"  Placed {info['geometries']} {cat} water features")
        except Exception as e:
            print(f"Skipping water: {e}")

        # --- Save meshes ------------------------------------------------------
        try:
            ds.rtx.save_meshes(ZARR)
        except Exception as e:
            print(f"Could not save mesh cache: {e}")

    # --- FIRMS fire markers ---------------------------------------------------
    if fire_data.get('features'):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="place_geojson called before")
            fire_info = ds.rtx.place_geojson(
                fire_data, z='elevation', height=20,
                geometry_id='fire', color=(1.0, 0.25, 0.0, 3.0),
                extrude=True, merge=True,
            )
            print(f"Placed {fire_info['geometries']} FIRMS fire markers")

    # --- Launch explorer ------------------------------------------------------
    print("\n" + "=" * 65)
    print("BAY AREA WILDFIRE & URBAN ANALYSIS — cuML + rtxpy")
    print("=" * 65)
    print("  G        cycle layers (elevation → terrain_class → wind_speed")
    print("                         → wind_exposure → fire_density")
    print("                         → fire_risk → urban_cluster)")
    print("  U        toggle satellite overlay")
    print("  Shift+W  toggle wind particles")
    print("  O / V    set observer / toggle viewshed")
    print("  H        full help overlay")
    print("=" * 65)

    ds.rtx.explore(
        z='elevation',
        scene_zarr=ZARR,
        width=2048,
        height=1600,
        render_scale=0.5,
        color_stretch='cbrt',
        subsample=1,
        wind_data=wind,
        weather_data=weather,
        ao_samples=1,
        repl=True,
    )

    print("Done")
