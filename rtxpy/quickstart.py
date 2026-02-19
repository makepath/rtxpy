"""One-call launcher: DEM fetch -> analysis layers -> feature placement -> explore()."""

import warnings
from pathlib import Path


def quickstart(
    name,
    bounds,
    crs,
    source='copernicus',
    features=None,
    tiles='satellite',
    tile_zoom=None,
    wind=True,
    cache_dir=None,
    **explore_kwargs,
):
    """Fetch terrain, place features, and launch the interactive viewer.

    Parameters
    ----------
    name : str
        Location name used to derive the zarr filename
        (``{name}_dem.zarr``) and GeoJSON cache filenames.
    bounds : tuple of float
        (west, south, east, north) in WGS84 degrees.
    crs : str
        EPSG code for the target projection (e.g. ``'EPSG:32620'``).
    source : str
        DEM source: ``'copernicus'``, ``'usgs_10m'``, ``'srtm'``.
        Default ``'copernicus'``.
    features : list or dict, optional
        Features to place on the terrain.  List form uses defaults::

            features=['buildings', 'roads', 'water', 'fire']

        Dict form allows per-feature overrides::

            features={'buildings': {'elev_scale': 0.33},
                      'fire': {'region': 'southeast_asia'}}

        Supported keys: ``'buildings'``, ``'roads'``, ``'water'``,
        ``'fire'``, ``'places'``, ``'infrastructure'``, ``'land_use'``,
        ``'restaurant_grades'``, ``'gtfs'``.
    tiles : str or None
        Tile provider: ``'satellite'``, ``'osm'``, or ``None`` to skip.
        Default ``'satellite'``.
    tile_zoom : int, optional
        Tile zoom level override.  ``None`` uses the provider default.
    wind : bool
        Fetch live wind data from Open-Meteo.  Default ``True``.
    cache_dir : str or Path, optional
        Directory for the zarr store and GeoJSON caches.  Defaults to
        the current working directory.
    **explore_kwargs
        Forwarded to ``ds.rtx.explore()``.  Defaults::

            width=2048, height=1600, render_scale=0.5,
            color_stretch='cbrt', subsample=1, repl=True
    """
    import numpy as np
    import xarray as xr
    from xrspatial import slope, aspect, quantile

    # -- paths ----------------------------------------------------------------
    if cache_dir is None:
        cache_dir = Path.cwd()
    else:
        cache_dir = Path(cache_dir)
    zarr_path = cache_dir / f"{name}_dem.zarr"

    # -- DEM ------------------------------------------------------------------
    from .remote_data import fetch_dem as _fetch_dem

    terrain = _fetch_dem(bounds=bounds, output_path=zarr_path,
                         source=source, crs=crs)
    terrain.data = np.ascontiguousarray(terrain.data)
    terrain = terrain.rtx.to_cupy()

    # -- Dataset with analysis layers -----------------------------------------
    print("Building Dataset with terrain analysis layers...")
    ds = xr.Dataset({
        'elevation': terrain.rename(None),
        'slope': slope(terrain),
        'aspect': aspect(terrain),
        'quantile': quantile(terrain),
    })

    # -- tiles ----------------------------------------------------------------
    if tiles:
        print(f"Loading {tiles} tiles...")
        ds.rtx.place_tiles(tiles, z='elevation', zoom=tile_zoom)

    # -- features -------------------------------------------------------------
    feat = _parse_features(features)
    _TEMPORAL = {'fire'}
    cacheable = {k: v for k, v in feat.items() if k not in _TEMPORAL}
    temporal = {k: v for k, v in feat.items() if k in _TEMPORAL}

    # Check for mesh cache in zarr and which features it contains
    has_cache = False
    cached_features = set()
    if cacheable:
        try:
            import zarr as _zarr
            store = _zarr.open(str(zarr_path), mode='r',
                               use_consolidated=False)
            if 'meshes' in store and len(list(store['meshes'])) > 0:
                has_cache = True
                # Read which feature keys were stored with this cache
                cached_features = set(
                    store['meshes'].attrs.get('feature_keys', []))
            del store
        except Exception:
            pass

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore",
                                message="place_geojson called before")
        if has_cache:
            ds.rtx.load_meshes(zarr_path)
            # Place any cacheable features missing from the cache.
            # If the cache pre-dates feature_keys tracking, we only
            # know which features were loaded by checking the scene's
            # geometry IDs against known feature prefixes.
            if cached_features:
                missing = {k: v for k, v in cacheable.items()
                           if k not in cached_features}
            else:
                # Legacy cache without feature_keys metadata —
                # check which features actually have geometries loaded
                loaded_gids = set(ds.rtx._get_terrain_da(
                    list(ds.data_vars)[0]).rtx._rtx.list_geometries())
                missing = {k: v for k, v in cacheable.items()
                           if not _has_geometry_for_feature(k, loaded_gids)}
            if missing:
                names = ', '.join(missing)
                print(f"Placing features not in cache: {names}")
                _place_features(ds, missing, name, bounds, crs, cache_dir)
                try:
                    ds.rtx.save_meshes(zarr_path)
                    _save_feature_keys(zarr_path, cacheable.keys())
                except Exception as e:
                    print(f"Could not update mesh cache: {e}")
            elif not cached_features:
                # Legacy cache is complete — stamp it with feature_keys
                try:
                    _save_feature_keys(zarr_path, cacheable.keys())
                except Exception:
                    pass
        elif cacheable:
            _place_features(ds, cacheable, name, bounds, crs, cache_dir)
            try:
                ds.rtx.save_meshes(zarr_path)
                _save_feature_keys(zarr_path, cacheable.keys())
            except Exception as e:
                print(f"Could not save mesh cache: {e}")

        # Temporal features (always fresh, not cached in zarr)
        if temporal:
            _place_features(ds, temporal, name, bounds, crs, cache_dir)

    # -- gtfs realtime --------------------------------------------------------
    # When loading from cache, _place_gtfs() is never called, so _gtfs_data
    # is not set.  Fetch the GTFS data directly for the realtime overlay.
    gtfs_data = ds.attrs.pop('_gtfs_data', None)
    if gtfs_data is None and 'gtfs' in feat:
        gtfs_opts = feat['gtfs']
        try:
            from .remote_data import fetch_gtfs as _fetch_gtfs
            gtfs_data = _fetch_gtfs(
                bounds=bounds,
                feed_url=gtfs_opts.get('feed_url'),
                gtfs_path=gtfs_opts.get('gtfs_path'),
                route_types=gtfs_opts.get('route_types'),
                cache_path=cache_dir / f"{name}_gtfs.json",
                crs=crs,
                realtime_url=gtfs_opts.get('realtime_url'))
        except Exception as e:
            print(f"Skipping GTFS realtime: {e}")

    # -- wind -----------------------------------------------------------------
    wind_data = None
    if wind:
        try:
            from .remote_data import fetch_wind as _fetch_wind
            wind_data = _fetch_wind(bounds, grid_size=15)
        except Exception as e:
            print(f"Skipping wind: {e}")

    # -- explore --------------------------------------------------------------
    defaults = dict(
        width=2048, height=1600, render_scale=0.5,
        color_stretch='cbrt', subsample=1, repl=True,
    )
    defaults.update(explore_kwargs)

    print("\nLaunching explore...\n")
    ds.rtx.explore(z='elevation', scene_zarr=zarr_path,
                   wind_data=wind_data, gtfs_data=gtfs_data, **defaults)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _has_geometry_for_feature(feature_key, loaded_gids):
    """Check whether any geometry ID in the scene matches *feature_key*."""
    _PREFIXES = {
        'buildings': ('building_',),
        'roads': ('road_major', 'road_minor'),
        'water': ('water_',),
        'places': ('places',),
        'infrastructure': ('infrastructure',),
        'land_use': ('land_use',),
        'restaurant_grades': ('grade_a', 'grade_b', 'grade_c'),
        'gtfs': ('gtfs_',),
    }
    prefixes = _PREFIXES.get(feature_key, (feature_key,))
    return any(
        any(gid == p or gid.startswith(p) for p in prefixes)
        for gid in loaded_gids
    )


def _save_feature_keys(zarr_path, keys):
    """Store feature keys in the zarr meshes group attributes."""
    import zarr as _zarr
    store = _zarr.open(str(zarr_path), mode='r+', use_consolidated=False)
    store['meshes'].attrs['feature_keys'] = sorted(keys)


def _parse_features(features):
    """Normalize *features* to ``{key: {opts}}`` dict."""
    if features is None:
        return {}
    if isinstance(features, (list, tuple)):
        return {f: {} for f in features}
    out = {}
    for key, val in features.items():
        if val is True:
            out[key] = {}
        elif isinstance(val, dict):
            out[key] = val
        elif val is False or val is None:
            continue
        else:
            out[key] = {}
    return out


def _place_features(ds, features, name, bounds, crs, cache_dir):
    """Place all requested features, catching errors per feature."""
    cache_dir = Path(cache_dir)
    for key, opts in features.items():
        handler = _FEATURE_HANDLERS.get(key)
        if handler is None:
            print(f"Unknown feature: {key!r}")
            continue
        try:
            handler(ds, opts, name, bounds, crs, cache_dir)
        except Exception as e:
            print(f"Skipping {key}: {e}")


# -- individual feature handlers ----------------------------------------------

def _place_buildings(ds, opts, name, bounds, crs, cache_dir):
    from .remote_data import fetch_buildings
    src = opts.get('source', 'overture')
    data = fetch_buildings(bounds=bounds, source=src,
                           cache_path=cache_dir / f"{name}_buildings.geojson")
    place_kw = {}
    if 'elev_scale' in opts:
        place_kw['elev_scale'] = opts['elev_scale']
    info = ds.rtx.place_buildings(data, z='elevation', **place_kw)
    print(f"Placed {info['geometries']} building geometries")


def _place_roads(ds, opts, name, bounds, crs, cache_dir):
    from .remote_data import fetch_roads
    for rt, gid, clr in [('major', 'road_major', (0.10, 0.10, 0.10)),
                          ('minor', 'road_minor', (0.55, 0.55, 0.55))]:
        data = fetch_roads(bounds=bounds, road_type=rt, source='overture',
                           cache_path=cache_dir / f"{name}_roads_{rt}.geojson")
        info = ds.rtx.place_roads(data, z='elevation',
                                  geometry_id=gid, color=clr)
        print(f"Placed {info['geometries']} {rt} road geometries")


def _place_water(ds, opts, name, bounds, crs, cache_dir):
    from .remote_data import fetch_water
    src = opts.get('source', 'overture')
    wt = opts.get('water_type', 'all')
    data = fetch_water(bounds=bounds, water_type=wt, source=src,
                       cache_path=cache_dir / f"{name}_water.geojson")
    results = ds.rtx.place_water(data, z='elevation')
    for cat, info in results.items():
        print(f"Placed {info['geometries']} {cat} water features")


def _place_fire(ds, opts, name, bounds, crs, cache_dir):
    from .remote_data import fetch_firms
    span = opts.get('date_span', '7d')
    region = opts.get('region', None)
    data = fetch_firms(bounds=bounds, date_span=span, region=region,
                       cache_path=cache_dir / f"{name}_fires.geojson",
                       crs=crs)
    if data.get('features'):
        info = ds.rtx.place_geojson(
            data, z='elevation', height=20,
            geometry_id='fire', color=(1.0, 0.25, 0.0, 3.0),
            extrude=True, merge=True,
        )
        print(f"Placed {info['geometries']} fire detection footprints")
    else:
        print("No fire detections in the last 7 days")


def _place_places(ds, opts, name, bounds, crs, cache_dir):
    from .remote_data import fetch_places
    category = opts.get('category', 'eat_and_drink')
    data = fetch_places(bounds=bounds, category=category,
                        cache_path=cache_dir / f"{name}_places.geojson",
                        crs=crs)
    if data.get('features'):
        info = ds.rtx.place_geojson(
            data, z='elevation', height=8,
            geometry_id='places', color=(1.0, 0.8, 0.0),
            merge=True,
        )
        print(f"Placed {info['geometries']} place markers")


def _place_infrastructure(ds, opts, name, bounds, crs, cache_dir):
    from .remote_data import fetch_infrastructure
    itype = opts.get('infra_type', 'communication')
    data = fetch_infrastructure(bounds=bounds, infra_type=itype,
                                cache_path=cache_dir / f"{name}_infra.geojson",
                                crs=crs)
    if data.get('features'):
        info = ds.rtx.place_geojson(
            data, z='elevation', height=30,
            geometry_id='infrastructure', color=(0.8, 0.2, 0.2),
            merge=True,
        )
        print(f"Placed {info['geometries']} infrastructure features")


def _place_land_use(ds, opts, name, bounds, crs, cache_dir):
    from .remote_data import fetch_land_use
    lt = opts.get('land_type', 'park')
    data = fetch_land_use(bounds=bounds, land_type=lt,
                          cache_path=cache_dir / f"{name}_land_use.geojson",
                          crs=crs)
    if data.get('features'):
        info = ds.rtx.place_geojson(
            data, z='elevation', height=2,
            geometry_id='land_use', color=(0.2, 0.7, 0.3, 0.5),
            extrude=True, merge=True,
        )
        print(f"Placed {info['geometries']} park polygons")


def _place_restaurant_grades(ds, opts, name, bounds, crs, cache_dir):
    from .remote_data import fetch_restaurant_grades
    data = fetch_restaurant_grades(
        bounds=bounds,
        cache_path=cache_dir / f"{name}_restaurants.geojson",
    )
    for grade, gid, clr in [
        ('A', 'grade_a', (0.20, 0.78, 0.40)),
        ('B', 'grade_b', (0.95, 0.75, 0.10)),
        ('C', 'grade_c', (0.90, 0.22, 0.20)),
    ]:
        subset = {
            "type": "FeatureCollection",
            "features": [f for f in data['features']
                         if f['properties'].get('grade') == grade],
        }
        if subset['features']:
            info = ds.rtx.place_geojson(
                subset, z='elevation', height=15,
                geometry_id=gid, color=clr, merge=True,
            )
            print(f"Placed {info['geometries']} grade {grade} restaurants")


def _place_gtfs(ds, opts, name, bounds, crs, cache_dir):
    from .remote_data import fetch_gtfs
    data = fetch_gtfs(bounds=bounds,
                      feed_url=opts.get('feed_url'),
                      gtfs_path=opts.get('gtfs_path'),
                      route_types=opts.get('route_types'),
                      cache_path=cache_dir / f"{name}_gtfs.json",
                      crs=crs,
                      realtime_url=opts.get('realtime_url'))
    if data['metadata']['n_routes'] > 0 or data['metadata']['n_stops'] > 0:
        results = ds.rtx.place_gtfs(data, z='elevation',
                                    stop_height=opts.get('stop_height', 8.0),
                                    route_width=opts.get('route_width'))
        for cat, groups in results.items():
            for label, info in groups.items():
                parts = []
                if 'routes' in info:
                    parts.append(f"{info['routes']['geometries']} routes")
                if 'stops' in info:
                    parts.append(f"{info['stops']['geometries']} stops")
                print(f"Placed {cat} [{label}]: {', '.join(parts)}")
        # Stash gtfs_data for realtime overlay in explore()
        ds.attrs['_gtfs_data'] = data
    else:
        print("No GTFS routes/stops found in bounds")


_FEATURE_HANDLERS = {
    'buildings': _place_buildings,
    'roads': _place_roads,
    'water': _place_water,
    'fire': _place_fire,
    'places': _place_places,
    'infrastructure': _place_infrastructure,
    'land_use': _place_land_use,
    'restaurant_grades': _place_restaurant_grades,
    'gtfs': _place_gtfs,
}
