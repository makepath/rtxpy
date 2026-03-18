"""Build a self-contained scene zarr from a bounding box.

Fetches 30 m elevation (Copernicus), Overture buildings and water,
Open-Meteo wind and weather, then writes everything into a single
zarr store conforming to the scene zarr specification.

Usage as a library::

    from rtxpy.scene import build_scene
    build_scene((-112.2, 36.0, -112.0, 36.2), "grand_canyon.zarr")

Usage from the command line::

    python -m rtxpy.scene -- -112.2 36.0 -112.0 36.2 grand_canyon.zarr

The resulting zarr can be opened directly with explore()::

    import rtxpy
    rtxpy.explore_scene("grand_canyon.zarr")
"""

import sys
from pathlib import Path

import numpy as np
from zarr.codecs import BloscCodec

_BLOSC = BloscCodec(cname='zstd', clevel=6, shuffle='bitshuffle')


# ---------------------------------------------------------------------------
# Scene builder
# ---------------------------------------------------------------------------

def build_scene(
    bounds,
    output_path,
    *,
    dem_source="copernicus",
    crs=None,
    buildings=True,
    roads=True,
    water=True,
    wind=True,
    weather=True,
    hydro=False,
    fires=False,
    name=None,
    cache_dir=None,
    tile_size=256,
    resume=False,
    progress=None,
):
    """Fetch data for a bounding box and write a scene zarr.

    Parameters
    ----------
    bounds : tuple or Location
        (west, south, east, north) in WGS84 degrees.  If a
        :class:`~rtxpy.scene_locations.Location` is passed and *crs*
        is None, its ``.crs`` is used automatically.
    output_path : str or Path
        Where to write the zarr store. Overwrites if it exists.
    dem_source : str
        DEM source: ``"copernicus"`` (30 m, default), ``"usgs_10m"``,
        ``"usgs_1m"``, or ``"srtm"``.
    crs : str or None
        Target CRS (e.g. ``"EPSG:32617"``). None auto-selects a local
        UTM zone from the bounding box center.
    buildings : bool
        Fetch and place Overture Maps building footprints.
    roads : bool
        Fetch and place Overture Maps road networks.
    water : bool
        Fetch and place Overture Maps water features.
    wind : bool
        Fetch Open-Meteo wind velocity grids.
    weather : bool
        Fetch Open-Meteo weather (cloud cover, temperature, etc.).
    hydro : bool
        Compute MFD hydrological flow from the DEM.
    fires : bool
        Fetch NASA FIRMS fire detections (last 24 h).
    name : str or None
        Human-readable scene name. Defaults to the output filename stem.
    cache_dir : str or Path or None
        Directory for intermediate tile caches. Defaults to a ``.cache``
        sibling of *output_path*.
    tile_size : int
        Zarr chunk size for the elevation array, in pixels.  This
        also controls LOD tile granularity — one chunk = one tile.
        Default 256.
    resume : bool
        If True, skip fetching data whose zarr group already exists.
        Useful for adding layers to an existing scene without
        re-downloading everything.
    progress : callable or None
        Optional callback ``progress(step, message)`` called at each
        stage. *step* is a string like ``'elevation'``, ``'buildings'``.
        If None, messages go to stdout via ``print()``.
    """
    import zarr
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from .scene_locations import Location

    # Auto-detect CRS: from Location objects, or UTM from bounds center
    if crs is None and isinstance(bounds, Location):
        crs = bounds.crs
    if crs is None:
        from .scene_locations import _utm_epsg
        crs = _utm_epsg(*bounds)

    output_path = Path(output_path)
    if name is None:
        name = output_path.stem

    if cache_dir is None:
        cache_dir = output_path.parent / ".cache"
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    def _log(step, msg):
        if progress is not None:
            progress(step, msg)
        else:
            print(msg)

    west, south, east, north = bounds

    # ---- 1. Elevation (always needed — zarr store is created here) ----
    _log('elevation',
         f"Fetching {dem_source} DEM for "
         f"({west:.4f}, {south:.4f}, {east:.4f}, {north:.4f})...")

    from .remote_data import fetch_dem
    dem = fetch_dem(bounds, str(output_path), source=dem_source, crs=crs,
                    cache_dir=str(cache_dir), tile_size=tile_size)
    _log('elevation', f"  DEM shape: {dem.shape}, CRS: {dem.rio.crs}")

    # Stamp root attributes per spec
    store = zarr.open(str(output_path), mode='r+')
    from .mesh_store import SCENE_VERSION
    store.attrs['rtxpy_scene_version'] = SCENE_VERSION
    store.attrs['name'] = name
    store.attrs['bounds_lonlat'] = [west, south, east, north]
    store.attrs['source'] = f"build_scene(dem={dem_source})"
    store.attrs['created_at'] = _iso_now()

    # Add epsg to spatial_ref if we can determine it
    if 'spatial_ref' in store:
        try:
            epsg = dem.rio.crs.to_epsg()
            if epsg is not None:
                store['spatial_ref'].attrs['epsg'] = epsg
        except Exception:
            pass

    # ---- 2. Triangulate for mesh placement ----
    need_meshes = buildings or roads or water or fires
    if need_meshes and not (resume and 'meshes' in store):
        _log('triangulate', "Triangulating terrain...")
        terrain = dem.copy()
        terrain.data = np.ascontiguousarray(terrain.data)
        terrain.rtx.triangulate()
    else:
        terrain = None

    # ---- 3. Parallel network fetches (buildings, roads, water, fires) ----
    # These are independent of each other and can overlap.
    geojson_results = {}

    if terrain is not None:
        fetch_jobs = {}
        with ThreadPoolExecutor(max_workers=4) as pool:
            if buildings:
                fetch_jobs['buildings'] = pool.submit(
                    _fetch_or_skip, "Overture buildings",
                    lambda: _fetch_buildings_overture(bounds, cache_dir))
            if roads:
                fetch_jobs['roads'] = pool.submit(
                    _fetch_or_skip, "Overture roads",
                    lambda: _fetch_roads_overture(bounds, cache_dir))
            if water:
                fetch_jobs['water'] = pool.submit(
                    _fetch_or_skip, "Overture water",
                    lambda: _fetch_water_overture(bounds, cache_dir))
            if fires:
                fetch_jobs['fires'] = pool.submit(
                    _fetch_or_skip, "FIRMS fire detections",
                    lambda: _fetch_firms(bounds, cache_dir))

            for key, future in fetch_jobs.items():
                geojson_results[key] = future.result()

    # ---- 4. Place geometry ----
    def _place(key, place_fn, label):
        gj = geojson_results.get(key)
        if gj and gj.get('features'):
            n = len(gj['features'])
            _log(key, f"  Placing {n} {label}...")
            place_fn(gj)

    if terrain is not None:
        _place('buildings', terrain.rtx.place_buildings, 'buildings')
        _place('roads',
               lambda gj: terrain.rtx.place_roads(gj, geometry_id='road'),
               'roads')
        _place('water', terrain.rtx.place_water, 'water features')
        _place('fires',
               lambda gj: terrain.rtx.place_geojson(
                   gj, geometry_id='fire', height=5.0,
                   color=(1.0, 0.2, 0.0, 1.0)),
               'fire detections')

        if _has_baked_meshes_da(terrain):
            _log('meshes', "Saving meshes to zarr...")
            terrain.rtx.save_meshes(str(output_path))
    elif resume and 'meshes' in store:
        _log('meshes', "Meshes already in zarr, skipping (resume=True)")

    # ---- 5. Parallel non-mesh data fetches (wind, weather) ----
    wind_data = None
    weather_data = None

    if wind or weather:
        net_jobs = {}
        with ThreadPoolExecutor(max_workers=2) as pool:
            if wind and not (resume and 'wind' in store):
                net_jobs['wind'] = pool.submit(
                    _fetch_or_skip, "wind",
                    lambda: _fetch_wind(bounds))
            elif resume and 'wind' in store:
                _log('wind', "Wind already in zarr, skipping (resume=True)")

            if weather and not (resume and 'weather' in store):
                net_jobs['weather'] = pool.submit(
                    _fetch_or_skip, "weather",
                    lambda: _fetch_weather(bounds))
            elif resume and 'weather' in store:
                _log('weather',
                     "Weather already in zarr, skipping (resume=True)")

            for key, future in net_jobs.items():
                if key == 'wind':
                    wind_data = future.result()
                else:
                    weather_data = future.result()

    if wind_data is not None:
        _save_wind(store, wind_data, bounds)
    if weather_data is not None:
        _save_weather(store, weather_data, bounds)

    # ---- 6. Hydro ----
    if hydro and not (resume and 'hydro' in store):
        hydro_data = _compute_hydro_or_skip(dem)
        if hydro_data is not None:
            _save_hydro(store, hydro_data)
    elif resume and 'hydro' in store:
        _log('hydro', "Hydro already in zarr, skipping (resume=True)")

    # ---- 7. LOD roughness ----
    _save_roughness(store, dem)

    # ---- Done ----
    from .mesh_store import validate_scene
    issues = validate_scene(str(output_path))
    errors = [msg for lvl, msg in issues if lvl == "error"]
    warnings = [msg for lvl, msg in issues if lvl == "warning"]
    if errors:
        _log('done',
             f"Validation: {len(errors)} errors, {len(warnings)} warnings")
        for e in errors:
            _log('done', f"  ERROR: {e}")
    elif warnings:
        _log('done', f"Validation: OK ({len(warnings)} warnings)")
    else:
        _log('done', "Validation: OK")

    _log('done', f"Scene written to {output_path}")
    return str(output_path)


# ---------------------------------------------------------------------------
# Fetch helpers (wrap imports + error handling)
# ---------------------------------------------------------------------------

def _fetch_or_skip(label, fn):
    """Call *fn*, print errors but don't crash the build."""
    try:
        print(f"Fetching {label}...")
        return fn()
    except Exception as exc:
        print(f"  Skipped {label}: {exc}")
        return None


def _fetch_buildings_overture(bounds, cache_dir):
    from .remote_data import fetch_buildings
    cache_path = Path(cache_dir) / "buildings_overture.json"
    return fetch_buildings(bounds, cache_path=str(cache_path),
                           source='overture')


def _fetch_roads_overture(bounds, cache_dir):
    from .remote_data import fetch_roads
    cache_path = Path(cache_dir) / "roads_overture.json"
    return fetch_roads(bounds, cache_path=str(cache_path),
                       source='overture')


def _fetch_water_overture(bounds, cache_dir):
    from .remote_data import fetch_water
    cache_path = Path(cache_dir) / "water_overture.json"
    return fetch_water(bounds, cache_path=str(cache_path),
                       source='overture')


def _fetch_firms(bounds, cache_dir):
    from .remote_data import fetch_firms
    cache_path = Path(cache_dir) / "firms_24h.json"
    return fetch_firms(bounds, cache_path=str(cache_path))


def _adaptive_grid_size(bounds, points_per_degree=50, lo=3, hi=12):
    """Pick a grid_size that scales with bbox extent.

    Open-Meteo encodes all grid_size² lat/lon pairs in the URL, so
    the total must stay under ~8 KB.  grid_size=12 → 144 points
    → ~3600 chars, well within limits.  For weather/wind the spatial
    resolution is coarse anyway (~10-25 km), so 12 is plenty.
    """
    west, south, east, north = bounds
    span = max(abs(east - west), abs(north - south))
    n = int(round(span * points_per_degree))
    return max(lo, min(n, hi))


def _fetch_wind(bounds):
    from .remote_data import fetch_wind
    return fetch_wind(bounds, grid_size=_adaptive_grid_size(bounds))


def _fetch_weather(bounds):
    from .remote_data import fetch_weather
    return fetch_weather(bounds, grid_size=_adaptive_grid_size(bounds))


def _has_baked_meshes_da(da):
    """Check whether any geometries were placed on a DataArray."""
    try:
        baked = getattr(da.rtx, '_baked_meshes', None)
        return baked is not None and len(baked) > 0
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Zarr writers for non-mesh groups
# ---------------------------------------------------------------------------

def _save_wind(store, wind_data, bounds):
    """Write wind group per scene zarr spec."""
    wg = store.create_group('wind', overwrite=True)

    u = np.asarray(wind_data['u'], dtype=np.float32)
    v = np.asarray(wind_data['v'], dtype=np.float32)
    wg.create_array('u', data=u, chunks=u.shape, compressors=_BLOSC)
    wg.create_array('v', data=v, chunks=v.shape, compressors=_BLOSC)

    # Store lats/lons so the viewer can interpolate onto the terrain
    if 'lats' in wind_data:
        lats = np.asarray(wind_data['lats'], dtype=np.float32)
        lons = np.asarray(wind_data['lons'], dtype=np.float32)
        wg.create_array('lats', data=lats, chunks=lats.shape, compressors=_BLOSC)
        wg.create_array('lons', data=lons, chunks=lons.shape, compressors=_BLOSC)

    west, south, east, north = bounds
    wg.attrs['grid_bounds'] = {'x0': west, 'y0': south,
                               'x1': east, 'y1': north}
    wg.attrs['grid_size'] = int(u.shape[0])
    wg.attrs['source'] = 'open-meteo'
    wg.attrs['source_time'] = _iso_now()
    print(f"  Wind grid: {u.shape}")


def _save_weather(store, weather_data, bounds):
    """Write weather group per scene zarr spec."""
    wg = store.create_group('weather', overwrite=True)

    for key in ('cloud_cover', 'temperature', 'humidity',
                'pressure', 'precipitation'):
        if key in weather_data:
            arr = np.asarray(weather_data[key], dtype=np.float32)
            wg.create_array(key, data=arr, chunks=arr.shape, compressors=_BLOSC)

    # Store lats/lons for spatial reference
    if 'lats' in weather_data:
        lats = np.asarray(weather_data['lats'], dtype=np.float32)
        lons = np.asarray(weather_data['lons'], dtype=np.float32)
        wg.create_array('lats', data=lats, chunks=lats.shape, compressors=_BLOSC)
        wg.create_array('lons', data=lons, chunks=lons.shape, compressors=_BLOSC)

    west, south, east, north = bounds
    wg.attrs['grid_bounds'] = {'x0': west, 'y0': south,
                               'x1': east, 'y1': north}
    wg.attrs['grid_size'] = int(weather_data.get(
        'cloud_cover', weather_data.get('temperature', np.empty(0))
    ).shape[0]) if any(k in weather_data for k in
                       ('cloud_cover', 'temperature')) else 0
    wg.attrs['source'] = 'open-meteo'
    wg.attrs['source_time'] = _iso_now()

    saved = [k for k in ('cloud_cover', 'temperature', 'humidity',
                         'pressure', 'precipitation')
             if k in weather_data]
    print(f"  Weather: {', '.join(saved)}")


def _save_hydro(store, hydro_data):
    """Write hydro group per scene zarr spec."""
    hg = store.create_group('hydro', overwrite=True)

    for key in ('flow_accum', 'flow_dir_mfd', 'stream_order', 'stream_link'):
        if key in hydro_data:
            arr = np.asarray(hydro_data[key], dtype=np.float32
                             if key == 'flow_accum' else np.int32
                             if key in ('stream_order', 'stream_link')
                             else np.float32)
            hg.create_array(key, data=arr, chunks=arr.shape, compressors=_BLOSC)

    # Copy tuning parameters
    for attr in ('n_particles', 'max_age', 'trail_len', 'speed',
                 'accum_threshold', 'color', 'alpha', 'dot_radius'):
        if attr in hydro_data:
            val = hydro_data[attr]
            if isinstance(val, (list, tuple)):
                hg.attrs[attr] = list(val)
            else:
                hg.attrs[attr] = val

    print(f"  Hydro: flow_accum + flow_dir_mfd"
          f"{' + stream_order' if 'stream_order' in hydro_data else ''}")


def _save_roughness(store, dem, tile_size=64):
    """Compute per-tile roughness and write elevation_roughness group.

    Roughness = std of residuals from bilinear interpolation of the four
    tile corner elevations.  This lets the LOD manager promote rough
    tiles to higher detail and demote flat ones.
    """
    elev = dem.values.astype(np.float32)
    h, w = elev.shape
    th = (h + tile_size - 1) // tile_size
    tw = (w + tile_size - 1) // tile_size

    roughness = np.zeros((th, tw), dtype=np.float32)
    for tr in range(th):
        for tc in range(tw):
            r0 = tr * tile_size
            c0 = tc * tile_size
            r1 = min(r0 + tile_size, h)
            c1 = min(c0 + tile_size, w)
            tile = elev[r0:r1, c0:c1]

            valid = tile[~np.isnan(tile)]
            if len(valid) < 4:
                roughness[tr, tc] = 0.0
                continue

            # Bilinear fit from corners
            corners = np.array([
                tile[0, 0] if not np.isnan(tile[0, 0]) else np.nanmean(tile),
                tile[0, -1] if not np.isnan(tile[0, -1]) else np.nanmean(tile),
                tile[-1, 0] if not np.isnan(tile[-1, 0]) else np.nanmean(tile),
                tile[-1, -1] if not np.isnan(tile[-1, -1]) else np.nanmean(tile),
            ], dtype=np.float32)

            rows, cols = tile.shape
            yy, xx = np.mgrid[0:rows, 0:cols].astype(np.float32)
            fy = yy / max(rows - 1, 1)
            fx = xx / max(cols - 1, 1)
            interp = (corners[0] * (1 - fx) * (1 - fy) +
                      corners[1] * fx * (1 - fy) +
                      corners[2] * (1 - fx) * fy +
                      corners[3] * fx * fy)

            residuals = tile - interp
            mask = ~np.isnan(residuals)
            if mask.any():
                roughness[tr, tc] = float(np.std(residuals[mask]))

    rg = store.create_group('elevation_roughness', overwrite=True)
    rg.create_array('values', data=roughness,
                     chunks=roughness.shape, compressors=_BLOSC)
    rg.attrs['tile_size'] = tile_size
    rg.attrs['shape'] = [th, tw]
    print(f"  Roughness: {th}x{tw} tiles, "
          f"range [{roughness.min():.1f}, {roughness.max():.1f}]")


def _compute_hydro_or_skip(dem):
    """Compute MFD hydrology from the DEM, or skip on failure."""
    try:
        print("Computing hydrology from DEM...")
        from xrspatial import flow_direction_mfd, flow_accumulation_mfd
        import xarray as xr

        elev = dem.copy()
        # Fill NaN with minimum elevation so MFD doesn't break
        elev_np = elev.values.copy()
        nan_mask = np.isnan(elev_np)
        if nan_mask.any():
            elev_np[nan_mask] = np.nanmin(elev_np)
            elev = xr.DataArray(elev_np, coords=elev.coords, dims=elev.dims)

        fdir = flow_direction_mfd(elev)
        facc = flow_accumulation_mfd(elev, fdir)

        hydro = {
            'flow_accum': facc.values,
            'flow_dir_mfd': fdir.values,
        }

        # Stream order if available
        try:
            from xrspatial import stream_order_mfd
            sorder = stream_order_mfd(facc, fdir)
            hydro['stream_order'] = sorder.values
        except ImportError:
            pass

        print(f"  Flow accumulation shape: {facc.shape}")
        return hydro
    except Exception as exc:
        print(f"  Skipped hydro: {exc}")
        return None


# ---------------------------------------------------------------------------
# Scene loader
# ---------------------------------------------------------------------------

def explore_scene(zarr_path, **explore_kwargs):
    """Open a scene zarr in the interactive viewer.

    Loads the DEM, reads wind/weather/hydro groups if present, and
    launches ``explore()`` with all the stored data.

    Parameters
    ----------
    zarr_path : str or Path
        Path to a scene zarr produced by :func:`build_scene`.
    **explore_kwargs
        Override any ``explore()`` parameter (e.g. ``subsample=2``,
        ``colormap='terrain'``, ``lod=True``).
    """
    import zarr
    import xarray as xr

    zarr_path = str(zarr_path)

    # Load elevation
    ds = xr.open_zarr(zarr_path)
    if 'elevation' not in ds:
        raise ValueError(f"No 'elevation' array in {zarr_path}")
    da = ds['elevation']

    # Attach CRS
    if 'spatial_ref' in ds:
        crs_wkt = ds['spatial_ref'].attrs.get('crs_wkt', '')
        if crs_wkt:
            import rioxarray  # noqa: F401
            da = da.rio.write_crs(crs_wkt)

    # Materialize to GPU — the render kernel needs cupy-backed data.
    # open_zarr returns dask-backed arrays; compute + transfer to GPU.
    import numpy as np
    data = da.values  # materializes dask → numpy
    data = np.ascontiguousarray(data, dtype=np.float32)
    try:
        import cupy as cp
        da = da.copy(data=cp.asarray(data))
    except ImportError:
        da = da.copy(data=data)

    # Read optional groups
    store = zarr.open(zarr_path, mode='r', use_consolidated=False)

    wind_data = _load_wind(store)
    weather_data = _load_weather(store)
    hydro_data = _load_hydro(store)

    # Build explore kwargs — stored values as defaults, explicit wins
    defaults = {}
    if wind_data is not None:
        defaults['wind_data'] = wind_data
    if weather_data is not None:
        defaults['weather_data'] = weather_data
    if hydro_data is not None:
        defaults['hydro_data'] = hydro_data

    # Read render settings
    if 'render' in store:
        rg = store['render']
        for attr in ('colormap', 'color_stretch', 'mesh_type', 'subsample',
                     'shadows', 'ambient', 'sun_azimuth', 'sun_altitude',
                     'fog_density', 'denoise', 'lod'):
            if attr in rg.attrs:
                defaults[attr] = rg.attrs[attr]
        if 'fog_color' in rg.attrs:
            defaults['fog_color'] = tuple(rg.attrs['fog_color'])
        if 'ao_enabled' in rg.attrs and rg.attrs['ao_enabled']:
            defaults['ao_samples'] = rg.attrs.get('ao_samples', 4)
        if 'gi_bounces' in rg.attrs:
            defaults['gi_bounces'] = rg.attrs['gi_bounces']

    # Read camera start position
    if 'camera' in store:
        cg = store['camera']
        if 'position' in cg.attrs:
            defaults['start_position'] = tuple(cg.attrs['position'])

    # Always pass scene_zarr for mesh loading
    defaults['scene_zarr'] = zarr_path

    # Build a ZarrChunkSource for chunk-driven terrain if the elevation
    # array has reasonable chunk sizes for LOD tiling.
    if 'terrain_source' not in explore_kwargs:
        try:
            from .chunk_source import ZarrChunkSource
            elev_arr = store['elevation']
            chunks = elev_arr.chunks
            if isinstance(chunks, (list, tuple)):
                ch, cw = int(chunks[0]), int(chunks[1])
            else:
                ch = cw = int(chunks)
            # Only use chunk source if chunks are tile-sized (not huge)
            if ch <= 512 and cw <= 512 and ch == cw:
                # Compute pixel spacing from DataArray coords
                x_coords = da.coords.get('x')
                y_coords = da.coords.get('y')
                if x_coords is not None and len(x_coords) >= 2:
                    psx = abs(float(x_coords[1] - x_coords[0]))
                    psy = abs(float(y_coords[1] - y_coords[0]))
                else:
                    psx = psy = 1.0
                cs = ZarrChunkSource(
                    zarr_path, 'elevation',
                    pixel_spacing_x=psx, pixel_spacing_y=psy)
                defaults['terrain_source'] = cs
        except Exception:
            pass  # fall back to in-memory path

    # Explicit kwargs override stored defaults
    defaults.update(explore_kwargs)

    da.rtx.explore(**defaults)


def _load_wind(store):
    """Read wind group into the dict format explore() expects."""
    if 'wind' not in store:
        return None
    wg = store['wind']
    if 'u' not in wg or 'v' not in wg:
        return None
    result = {
        'u': np.array(wg['u']),
        'v': np.array(wg['v']),
    }
    if 'lats' in wg:
        result['lats'] = np.array(wg['lats'])
    if 'lons' in wg:
        result['lons'] = np.array(wg['lons'])
    return result


def _load_weather(store):
    """Read weather group into the dict format explore() expects."""
    if 'weather' not in store:
        return None
    wg = store['weather']
    result = {}
    for key in ('cloud_cover', 'temperature', 'humidity',
                'pressure', 'precipitation'):
        if key in wg:
            result[key] = np.array(wg[key])
    if 'lats' in wg:
        result['lats'] = np.array(wg['lats'])
    if 'lons' in wg:
        result['lons'] = np.array(wg['lons'])
    return result if result else None


def _load_hydro(store):
    """Read hydro group into the dict format explore() expects."""
    if 'hydro' not in store:
        return None
    hg = store['hydro']
    if 'flow_accum' not in hg:
        return None
    result = {}
    for key in ('flow_accum', 'flow_dir_mfd', 'stream_order', 'stream_link'):
        if key in hg:
            result[key] = np.array(hg[key])
    # Copy tuning attrs
    for attr in ('n_particles', 'max_age', 'trail_len', 'speed',
                 'accum_threshold', 'color', 'alpha', 'dot_radius'):
        if attr in hg.attrs:
            result[attr] = hg.attrs[attr]
    return result


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _iso_now():
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv=None):
    """Command-line entry point for scene building."""
    import argparse

    parser = argparse.ArgumentParser(
        prog="rtxpy-build-scene",
        description="Build a scene zarr from a bounding box.",
    )
    parser.add_argument(
        "bounds", nargs=4, type=float, metavar=("WEST", "SOUTH", "EAST", "NORTH"),
        help="Bounding box in WGS84 degrees",
    )
    parser.add_argument(
        "output", type=str,
        help="Output zarr path (e.g. scene.zarr)",
    )
    parser.add_argument(
        "--dem-source", default="copernicus",
        choices=["copernicus", "srtm", "usgs_10m", "usgs_1m"],
        help="DEM source (default: copernicus, 30m)",
    )
    parser.add_argument(
        "--crs", default=None,
        help="Target CRS (e.g. EPSG:32617). Auto-detects UTM if omitted.",
    )
    parser.add_argument(
        "--no-buildings", action="store_true",
        help="Skip building footprints",
    )
    parser.add_argument(
        "--no-roads", action="store_true",
        help="Skip road networks",
    )
    parser.add_argument(
        "--no-water", action="store_true",
        help="Skip water features",
    )
    parser.add_argument(
        "--no-wind", action="store_true",
        help="Skip wind data",
    )
    parser.add_argument(
        "--no-weather", action="store_true",
        help="Skip weather data",
    )
    parser.add_argument(
        "--hydro", action="store_true",
        help="Compute MFD hydrology from the DEM",
    )
    parser.add_argument(
        "--fires", action="store_true",
        help="Fetch NASA FIRMS fire detections (last 24 h)",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Skip layers already present in an existing zarr",
    )
    parser.add_argument(
        "--name", default=None,
        help="Scene name (defaults to output filename stem)",
    )
    parser.add_argument(
        "--cache-dir", default=None,
        help="Directory for intermediate caches",
    )

    args = parser.parse_args(argv)

    build_scene(
        bounds=tuple(args.bounds),
        output_path=args.output,
        dem_source=args.dem_source,
        crs=args.crs,
        buildings=not args.no_buildings,
        roads=not args.no_roads,
        water=not args.no_water,
        wind=not args.no_wind,
        weather=not args.no_weather,
        hydro=args.hydro,
        fires=args.fires,
        resume=args.resume,
        name=args.name,
        cache_dir=args.cache_dir,
    )


if __name__ == "__main__":
    main()
