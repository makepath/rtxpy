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
    water=True,
    wind=True,
    weather=True,
    hydro=False,
    name=None,
    cache_dir=None,
):
    """Fetch data for a bounding box and write a scene zarr.

    Parameters
    ----------
    bounds : tuple
        (west, south, east, north) in WGS84 degrees.
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
    water : bool
        Fetch and place Overture Maps water features.
    wind : bool
        Fetch Open-Meteo wind velocity grids.
    weather : bool
        Fetch Open-Meteo weather (cloud cover, temperature, etc.).
    hydro : bool
        Compute MFD hydrological flow from the DEM.
    name : str or None
        Human-readable scene name. Defaults to the output filename stem.
    cache_dir : str or Path or None
        Directory for intermediate tile caches. Defaults to a ``.cache``
        sibling of *output_path*.
    """
    import zarr

    output_path = Path(output_path)
    if name is None:
        name = output_path.stem

    if cache_dir is None:
        cache_dir = output_path.parent / ".cache"
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    west, south, east, north = bounds

    # ---- 1. Elevation ----
    print(f"Fetching {dem_source} DEM for "
          f"({west:.4f}, {south:.4f}, {east:.4f}, {north:.4f})...")

    from .remote_data import fetch_dem
    dem = fetch_dem(bounds, str(output_path), source=dem_source, crs=crs,
                    cache_dir=str(cache_dir))
    print(f"  DEM shape: {dem.shape}, CRS: {dem.rio.crs}")

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
    print("Triangulating terrain...")
    terrain = dem.copy()
    terrain.data = np.ascontiguousarray(terrain.data)
    terrain.rtx.triangulate()

    # ---- 3. Buildings (Overture) ----
    if buildings:
        buildings_geojson = _fetch_or_skip(
            "Overture buildings",
            lambda: _fetch_buildings_overture(bounds, cache_dir),
        )
        if buildings_geojson and buildings_geojson.get('features'):
            n = len(buildings_geojson['features'])
            print(f"  Placing {n} buildings...")
            terrain.rtx.place_buildings(buildings_geojson)

    # ---- 4. Water (Overture) ----
    if water:
        water_geojson = _fetch_or_skip(
            "Overture water",
            lambda: _fetch_water_overture(bounds, cache_dir),
        )
        if water_geojson and water_geojson.get('features'):
            n = len(water_geojson['features'])
            print(f"  Placing {n} water features...")
            terrain.rtx.place_water(water_geojson)

    # ---- 5. Save meshes ----
    if _has_baked_meshes_da(terrain):
        print("Saving meshes to zarr...")
        terrain.rtx.save_meshes(str(output_path))

    # ---- 6. Wind ----
    if wind:
        wind_data = _fetch_or_skip(
            "wind",
            lambda: _fetch_wind(bounds),
        )
        if wind_data is not None:
            _save_wind(store, wind_data, bounds)

    # ---- 7. Weather ----
    if weather:
        weather_data = _fetch_or_skip(
            "weather",
            lambda: _fetch_weather(bounds),
        )
        if weather_data is not None:
            _save_weather(store, weather_data, bounds)

    # ---- 8. Hydro ----
    if hydro:
        hydro_data = _compute_hydro_or_skip(dem)
        if hydro_data is not None:
            _save_hydro(store, hydro_data)

    # ---- Done ----
    from .mesh_store import validate_scene
    issues = validate_scene(str(output_path))
    errors = [msg for lvl, msg in issues if lvl == "error"]
    warnings = [msg for lvl, msg in issues if lvl == "warning"]
    if errors:
        print(f"Validation: {len(errors)} errors, {len(warnings)} warnings")
        for e in errors:
            print(f"  ERROR: {e}")
    elif warnings:
        print(f"Validation: OK ({len(warnings)} warnings)")
    else:
        print("Validation: OK")

    print(f"Scene written to {output_path}")
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


def _fetch_water_overture(bounds, cache_dir):
    from .remote_data import fetch_water
    cache_path = Path(cache_dir) / "water_overture.json"
    return fetch_water(bounds, cache_path=str(cache_path),
                       source='overture')


def _fetch_wind(bounds):
    from .remote_data import fetch_wind
    return fetch_wind(bounds)


def _fetch_weather(bounds):
    from .remote_data import fetch_weather
    return fetch_weather(bounds)


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
    if 'wind' in store:
        del store['wind']
    wg = store.create_group('wind')

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
    if 'weather' in store:
        del store['weather']
    wg = store.create_group('weather')

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
    if 'hydro' in store:
        del store['hydro']
    hg = store.create_group('hydro')

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
        water=not args.no_water,
        wind=not args.no_wind,
        weather=not args.no_weather,
        hydro=args.hydro,
        name=args.name,
        cache_dir=args.cache_dir,
    )


if __name__ == "__main__":
    main()
