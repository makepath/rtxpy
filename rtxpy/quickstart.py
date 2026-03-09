"""One-call launcher: DEM fetch -> analysis layers -> feature placement -> explore()."""

import warnings
from pathlib import Path


def _rasterize_waterways_to_dem(water_geojson, terrain, elev_np, ocean):
    """Rasterize Overture waterway features into a burn-depth grid.

    Carves LineStrings (rivers/streams) and fills Polygons (lakes) into
    the DEM so the MFD algorithm routes flow through known channels.

    Returns a float32 array of burn depths (positive = carve down),
    or None if no features were rasterized.
    """
    import numpy as np

    features = water_geojson.get('features', [])
    if not features:
        return None

    H, W = elev_np.shape
    burn = np.zeros((H, W), dtype=np.float32)

    # Build lon/lat → pixel transformer from the terrain's CRS + affine
    try:
        from pyproj import Transformer
        terrain_crs = terrain.rio.crs
        to_crs = Transformer.from_crs('EPSG:4326', terrain_crs,
                                       always_xy=True)
    except Exception:
        return None

    # Affine: pixel (col, row) ↔ projected (x, y)
    x_coords = terrain.coords[terrain.dims[-1]].values
    y_coords = terrain.coords[terrain.dims[-2]].values
    x0, x1 = float(x_coords[0]), float(x_coords[-1])
    y0, y1 = float(y_coords[0]), float(y_coords[-1])
    dx = (x1 - x0) / max(W - 1, 1)
    dy = (y1 - y0) / max(H - 1, 1)

    def lonlat_to_pixel(lon, lat):
        """Convert (lon, lat) → (col, row) in the DEM grid."""
        px, py = to_crs.transform(lon, lat)
        col = (px - x0) / dx
        row = (py - y0) / dy
        return col, row

    def densify_line(pixel_coords):
        """Walk polyline at 1-pixel steps → list of (row, col)."""
        pts = []
        for i in range(len(pixel_coords) - 1):
            c0, r0 = pixel_coords[i]
            c1, r1 = pixel_coords[i + 1]
            dc, dr = c1 - c0, r1 - r0
            n = max(int(max(abs(dr), abs(dc))), 1)
            for s in range(n + 1):
                t = s / n
                pts.append((int(round(r0 + dr * t)),
                            int(round(c0 + dc * t))))
        return pts

    # Burn depths by waterway type
    _LINE_BURN = {
        'river': 5.0, 'canal': 3.0,
        'stream': 2.0, 'drain': 1.0, 'ditch': 0.5,
    }
    _POLY_BURN = 3.0  # lakes / reservoirs

    n_burned = 0
    for feat in features:
        geom = feat.get('geometry', {})
        gtype = geom.get('type', '')
        subtype = (feat.get('properties') or {}).get('subtype', '')

        if gtype == 'LineString':
            coords = geom.get('coordinates', [])
            if len(coords) < 2:
                continue
            depth = _LINE_BURN.get(subtype, 2.0)
            px = [lonlat_to_pixel(c[0], c[1]) for c in coords]
            for rr, cc in densify_line(px):
                if 0 <= rr < H and 0 <= cc < W and not ocean[rr, cc]:
                    if depth > burn[rr, cc]:
                        burn[rr, cc] = depth
            n_burned += 1

        elif gtype in ('Polygon', 'MultiPolygon'):
            all_rings = []
            if gtype == 'Polygon':
                all_rings = geom.get('coordinates', [])
            else:
                for poly in geom.get('coordinates', []):
                    all_rings.extend(poly)
            for ring in all_rings:
                if len(ring) < 3:
                    continue
                px = [lonlat_to_pixel(c[0], c[1]) for c in ring]
                # Outline
                for rr, cc in densify_line(px):
                    if 0 <= rr < H and 0 <= cc < W and not ocean[rr, cc]:
                        if _POLY_BURN > burn[rr, cc]:
                            burn[rr, cc] = _POLY_BURN
                # Fill interior via scanline
                pr = np.array([p[1] for p in px], dtype=np.float64)
                pc = np.array([p[0] for p in px], dtype=np.float64)
                r_min = max(int(pr.min()), 0)
                r_max = min(int(pr.max()), H - 1)
                n_verts = len(pr)
                for row in range(r_min, r_max + 1):
                    xings = []
                    for j in range(n_verts):
                        j1 = (j + 1) % n_verts
                        r0, r1 = pr[j], pr[j1]
                        if (r0 <= row < r1) or (r1 <= row < r0):
                            t = (row - r0) / (r1 - r0)
                            xings.append(pc[j] + t * (pc[j1] - pc[j]))
                    xings.sort()
                    for k in range(0, len(xings) - 1, 2):
                        c_lo = max(int(round(xings[k])), 0)
                        c_hi = min(int(round(xings[k + 1])), W - 1)
                        for cc in range(c_lo, c_hi + 1):
                            if not ocean[row, cc]:
                                if _POLY_BURN > burn[row, cc]:
                                    burn[row, cc] = _POLY_BURN
            n_burned += 1

    if n_burned > 0:
        # Buffer the burn by a few pixels with tapered depth —
        # widens channels so D8 catches more flow into them.
        from scipy.ndimage import maximum_filter as _max_filt
        from scipy.ndimage import gaussian_filter as _gauss_filt
        # Expand burn mask by 3 px radius
        buffered = _max_filt(burn, size=7)
        # Taper edges: Gaussian blur gives smooth falloff
        buffered = _gauss_filt(buffered, sigma=1.5).astype(np.float32)
        # Keep the sharper original where it's deeper
        burn = np.maximum(burn, buffered)
        burn[ocean] = 0.0

        n_cells = int((burn > 0).sum())
        print(f"  Burned {n_burned} waterway features into DEM "
              f"({n_cells} cells, max depth {burn.max():.1f} m)")
        return burn
    return None


def _burn_waterways_to_grids(water_geojson, terrain, sl_grid, so_grid,
                             ocean, next_link_id):
    """Burn Overture waterway features into stream_link and stream_order grids.

    Marks waterway cells in *sl_grid* (with *next_link_id* for cells not
    already part of the D8 stream network) and upgrades *so_grid* with
    equivalent Strahler orders.  Both arrays are modified in-place.

    Returns the number of features burned.
    """
    import numpy as np

    features = water_geojson.get('features', [])
    if not features:
        return 0

    H, W = sl_grid.shape

    try:
        from pyproj import Transformer
        terrain_crs = terrain.rio.crs
        to_crs = Transformer.from_crs('EPSG:4326', terrain_crs,
                                       always_xy=True)
    except Exception:
        return 0

    x_coords = terrain.coords[terrain.dims[-1]].values
    y_coords = terrain.coords[terrain.dims[-2]].values
    x0, x1 = float(x_coords[0]), float(x_coords[-1])
    y0, y1 = float(y_coords[0]), float(y_coords[-1])
    dx = (x1 - x0) / max(W - 1, 1)
    dy = (y1 - y0) / max(H - 1, 1)

    def lonlat_to_pixel(lon, lat):
        px, py = to_crs.transform(lon, lat)
        return (px - x0) / dx, (py - y0) / dy

    def densify_line(pixel_coords):
        pts = []
        for i in range(len(pixel_coords) - 1):
            c0, r0 = pixel_coords[i]
            c1, r1 = pixel_coords[i + 1]
            dc, dr = c1 - c0, r1 - r0
            n = max(int(max(abs(dr), abs(dc))), 1)
            for s in range(n + 1):
                t = s / n
                pts.append((int(round(r0 + dr * t)),
                            int(round(c0 + dc * t))))
        return pts

    _LINE_ORDER = {
        'river': 5, 'canal': 4,
        'stream': 2, 'drain': 1, 'ditch': 1,
    }

    def _mark(rr, cc, order):
        if 0 <= rr < H and 0 <= cc < W and not ocean[rr, cc]:
            if sl_grid[rr, cc] <= 0:
                sl_grid[rr, cc] = next_link_id
            if order > so_grid[rr, cc]:
                so_grid[rr, cc] = order

    n_burned = 0
    for feat in features:
        geom = feat.get('geometry', {})
        gtype = geom.get('type', '')
        subtype = (feat.get('properties') or {}).get('subtype', '')

        if gtype == 'LineString':
            coords = geom.get('coordinates', [])
            if len(coords) < 2:
                continue
            order = _LINE_ORDER.get(subtype, 2)
            px = [lonlat_to_pixel(c[0], c[1]) for c in coords]
            for rr, cc in densify_line(px):
                _mark(rr, cc, order)
            n_burned += 1

        elif gtype in ('Polygon', 'MultiPolygon'):
            all_rings = []
            if gtype == 'Polygon':
                all_rings = geom.get('coordinates', [])
            else:
                for poly in geom.get('coordinates', []):
                    all_rings.extend(poly)
            poly_order = max(_LINE_ORDER.get(subtype, 2), 5)
            for ring in all_rings:
                if len(ring) < 3:
                    continue
                px = [lonlat_to_pixel(c[0], c[1]) for c in ring]
                for rr, cc in densify_line(px):
                    _mark(rr, cc, poly_order)
                # Scanline fill
                pr = np.array([p[1] for p in px], dtype=np.float64)
                pc = np.array([p[0] for p in px], dtype=np.float64)
                r_min = max(int(pr.min()), 0)
                r_max = min(int(pr.max()), H - 1)
                n_verts = len(pr)
                for row in range(r_min, r_max + 1):
                    xings = []
                    for j in range(n_verts):
                        j1 = (j + 1) % n_verts
                        r0, r1 = pr[j], pr[j1]
                        if (r0 <= row < r1) or (r1 <= row < r0):
                            t = (row - r0) / (r1 - r0)
                            xings.append(pc[j] + t * (pc[j1] - pc[j]))
                    xings.sort()
                    for k in range(0, len(xings) - 1, 2):
                        c_lo = max(int(round(xings[k])), 0)
                        c_hi = min(int(round(xings[k + 1])), W - 1)
                        for cc in range(c_lo, c_hi + 1):
                            _mark(row, cc, poly_order)
            n_burned += 1

    return n_burned


def _trace_tributaries_flow_path(fd, fa, water_geojson, terrain, ocean,
                                 threshold=50):
    """Trace tributary network via xrspatial.flow_path.

    Pass 1: rasterize Overture waterway vertices as seed points, trace
    downstream through D8 → main channel mask.
    Pass 2: seed from high-accumulation cells NOT on the main channel,
    trace downstream → tributary mask.

    Returns (trib_cells, ww_net) — both bool arrays (H, W).
    """
    import numpy as np
    from xrspatial import flow_path as _flow_path

    features = water_geojson.get('features', [])
    H, W = ocean.shape

    # -- coordinate transform (lon/lat → pixel) ------------------------------
    try:
        from pyproj import Transformer
        terrain_crs = terrain.rio.crs
        to_crs = Transformer.from_crs('EPSG:4326', terrain_crs,
                                       always_xy=True)
    except Exception:
        return None, None

    x_coords = terrain.coords[terrain.dims[-1]].values
    y_coords = terrain.coords[terrain.dims[-2]].values
    x0, x1 = float(x_coords[0]), float(x_coords[-1])
    y0, y1 = float(y_coords[0]), float(y_coords[-1])
    dx = (x1 - x0) / max(W - 1, 1)
    dy = (y1 - y0) / max(H - 1, 1)

    def lonlat_to_pixel(lon, lat):
        px, py = to_crs.transform(lon, lat)
        return (px - x0) / dx, (py - y0) / dy

    def densify_line(pixel_coords):
        pts = []
        for i in range(len(pixel_coords) - 1):
            c0, r0 = pixel_coords[i]
            c1, r1 = pixel_coords[i + 1]
            dc_, dr_ = c1 - c0, r1 - r0
            n = max(int(max(abs(dr_), abs(dc_))), 1)
            for s in range(n + 1):
                t = s / n
                pts.append((int(round(r0 + dr_ * t)),
                            int(round(c0 + dc_ * t))))
        return pts

    # -- Pass 1: waterway seeds → main channel via flow_path -----------------
    ww_seeds = np.full((H, W), np.nan, dtype=np.float32)
    n_seed = 0
    label = 1.0
    for feat in features:
        geom = feat.get('geometry', {})
        gtype = geom.get('type', '')

        if gtype == 'LineString':
            coords = geom.get('coordinates', [])
            if len(coords) < 2:
                continue
            px = [lonlat_to_pixel(c[0], c[1]) for c in coords]
            for rr, cc in densify_line(px):
                if 0 <= rr < H and 0 <= cc < W and not ocean[rr, cc]:
                    ww_seeds[rr, cc] = label
                    n_seed += 1
            label += 1.0

        elif gtype in ('Polygon', 'MultiPolygon'):
            all_rings = []
            if gtype == 'Polygon':
                all_rings = geom.get('coordinates', [])
            else:
                for poly in geom.get('coordinates', []):
                    all_rings.extend(poly)
            for ring in all_rings:
                if len(ring) < 3:
                    continue
                px = [lonlat_to_pixel(c[0], c[1]) for c in ring]
                for rr, cc in densify_line(px):
                    if 0 <= rr < H and 0 <= cc < W and not ocean[rr, cc]:
                        ww_seeds[rr, cc] = label
                        n_seed += 1
            label += 1.0

    if n_seed == 0:
        return None, None

    ww_seeds_da = terrain.copy(data=ww_seeds)
    del ww_seeds
    ww_traced = _flow_path(fd, ww_seeds_da)
    del ww_seeds_da
    ww_traced_np = ww_traced.data.get() if hasattr(ww_traced.data, 'get') \
        else np.asarray(ww_traced.data)
    ww_net = np.isfinite(ww_traced_np)
    del ww_traced, ww_traced_np
    n_channel = int(ww_net.sum())

    # -- Pass 2: headwater seeds → tributaries via flow_path -----------------
    fa_np = fa.data.get() if hasattr(fa.data, 'get') else np.asarray(fa.data)
    fa_np = np.nan_to_num(fa_np, nan=0.0)

    hw_seeds = np.full((H, W), np.nan, dtype=np.float32)
    hw_mask = (fa_np >= threshold) & (~ww_net) & (~ocean)
    hw_seeds[hw_mask] = 1.0
    n_hw = int(hw_mask.sum())
    del fa_np, hw_mask

    if n_hw > 0:
        hw_seeds_da = terrain.copy(data=hw_seeds)
        del hw_seeds
        hw_traced = _flow_path(fd, hw_seeds_da)
        del hw_seeds_da
        hw_traced_np = hw_traced.data.get() if hasattr(hw_traced.data, 'get') \
            else np.asarray(hw_traced.data)
        trib_cells = np.isfinite(hw_traced_np) & (~ww_net)
        del hw_traced, hw_traced_np
        n_trib = int(trib_cells.sum())
    else:
        del hw_seeds
        trib_cells = np.zeros((H, W), dtype=bool)
        n_trib = 0

    print(f"  flow_path: {n_seed} waterway seeds → {n_channel} channel cells, "
          f"{n_hw} headwaters → {n_trib} tributary cells")
    return trib_cells, ww_net


def quickstart(
    name,
    bounds,
    crs,
    source='copernicus',
    features=None,
    tiles='satellite',
    tile_zoom=None,
    wind=True,
    weather=False,
    hydro=False,
    coast_distance=False,
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
    hydro : bool
        Compute flow accumulation from the terrain using xarray-spatial
        and enable MFD hydro flow particle animation (Shift+Y).
        Default ``False``.
    coast_distance : bool
        Compute terrain-aware surface distance from the coast using
        xrspatial's ``surface_distance`` (3-D Dijkstra).  Adds a
        ``coast_distance`` layer visible as a G-key overlay.
        Default ``False``.
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
    print("Building Dataset...")
    ds = xr.Dataset({
        'elevation': terrain.rename(None),
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

    # -- weather --------------------------------------------------------------
    weather_data = None
    if weather:
        try:
            from .remote_data import fetch_weather as _fetch_weather
            weather_data = _fetch_weather(bounds, grid_size=15)
        except Exception as e:
            print(f"Skipping weather: {e}")

    # -- wind -----------------------------------------------------------------
    wind_data = None
    if wind:
        try:
            from .remote_data import fetch_wind as _fetch_wind
            wind_data = _fetch_wind(bounds, grid_size=15)
        except Exception as e:
            print(f"Skipping wind: {e}")

    # -- hydro ----------------------------------------------------------------
    hydro_data = None
    if hydro:
        try:
            import gc
            from xrspatial import fill as _fill
            from xrspatial import flow_direction_mfd as _flow_direction_mfd
            from xrspatial import flow_accumulation_mfd as _flow_accumulation_mfd
            from xrspatial import stream_order_mfd as _stream_order_mfd
            from xrspatial import stream_link_mfd as _stream_link_mfd

            print("Conditioning DEM for hydrological flow...")
            from scipy.ndimage import uniform_filter as _uniform_filter

            # 1. Prepare elevation: ocean (0-fill) → low sentinel so
            #    fill() routes coastal depressions toward the sea.
            _data = terrain.data
            is_cupy = hasattr(_data, 'get')
            elev_np = _data.get() if is_cupy else np.array(_data)
            elev_np = elev_np.astype(np.float32)
            ocean = (elev_np == 0.0) | np.isnan(elev_np)
            elev_np[ocean] = -100.0

            # Pre-load waterway GeoJSON once (used in steps 1b, 6b,
            # 6c, and for hydro_data). Avoids 4× redundant file reads.
            _water_geojson = None
            if 'water' in feat:
                _wc = cache_dir / f"{name}_water.geojson"
                if _wc.exists():
                    import json as _json_ww
                    try:
                        with open(_wc) as _wf:
                            _water_geojson = _json_ww.load(_wf)
                    except Exception as _e:
                        print(f"  Waterway GeoJSON load failed: {_e}")

            # 1b. Burn Overture waterways into the DEM — carve known
            #     river/stream channels so MFD routes through them.
            if _water_geojson is not None:
                try:
                    _ww_burn = _rasterize_waterways_to_dem(
                        _water_geojson, terrain, elev_np, ocean)
                    if _ww_burn is not None:
                        elev_np -= _ww_burn
                        elev_np[ocean] = -100.0
                    del _ww_burn
                except Exception as _e:
                    print(f"  Waterway DEM burn skipped: {_e}")

            # 2. Smooth (15×15 ≈ 450 m at 30 m) to remove noise pits
            #    that fragment the drainage network.
            smoothed = _uniform_filter(elev_np, size=15, mode='nearest')
            smoothed[ocean] = -100.0
            del elev_np

            # 3. Fill remaining depressions, then resolve flats.
            if is_cupy:
                import cupy as _cp
                _sm = _cp.asarray(smoothed)
            else:
                _sm = smoothed
            del smoothed
            filled = _fill(terrain.copy(data=_sm))
            fill_depth = filled.data - _sm
            resolved = filled.data + fill_depth * 0.01
            del filled, fill_depth, _sm

            # 3b. Distance-to-ocean gradient — ensures flat areas
            #     drain coherently toward the coast.
            from scipy.ndimage import distance_transform_edt as _dist_edt
            dist_to_ocean = _dist_edt(~ocean).astype(np.float32)
            ocean_gradient = dist_to_ocean * 0.0001
            del dist_to_ocean

            if is_cupy:
                resolved = resolved + _cp.asarray(ocean_gradient)
                resolved[_cp.asarray(ocean)] = -100.0
            else:
                resolved += ocean_gradient
                resolved[ocean] = -100.0
            del ocean_gradient

            # 3c. Channel burning — compute an initial MFD flow
            #     accumulation, then lower high-accumulation cells
            #     to carve channels into the DEM. Re-fill and
            #     re-compute so streams connect into a network.
            xp = _cp if is_cupy else np   # array module for GPU/CPU
            _fd_mfd0 = _flow_direction_mfd(terrain.copy(data=resolved))
            _fa0 = _flow_accumulation_mfd(_fd_mfd0)
            del _fd_mfd0
            _fa0_data = xp.nan_to_num(_fa0.data, nan=0.0)
            del _fa0

            # Burn proportional to log(accumulation): cells with
            # more upstream area get carved deeper (up to ~2 m).
            _log_acc = xp.log10(xp.clip(_fa0_data, 1, None))
            _log_max = max(float(_log_acc.max()), 1.0)
            _burn = (_log_acc / _log_max) * 2.0  # 0–2 m carve
            if is_cupy:
                _burn[_cp.asarray(ocean)] = 0.0
            else:
                _burn[ocean] = 0.0
            del _fa0_data, _log_acc

            if is_cupy:
                resolved = resolved - _burn.astype(_cp.float32)
                resolved[_cp.asarray(ocean)] = -100.0
            else:
                resolved -= _burn.astype(np.float32)
                resolved[ocean] = -100.0
            del _burn

            # Re-fill after burning to remove any new micro-pits
            filled2 = _fill(terrain.copy(data=resolved))
            fill_depth2 = filled2.data - resolved
            resolved = filled2.data + fill_depth2 * 0.01
            del filled2, fill_depth2

            # Final jitter to break remaining ties
            if is_cupy:
                _cp.random.seed(0)
                resolved = resolved + _cp.random.uniform(
                    0, 0.0001, resolved.shape, dtype=_cp.float32)
                resolved[_cp.asarray(ocean)] = -100.0
            else:
                np.random.seed(0)
                resolved += np.random.uniform(
                    0, 0.0001, resolved.shape).astype(np.float32)
                resolved[ocean] = -100.0

            # Free CuPy cached blocks before the next heavy step
            if is_cupy:
                _cp.get_default_memory_pool().free_all_blocks()
            gc.collect()

            # 4. Compute final MFD flow direction and accumulation.
            _resolved_da = terrain.copy(data=resolved)
            fd_mfd = _flow_direction_mfd(_resolved_da)
            fa_mfd = _flow_accumulation_mfd(fd_mfd)
            del _resolved_da, resolved

            # 5. Compute Strahler stream order (MFD) — only stream
            #    cells (accum >= threshold) get an order; rest are NaN.
            so = _stream_order_mfd(fd_mfd, fa_mfd, threshold=50)

            # 5b. Compute stream link (MFD) — unique segment IDs.
            sl = _stream_link_mfd(fd_mfd, fa_mfd, threshold=50)

            # 6. Mask ocean back to NaN in the output grids.
            fa_mfd_out = fa_mfd.data
            fd_mfd_out = fd_mfd.data   # (8, H, W)
            so_out = so.data
            _sl_out = sl.data
            if is_cupy:
                ocean_gpu = _cp.asarray(ocean)
                fa_mfd_out[ocean_gpu] = _cp.nan
                fd_mfd_out[:, ocean_gpu] = _cp.nan
                so_out[ocean_gpu] = _cp.nan
                _sl_out[ocean_gpu] = _cp.nan
            else:
                fa_mfd_out[ocean] = np.nan
                fd_mfd_out[:, ocean] = np.nan
                so_out[ocean] = np.nan
                _sl_out[ocean] = np.nan

            # Clean stream_link for overlay (stay on GPU when possible)
            _sl_clean = xp.nan_to_num(_sl_out, nan=0.0).astype(xp.float32)

            # 6b. Burn Overture waterways into stream_link + stream_order
            #     so they appear in the overlay with the water shader.
            if _water_geojson is not None:
                try:
                    _so_np = so_out.get() if is_cupy else np.asarray(so_out)
                    _so_np = np.nan_to_num(_so_np, nan=0.0).astype(
                        np.float32)
                    _sl_np2 = _sl_clean.get() if is_cupy else np.asarray(
                        _sl_clean)
                    _sl_np2 = np.array(_sl_np2, dtype=np.float32)
                    _max_link = int(_sl_np2.max()) + 1
                    _n_ww = _burn_waterways_to_grids(
                        _water_geojson, terrain, _sl_np2, _so_np,
                        ocean, _max_link)
                    if _n_ww > 0:
                        if is_cupy:
                            _sl_clean = _cp.asarray(_sl_np2)
                            so_out = _cp.asarray(_so_np)
                        else:
                            _sl_clean = _sl_np2
                            so_out = _so_np
                        _sl_out = _sl_clean  # keep in sync
                        print(f"  Burned {_n_ww} waterway features "
                              f"into stream_link overlay")
                    del _so_np, _sl_np2
                except Exception as _e2:
                    print(f"  Waterway overlay burn skipped: {_e2}")

            # 6c. Trace tributary network via flow_path (D8-only).
            #     Compute a lightweight D8 fd/fa from the raw terrain
            #     since the conditioned DEM was already released.
            if _water_geojson is not None:
                try:
                    from xrspatial import (
                        flow_direction as _flow_direction,
                        flow_accumulation as _flow_accumulation,
                    )
                    _fd_d8 = _flow_direction(terrain)
                    _fa_d8 = _flow_accumulation(_fd_d8)
                    _trib, _ww_net = _trace_tributaries_flow_path(
                        _fd_d8, _fa_d8, _water_geojson, terrain, ocean)
                    del _fd_d8, _fa_d8
                    if _trib is not None:
                        _so_np3 = so_out.get() if is_cupy else np.asarray(so_out)
                        _so_np3 = np.nan_to_num(_so_np3, nan=0.0).astype(
                            np.float32)
                        _sl_np3 = _sl_clean.get() if is_cupy else np.asarray(
                            _sl_clean)
                        _sl_np3 = np.array(_sl_np3, dtype=np.float32)
                        _max_link3 = int(_sl_np3.max()) + 1
                        _new_ww = _ww_net & (_sl_np3 == 0) & (~ocean)
                        _sl_np3[_new_ww] = _max_link3
                        _so_np3[_new_ww] = np.maximum(
                            _so_np3[_new_ww], 3.0)
                        _max_link3 += 1
                        _new_trib = _trib & (_sl_np3 == 0) & (~ocean)
                        _sl_np3[_new_trib] = _max_link3
                        _so_np3[_new_trib] = np.maximum(
                            _so_np3[_new_trib], 1.0)
                        if is_cupy:
                            _sl_clean = _cp.asarray(_sl_np3)
                            so_out = _cp.asarray(_so_np3)
                        else:
                            _sl_clean = _sl_np3
                            so_out = _so_np3
                        _sl_out = _sl_clean
                        del _so_np3, _sl_np3, _trib, _ww_net
                except ImportError:
                    print("  flow_path tracing skipped "
                          "(xrspatial.flow_path unavailable)")
                except Exception as _e3:
                    print(f"  flow_path tracing skipped: {_e3}")

            # Drop xrspatial DataArray wrappers — we only need .data
            del fd_mfd, fa_mfd, so, sl

            ds['stream_link'] = terrain.copy(data=_sl_clean).rename(None)

            hydro_data = {
                'flow_accum': fa_mfd_out,
                'flow_dir_mfd': fd_mfd_out,
                'stream_order': so_out,
                'stream_link': _sl_out,
                'accum_threshold': 50,
            }
            del fa_mfd_out, fd_mfd_out, _sl_out
            # Pass overrides from explore_kwargs if present
            for key in ('n_particles', 'max_age', 'trail_len', 'speed',
                        'accum_threshold', 'color', 'alpha', 'dot_radius'):
                hydro_key = f'hydro_{key}'
                if hydro_key in explore_kwargs:
                    hydro_data[key] = explore_kwargs.pop(hydro_key)
            # Attach cached Overture waterway features for particle
            # injection and unified water-shader rendering.
            if _water_geojson is not None:
                ww_feats = [
                    f for f in _water_geojson.get('features', [])
                    if f.get('geometry', {}).get('type') in
                       ('LineString', 'Polygon', 'MultiPolygon')
                ]
                if ww_feats:
                    hydro_data['waterway_geojson'] = {
                        'type': 'FeatureCollection',
                        'features': ww_feats,
                    }
                    n_lines = sum(
                        1 for f in ww_feats
                        if f['geometry']['type'] == 'LineString')
                    n_polys = len(ww_feats) - n_lines
                    parts = []
                    if n_lines:
                        parts.append(f"{n_lines} LineStrings")
                    if n_polys:
                        parts.append(f"{n_polys} Polygons")
                    print(f"  Loaded {' + '.join(parts)} "
                          f"for hydro overlay")
                del ww_feats

            del _water_geojson, ocean, _sl_clean
            if is_cupy:
                _cp.get_default_memory_pool().free_all_blocks()
            gc.collect()

            print(f"  Flow direction + accumulation computed on "
                  f"{terrain.shape[0]}x{terrain.shape[1]} grid")
        except Exception as e:
            print(f"Skipping hydro: {e}")

    # -- coast distance -------------------------------------------------------
    if coast_distance:
        try:
            import gc as _gc_cd
            from xrspatial import surface_distance as _surface_distance
            from scipy.ndimage import binary_erosion as _binary_erosion

            _data = terrain.data
            _elev = _data.get() if hasattr(_data, 'get') else np.array(_data)
            _ocean = (_elev == 0.0) | np.isnan(_elev)
            _land = ~_ocean

            # Coast = land cells adjacent to ocean
            _coast = _land & ~_binary_erosion(_land)
            del _land

            # Target raster: 1.0 at coast, 0.0 elsewhere
            _targets = np.zeros_like(_elev, dtype=np.float32)
            _targets[_coast] = 1.0
            del _coast

            # Elevation with ocean as NaN barriers
            _elev_clean = _elev.astype(np.float32).copy()
            _elev_clean[_ocean] = np.nan
            del _elev, _ocean

            _dist = _surface_distance(
                terrain.copy(data=_targets),
                elevation=terrain.copy(data=_elev_clean),
                method='planar',
            )
            del _targets, _elev_clean
            ds['coast_distance'] = _dist.rename(None)
            del _dist
            _gc_cd.collect()
            print("Surface distance from coast computed")
        except Exception as e:
            print(f"Skipping coast distance: {e}")

    # -- explore --------------------------------------------------------------
    defaults = dict(
        width=2048, height=1600, render_scale=0.5,
        color_stretch='cbrt', subsample=1, repl=True,
    )
    defaults.update(explore_kwargs)

    print("\nLaunching explore...\n")
    ds.rtx.explore(z='elevation', scene_zarr=zarr_path,
                   wind_data=wind_data, weather_data=weather_data,
                   hydro_data=hydro_data,
                   gtfs_data=gtfs_data, **defaults)


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
