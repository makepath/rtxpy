"""Interactive playground for rtxpy terrain exploration.

This example demonstrates real-time terrain exploration using
GPU-accelerated ray tracing via the xarray accessor's explore mode.

Builds an xr.Dataset with elevation, slope, aspect, and quantile layers.
Press G to cycle between layers. Satellite tiles are draped on the terrain
automatically — press U to toggle tile overlay on/off.

GeoJSON landmarks (Wizard Island, Phantom Ship, etc.) and a simplified
Crater Lake outline + Rim Drive path are placed as 3D geometry. Press G/N
to cycle and jump between geometry layers.

Requirements:
    pip install rtxpy[all] matplotlib xarray rioxarray requests pyproj Pillow osmnx duckdb
"""

import warnings

import numpy as np
import xarray as xr
from pathlib import Path

from xrspatial import slope, aspect, quantile

# Import rtxpy to register the .rtx accessor
from rtxpy import fetch_dem, fetch_buildings, fetch_roads, fetch_water
import rtxpy

BOUNDS = (-122.3, 42.8, -121.9, 43.0)
CRS = 'EPSG:5070'
CACHE = Path(__file__).parent


def load_terrain():
    """Load Crater Lake terrain data, downloading if necessary."""
    terrain = fetch_dem(
        bounds=BOUNDS,
        output_path=CACHE / "crater_lake_national_park.zarr",
        source='srtm',
        crs=CRS,
    )

    # Subsample for faster interactive performance
    terrain = terrain[::2, ::2]

    # Ensure contiguous array before GPU transfer
    terrain.data = np.ascontiguousarray(terrain.data)

    # Get stats before GPU transfer
    elev_min = float(terrain.min())
    elev_max = float(terrain.max())

    # Convert to cupy for GPU processing using the accessor
    terrain = terrain.rtx.to_cupy()

    print(f"Terrain loaded: {terrain.shape}, elevation range: "
          f"{elev_min:.0f}m to {elev_max:.0f}m")

    return terrain


if __name__ == "__main__":
    # Load terrain data (downloads if needed)
    terrain = load_terrain()

    # Build Dataset with derived layers
    print("Building Dataset with terrain analysis layers...")
    ds = xr.Dataset({
        'elevation': terrain.rename(None),
        'slope': slope(terrain),
        'aspect': aspect(terrain),
        'quantile': quantile(terrain),
    })
    print(ds)

    # Drape satellite tiles on terrain (reprojected to match DEM CRS)
    print("Loading satellite tiles...")
    ds.rtx.place_tiles('satellite', z='elevation')

    # --- GeoJSON landmarks and outlines --------------------------------
    # Coordinates in WGS84 (lon, lat); pyproj reprojects to the DEM CRS.
    crater_lake_geojson = {
        "type": "FeatureCollection",
        "features": [
            # Points of interest
            {
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [-122.163, 42.947]},
                "properties": {"name": "Wizard Island"},
            },
            {
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [-122.069, 42.922]},
                "properties": {"name": "Phantom Ship"},
            },
            {
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [-122.142, 42.912]},
                "properties": {"name": "Rim Village"},
            },
            {
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [-122.082, 42.972]},
                "properties": {"name": "Cleetwood Cove"},
            },
            {
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [-122.016, 42.920]},
                "properties": {"name": "Mount Scott"},
            },
            # Simplified lake rim outline
            {
                "type": "Feature",
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[
                        [-122.110, 42.976],
                        [-122.073, 42.972],
                        [-122.048, 42.957],
                        [-122.040, 42.940],
                        [-122.050, 42.920],
                        [-122.064, 42.908],
                        [-122.090, 42.903],
                        [-122.120, 42.905],
                        [-122.148, 42.912],
                        [-122.168, 42.925],
                        [-122.172, 42.945],
                        [-122.162, 42.960],
                        [-122.140, 42.972],
                        [-122.110, 42.976],
                    ]],
                },
                "properties": {"name": "Crater Lake"},
            },
            # Rim Drive (simplified path)
            {
                "type": "Feature",
                "geometry": {
                    "type": "LineString",
                    "coordinates": [
                        [-122.142, 42.912],
                        [-122.160, 42.918],
                        [-122.175, 42.932],
                        [-122.178, 42.948],
                        [-122.168, 42.962],
                        [-122.148, 42.976],
                        [-122.118, 42.982],
                        [-122.088, 42.978],
                        [-122.065, 42.968],
                        [-122.045, 42.952],
                        [-122.035, 42.935],
                        [-122.040, 42.918],
                        [-122.055, 42.905],
                        [-122.080, 42.897],
                        [-122.110, 42.897],
                        [-122.135, 42.903],
                        [-122.142, 42.912],
                    ],
                },
                "properties": {"name": "Rim Drive"},
            },
        ],
    }

    ZARR = CACHE / "crater_lake_national_park.zarr"

    # Ensure meshes exist in zarr (first run places + saves them)
    try:
        import zarr as _zarr
        _zarr.open(str(ZARR), mode='r', use_consolidated=False)['meshes']
    except (KeyError, FileNotFoundError):
        # First run — place features from sources and bake into zarr
        # --- GeoJSON landmarks and outlines ----------------------------------
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="place_geojson called before")
            info = ds.rtx.place_geojson(
                crater_lake_geojson,
                z='elevation',
                height=15.0,
                label_field='name',
                geometry_id='landmark',
            )
        print(f"Placed {info['geometries']} GeoJSON geometries: "
              f"{', '.join(info['geometry_ids'])}")

        # --- Overture Maps buildings -----------------------------------------
        try:
            bldgs_data = fetch_buildings(bounds=BOUNDS, source='overture', crs=CRS,
                                         cache_path=CACHE / "crater_lake_buildings_overture.geojson")
            if bldgs_data['features']:
                info = ds.rtx.place_buildings(bldgs_data, z='elevation')
                print(f"Placed {info['geometries']} Overture buildings")
        except ImportError:
            print("Skipping Overture buildings (pip install duckdb)")
        except Exception as e:
            print(f"Skipping Overture buildings: {e}")

        # --- Overture Maps roads --------------------------------------------
        try:
            roads_data = fetch_roads(bounds=BOUNDS, road_type='all',
                                     source='overture', crs=CRS,
                                     cache_path=CACHE / "crater_lake_roads_overture.geojson")
            if roads_data['features']:
                info = ds.rtx.place_roads(roads_data, z='elevation',
                                          geometry_id='roads', height=5)
                print(f"Placed {info['geometries']} Overture road geometries")
        except ImportError:
            print("Skipping Overture roads (pip install duckdb)")
        except Exception as e:
            print(f"Skipping Overture roads: {e}")

        # --- Water features (Overture Maps) ---------------------------------
        try:
            water_data = fetch_water(bounds=BOUNDS, water_type='all', crs=CRS,
                                     cache_path=CACHE / "crater_lake_water.geojson")
            results = ds.rtx.place_water(water_data, z='elevation')
            for cat, info in results.items():
                print(f"Placed {info['geometries']} {cat} water features")
        except Exception as e:
            print(f"Skipping water: {e}")

        # Save all placed meshes into the DEM zarr
        ds.rtx.save_meshes(ZARR, z='elevation')

    # --- Wind data --------------------------------------------------------
    wind = None
    try:
        from rtxpy import fetch_wind
        wind = fetch_wind(BOUNDS, grid_size=15)
        # Crater Lake is smaller — more particles, faster, shorter lives
        # so they cover the field instead of clumping
        wind['n_particles'] = 15000
        wind['max_age'] = 120
        wind['speed_mult'] = 400.0
    except Exception as e:
        print(f"Skipping wind: {e}")

    # --- Weather data (clouds + rain via Shift+N) -------------------------
    weather = None
    try:
        from rtxpy import fetch_weather
        weather = fetch_weather(BOUNDS, grid_size=15)
    except Exception as e:
        print(f"Skipping weather: {e}")

    # --- Hydro flow data (off by default, Shift+Y to toggle) ---------------
    hydro = None
    try:
        from xrspatial import fill as _fill
        from xrspatial import flow_direction as _flow_direction
        from xrspatial import flow_accumulation as _flow_accumulation
        from xrspatial import stream_order as _stream_order
        from xrspatial import stream_link as _stream_link
        from scipy.ndimage import uniform_filter as _uniform_filter

        print("Conditioning DEM for hydrological flow...")
        _data = terrain.data
        _is_cupy = hasattr(_data, 'get')
        _elev = _data.get() if _is_cupy else np.array(_data)
        _elev = _elev.astype(np.float32)
        _ocean = (_elev == 0.0) | np.isnan(_elev)
        _elev[_ocean] = -100.0

        _smoothed = _uniform_filter(_elev, size=15, mode='nearest')
        _smoothed[_ocean] = -100.0

        if _is_cupy:
            import cupy as _cp
            _sm = _cp.asarray(_smoothed)
        else:
            _sm = _smoothed
        _filled = _fill(terrain.copy(data=_sm))
        _fd = _filled.data - _sm
        _resolved = _filled.data + _fd * 0.01
        if _is_cupy:
            _cp.random.seed(0)
            _resolved += _cp.random.uniform(0, 0.001, _resolved.shape,
                                            dtype=_cp.float32)
            _resolved[_cp.asarray(_ocean)] = -100.0
        else:
            np.random.seed(0)
            _resolved += np.random.uniform(
                0, 0.001, _resolved.shape).astype(np.float32)
            _resolved[_ocean] = -100.0

        fd = _flow_direction(terrain.copy(data=_resolved))
        fa = _flow_accumulation(fd)
        so = _stream_order(fd, fa, threshold=50)
        sl = _stream_link(fd, fa, threshold=50)

        fd_out, fa_out, so_out = fd.data, fa.data, so.data
        if _is_cupy:
            _ocean_gpu = _cp.asarray(_ocean)
            fd_out[_ocean_gpu] = _cp.nan
            fa_out[_ocean_gpu] = _cp.nan
            so_out[_ocean_gpu] = _cp.nan
        else:
            fd_out[_ocean] = np.nan
            fa_out[_ocean] = np.nan
            so_out[_ocean] = np.nan

        _sl_out = sl.data
        if _is_cupy:
            _sl_out[_ocean_gpu] = _cp.nan
        else:
            _sl_out[_ocean] = np.nan
        _sl_np = _sl_out.get() if _is_cupy else np.asarray(_sl_out)
        _sl_clean = np.nan_to_num(_sl_np, nan=0.0).astype(np.float32)
        if _is_cupy:
            _sl_clean = _cp.asarray(_sl_clean)
        ds['stream_link'] = terrain.copy(data=_sl_clean).rename(None)

        hydro = {
            'flow_dir': fd_out,
            'flow_accum': fa_out,
            'stream_order': so_out,
            'stream_link': _sl_out,
            'accum_threshold': 50,
            'enabled': False,
        }
        print(f"  Flow direction + accumulation computed on "
              f"{terrain.shape[0]}x{terrain.shape[1]} grid")
    except Exception as e:
        print(f"Skipping hydro: {e}")

    print("\nLaunching explore (press G to cycle layers, "
          "Shift+W for wind, Shift+N for clouds, Shift+Y for hydro)...\n")
    ds.rtx.explore(
        z='elevation',
        scene_zarr=ZARR,
        mesh_type='voxel',
        width=1024,
        height=768,
        render_scale=0.5,
        wind_data=wind,
        weather_data=weather,
        hydro_data=hydro,
        fog_density=3.0,
        repl=True,
    )

    print("Done")
