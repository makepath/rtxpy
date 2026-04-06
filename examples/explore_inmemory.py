"""In-memory terrain exploration — the classic DataArray path.

Demonstrates the simplest way to use rtxpy: load a DEM as an xarray
DataArray and call ``dem.rtx.explore()``.  The terrain is held in GPU
memory as a single raster; the LOD system tiles it internally using
an InMemoryChunkSource (automatic, no zarr required).

This is the right path when:
- Your DEM fits comfortably in GPU memory (up to ~8K x 8K)
- You're working with a DataArray from rioxarray, xarray-spatial, etc.
- You want to place geometry interactively with place_buildings(), etc.

Usage:
    python explore_inmemory.py
    python explore_inmemory.py --dem copernicus --subsample 2
    python explore_inmemory.py --buildings --roads --water --wind --hydro

Requirements:
    pip install rtxpy[all]
"""

import argparse
import warnings
from pathlib import Path

import numpy as np
import xarray as xr  # noqa
import xrspatial  # noqa

import rtxpy  # noqa
from rtxpy import fetch_buildings, fetch_roads, fetch_water

# Birmingham, Alabama — rolling Appalachian foothills
BOUNDS = (-86.9, 33.4, -86.7, 33.6)
CACHE = Path(__file__).parent


def _auto_utm(bounds):
    """Pick the correct UTM zone for a lon/lat bounding box."""
    west, south, east, north = bounds
    lon = (west + east) / 2
    lat = (south + north) / 2
    zone = int((lon + 180) / 6) + 1
    epsg = 32600 + zone if lat >= 0 else 32700 + zone
    return f'EPSG:{epsg}'


def main():
    parser = argparse.ArgumentParser(
        description="Explore terrain from an in-memory DataArray.")
    parser.add_argument("--dem", type=str, default="copernicus",
                        choices=["copernicus", "srtm", "usgs_10m"],
                        help="DEM source (default: copernicus)")
    parser.add_argument("--subsample", type=int, default=1,
                        help="Subsample factor (default: 1)")
    parser.add_argument("--bounds", type=float, nargs=4,
                        metavar=("W", "S", "E", "N"),
                        default=list(BOUNDS),
                        help="Bounding box in lon/lat degrees")
    parser.add_argument("--buildings", action="store_true",
                        help="Place Overture buildings on terrain")
    parser.add_argument("--roads", action="store_true",
                        help="Place Overture roads on terrain")
    parser.add_argument("--water", action="store_true",
                        help="Place water features on terrain")
    parser.add_argument("--wind", action="store_true",
                        help="Enable wind particle animation (Shift+W)")
    parser.add_argument("--hydro", action="store_true",
                        help="Enable hydro flow particles (Shift+Y)")
    args = parser.parse_args()

    bounds = tuple(args.bounds)
    crs = _auto_utm(bounds)

    # ---- Load DEM into memory as a DataArray ----------------------------
    print(f"Fetching {args.dem} DEM for {bounds} (CRS: {crs})...")
    dem = rtxpy.fetch_dem(
        bounds=bounds,
        output_path=f"explore_inmemory_{args.dem}.zarr",
        source=args.dem,
        crs=crs,
    )

    if args.subsample > 1:
        dem = dem[::args.subsample, ::args.subsample]

    # Ensure contiguous numpy then push to GPU
    dem.data = np.ascontiguousarray(dem.data)
    dem = dem.rtx.to_cupy()

    import cupy as cp
    emin = float(cp.nanmin(dem.data))
    emax = float(cp.nanmax(dem.data))
    print(f"Terrain: {dem.shape}, elevation {emin:.0f}–{emax:.0f} m")

    # ---- Place geometry (buildings, roads, water) -----------------------
    if args.buildings:
        try:
            bldgs = fetch_buildings(
                bounds=bounds, source='overture', crs=crs,
                cache_path=CACHE / "inmemory_buildings.geojson")
            if bldgs['features']:
                with warnings.catch_warnings():
                    warnings.filterwarnings(
                        "ignore", message="place_geojson called before")
                    info = dem.rtx.place_buildings(bldgs)
                print(f"Placed {info['geometries']} buildings")
            else:
                print("No buildings found in bounds")
        except Exception as e:
            print(f"Skipping buildings: {e}")

    if args.roads:
        try:
            roads = fetch_roads(
                bounds=bounds, road_type='all', source='overture', crs=crs,
                cache_path=CACHE / "inmemory_roads.geojson")
            if roads['features']:
                with warnings.catch_warnings():
                    warnings.filterwarnings(
                        "ignore", message="place_geojson called before")
                    info = dem.rtx.place_roads(roads, geometry_id='roads',
                                               height=5)
                print(f"Placed {info['geometries']} road geometries")
            else:
                print("No roads found in bounds")
        except Exception as e:
            print(f"Skipping roads: {e}")

    if args.water:
        try:
            water_data = fetch_water(
                bounds=bounds, water_type='all', crs=crs,
                cache_path=CACHE / "inmemory_water.geojson")
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore", message="place_geojson called before")
                results = dem.rtx.place_water(water_data)
            for cat, info in results.items():
                print(f"Placed {info['geometries']} {cat} water features")
        except Exception as e:
            print(f"Skipping water: {e}")

    # ---- Wind data (Open-Meteo) -----------------------------------------
    wind = None
    if args.wind:
        try:
            from rtxpy import fetch_wind
            wind = fetch_wind(bounds, grid_size=15)
            wind['n_particles'] = 15000
            wind['max_age'] = 120
            wind['speed_mult'] = 400.0
            print("Wind data loaded")
        except Exception as e:
            print(f"Skipping wind: {e}")

    # ---- Hydro flow (lazy GPU computation on first Shift+Y) -------------
    hydro = None
    if args.hydro:
        hydro = {'enabled': False}
        print("Hydro enabled (press Shift+Y to activate)")

    # ---- Explore --------------------------------------------------------
    # This is the classic path: a single DataArray in GPU memory.
    # Internally, the LOD system wraps it in an InMemoryChunkSource and
    # tiles it automatically.  No zarr, no chunk source, no streaming —
    # just a raster and a viewer.
    controls = []
    if args.wind:
        controls.append("Shift+W for wind")
    if args.hydro:
        controls.append("Shift+Y for hydro")
    if controls:
        print(f"\nControls: {', '.join(controls)}")

    dem.rtx.explore(
        width=1600,
        height=1200,
        render_scale=0.5,
        color_stretch='cbrt',
        wind_data=wind,
        hydro_data=hydro,
        repl=True,
    )

    print("Done")


if __name__ == "__main__":
    main()
