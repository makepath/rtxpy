"""
NYC Real LiDAR Point Cloud Demo
================================

Downloads real USGS 3DEP LiDAR LAZ tiles from the TNM API for
Manhattan and visualizes them over the NYC DEM using OptiX sphere
primitives with ASPRS classification coloring.

First run downloads ~2.7 GB of LAZ data; subsequent runs use cache.
LAZ decompression is parallelized across CPU cores.

LiDAR point clouds and buildings are cached in the zarr store after
the first run. Subsequent runs skip all LAZ decompression, filtering,
thinning, and coordinate transforms — loading directly from zarr.

Usage:
    python examples/nyc_lidar.py
"""

from pathlib import Path

import numpy as np
import xarray as xr
import zarr

import rtxpy


def _has_lidar_cache(zarr_path):
    """Check if the zarr store has cached LiDAR sphere geometries."""
    try:
        store = zarr.open(str(zarr_path), mode='r', use_consolidated=False)
        if 'meshes' not in store:
            return False
        mg = store['meshes']
        for gid in mg:
            if mg[gid].attrs.get('type', '') == 'sphere':
                return True
    except Exception:
        pass
    return False


def main():
    # Resolve paths relative to this script's directory
    _here = Path(__file__).resolve().parent

    # Full NYC DEM extent in WGS84
    dem_bounds = (-74.26, 40.49, -73.70, 40.92)
    # Manhattan LiDAR subset
    lidar_bounds = (-74.02, 40.70, -73.97, 40.88)
    cache_dir = _here / 'cache' / 'lidar'
    zarr_path = str(_here / 'nyc_dem.zarr')

    # Load NYC DEM (write CRS back — rioxarray doesn't auto-detect from zarr)
    ds = xr.open_zarr(zarr_path)
    terrain = ds['elevation'].load().astype(np.float32)
    terrain = terrain.rio.write_crs('EPSG:32618')
    terrain = terrain.rtx.to_cupy()

    # Build terrain mesh
    terrain.rtx.triangulate()

    # Satellite tiles
    terrain.rtx.place_tiles('satellite')

    # Check for cached lidar + buildings in zarr
    if _has_lidar_cache(zarr_path):
        print("Loading cached meshes from zarr...")
        terrain.rtx.load_meshes(zarr_path)
    else:
        # Use cached LAZ files if available, otherwise download
        cached = sorted(cache_dir.glob('*.laz')) if cache_dir.exists() else []
        if cached:
            laz_paths = cached
            print(f"Using {len(laz_paths)} cached LAZ file(s)")
        else:
            laz_paths = rtxpy.fetch_lidar(lidar_bounds, cache_dir=str(cache_dir))
        print(f"Got {len(laz_paths)} LAZ file(s)")

        # Buildings from Overture Maps (full DEM extent)
        buildings = rtxpy.fetch_buildings(
            bounds=dem_bounds, source='overture',
            cache_path=str(_here / 'cache' / 'nyc_lidar_buildings.geojson'))
        terrain.rtx.place_buildings(buildings)

        # Place all LAZ tiles in parallel (threaded LAZ decompression)
        # ASPRS classes: 1=unclassified, 2=ground, 10=rail, 17=bridge
        # Exclude: 7=low noise, 9=water, 18=high noise
        # thin=0.5 removes flight-line overlap striations (1 point per 0.5m cell)
        terrain.rtx.place_pointclouds(
            laz_paths,
            point_size=0.25,
            color='classification',
            classification=[1, 2, 10, 17],
            thin=0.5,
        )

        # Save all meshes (buildings + lidar) to zarr for next run
        terrain.rtx.save_meshes(zarr_path)

    # Fetch weather for cloud + rain overlay (Shift+N)
    weather = None
    try:
        from rtxpy import fetch_weather
        weather = fetch_weather(dem_bounds, grid_size=15)
    except Exception as e:
        print(f"Skipping weather: {e}")

    # Launch interactive viewer
    terrain.rtx.explore(width=1600, height=1200, weather_data=weather)


if __name__ == '__main__':
    main()
