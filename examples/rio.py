"""Interactive playground for Rio de Janeiro, Brazil.

Explore the terrain of Rio de Janeiro using GPU-accelerated ray tracing.
Elevation data is sourced from the Copernicus GLO-30 DEM (30 m).

Builds an xr.Dataset with elevation, slope, aspect, and quantile layers.
Press G to cycle between layers. Satellite tiles are draped on the terrain
automatically — press U to toggle tile overlay on/off.

Requirements:
    pip install rtxpy[all] matplotlib xarray rioxarray requests pyproj Pillow
"""

import warnings

import numpy as np
import xarray as xr

from xrspatial import slope, aspect, quantile
from pathlib import Path

from rtxpy import fetch_dem, fetch_buildings, fetch_roads, fetch_water
import rtxpy

# Rio de Janeiro bounding box (WGS84)
# Covers the city from Barra da Tijuca in the west to Ilha do Governador
# in the east, including Sugarloaf, Corcovado, and Tijuca Forest.
BOUNDS = (-43.42, -23.08, -43.10, -22.84)
CRS = 'EPSG:32723'  # UTM zone 23S
CACHE = Path(__file__).parent


def load_terrain():
    """Load Rio de Janeiro terrain data, downloading if necessary."""
    terrain = fetch_dem(
        bounds=BOUNDS,
        output_path=CACHE / "rio_dem.tif",
        source='copernicus',
        crs=CRS,
    )

    # Scale down elevation for visualization (optional)
    terrain.data = terrain.data * 0.025

    # Ensure contiguous array before GPU transfer
    terrain.data = np.ascontiguousarray(terrain.data)

    # Get stats before GPU transfer
    elev_min = float(np.nanmin(terrain.data))
    elev_max = float(np.nanmax(terrain.data))

    # Convert to cupy for GPU processing using the accessor
    terrain = terrain.rtx.to_cupy()

    print(f"Terrain loaded: {terrain.shape}, elevation range: "
          f"{elev_min:.0f}m to {elev_max:.0f}m (scaled)")

    return terrain


if __name__ == "__main__":
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

    # --- Microsoft Global Building Footprints --------------------------------
    try:
        bldg_data = fetch_buildings(bounds=BOUNDS, cache_path=CACHE / "rio_buildings.geojson")
        info = ds.rtx.place_buildings(bldg_data, z='elevation', elev_scale=0.025,
                                      mesh_cache=CACHE / "rio_buildings_mesh.npz")
        print(f"Placed {info['geometries']} building geometries")
    except Exception as e:
        print(f"Skipping buildings: {e}")

    # --- OpenStreetMap roads ------------------------------------------------
    try:
        for rt, gid, clr in [('major', 'road_major', (0.10, 0.10, 0.10)),
                              ('minor', 'road_minor', (0.55, 0.55, 0.55))]:
            data = fetch_roads(bounds=BOUNDS, road_type=rt,
                               cache_path=CACHE / f"rio_roads_{rt}.geojson")
            info = ds.rtx.place_roads(data, z='elevation', geometry_id=gid, color=clr,
                                      mesh_cache=CACHE / f"rio_roads_{rt}_mesh.npz")
            print(f"Placed {info['geometries']} {rt} road geometries")
    except Exception as e:
        print(f"Skipping roads: {e}")

    # --- OpenStreetMap water features ---------------------------------------
    try:
        water_data = fetch_water(bounds=BOUNDS, water_type='all',
                                 cache_path=CACHE / "rio_water.geojson")
        results = ds.rtx.place_water(water_data, z='elevation',
                                     mesh_cache_prefix=CACHE / "rio_water")
        for cat, info in results.items():
            print(f"Placed {info['geometries']} {cat} water features")
    except Exception as e:
        print(f"Skipping water: {e}")

    # --- Wind data --------------------------------------------------------
    wind = None
    try:
        from rtxpy import fetch_wind
        wind = fetch_wind(BOUNDS, grid_size=15)
    except Exception as e:
        print(f"Skipping wind: {e}")

    print("\nLaunching explore (press G to cycle layers, Shift+W for wind)...\n")
    ds.rtx.explore(
        z='elevation',
        width=2048,
        height=1600,
        render_scale=0.5,
        color_stretch='cbrt',
        subsample=4,
        wind_data=wind,
    )

    print("Done")
