"""Interactive playground for Los Angeles, California.

Explore the terrain of Los Angeles using GPU-accelerated ray tracing.
The area covers downtown LA, Echo Park, Silver Lake, Elysian Park,
Griffith Park, the Hollywood Hills, and the Hollywood Sign.

Elevation data is sourced from USGS 3DEP 1-meter lidar DEM.

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

from rtxpy import fetch_dem, fetch_buildings, fetch_roads, fetch_water, fetch_firms
import rtxpy
from _utils import print_controls, classify_water_features, scale_building_heights

# Los Angeles bounding box (WGS84)
# Focused area covering DTLA, Echo Park, Silver Lake, Griffith Park,
# Hollywood Hills, and the Hollywood Sign (~8 km × 8 km at 1 m resolution).
BOUNDS = (-118.32, 34.04, -118.23, 34.12)
CRS = 'EPSG:32611'  # UTM zone 11N


def load_terrain():
    """Load Los Angeles terrain data, downloading if necessary."""
    dem_path = Path(__file__).parent / "los_angeles_dem.tif"

    terrain = fetch_dem(
        bounds=BOUNDS,
        output_path=dem_path,
        source='usgs_1m',
        crs=CRS,
    )

    # Scale elevation for visualization (1 m pixels need less reduction
    # than 30 m Copernicus data to keep a similar visual slope ratio)
    terrain.data = terrain.data * 0.5

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
    # Load terrain data (downloads if needed)
    terrain = load_terrain()

    print_controls()

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
    ds.rtx.place_tiles('satellite', z='elevation', zoom=15)

    # --- Microsoft Global Building Footprints --------------------------------
    try:
        bldg_cache = Path(__file__).parent / "los_angeles_buildings.geojson"
        bldg_data = fetch_buildings(
            bounds=BOUNDS,
            cache_path=bldg_cache,
        )

        elev_scale = 0.5
        default_height_m = 8.0
        scale_building_heights(bldg_data, elev_scale, default_height_m)

        mesh_cache_path = Path(__file__).parent / "los_angeles_buildings_mesh.npz"
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="place_geojson called before")
            bldg_info = ds.rtx.place_geojson(
                bldg_data,
                z='elevation',
                height=default_height_m * elev_scale,
                height_field='height',
                geometry_id='building',
                densify=False,
                merge=True,
                extrude=True,
                mesh_cache=mesh_cache_path,
            )
        print(f"Placed {bldg_info['geometries']} building footprint geometries")
    except ImportError as e:
        print(f"Skipping buildings: {e}")

    # --- OpenStreetMap roads ------------------------------------------------
    try:
        # Major roads: motorways, trunk, primary, secondary
        major_cache = Path(__file__).parent / "los_angeles_roads_major.geojson"
        major_roads = fetch_roads(
            bounds=BOUNDS,
            road_type='major',
            cache_path=major_cache,
        )
        if major_roads.get('features'):
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="place_geojson called before")
                info = ds.rtx.place_geojson(
                    major_roads, z='elevation', height=1,
                    label_field='name', geometry_id='road_major',
                    color=(0.10, 0.10, 0.10),
                    densify=False,
                    merge=True,
                    mesh_cache=Path(__file__).parent / "los_angeles_roads_major_mesh.npz",
                )
            print(f"Placed {info['geometries']} major road geometries")

        # Minor roads: tertiary, residential, service
        minor_cache = Path(__file__).parent / "los_angeles_roads_minor.geojson"
        minor_roads = fetch_roads(
            bounds=BOUNDS,
            road_type='minor',
            cache_path=minor_cache,
        )
        if minor_roads.get('features'):
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="place_geojson called before")
                info = ds.rtx.place_geojson(
                    minor_roads, z='elevation', height=1,
                    label_field='name', geometry_id='road_minor',
                    color=(0.55, 0.55, 0.55),
                    densify=False,
                    merge=True,
                    mesh_cache=Path(__file__).parent / "los_angeles_roads_minor_mesh.npz",
                )
            print(f"Placed {info['geometries']} minor road geometries")

    except ImportError as e:
        print(f"Skipping roads: {e}")

    # --- OpenStreetMap water features ---------------------------------------
    try:
        water_cache = Path(__file__).parent / "los_angeles_water.geojson"
        water_data = fetch_water(
            bounds=BOUNDS,
            water_type='all',
            cache_path=water_cache,
        )

        major_features, minor_features, body_features = classify_water_features(water_data)

        if major_features:
            major_fc = {"type": "FeatureCollection", "features": major_features}
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="place_geojson called before")
                major_info = ds.rtx.place_geojson(
                    major_fc, z='elevation', height=0,
                    label_field='name', geometry_id='water_major',
                    color=(0.40, 0.70, 0.95, 2.25),
                    densify=False,
                    merge=True,
                    mesh_cache=Path(__file__).parent / "los_angeles_water_major_mesh.npz",
                )
            print(f"Placed {major_info['geometries']} major water features (rivers, canals)")

        if minor_features:
            minor_fc = {"type": "FeatureCollection", "features": minor_features}
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="place_geojson called before")
                minor_info = ds.rtx.place_geojson(
                    minor_fc, z='elevation', height=0,
                    label_field='name', geometry_id='water_minor',
                    color=(0.50, 0.75, 0.98, 2.25),
                    densify=False,
                    merge=True,
                    mesh_cache=Path(__file__).parent / "los_angeles_water_minor_mesh.npz",
                )
            print(f"Placed {minor_info['geometries']} minor water features (streams, drains)")

        if body_features:
            body_fc = {"type": "FeatureCollection", "features": body_features}
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="place_geojson called before")
                body_info = ds.rtx.place_geojson(
                    body_fc, z='elevation', height=0.5,
                    label_field='name', geometry_id='water_body',
                    color=(0.35, 0.55, 0.88, 2.25),
                    extrude=True,
                    merge=True,
                    mesh_cache=Path(__file__).parent / "los_angeles_water_body_mesh.npz",
                )
            print(f"Placed {body_info['geometries']} water bodies (lakes, reservoirs)")

    except ImportError as e:
        print(f"Skipping water features: {e}")

    # --- NASA FIRMS fire detections (LANDSAT 30 m, last 7 days) -----------
    try:
        fire_cache = Path(__file__).parent / "los_angeles_fires.geojson"
        fire_data = fetch_firms(
            bounds=BOUNDS,
            date_span='7d',
            cache_path=fire_cache,
            crs=CRS,
        )
        if fire_data.get('features'):
            elev_scale = 0.5
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="place_geojson called before")
                fire_info = ds.rtx.place_geojson(
                    fire_data, z='elevation', height=20 * elev_scale,
                    geometry_id='fire',
                    color=(1.0, 0.25, 0.0, 3.0),
                    extrude=True,
                    merge=True,
                )
            print(f"Placed {fire_info['geometries']} fire detection footprints")
        else:
            print("No fire detections in the last 7 days")
    except Exception as e:
        print(f"Skipping fire layer: {e}")

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
