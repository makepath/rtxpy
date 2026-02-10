"""Interactive playground for Rio de Janeiro, Brazil.

Explore the terrain of Rio de Janeiro using GPU-accelerated ray tracing.
Elevation data is sourced from the Copernicus GLO-30 DEM (30 m).

Builds an xr.Dataset with elevation, slope, aspect, and quantile layers.
Press G to cycle between layers. Satellite tiles are draped on the terrain
automatically — press U to toggle tile overlay on/off.

Requirements:
    pip install rtxpy[all] matplotlib xarray rioxarray requests pyproj Pillow
"""

import numpy as np
import xarray as xr

from xrspatial import slope, aspect, quantile
from pathlib import Path

import warnings

from rtxpy import fetch_dem, fetch_buildings, fetch_roads, fetch_water
import rtxpy

# Water feature classification
_MAJOR_WATER = {'river', 'canal'}
_MINOR_WATER = {'stream', 'drain', 'ditch'}

# Rio de Janeiro bounding box (WGS84)
# Covers the city from Barra da Tijuca in the west to Ilha do Governador
# in the east, including Sugarloaf, Corcovado, and Tijuca Forest.
BOUNDS = (-43.42, -23.08, -43.10, -22.84)
CRS = 'EPSG:32723'  # UTM zone 23S


def load_terrain():
    """Load Rio de Janeiro terrain data, downloading if necessary."""
    dem_path = Path(__file__).parent / "rio_dem.tif"

    terrain = fetch_dem(
        bounds=BOUNDS,
        output_path=dem_path,
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
    # Load terrain data (downloads if needed)
    terrain = load_terrain()

    print("\nControls:")
    print("  W/S/A/D or Arrow keys: Move camera")
    print("  Q/E or Page Up/Down: Move up/down")
    print("  I/J/K/L: Look around")
    print("  +/-: Adjust movement speed")
    print("  G: Cycle overlay layers")
    print("  O: Place observer (for viewshed)")
    print("  V: Toggle viewshed (teal glow)")
    print("  [/]: Adjust observer height")
    print("  T: Toggle shadows")
    print("  C: Cycle colormap")
    print("  U: Toggle tile overlay")
    print("  F: Screenshot")
    print("  H: Toggle help overlay")
    print("  X: Exit\n")

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
        bldg_cache = Path(__file__).parent / "rio_buildings.geojson"
        bldg_data = fetch_buildings(
            bounds=BOUNDS,
            cache_path=bldg_cache,
        )

        # Scale building heights to match the 0.025× terrain elevation.
        elev_scale = 0.025
        default_height_m = 8.0
        for feat in bldg_data.get("features", []):
            props = feat.get("properties", {})
            h = props.get("height", -1)
            if not isinstance(h, (int, float)) or h <= 0:
                h = default_height_m
            props["height"] = h * elev_scale

        mesh_cache_path = Path(__file__).parent / "rio_buildings_mesh.npz"
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
        major_cache = Path(__file__).parent / "rio_roads_major.geojson"
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
                    mesh_cache=Path(__file__).parent / "rio_roads_major_mesh.npz",
                )
            print(f"Placed {info['geometries']} major road geometries")

        # Minor roads: tertiary, residential, service
        minor_cache = Path(__file__).parent / "rio_roads_minor.geojson"
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
                    mesh_cache=Path(__file__).parent / "rio_roads_minor_mesh.npz",
                )
            print(f"Placed {info['geometries']} minor road geometries")

    except ImportError as e:
        print(f"Skipping roads: {e}")

    # --- OpenStreetMap water features ---------------------------------------
    try:
        water_cache = Path(__file__).parent / "rio_water.geojson"
        water_data = fetch_water(
            bounds=BOUNDS,
            water_type='all',
            cache_path=water_cache,
        )

        major_features = []
        minor_features = []
        body_features = []
        for f in water_data.get('features', []):
            ww = (f.get('properties') or {}).get('waterway', '')
            nat = (f.get('properties') or {}).get('natural', '')
            if ww in _MAJOR_WATER:
                major_features.append(f)
            elif ww in _MINOR_WATER:
                minor_features.append(f)
            elif nat == 'water':
                body_features.append(f)
            else:
                minor_features.append(f)

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
                    mesh_cache=Path(__file__).parent / "rio_water_major_mesh.npz",
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
                    mesh_cache=Path(__file__).parent / "rio_water_minor_mesh.npz",
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
                    mesh_cache=Path(__file__).parent / "rio_water_body_mesh.npz",
                )
            print(f"Placed {body_info['geometries']} water bodies (lakes, ponds)")

    except ImportError as e:
        print(f"Skipping water features: {e}")

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
