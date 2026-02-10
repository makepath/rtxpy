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
    pip install rtxpy[all] matplotlib xarray rioxarray requests pyproj Pillow osmnx
"""

import warnings

import numpy as np
import xarray as xr
from pathlib import Path

from xrspatial import slope, aspect, quantile

# Import rtxpy to register the .rtx accessor
from rtxpy import fetch_dem, fetch_roads, fetch_water
import rtxpy
from _utils import print_controls, classify_water_features


def load_terrain():
    """Load Crater Lake terrain data, downloading if necessary."""
    dem_path = Path(__file__).parent / "crater_lake_national_park.tif"

    terrain = fetch_dem(
        bounds=(-122.3, 42.8, -121.9, 43.0),
        output_path=dem_path,
        source='srtm',
        crs='EPSG:5070',
    )

    # Subsample for faster interactive performance
    terrain = terrain[::2, ::2]

    # Scale down elevation for visualization (optional)
    terrain.data = terrain.data * 0.1

    # Ensure contiguous array before GPU transfer
    terrain.data = np.ascontiguousarray(terrain.data)

    # Get stats before GPU transfer
    elev_min = float(terrain.min())
    elev_max = float(terrain.max())

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

    # Place GeoJSON on the elevation layer's RTX scene.
    # Pixel spacing is 1.0 (pixel-coord mode) which matches explore()'s
    # default, so the warning is expected — suppress it.
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

    # --- OpenStreetMap roads ------------------------------------------------
    try:
        roads_cache = Path(__file__).parent / "crater_lake_roads.geojson"
        roads_data = fetch_roads(
            bounds=(-122.3, 42.8, -121.9, 43.0),
            road_type='all',
            crs='EPSG:5070',
            cache_path=roads_cache,
        )
        roads_mesh = Path(__file__).parent / "crater_lake_roads_mesh.npz"
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="place_geojson called before")
            roads_info = ds.rtx.place_geojson(
                roads_data,
                z='elevation',
                height=5.0,
                label_field='name',
                geometry_id='roads',
                merge=True,
                mesh_cache=roads_mesh,
            )
        print(f"Placed {roads_info['geometries']} road geometries")
    except ImportError as e:
        print(f"Skipping roads: {e}")

    # --- OpenStreetMap water features ---------------------------------------
    # Split into major (rivers, canals → wider tubes) and minor (streams,
    # drains, ditches → thinner tubes), each in a distinct blue tone.
    try:
        water_cache = Path(__file__).parent / "crater_lake_water.geojson"
        water_data = fetch_water(
            bounds=(-122.3, 42.8, -121.9, 43.0),
            water_type='all',
            crs='EPSG:5070',
            cache_path=water_cache,
        )

        major_features, minor_features, body_features = classify_water_features(water_data)

        if major_features:
            major_fc = {"type": "FeatureCollection", "features": major_features}
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="place_geojson called before")
                major_info = ds.rtx.place_geojson(
                    major_fc, z='elevation', height=10.0,
                    label_field='name', geometry_id='water_major',
                    color=(0.40, 0.70, 0.95, 2.25),
                    merge=True,
                    mesh_cache=Path(__file__).parent / "crater_lake_water_major_mesh.npz",
                )
            print(f"Placed {major_info['geometries']} major water features (rivers, canals)")

        if minor_features:
            minor_fc = {"type": "FeatureCollection", "features": minor_features}
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="place_geojson called before")
                minor_info = ds.rtx.place_geojson(
                    minor_fc, z='elevation', height=4.0,
                    label_field='name', geometry_id='water_minor',
                    color=(0.50, 0.75, 0.98, 2.25),
                    merge=True,
                    mesh_cache=Path(__file__).parent / "crater_lake_water_minor_mesh.npz",
                )
            print(f"Placed {minor_info['geometries']} minor water features (streams, drains)")

        if body_features:
            body_fc = {"type": "FeatureCollection", "features": body_features}
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="place_geojson called before")
                body_info = ds.rtx.place_geojson(
                    body_fc, z='elevation', height=6.0,
                    label_field='name', geometry_id='water_body',
                    color=(0.35, 0.55, 0.88, 2.25),
                    merge=True,
                    mesh_cache=Path(__file__).parent / "crater_lake_water_body_mesh.npz",
                )
            print(f"Placed {body_info['geometries']} water bodies (lakes, ponds)")

    except ImportError as e:
        print(f"Skipping water features: {e}")

    # --- Wind data --------------------------------------------------------
    wind = None
    try:
        from rtxpy import fetch_wind
        wind = fetch_wind((-122.3, 42.8, -121.9, 43.0), grid_size=15)
        # Crater Lake is smaller — more particles, faster, shorter lives
        # so they cover the field instead of clumping
        wind['n_particles'] = 15000
        wind['max_age'] = 120
        wind['speed_mult'] = 80.0
    except Exception as e:
        print(f"Skipping wind: {e}")

    print("\nLaunching explore (press G to cycle layers, Shift+W for wind)...\n")
    ds.rtx.explore(
        z='elevation',
        mesh_type='voxel',
        width=1024,
        height=768,
        render_scale=0.5,
        wind_data=wind,
    )

    print("Done")
