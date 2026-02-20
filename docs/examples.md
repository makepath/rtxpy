# Examples

## Workflow 1: Analysis Arrays for Your Dataset

rtxpy analysis functions return standard xarray DataArrays. Use them just like xarray-spatial — the results slot into your existing pipeline:

```python
import xarray as xr
from rtxpy import fetch_dem
import rtxpy

# Get real data in one call
dem = fetch_dem(
    bounds=(-122.3, 42.8, -121.9, 43.0),
    output_path='crater_lake.zarr',
    source='srtm',
    crs='EPSG:5070',
)
dem = dem.rtx.to_cupy()

# Compute analysis layers — each returns a DataArray
hillshade = dem.rtx.hillshade(shadows=True)
viewshed = dem.rtx.viewshed(x=500, y=300, observer_elev=2)

# Add to your Dataset alongside other tools
from xrspatial import slope, aspect

ds = xr.Dataset({
    'elevation': dem,
    'hillshade': hillshade,
    'slope': slope(dem),
    'aspect': aspect(dem),
    'viewshed': viewshed,
})

# Standard xarray operations work normally
ds['viewshed'].plot()
print(f"Visible terrain: {float((ds['viewshed'] > 0).mean()) * 100:.1f}%")
ds.to_zarr('analysis_results.zarr')
```

## Workflow 2: Build a Dataset, Then Explore in 3D

The most common rtxpy workflow: build a Dataset from multiple analysis sources, fetch vector features from public data, then explore everything interactively:

```python
import xarray as xr
from xrspatial import slope, aspect, quantile
from rtxpy import fetch_dem, fetch_buildings, fetch_roads, fetch_water, fetch_wind
import rtxpy

BOUNDS = (-122.3, 42.8, -121.9, 43.0)
CRS = 'EPSG:5070'

# 1. Get terrain data (cached after first download)
terrain = fetch_dem(bounds=BOUNDS, output_path='crater_lake.zarr',
                    source='srtm', crs=CRS)
terrain = terrain[::2, ::2]  # subsample for interactive speed
terrain = terrain.rtx.to_cupy()

# 2. Build a Dataset with analysis layers
ds = xr.Dataset({
    'elevation': terrain,
    'slope': slope(terrain),
    'aspect': aspect(terrain),
    'quantile': quantile(terrain),
})

# 3. Drape satellite tiles on terrain
ds.rtx.place_tiles('satellite', z='elevation')

# 4. Fetch and place vector features
roads = fetch_roads(bounds=BOUNDS, crs=CRS,
                    cache_path='cache/roads.geojson')
ds.rtx.place_roads(roads, z='elevation', height=5)

water = fetch_water(bounds=BOUNDS, crs=CRS,
                    cache_path='cache/water.geojson')
ds.rtx.place_water(water, z='elevation')

bldgs = fetch_buildings(bounds=BOUNDS, source='overture', crs=CRS,
                        cache_path='cache/buildings.geojson')
ds.rtx.place_buildings(bldgs, z='elevation')

# 5. Save meshes for instant reload next time
ds.rtx.save_meshes('crater_lake.zarr', z='elevation')

# 6. Explore — G cycles layers, N toggles geometry, U for tiles
wind = fetch_wind(BOUNDS, grid_size=15)
ds.rtx.explore(
    z='elevation',
    scene_zarr='crater_lake.zarr',
    mesh_type='voxel',
    width=1024, height=768,
    render_scale=0.5,
    wind_data=wind,
)
```

Inside the viewer:
- **G** cycles through elevation / slope / aspect / quantile on the terrain
- **U** toggles satellite / OSM / topo basemap tiles
- **N** cycles geometry layers (roads, buildings, water, all, none)
- **Shift+W** toggles wind particle animation
- **V** toggles real-time viewshed (press **O** first to place an observer)

This is what `examples/playground.py` does — run it to see the full experience.

---

## Quick Recipes

### Get data and explore immediately

```python
from rtxpy import fetch_dem

dem = fetch_dem((-43.42, -23.08, -43.10, -22.84), 'rio.zarr', source='copernicus')
dem = dem.rtx.to_cupy()
dem.rtx.explore()
```

### Add buildings to a scene

```python
from rtxpy import fetch_dem, fetch_buildings

bounds = (-61.5, 10.6, -61.4, 10.7)
dem = fetch_dem(bounds, 'terrain.zarr', source='copernicus', crs='EPSG:32620')
dem = dem.rtx.to_cupy()

bldgs = fetch_buildings(bounds, source='overture', crs='EPSG:32620',
                        cache_path='cache/buildings.geojson')
dem.rtx.place_buildings(bldgs, elev_scale=0.33)
dem.rtx.explore()
```

### Render a static image

```python
import numpy as np

H, W = dem.shape
elev = dem.values

img = dem.rtx.render(
    camera_position=(W/2, -50, np.max(elev) + 200),
    look_at=(W/2, H/2, np.mean(elev)),
    shadows=True,
    colormap='terrain',
    output_path='render.png',
)
```

### Create a flyover GIF

```python
dem.rtx.flyover(
    'flyover.gif',
    duration=30,
    fps=10,
    colormap='terrain',
    shadows=True,
)
```

### Create a 360-degree panoramic GIF

```python
import numpy as np

H, W = dem.shape
elev = dem.values
peak_y, peak_x = np.unravel_index(np.argmax(elev), elev.shape)

dem.rtx.view(
    x=peak_x, y=peak_y, z=elev[peak_y, peak_x] + 50,
    output_path='panorama.gif',
    duration=10,
    fov=70,
)
```

### Place custom 3D models

```python
from rtxpy import load_mesh

# Load and place a GLB model at specific positions
dem.rtx.place_mesh(
    'models/tower.glb',
    positions=[(100, 200), (300, 400), (500, 100)],
    scale=0.5,
    swap_yz=True,
    rotation_z='random',
)

dem.rtx.explore()
```

### Use callable mesh source and positions

```python
from rtxpy import load_mesh
import numpy as np

def make_tree():
    return load_mesh('tree.glb', scale=1.5, swap_yz=True,
                     center_xy=True, base_at_zero=True)

def find_valleys(terrain):
    """Place trees in low-elevation areas."""
    threshold = np.percentile(terrain, 25)
    ys, xs = np.where(terrain < threshold)
    # Subsample to avoid too many instances
    idx = np.random.choice(len(xs), size=min(500, len(xs)), replace=False)
    return list(zip(xs[idx], ys[idx]))

dem.rtx.place_mesh(make_tree, find_valleys)
dem.rtx.explore()
```

### Use GeoJSON features

```python
# Points become glowing orbs, Lines become terrain-following ribbons,
# Polygons become outlines (or extruded with extrude=True)
dem.rtx.place_geojson(
    'trails.geojson',
    height=5.0,
    label_field='name',
    color=(0.9, 0.5, 0.1),
)

# Extrude building footprints into solid 3D geometry
dem.rtx.place_geojson(
    'buildings.geojson',
    height_field='height',
    extrude=True,
    merge=True,
    mesh_cache='cache/buildings.npz',
)
```

### Use per-feature 3D models with GeoJSON

```python
dem.rtx.place_geojson(
    'landcover.geojson',
    models={
        'forest': 'tree.glb',
        'urban': 'building.glb',
        'farmland': 'crop.glb',
    },
    model_field='landcover',
    fill_spacing=5.0,
    fill_scale=2.0,
)
```

### Stream large zarr datasets

```python
from rtxpy import fetch_dem

# Download a large area
dem = fetch_dem(
    bounds=(-122.5, 37.6, -122.3, 37.8),
    output_path='bay_area.zarr',
    source='usgs_10m',
)
dem = dem.rtx.to_cupy()

# Place and save meshes
from rtxpy import fetch_buildings, fetch_roads
bldgs = fetch_buildings(bounds, source='overture', crs=dem.rio.crs)
dem.rtx.place_buildings(bldgs)
roads = fetch_roads(bounds, source='overture', crs=dem.rio.crs)
dem.rtx.place_roads(roads)
dem.rtx.save_meshes('bay_area.zarr')

# Explore with chunk streaming — only nearby meshes are loaded
dem.rtx.explore(
    scene_zarr='bay_area.zarr',
    subsample=4,
    render_scale=0.5,
)
```

### Export terrain to STL

```python
from rtxpy import write_stl

verts, indices = dem.rtx.triangulate()
write_stl('terrain.stl', verts, indices)
```

### Compute hillshade with custom sun position

```python
shade = dem.rtx.hillshade(
    shadows=True,
    azimuth=90,          # sun from the east
    angle_altitude=15,   # low sun angle for dramatic shadows
)

# Save result
shade.rio.to_raster('hillshade.tif')
```

### Viewshed analysis

```python
# Find visibility from a specific point
vis = dem.rtx.viewshed(x=500, y=300, observer_elev=2)

# vis > 0 = visible (value is viewing angle), -1 = not visible
visible_area = float((vis > 0).sum()) / vis.size * 100
print(f"{visible_area:.1f}% of terrain visible")
```

### Wind particle visualization

```python
from rtxpy import fetch_wind

wind = fetch_wind(bounds=(-43.42, -23.08, -43.10, -22.84), grid_size=20)

# Customize particle behavior
wind['n_particles'] = 15000
wind['max_age'] = 120
wind['speed_mult'] = 400.0

dem.rtx.explore(wind_data=wind)
# Press Shift+W to toggle wind particles in the viewer
```

---

## Example Scripts

The `examples/` directory contains complete scripts:

| Script | Description |
|--------|-------------|
| `playground.py` | Full-featured Crater Lake demo with all data sources |
| `capetown.py` | Cape Town terrain with buildings and roads |
| `guanajuato.py` | Guanajuato, Mexico terrain exploration |
| `los_angeles.py` | Los Angeles terrain |
| `rio.py` | Rio de Janeiro terrain |
| `trinidad.py` | Trinidad & Tobago coastal terrain |
| `bay_area.py` | San Francisco Bay Area |
| `brooklyn.py` | Brooklyn, NYC terrain |
| `new_york_city.py` | New York City terrain |
| `victor_idaho.py` | Victor, Idaho (small town with 1m DEM) |

Run any example with:

```bash
python examples/playground.py
```
