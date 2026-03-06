# User Guide

This guide covers rtxpy workflows by task. For method signatures and parameter details, see the [API Reference](api-reference.md).

## Getting Data

rtxpy is designed so you can start working with real data immediately. Every `fetch_*` function downloads from public sources, caches the result, and returns data ready to use. On subsequent runs, the cached file loads instantly.

### Terrain (DEM)

```python
from rtxpy import fetch_dem

dem = fetch_dem(
    bounds=(-122.3, 42.8, -121.9, 43.0),  # (west, south, east, north) WGS84
    output_path='terrain.zarr',             # .zarr or .tif
    source='copernicus',                    # 'copernicus', 'srtm', 'usgs_10m', 'usgs_1m'
    crs='EPSG:5070',                        # optional reprojection
)
```

| Source | Resolution | Coverage | Notes |
|--------|-----------|----------|-------|
| `'copernicus'` | 30m | Global | Copernicus GLO-30 DSM |
| `'srtm'` | ~30m | Global | USGS 1-arc-second |
| `'usgs_10m'` | ~10m | US | USGS 3DEP 1/3-arc-second |
| `'usgs_1m'` | 1m | US (partial) | USGS 3DEP lidar, ~30 MB/tile |

The `.zarr` format uses int16 + scale_factor with Blosc zstd compression — much smaller than GeoTIFF for large areas.

### Buildings, roads, water, and more

```python
from rtxpy import fetch_buildings, fetch_roads, fetch_water, fetch_wind, fetch_firms

bounds = (-122.3, 42.8, -121.9, 43.0)
crs = 'EPSG:5070'

# Building footprints (Microsoft or Overture Maps)
bldgs = fetch_buildings(bounds, source='overture', crs=crs,
                        cache_path='cache/buildings.geojson')

# Road networks (OSM or Overture Maps)
roads = fetch_roads(bounds, road_type='all', source='osm', crs=crs,
                    cache_path='cache/roads.geojson')

# Water features (OSM)
water = fetch_water(bounds, water_type='all', crs=crs,
                    cache_path='cache/water.geojson')

# Current wind data (Open-Meteo)
wind = fetch_wind(bounds, grid_size=20)

# NASA FIRMS fire detections (LANDSAT 30m)
fires = fetch_firms(bounds, date_span='7d',
                    cache_path='cache/fires.geojson')
```

All vector data functions return GeoJSON FeatureCollections that feed directly into `place_buildings()`, `place_roads()`, `place_water()`, and `place_geojson()`. The `crs` parameter reprojects to match your DEM so coordinates align automatically.

### Loading your own data

```python
import rioxarray

# GeoTIFF
dem = rioxarray.open_rasterio('elevation.tif').squeeze()

# Zarr
import xarray as xr
ds = xr.open_zarr('terrain.zarr')
dem = ds['elevation']
```

### GPU transfer

```python
import rtxpy  # registers .rtx accessor

dem = dem.rtx.to_cupy()
```

Uses pinned memory for fast host-to-device transfer. If data is already on the GPU, returns unchanged.

## Terrain Analysis

rtxpy analysis functions work like xarray-spatial: call a method on a DataArray, get a DataArray back with the same coordinates. The results are standard xarray — plot them, save them, combine them with other tools.

```python
# Build a Dataset with layers from multiple sources
import xarray as xr
from xrspatial import slope, aspect, quantile

ds = xr.Dataset({
    'elevation': dem,
    'hillshade': dem.rtx.hillshade(shadows=True),  # rtxpy (GPU ray tracing)
    'slope': slope(dem),                             # xrspatial
    'aspect': aspect(dem),                           # xrspatial
    'viewshed': dem.rtx.viewshed(x=500, y=300, observer_elev=2),  # rtxpy
    'quantile': quantile(dem),                       # xrspatial
})

# Use like any xarray Dataset
ds['viewshed'].plot()
ds.to_zarr('analysis.zarr')
```

All analysis methods return xarray DataArrays with the same coordinates as the input.

### Hillshade

```python
shade = dem.rtx.hillshade(
    shadows=True,       # cast shadow rays (half-brightness in shadows)
    azimuth=225,        # sun direction (degrees clockwise from north)
    angle_altitude=25,  # sun elevation above horizon (degrees)
)
```

### Slope

```python
s = dem.rtx.slope()  # degrees: 0 = flat, 90 = vertical
```

### Aspect

```python
a = dem.rtx.aspect()  # degrees: 0=N, 90=E, 180=S, 270=W; -1=flat
```

### Viewshed

```python
vis = dem.rtx.viewshed(
    x=500, y=300,         # observer position (pixel coordinates)
    observer_elev=2,      # height above terrain at observer
    target_elev=0,        # height above terrain at targets
)
# Returns viewing angle for visible cells, -1 for not visible
```

## 3D Feature Placement

All placement methods add geometry to the shared RTX scene. Features persist and are visible in both `render()` and `explore()`.

### Buildings

```python
from rtxpy import fetch_buildings

bldgs = fetch_buildings(
    bounds=(-61.5, 10.6, -61.4, 10.7),
    source='overture',    # 'microsoft' or 'overture'
    crs='EPSG:32620',     # match your DEM's CRS
    cache_path='cache/buildings.geojson',
)

dem.rtx.place_buildings(
    bldgs,
    elev_scale=0.33,        # height scaling (auto-computed if omitted)
    default_height_m=8.0,   # fallback when feature has no height
    mesh_cache='cache/buildings_mesh.npz',  # cache for faster reload
)
```

`place_buildings()` extrudes building footprint polygons into 3D geometry. Heights come from the GeoJSON `height` property (meters), scaled by `elev_scale` to match terrain visualization units.

### Roads

```python
from rtxpy import fetch_roads

roads = fetch_roads(
    bounds=(-122.3, 42.8, -121.9, 43.0),
    road_type='major',   # 'major', 'minor', or 'all'
    source='overture',   # 'osm' or 'overture'
    crs='EPSG:5070',
    cache_path='cache/roads.geojson',
)

dem.rtx.place_roads(
    roads,
    geometry_id='road',
    color=(0.30, 0.30, 0.30),  # dark grey
    height=3,                   # ribbon height above terrain
    mesh_cache='cache/roads_mesh.npz',
)
```

Roads are rendered as flat ribbons that follow the terrain surface.

### Water

```python
from rtxpy import fetch_water

water = fetch_water(
    bounds=(-61.6, 10.4, -61.2, 10.7),
    water_type='all',    # 'waterway', 'waterbody', or 'all'
    crs='EPSG:32620',
    cache_path='cache/water.geojson',
)

dem.rtx.place_water(
    water,
    body_height=0.5,
    mesh_cache_prefix='cache/water',
)
```

Automatically classifies features into major waterways (rivers, canals), minor waterways (streams, drains), and water bodies (lakes) with distinct colors.

### GeoJSON

For arbitrary GeoJSON features:

```python
dem.rtx.place_geojson(
    'landmarks.geojson',  # path, dict, or FeatureCollection
    height=10.0,
    label_field='name',   # group by this property
    geometry_id='poi',
    color=(0.9, 0.2, 0.2),
    merge=True,           # single GAS for performance
    extrude=True,         # solid 3D geometry (for polygons)
    mesh_cache='cache/landmarks_mesh.npz',
)
```

Geometry type handling:
- **Points** → glowing orbs hovering above terrain (or custom mesh instances)
- **LineStrings** → flat ribbons following terrain
- **Polygons** → outline ribbons or extruded solid geometry (with `extrude=True`)

Use `models` and `model_field` for per-feature 3D models:

```python
dem.rtx.place_geojson(
    geojson,
    models={'forest': 'tree.glb', 'urban': 'building.glb'},
    model_field='landcover',
    fill_spacing=5.0,
    fill_scale=2.0,
)
```

### Custom 3D Models

```python
dem.rtx.place_mesh(
    'tower.glb',                            # GLB, OBJ, STL, PLY, etc.
    positions=[(100, 50), (200, 150)],      # (x, y) pixel coordinates
    geometry_id='tower',
    scale=0.1,
    rotation_z='random',                    # random rotation per instance
    swap_yz=True,                           # Y-up models → Z-up
    center_xy=True,
    base_at_zero=True,
)
```

Both `mesh_source` and `positions` accept callables:

```python
def make_tower():
    from rtxpy import load_mesh
    return load_mesh('tower.glb', scale=2.7, swap_yz=True,
                     center_xy=True, base_at_zero=True)

def pick_hilltops(terrain):
    # terrain is a 2D numpy array
    # return [(x, y), ...] pixel coords
    ...

dem.rtx.place_mesh(make_tower, pick_hilltops)
```

### Mesh Caching with Zarr

Save placed meshes to a zarr store for instant loading on subsequent runs:

```python
# First run: place features and save
dem.rtx.place_buildings(bldgs)
dem.rtx.place_roads(roads)
dem.rtx.save_meshes('terrain.zarr')

# Subsequent runs: load instantly
dem.rtx.load_meshes('terrain.zarr')
```

Meshes are spatially partitioned to match the DEM's chunk layout, enabling chunk-based streaming in `explore()` via the `scene_zarr` parameter.

## Rendering

### Static images

```python
import numpy as np

H, W = dem.shape
elev = dem.values if hasattr(dem, 'values') else dem.data.get()

img = dem.rtx.render(
    camera_position=(W/2, -50, np.max(elev) + 200),
    look_at=(W/2, H/2, np.mean(elev)),
    fov=60.0,
    width=1920, height=1080,
    sun_azimuth=225,
    sun_altitude=45,
    shadows=True,
    ambient=0.15,
    fog_density=0.002,
    fog_color=(0.7, 0.8, 0.9),
    colormap='terrain',
    color_range=None,           # auto from data range
    vertical_exaggeration=None, # auto-computed
    ao_samples=16,              # ambient occlusion rays per pixel
    ao_radius=None,             # auto ~5% of scene diagonal
    sun_angle=0.25,             # soft shadow cone half-angle
    aperture=5.0,               # depth-of-field lens radius
    focal_distance=0.0,         # auto from camera-to-lookat
    alpha=False,                # True for RGBA with transparent sky
    output_path='render.png',   # saves automatically
)
# Returns float32 array (H, W, 3) with values [0-1]
```

### Flyover animation

```python
dem.rtx.flyover(
    'flyover.gif',
    duration=30.0,              # seconds
    fps=10.0,
    orbit_scale=0.6,            # orbit radius as fraction of terrain size
    altitude_offset=500.0,      # camera height above max elevation
    fov_range=(30, 70),         # dynamic zoom (None for constant FOV)
    colormap='terrain',
    shadows=True,
)
```

### 360-degree panoramic view

```python
dem.rtx.view(
    x=500, y=300, z=2500,       # camera position
    output_path='panorama.gif',
    duration=10.0,
    fps=12.0,
    look_distance=1000.0,
    look_down_angle=10.0,
    fov=70.0,
)
```

### Colormaps

Any matplotlib colormap name works: `'terrain'`, `'viridis'`, `'plasma'`, `'cividis'`, `'gray'`, etc. Use `'hillshade'` for grayscale shading without elevation coloring.

## Interactive Viewer

### The typical workflow: Dataset → explore

The most common way to use `explore()` is to build an xarray Dataset with analysis layers, then explore them all together in 3D:

```python
import xarray as xr
from xrspatial import slope, aspect, quantile
from rtxpy import fetch_dem, fetch_roads, fetch_buildings, fetch_wind
import rtxpy

bounds = (-122.3, 42.8, -121.9, 43.0)
crs = 'EPSG:5070'

# 1. Get data
dem = fetch_dem(bounds, output_path='terrain.zarr', source='srtm', crs=crs)
dem = dem.rtx.to_cupy()

# 2. Build a Dataset with whatever layers you need
ds = xr.Dataset({
    'elevation': dem,
    'slope': slope(dem),
    'aspect': aspect(dem),
    'quantile': quantile(dem),
})

# 3. Fetch and place vector features
roads = fetch_roads(bounds, crs=crs, cache_path='cache/roads.geojson')
ds.rtx.place_roads(roads, z='elevation')

bldgs = fetch_buildings(bounds, source='overture', crs=crs,
                        cache_path='cache/buildings.geojson')
ds.rtx.place_buildings(bldgs, z='elevation')

# 4. Explore — G cycles layers, N toggles geometry, U drapes tiles
wind = fetch_wind(bounds)
ds.rtx.explore(z='elevation', wind_data=wind, mesh_type='voxel')
```

Inside the viewer, press **G** to cycle through elevation / slope / aspect / quantile on the terrain surface. The data variables in your Dataset become the layers you can visualize.

### Launching on a DataArray

```python
dem.rtx.explore(
    width=1024,
    height=768,
    render_scale=0.5,    # render at half resolution for speed
    mesh_type='tin',     # 'tin' or 'voxel'
    subsample=2,         # start at 2x subsample (R/Shift+R to change)
)
```

### With satellite tiles

```python
dem.rtx.place_tiles('satellite')  # 'satellite', 'osm', or 'topo'
dem.rtx.explore()
```

### With streaming zarr data

```python
dem.rtx.explore(scene_zarr='terrain.zarr')  # streams mesh chunks by camera position
```

### With wind animation

```python
from rtxpy import fetch_wind

wind = fetch_wind(bounds, grid_size=15)
dem.rtx.explore(wind_data=wind)  # Shift+W to toggle
```

### Keyboard Controls

| Key | Action |
|-----|--------|
| **W** / **Up** | Move forward |
| **S** / **Down** | Move backward |
| **A** / **Left** | Strafe left |
| **D** / **Right** | Strafe right |
| **Q** / **Page Up** | Move up |
| **E** / **Page Down** | Move down |
| **I** / **J** / **K** / **L** | Look up / left / down / right |
| **Click+Drag** | Pan (slippy-map style) |
| **Scroll wheel** | Zoom in/out (FOV) |
| **+** / **-** | Increase / decrease movement speed |
| **G** | Cycle terrain color (elevation, overlay layers) |
| **U** | Cycle basemap (none, satellite, osm, topo) |
| **N** | Cycle geometry layer visibility (none, all, per-group) |
| **P** | Jump to previous geometry in current group |
| **,** / **.** | Decrease / increase overlay alpha |
| **O** | Place viewshed observer at look-at point |
| **Shift+O** | Cycle drone mode (off, 3rd person, FPV) |
| **V** | Toggle viewshed overlay |
| **[** / **]** | Decrease / increase observer height |
| **R** | Decrease terrain resolution (coarser) |
| **Shift+R** | Increase terrain resolution (finer) |
| **Shift+A** | Toggle distance-based terrain LOD |
| **Z** | Decrease vertical exaggeration |
| **Shift+Z** | Increase vertical exaggeration |
| **B** | Toggle mesh type (TIN / voxel) |
| **Y** | Cycle color stretch (linear, sqrt, cbrt, log) |
| **T** | Toggle shadows |
| **0** | Toggle ambient occlusion (progressive) |
| **Shift+D** | Toggle OptiX AI Denoiser |
| **C** | Cycle colormap (gray, terrain, viridis, plasma, cividis) |
| **Shift+F** | Fetch / toggle FIRMS fire layer |
| **Shift+W** | Toggle wind particle animation |
| **F** | Save screenshot |
| **M** | Toggle minimap overlay |
| **H** | Toggle help overlay |
| **X** | Exit |

### Mouse controls

- **Click + drag** pans the view (slippy-map style)
- **Scroll wheel** adjusts FOV (zoom)

## Fetching Remote Data

See [Getting Data](#getting-data) at the top of this guide for the full list of `fetch_*` functions with sources and parameters. Quick reference:

| Function | Data | Sources |
|----------|------|---------|
| `fetch_dem()` | Elevation rasters | Copernicus 30m, SRTM 30m, USGS 10m, USGS 1m |
| `fetch_buildings()` | Building footprints | Microsoft, Overture Maps |
| `fetch_roads()` | Road networks | OSM, Overture Maps |
| `fetch_water()` | Water features | OSM |
| `fetch_firms()` | Fire detections | NASA FIRMS LANDSAT 30m |
| `fetch_wind()` | Wind speed/direction | Open-Meteo |
| `fetch_osm()` | Arbitrary OSM features | OSM via osmnx |

All functions cache automatically. Pass `crs=` to reproject to your DEM's coordinate system so features align without manual transforms.

## Mesh I/O

### Loading meshes

```python
from rtxpy import load_mesh

verts, indices = load_mesh(
    'model.glb',      # GLB, OBJ, STL, PLY, etc.
    scale=0.1,
    swap_yz=True,      # Y-up → Z-up
    center_xy=True,
    base_at_zero=True,
)
```

### Saving/loading scenes to zarr

```python
# Save all placed meshes
dem.rtx.save_meshes('terrain.zarr')

# Load back
dem.rtx.load_meshes('terrain.zarr')

# Load specific chunks only
dem.rtx.load_meshes('terrain.zarr', chunks=[(0, 0), (0, 1)])
```

### Exporting to STL

```python
from rtxpy import write_stl

verts, indices = dem.rtx.triangulate()
write_stl('terrain.stl', verts, indices)
```

## Render Graph

The render graph lets you define a pipeline of render passes as a DAG. Instead of editing one monolithic render function, you declare what each pass reads and writes. The graph handles execution order, buffer allocation, and capability gating.

```python
from rtxpy import RenderGraph, RenderPass, BufferDesc

rgb = BufferDesc(dtype="float32", channels=3, per_pixel=True)
scalar = BufferDesc(dtype="float32", channels=1, per_pixel=True)

class MyShadePass(RenderPass):
    def __init__(self):
        super().__init__("shade", outputs={"color": rgb})

    def execute(self, buffers):
        buffers["color"][:] = 0.5  # your shading logic here

class MyDenoisePass(RenderPass):
    def __init__(self):
        super().__init__(
            "denoise",
            inputs={"color": rgb},
            outputs={"denoised_color": rgb},
            requires=["optix_denoiser"],
        )

    def execute(self, buffers):
        buffers["denoised_color"][:] = buffers["color"]  # denoiser call here

class MyTonemapPass(RenderPass):
    def __init__(self):
        super().__init__(
            "tonemap",
            inputs={"denoised_color": rgb},
            outputs={"final": rgb},
        )

    def execute(self, buffers):
        buffers["final"][:] = buffers["denoised_color"] ** (1.0 / 2.2)

graph = RenderGraph(width=1920, height=1080)
graph.add_pass(MyShadePass())
graph.add_pass(MyDenoisePass())
graph.add_pass(MyTonemapPass())

# If denoiser unavailable, tonemap reads 'color' directly
graph.set_fallback("denoised_color", "color")

compiled = graph.compile(capabilities={"optix_denoiser": False})
result = compiled.execute()
# result["final"] contains the output image
```

The graph skips passes whose `requires` capabilities aren't present, wiring fallbacks so downstream passes still work. Buffer lifetime analysis reuses GPU memory when buffers don't overlap in time.

See the [API Reference](api-reference.md#render-graph) for the full interface.

## Performance Tips

- **Enable terrain LOD**: Press `Shift+A` in the viewer to activate distance-based LOD. Nearby tiles render at full detail while distant tiles are automatically subsampled, cutting triangle count without visible quality loss
- **Subsample large DEMs**: `dem[::2, ::2]` or `explore(subsample=4)` — 4x subsample is 16x less geometry
- **Lower render_scale**: `explore(render_scale=0.25)` renders at quarter resolution for faster interaction
- **Cache meshes**: Use `mesh_cache` parameter in `place_buildings()`, `place_roads()`, etc. to skip GeoJSON parsing on reload
- **Use zarr for streaming**: Save meshes to zarr with `save_meshes()`, then use `explore(scene_zarr=...)` to stream chunks by camera position
- **Merge geometry**: Use `merge=True` in `place_geojson()` to combine features into a single GAS — much faster for thousands of features
- **Reuse RTX instances**: The accessor caches its RTX instance automatically. For the low-level API, pass `rtx=` to avoid rebuilding acceleration structures
- **Zarr output for DEMs**: `fetch_dem(..., output_path='terrain.zarr')` uses int16 + Blosc compression, much smaller than GeoTIFF
