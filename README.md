# RTXpy

GPU-accelerated terrain analysis for the xarray ecosystem. Compute hillshade, viewshed, slope — get DataArrays back. Build a Dataset, then explore it interactively in 3D. Built-in data fetching (DEM, buildings, roads, water, fire, wind) makes it easy to go from a bounding box to a full scene.

![Crater Lake Viewshed Demo](examples/images/playground_demo.gif)

*Real-time viewshed analysis with GPU-accelerated ray tracing. Green areas are visible from the observer position (blue dot). Run `python examples/playground.py` to try it interactively.*

## Quick Start

Fetch terrain, analyze, and explore — all from a bounding box:

```python
from rtxpy import fetch_dem
import rtxpy

# Download 30m terrain (cached after first run)
dem = fetch_dem(
    bounds=(-122.3, 42.8, -121.9, 43.0),
    output_path='crater_lake.zarr',
    source='copernicus',
)
dem = dem.rtx.to_cupy()

# Analysis results are standard xarray DataArrays
hillshade = dem.rtx.hillshade(shadows=True)
viewshed = dem.rtx.viewshed(x=500, y=300, observer_elev=2)

# Interactive 3D terrain exploration
dem.rtx.explore()
```

Build a Dataset with multiple layers, then explore them together:

```python
import xarray as xr
from xrspatial import slope, aspect
from rtxpy import fetch_dem, fetch_buildings, fetch_roads

bounds = (-122.3, 42.8, -121.9, 43.0)
dem = fetch_dem(bounds, 'terrain.zarr', source='srtm', crs='EPSG:5070')
dem = dem.rtx.to_cupy()

ds = xr.Dataset({
    'elevation': dem,
    'slope': slope(dem),
    'aspect': aspect(dem),
})

# Fetch and place vector features
roads = fetch_roads(bounds, crs='EPSG:5070')
ds.rtx.place_roads(roads, z='elevation')

bldgs = fetch_buildings(bounds, source='overture', crs='EPSG:5070')
ds.rtx.place_buildings(bldgs, z='elevation')

# G cycles layers, N toggles geometry, U drapes satellite tiles
ds.rtx.explore(z='elevation', mesh_type='voxel')
```

## Scene files

Pack everything into a single zarr file — elevation, buildings, roads, water, wind, weather — then explore it offline without re-fetching. Good for sharing scenes or working on machines without network access.

### Build from the command line

```bash
# Grand Canyon — fetches 30m DEM, Overture buildings + roads + water, Open-Meteo wind/weather
rtxpy-build-scene -112.2 36.0 -112.0 36.2 grand_canyon.zarr

# Add fire detections, skip roads
rtxpy-build-scene -112.2 36.0 -112.0 36.2 grand_canyon.zarr --fires --no-roads

# Add wind to an existing scene without re-fetching everything
rtxpy-build-scene -112.2 36.0 -112.0 36.2 grand_canyon.zarr --resume
```

Options: `--no-buildings`, `--no-roads`, `--no-water`, `--no-wind`, `--no-weather`, `--hydro`, `--fires`, `--resume`.

### Build from Python

```python
from rtxpy.scene import build_scene, explore_scene
from rtxpy import LANDSCAPES

loc = LANDSCAPES['grand_canyon']
build_scene(loc, 'grand_canyon.zarr', crs=loc.crs)

# Load and launch the viewer in one call
explore_scene('grand_canyon.zarr')
```

119 preset locations ship with the package — countries, cities, and landscapes. Each carries a recommended projected CRS and its linear unit:

```python
from rtxpy import COUNTRIES, CITIES, LANDSCAPES
from rtxpy.scene_locations import find

loc = CITIES['tokyo']
loc.crs    # 'EPSG:32654' (UTM 54N)
loc.units  # 'meters'

build_scene(loc, 'tokyo.zarr', crs=loc.crs)

# Search by name
find('canyon')  # {'landscape/grand_canyon': Location(..., crs='EPSG:32612'), ...}
```

**Why these CRS?** The explore() viewer needs a projected (metric) CRS so that terrain spacing, building heights, and particle simulations all work in real-world units. Cities and landscapes use UTM — the zone is computed automatically from the bounding box center, giving meter-accurate coordinates anywhere on earth. Countries with well-established national grids use those instead (British National Grid for the UK, Lambert-93 for France, Conus Albers for the US, etc.) since they minimize distortion over the full country extent. All CRS in the current set use meters.

The zarr is self-contained: elevation stored as CF-encoded int16 with blosc compression, meshes spatially chunked to match the DEM grid, and wind/weather/hydro in their own groups. See `docs/proposals/scene-zarr-spec.md` for the full format.

## Prerequisites

- **NVIDIA GPU**: Maxwell architecture or newer (GTX 900+ / RTX series)
- **NVIDIA driver**: 455.28+ (Linux) or 456.71+ (Windows)
- **CUDA**: 12.x or newer

See [INSTALL.md](INSTALL.md) for detailed instructions and troubleshooting.

## Installation

### Conda (recommended)

```bash
conda create -n rtxpy python=3.12 -y
conda activate rtxpy
conda install -c makepath -c conda-forge rtxpy

# Additional deps for examples and interactive viewer
conda install -c conda-forge \
    xarray rioxarray xarray-spatial \
    pyproj pillow pyglfw moderngl scipy \
    "duckdb<1.4" requests matplotlib
```

### Pip + Conda hybrid (from source)

```bash
# GPU foundation via conda
conda create -n rtxpy-dev python=3.12 -y
conda activate rtxpy-dev
conda install -c conda-forge cupy numba zarr

# OptiX SDK headers (needed for pyoptix-contrib build and PTX compilation)
git clone --depth 1 https://github.com/NVIDIA/optix-dev.git /tmp/optix-dev
CMAKE_PREFIX_PATH=/tmp/optix-dev \
    pip install pyoptix-contrib

# Install rtxpy (editable)
pip install -e ".[all]"
```

### Development

```bash
# After completing the pip + conda hybrid setup above:
pip install -e ".[tests]"
pytest -v rtxpy/tests
```

## Features

- **Analysis arrays** — `hillshade()`, `slope()`, `aspect()`, `viewshed()` return xarray DataArrays that fit into your existing Dataset
- **Data fetching** — `fetch_dem()`, `fetch_buildings()`, `fetch_roads()`, `fetch_water()`, `fetch_wind()`, `fetch_firms()` — go from a bounding box to real data with automatic caching
- **3D feature placement** — extrude buildings, drape roads, scatter custom meshes on terrain
- **Interactive viewer** — `explore()` renders your Dataset in 3D with keyboard/mouse controls, satellite tiles, wind particles, and real-time viewshed
- **Rendering** — perspective camera with shadows, fog, AO, depth-of-field, colormaps for static images and GIF animations
- **Mesh I/O** — load GLB/OBJ/STL, save/load zarr scenes, export STL

## Documentation

- **[Getting Started](docs/getting-started.md)** — installation, prerequisites, first example, how the accessor works
- **[User Guide](docs/user-guide.md)** — task-oriented workflows for analysis, placement, rendering, and the interactive viewer
- **[API Reference](docs/api-reference.md)** — complete method signatures, parameters, and return values
- **[Examples](docs/examples.md)** — annotated walkthrough, quick recipes, and example scripts

## Low-Level API

For custom ray tracing without the xarray accessor:

```python
import numpy as np
from rtxpy import RTX

rtx = RTX()

verts = np.float32([0,0,0, 1,0,0, 0,1,0, 1,1,0])
triangles = np.int32([0,1,2, 2,1,3])
rtx.build(0, verts, triangles)

rays = np.float32([0.33, 0.33, 100, 0, 0, 0, -1, 1000])
hits = np.float32([0, 0, 0, 0])
rtx.trace(rays, hits, 1)

print(hits)  # [100.0, 0.0, 0.0, 1.0]
```

## Building the PTX Kernel

```bash
GPU_ARCH=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | tr -d '.')
nvcc -ptx -o rtxpy/kernel.ptx cuda/kernel.cu \
    -arch=sm_${GPU_ARCH} \
    -I/path/to/OptiX-SDK/include \
    -I cuda \
    --use_fast_math
```

## Building with Conda

```bash
conda install conda-build
conda build conda-recipe
conda install --use-local rtxpy
```

Auto-detects GPU architecture, downloads OptiX headers, compiles PTX, and installs everything. Override with `GPU_ARCH=86` or `OPTIX_VERSION=8.0.0`. See `conda-recipe/README.md` for details.

## WSL2 Support

See [Getting Started — WSL2 Setup](docs/getting-started.md#wsl2-setup) for instructions on getting OptiX working on WSL2.
