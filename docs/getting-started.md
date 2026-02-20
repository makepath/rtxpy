# Getting Started

rtxpy adds GPU-accelerated ray tracing to the xarray ecosystem. It works the way xarray-spatial does — you call a function on a DataArray and get a DataArray back — but the computations run on the GPU via NVIDIA OptiX. Hillshade, slope, viewshed, and other analysis results drop straight into your existing `xr.Dataset`. When you want to see your data in 3D, `explore()` launches an interactive viewer that renders the whole Dataset on the GPU.

## Prerequisites

- **NVIDIA GPU** with RTX support (Maxwell architecture or newer)
- **NVIDIA driver**: 456.71+ (Windows) or 455.28+ (Linux)
- **OptiX SDK** 7.6+ — set the `OptiX_INSTALL_DIR` environment variable
- **CUDA** 12.x+

## Installation

### Conda (recommended)

Linux:

```bash
conda install -c conda-forge cupy rioxarray matplotlib requests jupyter makepath::rtxpy
```

Windows:

```bash
conda install -c conda-forge cupy rioxarray matplotlib requests jupyter nvidia::cudatoolkit makepath::rtxpy
```

### Pip (from source)

First install the OptiX Python bindings:

```bash
export OptiX_INSTALL_DIR=/path/to/OptiX-SDK
pip install pyoptix-contrib
```

Then install rtxpy:

```bash
pip install rtxpy
```

### Development install

```bash
export OptiX_INSTALL_DIR=/path/to/OptiX-SDK
pip install pyoptix-contrib
pip install -ve .
```

### Conda build from source

```bash
conda install conda-build
conda build conda-recipe
conda install --use-local rtxpy
```

The conda build auto-detects your GPU architecture, downloads OptiX headers, compiles the PTX kernel, and installs everything. Override with `GPU_ARCH=86` or `OPTIX_VERSION=8.0.0` as needed. See `conda-recipe/README.md` for details.

## WSL2 Setup

OptiX requires extra steps on WSL2. The Windows GPU driver exposes CUDA to WSL, but the OptiX runtime libraries are not included by default.

### 1. OptiX Ray Tracing Libraries

Download the **Linux** display driver matching your Windows driver version (e.g., `NVIDIA-Linux-x86_64-590.44.01.run`) from the [NVIDIA driver archive](https://www.nvidia.com/en-us/drivers/unix/). You only need to extract it — do **not** install it inside WSL.

```bash
# Extract (do NOT run the installer)
chmod +x NVIDIA-Linux-x86_64-590.44.01.run
./NVIDIA-Linux-x86_64-590.44.01.run -x
```

Copy these files from the extracted directory into `C:\Windows\System32\lxss\lib` (which WSL mounts at `/usr/lib/wsl/lib`):

| Source file | Rename to |
|---|---|
| `libnvoptix.so.590.44.01` | `libnvoptix.so.1` |
| `libnvidia-rtcore.so.590.44.01` | *(keep original name)* |
| `libnvidia-ptxjitcompiler.so.590.44.01` | `libnvidia-ptxjitcompiler.so.1` |

Replace `590.44.01` with your actual driver version.

Then ensure the library path is set and restart WSL:

```bash
# In your ~/.bashrc or ~/.zshrc
export LD_LIBRARY_PATH=/usr/lib/wsl/lib:$LD_LIBRARY_PATH
```

```powershell
# From PowerShell — restart WSL
wsl --shutdown
```

Verify OptiX works:

```python
python -c "import optix; print(optix.version())"
```

### 2. AI Denoiser (Shift+D in explore)

The OptiX AI Denoiser requires a weights file called `nvoptix.bin` (~60 MB). Without it, pressing Shift+D in `explore()` prints a warning and denoising is skipped.

The extracted Linux driver package includes `nvoptix.bin`. Copy it to the standard location:

```bash
sudo mkdir -p /usr/share/nvidia
sudo cp NVIDIA-Linux-x86_64-590.44.01/nvoptix.bin /usr/share/nvidia/nvoptix.bin
```

Verify the denoiser works:

```python
python -c "
from rtxpy import RTX
from rtxpy.rtx import _ensure_denoiser
import numpy as np
rtx = RTX()
rtx.build(0, np.float32([0,0,0,1,0,0,0,1,0]), np.int32([0,1,2]))
print('Denoiser ready:', _ensure_denoiser(64, 64))
"
```

If you see `Denoiser ready: True`, the denoiser is working. If you get a warning about missing weights, double-check that `/usr/share/nvidia/nvoptix.bin` exists and is readable.

### Troubleshooting

- **`libnvoptix.so.1: cannot open shared object file`** — the library wasn't copied to `C:\Windows\System32\lxss\lib`, or `LD_LIBRARY_PATH` doesn't include `/usr/lib/wsl/lib`.
- **`Unable to load denoiser weights`** — `nvoptix.bin` is missing from `/usr/share/nvidia/`. Copy it from the extracted driver package.
- **Driver version mismatch** — the Linux driver version you extract must match your Windows driver version. Check with `nvidia-smi` in WSL.

Reference: [NVIDIA Forums — OptiX on WSL2](https://forums.developer.nvidia.com/t/problem-running-optix-7-6-in-wsl/239355/8)

## Get Data and Explore in 5 Lines

rtxpy includes `fetch_dem()` so you can go from zero to exploring real terrain immediately — no data prep needed:

```python
from rtxpy import fetch_dem

dem = fetch_dem(
    bounds=(-122.3, 42.8, -121.9, 43.0),  # Crater Lake (west, south, east, north)
    output_path='crater_lake.zarr',
    source='copernicus',
)
dem = dem.rtx.to_cupy()
dem.rtx.explore()
```

`fetch_dem()` downloads tiles from public sources, merges/clips/reprojects them, and caches the result. On subsequent runs the cached file loads instantly. Available sources:

| Source | Resolution | Coverage |
|--------|-----------|----------|
| `'copernicus'` | 30m | Global |
| `'srtm'` | ~30m | Global |
| `'usgs_10m'` | ~10m | US |
| `'usgs_1m'` | 1m (lidar) | US (partial) |

Output supports `.zarr` (chunked, compressed) or `.tif` (GeoTIFF).

Beyond DEMs, rtxpy can fetch buildings, roads, water, fire detections, and wind data — all with the same pattern of `fetch_*(bounds, ...)` with automatic caching. See [Fetching Remote Data](user-guide.md#fetching-remote-data) for the full list.

## Two Workflows

### Workflow 1: Analysis arrays for your Dataset

rtxpy analysis functions work like xarray-spatial — call a function, get a DataArray back, add it to your Dataset:

```python
import rtxpy
import xarray as xr
from rtxpy import fetch_dem

dem = fetch_dem(bounds=(-122.3, 42.8, -121.9, 43.0),
               output_path='crater_lake.zarr', source='srtm', crs='EPSG:5070')
dem = dem.rtx.to_cupy()

# Each call returns a DataArray with the same coords as the input
ds = xr.Dataset({
    'elevation': dem,
    'hillshade': dem.rtx.hillshade(shadows=True),
    'slope': dem.rtx.slope(),
    'aspect': dem.rtx.aspect(),
    'viewshed': dem.rtx.viewshed(x=500, y=300, observer_elev=2),
})

# Use the results like any other xarray data
ds['slope'].plot()
ds.to_zarr('analysis_results.zarr')
```

The result arrays are standard xarray DataArrays — plot them, save them, combine them with other tools. No special rtxpy serialization needed.

### Workflow 2: Build a Dataset, then explore it in 3D

When you have a Dataset with multiple analysis layers (from rtxpy, xarray-spatial, or any other tool), `explore()` lets you visualize them all interactively on a 3D terrain:

```python
import xarray as xr
from xrspatial import slope, aspect, quantile
from rtxpy import fetch_dem, fetch_buildings, fetch_roads
import rtxpy

dem = fetch_dem(bounds=(-122.3, 42.8, -121.9, 43.0),
               output_path='crater_lake.zarr', source='srtm', crs='EPSG:5070')
dem = dem.rtx.to_cupy()

# Build a Dataset with layers from any source
ds = xr.Dataset({
    'elevation': dem,
    'slope': slope(dem),           # xrspatial
    'aspect': aspect(dem),         # xrspatial
    'quantile': quantile(dem),     # xrspatial
})

# Fetch and place 3D features from public data sources
roads = fetch_roads(bounds=(-122.3, 42.8, -121.9, 43.0), crs='EPSG:5070')
ds.rtx.place_roads(roads, z='elevation')

buildings = fetch_buildings(bounds=(-122.3, 42.8, -121.9, 43.0),
                            source='overture', crs='EPSG:5070')
ds.rtx.place_buildings(buildings, z='elevation')

# Explore everything together — press G to cycle layers, N for geometry
ds.rtx.explore(z='elevation', mesh_type='voxel')
```

In the viewer, press **G** to cycle through elevation / slope / aspect / quantile on the terrain surface, **N** to toggle building and road layers, **U** to drape satellite tiles, and **V** for real-time viewshed.

## How the Accessor Works

rtxpy uses xarray's [accessor pattern](https://docs.xarray.dev/en/stable/internals/extending-xarray.html). When you `import rtxpy`, it registers the `.rtx` namespace on DataArrays and Datasets automatically.

```python
# DataArray accessor — analysis + placement
dem.rtx.hillshade()
dem.rtx.viewshed(x=500, y=300)
dem.rtx.place_buildings(bldgs)
dem.rtx.explore()

# Dataset accessor — explore with multiple layers
ds.rtx.explore(z='elevation')   # G key cycles through Dataset variables
ds.rtx.place_roads(roads, z='elevation')
```

The accessor lazily creates and caches an `RTX` instance (the OptiX scene) on first use. This means:

- Multiple calls share the same GPU acceleration structure
- Placed meshes (buildings, roads) persist across `render()` and `explore()` calls
- Call `dem.rtx.clear()` to reset the scene

## Next Steps

- [User Guide](user-guide.md) — task-oriented workflows for terrain analysis, 3D features, rendering, and the interactive viewer
- [API Reference](api-reference.md) — complete method signatures and parameters
- [Examples](examples.md) — annotated recipes and walkthrough of the playground demo
