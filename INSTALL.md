# Installation Guide

RTXpy requires an NVIDIA GPU and CUDA drivers. There are three ways to install it, listed from easiest to most flexible.

## Prerequisites

- **NVIDIA GPU**: Maxwell architecture or newer (GTX 900+ / RTX series)
- **NVIDIA driver**: 455.28+ (Linux) or 456.71+ (Windows)
- **CUDA**: 12.x or newer
- **OS**: Linux (x86_64) or Windows 10/11

Verify your GPU and driver:

```bash
nvidia-smi
```

## Option 1: Conda (recommended)

The makepath Anaconda channel provides pre-built packages for Python 3.10–3.13
with the OptiX bindings bundled — no SDK download required.

### Core install

```bash
conda create -n rtxpy python=3.12 -y
conda activate rtxpy
conda install -c makepath -c conda-forge rtxpy
```

This installs rtxpy with its core dependencies (numpy, numba, cupy, zarr,
pyoptix-contrib). Verify it works:

```python
from rtxpy import RTX
r = RTX()  # should succeed without errors
```

### Full install (examples + interactive viewer)

To run the examples (e.g. `examples/trinidad.py`, `examples/playground.py`)
you need the analysis, viewer, and data-fetching dependencies:

```bash
conda install -c conda-forge \
    xarray rioxarray xarray-spatial \
    pyproj pillow pyglfw moderngl scipy \
    "duckdb<1.4" requests matplotlib
```

> **Package name gotchas on conda-forge:**
> - Python GLFW bindings are `pyglfw`, not `glfw` (that's the C library only)
> - xrspatial is `xarray-spatial`, not `xrspatial`
> - Use `duckdb<1.4` — versions 1.4+ have a regression with Overture Maps queries

### Windows

Windows builds use the system CUDA Toolkit rather than conda CUDA packages:

```bash
conda create -n rtxpy python=3.12 -y
conda activate rtxpy
conda install -c makepath -c conda-forge rtxpy
conda install -c conda-forge xarray rioxarray xarray-spatial ^
    pyproj pillow pyglfw moderngl scipy "duckdb<1.4" requests matplotlib
```

Ensure the CUDA Toolkit 12.x is installed and `nvcc` is on your PATH.

## Option 2: Pip + Conda hybrid (development)

Use this when you want an editable install from the repo with the latest code.
Conda provides GPU packages (cupy, numba) that pip can't build, while pip
handles everything else.

### Step 1: GPU foundation via conda

```bash
conda create -n rtxpy-dev python=3.12 -y
conda activate rtxpy-dev
conda install -c conda-forge cupy numba zarr
```

### Step 2: OptiX Python bindings

Install the OptiX SDK headers (needed for the build) and the Python bindings:

```bash
# Get the OptiX SDK headers (no NVIDIA account required)
git clone --depth 1 https://github.com/NVIDIA/optix-dev.git /tmp/optix-dev

# Build and install the Python bindings from PyPI
CMAKE_PREFIX_PATH=/tmp/optix-dev \
    pip install pyoptix-contrib
```

> **Warning:** There is a *different* `optix` package on PyPI (optical system
> design library). Do **not** run `pip install optix` — it is unrelated to
> NVIDIA OptiX. The correct package is `pyoptix-contrib`.

### Step 3: Install rtxpy

```bash
# From the repo root (editable install with all extras)
pip install -e ".[all]"

# Additional deps for the example scripts
pip install rioxarray xarray-spatial "duckdb<1.4" matplotlib requests
```

Verify:

```bash
python -c "from rtxpy import RTX; RTX(); print('OK')"
```

## Option 3: Build the conda package locally

Build from the conda recipe in the repo. Useful for creating packages for
internal distribution or testing recipe changes.

```bash
# Install conda-build if needed
conda install -n base conda-build

# Build for a specific Python version
conda-build conda-recipe --python 3.12 -c conda-forge --no-test

# Install the locally built package
conda create -n rtxpy-local python=3.12 -y
conda activate rtxpy-local
conda install -c local -c conda-forge rtxpy
```

The build script automatically clones the OptiX SDK headers and installs
pyoptix-contrib during the build process.

## Running the examples

Once installed, try the interactive examples:

```bash
# Crater Lake (smaller, good for first run)
python examples/playground.py

# Trinidad & Tobago coastal resilience analysis
python examples/trinidad.py

# Los Angeles
python examples/los_angeles.py
```

Examples download terrain and vector data on first run (cached for subsequent
runs). Press `H` in the viewer for keyboard controls.

## Troubleshooting

### `ModuleNotFoundError: No module named 'optix'`

The NVIDIA OptiX Python bindings are missing. If using conda from makepath,
they should be bundled. If using pip, see [Step 2](#step-2-optix-python-bindings)
above.

### `Could NOT find OptiX (missing: OptiX_ROOT_DIR)`

When building pyoptix-contrib from source, set the path to the OptiX SDK headers:

```bash
CMAKE_PREFIX_PATH=/path/to/optix-dev pip install ...
```

### `No module named 'glfw'` (after conda install glfw)

The conda-forge `glfw` package is the C library. Install the Python bindings:

```bash
conda install -c conda-forge pyglfw
```

### DuckDB errors when fetching Overture Maps data

DuckDB 1.4.x has a regression with S3/httpfs. Pin to an older version:

```bash
conda install -c conda-forge "duckdb<1.4"
# or
pip install "duckdb<1.4"
```

### `ImportError` for rtxpy picks up local directory

When running from the repo root, Python may import the local `rtxpy/` directory
instead of the installed package. Either `cd` to a different directory or use
an editable install (`pip install -e .`).

### cupy / numba installation via pip fails

These packages have complex native dependencies. Install them via conda:

```bash
conda install -c conda-forge cupy numba
```
