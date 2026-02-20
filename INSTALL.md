# Installation Guide

RTXpy requires an NVIDIA GPU and CUDA drivers. There are three ways to install it, listed from easiest to most flexible.

## Prerequisites

- **NVIDIA GPU**: Turing architecture or newer (RTX 20xx+ / T4+)
- **NVIDIA driver**: 590+ (for OptiX 9.1)
- **CUDA**: 12.x or newer
- **OS**: Linux (x86_64) or Windows 10/11

Verify your GPU and driver:

```bash
nvidia-smi
```

## Option 1: Conda (recommended)

The [makepath Anaconda channel](https://anaconda.org/makepath/rtxpy) provides
pre-built packages for Python 3.10–3.13 with OptiX bindings, the compiled PTX
kernel, and all analysis/viewer dependencies bundled — no SDK download or
separate install steps required.

```bash
conda create -n rtxpy python=3.12 -y
conda activate rtxpy
conda install -c makepath -c conda-forge rtxpy
```

> **Important:** Install rtxpy in a single command with both `-c makepath` and
> `-c conda-forge` channels. Do not install cupy/numba separately first — conda-forge
> may default to CUDA 13 packages, which conflict with the current rtxpy build.
> Installing everything in one solve lets the solver pick compatible CUDA versions.

This installs rtxpy 0.0.6 with everything needed to run the examples and
interactive viewer (numpy, numba, cupy, zarr, xarray, rioxarray, scipy,
matplotlib, moderngl, duckdb, and more).

Verify it works:

```python
from rtxpy import RTX
r = RTX()  # should succeed without errors
```

### Using environment files

Pre-made environment files are provided in the `environments/` directory:

```bash
# Python 3.12 (recommended)
conda env create -f environments/test-py312.yml
conda activate rtxpy-test-py312

# Other Python versions: test-py310.yml, test-py311.yml, test-py313.yml
```

### Windows

Windows builds use the system CUDA Toolkit rather than conda CUDA packages:

```bash
conda create -n rtxpy python=3.12 -y
conda activate rtxpy
conda install -c makepath -c conda-forge rtxpy
```

Ensure the CUDA Toolkit 12.x is installed and `nvcc` is on your PATH.

> **Package name gotchas on conda-forge:**
> - Python GLFW bindings are `pyglfw`, not `glfw` (that's the C library only)
> - xrspatial is `xarray-spatial`, not `xrspatial`
> - Use `duckdb<1.4` — versions 1.4+ have a regression with Overture Maps queries (already pinned in the conda package)

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

Install [pyoptix-contrib](https://pypi.org/project/pyoptix-contrib/) — the
NVIDIA OptiX Python bindings. Building from source requires the OptiX SDK
headers:

```bash
# Get the OptiX SDK headers (no NVIDIA account required)
git clone --depth 1 https://github.com/NVIDIA/optix-dev.git /tmp/optix-dev

# Build and install the Python bindings
CMAKE_PREFIX_PATH=/tmp/optix-dev pip install pyoptix-contrib
```

> **Warning:** There is a *different* `optix` package on PyPI (optical system
> design library). Do **not** run `pip install optix` — it is unrelated to
> NVIDIA OptiX. The correct package is `pyoptix-contrib`.

### Step 3: Install rtxpy

```bash
# From the repo root (editable install with all extras)
pip install -e ".[all]"

# Additional deps for the example scripts
pip install rioxarray xarray-spatial "duckdb<1.4" matplotlib requests scipy
```

Or use the dev environment file to set up all conda deps in one step:

```bash
conda env create -f environments/dev.yml
conda activate rtxpy-dev

# Install pyoptix-contrib (requires OptiX headers)
git clone --depth 1 https://github.com/NVIDIA/optix-dev.git /tmp/optix-dev
CMAKE_PREFIX_PATH=/tmp/optix-dev pip install pyoptix-contrib

# Editable install of rtxpy
pip install -e ".[all]"
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

The build script automatically clones the OptiX 9.1 SDK headers, compiles the
PTX kernel, and installs pyoptix-contrib during the build process.

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
they should be bundled. If using pip, install them:

```bash
git clone --depth 1 https://github.com/NVIDIA/optix-dev.git /tmp/optix-dev
CMAKE_PREFIX_PATH=/tmp/optix-dev pip install pyoptix-contrib
```

### `Could NOT find OptiX (missing: OptiX_ROOT_DIR)`

When building pyoptix-contrib from source, set the path to the OptiX SDK headers:

```bash
CMAKE_PREFIX_PATH=/path/to/optix-dev pip install pyoptix-contrib
```

### `nothing provides cuda-cudart >=12.9.79,<13.0a0`

This happens when cupy or numba was installed first and pulled in CUDA 13
packages, which conflict with rtxpy's CUDA 12 build. Fix: create a fresh
environment and install rtxpy in a single solve:

```bash
conda create -n rtxpy python=3.12 -y
conda activate rtxpy
conda install -c makepath -c conda-forge rtxpy
```

This lets the solver pick consistent CUDA versions for all packages.

### DuckDB errors when fetching Overture Maps data

DuckDB 1.4.x has a regression with S3/httpfs. Pin to an older version:

```bash
conda install -c conda-forge "duckdb<1.4"
# or
pip install "duckdb<1.4"
```

The makepath conda package already pins `duckdb<1.4`.

### `Cannot detect window with OpenGL support` / `libEGL.so: cannot open shared object file`

The conda-forge `libegl` package only ships `libEGL.so.1` (versioned), but
ModernGL's loader needs the unversioned `libEGL.so` symlink. Install the
development package:

```bash
conda install -c conda-forge libegl-devel
```

### `No module named 'glfw'` (after conda install glfw)

The conda-forge `glfw` package is the C library. Install the Python bindings:

```bash
conda install -c conda-forge pyglfw
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
