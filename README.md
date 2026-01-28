# RTXpy

Ray tracing using CUDA, accessible from Python.

![Crater Lake Viewshed Demo](examples/images/playground_demo.gif)

*Real-time viewshed analysis with GPU-accelerated ray tracing. Green areas are visible from the observer position (white dot). Run `python examples/playground.py` to try it interactively.*

## Prerequisites

- NVIDIA GPU with RTX support (Maxwell architecture or newer)
- NVIDIA driver version:
  - 456.71 or newer for Windows
  - 455.28 or newer for Linux
- OptiX SDK 7.6+ (set `OptiX_INSTALL_DIR` environment variable)
- CUDA 12.x+

## Installation

First, install the OptiX Python bindings (otk-pyoptix):

```bash
export OptiX_INSTALL_DIR=/path/to/OptiX-SDK
pip install otk-pyoptix
```

Then install rtxpy:

```bash
pip install rtxpy
```

## Installation from source

To install RTXpy from source:

```bash
export OptiX_INSTALL_DIR=/path/to/OptiX-SDK
pip install otk-pyoptix
pip install -ve .
```

To run tests:

```bash
pip install -ve .[tests]
pytest -v rtxpy/tests
```

## Building kernel.ptx from source

If you need to rebuild the PTX kernel (e.g., for a different GPU architecture or OptiX version):

```bash
# Detect your GPU's compute capability (e.g., 75 for Turing, 86 for Ampere)
GPU_ARCH=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | tr -d '.')

# Compile for your GPU architecture
nvcc -ptx -o rtxpy/kernel.ptx cuda/kernel.cu \
    -arch=sm_${GPU_ARCH} \
    -I/path/to/OptiX-SDK/include \
    -I cuda \
    --use_fast_math
```

The CUDA source files are in the `cuda/` directory.

## Building with Conda

The easiest way to build rtxpy with all dependencies is using the included conda recipe:

```bash
# Install conda-build if not already installed
conda install conda-build

# Build the package (auto-detects GPU architecture)
conda build conda-recipe

# Or specify GPU architecture explicitly
GPU_ARCH=86 conda build conda-recipe  # For RTX 30xx/A100

# Install the built package
conda install --use-local rtxpy
```

The conda build automatically:
1. Clones OptiX SDK headers from NVIDIA/optix-dev
2. Detects your GPU architecture (or uses `GPU_ARCH` env var)
3. Compiles the PTX kernel for your GPU
4. Builds and installs otk-pyoptix
5. Installs rtxpy

You can also specify the OptiX version:
```bash
OPTIX_VERSION=7.7.0 conda build conda-recipe  # Requires driver 530.41+
OPTIX_VERSION=8.0.0 conda build conda-recipe  # Requires driver 535+
```

See `conda-recipe/README.md` for detailed documentation, GPU architecture reference, and troubleshooting.

## WSL2 Support

To get OptiX working on WSL2, follow the instructions from the NVIDIA forums:
https://forums.developer.nvidia.com/t/problem-running-optix-7-6-in-wsl/239355/8

Summary:
1. Install WSL 2 and enable CUDA
2. Download and extract the Linux display driver (e.g., `NVIDIA-Linux-x86_64-590.44.01.run`)
3. Extract with `./NVIDIA-Linux-x86_64-XXX.XX.run -x`
4. Copy the following files to `C:/Windows/System32/lxss/lib`:
   - `libnvoptix.so.XXX.00` (rename to `libnvoptix.so.1`)
   - `libnvidia-rtcore.so.XXX.00` (keep original name)
   - `libnvidia-ptxjitcompiler.so.XXX.00` (rename to `libnvidia-ptxjitcompiler.so.1`)
5. Add `/usr/lib/wsl/lib` to your `LD_LIBRARY_PATH`
6. Reset WSL cache with `wsl --shutdown` from PowerShell

## Usage

```python
import numpy as np
from rtxpy import RTX

# Create RTX instance
rtx = RTX()

# Define geometry (vertices and triangle indices)
verts = np.float32([0,0,0, 1,0,0, 0,1,0, 1,1,0])
triangles = np.int32([0,1,2, 2,1,3])

# Build acceleration structure
rtx.build(0, verts, triangles)

# Define rays: [ox, oy, oz, tmin, dx, dy, dz, tmax]
rays = np.float32([0.33, 0.33, 100, 0, 0, 0, -1, 1000])
hits = np.float32([0, 0, 0, 0])

# Trace rays
rtx.trace(rays, hits, 1)

# hits contains: [t, nx, ny, nz]
# t = distance to hit point (-1 if miss)
# nx, ny, nz = surface normal at hit point
print(hits)  # [100.0, 0.0, 0.0, 1.0]
```

For GPU-resident data, use CuPy arrays for better performance:

```python
import cupy

verts = cupy.float32([0,0,0, 1,0,0, 0,1,0, 1,1,0])
triangles = cupy.int32([0,1,2, 2,1,3])
rays = cupy.float32([0.33, 0.33, 100, 0, 0, 0, -1, 1000])
hits = cupy.float32([0, 0, 0, 0])

rtx.build(0, verts, triangles)
rtx.trace(rays, hits, 1)
```
