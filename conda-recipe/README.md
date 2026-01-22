# RTXpy Conda Recipe

This conda recipe builds rtxpy with all required dependencies including OptiX and CUDA support.

## Prerequisites

- NVIDIA GPU with compute capability 5.2+ (Maxwell or newer)
- NVIDIA driver 530.41+ (for OptiX 7.7 compatibility)
- conda-build installed

## Building the Package

### Basic build (auto-detect GPU architecture):

```bash
conda build conda-recipe
```

### Build for a specific GPU architecture:

```bash
# For Turing (RTX 20xx, T4) - sm_75
GPU_ARCH=75 conda build conda-recipe

# For Ampere (RTX 30xx, A100) - sm_86
GPU_ARCH=86 conda build conda-recipe

# For Ada Lovelace (RTX 40xx) - sm_89
GPU_ARCH=89 conda build conda-recipe

# For broad compatibility (Turing+) - sm_75 (default)
GPU_ARCH=75 conda build conda-recipe
```

### Build with a specific OptiX version:

```bash
# Use OptiX 7.6 (requires driver 522.25+)
OPTIX_VERSION=7.6.0 conda build conda-recipe

# Use OptiX 8.0 (requires driver 535+)
OPTIX_VERSION=8.0.0 conda build conda-recipe
```

## Installing the Built Package

```bash
conda install --use-local rtxpy
```

## What the Build Does

1. **Clones OptiX SDK headers** from NVIDIA/optix-dev (v7.7.0 by default)
2. **Detects GPU architecture** or uses the specified `GPU_ARCH`
3. **Compiles kernel.cu to PTX** for the target architecture
4. **Installs otk-pyoptix** from NVIDIA's repository
5. **Installs rtxpy** with the compiled PTX kernel

## GPU Architecture Reference

| GPU Series | Architecture | Compute Capability |
|------------|--------------|-------------------|
| GTX 900, Tesla M | Maxwell | sm_52 |
| GTX 1000, Tesla P | Pascal | sm_60, sm_61 |
| RTX 2000, Tesla T4 | Turing | sm_75 |
| RTX 3000, A100 | Ampere | sm_80, sm_86 |
| RTX 4000, L40 | Ada Lovelace | sm_89 |
| H100 | Hopper | sm_90 |

## OptiX Version / Driver Requirements

| OptiX Version | Minimum Driver |
|--------------|----------------|
| 7.6.0 | 522.25+ |
| 7.7.0 | 530.41+ |
| 8.0.0 | 535+ |
| 8.1.0 | 535+ |
| 9.0.0 | 560+ |
| 9.1.0 | 590+ |

## Troubleshooting

### "Unsupported ABI version" error
Your driver is too old for the OptiX version. Either:
- Update your NVIDIA driver, or
- Build with an older OptiX version: `OPTIX_VERSION=7.6.0 conda build conda-recipe`

### "Invalid target architecture" error
The PTX was compiled for a different GPU. Rebuild with your GPU's architecture:
```bash
GPU_ARCH=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | tr -d '.')
conda build conda-recipe
```
