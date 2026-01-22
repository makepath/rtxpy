#!/bin/bash
set -ex

echo "=== RTXpy Conda Build ==="

# ---------------------------------------------------------------------------
# Step 1: Install OptiX SDK headers
# ---------------------------------------------------------------------------
OPTIX_VERSION="${OPTIX_VERSION:-7.7.0}"
OPTIX_DIR="${SRC_DIR}/optix-sdk"

echo "=== Installing OptiX SDK headers (v${OPTIX_VERSION}) ==="
git clone --depth 1 --branch "v${OPTIX_VERSION}" \
    https://github.com/NVIDIA/optix-dev.git "${OPTIX_DIR}"

if [ ! -f "${OPTIX_DIR}/include/optix.h" ]; then
    echo "ERROR: OptiX headers not found after clone"
    exit 1
fi

export OptiX_INSTALL_DIR="${OPTIX_DIR}"
echo "OptiX headers installed at: ${OptiX_INSTALL_DIR}"

# ---------------------------------------------------------------------------
# Step 2: Detect GPU architecture and compile PTX
# ---------------------------------------------------------------------------
echo "=== Compiling PTX kernel ==="

# Try to detect GPU architecture, fall back to a compatible default
if command -v nvidia-smi &> /dev/null; then
    GPU_ARCH_DETECTED=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d '.')
    if [ -n "${GPU_ARCH_DETECTED}" ]; then
        GPU_ARCH="${GPU_ARCH:-${GPU_ARCH_DETECTED}}"
    fi
fi

# Default to sm_75 (Turing) - minimum supported by CUDA 12+
# PTX is forward-compatible, so this will JIT-compile on newer GPUs
GPU_ARCH="${GPU_ARCH:-75}"

echo "Target GPU architecture: sm_${GPU_ARCH}"

nvcc -ptx \
    -arch="sm_${GPU_ARCH}" \
    -I"${OptiX_INSTALL_DIR}/include" \
    -I"${SRC_DIR}/cuda" \
    --use_fast_math \
    -o "${SRC_DIR}/rtxpy/kernel.ptx" \
    "${SRC_DIR}/cuda/kernel.cu"

echo "PTX compiled successfully:"
head -15 "${SRC_DIR}/rtxpy/kernel.ptx"

# ---------------------------------------------------------------------------
# Step 3: Install otk-pyoptix from source
# ---------------------------------------------------------------------------
echo "=== Installing otk-pyoptix ==="
OTK_PYOPTIX_DIR="${SRC_DIR}/otk-pyoptix"

git clone --depth 1 https://github.com/NVIDIA/otk-pyoptix.git "${OTK_PYOPTIX_DIR}"
cd "${OTK_PYOPTIX_DIR}/optix"

# Install otk-pyoptix
${PYTHON} -m pip install . --no-deps --no-build-isolation -vv

# ---------------------------------------------------------------------------
# Step 4: Install rtxpy
# ---------------------------------------------------------------------------
echo "=== Installing rtxpy ==="
cd "${SRC_DIR}"

${PYTHON} -m pip install . --no-deps --no-build-isolation -vv

echo "=== RTXpy build complete ==="
