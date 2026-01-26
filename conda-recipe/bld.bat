@echo off
setlocal EnableDelayedExpansion

echo === RTXpy Conda Build (Windows) ===

REM ---------------------------------------------------------------------------
REM Step 1: Install OptiX SDK headers
REM ---------------------------------------------------------------------------
if "%OPTIX_VERSION%"=="" set OPTIX_VERSION=7.7.0
set OPTIX_DIR=%SRC_DIR%\optix-sdk

echo === Installing OptiX SDK headers (v%OPTIX_VERSION%) ===
git clone --depth 1 --branch "v%OPTIX_VERSION%" https://github.com/NVIDIA/optix-dev.git "%OPTIX_DIR%"
if errorlevel 1 exit /b 1

if not exist "%OPTIX_DIR%\include\optix.h" (
    echo ERROR: OptiX headers not found after clone
    exit /b 1
)

set OptiX_INSTALL_DIR=%OPTIX_DIR%
echo OptiX headers installed at: %OptiX_INSTALL_DIR%

REM ---------------------------------------------------------------------------
REM Step 2: Detect GPU architecture and compile PTX
REM ---------------------------------------------------------------------------
echo === Compiling PTX kernel ===

REM Try to detect GPU architecture, fall back to a compatible default
if "%GPU_ARCH%"=="" (
    for /f "tokens=*" %%a in ('nvidia-smi --query-gpu^=compute_cap --format^=csv^,noheader 2^>nul') do (
        set GPU_ARCH_RAW=%%a
        set GPU_ARCH=!GPU_ARCH_RAW:.=!
        goto :arch_found
    )
)
:arch_found

REM Default to sm_75 (Turing) - minimum supported by CUDA 12+
REM PTX is forward-compatible, so this will JIT-compile on newer GPUs
if "%GPU_ARCH%"=="" set GPU_ARCH=75

echo Target GPU architecture: sm_%GPU_ARCH%

nvcc -ptx ^
    -arch="sm_%GPU_ARCH%" ^
    -I"%OptiX_INSTALL_DIR%\include" ^
    -I"%SRC_DIR%\cuda" ^
    --use_fast_math ^
    -o "%SRC_DIR%\rtxpy\kernel.ptx" ^
    "%SRC_DIR%\cuda\kernel.cu"
if errorlevel 1 exit /b 1

echo PTX compiled successfully

REM ---------------------------------------------------------------------------
REM Step 3: Install otk-pyoptix from source
REM ---------------------------------------------------------------------------
echo === Installing otk-pyoptix ===
set OTK_PYOPTIX_DIR=%SRC_DIR%\otk-pyoptix

git clone --depth 1 https://github.com/NVIDIA/otk-pyoptix.git "%OTK_PYOPTIX_DIR%"
if errorlevel 1 exit /b 1

cd /d "%OTK_PYOPTIX_DIR%\optix"
"%PYTHON%" -m pip install . --no-deps --no-build-isolation -vv
if errorlevel 1 exit /b 1

REM ---------------------------------------------------------------------------
REM Step 4: Install rtxpy
REM ---------------------------------------------------------------------------
echo === Installing rtxpy ===
cd /d "%SRC_DIR%"

"%PYTHON%" -m pip install . --no-deps --no-build-isolation -vv
if errorlevel 1 exit /b 1

echo === RTXpy build complete ===
