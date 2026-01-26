@echo off
setlocal enabledelayedexpansion

:: GPU Test Script for Windows
:: Equivalent to .github/workflows/gpu-test.yml

echo ============================================
echo RTXpy GPU Test - Windows Local Runner
echo ============================================
echo.

:: Configuration
set OPTIX_DIR=C:\optix
set OPTIX_VERSION=v7.7.0

:: Get the directory where this script is located
set SCRIPT_DIR=%~dp0
cd /d "%SCRIPT_DIR%"

:: Step 1: Verify GPU
echo [1/9] Verifying NVIDIA GPU...
echo ----------------------------------------
nvidia-smi >nul 2>&1
if errorlevel 1 (
    echo ERROR: nvidia-smi not found. Please install NVIDIA drivers.
    exit /b 1
)
nvidia-smi
echo.
echo OptiX 7.7 requires driver 530.41+
echo OptiX 8.0 requires driver 535+
echo OptiX 9.1 requires driver 590+
echo.

:: Step 2: Verify CUDA Toolkit
echo [2/9] Verifying CUDA Toolkit...
echo ----------------------------------------
nvcc --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: nvcc not found. Please install CUDA Toolkit 12.6+
    echo Download from: https://developer.nvidia.com/cuda-downloads
    exit /b 1
)
nvcc --version
echo.

:: Step 3: Verify CMake
echo [3/9] Verifying CMake...
echo ----------------------------------------
cmake --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: cmake not found. Please install CMake.
    echo Download from: https://cmake.org/download/
    echo Or run: conda install -c conda-forge cmake
    exit /b 1
)
cmake --version
echo.

:: Step 4: Install OptiX SDK headers
echo [4/9] Setting up OptiX SDK headers...
echo ----------------------------------------
if exist "%OPTIX_DIR%\include\optix.h" (
    echo OptiX headers already installed at %OPTIX_DIR%
) else (
    echo Cloning OptiX SDK headers from NVIDIA/optix-dev...
    if exist "%OPTIX_DIR%" rmdir /s /q "%OPTIX_DIR%"
    git clone --depth 1 --branch %OPTIX_VERSION% https://github.com/NVIDIA/optix-dev.git "%OPTIX_DIR%"
    if errorlevel 1 (
        echo ERROR: Failed to clone OptiX headers
        exit /b 1
    )
    if not exist "%OPTIX_DIR%\include\optix.h" (
        echo ERROR: OptiX headers not found after clone
        exit /b 1
    )
    echo OptiX headers installed successfully
)
set OptiX_INSTALL_DIR=%OPTIX_DIR%
echo.

:: Step 5: Detect GPU architecture
echo [5/9] Detecting GPU architecture...
echo ----------------------------------------
for /f "tokens=*" %%i in ('nvidia-smi --query-gpu=compute_cap --format=csv,noheader') do set COMPUTE_CAP=%%i
:: Remove the dot from compute capability (e.g., 8.6 -> 86)
set GPU_ARCH=%COMPUTE_CAP:.=%
echo Detected GPU compute capability: sm_%GPU_ARCH%
echo.

:: Step 6: Compile PTX
echo [6/9] Compiling kernel.cu to PTX...
echo ----------------------------------------
if not exist "cuda\kernel.cu" (
    echo ERROR: cuda\kernel.cu not found. Are you in the rtxpy directory?
    exit /b 1
)
nvcc -ptx -arch=sm_%GPU_ARCH% -I"%OptiX_INSTALL_DIR%\include" -Icuda --use_fast_math -o rtxpy\kernel.ptx cuda\kernel.cu
if errorlevel 1 (
    echo ERROR: PTX compilation failed
    exit /b 1
)
echo PTX compiled successfully to rtxpy\kernel.ptx
echo.

:: Step 7: Install otk-pyoptix
echo [7/9] Installing otk-pyoptix...
echo ----------------------------------------
python -c "import optix" >nul 2>&1
if errorlevel 1 (
    echo otk-pyoptix not found, installing from source...
    if exist "%TEMP%\otk-pyoptix" rmdir /s /q "%TEMP%\otk-pyoptix"
    git clone --depth 1 https://github.com/NVIDIA/otk-pyoptix.git "%TEMP%\otk-pyoptix"
    if errorlevel 1 (
        echo ERROR: Failed to clone otk-pyoptix
        exit /b 1
    )
    pushd "%TEMP%\otk-pyoptix\optix"
    pip install .
    if errorlevel 1 (
        echo ERROR: Failed to install otk-pyoptix
        popd
        exit /b 1
    )
    popd
    echo otk-pyoptix installed successfully
) else (
    echo otk-pyoptix already installed
)
echo.

:: Step 8: Install rtxpy
echo [8/9] Installing rtxpy with test dependencies...
echo ----------------------------------------
pip install -U pip
pip install -ve .[tests,cuda12]
if errorlevel 1 (
    echo ERROR: Failed to install rtxpy
    exit /b 1
)
echo.

:: Step 9: Run tests
echo [9/9] Running GPU tests...
echo ----------------------------------------
echo.
echo === Running pytest ===
python -m pytest -v rtxpy/tests
set PYTEST_RESULT=%errorlevel%
echo.

echo === Running basic ray tracing test ===
python -c "from rtxpy import RTX; import numpy as np; verts = np.float32([0,0,0, 1,0,0, 0,1,0, 1,1,0]); triangles = np.int32([0,1,2, 2,1,3]); rays = np.float32([0.33,0.33,100, 0,0,0, -1,1000]); hits = np.float32([0,0,0,0]); optix = RTX(); res = optix.build(0, verts, triangles); assert res == 0, f'Build failed with {res}'; res = optix.trace(rays, hits, 1); assert res == 0, f'Trace failed with {res}'; print(f'Hit result: t={hits[0]}, normal=({hits[1]}, {hits[2]}, {hits[3]})'); assert hits[0] > 0, 'Expected a hit'; print('GPU ray tracing test PASSED!')"
set RAYTEST_RESULT=%errorlevel%
echo.

:: Summary
echo ============================================
echo Test Summary
echo ============================================
if %PYTEST_RESULT% equ 0 (
    echo pytest:           PASSED
) else (
    echo pytest:           FAILED
)
if %RAYTEST_RESULT% equ 0 (
    echo ray tracing test: PASSED
) else (
    echo ray tracing test: FAILED
)
echo ============================================

if %PYTEST_RESULT% neq 0 exit /b %PYTEST_RESULT%
if %RAYTEST_RESULT% neq 0 exit /b %RAYTEST_RESULT%

echo.
echo All GPU tests completed successfully!
exit /b 0
