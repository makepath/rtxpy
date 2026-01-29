@echo off
setlocal EnableDelayedExpansion

:: RTXpy Conda Build Script for Windows

echo ============================================
echo RTXpy Conda Build - Windows
echo ============================================
echo.

:: ---------------------------------------------------------------------------
:: Step 1: Install OptiX SDK headers
:: ---------------------------------------------------------------------------
echo [1/4] Installing OptiX SDK headers...
echo ----------------------------------------

if "%OPTIX_VERSION%"=="" set OPTIX_VERSION=7.7.0
set OPTIX_DIR=%SRC_DIR%\optix-sdk

echo Cloning OptiX SDK headers (v%OPTIX_VERSION%)...
git clone --depth 1 --branch "v%OPTIX_VERSION%" https://github.com/NVIDIA/optix-dev.git "%OPTIX_DIR%"
if errorlevel 1 (
    echo ERROR: Failed to clone OptiX headers
    exit /b 1
)

if not exist "%OPTIX_DIR%\include\optix.h" (
    echo ERROR: OptiX headers not found after clone
    exit /b 1
)

set OptiX_INSTALL_DIR=%OPTIX_DIR%
echo OptiX headers installed at: %OptiX_INSTALL_DIR%
echo.

:: ---------------------------------------------------------------------------
:: Step 2: Detect GPU architecture and compile PTX
:: ---------------------------------------------------------------------------
echo [2/4] Compiling PTX kernel...
echo ----------------------------------------

:: Try to detect GPU architecture, fall back to a compatible default
if "%GPU_ARCH%"=="" (
    echo Detecting GPU architecture...
    for /f "skip=1 tokens=*" %%a in ('nvidia-smi --query-gpu^=compute_cap --format^=csv 2^>nul') do (
        if not defined GPU_ARCH_RAW (
            set "GPU_ARCH_RAW=%%a"
            set "GPU_ARCH=!GPU_ARCH_RAW:.=!"
        )
    )
)

:: Default to sm_75 (Turing) - minimum supported by CUDA 12+
:: PTX is forward-compatible, so this will JIT-compile on newer GPUs
if "%GPU_ARCH%"=="" (
    echo No GPU detected, using default architecture
    set GPU_ARCH=75
)

echo Target GPU architecture: sm_%GPU_ARCH%

nvcc -ptx ^
    -arch="sm_%GPU_ARCH%" ^
    -I"%OptiX_INSTALL_DIR%\include" ^
    -I"%SRC_DIR%\cuda" ^
    --use_fast_math ^
    --allow-unsupported-compiler ^
    -o "%SRC_DIR%\rtxpy\kernel.ptx" ^
    "%SRC_DIR%\cuda\kernel.cu"
if errorlevel 1 (
    echo ERROR: PTX compilation failed
    exit /b 1
)

echo PTX compiled successfully to rtxpy\kernel.ptx
echo.

:: ---------------------------------------------------------------------------
:: Step 3: Install otk-pyoptix from source
:: ---------------------------------------------------------------------------
echo [3/4] Installing otk-pyoptix...
echo ----------------------------------------

set OTK_PYOPTIX_DIR=%SRC_DIR%\otk-pyoptix

echo Cloning otk-pyoptix repository...
git clone --depth 1 https://github.com/NVIDIA/otk-pyoptix.git "%OTK_PYOPTIX_DIR%"
if errorlevel 1 (
    echo ERROR: Failed to clone otk-pyoptix
    exit /b 1
)

:: Verify cmake is available (installed via conda)
where cmake >nul 2>&1
if errorlevel 1 (
    echo ERROR: cmake not found. Ensure cmake is in build requirements.
    exit /b 1
)
echo Found cmake at:
where cmake

:: Set up Visual Studio environment if not already set
:: This finds and activates the Visual Studio Build Tools
if not defined VSINSTALLDIR (
    echo Setting up Visual Studio environment...
    set "VCVARS_FOUND="

    if exist "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat" (
        echo Found VS 18 Community
        call "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat"
        set "VCVARS_FOUND=1"
    )
    if not defined VCVARS_FOUND if exist "C:\Program Files\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" (
        echo Found VS 2022 BuildTools
        call "C:\Program Files\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
        set "VCVARS_FOUND=1"
    )
    if not defined VCVARS_FOUND if exist "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" (
        echo Found VS 2022 Community
        call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
        set "VCVARS_FOUND=1"
    )
    if not defined VCVARS_FOUND if exist "C:\Program Files\Microsoft Visual Studio\2022\Professional\VC\Auxiliary\Build\vcvars64.bat" (
        echo Found VS 2022 Professional
        call "C:\Program Files\Microsoft Visual Studio\2022\Professional\VC\Auxiliary\Build\vcvars64.bat"
        set "VCVARS_FOUND=1"
    )
    if not defined VCVARS_FOUND if exist "C:\Program Files\Microsoft Visual Studio\2022\Enterprise\VC\Auxiliary\Build\vcvars64.bat" (
        echo Found VS 2022 Enterprise
        call "C:\Program Files\Microsoft Visual Studio\2022\Enterprise\VC\Auxiliary\Build\vcvars64.bat"
        set "VCVARS_FOUND=1"
    )
    if not defined VCVARS_FOUND if exist "C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\VC\Auxiliary\Build\vcvars64.bat" (
        echo Found VS 2019 BuildTools
        call "C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
        set "VCVARS_FOUND=1"
    )
    if not defined VCVARS_FOUND if exist "C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Auxiliary\Build\vcvars64.bat" (
        echo Found VS 2019 Community
        call "C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Auxiliary\Build\vcvars64.bat"
        set "VCVARS_FOUND=1"
    )
    if not defined VCVARS_FOUND if exist "C:\Program Files (x86)\Microsoft Visual Studio\2019\Professional\VC\Auxiliary\Build\vcvars64.bat" (
        echo Found VS 2019 Professional
        call "C:\Program Files (x86)\Microsoft Visual Studio\2019\Professional\VC\Auxiliary\Build\vcvars64.bat"
        set "VCVARS_FOUND=1"
    )
    if not defined VCVARS_FOUND if exist "C:\Program Files (x86)\Microsoft Visual Studio\2019\Enterprise\VC\Auxiliary\Build\vcvars64.bat" (
        echo Found VS 2019 Enterprise
        call "C:\Program Files (x86)\Microsoft Visual Studio\2019\Enterprise\VC\Auxiliary\Build\vcvars64.bat"
        set "VCVARS_FOUND=1"
    )
    if not defined VCVARS_FOUND (
        echo WARNING: Could not find Visual Studio. Build may fail.
        echo Please ensure Visual Studio 2019/2022 Build Tools are installed.
    )
)

:: Verify C++ compiler is available
where cl >nul 2>&1
if errorlevel 1 (
    echo ERROR: C++ compiler (cl.exe) not found.
    echo Please install Visual Studio Build Tools with C++ workload.
    exit /b 1
)
echo Found C++ compiler at:
where cl

cd /d "%OTK_PYOPTIX_DIR%\optix"
echo Building and installing otk-pyoptix...
"%PYTHON%" -m pip install . --no-deps --no-build-isolation -vv
if errorlevel 1 (
    echo ERROR: Failed to install otk-pyoptix
    exit /b 1
)

echo otk-pyoptix installed successfully
echo.

:: ---------------------------------------------------------------------------
:: Step 4: Install rtxpy
:: ---------------------------------------------------------------------------
echo [4/4] Installing rtxpy...
echo ----------------------------------------

cd /d "%SRC_DIR%"

echo Building and installing rtxpy package...
"%PYTHON%" -m pip install . --no-deps --no-build-isolation -vv
if errorlevel 1 (
    echo ERROR: Failed to install rtxpy
    exit /b 1
)

echo rtxpy installed successfully
echo.

:: ---------------------------------------------------------------------------
:: Build complete
:: ---------------------------------------------------------------------------
echo ============================================
echo RTXpy Conda Build Complete!
echo ============================================
echo.
echo Summary:
echo   OptiX SDK:    v%OPTIX_VERSION%
echo   GPU Target:   sm_%GPU_ARCH%
echo   PTX Output:   rtxpy\kernel.ptx
echo.

exit /b 0
