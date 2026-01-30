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

:: Verify C++ compiler is available (conda-build should set up VS environment)
where cl >nul 2>&1
if errorlevel 1 (
    echo.
    echo ERROR: C++ compiler ^(cl.exe^) not found.
    echo.
    echo Please ensure Visual Studio Build Tools are installed and activated.
    echo You can install them from: https://visualstudio.microsoft.com/visual-cpp-build-tools/
    echo.
    echo If already installed, run this build from a "Developer Command Prompt"
    echo or run vcvars64.bat before building.
    echo.
    exit /b 1
)
echo Found C++ compiler at:
where cl

:: Pre-clone pybind11 without submodules to avoid FetchContent submodule update failures
echo Pre-cloning pybind11 to avoid submodule issues...
set "PYBIND11_DIR=%SRC_DIR%\pybind11-src"
git clone --depth 1 --branch v2.13.6 https://github.com/pybind/pybind11.git "%PYBIND11_DIR%"
if errorlevel 1 (
    echo ERROR: Failed to clone pybind11
    exit /b 1
)

:: Tell CMake to use our pre-cloned pybind11 instead of fetching
set "FETCHCONTENT_SOURCE_DIR_PYBIND11=%PYBIND11_DIR%"
echo Using pre-cloned pybind11 at %PYBIND11_DIR%

cd /d "%OTK_PYOPTIX_DIR%\optix"

:: Patch CMakeLists.txt to use our pre-cloned pybind11 and skip submodule updates
echo Patching CMakeLists.txt to use local pybind11...

:: Convert backslashes to forward slashes for CMake
set "PYBIND11_DIR_CMAKE=%PYBIND11_DIR:\=/%"

:: Prepend the FETCHCONTENT_SOURCE_DIR_PYBIND11 setting to CMakeLists.txt
(
    echo set^(FETCHCONTENT_SOURCE_DIR_PYBIND11 "%PYBIND11_DIR_CMAKE%" CACHE PATH "pybind11 source" FORCE^)
    type CMakeLists.txt
) > "%SRC_DIR%\CMakeLists_new.txt"
move /y "%SRC_DIR%\CMakeLists_new.txt" CMakeLists.txt >nul

echo Patched CMakeLists.txt - first 2 lines:
powershell -Command "Get-Content CMakeLists.txt -Head 2"

:: Build otk-pyoptix using direct cmake + nmake (avoids conda-build injecting -A x64)
echo Building otk-pyoptix with CMake + NMake...

:: Create build directory
mkdir build 2>nul
cd build

:: Get Python include and library paths
for /f "delims=" %%i in ('"%PYTHON%" -c "import sysconfig; print(sysconfig.get_path('include'))"') do set "PYTHON_INCLUDE=%%i"
for /f "delims=" %%i in ('"%PYTHON%" -c "import sysconfig; print(sysconfig.get_config_var('LIBDIR') or sysconfig.get_path('stdlib'))"') do set "PYTHON_LIBDIR=%%i"
for /f "delims=" %%i in ('"%PYTHON%" -c "import sys; print(f'python{sys.version_info.major}{sys.version_info.minor}.lib')"') do set "PYTHON_LIB=%%i"

echo Python include: %PYTHON_INCLUDE%
echo Python libdir: %PYTHON_LIBDIR%

:: Configure with NMake Makefiles generator - explicit command line avoids env var issues
cmake -G "NMake Makefiles" ^
    -DCMAKE_BUILD_TYPE=Release ^
    -DCMAKE_C_COMPILER=cl ^
    -DCMAKE_CXX_COMPILER=cl ^
    -DPython_EXECUTABLE="%PYTHON%" ^
    -DPython_INCLUDE_DIR="%PYTHON_INCLUDE%" ^
    -DOptiX_INSTALL_DIR="%OptiX_INSTALL_DIR%" ^
    -DFETCHCONTENT_SOURCE_DIR_PYBIND11="%PYBIND11_DIR%" ^
    ..
if errorlevel 1 (
    echo ERROR: CMake configure failed
    exit /b 1
)

:: Build with nmake
nmake
if errorlevel 1 (
    echo ERROR: NMake build failed
    exit /b 1
)

:: Install the built module to site-packages
echo Installing optix module...
for /f "delims=" %%i in ('"%PYTHON%" -c "import site; print(site.getsitepackages()[0])"') do set SITE_PACKAGES=%%i
echo Site-packages: %SITE_PACKAGES%

:: Copy built files
copy /Y optix*.pyd "%SITE_PACKAGES%\" 2>nul
copy /Y *.pyd "%SITE_PACKAGES%\" 2>nul
if exist "..\optix\__init__.py" (
    mkdir "%SITE_PACKAGES%\optix" 2>nul
    xcopy /E /Y "..\optix\*" "%SITE_PACKAGES%\optix\"
)

:: Return to otk-pyoptix\optix directory
cd ..

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
