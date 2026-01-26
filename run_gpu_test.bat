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
echo [1/10] Verifying NVIDIA GPU...
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
echo [2/10] Verifying CUDA Toolkit...
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
echo [3/10] Verifying CMake...
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
echo [4/10] Setting up OptiX SDK headers...
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

:: Step 5: Set up Visual Studio environment for nvcc
echo [5/10] Setting up Visual Studio environment...
echo ----------------------------------------
:: Check if cl.exe is already available
where cl.exe >nul 2>&1
if not errorlevel 1 (
    echo Visual Studio environment already configured
    goto :vs_done
)

:: Use vswhere to find Visual Studio installation (most reliable method)
set "VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
set "VCVARS_PATH="

if exist "%VSWHERE%" (
    echo Using vswhere to locate Visual Studio...
    for /f "usebackq tokens=*" %%i in (`"%VSWHERE%" -latest -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath`) do (
        set "VS_PATH=%%i"
    )
    if defined VS_PATH (
        set "VCVARS_PATH=!VS_PATH!\VC\Auxiliary\Build\vcvars64.bat"
    )
)

:: Fallback: try common paths if vswhere didn't work
if not defined VCVARS_PATH (
    echo vswhere not found or no VS with C++ tools, trying common paths...
    for %%p in (
        "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
        "C:\Program Files\Microsoft Visual Studio\2022\Professional\VC\Auxiliary\Build\vcvars64.bat"
        "C:\Program Files\Microsoft Visual Studio\2022\Enterprise\VC\Auxiliary\Build\vcvars64.bat"
        "C:\Program Files\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
        "C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Auxiliary\Build\vcvars64.bat"
        "C:\Program Files (x86)\Microsoft Visual Studio\2019\Professional\VC\Auxiliary\Build\vcvars64.bat"
        "C:\Program Files (x86)\Microsoft Visual Studio\2019\Enterprise\VC\Auxiliary\Build\vcvars64.bat"
        "C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
    ) do (
        if not defined VCVARS_PATH (
            if exist %%p (
                set "VCVARS_PATH=%%~p"
            )
        )
    )
)

if not defined VCVARS_PATH (
    echo ERROR: Could not find Visual Studio with C++ tools.
    echo.
    echo Please install Visual Studio with C++ build tools:
    echo   winget install Microsoft.VisualStudio.2022.Community --silent --override "--wait --quiet --add Microsoft.VisualStudio.Workload.NativeDesktop --add Microsoft.VisualStudio.Component.VC.Tools.x86.x64 --includeRecommended"
    echo.
    echo Or run this script from "x64 Native Tools Command Prompt for VS 2022"
    exit /b 1
)

echo Found: %VCVARS_PATH%
echo Initializing Visual Studio environment...
call "%VCVARS_PATH%"
if errorlevel 1 (
    echo ERROR: Failed to initialize Visual Studio environment
    exit /b 1
)

:: Verify cl.exe is now available
where cl.exe >nul 2>&1
if errorlevel 1 (
    echo.
    echo ERROR: cl.exe still not found after running vcvars64.bat
    echo The C++ compiler may not be installed. Please ensure you have installed
    echo the "Desktop development with C++" workload in Visual Studio.
    echo.
    echo You can add it by running:
    echo   "%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vs_installer.exe" modify --installPath "!VS_PATH!" --add Microsoft.VisualStudio.Component.VC.Tools.x86.x64
    echo.
    echo Or run this script from "x64 Native Tools Command Prompt for VS 2022"
    exit /b 1
)
echo.
echo Visual Studio environment configured successfully
echo Compiler:
where cl.exe
:vs_done
echo.

:: Step 6: Detect GPU architecture
echo [6/10] Detecting GPU architecture...
echo ----------------------------------------
:: Use skip=1 to skip the CSV header line since noheader may not be supported
for /f "skip=1 tokens=*" %%i in ('nvidia-smi --query-gpu=compute_cap --format=csv') do (
    if not defined COMPUTE_CAP set "COMPUTE_CAP=%%i"
)
:: Remove the dot from compute capability (e.g., 8.6 -> 86)
set GPU_ARCH=%COMPUTE_CAP:.=%
echo Detected GPU compute capability: sm_%GPU_ARCH%
echo.

:: Step 7: Compile PTX
echo [7/10] Compiling kernel.cu to PTX...
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

:: Step 8: Install otk-pyoptix
echo [8/10] Installing otk-pyoptix...
echo ----------------------------------------

:: Verify OptiX_INSTALL_DIR is set and valid
if not defined OptiX_INSTALL_DIR (
    set "OptiX_INSTALL_DIR=%OPTIX_DIR%"
)
if not exist "%OptiX_INSTALL_DIR%\include\optix.h" (
    echo ERROR: OptiX headers not found at %OptiX_INSTALL_DIR%\include
    echo Please ensure step 4 completed successfully.
    exit /b 1
)
echo Using OptiX_INSTALL_DIR=%OptiX_INSTALL_DIR%

python -c "import optix" >nul 2>&1
if errorlevel 1 (
    echo otk-pyoptix not found, installing from source...
    if exist "%TEMP%\otk-pyoptix" rmdir /s /q "%TEMP%\otk-pyoptix"
    git clone --depth 1 https://github.com/NVIDIA/otk-pyoptix.git "%TEMP%\otk-pyoptix"
    if errorlevel 1 (
        echo ERROR: Failed to clone otk-pyoptix
        exit /b 1
    )

    :: Pre-clone pybind11 without submodules to avoid FetchContent submodule update failures
    echo Pre-cloning pybind11 to avoid submodule issues...
    set "PYBIND11_DIR=%TEMP%\pybind11-src"
    if exist "!PYBIND11_DIR!" rmdir /s /q "!PYBIND11_DIR!"
    git clone --depth 1 --branch v2.13.6 https://github.com/pybind/pybind11.git "!PYBIND11_DIR!"
    if errorlevel 1 (
        echo WARNING: Failed to pre-clone pybind11, will try without
    ) else (
        :: Tell CMake to use our pre-cloned pybind11 instead of fetching
        set "FETCHCONTENT_SOURCE_DIR_PYBIND11=!PYBIND11_DIR!"
        echo Using pre-cloned pybind11 at !PYBIND11_DIR!
    )

    pushd "%TEMP%\otk-pyoptix\optix"

    :: Set OptiX path for cmake/pip build process
    set "OptiX_INSTALL_DIR=%OptiX_INSTALL_DIR%"
    set "OPTIX_PATH=%OptiX_INSTALL_DIR%"
    set "CMAKE_PREFIX_PATH=%OptiX_INSTALL_DIR%;%CMAKE_PREFIX_PATH%"

    echo Building with OptiX_INSTALL_DIR=%OptiX_INSTALL_DIR%
    pip install . -v
    if errorlevel 1 (
        echo.
        echo ERROR: Failed to install otk-pyoptix
        echo.
        echo If the error mentions OptiX not found, try setting manually:
        echo   set OptiX_INSTALL_DIR=%OptiX_INSTALL_DIR%
        echo   set OPTIX_PATH=%OptiX_INSTALL_DIR%
        echo.
        popd
        exit /b 1
    )
    popd
    echo otk-pyoptix installed successfully
) else (
    echo otk-pyoptix already installed
)
echo.

:: Step 9: Install rtxpy
echo [9/10] Installing rtxpy with test dependencies...
echo ----------------------------------------
pip install -U pip
pip install -ve .[tests,cuda12]
if errorlevel 1 (
    echo ERROR: Failed to install rtxpy
    exit /b 1
)
echo.

:: Step 10: Run tests
echo [10/10] Running GPU tests...
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
