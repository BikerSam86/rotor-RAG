@echo off
REM Build script for MSVC - automatically finds and uses Visual Studio environment

echo ============================================================
echo Building Rotor C Library with MSVC
echo ============================================================

REM Try to find Visual Studio installation
set "VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"

if exist "%VSWHERE%" (
    echo Found vswhere.exe, locating Visual Studio...
    for /f "usebackq tokens=*" %%i in (`"%VSWHERE%" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath`) do (
        set "VS_PATH=%%i"
    )
)

if not defined VS_PATH (
    echo ERROR: Could not find Visual Studio installation!
    echo Please install Visual Studio Build Tools with C++ support.
    exit /b 1
)

echo Found Visual Studio at: %VS_PATH%

REM Set up MSVC environment
set "VCVARSALL=%VS_PATH%\VC\Auxiliary\Build\vcvarsall.bat"

if not exist "%VCVARSALL%" (
    echo ERROR: vcvarsall.bat not found!
    exit /b 1
)

echo Setting up MSVC environment...
call "%VCVARSALL%" x64

REM Now compile
echo.
echo Compiling rotor_core.c...
cd /d "%~dp0"

REM Add Windows SDK paths explicitly
set "SDK_VERSION=10.0.26100.0"
set "SDK_INCLUDE=%ProgramFiles(x86)%\Windows Kits\10\Include\%SDK_VERSION%"
set "SDK_LIB=%ProgramFiles(x86)%\Windows Kits\10\Lib\%SDK_VERSION%"

cl.exe /LD /O2 /arch:AVX2 /DROTOR_BUILD_DLL /I"include" /I"%SDK_INCLUDE%\ucrt" /I"%SDK_INCLUDE%\um" /I"%SDK_INCLUDE%\shared" c\rotor_core.c /Fe"build\rotor_core.dll" /link /LIBPATH:"%SDK_LIB%\ucrt\x64" /LIBPATH:"%SDK_LIB%\um\x64" /INCREMENTAL:NO

if %ERRORLEVEL% EQU 0 (
    echo.
    echo ============================================================
    echo SUCCESS! Library built: build\rotor_core.dll
    echo ============================================================
) else (
    echo.
    echo ============================================================
    echo BUILD FAILED!
    echo ============================================================
    exit /b 1
)
