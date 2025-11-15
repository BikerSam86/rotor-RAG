"""
Quick build script for C library using MSVC.
"""

import subprocess
import sys
from pathlib import Path

def main():
    # Paths
    native_dir = Path(__file__).parent
    build_dir = native_dir / "build"
    build_dir.mkdir(exist_ok=True)

    # Find Visual Studio
    vswhere_path = r"C:\Program Files (x86)\Microsoft Visual Studio\Installer\vswhere.exe"

    try:
        result = subprocess.run(
            [vswhere_path, "-latest", "-products", "*",
             "-requires", "Microsoft.VisualStudio.Component.VC.Tools.x86.x64",
             "-property", "installationPath"],
            capture_output=True,
            text=True,
            check=True
        )
        vs_path = result.stdout.strip()
        print(f"Found Visual Studio at: {vs_path}")
    except Exception as e:
        print(f"ERROR: Could not find Visual Studio: {e}")
        return 1

    # Setup MSVC environment and compile
    vcvarsall = Path(vs_path) / "VC" / "Auxiliary" / "Build" / "vcvarsall.bat"

    # SDK paths
    sdk_version = "10.0.26100.0"
    sdk_base = r"C:\Program Files (x86)\Windows Kits\10"
    sdk_include = f"{sdk_base}\\Include\\{sdk_version}"
    sdk_lib = f"{sdk_base}\\Lib\\{sdk_version}"

    # Build command
    compile_cmd = (
        f'cl.exe /LD /O2 /arch:AVX2 /DROTOR_BUILD_DLL '
        f'/I"include" '
        f'/I"{sdk_include}\\ucrt" '
        f'/I"{sdk_include}\\um" '
        f'/I"{sdk_include}\\shared" '
        f'c\\rotor_core.c '
        f'/Fe"build\\rotor_core.dll" '
        f'/link '
        f'/LIBPATH:"{sdk_lib}\\ucrt\\x64" '
        f'/LIBPATH:"{sdk_lib}\\um\\x64" '
        f'/INCREMENTAL:NO'
    )

    # Full command with environment setup
    full_cmd = f'"{vcvarsall}" x64 && cd /d "{native_dir}" && {compile_cmd}'

    print("\n" + "="*70)
    print("Building Rotor C Library")
    print("="*70)
    print(f"\nCompiling rotor_core.c...")
    print()

    # Run compilation
    result = subprocess.run(
        full_cmd,
        shell=True,
        cwd=str(native_dir)
    )

    if result.returncode == 0:
        print("\n" + "="*70)
        print("SUCCESS! Library built: build\\rotor_core.dll")
        print("="*70)
        return 0
    else:
        print("\n" + "="*70)
        print("BUILD FAILED!")
        print("="*70)
        return 1

if __name__ == "__main__":
    sys.exit(main())
