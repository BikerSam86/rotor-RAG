#!/usr/bin/env python3
"""
Build script for native C/CUDA libraries.
"""

import subprocess
import sys
import io
import os
import platform
from pathlib import Path

# Fix Windows encoding
if hasattr(sys.stdout, 'buffer') and not isinstance(sys.stdout, io.TextIOWrapper):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')


def build_c_library():
    """Build C library with SIMD support."""
    print("=" * 60)
    print("Building C Library with SIMD")
    print("=" * 60)

    native_dir = Path(__file__).parent
    build_dir = native_dir / "build"
    build_dir.mkdir(exist_ok=True)

    c_files = [
        native_dir / "c" / "rotor_core.c"
    ]

    include_dir = native_dir / "include"

    # Detect platform and compiler
    if sys.platform == "win32":
        # Try to use MSVC (cl) first, then fall back to clang/gcc
        compiler = "cl"
        output = build_dir / "rotor_core.dll"
        compile_flags = [
            "/LD",  # Create DLL
            "/O2",  # Optimize for speed
            "/arch:AVX2",  # Enable AVX2
            f"/I{include_dir}",
            f"/Fe{output}",  # Output file
            "/link", "/INCREMENTAL:NO"  # Linker options
        ]
    elif sys.platform == "darwin":
        compiler = "clang"
        output = build_dir / "librotor.dylib"
        compile_flags = [
            "-shared",
            "-O3",
            "-march=native",  # Use best available SIMD
            "-fPIC",
            f"-I{include_dir}",
            "-o", str(output)
        ]
    else:  # Linux
        compiler = "gcc"
        output = build_dir / "librotor.so"
        compile_flags = [
            "-shared",
            "-O3",
            "-mavx2",
            "-fPIC",
            f"-I{include_dir}",
            "-o", str(output)
        ]

    # Build command
    cmd = [compiler] + compile_flags + [str(f) for f in c_files]

    print(f"\nCompiling with: {' '.join(cmd)}\n")

    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(f"✓ C library built successfully: {output}")
        if result.stdout:
            print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Build failed!")
        print(f"Error: {e.stderr}")
        return False
    except FileNotFoundError:
        print(f"✗ Compiler '{compiler}' not found!")
        print("Please install a C compiler:")
        if sys.platform == "win32":
            print("  - Install Visual Studio or clang")
        elif sys.platform == "darwin":
            print("  - Install Xcode Command Line Tools: xcode-select --install")
        else:
            print("  - Install GCC: sudo apt-get install build-essential")
        return False


def build_cuda_library():
    """Build CUDA library (optional)."""
    print("\n" + "=" * 60)
    print("Building CUDA Library (Optional)")
    print("=" * 60)

    # Check if nvcc is available
    try:
        subprocess.run(["nvcc", "--version"], check=True, capture_output=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("✗ CUDA compiler (nvcc) not found. Skipping CUDA build.")
        print("  To enable CUDA support, install CUDA Toolkit from:")
        print("  https://developer.nvidia.com/cuda-downloads")
        return False

    native_dir = Path(__file__).parent
    build_dir = native_dir / "build"
    build_dir.mkdir(exist_ok=True)

    cu_files = [
        native_dir / "cuda" / "rotor_cuda.cu"
    ]

    include_dir = native_dir / "include"

    if sys.platform == "win32":
        output = build_dir / "librotor_cuda.dll"
        compile_flags = [
            "--shared",
            "-O3",
            f"-I{include_dir}",
            "-o", str(output)
        ]
    else:
        output = build_dir / "librotor_cuda.so"
        compile_flags = [
            "--shared",
            "-O3",
            "-Xcompiler", "-fPIC",
            f"-I{include_dir}",
            "-o", str(output)
        ]

    # Build command
    cmd = ["nvcc"] + compile_flags + [str(f) for f in cu_files]

    print(f"\nCompiling with: {' '.join(cmd)}\n")

    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(f"✓ CUDA library built successfully: {output}")
        if result.stdout:
            print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ CUDA build failed!")
        print(f"Error: {e.stderr}")
        return False


def main():
    """Main build function."""
    print("\nRotor Native Build Script")
    print("=" * 60)

    c_success = build_c_library()

    cuda_success = False
    if "--with-cuda" in sys.argv or "--cuda" in sys.argv:
        cuda_success = build_cuda_library()

    print("\n" + "=" * 60)
    print("Build Summary")
    print("=" * 60)
    print(f"C Library:    {'✓ Success' if c_success else '✗ Failed'}")
    print(f"CUDA Library: {'✓ Success' if cuda_success else '- Skipped' if not cuda_success else '✗ Failed'}")
    print("=" * 60)

    if c_success:
        print("\n✓ Native libraries ready!")
        print("\nTo use in Python:")
        print("  from rotor.native import get_backend")
        print("  backend = get_backend()")
        print("  result = backend.dot(a_bit0, a_bit1, b_bit0, b_bit1)")
    else:
        print("\n✗ Build failed. Python will fall back to NumPy implementation.")

    return 0 if c_success else 1


if __name__ == "__main__":
    sys.exit(main())
