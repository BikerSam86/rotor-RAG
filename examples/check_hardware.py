"""
Quick hardware check with psutil - let's see what beast we're running on!
"""

import sys
import io

# Fix Windows encoding
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import psutil
import platform

print("=" * 70)
print("🖥️  HARDWARE SPECS")
print("=" * 70)

# CPU Info
print("\n📊 CPU:")
print(f"  Processor: {platform.processor()}")
print(f"  Physical cores: {psutil.cpu_count(logical=False)}")
print(f"  Logical cores: {psutil.cpu_count(logical=True)}")
cpu_freq = psutil.cpu_freq()
if cpu_freq:
    print(f"  Base frequency: {cpu_freq.current:.2f} MHz")
    print(f"  Max frequency: {cpu_freq.max:.2f} MHz")

# RAM Info
print("\n💾 Memory:")
mem = psutil.virtual_memory()
print(f"  Total RAM: {mem.total / (1024**3):.2f} GB")
print(f"  Available: {mem.available / (1024**3):.2f} GB")
print(f"  Used: {mem.used / (1024**3):.2f} GB ({mem.percent}%)")

# Disk Info
print("\n💿 Disk:")
disk = psutil.disk_usage('C:\\')
print(f"  Total: {disk.total / (1024**3):.2f} GB")
print(f"  Used: {disk.used / (1024**3):.2f} GB ({disk.percent}%)")
print(f"  Free: {disk.free / (1024**3):.2f} GB")

# Platform
print("\n🖥️  Platform:")
print(f"  OS: {platform.system()} {platform.release()}")
print(f"  Version: {platform.version()}")
print(f"  Architecture: {platform.machine()}")
print(f"  Python: {platform.python_version()}")

print("\n" + "=" * 70)
print("🎉 This machine just loaded a 2.4B parameter model!")
print("=" * 70)
