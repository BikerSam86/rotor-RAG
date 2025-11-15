"""
Minimal Vulkan initialization test.
Tests: Instance creation, device detection, shader loading.
"""

import sys
import io
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

print("=" * 70)
print("VULKAN INITIALIZATION TEST")
print("=" * 70)

try:
    import vulkan as vk
    print("\n[1/5] Vulkan package imported")
    print(f"        Version: {vk.__version__}")
except ImportError as e:
    print(f"\n[ERROR] Failed to import vulkan: {e}")
    sys.exit(1)

# Test 1: Create instance
try:
    app_info = vk.VkApplicationInfo(
        sType=vk.VK_STRUCTURE_TYPE_APPLICATION_INFO,
        pApplicationName="Rotor Test",
        applicationVersion=vk.VK_MAKE_VERSION(1, 0, 0),
        pEngineName="Test",
        engineVersion=vk.VK_MAKE_VERSION(1, 0, 0),
        apiVersion=vk.VK_API_VERSION_1_0  # Use 1.0 for compatibility
    )

    instance_info = vk.VkInstanceCreateInfo(
        sType=vk.VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
        pApplicationInfo=app_info
    )

    instance = vk.vkCreateInstance(instance_info, None)
    print("\n[2/5] Vulkan instance created")
except Exception as e:
    print(f"\n[ERROR] Failed to create instance: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 2: Enumerate devices
try:
    physical_devices = vk.vkEnumeratePhysicalDevices(instance)
    print(f"\n[3/5] Found {len(physical_devices)} Vulkan device(s)")

    for i, device in enumerate(physical_devices):
        props = vk.vkGetPhysicalDeviceProperties(device)
        print(f"        Device {i}: {props.deviceName}")
        print(f"                   Type: {props.deviceType}")
        print(f"                   API Version: {props.apiVersion}")
except Exception as e:
    print(f"\n[ERROR] Failed to enumerate devices: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 3: Check compute queue support
try:
    device = physical_devices[0]
    queue_families = vk.vkGetPhysicalDeviceQueueFamilyProperties(device)

    compute_family = None
    for i, family in enumerate(queue_families):
        if family.queueFlags & vk.VK_QUEUE_COMPUTE_BIT:
            compute_family = i
            break

    if compute_family is not None:
        print(f"\n[4/5] Compute queue family found: {compute_family}")
    else:
        print("\n[ERROR] No compute queue family found!")
        sys.exit(1)
except Exception as e:
    print(f"\n[ERROR] Failed to check queue families: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 4: Load shader
try:
    shader_path = Path(__file__).parent.parent / "src" / "rotor" / "shaders" / "ternary_matmul.spv"

    if not shader_path.exists():
        print(f"\n[ERROR] Shader not found: {shader_path}")
        sys.exit(1)

    with open(shader_path, 'rb') as f:
        shader_code = f.read()

    print(f"\n[5/5] Shader loaded: {shader_path.name}")
    print(f"        Size: {len(shader_code)} bytes")
    print(f"        SPIR-V magic: {hex(int.from_bytes(shader_code[:4], 'little'))}")

    if int.from_bytes(shader_code[:4], 'little') == 0x07230203:
        print("        [OK] Valid SPIR-V magic number!")
    else:
        print("        [WARN] Invalid SPIR-V magic number")

except Exception as e:
    print(f"\n[ERROR] Failed to load shader: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Cleanup
vk.vkDestroyInstance(instance, None)

print("\n" + "=" * 70)
print("SUCCESS - All initialization tests passed!")
print("=" * 70)
print("\nVulkan is ready for compute operations.")
print("Next step: Implement buffer allocation and dispatch.")
print("\n" + "=" * 70)
