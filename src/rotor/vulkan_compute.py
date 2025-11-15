"""
Vulkan compute acceleration for ternary operations.
Works on Intel HD Graphics 615 and Steam Deck RDNA 2.
"""

import numpy as np
import vulkan as vk
from pathlib import Path
import struct
import subprocess
import os

class VulkanCompute:
    """Vulkan compute pipeline for ternary operations."""

    def __init__(self):
        """Initialize Vulkan compute."""
        self.instance = None
        self.physical_device = None
        self.device = None
        self.queue = None
        self.compute_queue_family = None

        self._init_vulkan()

    def _init_vulkan(self):
        """Initialize Vulkan instance and device."""
        # Create instance
        app_info = vk.VkApplicationInfo(
            pApplicationName="Rotor Ternary Compute",
            applicationVersion=vk.VK_MAKE_VERSION(1, 0, 0),
            pEngineName="Rotor",
            engineVersion=vk.VK_MAKE_VERSION(1, 0, 0),
            apiVersion=vk.VK_API_VERSION_1_0
        )

        create_info = vk.VkInstanceCreateInfo(
            pApplicationInfo=app_info
        )

        self.instance = vk.vkCreateInstance(create_info, None)

        # Get physical devices
        physical_devices = vk.vkEnumeratePhysicalDevices(self.instance)

        if not physical_devices:
            raise RuntimeError("No Vulkan-capable devices found!")

        # Pick first device (usually integrated GPU on laptops)
        self.physical_device = physical_devices[0]

        # Get device properties
        props = vk.vkGetPhysicalDeviceProperties(self.physical_device)
        device_name = props.deviceName if isinstance(props.deviceName, str) else props.deviceName.decode('utf-8')

        print(f"[Vulkan] Using device: {device_name}")
        print(f"  API Version: {vk.VK_VERSION_MAJOR(props.apiVersion)}.{vk.VK_VERSION_MINOR(props.apiVersion)}.{vk.VK_VERSION_PATCH(props.apiVersion)}")
        print(f"  Max Compute Shared Memory: {props.limits.maxComputeSharedMemorySize} bytes")
        print(f"  Max Work Group Size: {props.limits.maxComputeWorkGroupSize}")

        # Find compute queue family
        queue_families = vk.vkGetPhysicalDeviceQueueFamilyProperties(self.physical_device)

        for i, family in enumerate(queue_families):
            if family.queueFlags & vk.VK_QUEUE_COMPUTE_BIT:
                self.compute_queue_family = i
                print(f"  Compute Queue Family: {i}")
                break

        if self.compute_queue_family is None:
            raise RuntimeError("No compute queue family found!")

        # Create logical device
        queue_create_info = vk.VkDeviceQueueCreateInfo(
            queueFamilyIndex=self.compute_queue_family,
            queueCount=1,
            pQueuePriorities=[1.0]
        )

        device_create_info = vk.VkDeviceCreateInfo(
            queueCreateInfoCount=1,
            pQueueCreateInfos=[queue_create_info]
        )

        self.device = vk.vkCreateDevice(self.physical_device, device_create_info, None)

        # Get compute queue
        self.queue = vk.vkGetDeviceQueue(self.device, self.compute_queue_family, 0)

        print("[Vulkan] Device initialized successfully!")

    def compile_shader(self, shader_path: Path) -> bytes:
        """
        Compile GLSL shader to SPIR-V.

        Args:
            shader_path: Path to .comp shader file

        Returns:
            SPIR-V bytecode
        """
        spv_path = shader_path.with_suffix('.spv')

        # Check if SPIR-V already exists and is newer
        if spv_path.exists() and spv_path.stat().st_mtime > shader_path.stat().st_mtime:
            print(f"[Vulkan] Using cached SPIR-V: {spv_path}")
            return spv_path.read_bytes()

        # Try to compile with glslc (from Vulkan SDK)
        glslc = "glslc"
        try:
            result = subprocess.run(
                [glslc, str(shader_path), "-o", str(spv_path)],
                capture_output=True,
                text=True,
                check=True
            )
            print(f"[Vulkan] Compiled shader: {shader_path} -> {spv_path}")
            return spv_path.read_bytes()
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            print(f"[Vulkan] Warning: glslc not found or failed")
            print(f"  Please install Vulkan SDK or compile manually:")
            print(f"  glslc {shader_path} -o {spv_path}")
            raise RuntimeError("Shader compilation failed. Please compile manually with glslc.")

    def create_buffer(self, size: int, usage: int) -> tuple:
        """
        Create a Vulkan buffer.

        Args:
            size: Buffer size in bytes
            usage: Buffer usage flags

        Returns:
            (buffer, memory) tuple
        """
        buffer_info = vk.VkBufferCreateInfo(
            size=size,
            usage=usage,
            sharingMode=vk.VK_SHARING_MODE_EXCLUSIVE
        )

        buffer = vk.vkCreateBuffer(self.device, buffer_info, None)

        # Get memory requirements
        mem_reqs = vk.vkGetBufferMemoryRequirements(self.device, buffer)

        # Find suitable memory type
        mem_props = vk.vkGetPhysicalDeviceMemoryProperties(self.physical_device)

        memory_type_index = None
        for i in range(mem_props.memoryTypeCount):
            if (mem_reqs.memoryTypeBits & (1 << i)) and \
               (mem_props.memoryTypes[i].propertyFlags &
                (vk.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | vk.VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)):
                memory_type_index = i
                break

        if memory_type_index is None:
            raise RuntimeError("No suitable memory type found!")

        # Allocate memory
        alloc_info = vk.VkMemoryAllocateInfo(
            allocationSize=mem_reqs.size,
            memoryTypeIndex=memory_type_index
        )

        memory = vk.vkAllocateMemory(self.device, alloc_info, None)

        # Bind buffer to memory
        vk.vkBindBufferMemory(self.device, buffer, memory, 0)

        return buffer, memory

    def pack_ternary_weights_uint32(self, weights: np.ndarray) -> np.ndarray:
        """
        Pack ternary weights into uint32 format (16 values per uint32).

        Args:
            weights: Float array with values in {-1, 0, +1}

        Returns:
            Packed uint32 array
        """
        # Convert to encoding: -1->0, 0->1, +1->2
        encoded = (weights + 1).astype(np.uint32).flatten()

        # Pack 16 values per uint32
        num_packed = (len(encoded) + 15) // 16
        packed = np.zeros(num_packed, dtype=np.uint32)

        for i in range(len(encoded)):
            packed_idx = i // 16
            bit_offset = (i % 16) * 2
            packed[packed_idx] |= (encoded[i] << bit_offset)

        return packed

    def __del__(self):
        """Cleanup Vulkan resources."""
        if self.device:
            vk.vkDeviceWaitIdle(self.device)
            vk.vkDestroyDevice(self.device, None)
        if self.instance:
            vk.vkDestroyInstance(self.instance, None)


def test_vulkan_basic():
    """Test basic Vulkan initialization."""
    print("=" * 70)
    print("VULKAN COMPUTE TEST")
    print("=" * 70)

    try:
        compute = VulkanCompute()
        print("\n[OK] Vulkan compute initialized successfully!")

        # Try to compile shader
        shader_path = Path(__file__).parent / "shaders" / "ternary_matmul.comp"

        if shader_path.exists():
            print(f"\n[Shader] Found: {shader_path}")
            try:
                spv = compute.compile_shader(shader_path)
                print(f"[OK] Shader compiled! SPIR-V size: {len(spv)} bytes")
            except RuntimeError as e:
                print(f"[Info] {e}")
                print("\nTo compile manually:")
                print(f"  1. Install Vulkan SDK from https://vulkan.lunarg.com/")
                print(f"  2. Run: glslc {shader_path} -o {shader_path.with_suffix('.spv')}")
        else:
            print(f"\n[Warning] Shader not found: {shader_path}")

        print("\n" + "=" * 70)
        print("All ways, always!")
        print("=" * 70)

    except Exception as e:
        print(f"\n[Error] {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_vulkan_basic()
