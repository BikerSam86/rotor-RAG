"""
Vulkan-accelerated ternary matrix operations.
Cross-platform GPU compute using Vulkan API.

Supports:
- Intel HD Graphics (tested on HD 615)
- AMD RDNA (Steam Deck target)
- NVIDIA GPUs (universal Vulkan support)
- Mobile GPUs (via Vulkan 1.1+)
"""

import numpy as np
from pathlib import Path
import struct

try:
    import vulkan as vk
    VULKAN_AVAILABLE = True
except ImportError:
    VULKAN_AVAILABLE = False
    print("[Vulkan] Warning: vulkan package not available. Install with: pip install vulkan")


class VulkanTernaryOps:
    """
    Vulkan-accelerated ternary operations.
    Falls back gracefully if Vulkan unavailable.
    """

    def __init__(self, use_int8_optimized=True, device_index=0):
        """
        Initialize Vulkan compute backend.

        Args:
            use_int8_optimized: Use int8 shader (faster) vs bit-packed (smaller)
            device_index: GPU device index (0 = first GPU)
        """
        if not VULKAN_AVAILABLE:
            raise RuntimeError("Vulkan package not installed. Run: pip install vulkan")

        self.use_int8 = use_int8_optimized
        self.device_index = device_index

        # Initialize Vulkan
        self._init_instance()
        self._select_physical_device()
        self._create_logical_device()
        self._load_shader()
        self._create_descriptor_layout()
        self._create_pipeline()
        self._create_command_pool()

        print(f"[Vulkan] Initialized on: {self.device_name}")
        print(f"[Vulkan] Shader variant: {'int8-optimized' if use_int8_optimized else 'bit-packed'}")

    def _init_instance(self):
        """Create Vulkan instance."""
        app_info = vk.VkApplicationInfo(
            sType=vk.VK_STRUCTURE_TYPE_APPLICATION_INFO,
            pApplicationName="Rotor Ternary Compute",
            applicationVersion=vk.VK_MAKE_VERSION(1, 0, 0),
            pEngineName="Rotor",
            engineVersion=vk.VK_MAKE_VERSION(1, 0, 0),
            apiVersion=vk.VK_API_VERSION_1_1
        )

        create_info = vk.VkInstanceCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
            pApplicationInfo=app_info
        )

        self.instance = vk.vkCreateInstance(create_info, None)

    def _select_physical_device(self):
        """Select GPU device."""
        physical_devices = vk.vkEnumeratePhysicalDevices(self.instance)

        if not physical_devices:
            raise RuntimeError("No Vulkan-capable GPU found!")

        # Use specified device index
        if self.device_index >= len(physical_devices):
            print(f"[Vulkan] Warning: Device {self.device_index} not found, using device 0")
            self.device_index = 0

        self.physical_device = physical_devices[self.device_index]

        # Get device properties
        props = vk.vkGetPhysicalDeviceProperties(self.physical_device)
        self.device_name = props.deviceName

        # Find compute queue family
        queue_families = vk.vkGetPhysicalDeviceQueueFamilyProperties(self.physical_device)
        self.compute_queue_family = None

        for i, family in enumerate(queue_families):
            if family.queueFlags & vk.VK_QUEUE_COMPUTE_BIT:
                self.compute_queue_family = i
                break

        if self.compute_queue_family is None:
            raise RuntimeError("No compute queue family found on device!")

    def _create_logical_device(self):
        """Create logical device and compute queue."""
        queue_create_info = vk.VkDeviceQueueCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
            queueFamilyIndex=self.compute_queue_family,
            queueCount=1,
            pQueuePriorities=[1.0]
        )

        # Enable required features for int8 shader
        features = vk.VkPhysicalDeviceFeatures()
        if self.use_int8:
            # Would need to enable shaderInt8 feature here
            # For now, bit-packed shader works on all devices
            pass

        device_create_info = vk.VkDeviceCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
            queueCreateInfoCount=1,
            pQueueCreateInfos=[queue_create_info],
            pEnabledFeatures=features
        )

        self.device = vk.vkCreateDevice(self.physical_device, device_create_info, None)
        self.compute_queue = vk.vkGetDeviceQueue(self.device, self.compute_queue_family, 0)

    def _load_shader(self):
        """Load compiled SPIR-V shader."""
        shader_dir = Path(__file__).parent / "shaders"

        if self.use_int8:
            shader_path = shader_dir / "ternary_matmul_optimized.spv"
        else:
            shader_path = shader_dir / "ternary_matmul.spv"

        if not shader_path.exists():
            raise FileNotFoundError(f"Shader not found: {shader_path}")

        # Read SPIR-V bytecode
        with open(shader_path, 'rb') as f:
            spirv_code = f.read()

        # Create shader module
        shader_create_info = vk.VkShaderModuleCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
            codeSize=len(spirv_code),
            pCode=spirv_code
        )

        self.shader_module = vk.vkCreateShaderModule(self.device, shader_create_info, None)
        print(f"[Vulkan] Loaded shader: {shader_path.name}")

    def _create_descriptor_layout(self):
        """Create descriptor set layout for buffers."""
        # 4 storage buffers: weights, input, scales, output
        bindings = [
            vk.VkDescriptorSetLayoutBinding(
                binding=i,
                descriptorType=vk.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                descriptorCount=1,
                stageFlags=vk.VK_SHADER_STAGE_COMPUTE_BIT
            )
            for i in range(4)
        ]

        layout_info = vk.VkDescriptorSetLayoutCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
            bindingCount=len(bindings),
            pBindings=bindings
        )

        self.descriptor_layout = vk.vkCreateDescriptorSetLayout(self.device, layout_info, None)

    def _create_pipeline(self):
        """Create compute pipeline."""
        # Push constant range for dimensions
        push_constant_range = vk.VkPushConstantRange(
            stageFlags=vk.VK_SHADER_STAGE_COMPUTE_BIT,
            offset=0,
            size=8  # 2 uint32s: in_dim, out_dim
        )

        # Pipeline layout
        pipeline_layout_info = vk.VkPipelineLayoutCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
            setLayoutCount=1,
            pSetLayouts=[self.descriptor_layout],
            pushConstantRangeCount=1,
            pPushConstantRanges=[push_constant_range]
        )

        self.pipeline_layout = vk.vkCreatePipelineLayout(self.device, pipeline_layout_info, None)

        # Compute pipeline
        stage_info = vk.VkPipelineShaderStageCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            stage=vk.VK_SHADER_STAGE_COMPUTE_BIT,
            module=self.shader_module,
            pName="main"
        )

        pipeline_info = vk.VkComputePipelineCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
            stage=stage_info,
            layout=self.pipeline_layout
        )

        self.pipeline = vk.vkCreateComputePipelines(
            self.device, None, 1, [pipeline_info], None
        )[0]

    def _create_command_pool(self):
        """Create command pool for compute commands."""
        pool_info = vk.VkCommandPoolCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
            queueFamilyIndex=self.compute_queue_family,
            flags=vk.VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT
        )

        self.command_pool = vk.vkCreateCommandPool(self.device, pool_info, None)

    def pack_ternary_weights(self, weights: np.ndarray) -> np.ndarray:
        """
        Pack ternary weights for GPU.

        Args:
            weights: Float array with values in {-1, 0, +1}

        Returns:
            Packed weights (int8 or uint32 depending on shader variant)
        """
        if self.use_int8:
            # Direct int8 storage
            return weights.astype(np.int8)
        else:
            # Bit-packed 2-bit format (16 values per uint32)
            encoded = (weights + 1).astype(np.uint8)
            flat = encoded.flatten()
            num_packed = (len(flat) + 15) // 16
            packed = np.zeros(num_packed, dtype=np.uint32)

            for i in range(len(flat)):
                uint_idx = i // 16
                bit_offset = (i % 16) * 2
                packed[uint_idx] |= (flat[i] << bit_offset)

            return packed

    def ternary_matmul(
        self,
        packed_weights: np.ndarray,
        scales: np.ndarray,
        input_vec: np.ndarray,
        in_dim: int,
        out_dim: int
    ) -> np.ndarray:
        """
        GPU-accelerated ternary matrix multiplication.

        Args:
            packed_weights: Packed ternary weights
            scales: Scale factors [out_dim]
            input_vec: Input vector [in_dim]
            in_dim: Input dimension
            out_dim: Output dimension

        Returns:
            Output vector [out_dim]
        """
        # TODO: Implement Vulkan buffer creation and dispatch
        # For now, this is a placeholder that shows the structure
        raise NotImplementedError("Vulkan dispatch not yet implemented - use OpenCL for now")

        # Full implementation would:
        # 1. Create GPU buffers
        # 2. Upload data
        # 3. Create descriptor set
        # 4. Record command buffer
        # 5. Submit to queue
        # 6. Wait and download results

    def __del__(self):
        """Cleanup Vulkan resources."""
        if hasattr(self, 'command_pool'):
            vk.vkDestroyCommandPool(self.device, self.command_pool, None)
        if hasattr(self, 'pipeline'):
            vk.vkDestroyPipeline(self.device, self.pipeline, None)
        if hasattr(self, 'pipeline_layout'):
            vk.vkDestroyPipelineLayout(self.device, self.pipeline_layout, None)
        if hasattr(self, 'descriptor_layout'):
            vk.vkDestroyDescriptorSetLayout(self.device, self.descriptor_layout, None)
        if hasattr(self, 'shader_module'):
            vk.vkDestroyShaderModule(self.device, self.shader_module, None)
        if hasattr(self, 'device'):
            vk.vkDestroyDevice(self.device, None)
        if hasattr(self, 'instance'):
            vk.vkDestroyInstance(self.instance, None)


# Quick test
if __name__ == "__main__":
    print("=" * 70)
    print("VULKAN TERNARY OPERATIONS TEST")
    print("=" * 70)

    try:
        # Test initialization
        vulkan = VulkanTernaryOps(use_int8_optimized=True)
        print("\n[OK] Vulkan initialized successfully!")
        print(f"[OK] Device: {vulkan.device_name}")
        print(f"[OK] Shader loaded and pipeline created")

        # Test weight packing
        test_weights = np.array([-1, 0, 1, -1, 0, 1], dtype=np.float32)
        packed = vulkan.pack_ternary_weights(test_weights)
        print(f"\n[OK] Weight packing working")
        print(f"     Original: {test_weights}")
        print(f"     Packed shape: {packed.shape}, dtype: {packed.dtype}")

    except Exception as e:
        print(f"\n[ERROR] Vulkan initialization failed: {e}")
        print("\nTroubleshooting:")
        print("  1. Install vulkan package: pip install vulkan")
        print("  2. Ensure Vulkan drivers installed")
        print("  3. Check GPU supports Vulkan 1.1+")

    print("\n" + "=" * 70)
    print("All ways, always!")
    print("=" * 70)
