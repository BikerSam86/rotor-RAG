"""
Full Vulkan compute implementation for ternary matrix operations.
Complete buffer management, descriptor sets, and compute dispatch.
"""

import numpy as np
from pathlib import Path
import vulkan as vk


class VulkanTernaryCompute:
    """Complete Vulkan compute backend for ternary operations."""

    def __init__(self, use_int8_optimized=False):
        """
        Initialize Vulkan compute.

        Args:
            use_int8_optimized: Use int8 shader (requires feature support)
        """
        self.use_int8 = use_int8_optimized
        self._init_vulkan()

    def _init_vulkan(self):
        """Initialize all Vulkan resources."""
        # 1. Create instance
        app_info = vk.VkApplicationInfo(
            sType=vk.VK_STRUCTURE_TYPE_APPLICATION_INFO,
            pApplicationName="Rotor Ternary Compute",
            applicationVersion=vk.VK_MAKE_VERSION(1, 0, 0),
            pEngineName="Rotor",
            engineVersion=vk.VK_MAKE_VERSION(1, 0, 0),
            apiVersion=vk.VK_API_VERSION_1_0
        )

        instance_info = vk.VkInstanceCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
            pApplicationInfo=app_info
        )

        self.instance = vk.vkCreateInstance(instance_info, None)

        # 2. Select physical device
        physical_devices = vk.vkEnumeratePhysicalDevices(self.instance)
        if not physical_devices:
            raise RuntimeError("No Vulkan devices found!")

        self.physical_device = physical_devices[0]
        props = vk.vkGetPhysicalDeviceProperties(self.physical_device)
        self.device_name = props.deviceName
        print(f"[Vulkan] Using device: {self.device_name}")

        # 3. Find compute queue family
        queue_families = vk.vkGetPhysicalDeviceQueueFamilyProperties(self.physical_device)
        self.compute_queue_family = None

        for i, family in enumerate(queue_families):
            if family.queueFlags & vk.VK_QUEUE_COMPUTE_BIT:
                self.compute_queue_family = i
                break

        if self.compute_queue_family is None:
            raise RuntimeError("No compute queue family found!")

        # 4. Create logical device
        queue_create_info = vk.VkDeviceQueueCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
            queueFamilyIndex=self.compute_queue_family,
            queueCount=1,
            pQueuePriorities=[1.0]
        )

        device_create_info = vk.VkDeviceCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
            queueCreateInfoCount=1,
            pQueueCreateInfos=[queue_create_info],
            pEnabledFeatures=vk.VkPhysicalDeviceFeatures()
        )

        self.device = vk.vkCreateDevice(self.physical_device, device_create_info, None)
        self.queue = vk.vkGetDeviceQueue(self.device, self.compute_queue_family, 0)

        # 5. Load shader
        self._load_shader()

        # 6. Create descriptor set layout
        self._create_descriptor_layout()

        # 7. Create pipeline
        self._create_pipeline()

        # 8. Create command pool
        pool_info = vk.VkCommandPoolCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
            queueFamilyIndex=self.compute_queue_family,
            flags=vk.VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT
        )
        self.command_pool = vk.vkCreateCommandPool(self.device, pool_info, None)

        # 9. Create descriptor pool
        pool_size = vk.VkDescriptorPoolSize(
            type=vk.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            descriptorCount=4
        )

        descriptor_pool_info = vk.VkDescriptorPoolCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
            maxSets=1,
            poolSizeCount=1,
            pPoolSizes=[pool_size]
        )

        self.descriptor_pool = vk.vkCreateDescriptorPool(self.device, descriptor_pool_info, None)

        print("[Vulkan] Initialization complete!")

    def _load_shader(self):
        """Load SPIR-V shader."""
        shader_dir = Path(__file__).parent / "shaders"
        shader_name = "ternary_matmul_optimized.spv" if self.use_int8 else "ternary_matmul.spv"
        shader_path = shader_dir / shader_name

        if not shader_path.exists():
            raise FileNotFoundError(f"Shader not found: {shader_path}")

        with open(shader_path, 'rb') as f:
            shader_code = f.read()

        shader_info = vk.VkShaderModuleCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
            codeSize=len(shader_code),
            pCode=shader_code
        )

        self.shader_module = vk.vkCreateShaderModule(self.device, shader_info, None)
        print(f"[Vulkan] Loaded shader: {shader_name}")

    def _create_descriptor_layout(self):
        """Create descriptor set layout for 4 storage buffers."""
        bindings = [
            vk.VkDescriptorSetLayoutBinding(
                binding=i,
                descriptorType=vk.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                descriptorCount=1,
                stageFlags=vk.VK_SHADER_STAGE_COMPUTE_BIT,
                pImmutableSamplers=None
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
        # Push constant range
        push_constant = vk.VkPushConstantRange(
            stageFlags=vk.VK_SHADER_STAGE_COMPUTE_BIT,
            offset=0,
            size=8  # 2 uint32s
        )

        # Pipeline layout
        layout_info = vk.VkPipelineLayoutCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
            setLayoutCount=1,
            pSetLayouts=[self.descriptor_layout],
            pushConstantRangeCount=1,
            pPushConstantRanges=[push_constant]
        )

        self.pipeline_layout = vk.vkCreatePipelineLayout(self.device, layout_info, None)

        # Shader stage
        stage_info = vk.VkPipelineShaderStageCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            stage=vk.VK_SHADER_STAGE_COMPUTE_BIT,
            module=self.shader_module,
            pName="main"
        )

        # Pipeline
        pipeline_info = vk.VkComputePipelineCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
            stage=stage_info,
            layout=self.pipeline_layout
        )

        self.pipeline = vk.vkCreateComputePipelines(
            self.device, None, 1, [pipeline_info], None
        )[0]

    def _create_buffer(self, size, usage):
        """Create Vulkan buffer."""
        buffer_info = vk.VkBufferCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
            size=size,
            usage=usage,
            sharingMode=vk.VK_SHARING_MODE_EXCLUSIVE
        )

        buffer = vk.vkCreateBuffer(self.device, buffer_info, None)

        # Get memory requirements
        mem_reqs = vk.vkGetBufferMemoryRequirements(self.device, buffer)

        # Find memory type
        mem_props = vk.vkGetPhysicalDeviceMemoryProperties(self.physical_device)
        memory_type = None

        for i in range(mem_props.memoryTypeCount):
            if (mem_reqs.memoryTypeBits & (1 << i)) and \
               (mem_props.memoryTypes[i].propertyFlags & vk.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT):
                memory_type = i
                break

        if memory_type is None:
            raise RuntimeError("No suitable memory type found!")

        # Allocate memory
        alloc_info = vk.VkMemoryAllocateInfo(
            sType=vk.VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
            allocationSize=mem_reqs.size,
            memoryTypeIndex=memory_type
        )

        memory = vk.vkAllocateMemory(self.device, alloc_info, None)

        # Bind buffer to memory
        vk.vkBindBufferMemory(self.device, buffer, memory, 0)

        return buffer, memory

    def pack_weights(self, weights):
        """Pack ternary weights for GPU."""
        if self.use_int8:
            return weights.astype(np.int8)
        else:
            # Bit-packed format (4 weights per byte)
            encoded = (weights + 1).astype(np.uint8)
            flat = encoded.flatten()
            num_packed = (len(flat) + 3) // 4
            packed = np.zeros(num_packed, dtype=np.uint8)

            for i in range(len(flat)):
                byte_idx = i // 4
                bit_offset = (i % 4) * 2
                packed[byte_idx] |= (flat[i] << bit_offset)

            return packed

    def ternary_matmul(self, packed_weights, scales, input_vec, in_dim, out_dim):
        """
        Execute ternary matrix multiplication on GPU.

        Args:
            packed_weights: Packed ternary weights
            scales: Scale factors [out_dim]
            input_vec: Input vector [in_dim]
            in_dim: Input dimension
            out_dim: Output dimension

        Returns:
            Output vector [out_dim]
        """
        # Create buffers
        weight_size = packed_weights.nbytes
        input_size = input_vec.nbytes
        scale_size = scales.nbytes
        output_size = out_dim * 4  # float32

        weight_buf, weight_mem = self._create_buffer(
            weight_size, vk.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
        )
        input_buf, input_mem = self._create_buffer(
            input_size, vk.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
        )
        scale_buf, scale_mem = self._create_buffer(
            scale_size, vk.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
        )
        output_buf, output_mem = self._create_buffer(
            output_size, vk.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
        )

        # Upload data
        def upload_data(memory, data):
            data_ptr = vk.vkMapMemory(self.device, memory, 0, data.nbytes, 0)
            vk.ffi.memmove(data_ptr, vk.ffi.from_buffer(data), data.nbytes)
            vk.vkUnmapMemory(self.device, memory)

        upload_data(weight_mem, packed_weights)
        upload_data(input_mem, input_vec.astype(np.float32))
        upload_data(scale_mem, scales.astype(np.float32))

        # Create descriptor set
        alloc_info = vk.VkDescriptorSetAllocateInfo(
            sType=vk.VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
            descriptorPool=self.descriptor_pool,
            descriptorSetCount=1,
            pSetLayouts=[self.descriptor_layout]
        )

        descriptor_set = vk.vkAllocateDescriptorSets(self.device, alloc_info)[0]

        # Update descriptor set
        buffer_sizes = [weight_size, input_size, scale_size, output_size]
        buffer_infos = [
            vk.VkDescriptorBufferInfo(buffer=buf, offset=0, range=size)
            for buf, size in zip([weight_buf, input_buf, scale_buf, output_buf], buffer_sizes)
        ]

        writes = [
            vk.VkWriteDescriptorSet(
                sType=vk.VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                dstSet=descriptor_set,
                dstBinding=i,
                descriptorCount=1,
                descriptorType=vk.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                pBufferInfo=[buffer_infos[i]]
            )
            for i in range(4)
        ]

        vk.vkUpdateDescriptorSets(self.device, len(writes), writes, 0, None)

        # Create command buffer
        cmd_alloc_info = vk.VkCommandBufferAllocateInfo(
            sType=vk.VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
            commandPool=self.command_pool,
            level=vk.VK_COMMAND_BUFFER_LEVEL_PRIMARY,
            commandBufferCount=1
        )

        cmd_buffer = vk.vkAllocateCommandBuffers(self.device, cmd_alloc_info)[0]

        # Record commands
        begin_info = vk.VkCommandBufferBeginInfo(
            sType=vk.VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
            flags=vk.VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT
        )

        vk.vkBeginCommandBuffer(cmd_buffer, begin_info)

        # Bind pipeline and descriptor set
        vk.vkCmdBindPipeline(cmd_buffer, vk.VK_PIPELINE_BIND_POINT_COMPUTE, self.pipeline)
        vk.vkCmdBindDescriptorSets(
            cmd_buffer, vk.VK_PIPELINE_BIND_POINT_COMPUTE,
            self.pipeline_layout, 0, 1, [descriptor_set], 0, None
        )

        # Push constants
        push_data = np.array([in_dim, out_dim], dtype=np.uint32)
        vk.vkCmdPushConstants(
            cmd_buffer, self.pipeline_layout,
            vk.VK_SHADER_STAGE_COMPUTE_BIT,
            0, 8, vk.ffi.from_buffer(push_data)
        )

        # Dispatch
        group_count = (out_dim + 255) // 256
        vk.vkCmdDispatch(cmd_buffer, group_count, 1, 1)

        vk.vkEndCommandBuffer(cmd_buffer)

        # Submit
        submit_info = vk.VkSubmitInfo(
            sType=vk.VK_STRUCTURE_TYPE_SUBMIT_INFO,
            commandBufferCount=1,
            pCommandBuffers=[cmd_buffer]
        )

        vk.vkQueueSubmit(self.queue, 1, [submit_info], None)
        vk.vkQueueWaitIdle(self.queue)

        # Read results
        result = np.zeros(out_dim, dtype=np.float32)
        result_ptr = vk.vkMapMemory(self.device, output_mem, 0, output_size, 0)
        vk.ffi.memmove(vk.ffi.from_buffer(result), result_ptr, output_size)
        vk.vkUnmapMemory(self.device, output_mem)

        # Cleanup
        vk.vkFreeCommandBuffers(self.device, self.command_pool, 1, [cmd_buffer])
        vk.vkDestroyBuffer(self.device, weight_buf, None)
        vk.vkDestroyBuffer(self.device, input_buf, None)
        vk.vkDestroyBuffer(self.device, scale_buf, None)
        vk.vkDestroyBuffer(self.device, output_buf, None)
        vk.vkFreeMemory(self.device, weight_mem, None)
        vk.vkFreeMemory(self.device, input_mem, None)
        vk.vkFreeMemory(self.device, scale_mem, None)
        vk.vkFreeMemory(self.device, output_mem, None)

        return result

    def __del__(self):
        """Cleanup Vulkan resources."""
        if hasattr(self, 'descriptor_pool'):
            vk.vkDestroyDescriptorPool(self.device, self.descriptor_pool, None)
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
