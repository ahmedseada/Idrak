// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Reflection;
using System.Runtime.InteropServices;
using System.Runtime.Loader;

namespace Idrak.Backends.Vulkan;

/// <summary>Thrown when a Vulkan call fails.</summary>
public sealed class VulkanException(string message) : Exception(message);

/// <summary>
/// Bindings to the Vulkan loader (vulkan-1.dll on Windows, libvulkan.so.1 on Linux, libvulkan.so on Android), which ships with the graphics
/// driver: the core 1.0 and 1.1 functions a compute backend needs, called through the loader's exports (no SDK, no
/// validation layers), and VK_KHR_push_descriptor where the device has it (through vkGetDeviceProcAddr). Handles of dispatchable objects (instance, physical device, device, queue, command
/// buffer) are pointers; every other handle is a 64-bit number.
/// </summary>
internal static unsafe partial class VulkanDriver
{
    private const string Library = "vulkan-1";

    private static IntPtr s_handle;

    static VulkanDriver()
    {
        // The CUDA bindings own this assembly's import resolver, so the loader is found through the load context's
        // fallback instead (asked only for names the default search did not find: libvulkan.so.1 on Linux).
        var context = AssemblyLoadContext.GetLoadContext(typeof(VulkanDriver).Assembly) ?? AssemblyLoadContext.Default;
        context.ResolvingUnmanagedDll += Resolve;
    }

    /// <summary>Loads the Vulkan loader; returns false (with a reason) when there is none.</summary>
    public static bool TryLoad(out string reason)
    {
        if (OperatingSystem.IsMacOS() || OperatingSystem.IsIOS())
        {
            reason = "Vulkan is not used on macOS or iOS (Metal is the plan there)";
            return false;
        }

        if (s_handle != IntPtr.Zero)
        {
            reason = "";
            return true;
        }

        foreach (var name in CandidateNames())
        {
            if (NativeLibrary.TryLoad(name, out var handle))
            {
                s_handle = handle;                                             // kept for the process
                reason = "";
                return true;
            }
        }

        reason = $"the Vulkan loader ({string.Join(" / ", CandidateNames())}) was not found; install the GPU's graphics driver";
        return false;
    }

    /// <summary>Whether the loader exports <paramref name="function"/> (vkEnumerateInstanceVersion is missing from 1.0 loaders).</summary>
    public static bool Exports(string function) => s_handle != IntPtr.Zero && NativeLibrary.TryGetExport(s_handle, function, out _);

    private static IntPtr Resolve(Assembly assembly, string name)
    {
        if (name != Library || assembly != typeof(VulkanDriver).Assembly)
        {
            return IntPtr.Zero;
        }

        return TryLoad(out _) ? s_handle : IntPtr.Zero;
    }

    private static string[] CandidateNames() => CandidateNames(
        OperatingSystem.IsWindows() ? "windows" :
        OperatingSystem.IsAndroid() ? "android" :
        OperatingSystem.IsLinux() ? "linux" :
        OperatingSystem.IsFreeBSD() ? "freebsd" : "");

    /// <summary>The loader's file names tried in order on operating system <paramref name="os"/> ("windows", "android",
    /// "linux" or "freebsd"; none elsewhere). Android ships the loader as libvulkan.so, without a version suffix.</summary>
    internal static string[] CandidateNames(string os) => os switch
    {
        "windows" => ["vulkan-1.dll"],
        "android" => ["libvulkan.so"],
        "linux" or "freebsd" => ["libvulkan.so.1", "libvulkan.so"],
        _ => [],
    };

    public static void Check(int result, string call)
    {
        if (result != Success)
        {
            throw new VulkanException($"{call} failed: {Describe(result)}");
        }
    }

    // The specification's name for a result code, with the code.
    public static string Describe(int result) => result switch
    {
        Success => "VK_SUCCESS",
        NotReady => "VK_NOT_READY",
        Timeout => "VK_TIMEOUT",
        Incomplete => "VK_INCOMPLETE",
        ErrorOutOfHostMemory => "VK_ERROR_OUT_OF_HOST_MEMORY",
        ErrorOutOfDeviceMemory => "VK_ERROR_OUT_OF_DEVICE_MEMORY",
        -3 => "VK_ERROR_INITIALIZATION_FAILED",
        ErrorDeviceLost => "VK_ERROR_DEVICE_LOST",
        -5 => "VK_ERROR_MEMORY_MAP_FAILED",
        -6 => "VK_ERROR_LAYER_NOT_PRESENT",
        -7 => "VK_ERROR_EXTENSION_NOT_PRESENT",
        -8 => "VK_ERROR_FEATURE_NOT_PRESENT",
        -9 => "VK_ERROR_INCOMPATIBLE_DRIVER",
        -10 => "VK_ERROR_TOO_MANY_OBJECTS",
        -12 => "VK_ERROR_FRAGMENTED_POOL",
        -13 => "VK_ERROR_UNKNOWN",
        ErrorOutOfPoolMemory => "VK_ERROR_OUT_OF_POOL_MEMORY",
        -1000012000 => "VK_ERROR_INVALID_SHADER_NV",
        _ => "VK_ERROR",
    } + $" ({result})";

    // Result codes.
    public const int Success = 0, NotReady = 1, Timeout = 2, Incomplete = 5;
    public const int ErrorOutOfHostMemory = -1, ErrorOutOfDeviceMemory = -2, ErrorDeviceLost = -4, ErrorOutOfPoolMemory = -1000069000;

    // Structure types.
    public const uint StructureApplicationInfo = 0, StructureInstanceCreateInfo = 1, StructureDeviceQueueCreateInfo = 2,
        StructureDeviceCreateInfo = 3, StructureSubmitInfo = 4, StructureMemoryAllocateInfo = 5, StructureMappedMemoryRange = 6,
        StructureFenceCreateInfo = 8, StructureBufferCreateInfo = 12, StructureShaderModuleCreateInfo = 16,
        StructurePipelineCacheCreateInfo = 17, StructurePipelineShaderStageCreateInfo = 18, StructureComputePipelineCreateInfo = 29,
        StructurePipelineLayoutCreateInfo = 30, StructureDescriptorSetLayoutCreateInfo = 32, StructureDescriptorPoolCreateInfo = 33,
        StructureDescriptorSetAllocateInfo = 34, StructureWriteDescriptorSet = 35, StructureCommandPoolCreateInfo = 39,
        StructureCommandBufferAllocateInfo = 40, StructureCommandBufferBeginInfo = 42, StructureMemoryBarrier = 46,
        StructurePhysicalDeviceProperties2 = 1000059001, StructurePhysicalDeviceDriverProperties = 1000196000,
        StructurePhysicalDeviceIdProperties = 1000071004, StructurePhysicalDeviceSubgroupProperties = 1000094000,
        StructurePhysicalDeviceMaintenance3Properties = 1000168000, StructurePhysicalDeviceSubgroupSizeControlProperties = 1000225000,
        StructurePhysicalDevicePciBusInfoProperties = 1000212000;

    public static uint MakeVersion(uint major, uint minor) => (major << 22) | (minor << 12);

    public static (uint Major, uint Minor) VersionOf(uint version) => (version >> 22 & 0x7F, version >> 12 & 0x3FF);

    // Physical device types.
    public const uint DeviceTypeOther = 0, DeviceTypeIntegratedGpu = 1, DeviceTypeDiscreteGpu = 2, DeviceTypeVirtualGpu = 3, DeviceTypeCpu = 4;

    public const uint QueueCompute = 0x2;

    // Memory properties.
    public const uint MemoryDeviceLocal = 0x1, MemoryHostVisible = 0x2, MemoryHostCoherent = 0x4, MemoryHostCached = 0x8;
    public const uint HeapDeviceLocal = 0x1;

    // Buffers.
    public const uint BufferTransferSource = 0x1, BufferTransferDestination = 0x2, BufferStorage = 0x20;
    public const uint SharingExclusive = 0;
    public const ulong WholeSize = ulong.MaxValue;

    // Descriptors, pipelines and shaders.
    public const uint DescriptorStorageBuffer = 7;
    public const uint ShaderStageCompute = 0x20;
    public const uint PipelineBindPointCompute = 1;

    // Commands.
    public const uint CommandPoolResetCommandBuffer = 0x2;
    public const uint CommandBufferLevelPrimary = 0;
    public const uint CommandBufferOneTimeSubmit = 0x1;

    // Pipeline stages and memory access.
    public const uint StageTopOfPipe = 0x1, StageComputeShader = 0x800, StageTransfer = 0x1000, StageHost = 0x4000;
    public const uint AccessShaderRead = 0x20, AccessShaderWrite = 0x40, AccessTransferRead = 0x800, AccessTransferWrite = 0x1000,
        AccessHostRead = 0x2000, AccessHostWrite = 0x4000;

    // Instance and devices.

    [LibraryImport(Library)]
    public static partial int vkEnumerateInstanceVersion(out uint version);

    [LibraryImport(Library)]
    public static partial int vkCreateInstance(VkInstanceCreateInfo* info, void* allocator, out IntPtr instance);

    [LibraryImport(Library)]
    public static partial int vkEnumeratePhysicalDevices(IntPtr instance, ref uint count, IntPtr* devices);

    [LibraryImport(Library)]
    public static partial void vkGetPhysicalDeviceProperties(IntPtr physicalDevice, VkPhysicalDeviceProperties* properties);

    [LibraryImport(Library)]
    public static partial void vkGetPhysicalDeviceProperties2(IntPtr physicalDevice, VkPhysicalDeviceProperties2* properties);

    [LibraryImport(Library)]
    public static partial void vkGetPhysicalDeviceMemoryProperties(IntPtr physicalDevice, VkPhysicalDeviceMemoryProperties* properties);

    [LibraryImport(Library)]
    public static partial void vkGetPhysicalDeviceQueueFamilyProperties(IntPtr physicalDevice, ref uint count, VkQueueFamilyProperties* properties);

    [LibraryImport(Library)]
    public static partial int vkCreateDevice(IntPtr physicalDevice, VkDeviceCreateInfo* info, void* allocator, out IntPtr device);

    [LibraryImport(Library)]
    public static partial void vkDestroyDevice(IntPtr device, void* allocator);

    [LibraryImport(Library)]
    public static partial void vkGetDeviceQueue(IntPtr device, uint family, uint index, out IntPtr queue);

    [LibraryImport(Library)]
    public static partial int vkDeviceWaitIdle(IntPtr device);

    // Memory and buffers.

    [LibraryImport(Library)]
    public static partial int vkCreateBuffer(IntPtr device, VkBufferCreateInfo* info, void* allocator, out ulong buffer);

    [LibraryImport(Library)]
    public static partial void vkDestroyBuffer(IntPtr device, ulong buffer, void* allocator);

    [LibraryImport(Library)]
    public static partial void vkGetBufferMemoryRequirements(IntPtr device, ulong buffer, VkMemoryRequirements* requirements);

    [LibraryImport(Library)]
    public static partial int vkAllocateMemory(IntPtr device, VkMemoryAllocateInfo* info, void* allocator, out ulong memory);

    [LibraryImport(Library)]
    public static partial void vkFreeMemory(IntPtr device, ulong memory, void* allocator);

    [LibraryImport(Library)]
    public static partial int vkBindBufferMemory(IntPtr device, ulong buffer, ulong memory, ulong offset);

    [LibraryImport(Library)]
    public static partial int vkMapMemory(IntPtr device, ulong memory, ulong offset, ulong size, uint flags, void** data);

    [LibraryImport(Library)]
    public static partial void vkUnmapMemory(IntPtr device, ulong memory);

    [LibraryImport(Library)]
    public static partial int vkFlushMappedMemoryRanges(IntPtr device, uint count, VkMappedMemoryRange* ranges);

    [LibraryImport(Library)]
    public static partial int vkInvalidateMappedMemoryRanges(IntPtr device, uint count, VkMappedMemoryRange* ranges);

    // Descriptors.

    [LibraryImport(Library)]
    public static partial int vkCreateDescriptorSetLayout(IntPtr device, VkDescriptorSetLayoutCreateInfo* info, void* allocator, out ulong layout);

    [LibraryImport(Library)]
    public static partial void vkDestroyDescriptorSetLayout(IntPtr device, ulong layout, void* allocator);

    [LibraryImport(Library)]
    public static partial int vkCreateDescriptorPool(IntPtr device, VkDescriptorPoolCreateInfo* info, void* allocator, out ulong pool);

    [LibraryImport(Library)]
    public static partial void vkDestroyDescriptorPool(IntPtr device, ulong pool, void* allocator);

    [LibraryImport(Library)]
    public static partial int vkResetDescriptorPool(IntPtr device, ulong pool, uint flags);

    [LibraryImport(Library)]
    public static partial int vkAllocateDescriptorSets(IntPtr device, VkDescriptorSetAllocateInfo* info, ulong* sets);

    [LibraryImport(Library)]
    public static partial void vkUpdateDescriptorSets(IntPtr device, uint writeCount, VkWriteDescriptorSet* writes, uint copyCount, void* copies);

    // Shaders and pipelines.

    [LibraryImport(Library)]
    public static partial int vkCreatePipelineLayout(IntPtr device, VkPipelineLayoutCreateInfo* info, void* allocator, out ulong layout);

    [LibraryImport(Library)]
    public static partial void vkDestroyPipelineLayout(IntPtr device, ulong layout, void* allocator);

    [LibraryImport(Library)]
    public static partial int vkCreateShaderModule(IntPtr device, VkShaderModuleCreateInfo* info, void* allocator, out ulong module);

    [LibraryImport(Library)]
    public static partial void vkDestroyShaderModule(IntPtr device, ulong module, void* allocator);

    [LibraryImport(Library)]
    public static partial int vkCreatePipelineCache(IntPtr device, VkPipelineCacheCreateInfo* info, void* allocator, out ulong cache);

    [LibraryImport(Library)]
    public static partial void vkDestroyPipelineCache(IntPtr device, ulong cache, void* allocator);

    [LibraryImport(Library)]
    public static partial int vkCreateComputePipelines(IntPtr device, ulong cache, uint count, VkComputePipelineCreateInfo* infos, void* allocator, ulong* pipelines);

    [LibraryImport(Library)]
    public static partial void vkDestroyPipeline(IntPtr device, ulong pipeline, void* allocator);

    // Command buffers.

    [LibraryImport(Library)]
    public static partial int vkCreateCommandPool(IntPtr device, VkCommandPoolCreateInfo* info, void* allocator, out ulong pool);

    [LibraryImport(Library)]
    public static partial void vkDestroyCommandPool(IntPtr device, ulong pool, void* allocator);

    [LibraryImport(Library)]
    public static partial int vkAllocateCommandBuffers(IntPtr device, VkCommandBufferAllocateInfo* info, IntPtr* buffers);

    [LibraryImport(Library)]
    public static partial int vkResetCommandBuffer(IntPtr commandBuffer, uint flags);

    [LibraryImport(Library)]
    public static partial int vkBeginCommandBuffer(IntPtr commandBuffer, VkCommandBufferBeginInfo* info);

    [LibraryImport(Library)]
    public static partial int vkEndCommandBuffer(IntPtr commandBuffer);

    [LibraryImport(Library)]
    public static partial void vkCmdBindPipeline(IntPtr commandBuffer, uint bindPoint, ulong pipeline);

    [LibraryImport(Library)]
    public static partial void vkCmdBindDescriptorSets(IntPtr commandBuffer, uint bindPoint, ulong layout, uint firstSet, uint count, ulong* sets, uint dynamicOffsetCount, uint* dynamicOffsets);

    [LibraryImport(Library)]
    public static partial void vkCmdPushConstants(IntPtr commandBuffer, ulong layout, uint stages, uint offset, uint size, void* values);

    [LibraryImport(Library)]
    public static partial void vkCmdDispatch(IntPtr commandBuffer, uint groupsX, uint groupsY, uint groupsZ);

    [LibraryImport(Library)]
    public static partial void vkCmdPipelineBarrier(
        IntPtr commandBuffer, uint sourceStages, uint destinationStages, uint dependencyFlags,
        uint memoryBarrierCount, VkMemoryBarrier* memoryBarriers, uint bufferBarrierCount, void* bufferBarriers, uint imageBarrierCount, void* imageBarriers);

    [LibraryImport(Library)]
    public static partial void vkCmdCopyBuffer(IntPtr commandBuffer, ulong source, ulong destination, uint regionCount, VkBufferCopy* regions);

    [LibraryImport(Library)]
    public static partial void vkCmdFillBuffer(IntPtr commandBuffer, ulong buffer, ulong offset, ulong size, uint data);

    // Synchronization and submission.

    [LibraryImport(Library)]
    public static partial int vkCreateFence(IntPtr device, VkFenceCreateInfo* info, void* allocator, out ulong fence);

    [LibraryImport(Library)]
    public static partial void vkDestroyFence(IntPtr device, ulong fence, void* allocator);

    [LibraryImport(Library)]
    public static partial int vkResetFences(IntPtr device, uint count, ulong* fences);

    [LibraryImport(Library)]
    public static partial int vkWaitForFences(IntPtr device, uint count, ulong* fences, uint waitAll, ulong timeout);

    [LibraryImport(Library)]
    public static partial int vkQueueSubmit(IntPtr queue, uint count, VkSubmitInfo* submits, ulong fence);

    [LibraryImport(Library)]
    public static partial int vkQueueWaitIdle(IntPtr queue);

    [LibraryImport(Library)]
    public static partial int vkEnumerateDeviceExtensionProperties(IntPtr physicalDevice, byte* layerName, ref uint count, VkExtensionProperties* properties);

    [LibraryImport(Library)]
    public static partial IntPtr vkGetDeviceProcAddr(IntPtr device, byte* name);

    /// <summary>VK_KHR_push_descriptor: descriptors written straight into the command buffer (no sets to allocate).</summary>
    public const string PushDescriptorExtension = "VK_KHR_push_descriptor";

    /// <summary>VK_EXT_subgroup_size_control: the subgroup sizes a device runs (core in Vulkan 1.3).</summary>
    public const string SubgroupSizeControlExtension = "VK_EXT_subgroup_size_control";

    /// <summary>VK_EXT_pci_bus_info: the device's PCI address (orders devices that report no UUID).</summary>
    public const string PciBusInfoExtension = "VK_EXT_pci_bus_info";

    /// <summary>VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT_KHR.</summary>
    public const uint DescriptorSetLayoutPushDescriptor = 0x1;
}

/// <summary>VkExtensionProperties: an extension's name and version.</summary>
[StructLayout(LayoutKind.Sequential)]
internal unsafe struct VkExtensionProperties
{
    public fixed byte ExtensionName[256];
    public uint SpecVersion;
}

// The structures, laid out as in vulkan_core.h (natural C alignment; 64-bit handles; pointers as void*).

[StructLayout(LayoutKind.Sequential)]
internal unsafe struct VkApplicationInfo
{
    public uint SType;
    public void* PNext;
    public byte* ApplicationName;
    public uint ApplicationVersion;
    public byte* EngineName;
    public uint EngineVersion;
    public uint ApiVersion;
}

[StructLayout(LayoutKind.Sequential)]
internal unsafe struct VkInstanceCreateInfo
{
    public uint SType;
    public void* PNext;
    public uint Flags;
    public VkApplicationInfo* ApplicationInfo;
    public uint EnabledLayerCount;
    public byte** EnabledLayerNames;
    public uint EnabledExtensionCount;
    public byte** EnabledExtensionNames;
}

/// <summary>VkPhysicalDeviceProperties (824 bytes): the fields read here, at their offsets; the limits start at 296.</summary>
[StructLayout(LayoutKind.Explicit, Size = 824)]
internal unsafe struct VkPhysicalDeviceProperties
{
    [FieldOffset(0)] public uint ApiVersion;
    [FieldOffset(4)] public uint DriverVersion;
    [FieldOffset(8)] public uint VendorId;
    [FieldOffset(12)] public uint DeviceId;
    [FieldOffset(16)] public uint DeviceType;
    [FieldOffset(20)] public fixed byte DeviceName[256];
    [FieldOffset(296 + 28)] public uint MaxStorageBufferRange;
    [FieldOffset(296 + 32)] public uint MaxPushConstantsSize;
    [FieldOffset(296 + 36)] public uint MaxMemoryAllocationCount;
    [FieldOffset(296 + 76)] public uint MaxPerStageDescriptorStorageBuffers;
    [FieldOffset(296 + 216)] public uint MaxComputeSharedMemorySize;
    [FieldOffset(296 + 220)] public uint MaxComputeWorkGroupCountX;
    [FieldOffset(296 + 224)] public uint MaxComputeWorkGroupCountY;
    [FieldOffset(296 + 228)] public uint MaxComputeWorkGroupCountZ;
    [FieldOffset(296 + 232)] public uint MaxComputeWorkGroupInvocations;
    [FieldOffset(296 + 236)] public uint MaxComputeWorkGroupSizeX;
    [FieldOffset(296 + 240)] public uint MaxComputeWorkGroupSizeY;
    [FieldOffset(296 + 244)] public uint MaxComputeWorkGroupSizeZ;
    [FieldOffset(296 + 328)] public ulong MinStorageBufferOffsetAlignment;
    [FieldOffset(296 + 496)] public ulong NonCoherentAtomSize;
}

[StructLayout(LayoutKind.Sequential)]
internal unsafe struct VkPhysicalDeviceProperties2
{
    public uint SType;
    public void* PNext;
    public VkPhysicalDeviceProperties Properties;
}

[StructLayout(LayoutKind.Sequential)]
internal unsafe struct VkPhysicalDeviceDriverProperties
{
    public uint SType;
    public void* PNext;
    public uint DriverId;
    public fixed byte DriverName[256];
    public fixed byte DriverInfo[256];
    public uint ConformanceVersion;
}

/// <summary>VkPhysicalDevicePCIBusInfoPropertiesEXT (VK_EXT_pci_bus_info).</summary>
[StructLayout(LayoutKind.Sequential)]
internal unsafe struct VkPhysicalDevicePciBusInfoProperties
{
    public uint SType;
    public void* PNext;
    public uint PciDomain;
    public uint PciBus;
    public uint PciDevice;
    public uint PciFunction;
}

/// <summary>VkPhysicalDeviceIDProperties (Vulkan 1.1): the UUID that names the same GPU across APIs (CUDA's cuDeviceGetUuid too).</summary>
[StructLayout(LayoutKind.Sequential)]
internal unsafe struct VkPhysicalDeviceIdProperties
{
    public uint SType;
    public void* PNext;
    public fixed byte DeviceUuid[16];
    public fixed byte DriverUuid[16];
    public fixed byte DeviceLuid[8];
    public uint DeviceNodeMask;
    public uint DeviceLuidValid;
}

/// <summary>VkPhysicalDeviceSubgroupProperties (Vulkan 1.1).</summary>
[StructLayout(LayoutKind.Sequential)]
internal unsafe struct VkPhysicalDeviceSubgroupProperties
{
    public uint SType;
    public void* PNext;
    public uint SubgroupSize;
    public uint SupportedStages;
    public uint SupportedOperations;
    public uint QuadOperationsInAllStages;
}

/// <summary>VkPhysicalDeviceMaintenance3Properties (Vulkan 1.1).</summary>
[StructLayout(LayoutKind.Sequential)]
internal unsafe struct VkPhysicalDeviceMaintenance3Properties
{
    public uint SType;
    public void* PNext;
    public uint MaxPerSetDescriptors;
    public ulong MaxMemoryAllocationSize;
}

/// <summary>VkPhysicalDeviceSubgroupSizeControlProperties (Vulkan 1.3, or VK_EXT_subgroup_size_control).</summary>
[StructLayout(LayoutKind.Sequential)]
internal unsafe struct VkPhysicalDeviceSubgroupSizeControlProperties
{
    public uint SType;
    public void* PNext;
    public uint MinSubgroupSize;
    public uint MaxSubgroupSize;
    public uint MaxComputeWorkgroupSubgroups;
    public uint RequiredSubgroupSizeStages;
}

/// <summary>VkPhysicalDeviceMemoryProperties: 32 memory types (flags, heap) and 16 heaps (size, flags).</summary>
[StructLayout(LayoutKind.Explicit, Size = 520)]
internal unsafe struct VkPhysicalDeviceMemoryProperties
{
    [FieldOffset(0)] public uint MemoryTypeCount;
    [FieldOffset(4)] public fixed uint MemoryTypes[64];                      // (propertyFlags, heapIndex) pairs
    [FieldOffset(260)] public uint MemoryHeapCount;
    [FieldOffset(264)] public fixed ulong MemoryHeaps[32];                   // (size, flags) pairs

    public uint TypeFlags(int type) => MemoryTypes[2 * type];

    public int TypeHeap(int type) => (int)MemoryTypes[2 * type + 1];

    public ulong HeapSize(int heap) => MemoryHeaps[2 * heap];

    public ulong HeapFlags(int heap) => MemoryHeaps[2 * heap + 1];
}

[StructLayout(LayoutKind.Sequential)]
internal struct VkQueueFamilyProperties
{
    public uint QueueFlags;
    public uint QueueCount;
    public uint TimestampValidBits;
    public uint GranularityWidth, GranularityHeight, GranularityDepth;
}

[StructLayout(LayoutKind.Sequential)]
internal unsafe struct VkDeviceQueueCreateInfo
{
    public uint SType;
    public void* PNext;
    public uint Flags;
    public uint QueueFamilyIndex;
    public uint QueueCount;
    public float* QueuePriorities;
}

[StructLayout(LayoutKind.Sequential)]
internal unsafe struct VkDeviceCreateInfo
{
    public uint SType;
    public void* PNext;
    public uint Flags;
    public uint QueueCreateInfoCount;
    public VkDeviceQueueCreateInfo* QueueCreateInfos;
    public uint EnabledLayerCount;
    public byte** EnabledLayerNames;
    public uint EnabledExtensionCount;
    public byte** EnabledExtensionNames;
    public void* EnabledFeatures;
}

[StructLayout(LayoutKind.Sequential)]
internal unsafe struct VkBufferCreateInfo
{
    public uint SType;
    public void* PNext;
    public uint Flags;
    public ulong Size;
    public uint Usage;
    public uint SharingMode;
    public uint QueueFamilyIndexCount;
    public uint* QueueFamilyIndices;
}

[StructLayout(LayoutKind.Sequential)]
internal struct VkMemoryRequirements
{
    public ulong Size;
    public ulong Alignment;
    public uint MemoryTypeBits;
}

[StructLayout(LayoutKind.Sequential)]
internal unsafe struct VkMemoryAllocateInfo
{
    public uint SType;
    public void* PNext;
    public ulong AllocationSize;
    public uint MemoryTypeIndex;
}

[StructLayout(LayoutKind.Sequential)]
internal unsafe struct VkMappedMemoryRange
{
    public uint SType;
    public void* PNext;
    public ulong Memory;
    public ulong Offset;
    public ulong Size;
}

[StructLayout(LayoutKind.Sequential)]
internal unsafe struct VkDescriptorSetLayoutBinding
{
    public uint Binding;
    public uint DescriptorType;
    public uint DescriptorCount;
    public uint StageFlags;
    public void* ImmutableSamplers;
}

[StructLayout(LayoutKind.Sequential)]
internal unsafe struct VkDescriptorSetLayoutCreateInfo
{
    public uint SType;
    public void* PNext;
    public uint Flags;
    public uint BindingCount;
    public VkDescriptorSetLayoutBinding* Bindings;
}

[StructLayout(LayoutKind.Sequential)]
internal struct VkDescriptorPoolSize
{
    public uint Type;
    public uint DescriptorCount;
}

[StructLayout(LayoutKind.Sequential)]
internal unsafe struct VkDescriptorPoolCreateInfo
{
    public uint SType;
    public void* PNext;
    public uint Flags;
    public uint MaxSets;
    public uint PoolSizeCount;
    public VkDescriptorPoolSize* PoolSizes;
}

[StructLayout(LayoutKind.Sequential)]
internal unsafe struct VkDescriptorSetAllocateInfo
{
    public uint SType;
    public void* PNext;
    public ulong DescriptorPool;
    public uint DescriptorSetCount;
    public ulong* SetLayouts;
}

[StructLayout(LayoutKind.Sequential)]
internal struct VkDescriptorBufferInfo
{
    public ulong Buffer;
    public ulong Offset;
    public ulong Range;
}

[StructLayout(LayoutKind.Sequential)]
internal unsafe struct VkWriteDescriptorSet
{
    public uint SType;
    public void* PNext;
    public ulong DestinationSet;
    public uint DestinationBinding;
    public uint DestinationArrayElement;
    public uint DescriptorCount;
    public uint DescriptorType;
    public void* ImageInfo;
    public VkDescriptorBufferInfo* BufferInfo;
    public void* TexelBufferView;
}

[StructLayout(LayoutKind.Sequential)]
internal struct VkPushConstantRange
{
    public uint StageFlags;
    public uint Offset;
    public uint Size;
}

[StructLayout(LayoutKind.Sequential)]
internal unsafe struct VkPipelineLayoutCreateInfo
{
    public uint SType;
    public void* PNext;
    public uint Flags;
    public uint SetLayoutCount;
    public ulong* SetLayouts;
    public uint PushConstantRangeCount;
    public VkPushConstantRange* PushConstantRanges;
}

[StructLayout(LayoutKind.Sequential)]
internal unsafe struct VkShaderModuleCreateInfo
{
    public uint SType;
    public void* PNext;
    public uint Flags;
    public nuint CodeSize;
    public uint* Code;
}

[StructLayout(LayoutKind.Sequential)]
internal unsafe struct VkPipelineCacheCreateInfo
{
    public uint SType;
    public void* PNext;
    public uint Flags;
    public nuint InitialDataSize;
    public void* InitialData;
}

[StructLayout(LayoutKind.Sequential)]
internal unsafe struct VkPipelineShaderStageCreateInfo
{
    public uint SType;
    public void* PNext;
    public uint Flags;
    public uint Stage;
    public ulong Module;
    public byte* Name;
    public void* SpecializationInfo;
}

[StructLayout(LayoutKind.Sequential)]
internal unsafe struct VkComputePipelineCreateInfo
{
    public uint SType;
    public void* PNext;
    public uint Flags;
    public VkPipelineShaderStageCreateInfo Stage;
    public ulong Layout;
    public ulong BasePipelineHandle;
    public int BasePipelineIndex;
}

[StructLayout(LayoutKind.Sequential)]
internal unsafe struct VkCommandPoolCreateInfo
{
    public uint SType;
    public void* PNext;
    public uint Flags;
    public uint QueueFamilyIndex;
}

[StructLayout(LayoutKind.Sequential)]
internal unsafe struct VkCommandBufferAllocateInfo
{
    public uint SType;
    public void* PNext;
    public ulong CommandPool;
    public uint Level;
    public uint CommandBufferCount;
}

[StructLayout(LayoutKind.Sequential)]
internal unsafe struct VkCommandBufferBeginInfo
{
    public uint SType;
    public void* PNext;
    public uint Flags;
    public void* InheritanceInfo;
}

[StructLayout(LayoutKind.Sequential)]
internal unsafe struct VkMemoryBarrier
{
    public uint SType;
    public void* PNext;
    public uint SourceAccessMask;
    public uint DestinationAccessMask;
}

[StructLayout(LayoutKind.Sequential)]
internal struct VkBufferCopy
{
    public ulong SourceOffset;
    public ulong DestinationOffset;
    public ulong Size;
}

[StructLayout(LayoutKind.Sequential)]
internal unsafe struct VkFenceCreateInfo
{
    public uint SType;
    public void* PNext;
    public uint Flags;
}

[StructLayout(LayoutKind.Sequential)]
internal unsafe struct VkSubmitInfo
{
    public uint SType;
    public void* PNext;
    public uint WaitSemaphoreCount;
    public ulong* WaitSemaphores;
    public uint* WaitDestinationStageMask;
    public uint CommandBufferCount;
    public IntPtr* CommandBuffers;
    public uint SignalSemaphoreCount;
    public ulong* SignalSemaphores;
}
