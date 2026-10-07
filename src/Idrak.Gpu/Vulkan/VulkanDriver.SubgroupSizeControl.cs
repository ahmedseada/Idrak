// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Runtime.InteropServices;

namespace Idrak.Gpu.Vulkan;

// Subgroup size control (VK_EXT_subgroup_size_control, core in Vulkan 1.3): the features a device is asked for, and the
// required size a compute stage gives its pipeline.
internal static unsafe partial class VulkanDriver
{
    public const uint StructurePhysicalDeviceFeatures2 = 1000059000,
        StructurePipelineShaderStageRequiredSubgroupSizeCreateInfo = 1000225001,
        StructurePhysicalDeviceSubgroupSizeControlFeatures = 1000225002;

    // VkPipelineShaderStageCreateFlagBits.
    public const uint PipelineStageAllowVaryingSubgroupSize = 0x1, PipelineStageRequireFullSubgroups = 0x2;

    [LibraryImport(Library)]
    public static partial void vkGetPhysicalDeviceFeatures2(IntPtr physicalDevice, VkPhysicalDeviceFeatures2* features);
}

/// <summary>VkPhysicalDeviceFeatures2: the core features (55 VkBool32 values) and a chain of extension features.</summary>
[StructLayout(LayoutKind.Sequential)]
internal unsafe struct VkPhysicalDeviceFeatures2
{
    public uint SType;
    public void* PNext;
    public fixed uint Features[55];
}

/// <summary>VkPhysicalDeviceSubgroupSizeControlFeatures (Vulkan 1.3, or VK_EXT_subgroup_size_control).</summary>
[StructLayout(LayoutKind.Sequential)]
internal unsafe struct VkPhysicalDeviceSubgroupSizeControlFeatures
{
    public uint SType;
    public void* PNext;
    public uint SubgroupSizeControl;
    public uint ComputeFullSubgroups;
}

/// <summary>VkPipelineShaderStageRequiredSubgroupSizeCreateInfo: the subgroup size a stage requires.</summary>
[StructLayout(LayoutKind.Sequential)]
internal unsafe struct VkPipelineShaderStageRequiredSubgroupSizeCreateInfo
{
    public uint SType;
    public void* PNext;
    public uint RequiredSubgroupSize;
}
