// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Runtime.InteropServices;

namespace Idrak.Gpu.Vulkan;

// The memory budget (VK_EXT_memory_budget): per heap, what this process may take before allocations fail or slow the
// system (heapBudget), and what it takes now (heapUsage), read through vkGetPhysicalDeviceMemoryProperties2 (core 1.1).
internal static unsafe partial class VulkanDriver
{
    public const uint StructurePhysicalDeviceMemoryProperties2 = 1000059006,
        StructurePhysicalDeviceMemoryBudgetProperties = 1000237000;

    /// <summary>VK_EXT_memory_budget: the heaps' budget and usage, as the driver estimates them for this process.</summary>
    public const string MemoryBudgetExtension = "VK_EXT_memory_budget";

    [LibraryImport(Library)]
    public static partial void vkGetPhysicalDeviceMemoryProperties2(IntPtr physicalDevice, VkPhysicalDeviceMemoryProperties2* properties);
}

/// <summary>VkPhysicalDeviceMemoryProperties2: the memory properties and a chain of extension structures.</summary>
[StructLayout(LayoutKind.Sequential)]
internal unsafe struct VkPhysicalDeviceMemoryProperties2
{
    public uint SType;
    public void* PNext;
    public VkPhysicalDeviceMemoryProperties MemoryProperties;
}

/// <summary>VkPhysicalDeviceMemoryBudgetPropertiesEXT: per heap (VK_MAX_MEMORY_HEAPS, 16), the budget and the usage in bytes.</summary>
[StructLayout(LayoutKind.Sequential)]
internal unsafe struct VkPhysicalDeviceMemoryBudgetProperties
{
    public uint SType;
    public void* PNext;
    public fixed ulong HeapBudget[16];
    public fixed ulong HeapUsage[16];
}
