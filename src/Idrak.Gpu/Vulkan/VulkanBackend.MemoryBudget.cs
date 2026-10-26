// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using static Idrak.Gpu.Vulkan.VulkanDriver;

namespace Idrak.Gpu.Vulkan;

// The memory budget (VK_EXT_memory_budget). Without it, the memory a device can take is its storage heap less what is in
// use and less the reserve: on a device that shares the system's memory with the operating system and other programs
// (an integrated GPU, a phone), nearly all of it then looks free. Where the device reports the extension (and
// IDRAK_VULKAN_MEMORY_BUDGET is not 0), it is enabled at device creation and read each time the available or free memory
// is asked, so the device reports what the driver says this process can really take. Without it nothing changes.
internal sealed unsafe partial class VulkanBackend
{
    /// <summary>
    /// Tests: false makes backends created afterwards leave the memory budget unread (the heap less the reserve, as
    /// without the extension); IDRAK_VULKAN_MEMORY_BUDGET=0 sets it for a process.
    /// </summary>
    internal static bool? MemoryBudgetOverride { get; set; } =
        Environment.GetEnvironmentVariable("IDRAK_VULKAN_MEMORY_BUDGET") is "0" or "false" ? false : null;

    // Whether VK_EXT_memory_budget is enabled on this device and its budget read (decided once, at device creation).
    private readonly bool _memoryBudget;

    /// <summary>Whether the device's memory budget (VK_EXT_memory_budget) is read for its available and free memory.</summary>
    internal bool MemoryBudget => _memoryBudget;

    // Whether the device offers the memory budget and it is not turned off: the extension, and the loader's
    // vkGetPhysicalDeviceMemoryProperties2 (core 1.1) to read it through.
    private static bool UsesMemoryBudget(IntPtr physical) =>
        MemoryBudgetOverride is not false && Exports(nameof(vkGetPhysicalDeviceMemoryProperties2)) && HasExtension(physical, MemoryBudgetExtension);

    /// <summary>
    /// The storage heap's budget and this process's usage of it, as the driver reports them now, or null without the
    /// extension (or when the driver reports no budget for the heap).
    /// </summary>
    internal (long Budget, long Usage)? StorageHeapBudget()
    {
        if (!_memoryBudget)
        {
            return null;
        }

        var budget = new VkPhysicalDeviceMemoryBudgetProperties { SType = StructurePhysicalDeviceMemoryBudgetProperties };
        var properties = new VkPhysicalDeviceMemoryProperties2 { SType = StructurePhysicalDeviceMemoryProperties2, PNext = &budget };
        vkGetPhysicalDeviceMemoryProperties2(_physical.Handle, &properties);
        int heap = _physical.Memory.TypeHeap((int)_storageType);
        if ((uint)heap >= Math.Min(properties.MemoryProperties.MemoryHeapCount, 16u) || budget.HeapBudget[heap] == 0)
        {
            return null;                                                       // a driver that fills nothing in
        }

        return ((long)Math.Min(budget.HeapBudget[heap], long.MaxValue), (long)Math.Min(budget.HeapUsage[heap], long.MaxValue));
    }

    /// <summary>
    /// The bytes new tensors can take (<see cref="IMemoryReport.AvailableMemory"/>): <paramref name="heap"/> less
    /// <paramref name="inUse"/> (live tensors) and less <paramref name="reserve"/>, as without the budget; with a
    /// <paramref name="budget"/>, at most what the budget leaves this process: its budget less its usage, plus the
    /// <paramref name="cached"/> blocks the pool keeps. The usage counts every allocation of this process on the heap:
    /// live tensors, the pool's cached blocks, the pages' uncarved space and the staging buffer. The cached blocks are
    /// counted in it although a new tensor reuses them without a new allocation, so they are added back; the pages'
    /// uncarved space is not (it takes only blocks under a quarter page, so counting it would promise room a large tensor
    /// cannot get). The reserve is not taken from the budget again: the budget is the driver's estimate of what this
    /// process may take with the rest of the system's use already set aside, and the heap's bound keeps the reserve.
    /// Over budget (usage above it), only the cached blocks beyond the excess remain, and never less than 0.
    /// </summary>
    internal static long AvailableBytes(long heap, long inUse, long cached, long reserve, (long Budget, long Usage)? budget)
    {
        long onHeap = Math.Max(0, heap - inUse - reserve);
        return budget is { } b ? Math.Min(onHeap, Math.Max(0, b.Budget - b.Usage + cached)) : onHeap;
    }
}
