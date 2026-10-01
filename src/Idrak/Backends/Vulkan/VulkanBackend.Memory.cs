// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using static Idrak.Backends.Vulkan.VulkanDriver;

namespace Idrak.Backends.Vulkan;

// Device memory. Drivers cap the number of memory allocations (maxMemoryAllocationCount: 4096 on many Windows drivers),
// and a model has more tensors than that, so storages share allocations: each storage keeps a buffer of its own, bound
// into a page (one allocation of PageBytes, mapped once when the host sees it) at an offset carved from the page's free
// list. Storages larger than a quarter page get an allocation of their own. Carved ranges go back to their page when the
// cached blocks are freed (ReleaseCachedMemory, after waiting for the device), and empty pages go back to the driver then.
internal sealed unsafe partial class VulkanBackend
{
    // The pages, by memory type (also the lock over them and their free lists), and the live allocations (pages,
    // storages of their own, the staging buffer).
    private readonly List<Page> _pages = [];
    private int _allocations;

    /// <summary>The size of the shared allocations storages are carved from.</summary>
    public long PageBytes { get; }

    /// <summary>The largest storage carved from a page (a quarter page); larger storages get an allocation of their own.</summary>
    public long SubAllocationMax => PageBytes / 4;

    /// <summary>The memory allocations the backend makes at most: the driver's maxMemoryAllocationCount, or IDRAK_VULKAN_MAX_ALLOCATIONS when lower.</summary>
    internal int MaxAllocations { get; }

    /// <summary>The memory allocations live now (pages, storages of their own, the staging buffer).</summary>
    internal int Allocations => Volatile.Read(ref _allocations);

    /// <summary>The pages live now.</summary>
    internal int PageCount
    {
        get
        {
            lock (_pages)
            {
                return _pages.Count;
            }
        }
    }

    // The driver's cap, lowered by `cap` or IDRAK_VULKAN_MAX_ALLOCATIONS (to reproduce a low cap, e.g. AMD's 4096, on any device).
    private static int MemoryAllocationCap(uint driver, int? cap)
    {
        int limit = driver == 0 ? int.MaxValue : (int)Math.Min(driver, int.MaxValue);
        if (cap is null && int.TryParse(Environment.GetEnvironmentVariable("IDRAK_VULKAN_MAX_ALLOCATIONS"), out int value))
        {
            cap = value;
        }

        return cap is int c && c > 0 ? Math.Min(limit, c) : limit;
    }

    // The page size for a heap of `heapBytes`, from what the device reports: the power of two at most 1/128 of the heap
    // (64 MiB of an 8 GiB heap: a page's unused tail wastes under 1% of the heap), raised until half the driver's
    // allocation cap (`allocationCount`, maxMemoryAllocationCount) in pages covers the whole heap (so pages never run out
    // of allocations before the heap runs out of memory), at most the largest allocation (`maxAllocation`,
    // maxMemoryAllocationSize). `setting` (tests, IDRAK_VULKAN_PAGE_BYTES) instead when given.
    internal static long PageSize(ulong heapBytes, ulong maxAllocation, uint allocationCount, long? setting)
    {
        if (setting is long s and > 0)
        {
            return s;
        }

        ulong page = PowerOfTwoAtMost(heapBytes / 128);
        if (allocationCount >= 2)
        {
            while (page < heapBytes && page * (allocationCount / 2) < heapBytes)
            {
                page *= 2;
            }
        }

        if (maxAllocation > 0)
        {
            page = Math.Min(page, PowerOfTwoAtMost(maxAllocation));
        }

        return (long)Math.Max(page, sizeof(float));
    }

    // One shared allocation and its free ranges (sorted by offset, neighbors merged).
    private sealed class Page(ulong memory, uint type, long size, byte* mapped)
    {
        public readonly ulong Memory = memory;
        public readonly uint Type = type;
        public readonly long Size = size;
        public readonly byte* Mapped = mapped;
        public readonly List<(long Offset, long Size)> Free = [(0, size)];
        public long Used;

        // The offset of `size` bytes aligned to `alignment` (first fit), or -1 when no free range holds them.
        public long Take(long size, long alignment)
        {
            for (int i = 0; i < Free.Count; i++)
            {
                var (offset, length) = Free[i];
                long start = (offset + alignment - 1) / alignment * alignment, end = start + size;
                if (end > offset + length)
                {
                    continue;
                }

                Free.RemoveAt(i);
                if (end < offset + length)
                {
                    Free.Insert(i, (end, offset + length - end));
                }

                if (start > offset)
                {
                    Free.Insert(i, (offset, start - offset));
                }

                Used += size;
                return start;
            }

            return -1;
        }

        // Gives a range back, merged with the free ranges around it.
        public void Give(long offset, long size)
        {
            int i = 0;
            while (i < Free.Count && Free[i].Offset < offset)
            {
                i++;
            }

            Free.Insert(i, (offset, size));
            if (i + 1 < Free.Count && Free[i].Offset + Free[i].Size == Free[i + 1].Offset)
            {
                Free[i] = (Free[i].Offset, Free[i].Size + Free[i + 1].Size);
                Free.RemoveAt(i + 1);
            }

            if (i > 0 && Free[i - 1].Offset + Free[i - 1].Size == Free[i].Offset)
            {
                Free[i - 1] = (Free[i - 1].Offset, Free[i - 1].Size + Free[i].Size);
                Free.RemoveAt(i);
            }

            Used -= size;
        }
    }

    // A storage's buffer and where its memory is: `Reserved` bytes at `Offset` of a page, or (Page null) an allocation
    // of its own; mapped when the host sees it.
    internal sealed class VulkanBlock(ulong buffer, ulong memory, byte* mapped, int capacity, object? page, long offset, long reserved)
    {
        public readonly ulong Buffer = buffer;
        public readonly ulong Memory = memory;
        public readonly byte* Mapped = mapped;
        public readonly int Capacity = capacity;

        /// <summary>The page the memory is carved from (null: an allocation of its own).</summary>
        public readonly object? Page = page;

        public readonly long Offset = offset;
        public readonly long Reserved = reserved;

        /// <summary>The batch that last used the block (VulkanBackend.Dispatch.cs): host access waits for it.</summary>
        public ulong LastUse;

        /// <summary>The span between barriers (VulkanBackend.Dispatch.cs) in which a command last wrote the block.</summary>
        public ulong WrittenIn;

        /// <summary>The span between barriers in which a command last read the block.</summary>
        public ulong ReadIn;
    }

    // A buffer for `length` floats with memory of `memoryType`: carved from a page, or of its own when large (or
    // `dedicated`). When no memory is left, or the allocation cap is reached, frees cached blocks, empty pages and
    // unreachable tensors and tries once more.
    private VulkanBlock CreateBlock(int length, uint memoryType, bool map, bool dedicated = false)
    {
        long bytes = BlockBytes(length);
        ulong buffer = CreateBuffer(bytes);
        bool bound = false;
        try
        {
            VkMemoryRequirements requirements;
            vkGetBufferMemoryRequirements(_device, buffer, &requirements);
            long size = (long)requirements.Size;
            bool own = dedicated || size > SubAllocationMax;
            bool capped = false;
            for (int attempt = 0; attempt < 2; attempt++)
            {
                if (attempt == 1)
                {
                    GC.Collect();
                    GC.WaitForPendingFinalizers();
                    ReleaseCachedMemory();
                }

                if (own)
                {
                    ulong memory = AllocateMemory(size, memoryType, ref capped);
                    if (memory == 0)
                    {
                        continue;
                    }

                    Check(vkBindBufferMemory(_device, buffer, memory, 0), nameof(vkBindBufferMemory));
                    void* mapped = null;
                    if (map)
                    {
                        Check(vkMapMemory(_device, memory, 0, WholeSize, 0, &mapped), nameof(vkMapMemory));
                    }

                    bound = true;
                    return new VulkanBlock(buffer, memory, (byte*)mapped, length, null, 0, size);
                }

                // Ranges of memory the host flushes or invalidates start and end on nonCoherentAtomSize.
                long alignment = Math.Max((long)requirements.Alignment, 1);
                if (map && (_physical.Memory.TypeFlags((int)memoryType) & MemoryHostCoherent) == 0)
                {
                    alignment = Math.Max(alignment, (long)Math.Max(_physical.Properties.NonCoherentAtomSize, 1));
                }

                long reserved = (size + alignment - 1) / alignment * alignment;
                var (page, offset) = Carve(reserved, alignment, memoryType, map, ref capped);
                if (page is null)
                {
                    continue;
                }

                Check(vkBindBufferMemory(_device, buffer, page.Memory, (ulong)offset), nameof(vkBindBufferMemory));
                bound = true;
                return new VulkanBlock(buffer, page.Memory, page.Mapped == null ? null : page.Mapped + offset, length, page, offset, reserved);
            }

            throw new ResourceLimitExceededException(capped
                ? $"{Name} allows {MaxAllocations:N0} memory allocations, and they are all in use ({PageCount} pages of {PageBytes >> 20} MiB; {_memory.Usage})."
                : $"{Name} is out of memory: {bytes:N0} more bytes needed ({_memory.Usage}). Use smaller batches or shorter sequences.");
        }
        finally
        {
            if (!bound)
            {
                vkDestroyBuffer(_device, buffer, null);
            }
        }
    }

    // `size` bytes aligned to `alignment` in a page of `memoryType` (a new page when none has room); (null, 0) when no
    // page can be allocated.
    private (Page? Page, long Offset) Carve(long size, long alignment, uint memoryType, bool map, ref bool capped)
    {
        lock (_pages)
        {
            foreach (var page in _pages)
            {
                if (page.Type == memoryType && page.Size - page.Used >= size && page.Take(size, alignment) is var offset and >= 0)
                {
                    return (page, offset);
                }
            }

            ulong memory = AllocateMemory(PageBytes, memoryType, ref capped);
            if (memory == 0)
            {
                return (null, 0);
            }

            void* mapped = null;
            if (map)
            {
                Check(vkMapMemory(_device, memory, 0, WholeSize, 0, &mapped), nameof(vkMapMemory));
            }

            var fresh = new Page(memory, memoryType, PageBytes, (byte*)mapped);
            _pages.Add(fresh);
            return (fresh, fresh.Take(size, alignment));
        }
    }

    // A memory allocation, or 0 when the device is out of memory or the allocation cap is reached (`capped` set).
    private ulong AllocateMemory(long size, uint memoryType, ref bool capped)
    {
        if (Interlocked.Increment(ref _allocations) > MaxAllocations)
        {
            Interlocked.Decrement(ref _allocations);
            capped = true;
            return 0;
        }

        var info = new VkMemoryAllocateInfo { SType = StructureMemoryAllocateInfo, AllocationSize = (ulong)size, MemoryTypeIndex = memoryType };
        int result = vkAllocateMemory(_device, &info, null, out ulong memory);
        if (result != Success)
        {
            Interlocked.Decrement(ref _allocations);
            if (result is ErrorOutOfDeviceMemory or ErrorOutOfHostMemory)
            {
                return 0;
            }

            Check(result, nameof(vkAllocateMemory));
        }

        return memory;
    }

    // Destroys a block's buffer and gives its memory back: to its page, or to the driver. Only once the device no longer
    // uses it (ReleaseCachedMemory and Shutdown wait first).
    private void DestroyBlock(VulkanBlock block)
    {
        vkDestroyBuffer(_device, block.Buffer, null);
        if (block.Page is Page page)
        {
            lock (_pages)
            {
                page.Give(block.Offset, block.Reserved);
            }

            return;
        }

        vkFreeMemory(_device, block.Memory, null);                           // unmaps it too
        Interlocked.Decrement(ref _allocations);
    }

    // Gives the pages no block uses back to the driver.
    private void FreeEmptyPages()
    {
        lock (_pages)
        {
            for (int i = _pages.Count - 1; i >= 0; i--)
            {
                if (_pages[i].Used == 0)
                {
                    vkFreeMemory(_device, _pages[i].Memory, null);
                    Interlocked.Decrement(ref _allocations);
                    _pages.RemoveAt(i);
                }
            }
        }
    }

    // Makes host writes visible to the device (flush) or device writes visible to the host (invalidate) on memory that
    // is not host-coherent: the block's range of its page (aligned to nonCoherentAtomSize), or its whole allocation.
    private void FlushOrInvalidate(VulkanBlock block, bool flush)
    {
        var range = new VkMappedMemoryRange
        {
            SType = StructureMappedMemoryRange,
            Memory = block.Memory,
            Offset = (ulong)block.Offset,
            Size = block.Page is null ? WholeSize : (ulong)block.Reserved,
        };
        Check(flush ? vkFlushMappedMemoryRanges(_device, 1, &range) : vkInvalidateMappedMemoryRanges(_device, 1, &range),
            flush ? nameof(vkFlushMappedMemoryRanges) : nameof(vkInvalidateMappedMemoryRanges));
    }
}
