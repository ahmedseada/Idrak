// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Runtime.InteropServices;
using System.Text;
using Idrak.Backends.Cpu;
using static Idrak.Backends.Vulkan.VulkanDriver;

namespace Idrak.Backends.Vulkan;

/// <summary>
/// A GPU driven through Vulkan compute (Intel and AMD GPUs, and any other Vulkan 1.1 device). One compute queue; one
/// buffer and memory allocation per storage, pooled by exact size; kernels are SPIR-V dispatched in order
/// (<see cref="Dispatch"/>, VulkanBackend.Dispatch.cs). Operations without a kernel of their own run through the host
/// fallback (<see cref="HostCall"/>).
///
/// Memory: on an integrated GPU (or a CPU driver) storages live in memory both the device and the host see
/// (device-local and host-visible), mapped once, so uploads and downloads are plain copies. Elsewhere storages live in
/// device-local memory, and copies go through a host-visible staging buffer.
/// </summary>
internal sealed unsafe partial class VulkanBackend : Backend
{
    // The process's Vulkan instance and the usable devices, found once.
    private static readonly Lazy<(PhysicalDevice[] Devices, string Reason)> Probe = new(ProbeDevices);
    private static readonly Lazy<Lazy<VulkanBackend>[]> Instances = new(CreateInstances);
    private static IntPtr s_instance;

    private readonly PhysicalDevice _physical;
    private readonly IntPtr _device;
    private readonly IntPtr _queue;
    private readonly MemoryAccountant _memory;

    // Memory types: where storages live, and (without mapped storages) the staging buffer's.
    private readonly uint _storageType;
    private readonly bool _storageCoherent;
    private readonly int _stagingType = -1;
    private readonly bool _stagingCoherent;

    // Cached blocks by length (floats), and the number of live allocations (drivers cap it: maxMemoryAllocationCount).
    private readonly Dictionary<int, Stack<VulkanBlock>> _pool = [];
    private int _allocations;

    // The staging buffer, created on first use; copies larger than it go in chunks.
    private const long StagingBytes = 16L << 20;
    private VulkanBlock? _staging;

    private VulkanBackend(PhysicalDevice physical, bool preferMapped)
    {
        _physical = physical;
        var p = physical.Properties;
        Name = $"{physical.DeviceName} (Vulkan {VersionOf(p.ApiVersion).Major}.{VersionOf(p.ApiVersion).Minor}, {physical.Driver})";
        _memory = new MemoryAccountant(() => ComputeResources.GpuMemoryLimit, Name);
        MaxStorageBytes = p.MaxStorageBufferRange;

        // One compute queue.
        float priority = 1f;
        var queueInfo = new VkDeviceQueueCreateInfo
        {
            SType = StructureDeviceQueueCreateInfo,
            QueueFamilyIndex = physical.QueueFamily,
            QueueCount = 1,
            QueuePriorities = &priority,
        };
        var deviceInfo = new VkDeviceCreateInfo
        {
            SType = StructureDeviceCreateInfo,
            QueueCreateInfoCount = 1,
            QueueCreateInfos = &queueInfo,
        };
        Check(vkCreateDevice(physical.Handle, &deviceInfo, null, out _device), nameof(vkCreateDevice));
        vkGetDeviceQueue(_device, physical.QueueFamily, 0, out _queue);

        // Every storage buffer has the same memory requirements' type bits, so one probe buffer tells them.
        uint typeBits = StorageTypeBits();
        var memory = physical.Memory;
        bool shared = p.DeviceType is DeviceTypeIntegratedGpu or DeviceTypeCpu;
        int mapped = preferMapped && shared ? BestType(memory, typeBits, MemoryDeviceLocal | MemoryHostVisible, static f =>
            ((f & MemoryHostCached) != 0 ? 2 : 0) + ((f & MemoryHostCoherent) != 0 ? 1 : 0)) : -1;
        if (mapped >= 0)
        {
            _storageType = (uint)mapped;
            UnifiedMemory = true;
        }
        else
        {
            // Device memory the host does not see first (a discrete GPU's host-visible window is small and slow to read).
            int local = BestType(memory, typeBits, MemoryDeviceLocal, static f => (f & MemoryHostVisible) == 0 ? 1 : 0);
            _storageType = (uint)(local >= 0 ? local : BestType(memory, typeBits, 0, static _ => 0));
            _stagingType = BestType(memory, typeBits, MemoryHostVisible, static f =>
                ((f & MemoryHostCached) != 0 ? 4 : 0) + ((f & MemoryHostCoherent) != 0 ? 2 : 0) + ((f & MemoryDeviceLocal) == 0 ? 1 : 0));
            if (_stagingType < 0)
            {
                throw new VulkanException($"{Name} has no host-visible memory for copies.");
            }

            _stagingCoherent = (memory.TypeFlags(_stagingType) & MemoryHostCoherent) != 0;
        }

        _storageCoherent = (memory.TypeFlags((int)_storageType) & MemoryHostCoherent) != 0;
        StartQueue();
    }

    /// <summary>Whether storages live in memory the host maps directly (integrated GPUs): copies need no staging.</summary>
    public bool UnifiedMemory { get; }

    /// <summary>The largest storage a kernel can bind (maxStorageBufferRange); larger ones take the host fallback.</summary>
    public long MaxStorageBytes { get; }

    /// <summary>The PCI vendor id (0x8086 Intel, 0x1002 AMD, 0x10DE NVIDIA).</summary>
    public uint VendorId => _physical.Properties.VendorId;

    public override string Name { get; }

    public override BackendCapabilities Capabilities { get; } = CpuBackend.Instance.Capabilities with
    {
        MatrixUnits = false,
        MatrixUnitAttentionHeadDim = static _ => false,
        FusedKernels = false,
        Profiling = false,
    };

    /// <summary>The number of usable Vulkan devices (0 when there is no loader or driver).</summary>
    public static int DeviceCount => Probe.Value.Devices.Length;

    /// <summary>Why there are no Vulkan devices ("" when there are).</summary>
    public static string UnavailableReason => Probe.Value.Reason;

    /// <summary>The backend of device <paramref name="ordinal"/>, started on first use.</summary>
    public static VulkanBackend Get(int ordinal) => Instances.Value[ordinal].Value;

    public static bool IsInitialized(int ordinal) => Instances.IsValueCreated && Instances.Value[ordinal].IsValueCreated;

    /// <summary>Device <paramref name="ordinal"/>'s type (VK_PHYSICAL_DEVICE_TYPE_*) and PCI vendor id, without starting it.</summary>
    public static (uint Type, uint Vendor) DeviceKind(int ordinal) =>
        (Probe.Value.Devices[ordinal].Properties.DeviceType, Probe.Value.Devices[ordinal].Properties.VendorId);

    /// <summary>A second backend on device <paramref name="ordinal"/> (for tests: staging copies even where memory is shared).</summary>
    internal static VulkanBackend CreateSeparate(int ordinal, bool preferMapped) => new(Probe.Value.Devices[ordinal], preferMapped);

    private static Lazy<VulkanBackend>[] CreateInstances()
    {
        bool staging = Environment.GetEnvironmentVariable("IDRAK_VULKAN_STAGING") is "1" or "true";
        return [.. Probe.Value.Devices.Select(d => new Lazy<VulkanBackend>(() => new VulkanBackend(d, preferMapped: !staging)))];
    }

    // A physical device with what the backend reads of it.
    private sealed class PhysicalDevice(IntPtr handle, VkPhysicalDeviceProperties properties, string deviceName, string driver,
        VkPhysicalDeviceMemoryProperties memory, uint queueFamily)
    {
        public IntPtr Handle { get; } = handle;

        public VkPhysicalDeviceProperties Properties = properties;

        public string DeviceName { get; } = deviceName;

        public string Driver { get; } = driver;

        public VkPhysicalDeviceMemoryProperties Memory = memory;

        public uint QueueFamily { get; } = queueFamily;
    }

    private static (PhysicalDevice[], string) ProbeDevices()
    {
        if (Environment.GetEnvironmentVariable("IDRAK_DISABLE_VULKAN") is "1" or "true")
        {
            return ([], "disabled by the IDRAK_DISABLE_VULKAN environment variable");
        }

        if (!TryLoad(out string reason))
        {
            return ([], reason);
        }

        try
        {
            uint loaderVersion = MakeVersion(1, 0);
            if (Exports(nameof(vkEnumerateInstanceVersion)))
            {
                Check(vkEnumerateInstanceVersion(out loaderVersion), nameof(vkEnumerateInstanceVersion));
            }

            if (loaderVersion < MakeVersion(1, 1))
            {
                return ([], "the Vulkan loader supports only Vulkan 1.0 (1.1 is needed); update the graphics driver");
            }

            uint apiVersion = Math.Min(loaderVersion & ~0xFFFu, MakeVersion(1, 3));
            byte* name = stackalloc byte[] { (byte)'I', (byte)'d', (byte)'r', (byte)'a', (byte)'k', 0 };
            var application = new VkApplicationInfo
            {
                SType = StructureApplicationInfo,
                ApplicationName = name,
                EngineName = name,
                ApiVersion = apiVersion,
            };
            var instanceInfo = new VkInstanceCreateInfo { SType = StructureInstanceCreateInfo, ApplicationInfo = &application };
            int result = vkCreateInstance(&instanceInfo, null, out var instance);
            if (result != Success)
            {
                return ([], $"vkCreateInstance failed: {VulkanDriver.Describe(result)} (no Vulkan driver installed?)");
            }

            s_instance = instance;                                              // kept for the process
            uint count = 0;
            Check(vkEnumeratePhysicalDevices(instance, ref count, null), nameof(vkEnumeratePhysicalDevices));
            var handles = new IntPtr[count];
            fixed (IntPtr* h = handles)
            {
                result = vkEnumeratePhysicalDevices(instance, ref count, h);
                if (result is not (Success or Incomplete))
                {
                    Check(result, nameof(vkEnumeratePhysicalDevices));
                }
            }

            var devices = new List<PhysicalDevice>();
            var skipped = new List<string>();
            foreach (var handle in handles.AsSpan(0, (int)count))
            {
                VkPhysicalDeviceProperties properties;
                vkGetPhysicalDeviceProperties(handle, &properties);
                string deviceName = Utf8(properties.DeviceName, 256);
                if (properties.ApiVersion < MakeVersion(1, 1))
                {
                    skipped.Add($"{deviceName} supports only Vulkan 1.0");
                    continue;
                }

                if (ComputeFamily(handle) is not uint family)
                {
                    skipped.Add($"{deviceName} has no compute queue");
                    continue;
                }

                VkPhysicalDeviceMemoryProperties memory;
                vkGetPhysicalDeviceMemoryProperties(handle, &memory);
                devices.Add(new PhysicalDevice(handle, properties, deviceName, DriverName(handle, properties, apiVersion), memory, family));
            }

            return devices.Count > 0 ? ([.. devices], "")
                : ([], skipped.Count > 0 ? $"no usable Vulkan device ({string.Join("; ", skipped)})" : "the Vulkan driver reports no devices");
        }
        catch (Exception ex) when (ex is DllNotFoundException or EntryPointNotFoundException or VulkanException)
        {
            return ([], ex.Message);
        }
    }

    // The first queue family that runs compute work.
    private static uint? ComputeFamily(IntPtr physical)
    {
        uint count = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(physical, ref count, null);
        var families = new VkQueueFamilyProperties[count];
        fixed (VkQueueFamilyProperties* f = families)
        {
            vkGetPhysicalDeviceQueueFamilyProperties(physical, ref count, f);
        }

        for (uint i = 0; i < count; i++)
        {
            if ((families[i].QueueFlags & QueueCompute) != 0 && families[i].QueueCount > 0)
            {
                return i;
            }
        }

        return null;
    }

    // The driver's name and version text (Vulkan 1.2 devices report it), else its version number.
    private static string DriverName(IntPtr physical, VkPhysicalDeviceProperties properties, uint apiVersion)
    {
        if (properties.ApiVersion >= MakeVersion(1, 2) && apiVersion >= MakeVersion(1, 2))
        {
            var driver = new VkPhysicalDeviceDriverProperties { SType = StructurePhysicalDeviceDriverProperties };
            var all = new VkPhysicalDeviceProperties2 { SType = StructurePhysicalDeviceProperties2, PNext = &driver };
            vkGetPhysicalDeviceProperties2(physical, &all);
            string name = Utf8(driver.DriverName, 256), info = Utf8(driver.DriverInfo, 256);
            if (name.Length > 0)
            {
                return info.Length > 0 ? $"{name} {info}" : name;
            }
        }

        return $"driver 0x{properties.DriverVersion:X}";
    }

    private static string Utf8(byte* text, int capacity)
    {
        int length = 0;
        while (length < capacity && text[length] != 0)
        {
            length++;
        }

        return Encoding.UTF8.GetString(text, length).Trim();
    }

    // The memory type of `typeBits` with all of `required`, best by `score` then by heap size; -1 when none.
    private static int BestType(VkPhysicalDeviceMemoryProperties memory, uint typeBits, uint required, Func<uint, int> score)
    {
        int best = -1;
        (int Score, ulong Heap) bestKey = (int.MinValue, 0);
        for (int i = 0; i < (int)memory.MemoryTypeCount; i++)
        {
            uint flags = memory.TypeFlags(i);
            if ((typeBits & (1u << i)) == 0 || (flags & required) != required)
            {
                continue;
            }

            var key = (score(flags), memory.HeapSize(memory.TypeHeap(i)));
            if (best < 0 || key.Item1 > bestKey.Score || (key.Item1 == bestKey.Score && key.Item2 > bestKey.Heap))
            {
                (best, bestKey) = (i, key);
            }
        }

        return best;
    }

    private uint StorageTypeBits()
    {
        ulong buffer = CreateBuffer(4);
        VkMemoryRequirements requirements;
        vkGetBufferMemoryRequirements(_device, buffer, &requirements);
        vkDestroyBuffer(_device, buffer, null);
        return requirements.MemoryTypeBits;
    }

    private ulong CreateBuffer(long bytes)
    {
        var info = new VkBufferCreateInfo
        {
            SType = StructureBufferCreateInfo,
            Size = (ulong)bytes,
            Usage = BufferStorage | BufferTransferSource | BufferTransferDestination,
            SharingMode = SharingExclusive,
        };
        Check(vkCreateBuffer(_device, &info, null, out ulong buffer), nameof(vkCreateBuffer));
        return buffer;
    }

    private static long BlockBytes(int length) => (long)Math.Max(length, 1) * sizeof(float);

    // Storages: a buffer with its own memory, mapped for good when the host sees it.

    internal sealed class VulkanBlock(ulong buffer, ulong memory, byte* mapped, int capacity)
    {
        public readonly ulong Buffer = buffer;
        public readonly ulong Memory = memory;
        public readonly byte* Mapped = mapped;
        public readonly int Capacity = capacity;

        /// <summary>The batch that last used the block (VulkanBackend.Dispatch.cs): host access waits for it.</summary>
        public ulong LastUse;
    }

    private sealed class VulkanStorage(VulkanBackend backend, VulkanBlock block, int length) : Storage(backend, length)
    {
        public VulkanBlock? Block = block;
    }

    // The block of a storage of this device, or an error naming what is wrong.
    internal VulkanBlock BlockOf(Storage storage)
    {
        if (storage is not VulkanStorage s || !ReferenceEquals(s.Backend, this))
        {
            throw new ArgumentException($"The storage is not on {Name}.", nameof(storage));
        }

        return s.Block ?? throw new InvalidOperationException("The storage was evicted (Backend.Evict); restore it first.");
    }

    public override Storage Allocate(int length, bool zeroed)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(length);
        long bytes = BlockBytes(length);
        bool releaseCache = _memory.MustReleaseCacheFor(bytes);              // throws over the limit
        VulkanBlock? block = null;
        lock (_pool)
        {
            if (_pool.TryGetValue(length, out var bucket) && bucket.Count > 0)
            {
                block = bucket.Pop();
                _memory.Reused(bytes);
            }
        }

        if (block is null)
        {
            if (releaseCache)
            {
                ReleaseCachedMemory();
            }

            block = CreateBlock(length, _storageType, map: UnifiedMemory);
            _memory.Allocated(bytes);
        }

        if (zeroed)
        {
            Zero(block);
        }

        return new VulkanStorage(this, block, length);
    }

    // New memory for `length` floats; after an out-of-memory error, frees cached blocks and unreachable tensors and
    // tries once more.
    private VulkanBlock CreateBlock(int length, uint memoryType, bool map)
    {
        long bytes = BlockBytes(length);
        for (int attempt = 0; ; attempt++)
        {
            if (attempt == 1)
            {
                GC.Collect();
                GC.WaitForPendingFinalizers();
                ReleaseCachedMemory();
            }

            if (Volatile.Read(ref _allocations) >= _physical.Properties.MaxMemoryAllocationCount)
            {
                if (attempt == 0)
                {
                    continue;
                }

                throw new ResourceLimitExceededException(
                    $"{Name} allows {_physical.Properties.MaxMemoryAllocationCount:N0} memory allocations, and they are all in use.");
            }

            ulong buffer = CreateBuffer(bytes);
            VkMemoryRequirements requirements;
            vkGetBufferMemoryRequirements(_device, buffer, &requirements);
            var info = new VkMemoryAllocateInfo
            {
                SType = StructureMemoryAllocateInfo,
                AllocationSize = requirements.Size,
                MemoryTypeIndex = memoryType,
            };
            int result = vkAllocateMemory(_device, &info, null, out ulong memory);
            if (result is ErrorOutOfDeviceMemory or ErrorOutOfHostMemory)
            {
                vkDestroyBuffer(_device, buffer, null);
                if (attempt == 0)
                {
                    continue;
                }

                throw new ResourceLimitExceededException(
                    $"{Name} is out of memory: {bytes:N0} more bytes needed ({_memory.Usage}). Use smaller batches or shorter sequences.");
            }

            Check(result, nameof(vkAllocateMemory));
            Check(vkBindBufferMemory(_device, buffer, memory, 0), nameof(vkBindBufferMemory));
            void* mapped = null;
            if (map)
            {
                Check(vkMapMemory(_device, memory, 0, WholeSize, 0, &mapped), nameof(vkMapMemory));
            }

            Interlocked.Increment(ref _allocations);
            return new VulkanBlock(buffer, memory, (byte*)mapped, length);
        }
    }

    private void DestroyBlock(VulkanBlock block)
    {
        vkDestroyBuffer(_device, block.Buffer, null);
        vkFreeMemory(_device, block.Memory, null);                           // unmaps it too
        Interlocked.Decrement(ref _allocations);
    }

    // Called when the last reference is released, possibly from the finalizer thread, so it only touches the pool
    // (queued work may still use the block: it is reused in queue order, and freed only after a wait).
    public override void Return(Storage storage)
    {
        var block = ((VulkanStorage)storage).Block;
        if (block is null)
        {
            return;
        }

        lock (_pool)
        {
            if (!_pool.TryGetValue(block.Capacity, out var bucket))
            {
                _pool[block.Capacity] = bucket = new Stack<VulkanBlock>();
            }

            bucket.Push(block);
            _memory.Returned(BlockBytes(block.Capacity));
        }
    }

    private protected override void Detach(Storage storage) => ((VulkanStorage)storage).Block = null;

    private protected override void Attach(Storage storage, Storage fresh) =>
        ((VulkanStorage)storage).Block = ((VulkanStorage)fresh).Block;     // the fresh storage object is dropped, its block kept

    public override MemoryUsage GetMemoryUsage() => _memory.Usage;

    /// <summary>Frees every cached (currently unused) block, after the work queued on them has finished.</summary>
    public override void ReleaseCachedMemory()
    {
        lock (_gate)
        {
            SubmitAndWait();
            lock (_pool)
            {
                foreach (var (length, bucket) in _pool)
                {
                    while (bucket.Count > 0)
                    {
                        DestroyBlock(bucket.Pop());
                        _memory.Freed(BlockBytes(length));
                    }
                }

                _pool.Clear();
            }
        }
    }

    // Copies. Mapped storages are read and written in place once the queued work using them has finished; others go
    // through the staging buffer with copies recorded in queue order.

    public override void Upload(ReadOnlySpan<float> source, Storage destination)
    {
        var block = BlockOf(destination);
        if (source.Length > destination.Length)
        {
            throw new ArgumentException($"Uploading {source.Length} floats into a storage of {destination.Length}.", nameof(source));
        }

        lock (_gate)
        {
            if (block.Mapped != null)
            {
                WaitFor(block);
                source.CopyTo(new Span<float>(block.Mapped, source.Length));
                if (!_storageCoherent)
                {
                    FlushOrInvalidate(block.Memory, flush: true);
                }

                return;
            }

            var staging = Staging();
            for (long done = 0; done < source.Length;)
            {
                int count = (int)Math.Min(source.Length - done, StagingBytes / sizeof(float));
                source.Slice((int)done, count).CopyTo(new Span<float>(staging.Mapped, count));
                if (!_stagingCoherent)
                {
                    FlushOrInvalidate(staging.Memory, flush: true);
                }

                RecordCopy(staging, 0, block, done * sizeof(float), count * sizeof(float));
                SubmitAndWait();                                               // the staging buffer is reused next
                done += count;
            }
        }
    }

    public override void Download(Storage source, Span<float> destination) => DownloadRange(source, 0, destination);

    public override void DownloadRange(Storage source, int offset, Span<float> destination)
    {
        var block = BlockOf(source);
        if ((uint)offset > (uint)source.Length || destination.Length > source.Length - offset)
        {
            throw new ArgumentOutOfRangeException(nameof(offset), $"Floats {offset} to {offset + destination.Length} of a storage of {source.Length}.");
        }

        lock (_gate)
        {
            if (block.Mapped != null)
            {
                WaitFor(block);
                if (!_storageCoherent)
                {
                    FlushOrInvalidate(block.Memory, flush: false);
                }

                new ReadOnlySpan<float>(block.Mapped + (long)offset * sizeof(float), destination.Length).CopyTo(destination);
                return;
            }

            var staging = Staging();
            for (long done = 0; done < destination.Length;)
            {
                int count = (int)Math.Min(destination.Length - done, StagingBytes / sizeof(float));
                RecordCopy(block, (offset + done) * sizeof(float), staging, 0, count * sizeof(float));
                SubmitAndWait();
                if (!_stagingCoherent)
                {
                    FlushOrInvalidate(staging.Memory, flush: false);
                }

                new ReadOnlySpan<float>(staging.Mapped, count).CopyTo(destination.Slice((int)done, count));
                done += count;
            }
        }
    }

    private VulkanBlock Staging() =>
        _staging ??= CreateBlock((int)(StagingBytes / sizeof(float)), (uint)_stagingType, map: true);

    // Makes host writes visible to the device (flush) or device writes visible to the host (invalidate) on memory that
    // is not host-coherent; the whole allocation, so no alignment to nonCoherentAtomSize is needed.
    private void FlushOrInvalidate(ulong memory, bool flush)
    {
        var range = new VkMappedMemoryRange { SType = StructureMappedMemoryRange, Memory = memory, Offset = 0, Size = WholeSize };
        Check(flush ? vkFlushMappedMemoryRanges(_device, 1, &range) : vkInvalidateMappedMemoryRanges(_device, 1, &range),
            flush ? nameof(vkFlushMappedMemoryRanges) : nameof(vkInvalidateMappedMemoryRanges));
    }

    // Zeros a block: in place when it is mapped and idle, else with a fill in queue order.
    private void Zero(VulkanBlock block)
    {
        lock (_gate)
        {
            if (block.Mapped != null && block.LastUse <= _completed)
            {
                new Span<float>(block.Mapped, Math.Max(block.Capacity, 1)).Clear();

                if (!_storageCoherent)
                {
                    FlushOrInvalidate(block.Memory, flush: true);
                }

                return;
            }

            RecordFill(block);
        }
    }

    public override void Synchronize()
    {
        lock (_gate)
        {
            SubmitAndWait();
        }
    }

    /// <summary>Waits for the device, then frees everything it holds (for backends made by <see cref="CreateSeparate"/>; storages must be released first).</summary>
    internal void Shutdown()
    {
        lock (_gate)
        {
            SubmitAndWait();
            ReleaseCachedMemory();
            if (_staging is not null)
            {
                DestroyBlock(_staging);
                _staging = null;
            }

            StopQueue();
            vkDestroyDevice(_device, null);
        }
    }
}

/// <summary>
/// Vulkan devices: listed unless another backend drives them better (NVIDIA GPUs when CUDA is available) or they are
/// software drivers (lavapipe, SwiftShader: reached by name, e.g. for tests); discrete GPUs preferred to integrated ones.
/// </summary>
internal sealed class VulkanProvider : DeviceProvider
{
    private const uint VendorNvidia = 0x10DE;

    public override string Kind => "vulkan";

    public override string Display => "Vulkan";

    public override DeviceType Type => DeviceType.Vulkan;

    public override int Count => VulkanBackend.DeviceCount;

    public override string? UnavailableReason => VulkanBackend.UnavailableReason;

    public override Backend Create(int ordinal) => VulkanBackend.Get(ordinal);

    public override bool IsStarted(int ordinal) => VulkanBackend.IsInitialized(ordinal);

    public override bool Listed(int ordinal)
    {
        var (type, vendor) = VulkanBackend.DeviceKind(ordinal);
        return type != VulkanDriver.DeviceTypeCpu && !(vendor == VendorNvidia && Cuda.CudaBackend.DeviceCount > 0);
    }

    public override int? DefaultRank(int ordinal) => VulkanBackend.DeviceKind(ordinal).Type switch
    {
        VulkanDriver.DeviceTypeDiscreteGpu => 50,
        VulkanDriver.DeviceTypeIntegratedGpu => 40,
        _ => null,
    };
}
