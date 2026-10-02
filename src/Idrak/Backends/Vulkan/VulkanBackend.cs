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
/// Memory: storages live in one of the memories the device reports a storage could use (<see cref="StorageCandidates(VkPhysicalDeviceMemoryProperties, uint, uint)"/>):
/// device-local memory reached through a host-visible staging buffer, device-local memory the host maps (when about all
/// of it is mappable), or, on devices sharing the system's memory, host-visible system memory, host-cached or
/// write-combined, mapped. Which one is measured on the device when the backend starts and cached per device, driver and
/// power source (VulkanBackend.StorageMemory.cs); mapped storages are mapped once, so uploads and downloads are plain
/// copies (reads go through a host-cached staging buffer when that memory is not host-cached). IDRAK_VULKAN_STAGING=1
/// copies through staging everywhere; IDRAK_VULKAN_STORAGE names a memory.
///
/// Devices are numbered (vulkan:0, vulkan:1, …) by what they report, not the loader's order (<see cref="DeviceOrder"/>).
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

    // Memory types: where storages live, and (without mapped storages, or for reads of uncached mapped storages) the
    // staging buffer's.
    private uint _storageType;
    private bool _storageCoherent;
    private readonly int _stagingType;
    private readonly bool _stagingCoherent;

    // IDRAK_VULKAN_PAGE_BYTES or a test's page size (null: from the storage heap).
    private readonly long? _pageSetting;

    // Cached blocks by length (floats); their memory stays carved from its page until ReleaseCachedMemory.
    private readonly Dictionary<int, Stack<VulkanBlock>> _pool = [];

    // The staging buffer, created on first use; copies larger than it go in chunks.
    private VulkanBlock? _staging;

    // Whether dispatches push their descriptors (decided once the queue runs: VulkanBackend.Tuning.cs).
    private bool _usePush;

    private VulkanBackend(PhysicalDevice physical, bool preferMapped, bool? pushDescriptors = null, int? maxAllocations = null,
        long? stagingBytes = null, long? pageBytes = null)
    {
        _physical = physical;
        var p = physical.Properties;
        MaxAllocations = MemoryAllocationCap(p.MaxMemoryAllocationCount, maxAllocations);
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

        // Push descriptors are enabled wherever the device has them: a dispatch may then write its storages into the
        // command buffer instead of allocating and updating a descriptor set (recording a dispatch on the host: 9.3 → 0.7
        // µs, measured with lavapipe). Whether dispatches do is measured once the queue runs (VulkanBackend.Tuning.cs);
        // IDRAK_VULKAN_PUSH_DESCRIPTORS = 1 or 0 decides instead.
        pushDescriptors ??= Environment.GetEnvironmentVariable("IDRAK_VULKAN_PUSH_DESCRIPTORS") switch
        {
            "1" or "true" => true,
            "0" or "false" => false,
            _ => null,
        };
        bool push = HasExtension(physical.Handle, PushDescriptorExtension);
        fixed (byte* extensionName = "VK_KHR_push_descriptor\0"u8)
        {
            byte* extension = extensionName;
            var deviceInfo = new VkDeviceCreateInfo
            {
                SType = StructureDeviceCreateInfo,
                QueueCreateInfoCount = 1,
                QueueCreateInfos = &queueInfo,
                EnabledExtensionCount = push ? 1u : 0u,
                EnabledExtensionNames = &extension,
            };
            Check(vkCreateDevice(physical.Handle, &deviceInfo, null, out _device), nameof(vkCreateDevice));
        }

        vkGetDeviceQueue(_device, physical.QueueFamily, 0, out _queue);
        if (push)
        {
            fixed (byte* name = "vkCmdPushDescriptorSetKHR\0"u8)
            {
                _pushDescriptorSet = (delegate* unmanaged<IntPtr, uint, ulong, uint, uint, VkWriteDescriptorSet*, void>)vkGetDeviceProcAddr(_device, name);
            }
        }

        // Every storage buffer has the same memory requirements' type bits, so one probe buffer tells them.
        uint typeBits = StorageTypeBits();
        var memory = physical.Memory;

        // Staging memory (copies to device memory, reads of mapped memory the host does not cache): host-visible;
        // host-cached first (the host reads it fast), then coherent, then outside the device. Created on first use.
        _stagingType = BestType(memory, typeBits, MemoryHostVisible, static f =>
            ((f & MemoryHostCached) != 0 ? 4 : 0) + ((f & MemoryHostCoherent) != 0 ? 2 : 0) + ((f & MemoryDeviceLocal) == 0 ? 1 : 0));
        if (_stagingType < 0)
        {
            throw new VulkanException($"{Name} has no host-visible memory for copies.");
        }

        ulong maxAllocation = physical.Facts.MaxMemoryAllocationSize;
        _stagingCoherent = (memory.TypeFlags(_stagingType) & MemoryHostCoherent) != 0;
        StagingBytes = StagingSize(memory.HeapSize(memory.TypeHeap(_stagingType)), maxAllocation, p.NonCoherentAtomSize,
            stagingBytes ?? BytesSetting("IDRAK_VULKAN_STAGING_BYTES"));

        // Storage memory: one of the candidates the device reports, measured once the queue runs unless an override,
        // the cache or a single candidate decides it (VulkanBackend.StorageMemory.cs).
        _pageSetting = pageBytes ?? BytesSetting("IDRAK_VULKAN_PAGE_BYTES");
        _storageCandidates = [.. StorageCandidates(memory, typeBits, physical.Facts.DeviceType)];
        var (decided, how, timings) = DecideStorage(_storageCandidates, preferMapped ? null : StorageMemory.Staging,
            Environment.GetEnvironmentVariable("IDRAK_VULKAN_STORAGE"), CachedChoices());
        UseStorage(_storageCandidates.First(c => c.Kind == (decided ?? _storageCandidates[0].Kind)));
        (StorageChoice, StorageTimings) = (how, timings);
        StartQueue();
        TuneRuntime(pushDescriptors);
        TuneStorage(decided);
    }

    /// <summary>Whether dispatches push their descriptors (VK_KHR_push_descriptor) instead of allocating descriptor sets.</summary>
    public bool PushDescriptors => _usePush;

    /// <summary>Whether downloads of mapped storages go through a host-cached staging buffer (the mapped memory is not host-cached).</summary>
    public bool ReadsThroughStaging { get; private set; }

    /// <summary>The size of the memory heap storages live on (pages are sized from it).</summary>
    public long StorageHeapBytes { get; private set; }

    /// <summary>The staging buffer's size (copies larger than it go in chunks; the buffer is created on first use).</summary>
    public long StagingBytes { get; }

    // The staging buffer's size: the power of two at most 1/512 of the heap it lives on (16 MiB of an 8 GiB heap: chunks
    // large enough that a copy's fixed cost is small next to moving the chunk, memory held for good small next to the
    // heap), at most the largest allocation, at least nonCoherentAtomSize; `setting` (tests, IDRAK_VULKAN_STAGING_BYTES)
    // instead when given.
    internal static long StagingSize(ulong heapBytes, ulong maxAllocation, ulong atom, long? setting)
    {
        if (setting is long s and > 0)
        {
            return Math.Max(s / sizeof(float) * sizeof(float), sizeof(float));
        }

        ulong limit = heapBytes / 512;
        if (maxAllocation > 0)
        {
            limit = Math.Min(limit, maxAllocation);
        }

        long size = (long)Math.Max(PowerOfTwoAtMost(limit), Math.Max(atom, sizeof(float)));
        return size;
    }

    // The largest power of two at most `value` (1 for 0).
    private static ulong PowerOfTwoAtMost(ulong value) => value == 0 ? 1 : 1UL << (63 - System.Numerics.BitOperations.LeadingZeroCount(value));

    // vkCmdPushDescriptorSetKHR, or null without the extension.
    private readonly delegate* unmanaged<IntPtr, uint, ulong, uint, uint, VkWriteDescriptorSet*, void> _pushDescriptorSet;

    // Whether the device offers the extension `name`.
    private static bool HasExtension(IntPtr physical, string name)
    {
        uint count = 0;
        if (vkEnumerateDeviceExtensionProperties(physical, null, ref count, null) != Success || count == 0)
        {
            return false;
        }

        var properties = new VkExtensionProperties[count];
        fixed (VkExtensionProperties* p = properties)
        {
            if (vkEnumerateDeviceExtensionProperties(physical, null, ref count, p) is not (Success or Incomplete))
            {
                return false;
            }

            for (int i = 0; i < (int)count; i++)
            {
                if (Utf8(p[i].ExtensionName, 256) == name)
                {
                    return true;
                }
            }
        }

        return false;
    }

    /// <summary>Whether storages live in memory the host maps directly: uploads need no staging.</summary>
    public bool UnifiedMemory { get; private set; }

    /// <summary>The largest storage a kernel can bind (maxStorageBufferRange); larger ones take the host fallback.</summary>
    public long MaxStorageBytes { get; }

    public override string Name { get; }

    public override BackendCapabilities Capabilities { get; } = CpuBackend.Instance.Capabilities with
    {
        DecodeAttentionHeadDim = Math.Min(CpuBackend.Instance.Capabilities.DecodeAttentionHeadDim, VulkanKernels.AttentionMaxDim),
        TiledAttentionHeadDim = Math.Min(CpuBackend.Instance.Capabilities.TiledAttentionHeadDim, VulkanKernels.AttentionMaxDim),
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

    /// <summary>Device <paramref name="ordinal"/>'s type (VK_PHYSICAL_DEVICE_TYPE_*) and UUID (deviceUUID), without starting it.</summary>
    public static (uint Type, Guid Uuid) DeviceKind(int ordinal) =>
        (Probe.Value.Devices[ordinal].Properties.DeviceType, Probe.Value.Devices[ordinal].Facts.DeviceUuid);

    /// <summary>What orders device <paramref name="ordinal"/> among the others (<see cref="DeviceOrder"/>).</summary>
    internal static DeviceOrderKey OrderKey(int ordinal)
    {
        var d = Probe.Value.Devices[ordinal];
        return new DeviceOrderKey(d.Facts.DeviceType, d.Facts.DeviceUuid, d.Facts.DriverUuid, d.Facts.PciAddress, d.DeviceName);
    }

    /// <summary>Device <paramref name="ordinal"/>'s reported facts, without starting it.</summary>
    internal static VulkanDeviceFacts FactsOf(int ordinal) => Probe.Value.Devices[ordinal].Facts;

    /// <summary>
    /// A second backend on device <paramref name="ordinal"/> (for tests: staging copies even where memory is shared; pushed
    /// descriptors or descriptor sets whatever the device's choice; a given staging buffer or page size).
    /// </summary>
    internal static VulkanBackend CreateSeparate(int ordinal, bool preferMapped, bool? pushDescriptors = null, int? maxAllocations = null,
        long? stagingBytes = null, long? pageBytes = null) =>
        new(Probe.Value.Devices[ordinal], preferMapped, pushDescriptors, maxAllocations, stagingBytes, pageBytes);

    private static Lazy<VulkanBackend>[] CreateInstances()
    {
        bool staging = Environment.GetEnvironmentVariable("IDRAK_VULKAN_STAGING") is "1" or "true";
        return [.. Probe.Value.Devices.Select(d => new Lazy<VulkanBackend>(() => new VulkanBackend(d, preferMapped: !staging)))];
    }

    // A physical device with what the backend reads of it.
    private sealed class PhysicalDevice(IntPtr handle, VkPhysicalDeviceProperties properties, string deviceName, string driver,
        VkPhysicalDeviceMemoryProperties memory, uint queueFamily, VulkanDeviceFacts facts)
    {
        public VulkanDeviceFacts Facts { get; } = facts;

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
                var (facts, driver) = QueryFacts(handle, properties, apiVersion);
                devices.Add(new PhysicalDevice(handle, properties, deviceName, driver, memory, family, facts));
            }

            int[] order = DeviceOrder([.. devices.Select(d => new DeviceOrderKey(d.Facts.DeviceType, d.Facts.DeviceUuid, d.Facts.DriverUuid,
                d.Facts.PciAddress, d.DeviceName))]);
            return devices.Count > 0 ? ([.. order.Select(i => devices[i])], "")
                : ([], skipped.Count > 0 ? $"no usable Vulkan device ({string.Join("; ", skipped)})" : "the Vulkan driver reports no devices");
        }
        catch (Exception ex) when (ex is DllNotFoundException or EntryPointNotFoundException or VulkanException)
        {
            return ([], ex.Message);
        }
    }

    /// <summary>What orders a device among the others (<see cref="DeviceOrder"/>): all of it reported by the device.</summary>
    internal readonly record struct DeviceOrderKey(uint Type, Guid DeviceUuid, Guid DriverUuid,
        (uint Domain, uint Bus, uint Device, uint Function)? PciAddress, string Name);

    /// <summary>
    /// The order of the devices (indices into <paramref name="devices"/>, as the loader enumerated them) that vulkan:0,
    /// vulkan:1, … name. The loader's order is not stable across processes (on a laptop with two GPUs it changes with
    /// power state), so ordinals come from what each device reports instead: its type (discrete, integrated, virtual,
    /// CPU, other), then its deviceUUID's bytes, then its driverUUID's (one GPU through two drivers); a device without a
    /// UUID (all zeros) comes after those with one, by PCI address (when reported), then name. The loader's order only
    /// breaks what is left (identical reports). Never the vendor or the card.
    /// </summary>
    internal static int[] DeviceOrder(IReadOnlyList<DeviceOrderKey> devices)
    {
        int[] order = [.. Enumerable.Range(0, devices.Count)];
        Array.Sort(order, (a, b) => Compare(devices[a], devices[b]) is var c and not 0 ? c : a.CompareTo(b));
        return order;

        static int Compare(DeviceOrderKey a, DeviceOrderKey b)
        {
            int c = TypeRank(a.Type).CompareTo(TypeRank(b.Type));
            if (c != 0)
            {
                return c;
            }

            bool hasA = a.DeviceUuid != Guid.Empty, hasB = b.DeviceUuid != Guid.Empty;
            if (hasA != hasB)
            {
                return hasA ? -1 : 1;
            }

            if (hasA && (c = CompareBytes(a.DeviceUuid, b.DeviceUuid)) != 0)
            {
                return c;
            }

            if (!hasA)
            {
                if (a.PciAddress.HasValue != b.PciAddress.HasValue)
                {
                    return a.PciAddress.HasValue ? -1 : 1;
                }

                if (a.PciAddress is { } pa && b.PciAddress is { } pb && (c = pa.CompareTo(pb)) != 0)
                {
                    return c;
                }

                if ((c = string.CompareOrdinal(a.Name, b.Name)) != 0)
                {
                    return c;
                }
            }

            return CompareBytes(a.DriverUuid, b.DriverUuid) is var d and not 0 ? d : string.CompareOrdinal(a.Name, b.Name);
        }

        static int TypeRank(uint type) => type switch
        {
            DeviceTypeDiscreteGpu => 0,
            DeviceTypeIntegratedGpu => 1,
            DeviceTypeVirtualGpu => 2,
            DeviceTypeCpu => 3,
            _ => 4,
        };

        // The UUIDs' bytes as the device reported them (Guid's own order compares its fields, not its bytes).
        static int CompareBytes(Guid a, Guid b)
        {
            Span<byte> x = stackalloc byte[16], y = stackalloc byte[16];
            a.TryWriteBytes(x);
            b.TryWriteBytes(y);
            return x.SequenceCompareTo(y);
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

    private static string Utf8(byte* text, int capacity)
    {
        int length = 0;
        while (length < capacity && text[length] != 0)
        {
            length++;
        }

        return Encoding.UTF8.GetString(text, length).Trim();
    }

    // The memory type of `typeBits` with all of `required` and none of `forbidden` on a heap of at least `minHeap` bytes, best by `score` then by
    // heap size; -1 when none.
    private static int BestType(VkPhysicalDeviceMemoryProperties memory, uint typeBits, uint required, Func<uint, int> score, ulong minHeap = 0,
        uint forbidden = 0)
    {
        int best = -1;
        (int Score, ulong Heap) bestKey = (int.MinValue, 0);
        for (int i = 0; i < (int)memory.MemoryTypeCount; i++)
        {
            uint flags = memory.TypeFlags(i);
            if ((typeBits & (1u << i)) == 0 || (flags & required) != required || (flags & forbidden) != 0 || memory.HeapSize(memory.TypeHeap(i)) < minHeap)
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

            FreeEmptyPages();
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
                    FlushOrInvalidate(block, flush: true);
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
                    FlushOrInvalidate(staging, flush: true);
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
            if (block.Mapped != null && !ReadsThroughStaging)
            {
                WaitFor(block);
                if (!_storageCoherent)
                {
                    FlushOrInvalidate(block, flush: false);
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
                    FlushOrInvalidate(staging, flush: false);
                }

                new ReadOnlySpan<float>(staging.Mapped, count).CopyTo(destination.Slice((int)done, count));
                done += count;
            }
        }
    }

    private VulkanBlock Staging() =>
        _staging ??= CreateBlock((int)(StagingBytes / sizeof(float)), (uint)_stagingType, map: true, dedicated: true);

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
                    FlushOrInvalidate(block, flush: true);
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

            FreeEmptyPages();

            StopQueue();
            vkDestroyDevice(_device, null);
        }
    }
}

/// <summary>
/// Vulkan devices: listed unless another provider registered before this one reaches the same GPU (the same deviceUUID:
/// CUDA, when its driver is there) or they are CPU-type devices (software drivers: reached by name, e.g. for tests);
/// discrete GPUs preferred to integrated ones. Only what the device reports decides: its type and its UUID.
/// </summary>
internal sealed class VulkanProvider : DeviceProvider
{
    public override string Kind => "vulkan";

    public override string Display => "Vulkan";

    public override DeviceType Type => DeviceType.Vulkan;

    public override int Count => VulkanBackend.DeviceCount;

    public override string? UnavailableReason => VulkanBackend.UnavailableReason;

    public override Backend Create(int ordinal) => VulkanBackend.Get(ordinal);

    public override bool IsStarted(int ordinal) => VulkanBackend.IsInitialized(ordinal);

    public override Guid? DeviceUuid(int ordinal) => VulkanBackend.DeviceKind(ordinal).Uuid;

    public override bool Listed(int ordinal) =>
        VulkanBackend.DeviceKind(ordinal).Type != VulkanDriver.DeviceTypeCpu && DrivenElsewhere(ordinal) is null;

    // Not chosen by Device.Default until its kernels are tuned (a CPU can still beat an untuned integrated GPU);
    // IDRAK_VULKAN_DEFAULT=1 opts in. Reached by name ("vulkan:1") either way.
    public override int? DefaultRank(int ordinal) => Environment.GetEnvironmentVariable("IDRAK_VULKAN_DEFAULT") != "1" ? null
        : VulkanBackend.DeviceKind(ordinal).Type switch
        {
            VulkanDriver.DeviceTypeDiscreteGpu => 50,
            VulkanDriver.DeviceTypeIntegratedGpu => 40,
            _ => null,
        };

    public override string? Note(int ordinal) =>
        VulkanBackend.DeviceKind(ordinal).Type == VulkanDriver.DeviceTypeCpu ? "software driver: by name only"
        : DrivenElsewhere(ordinal) is var (display, name) ? $"driven by {display} as {name}: by name only"
        : DefaultRank(ordinal) is null ? "not the default device (IDRAK_VULKAN_DEFAULT=1 to prefer it)" : null;

    // The provider (display name) and device ("cuda:0") registered before this provider that reach the same GPU, or null.
    private (string Display, string Device)? DrivenElsewhere(int ordinal)
    {
        var all = DeviceProviders.All;
        int self = -1;
        for (int i = 0; i < all.Count && self < 0; i++)
        {
            self = ReferenceEquals(all[i], this) ? i : -1;
        }

        return DrivenBy(DeviceUuid(ordinal), self >= 0 ? all.Take(self) : all.Where(p => !ReferenceEquals(p, this)));
    }

    /// <summary>
    /// The first device of <paramref name="others"/> with UUID <paramref name="uuid"/> (its provider's display name and
    /// its device name), or null: the same physical GPU, reached through another API.
    /// </summary>
    internal static (string Display, string Device)? DrivenBy(Guid? uuid, IEnumerable<DeviceProvider> others)
    {
        if (uuid is not Guid id || id == Guid.Empty)
        {
            return null;
        }

        foreach (var provider in others)
        {
            int count = provider.Count;
            for (int i = 0; i < count; i++)
            {
                if (provider.DeviceUuid(i) == id)
                {
                    return (provider.Display, $"{provider.Kind}:{i}");
                }
            }
        }

        return null;
    }
}
