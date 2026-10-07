// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Diagnostics;
using System.Globalization;
using static Idrak.Gpu.Vulkan.VulkanDriver;

namespace Idrak.Gpu.Vulkan;

/// <summary>Where storages live and how the host reaches them (<see cref="VulkanBackend.StorageCandidates(VkPhysicalDeviceMemoryProperties, uint, uint)"/>).</summary>
internal enum StorageMemory
{
    /// <summary>Device-local memory the host does not see (or the device's only memory); copies through a staging buffer.</summary>
    Staging,

    /// <summary>Device-local, host-visible memory on a heap about as large as the device's memory, mapped once.</summary>
    MappedDevice,

    /// <summary>Host-visible, host-cached memory on the largest host-visible heap (shared-memory devices), mapped once.</summary>
    MappedCached,

    /// <summary>Host-visible memory that is not host-cached (write-combined) on the largest host-visible heap (shared-memory devices), mapped once.</summary>
    MappedUncached,
}

// Which memory storages live in is measured, not assumed. A device may report several memory types a storage could
// use, and which one its kernels read fastest is the device's own matter: an integrated GPU (measured on a laptop APU)
// read its device-local carve-out at 39 GB/s in an element-wise kernel but host-cached (snooped) system memory at 27,
// while that cached memory took uploads twice as fast. So the valid candidates (StorageCandidates) are each measured
// when the backend starts: a few-MiB element-wise kernel's bandwidth on the device, and upload and download bandwidth on
// the host, medians of rounds after a warm-up. The cost model (ChooseStorage) favors the device's reads, since decoding
// is bound by the device reading weights while each weight is uploaded once: the candidate the kernel runs fastest on,
// unless another runs within 3% of it (TuneMargin) and uploads faster. The choice and the measured numbers are cached
// with the other runtime choices (VulkanBackend.Tuning.cs: per device, driver and power source).
// IDRAK_VULKAN_STORAGE=staging|mapped-device|mapped-cached|mapped-uncached overrides it (when the device has that
// candidate), as does IDRAK_VULKAN_STAGING=1 (staging).
internal sealed unsafe partial class VulkanBackend
{
    /// <summary>A memory storages could live in: how the host reaches it, and the memory type.</summary>
    internal readonly record struct StorageCandidate(StorageMemory Kind, int Type);

    /// <summary>What one candidate measured: the device's element-wise bandwidth and the host's upload and download bandwidth, GB/s.</summary>
    internal readonly record struct StorageTiming(StorageMemory Kind, double Device, double Upload, double Download);

    private StorageCandidate[] _storageCandidates = [];

    /// <summary>The memory storages live in.</summary>
    public StorageMemory StorageKind { get; private set; }

    /// <summary>How <see cref="StorageKind"/> was chosen: "measured", "cached", "override" or "only candidate".</summary>
    public string StorageChoice { get; private set; } = "measured";

    /// <summary>The candidates' measured numbers (from this process or the cache; empty when not measured).</summary>
    internal IReadOnlyList<StorageTiming> StorageTimings { get; private set; } = [];

    /// <summary>The candidates the device offers, in <see cref="StorageMemory"/> order.</summary>
    internal IReadOnlyList<StorageCandidate> StorageCandidateList => _storageCandidates;

    /// <summary>The name of a storage memory (the IDRAK_VULKAN_STORAGE values).</summary>
    internal static string StorageName(StorageMemory kind) => kind switch
    {
        StorageMemory.Staging => "staging",
        StorageMemory.MappedDevice => "mapped-device",
        StorageMemory.MappedCached => "mapped-cached",
        _ => "mapped-uncached",
    };

    /// <summary>The storage memory a name (IDRAK_VULKAN_STORAGE) names, or null.</summary>
    internal static StorageMemory? ParseStorage(string? name) => name?.Trim().ToLowerInvariant() switch
    {
        "staging" => StorageMemory.Staging,
        "mapped-device" => StorageMemory.MappedDevice,
        "mapped-cached" => StorageMemory.MappedCached,
        "mapped-uncached" => StorageMemory.MappedUncached,
        _ => null,
    };

    // Whether a device shares the system's memory, by its reported type: an integrated GPU or a CPU driver. Only those
    // reach host memory without crossing a bus, so only they get host-memory candidates.
    private static bool SharesSystemMemory(uint deviceType) => deviceType is DeviceTypeIntegratedGpu or DeviceTypeCpu;

    /// <summary>
    /// The memories storages could live in, from the memory types and heaps the device reports (`typeBits`: the types a
    /// storage buffer allows), in <see cref="StorageMemory"/> order, without two of the same memory type among the mapped
    /// ones (staging stays: it reaches its memory another way):
    /// - staging: device-local memory the host does not see, first (else any device-local type, else any type);
    /// - mapped-device: a device-local, host-visible type on a heap at least half the size of the largest device-local
    ///   heap (the host maps about all of the device's memory, as on a discrete GPU with resizable BAR or an integrated
    ///   GPU with one heap; not a small window of it), host-cached first, then coherent;
    /// - on devices sharing the system's memory (integrated GPUs, CPU drivers, by reported type) also host-visible memory
    ///   on a heap at least half the size of the largest host-visible heap: mapped-cached (host-cached; device-local first)
    ///   and mapped-uncached (not host-cached: write-combined; coherent first). A discrete GPU would read those across
    ///   the bus.
    /// </summary>
    internal static List<StorageCandidate> StorageCandidates(VkPhysicalDeviceMemoryProperties memory, uint typeBits, uint deviceType)
    {
        var candidates = new List<StorageCandidate>();
        void Add(StorageMemory kind, int type)
        {
            if (type >= 0 && (kind == StorageMemory.Staging || !candidates.Any(c => c.Kind != StorageMemory.Staging && c.Type == type)))
            {
                candidates.Add(new StorageCandidate(kind, type));
            }
        }

        int local = BestType(memory, typeBits, MemoryDeviceLocal, static f => (f & MemoryHostVisible) == 0 ? 1 : 0);
        Add(StorageMemory.Staging, local >= 0 ? local : BestType(memory, typeBits, 0, static _ => 0));

        ulong largestLocal = 0;
        for (int heap = 0; heap < (int)memory.MemoryHeapCount; heap++)
        {
            if ((memory.HeapFlags(heap) & HeapDeviceLocal) != 0)
            {
                largestLocal = Math.Max(largestLocal, memory.HeapSize(heap));
            }
        }

        if (largestLocal > 0)
        {
            Add(StorageMemory.MappedDevice, BestType(memory, typeBits, MemoryDeviceLocal | MemoryHostVisible, static f =>
                ((f & MemoryHostCached) != 0 ? 2 : 0) + ((f & MemoryHostCoherent) != 0 ? 1 : 0), minHeap: largestLocal / 2));
        }

        if (SharesSystemMemory(deviceType))
        {
            ulong visible = 0;
            for (int i = 0; i < (int)memory.MemoryTypeCount; i++)
            {
                if ((typeBits & (1u << i)) != 0 && (memory.TypeFlags(i) & MemoryHostVisible) != 0)
                {
                    visible = Math.Max(visible, memory.HeapSize(memory.TypeHeap(i)));
                }
            }

            Add(StorageMemory.MappedCached, BestType(memory, typeBits, MemoryHostVisible | MemoryHostCached, static f =>
                ((f & MemoryDeviceLocal) != 0 ? 2 : 0) + ((f & MemoryHostCoherent) != 0 ? 1 : 0), minHeap: visible / 2));
            Add(StorageMemory.MappedUncached, BestType(memory, typeBits, MemoryHostVisible, static f =>
                ((f & MemoryHostCoherent) != 0 ? 2 : 0) + ((f & MemoryDeviceLocal) != 0 ? 1 : 0), minHeap: visible / 2, forbidden: MemoryHostCached));
        }

        return candidates;
    }

    /// <summary>The candidates for the memory types (flags, heap) and heaps (size, flags) given (tests: a device's report made up).</summary>
    internal static List<StorageCandidate> StorageCandidates(ReadOnlySpan<(uint Flags, int Heap)> types, ReadOnlySpan<(ulong Size, ulong Flags)> heaps,
        uint deviceType) => StorageCandidates(MemoryReport(types, heaps), uint.MaxValue, deviceType);

    /// <summary>A memory report made of the types (flags, heap) and heaps (size, flags) given (tests).</summary>
    internal static VkPhysicalDeviceMemoryProperties MemoryReport(ReadOnlySpan<(uint Flags, int Heap)> types, ReadOnlySpan<(ulong Size, ulong Flags)> heaps)
    {
        var memory = new VkPhysicalDeviceMemoryProperties { MemoryTypeCount = (uint)types.Length, MemoryHeapCount = (uint)heaps.Length };
        for (int i = 0; i < types.Length; i++)
        {
            (memory.MemoryTypes[2 * i], memory.MemoryTypes[2 * i + 1]) = (types[i].Flags, (uint)types[i].Heap);
        }

        for (int i = 0; i < heaps.Length; i++)
        {
            (memory.MemoryHeaps[2 * i], memory.MemoryHeaps[2 * i + 1]) = (heaps[i].Size, heaps[i].Flags);
        }

        return memory;
    }

    /// <summary>
    /// The cost model: the index of the timing to keep. The device reading storages matters most (decoding is bound by
    /// the device reading weights; each weight is uploaded once), so the candidate the kernel ran fastest on, unless
    /// another ran within 3% of it (<see cref="TuneMargin"/>) and uploads faster: among those, the fastest upload (ties:
    /// the earlier candidate). Downloads (activations, logits: small) do not count. -1 for none.
    /// </summary>
    internal static int ChooseStorage(IReadOnlyList<StorageTiming> timings)
    {
        int fastest = -1;
        for (int i = 0; i < timings.Count; i++)
        {
            if (timings[i].Device > 0 && (fastest < 0 || timings[i].Device > timings[fastest].Device))
            {
                fastest = i;
            }
        }

        if (fastest < 0)
        {
            return -1;
        }

        int chosen = fastest;
        for (int i = 0; i < timings.Count; i++)
        {
            if (timings[i].Device >= timings[fastest].Device * TuneMargin && timings[i].Upload > timings[chosen].Upload)
            {
                chosen = i;
            }
        }

        return chosen;
    }

    /// <summary>Timings as the cache keeps them: "kind=device/upload/download" separated by commas (GB/s).</summary>
    internal static string FormatTimings(IEnumerable<StorageTiming> timings) =>
        string.Join(",", timings.Select(t => string.Create(CultureInfo.InvariantCulture, $"{StorageName(t.Kind)}={t.Device:0.###}/{t.Upload:0.###}/{t.Download:0.###}")));

    /// <summary>Timings read back from <see cref="FormatTimings"/> (what does not parse is skipped).</summary>
    internal static List<StorageTiming> ParseTimings(string? text)
    {
        var timings = new List<StorageTiming>();
        foreach (string entry in (text ?? "").Split(',', StringSplitOptions.RemoveEmptyEntries))
        {
            string[] parts = entry.Split('=');
            string[] numbers = parts.Length == 2 ? parts[1].Split('/') : [];
            if (ParseStorage(parts[0]) is StorageMemory kind && numbers.Length == 3
                && double.TryParse(numbers[0], NumberStyles.Float, CultureInfo.InvariantCulture, out double device)
                && double.TryParse(numbers[1], NumberStyles.Float, CultureInfo.InvariantCulture, out double upload)
                && double.TryParse(numbers[2], NumberStyles.Float, CultureInfo.InvariantCulture, out double download))
            {
                timings.Add(new StorageTiming(kind, device, upload, download));
            }
        }

        return timings;
    }

    /// <summary>
    /// The storage memory to use before measuring and how it was decided: `forced` (IDRAK_VULKAN_STAGING, a test), else
    /// IDRAK_VULKAN_STORAGE when the device has it, else the cached choice when it is still a candidate, else a single
    /// candidate; null kind when it must be measured. Cached timings come back too.
    /// </summary>
    internal static (StorageMemory? Kind, string How, List<StorageTiming> Timings) DecideStorage(IReadOnlyList<StorageCandidate> candidates,
        StorageMemory? forced, string? setting, IReadOnlyDictionary<string, string> cached)
    {
        bool Has(StorageMemory kind) => candidates.Any(c => c.Kind == kind);
        var timings = ParseTimings(cached.GetValueOrDefault("storage-memory/timings"));
        if (forced is StorageMemory f && Has(f))
        {
            return (f, "override", timings);
        }

        if (ParseStorage(setting) is StorageMemory s && Has(s))
        {
            return (s, "override", timings);
        }

        if (ParseStorage(cached.GetValueOrDefault("storage-memory")) is StorageMemory c && Has(c))
        {
            return (c, "cached", timings);
        }

        return candidates.Count == 1 ? (candidates[0].Kind, "only candidate", []) : (null, "measured", []);
    }

    // Puts storages in `candidate`'s memory: the memory type, whether it is mapped (and read through the staging buffer
    // when not host-cached), and the page size from its heap. Only while no block is live (at start).
    private void UseStorage(StorageCandidate candidate)
    {
        var memory = _physical.Memory;
        uint flags = memory.TypeFlags(candidate.Type);
        StorageKind = candidate.Kind;
        _storageType = (uint)candidate.Type;
        _storageCoherent = (flags & MemoryHostCoherent) != 0;
        UnifiedMemory = candidate.Kind != StorageMemory.Staging;
        ReadsThroughStaging = UnifiedMemory && (flags & MemoryHostCached) == 0 && _stagingType >= 0
                              && (memory.TypeFlags(_stagingType) & MemoryHostCached) != 0;     // the host reads uncached memory slowly
        StorageHeapBytes = (long)memory.HeapSize(memory.TypeHeap(candidate.Type));
        PageBytes = PageSize((ulong)StorageHeapBytes, _physical.Facts.MaxMemoryAllocationSize, _physical.Properties.MaxMemoryAllocationCount, _pageSetting);
    }

    // Decides the storage memory once the queue runs (VulkanBackend.Tuning.cs's runtime policy first): the start-up
    // decision when there was one, else each candidate measured, the cost model's choice used and cached.
    private void TuneStorage(StorageMemory? decided)
    {
        if (decided is not null)
        {
            return;                                                        // already in use since the start
        }

        var timings = MeasureStorage();
        int chosen = ChooseStorage(timings);
        var pick = chosen >= 0 ? timings[chosen].Kind : _storageCandidates[0].Kind;
        UseStorage(_storageCandidates.First(c => c.Kind == pick));
        StorageTimings = timings;
        StorageChoice = "measured";
        if (chosen >= 0)
        {
            CacheChoices(new Dictionary<string, string> { ["storage-memory"] = StorageName(pick), ["storage-memory/timings"] = FormatTimings(timings) });
        }
    }

    // Measures every candidate in turn (each in use while measured): Rounds rounds after a warm-up, each an upload and a
    // download of a few-MiB storage and Runs element-wise additions over three such storages (two read, one
    // written), timed from the host with the device drained before and after; medians count. A candidate whose memory
    // runs out is left out. Staging over a memory type a mapped candidate also measured reuses its device number (the
    // kernel reads the same memory). Leaves no storage, page or staging buffer behind, and resets the dispatch counters.
    private List<StorageTiming> MeasureStorage()
    {
        const int Length = 1 << 20, Rounds = 5, Runs = 8;          // 4 MiB storages
        using var quiet = DeviceException.Handled();                // a candidate that fails is left out
        var data = new float[Length];
        for (int i = 0; i < Length; i++)
        {
            data[i] = i % 1000 * 1e-3f;
        }

        var timings = new List<StorageTiming>();
        var deviceByType = new Dictionary<int, double>();
        var order = _storageCandidates.Where(c => c.Kind != StorageMemory.Staging).Concat(_storageCandidates.Where(c => c.Kind == StorageMemory.Staging));
        foreach (var candidate in order)
        {
            UseStorage(candidate);
            Storage[] s = [];
            try
            {
                s = [Allocate(Length, zeroed: false), Allocate(Length, zeroed: false), Allocate(Length, zeroed: false)];
                List<double> device = [], upload = [], download = [];
                bool reuse = deviceByType.TryGetValue(candidate.Type, out double known);
                for (int round = -1; round < Rounds; round++)                // round -1 warms up, untimed
                {
                    Synchronize();
                    long start = Stopwatch.GetTimestamp();
                    Upload(data, s[0]);
                    Synchronize();
                    upload.Add(Stopwatch.GetElapsedTime(start).TotalSeconds);
                    Upload(data, s[1]);
                    if (!reuse)
                    {
                        Synchronize();
                        start = Stopwatch.GetTimestamp();
                        for (int d = 0; d < Runs; d++)
                        {
                            Binary(BinaryOp.Add, s[0], s[1], s[2], Length);
                        }

                        Synchronize();
                        device.Add(Stopwatch.GetElapsedTime(start).TotalSeconds);
                    }

                    start = Stopwatch.GetTimestamp();
                    Download(s[2], data);
                    download.Add(Stopwatch.GetElapsedTime(start).TotalSeconds);
                    if (round < 0)
                    {
                        (device, upload, download) = ([], [], []);
                    }
                }

                const double Bytes = (double)Length * sizeof(float), Giga = 1e9;
                double deviceRate = reuse ? known : 3 * Bytes * Runs / Math.Max(Median(device), 1e-9) / Giga;
                deviceByType.TryAdd(candidate.Type, deviceRate);
                timings.Add(new StorageTiming(candidate.Kind, Math.Round(deviceRate, 2), Math.Round(Bytes / Math.Max(Median(upload), 1e-9) / Giga, 2),
                    Math.Round(Bytes / Math.Max(Median(download), 1e-9) / Giga, 2)));
            }
            catch (Exception ex) when (ex is ResourceLimitExceededException or VulkanException)
            {
            }
            finally
            {
                foreach (var storage in s)
                {
                    storage.Release();
                }

                ReleaseCachedMemory();
                lock (_gate)
                {
                    if (_staging is not null)
                    {
                        DestroyBlock(_staging);
                        _staging = null;
                    }

                    (Dispatches, Submissions, Barriers) = (0, 0, 0);
                }
            }
        }

        // In candidate order (the cost model breaks ties by it).
        timings.Sort((a, b) => a.Kind.CompareTo(b.Kind));
        return timings;
    }

    /// <summary>The storage memory and how it was chosen, with the measured numbers (benchmark headers).</summary>
    internal string DescribeStorage()
    {
        string numbers = StorageTimings.Count == 0 ? "" : "; GB/s device/upload/download: " +
            string.Join(", ", StorageTimings.Select(t => string.Create(CultureInfo.InvariantCulture,
                $"{StorageName(t.Kind)}{(t.Kind == StorageKind ? "*" : "")} {t.Device:0.#}/{t.Upload:0.#}/{t.Download:0.#}")));
        return $"{StorageName(StorageKind)} ({StorageChoice} among {string.Join(", ", _storageCandidates.Select(c => $"{StorageName(c.Kind)}:{c.Type}"))}{numbers})";
    }
}
