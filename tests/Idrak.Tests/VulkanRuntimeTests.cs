// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Text.RegularExpressions;
using Idrak;
using Idrak.Backends;
using Idrak.Backends.Vulkan;

// The Vulkan runtime chooses by what the device reports or what is measured on it, never by vendor or card: devices
// another provider reaches are found by UUID, memory paths and sizes come from the reported heaps and limits, and the
// runtime policy (descriptor mode, batch size, batches in flight) is measured at start and kept per device and driver.
internal static partial class Tests
{
    private static readonly (string Name, Action<Device> Run)[] VulkanRuntimeGroup =
    [
        ("vulkan runtime: a GPU another provider reaches (same UUID) is found by UUID (first match; none for an empty or unknown UUID)", VulkanSameGpuByUuid),
        ("vulkan runtime: devices are numbered by what they report (type, then UUID), whatever order the loader lists them in", VulkanDeviceOrder),
        ("vulkan runtime: mapped memory, staging and page sizes come from the reported heaps and limits", VulkanSizesFromReport),
        ("vulkan runtime: storage memory candidates come from the reported heaps (host memory only on shared-memory devices); the cost model favors device reads", VulkanMappedByDeviceType),
        ("vulkan runtime: descriptor mode, batch size and batches in flight are measured at start, cached per device and driver, overridable", VulkanRuntimeMeasured),
        ("vulkan runtime: no vendor ids or card names in the code of src/Idrak/Backends/Vulkan (comments may say where something was measured)", VulkanNoVendorNames),
    ];

    // A provider whose devices report the UUIDs given.
    private sealed class FakeUuidProvider(string kind, params Guid?[] uuids) : DeviceProvider
    {
        public override string Kind => kind;

        public override string Display => kind.ToUpperInvariant();

        public override DeviceType Type => DeviceType.Cpu;

        public override int Count => uuids.Length;

        public override Backend Create(int ordinal) => throw new NotSupportedException();

        public override bool IsStarted(int ordinal) => false;

        public override Guid? DeviceUuid(int ordinal) => uuids[ordinal];
    }

    private static void VulkanSameGpuByUuid(Device device)
    {
        if (device.Type != DeviceType.Cpu)
        {
            return;                                                      // checked once
        }

        Guid a = Guid.NewGuid(), b = Guid.NewGuid(), c = Guid.NewGuid();
        DeviceProvider[] others = [new FakeUuidProvider("first", a, null, b), new FakeUuidProvider("second", c, b)];
        Check(VulkanProvider.DrivenBy(b, others) == ("FIRST", "first:2"), "the first provider reaching the GPU names it");
        Check(VulkanProvider.DrivenBy(c, others) == ("SECOND", "second:0"), "a later provider's device");
        Check(VulkanProvider.DrivenBy(Guid.NewGuid(), others) is null, "an unknown UUID: driven by nobody else");
        Check(VulkanProvider.DrivenBy(Guid.Empty, others) is null && VulkanProvider.DrivenBy(null, others) is null, "no UUID: never matched");
        Check(VulkanProvider.DrivenBy(a, []) is null, "no other provider");

        // The CUDA provider reports its devices' UUIDs (none without its driver), and a Vulkan device it reaches is
        // listed by name only, whatever its vendor.
        var cuda = DeviceProviders.Find("cuda")!;
        Check(cuda.Count == 0 || cuda.DeviceUuid(0) is not null, "CUDA devices report a UUID");
        for (int i = 0; i < VulkanBackend.DeviceCount; i++)
        {
            var vulkan = DeviceProviders.Find("vulkan")!;
            Check(vulkan.DeviceUuid(i) == VulkanBackend.DeviceKind(i).Uuid, $"vulkan:{i} reports its deviceUUID");
            var driver = VulkanProvider.DrivenBy(vulkan.DeviceUuid(i), [cuda]);
            Check(driver is null || (!vulkan.Listed(i) && vulkan.Note(i) == $"driven by CUDA as {driver.Value.Device}: by name only"),
                $"vulkan:{i}: driven by CUDA, so by name only");
        }
    }

    private static void VulkanDeviceOrder(Device device)
    {
        if (device.Type != DeviceType.Cpu)
        {
            return;
        }

        const uint Other = 0, Integrated = 1, Discrete = 2, Virtual = 3, Cpu = 4;
        static Guid Id(byte first) => new([first, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]);
        Guid driverA = Id(0xA0), driverB = Id(0xB0);
        VulkanBackend.DeviceOrderKey[] expected =
        [
            new(Discrete, Id(0x10), driverA, null, "gpu b"),             // discrete first, by UUID bytes
            new(Discrete, Id(0x90), driverA, null, "gpu a"),
            new(Integrated, Id(0x05), driverA, null, "igpu"),            // then integrated; one GPU through two drivers: by driver UUID
            new(Integrated, Id(0x05), driverB, null, "igpu"),
            new(Integrated, Guid.Empty, driverA, (0, 3, 0, 0), "z"),     // no UUID: after those with one, by PCI address
            new(Integrated, Guid.Empty, driverA, (0, 4, 0, 0), "a"),
            new(Integrated, Guid.Empty, driverA, null, "b"),             // then by name
            new(Virtual, Id(0x01), driverA, null, "virtual"),
            new(Cpu, Id(0x00), driverA, null, "software"),
            new(Other, Id(0x00), driverA, null, "other"),
        ];

        // Whatever order the loader lists them in (it changes between processes on a laptop with two GPUs), the same
        // numbering.
        var rng = new Random(7);
        for (int trial = 0; trial < 200; trial++)
        {
            var listed = expected.OrderBy(_ => rng.Next()).ToArray();
            int[] order = VulkanBackend.DeviceOrder(listed);
            Check(order.Select(i => listed[i]).SequenceEqual(expected), $"order {trial}: {string.Join(", ", order.Select(i => listed[i].Name))}");
        }

        Check(VulkanBackend.DeviceOrder([]).Length == 0, "no devices");

        // The UUID's bytes as reported, not Guid's field order: a smaller first byte comes first, whatever the 4th.
        Guid low = new([0x01, 0, 0, 0xFF, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]), high = new([0x02, 0, 0, 0x00, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
        Check(VulkanBackend.DeviceOrder([new(Discrete, high, driverA, null, "x"), new(Discrete, low, driverA, null, "x")]) is [1, 0], "UUID bytes in order");

        // The devices found here are numbered in that order.
        var keys = Enumerable.Range(0, VulkanBackend.DeviceCount).Select(VulkanBackend.OrderKey).ToArray();
        Check(VulkanBackend.DeviceOrder(keys).SequenceEqual(Enumerable.Range(0, keys.Length)), "vulkan:N follow the reported order");
        for (int i = 0; i < keys.Length; i++)
        {
            Console.WriteLine($"    vulkan:{i}: {keys[i].Name}, type {keys[i].Type}, uuid {keys[i].DeviceUuid:N}");
        }

        // The tuning cache names a device by its UUID (never its ordinal); without one, by PCI address and name.
        Check(VulkanBackend.DeviceKey(new VulkanDeviceFacts { DeviceUuid = Id(0x42) }, "x") == Id(0x42).ToString("N"), "tuning cache: keyed by UUID");
        string a = VulkanBackend.DeviceKey(new VulkanDeviceFacts(), "Soft GPU/1"), b = VulkanBackend.DeviceKey(new VulkanDeviceFacts(), "Soft GPU/2");
        Check(a != b && !a.Contains('/') && !a.Contains('\t'), $"tuning cache without a UUID: by name ({a}, {b})");
        Check(VulkanBackend.DeviceKey(new VulkanDeviceFacts { PciAddress = (0, 1, 0, 0) }, "same") != VulkanBackend.DeviceKey(new VulkanDeviceFacts { PciAddress = (0, 2, 0, 0) }, "same"),
            "tuning cache without a UUID: by PCI address too");
    }

    private static void VulkanMappedByDeviceType(Device device)
    {
        if (device.Type != DeviceType.Cpu)
        {
            return;
        }

        const uint Local = 0x1, Visible = 0x2, Coherent = 0x4, Cached = 0x8;
        const uint Integrated = 1, Discrete = 2, Cpu = 4;
        const ulong LocalHeap = 0x1, GiB = 1UL << 30, MiB = 1UL << 20;
        const StorageMemory Staging = StorageMemory.Staging, MappedDevice = StorageMemory.MappedDevice,
            MappedCached = StorageMemory.MappedCached, MappedUncached = StorageMemory.MappedUncached;

        // The candidates as "kind:type, …", and the page size storages get in a type (4096 allocations of at most 4 GiB).
        static string Candidates((uint, int)[] types, (ulong, ulong)[] heaps, uint deviceType) =>
            string.Join(", ", VulkanBackend.StorageCandidates(types, heaps, deviceType).Select(c => $"{VulkanBackend.StorageName(c.Kind)}:{c.Type}"));
        static long Pages(int type, (uint Flags, int Heap)[] types, (ulong Size, ulong Flags)[] heaps) =>
            VulkanBackend.PageSize(heaps[types[type].Heap].Size, 4 * GiB, 4096, null);

        // (a) An APU: a 2 GiB device-local carve-out, a 256 MiB device-local window the host sees, and system memory the
        // GPU reaches (host-visible, cached or not). Staging into the carve-out, or the system heap mapped, cached or
        // write-combined; the window is too small to hold storages. Pages from the chosen type's heap.
        (uint, int)[] apu = [(Local, 0), (Visible | Coherent, 1), (Local | Visible | Coherent, 2), (Visible | Coherent | Cached, 1)];
        (ulong, ulong)[] apuHeaps = [(2 * GiB, LocalHeap), (14 * GiB, 0), (256 * MiB, LocalHeap)];
        string found = Candidates(apu, apuHeaps, Integrated);
        Check(found == "staging:0, mapped-cached:3, mapped-uncached:1", $"APU: {found}");
        Check(Pages(0, apu, apuHeaps) == 16L << 20 && Pages(3, apu, apuHeaps) == 64L << 20 && Pages(1, apu, apuHeaps) == 64L << 20,
            "APU: pages of 16 MiB in the carve-out, 64 MiB in the 14 GiB system heap");
        Check(Candidates(apu, apuHeaps, Discrete) == "staging:0", "the same report from a discrete GPU: staging only (host memory is across the bus)");

        // The same APU with a device-local, cached type on the small window only: the window is still too small.
        (uint, int)[] apuWindowCached = [(Local, 0), (Visible | Coherent, 1), (Local | Visible | Coherent | Cached, 2)];
        found = Candidates(apuWindowCached, apuHeaps, Integrated);
        Check(found == "staging:0, mapped-uncached:1", $"APU, cached window: {found}");

        // (b) An integrated GPU with one large device-local, host-visible heap: staging, mapped (cached type; the same
        // type as mapped-cached, so once), and write-combined.
        (uint, int)[] igpu = [(Local, 0), (Local | Visible | Coherent, 0), (Local | Visible | Coherent | Cached, 0)];
        (ulong, ulong)[] igpuHeaps = [(16 * GiB, LocalHeap)];
        found = Candidates(igpu, igpuHeaps, Integrated);
        Check(found == "staging:0, mapped-device:2, mapped-uncached:1" && Pages(2, igpu, igpuHeaps) == 128L << 20, $"integrated, one heap: {found}, pages of 128 MiB");
        found = Candidates([(Local | Visible | Coherent | Cached, 0)], igpuHeaps, Cpu);
        Check(found == "staging:0, mapped-device:0", $"a CPU driver with one memory type: {found}");

        // (c) A discrete GPU with a 256 MiB BAR window: staging only, pages from device memory.
        (uint, int)[] window = [(Local, 0), (Visible | Coherent, 1), (Visible | Coherent | Cached, 1), (Local | Visible | Coherent, 2)];
        (ulong, ulong)[] windowHeaps = [(8 * GiB, LocalHeap), (32 * GiB, 0), (256 * MiB, LocalHeap)];
        found = Candidates(window, windowHeaps, Discrete);
        Check(found == "staging:0" && Pages(0, window, windowHeaps) == 64L << 20, $"discrete, 256 MiB window: {found}, pages of 64 MiB of the 8 GiB");

        // (d) The same GPU with resizable BAR (its whole memory host-visible): staging or mapped device memory.
        (uint, int)[] bar = [(Local, 0), (Visible | Coherent, 1), (Visible | Coherent | Cached, 1), (Local | Visible | Coherent, 0)];
        (ulong, ulong)[] barHeaps = [(8 * GiB, LocalHeap), (32 * GiB, 0)];
        found = Candidates(bar, barHeaps, Discrete);
        Check(found == "staging:0, mapped-device:3" && Pages(3, bar, barHeaps) == 64L << 20, $"discrete, resizable BAR: {found}");

        // Nothing host-visible at all: staging whatever the type.
        Check(Candidates([(Local, 0)], [(4 * GiB, LocalHeap)], Integrated) == "staging:0", "integrated without host-visible memory: staging");

        // The cost model, with timings made up (GB/s: device, upload, download). The APU above as measured: the
        // carve-out reads fastest, cached system memory uploads fastest but reads 30% slower: staging.
        static VulkanBackend.StorageTiming T(StorageMemory kind, double device, double upload) => new(kind, device, upload, upload);
        VulkanBackend.StorageTiming[] measured = [T(Staging, 39, 6.3), T(MappedCached, 27, 12.2)];
        Check(VulkanBackend.ChooseStorage(measured) == 0, "APU: the fastest device reads win over faster uploads");
        measured = [T(Staging, 39, 6.3), T(MappedCached, 27, 12.2), T(MappedUncached, 38.2, 9)];
        Check(VulkanBackend.ChooseStorage(measured) == 2, "within 3% of the fastest device reads and uploading faster: write-combined");
        measured = [T(Staging, 39, 6.3), T(MappedCached, 27, 12.2), T(MappedUncached, 37, 9)];
        Check(VulkanBackend.ChooseStorage(measured) == 0, "5% slower device reads: not chosen for faster uploads");
        measured = [T(Staging, 10, 2), T(MappedDevice, 10, 8)];
        Check(VulkanBackend.ChooseStorage(measured) == 1, "the same memory (a CPU driver): the faster upload");
        measured = [T(Staging, 500, 12), T(MappedDevice, 498, 10)];
        Check(VulkanBackend.ChooseStorage(measured) == 0, "resizable BAR uploading slower: staging");
        measured = [T(Staging, 10, 5), T(MappedDevice, 10, 5)];
        Check(VulkanBackend.ChooseStorage(measured) == 0, "a tie: the earlier candidate");
        Check(VulkanBackend.ChooseStorage([]) == -1 && VulkanBackend.ChooseStorage([T(Staging, 0, 1)]) == -1, "nothing measured: none");

        // The timings as the cache keeps them, read back.
        measured = [T(Staging, 39.25, 6.3), T(MappedUncached, 38.2, 9.125)];
        string text = VulkanBackend.FormatTimings(measured);
        Check(VulkanBackend.ParseTimings(text).SequenceEqual(measured) && VulkanBackend.ParseTimings("junk,staging=1/2").Count == 0
              && VulkanBackend.ParseTimings(null).Count == 0, $"timings round trip ({text})");

        // Deciding before measuring: a forced memory, then IDRAK_VULKAN_STORAGE, then the cache, each only when the device
        // has that candidate; a single candidate needs no measurement.
        var apuCandidates = VulkanBackend.StorageCandidates(apu, apuHeaps, Integrated);
        var none = new Dictionary<string, string>();
        var cache = new Dictionary<string, string> { ["storage-memory"] = "mapped-uncached", ["storage-memory/timings"] = text };
        Check(VulkanBackend.DecideStorage(apuCandidates, Staging, "mapped-cached", cache) is (Staging, "override", _), "IDRAK_VULKAN_STAGING first");
        Check(VulkanBackend.DecideStorage(apuCandidates, null, "Mapped-Cached", cache) is (MappedCached, "override", _), "IDRAK_VULKAN_STORAGE next");
        Check(VulkanBackend.DecideStorage(apuCandidates, null, "mapped-device", cache) is (MappedUncached, "cached", var t) && t.SequenceEqual(measured),
            "a memory the device lacks: ignored; the cached choice and its timings");
        Check(VulkanBackend.DecideStorage(apuCandidates, null, null, none) is (null, "measured", _), "nothing decided: measured");
        Check(VulkanBackend.DecideStorage(apuCandidates, null, null, new Dictionary<string, string> { ["storage-memory"] = "mapped-device" }) is (null, "measured", _),
            "a cached choice that is no longer a candidate: measured again");
        var windowCandidates = VulkanBackend.StorageCandidates(window, windowHeaps, Discrete);
        Check(VulkanBackend.DecideStorage(windowCandidates, null, "mapped-cached", none) is (Staging, "only candidate", _), "one candidate: no measurement");
        foreach (var kind in Enum.GetValues<StorageMemory>())
        {
            Check(VulkanBackend.ParseStorage(VulkanBackend.StorageName(kind)) == kind, $"{kind}: named {VulkanBackend.StorageName(kind)}");
        }
    }

    private static void VulkanSizesFromReport(Device device)
    {
        if (device.Type != DeviceType.Cpu)
        {
            return;
        }

        const uint Local = 0x1, Visible = 0x2, Coherent = 0x4, Cached = 0x8;
        const ulong LocalHeap = 0x1, GiB = 1UL << 30, MiB = 1UL << 20;

        // Mapped device memory: on a heap about as large as the device's memory, host-cached type first.
        static StorageMemory[] Kinds((uint, int)[] types, (ulong, ulong)[] heaps) => [.. VulkanBackend.StorageCandidates(types, heaps, 2).Select(c => c.Kind)];
        Check(VulkanBackend.StorageCandidates([(Local | Visible | Coherent, 0), (Local | Visible | Coherent | Cached, 0)], [(16 * GiB, LocalHeap)], 2)
                  .Single(c => c.Kind == StorageMemory.MappedDevice).Type == 1, "shared heap: mapped (cached type)");
        Check(Kinds([(Local, 0), (Visible | Coherent, 1), (Local | Visible | Coherent, 2)], [(8 * GiB, LocalHeap), (32 * GiB, 0), (256 * MiB, LocalHeap)])
                is [StorageMemory.Staging], "a 256 MiB window of an 8 GiB device: staging");
        Check(Kinds([(Local, 0), (Visible | Coherent, 1), (Local | Visible | Coherent, 0)], [(8 * GiB, LocalHeap), (32 * GiB, 0)])
                is [StorageMemory.Staging, StorageMemory.MappedDevice], "the whole device memory host-visible: mapped too");
        Check(Kinds([(Visible | Coherent, 0)], [(4 * GiB, 0)]) is [StorageMemory.Staging], "no device-local heap: staging");

        // Staging buffer: 1/512 of its heap as a power of two, within the largest allocation, at least the atom size.
        Check(VulkanBackend.StagingSize(8 * GiB, 4 * GiB, 64, null) == 16L << 20, "staging: 16 MiB of an 8 GiB heap");
        Check(VulkanBackend.StagingSize(48 * GiB, 4 * GiB, 64, null) == 64L << 20, "staging: 64 MiB of a 48 GiB heap (power of two)");
        Check(VulkanBackend.StagingSize(64 * GiB, 32 * MiB, 64, null) == 32L << 20, "staging: within maxMemoryAllocationSize");
        Check(VulkanBackend.StagingSize(16 * MiB, 0, 256, null) == 32L << 10 && VulkanBackend.StagingSize(0, 0, 256, null) == 256,
            "staging: small heaps, at least nonCoherentAtomSize");
        Check(VulkanBackend.StagingSize(8 * GiB, 4 * GiB, 64, 1_000_001) == 1_000_000, "staging: the setting, in whole floats");

        // Pages: 1/128 of the heap, raised until half the allocation cap covers the heap, within the largest allocation.
        Check(VulkanBackend.PageSize(8 * GiB, 4 * GiB, 4096, null) == 64L << 20, "pages: 64 MiB of an 8 GiB heap");
        Check(VulkanBackend.PageSize(16 * GiB, 16 * GiB, 64, null) == 512L << 20, "pages: raised so 32 allocations cover 16 GiB");
        Check(VulkanBackend.PageSize(64 * GiB, 128 * MiB, 4096, null) == 128L << 20, "pages: within maxMemoryAllocationSize");
        Check(VulkanBackend.PageSize(8 * GiB, 4 * GiB, 4096, 3 << 20) == 3L << 20, "pages: the setting");

        for (int i = 0; i < VulkanBackend.DeviceCount; i++)
        {
            var backend = (VulkanBackend)Device.Get("vulkan", i).Backend;
            var facts = backend.Facts;
            Check(facts.SubgroupSize >= 1 && facts.MinSubgroupSize <= facts.SubgroupSize && facts.SubgroupSize <= facts.MaxSubgroupSize,
                $"vulkan:{i}: subgroup sizes {facts.MinSubgroupSize} ≤ {facts.SubgroupSize} ≤ {facts.MaxSubgroupSize}");
            Check(facts.MaxWorkGroupInvocations >= 128 && facts.MaxWorkGroupSize.X >= 128 && facts.MaxSharedMemoryBytes >= 16384,
                $"vulkan:{i}: workgroup limits at least Vulkan's minimums");
            Check(facts.MaxMemoryAllocationSize >= 1 << 30 && facts.MaxStorageBufferRange >= 1 << 27, $"vulkan:{i}: allocation and storage limits");
            Check((facts.SubgroupOperations & VulkanDeviceFacts.SubgroupBasic) != 0, $"vulkan:{i}: basic subgroup operations (Vulkan 1.1)");
            Check(backend.PageBytes >= sizeof(float) && (ulong)backend.PageBytes <= facts.MaxMemoryAllocationSize, $"vulkan:{i}: page within the largest allocation");
            Console.WriteLine($"    vulkan:{i}: {backend.Describe()}");
        }
    }

    private static void VulkanRuntimeMeasured(Device device)
    {
        if (device.Type != DeviceType.Cpu || VulkanBackend.DeviceCount == 0)
        {
            return;
        }

        string? saved = VulkanBackend.TuningCacheFile;
        string file = Path.Combine(Path.GetTempPath(), $"idrak-vulkan-tuning-{Environment.ProcessId}.tsv");
        File.Delete(file);
        VulkanBackend.TuningCacheFile = file;
        try
        {
            for (int i = 0; i < VulkanBackend.DeviceCount; i++)
            {
                var first = VulkanBackend.CreateSeparate(i, preferMapped: true);
                var measured = first.Measured;
                bool push = first.PushDescriptors;
                int batch = first.MaxBatchCommands, inFlight = first.MaxInFlight;
                var (storage, storageChoice, storageTimings) = (first.StorageKind, first.StorageChoice, first.StorageTimings.ToArray());
                int candidates = first.StorageCandidateList.Count;
                Check(storageChoice == (candidates > 1 ? "measured" : "only candidate") && first.StorageCandidateList.Any(c => c.Kind == storage)
                      && (candidates == 1 || (storageTimings.Length == candidates && storageTimings.All(t => t.Device > 0 && t.Upload > 0 && t.Download > 0)
                                              && storageTimings[VulkanBackend.ChooseStorage(storageTimings)].Kind == storage)),
                    $"vulkan:{i}: storage memory {first.DescribeStorage()}");
                Check(first.UnifiedMemory == (storage != StorageMemory.Staging) && (candidates == 1 || (first.Allocations == 0 && first.PageCount == 0)),
                    $"vulkan:{i}: nothing left behind by the measurement");
                first.Shutdown();
                Check(first.PushDescriptorsChoice == "measured" && measured.Sets > 0 && measured.Record > 0 && batch >= 1 && inFlight >= 2,
                    $"vulkan:{i}: measured ({measured}; batches of {batch}, {inFlight} in flight)");
                Check(File.Exists(file) && File.ReadAllLines(file).Any(l => l.Contains("\tbatch-commands/", StringComparison.Ordinal) || l.Contains("/batch-commands/", StringComparison.Ordinal)),
                    $"vulkan:{i}: kept in {file}");

                var second = VulkanBackend.CreateSeparate(i, preferMapped: true);
                Check(second.StorageKind == storage && second.StorageChoice == (storageChoice == "measured" ? "cached" : storageChoice)
                      && second.StorageTimings.SequenceEqual(storageTimings), $"vulkan:{i}: storage memory read back from the cache ({second.DescribeStorage()})");
                Check(second.PushDescriptorsChoice == "cached" && second.PushDescriptors == push && second.MaxBatchCommands == batch && second.MaxInFlight == inFlight
                      && second.Measured == default, $"vulkan:{i}: read back from the cache, not measured again");
                second.Shutdown();

                var staging = VulkanBackend.CreateSeparate(i, preferMapped: false);
                Check(staging.StorageKind == StorageMemory.Staging && staging.StorageChoice == "override" && !staging.UnifiedMemory,
                    $"vulkan:{i}: staging on request (IDRAK_VULKAN_STAGING)");
                staging.Shutdown();
                Check(File.ReadAllLines(file).Any(l => l.Contains("/storage-memory\t", StringComparison.Ordinal)) == (candidates > 1)
                      && File.ReadAllLines(file).Any(l => l.Contains("/push-descriptors\t", StringComparison.Ordinal)),
                    $"vulkan:{i}: the storage memory and the runtime policy kept side by side");

                var sets = VulkanBackend.CreateSeparate(i, preferMapped: true, pushDescriptors: false);
                Check(!sets.PushDescriptors && sets.PushDescriptorsChoice == "override", $"vulkan:{i}: descriptor sets on request");
                sets.Shutdown();
                Console.WriteLine($"    vulkan:{i}: {(push ? "pushed descriptors" : "descriptor sets")}, batches of {batch}, {inFlight} in flight ({measured})");
            }
        }
        finally
        {
            VulkanBackend.TuningCacheFile = saved;
            File.Delete(file);
        }
    }

    private static void VulkanNoVendorNames(Device device)
    {
        _ = device;
        string folder = Path.Combine(RepositoryRoot(), "src", "Idrak", "Backends", "Vulkan");
        var vendorIds = new Regex(@"0x0*(10DE|1002|1022|8086|13B5|5143|106B|1010|14E4|10005|1AE0|19E5)\b", RegexOptions.IgnoreCase);
        var names = new Regex(@"\b(NVIDIA|AMD|ATI|Intel|Radeon|GeForce|Quadro|Tesla|RTX|GTX|Arc|Iris|UHD|Adreno|Mali|Qualcomm|Apple|PowerVR|Imagination|Broadcom|MoltenVK|lavapipe|llvmpipe|SwiftShader|RADV|ANV|NVK|Mesa|Snapdragon|Exynos)\b",
            RegexOptions.IgnoreCase);
        var found = new List<string>();
        foreach (string path in Directory.EnumerateFiles(folder, "*.cs", SearchOption.AllDirectories))
        {
            string[] lines = File.ReadAllLines(path);
            for (int n = 0; n < lines.Length; n++)
            {
                string code = CodeOf(lines[n]);
                if (vendorIds.Match(code) is { Success: true } id)
                {
                    found.Add($"{Path.GetFileName(path)}:{n + 1}: {id.Value}");
                }

                if (names.Match(code) is { Success: true } name)
                {
                    found.Add($"{Path.GetFileName(path)}:{n + 1}: {name.Value}");
                }
            }
        }

        Check(found.Count == 0, $"vendor ids or names in code: {string.Join("; ", found.Take(10))}");

        // The code of a line: nothing on a comment line, and a trailing // comment dropped (one outside a string).
        static string CodeOf(string line)
        {
            string trimmed = line.TrimStart();
            if (trimmed.StartsWith("//", StringComparison.Ordinal) || trimmed.StartsWith("*", StringComparison.Ordinal) || trimmed.StartsWith("/*", StringComparison.Ordinal))
            {
                return "";
            }

            bool inString = false;
            for (int i = 0; i + 1 < line.Length; i++)
            {
                if (line[i] == '"' && (i == 0 || line[i - 1] != '\\'))
                {
                    inString = !inString;
                }
                else if (!inString && line[i] == '/' && line[i + 1] == '/')
                {
                    return line[..i];
                }
            }

            return line;
        }
    }
}
