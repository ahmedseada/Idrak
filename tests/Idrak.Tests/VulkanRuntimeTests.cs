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
        ("vulkan runtime: mapped memory, staging and page sizes come from the reported heaps and limits", VulkanSizesFromReport),
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

    private static void VulkanSizesFromReport(Device device)
    {
        if (device.Type != DeviceType.Cpu)
        {
            return;
        }

        const uint Local = 0x1, Visible = 0x2, Coherent = 0x4, Cached = 0x8;
        const ulong LocalHeap = 0x1, GiB = 1UL << 30, MiB = 1UL << 20;

        // One heap the host and device share (integrated GPU, CPU driver): mapped, host-cached type first.
        Check(VulkanBackend.MappableType([(Local | Visible | Coherent, 0), (Local | Visible | Coherent | Cached, 0)], [(16 * GiB, LocalHeap)]) == 1,
            "shared memory: mapped (cached type)");

        // A discrete GPU with a small host-visible window of its memory: staging.
        Check(VulkanBackend.MappableType([(Local, 0), (Visible | Coherent, 1), (Local | Visible | Coherent, 2)],
                [(8 * GiB, LocalHeap), (32 * GiB, 0), (256 * MiB, LocalHeap)]) == -1, "a 256 MiB window of an 8 GiB device: staging");

        // The same GPU when the host maps all of its memory: mapped.
        Check(VulkanBackend.MappableType([(Local, 0), (Visible | Coherent, 1), (Local | Visible | Coherent, 0)],
                [(8 * GiB, LocalHeap), (32 * GiB, 0)]) == 2, "the whole device memory host-visible: mapped");

        // No device-local heap at all: staging.
        Check(VulkanBackend.MappableType([(Visible | Coherent, 0)], [(4 * GiB, 0)]) == -1, "no device-local heap: staging");

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
                first.Shutdown();
                Check(first.PushDescriptorsChoice == "measured" && measured.Sets > 0 && measured.Record > 0 && batch >= 1 && inFlight >= 2,
                    $"vulkan:{i}: measured ({measured}; batches of {batch}, {inFlight} in flight)");
                Check(File.Exists(file) && File.ReadAllLines(file).Any(l => l.Contains("\tbatch-commands/", StringComparison.Ordinal) || l.Contains("/batch-commands/", StringComparison.Ordinal)),
                    $"vulkan:{i}: kept in {file}");

                var second = VulkanBackend.CreateSeparate(i, preferMapped: true);
                Check(second.PushDescriptorsChoice == "cached" && second.PushDescriptors == push && second.MaxBatchCommands == batch && second.MaxInFlight == inFlight
                      && second.Measured == default, $"vulkan:{i}: read back from the cache, not measured again");
                second.Shutdown();

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
