// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Collections.Concurrent;
using Idrak;
using Idrak.Abstraction.Devices.Cpu;
using Idrak.Gpu.Vulkan;
using Idrak.Layers;
using Idrak.Models.Abstractions;

// Limits some devices report low and features some report: storages larger than one binding (maxStorageBufferRange,
// 128 MiB on some devices) bound in windows, forced on any device with a small binding range; the loader's file name per
// operating system; subgroup size control where reported (the size measured and stored, results as without it).
internal abstract partial class Tests
{
    internal static readonly (string Name, Action<Device> Run)[] VulkanLimitsGroup =
    [
        ("vulkan limits: the kernels' width is measured at start (a float32 product and one-row decoding work at each width up to the formula's), stored per device and driver, and read back; products at the measured width match the CPU", VulkanWidthMeasured),
        ("vulkan limits: the loader's file names per operating system (libvulkan.so on Android; Linux and Windows unchanged)", VulkanLoaderNames),
        ("vulkan limits: windows of a large storage are whole rows within the binding range, starting at aligned offsets", VulkanWindowRows),
        ("vulkan limits: with a 1 MiB binding range, gathers from a large table, uploads and downloads run on the device and match the CPU", VulkanLargeStorages),
        ("vulkan limits: subgroup size control only where reported; the size measured per kernel and stored; results as without it", VulkanSubgroupSizes),
        ("vulkan limits: a kernel the driver refuses (IDRAK_VULKAN_REFUSE_KERNELS) never takes the device down: it starts (the width probe too), the packed products run their next variant with the same bits, a kernel without variants runs on the host with the CPU's bits, one without its subgroup-size request where it only refused that; asked once per process; listed in the facts", VulkanRefusedKernels),
    ];

    internal static void VulkanLoaderNames(Device device)
    {
        if (device.Type != DeviceType.Cpu)
        {
            return;                                                          // pure: once, on the CPU
        }

        Check(VulkanDriver.CandidateNames("android") is ["libvulkan.so"], "Android: libvulkan.so");
        Check(VulkanDriver.CandidateNames("linux") is ["libvulkan.so.1", "libvulkan.so"], "Linux: libvulkan.so.1, then libvulkan.so");
        Check(VulkanDriver.CandidateNames("freebsd") is ["libvulkan.so.1", "libvulkan.so"], "FreeBSD: as Linux");
        Check(VulkanDriver.CandidateNames("windows") is ["vulkan-1.dll"], "Windows: vulkan-1.dll");
        Check(VulkanDriver.CandidateNames("macos").Length == 0 && VulkanDriver.CandidateNames("").Length == 0, "elsewhere: none");
    }

    internal static void VulkanWindowRows(Device device)
    {
        if (device.Type != DeviceType.Cpu)
        {
            return;
        }

        // 128 MiB, 16-byte alignment: int8 rows of 151936 bytes (a multiple of 16), groups of 32 rows.
        long rows = VulkanBackend.WindowRows(128L << 20, 16, 32, [151_936]);
        Check(rows == 864 && rows * 151_936 <= 128L << 20 && (rows + 32) * 151_936 > 128L << 20, $"int8 output layer: {rows} rows per window");

        // Rows of 3000 bytes, 256-byte alignment: windows start every 32 rows (3000 · 32 = 375 · 256).
        rows = VulkanBackend.WindowRows(1 << 20, 256, 1, [3000]);
        Check(rows % 32 == 0 && rows * 3000 <= 1 << 20 && rows * 3000 % 256 == 0 && (rows + 32) * 3000 > 1 << 20, $"3000-byte rows: {rows}");

        // Two storages together (packed words and the float32 output): the stricter of both.
        rows = VulkanBackend.WindowRows(1 << 20, 64, 1, [1004, 4004]);
        Check(rows > 0 && rows * 4004 <= 1 << 20 && rows * 1004 % 64 == 0 && rows * 4004 % 64 == 0, $"two storages: {rows}");

        // A row larger than the range: no window.
        Check(VulkanBackend.WindowRows(1 << 20, 16, 1, [(1 << 20) + 4]) == 0, "a row larger than a binding has no window");
    }

    // Ordinals of the Vulkan devices a test runs on: the device itself, none for others.
    internal static List<int> LimitOrdinals(Device device) => device.Type == DeviceType.Vulkan
        && Environment.GetEnvironmentVariable("IDRAK_VULKAN_KERNELS") is not ("0" or "false") ? [device.Ordinal] : [];

    internal static void VulkanLargeStorages(Device device)
    {
        foreach (int ordinal in LimitOrdinals(device))
        {
            var saved = VulkanBackend.StorageRangeOverride;
            VulkanBackend.StorageRangeOverride = 1 << 20;
            VulkanBackend backend;
            try
            {
                backend = VulkanBackend.CreateSeparate(ordinal, preferMapped: true);
            }
            finally
            {
                VulkanBackend.StorageRangeOverride = saved;
            }

            try
            {
                Check(backend.MaxStorageBytes == 1 << 20, $"binding range {backend.MaxStorageBytes}");
                LargeStorageCases(backend);
            }
            finally
            {
                backend.Shutdown();
            }
        }
    }

    internal static void LargeStorageCases(VulkanBackend backend)
    {
        var cpu = CpuBackend.Instance;
        long fallbacks = Idrak.Abstraction.Operations.Kernels.HostCalls(backend);
        var made = new List<Storage>();
        Storage On(Backend b, float[] values)
        {
            var s = b.Allocate(values.Length, zeroed: false);
            b.Upload(values, s);
            made.Add(s);
            return s;
        }

        Storage Empty(Backend b, int n)
        {
            var s = b.Allocate(n, zeroed: true);
            made.Add(s);
            return s;
        }

        float[] Read(Backend b, Storage s, int n)
        {
            var values = new float[n];
            b.Download(s, values);
            return values;
        }

        try
        {
            // Uploads and downloads of a storage of 3 MB (three windows), whole and from an offset in the last window.
            {
                var values = Values(750_001, 3);
                var s = On(backend, values);
                Check(Read(backend, s, values.Length).SequenceEqual(values), "round trip of a storage larger than a binding");
                var tail = new float[1000];
                backend.DownloadRange(s, 600_000, tail);
                Check(tail.SequenceEqual(values.Skip(600_000).Take(1000)), "download from an offset past the first binding");
            }

            // Gathers from tables of several windows (first and last rows, rows on both sides of a window's end).
            {
                const int Vocabulary = 3000, Dim = 130;
                var table = Values(Vocabulary * Dim, 5);
                float[] indices = [0, 2999, 1234, 2017, 7, 1500, 2999, 64];
                var (gt, ct, gi, ci) = (On(backend, table), On(cpu, table), On(backend, indices), On(cpu, indices));
                var (gy, cy) = (Empty(backend, indices.Length * Dim), Empty(cpu, indices.Length * Dim));
                backend.Gather(gt, gi, gy, indices.Length, Dim, Vocabulary);
                cpu.Gather(ct, ci, cy, indices.Length, Dim, Vocabulary);
                Check(Read(backend, gy, indices.Length * Dim).SequenceEqual(Read(cpu, cy, indices.Length * Dim)), "gather from a large float32 table");
            }

            Check(Idrak.Abstraction.Operations.Kernels.HostCalls(backend) == fallbacks, $"{Idrak.Abstraction.Operations.Kernels.HostCalls(backend) - fallbacks} host fallbacks");
        }
        finally
        {
            foreach (var s in made)
            {
                s.Release();
            }
        }
    }

    internal static void VulkanSubgroupSizes(Device device)
    {
        foreach (int ordinal in LimitOrdinals(device))
        {
            string cache = Path.Combine(Path.GetTempPath(), $"idrak-subgroups-{Environment.ProcessId}.tsv");
            string? savedCache = VulkanBackend.TuningCacheFile;
            VulkanBackend.TuningCacheFile = cache;
            var facts = VulkanBackend.FactsOf(ordinal);
            var values = Values(200_003, 11);
            float[] Run(VulkanBackend backend)
            {
                var x = backend.Allocate(values.Length, zeroed: false);
                var y = backend.Allocate(4, zeroed: true);
                var rows = backend.Allocate(64 * 3001, zeroed: false);
                var soft = backend.Allocate(64 * 3001, zeroed: false);
                try
                {
                    backend.Upload(values, x);
                    backend.Upload(values.AsSpan(0, 64 * 3001), rows);
                    backend.Sum(x, y, values.Length, 0.5f);
                    backend.Softmax(rows, soft, 64, 3001, log: false);
                    var result = new float[1 + 64 * 3001];
                    backend.Download(y, result.AsSpan(0, 1));
                    backend.Download(soft, result.AsSpan(1));
                    return result;
                }
                finally
                {
                    x.Release();
                    y.Release();
                    rows.Release();
                    soft.Release();
                }
            }

            try
            {
                var first = VulkanBackend.CreateSeparate(ordinal, preferMapped: true);
                float[] measured;
                int? chosen;
                try
                {
                    Console.WriteLine($"    vulkan:{ordinal}: subgroup size control {(first.SubgroupSizeControl ? $"on (sizes {facts.MinSubgroupSize}-{facts.MaxSubgroupSize}, candidates {string.Join(", ", first.SubgroupSizeCandidates(first.Width))}{(first.FullSubgroups ? ", full subgroups" : "")})" : "not reported")}");
                    measured = Run(first);
                    chosen = first.ChosenSubgroupSize("sum");
                    if (first.SubgroupSizeControl && first.Limits.SubgroupReductions && VulkanBackend.Autotune)
                    {
                        Check(chosen is not null, "a size was chosen for the sum kernel");
                        Check(File.Exists(cache) && File.ReadLines(cache).Any(l => l.Contains("SubgroupSize", StringComparison.Ordinal)), "the choice is stored");
                        Console.WriteLine($"    vulkan:{ordinal}: sum kernel: {(chosen == 0 ? "the default size" : $"subgroups of {chosen}")}");
                    }
                    else
                    {
                        Check(chosen is null, "no size chosen without subgroup size control");
                    }
                }
                finally
                {
                    first.Shutdown();
                }

                // A second backend reads the stored choice: nothing measured again.
                var second = VulkanBackend.CreateSeparate(ordinal, preferMapped: true);
                try
                {
                    var again = Run(second);
                    Check(second.ChosenSubgroupSize("sum") == chosen, "the stored size is used");
                    Check(again.SequenceEqual(measured), "the same results with the stored choice");
                    Check(second.Measurements == 0 || !second.SubgroupSizeControl, $"{second.Measurements} choices measured again");
                }
                finally
                {
                    second.Shutdown();
                }

                // Without subgroup size control: the same results (identical bits where the device runs one size).
                VulkanBackend.SubgroupSizeControlOverride = false;
                VulkanBackend off;
                try
                {
                    off = VulkanBackend.CreateSeparate(ordinal, preferMapped: true);
                }
                finally
                {
                    VulkanBackend.SubgroupSizeControlOverride = null;
                }

                try
                {
                    Check(!off.SubgroupSizeControl, "the override turns it off");
                    var plain = Run(off);
                    Check(off.ChosenSubgroupSize("sum") is null, "no size chosen when off");
                    if (facts.MinSubgroupSize == facts.MaxSubgroupSize)
                    {
                        Check(plain.SequenceEqual(measured), "identical bits with and without subgroup size control");
                    }
                    else
                    {
                        for (int i = 0; i < plain.Length; i++)
                        {
                            Check(Math.Abs(plain[i] - measured[i]) <= 1e-4f * Math.Max(1f, Math.Abs(plain[i])), $"value {i}: {plain[i]} without, {measured[i]} with");
                        }
                    }
                }
                finally
                {
                    off.Shutdown();
                }
            }
            finally
            {
                VulkanBackend.TuningCacheFile = savedCache;
                File.Delete(cache);
            }
        }
    }

    internal static void VulkanWidthMeasured(Device device)
    {
        if (device.Backend is not VulkanBackend vulkan)
        {
            return;
        }

        int ordinal = device.Ordinal;
        int wider = Math.Min(256, Math.Min(vulkan.Limits.MaxInvocations, vulkan.Limits.MaxSizeX));
        string cache = Path.Combine(Path.GetTempPath(), $"idrak-width-{Environment.ProcessId}.tsv");
        string? savedCache = VulkanBackend.TuningCacheFile;
        var savedFormula = VulkanBackend.ProbeFormulaOverride;
        VulkanBackend.TuningCacheFile = cache;
        VulkanBackend.ProbeFormulaOverride = wider;
        try
        {
            File.Delete(cache);
            var r = new Random(4);
            float[] a = [.. Enumerable.Range(0, 37 * 41).Select(_ => r.NextSingle() - 0.5f)];
            float[] b = [.. Enumerable.Range(0, 41 * 29).Select(_ => r.NextSingle() - 0.5f)];
            var expected = new float[37 * 29];
            for (int i = 0; i < 37; i++)
            {
                for (int j = 0; j < 29; j++)
                {
                    double sum = 0;
                    for (int t = 0; t < 41; t++)
                    {
                        sum += a[i * 41 + t] * b[t * 29 + j];
                    }

                    expected[i * 29 + j] = (float)sum;
                }
            }

            float[] Product(VulkanBackend backend)
            {
                var (sa, sb, sc) = (backend.Allocate(a.Length, false), backend.Allocate(b.Length, false), backend.Allocate(expected.Length, false));
                backend.Upload(a, sa);
                backend.Upload(b, sb);
                backend.BatchedMatMul(sa, sb, sc, 1, 37, 29, 41, false, false, 0f);
                var result = new float[expected.Length];
                backend.Download(sc, result);
                sa.Release();
                sb.Release();
                sc.Release();
                return result;
            }

            var first = VulkanBackend.CreateSeparate(ordinal, preferMapped: true);
            int width;
            try
            {
                width = first.Width;
                var l = first.Limits;
                int widest = Math.Max(wider, VulkanKernels.WidthFor(l.MaxInvocations, l.MaxSizeX, l.SharedBytes, l.SubgroupSize));  // the override only widens
                bool several = widest > Math.Max(VulkanKernels.MinWidth, first.Limits.SubgroupSize);
                Console.WriteLine($"    vulkan:{ordinal}: width {width} ({first.WidthChoice}, candidates up to {widest})");
                Check(!several || first.WidthChoice == "measured", $"measured when there are several widths ({first.WidthChoice})");
                Check(width >= VulkanKernels.MinWidth && width <= widest && (width & (width - 1)) == 0, $"a candidate width ({width})");
                AssertClose(expected, Product(first), 1e-4f, "the product at the measured width");
            }
            finally
            {
                first.Shutdown();
            }

            var second = VulkanBackend.CreateSeparate(ordinal, preferMapped: true);
            try
            {
                Check(second.Width == width, $"the stored width is read back ({second.Width}, {width})");
                Check(second.WidthChoice is "cached" or "formula", $"read from the cache ({second.WidthChoice})");
            }
            finally
            {
                second.Shutdown();
            }
        }
        finally
        {
            VulkanBackend.TuningCacheFile = savedCache;
            VulkanBackend.ProbeFormulaOverride = savedFormula;
            File.Delete(cache);
        }
    }
}
