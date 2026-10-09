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
internal static partial class Tests
{
    private static readonly (string Name, Action<Device> Run)[] VulkanLimitsGroup =
    [
        ("vulkan limits: the kernels' width is measured at start (a float32 product and one-row decoding work at each width up to the formula's), stored per device and driver, and read back; products at the measured width match the CPU", VulkanWidthMeasured),
        ("vulkan limits: the loader's file names per operating system (libvulkan.so on Android; Linux and Windows unchanged)", VulkanLoaderNames),
        ("vulkan limits: windows of a large storage are whole rows within the binding range, starting at aligned offsets", VulkanWindowRows),
        ("vulkan limits: with a 1 MiB binding range, large weights' products (int8, int4, bfloat16), gathers (by rows and by columns), dequantizations, uploads and downloads run on the device and match the CPU", VulkanLargeStorages),
        ("vulkan limits: with a small binding range, decoding steps whose embedding and head exceed it (int8, int4, bfloat16, a tied bfloat16 head read by columns) match the CPU with no host fallbacks", VulkanLargeDecoder),
        ("vulkan limits: subgroup size control only where reported; the size measured per kernel and stored; results as without it", VulkanSubgroupSizes),
    ];

    private static void VulkanLoaderNames(Device device)
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

    private static void VulkanWindowRows(Device device)
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
    private static List<int> LimitOrdinals(Device device) => device.Type == DeviceType.Vulkan
        && Environment.GetEnvironmentVariable("IDRAK_VULKAN_KERNELS") is not ("0" or "false") ? [device.Ordinal] : [];

    private static void VulkanLargeStorages(Device device)
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

    private static void LargeStorageCases(VulkanBackend backend)
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

        void Close(float[] want, float[] got, string what, double tolerance = 1e-4)
        {
            double worst = 0, scale = Math.Max(1e-6, want.Max(v => Math.Abs((double)v)));
            for (int i = 0; i < want.Length; i++)
            {
                worst = Math.Max(worst, Math.Abs(want[i] - (double)got[i]) / scale);
            }

            Check(worst <= tolerance, $"{what}: off by {worst:G3} of the largest value");
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

            // Packed products: weights of several windows, k a multiple of 32 or not, rows of x 1 to 9.
            foreach (var (k, n) in new[] { (1024, 3000), (2050, 1001), (96, 12_000) })
            {
                var w = Values(k * n, k + n);
                var int8 = Int8Weight.Quantize(w, k, n, Device.Cpu);
                var int4 = Int4Weight.Quantize(w, k, n, Device.Cpu);
                var bf16 = BFloat16Weight.FromValues(w, k, n, Device.Cpu);
                float[] q8 = int8.Packed.ToArray(), s8 = int8.Scales.ToArray(), q4 = int4.Packed.ToArray(), s4 = int4.Scales.ToArray(), b16 = bf16.Packed.ToArray();
                Check(q8.Length * 4L > backend.MaxStorageBytes, $"{k} x {n}: the int8 weights exceed a binding");
                foreach (int m in new[] { 1, 3, 9 })
                {
                    var x = Values(m * k, m);
                    var (gx, cx) = (On(backend, x), On(cpu, x));
                    var (g8, c8, gs8, cs8) = (On(backend, q8), On(cpu, q8), On(backend, s8), On(cpu, s8));
                    var (g4, c4, gs4, cs4) = (On(backend, q4), On(cpu, q4), On(backend, s4), On(cpu, s4));
                    var (gb, cb) = (On(backend, b16), On(cpu, b16));
                    var (gy, cy) = (Empty(backend, m * n), Empty(cpu, m * n));
                    string label = $"{m} x {k} -> {n}";

                    backend.Int8MatMul(gx, g8, gs8, gy, m, n, k);
                    cpu.Int8MatMul(cx, c8, cs8, cy, m, n, k);
                    Close(Read(cpu, cy, m * n), Read(backend, gy, m * n), $"int8 product {label}");

                    backend.Int4MatMul(gx, g4, gs4, gy, m, n, k);
                    cpu.Int4MatMul(cx, c4, cs4, cy, m, n, k);
                    Close(Read(cpu, cy, m * n), Read(backend, gy, m * n), $"int4 product {label}");

                    backend.BFloat16MatMul(gx, gb, gy, m, n, k);
                    cpu.BFloat16MatMul(cx, cb, cy, m, n, k);
                    Close(Read(cpu, cy, m * n), Read(backend, gy, m * n), $"bfloat16 product {label}");
                }

                int8.Packed.Dispose();
                int8.Scales.Dispose();
                int4.Packed.Dispose();
                int4.Scales.Dispose();
                bf16.Packed.Dispose();
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

                var bf16 = BFloat16Weight.FromValues(Values(5000 * 201, 6), 5000, 201, Device.Cpu);
                var packed = bf16.Packed.ToArray();
                bf16.Packed.Dispose();
                float[] ids = [4999, 0, 2500, 3333, 1, 4096];
                var (gp, cp, gj, cj) = (On(backend, packed), On(cpu, packed), On(backend, ids), On(cpu, ids));
                var (gz, cz) = (Empty(backend, ids.Length * 201), Empty(cpu, ids.Length * 201));
                backend.GatherBFloat16(gp, gj, gz, ids.Length, 201, 5000);
                cpu.GatherBFloat16(cp, cj, cz, ids.Length, 201, 5000);
                Check(Read(backend, gz, ids.Length * 201).SequenceEqual(Read(cpu, cz, ids.Length * 201)), "gather from a large bfloat16 table");

                // The same words as a [5000, 201] table read by columns (a tied head's weight as the embedding table).
                float[] columns = [200, 0, 101, 57, 1, 199];
                var (gc, cc) = (On(backend, columns), On(cpu, columns));
                var (gw, cw) = (Empty(backend, columns.Length * 5000), Empty(cpu, columns.Length * 5000));
                backend.GatherBFloat16Columns(gp, gc, gw, columns.Length, 5000, 201);
                cpu.GatherBFloat16Columns(cp, cc, cw, columns.Length, 5000, 201);
                Check(Read(backend, gw, columns.Length * 5000).SequenceEqual(Read(cpu, cw, columns.Length * 5000)), "column gather from a large bfloat16 table");
            }

            // Dequantizations whose float32 output (and int8, bfloat16 weights) exceed a binding.
            {
                const int K = 520, N = 1003;
                var w = Values(K * N, 9);
                var int8 = Int8Weight.Quantize(w, K, N, Device.Cpu);
                var int4 = Int4Weight.Quantize(w, K, N, Device.Cpu);
                var bf16 = BFloat16Weight.FromValues(w, K, N, Device.Cpu);
                var (gy, cy) = (Empty(backend, K * N), Empty(cpu, K * N));
                backend.Int8Dequantize(On(backend, int8.Packed.ToArray()), On(backend, int8.Scales.ToArray()), gy, K, N);
                cpu.Int8Dequantize(On(cpu, int8.Packed.ToArray()), On(cpu, int8.Scales.ToArray()), cy, K, N);
                Check(Read(backend, gy, K * N).SequenceEqual(Read(cpu, cy, K * N)), "int8 dequantization");
                backend.Int4Dequantize(On(backend, int4.Packed.ToArray()), On(backend, int4.Scales.ToArray()), gy, K, N);
                cpu.Int4Dequantize(On(cpu, int4.Packed.ToArray()), On(cpu, int4.Scales.ToArray()), cy, K, N);
                Check(Read(backend, gy, K * N).SequenceEqual(Read(cpu, cy, K * N)), "int4 dequantization");
                backend.BFloat16Dequantize(On(backend, bf16.Packed.ToArray()), gy, K, N);
                cpu.BFloat16Dequantize(On(cpu, bf16.Packed.ToArray()), cy, K, N);
                Check(Read(backend, gy, K * N).SequenceEqual(Read(cpu, cy, K * N)), "bfloat16 dequantization");
                int8.Packed.Dispose();
                int8.Scales.Dispose();
                int4.Packed.Dispose();
                int4.Scales.Dispose();
                bf16.Packed.Dispose();
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

    private static void VulkanLargeDecoder(Device device)
    {
        if (LimitOrdinals(device).Count == 0)
        {
            return;
        }

        // Vocabulary 4096, dim 96: the bfloat16 embedding (768 KiB), the int8 (384 KiB) and int4 (192 KiB) heads exceed a
        // 128 KiB binding; every other storage of a decoding step fits.
        var backend = (VulkanBackend)device.Backend;
        var spec = new DecoderSpec
        {
            Vocabulary = 4096, Dim = 96, Layers = 2, Heads = 4, KvHeads = 2, HeadDim = 16, FfDim = 192, MaxPositions = 32,
            Rope = new RopeSettings(500f), NormEpsilon = 1e-5f, QkNorm = true,
        };
        string cache = Path.Combine(Path.GetTempPath(), $"idrak-limits-{Environment.ProcessId}.tsv");
        string? savedCache = VulkanBackend.TuningCacheFile;
        VulkanBackend.TuningCacheFile = cache;                               // choices measured under the small range kept apart
        long savedRange = backend.LimitStorageRange(128 << 10);
        var counts = new ConcurrentDictionary<string, long>();
        try
        {
            foreach (var (name, options, tied) in new[]
            {
                ("int8", new DecoderBuildOptions { Seed = 4, Int8 = true }, false),
                ("int4", new DecoderBuildOptions { Seed = 4, Int4 = true }, false),
                ("bfloat16", new DecoderBuildOptions { Seed = 4, BFloat16 = true }, false),
                ("bfloat16 tied (the head's columns as the table)", new DecoderBuildOptions { Seed = 4, BFloat16 = true }, true),
            })
            {
                var built = spec with { TieEmbeddings = tied };
                using var gpu = built.Build(null, options with { Device = device });
                using var host = built.Build(null, options with { Device = Device.Cpu });
                gpu.Eval();
                host.Eval();
                using var noGrad = Autograd.NoGrad();
                using var gpuContext = new DecodingContext(device, 1, 32) { LastPositionOnly = true };
                using var hostContext = new DecodingContext(Device.Cpu, 1, 32) { LastPositionOnly = true };
                float[] tokens = [1f, 2000f, 4095f];
                for (int step = 0; step < 5; step++)
                {
                    if (step == 1)
                    {
                        counts.Clear();
                        CountHostCalls(backend, counts);                // the decoding steps after the prompt
                    }

                    using var scope = new TensorScope();
                    var want = host.ForwardCached(Tensor.From(tokens, [1, tokens.Length], Device.Cpu), hostContext).ToArray();
                    var got = gpu.ForwardCached(Tensor.From(tokens, [1, tokens.Length], device), gpuContext).ToArray();
                    double scale = want.Max(v => Math.Abs((double)v)), worst = 0;
                    for (int i = 0; i < want.Length; i++)
                    {
                        worst = Math.Max(worst, Math.Abs(want[i] - (double)got[i]));
                    }

                    Check(worst <= 2e-3 * Math.Max(scale, 1e-3), $"{name} decoder, step {step}: logits off by {worst:G3} (largest {scale:G3})");
                    int next = Array.IndexOf(want, want.Max());
                    tokens = [next];
                }

                CountHostCalls(backend, null);
                Check(counts.IsEmpty, $"{name} decoder: host fallbacks " + string.Join(", ", counts.Select(c => $"{c.Key} x{c.Value}")));
            }
        }
        finally
        {
            CountHostCalls(backend, null);
            backend.LimitStorageRange(savedRange);
            VulkanBackend.TuningCacheFile = savedCache;
            File.Delete(cache);
        }
    }

    private static void VulkanSubgroupSizes(Device device)
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
                    if (first.SubgroupSizeControl && first.Limits.SubgroupArithmetic && VulkanBackend.Autotune)
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

    private static void VulkanWidthMeasured(Device device)
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
