// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Idrak;
using Idrak.Backends;
using Idrak.Abstraction.Devices.Cpu;
using Idrak.Backends.Vulkan;

// The Vulkan kernels shaped by the device: every workgroup width a device may be given, the products split over k, the
// decoding attention split over the cached positions, the tiled float32 products, and the choices measured on the
// device (kept in a file per device). They match the CPU for every width, split count and edge shape, and give
// bit-identical results when run again. A software driver runs a workgroup's invocations one after another and hides
// races on workgroup memory; on a GPU a missing barrier shows up as results that change from run to run, which the
// repeated runs catch.
internal static partial class Tests
{
    private static void VulkanSplitKernels(Device device)
    {
        List<int> ordinals = device.Type switch
        {
            DeviceType.Vulkan => [device.Ordinal],
            DeviceType.Cpu => [.. Enumerable.Range(0, VulkanBackend.DeviceCount)],
            _ => [],
        };
        if (ordinals.Count == 0)
        {
            if (device.Type == DeviceType.Cpu)
            {
                Console.WriteLine($"    (no Vulkan device: {VulkanBackend.UnavailableReason}; skipped)");
            }

            return;
        }

        if (Environment.GetEnvironmentVariable("IDRAK_VULKAN_KERNELS") is "0" or "false")
        {
            Console.WriteLine("    (IDRAK_VULKAN_KERNELS=0: kernels off, skipped)");
            return;
        }

        // Measured choices go to a file of this test's own, removed afterwards.
        string cache = Path.Combine(Path.GetTempPath(), $"idrak-tuning-{Environment.ProcessId}.tsv");
        string? savedCache = VulkanBackend.TuningCacheFile;
        var savedWidth = DeviceLimits.WidthOverride;
        VulkanBackend.TuningCacheFile = cache;
        try
        {
            foreach (int ordinal in ordinals)
            {
                // The device's own width (mapped memory, then staging), then every other width its limits allow.
                DeviceLimits.WidthOverride = null;
                int deviceWidth = WithBackend(ordinal, mapped: true, b => SplitKernelCases(b, full: true));
                WithBackend(ordinal, mapped: false, b => SplitKernelCases(b, full: false));
                for (int width = VulkanKernels.MinWidth; width <= VulkanKernels.MaxWidth; width *= 2)
                {
                    DeviceLimits.WidthOverride = width;
                    if (width != deviceWidth)
                    {
                        int w = width;
                        WithBackend(ordinal, mapped: true, b =>
                        {
                            if (b.Width == w)
                            {
                                SplitKernelCases(b, full: false);
                            }
                        });
                    }
                }

                // Reductions through workgroup memory alone (as on devices without subgroup arithmetic).
                DeviceLimits.WidthOverride = null;
                DeviceLimits.SubgroupsOverride = false;
                try
                {
                    WithBackend(ordinal, mapped: true, b => SplitKernelCases(b, full: false));
                }
                finally
                {
                    DeviceLimits.SubgroupsOverride = null;
                }

                // Stored choices: a second backend of the same device and width measures nothing again.
                DeviceLimits.WidthOverride = null;
                WithBackend(ordinal, mapped: true, b =>
                {
                    Check(File.Exists(cache) && File.ReadLines(cache).Any(l => l.StartsWith("kernels/", StringComparison.Ordinal)), $"vulkan:{ordinal}: no stored kernel choices in {cache}");
                    long measured = b.Measurements;
                    SplitKernelCases(b, full: false);
                    Check(b.Measurements == measured, $"vulkan:{ordinal}: {b.Measurements - measured} choices measured again despite the stored ones");
                });
            }
        }
        finally
        {
            DeviceLimits.WidthOverride = savedWidth;
            VulkanBackend.TuningCacheFile = savedCache;
            File.Delete(cache);
        }
    }

    // Runs `check` on a separate backend of the device (released afterwards); returns its width.
    private static int WithBackend(int ordinal, bool mapped, Action<VulkanBackend> check)
    {
        var backend = VulkanBackend.CreateSeparate(ordinal, preferMapped: mapped);
        try
        {
            check(backend);
            return backend.Width;
        }
        finally
        {
            backend.Shutdown();
        }
    }

    private static void SplitKernelCases(VulkanBackend backend, bool full)
    {
        string label = $"width {backend.Width}{(backend.Limits.SubgroupArithmetic ? ", subgroups" : "")} ({(backend.UnifiedMemory ? "mapped" : "staging")})";
        var cpu = CpuBackend.Instance;
        var random = new Random(17);
        float[] R(int n, float lo = -1f, float hi = 1f) => [.. Enumerable.Range(0, n).Select(_ => lo + (hi - lo) * random.NextSingle())];
        float[] Bits(int n) => [.. Enumerable.Range(0, n).Select(_ => BitConverter.Int32BitsToSingle(random.Next() ^ (random.Next() << 16)))];
        float[] BFloat16Pairs(int words) => [.. Enumerable.Range(0, words).Select(_ =>
        {
            uint low = (uint)BitConverter.SingleToInt32Bits(2 * random.NextSingle() - 1) >> 16;
            uint high = (uint)BitConverter.SingleToInt32Bits(2 * random.NextSingle() - 1) & 0xFFFF0000u;
            return BitConverter.Int32BitsToSingle((int)(low | high));
        })];

        // Runs op on the backend's copies of inputs and returns storage `output` after it.
        float[] Run(Backend b, float[][] inputs, int output, Action<Backend, Storage[]> op)
        {
            var storages = inputs.Select(d => { var s = b.Allocate(d.Length, false); b.Upload(d, s); return s; }).ToArray();
            try
            {
                op(b, storages);
                var result = new float[inputs[output].Length];
                b.Download(storages[output], result);
                return result;
            }
            finally
            {
                foreach (var s in storages)
                {
                    s.Release();
                }
            }
        }

        // The CPU's result, the device's (no host fallback), close to it relative to the output's largest magnitude, and
        // the same bits on every repeated run.
        void Case(string what, float[][] inputs, int output, Action<Backend, Storage[]> op, int repeats, float tolerance = 1e-5f)
        {
            var expected = Run(cpu, inputs, output, op);
            long fallbacks = Idrak.Abstraction.Operations.Kernels.HostCalls(backend);
            var actual = Run(backend, inputs, output, op);
            Check(Idrak.Abstraction.Operations.Kernels.HostCalls(backend) == fallbacks, $"{label}: {what} took the host fallback");
            float scale = Math.Max(1f, expected.Max(MathF.Abs));
            for (int i = 0; i < expected.Length; i++)
            {
                if (!(MathF.Abs(expected[i] - actual[i]) <= tolerance * scale))
                {
                    throw new Exception($"{label}: {what}: element {i} is {actual[i]}, expected {expected[i]}");
                }
            }

            for (int r = 0; r < repeats; r++)
            {
                var again = Run(backend, inputs, output, op);
                for (int i = 0; i < actual.Length; i++)
                {
                    if (BitConverter.SingleToInt32Bits(again[i]) != BitConverter.SingleToInt32Bits(actual[i]))
                    {
                        throw new Exception($"{label}: {what}: run {r + 2} gave element {i} = {again[i]}, run 1 {actual[i]} (a race?)");
                    }
                }
            }
        }

        try
        {
            // Packed products: every row block (1, 2, 4, 8 rows; 11 rows = two blocks of 8), k not a multiple of 32 (a
            // partial int4 scale group) or of the slices, columns not a multiple of a word or of a column block; one
            // split, the measured choice (null), splits whose last chunk is short; both word counts per row of k.
            var gemvShapes = full
                ? new[] { (1, 1000, 129), (2, 300, 37), (3, 64, 70), (5, 257, 300), (8, 96, 33), (11, 200, 65), (1, 3072, 64) }
                : [(1, 1000, 129), (5, 257, 300)];
            foreach (var (m, k, n) in gemvShapes)
            {
                foreach (int? splits in full ? new int?[] { 1, null, 3, 7 } : [null, 3])
                {
                    foreach (int? words in splits is null ? new int?[] { null } : [32, 64])
                    {
                        (VulkanBackend.GemvSplits, VulkanBackend.GemvWords) = (splits, words);
                        int repeats = splits == 3 ? 3 : 0;
                        string shape = $"{m}×{k}→{n}, splits {splits?.ToString() ?? "measured"}, words {words?.ToString() ?? "measured"}";
                        int words8 = (n + 3) / 4, words4 = (n + 7) / 8, words2 = (n + 1) / 2;
                        Case($"int8 product {shape}", [R(m * k), Bits(k * words8), R(n, 0.001f, 0.02f), new float[m * n]], 3,
                            (b, s) => b.Int8MatMul(s[0], s[1], s[2], s[3], m, n, k), repeats);
                        Case($"int4 product {shape}", [R(m * k), Bits(k * words4), R((k + 31) / 32 * words4 * 8, 0.01f, 0.2f), new float[m * n]], 3,
                            (b, s) => b.Int4MatMul(s[0], s[1], s[2], s[3], m, n, k), repeats);
                        Case($"bfloat16 product {shape}", [R(m * k), BFloat16Pairs(k * words2), new float[m * n]], 2,
                            (b, s) => b.BFloat16MatMul(s[0], s[1], s[2], m, n, k), repeats);
                    }
                }
            }

            (VulkanBackend.GemvSplits, VulkanBackend.GemvWords) = (null, null);

            // Decoding attention: head sizes from 1 to 256 (one to many parts of the values per dimension, or several
            // dimensions per invocation on narrow workgroups), rows of a head seeing different lengths (steps 2), lengths
            // from one position to more than a tile, splits from one to more than the positions need (empty splits).
            const int Heads = 2, RowsPerHead = 3, Steps = 2;
            var attentionShapes = full
                ? new[] { (1, 8, 1), (48, 40, 21), (64, 300, 299), (100, 700, 600), (128, 600, 520), (256, 300, 270), (33, 2048, 2000) }
                : [(48, 40, 21), (256, 1200, 1100)];
            foreach (var (dim, capacity, length) in attentionShapes)
            {
                foreach (int? splits in full ? new int?[] { 1, null, 5 } : [null, 5])
                {
                    VulkanBackend.AttentionSplits = splits;
                    int repeats = splits == 5 ? 3 : 0;
                    string shape = $"head size {dim}, {length} positions, splits {splits?.ToString() ?? "measured"}";
                    int rows = Heads * RowsPerHead, slots = Heads * capacity;
                    float[] position = [length - 1];
                    Case($"attention, float32 cache, {shape}", [R(rows * dim), R(slots * dim), R(slots * dim), position, new float[rows * dim]], 4,
                        (b, s) => b.AttentionDecode(s[0], s[1], s[2], s[3], s[4], Heads, RowsPerHead, Steps, capacity, dim, 0.3f), repeats);
                    int words8 = (dim + 3) / 4, words2 = (dim + 1) / 2;
                    Case($"attention, int8 cache, {shape}", [R(rows * dim), Bits(slots * words8), Bits(slots * words8), R(slots, 0.001f, 0.01f), R(slots, 0.001f, 0.01f), position, new float[rows * dim]], 6,
                        (b, s) => b.AttentionInt8(s[0], s[1], s[2], s[3], s[4], s[5], s[6], Heads, RowsPerHead, Steps, capacity, dim, 0.3f, tiled: false), repeats);
                    Case($"attention, bfloat16 cache, {shape}", [R(rows * dim), BFloat16Pairs(slots * words2), BFloat16Pairs(slots * words2), position, new float[rows * dim]], 4,
                        (b, s) => b.AttentionBFloat16(s[0], s[1], s[2], s[3], s[4], Heads, RowsPerHead, Steps, capacity, dim, 0.3f, tiled: false), repeats);
                }
            }

            VulkanBackend.AttentionSplits = null;

            // The float32 products (small, tiled, register-blocked, and the measured choice): partial blocks, k not a
            // multiple of the staging step, transposes, beta.
            var matShapes = full ? new[] { (1, 64, 64, 64), (2, 70, 130, 33), (3, 129, 65, 200), (1, 32, 200, 47) } : [(2, 70, 130, 33)];
            foreach (var (batch, m, n, k) in matShapes)
            {
                foreach (var (transA, transB) in new[] { (false, false), (true, false), (false, true), (true, true) })
                {
                    foreach (int? kernel in new int?[] { null, 0, 1, 2 })
                    {
                        VulkanBackend.MatMulKernel = kernel;
                        float beta = kernel == 1 ? 0f : 0.5f;
                        Case($"batched matmul {batch}×{m}×{n}×{k}, transA {transA}, transB {transB}, beta {beta}, kernel {kernel?.ToString() ?? "measured"}",
                            [R(batch * m * k), R(batch * k * n), R(batch * m * n)], 2,
                            (b, s) => b.BatchedMatMul(s[0], s[1], s[2], batch, m, n, k, transA, transB, beta), kernel == 2 ? 3 : 0);
                    }
                }
            }

            // Row kernels at a measured choice of narrow or wide, in place.
            VulkanBackend.MatMulKernel = null;
            foreach (int cols in new[] { 7, 300 })
            {
                Case($"softmax in place, 50 × {cols}", [R(50 * cols, -4, 4)], 0, (b, s) => b.Softmax(s[0], s[0], 50, cols, log: false), 1);
            }
        }
        finally
        {
            (VulkanBackend.GemvSplits, VulkanBackend.GemvWords, VulkanBackend.AttentionSplits, VulkanBackend.MatMulKernel) = (null, null, null, null);
        }
    }
}
