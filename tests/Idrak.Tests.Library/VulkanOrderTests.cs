// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Diagnostics;
using Idrak.Gpu.Vulkan;

// Every choice a tuning op of the Vulkan backend can make gives the same bits (docs/optimization.md rule 86): the
// measurement decides only speed, so two processes that measured different choices (two cluster ranks, two drivers of
// one card) agree. Each test runs every choice of one op, forced through the backend's own overrides, at every width the
// device allows with and without subgroups, and compares each output with the first configuration's bit for bit; a
// failure names the op, the case, the choice and the first differing index. IDRAK_ORDER_TIMES=1 also prints each
// choice's median time (a bench of the variants on the device in use).
internal abstract partial class Tests
{
    // One case of an op under one choice: Execute runs it (setting the choice's overrides and clearing them after), Read
    // downloads its outputs, Dispose frees its storages.
    internal sealed record OrderCase(Action Execute, Func<float[][]> Read, Action Dispose);

    internal sealed record OrderRun(string Case, string Choice, Func<VulkanBackend, OrderCase> Make);

    // Runs `runs(backend)` on a backend of every width and both reductions, comparing every case's outputs over every
    // choice and configuration with the first one's.
    internal static void VulkanOrderOver(Device device, string op, Func<VulkanBackend, IEnumerable<OrderRun>> runs)
    {
        List<int> ordinals = device.Type switch
        {
            DeviceType.Vulkan => [device.Ordinal],
            DeviceType.Cpu => [.. Enumerable.Range(0, VulkanBackend.DeviceCount)],
            _ => [],
        };
        if (ordinals.Count == 0 || Environment.GetEnvironmentVariable("IDRAK_VULKAN_KERNELS") is "0" or "false")
        {
            Skip(ordinals.Count > 0 ? "IDRAK_VULKAN_KERNELS=0: the Vulkan kernels are off"
                : device.Type == DeviceType.Cpu ? $"no Vulkan device: {VulkanBackend.UnavailableReason}" : $"Vulkan devices are checked from the CPU's pass and their own (vulkan:N), not from {device}");
        }

        bool timing = Environment.GetEnvironmentVariable("IDRAK_ORDER_TIMES") is "1" or "true";
        bool savedAutotune = VulkanBackend.Autotune;
        var savedWidth = DeviceLimits.WidthOverride;
        VulkanBackend.Autotune = false;                                    // forced choices only; nothing measured or stored
        try
        {
            foreach (int ordinal in ordinals)
            {
                var reference = new Dictionary<string, (string Label, float[][] Values)>();
                var differences = new List<string>();
                var seen = new List<string>();
                int compared = 0;
                for (int width = VulkanKernels.MinWidth; width <= VulkanKernels.MaxWidth; width *= 2)
                {
                    foreach (bool subgroups in new[] { true, false })
                    {
                        DeviceLimits.WidthOverride = width;
                        DeviceLimits.SubgroupsOverride = subgroups ? null : false;
                        VulkanBackend backend;
                        try
                        {
                            backend = VulkanBackend.CreateSeparate(ordinal, preferMapped: true);
                        }
                        finally
                        {
                            DeviceLimits.SubgroupsOverride = null;
                        }

                        try
                        {
                            if (backend.Width != width || subgroups && !backend.Limits.SubgroupReductions)
                            {
                                continue;                                  // a width past the device's limits, or no subgroups
                            }

                            string config = $"width {width}{(subgroups ? $", butterfly {backend.Limits.Butterfly}" : ", workgroup memory")}";
                            seen.Add(config);
                            foreach (var run in runs(backend))
                            {
                                var made = run.Make(backend);
                                try
                                {
                                    made.Execute();
                                    var values = made.Read();
                                    string label = $"{config}, {run.Choice}";
                                    if (timing)
                                    {
                                        Console.WriteLine($"    time {op}: {run.Case}: {label}: {OrderMedianMs(backend, made.Execute):F3} ms");
                                    }

                                    if (!reference.TryGetValue(run.Case, out var known))
                                    {
                                        reference[run.Case] = (label, values);
                                        continue;
                                    }

                                    compared++;
                                    for (int o = 0; o < values.Length; o++)
                                    {
                                        var (a, b) = (values[o], known.Values[o]);
                                        int first = -1, differ = 0;
                                        for (int i = 0; i < a.Length; i++)
                                        {
                                            if (BitConverter.SingleToInt32Bits(a[i]) != BitConverter.SingleToInt32Bits(b[i]))
                                            {
                                                differ++;
                                                first = first < 0 ? i : first;
                                            }
                                        }

                                        if (differ > 0)
                                        {
                                            differences.Add($"{op}: {run.Case}: {label}: output {o}: {differ} of {a.Length} values differ from {known.Label}; the first, index {first}, is {a[first]:R} against {b[first]:R}");
                                        }
                                    }
                                }
                                finally
                                {
                                    made.Dispose();
                                }
                            }
                        }
                        finally
                        {
                            backend.Shutdown();
                        }
                    }
                }

                Check(seen.Count >= 2, $"vulkan:{ordinal}: {op}: only {seen.Count} configuration ({string.Join(", ", seen)}) to compare");
                Check(differences.Count == 0, $"vulkan:{ordinal}: {differences.Count} outputs differ; {string.Join(" | ", differences.Take(6))}");
                Console.WriteLine($"    vulkan:{ordinal}: {op}: {reference.Count} cases bit for bit over {compared} other choices and configurations ({string.Join("; ", seen)})");
            }
        }
        finally
        {
            VulkanBackend.Autotune = savedAutotune;
            DeviceLimits.WidthOverride = savedWidth;
        }
    }

    // Median milliseconds of one run of `execute` (as many back to back as take about 20 ms, five rounds).
    internal static double OrderMedianMs(VulkanBackend backend, Action execute)
    {
        backend.Synchronize();
        var watch = Stopwatch.StartNew();
        execute();
        backend.Synchronize();
        double once = Math.Max(watch.Elapsed.TotalMilliseconds, 1e-3);
        int repeats = (int)Math.Clamp(Math.Ceiling(20 / once), 1, 64);
        var times = new double[5];
        for (int round = 0; round < times.Length; round++)
        {
            watch.Restart();
            for (int i = 0; i < repeats; i++)
            {
                execute();
            }

            backend.Synchronize();
            times[round] = watch.Elapsed.TotalMilliseconds / repeats;
        }

        Array.Sort(times);
        return times[2];
    }

    // Storages of one case on a backend: uploaded inputs, zeroed outputs, downloaded at the end, freed together.
    internal sealed class OrderStorages(VulkanBackend backend)
    {
        private readonly List<Storage> _made = [];

        public Storage Up(float[] values)
        {
            var s = backend.Allocate(values.Length, false);
            backend.Upload(values, s);
            _made.Add(s);
            return s;
        }

        public Storage Out(int n)
        {
            var s = backend.Allocate(n, true);
            _made.Add(s);
            return s;
        }

        public float[] Down(Storage s)
        {
            var values = new float[s.Length];
            backend.Download(s, values);
            return values;
        }

        public void Dispose()
        {
            foreach (var s in _made)
            {
                s.Release();
            }
        }
    }

    internal static float[] OrderValues(Random random, int n, float lo = -1f, float hi = 1f) => [.. Enumerable.Range(0, n).Select(_ => lo + (hi - lo) * random.NextSingle())];

    internal static float[] OrderBits(Random random, int n) => [.. Enumerable.Range(0, n).Select(_ => BitConverter.Int32BitsToSingle(random.Next() ^ (random.Next() << 16)))];

    // Words of two bfloat16 values in [-1, 1).
    internal static float[] OrderBFloat16(Random random, int words) => [.. Enumerable.Range(0, words).Select(_ =>
    {
        uint low = (uint)BitConverter.SingleToInt32Bits(2 * random.NextSingle() - 1) >> 16;
        uint high = (uint)BitConverter.SingleToInt32Bits(2 * random.NextSingle() - 1) & 0xFFFF0000u;
        return BitConverter.Int32BitsToSingle((int)(low | high));
    })];

    internal static readonly int[] OrderSplits = [1, 2, 4, 8, 16, 32, 64];

    // ------------------------------------------------------------------ float32 products (VulkanTuneOp.MatMul)

    internal static void VulkanOrderMatMul(Device device) => VulkanOrderOver(device, "float32 products", backend =>
    {
        var random = new Random(47);
        var runs = new List<OrderRun>();
        foreach (var (batch, m, n, k) in new[] { (2, 33, 70, 100), (1, 130, 65, 257), (3, 5, 9, 1000) })
        {
            foreach (var (ta, tb) in new[] { (false, false), (true, false), (false, true), (true, true) })
            {
                foreach (float beta in new[] { 0f, 0.5f })
                {
                    var a = OrderValues(random, batch * m * k);
                    var bv = OrderValues(random, batch * k * n);
                    var c0 = OrderValues(random, batch * m * n);
                    foreach (var (kernel, name) in new[] { (0, "small"), (1, "tiled"), (2, "register-blocked") })
                    {
                        runs.Add(new OrderRun($"{batch} × {m}×{k}·{k}×{n}{(ta ? ", a transposed" : "")}{(tb ? ", b transposed" : "")}, beta {beta}", name, b =>
                        {
                            var s = new OrderStorages(b);
                            var (sa, sb) = (s.Up(a), s.Up(bv));
                            var c = s.Up(c0);
                            return new OrderCase(() =>
                            {
                                b.Upload(c0, c);                           // beta reads c: the same c every run
                                VulkanBackend.MatMulKernel = kernel;
                                try
                                {
                                    b.BatchedMatMulKernel(sa, sb, c, batch, m, n, k, ta, tb, beta);
                                }
                                finally
                                {
                                    VulkanBackend.MatMulKernel = null;
                                }
                            }, () => [s.Down(c)], s.Dispose);
                        }));
                    }
                }
            }
        }

        return runs;
    });

    // ------------------------------------------------------------------ decoding attention (VulkanTuneOp.Attention)

    internal static void VulkanOrderAttention(Device device) => VulkanOrderOver(device, "decoding attention", backend =>
    {
        var random = new Random(59);
        var runs = new List<OrderRun>();
        // (head size, heads, rows per head, steps, capacity, position, window): one row and grouped rows, a head of 6, 64
        // and 256, positions past several blocks, windows shorter than what a row sees.
        foreach (var (dim, heads, rowsPerHead, steps, capacity, position, window) in new[]
                 {
                     (64, 2, 1, 1, 300, 250, 0), (6, 3, 2, 2, 200, 120, 0), (128, 1, 4, 2, 520, 500, 100), (256, 2, 1, 1, 90, 60, 0), (64, 1, 2, 1, 1100, 1050, 0),
                 })
        {
            int rows = heads * rowsPerHead, slots = heads * capacity;
            var q = OrderValues(random, rows * dim, -2f, 2f);
            var keysF = OrderValues(random, slots * dim);
            var valuesF = OrderValues(random, slots * dim);
            // Rows of different lengths (one head per row): each head's own start per step, the last head all padding
            // where there are several.
            var starts = Enumerable.Range(0, heads * steps).Select(i => heads > 1 && i / steps == heads - 1
                ? (float)(position + i % steps + 5) : random.Next(0, position + i % steps + 1)).ToArray();
            var variant = new AttentionVariant(window, window > 0 ? 30f : 0f);
            float scale = 1f / MathF.Sqrt(dim);
            foreach (string cache in new[] { "float32", "float32 per-row starts" })
            {
                foreach (int splits in OrderSplits)
                {
                    runs.Add(new OrderRun($"{cache} cache, head size {dim}, {heads} heads × {rowsPerHead} rows, capacity {capacity}, position {position}{(window > 0 ? $", window {window}, soft-cap 30" : "")}",
                        $"{splits} splits", b =>
                        {
                            var s = new OrderStorages(b);
                            var (qs, pos, y) = (s.Up(q), s.Up([position]), s.Out(rows * dim));
                            Storage keys, values;
                            Storage? st = null;
                            switch (cache)
                            {
                                case "float32":
                                    (keys, values) = (s.Up(keysF), s.Up(valuesF));
                                    break;
                                default:
                                    (keys, values, st) = (s.Up(keysF), s.Up(valuesF), s.Up(starts));
                                    break;
                            }

                            return new OrderCase(() =>
                            {
                                VulkanBackend.AttentionSplits = splits;
                                VulkanBackend.RowsAttentionKernel = false;               // the decoding kernel, whose splits these are
                                try
                                {
                                    switch (cache)
                                    {
                                        case "float32":
                                            b.AttentionDecodeKernel(qs, keys, values, pos, y, heads, rowsPerHead, steps, capacity, dim, scale, variant);
                                            break;
                                        default:
                                            b.AttentionRowsKernel(qs, keys, values, pos, y, st!, heads, 1, rowsPerHead, steps, capacity, dim, scale, variant);
                                            break;
                                    }
                                }
                                finally
                                {
                                    VulkanBackend.AttentionSplits = null;
                                    VulkanBackend.RowsAttentionKernel = null;
                                }
                            }, () => [s.Down(y)], s.Dispose);
                        }));
                }
            }
        }

        return runs;
    });

    // ------------------------------------------------------------------ two-pass reductions (VulkanTuneOp.Splits)

    internal static void VulkanOrderSplits(Device device) => VulkanOrderOver(device, "two-pass reductions", backend =>
    {
        var random = new Random(61);
        var runs = new List<OrderRun>();
        void Add(string name, Func<OrderStorages, VulkanBackend, (Action Run, Storage[] Outputs)> make)
        {
            foreach (int splits in OrderSplits.Append(256))
            {
                runs.Add(new OrderRun(name, $"{splits} splits", b =>
                {
                    var s = new OrderStorages(b);
                    var (run, outputs) = make(s, b);
                    return new OrderCase(() =>
                    {
                        foreach (var o in outputs)
                        {
                            b.Upload(new float[o.Length], o);              // accumulating outputs start from 0 every run
                        }

                        VulkanBackend.ReductionSplits = splits;
                        try
                        {
                            run();
                        }
                        finally
                        {
                            VulkanBackend.ReductionSplits = null;
                        }
                    }, () => [.. outputs.Select(s.Down)], s.Dispose);
                }));
            }
        }

        foreach (var (rows, cols) in new[] { (1000, 37), (4099, 5) })
        {
            var x = OrderValues(random, rows * cols, -2f, 3f);
            var dy = OrderValues(random, rows * cols);
            var gamma = OrderValues(random, cols);
            Add($"column sums of {rows}×{cols}", (s, b) =>
            {
                var (xs, y) = (s.Up(x), s.Out(cols));
                return (() => b.SumColumnsKernel(xs, 0, cols, y, rows, cols), [y]);
            });
            Add($"layer-norm gamma and beta gradients of {rows}×{cols}", (s, b) =>
            {
                var stats = new float[2 * rows];
                for (int r = 0; r < rows; r++)
                {
                    (stats[r], stats[rows + r]) = (0.1f * (r % 7), 1f + 0.01f * (r % 5));
                }

                var (xs, g, d, st, dgamma, dbeta) = (s.Up(x), s.Up(gamma), s.Up(dy), s.Up(stats), s.Out(cols), s.Out(cols));
                return (() => b.LayerNormBackwardKernel(xs, g, d, st, null, dgamma, dbeta, rows, cols), [dgamma, dbeta]);
            });
        }

        // Group statistics: runs shorter than any width (the column sums) and longer than every width (a workgroup per group).
        foreach (var (outer, groups, inner) in new[] { (300, 3, 7), (5, 2, 3000), (2, 3, 1100) })
        {
            var x = OrderValues(random, outer * groups * inner, -1f, 4f);
            var b2 = OrderValues(random, outer * groups * inner);
            Add($"group moments of {outer}×{groups}×{inner}", (s, b) =>
            {
                var (xs, mean, variance, inv) = (s.Up(x), s.Out(groups), s.Out(groups), s.Out(groups));
                return (() => b.NormStatsKernel(xs, mean, variance, inv, outer, groups, inner, 1e-5f), [mean, variance, inv]);
            });
            Add($"group sums of {outer}×{groups}×{inner}", (s, b) =>
            {
                var (xs, bs, sumA, sumAB) = (s.Up(x), s.Up(b2), s.Out(groups), s.Out(groups));
                return (() => b.GroupReduceKernel(xs, bs, sumA, sumAB, outer, groups, inner), [sumA, sumAB]);
            });
        }

        return runs;
    });
}
