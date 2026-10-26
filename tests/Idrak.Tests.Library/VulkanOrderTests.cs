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
            if (device.Type == DeviceType.Cpu)
            {
                Console.WriteLine($"    (no Vulkan device or kernels off: {VulkanBackend.UnavailableReason}; skipped)");
            }

            return;
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

    // Packed weights of a format for [k, n]: (words, scales) as the products read them (scales null for bfloat16).
    internal static (float[] Words, float[]? Scales) OrderWeights(Random random, PackedFormat format, int k, int n) => format switch
    {
        PackedFormat.Int8 => (OrderBits(random, k * ((n + 3) / 4)), OrderValues(random, n, 0.001f, 0.02f)),
        PackedFormat.Int4 => (OrderBits(random, k * ((n + 7) / 8)), OrderValues(random, (k + 31) / 32 * ((n + 7) / 8) * 8, 0.01f, 0.2f)),
        _ => (OrderBFloat16(random, k * ((n + 1) / 2)), null),
    };

    internal static readonly PackedFormat[] OrderFormats = [PackedFormat.Int8, PackedFormat.Int4, PackedFormat.BFloat16];

    internal static readonly int[] OrderSplits = [1, 2, 4, 8, 16, 32, 64];

    // ------------------------------------------------------------------ packed products (VulkanTuneOp.Gemv)

    internal static void VulkanOrderGemv(Device device) => VulkanOrderOver(device, "packed products", backend =>
    {
        var random = new Random(41);
        var runs = new List<OrderRun>();
        // (m, k, n): one row, a block of 4 and of 8; k not a multiple of 32 (a partial int4 scale group), columns not a
        // multiple of a word.
        foreach (var (m, k, n) in new[] { (1, 1000, 77), (3, 2112, 130), (8, 520, 40) })
        {
            foreach (var format in OrderFormats)
            {
                var x = OrderValues(random, m * k);
                var (words, scales) = OrderWeights(random, format, k, n);
                foreach (int wordCount in VulkanKernels.GemvWordCounts)
                {
                    // The weights larger than a 64 KiB binding (k = 2112): windows of rows of k, a dispatch each.
                    foreach (var (splits, window) in OrderSplits.Select(s => (s, false)).Concat(k == 2112 ? [(1, true), (8, true)] : []))
                    {
                        runs.Add(new OrderRun($"{format} {m}×{k}→{n}", $"{wordCount} words, {splits} splits{(window ? ", weights in windows of a 64 KiB binding" : "")}", b =>
                        {
                            var s = new OrderStorages(b);
                            var (xs, q, sc, y) = (s.Up(x), s.Up(words), scales is null ? null : s.Up(scales), s.Out(m * n));
                            return new OrderCase(() =>
                            {
                                (VulkanBackend.GemvSplits, VulkanBackend.GemvWords) = (splits, wordCount);
                                long range = window ? b.LimitStorageRange(64 << 10) : 0;
                                try
                                {
                                    switch (format)
                                    {
                                        case PackedFormat.Int8:
                                            b.Int8MatMulKernel(xs, q, sc!, y, m, n, k);
                                            break;
                                        case PackedFormat.Int4:
                                            b.Int4MatMulKernel(xs, q, sc!, y, m, n, k);
                                            break;
                                        default:
                                            b.BFloat16MatMulKernel(xs, q, y, m, n, k);
                                            break;
                                    }
                                }
                                finally
                                {
                                    (VulkanBackend.GemvSplits, VulkanBackend.GemvWords) = (null, null);
                                    if (window)
                                    {
                                        b.LimitStorageRange(range);
                                    }
                                }
                            }, () => [s.Down(y)], s.Dispose);
                        }));
                    }
                }
            }
        }

        return runs;
    });

    // ------------------------------------------------------------------ fused packed products (VulkanTuneOp.FusedGemv)

    internal static void VulkanOrderFused(Device device) => VulkanOrderOver(device, "fused packed products", backend =>
    {
        var random = new Random(43);
        var runs = new List<OrderRun>();
        foreach (var (m, k) in new[] { (1, 1000), (5, 640) })
        {
            foreach (var format in OrderFormats)
            {
                int[] columns = [77, 40, 64];
                var x = OrderValues(random, m * k);
                var weights = columns.Select(n => OrderWeights(random, format, k, n)).ToArray();
                var upWeights = OrderWeights(random, format, k, columns[0]);
                var biases = columns.Select(n => OrderValues(random, n)).ToArray();
                var gate = OrderValues(random, m * k);
                var residual = OrderValues(random, m * columns[0]);
                var gain = OrderValues(random, columns[0]);
                foreach (int wordCount in VulkanKernels.GemvWordCounts)
                {
                    foreach (int splits in new[] { 1, 4, 32 })
                    {
                        string choice = $"{wordCount} words, {splits} splits";
                        void Forced(Action body)
                        {
                            (VulkanBackend.GemvSplits, VulkanBackend.GemvWords) = (splits, wordCount);
                            try
                            {
                                body();
                            }
                            finally
                            {
                                (VulkanBackend.GemvSplits, VulkanBackend.GemvWords) = (null, null);
                            }
                        }

                        runs.Add(new OrderRun($"{format} {m}×{k}, three products sharing an input", choice, b =>
                        {
                            var s = new OrderStorages(b);
                            var xs = s.Up(x);
                            var products = Enumerable.Range(0, 3).Select(p => (Packed: s.Up(weights[p].Words), Scales: weights[p].Scales is null ? null : s.Up(weights[p].Scales!),
                                Bias: p == 1 ? null : s.Up(biases[p]), Output: s.Out(m * columns[p]), Columns: columns[p])).ToArray();
                            return new OrderCase(() => Forced(() => Check(b.PackedMatMulManyKernel(format, xs, m, k, products.Select(p => (p.Packed, p.Scales, p.Bias, p.Output, p.Columns)).ToArray()),
                                "the fused products declined")), () => [.. products.Select(p => s.Down(p.Output))], s.Dispose);
                        }));
                        runs.Add(new OrderRun($"{format} {m}×{k}, gate and up with GELU", choice, b =>
                        {
                            var s = new OrderStorages(b);
                            var xs = s.Up(x);
                            int n = columns[0];
                            var products = new[] { weights[0], upWeights }.Select((w, p) => (Packed: s.Up(w.Words), Scales: w.Scales is null ? null : s.Up(w.Scales),
                                Bias: p == 0 ? s.Up(biases[0]) : null, Output: s.Out(m * n), Columns: n)).ToArray();
                            var hidden = s.Out(m * n);
                            return new OrderCase(() => Forced(() => Check(b.PackedMatMulGatedPairKernel(format, 1, xs, m, k, products.Select(p => (p.Packed, p.Scales, p.Bias, p.Output, p.Columns)).ToArray(), hidden),
                                "the fused pair declined")), () => [s.Down(products[0].Output), s.Down(products[1].Output), s.Down(hidden)], s.Dispose);
                        }));
                        runs.Add(new OrderRun($"{format} {m}×{k}, the gated down projection", choice, b =>
                        {
                            var s = new OrderStorages(b);
                            var (g, u, q) = (s.Up(gate), s.Up(x), s.Up(weights[0].Words));
                            var sc = weights[0].Scales is null ? null : s.Up(weights[0].Scales!);
                            var y = s.Out(m * columns[0]);
                            return new OrderCase(() => Forced(() => Check(b.PackedMatMulGatedKernel(format, 0, g, u, q, sc, y, m, columns[0], k), "the gated product declined")),
                                () => [s.Down(y)], s.Dispose);
                        }));
                        runs.Add(new OrderRun($"{format} {m}×{k}, projection, residual and RMS norm", choice, b =>
                        {
                            var s = new OrderStorages(b);
                            int n = columns[0];
                            var (xs, q) = (s.Up(x), s.Up(weights[0].Words));
                            var sc = weights[0].Scales is null ? null : s.Up(weights[0].Scales!);
                            var (y, res, sum, gn, normalized) = (s.Out(m * n), s.Up(residual), s.Out(m * n), s.Up(gain), s.Out(m * n));
                            return new OrderCase(() => Forced(() => Check(b.PackedMatMulAddRmsNormKernel(format, xs, q, sc, y, m, n, k, res, sum, gn, normalized, 1e-6f, 0f),
                                "the fused norm declined")), () => [s.Down(y), s.Down(sum), s.Down(normalized)], s.Dispose);
                        }));
                    }
                }
            }
        }

        return runs;
    });

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

    // ------------------------------------------------------------------ prompt-sized packed products (VulkanTuneOp.PackedPrompt)

    internal static void VulkanOrderPrompt(Device device) => VulkanOrderOver(device, "prompt-sized packed products", backend =>
    {
        var random = new Random(53);
        var runs = new List<OrderRun>();
        foreach (var (m, k, n) in new[] { (33, 100, 65), (20, 300, 130), (70, 33, 7) })
        {
            foreach (var format in OrderFormats)
            {
                var x = OrderValues(random, m * k);
                var (words, scales) = OrderWeights(random, format, k, n);
                foreach (var (kernel, name) in new[] { (0, "few-rows kernels"), (1, "expanded weights"), (2, "tiled") })
                {
                    foreach (var (matmul, matName) in kernel == 1 ? new[] { (0, "small"), (1, "tiled"), (2, "register-blocked") } : [(-1, "")])
                    {
                        runs.Add(new OrderRun($"{format} {m}×{k}→{n}", matmul < 0 ? name : $"{name} through the {matName} float32 product", b =>
                        {
                            var s = new OrderStorages(b);
                            var (xs, q, sc, y) = (s.Up(x), s.Up(words), scales is null ? null : s.Up(scales), s.Out(m * n));
                            return new OrderCase(() =>
                            {
                                VulkanBackend.PackedPromptKernel = kernel;
                                VulkanBackend.MatMulKernel = matmul < 0 ? null : matmul;
                                try
                                {
                                    Check(b.PackedMatMulLargeKernel(format, xs, q, sc, y, m, n, k), $"{format} {m}×{k}→{n}: {name}: the prompt product declined");
                                }
                                finally
                                {
                                    (VulkanBackend.PackedPromptKernel, VulkanBackend.MatMulKernel) = (null, null);
                                }
                            }, () => [s.Down(y)], s.Dispose);
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
            int bfWords = (dim + 1) / 2, i8Words = (dim + 3) / 4;
            var keysB = OrderBFloat16(random, slots * bfWords);
            var valuesB = OrderBFloat16(random, slots * bfWords);
            var keysI = OrderBits(random, slots * i8Words);
            var valuesI = OrderBits(random, slots * i8Words);
            var keyScales = OrderValues(random, slots, 0.01f, 0.05f);
            var valueScales = OrderValues(random, slots, 0.01f, 0.05f);
            // Rows of different lengths (one head per row): each head's own start per step, the last head all padding
            // where there are several.
            var starts = Enumerable.Range(0, heads * steps).Select(i => heads > 1 && i / steps == heads - 1
                ? (float)(position + i % steps + 5) : random.Next(0, position + i % steps + 1)).ToArray();
            var variant = new AttentionVariant(window, window > 0 ? 30f : 0f);
            float scale = 1f / MathF.Sqrt(dim);
            foreach (string cache in new[] { "float32", "bfloat16", "int8", "float32 per-row starts", "bfloat16 per-row starts" })
            {
                foreach (int splits in OrderSplits)
                {
                    runs.Add(new OrderRun($"{cache} cache, head size {dim}, {heads} heads × {rowsPerHead} rows, capacity {capacity}, position {position}{(window > 0 ? $", window {window}, soft-cap 30" : "")}",
                        $"{splits} splits", b =>
                        {
                            var s = new OrderStorages(b);
                            var (qs, pos, y) = (s.Up(q), s.Up([position]), s.Out(rows * dim));
                            Storage keys, values;
                            Storage? ks = null, vs = null, st = null;
                            switch (cache)
                            {
                                case "float32":
                                    (keys, values) = (s.Up(keysF), s.Up(valuesF));
                                    break;
                                case "bfloat16":
                                    (keys, values) = (s.Up(keysB), s.Up(valuesB));
                                    break;
                                case "float32 per-row starts":
                                    (keys, values, st) = (s.Up(keysF), s.Up(valuesF), s.Up(starts));
                                    break;
                                case "bfloat16 per-row starts":
                                    (keys, values, st) = (s.Up(keysB), s.Up(valuesB), s.Up(starts));
                                    break;
                                default:
                                    (keys, values, ks, vs) = (s.Up(keysI), s.Up(valuesI), s.Up(keyScales), s.Up(valueScales));
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
                                        case "bfloat16":
                                            b.AttentionBFloat16Kernel(qs, keys, values, pos, y, heads, rowsPerHead, steps, capacity, dim, scale, false, variant);
                                            break;
                                        case "float32 per-row starts":
                                            b.AttentionRowsKernel(qs, keys, values, pos, y, st!, heads, 1, rowsPerHead, steps, capacity, dim, scale, variant);
                                            break;
                                        case "bfloat16 per-row starts":
                                            b.AttentionRowsBFloat16Kernel(qs, keys, values, pos, y, st!, heads, 1, rowsPerHead, steps, capacity, dim, scale, variant);
                                            break;
                                        default:
                                            b.AttentionInt8Kernel(qs, keys, values, ks!, vs!, pos, y, heads, rowsPerHead, steps, capacity, dim, scale, false, variant);
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
