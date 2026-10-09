// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Idrak;
using Idrak.Abstraction.Operations;
using Idrak.Abstraction.Testing;
using Idrak.Layers;

// Attention over one range of keys per query row (Tensor.AttentionSpans, Backend.AttentionSpans): every rule KeySpans
// builds against the composed path (the full scores, a mask, softmax), forward and gradients, on each device; the
// rules' ranges as Gemma 3 masks them; and the bidirectional multi-head attention layer, which now runs on it.
internal static partial class Tests
{
    private static readonly (string Name, Action<Device> Run)[] SpanAttentionGroup =
    [
        ("attention spans: bidirectional, causal, window, segment and image-block ranges (two tables, grouped heads, a soft-cap) match the composed scores, forward and gradients; empty ranges give zeros and no gradient", SpanAttentionMatchesComposed),
        ("attention spans: the rules' ranges are Gemma 3's masks (a key seen when not after the row or in the row's image block, within the window on sliding layers), and transformers' for the tiny model's image prompt", SpanRulesAreGemmaMasks),
        ("attention spans: bidirectional multi-head attention runs on the key-range kernel or the full scores as measured or forced, and matches the composed path, forward and gradients", BidirectionalAttentionLayer),
        ("attention spans: the fastest path (forced key ranges, forced composed, measured) gives AttentionSpans' result with every key and with a dense mask; the memory guard; IDRAK_ATTENTION_PATH", FastestAttention),
        ("attention spans: float32 and bfloat16 tensor cores (head sizes up to 128, rows past blocks of 64 and 128, every rule, grouped heads, tables, a soft-cap) match the CPU", SpanAttentionPrecisions),
        ("attention spans: the CUDA span modules (float32 and tensor-core, every padded head size) declare their kernels and parameters, and are judged by the shared memory a device reports", SpanModules),
        ("attention spans backward: the conformance kit's key-range cases (forward and gradient; random ranges, bidirectional, causal, windows, image blocks, grouped heads, tables, head sizes 1 to 256, a soft-cap) replay on the device call by call", SpanGradientCases),
        ("attention spans backward: float32 and bfloat16 (MixedPrecision): the gradient on the device matches the CPU (head sizes 4 to 256, rows and keys past several tiles, every rule, grouped heads, tables, a soft-cap), on the device's own kernels", SpanGradientPrecisions),
        ("attention spans backward: training through the fastest path (forced key ranges, forced composed, measured with the gradient) gives the same gradients; the key-range path runs AttentionSpansBackward", SpanTrainingPaths),
    ];

    // The composed path for query head h: softmax(cap(scale · q kᵀ) + mask) v with the mask from the table's ranges.
    private static Tensor ComposedSpans(Tensor q, Tensor k, Tensor v, KeySpans spans, int tables, float scale, AttentionVariant variant)
    {
        int heads = q.Shape[0], rows = q.Shape[1], kvHeads = k.Shape[0], keyRows = k.Shape[1];
        var outputs = new List<Tensor>();
        for (int h = 0; h < heads; h++)
        {
            int g = h / (heads / kvHeads), table = h / (heads / tables);
            var mask = new float[rows * keyRows];
            for (int i = 0; i < rows; i++)
            {
                for (int c = 0; c < keyRows; c++)
                {
                    mask[i * keyRows + c] = c >= spans.Starts[table * rows + i] && c < spans.Ends[table * rows + i] ? 0f : -1e9f;
                }
            }

            var scores = q.Narrow(0, h, 1).MatMul(k.Narrow(0, g, 1), transposeB: true) * scale;
            if (variant.Softcap > 0f)
            {
                scores = (scores * (1f / variant.Softcap)).Tanh() * variant.Softcap;
            }

            outputs.Add((scores + Tensor.From(mask, [rows, keyRows], q.Device)).Softmax().MatMul(v.Narrow(0, g, 1)));
        }

        return Tensor.Concat(outputs, 0);
    }

    private static void SpanAttentionMatchesComposed(Device device)
    {
        var random = new Random(71);
        int rows = 37, dim = 13;
        float scale = 1f / MathF.Sqrt(dim);
        (string Name, Func<KeySpans> Spans)[] rules =
        [
            ("bidirectional", () => KeySpans.Bidirectional(rows, rows)),
            ("causal", () => KeySpans.Causal(rows)),
            ("window", () => KeySpans.Causal(rows, 5)),
            ("segments", () => KeySpans.Segments([10, 1, 20, 6], window: 7)),
            ("image blocks", () => KeySpans.ImageBlocks(rows, [(3, 9), (20, 16)])),
            ("image blocks in a window", () => KeySpans.ImageBlocks(rows, [(3, 9), (20, 16)], window: 4)),
        ];
        foreach (var (name, make) in rules)
        {
            foreach (var variant in new[] { default(AttentionVariant), new AttentionVariant(0, 3f) })
            {
                int kvHeads = 2, heads = 6, tables = 2;
                var spans = KeySpans.Concat(make(), name == "bidirectional" ? KeySpans.Causal(rows) : make());
                var (starts, ends) = spans.ToTensors(device);
                var w = Tensor.From(RandomArray(random, heads * rows * dim), [heads, rows, dim], device);
                Tensor Leaf(int count, int[] shape) => Tensor.From(RandomArray(random, count), shape, device, requiresGrad: true);
                var (q, k, v) = (Leaf(heads * rows * dim, [heads, rows, dim]), Leaf(kvHeads * rows * dim, [kvHeads, rows, dim]), Leaf(kvHeads * rows * dim, [kvHeads, rows, dim]));
                var y = Tensor.AttentionSpans(q, k, v, starts, ends, scale, variant);
                (y * w).Sum().Backward();
                float[] gq = q.Grad!.ToArray(), gk = k.Grad!.ToArray(), gv = v.Grad!.ToArray();
                foreach (var t in new[] { q, k, v })
                {
                    t.ZeroGrad();
                }

                var expected = ComposedSpans(q, k, v, spans, tables, scale, variant);
                (expected * w).Sum().Backward();
                string what = $"{name}, cap {variant.Softcap} on {device}";
                AssertClose(expected.ToArray(), y.ToArray(), 2e-4f, what);
                AssertClose(q.Grad!.ToArray(), gq, 1e-3f, what + ": dq");
                AssertClose(k.Grad!.ToArray(), gk, 1e-3f, what + ": dkeys");
                AssertClose(v.Grad!.ToArray(), gv, 1e-3f, what + ": dvalues");
            }
        }

        // Empty ranges (rows 0 and 2) and ranges past the keys (row 1, clamped): zeros, and no gradient from those rows.
        var empty = new KeySpans([3, 0, 4, 0], [3, 9, 1, 2]);
        var (es, ee) = empty.ToTensors(device);
        var eq = Tensor.From(RandomArray(random, 4 * 3), [1, 4, 3], device, requiresGrad: true);
        var ek = Tensor.From(RandomArray(random, 5 * 3), [1, 5, 3], device, requiresGrad: true);
        var ey = Tensor.AttentionSpans(eq, ek, ek, es, ee, 0.5f);
        var values = ey.ToArray();
        Check(values[..3].All(x => x == 0f) && values[6..9].All(x => x == 0f) && values.All(float.IsFinite), "empty ranges give zeros");
        ey.Sum().Backward();
        var dq = eq.Grad!.ToArray();
        Check(dq[..3].All(x => x == 0f) && dq[6..9].All(x => x == 0f) && dq.All(float.IsFinite), "empty ranges give no gradient");
    }

    private static void SpanRulesAreGemmaMasks(Device device)
    {
        int rows = 23;
        (int Start, int Length)[] blocks = [(2, 6), (12, 4), (19, 4)];
        foreach (int window in new[] { 0, 1, 3, 8, 40 })
        {
            var spans = KeySpans.ImageBlocks(rows, blocks, window);
            for (int q = 0; q < rows; q++)
            {
                int block = Array.FindIndex(blocks, b => q >= b.Start && q < b.Start + b.Length);
                for (int k = 0; k < rows; k++)
                {
                    bool sameBlock = block >= 0 && k >= blocks[block].Start && k < blocks[block].Start + blocks[block].Length;
                    bool seen = (k <= q || sameBlock) && (window == 0 || k > q - window);
                    bool ranged = k >= spans.Starts[q] && k < spans.Ends[q];
                    Check(seen == ranged, $"window {window}: row {q}, key {k} is {(seen ? "" : "not ")}seen by Gemma 3's mask");
                }
            }
        }

        // The tiny model's prompt with an image (plan 11, phase 0): transformers' masks as each row's first and last key,
        // for the global and the sliding-window layers, against the rule built from the prompt's image tokens.
        string manifest = Path.Combine(RepositoryRoot(), "tests", "Idrak.Tests", "data", "vlm", "manifest.json");
        var facts = System.Text.Json.Nodes.JsonNode.Parse(File.ReadAllText(manifest))!["facts"]!;
        var prompt = facts["image_prompt"]!;
        int[] types = [.. prompt["token_type_ids"]!.AsArray().Select(n => (int)n!)];
        var runs = new List<(int Start, int Length)>();
        for (int i = 0; i < types.Length; i++)
        {
            if (types[i] == 1 && (i == 0 || types[i - 1] != 1))
            {
                int end = i;
                while (end < types.Length && types[end] == 1)
                {
                    end++;
                }

                runs.Add((i, end - i));
            }
        }

        int sliding = (int)facts["model_facts"]!["sliding_window"]!;
        foreach (var (layer, window) in new[] { ("full_attention", 0), ("sliding_attention", sliding) })
        {
            var expected = prompt["mask_rows_first_last_key"]![layer]!.AsArray();
            var spans = KeySpans.ImageBlocks(types.Length, runs, window);
            for (int q = 0; q < types.Length; q++)
            {
                int first = (int)expected[q]![0]!, last = (int)expected[q]![1]!;
                Check(spans.Starts[q] == first && spans.Ends[q] == last + 1,
                    $"{layer}: row {q} sees [{spans.Starts[q]}, {spans.Ends[q]}), transformers [{first}, {last}]");
            }
        }
    }

    private static void BidirectionalAttentionLayer(Device device)
    {
        using var attention = new MultiHeadAttention(16, 4, device: device, random: new Random(72));
        var x = Tensor.From(RandomArray(new Random(73), 2 * 9 * 16), [2, 9, 16], device, requiresGrad: true);
        var y = attention.Forward(x);

        // The same layer composed by hand from its weights: the full scores, softmax, values.
        var weights = attention.Parameters().ToArray();                          // qkv weight, bias, output weight, bias
        var qkv = x.MatMul(weights[0]) + weights[1];
        Tensor Heads(int part) => qkv.Narrow(2, part * 16, 16).Reshape(2, 9, 4, 4).Permute(0, 2, 1, 3).Reshape(8, 9, 4);
        var context = (Heads(0).MatMul(Heads(1), transposeB: true) * 0.5f).Softmax().MatMul(Heads(2)).Reshape(2, 4, 9, 4).Permute(0, 2, 1, 3).Reshape(2, 9, 16);
        var expected = context.MatMul(weights[2]) + weights[3];
        AssertClose(expected.ToArray(), y.ToArray(), 2e-4f, $"bidirectional attention on {device}");

        y.Square().Sum().Backward();
        var grads = attention.Parameters().Select(p => p.Grad!.ToArray()).Append(x.Grad!.ToArray()).ToArray();
        foreach (var p in attention.Parameters())
        {
            p.ZeroGrad();
        }

        x.ZeroGrad();
        expected.Square().Sum().Backward();
        var reference = attention.Parameters().Select(p => p.Grad!.ToArray()).Append(x.Grad!.ToArray()).ToArray();
        for (int i = 0; i < grads.Length; i++)
        {
            AssertClose(reference[i], grads[i], 1e-3f, $"bidirectional attention gradient {i} on {device}");
        }

        // The layer takes Tensor.AttentionFastest: with the key-range path forced, no [T, T] scores (the key-range kernel,
        // never a softmax over scores); with the composed one forced (where the device reports the memory for the
        // scores), the full scores; measured, either; the same output each way.
        var previous = AttentionPaths.Forced;
        try
        {
            var outputs = new List<float[]>();
            foreach (var path in new[] { AttentionPath.Spans, AttentionPath.Composed, AttentionPath.Measured })
            {
                AttentionPaths.Forced = path;
                using (Autograd.NoGrad())
                using (var trace = Kernels.Trace(device.Backend))
                {
                    outputs.Add(attention.Forward(x).ToArray());
                    if (path == AttentionPath.Spans)
                    {
                        Check(trace.Calls(Ops.AttentionSpans) == 1 && trace.Calls(Ops.ScaleMaskSoftmax) == 0 && trace.Calls(Ops.Softmax) == 0,
                            "the layer runs AttentionSpans, not a softmax over full scores");
                        Check(device.Backend.Kind is not ("cpu" or "vulkan" or "cuda") || !trace.HostCallsByOperation.ContainsKey("AttentionSpans"),
                            "the CPU, Vulkan and CUDA run AttentionSpans on their own kernels");
                    }
                    else if (path == AttentionPath.Composed && device.Backend.AvailableMemory() is not null)
                    {
                        Check(trace.Calls(Ops.AttentionSpans) == 0 && trace.Calls(Ops.ScaleMaskSoftmax) == 1, "forced, the layer composes the full scores");
                    }
                    else
                    {
                        Check(trace.Calls(Ops.AttentionSpans) + trace.Calls(Ops.ScaleMaskSoftmax) >= 1, "measured, the layer takes a path");
                    }
                }
            }

            AssertClose(outputs[0], outputs[1], 1e-4f, $"the layer composed and on key ranges on {device}");
            AssertClose(outputs[0], outputs[2], 1e-4f, $"the layer on its measured path on {device}");
        }
        finally
        {
            AttentionPaths.Forced = previous;
        }
    }

    // Tensor.AttentionFastest: each path (forced) gives AttentionSpans' result, for every key (no mask) and for a rule
    // with its dense mask; the memory guard and the override's parsing.
    private static void FastestAttention(Device device)
    {
        var random = new Random(74);
        const int Heads = 4, Rows = 70, Dim = 16;
        float scale = 1f / MathF.Sqrt(Dim);
        using var q = Tensor.From(RandomArray(random, Heads * Rows * Dim), [Heads, Rows, Dim], device);
        using var k = Tensor.From(RandomArray(random, Heads * Rows * Dim), [Heads, Rows, Dim], device);
        using var v = Tensor.From(RandomArray(random, Heads * Rows * Dim), [Heads, Rows, Dim], device);
        var image = KeySpans.ImageBlocks(Rows, [(5, 20), (40, 9)], window: 30);
        var dense = new float[Rows * Rows];
        for (int i = 0; i < Rows; i++)
        {
            for (int c = 0; c < Rows; c++)
            {
                dense[i * Rows + c] = c >= image.Starts[i] && c < image.Ends[i] ? 0f : -1e9f;
            }
        }

        using var mask = Tensor.From(dense, [Rows, Rows], device);
        var previous = AttentionPaths.Forced;
        try
        {
            foreach (var (name, spans, everyKey, withMask) in new[] { ("every key", KeySpans.Bidirectional(Rows, Rows), true, (Tensor?)null), ("image blocks", image, false, mask) })
            {
                var (starts, ends) = spans.ToTensors(device);
                using var expected = Tensor.AttentionSpans(q, k, v, starts, ends, scale);
                foreach (var path in new[] { AttentionPath.Spans, AttentionPath.Composed, AttentionPath.Measured })
                {
                    AttentionPaths.Forced = path;
                    using var trace = Kernels.Trace(device.Backend);
                    using var y = Tensor.AttentionFastest(q, k, v, starts, ends, scale, everyKey, withMask);
                    AssertClose(expected.ToArray(), y.ToArray(), 1e-4f, $"{name}, {path} on {device}");
                    if (path == AttentionPath.Composed && device.Backend.AvailableMemory() is not null)
                    {
                        Check(trace.Calls(Ops.ScaleMaskSoftmax) == 1 && trace.Calls(Ops.AttentionSpans) == 0, $"{name}: forced composed composes");
                    }
                }

                // Without a mask or every key, only the key ranges can run.
                AttentionPaths.Forced = AttentionPath.Composed;
                using (var trace = Kernels.Trace(device.Backend))
                using (var y = Tensor.AttentionFastest(q, k, v, starts, ends, scale, everyKey: false))
                {
                    Check(trace.Calls(Ops.AttentionSpans) == 1 && trace.Calls(Ops.ScaleMaskSoftmax) == 0, $"{name}: without a mask, the key ranges");
                }
            }
        }
        finally
        {
            AttentionPaths.Forced = previous;
        }

        // The memory guard: the scores (and weights; four times the scores with the gradient) within half the memory the
        // device reports available, never when it reports none.
        Check(AttentionPaths.ComposedBytes(16, 4096, 4096, recording: false) == 2L << 30, "16 heads of 4,096² scores and weights take 2 GiB");
        Check(AttentionPaths.ComposedBytes(16, 4096, 4096, recording: true) == 4L << 30, "with the gradient, 4 GiB");
        Check(!AttentionPaths.ComposedFits(null, 1), "no memory reported: never composed");
        Check(AttentionPaths.ComposedFits(1000, 500) && !AttentionPaths.ComposedFits(1000, 501), "composed within half the available memory");
        Check(AttentionPaths.FromEnvironment("spans") == AttentionPath.Spans && AttentionPaths.FromEnvironment(" Composed ") == AttentionPath.Composed
              && AttentionPaths.FromEnvironment(null) == AttentionPath.Measured && AttentionPaths.FromEnvironment("other") == AttentionPath.Measured,
            "IDRAK_ATTENTION_PATH: spans, composed, else measured");
    }

    // AttentionSpans in float32 and under MixedPrecision (bfloat16 tensor cores where the device has them: within
    // bfloat16's error of the CPU's float32, as the other tensor-core attention; float32 everywhere else): every kernel's
    // head sizes (multiples of 4 up to 128 and an odd one), rows past one block of 64 and of 128, key ranges of every
    // rule and empty ones, grouped heads, a table per head, a soft-cap.
    private static void SpanAttentionPrecisions(Device device)
    {
        bool tensorCores = MixedPrecision.TensorCoresUnavailable(device) is null;
        var random = new Random(75);
        static double RelativeError(float[] expected, float[] actual) =>
            Math.Sqrt(expected.Zip(actual).Sum(p => (double)(p.First - p.Second) * (p.First - p.Second)) / Math.Max(1e-12, expected.Sum(v => (double)v * v)));
        foreach (var precision in new[] { MatMulPrecision.Float32, MatMulPrecision.BFloat16 })
        {
            foreach (var (dim, rows, keyRows, rule, kvHeads, tables, cap) in new[]
                     {
                         (72, 150, 150, "bidirectional", 2, 1, 0f), (64, 130, 200, "bidirectional", 1, 1, 0f), (128, 129, 129, "causal", 2, 4, 0f),
                         (80, 100, 100, "image blocks", 1, 1, 0f), (4, 70, 90, "random", 2, 4, 0f), (100, 65, 65, "window", 4, 1, 0f),
                         (72, 90, 90, "random", 1, 1, 2.5f), (13, 40, 40, "bidirectional", 1, 1, 0f),
                     })
            {
                const int Heads = 4;
                float scale = 1f / MathF.Sqrt(dim);
                KeySpans Table() => rule switch
                {
                    "bidirectional" => KeySpans.Bidirectional(rows, keyRows),
                    "causal" => KeySpans.Causal(rows),
                    "window" => KeySpans.Causal(rows, 17),
                    "image blocks" => KeySpans.ImageBlocks(rows, [(3, 40), (60, 33)], window: 20),
                    _ => new KeySpans([.. Enumerable.Range(0, rows).Select(_ => random.Next(0, keyRows + 2))],
                        [.. Enumerable.Range(0, rows).Select(_ => random.Next(0, keyRows + 3))]),
                };
                var spans = KeySpans.Concat([.. Enumerable.Range(0, tables).Select(_ => Table())]);
                float[] qv = RandomArray(random, Heads * rows * dim), kv = RandomArray(random, kvHeads * keyRows * dim), vv = RandomArray(random, kvHeads * keyRows * dim);
                (float[] Y, float[] Lse) Run(Device on)
                {
                    using var q = Tensor.From(qv, [Heads, rows, dim], on);
                    using var k = Tensor.From(kv, [kvHeads, keyRows, dim], on);
                    using var v = Tensor.From(vv, [kvHeads, keyRows, dim], on);
                    var (starts, ends) = spans.ToTensors(on);
                    using var y = Tensor.Zeros([Heads * rows * dim], on);
                    using var lse = Tensor.Zeros([Heads * rows], on);
                    using (MixedPrecision.Use(precision))
                    {
                        on.Backend.AttentionSpans(q.Storage, k.Storage, v.Storage, starts.Storage, ends.Storage, y.Storage, lse.Storage, Heads, kvHeads,
                            Heads / tables, rows, keyRows, dim, scale, new AttentionVariant(0, cap));
                    }

                    return (y.ToArray(), lse.ToArray());
                }

                var expected = Run(Device.Cpu);
                var got = Run(device);
                string what = $"{precision}, dim {dim}, {rows} rows, {keyRows} keys, {rule}, {kvHeads} key/value heads, {tables} tables, cap {cap} on {device}";
                double tolerance = precision != MatMulPrecision.Float32 && tensorCores && cap == 0f && dim % 4 == 0 ? 1.5e-2 : 1e-5;
                Check(RelativeError(expected.Y, got.Y) < tolerance, $"{what}: output error {RelativeError(expected.Y, got.Y):G3}");
                for (int i = 0; i < expected.Lse.Length; i++)
                {
                    bool same = float.IsNegativeInfinity(expected.Lse[i])
                        ? float.IsNegativeInfinity(got.Lse[i])
                        : Math.Abs(expected.Lse[i] - got.Lse[i]) <= 50 * tolerance * Math.Max(1, Math.Abs(expected.Lse[i]));
                    Check(same, $"{what}: log-sum-exp {i}: {expected.Lse[i]} against {got.Lse[i]}");
                }
            }
        }
    }

    // Runs without a GPU (the PTX is generated on the host): each padded head size's modules, their kernels' parameters
    // (CudaBackend.Launch checks every launch against them), and the shared memory they need against what a device
    // reports: the float32 kernel's tiles grow with the head size, past 48 KiB from 104 on (devices that cannot opt in
    // to more keep the first kernel there), the tensor-core tiles stay within 48 KiB of static memory.
    private static void SpanModules(Device device)
    {
        if (device.Type != DeviceType.Cpu)
        {
            return;                                                              // nothing device-specific: once
        }

        var documented = Idrak.Gpu.Cuda.CudaDeviceLimits.Documented(8, 6, 30);
        var old = documented with { SharedPerBlockOptin = 48 << 10 };
        for (int d = 8; d <= Idrak.Gpu.Cuda.PtxKernels.FlashMaxDim; d += 8)
        {
            var (_, kernels, source) = Idrak.Gpu.Cuda.PtxKernels.SpanFloatModule(d);
            var counts = Idrak.Gpu.Cuda.PtxKernels.ParameterCountsOf(source);
            Check(kernels.SequenceEqual([Idrak.Gpu.Cuda.PtxKernels.SpanFloatName(d)]) && counts[kernels[0]] == 14, $"f32 d{d}: one kernel of 14 parameters");
            // The PTX is written from raw string literals, so its line endings are the checkout's (CRLF on Windows with autocrlf).
            Check(source.ReplaceLineEndings("\n").StartsWith(".version 6.0\n.target sm_50", StringComparison.Ordinal), $"f32 d{d}: PTX 6.0 for sm_50, as the main module");
            int shared = Idrak.Gpu.Cuda.PtxKernels.SpanFloatShared(d);
            Check(shared <= documented.SharedPerBlockOptin && (shared <= old.SharedPerBlockOptin) == (d < 104), $"f32 d{d}: {shared} bytes of shared memory");
            if (d % 16 == 0)
            {
                var tensor = Idrak.Gpu.Cuda.PtxKernels.SpanTensorModule(d);
                var tensorCounts = Idrak.Gpu.Cuda.PtxKernels.ParameterCountsOf(tensor.Source);
                Check(tensor.Kernels.SequenceEqual(Idrak.Gpu.Cuda.PtxKernels.SpanTensorWarps.Select(w => Idrak.Gpu.Cuda.PtxKernels.SpanTensorName(d, w)))
                      && tensor.Kernels.All(k => tensorCounts[k] == 13), $"tc d{d}: a kernel per warp count, 13 parameters each");
                Check(Idrak.Gpu.Cuda.PtxKernels.Fits(tensor.Source, old) is null, $"tc d{d}: within 48 KiB");
            }
        }

        Check(Idrak.Gpu.Cuda.PtxKernels.SpanFloatDim(72) == 72 && Idrak.Gpu.Cuda.PtxKernels.SpanFloatDim(68) == 72 && Idrak.Gpu.Cuda.PtxKernels.SpanFloatDim(13) == 0
              && Idrak.Gpu.Cuda.PtxKernels.SpanFloatDim(132) == 0, "float32 head sizes: multiples of 4 up to 128, padded to 8");
        Check(Idrak.Gpu.Cuda.PtxKernels.SpanTensorDim(72) == 80 && Idrak.Gpu.Cuda.PtxKernels.SpanTensorDim(64) == 64 && Idrak.Gpu.Cuda.PtxKernels.SpanTensorDim(6) == 0,
            "tensor-core head sizes: multiples of 4 up to 128, padded to 16");
    }

    // The kit's two key-range cases on this device: every call the CPU made (AttentionSpans, AttentionSpansBackward, and
    // the composed path's products and softmax) made again here and compared with what the CPU wrote.
    private static void SpanGradientCases(Device device)
    {
        var report = Conformance.Check(device, new DeviceCheckOptions { Filter = name => name.StartsWith("attention over key ranges", StringComparison.Ordinal) });
        report.ThrowIfFailed();
        foreach (string operation in new[] { "AttentionSpans", "AttentionSpansBackward" })
        {
            var entry = report.Entries.Single(e => e.Name == operation);
            Check(entry.Status == CheckStatus.Passed, $"{operation} on {device}: {entry}");
        }
    }

    // AttentionSpansBackward on the device against the CPU's, given the same forward results (the CPU's output and
    // log-sum-exp) in float32 and under MixedPrecision (the gradient's kernels sum in float32 either way), at head sizes
    // past the kernels' tiles; then a forward and backward pass through tensors on each, within bfloat16's error where
    // the device's forward runs on bfloat16 tensor cores. The device's own kernels take the gradient where it has them:
    // CUDA to head size 128, Vulkan to 256 when it binds the kernels' storages.
    private static void SpanGradientPrecisions(Device device)
    {
        bool tensorCores = MixedPrecision.TensorCoresUnavailable(device) is null;
        var random = new Random(76);
        static double RelativeError(float[] expected, float[] actual) =>
            Math.Sqrt(expected.Zip(actual).Sum(p => (double)(p.First - p.Second) * (p.First - p.Second)) / Math.Max(1e-12, expected.Sum(v => (double)v * v)));
        foreach (var precision in new[] { MatMulPrecision.Float32, MatMulPrecision.BFloat16 })
        {
            foreach (var (dim, rows, keyRows, rule, kvHeads, tables, cap) in new[]
                     {
                         (72, 150, 150, "bidirectional", 2, 1, 0f), (64, 130, 200, "bidirectional", 1, 1, 0f), (128, 129, 129, "causal", 2, 4, 0f),
                         (80, 100, 100, "image blocks", 1, 1, 0f), (4, 70, 90, "random", 2, 4, 0f), (100, 65, 65, "window", 4, 1, 0f),
                         (72, 90, 90, "random", 1, 1, 2.5f), (13, 40, 40, "bidirectional", 1, 1, 0f), (256, 70, 70, "image blocks", 2, 2, 0f),
                         (200, 45, 60, "random", 1, 1, 0f),
                     })
            {
                const int Heads = 4;
                float scale = 1f / MathF.Sqrt(dim);
                var variant = new AttentionVariant(0, cap);
                KeySpans Table() => rule switch
                {
                    "bidirectional" => KeySpans.Bidirectional(rows, keyRows),
                    "causal" => KeySpans.Causal(rows),
                    "window" => KeySpans.Causal(rows, 17),
                    "image blocks" => KeySpans.ImageBlocks(rows, [(3, Math.Min(40, rows / 2)), (rows / 2 + 5, rows / 3)], window: 20),
                    _ => new KeySpans([.. Enumerable.Range(0, rows).Select(_ => random.Next(0, keyRows + 2))],
                        [.. Enumerable.Range(0, rows).Select(_ => random.Next(0, keyRows + 3))]),
                };
                var spans = KeySpans.Concat([.. Enumerable.Range(0, tables).Select(_ => Table())]);
                float[] qv = RandomArray(random, Heads * rows * dim), kv = RandomArray(random, kvHeads * keyRows * dim), vv = RandomArray(random, kvHeads * keyRows * dim);
                float[] dyv = RandomArray(random, Heads * rows * dim);
                string what = $"{precision}, dim {dim}, {rows} rows, {keyRows} keys, {rule}, {kvHeads} key/value heads, {tables} tables, cap {cap} on {device}";

                // The CPU's forward results, given to both.
                float[] yv, lsev;
                using (var q = Tensor.From(qv, [Heads, rows, dim], Device.Cpu))
                using (var k = Tensor.From(kv, [kvHeads, keyRows, dim], Device.Cpu))
                using (var v = Tensor.From(vv, [kvHeads, keyRows, dim], Device.Cpu))
                using (var y = Tensor.Zeros([Heads * rows * dim], Device.Cpu))
                using (var lse = Tensor.Zeros([Heads * rows], Device.Cpu))
                {
                    var (starts, ends) = spans.ToTensors(Device.Cpu);
                    Device.Cpu.Backend.AttentionSpans(q.Storage, k.Storage, v.Storage, starts.Storage, ends.Storage, y.Storage, lse.Storage, Heads, kvHeads,
                        Heads / tables, rows, keyRows, dim, scale, variant);
                    (yv, lsev) = (y.ToArray(), lse.ToArray());
                }

                (float[] Dq, float[] Dk, float[] Dv, IReadOnlyDictionary<string, long> HostCalls) Gradient(Device on)
                {
                    using var q = Tensor.From(qv, [Heads, rows, dim], on);
                    using var k = Tensor.From(kv, [kvHeads, keyRows, dim], on);
                    using var v = Tensor.From(vv, [kvHeads, keyRows, dim], on);
                    using var y = Tensor.From(yv, [Heads * rows * dim], on);
                    using var lse = Tensor.From(lsev, [Heads * rows], on);
                    using var dy = Tensor.From(dyv, [Heads * rows * dim], on);
                    using var dq = Tensor.Zeros([Heads * rows * dim], on);
                    using var dk = Tensor.Zeros([kvHeads * keyRows * dim], on);
                    using var dv = Tensor.Zeros([kvHeads * keyRows * dim], on);
                    var (starts, ends) = spans.ToTensors(on);
                    using (var trace = Kernels.Trace(on.Backend))
                    using (MixedPrecision.Use(precision))
                    {
                        on.Backend.AttentionSpansBackward(q.Storage, k.Storage, v.Storage, starts.Storage, ends.Storage, y.Storage, lse.Storage, dy.Storage,
                            dq.Storage, dk.Storage, dv.Storage, Heads, kvHeads, Heads / tables, rows, keyRows, dim, scale, variant);
                        on.Synchronize();
                        return (dq.ToArray(), dk.ToArray(), dv.ToArray(), trace.HostCallsByOperation);
                    }
                }

                var expected = Gradient(Device.Cpu);
                var got = Gradient(device);
                Check(RelativeError(expected.Dq, got.Dq) < 1e-4, $"{what}: dq error {RelativeError(expected.Dq, got.Dq):G3}");
                Check(RelativeError(expected.Dk, got.Dk) < 1e-4, $"{what}: dkeys error {RelativeError(expected.Dk, got.Dk):G3}");
                Check(RelativeError(expected.Dv, got.Dv) < 1e-4, $"{what}: dvalues error {RelativeError(expected.Dv, got.Dv):G3}");
                bool ownKernels = device.Backend.Kind switch
                {
                    "cuda" => dim <= Idrak.Gpu.Cuda.PtxKernels.FlashMaxDim,
                    "vulkan" => device.Backend is Idrak.Gpu.Vulkan.VulkanBackend vulkan && vulkan.SpanGradientOnDevice,
                    _ => device.Type == DeviceType.Cpu,
                };
                Check(!ownKernels || !got.HostCalls.ContainsKey("AttentionSpansBackward"), $"{what}: the gradient ran on the device's own kernels");

                // Forward and backward through tensors, in the precision.
                (float[] Y, float[] Dq, float[] Dk, float[] Dv) Pass(Device on)
                {
                    using var q = Tensor.From(qv, [Heads, rows, dim], on, requiresGrad: true);
                    using var k = Tensor.From(kv, [kvHeads, keyRows, dim], on, requiresGrad: true);
                    using var v = Tensor.From(vv, [kvHeads, keyRows, dim], on, requiresGrad: true);
                    using var w = Tensor.From(dyv, [Heads, rows, dim], on);
                    var (starts, ends) = spans.ToTensors(on);
                    using (MixedPrecision.Use(on.Type == DeviceType.Cpu ? MatMulPrecision.Float32 : precision))
                    {
                        var y = Tensor.AttentionSpans(q, k, v, starts, ends, scale, variant);
                        (y * w).Sum().Backward();
                        return (y.ToArray(), q.Grad!.ToArray(), k.Grad!.ToArray(), v.Grad!.ToArray());
                    }
                }

                var reference = Pass(Device.Cpu);
                var pass = Pass(device);
                // bfloat16's error where the forward runs on tensor cores (the gradient then starts from its output and log-sum-exp).
                bool reduced = precision != MatMulPrecision.Float32 && tensorCores && cap == 0f && dim % 4 == 0 && dim <= 128;
                double tolerance = reduced ? 5e-2 : 1e-4;
                Check(RelativeError(reference.Y, pass.Y) < (reduced ? 1.5e-2 : 1e-4), $"{what}, through tensors: output error {RelativeError(reference.Y, pass.Y):G3}");
                Check(RelativeError(reference.Dq, pass.Dq) < tolerance, $"{what}, through tensors: dq error {RelativeError(reference.Dq, pass.Dq):G3}");
                Check(RelativeError(reference.Dk, pass.Dk) < tolerance, $"{what}, through tensors: dkeys error {RelativeError(reference.Dk, pass.Dk):G3}");
                Check(RelativeError(reference.Dv, pass.Dv) < tolerance, $"{what}, through tensors: dvalues error {RelativeError(reference.Dv, pass.Dv):G3}");
            }
        }
    }

    // Tensor.AttentionFastest with the gradient recorded: each path (forced key ranges, forced composed where the device
    // reports the memory, measured with the gradient) gives the same gradients; the key-range path records
    // AttentionSpansBackward and, on CUDA and Vulkan, runs it on the device.
    private static void SpanTrainingPaths(Device device)
    {
        var random = new Random(77);
        const int Heads = 4, Rows = 70, Dim = 16;
        float scale = 1f / MathF.Sqrt(Dim);
        float[] qv = RandomArray(random, Heads * Rows * Dim), kv = RandomArray(random, Heads * Rows * Dim), vv = RandomArray(random, Heads * Rows * Dim);
        float[] wv = RandomArray(random, Heads * Rows * Dim);
        var (starts, ends) = KeySpans.Bidirectional(Rows, Rows).ToTensors(device);
        var previous = AttentionPaths.Forced;
        try
        {
            var gradients = new List<float[][]>();
            foreach (var path in new[] { AttentionPath.Spans, AttentionPath.Composed, AttentionPath.Measured })
            {
                AttentionPaths.Forced = path;
                using var q = Tensor.From(qv, [Heads, Rows, Dim], device, requiresGrad: true);
                using var k = Tensor.From(kv, [Heads, Rows, Dim], device, requiresGrad: true);
                using var v = Tensor.From(vv, [Heads, Rows, Dim], device, requiresGrad: true);
                using var w = Tensor.From(wv, [Heads, Rows, Dim], device);
                using var trace = Kernels.Trace(device.Backend);
                (Tensor.AttentionFastest(q, k, v, starts, ends, scale, everyKey: true) * w).Sum().Backward();
                gradients.Add([q.Grad!.ToArray(), k.Grad!.ToArray(), v.Grad!.ToArray()]);
                if (path == AttentionPath.Spans)
                {
                    Check(trace.Calls(Ops.AttentionSpansBackward) == 1 && trace.Calls(Ops.SoftmaxBackward) == 0, "the key-range path's gradient is AttentionSpansBackward");
                    Check(device.Backend.Kind is not ("cuda" or "vulkan") || !trace.HostCallsByOperation.ContainsKey("AttentionSpansBackward"),
                        "CUDA and Vulkan run AttentionSpansBackward on their own kernels");
                }
                else if (path == AttentionPath.Composed && device.Backend.AvailableMemory() is not null)
                {
                    Check(trace.Calls(Ops.AttentionSpansBackward) == 0, "forced composed, no key-range gradient");
                }
            }

            for (int i = 1; i < gradients.Count; i++)
            {
                for (int t = 0; t < 3; t++)
                {
                    AssertClose(gradients[0][t], gradients[i][t], 1e-3f, $"gradient {t} of path {i} against the key ranges' on {device}");
                }
            }
        }
        finally
        {
            AttentionPaths.Forced = previous;
        }
    }
}
