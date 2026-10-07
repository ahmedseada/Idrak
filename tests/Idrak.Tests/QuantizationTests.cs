// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Idrak;
using Idrak.Data;
using Idrak.Generation;
using Idrak.Layers;
using Idrak.Models.Abstractions;
using Idrak.Onnx;
using Idrak.Optimizers;
using Idrak.Training;

// Int8 weight-only quantization and half-precision weight files.
internal static partial class Tests
{
    private static readonly (string Name, Action<Device> Run)[] Quantization =
    [
        ("int8: quantize/dequantize per column; direct and dequantized products match; input gradient", Int8Kernels),
        ("int8: quantized GPT: cached decoding matches the full pass; small logit change; ONNX export dequantizes", Int8Gpt),
        ("int8: save/load (a float model loads a quantized file), move between devices, dequantize, memory", Int8SaveLoad),
        ("int8: QLoRA — adapters train on top of frozen int8 weights", Int8Lora),
        ("bf16: rounding, direct and expanded products, gradient, model conversion, save/load, decoder build", BFloat16Weights),
        ("int4: group scales, direct and expanded products, gradient, model conversion, save/load", Int4Weights),
        ("packed projections sharing an input (int8, int4, bf16; biases; splits) and gated down projections match separate layers", PackedMany),
        ("weights: Float16 and BFloat16 files are half the size and round as expected", HalfPrecisionFiles),
        ("matmul: few rows (column-parallel path) match the reference, with beta and transposes", FewRowMatMul),
        ("int8 KV cache: write, scores and context kernels match a dequantized cache", Int8CacheKernels),
        ("int8 KV cache: decoding stays close to float32 in less memory; graph replay matches direct steps", Int8CacheDecoding),
    ];

    private static void BFloat16Weights(Device device)
    {
        // Rounding: to nearest, ties to even; exactly representable values stay exact.
        float Bf(float v) => BitConverter.Int32BitsToSingle(BFloat16Weight.Round(v) << 16);
        Check(Bf(1f) == 1f && Bf(-2.5f) == -2.5f && Bf(0f) == 0f, "exact values stay exact");
        Check(Bf(1f + 1f / 256f) == 1f && Bf(1f + 3f / 256f) == 1f + 4f / 256f, "ties round to even");
        Check(float.IsNaN(Bf(float.NaN)), "NaN stays NaN");

        var r = new Random(71);
        foreach (var (k, n) in new[] { (37, 23), (1030, 301), (700, 2050) })
        {
            var values = Enumerable.Range(0, k * n).Select(_ => (float)(r.NextDouble() * 2 - 1)).ToArray();
            using var h = BFloat16Weight.FromValues(values, k, n, device);
            using var expanded = h.Dequantize();
            var e = expanded.ToArray();
            Check(e.Select((v, i) => v == Bf(values[i])).All(x => x), $"[{k}, {n}]: expansion equals the rounded values");
            foreach (int m in new[] { 1, 3, 8, 20 })                       // ≤ 8 rows read bf16 directly; 20 expand first
            {
                var input = Enumerable.Range(0, m * k).Select(_ => (float)(r.NextDouble() * 2 - 1)).ToArray();
                using var x = Tensor.From(input, [m, k], device, requiresGrad: true);
                using var reference = Tensor.From(input, [m, k], device, requiresGrad: true);
                var y = x.MatMulBFloat16(h);
                var want = reference.MatMul(expanded);
                AssertClose(want.ToArray(), y.ToArray(), 2e-3f, $"bf16 product [{m}x{k}] × [{k}x{n}]");
                if (k == 37)
                {
                    y.Sum().Backward();
                    want.Sum().Backward();
                    AssertClose(reference.Grad!.ToArray(), x.Grad!.ToArray(), 1e-4f, $"input gradient, {m} rows");
                }
            }
        }

        // A whole model: bf16 predictions close to float32; save/load keeps bf16; back to float32.
        using var model = TinyGpt(device);
        using var ids = Tensor.From([1f, 4f, 2f, 7f, 3f], [1, 5], device);
        var original = model.Predict(ids).ToArray();
        Check(model.ToBFloat16() > 0 && model.Descendants().OfType<Linear>().All(l => l.BFloat16 is not null), "every Linear is bf16");
        var half = model.Predict(ids).ToArray();
        AssertClose(original, half, 0.02f * original.Max(MathF.Abs), "bf16 model predictions");
        string path = Path.GetTempFileName();
        try
        {
            model.Save(path);
            using var reloaded = TinyGpt(device);
            reloaded.Load(path);
            Check(reloaded.Descendants().OfType<Linear>().All(l => l.BFloat16 is not null), "a float model loads a bf16 file as bf16");
            Check(reloaded.Predict(ids).ToArray().SequenceEqual(half), "reloaded bf16 predictions are identical");
        }
        finally
        {
            File.Delete(path);
        }

        // (Not bit-identical on CUDA: few-row bf16 and float products add in different orders.)
        Check(model.ToFloat32() > 0 && model.Descendants().OfType<Linear>().All(l => !l.Packed), "back to float32");
        AssertClose(half, model.Predict(ids).ToArray(), 1e-4f, "float32 keeps the rounded weights");
    }

    private static void Int4Weights(Device device)
    {
        var r = new Random(73);
        foreach (var (k, n) in new[] { (37, 23), (1030, 301), (700, 2050), (4096, 264) })
        {
            var values = Enumerable.Range(0, k * n).Select(_ => (float)(r.NextDouble() * 2 - 1)).ToArray();
            using var q = Int4Weight.Quantize(values, k, n, device);

            // The plain rule: per group of 32 rows and column, d = (the value of largest magnitude) / -8, q = clamp(round(v / d), -8, 7).
            var rule = new float[k * n];
            for (int g = 0; g < (k + 31) / 32; g++)
            {
                for (int j = 0; j < n; j++)
                {
                    float extreme = 0f;
                    for (int row = g * 32; row < Math.Min(k, g * 32 + 32); row++)
                    {
                        extreme = MathF.Abs(values[row * n + j]) > MathF.Abs(extreme) ? values[row * n + j] : extreme;
                    }

                    float d = extreme / -8f;
                    for (int row = g * 32; row < Math.Min(k, g * 32 + 32); row++)
                    {
                        rule[row * n + j] = d == 0f ? 0f : Math.Clamp(MathF.Round(values[row * n + j] / d), -8f, 7f) * d;
                    }
                }
            }

            using var expanded = q.Dequantize();
            var e = expanded.ToArray();
            // The searched scales never do worse than the plain rule (extreme / -8), group by group.
            double searched = values.Select((v, i) => (double)(v - e[i]) * (v - e[i])).Sum();
            double plain = values.Select((v, i) => (double)(v - rule[i]) * (v - rule[i])).Sum();
            Check(searched <= plain * (1 + 1e-6), $"[{k}, {n}]: squared error {searched:G4} vs plain rule {plain:G4}");
            float error = values.Select((v, i) => MathF.Abs(v - e[i])).Max();
            Check(error <= 0.25f, $"[{k}, {n}]: largest error {error}");
            foreach (int m in new[] { 1, 3, 8, 20 })                       // ≤ 8 rows read nibbles directly; 20 expand first
            {
                var input = Enumerable.Range(0, m * k).Select(_ => (float)(r.NextDouble() * 2 - 1)).ToArray();
                using var x = Tensor.From(input, [m, k], device, requiresGrad: true);
                using var reference = Tensor.From(input, [m, k], device, requiresGrad: true);
                var y = x.MatMulInt4(q);
                var want = reference.MatMul(expanded);
                AssertClose(want.ToArray(), y.ToArray(), 2e-3f * MathF.Sqrt(k), $"int4 product [{m}x{k}] × [{k}x{n}]");
                if (k == 37)
                {
                    y.Sum().Backward();
                    want.Sum().Backward();
                    AssertClose(reference.Grad!.ToArray(), x.Grad!.ToArray(), 1e-4f, $"input gradient, {m} rows");
                }
            }
        }

        // A whole model: int4 predictions near float32; save/load keeps int4; back to float32 keeps the rounded weights.
        using var model = TinyGpt(device);
        using var ids = Tensor.From([1f, 4f, 2f, 7f, 3f], [1, 5], device);
        var original = model.Predict(ids).ToArray();
        Check(model.QuantizeInt4() > 0 && model.Descendants().OfType<Linear>().All(l => l.Int4 is not null), "every Linear is int4");
        var nibbles = model.Predict(ids).ToArray();
        AssertClose(original, nibbles, 0.25f * original.Max(MathF.Abs), "int4 model predictions");
        long bytes = model.Descendants().OfType<Linear>().Sum(l => l.Int4!.Bytes);
        long floats = model.Descendants().OfType<Linear>().Sum(l => 4L * l.InFeatures * l.OutFeatures);
        Check(bytes < floats / 5, $"int4 memory {bytes} vs float32 {floats}");
        string path = Path.GetTempFileName();
        try
        {
            model.Save(path);
            using var reloaded = TinyGpt(device);
            reloaded.Load(path);
            Check(reloaded.Descendants().OfType<Linear>().All(l => l.Int4 is not null), "a float model loads an int4 file as int4");
            Check(reloaded.Predict(ids).ToArray().SequenceEqual(nibbles), "reloaded int4 predictions are identical");
        }
        finally
        {
            File.Delete(path);
        }

        Check(model.ToFloat32() > 0 && model.Descendants().OfType<Linear>().All(l => !l.Packed), "back to float32");
        AssertClose(nibbles, model.Predict(ids).ToArray(), 1e-4f, "float32 keeps the rounded weights");
    }

    private static void PackedMany(Device device)
    {
        var r = new Random(75);
        foreach (int k in new[] { 64, 1024, 3000 })
        {
            foreach (var convert in new Action<Linear>[] { l => l.QuantizeInt8(), l => l.QuantizeInt4(), l => l.ToBFloat16() })
            {
                using var a = new Linear(k, 2048, bias: false, device, r);
                using var b = new Linear(k, 520, bias: true, device, r);
                using var c = new Linear(k, 1030, bias: true, device, r);
                b.Bias!.Load([.. Enumerable.Range(0, 520).Select(i => MathF.Sin(i))]);
                c.Bias!.Load([.. Enumerable.Range(0, 1030).Select(i => MathF.Cos(i))]);
                foreach (var layer in new[] { a, b, c })
                {
                    convert(layer);
                }

                // Prompts (many rows): packed weights expanded as the tiles load, against a float product with the expanded weights.
                foreach (int m in new[] { 100, 300 })
                {
                    using var noGrad = Autograd.NoGrad();
                    using var scope = new TensorScope();
                    using var x = Tensor.From([.. Enumerable.Range(0, m * k).Select(_ => (float)(r.NextDouble() * 2 - 1))], [m, k], device);
                    using var w = Tensor.From(a.WeightValues(), [k, 2048], device);
                    AssertClose(x.MatMul(w).ToArray(), a.Forward(x).ToArray(), 1e-3f * MathF.Sqrt(k), $"{a} prompt product, k {k}, m {m}");
                }

                foreach (int m in new[] { 1, 3, 8 })
                {
                    using var noGrad = Autograd.NoGrad();
                    using var scope = new TensorScope();
                    using var x = Tensor.From([.. Enumerable.Range(0, m * k).Select(_ => (float)(r.NextDouble() * 2 - 1))], [1, m, k], device);
                    var together = Linear.ForwardMany(x, a, b, c);
                    var pair = Linear.ForwardMany(x, b, c);
                    var separate = new[] { a.Forward(x), b.Forward(x), c.Forward(x) };
                    for (int j = 0; j < 3; j++)
                    {
                        AssertClose(separate[j].ToArray(), together[j].ToArray(), 1e-3f * MathF.Sqrt(k), $"{a} k {k}, m {m}, product {j}");
                    }

                    AssertClose(separate[1].ToArray(), pair[0].ToArray(), 1e-3f * MathF.Sqrt(k), $"pair, k {k}, m {m}");
                    AssertClose(separate[2].ToArray(), pair[1].ToArray(), 1e-3f * MathF.Sqrt(k), $"pair, k {k}, m {m}");

                    // The down projection reading act(gate) · up directly (SiLU, GELU).
                    using var gate = Tensor.From([.. Enumerable.Range(0, m * 2048).Select(_ => (float)(r.NextDouble() * 4 - 2))], [1, m, 2048], device);
                    using var up = Tensor.From([.. Enumerable.Range(0, m * 2048).Select(_ => (float)(r.NextDouble() * 2 - 1))], [1, m, 2048], device);
                    using var down = new Linear(2048, 700, bias: false, device, r);
                    convert(down);
                    foreach (int activation in new[] { 0, 1 })
                    {
                        var fused = Linear.MatMulPackedGated(gate, up, activation, down);
                        Check(fused is not null || device.Type != DeviceType.Cuda || down.Int4 is null, "fused gated product on CUDA (int4)");
                        if (fused is not null)
                        {
                            var reference = down.Forward(Tensor.GatedActivation(gate, up, activation));
                            AssertClose(reference.ToArray(), fused.ToArray(), 2e-3f * MathF.Sqrt(2048), $"{down} gated {activation}, m {m}");
                        }
                    }
                }
            }
        }
    }

    private static void Int8CacheKernels(Device device)
    {
        const int Rows = 3, Steps = 2, Capacity = 5, Dim = 10, Start = 1;           // Dim not a multiple of 4
        var r = new Random(49);
        float[] Random(int n) => [.. Enumerable.Range(0, n).Select(_ => (float)(r.NextDouble() * 2 - 1))];
        using var cache = new KeyValueCache(Rows, Capacity, Dim, device, KeyValueFormat.Int8);
        var source = Random(Rows * Steps * Dim);
        using (var src = Tensor.From(source, [Rows, Steps, Dim], device))
        using (var position = Tensor.From([(float)Start], [1], device))
        {
            Tensor.WriteKeyValuesInt8(src, cache.Keys, cache.KeyScales!, position);
            Tensor.WriteKeyValuesInt8(src, cache.Values, cache.ValueScales!, position);
        }

        // The cache as floats, from the rule: scale = max |row| / 127, byte = round-half-even(x / scale).
        var expected = new float[Rows * Capacity * Dim];
        for (int row = 0; row < Rows; row++)
        {
            for (int t = 0; t < Steps; t++)
            {
                var x = source.AsSpan((row * Steps + t) * Dim, Dim);
                float max = 0f;
                foreach (float v in x)
                {
                    max = MathF.Max(max, MathF.Abs(v));
                }

                float scale = max / 127f;
                for (int d = 0; d < Dim; d++)
                {
                    expected[(row * Capacity + Start + t) * Dim + d] = MathF.Round(x[d] * (1f / scale), MidpointRounding.ToEven) * scale;
                }
            }
        }

        var bytes = System.Runtime.InteropServices.MemoryMarshal.Cast<float, sbyte>(cache.Keys.ToArray()).ToArray();
        var scales = cache.KeyScales!.ToArray();
        int stride = (Dim + 3) / 4 * 4;
        var stored = new float[expected.Length];
        for (int slot = 0; slot < Rows * Capacity; slot++)
        {
            for (int d = 0; d < Dim; d++)
            {
                stored[slot * Dim + d] = bytes[slot * stride + d] * scales[slot];
            }
        }

        AssertClose(expected, stored, 1e-6f, "quantized cache contents");

        var q = Random(Rows * Steps * Dim);
        using var tq = Tensor.From(q, [Rows, Steps, Dim], device);
        var scores = Tensor.AttentionScoresInt8(tq, cache).ToArray();
        var weights = Random(Rows * Steps * Capacity);
        using var tw = Tensor.From(weights, [Rows, Steps, Capacity], device);
        var context = Tensor.AttentionContextInt8(tw, cache).ToArray();
        var wantScores = new float[Rows * Steps * Capacity];
        var wantContext = new float[Rows * Steps * Dim];
        for (int row = 0; row < Rows; row++)
        {
            for (int t = 0; t < Steps; t++)
            {
                for (int c = 0; c < Capacity; c++)
                {
                    float sum = 0f;
                    for (int d = 0; d < Dim; d++)
                    {
                        sum += q[(row * Steps + t) * Dim + d] * expected[(row * Capacity + c) * Dim + d];
                        wantContext[(row * Steps + t) * Dim + d] += weights[(row * Steps + t) * Capacity + c] * expected[(row * Capacity + c) * Dim + d];
                    }

                    wantScores[(row * Steps + t) * Capacity + c] = sum;
                }
            }
        }

        AssertClose(wantScores, scores, 1e-5f, "q · int8 keys");
        AssertClose(wantContext, context, 1e-5f, "weights · int8 values");
    }

    private static void Int8CacheDecoding(Device device)
    {
        const int T = 10, V = 11;
        using var model = TinyGpt(device);
        var random = new Random(50);
        var ids = Enumerable.Range(0, T).Select(_ => (float)random.Next(V)).ToArray();
        List<float[]> Decode(KeyValueFormat format, out long bytes)
        {
            var logits = new List<float[]>();
            using var context = new DecodingContext(device, 1, 12, format);
            using (Autograd.NoGrad())
            {
                using var prompt = Tensor.From(ids.AsSpan(0, 4), [1, 4], device);
                logits.Add(model.ForwardCached(prompt, context).ToArray());
                for (int t = 4; t < T; t++)
                {
                    using var next = Tensor.From(ids.AsSpan(t, 1), [1, 1], device);
                    logits.Add(model.ForwardCached(next, context).ToArray());
                }
            }

            bytes = context.CacheBytes;
            return logits;
        }

        var exact = Decode(KeyValueFormat.Float32, out long floatBytes);
        var int8 = Decode(KeyValueFormat.Int8, out long int8Bytes);
        float range = exact.SelectMany(l => l).Max(MathF.Abs);
        for (int step = 0; step < exact.Count; step++)
        {
            float change = exact[step].Zip(int8[step], (a, b) => MathF.Abs(a - b)).Max();
            Check(change < 0.05f * range, $"step {step}: logit change {change} (range {range})");
        }

        // Per cached row: ceil(dh / 4) packed elements + one scale, instead of dh floats (dh = 4 here, so exactly half;
        // at dh = 64 it is 17 / 64 of the memory).
        int dh = model.Descendants().OfType<MultiHeadAttention>().First() is var mha ? mha.Dim / mha.Heads : 0;
        Check(int8Bytes * dh == floatBytes * ((dh + 3) / 4 + 1), $"cache {int8Bytes:N0} bytes vs {floatBytes:N0} (head size {dh})");

        // Recorded decoding steps (CUDA graphs on the GPU) give the same tokens as direct steps with an int8 cache.
        const int B = 2, Steps = 8;
        int[][] Run(bool useGraph)
        {
            using var context = new DecodingContext(device, B, 12, KeyValueFormat.Int8);
            using var sampler = new TokenSampler(device, B, V, Steps + 1) { Temperature = 0.9f, Seed = 7 };
            using (Autograd.NoGrad())
            {
                using (var scope = new TensorScope())
                {
                    sampler.Sample(model.ForwardCached(Tensor.From([1f, 2f, 3f, 4f], [B, 2], device), context));
                }

                void Step() => sampler.Sample(model.ForwardCached(sampler.Ids.Reshape(B, 1), context));
                using var graph = useGraph ? context.CaptureStep(Step) : null;
                for (int s = 0; s < Steps; s++)
                {
                    if (graph is not null)
                    {
                        context.ReplayStep(graph);
                    }
                    else
                    {
                        using var scope = new TensorScope();
                        Step();
                    }
                }

                if (graph is not null && device.Type == DeviceType.Cuda)
                {
                    Check(graph.IsRecorded, $"CUDA graph recording failed: {graph.FailureReason}");
                }
            }

            return [.. sampler.Read(0, Steps + 1).Select(step => step.Select(t => t.Id).ToArray())];
        }

        var direct = Run(useGraph: false);
        var replayed = Run(useGraph: true);
        for (int s = 0; s < direct.Length; s++)
        {
            Check(direct[s].SequenceEqual(replayed[s]), $"step {s}: direct [{string.Join(",", direct[s])}] vs graph [{string.Join(",", replayed[s])}]");
        }

        var generator = new TextGenerator(model, new CharTokenizer("abcdefghijk"), 12) { CacheFormat = KeyValueFormat.Int8 };
        var (text, _, stats) = generator.Generate("abc", new GenerationOptions { Temperature = 0f, TopK = 1, NumPredict = 6 });
        Check(stats.GeneratedTokens == 6 && text.Length == 6, $"generator with an int8 cache: '{text}'");
    }

    private static void FewRowMatMul(Device device)
    {
        var r = new Random(48);
        foreach (var (m, n, k) in new[] { (1, 700, 300), (3, 1030, 257), (4, 512, 512), (8, 300, 1000) })
        {
            foreach (var (transA, transB) in new[] { (false, false), (true, false), (false, true), (true, true) })
            {
                var a = Enumerable.Range(0, m * k).Select(_ => (float)(r.NextDouble() * 2 - 1)).ToArray();
                var b = Enumerable.Range(0, k * n).Select(_ => (float)(r.NextDouble() * 2 - 1)).ToArray();
                using var ta = Tensor.From(a, transA ? [k, m] : [m, k], device);
                using var tb = Tensor.From(b, transB ? [n, k] : [k, n], device);
                var actual = ta.MatMul(tb, transA, transB).ToArray();
                var expected = new float[m * n];
                for (int i = 0; i < m; i++)
                {
                    for (int j = 0; j < n; j++)
                    {
                        double sum = 0;
                        for (int p = 0; p < k; p++)
                        {
                            sum += (double)a[transA ? p * m + i : i * k + p] * b[transB ? j * k + p : p * n + j];
                        }

                        expected[i * n + j] = (float)sum;
                    }
                }

                AssertClose(expected, actual, 1e-3f, $"[{m}x{k}]{(transA ? "ᵀ" : "")} × [{k}x{n}]{(transB ? "ᵀ" : "")}");
            }
        }

        // Batched, as in decoding attention: weights · values and queries · keysᵀ for 3 heads of 2 rows.
        foreach (var (transB, M) in new[] { (false, 2), (true, 2), (false, 150), (true, 150) })
        {
            const int B = 3, K = 600, N = 200;
            var a = Enumerable.Range(0, B * M * K).Select(_ => (float)(r.NextDouble() * 2 - 1)).ToArray();
            var b = Enumerable.Range(0, B * K * N).Select(_ => (float)(r.NextDouble() * 2 - 1)).ToArray();
            using var ta = Tensor.From(a, [B, M, K], device);
            using var tb = Tensor.From(b, transB ? [B, N, K] : [B, K, N], device);
            var actual = ta.MatMul(tb, transposeB: transB).ToArray();
            var expected = new float[B * M * N];
            for (int h = 0; h < B; h++)
            {
                for (int i = 0; i < M; i++)
                {
                    for (int j = 0; j < N; j++)
                    {
                        double sum = 0;
                        for (int p = 0; p < K; p++)
                        {
                            sum += (double)a[(h * M + i) * K + p] * b[h * K * N + (transB ? j * K + p : p * N + j)];
                        }

                        expected[(h * M + i) * N + j] = (float)sum;
                    }
                }
            }

            AssertClose(expected, actual, 1e-3f, $"batched [{B}x{M}x{K}] × [{B}x{K}x{N}]{(transB ? "ᵀ" : "")}");
        }

        // Several products of one input in one pass (query/key/value when decoding), with and without biases.
        foreach (int rows in new[] { 1, 3, 8 })
        {
            const int K = 520;
            int[] widths = [1024, 300, 777];
            using var input = Tensor.From([.. Enumerable.Range(0, rows * K).Select(_ => (float)(r.NextDouble() * 2 - 1))], [1, rows, K], device);
            var ws = widths.Select(n => Tensor.From([.. Enumerable.Range(0, K * n).Select(_ => (float)(r.NextDouble() * 2 - 1))], [K, n], device)).ToList();
            var bs = widths.Select((n, j) => j == 1 ? null : Tensor.From([.. Enumerable.Range(0, n).Select(_ => (float)r.NextDouble())], [n], device)).ToList();
            foreach (int count in new[] { 2, 3 })
            {
                Tensor[] many;
                using (Autograd.NoGrad())
                {
                    many = Tensor.MatMulMany(input, ws[..count], bs[..count]);
                }

                for (int j = 0; j < count; j++)
                {
                    var single = input.Reshape(rows, K).MatMul(ws[j]).ToArray();
                    if (bs[j] is { } bias)
                    {
                        var bv = bias.ToArray();
                        single = [.. single.Select((v, i) => v + bv[i % widths[j]])];
                    }

                    Check(many[j].Shape.SequenceEqual([1, rows, widths[j]]), $"product {j} shape");
                    AssertClose(single, many[j].ToArray(), 1e-3f, $"{count} products, {rows} rows: product {j}");
                }
            }

            ws.ForEach(w => w.Dispose());
            bs.ForEach(b => b?.Dispose());
        }

        // beta = 1 accumulates (the gradient path): dx += dy · wᵀ with one row.
        using var w = Tensor.From([.. Enumerable.Range(0, 300 * 700).Select(i => MathF.Sin(i * 0.01f))], [300, 700], device);
        using var x = Tensor.From([.. Enumerable.Range(0, 300).Select(i => MathF.Cos(i * 0.1f))], [1, 300], device, requiresGrad: true);
        var y = x.MatMul(w);
        y.Sum().Backward();
        y = x.MatMul(w);
        y.Sum().Backward();                                                       // accumulates a second time
        var once = w.ToArray().Chunk(700).Select(row => row.Sum()).ToArray();
        AssertClose([.. once.Select(v => 2 * v)], x.Grad!.ToArray(), 1e-2f, "accumulated gradient");
    }

    private static void Int8Kernels(Device device)
    {
        const int K = 37, N = 23;                                              // N not a multiple of 4: padded rows
        var r = new Random(40);
        var values = Enumerable.Range(0, K * N).Select(_ => (float)(r.NextDouble() * 2 - 1) * (1 + r.Next(3))).ToArray();
        using var weight = Tensor.From(values, [K, N], device);
        using var q = Int8Weight.Quantize(weight);
        using var dequantized = q.Dequantize();
        var deq = dequantized.ToArray();
        for (int j = 0; j < N; j++)
        {
            float scale = Enumerable.Range(0, K).Max(i => MathF.Abs(values[i * N + j])) / 127f;
            for (int i = 0; i < K; i++)
            {
                float expected = MathF.Round(values[i * N + j] / scale) * scale;
                Check(MathF.Abs(deq[i * N + j] - expected) < 1e-5f, $"dequantized [{i}, {j}]");
                Check(MathF.Abs(deq[i * N + j] - values[i * N + j]) <= scale / 2 + 1e-6f, $"rounding error [{i}, {j}]");
            }
        }

        foreach (int m in new[] { 1, 3, 8, 20 })                           // ≤ 8 rows: direct int8 kernel; 20: dequantize + matmul
        {
            var input = Enumerable.Range(0, m * K).Select(_ => (float)(r.NextDouble() * 2 - 1)).ToArray();
            using var x = Tensor.From(input, [m, K], device, requiresGrad: true);
            using var reference = Tensor.From(input, [m, K], device, requiresGrad: true);
            var y = x.MatMulInt8(q);
            var want = reference.MatMul(dequantized);
            AssertClose(want.ToArray(), y.ToArray(), 1e-4f, $"int8 product, {m} rows");
            y.Sum().Backward();
            want.Sum().Backward();
            AssertClose(reference.Grad!.ToArray(), x.Grad!.ToArray(), 1e-4f, $"input gradient, {m} rows");
        }

        // Larger products: the k dimension is split across blocks (partial sums added in order) and N % 4 != 0.
        foreach (var (m, k, n) in new[] { (1, 1030, 301), (5, 4100, 130), (8, 700, 2050) })
        {
            using var w = Tensor.From([.. Enumerable.Range(0, k * n).Select(_ => (float)(r.NextDouble() * 2 - 1))], [k, n], device);
            using var wq = Int8Weight.Quantize(w);
            using var wd = wq.Dequantize();
            using var x = Tensor.From([.. Enumerable.Range(0, m * k).Select(_ => (float)(r.NextDouble() * 2 - 1))], [m, k], device);
            AssertClose(x.MatMul(wd).ToArray(), x.MatMulInt8(wq).ToArray(), 2e-3f, $"int8 product [{m}x{k}] × [{k}x{n}]");
        }

        using var batched = Tensor.From([.. Enumerable.Range(0, 2 * 3 * K).Select(i => MathF.Sin(i))], [2, 3, K], device);
        using var flatInput = batched.Reshape(6, K);
        AssertClose(flatInput.MatMul(dequantized).ToArray(), batched.MatMulInt8(q).ToArray(), 1e-4f, "[batch, time, k] input");
    }

    private static void Int8Gpt(Device device)
    {
        const int T = 10, V = 11;
        using var model = TinyGpt(device);
        var random = new Random(41);
        var ids = Enumerable.Range(0, T).Select(_ => (float)random.Next(V)).ToArray();
        using var sequence = Tensor.From(ids, [1, T], device);
        float[] original;
        using (Autograd.NoGrad())
        {
            original = model.Forward(sequence).ToArray();
        }

        int quantized = model.QuantizeInt8();
        Check(quantized == model.Descendants().OfType<Linear>().Count() && quantized > 0, $"{quantized} layers quantized");
        using (Autograd.NoGrad())
        {
            var full = model.Forward(sequence).ToArray();                  // 10 rows per product: dequantize + matmul
            float change = full.Zip(original, (a, b) => MathF.Abs(a - b)).Max(), range = original.Max(MathF.Abs);
            Check(change < 0.05f * range && change > 0, $"logit change {change} (logit range {range})");

            using var context = new DecodingContext(device, 1, 12);        // one row per product: the direct int8 kernel
            using var prompt = Tensor.From(ids.AsSpan(0, 4), [1, 4], device);
            AssertClose(full[..(4 * V)], model.ForwardCached(prompt, context).ToArray(), 1e-4f, "prefill");
            for (int t = 4; t < T; t++)
            {
                using var next = Tensor.From(ids.AsSpan(t, 1), [1, 1], device);
                AssertClose(full[(t * V)..((t + 1) * V)], model.ForwardCached(next, context).ToArray(), 1e-4f, $"cached step {t}");
            }

            using var exported = OnnxImport.Load(OnnxExport.For(model).Input(T).ToBytes(), device);
            AssertClose(full, exported.Model.Forward(sequence).ToArray(), 1e-4f, "ONNX export of an int8 model (dequantized weights)");
        }
    }

    private static void Int8SaveLoad(Device device)
    {
        var builder = Network.Input(64).Seed(42).Linear(256).ReLU().Linear(256).ReLU().Linear(10);
        using var model = builder.OnDevice(device).Build();
        using var x = Tensor.From([.. Enumerable.Range(0, 5 * 64).Select(i => MathF.Cos(i * 0.1f))], [5, 64], device);
        long floatBytes = model.Parameters().Sum(p => 4L * p.Size);
        model.QuantizeInt8();
        using var expected = model.Predict(x);
        long int8Bytes = model.Parameters().Concat(model.Buffers()).Sum(p => 4L * p.Size);
        Check(int8Bytes < floatBytes * 0.3, $"memory {int8Bytes:N0} bytes vs {floatBytes:N0} as float32");
        Check(model.Parameters().All(p => p.Rank == 1), "only biases remain trainable parameters");
        Check(model.ToString()!.Length > 0 && model.OfType<Linear>().All(l => l.ToString().Contains("int8")), "layers show int8");

        string path = Path.Combine(Path.GetTempPath(), $"ns-{Guid.NewGuid():N}.ikw");
        string floatPath = Path.ChangeExtension(path, ".f32.ikw");
        try
        {
            model.Save(path);
            using (var reference = builder.OnDevice(device).Build())
            {
                reference.Save(floatPath);
            }

            Check(new FileInfo(path).Length < new FileInfo(floatPath).Length * 0.3, $"file {new FileInfo(path).Length:N0} vs {new FileInfo(floatPath).Length:N0} bytes");
            using var loaded = builder.OnDevice(device).Build();                  // a float model: the file makes it int8
            loaded.Load(path);
            Check(loaded.OfType<Linear>().All(l => l.Int8 is not null), "loading quantizes the layers listed in the file");
            AssertClose(expected.ToArray(), loaded.Predict(x).ToArray(), 1e-6f, "reloaded int8 model");
        }
        finally
        {
            File.Delete(path);
            File.Delete(floatPath);
        }

        using var moved = builder.OnDevice(Device.Cpu).Build();
        moved.QuantizeInt8();
        using var cpuResult = moved.Predict(x.To(Device.Cpu));
        moved.To(device);
        AssertClose(cpuResult.ToArray(), moved.Predict(x).ToArray(), 1e-4f, "int8 model moved to the device");

        var deq = model.OfType<Linear>().First().Int8!.Dequantize();
        model.DequantizeInt8();
        Check(model.OfType<Linear>().All(l => l.Int8 is null), "dequantized");
        AssertClose(deq.ToArray(), model.OfType<Linear>().First().Weight.ToArray(), 0f, "dequantized weights are the int8 values");
        deq.Dispose();
        AssertClose(expected.ToArray(), model.Predict(x).ToArray(), 1e-4f, "same outputs after dequantizing");
        Check(model.Parameters().Any(p => p.Rank == 2 && p.RequiresGrad), "weights are trainable again");
    }

    private static void Int8Lora(Device device)
    {
        var random = new Random(43);
        var features = new float[64, 8];
        var targets = new float[64, 1];
        for (int i = 0; i < 64; i++)
        {
            for (int j = 0; j < 8; j++)
            {
                features[i, j] = (float)(random.NextDouble() * 2 - 1);
            }

            targets[i, 0] = features[i, 0] * 2 - features[i, 3] + 0.5f;
        }

        using var model = Network.Input(8).OnDevice(device).Seed(44).Linear(32).Tanh().Linear(1).Build();
        model.QuantizeInt8();
        var before = model.OfType<Linear>().Select(l => l.Int8!.Dequantize()).ToList();
        int adapters = model.AddLora(rank: 4, alpha: 8, targets: _ => true, freezeBase: true, random: new Random(45));
        Check(adapters == 2, $"{adapters} adapters");
        var data = Dataset.FromArrays(features, targets);
        using var trainer = new Trainer(model, Losses.MeanSquaredError, p => new Adam(p, 0.02f));
        var history = trainer.Fit(new DataLoader(data, 16, shuffle: true, device: device, seed: 46), 40);
        Check(history.Epochs[^1].Loss < history.Epochs[0].Loss * 0.5, $"loss {history.Epochs[0].Loss:F4} → {history.Epochs[^1].Loss:F4}");
        foreach (var (linear, weight) in model.OfType<Linear>().Zip(before))
        {
            using var after = linear.Int8!.Dequantize();
            AssertClose(weight.ToArray(), after.ToArray(), 0f, "int8 weights unchanged by training");
            weight.Dispose();
        }
    }

    private static void HalfPrecisionFiles(Device device)
    {
        var builder = Network.Input(32).Seed(47).Linear(128).ReLU().Linear(32);
        using var model = builder.OnDevice(device).Build();
        var original = model.Parameters().SelectMany(p => p.ToArray()).ToArray();
        string dir = Path.Combine(Path.GetTempPath(), $"ns-{Guid.NewGuid():N}");
        Directory.CreateDirectory(dir);
        try
        {
            var sizes = new Dictionary<WeightFormat, long>();
            foreach (var format in new[] { WeightFormat.Float32, WeightFormat.Float16, WeightFormat.BFloat16 })
            {
                string path = Path.Combine(dir, $"{format}.ikw");
                model.Save(path, format);
                sizes[format] = new FileInfo(path).Length;
                using var loaded = builder.OnDevice(device).Build();
                loaded.Load(path);
                var values = loaded.Parameters().SelectMany(p => p.ToArray()).ToArray();
                float relative = format switch { WeightFormat.Float32 => 0f, WeightFormat.Float16 => 1f / 1024, _ => 1f / 128 };
                for (int i = 0; i < values.Length; i++)
                {
                    Check(MathF.Abs(values[i] - original[i]) <= relative * MathF.Abs(original[i]) + 1e-7f, $"{format} value {i}: {values[i]} vs {original[i]}");
                }
            }

            Check(sizes[WeightFormat.Float16] < sizes[WeightFormat.Float32] * 0.52 && sizes[WeightFormat.BFloat16] == sizes[WeightFormat.Float16],
                $"sizes {string.Join(", ", sizes)}");

            // A quantized model saved as Float16 keeps its int8 bytes exact.
            model.QuantizeInt8();
            using var expected = model.Predict(Tensor.Ones([2, 32], device));
            string mixed = Path.Combine(dir, "int8-f16.ikw");
            model.Save(mixed, WeightFormat.Float16);
            using var reloaded = builder.OnDevice(device).Build();
            reloaded.Load(mixed);
            AssertClose(expected.ToArray(), reloaded.Predict(Tensor.Ones([2, 32], device)).ToArray(), 1e-2f, "int8 + float16 biases");
        }
        finally
        {
            Directory.Delete(dir, recursive: true);
        }
    }
}
