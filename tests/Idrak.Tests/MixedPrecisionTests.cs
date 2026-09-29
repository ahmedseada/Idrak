using Idrak;
using Idrak.Backends.Cuda;
using Idrak.Layers;
using Idrak.Optimizers;

internal static partial class Tests
{
    private static readonly (string Name, Action<Device> Run)[] MixedPrecisionGroup =
    [
        ("mixed precision: bfloat16 tensor-core products (every transpose, edges, batches, beta, split k) match bfloat16-rounded references", TensorCoreProducts),
        ("mixed precision: tensor-core flash attention (forward, log-sum-exp, dq, dk, dv; head sizes 64 and 128) matches float32", TensorCoreAttention),
        ("layer norm: the fused training kernels match the composed operations (output, input, gamma and beta gradients)", LayerNormTraining),
        ("mixed precision: strided products and GELU epilogues match references (slices, activation, its gradient)", StridedProducts),
        ("mixed precision: 8-bit tensor-core products (fp8, int8; every layout, beta, bias) track float32", EightBitProducts),
        ("mixed precision: 8-bit column quantizers: delayed scaling (given maxima with headroom, recorded maxima, correction of saturated columns) matches the two-launch pair", ColumnQuantizers),
        ("mixed precision: prompts through int8 / int4 / bfloat16 weights on tensor cores match the float32 kernels; several layers in one launch match their own products", PackedTensorCoreProducts),
        ("mixed precision: fused decoder blocks (packed q/k/v, attention in place, GELU inside the products) match the composed ones", FusedDecoderBlocks),
        ("mixed precision: product + bias in one pass matches the product and a bias addition (output and gradients)", MatMulBiasPass),
        ("optimizer: 8-bit AdamW (dynamic code map, nearest codes, tracks 32-bit AdamW, CPU parity)", EightBitAdam),
        ("decoder: GPT-2 style (learned positions, LayerNorm, GELU, dropout): checkpointed blocks replay the dropout masks", GptStyleDecoder),
    ];

    private static void TensorCoreAttention(Device device)
    {
        bool tensorCores = MixedPrecision.TensorCoresUnavailable(device) is null;
        var random = new Random(8);
        float[] Values(int n) => [.. Enumerable.Range(0, n).Select(_ => random.NextSingle() * 2 - 1)];
        static double RelativeError(float[] expected, float[] actual) =>
            Math.Sqrt(expected.Zip(actual).Sum(p => (double)(p.First - p.Second) * (p.First - p.Second)) / Math.Max(1e-12, expected.Sum(v => (double)v * v)));

        foreach (int dim in new[] { 64, 128 })
        {
            foreach (var (rows, steps, capacity, offset, backward) in new[] { (100, 100, 100, 0, true), (130, 65, 65, 0, true), (70, 70, 150, 40, false), (192, 192, 192, 0, true) })
            {
                const int Heads = 3;
                float scale = 1f / MathF.Sqrt(dim);
                float[] q = Values(Heads * rows * dim), k = Values(Heads * capacity * dim), v = Values(Heads * capacity * dim);
                float[] dOut = Values(Heads * rows * dim), dq0 = Values(Heads * rows * dim), dk0 = Values(Heads * capacity * dim), dv0 = Values(Heads * capacity * dim);
                (float[] Y, float[] Lse, float[] Dq, float[] Dk, float[] Dv) Run(Device on)
                {
                    var backend = on.Backend;
                    using var tq = Tensor.From(q, [q.Length], on);
                    using var tk = Tensor.From(k, [k.Length], on);
                    using var tv = Tensor.From(v, [v.Length], on);
                    using var position = Tensor.From([offset], [1], on);
                    using var y = Tensor.Zeros([q.Length], on);
                    using var lse = Tensor.Zeros([Heads * rows], on);
                    using var tdo = Tensor.From(dOut, [dOut.Length], on);
                    using var dq = Tensor.From(dq0, [dq0.Length], on);
                    using var dk = Tensor.From(dk0, [dk0.Length], on);
                    using var dv = Tensor.From(dv0, [dv0.Length], on);
                    using (MixedPrecision.BFloat16())
                    {
                        backend.AttentionTiled(tq.Storage, tk.Storage, tv.Storage, position.Storage, y.Storage, lse.Storage, Heads, rows, steps, capacity, dim, scale);
                        if (backward)
                        {
                            backend.AttentionTiledBackward(tq.Storage, tk.Storage, tv.Storage, y.Storage, lse.Storage, tdo.Storage, dq.Storage, dk.Storage, dv.Storage,
                                Heads, rows, steps, capacity, dim, scale);
                        }
                    }

                    return (y.ToArray(), lse.ToArray(), dq.ToArray(), dk.ToArray(), dv.ToArray());
                }

                var expected = Run(Device.Cpu);
                var got = Run(device);
                string what = $"d {dim}, rows {rows}, steps {steps}, capacity {capacity}, offset {offset}";
                double tolerance = tensorCores ? 1.5e-2 : 1e-4;
                Check(RelativeError(expected.Y, got.Y) < tolerance, $"{what}: output error {RelativeError(expected.Y, got.Y):G3}");
                Check(RelativeError(expected.Lse, got.Lse) < tolerance, $"{what}: log-sum-exp error {RelativeError(expected.Lse, got.Lse):G3}");
                if (backward)
                {
                    // Relative to the gradients alone (the tensors started from random values they are added to).
                    float[] Minus(float[] a, float[] b) => [.. a.Zip(b).Select(p => p.First - p.Second)];
                    foreach (var (name, e, g, start) in new[] { ("dq", expected.Dq, got.Dq, dq0), ("dk", expected.Dk, got.Dk, dk0), ("dv", expected.Dv, got.Dv, dv0) })
                    {
                        double error = RelativeError(Minus(e, start), Minus(g, start));
                        Check(error < 2 * tolerance, $"{what}: {name} error {error:G3}");
                    }
                }
            }
        }
    }

    private static void MatMulBiasPass(Device device)
    {
        var random = new Random(6);
        const int M = 150, K = 96, N = 130;
        float[] Values(int n) => [.. Enumerable.Range(0, n).Select(_ => random.NextSingle() * 2 - 1)];
        float[] x0 = Values(M * K), w0 = Values(K * N), b0 = Values(N), g0 = Values(M * N);
        (float[] Y, float[] Dx, float[] Dw, float[] Db) Run(bool fused)
        {
            using var precision = MixedPrecision.BFloat16();
            using var scope = new TensorScope();
            var x = Tensor.From(x0, [3, M / 3, K], device, requiresGrad: true);
            var w = Tensor.From(w0, [K, N], device, requiresGrad: true);
            var b = Tensor.From(b0, [N], device, requiresGrad: true);
            var y = fused ? Tensor.MatMulBias(x, w, b) : x.MatMul(w) + b;
            (y * Tensor.From(g0, y.Shape, device)).Sum().Backward();
            return (y.ToArray(), x.Grad!.ToArray(), w.Grad!.ToArray(), b.Grad!.ToArray());
        }

        var separate = Run(false);
        var fusedRun = Run(true);
        AssertClose(separate.Y, fusedRun.Y, 1e-5f, "output");
        AssertClose(separate.Dx, fusedRun.Dx, 1e-4f, "input gradient");
        AssertClose(separate.Dw, fusedRun.Dw, 1e-4f, "weight gradient");
        AssertClose(separate.Db, fusedRun.Db, 1e-4f, "bias gradient");
    }

    private static float GeluTanh(float x) => 0.5f * x * (1f + MathF.Tanh(0.7978845608f * (x + 0.044715f * x * x * x)));

    private static float GeluTanhGradient(float x)
    {
        float t = MathF.Tanh(0.7978845608f * (x + 0.044715f * x * x * x));
        return 0.5f * (1f + t) + 0.5f * x * (1f - t * t) * 0.7978845608f * (1f + 3f * 0.044715f * x * x);
    }

    private static void StridedProducts(Device device)
    {
        if (MixedPrecision.TensorCoresUnavailable(device) is not null)
        {
            return;                                                          // CPU and older GPUs: the fused paths do not run
        }

        var backend = device.Backend;
        var random = new Random(12);
        float[] Values(int n) => [.. Enumerable.Range(0, n).Select(_ => random.NextSingle() * 2 - 1)];
        // c[:, 5:5+n] of a [m, ldc] block = a[:, 3:3+k] (lda) · b[2:2+k rows, 7:7+n] (ldb), every transpose.
        const int M = 70, N = 90, K = 50, Lda = 64, Ldb = 101, Ldc = 104;
        foreach (bool ta in new[] { false, true })
        {
            foreach (bool tb in new[] { false, true })
            {
                int aRows = ta ? K : M, bRows = tb ? N : K;
                float[] a = Values((aRows + 3) * Lda), b = Values((bRows + 2) * Ldb), c = Values(M * Ldc), bias = Values(N);
                using var ta_ = Tensor.From(a, [a.Length], device);
                using var tb_ = Tensor.From(b, [b.Length], device);
                using var tc_ = Tensor.From(c, [c.Length], device);
                using var tbias = Tensor.From(bias, [N], device);
                Check(backend.GemmStrided(ta_.Storage, 3, Lda, ta, tb_.Storage, 2L * Ldb + 7, Ldb, tb, tc_.Storage, 5, Ldc, M, N, K, 0.5f, tbias.Storage),
                    "strided product runs");
                var got = tc_.ToArray();
                for (int i = 0; i < M; i++)
                {
                    for (int j = 0; j < N; j++)
                    {
                        double sum = 0;
                        for (int q = 0; q < K; q++)
                        {
                            float x = ta ? a[3 + q * Lda + i] : a[3 + i * Lda + q], y = tb ? b[2 * Ldb + 7 + j * Ldb + q] : b[2 * Ldb + 7 + q * Ldb + j];
                            sum += (double)RoundBFloat16(x) * RoundBFloat16(y);
                        }

                        double want = sum + 0.5 * c[5 + i * Ldc + j] + bias[j];
                        Check(Math.Abs(got[5 + i * Ldc + j] - want) < 1e-3 * Math.Max(1, Math.Abs(want)), $"{(ta ? 't' : 'n')}{(tb ? 't' : 'n')} [{i}, {j}]: {got[5 + i * Ldc + j]} vs {want}");
                    }
                }

                Check(got[4] == c[4] && got[5 + N] == c[5 + N], "outside the slice untouched");
            }
        }

        // GELU epilogue (with the pre-activations kept) and the GELU-gradient epilogue.
        {
            float[] a = Values(M * K), b = Values(K * N), bias = Values(N), pre0 = Values(M * N);
            using var ta_ = Tensor.From(a, [a.Length], device);
            using var tb_ = Tensor.From(b, [b.Length], device);
            using var tbias = Tensor.From(bias, [N], device);
            using var act = Tensor.Zeros([M * N], device);
            using var pre = Tensor.Zeros([M * N], device);
            Check(backend.GemmStrided(ta_.Storage, 0, K, false, tb_.Storage, 0, N, false, act.Storage, 0, N, M, N, K, 0f, tbias.Storage,
                Idrak.Backends.GemmEpilogue.Gelu, pre.Storage), "GELU epilogue runs");
            var acts = act.ToArray();
            var pres = pre.ToArray();
            // The gradient variant multiplies by a transposed weight (dPre = g · Wᵀ): b stored [n, k].
            float[] bt = new float[N * K];
            for (int q = 0; q < K; q++)
            {
                for (int j = 0; j < N; j++)
                {
                    bt[j * K + q] = b[q * N + j];
                }
            }

            using var tbt = Tensor.From(bt, [bt.Length], device);
            using var saved = Tensor.From(pre0, [pre0.Length], device);
            using var grad = Tensor.Zeros([M * N], device);
            Check(backend.GemmStrided(ta_.Storage, 0, K, false, tbt.Storage, 0, K, true, grad.Storage, 0, N, M, N, K, 0f, null,
                Idrak.Backends.GemmEpilogue.GeluGradient, saved.Storage), "GELU-gradient epilogue runs");
            var grads = grad.ToArray();
            for (int i = 0; i < M; i++)
            {
                for (int j = 0; j < N; j++)
                {
                    double sum = 0;
                    for (int q = 0; q < K; q++)
                    {
                        sum += (double)RoundBFloat16(a[i * K + q]) * RoundBFloat16(b[q * N + j]);
                    }

                    float p = (float)(sum + bias[j]);
                    int o = i * N + j;
                    Check(Math.Abs(pres[o] - p) < 1e-3f * Math.Max(1, Math.Abs(p)), $"pre-activation [{i}, {j}]");
                    Check(Math.Abs(acts[o] - GeluTanh(p)) < 2e-3f * Math.Max(1, Math.Abs(p)), $"gelu [{i}, {j}]: {acts[o]} vs {GeluTanh(p)}");
                    float want = (float)sum * GeluTanhGradient(pre0[o]);
                    Check(Math.Abs(grads[o] - want) < 2e-3f * Math.Max(1, Math.Abs(want)), $"gelu gradient [{i}, {j}]: {grads[o]} vs {want}");
                }
            }
        }
    }

    private static double Relative(float[] expected, float[] actual) =>
        Math.Sqrt(expected.Zip(actual).Sum(p => (double)(p.First - p.Second) * (p.First - p.Second)) / Math.Max(1e-30, expected.Sum(v => (double)v * v)));

    private static void ColumnQuantizers(Device device)
    {
        if (device.Type != DeviceType.Cuda)
        {
            return;                                                  // CUDA kernels only
        }

        var backend = (CudaBackend)device.Backend;
        var random = new Random(12);
        foreach (bool fp8 in new[] { true, false })
        {
            if (!backend.EightBitReady(fp8))
            {
                Console.WriteLine($"    ({(fp8 ? "fp8" : "int8")} products unavailable on this GPU; skipped)");
                continue;
            }

            // x [k][ld] with `cols` used columns, one outlier column; the columns become the quantized rows.
            foreach (var (k, cols, ld) in new[] { (700, 290, 300), (1100, 64, 64), (33, 5, 7) })
            {
                var values = new float[k * ld];
                for (int i = 0; i < values.Length; i++)
                {
                    values[i] = (random.NextSingle() * 2 - 1) * (i % ld == 3 ? 40f : 1f);
                }

                var maxima = new float[cols];
                for (int r = 0; r < k; r++)
                {
                    for (int c = 0; c < cols; c++)
                    {
                        maxima[c] = MathF.Max(maxima[c], MathF.Abs(values[r * ld + c]));
                    }
                }

                int kp = PtxKernels.EightBitPaddedK(k);
                using var x = Tensor.From(values, [values.Length], device);
                (byte[] Bytes, float[] Scales) Run(int variant, float[]? given = null, Tensor? record = null)
                {
                    using var output = Tensor.Zeros([cols * kp / 4], device);
                    using var scale = Tensor.Zeros([cols], device);
                    using var maximaTensor = given is null ? null : Tensor.From(given, [cols], device);
                    backend.QuantizeColumnsVariant(fp8, x.Storage, ld, output.Storage, scale.Storage, cols, k, variant, maximaTensor?.Storage, record?.Storage);
                    return (System.Runtime.InteropServices.MemoryMarshal.AsBytes(output.ToArray().AsSpan()).ToArray(), scale.ToArray());
                }

                string what = $"{(fp8 ? "fp8" : "int8")} {k}x{cols} (ld {ld})";
                var (bytes, scales) = Run(0);
                // Kept maxima get twice their value as headroom: half the true maxima give the exact result.
                using var recorded = Tensor.Zeros([cols], device);
                var (delayed, delayedScales) = Run(2, [.. maxima.Select(m => m / 2)], recorded);
                Check(bytes.AsSpan().SequenceEqual(delayed) && scales.AsSpan().SequenceEqual(delayedScales), $"{what}: given maxima (with headroom)");
                Check(recorded.ToArray().AsSpan().SequenceEqual(maxima), $"{what}: recorded maxima");
                using var recordedStale = Tensor.Zeros([cols], device);
                var (_, staleScales) = Run(2, [.. maxima.Select(m => m / 4)], recordedStale);
                Check(staleScales.Zip(scales).All(p => MathF.Abs(p.First * 2 - p.Second) <= 1e-6f * p.Second), $"{what}: stale maxima set the scales");
                Check(recordedStale.ToArray().AsSpan().SequenceEqual(maxima), $"{what}: maxima recorded under stale scales");

                // Delayed scaling as training runs it: values more than twice the kept maxima → every strip corrected to
                // the exact result; within the headroom → no correction (scales from the kept maxima), true maxima recorded.
                using var recordedSmall = Tensor.Zeros([cols], device);
                var (corrected, correctedScales) = Run(3, [.. maxima.Select(m => m / 4)], recordedSmall);
                Check(bytes.AsSpan().SequenceEqual(corrected) && scales.AsSpan().SequenceEqual(correctedScales), $"{what}: saturated columns corrected");
                using var recordedLarge = Tensor.Zeros([cols], device);
                var (_, largeScales) = Run(3, maxima, recordedLarge);
                Check(largeScales.Zip(scales).All(p => MathF.Abs(p.First - p.Second * 2) <= 1e-6f * p.First), $"{what}: kept maxima within the headroom");
                Check(recordedLarge.ToArray().AsSpan().SequenceEqual(maxima), $"{what}: maxima recorded without correction");
            }
        }
    }

    private static void EightBitProducts(Device device)
    {
        if (device.Type != DeviceType.Cuda || MixedPrecision.TensorCoresUnavailable(device) is not null)
        {
            return;
        }

        var backend = (CudaBackend)device.Backend;
        var random = new Random(21);
        float[] Values(int n) => [.. Enumerable.Range(0, n).Select(_ => random.NextSingle() * 2 - 1)];
        foreach (bool fp8 in new[] { true, false })
        {
            if (!backend.EightBitReady(fp8))
            {
                Console.WriteLine($"    ({(fp8 ? "fp8" : "int8")} products unavailable on this GPU)");
                continue;
            }

            foreach (var (m, n, k) in new[] { (100, 70, 90), (256, 130, 200), (129, 256, 64) })
            {
                foreach (bool ta in new[] { false, true })
                {
                    foreach (bool tb in new[] { false, true })
                    {
                        float[] a = Values(m * k), b = Values(k * n), c0 = Values(m * n), bias = Values(n);
                        using var ta_ = Tensor.From(a, [a.Length], device);
                        using var tb_ = Tensor.From(b, [b.Length], device);
                        using var tc_ = Tensor.From(c0, [c0.Length], device);
                        using var tbias = Tensor.From(bias, [n], device);
                        backend.Gemm8(fp8, P(ta_), ta ? m : k, ta, P(tb_), tb ? k : n, tb, P(tc_), n, m, n, k, 0.5f, P(tbias), Idrak.Backends.GemmEpilogue.None, 0UL);
                        var got = tc_.ToArray();
                        var want = new float[m * n];
                        for (int i = 0; i < m; i++)
                        {
                            for (int j = 0; j < n; j++)
                            {
                                double sum = 0;
                                for (int q = 0; q < k; q++)
                                {
                                    sum += (double)(ta ? a[q * m + i] : a[i * k + q]) * (tb ? b[j * k + q] : b[q * n + j]);
                                }

                                want[i * n + j] = (float)(sum + 0.5 * c0[i * n + j] + bias[j]);
                            }
                        }

                        double error = Relative(want, got);
                        Check(error < (fp8 ? 0.05 : 0.01), $"{(fp8 ? "fp8" : "int8")} {m}×{n}×{k} {(ta ? 't' : 'n')}{(tb ? 't' : 'n')}: relative error {error:G3}");
                    }
                }
            }
        }
    }

    private static void PackedTensorCoreProducts(Device device)
    {
        if (MixedPrecision.TensorCoresUnavailable(device) is not null)
        {
            return;
        }

        var random = new Random(22);
        // k = 1056 splits k over several blocks (few output tiles), with an uneven last chunk; 150 rows take the 64-row
        // tiles (the last 128-row tile would be mostly empty), 200 rows the 128-row ones.
        foreach (var (M, K, N) in new[] { (150, 256, 200), (150, 1056, 200), (200, 1056, 200) })
        {
            float[] x = [.. Enumerable.Range(0, M * K).Select(_ => random.NextSingle() * 2 - 1)];
            float[] w = [.. Enumerable.Range(0, K * N).Select(_ => (random.NextSingle() * 2 - 1) * 0.1f)];
            using var weights = Tensor.From(w, [K, N], device);
            using var int8 = Int8Weight.Quantize(weights);
            using var int4 = Int4Weight.Quantize(weights);
            using var bf16 = BFloat16Weight.Convert(weights);
            using var input = Tensor.From(x, [M, K], device);
            foreach (var (name, run) in new (string, Func<Tensor>)[]
            {
                ("int8", () => input.MatMulInt8(int8)),
                ("int4", () => input.MatMulInt4(int4)),
                ("bf16", () => input.MatMulBFloat16(bf16)),
            })
            {
                float[] plain, tensor;
                using (MixedPrecision.Use(MatMulPrecision.Float32))
                using (var y = run())
                {
                    plain = y.ToArray();
                }

                using (MixedPrecision.BFloat16())
                using (var y = run())
                {
                    tensor = y.ToArray();
                }

                double error = Relative(plain, tensor);
                Check(error < 0.01, $"{name} {M}x{K}x{N}: tensor-core prompt product differs from the float32 kernel by {error:G3}");
            }
        }

        // Several layers of one input in one launch (queries/keys/values, gate/up): widths 256 + 128 + 384 and 128 + 128,
        // k 1056 (split); each output must equal that layer's own prompt product.
        foreach (var widths in new[] { new[] { 256, 128, 384 }, new[] { 128, 128 } })
        {
            const int M = 150, K = 1056;
            using var input = Tensor.From([.. Enumerable.Range(0, M * K).Select(_ => random.NextSingle() * 2 - 1)], [M, K], device);
            foreach (string format in new[] { "int8", "int4", "bf16" })
            {
                var layers = widths.Select(n =>
                {
                    using var weights = Tensor.From([.. Enumerable.Range(0, K * n).Select(_ => (random.NextSingle() * 2 - 1) * 0.1f)], [K, n], device);
                    return format switch
                    {
                        "int8" => Linear.FromInt8(Int8Weight.Quantize(weights)),
                        "int4" => Linear.FromInt4(Int4Weight.Quantize(weights)),
                        _ => Linear.FromBFloat16(BFloat16Weight.Convert(weights)),
                    };
                }).ToArray();
                try
                {
                    using (MixedPrecision.BFloat16())
                    using (Autograd.NoGrad())
                    using (var scope = new TensorScope())
                    {
                        var together = Tensor.MatMulPackedMany(input, format switch { "int8" => 0, "int4" => 1, _ => 2 }, layers);
                        Check(together is not null, $"{format} {string.Join('+', widths)}: one launch");
                        for (int j = 0; j < layers.Length; j++)
                        {
                            double error = Relative(layers[j].Forward(input).ToArray(), together![j].ToArray());
                            Check(error < 1e-5, $"{format} {string.Join('+', widths)}: layer {j} differs from its own product by {error:G3}");
                        }
                    }
                }
                finally
                {
                    foreach (var layer in layers)
                    {
                        layer.Dispose();
                    }
                }
            }
        }
    }

    private static void FusedDecoderBlocks(Device device)
    {
        if (MixedPrecision.TensorCoresUnavailable(device) is not null)
        {
            return;
        }

        foreach (var (heads, kvHeads) in new[] { (2, 2), (4, 2) })
        {
            var spec = new DecoderSpec
            {
                Vocabulary = 50, Dim = 128, Layers = 2, Heads = heads, KvHeads = kvHeads, HeadDim = 64, FfDim = 256, MaxPositions = 80,
                Norm = DecoderNorm.Layer, NormEpsilon = 1e-5f, Gated = false, Activation = FeedForwardActivation.Gelu, Rope = null,
                QkvBias = true, OutputBias = true, FeedForwardBias = true, TieEmbeddings = true, LearnedPositions = true,
            };
            float[] ids = [.. Enumerable.Range(0, 2 * 70).Select(i => (float)(i * 7 % 50))];
            (float[] Loss, float[][] Grads) Run(bool fused)
            {
                Idrak.Layers.FusedTraining.Enabled = fused;
                try
                {
                    using var model = spec.Build(options: new DecoderBuildOptions { Device = device, Seed = 3, InitStd = 0.05f });
                    model.Train();
                    using var precision = MixedPrecision.BFloat16();
                    using var scope = new TensorScope();
                    var h = Tensor.From(ids, [2, 70], device);
                    foreach (var module in model)
                    {
                        h = module is DecoderBlock block ? block.ForwardCheckpointed(h) : module.Forward(h);
                    }

                    var loss = (h * Tensor.From([.. Enumerable.Range(0, h.Size).Select(i => MathF.Sin(i))], h.Shape, device)).Sum();
                    loss.Backward();
                    return ([loss.Item()], [.. model.Parameters().Select(p => p.Grad!.ToArray())]);
                }
                finally
                {
                    Idrak.Layers.FusedTraining.Enabled = true;
                }
            }

            var composed = Run(false);
            var fusedRun = Run(true);
            AssertClose(composed.Loss, fusedRun.Loss, 2e-3f, $"heads {heads}/{kvHeads}: loss");
            for (int i = 0; i < composed.Grads.Length; i++)
            {
                double norm = Math.Sqrt(composed.Grads[i].Sum(v => (double)v * v)), diff = Math.Sqrt(composed.Grads[i].Zip(fusedRun.Grads[i]).Sum(p => (double)(p.First - p.Second) * (p.First - p.Second)));
                Check(diff <= 2e-2 * norm + 1e-6, $"heads {heads}/{kvHeads}: gradient {i} differs by {diff:G3} (norm {norm:G3})");
            }
        }
    }

    private static void LayerNormTraining(Device device)
    {
        var random = new Random(4);
        foreach (var (rows, cols) in new[] { (37, 50), (5, 300), (64, 2048) })
        {
            float[] Values(int n) => [.. Enumerable.Range(0, n).Select(_ => random.NextSingle() * 4 - 1)];
            float[] x0 = Values(rows * cols), g0 = Values(cols), b0 = Values(cols), w = Values(rows * cols);
            (float[] Y, float[] Dx, float[] Dg, float[] Db) Run(bool fused)
            {
                using var scope = new TensorScope();
                var x = Tensor.From(x0, [rows, cols], device, requiresGrad: true);
                var gamma = Tensor.From(g0, [cols], device, requiresGrad: true);
                var beta = Tensor.From(b0, [cols], device, requiresGrad: true);
                var y = fused ? x.LayerNormTrain(gamma, beta, 1e-5f)
                    : x.Normalize(1, rows, cols, 1e-5f, out _, out _).GroupAffine(gamma, beta, cols, 1);
                (y * Tensor.From(w, [rows, cols], device)).Sum().Backward();
                return (y.ToArray(), x.Grad!.ToArray(), gamma.Grad!.ToArray(), beta.Grad!.ToArray());
            }

            var composed = Run(false);
            var fusedRun = Run(true);
            AssertClose(composed.Y, fusedRun.Y, 1e-4f, $"{rows}×{cols}: output");
            AssertClose(composed.Dx, fusedRun.Dx, 1e-3f, $"{rows}×{cols}: input gradient");
            AssertClose(composed.Dg, fusedRun.Dg, 1e-3f, $"{rows}×{cols}: gamma gradient");
            AssertClose(composed.Db, fusedRun.Db, 1e-3f, $"{rows}×{cols}: beta gradient");
        }
    }

    private static void EightBitAdam(Device device)
    {
        var signedMap = AdamW8Bit.DynamicMap(signed: true);
        var unsignedMap = AdamW8Bit.DynamicMap(signed: false);
        Check(signedMap.Length == 256 && unsignedMap.Length == 256, "256 codes");
        Check(signedMap.Zip(signedMap.Skip(1)).All(p => p.First <= p.Second) && signedMap.Contains(0f) && signedMap[^1] == 1f && signedMap[0] > -1f,
            "signed map sorted in [-1, 1] with 0 and 1");
        Check(unsignedMap[0] == 0f && unsignedMap[^1] == 1f && unsignedMap.Distinct().Count() == 256, "unsigned map: 256 distinct codes in [0, 1]");
        var random = new Random(5);
        for (int i = 0; i < 2000; i++)
        {
            float x = random.NextSingle() * 2 - 1;
            var map = i % 2 == 0 ? signedMap : unsignedMap;
            x = i % 2 == 0 ? x : MathF.Abs(x);
            int got = AdamW8Bit.Nearest(map, x);
            float best = map.Min(c => MathF.Abs(c - x));
            Check(MathF.Abs(map[got] - x) <= best, $"nearest code to {x}: {map[got]}, best distance {best}");
        }

        const int N = 20000;
        float[] start = [.. Enumerable.Range(0, N).Select(_ => random.NextSingle() * 2 - 1)];
        float[] target = [.. Enumerable.Range(0, N).Select(_ => random.NextSingle() * 2 - 1)];
        (float Loss, float[] Values) Run(Device on, bool eightBit)
        {
            using var p = Tensor.Persistent(start, [N], on, requiresGrad: true);
            using var t = Tensor.From(target, [N], on);
            using Optimizer optimizer = eightBit ? new AdamW8Bit([p], 0.01f, 0.9f, 0.95f, weightDecay: 0.1f) : new AdamW([p], 0.01f, 0.9f, 0.95f, weightDecay: 0.1f);
            float loss = 0;
            for (int step = 0; step < 100; step++)
            {
                using var scope = new TensorScope();
                optimizer.ZeroGrad();
                var l = (p - t).Square().Sum() * 0.5f;
                l.Backward();
                optimizer.ClipGradientNorm(50f);                     // active early on (the norm starts near 115)
                optimizer.Step();
                loss = l.Item();
            }

            return (loss, p.ToArray());
        }

        var full = Run(device, eightBit: false);
        var quantized = Run(device, eightBit: true);
        Check(quantized.Loss < 0.2f * N / 6 && Math.Abs(quantized.Loss - full.Loss) < 0.05f * full.Loss + 1f,
            $"8-bit AdamW loss {quantized.Loss:F2}, 32-bit {full.Loss:F2}");
        if (device.Type != DeviceType.Cpu)
        {
            var cpu = Run(Device.Cpu, eightBit: true);
            AssertClose(cpu.Values, quantized.Values, 2e-3f, "8-bit AdamW on the CPU and the device");
        }
    }

    private static void GptStyleDecoder(Device device)
    {
        var spec = new DecoderSpec
        {
            Vocabulary = 23, Dim = 16, Layers = 2, Heads = 2, KvHeads = 2, HeadDim = 8, FfDim = 64, MaxPositions = 12,
            Norm = DecoderNorm.Layer, NormEpsilon = 1e-5f, Gated = false, Activation = FeedForwardActivation.Gelu, Rope = null,
            FeedForwardBias = true, TieEmbeddings = true, LearnedPositions = true, Dropout = 0.2f,
        };
        Check(DecoderSpec.FromJson(spec.ToJson()) == spec, "JSON round trip");
        float[] ids = [3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5, 8, 9, 7];
        (float Loss, float[][] Grads) Run(bool checkpointed)
        {
            using var model = spec.Build(options: new DecoderBuildOptions { Device = device, Seed = 7, InitStd = 0.02f });
            model.Train();
            using var scope = new TensorScope();
            var hidden = Tensor.From(ids, [2, 7], device);
            foreach (var module in model)
            {
                hidden = checkpointed && module is DecoderBlock block ? block.ForwardCheckpointed(hidden) : module.Forward(hidden);
            }

            var weights = Tensor.From([.. Enumerable.Range(0, hidden.Size).Select(i => MathF.Cos(i))], hidden.Shape, device);
            var loss = (hidden * weights).Sum();
            loss.Backward();
            var position = model.OfType<PositionEmbedding>().Single().Weight.Grad!.ToArray();
            Check(position.Take(7 * spec.Dim).Any(v => v != 0) && position.Skip(7 * spec.Dim).All(v => v == 0), "positions used get gradients, the rest none");
            return (loss.Item(), [.. model.Parameters().Select(p => p.Grad!.ToArray())]);
        }

        var plain = Run(false);
        var checkpointed = Run(true);
        AssertClose([plain.Loss], [checkpointed.Loss], 1e-4f, "loss");
        for (int i = 0; i < plain.Grads.Length; i++)
        {
            AssertClose(plain.Grads[i], checkpointed.Grads[i], 1e-3f, $"gradient {i}");
        }

        using var eval = spec.Build(options: new DecoderBuildOptions { Device = device, Seed = 7, InitStd = 0.02f });
        eval.Eval();
        using var x = Tensor.From(ids, [2, 7], device);
        using var first = eval.Forward(x);
        using var second = eval.Forward(x);
        AssertClose(first.ToArray(), second.ToArray(), 0f, "no dropout in evaluation");
    }

    // dotnet run -c Release --project tests/Idrak.Tests -- --bench-gemm
    // TFLOPS of the training products (a 1.25B GPT's shapes) in every layout, with beta 0 and 1, the transposed
    // operands handled by the transposing kernels directly or copied first, and a fresh output buffer each call.
    // Decoding-sized packed products (one row, Qwen3-0.6B shapes, int8 weights) for each forced number of k splits:
    // 100 calls recorded in a graph and replayed, so host launch costs are excluded; GB/s counts the weight bytes.
    internal static int BenchGemv()
    {
        if (!Device.IsCudaAvailable)
        {
            Console.WriteLine("needs a CUDA device");
            return 1;
        }

        var device = Device.Cuda();
        Console.WriteLine(device.Name);
        var random = new Random(2);
        Linear Layer(int k, int n) => Linear.FromInt8(Int8Weight.Quantize([.. Enumerable.Range(0, k * n).Select(_ => random.NextSingle() - 0.5f)], k, n, device));
        using var x1024 = Tensor.From([.. Enumerable.Range(0, 1024).Select(_ => random.NextSingle() - 0.5f)], [1, 1024], device);
        using var x2048 = Tensor.From([.. Enumerable.Range(0, 2048).Select(_ => random.NextSingle() - 0.5f)], [1, 2048], device);
        using var x3072 = Tensor.From([.. Enumerable.Range(0, 3072).Select(_ => random.NextSingle() - 0.5f)], [1, 3072], device);
        using Linear q = Layer(1024, 2048), k = Layer(1024, 1024), v = Layer(1024, 1024), o = Layer(2048, 1024);
        using Linear gate = Layer(1024, 3072), up = Layer(1024, 3072), down = Layer(3072, 1024), head = Layer(1024, 151936);
        var norm = new RMSNorm(1024, device: device);
        var shapes = new (string Name, long Bytes, Action Run)[]
        {
            ("q/k/v 1024 -> 2048+1024+1024", 1024L * 4096, () => Tensor.MatMulPackedMany(x1024, 0, [q, k, v])),
            ("gate/up + act 1024 -> 2x3072", 1024L * 6144, () => Tensor.MatMulPackedGatedPair(x1024, gate, up, 0)),
            ("o + add + norm 2048 -> 1024", 2048L * 1024, () => Tensor.MatMulPackedAddRmsNorm(x2048, o, x1024, norm)),
            ("down + add + norm 3072 -> 1024", 3072L * 1024, () => Tensor.MatMulPackedAddRmsNorm(x3072, down, x1024, norm)),
            ("head 1024 -> 151936", 1024L * 151936, () => x1024.MatMulInt8(head.Int8!)),
        };
        int?[] splits = [null, 1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64];
        Console.WriteLine($"{"product",-32} " + string.Join(" ", splits.Select(s => $"{(s is null ? "auto" : s.ToString()),15}")));
        foreach (var (name, bytes, run) in shapes)
        {
            var line = new System.Text.StringBuilder($"{name,-32} ");
            foreach (var split in splits)
            {
                CudaBackend.GemvSplits = split;
                try
                {
                    using var graph = ComputeGraph.Capture(device, () =>
                    {
                        for (int i = 0; i < 100; i++)
                        {
                            using var calls = new TensorScope();
                            run();
                        }
                    });
                    graph.Replay();
                    device.Synchronize();
                    var watch = System.Diagnostics.Stopwatch.StartNew();
                    for (int r = 0; r < 10; r++)
                    {
                        graph.Replay();
                    }

                    device.Synchronize();
                    double us = watch.Elapsed.TotalMilliseconds * 1000 / 1000;
                    line.Append($"{$"{us:F1}us {bytes / (us * 1e3):F0}GB/s",15} ");
                }
                finally
                {
                    CudaBackend.GemvSplits = null;
                }
            }

            Console.WriteLine(line);
        }

        // Timed like the products above: `run` 100 times per graph; the forced setting is applied by `force`.
        string Timed(Action run, Action<int?> force, int? setting)
        {
            force(setting);
            try
            {
                using var graph = ComputeGraph.Capture(device, () =>
                {
                    for (int i = 0; i < 100; i++)
                    {
                        using var calls = new TensorScope();
                        run();
                    }
                });
                graph.Replay();
                device.Synchronize();
                var watch = System.Diagnostics.Stopwatch.StartNew();
                for (int r = 0; r < 10; r++)
                {
                    graph.Replay();
                }

                device.Synchronize();
                return $"{watch.Elapsed.TotalMilliseconds:F1}us";      // 1000 calls: ms total = µs per call
            }
            finally
            {
                force(null);
            }
        }

        // Decoding attention over a bfloat16 cache (8 key/value heads × 2 query heads, head size 128), forced block splits.
        Console.WriteLine();
        int?[] attentionSplits = [null, 1, 2, 4, 8, 12, 16, 24, 32, 48, 64];
        Console.WriteLine($"{"attention, cached positions",-32} " + string.Join(" ", attentionSplits.Select(s => $"{(s is null ? "auto" : s.ToString()),9}")));
        using var cache = new KeyValueCache(8, 4096, 128, device, KeyValueFormat.BFloat16);
        using var query = Tensor.From([.. Enumerable.Range(0, 8 * 2 * 128).Select(_ => random.NextSingle() - 0.5f)], [8, 2, 128], device);
        foreach (int length in new[] { 200, 1000, 4000 })
        {
            using var position = Tensor.From([length - 1f], [1], device);
            Console.WriteLine($"{length,-32} " + string.Join(" ", attentionSplits.Select(split =>
                $"{Timed(() => Tensor.AttentionBFloat16(query, cache, position, 1, 0.088f, tiled: false), v => CudaBackend.DecodeSplits = v, split),9}")));
        }

        // Prompt-sized products (180 rows) through int8 weights on tensor cores, forced k splits, with 128-row tiles and with
        // 64-row ones (the heuristic's choice for 180 rows).
        foreach (int tileRows in new[] { 128, 64 })
        {
            CudaBackend.PromptTileRowsOverride = tileRows;
            try
            {
                Console.WriteLine();
                int?[] promptSplits = [null, 1, 2, 4, 6, 8, 12, 16];
                Console.WriteLine($"{$"180 rows, int8, {tileRows}-row tiles",-32} " + string.Join(" ", promptSplits.Select(s => $"{(s is null ? "auto" : s.ToString()),9}")));
                using (MixedPrecision.BFloat16())
                {
                    foreach (var (kIn, nOut, layer) in new[] { (1024, 1024, k), (1024, 2048, q), (1024, 3072, gate), (2048, 1024, o), (3072, 1024, down) })
                    {
                        using var x = Tensor.From([.. Enumerable.Range(0, 180 * kIn).Select(_ => random.NextSingle() - 0.5f)], [180, kIn], device);
                        Console.WriteLine($"{$"{kIn} -> {nOut}",-32} " + string.Join(" ", promptSplits.Select(split =>
                            $"{Timed(() => x.MatMulInt8(layer.Int8!), v => CudaBackend.PackedSplits = v, split),9}")));
                    }

                    // The layers sharing an input in one launch, against their separate launches (auto splits).
                    using var prompt = Tensor.From([.. Enumerable.Range(0, 180 * 1024).Select(_ => random.NextSingle() - 0.5f)], [180, 1024], device);
                    foreach (var (name, layers) in new[] { ("q+k+v one launch (1024 -> 4096)", new[] { q, k, v }), ("gate+up one launch (1024 -> 6144)", new[] { gate, up }) })
                    {
                        Console.WriteLine($"{name,-32} " + string.Join(" ", promptSplits.Select(split =>
                            $"{Timed(() => Tensor.MatMulPackedMany(prompt, 0, layers), v => CudaBackend.PackedSplits = v, split),9}")));
                        Console.WriteLine($"{"  same, separate launches",-32} {Timed(() => { foreach (var layer in layers) { prompt.MatMulInt8(layer.Int8!); } }, _ => { }, null),9}");
                    }
                }
            }
            finally
            {
                CudaBackend.PromptTileRowsOverride = null;
            }
        }

        return 0;
    }

    // FP8 operand quantizers for x [k][n] (n columns → n quantized rows): the two-launch column pair (absmax_cols +
    // quant_cols, current), delayed scaling (quant_cols with maxima kept from an earlier pass, recording x's own; then
    // with its correction pass), and the row quantizer of the same tensor for reference. 20 calls per graph;
    // GB/s counts one read of x. Then the error of delayed scaling: x' = x drifted (next step), quantized with x's maxima
    // (stale) against its own (exact), relative RMS after dequantizing, without and with a new outlier.
    internal static int BenchFp8Quantizers()
    {
        if (!Device.IsCudaAvailable)
        {
            Console.WriteLine("needs a CUDA device");
            return 1;
        }

        var device = Device.Cuda();
        var backend = (CudaBackend)device.Backend;
        if (!backend.EightBitReady(fp8: true))
        {
            Console.WriteLine("needs FP8 tensor cores (compute capability 8.9+)");
            return 1;
        }

        Console.WriteLine(device.Name);
        var random = new Random(4);
        double Time(Action run)
        {
            using var graph = ComputeGraph.Capture(device, () =>
            {
                for (int i = 0; i < 20; i++)
                {
                    using var calls = new TensorScope();
                    run();
                }
            });
            graph.Replay();
            device.Synchronize();
            var watch = System.Diagnostics.Stopwatch.StartNew();
            for (int r = 0; r < 5; r++)
            {
                graph.Replay();
            }

            device.Synchronize();
            return watch.Elapsed.TotalMilliseconds * 1000 / 100;          // µs per call
        }

        Console.WriteLine($"{"x [k][n]",-16} {"MB",6} {"current (2 launches)",22} {"D: kept maxima",22} {"D + correction",22} {"rows (reference)",22}");
        foreach (var (k, n) in new[] { (4096, 2048), (4096, 8192), (2048, 8192), (8192, 2048), (2048, 2048), (12288, 768), (12288, 3072) })
        {
            using var x = Tensor.From([.. Enumerable.Range(0, k * n).Select(_ => random.NextSingle() * 2 - 1)], [k * n], device);
            int kp = PtxKernels.EightBitPaddedK(k), np = PtxKernels.EightBitPaddedK(n);
            using var output = Tensor.Zeros([Math.Max(n * kp, k * np) / 4], device);
            using var scale = Tensor.Zeros([Math.Max(n, k)], device);
            using var maxima = Tensor.Ones([n], device);
            using var record = Tensor.Zeros([n], device);
            double mb = 4.0 * k * n / 1e6;
            string Cell(double us) => $"{us,8:F1}us {mb * 1e-3 / (us * 1e-6),7:F0}GB/s";
            double current = Time(() => backend.QuantizeColumnsVariant(true, x.Storage, n, output.Storage, scale.Storage, n, k, 0));
            double delayed = Time(() => backend.QuantizeColumnsVariant(true, x.Storage, n, output.Storage, scale.Storage, n, k, 2, maxima.Storage, record.Storage));
            double corrected = Time(() => backend.QuantizeColumnsVariant(true, x.Storage, n, output.Storage, scale.Storage, n, k, 3, maxima.Storage, record.Storage));
            double rows = Time(() => backend.QuantizeRowsForBenchmark(true, x.Storage, n, output.Storage, scale.Storage, k, n));
            Console.WriteLine($"{$"{k}x{n}",-16} {mb,6:F0} {Cell(current),22} {Cell(delayed),22} {Cell(corrected),22} {Cell(rows),22}");
        }

        // Error of delayed scaling (fp8 e4m3), x [4096][2048] of normal values with a few larger columns.
        Console.WriteLine();
        Console.WriteLine($"{"delayed scaling error",-44} {"exact maxima",14} {"kept maxima",14} {"+ correction",14}");
        const int K = 4096, N = 2048;
        double Normal() => Math.Sqrt(-2 * Math.Log(1 - random.NextDouble())) * Math.Cos(2 * Math.PI * random.NextDouble());
        var previous = new float[K * N];
        for (int i = 0; i < previous.Length; i++)
        {
            previous[i] = (float)(Normal() * (i % N % 97 == 0 ? 8 : 1));
        }

        var previousMaxima = new float[N];
        for (int i = 0; i < previous.Length; i++)
        {
            previousMaxima[i % N] = MathF.Max(previousMaxima[i % N], MathF.Abs(previous[i]));
        }

        foreach (var (name, drift, outlier) in new[] { ("next step: x·1.02 + 1% noise", 1.02f, 0f), ("same, one new outlier ×20 per column", 1.02f, 20f) })
        {
            var current = new float[K * N];
            for (int i = 0; i < current.Length; i++)
            {
                current[i] = previous[i] * drift + (float)(0.01 * Normal());
            }

            if (outlier > 0)
            {
                for (int c = 0; c < N; c++)
                {
                    current[random.Next(K) * N + c] = outlier * previousMaxima[c];
                }
            }

            using var x = Tensor.From(current, [current.Length], device);
            using var kept = Tensor.From(previousMaxima, [N], device);
            using var exactMaxima = Tensor.Zeros([N], device);
            int kp = PtxKernels.EightBitPaddedK(K);
            double Error(int variant)
            {
                using var output = Tensor.Zeros([N * kp / 4], device);
                using var scale = Tensor.Zeros([N], device);
                using var record = Tensor.Zeros([N], device);
                backend.QuantizeColumnsVariant(true, x.Storage, N, output.Storage, scale.Storage, N, K, variant, kept.Storage, record.Storage);

                var bytes = System.Runtime.InteropServices.MemoryMarshal.AsBytes(output.ToArray().AsSpan()).ToArray();
                var scales = scale.ToArray();
                double error = 0, total = 0;
                for (int c = 0; c < N; c++)
                {
                    for (int r = 0; r < K; r++)
                    {
                        double value = current[r * N + c], back = DecodeE4M3(bytes[c * kp + r]) * scales[c];
                        error += (back - value) * (back - value);
                        total += value * value;
                    }
                }

                return Math.Sqrt(error / total);
            }

            Console.WriteLine($"{name,-44} {Error(0),14:P3} {Error(2),14:P3} {Error(3),14:P3}");
        }

        return 0;
    }

    // An FP8 e4m3 byte as a float (bias 7, subnormals, 0x7F / 0xFF NaN).
    private static float DecodeE4M3(byte b)
    {
        int sign = b >> 7, exponent = (b >> 3) & 15, mantissa = b & 7;
        if (exponent == 15 && mantissa == 7)
        {
            return float.NaN;
        }

        float magnitude = exponent == 0 ? mantissa / 8f * MathF.Pow(2, -6) : (1 + mantissa / 8f) * MathF.Pow(2, exponent - 7);
        return sign == 1 ? -magnitude : magnitude;
    }

    internal static int BenchGemm()
    {
        if (!Device.IsCudaAvailable)
        {
            Console.WriteLine("needs a CUDA device");
            return 1;
        }

        var device = Device.Cuda();
        var backend = (CudaBackend)device.Backend;
        Console.WriteLine($"{device.Name}; tensor cores: {MixedPrecision.TensorCoresUnavailable(device) ?? "yes"}");
        var random = new Random(1);
        Tensor Random(int n) => Tensor.From([.. Enumerable.Range(0, n).Select(_ => random.NextSingle() - 0.5f)], [n], device);
        double Time(Action run, int repeats)
        {
            run();
            device.Synchronize();
            var watch = System.Diagnostics.Stopwatch.StartNew();
            for (int i = 0; i < repeats; i++)
            {
                run();
            }

            device.Synchronize();
            return watch.Elapsed.TotalMilliseconds / repeats;
        }

        using var scope = MixedPrecision.BFloat16();
        Console.WriteLine($"{"m x n x k",-20} {"layout",6} {"beta",4} {"direct",14} {"pretransposed",14} {"fresh output",14}");
        foreach (var (m, n, k) in new[] { (4096, 2048, 2048), (4096, 8192, 2048), (4096, 2048, 8192), (2048, 2048, 4096), (2048, 8192, 4096), (8192, 2048, 4096) })
        {
            using var a = Random(m * k);
            using var b = Random(k * n);
            using var c = Random(m * n);
            double flops = 2.0 * m * n * k;
            foreach (var (ta, tb) in new[] { (false, false), (false, true), (true, false) })
            {
                foreach (float beta in new[] { 0f, 1f })
                {
                    string Tflops(double ms) => $"{flops / (ms * 1e9),7:F1} TFLOPS";
                    CudaBackend.PretransposeForTensorCores = false;
                    double direct = Time(() => backend.BatchedMatMul(a.Storage, b.Storage, c.Storage, 1, m, n, k, ta, tb, beta), 10);
                    CudaBackend.PretransposeForTensorCores = true;
                    double copied = ta || tb ? Time(() => backend.BatchedMatMul(a.Storage, b.Storage, c.Storage, 1, m, n, k, ta, tb, beta), 10) : direct;
                    CudaBackend.PretransposeForTensorCores = false;
                    double fresh = Time(() =>
                    {
                        using var output = Tensor.Zeros([m * n], device);
                        backend.BatchedMatMul(a.Storage, b.Storage, output.Storage, 1, m, n, k, ta, tb, beta);
                    }, 10);
                    Console.WriteLine($"{$"{m}x{n}x{k}",-20} {(ta ? "t" : "n") + (tb ? "t" : "n"),6} {beta,4} {Tflops(direct),14} {Tflops(copied),14} {Tflops(fresh),14}");
                }
            }
        }

        // Weight-gradient shapes (small output, long k) with forced k splits (1: none).
        Console.WriteLine();
        int?[] splitCounts = [null, 1, 2, 3, 4, 6, 8];
        Console.WriteLine($"{"m x n x k (tn)",-20} " + string.Join(" ", splitCounts.Select(s => $"{(s is null ? "auto" : $"split {s}"),13}")));
        foreach (var (m, n, k) in new[] { (768, 768, 12288), (3072, 768, 12288), (768, 3072, 12288), (1024, 1024, 8192), (1024, 3072, 8192), (2048, 2048, 4096), (2048, 8192, 4096) })
        {
            using var a = Random(k * m);
            using var b = Random(k * n);
            using var c = Random(m * n);
            var line = new System.Text.StringBuilder($"{$"{m}x{n}x{k}",-20} ");
            foreach (var split in splitCounts)
            {
                CudaBackend.TensorSplitsOverride = split;
                CudaBackend.PretransposeForTensorCores = false;
                double ms = Time(() => backend.BatchedMatMul(a.Storage, b.Storage, c.Storage, 1, m, n, k, true, false, 1f), 10);
                line.Append($"{$"{2.0 * m * n * k / (ms * 1e9):F1} TFLOPS",13} ");
            }

            CudaBackend.TensorSplitsOverride = null;
            Console.WriteLine(line);
        }

        // 8-bit products (quantization of both operands included) against bfloat16.
        Console.WriteLine();
        Console.WriteLine($"{"m x n x k",-20} {"layout",6} {"bf16",14} {"fp8",14} {"int8",14}");
        foreach (var (m, n, k) in new[] { (4096, 2048, 2048), (4096, 8192, 2048), (4096, 2048, 8192), (2048, 8192, 4096), (8192, 8192, 8192) })
        {
            using var a = Random(m * k);
            using var b = Random(k * n);
            using var c = Random(m * n);
            double flops = 2.0 * m * n * k;
            foreach (var (ta, tb) in new[] { (false, false), (false, true), (true, false) })
            {
                string Tflops(double ms) => double.IsNaN(ms) ? $"{"n/a",14}" : $"{flops / (ms * 1e9),7:F1} TFLOPS";
                double bf16 = Time(() => backend.GemmStrided(a.Storage, 0, ta ? m : k, ta, b.Storage, 0, tb ? k : n, tb, c.Storage, 0, n, m, n, k, 0f), 10);
                double fp8 = backend.EightBitReady(fp8: true)
                    ? Time(() => backend.Gemm8(true, P(a), ta ? m : k, ta, P(b), tb ? k : n, tb, P(c), n, m, n, k, 0f, 0UL, Idrak.Backends.GemmEpilogue.None, 0UL), 10)
                    : double.NaN;
                double int8 = backend.EightBitReady(fp8: false)
                    ? Time(() => backend.Gemm8(false, P(a), ta ? m : k, ta, P(b), tb ? k : n, tb, P(c), n, m, n, k, 0f, 0UL, Idrak.Backends.GemmEpilogue.None, 0UL), 10)
                    : double.NaN;
                Console.WriteLine($"{$"{m}x{n}x{k}",-20} {(ta ? "t" : "n") + (tb ? "t" : "n"),6} {Tflops(bf16),14} {Tflops(fp8),14} {Tflops(int8),14}");
            }
        }

        Console.WriteLine();
        foreach (var (rows, cols) in new[] { (2048, 2048), (2048, 8192), (4096, 2048), (4096, 8192) })
        {
            using var x = Random(rows * cols);
            using var y = Tensor.Zeros([rows * cols], device);
            double ms = Time(() => backend.TransposeForBenchmark(x.Storage, y.Storage, rows, cols), 20);
            Console.WriteLine($"transpose {rows}x{cols}: {ms:F3} ms, {2.0 * rows * cols * 4 / (ms * 1e6):F0} GB/s");
        }

        return 0;
    }

    private static ulong P(Tensor t) => ((Idrak.Backends.Cuda.CudaStorage)t.Storage).Pointer;

    private static float RoundBFloat16(float x)
    {
        uint bits = BitConverter.SingleToUInt32Bits(x);
        bits += 0x7FFFu + ((bits >> 16) & 1u);
        return BitConverter.UInt32BitsToSingle(bits & 0xFFFF0000u);
    }

    private static void TensorCoreProducts(Device device)
    {
        Check(PtxKernels.TensorCoreNames.Where(k => k.StartsWith("gemm_tc")).All(k => PtxKernels.TensorCoreParameterCounts.TryGetValue(k, out int n) && n == (k.Contains("_multi") ? 23 : 15) + (k.Contains("_lr_") ? (k.Contains("_multi") ? 7 : 3) : 0))
              && PtxKernels.TensorCoreNames.All(PtxKernels.TensorCoreParameterCounts.ContainsKey), "tensor-core kernel signatures");
        var random = new Random(3);
        // A GPU that has tensor cores must load the module: a JIT error would otherwise fall back to float32 silently.
        string? unavailable = MixedPrecision.TensorCoresUnavailable(device);
        Check(device.Type == DeviceType.Cpu || unavailable is null || unavailable.StartsWith("compute capability", StringComparison.Ordinal),
            $"tensor-core module: {unavailable}");
        bool tensorCores = device.Type == DeviceType.Cuda && unavailable is null;
        if (device.Type == DeviceType.Cuda && !(unavailable?.StartsWith("compute capability", StringComparison.Ordinal) ?? false))
        {
            var errors = ((CudaBackend)device.Backend).TensorCoreModuleErrors;
            Check(errors.Count == 0, $"tensor-core modules that did not load: {string.Join(" | ", errors)}");
        }
        long launchesBefore = device.Type == DeviceType.Cuda ? ((CudaBackend)device.Backend).TensorCoreLaunches : 0;
        bool anyDifferent = false;
        // 130×150×4200: few output tiles and a long k, so k is split over blocks (beta 0 and 1; an uneven last chunk).
        foreach (var (m, n, k, batch) in new[] { (64, 64, 32, 1), (200, 130, 71, 2), (257, 300, 129, 1), (128, 256, 512, 1), (130, 150, 4200, 1) })
        {
            foreach (bool ta in new[] { false, true })
            {
                foreach (bool tb in new[] { false, true })
                {
                    foreach (float beta in new[] { 0f, 0.5f, 1f })
                    {
                        float[] a = [.. Enumerable.Range(0, batch * m * k).Select(_ => random.NextSingle() * 2 - 1)];
                        float[] b = [.. Enumerable.Range(0, batch * k * n).Select(_ => random.NextSingle() * 2 - 1)];
                        float[] c = [.. Enumerable.Range(0, batch * m * n).Select(_ => random.NextSingle() * 2 - 1)];
                        using var ta_ = Tensor.From(a, [a.Length], device);
                        using var tb_ = Tensor.From(b, [b.Length], device);
                        using var tc_ = Tensor.From(c, [c.Length], device);
                        using (MixedPrecision.BFloat16())
                        {
                            device.Backend.BatchedMatMul(ta_.Storage, tb_.Storage, tc_.Storage, batch, m, n, k, ta, tb, beta);
                        }

                        var got = tc_.ToArray();
                        var rounded = new double[m * n * batch];
                        var exact = new double[m * n * batch];
                        for (int s = 0; s < batch; s++)
                        {
                            for (int i = 0; i < m; i++)
                            {
                                for (int j = 0; j < n; j++)
                                {
                                    double sum = 0, sumExact = 0;
                                    for (int q = 0; q < k; q++)
                                    {
                                        float x = a[s * m * k + (ta ? q * m + i : i * k + q)], y = b[s * k * n + (tb ? j * k + q : q * n + j)];
                                        sum += (double)RoundBFloat16(x) * RoundBFloat16(y);
                                        sumExact += (double)x * y;
                                    }

                                    int o = s * m * n + i * n + j;
                                    rounded[o] = sum + beta * c[o];
                                    exact[o] = sumExact + beta * c[o];
                                }
                            }
                        }

                        string what = $"{m}×{n}×{k} batch {batch}, {(ta ? "t" : "n")}{(tb ? "t" : "n")}, beta {beta}";
                        var reference = tensorCores ? rounded : exact;
                        AssertClose([.. reference.Select(v => (float)v)], got, 1e-3f, what);
                        anyDifferent |= got.Zip(exact).Any(p => Math.Abs(p.First - p.Second) > 1e-3 * Math.Max(1, Math.Abs(p.Second)));
                    }
                }
            }
        }

        if (tensorCores)
        {
            Check(((CudaBackend)device.Backend).TensorCoreLaunches > launchesBefore && anyDifferent, "the products ran on the tensor cores in bfloat16");
        }
        else if (device.Type == DeviceType.Cuda)
        {
            Console.WriteLine($"    (no tensor cores: {((CudaBackend)device.Backend).TensorCoreUnavailableReason}; float32 checked)");
        }
    }
}
