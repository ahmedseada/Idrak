// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Idrak;
using Idrak.Backends;
using Idrak.Backends.Vulkan;
using Idrak.Layers;

// Sliding windows and soft-capped scores in the attention kernels (AttentionVariant): every kernel against a direct
// reference written out here, the Vulkan kernels against the CPU at every width with no host fallback, and decoders
// with windows and caps through the kernels against the composed path (basic operations over every cached slot, the
// path these layers took before the kernels had a window, still forced with IDRAK_WINDOW_KERNELS=0).
internal static partial class Tests
{
    private static readonly (string Name, Action<Device> Run)[] WindowKernelGroup =
    [
        ("window kernels: tiled, decoding (float32, int8, bfloat16 caches, split positions), per-row and packed attention with windows inside and beyond the keys and soft-caps match a direct reference; a window beyond the keys is plain attention bit for bit", WindowedAttentionKernels),
        ("window kernels: gradients with windows and soft-caps (tiled and packed) match finite differences and the CPU", WindowedAttentionGradients),
        ("window kernels: vulkan decoding, tiled and gradient kernels with windows and soft-caps match the CPU as kernels at every width and split count", VulkanWindowedKernels),
        ("window kernels: Mistral, Gemma 2 and Gemma 3 style decoders give the composed path's logits through the kernels (prompt, training and gradients, cached decoding in every cache format, recorded steps, packed sequences, rows of different lengths)", WindowedDecodersMatchComposed),
    ];

    // The variants checked: windows inside the keys, one beyond them, a cap alone, both.
    private static readonly AttentionVariant[] WindowVariants = [new(3, 0f), new(17, 0f), new(0, 1.5f), new(4, 2f), new(1, 0f), new(1000, 0f)];

    // Attention written out (double): row i of head h sees positions [max(begin, end - window), end) of keys and values
    // [heads, capacity, dim], the scaled scores capped. range(h, i) gives (begin, end); writes the log-sum-exp when given.
    private static float[] DirectAttention(float[] q, float[] k, float[] v, int heads, int rowsPerHead, int capacity, int dim, float scale,
        AttentionVariant variant, Func<int, int, (int Begin, int End)> range, float[]? logSumExp = null)
    {
        var y = new float[heads * rowsPerHead * dim];
        for (int h = 0; h < heads; h++)
        {
            for (int i = 0; i < rowsPerHead; i++)
            {
                var (begin, end) = range(h, i);
                if (variant.Window > 0)
                {
                    begin = Math.Max(begin, end - variant.Window);
                }

                if (end <= begin)
                {
                    continue;                                                    // sees nothing: zeros
                }

                var scores = new double[end - begin];
                for (int c = begin; c < end; c++)
                {
                    double dot = 0;
                    for (int d = 0; d < dim; d++)
                    {
                        dot += (double)q[(h * rowsPerHead + i) * dim + d] * k[(h * capacity + c) * dim + d];
                    }

                    double s = dot * scale;
                    scores[c - begin] = variant.Softcap > 0 ? variant.Softcap * Math.Tanh(s / variant.Softcap) : s;
                }

                double max = scores.Max(), sum = scores.Sum(x => Math.Exp(x - max));
                if (logSumExp is not null)
                {
                    logSumExp[h * rowsPerHead + i] = (float)(max + Math.Log(sum));
                }

                for (int d = 0; d < dim; d++)
                {
                    double value = 0;
                    for (int c = begin; c < end; c++)
                    {
                        value += Math.Exp(scores[c - begin] - max) / sum * v[(h * capacity + c) * dim + d];
                    }

                    y[(h * rowsPerHead + i) * dim + d] = (float)value;
                }
            }
        }

        return y;
    }

    private static void WindowedAttentionKernels(Device device)
    {
        var r = new Random(71);
        float[] R(int n) => [.. Enumerable.Range(0, n).Select(_ => (float)(r.NextDouble() * 2 - 1))];
        float Bf(float x) => BitConverter.Int32BitsToSingle(BFloat16Weight.Round(x) << 16);

        // (heads, rows per head, steps, capacity, head size, offset): prompts (tiled), decoding steps with grouped rows, large
        // caches split over positions (windows narrower than a split, splits before the window empty), head size 6.
        foreach (var (heads, rowsPerHead, steps, capacity, dim, offset) in new[] { (2, 40, 20, 64, 64, 10), (3, 18, 9, 40, 33, 0), (2, 2, 1, 700, 128, 650),
            (4, 1, 1, 1000, 64, 400), (1, 4, 2, 300, 40, 150), (2, 12, 6, 30, 6, 4) })
        {
            float scale = 1f / MathF.Sqrt(dim);
            float[] q = R(heads * rowsPerHead * dim), k = R(heads * capacity * dim), v = R(heads * capacity * dim);
            using var tq = Tensor.From(q, [heads, rowsPerHead, dim], device);
            using var tk = Tensor.From(k, [heads, capacity, dim], device);
            using var tv = Tensor.From(v, [heads, capacity, dim], device);
            using var position = Tensor.From([(float)offset], [1], device);
            using var zero = Tensor.From([0f], [1], device);
            (int, int) Causal(int h, int i) => (0, Math.Min(offset + i % steps, capacity - 1) + 1);

            using var cache = new KeyValueCache(heads, capacity, dim, device, KeyValueFormat.Float32);
            cache.Keys.Load(k);
            cache.Values.Load(v);
            using var cache8 = new KeyValueCache(heads, capacity, dim, device, KeyValueFormat.Int8);
            Tensor.WriteKeyValuesInt8(tk, cache8.Keys, cache8.KeyScales!, zero);
            Tensor.WriteKeyValuesInt8(tv, cache8.Values, cache8.ValueScales!, zero);
            using var cache16 = new KeyValueCache(heads, capacity, dim, device, KeyValueFormat.BFloat16);
            Tensor.WriteKeyValuesBFloat16(tk, cache16.Keys, zero, dim);
            Tensor.WriteKeyValuesBFloat16(tv, cache16.Values, zero, dim);
            float[] k8, v8;
            using (var ek = cache8.Layout.Expand(cache8, keys: true))
            using (var ev = cache8.Layout.Expand(cache8, keys: false))
            {
                (k8, v8) = (ek.ToArray(), ev.ToArray());                         // the int8 cache's values as the kernels read them
            }

            float[] k16 = [.. k.Select(Bf)], v16 = [.. v.Select(Bf)];
            foreach (var variant in WindowVariants)
            {
                string name = $"{heads}×{rowsPerHead} rows, steps {steps}, capacity {capacity}, dim {dim}, offset {offset}, window {variant.Window}, cap {variant.Softcap}";
                var expected = DirectAttention(q, k, v, heads, rowsPerHead, capacity, dim, scale, variant, Causal);
                AssertClose(expected, Tensor.AttentionTiled(tq, tk, tv, position, steps, scale, variant).ToArray(), 2e-4f, "tiled: " + name);
                AssertClose(expected, Tensor.AttentionDecode(tq, cache, position, steps, scale, variant).ToArray(), 2e-4f, "decoding: " + name);
                var expected8 = DirectAttention(q, k8, v8, heads, rowsPerHead, capacity, dim, scale, variant, Causal);
                AssertClose(expected8, Tensor.AttentionInt8(tq, cache8, position, steps, scale, tiled: false, variant).ToArray(), 2e-4f, "int8 decoding: " + name);
                AssertClose(expected8, Tensor.AttentionInt8(tq, cache8, position, steps, scale, tiled: true, variant).ToArray(), 2e-4f, "int8 tiled: " + name);
                var expected16 = DirectAttention(q, k16, v16, heads, rowsPerHead, capacity, dim, scale, variant, Causal);
                AssertClose(expected16, Tensor.AttentionBFloat16(tq, cache16, position, steps, scale, tiled: false, variant).ToArray(), 2e-4f, "bf16 decoding: " + name);
                AssertClose(expected16, Tensor.AttentionBFloat16(tq, cache16, position, steps, scale, tiled: true, variant).ToArray(), 2e-4f, "bf16 tiled: " + name);
            }

            // A window covering every key (and no cap) runs the plain kernels' arithmetic: the same bits.
            var beyond = new AttentionVariant(capacity + 5, 0f);
            Check(Tensor.AttentionTiled(tq, tk, tv, position, steps, scale, beyond).ToArray().SequenceEqual(Tensor.AttentionTiled(tq, tk, tv, position, steps, scale).ToArray()),
                $"tiled, capacity {capacity}: a window beyond the keys changes no bit");
            Check(Tensor.AttentionDecode(tq, cache, position, steps, scale, beyond).ToArray().SequenceEqual(Tensor.AttentionDecode(tq, cache, position, steps, scale).ToArray()),
                $"decoding, capacity {capacity}: a window beyond the keys changes no bit");
            Check(Tensor.AttentionInt8(tq, cache8, position, steps, scale, false, beyond).ToArray().SequenceEqual(Tensor.AttentionInt8(tq, cache8, position, steps, scale, false).ToArray()),
                $"int8, capacity {capacity}: a window beyond the keys changes no bit");
        }

        // Rows of different lengths (per-row starts) and packed sequences: where the device has these passes for a window.
        {
            const int Rows = 3, Kv = 2, Group = 2, Steps = 7, Capacity = 24, Dim = 16, Offset = 9;
            int heads = Rows * Kv, rowsPerHead = Group * Steps;
            float scale = 0.25f;
            float[] q = R(heads * rowsPerHead * Dim), k = R(heads * Capacity * Dim), v = R(heads * Capacity * Dim);
            int[] rowStarts = [0, 5, 13];
            float[] starts = [.. Enumerable.Range(0, Rows * Steps).Select(i => (float)rowStarts[i / Steps])];
            using var tq = Tensor.From(q, [heads, rowsPerHead, Dim], device);
            using var tk = Tensor.From(k, [heads, Capacity, Dim], device);
            using var tv = Tensor.From(v, [heads, Capacity, Dim], device);
            using var position = Tensor.From([(float)Offset], [1], device);
            using var tStarts = Tensor.From(starts, [Rows * Steps], device);
            foreach (var variant in WindowVariants.Prepend(default))
            {
                var y = Tensor.AttentionRows(tq, tk, tv, position, Steps, scale, tStarts, Kv, variant);
                bool supported = device.Backend.SupportsSegmentedAttention(Dim, variant);
                if (y is null)
                {
                    Check(!supported, $"per-row attention, window {variant.Window}, cap {variant.Softcap}: refused although supported");
                    continue;
                }

                var expected = DirectAttention(q, k, v, heads, rowsPerHead, Capacity, Dim, scale, variant,
                    (h, i) => (rowStarts[h / Kv], Math.Min(Offset + i % Steps, Capacity - 1) + 1));
                AssertClose(expected, y.ToArray(), 2e-4f, $"per-row attention, window {variant.Window}, cap {variant.Softcap}");
                y.Dispose();
            }

            int[][] lengths = [[3, 9, 4], [16], [1, 6, 2]];
            const int Length = 16;
            using var packing = PackedSequences.Create(lengths, Length, device);
            var packedStarts = packing.Starts.ToArray();
            int packedRows = Kv * Group * Length;
            float[] pq = R(heads * packedRows * Dim), pk = R(heads * Length * Dim), pv = R(heads * Length * Dim);
            using var tpq = Tensor.From(pq, [heads, packedRows, Dim], device);
            using var tpk = Tensor.From(pk, [heads, Length, Dim], device);
            using var tpv = Tensor.From(pv, [heads, Length, Dim], device);
            foreach (var variant in WindowVariants.Prepend(default))
            {
                using (Autograd.NoGrad())
                {
                    var y = Tensor.CausalAttentionSegmented(tpq, tpk, tpv, packing, Kv, scale, variant);
                    bool supported = device.Backend.SupportsSegmentedAttention(Dim, variant);
                    if (y is null)
                    {
                        Check(!supported, $"packed attention, window {variant.Window}, cap {variant.Softcap}: refused although supported");
                        continue;
                    }

                    var expected = DirectAttention(pq, pk, pv, heads, packedRows, Length, Dim, scale, variant,
                        (h, i) => ((int)packedStarts[h / Kv * Length + i % Length], i % Length + 1));
                    AssertClose(expected, y.ToArray(), 2e-4f, $"packed attention, window {variant.Window}, cap {variant.Softcap}");
                    y.Dispose();
                }
            }
        }
    }

    private static void WindowedAttentionGradients(Device device)
    {
        const int Heads = 2, Steps = 6, Rows = 2 * Steps, Dim = 6;
        var r = new Random(72);
        float[] Random(int n) => [.. Enumerable.Range(0, n).Select(_ => (float)(r.NextDouble() * 2 - 1))];
        using var q = Tensor.From(Random(Heads * Rows * Dim), [Heads, Rows, Dim], device);
        using var k = Tensor.From(Random(Heads * Steps * Dim), [Heads, Steps, Dim], device);
        using var v = Tensor.From(Random(Heads * Steps * Dim), [Heads, Steps, Dim], device);
        using var weights = Tensor.From(Random(Heads * Rows * Dim), [Heads, Rows, Dim], device);
        using var zero = Tensor.From([0f], [1], device);
        const float Scale = 0.9f;
        foreach (var variant in new AttentionVariant[] { new(2, 0f), new(0, 0.8f), new(3, 1.2f) })
        {
            GradCheck(device, [Heads, Rows, Dim], x => (Tensor.CausalAttention(x, k, v, zero, Steps, Scale, variant) * weights).Sum(), scale: 2f);
            GradCheck(device, [Heads, Steps, Dim], x => (Tensor.CausalAttention(q, x, v, zero, Steps, Scale, variant) * weights).Sum(), scale: 2f);
            GradCheck(device, [Heads, Steps, Dim], x => (Tensor.CausalAttention(q, k, x, zero, Steps, Scale, variant) * weights).Sum(), scale: 2f);
        }

        // Several tiles, head size not a multiple of 32: the device's gradients equal the CPU's; packed sequences too (where
        // the device has them for a window).
        {
            const int H = 2, T = 70, R = 2 * T, E = 40;
            float[] qs = Random(H * R * E), ks = Random(H * T * E), vs = Random(H * T * E), ws = Random(H * R * E);
            int[][] lengths = [[30, 25, 15], [70]];
            foreach (var variant in new AttentionVariant[] { new(9, 0f), new(40, 3f), new(0, 0.5f) })
            {
                (float[] Q, float[] K, float[] V)? Gradients(Device on, bool packed)
                {
                    using var tq = Tensor.From(qs, [H, R, E], on, requiresGrad: true);
                    using var tk = Tensor.From(ks, [H, T, E], on, requiresGrad: true);
                    using var tv = Tensor.From(vs, [H, T, E], on, requiresGrad: true);
                    using var tw = Tensor.From(ws, [H, R, E], on);
                    using var tz = Tensor.From([0f], [1], on);
                    using var packing = packed ? PackedSequences.Create(lengths, T, on) : null;
                    var y = packed ? Tensor.CausalAttentionSegmented(tq, tk, tv, packing!, 1, 0.3f, variant) : Tensor.CausalAttention(tq, tk, tv, tz, T, 0.3f, variant);
                    if (y is null)
                    {
                        return null;
                    }

                    (y * tw).Sum().Backward();
                    return (tq.Grad!.ToArray(), tk.Grad!.ToArray(), tv.Grad!.ToArray());
                }

                foreach (bool packed in new[] { false, true })
                {
                    string what = $"{(packed ? "packed" : "tiled")}, window {variant.Window}, cap {variant.Softcap}";
                    var cpu = Gradients(Device.Cpu, packed)!.Value;
                    var mine = Gradients(device, packed);
                    if (mine is null)
                    {
                        Check(packed && !device.Backend.SupportsSegmentedAttention(E, variant), $"{what}: no gradient");
                        continue;
                    }

                    var (dq, dk, dv) = mine.Value;

                    AssertClose(cpu.Q, dq, 2e-4f, $"query gradient, {what}");
                    AssertClose(cpu.K, dk, 2e-4f, $"key gradient, {what}");
                    AssertClose(cpu.V, dv, 2e-4f, $"value gradient, {what}");
                }
            }
        }
    }

    private static void VulkanWindowedKernels(Device device) => ForEachPromptBackend(device, (backend, label, full) =>
    {
        var random = new Random(73);
        float[] R(int n) => [.. Enumerable.Range(0, n).Select(_ => 2 * random.NextSingle() - 1)];
        float[] Bits(int n) => [.. Enumerable.Range(0, n).Select(_ => BitConverter.Int32BitsToSingle(random.Next() ^ (random.Next() << 16)))];
        float[] BFloat16Pairs(int words) => [.. Enumerable.Range(0, words).Select(_ =>
        {
            uint low = (uint)BitConverter.SingleToInt32Bits(2 * random.NextSingle() - 1) >> 16;
            uint high = (uint)BitConverter.SingleToInt32Bits(2 * random.NextSingle() - 1) & 0xFFFF0000u;
            return BitConverter.Int32BitsToSingle((int)(low | high));
        })];
        float[] Scales(int n) => [.. Enumerable.Range(0, n).Select(_ => 0.002f + 0.01f * random.NextSingle())];

        // (head size, key/value heads, groups, steps, capacity, position): decoding steps, short prompts, a prompt longer than
        // a block, odd sizes, a window inside and one beyond the keys.
        var shapes = full
            ? new[] { (64, 2, 2, 1, 300, 250), (33, 3, 1, 3, 80, 40), (128, 1, 4, 20, 64, 12), (7, 2, 3, 40, 100, 50), (256, 1, 1, 2, 40, 30) }
            : [(64, 2, 2, 1, 300, 250), (33, 3, 1, 3, 80, 40)];
        AttentionVariant[] variants = full ? [new(5, 0f), new(0, 1.2f), new(17, 2.5f), new(1000, 0f)] : [new(5, 0f), new(17, 2.5f)];
        try
        {
            foreach (var (dim, heads, groups, steps, capacity, position) in shapes)
            {
                int rowsPerHead = groups * steps, rows = heads * rowsPerHead, slots = heads * capacity;
                float scale = 1f / MathF.Sqrt(dim);
                float[] q = R(rows * dim), keys = R(slots * dim), values = R(slots * dim);
                float[] k16 = BFloat16Pairs(slots * ((dim + 1) / 2)), v16 = BFloat16Pairs(slots * ((dim + 1) / 2));
                float[] k8 = Bits(slots * ((dim + 3) / 4)), v8 = Bits(slots * ((dim + 3) / 4)), ks = Scales(slots), vs = Scales(slots);
                foreach (var variant in variants)
                {
                    string shape = $"{label}: head size {dim}, {heads} heads × {groups} groups, {steps} steps, capacity {capacity}, position {position}, window {variant.Window}, cap {variant.Softcap}";
                    foreach (int? splits in full ? new int?[] { 1, 3, null } : [null])
                    {
                        VulkanBackend.AttentionSplits = splits;
                        string split = $"{shape}, {(splits is null ? "measured" : $"{splits} splits")}";
                        PromptCase(backend, $"decoding {split}", [q, keys, values, [position], new float[rows * dim]], [4],
                            (b, s) => b.AttentionDecode(s[0], s[1], s[2], s[3], s[4], heads, rowsPerHead, steps, capacity, dim, scale, variant), "attention_decode");
                        PromptCase(backend, $"bfloat16 decoding {split}", [q, k16, v16, [position], new float[rows * dim]], [4],
                            (b, s) => b.AttentionBFloat16(s[0], s[1], s[2], s[3], s[4], heads, rowsPerHead, steps, capacity, dim, scale, false, variant), "attention_bf16");
                        PromptCase(backend, $"int8 decoding {split}", [q, k8, v8, ks, vs, [position], new float[rows * dim]], [6],
                            (b, s) => b.AttentionInt8(s[0], s[1], s[2], s[3], s[4], s[5], s[6], heads, rowsPerHead, steps, capacity, dim, scale, false, variant), "attention_int8");
                    }

                    VulkanBackend.AttentionSplits = null;
                    VulkanBackend.TiledAttentionKernel = true;
                    float[][] inputs = [q, keys, values, [position], new float[rows * dim], new float[rows]];
                    PromptCase(backend, $"tiled {shape}, with the log-sum-exp", inputs, [4, 5],
                        (b, s) => b.AttentionTiled(s[0], s[1], s[2], s[3], s[4], s[5], heads, rowsPerHead, steps, capacity, dim, scale, variant), "attention_tiled_lse");
                    PromptCase(backend, $"tiled {shape}", inputs, [4],
                        (b, s) => b.AttentionTiled(s[0], s[1], s[2], s[3], s[4], null, heads, rowsPerHead, steps, capacity, dim, scale, variant), "attention_tiled");
                    VulkanBackend.TiledAttentionKernel = null;

                    // Gradients (causal offset 0, keys and values [heads, steps, dim]): dq, dk, dv added to what they hold.
                    if (dim <= VulkanKernels.AttentionMaxDim)
                    {
                        int keyRows = heads * steps;
                        float[] bk = R(keyRows * dim), bv = R(keyRows * dim), output = new float[rows * dim], lse = new float[rows];
                        RunCpu(Cpu(), [q, bk, bv, [0f], output, lse], s => Cpu().AttentionTiled(s[0], s[1], s[2], s[3], s[4], s[5],
                            heads, rowsPerHead, steps, steps, dim, scale, variant), [(4, output), (5, lse)]);
                        PromptCase(backend, $"gradient {shape}", [q, bk, bv, output, lse, R(rows * dim), R(rows * dim), R(keyRows * dim), R(keyRows * dim)], [6, 7, 8],
                            (b, s) => b.AttentionTiledBackward(s[0], s[1], s[2], s[3], s[4], s[5], s[6], s[7], s[8], heads, rowsPerHead, steps, steps, dim, scale, variant),
                            "attention_backward_dq");
                    }
                }

                // A window beyond the keys: the plain kernels' bits.
                float[] Plain(AttentionVariant variant) => RunOn(backend, [q, keys, values, [position], new float[rows * dim]], 4,
                    s => backend.AttentionDecode(s[0], s[1], s[2], s[3], s[4], heads, rowsPerHead, steps, capacity, dim, scale, variant));
                Check(Plain(new(capacity + 1, 0f)).SequenceEqual(Plain(default)), $"{label}, head size {dim}: a window beyond the keys changes no bit");
            }
        }
        finally
        {
            VulkanBackend.AttentionSplits = null;
            VulkanBackend.TiledAttentionKernel = null;
        }

        static Backend Cpu() => Idrak.Backends.Cpu.CpuBackend.Instance;

        // Runs op on the CPU and copies outputs (index, array) back into the given arrays.
        static void RunCpu(Backend b, float[][] inputs, Action<Storage[]> op, (int Index, float[] Into)[] outputs)
        {
            var storages = inputs.Select(d => { var s = b.Allocate(d.Length, false); b.Upload(d, s); return s; }).ToArray();
            try
            {
                op(storages);
                foreach (var (index, into) in outputs)
                {
                    b.Download(storages[index], into);
                }
            }
            finally
            {
                foreach (var s in storages)
                {
                    s.Release();
                }
            }
        }

        static float[] RunOn(Backend b, float[][] inputs, int output, Action<Storage[]> op)
        {
            var result = new float[inputs[output].Length];
            RunCpu(b, inputs, op, [(output, result)]);
            return result;
        }
    });

    private static void WindowedDecodersMatchComposed(Device device)
    {
        int[] ids = [1, 4, 9, 16, 2, 7, 11, 3, 8, 20, 5, 13, 6, 0, 17, 10, 21, 12];
        int count = ids.Length, V = SmallSpec.Vocabulary;
        var spec0 = SmallSpec with { MaxPositions = 40 };
        var variants = new (string Name, DecoderSpec Spec)[]
        {
            ("Mistral style: window 3 in every layer", spec0 with { SlidingWindow = 3 }),
            ("Gemma 2 style: alternating window, soft-capped scores and logits, score scale, post-norms", spec0 with
            {
                SlidingWindow = 3, SlidingWindowLayers = [true, false], AttentionSoftcap = 1.5f, LogitSoftcap = 2f, AttentionScale = 0.3f,
                PostNorms = true, NormOffset = 1f, Activation = FeedForwardActivation.Gelu, TieEmbeddings = true, EmbeddingScale = 4f,
            }),
            ("Gemma 3 style: local rotary base in the windowed layer, q/k norm, scaled global rope", spec0 with
            {
                SlidingWindow = 4, SlidingWindowLayers = [true, false], SlidingWindowRope = new RopeSettings(50f), QkNorm = true,
                Rope = new RopeSettings(5000f, Scaling: RopeScaling.Linear(2)), AttentionScale = 0.5f, PostNorms = true,
            }),
            ("soft-capped scores only", spec0 with { AttentionSoftcap = 1f }),
            ("window 9 with a wider head", spec0 with { SlidingWindow = 9, HeadDim = 32, Dim = 32 }),
        };

        bool saved = CausalSelfAttention.WindowKernels;
        try
        {
            using var sequence = Tensor.From([.. ids.Select(i => (float)i)], [1, count], device);
            foreach (var (name, spec) in variants)
            {
                using var model = spec.Build(new RandomWeights(85), new DecoderBuildOptions { Device = device });
                T Both<T>(Func<T> run, out T composed)
                {
                    CausalSelfAttention.WindowKernels = false;
                    composed = run();
                    CausalSelfAttention.WindowKernels = true;
                    return run();
                }

                // The prompt, and the training path with every parameter's gradient.
                var kernels = Both(() => model.Predict(sequence).ToArray(), out var composed);
                AssertClose(composed, kernels, 1e-4f, $"{name}: prompt");
                (float[] Logits, float[] Gradients) Train()
                {
                    model.Train();
                    foreach (var p in model.Parameters())
                    {
                        p.ZeroGrad();
                    }

                    using var scope = new TensorScope();
                    var output = model.Forward(sequence);
                    (output * output).Sum().Backward();
                    var result = (output.ToArray(), model.Parameters().SelectMany(p => p.Grad?.ToArray() ?? []).ToArray());
                    model.Eval();
                    return result;
                }

                var trained = Both(Train, out var trainedComposed);
                AssertClose(trainedComposed.Logits, trained.Logits, 1e-4f, $"{name}: training forward");
                AssertClose(trainedComposed.Gradients, trained.Gradients, 2e-3f, $"{name}: gradients");

                // Cached decoding: a prompt (2, 5 or 9 positions; 9 takes the tiled kernels), then one position at a time.
                foreach (var format in new[] { KeyValueFormat.Float32, KeyValueFormat.Int8, KeyValueFormat.BFloat16 })
                {
                    foreach (int prompt in new[] { 2, 5, 9 })
                    {
                        List<float[]> Decode()
                        {
                            var steps = new List<float[]>();
                            using var context = new DecodingContext(device, 1, 32, format);
                            using (Autograd.NoGrad())
                            {
                                using var first = Tensor.From([.. ids.Take(prompt).Select(i => (float)i)], [1, prompt], device);
                                steps.Add(model.ForwardCached(first, context).ToArray());
                                for (int t = prompt; t < count; t++)
                                {
                                    using var next = Tensor.From([(float)ids[t]], [1, 1], device);
                                    steps.Add(model.ForwardCached(next, context).ToArray());
                                }
                            }

                            return steps;
                        }

                        var decoded = Both(Decode, out var decodedComposed);
                        float tolerance = format == KeyValueFormat.Float32 ? 1e-4f : 1e-3f;
                        for (int s = 0; s < decoded.Count; s++)
                        {
                            AssertClose(decodedComposed[s], decoded[s], tolerance, $"{name}, {format}: prompt {prompt}, step {s}");
                        }
                    }
                }

                // A recorded step replayed past the window samples what direct steps sample (recorded before the window masks).
                int[][] Sampled(bool useGraph)
                {
                    using var context = new DecodingContext(device, 1, 24);
                    using var sampler = new TokenSampler(device, 1, V, 12) { Temperature = 0.9f, Seed = 11 };
                    using (Autograd.NoGrad())
                    {
                        using (var scope = new TensorScope())
                        {
                            sampler.Sample(model.ForwardCached(Tensor.From([1f, 2f], [1, 2], device), context));
                        }

                        void Step() => sampler.Sample(model.ForwardCached(sampler.Ids.Reshape(1, 1), context));
                        using var graph = useGraph ? context.CaptureStep(Step) : null;
                        for (int s = 0; s < 11; s++)
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
                    }

                    return [.. sampler.Read(0, 12).Select(step => step.Select(t => t.Id).ToArray())];
                }

                var direct = Sampled(useGraph: false);
                var replayed = Sampled(useGraph: true);
                Check(direct.Zip(replayed).All(p => p.First.SequenceEqual(p.Second)), $"{name}: replayed steps sample what direct steps do");

                // Packed sequences: each sequence's logits as when it runs alone (where the device packs these layers).
                bool packs = PackedSequences.Supports(model);
                bool shouldPack = model.Descendants().OfType<CausalSelfAttention>().All(a => device.Backend.SupportsSegmentedAttention(a.HeadDim, a.Variant));
                Check(packs == shouldPack, $"{name}: packing offered {packs}, the device's packed attention says {shouldPack}");
                if (packs)
                {
                    int[][] lengths = [[7, 11], [18]];
                    using var packing = PackedSequences.Create(lengths, count, device);
                    float[] packedLogits;
                    model.Train();
                    using (var scope = new TensorScope())
                    using (packing.Use())
                    {
                        var input = ids.Take(7).Concat(ids.Take(11)).Concat(ids).Select(i => (float)i).ToArray();
                        packedLogits = model.Forward(Tensor.From(input, [2, count], device)).ToArray();
                    }

                    model.Eval();
                    int offset = 0;
                    foreach (int n in new[] { 7, 11, 18 })
                    {
                        using var alone = Tensor.From([.. ids.Take(n).Select(i => (float)i)], [1, n], device);
                        AssertClose(model.Predict(alone).ToArray(), packedLogits[(offset * V)..((offset + n) * V)], 1e-4f, $"{name}: packed sequence of {n}");
                        offset += n;
                    }
                }

                // Rows of different lengths decoded together: each row as when it runs alone.
                if (model.Descendants().OfType<CausalSelfAttention>().All(a => a.SupportsSegmented(device.Backend)))
                {
                    int[] lengthsOfRows = [5, 12, 9];
                    int longest = lengthsOfRows.Max();
                    var batched = new List<float[]>[3];
                    using (Autograd.NoGrad())
                    {
                        using (var scope = new TensorScope())
                        using (var context = new DecodingContext(device, 3, 32))
                        {
                            context.SetRowStarts([.. lengthsOfRows.Select(n => longest - n)]);
                            context.LastPositionOnly = true;
                            var input = new float[3 * longest];
                            for (int row = 0; row < 3; row++)
                            {
                                ids.Skip(row).Take(lengthsOfRows[row]).Select(i => (float)i).ToArray().CopyTo(input, row * longest + longest - lengthsOfRows[row]);
                                batched[row] = [];
                            }

                            void Collect(Tensor logits)
                            {
                                var values = logits.ToArray();
                                for (int row = 0; row < 3; row++)
                                {
                                    batched[row].Add(values[(row * V)..((row + 1) * V)]);
                                }
                            }

                            Collect(model.ForwardCached(Tensor.From(input, [3, longest], device), context));
                            for (int s = 0; s < 4; s++)
                            {
                                Collect(model.ForwardCached(Tensor.From([.. Enumerable.Range(0, 3).Select(row => (float)ids[(row + s) % count])], [3, 1], device), context));
                            }
                        }

                        for (int row = 0; row < 3; row++)
                        {
                            using var scope = new TensorScope();
                            using var context = new DecodingContext(device, 1, 32) { LastPositionOnly = true };
                            var alone = new List<float[]> { model.ForwardCached(Tensor.From([.. ids.Skip(row).Take(lengthsOfRows[row]).Select(i => (float)i)], [1, lengthsOfRows[row]], device), context).ToArray() };
                            for (int s = 0; s < 4; s++)
                            {
                                alone.Add(model.ForwardCached(Tensor.From([(float)ids[(row + s) % count]], [1, 1], device), context).ToArray());
                            }

                            for (int s = 0; s < alone.Count; s++)
                            {
                                AssertClose(alone[s], batched[row][s], 1e-4f, $"{name}: row {row} of different lengths, step {s}");
                            }
                        }
                    }
                }

                // The composed path refuses packing again.
                CausalSelfAttention.WindowKernels = false;
                Check(!PackedSequences.Supports(model), $"{name}: IDRAK_WINDOW_KERNELS=0 refuses packing");
                CausalSelfAttention.WindowKernels = true;
            }
        }
        finally
        {
            CausalSelfAttention.WindowKernels = saved;
        }
    }
}
