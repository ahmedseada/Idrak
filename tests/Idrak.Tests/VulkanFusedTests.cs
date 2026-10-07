// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Collections.Concurrent;
using Idrak;
using Idrak.Gpu.Vulkan;
using Idrak.Layers;

// The fused decoding kernels on Vulkan (BackendCapabilities.FusedKernels): each fused operation against the unfused steps
// it replaces on the same device and against the CPU, without host fallback and with the same bits run after run; and
// decoders running fewer dispatches per token with them. Run on a Vulkan device; elsewhere they do nothing.
internal static partial class Tests
{
    private static readonly (string Name, Action<Device> Run)[] VulkanFusedGroup =
    [
        ("vulkan fused: packed products sharing an input, gate/up with the activation, the gated down projection and projection + residual + RMS norm (int8, int4, bfloat16; odd widths, biases, splits of k) match the unfused steps and the CPU", VulkanFusedProducts),
        ("vulkan fused: attention heads normalized, rotated and laid out or written into float32 and bfloat16 caches in one pass match the unfused steps and the CPU", VulkanFusedHeads),
        ("vulkan fused: decoders (float32, int8, int4, bfloat16 weights; float32, bfloat16, int8 caches) give the unfused logits; a dim-1024, 8-layer int8 decoder runs fewer dispatches per token, the same kernels in direct and recorded steps", VulkanFusedDecoder),
    ];

    // The Vulkan backend of `device` when the fused kernels can be tested on it.
    private static bool FusedVulkan(Device device, out VulkanBackend backend)
    {
        backend = null!;
        if (device.Type != DeviceType.Vulkan)
        {
            return false;                                                // run on the Vulkan device itself
        }

        if (Environment.GetEnvironmentVariable("IDRAK_VULKAN_KERNELS") is "0" or "false")
        {
            Console.WriteLine("    (IDRAK_VULKAN_KERNELS=0: kernels off, skipped)");
            return false;
        }

        backend = (VulkanBackend)device.Backend;
        Check(backend.Capabilities.FusedKernels, "Vulkan reports fused kernels");
        return true;
    }

    private static readonly PackedFormat[] FusedFormats = [PackedFormat.Int8, PackedFormat.Int4, PackedFormat.BFloat16];

    private static void VulkanFusedProducts(Device device)
    {
        if (!FusedVulkan(device, out var backend))
        {
            return;
        }

        var cpu = Device.Cpu;
        var random = new Random(71);
        var (savedSplits, savedWords) = (VulkanBackend.GemvSplits, VulkanBackend.GemvWords);
        using var noGrad = Autograd.NoGrad();
        try
        {
            // (rows, k, columns of each product, biases): odd widths, a k that is no multiple of 32 (int4's groups), 1 to 8 rows.
            foreach (var (m, k, columns, biased) in new (int, int, int[], bool)[]
            {
                (1, 200, [70, 33, 45], true), (3, 96, [64, 64], false), (8, 136, [37, 37, 37], true), (2, 64, [9], true), (5, 160, [130, 7], false),
            })
            {
                // Splits: measured (or the formula's), one, two and four; words per row of k: 32 and 64.
                foreach (var (splits, words) in new (int?, int?)[] { (null, null), (1, 32), (2, 64), (4, 32) })
                {
                    (VulkanBackend.GemvSplits, VulkanBackend.GemvWords) = (splits, words);
                    foreach (var format in FusedFormats)
                    {
                        string what = $"{format}, {m} x {k} -> [{string.Join(", ", columns)}]{(biased ? " with biases" : "")}, splits {splits?.ToString() ?? "measured"}";
                        FusedMany(backend, device, cpu, random, format, m, k, columns, biased, what);
                        if (columns.Length == 2 && columns[0] == columns[1] || columns.Length == 3)
                        {
                            foreach (int activation in new[] { 0, 1, 2 })
                            {
                                FusedPair(backend, device, cpu, random, format, m, k, columns[0], biased, activation, $"{what}, gate/up pair, activation {activation}");
                            }
                        }

                        foreach (int activation in new[] { 0, 1 })
                        {
                            FusedGated(backend, device, cpu, random, format, m, k, columns[0], activation, $"{what}, gated down projection, activation {activation}");
                        }

                        FusedAddNorm(backend, device, cpu, random, format, m, k, columns[0], offset: biased ? 1f : 0f, $"{what}, projection + residual + RMS norm");
                    }
                }
            }
        }
        finally
        {
            (VulkanBackend.GemvSplits, VulkanBackend.GemvWords) = (savedSplits, savedWords);
        }
    }

    // Runs `fused` twice (the same bits both times, no host fallback, accepted) and returns its outputs.
    private static float[][] RunFused(VulkanBackend backend, Func<bool> fused, Tensor[] outputs, string what)
    {
        long fallbacks = Idrak.Abstraction.Operations.Kernels.HostCalls(backend);
        Check(fused(), $"{what}: the fused kernel declined");
        var first = outputs.Select(o => o.ToArray()).ToArray();
        Check(fused(), $"{what}: the fused kernel declined the second time");
        var second = outputs.Select(o => o.ToArray()).ToArray();
        Check(Idrak.Abstraction.Operations.Kernels.HostCalls(backend) == fallbacks, $"{what}: took the host fallback");
        for (int i = 0; i < first.Length; i++)
        {
            Check(first[i].Select(BitConverter.SingleToInt32Bits).SequenceEqual(second[i].Select(BitConverter.SingleToInt32Bits)), $"{what}: output {i} differs run after run");
        }

        return first;
    }

    private static void FusedMany(VulkanBackend backend, Device device, Device cpu, Random random, PackedFormat format, int m, int k, int[] columns, bool biased,
        string what)
    {
        var xs = RandomArray(random, m * k);
        var weights = columns.Select(n => RandomArray(random, k * n)).ToArray();
        var biases = columns.Select(n => biased ? RandomArray(random, n) : null).ToArray();
        using var x = Tensor.From(xs, [m, k], device);
        using var xc = Tensor.From(xs, [m, k], cpu);
        var packed = columns.Select((n, j) => PackedWeight.FromValues(format, weights[j], k, n, device)).ToArray();
        var packedCpu = columns.Select((n, j) => PackedWeight.FromValues(format, weights[j], k, n, cpu)).ToArray();
        var bias = biases.Select(b => b is null ? null : Tensor.From(b, device)).ToArray();
        var outputs = columns.Select(n => Tensor.Zeros([m, n], device)).ToArray();
        try
        {
            var fused = RunFused(backend, () =>
            {
                var products = new (Storage, Storage?, Storage?, Storage, int)[columns.Length];
                for (int j = 0; j < columns.Length; j++)
                {
                    products[j] = (packed[j].PackedValues.Storage, packed[j].ScaleValues?.Storage, bias[j]?.Storage, outputs[j].Storage, columns[j]);
                }

                return backend.PackedMatMulMany(format, x.Storage, m, k, products);
            }, outputs, what);
            for (int j = 0; j < columns.Length; j++)
            {
                using var product = packed[j].MatMul(x);
                using var biasedProduct = bias[j] is null ? null : product + bias[j]!;
                using var productCpu = packedCpu[j].MatMul(xc);
                using var biasCpu = biases[j] is null ? null : Tensor.From(biases[j]!, cpu);
                using var biasedCpu = biasCpu is null ? null : productCpu + biasCpu;
                AssertClose((biasedProduct ?? product).ToArray(), fused[j], 1e-4f, $"{what}: product {j} against the unfused product");
                AssertClose((biasedCpu ?? productCpu).ToArray(), fused[j], 1e-4f, $"{what}: product {j} against the CPU");
            }
        }
        finally
        {
            foreach (var d in packed.Concat(packedCpu).Cast<IDisposable>().Concat(bias.OfType<Tensor>()).Concat(outputs))
            {
                d.Dispose();
            }
        }
    }

    private static void FusedPair(VulkanBackend backend, Device device, Device cpu, Random random, PackedFormat format, int m, int k, int n, bool biased,
        int activation, string what)
    {
        var xs = RandomArray(random, m * k);
        var (wg, wu) = (RandomArray(random, k * n), RandomArray(random, k * n));
        var (bg, bu) = biased ? (RandomArray(random, n), RandomArray(random, n)) : (null, null);
        using var x = Tensor.From(xs, [m, k], device);
        using var xc = Tensor.From(xs, [m, k], cpu);
        using var gate = PackedWeight.FromValues(format, wg, k, n, device);
        using var up = PackedWeight.FromValues(format, wu, k, n, device);
        using var gateCpu = PackedWeight.FromValues(format, wg, k, n, cpu);
        using var upCpu = PackedWeight.FromValues(format, wu, k, n, cpu);
        using var gateBias = bg is null ? null : Tensor.From(bg, device);
        using var upBias = bu is null ? null : Tensor.From(bu, device);
        using var gateOut = Tensor.Zeros([m, n], device);
        using var upOut = Tensor.Zeros([m, n], device);
        using var hidden = Tensor.Zeros([m, n], device);
        var fused = RunFused(backend, () => backend.PackedMatMulGatedPair(format, activation, x.Storage, m, k,
            [(gate.PackedValues.Storage, gate.ScaleValues?.Storage, gateBias?.Storage, gateOut.Storage, n),
                (up.PackedValues.Storage, up.ScaleValues?.Storage, upBias?.Storage, upOut.Storage, n)], hidden.Storage), [gateOut, upOut, hidden], what);

        Tensor Biased(Tensor product, float[]? b, Device d)
        {
            if (b is null)
            {
                return product;
            }

            using var t = Tensor.From(b, d);
            using (product)
            {
                return product + t;
            }
        }

        foreach (var (d, g, u, input) in new[] { (device, gate, up, x), (cpu, gateCpu, upCpu, xc) })
        {
            using var gv = Biased(g.MatMul(input), bg, d);
            using var uv = Biased(u.MatMul(input), bu, d);
            using var h = Tensor.GatedActivation(gv, uv, activation);
            string against = d == cpu ? "the CPU" : "the unfused steps";
            AssertClose(gv.ToArray(), fused[0], 1e-4f, $"{what}: gate against {against}");
            AssertClose(uv.ToArray(), fused[1], 1e-4f, $"{what}: up against {against}");
            AssertClose(h.ToArray(), fused[2], 1e-4f, $"{what}: act(gate) · up against {against}");
        }
    }

    private static void FusedGated(VulkanBackend backend, Device device, Device cpu, Random random, PackedFormat format, int m, int k, int n, int activation,
        string what)
    {
        var (gs, us, ws) = (RandomArray(random, m * k, 3f), RandomArray(random, m * k), RandomArray(random, k * n));
        using var gate = Tensor.From(gs, [m, k], device);
        using var up = Tensor.From(us, [m, k], device);
        using var weight = PackedWeight.FromValues(format, ws, k, n, device);
        using var y = Tensor.Zeros([m, n], device);
        var fused = RunFused(backend, () => backend.PackedMatMulGated(format, activation, gate.Storage, up.Storage, weight.PackedValues.Storage,
            weight.ScaleValues?.Storage, y.Storage, m, n, k), [y], what);
        using (var hidden = Tensor.GatedActivation(gate, up, activation))
        using (var product = weight.MatMul(hidden))
        {
            AssertClose(product.ToArray(), fused[0], 1e-4f, $"{what}: against the unfused steps");
        }

        using var gc = Tensor.From(gs, [m, k], cpu);
        using var uc = Tensor.From(us, [m, k], cpu);
        using var wc = PackedWeight.FromValues(format, ws, k, n, cpu);
        using var hc = Tensor.GatedActivation(gc, uc, activation);
        using var pc = wc.MatMul(hc);
        AssertClose(pc.ToArray(), fused[0], 1e-4f, $"{what}: against the CPU");
    }

    private static void FusedAddNorm(VulkanBackend backend, Device device, Device cpu, Random random, PackedFormat format, int m, int k, int n, float offset,
        string what)
    {
        const float Eps = 1e-5f;
        var (xs, ws, rs, gs) = (RandomArray(random, m * k), RandomArray(random, k * n), RandomArray(random, m * n, 4f), RandomArray(random, n));
        using var x = Tensor.From(xs, [m, k], device);
        using var weight = PackedWeight.FromValues(format, ws, k, n, device);
        using var residual = Tensor.From(rs, [m, n], device);
        using var gain = Tensor.From(gs, device);
        using var y = Tensor.Zeros([m, n], device);
        using var sum = Tensor.Zeros([m, n], device);
        using var normalized = Tensor.Zeros([m, n], device);
        var fused = RunFused(backend, () => backend.PackedMatMulAddRmsNorm(format, x.Storage, weight.PackedValues.Storage, weight.ScaleValues?.Storage, y.Storage,
            m, n, k, residual.Storage, sum.Storage, gain.Storage, normalized.Storage, Eps, offset), [y, sum, normalized], what);
        foreach (var d in new[] { device, cpu })
        {
            using var xd = Tensor.From(xs, [m, k], d);
            using var wd = PackedWeight.FromValues(format, ws, k, n, d);
            using var rd = Tensor.From(rs, [m, n], d);
            using var gd = Tensor.From(gs, d);
            using var product = wd.MatMul(xd);
            var (s, normal) = Tensor.AddRmsNormAffine(rd, product, gd, Eps, offset);
            using (s)
            using (normal)
            {
                string against = d == cpu ? "the CPU" : "the unfused steps";
                AssertClose(product.ToArray(), fused[0], 1e-4f, $"{what}: projection against {against}");
                AssertClose(s.ToArray(), fused[1], 1e-4f, $"{what}: sum against {against}");
                AssertClose(normal.ToArray(), fused[2], 1e-4f, $"{what}: normalized against {against}");
            }
        }
    }

    private static void VulkanFusedHeads(Device device)
    {
        if (!FusedVulkan(device, out var backend))
        {
            return;
        }

        var random = new Random(72);
        using var noGrad = Autograd.NoGrad();
        // (batch, steps, heads, kvHeads, head size, rotary half, interleaved): grouped heads, odd head sizes, partial rotation.
        foreach (var (n, t, heads, kv, d, half, interleaved) in new[]
        {
            (1, 1, 4, 2, 64, 32, false), (2, 3, 6, 3, 7, 3, false), (1, 2, 8, 8, 12, 4, true), (2, 1, 4, 1, 130, 65, false),
        })
        {
            foreach (bool norm in new[] { true, false })
            {
                foreach (bool rope in new[] { true, false })
                {
                    foreach (var cache in new[] { KeyValueFormat.Float32, KeyValueFormat.BFloat16, (KeyValueFormat?)null })
                    {
                        string what = $"{n} x {t} steps, {heads}/{kv} heads of {d}, {(rope ? $"half {half}{(interleaved ? " interleaved" : "")}" : "no rotation")}, "
                            + $"{(norm ? "normalized" : "no norm")}, {(cache is { } c ? $"{c} cache" : "no cache")}";
                        FusedHeadsCase(backend, device, random, n, t, heads, kv, d, rope ? half : 0, interleaved, norm, cache, what);
                    }
                }
            }
        }
    }

    private static void FusedHeadsCase(VulkanBackend backend, Device device, Random random, int n, int t, int heads, int kv, int d, int half, bool interleaved,
        bool norm, KeyValueFormat? cache, string what)
    {
        const int Positions = 16, Capacity = 9, Start = 4;
        var qs = RandomArray(random, n * t * heads * d, 2f);
        var ks = RandomArray(random, n * t * kv * d, 2f);
        var vs = RandomArray(random, n * t * kv * d, 2f);
        var (gq, gk) = (RandomArray(random, d), RandomArray(random, d));
        var cosTable = new float[Positions * Math.Max(1, half)];
        var sinTable = new float[cosTable.Length];
        for (int p = 0; p < Positions; p++)
        {
            for (int i = 0; i < half; i++)
            {
                float angle = p / MathF.Pow(500f, 2f * i / Math.Max(1, 2 * half));
                (cosTable[p * half + i], sinTable[p * half + i]) = (MathF.Cos(angle), MathF.Sin(angle));
            }
        }

        var steps = Enumerable.Range(0, t).Select(s => (float)(Start + s)).ToArray();
        bool bfloat16 = cache == KeyValueFormat.BFloat16;
        int words = bfloat16 ? (d + 1) / 2 : d;

        // The unfused steps on a device: norms and rotation as the decoder runs them without the fused pass, the layouts
        // by permutation, the cache writes by the cache's own kernels. Returns queries, then keys and values (laid out, or
        // the caches' words).
        float[][] Unfused(Device on)
        {
            using var q = Tensor.From(qs, [n, t, heads, d], on);
            using var k = Tensor.From(ks, [n, t, kv, d], on);
            using var v = Tensor.From(vs, [n, t, kv, d], on);
            using var gainQ = Tensor.From(gq, on);
            using var gainK = Tensor.From(gk, on);
            using var cos = Tensor.From(cosTable, [Positions, Math.Max(1, half)], on);
            using var sin = Tensor.From(sinTable, [Positions, Math.Max(1, half)], on);
            using var positions = Tensor.From(steps, [t], on);
            Tensor qn, kn;
            if (norm && half > 0)
            {
                (qn, kn) = Tensor.RmsNormRopePair(q, gainQ, 1e-5f, 0.5f, k, gainK, 1e-6f, 0f, cos, sin, positions, half, interleaved);
            }
            else if (norm)
            {
                (qn, kn) = (q.RmsNormAffine(gainQ, 1e-5f, 0.5f), k.RmsNormAffine(gainK, 1e-6f, 0f));
            }
            else if (half > 0)
            {
                (qn, kn) = (q.Rope(cos, sin, positions, half, interleaved), k.Rope(cos, sin, positions, half, interleaved));
            }
            else
            {
                (qn, kn) = (q.Reshape(n, t, heads, d), k.Reshape(n, t, kv, d));
            }

            using (qn)
            using (kn)
            using (var queries = qn.Reshape(n, t, kv, heads / kv, d).Permute(0, 2, 3, 1, 4).Reshape(n * kv, heads / kv * t, d))
            using (var keys = kn.Permute(0, 2, 1, 3).Reshape(n * kv, t, d))
            using (var values = v.Permute(0, 2, 1, 3).Reshape(n * kv, t, d))
            {
                if (cache is null)
                {
                    return [queries.ToArray(), keys.ToArray(), values.ToArray()];
                }

                using var keyCache = Tensor.Zeros([n * kv, Capacity, words], on);
                using var valueCache = Tensor.Zeros([n * kv, Capacity, words], on);
                using var position = Tensor.From([(float)Start], [1], on);
                foreach (var (source, target) in new[] { (keys, keyCache), (values, valueCache) })
                {
                    if (bfloat16)
                    {
                        on.Backend.KeyValueWriteBFloat16(source.Storage, target.Storage, position.Storage, n * kv, t, Capacity, d);
                    }
                    else
                    {
                        on.Backend.KeyValueWrite(source.Storage, target.Storage, position.Storage, n * kv, t, Capacity, d);
                    }
                }

                return [queries.ToArray(), keyCache.ToArray(), valueCache.ToArray()];
            }
        }

        float[][] fused;
        {
            using var q = Tensor.From(qs, [n, t, heads * d], device);
            using var k = Tensor.From(ks, [n, t, kv * d], device);
            using var v = Tensor.From(vs, [n, t, kv * d], device);
            using var gainQ = Tensor.From(gq, device);
            using var gainK = Tensor.From(gk, device);
            using var cos = Tensor.From(cosTable, [Positions, Math.Max(1, half)], device);
            using var sin = Tensor.From(sinTable, [Positions, Math.Max(1, half)], device);
            using var positions = Tensor.From(steps, [t], device);
            using var position = Tensor.From([(float)Start], [1], device);
            using var yq = Tensor.Zeros([n * kv, heads / kv * t, d], device);
            using var yk = Tensor.Zeros(cache is null ? [n * kv, t, d] : [n * kv, Capacity, words], device);
            using var yv = Tensor.Zeros(cache is null ? [n * kv, t, d] : [n * kv, Capacity, words], device);
            fused = RunFused(backend, () => backend.NormRopeHeads(q.Storage, k.Storage, v.Storage, n, t, heads, kv, d,
                norm ? gainQ.Storage : null, 1e-5f, 0.5f, norm ? gainK.Storage : null, 1e-6f, 0f, half > 0 ? cos.Storage : null, half > 0 ? sin.Storage : null,
                positions.Storage, half, interleaved, yq.Storage, yk.Storage, yv.Storage, cache is null ? null : position.Storage,
                cache is null ? t : Capacity, cache is null ? d : bfloat16 ? 2 * words : d, bfloat16), [yq, yk, yv], what);
        }

        // bfloat16 caches: the stored halves as floats (a value rounded on either side of a tie may differ by one bfloat16 step).
        float[] Values(float[] stored) => !bfloat16 ? stored : [.. stored.SelectMany(w =>
        {
            uint bits = (uint)BitConverter.SingleToInt32Bits(w);
            return new[] { BitConverter.Int32BitsToSingle((int)(bits << 16)), BitConverter.Int32BitsToSingle((int)(bits & 0xFFFF0000u)) };
        })];

        foreach (var (against, expected) in new[] { ("the unfused steps", Unfused(device)), ("the CPU", Unfused(Device.Cpu)) })
        {
            AssertClose(expected[0], fused[0], 1e-5f, $"{what}: queries against {against}");
            AssertClose(Values(expected[1]), Values(fused[1]), bfloat16 ? 1e-2f : 1e-5f, $"{what}: keys against {against}");
            AssertClose(Values(expected[2]), Values(fused[2]), bfloat16 ? 1e-2f : 0f, $"{what}: values against {against}");
        }
    }

    private static void VulkanFusedDecoder(Device device)
    {
        if (!FusedVulkan(device, out var backend))
        {
            return;
        }

        // Small decoders with q/k norms and biases on the projections: every weight format and cache format, the fused
        // logits against the unfused ones.
        foreach (var (name, options) in new[]
        {
            ("float32 weights", new DecoderBuildOptions { Device = device, Seed = 4 }),
            ("int8 weights", new DecoderBuildOptions { Device = device, Seed = 4, Int8 = true }),
            ("int4 weights", new DecoderBuildOptions { Device = device, Seed = 4, Int4 = true }),
            ("bfloat16 weights", new DecoderBuildOptions { Device = device, Seed = 4, BFloat16 = true }),
        })
        {
            foreach (var spec in new[] { SmallSpec with { QkNorm = true }, SmallSpec with { QkvBias = true, Rope = new RopeSettings(100f, RotaryDim: 4, Interleaved: true) } })
            {
                using var model = spec.Build(null, options);
                foreach (var format in new[] { KeyValueFormat.Float32, KeyValueFormat.BFloat16, KeyValueFormat.Int8 })
                {
                    var (unfused, _, _) = FusedDecode(backend, device, model, spec.Vocabulary, format, fused: false, steps: 4);
                    var (fusedLogits, _, fallbacks) = FusedDecode(backend, device, model, spec.Vocabulary, format, fused: true, steps: 4);
                    float range = Math.Max(1f, unfused.Max(MathF.Abs));
                    AssertClose(unfused, fusedLogits, 1e-3f * range, $"{name}{(spec.QkNorm ? ", q/k norms" : ", biases, partial interleaved rotation")}, {format} cache: logits");
                    Check(fallbacks == 0, $"{name}, {format} cache: {fallbacks} host fallbacks in fused decoding steps");
                }
            }
        }

        // The medium decoder of --bench-vulkan: dispatches per token without and with the fused kernels.
        var medium = new DecoderSpec
        {
            Vocabulary = 40, Dim = 1024, Layers = 8, Heads = 16, KvHeads = 8, HeadDim = 64, FfDim = 3072, MaxPositions = 256,
            Rope = new RopeSettings(1_000_000f), NormEpsilon = 1e-6f, QkNorm = true,
        };
        using (var model = medium.Build(null, new DecoderBuildOptions { Device = device, Seed = 1, Int8 = true }))
        {
            var (before, perTokenBefore, _) = FusedDecode(backend, device, model, medium.Vocabulary, KeyValueFormat.Float32, fused: false, steps: 3);
            var kernels = new ConcurrentDictionary<string, long>();
            var (after, perTokenAfter, fallbacks) = FusedDecode(backend, device, model, medium.Vocabulary, KeyValueFormat.Float32, fused: true, steps: 3, kernels);
            float range = Math.Max(1f, before.Max(MathF.Abs));
            AssertClose(before, after, 1e-3f * range, "medium int8 decoder: logits");
            Check(fallbacks == 0, $"medium int8 decoder: {fallbacks} host fallbacks");
            Check(kernels.Keys.Any(k => k.StartsWith("int8_gemv_many_", StringComparison.Ordinal)) && kernels.Keys.Any(k => k.StartsWith("int8_gemv_pair_", StringComparison.Ordinal))
                && kernels.ContainsKey("norm_rope_heads") && kernels.ContainsKey("gemv_add_rms_norm") | kernels.ContainsKey("gemv_add_rms_norm_narrow"),
                "medium int8 decoder: the fused kernels ran (" + string.Join(", ", kernels.Keys.Order()) + ")");
            Check(perTokenAfter < perTokenBefore, $"medium int8 decoder: {perTokenAfter:F1} dispatches per token fused, {perTokenBefore:F1} unfused");
            Console.WriteLine($"    medium int8 decoder (dim 1024, 8 layers): {perTokenBefore:F1} dispatches per token unfused, {perTokenAfter:F1} fused");

            // A direct step launches the kernels a recorded step does: the residual addition fused with the next block's
            // norm is kept for that norm rather than freed with its layer and computed again.
            var (direct, recorded) = StepKernels(backend, device, model);
            string Listed(IDictionary<string, long> counts) => string.Join(", ", counts.OrderBy(p => p.Key, StringComparer.Ordinal).Select(p => $"{p.Key} {p.Value}"));
            Check(direct.Count == recorded.Count && direct.All(p => recorded.TryGetValue(p.Key, out long n) && n == p.Value),
                $"medium int8 decoder: a direct step runs {Listed(direct)}; a recorded step {Listed(recorded)}");
            Check(direct.GetValueOrDefault("rms_norm_affine") + direct.GetValueOrDefault("rms_norm_affine_narrow") == 1,
                $"medium int8 decoder: a direct step normalizes on its own only the embedding, the rest with the addition before ({Listed(direct)})");
            Console.WriteLine($"    medium int8 decoder: {direct.Values.Sum()} dispatches in a direct step, {recorded.Values.Sum()} recorded");
        }
    }

    // The kernels of one direct decoding step and of the same step recorded as a graph, by name.
    private static (ConcurrentDictionary<string, long> Direct, ConcurrentDictionary<string, long> Recorded) StepKernels(
        VulkanBackend backend, Device device, Sequential model)
    {
        model.Eval();
        using var noGrad = Autograd.NoGrad();
        using var context = new DecodingContext(device, 1, 32, KeyValueFormat.Float32);
        using var token = Tensor.From([5f], [1, 1], device);
        using (new TensorScope())
        {
            model.ForwardCached(Tensor.From([1f, 2f, 3f], [1, 3], device), context).ToArray();
        }

        var direct = new ConcurrentDictionary<string, long>();
        var recorded = new ConcurrentDictionary<string, long>();
        for (int s = 0; s < 3; s++)
        {
            using var scope = new TensorScope();
            device.Synchronize();
            backend.DispatchesByKernel = s == 2 ? direct : null;            // after two steps, so measured choices are settled
            model.ForwardCached(token, context);
            device.Synchronize();
            backend.DispatchesByKernel = null;
        }

        backend.DispatchesByKernel = recorded;
        try
        {
            using var graph = context.CaptureStep(() => model.ForwardCached(token, context));
        }
        finally
        {
            backend.DispatchesByKernel = null;
        }

        return (direct, recorded);
    }

    // Decodes a prompt and then `steps` fixed tokens (after two warm-up tokens, so measured choices are settled) with the
    // fused kernels on or off; returns the steps' logits, their dispatches per token and host fallbacks.
    private static (float[] Logits, double DispatchesPerToken, long HostCalls) FusedDecode(VulkanBackend backend, Device device, Sequential model, int vocabulary,
        KeyValueFormat format, bool fused, int steps, ConcurrentDictionary<string, long>? kernels = null)
    {
        VulkanBackend.FusedOff = !fused;
        try
        {
            model.Eval();
            using var noGrad = Autograd.NoGrad();
            using var context = new DecodingContext(device, 1, 32, format);
            using (new TensorScope())
            {
                model.ForwardCached(Tensor.From([1f, 2f, 3f], [1, 3], device), context).ToArray();
            }

            var logits = new List<float>();
            long dispatches = 0, fallbacks = 0;
            for (int s = 0; s < steps + 2; s++)
            {
                bool measured = s >= 2;
                using var scope = new TensorScope();
                using var token = Tensor.From([(float)((5 + 3 * s) % vocabulary)], [1, 1], device);
                device.Synchronize();
                var (d0, h0) = (backend.Dispatches, Idrak.Abstraction.Operations.Kernels.HostCalls(backend));
                backend.DispatchesByKernel = measured ? kernels : null;
                var output = model.ForwardCached(token, context);
                device.Synchronize();
                backend.DispatchesByKernel = null;
                if (measured)
                {
                    dispatches += backend.Dispatches - d0;
                    fallbacks += Idrak.Abstraction.Operations.Kernels.HostCalls(backend) - h0;
                    logits.AddRange(output.ToArray());
                }
            }

            return ([.. logits], dispatches / (double)steps, fallbacks);
        }
        finally
        {
            VulkanBackend.FusedOff = false;
        }
    }
}
