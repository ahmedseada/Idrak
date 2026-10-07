// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Collections.Concurrent;
using Idrak;
using Idrak.Backends;
using Idrak.Abstraction.Devices.Cpu;
using Idrak.Backends.Vulkan;
using Idrak.Layers;
using Idrak.Optimizers;

// Training on Vulkan: the training operations run as generated kernels (no host fallback), match the CPU over several
// shapes (odd sizes, both forms of the reductions), give the same bits run after run, and a small network trains to the
// CPU's losses with no host fallback in its steps. On other devices the checks do nothing.
internal static partial class Tests
{
    private static readonly (string Name, Action<Device> Run)[] VulkanTrainingGroup =
    [
        ("vulkan training: layer norm gradient, batch-norm statistics and gradients, group reductions, column sums and products with a bias run as kernels and match the CPU", VulkanTrainingNormsMatchCpu),
        ("cpu: column sums of a strided block (the base fallback's own target) add into y as the copy-then-sum reference does", CpuSumColumns),
        ("vulkan training: fused AdamW (with and without clipping) and 8-bit Adam (the CPU's codes) run as kernels and match the CPU", VulkanTrainingOptimizersMatchCpu),
        ("vulkan training: the tiled causal attention gradient (head sizes up to 256, grouped queries, capacity past the steps) runs as kernels and matches the CPU", VulkanTrainingAttentionMatchesCpu),
        ("vulkan training: the reductions and the attention gradient give the same bits run after run", VulkanTrainingRepeatable),
        ("vulkan training: an MLP with LayerNorm and BatchNorm trained by AdamW (clipped, fused) and 8-bit AdamW gives the CPU's losses with no host fallback", VulkanTrainingEndToEnd),
    ];

    // Runs `op` on the CPU (`cpuOp` when the CPU computes it another way) and on the Vulkan backend from the same inputs
    // and compares every storage after it; on Vulkan it must dispatch kernels and never take the host fallback.
    private static void TrainingCase(VulkanBackend backend, string what, float[][] inputs, Action<Backend, Storage[]> op, float tolerance = 2e-5f,
        Action<Backend, Storage[]>? cpuOp = null)
    {
        var cpu = CpuBackend.Instance;
        var host = inputs.Select(d => { var s = cpu.Allocate(d.Length, false); cpu.Upload(d, s); return s; }).ToArray();
        var gpu = inputs.Select(d => { var s = backend.Allocate(d.Length, false); backend.Upload(d, s); return s; }).ToArray();
        try
        {
            long dispatches = backend.Dispatches, fallbacks = Idrak.Abstraction.Operations.Kernels.HostCalls(backend);
            (cpuOp ?? op)(cpu, host);
            op(backend, gpu);
            Check(backend.Dispatches > dispatches, $"{what}: dispatched no kernel");
            Check(Idrak.Abstraction.Operations.Kernels.HostCalls(backend) == fallbacks, $"{what}: took the host fallback");
            for (int i = 0; i < inputs.Length; i++)
            {
                var (expected, actual) = (new float[inputs[i].Length], new float[inputs[i].Length]);
                cpu.Download(host[i], expected);
                backend.Download(gpu[i], actual);
                AssertClose(expected, actual, tolerance, $"{what}, storage {i}");
            }
        }
        finally
        {
            foreach (var s in host.Concat(gpu))
            {
                s.Release();
            }
        }
    }

    // The Vulkan backend of `device`, or null (with the reason skipped) when the checks do not apply.
    private static VulkanBackend? TrainingBackend(Device device)
    {
        if (device.Type != DeviceType.Vulkan)
        {
            return null;
        }

        if (Environment.GetEnvironmentVariable("IDRAK_VULKAN_KERNELS") is "0" or "false")
        {
            Console.WriteLine("    (IDRAK_VULKAN_KERNELS=0: kernels off, skipped)");
            return null;
        }

        return (VulkanBackend)device.Backend;
    }

    private static float[] Uniform(Random random, int n, float lo = -2f, float hi = 2f) => [.. Enumerable.Range(0, n).Select(_ => lo + (hi - lo) * random.NextSingle())];

    private static void VulkanTrainingNormsMatchCpu(Device device)
    {
        if (TrainingBackend(device) is not { } backend)
        {
            return;
        }

        var random = new Random(21);
        float[] R(int n, float lo = -2f, float hi = 2f) => Uniform(random, n, lo, hi);

        // Layer norm: the forward kernel's stats, then every gradient, only dgamma, and dx alone; narrow and wide rows.
        foreach (var (rows, cols) in new[] { (1, 1), (7, 33), (40, 700), (300, 5), (3, 1031) })
        {
            int n = rows * cols;
            TrainingCase(backend, $"layer norm backward {rows}×{cols}",
                [R(n), R(cols), R(cols), new float[n], new float[2 * rows], R(n), R(n), R(cols), R(cols), R(cols)], (b, s) =>
                {
                    b.LayerNormTrain(s[0], s[1], s[2], s[3], s[4], rows, cols, 1e-5f);
                    b.LayerNormBackward(s[0], s[1], s[5], s[4], s[6], s[7], s[8], rows, cols);
                    b.LayerNormBackward(s[0], s[1], s[5], s[4], null, s[9], null, rows, cols);
                    b.LayerNormBackward(s[0], s[1], s[5], s[4], s[6], null, null, rows, cols);
                }, 2e-4f);
        }

        // Grouped normalization over [outer, groups, inner]: short runs (column sums), long runs (a workgroup per group;
        // inner past any width), one group, two elements per group (one would have variance 0, which the CPU's E[x²] - mean²
        // in float products misses by its rounding).
        foreach (var (outer, groups, inner) in new[] { (5, 3, 7), (64, 16, 1), (2, 3, 1100), (1, 37, 9), (300, 1, 1), (3, 2, 2051), (2, 1, 1) })
        {
            int n = outer * groups * inner;
            TrainingCase(backend, $"batch-norm statistics, apply, gradient {outer}×{groups}×{inner}",
                [R(n, -1f, 3f), new float[groups], new float[groups], new float[groups], new float[n], R(n), R(groups), R(groups), R(n)], (b, s) =>
                {
                    b.NormStats(s[0], s[1], s[2], s[3], outer, groups, inner, 1e-5f);
                    b.NormApply(s[0], s[1], s[3], s[4], outer, groups, inner);
                    b.GroupReduce(s[5], s[4], s[6], s[7], outer, groups, inner);
                    b.NormBackward(s[5], s[4], s[6], s[7], s[3], s[8], outer, groups, inner);
                }, 2e-4f);
            TrainingCase(backend, $"group scale and shift, group sums {outer}×{groups}×{inner}",
                [R(n), R(groups), R(groups), R(n), R(n), R(groups)], (b, s) =>
                {
                    b.GroupScaleShift(s[0], s[1], s[2], s[3], n, groups, inner, accumulate: false);
                    b.GroupScaleShift(s[0], s[1], null, s[4], n, groups, inner, accumulate: true);
                    b.GroupScaleShift(s[3], null, s[2], s[4], n, groups, inner, accumulate: true);
                    b.GroupScaleShift(s[4], null, null, s[4], n, groups, inner, accumulate: false);
                    b.GroupReduce(s[0], null, s[5], null, outer, groups, inner);
                }, 1e-4f);
        }

        // Column sums of a strided block (the bias gradient of a slice). The CPU has no such pass of its own: its reference
        // copies the block out and sums its rows.
        foreach (var (rows, cols, ld, offset) in new[] { (3, 5, 9, 2), (1000, 7, 7, 0), (17, 300, 310, 5), (1, 1, 1, 0) })
        {
            TrainingCase(backend, $"column sums {rows}×{cols} (row stride {ld}, offset {offset})", [R(offset + rows * ld), R(cols)],
                (b, s) => b.SumColumns(s[0], offset, ld, s[1], rows, cols), 1e-4f,
                cpuOp: (b, s) =>
                {
                    var block = b.Allocate(rows * cols, zeroed: false);
                    b.Copy2D(s[0], offset, ld, block, 0, cols, rows, cols, accumulate: false);
                    b.SumRows(block, s[1], rows, cols);
                    block.Release();
                });
        }

        // c = a·b + bias (the CPU has no such pass: it multiplies and adds the bias).
        foreach (var (m, n, k) in new[] { (5, 7, 9), (70, 65, 80), (1, 3, 1), (33, 1, 17) })
        {
            TrainingCase(backend, $"product with a bias {m}×{n}×{k}", [R(m * k), R(k * n), R(n), R(m * n)],
                (b, s) => Check(b.MatMulBias(s[0], s[1], s[2], s[3], m, n, k), "the Vulkan backend has a product with a bias"), 1e-4f,
                cpuOp: (b, s) =>
                {
                    b.MatMul(s[0], s[1], s[3], m, n, k, false, false, 0f);
                    b.AddRowVector(s[3], s[2], s[3], m, n);
                });
        }
    }

    private static void VulkanTrainingOptimizersMatchCpu(Device device)
    {
        if (TrainingBackend(device) is not { } backend)
        {
            return;
        }

        var random = new Random(22);
        float[] R(int n, float lo = -2f, float hi = 2f) => Uniform(random, n, lo, hi);

        // Fused AdamW over tensors of 1, odd and many elements, clipped (the norm well above 1) and not, three steps each
        // (the moments carried over), gradients zeroed and not.
        int[] sizes = [1, 1001, 70_001];
        foreach (float maxNorm in new[] { 1f, 0f })
        {
            foreach (bool zero in new[] { true, false })
            {
                float[][] inputs = [.. sizes.SelectMany(n => new[] { R(n), R(n), new float[n], new float[n] })];
                IDisposable? cpuCache = null, gpuCache = null;
                try
                {
                    TrainingCase(backend, $"fused AdamW, max norm {maxNorm}, zero gradients {zero}", inputs, (b, s) =>
                    {
                        var tensors = sizes.Select((n, t) => (s[4 * t], s[4 * t + 1], s[4 * t + 2], s[4 * t + 3], n)).ToArray();
                        for (int step = 0; step < 3; step++)
                        {
                            ref var cache = ref b is CpuBackend ? ref cpuCache : ref gpuCache;
                            Check(b.FusedAdamW(tensors, ref cache, maxNorm, 0.01f * (step + 1), 0.999f, 0.9f, 0.95f, 1e-8f, zero), "fused AdamW runs");
                        }
                    }, 1e-5f);
                }
                finally
                {
                    cpuCache?.Dispose();
                    gpuCache?.Dispose();
                }
            }
        }

        // 8-bit Adam: whole blocks and a last partial block and word, a few steps; the parameters match and the codes are
        // the CPU's (the same map, the same nearest-code search).
        var map = AdamW8Bit.DynamicMap(signed: true).Concat(AdamW8Bit.DynamicMap(signed: false)).ToArray();
        foreach (int n in new[] { 1, 1001, 20_000 })
        {
            int blocks = (n + AdamW8Bit.BlockSize - 1) / AdamW8Bit.BlockSize, words = (n + 3) / 4;
            var p = R(n);
            var grads = Enumerable.Range(0, 4).Select(_ => R(n, -0.5f, 0.5f)).ToArray();
            (float[] P, uint[] M, uint[] V, float[] Scales) Run(Backend b)
            {
                Storage Put(float[] values)
                {
                    var s = b.Allocate(values.Length, zeroed: false);
                    b.Upload(values, s);
                    return s;
                }

                var (ps, ms, vs, scales, codes) = (Put(p), Put(new float[words]), Put(new float[words]), Put(new float[2 * blocks]), Put(map));
                try
                {
                    long fallbacks = Idrak.Abstraction.Operations.Kernels.HostCalls(b);
                    foreach (var g in grads)
                    {
                        var gs = Put(g);
                        b.AdamStep8Bit(ps, gs, ms, vs, scales, codes, n, 0.01f, 0.9f, 0.95f, 1e-8f, 0.7f, 0.999f);
                        gs.Release();
                    }

                    Check(b is CpuBackend || Idrak.Abstraction.Operations.Kernels.HostCalls(b) == fallbacks, "8-bit Adam took the host fallback");
                    var (pv, mv, vv, sv) = (new float[n], new float[words], new float[words], new float[2 * blocks]);
                    b.Download(ps, pv);
                    b.Download(ms, mv);
                    b.Download(vs, vv);
                    b.Download(scales, sv);
                    return (pv, [.. mv.Select(BitConverter.SingleToUInt32Bits)], [.. vv.Select(BitConverter.SingleToUInt32Bits)], sv);
                }
                finally
                {
                    foreach (var s in new[] { ps, ms, vs, scales, codes })
                    {
                        s.Release();
                    }
                }
            }

            var expected = Run(CpuBackend.Instance);
            var actual = Run(backend);
            // A moment within an ulp of the midpoint of two codes may round to the other code (checked below), and its
            // parameter then moves by a slightly different step in the later steps: such elements (as few as the codes
            // that may differ) are allowed a tenth of the learning rate, every other one 1e-5.
            int moved = 0;
            for (int i = 0; i < n; i++)
            {
                float d = Math.Abs(expected.P[i] - actual.P[i]);
                Check(d <= 1e-3f, $"8-bit Adam, {n} elements: parameters: element {i} is {actual.P[i]}, expected {expected.P[i]}");
                moved += d > 1e-5f ? 1 : 0;
            }

            Check(moved <= Math.Max(1, n / 1000), $"8-bit Adam, {n} elements: {moved} parameters differ from the CPU's by more than 1e-5");
            AssertClose(expected.Scales, actual.Scales, 1e-5f, $"8-bit Adam, {n} elements: block scales");
            int differ = 0;
            for (int i = 0; i < n; i++)
            {
                int shift = 8 * (i % 4);
                foreach (var (e, a) in new[] { (expected.M, actual.M), (expected.V, actual.V) })
                {
                    int ce = (int)(e[i / 4] >> shift & 255), ca = (int)(a[i / 4] >> shift & 255);
                    Check(Math.Abs(ce - ca) <= 1, $"8-bit Adam, {n} elements: code {ca} of element {i}, the CPU's {ce}");
                    differ += ce == ca ? 0 : 1;
                }
            }

            // Nearest codes agree unless a moment falls within an ulp of the midpoint of two codes.
            Check(differ <= n / 1000, $"8-bit Adam, {n} elements: {differ} codes differ from the CPU's");
            for (int w = n / 4; w < words; w++)
            {
                Check(actual.M[w] >> (8 * (n % 4)) == 0 && actual.V[w] >> (8 * (n % 4)) == 0, $"8-bit Adam, {n} elements: bytes past the end stay zero");
            }
        }
    }

    // Queries, keys, values, the forward output and log-sum-exp (from the CPU), dOutput and the gradients' starting values.
    private static float[][] AttentionInputs(Random random, int heads, int rowsPerHead, int steps, int capacity, int dim, float scale)
    {
        var cpu = CpuBackend.Instance;
        float[] q = Uniform(random, heads * rowsPerHead * dim, -1f, 1f), keys = Uniform(random, heads * capacity * dim, -1f, 1f);
        float[] values = Uniform(random, heads * capacity * dim, -1f, 1f);
        var y = new float[heads * rowsPerHead * dim];
        var lse = new float[heads * rowsPerHead];
        Storage Put(float[] v)
        {
            var s = cpu.Allocate(v.Length, false);
            cpu.Upload(v, s);
            return s;
        }

        var (qs, ks, vs, zero, ys, ls) = (Put(q), Put(keys), Put(values), Put([0f]), Put(y), Put(lse));
        cpu.AttentionTiled(qs, ks, vs, zero, ys, ls, heads, rowsPerHead, steps, capacity, dim, scale);
        cpu.Download(ys, y);
        cpu.Download(ls, lse);
        foreach (var s in new[] { qs, ks, vs, zero, ys, ls })
        {
            s.Release();
        }

        return [q, keys, values, y, lse, Uniform(random, y.Length, -1f, 1f), Uniform(random, q.Length, -0.1f, 0.1f),
            Uniform(random, keys.Length, -0.1f, 0.1f), Uniform(random, keys.Length, -0.1f, 0.1f)];
    }

    private static void VulkanTrainingAttentionMatchesCpu(Device device)
    {
        if (TrainingBackend(device) is not { } backend)
        {
            return;
        }

        var random = new Random(23);
        foreach (var (heads, group, steps, extra, dim) in new[] { (2, 1, 3, 0, 8), (3, 2, 17, 3, 48), (1, 1, 70, 0, 130), (2, 3, 5, 2, 256), (1, 1, 1, 0, 1) })
        {
            int rowsPerHead = group * steps, capacity = steps + extra;
            float scale = 1f / MathF.Sqrt(dim);
            TrainingCase(backend, $"attention backward, {heads} heads × {rowsPerHead} rows, {steps} steps, capacity {capacity}, head size {dim}",
                AttentionInputs(random, heads, rowsPerHead, steps, capacity, dim, scale),
                (b, s) => b.AttentionTiledBackward(s[0], s[1], s[2], s[3], s[4], s[5], s[6], s[7], s[8], heads, rowsPerHead, steps, capacity, dim, scale), 1e-4f);
        }
    }

    private static void VulkanTrainingRepeatable(Device device)
    {
        if (TrainingBackend(device) is not { } backend)
        {
            return;
        }

        var random = new Random(24);
        // Each case runs twice from the same inputs; every storage must come out with the same bits.
        void Twice(string what, float[][] inputs, Action<Storage[]> op)
        {
            float[][]? first = null;
            for (int run = 0; run < 2; run++)
            {
                var storages = inputs.Select(d => { var s = backend.Allocate(d.Length, false); backend.Upload(d, s); return s; }).ToArray();
                try
                {
                    op(storages);
                    var results = storages.Select(s => { var v = new float[s.Length]; backend.Download(s, v); return v; }).ToArray();
                    if (first is null)
                    {
                        first = results;
                        continue;
                    }

                    for (int i = 0; i < results.Length; i++)
                    {
                        Check(first[i].Select(BitConverter.SingleToInt32Bits).SequenceEqual(results[i].Select(BitConverter.SingleToInt32Bits)), $"{what}: storage {i} differs between runs");
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
        }

        int rows = 3000, cols = 37, n = rows * cols;
        Twice("layer norm backward", [Uniform(random, n), Uniform(random, cols), Uniform(random, cols), new float[n], new float[2 * rows], Uniform(random, n), new float[n], new float[cols], new float[cols]],
            s =>
            {
                backend.LayerNormTrain(s[0], s[1], s[2], s[3], s[4], rows, cols, 1e-5f);
                backend.LayerNormBackward(s[0], s[1], s[5], s[4], s[6], s[7], s[8], rows, cols);
            });
        Twice("batch-norm statistics and group sums", [Uniform(random, n), new float[cols], new float[cols], new float[cols], new float[cols], new float[1]], s =>
        {
            backend.NormStats(s[0], s[1], s[2], s[3], rows, cols, 1, 1e-5f);
            backend.GroupReduce(s[0], s[0], s[4], s[3], rows, cols, 1);
            backend.NormStats(s[0], s[5], s[5], s[5], 1, 1, n, 1e-5f);
        });
        int heads = 2, steps = 40, dim = 64;
        Twice("attention backward", AttentionInputs(random, heads, steps, steps, steps, dim, 0.125f),
            s => backend.AttentionTiledBackward(s[0], s[1], s[2], s[3], s[4], s[5], s[6], s[7], s[8], heads, steps, steps, steps, dim, 0.125f));
    }

    private static void VulkanTrainingEndToEnd(Device device)
    {
        if (TrainingBackend(device) is null)
        {
            return;
        }

        var r = new Random(25);
        var x = Enumerable.Range(0, 48 * 12).Select(_ => r.NextSingle() * 2 - 1).ToArray();
        var t = Enumerable.Range(0, 48 * 5).Select(i => i % 5 == i / 5 % 5 ? 1f : 0f).ToArray();

        // Losses of 8 steps on a device; fallbacks (by operation) counted over the training steps on Vulkan.
        (float[] Losses, float[][] Weights) Train(Device d, bool eightBit, ConcurrentDictionary<string, long>? fallbacks)
        {
            using var model = new Sequential(
                new Linear(12, 40, device: d, random: new Random(5)), new GELU(), new LayerNorm(40, device: d),
                new Linear(40, 24, device: d, random: new Random(6)), new BatchNorm(24, device: d), new ReLU(),
                new Linear(24, 5, device: d, random: new Random(7)));
            model.Train();
            using Optimizer optimizer = eightBit ? new AdamW8Bit(model.Parameters(), 1e-2f, minimumSize: 256) : new AdamW(model.Parameters(), 1e-2f, weightDecay: 0.05f);
            var losses = new List<float>();
            for (int step = 0; step < 8; step++)
            {
                using var scope = new TensorScope();
                var input = Tensor.From(x, [48, 12], d);
                var target = Tensor.From(t, [48, 5], d);
                d.Synchronize();
                if (fallbacks is not null)
                {
                    CountHostCalls(d.Backend, fallbacks);
                }

                try
                {
                    optimizer.ZeroGrad();
                    var loss = Losses.CrossEntropy(model.Forward(input), target);
                    loss.Backward();
                    if (eightBit)
                    {
                        optimizer.ClipGradientNorm(0.5f);
                        optimizer.Step();
                    }
                    else
                    {
                        optimizer.ClipAndStep(0.5f);                       // one fused pass
                    }

                    d.Synchronize();
                    CountHostCalls(d.Backend, null);
                    losses.Add(loss.ToArray()[0]);
                }
                finally
                {
                    CountHostCalls(d.Backend, null);
                }
            }

            return ([.. losses], [.. model.Parameters().Select(p => p.ToArray())]);
        }

        foreach (bool eightBit in new[] { false, true })
        {
            string name = eightBit ? "8-bit AdamW" : "AdamW";
            var counts = new ConcurrentDictionary<string, long>();
            var expected = Train(Device.Cpu, eightBit, null);
            var actual = Train(device, eightBit, counts);
            Check(counts.IsEmpty, $"{name}: host fallbacks in the training steps: {string.Join(", ", counts.OrderBy(c => c.Key).Select(c => $"{c.Key} ×{c.Value}"))}");
            AssertClose(expected.Losses, actual.Losses, 2e-3f, $"{name}: losses");
            Check(actual.Losses[^1] < actual.Losses[0], $"{name}: the loss falls ({actual.Losses[0]} to {actual.Losses[^1]})");
            for (int i = 0; i < expected.Weights.Length; i++)
            {
                AssertClose(expected.Weights[i], actual.Weights[i], 5e-3f, $"{name}: parameter {i} after training");
            }
        }
    }

    // CpuBackend once lacked SumColumns, so the base method's host fallback called the CPU's own (inherited) fallback
    // until the stack overflowed.
    private static void CpuSumColumns(Device device)
    {
        if (device.Type != DeviceType.Cpu)
        {
            return;
        }

        var cpu = CpuBackend.Instance;
        var r = new Random(9);
        const int rows = 7, cols = 13, ld = 20, offset = 3;
        float[] data = [.. Enumerable.Range(0, offset + rows * ld).Select(_ => r.NextSingle() - 0.5f)];
        float[] start = [.. Enumerable.Range(0, cols).Select(_ => r.NextSingle())];
        var x = cpu.Allocate(data.Length, false);
        var y = cpu.Allocate(cols, false);
        cpu.Upload(data, x);
        cpu.Upload(start, y);
        cpu.SumColumns(x, offset, ld, y, rows, cols);
        var actual = new float[cols];
        cpu.Download(y, actual);
        var expected = (float[])start.Clone();
        for (int j = 0; j < cols; j++)
        {
            for (int i = 0; i < rows; i++)
            {
                expected[j] += data[offset + i * ld + j];
            }
        }

        AssertClose(expected, actual, 1e-6f, "column sums added into y");
        x.Release();
        y.Release();
    }
}
