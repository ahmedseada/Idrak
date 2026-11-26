// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Idrak;
using Idrak.Generation;
using Idrak.Gpu.Vulkan;
using Idrak.Layers;
using Idrak.Models.Abstractions;

// Sliding windows and soft-capped scores in the attention kernels (AttentionVariant): every kernel against a direct
// reference written out here, the Vulkan kernels against the CPU at every width with no host fallback, and decoders
// with windows and caps through the kernels against the composed path (basic operations over every cached slot, the
// path these layers took before the kernels had a window, still forced with IDRAK_WINDOW_KERNELS=0).
internal abstract partial class Tests
{
    internal static readonly (string Name, Action<Device> Run)[] WindowKernelGroup =
    [
        ("window kernels: gradients with windows and soft-caps (tiled and packed) match finite differences and the CPU; packed attention on a device with its own pass (Vulkan's span kernels) runs nothing on the host", WindowedAttentionGradients),
        ("window kernels: packed attention gives each sequence the same values bit for bit wherever it is packed and whatever is packed beside it (plain, a window, a soft-cap; inference and training passes)", PackedAttentionPlaceIndependent),
        ("window kernels: vulkan decoding, tiled and gradient kernels with windows and soft-caps match the CPU as kernels at every width and split count", VulkanWindowedKernels),
    ];

    // The variants checked: windows inside the keys, one beyond them, a cap alone, both.
    internal static readonly AttentionVariant[] WindowVariants = [new(3, 0f), new(17, 0f), new(0, 1.5f), new(4, 2f), new(1, 0f), new(1000, 0f)];

    // Attention written out (double): row i of head h sees positions [max(begin, end - window), end) of keys and values
    // [heads, capacity, dim], the scaled scores capped. range(h, i) gives (begin, end); writes the log-sum-exp when given.
    internal static float[] DirectAttention(float[] q, float[] k, float[] v, int heads, int rowsPerHead, int capacity, int dim, float scale,
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

    // The same five sequences packed two ways (other rows, other offsets, other neighbours): each one's outputs are the same
    // bits in both, as a behaviour pass and a training step that pack it apart need (their first ratio exactly 1).
    internal static void PackedAttentionPlaceIndependent(Device device)
    {
        const int Length = 16, Dim = 8;
        int[] sizes = [5, 7, 4, 9, 6];
        int[][][] layouts = [[[0, 1, 2], [3, 4]], [[4, 0, 2], [1, 3]]];
        var r = new Random(31);
        var data = sizes.Select(n => (Q: RandomArray(r, n * Dim, 1f), K: RandomArray(r, n * Dim, 1f), V: RandomArray(r, n * Dim, 1f))).ToArray();
        foreach (var variant in new AttentionVariant[] { default, new(3, 0f), new(0, 0.7f) })
        {
            foreach (bool training in new[] { false, true })
            {
                var seen = new float[layouts.Length][][];
                for (int l = 0; l < layouts.Length; l++)
                {
                    var layout = layouts[l];
                    int heads = layout.Length;
                    float[] q = new float[heads * Length * Dim], k = new float[heads * Length * Dim], v = new float[heads * Length * Dim];
                    var at = new (int Head, int Offset)[sizes.Length];
                    for (int h = 0; h < heads; h++)
                    {
                        int offset = 0;
                        foreach (int i in layout[h])
                        {
                            at[i] = (h, offset);
                            data[i].Q.CopyTo(q, (h * Length + offset) * Dim);
                            data[i].K.CopyTo(k, (h * Length + offset) * Dim);
                            data[i].V.CopyTo(v, (h * Length + offset) * Dim);
                            offset += sizes[i];
                        }
                    }

                    using var scope = new TensorScope();
                    using var packing = PackedSequences.Create([.. layout.Select(row => row.Select(i => sizes[i]).ToArray())], Length, device);
                    var tq = Tensor.From(q, [heads, Length, Dim], device, requiresGrad: training);
                    var tk = Tensor.From(k, [heads, Length, Dim], device);
                    var tv = Tensor.From(v, [heads, Length, Dim], device);
                    Tensor? y;
                    using (training ? null : (Autograd.NoGradScope?)Autograd.NoGrad())
                    {
                        y = PackedSequences.CausalAttentionSegmented(tq, tk, tv, packing, 1, 0.5f, variant);
                    }

                    if (y is null)
                    {
                        Check(!device.Backend.SupportsSegmentedAttention(Dim, variant), $"packed attention, window {variant.Window}, cap {variant.Softcap}: refused although supported");
                        return;
                    }

                    var values = y.ToArray();
                    seen[l] = [.. at.Select((p, i) => values.AsSpan((p.Head * Length + p.Offset) * Dim, sizes[i] * Dim).ToArray())];
                }

                for (int i = 0; i < sizes.Length; i++)
                {
                    Check(seen[0][i].AsSpan().SequenceEqual(seen[1][i]),
                        $"window {variant.Window}, cap {variant.Softcap}, {(training ? "training" : "inference")}: sequence {i} packed apart gives other bits (largest gap {Gap(seen[0][i], seen[1][i]):E2})");
                }
            }
        }
    }

    internal static void WindowedAttentionGradients(Device device)
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
                    var y = packed ? PackedSequences.CausalAttentionSegmented(tq, tk, tv, packing!, 1, 0.3f, variant) : Tensor.CausalAttention(tq, tk, tv, tz, T, 0.3f, variant);
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
                    long host = Idrak.Abstraction.Operations.Kernels.HostCalls(device.Backend);
                    var mine = Gradients(device, packed);
                    long hosted = Idrak.Abstraction.Operations.Kernels.HostCalls(device.Backend) - host;
                    Check(!packed || device.Backend is not ISegmentedAttentionBackend || hosted == 0, $"{what}: {hosted} operations ran on the host");
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

    internal static void VulkanWindowedKernels(Device device) => ForEachPromptBackend(device, (backend, label, full) =>
    {
        var random = new Random(73);
        float[] R(int n) => [.. Enumerable.Range(0, n).Select(_ => 2 * random.NextSingle() - 1)];

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
                foreach (var variant in variants)
                {
                    string shape = $"{label}: head size {dim}, {heads} heads × {groups} groups, {steps} steps, capacity {capacity}, position {position}, window {variant.Window}, cap {variant.Softcap}";
                    foreach (int? splits in full ? new int?[] { 1, 3, null } : [null])
                    {
                        VulkanBackend.AttentionSplits = splits;
                        string split = $"{shape}, {(splits is null ? "measured" : $"{splits} splits")}";
                        PromptCase(backend, $"decoding {split}", [q, keys, values, [position], new float[rows * dim]], [4],
                            (b, s) => b.AttentionDecode(s[0], s[1], s[2], s[3], s[4], heads, rowsPerHead, steps, capacity, dim, scale, variant), "attention_decode");
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

        static Backend Cpu() => Idrak.Abstraction.Devices.Cpu.CpuBackend.Instance;

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

}
