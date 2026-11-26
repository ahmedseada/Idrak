// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Idrak;
using Idrak.Abstraction.Devices.Cpu;

// Typed operands of the operations read where they read them (a row, a run of a group, a plane, a head's keys, a window of
// a strided view): an input of every built-in type gives the bits its values give as float32.
internal abstract partial class Tests
{
    // Runs `op` on the typed input and on its values as float32, each into fresh outputs of `sizes` floats (zeros, or
    // `start`'s values): every output the same bits.
    private static void SameOutputsOnTyped(string what, Tensor typed, Tensor plain, int[] sizes, Action<Storage, Storage[]> op, float[]?[]? start = null)
    {
        var backend = Device.Cpu.Backend;
        var results = new float[2][][];
        for (int run = 0; run < 2; run++)
        {
            var outputs = sizes.Select(s => backend.Allocate(Math.Max(1, s), zeroed: true)).ToArray();
            try
            {
                for (int i = 0; i < outputs.Length; i++)
                {
                    if (start?[i] is { } values)
                    {
                        backend.Upload(values, outputs[i]);
                    }
                }

                op(run == 0 ? typed.Storage : plain.Storage, outputs);
                results[run] = [.. outputs.Select((o, i) =>
                {
                    var values = new float[sizes[i]];
                    backend.Download(o, values);
                    return values;
                })];
            }
            finally
            {
                foreach (var o in outputs)
                {
                    o.Release();
                }
            }
        }

        for (int i = 0; i < sizes.Length; i++)
        {
            float[] x = results[0][i], y = results[1][i];
            int at = Enumerable.Range(0, x.Length).FirstOrDefault(j => BitConverter.SingleToInt32Bits(x[j]) != BitConverter.SingleToInt32Bits(y[j]), -1);
            Check(at < 0, $"{what}: output {i} element {at} is {(at < 0 ? 0 : x[at])} from the typed input, {(at < 0 ? 0 : y[at])} from its values as float32");
        }
    }

    internal static void WideResultsStayWide(Device device)
    {
        if (device.Type != DeviceType.Cpu)
        {
            return;
        }

        // 2^40 + 1 needs float64 (float32 holds 2^40); every result of an operation computed in float64 is held as float64.
        double big = Math.Pow(2, 40) + 1;
        foreach (var type in new[] { Idrak.Abstraction.Types.F64.Type, Idrak.Abstraction.Types.I64.Type })
        {
            using var scope = new TensorScope();
            var stored = new byte[type.ByteCount(4)];
            type.Write(new double[] { big, 3, -5, 7 }, stored);
            var x = Tensor.Stored(type, stored, [2, 2], Device.Cpu);
            var small = new byte[type.ByteCount(4)];
            type.Write(new double[] { 1, 2, 3, 4 }, small);
            var w = Tensor.Stored(type, small, [2, 2], Device.Cpu);
            foreach (var (what, result, expected) in new (string, Tensor, double[])[]
                     {
                         ("x + x", x + x, [2 * big, 6, -10, 14]),
                         ("x · 2 - 1", x * 2f - 1f, [2 * big - 1, 5, -11, 13]),
                         ("|x|", x.Abs(), [big, 3, 5, 7]),
                         ("the sum", x.Sum(), [big + 5]),
                         ("x · w", x.MatMul(w), [big + 9, 2 * big + 12, 16, 18]),
                     })
            {
                var values = new double[expected.Length];
                result.Type.Read(result.ToBytes(), values);
                Check(result.Type.Description == Idrak.Abstraction.Types.F64.Type.Description && values.SequenceEqual(expected),
                    $"{type.Name}: {what} is {result.Type.Name} [{string.Join(", ", values)}], expected float64 [{string.Join(", ", expected)}]");
            }
        }
    }

    internal static void TypedOperandsWhereRead(Device device)
    {
        if (device.Type != DeviceType.Cpu)
        {
            return;
        }

        const int Rows = 48, Cols = 256, Count = Rows * Cols;                 // rows of whole blocks of every built-in type
        var backend = Device.Cpu.Backend;
        var random = new Random(31);
        Tensor Floats(int count) => Tensor.From(RandomArray(random, count), [count], Device.Cpu);
        foreach (var type in TypedWritable(wide: false))
        {
            using var scope = new TensorScope();
            var (typed, plain) = TypedPair(type, Count, 17);
            string name = type.Name;
            var other = Floats(Count);

            // Layout: a transpose, a slice, sums and broadcasts over an axis, column sums.
            SameOutputsOnTyped($"{name}: permute", typed, plain, [Count], (x, o) => backend.Permute(x, o[0], [Cols, Rows], [1, Cols], false));
            SameOutputsOnTyped($"{name}: strided copy", typed, plain, [47 * 200], (x, o) => backend.StridedCopy(x, 3, [47, 200], [Cols, 1], o[0]));
            SameOutputsOnTyped($"{name}: strided copy of a broadcast", typed, plain, [6 * 5 * 7], (x, o) => backend.StridedCopy(x, 9, [6, 5, 7], [Cols, 0, 2], o[0]));
            SameOutputsOnTyped($"{name}: sum over an axis", typed, plain, [4 * Cols], (x, o) => backend.SumAxis(x, o[0], 4, 12, Cols, 0.5f, false));
            SameOutputsOnTyped($"{name}: sum over the first axis", typed, plain, [Cols], (x, o) => backend.SumAxis(x, o[0], 1, Rows, Cols, 1f, false));
            SameOutputsOnTyped($"{name}: broadcast over an axis", typed, plain, [Count], (x, o) => backend.BroadcastAxis(x, o[0], 4, 12, Cols, 0.5f));
            SameOutputsOnTyped($"{name}: sum of rows", typed, plain, [Cols], (x, o) => backend.SumRows(x, o[0], Rows, Cols));
            SameOutputsOnTyped($"{name}: sum of columns", typed, plain, [200], (x, o) => backend.SumColumns(x, Cols, Cols, o[0], 40, 200));

            // Grouped normalizations and the weight normalization.
            var mean = Floats(12);
            var inv = Tensor.From([.. Enumerable.Range(0, 12).Select(i => 0.5f + i * 0.1f)], [12], Device.Cpu);
            SameOutputsOnTyped($"{name}: group statistics", typed, plain, [12, 12, 12], (x, o) => backend.NormStats(x, o[0], o[1], o[2], 4, 12, Cols, 1e-5f));
            SameOutputsOnTyped($"{name}: group sums", typed, plain, [12, 12], (x, o) => backend.GroupReduce(x, other.Storage, o[0], o[1], 4, 12, Cols));
            SameOutputsOnTyped($"{name}: group apply", typed, plain, [Count], (x, o) => backend.NormApply(x, mean.Storage, inv.Storage, o[0], 4, 12, Cols));
            SameOutputsOnTyped($"{name}: group gradient", typed, plain, [Count], (x, o) => backend.NormBackward(x, other.Storage, mean.Storage, inv.Storage, inv.Storage, o[0], 4, 12, Cols));
            SameOutputsOnTyped($"{name}: group scale and shift", typed, plain, [Count], (x, o) => backend.GroupScaleShift(x, inv.Storage, mean.Storage, o[0], Count, 12, Cols, false));
            var gain = Floats(Rows);
            SameOutputsOnTyped($"{name}: weight normalization", typed, plain, [Count], (x, o) => backend.WeightNorm(gain.Storage, x, o[0], Rows, Cols));
            SameOutputsOnTyped($"{name}: weight normalization's gradient", typed, plain, [Rows, Count], (x, o) => backend.WeightNormBackward(gain.Storage, x, other.Storage, o[0], o[1], Rows, Cols));

            // The layer norm's gradients with a typed x and a typed dy (the parameters' gradients by columns).
            var gamma = Floats(Cols);
            var beta = Floats(Cols);
            var stats = Tensor.Zeros([2 * Rows], Device.Cpu);
            var normed = Tensor.Zeros([Count], Device.Cpu);
            backend.LayerNormTrain(plain.Storage, gamma.Storage, beta.Storage, normed.Storage, stats.Storage, Rows, Cols, 1e-5f);
            SameOutputsOnTyped($"{name}: layer norm's gradients, typed x", typed, plain, [Count, Cols, Cols],
                (x, o) => backend.LayerNormBackward(x, gamma.Storage, other.Storage, stats.Storage, o[0], o[1], o[2], Rows, Cols));
            SameOutputsOnTyped($"{name}: layer norm's gradients, typed dy", typed, plain, [Count, Cols, Cols],
                (dy, o) => backend.LayerNormBackward(other.Storage, gamma.Storage, dy, stats.Storage, o[0], o[1], o[2], Rows, Cols));

            // Rows read by index, classes, a scatter, the rotary tables.
            SameOutputsOnTyped($"{name}: class match", typed, plain, [Rows], (x, o) => backend.ClassMatch(x, other.Storage, o[0], Rows, Cols, 0.5f));
            var ids = Tensor.Indices([.. Enumerable.Range(0, Rows).Select(i => i * 7 % 10)], Device.Cpu);
            SameOutputsOnTyped($"{name}: scatter add", typed, plain, [10 * Cols], (x, o) => backend.ScatterAdd(x, ids.Storage, o[0], Rows, Cols, 10));
            var positions = Tensor.Indices([5, 40, 17], Device.Cpu);
            SameOutputsOnTyped($"{name}: rotary tables", typed, plain, [6 * 64], (t, o) => backend.Rope(other.Storage, o[0], t, t, positions.Storage, 6, 2, 3, 64, 32, false, 1f));

            // Attention over a typed cache (keys and values), a typed query; the gradients with a typed cache; the cache's writes.
            const int Heads = 2, Steps = 4, Dim = 64, Capacity = Count / (Heads * Dim);
            var q = Floats(Heads * Steps * Dim);
            var at = Tensor.Indices([80], Device.Cpu);
            SameOutputsOnTyped($"{name}: attention over a typed cache", typed, plain, [Heads * Steps * Dim, Heads * Steps],
                (kv, o) => backend.AttentionTiled(q.Storage, kv, kv, at.Storage, o[0], o[1], Heads, Steps, Steps, Capacity, Dim, 0.125f));
            SameOutputsOnTyped($"{name}: attention, a typed query", typed, plain, [Heads * Steps * Dim],
                (tq, o) => backend.AttentionDecode(tq, other.Storage, other.Storage, at.Storage, o[0], Heads, Steps, Steps, Capacity, Dim, 0.125f));
            var output = Tensor.Zeros([Heads * Steps * Dim], Device.Cpu);
            var lse = Tensor.Zeros([Heads * Steps], Device.Cpu);
            var zero = Tensor.Indices([0], Device.Cpu);
            backend.AttentionTiled(q.Storage, plain.Storage, plain.Storage, zero.Storage, output.Storage, lse.Storage, Heads, Steps, Steps, Capacity, Dim, 0.125f);
            var dOutput = Floats(Heads * Steps * Dim);
            SameOutputsOnTyped($"{name}: attention's gradients over a typed cache", typed, plain, [Heads * Steps * Dim, Count, Count],
                (kv, o) => backend.AttentionTiledBackward(q.Storage, kv, kv, output.Storage, lse.Storage, dOutput.Storage, o[0], o[1], o[2], Heads, Steps, Steps, Capacity, Dim, 0.125f));
            SameOutputsOnTyped($"{name}: a cache's write from a typed source", typed, plain, [Heads * 96 * Dim],
                (s, o) => backend.KeyValueWrite(s, o[0], at.Storage, Heads, Steps, 96, Dim));

            // Images: pools, interpolation, windows unfolded and folded back, and their gradients from a typed dy.
            var g = new ConvGeometry(2, 3, 32, 64, 3, 3, 2, 2, 1, 1);
            int pooled = g.N * g.C * g.OH * g.OW;
            SameOutputsOnTyped($"{name}: max pool", typed, plain, [pooled, pooled], (x, o) => backend.MaxPool(x, o[0], o[1], in g));
            SameOutputsOnTyped($"{name}: average pool", typed, plain, [pooled], (x, o) => backend.AvgPool(x, o[0], in g, false, 1, 1));
            SameOutputsOnTyped($"{name}: adaptive average pool", typed, plain, [6 * 35], (x, o) => backend.AdaptiveAvgPool(x, o[0], 6, 32, 64, 5, 7));
            SameOutputsOnTyped($"{name}: adaptive max pool", typed, plain, [6 * 35, 6 * 35], (x, o) => backend.AdaptiveMaxPool(x, o[0], o[1], 6, 32, 64, 5, 7));
            SameOutputsOnTyped($"{name}: interpolation", typed, plain, [6 * 20 * 50],
                (x, o) => backend.Interpolate2d(x, o[0], 6, 32, 64, 20, 50, InterpolationMode.Bilinear, false, 32f / 20, 64f / 50));
            var small = new ConvGeometry(1, 2, 16, 16, 3, 3, 1, 1, 1, 1);
            SameOutputsOnTyped($"{name}: windows unfolded", typed, plain, [small.Positions * small.PatchSize], (x, o) => backend.Im2Col(x, o[0], in small));
            SameOutputsOnTyped($"{name}: windows folded back", typed, plain, [2 * 16 * 16], (x, o) => backend.Col2Im(x, o[0], in small));
            var argmax = Tensor.Zeros([pooled], Device.Cpu);
            var pooledOut = Tensor.Zeros([pooled], Device.Cpu);
            backend.MaxPool(other.Storage, pooledOut.Storage, argmax.Storage, in g);
            SameOutputsOnTyped($"{name}: max pool's gradient", typed, plain, [Count], (dy, o) => backend.MaxPoolBackward(dy, argmax.Storage, o[0], in g));
            SameOutputsOnTyped($"{name}: average pool's gradient", typed, plain, [Count], (dy, o) => backend.AvgPoolBackward(dy, o[0], in g, false, 1, 1));
            SameOutputsOnTyped($"{name}: adaptive average pool's gradient", typed, plain, [Count], (dy, o) => backend.AdaptiveAvgPoolBackward(dy, o[0], 6, 32, 64, 5, 7));
            SameOutputsOnTyped($"{name}: interpolation's gradient", typed, plain, [Count],
                (dy, o) => backend.Interpolate2dBackward(dy, o[0], 6, 32, 64, 20, 50, InterpolationMode.Bilinear, false, 32f / 20, 64f / 50));

            // Sequences: CTC over typed log-probabilities, the best k, a weighted cross-entropy's weights.
            var targets = Tensor.Indices([1, 3, 2, 5, 4, 1], Device.Cpu);
            SameOutputsOnTyped($"{name}: CTC loss", typed, plain, [2],
                (lp, o) => backend.CtcLoss(lp, targets.Storage, o[0], [20, 17], [3, 3], [0, 3], 20, 2, 8, 0, false, false));
            SameOutputsOnTyped($"{name}: the best k", typed, plain, [Rows * 5, Rows * 5], (x, o) => backend.LogSoftmaxTopK(x, o[0], o[1], Rows, Cols, 5));

            // Routing, the experts' product over typed rows, the box and focal losses, the optimizers' typed gradients, a
            // typed A of the product with its bias.
            SameOutputsOnTyped($"{name}: routing", typed, plain, [Rows * 2, 257, Rows * 2, Rows * 2],
                (p, o) => backend.RouteTopK(p, o[0], o[1], o[2], o[3], Rows, Cols, 2, true));
            var offsets = Tensor.Indices([0, 20, Rows], Device.Cpu);
            var w0 = Floats(Cols * 8);
            var w1 = Floats(Cols * 8);
            var bias = Floats(8);
            SameOutputsOnTyped($"{name}: experts' product", typed, plain, [Rows * 8],
                (x, o) => backend.ExpertMatMul(x, offsets.Storage, [(w0.Storage, bias.Storage), (w1.Storage, null)], o[0], Rows, Cols, 8));
            SameOutputsOnTyped($"{name}: focal loss", typed, plain, [Count], (x, o) => backend.SigmoidFocalLoss(x, other.Storage, o[0], Count, 0.25f, 2f));
            SameOutputsOnTyped($"{name}: box loss", typed, plain, [Count / 4], (x, o) => backend.BoxIouLoss(x, other.Storage, o[0], Count / 4, BoxOverlap.CIoU, 1e-7f));
            SameOutputsOnTyped($"{name}: SGD with momentum", typed, plain, [Count, Count], (gr, o) => backend.SgdStep(o[0], gr, o[1], Count, 0.1f, 0.9f));
            SameOutputsOnTyped($"{name}: Adam", typed, plain, [Count, Count, Count], (gr, o) => backend.AdamStep(o[0], gr, o[1], o[2], Count, 1e-3f, 0.9f, 0.999f, 1e-8f));
            var weight = Floats(Cols * 16);
            var biasRow = Floats(16);
            SameOutputsOnTyped($"{name}: product with a bias, a typed A", typed, plain, [Rows * 16], (a, o) => backend.MatMulBias(a, weight.Storage, biasRow.Storage, o[0], Rows, 16, Cols));
        }

        // A typed cache written: each row as the cache's type writes it (whole groups of its blocks).
        foreach (var type in new[] { Idrak.Abstraction.Types.BF16.Type, BuiltInTypes.All.Types.First(t => t.Name == "q8_0") })
        {
            using var scope = new TensorScope();
            const int Heads = 2, Steps = 4, Dim = 64, Capacity = 8;
            var source = Tensor.From(RandomArray(new Random(3), Heads * Steps * Dim), [Heads * Steps * Dim], Device.Cpu);
            var cache = Tensor.Stored(type, new float[Heads * Capacity * Dim], [Heads * Capacity * Dim], Device.Cpu);
            backend.KeyValueWrite(source.Storage, cache.Storage, Tensor.Indices([2], Device.Cpu).Storage, Heads, Steps, Capacity, Dim);
            var read = cache.ToArray();
            var values = source.ToArray();
            var expected = new float[Heads * Capacity * Dim];
            for (int h = 0; h < Heads; h++)
            {
                var stored = new byte[type.ByteCount(Steps * Dim)];
                type.Write(values.AsSpan(h * Steps * Dim, Steps * Dim), stored);
                type.Read(stored, expected.AsSpan((h * Capacity + 2) * Dim, Steps * Dim));
            }

            Check(read.Select(BitConverter.SingleToInt32Bits).SequenceEqual(expected.Select(BitConverter.SingleToInt32Bits)), $"{type.Name}: a typed cache's rows as its type writes them");
        }
    }
}
