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

    // The values of `values` written through `type` and read back (what an operand of the type holds after a write).
    private static float[] ThroughType(IElementType type, float[] values)
    {
        var stored = new byte[type.ByteCount(values.Length)];
        type.Write(values, stored);
        var read = new float[values.Length];
        type.Read(stored, read);
        return read;
    }

    internal static void TypedOperandsInPlaceAndViews(Device device)
    {
        if (device.Type != DeviceType.Cpu)
        {
            return;
        }

        const int Rows = 48, Cols = 256, Count = Rows * Cols;
        var backend = Device.Cpu.Backend;
        var random = new Random(41);
        Tensor Floats(int count) => Tensor.From(RandomArray(random, count), [count], Device.Cpu);
        foreach (var type in TypedWritable(wide: false))
        {
            using var scope = new TensorScope();
            var (typed, plain) = TypedPair(type, Count, 19);
            string name = type.Name;
            var other = Floats(Count);

            // The paths the first test does not reach: sampling, the heads' norm and rotation, resizing, the recurrent
            // steps' typed weights forward and back.
            var step = Tensor.Indices([3], Device.Cpu);
            SameOutputsOnTyped($"{name}: sampling", typed, plain, [6, 13 * 6 * 4],
                (z, o) => backend.SampleRows(z, o[0], o[1], step.Storage, 6, 300, 2000, 7, 0.8f, 40, 0.9f, 0.02f, 11));
            var gains = Floats(64);
            var positions = Tensor.Indices([5, 40, 17], Device.Cpu);
            var at = Tensor.Indices([2], Device.Cpu);
            SameOutputsOnTyped($"{name}: the heads' norm and rotation", typed, plain, [4 * 3 * 64, 2 * 8 * 64, 2 * 8 * 64],
                (t, o) => backend.NormRopeHeads(t, t, t, 1, 3, 4, 2, 64, gains.Storage, 1e-6f, 0f, gains.Storage, 1e-6f, 1f, t, t, positions.Storage, 16, false, o[0], o[1], o[2],
                    at.Storage, 8, 64));
            var (coefficients, scales) = ResizeTables(24, 16, 64, 32, bytes: false);
            SameOutputsOnTyped($"{name}: resize and normalize", typed, plain, [6 * 16 * 24],
                (x, o) => backend.ResizeNormalize(x, coefficients.Storage, scales.Storage, o[0], 6, 3, 32, 64, 16, 24, 2, 2, false));
            var projected = Floats(2 * 3 * 64);
            SameOutputsOnTyped($"{name}: LSTM steps, typed weights", typed, plain, [32, 96, 384, 96],
                (w, o) => backend.LstmSequence(projected.Storage, w, o[0], o[1], o[2], o[3], 3, 2, 16, false));
            SameOutputsOnTyped($"{name}: GRU steps, typed weights", typed, plain, [96, 2 * 3 * 64],
                (w, o) => backend.GruSequence(projected.Storage, w, null, o[0], o[1], 3, 2, 16, false));
            var gates = Tensor.Zeros([384], Device.Cpu);
            var cells = Tensor.Zeros([96], Device.Cpu);
            backend.LstmSequence(projected.Storage, plain.Storage, Tensor.Zeros([32], Device.Cpu).Storage, Tensor.Zeros([96], Device.Cpu).Storage, gates.Storage, cells.Storage,
                3, 2, 16, false);
            var dOutput = Floats(96);
            SameOutputsOnTyped($"{name}: LSTM steps' gradient, typed weights", typed, plain, [32, 384],
                (w, o) => backend.LstmSequenceBackward(gates.Storage, cells.Storage, dOutput.Storage, w, o[0], o[1], 3, 2, 16, false));

            // The recurrent cells' typed projections and recurrent products (h = 16), and typed states written back through
            // their type (h = 256: rows of whole blocks of every type).
            var recurrent = Floats(2 * 64);
            SameOutputsOnTyped($"{name}: LSTM cell, typed projections", typed, plain, [32, 32, 96, 384, 96],
                (p, o) => backend.LstmCell(p, recurrent.Storage, o[0], o[1], o[2], o[3], o[4], 1, 3, 2, 16));
            SameOutputsOnTyped($"{name}: LSTM cell, a typed recurrent product", typed, plain, [32, 32, 96, 384, 96],
                (q, o) => backend.LstmCell(projected.Storage, q, o[0], o[1], o[2], o[3], o[4], 1, 3, 2, 16));
            SameOutputsOnTyped($"{name}: GRU cell, typed projections and recurrent product", typed, plain, [32, 96, 384],
                (p, o) => backend.GruCell(p, p, null, o[0], o[1], o[2], 1, 3, 2, 16));
            var wideProjected = Floats(2 * 3 * 1024);
            var wideRecurrent = Floats(2 * 1024);
            InPlace($"{name}: an LSTM cell's typed state", type, 2 * 256, 1, (s, _) =>
            {
                using var outputs = Tensor.Zeros([2 * 3 * 256], Device.Cpu);
                using var hidden = Tensor.Zeros([2 * 256], Device.Cpu);
                backend.LstmCell(wideProjected.Storage, wideRecurrent.Storage, s[0], hidden.Storage, outputs.Storage, null, null, 1, 3, 2, 256);
            }, other);
            InPlace($"{name}: a GRU cell's typed hidden state", type, 2 * 256, 1, (s, _) =>
            {
                using var outputs = Tensor.Zeros([2 * 3 * 256], Device.Cpu);
                backend.GruCell(wideProjected.Storage, wideRecurrent.Storage, null, s[0], outputs.Storage, null, 1, 3, 2, 256);
            }, other);

            // Operands written in place: the optimizers' parameters and moments, the cross-entropy's logits. The typed run
            // gives the float32 run's results written through the type.
            InPlace($"{name}: SGD's parameters and momentum", type, Count, 2, (s, g) => backend.SgdStep(s[0], g, s[1], Count, 0.1f, 0.9f), other);
            InPlace($"{name}: Adam's parameters and moments", type, Count, 3, (s, g) => backend.AdamStep(s[0], g, s[1], s[2], Count, 1e-3f, 0.9f, 0.999f, 1e-8f), other);
            var targets = Tensor.Indices([.. Enumerable.Range(0, Rows).Select(i => i * 31 % Cols)], Device.Cpu);
            var weights = Floats(Rows);
            InPlace($"{name}: the cross-entropy's logits", type, Count, 1, (s, _) =>
            {
                using var losses = Tensor.Zeros([Rows], Device.Cpu);
                backend.SoftmaxCrossEntropyRows(s[0], targets.Storage, weights.Storage, losses.Storage, Rows, Cols, 0.5f);
            }, other);

            // Views that are no plane (the last dimension not of step 1), typed, read in place: the product, the
            // attention, a cache's write; each gives what its contiguous float32 copy gives.
            var bView = new TensorLayout([2, 8, 12], [96, 1, 8], 0);
            var aMatrix = Floats(2 * 5 * 8);
            var contiguousB = Tensor.Zeros([2 * 8 * 12], Device.Cpu);
            backend.StridedCopy(plain.Storage, 0, [2, 8, 12], [96, 1, 8], contiguousB.Storage);
            using var expected = Tensor.Zeros([2 * 5 * 12], Device.Cpu);
            backend.BatchedMatMul(aMatrix.Storage, contiguousB.Storage, expected.Storage, 2, 5, 12, 8, false, false, 0f);
            var want = expected.ToArray();
            foreach (var (input, label) in new[] { (typed, "typed"), (plain, "float32") })
            {
                using var product = Tensor.Zeros([2 * 5 * 12], Device.Cpu);
                backend.BatchedMatMulStrided(aMatrix.Storage, TensorLayout.Contiguous([2, 5, 8]), input.Storage, bView, product.Storage, 2, 5, 12, 8, false, false, 0f);
                Check(product.ToArray().Select(BitConverter.SingleToInt32Bits).SequenceEqual(want.Select(BitConverter.SingleToInt32Bits)),
                    $"{name}: the product of a {label} view that is no plane, read in place, gives its contiguous copy's bits");
            }

            var kvView = new TensorLayout([2, 16, 64], [1024, 1, 16], 0);              // the rows and the columns swapped: no plane
            var q = Floats(2 * 4 * 64);
            var contiguousKv = Tensor.Zeros([2 * 16 * 64], Device.Cpu);
            backend.StridedCopy(plain.Storage, 0, [2, 16, 64], [1024, 1, 16], contiguousKv.Storage);
            var position = Tensor.Indices([10], Device.Cpu);
            using var attended = Tensor.Zeros([2 * 4 * 64], Device.Cpu);
            backend.AttentionTiled(q.Storage, contiguousKv.Storage, contiguousKv.Storage, position.Storage, attended.Storage, null, 2, 4, 4, 16, 64, 0.125f);
            using var strided = Tensor.Zeros([2 * 4 * 64], Device.Cpu);
            backend.AttentionTiledStrided(q.Storage, TensorLayout.Contiguous([2, 4, 64]), plain.Storage, kvView, plain.Storage, kvView, position.Storage, strided.Storage, null,
                2, 4, 4, 16, 64, 0.125f);
            Check(strided.ToArray().Select(BitConverter.SingleToInt32Bits).SequenceEqual(attended.ToArray().Select(BitConverter.SingleToInt32Bits)),
                $"{name}: attention over a cache view that is no plane gives its contiguous copy's bits");
            using var stridedTyped = Tensor.Zeros([2 * 4 * 64], Device.Cpu);
            backend.AttentionTiledStrided(q.Storage, TensorLayout.Contiguous([2, 4, 64]), typed.Storage, kvView, typed.Storage, kvView, position.Storage, stridedTyped.Storage,
                null, 2, 4, 4, 16, 64, 0.125f);
            Check(stridedTyped.ToArray().Select(BitConverter.SingleToInt32Bits).SequenceEqual(attended.ToArray().Select(BitConverter.SingleToInt32Bits)),
                $"{name}: attention over a typed cache view that is no plane gives its values' bits");
            var sourceView = new TensorLayout([2, 4, 64], [1024, 1, 16], 0);
            var contiguousSource = Tensor.Zeros([2 * 4 * 64], Device.Cpu);
            backend.StridedCopy(plain.Storage, 0, [2, 4, 64], [1024, 1, 16], contiguousSource.Storage);
            using var cacheWanted = Tensor.Zeros([2 * 8 * 64], Device.Cpu);
            backend.KeyValueWrite(contiguousSource.Storage, cacheWanted.Storage, at.Storage, 2, 4, 8, 64);
            using var cacheGot = Tensor.Zeros([2 * 8 * 64], Device.Cpu);
            backend.KeyValueWriteStrided(typed.Storage, sourceView, cacheGot.Storage, at.Storage, 2, 4, 8, 64);
            Check(cacheGot.ToArray().Select(BitConverter.SingleToInt32Bits).SequenceEqual(cacheWanted.ToArray().Select(BitConverter.SingleToInt32Bits)),
                $"{name}: a cache's write from a typed source view that is no plane gives its values' bits");
        }

        // Bytes: an 8-bit unsigned image read through its type gives the packed words' resizing.
        {
            using var scope = new TensorScope();
            var u8 = Idrak.Abstraction.Types.U8.Type;
            var pixels = new float[6 * 32 * 64];
            for (int i = 0; i < pixels.Length; i++)
            {
                pixels[i] = (i * 37 + i / 64 * 11) % 256;
            }

            var typedImage = Tensor.Stored(u8, pixels, [pixels.Length], Device.Cpu);
            var words = new float[typedImage.Storage.Length];
            backend.Download(typedImage.Storage, words);
            var packed = Tensor.From(words, [words.Length], Device.Cpu);              // the same bytes, four a word
            var (coefficients, table) = ResizeTables(24, 16, 64, 32, bytes: true);
            SameOutputsOnTyped("uint8: resize and normalize of bytes", typedImage, packed, [6 * 16 * 24],
                (x, o) => backend.ResizeNormalize(x, coefficients.Storage, table.Storage, o[0], 6, 3, 32, 64, 16, 24, 2, 2, true));
        }

        // Runs `op` on `count`-value operands written in place: float32 ones holding a type's values, then operands of the
        // type: the typed operands hold the float32 results written through the type.
        void InPlace(string what, IElementType type, int count, int operands, Action<Storage[], Storage> op, Tensor gradient)
        {
            using var scope = new TensorScope();
            // The typed operands first (moments of Adam are not negative), and float32 ones holding the values they hold.
            var typedOnes = Enumerable.Range(0, operands).Select(i =>
            {
                var values = RandomArray(new Random(50 + i), count);
                return Tensor.Stored(type, i == 0 ? values : [.. values.Select(MathF.Abs)], [count], Device.Cpu);
            }).ToArray();
            var floats = typedOnes.Select(t => Tensor.From(t.ToArray(), [count], Device.Cpu)).ToArray();
            op([.. floats.Select(f => f.Storage)], gradient.Storage);
            op([.. typedOnes.Select(t => t.Storage)], gradient.Storage);
            for (int i = 0; i < operands; i++)
            {
                var wanted = ThroughType(type, floats[i].ToArray());
                var got = typedOnes[i].ToArray();
                int bad = Enumerable.Range(0, count).FirstOrDefault(j => BitConverter.SingleToInt32Bits(wanted[j]) != BitConverter.SingleToInt32Bits(got[j]), -1);
                Check(bad < 0, $"{what}: operand {i} element {bad} is {(bad < 0 ? 0 : got[bad])}, the float32 run's written through the type {(bad < 0 ? 0 : wanted[bad])}");
            }
        }
    }

    // ResizeNormalize's tables for a [height, width] plane to [outHeight, outWidth]: two taps a direction (each output's
    // first input and its neighbour), as weights (floats) or Pillow's fixed-point integers (bytes); each channel's scale and
    // shift (floats) or byte table.
    private static (Tensor Coefficients, Tensor Values) ResizeTables(int outWidth, int outHeight, int width, int height, bool bytes)
    {
        var c = new List<float>();
        void Taps(int outputs, int size)
        {
            for (int o = 0; o < outputs; o++)
            {
                int first = Math.Min(size - 2, o * size / outputs);
                c.Add(BitConverter.Int32BitsToSingle(first));
                c.Add(BitConverter.Int32BitsToSingle(2));
                c.Add(bytes ? BitConverter.Int32BitsToSingle(1 << 21) : 0.75f);
                c.Add(bytes ? BitConverter.Int32BitsToSingle(1 << 21) : 0.25f);
            }
        }

        Taps(outWidth, width);
        Taps(outHeight, height);
        float[] values = bytes ? [.. Enumerable.Range(0, 3 * 256).Select(i => i % 256 / 255f - 0.5f)] : [0.5f, -0.25f, 2f, 0.125f, 1f, 0f];
        return (Tensor.From([.. c], [c.Count], Device.Cpu), Tensor.From(values, [values.Length], Device.Cpu));
    }

    // A float64 tensor of `values` (any of them past float32's 24 bits) and its values read back as float64.
    private static Tensor Float64(double[] values, int[] shape, bool requiresGrad = false)
    {
        var type = Idrak.Abstraction.Types.F64.Type;
        var stored = new byte[type.ByteCount(values.Length)];
        type.Write(values, stored);
        var t = Tensor.Stored(type, stored, shape, Device.Cpu);
        t.RequiresGrad = requiresGrad;
        return t;
    }

    private static double[] Doubles(Tensor t)
    {
        var values = new double[t.Size];
        t.Type.Read(t.ToBytes(), values);
        return values;
    }

    internal static void WideLanesAndGradients(Device device)
    {
        if (device.Type != DeviceType.Cpu)
        {
            return;
        }

        double big = Math.Pow(2, 40) + 1, mid = Math.Pow(2, 20) + 1;
        double[] values = [big, 3, -5, mid, 0.5, -big];
        using var scope = new TensorScope();
        void Same(string what, Tensor result, double[] expected) =>
            Check(result.Type.Description == Idrak.Abstraction.Types.F64.Type.Description && Doubles(result).SequenceEqual(expected),
                $"float64 {what}: {result.Type.Name} [{string.Join(", ", Doubles(result))}], expected [{string.Join(", ", expected)}]");

        var x = Float64(values, [2, 3]);
        Same("power", x.Pow(2f), [.. values.Select(v => Idrak.Abstraction.Devices.Cpu.CpuMath.PowDouble(v, 2))]);
        Same("clamp", x.Clamp(-4f, 1e13f), [.. values.Select(v => Math.Min(Math.Max(v, -4), (double)1e13f))]);
        Same("leaky ReLU", x.LeakyRelu(0.25f), [.. values.Select(v => v > 0 ? v : 0.25 * v)]);
        var condition = Tensor.From([1f, 0f, 1f, 0f, 1f, 0f], [2, 3], Device.Cpu);
        var other = Float64([.. values.Select(v => v + 1)], [2, 3]);
        Same("where", Tensor.Where(condition, x, other), [.. values.Select((v, i) => i % 2 == 0 ? v : v + 1)]);
        Same("a transposed copy", x.Transpose(0, 1).Contiguous(), [big, mid, 3, 0.5, -5, -big]);
        var bias = Float64([1, 2, 3], [3]);
        Same("a row's bias", x + bias, [.. values.Select((v, i) => v + 1 + i % 3)]);

        // Gradients of float64 tensors are float64: y = Σ (x² + 3x + w·x) has dx = 2x + 3 + w and dw = x.
        var leaf = Float64(values, [6], requiresGrad: true);
        var weight = Float64([1, 2, 3, 4, 5, 6], [6], requiresGrad: true);
        var loss = (leaf.Square() + leaf * 3f + weight * leaf).Sum();
        loss.Backward();
        Same("x's gradient", leaf.Grad!, [.. values.Select((v, i) => 2 * v + 3 + (i + 1))]);
        Same("w's gradient", weight.Grad!, values);

        // The product's gradients: y = Σ a·b, da = Σ_j b[k, j] a row, db = Σ_i a[i, k] a column.
        var a = Float64([big, 1, 2, mid], [2, 2], requiresGrad: true);
        var b = Float64([1, 2, 3, 4], [2, 2], requiresGrad: true);
        a.MatMul(b).Sum().Backward();
        Same("the product's left gradient", a.Grad!, [3, 7, 3, 7]);
        Same("the product's right gradient", b.Grad!, [big + 2, big + 2, 1 + mid, 1 + mid]);
    }

    internal static void TypedViewsKeepTheirType(Device device)
    {
        if (device.Type != DeviceType.Cpu)
        {
            return;
        }

        const int Rows = 12, Cols = 256;
        foreach (var type in TypedWritable(wide: false).Concat(TypedWritable(wide: true)))
        {
            using var scope = new TensorScope();
            var values = new float[Rows * Cols];
            var random = new Random(23);
            for (int i = 0; i < values.Length; i++)
            {
                values[i] = random.NextSingle() * 4f - 2f;
            }

            var typed = Tensor.Stored(type, values, [Rows, Cols], Device.Cpu);
            var plain = Tensor.From(typed.ToArray(), [Rows, Cols], Device.Cpu);
            bool perValue = type.Description.UnitValues == 1;
            // One value a block: a transpose; any type: rows of whole blocks (a narrow of the first dimension).
            var (view, plainView, what) = perValue
                ? (typed.Transpose(0, 1), plain.Transpose(0, 1), "a transpose")
                : (typed.Narrow(0, 2, 5), plain.Narrow(0, 2, 5), "rows of whole blocks");
            var copy = view.Contiguous();
            var wanted = plainView.Contiguous().ToArray();
            Check(copy.Type.Description == type.Description && copy.ToArray().Select(BitConverter.SingleToInt32Bits).SequenceEqual(wanted.Select(BitConverter.SingleToInt32Bits)),
                $"{type.Name}: {what} copied keeps its type ({copy.Type.Name}) and its values");
            if (CpuBackend.WideValues(type))
            {
                continue;                                                   // computed in float64: not its float32 values' bits
            }

            var exp = view.Exp().ToArray();
            var expPlain = plainView.Exp().ToArray();
            Check(exp.Select(BitConverter.SingleToInt32Bits).SequenceEqual(expPlain.Select(BitConverter.SingleToInt32Bits)),
                $"{type.Name}: exp of {what} gives its float32 values' bits");
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
