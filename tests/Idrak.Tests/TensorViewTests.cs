// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Idrak.Abstraction.Devices;
using Idrak.Abstraction.Operations;

// Plan 37, V0 and V1: every tensor's layout (shape, strides, offset), and views on the CPU (permute, transpose, narrow,
// select, expand share their input's storage), with results and gradients unchanged against today's copies.
internal abstract partial class DesktopTests : Tests
{
    private static readonly (string Name, Action<Device> Run)[] TensorLayoutGroup =
    [
        ("tensor layout: a tensor made by an operation is contiguous from element 0 (row-major strides); layouts permute, narrow, select and expand without moving an element (each element where it was); Contiguous() of a contiguous tensor is itself; the CPU lists the strided copy and the product among its strided inputs, a device that does not none", LayoutBasics),
        ("tensor layout: a reshape is written as strides where dimensions sit one after another (merged or split, size-1 dimensions anywhere) and not where a permuted pair would merge; rows of a layout start where its strides put them; the CPU lists its strided product and attention, forward and backward", LayoutReshapes),
    ];

    private static readonly (string Name, Action<Device> Run)[] TensorViewGroup =
    [
        ("tensor views: permute, transpose, narrow, select, expand, a reshape of a slice and chains of them share their input's storage (views, counted) and equal the copies made with views off, element for element; Contiguous() copies once", ViewsEqualCopies),
        ("tensor views: gradients through views (permute, narrow, select, expand, chains, into products, element-wise ops, softmax and sums) equal those through the copies made with views off, bit for bit", ViewGradients),
        ("tensor views: a view keeps the values it was made with when its input is written afterwards (CopyFrom, Fill, a kernel through Storage); a write to a view's storage does not reach its input; products read a transposed view in place and element-wise ops, softmax, sums and concatenation read views through short-lived copies, none kept", ViewsKeepTheirValues),
        ("tensor views: views off (IDRAK_VIEWS=0, TensorViews.Enabled) make today's copies and no view; a device copying the CPU's capabilities (the minimal test device) reads its views the same", ViewsOff),
        ("tensor views: the tiled attention reads heads split from projections and permuted (q, keys, values views) where they are, and the merged heads go to the output product as a view, forward and gradients bit for bit against views off, no view copied", AttentionReadsViews),
        ("tensor views: the product reads a strided right operand itself (few rows and many, transposed or not, batched), forward and gradients bit for bit against views off, no view copied", ProductReadsViews),
        ("tensor views: values released while a view still reads them stay until the view is copied to its own (then they go), and Release keeps a view's source", ReleaseKeepsViewedValues),
    ];

    private static void LayoutBasics(Device device)
    {
        _ = device;
        if (!FirstRun(nameof(LayoutBasics)))
        {
            return;
        }

        using var t = Tensor.Zeros([2, 3, 4], Device.Cpu);
        var layout = t.Layout;
        Check(layout.IsContiguousFromStart && layout.Strides.SequenceEqual(new long[] { 12, 4, 1 }) && layout.Offset == 0 && layout.Elements == 24 && !t.IsView
              && ReferenceEquals(t.Contiguous(), t), $"a made tensor: {layout}");
        var permuted = layout.Permute([2, 0, 1]);
        var narrowed = layout.Narrow(1, 1, 2);
        var selected = layout.Select(0, 1);
        var expanded = TensorLayout.Contiguous([3, 1]).Expand([2, 3, 5]);
        Check(permuted.Shape.SequenceEqual([4, 2, 3]) && permuted.Strides.SequenceEqual(new long[] { 1, 12, 4 }) && !permuted.IsContiguous
              && permuted.ElementAt([3, 1, 2]) == layout.ElementAt([1, 2, 3]),
            $"permute: {permuted}");
        Check(narrowed.Shape.SequenceEqual([2, 2, 4]) && narrowed.Offset == 4 && !narrowed.IsContiguous && narrowed.ElementAt([1, 1, 3]) == layout.ElementAt([1, 2, 3])
              && layout.Narrow(0, 1, 1).IsContiguous && !layout.Narrow(0, 1, 1).IsContiguousFromStart, $"narrow: {narrowed}");
        Check(selected.Shape.SequenceEqual([3, 4]) && selected.Offset == 12 && selected.IsContiguous && selected.ElementAt([2, 3]) == layout.ElementAt([1, 2, 3]), $"select: {selected}");
        Check(expanded.Shape.SequenceEqual([2, 3, 5]) && expanded.Strides.SequenceEqual(new long[] { 0, 1, 0 }) && expanded.ElementAt([1, 2, 4]) == 2, $"expand: {expanded}");
        Check(Device.Cpu.Backend.Capabilities.StridedInputs.Contains(Ops.StridedCopy) && Device.Cpu.Backend.Capabilities.StridedInputs.Contains(Ops.BatchedMatMul)
              && MinimalBackend.Instance.Capabilities.StridedInputs.Contains(Ops.StridedCopy)
              && new BackendCapabilities { FewRows = 1, DecodeAttentionHeadDim = 1, TiledAttentionHeadDim = 1 }.StridedInputs.Count == 0,
            "the CPU reads strides in its strided copy and its products; a device that says nothing in none");
    }

    private static void LayoutReshapes(Device device)
    {
        _ = device;
        if (!FirstRun(nameof(LayoutReshapes)))
        {
            return;
        }

        var layout = TensorLayout.Contiguous([2, 3, 4, 5]);
        var permuted = layout.Permute([0, 2, 1, 3]);                      // [2, 4, 3, 5] strides [60, 5, 20, 1]
        var split = permuted.Reshape([2, 2, 2, 3, 5]);
        var swapped = layout.Permute([1, 0, 2, 3]);                       // [3, 2, 4, 5] strides [20, 60, 5, 1]
        var merged = swapped.Reshape([3, 2, 20]);
        var ones = permuted.Reshape([1, 2, 4, 1, 3, 5, 1]);
        Check(split is not null && split.Strides.SequenceEqual(new long[] { 60, 10, 5, 20, 1 }), $"a split: {split}");
        Check(merged is not null && merged.Strides.SequenceEqual(new long[] { 20, 60, 1 }) && !merged.IsContiguous, $"a merge of neighbours: {merged}");
        Check(ones is not null && ones.Strides[1] == 60 && ones.Strides[2] == 5 && ones.Strides[4] == 20 && ones.Strides[5] == 1, $"size-1 dimensions: {ones}");
        Check(permuted.Reshape([2, 4, 15]) is null && permuted.Reshape([8, 15]) is null && swapped.Reshape([6, 20]) is null,
            "a merge of a permuted pair is not strides");
        Check(layout.Narrow(0, 1, 1).Reshape([12, 5]) is { IsContiguous: true, Offset: 60 }, "a slice of the first dimension reshaped: contiguous from its offset");
        var flat = TensorLayout.Contiguous([2, 3]).Reshape([6]);
        Check(flat is { IsContiguousFromStart: true }, $"a contiguous layout flattened: {flat}");

        // Rows: [2, 4, 3, 5] permuted, read as rows of 5 (the last dimension) and of 15 (no: the last dimension holds 5).
        Check(permuted.HasRows(5) && !permuted.HasRows(15) && !permuted.HasRows(2) && layout.Permute([0, 1, 3, 2]).HasRows(1)
              && !layout.Permute([0, 1, 3, 2]).HasRows(4), "rows sit one after another only along a last dimension of stride 1");
        var offsets = new long[2 * 4 * 3];
        permuted.RowOffsets(5, offsets);
        bool rowsRight = true;
        for (int a = 0, r = 0; a < 2; a++)
        {
            for (int b = 0; b < 4; b++)
            {
                for (int c = 0; c < 3; c++, r++)
                {
                    rowsRight &= offsets[r] == layout.ElementAt([a, c, b, 0]);
                }
            }
        }

        Check(rowsRight, $"each row where its strides put it: {string.Join(", ", offsets.Take(8))}");
        var caps = Device.Cpu.Backend.Capabilities.StridedInputs;
        Check(caps.Contains(Ops.BatchedMatMulStrided) && caps.Contains(Ops.AttentionTiledStrided) && caps.Contains(Ops.AttentionTiledStridedBackward),
            "the CPU's product and tiled attention read strides, forward and backward");
    }

    // Runs `make` with views on and off; returns both results' values (and with views on, whether `isView` held).
    private static (float[] On, float[] Off) WithViews(Func<Tensor> make, Action<Tensor>? onView = null)
    {
        bool before = TensorViews.Enabled;
        try
        {
            TensorViews.Enabled = true;
            float[] on;
            using (new TensorScope())
            {
                var y = make();
                onView?.Invoke(y);
                on = y.ToArray();
            }

            TensorViews.Enabled = false;
            float[] off;
            using (new TensorScope())
            {
                off = make().ToArray();
            }

            return (on, off);
        }
        finally
        {
            TensorViews.Enabled = before;
        }
    }

    private static int[] BitsOf(float[] values) => [.. values.Select(BitConverter.SingleToInt32Bits)];

    private static void ViewsEqualCopies(Device device)
    {
        if (device.Type != DeviceType.Cpu)
        {
            return;                                                      // views on the GPUs are V2
        }

        var random = new Random(3);
        float[] values = [.. Enumerable.Range(0, 2 * 3 * 4 * 5).Select(_ => random.NextSingle() * 2 - 1)];
        using var x = Tensor.From(values, [2, 3, 4, 5], device);
        long made = TensorViews.Made;
        var cases = new (string Name, Func<Tensor> Make)[]
        {
            ("permute", () => x.Permute(0, 2, 1, 3)),
            ("transpose", () => x.Transpose(-2, -1)),
            ("narrow", () => x.Narrow(2, 1, 2)),
            ("narrow of the first dimension, reshaped", () => x.Narrow(0, 1, 1).Reshape(3, 20)),
            ("select", () => x.Select(1, 2)),
            ("expand", () => x.Narrow(2, 0, 1).Expand(3, 2, 3, 4, 5)),
            ("a chain", () => x.Permute(3, 1, 0, 2).Narrow(1, 1, 2).Select(2, 1).Transpose(0, 1)),
            ("contiguous of a chain", () => x.Transpose(0, 3).Narrow(0, 2, 3).Contiguous()),
            ("a reshape of a permute strides express", () => x.Permute(1, 0, 2, 3).Reshape(3, 2, 20)),
            ("a reshape of a permute strides do not express", () => x.Permute(0, 2, 1, 3).Reshape(2, 4, 15)),
            ("a view of such a reshape", () => x.Permute(0, 2, 1, 3).Reshape(8, 15).Transpose(0, 1).Narrow(0, 2, 7)),
            ("heads split, permuted and merged", () => x.Reshape(2, 3, 2, 2, 5).Permute(0, 2, 3, 1, 4).Reshape(4, 6, 5)),
        };
        foreach (var (name, make) in cases)
        {
            var (on, off) = WithViews(make, y => Check(name.StartsWith("contiguous", StringComparison.Ordinal) ? !y.IsView : y.IsView, $"{name}: a view"));
            Check(BitsOf(on).SequenceEqual(BitsOf(off)), $"{name}: the view's elements are the copy's");
        }

        Check(TensorViews.Made - made >= cases.Length, $"views counted ({TensorViews.Made - made})");
        Check(BitsOf(x.ToArray()).SequenceEqual(BitsOf(values)), "the input unchanged");
    }

    private static void ViewGradients(Device device)
    {
        if (device.Type != DeviceType.Cpu)
        {
            return;
        }

        var random = new Random(5);
        float[] xs = [.. Enumerable.Range(0, 3 * 4 * 6).Select(_ => random.NextSingle() * 2 - 1)];
        float[] ws = [.. Enumerable.Range(0, 6 * 4).Select(_ => random.NextSingle() * 2 - 1)];
        var cases = new (string Name, Func<Tensor, Tensor, Tensor> Loss)[]
        {
            ("permute into a product", (x, w) => x.Permute(0, 2, 1).Reshape(18, 4).MatMul(w.Narrow(0, 0, 4)).Sum()),
            ("transpose read by the product", (x, w) => x.Reshape(12, 6).MatMul(w.Transpose(0, 1).Transpose(0, 1)).Square().Sum()),
            ("narrow and select into element-wise ops", (x, w) => (x.Narrow(2, 1, 4).Select(0, 2).Exp() * w.Narrow(0, 2, 4)).Sum()),
            ("expand", (x, w) => (x.Narrow(1, 0, 1).Expand(3, 4, 6) * x).Sum()),
            ("a chain into softmax", (x, w) => x.Transpose(1, 2).Narrow(1, 1, 3).Softmax().Square().Sum()),
            ("a product of two views", (x, w) => x.Select(0, 1).Transpose(0, 1).MatMul(w.Narrow(1, 0, 3).Transpose(0, 1).Transpose(0, 1).Narrow(0, 0, 4)).Sum()),
        };
        foreach (var (name, loss) in cases)
        {
            float[][] Gradients(bool views)
            {
                bool before = TensorViews.Enabled;
                TensorViews.Enabled = views;
                try
                {
                    using var scope = new TensorScope();
                    var x = Tensor.From(xs, [3, 4, 6], device, requiresGrad: true);
                    var w = Tensor.From(ws, [6, 4], device, requiresGrad: true);
                    loss(x, w).Backward();
                    return [x.Grad!.ToArray(), w.Grad?.ToArray() ?? []];
                }
                finally
                {
                    TensorViews.Enabled = before;
                }
            }

            var on = Gradients(true);
            var off = Gradients(false);
            Check(BitsOf(on[0]).SequenceEqual(BitsOf(off[0])) && BitsOf(on[1]).SequenceEqual(BitsOf(off[1])), $"{name}: gradients through views equal those through copies");
        }
    }

    private static void ViewsKeepTheirValues(Device device)
    {
        if (device.Type != DeviceType.Cpu)
        {
            return;
        }

        bool before = TensorViews.Enabled;
        TensorViews.Enabled = true;
        try
        {
            using var scope = new TensorScope();
            float[] values = [.. Enumerable.Range(0, 24).Select(i => (float)i)];
            var x = Tensor.From(values, [4, 6], device);
            var t = x.Transpose(0, 1);
            var n = x.Narrow(1, 2, 3);
            var expectedT = Enumerable.Range(0, 24).Select(i => (float)(i % 4 * 6 + i / 4)).ToArray();
            var expectedN = Enumerable.Range(0, 12).Select(i => (float)(i / 3 * 6 + 2 + i % 3)).ToArray();
            x.CopyFrom(new float[24]);                                   // the input written after the views were made
            Check(t.ToArray().SequenceEqual(expectedT) && n.ToArray().SequenceEqual(expectedN), "CopyFrom: the views keep their values");

            var y = Tensor.From(values, [4, 6], device);
            var s = y.Select(0, 3);
            device.Backend.Fill(y.Storage, 24, 7f);                       // a kernel writing through Storage
            Check(s.ToArray().SequenceEqual(values[18..]), "a kernel through Storage: the view keeps its values");

            var z = Tensor.From(values, [4, 6], device);
            var v = z.Narrow(0, 1, 2);
            device.Backend.Fill(v.Storage, 12, -1f);                      // a write to the view's storage
            Check(z.ToArray().SequenceEqual(values) && v.ToArray().All(e => e == -1f), "a write to the view's storage stays the view's");

            // Read in place: a transposed operand, element-wise ops, softmax, sums and concatenation keep no copy.
            var a = Tensor.From(values, [4, 6], device);
            long materialized = TensorViews.Materialized;
            var at = a.Transpose(0, 1);
            var product = at.MatMul(a);                                  // [6, 4] x [4, 6]
            var e = (a.Narrow(1, 1, 3) * 2f).Exp().Softmax();
            var sum = a.Permute(1, 0).Sum();
            var joined = Tensor.Concat([a.Narrow(0, 0, 2), a.Narrow(0, 2, 2)]);
            Check(TensorViews.Materialized == materialized && at.IsView && product.Shape.SequenceEqual([6, 6]) && e.Shape.SequenceEqual([4, 3])
                  && joined.ToArray().SequenceEqual(values) && MathF.Abs(sum.Item() - values.Sum()) < 1e-3f,
                $"read in place: {TensorViews.Materialized - materialized} views copied to their own");
            var expected = new float[36];
            for (int i = 0; i < 6; i++)
            {
                for (int j = 0; j < 6; j++)
                {
                    for (int r = 0; r < 4; r++)
                    {
                        expected[i * 6 + j] += values[r * 6 + i] * values[r * 6 + j];
                    }
                }
            }

            AssertClose(expected, product.ToArray(), 1e-5f, "the transposed view's product");
        }
        finally
        {
            TensorViews.Enabled = before;
        }
    }

    private static void ViewsOff(Device device)
    {
        _ = device;
        if (!FirstRun(nameof(ViewsOff)))
        {
            return;
        }

        bool before = TensorViews.Enabled;
        try
        {
            TensorViews.Enabled = false;
            using var scope = new TensorScope();
            var x = Tensor.From([.. Enumerable.Range(0, 12).Select(i => (float)i)], [3, 4], Device.Cpu);
            long made = TensorViews.Made;
            var copies = new[] { x.Transpose(0, 1), x.Narrow(0, 1, 1), x.Select(1, 2), x.Narrow(1, 0, 1).Expand(3, 5) };
            Check(TensorViews.Made == made && copies.All(c => !c.IsView), "views off: copies, no view made");
            TensorViews.Enabled = true;
            var minimal = Tensor.From([1f, 2f, 3f, 4f], [2, 2], Device.Get("minimal"));
            var onMinimal = minimal.Transpose(0, 1);
            Check(onMinimal.ToArray().SequenceEqual([1f, 3f, 2f, 4f]), "a device copying the CPU's capabilities: its views read the same");
        }
        finally
        {
            TensorViews.Enabled = before;
        }
    }

    // Runs `run` with views on or off, and counts the views copied (to their own or short-lived) while it ran.
    private static (T Result, long Copied) UnderViews<T>(bool views, Func<T> run)
    {
        bool before = TensorViews.Enabled;
        TensorViews.Enabled = views;
        try
        {
            long copied = TensorViews.Materialized + TensorViews.Scratch;
            var result = run();
            return (result, TensorViews.Materialized + TensorViews.Scratch - copied);
        }
        finally
        {
            TensorViews.Enabled = before;
        }
    }

    private static bool SameBits(float[][] a, float[][] b) => a.Length == b.Length && a.Zip(b).All(p => BitsOf(p.First).SequenceEqual(BitsOf(p.Second)));

    private static void AttentionReadsViews(Device device)
    {
        if (device.Type != DeviceType.Cpu)
        {
            return;
        }

        const int n = 2, t = 5, kvHeads = 2, group = 2, heads = kvHeads * group, d = 8, outputs = 6;
        var random = new Random(9);
        float[] Values(int count) => [.. Enumerable.Range(0, count).Select(_ => random.NextSingle() * 2 - 1)];
        float[] qs = Values(n * t * heads * d), ks = Values(n * t * kvHeads * d), vs = Values(n * t * kvHeads * d), ws = Values(heads * d * outputs);
        float[][] Run(bool views)
        {
            using var scope = new TensorScope();
            var qp = Tensor.From(qs, [n, t, heads * d], device, requiresGrad: true);
            var kp = Tensor.From(ks, [n, t, kvHeads * d], device, requiresGrad: true);
            var vp = Tensor.From(vs, [n, t, kvHeads * d], device, requiresGrad: true);
            var w = Tensor.From(ws, [heads * d, outputs], device, requiresGrad: true);

            // The decoder's head layout: the query heads sharing a key/value head stacked, keys and values by head.
            var q = qp.Reshape(n, t, kvHeads, group, d).Permute(0, 2, 3, 1, 4).Reshape(n * kvHeads, group * t, d);
            var k = kp.Reshape(n, t, kvHeads, d).Permute(0, 2, 1, 3).Reshape(n * kvHeads, t, d);
            var v = vp.Reshape(n, t, kvHeads, d).Permute(0, 2, 1, 3).Reshape(n * kvHeads, t, d);
            var y = Tensor.CausalAttention(q, k, v, Tensor.From([0f], [1], device), t, 1f / MathF.Sqrt(d));
            var merged = y.Reshape(n, kvHeads, group, t, d).Permute(0, 3, 1, 2, 4).Reshape(n, t, heads * d);
            var output = merged.MatMul(w);
            output.Square().Sum().Backward();
            if (views)
            {
                Check(q.IsView && k.IsView && v.IsView && merged.IsView, $"views on: the heads and the merged heads stay views ({q.IsView}, {k.IsView}, {v.IsView}, {merged.IsView})");
            }

            return [output.ToArray(), qp.Grad!.ToArray(), kp.Grad!.ToArray(), vp.Grad!.ToArray(), w.Grad!.ToArray()];
        }

        var (on, copied) = UnderViews(true, () => Run(true));
        var (off, _) = UnderViews(false, () => Run(false));
        Check(SameBits(on, off), "the attention and its gradients through views equal those through copies, bit for bit");
        Check(copied == 0, $"no view copied, to its own or short-lived ({copied})");
    }

    private static void ProductReadsViews(Device device)
    {
        if (device.Type != DeviceType.Cpu)
        {
            return;
        }

        var random = new Random(13);
        // Each case: the left operand's and the right operand's source shapes, and the product of views of them.
        var cases = new (string Name, int[] A, int[] B, Func<Tensor, Tensor, Tensor> Product)[]
        {
            ("few rows, a transposed right operand", [2, 6], [5, 6], (a, b) => a.MatMul(b.Transpose(0, 1))),
            ("few rows, a transposed right operand read transposed", [2, 6], [6, 5], (a, b) => a.MatMul(b.Transpose(0, 1), transposeB: true)),
            ("many rows, a transposed right operand read transposed", [40, 6], [6, 5], (a, b) => a.MatMul(b.Transpose(0, 1), transposeB: true)),
            ("many rows, a slice of the right operand", [24, 6], [6, 9], (a, b) => a.MatMul(b.Narrow(1, 2, 5))),
            ("batched, a permuted right operand", [3, 7, 6], [3, 5, 6], (a, b) => a.MatMul(b.Permute(0, 2, 1))),
            ("batched, a permuted right operand read transposed", [3, 20, 6], [3, 6, 5], (a, b) => a.MatMul(b.Permute(0, 2, 1), transposeB: true)),
            ("both operands views", [6, 20], [7, 6], (a, b) => a.Transpose(0, 1).Narrow(0, 1, 18).MatMul(b.Narrow(0, 1, 5).Transpose(0, 1))),
        };
        foreach (var (name, aShape, bShape, product) in cases)
        {
            float[] av = [.. Enumerable.Range(0, aShape.Aggregate(1, (x, y) => x * y)).Select(_ => random.NextSingle() * 2 - 1)];
            float[] bv = [.. Enumerable.Range(0, bShape.Aggregate(1, (x, y) => x * y)).Select(_ => random.NextSingle() * 2 - 1)];
            float[][] Run()
            {
                using var scope = new TensorScope();
                var a = Tensor.From(av, aShape, device, requiresGrad: true);
                var b = Tensor.From(bv, bShape, device, requiresGrad: true);
                var c = product(a, b);
                c.Square().Sum().Backward();
                return [c.ToArray(), a.Grad!.ToArray(), b.Grad!.ToArray()];
            }

            var (on, copied) = UnderViews(true, Run);
            var (off, _) = UnderViews(false, Run);
            Check(SameBits(on, off), $"{name}: the product and its gradients through views equal those through copies, bit for bit");
            Check(copied == 0, $"{name}: no view copied ({copied})");
        }
    }

    private static void ReleaseKeepsViewedValues(Device device)
    {
        if (device.Type != DeviceType.Cpu)
        {
            return;
        }

        bool before = TensorViews.Enabled;
        TensorViews.Enabled = true;
        try
        {
            using var scope = new TensorScope();
            float[] values = [.. Enumerable.Range(0, 12).Select(i => (float)i)];
            var x = Tensor.From(values, [3, 4], device, requiresGrad: true);
            var p = x * 2f;
            var v = p.Transpose(0, 1);
            float[] expected = [.. Enumerable.Range(0, 12).Select(i => values[i % 3 * 4 + i / 3] * 2f)];
            ActivationMemory.Release([v], p);
            Check(!p.Held.Evicted && v.IsView, "Release keeps a view's source");
            ActivationMemory.Release(p);
            Check(!p.Held.Evicted && v.IsView && v.ToArray().SequenceEqual(expected), "values released while a view reads them stay, and the view reads them");
            _ = v.Storage;                                                 // the view copied to its own
            Check(!v.IsView && p.Held.Evicted && v.ToArray().SequenceEqual(expected), "once the view has its own copy, the released values go");
        }
        finally
        {
            TensorViews.Enabled = before;
        }
    }
}
