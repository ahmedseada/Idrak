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
    ];

    private static readonly (string Name, Action<Device> Run)[] TensorViewGroup =
    [
        ("tensor views: permute, transpose, narrow, select, expand, a reshape of a slice and chains of them share their input's storage (views, counted) and equal the copies made with views off, element for element; Contiguous() copies once", ViewsEqualCopies),
        ("tensor views: gradients through views (permute, narrow, select, expand, chains, into products, element-wise ops, softmax and sums) equal those through the copies made with views off, bit for bit", ViewGradients),
        ("tensor views: a view keeps the values it was made with when its input is written afterwards (CopyFrom, Fill, a kernel through Storage); a write to a view's storage does not reach its input; products read a transposed view in place and element-wise ops, softmax, sums and concatenation read views through short-lived copies, none kept", ViewsKeepTheirValues),
        ("tensor views: views off (IDRAK_VIEWS=0, TensorViews.Enabled) make today's copies and no view; a device copying the CPU's capabilities (the minimal test device) reads its views the same", ViewsOff),
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
}
