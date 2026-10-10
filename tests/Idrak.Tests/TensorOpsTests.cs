// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Idrak;

// The element-wise operations beyond the activations (square root, trigonometry, SiLU, sign, powers, clamping,
// extremes, selection), operations with a backward step of one's own (Autograd.Function), and writing into an existing
// tensor in place.
internal static partial class Tests
{
    private static readonly (string Name, Action<Device> Run)[] TensorOpsGroup =
    [
        ("tensor ops: sqrt, sin, cos (also far from zero), silu, sign, pow, clamp, maximum, minimum and where match the host's math; on Vulkan and CUDA as kernels, without the host fallback", TensorOpsForward),
        ("gradient: sqrt, sin, cos, silu, sign", TensorOpsUnaryGradients),
        ("gradient: pow (fractional, odd, negative and zero exponents), clamp, scalar maximum and minimum", TensorOpsParameterGradients),
        ("gradient: element-wise maximum and minimum (both inputs, and one tensor on both sides), where", TensorOpsSelectionGradients),
        ("custom function: softplus and a two-input function match finite differences, pass gradients on to earlier operations and free their own tensors; bad gradients are refused", CustomFunctionGradients),
        ("in-place writes: CopyFrom, Fill, Scale, AddScaled; refused on recorded results and checked before writing; a backward step after a write refuses to run; Persistent tensors outlive scopes", InPlaceWrites),
    ];

    // Away from every kink the tests use (0, the clamp bounds -0.5 and 0.6, the scalar extremes 0.4 and -0.2, and the
    // crossing of x with 0.3 - 0.5x at 0.2), by more than the finite differences' step.
    private static readonly float[] OpValues = [-1.7f, -1.1f, -0.83f, -0.61f, -0.37f, -0.13f, 0.11f, 0.29f, 0.53f, 0.77f, 1.21f, 1.9f];

    private static void TensorOpsForward(Device device)
    {
        float[] values = [.. OpValues, 40f, -25.5f, 100.25f, 3.5f];
        float[] positive = [.. values.Select(v => MathF.Abs(v) + 0.25f)];
        float[] other = [.. values.Select(v => 0.3f - 0.5f * v)];
        float[] mask = [.. values.Select((_, i) => i % 3 == 0 ? 0f : i % 3 == 1 ? 1f : -2f)];
        int[] shape = [values.Length];
        long hostCalls = device.Backend.HostCalls;
        using var x = Tensor.From(values, shape, device);
        using var p = Tensor.From(positive, shape, device);
        using var o = Tensor.From(other, shape, device);
        using var m = Tensor.From(mask, shape, device);
        void Same(Func<float, float> expected, float[] input, Tensor actual, string what, float tolerance = 1e-5f)
        {
            AssertClose([.. input.Select(expected)], actual.ToArray(), tolerance, what);
        }

        Same(MathF.Sqrt, positive, p.Sqrt(), "sqrt");
        Same(MathF.Sin, values, x.Sin(), "sin", 1e-4f);
        Same(MathF.Cos, values, x.Cos(), "cos", 1e-4f);
        Same(v => v / (1f + MathF.Exp(-v)), values, x.Silu(), "silu");
        Same(v => MathF.Sign(v), values, x.Sign(), "sign");
        Same(v => MathF.Pow(v, 2.5f), positive, p.Pow(2.5f), "pow 2.5");
        Same(v => MathF.Pow(v, 3f), OpValues, Tensor.From(OpValues, [OpValues.Length], device).Pow(3f), "pow 3 (negative bases)");
        Same(v => MathF.Pow(v, 2f), OpValues, Tensor.From(OpValues, [OpValues.Length], device).Pow(2f), "pow 2 (negative bases)");
        Same(v => 1f / v, positive, p.Pow(-1f), "pow -1");
        Check(x.Pow(0.5f).ToArray().Select((v, i) => values[i] < 0 ? float.IsNaN(v) : MathF.Abs(v - MathF.Sqrt(values[i])) < 1e-4f * MathF.Max(1, MathF.Sqrt(values[i]))).All(b => b),
              "pow 0.5: NaN for negative bases");
        using var zeros = Tensor.Zeros([3], device);
        Check(zeros.Pow(0f).ToArray().All(v => v == 1f) && zeros.Pow(2f).ToArray().All(v => v == 0f) && zeros.Pow(-1f).ToArray().All(float.IsPositiveInfinity), "pow at zero: 1, 0, infinity");
        Same(v => Math.Clamp(v, -0.5f, 0.6f), values, x.Clamp(-0.5f, 0.6f), "clamp", 0f);
        Same(v => MathF.Max(v, 0.4f), values, x.Maximum(0.4f), "maximum (scalar)", 0f);
        Same(v => MathF.Min(v, -0.2f), values, x.Minimum(-0.2f), "minimum (scalar)", 0f);
        AssertClose([.. values.Select((v, i) => MathF.Max(v, other[i]))], x.Maximum(o).ToArray(), 0f, "maximum");
        AssertClose([.. values.Select((v, i) => MathF.Min(v, other[i]))], x.Minimum(o).ToArray(), 0f, "minimum");
        AssertClose([.. values.Select((v, i) => mask[i] != 0f ? v : other[i])], Tensor.Where(m, x, o).ToArray(), 0f, "where");

        // The backward steps too, so the check below covers every kernel.
        using (var g = Tensor.From(values, shape, device, requiresGrad: true))
        {
            var loss = g.Sqrt().Sum() + g.Sin().Sum() + g.Cos().Sum() + g.Silu().Sum() + g.Sign().Sum() + g.Pow(3f).Sum() + g.Clamp(-0.5f, 0.6f).Sum()
                       + g.Maximum(o).Sum() + g.Minimum(o).Sum() + Tensor.Where(m, g, o).Sum();
            loss.Backward();
            Check(g.Grad is not null, "the backward steps ran");
        }

        if (device.Type is DeviceType.Vulkan or DeviceType.Cuda)
        {
            Check(device.Backend.HostCalls == hostCalls, $"every operation ran as a {device.Type} kernel ({device.Backend.HostCalls - hostCalls} host fallbacks)");
        }

        try
        {
            x.Clamp(1f, -1f);
            Check(false, "Clamp with min > max is refused");
        }
        catch (ArgumentException)
        {
        }
    }

    private static void TensorOpsUnaryGradients(Device device)
    {
        GradCheckAt(device, OpValues, x => (x + 2f).Sqrt().Sum());
        GradCheckAt(device, [.. OpValues.Select(v => v * 3f)], x => x.Sin().Sum());
        GradCheckAt(device, [.. OpValues.Select(v => v * 3f)], x => (x.Cos() * x).Sum());
        GradCheckAt(device, [.. OpValues.Select(v => v * 4f)], x => x.Silu().Sum());
        GradCheckAt(device, OpValues, x => (x.Sign() * x).Sum());               // sign's own gradient is zero
    }

    private static void TensorOpsParameterGradients(Device device)
    {
        GradCheckAt(device, OpValues, x => (x + 2f).Pow(2.5f).Sum());
        GradCheckAt(device, OpValues, x => x.Pow(3f).Sum());
        GradCheckAt(device, OpValues, x => (x + 2f).Pow(-1f).Sum());
        GradCheckAt(device, OpValues, x => (x.Pow(0f) * x).Sum());
        GradCheckAt(device, OpValues, x => (x.Clamp(-0.5f, 0.6f) * x).Sum());
        GradCheckAt(device, OpValues, x => x.Maximum(0.4f).Square().Sum());
        GradCheckAt(device, OpValues, x => x.Minimum(-0.2f).Square().Sum());
    }

    private static void TensorOpsSelectionGradients(Device device)
    {
        GradCheckAt(device, OpValues, x => x.Maximum(0.3f - 0.5f * x).Square().Sum());
        GradCheckAt(device, OpValues, x => x.Minimum(0.3f - 0.5f * x).Square().Sum());
        GradCheckAt(device, OpValues, x => x.Maximum(x).Square().Sum());
        using var m = Tensor.From([.. OpValues.Select((_, i) => i % 2 == 0 ? 1f : 0f)], [OpValues.Length], device);
        GradCheckAt(device, OpValues, x => Tensor.Where(m, x.Square(), x * 3f).Sum());
        GradCheckAt(device, OpValues, x => Tensor.Where(m, x, x).Square().Sum());

        // Only one side requires gradients (the tensors a check's function reads must outlive its backward step).
        using var constant = Tensor.From([.. OpValues.Select(v => 0.3f - 0.5f * v)], [OpValues.Length], device);
        GradCheckAt(device, OpValues, x => constant.Maximum(x).Square().Sum());
    }

    private static void CustomFunctionGradients(Device device)
    {
        var softplus = Autograd.Function("softplus", x => (x[0].Exp() + 1f).Log(), (x, y, g) => [g * x[0].Sigmoid()]);
        Check(softplus.Name == "softplus", "its name");
        GradCheckAt(device, [.. OpValues.Select(v => v * 3f)], x => softplus.Apply(x).Sum());
        GradCheckAt(device, OpValues, x => softplus.Apply(x * 2f + 0.5f).Square().Mean());   // through earlier and later operations

        // Two inputs, one of them not differentiated: f(a, b) = a² · b, df/da = 2ab, df/db = a².
        var product = Autograd.Function("square-times", x => x[0].Square() * x[1], (x, y, g) => [g * x[0] * x[1] * 2f, g * x[0].Square()]);
        using var other = Tensor.From([.. OpValues.Select(v => 0.7f - v)], [OpValues.Length], device);
        GradCheckAt(device, OpValues, a => product.Apply(a, other).Sum());
        GradCheckAt(device, OpValues, b => product.Apply(other, b).Sum());
        GradCheckAt(device, OpValues, x => product.Apply(x, x).Sum());               // the same tensor twice: 3x²

        // Without gradients it only runs forward; an input returned as the result is copied.
        using (var scope = new TensorScope())
        {
            var x = Tensor.From(OpValues, [OpValues.Length], device);
            var identity = Autograd.Function("identity", v => v[0], (v, y, g) => [g]);
            var y = identity.Apply(x);
            Check(!ReferenceEquals(x, y) && y.ToArray().SequenceEqual(OpValues) && !y.RequiresGrad, "identity without gradients: a copy");
            var xg = Tensor.From(OpValues, [OpValues.Length], device, requiresGrad: true);
            var yg = identity.Apply(xg);
            Check(yg.RequiresGrad && !yg.IsLeaf, "identity with gradients: recorded");
            (yg * 3f).Sum().Backward();
            Check(xg.Grad!.ToArray().All(v => v == 3f), "identity's gradient");
        }

        // The forward step's own tensors are freed; the result stays in the enclosing scope.
        Tensor kept;
        using (var scope = new TensorScope())
        {
            var x = Tensor.From(OpValues, [OpValues.Length], device);
            kept = softplus.Apply(x);
            Check(scope.Owns(kept), "the result belongs to the enclosing scope");
        }

        Check(kept.IsDisposed, "and is disposed with it");

        // A wrong number of gradients, or a wrong shape, is refused.
        foreach (var (name, bad) in new (string, DifferentiableFunction)[]
        {
            ("count", Autograd.Function("bad-count", x => x[0] * 2f, (x, y, g) => [g, g])),
            ("shape", Autograd.Function("bad-shape", x => x[0] * 2f, (x, y, g) => [g.Reshape(3, -1)])),
        })
        {
            using var scope = new TensorScope();
            var x = Tensor.From(OpValues, [OpValues.Length], device, requiresGrad: true);
            try
            {
                bad.Apply(x).Sum().Backward();
                Check(false, $"a gradient of the wrong {name} is refused");
            }
            catch (InvalidOperationException ex)
            {
                Check(ex.Message.Contains(name == "count" ? "bad-count" : "bad-shape", StringComparison.Ordinal), ex.Message);
            }
        }
    }

    private static void InPlaceWrites(Device device)
    {
        using var t = Tensor.Zeros([2, 3], device);
        t.CopyFrom([1f, 2f, 3f, 4f, 5f, 6f]);
        Check(t.ToArray().SequenceEqual([1f, 2f, 3f, 4f, 5f, 6f]), "CopyFrom values");
        t.Scale(2f);
        Check(t.ToArray().SequenceEqual([2f, 4f, 6f, 8f, 10f, 12f]), "Scale");
        using var other = Tensor.Ones([6], device);
        t.AddScaled(other, -2f);
        Check(t.ToArray().SequenceEqual([0f, 2f, 4f, 6f, 8f, 10f]), "AddScaled (same size, another shape)");
        using var cpu = Tensor.From([6f, 5f, 4f, 3f, 2f, 1f], [3, 2], Device.Cpu);
        t.CopyFrom(cpu);
        Check(t.ToArray().SequenceEqual([6f, 5f, 4f, 3f, 2f, 1f]), "CopyFrom a tensor (from the CPU)");
        t.Fill(0.5f);
        Check(t.ToArray().All(v => v == 0.5f), "Fill");

        // Wrong sizes are refused before anything is written or counted.
        int version = t.Storage.Version;
        foreach (var write in new Action[] { () => t.CopyFrom([1f]), () => t.AddScaled(Tensor.Ones([5], device)), () => t.CopyFrom(Tensor.Ones([7], device)) })
        {
            try
            {
                write();
                Check(false, "a write of the wrong size is refused");
            }
            catch (ArgumentException)
            {
            }
        }

        Check(t.Storage.Version == version && t.ToArray().All(v => v == 0.5f), "a refused write changes nothing");

        // A recorded result cannot be written; a leaf that requires gradients can.
        using (var scope = new TensorScope())
        {
            var w = Tensor.From([1f, 2f, 3f], device: device, requiresGrad: true);
            var y = w * 2f;
            try
            {
                y.Fill(0f);
                Check(false, "a recorded result is refused");
            }
            catch (InvalidOperationException ex)
            {
                Check(ex.Message.Contains("recorded operation", StringComparison.Ordinal), ex.Message);
            }

            // A write after the operation recorded its use of w: the backward step refuses to compute from the new values.
            var loss = (w * w).Sum();
            w.AddScaled(Tensor.Ones([3], device), 1f);
            try
            {
                loss.Backward();
                Check(false, "a backward step after an in-place write refuses to run");
            }
            catch (InvalidOperationException ex)
            {
                Check(ex.Message.Contains("changed in place", StringComparison.Ordinal), ex.Message);
            }

            // Writing after Backward (as an optimizer does) is fine, and so is writing into a detached copy of a value
            // nothing reads backward.
            var w2 = Tensor.From([1f, 2f, 3f], device: device, requiresGrad: true);
            (w2 * w2).Sum().Backward();
            w2.AddScaled(w2.Grad!, -0.25f);
            Check(w2.ToArray().SequenceEqual([0.5f, 1f, 1.5f]), "a gradient step in place after Backward");
        }

        // Persistent tensors stay alive after a scope; ordinary ones are disposed with it.
        Tensor persistent, zeros, ordinary;
        using (new TensorScope())
        {
            persistent = Tensor.Persistent([1f, 2f], [2], device, requiresGrad: true);
            zeros = Tensor.PersistentZeros([2, 2], device);
            ordinary = Tensor.From([1f, 2f], [2], device);
        }

        Check(!persistent.IsDisposed && persistent.RequiresGrad && persistent.ToArray().SequenceEqual([1f, 2f]) && zeros.ToArray().All(v => v == 0f) && ordinary.IsDisposed,
              "Persistent tensors outlive the scope");
        persistent.Dispose();
        zeros.Dispose();
    }

    /// <summary>Compares autograd's gradient of f at <paramref name="values"/> with central finite differences.</summary>
    private static void GradCheckAt(Device device, float[] values, Func<Tensor, Tensor> f, float tolerance = 2e-2f)
    {
        using var scope = new TensorScope();
        int[] shape = [values.Length];
        var x = Tensor.From(values, shape, device, requiresGrad: true);
        f(x).Backward();
        var analytic = x.Grad?.ToArray() ?? new float[values.Length];
        const float h = 1e-2f;
        var numeric = new float[values.Length];
        using (Autograd.NoGrad())
        {
            for (int i = 0; i < values.Length; i++)
            {
                var plus = (float[])values.Clone();
                var minus = (float[])values.Clone();
                plus[i] += h;
                minus[i] -= h;
                numeric[i] = (f(Tensor.From(plus, shape, device)).Item() - f(Tensor.From(minus, shape, device)).Item()) / (2 * h);
            }
        }

        AssertClose(numeric, analytic, tolerance, "gradient");
    }
}
