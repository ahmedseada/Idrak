// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Idrak.Backends;
using Idrak.Diagnostics;

namespace Idrak;

public sealed partial class Tensor
{
    // ---------------------------------------------------------------- element-wise math

    /// <summary>√x, element-wise (NaN for negative x).</summary>
    public Tensor Sqrt() => Unary(UnaryOp.Sqrt);

    /// <summary>sin(x), element-wise (x in radians).</summary>
    public Tensor Sin() => Unary(UnaryOp.Sin);

    /// <summary>cos(x), element-wise (x in radians).</summary>
    public Tensor Cos() => Unary(UnaryOp.Cos);

    /// <summary>Sigmoid linear unit (swish), x · sigmoid(x), element-wise.</summary>
    public Tensor Silu() => Unary(UnaryOp.Silu);

    /// <summary>
    /// The sign of each element: -1, 0 or 1. Its gradient is zero. Combined with <see cref="Relu"/> it makes a mask:
    /// <c>x.Sign().Relu()</c> is 1 where x &gt; 0 and 0 elsewhere (see <see cref="Where"/>).
    /// </summary>
    public Tensor Sign() => Unary(UnaryOp.Sign);

    /// <summary>
    /// x^<paramref name="exponent"/>, element-wise, as <see cref="MathF.Pow"/>: negative x give NaN unless the exponent is
    /// a whole number. <c>Pow(-1)</c> is the reciprocal, <c>Pow(0.5f)</c> the square root.
    /// </summary>
    public Tensor Pow(float exponent)
    {
        ThrowIfDisposed();
        long start = Telemetry.Start(TelemetryLevel.Operations);
        var y = Empty(_shape, Device);
        Backend.Pow(Storage, y.Storage, Size, exponent);
        if (WillRecord(this))
        {
            var x = this;
            y.Record("pow", g => x.Backend.PowBackward(x.Storage, g.Storage, x.GradStorage(), x.Size, exponent), x);
        }

        return Traced("pow", y, start);
    }

    /// <summary>
    /// Each element limited to [<paramref name="min"/>, <paramref name="max"/>]. The gradient passes where the element is
    /// within the range (its ends included) and is zero elsewhere. Either bound may be infinite.
    /// </summary>
    public Tensor Clamp(float min, float max)
    {
        if (!(min <= max))
        {
            throw new ArgumentException($"Clamp needs min <= max, but got [{min}, {max}].");
        }

        ThrowIfDisposed();
        long start = Telemetry.Start(TelemetryLevel.Operations);
        var y = Empty(_shape, Device);
        Backend.Clamp(Storage, y.Storage, Size, min, max);
        if (WillRecord(this))
        {
            var x = this;
            y.Record("clamp", g => x.Backend.ClampBackward(x.Storage, g.Storage, x.GradStorage(), x.Size, min, max), x);
        }

        return Traced("clamp", y, start);
    }

    /// <summary>max(x, <paramref name="value"/>), element-wise (the gradient passes where x ≥ value).</summary>
    public Tensor Maximum(float value) => Clamp(value, float.PositiveInfinity);

    /// <summary>min(x, <paramref name="value"/>), element-wise (the gradient passes where x ≤ value).</summary>
    public Tensor Minimum(float value) => Clamp(float.NegativeInfinity, value);

    /// <summary>
    /// The larger of this tensor's and <paramref name="other"/>'s elements (same shape). The gradient goes to the larger
    /// element; on ties, to this tensor's.
    /// </summary>
    public Tensor Maximum(Tensor other) => Extremum(BinaryOp.Maximum, this, other);

    /// <summary>
    /// The smaller of this tensor's and <paramref name="other"/>'s elements (same shape). The gradient goes to the smaller
    /// element; on ties, to this tensor's.
    /// </summary>
    public Tensor Minimum(Tensor other) => Extremum(BinaryOp.Minimum, this, other);

    /// <summary>
    /// Element-wise selection: <paramref name="a"/>'s element where <paramref name="condition"/>'s is non-zero,
    /// <paramref name="b"/>'s elsewhere (all three of the same shape). The gradient goes to the selected element; the
    /// condition gets none.
    /// </summary>
    /// <example><c>var leaky = Tensor.Where(x.Sign().Relu(), x, x * 0.01f);   // leaky ReLU</c></example>
    public static Tensor Where(Tensor condition, Tensor a, Tensor b)
    {
        condition.ThrowIfDisposed();
        a.ThrowIfDisposed();
        b.ThrowIfDisposed();
        CheckSameDevice(condition, a);
        CheckSameDevice(a, b);
        CheckSameShape(condition, a);
        CheckSameShape(a, b);
        long start = Telemetry.Start(TelemetryLevel.Operations);
        var y = Empty(a._shape, a.Device);
        a.Backend.Where(condition.Storage, a.Storage, b.Storage, y.Storage, a.Size);
        if (WillRecord(a, b))
        {
            y.Record("where", g =>
            {
                if (ReferenceEquals(a, b))
                {
                    a.AddGradient(g, adopt: false);                     // either way the gradient reaches the same tensor
                    return;
                }

                a.Backend.WhereBackward(condition.Storage, g.Storage, a.RequiresGrad ? a.GradStorage() : null, b.RequiresGrad ? b.GradStorage() : null, a.Size);
            }, a, b, condition);
        }

        return Traced("where", y, start);
    }

    /// <summary>Runs <paramref name="function"/>'s forward step and records it as one operation (see <see cref="DifferentiableFunction"/>).</summary>
    internal static Tensor Apply(DifferentiableFunction function, Tensor[] inputs)
    {
        foreach (var input in inputs)
        {
            ArgumentNullException.ThrowIfNull(input, nameof(inputs));
            input.ThrowIfDisposed();
            CheckSameDevice(inputs[0], input);
        }

        long start = Telemetry.Start(TelemetryLevel.Operations);
        Tensor output;
        using (var scope = new TensorScope())
        {
            using (Autograd.NoGrad())
            {
                output = function.Forward(inputs);
            }

            if (!scope.Owns(output))
            {
                output = output.Clone();                                    // an input, or a tensor made elsewhere: a result of its own
            }

            scope.Keep(output);
        }

        if (inputs.Any(input => input.RequiresGrad) && Autograd.IsEnabled)
        {
            var y = output;
            y.Record(function.Name, g =>
            {
                using var scope = new TensorScope();
                IReadOnlyList<Tensor?> gradients;
                using (Autograd.NoGrad())
                {
                    gradients = function.Backward(inputs, y, g);
                }

                for (int i = 0; i < inputs.Length; i++)
                {
                    if (!inputs[i].RequiresGrad || gradients[i] is not { } gradient)
                    {
                        continue;
                    }

                    if (gradient.Device != inputs[i].Device || !gradient.Shape.SequenceEqual(inputs[i].Shape))
                    {
                        throw new InvalidOperationException($"The backward step of {function.Name} returned a gradient of shape {FormatShape(gradient._shape)} on {gradient.Device} "
                            + $"for input {i}, of shape {FormatShape(inputs[i]._shape)} on {inputs[i].Device}.");
                    }

                    inputs[i].AddGradient(gradient, adopt: false);
                }
            }, inputs);
        }

        return Traced(function.Name, output, start);
    }

    private static Tensor Extremum(BinaryOp op, Tensor a, Tensor b)
    {
        a.ThrowIfDisposed();
        b.ThrowIfDisposed();
        CheckSameDevice(a, b);
        CheckSameShape(a, b);
        long start = Telemetry.Start(TelemetryLevel.Operations);
        var c = Empty(a._shape, a.Device);
        a.Backend.Binary(op, a.Storage, b.Storage, c.Storage, a.Size);
        if (WillRecord(a, b))
        {
            c.Record(BinaryNames[(int)op], g =>
            {
                if (ReferenceEquals(a, b))
                {
                    a.AddGradient(g, adopt: false);
                    return;
                }

                a.Backend.ExtremumBackward(op, a.Storage, b.Storage, g.Storage, a.RequiresGrad ? a.GradStorage() : null, b.RequiresGrad ? b.GradStorage() : null, a.Size);
            }, a, b);
        }

        return Traced(BinaryNames[(int)op], c, start);
    }
}
