// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Idrak.Abstraction.Devices;
using Idrak.Abstraction.Diagnostics;

namespace Idrak.Abstraction;

public sealed partial class Tensor
{
    // ---------------------------------------------------------------- operators

    /// <summary>Element-wise sum. A 1-D right operand whose length matches the last dimension is broadcast across rows (e.g. adding a bias).</summary>
    public static Tensor operator +(Tensor a, Tensor b) => Add(a, b);

    /// <summary>Element-wise difference of two tensors of the same shape.</summary>
    public static Tensor operator -(Tensor a, Tensor b) => ElementWise(BinaryOp.Sub, a, b);

    /// <summary>Element-wise (Hadamard) product of two tensors of the same shape. Use <see cref="MatMul"/> for matrix multiplication.</summary>
    public static Tensor operator *(Tensor a, Tensor b) => ElementWise(BinaryOp.Mul, a, b);

    /// <summary>Adds a constant to every element.</summary>
    public static Tensor operator +(Tensor a, float b) => a.Affine(1f, b);

    /// <summary>Adds a constant to every element.</summary>
    public static Tensor operator +(float a, Tensor b) => b.Affine(1f, a);

    /// <summary>Subtracts a constant from every element.</summary>
    public static Tensor operator -(Tensor a, float b) => a.Affine(1f, -b);

    /// <summary>Subtracts every element from a constant.</summary>
    public static Tensor operator -(float a, Tensor b) => b.Affine(-1f, a);

    /// <summary>Multiplies every element by a constant.</summary>
    public static Tensor operator *(Tensor a, float b) => a.Affine(b, 0f);

    /// <summary>Multiplies every element by a constant.</summary>
    public static Tensor operator *(float a, Tensor b) => b.Affine(a, 0f);

    /// <summary>Divides every element by a constant.</summary>
    public static Tensor operator /(Tensor a, float b) => a.Affine(1f / b, 0f);

    /// <summary>Negates every element.</summary>
    public static Tensor operator -(Tensor a) => a.Affine(-1f, 0f);

    // ---------------------------------------------------------------- activations

    /// <summary>Logistic sigmoid, 1 / (1 + e^-x), element-wise.</summary>
    public Tensor Sigmoid() => Unary(UnaryOp.Sigmoid);

    /// <summary>Hyperbolic tangent, element-wise.</summary>
    public Tensor Tanh() => Unary(UnaryOp.Tanh);

    /// <summary>Rectified linear unit, max(x, 0), element-wise.</summary>
    public Tensor Relu() => Unary(UnaryOp.Relu);

    /// <summary>x², element-wise.</summary>
    public Tensor Square() => Unary(UnaryOp.Square);

    /// <summary>|x|, element-wise.</summary>
    public Tensor Abs() => Unary(UnaryOp.Abs);

    /// <summary>
    /// Inverted dropout: zeroes each element with probability <paramref name="p"/> and scales the
    /// survivors by 1 / (1 - p). The mask is derived from <paramref name="seed"/>, so it costs no memory.
    /// </summary>
    public Tensor Dropout(float p, uint seed)
    {
        ThrowIfDisposed();
        if (p is < 0f or >= 1f)
        {
            throw new ArgumentOutOfRangeException(nameof(p), p, "Dropout probability must be in [0, 1).");
        }

        long start = Telemetry.Start(TelemetryLevel.Operations);
        var y = Empty(_shape, Device);
        Backend.Dropout(Storage, y.Storage, Size, p, seed);
        if (WillRecord(this))
        {
            var x = this;
            y.Record("dropout", g => x.Backend.DropoutBackward(g.Storage, x.GradStorage(), x.Size, p, seed), x);
        }

        return Traced("dropout", y, start);
    }

    /// <summary>residual + x.Dropout(p, seed) in one pass (a residual connection with dropout on its branch).</summary>
    internal static Tensor AddDropout(Tensor residual, Tensor x, float p, uint seed)
    {
        residual.ThrowIfDisposed();
        x.ThrowIfDisposed();
        CheckSameShape(residual, x);
        long start = Telemetry.Start(TelemetryLevel.Operations);
        var y = Empty(x._shape, x.Device);
        x.Backend.AddDropout(residual.Storage, x.Storage, y.Storage, x.Size, p, seed);
        if (WillRecord(residual, x))
        {
            y.Record("add_dropout", g =>
            {
                if (x.RequiresGrad)
                {
                    x.Backend.DropoutBackward(g.Storage, x.GradStorage(), x.Size, p, seed);
                }

                if (residual.RequiresGrad)
                {
                    residual.AddGradient(g, adopt: !ReferenceEquals(residual, x));   // last: it may take g's buffer
                }
            }, residual, x);
        }

        return Traced("add_dropout", y, start);
    }

    // ---------------------------------------------------------------- linear algebra and reductions

    /// <summary>Sum of all elements, as a scalar tensor.</summary>
    public Tensor Sum() => Reduce(1f);

    /// <summary>Mean of all elements, as a scalar tensor.</summary>
    public Tensor Mean() => Reduce(Size == 0 ? 0f : 1f / Size);

    /// <summary>
    /// Returns a tensor with the same data viewed with a different shape. The data is shared, not copied.
    /// One dimension may be -1 to infer it from the others.
    /// </summary>
    public Tensor Reshape(params ReadOnlySpan<int> shape)
    {
        ThrowIfDisposed();
        var resolved = shape.ToArray();
        int inferred = Array.IndexOf(resolved, -1);
        if (inferred >= 0)
        {
            resolved[inferred] = 1;
            int known = ElementCount(resolved);
            resolved[inferred] = known == 0 ? 0 : Size / known;
        }

        if (ElementCount(resolved) != Size)
        {
            throw new ArgumentException($"Cannot reshape {FormatShape(_shape)} ({Size} elements) to {FormatShape(shape)}.");
        }

        Storage.AddRef();
        var y = new Tensor(resolved, Storage, Device, track: true);
        if (WillRecord(this))
        {
            var x = this;
            y.Record("reshape", g => x.AddGradient(g, adopt: true), x);
        }

        return y;
    }

    // ---------------------------------------------------------------- implementations

    private static readonly string[] UnaryNames = ["sigmoid", "tanh", "relu", "square", "abs", "exp", "log", "gelu", "sqrt", "sin", "cos", "silu", "sign"];

    private Tensor Unary(UnaryOp op)
    {
        ThrowIfDisposed();
        long start = Telemetry.Start(TelemetryLevel.Operations);
        var y = Empty(_shape, Device);
        Backend.Unary(op, Storage, y.Storage, Size);
        if (WillRecord(this))
        {
            var x = this;
            y.Record(UnaryNames[(int)op], g => x.Backend.UnaryBackward(op, x.Storage, y.Storage, g.Storage, x.GradStorage(), x.Size), x);
        }

        return Traced(UnaryNames[(int)op], y, start);
    }

    /// <summary>y = alpha * x + beta.</summary>
    private Tensor Affine(float alpha, float beta)
    {
        ThrowIfDisposed();
        long start = Telemetry.Start(TelemetryLevel.Operations);
        var y = Empty(_shape, Device);
        Backend.Affine(Storage, y.Storage, Size, alpha, beta);
        if (WillRecord(this))
        {
            var x = this;
            y.Record("affine", g => x.Backend.Axpy(g.Storage, x.GradStorage(), x.Size, alpha), x);
        }

        return Traced("affine", y, start);
    }

    private Tensor Reduce(float scale)
    {
        ThrowIfDisposed();
        long start = Telemetry.Start(TelemetryLevel.Operations);
        var y = Empty([], Device);
        Backend.Sum(Storage, y.Storage, Size, scale);
        if (WillRecord(this))
        {
            var x = this;
            y.Record("sum", g => x.Backend.AddBroadcastScalar(g.Storage, x.GradStorage(), x.Size, scale), x);
        }

        return Traced("sum", y, start);
    }

    private static Tensor Add(Tensor a, Tensor b)
    {
        a.ThrowIfDisposed();
        b.ThrowIfDisposed();
        // b broadcasts when its shape equals the trailing dimensions of a (a bias [F] over [N, F], a mask [T, T] over [B, T, T]).
        bool rowBroadcast = b.Rank >= 1 && b.Rank < a.Rank && a.Shape[(a.Rank - b.Rank)..].SequenceEqual(b.Shape);
        if (!rowBroadcast)
        {
            return ElementWise(BinaryOp.Add, a, b);
        }

        CheckSameDevice(a, b);
        long start = Telemetry.Start(TelemetryLevel.Operations);
        int cols = b.Size;
        int rows = cols == 0 ? 0 : a.Size / cols;
        var c = Empty(a._shape, a.Device);
        a.Backend.AddRowVector(a.Storage, b.Storage, c.Storage, rows, cols);
        if (WillRecord(a, b))
        {
            c.Record("add_bias", g =>
            {
                if (b.RequiresGrad)
                {
                    b.Backend.SumRows(g.Storage, b.GradStorage(), rows, cols);
                }

                if (a.RequiresGrad)
                {
                    a.AddGradient(g, adopt: true);                  // last: a may take g's buffer
                }
            }, a, b);
        }

        return Traced("add_bias", c, start);
    }

    private static Tensor ElementWise(BinaryOp op, Tensor a, Tensor b)
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
                var backend = a.Backend;
                int n = a.Size;
                switch (op)
                {
                    case BinaryOp.Add:
                        // b first (added), then a may take g's buffer (x + x: a takes it, then... b is a, already done).
                        if (b.RequiresGrad && !ReferenceEquals(a, b)) b.AddGradient(g, adopt: false);
                        if (a.RequiresGrad) a.AddGradient(g, adopt: true);
                        if (b.RequiresGrad && ReferenceEquals(a, b)) backend.Axpy(g.Storage, a.GradStorage(), n, 1f);
                        break;
                    case BinaryOp.Sub:
                        if (b.RequiresGrad) backend.Axpy(g.Storage, b.GradStorage(), n, -1f);
                        if (a.RequiresGrad) a.AddGradient(g, adopt: !ReferenceEquals(a, b));
                        break;
                    case BinaryOp.Mul:
                        if (a.RequiresGrad) backend.MulAdd(g.Storage, b.Storage, a.GradStorage(), n);
                        if (b.RequiresGrad) backend.MulAdd(g.Storage, a.Storage, b.GradStorage(), n);
                        break;
                }
            }, a, b);
        }

        return Traced(BinaryNames[(int)op], c, start);
    }

    private static readonly string[] BinaryNames = ["add", "sub", "mul", "maximum", "minimum"];

    private static void CheckSameDevice(Tensor a, Tensor b)
    {
        if (a.Device != b.Device)
        {
            throw new InvalidOperationException($"Tensors are on different devices ({a.Device} and {b.Device}). Move one with .To(device).");
        }
    }

    private static void CheckSameShape(Tensor a, Tensor b)
    {
        if (!a.Shape.SequenceEqual(b.Shape))
        {
            throw new ArgumentException($"Shapes {FormatShape(a._shape)} and {FormatShape(b._shape)} do not match.");
        }
    }
}
