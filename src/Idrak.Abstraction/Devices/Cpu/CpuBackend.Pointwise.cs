// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Numerics;
using System.Runtime.InteropServices;

namespace Idrak.Abstraction.Devices.Cpu;

// The element-wise operations beyond the activations: square roots, trigonometry, SiLU and sign, powers, clamping,
// element-wise extremes and selection by a mask, with their backward steps. Scalar loops through MathF (exact to its
// rounding), split across cores like the other element-wise kernels; the reference the GPU kernels are checked against.
// Extremes, clamping and selection (and their backward steps) also run on whole vectors: Vector.Min/Max follow
// MathF.Min/Max on every input (NaN, signed zeros), and a gradient is added only where it lands, so the bits are the same.
internal sealed partial class CpuBackend
{
    public override void ExtremumBackwardKernel(BinaryOp op, Storage a, Storage b, Storage dy, Storage? da, Storage? db, int n) =>
        Run(new ExtremumBackwardLoop(op == BinaryOp.Minimum, D(a), D(b), D(dy), da is null ? null : D(da), db is null ? null : D(db)), n);

    public override void PowKernel(Storage x, Storage y, int n, float exponent) => Run(new PowLoop(D(x), D(y), exponent), n);

    public override void PowBackwardKernel(Storage x, Storage dy, Storage dx, int n, float exponent) => Run(new PowBackwardLoop(D(x), D(dy), D(dx), exponent), n);

    public override void ClampKernel(Storage x, Storage y, int n, float min, float max) => Run(new ClampLoop(D(x), D(y), min, max), n);

    public override void ClampBackwardKernel(Storage x, Storage dy, Storage dx, int n, float min, float max) => Run(new ClampBackwardLoop(D(x), D(dy), D(dx), min, max), n);

    public override void WhereKernel(Storage condition, Storage a, Storage b, Storage y, int n) => Run(new WhereLoop(D(condition), D(a), D(b), D(y)), n);

    public override void WhereBackwardKernel(Storage condition, Storage dy, Storage? da, Storage? db, int n) =>
        Run(new WhereBackwardLoop(D(condition), D(dy), da is null ? null : D(da), db is null ? null : D(db)), n);

    // sign(x): -1, 0 or 1, and NaN for NaN (MathF.Sign throws on NaN).
    private static float SignOf(float x) => x > 0f ? 1f : x < 0f ? -1f : x;

    private static float SigmoidOf(float x) => 1f / (1f + MathF.Exp(-x));

    private readonly struct MathLoop(UnaryOp op, float[] x, float[] y) : IRangeKernel
    {
        public void Execute(int start, int end)
        {
            switch (op)
            {
                case UnaryOp.Sqrt:
                    for (int i = start; i < end; i++) y[i] = MathF.Sqrt(x[i]);
                    break;
                case UnaryOp.Sin:
                    for (int i = start; i < end; i++) y[i] = MathF.Sin(x[i]);
                    break;
                case UnaryOp.Cos:
                    for (int i = start; i < end; i++) y[i] = MathF.Cos(x[i]);
                    break;
                case UnaryOp.Silu:
                    for (int i = start; i < end; i++) y[i] = x[i] * SigmoidOf(x[i]);
                    break;
                case UnaryOp.Sign:
                    for (int i = start; i < end; i++) y[i] = SignOf(x[i]);
                    break;
                default:
                    throw new ArgumentOutOfRangeException(nameof(op));
            }
        }
    }

    // dx += dy · op'(x); sqrt's slope comes from y = sqrt(x), sign's is zero.
    private readonly struct MathBackwardLoop(UnaryOp op, float[] x, float[] y, float[] dy, float[] dx) : IRangeKernel
    {
        public void Execute(int start, int end)
        {
            switch (op)
            {
                case UnaryOp.Sqrt:
                    for (int i = start; i < end; i++) dx[i] += dy[i] * 0.5f / y[i];
                    break;
                case UnaryOp.Sin:
                    for (int i = start; i < end; i++) dx[i] += dy[i] * MathF.Cos(x[i]);
                    break;
                case UnaryOp.Cos:
                    for (int i = start; i < end; i++) dx[i] -= dy[i] * MathF.Sin(x[i]);
                    break;
                case UnaryOp.Silu:
                    for (int i = start; i < end; i++)
                    {
                        float s = SigmoidOf(x[i]);
                        dx[i] += dy[i] * s * (1f + x[i] * (1f - s));
                    }

                    break;
                case UnaryOp.Sign:
                    break;
                default:
                    throw new ArgumentOutOfRangeException(nameof(op));
            }
        }
    }

    private readonly struct ExtremumLoop(bool minimum, float[] a, float[] b, float[] c) : IRangeKernel
    {
        public void Execute(int start, int end)
        {
            int i = start;
            if (Vector.IsHardwareAccelerated)
            {
                var av = MemoryMarshal.Cast<float, Vector<float>>(a.AsSpan(start, end - start));
                var bv = MemoryMarshal.Cast<float, Vector<float>>(b.AsSpan(start, end - start));
                var cv = MemoryMarshal.Cast<float, Vector<float>>(c.AsSpan(start, end - start));
                for (int v = 0; v < cv.Length; v++)
                {
                    cv[v] = minimum ? Vector.Min(av[v], bv[v]) : Vector.Max(av[v], bv[v]);
                }

                i += cv.Length * Vector<float>.Count;
            }

            if (minimum)
            {
                for (; i < end; i++) c[i] = MathF.Min(a[i], b[i]);
            }
            else
            {
                for (; i < end; i++) c[i] = MathF.Max(a[i], b[i]);
            }
        }
    }

    // Ties go to a.
    private readonly struct ExtremumBackwardLoop(bool minimum, float[] a, float[] b, float[] dy, float[]? da, float[]? db) : IRangeKernel
    {
        public void Execute(int start, int end)
        {
            int i = start;
            if (Vector.IsHardwareAccelerated)
            {
                int length = end - start;
                var av = MemoryMarshal.Cast<float, Vector<float>>(a.AsSpan(start, length));
                var bv = MemoryMarshal.Cast<float, Vector<float>>(b.AsSpan(start, length));
                var gv = MemoryMarshal.Cast<float, Vector<float>>(dy.AsSpan(start, length));
                var dav = da is null ? default : MemoryMarshal.Cast<float, Vector<float>>(da.AsSpan(start, length));
                var dbv = db is null ? default : MemoryMarshal.Cast<float, Vector<float>>(db.AsSpan(start, length));
                for (int v = 0; v < gv.Length; v++)
                {
                    var first = minimum ? Vector.LessThanOrEqual(av[v], bv[v]) : Vector.GreaterThanOrEqual(av[v], bv[v]);
                    if (da is not null) dav[v] = Vector.ConditionalSelect(first, dav[v] + gv[v], dav[v]);
                    if (db is not null) dbv[v] = Vector.ConditionalSelect(first, dbv[v], dbv[v] + gv[v]);
                }

                i += gv.Length * Vector<float>.Count;
            }

            for (; i < end; i++)
            {
                bool first = minimum ? a[i] <= b[i] : a[i] >= b[i];
                if (da is not null && first) da[i] += dy[i];
                if (db is not null && !first) db[i] += dy[i];
            }
        }
    }

    private readonly struct PowLoop(float[] x, float[] y, float exponent) : IRangeKernel
    {
        public void Execute(int start, int end)
        {
            for (int i = start; i < end; i++) y[i] = MathF.Pow(x[i], exponent);
        }
    }

    private readonly struct PowBackwardLoop(float[] x, float[] dy, float[] dx, float exponent) : IRangeKernel
    {
        public void Execute(int start, int end)
        {
            if (exponent == 0f)
            {
                return;                                                     // a constant: no gradient (not 0 · x^-1, NaN at 0)
            }

            for (int i = start; i < end; i++) dx[i] += dy[i] * exponent * MathF.Pow(x[i], exponent - 1f);
        }
    }

    private readonly struct ClampLoop(float[] x, float[] y, float min, float max) : IRangeKernel
    {
        public void Execute(int start, int end)
        {
            int i = start;
            if (Vector.IsHardwareAccelerated)
            {
                var xv = MemoryMarshal.Cast<float, Vector<float>>(x.AsSpan(start, end - start));
                var yv = MemoryMarshal.Cast<float, Vector<float>>(y.AsSpan(start, end - start));
                Vector<float> low = new(min), high = new(max);
                for (int v = 0; v < yv.Length; v++)
                {
                    yv[v] = Vector.Min(Vector.Max(xv[v], low), high);
                }

                i += yv.Length * Vector<float>.Count;
            }

            for (; i < end; i++) y[i] = MathF.Min(MathF.Max(x[i], min), max);
        }
    }

    private readonly struct ClampBackwardLoop(float[] x, float[] dy, float[] dx, float min, float max) : IRangeKernel
    {
        public void Execute(int start, int end)
        {
            int i = start;
            if (Vector.IsHardwareAccelerated)
            {
                var xv = MemoryMarshal.Cast<float, Vector<float>>(x.AsSpan(start, end - start));
                var gv = MemoryMarshal.Cast<float, Vector<float>>(dy.AsSpan(start, end - start));
                var dv = MemoryMarshal.Cast<float, Vector<float>>(dx.AsSpan(start, end - start));
                Vector<float> low = new(min), high = new(max);
                for (int v = 0; v < dv.Length; v++)
                {
                    var inside = Vector.GreaterThanOrEqual(xv[v], low) & Vector.LessThanOrEqual(xv[v], high);
                    dv[v] = Vector.ConditionalSelect(inside, dv[v] + gv[v], dv[v]);
                }

                i += dv.Length * Vector<float>.Count;
            }

            for (; i < end; i++)
            {
                if (x[i] >= min && x[i] <= max) dx[i] += dy[i];
            }
        }
    }

    private readonly struct WhereLoop(float[] condition, float[] a, float[] b, float[] y) : IRangeKernel
    {
        public void Execute(int start, int end)
        {
            int i = start;
            if (Vector.IsHardwareAccelerated)
            {
                int length = end - start;
                var cv = MemoryMarshal.Cast<float, Vector<float>>(condition.AsSpan(start, length));
                var av = MemoryMarshal.Cast<float, Vector<float>>(a.AsSpan(start, length));
                var bv = MemoryMarshal.Cast<float, Vector<float>>(b.AsSpan(start, length));
                var yv = MemoryMarshal.Cast<float, Vector<float>>(y.AsSpan(start, length));
                for (int v = 0; v < yv.Length; v++)
                {
                    yv[v] = Vector.ConditionalSelect(Vector.Equals(cv[v], Vector<float>.Zero), bv[v], av[v]);   // NaN selects a
                }

                i += yv.Length * Vector<float>.Count;
            }

            for (; i < end; i++) y[i] = condition[i] != 0f ? a[i] : b[i];
        }
    }

    private readonly struct WhereBackwardLoop(float[] condition, float[] dy, float[]? da, float[]? db) : IRangeKernel
    {
        public void Execute(int start, int end)
        {
            int i = start;
            if (Vector.IsHardwareAccelerated)
            {
                int length = end - start;
                var cv = MemoryMarshal.Cast<float, Vector<float>>(condition.AsSpan(start, length));
                var gv = MemoryMarshal.Cast<float, Vector<float>>(dy.AsSpan(start, length));
                var dav = da is null ? default : MemoryMarshal.Cast<float, Vector<float>>(da.AsSpan(start, length));
                var dbv = db is null ? default : MemoryMarshal.Cast<float, Vector<float>>(db.AsSpan(start, length));
                for (int v = 0; v < gv.Length; v++)
                {
                    var second = Vector.Equals(cv[v], Vector<float>.Zero);
                    if (da is not null) dav[v] = Vector.ConditionalSelect(second, dav[v], dav[v] + gv[v]);
                    if (db is not null) dbv[v] = Vector.ConditionalSelect(second, dbv[v] + gv[v], dbv[v]);
                }

                i += gv.Length * Vector<float>.Count;
            }

            for (; i < end; i++)
            {
                bool first = condition[i] != 0f;
                if (da is not null && first) da[i] += dy[i];
                if (db is not null && !first) db[i] += dy[i];
            }
        }
    }
}
