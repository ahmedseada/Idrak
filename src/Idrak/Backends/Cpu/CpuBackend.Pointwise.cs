// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

namespace Idrak.Backends.Cpu;

// The element-wise operations beyond the activations: square roots, trigonometry, SiLU and sign, powers, clamping,
// element-wise extremes and selection by a mask, with their backward steps. Scalar loops through MathF (exact to its
// rounding), split across cores like the other element-wise kernels; the reference the GPU kernels are checked against.
internal sealed partial class CpuBackend
{
    public override void ExtremumBackward(BinaryOp op, Storage a, Storage b, Storage dy, Storage? da, Storage? db, int n) =>
        Run(new ExtremumBackwardKernel(op == BinaryOp.Minimum, D(a), D(b), D(dy), da is null ? null : D(da), db is null ? null : D(db)), n);

    public override void Pow(Storage x, Storage y, int n, float exponent) => Run(new PowKernel(D(x), D(y), exponent), n);

    public override void PowBackward(Storage x, Storage dy, Storage dx, int n, float exponent) => Run(new PowBackwardKernel(D(x), D(dy), D(dx), exponent), n);

    public override void Clamp(Storage x, Storage y, int n, float min, float max) => Run(new ClampKernel(D(x), D(y), min, max), n);

    public override void ClampBackward(Storage x, Storage dy, Storage dx, int n, float min, float max) => Run(new ClampBackwardKernel(D(x), D(dy), D(dx), min, max), n);

    public override void Where(Storage condition, Storage a, Storage b, Storage y, int n) => Run(new WhereKernel(D(condition), D(a), D(b), D(y)), n);

    public override void WhereBackward(Storage condition, Storage dy, Storage? da, Storage? db, int n) =>
        Run(new WhereBackwardKernel(D(condition), D(dy), da is null ? null : D(da), db is null ? null : D(db)), n);

    // sign(x): -1, 0 or 1, and NaN for NaN (MathF.Sign throws on NaN).
    private static float SignOf(float x) => x > 0f ? 1f : x < 0f ? -1f : x;

    private static float SigmoidOf(float x) => 1f / (1f + MathF.Exp(-x));

    private readonly struct MathKernel(UnaryOp op, float[] x, float[] y) : IRangeKernel
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
    private readonly struct MathBackwardKernel(UnaryOp op, float[] x, float[] y, float[] dy, float[] dx) : IRangeKernel
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

    private readonly struct ExtremumKernel(bool minimum, float[] a, float[] b, float[] c) : IRangeKernel
    {
        public void Execute(int start, int end)
        {
            if (minimum)
            {
                for (int i = start; i < end; i++) c[i] = MathF.Min(a[i], b[i]);
            }
            else
            {
                for (int i = start; i < end; i++) c[i] = MathF.Max(a[i], b[i]);
            }
        }
    }

    // Ties go to a.
    private readonly struct ExtremumBackwardKernel(bool minimum, float[] a, float[] b, float[] dy, float[]? da, float[]? db) : IRangeKernel
    {
        public void Execute(int start, int end)
        {
            for (int i = start; i < end; i++)
            {
                bool first = minimum ? a[i] <= b[i] : a[i] >= b[i];
                if (da is not null && first) da[i] += dy[i];
                if (db is not null && !first) db[i] += dy[i];
            }
        }
    }

    private readonly struct PowKernel(float[] x, float[] y, float exponent) : IRangeKernel
    {
        public void Execute(int start, int end)
        {
            for (int i = start; i < end; i++) y[i] = MathF.Pow(x[i], exponent);
        }
    }

    private readonly struct PowBackwardKernel(float[] x, float[] dy, float[] dx, float exponent) : IRangeKernel
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

    private readonly struct ClampKernel(float[] x, float[] y, float min, float max) : IRangeKernel
    {
        public void Execute(int start, int end)
        {
            for (int i = start; i < end; i++) y[i] = MathF.Min(MathF.Max(x[i], min), max);
        }
    }

    private readonly struct ClampBackwardKernel(float[] x, float[] dy, float[] dx, float min, float max) : IRangeKernel
    {
        public void Execute(int start, int end)
        {
            for (int i = start; i < end; i++)
            {
                if (x[i] >= min && x[i] <= max) dx[i] += dy[i];
            }
        }
    }

    private readonly struct WhereKernel(float[] condition, float[] a, float[] b, float[] y) : IRangeKernel
    {
        public void Execute(int start, int end)
        {
            for (int i = start; i < end; i++) y[i] = condition[i] != 0f ? a[i] : b[i];
        }
    }

    private readonly struct WhereBackwardKernel(float[] condition, float[] dy, float[]? da, float[]? db) : IRangeKernel
    {
        public void Execute(int start, int end)
        {
            for (int i = start; i < end; i++)
            {
                bool first = condition[i] != 0f;
                if (da is not null && first) da[i] += dy[i];
                if (db is not null && !first) db[i] += dy[i];
            }
        }
    }
}
