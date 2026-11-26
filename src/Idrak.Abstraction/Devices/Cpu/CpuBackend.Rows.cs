// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Runtime.CompilerServices;

namespace Idrak.Abstraction.Devices.Cpu;

// The operations a row at a time: softmax, the normalizations, the rotary embedding, a row's bias and its arg-max, each
// written once over the float lanes (Rows): a row's sums in the sixteen canonical lanes, its maps by the row's width, so
// every width gives the same bits.
internal sealed unsafe partial class CpuBackend
{
    // ------------------------------------------------------------------ maps inside a row

    // y = e^(x - shift), returning nothing (the sum is taken apart).
    private struct ExpShiftMap(float* x, float* y, float shift) : IRowMap
    {
        public readonly void Step<TV, TF>(nint i) where TV : unmanaged where TF : struct, IFloats<TV> =>
            TF.Store(y + i, Fn<TV, TF>.Exp(TF.Sub(TF.Load(x + i), TF.All(shift))));
    }

    // Σ e^(x - shift).
    private struct ExpSumTerms(float* x, float shift) : ITerms
    {
        public readonly TV Take<TV, TF>(TV sum, nint i) where TV : unmanaged where TF : struct, IFloats<TV> =>
            TF.Add(sum, Fn<TV, TF>.Exp(TF.Sub(TF.Load(x + i), TF.All(shift))));
    }

    // y = x · scale.
    private struct ScaleMap(float* x, float* y, float scale) : IRowMap
    {
        public readonly void Step<TV, TF>(nint i) where TV : unmanaged where TF : struct, IFloats<TV> => TF.Store(y + i, TF.Mul(TF.Load(x + i), TF.All(scale)));
    }

    // y = x / d.
    private struct DivideMap(float* x, float* y, float d) : IRowMap
    {
        public readonly void Step<TV, TF>(nint i) where TV : unmanaged where TF : struct, IFloats<TV> => TF.Store(y + i, TF.Div(TF.Load(x + i), TF.All(d)));
    }

    // y = x - shift.
    private struct ShiftMap(float* x, float* y, float shift) : IRowMap
    {
        public readonly void Step<TV, TF>(nint i) where TV : unmanaged where TF : struct, IFloats<TV> => TF.Store(y + i, TF.Sub(TF.Load(x + i), TF.All(shift)));
    }

    // y = scale · x + mask (no mask: scale · x), rounded once.
    private struct ScaleMaskMap(float* x, float* mask, float* y, float scale) : IRowMap
    {
        public readonly void Step<TV, TF>(nint i) where TV : unmanaged where TF : struct, IFloats<TV> =>
            TF.Store(y + i, mask == null ? TF.Mul(TF.All(scale), TF.Load(x + i)) : TF.Fma(TF.All(scale), TF.Load(x + i), TF.Load(mask + i)));
    }

    // The softmax of a row of n into y: its maximum, e^(x - max) and their sum, then each divided by the sum (log: x - max
    // - ln sum).
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void SoftmaxRow<TV, TF>(float* x, float* y, int n, bool log) where TV : unmanaged where TF : struct, IFloats<TV>
    {
        float max = MaxOf<TV, TF>(x, n);
        if (log)
        {
            // The sum of e^(x - max) without writing y first (y may be x itself).
            var expTerms = new ExpSumTerms(x, max);
            float total = Reduce16<TV, TF, ExpSumTerms>(ref expTerms, n);
            var shift = new ShiftMap(x, y, max + Fn<float, F1>.Log(total));
            RowMap<TV, TF, ShiftMap>(ref shift, n);
            return;
        }

        var exp = new ExpShiftMap(x, y, max);
        RowMap<TV, TF, ExpShiftMap>(ref exp, n);
        var terms = new SumTerms(y);
        float sum = Reduce16<TV, TF, SumTerms>(ref terms, n);
        var divide = new DivideMap(y, y, sum);
        RowMap<TV, TF, DivideMap>(ref divide, n);
    }

    public override void SoftmaxKernel(Storage x, Storage y, int rows, int cols, bool log)
    {
        fixed (float* px = V(x), py = V(y))
        {
            Rows(log ? "LogSoftmax" : "Softmax", new SoftmaxRows(px, py, cols, log, rows), rows, cols, 24);
        }
    }

    private struct SoftmaxRows(float* x, float* y, int cols, bool log, int rows) : IRowKernel
    {
        private float* _y = y;

        public readonly void Row<TV, TF>(int r) where TV : unmanaged where TF : struct, IFloats<TV> =>
            SoftmaxRow<TV, TF>(x + (long)r * cols, _y + (long)r * cols, cols, log);

        public void Redirect(ref Scratch scratch) => _y = scratch.Take((long)rows * cols);
    }

    public override void ScaleMaskSoftmaxKernel(Storage x, Storage? mask, Storage y, int rows, int cols, int maskRows, float scale)
    {
        fixed (float* px = V(x), pm = mask is null ? null : V(mask), py = V(y))
        {
            Rows("ScaleMaskSoftmax", new ScaleMaskSoftmaxRows(px, pm, py, cols, Math.Max(1, maskRows), scale, rows), rows, cols, 24);
        }
    }

    // y = softmax(scale · x + mask[r % maskRows]).
    private struct ScaleMaskSoftmaxRows(float* x, float* mask, float* y, int cols, int maskRows, float scale, int rows) : IRowKernel
    {
        private float* _y = y;

        public readonly void Row<TV, TF>(int r) where TV : unmanaged where TF : struct, IFloats<TV>
        {
            float* row = _y + (long)r * cols;
            var scaled = new ScaleMaskMap(x + (long)r * cols, mask == null ? null : mask + (long)(r % maskRows) * cols, row, scale);
            RowMap<TV, TF, ScaleMaskMap>(ref scaled, cols);
            SoftmaxRow<TV, TF>(row, row, cols, log: false);
        }

        public void Redirect(ref Scratch scratch) => _y = scratch.Take((long)rows * cols);
    }

    public override void SoftmaxBackwardKernel(Storage y, Storage dy, Storage dx, int rows, int cols, bool log)
    {
        fixed (float* py = V(y), pdy = V(dy), pdx = V(dx))
        {
            Rows(log ? "LogSoftmaxBackward" : "SoftmaxBackward", new SoftmaxBackRows(py, pdy, pdx, cols, log, rows), rows, cols, 24);
        }
    }

    // Softmax: dx += y (dy - Σ dy·y); log-softmax: dx += dy - e^y Σ dy.
    private struct SoftmaxBackRows(float* y, float* dy, float* dx, int cols, bool log, int rows) : IRowKernel
    {
        private float* _dx = dx;

        public readonly void Row<TV, TF>(int r) where TV : unmanaged where TF : struct, IFloats<TV>
        {
            long at = (long)r * cols;
            float s;
            if (log)
            {
                var terms = new SumTerms(dy + at);
                s = Reduce16<TV, TF, SumTerms>(ref terms, cols);
            }
            else
            {
                var terms = new DotTerms(dy + at, y + at);
                s = Reduce16<TV, TF, DotTerms>(ref terms, cols);
            }

            var map = new SoftmaxBackMap(y + at, dy + at, _dx + at, s, log);
            RowMap<TV, TF, SoftmaxBackMap>(ref map, cols);
        }

        public void Redirect(ref Scratch scratch) => _dx = scratch.Take((long)rows * cols);
    }

    private struct SoftmaxBackMap(float* y, float* dy, float* dx, float s, bool log) : IRowMap
    {
        public readonly void Step<TV, TF>(nint i) where TV : unmanaged where TF : struct, IFloats<TV>
        {
            TV v = TF.Load(y + i), g = TF.Load(dy + i), d = TF.Load(dx + i);
            TF.Store(dx + i, log ? TF.Add(d, TF.Fma(Fn<TV, TF>.Neg(Fn<TV, TF>.Exp(v)), TF.All(s), g)) : TF.Fma(v, TF.Sub(g, TF.All(s)), d));
        }
    }

    // ------------------------------------------------------------------ RMS normalization

    // 1 / √(mean(x²) + eps) of a row of n.
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static float RmsInverse<TV, TF>(float* x, int n, float eps) where TV : unmanaged where TF : struct, IFloats<TV>
    {
        var terms = new SquareTerms(x);
        float sum = Reduce16<TV, TF, SquareTerms>(ref terms, n);
        return 1f / MathF.Sqrt(sum / n + eps);
    }

    // y = x · inv · (gain + offset) (no gain: x · inv).
    private struct GainMap(float* x, float* gain, float* y, float inv, float offset) : IRowMap
    {
        public readonly void Step<TV, TF>(nint i) where TV : unmanaged where TF : struct, IFloats<TV>
        {
            var v = TF.Mul(TF.Load(x + i), TF.All(inv));
            TF.Store(y + i, gain == null ? v : TF.Mul(v, TF.Add(TF.Load(gain + i), TF.All(offset))));
        }
    }

    public override void RmsNormKernel(Storage x, Storage y, Storage inv, int rows, int cols, float eps)
    {
        fixed (float* px = V(x), py = V(y), pi = V(inv))
        {
            Rows("RmsNorm", new RmsRows(px, null, null, null, py, pi, cols, eps, 0f, rows), rows, cols);
        }
    }

    public override void RmsNormAffineKernel(Storage x, Storage gain, Storage y, int rows, int cols, float eps, float offset)
    {
        fixed (float* px = V(x), pg = V(gain), py = V(y))
        {
            Rows("RmsNormAffine", new RmsRows(px, null, null, pg, py, null, cols, eps, offset, rows), rows, cols);
        }
    }

    public override void AddRmsNormAffineKernel(Storage a, Storage b, Storage sum, Storage gain, Storage y, int rows, int cols, float eps, float offset)
    {
        fixed (float* pa = V(a), pb = V(b), ps = V(sum), pg = V(gain), py = V(y))
        {
            Rows("AddRmsNormAffine", new RmsRows(pa, pb, ps, pg, py, null, cols, eps, offset, rows), rows, cols);
        }
    }

    // A row RMS-normalized (x, or sum = a + b first), times gain + offset where given; its inverse kept where asked.
    private struct RmsRows(float* x, float* b, float* sum, float* gain, float* y, float* inv, int cols, float eps, float offset, int rows) : IRowKernel
    {
        private float* _y = y, _sum = sum, _inv = inv;

        public readonly void Row<TV, TF>(int r) where TV : unmanaged where TF : struct, IFloats<TV>
        {
            long at = (long)r * cols;
            float* source = x + at;
            float factor;
            if (b != null)
            {
                var terms = new AddSquareTerms(x + at, b + at, _sum + at);
                float squares = Reduce16<TV, TF, AddSquareTerms>(ref terms, cols);
                factor = 1f / MathF.Sqrt(squares / cols + eps);
                source = _sum + at;
            }
            else
            {
                factor = RmsInverse<TV, TF>(source, cols, eps);
            }

            if (_inv != null)
            {
                _inv[r] = factor;
            }

            var map = new GainMap(source, gain, _y + at, factor, offset);
            RowMap<TV, TF, GainMap>(ref map, cols);
        }

        public void Redirect(ref Scratch scratch)
        {
            _y = scratch.Take((long)rows * cols);
            _sum = _sum == null ? null : scratch.Take((long)rows * cols);
            _inv = _inv == null ? null : scratch.Take(rows);
        }
    }

    public override void RmsNormBackwardKernel(Storage dy, Storage y, Storage inv, Storage dx, int rows, int cols)
    {
        fixed (float* pdy = V(dy), py = V(y), pi = V(inv), pdx = V(dx))
        {
            Rows("RmsNormBackward", new RmsBackRows(pdy, py, pi, pdx, cols, rows), rows, cols);
        }
    }

    // dx += inv · (dy - y · mean(dy · y)).
    private struct RmsBackRows(float* dy, float* y, float* inv, float* dx, int cols, int rows) : IRowKernel
    {
        private float* _dx = dx;

        public readonly void Row<TV, TF>(int r) where TV : unmanaged where TF : struct, IFloats<TV>
        {
            long at = (long)r * cols;
            var terms = new DotTerms(dy + at, y + at);
            float mean = Reduce16<TV, TF, DotTerms>(ref terms, cols) / cols;
            var map = new RmsBackMap(dy + at, y + at, _dx + at, inv[r], mean);
            RowMap<TV, TF, RmsBackMap>(ref map, cols);
        }

        public void Redirect(ref Scratch scratch) => _dx = scratch.Take((long)rows * cols);
    }

    private struct RmsBackMap(float* dy, float* y, float* dx, float inv, float mean) : IRowMap
    {
        public readonly void Step<TV, TF>(nint i) where TV : unmanaged where TF : struct, IFloats<TV> =>
            TF.Store(dx + i, TF.Fma(TF.All(inv), TF.Fma(Fn<TV, TF>.Neg(TF.Load(y + i)), TF.All(mean), TF.Load(dy + i)), TF.Load(dx + i)));
    }

    // ------------------------------------------------------------------ layer normalization

    public override void LayerNormFusedKernel(Storage x, Storage gamma, Storage beta, Storage y, int rows, int cols, float eps)
    {
        fixed (float* px = V(x), pg = V(gamma), pb = V(beta), py = V(y))
        {
            Rows("LayerNorm", new LayerRows(px, pg, pb, py, null, cols, eps, rows), rows, cols);
        }
    }

    public override void LayerNormTrainKernel(Storage x, Storage gamma, Storage beta, Storage y, Storage stats, int rows, int cols, float eps)
    {
        fixed (float* px = V(x), pg = V(gamma), pb = V(beta), py = V(y), ps = V(stats))
        {
            Rows("LayerNorm", new LayerRows(px, pg, pb, py, ps, cols, eps, rows), rows, cols);
        }
    }

    // A row's mean and 1 / √(var + eps), the variance from the deviations (two passes).
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static (float Mean, float Inv) MeanInverse<TV, TF>(float* x, int n, float eps) where TV : unmanaged where TF : struct, IFloats<TV>
    {
        var sum = new SumTerms(x);
        float mean = Reduce16<TV, TF, SumTerms>(ref sum, n) / n;
        var deviations = new DeviationTerms(x, mean);
        float variance = Reduce16<TV, TF, DeviationTerms>(ref deviations, n) / n;
        return (mean, 1f / MathF.Sqrt(variance + eps));
    }

    // y = (x - mean) · inv · gamma + beta; stats[r] = mean and stats[rows + r] = inv where kept.
    private struct LayerRows(float* x, float* gamma, float* beta, float* y, float* stats, int cols, float eps, int rows) : IRowKernel
    {
        private float* _y = y, _stats = stats;

        public readonly void Row<TV, TF>(int r) where TV : unmanaged where TF : struct, IFloats<TV>
        {
            long at = (long)r * cols;
            var (mean, inv) = MeanInverse<TV, TF>(x + at, cols, eps);
            if (_stats != null)
            {
                _stats[r] = mean;
                _stats[rows + r] = inv;
            }

            var map = new LayerMap(x + at, gamma, beta, _y + at, mean, inv);
            RowMap<TV, TF, LayerMap>(ref map, cols);
        }

        public void Redirect(ref Scratch scratch)
        {
            _y = scratch.Take((long)rows * cols);
            _stats = _stats == null ? null : scratch.Take(2L * rows);
        }
    }

    private struct LayerMap(float* x, float* gamma, float* beta, float* y, float mean, float inv) : IRowMap
    {
        public readonly void Step<TV, TF>(nint i) where TV : unmanaged where TF : struct, IFloats<TV>
        {
            var normalized = TF.Mul(TF.Sub(TF.Load(x + i), TF.All(mean)), TF.All(inv));
            TF.Store(y + i, TF.Fma(normalized, TF.Load(gamma + i), TF.Load(beta + i)));
        }
    }

    public override void LayerNormBackwardKernel(Storage x, Storage gamma, Storage dy, Storage stats, Storage? dx, Storage? dgamma, Storage? dbeta, int rows, int cols)
    {
        fixed (float* px = V(x), pg = V(gamma), pdy = V(dy), ps = V(stats), pdx = dx is null ? null : V(dx), pdg = dgamma is null ? null : V(dgamma),
            pdb = dbeta is null ? null : V(dbeta))
        {
            if (pdx != null)
            {
                Rows("LayerNormBackward", new LayerBackRows(px, pg, pdy, ps, pdx, cols, rows), rows, cols, 8);
            }

            if (pdg != null || pdb != null)
            {
                Elements("LayerNormBackward columns", new LayerColumns(px, pdy, ps, pdg, pdb, rows, cols), cols, rows * 2);
            }
        }
    }

    // dx += inv / n · (n · dxhat - Σ dxhat - xhat · Σ dxhat · xhat), dxhat = dy · gamma, xhat = (x - mean) · inv.
    private struct LayerBackRows(float* x, float* gamma, float* dy, float* stats, float* dx, int cols, int rows) : IRowKernel
    {
        private float* _dx = dx;

        public readonly void Row<TV, TF>(int r) where TV : unmanaged where TF : struct, IFloats<TV>
        {
            long at = (long)r * cols;
            float mean = stats[r], inv = stats[rows + r];
            var terms = new LayerBackTerms(gamma, dy + at);
            float* lanes = stackalloc float[16];
            Lanes16<TV, TF, LayerBackTerms>(ref terms, cols, lanes);
            float sumHat = Tree16(lanes);
            var weighted = new LayerBackWeighted(x + at, gamma, dy + at, mean, inv);
            Lanes16<TV, TF, LayerBackWeighted>(ref weighted, cols, lanes);
            float sumHatX = Tree16(lanes);
            var map = new LayerBackMap(x + at, gamma, dy + at, _dx + at, mean, inv, sumHat, sumHatX, cols);
            RowMap<TV, TF, LayerBackMap>(ref map, cols);
        }

        public void Redirect(ref Scratch scratch) => _dx = scratch.Take((long)rows * cols);
    }

    // Σ dy · gamma.
    private struct LayerBackTerms(float* gamma, float* dy) : ITerms
    {
        public readonly TV Take<TV, TF>(TV sum, nint i) where TV : unmanaged where TF : struct, IFloats<TV> => TF.Fma(TF.Load(dy + i), TF.Load(gamma + i), sum);
    }

    // Σ dy · gamma · xhat.
    private struct LayerBackWeighted(float* x, float* gamma, float* dy, float mean, float inv) : ITerms
    {
        public readonly TV Take<TV, TF>(TV sum, nint i) where TV : unmanaged where TF : struct, IFloats<TV>
        {
            var hat = TF.Mul(TF.Sub(TF.Load(x + i), TF.All(mean)), TF.All(inv));
            return TF.Fma(TF.Mul(TF.Load(dy + i), TF.Load(gamma + i)), hat, sum);
        }
    }

    private struct LayerBackMap(float* x, float* gamma, float* dy, float* dx, float mean, float inv, float sumHat, float sumHatX, int n) : IRowMap
    {
        public readonly void Step<TV, TF>(nint i) where TV : unmanaged where TF : struct, IFloats<TV>
        {
            var hat = TF.Mul(TF.Sub(TF.Load(x + i), TF.All(mean)), TF.All(inv));
            var dhat = TF.Mul(TF.Load(dy + i), TF.Load(gamma + i));
            var inner = TF.Sub(TF.Sub(TF.Mul(TF.All((float)n), dhat), TF.All(sumHat)), TF.Mul(hat, TF.All(sumHatX)));
            TF.Store(dx + i, TF.Fma(TF.All(inv / n), inner, TF.Load(dx + i)));
        }
    }

    // dgamma[j] += Σ_r dy · xhat, dbeta[j] += Σ_r dy, each column's rows in order.
    private struct LayerColumns(float* x, float* dy, float* stats, float* dgamma, float* dbeta, int rows, int cols) : IMap
    {
        private float* _dg = dgamma, _db = dbeta;

        public readonly void Step<TV, TF>(nint j) where TV : unmanaged where TF : struct, IFloats<TV>
        {
            TV g = TF.All(0f), b = g;
            for (int r = 0; r < rows; r++)
            {
                long at = (long)r * cols + j;
                var d = TF.Load(dy + at);
                var hat = TF.Mul(TF.Sub(TF.Load(x + at), TF.All(stats[r])), TF.All(stats[rows + r]));
                g = TF.Fma(d, hat, g);
                b = TF.Add(b, d);
            }

            if (_dg != null)
            {
                TF.Store(_dg + j, TF.Add(TF.Load(_dg + j), g));
            }

            if (_db != null)
            {
                TF.Store(_db + j, TF.Add(TF.Load(_db + j), b));
            }
        }

        public void Redirect(ref Scratch scratch)
        {
            _dg = _dg == null ? null : scratch.Take(cols);
            _db = _db == null ? null : scratch.Take(cols);
        }
    }

    // ------------------------------------------------------------------ the rotary embedding

    public override void RopeKernel(Storage x, Storage y, Storage cos, Storage sin, Storage positions, int rows, int heads, int steps, int dim, int half,
        bool interleaved, float sign)
    {
        fixed (float* px = V(x), py = V(y), pc = V(cos), ps = V(sin), pp = V(positions))
        {
            Rows("Rope", new RopeRows(px, py, pc, ps, pp, null, heads, steps, dim, dim, half, interleaved, sign, 0f, 0f, rows), rows, 2 * half);
        }
    }

    public override void RmsNormRopeKernel(Storage x, Storage gain, Storage cos, Storage sin, Storage positions, Storage y, int rows, int cols,
        float eps, float offset, int heads, int steps, int half, bool interleaved)
    {
        fixed (float* px = V(x), pg = V(gain), pc = V(cos), ps = V(sin), pp = V(positions), py = V(y))
        {
            Rows("RmsNormRope", new RopeRows(px, py, pc, ps, pp, pg, heads, steps, cols, cols, half, interleaved, 1f, eps, offset, rows), rows, cols);
        }
    }

    // Row r (of batch · steps · heads, step (r / heads) mod steps): RMS-normalized with gain first where given (the whole
    // row), then each pair p < half rotated by cos/sin[positions[step], p], sin times sign: (a, b) → (a·c − b·s, a·s + b·c).
    private struct RopeRows(float* x, float* y, float* cos, float* sin, float* positions, float* gain, int heads, int steps, int dim, int stride,
        int half, bool interleaved, float sign, float eps, float offset, int rows) : IRowKernel
    {
        private float* _y = y;

        public readonly void Row<TV, TF>(int r) where TV : unmanaged where TF : struct, IFloats<TV>
        {
            long at = (long)r * stride;
            float* source = x + at, target = _y + at;
            if (gain != null)
            {
                float inv = RmsInverse<TV, TF>(source, dim, eps);
                var map = new GainMap(source, gain, target, inv, offset);
                RowMap<TV, TF, GainMap>(ref map, dim);
                source = target;
            }

            int step = r / heads % steps;
            long position = (long)positions[step];
            float* c = cos + position * half, s = sin + position * half;
            if (interleaved)
            {
                for (int p = 0; p < half; p++)
                {
                    float a = source[2 * p], b = source[2 * p + 1], cp = c[p], sp = sign * s[p];
                    target[2 * p] = MathF.FusedMultiplyAdd(a, cp, -(b * sp));
                    target[2 * p + 1] = MathF.FusedMultiplyAdd(a, sp, b * cp);
                }
            }
            else
            {
                var map = new RopeHalfMap(source, target, c, s, half, sign);
                RowMap<TV, TF, RopeHalfMap>(ref map, half);
            }
        }

        public void Redirect(ref Scratch scratch) => _y = scratch.Take((long)rows * stride);
    }

    // Pairs (p, p + half) of a row rotated, both read before either is written (the row may be rotated in place).
    private struct RopeHalfMap(float* x, float* y, float* cos, float* sin, int half, float sign) : IRowMap
    {
        public readonly void Step<TV, TF>(nint p) where TV : unmanaged where TF : struct, IFloats<TV>
        {
            TV a = TF.Load(x + p), b = TF.Load(x + p + half), c = TF.Load(cos + p), s = TF.Mul(TF.All(sign), TF.Load(sin + p));
            var first = TF.Fma(a, c, Fn<TV, TF>.Neg(TF.Mul(b, s)));
            var second = TF.Fma(a, s, TF.Mul(b, c));
            TF.Store(y + p, first);
            TF.Store(y + p + half, second);
        }
    }

    // ------------------------------------------------------------------ a row's vector

    public override void AddRowVectorKernel(Storage a, Storage v, Storage c, int rows, int cols)
    {
        fixed (float* pa = V(a), pv = V(v), pc = V(c))
        {
            Rows("AddRowVector", new RowVectorRows(pa, pv, pc, cols, rows, gelu: false), rows, cols, 1);
        }
    }

    public override void BiasGeluKernel(Storage x, Storage bias, Storage y, int n, int cols)
    {
        fixed (float* px = V(x), pb = V(bias), py = V(y))
        {
            Rows("BiasGelu", new RowVectorRows(px, pb, py, cols, n / Math.Max(1, cols), gelu: true), n / Math.Max(1, cols), cols, 24);
        }
    }

    // c = a + v a row (gelu: gelu(a + v)).
    private struct RowVectorRows(float* a, float* v, float* c, int cols, int rows, bool gelu) : IRowKernel
    {
        private float* _c = c;

        public readonly void Row<TV, TF>(int r) where TV : unmanaged where TF : struct, IFloats<TV>
        {
            long at = (long)r * cols;
            var map = new RowVectorMap(a + at, v, _c + at, gelu);
            RowMap<TV, TF, RowVectorMap>(ref map, cols);
        }

        public void Redirect(ref Scratch scratch) => _c = scratch.Take((long)rows * cols);
    }

    private struct RowVectorMap(float* a, float* v, float* c, bool gelu) : IRowMap
    {
        public readonly void Step<TV, TF>(nint i) where TV : unmanaged where TF : struct, IFloats<TV>
        {
            var sum = TF.Add(TF.Load(a + i), TF.Load(v + i));
            TF.Store(c + i, gelu ? Fn<TV, TF>.Gelu(sum) : sum);
        }
    }

    public override void SumRowsKernel(Storage x, Storage y, int rows, int cols)
    {
        fixed (float* px = V(x), py = V(y))
        {
            Elements("SumRows", new ColumnSums(px, 0, cols, py, rows, cols), cols, Math.Max(1, rows));
        }
    }

    public override void SumColumnsKernel(Storage x, long offset, int ld, Storage y, int rows, int cols)
    {
        fixed (float* px = V(x), py = V(y))
        {
            Elements("SumColumns", new ColumnSums(px, offset, ld, py, rows, cols), cols, Math.Max(1, rows));
        }
    }

    // y[j] += Σ_r x[offset + r · ld + j], the rows in order.
    private struct ColumnSums(float* x, long offset, int ld, float* y, int rows, int cols) : IMap
    {
        private float* _y = y;

        public readonly void Step<TV, TF>(nint j) where TV : unmanaged where TF : struct, IFloats<TV>
        {
            var s = TF.All(0f);
            for (int r = 0; r < rows; r++)
            {
                s = TF.Add(s, TF.Load(x + offset + (long)r * ld + j));
            }

            TF.Store(_y + j, TF.Add(TF.Load(_y + j), s));
        }

        public void Redirect(ref Scratch scratch) => _y = scratch.Take(cols);
    }

    public override void Copy2D(Storage src, int srcOffset, int srcStride, Storage dst, int dstOffset, int dstStride, int rows, int cols, bool accumulate)
    {
        fixed (float* ps = V(src), pd = V(dst))
        {
            Rows("Copy2D", new CopyRows(ps + srcOffset, srcStride, pd + dstOffset, dstStride, cols, accumulate, rows), rows, cols, 1);
        }
    }

    // dst row r (+)= src row r.
    private struct CopyRows(float* src, int srcStride, float* dst, int dstStride, int cols, bool accumulate, int rows) : IRowKernel
    {
        private float* _dst = dst;
        private int _dstStride = dstStride;

        public readonly void Row<TV, TF>(int r) where TV : unmanaged where TF : struct, IFloats<TV>
        {
            float* s = src + (long)r * srcStride, d = _dst + (long)r * _dstStride;
            if (!accumulate)
            {
                new ReadOnlySpan<float>(s, cols).CopyTo(new Span<float>(d, cols));
                return;
            }

            var map = new AccumulateMap(s, d);
            RowMap<TV, TF, AccumulateMap>(ref map, cols);
        }

        public void Redirect(ref Scratch scratch)
        {
            _dst = scratch.Take((long)rows * cols);
            _dstStride = cols;
        }
    }

    // y += x.
    private struct AccumulateMap(float* x, float* y) : IRowMap
    {
        public readonly void Step<TV, TF>(nint i) where TV : unmanaged where TF : struct, IFloats<TV> => TF.Store(y + i, TF.Add(TF.Load(y + i), TF.Load(x + i)));
    }

    // ------------------------------------------------------------------ indices and classes

    public override void ArgMaxKernel(Storage x, Storage y, int rows, int cols)
    {
        var px = V(x);
        fixed (float* py = D(y))
        {
            var into = IndicesOf(y, py);
            for (int r = 0; r < rows; r++)
            {
                var row = px.AsSpan(r * cols, cols);
                int best = 0;
                float value = cols > 0 ? row[0] : 0f;
                for (int j = 1; j < cols; j++)
                {
                    // The first of the largest; a NaN is the largest (as PyTorch's argmax takes it).
                    if (row[j] > value || float.IsNaN(row[j]) && !float.IsNaN(value))
                    {
                        (best, value) = (j, row[j]);
                    }
                }

                into.Set(r, best);
            }
        }
    }

    public override void ClassMatchKernel(Storage predictions, Storage targets, Storage y, int rows, int cols, float threshold)
    {
        var p = V(predictions);
        var t = V(targets);
        var py = V(y);
        for (int r = 0; r < rows; r++)
        {
            if (cols == 1)
            {
                py[r] = p[r] >= threshold == t[r] >= 0.5f ? 1f : 0f;
                continue;
            }

            int bp = 0, bt = 0;
            for (int j = 1; j < cols; j++)
            {
                if (p[r * cols + j] > p[r * cols + bp])
                {
                    bp = j;
                }

                if (t[r * cols + j] > t[r * cols + bt])
                {
                    bt = j;
                }
            }

            py[r] = bp == bt ? 1f : 0f;
        }
    }

    public override void GatherKernel(Storage table, Storage indices, Storage y, int count, int dim, int vocabulary)
    {
        fixed (float* pt = D(table), pi = D(indices), py = V(y))
        {
            var ids = IndicesOf(indices, pi);
            if (NativeFloats(table))
            {
                Rows("Gather", new GatherRows(pt, ids, py, dim, vocabulary, count), count, dim, 1);
                return;
            }

            // The described read: each row's values read through the table's type from its stored bytes (a row whole
            // groups of the type's blocks), straight into y; no row of the table is copied wider anywhere.
            var type = table.Type;
            var d = type.Description;
            if (dim % d.UnitValues != 0)
            {
                throw new NotSupportedException($"The CPU reads rows of {dim} values of {type.Name} whole groups of {d.UnitValues}: this table's rows are not.");
            }

            long rowBytes = type.ByteCount(dim);
            var bytes = new ReadOnlySpan<byte>(pt, (int)Math.Min(int.MaxValue, (long)table.Length * sizeof(float)));
            for (int i = 0; i < count; i++)
            {
                long index = ids[i];
                var target = new Span<float>(py + (long)i * dim, dim);
                if (index >= 0 && index < vocabulary)
                {
                    type.Read(bytes.Slice(checked((int)(index * rowBytes)), (int)rowBytes), target);
                }
                else
                {
                    target.Clear();
                }
            }
        }
    }

    // y[i] = table[indices[i]] (zeros for an index outside the table).
    private struct GatherRows(float* table, Indices indices, float* y, int dim, int vocabulary, int count) : IRowKernel
    {
        private float* _y = y;

        public readonly void Row<TV, TF>(int i) where TV : unmanaged where TF : struct, IFloats<TV>
        {
            long index = indices[i];
            var target = new Span<float>(_y + (long)i * dim, dim);
            if (index >= 0 && index < vocabulary)
            {
                new ReadOnlySpan<float>(table + index * dim, dim).CopyTo(target);
            }
            else
            {
                target.Clear();
            }
        }

        public void Redirect(ref Scratch scratch) => _y = scratch.Take((long)count * dim);
    }

    public override void OneHotKernel(Storage indices, Storage y, int count, int classes)
    {
        var py = V(y);
        py.AsSpan(0, count * classes).Clear();
        fixed (float* pi = D(indices))
        {
            var ids = IndicesOf(indices, pi);
            for (int i = 0; i < count; i++)
            {
                long index = ids[i];
                if (index >= 0 && index < classes)
                {
                    py[i * classes + (int)index] = 1f;
                }
            }
        }
    }

    public override void ScatterAddKernel(Storage dy, Storage indices, Storage dtable, int count, int dim, int vocabulary)
    {
        fixed (float* pdy = V(dy), pi = D(indices), pt = V(dtable))
        {
            Elements("ScatterAdd", new ScatterColumns(pdy, IndicesOf(indices, pi), pt, count, dim, vocabulary), dim, Math.Max(1, count));
        }
    }

    // dtable[indices[i], j] += dy[i, j] for each column j, the rows i in position order (each id's rows in order).
    private struct ScatterColumns(float* dy, Indices indices, float* table, int count, int dim, int vocabulary) : IMap
    {
        private float* _table = table;

        public readonly void Step<TV, TF>(nint j) where TV : unmanaged where TF : struct, IFloats<TV>
        {
            for (int i = 0; i < count; i++)
            {
                long index = indices[i];
                if (index >= 0 && index < vocabulary)
                {
                    float* row = _table + index * dim + j;
                    TF.Store(row, TF.Add(TF.Load(row), TF.Load(dy + (long)i * dim + j)));
                }
            }
        }

        public void Redirect(ref Scratch scratch) => _table = scratch.Take((long)vocabulary * dim);
    }
}
