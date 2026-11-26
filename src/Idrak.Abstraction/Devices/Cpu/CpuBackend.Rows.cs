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

    // Each row operation takes its row inputs through TypedInput: float32's own words as they are (one run over every row),
    // another type's values read a chunk of rows at a time (the run a chunk at a time, its kernel indexing rows as in the
    // whole run); a vector every row reads (a gain, a bias) read once for the call, its own size.
    public override void SoftmaxKernel(Storage x, Storage y, int rows, int cols, bool log)
    {
        fixed (float* wx = D(x), py = V(y))
        {
            int step = RowStep(rows, cols, !NativeFloats(x));
            var tx = new TypedInput(x, wx, 0, (long)step * cols, "Softmax");
            for (int r = 0; r < rows; r += step)
            {
                int e = Math.Min(rows, r + step);
                Rows(log ? "LogSoftmax" : "Softmax", new SoftmaxRows(tx.Range((long)r * cols, (long)(e - r) * cols), py, cols, log, rows), e, cols, 24, r);
            }
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
        fixed (float* wx = D(x), wm = mask is null ? null : D(mask), py = V(y))
        {
            int masks = Math.Max(1, maskRows);
            int step = RowStep(rows, cols, !NativeFloats(x));
            TypedInput tx = new(x, wx, 0, (long)step * cols, "ScaleMaskSoftmax"), tm = mask is null ? default : new(mask, wm, 1, (long)masks * cols, "ScaleMaskSoftmax");
            float* pm = mask is null ? null : tm.Range(0, (long)masks * cols);
            for (int r = 0; r < rows; r += step)
            {
                int e = Math.Min(rows, r + step);
                Rows("ScaleMaskSoftmax", new ScaleMaskSoftmaxRows(tx.Range((long)r * cols, (long)(e - r) * cols), pm, py, cols, masks, scale, rows), e, cols, 24, r);
            }
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
        fixed (float* wy = D(y), wdy = D(dy), pdx = V(dx))
        {
            int step = RowStep(rows, cols, !NativeFloats(y) || !NativeFloats(dy));
            TypedInput ty = new(y, wy, 0, (long)step * cols, "SoftmaxBackward"), tdy = new(dy, wdy, 1, (long)step * cols, "SoftmaxBackward");
            for (int r = 0; r < rows; r += step)
            {
                int e = Math.Min(rows, r + step);
                long at = (long)r * cols, count = (long)(e - r) * cols;
                Rows(log ? "LogSoftmaxBackward" : "SoftmaxBackward", new SoftmaxBackRows(ty.Range(at, count), tdy.Range(at, count), pdx, cols, log, rows), e, cols, 24, r);
            }
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
        fixed (float* wx = D(x), py = V(y), pi = V(inv))
        {
            int step = RowStep(rows, cols, !NativeFloats(x));
            var tx = new TypedInput(x, wx, 0, (long)step * cols, "RmsNorm");
            for (int r = 0; r < rows; r += step)
            {
                int e = Math.Min(rows, r + step);
                Rows("RmsNorm", new RmsRows(tx.Range((long)r * cols, (long)(e - r) * cols), null, null, null, py, pi, cols, eps, 0f, rows), e, cols, 4, r);
            }
        }
    }

    public override void RmsNormAffineKernel(Storage x, Storage gain, Storage y, int rows, int cols, float eps, float offset)
    {
        fixed (float* wx = D(x), wg = D(gain), py = V(y))
        {
            int step = RowStep(rows, cols, !NativeFloats(x));
            TypedInput tx = new(x, wx, 0, (long)step * cols, "RmsNormAffine"), tg = new(gain, wg, 1, cols, "RmsNormAffine");
            float* pg = tg.Range(0, cols);
            for (int r = 0; r < rows; r += step)
            {
                int e = Math.Min(rows, r + step);
                Rows("RmsNormAffine", new RmsRows(tx.Range((long)r * cols, (long)(e - r) * cols), null, null, pg, py, null, cols, eps, offset, rows), e, cols, 4, r);
            }
        }
    }

    public override void AddRmsNormAffineKernel(Storage a, Storage b, Storage sum, Storage gain, Storage y, int rows, int cols, float eps, float offset)
    {
        fixed (float* wa = D(a), wb = D(b), ps = V(sum), wg = D(gain), py = V(y))
        {
            int step = RowStep(rows, cols, !NativeFloats(a) || !NativeFloats(b));
            TypedInput ta = new(a, wa, 0, (long)step * cols, "AddRmsNormAffine"), tb = new(b, wb, 1, (long)step * cols, "AddRmsNormAffine"),
                tg = new(gain, wg, 2, cols, "AddRmsNormAffine");
            float* pg = tg.Range(0, cols);
            for (int r = 0; r < rows; r += step)
            {
                int e = Math.Min(rows, r + step);
                long at = (long)r * cols, count = (long)(e - r) * cols;
                Rows("AddRmsNormAffine", new RmsRows(ta.Range(at, count), tb.Range(at, count), ps, pg, py, null, cols, eps, offset, rows), e, cols, 4, r);
            }
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
        fixed (float* wdy = D(dy), wy = D(y), pi = V(inv), pdx = V(dx))
        {
            int step = RowStep(rows, cols, !NativeFloats(dy) || !NativeFloats(y));
            TypedInput tdy = new(dy, wdy, 0, (long)step * cols, "RmsNormBackward"), ty = new(y, wy, 1, (long)step * cols, "RmsNormBackward");
            for (int r = 0; r < rows; r += step)
            {
                int e = Math.Min(rows, r + step);
                long at = (long)r * cols, count = (long)(e - r) * cols;
                Rows("RmsNormBackward", new RmsBackRows(tdy.Range(at, count), ty.Range(at, count), pi, pdx, cols, rows), e, cols, 4, r);
            }
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

    public override void LayerNormFusedKernel(Storage x, Storage gamma, Storage beta, Storage y, int rows, int cols, float eps) =>
        LayerNorm(x, gamma, beta, y, null, rows, cols, eps);

    public override void LayerNormTrainKernel(Storage x, Storage gamma, Storage beta, Storage y, Storage stats, int rows, int cols, float eps) =>
        LayerNorm(x, gamma, beta, y, stats, rows, cols, eps);

    private static void LayerNorm(Storage x, Storage gamma, Storage beta, Storage y, Storage? stats, int rows, int cols, float eps)
    {
        fixed (float* wx = D(x), wg = D(gamma), wb = D(beta), py = V(y), ps = stats is null ? null : V(stats))
        {
            int step = RowStep(rows, cols, !NativeFloats(x));
            TypedInput tx = new(x, wx, 0, (long)step * cols, "LayerNorm"), tg = new(gamma, wg, 1, cols, "LayerNorm"), tb = new(beta, wb, 2, cols, "LayerNorm");
            float* pg = tg.Range(0, cols), pb = tb.Range(0, cols);
            for (int r = 0; r < rows; r += step)
            {
                int e = Math.Min(rows, r + step);
                Rows("LayerNorm", new LayerRows(tx.Range((long)r * cols, (long)(e - r) * cols), pg, pb, py, ps, cols, eps, rows), e, cols, 4, r);
            }
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
        fixed (float* wx = D(x), wg = D(gamma), wdy = D(dy), ps = V(stats), pdx = dx is null ? null : V(dx), pdg = dgamma is null ? null : V(dgamma),
            pdb = dbeta is null ? null : V(dbeta))
        {
            bool typed = !NativeFloats(x) || !NativeFloats(dy);
            int step = RowStep(rows, cols, typed);
            TypedInput tg = new(gamma, wg, 0, cols, "LayerNormBackward"), tdy = new(dy, wdy, 1, (long)step * cols, "LayerNormBackward"),
                tx = new(x, wx, 2, (long)step * cols, "LayerNormBackward");
            float* pg = tg.Range(0, cols);
            if (pdx != null)
            {
                for (int r = 0; r < rows; r += step)
                {
                    int e = Math.Min(rows, r + step);
                    long at = (long)r * cols, count = (long)(e - r) * cols;
                    Rows("LayerNormBackward", new LayerBackRows(tx.Range(at, count), pg, tdy.Range(at, count), ps, pdx, cols, rows), e, cols, 8, r);
                }
            }

            if (pdg == null && pdb == null)
            {
                return;
            }

            if (!typed)
            {
                Elements("LayerNormBackward columns", new LayerColumns(wx, wdy, ps, pdg, pdb, rows, cols, 0, rows, null, null), cols, rows * 2);
                return;
            }

            // A typed x or dy: its rows a chunk at a time, each column's sums carried in the thread's working rows from
            // one chunk to the next (the same terms in the same order), then added to the gradients.
            float* sumG = WorkRow(0, cols), sumB = WorkRow(1, cols);
            new Span<float>(sumG, cols).Clear();
            new Span<float>(sumB, cols).Clear();
            for (int r = 0; r < rows; r += step)
            {
                int e = Math.Min(rows, r + step);
                long at = (long)r * cols, count = (long)(e - r) * cols;
                Elements("LayerNormBackward columns", new LayerColumns(tx.Range(at, count), tdy.Range(at, count), ps, pdg, pdb, rows, cols, r, e, sumG, sumB), cols,
                    (e - r) * 2);
            }

            for (int j = 0; j < cols; j++)
            {
                if (pdg != null)
                {
                    pdg[j] += sumG[j];
                }

                if (pdb != null)
                {
                    pdb[j] += sumB[j];
                }
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

    // dgamma[j] += Σ_r dy · xhat, dbeta[j] += Σ_r dy, each column's rows in order: rows [first, last), the sums from zero
    // added to the gradients, or (sumG, sumB given: a chunk of a typed run's rows) carried on in them.
    private struct LayerColumns(float* x, float* dy, float* stats, float* dgamma, float* dbeta, int rows, int cols, int first, int last, float* sumG, float* sumB) : IMap
    {
        private float* _dg = dgamma, _db = dbeta, _sg = sumG, _sb = sumB;

        public readonly void Step<TV, TF>(nint j) where TV : unmanaged where TF : struct, IFloats<TV>
        {
            TV g = _sg == null ? TF.All(0f) : TF.Load(_sg + j), b = _sb == null ? TF.All(0f) : TF.Load(_sb + j);
            for (int r = first; r < last; r++)
            {
                long at = (long)r * cols + j;
                var d = TF.Load(dy + at);
                var hat = TF.Mul(TF.Sub(TF.Load(x + at), TF.All(stats[r])), TF.All(stats[rows + r]));
                g = TF.Fma(d, hat, g);
                b = TF.Add(b, d);
            }

            if (_sg != null)
            {
                TF.Store(_sg + j, g);
                TF.Store(_sb + j, b);
                return;
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
            _sg = _sg == null ? null : scratch.Take(cols);
            _sb = _sb == null ? null : scratch.Take(cols);
        }
    }

    // ------------------------------------------------------------------ the rotary embedding

    public override void RopeKernel(Storage x, Storage y, Storage cos, Storage sin, Storage positions, int rows, int heads, int steps, int dim, int half,
        bool interleaved, float sign)
    {
        fixed (float* wx = D(x), py = V(y), pc = D(cos), ps = D(sin), pp = D(positions))
        {
            int step = RowStep(rows, dim, !NativeFloats(x));
            var tx = new TypedInput(x, wx, 0, (long)step * dim, "Rope");
            var ids = IndicesOf(positions, pp);
            TypedRanges tc = new(cos, 0, "Rope"), ts = new(sin, 1, "Rope");
            for (int r = 0; r < rows; r += step)
            {
                int e = Math.Min(rows, r + step);
                Rows("Rope", new RopeRows(tx.Range((long)r * dim, (long)(e - r) * dim), py, pc, ps, ids, null, heads, steps, dim, dim, half, interleaved, sign, 0f, 0f, rows,
                    tc, ts), e, 2 * half, 4, r);
            }
        }
    }

    public override void RmsNormRopeKernel(Storage x, Storage gain, Storage cos, Storage sin, Storage positions, Storage y, int rows, int cols,
        float eps, float offset, int heads, int steps, int half, bool interleaved)
    {
        fixed (float* wx = D(x), wg = D(gain), pc = D(cos), ps = D(sin), pp = D(positions), py = V(y))
        {
            int step = RowStep(rows, cols, !NativeFloats(x));
            TypedInput tx = new(x, wx, 0, (long)step * cols, "RmsNormRope"), tg = new(gain, wg, 1, cols, "RmsNormRope");
            float* pg = tg.Range(0, cols);
            var ids = IndicesOf(positions, pp);
            TypedRanges tc = new(cos, 0, "RmsNormRope"), ts = new(sin, 1, "RmsNormRope");
            for (int r = 0; r < rows; r += step)
            {
                int e = Math.Min(rows, r + step);
                Rows("RmsNormRope", new RopeRows(tx.Range((long)r * cols, (long)(e - r) * cols), py, pc, ps, ids, pg, heads, steps, cols, cols, half, interleaved, 1f, eps,
                    offset, rows, tc, ts), e, cols, 4, r);
            }
        }
    }

    // Row r (of batch · steps · heads, step (r / heads) mod steps): RMS-normalized with gain first where given (the whole
    // row), then each pair p < half rotated by cos/sin[positions[step], p], sin times sign: (a, b) → (a·c − b·s, a·s + b·c).
    // The positions are indices, read as their storage's type holds them; typed tables a row (a position's) at a time.
    private struct RopeRows(float* x, float* y, float* cos, float* sin, Indices positions, float* gain, int heads, int steps, int dim, int stride,
        int half, bool interleaved, float sign, float eps, float offset, int rows, TypedRanges tc, TypedRanges ts) : IRowKernel
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
            long position = positions[step];
            float* c = tc.At(cos, position * half, half) + position * half, s = ts.At(sin, position * half, half) + position * half;
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

    public override bool NormRopeHeadsKernel(Storage q, Storage k, Storage v, int batch, int steps, int heads, int kvHeads, int cols,
        Storage? gainQ, float epsQ, float offsetQ, Storage? gainK, float epsK, float offsetK, Storage? cos, Storage? sin, Storage? positions,
        int half, bool interleaved, Storage yq, Storage yk, Storage yv, Storage? position, int capacity, int stride)
    {
        if (heads <= 0 || kvHeads <= 0 || heads % kvHeads != 0 || cols <= 0 || stride < cols || (gainQ is null) != (gainK is null)
            || (half > 0 && (cos is null || sin is null || positions is null)) || 2 * half > cols)
        {
            return false;
        }

        if (batch <= 0 || steps <= 0)
        {
            return true;
        }

        fixed (float* pq = D(q), pk = D(k), pv = D(v), wgq = gainQ is null ? null : D(gainQ), wgk = gainK is null ? null : D(gainK),
            pc = half > 0 ? D(cos!) : null, ps = half > 0 ? D(sin!) : null, ppos = half > 0 ? D(positions!) : null, pyq = V(yq), pyk = V(yk), pyv = V(yv),
            pat = position is null ? null : D(position))
        {
            // The gains (a vector every head reads) read once for the call through their types: float32's own words, or a
            // gain kept as its checkpoint stores it read into the thread's buffer.
            float* pgq = gainQ is null ? null : new TypedInput(gainQ, wgq, 0, cols, "NormRopeHeads").Range(0, cols);
            float* pgk = gainK is null ? null : new TypedInput(gainK, wgk, 1, cols, "NormRopeHeads").Range(0, cols);
            long start = position is null ? 0 : IndicesOf(position, pat)[0];
            if (start < 0 || start + steps > capacity)
            {
                throw new ArgumentOutOfRangeException(nameof(position), start, $"{steps} steps from row {start} do not fit a cache of {capacity} rows.");
            }

            // Typed projections and tables read through their types a row at a time where a row reads them.
            var reads = new HeadReads(new(q, 0, "NormRopeHeads"), new(k, 1, "NormRopeHeads"), new(v, 2, "NormRopeHeads"),
                half > 0 ? new(cos!, 3, "NormRopeHeads") : default, half > 0 ? new(sin!, 4, "NormRopeHeads") : default);
            var kernel = new HeadRows(pq, pk, pv, pgq, pgk, epsQ, offsetQ, epsK, offsetK, pc, ps, half > 0 ? IndicesOf(positions!, ppos) : default, half, interleaved,
                pyq, pyk, pyv, start, batch, steps, heads, kvHeads, cols, capacity, stride, reads);
            int rows = batch * steps * (heads + 2 * kvHeads);
            Rows("NormRopeHeads", kernel, rows, cols);
        }

        return true;
    }

    // Row r: a query head (r < batch·steps·heads), then a key head, then a value head, of step s of sequence b as the
    // projections hold them ([batch, steps, heads·cols]): a query or key RMS-normalized with its gain where given, then
    // rotated by the step's position where half > 0, into yq [batch, heads, steps, cols] or the cache yk [batch, kvHeads,
    // capacity, stride] at row start + s; a value copied into yv as a key is placed.
    // NormRopeHeads' typed inputs: the projections' and the tables' rows read through their types where a row reads them.
    private readonly struct HeadReads(TypedRanges q, TypedRanges k, TypedRanges v, TypedRanges cos, TypedRanges sin)
    {
        public readonly TypedRanges Q = q, K = k, V = v, Cos = cos, Sin = sin;
    }

    private struct HeadRows(float* q, float* k, float* v, float* gainQ, float* gainK, float epsQ, float offsetQ, float epsK, float offsetK, float* cos, float* sin,
        Indices positions, int half, bool interleaved, float* yq, float* yk, float* yv, long start, int batch, int steps, int heads, int kvHeads, int cols,
        int capacity, int stride, HeadReads reads) : IRowKernel
    {
        private float* _yq = yq, _yk = yk, _yv = yv;

        public readonly void Row<TV, TF>(int r) where TV : unmanaged where TF : struct, IFloats<TV>
        {
            int queries = batch * steps * heads, keys = batch * steps * kvHeads;
            int part = r < queries ? 0 : r < queries + keys ? 1 : 2;
            int index = part == 0 ? r : part == 1 ? r - queries : r - queries - keys, width = part == 0 ? heads : kvHeads;
            int head = index % width, bs = index / width, s = bs % steps, b = bs / steps;
            long first = ((long)bs * width + head) * cols;
            var reader = part == 0 ? reads.Q : part == 1 ? reads.K : reads.V;
            float* source = reader.At(part == 0 ? q : part == 1 ? k : v, first, cols) + first;
            float* target = part == 0 ? _yq + (((long)b * heads + head) * steps + s) * cols
                : (part == 1 ? _yk : _yv) + (((long)b * kvHeads + head) * capacity + start + s) * stride;
            float* gain = part == 0 ? gainQ : part == 1 ? gainK : null;
            if (gain != null)
            {
                float eps = part == 0 ? epsQ : epsK;
                float inv = RmsInverse<TV, TF>(source, cols, eps);
                var map = new GainMap(source, gain, target, inv, part == 0 ? offsetQ : offsetK);
                RowMap<TV, TF, GainMap>(ref map, cols);
            }
            else
            {
                new ReadOnlySpan<float>(source, cols).CopyTo(new Span<float>(target, cols));
            }

            if (part == 2 || half <= 0)
            {
                return;
            }

            long at = positions[s];
            float* c = reads.Cos.At(cos, at * half, half) + at * half, sn = reads.Sin.At(sin, at * half, half) + at * half;
            if (interleaved)
            {
                for (int p = 0; p < half; p++)
                {
                    float a = target[2 * p], o = target[2 * p + 1], cp = c[p], sp = sn[p];
                    target[2 * p] = MathF.FusedMultiplyAdd(a, cp, -(o * sp));
                    target[2 * p + 1] = MathF.FusedMultiplyAdd(a, sp, o * cp);
                }
            }
            else
            {
                var rope = new RopeHalfMap(target, target, c, sn, half, 1f);
                RowMap<TV, TF, RopeHalfMap>(ref rope, half);
            }
        }

        public void Redirect(ref Scratch scratch)
        {
            _yq = scratch.Take((long)batch * heads * steps * cols);
            _yk = scratch.Take((long)batch * kvHeads * capacity * stride);
            _yv = scratch.Take((long)batch * kvHeads * capacity * stride);
        }
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
        RowVector(a, v, c, rows, cols, gelu: false);
    }

    public override void BiasGeluKernel(Storage x, Storage bias, Storage y, int n, int cols) => RowVector(x, bias, y, n / Math.Max(1, cols), cols, gelu: true);

    private static void RowVector(Storage a, Storage v, Storage c, int rows, int cols, bool gelu)
    {
        if (Wide64(a, v, c))
        {
            RowVector64(a, v, c, rows, cols, gelu);
            return;
        }

        fixed (float* wa = D(a), wv = D(v), pc = V(c))
        {
            string name = gelu ? "BiasGelu" : "AddRowVector";
            int step = RowStep(rows, cols, !NativeFloats(a));
            TypedInput ta = new(a, wa, 0, (long)step * cols, name), tv = new(v, wv, 1, cols, name);
            float* pv = tv.Range(0, cols);
            for (int r = 0; r < rows; r += step)
            {
                int e = Math.Min(rows, r + step);
                Rows(name, new RowVectorRows(ta.Range((long)r * cols, (long)(e - r) * cols), pv, pc, cols, rows, gelu), e, cols, gelu ? 24 : 1, r);
            }
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

    public override void SumRowsKernel(Storage x, Storage y, int rows, int cols) => ColumnSumsOf(x, 0, cols, y, rows, cols, "SumRows");

    public override void SumColumnsKernel(Storage x, long offset, int ld, Storage y, int rows, int cols) => ColumnSumsOf(x, offset, ld, y, rows, cols, "SumColumns");

    // y[j] += Σ_r x[offset + r · ld + j]: float32's own x in one run; a typed x a chunk of rows at a time, each column's
    // sum carried in the thread's working row from one chunk to the next (the same terms in the same order), then added.
    private static void ColumnSumsOf(Storage x, long offset, int ld, Storage y, int rows, int cols, string name)
    {
        if (Wide(x, y))
        {
            ColumnSums64(x, offset, ld, y, rows, cols);
            return;
        }

        fixed (float* wx = D(x), py = V(y))
        {
            if (NativeFloats(x))
            {
                Elements(name, new ColumnSums(wx, offset, ld, py, 0, rows, cols, null), cols, Math.Max(1, rows));
                return;
            }

            if (rows <= 0 || cols <= 0)
            {
                return;
            }

            int step = Math.Max(1, ChunkValues / Math.Max(1, ld));
            var tx = new TypedInput(x, wx, 0, (long)(Math.Min(step, rows) - 1) * ld + cols, name);
            float* sums = WorkRow(0, cols);
            new Span<float>(sums, cols).Clear();
            for (int r = 0; r < rows; r += step)
            {
                int e = Math.Min(rows, r + step);
                long at = offset + (long)r * ld;
                Elements(name, new ColumnSums(tx.Range(at, (long)(e - r - 1) * ld + cols), offset, ld, py, r, e, cols, sums), cols, e - r);
            }

            for (int j = 0; j < cols; j++)
            {
                py[j] += sums[j];
            }
        }
    }

    // y[j] += Σ_r x[offset + r · ld + j], rows [first, last) in order; with `sums` (a chunk of a typed run's rows) the sums
    // carried on in it instead.
    private struct ColumnSums(float* x, long offset, int ld, float* y, int first, int last, int cols, float* sums) : IMap
    {
        private float* _y = y, _s = sums;

        public readonly void Step<TV, TF>(nint j) where TV : unmanaged where TF : struct, IFloats<TV>
        {
            var s = _s == null ? TF.All(0f) : TF.Load(_s + j);
            for (int r = first; r < last; r++)
            {
                s = TF.Add(s, TF.Load(x + offset + (long)r * ld + j));
            }

            if (_s != null)
            {
                TF.Store(_s + j, s);
                return;
            }

            TF.Store(_y + j, TF.Add(TF.Load(_y + j), s));
        }

        public void Redirect(ref Scratch scratch)
        {
            _y = scratch.Take(cols);
            _s = _s == null ? null : scratch.Take(cols);
        }
    }

    public override void Copy2D(Storage src, int srcOffset, int srcStride, Storage dst, int dstOffset, int dstStride, int rows, int cols, bool accumulate)
    {
        if (Wide(src, dst))
        {
            // Values that need float64 (or a float64 destination): each row read in float64 and written (added) in dst's type.
            fixed (float* ws = D(src), wd = D(dst))
            {
                double* row = DoubleBuffer(0, Math.Max(1, cols));
                for (int r = 0; r < rows; r++)
                {
                    ReadDoubles(src, ws, srcOffset + (long)r * srcStride, cols, row);
                    WriteDoubles(dst, wd, dstOffset + (long)r * dstStride, cols, row, accumulate);
                }
            }

            return;
        }

        fixed (float* ws = D(src), pd = V(dst))
        {
            if (NativeFloats(src))
            {
                Rows("Copy2D", new CopyRows(ws + srcOffset, srcStride, pd + dstOffset, dstStride, cols, accumulate, rows), rows, cols, 1);
                return;
            }

            // A typed source: each row's values read through its type, then copied (or added) as float32.
            var ts = new TypedInput(src, ws, 0, cols, "Copy2D");
            for (int r = 0; r < rows; r++)
            {
                float* row = ts.Range(srcOffset + (long)r * srcStride, cols) + srcOffset + (long)r * srcStride;
                Rows("Copy2D", new CopyRows(row, 0, pd + dstOffset + (long)r * dstStride, 0, cols, accumulate, 1), 1, cols, 1);
            }
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
        fixed (float* wx = D(x), py = D(y))
        {
            var into = IndicesOf(y, py);
            int step = RowStep(rows, cols, !NativeFloats(x));
            var tx = new TypedInput(x, wx, 0, (long)step * cols, "ArgMax");
            float* px = wx;
            for (int r = 0; r < rows; r++)
            {
                if (r % step == 0)
                {
                    px = tx.Range((long)r * cols, (long)(Math.Min(rows, r + step) - r) * cols);
                }

                var row = new ReadOnlySpan<float>(px + (long)r * cols, cols);
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
        var py = V(y);
        fixed (float* wp = D(predictions), wt = D(targets))
        {
            // Typed predictions or targets read through their types a chunk of rows at a time.
            int step = RowStep(rows, cols, !NativeFloats(predictions) || !NativeFloats(targets));
            TypedInput tp = new(predictions, wp, 0, (long)step * cols, "ClassMatch"), tt = new(targets, wt, 1, (long)step * cols, "ClassMatch");
            float* p = wp, t = wt;
            for (int r = 0; r < rows; r++)
            {
                if (r % step == 0)
                {
                    long count = (long)(Math.Min(rows, r + step) - r) * cols;
                    p = tp.Range((long)r * cols, count);
                    t = tt.Range((long)r * cols, count);
                }

                if (cols == 1)
                {
                    py[r] = p[r] >= threshold == t[r] >= 0.5f ? 1f : 0f;
                    continue;
                }

                int bp = 0, bt = 0;
                long at = (long)r * cols;
                for (int j = 1; j < cols; j++)
                {
                    if (p[at + j] > p[at + bp])
                    {
                        bp = j;
                    }

                    if (t[at + j] > t[at + bt])
                    {
                        bt = j;
                    }
                }

                py[r] = bp == bt ? 1f : 0f;
            }
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
                    ReadValues(type, bytes.Slice(checked((int)(index * rowBytes)), (int)rowBytes), target);
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
        fixed (float* wdy = D(dy), pi = D(indices), pt = V(dtable))
        {
            // A typed dy read through its type a chunk of rows at a time: each table element still takes its rows in order.
            int step = RowStep(count, dim, !NativeFloats(dy));
            var tdy = new TypedInput(dy, wdy, 0, (long)step * dim, "ScatterAdd");
            var ids = IndicesOf(indices, pi);
            for (int i = 0; i < count; i += step)
            {
                int e = Math.Min(count, i + step);
                Elements("ScatterAdd", new ScatterColumns(tdy.Range((long)i * dim, (long)(e - i) * dim), ids, pt, i, e, dim, vocabulary), dim, Math.Max(1, e - i));
            }
        }
    }

    // dtable[indices[i], j] += dy[i, j] for each column j, the rows i of [first, last) in position order (each id's rows in
    // order).
    private struct ScatterColumns(float* dy, Indices indices, float* table, int first, int last, int dim, int vocabulary) : IMap
    {
        private float* _table = table;

        public readonly void Step<TV, TF>(nint j) where TV : unmanaged where TF : struct, IFloats<TV>
        {
            for (int i = first; i < last; i++)
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
