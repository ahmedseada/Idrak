// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Runtime.CompilerServices;

namespace Idrak.Abstraction.Devices.Cpu;

// The layout operations (permutes, strided copies, sums and broadcasts over an axis) and the grouped normalizations
// (batch statistics over [outer, groups, inner], their gradient parts, weight normalization), written once over the float
// lanes. A group's sums take its elements in order (o, i) into the sixteen canonical lanes, so every width gives the same bits.
internal sealed unsafe partial class CpuBackend
{
    public override void PermuteKernel(Storage x, Storage y, ReadOnlySpan<int> outShape, ReadOnlySpan<int> inStrides, bool accumulate)
    {
        var py = V(y);
        if (!NativeFloats(x))
        {
            Span<long> strides = stackalloc long[Math.Max(1, inStrides.Length)];
            for (int d = 0; d < inStrides.Length; d++)
            {
                strides[d] = inStrides[d];
            }

            StridedTyped(x, 0, outShape, strides[..inStrides.Length], py, accumulate, "Permute");
            return;
        }

        var px = D(x);
        int rank = outShape.Length;
        long total = 1;
        foreach (int d in outShape)
        {
            total *= d;
        }

        if (total == 0)
        {
            return;
        }

        Span<int> index = stackalloc int[Math.Max(1, rank)];
        index.Clear();
        long at = 0;
        for (long o = 0; o < total; o++)
        {
            py[o] = accumulate ? py[o] + px[at] : px[at];
            for (int d = rank - 1; d >= 0; d--)
            {
                at += inStrides[d];
                if (++index[d] < outShape[d])
                {
                    break;
                }

                at -= (long)inStrides[d] * outShape[d];
                index[d] = 0;
            }
        }
    }

    public override void StridedCopyKernel(Storage x, long offset, ReadOnlySpan<int> shape, ReadOnlySpan<long> strides, Storage y)
    {
        var py = V(y);
        if (!NativeFloats(x))
        {
            StridedTyped(x, offset, shape, strides, py, accumulate: false, "StridedCopy");
            return;
        }

        var px = D(x);
        int rank = shape.Length;
        long total = 1;
        foreach (int d in shape)
        {
            total *= d;
        }

        if (total == 0)
        {
            return;
        }

        if (rank == 0)
        {
            py[0] = px[offset];
            return;
        }

        // The last dimension a run where its stride is 1 (a slice's rows), else element by element.
        Span<int> index = stackalloc int[rank];
        index.Clear();
        int last = shape[rank - 1];
        long runs = total / last, at = offset;
        for (long run = 0; run < runs; run++)
        {
            long into = run * last;
            if (strides[rank - 1] == 1)
            {
                px.AsSpan((int)at, last).CopyTo(py.AsSpan((int)into, last));
            }
            else
            {
                for (int j = 0; j < last; j++)
                {
                    py[into + j] = px[at + j * strides[rank - 1]];
                }
            }

            for (int d = rank - 2; d >= 0; d--)
            {
                at += strides[d];
                if (++index[d] < shape[d])
                {
                    break;
                }

                at -= strides[d] * shape[d];
                index[d] = 0;
            }
        }
    }

    // A strided view of a typed x copied (or added) into the contiguous y: the output's positions taken in the order of the
    // view's strides, the largest outermost, so its addresses run on through x and each window of x is read through its
    // type once where the view reads x in order (a permutation, a slice, a broadcast); every value as the type reads it.
    private static void StridedTyped(Storage x, long offset, ReadOnlySpan<int> shape, ReadOnlySpan<long> strides, float[] y, bool accumulate, string operation)
    {
        int rank = shape.Length;
        long total = 1;
        foreach (int d in shape)
        {
            total *= d;
        }

        if (total == 0)
        {
            return;
        }

        // The dimensions by the magnitude of their strides, the largest first (ties in their own order), and the output's
        // contiguous steps.
        Span<int> order = stackalloc int[Math.Max(1, rank)];
        Span<long> step = stackalloc long[Math.Max(1, rank)];
        Span<int> index = stackalloc int[Math.Max(1, rank)];
        index.Clear();
        long run = 1;
        for (int d = rank - 1; d >= 0; d--)
        {
            order[d] = d;
            step[d] = run;
            run *= shape[d];
        }

        for (int a = 1; a < rank; a++)
        {
            int d = order[a], b = a - 1;
            for (; b >= 0 && Math.Abs(strides[order[b]]) < Math.Abs(strides[d]); b--)
            {
                order[b + 1] = order[b];
            }

            order[b + 1] = d;
        }

        var reader = new TypedRanges(x, 0, operation);
        fixed (float* words = D(x))
        {
            float* values = null;
            long from = 0, to = 0, at = offset, o = 0;
            for (long t = 0; t < total; t++)
            {
                float v = reader.Windowed(words, at, ChunkValues, ref values, ref from, ref to);
                y[o] = accumulate ? y[o] + v : v;
                for (int q = rank - 1; q >= 0; q--)
                {
                    int d = order[q];
                    at += strides[d];
                    o += step[d];
                    if (++index[d] < shape[d])
                    {
                        break;
                    }

                    at -= strides[d] * shape[d];
                    o -= step[d] * shape[d];
                    index[d] = 0;
                }
            }
        }
    }

    public override void SumAxisKernel(Storage x, Storage y, int outer, int dim, int inner, float scale, bool accumulate)
    {
        fixed (float* px = D(x), py = V(y))
        {
            if (!NativeFloats(x))
            {
                // Each row's values read through x's type a chunk at a time, the sums kept in the thread's working row.
                var tx = new TypedRanges(x, 0, "SumAxis");
                Rows("SumAxis", new TypedAxisRows(tx, px, py, dim, inner, scale, accumulate, outer), outer, inner * Math.Max(1, dim), 1);
                return;
            }

            Rows("SumAxis", new AxisRows(px, py, dim, inner, scale, accumulate, outer, broadcast: false), outer, inner * Math.Max(1, dim), 1);
        }
    }

    public override void BroadcastAxisKernel(Storage dy, Storage dx, int outer, int dim, int inner, float scale)
    {
        fixed (float* pdy = D(dy), pdx = V(dx))
        {
            if (!NativeFloats(dy))
            {
                var tdy = new TypedRanges(dy, 0, "BroadcastAxis");
                Rows("BroadcastAxis", new TypedBroadcastRows(tdy, pdy, pdx, dim, inner, scale, outer), outer, inner * Math.Max(1, dim), 1);
                return;
            }

            Rows("BroadcastAxis", new AxisRows(pdy, pdx, dim, inner, scale, true, outer, broadcast: true), outer, inner * Math.Max(1, dim), 1);
        }
    }

    // SumAxis over a typed x: row o's [dim, inner] values read a chunk at a time (in their order, so each sum takes its terms
    // in d's order from zero, as AxisMap's), the sums in the thread's working row, then y[o, i] (+)= scale · sum: AxisMap's
    // bits.
    private struct TypedAxisRows(TypedRanges reader, float* x, float* y, int dim, int inner, float scale, bool accumulate, int outer) : IRowKernel
    {
        private float* _y = y;

        public readonly void Row<TV, TF>(int o) where TV : unmanaged where TF : struct, IFloats<TV>
        {
            float* sums = WorkRow(0, Math.Max(1, inner));
            new Span<float>(sums, inner).Clear();
            long first = (long)o * dim * inner, count = (long)dim * inner;
            for (long c = 0; c < count; c += ChunkValues)
            {
                long end = Math.Min(count, c + ChunkValues);
                float* values = reader.At(x, first + c, end - c) + first;
                for (long e = c; e < end;)
                {
                    int i = (int)(e % inner), length = (int)Math.Min(inner - i, end - e);
                    var add = new AccumulateMap(values + e, sums + i);
                    RowMap<TV, TF, AccumulateMap>(ref add, length);
                    e += length;
                }
            }

            var store = new AxisStoreMap(sums, _y + (long)o * inner, scale, accumulate);
            RowMap<TV, TF, AxisStoreMap>(ref store, inner);
        }

        public void Redirect(ref Scratch scratch) => _y = scratch.Take((long)outer * inner);
    }

    // y (+)= scale · s (AxisMap's last step).
    private struct AxisStoreMap(float* s, float* y, float scale, bool accumulate) : IRowMap
    {
        public readonly void Step<TV, TF>(nint i) where TV : unmanaged where TF : struct, IFloats<TV>
        {
            var value = TF.Mul(TF.All(scale), TF.Load(s + i));
            TF.Store(y + i, accumulate ? TF.Add(TF.Load(y + i), value) : value);
        }
    }

    // BroadcastAxis from a typed dy: row o's inner values read through dy's type a chunk at a time, each chunk added into
    // every d of the row (AxisMap's).
    private struct TypedBroadcastRows(TypedRanges reader, float* dy, float* dx, int dim, int inner, float scale, int outer) : IRowKernel
    {
        private float* _dx = dx;

        public readonly void Row<TV, TF>(int o) where TV : unmanaged where TF : struct, IFloats<TV>
        {
            long first = (long)o * inner;
            for (int i0 = 0; i0 < inner; i0 += ChunkValues)
            {
                int length = Math.Min(ChunkValues, inner - i0);
                float* values = reader.At(dy, first + i0, length);
                var map = new AxisMap(values + i0, _dx + i0, o, dim, inner, scale, true, broadcast: true);
                RowMap<TV, TF, AxisMap>(ref map, length);
            }
        }

        public void Redirect(ref Scratch scratch) => _dx = scratch.Take((long)outer * dim * inner);
    }

    // Sum: y[o, i] (+)= scale · Σ_d x[o, d, i] (d in order). Broadcast: dx[o, d, i] += scale · dy[o, i].
    private struct AxisRows(float* x, float* y, int dim, int inner, float scale, bool accumulate, int outer, bool broadcast) : IRowKernel
    {
        private float* _y = y;

        public readonly void Row<TV, TF>(int o) where TV : unmanaged where TF : struct, IFloats<TV>
        {
            var map = new AxisMap(x, _y, o, dim, inner, scale, accumulate, broadcast);
            RowMap<TV, TF, AxisMap>(ref map, inner);
        }

        public void Redirect(ref Scratch scratch) => _y = scratch.Take(broadcast ? (long)outer * dim * inner : (long)outer * inner);
    }

    private struct AxisMap(float* x, float* y, int o, int dim, int inner, float scale, bool accumulate, bool broadcast) : IRowMap
    {
        public readonly void Step<TV, TF>(nint i) where TV : unmanaged where TF : struct, IFloats<TV>
        {
            if (broadcast)
            {
                var g = TF.Load(x + (long)o * inner + i);
                for (int d = 0; d < dim; d++)
                {
                    float* at = y + ((long)o * dim + d) * inner + i;
                    TF.Store(at, TF.Fma(TF.All(scale), g, TF.Load(at)));
                }

                return;
            }

            var s = TF.All(0f);
            for (int d = 0; d < dim; d++)
            {
                s = TF.Add(s, TF.Load(x + ((long)o * dim + d) * inner + i));
            }

            float* to = y + (long)o * inner + i;
            var value = TF.Mul(TF.All(scale), s);
            TF.Store(to, accumulate ? TF.Add(TF.Load(to), value) : value);
        }
    }

    // ------------------------------------------------------------------ grouped normalization

    // The sixteen lanes of a group's Σ term(x[o, g, i]) (and of Σ a·b with a second input), elements taken in order (o, i):
    // by the width where a run of inner holds whole blocks of 16 (its lanes then i mod 16), else a float at a time.
    // A typed operand's run is read through its type in pieces of this many values (a multiple of 16: each piece's lanes
    // continue the run's).
    private const int GroupPiece = 4096;

    private static float GroupSum<TV, TF>(float* a, float* b, int outer, int groups, int inner, int g, float mean, int kind, TypedRanges ta = default,
        TypedRanges tb = default) where TV : unmanaged where TF : struct, IFloats<TV>
    {
        float* lanes = stackalloc float[16];
        for (int j = 0; j < 16; j++)
        {
            lanes[j] = 0f;
        }

        // A typed operand's run read a piece at a time (the same terms in the same lanes); float32's own read whole.
        int piece = ta.Typed || tb.Typed ? GroupPiece : Math.Max(1, inner);
        long e = 0;
        for (int o = 0; o < outer; o++)
        {
            long run = ((long)o * groups + g) * inner;
            for (int i0 = 0; i0 < inner; i0 += piece)
            {
                int i1 = Math.Min(inner, i0 + piece);
                float* ra = ta.At(a, run + i0, i1 - i0) + run, rb = b == null ? null : tb.At(b, run + i0, i1 - i0) + run;
                int i = i0;
                if (TF.Count > 1 && inner % 16 == 0)
                {
                    for (; i < i1; i += TF.Count)
                    {
                        var v = TF.Load(ra + i);
                        var term = kind switch
                        {
                            0 => v,
                            1 => TF.Mul(TF.Sub(v, TF.All(mean)), TF.Sub(v, TF.All(mean))),
                            _ => TF.Mul(v, TF.Load(rb + i)),
                        };
                        float* lane = lanes + (i & 15);
                        TF.Store(lane, TF.Add(TF.Load(lane), term));
                    }

                    e += i1 - i0;
                    continue;
                }

                for (; i < i1; i++, e++)
                {
                    float v = ra[i];
                    float term = kind switch
                    {
                        0 => v,
                        1 => (v - mean) * (v - mean),
                        _ => v * rb[i],
                    };
                    lanes[e & 15] += term;
                }
            }
        }

        return Tree16(lanes);
    }

    // The grouped normalizations read a typed x (and the gradient parts a typed dxhat or xhat) through its type where they
    // read it: a group's runs a piece at a time, a run of a row kernel when it runs; the per-group vectors once, their own
    // size.
    public override void NormStatsKernel(Storage x, Storage mean, Storage variance, Storage invStd, int outer, int groups, int inner, float eps)
    {
        fixed (float* px = D(x), pm = V(mean), pv = V(variance), pi = V(invStd))
        {
            Rows("NormStats", new StatsRows(px, pm, pv, pi, outer, groups, inner, eps, new TypedRanges(x, 0, "NormStats")), groups, outer * inner, 2);
        }
    }

    private struct StatsRows(float* x, float* mean, float* variance, float* invStd, int outer, int groups, int inner, float eps, TypedRanges tx) : IRowKernel
    {
        private float* _m = mean, _v = variance, _i = invStd;

        public readonly void Row<TV, TF>(int g) where TV : unmanaged where TF : struct, IFloats<TV>
        {
            float count = (float)outer * inner;
            float m = GroupSum<TV, TF>(x, null, outer, groups, inner, g, 0f, 0, tx) / count;
            float v = GroupSum<TV, TF>(x, null, outer, groups, inner, g, m, 1, tx) / count;
            _m[g] = m;
            _v[g] = v;
            _i[g] = 1f / MathF.Sqrt(v + eps);
        }

        public void Redirect(ref Scratch scratch)
        {
            _m = scratch.Take(groups);
            _v = scratch.Take(groups);
            _i = scratch.Take(groups);
        }
    }

    public override void GroupReduceKernel(Storage a, Storage? b, Storage sumA, Storage? sumAB, int outer, int groups, int inner)
    {
        fixed (float* pa = D(a), pb = b is null ? null : D(b), psa = V(sumA), psab = sumAB is null ? null : V(sumAB))
        {
            Rows("GroupReduce", new ReduceRows(pa, pb, psa, psab, outer, groups, inner, new TypedRanges(a, 0, "GroupReduce"),
                b is null ? default : new TypedRanges(b, 1, "GroupReduce")), groups, outer * inner, 2);
        }
    }

    private struct ReduceRows(float* a, float* b, float* sumA, float* sumAB, int outer, int groups, int inner, TypedRanges ta, TypedRanges tb) : IRowKernel
    {
        private float* _sa = sumA, _sab = sumAB;

        public readonly void Row<TV, TF>(int g) where TV : unmanaged where TF : struct, IFloats<TV>
        {
            _sa[g] += GroupSum<TV, TF>(a, null, outer, groups, inner, g, 0f, 0, ta);
            if (_sab != null && b != null)
            {
                _sab[g] += GroupSum<TV, TF>(a, b, outer, groups, inner, g, 0f, 2, ta, tb);
            }
        }

        public void Redirect(ref Scratch scratch)
        {
            _sa = scratch.Take(groups);
            _sab = _sab == null ? null : scratch.Take(groups);
        }
    }

    public override void NormApplyKernel(Storage x, Storage mean, Storage invStd, Storage y, int outer, int groups, int inner)
    {
        fixed (float* px = D(x), wm = D(mean), wi = D(invStd), py = V(y))
        {
            TypedInput tm = new(mean, wm, 0, groups, "NormApply"), ti = new(invStd, wi, 1, groups, "NormApply");
            float* pm = tm.Range(0, groups), pi = ti.Range(0, groups);
            Rows("NormApply", new GroupRows(px, null, pm, pi, null, py, groups, inner, 0, (long)outer * groups, 0f, new TypedRanges(x, 0, "NormApply")), outer * groups,
                inner, 1);
        }
    }

    public override void NormBackwardKernel(Storage dxhat, Storage xhat, Storage sum1, Storage sum2, Storage invStd, Storage dx, int outer, int groups, int inner)
    {
        fixed (float* pd = D(dxhat), ph = D(xhat), w1 = D(sum1), w2 = D(sum2), wi = D(invStd), px = V(dx))
        {
            TypedInput t1 = new(sum1, w1, 0, groups, "NormBackward"), t2 = new(sum2, w2, 1, groups, "NormBackward"), ti = new(invStd, wi, 2, groups, "NormBackward");
            float* p1 = t1.Range(0, groups), p2 = t2.Range(0, groups), pi = ti.Range(0, groups);
            Rows("NormBackward", new GroupRows(pd, ph, p1, p2, pi, px, groups, inner, 1, (long)outer * groups, (float)outer * inner, new TypedRanges(dxhat, 0, "NormBackward"),
                new TypedRanges(xhat, 1, "NormBackward")), outer * groups, inner, 2);
        }
    }

    public override void GroupScaleShiftKernel(Storage x, Storage? scale, Storage? shift, Storage y, int n, int groups, int inner, bool accumulate)
    {
        fixed (float* px = D(x), ws = scale is null ? null : D(scale), wt = shift is null ? null : D(shift), py = V(y))
        {
            TypedInput ts = scale is null ? default : new(scale, ws, 0, groups, "GroupScaleShift"), tt = shift is null ? default : new(shift, wt, 1, groups, "GroupScaleShift");
            float* ps = scale is null ? null : ts.Range(0, groups), pt = shift is null ? null : tt.Range(0, groups);
            int runs = n / Math.Max(1, inner);
            Rows("GroupScaleShift", new GroupRows(px, null, ps, pt, null, py, groups, inner, accumulate ? 3 : 2, runs, 0f, new TypedRanges(x, 0, "GroupScaleShift")), runs,
                inner, 1);
        }
    }

    // A run of inner elements of group g = run mod groups. Kind 0: y = (x - mean[g]) · inv[g]; 1: dx += inv[g] / M ·
    // (M · dxhat - sum1[g] - xhat · sum2[g]); 2 and 3 (accumulate): y (+)= x · scale[g] + shift[g] (1 and 0 where absent).
    // A typed x (or x2) is read through its type a run at a time, when the run's row runs.
    private struct GroupRows(float* x, float* x2, float* p, float* q, float* inv, float* y, int groups, int inner, int kind, long runs, float count, TypedRanges tx,
        TypedRanges tx2 = default) : IRowKernel
    {
        private float* _y = y;

        public readonly void Row<TV, TF>(int run) where TV : unmanaged where TF : struct, IFloats<TV>
        {
            int g = run % groups;
            long at = (long)run * inner;
            var map = new GroupMap(tx.At(x, at, inner) + at, x2 == null ? null : tx2.At(x2, at, inner) + at, _y + at, kind,
                kind == 0 ? p[g] : kind == 1 ? p[g] : p == null ? 1f : p[g],
                kind == 0 ? q[g] : kind == 1 ? q[g] : q == null ? 0f : q[g],
                kind == 1 ? inv[g] : 0f, count);
            RowMap<TV, TF, GroupMap>(ref map, inner);
        }

        public void Redirect(ref Scratch scratch) => _y = scratch.Take(runs * inner);
    }

    private struct GroupMap(float* x, float* x2, float* y, int kind, float a, float b, float inv, float count) : IRowMap
    {
        public readonly void Step<TV, TF>(nint i) where TV : unmanaged where TF : struct, IFloats<TV>
        {
            var v = TF.Load(x + i);
            switch (kind)
            {
                case 0:
                    TF.Store(y + i, TF.Mul(TF.Sub(v, TF.All(a)), TF.All(b)));
                    break;
                case 1:
                {
                    var inner = TF.Sub(TF.Sub(TF.Mul(TF.All(count), v), TF.All(a)), TF.Mul(TF.Load(x2 + i), TF.All(b)));
                    TF.Store(y + i, TF.Fma(TF.All(inv / count), inner, TF.Load(y + i)));
                    break;
                }

                case 2:
                    TF.Store(y + i, TF.Fma(v, TF.All(a), TF.All(b)));
                    break;
                default:
                    TF.Store(y + i, TF.Add(TF.Load(y + i), TF.Fma(v, TF.All(a), TF.All(b))));
                    break;
            }
        }
    }

    // ------------------------------------------------------------------ weight normalization

    public override void WeightNormKernel(Storage gain, Storage direction, Storage weight, int rows, int columns)
    {
        fixed (float* wg = D(gain), pd = D(direction), pw = V(weight))
        {
            var tg = new TypedInput(gain, wg, 0, rows, "WeightNorm");
            Rows("WeightNorm", new WeightNormRows(tg.Range(0, rows), pd, pw, null, null, columns, rows, backward: false, new TypedRanges(direction, 0, "WeightNorm"), default),
                rows, columns, 2);
        }
    }

    public override void WeightNormBackwardKernel(Storage gain, Storage direction, Storage dweight, Storage? dgain, Storage? ddirection, int rows, int columns)
    {
        fixed (float* wg = D(gain), pd = D(direction), pw = D(dweight), pdg = dgain is null ? null : V(dgain), pdd = ddirection is null ? null : V(ddirection))
        {
            var tg = new TypedInput(gain, wg, 0, rows, "WeightNormBackward");
            Rows("WeightNormBackward", new WeightNormRows(tg.Range(0, rows), pd, pw, pdg, pdd, columns, rows, backward: true, new TypedRanges(direction, 0, "WeightNormBackward"),
                new TypedRanges(dweight, 1, "WeightNormBackward")), rows, columns, 4);
        }
    }

    // weight[r] = gain[r] · dir[r] / ‖dir[r]‖; backward: dgain[r] += dot / n, ddir[r] += gain[r] / n · (dw - dir · dot / n²).
    // A typed direction (or dweight) is read through its type a row at a time, when the row runs.
    private struct WeightNormRows(float* gain, float* direction, float* weight, float* dgain, float* ddirection, int columns, int rows, bool backward, TypedRanges td,
        TypedRanges tw) : IRowKernel
    {
        private float* _w = weight, _dg = dgain, _dd = ddirection;

        public readonly void Row<TV, TF>(int r) where TV : unmanaged where TF : struct, IFloats<TV>
        {
            long at = (long)r * columns;
            float* dir = td.At(direction, at, columns) + at;
            var squares = new SquareTerms(dir);
            float norm = MathF.Sqrt(Reduce16<TV, TF, SquareTerms>(ref squares, columns));
            if (!backward)
            {
                var map = new ScaleMap(dir, _w + at, gain[r] / norm);
                RowMap<TV, TF, ScaleMap>(ref map, columns);
                return;
            }

            float* dw = tw.At(_w, at, columns) + at;                 // backward: _w is dweight, never redirected
            var dotTerms = new DotTerms(dw, dir);
            float dot = Reduce16<TV, TF, DotTerms>(ref dotTerms, columns);
            if (_dg != null)
            {
                _dg[r] += dot / norm;
            }

            if (_dd != null)
            {
                float factor = gain[r] / norm, along = dot / (norm * norm);
                for (int j = 0; j < columns; j++)
                {
                    _dd[r * (long)columns + j] += factor * (dw[j] - dir[j] * along);
                }
            }
        }

        public void Redirect(ref Scratch scratch)
        {
            if (!backward)
            {
                _w = scratch.Take((long)rows * columns);
            }

            _dg = _dg == null ? null : scratch.Take(rows);
            _dd = _dd == null ? null : scratch.Take((long)rows * columns);
        }
    }
}
