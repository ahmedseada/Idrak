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
        var px = V(x);
        var py = V(y);
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
        var px = V(x);
        var py = V(y);
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

    public override void SumAxisKernel(Storage x, Storage y, int outer, int dim, int inner, float scale, bool accumulate)
    {
        fixed (float* px = V(x), py = V(y))
        {
            Rows("SumAxis", new AxisRows(px, py, dim, inner, scale, accumulate, outer, broadcast: false), outer, inner * Math.Max(1, dim), 1);
        }
    }

    public override void BroadcastAxisKernel(Storage dy, Storage dx, int outer, int dim, int inner, float scale)
    {
        fixed (float* pdy = V(dy), pdx = V(dx))
        {
            Rows("BroadcastAxis", new AxisRows(pdy, pdx, dim, inner, scale, true, outer, broadcast: true), outer, inner * Math.Max(1, dim), 1);
        }
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
    private static float GroupSum<TV, TF>(float* a, float* b, int outer, int groups, int inner, int g, float mean, int kind) where TV : unmanaged where TF : struct, IFloats<TV>
    {
        float* lanes = stackalloc float[16];
        for (int j = 0; j < 16; j++)
        {
            lanes[j] = 0f;
        }

        long e = 0;
        for (int o = 0; o < outer; o++)
        {
            float* ra = a + ((long)o * groups + g) * inner, rb = b == null ? null : b + ((long)o * groups + g) * inner;
            int i = 0;
            if (TF.Count > 1 && inner % 16 == 0)
            {
                for (; i < inner; i += TF.Count)
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

                e += inner;
                continue;
            }

            for (; i < inner; i++, e++)
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

        return Tree16(lanes);
    }

    public override void NormStatsKernel(Storage x, Storage mean, Storage variance, Storage invStd, int outer, int groups, int inner, float eps)
    {
        fixed (float* px = V(x), pm = V(mean), pv = V(variance), pi = V(invStd))
        {
            Rows("NormStats", new StatsRows(px, pm, pv, pi, outer, groups, inner, eps), groups, outer * inner, 2);
        }
    }

    private struct StatsRows(float* x, float* mean, float* variance, float* invStd, int outer, int groups, int inner, float eps) : IRowKernel
    {
        private float* _m = mean, _v = variance, _i = invStd;

        public readonly void Row<TV, TF>(int g) where TV : unmanaged where TF : struct, IFloats<TV>
        {
            float count = (float)outer * inner;
            float m = GroupSum<TV, TF>(x, null, outer, groups, inner, g, 0f, 0) / count;
            float v = GroupSum<TV, TF>(x, null, outer, groups, inner, g, m, 1) / count;
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
        fixed (float* pa = V(a), pb = b is null ? null : V(b), psa = V(sumA), psab = sumAB is null ? null : V(sumAB))
        {
            Rows("GroupReduce", new ReduceRows(pa, pb, psa, psab, outer, groups, inner), groups, outer * inner, 2);
        }
    }

    private struct ReduceRows(float* a, float* b, float* sumA, float* sumAB, int outer, int groups, int inner) : IRowKernel
    {
        private float* _sa = sumA, _sab = sumAB;

        public readonly void Row<TV, TF>(int g) where TV : unmanaged where TF : struct, IFloats<TV>
        {
            _sa[g] += GroupSum<TV, TF>(a, null, outer, groups, inner, g, 0f, 0);
            if (_sab != null && b != null)
            {
                _sab[g] += GroupSum<TV, TF>(a, b, outer, groups, inner, g, 0f, 2);
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
        fixed (float* px = V(x), pm = V(mean), pi = V(invStd), py = V(y))
        {
            Rows("NormApply", new GroupRows(px, null, pm, pi, null, py, groups, inner, 0, (long)outer * groups, 0f), outer * groups, inner, 1);
        }
    }

    public override void NormBackwardKernel(Storage dxhat, Storage xhat, Storage sum1, Storage sum2, Storage invStd, Storage dx, int outer, int groups, int inner)
    {
        fixed (float* pd = V(dxhat), ph = V(xhat), p1 = V(sum1), p2 = V(sum2), pi = V(invStd), px = V(dx))
        {
            Rows("NormBackward", new GroupRows(pd, ph, p1, p2, pi, px, groups, inner, 1, (long)outer * groups, (float)outer * inner), outer * groups, inner, 2);
        }
    }

    public override void GroupScaleShiftKernel(Storage x, Storage? scale, Storage? shift, Storage y, int n, int groups, int inner, bool accumulate)
    {
        fixed (float* px = V(x), ps = scale is null ? null : V(scale), pt = shift is null ? null : V(shift), py = V(y))
        {
            int runs = n / Math.Max(1, inner);
            Rows("GroupScaleShift", new GroupRows(px, null, ps, pt, null, py, groups, inner, accumulate ? 3 : 2, runs, 0f), runs, inner, 1);
        }
    }

    // A run of inner elements of group g = run mod groups. Kind 0: y = (x - mean[g]) · inv[g]; 1: dx += inv[g] / M ·
    // (M · dxhat - sum1[g] - xhat · sum2[g]); 2 and 3 (accumulate): y (+)= x · scale[g] + shift[g] (1 and 0 where absent).
    private struct GroupRows(float* x, float* x2, float* p, float* q, float* inv, float* y, int groups, int inner, int kind, long runs, float count) : IRowKernel
    {
        private float* _y = y;

        public readonly void Row<TV, TF>(int run) where TV : unmanaged where TF : struct, IFloats<TV>
        {
            int g = run % groups;
            long at = (long)run * inner;
            var map = new GroupMap(x + at, x2 == null ? null : x2 + at, _y + at, kind,
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
        fixed (float* pg = V(gain), pd = V(direction), pw = V(weight))
        {
            Rows("WeightNorm", new WeightNormRows(pg, pd, pw, null, null, columns, rows, backward: false), rows, columns, 2);
        }
    }

    public override void WeightNormBackwardKernel(Storage gain, Storage direction, Storage dweight, Storage? dgain, Storage? ddirection, int rows, int columns)
    {
        fixed (float* pg = V(gain), pd = V(direction), pw = V(dweight), pdg = dgain is null ? null : V(dgain), pdd = ddirection is null ? null : V(ddirection))
        {
            Rows("WeightNormBackward", new WeightNormRows(pg, pd, pw, pdg, pdd, columns, rows, backward: true), rows, columns, 4);
        }
    }

    // weight[r] = gain[r] · dir[r] / ‖dir[r]‖; backward: dgain[r] += dot / n, ddir[r] += gain[r] / n · (dw - dir · dot / n²).
    private struct WeightNormRows(float* gain, float* direction, float* weight, float* dgain, float* ddirection, int columns, int rows, bool backward) : IRowKernel
    {
        private float* _w = weight, _dg = dgain, _dd = ddirection;

        public readonly void Row<TV, TF>(int r) where TV : unmanaged where TF : struct, IFloats<TV>
        {
            long at = (long)r * columns;
            var squares = new SquareTerms(direction + at);
            float norm = MathF.Sqrt(Reduce16<TV, TF, SquareTerms>(ref squares, columns));
            if (!backward)
            {
                var map = new ScaleMap(direction + at, _w + at, gain[r] / norm);
                RowMap<TV, TF, ScaleMap>(ref map, columns);
                return;
            }

            var dotTerms = new DotTerms(_w + at, direction + at);
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
                    _dd[at + j] += factor * (_w[at + j] - direction[at + j] * along);
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
