// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Buffers;
using System.Runtime.CompilerServices;

namespace Idrak.Abstraction.Devices.Cpu;

// Attention a query row at a time, written once over the float lanes: each score the dot product of the row and a key in
// the sixteen canonical lanes, times the scale (soft-capped where the variant caps), the scores' softmax as a row's
// (their maximum, e^(s - max), the sum), and the output Σ_c e_c · v_c in key order divided by the sum; the log-sum-exp
// max + ln sum where asked. The cached decode, the tiled prompt and the key ranges share it, each giving its rows' key
// ranges; the gradients recompute the weights, never store them, each key/value head's gradients added by one thread in
// row order.
internal sealed unsafe partial class CpuBackend
{
    // y = Σ_c e_c · v_c over keys [start, end) of a head (scores into `scores`), divided by Σ e_c; its log-sum-exp returned
    // (-∞ for an empty range, whose row gets zeros).
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static float AttendRow<TV, TF>(float* q, float* keys, float* values, int start, int end, int dim, float scale, float cap, float* y, float* scores)
        where TV : unmanaged where TF : struct, IFloats<TV>
    {
        var output = new Span<float>(y, dim);
        output.Clear();
        if (end <= start)
        {
            return float.NegativeInfinity;
        }

        int n = end - start;
        float max = float.NegativeInfinity;
        for (int c = 0; c < n; c++)
        {
            var terms = new DotTerms(q, keys + (long)(start + c) * dim);
            float s = Reduce16<TV, TF, DotTerms>(ref terms, dim) * scale;
            if (cap > 0f)
            {
                s = Fn<float, F1>.Tanh(s / cap) * cap;
            }

            scores[c] = s;
            max = Fn<float, F1>.Max(max, s);
        }

        var exp = new ExpShiftMap(scores, scores, max);
        RowMap<TV, TF, ExpShiftMap>(ref exp, n);
        var sumTerms = new SumTerms(scores);
        float sum = Reduce16<TV, TF, SumTerms>(ref sumTerms, n);
        for (int c = 0; c < n; c++)
        {
            var axpy = new WeightedMap(values + (long)(start + c) * dim, y, scores[c]);
            RowMap<TV, TF, WeightedMap>(ref axpy, dim);
        }

        var divide = new DivideMap(y, y, sum);
        RowMap<TV, TF, DivideMap>(ref divide, dim);
        return max + Fn<float, F1>.Log(sum);
    }

    // y += w · x, rounded once.
    private struct WeightedMap(float* x, float* y, float w) : IRowMap
    {
        public readonly void Step<TV, TF>(nint i) where TV : unmanaged where TF : struct, IFloats<TV> => TF.Store(y + i, TF.Fma(TF.All(w), TF.Load(x + i), TF.Load(y + i)));
    }

    // A thread's scores buffer (at least `count` floats), kept for the thread.
    [ThreadStatic]
    private static float[]? t_scores;

    private static float[] ScoresBuffer(int count)
    {
        var buffer = t_scores;
        if (buffer is null || buffer.Length < count)
        {
            t_scores = buffer = GC.AllocateUninitializedArray<float>(Math.Max(count, 256), pinned: true);
        }

        return buffer;
    }

    // ------------------------------------------------------------------ causal attention over a cache (decode and prompt)

    public override void AttentionDecodeKernel(Storage q, Storage keys, Storage values, Storage position, Storage y, int heads, int rowsPerHead,
        int steps, int capacity, int dim, float scale, AttentionVariant variant = default) =>
        Causal(q, keys, values, position, y, null, heads, rowsPerHead, steps, capacity, dim, scale, variant, "AttentionDecode");

    public override void AttentionTiledKernel(Storage q, Storage keys, Storage values, Storage position, Storage y, Storage? logSumExp, int heads,
        int rowsPerHead, int steps, int capacity, int dim, float scale, AttentionVariant variant = default) =>
        Causal(q, keys, values, position, y, logSumExp, heads, rowsPerHead, steps, capacity, dim, scale, variant, "AttentionTiled");

    private static void Causal(Storage q, Storage keys, Storage values, Storage position, Storage y, Storage? logSumExp, int heads, int rowsPerHead,
        int steps, int capacity, int dim, float scale, AttentionVariant variant, string operation)
    {
        int rows = heads * rowsPerHead;
        if (rows <= 0 || dim <= 0)
        {
            return;
        }

        fixed (float* pq = V(q), pk = V(keys), pv = V(values), pp = D(position), py = V(y), pl = logSumExp is null ? null : V(logSumExp))
        {
            long start = IndicesOf(position, pp)[0];
            var kernel = new CausalRows(pq, pk, pv, py, pl, (int)start, rowsPerHead, Math.Max(1, steps), capacity, dim, scale, variant, rows);
            Rows(operation, kernel, rows, dim * Math.Min(capacity, (int)start + Math.Max(1, steps)), 2);
        }
    }

    // Row h · rowsPerHead + i sees cached positions up to position + i mod steps (from its window's first).
    private struct CausalRows(float* q, float* keys, float* values, float* y, float* lse, int position, int rowsPerHead, int steps, int capacity, int dim,
        float scale, AttentionVariant variant, int rows) : IRowKernel
    {
        private float* _y = y, _lse = lse;

        public readonly void Row<TV, TF>(int row) where TV : unmanaged where TF : struct, IFloats<TV>
        {
            int h = row / rowsPerHead, i = row % rowsPerHead;
            int end = Math.Min(capacity, position + i % steps + 1), start = variant.Start(end);
            var scores = ScoresBuffer(Math.Max(1, end - start));
            fixed (float* ps = scores)
            {
                long head = (long)h * capacity * dim;
                long outRow = variant.OutputRow(row, rowsPerHead, steps);
                float l = AttendRow<TV, TF>(q + (long)row * dim, keys + head, values + head, start, end, dim, scale, variant.Softcap, _y + outRow * dim, ps);
                if (_lse != null)
                {
                    _lse[row] = l;
                }
            }
        }

        public void Redirect(ref Scratch scratch)
        {
            _y = scratch.Take((long)rows * dim);
            _lse = _lse == null ? null : scratch.Take(rows);
        }
    }

    // ------------------------------------------------------------------ attention over key ranges

    public override void AttentionSpansKernel(Storage q, Storage keys, Storage values, Storage starts, Storage ends, Storage y, Storage? logSumExp, int heads,
        int kvHeads, int headsPerTable, int rows, int keyRows, int dim, float scale, AttentionVariant variant = default)
    {
        if (heads <= 0 || rows <= 0 || dim <= 0)
        {
            return;
        }

        fixed (float* pq = V(q), pk = V(keys), pv = V(values), ps = D(starts), pe = D(ends), py = V(y), pl = logSumExp is null ? null : V(logSumExp))
        {
            var kernel = new SpanRows(pq, pk, pv, IndicesOf(starts, ps), IndicesOf(ends, pe), py, pl, heads, Math.Max(1, kvHeads), Math.Max(1, headsPerTable), rows,
                keyRows, dim, scale, variant.Softcap);
            Rows("AttentionSpans", kernel, heads * rows, dim * Math.Max(1, keyRows), 2);
        }
    }

    // Query head h's row i over keys [starts, ends) of its table's row, its key/value head h / (heads / kvHeads).
    private struct SpanRows(float* q, float* keys, float* values, Indices starts, Indices ends, float* y, float* lse, int heads, int kvHeads, int headsPerTable,
        int rows, int keyRows, int dim, float scale, float cap) : IRowKernel
    {
        private float* _y = y, _lse = lse;

        public readonly void Row<TV, TF>(int row) where TV : unmanaged where TF : struct, IFloats<TV>
        {
            int h = row / rows, i = row % rows, g = h / Math.Max(1, heads / kvHeads), b = h / headsPerTable;
            var (start, end) = SpanOf(starts, ends, (long)b * rows + i, keyRows);
            var scores = ScoresBuffer(Math.Max(1, end - start));
            fixed (float* ps = scores)
            {
                long head = (long)g * keyRows * dim;
                float l = AttendRow<TV, TF>(q + (long)row * dim, keys + head, values + head, start, end, dim, scale, cap, _y + (long)row * dim, ps);
                if (_lse != null)
                {
                    _lse[row] = l;
                }
            }
        }

        public void Redirect(ref Scratch scratch)
        {
            _y = scratch.Take((long)heads * rows * dim);
            _lse = _lse == null ? null : scratch.Take((long)heads * rows);
        }
    }

    // A row's key range, clamped to [0, keyRows].
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static (int Start, int End) SpanOf(Indices starts, Indices ends, long at, int keyRows) =>
        ((int)Math.Clamp(starts[at], 0, keyRows), (int)Math.Clamp(ends[at], 0, keyRows));

    // ------------------------------------------------------------------ the gradients

    public override void AttentionSpansBackwardKernel(Storage q, Storage keys, Storage values, Storage starts, Storage ends, Storage output, Storage logSumExp,
        Storage dOutput, Storage dq, Storage dkeys, Storage dvalues, int heads, int kvHeads, int headsPerTable, int rows, int keyRows, int dim, float scale,
        AttentionVariant variant = default)
    {
        if (heads <= 0 || rows <= 0 || dim <= 0)
        {
            return;
        }

        fixed (float* pq = V(q), pk = V(keys), pv = V(values), ps = D(starts), pe = D(ends), po = V(output), pl = V(logSumExp), pdo = V(dOutput), pdq = V(dq),
            pdk = V(dkeys), pdv = V(dvalues))
        {
            var kernel = new BackwardHeads(pq, pk, pv, po, pl, pdo, pdq, pdk, pdv, IndicesOf(starts, ps), IndicesOf(ends, pe), heads, Math.Max(1, kvHeads),
                Math.Max(1, headsPerTable), rows, keyRows, 0, dim, scale, variant, spans: true);
            Rows("AttentionSpansBackward", kernel, Math.Max(1, kvHeads), heads / Math.Max(1, kvHeads) * rows * dim * Math.Max(1, keyRows), 4);
        }
    }

    public override void AttentionTiledBackwardKernel(Storage q, Storage keys, Storage values, Storage output, Storage logSumExp, Storage dOutput,
        Storage dq, Storage dkeys, Storage dvalues, int heads, int rowsPerHead, int steps, int capacity, int dim, float scale, AttentionVariant variant = default)
    {
        if (heads <= 0 || rowsPerHead <= 0 || dim <= 0)
        {
            return;
        }

        fixed (float* pq = V(q), pk = V(keys), pv = V(values), po = V(output), pl = V(logSumExp), pdo = V(dOutput), pdq = V(dq), pdk = V(dkeys), pdv = V(dvalues))
        {
            var kernel = new BackwardHeads(pq, pk, pv, po, pl, pdo, pdq, pdk, pdv, default, default, heads, heads, 1, rowsPerHead, capacity, Math.Max(1, steps),
                dim, scale, variant, spans: false);
            Rows("AttentionTiledBackward", kernel, heads, rowsPerHead * dim * Math.Max(1, steps), 4);
        }
    }

    // A key/value head's gradients: its query heads' rows in order (spans: ranges from the tables; else causal from 0 with
    // the variant's window), each adding to dq its own row and to dkeys and dvalues the head's.
    private struct BackwardHeads(float* q, float* keys, float* values, float* output, float* lse, float* dOutput, float* dq, float* dkeys, float* dvalues,
        Indices starts, Indices ends, int heads, int kvHeads, int headsPerTable, int rows, int keyRows, int steps, int dim, float scale,
        AttentionVariant variant, bool spans) : IRowKernel
    {
        private float* _dq = dq, _dk = dkeys, _dv = dvalues;

        public readonly void Row<TV, TF>(int g) where TV : unmanaged where TF : struct, IFloats<TV>
        {
            int group = Math.Max(1, heads / kvHeads);
            long head = (long)g * keyRows * dim;
            for (int h = g * group; h < (g + 1) * group; h++)
            {
                for (int i = 0; i < rows; i++)
                {
                    int start, end;
                    if (spans)
                    {
                        (start, end) = SpanOf(starts, ends, (long)(h / headsPerTable) * rows + i, keyRows);
                    }
                    else
                    {
                        end = Math.Min(keyRows, i % steps + 1);
                        start = variant.Start(end);
                    }

                    if (end <= start)
                    {
                        continue;
                    }

                    long row = ((long)h * rows + i) * dim;
                    RowGradient<TV, TF>(q + row, keys + head, values + head, output + row, lse[(long)h * rows + i], dOutput + row, _dq + row, _dk + head, _dv + head,
                        start, end, dim, scale, variant.Softcap);
                }
            }
        }

        public void Redirect(ref Scratch scratch)
        {
            _dq = scratch.Take((long)heads * rows * dim);
            _dk = scratch.Take((long)kvHeads * keyRows * dim);
            _dv = scratch.Take((long)kvHeads * keyRows * dim);
        }
    }

    // One row's gradients: P_c = e^(s_c - lse), dP_c = dO · v_c, dS_c = P_c (dP_c - Δ) (times the cap's slope), Δ = dO · O;
    // dv_c += P_c dO, dq += scale Σ dS_c k_c, dk_c += scale dS_c q, the keys in order.
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void RowGradient<TV, TF>(float* q, float* keys, float* values, float* o, float lse, float* dO, float* dq, float* dk, float* dv, int start, int end,
        int dim, float scale, float cap) where TV : unmanaged where TF : struct, IFloats<TV>
    {
        var delta = new DotTerms(dO, o);
        float d = Reduce16<TV, TF, DotTerms>(ref delta, dim);
        for (int c = start; c < end; c++)
        {
            float* k = keys + (long)c * dim, v = values + (long)c * dim;
            var scoreTerms = new DotTerms(q, k);
            float s = Reduce16<TV, TF, DotTerms>(ref scoreTerms, dim) * scale, slope = 1f;
            if (cap > 0f)
            {
                s = Fn<float, F1>.Tanh(s / cap) * cap;
                slope = 1f - s / cap * (s / cap);
            }

            float p = Fn<float, F1>.Exp(s - lse);
            var dpTerms = new DotTerms(dO, v);
            float dp = Reduce16<TV, TF, DotTerms>(ref dpTerms, dim);
            float ds = p * (dp - d) * slope * scale;
            var toV = new WeightedMap(dO, dv + (long)c * dim, p);
            RowMap<TV, TF, WeightedMap>(ref toV, dim);
            var toQ = new WeightedMap(k, dq, ds);
            RowMap<TV, TF, WeightedMap>(ref toQ, dim);
            var toK = new WeightedMap(q, dk + (long)c * dim, ds);
            RowMap<TV, TF, WeightedMap>(ref toK, dim);
        }
    }

    // ------------------------------------------------------------------ the cache and its mask

    public override void KeyValueWriteKernel(Storage source, Storage cache, Storage position, int heads, int steps, int capacity, int dim)
    {
        var s = V(source);
        var c = V(cache);
        long start;
        fixed (float* pp = D(position))
        {
            start = IndicesOf(position, pp)[0];
        }

        int count = (int)Math.Clamp(capacity - start, 0, steps);
        for (int h = 0; h < heads; h++)
        {
            s.AsSpan(h * steps * dim, count * dim).CopyTo(c.AsSpan((int)((h * (long)capacity + start) * dim), count * dim));
        }
    }

    public override void DecoderMaskKernel(Storage position, Storage mask, int rows, int capacity)
    {
        var m = V(mask);
        long start;
        fixed (float* pp = D(position))
        {
            start = IndicesOf(position, pp)[0];
        }

        for (int i = 0; i < rows; i++)
        {
            var row = m.AsSpan(i * capacity, capacity);
            int visible = (int)Math.Clamp(start + i + 1, 0, capacity);
            row[..visible].Clear();
            row[visible..].Fill(-1e9f);
        }
    }
}
