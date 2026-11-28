// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Buffers;
using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;

namespace Idrak.Abstraction.Devices.Cpu;

// Attention a query row at a time, written once over the float lanes: each score the dot product of the row and a key in
// the sixteen canonical lanes, times the scale (soft-capped where the variant caps), the scores' softmax as a row's
// (their maximum, e^(s - max), the sum), and the output Σ_c e_c · v_c in key order divided by the sum; the log-sum-exp
// max + ln sum where asked. The cached decode, the tiled prompt and the key ranges share it, each giving its rows' key
// ranges; the gradients recompute the weights, never store them, each key/value head's gradients added by one thread in
// row order.
internal sealed unsafe partial class CpuBackend : ISegmentedAttentionBackend, IRowAttentionBackend
{
    // y = Σ_c e_c · v_c over keys [start, end) of a head (scores into `scores`), divided by Σ e_c; its log-sum-exp returned
    // (-∞ for an empty range, whose row gets zeros).
    // Key c at element c · keyStep of `keys`, its value at c · valueStep of `values` (dim for packed rows; a view's row steps
    // otherwise). Typed keys and values (a typed cache) are read through their types a block of rows at a time where the
    // pass reads them (a chunk's worth of rows, one at least), never the whole cache.
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static float AttendRow<TV, TF>(float* q, RowsIn keys, RowsIn values, int start, int end, int dim, float scale, float cap, float* y, float* scores,
        long keyStep, long valueStep)
        where TV : unmanaged where TF : struct, IFloats<TV> =>
        AttendRows<TV, TF, No>(q, null, keys, values, start, end, dim, scale, cap, y, null, scores, keyStep, valueStep, out _);

    // AttendRow for one query row (TTwo No) or two (Yes: q1's row into y1, its scores after q0's n, its log-sum-exp into
    // `second`) over the same keys [start, end): each the same numbers by the same operations as alone. The keys' dot products
    // are made several at once in registers, each its sixteen lanes in order and Tree16's tree (DotsAtOnce), the rows'
    // weighted values summed in registers a run of the row at a time, each output in key order.
    // Its parts are methods of their own, each compiled optimized from its first call (a method the JIT inlines all of
    // stays small: in one method this size the JIT stopped inlining, and the vectors' operations were calls, compiled
    // unoptimized for their first calls).
    [MethodImpl(MethodImplOptions.NoInlining | MethodImplOptions.AggressiveOptimization)]
    private static float AttendRows<TV, TF, TTwo>(float* q0, float* q1, RowsIn keys, RowsIn values, int start, int end, int dim, float scale, float cap,
        float* y0, float* y1, float* scores, long keyStep, long valueStep, out float second)
        where TV : unmanaged where TF : struct, IFloats<TV> where TTwo : struct, IFlag
    {
        bool two = typeof(TTwo) == typeof(Yes);
        new Span<float>(y0, dim).Clear();
        if (two)
        {
            new Span<float>(y1, dim).Clear();
        }

        if (end <= start)
        {
            second = float.NegativeInfinity;
            return float.NegativeInfinity;
        }

        int n = end - start;
        float* s0 = scores, s1 = scores + n;
        float max0 = float.NegativeInfinity, max1 = float.NegativeInfinity;
        int block = keys.Typed ? BlockRows(n, keyStep, dim) : n;
        for (int first = 0; first < n; first += block)
        {
            int last = Math.Min(n, first + block);
            float* k0 = keys.Row((start + first) * keyStep, (last - first - 1) * keyStep + dim) - first * keyStep;
            ScoreKeys<TV, TF, TTwo>(q0, q1, k0, keyStep, first, last, dim, scale, cap, s0, s1, ref max0, ref max1);
        }

        float sum0 = ExpSum<TV, TF>(s0, n, max0), sum1 = two ? ExpSum<TV, TF>(s1, n, max1) : 0f;
        block = values.Typed ? BlockRows(n, valueStep, dim) : n;
        for (int first = 0; first < n; first += block)
        {
            int last = Math.Min(n, first + block);
            float* v0 = values.Row((start + first) * valueStep, (last - first - 1) * valueStep + dim) - first * valueStep;
            WeightedRuns<TV, TF, TTwo>(v0, valueStep, s0, s1, first, last, y0, y1, dim);
        }

        var divide = new DivideMap(y0, y0, sum0);
        RowMap<TV, TF, DivideMap>(ref divide, dim);
        if (two)
        {
            var divide1 = new DivideMap(y1, y1, sum1);
            RowMap<TV, TF, DivideMap>(ref divide1, dim);
            second = max1 + LogOf(sum1);
        }
        else
        {
            second = float.NegativeInfinity;
        }

        return max0 + LogOf(sum0);
    }

    // The scores of keys [first, last) (key c at k + c · keyStep) for q0 (into s0 at c) and q1 (Yes: into s1), each row's
    // largest so far kept: four dot products at once where the width holds a sum's sixteen lanes in one or two vectors,
    // then each times the scale (and capped) in key order.
    [MethodImpl(MethodImplOptions.NoInlining | MethodImplOptions.AggressiveOptimization)]
    private static void ScoreKeys<TV, TF, TTwo>(float* q0, float* q1, float* k, long keyStep, int first, int last, int dim, float scale, float cap,
        float* s0, float* s1, ref float max0, ref float max1)
        where TV : unmanaged where TF : struct, IFloats<TV> where TTwo : struct, IFlag
    {
        bool two = typeof(TTwo) == typeof(Yes);
        int c = first;
        if ((TF.Count == 8 || TF.Count == 16) && (dim & 15) == 0)
        {
            for (; c + 4 <= last && !two; c += 4)
            {
                DotsAtOnce<TV, TF>(q0, q0, k + c * keyStep, k + (c + 1) * keyStep, k + (c + 2) * keyStep, k + (c + 3) * keyStep, dim).Store(s0 + c);
            }

            for (; c + 2 <= last && two; c += 2)
            {
                float* ka = k + c * keyStep, kb = ka + keyStep;
                var d = DotsAtOnce<TV, TF>(q0, q1, ka, kb, ka, kb, dim);
                s0[c] = d.GetElement(0);
                s0[c + 1] = d.GetElement(1);
                s1[c] = d.GetElement(2);
                s1[c + 1] = d.GetElement(3);
            }
        }

        for (; c < last; c++)
        {
            s0[c] = DotOf<TV, TF>(q0, k + c * keyStep, dim);
            if (two)
            {
                s1[c] = DotOf<TV, TF>(q1, k + c * keyStep, dim);
            }
        }

        ScaleKeys(s0, first, last, scale, cap, ref max0);
        if (two)
        {
            ScaleKeys(s1, first, last, scale, cap, ref max1);
        }
    }

    // q · k over dim values: Reduce16's sum.
    [MethodImpl(MethodImplOptions.NoInlining | MethodImplOptions.AggressiveOptimization)]
    private static float DotOf<TV, TF>(float* q, float* k, int dim) where TV : unmanaged where TF : struct, IFloats<TV>
    {
        var terms = new DotTerms(q, k);
        return Reduce16<TV, TF, DotTerms>(ref terms, dim);
    }

    // Each key's dot product in s [first, last) made its score: times the scale, soft-capped where the variant caps (the
    // CPU's own tanh), in key order, the row's largest kept.
    [MethodImpl(MethodImplOptions.NoInlining | MethodImplOptions.AggressiveOptimization)]
    private static void ScaleKeys(float* s, int first, int last, float scale, float cap, ref float max)
    {
        float m = max;
        for (int c = first; c < last; c++)
        {
            float v = s[c] * scale;
            if (cap > 0f)
            {
                v = Fn<float, F1>.Tanh(v / cap) * cap;
            }

            s[c] = v;
            m = Fn<float, F1>.Max(m, v);
        }

        max = m;
    }

    // A row's scores made e^(s - max) in place, and their sum (Reduce16's).
    [MethodImpl(MethodImplOptions.NoInlining | MethodImplOptions.AggressiveOptimization)]
    private static float ExpSum<TV, TF>(float* s, int n, float max) where TV : unmanaged where TF : struct, IFloats<TV>
    {
        var exp = new ExpShiftMap(s, s, max);
        RowMap<TV, TF, ExpShiftMap>(ref exp, n);
        var terms = new SumTerms(s);
        return Reduce16<TV, TF, SumTerms>(ref terms, n);
    }

    // ln x (the CPU's own), optimized from its first call.
    [MethodImpl(MethodImplOptions.NoInlining | MethodImplOptions.AggressiveOptimization)]
    private static float LogOf(float x) => Fn<float, F1>.Log(x);

    // Four dot products of dim (a multiple of 16) values at once: qa · ka, qa · kb, qb · kc, qb · kd, each Reduce16's sum
    // (element i into lane i mod 16, each lane in order, then Tree16's tree), its lanes held in registers: the 256-bit
    // lanes two vectors a sum (lanes 0-7, 8-15), the 512-bit one.
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector128<float> DotsAtOnce<TV, TF>(float* qa, float* qb, float* ka, float* kb, float* kc, float* kd, int dim)
        where TV : unmanaged where TF : struct, IFloats<TV>
    {
        if (TF.Count == 16)
        {
            Vector512<float> a = default, b = default, c = default, d = default;
            for (nint i = 0; i < dim; i += 16)
            {
                var x = Vector512.Load(qa + i);
                var z = Vector512.Load(qb + i);
                a = F512.Fma(x, Vector512.Load(ka + i), a);
                b = F512.Fma(x, Vector512.Load(kb + i), b);
                c = F512.Fma(z, Vector512.Load(kc + i), c);
                d = F512.Fma(z, Vector512.Load(kd + i), d);
            }

            return Vector128.Create(Tree8(a.GetLower() + a.GetUpper()), Tree8(b.GetLower() + b.GetUpper()), Tree8(c.GetLower() + c.GetUpper()),
                Tree8(d.GetLower() + d.GetUpper()));
        }

        Vector256<float> a0 = default, a1 = default, b0 = default, b1 = default, c0 = default, c1 = default, d0 = default, d1 = default;
        for (nint i = 0; i < dim; i += 16)
        {
            var x0 = Vector256.Load(qa + i);
            var x1 = Vector256.Load(qa + i + 8);
            a0 = F256.Fma(x0, Vector256.Load(ka + i), a0);
            a1 = F256.Fma(x1, Vector256.Load(ka + i + 8), a1);
            b0 = F256.Fma(x0, Vector256.Load(kb + i), b0);
            b1 = F256.Fma(x1, Vector256.Load(kb + i + 8), b1);
            var z0 = Vector256.Load(qb + i);
            var z1 = Vector256.Load(qb + i + 8);
            c0 = F256.Fma(z0, Vector256.Load(kc + i), c0);
            c1 = F256.Fma(z1, Vector256.Load(kc + i + 8), c1);
            d0 = F256.Fma(z0, Vector256.Load(kd + i), d0);
            d1 = F256.Fma(z1, Vector256.Load(kd + i + 8), d1);
        }

        return Vector128.Create(Tree8(a0 + a1), Tree8(b0 + b1), Tree8(c0 + c1), Tree8(d0 + d1));
    }

    // Tree16's tree past its first step (lanes j + 8 added into j): lanes j + 4, then + 2, then + 1, the same additions.
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static float Tree8(Vector256<float> v)
    {
        var w = v.GetLower() + v.GetUpper();
        var x = w + Vector128.Shuffle(w, Vector128.Create(2, 3, 2, 3));
        return (x + Vector128.Shuffle(x, Vector128.Create(1, 1, 1, 1))).ToScalar();
    }

    // y0 (and y1) += Σ_c w_c · v_c over keys [first, last) of the values at v (key c at c · step), w the rows' weights (s0,
    // s1 at c): a run of each row a few vectors long summed in registers over the keys, each output its terms fused in key
    // order onto what y held, as WeightedMap a key at a time; the run's tail a float at a time.
    [MethodImpl(MethodImplOptions.NoInlining | MethodImplOptions.AggressiveOptimization)]
    private static void WeightedRuns<TV, TF, TTwo>(float* v, long step, float* s0, float* s1, int first, int last, float* y0, float* y1, int dim)
        where TV : unmanaged where TF : struct, IFloats<TV> where TTwo : struct, IFlag
    {
        bool two = typeof(TTwo) == typeof(Yes);
        int width = 4 * TF.Count, d = 0;
        if (TF.Count > 1)
        {
            for (; d + width <= dim; d += width)
            {
                TV a0 = TF.Load(y0 + d), a1 = TF.Load(y0 + d + TF.Count), a2 = TF.Load(y0 + d + 2 * TF.Count), a3 = TF.Load(y0 + d + 3 * TF.Count);
                TV b0 = two ? TF.Load(y1 + d) : a0, b1 = two ? TF.Load(y1 + d + TF.Count) : a1, b2 = two ? TF.Load(y1 + d + 2 * TF.Count) : a2,
                    b3 = two ? TF.Load(y1 + d + 3 * TF.Count) : a3;
                for (int c = first; c < last; c++)
                {
                    float* at = v + c * step + d;
                    TV x0 = TF.Load(at), x1 = TF.Load(at + TF.Count), x2 = TF.Load(at + 2 * TF.Count), x3 = TF.Load(at + 3 * TF.Count);
                    var w = TF.All(s0[c]);
                    a0 = TF.Fma(w, x0, a0);
                    a1 = TF.Fma(w, x1, a1);
                    a2 = TF.Fma(w, x2, a2);
                    a3 = TF.Fma(w, x3, a3);
                    if (two)
                    {
                        var u = TF.All(s1[c]);
                        b0 = TF.Fma(u, x0, b0);
                        b1 = TF.Fma(u, x1, b1);
                        b2 = TF.Fma(u, x2, b2);
                        b3 = TF.Fma(u, x3, b3);
                    }
                }

                TF.Store(y0 + d, a0);
                TF.Store(y0 + d + TF.Count, a1);
                TF.Store(y0 + d + 2 * TF.Count, a2);
                TF.Store(y0 + d + 3 * TF.Count, a3);
                if (two)
                {
                    TF.Store(y1 + d, b0);
                    TF.Store(y1 + d + TF.Count, b1);
                    TF.Store(y1 + d + 2 * TF.Count, b2);
                    TF.Store(y1 + d + 3 * TF.Count, b3);
                }
            }

            for (; d + TF.Count <= dim; d += TF.Count)
            {
                TV a = TF.Load(y0 + d), b = two ? TF.Load(y1 + d) : a;
                for (int c = first; c < last; c++)
                {
                    var x = TF.Load(v + c * step + d);
                    a = TF.Fma(TF.All(s0[c]), x, a);
                    if (two)
                    {
                        b = TF.Fma(TF.All(s1[c]), x, b);
                    }
                }

                TF.Store(y0 + d, a);
                if (two)
                {
                    TF.Store(y1 + d, b);
                }
            }
        }

        for (; d < dim; d++)
        {
            float a = y0[d], b = two ? y1[d] : 0f;
            for (int c = first; c < last; c++)
            {
                float x = v[c * step + d];
                a = F1.Fma(s0[c], x, a);
                if (two)
                {
                    b = F1.Fma(s1[c], x, b);
                }
            }

            y0[d] = a;
            if (two)
            {
                y1[d] = b;
            }
        }
    }

    // The rows of `step` elements a typed operand's block takes: a chunk's worth of values, one row at least.
    private static int BlockRows(int n, long step, int dim) => (int)Math.Clamp(ChunkValues / Math.Max(1, Math.Max(step, dim)), 1, Math.Max(1, n));

    // y += w · x, rounded once.
    private struct WeightedMap(float* x, float* y, float w) : IRowMap
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public readonly void Step<TV, TF>(nint i) where TV : unmanaged where TF : struct, IFloats<TV> => TF.Store(y + i, TF.Fma(TF.All(w), TF.Load(x + i), TF.Load(y + i)));
    }

    // A thread's scores buffer (at least `count` floats), kept for the thread.
    [ThreadStatic]
    private static float[]? t_scores;

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
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
        Causal(q, keys, values, position, y, logSumExp, heads, rowsPerHead, steps, capacity, dim, scale, variant, "AttentionTiled", pairs: true);

    // `at`: views' planes read where they lie (q, keys and values each from its element offset, steps by HeadSteps), or
    // null for packed operands.
    // `views`: operands of any strides, each row gathered from where its values lie (RowsIn's view mode), the steps packed.
    // `pairs`: a prompt's (many rows): query rows that share their keys taken two at a time (CausalRows' paired units).
    private static void Causal(Storage q, Storage keys, Storage values, Storage position, Storage y, Storage? logSumExp, int heads, int rowsPerHead,
        int steps, int capacity, int dim, float scale, AttentionVariant variant, string operation, (long Q, long K, long V, HeadSteps Steps)? at = null,
        (TensorLayout? Q, TensorLayout? K, TensorLayout? V)? views = null, bool pairs = false)
    {
        int rows = heads * rowsPerHead;
        if (rows <= 0 || dim <= 0)
        {
            return;
        }

        fixed (float* pq = D(q), pk = D(keys), pv = D(values), pp = D(position), py = V(y), pl = logSumExp is null ? null : V(logSumExp))
        {
            long start = IndicesOf(position, pp)[0];
            var (oq, ok, ov, layout) = at ?? (0, 0, 0, default);
            var (vq, vk, vv) = views ?? (null, null, null);
            var rowsOfQ = Read(q, pq, oq, 0, operation, vq);
            steps = Math.Max(1, steps);

            // A prompt's query heads that share a key/value head (rowsPerHead a multiple of steps) see the same keys at each
            // position: two of them a unit, the keys read once for both (q's rows where they lie: a typed or gathered q's
            // row is the thread's buffer).
            int paired = pairs && !rowsOfQ.Typed && rowsPerHead % steps == 0 && rowsPerHead / steps >= 2 ? steps * ((rowsPerHead / steps + 1) / 2) : 0;
            var kernel = new CausalRows(rowsOfQ, Read(keys, pk, ok, 1, operation, vk), Read(values, pv, ov, 2, operation, vv), py, pl,
                (int)start, rowsPerHead, steps, capacity, dim, scale, variant, rows, default, false, 1, layout, paired);
            Rows(operation, kernel, paired > 0 ? heads * paired : rows, dim * Math.Min(capacity, (int)start + steps), paired > 0 ? 4 : 2);
        }
    }

    // An attention input's rows from element `offset` (a view's plane), read where they lie or through the input's type
    // into the thread's buffer of `slot` (a typed q, a typed cache).
    private static RowsIn Read(Storage s, float* words, long offset, int slot, string operation, TensorLayout? view = null) =>
        new(new TypedRanges(s, slot, operation), words, offset, view is { IsContiguousFromStart: false } ? view : null);

    // Where a head's rows lie: q's row i of head h at h · QHead + i · QRow, its keys' at h · KHead + c · KRow, its values'
    // at h · VHead + c · VRow (a view's planes read where they are); all 0: packed ([heads, rows, dim] each).
    private readonly record struct HeadSteps(long QHead, long QRow, long KHead, long KRow, long VHead, long VRow);

    // Row h · rowsPerHead + i sees cached positions up to position + i mod steps (from its window's first). `paired` (units
    // a head; 0: a row a unit): unit u of head h takes rows i + 2j · steps and i + (2j + 1) · steps (the second where the
    // head has it), u = j · steps + i, which see the same keys.
    private struct CausalRows(RowsIn q, RowsIn keys, RowsIn values, float* y, float* lse, int position, int rowsPerHead, int steps, int capacity, int dim,
        float scale, AttentionVariant variant, int rows, Indices starts, bool started, int headsPerRow, HeadSteps layout = default, int paired = 0) : IRowKernel
    {
        private float* _y = y, _lse = lse;

        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        public readonly void Row<TV, TF>(int unit) where TV : unmanaged where TF : struct, IFloats<TV>
        {
            int row = unit, other = -1;
            if (paired > 0)
            {
                int w = unit % paired, j = 2 * (w / steps);
                row = unit / paired * rowsPerHead + j * steps + w % steps;
                other = j + 1 < rowsPerHead / steps ? row + steps : -1;
            }

            int h = row / rowsPerHead, i = row % rowsPerHead;
            int end = Math.Min(capacity, position + i % steps + 1), start = variant.Start(end);
            if (started)
            {
                start = (int)Math.Clamp(Math.Max(start, starts[(long)(h / headsPerRow) * steps + i % steps]), 0, Math.Max(0, end));
            }
            var scores = ScoresBuffer(Math.Max(1, end - start) * (other >= 0 ? 2 : 1));
            fixed (float* ps = scores)
            {
                long outRow = variant.OutputRow(row, rowsPerHead, steps);
                var at = layout.QRow == 0 && layout.KRow == 0 && layout.VRow == 0
                    ? new HeadSteps((long)rowsPerHead * dim, dim, (long)capacity * dim, dim, (long)capacity * dim, dim)
                    : layout;
                float* q0 = q.Row(h * at.QHead + i * at.QRow, dim);
                if (other >= 0)
                {
                    float* q1 = q.Row(h * at.QHead + (i + steps) * at.QRow, dim);
                    float l = AttendRows<TV, TF, Yes>(q0, q1, keys.Plus(h * at.KHead), values.Plus(h * at.VHead), start, end, dim, scale, variant.Softcap,
                        _y + outRow * dim, _y + variant.OutputRow(other, rowsPerHead, steps) * dim, ps, at.KRow, at.VRow, out float l1);
                    if (_lse != null)
                    {
                        _lse[row] = l;
                        _lse[other] = l1;
                    }
                }
                else
                {
                    float l = AttendRow<TV, TF>(q0, keys.Plus(h * at.KHead), values.Plus(h * at.VHead), start, end, dim, scale, variant.Softcap,
                        _y + outRow * dim, ps, at.KRow, at.VRow);
                    if (_lse != null)
                    {
                        _lse[row] = l;
                    }
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

        fixed (float* pq = D(q), pk = D(keys), pv = D(values), ps = D(starts), pe = D(ends), py = V(y), pl = logSumExp is null ? null : V(logSumExp))
        {
            var kernel = new SpanRows(Read(q, pq, 0, 0, "AttentionSpans"), Read(keys, pk, 0, 1, "AttentionSpans"), Read(values, pv, 0, 2, "AttentionSpans"),
                IndicesOf(starts, ps), IndicesOf(ends, pe), py, pl, heads, Math.Max(1, kvHeads), Math.Max(1, headsPerTable), rows, keyRows, dim, scale, variant.Softcap);
            Rows("AttentionSpans", kernel, heads * rows, dim * Math.Max(1, keyRows), 2);
        }
    }

    // Query head h's row i over keys [starts, ends) of its table's row, its key/value head h / (heads / kvHeads).
    private struct SpanRows(RowsIn q, RowsIn keys, RowsIn values, Indices starts, Indices ends, float* y, float* lse, int heads, int kvHeads, int headsPerTable,
        int rows, int keyRows, int dim, float scale, float cap) : IRowKernel
    {
        private float* _y = y, _lse = lse;

        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        public readonly void Row<TV, TF>(int row) where TV : unmanaged where TF : struct, IFloats<TV>
        {
            int h = row / rows, i = row % rows, g = h / Math.Max(1, heads / kvHeads), b = h / headsPerTable;
            var (start, end) = SpanOf(starts, ends, (long)b * rows + i, keyRows);
            var scores = ScoresBuffer(Math.Max(1, end - start));
            fixed (float* ps = scores)
            {
                long head = (long)g * keyRows * dim;
                float l = AttendRow<TV, TF>(q.Row((long)row * dim, dim), keys.Plus(head), values.Plus(head), start, end, dim, scale, cap, _y + (long)row * dim, ps, dim, dim);
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

        fixed (float* pq = D(q), pk = D(keys), pv = D(values), ps = D(starts), pe = D(ends), po = D(output), pl = V(logSumExp), pdo = D(dOutput), pdq = V(dq),
            pdk = V(dkeys), pdv = V(dvalues))
        {
            var kernel = new BackwardHeads(BackwardReads(q, pq, keys, pk, values, pv, output, po, dOutput, pdo, "AttentionSpansBackward"), pl, pdq, pdk, pdv,
                IndicesOf(starts, ps), IndicesOf(ends, pe), heads, Math.Max(1, kvHeads), Math.Max(1, headsPerTable), rows, keyRows, 0, dim, scale, variant, mode: 1);
            Rows("AttentionSpansBackward", kernel, Math.Max(1, kvHeads), heads / Math.Max(1, kvHeads) * rows * dim * Math.Max(1, keyRows), 4);
        }
    }

    public override void AttentionTiledBackwardKernel(Storage q, Storage keys, Storage values, Storage output, Storage logSumExp, Storage dOutput,
        Storage dq, Storage dkeys, Storage dvalues, int heads, int rowsPerHead, int steps, int capacity, int dim, float scale, AttentionVariant variant = default) =>
        TiledBackward(q, keys, values, output, logSumExp, dOutput, dq, dkeys, dvalues, heads, rowsPerHead, steps, capacity, dim, scale, variant, null, null, null);

    // The tiled gradients; q, keys and values views of any strides read in place (RowsIn's view mode) where given.
    private static void TiledBackward(Storage q, Storage keys, Storage values, Storage output, Storage logSumExp, Storage dOutput, Storage dq, Storage dkeys,
        Storage dvalues, int heads, int rowsPerHead, int steps, int capacity, int dim, float scale, AttentionVariant variant, TensorLayout? qView, TensorLayout? keysView,
        TensorLayout? valuesView)
    {
        if (heads <= 0 || rowsPerHead <= 0 || dim <= 0)
        {
            return;
        }

        fixed (float* pq = D(q), pk = D(keys), pv = D(values), po = D(output), pl = V(logSumExp), pdo = D(dOutput), pdq = V(dq), pdk = V(dkeys), pdv = V(dvalues))
        {
            var kernel = new BackwardHeads(BackwardReads(q, pq, keys, pk, values, pv, output, po, dOutput, pdo, "AttentionTiledBackward", qView, keysView, valuesView), pl,
                pdq, pdk, pdv, default, default, heads, heads, 1, rowsPerHead, capacity, Math.Max(1, steps), dim, scale, variant, mode: 0);
            Rows("AttentionTiledBackward", kernel, heads, rowsPerHead * dim * Math.Max(1, steps), 4);
        }
    }

    // The gradients' inputs, each read where it lies or through its type a row (a block of keys) at a time.
    private readonly struct AttentionReads(RowsIn q, RowsIn keys, RowsIn values, RowsIn output, RowsIn dOutput)
    {
        public readonly RowsIn Q = q, Keys = keys, Values = values, Output = output, DOutput = dOutput;
    }

    private static AttentionReads BackwardReads(Storage q, float* pq, Storage keys, float* pk, Storage values, float* pv, Storage output, float* po, Storage dOutput,
        float* pdo, string operation, TensorLayout? qView = null, TensorLayout? keysView = null, TensorLayout? valuesView = null) =>
        new(Read(q, pq, 0, 0, operation, qView), Read(keys, pk, 0, 1, operation, keysView), Read(values, pv, 0, 2, operation, valuesView), Read(output, po, 0, 3, operation),
            Read(dOutput, pdo, 0, 4, operation));

    // A key/value head's gradients: its query heads' rows in order (spans: ranges from the tables; else causal from 0 with
    // the variant's window), each adding to dq its own row and to dkeys and dvalues the head's.
    private struct BackwardHeads(AttentionReads reads, float* lse, float* dq, float* dkeys, float* dvalues, Indices starts, Indices ends, int heads, int kvHeads,
        int headsPerTable, int rows, int keyRows, int steps, int dim, float scale, AttentionVariant variant, int mode) : IRowKernel
    {
        private float* _dq = dq, _dk = dkeys, _dv = dvalues;

        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        public readonly void Row<TV, TF>(int g) where TV : unmanaged where TF : struct, IFloats<TV>
        {
            int group = Math.Max(1, heads / kvHeads);
            long head = (long)g * keyRows * dim;
            for (int h = g * group; h < (g + 1) * group; h++)
            {
                for (int i = 0; i < rows; i++)
                {
                    int start, end;
                    if (mode == 1)
                    {
                        (start, end) = SpanOf(starts, ends, (long)(h / headsPerTable) * rows + i, keyRows);
                    }
                    else
                    {
                        end = Math.Min(keyRows, i % steps + 1);
                        start = variant.Start(end);
                        if (mode == 2)
                        {
                            start = (int)Math.Clamp(Math.Max(start, starts[(long)(h / headsPerTable) * steps + i % steps]), 0, end);
                        }
                    }

                    if (end <= start)
                    {
                        continue;
                    }

                    long row = ((long)h * rows + i) * dim;
                    RowGradient<TV, TF>(reads.Q.Row(row, dim), reads.Keys.Plus(head), reads.Values.Plus(head), reads.Output.Row(row, dim), lse[(long)h * rows + i],
                        reads.DOutput.Row(row, dim), _dq + row, _dk + head, _dv + head, start, end, dim, scale, variant.Softcap);
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
    private static void RowGradient<TV, TF>(float* q, RowsIn keys, RowsIn values, float* o, float lse, float* dO, float* dq, float* dk, float* dv, int start, int end,
        int dim, float scale, float cap) where TV : unmanaged where TF : struct, IFloats<TV>
    {
        var delta = new DotTerms(dO, o);
        float d = Reduce16<TV, TF, DotTerms>(ref delta, dim);
        for (int c = start; c < end; c++)
        {
            float* k = keys.Row((long)c * dim, dim), v = values.Row((long)c * dim, dim);
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

    // ------------------------------------------------------------------ packed sequences and rows of different lengths

    public bool SupportsSegmented(int dim, AttentionVariant variant) => true;

    public bool SupportsRows(int dim, AttentionVariant variant) => true;

    bool ISegmentedAttentionBackend.AttentionSegmented(Storage q, Storage keys, Storage values, Storage y, Storage? logSumExp, Storage starts, Storage ends, int heads,
        int headsPerRow, int rowsPerHead, int steps, int dim, float scale, AttentionVariant variant)
    {
        if (heads <= 0 || rowsPerHead <= 0 || dim <= 0)
        {
            return true;
        }

        fixed (float* pq = D(q), pk = D(keys), pv = D(values), ps = D(starts), py = V(y), pl = logSumExp is null ? null : V(logSumExp))
        {
            var plain = variant with { Merged = 0 };
            var kernel = new CausalRows(Read(q, pq, 0, 0, "AttentionSegmented"), Read(keys, pk, 0, 1, "AttentionSegmented"), Read(values, pv, 0, 2, "AttentionSegmented"),
                py, pl, 0, rowsPerHead, Math.Max(1, steps), steps, dim, scale, plain, heads * rowsPerHead, IndicesOf(starts, ps), true, Math.Max(1, headsPerRow));
            Rows("AttentionSegmented", kernel, heads * rowsPerHead, dim * Math.Max(1, steps), 2);
        }

        return true;
    }

    bool ISegmentedAttentionBackend.AttentionSegmentedBackward(Storage q, Storage keys, Storage values, Storage output, Storage logSumExp, Storage dOutput, Storage dq, Storage dkeys,
        Storage dvalues, Storage starts, Storage ends, int heads, int headsPerRow, int rowsPerHead, int steps, int dim, float scale, AttentionVariant variant)
    {
        if (heads <= 0 || rowsPerHead <= 0 || dim <= 0)
        {
            return true;
        }

        fixed (float* pq = D(q), pk = D(keys), pv = D(values), po = D(output), pl = V(logSumExp), pdo = D(dOutput), pdq = V(dq), pdk = V(dkeys), pdv = V(dvalues),
            ps = D(starts))
        {
            var kernel = new BackwardHeads(BackwardReads(q, pq, keys, pk, values, pv, output, po, dOutput, pdo, "AttentionSegmentedBackward"), pl, pdq, pdk, pdv,
                IndicesOf(starts, ps), default, heads, heads, Math.Max(1, headsPerRow), rowsPerHead, steps, Math.Max(1, steps), dim, scale, variant, mode: 2);
            Rows("AttentionSegmentedBackward", kernel, heads, rowsPerHead * dim * Math.Max(1, steps), 4);
        }

        return true;
    }

    bool IRowAttentionBackend.AttentionRows(Storage q, Storage keys, Storage values, Storage position, Storage y, Storage starts, int heads, int headsPerRow, int rowsPerHead,
        int steps, int capacity, int dim, float scale, AttentionVariant variant)
    {
        int rows = heads * rowsPerHead;
        if (rows <= 0 || dim <= 0)
        {
            return true;
        }

        fixed (float* pq = D(q), pk = D(keys), pv = D(values), pp = D(position), py = V(y), ps = D(starts))
        {
            long start = IndicesOf(position, pp)[0];
            var kernel = new CausalRows(Read(q, pq, 0, 0, "AttentionRows"), Read(keys, pk, 0, 1, "AttentionRows"), Read(values, pv, 0, 2, "AttentionRows"), py, null,
                (int)start, rowsPerHead, Math.Max(1, steps), capacity, dim, scale, variant, rows, IndicesOf(starts, ps), true, Math.Max(1, headsPerRow));
            Rows("AttentionRows", kernel, rows, dim * Math.Min(capacity, (int)start + Math.Max(1, steps)), 2);
        }

        return true;
    }

    // ------------------------------------------------------------------ the cache and its mask

    // Each head's new rows into the cache: the source's values (read through its type a chunk at a time where typed) as the
    // cache holds them: float32's own words, or written through the cache's type (a typed cache: its rows whole groups of
    // the type's blocks); a source of the cache's own description copied as its stored bytes (nothing read or rounded).
    public override void KeyValueWriteKernel(Storage source, Storage cache, Storage position, int heads, int steps, int capacity, int dim)
    {
        long start;
        fixed (float* pp = D(position))
        {
            start = IndicesOf(position, pp)[0];
        }

        long count = Math.Clamp(capacity - start, 0, steps) * (long)dim;
        if (count <= 0)
        {
            return;
        }

        fixed (float* ws = D(source), wc = D(cache))
        {
            var cacheType = NativeFloats(cache) ? null : cache.Type;
            bool sameBytes = cacheType is not null && source.Typed && source.Type.Description == cacheType.Description;
            TypedRanges ts = sameBytes ? default : new(source, 0, "KeyValueWrite");
            long unit = cacheType is null ? 1 : Math.Max(1, cacheType.Description.UnitValues), piece = Math.Max(unit, ChunkValues / unit * unit);
            for (int h = 0; h < heads; h++)
            {
                long from = (long)h * steps * dim, to = (h * (long)capacity + start) * dim;
                if (cacheType is not null && (to % unit != 0 || count % unit != 0 || (sameBytes && from % unit != 0)))
                {
                    throw new NotSupportedException($"The CPU's KeyValueWrite writes a cache of {cacheType.Name} whole groups of {unit} values: rows of {dim} from row {start} are not.");
                }

                if (sameBytes)
                {
                    long bytes = cacheType!.ByteCount(count);
                    Buffer.MemoryCopy((byte*)ws + cacheType.ByteCount(from), (byte*)wc + cacheType.ByteCount(to), bytes, bytes);
                    continue;
                }

                for (long c = 0; c < count; c += piece)
                {
                    int length = (int)Math.Min(piece, count - c);
                    var values = new ReadOnlySpan<float>(ts.At(ws, from + c, length) + from + c, length);
                    if (cacheType is null)
                    {
                        values.CopyTo(new Span<float>(wc + to + c, length));
                    }
                    else
                    {
                        cacheType.Write(values, new Span<byte>((byte*)wc + cacheType.ByteCount(to + c), (int)cacheType.ByteCount(length)));
                    }
                }
            }
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
