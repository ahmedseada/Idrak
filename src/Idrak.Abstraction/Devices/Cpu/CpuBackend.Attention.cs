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
        int block = keys.Typed ? BlockRows(n, keyStep, dim) : n, first = 0, last = 0;
        float* k0 = null;
        for (int c = 0; c < n; c++)
        {
            if (c >= last)
            {
                (first, last) = (c, Math.Min(n, c + block));
                k0 = keys.Row((start + c) * keyStep, (last - c - 1) * keyStep + dim);
            }

            var terms = new DotTerms(q, k0 + (c - first) * keyStep);
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
        block = values.Typed ? BlockRows(n, valueStep, dim) : n;
        last = 0;
        float* v0 = null;
        for (int c = 0; c < n; c++)
        {
            if (c >= last)
            {
                (first, last) = (c, Math.Min(n, c + block));
                v0 = values.Row((start + c) * valueStep, (last - c - 1) * valueStep + dim);
            }

            var axpy = new WeightedMap(v0 + (c - first) * valueStep, y, scores[c]);
            RowMap<TV, TF, WeightedMap>(ref axpy, dim);
        }

        var divide = new DivideMap(y, y, sum);
        RowMap<TV, TF, DivideMap>(ref divide, dim);
        return max + Fn<float, F1>.Log(sum);
    }

    // The rows of `step` elements a typed operand's block takes: a chunk's worth of values, one row at least.
    private static int BlockRows(int n, long step, int dim) => (int)Math.Clamp(ChunkValues / Math.Max(1, Math.Max(step, dim)), 1, Math.Max(1, n));

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

    // `at`: views' planes read where they lie (q, keys and values each from its element offset, steps by HeadSteps), or
    // null for packed operands.
    private static void Causal(Storage q, Storage keys, Storage values, Storage position, Storage y, Storage? logSumExp, int heads, int rowsPerHead,
        int steps, int capacity, int dim, float scale, AttentionVariant variant, string operation, (long Q, long K, long V, HeadSteps Steps)? at = null)
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
            var kernel = new CausalRows(Read(q, pq, oq, 0, operation), Read(keys, pk, ok, 1, operation), Read(values, pv, ov, 2, operation), py, pl, (int)start,
                rowsPerHead, Math.Max(1, steps), capacity, dim, scale, variant, rows, default, false, 1, layout);
            Rows(operation, kernel, rows, dim * Math.Min(capacity, (int)start + Math.Max(1, steps)), 2);
        }
    }

    // An attention input's rows from element `offset` (a view's plane), read where they lie or through the input's type
    // into the thread's buffer of `slot` (a typed q, a typed cache).
    private static RowsIn Read(Storage s, float* words, long offset, int slot, string operation) => new(new TypedRanges(s, slot, operation), words, offset);

    // Where a head's rows lie: q's row i of head h at h · QHead + i · QRow, its keys' at h · KHead + c · KRow, its values'
    // at h · VHead + c · VRow (a view's planes read where they are); all 0: packed ([heads, rows, dim] each).
    private readonly record struct HeadSteps(long QHead, long QRow, long KHead, long KRow, long VHead, long VRow);

    // Row h · rowsPerHead + i sees cached positions up to position + i mod steps (from its window's first).
    private struct CausalRows(RowsIn q, RowsIn keys, RowsIn values, float* y, float* lse, int position, int rowsPerHead, int steps, int capacity, int dim,
        float scale, AttentionVariant variant, int rows, Indices starts, bool started, int headsPerRow, HeadSteps layout = default) : IRowKernel
    {
        private float* _y = y, _lse = lse;

        public readonly void Row<TV, TF>(int row) where TV : unmanaged where TF : struct, IFloats<TV>
        {
            int h = row / rowsPerHead, i = row % rowsPerHead;
            int end = Math.Min(capacity, position + i % steps + 1), start = variant.Start(end);
            if (started)
            {
                start = (int)Math.Clamp(Math.Max(start, starts[(long)(h / headsPerRow) * steps + i % steps]), 0, Math.Max(0, end));
            }
            var scores = ScoresBuffer(Math.Max(1, end - start));
            fixed (float* ps = scores)
            {
                long outRow = variant.OutputRow(row, rowsPerHead, steps);
                var at = layout.QRow == 0 && layout.KRow == 0 && layout.VRow == 0
                    ? new HeadSteps((long)rowsPerHead * dim, dim, (long)capacity * dim, dim, (long)capacity * dim, dim)
                    : layout;
                float l = AttendRow<TV, TF>(q.Row(h * at.QHead + i * at.QRow, dim), keys.Plus(h * at.KHead), values.Plus(h * at.VHead), start, end, dim, scale,
                    variant.Softcap, _y + outRow * dim, ps, at.KRow, at.VRow);
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
        Storage dq, Storage dkeys, Storage dvalues, int heads, int rowsPerHead, int steps, int capacity, int dim, float scale, AttentionVariant variant = default)
    {
        if (heads <= 0 || rowsPerHead <= 0 || dim <= 0)
        {
            return;
        }

        fixed (float* pq = D(q), pk = D(keys), pv = D(values), po = D(output), pl = V(logSumExp), pdo = D(dOutput), pdq = V(dq), pdk = V(dkeys), pdv = V(dvalues))
        {
            var kernel = new BackwardHeads(BackwardReads(q, pq, keys, pk, values, pv, output, po, dOutput, pdo, "AttentionTiledBackward"), pl, pdq, pdk, pdv, default,
                default, heads, heads, 1, rowsPerHead, capacity, Math.Max(1, steps), dim, scale, variant, mode: 0);
            Rows("AttentionTiledBackward", kernel, heads, rowsPerHead * dim * Math.Max(1, steps), 4);
        }
    }

    // The gradients' inputs, each read where it lies or through its type a row (a block of keys) at a time.
    private readonly struct AttentionReads(RowsIn q, RowsIn keys, RowsIn values, RowsIn output, RowsIn dOutput)
    {
        public readonly RowsIn Q = q, Keys = keys, Values = values, Output = output, DOutput = dOutput;
    }

    private static AttentionReads BackwardReads(Storage q, float* pq, Storage keys, float* pk, Storage values, float* pv, Storage output, float* po, Storage dOutput,
        float* pdo, string operation) =>
        new(Read(q, pq, 0, 0, operation), Read(keys, pk, 0, 1, operation), Read(values, pv, 0, 2, operation), Read(output, po, 0, 3, operation),
            Read(dOutput, pdo, 0, 4, operation));

    // A key/value head's gradients: its query heads' rows in order (spans: ranges from the tables; else causal from 0 with
    // the variant's window), each adding to dq its own row and to dkeys and dvalues the head's.
    private struct BackwardHeads(AttentionReads reads, float* lse, float* dq, float* dkeys, float* dvalues, Indices starts, Indices ends, int heads, int kvHeads,
        int headsPerTable, int rows, int keyRows, int steps, int dim, float scale, AttentionVariant variant, int mode) : IRowKernel
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
