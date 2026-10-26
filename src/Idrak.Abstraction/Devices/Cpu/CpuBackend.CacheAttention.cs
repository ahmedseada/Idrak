// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Buffers;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;

namespace Idrak.Abstraction.Devices.Cpu;

// Attention over int8 and bfloat16 key/value caches read in place: the decoding kernels (AttentionInt8, AttentionBFloat16,
// AttentionRowsBFloat16) through one engine, and the masked pass over every slot of a bfloat16 cache that gives the
// composed products' bits (IMaskedCacheAttentionBackend).
//
// Every output keeps the arithmetic of the kernel it replaces, operation for operation: a score is one dot product of the
// query with one cached row, a row's softmax is one pass over its scores in order, and each output element is one chain
// of weighted adds over the positions in order. So the work may be split by query rows, by positions (the scores) or by
// output elements (the weighted values) without changing a bit; only how it is split follows the machine.
internal sealed partial class CpuBackend : IMaskedCacheAttentionBackend
{
    /// <summary>
    /// Forces how these kernels split their work (true: the scores by positions or slots, then the weighted values by output
    /// elements; false: one query row, or for the masked pass one head, per work item; null: chosen by the rows or heads
    /// against the threads). Results are the same either way: the tests compare them.
    /// </summary>
    internal static bool? CacheAttentionSplit { get; set; }

    // A cache's rows as a decoding kernel reads them: the format's score, weight and weighted values.
    private interface ICacheRows
    {
        // Bytes per cached value: the weighted values are split in chunks of at least a cache line of them.
        int ElementBytes { get; }

        // The scaled score of `query` against the key at `position` of `head`, before any soft cap.
        float Score(ReadOnlySpan<float> query, int head, int position);

        // The weight of the value at `position` from its softmax term and the row's sum.
        float Weight(float term, float sum, int head, int position);

        // output[j] = Σ_c weights[c] · value_(begin + c)[from + j] for the values of `head` from position `begin` (output: a
        // chunk of the output row): each element one chain of a product then a sum per position, from zero, in order.
        void Accumulate(ReadOnlySpan<float> weights, int head, int begin, int from, Span<float> output);
    }

    // An int8 cache: each row's bytes with its scale (AttentionInt8Kernel's arithmetic).
    private readonly struct Int8Rows : ICacheRows
    {
        private readonly float[] _keys, _values, _keyScales, _valueScales;
        private readonly int _capacity, _dim, _words;
        private readonly float _scale;

        public Int8Rows(float[] keys, float[] values, float[] keyScales, float[] valueScales, int capacity, int dim, float scale)
        {
            (_keys, _values, _keyScales, _valueScales) = (keys, values, keyScales, valueScales);
            (_capacity, _dim, _words, _scale) = (capacity, dim, (dim + 3) / 4, scale);
        }

        public int ElementBytes => 1;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public float Score(ReadOnlySpan<float> query, int head, int position)
        {
            long slot = (long)head * _capacity + position;
            return Dot(query, Row(_keys, slot)) * _keyScales[slot] * _scale;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public float Weight(float term, float sum, int head, int position) => term / sum * _valueScales[(long)head * _capacity + position];

        // AddScaled's arithmetic (each byte widened to a float, times the weight, plus the sum), four vectors of the output
        // in registers across the positions, then single elements.
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public void Accumulate(ReadOnlySpan<float> weights, int head, int begin, int from, Span<float> output)
        {
            int stride = _words * 4, length = output.Length, count = weights.Length, j = 0;
            ref sbyte values = ref Unsafe.Add(ref Unsafe.As<float, sbyte>(ref MemoryMarshal.GetArrayDataReference(_values)),
                (nint)(((long)head * _capacity + begin) * stride + from));
            ref float wr = ref MemoryMarshal.GetReference(weights);
            ref float yr = ref MemoryMarshal.GetReference(output);
            if (Vector.IsHardwareAccelerated)
            {
                int w = Vector<float>.Count;
                for (; j <= length - Vector<sbyte>.Count; j += Vector<sbyte>.Count)
                {
                    Vector<float> a0 = default, a1 = default, a2 = default, a3 = default;
                    ref sbyte value = ref Unsafe.Add(ref values, j);
                    for (int c = 0; c < count; c++, value = ref Unsafe.Add(ref value, stride))
                    {
                        var s = new Vector<float>(Unsafe.Add(ref wr, c));
                        Vector.Widen(Vector.LoadUnsafe(ref value), out var lowShorts, out var highShorts);
                        Vector.Widen(lowShorts, out var i0, out var i1);
                        Vector.Widen(highShorts, out var i2, out var i3);
                        a0 = Vector.ConvertToSingle(i0) * s + a0;
                        a1 = Vector.ConvertToSingle(i1) * s + a1;
                        a2 = Vector.ConvertToSingle(i2) * s + a2;
                        a3 = Vector.ConvertToSingle(i3) * s + a3;
                    }

                    a0.StoreUnsafe(ref yr, (nuint)j);
                    a1.StoreUnsafe(ref yr, (nuint)(j + w));
                    a2.StoreUnsafe(ref yr, (nuint)(j + 2 * w));
                    a3.StoreUnsafe(ref yr, (nuint)(j + 3 * w));
                }
            }

            for (; j < length; j++)
            {
                float acc = 0f;
                ref sbyte value = ref Unsafe.Add(ref values, j);
                for (int c = 0; c < count; c++, value = ref Unsafe.Add(ref value, stride))
                {
                    acc += Unsafe.Add(ref wr, c) * value;
                }

                Unsafe.Add(ref yr, j) = acc;
            }
        }

        private ReadOnlySpan<sbyte> Row(float[] cache, long slot) => MemoryMarshal.Cast<float, sbyte>(cache.AsSpan((int)(slot * _words), _words))[.._dim];
    }

    // A bfloat16 cache: each row's halves widened as they are read (the float32 kernel's arithmetic on the widened values).
    private readonly struct BFloat16Rows : ICacheRows
    {
        private readonly float[] _keys, _values;
        private readonly int _capacity, _dim, _words;
        private readonly float _scale;

        public BFloat16Rows(float[] keys, float[] values, int capacity, int dim, float scale)
        {
            (_keys, _values) = (keys, values);
            (_capacity, _dim, _words, _scale) = (capacity, dim, (dim + 1) / 2, scale);
        }

        public int ElementBytes => 2;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public float Score(ReadOnlySpan<float> query, int head, int position) =>
            DotBFloat16(query, Halves(_keys, (long)head * _capacity + position)) * _scale;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public float Weight(float term, float sum, int head, int position) => term / sum;

        // AddScaled's arithmetic on the widened halves (times the weight, plus the sum), four vectors of the output in registers
        // across the positions, then single elements.
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public void Accumulate(ReadOnlySpan<float> weights, int head, int begin, int from, Span<float> output)
        {
            int stride = _words * 2, length = output.Length, count = weights.Length, j = 0;
            ref ushort values = ref Unsafe.Add(ref Unsafe.As<float, ushort>(ref MemoryMarshal.GetArrayDataReference(_values)),
                (nint)(((long)head * _capacity + begin) * stride + from));
            ref float wr = ref MemoryMarshal.GetReference(weights);
            ref float yr = ref MemoryMarshal.GetReference(output);
            if (Vector.IsHardwareAccelerated)
            {
                int w = Vector<float>.Count;
                for (; j <= length - 4 * w; j += 4 * w)
                {
                    Vector<float> a0 = default, a1 = default, a2 = default, a3 = default;
                    ref ushort value = ref Unsafe.Add(ref values, j);
                    for (int c = 0; c < count; c++, value = ref Unsafe.Add(ref value, stride))
                    {
                        var s = new Vector<float>(Unsafe.Add(ref wr, c));
                        Vector.Widen(Vector.LoadUnsafe(ref value), out var l0, out var h0);
                        Vector.Widen(Vector.LoadUnsafe(ref value, (nuint)(2 * w)), out var l1, out var h1);
                        a0 = Vector.AsVectorSingle(l0 << 16) * s + a0;
                        a1 = Vector.AsVectorSingle(h0 << 16) * s + a1;
                        a2 = Vector.AsVectorSingle(l1 << 16) * s + a2;
                        a3 = Vector.AsVectorSingle(h1 << 16) * s + a3;
                    }

                    a0.StoreUnsafe(ref yr, (nuint)j);
                    a1.StoreUnsafe(ref yr, (nuint)(j + w));
                    a2.StoreUnsafe(ref yr, (nuint)(j + 2 * w));
                    a3.StoreUnsafe(ref yr, (nuint)(j + 3 * w));
                }

                for (; j <= length - 2 * w; j += 2 * w)
                {
                    Vector<float> a0 = default, a1 = default;
                    ref ushort value = ref Unsafe.Add(ref values, j);
                    for (int c = 0; c < count; c++, value = ref Unsafe.Add(ref value, stride))
                    {
                        var s = new Vector<float>(Unsafe.Add(ref wr, c));
                        Vector.Widen(Vector.LoadUnsafe(ref value), out var l0, out var h0);
                        a0 = Vector.AsVectorSingle(l0 << 16) * s + a0;
                        a1 = Vector.AsVectorSingle(h0 << 16) * s + a1;
                    }

                    a0.StoreUnsafe(ref yr, (nuint)j);
                    a1.StoreUnsafe(ref yr, (nuint)(j + w));
                }
            }

            for (; j < length; j++)
            {
                float acc = 0f;
                ref ushort value = ref Unsafe.Add(ref values, j);
                for (int c = 0; c < count; c++, value = ref Unsafe.Add(ref value, stride))
                {
                    acc += Unsafe.Add(ref wr, c) * BitConverter.Int32BitsToSingle(value << 16);
                }

                Unsafe.Add(ref yr, j) = acc;
            }
        }

        private ReadOnlySpan<ushort> Halves(float[] cache, long slot) => MemoryMarshal.Cast<float, ushort>(cache.AsSpan((int)(slot * _words), _words))[.._dim];
    }

    public override void AttentionInt8Kernel(Storage q, Storage keys, Storage values, Storage keyScales, Storage valueScales, Storage position,
        Storage y, int heads, int rowsPerHead, int steps, int capacity, int dim, float scale, bool tiled, AttentionVariant variant = default)
    {
        int position0 = (int)D(position)[0];
        var rows = new Int8Rows(D(keys), D(values), D(keyScales), D(valueScales), capacity, dim, scale);
        AttendCache(rows, D(q), D(y), heads, rowsPerHead, steps, capacity, dim, position0, null, 1, variant);
    }

    public override void AttentionBFloat16Kernel(Storage q, Storage keys, Storage values, Storage position, Storage y, int heads, int rowsPerHead,
        int steps, int capacity, int dim, float scale, bool tiled, AttentionVariant variant = default)
    {
        // Straight from the bfloat16 cache: each cached key and value row is widened as it is read (no float copy of the
        // filled cache per call).
        int position0 = (int)D(position)[0];
        AttendCache(new BFloat16Rows(D(keys), D(values), capacity, dim, scale), D(q), D(y), heads, rowsPerHead, steps, capacity, dim, position0, null, 1, variant);
    }

    bool IRowAttentionBackend.AttentionRowsBFloat16(Storage q, Storage keys, Storage values, Storage position, Storage y, Storage starts, int heads, int headsPerRow,
        int rowsPerHead, int steps, int capacity, int dim, float scale, AttentionVariant variant)
    {
        // AttentionRows over a bfloat16 cache: each row from its own start (a row with none, padding, gets zeros).
        int position0 = (int)D(position)[0];
        AttendCache(new BFloat16Rows(D(keys), D(values), capacity, dim, scale), D(q), D(y), heads, rowsPerHead, steps, capacity, dim, position0, D(starts),
            headsPerRow, variant);
        return true;
    }

    // Query row r (of head r / rowsPerHead) attends the positions [begin, end): end follows its step, begin its window and,
    // for rows of different lengths, its row's start (`starts`, null when every row starts at 0).
    private static void AttendCache<TRows>(TRows cache, float[] qv, float[] yv, int heads, int rowsPerHead, int steps, int capacity, int dim, int position0,
        float[]? starts, int headsPerRow, AttentionVariant variant) where TRows : struct, ICacheRows
    {
        int rows = heads * rowsPerHead;
        long work = (long)rows * dim * Math.Max(1, position0);
        int[] begins = ArrayPool<int>.Shared.Rent(rows), counts = ArrayPool<int>.Shared.Rent(rows), offsets = ArrayPool<int>.Shared.Rent(rows + 1);
        float[]? scores = null;
        try
        {
            long total = 0;
            for (int r = 0; r < rows; r++)
            {
                int t = r % rowsPerHead % steps, end = Math.Min(position0 + t, capacity - 1) + 1, begin = variant.Start(end);
                if (starts is not null)
                {
                    begin = Math.Max((int)starts[r / rowsPerHead / headsPerRow * steps + t], begin);
                }

                (begins[r], counts[r]) = (begin, Math.Max(0, end - begin));
                offsets[r] = (int)Math.Min(total, int.MaxValue);
                total += counts[r];
            }

            offsets[rows] = (int)Math.Min(total, int.MaxValue);
            int threads = ComputeResources.MaxCpuThreads;
            bool split = total <= int.MaxValue
                         && (CacheAttentionSplit ?? (rows < 2 * threads && ComputeResources.AllowParallel && CpuTuning.SplitElements(work)));
            if (!split)
            {
                // Enough query rows for every thread: each row whole, as the float32 kernel runs.
                For(rows, work, (first, last) => AttendRows(cache, qv, yv, first, last, rowsPerHead, capacity, dim, begins, counts, variant));
                return;
            }

            // Few query rows (one decoding step): the scores split by positions, then the weighted values split by output
            // elements in chunks of at least a cache line of values, so every thread has work. Each chunk takes its row's
            // softmax of the scores itself (the same terms and weights, a pass over the row's scores): no third pass to wait on.
            scores = ArrayPool<float>.Shared.Rent((int)Math.Max(1, total));
            var s = scores;
            For((int)total, total * dim, (first, last) => ScoreRange(cache, qv, s, first, last, rows, rowsPerHead, dim, begins, offsets, variant));
            int step = Math.Max(1, Vector<byte>.Count / cache.ElementBytes);
            int least = Math.Max(CpuInfo.Current.CacheLine / cache.ElementBytes, step);
            int wanted = Math.Max(1, (2 * threads + rows - 1) / rows);
            int width = (Math.Max(least, (dim + wanted - 1) / wanted) + step - 1) / step * step, chunks = (dim + width - 1) / width;
            For(rows * chunks, total * dim, (first, last) => WeightedValues(cache, s, yv, first, last, chunks, width, rowsPerHead, capacity, dim, begins, counts, offsets));
        }
        finally
        {
            ArrayPool<int>.Shared.Return(begins);
            ArrayPool<int>.Shared.Return(counts);
            ArrayPool<int>.Shared.Return(offsets);
            if (scores is not null)
            {
                ArrayPool<float>.Shared.Return(scores);
            }
        }
    }

    // Query rows [first, last) each whole: scores, softmax, weighted values.
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static void AttendRows<TRows>(TRows cache, float[] qv, float[] yv, int first, int last, int rowsPerHead, int capacity, int dim,
        int[] begins, int[] counts, AttentionVariant variant) where TRows : struct, ICacheRows
    {
        var scores = ArrayPool<float>.Shared.Rent(capacity);
        try
        {
            for (int row = first; row < last; row++)
            {
                int h = row / rowsPerHead, begin = begins[row], count = counts[row];
                var output = yv.AsSpan(row * dim, dim);
                if (count == 0)
                {
                    output.Clear();                                              // padding: sees nothing
                    continue;
                }

                var query = qv.AsSpan(row * dim, dim);
                float max = float.NegativeInfinity;
                for (int c = 0; c < count; c++)
                {
                    scores[c] = variant.Cap(cache.Score(query, h, begin + c));
                    max = MathF.Max(max, scores[c]);
                }

                float sum = CpuMath.ExpShifted(scores.AsSpan(0, count), max);
                for (int c = 0; c < count; c++)
                {
                    scores[c] = cache.Weight(scores[c], sum, h, begin + c);
                }

                cache.Accumulate(scores.AsSpan(0, count), h, begin, 0, output);
            }
        }
        finally
        {
            ArrayPool<float>.Shared.Return(scores);
        }
    }

    // Scores [first, last) of every row's positions laid end to end (row r's from offsets[r]).
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static void ScoreRange<TRows>(TRows cache, float[] qv, float[] scores, int first, int last, int rows, int rowsPerHead, int dim,
        int[] begins, int[] offsets, AttentionVariant variant) where TRows : struct, ICacheRows
    {
        int lo = 0, hi = rows - 1;                                           // the row holding `first`: offsets[row] <= first < offsets[row + 1]
        while (lo < hi)
        {
            int mid = (lo + hi + 1) / 2;
            if (offsets[mid] <= first)
            {
                lo = mid;
            }
            else
            {
                hi = mid - 1;
            }
        }

        int row = lo;
        for (int i = first; i < last; i++)
        {
            while (i >= offsets[row + 1])
            {
                row++;
            }

            scores[i] = variant.Cap(cache.Score(qv.AsSpan(row * dim, dim), row / rowsPerHead, begins[row] + i - offsets[row]));
        }
    }

    // Output chunks [first, last) (chunk `chunks`·row + i: elements i·width onwards of that row): the row's weights from its
    // scores (the softmax terms over them, as the whole-row kernel takes them), then each element's weighted adds over the
    // row's positions in order.
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static void WeightedValues<TRows>(TRows cache, float[] scores, float[] yv, int first, int last, int chunks, int width, int rowsPerHead, int capacity,
        int dim, int[] begins, int[] counts, int[] offsets) where TRows : struct, ICacheRows
    {
        var weights = ArrayPool<float>.Shared.Rent(capacity);
        try
        {
            int weighted = -1;                                               // the row whose weights `weights` holds
            for (int item = first; item < last; item++)
            {
                int row = item / chunks, from = item % chunks * width, h = row / rowsPerHead, begin = begins[row], count = counts[row];
                var output = yv.AsSpan(row * dim + from, Math.Min(width, dim - from));
                if (count == 0)
                {
                    output.Clear();                                          // padding: sees nothing
                    continue;
                }

                if (row != weighted)
                {
                    var terms = weights.AsSpan(0, count);
                    scores.AsSpan(offsets[row], count).CopyTo(terms);
                    float max = float.NegativeInfinity;
                    foreach (float score in terms)
                    {
                        max = MathF.Max(max, score);
                    }

                    float sum = CpuMath.ExpShifted(terms, max);
                    for (int c = 0; c < count; c++)
                    {
                        terms[c] = cache.Weight(terms[c], sum, h, begin + c);
                    }

                    weighted = row;
                }

                cache.Accumulate(weights.AsSpan(0, count), h, begin, from, output);
            }
        }
        finally
        {
            ArrayPool<float>.Shared.Return(weights);
        }
    }

    bool IMaskedCacheAttentionBackend.AttentionMaskedBFloat16(Storage q, Storage keys, Storage values, Storage? mask, Storage y, int heads, int rowsPerHead,
        int maskRows, int capacity, int dim, float scale)
    {
        // The products this equals: q · keysᵀ with few query rows reads the keys in place and sums each score in vector
        // lanes (CpuMatMul.DotLanesBFloat16); more rows transpose the keys and run the tiled product, which the composed path does
        // faster than a pass of single dot products would.
        if (rowsPerHead > CpuTuning.TransposedInPlace)
        {
            return false;
        }

        float[] qv = D(q), kv = D(keys), vv = D(values), yv = D(y);
        float[]? mv = mask is null ? null : D(mask);
        var masks = new MaskRows(mv, maskRows, scale);
        int rows = heads * rowsPerHead, words = (dim + 1) / 2, threads = ComputeResources.MaxCpuThreads;
        long work = (long)rows * capacity * dim;
        var scores = ArrayPool<float>.Shared.Rent(Math.Max(1, rows * capacity));
        try
        {
            if (!(CacheAttentionSplit ?? (heads < 2 * threads && ComputeResources.AllowParallel && CpuTuning.SplitElements(work))))
            {
                // Enough heads for every thread: each head whole (its scores, softmax and weighted values), one pass.
                For(heads, work, (first, last) =>
                {
                    MaskedScores(qv, kv, scores, first * capacity, last * capacity, rowsPerHead, capacity, dim, words);
                    MaskedContext(scores, masks, vv, yv, first, last, 1, dim, rowsPerHead, capacity, dim, words);
                });
                return true;
            }

            // Few heads: the scores of every slot split by slots (each key row widened once for the head's query rows), then
            // the weights times the widened values split by output elements (each element one FMA chain over the slots in
            // order), each chunk taking its head's softmax rows itself.
            For(heads * capacity, work, (first, last) => MaskedScores(qv, kv, scores, first, last, rowsPerHead, capacity, dim, words));
            int step = Math.Max(1, Vector<byte>.Count / 2);
            int least = Math.Max(CpuInfo.Current.CacheLine / 2, step), wanted = Math.Max(1, (2 * threads + heads - 1) / heads);
            int width = (Math.Max(least, (dim + wanted - 1) / wanted) + step - 1) / step * step, chunks = (dim + width - 1) / width;
            For(heads * chunks, work, (first, last) => MaskedContext(scores, masks, vv, yv, first, last, chunks, width, rowsPerHead, capacity, dim, words));
        }
        finally
        {
            ArrayPool<float>.Shared.Return(scores);
        }

        return true;
    }

    // The mask of the masked pass (null: none), its rows repeating every maskRows query rows, and the scale of the scores.
    private readonly record struct MaskRows(float[]? Mask, int Rows, float Scale);

    // Slots [first, last) of every head (slot h·capacity + c): each query row's score as the transposed-B product sums it.
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static void MaskedScores(float[] qv, float[] kv, float[] scores, int first, int last, int rowsPerHead, int capacity, int dim, int words)
    {
        ref ushort halves = ref Unsafe.As<float, ushort>(ref MemoryMarshal.GetArrayDataReference(kv));
        ref float q0 = ref MemoryMarshal.GetArrayDataReference(qv);
        for (int slot = first; slot < last; slot++)
        {
            int h = slot / capacity, c = slot % capacity;
            ref ushort key = ref Unsafe.Add(ref halves, (nint)slot * words * 2);
            for (int i = 0; i < rowsPerHead; i++)
            {
                int r = h * rowsPerHead + i;
                scores[(long)r * capacity + c] = CpuMatMul.DotLanesBFloat16(ref Unsafe.Add(ref q0, (nint)r * dim), ref key, dim);
            }
        }
    }

    // Output chunks [first, last) (chunk `chunks`·head + i: elements i·width onwards of the head's rows): the head's softmax
    // rows (ScaleMaskSoftmax's calls on each row of scores), then each element's FMA chain over the slots in order, kept in
    // registers for a block of elements across all the slots.
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static void MaskedContext(float[] scores, MaskRows masks, float[] vv, float[] yv, int first, int last, int chunks, int width, int rowsPerHead,
        int capacity, int dim, int words)
    {
        var weights = ArrayPool<float>.Shared.Rent(rowsPerHead * capacity);
        try
        {
            int weighted = -1;                                               // the head whose softmax rows `weights` holds
            for (int item = first; item < last; item++)
            {
                int h = item / chunks, from = item % chunks * width, length = Math.Min(width, dim - from);
                if (h != weighted)
                {
                    for (int i = 0; i < rowsPerHead; i++)
                    {
                        int r = h * rowsPerHead + i;
                        var ws = weights.AsSpan(i * capacity, capacity);
                        var ms = masks.Mask is null ? default : masks.Mask.AsSpan(r % masks.Rows * capacity, capacity);
                        float max = CpuMath.ScaleAddMax(scores.AsSpan(r * capacity, capacity), ms, ws, masks.Scale);
                        CpuMath.Scale(ws, 1f / CpuMath.ExpShifted(ws, max));
                    }

                    weighted = h;
                }

                for (int i = 0; i < rowsPerHead; i++)
                {
                    WeightedHalves(weights.AsSpan(i * capacity, capacity), vv, yv.AsSpan((h * rowsPerHead + i) * dim + from, length), (long)h * capacity * words * 2 + from,
                        words * 2);
                }
            }
        }
        finally
        {
            ArrayPool<float>.Shared.Return(weights);
        }
    }

    // y[j] = Σ_c weights[c] · widened value_c[j] as one FMA chain per element from zero over c in order (a zero weight
    // skipped, as the few-row product skips it: it adds nothing to a finite sum), value_c the halves from `start` + c ·
    // `stride`. Four vectors of y stay in registers across every c, then two, then single elements.
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void WeightedHalves(ReadOnlySpan<float> weights, float[] vv, Span<float> y, long start, int stride)
    {
        int w = Vector<float>.Count, length = y.Length, slots = weights.Length, j = 0;
        ref ushort halves = ref Unsafe.Add(ref Unsafe.As<float, ushort>(ref MemoryMarshal.GetArrayDataReference(vv)), (nint)start);
        ref float wr = ref MemoryMarshal.GetReference(weights);
        ref float yr = ref MemoryMarshal.GetReference(y);
        for (; j <= length - 4 * w; j += 4 * w)
        {
            Vector<float> a0 = default, a1 = default, a2 = default, a3 = default;
            ref ushort value = ref Unsafe.Add(ref halves, j);
            for (int c = 0; c < slots; c++, value = ref Unsafe.Add(ref value, stride))
            {
                float weight = Unsafe.Add(ref wr, c);
                if (weight != 0f)
                {
                    var x = new Vector<float>(weight);
                    Vector.Widen(Vector.LoadUnsafe(ref value), out var l0, out var h0);
                    Vector.Widen(Vector.LoadUnsafe(ref value, (nuint)(2 * w)), out var l1, out var h1);
                    a0 = Vector.FusedMultiplyAdd(x, Vector.AsVectorSingle(l0 << 16), a0);
                    a1 = Vector.FusedMultiplyAdd(x, Vector.AsVectorSingle(h0 << 16), a1);
                    a2 = Vector.FusedMultiplyAdd(x, Vector.AsVectorSingle(l1 << 16), a2);
                    a3 = Vector.FusedMultiplyAdd(x, Vector.AsVectorSingle(h1 << 16), a3);
                }
            }

            a0.StoreUnsafe(ref yr, (nuint)j);
            a1.StoreUnsafe(ref yr, (nuint)(j + w));
            a2.StoreUnsafe(ref yr, (nuint)(j + 2 * w));
            a3.StoreUnsafe(ref yr, (nuint)(j + 3 * w));
        }

        for (; j <= length - 2 * w; j += 2 * w)
        {
            Vector<float> a0 = default, a1 = default;
            ref ushort value = ref Unsafe.Add(ref halves, j);
            for (int c = 0; c < slots; c++, value = ref Unsafe.Add(ref value, stride))
            {
                float weight = Unsafe.Add(ref wr, c);
                if (weight != 0f)
                {
                    var x = new Vector<float>(weight);
                    Vector.Widen(Vector.LoadUnsafe(ref value), out var l0, out var h0);
                    a0 = Vector.FusedMultiplyAdd(x, Vector.AsVectorSingle(l0 << 16), a0);
                    a1 = Vector.FusedMultiplyAdd(x, Vector.AsVectorSingle(h0 << 16), a1);
                }
            }

            a0.StoreUnsafe(ref yr, (nuint)j);
            a1.StoreUnsafe(ref yr, (nuint)(j + w));
        }

        for (; j < length; j++)
        {
            float acc = 0f;
            ref ushort value = ref Unsafe.Add(ref halves, j);
            for (int c = 0; c < slots; c++, value = ref Unsafe.Add(ref value, stride))
            {
                float weight = Unsafe.Add(ref wr, c);
                if (weight != 0f)
                {
                    acc = MathF.FusedMultiplyAdd(weight, BitConverter.Int32BitsToSingle(value << 16), acc);
                }
            }

            Unsafe.Add(ref yr, j) = acc;
        }
    }

    // Σ a[j] · widened h[j]: Dot(a, widened) operation for operation (one vector sum of products, its lanes added, the scalar tail).
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static float DotBFloat16(ReadOnlySpan<float> a, ReadOnlySpan<ushort> h)
    {
        int j = 0;
        float total = 0f;
        if (Vector.IsHardwareAccelerated && a.Length >= Vector<float>.Count)
        {
            int w = Vector<float>.Count;
            var sum = Vector<float>.Zero;
            for (; j <= a.Length - 2 * w; j += 2 * w)
            {
                Vector.Widen(new Vector<ushort>(h[j..]), out var low, out var high);
                sum += new Vector<float>(a[j..]) * Vector.AsVectorSingle(low << 16);
                sum += new Vector<float>(a[(j + w)..]) * Vector.AsVectorSingle(high << 16);
            }

            if (j <= a.Length - w)
            {
                sum += new Vector<float>(a[j..]) * Widened(h.Slice(j, w));
                j += w;
            }

            total = Vector.Sum(sum);
        }

        for (; j < a.Length; j++)
        {
            total += a[j] * BitConverter.Int32BitsToSingle(h[j] << 16);
        }

        return total;
    }

    // One vector of widened halves (h holds Vector<float>.Count of them): the last whole vector of a row whose halves do not
    // fill two.
    [MethodImpl(MethodImplOptions.NoInlining)]
    private static Vector<float> Widened(ReadOnlySpan<ushort> h)
    {
        Span<float> values = stackalloc float[Vector<float>.Count];
        HalfFloats.WidenBFloat16(h, values);
        return new Vector<float>(values);
    }
}
