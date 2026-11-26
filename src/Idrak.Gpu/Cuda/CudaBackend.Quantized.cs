// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Numerics;
using System.Runtime.CompilerServices;
using Idrak.Abstraction.Operations;
using static Idrak.Gpu.Cuda.CudaDriver;

namespace Idrak.Gpu.Cuda;

// Inference kernels: norms, rotary embeddings, attention over caches and over spans, gated activations.
internal sealed unsafe partial class CudaBackend : IRowAttentionBackend
{
    /// <summary>Benchmarks only: the number of blocks decoding attention splits the cache over instead of the heuristic's.</summary>
    internal static int? DecodeSplits { get; set; }

    // The partial sums of split products (the described lane's k splits), kept between calls.
    private readonly KeptScratch _splitPartials = new();

    private Storage? _splitCounters;

    // Zeroed arrival counters for split products, one per column block; the last block of a range resets its counter.
    private Storage SplitCounters(int blocks)
    {
        // A larger buffer replaces a smaller one without freeing it: recorded graphs may still use the old one.
        var current = Volatile.Read(ref _splitCounters);
        if (current is null || current.Length < blocks)
        {
            current = Allocate(Math.Max(blocks, 4096), zeroed: true);
            Volatile.Write(ref _splitCounters, current);
        }

        return current;
    }

    public override void RmsNormKernel(Storage x, Storage y, Storage inv, int rows, int cols, float eps) =>
        LaunchRows(K("rms_norm_f32"), rows, P(x), P(y), P(inv), U(cols), F(eps), U(rows));

    public override void RmsNormBackwardKernel(Storage dy, Storage y, Storage inv, Storage dx, int rows, int cols) =>
        LaunchRows(K("rms_norm_backward_f32"), rows, P(dy), P(y), P(inv), P(dx), U(cols), U(rows));

    public override void RopeKernel(Storage x, Storage y, Storage cos, Storage sin, Storage positions, int rows, int heads, int steps, int dim, int half, bool interleaved, float sign)
    {
        int n = rows * half;
        Launch1D(K("rope_f32"), n, P(x), P(y), P(cos), P(sin), P(positions), U(heads), U(steps), U(dim), U(half), U(interleaved ? 1 : 0), F(sign), U(n));
    }

    public override void AttentionDecodeKernel(Storage q, Storage keys, Storage values, Storage position, Storage y, int heads, int rowsPerHead,
        int steps, int capacity, int dim, float scale, AttentionVariant variant = default)
    {
        if (dim > PtxKernels.DecodeMaxDim)
        {
            throw new NotSupportedException($"Decoding attention supports head sizes up to {PtxKernels.DecodeMaxDim} on CUDA.");
        }

        DecodeSplit(WindowedTuning(0, variant, capacity),
            new DecodeAttention(0, q, keys, values, y, rowsPerHead, steps, capacity, dim, scale, variant, heads * rowsPerHead), position);
    }

    // The tuning variant of a decoding attention kernel (0 float32, 6 rows of different lengths): a window shorter than the
    // capacity is measured under its own (the variant + 3), since its rows read only the window's positions.
    private static int WindowedTuning(int format, AttentionVariant variant, int capacity) =>
        variant.Window > 0 && variant.Window < capacity ? format + 3 : format;

    // Decoding attention has one block per query row (few rows: the heads of one token), so the cached positions are
    // split over several blocks per row; each writes (max, sum, weighted values) for its chunk and the last block of a
    // row to finish merges them (counted in the split counters). The split count depends only on the shapes, so
    // recorded graphs stay valid as the cache fills (chunks are computed on the device from the current length).
    // How many splits is measured once per shape (kernel `variant`, rows, capacity, head size) on this card, over
    // filled lengths 64, 128, ... up to the capacity (scratch positions), by the geometric mean of their times, so each
    // doubling of the context counts alike. (Measured at a full cache alone, one card chose 64 splits, 2%
    // faster there than 8, and ran 200 positions in 36.8 µs against 12.0 with 8.) Candidates: 1, 2, 3, 4, 6, 8, ... up to one split
    // per 64 cached positions (at most 64). The formula, about five blocks per SM (--bench-gemv: 16 rows → 22 splits,
    // 4000 positions 75.9 → ~35 µs, 1000 positions 22 → ~15 µs), is the default until then and when nothing can be
    // measured.
    // A short filled length split as many ways reads a few positions per block and merges many parts: on one card
    // 200 positions took 9.0 µs with the 24 splits measured at 4096, 7.6 with 16. So each block also takes at least
    // `minChunk` positions (the blocks past the filled length then add an empty part), measured per shape after the
    // splits over the same lengths; 1, the plain chunks, is the reference and stays unless another is faster. `attention`: the kernel's operands (RunDecode).
    // Rows and capacity are keyed by their size class (TuneSizes): a prompt's pieces and every conversation's own length
    // would otherwise measure the splits again at nearly every call; a class-mate's choice that is not one of this shape's
    // candidates gives the formula's (TryTuned).
    private void DecodeSplit(int variant, in DecodeAttention attention, Storage position)
    {
        int rows = attention.Rows, capacity = attention.Capacity, dim = attention.Dim;
        var counters = SplitCounters(rows);
        int limit = Math.Clamp(capacity / 64, 1, 64);
        int splits, minChunk = 1;
        if (DecodeSplits is int forced)
        {
            splits = Math.Clamp(forced, 1, 64);
            minChunk = DecodeMinChunk ?? 1;
        }
        else
        {
            // Once both are known, they are read without allocating (no candidate arrays, no closures).
            int formula = Math.Clamp((5 * Math.Max(1, _multiprocessors) + rows - 1) / rows, 1, Math.Min(32, limit));
            Span<int> candidates = stackalloc int[SplitCountsLength];
            candidates = candidates[..DecodeSplitCandidates(limit, formula, candidates)];
            Span<int> chunks = stackalloc int[DecodeChunksLength];
            bool known = TryTuned(new TuneKey(TuneOp.DecodeSplits, variant, TuneSizes.Class(rows), TuneSizes.Class(capacity), dim), candidates, formula, out splits);
            if (known && DecodeMinChunk is int forcedChunk)
            {
                minChunk = forcedChunk;
            }
            else if (known)
            {
                chunks = chunks[..(splits > 1 ? DecodeMinChunks(capacity, splits, chunks) : 0)];
                known = TryTuned(new TuneKey(TuneOp.DecodeMinChunk, variant, TuneSizes.Class(rows), TuneSizes.Class(capacity), dim, splits), chunks, 1, out minChunk);
            }

            if (!known)
            {
                (splits, minChunk) = MeasureDecode(variant, attention, counters, candidates.ToArray(), formula);
            }
        }

        RunDecode(attention, counters, splits, minChunk, P(position));
    }

    // Split counts decoding attention is measured with: SplitCounts up to `limit`, and the formula's choice in order.
    private static int DecodeSplitCandidates(int limit, int formula, Span<int> values)
    {
        int count = SplitCounts(limit, values);
        if (!values[..count].Contains(formula))
        {
            int at = count++;
            for (; at > 0 && values[at - 1] > formula; at--)
            {
                values[at] = values[at - 1];
            }

            values[at] = formula;
        }

        return count;
    }

    // Measures the split count, then the least chunk, over the filled lengths of DecodeTuneLengths (scratch positions).
    private (int Splits, int MinChunk) MeasureDecode(int variant, DecodeAttention attention, Storage counters, int[] candidates, int formula)
    {
        int rows = attention.Rows, capacity = attention.Capacity, dim = attention.Dim;
        int[] lengths = DecodeTuneLengths(capacity);
        Storage? positions = null;
        ulong At(int i)
        {
            if (positions is null)
            {
                positions = Allocate(lengths.Length, zeroed: false);
                Upload([.. lengths.Select(n => n - 1f)], positions);     // every row then reads that many positions
            }

            return P(positions) + (ulong)(4 * i);
        }

        try
        {
            int splits = Tune(new TuneKey(TuneOp.DecodeSplits, variant, TuneSizes.Class(rows), TuneSizes.Class(capacity), dim), candidates, formula, c =>
            {
                for (int i = 0; i < lengths.Length; i++)
                {
                    RunDecode(attention, counters, c, 1, At(i));             // writes y, which the launch after writes again
                }
            }, part: (c, i) => RunDecode(attention, counters, c, 1, At(i)), parts: lengths.Length);
            int[] chunks = splits > 1 ? DecodeMinChunks(capacity, splits) : [];
            int minChunk = DecodeMinChunk ?? Tune(new TuneKey(TuneOp.DecodeMinChunk, variant, TuneSizes.Class(rows), TuneSizes.Class(capacity), dim, splits), chunks, 1, c =>
            {
                for (int i = 0; i < lengths.Length; i++)
                {
                    RunDecode(attention, counters, splits, c, At(i));
                }
            }, part: (c, i) => RunDecode(attention, counters, splits, c, At(i)), parts: lengths.Length);
            return (splits, minChunk);
        }
        finally
        {
            positions?.Release();
        }
    }

    // A decoding attention launch but for its split count, least chunk, partial results, counters and position: the
    // kernel (0 attention_decode_f32; 3 rows of different lengths, attention_rows_f32, each from
    // starts[(head / HeadsPerRow) · steps + step]) and its operands (DecodeSplit).
    private readonly record struct DecodeAttention(int Format, Storage Q, Storage Keys, Storage Values, Storage Y, int RowsPerHead, int Steps, int Capacity,
        int Dim, float Scale, AttentionVariant Variant, int Rows, Storage? Starts = null, int HeadsPerRow = 1);

    private void RunDecode(in DecodeAttention a, Storage counters, int splits, int minChunk, ulong at)
    {
        if (splits == 1)
        {
            LaunchDecode(a, 1, 1, a.Y, counters, at);
            return;
        }

        var part = Allocate(a.Rows * splits * (a.Dim + 2), zeroed: false);
        try
        {
            LaunchDecode(a, splits, minChunk, part, counters, at);
        }
        finally
        {
            part.Release();
        }
    }

    private void LaunchDecode(in DecodeAttention a, int splits, int minChunk, Storage part, Storage counters, ulong at)
    {
        if (a.Format == 3)
        {
            Launch(K("attention_rows_f32"), (uint)a.Rows, (uint)splits, 1, PtxKernels.RowThreads, 1,
                P(a.Q), P(a.Keys), P(a.Values), at, P(a.Y), P(part), P(counters), P(a.Starts!), U(a.RowsPerHead), U(a.Steps), U(a.Capacity), U(a.Dim), F(a.Scale),
                U(minChunk), U(a.Variant.Window), F(a.Variant.Softcap), U(a.HeadsPerRow), U(a.Rows));
            return;
        }

        Launch(K("attention_decode_f32"), (uint)a.Rows, (uint)splits, 1, PtxKernels.RowThreads, 1,
            P(a.Q), P(a.Keys), P(a.Values), at, P(a.Y), P(part), P(counters), U(a.RowsPerHead), U(a.Steps), U(a.Capacity), U(a.Dim), F(a.Scale), U(minChunk),
            U(a.Variant.Window), F(a.Variant.Softcap), U(a.Rows));
    }

    /// <summary>Tests and benchmarks: the least positions per block of decoding attention, instead of the measured one.</summary>
    internal static int? DecodeMinChunk { get; set; }

    // Filled lengths the least chunk of decoding attention is timed at: 64, 128, ... below the capacity, then the capacity.
    internal static int[] DecodeTuneLengths(int capacity)
    {
        var lengths = new List<int>();
        for (int n = 64; n < capacity; n *= 2)
        {
            lengths.Add(n);
        }

        lengths.Add(Math.Max(1, capacity));
        return [.. lengths];
    }

    // Candidates for the least positions per block: 1 (plain chunks), then 8 (one per warp of a block) and its doublings
    // while a full cache still gives every split more than that (above, the full cache would run on fewer blocks than
    // measured best).
    internal static int[] DecodeMinChunks(int capacity, int splits)
    {
        Span<int> chunks = stackalloc int[DecodeChunksLength];
        return chunks[..DecodeMinChunks(capacity, splits, chunks)].ToArray();
    }

    // The same into `chunks` (DecodeChunksLength long), without allocating; returns how many.
    private static int DecodeMinChunks(int capacity, int splits, Span<int> chunks)
    {
        int count = 0;
        chunks[count++] = 1;
        for (int c = 8; c * splits < capacity; c *= 2)
        {
            chunks[count++] = c;
        }

        return count;
    }

    // 1, then 8 and its doublings below any int capacity: at most 29.
    private const int DecodeChunksLength = 32;

    public override void RmsNormAffineKernel(Storage x, Storage gain, Storage y, int rows, int cols, float eps, float offset) =>
        LaunchRows(K("rms_norm_affine_f32"), rows, P(x), P(gain), P(y), U(cols), F(eps), F(offset), U(rows));

    public override void GatedActivationKernel(Storage gate, Storage up, Storage y, int n, int kind) =>
        Launch1D(K("gated_act_f32"), n, P(gate), P(up), P(y), U(kind), U(n));

    public override void GatedActivationBackwardKernel(Storage gate, Storage up, Storage dy, Storage dgate, Storage dup, int n, int kind, int flags) =>
        Launch1D(K("gated_act_bwd_f32"), n, P(gate), P(up), P(dy), P(dgate), P(dup), U(kind), U(flags), U(n));

    // Windows and soft-caps included (attention_flash_f32 and its backward).
    public override void AttentionTiledKernel(Storage q, Storage keys, Storage values, Storage position, Storage y, Storage? logSumExp, int heads,
        int rowsPerHead, int steps, int capacity, int dim, float scale, AttentionVariant variant = default)
    {
        NotMerged(variant, nameof(AttentionTiled));

        if (dim > PtxKernels.FlashMaxDim)
        {
            base.AttentionTiledKernel(q, keys, values, position, y, logSumExp, heads, rowsPerHead, steps, capacity, dim, scale, variant);
            return;
        }

        Launch(K("attention_flash_f32"), (uint)((rowsPerHead + PtxKernels.FlashTile - 1) / PtxKernels.FlashTile), (uint)heads, 1, 128, 1,
            P(q), P(keys), P(values), P(position), P(y), logSumExp is null ? 0UL : P(logSumExp),
            U(rowsPerHead), U(steps), U(capacity), U(dim), F(scale), U(variant.Window), F(variant.Softcap));
    }

    // No pass writes token-major here (MergesAttention): a call asking for it is a caller's fault.
    private static void NotMerged(AttentionVariant variant, string operation)
    {
        if (variant.Merged > 0)
        {
            throw new InvalidOperationException($"{operation} was asked for token-major output (AttentionVariant.Merged) on a path that does not write it; ask Backend.MergesAttention first.");
        }
    }

    // Chunks of rows in parallel; several chunks' sums are added in chunk order by fold_chunks_f32 (rule 86: no atomics).
    public override void SumColumnsKernel(Storage x, long offset, int ld, Storage y, int rows, int cols)
    {
        const int Chunk = 64;
        int chunks = (rows + Chunk - 1) / Chunk;
        var partials = chunks > 1 ? Allocate(chunks * cols, zeroed: false) : null;
        try
        {
            Launch(K("sum_cols_strided_f32"), (uint)((cols + _shapes.BlockSize - 1) / _shapes.BlockSize), (uint)chunks, 1, (uint)_shapes.BlockSize, 1,
                P(x) + (ulong)offset * 4, P(y), partials is null ? P(y) : P(partials), U(rows), U(cols), U(ld), U(Chunk));
            if (partials is not null)
            {
                Launch1D(K("fold_chunks_f32"), cols, P(partials), P(y), U(cols), U(chunks), U(cols));
            }
        }
        finally
        {
            partials?.Release();                                               // reused in stream order
        }
    }

    public override void AttentionTiledBackwardKernel(Storage q, Storage keys, Storage values, Storage output, Storage logSumExp, Storage dOutput,
        Storage dq, Storage dkeys, Storage dvalues, int heads, int rowsPerHead, int steps, int capacity, int dim, float scale, AttentionVariant variant = default)
    {
        int rows = heads * rowsPerHead;
        var delta = Allocate(rows, zeroed: false);
        try
        {
            Launch1D(K("attn_bwd_d_f32"), rows, P(output), P(dOutput), P(delta), U(dim), U(rows));
            ReadOnlySpan<ulong> args = [P(q), P(keys), P(values), P(dOutput), P(logSumExp), P(delta), P(dq), P(dkeys), P(dvalues),
                U(rowsPerHead), U(steps), U(capacity), U(dim), F(scale), U(variant.Window), F(variant.Softcap)];
            Launch(K("attn_bwd_kv_f32"), (uint)((capacity + 15) / 16), (uint)heads, 1, 128, 1, args);
            Launch(K("attn_bwd_q_f32"), (uint)((rowsPerHead + 31) / 32), (uint)heads, 1, 128, 1, args);
        }
        finally
        {
            delta.Release();
        }
    }

    // Rows of different lengths: on the decoding kernel with per-row starts (attention_rows_f32), which takes every head
    // size up to DecodeMaxDim, windows and caps.
    bool IRowAttentionBackend.SupportsRows(int dim, AttentionVariant variant) => dim <= PtxKernels.DecodeMaxDim;

    bool IRowAttentionBackend.AttentionRows(Storage q, Storage keys, Storage values, Storage position, Storage y, Storage starts, int heads, int headsPerRow,
        int rowsPerHead, int steps, int capacity, int dim, float scale, AttentionVariant variant)
    {
        if (dim > PtxKernels.DecodeMaxDim)
        {
            return false;
        }

        // The decoding kernel, each row also from its sequence's start (its splits measured as a kernel of its own).
        DecodeSplit(WindowedTuning(6, variant, capacity),
            new DecodeAttention(3, q, keys, values, y, rowsPerHead, steps, capacity, dim, scale, variant, heads * rowsPerHead, starts, headsPerRow),
            position);
        return true;
    }

    public override void SoftmaxCrossEntropyRowsKernel(Storage logits, Storage targets, Storage weights, Storage losses, int rows, int vocabulary, float scale) =>
        LaunchRows(K("softmax_ce_rows_f32"), rows, PtxKernels.SoftmaxCrossEntropyThreads, P(logits), P(targets), P(weights), P(losses), U(vocabulary), F(scale), U(rows));

    public override void AddRmsNormAffineKernel(Storage a, Storage b, Storage sum, Storage gain, Storage y, int rows, int cols, float eps, float offset) =>
        LaunchRows(K("add_rms_norm_affine_f32"), rows, P(a), P(b), P(sum), P(gain), P(y), U(cols), F(eps), F(offset), U(rows));

    public override void RmsNormRopeKernel(Storage x, Storage gain, Storage cos, Storage sin, Storage positions, Storage y, int rows, int cols,
        float eps, float offset, int heads, int steps, int half, bool interleaved) =>
        LaunchRows(K("rms_norm_rope_f32"), rows, P(x), P(gain), P(cos), P(sin), P(positions), P(y),
            U(cols), F(eps), F(offset), U(heads), U(steps), U(half), U(interleaved ? 1 : 0), U(rows));

    public override void RmsNormRopePairKernel(Storage x, Storage gain, Storage y, int rows1, float eps, float offset, int heads,
        Storage x2, Storage gain2, Storage y2, int rows2, float eps2, float offset2, int heads2,
        Storage cos, Storage sin, Storage positions, int cols, int steps, int half, bool interleaved) =>
        LaunchRows(K("rms_norm_rope2_f32"), rows1 + rows2, P(x), P(gain), P(cos), P(sin), P(positions), P(y), P(x2), P(gain2), P(y2),
            U(cols), F(eps), F(offset), U(heads), U(steps), U(half), U(interleaved ? 1 : 0), U(rows1), F(eps2), F(offset2), U(heads2),
            U(rows1 + rows2));

    public override bool NormRopeHeadsKernel(Storage q, Storage k, Storage v, int batch, int steps, int heads, int kvHeads, int cols,
        Storage? gainQ, float epsQ, float offsetQ, Storage? gainK, float epsK, float offsetK, Storage? cos, Storage? sin, Storage? positions,
        int half, bool interleaved, Storage yq, Storage yk, Storage yv, Storage? position, int capacity, int stride)
    {
        int rows1 = batch * steps * heads, rows2 = batch * steps * kvHeads;
        int flags = (gainQ is not null ? 1 : 0) | (position is not null ? 4 : 0);
        static ulong Optional(Storage? s) => s is null ? 0UL : P(s);
        ReadOnlySpan<ulong> args = [P(q), Optional(gainQ), P(k), Optional(gainK), P(v), Optional(cos), Optional(sin),
            Optional(positions), Optional(position), P(yq), P(yk), P(yv), U(cols), F(epsQ), F(offsetQ), U(heads), F(epsK), F(offsetK), U(kvHeads),
            U(steps), U(cos is null ? 0 : half), U(interleaved ? 1 : 0), U(rows1), U(rows2), U(flags), U(capacity), U(stride), U(rows1 + 2 * rows2)];

        // A row is one head: its block's threads measured once per head size and row count class (about one thread a pair of
        // values before), since 256 threads over 64 or 128 values leave most idle and pay the block's barriers for nothing.
        // Each run writes the outputs (and cache slots) the final run writes again.
        int rows = rows1 + 2 * rows2, formula = RowThreadsFor(cols);
        var key = new TuneKey(TuneOp.RowThreads, 0, cols, TuneSizes.Class(rows), interleaved ? 1 : 0);
        if (!TryTuned(key, RowThreadCounts, formula, out int threads))
        {
            threads = MeasureRowThreads(key, formula, K("norm_rope_heads_f32"), rows, args.ToArray());
        }

        LaunchRows(K("norm_rope_heads_f32"), rows, threads, args);
        return true;
    }

    // Block sizes a short-row kernel is measured with (each a whole number of warps the reductions take).
    private static readonly int[] RowThreadCounts = [32, 64, 128, 256];

    // Before measuring: the power of two near half the row's length, from one warp to RowThreads.
    private static int RowThreadsFor(int cols) => Math.Clamp((int)BitOperations.RoundUpToPowerOf2((uint)Math.Max(1, cols / 2)), 32, PtxKernels.RowThreads);

    [MethodImpl(MethodImplOptions.NoInlining)]
    private int MeasureRowThreads(TuneKey key, int formula, IntPtr function, int rows, ulong[] args) =>
        Tune(key, RowThreadCounts, formula, threads => LaunchRows(function, rows, threads, args));
}
