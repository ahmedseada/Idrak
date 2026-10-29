// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Globalization;

namespace Idrak.Gpu.Hip;

// HIP's attention on its own GCN kernels (GcnKernels.Attention.cs), where CUDA overrides it: decoding and the tiled form
// over a float32, bfloat16 or int8 key/value cache (the log-sum-exp for training, windows and soft-caps), attention over
// key ranges (grouped heads, tables of ranges, the log-sum-exp) and its strided form read in place from a layer's packed
// rows, and the int8 cache's scores and weighted values. One block of 256 takes a query row (min(rows, the device's
// grid) blocks; the kernels walk the rows grid-stride, rule 84) and, for few rows over many keys, a split of its keys
// in grid y (rule 81): the splits and their positions follow the shapes alone (IDRAK_HIP_ATTENTION_SPLIT positions a
// split, 512 by default, at most 64 splits and 4,096 blocks), so a row's sums keep one order on every device (rule 86),
// and attention_merge_f32 adds the splits' parts in split order. Each launch is labelled for --profile with its shape
// (rows x positions x head size, the positions the cache's capacity or the key rows: an upper bound, the filled length
// being on the device) and its floating-point operations (Cost). The host fallback takes over when the kernels are
// unavailable, a head is past GcnKernels.AttentionMaxDim, a block may not have the kernels' LDS, or a key/value head
// spans 4 GiB or more (the kernels' 32-bit offsets within a head).
internal sealed unsafe partial class HipBackend
{
    /// <summary>The setting for the key positions one split of a row's keys takes (a multiple of 256; 0 turns splits off; default 512).</summary>
    internal const string AttentionSplitVariable = "IDRAK_HIP_ATTENTION_SPLIT";

    // The part floats a split launch may take (rows × splits × (dim + 2)), and the blocks (rows × splits) it adds splits up to.
    private const long AttentionMostPartFloats = 1L << 24;
    private const int AttentionSplitBlocks = 4096;

    /// <summary>
    /// Key positions a split of a row's keys takes (IDRAK_HIP_ATTENTION_SPLIT, <paramref name="setting"/>): 512 by default,
    /// a whole number rounded up to a multiple of the 256-key tile, 0 for no splits.
    /// </summary>
    internal static int AttentionSplitPositions(string? setting)
    {
        if (!int.TryParse(setting, NumberStyles.None, CultureInfo.InvariantCulture, out int positions))
        {
            return 2 * GcnKernels.AttentionTile;
        }

        return positions == 0 ? 0 : (int)Math.Min(1 << 30, ((long)positions + GcnKernels.AttentionTile - 1) / GcnKernels.AttentionTile * GcnKernels.AttentionTile);
    }

    private static readonly int SplitPositions = AttentionSplitPositions(Environment.GetEnvironmentVariable(AttentionSplitVariable));

    /// <summary>
    /// The splits of <paramref name="rows"/> rows' keys over <paramref name="positions"/> positions of head size
    /// <paramref name="dim"/>, <paramref name="perSplit"/> positions a split: (1, positions) without splits, else the split
    /// count and the positions a split takes (a multiple of the tile). From the shapes alone (rule 86): at most
    /// <see cref="GcnKernels.AttentionMostSplits"/> splits, rows × splits up to 4,096 blocks, parts within 2^24 floats.
    /// </summary>
    internal static (int Splits, int Chunk) AttentionSplits(int rows, int positions, int dim, int perSplit)
    {
        int whole = Math.Max(positions, 1);
        if (perSplit <= 0 || positions <= perSplit || rows <= 0)
        {
            return (1, whole);
        }

        long splits = Math.Min(Math.Min(((long)positions + perSplit - 1) / perSplit, GcnKernels.AttentionMostSplits), Math.Max(1, AttentionSplitBlocks / rows));
        while (splits > 1 && rows * splits * (dim + 2L) > AttentionMostPartFloats)
        {
            splits--;
        }

        if (splits <= 1)
        {
            return (1, whole);
        }

        long chunk = ((positions + splits - 1) / splits + GcnKernels.AttentionTile - 1) / GcnKernels.AttentionTile * GcnKernels.AttentionTile;
        splits = (positions + chunk - 1) / chunk;
        return splits <= 1 ? (1, whole) : ((int)splits, (int)chunk);
    }

    /// <summary>The cost of an attention or shape kernel from the tables at the top of GcnKernels.Attention.cs and .Shapes.cs; (0, 0) for another.</summary>
    private static (double Flops, double Bytes) AttentionShapeCost(string kernel, bool variant) => kernel switch
    {
        "attention_decode_f32" or "attention_spans_f32" => (4, 8),                // per row × position × dimension
        "attention_decode_bf16" => (4, 4),
        "attention_decode_int8" => (4, 2),
        "attention_merge_f32" => (3, 4),                                          // per row × split × dimension
        "attn_scores_int8" => (2, 5),                                             // per score × dimension
        "attn_context_int8" => (2, 3),                                            // per output × position
        "kv_write_f32" or "gather_f32" or "gather_bf16_cols_f32" => (0, 8),
        "kv_write_bf16" or "gather_bf16_f32" => (0, 6),
        "kv_write_int8" => (3, 9),
        "decoder_mask_f32" => (0, 4),
        "permute_f32" or "copy2d_f32" => variant ? (1, 12) : (0, 8),             // variant: adding into the output
        _ => (0, 0),
    };

    // An attention kernel for head size dim, or null: the kernels unavailable, a head past the kernels' slots, or a block
    // that may not have their LDS (the device's reported limit, rule 84).
    private GcnFunction? AttentionKernel(string name, int dim) =>
        dim >= 1 && dim <= GcnKernels.AttentionMaxDim && GcnKernels.AttentionLdsBytes <= DeviceLimits.SharedMemoryPerBlock ? Gcn(name) : null;

    // The splits a launch of `rows` rows over `positions` keys takes here, or null when the device's grid has no room for them.
    private (int Splits, int Chunk)? SplitsFor(int rows, int positions, int dim)
    {
        var split = AttentionSplits(rows, positions, dim, SplitPositions);
        return split.Splits <= DeviceLimits.MaxGridDimY ? split : null;
    }

    // Launches an attention kernel over min(rows, the device's grid) × splits blocks with `values` (its part pointer at
    // `partArgument` filled in), then with several splits attention_merge_f32 over the parts into y (laid out as
    // `layout`: heads, rows, ybatch, yhead, yrow of the kernel), each labelled for the profile.
    private void LaunchAttention(GcnFunction function, GcnFunction merge, int rows, int positions, int dim, (int Splits, int Chunk) split, Span<ulong> values,
        int partArgument, ulong y, ulong lse, (int Heads, int Rows, int YBatch, int YHead, int YRow) layout)
    {
        var part = split.Splits > 1 ? Allocate((int)((long)rows * split.Splits * (dim + 2)), zeroed: false) : null;
        try
        {
            values[partArgument] = part is null ? 0UL : P(part);
            if (Profiling)
            {
                LabelNextLaunch(string.Create(CultureInfo.InvariantCulture, $"{function.Name} {rows}x{positions}x{dim}"),
                    Cost(function.Name).Flops * rows * (double)positions * dim);
            }

            uint grid = (uint)Math.Min(rows, DeviceLimits.MaxGridDimX);
            LaunchKernel(function, grid, (uint)split.Splits, values);
            if (part is not null)
            {
                if (Profiling)
                {
                    LabelNextLaunch(string.Create(CultureInfo.InvariantCulture, $"{merge.Name} {rows}x{split.Splits}x{dim}"),
                        Cost(merge.Name).Flops * rows * (double)split.Splits * dim);
                }

                LaunchKernel(merge, grid, 1, P(part), y, lse, U(split.Splits), U(dim), U(layout.Heads), U(layout.Rows), U(layout.YBatch), U(layout.YHead),
                    U(layout.YRow), U(rows));
            }
        }
        finally
        {
            part?.Release();                                                     // reused in stream order
        }
    }

    // Attention over a cache up to the device position (attention_decode_*): false when it cannot run here (the host's).
    // With `starts` (attention_rows_*, HipBackend.Decoding.cs), each row also from its sequence's start, headsPerRow heads a
    // packed row of the table.
    private bool CacheAttention(string kernel, Storage q, Storage keys, Storage values, Storage? keyScales, Storage? valueScales, Storage position, Storage y,
        Storage? logSumExp, int heads, int rowsPerHead, int steps, int capacity, int dim, float scale, int words, AttentionVariant variant, Storage? starts = null,
        int headsPerRow = 0)
    {
        long rows = (long)heads * rowsPerHead;
        int rowWords = words > 0 ? words : dim;
        if (AttentionKernel(kernel, dim) is not { } function || Gcn("attention_merge_f32") is not { } merge || steps <= 0 || capacity <= 0 || rows > int.MaxValue
            || (long)capacity * rowWords * sizeof(float) >= 1L << 32 || SplitsFor((int)rows, capacity, dim) is not { } split)
        {
            return false;
        }

        if (rows <= 0)
        {
            return true;
        }

        bool int8 = keyScales is not null;
        Span<ulong> arguments = stackalloc ulong[23];
        int at = 0;
        arguments[at++] = P(q);
        arguments[at++] = P(keys);
        arguments[at++] = P(values);
        if (int8)
        {
            arguments[at++] = P(keyScales!);
            arguments[at++] = P(valueScales!);
        }

        arguments[at++] = P(position);
        arguments[at++] = P(y);
        arguments[at++] = logSumExp is null ? 0UL : P(logSumExp);
        int partArgument = at++;
        if (starts is not null)
        {
            arguments[at++] = P(starts);
        }

        arguments[at++] = U(rowsPerHead);
        arguments[at++] = U(steps);
        arguments[at++] = U(capacity);
        arguments[at++] = U(dim);
        arguments[at++] = F(scale);
        if (words > 0)
        {
            arguments[at++] = U(words);
        }

        arguments[at++] = U(Math.Max(0, variant.Window));
        arguments[at++] = F(variant.Softcap);
        arguments[at++] = U(split.Chunk);
        arguments[at++] = U(split.Splits);
        if (starts is not null)
        {
            arguments[at++] = U(headsPerRow);
        }

        arguments[at++] = U((int)rows);
        LaunchAttention(function, merge, (int)rows, capacity, dim, split, arguments[..at], partArgument, P(y), logSumExp is null ? 0UL : P(logSumExp),
            (1, (int)rows, 0, 0, dim));
        return true;
    }

    // Attention over key ranges (attention_spans_f32) with the layouts of AttentionSpans (batch 1) or its strided form:
    // false when it cannot run here (the host's).
    private bool SpanAttention(ulong q, ulong keys, ulong values, Storage starts, Storage ends, Storage y, Storage? logSumExp, int batch, int heads, int kvHeads,
        int headsPerTable, int rows, int keyRows, int dim, float scale, float softcap, (int Batch, int Head, int Row) qLayout, (int Batch, int Head, int Row) kLayout,
        (int Batch, int Head, int Row) yLayout)
    {
        long all = (long)batch * heads * rows;
        if (AttentionKernel("attention_spans_f32", dim) is not { } function || Gcn("attention_merge_f32") is not { } merge || heads <= 0 || kvHeads <= 0
            || heads % kvHeads != 0 || headsPerTable <= 0 || rows <= 0 || keyRows < 0 || all > int.MaxValue
            || (long)keyRows * kLayout.Row * sizeof(float) >= 1L << 32 || SplitsFor((int)all, keyRows, dim) is not { } split)
        {
            return false;
        }

        if (all == 0)
        {
            return true;
        }

        Span<ulong> arguments =
        [
            q, keys, values, P(starts), P(ends), P(y), logSumExp is null ? 0UL : P(logSumExp), 0UL,
            U(heads), U(heads / kvHeads), U(headsPerTable), U(rows), U(keyRows), U(dim), F(scale), F(softcap), U(split.Chunk), U(split.Splits),
            U(qLayout.Batch), U(qLayout.Head), U(qLayout.Row), U(kLayout.Batch), U(kLayout.Head), U(kLayout.Row), U(yLayout.Batch), U(yLayout.Head), U(yLayout.Row),
            U((int)all),
        ];
        LaunchAttention(function, merge, (int)all, keyRows, dim, split, arguments, 7, P(y), logSumExp is null ? 0UL : P(logSumExp),
            (heads, rows, yLayout.Batch, yLayout.Head, yLayout.Row));
        return true;
    }

    public override void AttentionDecodeKernel(Storage q, Storage keys, Storage values, Storage position, Storage y, int heads, int rowsPerHead,
        int steps, int capacity, int dim, float scale, AttentionVariant variant = default)
    {
        if (!CacheAttention("attention_decode_f32", q, keys, values, null, null, position, y, null, heads, rowsPerHead, steps, capacity, dim, scale, 0, variant))
        {
            base.AttentionDecodeKernel(q, keys, values, position, y, heads, rowsPerHead, steps, capacity, dim, scale, variant);
        }
    }

    // The tiled form: the same kernel, one block a query row (many rows fill the device), with the log-sum-exp when asked.
    public override void AttentionTiledKernel(Storage q, Storage keys, Storage values, Storage position, Storage y, Storage? logSumExp, int heads,
        int rowsPerHead, int steps, int capacity, int dim, float scale, AttentionVariant variant = default)
    {
        if (!CacheAttention("attention_decode_f32", q, keys, values, null, null, position, y, logSumExp, heads, rowsPerHead, steps, capacity, dim, scale, 0, variant))
        {
            base.AttentionTiledKernel(q, keys, values, position, y, logSumExp, heads, rowsPerHead, steps, capacity, dim, scale, variant);
        }
    }

    public override void AttentionBFloat16Kernel(Storage q, Storage keys, Storage values, Storage position, Storage y, int heads, int rowsPerHead,
        int steps, int capacity, int dim, float scale, bool tiled, AttentionVariant variant = default)
    {
        if (!CacheAttention("attention_decode_bf16", q, keys, values, null, null, position, y, null, heads, rowsPerHead, steps, capacity, dim, scale, (dim + 1) / 2,
                variant))
        {
            base.AttentionBFloat16Kernel(q, keys, values, position, y, heads, rowsPerHead, steps, capacity, dim, scale, tiled, variant);
        }
    }

    public override void AttentionInt8Kernel(Storage q, Storage keys, Storage values, Storage keyScales, Storage valueScales, Storage position,
        Storage y, int heads, int rowsPerHead, int steps, int capacity, int dim, float scale, bool tiled, AttentionVariant variant = default)
    {
        if (!CacheAttention("attention_decode_int8", q, keys, values, keyScales, valueScales, position, y, null, heads, rowsPerHead, steps, capacity, dim, scale,
                (dim + 3) / 4, variant))
        {
            base.AttentionInt8Kernel(q, keys, values, keyScales, valueScales, position, y, heads, rowsPerHead, steps, capacity, dim, scale, tiled, variant);
        }
    }

    public override void AttentionSpansKernel(Storage q, Storage keys, Storage values, Storage starts, Storage ends, Storage y, Storage? logSumExp, int heads,
        int kvHeads, int headsPerTable, int rows, int keyRows, int dim, float scale, AttentionVariant variant = default)
    {
        int block = rows * dim;
        if (!SpanAttention(P(q), P(keys), P(values), starts, ends, y, logSumExp, 1, heads, kvHeads, headsPerTable, rows, keyRows, dim, scale, variant.Softcap,
                (0, block, dim), (0, keyRows * dim, dim), (0, block, dim)))
        {
            base.AttentionSpansKernel(q, keys, values, starts, ends, y, logSumExp, heads, kvHeads, headsPerTable, rows, keyRows, dim, scale, variant);
        }
    }

    // In place over the packed rows: the query, key and value heads read at their strides, no copies (the base copies the
    // heads out and back).
    public override void AttentionSpansStridedKernel(Storage q, long qOffset, Storage k, long kOffset, Storage v, long vOffset, int qRow, int kRow,
        Storage starts, Storage ends, Storage y, int batch, int heads, int kvHeads, int headsPerTable, int rows, int dim, float scale)
    {
        long qBatch = (long)rows * qRow, kBatch = (long)rows * kRow, yBatch = (long)rows * heads * dim;
        if (batch <= 0 || qOffset < 0 || kOffset < 0 || vOffset < 0 || Math.Max(qBatch, Math.Max(kBatch, yBatch)) > int.MaxValue || (long)heads * dim > int.MaxValue
            || !SpanAttention(P(q) + (ulong)qOffset * sizeof(float), P(k) + (ulong)kOffset * sizeof(float), P(v) + (ulong)vOffset * sizeof(float), starts, ends, y,
                null, batch, heads, kvHeads, headsPerTable, rows, rows, dim, scale, 0f, ((int)qBatch, dim, qRow), ((int)kBatch, dim, kRow),
                ((int)yBatch, dim, heads * dim)))
        {
            base.AttentionSpansStridedKernel(q, qOffset, k, kOffset, v, vOffset, qRow, kRow, starts, ends, y, batch, heads, kvHeads, headsPerTable, rows, dim, scale);
        }
    }

    public override void AttentionScoresInt8Kernel(Storage q, Storage cache, Storage scales, Storage y, int rows, int steps, int capacity, int dim)
    {
        int words = (dim + 3) / 4;
        long n = (long)rows * steps * capacity;
        if (dim <= 0 || steps <= 0 || capacity <= 0 || n >= GcnKernels.ShapeMostElements || (long)rows * steps * dim >= GcnKernels.ShapeMostElements
            || (long)rows * capacity * words >= GcnKernels.ShapeMostElements || ShapeKernel("attn_scores_int8") is not { } f)
        {
            base.AttentionScoresInt8Kernel(q, cache, scales, y, rows, steps, capacity, dim);
            return;
        }

        LaunchShape(f, (int)n, Cost(f.Name).Flops * n * dim, P(q), P(cache), P(scales), P(y), U(steps), U(capacity), U(dim), U(words), U((int)n));
    }

    public override void AttentionContextInt8Kernel(Storage weights, Storage cache, Storage scales, Storage y, int rows, int steps, int capacity, int dim)
    {
        int words = (dim + 3) / 4;
        long n = (long)rows * steps * words;
        if (dim <= 0 || steps <= 0 || capacity <= 0 || (long)rows * steps * dim >= GcnKernels.ShapeMostElements || (long)rows * steps * capacity >= GcnKernels.ShapeMostElements
            || (long)rows * capacity * words >= GcnKernels.ShapeMostElements || ShapeKernel("attn_context_int8") is not { } f)
        {
            base.AttentionContextInt8Kernel(weights, cache, scales, y, rows, steps, capacity, dim);
            return;
        }

        LaunchShape(f, (int)n, Cost(f.Name).Flops * rows * steps * (double)capacity * dim, P(weights), P(cache), P(scales), P(y), U(steps), U(capacity), U(dim),
            U(words), U((int)n));
    }
}
