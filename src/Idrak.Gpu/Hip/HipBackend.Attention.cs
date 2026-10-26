// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

namespace Idrak.Gpu.Hip;

// Attention (HipKernels.Attention.cs) over key ranges (AttentionSpans: a decoder's self-attention over its cache, cross-
// attention over an encoder's memory, an encoder's own) and over a key/value cache up to the device position
// (AttentionDecode, AttentionTiled), and the cache writes and decoder masks that read the position from device memory.
// Heads larger than HipKernels.AttentionMaxDim, and the gradients, take the host fallback.
internal sealed unsafe partial class HipBackend
{
    public override void AttentionSpansKernel(Storage q, Storage keys, Storage values, Storage starts, Storage ends, Storage y, Storage? logSumExp, int heads,
        int kvHeads, int headsPerTable, int rows, int keyRows, int dim, float scale, AttentionVariant variant = default)
    {
        if (kvHeads <= 0 || heads % kvHeads != 0
            || !Attend(q, keys, values, starts, ends, y, logSumExp, heads, heads / kvHeads, headsPerTable, rows, keyRows, dim, scale, variant.Softcap, false, 1, 0))
        {
            base.AttentionSpansKernel(q, keys, values, starts, ends, y, logSumExp, heads, kvHeads, headsPerTable, rows, keyRows, dim, scale, variant);
        }
    }

    public override void AttentionDecodeKernel(Storage q, Storage keys, Storage values, Storage position, Storage y, int heads, int rowsPerHead,
        int steps, int capacity, int dim, float scale, AttentionVariant variant = default)
    {
        if (steps <= 0
            || !Attend(q, keys, values, position, position, y, null, heads, 1, 1, rowsPerHead, capacity, dim, scale, variant.Softcap, true, steps, variant.Window))
        {
            base.AttentionDecodeKernel(q, keys, values, position, y, heads, rowsPerHead, steps, capacity, dim, scale, variant);
        }
    }

    // Inference and training alike: the kernel also writes each row's log-sum-exp when asked (the gradient runs on the host).
    public override void AttentionTiledKernel(Storage q, Storage keys, Storage values, Storage position, Storage y, Storage? logSumExp, int heads,
        int rowsPerHead, int steps, int capacity, int dim, float scale, AttentionVariant variant = default)
    {
        if (steps <= 0
            || !Attend(q, keys, values, position, position, y, logSumExp, heads, 1, 1, rowsPerHead, capacity, dim, scale, variant.Softcap, true, steps, variant.Window))
        {
            base.AttentionTiledKernel(q, keys, values, position, y, logSumExp, heads, rowsPerHead, steps, capacity, dim, scale, variant);
        }
    }

    // One launch: a block per query row (attention_row_f32) for a step's few rows, a block per HipKernels.AttentionRows rows
    // (attention_rows_f32, each key and value read once for them) from that many rows on; heads along grid y. causal: the
    // ranges from the position in `starts` (with steps and window), else from starts and ends. False: no kernels, a head
    // larger than the kernels take, or more heads or row blocks than a grid holds.
    private bool Attend(Storage q, Storage keys, Storage values, Storage starts, Storage ends, Storage y, Storage? logSumExp, int heads, int group,
        int headsPerTable, int rows, int keyRows, int dim, float scale, float softcap, bool causal, int steps, int window)
    {
        if (dim <= 0 || dim > HipKernels.AttentionMaxDim || group <= 0 || headsPerTable <= 0 || keyRows < 0 || Kernels() is null)
        {
            return false;
        }

        if (heads <= 0 || rows <= 0)
        {
            return true;
        }

        int tiled = HipKernels.AttentionRows(_blockSize);
        int perBlock = rows >= tiled ? tiled : 1;
        return Grid(perBlock == 1 ? "attention_row_f32" : "attention_rows_f32", (rows + perBlock - 1) / perBlock, heads,
            P(q), P(keys), P(values), P(starts), P(ends), P(y), logSumExp is null ? 0UL : P(logSumExp),
            U(group), U(headsPerTable), U(rows), U(keyRows), U(dim), F(scale), F(softcap), U(causal ? 1 : 0), U(steps), U(window));
    }

    public override void KeyValueWriteKernel(Storage source, Storage cache, Storage position, int heads, int steps, int capacity, int dim)
    {
        if (!Elements("kv_write_f32", (long)heads * steps * dim, P(source), P(cache), P(position), U(heads), U(steps), U(capacity), U(dim)))
        {
            base.KeyValueWriteKernel(source, cache, position, heads, steps, capacity, dim);
        }
    }

    public override void DecoderMaskKernel(Storage position, Storage mask, int rows, int capacity)
    {
        if (!Elements("decoder_mask_f32", (long)rows * capacity, P(position), P(mask), U(rows), U(capacity)))
        {
            base.DecoderMaskKernel(position, mask, rows, capacity);
        }
    }
}
