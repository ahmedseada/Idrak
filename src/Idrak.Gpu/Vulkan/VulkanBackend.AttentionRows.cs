// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

namespace Idrak.Gpu.Vulkan;

// Attention for rows of different lengths decoded together (Backend.AttentionRows over a float32 cache,
// AttentionRowsBFloat16 over a bfloat16 one): each row of a padded batch attends from its own start, so a batch of
// prompts of different lengths decodes on the device instead of copying the cache to the host every step. Two kernel
// families serve it, each with the row's start read from the starts buffer: the decoding kernel (attention_rows_f32,
// attention_rows_bf16: a workgroup per query row, the positions split as decoding attention's, its splits measured as
// Attend's) and the tiled kernel (attention_tiled_rows_f32, attention_tiled_rows_bf16: a workgroup per block of query
// rows of a head, each key and value read once per block). Which one serves a shape is measured on the device
// (VulkanTuneOp.RowsAttention: the tiled kernel at every candidate width against the decoding kernel), as attention over
// many rows is (VulkanBackend.Prompt.cs); until measured, the formula's choice is the tiled kernel where a head has at
// least a block of rows at the device's width, else the decoding kernel.
internal sealed partial class VulkanBackend
{
    /// <summary>Tests and benchmarks only: attention for rows of different lengths, tiled (true) or the decoding kernel (false), instead of the measured choice.</summary>
    internal static bool? RowsAttentionKernel { get; set; }

    // The kernels of each cache format: 0 float32, 1 bfloat16.
    private static readonly string[] RowsDecodeKernels = ["attention_rows_f32", "attention_rows_bf16"];
    private static readonly string[] RowsTiledKernels = ["attention_tiled_rows_f32", "attention_tiled_rows_bf16"];

    // Whether both formats' kernels bind on this device (their storages within maxPerStageDescriptorStorageBuffers).
    private bool RowsBindable => Bindable(RowsDecodeKernels[0]) && Bindable(RowsDecodeKernels[1]) && Bindable(RowsTiledKernels[0]) && Bindable(RowsTiledKernels[1]);

    // Both kernels take every head size up to AttentionMaxDim, a sliding window and soft-capped scores.
    public override bool SupportsAttentionRows(int dim, AttentionVariant variant = default) =>
        dim > 0 && dim <= VulkanKernels.AttentionMaxDim && !KernelsOff && RowsBindable;

    public override bool AttentionRowsKernel(Storage q, Storage keys, Storage values, Storage position, Storage y, Storage starts, int heads, int headsPerRow,
        int rowsPerHead, int steps, int capacity, int dim, float scale, AttentionVariant variant = default)
    {
        if (!RowsFit(dim, steps, headsPerRow, q, keys, values, position, y, starts))
        {
            return base.AttentionRowsKernel(q, keys, values, position, y, starts, heads, headsPerRow, rowsPerHead, steps, capacity, dim, scale, variant);
        }

        AttendRows(0, [q, keys, values, starts, position, y], heads, headsPerRow, rowsPerHead, steps, capacity, dim, scale, variant);
        return true;
    }

    public override bool AttentionRowsBFloat16Kernel(Storage q, Storage keys, Storage values, Storage position, Storage y, Storage starts, int heads,
        int headsPerRow, int rowsPerHead, int steps, int capacity, int dim, float scale, AttentionVariant variant = default)
    {
        if (!RowsFit(dim, steps, headsPerRow, q, keys, values, position, y, starts))
        {
            return base.AttentionRowsBFloat16Kernel(q, keys, values, position, y, starts, heads, headsPerRow, rowsPerHead, steps, capacity, dim, scale, variant);
        }

        AttendRows(1, [q, keys, values, starts, position, y], heads, headsPerRow, rowsPerHead, steps, capacity, dim, scale, variant);
        return true;
    }

    // Whether the kernels take the call: a head size they hold, steps and heads per row, every storage in one binding.
    private bool RowsFit(int dim, int steps, int headsPerRow, params ReadOnlySpan<Storage> storages) =>
        dim > 0 && dim <= VulkanKernels.AttentionMaxDim && steps > 0 && headsPerRow > 0 && Fit(storages) && RowsBindable;

    // Runs attention for rows of different lengths (storages: q, keys, values, starts, position, y) with the forced,
    // stored, measured or formula's choice.
    private void AttendRows(int format, Span<Storage> storages, int heads, int headsPerRow, int rowsPerHead, int steps, int capacity, int dim, float scale,
        AttentionVariant variant)
    {
        int rows = heads * rowsPerHead;
        if (rows <= 0)
        {
            return;
        }

        int fallback = rowsPerHead >= VulkanKernels.TiledAttentionRows(Width) ? WithWidth(Width, TiledKernel) : WithWidth(Width, TiledDecode);
        int choice;
        if (RowsAttentionKernel is bool forced)
        {
            choice = forced ? WithWidth(Width, TiledKernel) : WithWidth(Width, TiledDecode);
        }
        else
        {
            // A window shorter than the capacity under a key of its own (variants 2 and 3, the window last).
            bool windowed = variant.Window > 0 && variant.Window < capacity;
            // Sizes that follow the data by their class (TuneSizes), as decoding attention's; a class-mate's choice this
            // shape cannot take gives the formula's.
            var key = new VulkanTuneKey(VulkanTuneOp.RowsAttention, format + (windowed ? 2 : 0), TuneSizes.Class(rows), TuneSizes.Class(rowsPerHead),
                TuneSizes.Class(steps), TuneSizes.Class(capacity), dim, windowed ? variant.Window : 0);
            if (!TryTuned(key, out choice))
            {
                choice = fallback;
                if (CanTune)
                {
                    choice = TuneRowsAttention(key, format, storages.ToArray(), fallback, heads, headsPerRow, rowsPerHead, steps, capacity, dim, scale, variant);
                }
            }
            else if (!RowsChoiceValid(choice))
            {
                choice = fallback;
            }
        }

        RunRowsAttention(choice, format, storages, heads, headsPerRow, rowsPerHead, steps, capacity, dim, scale, variant);
    }

    // Whether a choice (a stored one too) can run here: the tiled kernel at a candidate width, or the decoding kernel.
    private bool RowsChoiceValid(int choice) => (choice & Rest) switch
    {
        TiledKernel => Array.IndexOf(CandidateWidths, WidthOf(choice)) >= 0,
        TiledDecode => true,
        _ => false,
    };

    // Measures the tiled kernel at each candidate width and the decoding kernel over a nearly full cache (the position
    // read from a scratch storage holding capacity - steps, so the last step sees every position, where attention costs
    // most; the rows' own starts as given), the outputs written to a scratch storage. The decoding kernel's own choice
    // (its splits) is settled first, outside the timing. Kept apart from AttendRows so its usual calls make no closure.
    private int TuneRowsAttention(VulkanTuneKey key, int format, Storage[] storages, int fallback, int heads, int headsPerRow, int rowsPerHead, int steps,
        int capacity, int dim, float scale, AttentionVariant variant)
    {
        if (!TunesOp(key.Op))
        {
            return fallback;                                                   // IDRAK_VULKAN_TUNE_OPS leaves this one out
        }

        int[] candidates = [.. CandidateWidths.Select(w => WithWidth(w, TiledKernel)), WithWidth(Width, TiledDecode)];
        int chosen = fallback;
        WithScratch([0, 0, 0, 0, 1, storages[5].Length], scratch =>
        {
            Fill(scratch[4], 1, Math.Max(0, capacity - steps));
            var bound = new Storage[storages.Length];
            for (int i = 0; i < bound.Length; i++)
            {
                bound[i] = scratch[i] ?? storages[i];
            }

            RunRowsAttention(WithWidth(Width, TiledDecode), format, bound, heads, headsPerRow, rowsPerHead, steps, capacity, dim, scale, variant);
            chosen = Tune(key, candidates, fallback, c => RunRowsAttention(c, format, bound, heads, headsPerRow, rowsPerHead, steps, capacity, dim, scale, variant));
        });
        return chosen;
    }

    // Runs attention for rows of different lengths with `choice`: the decoding kernel (Attend, its splits measured under
    // formats 6 and 7), or the tiled kernel at its width (a workgroup per block of TiledAttentionRows rows of a head, at
    // most what the device takes; the kernel loops over the rest).
    private void RunRowsAttention(int choice, int format, Span<Storage> storages, int heads, int headsPerRow, int rowsPerHead, int steps, int capacity, int dim,
        float scale, AttentionVariant variant)
    {
        if ((choice & Rest) == TiledDecode)
        {
            Attend(RowsDecodeKernels[format], 6 + format, storages, heads, rowsPerHead, steps, capacity, dim, scale, variant, headsPerRow);
            return;
        }

        int width = WidthOf(choice), rowsPerBlock = VulkanKernels.TiledAttentionRows(width);
        long blocks = (long)heads * ((rowsPerHead + rowsPerBlock - 1) / rowsPerBlock);
        Span<byte> b = stackalloc byte[36];
        var push = new Push(b).I(heads).I(rowsPerHead).I(steps).I(capacity).I(dim).F(scale).I(variant.Window).F(variant.Softcap).I(headsPerRow).Bytes;
        RunAt(RowsTiledKernels[format], width, RowGroups(blocks), 1, 1, storages, push);
    }
}
