// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Idrak.Abstraction.Operations;

namespace Idrak.Gpu.Vulkan;

// Prompt processing (many rows at once) as kernels: attention over many query rows (attention_tiled, with each row's
// log-sum-exp for training). Which kernel serves a shape is measured on the device, as the other choices
// (VulkanBackend.KernelTuning.cs): the tiled kernels at each candidate width against the decoding attention, so a device
// where the older path is faster keeps it.
internal sealed partial class VulkanBackend
{
    /// <summary>Tests and benchmarks only: the attention for many rows, tiled (true) or the decoding kernel (false), instead of the measured choice.</summary>
    internal static bool? TiledAttentionKernel { get; set; }

    // The choices' own part (WithWidth adds the width).
    private const int TiledDecode = 1, TiledKernel = 2;
    // ------------------------------------------------------------------ attention over many rows

    public override void AttentionTiledKernel(Storage q, Storage keys, Storage values, Storage position, Storage y, Storage? logSumExp, int heads,
        int rowsPerHead, int steps, int capacity, int dim, float scale, AttentionVariant variant = default)
    {
        if (dim <= 0 || dim > VulkanKernels.AttentionMaxDim || steps <= 0 || !Fit(q, keys, values, position, y) || logSumExp is not null && !Fit(logSumExp))
        {
            base.AttentionTiledKernel(q, keys, values, position, y, logSumExp, heads, rowsPerHead, steps, capacity, dim, scale, variant);
            return;
        }

        int rows = heads * rowsPerHead;
        if (rows <= 0)
        {
            return;
        }

        bool lse = logSumExp is not null;
        Span<Storage> all = [q, keys, values, position, y, logSumExp ?? y];
        var storages = lse ? all : all[..5];
        int fallback = WithWidth(Width, TiledKernel);
        int choice;
        if (TiledAttentionKernel is bool forced)
        {
            choice = forced || lse ? fallback : WithWidth(Width, TiledDecode);
        }
        else
        {
            // A window shorter than the capacity under a key of its own (variants 2 and 3, the window last).
            bool windowed = variant.Window > 0 && variant.Window < capacity;
            var key = windowed
                ? new VulkanTuneKey(VulkanTuneOp.TiledAttention, lse ? 3 : 2, rows, rowsPerHead, steps, capacity, dim, variant.Window)
                : new VulkanTuneKey(VulkanTuneOp.TiledAttention, lse ? 1 : 0, rows, rowsPerHead, steps, capacity, dim);
            if (!TryTuned(key, out choice) || !TiledAttentionValid(choice, lse))
            {
                choice = fallback;
                if (Autotune && !t_timing)
                {
                    choice = TuneTiledAttention(key, storages.ToArray(), fallback, heads, rowsPerHead, steps, capacity, dim, scale, variant);
                }
            }
        }

        RunTiledAttention(choice, storages, heads, rowsPerHead, steps, capacity, dim, scale, variant);
    }

    // Whether a choice (a stored one too) can run here: the tiled kernel at a candidate width, or the decoding kernel
    // where no log-sum-exp is asked for.
    private bool TiledAttentionValid(int choice, bool lse) => (choice & Rest) switch
    {
        TiledKernel => Array.IndexOf(CandidateWidths, WidthOf(choice)) >= 0,
        TiledDecode => !lse,
        _ => false,
    };

    // Measures the tiled kernel at each candidate width (and the decoding kernel without a log-sum-exp) over a nearly
    // full cache: the position read from a scratch storage holding capacity - steps, so the last row sees every position
    // (where attention costs most), the outputs written to scratch storages. The decoding kernel's own choice (its
    // splits) is settled first, outside the timing.
    private int TuneTiledAttention(VulkanTuneKey key, Storage[] storages, int fallback, int heads, int rowsPerHead, int steps, int capacity, int dim, float scale,
        AttentionVariant variant)
    {
        bool lse = storages.Length == 6;
        var candidates = CandidateWidths.Select(w => WithWidth(w, TiledKernel)).ToList();
        if (!lse)
        {
            candidates.Add(WithWidth(Width, TiledDecode));
        }

        int chosen = fallback;
        int[] lengths = [0, 0, 0, 1, storages[4].Length, .. lse ? new[] { storages[5].Length } : []];
        WithScratch(lengths, scratch =>
        {
            Fill(scratch[3], 1, Math.Max(0, capacity - steps));
            var bound = new Storage[storages.Length];
            for (int i = 0; i < bound.Length; i++)
            {
                bound[i] = scratch[i] ?? storages[i];
            }

            if (!lse)
            {
                RunTiledAttention(WithWidth(Width, TiledDecode), bound, heads, rowsPerHead, steps, capacity, dim, scale, variant);
            }

            chosen = Tune(key, [.. candidates], fallback, c => RunTiledAttention(c, bound, heads, rowsPerHead, steps, capacity, dim, scale, variant));
        });
        return chosen;
    }

    // Runs attention for many rows with `choice`: the decoding kernel, or the tiled kernel at its width (a workgroup per
    // block of TiledAttentionRows rows of a head, at most what the device takes; the kernel loops over the rest).
    private void RunTiledAttention(int choice, Span<Storage> storages, int heads, int rowsPerHead, int steps, int capacity, int dim, float scale,
        AttentionVariant variant)
    {
        if ((choice & Rest) == TiledDecode)
        {
            Attend("attention_decode", 0, storages[..5], heads, rowsPerHead, steps, capacity, dim, scale, variant);
            return;
        }

        int width = WidthOf(choice), rowsPerBlock = VulkanKernels.TiledAttentionRows(width);
        long blocks = (long)heads * ((rowsPerHead + rowsPerBlock - 1) / rowsPerBlock);
        Span<byte> b = stackalloc byte[32];
        var push = new Push(b).I(heads).I(rowsPerHead).I(steps).I(capacity).I(dim).F(scale).I(variant.Window).F(variant.Softcap).Bytes;

        RunAt(storages.Length == 6 ? "attention_tiled_lse" : "attention_tiled", width, RowGroups(blocks), 1, 1, storages, push);
    }

    // ------------------------------------------------------------------ packed products for many rows

}
