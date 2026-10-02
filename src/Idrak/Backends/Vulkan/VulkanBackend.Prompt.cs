// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

namespace Idrak.Backends.Vulkan;

// Prompt processing (many rows at once) as kernels: attention over many query rows (attention_tiled, with each row's
// log-sum-exp for training) and the products through packed weights for many rows (int8_gemm, int4_gemm, bf16_gemm).
// Which kernel serves a shape is measured on the device, as the other choices (VulkanBackend.KernelTuning.cs): the tiled
// kernels at each candidate width against what ran before them (the decoding attention; the few-rows packed kernels,
// or expanding the weights to float32 for the float product), and the cooperative-matrix kernel where the device has
// cooperative matrices (VulkanBackend.Matrix.cs), so a device where the older path is faster keeps it.
internal sealed partial class VulkanBackend
{
    /// <summary>Tests and benchmarks only: the attention for many rows, tiled (true) or the decoding kernel (false), instead of the measured choice.</summary>
    internal static bool? TiledAttentionKernel { get; set; }

    /// <summary>Tests and benchmarks only: the prompt-sized packed product (0 the few-rows kernels, 1 expanding the weights, 2 the tiled kernel, 3 cooperative matrices) instead of the measured choice.</summary>
    internal static int? PackedPromptKernel { get; set; }

    // The choices' own part (WithWidth adds the width).
    private const int TiledDecode = 1, TiledKernel = 2;
    private const int PromptFewRows = 1, PromptExpand = 2, PromptTiled = 3, PromptCoop = 4;

    // ------------------------------------------------------------------ attention over many rows

    public override void AttentionTiled(Storage q, Storage keys, Storage values, Storage position, Storage y, Storage? logSumExp, int heads,
        int rowsPerHead, int steps, int capacity, int dim, float scale)
    {
        if (dim <= 0 || dim > VulkanKernels.AttentionMaxDim || steps <= 0 || !Fit(q, keys, values, position, y) || logSumExp is not null && !Fit(logSumExp))
        {
            base.AttentionTiled(q, keys, values, position, y, logSumExp, heads, rowsPerHead, steps, capacity, dim, scale);
            return;
        }

        int rows = heads * rowsPerHead;
        if (rows <= 0)
        {
            return;
        }

        bool lse = logSumExp is not null;
        Storage[] storages = lse ? [q, keys, values, position, y, logSumExp!] : [q, keys, values, position, y];
        int fallback = WithWidth(Width, TiledKernel);
        int choice;
        if (TiledAttentionKernel is bool forced)
        {
            choice = forced || lse ? fallback : WithWidth(Width, TiledDecode);
        }
        else
        {
            var key = new VulkanTuneKey(VulkanTuneOp.TiledAttention, lse ? 1 : 0, rows, rowsPerHead, steps, capacity, dim);
            if (!TryTuned(key, out choice) || !TiledAttentionValid(choice, lse))
            {
                choice = fallback;
                if (Autotune && !t_timing)
                {
                    choice = TuneTiledAttention(key, storages, fallback, heads, rowsPerHead, steps, capacity, dim, scale);
                }
            }
        }

        RunTiledAttention(choice, storages, heads, rowsPerHead, steps, capacity, dim, scale);
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
    private int TuneTiledAttention(VulkanTuneKey key, Storage[] storages, int fallback, int heads, int rowsPerHead, int steps, int capacity, int dim, float scale)
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
                RunTiledAttention(WithWidth(Width, TiledDecode), bound, heads, rowsPerHead, steps, capacity, dim, scale);
            }

            chosen = Tune(key, [.. candidates], fallback, c => RunTiledAttention(c, bound, heads, rowsPerHead, steps, capacity, dim, scale));
        });
        return chosen;
    }

    // Runs attention for many rows with `choice`: the decoding kernel, or the tiled kernel at its width (a workgroup per
    // block of TiledAttentionRows rows of a head, at most what the device takes; the kernel loops over the rest).
    private void RunTiledAttention(int choice, Storage[] storages, int heads, int rowsPerHead, int steps, int capacity, int dim, float scale)
    {
        if ((choice & Rest) == TiledDecode)
        {
            Attend("attention_decode", 0, storages.AsSpan(0, 5), heads, rowsPerHead, steps, capacity, dim, scale);
            return;
        }

        int width = WidthOf(choice), rowsPerBlock = VulkanKernels.TiledAttentionRows(width);
        long blocks = (long)heads * ((rowsPerHead + rowsPerBlock - 1) / rowsPerBlock);
        Span<byte> b = stackalloc byte[24];
        var push = new Push(b).I(heads).I(rowsPerHead).I(steps).I(capacity).I(dim).F(scale).Bytes;
        RunAt(storages.Length == 6 ? "attention_tiled_lse" : "attention_tiled", width, RowGroups(blocks), 1, 1, storages, push);
    }

    // ------------------------------------------------------------------ packed products for many rows

    public override bool PackedMatMulLarge(Layers.PackedFormat format, Storage x, Storage packed, Storage? scales, Storage y, int m, int n, int k)
    {
        var f = (VulkanKernels.PackedFormat)(int)format;
        if (m < 1 || n < 1 || k < 1 || f != VulkanKernels.PackedFormat.BFloat16 && scales is null || !FitPacked(x, packed, scales, y))
        {
            return false;
        }

        // Expanding the weights is the caller's own path: false lets it run (when that was measured faster here).
        int choice = PromptChoice(f, x, packed, scales, y, m, n, k);
        return choice != 0 && (choice & Rest) != PromptExpand && RunPrompt(choice, f, x, packed, scales, y, m, n, k);
    }

    // y = x · w for a packed product given to Int4MatMul or BFloat16MatMul (whose callers have no expanding path of their
    // own): above the few-rows limit, the measured prompt path (expansion included); else, or when that cannot run, the
    // few-rows kernels. False when nothing fits the device (the caller falls back).
    private bool PackedRows(VulkanKernels.PackedFormat format, Storage x, Storage weights, Storage? scales, Storage y, int m, int n, int k)
    {
        if (m > Capabilities.FewRows && n > 0 && k > 0 && FitPacked(x, weights, scales, y))   // larger weights: the windowed few-rows path
        {
            int choice = PromptChoice(format, x, weights, scales, y, m, n, k);
            if (choice != 0 && RunPrompt(choice, format, x, weights, scales, y, m, n, k))
            {
                return true;
            }
        }

        return PackedProduct(format, x, weights, scales, y, m, n, k);
    }

    private bool FitPacked(Storage x, Storage packed, Storage? scales, Storage y) => scales is null ? Fit(x, packed, y) : Fit(x, packed, scales, y);

    // The path of a prompt-sized packed product (0 when none can run): measured per shape among the tiled kernel at each
    // candidate width, the few-rows kernels and expanding the weights. The formula's choice (kept while nothing is
    // measured) is the few-rows kernels where they run: what int4 and bfloat16 products ran before, and faster than
    // expanding int8 weights on the device measured (a CPU driver, where they also beat the tiled kernel up to 128 rows);
    // the tiled kernel reads each weight once per block of PackedGemmEdge rows instead of once per 8, which the
    // measurement finds where it pays.
    private int PromptChoice(VulkanKernels.PackedFormat format, Storage x, Storage weights, Storage? scales, Storage y, int m, int n, int k)
    {
        var candidates = PromptCandidates(format, m, n, k);
        if (candidates.Length == 0)
        {
            return 0;
        }

        int fallback = Array.IndexOf(candidates, WithWidth(Width, PromptFewRows)) >= 0 ? WithWidth(Width, PromptFewRows)
            : Array.IndexOf(candidates, WithWidth(Width, PromptTiled)) >= 0 ? WithWidth(Width, PromptTiled) : candidates[0];
        if (PackedPromptKernel is int forced)
        {
            int wanted = WithWidth(Width, forced switch { 0 => PromptFewRows, 1 => PromptExpand, 2 => PromptTiled, _ => PromptCoop });
            return Array.IndexOf(candidates, wanted) >= 0 ? wanted : fallback;
        }

        var key = new VulkanTuneKey(VulkanTuneOp.PackedPrompt, (int)format, m, n, k);
        if (TryTuned(key, out int choice) && Array.IndexOf(candidates, choice) >= 0)
        {
            return choice;
        }

        if (!Autotune || t_timing)
        {
            return fallback;
        }

        // The other paths' own choices (k splits, product kernels) are settled first, outside the timing; every
        // candidate writes a scratch output.
        WithScratch([y.Length], scratch =>
        {
            foreach (int c in candidates)
            {
                if ((c & Rest) is not (PromptTiled or PromptCoop))
                {
                    RunPrompt(c, format, x, weights, scales, scratch[0], m, n, k);
                }
            }

            choice = Tune(key, candidates, fallback, c => RunPrompt(c, format, x, weights, scales, scratch[0], m, n, k));
        });
        return choice;
    }

    // Candidates of a prompt-sized packed product that can run here: the tiled kernel at each candidate width whose
    // blocks the device's workgroup counts take, the few-rows kernels when their row blocks fit, expanding the weights
    // when a float copy fits one storage, the cooperative-matrix kernel where the device has one and its blocks fit.
    private int[] PromptCandidates(VulkanKernels.PackedFormat format, int m, int n, int k)
    {
        var candidates = new List<int>();
        foreach (int width in CandidateWidths)
        {
            int edge = VulkanKernels.PackedGemmEdge(width);
            if ((n + edge - 1) / edge <= Limits.MaxGroupsX && (m + edge - 1) / edge <= Limits.MaxGroupsY)
            {
                candidates.Add(WithWidth(width, PromptTiled));
            }
        }

        if ((m + 7) / 8 <= Limits.MaxGroupsZ && GemvCandidates(VulkanKernels.ColumnsPerWord(format), n, 1).Length > 0)
        {
            candidates.Add(WithWidth(Width, PromptFewRows));
        }

        if ((long)k * n <= int.MaxValue && BlockBytes(k * n) <= MaxStorageBytes)
        {
            candidates.Add(WithWidth(Width, PromptExpand));
        }

        const int Block = VulkanKernels.CoopBlock;
        if ((n + Block - 1) / Block <= Limits.MaxGroupsX && (m + Block - 1) / Block <= Limits.MaxGroupsY && CoopKernel(format) is not null)
        {
            candidates.Add(WithWidth(Width, PromptCoop));
        }

        return [.. candidates];
    }

    // Runs a prompt-sized packed product with `choice`; false when the few-rows kernels cannot take it.
    private bool RunPrompt(int choice, VulkanKernels.PackedFormat format, Storage x, Storage weights, Storage? scales, Storage y, int m, int n, int k)
    {
        switch (choice & Rest)
        {
            case PromptFewRows:
                return PackedProduct(format, x, weights, scales, y, m, n, k);
            case PromptExpand:
                var w = Allocate(k * n, zeroed: false);
                try
                {
                    switch (format)
                    {
                        case VulkanKernels.PackedFormat.Int8:
                            Int8Dequantize(weights, scales!, w, k, n);
                            break;
                        case VulkanKernels.PackedFormat.Int4:
                            Int4Dequantize(weights, scales!, w, k, n);
                            break;
                        default:
                            BFloat16Dequantize(weights, w, k, n);
                            break;
                    }

                    BatchedMatMul(x, w, y, 1, m, n, k, false, false, 0f);
                }
                finally
                {
                    w.Release();                                               // reused in queue order
                }

                return true;
            case PromptCoop:
                const int Block = VulkanKernels.CoopBlock;
                Span<byte> pushed = stackalloc byte[12];
                var coopPush = new Push(pushed).I(m).I(n).I(k).Bytes;
                var coop = CoopKernel(format)!;
                uint blocksX = (uint)((n + Block - 1) / Block), blocksY = (uint)((m + Block - 1) / Block);
                if (scales is null)
                {
                    DispatchKernel(coop, blocksX, blocksY, 1, [x, weights, y], coopPush);
                }
                else
                {
                    DispatchKernel(coop, blocksX, blocksY, 1, [x, weights, scales, y], coopPush);
                }

                return true;
            default:
                int width = WidthOf(choice), edge = VulkanKernels.PackedGemmEdge(width);
                Span<byte> b = stackalloc byte[12];
                var push = new Push(b).I(m).I(n).I(k).Bytes;
                uint groupsX = (uint)((n + edge - 1) / edge), groupsY = (uint)((m + edge - 1) / edge);
                string kernel = VulkanKernels.PackedGemmName(format);
                if (scales is null)
                {
                    RunAt(kernel, width, groupsX, groupsY, 1, [x, weights, y], push);
                }
                else
                {
                    RunAt(kernel, width, groupsX, groupsY, 1, [x, weights, scales, y], push);
                }

                return true;
        }
    }
}
