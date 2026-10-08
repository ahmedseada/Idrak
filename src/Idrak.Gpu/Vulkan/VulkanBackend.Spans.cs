// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

namespace Idrak.Gpu.Vulkan;

// Attention over one range of keys per query row (attention_spans, with each row's log-sum-exp: attention_spans_lse), a
// workgroup per block of rows of a head as attention_tiled (its rows and key tile from the device's workgroup width).
// Its gradient takes the host fallback for now. And the measured choice between it and the composed scores
// (Backend.PrefersComposedAttention, for Tensor.AttentionFastest), measured and kept as the other kernel choices: per
// device, driver and width, in the runtime's tuning file.
internal sealed partial class VulkanBackend
{
    public override void AttentionSpansKernel(Storage q, Storage keys, Storage values, Storage starts, Storage ends, Storage y, Storage? logSumExp, int heads,
        int kvHeads, int headsPerTable, int rows, int keyRows, int dim, float scale, AttentionVariant variant = default)
    {
        if (dim <= 0 || dim > VulkanKernels.AttentionMaxDim || kvHeads <= 0 || headsPerTable <= 0 || !Fit(q, keys, values, starts, ends, y)
            || logSumExp is not null && !Fit(logSumExp))
        {
            base.AttentionSpansKernel(q, keys, values, starts, ends, y, logSumExp, heads, kvHeads, headsPerTable, rows, keyRows, dim, scale, variant);
            return;
        }

        if (heads <= 0 || rows <= 0)
        {
            return;
        }

        // The workgroup width (rows per block and keys per tile follow from it): the device's own until measured, then
        // the fastest of the candidate widths for this shape, as attention_tiled's.
        bool lse = logSumExp is not null;
        int width = Width;
        var key = new VulkanTuneKey(VulkanTuneOp.SpanAttention, lse ? 1 : 0, heads, rows, keyRows, dim, kvHeads, headsPerTable);
        if (TryTuned(key, out int stored) && Array.IndexOf(CandidateWidths, stored) >= 0)
        {
            width = stored;
        }
        else if (CanTune && CandidateWidths.Length > 1)
        {
            WithScratch([y.Length, lse ? logSumExp!.Length : 0], scratch => width = Tune(key, CandidateWidths, Width,
                w => RunSpans(w, q, keys, values, starts, ends, scratch[0], lse ? scratch[1] : null, heads, kvHeads, headsPerTable, rows, keyRows, dim, scale, variant)));
        }

        RunSpans(width, q, keys, values, starts, ends, y, logSumExp, heads, kvHeads, headsPerTable, rows, keyRows, dim, scale, variant);
    }

    private void RunSpans(int width, Storage q, Storage keys, Storage values, Storage starts, Storage ends, Storage y, Storage? logSumExp, int heads,
        int kvHeads, int headsPerTable, int rows, int keyRows, int dim, float scale, AttentionVariant variant)
    {
        int rowsPerBlock = VulkanKernels.TiledAttentionRows(width);
        long blocks = (long)heads * ((rows + rowsPerBlock - 1) / rowsPerBlock);
        Span<byte> b = stackalloc byte[32];
        var push = new Push(b).I(heads).I(rows).I(keyRows).I(dim).I(heads / kvHeads).I(headsPerTable).F(scale).F(variant.Softcap).Bytes;
        if (logSumExp is null)
        {
            RunAt("attention_spans", width, RowGroups(blocks), 1, 1, [q, keys, values, starts, ends, y], push);
        }
        else
        {
            RunAt("attention_spans_lse", width, RowGroups(blocks), 1, 1, [q, keys, values, starts, ends, y, logSumExp], push);
        }
    }

    public override bool PrefersComposedAttention(Storage q, Storage keys, Storage values, Storage starts, Storage ends, Storage? mask, int heads, int rows,
        int keyRows, int dim, float scale)
    {
        // 0 = AttentionSpans (the default, also while nothing can be measured), 1 = composed; keyed by the precision too
        // (the products may run on cooperative matrices under MixedPrecision).
        var key = new VulkanTuneKey(VulkanTuneOp.AttentionPath, (int)MixedPrecision.Current, heads, rows, keyRows, dim, mask is null ? 0 : 1);
        if (TryTuned(key, out int known) && known is 0 or 1)
        {
            return known == 1;
        }

        long scores = (long)heads * rows * keyRows, outputs = (long)heads * rows * dim;
        if (!CanTune || scores > int.MaxValue || outputs > int.MaxValue)
        {
            return false;
        }

        int chosen = 0;
        try
        {
            WithScratch([(int)scores, (int)scores, (int)outputs], scratch =>
            {
                void Run(int c)
                {
                    if (c == 0)
                    {
                        AttentionSpans(q, keys, values, starts, ends, scratch[2], null, heads, heads, heads, rows, keyRows, dim, scale);
                    }
                    else
                    {
                        ComposedAttention(q, keys, values, mask, scratch[0], scratch[1], scratch[2], heads, rows, keyRows, dim, scale);
                    }
                }

                // One untimed run of each (pipelines, the choices each path needs itself), then one timed run of each: a
                // path twice as fast as the other is kept at once. Tune's rounds settle closer calls, but keep the
                // default without measuring where a run is too slow for its budget (a software driver at thousands of
                // rows), which is where a clear winner matters most.
                Run(0);
                Run(1);
                double spans, composed;
                t_timing = true;
                try
                {
                    spans = TimeRuns(0, 1, Run);
                    composed = TimeRuns(1, 1, Run);
                }
                finally
                {
                    t_timing = false;
                }

                if (composed * 2 < spans || spans * 2 < composed)
                {
                    chosen = composed < spans ? 1 : 0;
                    Interlocked.Increment(ref _measurements);
                    lock (_tuned)
                    {
                        _tuned[key] = chosen;
                    }

                    SaveTuning(key, chosen);
                }
                else
                {
                    chosen = Tune(key, [0, 1], 0, Run);
                }
            });
        }
        catch (ResourceLimitExceededException)
        {
            return false;
        }

        return chosen == 1;
    }

    // What the device reports holding no live tensors: the storage heap less what is in use (Vulkan reports no free
    // memory of its own without the memory-budget extension; blocks kept in the pool count as available).
    public override long? AvailableMemory() => StorageHeapBytes > 0 ? Math.Max(0, StorageHeapBytes - _memory.Usage.InUse) : null;
}
