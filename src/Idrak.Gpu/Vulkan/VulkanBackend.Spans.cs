// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

namespace Idrak.Gpu.Vulkan;

// Attention over one range of keys per query row (attention_spans, with each row's log-sum-exp: attention_spans_lse), a
// workgroup per block of rows of a head as attention_tiled (its rows and key tile from the device's workgroup width).
// Its gradient: Δ per row (attention_spans_delta), then attention_spans_backward_dq and attention_spans_backward_dkv,
// tiled the same way (their width measured per shape as the forward's); a device that cannot bind their storages takes
// the host fallback. And the measured choice between it and the composed scores (Backend.PrefersComposedAttention, for
// Tensor.AttentionFastest; with their gradients when training), measured and kept as the other kernel choices: per
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

    /// <summary>Whether the gradient of AttentionSpans runs on this device's kernels (the kernels are on and the device binds their storages; tests).</summary>
    internal bool SpanGradientOnDevice => !KernelsOff && Bindable("attention_spans_backward_dq") && Bindable("attention_spans_backward_dkv");

    // Δ = dOutput · output per row, then the queries' pass and the keys' pass, at the width measured for the shape (the
    // device's own until measured), as the forward's.
    public override void AttentionSpansBackwardKernel(Storage q, Storage keys, Storage values, Storage starts, Storage ends, Storage output, Storage logSumExp,
        Storage dOutput, Storage dq, Storage dkeys, Storage dvalues, int heads, int kvHeads, int headsPerTable, int rows, int keyRows, int dim, float scale,
        AttentionVariant variant = default)
    {
        long total = (long)heads * rows;
        if (dim <= 0 || dim > VulkanKernels.AttentionMaxDim || kvHeads <= 0 || headsPerTable <= 0 || total > int.MaxValue
            || !Fit(q, keys, values, starts, ends, output, logSumExp, dOutput, dq, dkeys, dvalues)
            || !Bindable("attention_spans_backward_dq") || !Bindable("attention_spans_backward_dkv"))
        {
            base.AttentionSpansBackwardKernel(q, keys, values, starts, ends, output, logSumExp, dOutput, dq, dkeys, dvalues, heads, kvHeads, headsPerTable, rows,
                keyRows, dim, scale, variant);
            return;
        }

        if (heads <= 0 || rows <= 0 || keyRows <= 0)
        {
            return;
        }

        var delta = Allocate((int)total, zeroed: false);
        try
        {
            Span<byte> b = stackalloc byte[8];
            Grid("attention_spans_delta", total, [output, dOutput, delta], new Push(b).I((int)total).I(dim).Bytes);
            int width = Width;
            var key = new VulkanTuneKey(VulkanTuneOp.SpanAttentionBackward, 0, heads, rows, keyRows, dim, kvHeads, headsPerTable);
            if (TryTuned(key, out int stored) && Array.IndexOf(CandidateWidths, stored) >= 0)
            {
                width = stored;
            }
            else if (CanTune && CandidateWidths.Length > 1)
            {
                // The candidates add into scratch gradients.
                WithScratch([dq.Length, dkeys.Length, dvalues.Length], scratch => width = Tune(key, CandidateWidths, Width,
                    w => RunSpansBackward(w, q, keys, values, starts, ends, logSumExp, dOutput, delta, scratch[0], scratch[1], scratch[2], heads, kvHeads,
                        headsPerTable, rows, keyRows, dim, scale, variant)));
            }

            RunSpansBackward(width, q, keys, values, starts, ends, logSumExp, dOutput, delta, dq, dkeys, dvalues, heads, kvHeads, headsPerTable, rows, keyRows,
                dim, scale, variant);
        }
        finally
        {
            delta.Release();                                                   // reused in queue order
        }
    }

    private void RunSpansBackward(int width, Storage q, Storage keys, Storage values, Storage starts, Storage ends, Storage logSumExp, Storage dOutput,
        Storage delta, Storage dq, Storage dkeys, Storage dvalues, int heads, int kvHeads, int headsPerTable, int rows, int keyRows, int dim, float scale,
        AttentionVariant variant)
    {
        int perBlock = VulkanKernels.TiledAttentionRows(width);
        long rowBlocks = (long)heads * ((rows + perBlock - 1) / perBlock), keyBlocks = (long)kvHeads * ((keyRows + perBlock - 1) / perBlock);
        Span<byte> b = stackalloc byte[32];
        var push = new Push(b).I(heads).I(rows).I(keyRows).I(dim).I(heads / kvHeads).I(headsPerTable).F(scale).F(variant.Softcap).Bytes;
        RunAt("attention_spans_backward_dq", width, RowGroups(rowBlocks), 1, 1, [q, keys, values, starts, ends, logSumExp, dOutput, delta, dq], push);
        RunAt("attention_spans_backward_dkv", width, RowGroups(keyBlocks), 1, 1, [q, keys, values, starts, ends, logSumExp, dOutput, delta, dkeys, dvalues], push);
    }

    public override bool PrefersComposedAttention(Storage q, Storage keys, Storage values, Storage starts, Storage ends, Storage? mask, int heads, int rows,
        int keyRows, int dim, float scale, bool training = false)
    {
        // 0 = AttentionSpans (the default, also while nothing can be measured), 1 = composed; keyed by the precision too
        // (the products may run on cooperative matrices under MixedPrecision) and by whether the gradient is timed too.
        var key = new VulkanTuneKey(VulkanTuneOp.AttentionPath, (int)MixedPrecision.Current, heads, rows, keyRows, dim, (mask is null ? 0 : 1) | (training ? 2 : 0));
        if (TryTuned(key, out int known) && known is 0 or 1)
        {
            return known == 1;
        }

        long scores = (long)heads * rows * keyRows, outputs = (long)heads * rows * dim, keyFloats = (long)heads * keyRows * dim;
        if (!CanTune || scores > int.MaxValue || outputs > int.MaxValue || keyFloats > int.MaxValue)
        {
            return false;
        }

        int chosen = 0;
        try
        {
            // Scratch: the scores, the weights, the output (also dOutput in the gradient); with the gradient the softmax's
            // gradient, the log-sum-exp, dq, dkeys and dvalues.
            int[] lengths = training
                ? [(int)scores, (int)scores, (int)outputs, (int)scores, heads * rows, (int)outputs, (int)keyFloats, (int)keyFloats]
                : [(int)scores, (int)scores, (int)outputs];
            WithScratch(lengths, scratch =>
            {
                void Run(int c)
                {
                    if (!training && c == 0)
                    {
                        AttentionSpans(q, keys, values, starts, ends, scratch[2], null, heads, heads, heads, rows, keyRows, dim, scale);
                    }
                    else if (!training)
                    {
                        ComposedAttention(q, keys, values, mask, scratch[0], scratch[1], scratch[2], heads, rows, keyRows, dim, scale);
                    }
                    else if (c == 0)
                    {
                        AttentionSpans(q, keys, values, starts, ends, scratch[2], scratch[4], heads, heads, heads, rows, keyRows, dim, scale);
                        AttentionSpansBackward(q, keys, values, starts, ends, scratch[2], scratch[4], scratch[2], scratch[5], scratch[6], scratch[7], heads, heads,
                            heads, rows, keyRows, dim, scale);
                    }
                    else
                    {
                        ComposedAttentionTraining(q, keys, values, mask, scratch[0], scratch[1], scratch[3], scratch[2], scratch[2], scratch[5], scratch[6],
                            scratch[7], heads, rows, keyRows, dim, scale);
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
