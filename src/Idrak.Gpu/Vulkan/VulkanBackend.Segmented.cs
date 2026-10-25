// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

namespace Idrak.Gpu.Vulkan;

// Causal attention over packed sequences (ISegmentedAttentionBackend) on the span kernels (VulkanBackend.Spans.cs): the
// range each query row sees, from its sequence's start (raised to the window's first position) to itself, is written on
// the device by segment_spans, so nothing is read back and nothing runs on the host (a training step with packed
// sequences records as a graph). A head size the span kernels do not take, or heads that do not divide into packed rows,
// go to the host fallback as before; the head sizes answered are the CPU's, which that fallback runs.
internal sealed partial class VulkanBackend : ISegmentedAttentionBackend
{
    bool ISegmentedAttentionBackend.SupportsSegmented(int dim, AttentionVariant variant) => Device.Cpu.Backend.SupportsSegmentedAttention(dim, variant);

    bool ISegmentedAttentionBackend.AttentionSegmented(Storage q, Storage keys, Storage values, Storage y, Storage? logSumExp, Storage starts, Storage ends,
        int heads, int headsPerRow, int rowsPerHead, int steps, int dim, float scale, AttentionVariant variant)
    {
        if (!SegmentsOnDevice(heads, headsPerRow, rowsPerHead, steps, dim) || !Fit(starts))
        {
            return AttentionSegmentedOnHost(q, keys, values, y, logSumExp, starts, ends, heads, headsPerRow, rowsPerHead, steps, dim, scale, variant);
        }

        var (spanStarts, spanEnds) = SegmentSpans(starts, heads / headsPerRow, rowsPerHead, steps, variant.Window);
        try
        {
            AttentionSpansKernel(q, keys, values, spanStarts, spanEnds, y, logSumExp, heads, heads, headsPerRow, rowsPerHead, steps, dim, scale, variant);
        }
        finally
        {
            spanStarts.Release();                                              // reused in queue order
            spanEnds.Release();
        }

        return true;
    }

    bool ISegmentedAttentionBackend.AttentionSegmentedBackward(Storage q, Storage keys, Storage values, Storage output, Storage logSumExp, Storage dOutput,
        Storage dq, Storage dkeys, Storage dvalues, Storage starts, Storage ends, int heads, int headsPerRow, int rowsPerHead, int steps, int dim, float scale,
        AttentionVariant variant)
    {
        if (!SegmentsOnDevice(heads, headsPerRow, rowsPerHead, steps, dim) || !Fit(starts))
        {
            return AttentionSegmentedBackwardOnHost(q, keys, values, output, logSumExp, dOutput, dq, dkeys, dvalues, starts, ends, heads, headsPerRow,
                rowsPerHead, steps, dim, scale, variant);
        }

        var (spanStarts, spanEnds) = SegmentSpans(starts, heads / headsPerRow, rowsPerHead, steps, variant.Window);
        try
        {
            AttentionSpansBackwardKernel(q, keys, values, spanStarts, spanEnds, output, logSumExp, dOutput, dq, dkeys, dvalues, heads, heads, headsPerRow,
                rowsPerHead, steps, dim, scale, variant);
        }
        finally
        {
            spanStarts.Release();
            spanEnds.Release();
        }

        return true;
    }

    // Whether the span kernels take this attention: their head sizes, whole packed rows of heads, and a table that fits
    // one storage (the kernels themselves fall back for storages they cannot bind).
    private static bool SegmentsOnDevice(int heads, int headsPerRow, int rowsPerHead, int steps, int dim) =>
        !KernelsOff && dim > 0 && dim <= VulkanKernels.AttentionMaxDim && heads > 0 && headsPerRow > 0 && heads % headsPerRow == 0 && rowsPerHead > 0
        && steps > 0 && (long)heads / headsPerRow * rowsPerHead <= int.MaxValue;

    // Each query row's range of keys ([tables, rows] starts and ends), written on the device from the packed rows' starts.
    private (Storage Starts, Storage Ends) SegmentSpans(Storage starts, int tables, int rows, int steps, int window)
    {
        int count = tables * rows;
        var spanStarts = Allocate(count, zeroed: false);
        var spanEnds = Allocate(count, zeroed: false);
        Span<byte> b = stackalloc byte[16];
        Grid("segment_spans", count, [starts, spanStarts, spanEnds], new Push(b).I(tables).I(rows).I(steps).I(Math.Max(window, 0)).Bytes);
        return (spanStarts, spanEnds);
    }
}
