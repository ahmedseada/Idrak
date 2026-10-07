// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

namespace Idrak.Gpu.Vulkan;

// The operations of a decoding step that used to run on the host (VulkanKernels.Sampling.cs): sampling, penalties, the
// token history, and the queries' and keys' normalization with rotary positions. With them a decoding step queues only
// device work: the host waits for the device when it reads the sampled tokens, not on every token.
internal sealed partial class VulkanBackend
{
    public override void HistoryPushKernel(Storage ids, Storage history, Storage length, int rows, int capacity)
    {
        if (!Fit(ids, history, length))
        {
            base.HistoryPushKernel(ids, history, length, rows, capacity);
            return;
        }

        Span<byte> b = stackalloc byte[8];
        Grid("history_push", rows, [ids, history, length], new Push(b).I(rows).I(capacity).Bytes);
    }

    public override void PenalizeRowsKernel(Storage logits, Storage work, Storage history, Storage length, int rows, int vocabulary,
        int rowStride, int rowOffset, int capacity, int lastN, float repeat, float presence, float frequency)
    {
        if (!Fit(logits, work, history, length))
        {
            base.PenalizeRowsKernel(logits, work, history, length, rows, vocabulary, rowStride, rowOffset, capacity, lastN, repeat, presence, frequency);
            return;
        }

        if (rows <= 0)
        {
            return;
        }

        Span<byte> b = stackalloc byte[36];
        bool narrow = vocabulary <= VulkanKernels.NarrowVocabulary;          // an invocation per row, no barriers
        Run(narrow ? "penalize_rows_narrow" : "penalize_rows", narrow ? GridGroups(rows) : RowGroups(rows), 1, 1, [logits, work, history, length], new Push(b).I(rows).I(vocabulary).I(rowStride).I(rowOffset)
            .I(capacity).I(Math.Min(lastN, capacity)).F(repeat).F(presence).F(frequency).Bytes);
    }

    // Top-k over a vocabulary of more than this many scores first keeps each slice's candidates (topk_slots), so the
    // sampler's cut-offs read a few thousand slots instead of the whole row once per k.
    private const int SlotVocabulary = 2 * VulkanKernels.SliceLength;

    public override void SampleRowsKernel(Storage logits, Storage ids, Storage stats, Storage step, int rows, int vocabulary,
        int rowStride, int rowOffset, float temperature, int topK, float topP, float minP, uint seed)
    {
        if (!Fit(logits, ids, stats, step))
        {
            base.SampleRowsKernel(logits, ids, stats, step, rows, vocabulary, rowStride, rowOffset, temperature, topK, topP, minP, seed);
            return;
        }

        if (rows <= 0)
        {
            return;
        }

        // The host computes what the CPU computes on the host, so both use the same values.
        float invT = 1f / MathF.Max(temperature, 1e-3f);
        float logMinP = minP > 0f ? MathF.Log(minP) : 0f;
        int blocks = (vocabulary + VulkanKernels.SliceLength - 1) / VulkanKernels.SliceLength;
        bool useSlots = topK > 0 && topK < vocabulary && topK <= VulkanKernels.MaxTopKSlots && vocabulary > SlotVocabulary;
        int slotCount = useSlots ? blocks * topK : 0;
        Span<byte> b = stackalloc byte[44];
        var push = new Push(b).I(vocabulary).I(rowStride).I(rowOffset).I(rows).F(invT).I(topK).F(topP).F(minP).F(logMinP).U(seed).I(slotCount).Bytes;
        if (!useSlots)
        {
            bool narrow = vocabulary <= VulkanKernels.NarrowVocabulary;
            Run(narrow ? "sample_rows_narrow" : "sample_rows", narrow ? GridGroups(rows) : RowGroups(rows), 1, 1, [logits, ids, stats, step], push);
            return;
        }

        var slots = Allocate(rows * slotCount, zeroed: false);
        var counts = Allocate(rows * slotCount, zeroed: false);
        try
        {
            Span<byte> s = stackalloc byte[28];
            Run("topk_slots", RowGroups((long)rows * blocks), 1, 1, [logits, slots, counts],
                new Push(s).I(vocabulary).I(rowStride).I(rowOffset).F(invT).I(topK).I(blocks).I(rows).Bytes);   // a workgroup per (row, slice)
            Run("sample_rows_slots", RowGroups(rows), 1, 1, [logits, ids, stats, step, slots, counts], push);
        }
        finally
        {
            slots.Release();                                                   // reused in queue order
            counts.Release();
        }
    }

    public override void RmsNormRopeKernel(Storage x, Storage gain, Storage cos, Storage sin, Storage positions, Storage y, int rows, int cols,
        float eps, float offset, int heads, int steps, int half, bool interleaved)
    {
        if (!Fit(x, gain, cos, sin, positions, y))
        {
            base.RmsNormRopeKernel(x, gain, cos, sin, positions, y, rows, cols, eps, offset, heads, steps, half, interleaved);
            return;
        }

        Span<byte> b = stackalloc byte[32];
        Rows("rms_norm_rope", rows, cols, [x, gain, y, cos, sin, positions],
            new Push(b).I(rows).I(cols).F(eps).F(offset).I(heads).I(steps).I(half).B(interleaved).Bytes);
    }

    public override void RmsNormRopePairKernel(Storage x, Storage gain, Storage y, int rows1, float eps, float offset, int heads,
        Storage x2, Storage gain2, Storage y2, int rows2, float eps2, float offset2, int heads2,
        Storage cos, Storage sin, Storage positions, int cols, int steps, int half, bool interleaved)
    {
        if (!Fit(x, gain, y, x2, gain2, y2, cos, sin, positions))
        {
            base.RmsNormRopePairKernel(x, gain, y, rows1, eps, offset, heads, x2, gain2, y2, rows2, eps2, offset2, heads2, cos, sin, positions, cols, steps, half, interleaved);
            return;
        }

        Span<byte> b = stackalloc byte[48];
        Rows("rms_norm_rope_pair", rows1 + rows2, cols, [x, gain, y, x2, gain2, y2, cos, sin, positions],
            new Push(b).I(rows1).I(cols).F(eps).F(offset).I(heads).I(rows2).F(eps2).F(offset2).I(heads2).I(steps).I(half).B(interleaved).Bytes);
    }
}
