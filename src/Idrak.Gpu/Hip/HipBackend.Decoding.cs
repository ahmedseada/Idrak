// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Globalization;

namespace Idrak.Gpu.Hip;

// HIP's decoding operations on its own GCN kernels (GcnKernels.Decoding.cs), and the decoding contracts CUDA implements:
// - IRowAttentionBackend: rows of different lengths decoded together over a float32 cache
//   (attention_rows_f32: the decoding kernel with each row's start), and ISegmentedAttentionBackend:
//   packed sequences, their key ranges written on the device (segment_spans_f32) for attention_spans_f32. Their head sizes
//   are the CPU's; a shape the kernels cannot take has no pass here. The segmented gradient is G5b's: none yet.
// - The rows of decoding: the arg-max, class matches, the residual addition with its norm, the rotary embedding alone
//   and after the norm (one tensor, queries and keys in one launch, and the attention layer's heads into their layouts
//   and cache, NormRopeHeads true as on CUDA), one-hot rows, the token history and its repetition penalties.
// - The sampler: CUDA's paths (the whole row; top-k ≤ 64 over a vocabulary past four candidate slices through the
//   candidates; greedy rows sampled by their last candidate block, one launch), each draw the CPU's (GcnKernels' Sampler).
// Each launch is labelled for --profile with its shape and floating-point operations (Cost, ProductCost).
internal sealed unsafe partial class HipBackend : IRowAttentionBackend, ISegmentedAttentionBackend
{
    /// <summary>
    /// The setting for how a split sum is finished: <c>block</c> (the default, CUDA's
    /// form): the last block of the launch, counted by a completion counter, in the same launch (GcnKernels.Finish.cs);
    /// <c>launch</c>: a second kernel, as before plan 35's G6b. Both add in the same order (rule 86); which is faster may
    /// depend on the device and the shape (a launch's cost against one block doing the tail), so both stay.
    /// </summary>
    internal const string FinishVariable = "IDRAK_HIP_FINISH";

    /// <summary>Whether <c>IDRAK_HIP_FINISH</c> (<paramref name="setting"/>) asks for the last block's finish: anything but <c>launch</c>.</summary>
    internal static bool FinishByLastBlockSetting(string? setting) => !string.Equals(setting?.Trim(), "launch", StringComparison.OrdinalIgnoreCase);

    /// <summary>
    /// Whether split sums are finished by their launch's last block (one launch) or by a second launch: <c>IDRAK_HIP_FINISH</c>, settable by tests and benchmarks to time both forms.
    /// </summary>
    internal static bool FinishByLastBlock { get; set; } = FinishByLastBlockSetting(Environment.GetEnvironmentVariable(FinishVariable));

    // The completion counters, zeroed once and kept (rule 77: never a buffer per call); every launch that uses them
    // leaves them zero (its last block resets each), so the next launch, in stream order, finds them so.
    private Storage? _counters;

    // At least `count` zeroed completion counters (a larger buffer replaces a smaller one, which the pool takes back in
    // stream order).
    private Storage Counters(int count)
    {
        var current = _counters;
        if (current is null || current.Length < count)
        {
            var larger = Allocate(Math.Max(count, 4096), zeroed: true);
            current?.Release();
            _counters = current = larger;
        }

        return current;
    }

    /// <summary>The cost of a decoding kernel (the table at the top of GcnKernels.Decoding.cs; k: a finish's splits); (0, 0) for another.</summary>
    private static (double Flops, double Bytes) DecodingCost(string kernel, int k) => kernel switch
    {
        "attention_rows_f32" => (4, 8),                                                          // per row × position × dimension
        "segment_spans_f32" => (0, 12),                                                          // per row
        "argmax_f32" => (1, 4),
        "class_match_f32" => (2, 8),
        "add_rms_norm_affine_f32" => (6, 24),
        "rms_norm_rope_f32" => (7, 20),
        "rope_f32" => (7, 24),                                                                   // per pair
        "one_hot_f32" => (0, 8),
        "history_push_f32" => (0, 8),                                                            // per row
        "penalize_rows_f32" => (0, 8),
        "rms_norm_rope2_f32" or "norm_rope_heads_f32" => (7, 20),                               // per element
        "topk_candidates_f32" => (3.0 * k + 3, 4.25),                                            // per score; k: the top-k
        "topk_greedy_f32" => (3.0 * k + 7, 4.25),
        "sample_rows_f32" or "sample_candidates_f32" or "sample_candidates_greedy_f32" => (150, 36), // per score or candidate
        "pow_f32" => (12, 8),
        "clamp_f32" => (4, 8),
        "leaky_relu_f32" => (1, 8),
        "where_f32" => (1, 16),
        "inv_sqrt_f32" => (25, 8),
        "bias_gelu_f32" => (47, 12),
        "add_scalar_f32" => (1, 8),
        "gated_silu_f32" => (30, 12),
        "gated_gelu_f32" => (47, 12),
        "gated_relu_f32" => (2, 12),
        "gated_tanh_f32" => (67, 12),
        _ => (0, 0),
    };

    // ---- IRowAttentionBackend, ISegmentedAttentionBackend ----------------------------------------------------------

    // The head sizes and variants the CPU takes: the kernels take those up to GcnKernels.AttentionMaxDim, the host the rest.
    bool IRowAttentionBackend.SupportsRows(int dim, AttentionVariant variant) => Device.Cpu.Backend.SupportsAttentionRows(dim, variant);

    bool IRowAttentionBackend.AttentionRows(Storage q, Storage keys, Storage values, Storage position, Storage y, Storage starts, int heads, int headsPerRow,
        int rowsPerHead, int steps, int capacity, int dim, float scale, AttentionVariant variant)
    {
        using (EnterOperation())
        {
            if (headsPerRow > 0 && CacheAttention("attention_rows_f32", q, keys, values, position, y, null, heads, rowsPerHead, steps, capacity, dim, scale,
                    variant, starts, headsPerRow))
            {
                return true;
            }
        }

        return false;
    }

    bool ISegmentedAttentionBackend.SupportsSegmented(int dim, AttentionVariant variant) => Device.Cpu.Backend.SupportsSegmentedAttention(dim, variant);

    bool ISegmentedAttentionBackend.AttentionSegmented(Storage q, Storage keys, Storage values, Storage y, Storage? logSumExp, Storage starts, Storage ends, int heads,
        int headsPerRow, int rowsPerHead, int steps, int dim, float scale, AttentionVariant variant)
    {
        using (EnterOperation())
        {
            if (Segmented(q, keys, values, y, logSumExp, starts, heads, headsPerRow, rowsPerHead, steps, dim, scale, variant))
            {
                return true;
            }
        }

        return false;
    }

    // The gradient's kernels are G5b's (the attention backward passes): none on the device yet.
    bool ISegmentedAttentionBackend.AttentionSegmentedBackward(Storage q, Storage keys, Storage values, Storage output, Storage logSumExp, Storage dOutput, Storage dq,
        Storage dkeys, Storage dvalues, Storage starts, Storage ends, int heads, int headsPerRow, int rowsPerHead, int steps, int dim, float scale,
        AttentionVariant variant) => false;

    // Packed sequences on the device: each row's range of keys (segment_spans_f32), then attention over those ranges with
    // the keys and values [heads, steps, dim] (attention_spans_f32, the soft-cap its own); false when the kernels cannot
    // take it.
    private bool Segmented(Storage q, Storage keys, Storage values, Storage y, Storage? logSumExp, Storage starts, int heads, int headsPerRow, int rowsPerHead, int steps,
        int dim, float scale, AttentionVariant variant)
    {
        long count = heads > 0 && headsPerRow > 0 ? (long)heads / headsPerRow * rowsPerHead : 0;
        if (count <= 0 || heads % headsPerRow != 0 || steps <= 0 || !ShapeFits(count, (long)heads * rowsPerHead * dim, (long)heads * steps * dim)
            || ShapeKernel("segment_spans_f32") is not { } spans || AttentionKernel("attention_spans_f32", dim) is null)
        {
            return false;
        }

        var begins = Allocate((int)count, zeroed: false);
        var ends = Allocate((int)count, zeroed: false);
        try
        {
            LaunchShape(spans, (int)count, 0, P(starts), P(begins), P(ends), U(rowsPerHead), U(steps), U(Math.Max(0, variant.Window)), U((int)count));
            return SpanAttention(P(q), P(keys), P(values), begins, ends, y, logSumExp, 1, heads, heads, headsPerRow, rowsPerHead, steps, dim, scale, variant.Softcap,
                (0, rowsPerHead * dim, dim), (0, steps * dim, dim), (0, rowsPerHead * dim, dim));
        }
        finally
        {
            begins.Release();                                                    // reused in stream order
            ends.Release();
        }
    }

    // ---- The rows of decoding ---------------------------------------------------------------------------------------

    public override void ArgMaxKernel(Storage x, Storage y, int rows, int cols)
    {
        if (RowKernel("argmax_f32", cols) is not { } f)
        {
            base.ArgMaxKernel(x, y, rows, cols);
            return;
        }

        LaunchRows(f, rows, cols, Cost(f.Name).Flops, P(x), P(y), U(cols), U(rows));
    }

    public override void ClassMatchKernel(Storage predictions, Storage targets, Storage y, int rows, int cols, float threshold)
    {
        if (RowKernel("class_match_f32", cols) is not { } f)
        {
            base.ClassMatchKernel(predictions, targets, y, rows, cols, threshold);
            return;
        }

        LaunchRows(f, rows, cols, Cost(f.Name).Flops, P(predictions), P(targets), P(y), U(cols), F(threshold), U(rows));
    }

    public override void AddRmsNormAffineKernel(Storage a, Storage b, Storage sum, Storage gain, Storage y, int rows, int cols, float eps, float offset)
    {
        if (RowKernel("add_rms_norm_affine_f32", cols) is not { } f)
        {
            base.AddRmsNormAffineKernel(a, b, sum, gain, y, rows, cols, eps, offset);
            return;
        }

        LaunchRows(f, rows, cols, Cost(f.Name).Flops, P(a), P(b), P(sum), P(gain), P(y), U(cols), F(eps), F(offset), U(rows));
    }

    public override void RmsNormRopeKernel(Storage x, Storage gain, Storage cos, Storage sin, Storage positions, Storage y, int rows, int cols,
        float eps, float offset, int heads, int steps, int half, bool interleaved)
    {
        if (heads <= 0 || steps <= 0 || half < 0 || 2L * half > cols || RowKernel("rms_norm_rope_f32", cols) is not { } f)
        {
            base.RmsNormRopeKernel(x, gain, cos, sin, positions, y, rows, cols, eps, offset, heads, steps, half, interleaved);
            return;
        }

        LaunchRows(f, rows, cols, Cost(f.Name).Flops, P(x), P(gain), P(cos), P(sin), P(positions), P(y), U(cols), F(eps), F(offset), U(heads), U(steps), U(half),
            U(interleaved ? 1 : 0), U(rows));
    }

    public override void RmsNormRopePairKernel(Storage x, Storage gain, Storage y, int rows1, float eps, float offset, int heads,
        Storage x2, Storage gain2, Storage y2, int rows2, float eps2, float offset2, int heads2,
        Storage cos, Storage sin, Storage positions, int cols, int steps, int half, bool interleaved)
    {
        if (rows1 < 0 || rows2 < 0 || heads <= 0 || heads2 <= 0 || steps <= 0 || half < 0 || 2L * half > cols || !ShapeFits((long)rows1 * cols, (long)rows2 * cols)
            || RowKernel("rms_norm_rope2_f32", cols) is not { } f)
        {
            base.RmsNormRopePairKernel(x, gain, y, rows1, eps, offset, heads, x2, gain2, y2, rows2, eps2, offset2, heads2, cos, sin, positions, cols, steps, half, interleaved);
            return;
        }

        LaunchRows(f, rows1 + rows2, cols, Cost(f.Name).Flops, P(x), P(gain), P(cos), P(sin), P(positions), P(y), P(x2), P(gain2), P(y2), U(cols), F(eps), F(offset),
            U(heads), U(steps), U(half), U(interleaved ? 1 : 0), U(rows1), F(eps2), F(offset2), U(heads2), U(rows1 + rows2));
    }

    // The attention layer's queries, keys and values into the layouts attention reads in one launch (norm_rope_heads_f32,
    // CUDA's flags: 1 the norms, 4 a cache position); false where the kernel cannot take it (the same
    // refusals as Vulkan's, a tensor of 2^30 elements or more), so callers run the steps one by one.
    public override bool NormRopeHeadsKernel(Storage q, Storage k, Storage v, int batch, int steps, int heads, int kvHeads, int cols,
        Storage? gainQ, float epsQ, float offsetQ, Storage? gainK, float epsK, float offsetK, Storage? cos, Storage? sin, Storage? positions,
        int half, bool interleaved, Storage yq, Storage yk, Storage yv, Storage? position, int capacity, int stride)
    {
        int rot = cos is null ? 0 : half;
        long rows1 = (long)batch * steps * heads, rows2 = (long)batch * steps * kvHeads;
        if (cols <= 0 || batch <= 0 || steps <= 0 || heads <= 0 || kvHeads <= 0 || capacity <= 0 || stride <= 0 || (gainQ is null) != (gainK is null)
            || (cos is null) != (sin is null) || cos is not null && positions is null || rot < 0 || 2L * rot > cols
            || rows1 + 2 * rows2 > int.MaxValue || !ShapeFits(rows1 * cols, rows2 * cols, (long)batch * kvHeads * capacity * stride)
            || RowKernel("norm_rope_heads_f32", cols) is not { } f)
        {
            return false;
        }

        static ulong Optional(Storage? s) => s is null ? 0UL : P(s);
        int flags = (gainQ is not null ? 1 : 0) | (position is not null ? 4 : 0);
        LaunchRows(f, (int)(rows1 + 2 * rows2), cols, Cost(f.Name).Flops, P(q), Optional(gainQ), P(k), Optional(gainK), P(v), Optional(cos), Optional(sin),
            Optional(positions), Optional(position), P(yq), P(yk), P(yv), U(cols), F(epsQ), F(offsetQ), U(heads), F(epsK), F(offsetK), U(kvHeads), U(steps), U(rot),
            U(interleaved ? 1 : 0), U((int)rows1), U((int)rows2), U(flags), U(capacity), U(stride), U((int)(rows1 + 2 * rows2)));
        return true;
    }

    public override void RopeKernel(Storage x, Storage y, Storage cos, Storage sin, Storage positions, int rows, int heads, int steps, int dim, int half, bool interleaved,
        float sign)
    {
        long n = (long)rows * half;
        if (heads <= 0 || steps <= 0 || half <= 0 || 2L * half > dim || !ShapeFits(n, (long)rows * dim) || ShapeKernel("rope_f32") is not { } f)
        {
            base.RopeKernel(x, y, cos, sin, positions, rows, heads, steps, dim, half, interleaved, sign);
            return;
        }

        LaunchShape(f, (int)n, Cost(f.Name).Flops * n, P(x), P(y), P(cos), P(sin), P(positions), U(heads), U(steps), U(dim), U(half), U(interleaved ? 1 : 0), F(sign),
            U((int)n));
    }

    public override void OneHotKernel(Storage indices, Storage y, int count, int classes)
    {
        long n = (long)count * classes;
        if (classes <= 0 || !ShapeFits(n) || ShapeKernel("one_hot_f32") is not { } f)
        {
            base.OneHotKernel(indices, y, count, classes);
            return;
        }

        LaunchShape(f, (int)n, 0, P(indices), P(y), U(classes), U(classes - 1), U((int)n));
    }

    public override void HistoryPushKernel(Storage ids, Storage history, Storage length, int rows, int capacity)
    {
        if (capacity <= 0 || !ShapeFits((long)rows * capacity) || ShapeKernel("history_push_f32") is not { } f)
        {
            base.HistoryPushKernel(ids, history, length, rows, capacity);
            return;
        }

        LaunchShape(f, rows, 0, P(ids), P(history), P(length), U(capacity), U(rows), U(rows));
    }

    public override void PenalizeRowsKernel(Storage logits, Storage work, Storage history, Storage length, int rows, int vocabulary,
        int rowStride, int rowOffset, int capacity, int lastN, float repeat, float presence, float frequency)
    {
        if (capacity <= 0 || lastN < 0 || !ShapeFits((long)rows * vocabulary, (long)rows * rowStride + rowOffset, (long)rows * capacity)
            || RowKernel("penalize_rows_f32", vocabulary) is not { } f)
        {
            base.PenalizeRowsKernel(logits, work, history, length, rows, vocabulary, rowStride, rowOffset, capacity, lastN, repeat, presence, frequency);
            return;
        }

        LaunchRows(f, rows, vocabulary, 0, P(logits), P(work), P(history), P(length), U(vocabulary), U(rowStride), U(rowOffset), U(capacity),
            U(Math.Min(lastN, capacity)), F(repeat), F(presence), F(frequency), U(rows), U(rows));
    }

    // ---- The sampler -------------------------------------------------------------------------------------------------

    /// <summary>Tests and benchmarks: false samples greedy rows through the two candidate kernels alone, as CUDA's GreedyRows.</summary>
    internal static bool GreedyRows { get; set; } = true;

    // CUDA's paths: the whole row on sample_rows_f32; top-k ≤ 64 over a vocabulary past four candidate slices on the
    // candidates (topk_candidates_f32, then sample_candidates_f32), and greedy rows (top-k 1, min-p at most 0.5, which can
    // never raise the cut-off past the best score) on topk_greedy_f32 then sample_candidates_greedy_f32, which finishes a
    // row of one best score from its blocks' parts. The host's sampler past the kernels' limits.
    public override void SampleRowsKernel(Storage logits, Storage ids, Storage stats, Storage step, int rows, int vocabulary,
        int rowStride, int rowOffset, float temperature, int topK, float topP, float minP, uint seed)
    {
        if (rows <= 0)
        {
            return;
        }

        var settings = GcnKernels.SamplerSettings.Of(vocabulary, rowStride, rowOffset, temperature, topK, topP, minP, seed, rows);
        bool candidates = topK > 0 && topK <= GcnKernels.CandidateSlots && vocabulary > 4 * GcnKernels.CandidateSlice;
        bool fits = vocabulary > 0 && rowStride >= 0 && rowOffset >= 0 && ShapeFits((long)(rows - 1) * rowStride + rowOffset + vocabulary);
        if (!fits || !(candidates ? SampleCandidates(logits, ids, stats, step, rows, vocabulary, topK, minP, settings) : SampleWhole(logits, ids, stats, step, rows, vocabulary, settings)))
        {
            base.SampleRowsKernel(logits, ids, stats, step, rows, vocabulary, rowStride, rowOffset, temperature, topK, topP, minP, seed);
        }
    }

    // The settings' values after the pointers, as the sampler kernels take them (ln(min-p) where CUDA has min-p).
    private static void SamplerValues(Span<ulong> values, in GcnKernels.SamplerSettings s) =>
        (values[0], values[1], values[2], values[3], values[4], values[5], values[6], values[7], values[8]) =
        (s.Vocabulary, s.RowStride, s.RowOffset, F(s.InverseTemperature), s.TopK, F(s.TopP), F(s.LogMinP), s.Seed, s.Rows);

    private bool SampleWhole(Storage logits, Storage ids, Storage stats, Storage step, int rows, int vocabulary, in GcnKernels.SamplerSettings settings)
    {
        if (RowKernel(GcnKernels.SamplerKernel(GcnKernels.SamplerForm.Rows), vocabulary) is not { } f)
        {
            return false;
        }

        Span<ulong> values = stackalloc ulong[14];
        (values[0], values[1], values[2], values[3]) = (P(logits), P(ids), P(stats), P(step));
        SamplerValues(values[4..], settings);
        values[13] = U(rows);
        LaunchRows(f, rows, vocabulary, Cost(f.Name).Flops, values);
        return true;
    }

    // Greedy rows sample themselves in topk_greedy_f32's last block of each row (IDRAK_HIP_FINISH=block, the default: one
    // launch), else in sample_candidates_greedy_f32 after it; top-k rows through topk_candidates_f32, then
    // sample_candidates_f32.
    private bool SampleCandidates(Storage logits, Storage ids, Storage stats, Storage step, int rows, int vocabulary, int topK, float minP,
        in GcnKernels.SamplerSettings settings)
    {
        int blocks = (vocabulary + GcnKernels.CandidateSlice - 1) / GcnKernels.CandidateSlice, slots = blocks * GcnKernels.CandidateSlots;
        bool greedy = GreedyRows && topK == 1 && minP <= 0.5f, lastBlock = greedy && FinishByLastBlock;
        var form = greedy ? GcnKernels.SamplerForm.Greedy : GcnKernels.SamplerForm.Candidates;
        if (!ShapeFits((long)rows * slots, (long)rows * blocks * 3) || Gcn(greedy ? "topk_greedy_f32" : "topk_candidates_f32") is not { } stage
            || Gcn(GcnKernels.SamplerKernel(form)) is not { } sampler)
        {
            return false;
        }

        var candidateScores = Allocate(rows * slots, zeroed: false);
        var candidateIds = Allocate(rows * slots, zeroed: false);
        var flags = Allocate(rows * blocks, zeroed: false);                       // every block writes its own
        var parts = greedy ? Allocate(rows * blocks * 3, zeroed: false) : null;
        try
        {
            Span<ulong> first = stackalloc ulong[21];
            int at = 0;
            first[at++] = P(logits);
            first[at++] = P(candidateScores);
            first[at++] = P(candidateIds);
            first[at++] = P(flags);
            if (parts is not null)
            {
                first[at++] = P(parts);
                first[at++] = P(ids);
                first[at++] = P(stats);
                first[at++] = P(step);
                first[at++] = lastBlock ? P(Counters(rows)) : 0UL;
            }

            (first[at], first[at + 1], first[at + 2], first[at + 3], first[at + 4], first[at + 5]) =
                (settings.Vocabulary, settings.RowStride, settings.RowOffset, F(settings.InverseTemperature), settings.TopK, U(blocks));
            at += 6;
            if (parts is not null)
            {
                (first[at], first[at + 1], first[at + 2], first[at + 3], first[at + 4]) = (F(settings.TopP), F(settings.LogMinP), settings.Seed, settings.Rows, U(slots));
                at += 5;
            }

            first[at++] = U(rows * blocks);
            if (lastBlock)
            {
                if (Profiling)
                {
                    LabelNextLaunch(string.Create(CultureInfo.InvariantCulture, $"{stage.Name} {rows * blocks}x{GcnKernels.CandidateSlice} (last block samples)"),
                        Cost(stage.Name, topK).Flops * rows * (double)blocks * GcnKernels.CandidateSlice + Cost(sampler.Name).Flops * rows * (double)slots);
                }

                LaunchKernel(stage, (uint)Math.Min(rows * blocks, DeviceLimits.MaxGridDimX), 1, first[..at]);
                return true;
            }

            LaunchRows(stage, rows * blocks, GcnKernels.CandidateSlice, Cost(stage.Name, topK).Flops, first[..at]);

            Span<ulong> second = stackalloc ulong[19];
            at = 0;
            (second[0], second[1], second[2], second[3], second[4], second[5], second[6]) =
                (P(logits), P(ids), P(stats), P(step), P(candidateScores), P(candidateIds), P(flags));
            at = 7;
            if (parts is not null)
            {
                second[at++] = P(parts);
            }

            SamplerValues(second[at..], settings);
            at += 9;
            second[at++] = U(slots);
            second[at++] = U(rows);
            LaunchRows(sampler, rows, slots, Cost(sampler.Name).Flops, second[..at]);
        }
        finally
        {
            candidateScores.Release();                                       // reused in stream order
            candidateIds.Release();
            flags.Release();
            parts?.Release();
        }

        return true;
    }
}
