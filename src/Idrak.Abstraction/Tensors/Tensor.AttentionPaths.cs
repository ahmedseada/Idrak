// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Idrak.Abstraction.Devices;

namespace Idrak.Abstraction;

/// <summary>How <see cref="Tensor.AttentionFastest"/> computes attention where both paths can.</summary>
public enum AttentionPath
{
    /// <summary>The faster on the device as measured there (once per shape and precision), AttentionSpans until measured.</summary>
    Measured,

    /// <summary>Always <see cref="Tensor.AttentionSpans"/>: tiled over the keys, the scores never stored.</summary>
    Spans,

    /// <summary>The full scores [heads, rows, keyRows], a softmax and the values (as long as the device has the memory).</summary>
    Composed,
}

/// <summary>
/// The choice between <see cref="Tensor.AttentionSpans"/> and the composed scores for <see cref="Tensor.AttentionFastest"/>:
/// measured on the device by default; <see cref="Forced"/> (environment: <c>IDRAK_ATTENTION_PATH</c>=spans or composed)
/// sets one for every call.
/// </summary>
public static class AttentionPaths
{
    /// <summary>
    /// The path every <see cref="Tensor.AttentionFastest"/> takes, or <see cref="AttentionPath.Measured"/> (the default) for
    /// the device's own measurement. From <c>IDRAK_ATTENTION_PATH</c> (spans or composed) at start. A forced composed path
    /// still needs a device memory guard's room: without it AttentionSpans runs.
    /// </summary>
    public static AttentionPath Forced { get; set; } = FromEnvironment(Environment.GetEnvironmentVariable("IDRAK_ATTENTION_PATH"));

    /// <summary>
    /// The share of the device's available memory the composed path's scores may take (the rest is the margin for
    /// everything else the pass allocates): half.
    /// </summary>
    internal const double ScoreShare = 0.5;

    internal static AttentionPath FromEnvironment(string? value) => value?.Trim().ToLowerInvariant() switch
    {
        "spans" => AttentionPath.Spans,
        "composed" => AttentionPath.Composed,
        _ => AttentionPath.Measured,
    };

    /// <summary>
    /// Bytes the composed path holds at once for <paramref name="heads"/> · <paramref name="rows"/> · <paramref name="keyRows"/>
    /// scores: the scores and the weights (inference), and with the gradient recorded also the scaled and masked scores
    /// kept for the backward pass.
    /// </summary>
    internal static long ComposedBytes(int heads, int rows, int keyRows, bool recording) =>
        (recording ? 4L : 2L) * heads * rows * keyRows * sizeof(float);

    /// <summary>Whether the composed path's scores fit the memory the device reports available, with the margin (none reported: no).</summary>
    internal static bool ComposedFits(long? available, long bytes) => available is { } free && bytes <= free * ScoreShare;
}

public sealed partial class Tensor
{
    /// <summary>
    /// <see cref="AttentionSpans"/>, or the same attention through the full scores (q · keysᵀ, softmax with the mask,
    /// · values) where that is faster on the device: the device measures both once per shape and precision and keeps the
    /// choice with its other measured choices (per device and driver; <see cref="Backend.PrefersComposedAttention"/>; while
    /// the gradient is recorded, each path timed with its gradient).
    /// The composed path is taken only when the device reports the memory for its scores with margin
    /// (<see cref="Backend.AvailableMemory"/>), as many key/value heads as query heads, one table of ranges and no
    /// soft-cap; until measured, and on devices that measure nothing (the CPU), AttentionSpans runs.
    /// <see cref="AttentionPaths.Forced"/> (<c>IDRAK_ATTENTION_PATH</c>) forces either. The two give the same result
    /// within rounding.
    /// </summary>
    /// <param name="q">[heads, rows, dim].</param>
    /// <param name="keys">[kvHeads, keyRows, dim].</param>
    /// <param name="values">[kvHeads, keyRows, dim].</param>
    /// <param name="starts">Each row's first key, as for <see cref="AttentionSpans"/>.</param>
    /// <param name="ends">Each row's end (exclusive).</param>
    /// <param name="scale">The scores' scale.</param>
    /// <param name="everyKey">True when every row sees every key (a bidirectional pass): the composed path needs no mask.</param>
    /// <param name="mask">
    /// The same ranges as an additive mask [rows, keyRows] (0 where a row sees the key, a large negative number where it
    /// does not), for the composed path; with neither this nor <paramref name="everyKey"/>, AttentionSpans always runs.
    /// </param>
    /// <exception cref="ArgumentException">The shapes do not fit together.</exception>
    public static Tensor AttentionFastest(Tensor q, Tensor keys, Tensor values, Tensor starts, Tensor ends, float scale, bool everyKey, Tensor? mask = null)
    {
        ArgumentNullException.ThrowIfNull(q);
        ArgumentNullException.ThrowIfNull(keys);
        ArgumentNullException.ThrowIfNull(values);
        ArgumentNullException.ThrowIfNull(starts);
        ArgumentNullException.ThrowIfNull(ends);
        return ComposedChosen(q, keys, values, starts, ends, scale, everyKey, mask)
            ? ComposedAttention(q, keys, values, everyKey ? null : mask, scale)
            : AttentionSpans(q, keys, values, starts, ends, scale);
    }

    // Whether AttentionFastest takes the composed path: possible (shapes, a mask or every key, the memory), then forced or
    // measured.
    private static bool ComposedChosen(Tensor q, Tensor keys, Tensor values, Tensor starts, Tensor ends, float scale, bool everyKey, Tensor? mask)
    {
        var forced = AttentionPaths.Forced;
        if (forced == AttentionPath.Spans || q.Rank != 3 || keys.Rank != 3 || !keys._shape.AsSpan().SequenceEqual(values._shape)
            || keys._shape[2] != q._shape[2] || keys.Device != q.Device || values.Device != q.Device)
        {
            return false;
        }

        int heads = q._shape[0], rows = q._shape[1], dim = q._shape[2], keyRows = keys._shape[1];
        if (heads == 0 || rows == 0 || keyRows == 0 || dim == 0 || keys._shape[0] != heads || starts.Size != rows || ends.Size != rows)
        {
            return false;                                                  // grouped heads or several tables: AttentionSpans
        }

        if (!everyKey && (mask is null || mask.Size != (long)rows * keyRows || mask.Device != q.Device))
        {
            return false;
        }

        bool recording = Autograd.IsEnabled && (q.RequiresGrad || keys.RequiresGrad || values.RequiresGrad);
        var backend = q.Backend;
        var usage = backend.GetMemoryUsage();
        long? available = backend.AvailableMemory();
        if (available is { } free && usage.Limit is { } limit)
        {
            available = Math.Min(free, limit - usage.InUse);
        }

        if (!AttentionPaths.ComposedFits(available, AttentionPaths.ComposedBytes(heads, rows, keyRows, recording)))
        {
            return false;                                                  // the memory guard: AttentionSpans never holds the scores
        }

        return forced == AttentionPath.Composed
               || backend.PrefersComposedAttention(q.Storage, keys.Storage, values.Storage, starts.Storage, ends.Storage, everyKey ? null : mask?.Storage,
                   heads, rows, keyRows, dim, scale, recording);
    }

    // softmax(scale · q · keysᵀ + mask) · values over the full scores: one fused softmax for inference, the differentiable
    // operations when the gradient is recorded.
    private static Tensor ComposedAttention(Tensor q, Tensor keys, Tensor values, Tensor? mask, float scale)
    {
        if (!Autograd.IsEnabled || !(q.RequiresGrad || keys.RequiresGrad || values.RequiresGrad))
        {
            using var scores = q.MatMul(keys, transposeB: true);
            using var weights = scores.ScaleMaskSoftmax(scale, mask);
            return weights.MatMul(values);
        }

        var scaled = q.MatMul(keys, transposeB: true) * scale;
        if (mask is not null)
        {
            scaled = scaled + mask.Reshape(q._shape[1], keys._shape[1]);
        }

        return scaled.Softmax().MatMul(values);
    }
}
