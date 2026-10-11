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
    /// (<see cref="Backend.AvailableMemory"/>); until measured, and on devices that measure nothing (the CPU), AttentionSpans
    /// runs. Grouped heads (fewer key/value heads than query heads) and several tables of ranges compose as well (a table
    /// at a time, each table's query heads a whole number of groups); where the device has its own kernel for the head
    /// size they take AttentionSpans unless forced, and where it has none (<see cref="BackendCapabilities.SpanAttentionHeadDim"/>:
    /// AttentionSpans would run on the host) they compose whenever the scores fit, without measuring.
    /// <see cref="AttentionPaths.Forced"/> (<c>IDRAK_ATTENTION_PATH</c>) forces either. The two give the same result
    /// within rounding for rows that see at least one key (a row whose range is empty gets zeros from AttentionSpans and
    /// the mean of the values from the composed path).
    /// </summary>
    /// <param name="q">[heads, rows, dim].</param>
    /// <param name="keys">[kvHeads, keyRows, dim].</param>
    /// <param name="values">[kvHeads, keyRows, dim].</param>
    /// <param name="starts">Each row's first key, as for <see cref="AttentionSpans"/>.</param>
    /// <param name="ends">Each row's end (exclusive).</param>
    /// <param name="scale">The scores' scale.</param>
    /// <param name="everyKey">True when every row sees every key (a bidirectional pass): the composed path needs no mask.</param>
    /// <param name="mask">
    /// The same ranges as an additive mask [tables · rows, keyRows] (0 where a row sees the key, a large negative number
    /// where it does not; one table's rows after another's, as <paramref name="starts"/>), for the composed path; with
    /// neither this nor <paramref name="everyKey"/>, AttentionSpans always runs.
    /// </param>
    /// <exception cref="ArgumentException">The shapes do not fit together.</exception>
    public static Tensor AttentionFastest(Tensor q, Tensor keys, Tensor values, Tensor starts, Tensor ends, float scale, bool everyKey, Tensor? mask = null)
    {
        ArgumentNullException.ThrowIfNull(q);
        ArgumentNullException.ThrowIfNull(keys);
        ArgumentNullException.ThrowIfNull(values);
        ArgumentNullException.ThrowIfNull(starts);
        ArgumentNullException.ThrowIfNull(ends);
        return ComposedChosen(q, keys, values, starts, ends, scale, everyKey, mask, out int tables)
            ? ComposedAttention(q, keys, values, everyKey ? null : mask, scale, tables)
            : AttentionSpans(q, keys, values, starts, ends, scale);
    }

    // Whether AttentionFastest takes the composed path: possible (shapes, a mask or every key, the memory), then forced, the
    // device without its own kernel for the head size, or measured.
    private static bool ComposedChosen(Tensor q, Tensor keys, Tensor values, Tensor starts, Tensor ends, float scale, bool everyKey, Tensor? mask,
        out int tables)
    {
        tables = 1;
        var forced = AttentionPaths.Forced;
        if (forced == AttentionPath.Spans || q.Rank != 3 || keys.Rank != 3 || !keys._shape.AsSpan().SequenceEqual(values._shape)
            || keys._shape[2] != q._shape[2] || keys.Device != q.Device || values.Device != q.Device)
        {
            return false;
        }

        int heads = q._shape[0], rows = q._shape[1], dim = q._shape[2], kvHeads = keys._shape[0], keyRows = keys._shape[1];
        if (heads == 0 || rows == 0 || keyRows == 0 || dim == 0 || kvHeads == 0 || heads % kvHeads != 0 || starts.Size != ends.Size
            || starts.Size % rows != 0)
        {
            return false;
        }

        tables = starts.Size / rows;
        int group = heads / kvHeads;
        if (tables == 0 || heads % tables != 0 || heads / tables % group != 0)
        {
            return false;                                                  // a table's query heads not whole groups: AttentionSpans
        }

        var backend = q.Backend;
        bool ownKernel = dim <= backend.Capabilities.SpanAttentionHeadDim;
        bool plain = group == 1 && tables == 1;                            // what the device measures
        if (!plain && ownKernel && forced != AttentionPath.Composed)
        {
            return false;                                                  // grouped heads or several tables: the device's own kernel
        }

        if (!everyKey && (mask is null || mask.Size != (long)tables * rows * keyRows || mask.Device != q.Device))
        {
            return false;
        }

        bool recording = Autograd.IsEnabled && (q.RequiresGrad || keys.RequiresGrad || values.RequiresGrad);
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
               || !ownKernel                                               // the alternative runs on the host
               || plain && backend.PrefersComposedAttention(q.Storage, keys.Storage, values.Storage, starts.Storage, ends.Storage,
                   everyKey ? null : mask?.Storage, heads, rows, keyRows, dim, scale, recording);
    }

    // softmax(scale · q · keysᵀ + mask) · values over the full scores, a table of ranges at a time (its query heads and their
    // key/value heads, its rows of the mask): one fused softmax for inference, the differentiable operations when the
    // gradient is recorded.
    private static Tensor ComposedAttention(Tensor q, Tensor keys, Tensor values, Tensor? mask, float scale, int tables)
    {
        int heads = q._shape[0], rows = q._shape[1], keyRows = keys._shape[1], kvHeads = keys._shape[0];
        if (tables == 1)
        {
            return ComposedTable(q, keys, values, mask?.Reshape(rows, keyRows), scale);
        }

        bool recording = Autograd.IsEnabled && (q.RequiresGrad || keys.RequiresGrad || values.RequiresGrad);
        int perTable = heads / tables, kvPerTable = kvHeads / tables;
        var parts = new Tensor[tables];
        for (int b = 0; b < tables; b++)
        {
            var qb = q.Narrow(0, b * perTable, perTable);
            var kb = keys.Narrow(0, b * kvPerTable, kvPerTable);
            var vb = values.Narrow(0, b * kvPerTable, kvPerTable);
            var mb = mask?.Reshape(tables * rows, keyRows).Narrow(0, b * rows, rows);
            parts[b] = ComposedTable(qb, kb, vb, mb, scale);
            if (!recording)
            {
                qb.Dispose();
                kb.Dispose();
                vb.Dispose();
                mb?.Dispose();
            }
        }

        var y = Concat(parts, 0);
        if (!recording)
        {
            foreach (var part in parts)
            {
                part.Dispose();
            }
        }

        return y;
    }

    // One table: q [heads, rows, dim] over keys and values [kvHeads, keyRows, dim] (each key/value head read by
    // heads / kvHeads query heads, as one product of [group · rows] rows), mask [rows, keyRows] or none.
    private static Tensor ComposedTable(Tensor q, Tensor keys, Tensor values, Tensor? mask, float scale)
    {
        int heads = q._shape[0], rows = q._shape[1], dim = q._shape[2], kvHeads = keys._shape[0], keyRows = keys._shape[1];
        int group = heads / kvHeads;
        if (!Autograd.IsEnabled || !(q.RequiresGrad || keys.RequiresGrad || values.RequiresGrad))
        {
            using var grouped = group == 1 ? null : q.Reshape(kvHeads, group * rows, dim);
            using var scores = (grouped ?? q).MatMul(keys, transposeB: true);
            using var weights = scores.ScaleMaskSoftmax(scale, mask);      // the mask's rows repeat over every head's rows
            if (group == 1)
            {
                return weights.MatMul(values);
            }

            using var attended = weights.MatMul(values);
            return attended.Reshape(heads, rows, dim);
        }

        var scaled = (group == 1 ? q : q.Reshape(kvHeads, group * rows, dim)).MatMul(keys, transposeB: true) * scale;
        if (mask is not null)
        {
            scaled = (scaled.Reshape(heads, rows, keyRows) + mask).Reshape(kvHeads, group * rows, keyRows);   // the mask over each head's rows
        }

        var y = scaled.Softmax().MatMul(values);
        return group == 1 ? y : y.Reshape(heads, rows, dim);
    }
}
