// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

namespace Idrak.Audio;

/// <summary>
/// Length regulation: a text's per-label states repeated for each label's frames, so a decoder reads one row per frame
/// (plan 21, S6). One gather of rows by indices made once on the host per batch (no per-frame work on the device); the
/// gather's gradient adds each frame's gradient back to its label's row, in frame order.
/// </summary>
public static class LengthRegulation
{
    /// <summary>
    /// The row each frame reads: frame f of text n (its durations in order) reads row n · <paramref name="length"/> + its
    /// label; the frames after a text's own (to the longest text's frames) read its first row and are padding. Also each
    /// text's frames.
    /// </summary>
    /// <exception cref="ArgumentException">A duration is negative, or a text has more durations than <paramref name="length"/>.</exception>
    public static (int[] Rows, int[] Frames) Rows(IReadOnlyList<int[]> durations, int length)
    {
        ArgumentNullException.ThrowIfNull(durations);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(length);
        int n = durations.Count;
        var frames = new int[n];
        int longest = 0;
        for (int b = 0; b < n; b++)
        {
            var own = durations[b] ?? throw new ArgumentException($"Text {b} has no durations.", nameof(durations));
            if (own.Length > length)
            {
                throw new ArgumentException($"Text {b} has {own.Length} durations for {length} positions.", nameof(durations));
            }

            long sum = 0;
            foreach (int d in own)
            {
                if (d < 0)
                {
                    throw new ArgumentException($"Text {b} has a negative duration ({d}).", nameof(durations));
                }

                sum += d;
            }

            frames[b] = checked((int)sum);
            longest = Math.Max(longest, frames[b]);
        }

        var rows = new int[n * longest];
        for (int b = 0; b < n; b++)
        {
            var target = rows.AsSpan(b * longest, longest);
            target.Fill(b * length);                                               // padding: the text's first row
            int at = 0;
            for (int label = 0; label < durations[b].Length; label++)
            {
                target.Slice(at, durations[b][label]).Fill(b * length + label);
                at += durations[b][label];
            }
        }

        return (rows, frames);
    }

    /// <summary>
    /// <paramref name="hidden"/> [batch, length, width] expanded to [batch, frames, width] (frames the longest text's sum of
    /// durations): frame f of text n is the row of the label it falls in; frames after a text's own repeat its first row.
    /// </summary>
    /// <exception cref="ArgumentException">The states are not [batch, length, width] for the durations' batch, or no text has a frame.</exception>
    public static (Tensor Frames, int[] Lengths) Expand(Tensor hidden, IReadOnlyList<int[]> durations)
    {
        ArgumentNullException.ThrowIfNull(hidden);
        ArgumentNullException.ThrowIfNull(durations);
        if (hidden.Rank != 3 || hidden.Shape[0] != durations.Count)
        {
            throw new ArgumentException($"Length regulation takes states [{durations.Count}, length, width], got {Tensor.FormatShape(hidden.Shape)}.", nameof(hidden));
        }

        int n = hidden.Shape[0], length = hidden.Shape[1], width = hidden.Shape[2];
        var (rows, frames) = Rows(durations, length);
        if (rows.Length == 0)
        {
            throw new ArgumentException("No text has a frame to make (every duration is 0).", nameof(durations));
        }

        var indices = new float[rows.Length];
        for (int i = 0; i < rows.Length; i++)
        {
            indices[i] = rows[i];
        }

        int longest = rows.Length / n;
        var gathered = hidden.Reshape(n * length, width).EmbeddingLookup(Tensor.From(indices, [indices.Length], hidden.Device));
        return (gathered.Reshape(n, longest, width), frames);
    }

    /// <summary>
    /// Whole frames from predicted log(frames + <paramref name="offset"/>) values: exp(value) − offset over
    /// <paramref name="rate"/> (2 speaks twice as fast), rounded, never below 0; when every label rounds to none, the
    /// label predicted longest gets one frame so there is something to speak.
    /// </summary>
    /// <exception cref="ArgumentOutOfRangeException">The rate is not positive, or the offset is negative.</exception>
    public static int[] Durations(ReadOnlySpan<float> logDurations, float offset, float rate = 1f)
    {
        if (!(rate > 0))
        {
            throw new ArgumentOutOfRangeException(nameof(rate), $"The speaking rate is positive, got {rate}.");
        }

        ArgumentOutOfRangeException.ThrowIfNegative(offset);
        var durations = new int[logDurations.Length];
        long total = 0;
        int longest = 0;
        for (int i = 0; i < durations.Length; i++)
        {
            double frames = (Math.Exp(logDurations[i]) - offset) / rate;
            durations[i] = frames > 0 ? (int)Math.Min(int.MaxValue / 2, Math.Round(frames, MidpointRounding.AwayFromZero)) : 0;
            total += durations[i];
            if (logDurations[i] > logDurations[longest])
            {
                longest = i;
            }
        }

        if (total == 0 && durations.Length > 0)
        {
            durations[longest] = 1;
        }

        return durations;
    }
}
