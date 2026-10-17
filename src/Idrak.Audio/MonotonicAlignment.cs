// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

namespace Idrak.Audio;

/// <summary>
/// The monotonic alignment search (VITS's <c>monotonic_align</c>, Glow-TTS's): the path through a sequence's
/// [frames, tokens] scores (each frame's log-likelihood under each token's prior) that visits every token in order, at
/// least a frame each, with the largest sum; on the host, in float as the Cython code adds, so its paths are VITS's exactly.
/// </summary>
public static class MonotonicAlignment
{
    /// <summary>The token of each frame along the best path of <paramref name="scores"/> [frames, tokens] (row-major); frames are at least tokens.</summary>
    /// <exception cref="ArgumentException">Fewer frames than tokens, or the scores are not frames × tokens.</exception>
    public static int[] Search(ReadOnlySpan<float> scores, int frames, int tokens)
    {
        if (tokens < 1 || frames < tokens || scores.Length != frames * tokens)
        {
            throw new ArgumentException($"The alignment search takes [frames, tokens] scores with at least as many frames as tokens; got {scores.Length} values for {frames} x {tokens}.");
        }

        var value = scores.ToArray();
        const float Never = -1e9f;
        for (int y = 0; y < frames; y++)
        {
            int last = Math.Min(tokens, y + 1);
            for (int x = Math.Max(0, tokens + y - frames); x < last; x++)
            {
                float current = x == y ? Never : value[(y - 1) * tokens + x];
                float previous = x == 0 ? y == 0 ? 0f : Never : value[(y - 1) * tokens + x - 1];
                value[y * tokens + x] += Math.Max(previous, current);
            }
        }

        var path = new int[frames];
        int index = tokens - 1;
        for (int y = frames - 1; y >= 0; y--)
        {
            path[y] = index;
            if (index != 0 && (index == y || value[(y - 1) * tokens + index] < value[(y - 1) * tokens + index - 1]))
            {
                index--;
            }
        }

        return path;
    }

    /// <summary>Each token's frames along <paramref name="path"/> (a frame's token).</summary>
    public static int[] Durations(ReadOnlySpan<int> path, int tokens)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(tokens);
        var durations = new int[tokens];
        foreach (int token in path)
        {
            durations[token]++;
        }

        return durations;
    }
}
