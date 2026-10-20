// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Buffers;
using System.Text;
using Idrak.Text.Abstractions;

namespace Idrak.Text;

/// <summary>
/// "semi-global" (<see cref="TextMatchers.SemiGlobal"/>): the span of a text nearest a draft by edit distance over
/// Unicode scalars, the span free to start and end anywhere in the text (Sellers, "The theory and computation of
/// evolutionary distances: pattern recognition", 1980): one row of the table per draft character, two rows kept, each
/// cell carrying where its span starts. Of equal distances the first end wins; with
/// <see cref="TextMatchOptions.WordBoundaries"/> the span's ends then move out to white space or the text's ends. Time
/// O(draft · text), memory O(text), one pooled buffer per call.
/// </summary>
public sealed class SemiGlobalMatcher : ITextMatcher
{
    /// <inheritdoc />
    public string Name => TextMatchers.SemiGlobal;

    /// <inheritdoc />
    public TextMatch? Best(string draft, string text, int from, TextMatchOptions options)
    {
        ArgumentNullException.ThrowIfNull(draft);
        ArgumentNullException.ThrowIfNull(text);
        ArgumentNullException.ThrowIfNull(options);
        ArgumentOutOfRangeException.ThrowIfNegative(from);
        if (from >= text.Length || draft.Length == 0)
        {
            return null;
        }

        // One rented buffer: the draft's scalars, the text's scalars from `from` with each one's char index, and two rows
        // of the table (only the last row is read, so the rows before it are not kept).
        int size = text.Length - from + 1;
        int[] buffer = ArrayPool<int>.Shared.Rent(draft.Length + 6 * size);
        try
        {
            var d = buffer.AsSpan(0, draft.Length);
            int n = 0;
            foreach (var rune in draft.EnumerateRunes())
            {
                d[n++] = rune.Value;
            }

            int m = 0;
            var t = buffer.AsSpan(draft.Length, size);
            var index = buffer.AsSpan(draft.Length + size, size);
            for (int i = from; i < text.Length;)
            {
                var rune = Rune.GetRuneAt(text, i);
                (t[m], index[m]) = (rune.Value, i);
                m++;
                i += rune.Utf16SequenceLength;
            }

            index[m] = text.Length;                                                   // a span starting past the last scalar

            // cost[j]: distance of d[..i] to the best span ending at t[j - 1]; start[j]: where that span starts. Row i - 1
            // in the previous pair, row i in the current one.
            var cost = buffer.AsSpan(draft.Length + 2 * size, m + 1);
            var start = buffer.AsSpan(draft.Length + 3 * size, m + 1);
            var nextCost = buffer.AsSpan(draft.Length + 4 * size, m + 1);
            var nextStart = buffer.AsSpan(draft.Length + 5 * size, m + 1);
            for (int j = 0; j <= m; j++)
            {
                (cost[j], start[j]) = (0, j);
            }

            for (int i = 1; i <= n; i++)
            {
                (nextCost[0], nextStart[0]) = (i, 0);
                int letter = d[i - 1];
                for (int j = 1; j <= m; j++)
                {
                    int substitute = cost[j - 1] + (letter == t[j - 1] ? 0 : 1);
                    int delete = cost[j] + 1;
                    int insert = nextCost[j - 1] + 1;
                    if (substitute <= delete && substitute <= insert)
                    {
                        (nextCost[j], nextStart[j]) = (substitute, start[j - 1]);
                    }
                    else if (delete <= insert)
                    {
                        (nextCost[j], nextStart[j]) = (delete, start[j]);
                    }
                    else
                    {
                        (nextCost[j], nextStart[j]) = (insert, nextStart[j - 1]);
                    }
                }

                var swap = cost;
                cost = nextCost;
                nextCost = swap;
                swap = start;
                start = nextStart;
                nextStart = swap;
            }

            int bestEnd = 1;
            for (int j = 2; j <= m; j++)
            {
                if (cost[j] < cost[bestEnd])
                {
                    bestEnd = j;
                }
            }

            int first = index[start[bestEnd]], last = bestEnd >= m ? text.Length : index[bestEnd];
            first = Math.Min(first, last);
            if (options.WordBoundaries)
            {
                // Out to word boundaries: a line breaks between words.
                while (first > from && !char.IsWhiteSpace(text[first - 1]) && first < last)
                {
                    first--;
                }

                while (last < text.Length && !char.IsWhiteSpace(text[last]))
                {
                    last++;
                }
            }

            return new TextMatch(first, last - first, cost[bestEnd]);
        }
        finally
        {
            ArrayPool<int>.Shared.Return(buffer);
        }
    }

    /// <inheritdoc />
    public override string ToString() => "Sellers' semi-global edit distance over Unicode scalars, the span's ends at word boundaries";
}
