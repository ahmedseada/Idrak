// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

namespace Idrak.Abstraction;

/// <summary>
/// Which keys each query row sees, as one half-open range per row: row i sees keys c with <see cref="Starts"/>[i] ≤ c &lt;
/// <see cref="Ends"/>[i] (empty when the end is not past the start: that row's attention output is zeros). The rules the
/// library needs are built here (bidirectional, causal, sliding window, packed sequences, image blocks) for
/// <see cref="Tensor.AttentionSpans"/>; several tables (one per sequence of a batch) are joined with
/// <see cref="Concat"/>. Positions count from 0 in the query rows and in the keys alike (the keys are the same sequence).
/// </summary>
public sealed class KeySpans
{
    private readonly int[] _starts;
    private readonly int[] _ends;

    /// <summary>Ranges given row by row (half-open).</summary>
    /// <exception cref="ArgumentException">The two lists differ in length, or a bound is negative.</exception>
    public KeySpans(IReadOnlyList<int> starts, IReadOnlyList<int> ends)
    {
        ArgumentNullException.ThrowIfNull(starts);
        ArgumentNullException.ThrowIfNull(ends);
        if (starts.Count != ends.Count)
        {
            throw new ArgumentException($"{starts.Count} starts and {ends.Count} ends: one of each per row.", nameof(ends));
        }

        _starts = [.. starts];
        _ends = [.. ends];
        if (_starts.Any(s => s < 0) || _ends.Any(e => e < 0))
        {
            throw new ArgumentException("A range's bounds are key positions, 0 or more.", nameof(starts));
        }
    }

    /// <summary>The rows (over every table joined).</summary>
    public int Count => _starts.Length;

    /// <summary>Each row's first key.</summary>
    public IReadOnlyList<int> Starts => _starts;

    /// <summary>Each row's end (exclusive).</summary>
    public IReadOnlyList<int> Ends => _ends;

    /// <summary>Every row sees every key (an encoder; SigLIP's vision layers).</summary>
    public static KeySpans Bidirectional(int rows, int keys)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(rows);
        ArgumentOutOfRangeException.ThrowIfNegative(keys);
        return new KeySpans(new int[rows], Enumerable.Repeat(keys, rows).ToArray());
    }

    /// <summary>
    /// Row i sees keys up to its own (i + 1 exclusive), only the last <paramref name="window"/> of them with a window
    /// (from max(0, i - window + 1), as transformers masks a sliding window; 0: every earlier key).
    /// </summary>
    public static KeySpans Causal(int rows, int window = 0)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(rows);
        ArgumentOutOfRangeException.ThrowIfNegative(window);
        var starts = new int[rows];
        var ends = new int[rows];
        for (int i = 0; i < rows; i++)
        {
            starts[i] = WindowStart(i, window);
            ends[i] = i + 1;
        }

        return new KeySpans(starts, ends);
    }

    /// <summary>
    /// Packed sequences of the given lengths, one after another in one row of positions: each row sees its own
    /// sequence's keys causally (within a <paramref name="window"/> when given), nothing of the others.
    /// </summary>
    public static KeySpans Segments(IReadOnlyList<int> lengths, int window = 0)
    {
        ArgumentNullException.ThrowIfNull(lengths);
        ArgumentOutOfRangeException.ThrowIfNegative(window);
        var starts = new List<int>();
        var ends = new List<int>();
        int begin = 0;
        foreach (int length in lengths)
        {
            ArgumentOutOfRangeException.ThrowIfNegative(length, nameof(lengths));
            for (int i = begin; i < begin + length; i++)
            {
                starts.Add(Math.Max(begin, WindowStart(i, window)));
                ends.Add(i + 1);
            }

            begin += length;
        }

        return new KeySpans(starts, ends);
    }

    /// <summary>
    /// Causal attention in which the rows of an image block (a run of image tokens) also see the whole block: row i sees
    /// keys from 0 (with a <paramref name="window"/>, from max(0, i - window + 1)) up to i, or to the end of its block when
    /// it is inside one. That is Gemma 3's mask: a key is seen when it is not after the row or shares the row's image
    /// block, and, on sliding-window layers, also lies within the window.
    /// </summary>
    /// <param name="rows">Rows (and keys) of the sequence.</param>
    /// <param name="blocks">Each image block's first position and length (blocks do not overlap).</param>
    /// <param name="window">Keys a sliding-window layer sees, its own included (0: every earlier key).</param>
    /// <exception cref="ArgumentException">A block lies outside the rows or overlaps another.</exception>
    public static KeySpans ImageBlocks(int rows, IReadOnlyList<(int Start, int Length)> blocks, int window = 0)
    {
        ArgumentNullException.ThrowIfNull(blocks);
        var spans = Causal(rows, window);
        var taken = new bool[rows];
        foreach (var (start, length) in blocks)
        {
            if (start < 0 || length < 0 || start + length > rows)
            {
                throw new ArgumentException($"The image block [{start}, {start + length}) lies outside {rows} rows.", nameof(blocks));
            }

            for (int i = start; i < start + length; i++)
            {
                if (taken[i])
                {
                    throw new ArgumentException($"Image blocks overlap at row {i}.", nameof(blocks));
                }

                taken[i] = true;
                spans._ends[i] = start + length;
            }
        }

        return spans;
    }

    /// <summary>Tables one after another: the ranges of each sequence of a batch (each table's positions its own).</summary>
    public static KeySpans Concat(params KeySpans[] tables)
    {
        ArgumentNullException.ThrowIfNull(tables);
        return new KeySpans([.. tables.SelectMany(t => t._starts)], [.. tables.SelectMany(t => t._ends)]);
    }

    /// <summary>The starts and ends as tensors of [<see cref="Count"/>] integers held as floats, on <paramref name="device"/>.</summary>
    public (Tensor Starts, Tensor Ends) ToTensors(Device? device = null) =>
        (Tensor.From(_starts.Select(s => (float)s).ToArray(), device), Tensor.From(_ends.Select(e => (float)e).ToArray(), device));

    private static int WindowStart(int row, int window) => window > 0 ? Math.Max(0, row - window + 1) : 0;
}
