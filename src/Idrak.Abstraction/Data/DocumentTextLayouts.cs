// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Buffers;
using System.Globalization;
using System.Runtime.CompilerServices;

namespace Idrak.Abstraction.Data;

/// <summary>The direction a layout gives a page's lines.</summary>
public enum DocumentTextDirection : byte
{
    /// <summary>Each line's own, from its first strong character; a line with none takes the page's.</summary>
    Auto,

    /// <summary>Every line left to right.</summary>
    LeftToRight,

    /// <summary>Every line right to left.</summary>
    RightToLeft,
}

/// <summary>
/// Arranges a page's text runs into lines and text, from their places on the page and the reading order the options
/// name; register one with <c>Register</c>. Formats call the layout <see cref="DocumentOptions.Layout"/> names (<c>Arrange</c>). Safe to use from several threads
/// at once.
/// </summary>
public interface IDocumentTextLayout
{
    /// <summary>
    /// The lines and text of a page of <paramref name="width"/> x <paramref name="height"/> points holding
    /// <paramref name="runs"/> (in content order).
    /// </summary>
    /// <exception cref="Exception">The reading order the options name is not registered.</exception>
    DocumentText Arrange(IReadOnlyList<DocumentTextRun> runs, float width, float height, DocumentOptions options);
}

/// <summary>
/// The document text layouts, by name (ignoring case). The library's, in Idrak.Abstraction: <see cref="Lines"/> (runs
/// grouped into lines by baseline and size, each line's direction from its first strong character by Unicode's rule
/// through the reading order, runs ordered by their places in that direction, words split where a gap passes a share of
/// the font size, then the reading order for logical order), <see cref="Content"/> (the runs in content order, joined,
/// no layout) and <see cref="Tagged"/> (the order a tagged document's author gave).
/// </summary>
public static class DocumentTextLayouts
{
    /// <summary>"lines": the runs laid out into lines by their places on the page.</summary>
    public const string Lines = "lines";

    /// <summary>"content": the runs in content order, joined, no layout.</summary>
    public const string Content = "content";

    /// <summary>
    /// "tagged": the runs in the order the document's author tagged them (<see cref="DocumentTextRun.Tag"/>): each
    /// structure element's runs laid out by <see cref="Lines"/>, the elements in the structure's order, untagged runs
    /// after them, artifacts left out; a page with no tagged run is laid out by <see cref="Lines"/>.
    /// </summary>
    public const string Tagged = "tagged";
}

// "lines": runs grouped into lines by baseline and size; each line's direction from its first strong character (the
// reading order's RightToLeft over its runs in content order), a line with none taking the page's; its runs ordered by
// their places in that direction, a space where the gap between two passes the word gap; the text met that way (the
// columns, read in the line's direction) put in logical order by the reading order. Runs without a place (an empty box)
// are lines of their own, in content order, after the placed ones. It allocates what it gives back (the lines, each
// line's runs and text, the page's text) and the strings the reading order is asked about; its working space (the runs'
// sort keys, the lines' starts, the text being built) comes from the shared array pools.
internal sealed class LinesLayout(IReadingOrder order) : IDocumentTextLayout
{
    public string Name => DocumentTextLayouts.Lines;

    public string Summary => "runs grouped into lines by baseline and size, ordered by their places in each line's direction (its first strong character), words split at gaps, then the reading order";

    public DocumentText Arrange(IReadOnlyList<DocumentTextRun> runs, float width, float height, DocumentOptions options)
    {
        ArgumentNullException.ThrowIfNull(runs);
        ArgumentNullException.ThrowIfNull(options);
        if (runs.Count == 0)
        {
            return DocumentText.Empty;
        }

        int count = runs.Count;
        var keys = ArrayPool<RunKey>.Shared.Rent(count);
        var starts = ArrayPool<int>.Shared.Rent(count + 1);
        var text = ArrayPool<char>.Shared.Rent(256);
        try
        {
            int placedLines = Group(runs, options.LineTolerance, keys, starts, out int placed, out int unplaced);
            var lines = new DocumentLine[placedLines + unplaced];
            bool? page = null;
            bool pageKnown = false;
            for (int l = 0; l < lines.Length; l++)
            {
                // A placed line's keys, or an unplaced run's (a line of its own, after the placed ones).
                var line = l < placedLines ? keys.AsSpan(starts[l], starts[l + 1] - starts[l]) : keys.AsSpan(placed + l - placedLines, 1);
                bool rightToLeft;
                if (options.Direction != DocumentTextDirection.Auto)
                {
                    rightToLeft = options.Direction == DocumentTextDirection.RightToLeft;
                }
                else if (order.RightToLeft(ContentText(runs, line, ref text)) is { } own)
                {
                    rightToLeft = own;
                }
                else
                {
                    if (!pageKnown)
                    {
                        (page, pageKnown) = (order.RightToLeft(PageText(runs, ref text)), true);
                    }

                    rightToLeft = page ?? false;
                }

                lines[l] = Line(runs, line, rightToLeft, order, options.WordGap, ref text);
            }

            return new DocumentText(lines, Joined(lines, lineFeeds: true));
        }
        finally
        {
            ArrayPool<char>.Shared.Return(text);
            ArrayPool<int>.Shared.Return(starts);
            ArrayPool<RunKey>.Shared.Return(keys);
        }
    }

    // The runs with text as sort keys: the placed ones first, sorted by baseline (then x, then content order) and swept, a
    // run joining the line when its baseline is within the tolerance of the line's largest run's (`starts` gets each line's
    // first key, and the placed count after the last); then the unplaced ones in content order, a line each. Returns the
    // number of placed lines.
    private static int Group(IReadOnlyList<DocumentTextRun> runs, float tolerance, RunKey[] keys, int[] starts, out int placed, out int unplaced)
    {
        int count = runs.Count;
        placed = 0;
        unplaced = 0;
        for (int i = 0; i < count; i++)
        {
            var run = runs[i];
            if (run.Text.Length == 0)
            {
                continue;
            }

            if (run.Box.IsEmpty)
            {
                keys[count - ++unplaced] = new RunKey(0f, 0f, run.Order, i);   // from the end; moved after the placed ones below
            }
            else
            {
                keys[placed++] = new RunKey(run.Baseline ?? run.Box.Bottom, run.Box.X, run.Order, i);
            }
        }

        keys.AsSpan(count - unplaced, unplaced).CopyTo(keys.AsSpan(placed, unplaced));
        keys.AsSpan(0, placed).Sort();
        keys.AsSpan(placed, unplaced).Sort();

        int lines = 0;
        float lineBaseline = 0f, lineSize = 0f;
        for (int k = 0; k < placed; k++)
        {
            float baseline = keys[k].First;
            float size = MathF.Max(runs[keys[k].Index].FontSize, 0f);
            if (k > 0 && MathF.Abs(baseline - lineBaseline) <= tolerance * MathF.Max(size, lineSize))
            {
                if (size > lineSize)
                {
                    (lineBaseline, lineSize) = (baseline, size);
                }

                continue;
            }

            starts[lines++] = k;
            (lineBaseline, lineSize) = (baseline, size);
        }

        starts[lines] = placed;
        return lines;
    }

    // The text of a line's runs in content order: what its direction is read from (a run alone: its own text). The line's
    // keys are left in content order.
    private static string ContentText(IReadOnlyList<DocumentTextRun> runs, Span<RunKey> line, ref char[] buffer)
    {
        if (line.Length == 1)
        {
            return runs[line[0].Index].Text;
        }

        for (int k = 0; k < line.Length; k++)
        {
            line[k] = new RunKey(0f, 0f, line[k].Order, line[k].Index);
        }

        line.Sort();
        int length = 0;
        foreach (var key in line)
        {
            Append(ref buffer, ref length, runs[key.Index].Text);
        }

        return new string(buffer, 0, length);
    }

    // The text of all the page's runs in content order: the direction of a line that has none of its own.
    private static string PageText(IReadOnlyList<DocumentTextRun> runs, ref char[] buffer)
    {
        int count = runs.Count;
        var keys = ArrayPool<RunKey>.Shared.Rent(count);
        try
        {
            for (int i = 0; i < count; i++)
            {
                keys[i] = new RunKey(0f, 0f, runs[i].Order, i);
            }

            var all = keys.AsSpan(0, count);
            all.Sort();
            int length = 0;
            foreach (var key in all)
            {
                Append(ref buffer, ref length, runs[key.Index].Text);
            }

            return new string(buffer, 0, length);
        }
        finally
        {
            ArrayPool<RunKey>.Shared.Return(keys);
        }
    }

    private static DocumentLine Line(IReadOnlyList<DocumentTextRun> runs, Span<RunKey> line, bool rightToLeft, IReadingOrder order, float wordGap, ref char[] columns)
    {
        // By place in the line's direction: from the left edge, or from the right edge for a right-to-left line.
        for (int k = 0; k < line.Length; k++)
        {
            var box = runs[line[k].Index].Box;
            line[k] = new RunKey(rightToLeft ? box.Right : box.X, 0f, line[k].Order, line[k].Index, descending: rightToLeft);
        }

        line.Sort();
        var group = new int[line.Length];
        var bounds = DocumentBox.Empty;
        int length = 0;
        DocumentTextRun? previous = null;
        for (int k = 0; k < line.Length; k++)
        {
            group[k] = line[k].Index;
            var run = runs[group[k]];
            if (previous is not null)
            {
                float gap = rightToLeft ? previous.Box.X - run.Box.Right : run.Box.X - previous.Box.Right;
                if (gap > wordGap * MathF.Max(previous.FontSize, run.FontSize) && length > 0 && !char.IsWhiteSpace(columns[length - 1]) && !char.IsWhiteSpace(run.Text[0]))
                {
                    Append(ref columns, ref length, " ");
                }
            }

            // The run's characters as the line's columns meet them in its direction.
            if (run.Logical)
            {
                Append(ref columns, ref length, order.ToColumns(run.Text, rightToLeft));
            }
            else if (rightToLeft)
            {
                Reserve(ref columns, length, run.Text.Length);
                Reverse(run.Text, columns.AsSpan(length, run.Text.Length));
                length += run.Text.Length;
            }
            else
            {
                Append(ref columns, ref length, run.Text);
            }

            bounds = bounds.Union(run.Box);
            previous = run;
        }

        return new DocumentLine(order.FromColumns(new string(columns, 0, length), rightToLeft), bounds, rightToLeft, group);
    }

    // A text read from its end: its text elements (a base and its marks, a surrogate pair) in reverse order, each kept whole.
    internal static string Reversed(string text) =>
        text.Length < 2 ? text : string.Create(text.Length, text, static (destination, source) => Reverse(source, destination));

    // `source` read from its end into `destination` (as long), its text elements kept whole.
    private static void Reverse(ReadOnlySpan<char> source, Span<char> destination)
    {
        int at = destination.Length;
        while (source.Length > 0)
        {
            int length = Math.Max(1, StringInfo.GetNextTextElementLength(source));
            at -= length;
            source[..length].CopyTo(destination[at..]);
            source = source[length..];
        }
    }

    // `text` appended to the pooled `buffer` holding `length` characters (a null text appends nothing).
    private static void Append(ref char[] buffer, ref int length, ReadOnlySpan<char> text)
    {
        Reserve(ref buffer, length, text.Length);
        text.CopyTo(buffer.AsSpan(length));
        length += text.Length;
    }

    // Room in the pooled `buffer` for `more` characters after the `used` ones: a larger one rented, the old one returned.
    private static void Reserve(ref char[] buffer, int used, int more)
    {
        if (used + more <= buffer.Length)
        {
            return;
        }

        var larger = ArrayPool<char>.Shared.Rent(Math.Max(used + more, buffer.Length * 2));
        buffer.AsSpan(0, used).CopyTo(larger);
        ArrayPool<char>.Shared.Return(buffer);
        buffer = larger;
    }

    /// <summary>The lines' texts joined into one string, made once: by line feeds, or end to end (a missing text joins as empty).</summary>
    internal static string Joined(DocumentLine[] lines, bool lineFeeds)
    {
        if (lines.Length == 1)
        {
            return lines[0].Text ?? "";
        }

        int length = lineFeeds ? Math.Max(lines.Length - 1, 0) : 0;
        foreach (var line in lines)
        {
            length += line.Text?.Length ?? 0;
        }

        return string.Create(length, (lines, lineFeeds), static (destination, state) =>
        {
            int at = 0;
            for (int i = 0; i < state.lines.Length; i++)
            {
                if (i > 0 && state.lineFeeds)
                {
                    destination[at++] = '\n';
                }

                var text = state.lines[i].Text.AsSpan();
                text.CopyTo(destination[at..]);
                at += text.Length;
            }
        });
    }

    /// <summary>
    /// A run's sort key: by <see cref="First"/> (descending when <see cref="Descending"/>), then <see cref="Second"/>, then its
    /// place in the content, then its index, so no two keys compare equal and every sort gives the one order. Sorted as a
    /// span of structs: no comparison delegate, nothing boxed.
    /// </summary>
    internal readonly struct RunKey(float first, float second, int order, int index, bool descending = false) : IComparable<RunKey>
    {
        public float First { get; } = first;

        public float Second { get; } = second;

        public int Order { get; } = order;

        public int Index { get; } = index;

        public bool Descending { get; } = descending;

        public int CompareTo(RunKey other)
        {
            int c = Descending ? other.First.CompareTo(First) : First.CompareTo(other.First);
            c = c != 0 ? c : Second.CompareTo(other.Second);
            c = c != 0 ? c : Order.CompareTo(other.Order);
            return c != 0 ? c : Index.CompareTo(other.Index);
        }
    }
}

// "content": the runs in content order, their text joined as it is; one line per run (left to right), no layout.
internal sealed class ContentLayout : IDocumentTextLayout
{
    public string Name => DocumentTextLayouts.Content;

    public string Summary => "the runs in content order, joined as they are, one line per run; no layout";

    public DocumentText Arrange(IReadOnlyList<DocumentTextRun> runs, float width, float height, DocumentOptions options)
    {
        ArgumentNullException.ThrowIfNull(runs);
        if (runs.Count == 0)
        {
            return DocumentText.Empty;
        }

        int count = runs.Count;
        var keys = ArrayPool<LinesLayout.RunKey>.Shared.Rent(count);
        try
        {
            for (int i = 0; i < count; i++)
            {
                keys[i] = new LinesLayout.RunKey(0f, 0f, runs[i].Order, i);
            }

            keys.AsSpan(0, count).Sort();
            var lines = new DocumentLine[count];
            for (int k = 0; k < count; k++)
            {
                int index = keys[k].Index;
                var run = runs[index];
                lines[k] = new DocumentLine(run.Text, run.Box, false, [index]);
            }

            return new DocumentText(lines, LinesLayout.Joined(lines, lineFeeds: false));
        }
        finally
        {
            ArrayPool<LinesLayout.RunKey>.Shared.Return(keys);
        }
    }
}
