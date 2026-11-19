// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Buffers;
using System.Runtime.InteropServices;
using System.Text;

namespace Idrak.Imaging.Shaping;

/// <summary>
/// "bidi" (<see cref="ReadingOrders.Bidi"/>): the Unicode bidirectional algorithm (<see cref="Bidi"/>, UAX #9) on one line
/// as a paragraph of the line's direction: levels resolved (rules P, X, W, N, I), the line reordered (L1, L2), the visual
/// order read in the line's direction. Characters are not mirrored. The algorithm maps some different texts to the same
/// columns (a number before a Latin word and after it, in a right-to-left line), so <see cref="FromColumns"/> returns the
/// logical text whose columns are the ones given, found from the columns' own levels and refined until its columns agree
/// (a few rounds); where several texts agree it returns one of them. A text whose every character resolves to the line's own
/// level (no character of the other direction, no number that rises, no explicit control) is its own columns: it is given
/// back as it is, nothing allocated. Any other text allocates only the string it gives back: the candidates, their
/// columns and the algorithm's working space come from the shared array pools.
/// </summary>
internal sealed class BidiReadingOrder : IReadingOrder
{
    private const int Rounds = 8;

    public string Name => ReadingOrders.Bidi;

    public bool? RightToLeft(string text) => FirstStrong.RightToLeft(text);

    // The reordering is a method of its own in each direction, so that a line the order leaves as it is (most lines) has
    // only these few lines compiled on its first call.
    public string ToColumns(string logical, bool rightToLeft)
    {
        ArgumentNullException.ThrowIfNull(logical);
        return Unchanged(logical, rightToLeft) ? logical : Reordered(logical, rightToLeft);
    }

    public string FromColumns(string columns, bool rightToLeft)
    {
        ArgumentNullException.ThrowIfNull(columns);
        return Unchanged(columns, rightToLeft) ? columns : Restored(columns, rightToLeft);   // unchanged: its own logical order, ToColumns gives it back
    }

    // ToColumns of a text that is not its own columns.
    private static string Reordered(string logical, bool rightToLeft)
    {
        // Working space for a text of n UTF-16 units: its code points' starts (n + 1), an order and the code points (n
        // each), their classes and levels (3n bytes), and the columns (n).
        int n = logical.Length;
        int[] ints = ArrayPool<int>.Shared.Rent(3 * n + 1);
        byte[] bytes = ArrayPool<byte>.Shared.Rent(3 * n);
        char[] chars = ArrayPool<char>.Shared.Rent(n);
        try
        {
            var span = ints.AsSpan();
            int length = Columns(logical, rightToLeft, span[..(n + 1)], span.Slice(n + 1, n), span.Slice(2 * n + 1, n), bytes, chars);
            return new string(chars, 0, length);
        }
        finally
        {
            ArrayPool<char>.Shared.Return(chars);
            ArrayPool<byte>.Shared.Return(bytes);
            ArrayPool<int>.Shared.Return(ints);
        }
    }

    // FromColumns of columns that are not their own logical order.
    private static string Restored(string columns, bool rightToLeft)
    {
        // Working space for texts of n UTF-16 units at most (a candidate has the columns' code points, or fewer where two
        // lone surrogates meet): the code points' starts of the columns and of a candidate (n + 1 each), an order, its
        // inverse and the code points (n each), their classes and levels (3n bytes); the candidate, the next one and the
        // next one's columns (n chars each).
        int n = columns.Length;
        int[] ints = ArrayPool<int>.Shared.Rent(5 * n + 2);
        byte[] bytes = ArrayPool<byte>.Shared.Rent(3 * n);
        char[] chars = ArrayPool<char>.Shared.Rent(3 * n);
        try
        {
            var span = ints.AsSpan();
            Span<int> units = span[..(n + 1)], textUnits = span.Slice(n + 1, n + 1), order = span.Slice(2 * n + 2, n), placed = span.Slice(3 * n + 2, n),
                points = span.Slice(4 * n + 2, n);
            Span<char> candidate = chars.AsSpan(0, n), next = chars.AsSpan(n, n), check = chars.AsSpan(2 * n, n);
            Units(columns, units);
            columns.CopyTo(candidate);
            int candidateLength = n;
            bool candidateIsColumns = true;                                       // the best so far: the columns themselves, or the candidate
            for (int round = 0; round < Rounds; round++)
            {
                // The order a candidate's levels give, undone on the columns: candidate'[order[k]] = columns[k].
                var current = candidate[..candidateLength];
                int count = Units(current, textUnits);
                Order(current, textUnits[..(count + 1)], rightToLeft, order[..count], points, bytes);
                for (int k = 0; k < count; k++)
                {
                    placed[order[k]] = k;
                }

                int nextLength = Gather(columns, units, placed[..count], next);
                var gathered = next[..nextLength];
                if (SameColumns(gathered, columns, rightToLeft, textUnits, order, points, bytes, check))
                {
                    return new string(gathered);
                }

                if (gathered.SequenceEqual(current))
                {
                    break;
                }

                var swap = candidate;
                candidate = next;
                next = swap;
                candidateLength = nextLength;
                candidateIsColumns = false;
            }

            return candidateIsColumns ? columns : new string(candidate[..candidateLength]);
        }
        finally
        {
            ArrayPool<char>.Shared.Return(chars);
            ArrayPool<byte>.Shared.Return(bytes);
            ArrayPool<int>.Shared.Return(ints);
        }
    }

    public override string ToString() => "the Unicode bidirectional algorithm (UAX #9), a paragraph of the line's direction";

    // Whether every code point of `text` (read as Units reads them) resolves to the paragraph's level, so that the visual
    // order is the logical one (read from the right, a right-to-left line's too): no explicit embedding, override or isolate
    // (X1-X10) and no Arabic number (I1, I2); in a left-to-right line no R or AL (European numbers then follow sos L, W7,
    // and neutrals and brackets the strong L around them, N0-N2); in a right-to-left line no L or European number (all the
    // rest resolves to R). Separators and trailing whitespace go to the paragraph's level anyway (L1).
    private static bool Unchanged(ReadOnlySpan<char> text, bool rightToLeft)
    {
        for (int i = 0; i < text.Length; i++)
        {
            int point = text[i];
            if (char.IsHighSurrogate(text[i]) && i + 1 < text.Length && char.IsLowSurrogate(text[i + 1]))
            {
                point = char.ConvertToUtf32(text[i], text[i + 1]);
                i++;
            }

            var type = Bidi.ClassOf(point);
            if (type >= BidiClass.LRE || type == BidiClass.AN || (rightToLeft ? type is BidiClass.L or BidiClass.EN : type is BidiClass.R or BidiClass.AL))
            {
                return false;
            }
        }

        return true;
    }

    // Whether `text`'s columns (ToColumns) are `columns`; the rest is working space, as ToColumns takes it.
    private static bool SameColumns(ReadOnlySpan<char> text, string columns, bool rightToLeft, Span<int> units, Span<int> order, Span<int> points, Span<byte> bytes,
        Span<char> destination)
    {
        if (Unchanged(text, rightToLeft))
        {
            return text.SequenceEqual(columns);
        }

        int length = Columns(text, rightToLeft, units, order, points, bytes, destination);
        return destination[..length].SequenceEqual(columns);
    }

    // `text`'s columns into `destination` (as long as it); returns their length. `units` (one more than the text's length),
    // `order`, `points` (the text's length each) and `bytes` (three times it) are working space.
    private static int Columns(ReadOnlySpan<char> text, bool rightToLeft, Span<int> units, Span<int> order, Span<int> points, Span<byte> bytes, Span<char> destination)
    {
        int count = Units(text, units);
        Order(text, units[..(count + 1)], rightToLeft, order[..count], points, bytes);
        return Gather(text, units, order[..count], destination);
    }

    // Each code point's start in the text (UTF-16) into `starts`, with the text's length after the last; returns the
    // number of code points.
    private static int Units(ReadOnlySpan<char> text, Span<int> starts)
    {
        int count = 0;
        for (int i = 0; i < text.Length;)
        {
            starts[count++] = i;
            i += char.IsHighSurrogate(text[i]) && i + 1 < text.Length && char.IsLowSurrogate(text[i + 1]) ? 2 : 1;
        }

        starts[count] = text.Length;
        return count;
    }

    // The code points' indexes in column order into `order` (one per code point): the visual order (left to right),
    // reversed for a right-to-left line. `points` (as many as the code points) and `bytes` (three times as many) are
    // working space: the code points, their classes, and their levels as Bidi.Levels resolves them and as the line has them.
    private static void Order(ReadOnlySpan<char> text, ReadOnlySpan<int> units, bool rightToLeft, Span<int> order, Span<int> points, Span<byte> bytes)
    {
        int count = order.Length;
        if (count == 0)
        {
            return;
        }

        points = points[..count];
        var classes = MemoryMarshal.Cast<byte, BidiClass>(bytes[..count]);
        var resolved = MemoryMarshal.Cast<byte, sbyte>(bytes.Slice(count, count));
        var levels = bytes.Slice(2 * count, count);
        for (int k = 0; k < count; k++)
        {
            points[k] = units[k + 1] - units[k] == 2 ? char.ConvertToUtf32(text[units[k]], text[units[k] + 1]) : text[units[k]];
            classes[k] = Bidi.ClassOf(points[k]);
        }

        int previous = Bidi.Levels(classes, points, rightToLeft ? 1 : 0, resolved);
        for (int k = 0; k < count; k++)
        {
            previous = resolved[k] < 0 ? previous : resolved[k];                    // a character X9 removes takes the level before it
            levels[k] = (byte)previous;
        }

        Bidi.Reorder(levels, classes, 0, count, rightToLeft, order, levels);   // the line's levels after L1 in place: not needed here
        if (rightToLeft)
        {
            order.Reverse();
        }
    }

    // The code points of `text` in the order of `order` (indexes into its code points, `units` their starts) into
    // `destination`; returns the length written.
    private static int Gather(ReadOnlySpan<char> text, ReadOnlySpan<int> units, ReadOnlySpan<int> order, Span<char> destination)
    {
        int length = 0;
        foreach (int k in order)
        {
            int width = units[k + 1] - units[k];
            text.Slice(units[k], width).CopyTo(destination[length..]);
            length += width;
        }

        return length;
    }
}

/// <summary>
/// "runs" (<see cref="ReadingOrders.Runs"/>): in a right-to-left line, every run of left-to-right characters (digits,
/// Arabic-Indic ones too, and letters of left-to-right scripts), with the spaces and separators between two of them
/// (<c>space . , : / -</c> and the Arabic decimal and thousands separators), reversed; in a left-to-right line, every run
/// of right-to-left characters (letters and marks of the right-to-left blocks, not their digits) with the same separators
/// reversed. Its own inverse, so <see cref="FromColumns"/> is <see cref="ToColumns"/>. Text is read as Unicode scalars
/// (a lone surrogate becomes U+FFFD); a text with nothing to reverse is given back as it is, nothing allocated.
/// </summary>
internal sealed class RunsReadingOrder : IReadingOrder
{
    public string Name => ReadingOrders.Runs;

    public bool? RightToLeft(string text) => FirstStrong.RightToLeft(text);

    public string ToColumns(string logical, bool rightToLeft)
    {
        ArgumentNullException.ThrowIfNull(logical);
        return ReverseRuns(logical, rightToLeft);
    }

    public string FromColumns(string columns, bool rightToLeft)
    {
        ArgumentNullException.ThrowIfNull(columns);
        return ReverseRuns(columns, rightToLeft);
    }

    public override string ToString() => "the runs drawn against the line's direction reversed (digits and Latin in a right-to-left line); its own inverse";

    /// <summary>Whether a character is drawn left to right inside a right-to-left line: a digit (Arabic-Indic too) or a letter of a left-to-right script.</summary>
    internal static bool IsLeftToRight(Rune rune) => Rune.IsDigit(rune) || Rune.IsLetter(rune) && !InRightToLeftBlock(rune);

    // A character of a right-to-left run inside a left-to-right line: a letter or mark of the right-to-left blocks.
    private static bool IsRightToLeft(Rune rune) => InRightToLeftBlock(rune) && !Rune.IsDigit(rune) && (Rune.IsLetter(rune) || IsMark(rune));

    private static bool IsMark(Rune rune) => Rune.GetUnicodeCategory(rune) is System.Globalization.UnicodeCategory.NonSpacingMark
        or System.Globalization.UnicodeCategory.SpacingCombiningMark or System.Globalization.UnicodeCategory.EnclosingMark;

    private static bool InRightToLeftBlock(Rune rune) => rune.Value is >= 0x0590 and <= 0x08FF or >= 0xFB1D and <= 0xFDFF or >= 0xFE70 and <= 0xFEFF
        or >= 0x10800 and <= 0x10FFF or >= 0x1E800 and <= 0x1EFFF;

    // Inside a number or a phrase: a space, a separator or punctuation that keeps the run together.
    private static bool IsJoiner(Rune rune) => rune.Value is ' ' or '.' or ',' or ':' or '/' or '-' or '٫' or '٬';

    // A run: characters drawn against the line's direction, with the joiners between two of them (not at its ends).
    private static string ReverseRuns(string text, bool rightToLeft)
    {
        if (NothingAgainst(text, rightToLeft))
        {
            return text;
        }

        var runes = text.EnumerateRunes().ToArray();
        var builder = new StringBuilder(text.Length);
        Span<char> pair = stackalloc char[2];
        bool Against(Rune rune) => rightToLeft ? IsLeftToRight(rune) : IsRightToLeft(rune);
        for (int i = 0; i < runes.Length;)
        {
            if (!Against(runes[i]))
            {
                builder.Append(pair[..runes[i++].EncodeToUtf16(pair)]);
                continue;
            }

            int end = i, j = i;
            while (j < runes.Length && (Against(runes[j]) || IsJoiner(runes[j])))
            {
                if (Against(runes[j]))
                {
                    end = j;
                }

                j++;
            }

            for (int k = end; k >= i; k--)
            {
                builder.Append(pair[..runes[k].EncodeToUtf16(pair)]);
            }

            i = end + 1;
        }

        return builder.ToString();
    }

    // Whether no scalar of `text` is drawn against the line's direction and none is a lone surrogate (which reads as
    // U+FFFD): every scalar is then written back as it is, so the text is its own columns.
    private static bool NothingAgainst(string text, bool rightToLeft)
    {
        var rest = text.AsSpan();
        while (!rest.IsEmpty)
        {
            if (Rune.DecodeFromUtf16(rest, out var rune, out int used) != OperationStatus.Done || (rightToLeft ? IsLeftToRight(rune) : IsRightToLeft(rune)))
            {
                return false;
            }

            rest = rest[used..];
        }

        return true;
    }
}

/// <summary>
/// A text's direction by its first strong character, Unicode's rules P2 and P3 (UAX #9): the bidirectional classes of
/// <see cref="Bidi.ClassOf"/>, characters inside isolates skipped, the search ending at a paragraph separator. Both reading
/// orders answer <see cref="IReadingOrder.RightToLeft"/> with it. Nothing is allocated.
/// </summary>
internal static class FirstStrong
{
    /// <summary>True for R or AL first, false for L first, null when the text has no strong character.</summary>
    public static bool? RightToLeft(string text)
    {
        ArgumentNullException.ThrowIfNull(text);
        int isolates = 0;
        foreach (var rune in text.EnumerateRunes())
        {
            switch (UnicodeText.Bidi(rune.Value))
            {
                case BidiClass.LRI or BidiClass.RLI or BidiClass.FSI:
                    isolates++;
                    break;
                case BidiClass.PDI when isolates > 0:
                    isolates--;
                    break;
                case BidiClass.B:
                    return null;
                case BidiClass.L when isolates == 0:
                    return false;
                case BidiClass.R or BidiClass.AL when isolates == 0:
                    return true;
            }
        }

        return null;
    }
}
