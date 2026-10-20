// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Text;

namespace Idrak.Imaging.Shaping;

/// <summary>
/// "bidi" (<see cref="ReadingOrders.Bidi"/>): the Unicode bidirectional algorithm (<see cref="Bidi"/>, UAX #9) on one line
/// as a paragraph of the line's direction: levels resolved (rules P, X, W, N, I), the line reordered (L1, L2), the visual
/// order read in the line's direction. Characters are not mirrored. The algorithm maps some different texts to the same
/// columns (a number before a Latin word and after it, in a right-to-left line), so <see cref="FromColumns"/> returns the
/// logical text whose columns are the ones given, found from the columns' own levels and refined until its columns agree
/// (a few rounds); where several texts agree it returns one of them.
/// </summary>
internal sealed class BidiReadingOrder : IReadingOrder
{
    private const int Rounds = 8;

    public string Name => ReadingOrders.Bidi;

    public string ToColumns(string logical, bool rightToLeft)
    {
        ArgumentNullException.ThrowIfNull(logical);
        var units = Units(logical);
        return Gather(logical, units, Order(logical, units, rightToLeft));
    }

    public string FromColumns(string columns, bool rightToLeft)
    {
        ArgumentNullException.ThrowIfNull(columns);
        var units = Units(columns);
        string candidate = columns;
        string best = columns;
        for (int round = 0; round < Rounds; round++)
        {
            // The order a candidate's levels give, undone on the columns: candidate'[order[k]] = columns[k].
            var order = Order(candidate, Units(candidate), rightToLeft);
            var placed = new int[order.Length];
            for (int k = 0; k < order.Length; k++)
            {
                placed[order[k]] = k;
            }

            string next = Gather(columns, units, placed);
            if (string.Equals(ToColumns(next, rightToLeft), columns, StringComparison.Ordinal))
            {
                return next;
            }

            if (string.Equals(next, candidate, StringComparison.Ordinal))
            {
                break;
            }

            (best, candidate) = (next, next);
        }

        return best;
    }

    public override string ToString() => "the Unicode bidirectional algorithm (UAX #9), a paragraph of the line's direction";

    // Each code point's start in the text (UTF-16), with the text's length at the end.
    private static int[] Units(string text)
    {
        var starts = new List<int>(text.Length + 1);
        for (int i = 0; i < text.Length;)
        {
            starts.Add(i);
            i += char.IsHighSurrogate(text[i]) && i + 1 < text.Length && char.IsLowSurrogate(text[i + 1]) ? 2 : 1;
        }

        starts.Add(text.Length);
        return [.. starts];
    }

    // The code points' indexes in column order: the visual order (left to right), reversed for a right-to-left line.
    private static int[] Order(string text, int[] units, bool rightToLeft)
    {
        int count = units.Length - 1;
        if (count == 0)
        {
            return [];
        }

        var points = new int[count];
        var classes = new BidiClass[count];
        for (int k = 0; k < count; k++)
        {
            points[k] = units[k + 1] - units[k] == 2 ? char.ConvertToUtf32(text[units[k]], text[units[k] + 1]) : text[units[k]];
            classes[k] = Bidi.ClassOf(points[k]);
        }

        var resolved = Bidi.Levels(classes, points, rightToLeft ? 1 : 0, out int paragraph);
        var levels = new byte[count];
        int previous = paragraph;
        for (int k = 0; k < count; k++)
        {
            previous = resolved[k] < 0 ? previous : resolved[k];                    // a character X9 removes takes the level before it
            levels[k] = (byte)previous;
        }

        var (order, _) = Bidi.Reorder(levels, classes, 0, count, rightToLeft);
        if (rightToLeft)
        {
            Array.Reverse(order);
        }

        return order;
    }

    // The code points of `text` in the order of `order` (indexes into its code points).
    private static string Gather(string text, int[] units, int[] order)
    {
        var builder = new StringBuilder(text.Length);
        foreach (int k in order)
        {
            builder.Append(text, units[k], units[k + 1] - units[k]);
        }

        return builder.ToString();
    }
}

/// <summary>
/// "runs" (<see cref="ReadingOrders.Runs"/>): in a right-to-left line, every run of left-to-right characters (digits,
/// Arabic-Indic ones too, and letters of left-to-right scripts), with the spaces and separators between two of them
/// (<c>space . , : / -</c> and the Arabic decimal and thousands separators), reversed; in a left-to-right line, every run
/// of right-to-left characters (letters and marks of the right-to-left blocks, not their digits) with the same separators
/// reversed. Its own inverse, so <see cref="FromColumns"/> is <see cref="ToColumns"/>. Text is read as Unicode scalars
/// (a lone surrogate becomes U+FFFD).
/// </summary>
internal sealed class RunsReadingOrder : IReadingOrder
{
    public string Name => ReadingOrders.Runs;

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
}
