// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

namespace Idrak.Imaging.Shaping;

/// <summary>
/// The Unicode bidirectional algorithm (UAX #9) for a paragraph, without explicit embeddings, overrides or isolates (their
/// controls count as boundary neutrals; LRM, RLM and ALM count as the strong characters they are): the paragraph's
/// direction (P2, P3), weak types (W1 to W7), bracket pairs (N0), neutrals (N1, N2), implicit levels (I1, I2), and per
/// line the trailing whitespace reset (L1) and the reordering (L2). Levels are per UTF-16 index.
/// </summary>
internal static class Bidi
{
    /// <summary>The resolved embedding level of every UTF-16 index of a paragraph, and whether the paragraph is right to left.</summary>
    /// <param name="text">The paragraph (no paragraph separator inside).</param>
    /// <param name="rightToLeft">The paragraph's direction, or null for its first strong character's (left to right without one).</param>
    public static (byte[] Levels, BidiClass[] Classes, bool RightToLeft) Resolve(string text, bool? rightToLeft = null)
    {
        int n = text.Length;
        var original = new BidiClass[n];
        for (int i = 0; i < n;)
        {
            int cp = UnicodeText.CodePointAt(text, i, out int w);
            var c = UnicodeText.Bidi(cp);
            if (c is BidiClass.LRE or BidiClass.RLE or BidiClass.LRO or BidiClass.RLO or BidiClass.PDF or BidiClass.LRI or BidiClass.RLI
                or BidiClass.FSI or BidiClass.PDI)
            {
                c = BidiClass.BN;                                                   // explicit controls are not followed
            }

            for (int k = 0; k < w; k++)
            {
                original[i + k] = c;
            }

            i += w;
        }

        bool paragraph = rightToLeft ?? FirstStrong(original) is BidiClass.R or BidiClass.AL;
        var level = (byte)(paragraph ? 1 : 0);
        var embedding = paragraph ? BidiClass.R : BidiClass.L;
        var t = (BidiClass[])original.Clone();

        // W1: a non-spacing mark takes the type before it (the start of text: the paragraph's direction).
        var previous = embedding;
        for (int i = 0; i < n; i++)
        {
            if (t[i] == BidiClass.BN)
            {
                continue;
            }

            if (t[i] == BidiClass.NSM)
            {
                t[i] = previous;
            }

            previous = t[i];
        }

        // W2: European digits after Arabic letters are Arabic digits. W3: Arabic letters are R.
        var lastStrong = embedding;
        for (int i = 0; i < n; i++)
        {
            switch (t[i])
            {
                case BidiClass.L or BidiClass.R or BidiClass.AL:
                    lastStrong = t[i];
                    break;
                case BidiClass.EN when lastStrong == BidiClass.AL:
                    t[i] = BidiClass.AN;
                    break;
            }
        }

        for (int i = 0; i < n; i++)
        {
            if (t[i] == BidiClass.AL)
            {
                t[i] = BidiClass.R;
            }
        }

        // W4: one separator between two numbers of a kind joins them.
        for (int i = 1; i < n - 1; i++)
        {
            int a = PreviousIndex(t, i), b = NextIndex(t, i);
            if (a < 0 || b < 0)
            {
                continue;
            }

            if (t[i] == BidiClass.ES && t[a] == BidiClass.EN && t[b] == BidiClass.EN)
            {
                t[i] = BidiClass.EN;
            }
            else if (t[i] == BidiClass.CS && t[a] == t[b] && t[a] is BidiClass.EN or BidiClass.AN)
            {
                t[i] = t[a];
            }
        }

        // W5: terminators next to European digits are European digits.
        for (int i = 0; i < n; i++)
        {
            if (t[i] != BidiClass.ET)
            {
                continue;
            }

            int end = i;
            while (end < n && t[end] is BidiClass.ET or BidiClass.BN)
            {
                end++;
            }

            int before = PreviousIndex(t, i);
            bool touches = (before >= 0 && t[before] == BidiClass.EN) || (end < n && t[end] == BidiClass.EN);
            if (touches)
            {
                for (int k = i; k < end; k++)
                {
                    if (t[k] == BidiClass.ET)
                    {
                        t[k] = BidiClass.EN;
                    }
                }
            }

            i = end - 1;
        }

        // W6: the remaining separators and terminators are neutral. W7: European digits after L are L.
        for (int i = 0; i < n; i++)
        {
            if (t[i] is BidiClass.ES or BidiClass.ET or BidiClass.CS)
            {
                t[i] = BidiClass.ON;
            }
        }

        lastStrong = embedding;
        for (int i = 0; i < n; i++)
        {
            if (t[i] is BidiClass.L or BidiClass.R)
            {
                lastStrong = t[i];
            }
            else if (t[i] == BidiClass.EN && lastStrong == BidiClass.L)
            {
                t[i] = BidiClass.L;
            }
        }

        PairedBrackets(text, t, original, embedding);

        // N1, N2: a neutral run between two strong types of one direction takes it (numbers count as R); else the embedding's.
        for (int i = 0; i < n; i++)
        {
            if (!IsNeutral(t[i]))
            {
                continue;
            }

            int end = i;
            while (end < n && IsNeutral(t[end]))
            {
                end++;
            }

            var before = i == 0 ? embedding : Strength(t, i - 1, -1, embedding);
            var after = end == n ? embedding : Strength(t, end, 1, embedding);
            var resolved = before == after ? before : embedding;
            for (int k = i; k < end; k++)
            {
                t[k] = resolved;
            }

            i = end - 1;
        }

        // I1, I2: the levels.
        var levels = new byte[n];
        for (int i = 0; i < n; i++)
        {
            levels[i] = (level % 2, t[i]) switch
            {
                (0, BidiClass.R) => (byte)(level + 1),
                (0, BidiClass.AN or BidiClass.EN) => (byte)(level + 2),
                (1, BidiClass.L or BidiClass.EN or BidiClass.AN) => (byte)(level + 1),
                _ => level,
            };
        }

        // Boundary neutrals (removed by X9) take the level of the character before them.
        for (int i = 0; i < n; i++)
        {
            if (original[i] == BidiClass.BN)
            {
                levels[i] = i > 0 ? levels[i - 1] : level;
            }
        }

        // Surrogate pairs keep one level.
        for (int i = 1; i < n; i++)
        {
            if (char.IsLowSurrogate(text[i]) && char.IsHighSurrogate(text[i - 1]))
            {
                levels[i] = levels[i - 1];
            }
        }

        return (levels, original, paragraph);
    }

    /// <summary>
    /// The visual order of a line [start, end) of a resolved paragraph: L1 (separators and trailing whitespace back to the
    /// paragraph level) then L2 (each level and above reversed, from the highest down to the lowest odd one). Returns the
    /// text indices left to right, and the line's levels after L1.
    /// </summary>
    public static (int[] Order, byte[] Levels) Reorder(byte[] levels, BidiClass[] classes, int start, int end, bool rightToLeft)
    {
        int n = end - start;
        var line = new byte[n];
        Array.Copy(levels, start, line, 0, n);
        byte paragraph = (byte)(rightToLeft ? 1 : 0);
        bool trailing = true;
        for (int i = n - 1; i >= 0; i--)
        {
            var c = classes[start + i];
            if (c is BidiClass.S or BidiClass.B)
            {
                line[i] = paragraph;
                trailing = true;
            }
            else if (c is BidiClass.WS or BidiClass.BN)
            {
                if (trailing)
                {
                    line[i] = paragraph;
                }
            }
            else
            {
                trailing = false;
            }
        }

        var order = new int[n];
        for (int i = 0; i < n; i++)
        {
            order[i] = start + i;
        }

        if (n == 0)
        {
            return (order, line);
        }

        byte highest = 0, lowestOdd = byte.MaxValue;
        foreach (byte l in line)
        {
            highest = Math.Max(highest, l);
            if (l % 2 == 1)
            {
                lowestOdd = Math.Min(lowestOdd, l);
            }
        }

        for (int level = highest; level >= lowestOdd && level > 0; level--)
        {
            for (int i = 0; i < n; i++)
            {
                if (line[i] < level)
                {
                    continue;
                }

                int j = i;
                while (j < n && line[j] >= level)
                {
                    j++;
                }

                Array.Reverse(order, i, j - i);
                Array.Reverse(line, i, j - i);
                i = j;
            }

            // Reversing the levels with the order keeps each level with its character; undo for the next pass.
        }

        // Levels follow the characters in visual order now; give them back in text order for the caller.
        var textLevels = new byte[n];
        Array.Copy(levels, start, textLevels, 0, n);
        for (int i = 0; i < n; i++)
        {
            int k = order[i] - start;
            textLevels[k] = line[i];
        }

        return (order, textLevels);
    }

    private static BidiClass FirstStrong(BidiClass[] classes)
    {
        foreach (var c in classes)
        {
            if (c is BidiClass.L or BidiClass.R or BidiClass.AL)
            {
                return c;
            }
        }

        return BidiClass.L;
    }

    private static bool IsNeutral(BidiClass c) => c is BidiClass.B or BidiClass.S or BidiClass.WS or BidiClass.ON or BidiClass.BN;

    // The direction a neighbour gives a neutral run: L, or R for R and numbers; looking from `at` in `step` direction past boundary neutrals.
    private static BidiClass Strength(BidiClass[] t, int at, int step, BidiClass embedding)
    {
        for (int i = at; i >= 0 && i < t.Length; i += step)
        {
            switch (t[i])
            {
                case BidiClass.L:
                    return BidiClass.L;
                case BidiClass.R or BidiClass.AL or BidiClass.EN or BidiClass.AN:
                    return BidiClass.R;
                case BidiClass.BN:
                    continue;
                default:
                    return BidiClass.ON;
            }
        }

        return embedding;
    }

    private static int PreviousIndex(BidiClass[] t, int i)
    {
        for (int k = i - 1; k >= 0; k--)
        {
            if (t[k] != BidiClass.BN)
            {
                return k;
            }
        }

        return -1;
    }

    private static int NextIndex(BidiClass[] t, int i)
    {
        for (int k = i + 1; k < t.Length; k++)
        {
            if (t[k] != BidiClass.BN)
            {
                return k;
            }
        }

        return -1;
    }

    // N0: brackets that pair up take the embedding direction when a strong type of it lies inside, else the opposite
    // one found inside when the context before the opener is also opposite; marks after a bracket follow it.
    private static void PairedBrackets(string text, BidiClass[] t, BidiClass[] original, BidiClass embedding)
    {
        var pairs = new List<(int Open, int Close)>();
        var stack = new Stack<(int At, int Closer)>();
        for (int i = 0; i < text.Length; i++)
        {
            if (t[i] != BidiClass.ON)
            {
                continue;
            }

            int c = text[i];
            int closer = c switch { '(' => ')', '[' => ']', '{' => '}', '〈' => '〉', '〈' => '〉', _ => 0 };
            if (closer != 0)
            {
                if (stack.Count == 63)
                {
                    break;
                }

                stack.Push((i, closer));
            }
            else if (c is ')' or ']' or '}' or '〉' or '〉')
            {
                foreach (var (at, wanted) in stack)
                {
                    if (wanted == c)
                    {
                        while (stack.Peek().At != at)
                        {
                            stack.Pop();
                        }

                        stack.Pop();
                        pairs.Add((at, i));
                        break;
                    }
                }
            }
        }

        pairs.Sort((a, b) => a.Open.CompareTo(b.Open));
        foreach (var (open, close) in pairs)
        {
            BidiClass? inside = null;
            bool sameFound = false;
            for (int k = open + 1; k < close; k++)
            {
                var s = t[k] switch { BidiClass.L => BidiClass.L, BidiClass.R or BidiClass.EN or BidiClass.AN => BidiClass.R, _ => (BidiClass?)null };
                if (s is null)
                {
                    continue;
                }

                if (s == embedding)
                {
                    sameFound = true;
                    break;
                }

                inside = s;
            }

            BidiClass? resolved = null;
            if (sameFound)
            {
                resolved = embedding;
            }
            else if (inside is { } opposite)
            {
                var before = embedding;
                for (int k = open - 1; k >= 0; k--)
                {
                    var s = t[k] switch { BidiClass.L => BidiClass.L, BidiClass.R or BidiClass.EN or BidiClass.AN => BidiClass.R, _ => (BidiClass?)null };
                    if (s is { } found)
                    {
                        before = found;
                        break;
                    }
                }

                resolved = before == opposite ? opposite : embedding;
            }

            if (resolved is { } r)
            {
                t[open] = t[close] = r;
                for (int k = close + 1; k < t.Length && original[k] == BidiClass.NSM; k++)
                {
                    t[k] = r;
                }

                for (int k = open + 1; k < close && original[k] == BidiClass.NSM; k++)
                {
                    if (k == open + 1)
                    {
                        t[k] = r;
                    }
                }
            }
        }
    }
}

/// <summary>
/// A paragraph cut into runs one shaper takes: one direction (bidi level) and one script each, characters of no script of
/// their own (spaces, punctuation, digits, marks) given their neighbours' script.
/// </summary>
internal static class Itemizer
{
    /// <summary>The runs of [start, end) of a resolved paragraph in text order, each with its level.</summary>
    public static List<(TextRun Run, byte Level)> Runs(string text, byte[] levels, int start, int end, string? language)
    {
        // Characters of no script of their own (spaces, punctuation, digits, marks) take the script before them at the
        // same level, else the one after them at the same level, else the one before, else "Zyyy".
        var scripts = new string[text.Length];
        for (int i = start; i < end;)
        {
            int cp = UnicodeText.CodePointAt(text, i, out int w);
            string s = UnicodeText.Script(cp);
            for (int k = 0; k < w; k++)
            {
                scripts[i + k] = s is "Zyyy" or "Zinh" or "Zzzz" ? "" : s;
            }

            i += w;
        }

        var resolved = (string[])scripts.Clone();
        for (int i = start; i < end; i++)
        {
            if (scripts[i].Length > 0)
            {
                continue;
            }

            string? found = null;
            for (int k = i - 1; k >= start && levels[k] == levels[i] && found is null; k--)
            {
                if (resolved[k].Length > 0)
                {
                    found = resolved[k];
                }
            }

            for (int k = i + 1; k < end && levels[k] == levels[i] && found is null; k++)
            {
                if (scripts[k].Length > 0)
                {
                    found = scripts[k];
                }
            }

            for (int k = i - 1; k >= start && found is null; k--)
            {
                if (resolved[k].Length > 0)
                {
                    found = resolved[k];
                }
            }

            resolved[i] = found ?? "Zyyy";
        }

        scripts = resolved;
        var runs = new List<(TextRun, byte)>();
        int runStart = start;
        for (int i = start + 1; i <= end; i++)
        {
            if (i == end || levels[i] != levels[runStart] || scripts[i] != scripts[runStart])
            {
                if (i < end && char.IsLowSurrogate(text[i]))
                {
                    continue;
                }

                runs.Add((new TextRun(runStart, i - runStart, scripts[runStart], levels[runStart] % 2 == 1, language), levels[runStart]));
                runStart = i;
            }
        }

        return runs;
    }
}
