// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Globalization;
using System.Text;

namespace Idrak.Cli.Shared;

/// <summary>
/// Arabic contextual shaping for terminals that draw one glyph per cell and do not join letters: each letter becomes
/// its presentation form (isolated, final, initial or medial, from the Arabic Presentation Forms blocks) by whether it
/// joins the letters around it (the joining types of ArabicShaping.txt, nonspacing marks being transparent), and lam
/// followed by alef becomes the lam-alef ligature (isolated or final). Text in logical order in, logical order out:
/// the bidirectional reordering comes after (<see cref="VisualText"/>).
/// </summary>
internal static class ArabicShaping
{
    private static readonly Dictionary<int, int[]> FormTable = BuildForms();
    private static readonly Dictionary<int, (int Isolated, int Final)> LamAlefTable = BuildLamAlef();

    private const int Lam = 0x0644;

    /// <summary>The joining type of a code point: 'U' (none), 'D' (both sides), 'R', 'L', 'C' (causes joining) or 'T' (transparent).</summary>
    public static char JoiningType(int codePoint)
    {
        var table = UnicodeTables.Joining;
        int low = 0, high = table.Length / 3 - 1;
        while (low <= high)
        {
            int middle = (low + high) / 2;
            if (codePoint < table[middle * 3])
            {
                high = middle - 1;
            }
            else if (codePoint > table[middle * 3 + 1])
            {
                low = middle + 1;
            }
            else
            {
                return (char)table[middle * 3 + 2];
            }
        }

        var category = CharUnicodeInfo.GetUnicodeCategory(codePoint);
        return category is UnicodeCategory.NonSpacingMark or UnicodeCategory.EnclosingMark or UnicodeCategory.Format ? 'T' : 'U';
    }

    /// <summary>Whether <paramref name="text"/> holds a letter this class shapes (a quick test before <see cref="Shape"/>).</summary>
    public static bool HasArabic(string text)
    {
        foreach (char c in text)
        {
            if (FormTable.ContainsKey(c))
            {
                return true;
            }
        }

        return false;
    }

    /// <summary>
    /// <paramref name="text"/> with its Arabic letters in their presentation forms and lam-alef pairs as ligatures; other
    /// characters (marks included) are kept as they are.
    /// </summary>
    public static string Shape(string text)
    {
        if (!HasArabic(text))
        {
            return text;
        }

        var output = new StringBuilder(text.Length);
        foreach (int cp in Shape(VisualText.CodePoints(text), out _))
        {
            VisualText.Append(output, cp);
        }

        return output.ToString();
    }

    /// <summary>
    /// <paramref name="points"/> shaped as <see cref="Shape(string)"/> does, with the index in <paramref name="points"/>
    /// each output code point comes from (a ligature from its lam).
    /// </summary>
    public static int[] Shape(int[] points, out int[] from)
    {
        var types = Array.ConvertAll(points, JoiningType);
        var output = new List<int>(points.Length);
        var origins = new List<int>(points.Length);
        for (int i = 0; i < points.Length; i++)
        {
            int cp = points[i];
            origins.Add(i);
            // A letter that never joins (hamza) keeps its own code point, as its isolated form looks the same.
            if (types[i] == 'U' || !FormTable.TryGetValue(cp, out var forms))
            {
                output.Add(cp);
                continue;
            }

            int previous = Neighbour(types, i, -1);
            int next = Neighbour(types, i, 1);
            bool joinsBefore = types[i] is 'D' or 'R' or 'C' && previous >= 0 && types[previous] is 'D' or 'L' or 'C';
            // Lam then alef (marks between them allowed): the ligature, final when the lam joins the letter before it.
            if (cp == Lam && next >= 0 && LamAlefTable.TryGetValue(points[next], out var ligature))
            {
                output.Add(joinsBefore ? ligature.Final : ligature.Isolated);
                for (int m = i + 1; m < next; m++)
                {
                    output.Add(points[m]);
                    origins.Add(m);
                }

                i = next;
                continue;
            }

            bool joinsAfter = types[i] is 'D' or 'L' or 'C' && next >= 0 && types[next] is 'D' or 'R' or 'C';
            // forms: isolated, final, initial, medial (0 where the letter has none, so it falls back to a simpler form).
            int form = (joinsBefore, joinsAfter) switch
            {
                (true, true) => forms[3] != 0 ? forms[3] : forms[1],
                (true, false) => forms[1],
                (false, true) => forms[2] != 0 ? forms[2] : forms[0],
                _ => forms[0],
            };
            output.Add(form != 0 ? form : forms[0] != 0 ? forms[0] : cp);
        }

        from = [.. origins];
        return [.. output];
    }

    // The nearest character before (step -1) or after (+1) index that is not transparent, or -1.
    private static int Neighbour(char[] types, int index, int step)
    {
        for (int i = index + step; i >= 0 && i < types.Length; i += step)
        {
            if (types[i] != 'T')
            {
                return i;
            }
        }

        return -1;
    }

    private static Dictionary<int, int[]> BuildForms()
    {
        var table = UnicodeTables.Forms;
        var forms = new Dictionary<int, int[]>(table.Length / 5);
        for (int i = 0; i < table.Length; i += 5)
        {
            forms[table[i]] = [table[i + 1], table[i + 2], table[i + 3], table[i + 4]];
        }

        return forms;
    }

    private static Dictionary<int, (int, int)> BuildLamAlef()
    {
        var table = UnicodeTables.LamAlef;
        var ligatures = new Dictionary<int, (int, int)>();
        for (int i = 0; i < table.Length; i += 3)
        {
            ligatures[table[i]] = (table[i + 1], table[i + 2]);
        }

        return ligatures;
    }
}
