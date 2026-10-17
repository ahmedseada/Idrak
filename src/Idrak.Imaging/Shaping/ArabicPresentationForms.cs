// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Text;

namespace Idrak.Imaging.Shaping;

/// <summary>The Unicode joining types (ArabicShaping.txt), the ways a letter of a cursive script connects to its neighbours.</summary>
public enum JoiningType : byte
{
    /// <summary>Non-joining: the letter connects to neither side (hamza, and every character of a non-joining script).</summary>
    U,

    /// <summary>Right-joining: the letter connects to the one before it only (alef, dal, reh, waw).</summary>
    R,

    /// <summary>Left-joining: the letter connects to the one after it only.</summary>
    L,

    /// <summary>Dual-joining: the letter connects on both sides (beh, seen, lam, meem).</summary>
    D,

    /// <summary>Join-causing: the character joins both sides without a form of its own (tatweel, the zero width joiner).</summary>
    C,

    /// <summary>Transparent: the character does not affect joining across it (vowel marks, shadda, format characters).</summary>
    T,
}

/// <summary>
/// Arabic contextual shaping to presentation forms, for terminals and other renderers that draw one glyph per code point and
/// do not join letters: each letter becomes its isolated, final, initial or medial form (the Arabic Presentation Forms blocks)
/// by whether it joins the letters around it (the joining types of ArabicShaping.txt, marks being transparent, a zero width
/// non-joiner breaking a join and a zero width joiner causing one), and lam followed by alef becomes the lam-alef ligature
/// (isolated or final). Text in logical order in, logical order out: the bidirectional reordering (<see cref="Bidi"/>) comes
/// after. Fonts with OpenType tables are shaped through their own features instead (the shapers registered by script).
/// </summary>
public static class ArabicPresentationForms
{
    private const int Lam = 0x0644;

    /// <summary>
    /// The joining type of a code point: the Unicode property, with the zero width joiner (U+200D) join-causing and the zero
    /// width non-joiner (U+200C) non-joining, as shaping reads them (the property calls both transparent).
    /// </summary>
    public static JoiningType JoiningTypeOf(int codePoint) => codePoint switch
    {
        0x200D => JoiningType.C,
        0x200C => JoiningType.U,
        _ => UnicodeText.Joining(codePoint),
    };

    /// <summary>Whether a code point is an Arabic letter with presentation forms.</summary>
    public static bool HasForms(int codePoint)
    {
        Span<int> forms = stackalloc int[4];
        return UnicodeText.AllPresentationForms(codePoint, forms);
    }

    /// <summary>Whether <paramref name="text"/> holds a letter this class shapes (a quick test before <see cref="Shape(string)"/>).</summary>
    public static bool HasArabic(ReadOnlySpan<char> text)
    {
        // Every letter with presentation forms lies in the Arabic block: a line without it is one vectorized search.
        int first = text.IndexOfAnyInRange('؀', 'ۿ');
        if (first < 0)
        {
            return false;
        }

        Span<int> forms = stackalloc int[4];
        for (int i = first; i < text.Length; i++)
        {
            if (text[i] is >= '؀' and <= 'ۿ' && UnicodeText.AllPresentationForms(text[i], forms))
            {
                return true;
            }
        }

        return false;
    }

    /// <summary>
    /// <paramref name="text"/> with its Arabic letters in their presentation forms and lam-alef pairs as ligatures; other
    /// characters (marks included) are kept as they are, and text without Arabic letters is returned as it is.
    /// </summary>
    public static string Shape(string text)
    {
        ArgumentNullException.ThrowIfNull(text);
        if (!HasArabic(text))
        {
            return text;
        }

        var shaped = Shape(CodePoints(text), out _);
        var output = new StringBuilder(text.Length);
        foreach (int cp in shaped)
        {
            if (cp < 0x10000)
            {
                output.Append((char)cp);
            }
            else
            {
                output.Append(char.ConvertFromUtf32(cp));
            }
        }

        return output.ToString();
    }

    /// <summary>
    /// <paramref name="points"/> (code points in logical order) shaped as <see cref="Shape(string)"/> does, with the index in
    /// <paramref name="points"/> each output code point comes from in <paramref name="from"/> (a ligature from its lam).
    /// </summary>
    public static int[] Shape(ReadOnlySpan<int> points, out int[] from)
    {
        var types = new JoiningType[points.Length];
        for (int i = 0; i < points.Length; i++)
        {
            types[i] = JoiningTypeOf(points[i]);
        }

        var output = new List<int>(points.Length);
        var origins = new List<int>(points.Length);
        Span<int> forms = stackalloc int[4];
        for (int i = 0; i < points.Length; i++)
        {
            int cp = points[i];
            origins.Add(i);
            // A letter that never joins (hamza) keeps its own code point, as its isolated form looks the same.
            if (types[i] == JoiningType.U || !UnicodeText.AllPresentationForms(cp, forms))
            {
                output.Add(cp);
                continue;
            }

            int previous = Neighbour(types, i, -1);
            int next = Neighbour(types, i, 1);
            bool joinsBefore = types[i] is JoiningType.D or JoiningType.R or JoiningType.C && previous >= 0 && types[previous] is JoiningType.D or JoiningType.L or JoiningType.C;
            // Lam then alef (marks between them allowed): the ligature, final when the lam joins the letter before it.
            if (cp == Lam && next >= 0 && UnicodeText.PresentationLamAlef(points[next], final: joinsBefore) is int ligature)
            {
                output.Add(ligature);
                for (int m = i + 1; m < next; m++)
                {
                    output.Add(points[m]);
                    origins.Add(m);
                }

                i = next;
                continue;
            }

            bool joinsAfter = types[i] is JoiningType.D or JoiningType.L or JoiningType.C && next >= 0 && types[next] is JoiningType.D or JoiningType.R or JoiningType.C;
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
    private static int Neighbour(JoiningType[] types, int index, int step)
    {
        for (int i = index + step; i >= 0 && i < types.Length; i += step)
        {
            if (types[i] != JoiningType.T)
            {
                return i;
            }
        }

        return -1;
    }

    // The code points of a string, a lone surrogate kept as its own value so the text round-trips.
    private static int[] CodePoints(string text)
    {
        int count = 0;
        for (int i = 0; i < text.Length; i++, count++)
        {
            if (char.IsHighSurrogate(text[i]) && i + 1 < text.Length && char.IsLowSurrogate(text[i + 1]))
            {
                i++;
            }
        }

        var points = new int[count];
        for (int i = 0, k = 0; i < text.Length; i++, k++)
        {
            if (char.IsHighSurrogate(text[i]) && i + 1 < text.Length && char.IsLowSurrogate(text[i + 1]))
            {
                points[k] = char.ConvertToUtf32(text[i], text[i + 1]);
                i++;
            }
            else
            {
                points[k] = text[i];
            }
        }

        return points;
    }
}
