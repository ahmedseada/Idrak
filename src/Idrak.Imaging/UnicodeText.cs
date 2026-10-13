// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

namespace Idrak.Imaging;

/// <summary>The Unicode bidirectional classes (UAX #9).</summary>
internal enum BidiClass : byte
{
    L, R, AL, EN, ES, ET, AN, CS, NSM, BN, B, S, WS, ON, LRE, LRO, RLE, RLO, PDF, LRI, RLI, FSI, PDI,
}

/// <summary>The Unicode joining types (ArabicShaping.txt): non-joining, right, left, dual, join-causing, transparent.</summary>
internal enum JoiningType : byte
{
    U, R, L, D, C, T,
}

/// <summary>
/// The Unicode properties text shaping reads that .NET does not give: script, bidirectional class, joining type,
/// canonical combining class, Indic categories and bidi mirrors, from tables made by tools/unicode/make_text_tables.py
/// (UnicodeText.Data.cs). Each is a binary search over range starts; nothing is allocated.
/// </summary>
internal static partial class UnicodeText
{
    /// <summary>The ISO 15924 code of a code point's script ("Zyyy" common, "Zinh" inherited, "Zzzz" unknown).</summary>
    public static string Script(int codePoint) => ScriptTags[Find(ScriptStarts, ScriptValues, codePoint)];

    /// <summary>A code point's bidirectional class.</summary>
    public static BidiClass Bidi(int codePoint) => (BidiClass)Find(BidiStarts, BidiValues, codePoint);

    /// <summary>A code point's joining type (U for most).</summary>
    public static JoiningType Joining(int codePoint) => (JoiningType)Find(JoiningStarts, JoiningValues, codePoint);

    /// <summary>A code point's canonical combining class (0 for a base character).</summary>
    public static int CombiningClass(int codePoint) => Find(CombiningStarts, CombiningValues, codePoint);

    /// <summary>A code point's Indic syllabic category ("Consonant", "Vowel_Dependent", ...; "Other").</summary>
    public static string IndicSyllabic(int codePoint) => SyllabicNames[Find(SyllabicStarts, SyllabicValues, codePoint)];

    /// <summary>A code point's Indic positional category ("Left", "Top", ...; "NA").</summary>
    public static string IndicPositional(int codePoint) => PositionalNames[Find(PositionalStarts, PositionalValues, codePoint)];

    /// <summary>The bidi mirror of a code point ("(" for ")"), or the code point itself.</summary>
    public static int Mirror(int codePoint)
    {
        var pairs = Mirrors;
        int lo = 0, hi = pairs.Length / 2 - 1;
        while (lo <= hi)
        {
            int mid = (lo + hi) >>> 1;
            int key = pairs[2 * mid];
            if (key == codePoint)
            {
                return pairs[2 * mid + 1];
            }

            if (key < codePoint)
            {
                lo = mid + 1;
            }
            else
            {
                hi = mid - 1;
            }
        }

        return codePoint;
    }

    /// <summary>Whether a code point is a default-ignorable one shaping hides (zero width joiners and marks, bidi controls, variation selectors).</summary>
    public static bool IsDefaultIgnorable(int c) =>
        c is 0x00AD or 0x034F or 0x061C or 0x115F or 0x1160 or 0x17B4 or 0x17B5 or 0x3164 or 0xFEFF or 0xFFA0
        || c is >= 0x180B and <= 0x180F || c is >= 0x200B and <= 0x200F || c is >= 0x202A and <= 0x202E || c is >= 0x2060 and <= 0x206F
        || c is >= 0xFE00 and <= 0xFE0F || c is >= 0xFFF0 and <= 0xFFF8 || c is >= 0x1BCA0 and <= 0x1BCA3 || c is >= 0x1D173 and <= 0x1D17A
        || c is >= 0xE0000 and <= 0xE0FFF;

    /// <summary>Whether a code point is a combining mark (general category Mn, Mc or Me).</summary>
    public static bool IsMark(int codePoint)
    {
        var category = codePoint <= 0xFFFF ? char.GetUnicodeCategory((char)codePoint) : System.Globalization.CharUnicodeInfo.GetUnicodeCategory(codePoint);
        return category is System.Globalization.UnicodeCategory.NonSpacingMark or System.Globalization.UnicodeCategory.SpacingCombiningMark
            or System.Globalization.UnicodeCategory.EnclosingMark;
    }

    /// <summary>The code point at <paramref name="index"/> of <paramref name="text"/> and its length (a lone surrogate reads as U+FFFD).</summary>
    public static int CodePointAt(string text, int index, out int width)
    {
        char c = text[index];
        if (char.IsHighSurrogate(c) && index + 1 < text.Length && char.IsLowSurrogate(text[index + 1]))
        {
            width = 2;
            return char.ConvertToUtf32(c, text[index + 1]);
        }

        width = 1;
        return char.IsSurrogate(c) ? 0xFFFD : c;
    }

    /// <summary>The presentation form (U+FB50..U+FEFF) of an Arabic letter: form 0 isolated, 1 final, 2 initial, 3 medial; null when Unicode has none.</summary>
    public static int? PresentationForm(int codePoint, int form)
    {
        var table = PresentationForms;
        int lo = 0, hi = table.Length / 3 - 1;
        while (lo <= hi)
        {
            int mid = (lo + hi) >>> 1;
            int key = table[3 * mid] * 4 + table[3 * mid + 1], wanted = codePoint * 4 + form;
            if (key == wanted)
            {
                return table[3 * mid + 2];
            }

            if (key < wanted)
            {
                lo = mid + 1;
            }
            else
            {
                hi = mid - 1;
            }
        }

        return null;
    }

    /// <summary>The lam-alef ligature of lam and <paramref name="alef"/> (isolated, or final when the lam joins the letter before), or null.</summary>
    public static int? PresentationLamAlef(int alef, bool final)
    {
        var table = LamAlef;
        for (int i = 0; i < table.Length; i += 3)
        {
            if (table[i] == alef && table[i + 1] == (final ? 1 : 0))
            {
                return table[i + 2];
            }
        }

        return null;
    }

    // The value of the range holding the code point: the last start at or below it.
    private static int Find(ReadOnlySpan<int> starts, ReadOnlySpan<byte> values, int codePoint)
    {
        if ((uint)codePoint > 0x10FFFF)
        {
            return 0;
        }

        int lo = 0, hi = starts.Length - 1;
        while (lo < hi)
        {
            int mid = (lo + hi + 1) >>> 1;
            if (starts[mid] <= codePoint)
            {
                lo = mid;
            }
            else
            {
                hi = mid - 1;
            }
        }

        return values[lo];
    }
}
