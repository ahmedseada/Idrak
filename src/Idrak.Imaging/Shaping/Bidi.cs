// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Numerics;

namespace Idrak.Imaging.Shaping;

/// <summary>The bidirectional character types of the Unicode Bidirectional Algorithm (UAX #9, table 4).</summary>
public enum BidiClass : byte
{
    /// <summary>Left-to-right (letters of most scripts).</summary>
    L,

    /// <summary>Right-to-left (Hebrew letters, and the right-to-left mark).</summary>
    R,

    /// <summary>Arabic letter (Arabic, Syriac and Thaana letters, and the Arabic letter mark).</summary>
    AL,

    /// <summary>European number (the digits 0-9 and their look-alikes).</summary>
    EN,

    /// <summary>European number separator (plus and minus).</summary>
    ES,

    /// <summary>European number terminator (currency signs, percent, degree).</summary>
    ET,

    /// <summary>Arabic number (the Arabic-Indic digits).</summary>
    AN,

    /// <summary>Common number separator (colon, comma, full stop, no-break space).</summary>
    CS,

    /// <summary>Nonspacing mark (combining marks).</summary>
    NSM,

    /// <summary>Boundary neutral (format characters and controls, removed by rule X9).</summary>
    BN,

    /// <summary>Paragraph separator.</summary>
    B,

    /// <summary>Segment separator (tab).</summary>
    S,

    /// <summary>Whitespace.</summary>
    WS,

    /// <summary>Other neutral (punctuation and symbols).</summary>
    ON,

    /// <summary>Left-to-right embedding (U+202A).</summary>
    LRE,

    /// <summary>Left-to-right override (U+202D).</summary>
    LRO,

    /// <summary>Right-to-left embedding (U+202B).</summary>
    RLE,

    /// <summary>Right-to-left override (U+202E).</summary>
    RLO,

    /// <summary>Pop directional format (U+202C), the end of an embedding or override.</summary>
    PDF,

    /// <summary>Left-to-right isolate (U+2066).</summary>
    LRI,

    /// <summary>Right-to-left isolate (U+2067).</summary>
    RLI,

    /// <summary>First strong isolate (U+2068), its direction from the first strong character inside it.</summary>
    FSI,

    /// <summary>Pop directional isolate (U+2069), the end of an isolate.</summary>
    PDI,
}

/// <summary>
/// The Unicode Bidirectional Algorithm (UAX #9, the revision of Unicode <see cref="UnicodeVersion"/>) for one paragraph,
/// complete: explicit embeddings, overrides and isolates (X1-X10), weak and neutral types with the paired brackets of
/// BidiBrackets.txt (W1-W7, N0-N2), implicit levels (I1-I2), and the line rules (L1 levels, L2 reordering). L3 (combining
/// marks after their base) and L4 (mirrored glyphs, <see cref="Mirror"/>) are left to the caller, as the standard leaves
/// them to the renderer. <see cref="Levels"/> and <see cref="VisualOrder"/> work on code points and their classes (a
/// terminal renderer); <see cref="Resolve"/> and <see cref="Reorder"/> on a string's UTF-16 indexes, a line at a time (text
/// layout). Checked against the Unicode conformance files BidiTest.txt and BidiCharacterTest.txt.
/// </summary>
public static class Bidi
{
    private const int MaxDepth = 125;

    /// <summary>The Unicode version the character tables come from.</summary>
    public static string UnicodeVersion => UnicodeText.Version;

    /// <summary>The bidirectional class of a code point.</summary>
    public static BidiClass ClassOf(int codePoint) => UnicodeText.Bidi(codePoint);

    /// <summary>The mirrored glyph of a code point shown right to left (BidiMirroring.txt), or the code point itself.</summary>
    public static int Mirror(int codePoint) => UnicodeText.Mirror(codePoint);

    /// <summary>Whether a class is removed by rule X9 (embedding and override controls, boundary neutrals).</summary>
    public static bool Removed(BidiClass type) => type is BidiClass.RLE or BidiClass.LRE or BidiClass.RLO or BidiClass.LRO or BidiClass.PDF or BidiClass.BN;

    /// <summary>
    /// The paragraph level by rules P2 and P3: 1 when the first strong character (outside isolates) is R or AL, 0 when it
    /// is L or there is none (<paramref name="fallback"/> then).
    /// </summary>
    public static int ParagraphLevel(ReadOnlySpan<BidiClass> types, int fallback = 0)
    {
        int isolates = 0;
        foreach (var type in types)
        {
            switch (type)
            {
                case BidiClass.LRI or BidiClass.RLI or BidiClass.FSI:
                    isolates++;
                    break;
                case BidiClass.PDI when isolates > 0:
                    isolates--;
                    break;
                case BidiClass.B:
                    return fallback;
                case BidiClass.L when isolates == 0:
                    return 0;
                case BidiClass.R or BidiClass.AL when isolates == 0:
                    return 1;
            }
        }

        return fallback;
    }

    /// <summary>
    /// The resolved embedding level of each character of one paragraph (after L1), or -1 for a character rule X9 removes.
    /// <paramref name="codePoints"/> identifies paired brackets (rule N0); it may be empty when only classes are known.
    /// <paramref name="paragraphLevel"/> is 0, 1, or -1 to find it by P2 and P3 (<paramref name="resolvedParagraphLevel"/>).
    /// </summary>
    public static sbyte[] Levels(ReadOnlySpan<BidiClass> types, ReadOnlySpan<int> codePoints, int paragraphLevel, out int resolvedParagraphLevel)
    {
        if (!codePoints.IsEmpty && codePoints.Length != types.Length)
        {
            throw new ArgumentException($"{codePoints.Length} code points for {types.Length} classes.", nameof(codePoints));
        }

        if (paragraphLevel is < -1 or > 1)
        {
            throw new ArgumentOutOfRangeException(nameof(paragraphLevel), paragraphLevel, "The paragraph level is 0, 1, or -1 to find it.");
        }

        int n = types.Length;
        int paragraph = paragraphLevel is 0 or 1 ? paragraphLevel : ParagraphLevel(types);
        resolvedParagraphLevel = paragraph;
        var levels = new sbyte[n];
        if (n == 0)
        {
            return levels;
        }

        var original = types.ToArray();
        var result = types.ToArray();
        MatchIsolates(original, out var matchingPdi, out var matchingInitiator);

        // X1-X8: explicit levels and directions.
        var stackLevel = new int[MaxDepth + 2];
        var stackOverride = new BidiClass[MaxDepth + 2];
        var stackIsolate = new bool[MaxDepth + 2];
        int depth = 0;
        stackLevel[0] = paragraph;
        stackOverride[0] = BidiClass.ON;
        int overflowIsolates = 0, overflowEmbeddings = 0, validIsolates = 0;
        for (int i = 0; i < n; i++)
        {
            var type = original[i];
            switch (type)
            {
                case BidiClass.RLE or BidiClass.LRE or BidiClass.RLO or BidiClass.LRO:
                {
                    bool rtl = type is BidiClass.RLE or BidiClass.RLO;
                    int level = rtl ? (stackLevel[depth] + 1) | 1 : (stackLevel[depth] + 2) & ~1;
                    if (level <= MaxDepth && overflowIsolates == 0 && overflowEmbeddings == 0)
                    {
                        depth++;
                        stackLevel[depth] = level;
                        stackOverride[depth] = type == BidiClass.RLO ? BidiClass.R : type == BidiClass.LRO ? BidiClass.L : BidiClass.ON;
                        stackIsolate[depth] = false;
                    }
                    else if (overflowIsolates == 0)
                    {
                        overflowEmbeddings++;
                    }

                    levels[i] = (sbyte)stackLevel[depth];
                    break;
                }

                case BidiClass.RLI or BidiClass.LRI or BidiClass.FSI:
                {
                    levels[i] = (sbyte)stackLevel[depth];
                    if (stackOverride[depth] != BidiClass.ON)
                    {
                        result[i] = stackOverride[depth];
                    }

                    bool rtl = type == BidiClass.RLI;
                    if (type == BidiClass.FSI)
                    {
                        int end = matchingPdi[i] >= 0 ? matchingPdi[i] : n;
                        rtl = ParagraphLevel(original.AsSpan(i + 1, end - i - 1)) == 1;
                    }

                    int level = rtl ? (stackLevel[depth] + 1) | 1 : (stackLevel[depth] + 2) & ~1;
                    if (level <= MaxDepth && overflowIsolates == 0 && overflowEmbeddings == 0)
                    {
                        validIsolates++;
                        depth++;
                        stackLevel[depth] = level;
                        stackOverride[depth] = BidiClass.ON;
                        stackIsolate[depth] = true;
                    }
                    else
                    {
                        overflowIsolates++;
                    }

                    break;
                }

                case BidiClass.PDI:
                    if (overflowIsolates > 0)
                    {
                        overflowIsolates--;
                    }
                    else if (validIsolates > 0)
                    {
                        overflowEmbeddings = 0;
                        while (!stackIsolate[depth])
                        {
                            depth--;
                        }

                        depth--;
                        validIsolates--;
                    }

                    levels[i] = (sbyte)stackLevel[depth];
                    if (stackOverride[depth] != BidiClass.ON)
                    {
                        result[i] = stackOverride[depth];
                    }

                    break;

                case BidiClass.PDF:
                    if (overflowIsolates == 0)
                    {
                        if (overflowEmbeddings > 0)
                        {
                            overflowEmbeddings--;
                        }
                        else if (!stackIsolate[depth] && depth > 0)
                        {
                            depth--;
                        }
                    }

                    levels[i] = (sbyte)stackLevel[depth];
                    break;

                case BidiClass.B:
                    levels[i] = (sbyte)paragraph;
                    break;

                default:
                    levels[i] = (sbyte)stackLevel[depth];
                    if (stackOverride[depth] != BidiClass.ON && type != BidiClass.BN)
                    {
                        result[i] = stackOverride[depth];
                    }

                    break;
            }
        }

        // X9 removes the embedding controls and boundary neutrals; X10 splits the rest into isolating run sequences.
        // Each sequence's sos and eos come from the explicit levels, before the implicit rules raise any of them.
        var explicitLevels = (sbyte[])levels.Clone();
        foreach (var sequence in RunSequences(original, levels, matchingPdi, matchingInitiator))
        {
            ResolveSequence(sequence, original, result, explicitLevels, levels, codePoints, paragraph);
        }

        // L1: separators, and whitespace and isolate controls before them or at the end of the line, to the paragraph level.
        bool trailing = true;
        for (int i = n - 1; i >= 0; i--)
        {
            var type = original[i];
            if (type is BidiClass.S or BidiClass.B)
            {
                levels[i] = (sbyte)paragraph;
                trailing = true;
            }
            else if (IsWhitespaceOrIsolate(type) || Removed(type))
            {
                if (trailing)
                {
                    levels[i] = (sbyte)paragraph;
                }
            }
            else
            {
                trailing = false;
            }
        }

        for (int i = 0; i < n; i++)
        {
            if (Removed(original[i]))
            {
                levels[i] = -1;
            }
        }

        return levels;
    }

    /// <summary>Rule L2: the indexes of the characters in visual order (left to right), leaving out those with level -1.</summary>
    public static int[] VisualOrder(ReadOnlySpan<sbyte> levels)
    {
        int kept = 0;
        foreach (sbyte level in levels)
        {
            if (level >= 0)
            {
                kept++;
            }
        }

        var indexes = new int[kept];
        var at = new sbyte[kept];
        int highest = 0, lowestOdd = int.MaxValue;
        for (int i = 0, k = 0; i < levels.Length; i++)
        {
            if (levels[i] < 0)
            {
                continue;
            }

            indexes[k] = i;
            at[k++] = levels[i];
            highest = Math.Max(highest, levels[i]);
            if ((levels[i] & 1) == 1)
            {
                lowestOdd = Math.Min(lowestOdd, levels[i]);
            }
        }

        Reverse(indexes, at, highest, lowestOdd);
        return indexes;
    }

    /// <summary>
    /// The resolved embedding level of every UTF-16 index of a paragraph (a character X9 removes takes the level before
    /// it), the paragraph's classes, and whether it is right to left.
    /// </summary>
    /// <param name="text">The paragraph (no paragraph separator inside).</param>
    /// <param name="rightToLeft">The paragraph's direction, or null for its first strong character's (left to right without one).</param>
    public static (byte[] Levels, BidiClass[] Classes, bool RightToLeft) Resolve(string text, bool? rightToLeft = null)
    {
        ArgumentNullException.ThrowIfNull(text);
        int n = text.Length;
        var points = new int[n];
        var starts = new int[n];
        int count = 0;
        for (int i = 0; i < n;)
        {
            points[count] = UnicodeText.CodePointAt(text, i, out int width);
            starts[count++] = i;
            i += width;
        }

        var types = new BidiClass[count];
        for (int k = 0; k < count; k++)
        {
            types[k] = UnicodeText.Bidi(points[k]);
        }

        var resolved = Levels(types, points.AsSpan(0, count), rightToLeft switch { true => 1, false => 0, null => -1 }, out int paragraph);
        var levels = new byte[n];
        var classes = new BidiClass[n];
        int previous = paragraph;
        for (int k = 0; k < count; k++)
        {
            int level = resolved[k] < 0 ? previous : resolved[k];
            previous = level;
            int end = k + 1 < count ? starts[k + 1] : n;
            for (int i = starts[k]; i < end; i++)
            {
                levels[i] = (byte)level;
                classes[i] = types[k];
            }
        }

        return (levels, classes, paragraph == 1);
    }

    /// <summary>
    /// The visual order of a line [start, end) of a resolved paragraph: L1 (separators, and trailing whitespace and controls,
    /// back to the paragraph level) then L2 (each level and above reversed, from the highest down to the lowest odd one).
    /// Returns the text indexes left to right, and the line's levels after L1 in text order.
    /// </summary>
    public static (int[] Order, byte[] Levels) Reorder(ReadOnlySpan<byte> levels, ReadOnlySpan<BidiClass> classes, int start, int end, bool rightToLeft)
    {
        if (classes.Length != levels.Length)
        {
            throw new ArgumentException($"{classes.Length} classes for {levels.Length} levels.", nameof(classes));
        }

        ArgumentOutOfRangeException.ThrowIfNegative(start);
        ArgumentOutOfRangeException.ThrowIfLessThan(end, start);
        ArgumentOutOfRangeException.ThrowIfGreaterThan(end, levels.Length);
        int n = end - start;
        var line = levels.Slice(start, n).ToArray();
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
            else if (IsWhitespaceOrIsolate(c) || Removed(c))
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

        int highest = 0, lowestOdd = int.MaxValue;
        foreach (byte level in line)
        {
            highest = Math.Max(highest, level);
            if ((level & 1) == 1)
            {
                lowestOdd = Math.Min(lowestOdd, level);
            }
        }

        Reverse(order, line, highest, lowestOdd);

        // The levels follow the characters in visual order now; give them back in text order.
        var textLevels = new byte[n];
        for (int i = 0; i < n; i++)
        {
            textLevels[order[i] - start] = line[i];
        }

        return (order, textLevels);
    }

    private static bool IsWhitespaceOrIsolate(BidiClass type) => type is BidiClass.WS or BidiClass.LRI or BidiClass.RLI or BidiClass.FSI or BidiClass.PDI;

    // L2 on indexes with their levels: from the highest level down to the lowest odd one, every run at or above it reversed.
    private static void Reverse<TLevel>(int[] indexes, TLevel[] at, int highest, int lowestOdd) where TLevel : struct, INumber<TLevel>
    {
        for (int level = highest; level >= lowestOdd; level--)
        {
            var threshold = TLevel.CreateChecked(level);
            for (int i = 0; i < at.Length; i++)
            {
                if (at[i] < threshold)
                {
                    continue;
                }

                int end = i;
                while (end + 1 < at.Length && at[end + 1] >= threshold)
                {
                    end++;
                }

                Array.Reverse(indexes, i, end - i + 1);
                Array.Reverse(at, i, end - i + 1);
                i = end;
            }
        }
    }

    // BD9: the matching PDI of each isolate initiator and the initiator of each matched PDI (-1 when none).
    private static void MatchIsolates(BidiClass[] types, out int[] matchingPdi, out int[] matchingInitiator)
    {
        matchingPdi = new int[types.Length];
        matchingInitiator = new int[types.Length];
        Array.Fill(matchingPdi, -1);
        Array.Fill(matchingInitiator, -1);
        var open = new Stack<int>();
        for (int i = 0; i < types.Length; i++)
        {
            if (types[i] is BidiClass.LRI or BidiClass.RLI or BidiClass.FSI)
            {
                open.Push(i);
            }
            else if (types[i] == BidiClass.PDI && open.Count > 0)
            {
                int start = open.Pop();
                matchingPdi[start] = i;
                matchingInitiator[i] = start;
            }
            else if (types[i] == BidiClass.B)
            {
                open.Clear();
            }
        }
    }

    // BD13 with X10: the isolating run sequences, each a list of character indexes in order (removed characters left out).
    private static List<List<int>> RunSequences(BidiClass[] types, sbyte[] levels, int[] matchingPdi, int[] matchingInitiator)
    {
        int n = types.Length;
        var runs = new List<List<int>>();
        var runAt = new int[n];                                                   // the run starting at a character, or -1
        Array.Fill(runAt, -1);
        List<int>? run = null;
        int runLevel = -1;
        for (int i = 0; i < n; i++)
        {
            if (Removed(types[i]))
            {
                continue;
            }

            if (run is null || levels[i] != runLevel)
            {
                run = [];
                runAt[i] = runs.Count;
                runs.Add(run);
                runLevel = levels[i];
            }

            run.Add(i);
        }

        var sequences = new List<List<int>>(runs.Count);
        foreach (var r in runs)
        {
            if (types[r[0]] == BidiClass.PDI && matchingInitiator[r[0]] >= 0)
            {
                continue;                                                        // continues an earlier sequence
            }

            var sequence = new List<int>(r);
            var last = r;
            while (types[last[^1]] is BidiClass.LRI or BidiClass.RLI or BidiClass.FSI && matchingPdi[last[^1]] is int pdi && pdi >= 0 && runAt[pdi] >= 0)
            {
                var next = runs[runAt[pdi]];
                sequence.AddRange(next);
                last = next;
            }

            sequences.Add(sequence);
        }

        return sequences;
    }

    private static bool IsStrongOrNumber(BidiClass type) => type is BidiClass.L or BidiClass.R or BidiClass.EN or BidiClass.AN;

    // N1, N2 and N0 see numbers as R.
    private static BidiClass AsStrong(BidiClass type) => type is BidiClass.EN or BidiClass.AN ? BidiClass.R : type;

    private static bool IsNeutralOrIsolate(BidiClass type) =>
        type is BidiClass.B or BidiClass.S or BidiClass.WS or BidiClass.ON or BidiClass.LRI or BidiClass.RLI or BidiClass.FSI or BidiClass.PDI;

    // W1-W7, N0-N2 and I1-I2 on one isolating run sequence.
    private static void ResolveSequence(List<int> sequence, BidiClass[] original, BidiClass[] result, sbyte[] explicitLevels, sbyte[] levels, ReadOnlySpan<int> codePoints,
        int paragraph)
    {
        int count = sequence.Count;
        int level = levels[sequence[0]];
        int Neighbour(int index, int step)
        {
            for (int i = index + step; i >= 0 && i < original.Length; i += step)
            {
                if (!Removed(original[i]))
                {
                    return i;
                }
            }

            return -1;
        }

        int before = Neighbour(sequence[0], -1);
        int last = sequence[^1];
        int after = original[last] is BidiClass.LRI or BidiClass.RLI or BidiClass.FSI ? -1 : Neighbour(last, 1);
        var sos = (Math.Max(level, before < 0 ? paragraph : explicitLevels[before]) & 1) == 1 ? BidiClass.R : BidiClass.L;
        var eos = (Math.Max(level, after < 0 ? paragraph : explicitLevels[after]) & 1) == 1 ? BidiClass.R : BidiClass.L;
        var t = new BidiClass[count];
        for (int i = 0; i < count; i++)
        {
            t[i] = result[sequence[i]];
        }

        // W1: nonspacing marks take the type before them (ON after an isolate control, sos at the start).
        for (int i = 0; i < count; i++)
        {
            if (t[i] == BidiClass.NSM)
            {
                t[i] = i == 0 ? sos : t[i - 1] is BidiClass.LRI or BidiClass.RLI or BidiClass.FSI or BidiClass.PDI ? BidiClass.ON : t[i - 1];
            }
        }

        // W2: European numbers after Arabic letters are Arabic numbers. W3: Arabic letters are R.
        var strong = sos;
        for (int i = 0; i < count; i++)
        {
            if (t[i] is BidiClass.L or BidiClass.R or BidiClass.AL)
            {
                strong = t[i];
            }
            else if (t[i] == BidiClass.EN && strong == BidiClass.AL)
            {
                t[i] = BidiClass.AN;
            }
        }

        for (int i = 0; i < count; i++)
        {
            if (t[i] == BidiClass.AL)
            {
                t[i] = BidiClass.R;
            }
        }

        // W4: one separator between two numbers of the same kind joins them.
        for (int i = 1; i < count - 1; i++)
        {
            if (t[i] == BidiClass.ES && t[i - 1] == BidiClass.EN && t[i + 1] == BidiClass.EN)
            {
                t[i] = BidiClass.EN;
            }
            else if (t[i] == BidiClass.CS && t[i - 1] is BidiClass.EN or BidiClass.AN && t[i + 1] == t[i - 1])
            {
                t[i] = t[i - 1];
            }
        }

        // W5: terminators next to European numbers become European numbers.
        for (int i = 0; i < count; i++)
        {
            if (t[i] != BidiClass.ET)
            {
                continue;
            }

            int end = i;
            while (end + 1 < count && t[end + 1] == BidiClass.ET)
            {
                end++;
            }

            bool number = i > 0 && t[i - 1] == BidiClass.EN || end + 1 < count && t[end + 1] == BidiClass.EN;
            if (number)
            {
                Array.Fill(t, BidiClass.EN, i, end - i + 1);
            }

            i = end;
        }

        // W6: the remaining separators and terminators are neutral.
        for (int i = 0; i < count; i++)
        {
            if (t[i] is BidiClass.ES or BidiClass.ET or BidiClass.CS)
            {
                t[i] = BidiClass.ON;
            }
        }

        // W7: European numbers after L (or sos L) are L.
        strong = sos;
        for (int i = 0; i < count; i++)
        {
            if (t[i] is BidiClass.L or BidiClass.R)
            {
                strong = t[i];
            }
            else if (t[i] == BidiClass.EN && strong == BidiClass.L)
            {
                t[i] = BidiClass.L;
            }
        }

        var embedding = (level & 1) == 1 ? BidiClass.R : BidiClass.L;
        if (!codePoints.IsEmpty)
        {
            ResolveBrackets(sequence, original, codePoints, t, sos, embedding);
        }

        // N1: neutrals between two strong types of one direction take it (numbers count as R); N2: else the embedding's.
        for (int i = 0; i < count; i++)
        {
            if (!IsNeutralOrIsolate(t[i]))
            {
                continue;
            }

            int end = i;
            while (end + 1 < count && IsNeutralOrIsolate(t[end + 1]))
            {
                end++;
            }

            var leading = i == 0 ? sos : AsStrong(t[i - 1]);
            var trailing = end + 1 == count ? eos : AsStrong(t[end + 1]);
            Array.Fill(t, leading == trailing ? leading : embedding, i, end - i + 1);
            i = end;
        }

        // I1, I2: implicit levels.
        for (int i = 0; i < count; i++)
        {
            int index = sequence[i];
            if ((levels[index] & 1) == 0)
            {
                levels[index] += (sbyte)(t[i] == BidiClass.R ? 1 : t[i] is BidiClass.AN or BidiClass.EN ? 2 : 0);
            }
            else if (t[i] is BidiClass.L or BidiClass.EN or BidiClass.AN)
            {
                levels[index] += 1;
            }
        }
    }

    // The two angle brackets with canonical equivalents (BD16: compared after canonical decomposition).
    private static int Canonical(int codePoint) => codePoint switch
    {
        0x2329 => 0x3008,
        0x232A => 0x3009,
        _ => codePoint,
    };

    // BD16 and N0: paired brackets take the direction of what they enclose or, failing that, of the context before them.
    private static void ResolveBrackets(List<int> sequence, BidiClass[] original, ReadOnlySpan<int> codePoints, BidiClass[] t, BidiClass sos, BidiClass embedding)
    {
        var pairs = new List<(int Open, int Close)>();
        var stack = new List<(int Pair, int Position)>();
        for (int i = 0; i < sequence.Count; i++)
        {
            if (t[i] != BidiClass.ON || !UnicodeText.Bracket(codePoints[sequence[i]], out int pair, out bool opening))
            {
                continue;
            }

            if (opening)
            {
                if (stack.Count == 63)
                {
                    break;
                }

                stack.Add((Canonical(pair), i));
            }
            else
            {
                int wanted = Canonical(codePoints[sequence[i]]);
                for (int s = stack.Count - 1; s >= 0; s--)
                {
                    if (stack[s].Pair == wanted)
                    {
                        pairs.Add((stack[s].Position, i));
                        stack.RemoveRange(s, stack.Count - s);
                        break;
                    }
                }
            }
        }

        pairs.Sort((a, b) => a.Open.CompareTo(b.Open));
        var opposite = embedding == BidiClass.L ? BidiClass.R : BidiClass.L;
        foreach (var (open, close) in pairs)
        {
            bool matching = false, other = false;
            for (int i = open + 1; i < close; i++)
            {
                if (!IsStrongOrNumber(t[i]))
                {
                    continue;
                }

                var direction = AsStrong(t[i]);
                matching |= direction == embedding;
                other |= direction == opposite;
            }

            BidiClass set;
            if (matching)
            {
                set = embedding;
            }
            else if (other)
            {
                var context = sos;
                for (int i = open - 1; i >= 0; i--)
                {
                    if (IsStrongOrNumber(t[i]))
                    {
                        context = AsStrong(t[i]);
                        break;
                    }
                }

                set = context == opposite ? opposite : embedding;
            }
            else
            {
                continue;
            }

            Set(open, set);
            Set(close, set);
        }

        // A bracket's new type, and that of the nonspacing marks after it.
        void Set(int bracket, BidiClass direction)
        {
            t[bracket] = direction;
            for (int i = bracket + 1; i < sequence.Count && original[sequence[i]] == BidiClass.NSM; i++)
            {
                t[i] = direction;
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
