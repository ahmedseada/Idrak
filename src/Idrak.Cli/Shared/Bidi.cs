// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

namespace Idrak.Cli.Shared;

/// <summary>The bidirectional character types of the Unicode Bidirectional Algorithm (UAX #9, table 4).</summary>
internal enum BidiClass : byte
{
    L, R, AL, EN, ES, ET, AN, CS, NSM, BN, B, S, WS, ON, LRE, LRO, RLE, RLO, PDF, LRI, RLI, FSI, PDI,
}

/// <summary>
/// The Unicode Bidirectional Algorithm (UAX #9, revision of Unicode <see cref="UnicodeTables.Version"/>) for one
/// paragraph: explicit embeddings, overrides and isolates (X1-X10), weak and neutral types with bracket pairs (W1-W7,
/// N0-N2), implicit levels (I1-I2), and the line rules (L1 levels, L2 reordering). L3 (combining marks after their
/// base) and L4 (mirrored glyphs) are left to the caller, as the standard leaves them to the renderer. Checked against
/// the Unicode conformance files BidiTest.txt and BidiCharacterTest.txt (plans/idrak-cli.md, "Arabic messages").
/// </summary>
internal static class Bidi
{
    private const int MaxDepth = 125;

    /// <summary>The bidirectional class of a code point.</summary>
    public static BidiClass ClassOf(int codePoint)
    {
        var table = UnicodeTables.BidiRanges;
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
                return (BidiClass)table[middle * 3 + 2];
            }
        }

        return BidiClass.L;
    }

    /// <summary>The mirrored glyph of a code point shown right to left (BidiMirroring.txt), or the code point itself.</summary>
    public static int Mirror(int codePoint) => Mirrors.TryGetValue(codePoint, out int mirror) ? mirror : codePoint;

    private static readonly Dictionary<int, int> Mirrors = Pairs(UnicodeTables.Mirrors, 2).ToDictionary(e => e[0], e => e[1]);

    private static readonly Dictionary<int, (int Pair, bool Opening)> Brackets =
        Pairs(UnicodeTables.Brackets, 3).ToDictionary(e => e[0], e => (e[1], e[2] == 1));

    // A flat table read as entries of the given size.
    private static IEnumerable<int[]> Pairs(int[] table, int size)
    {
        for (int i = 0; i < table.Length; i += size)
        {
            yield return table[i..(i + size)];
        }
    }

    // A paired bracket (BidiBrackets.txt): its pair and whether it opens; pair 0 when it is no bracket.
    private static (int Pair, bool Opening) Bracket(int codePoint) => Brackets.TryGetValue(codePoint, out var bracket) ? bracket : (0, false);

    // The two angle brackets with canonical equivalents (BD16: compared after canonical decomposition).
    private static int Canonical(int codePoint) => codePoint switch
    {
        0x2329 => 0x3008,
        0x232A => 0x3009,
        _ => codePoint,
    };

    /// <summary>Whether a class is removed by rule X9 (embedding and override controls, boundary neutrals).</summary>
    public static bool Removed(BidiClass type) => type is BidiClass.RLE or BidiClass.LRE or BidiClass.RLO or BidiClass.LRO or BidiClass.PDF or BidiClass.BN;

    /// <summary>
    /// The paragraph level by rules P2 and P3: 1 when the first strong character (outside isolates) is R or AL, 0 when
    /// it is L or there is none (<paramref name="fallback"/> then).
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
    /// The resolved embedding level of each character of one paragraph (after L1), or -1 for a character rule X9
    /// removes. <paramref name="codePoints"/> identifies paired brackets (rule N0); it may be empty when only classes
    /// are known. <paramref name="paragraphLevel"/> is 0, 1, or -1 to find it by P2 and P3.
    /// </summary>
    public static sbyte[] Levels(ReadOnlySpan<BidiClass> types, ReadOnlySpan<int> codePoints, int paragraphLevel, out int resolvedParagraphLevel)
    {
        int n = types.Length;
        int paragraph = paragraphLevel is 0 or 1 ? paragraphLevel : ParagraphLevel(types);
        resolvedParagraphLevel = paragraph;
        var original = types.ToArray();
        var result = types.ToArray();
        var levels = new sbyte[n];
        var matchingPdi = MatchIsolates(original, out var matchingInitiator);

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
            else if (type is BidiClass.WS or BidiClass.LRI or BidiClass.RLI or BidiClass.FSI or BidiClass.PDI || Removed(type))
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

    /// <summary>
    /// Rule L2: the indexes of the characters in visual order (left to right), leaving out those with level -1.
    /// </summary>
    public static int[] VisualOrder(ReadOnlySpan<sbyte> levels)
    {
        var order = new List<int>(levels.Length);
        var kept = new List<sbyte>(levels.Length);
        int highest = 0, lowestOdd = int.MaxValue;
        for (int i = 0; i < levels.Length; i++)
        {
            if (levels[i] < 0)
            {
                continue;
            }

            order.Add(i);
            kept.Add(levels[i]);
            highest = Math.Max(highest, levels[i]);
            if ((levels[i] & 1) == 1)
            {
                lowestOdd = Math.Min(lowestOdd, levels[i]);
            }
        }

        var indexes = order.ToArray();
        var at = kept.ToArray();
        for (int level = highest; level >= lowestOdd; level--)
        {
            for (int i = 0; i < at.Length; i++)
            {
                if (at[i] < level)
                {
                    continue;
                }

                int end = i;
                while (end + 1 < at.Length && at[end + 1] >= level)
                {
                    end++;
                }

                Array.Reverse(indexes, i, end - i + 1);
                Array.Reverse(at, i, end - i + 1);
                i = end;
            }
        }

        return indexes;
    }

    // BD9: the matching PDI of each isolate initiator and the initiator of each matched PDI (-1 when none).
    private static int[] MatchIsolates(BidiClass[] types, out int[] matchingInitiator)
    {
        var pdi = new int[types.Length];
        matchingInitiator = new int[types.Length];
        Array.Fill(pdi, -1);
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
                pdi[start] = i;
                matchingInitiator[i] = start;
            }
            else if (types[i] == BidiClass.B)
            {
                open.Clear();
            }
        }

        return pdi;
    }

    // BD13 with X10: the isolating run sequences, each a list of character indexes in order (removed characters left out).
    private static List<List<int>> RunSequences(BidiClass[] types, sbyte[] levels, int[] matchingPdi, int[] matchingInitiator)
    {
        var runs = new List<List<int>>();
        List<int>? run = null;
        int runLevel = -1;
        for (int i = 0; i < types.Length; i++)
        {
            if (Removed(types[i]))
            {
                continue;
            }

            if (run is null || levels[i] != runLevel)
            {
                run = [];
                runs.Add(run);
                runLevel = levels[i];
            }

            run.Add(i);
        }

        var runStartingAt = new Dictionary<int, List<int>>();
        foreach (var r in runs)
        {
            runStartingAt[r[0]] = r;
        }

        var sequences = new List<List<int>>();
        foreach (var r in runs)
        {
            if (types[r[0]] == BidiClass.PDI && matchingInitiator[r[0]] >= 0)
            {
                continue;                                                        // continues an earlier sequence
            }

            var sequence = new List<int>(r);
            var last = r;
            while (types[last[^1]] is BidiClass.LRI or BidiClass.RLI or BidiClass.FSI && matchingPdi[last[^1]] is int pdi && pdi >= 0
                   && runStartingAt.TryGetValue(pdi, out var next))
            {
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

    // BD16 and N0: paired brackets take the direction of what they enclose or, failing that, of the context before them.
    private static void ResolveBrackets(List<int> sequence, BidiClass[] original, ReadOnlySpan<int> codePoints, BidiClass[] t, BidiClass sos, BidiClass embedding)
    {
        var pairs = new List<(int Open, int Close)>();
        var stack = new List<(int Pair, int Position)>();
        for (int i = 0; i < sequence.Count; i++)
        {
            if (t[i] != BidiClass.ON)
            {
                continue;
            }

            int codePoint = codePoints[sequence[i]];
            var (pair, opening) = Bracket(codePoint);
            if (pair == 0)
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
                for (int s = stack.Count - 1; s >= 0; s--)
                {
                    if (stack[s].Pair == Canonical(codePoint))
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

            BidiClass? set = null;
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

            if (set is not { } direction2)
            {
                continue;
            }

            foreach (int bracket in (int[])[open, close])
            {
                t[bracket] = direction2;
                // Nonspacing marks after a bracket that changed take its new type.
                for (int i = bracket + 1; i < sequence.Count && original[sequence[i]] == BidiClass.NSM; i++)
                {
                    t[i] = direction2;
                }
            }
        }
    }
}
