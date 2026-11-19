// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Buffers;
using System.Runtime.InteropServices;

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
/// them to the renderer. <c>Levels</c> and <c>VisualOrder</c> work on code points and their classes (a terminal
/// renderer); <see cref="Resolve"/> and <c>Reorder</c> on a string's UTF-16 indexes, a line at a time (text layout). Each
/// of <c>Levels</c>, <c>VisualOrder</c> and <c>Reorder</c> has a form that writes into the caller's buffers
/// (<see cref="Levels(ReadOnlySpan{BidiClass}, ReadOnlySpan{int}, int, Span{sbyte})"/>,
/// <see cref="VisualOrder(ReadOnlySpan{sbyte}, Span{int})"/>,
/// <see cref="Reorder(ReadOnlySpan{byte}, ReadOnlySpan{BidiClass}, int, int, bool, Span{int}, Span{byte})"/>) and
/// allocates nothing, its working space taken from the shared array pools; the forms that return arrays
/// allocate those alone. Checked against every case of the Unicode conformance files BidiTest.txt and
/// BidiCharacterTest.txt of <see cref="UnicodeVersion"/> (the test group "bidi:", which reads them from the repository's
/// test data), through both forms.
/// </summary>
public static class Bidi
{
    private const int MaxDepth = 125;

    // X1: the entries of the directional status stack, the paragraph's own and one for each level up to MaxDepth.
    private const int StackEntries = MaxDepth + 2;

    // BD16: the depth of the bracket stack; an opening bracket past it ends the pairing in its isolating run sequence.
    private const int MaxBracketDepth = 63;

    // The working space Levels takes from the shared pools, per character. Ints: the isolates' matches both ways, the
    // level runs (the run starting at a character, each run's first and last character), the isolating run sequence being
    // resolved and its brackets' pairs. Bytes: the classes after X1-X8, the explicit levels, the sequence's classes. After
    // them, the two fixed stacks: the bracket stack (a bracket and its place, ints) and the directional status stack (a
    // level, an override and an isolate flag, bytes). They are pooled, not stackalloc'ed: a method with a loop and
    // stackalloc cannot start in the tiered JIT's quick code, so it is compiled fully optimized at its first call (a few
    // milliseconds more on a process's first line).
    private const int IntsPerCharacter = 7;
    private const int BytesPerCharacter = 3;
    private const int BracketStackInts = 2 * MaxBracketDepth;
    private const int StatusStackBytes = 3 * StackEntries;

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
    /// It allocates the array it returns alone (<see cref="Levels(ReadOnlySpan{BidiClass}, ReadOnlySpan{int}, int, Span{sbyte})"/>
    /// writes into the caller's).
    /// </summary>
    public static sbyte[] Levels(ReadOnlySpan<BidiClass> types, ReadOnlySpan<int> codePoints, int paragraphLevel, out int resolvedParagraphLevel)
    {
        CheckParagraph(types, codePoints, paragraphLevel);
        var levels = new sbyte[types.Length];
        resolvedParagraphLevel = Levels(types, codePoints, paragraphLevel, levels);
        return levels;
    }

    /// <summary>
    /// The resolved embedding level of each character of one paragraph (after L1), or -1 for a character rule X9 removes,
    /// written into the first <c>types.Length</c> entries of <paramref name="levels"/>; returns the paragraph level.
    /// <paramref name="codePoints"/> identifies paired brackets (rule N0); it may be empty when only classes are known.
    /// <paramref name="paragraphLevel"/> is 0, 1, or -1 to find it by P2 and P3. Nothing is allocated: the working space
    /// comes from the shared array pools (<see cref="ArrayPool{T}.Shared"/>).
    /// </summary>
    /// <exception cref="ArgumentException"><paramref name="codePoints"/> is neither empty nor as long as <paramref name="types"/>, or <paramref name="levels"/> is shorter than it.</exception>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="paragraphLevel"/> is not -1, 0 or 1.</exception>
    public static int Levels(ReadOnlySpan<BidiClass> types, ReadOnlySpan<int> codePoints, int paragraphLevel, Span<sbyte> levels)
    {
        CheckParagraph(types, codePoints, paragraphLevel);
        int n = types.Length;
        if (levels.Length < n)
        {
            throw new ArgumentException($"{levels.Length} levels for {n} classes.", nameof(levels));
        }

        int paragraph = paragraphLevel is 0 or 1 ? paragraphLevel : ParagraphLevel(types);
        if (n == 0)
        {
            return paragraph;
        }

        levels = levels[..n];
        int[] ints = ArrayPool<int>.Shared.Rent(IntsPerCharacter * n + BracketStackInts);
        byte[] bytes = ArrayPool<byte>.Shared.Rent(BytesPerCharacter * n + StatusStackBytes);
        try
        {
            var matchingPdi = ints.AsSpan(0, n);
            var matchingInitiator = ints.AsSpan(n, n);
            var runAt = ints.AsSpan(2 * n, n);
            var runFirst = ints.AsSpan(3 * n, n);
            var runLast = ints.AsSpan(4 * n, n);
            var sequence = ints.AsSpan(5 * n, n);
            var pairs = ints.AsSpan(6 * n, n);
            var bracketStack = ints.AsSpan(7 * n, BracketStackInts);
            var result = MemoryMarshal.Cast<byte, BidiClass>(bytes.AsSpan(0, n));
            var explicitLevels = MemoryMarshal.Cast<byte, sbyte>(bytes.AsSpan(n, n));
            var sequenceTypes = MemoryMarshal.Cast<byte, BidiClass>(bytes.AsSpan(2 * n, n));
            var statusStack = bytes.AsSpan(3 * n, StatusStackBytes);
            types.CopyTo(result);
            MatchIsolates(types, matchingPdi, matchingInitiator, sequence);      // the sequence's space holds the open isolates meanwhile
            Explicit(types, result, levels, matchingPdi, paragraph, statusStack);

            // X9 removes the embedding controls and boundary neutrals; X10 splits the rest into isolating run sequences,
            // each resolved as soon as it is gathered. Each sequence's sos and eos come from the explicit levels, before
            // the implicit rules raise any of them.
            levels.CopyTo(explicitLevels);
            int runs = LevelRuns(types, explicitLevels, runAt, runFirst, runLast);
            for (int r = 0; r < runs; r++)
            {
                if (types[runFirst[r]] == BidiClass.PDI && matchingInitiator[runFirst[r]] >= 0)
                {
                    continue;                                                    // continues an earlier sequence
                }

                // The run, then the run its last character's matching PDI starts, while the run ends with an isolate initiator.
                int count = 0, run = r;
                while (true)
                {
                    count = AddRun(types, runFirst[run], runLast[run], sequence, count);
                    int last = runLast[run];
                    if (types[last] is not (BidiClass.LRI or BidiClass.RLI or BidiClass.FSI) || matchingPdi[last] < 0 || runAt[matchingPdi[last]] < 0)
                    {
                        break;
                    }

                    run = runAt[matchingPdi[last]];
                }

                ResolveSequence(sequence[..count], types, result, explicitLevels, levels, codePoints, paragraph, sequenceTypes[..count], pairs[..count], bracketStack);
            }

            // L1: separators, and whitespace and isolate controls before them or at the end of the line, to the paragraph level.
            bool trailing = true;
            for (int i = n - 1; i >= 0; i--)
            {
                var type = types[i];
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
                if (Removed(types[i]))
                {
                    levels[i] = -1;
                }
            }
        }
        finally
        {
            ArrayPool<byte>.Shared.Return(bytes);
            ArrayPool<int>.Shared.Return(ints);
        }

        return paragraph;
    }

    /// <summary>
    /// Rule L2: the indexes of the characters in visual order (left to right), leaving out those with level -1. It
    /// allocates the array it returns alone (<see cref="VisualOrder(ReadOnlySpan{sbyte}, Span{int})"/> writes into the caller's).
    /// </summary>
    public static int[] VisualOrder(ReadOnlySpan<sbyte> levels)
    {
        var order = new int[Kept(levels)];
        VisualOrder(levels, order);
        return order;
    }

    /// <summary>
    /// Rule L2 into <paramref name="order"/>: the indexes of the characters in visual order (left to right), leaving out
    /// those with level -1; returns how many there are. <paramref name="order"/> holds at least that many (as many as
    /// <paramref name="levels"/> always does). Nothing is allocated.
    /// </summary>
    /// <exception cref="ArgumentException"><paramref name="order"/> is shorter than the number of characters kept.</exception>
    public static int VisualOrder(ReadOnlySpan<sbyte> levels, Span<int> order)
    {
        int kept = Kept(levels);
        if (order.Length < kept)
        {
            throw new ArgumentException($"{order.Length} places for {kept} characters.", nameof(order));
        }

        order = order[..kept];
        int highest = 0, lowestOdd = int.MaxValue;
        for (int i = 0, k = 0; i < levels.Length; i++)
        {
            int level = levels[i];
            if (level < 0)
            {
                continue;
            }

            order[k++] = i;
            highest = Math.Max(highest, level);
            if ((level & 1) == 1)
            {
                lowestOdd = Math.Min(lowestOdd, level);
            }
        }

        Reverse(order, MemoryMarshal.Cast<sbyte, byte>(levels), 0, highest, lowestOdd);      // the kept characters' levels are 0 or more: the same as bytes
        return kept;
    }

    /// <summary>
    /// The resolved embedding level of every UTF-16 index of a paragraph (a character X9 removes takes the level before
    /// it), the paragraph's classes, and whether it is right to left. It allocates the two arrays it returns alone.
    /// </summary>
    /// <param name="text">The paragraph (no paragraph separator inside).</param>
    /// <param name="rightToLeft">The paragraph's direction, or null for its first strong character's (left to right without one).</param>
    public static (byte[] Levels, BidiClass[] Classes, bool RightToLeft) Resolve(string text, bool? rightToLeft = null)
    {
        ArgumentNullException.ThrowIfNull(text);
        int n = text.Length;
        var levels = new byte[n];
        var classes = new BidiClass[n];
        int[] ints = ArrayPool<int>.Shared.Rent(2 * n);
        byte[] bytes = ArrayPool<byte>.Shared.Rent(2 * n);
        try
        {
            // The code points with their starts, their classes, and their levels as Levels resolves them.
            var points = ints.AsSpan(0, n);
            var starts = ints.AsSpan(n, n);
            int count = 0;
            for (int i = 0; i < n;)
            {
                points[count] = UnicodeText.CodePointAt(text, i, out int width);
                starts[count++] = i;
                i += width;
            }

            var types = MemoryMarshal.Cast<byte, BidiClass>(bytes.AsSpan(0, count));
            var resolved = MemoryMarshal.Cast<byte, sbyte>(bytes.AsSpan(n, count));
            for (int k = 0; k < count; k++)
            {
                types[k] = UnicodeText.Bidi(points[k]);
            }

            int paragraph = Levels(types, points[..count], rightToLeft switch { true => 1, false => 0, null => -1 }, resolved);
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
        finally
        {
            ArrayPool<byte>.Shared.Return(bytes);
            ArrayPool<int>.Shared.Return(ints);
        }
    }

    /// <summary>
    /// The visual order of a line [start, end) of a resolved paragraph: L1 (separators, and trailing whitespace and controls,
    /// back to the paragraph level) then L2 (each level and above reversed, from the highest down to the lowest odd one).
    /// Returns the text indexes left to right, and the line's levels after L1 in text order. It allocates the two arrays it
    /// returns alone (<see cref="Reorder(ReadOnlySpan{byte}, ReadOnlySpan{BidiClass}, int, int, bool, Span{int}, Span{byte})"/>
    /// writes into the caller's).
    /// </summary>
    public static (int[] Order, byte[] Levels) Reorder(ReadOnlySpan<byte> levels, ReadOnlySpan<BidiClass> classes, int start, int end, bool rightToLeft)
    {
        CheckLine(levels, classes, start, end);
        var order = new int[end - start];
        var lineLevels = new byte[end - start];
        Reorder(levels, classes, start, end, rightToLeft, order, lineLevels);
        return (order, lineLevels);
    }

    /// <summary>
    /// The visual order of a line [start, end) of a resolved paragraph, as
    /// <see cref="Reorder(ReadOnlySpan{byte}, ReadOnlySpan{BidiClass}, int, int, bool)"/> gives it, into the caller's
    /// buffers: the text indexes left to right into <paramref name="order"/>, and the line's levels after L1 in text order
    /// into <paramref name="lineLevels"/>, each written for its first <c>end - start</c> entries. An empty
    /// <paramref name="order"/> leaves L2 out (the levels alone). <paramref name="lineLevels"/> may be the same memory as
    /// <paramref name="levels"/>' [start, end), changed in place. Nothing is allocated.
    /// </summary>
    /// <exception cref="ArgumentException">The classes and levels differ in length, or a buffer is shorter than the line.</exception>
    /// <exception cref="ArgumentOutOfRangeException">[start, end) is not within the paragraph.</exception>
    public static void Reorder(ReadOnlySpan<byte> levels, ReadOnlySpan<BidiClass> classes, int start, int end, bool rightToLeft, Span<int> order, Span<byte> lineLevels)
    {
        CheckLine(levels, classes, start, end);
        int n = end - start;
        if (lineLevels.Length < n)
        {
            throw new ArgumentException($"{lineLevels.Length} levels for a line of {n}.", nameof(lineLevels));
        }

        if (!order.IsEmpty && order.Length < n)
        {
            throw new ArgumentException($"{order.Length} places for a line of {n}.", nameof(order));
        }

        var line = lineLevels[..n];
        levels.Slice(start, n).CopyTo(line);
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

        if (order.IsEmpty)
        {
            return;
        }

        order = order[..n];
        int highest = 0, lowestOdd = int.MaxValue;
        for (int i = 0; i < n; i++)
        {
            order[i] = start + i;
            highest = Math.Max(highest, line[i]);
            if ((line[i] & 1) == 1)
            {
                lowestOdd = Math.Min(lowestOdd, line[i]);
            }
        }

        Reverse(order, line, start, highest, lowestOdd);
    }

    // What every form of Levels checks first.
    private static void CheckParagraph(ReadOnlySpan<BidiClass> types, ReadOnlySpan<int> codePoints, int paragraphLevel)
    {
        if (!codePoints.IsEmpty && codePoints.Length != types.Length)
        {
            throw new ArgumentException($"{codePoints.Length} code points for {types.Length} classes.", nameof(codePoints));
        }

        if (paragraphLevel is < -1 or > 1)
        {
            throw new ArgumentOutOfRangeException(nameof(paragraphLevel), paragraphLevel, "The paragraph level is 0, 1, or -1 to find it.");
        }
    }

    // What every form of Reorder checks first.
    private static void CheckLine(ReadOnlySpan<byte> levels, ReadOnlySpan<BidiClass> classes, int start, int end)
    {
        if (classes.Length != levels.Length)
        {
            throw new ArgumentException($"{classes.Length} classes for {levels.Length} levels.", nameof(classes));
        }

        ArgumentOutOfRangeException.ThrowIfNegative(start);
        ArgumentOutOfRangeException.ThrowIfLessThan(end, start);
        ArgumentOutOfRangeException.ThrowIfGreaterThan(end, levels.Length);
    }

    // The characters X9 keeps (level 0 or more).
    private static int Kept(ReadOnlySpan<sbyte> levels)
    {
        int kept = 0;
        foreach (sbyte level in levels)
        {
            if (level >= 0)
            {
                kept++;
            }
        }

        return kept;
    }

    private static bool IsWhitespaceOrIsolate(BidiClass type) => type is BidiClass.WS or BidiClass.LRI or BidiClass.RLI or BidiClass.FSI or BidiClass.PDI;

    // L2 on the places of `order`: from the highest level down to the lowest odd one, every run of places at or above it
    // reversed. A place's level is that of the index it holds, levels[order[i] - offset], so the levels stay in text order.
    private static void Reverse(Span<int> order, ReadOnlySpan<byte> levels, int offset, int highest, int lowestOdd)
    {
        for (int level = highest; level >= lowestOdd; level--)
        {
            for (int i = 0; i < order.Length; i++)
            {
                if (levels[order[i] - offset] < level)
                {
                    continue;
                }

                int end = i;
                while (end + 1 < order.Length && levels[order[end + 1] - offset] >= level)
                {
                    end++;
                }

                order.Slice(i, end - i + 1).Reverse();
                i = end;
            }
        }
    }

    // BD9: the matching PDI of each isolate initiator and the initiator of each matched PDI (-1 when none); `open` holds
    // the initiators not matched yet (as many as the characters at most).
    private static void MatchIsolates(ReadOnlySpan<BidiClass> types, Span<int> matchingPdi, Span<int> matchingInitiator, Span<int> open)
    {
        matchingPdi.Fill(-1);
        matchingInitiator.Fill(-1);
        int depth = 0;
        for (int i = 0; i < types.Length; i++)
        {
            if (types[i] is BidiClass.LRI or BidiClass.RLI or BidiClass.FSI)
            {
                open[depth++] = i;
            }
            else if (types[i] == BidiClass.PDI && depth > 0)
            {
                int start = open[--depth];
                matchingPdi[start] = i;
                matchingInitiator[i] = start;
            }
            else if (types[i] == BidiClass.B)
            {
                depth = 0;
            }
        }
    }

    // X1-X8: each character's explicit level into `levels`, and an override's direction into `result`. `status` holds the
    // directional status stack (StatusStackBytes: StackEntries entries, the most X1 allows, of a level, an override and an
    // isolate flag).
    private static void Explicit(ReadOnlySpan<BidiClass> original, Span<BidiClass> result, Span<sbyte> levels, ReadOnlySpan<int> matchingPdi, int paragraph, Span<byte> status)
    {
        int n = original.Length;
        status.Clear();
        var stackLevel = status[..StackEntries];
        var stackOverride = MemoryMarshal.Cast<byte, BidiClass>(status.Slice(StackEntries, StackEntries));
        var stackIsolate = status.Slice(2 * StackEntries, StackEntries);       // 1 for an isolate's entry
        int depth = 0;
        stackLevel[0] = (byte)paragraph;
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
                        stackLevel[depth] = (byte)level;
                        stackOverride[depth] = type == BidiClass.RLO ? BidiClass.R : type == BidiClass.LRO ? BidiClass.L : BidiClass.ON;
                        stackIsolate[depth] = 0;
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
                        rtl = ParagraphLevel(original.Slice(i + 1, end - i - 1)) == 1;
                    }

                    int level = rtl ? (stackLevel[depth] + 1) | 1 : (stackLevel[depth] + 2) & ~1;
                    if (level <= MaxDepth && overflowIsolates == 0 && overflowEmbeddings == 0)
                    {
                        validIsolates++;
                        depth++;
                        stackLevel[depth] = (byte)level;
                        stackOverride[depth] = BidiClass.ON;
                        stackIsolate[depth] = 1;
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
                        while (stackIsolate[depth] == 0)
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
                        else if (stackIsolate[depth] == 0 && depth > 0)
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
    }

    // BD7 with X9: the level runs of the characters X9 keeps, each by its first and last character, and the run that starts
    // at each character (`runAt`, -1 where none does). Returns the number of runs.
    private static int LevelRuns(ReadOnlySpan<BidiClass> types, ReadOnlySpan<sbyte> levels, Span<int> runAt, Span<int> first, Span<int> last)
    {
        runAt.Fill(-1);
        int runs = 0, runLevel = -1;
        for (int i = 0; i < types.Length; i++)
        {
            if (Removed(types[i]))
            {
                continue;
            }

            if (runs == 0 || levels[i] != runLevel)
            {
                runAt[i] = runs;
                first[runs++] = i;
                runLevel = levels[i];
            }

            last[runs - 1] = i;
        }

        return runs;
    }

    // The characters of [first, last] that X9 keeps, appended to `sequence` after its first `count`; returns the new count.
    private static int AddRun(ReadOnlySpan<BidiClass> types, int first, int last, Span<int> sequence, int count)
    {
        for (int i = first; i <= last; i++)
        {
            if (!Removed(types[i]))
            {
                sequence[count++] = i;
            }
        }

        return count;
    }

    // The nearest character X9 keeps from `index` going `step` (-1 or 1), or -1.
    private static int Neighbour(ReadOnlySpan<BidiClass> original, int index, int step)
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

    private static bool IsStrongOrNumber(BidiClass type) => type is BidiClass.L or BidiClass.R or BidiClass.EN or BidiClass.AN;

    // N1, N2 and N0 see numbers as R.
    private static BidiClass AsStrong(BidiClass type) => type is BidiClass.EN or BidiClass.AN ? BidiClass.R : type;

    private static bool IsNeutralOrIsolate(BidiClass type) =>
        type is BidiClass.B or BidiClass.S or BidiClass.WS or BidiClass.ON or BidiClass.LRI or BidiClass.RLI or BidiClass.FSI or BidiClass.PDI;

    // W1-W7, N0-N2 and I1-I2 on one isolating run sequence (its characters' indexes in order). `t` (the sequence's classes
    // as the rules change them) and `pairs` (its brackets' pairs) are working space as long as the sequence; `bracketStack`
    // the bracket stack's (BracketStackInts).
    private static void ResolveSequence(ReadOnlySpan<int> sequence, ReadOnlySpan<BidiClass> original, ReadOnlySpan<BidiClass> result, ReadOnlySpan<sbyte> explicitLevels,
        Span<sbyte> levels, ReadOnlySpan<int> codePoints, int paragraph, Span<BidiClass> t, Span<int> pairs, Span<int> bracketStack)
    {
        int count = sequence.Length;
        int level = levels[sequence[0]];
        int before = Neighbour(original, sequence[0], -1);
        int last = sequence[^1];
        int after = original[last] is BidiClass.LRI or BidiClass.RLI or BidiClass.FSI ? -1 : Neighbour(original, last, 1);
        var sos = (Math.Max(level, before < 0 ? paragraph : explicitLevels[before]) & 1) == 1 ? BidiClass.R : BidiClass.L;
        var eos = (Math.Max(level, after < 0 ? paragraph : explicitLevels[after]) & 1) == 1 ? BidiClass.R : BidiClass.L;
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
                t.Slice(i, end - i + 1).Fill(BidiClass.EN);
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
            ResolveBrackets(sequence, original, codePoints, t, sos, embedding, pairs, bracketStack);
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
            t.Slice(i, end - i + 1).Fill(leading == trailing ? leading : embedding);
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
    // `pairs` gets each opening bracket's closing one (-1 where none), so the pairs are taken in the order of their opening
    // brackets. `stack` holds the bracket stack (BracketStackInts: MaxBracketDepth entries of a bracket and its place).
    private static void ResolveBrackets(ReadOnlySpan<int> sequence, ReadOnlySpan<BidiClass> original, ReadOnlySpan<int> codePoints, Span<BidiClass> t, BidiClass sos,
        BidiClass embedding, Span<int> pairs, Span<int> stack)
    {
        var stackPair = stack[..MaxBracketDepth];
        var stackPosition = stack.Slice(MaxBracketDepth, MaxBracketDepth);
        int depth = 0;
        pairs.Fill(-1);
        for (int i = 0; i < sequence.Length; i++)
        {
            if (t[i] != BidiClass.ON || !UnicodeText.Bracket(codePoints[sequence[i]], out int pair, out bool opening))
            {
                continue;
            }

            if (opening)
            {
                if (depth == MaxBracketDepth)
                {
                    break;
                }

                stackPair[depth] = Canonical(pair);
                stackPosition[depth++] = i;
            }
            else
            {
                int wanted = Canonical(codePoints[sequence[i]]);
                for (int s = depth - 1; s >= 0; s--)
                {
                    if (stackPair[s] == wanted)
                    {
                        pairs[stackPosition[s]] = i;
                        depth = s;
                        break;
                    }
                }
            }
        }

        var opposite = embedding == BidiClass.L ? BidiClass.R : BidiClass.L;
        for (int open = 0; open < sequence.Length; open++)
        {
            int close = pairs[open];
            if (close < 0)
            {
                continue;
            }

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

            SetBracket(t, sequence, original, open, set);
            SetBracket(t, sequence, original, close, set);
        }
    }

    // A bracket's new type, and that of the nonspacing marks after it.
    private static void SetBracket(Span<BidiClass> t, ReadOnlySpan<int> sequence, ReadOnlySpan<BidiClass> original, int bracket, BidiClass direction)
    {
        t[bracket] = direction;
        for (int i = bracket + 1; i < sequence.Length && original[sequence[i]] == BidiClass.NSM; i++)
        {
            t[i] = direction;
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
