// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Globalization;
using System.Numerics;
using System.Text;
using Idrak.Imaging;
using Idrak.Imaging.Shaping;

// The pooled path through UAX #9 (plan 28, "Bidi pooled", 2026-11-19): Bidi's span forms, its array forms and the bidi
// reading order against the algorithm as it was before (kept at the end of this file word for word, its arrays, lists and
// LINQ and all), on random mixed texts from a fixed seed; and what the forms allocate once warm.
internal abstract partial class DesktopTests
{
    // The span forms of Levels and VisualOrder, into buffers longer than the paragraph that hold other values first: the
    // levels, the order, the paragraph level, and whether the entries past what they write were left as they were.
    private static (sbyte[] Levels, int[] Order, int Paragraph, bool Untouched) BidiSpanForms(ReadOnlySpan<BidiClass> types, ReadOnlySpan<int> codePoints, int paragraphLevel)
    {
        const sbyte OtherLevel = 99;
        const int OtherPlace = -7, Extra = 3;
        var levels = new sbyte[types.Length + Extra];
        var order = new int[types.Length + Extra];
        levels.AsSpan().Fill(OtherLevel);
        order.AsSpan().Fill(OtherPlace);
        int paragraph = Bidi.Levels(types, codePoints, paragraphLevel, levels);
        int kept = Bidi.VisualOrder(levels.AsSpan(0, types.Length), order);
        bool untouched = !levels.AsSpan(types.Length).ContainsAnyExcept(OtherLevel) && !order.AsSpan(kept).ContainsAnyExcept(OtherPlace);
        return (levels[..types.Length], order[..kept], paragraph, untouched);
    }

    // Code points of each kind for random mixed texts: L (with a supplementary letter and LRM), R (with supplementary
    // letters and RLM), AL (with ALM), EN (European and extended Arabic-Indic digits), AN (Arabic-Indic digits and their
    // separators), NSM, separators and terminators, whitespace, tabs and other neutrals, brackets (the two with canonical
    // equivalents too), the explicit controls and boundary neutrals, and lone surrogates.
    private static readonly int[][] BidiRandomKinds =
    [
        [.. Enumerable.Range('a', 26), .. Enumerable.Range('A', 26), 0x1D400, 0x200E],
        [.. Enumerable.Range(0x05D0, 27), 0x10800, 0x1E900, 0x200F],
        [.. Enumerable.Range(0x0627, 20), .. Enumerable.Range(0x0641, 10), 0x061C],
        [.. Enumerable.Range('0', 10), .. Enumerable.Range(0x06F0, 10)],
        [.. Enumerable.Range(0x0660, 10), 0x066B, 0x066C],
        [.. Enumerable.Range(0x064B, 8), 0x0300, 0x0301],
        ['+', '-', '$', '%', '#', 0x00B0, ',', '.', ':', '/', 0x00A0],
        [' ', ' ', ' ', '\t', '!', '"', '&', '*', ';', '?', '@', 0x2029],
        ['(', ')', '[', ']', '{', '}', 0x2329, 0x232A, 0x3008, 0x3009],
        [0x202A, 0x202B, 0x202C, 0x202D, 0x202E, 0x2066, 0x2067, 0x2068, 0x2069, 0x200B, 0x00AD],
        [0xD800, 0xDC00],
    ];

    private static readonly int[] BidiRandomWeights = [6, 4, 5, 4, 2, 2, 3, 5, 4, 3, 1];

    // Pairs repeated past the algorithm's limits now and then: embeddings and overrides past MaxDepth (125), isolates past
    // it, and opening brackets past the bracket stack (63).
    private static readonly (int A, int B)[] BidiRandomDeep = [(0x202B, 0x202A), (0x202E, 0x202D), (0x2067, 0x2066), (0x2068, 0x2068), ('(', '['), (0x2066, 0x202B)];

    // A random mixed text as code points: mostly short, one in ten long (100-320), one in 400 positions a deep run.
    private static int[] BidiRandomPoints(Random random)
    {
        int length = random.Next(10) == 0 ? random.Next(100, 320) : random.Next(0, 40);
        int total = BidiRandomWeights.Sum();
        var points = new List<int>(length);
        for (int i = 0; i < length; i++)
        {
            if (random.Next(400) == 0)
            {
                var (a, b) = BidiRandomDeep[random.Next(BidiRandomDeep.Length)];
                for (int k = random.Next(60, 140); k > 0; k--)
                {
                    points.Add(k % 2 == 0 ? a : b);
                }

                continue;
            }

            int pick = random.Next(total), kind = 0;
            while (pick >= BidiRandomWeights[kind])
            {
                pick -= BidiRandomWeights[kind++];
            }

            points.Add(BidiRandomKinds[kind][random.Next(BidiRandomKinds[kind].Length)]);
        }

        return [.. points];
    }

    // The code points as a string (a lone surrogate as its own UTF-16 unit).
    private static string BidiRandomString(int[] points)
    {
        var text = new StringBuilder(points.Length);
        foreach (int point in points)
        {
            if (point is >= 0xD800 and <= 0xDFFF)
            {
                text.Append((char)point);
            }
            else
            {
                text.Append(char.ConvertFromUtf32(point));
            }
        }

        return text.ToString();
    }

    private static string BidiShow(ReadOnlySpan<int> points)
    {
        var text = new StringBuilder();
        foreach (int point in points)
        {
            text.Append(point.ToString("X4", CultureInfo.InvariantCulture)).Append(' ');
        }

        return text.ToString().TrimEnd();
    }

    private static void BidiPooledAgainstReference()
    {
        const int Texts = 20_000;
        var random = new Random(20261119);
        var order = ReadingOrders.Default(ReadingOrders.Bidi) ?? throw new InvalidOperationException("the library registers no bidi reading order");
        int compared = 0, failed = 0;
        var first = new List<string>();
        void Expect(bool same, Func<string> what)
        {
            compared++;
            if (!same)
            {
                failed++;
                if (first.Count < 5)
                {
                    first.Add(what());
                }
            }
        }

        for (int t = 0; t < Texts; t++)
        {
            int[] points = BidiRandomPoints(random);
            BidiClass[] classes = [.. points.Select(Bidi.ClassOf)];
            string text = BidiRandomString(points);

            // Levels and VisualOrder at each paragraph level, with the code points (brackets paired) and without.
            foreach (int paragraph in (int[])[-1, 0, 1])
            {
                foreach (bool brackets in (bool[])[true, false])
                {
                    ReadOnlySpan<int> codePoints = brackets ? points : [];
                    var expected = ReferenceBidi.Levels(classes, codePoints, paragraph, out int expectedParagraph);
                    var expectedOrder = ReferenceBidi.VisualOrder(expected);
                    var array = Bidi.Levels(classes, codePoints, paragraph, out int arrayParagraph);
                    var arrayOrder = Bidi.VisualOrder(array);
                    var spans = BidiSpanForms(classes, codePoints, paragraph);
                    Expect(array.AsSpan().SequenceEqual(expected) && arrayParagraph == expectedParagraph && arrayOrder.AsSpan().SequenceEqual(expectedOrder)
                           && spans.Levels.AsSpan().SequenceEqual(expected) && spans.Paragraph == expectedParagraph && spans.Order.AsSpan().SequenceEqual(expectedOrder) && spans.Untouched,
                        () => $"Levels of {BidiShow(points)} at paragraph level {paragraph}{(brackets ? "" : " (classes alone)")}: array {string.Join(' ', array)} "
                              + $"(paragraph {arrayParagraph}), span {string.Join(' ', spans.Levels)} (paragraph {spans.Paragraph}, the rest untouched {spans.Untouched}), "
                              + $"the reference {string.Join(' ', expected)} (paragraph {expectedParagraph}); orders {string.Join(' ', arrayOrder)} / {string.Join(' ', spans.Order)} / {string.Join(' ', expectedOrder)}");
                }
            }

            // Resolve, then Reorder of the whole paragraph and of a random line of it: the array form, the span form into
            // buffers holding other values, the span form in place over the levels, and L1 alone.
            foreach (bool? direction in (bool?[])[null, false, true])
            {
                var (levels, kinds, rtl) = Bidi.Resolve(text, direction);
                var (referenceLevels, referenceKinds, referenceRtl) = ReferenceBidi.Resolve(text, direction);
                Expect(levels.AsSpan().SequenceEqual(referenceLevels) && kinds.AsSpan().SequenceEqual(referenceKinds) && rtl == referenceRtl,
                    () => $"Resolve of {BidiShow(points)} ({direction?.ToString() ?? "found"}): {string.Join(' ', levels)}, the reference {string.Join(' ', referenceLevels)}");
                int start = random.Next(text.Length + 1), end = random.Next(start, text.Length + 1);
                foreach (var (s, e) in (ReadOnlySpan<(int, int)>)[(0, text.Length), (start, end)])
                {
                    var (expectedOrder, expectedLine) = ReferenceBidi.Reorder(referenceLevels, referenceKinds, s, e, referenceRtl);
                    var (arrayOrder, arrayLine) = Bidi.Reorder(levels, kinds, s, e, rtl);
                    var spanOrder = new int[e - s + 2];
                    var spanLine = new byte[e - s + 2];
                    spanOrder.AsSpan().Fill(-7);
                    spanLine.AsSpan().Fill(200);
                    Bidi.Reorder(levels, kinds, s, e, rtl, spanOrder, spanLine);
                    var inPlace = (byte[])levels.Clone();
                    var inPlaceOrder = new int[e - s];
                    Bidi.Reorder(inPlace, kinds, s, e, rtl, inPlaceOrder, inPlace.AsSpan(s, e - s));
                    var alone = new byte[e - s];
                    Bidi.Reorder(levels, kinds, s, e, rtl, Span<int>.Empty, alone);
                    Expect(arrayOrder.AsSpan().SequenceEqual(expectedOrder) && arrayLine.AsSpan().SequenceEqual(expectedLine)
                           && spanOrder.AsSpan(0, e - s).SequenceEqual(expectedOrder) && spanLine.AsSpan(0, e - s).SequenceEqual(expectedLine)
                           && !spanOrder.AsSpan(e - s).ContainsAnyExcept(-7) && !spanLine.AsSpan(e - s).ContainsAnyExcept((byte)200)
                           && inPlaceOrder.AsSpan().SequenceEqual(expectedOrder) && inPlace.AsSpan(s, e - s).SequenceEqual(expectedLine) && alone.AsSpan().SequenceEqual(expectedLine),
                        () => $"Reorder of [{s}, {e}) of {BidiShow(points)} ({direction?.ToString() ?? "found"}): order {string.Join(' ', arrayOrder)} / {string.Join(' ', spanOrder)} / "
                              + $"{string.Join(' ', inPlaceOrder)}, the reference {string.Join(' ', expectedOrder)}; levels {string.Join(' ', arrayLine)} / {string.Join(' ', spanLine)} / "
                              + $"{string.Join(' ', inPlace[s..e])} / {string.Join(' ', alone)}, the reference {string.Join(' ', expectedLine)}");
                }
            }

            // The bidi reading order both ways, the text read as logical and as columns, in either direction.
            foreach (bool rtl in (bool[])[false, true])
            {
                string columns = order.ToColumns(text, rtl), logical = order.FromColumns(text, rtl);
                string expectedColumns = ReferenceBidiOrder.ToColumns(text, rtl), expectedLogical = ReferenceBidiOrder.FromColumns(text, rtl);
                Expect(columns == expectedColumns && logical == expectedLogical,
                    () => $"the bidi order of {BidiShow(points)} ({(rtl ? "right to left" : "left to right")}): columns {BidiShow([.. columns.Select(c => (int)c)])}, "
                          + $"the reference {BidiShow([.. expectedColumns.Select(c => (int)c)])}; logical {BidiShow([.. logical.Select(c => (int)c)])}, "
                          + $"the reference {BidiShow([.. expectedLogical.Select(c => (int)c)])}");
            }
        }

        Check(failed == 0, $"{failed} of {compared} comparisons on {Texts} random texts differ from the reference; the first: {string.Join("; ", first)}");
        Console.WriteLine($"    {Texts} random mixed texts, {compared} comparisons: the span forms, the array forms and the bidi reading order give the reference's levels, orders and texts");
    }

    // What `call` allocates on this thread over 1,000 calls once warm (300 calls first), as the lines layout's test measures.
    private static long BidiAllocated(Func<object?> call)
    {
        for (int i = 0; i < 300; i++)
        {
            call();
        }

        long before = GC.GetAllocatedBytesForCurrentThread();
        for (int i = 0; i < 1_000; i++)
        {
            call();
        }

        return GC.GetAllocatedBytesForCurrentThread() - before;
    }

    private static void BidiPooledAllocation()
    {
        // A right-to-left line of a report: Arabic words with a year in brackets, a product name and version, a percentage
        // and a bracketed note.
        string text = "صدر التقرير السنوي (2026) مع Idrak 4.1 للفريق، بنسبة 12% اليوم [النسخة الثانية]";
        int[] points = [.. text.EnumerateRunes().Select(r => r.Value)];
        BidiClass[] classes = [.. points.Select(Bidi.ClassOf)];
        var levels = new sbyte[points.Length];
        var order = new int[points.Length];
        var (paragraphLevels, kinds, rtl) = Bidi.Resolve(text);
        var lineOrder = new int[text.Length];
        var lineLevels = new byte[text.Length];
        Check(rtl && Bidi.Levels(classes, points, -1, levels) == 1, "the line is right to left");

        long spans = BidiAllocated(() =>
        {
            Bidi.Levels(classes, points, -1, levels);
            Bidi.VisualOrder(levels, order);
            Bidi.Reorder(paragraphLevels, kinds, 0, text.Length, rtl, lineOrder, lineLevels);
            Bidi.Reorder(paragraphLevels, kinds, 0, text.Length, rtl, Span<int>.Empty, lineLevels);
            return null;
        });
        Check(spans == 0, $"the span forms of Levels, VisualOrder and Reorder on a line of {points.Length} code points allocated {spans} bytes in 1,000 calls once warm"
                          + (spans == 0 ? "" : $" ({AllocatedTypes(() => Bidi.Levels(classes, points, -1, levels))}; {AllocatedTypes(() => Bidi.Reorder(paragraphLevels, kinds, 0, text.Length, rtl, lineOrder, lineLevels))})"));

        // The array forms and Resolve give back their arrays alone: the same arrays made by hand.
        int kept = Bidi.VisualOrder(levels, order);
        long arrays = BidiAllocated(() => (Bidi.Levels(classes, points, -1, out _), Bidi.VisualOrder(levels), Bidi.Reorder(paragraphLevels, kinds, 0, text.Length, rtl), Bidi.Resolve(text)));
        long given = BidiAllocated(() => (new sbyte[points.Length], new int[kept], (new int[text.Length], new byte[text.Length]), (new byte[text.Length], new BidiClass[text.Length], rtl)));
        Check(arrays <= given, $"the array forms and Resolve allocated {arrays} bytes in 1,000 calls once warm, {given} for what they give back"
                               + (arrays <= given ? "" : $" ({AllocatedTypes(() => Bidi.Resolve(text))})"));
        Console.WriteLine($"    once warm, 1,000 calls: the span forms {spans} bytes; the array forms and Resolve {arrays} bytes, {given} for what they give back");
    }

    // The algorithm as it was before its pooled path (Idrak.Imaging/Shaping/Bidi.cs at ecb3bd0), word for word: the reference.
    private static class ReferenceBidi
    {
        private const int MaxDepth = 125;

        // <summary>The Unicode version the character tables come from.</summary>
        public static string UnicodeVersion => UnicodeText.Version;

        // <summary>The bidirectional class of a code point.</summary>
        public static BidiClass ClassOf(int codePoint) => UnicodeText.Bidi(codePoint);

        // <summary>The mirrored glyph of a code point shown right to left (BidiMirroring.txt), or the code point itself.</summary>
        public static int Mirror(int codePoint) => UnicodeText.Mirror(codePoint);

        // <summary>Whether a class is removed by rule X9 (embedding and override controls, boundary neutrals).</summary>
        public static bool Removed(BidiClass type) => type is BidiClass.RLE or BidiClass.LRE or BidiClass.RLO or BidiClass.LRO or BidiClass.PDF or BidiClass.BN;

        // <summary>
        // The paragraph level by rules P2 and P3: 1 when the first strong character (outside isolates) is R or AL, 0 when it
        // is L or there is none (<paramref name="fallback"/> then).
        // </summary>
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

        // <summary>
        // The resolved embedding level of each character of one paragraph (after L1), or -1 for a character rule X9 removes.
        // <paramref name="codePoints"/> identifies paired brackets (rule N0); it may be empty when only classes are known.
        // <paramref name="paragraphLevel"/> is 0, 1, or -1 to find it by P2 and P3 (<paramref name="resolvedParagraphLevel"/>).
        // </summary>
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

        // <summary>Rule L2: the indexes of the characters in visual order (left to right), leaving out those with level -1.</summary>
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

        // <summary>
        // The resolved embedding level of every UTF-16 index of a paragraph (a character X9 removes takes the level before
        // it), the paragraph's classes, and whether it is right to left.
        // </summary>
        // <param name="text">The paragraph (no paragraph separator inside).</param>
        // <param name="rightToLeft">The paragraph's direction, or null for its first strong character's (left to right without one).</param>
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

        // <summary>
        // The visual order of a line [start, end) of a resolved paragraph: L1 (separators, and trailing whitespace and controls,
        // back to the paragraph level) then L2 (each level and above reversed, from the highest down to the lowest odd one).
        // Returns the text indexes left to right, and the line's levels after L1 in text order.
        // </summary>
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

    // The bidi reading order as it was before its pooled path (ReadingOrderImplementations.cs at ecb3bd0), word for word
    // but static and on the reference algorithm.
    private static class ReferenceBidiOrder
    {
        private const int Rounds = 8;

        public static string ToColumns(string logical, bool rightToLeft)
        {
            ArgumentNullException.ThrowIfNull(logical);
            if (Unchanged(logical, rightToLeft))
            {
                return logical;
            }

            var units = Units(logical);
            return Gather(logical, units, Order(logical, units, rightToLeft));
        }

        public static string FromColumns(string columns, bool rightToLeft)
        {
            ArgumentNullException.ThrowIfNull(columns);
            if (Unchanged(columns, rightToLeft))
            {
                return columns;                                                     // its own logical order: ToColumns gives it back
            }

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

        // Whether every code point of `text` (read as Units reads them) resolves to the paragraph's level, so that the visual
        // order is the logical one (read from the right, a right-to-left line's too): no explicit embedding, override or isolate
        // (X1-X10) and no Arabic number (I1, I2); in a left-to-right line no R or AL (European numbers then follow sos L, W7,
        // and neutrals and brackets the strong L around them, N0-N2); in a right-to-left line no L or European number (all the
        // rest resolves to R). Separators and trailing whitespace go to the paragraph's level anyway (L1).
        private static bool Unchanged(string text, bool rightToLeft)
        {
            for (int i = 0; i < text.Length; i++)
            {
                int point = text[i];
                if (char.IsHighSurrogate(text[i]) && i + 1 < text.Length && char.IsLowSurrogate(text[i + 1]))
                {
                    point = char.ConvertToUtf32(text[i], text[i + 1]);
                    i++;
                }

                var type = ReferenceBidi.ClassOf(point);
                if (type >= BidiClass.LRE || type == BidiClass.AN || (rightToLeft ? type is BidiClass.L or BidiClass.EN : type is BidiClass.R or BidiClass.AL))
                {
                    return false;
                }
            }

            return true;
        }

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
                classes[k] = ReferenceBidi.ClassOf(points[k]);
            }

            var resolved = ReferenceBidi.Levels(classes, points, rightToLeft ? 1 : 0, out int paragraph);
            var levels = new byte[count];
            int previous = paragraph;
            for (int k = 0; k < count; k++)
            {
                previous = resolved[k] < 0 ? previous : resolved[k];                    // a character X9 removes takes the level before it
                levels[k] = (byte)previous;
            }

            var (order, _) = ReferenceBidi.Reorder(levels, classes, 0, count, rightToLeft);
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
}
