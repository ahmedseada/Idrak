// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Globalization;
using System.IO.Compression;
using System.Text;
using Idrak.Imaging.Shaping;

// Plan 36, G-1 piece 4 (T7): the Unicode Bidirectional Algorithm against Unicode's own conformance files, BidiTest.txt and
// BidiCharacterTest.txt of the Unicode version the tables come from (18.0.0), kept gzipped in tests/Idrak.Tests/data/unicode
// (from https://github.com/unicode-org/unicodetools, unicodetools/data/ucd/18.0.0; the files carry Unicode's notice).
internal abstract partial class DesktopTests : Tests
{
    private static readonly (string Name, Action<Device> Run)[] BidiConformanceGroup =
    [
        ("bidi: every case of Unicode's BidiTest.txt (the tables' Unicode version): the levels and the visual order of each class sequence at each paragraph level its bitset names (auto, left to right, right to left)",
            d => { if (d == Device.Cpu) BidiClassConformance(); }),
        ("bidi: every line of Unicode's BidiCharacterTest.txt (the tables' Unicode version): the resolved paragraph level, the levels and the visual order of each code point sequence, paired brackets included",
            d => { if (d == Device.Cpu) BidiCharacterConformance(); }),
    ];

    // A gzipped file of tests/Idrak.Tests/data/unicode, its first line checked to name the tables' Unicode version.
    private static StreamReader BidiData(string name)
    {
        var reader = new StreamReader(new GZipStream(File.OpenRead(TestData("unicode/" + name)), CompressionMode.Decompress), Encoding.UTF8);
        string first = reader.ReadLine() ?? "";
        Check(first.Contains(Bidi.UnicodeVersion, StringComparison.Ordinal), $"{name} is for Unicode {Bidi.UnicodeVersion}, as the tables are: {first}");
        return reader;
    }

    // Levels as the files write them ("x" for a character rule X9 removes: -1 here), separated by spaces.
    private static sbyte[] BidiLevels(string text) =>
        [.. text.Split([' ', '\t'], StringSplitOptions.RemoveEmptyEntries).Select(l => l == "x" ? (sbyte)-1 : sbyte.Parse(l, CultureInfo.InvariantCulture))];

    private static int[] BidiIndexes(string text) => [.. text.Split([' ', '\t'], StringSplitOptions.RemoveEmptyEntries).Select(i => int.Parse(i, CultureInfo.InvariantCulture))];

    private static void BidiClassConformance()
    {
        using var reader = BidiData("bidi-test.txt.gz");
        sbyte[] levels = [];
        int[] order = [];
        int cases = 0, failed = 0;
        var first = new List<string>();
        (int Bit, int Level)[] paragraphs = [(1, -1), (2, 0), (4, 1)];
        for (string? line = reader.ReadLine(); line is not null; line = reader.ReadLine())
        {
            if (line.Length == 0 || line[0] == '#')
            {
                continue;
            }

            if (line.StartsWith("@Levels:", StringComparison.Ordinal))
            {
                levels = BidiLevels(line["@Levels:".Length..]);
                continue;
            }

            if (line.StartsWith("@Reorder:", StringComparison.Ordinal))
            {
                order = BidiIndexes(line["@Reorder:".Length..]);
                continue;
            }

            int semicolon = line.IndexOf(';', StringComparison.Ordinal);
            BidiClass[] types = [.. line[..semicolon].Split([' ', '\t'], StringSplitOptions.RemoveEmptyEntries).Select(Enum.Parse<BidiClass>)];
            int bits = int.Parse(line[(semicolon + 1)..].Trim(), CultureInfo.InvariantCulture);
            foreach (var (bit, paragraph) in paragraphs)
            {
                if ((bits & bit) == 0)
                {
                    continue;
                }

                cases++;
                var actual = Bidi.Levels(types, ReadOnlySpan<int>.Empty, paragraph, out _);
                int[] visual = Bidi.VisualOrder(actual);
                if (!actual.AsSpan().SequenceEqual(levels) || !visual.AsSpan().SequenceEqual(order))
                {
                    failed++;
                    if (first.Count < 5)
                    {
                        first.Add($"{line[..semicolon]} at paragraph level {paragraph}: levels {string.Join(' ', actual)} (want {string.Join(' ', levels)}), "
                                  + $"order {string.Join(' ', visual)} (want {string.Join(' ', order)})");
                    }
                }
            }
        }

        Check(cases > 100_000, $"only {cases} cases read: the file or its reading is broken");
        Check(failed == 0, $"{failed} of {cases} BidiTest.txt cases differ; the first: {string.Join("; ", first)}");
        Console.WriteLine($"    BidiTest.txt: {cases} cases, all equal");
    }

    private static void BidiCharacterConformance()
    {
        using var reader = BidiData("bidi-character-test.txt.gz");
        int cases = 0, failed = 0;
        var first = new List<string>();
        for (string? line = reader.ReadLine(); line is not null; line = reader.ReadLine())
        {
            if (line.Length == 0 || line[0] == '#')
            {
                continue;
            }

            // code points; paragraph direction (0 left to right, 1 right to left, 2 auto); resolved paragraph level; levels; visual order
            string[] fields = line.Split(';');
            int[] codePoints = [.. fields[0].Split(' ', StringSplitOptions.RemoveEmptyEntries).Select(c => int.Parse(c, NumberStyles.HexNumber, CultureInfo.InvariantCulture))];
            int direction = int.Parse(fields[1], CultureInfo.InvariantCulture), paragraph = int.Parse(fields[2], CultureInfo.InvariantCulture);
            sbyte[] levels = BidiLevels(fields[3]);
            int[] order = BidiIndexes(fields[4]);
            BidiClass[] types = [.. codePoints.Select(Bidi.ClassOf)];
            cases++;
            var actual = Bidi.Levels(types, codePoints, direction == 2 ? -1 : direction, out int resolved);
            int[] visual = Bidi.VisualOrder(actual);
            if (resolved != paragraph || !actual.AsSpan().SequenceEqual(levels) || !visual.AsSpan().SequenceEqual(order))
            {
                failed++;
                if (first.Count < 5)
                {
                    first.Add($"{fields[0]} (direction {direction}): paragraph {resolved} (want {paragraph}), levels {string.Join(' ', actual)} (want {fields[3]}), "
                              + $"order {string.Join(' ', visual)} (want {fields[4]})");
                }
            }
        }

        Check(cases > 50_000, $"only {cases} lines read: the file or its reading is broken");
        Check(failed == 0, $"{failed} of {cases} BidiCharacterTest.txt lines differ; the first: {string.Join("; ", first)}");
        Console.WriteLine($"    BidiCharacterTest.txt: {cases} lines, all equal");
    }
}
