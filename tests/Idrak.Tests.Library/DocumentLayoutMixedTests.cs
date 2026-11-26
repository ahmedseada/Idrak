// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Text;
using Idrak.Imaging.Shaping;

// Plan 28, "Bidi pooled" (2026-11-19): with UAX #9's working space taken from the shared pools, a mixed-direction line read
// through the lines layout and the bidi reading order allocates only what it gives back, as an unmixed line does.
internal abstract partial class Tests
{
    // What `call` allocates on this thread over 1,000 calls once warm (300 calls first), as LayoutsLinesAllocation measures.
    private static long MixedLinesAllocated(Func<object> call)
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

    private static void LayoutsLinesMixedAllocation(Device device)
    {
        CpuOnly(device);
        var order = new BidiReadingOrder();
        var options = new DocumentOptions { Layout = DocumentTextLayouts.Lines };
        var layout = new LinesLayout(order);   // the default layout with the default reading order, as the PDF reader reaches them

        // Three mixed lines in logical order: a release note in Arabic with a year and a product name (right to left), a report
        // line in English with Arabic words, a version and a percentage (left to right), and a long Arabic line of a
        // newsletter with versions, years, percentages and Latin words (right to left).
        var newsletter = new StringBuilder();
        while (newsletter.Length < 300)
        {
            newsletter.Append("أصدر الفريق النسخة 4.1 من Idrak في 2026 بتحسن 12% مع release notes جديدة ");
        }

        (string Logical, bool RightToLeft)[] mixed =
        [
            ("صدر التقرير 2026 مع Idrak", true),
            ("The quarterly report lists التقرير السنوي and version 4.1 with الفريق at 12% growth", false),
            (newsletter.ToString(0, 300).TrimEnd(), true),
        ];

        // Each line as a page shows it: its columns (read in the line's direction) cut into words at its spaces, a run each,
        // placed from the line's start edge with a gap between them; a right-to-left line's runs hold their glyphs left to
        // right (each word's text reversed) and come in content order from the right.
        var runs = new List<DocumentTextRun>();
        var columns = new string[mixed.Length];
        int content = 0;
        for (int l = 0; l < mixed.Length; l++)
        {
            var (logical, rightToLeft) = mixed[l];
            columns[l] = order.ToColumns(logical, rightToLeft);
            string[] words = columns[l].Split(' ');
            Check(words.All(w => w.Length > 0), $"line {l}: its columns have single spaces between words: \"{columns[l]}\"");
            const float Size = 10f, Gap = 10f;
            float baseline = 50f + 40f * l, edge = rightToLeft ? 4000f : 50f;
            foreach (string word in words)
            {
                float width = word.Length * Size * 0.5f;
                float x = rightToLeft ? edge - width : edge;
                runs.Add(Placed(rightToLeft ? LinesLayout.Reversed(word) : word, x, baseline, Size, content++, width));
                edge = rightToLeft ? x - Gap : x + width + Gap;
            }
        }

        DocumentTextRun[] page = [.. runs];
        var text = layout.Arrange(page, 4100f, 792f, options);
        Check(text.Lines.Count == mixed.Length, $"{text.Lines.Count} lines: \"{text.Text}\"");
        for (int l = 0; l < mixed.Length; l++)
        {
            var line = text.Lines[l];
            string expected = order.FromColumns(columns[l], mixed[l].RightToLeft);
            Check(line.Text == expected && line.RightToLeft == mixed[l].RightToLeft && order.ToColumns(line.Text, line.RightToLeft) == columns[l] && line.Text != columns[l],
                $"line {l} (right to left {line.RightToLeft}): \"{line.Text}\", the order's \"{expected}\" from the columns \"{columns[l]}\"");
        }

        // What it gives back and asks, made by hand: each line's text and runs, the content-order text its direction is read
        // from, and the columns the reading order is asked about (a mixed line's text is another string); the page's text.
        int[] lengths = [.. text.Lines.Select(l => l.Text.Length)];
        int[] counts = [.. text.Lines.Select(l => l.Runs.Count)];
        int[] asked = [.. text.Lines.Select(l => l.Runs.Count > 1 ? l.Runs.Sum(r => page[r].Text.Length) : 0)];
        int[] columnLengths = [.. columns.Select(c => c.Length)];
        int pageLength = text.Text.Length;
        object Given()
        {
            var lines = new DocumentLine[lengths.Length];
            for (int i = 0; i < lines.Length; i++)
            {
                string question = asked[i] > 0 ? new string('x', asked[i]) : "";
                string asColumns = new string('x', columnLengths[i]);
                lines[i] = new DocumentLine(new string('x', lengths[i]), DocumentBox.Empty, question.Length + asColumns.Length > 0, new int[counts[i]]);
            }

            return new DocumentText(lines, new string('x', pageLength));
        }

        long given = MixedLinesAllocated(Given), arranged = MixedLinesAllocated(() => layout.Arrange(page, 4100f, 792f, options));
        Check(arranged <= given, $"{page.Length} runs in {text.Lines.Count} mixed lines arranged 1,000 times allocated {arranged} bytes once warm, {given} for what it gives back and asks"
            + (arranged <= given ? "" : $" ({AllocatedTypes(() => layout.Arrange(page, 4100f, 792f, options))})"));

        // The reading order alone on each line: the string it gives back, nothing more.
        for (int l = 0; l < mixed.Length; l++)
        {
            string logical = text.Lines[l].Text, lineColumns = columns[l];
            bool rightToLeft = mixed[l].RightToLeft;
            long from = MixedLinesAllocated(() => order.FromColumns(lineColumns, rightToLeft)), fromGiven = MixedLinesAllocated(() => new string('x', logical.Length));
            long to = MixedLinesAllocated(() => order.ToColumns(logical, rightToLeft)), toGiven = MixedLinesAllocated(() => new string('x', lineColumns.Length));
            Check(from <= fromGiven && to <= toGiven, $"a mixed line of {logical.Length} characters (right to left {rightToLeft}), 1,000 calls once warm: FromColumns allocated {from} bytes "
                + $"({fromGiven} for the string it gives back), ToColumns {to} ({toGiven})"
                + (from <= fromGiven ? "" : $" ({AllocatedTypes(() => order.FromColumns(lineColumns, rightToLeft))})"));
            Console.WriteLine($"    a mixed line of {logical.Length} characters: FromColumns {from / 1000.0:F0} bytes a call, ToColumns {to / 1000.0:F0}, the strings they give back {fromGiven / 1000.0:F0} and {toGiven / 1000.0:F0}");
        }

        Console.WriteLine($"    {page.Length} runs in {text.Lines.Count} mixed lines: {arranged / 1000.0:F0} bytes a page once warm, {given / 1000.0:F0} for what it gives back and asks");
    }
}
