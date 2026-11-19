// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Diagnostics;
using System.Globalization;
using System.Text;
using System.Text.Json.Nodes;
using Idrak.Data.Pdf;

// Plan 28, phase D3: PDF text (the content interpreter, fonts to text, widths, run boxes, the lines layout's use),
// checked on the D3 samples tools/documents/make_fixtures.py writes into the ignored test-output/documents/samples
// (IDRAK_DOCUMENTS_SAMPLES moves them), and on hand-made bytes.
internal abstract partial class Tests
{
    internal static readonly (string Name, Action<Device> Run)[] DocumentPdfTextGroup =
    [
        ("documents pdf: text: reading every D3 sample page's text, timed cold and warm apart, with the managed allocation per page (the same text on every pass)", PdfTextTimed),
        ("documents pdf: text: every D3 sample page's runs as written (text, box within float rounding, font size, font, hidden, order, /ActualText), its lines with their direction and its text in logical order through bidi (right-to-left and mixed lines, ligatures kept whole), unmapped glyphs counted, rotated and cropped pages, vertical text, forms and text matrices", PdfTextSamples),
        ("documents pdf: text: a form that draws itself trips MaxDepth when its page is read, naming the limit and the file; pages read in parallel give the same runs, lines, glyphs and issues as read in order", PdfTextHostileAndParallel),
        ("documents pdf: text: the options (IncludeHidden leaves invisible runs out with the others' order kept, WordGap decides TJ word gaps, the content layout gives one line a run), glyphs and faces when asked (embedded programs through FontFormats, none for the standard 14)", PdfTextOptions),
        ("documents pdf: text: glyph names to text by the Adobe Glyph List's rules, the standard fonts and their common other names, CMaps (code spaces, bfchar, bfrange with arrays, cidrange, notdef), and a hand-made page (a font the page lacks counted, an inline image holding \" EIQ\", spans named in /Properties, unknown operators passed over, paths in the order)", PdfTextHandMade),
        ("documents pdf: text: once warm, a page of a document kept open allocates what it gives back (runs, their text, lines, text, issues), what the lines layout makes of its runs, and at most a named bound of the reader's own (the content stream's decoding, the list the layout is given): many fonts, right to left, two-byte codes, unmapped glyphs, TrueType", PdfTextAllocation),
        ("documents pdf: text: TrueType glyphs found by their names in the program's post table (ISO 32000-2, 9.6.5.4), its cmap holding no letter: format 1 by the 258 standard Macintosh names' order, format 2 by their indices; the sample's text, boxes and glyph ids", PdfTextMacNames),
        ("documents pdf: text: a bare CFF with no /Encoding reads its codes through the program's own encoding: the predefined Expert encoding over the Expert and Expert Subset charsets (each glyph named as the charset names it); the sample's text by the glyph names, boxes and glyph ids", PdfTextCffExpert),
        ("documents pdf: text: vertical text whose /W2 gives position vectors of its own (ISO 32000-2, 9.7.4.3): each glyph drawn from its vertical origin moved by its vector, its box across the column from -vx to its width less vx; the sample's text and boxes", PdfTextVerticalW2),
    ];

    // A kept-open page's warm allocation beyond what it gives back and what the layout makes of its runs: the reader's own,
    // which is the content stream's decoding (its filter list, window, buffer and the inflater's objects), the page's issues
    // and the list of runs the layout is given (a reference a run). The numbers seen are in plan 28's note "Reader memory".
    private const long PdfReaderOwnBytes = 1024, PdfReaderOwnBytesPerRun = 8;

    // The D3 samples that open (not hostile, not encrypted).
    private static List<PdfSampleEntry> PdfTextSamplesReadable()
    {
        var samples = PdfSamples().Where(s => s.Phases.Contains("D3") && s.Readable).ToList();
        Check(samples.Count >= 19, $"{samples.Count} readable D3 samples; the generator writes 19 (make them again: tools/documents/make_fixtures.py)");
        return samples;
    }

    private static void PdfTextTimed(Device device)
    {
        CpuOnly(device);
        var samples = PdfTextSamplesReadable();
        var files = samples.Select(s => (s.Name, Bytes: File.ReadAllBytes(s.Path))).ToList();
        int pages = samples.Sum(s => s.Expected["pages"]!.AsArray().Count);

        string Pass()
        {
            var text = new StringBuilder();
            foreach (var (name, bytes) in files)
            {
                using var document = DocumentFormats.Open(bytes.AsMemory(), name + ".pdf");
                for (int i = 0; i < document.PageCount; i++)
                {
                    text.Append(document.ReadPage(i, DocumentParts.Text).Text).Append('\f');
                }
            }

            return text.ToString();
        }

        var clock = Stopwatch.StartNew();
        string first = Pass();
        double cold = clock.Elapsed.TotalMilliseconds;
        var warm = new List<double>();
        long allocated = 0;
        for (int pass = 0; pass < 20; pass++)
        {
            long before = GC.GetAllocatedBytesForCurrentThread();
            long start = Stopwatch.GetTimestamp();
            string again = Pass();
            warm.Add(Stopwatch.GetElapsedTime(start).TotalMilliseconds);
            allocated = GC.GetAllocatedBytesForCurrentThread() - before;
            Check(again == first, $"pass {pass + 1} read other text than the first");
        }

        warm.Sort();
        double median = warm[warm.Count / 2];
        Console.WriteLine(string.Create(CultureInfo.InvariantCulture,
            $"    {samples.Count} D3 samples, {pages} pages (opened from memory, text read): cold {cold:F1} ms, warm median {median:F2} ms a pass ({median * 1000 / pages:F0} µs a page), {allocated / pages / 1024.0:F1} KiB allocated a page"));

        // The pages of documents kept open: the fonts are read once, each page's text again.
        var open = files.Select(f => DocumentFormats.Open(f.Bytes.AsMemory(), f.Name + ".pdf")).ToList();
        try
        {
            string Again()
            {
                var text = new StringBuilder();
                foreach (var document in open)
                {
                    for (int i = 0; i < document.PageCount; i++)
                    {
                        text.Append(document.ReadPage(i, DocumentParts.Text).Text).Append('\f');
                    }
                }

                return text.ToString();
            }

            Check(Again() == first, "the pages of open documents read other text");
            var kept = new List<double>();
            long keptAllocated = 0;
            for (int pass = 0; pass < 20; pass++)
            {
                long before = GC.GetAllocatedBytesForCurrentThread();
                long start = Stopwatch.GetTimestamp();
                string again = Again();
                kept.Add(Stopwatch.GetElapsedTime(start).TotalMilliseconds);
                keptAllocated = GC.GetAllocatedBytesForCurrentThread() - before;
                Check(again == first, $"pass {pass + 1} over open documents read other text");
            }

            kept.Sort();
            double keptMedian = kept[kept.Count / 2];
            Console.WriteLine(string.Create(CultureInfo.InvariantCulture,
                $"    the same pages of documents kept open (fonts read once): warm median {keptMedian:F2} ms a pass ({keptMedian * 1000 / pages:F0} µs a page), {keptAllocated / pages / 1024.0:F1} KiB allocated a page"));
        }
        finally
        {
            open.ForEach(d => d.Dispose());
        }
    }

    private static void PdfTextSamples(Device device)
    {
        CpuOnly(device);
        int checkedRuns = 0, checkedPages = 0;
        foreach (var sample in PdfTextSamplesReadable())
        {
            using var document = DocumentFormats.Open(sample.Path);
            foreach (var expected in sample.Expected["pages"]!.AsArray())
            {
                int index = (int)expected!["index"]!;
                var page = document.ReadPage(index, DocumentParts.Text);
                checkedRuns += PdfCheckTextPage($"{sample.Name}: page {index}", page, expected.AsObject());
                checkedPages++;
            }
        }

        Check(checkedPages >= 30 && checkedRuns >= 100, $"{checkedPages} pages and {checkedRuns} runs checked");
    }

    // A page's runs, lines, text and unmapped glyphs against the sample's JSON; the number of runs checked.
    private static int PdfCheckTextPage(string what, DocumentPage page, JsonObject expected)
    {
        var runs = expected["runs"]!.AsArray().Select(r => r!.AsObject()).OrderBy(r => (int)r["order"]!).ToList();
        Check(page.Runs.Count == runs.Count, $"{what}: {page.Runs.Count} runs ({string.Join(" | ", page.Runs.Select(r => r.Text))}), expected {runs.Count} ({string.Join(" | ", runs.Select(r => (string?)r["text"]))})");
        for (int i = 0; i < runs.Count; i++)
        {
            var run = page.Runs[i];
            var e = runs[i];
            string at = $"{what}: run {i} \"{run.Text}\"";
            Check(run.Text == (string?)e["text"], $"{at}: the text, expected \"{e["text"]}\"");
            Check(run.Order == (int)e["order"]!, $"{at}: order {run.Order}, expected {e["order"]}");
            var box = e["box"]!.AsArray();
            float[] b = [(float)box[0]!, (float)box[1]!, (float)box[2]!, (float)box[3]!];
            const float Tolerance = 0.005f;
            Check(MathF.Abs(run.Box.X - b[0]) <= Tolerance && MathF.Abs(run.Box.Y - b[1]) <= Tolerance && MathF.Abs(run.Box.Width - b[2]) <= Tolerance && MathF.Abs(run.Box.Height - b[3]) <= Tolerance,
                $"{at}: the box {run.Box}, expected ({b[0]}, {b[1]}, {b[2]} x {b[3]})");
            Check(MathF.Abs(run.FontSize - (float)e["fontSize"]!) <= 1e-3f, $"{at}: font size {run.FontSize}, expected {e["fontSize"]}");
            Check(run.Hidden == (bool)e["hidden"]!, $"{at}: hidden {run.Hidden}, expected {e["hidden"]} (render mode {e["renderMode"]})");
            Check(run.Font == (string?)e["font"], $"{at}: the font {run.Font}, expected {e["font"]}");
            Check(run.Logical == (e["actualText"]?.GetValue<bool>() == true), $"{at}: logical {run.Logical} (an /ActualText span's run is)");
        }

        var lines = expected["lines"]!.AsArray();
        Check(page.Lines.Count == lines.Count, $"{what}: {page.Lines.Count} lines ({string.Join(" | ", page.Lines.Select(l => l.Text))}), expected {lines.Count}");
        for (int i = 0; i < lines.Count; i++)
        {
            var line = page.Lines[i];
            Check(line.Text == (string?)lines[i]!["text"] && line.RightToLeft == ((string?)lines[i]!["direction"] == "rtl"),
                $"{what}: line {i} \"{line.Text}\" ({(line.RightToLeft ? "rtl" : "ltr")}), expected \"{lines[i]!["text"]}\" ({lines[i]!["direction"]})");
        }

        Check(page.Text == (string?)expected["text"], $"{what}: the page's text \"{page.Text}\", expected \"{expected["text"]}\"");
        int unmapped = page.Issues.Where(i => i.Kind == DocumentIssue.UnmappedGlyphs).Sum(i => i.Count);
        Check(unmapped == (int)expected["issues"]!["unmappedGlyphs"]!, $"{what}: {unmapped} unmapped glyphs, expected {expected["issues"]!["unmappedGlyphs"]}");
        Check(page.Issues.All(i => i.Kind == DocumentIssue.UnmappedGlyphs), $"{what}: issues {string.Join("; ", page.Issues.Select(i => $"{i.Kind}: {i.Detail}"))}");
        return runs.Count;
    }

    private static void PdfTextAllocation(Device device)
    {
        CpuOnly(device);
        var options = new DocumentOptions { Layout = DocumentTextLayouts.Lines, ReadingOrder = ReadingOrders.Bidi };
        var samples = PdfTextSamplesReadable();
        static long Allocated(Func<object?> call)
        {
            for (int i = 0; i < 1_000; i++)
            {
                call();
            }

            long before = GC.GetAllocatedBytesForCurrentThread();
            for (int i = 0; i < 1_000; i++)
            {
                call();
            }

            return (GC.GetAllocatedBytesForCurrentThread() - before) / 1_000;
        }

        // Upright pages whose runs are the layout's pieces as they are (shown by Tj, no TJ gap, no glyph of several
        // characters, which is a piece of its own): many fonts, right to left, two-byte codes (Type 0, one with no text
        // map: its unmapped glyphs an issue), a TrueType font.
        foreach (string name in (string[])["standard14", "rtl", "cidtogidmap-stream", "type0-no-cmap", "truetype-simple"])
        {
            var sample = samples.Single(s => s.Name == name);
            using var document = DocumentFormats.Open(File.ReadAllBytes(sample.Path).AsMemory(), name + ".pdf", options);
            DocumentPage Read() => document.ReadPage(0, DocumentParts.Text);
            var page = Read();
            var again = Read();
            Check(PdfTextSignature(page) == PdfTextSignature(again), $"{name}: the page read again from the document kept open reads otherwise");

            // What the page gives back, made by hand: the same objects of the same sizes, kept in a holder made once.
            int[] strings = PdfPageStrings(page, again);
            int[] lineRuns = [.. page.Lines.Select(l => l.Runs is List<int> indices ? indices.Capacity : l.Runs.Count)];
            int lines = page.Lines is List<DocumentLine> kept ? kept.Capacity : page.Lines.Count, issues = page.Issues.Count, runCount = page.Runs.Count;
            var holder = new object?[strings.Length + 4];
            object? Given()
            {
                for (int i = 0; i < strings.Length; i++)
                {
                    holder[i] = new string('x', strings[i]);
                }

                var runs = new DocumentTextRun[runCount];
                for (int i = 0; i < runs.Length; i++)
                {
                    runs[i] = new DocumentTextRun("", DocumentBox.Empty, 0f);
                }

                var made = new List<DocumentLine>(lines);
                foreach (int capacity in lineRuns)
                {
                    made.Add(new DocumentLine("", DocumentBox.Empty, false, new List<int>(capacity)));
                }

                holder[^4] = runs;
                holder[^3] = made;
                holder[^2] = issues > 0 ? new List<DocumentIssue>(issues) : null;
                holder[^1] = new DocumentPage(0, 1f, 1f);
                return holder;
            }

            // What the lines layout makes of the page's runs (its pieces), less the texts it gives that the page gives back.
            var arranged = DocumentTextLayouts.Arrange(page.Runs, page.Width, page.Height, options);
            var runTexts = new HashSet<string>(page.Runs.Select(r => r.Text), ReferenceEqualityComparer.Instance);
            int[] texts = [.. arranged.Lines.Select(l => l.Text).Append(arranged.Text).Where(t => t.Length > 0 && !runTexts.Contains(t)).Distinct<string>(ReferenceEqualityComparer.Instance).Select(t => t.Length)];
            var textHolder = new object[texts.Length];
            object? Texts()
            {
                for (int i = 0; i < texts.Length; i++)
                {
                    textHolder[i] = new string('x', texts[i]);
                }

                return textHolder;
            }

            long read = Allocated(Read), given = Allocated(Given);
            long layout = Allocated(() => DocumentTextLayouts.Arrange(page.Runs, page.Width, page.Height, options)) - Allocated(Texts);
            long own = read - given - layout, bound = PdfReaderOwnBytes + PdfReaderOwnBytesPerRun * runCount;
            Console.WriteLine(string.Create(CultureInfo.InvariantCulture,
                $"    {name}: {read:N0} B a page read once warm: {given:N0} given back ({runCount} runs, {page.Lines.Count} lines), {layout:N0} the layout's own, {own:N0} the reader's own (bound {bound:N0})"));
            Check(own <= bound, $"{name}: a page of a document kept open allocates {read} bytes once warm: {given} for what it gives back, {layout} for what the layout makes of its runs, "
                + $"so {own} of the reader's own, past the bound of {bound} (PdfReaderOwnBytes {PdfReaderOwnBytes} + PdfReaderOwnBytesPerRun {PdfReaderOwnBytesPerRun} a run)"
                + (own <= bound ? "" : $": {AllocatedTypes(() => Read())}"));
        }
    }

    // The lengths of the strings a page read made (its runs', lines', text's and issues', each object once), less those
    // that are the same object on another read (kept by the reader from page to page).
    private static int[] PdfPageStrings(DocumentPage page, DocumentPage again)
    {
        var seen = new HashSet<string>(ReferenceEqualityComparer.Instance);
        var lengths = new List<int>();
        void Add(string text, string other)
        {
            if (text.Length > 0 && !ReferenceEquals(text, other) && seen.Add(text))
            {
                lengths.Add(text.Length);
            }
        }

        for (int i = 0; i < page.Runs.Count; i++)
        {
            Add(page.Runs[i].Text, again.Runs[i].Text);
        }

        for (int i = 0; i < page.Lines.Count; i++)
        {
            Add(page.Lines[i].Text, again.Lines[i].Text);
        }

        Add(page.Text, again.Text);
        for (int i = 0; i < page.Issues.Count; i++)
        {
            Add(page.Issues[i].Detail, again.Issues[i].Detail);
        }

        return [.. lengths];
    }

    private static void PdfTextHostileAndParallel(Device device)
    {
        CpuOnly(device);
        var hostile = PdfSamples().Single(s => s.Name == "hostile-form-recursion");
        var rule = hostile.Expected["hostile"]!;
        var options = new DocumentOptions { MaxDepth = (int)rule["options"]!["MaxDepth"]! };
        var clock = Stopwatch.StartNew();
        using (var document = DocumentFormats.Open(hostile.Path, options))
        {
            try
            {
                document.ReadPage(0, DocumentParts.Text);
                Check(false, "a form that draws itself was read without an error");
            }
            catch (InvalidDataException e)
            {
                Check(e.Message.Contains("MaxDepth", StringComparison.Ordinal) && e.Message.Contains("DocumentOptions.MaxDepth", StringComparison.Ordinal)
                      && e.Message.Contains(Path.GetFileName(hostile.Path), StringComparison.Ordinal), $"\"{e.Message}\" does not name MaxDepth, its option and the file");
            }

            Check(document.ReadPage(0, DocumentParts.None).Width == 420f, "the page's size still reads without its content");
        }

        Check(clock.Elapsed.TotalSeconds < 20, $"took {clock.Elapsed.TotalSeconds:F1} s to fail");

        // Every D3 page with its glyphs, in order, then all at once from shared documents.
        var samples = PdfTextSamplesReadable();
        var documents = samples.Select(s => DocumentFormats.Open(s.Path)).ToList();
        try
        {
            var jobs = documents.SelectMany(d => Enumerable.Range(0, d.PageCount).Select(i => (Document: d, Index: i))).ToList();
            var inOrder = jobs.Select(j => PdfTextSignature(j.Document.ReadPage(j.Index, DocumentParts.Text | DocumentParts.Glyphs))).ToList();
            using var fresh = new PdfDocumentList(samples.Select(s => DocumentFormats.Open(s.Path)));
            var freshJobs = fresh.Items.SelectMany(d => Enumerable.Range(0, d.PageCount).Select(i => (Document: d, Index: i))).ToList();
            var parallel = new string[freshJobs.Count];
            Parallel.For(0, freshJobs.Count, new ParallelOptions { MaxDegreeOfParallelism = Math.Max(2, Environment.ProcessorCount) },
                k => parallel[k] = PdfTextSignature(freshJobs[k].Document.ReadPage(freshJobs[k].Index, DocumentParts.Text | DocumentParts.Glyphs)));
            for (int k = 0; k < inOrder.Count; k++)
            {
                Check(parallel[k] == inOrder[k], $"page {freshJobs[k].Index} of a document read in parallel:\n{parallel[k]}\nin order:\n{inOrder[k]}");
            }

            Check(inOrder.Count >= 30, $"{inOrder.Count} pages read both ways");
        }
        finally
        {
            documents.ForEach(d => d.Dispose());
        }
    }

    private sealed class PdfDocumentList(IEnumerable<IDocument> items) : IDisposable
    {
        public List<IDocument> Items { get; } = [.. items];

        public void Dispose() => Items.ForEach(d => d.Dispose());
    }

    // What a page read gives, as text: each run (text, box, size, mode, order, glyphs), the lines, the text, the issues.
    private static string PdfTextSignature(DocumentPage page)
    {
        var text = new StringBuilder();
        foreach (var run in page.Runs)
        {
            text.Append(CultureInfo.InvariantCulture, $"run \"{run.Text}\" {run.Box} {run.FontSize} {run.Mode} {run.Order} {run.Logical} {run.Font} face {run.Face?.Family}");
            foreach (var glyph in run.Glyphs.Span)
            {
                text.Append(CultureInfo.InvariantCulture, $" {glyph.Glyph}@{glyph.X:F3},{glyph.Y:F3}+{glyph.Advance:F3}");
            }

            text.Append('\n');
        }

        foreach (var line in page.Lines)
        {
            text.Append(CultureInfo.InvariantCulture, $"line \"{line.Text}\" {line.Box} {line.RightToLeft} [{string.Join(",", line.Runs)}]\n");
        }

        text.Append("text \"").Append(page.Text).Append("\"\n");
        foreach (var issue in page.Issues)
        {
            text.Append(CultureInfo.InvariantCulture, $"issue {issue.Kind} {issue.Count} {issue.Detail}\n");
        }

        return text.ToString();
    }

    private static void PdfTextOptions(Device device)
    {
        CpuOnly(device);
        var samples = PdfTextSamplesReadable().ToDictionary(s => s.Name);

        // IncludeHidden false: modes 3 and 7 left out, the other runs keep their places in the content.
        using (var document = DocumentFormats.Open(samples["render-modes"].Path, new DocumentOptions { IncludeHidden = false }))
        {
            var page = document.ReadPage(0, DocumentParts.Text);
            Check(page.Runs.Select(r => r.Order).SequenceEqual([0, 1, 2, 4, 5, 6]) && page.Runs.All(r => !r.Hidden) && page.Lines.Count == 6,
                $"without hidden text: orders {string.Join(" ", page.Runs.Select(r => r.Order))}, {page.Lines.Count} lines");
        }

        using (var document = DocumentFormats.Open(samples["render-modes"].Path))
        {
            var page = document.ReadPage(0, DocumentParts.Text);
            Check(page.Runs.Count == 8 && page.Runs[3].Mode == DocumentTextMode.Hidden && page.Runs[7].Mode == DocumentTextMode.Hidden && page.Runs[1].Mode == DocumentTextMode.Stroke
                  && page.Runs[2].Mode == DocumentTextMode.FillStroke && page.Runs[4].Mode == DocumentTextMode.Fill, $"render modes: {string.Join(" ", page.Runs.Select(r => r.Mode))}");
        }

        // The word gap: TJ offsets of 0.32 em are word breaks at the default share (0.2), not at 0.5.
        foreach (var (gap, line) in new[] { (0.2f, "Annual budget review"), (0.5f, "Annualbudgetreview") })
        {
            using var document = DocumentFormats.Open(samples["text-state"].Path, new DocumentOptions { WordGap = gap });
            var page = document.ReadPage(0, DocumentParts.Text);
            Check(page.Lines[1].Text == line && page.Lines[0].Text == "Words with kerning", $"word gap {gap}: \"{page.Lines[1].Text}\", \"{page.Lines[0].Text}\"");
        }

        // The content layout: one line a run, in content order, the runs' text joined.
        using (var document = DocumentFormats.Open(samples["mixed-direction"].Path, new DocumentOptions { Layout = DocumentTextLayouts.Content }))
        {
            var page = document.ReadPage(0, DocumentParts.Text);
            Check(page.Lines.Count == page.Runs.Count && page.Lines.Select(l => l.Text).SequenceEqual(page.Runs.Select(r => r.Text))
                  && page.Lines.Select(l => l.Runs.Single()).SequenceEqual(Enumerable.Range(0, page.Runs.Count)) && page.Text == string.Concat(page.Runs.Select(r => r.Text)),
                $"the content layout: {page.Lines.Count} lines for {page.Runs.Count} runs: {string.Join(" | ", page.Lines.Select(l => l.Text))}");
        }

        // Glyphs when asked: an embedded program read as a face (its glyph ids), none for the standard 14 (the codes).
        using (var document = DocumentFormats.Open(samples["type0-identity"].Path))
        {
            var plain = document.ReadPage(0, DocumentParts.Text);
            Check(plain.Runs.All(r => r.Glyphs.IsEmpty && r.Face is null), "no glyphs or faces when text alone is asked");
            var page = document.ReadPage(0, DocumentParts.Text | DocumentParts.Glyphs);
            foreach (var run in page.Runs)
            {
                var glyphs = run.Glyphs.Span;
                Check(run.Face is { } face && face.Family.Contains("DejaVu", StringComparison.Ordinal) && glyphs.Length > 0 && glyphs.Length <= run.Text.Length
                      && glyphs.ToArray().All(g => g.Glyph > 0 && g.Glyph < face.GlyphCount && g.Advance > 0 && g.X >= run.Box.X - 0.01f && g.X <= run.Box.Right + 0.01f),
                    $"type0-identity: \"{run.Text}\": face {run.Face?.Family}, {glyphs.Length} glyphs");
                float right = glyphs[^1].X + glyphs[^1].Advance;
                Check(MathF.Abs(glyphs[0].X - run.Box.X) < 0.01f && MathF.Abs(right - run.Box.Right) < 0.01f, $"\"{run.Text}\": the glyphs span {glyphs[0].X} to {right}, the box {run.Box}");
            }
        }

        using (var document = DocumentFormats.Open(samples["standard14"].Path))
        {
            var page = document.ReadPage(0, DocumentParts.Text | DocumentParts.Glyphs);
            var first = page.Runs[0];
            Check(first.Face is null && first.Glyphs.Length == first.Text.Length && first.Glyphs.Span[0].Glyph == 'H', $"standard 14: face {first.Face?.Family}, {first.Glyphs.Length} glyphs");
        }
    }

    private static void PdfTextHandMade(Device device)
    {
        CpuOnly(device);

        // Glyph names to text: the list, uniXXXX (several), uXXXXX, components, suffixes, the Zapf Dingbats list for that font only.
        char[] chars = new char[PdfFontTables.MaxGlyphText];
        string Name(string glyph, bool zapf = false) => new(chars, 0, PdfFontTables.GlyphText(Encoding.ASCII.GetBytes(glyph), zapf, chars));
        Check(Name("Euro") == "€" && Name("uni2192") == "→" && Name("uni00410042") == "AB" && Name("u1F4C8") == "\U0001F4C8" && Name("f_f") == "ff" && Name("a.sc") == "a"
              && Name("g17") == "" && Name(".notdef") == "" && Name("uni20ac") == "" && Name("uniD800") == "" && Name("xx1_A") == "A" && Name("space") == " "
              && Name("a20", zapf: true) == "✔" && Name("a20") == "" && Name("T_h.alt") == "Th",
            "glyph names by the Adobe Glyph List's rules");

        // The standard fonts, and the names files use for them.
        Check(PdfFontTables.StandardFontIndex("Helvetica") == Array.IndexOf(PdfFontTables.StandardFonts, "Helvetica")
              && PdfFontTables.StandardFontIndex("ABCDEF+Arial,Bold") == Array.IndexOf(PdfFontTables.StandardFonts, "Helvetica-Bold")
              && PdfFontTables.StandardFontIndex("TimesNewRomanPS-BoldItalicMT") == Array.IndexOf(PdfFontTables.StandardFonts, "Times-BoldItalic")
              && PdfFontTables.StandardFontIndex("CourierNew,Italic") == Array.IndexOf(PdfFontTables.StandardFonts, "Courier-Oblique")
              && PdfFontTables.StandardFontIndex("DejaVuSans") == -1 && PdfFontTables.StandardFontIndex(null) == -1, "standard font names");
        int helvetica = PdfFontTables.StandardFontIndex("Helvetica");
        Check(PdfFontTables.StandardWidth(helvetica, 'H') == 722 && PdfFontTables.StandardWidth(helvetica, ' ') == 278 && PdfFontTables.StandardExtents(helvetica) == (718, -207)
              && PdfFontTables.StandardWidth(PdfFontTables.ZapfDingbats, ' ') == 278 && PdfFontTables.StandardWidth(PdfFontTables.ZapfDingbats, '✔') == 846
              && PdfFontTables.EncodingName(PdfFontTables.WinAnsiEncoding, 0x80).SequenceEqual("Euro"u8) && PdfFontTables.EncodingName(PdfFontTables.SymbolEncoding, 0x61).SequenceEqual("alpha"u8)
              && PdfFontTables.EncodingName(PdfFontTables.StandardEncoding, 0x80).IsEmpty, "the standard fonts' metrics and the encodings");

        // CMaps: a mixed code space, bfchar, bfrange by a start and by an array, cidrange, notdef.
        var map = PdfCMap.Parse(Encoding.ASCII.GetBytes(
            "/CIDInit /ProcSet findresource begin 12 dict begin begincmap /WMode 1 def\n2 begincodespacerange <00> <7F> <8000> <FFFF> endcodespacerange\n"
            + "2 beginbfchar <41> <0042> <8001> <00660069> endbfchar\n2 beginbfrange <61> <63> <0061> <8010> <8012> [<0031> <00320033> <0034>] endbfrange\n"
            + "1 begincidrange <8000> <80FF> 100 endcidrange 1 beginnotdefrange <00> <1F> 1 endnotdefrange endcmap end end"), null, vertical: false);
        string Unicode(uint code, int length)
        {
            Span<char> text = stackalloc char[16];
            return new string(text[..map.Unicode(code, length, text)]);
        }

        byte[] codes = [0x41, 0x80, 0x01, 0x62, 0x80, 0x11];
        var split = new List<(uint Code, int Length)>();
        for (int at = 0; at < codes.Length;)
        {
            int length = map.NextCode(codes, at, out uint code);
            split.Add((code, length));
            at += length;
        }

        Check(map.Vertical && split.SequenceEqual([(0x41u, 1), (0x8001u, 2), (0x62u, 1), (0x8011u, 2)]), $"code space: {string.Join(" ", split)}");
        Check(Unicode(0x41, 1) == "B" && Unicode(0x8001, 2) == "fi" && Unicode(0x62, 1) == "b" && Unicode(0x63, 1) == "c" && Unicode(0x8011, 2) == "23" && Unicode(0x8012, 2) == "4"
              && Unicode(0x64, 1) == "" && Unicode(0x41, 2) == "", "bfchar, bfrange by a start and by an array");
        Check(map.Cid(0x8005, 2) == 105 && map.Cid(0x05, 1) == 1 && map.Cid(0x41, 1) == 0 && PdfCMap.Identity(true).Cid(0x1234, 2) == 0x1234, "cidrange, notdef, identity");

        // A hand-made page: a font the page lacks, an inline image whose bytes hold " EIQ", a span named in /Properties, an
        // unknown operator, a path, a TJ word gap.
        string content = "BT /F9 12 Tf 72 700 Td (No such font) Tj ET\n"
                         + "q 40 0 0 10 100 100 cm BI /W 4 /H 1 /BPC 8 /CS /G ID  EIQ\nEI Q\n"
                         + "BT /F1 12 Tf 72 650 Td (After the image) Tj ET\n"
                         + "/P <</MCID 0>> BDC BT /F1 12 Tf 72 600 Td (Tagged) Tj ET EMC\n"
                         + "/Span /MC0 BDC BT /F1 12 Tf 72 550 Td (x) Tj ET EMC\n"
                         + "1 2 3 frobnicate 0 0 m 10 10 l S 5 5 m 6 6 l n\n"
                         + "BT /F1 12 Tf 72 500 Td [(Word) -500 (gap)] TJ ET\n";
        byte[] file = PdfHandMadeFile([
            "<< /Type /Catalog /Pages 2 0 R >>",
            "<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
            "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R /Resources << /Font << /F1 5 0 R >> /Properties << /MC0 << /ActualText (Named span) >> >> >> >>",
            PdfHandMadeStream("", Encoding.Latin1.GetBytes(content)),
            "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /Encoding /WinAnsiEncoding >>",
        ]);
        using var document = DocumentFormats.Open(file.AsMemory(), "hand-made.pdf");
        var page = document.ReadPage(0, DocumentParts.Text);
        Check(page.Runs.Select(r => r.Text).SequenceEqual(["No such font", "After the image", "Tagged", "Named span", "Wordgap"])
              && page.Runs.Select(r => r.Order).SequenceEqual([0, 2, 3, 4, 6]) && page.Runs[3].Logical,
            $"the hand-made page's runs: {string.Join(" | ", page.Runs.Select(r => $"{r.Order} {r.Text}"))}");
        Check(page.Text == "No such font\nAfter the image\nTagged\nNamed span\nWord gap", $"the hand-made page's text: \"{page.Text}\"");
        Check(page.Issues.Count == 1 && page.Issues[0].Kind == DocumentIssue.MissingFont && page.Issues[0].Detail.Contains("/F9", StringComparison.Ordinal),
            $"the font the page lacks: {string.Join("; ", page.Issues.Select(i => $"{i.Kind}: {i.Detail}"))}");
        float helveticaWidth = 12 * "No such font".Sum(c => PdfFontTables.StandardWidth(helvetica, c)) / 1000f;
        Check(MathF.Abs(page.Runs[0].Box.Width - helveticaWidth) < 0.01f, $"a font the page lacks is read as Helvetica: {page.Runs[0].Box}, expected {helveticaWidth} wide");
    }
}
