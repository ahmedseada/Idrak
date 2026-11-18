// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Buffers.Binary;
using System.Collections.Concurrent;
using System.Text;
using Idrak.Abstraction.Testing;
using Idrak.Imaging;
using Idrak.PluginTests;

// Plan 38, phase F0: the handwriting contracts (outline tracers and glyph sources, their registries, values and suites),
// checked on the outside toys (tests/Idrak.PluginTests) and on broken ones of the test's own; and the face made from
// glyphs (MadeFontFace), read back, shaped and read from several threads. Its export to .ttf waits for plan 28's
// TrueType writer (F3 tests it).
internal abstract partial class Tests
{
    internal static readonly (string Name, Action<Device> Run)[] OutlineTracerGroup =
    [
        ("outline tracers: a tracer of the test's own registers under its name, is found by name in any case and by span, lists its keys and summary, unregisters; a name nobody registered is refused naming the registry, the registered ones and what to do", OutlineTracerRegistry),
        ("outline tracers: coverage is checked on the way in (its size, every value from 0 to 1, naming the pixel), finds its ink's box in the same pass and crops a part; a placement maps pixels and font units both ways", GlyphCoverageValues),
        ("outline tracers: the suite passes on the outside toy tracer (its defaults, another level, DejaVu's glyphs) and fails a tracer that fills holes, one that traces every run twice, one that inks empty coverage, one that leaves y down and one that takes any option", OutlineTracerSuiteCatches),
    ];

    internal static readonly (string Name, Action<Device> Run)[] GlyphSourceGroup =
    [
        ("glyph sources: a source of the test's own registers under its name, is found by name in any case and by span, lists its keys and summary, unregisters; a name nobody registered is refused naming the registry, the registered ones and what to do", GlyphSourceRegistry),
        ("glyph sources: a labelled glyph is checked as it is made (its text, a form as data with no spaces, ink its box's size, a finite baseline, a size above 0) and gives a tracer's placement from its baseline and size; a request needs an image and settings", LabelledGlyphValues),
        ("glyph sources: the suite passes on the outside toy source (grey and colour pages, a character with no ink reported) and fails a source that labels a glyph twice, one that puts a glyph outside its region, one that drops a character unreported, one that answers differently each time and one that takes any option", GlyphSourceSuiteCatches),
    ];

    internal static readonly (string Name, Action<Device> Run)[] MadeFontGroup =
    [
        ("made fonts: a face made from glyphs gives back its characters, outlines (copies of those given), advances, metrics and names; glyphs with no character and characters sharing a glyph; the missing glyph given, or empty at half an em; its one table the name table of the names given, no layout tables", MadeFaceReadsBack),
        ("made fonts: generic text shaped through TextShapers gives the made glyphs with their advances, a character the face lacks the missing glyph", MadeFaceShapes),
        ("made fonts: the face answers the same from several threads at once (characters, advances, outlines, its table, shaping)", MadeFaceThreads),
        ("made fonts: the builder refuses what a font cannot hold, naming it (no family, units per em, the ascender and descender, an advance, a point not finite or too far, a contour without its move, a character twice or not a character, a glyph not added, a name with no text) and stays as it was", MadeFaceRefusals),
        ("made fonts: the outside toy source's glyphs, traced by the outside toy tracer at their baseline and size, make a face whose text shapes to them, each as wide as its ink", MadeFaceFromToys),
    ];

    private const string LetterText = "Thank you for your letter of 12 May.";

    internal static void OutlineTracerRegistry(Device device)
    {
        CpuOnly(device);
        var mine = new ToyOutlineTracer("test-runs");
        OutlineTracers.Register(mine);
        try
        {
            const string Line = "--tracer=test-runs";
            Check(ReferenceEquals(OutlineTracers.Get("TEST-RUNS"), mine) && ReferenceEquals(OutlineTracers.Find(Line.AsSpan(9)), mine) && OutlineTracers.Names.Contains("test-runs")
                  && OutlineTracers.Origin("test-runs") == typeof(Tests).Assembly.GetName().Name && OutlineTracers.Default("test-runs") is null
                  && OutlineTracers.Describe().Contains("test-runs: each row's runs of ink", StringComparison.Ordinal) && OutlineTracers.AllKeys().Contains(ToyOutlineTracer.LevelKey),
                $"registered: {OutlineTracers.Describe()}");
        }
        finally
        {
            Check(OutlineTracers.Unregister("test-runs") && !OutlineTracers.Unregister("test-runs") && OutlineTracers.Find("test-runs") is null, "unregistered once");
        }

        try
        {
            OutlineTracers.Get("a-tracer-nobody-registered");
            Check(false, "a name nobody registered was found");
        }
        catch (NotRegisteredException e)
        {
            Check(e.Registry == nameof(OutlineTracers) && e.Name == "a-tracer-nobody-registered" && e.Registered.SequenceEqual(OutlineTracers.Names)
                  && e.Message.StartsWith($"Outline tracer 'a-tracer-nobody-registered' is not registered (registered: {(OutlineTracers.Names.Count == 0 ? "none" : string.Join(", ", OutlineTracers.Names))})", StringComparison.Ordinal)
                  && e.Message.Contains("OutlineTracers.Register", StringComparison.Ordinal) && e.Message.Contains("reference a package", StringComparison.Ordinal), e.Message);
        }
    }

    internal static void GlyphCoverageValues(Device device)
    {
        CpuOnly(device);
        ThrowsNaming<ArgumentException>(() => new GlyphCoverage(new float[5], 2, 3), "values of another size", "5 values are not 2 x 3");
        ThrowsNaming<ArgumentException>(() => new GlyphCoverage([0f, 1.5f, 0f, 0f], 2, 2), "a value past 1", "pixel (1, 0) holds 1.5");
        ThrowsNaming<ArgumentException>(() => new GlyphCoverage([0f, 0f, float.NaN, 0f], 2, 2), "a value not a number", "pixel (0, 1) holds NaN");
        ThrowsNaming<ArgumentException>(() => new GlyphCoverage([0f, -0.25f], 2, 1), "a value below 0", "pixel (1, 0) holds -0.25");

        float[] values =
        [
            0, 0, 0, 0,
            0, 0.5f, 0, 0,
            0, 0, 1, 0,
        ];
        var coverage = new GlyphCoverage(values, 4, 3);
        Check(coverage.InkBox == new PixelBox(1, 1, 2, 2) && !coverage.IsBlank && coverage[2, 2] == 1f && coverage.Row(1)[1] == 0.5f && coverage.Values.Length == 12,
            $"the ink's box {coverage.InkBox}");
        var part = coverage.Crop(new PixelBox(1, 1, 2, 2));
        Check(part.Width == 2 && part.Height == 2 && part.Values.Span.SequenceEqual([0.5f, 0f, 0f, 1f]) && part.InkBox == new PixelBox(0, 0, 2, 2), "a part cropped");
        ThrowsNaming<ArgumentOutOfRangeException>(() => coverage.Crop(new PixelBox(3, 0, 2, 1)), "a crop past the edge", "reaches outside");
        ThrowsNaming<ArgumentOutOfRangeException>(() => _ = coverage[4, 0], "a pixel past the edge", "(4, 0)");
        var blank = GlyphCoverage.Blank(3, 2);
        Check(blank.IsBlank && blank.InkBox == default && blank.Width == 3 && blank.Height == 2, "blank coverage");

        var at = new OutlinePlacement(0.05f, 3f, 40f);
        var (ux, uy) = at.ToUnits(13f, 20f);
        var (px, py) = at.ToPixels(ux, uy);
        Check(MathF.Abs(ux - 200f) < 1e-3f && MathF.Abs(uy - 400f) < 1e-3f && MathF.Abs(px - 13f) < 1e-4f && MathF.Abs(py - 20f) < 1e-4f && at.IsValid,
            $"pixels to units and back: ({ux}, {uy}), ({px}, {py})");
        Check(!new OutlinePlacement(0f, 0f, 0f).IsValid && !new OutlinePlacement(1f, float.NaN, 0f).IsValid && !new OutlinePlacement(float.PositiveInfinity, 0f, 0f).IsValid,
            "a placement with no scale, or not finite, is not valid");
    }

    internal static void OutlineTracerSuiteCatches(Device device)
    {
        CpuOnly(device);
        var toy = new ToyOutlineTracer();
        var face = SubsetFace();
        var outlines = "aegB8%&".Select(c => face.Outline(face.Glyph(c))).ToList();
        Conformance.CheckOutlineTracer(toy, outlines, face.Metrics.UnitsPerEm, [PageOptionsOf((ToyOutlineTracer.LevelKey, "0.3"))]).ThrowIfFailed();

        void Fails(IOutlineTracer broken, string property)
        {
            var report = Conformance.CheckOutlineTracer(broken);
            Check(!report.Passed && report.Failures.Any(f => f.Check == property), $"{property} is not caught:\n{report}");
        }

        Fails(new BrokenTracer(toy, "fills holes"), "holes kept");
        Fails(new BrokenTracer(toy, "twice"), "one winding");
        Fails(new BrokenTracer(toy, "inks blank"), "coverage with no ink gives an empty outline");
        Fails(new BrokenTracer(toy, "y down"), "drawn and traced back within the promised tolerance");
        Fails(new BrokenTracer(toy, "greedy"), "an option it does not take is refused");
    }

    internal static void GlyphSourceRegistry(Device device)
    {
        CpuOnly(device);
        var mine = new ToyGlyphSource("test-columns");
        GlyphSources.Register(mine);
        try
        {
            const string Line = "--source=test-columns";
            Check(ReferenceEquals(GlyphSources.Get("Test-Columns"), mine) && ReferenceEquals(GlyphSources.Find(Line.AsSpan(9)), mine) && GlyphSources.Names.Contains("test-columns")
                  && GlyphSources.Origin("test-columns") == typeof(Tests).Assembly.GetName().Name && GlyphSources.Default("test-columns") is null
                  && GlyphSources.Describe().Contains("test-columns: the ink between blank columns", StringComparison.Ordinal) && GlyphSources.AllKeys().Contains(ToyGlyphSource.LevelKey),
                $"registered: {GlyphSources.Describe()}");
        }
        finally
        {
            Check(GlyphSources.Unregister("test-columns") && !GlyphSources.Unregister("test-columns") && GlyphSources.Find("test-columns") is null, "unregistered once");
        }

        try
        {
            GlyphSources.Get("a-source-nobody-registered");
            Check(false, "a name nobody registered was found");
        }
        catch (NotRegisteredException e)
        {
            Check(e.Registry == nameof(GlyphSources) && e.Name == "a-source-nobody-registered" && e.Registered.SequenceEqual(GlyphSources.Names)
                  && e.Message.StartsWith($"Glyph source 'a-source-nobody-registered' is not registered (registered: {(GlyphSources.Names.Count == 0 ? "none" : string.Join(", ", GlyphSources.Names))})", StringComparison.Ordinal)
                  && e.Message.Contains("GlyphSources.Register", StringComparison.Ordinal) && e.Message.Contains("reference a package", StringComparison.Ordinal), e.Message);
        }
    }

    internal static void LabelledGlyphValues(Device device)
    {
        CpuOnly(device);
        var ink = new GlyphCoverage([0f, 1f, 1f, 0.5f, 1f, 0f], 2, 3);
        var box = new PixelBox(10, 4, 2, 3);
        var region = new PixelBox(8, 0, 6, 12);
        var glyph = new LabelledGlyph("7", "isol", ink, box, region, 7f, 20f);
        var at = glyph.Placement(1000);
        Check(glyph.Text == "7" && glyph.Form == "isol" && ReferenceEquals(glyph.Ink, ink) && glyph.Box == box && glyph.Region == region
              && at == new OutlinePlacement(0.02f, 0f, 3f) && glyph.ToString().Contains("'7' (isol)", StringComparison.Ordinal), $"the glyph: {glyph}, placed {at}");
        Check(new LabelledGlyph("ab", null, ink, box, region, 0f, 1f).Form is null, "a glyph with no form, of two letters (a ligature)");

        ThrowsNaming<ArgumentException>(() => new LabelledGlyph("", null, ink, box, region, 7f, 20f), "no text", "text");
        ThrowsNaming<ArgumentException>(() => new LabelledGlyph("7", "in it", ink, box, region, 7f, 20f), "a form with a space", "'in it'");
        ThrowsNaming<ArgumentException>(() => new LabelledGlyph("7", "", ink, box, region, 7f, 20f), "an empty form", "joining form");
        ThrowsNaming<ArgumentException>(() => new LabelledGlyph("7", null, ink, new PixelBox(10, 4, 3, 3), region, 7f, 20f), "ink not its box's size", "2 x 3 pixels", "3 x 3");
        ThrowsNaming<ArgumentException>(() => new LabelledGlyph("7", null, ink, box, region, float.NaN, 20f), "a baseline not a number", "baseline");
        ThrowsNaming<ArgumentException>(() => new LabelledGlyph("7", null, ink, box, region, 7f, 0f), "a size of 0", "pixels per em");
        ThrowsNaming<ArgumentException>(() => new LabelledGlyph("7", null, ink, box, region, 7f, float.PositiveInfinity), "an endless size", "pixels per em");
        ThrowsNaming<ArgumentOutOfRangeException>(() => glyph.Placement(0), "a face of no units", "unitsPerEm");

        Check(FoundGlyphs.None.Glyphs.Count == 0 && FoundGlyphs.None.LeftOut.Count == 0, "nothing found");
        var image = new ImageData(new float[4], 1, 2, 2);
        ThrowsNaming<ArgumentNullException>(() => _ = new GlyphSourceRequest(null!, "7", GlyphSourceSettings.None), "a request with no image", "Image");
        ThrowsNaming<ArgumentNullException>(() => _ = new GlyphSourceRequest(image, "7", null!), "a request with no settings", "Settings");
        Check(new GlyphSourceRequest(image, null, GlyphSourceSettings.None).Text is null && GlyphSourceSettings.None.Options.Count == 0 && OutlineTracerSettings.None.Options.Count == 0,
            "a request with no known text; settings of no options");
    }

    internal static void GlyphSourceSuiteCatches(Device device)
    {
        CpuOnly(device);
        var samples = HandwritingPluginTests.Samples();
        var toy = new ToyGlyphSource();
        Conformance.CheckGlyphSource(toy, samples, new ContractCheckOptions { RandomCases = 10 }).ThrowIfFailed();

        void Fails(IGlyphSource broken, string property)
        {
            var report = Conformance.CheckGlyphSource(broken, samples);
            Check(!report.Passed && report.Failures.Any(f => f.Check == property), $"{property} is not caught:\n{report}");
        }

        Fails(new BrokenSource(toy, "twice"), "every glyph labelled once");
        Fails(new BrokenSource(toy, "outside"), "every glyph inside its cell or its ink piece");
        Fails(new BrokenSource(toy, "drops"), "the known text's characters all accounted for or reported");
        Fails(new BrokenSource(toy, "unstable"), "the same glyphs again and from several threads");
        Fails(new BrokenSource(toy, "greedy"), "an option it does not take is refused");
    }

    internal static void MadeFaceReadsBack(Device device)
    {
        CpuOnly(device);
        var font = SubsetFace();
        var builder = new MadeFontFaceBuilder("A Hand", font.Metrics).Name(FontNameKind.Style, "Written").Name(FontNameKind.Designer, "A writer");
        int[] codes = [.. LetterText.EnumerateRunes().Select(r => r.Value).Distinct()];
        foreach (int code in codes)
        {
            int g = font.Glyph(code);
            builder.Add(code, font.Outline(g), font.Advance(g));
        }

        // A glyph from arrays the test keeps changing after: the face keeps what it was given.
        OutlineVerb[] verbs = [OutlineVerb.Move, OutlineVerb.Line, OutlineVerb.Line, OutlineVerb.Close];
        float[] points = [0f, 0f, 500f, 0f, 250f, 700f];
        int triangle = builder.Add('^', new GlyphOutline(verbs, points), 600);
        int alternate = builder.Add(font.Outline(font.Glyph('a')), font.Advance(font.Glyph('a')));   // a second fill of 'a', with no character
        builder.Map('A', triangle);                                                                      // two characters, one glyph
        var made = builder.Build();
        points[0] = 99f;
        verbs[1] = OutlineVerb.Cubic;

        Check(made.Family == "A Hand" && made.Style == "Written" && made.Format == MadeFontFace.FormatName && made.Metrics == font.Metrics
              && made.GlyphCount == codes.Length + 3 && made.Names[FontNameKind.Designer] == "A writer" && made.Names.Keys.SequenceEqual([FontNameKind.Family, FontNameKind.Style, FontNameKind.Designer]),
            $"the face: {made}, names {string.Join(", ", made.Names)}");
        Check(made.Characters.SequenceEqual(codes.Append('^').Append('A').Order()), $"the characters: {string.Join(" ", made.Characters)}");
        foreach (int code in codes)
        {
            int had = font.Glyph(code), glyph = made.Glyph(code);
            Check(glyph > 0 && made.Advance(glyph) == font.Advance(had) && made.Outline(glyph).Verbs.Span.SequenceEqual(font.Outline(had).Verbs.Span)
                  && made.Outline(glyph).Points.Span.SequenceEqual(font.Outline(had).Points.Span), $"U+{code:X4}: glyph {glyph}, its advance and outline");
        }

        Check(made.Glyph('^') == triangle && made.Glyph('A') == triangle && made.Outline(triangle).Points.Span[0] == 0f && made.Outline(triangle).Verbs.Span[1] == OutlineVerb.Line
              && made.Advance(triangle) == 600, "a glyph keeps what it was given, not what its arrays hold after");
        Check(made.Outline(alternate).Points.Span.SequenceEqual(font.Outline(font.Glyph('a')).Points.Span) && !made.Characters.Any(c => made.Glyph(c) == alternate),
            "a glyph with no character, reached by its id");
        Check(made.Glyph('Z') == 0 && made.Glyph(-1) == 0 && made.Advance(-1) == 0 && made.Advance(made.GlyphCount) == 0 && made.Outline(made.GlyphCount).IsEmpty,
            "a character or glyph the face does not have");
        Check(made.Outline(0).IsEmpty && made.Advance(0) == font.Metrics.UnitsPerEm / 2, $"the missing glyph, not given: empty, {made.Advance(0)} wide");
        var square = new GlyphOutline([OutlineVerb.Move, OutlineVerb.Line, OutlineVerb.Line, OutlineVerb.Line, OutlineVerb.Close], [50f, 0f, 450f, 0f, 450f, 700f, 50f, 700f]);
        var withMissing = builder.Missing(square, 500).Build();
        Check(withMissing.Advance(0) == 500 && withMissing.Outline(0).Points.Length == 8 && made.Advance(0) == font.Metrics.UnitsPerEm / 2, "the missing glyph given; the face built before keeps its own");

        // The name table: format 0, Windows Unicode English records of the names given, sorted by id.
        var table = made.Table("name").Span;
        int count = BinaryPrimitives.ReadUInt16BigEndian(table[2..]), storage = BinaryPrimitives.ReadUInt16BigEndian(table[4..]);
        var names = new List<(int Id, string Text)>();
        for (int i = 0; i < count; i++)
        {
            var record = table.Slice(6 + 12 * i, 12);
            Check(BinaryPrimitives.ReadUInt16BigEndian(record) == 3 && BinaryPrimitives.ReadUInt16BigEndian(record[2..]) == 1 && BinaryPrimitives.ReadUInt16BigEndian(record[4..]) == 0x409,
                $"record {i}: Windows, Unicode, English");
            int length = BinaryPrimitives.ReadUInt16BigEndian(record[8..]), offset = BinaryPrimitives.ReadUInt16BigEndian(record[10..]);
            names.Add((BinaryPrimitives.ReadUInt16BigEndian(record[6..]), Encoding.BigEndianUnicode.GetString(table.Slice(storage + offset, length))));
        }

        Check(BinaryPrimitives.ReadUInt16BigEndian(table) == 0 && storage == 6 + 12 * count
              && names.SequenceEqual([(1, "A Hand"), (2, "Written"), (9, "A writer")]), $"the name table: {string.Join(", ", names)}");
        Check(made.Table("GSUB").IsEmpty && made.Table("GPOS").IsEmpty && made.Table("GDEF").IsEmpty && made.Table("cmap").IsEmpty && made.Table("NAME").IsEmpty,
            "no layout tables, nothing else made up");
        var plain = new MadeFontFaceBuilder("Plain", new FontMetrics(1000, 800, -200, 0)).Build();
        Check(plain.Style == "Regular" && plain.GlyphCount == 1 && plain.Characters.Count == 0 && plain.Names.Count == 1, $"a face of no glyphs, no style given: {plain}");
    }

    internal static void MadeFaceShapes(Device device)
    {
        CpuOnly(device);
        var made = LetterFace(leaveOut: 'M');
        foreach (string text in new[] { LetterText, "The report is attached: 12 pages." })
        {
            var shaped = TextShapers.Shape(made, text, new TextRun(0, text.Length, "Latn", false));
            Check(shaped.Length == text.Length, $"{shaped.Length} glyphs for the {text.Length} characters of '{text}'");
            for (int i = 0; i < text.Length; i++)
            {
                int glyph = made.Glyph(text[i]);
                Check(shaped[i].Glyph == glyph && shaped[i].Cluster == i && shaped[i].XAdvance == made.Advance(glyph) && shaped[i].XOffset == 0 && shaped[i].YOffset == 0,
                    $"'{text[i]}' at {i}: {shaped[i]}, expected glyph {glyph} {made.Advance(glyph)} wide");
            }

            Check(text.Contains('M') == shaped.Any(s => s.Glyph == 0), $"a character the face lacks is the missing glyph: '{text}'");
        }
    }

    internal static void MadeFaceThreads(Device device)
    {
        CpuOnly(device);
        var made = LetterFace(leaveOut: null);
        string Answers()
        {
            var answer = new StringBuilder();
            for (int code = 0; code < 0x250; code++)
            {
                int glyph = made.Glyph(code);
                answer.Append(glyph).Append(',').Append(made.Advance(glyph)).Append(',').Append(made.Outline(glyph).Points.Length).Append(';');
            }

            for (int g = 0; g < made.GlyphCount; g++)
            {
                foreach (float p in made.Outline(g).Points.Span)
                {
                    answer.Append(BitConverter.SingleToInt32Bits(p)).Append(' ');
                }
            }

            answer.Append(Convert.ToBase64String(made.Table("name").Span));
            foreach (var s in TextShapers.Shape(made, LetterText, new TextRun(0, LetterText.Length, "Latn", false)))
            {
                answer.Append(s);
            }

            return answer.ToString();
        }

        string expected = Answers();
        var differing = new ConcurrentBag<int>();
        Parallel.For(0, 16, new ParallelOptions { MaxDegreeOfParallelism = Math.Max(4, Environment.ProcessorCount) }, i =>
        {
            if (Answers() != expected)
            {
                differing.Add(i);
            }
        });
        Check(differing.IsEmpty, $"{differing.Count} of 16 readers on several threads got other answers");
    }

    internal static void MadeFaceRefusals(Device device)
    {
        CpuOnly(device);
        var metrics = new FontMetrics(1000, 800, -200, 90);
        var line = new GlyphOutline([OutlineVerb.Move, OutlineVerb.Line, OutlineVerb.Line, OutlineVerb.Close], [0f, 0f, 400f, 0f, 200f, 600f]);
        ThrowsNaming<ArgumentException>(() => new MadeFontFaceBuilder(" ", metrics), "no family", "family name");
        ThrowsNaming<ArgumentException>(() => new MadeFontFaceBuilder("A Hand", metrics with { UnitsPerEm = 8 }), "eight units per em", "Units per em is from 16 to 16,384", "not 8");
        ThrowsNaming<ArgumentException>(() => new MadeFontFaceBuilder("A Hand", metrics with { Descender = 120 }), "a descender above the baseline", "descender 120");
        ThrowsNaming<ArgumentException>(() => new MadeFontFaceBuilder("A Hand", metrics with { Ascender = 0, Descender = 0 }), "both on the baseline", "not both on it");
        ThrowsNaming<ArgumentException>(() => new MadeFontFaceBuilder("A Hand", metrics with { LineGap = -5 }), "a line gap below 0", "line gap");

        var builder = new MadeFontFaceBuilder("A Hand", metrics);
        int first = builder.Add('7', line, 500);
        ThrowsNaming<ArgumentException>(() => builder.Add('8', line, -1), "an advance below 0", "advance is from 0 to 65,535", "not -1");
        ThrowsNaming<ArgumentException>(() => builder.Add('8', line, 70_000), "an advance past 16 bits", "not 70000");
        ThrowsNaming<ArgumentException>(() => builder.Add('8', new GlyphOutline([OutlineVerb.Move, OutlineVerb.Line], [0f, 0f, float.NaN, 5f]), 500), "a point not a number",
            "(NaN, 5)", "finite coordinates within ±32,767");
        ThrowsNaming<ArgumentException>(() => builder.Add('8', new GlyphOutline([OutlineVerb.Move, OutlineVerb.Line], [0f, 0f, 40_000f, 5f]), 500), "a point too far", "(40000, 5)");
        ThrowsNaming<ArgumentException>(() => builder.Add('8', new GlyphOutline([OutlineVerb.Line, OutlineVerb.Close], [0f, 0f]), 500), "a contour without its move",
            "step 1 (Line) before a move");
        ThrowsNaming<ArgumentException>(() => builder.Add('8', new GlyphOutline([OutlineVerb.Move, OutlineVerb.Close, OutlineVerb.Line], [0f, 0f, 5f, 5f]), 500),
            "a step after a close", "step 3 (Line) before a move");
        ThrowsNaming<ArgumentException>(() => builder.Add('7', line, 500), "a character twice", "U+0037 is given twice", $"glyph {first}");
        ThrowsNaming<ArgumentException>(() => builder.Add(0xD800, line, 500), "a surrogate", "U+D800 is not a character");
        ThrowsNaming<ArgumentException>(() => builder.Add(0x110000, line, 500), "past U+10FFFF", "is not a character");
        ThrowsNaming<ArgumentException>(() => builder.Map('9', 42), "a glyph not added", "Glyph 42 is not added yet");
        ThrowsNaming<ArgumentException>(() => builder.Map('9', 0), "the missing glyph", "missing-character glyph");
        ThrowsNaming<ArgumentException>(() => builder.Map('7', first), "a character given before", "U+0037 is given twice");
        ThrowsNaming<ArgumentException>(() => builder.Name(FontNameKind.Designer, " "), "a name with no text", "Designer needs text");
        ThrowsNaming<ArgumentException>(() => builder.Name((FontNameKind)18, "x"), "no kind of name", "18 is not a kind of font name");
        ThrowsNaming<ArgumentException>(() => builder.Missing(line, -3), "a missing glyph's advance", "The missing glyph's advance");
        ThrowsNaming<ArgumentException>(() => builder.Name(FontNameKind.Description, new string('x', 40_000)).Build(), "names too long for a name table", "a name table holds at most 65,535");

        var made = builder.Name(FontNameKind.Description, "A hand for letters.").Build();
        Check(made.GlyphCount == 2 && made.Characters.SequenceEqual([(int)'7']) && made.Glyph('8') == 0 && made.Glyph('9') == 0 && made.Advance(0) == 500
              && made.Names[FontNameKind.Description] == "A hand for letters.", $"refused input left the builder as it was: {made}");
    }

    internal static void MadeFaceFromToys(Device device)
    {
        CpuOnly(device);
        var sample = HandwritingPluginTests.Samples()[0];
        var source = new ToyGlyphSource();
        var tracer = new ToyOutlineTracer();
        var found = source.Find(new GlyphSourceRequest(sample.Image, sample.Text, source.Read(PageOptions.None)));
        var builder = new MadeFontFaceBuilder("A Toy Hand", new FontMetrics(1000, 1000, -250, 0)).Name(FontNameKind.Style, "Regular");
        var settings = tracer.Read(PageOptions.None);
        foreach (var glyph in found.Glyphs)
        {
            var outline = tracer.Trace(glyph.Ink, glyph.Placement(1000), settings);
            builder.Add(char.ConvertToUtf32(glyph.Text, 0), outline, (int)MathF.Round(glyph.Box.Width * 1000f / glyph.Size));
        }

        var made = builder.Build();
        const string Text = "1074";
        var shaped = TextShapers.Shape(made, Text, new TextRun(0, Text.Length, "Latn", false));
        Check(shaped.Select(s => s.Glyph).SequenceEqual([1, 2, 3, 4]), $"shaped: {string.Join(" ", shaped)}");
        for (int i = 0; i < Text.Length; i++)
        {
            var glyph = found.Glyphs[i];
            var (minX, minY, maxX, maxY) = made.Outline(shaped[i].Glyph).Bounds();
            Check(shaped[i].XAdvance == (int)MathF.Round(glyph.Box.Width * 1000f / glyph.Size) && minX >= -0.5f && maxX <= shaped[i].XAdvance + 1f && minY >= -0.5f && maxY <= 1000.5f,
                $"'{Text[i]}': {shaped[i]}, its outline spans ({minX}, {minY}) to ({maxX}, {maxY})");
        }
    }

    // A face made from DejaVu's glyphs of the letter's text and the second sample line's, without the character leaveOut.
    private static MadeFontFace LetterFace(char? leaveOut)
    {
        var font = SubsetFace();
        var builder = new MadeFontFaceBuilder("A Letter Hand", font.Metrics);
        foreach (int code in (LetterText + "The report is attached: 12 pages.").EnumerateRunes().Select(r => r.Value).Distinct())
        {
            if (code != leaveOut)
            {
                int g = font.Glyph(code);
                builder.Add(code, font.Outline(g), font.Advance(g));
            }
        }

        return builder.Build();
    }

    // The outside toy tracer broken one way, to prove the suite catches it.
    private sealed class BrokenTracer(ToyOutlineTracer toy, string how) : IOutlineTracer
    {
        public string Name => $"broken ({how})";

        public string Summary => $"the toy tracer, broken: {how}";

        public IReadOnlySet<string> Keys => toy.Keys;

        public OutlineTracerSettings Read(IReadOnlyDictionary<string, string> options) =>
            how == "greedy" ? toy.Read(PageOptions.None) : toy.Read(options);

        public float Tolerance(OutlineTracerSettings settings) => toy.Tolerance(settings);

        public GlyphOutline Trace(GlyphCoverage coverage, OutlinePlacement placement, OutlineTracerSettings settings)
        {
            switch (how)
            {
                case "fills holes":
                {
                    // Each row inked from its first ink to its last.
                    var values = coverage.Values.ToArray();
                    for (int y = 0; y < coverage.Height; y++)
                    {
                        var row = values.AsSpan(y * coverage.Width, coverage.Width);
                        int first = row.IndexOfAnyExcept(0f), last = row.LastIndexOfAnyExcept(0f);
                        if (first >= 0)
                        {
                            row[first..(last + 1)].Fill(1f);
                        }
                    }

                    return toy.Trace(new GlyphCoverage(values, coverage.Width, coverage.Height), placement, settings);
                }

                case "twice":
                {
                    var once = toy.Trace(coverage, placement, settings);
                    return once.IsEmpty ? once : new GlyphOutline([.. once.Verbs.Span, .. once.Verbs.Span], [.. once.Points.Span, .. once.Points.Span]);
                }

                case "inks blank":
                    return coverage.IsBlank
                        ? new GlyphOutline([OutlineVerb.Move, OutlineVerb.Line, OutlineVerb.Line, OutlineVerb.Close], [0f, 0f, 100f, 0f, 50f, 100f])
                        : toy.Trace(coverage, placement, settings);
                case "y down":
                {
                    var traced = toy.Trace(coverage, placement, settings);
                    float[] points = traced.Points.ToArray();
                    for (int i = 1; i < points.Length; i += 2)
                    {
                        points[i] = -points[i];
                    }

                    return traced.IsEmpty ? traced : new GlyphOutline(traced.Verbs.ToArray(), points);
                }

                default:
                    return toy.Trace(coverage, placement, settings);
            }
        }
    }

    // The outside toy source broken one way, to prove the suite catches it.
    private sealed class BrokenSource(ToyGlyphSource toy, string how) : IGlyphSource
    {
        private int _calls;

        public string Name => $"broken ({how})";

        public string Summary => $"the toy source, broken: {how}";

        public IReadOnlySet<string> Keys => toy.Keys;

        public GlyphSourceSettings Read(IReadOnlyDictionary<string, string> options) =>
            how == "greedy" ? toy.Read(PageOptions.None) : toy.Read(options);

        public FoundGlyphs Find(GlyphSourceRequest request)
        {
            var found = toy.Find(request);
            var glyphs = found.Glyphs.ToList();
            if (glyphs.Count == 0)
            {
                return found;
            }

            switch (how)
            {
                case "twice":
                    glyphs.Add(glyphs[0]);
                    break;
                case "outside":
                    var g = glyphs[0];
                    glyphs[0] = new LabelledGlyph(g.Text, g.Form, g.Ink, g.Box, g.Region with { Width = Math.Max(1, g.Box.X - g.Region.X) }, g.Baseline, g.Size);
                    break;
                case "drops":
                    glyphs.RemoveAt(glyphs.Count - 1);
                    break;
                case "unstable":
                    if (Interlocked.Increment(ref _calls) % 2 == 0)
                    {
                        var h = glyphs[0];
                        glyphs[0] = new LabelledGlyph(h.Text, h.Form, h.Ink, h.Box, h.Region, h.Baseline + 1f, h.Size);
                    }

                    break;
            }

            return found with { Glyphs = glyphs };
        }
    }
}
