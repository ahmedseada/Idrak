// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Diagnostics;
using System.Globalization;
using Idrak.Abstraction.Testing;
using Idrak.Imaging;
using Idrak.Imaging.Shaping;

// Plan 38, phase F3: the face. A "filled sheet" is simulated without a human: glyphs of a repository font drawn into cells
// at a sheet's size (96 px per em), labelled with their text and form (generic characters; the joined font's forms come
// from the test's own data), traced with "contour" and made into a font by the library's maker ("traced"); the made face
// is checked against the font it was drawn from (shaping by role, metrics), for its alternates, drawn glyphs and export.
internal abstract partial class Tests
{
    internal static readonly (string Name, Action<Device> Run)[] FontMakerGroup =
    [
        ("made fonts: making a face from a sheet's worth of glyphs, measured: the first make apart from warm ones, time per glyph on every core and on one thread (the same face both ways), and managed allocation", MadeFontMeasured),
        ("font makers: traced's options are read in any case and refused naming the key or the range (a key it does not take, units per em, a side bearing, an ink share, a script or feature tag, a character to measure, threads); a request needs glyphs and a family; no glyphs is refused", FontMakerOptions),
        ("made fonts: a sheet filled with DejaVu's glyphs (letters, punctuation, the fi, fl and ffi ligatures, second fills of a and e), traced with contour, makes a face whose generic lines shape like DejaVu's glyph for glyph by role; its ascender, descender and x-height within 0.02 em of the same measures on DejaVu's own outlines, its x-height within 0.03 em of DejaVu's x; every advance its ink and a side bearing of half the pen either side", MadeFontFromLatinSheet),
        ("made fonts: a sheet filled with Cairo's letters in the forms the test's data gives, lam-alef in two forms and two marks makes a face whose generic lines shape like Cairo's glyph for glyph by role (forms and ligatures); its joined forms take no side bearing; its marks sit their gap above or below their bases' ink; the digits and marks the sheet lacks are drawn, the written marks are not", MadeFontFromJoinedSheet),
        ("made fonts: second fills are alternates: the GSUB lists them under aalt, salt and ALTS; +salt takes the second fill; ALTS=random picks per glyph from the run's seed (the same seed the same picks, other seeds others, both fills seen), none without it; a page drawn in the made font picks them from its seed and says so (the same seed, the same page; off when the chance is 0); a font with no alternates asks for none", MadeFontAlternates),
        ("made fonts: the made faces written as TrueType read back as OpenType with the same outlines (every point), advances, characters, GSUB, GDEF and GPOS, and shape the sample lines the same (with alternates picked too); written to test-output/fonts/made for check_font.py", MadeFontExport),
    ];

    private const char Zwj = '\u200D';

    // Generic lines: an ordinary letter and a report (Latin), a sales report and a team meeting (Arabic).
    private static readonly string[] LatinLines = ["Thank you for your letter of 12 May.", "The report is attached: 12 pages.", "The office files are filed in March."];

    private static readonly string[] JoinedLines = ["\u062A\u0642\u0631\u064A\u0631 \u0627\u0644\u0645\u0628\u064A\u0639\u0627\u062A \u0627\u0644\u0634\u0647\u0631\u064A",
        "\u0627\u062C\u062A\u0645\u0627\u0639 \u0627\u0644\u0641\u0631\u064A\u0642 \u064A\u0648\u0645 \u0627\u0644\u062B\u0644\u0627\u062B\u0627\u0621", "\u0633\u0644\u0627\u0645"];

    // The Latin sheet: letters and punctuation, three ligatures.
    private const string SheetCharacters = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ.,:;!?'-()";

    private static readonly string[] SheetLigatures = ["fi", "fl", "ffi"];

    // The joined sheet's data: each letter and the forms a sheet asks it in (as a sheet's character list gives them).
    private static readonly (string Letter, string Forms)[] JoinedLetters =
    [
        ("\u0627", "isol fina"), ("\u0628", "isol init medi fina"), ("\u062A", "isol init medi fina"), ("\u062B", "isol init medi fina"), ("\u062C", "isol init medi fina"),
        ("\u0631", "isol fina"), ("\u0633", "isol init medi fina"), ("\u0634", "isol init medi fina"), ("\u0639", "isol init medi fina"), ("\u0641", "isol init medi fina"),
        ("\u0642", "isol init medi fina"), ("\u0644", "isol init medi fina"), ("\u0645", "isol init medi fina"), ("\u0647", "isol init medi fina"), ("\u0648", "isol fina"),
        ("\u064A", "isol init medi fina"), ("\u0621", "isol"),
    ];

    private const string LamAlef = "\u0644\u0627", Fatha = "\u064E", Kasra = "\u0650";

    private static readonly Lazy<(IFontFace Source, List<SheetCell> Cells, FoundGlyphs Found, MadeFont Made)> LatinMade = new(() => MakeFrom(SubsetFace(), LatinSheet));

    private static readonly Lazy<(IFontFace Source, List<SheetCell> Cells, FoundGlyphs Found, MadeFont Made)> JoinedMade = new(() =>
        MakeFrom(new OpenTypeFormat().Read(File.ReadAllBytes(FontData("arabic/cairo.ttf")), "cairo").Single(), JoinedSheet));

    // What a sheet's cell holds: its text and form, the source font's glyph written there, and a slant (a second fill).
    private readonly record struct SheetCell(string Text, string? Form, int Glyph, float Slant = 0f);

    internal static void MadeFontMeasured(Device device)
    {
        CpuOnly(device);
        var maker = new TracedFontMaker();
        var dejaVu = SubsetFace();
        var cairo = new OpenTypeFormat().Read(File.ReadAllBytes(FontData("arabic/cairo.ttf")), "cairo").Single();
        var sheets = new List<FoundGlyphs> { FillSheet(dejaVu, LatinSheet(dejaVu)), FillSheet(cairo, JoinedSheet(cairo)) };
        int glyphs = sheets.Sum(s => s.Glyphs.Count);
        var every = maker.Read(PageOptions.None);
        var one = maker.Read(PageOptionsOf((TracedFontMaker.ThreadsKey, "1")));

        // The first make of the process's maker (cold: its methods compiled as they are first called; rule 76).
        long allocated = GC.GetTotalAllocatedBytes(precise: true);
        long start = Stopwatch.GetTimestamp();
        var first = maker.Make(new FontMakerRequest(sheets[0], "A Measured Hand", every));
        double cold = Stopwatch.GetElapsedTime(start).TotalMilliseconds;
        long coldBytes = GC.GetTotalAllocatedBytes(precise: true) - allocated;

        // Warm: the make (traced, measured, placed, mapped) and the face's completion (the layout tables, anchors and drawn
        // glyphs the map's face makes on first use) timed apart, both sheets a round; managed allocation of both.
        (double Make, double Complete, double Bytes, string[] Faces) Timed(FontMakerSettings settings, int rounds)
        {
            var made = new MadeFont[sheets.Count];
            double make = 0, complete = 0, bytes = 0;
            for (int round = 0; round < rounds; round++)
            {
                for (int s = 0; s < sheets.Count; s++)
                {
                    long before = GC.GetTotalAllocatedBytes(precise: true);
                    long at = Stopwatch.GetTimestamp();
                    made[s] = maker.Make(new FontMakerRequest(sheets[s], "A Measured Hand", settings));
                    long half = Stopwatch.GetTimestamp();
                    Completed(made[s].Face);
                    long end = Stopwatch.GetTimestamp();
                    bytes += GC.GetTotalAllocatedBytes(precise: true) - before;
                    make += Stopwatch.GetElapsedTime(at, half).TotalMilliseconds;
                    complete += Stopwatch.GetElapsedTime(half, end).TotalMilliseconds;
                }
            }

            return (make / rounds, complete / rounds, bytes / rounds, [.. made.Select(FaceText)]);
        }

        Completed(first.Face);
        double coldComplete = Stopwatch.GetElapsedTime(start).TotalMilliseconds - cold;
        Timed(every, 3);                                                     // warm-up: the tiered JIT settles
        Timed(one, 3);
        var parallel = Timed(every, 5);
        var single = Timed(one, 5);
        Check(parallel.Faces.SequenceEqual(single.Faces), "the faces made on every core and on one thread are the same (rule 71)");
        Check(FaceText(first) == parallel.Faces[0], "the first make made the face the warm ones make");
        Console.WriteLine(string.Create(CultureInfo.InvariantCulture,
            $"    traced, {glyphs} glyphs at 96 px per em (DejaVu's sheet, {sheets[0].Glyphs.Count}; Cairo's, {sheets[1].Glyphs.Count}): the first make (DejaVu's) {cold:F0} ms and its completion {coldComplete:F0} ms, {coldBytes / 1e6:F1} MB; warm, both sheets: made on {ComputeResources.MaxCpuThreads} threads {parallel.Make:F0} ms ({parallel.Make * 1000 / glyphs:F0} µs a glyph), on one {single.Make:F0} ms ({single.Make * 1000 / glyphs:F0} µs a glyph, {single.Make / parallel.Make:F2}x); completed (tables, anchors, drawn glyphs) {parallel.Complete:F0} ms; {parallel.Bytes / glyphs / 1024:F0} KB allocated a glyph"));
    }

    // A face's parts made on first use: its layout tables (and the anchors measured for them) and its drawn glyphs.
    private static void Completed(IFontFace face)
    {
        _ = face.Table("GSUB");
        _ = face.Table("GDEF");
        _ = face.Table("GPOS");
        _ = face.Outline(face.GlyphCount - 1);
    }

    internal static void FontMakerOptions(Device device)
    {
        CpuOnly(device);
        var maker = new TracedFontMaker();
        void Refused(string key, string value, params string[] naming) =>
            ThrowsNaming<ArgumentException>(() => maker.Read(PageOptionsOf((key, value))), $"{key}={value}", naming);

        Refused("glyph-size", "3", "does not take 'glyph-size'", TracedFontMaker.UnitsKey, TracedFontMaker.ThreadsKey);
        Refused(TracedFontMaker.UnitsKey, "8", "units-per-em", "from 16 to 16384");
        Refused(TracedFontMaker.SideBearingKey, "0.7", "side-bearing", "from 0 to 0.5");
        Refused(TracedFontMaker.InkShareKey, "0.2", "ink-share", "from 0.5 to 1");
        Refused(TracedFontMaker.ScriptsKey, "latin", "scripts", "'latin'", "four printable characters");
        Refused(TracedFontMaker.LigatureFeatureKey, "lig", "ligature-feature", "'lig'");
        Refused(TracedFontMaker.MeasureKey, "12", "measure", "'12'", "U+XXXX or one character");
        Refused(TracedFontMaker.ThreadsKey, "0", "threads", "from 1 to 1024");
        Refused(TracedFontMaker.AlternatesKey, "maybe", "alternates", "on or off");

        var read = maker.Read(PageOptionsOf(("UNITS-PER-EM", "2048"), (TracedFontMaker.SideBearingKey, "auto"), (TracedFontMaker.ScriptsKey, "latn:DFLT"),
            (TracedFontMaker.MeasureKey, "U+0031:x"), (TracedFontMaker.JoinedFormsKey, "none"), (TracedFontMaker.ThreadsKey, "auto")));
        Check(read.Options.Count == 6 && ReferenceEquals(maker.Read(PageOptions.None), maker.Read(PageOptions.None)), "options read in any case; no options are the defaults, read once");

        var found = LatinMade.Value.Found;
        ThrowsNaming<ArgumentException>(() => _ = new FontMakerRequest(found, " ", FontMakerSettings.None), "a request with no family", "family name");
        ThrowsNaming<ArgumentNullException>(() => _ = new FontMakerRequest(null!, "A Hand", FontMakerSettings.None), "a request with no glyphs", "Glyphs");
        ThrowsNaming<ArgumentException>(() => maker.Make(new FontMakerRequest(FoundGlyphs.None, "A Hand", FontMakerSettings.None)), "no glyphs", "has none");
        var request = new FontMakerRequest(found, "A Hand", FontMakerSettings.None);
        Check(request.Tracer == OutlineTracers.Contour && request.TracerOptions is null && request.Style is null, "a request traces with contour's defaults unless told");
    }

    internal static void MadeFontFromLatinSheet(Device device)
    {
        CpuOnly(device);
        var (source, cells, found, made) = LatinMade.Value;
        var face = made.Face;
        Check(made.LeftOut.Count == 0 && made.Glyphs.Count == cells.Count && face.Family == "A Latin Hand" && face.Metrics.UnitsPerEm == 1000,
            $"every cell made: {made.Glyphs.Count} of {cells.Count}, left out {string.Join("; ", made.LeftOut)}");
        ShapesByRole(source, face, made, cells, LatinLines, "Latn", false, new OpenTypeShaper());

        // The metrics: the same measures on DejaVu's own outlines (the cells' glyphs, drawn and slanted as on the sheet), in ems.
        int upem = source.Metrics.UnitsPerEm;
        var inks = cells.Select(c => InkMeasure.Of(Slanted(source.Outline(c.Glyph), c.Slant), upem, 0.13 * upem)).OfType<InkStats>().ToList();
        var own = TracedFontMaker.Heights(inks, 0.95);
        var m = made.Metrics;
        double xTop = source.Outline(source.Glyph('x')).Bounds().MaxY / upem;
        string metrics = string.Create(CultureInfo.InvariantCulture,
            $"made {m}; DejaVu's outlines: ascender {own.Ascender / upem:0.###} em, descender {own.Descender / upem:0.###}, x-height {own.XHeight / upem:0.###}; DejaVu's x {xTop:0.###}");
        Check(Math.Abs(m.Ascender / 1000.0 - own.Ascender / upem) <= 0.02 && Math.Abs(m.Descender / 1000.0 - own.Descender / upem) <= 0.02
              && Math.Abs(m.XHeight / 1000.0 - own.XHeight / upem) <= 0.02 && Math.Abs(m.XHeight / 1000.0 - xTop) <= 0.03, metrics);
        Console.WriteLine($"    {metrics}");
        CheckAdvances(made, joined: []);
        Check(m.SideBearing == (int)Math.Round(m.Pen / 2.0) && m.Space > 0 && face.Glyph(' ') != 0 && face.Advance(face.Glyph(' ')) == m.Space && face.Glyph(0xA0) == face.Glyph(' '),
            $"the side bearing half the pen, the space measured: {m}");
        Check(face.Table("OS/2").Length >= 96 && System.Buffers.Binary.BinaryPrimitives.ReadInt16BigEndian(face.Table("OS/2").Span[86..]) == m.XHeight,
            "the made face's OS/2 holds the x-height measured");
    }

    internal static void MadeFontFromJoinedSheet(Device device)
    {
        CpuOnly(device);
        var (source, cells, _, made) = JoinedMade.Value;
        var face = made.Face;
        Check(made.LeftOut.Count == 0 && made.Glyphs.Count == cells.Count, $"every cell made: {made.Glyphs.Count} of {cells.Count}, left out {string.Join("; ", made.LeftOut)}");
        ShapesByRole(source, face, made, cells, JoinedLines, "Arab", true, new ArabicShaper());
        CheckAdvances(made, joined: ["init", "medi", "fina"]);

        // Marks: a fatha above and a kasra below a beh, their ink the map's gap from the base's ink (around its middle).
        int upem = face.Metrics.UnitsPerEm, gap = made.Map.AnchorGap;
        foreach (var (mark, above) in new[] { (Fatha, true), (Kasra, false) })
        {
            string text = "\u0628" + mark;
            var shaped = new ArabicShaper().Shape(face, text, new TextRun(0, text.Length, "Arab", true));
            int markGlyph = face.Glyph(mark[0]), baseGlyph = face.Glyph('\u0628');
            var m = shaped.Single(s => s.Glyph == markGlyph);
            var b = shaped.Single(s => s.Glyph == baseGlyph);
            var markInk = InkMeasure.Of(face.Outline(markGlyph), upem, made.Map.AnchorWindow)!.Value;
            var baseInk = InkMeasure.Of(face.Outline(baseGlyph), upem, made.Map.AnchorWindow)!.Value;
            double apart = above ? markInk.MinY + m.YOffset - baseInk.Top : baseInk.Bottom - (markInk.MaxY + m.YOffset);
            Check(Math.Abs(apart - gap) <= 6 && m.XAdvance == 0 && b.XAdvance > 0, string.Create(CultureInfo.InvariantCulture,
                $"U+{(int)mark[0]:X4} {(above ? "above" : "below")} its base: {apart:0.#} units from its ink (the gap {gap}); {m}, {b}"));
        }

        // Drawn: the digits (the sheet has none) and the marks it lacks; the written marks are its own.
        var mapped = (MappedFontFace)face;
        var drawn = mapped.DrawnCharacters.ToHashSet();
        foreach (int code in new[] { '1', '7', 0x0661, 0x0667, 0x0651, 0x064B })
        {
            int glyph = face.Glyph(code);
            Check(glyph != 0 && drawn.Contains(code) && !face.Outline(glyph).IsEmpty, string.Create(CultureInfo.InvariantCulture, $"U+{code:X4} drawn into the made face: glyph {glyph}"));
        }

        Check(!drawn.Contains(Fatha[0]) && !drawn.Contains(Kasra[0]) && face.Glyph(Fatha[0]) == made.Glyphs.Single(g => g.Text == Fatha).Glyph,
            "the written marks are the sheet's, not drawn");
        var digit = InkMeasure.Of(face.Outline(face.Glyph('7')), upem, made.Map.AnchorWindow)!.Value;
        Check(digit.Height > 0.3 * made.Metrics.XHeight && digit.Height < 3 * made.Metrics.XHeight && digit.Thickness < 3 * made.Metrics.Pen,
            string.Create(CultureInfo.InvariantCulture, $"a drawn digit sized from the written glyphs: {digit.Height:0} high, pen {digit.Thickness:0}; {made.Metrics}"));
    }

    internal static void MadeFontAlternates(Device device)
    {
        CpuOnly(device);
        var (_, _, _, latin) = LatinMade.Value;
        var (_, _, _, joined) = JoinedMade.Value;
        var face = latin.Face;
        var substitutions = FontLayout.For(face).Substitutions;
        Check(substitutions.HasFeature("aalt") && substitutions.HasFeature("salt") && substitutions.HasFeature(FontMap.AlternatesFeature),
            "the made face's GSUB lists aalt, salt and ALTS");
        Check(latin.Map.Alternates.Count == 2 && joined.Map.Alternates.Count == 2, $"two characters with second fills in each map: {latin.Map.Alternates.Count}, {joined.Map.Alternates.Count}");

        int a0 = FillOf(latin, "a", null, 0), a1 = FillOf(latin, "a", null, 1), e0 = FillOf(latin, "e", null, 0), e1 = FillOf(latin, "e", null, 1);
        const string Text = "a cake and a pear are made available each age";
        var run = new TextRun(0, Text.Length, "Latn", false);
        int[] Glyphs(string[]? features, int seed) => [.. new OpenTypeShaper().Shape(face, Text, run with { Seed = seed }, features).Select(s => s.Glyph)];
        int[] plain = Glyphs(null, 0), salt = Glyphs(["+salt"], 0);
        Check(plain.Contains(a0) && !plain.Contains(a1) && !plain.Contains(e1) && salt.Contains(a1) && salt.Contains(e1) && !salt.Contains(a0) && !salt.Contains(e0),
            "shaped plainly: the first fills; with +salt: the second");

        var picks = Enumerable.Range(1, 12).Select(seed => Glyphs(["ALTS=random"], seed)).ToList();
        Check(picks.All(p => p.Length == plain.Length) && picks.SelectMany(p => p).Contains(a1) && picks.SelectMany(p => p).Contains(a0) && picks.SelectMany(p => p).Contains(e1),
            "ALTS=random picks both fills");
        Check(Glyphs(["ALTS=random"], 5).SequenceEqual(picks[4]) && picks.Select(p => string.Join(",", p)).Distinct().Count() > 6, "the same seed, the same picks; other seeds, others");
        Check(picks.All(p => p.Select((g, i) => (g, i)).All(x => x.g == plain[x.i] || (plain[x.i] == a0 && x.g == a1) || (plain[x.i] == e0 && x.g == e1))),
            "a pick changes only the glyphs with alternates, each to one of its fills");

        // The joined face: a beh written alone and a meem between letters each have a second fill.
        const string Joined = "\u0628 \u0628 \u0628 \u0628 \u0644\u0645\u0627 \u0644\u0645\u0627 \u0644\u0645\u0627";
        var joinedRun = new TextRun(0, Joined.Length, "Arab", true);
        var seen = Enumerable.Range(1, 8).SelectMany(seed => new ArabicShaper().Shape(joined.Face, Joined, joinedRun with { Seed = seed }, ["ALTS=random"])).Select(s => s.Glyph).ToHashSet();
        Check(seen.Contains(FillOf(joined, "\u0628", "isol", 1)) && seen.Contains(FillOf(joined, "\u0628", "isol", 0)) && seen.Contains(FillOf(joined, "\u0645", "medi", 1)),
            "the joined face's forms pick their fills after forms are set");

        // A page drawn in the made font: the alternates asked for and picked from its seed; the same seed, the same page.
        var content = new PageContent { Body = [LatinLines[0], "A cake and a pear are made available each age."], Language = "en" };
        var style = new PageStyle { Fonts = [face], Color = false };
        var plainPage = new PlainLayout();
        var page = plainPage.Render(content, style, 5);
        var again = plainPage.Render(content, style, 5);
        Check(page.Description.Contains(", alternates", StringComparison.Ordinal) && page.Image.Pixels.AsSpan().SequenceEqual(again.Image.Pixels), $"the page: {page.Description}");
        var dejaVu = plainPage.Render(content, TestPageStyle(color: false), 5);
        Check(!dejaVu.Description.Contains(", alternates", StringComparison.Ordinal) && PageDrawing.Asked(SubsetFace(), [FontMap.AlternatesFeature]) is null
              && PageDrawing.Asked(face, [FontMap.AlternatesFeature]) is ["ALTS=random"], $"a font with no alternates asks for none: {dejaVu.Description}");
        var drawing = FontMaps.Drawing;
        try
        {
            FontMaps.Drawing = drawing with { Alternates = 0 };
            Check(!plainPage.Render(content, style, 5).Description.Contains(", alternates", StringComparison.Ordinal), "a chance of 0 asks for none");
        }
        finally
        {
            FontMaps.Drawing = drawing;
        }
    }

    internal static void MadeFontExport(Device device)
    {
        CpuOnly(device);
        foreach (var (name, made, lines, script, rtl, shaper) in new[] { ("latin-hand", LatinMade.Value.Made, LatinLines, "Latn", false, new OpenTypeShaper()), ("joined-hand", JoinedMade.Value.Made, JoinedLines, "Arab", true, new ArabicShaper()) })
        {
            var face = made.Face;
            using var stream = new MemoryStream();
            new TrueTypeWriter().Write(face, stream, new FontWriteOptions());
            byte[] bytes = stream.ToArray();
            var back = new OpenTypeFormat().Read(bytes, name).Single();
            Check(back.GlyphCount == face.GlyphCount && back.Family == face.Family && back.Metrics == face.Metrics && back.Format == "opentype",
                $"{name}: {back.GlyphCount} glyphs read back of {face.GlyphCount}, {back.Family} {back.Metrics}");
            for (int g = 0; g < face.GlyphCount; g++)
            {
                var (a, b) = (face.Outline(g), back.Outline(g));
                Check(a.Verbs.Span.SequenceEqual(b.Verbs.Span) && a.Points.Span.SequenceEqual(b.Points.Span) && face.Advance(g) == back.Advance(g),
                    $"{name}: glyph {g} read back with {b.Points.Length / 2} points and advance {back.Advance(g)}, written {a.Points.Length / 2} and {face.Advance(g)}");
            }

            for (int code = 0; code <= 0xFFFF; code++)
            {
                if (face.Glyph(code) != back.Glyph(code))
                {
                    Check(false, string.Create(CultureInfo.InvariantCulture, $"{name}: U+{code:X4} reads back as glyph {back.Glyph(code)}, written {face.Glyph(code)}"));
                }
            }

            foreach (string tag in (string[])["GSUB", "GDEF", "GPOS"])
            {
                Check(!face.Table(tag).IsEmpty && face.Table(tag).Span.SequenceEqual(back.Table(tag).Span), $"{name}: its {tag} read back as written");
            }

            foreach (string line in lines)
            {
                var run = new TextRun(0, line.Length, script, rtl);
                foreach (var (features, seed) in new (string[]?, int)[] { (null, 0), (["ALTS=random"], 3), (["+salt"], 0) })
                {
                    Check(shaper.Shape(face, line, run with { Seed = seed }, features).SequenceEqual(shaper.Shape(back, line, run with { Seed = seed }, features)),
                        $"{name}: '{line}' shapes the same read back ({(features is null ? "defaults" : features[0])})");
                }
            }

            // A copy kept for check_font.py.
            if (TestPaths.Root is { } root)
            {
                string kept = Path.Combine(root, "test-output", "fonts", "made");
                Directory.CreateDirectory(kept);
                File.WriteAllBytes(Path.Combine(kept, name + ".ttf"), bytes);
            }
        }
    }

    // The glyph of a text, form and fill in a made font.
    private static int FillOf(MadeFont made, string text, string? form, int fill) =>
        made.Glyphs.Single(g => g.Text == text && g.Form == form && g.Fill == fill).Glyph;

    // A sheet's cells drawn from `source`, labelled, made into a font by the library's maker with its defaults.
    private static (IFontFace Source, List<SheetCell> Cells, FoundGlyphs Found, MadeFont Made) MakeFrom(IFontFace source, Func<IFontFace, List<SheetCell>> sheet)
    {
        var cells = sheet(source);
        var found = FillSheet(source, cells);
        string family = cells.Any(c => c.Form is not null) ? "A Joined Hand" : "A Latin Hand";
        var maker = new TracedFontMaker();
        return (source, cells, found, maker.Make(new FontMakerRequest(found, family, maker.Read(PageOptions.None)) { Style = "Regular" }));
    }

    // The Latin sheet: its characters, the ligatures as the font shapes them, second fills of a and e (slanted either way).
    private static List<SheetCell> LatinSheet(IFontFace font)
    {
        var cells = new List<SheetCell>();
        foreach (char c in SheetCharacters)
        {
            int glyph = font.Glyph(c);
            Check(glyph != 0, $"DejaVu has '{c}'");
            cells.Add(new SheetCell(c.ToString(), null, glyph));
        }

        foreach (string ligature in SheetLigatures)
        {
            var shaped = new OpenTypeShaper().Shape(font, ligature, new TextRun(0, ligature.Length, "Latn", false));
            Check(shaped.Length == 1, $"DejaVu joins '{ligature}' into one glyph");
            cells.Add(new SheetCell(ligature, null, shaped[0].Glyph));
        }

        cells.Add(new SheetCell("a", null, font.Glyph('a'), 0.18f));
        cells.Add(new SheetCell("e", null, font.Glyph('e'), -0.14f));
        return cells;
    }

    // The joined sheet: each letter in the forms the data gives (the font's glyph for each, shaped in a joining context),
    // lam-alef alone and after a letter, a fatha and a kasra, second fills of beh alone and meem between letters.
    private static List<SheetCell> JoinedSheet(IFontFace font)
    {
        int Shaped(string text, int at)
        {
            var run = new TextRun(0, text.Length, "Arab", true) { Clusters = TextClusters.Characters };
            return new ArabicShaper().Shape(font, text, run).Single(s => s.Cluster == at).Glyph;
        }

        var cells = new List<SheetCell>();
        foreach (var (letter, forms) in JoinedLetters)
        {
            foreach (string form in forms.Split(' '))
            {
                var (text, at) = form switch
                {
                    "init" => (letter + Zwj, 0),
                    "medi" => (Zwj + letter + Zwj, 1),
                    "fina" => (Zwj + letter, 1),
                    _ => (letter, 0),
                };
                cells.Add(new SheetCell(letter, form, Shaped(text, at)));
            }
        }

        cells.Add(new SheetCell(LamAlef, "isol", Shaped(LamAlef, 0)));
        cells.Add(new SheetCell(LamAlef, "fina", Shaped(Zwj + LamAlef, 1)));
        cells.Add(new SheetCell(Fatha, null, font.Glyph(Fatha[0])));
        cells.Add(new SheetCell(Kasra, null, font.Glyph(Kasra[0])));
        cells.Add(new SheetCell("\u0628", "isol", Shaped("\u0628", 0), 0.16f));
        cells.Add(new SheetCell("\u0645", "medi", Shaped(Zwj + "\u0645" + Zwj, 1), -0.12f));
        return cells;
    }

    // The cells written as a glyph source gives them: each glyph drawn into a cell of the page at 96 px per em on a baseline
    // a margin below the font's ascender, kept at a scan's 8-bit levels, its ink cut out, labelled with the cell's text and
    // form; cells in rows.
    private static FoundGlyphs FillSheet(IFontFace source, List<SheetCell> cells, float size = 96f)
    {
        const int Margin = 8, RowWidth = 1600;
        var metrics = source.Metrics;
        float scale = size / metrics.UnitsPerEm;
        int height = (int)MathF.Ceiling((metrics.Ascender - metrics.Descender) * scale) + 2 * Margin;
        float baseline = Margin + metrics.Ascender * scale;
        var glyphs = new List<LabelledGlyph>(cells.Count);
        int x = 0, y = 0;
        foreach (var cell in cells)
        {
            var outline = Slanted(source.Outline(cell.Glyph), cell.Slant);
            var (minX, _, maxX, _) = outline.Bounds();
            float left = MathF.Min(0, minX);
            int width = (int)MathF.Ceiling((MathF.Max(maxX, source.Advance(cell.Glyph)) - left) * scale) + 2 * Margin;
            if (x + width > RowWidth)
            {
                (x, y) = (0, y + height);
            }

            var values = new float[width * height];
            new CoverageRasterizer().Fill(outline, scale, Margin - left * scale, baseline, values, width, height);
            for (int i = 0; i < values.Length; i++)
            {
                values[i] = MathF.Round(Math.Clamp(values[i], 0f, 1f) * 255f) / 255f;
            }

            var coverage = new GlyphCoverage(values, width, height);
            var ink = coverage.InkBox;
            glyphs.Add(new LabelledGlyph(cell.Text, cell.Form, coverage.Crop(ink), new PixelBox(x + ink.X, y + ink.Y, ink.Width, ink.Height), new PixelBox(x, y, width, height),
                y + baseline, size));
            x += width;
        }

        return new FoundGlyphs(glyphs, []);
    }

    // An outline slanted: x moves by `slant` times y (a second fill written a little differently).
    private static GlyphOutline Slanted(GlyphOutline outline, float slant)
    {
        if (slant == 0f || outline.IsEmpty)
        {
            return outline;
        }

        float[] points = outline.Points.ToArray();
        for (int i = 0; i < points.Length; i += 2)
        {
            points[i] += slant * points[i + 1];
        }

        return new GlyphOutline(outline.Verbs.ToArray(), points);
    }

    // A glyph's role: its text, and its form unless plain (a single character's isol is its plain glyph).
    private static string RoleOf(string text, string? form) =>
        form is null || form == "isol" && text.EnumerateRunes().Count() == 1 ? text : $"{text}.{form}";

    // The lines shaped with the made face and the source font give glyphs of the same roles, one for one: the sheet's
    // cells name the source's glyphs, the made font's glyphs name theirs, and a glyph of neither (a space, a drawn digit)
    // is named by the line's character it answers.
    private static void ShapesByRole(IFontFace source, IFontFace face, MadeFont made, List<SheetCell> cells, string[] lines, string script, bool rtl, ITextShaper shaper)
    {
        var sourceRoles = new Dictionary<int, HashSet<string>>();
        foreach (var cell in cells.Where(c => c.Slant == 0f))
        {
            (sourceRoles.TryGetValue(cell.Glyph, out var roles) ? roles : sourceRoles[cell.Glyph] = []).Add(RoleOf(cell.Text, cell.Form));
        }

        var madeRoles = made.Glyphs.Where(g => g.Fill == 0).ToDictionary(g => g.Glyph, g => RoleOf(g.Text, g.Form));
        foreach (string line in lines)
        {
            var run = new TextRun(0, line.Length, script, rtl);
            var expected = shaper.Shape(source, line, run);
            var shaped = shaper.Shape(face, line, run);
            string Named(IFontFace font, int glyph) =>
                line.EnumerateRunes().Select(r => r.Value).Where(c => font.Glyph(c) == glyph).Select(char.ConvertFromUtf32).FirstOrDefault() ?? $"#{glyph}";
            var want = expected.Select(s => sourceRoles.TryGetValue(s.Glyph, out var roles) ? roles : [Named(source, s.Glyph)]).ToList();
            var got = shaped.Select(s => madeRoles.TryGetValue(s.Glyph, out string? role) ? role : Named(face, s.Glyph)).ToList();
            Check(got.Count == want.Count && got.Zip(want).All(p => p.Second.Contains(p.First)),
                $"'{line}': the made face's glyphs [{string.Join(" ", got)}], the source's [{string.Join(" ", want.Select(w => string.Join("|", w)))}]");
        }
    }

    // Each made glyph that is not a mark: its ink a side bearing in from its origin and from its advance (none for the
    // joined forms), to the rounding of whole units.
    private static void CheckAdvances(MadeFont made, string[] joined)
    {
        var face = made.Face;
        foreach (var glyph in made.Glyphs.Where(g => !(g.Text.Length == 1 && CharUnicodeInfo.GetUnicodeCategory(g.Text[0]) == UnicodeCategory.NonSpacingMark)))
        {
            int bearing = glyph.Form is { } form && joined.Contains(form) ? 0 : made.Metrics.SideBearing;
            var (minX, _, maxX, _) = face.Outline(glyph.Glyph).Bounds();
            int advance = face.Advance(glyph.Glyph);
            Check(MathF.Abs(minX - bearing) <= 0.5f && advance == (int)MathF.Round(maxX + bearing),
                string.Create(CultureInfo.InvariantCulture, $"'{glyph.Text}'{(glyph.Form is null ? "" : $" ({glyph.Form})")} fill {glyph.Fill}: ink from {minX} to {maxX}, advance {advance}, bearing {bearing}"));
        }
    }

    // A made font as text: its metrics and every glyph's steps, points and advance, and its layout tables.
    private static string FaceText(MadeFont made)
    {
        var face = made.Face;
        var text = new System.Text.StringBuilder().Append(made.Metrics).Append('|');
        for (int g = 0; g < face.GlyphCount; g++)
        {
            text.Append(face.Advance(g)).Append(':');
            foreach (float p in face.Outline(g).Points.Span)
            {
                text.Append(BitConverter.SingleToInt32Bits(p)).Append(',');
            }

            text.Append(';');
        }

        foreach (string tag in (string[])["GSUB", "GDEF", "GPOS"])
        {
            text.Append(Convert.ToBase64String(face.Table(tag).Span));
        }

        return text.ToString();
    }
}
