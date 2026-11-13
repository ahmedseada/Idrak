// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Text;
using System.Text.Json.Nodes;
using Idrak.Abstraction.Testing;
using Idrak.Imaging;

// Legacy fonts used directly: Arabic letter forms under the Private Use Area (F200 plus a code page byte) or found by
// glyph names, no Unicode map and no shaping tables. The library's font maps (data) make them answer Unicode and carry
// GSUB, GDEF and GPOS; what they lack of digits and marks is drawn in, measured from them. Fixtures: three legacy fonts
// (FS_Hand_Style, MCS-HAND, mohareb-hand) and two Unicode fonts that must stay untouched (mcs-diwany-jaly-s-u,
// arbfonts-diwany-thuluth).
internal abstract partial class DesktopTests
{
    private static readonly string[] LegacyFonts = ["FS_Hand_Style", "MCS-HAND", "mohareb-hand"];
    private static readonly string[] UnicodeControls = ["mcs-diwany-jaly-s-u", "arbfonts-diwany-thuluth"];
    private const string PuaMap = "arabic-pua-f200", NamesMap = "arabic-glyph-names";

    // The face as the OpenType format reads it, no map applied.
    private static IFontFace RawFace(string name) => new OpenTypeFormat().Read(File.ReadAllBytes(FontData($"arabic/{name}.ttf")), name)[0];

    private static IFontFace MappedFace(string name) => FontFormats.Read(FontData($"arabic/{name}.ttf")).Single();

    // The font's own glyph the PUA layout gives a letter's form (through the layout's fallback), 0 when it has none.
    private static int LayoutGlyph(IFontFace raw, int letter, string form)
    {
        var map = FontMaps.Get(PuaMap);
        if (!map.Letters[letter].Forms.TryGetValue(form, out var refs))
        {
            return 0;
        }

        int glyph = raw.Glyph(refs[0].Value);
        return glyph != 0 ? glyph : map.Fallbacks.TryGetValue(refs[0].Text, out var fallback) ? raw.Glyph(fallback[0].Value) : 0;
    }

    // The glyph a letter is drawn with in a form: shaped between tatweels (a joining character every map gives).
    private static int ShapedForm(IFontFace face, int letter, string form)
    {
        string c = char.ConvertFromUtf32(letter), t = "ـ";
        string text = form switch { "init" => c + t, "medi" => t + c + t, "fina" => t + c, _ => c };
        int at = form is "medi" or "fina" ? 1 : 0;
        var glyphs = TextShapers.Shape(face, text, new TextRun(0, text.Length, "Arab", true));
        return glyphs.Single(g => g.Cluster == at).Glyph;
    }

    private static void LegacyFontsMapped(Device device)
    {
        Check(FontMaps.Names.Contains(PuaMap) && FontMaps.Names.Contains(NamesMap) && FontMaps.Origin(PuaMap) == Overrides.Library
              && FontMappers.Names.Contains(FontMappers.OpenType), $"maps {string.Join(", ", FontMaps.Names)}; mappers {string.Join(", ", FontMappers.Names)}");
        var pua = FontMaps.Get(PuaMap);
        foreach (string name in LegacyFonts)
        {
            var raw = RawFace(name);
            var face = MappedFace(name);
            Check(FontMaps.MappedBy(face)?.Name == PuaMap && raw.Glyph(0x0628) == 0, $"{name}: mapped by {FontMaps.MappedBy(face)?.Name ?? "nothing"}");
            var lacking = Enumerable.Range(0x0621, 0x064A - 0x0621 + 1).Where(c => pua.Letters.ContainsKey(c) && face.Glyph(c) == 0).ToList();
            Check(lacking.Count == 0, $"{name}: no glyph for {string.Join(" ", lacking.Select(c => $"U+{c:X4}"))}");
            Check(Enumerable.Range(0x0660, 10).All(c => face.Glyph(c) == raw.Glyph(0xF230 + c - 0x0660)), $"{name}: the Arabic-Indic digits are the font's own");

            // Every letter in every form the layout gives is drawn with the font's glyph of that form, through the made GSUB.
            var wrong = new List<string>();
            foreach (var (letter, _) in pua.Letters)
            {
                foreach (string form in FontMap.Forms)
                {
                    int expected = LayoutGlyph(raw, letter, form);
                    if (expected != 0 && ShapedForm(face, letter, form) != expected)
                    {
                        wrong.Add($"U+{letter:X4}.{form}: {ShapedForm(face, letter, form)}, the font's {expected}");
                    }
                }
            }

            Check(wrong.Count == 0, $"{name}: {wrong.Count} forms differ: {string.Join("; ", wrong.Take(6))}");
        }

        // mohareb-hand has no isolated khaa: the layout's fallback gives its final form.
        var mohareb = RawFace("mohareb-hand");
        Check(mohareb.Glyph(0xF264) == 0 && MappedFace("mohareb-hand").Glyph(0x062E) == mohareb.Glyph(0xF263), "mohareb-hand: khaa falls back to its final form");

        // FS_Hand_Style fits both rule sets: the glyph names give every letter form the PUA layout gives.
        var fs = RawFace("FS_Hand_Style");
        var byNames = FontMaps.Apply(fs, FontMaps.Get(NamesMap));
        var byLayout = FontMaps.Apply(fs, pua);
        var differ = new List<string>();
        int compared = 0;
        foreach (var (letter, _) in pua.Letters)
        {
            foreach (string form in FontMap.Forms)
            {
                if (LayoutGlyph(fs, letter, form) == 0)
                {
                    continue;
                }

                compared++;
                int a = ShapedForm(byNames, letter, form), b = ShapedForm(byLayout, letter, form);
                if (a != b)
                {
                    differ.Add($"U+{letter:X4}.{form}: names {a}, layout {b}");
                }
            }
        }

        Check(compared > 100 && differ.Count == 0, $"{compared} forms compared; {string.Join("; ", differ.Take(6))}");
        string lamAlef = "لا لأ لإ لآ";
        Check(TextShapers.Shape(byNames, lamAlef, new TextRun(0, lamAlef.Length, "Arab", true)).Select(g => g.Glyph)
                .SequenceEqual(TextShapers.Shape(byLayout, lamAlef, new TextRun(0, lamAlef.Length, "Arab", true)).Select(g => g.Glyph)),
            "the lam-alef ligatures agree");
        Check(FontMaps.Apply(fs, FontMaps.Get(NamesMap)) is var names && FontMaps.MappedBy(names) == FontMaps.Get(NamesMap), "the face says which map made it");
    }

    private static void LegacyFontsShape(Device device)
    {
        foreach (string name in LegacyFonts)
        {
            var raw = RawFace(name);
            var face = MappedFace(name);
            int G(int pua) => raw.Glyph(pua) != 0 ? raw.Glyph(pua) : raw.Glyph(pua - 1);   // the khaa fallback (none of these use it)
            ShapedGlyph[] Shape(string text) => TextShapers.Shape(face, text, new TextRun(0, text.Length, "Arab", true));

            // مينا: initial meem, medial ya and noon, final alef; drawn right to left (the glyphs come back reversed).
            var mina = Shape("مينا").Select(g => g.Glyph).Reverse().ToArray();
            Check(mina.SequenceEqual([G(0xF2BF), G(0xF2CE), G(0xF2C4), G(0xF242)]), $"{name}: مينا is {string.Join(" ", mina)}");

            // Each lam-alef is one ligature: isolated after a space, final after a joining letter.
            foreach (var (text, pua) in new[] { ("لا", 0xF2DC), ("لأ", 0xF2DE), ("لإ", 0xF2E2), ("لآ", 0xF2E0), ("بلا", 0xF2DD) })
            {
                var glyphs = Shape(text);
                Check(glyphs[0].Glyph == G(pua), $"{name}: {text} ends in glyph {glyphs[0].Glyph}, the ligature is {G(pua)} ({glyphs.Length} glyphs)");
            }

            string digits = "١٠ ٢٣";
            var shaped = TextShapers.Shape(face, digits, new TextRun(0, digits.Length, "Arab", true));
            Check(shaped.Select(g => g.Glyph).Order().SequenceEqual(new[] { G(0xF231), G(0xF230), G(0xF200), G(0xF232), G(0xF233) }.Order()), $"{name}: digits {string.Join(" ", shaped.Select(g => g.Glyph))}");

            // A mark above and one below each sit on their base's ink, no wider than nothing.
            foreach (string text in (string[])["بَ", "بِ", "نُ", "سْ", "لّ"])
            {
                var g = Shape(text);
                var mark = g.Single(x => x.Cluster == 0 && x.Glyph != g.Last().Glyph);
                var baseGlyph = g.Last();
                var ink = InkMeasure.Of(face.Outline(baseGlyph.Glyph), face.Metrics.UnitsPerEm, 130)!.Value;
                var (_, minY, _, maxY) = face.Outline(mark.Glyph).Bounds();
                bool below = text[1] == 'ِ';
                Check(mark.XAdvance == 0 && !face.Outline(mark.Glyph).IsEmpty, $"{name}: {text}'s mark {mark.Glyph} has width {mark.XAdvance} or no ink");
                Check(below ? maxY + mark.YOffset <= ink.Bottom : minY + mark.YOffset >= ink.Top,
                    $"{name}: {text}'s mark spans {minY + mark.YOffset}..{maxY + mark.YOffset}, its base's ink {ink.Bottom}..{ink.Top}");
            }

            // The marks the font draws are its own; FS_Hand_Style's have no ink and are drawn in.
            int fatha = face.Glyph(0x064E);
            Check(name == "FS_Hand_Style" ? fatha >= raw.GlyphCount : fatha == raw.Glyph(0xF2F2), $"{name}: fatha is glyph {fatha} (the font has {raw.GlyphCount})");
        }
    }

    private static void LegacyFontsDrawGlyphs(Device device)
    {
        var drawnMarks = Enumerable.Range(0x064B, 0x065F - 0x064B + 1).Append(0x0670).ToArray();
        foreach (string name in LegacyFonts)
        {
            var raw = RawFace(name);
            var face = MappedFace(name);
            int upem = face.Metrics.UnitsPerEm;
            Check(raw.Glyph('5') == 0 && Enumerable.Range('0', 10).All(c => face.Glyph(c) >= raw.GlyphCount), $"{name}: the Latin digits are drawn in");
            Check(drawnMarks.All(c => face.Glyph(c) != 0 && !face.Outline(face.Glyph(c)).IsEmpty && face.Advance(face.Glyph(c)) == 0),
                $"{name}: every mark answers an inked glyph of no width: {string.Join(" ", drawnMarks.Where(c => face.Glyph(c) == 0 || face.Outline(face.Glyph(c)).IsEmpty).Select(c => $"U+{c:X4}"))}");
            Check(face.GlyphCount > raw.GlyphCount && face.Table("GDEF").Length > 0 && face.Table("GPOS").Length > 0, $"{name}: {face.GlyphCount} glyphs");

            // Measured from the font: the drawn digits' height and pen are its own Arabic-Indic digits'. Both sets are measured
            // the same way (the tall glyphs' median); the hand wobbles points by 1.2% of the box and the pen by 6%, and the ink
            // is read at 256 pixels per em (8 units in 2,048): 6% on the height, 25% on the pen (a diagonal stroke reads wider).
            InkStats[] Inks(IEnumerable<int> glyphs) => [.. glyphs.Select(g => InkMeasure.Of(face.Outline(g), upem, 130)).OfType<InkStats>()];
            (double Height, double Pen) Typical(InkStats[] inks)
            {
                double tallest = inks.Max(i => i.Height);
                var tall = inks.Where(i => i.Height >= tallest / 2).ToList();
                return (InkMeasure.Median([.. tall.Select(i => i.Height)]), InkMeasure.Median([.. tall.Select(i => i.Thickness)]));
            }

            var own = Typical(Inks(Enumerable.Range(0x0660, 10).Select(face.Glyph)));
            var drawn = Typical(Inks(Enumerable.Range('0', 10).Select(face.Glyph)));
            Check(Math.Abs(drawn.Height - own.Height) <= 0.06 * own.Height && Math.Abs(drawn.Pen - own.Pen) <= 0.25 * own.Pen,
                $"{name}: drawn digits {drawn.Height:F0} high, pen {drawn.Pen:F0}; the font's {own.Height:F0}, {own.Pen:F0}");

            // Every contour clockwise (the same winding: overlaps fill once), inside the measured height and its slant.
            var drawing = FontMaps.Drawing;
            foreach (int c in Enumerable.Range('0', 10))
            {
                var outline = face.Outline(face.Glyph(c));
                var areas = Contours(outline).Select(p => StrokeDrawing.Area(p)).ToList();
                var (minX, minY, maxX, maxY) = outline.Bounds();
                Check(areas.All(a => a < 0), $"{name}: digit {(char)c} has a contour wound the other way");
                Check(minY >= -0.06 * own.Height && maxY <= 1.08 * own.Height && minX >= 0 && maxX <= face.Advance(face.Glyph(c)),
                    $"{name}: digit {(char)c} spans x {minX}..{maxX} (advance {face.Advance(face.Glyph(c))}), y {minY}..{maxY}");
                Check(maxX - minX <= 1.5 * own.Height + drawing.Slant * maxY, $"{name}: digit {(char)c} is {maxX - minX} wide");
            }

            // The same seed draws the same outlines; another seed others.
            var again = MappedFace(name);
            var other = Remap(raw, FontMaps.Drawing with { Seed = 8 });
            Check(Enumerable.Range('0', 10).All(c => face.Outline(face.Glyph(c)).Points.Span.SequenceEqual(again.Outline(again.Glyph(c)).Points.Span)), $"{name}: the same seed, the same digits");
            Check(!face.Outline(face.Glyph('3')).Points.Span.SequenceEqual(other.Outline(other.Glyph('3')).Points.Span), $"{name}: another seed, other digits");
        }

        // Turned off, nothing is drawn; a font that maps its own marks keeps them.
        var mcs = RawFace("MCS-HAND");
        var plain = Remap(mcs, FontMaps.Drawing with { Digits = false, Marks = false, Letters = false });
        Check(plain.Glyph('5') == 0 && plain.Glyph(0x0653) == 0 && plain.GlyphCount == mcs.GlyphCount, $"drawing off: '5' is {plain.Glyph('5')}, {plain.GlyphCount} glyphs");
        var mapped = MappedFace("MCS-HAND");
        Check(Enumerable.Range(0x064B, 8).All(c => mapped.Glyph(c) < mcs.GlyphCount), "MCS-HAND keeps its own marks");

        // The Unicode controls keep their own digits and marks.
        foreach (string name in UnicodeControls)
        {
            var raw = RawFace(name);
            var face = MappedFace(name);
            Check(Enumerable.Range('0', 10).All(c => face.Glyph(c) == raw.Glyph(c)) && face.GlyphCount == raw.GlyphCount, $"{name}: its digits are its own");
        }
    }

    // A raw face mapped by the PUA layout with other drawing settings (put back after).
    private static IFontFace Remap(IFontFace raw, FontMapDrawing drawing)
    {
        var before = FontMaps.Drawing;
        try
        {
            FontMaps.Drawing = drawing;
            return FontMaps.Apply(raw, FontMaps.Get(PuaMap));
        }
        finally
        {
            FontMaps.Drawing = before;
        }
    }

    private static IEnumerable<(double X, double Y)[]> Contours(GlyphOutline outline)
    {
        var verbs = outline.Verbs.ToArray();
        var points = outline.Points.ToArray();
        var current = new List<(double, double)>();
        int p = 0;
        foreach (var verb in verbs)
        {
            switch (verb)
            {
                case OutlineVerb.Move:
                case OutlineVerb.Line:
                    current.Add((points[p], points[p + 1]));
                    p += 2;
                    break;
                case OutlineVerb.Close:
                    yield return [.. current];
                    current.Clear();
                    break;
                default:
                    throw new InvalidOperationException("drawn glyphs have straight lines only");
            }
        }
    }

    private static void LegacyFontsLeaveOthersAlone(Device device)
    {
        // A font no map fits answers, measures and draws exactly as the OpenType format reads it.
        foreach (string file in UnicodeControls.Select(n => $"arabic/{n}.ttf").Concat(["arabic/amiri.ttf", "arabic/cairo.ttf", "dejavu-sans-subset.ttf", "dejavu-sans-nogsub.ttf", "features.ttf"]))
        {
            var raw = new OpenTypeFormat().Read(File.ReadAllBytes(FontData(file)), Path.GetFileNameWithoutExtension(file))[0];
            var face = FontFormats.Read(FontData(file)).Single();
            Check(FontMaps.MappedBy(face) is null && face.GetType() == raw.GetType() && face.GlyphCount == raw.GlyphCount, $"{file}: mapped by {FontMaps.MappedBy(face)?.Name}");
            Check(Enumerable.Range(0, 0x10000).All(c => face.Glyph(c) == raw.Glyph(c)) && Enumerable.Range(0, raw.GlyphCount).All(g => face.Advance(g) == raw.Advance(g)),
                $"{file}: the same glyphs and advances");
            Check(new[] { "GSUB", "GPOS", "GDEF", "cmap", "glyf" }.All(t => face.Table(t).Span.SequenceEqual(raw.Table(t).Span)), $"{file}: the same tables");
            string text = file.Contains("dejavu", StringComparison.Ordinal) || file.Contains("features", StringComparison.Ordinal) ? "Idrak 2026 بسم" : "بسم الله الرحمن ١٠ 23";
            Check(Render(face, text).SequenceEqual(Render(raw, text)), $"{file}: the same rendered line");
            Check(ReferenceEquals(FontMaps.Apply(raw), raw), $"{file}: Apply gives the same face back");
        }

        // A map beside a font maps it; turned off, it does not; a map that does not fit is refused naming its file.
        string folder = Path.Combine(Path.GetTempPath(), $"idrak-font-maps-{Guid.NewGuid():N}");
        Directory.CreateDirectory(folder);
        bool builtin = FontMaps.Builtin, sidecars = FontMaps.Sidecars;
        try
        {
            string font = Path.Combine(folder, "hand.ttf");
            File.Copy(FontData("arabic/MCS-HAND.ttf"), font);
            FontMaps.Builtin = false;
            Check(FontMaps.MappedBy(FontFormats.Read(font).Single()) is null, "the library's maps off: the legacy font is read as it is");
            var raw = RawFace("MCS-HAND");
            File.WriteAllText(Path.Combine(folder, "hand.map.json"), """
                // a map of two letters by glyph id, beside the font
                {"name": "beside", "scripts": ["arab", "DFLT"], "letters": {"U+0628": {"isol": "#BAA", "init": "#INBAA"}, "U+0627": {"isol": "U+F241", "fina": "U+F242"}}}
                """.Replace("BAA", raw.Glyph(0xF24C).ToString(System.Globalization.CultureInfo.InvariantCulture), StringComparison.Ordinal)
                .Replace("#IN" + raw.Glyph(0xF24C).ToString(System.Globalization.CultureInfo.InvariantCulture), "#" + raw.Glyph(0xF249).ToString(System.Globalization.CultureInfo.InvariantCulture), StringComparison.Ordinal),
                new UTF8Encoding(false));
            var beside = FontFormats.Read(font).Single();
            Check(FontMaps.MappedBy(beside)?.Name == "beside" && beside.Glyph(0x0628) == raw.Glyph(0xF24C) && beside.Glyph(0x062A) == 0,
                $"the map beside the font: {FontMaps.MappedBy(beside)?.Name}, ب {beside.Glyph(0x0628)}");
            Check(SynthCli(["data", "fonts", font, "-j"]) is var report && JsonNode.Parse(report.Out)!["fonts"]![0]!["mapped"]!["map"]!.GetValue<string>() == "beside",
                "data fonts names the map");
            FontMaps.Sidecars = false;
            Check(FontMaps.MappedBy(FontFormats.Read(font).Single()) is null, "the maps beside fonts off: not mapped");
            FontMaps.Sidecars = true;
            File.WriteAllText(Path.Combine(folder, "hand.map.json"), """{"name": "none", "scripts": ["arab"], "letters": {"U+0628": {"isol": "U+E999"}}}""");
            var refused = Refused<InvalidDataException>(() => FontFormats.Read(font), "a map that fits no letter of its font");
            Check(refused.Message.Contains("hand.map.json", StringComparison.Ordinal), refused.Message);

            foreach (var (json, key) in new[]
            {
                ("""{"scripts": ["arab"], "letters": {"U+0628": {"isol": "#1"}}}""", "name"),
                ("""{"name": "x", "scripts": ["arabic"], "letters": {"U+0628": {"isol": "#1"}}}""", "scripts"),
                ("""{"name": "x", "scripts": ["arab"], "letters": {"U+0628": {"init": "#1"}}}""", "letters.U+0628"),
                ("""{"name": "x", "scripts": ["arab"], "letters": {"U+0628": ["baa"]}}""", "letters.U+0628"),
                ("""{"name": "x", "scripts": ["arab"], "letters": {"U+0628": {"isol": "#1"}}, "marks": {"U+064E": {"glyph": "#2", "place": "over"}}}""", "marks.U+064E.place"),
                ("""{"name": "x", "scripts": ["arab"], "letters": {"U+0628": {"isol": "#1"}}, "ligatures": [{"feature": "rlig", "components": ["U+0644.init", "U+0627"], "glyph": "#3"}]}""", "ligatures[0].components"),
                ("""{"name": "x", "scripts": ["arab"], "letters": {"U+0628": {"isol": "#1"}}, "detect": {"share": 2}}""", "detect.share"),
            })
            {
                string path = Path.Combine(folder, "bad.map.json");
                File.WriteAllText(path, json);
                var e = Refused<InvalidDataException>(() => FontMap.Read(path), $"a bad map ({key})");
                Check(e.Message.StartsWith(path + ":", StringComparison.Ordinal) && e.Message.Contains($"'{key}'", StringComparison.Ordinal), e.Message);
            }
        }
        finally
        {
            FontMaps.Builtin = builtin;
            FontMaps.Sidecars = sidecars;
            Directory.Delete(folder, recursive: true);
        }

        Conformance.CheckFontMapper(FontMappers.OpenType, RawFace("MCS-HAND"), FontMaps.Get(PuaMap), FontFormats.Read(FontData("arabic/amiri.ttf")).Single()).ThrowIfFailed();
        Conformance.CheckFontMapper(FontMappers.OpenType, RawFace("FS_Hand_Style"), FontMaps.Get(NamesMap), FontFormats.Read(FontData("arabic/cairo.ttf")).Single()).ThrowIfFailed();
    }

    // The exception the call throws (a failed check when it throws none).
    private static T Refused<T>(Func<object> call, string what) where T : Exception
    {
        try
        {
            call();
        }
        catch (T e)
        {
            return e;
        }

        Check(false, $"{what}: not refused");
        throw new InvalidOperationException(what);
    }

    private static float[] Render(IFontFace face, string text)
    {
        var block = TextLayout.Layout(text, new TextStyle(face, 40));
        var ink = new float[600 * 80];
        TextRenderer.Draw(block, 10, 10, ink, 600, 80);
        return ink;
    }

    // The PUA layout's map without some characters (a font that lacks them), read from the library's data.
    private static FontMap LayoutWithout(Func<int, bool> drop)
    {
        using var stream = typeof(OpenTypeFormat).Assembly.GetManifestResourceStream("Idrak.Imaging.font-maps.arabic-pua-f200.json")!;
        var json = JsonNode.Parse(stream)!.AsObject();
        json["name"] = "pua-without";
        static int Code(string text) => Convert.ToInt32(text[2..].Split('.')[0], 16);
        foreach (string section in (string[])["letters", "characters", "marks"])
        {
            var obj = json[section]!.AsObject();
            foreach (string key in obj.Select(p => p.Key).Where(k => drop(Code(k))).ToList())
            {
                obj.Remove(key);
            }
        }

        var aliases = json["aliases"]!.AsObject();
        foreach (string key in aliases.Select(p => p.Key).Where(k => drop(Code(k)) || drop(Code((string)aliases[k]!))).ToList())
        {
            aliases.Remove(key);
        }

        var ligatures = json["ligatures"]!.AsArray();
        foreach (var lig in ligatures.Where(l => l!["components"]!.AsArray().Any(c => drop(Code((string)c!)))).ToList())
        {
            ligatures.Remove(lig);
        }

        return FontMap.Parse(Encoding.UTF8.GetBytes(json.ToJsonString()), "pua-without.map.json");
    }

    private static void LegacyFontsCompose(Device device)
    {
        foreach (string name in LegacyFonts)
        {
            var raw = RawFace(name);
            var face = MappedFace(name);
            ShapedGlyph[] Shape(IFontFace font, string text) => TextShapers.Shape(font, text, new TextRun(0, text.Length, "Arab", true));

            // The Allah ligature (liga) where the font has it.
            var allah = Shape(face, "الله");
            Check(allah.Length == 1 && allah[0].Glyph == raw.Glyph(0xF201), $"{name}: الله is {string.Join(" ", allah.Select(g => g.Glyph))}, the ligature {raw.Glyph(0xF201)}");

            // Letters the font lacks are composed from its own forms: پ from ب with three dots, ی from ى and dotless ي, in every form.
            foreach (int letter in (int[])[0x067E, 0x0686, 0x0698, 0x06A4, 0x06A9, 0x06AF, 0x06CC, 0x0671])
            {
                Check(raw.Glyph(letter) == 0 && face.Glyph(letter) >= raw.GlyphCount && !face.Outline(face.Glyph(letter)).IsEmpty, $"{name}: U+{letter:X4} is glyph {face.Glyph(letter)}");
            }

            var peh = Shape(face, "پپپ").Select(g => g.Glyph).Reverse().ToArray();
            var beh = Shape(face, "ببب").Select(g => g.Glyph).Reverse().ToArray();
            Check(peh.Distinct().Count() == 3 && peh.All(g => g >= raw.GlyphCount), $"{name}: پپپ is {string.Join(" ", peh)}");
            for (int i = 0; i < 3; i++)
            {
                int composed = Composition.Split(face.Outline(peh[i])).Count, body = Composition.Split(face.Outline(beh[i])).Count;
                Check(composed == body + 2 && face.Advance(peh[i]) == face.Advance(beh[i]), $"{name}: پ form {i} has {composed} contours, ب {body} (one dot becomes three)");
            }

            var yeh = Shape(face, "ییی").Select(g => g.Glyph).Reverse().ToArray();
            var ya = Shape(face, "ييي").Select(g => g.Glyph).Reverse().ToArray();
            Check(Composition.Split(face.Outline(yeh[0])).Count < Composition.Split(face.Outline(ya[0])).Count
                  && face.Outline(yeh[2]).Points.Span.SequenceEqual(face.Outline(Shape(face, "بى").First().Glyph).Points.Span),
                $"{name}: ی is dotless in its initial form and alef maksura's final form at the end");
        }

        // A font without the hamza seats and the Arabic-Indic digits: أ ؤ ئ are its seat with a hamza drawn on it (Unicode's
        // decomposition, the hamza placed through the anchors), لأ the lam-alef with the hamza, the digits drawn.
        var mcs = RawFace("MCS-HAND");
        var without = FontMaps.Apply(mcs, LayoutWithout(c => c is 0x0622 or 0x0623 or 0x0624 or 0x0625 or 0x0626 or >= 0x0660 and <= 0x0669 or >= 0x06F0 and <= 0x06F9
            or >= 0xFE81 and <= 0xFE8C or >= 0xFEF5 and <= 0xFEFA));
        foreach (var (text, seat) in new[] { ("أ", 0xF241), ("ؤ", 0xF2CB), ("إ", 0xF241) })
        {
            var g = TextShapers.Shape(without, text, new TextRun(0, text.Length, "Arab", true));
            var seatGlyph = g.SingleOrDefault(x => x.Glyph == mcs.Glyph(seat));
            var hamza = g.SingleOrDefault(x => x.Glyph >= mcs.GlyphCount);
            bool below = text == "إ";
            var ink = InkMeasure.Of(mcs.Outline(mcs.Glyph(seat)), mcs.Metrics.UnitsPerEm, 130)!.Value;
            var (_, minY, _, maxY) = without.Outline(hamza.Glyph).Bounds();
            Check(g.Length == 2 && seatGlyph.Glyph != 0 && hamza.Glyph != 0 && (below ? maxY + hamza.YOffset <= ink.Bottom : minY + hamza.YOffset >= ink.Top),
                $"{text}: {string.Join(" ", g.Select(x => $"{x.Glyph}({x.XOffset},{x.YOffset})"))}");
        }

        var lamHamza = TextShapers.Shape(without, "لأ", new TextRun(0, 2, "Arab", true));
        Check(lamHamza.Any(x => x.Glyph == mcs.Glyph(0xF2DC)) && lamHamza.Any(x => x.Glyph >= mcs.GlyphCount), $"لأ: {string.Join(" ", lamHamza.Select(x => x.Glyph))}");
        Check(Enumerable.Range(0x0660, 10).All(c => without.Glyph(c) >= mcs.GlyphCount) && Enumerable.Range(0x06F0, 10).All(c => without.Glyph(c) >= mcs.GlyphCount),
            "the Arabic-Indic digits are drawn when the font lacks them");

        // Turned off, nothing is composed.
        var plain = Remap(mcs, FontMaps.Drawing with { Letters = false });
        Check(plain.Glyph(0x067E) == 0, "letters off: پ is not composed");
    }

    private static void LegacyFontsSynth(Device device)
    {
        string folder = Path.Combine(Path.GetTempPath(), $"idrak-legacy-synth-{Guid.NewGuid():N}");
        try
        {
            string legacy = Path.Combine(folder, "legacy"), all = Path.Combine(folder, "all");
            Directory.CreateDirectory(legacy);
            Directory.CreateDirectory(all);
            foreach (string name in LegacyFonts)
            {
                File.Copy(FontData($"arabic/{name}.ttf"), Path.Combine(legacy, $"{name}.ttf"));
                File.Copy(FontData($"arabic/{name}.ttf"), Path.Combine(all, $"{name}.ttf"));
            }

            foreach (string name in UnicodeControls)
            {
                File.Copy(FontData($"arabic/{name}.ttf"), Path.Combine(all, $"{name}.ttf"));
            }

            foreach (var (fonts, text) in new[] { (legacy, "١٠ مينا ناجي عطية 6 10 7 23 صفر"), (all, "لا إله إلا الله، مينا ناجي عطية ١٠ ٢٣") })
            {
                string input = Path.Combine(folder, "text.txt");
                File.WriteAllText(input, text + "\n---\n" + text + "\n", new UTF8Encoding(false));
                string output = Path.Combine(folder, Path.GetFileName(fonts), "pages.jsonl");
                var (code, stdout, stderr) = SynthCli(["data", "synth", input, "-o", output, "--fonts", fonts, "--layout", "plain", "--scan", "none", "--page", "crop=1", "-n", "1", "-j"]);
                Check(code == 0, $"{Path.GetFileName(fonts)}: exit {code}: {stderr}");
                var json = JsonNode.Parse(stdout)!;
                Check((int)json["pages"]! == 2 && json["skipped"]!.AsArray().Count == 0, $"{Path.GetFileName(fonts)}: {json["pages"]} pages, left out: {json["skipped"]!.ToJsonString()}");
                Check(!stderr.Contains("no font has", StringComparison.Ordinal) && !stdout.Contains("no font has", StringComparison.Ordinal), $"{Path.GetFileName(fonts)}: {stdout}");
            }

            var off = SynthCli(["data", "synth", Path.Combine(folder, "text.txt"), "-o", Path.Combine(folder, "off", "pages.jsonl"), "--fonts", legacy, "--font-maps", "builtin=off", "-n", "1", "-j"]);
            Check(off.Code != 0 || JsonNode.Parse(off.Out)!["skipped"]!.AsArray().Count > 0, $"the library's maps off: the legacy fonts draw nothing: exit {off.Code}");
            var bad = SynthCli(["data", "synth", Path.Combine(folder, "text.txt"), "-o", Path.Combine(folder, "bad", "pages.jsonl"), "--fonts", legacy, "--font-maps", "colour=red"]);
            Check(bad.Code == 2 && bad.Err.Contains("colour", StringComparison.Ordinal), $"an unknown setting: exit {bad.Code} {bad.Err}");
            Check(FontMaps.Builtin && FontMaps.Drawing == new FontMapDrawing(), "the command puts the settings back");
        }
        finally
        {
            Directory.Delete(folder, recursive: true);
        }
    }
}
