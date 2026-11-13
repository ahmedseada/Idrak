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

    // A word holding a pair: its two letters, then a joining letter when the second is in its medial form.
    private static string PairWord(FontMapLigature pair) =>
        char.ConvertFromUtf32(pair.Components[0].CodePoint) + char.ConvertFromUtf32(pair.Components[1].CodePoint) + (pair.Components[1].Form == "medi" ? "م" : "");

    private static void LegacyFontsPairs(Device device)
    {
        var pairs = FontMaps.Get(PuaMap).Ligatures.Where(l => l.Feature == FontMap.PairFeature).ToList();
        Check(pairs.Count >= 40 && pairs.All(p => p.Components.Count == 2), $"{pairs.Count} letter pairs in the layout");
        foreach (string name in LegacyFonts)
        {
            var raw = RawFace(name);
            var face = MappedFace(name);
            var mine = pairs.Where(p => p.Fonts.Count == 0 || p.Fonts.Contains(face.Family, StringComparer.OrdinalIgnoreCase)).ToList();
            Check(pairs.Count - mine.Count == (name == "FS_Hand_Style" ? 1 : 0), $"{name}: {mine.Count} of {pairs.Count} pairs are its own");
            foreach (var pair in mine)
            {
                int glyph = raw.Glyph(pair.Glyph[0].Value);
                string word = PairWord(pair);
                var run = new TextRun(0, word.Length, "Arab", true);
                var on = TextShapers.Shape(face, word, run, ["+" + FontMap.PairFeature]);
                var off = TextShapers.Shape(face, word, run);
                Check(glyph != 0 && !raw.Outline(glyph).IsEmpty && on.Any(g => g.Glyph == glyph) && !off.Any(g => g.Glyph == glyph) && off.Length == on.Length + 1,
                    $"{name}: {word} ({pair.Glyph[0]}) with pairs {string.Join(" ", on.Select(g => g.Glyph))}, without {string.Join(" ", off.Select(g => g.Glyph))}");
            }
        }

        // On a page: the same text whether pairs are joined or not; the choice per page, the same for the same seed.
        var content = new PageContent { Body = ["بجانب لحم مجلس نحن تحت يمين في بيت لجنة تمام", "خمسة لمس ممر لبن تين حمل جمل"], Language = "ar" };
        var style = new PageStyle { Fonts = [MappedFace("MCS-HAND")], Fallbacks = [SubsetFace()] };
        var before = FontMaps.Drawing;
        try
        {
            FontMaps.Drawing = before with { Pairs = 1 };
            var joined = PageLayouts.Get("plain").Render(content, style, 3);
            FontMaps.Drawing = before with { Pairs = 0 };
            var apart = PageLayouts.Get("plain").Render(content, style, 3);
            Check(joined.Text == apart.Text && joined.Description.Contains(", pairs", StringComparison.Ordinal) && !apart.Description.Contains(", pairs", StringComparison.Ordinal)
                  && !joined.Image.Pixels.AsSpan().SequenceEqual(apart.Image.Pixels), "pairs on and off: the same text, other ink");
            FontMaps.Drawing = before with { Pairs = 0.5 };
            var chosen = Enumerable.Range(1, 24).Select(seed => PageLayouts.Get("letter").Render(content, style, seed).Description.Contains(", pairs", StringComparison.Ordinal)).ToList();
            var again = Enumerable.Range(1, 24).Select(seed => PageLayouts.Get("letter").Render(content, style, seed).Description.Contains(", pairs", StringComparison.Ordinal)).ToList();
            Check(chosen.SequenceEqual(again) && chosen.Count(c => c) is > 4 and < 20, $"at 0.5, {chosen.Count(c => c)} of 24 pages join pairs, the same again");
        }
        finally
        {
            FontMaps.Drawing = before;
        }
    }

    private static void LegacyFontsSecondStyleAndWords(Device device)
    {
        var map = FontMaps.Get(PuaMap);
        var second = map.Ligatures.Where(l => l.Feature == FontMap.SecondStyleFeature).ToList();
        var words = map.Ligatures.Where(l => l.Feature == FontMap.WordFeature).ToList();
        Check(second.Count == 5 && words.Count == 2, $"{second.Count} second-style pairs, {words.Count} words in the layout");
        string[] pair = ["+" + FontMap.PairFeature], style2 = ["+" + FontMap.SecondStyleFeature], both = [.. pair, .. style2], word = ["+" + FontMap.WordFeature];
        foreach (string name in LegacyFonts)
        {
            var raw = RawFace(name);
            var face = MappedFace(name);
            ShapedGlyph[] Shape(string text, string[]? features) => TextShapers.Shape(face, text, new TextRun(0, text.Length, "Arab", true), features);
            foreach (var lig in second)
            {
                int glyph = raw.Glyph(lig.Glyph[0].Value);
                string text = string.Concat(lig.Components.Select(c => char.ConvertFromUtf32(c.CodePoint)));
                var first = map.Ligatures.FirstOrDefault(l => l.Feature == FontMap.PairFeature && l.Components.SequenceEqual(lig.Components));
                Check(glyph != 0 && !raw.Outline(glyph).IsEmpty && Shape(text, style2).Single().Glyph == glyph && Shape(text, both).Single().Glyph == glyph,
                    $"{name}: {text} in its second style is {lig.Glyph[0]}, the second style winning over the first");
                var plain = Shape(text, pair);
                Check(first is null ? plain.Length == 2 : plain.Single().Glyph == raw.Glyph(first.Glyph[0].Value) && plain.Single().Glyph != glyph,
                    $"{name}: {text} with the first style only: {string.Join(" ", plain.Select(g => g.Glyph))}");
            }

            foreach (var lig in words)
            {
                int glyph = raw.Glyph(lig.Glyph[0].Value);
                string text = string.Concat(lig.Components.Select(c => char.ConvertFromUtf32(c.CodePoint)));
                Check(glyph != 0 && Shape(text, word).Single().Glyph == glyph && Shape(text, null).Length == 3,
                    $"{name}: the word {text} drawn whole is {lig.Glyph[0]}, letter by letter without words");
            }

            Check(Shape("لمحة", word).Length == 4, $"{name}: لمحة (its hah medial) is no whole word");
        }

        // On a page: the same text with the second style and the words on or off; each chosen per page, the same for a seed.
        var content = new PageContent { Body = ["لمح الفريق التقرير ثم لم يتأخر", "وتم النشر بمقر الشركة لمح"], Language = "ar" };
        var style = new PageStyle { Fonts = [MappedFace("MCS-HAND")], Fallbacks = [SubsetFace()] };
        var before = FontMaps.Drawing;
        try
        {
            FontMaps.Drawing = before with { Pairs = 0, SecondStyle = 1, Words = 1 };
            var on = PageLayouts.Get("plain").Render(content, style, 3);
            FontMaps.Drawing = before with { Pairs = 0, SecondStyle = 0, Words = 0 };
            var off = PageLayouts.Get("plain").Render(content, style, 3);
            Check(on.Text == off.Text && on.Description.Contains(", second style, words", StringComparison.Ordinal) && !off.Description.Contains("second style", StringComparison.Ordinal)
                  && !off.Description.Contains("words", StringComparison.Ordinal) && !on.Image.Pixels.AsSpan().SequenceEqual(off.Image.Pixels),
                $"second style and words on and off: the same text, other ink ({on.Description} / {off.Description})");
            FontMaps.Drawing = before with { Pairs = 0.5, SecondStyle = 0.5, Words = 0.5 };
            string Chosen(int seed) => PageLayouts.Get("letter").Render(content, style, seed).Description;
            var first = Enumerable.Range(1, 24).Select(Chosen).ToList();
            var again = Enumerable.Range(1, 24).Select(Chosen).ToList();
            int seconds = first.Count(d => d.Contains("second style", StringComparison.Ordinal)), wholes = first.Count(d => d.Contains(", words", StringComparison.Ordinal));
            Check(first.SequenceEqual(again) && seconds is > 4 and < 20 && wholes is > 4 and < 20,
                $"at 0.5, {seconds} of 24 pages write the second style and {wholes} whole words, the same again");
        }
        finally
        {
            FontMaps.Drawing = before;
        }
    }

    // The signed area of a contour's points (control points included).
    private static double ContourArea(float[] p)
    {
        double sum = 0;
        for (int i = 0, n = p.Length / 2; i < n; i++)
        {
            int j = (i + 1) % n;
            sum += (double)p[2 * i] * p[2 * j + 1] - (double)p[2 * j] * p[2 * i + 1];
        }

        return sum / 2;
    }

    private static (float X0, float Y0, float X1, float Y1) ContourBox(float[] p)
    {
        float x0 = float.MaxValue, y0 = float.MaxValue, x1 = float.MinValue, y1 = float.MinValue;
        for (int i = 0; i < p.Length; i += 2)
        {
            (x0, y0, x1, y1) = (Math.Min(x0, p[i]), Math.Min(y0, p[i + 1]), Math.Max(x1, p[i]), Math.Max(y1, p[i + 1]));
        }

        return (x0, y0, x1, y1);
    }

    private static void LegacyFontsKehehGaf(Device device)
    {
        foreach (string name in LegacyFonts)
        {
            var face = MappedFace(name);
            float em = face.Metrics.UnitsPerEm;
            foreach (string form in FontMap.Forms)
            {
                var kaf = Composition.Split(face.Outline(ShapedForm(face, 0x0643, form)));
                var keheh = Composition.Split(face.Outline(ShapedForm(face, 0x06A9, form)));
                var gaf = Composition.Split(face.Outline(ShapedForm(face, 0x06AF, form)));
                var main = kaf.MaxBy(c => Math.Abs(ContourArea(c.Points)));
                var box = ContourBox(main.Points);
                int marks = kaf.Count(c => c != main && Math.Sign(ContourArea(c.Points)) == Math.Sign(ContourArea(main.Points))
                    && ContourBox(c.Points) is var b && b.X0 >= box.X0 && b.Y0 >= box.Y0 && b.X1 <= box.X1 && b.Y1 <= box.Y1);
                var innerMarks = kaf.Where(c => c != main && Math.Sign(ContourArea(c.Points)) == Math.Sign(ContourArea(main.Points))
                    && ContourBox(c.Points) is var b && b.X0 >= box.X0 && b.Y0 >= box.Y0 && b.X1 <= box.X1 && b.Y1 <= box.Y1).ToList();
                Check(innerMarks.Count == marks && !keheh.Any(c => innerMarks.Any(m => m.Points.AsSpan().SequenceEqual(c.Points))),
                    $"{name} {form}: ک keeps none of kaf's {marks} inner marks");

                // A stem the strip left upright gets the letter's own slanted top stroke; elsewhere ک is kaf less its marks.
                var top = Composition.TopStroke(main.Points, InkMeasure.Of(face.Outline(ShapedForm(face, 0x0643, form)), (int)em, 130)!.Value.Thickness);
                bool drawn = marks > 0 && top is { Upright: true };
                Check(drawn ? keheh.Count > kaf.Count - marks : keheh.Count == kaf.Count - marks,
                    $"{name} {form}: ک has {keheh.Count} contours, kaf {kaf.Count} with {marks} inner marks, {(drawn ? "a slanted top drawn" : "nothing drawn")}");
                if (drawn)
                {
                    // Its top now reaches beyond the stem (lam's top is the stem alone), and گ's bar runs parallel to the drawn stroke.
                    double pen = InkMeasure.Of(face.Outline(ShapedForm(face, 0x0643, form)), (int)em, 130)!.Value.Thickness;
                    double kehehTop = TopSpan(face, ShapedForm(face, 0x06A9, form)), lamTop = TopSpan(face, ShapedForm(face, 0x0644, form));
                    Check(kehehTop >= lamTop + pen && lamTop < 2 * pen, $"{name} {form}: ک's top spans {kehehTop:F0} units, ل's {lamTop:F0} (the stem), the pen {pen:F0}");
                    var strokePoints = keheh.Where(c => !kaf.Any(k => k.Points.AsSpan().SequenceEqual(c.Points))).SelectMany(c => c.Points).ToArray();
                    double strokeAngle = Axis(strokePoints), barAngle = Axis(gaf[^1].Points);
                    double difference = Math.Abs(strokeAngle - barAngle) % 180;
                    Check(Math.Min(difference, 180 - difference) < 4, $"{name} {form}: گ's bar at {barAngle:F1}°, the drawn top stroke at {strokeAngle:F1}°");
                }

                Check(gaf.Count == keheh.Count + 1, $"{name} {form}: گ has {gaf.Count} contours, ک {keheh.Count}: one bar more");
                var bar = ContourBox(gaf[^1].Points);
                var body = ContourBox(keheh.MaxBy(c => Math.Abs(ContourArea(c.Points))).Points);
                double length = Math.Sqrt(Math.Pow(bar.X1 - bar.X0, 2) + Math.Pow(bar.Y1 - bar.Y0, 2));
                Check((bar.Y0 + bar.Y1) / 2 > (body.Y0 + body.Y1) / 2 && length >= 0.12 * em, $"{name} {form}: گ's bar {bar} over the body {body}, {length / em:F2} em long");
            }
        }

        // The drawn stroke keeps the letter at its own height: ک no taller than kaf's tallest form and a pen (the stroke ends
        // half a pen over it, its round end half a pen more), within the one pixel ink is read at (256 pixels per em).
        var mohareb = MappedFace("mohareb-hand");
        int upem = mohareb.Metrics.UnitsPerEm;
        double tallest = FontMap.Forms.Max(f => InkMeasure.Of(mohareb.Outline(ShapedForm(mohareb, 0x0643, f)), upem, 130)!.Value.MaxY);
        double pen0 = InkMeasure.Of(mohareb.Outline(ShapedForm(mohareb, 0x0643, "isol")), upem, 130)!.Value.Thickness;
        foreach (string form in (string[])["isol", "fina"])
        {
            double top = InkMeasure.Of(mohareb.Outline(ShapedForm(mohareb, 0x06A9, form)), upem, 130)!.Value.MaxY;
            Check(top <= tallest + pen0 + upem / 256.0, $"mohareb-hand {form}: ک's ink reaches {top:F0}, kaf's tallest form {tallest:F0}, the pen {pen0:F0}");
        }

        // ک never reads as ل: in every font and form, the two glyphs' inks overlap on at most 80% of their union.
        foreach (string name in LegacyFonts)
        {
            var face = MappedFace(name);
            foreach (string form in FontMap.Forms)
            {
                double overlap = InkOverlap(face, ShapedForm(face, 0x06A9, form), ShapedForm(face, 0x0644, form));
                Check(overlap <= 0.8, $"{name} {form}: ک and ل share {overlap:P0} of their ink");
            }
        }
    }

    // The ink of a glyph at 256 pixels per em, origin at a fixed place: rows of booleans.
    private static bool[,] InkGrid(IFontFace face, int glyph)
    {
        const int Size = 640;
        float scale = 256f / face.Metrics.UnitsPerEm;
        var coverage = new float[Size * Size];
        OutlineRasterizers.Get().Fill(face.Outline(glyph), scale, 160, 420, coverage, Size, Size);
        var grid = new bool[Size, Size];
        for (int y = 0; y < Size; y++)
        {
            for (int x = 0; x < Size; x++)
            {
                grid[y, x] = coverage[y * Size + x] > 0.43f;
            }
        }

        return grid;
    }

    // The share of two glyphs' ink (the union) both cover.
    private static double InkOverlap(IFontFace face, int a, int b)
    {
        var (ga, gb) = (InkGrid(face, a), InkGrid(face, b));
        int both = 0, either = 0;
        for (int y = 0; y < ga.GetLength(0); y++)
        {
            for (int x = 0; x < ga.GetLength(1); x++)
            {
                both += ga[y, x] && gb[y, x] ? 1 : 0;
                either += ga[y, x] || gb[y, x] ? 1 : 0;
            }
        }

        return either == 0 ? 1 : (double)both / either;
    }

    // How wide, in font units, a glyph's ink is in the top fifth of its height.
    private static double TopSpan(IFontFace face, int glyph)
    {
        var grid = InkGrid(face, glyph);
        int top = -1, bottom = -1;
        for (int y = 0; y < grid.GetLength(0) && top < 0; y++)
        {
            for (int x = 0; x < grid.GetLength(1); x++)
            {
                top = grid[y, x] ? y : top;
            }
        }

        for (int y = grid.GetLength(0) - 1; y >= 0 && bottom < 0; y--)
        {
            for (int x = 0; x < grid.GetLength(1); x++)
            {
                bottom = grid[y, x] ? y : bottom;
            }
        }

        int left = int.MaxValue, right = -1;
        for (int y = top; y >= 0 && y <= top + (bottom - top) / 5; y++)
        {
            for (int x = 0; x < grid.GetLength(1); x++)
            {
                if (grid[y, x])
                {
                    (left, right) = (Math.Min(left, x), Math.Max(right, x));
                }
            }
        }

        return right < 0 ? 0 : (right - left + 1) * face.Metrics.UnitsPerEm / 256.0;
    }

    // The direction (degrees, -90 to 90) of the main axis of a set of points.
    private static double Axis(float[] p)
    {
        double mx = 0, my = 0;
        int n = p.Length / 2;
        for (int i = 0; i < p.Length; i += 2)
        {
            (mx, my) = (mx + p[i], my + p[i + 1]);
        }

        (mx, my) = (mx / n, my / n);
        double cxx = 0, cyy = 0, cxy = 0;
        for (int i = 0; i < p.Length; i += 2)
        {
            double dx = p[i] - mx, dy = p[i + 1] - my;
            (cxx, cyy, cxy) = (cxx + dx * dx, cyy + dy * dy, cxy + dx * dy);
        }

        double angle = 0.5 * Math.Atan2(2 * cxy, cxx - cyy) * 180 / Math.PI;
        return angle > 90 ? angle - 180 : angle < -90 ? angle + 180 : angle;
    }
}
