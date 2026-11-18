// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Buffers.Binary;
using System.Diagnostics;
using System.Runtime.InteropServices;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json.Nodes;
using Idrak.Abstraction.Testing;
using Idrak.Imaging;

// Plan 28, D8 (the built-in part): the "truetype" font writer, subsets (against fontTools' subsetter on the D8 samples,
// tools/documents/make_fixtures.py), legacy fonts converted once into complete TrueType fonts kept in a cache, and the
// page's hand features read from the font's own GSUB. The fonts written here are also saved to test-output/fonts/written
// for tools/documents/check_font.py (fontTools and the OpenType Sanitizer).
internal abstract partial class DesktopTests
{
    internal static readonly (string Name, Action<Device> Run)[] FontWriteGroup =
    [
        ("fonts write: every test font (TrueType, CFF, a collection, variable fonts' default instance, and the legacy fonts as their maps make them) written as TrueType reads back with the same metrics, names, glyph count, characters, advances, outlines (the same points from TrueType and for a mapped face's drawn and composed glyphs; cubic curves within the curve tolerance and CFF's straight glyphs within half a unit, measured) and shaping; the same face, the same bytes", FontWriteEveryFont),
        ("fonts write: subsets keep exactly the glyphs fontTools' subsetter keeps on the D8 samples (glyph 0, the glyphs asked for, the GSUB closure with layout, composite parts), under their own ids, the others empty with their advances; layout tables kept or dropped", FontWriteSubsets),
        ("fonts write: truetype is the library's writer and passes the font writer contract (TrueType and mapped faces with the same points, CFF within the tolerance); names given over the face's, hinting kept only when asked, a face of no tables (a cubic glyph, a character past the basic plane), the tables by tag with their checksums, a glyph the face lacks refused", FontWriteContract),
        ("fonts write: a legacy font is converted to TrueType once and kept: read back it is a plain font (the maps leave it alone) that shapes as its mapped face, the next read finds it in the cache (counted; cold and warm timed), another drawing setting makes another, a damaged one is made again, off reads the mapped face; --font-maps cache=off", FontWriteLegacyCache),
        ("fonts write: a page turns on the hand's features (PAIR, PAI2, WORD) from the font's own GSUB: the same answer as the map's rule for every mapped face and hand, the same for its converted font, none for Unicode fonts; a page of the converted font is the mapped face's, text, description and pixels", FontWriteHandFeatures),
        ("fonts write: cached and mapped faces are identical (decision 25): every legacy font under fonts/arabic a map fits gives, with the converted-font cache off and on, the same outline (every step and point) and advance for every glyph, the same characters and layout tables, as does every other library map that applies to it written and read back; generic pages (two layouts, three texts with drawn digits, marks and composed letters, the hand's features off and on) drawn from both are byte for byte the same", FontWriteCacheIdentical),
    ];

    private static readonly (string Text, string Script, bool RightToLeft)[] FontWriteLines =
    [
        ("Quarterly report: the office moves in May.", "Latn", false), ("fi fl ffi office café 2024", "Latn", false), ("AVATAR To Ta Wa", "Latn", false),
        ("تقرير المبيعات الشهري", "Arab", true), ("اجتماع الفريق يوم الثلاثاء", "Arab", true), ("لا ١٢٣ سعر", "Arab", true),
        ("لمح الفريق التقرير ثم لم يتأخر", "Arab", true), ("تَقرِيرٌ مُفَصَّل", "Arab", true),
    ];

    private static readonly string[] HandTags = [FontMap.PairFeature, FontMap.SecondStyleFeature, FontMap.WordFeature];

    // Every face of the repository's test fonts as the library reads them, a legacy font as its map makes it (the mapped
    // face itself, not a cached one).
    private static List<(string Name, IFontFace Face)> FontWriteFaces()
    {
        var faces = new List<(string, IFontFace)>();
        string root = Path.Combine(DataFolder(), "fonts");
        foreach (string path in Directory.EnumerateFiles(root, "*", SearchOption.AllDirectories).Order(StringComparer.Ordinal))
        {
            if (Path.GetExtension(path).ToLowerInvariant() is not (".ttf" or ".otf" or ".ttc"))
            {
                continue;
            }

            string relative = Path.GetRelativePath(root, path).Replace('\\', '/');
            var read = new OpenTypeFormat().Read(File.ReadAllBytes(path), Path.GetFileNameWithoutExtension(path));
            for (int i = 0; i < read.Count; i++)
            {
                faces.Add((read.Count > 1 ? $"{relative} #{i}" : relative, FontMaps.Apply(read[i])));
            }
        }

        return faces;
    }

    private static byte[] WriteFont(IFontWriter writer, IFontFace face, FontWriteOptions options)
    {
        using var stream = new MemoryStream();
        writer.Write(face, stream, options);
        return stream.ToArray();
    }

    // A written font kept in test-output/fonts/written for tools/documents/check_font.py (where there is a checkout).
    private static void SaveWritten(string name, byte[] bytes)
    {
        if (TestPaths.Root is not { } root)
        {
            return;
        }

        string folder = Path.Combine(root, "test-output", "fonts", "written");
        Directory.CreateDirectory(folder);
        File.WriteAllBytes(Path.Combine(folder, name.Replace('/', '-').Replace(" #", "-", StringComparison.Ordinal)), bytes);
    }

    // The glyph ids below which a face's outlines are the library's own TrueType records (written as they are).
    private static int RecordGlyphs(IFontFace face) => face switch
    {
        OpenTypeFace { HasGlyphRecords: true } => face.GlyphCount,
        MappedFontFace { Inner: OpenTypeFace { HasGlyphRecords: true } inner } => inner.GlyphCount,
        _ => 0,
    };

    // A glyph's left side bearing as the font's hmtx gives it.
    private static int Bearing(IFontFace face, int glyph)
    {
        var hmtx = face.Table("hmtx").Span;
        int metrics = BinaryPrimitives.ReadUInt16BigEndian(face.Table("hhea").Span[34..]);
        return BinaryPrimitives.ReadInt16BigEndian(hmtx[(glyph < metrics ? 4 * glyph + 2 : 4 * metrics + 2 * (glyph - metrics))..]);
    }

    private static int ContourCount(GlyphOutline outline)
    {
        int count = 0;
        foreach (var verb in outline.Verbs.Span)
        {
            count += verb == OutlineVerb.Move ? 1 : 0;
        }

        return count;
    }

    private static float BoundsDifference(GlyphOutline a, GlyphOutline b)
    {
        var (x0, y0, x1, y1) = a.Bounds();
        var (u0, v0, u1, v1) = b.Bounds();
        return MathF.Max(MathF.Max(MathF.Abs(x0 - u0), MathF.Abs(y0 - v0)), MathF.Max(MathF.Abs(x1 - u1), MathF.Abs(y1 - v1)));
    }

    // Each contour as points along it: `steps` points a segment (the start of each, its end on the last).
    private static List<List<(double X, double Y)>> Along(GlyphOutline outline, int steps)
    {
        var contours = new List<List<(double X, double Y)>>();
        var verbs = outline.Verbs.ToArray();
        var p = outline.Points.ToArray();
        int k = 0;
        List<(double X, double Y)>? current = null;
        (double X, double Y) at = default, start = default;
        foreach (var verb in verbs)
        {
            switch (verb)
            {
                case OutlineVerb.Move:
                    current = [];
                    contours.Add(current);
                    at = start = (p[k], p[k + 1]);
                    current.Add(at);
                    k += 2;
                    break;
                case OutlineVerb.Line:
                    {
                        var end = (p[k], p[k + 1]);
                        for (int s = 1; s <= steps; s++)
                        {
                            double t = s / (double)steps;
                            current!.Add((at.X + (end.Item1 - at.X) * t, at.Y + (end.Item2 - at.Y) * t));
                        }

                        at = end;
                        k += 2;
                        break;
                    }

                case OutlineVerb.Quadratic:
                    {
                        double cx = p[k], cy = p[k + 1], ex = p[k + 2], ey = p[k + 3];
                        for (int s = 1; s <= steps; s++)
                        {
                            double t = s / (double)steps, u = 1 - t;
                            current!.Add((u * u * at.X + 2 * u * t * cx + t * t * ex, u * u * at.Y + 2 * u * t * cy + t * t * ey));
                        }

                        at = (ex, ey);
                        k += 4;
                        break;
                    }

                case OutlineVerb.Cubic:
                    {
                        double ax = p[k], ay = p[k + 1], bx = p[k + 2], by = p[k + 3], ex = p[k + 4], ey = p[k + 5];
                        for (int s = 1; s <= steps; s++)
                        {
                            double t = s / (double)steps, u = 1 - t;
                            current!.Add((u * u * u * at.X + 3 * u * u * t * ax + 3 * u * t * t * bx + t * t * t * ex,
                                u * u * u * at.Y + 3 * u * u * t * ay + 3 * u * t * t * by + t * t * t * ey));
                        }

                        at = (ex, ey);
                        k += 6;
                        break;
                    }

                default:
                    current?.Add(start);                                        // the closing line back to the start
                    at = start;
                    break;
            }
        }

        return contours;
    }

    // How far the points of `a` stray from the polylines of `b`, contour by contour (infinite when they differ in number).
    private static double StrayFrom(List<List<(double X, double Y)>> a, List<List<(double X, double Y)>> b)
    {
        if (a.Count != b.Count)
        {
            return double.PositiveInfinity;
        }

        double worst = 0;
        for (int c = 0; c < a.Count; c++)
        {
            var line = b[c];
            foreach (var (x, y) in a[c])
            {
                double best = double.PositiveInfinity;
                for (int i = 0; i + 1 < line.Count; i++)
                {
                    var (x0, y0) = line[i];
                    var (x1, y1) = line[i + 1];
                    double dx = x1 - x0, dy = y1 - y0, length = dx * dx + dy * dy;
                    double t = length == 0 ? 0 : Math.Clamp(((x - x0) * dx + (y - y0) * dy) / length, 0, 1);
                    double ex = x0 + t * dx - x, ey = y0 + t * dy - y;
                    best = Math.Min(best, ex * ex + ey * ey);
                }

                worst = Math.Max(worst, Math.Sqrt(best));
            }
        }

        return worst;
    }

    // Both ways: the written outline's points from the source's curve, and the source's from the written one.
    private static double Deviation(GlyphOutline source, GlyphOutline written) =>
        Math.Max(StrayFrom(Along(written, 16), Along(source, 64)), StrayFrom(Along(source, 16), Along(written, 16)));

    private static void FontWriteEveryFont(Device device)
    {
        CpuOnly(device);
        var writer = FontWriters.Get(TrueTypeWriter.WriterName);
        float tolerance = new FontWriteOptions().CurveTolerance;
        double worstCubic = 0, worstStraight = 0;
        int written = 0, exactGlyphs = 0, drawnGlyphs = 0, cubicGlyphs = 0, straightGlyphs = 0, lines = 0;
        foreach (var (name, face) in FontWriteFaces())
        {
            byte[] bytes = WriteFont(writer, face, new FontWriteOptions());
            Check(bytes.AsSpan().SequenceEqual(WriteFont(writer, face, new FontWriteOptions())), $"{name}: the same face wrote other bytes");
            SaveWritten($"{name}.written.ttf", bytes);
            written++;
            var read = new OpenTypeFormat().Read(bytes, "written")[0];
            Check(read.Metrics == face.Metrics && read.GlyphCount == face.GlyphCount && read.Family == face.Family && read.Style == face.Style,
                $"{name}: read back {read.Family} {read.Style}, {read.GlyphCount} glyphs, {read.Metrics}; the face {face.Family} {face.Style}, {face.GlyphCount}, {face.Metrics}");
            Check(FontFormats.Read(bytes, "written").Single() is var again && FontMaps.MappedBy(again) is null && again.GlyphCount == face.GlyphCount,
                $"{name}: read back through FontFormats, a map took it");
            var gdef = read.Table("GDEF").Span;
            Check(read.Table("fvar").IsEmpty && (gdef.Length < 18 || BinaryPrimitives.ReadUInt16BigEndian(gdef[2..]) < 3 || BinaryPrimitives.ReadUInt32BigEndian(gdef[14..]) == 0),
                $"{name}: written as its default instance, its GDEF still names a variation store (the Sanitizer refuses one without fvar)");
            for (int c = 0; c <= 0x10FFFF; c++)
            {
                if (face.Glyph(c) != read.Glyph(c))
                {
                    Check(false, $"{name}: U+{c:X4} reads back as glyph {read.Glyph(c)}, the face's {face.Glyph(c)}");
                }
            }

            int records = RecordGlyphs(face);
            for (int g = 0; g < face.GlyphCount; g++)
            {
                Check(face.Advance(g) == read.Advance(g), $"{name}: glyph {g}'s advance {read.Advance(g)}, the face's {face.Advance(g)}");
                GlyphOutline a = face.Outline(g), b = read.Outline(g);
                if (g < records)
                {
                    Check(a.Verbs.Span.SequenceEqual(b.Verbs.Span) && a.Points.Span.SequenceEqual(b.Points.Span),
                        $"{name}: glyph {g} reads back with {b.Verbs.Length} steps and {b.Points.Length / 2} points, the face's {a.Verbs.Length} and {a.Points.Length / 2}");
                    Check(a.IsEmpty || Bearing(read, g) == Bearing(face, g), $"{name}: glyph {g}'s left side bearing {Bearing(read, g)}, the font's {Bearing(face, g)}");
                    exactGlyphs++;
                }
                else if (face is MappedFontFace)
                {
                    // A mapped face keeps its drawn and composed glyphs as the writer keeps them (decision 25).
                    Check(a.Verbs.Span.SequenceEqual(b.Verbs.Span) && a.Points.Span.SequenceEqual(b.Points.Span),
                        $"{name}: drawn glyph {g} reads back with {b.Verbs.Length} steps and {b.Points.Length / 2} points, the face's {a.Verbs.Length} and {a.Points.Length / 2}: "
                        + "the mapped face does not keep its drawn and composed glyphs as the TrueType writer does (their rounding to whole units, GlyphEncoder.Whole)");
                    drawnGlyphs++;
                }
                else if (a.Verbs.Span.Contains(OutlineVerb.Cubic))
                {
                    double deviation = Deviation(a, b);
                    worstCubic = Math.Max(worstCubic, deviation);
                    Check(deviation <= tolerance + 1, $"{name}: glyph {g}'s quadratics stray {deviation:F3} units from its cubic curves (tolerance {tolerance}, and the rounding to whole units)");
                    cubicGlyphs++;
                }
                else
                {
                    float difference = BoundsDifference(a, b);
                    worstStraight = Math.Max(worstStraight, difference);
                    Check(a.IsEmpty == b.IsEmpty && ContourCount(a) == ContourCount(b) && difference <= 0.5f + 1e-3f,
                        $"{name}: glyph {g} reads back with {ContourCount(b)} contours within {difference} units, the face's {ContourCount(a)}");
                    straightGlyphs++;
                }
            }

            int shaped = 0;
            foreach (var (text, script, rightToLeft) in FontWriteLines)
            {
                if (text.EnumerateRunes().Any(r => !Rune.IsWhiteSpace(r) && face.Glyph(r.Value) == 0))
                {
                    continue;                                                   // a line of characters the font lacks
                }

                var run = new TextRun(0, text.Length, script, rightToLeft);
                foreach (var features in (string[]?[])[null, [.. HandTags.Select(t => "+" + t)]])
                {
                    var expected = TextShapers.Shape(face, text, run, features);
                    var actual = TextShapers.Shape(read, text, run, features);
                    Check(expected.SequenceEqual(actual), $"{name}: \"{text}\" ({(features is null ? "default features" : "with the hand's")}) shapes otherwise: {Comparisons.Sequences(expected, actual)}");
                }

                shaped++;
            }

            Check(shaped > 0, $"{name}: no sample line it can set");
            lines += shaped;
        }

        Console.WriteLine($"    {written} faces written and read back: {exactGlyphs} glyphs the same points, {drawnGlyphs} drawn and composed glyphs the same points, "
                          + $"{cubicGlyphs} cubic glyphs within {worstCubic:F3} units (tolerance {tolerance}), {straightGlyphs} other CFF glyphs (straight lines or none) within {worstStraight:F3}; {lines} lines shaped the same");
    }

    // The D8 samples' folder (IDRAK_DOCUMENTS_SAMPLES, else test-output/documents/samples); missing, the test fails naming the command.
    private static string FontWriteSamples()
    {
        string folder = Environment.GetEnvironmentVariable("IDRAK_DOCUMENTS_SAMPLES") is { Length: > 0 } set ? set : RepositoryPath("test-output", "documents", "samples");
        Check(File.Exists(Path.Combine(folder, "d8", "amiri.subset-layout.ttf.json")),
            $"The D8 font samples are not in {folder}. Make them with: python -I tools/documents/make_fixtures.py -o test-output/documents/samples "
            + "(after python -m pip install -r tools/documents/requirements.txt; see tools/documents/README.md), or set IDRAK_DOCUMENTS_SAMPLES to where they are.");
        return Path.Combine(folder, "d8");
    }

    private static void FontWriteSubsets(Device device)
    {
        CpuOnly(device);
        string samples = FontWriteSamples();
        var writer = FontWriters.Get(TrueTypeWriter.WriterName);
        int subsets = 0;
        foreach (var (stem, file) in ((string, string)[])[("dejavu-sans-subset", "dejavu-sans-subset.ttf"), ("dejavu-sans-cff", "dejavu-sans-cff.otf"), ("features", "features.ttf"), ("amiri", "arabic/amiri.ttf")])
        {
            byte[] source = File.ReadAllBytes(FontData(file));
            var made = JsonNode.Parse(File.ReadAllText(Path.Combine(samples, $"{stem}.source.json")))!;
            Check((string)made["sha256"]! == Convert.ToHexStringLower(SHA256.HashData(source)), $"{stem}: the samples were made from another {file}: make them again (tools/documents/make_fixtures.py)");
            var face = FontFormats.Read(FontData(file)).Single();
            bool cff = file.EndsWith(".otf", StringComparison.Ordinal);
            foreach (bool layout in (bool[])[false, true])
            {
                string sample = $"{stem}.{(layout ? "subset-layout" : "subset")}.{Path.GetExtension(file)[1..]}";
                var expected = JsonNode.Parse(File.ReadAllText(Path.Combine(samples, sample + ".json")))!;
                Check((string)expected["sha256"]! == Convert.ToHexStringLower(SHA256.HashData(File.ReadAllBytes(Path.Combine(samples, sample)))), $"{sample}: not the sample its JSON describes: make the samples again");
                var glyphs = expected["font"]!["glyphs"]!;
                int[] requested = [.. glyphs["requested"]!.AsArray().Select(g => (int)g!)];
                int[] theirs = [.. glyphs["kept"]!.AsArray().Select(g => (int)g!)];
                var options = new FontWriteOptions { Glyphs = requested, KeepLayout = layout };
                var kept = TrueTypeWriter.Kept(face, options);
                int[] ours = [.. Enumerable.Range(0, kept.Length).Where(g => kept[g])];
                Check(ours.SequenceEqual(theirs), $"{sample}: {ours.Length} glyphs kept, fontTools' subsetter {theirs.Length}: here only {string.Join(" ", ours.Except(theirs))}; "
                                                  + $"fontTools' only {string.Join(" ", theirs.Except(ours))}");

                byte[] bytes = WriteFont(writer, face, options);
                SaveWritten($"{stem}.{(layout ? "subset-layout" : "subset")}.ttf", bytes);
                var read = new OpenTypeFormat().Read(bytes, sample)[0];
                Check(read.GlyphCount == face.GlyphCount, $"{sample}: {read.GlyphCount} glyphs, the face's {face.GlyphCount} (ids are kept)");
                for (int g = 0; g < face.GlyphCount; g++)
                {
                    GlyphOutline a = face.Outline(g), b = read.Outline(g);
                    Check(read.Advance(g) == face.Advance(g), $"{sample}: glyph {g}'s advance {read.Advance(g)}, the face's {face.Advance(g)}");
                    if (!kept[g])
                    {
                        Check(b.IsEmpty, $"{sample}: glyph {g}, left out, has an outline");
                    }
                    else if (!cff)
                    {
                        Check(a.Verbs.Span.SequenceEqual(b.Verbs.Span) && a.Points.Span.SequenceEqual(b.Points.Span), $"{sample}: kept glyph {g} reads back otherwise");
                    }
                    else
                    {
                        Check(a.IsEmpty == b.IsEmpty && Deviation(a, b) <= options.CurveTolerance + 1, $"{sample}: kept glyph {g} strays {Deviation(a, b):F3} units");
                    }
                }

                foreach (var rune in ((string)expected["font"]!["glyphs"]!["characters"]!).EnumerateRunes())
                {
                    int glyph = face.Glyph(rune.Value);
                    Check(read.Glyph(rune.Value) == (glyph != 0 && kept[glyph] ? glyph : 0), $"{sample}: U+{rune.Value:X4} reads back as {read.Glyph(rune.Value)}");
                }

                foreach (string tag in (string[])["GDEF", "GPOS", "GSUB"])
                {
                    Check(layout ? read.Table(tag).Span.SequenceEqual(face.Table(tag).Span) : read.Table(tag).IsEmpty,
                        $"{sample}: its {tag} is {read.Table(tag).Length} bytes, {(layout ? $"the face's {face.Table(tag).Length}" : "dropped without layout")}");
                }

                subsets++;
            }
        }

        Console.WriteLine($"    {subsets} subsets keep the glyphs fontTools' subsetter keeps");
    }

    // The file's structure: the directory by tag, each table 4-byte aligned with its checksum, the data in the order the
    // specification recommends, and the whole file summing to 0xB1B0AFBA.
    private static void FontWriteStructure(string name, byte[] font)
    {
        int count = BinaryPrimitives.ReadUInt16BigEndian(font.AsSpan(4));
        var tables = new List<(string Tag, uint Sum, int Offset, int Length)>();
        for (int i = 0; i < count; i++)
        {
            var r = font.AsSpan(12 + 16 * i, 16);
            tables.Add((Encoding.ASCII.GetString(r[..4]), BinaryPrimitives.ReadUInt32BigEndian(r[4..]), (int)BinaryPrimitives.ReadUInt32BigEndian(r[8..]),
                (int)BinaryPrimitives.ReadUInt32BigEndian(r[12..])));
        }

        Check(tables.Select(t => t.Tag).SequenceEqual(tables.Select(t => t.Tag).Order(StringComparer.Ordinal)), $"{name}: the directory is not in tag order");
        int power = 1 << (int)Math.Log2(count);
        Check(BinaryPrimitives.ReadUInt16BigEndian(font.AsSpan(6)) == 16 * power && BinaryPrimitives.ReadUInt16BigEndian(font.AsSpan(8)) == (int)Math.Log2(count),
            $"{name}: searchRange and entrySelector");
        foreach (var (tag, sum, offset, length) in tables)
        {
            var data = font.AsSpan(offset, (length + 3) & ~3).ToArray();
            if (tag == "head")
            {
                data.AsSpan(8, 4).Clear();
            }

            Check(offset % 4 == 0 && TrueTypeFont.Checksum(data) == sum, $"{name}: table {tag}'s checksum or alignment");
        }

        Check(TrueTypeFont.Checksum(font) == 0xB1B0AFBA, $"{name}: the file does not sum to 0xB1B0AFBA (head's checkSumAdjustment)");
        string[] recommended = ["head", "hhea", "maxp", "OS/2", "hmtx", "LTSH", "VDMX", "hdmx", "cmap", "fpgm", "prep", "cvt ", "loca", "glyf", "kern", "name", "post", "gasp", "PCLT", "DSIG"];
        var laid = tables.OrderBy(t => t.Offset).Select(t => t.Tag).ToList();
        var expected = recommended.Where(laid.Contains).Concat(laid.Where(t => !recommended.Contains(t)).Order(StringComparer.Ordinal)).ToList();
        Check(laid.SequenceEqual(expected), $"{name}: the tables are laid out {string.Join(" ", laid)}, the recommended order {string.Join(" ", expected)}");
    }

    private static void FontWriteContract(Device device)
    {
        CpuOnly(device);
        Check(FontWriters.Names.Contains(TrueTypeWriter.WriterName) && FontWriters.Origin(TrueTypeWriter.WriterName) == Overrides.Library
              && FontWriters.Default(TrueTypeWriter.WriterName) is TrueTypeWriter && FontWriters.For("report.TTF").Name == TrueTypeWriter.WriterName,
            $"the truetype writer registered by the library: {string.Join(", ", FontWriters.Names)}");
        var writer = FontWriters.Get(TrueTypeWriter.WriterName);
        var subset = SubsetFace();
        var features = FontFormats.Read(FontData("features.ttf")).Single();
        var amiri = FontFormats.Read(FontData("arabic/amiri.ttf")).Single();
        var cff = FontFormats.Read(FontData("dejavu-sans-cff.otf")).Single();
        var mapped = FontMaps.Apply(RawFace("MCS-HAND"));
        Conformance.CheckFontWriter(writer, [subset, features, amiri]).ThrowIfFailed();
        Conformance.CheckFontWriter(writer, [cff], exact: false).ThrowIfFailed();
        Conformance.CheckFontWriter(writer, [mapped]).ThrowIfFailed();                   // its drawn glyphs as the writer keeps them (decision 25)

        foreach (var (name, face) in new[] { ("dejavu-sans-subset", subset), ("dejavu-sans-cff", cff), ("MCS-HAND mapped", mapped) })
        {
            FontWriteStructure(name, WriteFont(writer, face, new FontWriteOptions()));
        }

        // Saved by its extension, the same bytes.
        string folder = Path.Combine(Path.GetTempPath(), $"idrak-font-write-{Guid.NewGuid():N}");
        try
        {
            string path = Path.Combine(folder, "sub", "features.ttf");
            FontWriters.Save(features, path);
            Check(File.ReadAllBytes(path).AsSpan().SequenceEqual(WriteFont(writer, features, new FontWriteOptions())), "FontWriters.Save wrote other bytes");
        }
        finally
        {
            if (Directory.Exists(folder))
            {
                Directory.Delete(folder, recursive: true);
            }
        }

        // The names: the caller's over the face's, every other record kept (the copyright and the licence).
        var names = new Dictionary<FontNameKind, string> { [FontNameKind.Family] = "Report Sans", [FontNameKind.Description] = "A test of the name table" };
        var named = new OpenTypeFormat().Read(WriteFont(writer, subset, new FontWriteOptions { Names = names }), "named")[0];
        static List<(int Id, string Text)> Records(IFontFace face)
        {
            var t = face.Table("name").Span;
            int count = BinaryPrimitives.ReadUInt16BigEndian(t[2..]), storage = BinaryPrimitives.ReadUInt16BigEndian(t[4..]);
            var list = new List<(int, string)>();
            for (int i = 0; i < count; i++)
            {
                var r = t.Slice(6 + 12 * i, 12);
                int platform = BinaryPrimitives.ReadUInt16BigEndian(r), id = BinaryPrimitives.ReadUInt16BigEndian(r[6..]);
                int length = BinaryPrimitives.ReadUInt16BigEndian(r[8..]), offset = BinaryPrimitives.ReadUInt16BigEndian(r[10..]);
                if (platform == 3)
                {
                    list.Add((id, Encoding.BigEndianUnicode.GetString(t.Slice(storage + offset, length))));
                }
            }

            return list;
        }

        var before = Records(subset);
        var after = Records(named);
        Check(named.Family == "Report Sans" && after.Contains((10, "A test of the name table")) && before.Any(r => r.Id is 0 or 13)
              && before.Where(r => r.Id is not (1 or 10)).All(after.Contains) && !after.Any(r => r.Id == 1 && r.Text != "Report Sans"),
            $"the names given over the face's: {named.Family}; {after.Count} records, the face's {before.Count}");

        // Hinting: dropped by default, kept for a whole face when asked (the programs and maxp's limits as the face's).
        var hinted = RawFace("MCS-HAND");
        var plain = new OpenTypeFormat().Read(WriteFont(writer, hinted, new FontWriteOptions()), "plain")[0];
        var kept = new OpenTypeFormat().Read(WriteFont(writer, hinted, new FontWriteOptions { KeepHinting = true }), "hinted")[0];
        var keptSubset = new OpenTypeFormat().Read(WriteFont(writer, hinted, new FontWriteOptions { KeepHinting = true, Glyphs = [3, 4] }), "subset")[0];
        Check(new[] { "fpgm", "cvt " }.All(t => !hinted.Table(t).IsEmpty && kept.Table(t).Span.SequenceEqual(hinted.Table(t).Span) && plain.Table(t).IsEmpty && keptSubset.Table(t).IsEmpty)
              && kept.Table("maxp").Span[14..26].SequenceEqual(hinted.Table("maxp").Span[14..26]),
            "hinting kept for a whole face when asked, dropped otherwise");

        // A face of the test's own, with no tables: a square of lines, a ring of quadratic curves, a circle of cubic ones,
        // a space, and a character past the basic plane.
        var own = new FontWriteTestFace();
        byte[] ownBytes = WriteFont(writer, own, new FontWriteOptions());
        FontWriteStructure("a face of no tables", ownBytes);
        SaveWritten("test-face.ttf", ownBytes);
        var back = new OpenTypeFormat().Read(ownBytes, "own")[0];
        Check(back.Family == own.Family && back.Style == own.Style && back.Metrics == own.Metrics && back.GlyphCount == own.GlyphCount, $"own face: {back.Family} {back.Style} {back.Metrics}");
        Check(back.Glyph(0x1F600) == 2 && back.Glyph('O') == 1 && back.Glyph(' ') == 3 && back.Glyph('x') == 0, "own face: its characters, one past the basic plane (cmap format 12)");
        Check(back.Outline(1).Verbs.Span.SequenceEqual(own.Outline(1).Verbs.Span) && back.Outline(1).Points.Span.SequenceEqual(own.Outline(1).Points.Span),
            "own face: the quadratic ring reads back the same");
        Check(BoundsDifference(own.Outline(0), back.Outline(0)) == 0 && Deviation(own.Outline(2), back.Outline(2)) <= new FontWriteOptions().CurveTolerance + 1 && back.Outline(3).IsEmpty,
            $"own face: the square's box, the circle within {Deviation(own.Outline(2), back.Outline(2)):F3} units");
        var os2 = back.Table("OS/2").Span;
        var post = back.Table("post").Span;
        Check(BinaryPrimitives.ReadUInt16BigEndian(os2) == 4 && (BinaryPrimitives.ReadUInt32BigEndian(os2[42..]) & 1) == 1 && (BinaryPrimitives.ReadUInt32BigEndian(os2[46..]) & (1u << 25)) != 0
              && BinaryPrimitives.ReadUInt16BigEndian(os2[62..]) == 0xA1 && BinaryPrimitives.ReadUInt32BigEndian(post) == 0x00030000 && BinaryPrimitives.ReadUInt16BigEndian(back.Table("head").Span[44..]) == 3,
            "own face: OS/2 made from it (version 4, Basic Latin and past the basic plane, bold italic), post 3 (no names), head's style");

        _ = Refused<ArgumentOutOfRangeException>(() => WriteFont(writer, subset, new FontWriteOptions { Glyphs = [subset.GlyphCount] }), "a subset of a glyph the face lacks");
    }

    private static void FontWriteLegacyCache(Device device)
    {
        CpuOnly(device);
        string folder = Path.Combine(Path.GetTempPath(), $"idrak-font-cache-{Guid.NewGuid():N}");
        bool enabled = FontCache.Enabled;
        var drawing = FontMaps.Drawing;
        try
        {
            Directory.CreateDirectory(folder);
            FontCache.Folder = folder;
            FontCache.Enabled = true;
            static void Touch(IFontFace face)
            {
                for (int g = 0; g < face.GlyphCount; g++)
                {
                    _ = face.Outline(g);
                }

                _ = face.Table("GSUB").Length + face.Table("GDEF").Length + face.Table("GPOS").Length;
            }

            int files = 0;
            foreach (string name in LegacyFonts)
            {
                string path = FontData($"arabic/{name}.ttf");
                FontCache.Enabled = false;
                long start = Stopwatch.GetTimestamp();
                var mapped = FontFormats.Read(path).Single();
                Touch(mapped);
                double mappedMs = Stopwatch.GetElapsedTime(start).TotalMilliseconds;
                Check(mapped is MappedFontFace, $"{name}: the cache off, the mapped face as before");

                FontCache.Enabled = true;
                long converted = FontCache.Converted, hits = FontCache.Hits;
                string[] keptBefore = Directory.GetFiles(folder, "*.ttf");
                start = Stopwatch.GetTimestamp();
                var cold = FontFormats.Read(path).Single();
                Touch(cold);
                double coldMs = Stopwatch.GetElapsedTime(start).TotalMilliseconds;
                files++;
                Check(FontCache.Converted == converted + 1 && FontCache.Hits == hits && Directory.GetFiles(folder, "*.ttf").Length == files && Directory.GetFiles(folder, "*.tmp").Length == 0,
                    $"{name}: converted {FontCache.Converted - converted}, found {FontCache.Hits - hits}; {Directory.GetFiles(folder).Length} files kept");
                var warmTimes = new List<double>();
                IFontFace warm = cold;
                for (int i = 0; i < 5; i++)
                {
                    start = Stopwatch.GetTimestamp();
                    warm = FontFormats.Read(path).Single();
                    Touch(warm);
                    warmTimes.Add(Stopwatch.GetElapsedTime(start).TotalMilliseconds);
                }

                Check(FontCache.Hits == hits + 5 && FontCache.Converted == converted + 1, $"{name}: the next reads found it kept ({FontCache.Hits - hits} of 5)");
                Check(cold is OpenTypeFace && warm is OpenTypeFace && FontMaps.MappedBy(warm) == FontMaps.MappedBy(mapped) && FontMaps.MappedBy(cold) is not null,
                    $"{name}: a plain TrueType face, still known as the map's");

                byte[] bytes = File.ReadAllBytes(Directory.GetFiles(folder, "*.ttf").Except(keptBefore).Single());
                SaveWritten($"{name}.converted.ttf", bytes);
                var plain = new OpenTypeFormat().Read(bytes, name)[0];
                Check(ReferenceEquals(FontMaps.Apply(plain), plain) && !plain.Table("GSUB").IsEmpty, $"{name}: the maps leave the converted font alone (it has its own GSUB)");

                // The mapped face's answers: characters, advances, every glyph's steps and points (its own, and the drawn and
                // composed ones as the writer keeps them, decision 25), the layout tables byte for byte, the shaping of every
                // line with and without the hand's features.
                Check(warm.GlyphCount == mapped.GlyphCount && warm.Family == mapped.Family && warm.Metrics == mapped.Metrics, $"{name}: {warm.GlyphCount} glyphs, the mapped face's {mapped.GlyphCount}");
                for (int c = 0; c <= 0x10FFFF; c++)
                {
                    if (warm.Glyph(c) != mapped.Glyph(c))
                    {
                        Check(false, $"{name}: U+{c:X4} is glyph {warm.Glyph(c)}, the mapped face's {mapped.Glyph(c)}");
                    }
                }

                int own = ((MappedFontFace)mapped).Inner.GlyphCount;
                for (int g = 0; g < mapped.GlyphCount; g++)
                {
                    GlyphOutline a = mapped.Outline(g), b = warm.Outline(g);
                    if (warm.Advance(g) != mapped.Advance(g) || !a.Verbs.Span.SequenceEqual(b.Verbs.Span) || !a.Points.Span.SequenceEqual(b.Points.Span))
                    {
                        Check(false, $"{name}: glyph {g} ({(g < own ? "the font's own" : "drawn")}) reads back with {b.Verbs.Length} steps, {b.Points.Length / 2} points and advance {warm.Advance(g)}, "
                                     + $"the mapped face's {a.Verbs.Length}, {a.Points.Length / 2} and {mapped.Advance(g)}"
                                     + (g < own ? "" : ": the mapped face does not keep its drawn and composed glyphs as the TrueType writer does (their rounding to whole units, GlyphEncoder.Whole)"));
                    }
                }

                foreach (string tag in (string[])["GSUB", "GDEF", "GPOS"])
                {
                    Check(warm.Table(tag).Span.SequenceEqual(mapped.Table(tag).Span), $"{name}: its {tag} is not the mapped face's");
                }

                foreach (var (text, script, rightToLeft) in FontWriteLines.Where(l => l.Script == "Arab"))
                {
                    var run = new TextRun(0, text.Length, script, rightToLeft);
                    foreach (var features in (string[]?[])[null, [.. HandTags.Select(t => "+" + t)]])
                    {
                        Check(TextShapers.Shape(warm, text, run, features).SequenceEqual(TextShapers.Shape(mapped, text, run, features)), $"{name}: \"{text}\" shapes otherwise converted");
                    }
                }

                warmTimes.Sort();
                Console.WriteLine($"    {name}: mapped and drawn at every read {mappedMs:F0} ms; converted once {coldMs:F0} ms (cold); read from the cache {warmTimes[2]:F1} ms (warm, median of 5)");
            }

            // Another drawing setting is another font; a page's chance of the hand's features is not.
            string mcs = FontData("arabic/MCS-HAND.ttf");
            long made = FontCache.Converted, found = FontCache.Hits;
            FontMaps.Drawing = drawing with { Seed = drawing.Seed + 1 };
            _ = FontFormats.Read(mcs);
            FontMaps.Drawing = drawing with { Pairs = 0.25, Words = 1 };
            _ = FontFormats.Read(mcs);
            FontMaps.Drawing = drawing;
            Check(FontCache.Converted == made + 1 && FontCache.Hits == found + 1, $"another seed: {FontCache.Converted - made} converted; other chances: {FontCache.Hits - found} found");

            // A damaged kept file is made again.
            foreach (string file in Directory.GetFiles(folder, "*.ttf"))
            {
                File.WriteAllBytes(file, File.ReadAllBytes(file)[..100]);
            }

            made = FontCache.Converted;
            var again = FontFormats.Read(mcs).Single();
            Check(FontCache.Converted == made + 1 && again is OpenTypeFace && again.GlyphCount == FontMaps.Apply(RawFace("MCS-HAND")).GlyphCount, "a damaged kept font is converted again");

            // The variable's values, and the command's setting (put back after).
            Check(FontCache.FolderFrom(null) == Path.Combine(ComputeResources.CacheFolder, "fonts") && FontCache.FolderFrom("on") == FontCache.FolderFrom(null)
                  && FontCache.FolderFrom("kept-fonts") == Path.GetFullPath("kept-fonts"), $"IDRAK_FONT_CACHE: {FontCache.FolderFrom(null)}");
            var (code, output, error) = SynthCli(["data", "fonts", mcs, "--font-maps", "cache=off", "-j"]);
            Check(code is 0 or 1 && (string?)JsonNode.Parse(output)!["fonts"]![0]!["mapped"]?["map"] == PuaMap && FontCache.Enabled && FontCache.Folder == Path.GetFullPath(folder),
                $"data fonts --font-maps cache=off: exit {code}, {error}; the settings put back");
            Check(SynthCli(["data", "fonts", mcs, "--font-maps", "cache="]).Code == 2, "--font-maps cache= with nothing: a usage error");
        }
        finally
        {
            FontCache.Enabled = enabled;
            FontCache.Folder = null;
            FontMaps.Drawing = drawing;
            if (Directory.Exists(folder))
            {
                Directory.Delete(folder, recursive: true);
            }
        }
    }

    // The rule before the fix: the hand's features the face's map lists ligatures under.
    private static string[]? HandRuleOfTheMap(IFontFace face, IReadOnlyList<string> hand)
    {
        if (hand.Count == 0 || FontMaps.MappedBy(face) is not { } map)
        {
            return null;
        }

        var on = hand.Where(tag => map.Ligatures.Any(l => l.Feature == tag)).Select(tag => "+" + tag).ToArray();
        return on.Length == 0 ? null : on;
    }

    private static void FontWriteHandFeatures(Device device)
    {
        CpuOnly(device);
        var writer = FontWriters.Get(TrueTypeWriter.WriterName);
        var hands = Enumerable.Range(0, 1 << HandTags.Length).Select(m => HandTags.Where((_, i) => (m & (1 << i)) != 0).ToArray()).ToList();
        static bool Same(string[]? a, string[]? b) => a is null ? b is null : b is not null && a.SequenceEqual(b);
        var faces = LegacyFonts.Select(n => (Name: $"{n} ({PuaMap})", Face: FontMaps.Apply(RawFace(n), FontMaps.Get(PuaMap)))).ToList();
        faces.Add(($"FS_Hand_Style ({NamesMap})", FontMaps.Apply(RawFace("FS_Hand_Style"), FontMaps.Get(NamesMap))));
        int compared = 0;
        foreach (var (name, face) in faces)
        {
            var converted = new OpenTypeFormat().Read(WriteFont(writer, face, new FontWriteOptions()), name)[0];
            foreach (var hand in hands)
            {
                string[]? before = HandRuleOfTheMap(face, hand), now = PageDrawing.Asked(face, hand), fromFont = PageDrawing.Asked(converted, hand);
                Check(Same(before, now) && Same(now, fromFont), $"{name}, hand [{string.Join(" ", hand)}]: the map's rule [{string.Join(" ", before ?? Array.Empty<string>())}], "
                                                                 + $"the GSUB's [{string.Join(" ", now ?? Array.Empty<string>())}], the converted font's [{string.Join(" ", fromFont ?? Array.Empty<string>())}]");
                compared++;
            }
        }

        foreach (string file in UnicodeControls.Select(n => $"arabic/{n}.ttf").Append("arabic/amiri.ttf").Append("dejavu-sans-subset.ttf"))
        {
            var face = FontFormats.Read(FontData(file)).Single();
            Check(PageDrawing.Asked(face, HandTags) is null, $"{file}: a Unicode font turned on {string.Join(" ", PageDrawing.Asked(face, HandTags) ?? Array.Empty<string>())}");
        }

        // A page drawn with the converted font: the mapped face's text and description (its hand's features on).
        var mcsMapped = FontMaps.Apply(RawFace("MCS-HAND"));
        var mcsConverted = new OpenTypeFormat().Read(WriteFont(writer, mcsMapped, new FontWriteOptions()), "MCS-HAND")[0];
        var content = new PageContent { Body = ["لمح الفريق التقرير ثم لم يتأخر", "بجانب لحم مجلس نحن تحت يمين في بيت لجنة تمام"], Language = "ar" };
        var drawing = FontMaps.Drawing;
        try
        {
            FontMaps.Drawing = drawing with { Pairs = 1, SecondStyle = 1, Words = 1 };
            var fromMapped = PageLayouts.Get("plain").Render(content, new PageStyle { Fonts = [mcsMapped], Fallbacks = [SubsetFace()] }, 3);
            var fromConverted = PageLayouts.Get("plain").Render(content, new PageStyle { Fonts = [mcsConverted], Fallbacks = [SubsetFace()] }, 3);
            Check(fromMapped.Description.Contains(", pairs, second style, words", StringComparison.Ordinal) && fromConverted.Description == fromMapped.Description
                  && fromConverted.Text == fromMapped.Text, $"the converted font's page: {fromConverted.Description}; the mapped face's {fromMapped.Description}");
            var (same, changed, largest) = ComparePageBytes(fromMapped, fromConverted);
            Check(same, $"the converted font's page differs from the mapped face's in {changed} values, by up to {largest:F3} levels: "
                        + "the mapped face does not keep its drawn and composed glyphs as the TrueType writer does (their rounding to whole units, GlyphEncoder.Whole, decision 25)");
            Console.WriteLine($"    {compared} faces and hands asked the same; the converted font's page is the mapped face's, byte for byte");
        }
        finally
        {
            FontMaps.Drawing = drawing;
        }
    }

    // Plan 28, decision 25 (D8c): a legacy font's mapped face (the converted-font cache off) and its converted font (on)
    // are the same face, glyph for glyph, and draw the same pages byte for byte.
    private static void FontWriteCacheIdentical(Device device)
    {
        CpuOnly(device);
        string folder = Path.Combine(Path.GetTempPath(), $"idrak-font-cache-same-{Guid.NewGuid():N}");
        bool enabled = FontCache.Enabled;
        var drawing = FontMaps.Drawing;
        var writer = FontWriters.Get(TrueTypeWriter.WriterName);
        var fallback = SubsetFace();
        (string Name, PageContent Content)[] contents =
        [
            ("a report", new PageContent { Body = ["لمح الفريق التقرير ثم لم يتأخر", "بجانب لحم مجلس نحن تحت يمين في بيت لجنة تمام"], Language = "ar" }),
            ("marks and digits", new PageContent { Title = "تَقرِيرٌ مُفَصَّل", Body = ["اِجتِمَاعُ الفَرِيقِ يَومَ الثُّلَاثَاءِ", "المبيعات 2026 و ١٢٣ وحدة"], Language = "ar" }),
            ("composed letters", new PageContent { Body = ["گزارش پروژه چاپ شد", "کتاب ژورنال ڤيديو ٱلشركة"], Language = "ar" }),
        ];
        int fonts = 0, faces = 0, glyphs = 0, pages = 0;
        try
        {
            FontCache.Folder = folder;
            foreach (string path in Directory.EnumerateFiles(Path.Combine(DataFolder(), "fonts", "arabic"), "*.ttf").Order(StringComparer.Ordinal))
            {
                string name = Path.GetFileNameWithoutExtension(path);
                FontCache.Enabled = false;
                var mapped = FontFormats.Read(path).Single();
                if (FontMaps.MappedBy(mapped) is not { } map)
                {
                    continue;                                                   // a Unicode font: read as it is either way
                }

                FontCache.Enabled = true;
                var converted = FontFormats.Read(path).Single();
                Check(mapped is MappedFontFace && converted is OpenTypeFace, $"{name}: the cache off gives {mapped.GetType().Name}, on {converted.GetType().Name}");
                glyphs += SameFace($"{name} ({map.Name}, the cache off and on)", mapped, converted);
                faces++;

                // Every other library map the mapper applies to the font, converted as the cache converts.
                var raw = new OpenTypeFormat().Read(File.ReadAllBytes(path), name)[0];
                foreach (string other in FontMaps.Names.Where(n => n != map.Name && FontMaps.Origin(n) == Overrides.Library).Order(StringComparer.Ordinal))
                {
                    if (FontMappers.Get(FontMaps.Mapper).Map(raw, FontMaps.Get(other), detect: false, FontMaps.Drawing) is not { } face)
                    {
                        continue;
                    }

                    var written = new OpenTypeFormat().Read(WriteFont(writer, face, new FontWriteOptions { KeepLayout = true, KeepHinting = true }), name)[0];
                    glyphs += SameFace($"{name} ({other}, written)", face, written);
                    faces++;
                }

                foreach (bool hand in (bool[])[false, true])
                {
                    foreach (string layout in (string[])["plain", "letter"])
                    {
                        foreach (var (what, content) in contents)
                        {
                            FontMaps.Drawing = hand ? drawing with { Pairs = 1, SecondStyle = 1, Words = 1 } : drawing;
                            var off = PageLayouts.Get(layout).Render(content, new PageStyle { Fonts = [mapped], Fallbacks = [fallback], Width = 900, Height = 700 }, 3);
                            var on = PageLayouts.Get(layout).Render(content, new PageStyle { Fonts = [converted], Fallbacks = [fallback], Width = 900, Height = 700 }, 3);
                            FontMaps.Drawing = drawing;
                            var (same, changed, largest) = ComparePageBytes(off, on);
                            Check(same && off.Text == on.Text && off.Description == on.Description,
                                $"{name}, {layout}, {what}{(hand ? ", the hand's features on" : "")}: the page with the cache on differs from the page with it off in {changed} values, "
                                + $"by up to {largest:F3} levels{(off.Description == on.Description ? "" : $" (described {on.Description}, off {off.Description})")}: the mapped face does not keep "
                                + "its drawn and composed glyphs as the TrueType writer does (their rounding to whole units, GlyphEncoder.Whole)");
                            pages++;
                        }
                    }
                }

                fonts++;
            }

            Check(fonts >= LegacyFonts.Length, $"{fonts} legacy fonts under fonts/arabic, the test fonts {string.Join(", ", LegacyFonts)} at least");
            Console.WriteLine($"    {fonts} legacy fonts, {faces} mapped faces: {glyphs} glyphs the same steps, points and advances with the cache off and on; {pages} pages byte for byte the same");
        }
        finally
        {
            FontCache.Enabled = enabled;
            FontCache.Folder = null;
            FontMaps.Drawing = drawing;
            if (Directory.Exists(folder))
            {
                Directory.Delete(folder, recursive: true);
            }
        }
    }

    // A mapped face and its converted font as the same face: names, metrics, every glyph's steps, points and advance,
    // every character and the layout tables. Returns the glyphs compared.
    private static int SameFace(string name, IFontFace mapped, IFontFace converted)
    {
        Check(converted.GlyphCount == mapped.GlyphCount && converted.Metrics == mapped.Metrics && converted.Family == mapped.Family && converted.Style == mapped.Style,
            $"{name}: {converted.Family} {converted.Style}, {converted.GlyphCount} glyphs, {converted.Metrics}; mapped {mapped.Family} {mapped.Style}, {mapped.GlyphCount}, {mapped.Metrics}");
        int own = ((MappedFontFace)mapped).Inner.GlyphCount;
        for (int g = 0; g < mapped.GlyphCount; g++)
        {
            GlyphOutline a = mapped.Outline(g), b = converted.Outline(g);
            if (mapped.Advance(g) != converted.Advance(g) || !a.Verbs.Span.SequenceEqual(b.Verbs.Span) || !a.Points.Span.SequenceEqual(b.Points.Span))
            {
                Check(false, $"{name}: glyph {g} ({(g < own ? "the font's own" : "drawn or composed")}) has {b.Verbs.Length} steps, {b.Points.Length / 2} points and advance {converted.Advance(g)} "
                             + $"converted, {a.Verbs.Length}, {a.Points.Length / 2} and {mapped.Advance(g)} mapped"
                             + (g < own ? "" : ": the mapped face does not keep its drawn and composed glyphs as the TrueType writer does (their rounding to whole units, GlyphEncoder.Whole)"));
            }
        }

        for (int c = 0; c <= 0x10FFFF; c++)
        {
            if (mapped.Glyph(c) != converted.Glyph(c))
            {
                Check(false, $"{name}: U+{c:X4} is glyph {converted.Glyph(c)} converted, {mapped.Glyph(c)} mapped");
            }
        }

        foreach (string tag in (string[])["GSUB", "GDEF", "GPOS"])
        {
            Check(converted.Table(tag).Span.SequenceEqual(mapped.Table(tag).Span), $"{name}: its {tag} converted is {converted.Table(tag).Length} bytes, mapped {mapped.Table(tag).Length}, and not the same");
        }

        return mapped.GlyphCount;
    }

    // Whether two pages are the same image byte for byte; else how many of their values differ, by how many levels at most.
    private static (bool Same, int Changed, double Largest) ComparePageBytes(RenderedPage a, RenderedPage b)
    {
        float[] p = a.Image.Pixels, q = b.Image.Pixels;
        if (a.Image.Width != b.Image.Width || a.Image.Height != b.Image.Height || a.Image.Channels != b.Image.Channels || p.Length != q.Length)
        {
            return (false, Math.Max(p.Length, q.Length), double.PositiveInfinity);
        }

        if (MemoryMarshal.AsBytes(p.AsSpan()).SequenceEqual(MemoryMarshal.AsBytes(q.AsSpan())))
        {
            return (true, 0, 0);
        }

        int changed = 0;
        double largest = 0;
        for (int i = 0; i < p.Length; i++)
        {
            if (BitConverter.SingleToInt32Bits(p[i]) != BitConverter.SingleToInt32Bits(q[i]))
            {
                changed++;
                largest = Math.Max(largest, Math.Abs(p[i] - q[i]) * 255);
            }
        }

        return (false, changed, largest);
    }

    // A face of the test's own, no tables: a square of lines, a ring of quadratic curves (closed on its start), a circle of
    // cubic curves, and a space; 'O', 'C', a space and U+1F600 (past the basic plane).
    private sealed class FontWriteTestFace : IFontFace
    {
        private const float K = 0.5523f * 250f;

        private static readonly GlyphOutline[] Outlines =
        [
            new([OutlineVerb.Move, OutlineVerb.Line, OutlineVerb.Line, OutlineVerb.Line, OutlineVerb.Close], [50, 0, 550, 0, 550, 700, 50, 700]),
            new([OutlineVerb.Move, OutlineVerb.Quadratic, OutlineVerb.Quadratic, OutlineVerb.Quadratic, OutlineVerb.Quadratic, OutlineVerb.Close],
                [100, 100, 300, 0, 500, 100, 600, 300, 500, 500, 300, 600, 100, 500, 0, 300, 100, 100]),
            new([OutlineVerb.Move, OutlineVerb.Cubic, OutlineVerb.Cubic, OutlineVerb.Cubic, OutlineVerb.Cubic, OutlineVerb.Close],
                [550, 350, 550, 350 + K, 300 + K, 600, 300, 600, 300 - K, 600, 50, 350 + K, 50, 350, 50, 350 - K, 300 - K, 100, 300, 100, 300 + K, 100, 550, 350 - K, 550, 350]),
            GlyphOutline.Empty,
        ];

        public string Family => "Test Sans";

        public string Style => "Bold Italic";

        public string Format => "test";

        public FontMetrics Metrics => new(1000, 800, -200, 90);

        public int GlyphCount => Outlines.Length;

        public int Glyph(int codePoint) => codePoint switch { 'O' => 1, 'C' => 2, ' ' => 3, 0x1F600 => 2, _ => 0 };

        public int Advance(int glyph) => glyph == 3 ? 250 : 600;

        public GlyphOutline Outline(int glyph) => (uint)glyph < (uint)Outlines.Length ? Outlines[glyph] : GlyphOutline.Empty;

        public ReadOnlyMemory<byte> Table(string tag) => ReadOnlyMemory<byte>.Empty;
    }
}
