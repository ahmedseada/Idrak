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
// tools/documents/make_fixtures.py). The fonts written here are also saved to test-output/fonts/written
// for tools/documents/check_font.py (fontTools and the OpenType Sanitizer).
internal abstract partial class DesktopTests
{
    internal static readonly (string Name, Action<Device> Run)[] FontWriteGroup =
    [
        ("fonts write: every test font (TrueType, CFF, a collection, variable fonts' default instance, and the legacy fonts as read) written as TrueType reads back with the same metrics, names, glyph count, characters, advances, outlines (the same points from TrueType and for a mapped face's drawn and composed glyphs; cubic curves within the curve tolerance and CFF's straight glyphs within half a unit, measured); the same face, the same bytes", FontWriteEveryFont),
        ("fonts write: subsets keep exactly the glyphs fontTools' subsetter keeps on the D8 samples (glyph 0, the glyphs asked for, the GSUB closure with layout, composite parts), under their own ids, the others empty with their advances; layout tables kept or dropped", FontWriteSubsets),
    ];

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
                faces.Add((read.Count > 1 ? $"{relative} #{i}" : relative, read[i]));
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
        var writer = new TrueTypeWriter();
        float tolerance = new FontWriteOptions().CurveTolerance;
        double worstCubic = 0, worstStraight = 0;
        int written = 0, exactGlyphs = 0, drawnGlyphs = 0, cubicGlyphs = 0, straightGlyphs = 0;
        foreach (var (name, face) in FontWriteFaces())
        {
            byte[] bytes = WriteFont(writer, face, new FontWriteOptions());
            Check(bytes.AsSpan().SequenceEqual(WriteFont(writer, face, new FontWriteOptions())), $"{name}: the same face wrote other bytes");
            SaveWritten($"{name}.written.ttf", bytes);
            written++;
            var read = new OpenTypeFormat().Read(bytes, "written")[0];
            Check(read.Metrics == face.Metrics && read.GlyphCount == face.GlyphCount && read.Family == face.Family && read.Style == face.Style,
                $"{name}: read back {read.Family} {read.Style}, {read.GlyphCount} glyphs, {read.Metrics}; the face {face.Family} {face.Style}, {face.GlyphCount}, {face.Metrics}");
            Check(new OpenTypeFormat().Read(bytes, "written").Single() is var again && FontMaps.MappedBy(again) is null && again.GlyphCount == face.GlyphCount,
                $"{name}: read back as OpenType, a map took it");
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
        }

        Console.WriteLine($"    {written} faces written and read back: {exactGlyphs} glyphs the same points, {drawnGlyphs} drawn and composed glyphs the same points, "
                          + $"{cubicGlyphs} cubic glyphs within {worstCubic:F3} units (tolerance {tolerance}), {straightGlyphs} other CFF glyphs (straight lines or none) within {worstStraight:F3}");
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
        var writer = new TrueTypeWriter();
        int subsets = 0;
        foreach (var (stem, file) in ((string, string)[])[("dejavu-sans-subset", "dejavu-sans-subset.ttf"), ("dejavu-sans-cff", "dejavu-sans-cff.otf"), ("features", "features.ttf"), ("amiri", "arabic/amiri.ttf")])
        {
            byte[] source = File.ReadAllBytes(FontData(file));
            var made = JsonNode.Parse(File.ReadAllText(Path.Combine(samples, $"{stem}.source.json")))!;
            Check((string)made["sha256"]! == Convert.ToHexStringLower(SHA256.HashData(source)), $"{stem}: the samples were made from another {file}: make them again (tools/documents/make_fixtures.py)");
            var face = new OpenTypeFormat().Read(File.ReadAllBytes(FontData(file)), Path.GetFileNameWithoutExtension(file)).Single();
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
}
