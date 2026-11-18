// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Buffers.Binary;
using System.Diagnostics;
using System.IO.Compression;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json.Nodes;
using Idrak.Abstraction.Plugins;
using Idrak.Abstraction.Testing;
using Idrak.Imaging;
using Idrak.Imaging.Plugins;
using Idrak.Imaging.Plugins.Fonts;

// Plan 28, D8 (the plug-in part): WOFF 1.0 read and written by Idrak.Imaging.Plugins, a font format registered into
// FontFormats and FontWriters from outside the library (decisions 20 and 21), against fontTools' WOFF samples
// (tools/documents/make_fixtures.py: their dumps and HarfBuzz's shaping) and the truetype writer it packs. The WOFF files
// written here are kept in test-output/fonts/woff for tools/documents/check_font.py (fontTools and the OpenType Sanitizer).
internal abstract partial class DesktopTests
{
    internal static readonly (string Name, Action<Device> Run)[] FontWoffGroup =
    [
        ("fonts woff: every D8 sample .woff (fontTools' WOFF 1.0 of the four test fonts) reads through FontFormats with the plug-in: every table byte for byte its source's, the dump's glyph count, table lengths, advances and characters, the source face's outlines and characters exactly, and the sample lines shaped as from the source (and as HarfBuzz shapes them wherever it shapes the source the same)", WoffSamples),
        ("fonts woff: every test font (TrueType, CFF, a collection, variable fonts' default instance, the legacy fonts as mapped) written as WOFF unpacks to exactly what the truetype writer writes and reads back unchanged (tables, glyphs, advances, outlines, every character, shaping); the same face, the same bytes; each table compressed only when smaller, 4-byte aligned and zero-padded, the directory by tag, the data in the font's order; kept in test-output/fonts/woff", WoffEveryFont),
        ("fonts woff: woff passes the font writer contract (FontWriterSuite, read back through FontFormats) on TrueType, CFF and mapped faces; FontWriters.Save and FontFormats.Read find it by extension and first bytes, ReadFolder reads .woff beside .ttf; Encode packs a TrueType and a CFF file as they are; the compression level (IDRAK_WOFF_COMPRESSION; 0 stores every table) and the limit (IDRAK_WOFF_MAX_BYTES) are settings, bad values refused naming the variable", WoffContract),
        ("fonts woff: the plug-in registers from outside the library (Idrak.Imaging.Plugins, never a library default) into FontFormats and FontWriters, once, and unregisters leaving them as found (a WOFF then refused naming the formats); the truetype writer and opentype reader it builds on are reached through the registries: an app's over the library's is used, its failure reaches the caller by default, falls back per call under FallBack, is compared under Shadow; IdrakPlugins.Load (-P) registers the package", WoffRegistration),
        ("fonts woff: damaged and hostile WOFF files fail fast with an InvalidDataException naming the file and the cause: cut short, a table past the end, longer or shorter than declared, a damaged zlib stream, a decompression bomb past the limit refused before inflating, one lying about its length stopped at what it declares, a length zlib cannot give, unsorted, repeated, overlapping or misaligned tables, a wrong font size, extra bytes, no tables, a WOFF inside a WOFF", WoffHostile),
        ("fonts woff: idrak data fonts reads .woff files (named or in a folder) and data synth's fonts too; their help lists .woff while the tool's imaging package is registered, and not when IDRAK_PLUGIN_PACKAGES leaves it out", WoffCommands),
    ];

    // The D8 WOFF samples: each one's stem and its source under the test fonts.
    private static readonly (string Stem, string Source)[] WoffSampleFonts =
        [("dejavu-sans-subset", "dejavu-sans-subset.ttf"), ("dejavu-sans-cff", "dejavu-sans-cff.otf"), ("features", "features.ttf"), ("amiri", "arabic/amiri.ttf")];

    // The plug-in registered for the body, as an app registers it, and left as it was found.
    private static void WithWoff(Action body)
    {
        bool registered = WoffPlugin.IsRegistered;
        WoffPlugin.Register();
        try
        {
            body();
        }
        finally
        {
            if (!registered)
            {
                WoffPlugin.Unregister();
            }
        }
    }

    // A WOFF file kept in test-output/fonts/woff for tools/documents/check_font.py (where there is a checkout).
    private static void SaveWoff(string name, byte[] bytes)
    {
        if (TestPaths.Root is not { } root)
        {
            return;
        }

        string folder = Path.Combine(root, "test-output", "fonts", "woff");
        Directory.CreateDirectory(folder);
        File.WriteAllBytes(Path.Combine(folder, name.Replace('/', '-').Replace(" #", "-", StringComparison.Ordinal)), bytes);
    }

    // An sfnt's table directory: tag, checksum, offset, length.
    private static List<(string Tag, uint Checksum, int Offset, int Length)> SfntDirectory(ReadOnlySpan<byte> sfnt)
    {
        int count = BinaryPrimitives.ReadUInt16BigEndian(sfnt[4..]);
        var tables = new List<(string, uint, int, int)>(count);
        for (int i = 0; i < count; i++)
        {
            var e = sfnt.Slice(12 + 16 * i, 16);
            tables.Add((Encoding.ASCII.GetString(e[..4]), BinaryPrimitives.ReadUInt32BigEndian(e[4..]), (int)BinaryPrimitives.ReadUInt32BigEndian(e[8..]),
                (int)BinaryPrimitives.ReadUInt32BigEndian(e[12..])));
        }

        return tables;
    }

    // A WOFF's table directory: tag, offset, stored length, length, checksum.
    private static List<(string Tag, int Offset, int Stored, int Length, uint Checksum)> WoffDirectory(ReadOnlySpan<byte> woff)
    {
        int count = BinaryPrimitives.ReadUInt16BigEndian(woff[12..]);
        var tables = new List<(string, int, int, int, uint)>(count);
        for (int i = 0; i < count; i++)
        {
            var e = woff.Slice(44 + 20 * i, 20);
            tables.Add((Encoding.ASCII.GetString(e[..4]), (int)BinaryPrimitives.ReadUInt32BigEndian(e[4..]), (int)BinaryPrimitives.ReadUInt32BigEndian(e[8..]),
                (int)BinaryPrimitives.ReadUInt32BigEndian(e[12..]), BinaryPrimitives.ReadUInt32BigEndian(e[16..])));
        }

        return tables;
    }

    private static byte[] Inflated(ReadOnlySpan<byte> stored)
    {
        using var zlib = new ZLibStream(new MemoryStream(stored.ToArray()), CompressionMode.Decompress);
        using var all = new MemoryStream();
        zlib.CopyTo(all);
        return all.ToArray();
    }

    private static int Deflated(ReadOnlySpan<byte> table, int level)
    {
        using var all = new MemoryStream();
        using (var zlib = new ZLibStream(all, new ZLibCompressionOptions { CompressionLevel = level }, leaveOpen: true))
        {
            zlib.Write(table);
        }

        return (int)all.Length;
    }

    // The WOFF file's layout against the sfnt it packs (WOFF 1.0): the header, the directory by tag with each table's
    // length and checksum, every table on a 4-byte boundary in the sfnt's order and zero-padded, inflating to the table, and
    // compressed only when that is smaller (at the writer's level). How many tables were compressed and stored.
    private static (int Compressed, int Stored) WoffStructure(string name, byte[] woff, byte[] sfnt, int level = 9)
    {
        var tables = SfntDirectory(sfnt);
        var entries = WoffDirectory(woff);
        var head = tables.Single(t => t.Tag == "head");
        Check(BinaryPrimitives.ReadUInt32BigEndian(woff) == 0x774F4646 && BinaryPrimitives.ReadUInt32BigEndian(woff.AsSpan(4)) == BinaryPrimitives.ReadUInt32BigEndian(sfnt)
              && BinaryPrimitives.ReadUInt32BigEndian(woff.AsSpan(8)) == woff.Length && entries.Count == tables.Count && BinaryPrimitives.ReadUInt16BigEndian(woff.AsSpan(14)) == 0
              && BinaryPrimitives.ReadUInt32BigEndian(woff.AsSpan(16)) == sfnt.Length
              && BinaryPrimitives.ReadUInt32BigEndian(woff.AsSpan(20)) == BinaryPrimitives.ReadUInt32BigEndian(sfnt.AsSpan(head.Offset + 4))
              && woff.AsSpan(24, 20).IndexOfAnyExcept((byte)0) < 0,
            $"{name}: the WOFF header (signature, flavor, length, tables, reserved, the font's size, head's revision as the version, no metadata or private data)");
        Check(entries.Select(e => e.Tag).SequenceEqual(tables.Select(t => t.Tag).Order(StringComparer.Ordinal)), $"{name}: the WOFF directory is not the font's tables by tag");
        int compressed = 0, stored = 0;
        foreach (var entry in entries)
        {
            var table = tables.Single(t => t.Tag == entry.Tag);
            var data = sfnt.AsSpan(table.Offset, table.Length);
            Check(entry.Offset % 4 == 0 && entry.Length == table.Length && entry.Checksum == table.Checksum && entry.Stored <= entry.Length,
                $"{name}: table {entry.Tag} at {entry.Offset}, {entry.Stored} bytes stored for {entry.Length} (the font's {table.Length}), checksum {entry.Checksum:X8} (the font's {table.Checksum:X8})");
            var bytes = woff.AsSpan(entry.Offset, entry.Stored);
            if (entry.Stored < entry.Length)
            {
                Check(Inflated(bytes).AsSpan().SequenceEqual(data), $"{name}: table {entry.Tag} does not inflate to the font's");
                compressed++;
            }
            else
            {
                Check(bytes.SequenceEqual(data) && (level == 0 || data.Length == 0 || Deflated(data, level) >= data.Length),
                    $"{name}: table {entry.Tag} stored as it is, though compressing it is smaller, or stored otherwise");
                stored++;
            }
        }

        // The data in the font's own order, each block where the last ended (rounded up to 4 bytes, zeros between).
        var laid = entries.OrderBy(e => e.Offset).ToList();
        Check(laid.Select(e => e.Tag).SequenceEqual(tables.OrderBy(t => t.Offset).Select(t => t.Tag)), $"{name}: the WOFF's tables are not in the font's order");
        int at = 44 + 20 * entries.Count;
        foreach (var entry in laid)
        {
            Check(entry.Offset == at, $"{name}: table {entry.Tag} at {entry.Offset}, the last block ended at {at}");
            int end = entry.Offset + entry.Stored;
            at = (end + 3) & ~3;
            Check(woff.AsSpan(end, at - end).IndexOfAnyExcept((byte)0) < 0, $"{name}: table {entry.Tag}'s padding is not zeros");
        }

        Check(at == woff.Length, $"{name}: the last table ends at {at}, the file at {woff.Length}");
        return (compressed, stored);
    }

    private static void WoffSamples(Device device)
    {
        CpuOnly(device);
        string samples = FontWriteSamples();
        WithWoff(() =>
        {
            int glyphs = 0, characters = 0, lines = 0, harfBuzz = 0;
            foreach (var (stem, file) in WoffSampleFonts)
            {
                string path = Path.Combine(samples, stem + ".woff");
                byte[] bytes = File.ReadAllBytes(path), source = File.ReadAllBytes(FontData(file));
                var info = JsonNode.Parse(File.ReadAllText(path + ".json"))!;
                Check((string)info["sha256"]! == Convert.ToHexStringLower(SHA256.HashData(bytes)), $"{stem}.woff: not the sample its JSON describes: make the samples again");
                Check((string)JsonNode.Parse(File.ReadAllText(Path.Combine(samples, $"{stem}.source.json")))!["sha256"]! == Convert.ToHexStringLower(SHA256.HashData(source)),
                    $"{stem}: the samples were made from another {file}: make them again (tools/documents/make_fixtures.py)");
                var dump = JsonNode.Parse(File.ReadAllText(path + ".dump.json"))!;

                var faces = FontFormats.Read(path);
                Check(faces.Count == 1 && FontFormats.Read(bytes, stem).Count == 1, $"{stem}.woff: {faces.Count} faces read by its extension and first bytes");
                var face = faces[0];
                var from = FontFormats.Read(source, stem).Single();
                Check(face.Family == from.Family && face.Style == from.Style && face.Format == from.Format && face.Metrics == from.Metrics
                      && face.Metrics.UnitsPerEm == (int)dump["unitsPerEm"]! && face.GlyphCount == from.GlyphCount && face.GlyphCount == (int)dump["glyphCount"]!,
                    $"{stem}.woff: {face.Family} {face.Style} ({face.Format}), {face.GlyphCount} glyphs, {face.Metrics}; the source {from.Family} {from.Style} ({from.Format}), "
                    + $"{from.GlyphCount}, {from.Metrics}; the dump {dump["glyphCount"]} glyphs");

                // Every table the dump lists, at its length, byte for byte the source's (fontTools kept them all).
                foreach (var table in dump["tables"]!.AsArray())
                {
                    string tag = (string)table!["tag"]!;
                    Check(face.Table(tag).Length == (int)table["length"]! && face.Table(tag).Span.SequenceEqual(from.Table(tag).Span),
                        $"{stem}.woff: table {tag} is {face.Table(tag).Length} bytes, the dump's {table["length"]}, or not the source's");
                }

                var advances = dump["hmtx"]!.AsArray();
                for (int g = 0; g < face.GlyphCount; g++)
                {
                    GlyphOutline a = face.Outline(g), b = from.Outline(g);
                    Check(face.Advance(g) == (int)advances[g]![0]! && face.Advance(g) == from.Advance(g)
                          && a.Verbs.Span.SequenceEqual(b.Verbs.Span) && a.Points.Span.SequenceEqual(b.Points.Span),
                        $"{stem}.woff: glyph {g}: advance {face.Advance(g)} (the dump's {advances[g]![0]}, the source's {from.Advance(g)}), "
                        + $"{a.Verbs.Length} steps and {a.Points.Length / 2} points, the source's {b.Verbs.Length} and {b.Points.Length / 2}");
                }

                foreach (var (key, glyph) in dump["cmap"]!["best"]!.AsObject())
                {
                    int code = int.Parse(key, System.Globalization.CultureInfo.InvariantCulture);
                    Check(face.Glyph(code) == (int)glyph!, $"{stem}.woff: U+{code:X4} reads as {face.Glyph(code)}, the dump's {glyph}");
                }

                for (int c = 0; c <= 0x10FFFF; c++)
                {
                    if (face.Glyph(c) != from.Glyph(c))
                    {
                        Check(false, $"{stem}.woff: U+{c:X4} reads as glyph {face.Glyph(c)}, the source's {from.Glyph(c)}");
                    }
                }

                // The sample lines: as shaped from the source; and as HarfBuzz shapes them wherever it shapes the source the same.
                var shaped = JsonNode.Parse(File.ReadAllText(path + ".shaping.json"))!["lines"]!.AsArray();
                var sourceShaped = JsonNode.Parse(File.ReadAllText(Path.Combine(samples, $"{stem}.source.shaping.json")))!["lines"]!.AsArray();
                Check(shaped.Count == sourceShaped.Count && shaped.Count > 0, $"{stem}: {shaped.Count} shaped lines, the source {sourceShaped.Count}");
                for (int l = 0; l < shaped.Count; l++)
                {
                    string text = (string)shaped[l]!["text"]!;
                    static ShapedGlyph[] HarfBuzz(JsonNode line) => [.. line["glyphs"]!.AsArray().Select(g => new ShapedGlyph((int)g!["gid"]!, (int)g["cluster"]!, (int)g["xAdvance"]!,
                        (int)g["yAdvance"]!, (int)g["xOffset"]!, (int)g["yOffset"]!))];
                    var reference = HarfBuzz(shaped[l]!);
                    Check((string)sourceShaped[l]!["text"]! == text && reference.SequenceEqual(HarfBuzz(sourceShaped[l]!)),
                        $"{stem}: HarfBuzz shapes \"{text}\" otherwise from the WOFF sample and its source: make the samples again");
                    var run = new TextRun(0, text.Length, (string)shaped[l]!["script"]!, (string)shaped[l]!["direction"]! == "rtl");
                    var ours = TextShapers.Shape(face, text, run);
                    var fromSource = TextShapers.Shape(from, text, run);
                    Check(ours.SequenceEqual(fromSource), $"{stem}.woff: \"{text}\" shapes otherwise than from the source: {Comparisons.Sequences(fromSource, ours)}");
                    if (fromSource.SequenceEqual(reference))
                    {
                        Check(ours.SequenceEqual(reference), $"{stem}.woff: \"{text}\" is not shaped as HarfBuzz shapes it: {Comparisons.Sequences(reference, ours)}");
                        harfBuzz++;
                    }

                    lines++;
                }

                glyphs += face.GlyphCount;
                characters += dump["cmap"]!["best"]!.AsObject().Count;
            }

            Console.WriteLine($"    {WoffSampleFonts.Length} WOFF samples read as their sources: {glyphs} glyphs the same outlines and advances, {characters} characters, "
                              + $"{lines} lines shaped the same ({harfBuzz} of them as HarfBuzz shapes them, glyph for glyph)");
        });
    }

    private static void WoffEveryFont(Device device)
    {
        CpuOnly(device);
        WithWoff(() =>
        {
            var woff = FontWriters.Get(WoffWriter.WriterName);
            var truetype = FontWriters.Get(TrueTypeWriter.WriterName);
            var format = new WoffFormat();
            int written = 0, compressed = 0, stored = 0, lines = 0, glyphs = 0;
            long packed = 0, unpacked = 0;
            var watch = Stopwatch.StartNew();
            foreach (var (name, face) in FontWriteFaces())
            {
                byte[] bytes = WriteFont(woff, face, new FontWriteOptions());
                Check(bytes.AsSpan().SequenceEqual(WriteFont(woff, face, new FontWriteOptions())), $"{name}: the same face wrote other bytes");
                byte[] sfnt = WriteFont(truetype, face, new FontWriteOptions());
                byte[] decoded = format.Decode(bytes, name);
                Check(decoded.AsSpan().SequenceEqual(sfnt), $"{name}: the WOFF unpacks to {decoded.Length} bytes, not the {sfnt.Length} the truetype writer writes");
                var (c, s) = WoffStructure(name, bytes, sfnt);
                compressed += c;
                stored += s;
                SaveWoff($"{name}.woff", bytes);

                var read = FontFormats.Read(bytes, "written").Single();
                var plain = new OpenTypeFormat().Read(sfnt, "written")[0];
                Check(read.Metrics == face.Metrics && read.GlyphCount == face.GlyphCount && read.Family == face.Family && read.Style == face.Style && FontMaps.MappedBy(read) is null,
                    $"{name}: read back {read.Family} {read.Style}, {read.GlyphCount} glyphs, {read.Metrics}; the face {face.Family} {face.Style}, {face.GlyphCount}, {face.Metrics}");
                foreach (var (tag, _, _, _) in SfntDirectory(sfnt))
                {
                    Check(read.Table(tag).Span.SequenceEqual(plain.Table(tag).Span), $"{name}: table {tag} reads back otherwise from the WOFF than from the TrueType file");
                }

                for (int g = 0; g < face.GlyphCount; g++)
                {
                    GlyphOutline a = plain.Outline(g), b = read.Outline(g);
                    Check(read.Advance(g) == face.Advance(g) && a.Verbs.Span.SequenceEqual(b.Verbs.Span) && a.Points.Span.SequenceEqual(b.Points.Span),
                        $"{name}: glyph {g} reads back from the WOFF with advance {read.Advance(g)} (the face's {face.Advance(g)}), {b.Verbs.Length} steps and {b.Points.Length / 2} points, "
                        + $"from the TrueType file {a.Verbs.Length} and {a.Points.Length / 2}");
                }

                for (int code = 0; code <= 0x10FFFF; code++)
                {
                    if (face.Glyph(code) != read.Glyph(code))
                    {
                        Check(false, $"{name}: U+{code:X4} reads back as glyph {read.Glyph(code)}, the face's {face.Glyph(code)}");
                    }
                }

                foreach (var (text, script, rightToLeft) in FontWriteLines)
                {
                    if (text.EnumerateRunes().Any(r => !Rune.IsWhiteSpace(r) && face.Glyph(r.Value) == 0))
                    {
                        continue;                                               // a line of characters the font lacks
                    }

                    var run = new TextRun(0, text.Length, script, rightToLeft);
                    foreach (var features in (string[]?[])[null, [.. HandTags.Select(t => "+" + t)]])
                    {
                        var expected = TextShapers.Shape(face, text, run, features);
                        var actual = TextShapers.Shape(read, text, run, features);
                        Check(expected.SequenceEqual(actual), $"{name}: \"{text}\" ({(features is null ? "default features" : "with the hand's")}) shapes otherwise: {Comparisons.Sequences(expected, actual)}");
                    }

                    lines++;
                }

                written++;
                glyphs += face.GlyphCount;
                packed += bytes.Length;
                unpacked += sfnt.Length;
            }

            Check(written >= 10 && compressed > stored, $"{written} faces written, {compressed} tables compressed and {stored} stored");
            Console.WriteLine($"    {written} faces ({glyphs} glyphs) written as WOFF and read back unchanged: {packed:N0} bytes for {unpacked:N0} of TrueType "
                              + $"({100.0 * packed / unpacked:F1}%), {compressed} tables compressed, {stored} stored as they are; {lines} lines shaped the same; {watch.Elapsed.TotalSeconds:F1} s");
        });
    }

    private static void WoffContract(Device device)
    {
        CpuOnly(device);
        WithWoff(() =>
        {
            var writer = FontWriters.Get(WoffWriter.WriterName);
            var subset = SubsetFace();
            var features = FontFormats.Read(FontData("features.ttf")).Single();
            var amiri = FontFormats.Read(FontData("arabic/amiri.ttf")).Single();
            var cff = FontFormats.Read(FontData("dejavu-sans-cff.otf")).Single();
            var mapped = FontMaps.Apply(RawFace("MCS-HAND"));
            Conformance.CheckFontWriter(writer, [subset, features, amiri]).ThrowIfFailed();
            Conformance.CheckFontWriter(writer, [cff], exact: false).ThrowIfFailed();
            Conformance.CheckFontWriter(writer, [mapped]).ThrowIfFailed();

            // Saved and read by its extension and first bytes; a folder's .woff read beside its .ttf.
            string folder = Path.Combine(Path.GetTempPath(), $"idrak-font-woff-{Guid.NewGuid():N}");
            try
            {
                string path = Path.Combine(folder, "sub", "features.woff");
                FontWriters.Save(features, path);
                byte[] saved = File.ReadAllBytes(path);
                Check(saved.AsSpan().SequenceEqual(WriteFont(writer, features, new FontWriteOptions())) && FontFormats.Read(path).Single().GlyphCount == features.GlyphCount,
                    "FontWriters.Save wrote other bytes, or FontFormats.Read did not read them back");
                File.Copy(FontData("dejavu-sans-subset.ttf"), Path.Combine(folder, "sub", "dejavu.ttf"));
                File.WriteAllText(Path.Combine(folder, "sub", "notes.txt"), "Not a font.");
                var all = FontFormats.ReadFolder(folder);
                Check(all.Count == 2 && all.Any(f => f.Family == features.Family) && all.Any(f => f.Family == subset.Family), $"ReadFolder: {string.Join(", ", all.Select(f => f.Family))}");
            }
            finally
            {
                if (Directory.Exists(folder))
                {
                    Directory.Delete(folder, recursive: true);
                }
            }

            // Encode packs a font file as it is: TrueType and CFF, every table kept byte for byte, the flavor its version.
            var format = new WoffFormat();
            foreach (string file in (string[])["features.ttf", "dejavu-sans-cff.otf"])
            {
                byte[] source = File.ReadAllBytes(FontData(file));
                using var packed = new MemoryStream();
                new WoffWriter().Encode(source, packed);
                byte[] woff = packed.ToArray(), sfnt = format.Decode(woff, file);
                Check(BinaryPrimitives.ReadUInt32BigEndian(woff.AsSpan(4)) == BinaryPrimitives.ReadUInt32BigEndian(source) && BinaryPrimitives.ReadUInt32BigEndian(sfnt) == BinaryPrimitives.ReadUInt32BigEndian(source),
                    $"{file}: the flavor is not the font's version");
                var tables = SfntDirectory(source);
                var unpacked = SfntDirectory(sfnt);
                Check(unpacked.Select(t => t.Tag).SequenceEqual(tables.Select(t => t.Tag).Order(StringComparer.Ordinal)), $"{file}: the unpacked font's tables by tag");
                foreach (var table in tables)
                {
                    var back = unpacked.Single(t => t.Tag == table.Tag);
                    Check(back.Checksum == table.Checksum && sfnt.AsSpan(back.Offset, back.Length).SequenceEqual(source.AsSpan(table.Offset, table.Length)),
                        $"{file}: table {table.Tag} unpacks otherwise");
                }

                var read = FontFormats.Read(woff, file).Single();
                var original = FontFormats.Read(source, file).Single();
                Check(read.GlyphCount == original.GlyphCount && read.Format == original.Format && read.Family == original.Family, $"{file}: read back otherwise as WOFF");
            }

            // The compression level: 0 stores every table, 1 and 9 compress, each unpacking to the same font.
            byte[] font = WriteFont(FontWriters.Get(TrueTypeWriter.WriterName), amiri, new FontWriteOptions());
            var sizes = new List<int>();
            foreach (int level in (int[])[0, 1, 9])
            {
                var levelled = new WoffWriter { CompressionLevel = level };
                using var packed = new MemoryStream();
                levelled.Write(amiri, packed, new FontWriteOptions());
                byte[] woff = packed.ToArray();
                Check(format.Decode(woff).AsSpan().SequenceEqual(font), $"level {level}: unpacks to another font");
                var (compressedTables, storedTables) = WoffStructure($"amiri at level {level}", woff, font, level);
                Check(level == 0 ? compressedTables == 0 && woff.Length == 44 + 20 * storedTables + font.Length - 12 - 16 * storedTables : compressedTables > 0,
                    $"level {level}: {compressedTables} tables compressed, {storedTables} stored, {woff.Length} bytes for {font.Length}");
                sizes.Add(woff.Length);
            }

            Check(sizes[0] > sizes[1] && sizes[0] > sizes[2], $"compressing makes the file smaller: {string.Join(", ", sizes)} bytes at levels 0, 1 and 9");

            // The settings: the variables read when a format or writer is made, bad values and options refused naming them.
            Check(new WoffWriter().CompressionLevel == 9 && new WoffWriter().SfntWriter == TrueTypeWriter.WriterName && new WoffFormat().MaxFontBytes is > 0 and <= int.MaxValue,
                $"the defaults: level {new WoffWriter().CompressionLevel}, limit {new WoffFormat().MaxFontBytes}");
            WithEnvironment(new Dictionary<string, string?> { [WoffWriter.CompressionLevelVariable] = "0", [WoffFormat.MaxFontBytesVariable] = "64k" }, () =>
            {
                Check(new WoffWriter().CompressionLevel == 0 && new WoffFormat().MaxFontBytes == 65536, "IDRAK_WOFF_COMPRESSION=0 and IDRAK_WOFF_MAX_BYTES=64k");
                ThrowsNaming<InvalidDataException>(() => new WoffFormat().Decode(File.ReadAllBytes(Path.Combine(FontWriteSamples(), "amiri.woff")), "amiri"), "a WOFF past the limit",
                    "amiri: ", "more than the limit of 65,536", "WoffFormat.MaxFontBytes", WoffFormat.MaxFontBytesVariable);
            });
            WithEnvironment(new Dictionary<string, string?> { [WoffWriter.CompressionLevelVariable] = "12" }, () =>
                ThrowsNaming<ArgumentException>(() => _ = new WoffWriter(), "a level past 9", "IDRAK_WOFF_COMPRESSION=12", "0 to 9"));
            WithEnvironment(new Dictionary<string, string?> { [WoffFormat.MaxFontBytesVariable] = "lots" }, () =>
                ThrowsNaming<ArgumentException>(() => _ = new WoffFormat(), "a limit that is no number", "IDRAK_WOFF_MAX_BYTES=lots", "256m"));
            ThrowsNaming<ArgumentOutOfRangeException>(() => _ = new WoffWriter { CompressionLevel = -1 }, "level -1", "0 to 9", WoffWriter.CompressionLevelVariable);
            ThrowsNaming<ArgumentOutOfRangeException>(() => _ = new WoffFormat { MaxFontBytes = 0 }, "a limit of 0", "more than 0", WoffFormat.MaxFontBytesVariable);
            ThrowsNaming<ArgumentException>(() => _ = new WoffWriter { SfntWriter = "WOFF" }, "woff as its own sfnt writer", "its own sfnt writer");
            ThrowsNaming<NotRegisteredException>(() => new WoffWriter { SfntWriter = "fonts-woff-test-none" }.Write(features, Stream.Null, new FontWriteOptions()), "an sfnt writer nobody registered",
                "'fonts-woff-test-none'", TrueTypeWriter.WriterName);
            ThrowsNaming<InvalidDataException>(() => new WoffWriter().Encode(File.ReadAllBytes(Path.Combine(FontWriteSamples(), "amiri.woff")), Stream.Null), "a WOFF packed again",
                "one sfnt font", "wOFF");
        });
    }

    // An app's font writers and format over the library's, for the registration proof.
    private sealed class CountingFontWriter : IFontWriter
    {
        public int Calls;

        public string Name => TrueTypeWriter.WriterName;

        public IReadOnlyCollection<string> Extensions { get; } = [".ttf"];

        public void Write(IFontFace face, Stream destination, FontWriteOptions options)
        {
            Interlocked.Increment(ref Calls);
            FontWriters.Default(TrueTypeWriter.WriterName)!.Write(face, destination, options);
        }
    }

    private sealed class FailingFontWriter : IFontWriter
    {
        public string Name => TrueTypeWriter.WriterName;

        public IReadOnlyCollection<string> Extensions { get; } = [".ttf"];

        public void Write(IFontFace face, Stream destination, FontWriteOptions options) => throw new InvalidOperationException("fonts woff test: the app's truetype writer failed");
    }

    // Writes other bytes: the library's font with a description of its own.
    private sealed class DifferingFontWriter : IFontWriter
    {
        public string Name => TrueTypeWriter.WriterName;

        public IReadOnlyCollection<string> Extensions { get; } = [".ttf"];

        public void Write(IFontFace face, Stream destination, FontWriteOptions options) =>
            FontWriters.Default(TrueTypeWriter.WriterName)!.Write(face, destination, options with { Names = new Dictionary<FontNameKind, string> { [FontNameKind.Description] = "A differing writer" } });
    }

    private sealed class FailingFontFormat : IFontFormat
    {
        public string Name => "opentype";

        public IReadOnlyCollection<string> Extensions { get; } = [".ttf", ".otf", ".ttc"];

        public bool Accepts(ReadOnlySpan<byte> header) => FontFormats.Default("opentype")!.Accepts(header);

        public IReadOnlyList<IFontFace> Read(ReadOnlyMemory<byte> file, string name) => throw new InvalidOperationException("fonts woff test: the app's opentype reader failed");
    }

    private static SlotOverride? WoffRow(string registry, string name) => Overrides.Report().FirstOrDefault(o => o.Registry == registry && o.Name == name);

    private static void WoffRegistration(Device device)
    {
        CpuOnly(device);
        bool registered = WoffPlugin.IsRegistered;
        string package = typeof(WoffPlugin).Assembly.GetName().Name!, app = typeof(DesktopTests).Assembly.GetName().Name!;
        var features = FontFormats.Read(FontData("features.ttf")).Single();
        byte[] Woff() => WriteFont(FontWriters.Get(WoffWriter.WriterName), features, new FontWriteOptions());
        try
        {
            // Unregistered: no woff anywhere, a .woff path refused naming what there is.
            ImagingPlugins.UnregisterAll();
            var formats = FontFormats.Names.ToList();
            var writers = FontWriters.Names.ToList();
            Check(!WoffPlugin.IsRegistered && IdrakPlugins.Find(WoffFormat.FormatName) is null && !formats.Contains(WoffFormat.FormatName) && !writers.Contains(WoffWriter.WriterName)
                  && !FontFormats.Extensions.Contains(".woff") && !FontWriters.Extensions.Contains(".woff"),
                $"unregistered: formats {string.Join(", ", formats)}, writers {string.Join(", ", writers)}");
            ThrowsNaming<NotRegisteredException>(() => FontWriters.Get(WoffWriter.WriterName), "the woff writer, unregistered", "'woff'", "FontWriters.Register", TrueTypeWriter.WriterName);
            ThrowsNaming<NotSupportedException>(() => FontWriters.For("report.woff"), "a .woff path, unregistered", "'.woff'", ".ttf");

            // Registered as an app registers it: once however often, under the package's name, never a library default.
            WoffPlugin.Register();
            ImagingPlugins.RegisterAll();
            WoffPlugin.Register();
            Check(WoffPlugin.IsRegistered && ReferenceEquals(IdrakPlugins.Find(WoffFormat.FormatName), WoffPlugin.Plugin) && IdrakPlugins.Origin(WoffFormat.FormatName) == package,
                $"the plug-in recorded: {IdrakPlugins.Origin(WoffFormat.FormatName)}");
            Check(FontFormats.Names.Count(n => n == WoffFormat.FormatName) == 1 && FontFormats.Get("WOFF") is WoffFormat && FontFormats.Origin(WoffFormat.FormatName) == package
                  && FontFormats.Default(WoffFormat.FormatName) is null && FontFormats.Extensions.Contains(".woff") && FontFormats.Names.SequenceEqual([WoffFormat.FormatName, .. formats]),
                $"the format from outside, asked first: {string.Join(", ", FontFormats.Names)}, origin {FontFormats.Origin(WoffFormat.FormatName)}");
            Check(FontWriters.Names.Count(n => n == WoffWriter.WriterName) == 1 && FontWriters.Get("Woff") is WoffWriter && FontWriters.For("Report.WOFF") is WoffWriter
                  && FontWriters.Origin(WoffWriter.WriterName) == package && FontWriters.Default(WoffWriter.WriterName) is null,
                $"the writer from outside: {string.Join(", ", FontWriters.Names)}, origin {FontWriters.Origin(WoffWriter.WriterName)}");
            byte[] library = Woff();
            Check(FontFormats.Read(library, "features").Single().GlyphCount == features.GlyphCount, "a WOFF written and read back");

            // The truetype writer it packs, reached through FontWriters: the app's over the library's is used, fails by
            // default, falls back under FallBack, is compared under Shadow.
            var counting = new CountingFontWriter();
            FontWriters.Register(counting);
            try
            {
                Check(FontWriters.Origin(TrueTypeWriter.WriterName) == app && FontWriters.Default(TrueTypeWriter.WriterName) is TrueTypeWriter, "the app's truetype writer over the library's");
                Check(Woff().AsSpan().SequenceEqual(library) && counting.Calls == 1, $"the woff writer went through the app's truetype writer ({counting.Calls} calls)");
                FontWriters.Register(new FailingFontWriter());
                ThrowsNaming<InvalidOperationException>(() => Woff(), "the app's truetype writer failing (Throw)", "the app's truetype writer failed");
                FontWriters.SetPolicy(TrueTypeWriter.WriterName, SlotPolicy.FallBack);
                Check(Woff().AsSpan().SequenceEqual(library), "under FallBack the library's truetype writer writes the font the WOFF packs");
                Check(WoffRow(nameof(FontWriters), TrueTypeWriter.WriterName) is { Failures: >= 2, FallBacks: >= 1 }, $"the report: {WoffRow(nameof(FontWriters), TrueTypeWriter.WriterName)}");
                FontWriters.Register(new DifferingFontWriter());
                FontWriters.SetPolicy(TrueTypeWriter.WriterName, SlotPolicy.Shadow, shadowRate: 1);
                Check(Woff().AsSpan().SequenceEqual(library), "under Shadow the library's font is packed");
                Check(WoffRow(nameof(FontWriters), TrueTypeWriter.WriterName) is { Compared: >= 1, Differed: >= 1 }, $"the comparison: {WoffRow(nameof(FontWriters), TrueTypeWriter.WriterName)}");
            }
            finally
            {
                FontWriters.Unregister(TrueTypeWriter.WriterName);
                FontWriters.SetPolicy(TrueTypeWriter.WriterName, SlotPolicy.Throw);
            }

            Check(FontWriters.Origin(TrueTypeWriter.WriterName) == Overrides.Library && Woff().AsSpan().SequenceEqual(library), "the library's truetype writer is back");

            // The font inside handed to the opentype reader through FontFormats: the app's fails by default, falls back.
            FontFormats.Register(new FailingFontFormat());
            try
            {
                ThrowsNaming<InvalidOperationException>(() => FontFormats.Read(library, "features"), "the app's opentype reader failing (Throw)", "the app's opentype reader failed");
                FontFormats.SetPolicy("opentype", SlotPolicy.FallBack);
                Check(FontFormats.Read(library, "features").Single().GlyphCount == features.GlyphCount, "under FallBack the library's reader reads the font inside the WOFF");
                Check(WoffRow(nameof(FontFormats), "opentype") is { FallBacks: >= 1 }, $"the report: {WoffRow(nameof(FontFormats), "opentype")}");
            }
            finally
            {
                FontFormats.Unregister("opentype");
                FontFormats.SetPolicy("opentype", SlotPolicy.Throw);
            }

            Check(FontFormats.Origin("opentype") == Overrides.Library, "the library's opentype reader is back");

            // -P: a host that loads the package's assembly registers it through its attribute.
            ImagingPlugins.UnregisterAll();
            Check(IdrakPlugins.PackageOf(typeof(WoffPlugin).Assembly) is { Name: "Idrak.Imaging.Plugins" } declared && declared.All.SequenceEqual(ImagingPlugins.All)
                  && IdrakPlugins.Load(typeof(WoffPlugin).Assembly) && WoffPlugin.IsRegistered && FontFormats.Origin(WoffFormat.FormatName) == package,
                "IdrakPlugins.Load registers the package its assembly declares");

            // Unregistered again: the registries as found, a WOFF refused naming the formats there are.
            ImagingPlugins.UnregisterAll();
            Check(FontFormats.Names.SequenceEqual(formats) && FontWriters.Names.SequenceEqual(writers) && !WoffPlugin.IsRegistered,
                $"UnregisterAll leaves the registries as found: {string.Join(", ", FontFormats.Names)}; {string.Join(", ", FontWriters.Names)}");
            ThrowsNaming<InvalidDataException>(() => FontFormats.Read(library, "features"), "a WOFF with no woff format", "features: not a font of the registered formats", "opentype",
                "FontFormats.Register");
        }
        finally
        {
            if (registered)
            {
                WoffPlugin.Register();
            }
            else
            {
                WoffPlugin.Unregister();
            }
        }
    }

    private static void WoffHostile(Device device)
    {
        CpuOnly(device);
        WithWoff(() =>
        {
            var features = FontFormats.Read(FontData("features.ttf")).Single();
            byte[] good = WriteFont(FontWriters.Get(WoffWriter.WriterName), features, new FontWriteOptions());
            var format = new WoffFormat();
            var entries = WoffDirectory(good);
            int count = entries.Count, glyf = entries.FindIndex(e => e.Tag == "glyf");
            Check(entries[glyf].Stored < entries[glyf].Length && count >= 4, "the sample's glyf is compressed");
            var cases = new List<(string What, byte[] Bytes, string[] Says)>();

            byte[] Changed(Action<byte[]> change)
            {
                byte[] copy = [.. good];
                change(copy);
                return copy;
            }

            static string Number(long value) => value.ToString("N0", System.Globalization.CultureInfo.InvariantCulture);
            void U32(byte[] b, int at, long value) => BinaryPrimitives.WriteUInt32BigEndian(b.AsSpan(at), (uint)value);
            int Entry(int index) => 44 + 20 * index;

            // glyf declaring another length, the font's size in the header changed to match (so only the table is wrong).
            void Declare(byte[] b, long length)
            {
                U32(b, Entry(glyf) + 12, length);
                U32(b, 16, BinaryPrimitives.ReadUInt32BigEndian(good.AsSpan(16)) - ((entries[glyf].Length + 3) & ~3) + ((length + 3) & ~3));
            }

            byte[] cut = good[..(entries[glyf].Offset + 8)];
            U32(cut, 8, cut.Length);
            cases.Add(("cut in half", good[..(good.Length / 2)], ["the file is cut", $"gives {Number(good.Length)} bytes"]));
            cases.Add(("cut inside a table, the header's length changed to match", cut, ["runs past the end of the file", "the file is cut or damaged"]));
            cases.Add(("a table longer than it declares", Changed(b => Declare(b, entries[glyf].Length - 4)), ["the WOFF table 'glyf'", "inflates past the", "nothing more was inflated"]));
            cases.Add(("a table shorter than it declares", Changed(b => Declare(b, entries[glyf].Length + 4)), ["the WOFF table 'glyf'", "inflates to", "fewer than the"]));
            cases.Add(("a damaged zlib stream", Changed(b => b[entries[glyf].Offset] = 0), ["the WOFF table 'glyf' does not inflate"]));
            cases.Add(("a length zlib cannot give", Changed(b => Declare(b, 1033L * entries[glyf].Stored)), ["the WOFF table 'glyf' claims", "more than zlib can give", "1,032 to 1"]));
            cases.Add(("a table storing more than its length", Changed(b => Declare(b, entries[glyf].Stored - 4)), ["the WOFF table 'glyf' stores", "more than the table itself"]));
            cases.Add(("the directory out of order", Changed(b => { b.AsSpan(Entry(0), 20).CopyTo(b.AsSpan(Entry(1))); good.AsSpan(Entry(1), 20).CopyTo(b.AsSpan(Entry(0))); }),
                ["not in tag order"]));
            cases.Add(("a table listed twice", Changed(b => good.AsSpan(Entry(0), 4).CopyTo(b.AsSpan(Entry(1)))), ["is listed twice"]));
            cases.Add(("a table not on a 4-byte boundary", Changed(b => U32(b, Entry(glyf) + 4, entries[glyf].Offset + 2)), ["the WOFF table 'glyf'", "not on a 4-byte boundary"]));
            cases.Add(("a table inside the directory", Changed(b => U32(b, Entry(glyf) + 4, 44)), ["the WOFF table 'glyf'", "inside the header and directory"]));
            int small = entries.FindIndex(e => e.Tag == "head"), other = entries.FindIndex(e => e.Tag == "hhea");
            cases.Add(("two tables overlapping", Changed(b => U32(b, Entry(other) + 4, entries[small].Offset)), ["overlap", "a damaged file"]));
            cases.Add(("a wrong font size", Changed(b => U32(b, 16, BinaryPrimitives.ReadUInt32BigEndian(good.AsSpan(16)) + 4)), ["its WOFF header gives the font inside", "its tables make"]));
            cases.Add(("bytes after its end", [.. good, 0, 0, 0, 0], ["4 bytes follow the"]));
            cases.Add(("no tables", Changed(b => BinaryPrimitives.WriteUInt16BigEndian(b.AsSpan(12), 0)), ["a WOFF file of no tables"]));
            cases.Add(("the reserved field set", Changed(b => BinaryPrimitives.WriteUInt16BigEndian(b.AsSpan(14), 1)), ["reserved field is 1"]));
            cases.Add(("a WOFF inside a WOFF", Changed(b => U32(b, 4, 0x774F4646)), ["flavor is wOFF", "never a collection or another WOFF"]));
            cases.Add(("metadata outside the file", Changed(b => { U32(b, 24, good.Length - 4); U32(b, 28, 64); }), ["the WOFF's metadata", "lies outside the file's data"]));
            cases.Add(("too short for a header", good[..40], ["the file is cut", "fewer than a WOFF header's 44"]));

            var watch = Stopwatch.StartNew();
            foreach (var (what, bytes, says) in cases)
            {
                ThrowsNaming<InvalidDataException>(() => format.Decode(bytes, "hostile"), what, ["hostile: ", .. says]);
                if (bytes.Length >= 4 && format.Accepts(bytes))
                {
                    ThrowsNaming<InvalidDataException>(() => FontFormats.Read(bytes, "hostile"), $"{what}, through FontFormats", ["hostile: ", .. says]);
                }
            }

            double each = watch.Elapsed.TotalMilliseconds / cases.Count;

            // Decompression bombs: 16 MiB of zeros in one table.
            using var zeros = new MemoryStream();
            using (var zlib = new ZLibStream(zeros, CompressionLevel.SmallestSize, leaveOpen: true))
            {
                byte[] block = new byte[64 * 1024];
                for (int i = 0; i < 256; i++)
                {
                    zlib.Write(block);
                }
            }

            byte[] Bomb(long declared)
            {
                int stored = (int)zeros.Length, end = 64 + ((stored + 3) & ~3);
                var bomb = new byte[end];
                U32(bomb, 0, 0x774F4646);
                U32(bomb, 4, 0x00010000);
                U32(bomb, 8, end);
                BinaryPrimitives.WriteUInt16BigEndian(bomb.AsSpan(12), 1);
                U32(bomb, 16, 12 + 16 + ((declared + 3) & ~3));
                Encoding.ASCII.GetBytes("glyf").CopyTo(bomb, 44);
                U32(bomb, 48, 64);
                U32(bomb, 52, stored);
                U32(bomb, 56, declared);
                zeros.GetBuffer().AsSpan(0, stored).CopyTo(bomb.AsSpan(64));
                return bomb;
            }

            // An honest one past the limit: refused before inflating, nothing of its size allocated.
            var limited = new WoffFormat { MaxFontBytes = 1 << 20 };
            byte[] honest = Bomb(16 << 20);
            long before = GC.GetAllocatedBytesForCurrentThread();
            ThrowsNaming<InvalidDataException>(() => limited.Decode(honest, "bomb"), "an honest bomb past the limit",
                "bomb: ", "would take 16,777,244 bytes", "more than the limit of 1,048,576", "WoffFormat.MaxFontBytes", WoffFormat.MaxFontBytesVariable, "refused before anything was inflated");
            long honestBytes = GC.GetAllocatedBytesForCurrentThread() - before;

            // One that lies: it declares twice its stored bytes, inflating stops one byte past them.
            long lie = (2 * zeros.Length + 3) & ~3L;
            byte[] liar = Bomb(lie);
            before = GC.GetAllocatedBytesForCurrentThread();
            ThrowsNaming<InvalidDataException>(() => format.Decode(liar, "bomb"), "a bomb lying about its length", "bomb: ", $"the WOFF table 'glyf' inflates past the {Number(lie)} bytes it declares");
            long liarBytes = GC.GetAllocatedBytesForCurrentThread() - before;
            ThrowsNaming<InvalidDataException>(() => FontFormats.Read(liar, "bomb"), "a lying bomb through FontFormats", "bomb: ", $"inflates past the {Number(lie)} bytes");
            Check(honestBytes < 256 * 1024 && liarBytes < 1024 * 1024, $"a bomb costs {honestBytes:N0} bytes refused by the limit, {liarBytes:N0} stopped at its length");

            // Claiming a size zlib cannot give from its stored bytes: refused before the limit is even asked.
            ThrowsNaming<InvalidDataException>(() => format.Decode(Bomb(1L << 31), "bomb"), "a bomb claiming 2 GiB", "bomb: ", "more than zlib can give");
            Console.WriteLine($"    {cases.Count} damaged and hostile files refused, {each:F2} ms each; 16 MiB bombs: {honestBytes:N0} bytes allocated refusing one past the limit, "
                              + $"{liarBytes:N0} stopping one that lies ({lie:N0} declared, {zeros.Length:N0} stored)");
        });
    }

    private static void WoffCommands(Device device)
    {
        CpuOnly(device);
        bool registered = WoffPlugin.IsRegistered;
        string folder = Path.Combine(Path.GetTempPath(), $"idrak-font-woff-cli-{Guid.NewGuid():N}");
        try
        {
            WoffPlugin.Register();
            string fonts = Path.Combine(folder, "fonts");
            Directory.CreateDirectory(fonts);
            var amiri = FontFormats.Read(FontData("arabic/amiri.ttf")).Single();
            string woff = Path.Combine(fonts, "amiri.woff");
            FontWriters.Save(amiri, woff);
            string lines = Path.Combine(folder, "lines.txt");
            // The generic Arabic lines whose characters the repository's Amiri holds (it has no Arabic-Indic digits).
            var arabic = FontWriteLines.Where(l => l.Script == "Arab" && l.Text.EnumerateRunes().All(r => Rune.IsWhiteSpace(r) || amiri.Glyph(r.Value) != 0)).Select(l => l.Text).ToList();
            Check(arabic.Count >= 3, $"{arabic.Count} generic Arabic lines Amiri can set");
            File.WriteAllLines(lines, arabic, new UTF8Encoding(false));
            foreach (string target in (string[])[fonts, woff])
            {
                var (code, output, error) = SynthCli(["data", "fonts", target, "--lines", lines, "-j"]);
                var read = JsonNode.Parse(output)?["fonts"]?.AsArray();
                Check(code == 0 && read is { Count: 1 } && (string?)read[0]!["font"] == "amiri.woff" && (string?)read[0]!["family"] == amiri.Family
                      && read[0]!["missing"]!.AsArray().Count == 0 && read[0]!["lacking"]!.AsArray().Count == 0,
                    $"data fonts {target}: exit {code}: {output} {error}");
            }

            var synthFonts = Idrak.Text.SyntheticPages.Fonts(fonts);
            Check(synthFonts.Count == 1 && synthFonts[0].Family == amiri.Family, $"data synth's fonts: {synthFonts.Count}");
            foreach (string command in (string[])["fonts", "synth"])
            {
                var (code, help, _) = SynthCli(["data", command, "--help"]);
                Check(code == 0 && help.Contains(".ttf, .otf, .ttc, .woff", StringComparison.Ordinal), $"data {command} --help does not list .woff: {help}");
            }

            // Without the imaging package: the help lists the library's formats, and a folder of .woff holds no font.
            WoffPlugin.Unregister();
            WithEnvironment(new Dictionary<string, string?> { ["IDRAK_PLUGIN_PACKAGES"] = "text" }, () =>
            {
                foreach (string command in (string[])["fonts", "synth"])
                {
                    var (code, help, _) = SynthCli(["data", command, "--help"]);
                    Check(code == 0 && help.Contains("(.ttf, .otf, .ttc:", StringComparison.Ordinal) && !help.Contains(".woff", StringComparison.Ordinal),
                        $"data {command} --help lists .woff without the plug-in: {help}");
                }

                var (refused, _, why) = SynthCli(["data", "fonts", fonts, "--lines", lines]);
                Check(refused == 2 && why.Contains("No font (.ttf, .otf, .ttc) in", StringComparison.Ordinal), $"data fonts on .woff alone without the plug-in: exit {refused}: {why}");
            });
        }
        finally
        {
            if (registered)
            {
                WoffPlugin.Register();
            }
            else
            {
                WoffPlugin.Unregister();
            }

            if (Directory.Exists(folder))
            {
                Directory.Delete(folder, recursive: true);
            }
        }
    }
}
