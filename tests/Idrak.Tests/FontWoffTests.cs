// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Buffers.Binary;
using System.Diagnostics;
using System.IO.Compression;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json.Nodes;
using Idrak.Abstraction.Testing;
using Idrak.Imaging;
using Idrak.Imaging.Plugins;
using Idrak.Imaging.Plugins.Fonts;

// Plan 28, D8 (the plug-in part): WOFF 1.0 read and written by Idrak.Imaging.Plugins (WoffFormat, WoffWriter), against fontTools' WOFF samples
// (tools/documents/make_fixtures.py: their dumps and HarfBuzz's shaping) and the truetype writer it packs. The WOFF files
// written here are kept in test-output/fonts/woff for tools/documents/check_font.py (fontTools and the OpenType Sanitizer).
internal abstract partial class DesktopTests
{
    internal static readonly (string Name, Action<Device> Run)[] FontWoffGroup =
    [
        ("fonts woff: every D8 sample .woff (fontTools' WOFF 1.0 of the four test fonts) reads through WoffFormat: every table byte for byte its source's, the dump's glyph count, table lengths, advances and characters, the source face's outlines and characters exactly", WoffSamples),
        ("fonts woff: every test font (TrueType, CFF, a collection, variable fonts' default instance, the legacy fonts as read) written as WOFF unpacks to exactly what the truetype writer writes and reads back unchanged (tables, glyphs, advances, outlines, every character); the same face, the same bytes; each table compressed only when smaller, 4-byte aligned and zero-padded, the directory by tag, the data in the font's order; kept in test-output/fonts/woff", WoffEveryFont),
        ("fonts woff: damaged and hostile WOFF files fail fast with an InvalidDataException naming the file and the cause: cut short, a table past the end, longer or shorter than declared, a damaged zlib stream, a decompression bomb past the limit refused before inflating, one lying about its length stopped at what it declares, a length zlib cannot give, unsorted, repeated, overlapping or misaligned tables, a wrong font size, extra bytes, no tables, a WOFF inside a WOFF", WoffHostile),
    ];

    // The D8 WOFF samples: each one's stem and its source under the test fonts.
    private static readonly (string Stem, string Source)[] WoffSampleFonts =
        [("dejavu-sans-subset", "dejavu-sans-subset.ttf"), ("dejavu-sans-cff", "dejavu-sans-cff.otf"), ("features", "features.ttf"), ("amiri", "arabic/amiri.ttf")];

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
        int glyphs = 0, characters = 0;
        foreach (var (stem, file) in WoffSampleFonts)
        {
            string path = Path.Combine(samples, stem + ".woff");
            byte[] bytes = File.ReadAllBytes(path), source = File.ReadAllBytes(FontData(file));
            var info = JsonNode.Parse(File.ReadAllText(path + ".json"))!;
            Check((string)info["sha256"]! == Convert.ToHexStringLower(SHA256.HashData(bytes)), $"{stem}.woff: not the sample its JSON describes: make the samples again");
            Check((string)JsonNode.Parse(File.ReadAllText(Path.Combine(samples, $"{stem}.source.json")))!["sha256"]! == Convert.ToHexStringLower(SHA256.HashData(source)),
                $"{stem}: the samples were made from another {file}: make them again (tools/documents/make_fixtures.py)");
            var dump = JsonNode.Parse(File.ReadAllText(path + ".dump.json"))!;

            var faces = new WoffFormat().Read(bytes, stem);
            Check(faces.Count == 1, $"{stem}.woff: {faces.Count} faces");
            var face = faces[0];
            var from = new OpenTypeFormat().Read(source, stem).Single();
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

            glyphs += face.GlyphCount;
            characters += dump["cmap"]!["best"]!.AsObject().Count;
        }

        Console.WriteLine($"    {WoffSampleFonts.Length} WOFF samples read as their sources: {glyphs} glyphs the same outlines and advances, {characters} characters");
    }

    private static void WoffEveryFont(Device device)
    {
        CpuOnly(device);
        var woff = new WoffWriter();
        var truetype = new TrueTypeWriter();
        var format = new WoffFormat();
        int written = 0, compressed = 0, stored = 0, glyphs = 0;
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

            var read = format.Read(bytes, "written").Single();
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

            written++;
            glyphs += face.GlyphCount;
            packed += bytes.Length;
            unpacked += sfnt.Length;
        }

        Check(written >= 10 && compressed > stored, $"{written} faces written, {compressed} tables compressed and {stored} stored");
        Console.WriteLine($"    {written} faces ({glyphs} glyphs) written as WOFF and read back unchanged: {packed:N0} bytes for {unpacked:N0} of TrueType "
                          + $"({100.0 * packed / unpacked:F1}%), {compressed} tables compressed, {stored} stored as they are; {watch.Elapsed.TotalSeconds:F1} s");
    }

    private static void WoffHostile(Device device)
    {
        CpuOnly(device);
        var features = new OpenTypeFormat().Read(File.ReadAllBytes(FontData("features.ttf")), "features").Single();
        byte[] good = WriteFont(new WoffWriter(), features, new FontWriteOptions());
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
        Check(honestBytes < 256 * 1024 && liarBytes < 1024 * 1024, $"a bomb costs {honestBytes:N0} bytes refused by the limit, {liarBytes:N0} stopped at its length");

        // Claiming a size zlib cannot give from its stored bytes: refused before the limit is even asked.
        ThrowsNaming<InvalidDataException>(() => format.Decode(Bomb(1L << 31), "bomb"), "a bomb claiming 2 GiB", "bomb: ", "more than zlib can give");
        Console.WriteLine($"    {cases.Count} damaged and hostile files refused, {each:F2} ms each; 16 MiB bombs: {honestBytes:N0} bytes allocated refusing one past the limit, "
                          + $"{liarBytes:N0} stopping one that lies ({lie:N0} declared, {zeros.Length:N0} stored)");
    }
}
