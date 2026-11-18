// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Buffers.Binary;
using System.Text;

namespace Idrak.Imaging;

/// <summary>
/// "truetype": writes any face as a TrueType font (.ttf), the format Idrak works from (plan 28, decision 18). The tables:
/// head, hhea, maxp, hmtx; cmap (formats 4 and 12, from the face's characters); glyf and loca (the library's own
/// TrueType faces keep their glyph records, composites included, so their outlines read back the same; any other outline
/// is written as quadratic contours, a cubic curve turned into quadratics within <see cref="FontWriteOptions.CurveTolerance"/>
/// and every point rounded to whole font units); name (the face's own records, copyright and licence included, with the
/// strings <see cref="FontWriteOptions.Names"/> gives over them); OS/2 (the face's, its character and code page ranges,
/// average width and longest context measured from what is written); post (glyph names when the face has them); and the
/// face's layout tables (GDEF, GPOS, GSUB, a mapped face's made ones included) as they are. Hinting (cvt, fpgm, prep,
/// gasp and the glyphs' programs) is kept only for a whole face with <see cref="FontWriteOptions.KeepHinting"/>. The tables
/// are listed by tag and laid out in the order the OpenType specification recommends, each with its checksum, and head's
/// checkSumAdjustment set. A subset (<see cref="FontWriteOptions.Glyphs"/>) keeps glyph 0, the glyphs asked for, with
/// <see cref="FontWriteOptions.KeepLayout"/> every glyph GSUB can make from them (fontTools' closure), and the parts of
/// composite glyphs, each under its own id; the others are written empty with their advances, and without layout the
/// layout tables are left out. The same face and options write the same bytes.
/// </summary>
public sealed class TrueTypeWriter : IFontWriter
{
    /// <summary>The writer's name.</summary>
    public const string WriterName = "truetype";

    /// <inheritdoc/>
    public string Name => WriterName;

    /// <inheritdoc/>
    public IReadOnlyCollection<string> Extensions { get; } = [".ttf"];

    /// <inheritdoc/>
    /// <exception cref="NotSupportedException">The face has more than 65,535 glyphs, an advance outside 0 to 65,535, or a point TrueType's 16 bits cannot hold.</exception>
    /// <exception cref="ArgumentOutOfRangeException">A subset names a glyph the face does not have.</exception>
    public void Write(IFontFace face, Stream destination, FontWriteOptions options)
    {
        ArgumentNullException.ThrowIfNull(face);
        ArgumentNullException.ThrowIfNull(destination);
        ArgumentNullException.ThrowIfNull(options);
        destination.Write(new TrueTypeFont(face, options).Build());
    }

    /// <summary>The glyphs a write of <paramref name="face"/> with <paramref name="options"/> keeps (every glyph when it is written whole).</summary>
    internal static bool[] Kept(IFontFace face, FontWriteOptions options) => new TrueTypeFont(face, options).Kept;
}

/// <summary>One TrueType font being written: its glyphs, then its tables, then the file.</summary>
internal sealed class TrueTypeFont
{
    // The tables' data in the order the OpenType specification recommends for TrueType outlines; others follow by tag.
    private static readonly string[] DataOrder =
        ["head", "hhea", "maxp", "OS/2", "hmtx", "LTSH", "VDMX", "hdmx", "cmap", "fpgm", "prep", "cvt ", "loca", "glyf", "kern", "name", "post", "gasp", "PCLT", "DSIG"];

    private static readonly Dictionary<string, int> StandardNames = PostNames.Standard.Select((n, i) => (n, i)).ToDictionary(p => p.n, p => p.i, StringComparer.Ordinal);

    private readonly IFontFace _face;
    private readonly FontWriteOptions _options;
    private readonly int _count;
    private readonly string _name;
    private readonly OpenTypeFace? _records;
    private readonly int _recordCount;
    private readonly bool _hinting;
    private readonly GlyphRecords.Layout?[] _layouts;
    private readonly Dictionary<int, (int Points, int Contours, int Depth)> _flattened = [];
    private bool[]? _kept;

    public TrueTypeFont(IFontFace face, FontWriteOptions options)
    {
        _face = face;
        _options = options;
        _count = face.GlyphCount;
        _name = $"{face.Family} {face.Style}";
        if (_count is <= 0 or > 0xFFFF)
        {
            throw new NotSupportedException($"{_name}: {_count:N0} glyphs; a TrueType font holds 1 to 65,535.");
        }

        (_records, _recordCount) = Records(face);
        _hinting = options.KeepHinting && options.Glyphs is null && _records is not null;
        _layouts = new GlyphRecords.Layout?[_recordCount];
    }

    /// <summary>The glyphs written with their outlines.</summary>
    public bool[] Kept => _kept ??= Closure();

    // A glyph as written: its record (null: empty), contours (-1: composite), points (a simple glyph's), box, program size and
    // left side bearing.
    private readonly record struct Written(byte[]? Data, int Contours, int Points, int XMin, int YMin, int XMax, int YMax, int Instructions, int Left);

    // The library's own TrueType records behind a face, for glyph ids below the count: an OpenType face's, or the face a
    // map was applied to (its glyphs keep their ids).
    private static (OpenTypeFace? Face, int Count) Records(IFontFace face) => face switch
    {
        OpenTypeFace { HasGlyphRecords: true } open => (open, open.GlyphCount),
        MappedFontFace mapped when Records(mapped.Inner) is ({ } inner, int count) => (inner, Math.Min(count, mapped.GlyphCount)),
        _ => (null, 0),
    };

    // The code points to ask the face about: its character map's (and a map's) where the library made the face, else every one.
    private static IEnumerable<int> CharacterCodes(IFontFace face) => face switch
    {
        OpenTypeFace open => open.MappedCodes(),
        MappedFontFace mapped => CharacterCodes(mapped.Inner).Concat(mapped.MappedCodes).Distinct().Order(),
        _ => EveryCode(),
    };

    private static IEnumerable<int> EveryCode()
    {
        for (int c = 0; c <= 0x10FFFF; c++)
        {
            if (c is < 0xD800 or > 0xDFFF)
            {
                yield return c;
            }
        }
    }

    private ReadOnlySpan<byte> Record(int glyph) => _records!.GlyphRecord(glyph);

    private GlyphRecords.Layout Layout(int glyph) => _layouts[glyph] ??= GlyphRecords.Measure(Record(glyph), _name, glyph);

    // Glyph 0, the glyphs asked for, the GSUB closure (with layout), then the parts of composites, as fontTools' subsetter
    // orders them.
    private bool[] Closure()
    {
        var kept = new bool[_count];
        if (_options.Glyphs is null)
        {
            Array.Fill(kept, true);
            return kept;
        }

        var glyphs = new HashSet<int> { 0 };
        foreach (int g in _options.Glyphs)
        {
            if ((uint)g >= (uint)_count)
            {
                throw new ArgumentOutOfRangeException(nameof(FontWriteOptions.Glyphs), g, $"{_name} has glyphs 0 to {_count - 1}: a subset cannot keep glyph {g}.");
            }

            glyphs.Add(g);
        }

        if (_options.KeepLayout)
        {
            LayoutClosure.Close(_face.Table("GSUB"), glyphs, _count);
        }

        var pending = new Queue<int>(glyphs.Order());
        var parts = new List<int>();
        while (pending.TryDequeue(out int g))
        {
            if (g >= _recordCount || Record(g).IsEmpty || Layout(g).Contours >= 0)
            {
                continue;
            }

            parts.Clear();
            GlyphRecords.Components(Record(g), Layout(g), parts);
            foreach (int part in parts)
            {
                if ((uint)part < (uint)_count && glyphs.Add(part))
                {
                    pending.Enqueue(part);
                }
            }
        }

        foreach (int g in glyphs)
        {
            kept[g] = true;
        }

        return kept;
    }

    /// <summary>The font file.</summary>
    public byte[] Build()
    {
        var kept = Kept;
        var glyphs = new Written[_count];
        for (int g = 0; g < _count; g++)
        {
            glyphs[g] = kept[g] ? Glyph(g) : default;
        }

        var advances = new int[_count];
        for (int g = 0; g < _count; g++)
        {
            int advance = _face.Advance(g);
            advances[g] = advance is >= 0 and <= 0xFFFF ? advance
                : throw new NotSupportedException($"{_name}: glyph {g}'s advance {advance} lies outside the 0 to 65,535 TrueType holds.");
        }

        var characters = new List<(int Code, int Glyph)>();
        foreach (int code in CharacterCodes(_face))
        {
            int glyph = _face.Glyph(code);
            if (glyph > 0 && glyph < _count && kept[glyph])
            {
                characters.Add((code, glyph));
            }
        }

        var tables = new Dictionary<string, byte[]>(StringComparer.Ordinal);
        var (glyf, loca, longOffsets) = GlyphTables(glyphs);
        tables["glyf"] = glyf;
        tables["loca"] = loca;
        var box = Box(glyphs);
        tables["head"] = Head(box, longOffsets);
        tables["hhea"] = Hhea(glyphs, advances, out int metrics);
        tables["hmtx"] = Hmtx(glyphs, advances, metrics);
        tables["maxp"] = Maxp(glyphs);
        tables["cmap"] = Cmap(characters);
        tables["name"] = Name();
        tables["post"] = Post(characters);
        if (_options.KeepLayout)
        {
            foreach (string tag in (string[])["GDEF", "GPOS", "GSUB"])
            {
                if (_face.Table(tag) is { IsEmpty: false } table)
                {
                    tables[tag] = Static(tag, table.Span);
                }
            }
        }

        if (_hinting)
        {
            foreach (string tag in (string[])["cvt ", "fpgm", "prep", "gasp"])
            {
                if (_face.Table(tag) is { IsEmpty: false } table)
                {
                    tables[tag] = table.ToArray();
                }
            }
        }

        int context = LayoutClosure.MaxContext(tables.GetValueOrDefault("GSUB"), tables.GetValueOrDefault("GPOS"));
        tables["OS/2"] = Os2(characters, glyphs, advances, box, context);
        return Assemble(tables);
    }

    // A layout table as it is, but for what only a variable font's other tables give meaning to: the font is written as
    // its default instance (no fvar), so GDEF's item variation store and GSUB's and GPOS's feature variations are
    // unlinked (their offsets zeroed; the shapers use neither at the default instance).
    private static byte[] Static(string tag, ReadOnlySpan<byte> table)
    {
        var copy = table.ToArray();
        int major = copy.Length >= 4 ? BinaryPrimitives.ReadUInt16BigEndian(copy) : 0, minor = copy.Length >= 4 ? BinaryPrimitives.ReadUInt16BigEndian(copy.AsSpan(2)) : 0;
        if (major == 1 && tag == "GDEF" && minor >= 3 && copy.Length >= 18)
        {
            copy.AsSpan(14, 4).Clear();
        }
        else if (major == 1 && tag != "GDEF" && minor >= 1 && copy.Length >= 14)
        {
            copy.AsSpan(10, 4).Clear();
        }

        return copy;
    }

    private Written Glyph(int g)
    {
        if (g < _recordCount)
        {
            var record = Record(g);
            if (record.IsEmpty)
            {
                return default;
            }

            var layout = Layout(g);
            if (layout.Contours == 0)
            {
                return default;                                                 // a record of no contour: nothing to draw
            }

            var (xMin, yMin, xMax, yMax) = GlyphRecords.Box(record);
            return new Written(GlyphRecords.Copy(record, layout, _hinting), layout.Contours, layout.Points, xMin, yMin, xMax, yMax, _hinting ? layout.Instructions : 0,
                LeftBearing(g) ?? xMin);
        }

        var encoded = GlyphEncoder.Encode(_face.Outline(g), _options.CurveTolerance, _name, g);
        return encoded.Contours == 0 ? default
            : new Written(encoded.Data, encoded.Contours, encoded.Points, encoded.XMin, encoded.YMin, encoded.XMax, encoded.YMax, 0, encoded.XMin);
    }

    // A copied record's left side bearing as its font's hmtx gives it (where it is not the box's left, a font without head's
    // "lsb at x=0" flag places the outline by it, so it is kept as it was); null when the font gives none.
    private int? LeftBearing(int g)
    {
        var hhea = _records!.Table("hhea").Span;
        var hmtx = _records.Table("hmtx").Span;
        if (hhea.Length < 36)
        {
            return null;
        }

        int metrics = BinaryPrimitives.ReadUInt16BigEndian(hhea[34..]);
        int at = g < metrics ? 4 * g + 2 : 4 * metrics + 2 * (g - metrics);
        return metrics > 0 && at + 2 <= hmtx.Length ? BinaryPrimitives.ReadInt16BigEndian(hmtx[at..]) : null;
    }

    // glyf (each record padded to 4 bytes) and loca (short offsets while they fit).
    private static (byte[] Glyf, byte[] Loca, bool Long) GlyphTables(Written[] glyphs)
    {
        var glyf = new FontBuffer(glyphs.Sum(g => g.Data?.Length + 3 ?? 0));
        var offsets = new int[glyphs.Length + 1];
        for (int g = 0; g < glyphs.Length; g++)
        {
            offsets[g] = glyf.Length;
            if (glyphs[g].Data is { } data)
            {
                glyf.Bytes(data);
                glyf.Align(4);
            }
        }

        offsets[^1] = glyf.Length;
        bool longOffsets = glyf.Length > 0x1FFFE;
        var loca = new FontBuffer(offsets.Length * (longOffsets ? 4 : 2));
        foreach (int offset in offsets)
        {
            if (longOffsets)
            {
                loca.U32((uint)offset);
            }
            else
            {
                loca.U16(offset / 2);
            }
        }

        return (glyf.ToArray(), loca.ToArray(), longOffsets);
    }

    private static (int XMin, int YMin, int XMax, int YMax) Box(Written[] glyphs)
    {
        bool any = false;
        int xMin = 0, yMin = 0, xMax = 0, yMax = 0;
        foreach (var g in glyphs)
        {
            if (g.Data is null)
            {
                continue;
            }

            (xMin, yMin, xMax, yMax) = any ? (Math.Min(xMin, g.XMin), Math.Min(yMin, g.YMin), Math.Max(xMax, g.XMax), Math.Max(yMax, g.YMax)) : (g.XMin, g.YMin, g.XMax, g.YMax);
            any = true;
        }

        return (xMin, yMin, xMax, yMax);
    }

    private byte[] Head((int XMin, int YMin, int XMax, int YMax) box, bool longOffsets)
    {
        var source = _face.Table("head").Span;
        bool copy = source.Length >= 54;
        var w = new FontBuffer(54);
        w.U32(0x00010000);
        w.U32(copy ? BinaryPrimitives.ReadUInt32BigEndian(source[4..]) : 0x00010000);      // fontRevision
        w.U32(0);                                                                          // checkSumAdjustment, set last
        w.U32(0x5F0F3CF5);
        w.U16(copy ? BinaryPrimitives.ReadUInt16BigEndian(source[16..]) : 0x000B);          // flags
        w.U16(_face.Metrics.UnitsPerEm);
        if (copy)
        {
            w.Bytes(source[20..36]);                                                       // created and modified, as the face says
        }
        else
        {
            w.Zeros(16);
        }

        w.I16(box.XMin);
        w.I16(box.YMin);
        w.I16(box.XMax);
        w.I16(box.YMax);
        w.U16(copy ? BinaryPrimitives.ReadUInt16BigEndian(source[44..]) : MacStyle());
        w.U16(copy ? BinaryPrimitives.ReadUInt16BigEndian(source[46..]) : 8);               // lowestRecPPEM
        w.I16(copy ? BinaryPrimitives.ReadInt16BigEndian(source[48..]) : 2);                // fontDirectionHint
        w.I16(longOffsets ? 1 : 0);
        w.I16(0);
        return w.ToArray();
    }

    private int MacStyle() => (Bold ? 1 : 0) | (Italic ? 2 : 0);

    private bool Bold => _face.Style.Contains("Bold", StringComparison.OrdinalIgnoreCase);

    private bool Italic => _face.Style.Contains("Italic", StringComparison.OrdinalIgnoreCase) || _face.Style.Contains("Oblique", StringComparison.OrdinalIgnoreCase);

    private byte[] Hhea(Written[] glyphs, int[] advances, out int metrics)
    {
        metrics = advances.Length;
        while (metrics > 1 && advances[metrics - 1] == advances[metrics - 2])
        {
            metrics--;
        }

        int minLeft = 0, minRight = 0, extent = 0;
        bool any = false;
        for (int g = 0; g < glyphs.Length; g++)
        {
            if (glyphs[g].Data is null)
            {
                continue;
            }

            int width = glyphs[g].XMax - glyphs[g].XMin, left = glyphs[g].Left, right = advances[g] - left - width;
            (minLeft, minRight, extent) = any ? (Math.Min(minLeft, left), Math.Min(minRight, right), Math.Max(extent, left + width)) : (left, right, left + width);
            any = true;
        }

        var source = _face.Table("hhea").Span;
        var m = _face.Metrics;
        var w = new FontBuffer(36);
        w.U32(0x00010000);
        w.I16(m.Ascender);
        w.I16(m.Descender);
        w.I16(m.LineGap);
        w.U16(advances.Max());
        w.I16(minLeft);
        w.I16(minRight);
        w.I16(extent);
        if (source.Length >= 24)
        {
            w.Bytes(source[18..24]);                                                       // the caret's slope and offset
        }
        else
        {
            w.I16(1);
            w.I16(0);
            w.I16(0);
        }

        w.Zeros(10);                                                                       // reserved, metricDataFormat 0
        w.U16(metrics);
        return w.ToArray();
    }

    private static byte[] Hmtx(Written[] glyphs, int[] advances, int metrics)
    {
        var w = new FontBuffer(4 * metrics + 2 * (glyphs.Length - metrics));
        for (int g = 0; g < glyphs.Length; g++)
        {
            if (g < metrics)
            {
                w.U16(advances[g]);
            }

            w.I16(glyphs[g].Data is null ? 0 : glyphs[g].Left);
        }

        return w.ToArray();
    }

    // A composite's points and contours with its parts resolved, and how deep its parts nest (1: parts that are simple).
    private (int Points, int Contours, int Depth) Flattened(int g, int depth)
    {
        if (_flattened.TryGetValue(g, out var known))
        {
            return known;
        }

        if (depth > 16)
        {
            throw new InvalidDataException($"{_name}: glyph {g}'s parts nest deeper than 16.");
        }

        if (g >= _recordCount || Record(g).IsEmpty)
        {
            return (0, 0, 0);
        }

        var layout = Layout(g);
        if (layout.Contours >= 0)
        {
            return (layout.Points, layout.Contours, 0);
        }

        var parts = new List<int>();
        GlyphRecords.Components(Record(g), layout, parts);
        int points = 0, contours = 0, deepest = 0;
        foreach (int part in parts)
        {
            var (p, c, d) = Flattened(part, depth + 1);
            points += p;
            contours += c;
            deepest = Math.Max(deepest, d);
        }

        return _flattened[g] = (points, contours, deepest + 1);
    }

    private byte[] Maxp(Written[] glyphs)
    {
        int maxPoints = 0, maxContours = 0, maxCompositePoints = 0, maxCompositeContours = 0, maxElements = 0, maxDepth = 0, maxInstructions = 0;
        var parts = new List<int>();
        for (int g = 0; g < glyphs.Length; g++)
        {
            var glyph = glyphs[g];
            if (glyph.Data is null)
            {
                continue;
            }

            maxInstructions = Math.Max(maxInstructions, glyph.Instructions);
            if (glyph.Contours > 0)
            {
                maxPoints = Math.Max(maxPoints, glyph.Points);
                maxContours = Math.Max(maxContours, glyph.Contours);
                continue;
            }

            var (points, contours, depth) = Flattened(g, 0);
            parts.Clear();
            GlyphRecords.Components(Record(g), Layout(g), parts);
            maxCompositePoints = Math.Max(maxCompositePoints, points);
            maxCompositeContours = Math.Max(maxCompositeContours, contours);
            maxElements = Math.Max(maxElements, parts.Count);
            maxDepth = Math.Max(maxDepth, depth);
        }

        var source = _face.Table("maxp").Span;
        bool programs = _hinting && source.Length >= 32 && BinaryPrimitives.ReadUInt32BigEndian(source) == 0x00010000;
        var w = new FontBuffer(32);
        w.U32(0x00010000);
        w.U16(_count);
        w.U16(maxPoints);
        w.U16(maxContours);
        w.U16(maxCompositePoints);
        w.U16(maxCompositeContours);
        if (programs)
        {
            w.Bytes(source[14..26]);                                                       // zones, twilight points, storage, definitions, stack: the programs'
            w.U16(Math.Max(maxInstructions, BinaryPrimitives.ReadUInt16BigEndian(source[26..])));
        }
        else
        {
            w.U16(1);                                                                      // no programs: one zone, nothing else
            w.Zeros(10);
            w.U16(maxInstructions);
        }

        w.U16(maxElements);
        w.U16(maxDepth);
        return w.ToArray();
    }

    // cmap: format 4 for the basic plane (when it fits its 16-bit sizes) and format 12 for every character, each under its
    // Unicode and Windows encodings.
    private static byte[] Cmap(List<(int Code, int Glyph)> characters)
    {
        byte[]? format4 = Format4(characters);
        byte[] format12 = Format12(characters);
        var encodings = new List<(int Platform, int Encoding, byte[] Table)>();
        if (format4 is not null)
        {
            encodings.Add((0, 3, format4));
        }

        encodings.Add((0, 4, format12));
        if (format4 is not null)
        {
            encodings.Add((3, 1, format4));
        }

        encodings.Add((3, 10, format12));
        var w = new FontBuffer(4 + 8 * encodings.Count + (format4?.Length ?? 0) + format12.Length);
        w.U16(0);
        w.U16(encodings.Count);
        int at = 4 + 8 * encodings.Count;
        var placed = new Dictionary<byte[], int>(ReferenceEqualityComparer.Instance);
        var order = new List<byte[]>();
        foreach (var (platform, encoding, table) in encodings)
        {
            if (!placed.TryGetValue(table, out int offset))
            {
                placed[table] = offset = at;
                at += table.Length;
                order.Add(table);
            }

            w.U16(platform);
            w.U16(encoding);
            w.U32((uint)offset);
        }

        foreach (var table in order)
        {
            w.Bytes(table);
        }

        return w.ToArray();
    }

    // Format 4: segments of consecutive codes, each by a constant delta, or by a glyph array where that is smaller; null
    // when neither fits the format's 16-bit sizes.
    private static byte[]? Format4(List<(int Code, int Glyph)> characters)
    {
        var basic = characters.Where(c => c.Code < 0xFFFF).ToList();
        var runs = new List<(int Start, int End)>();                          // runs of consecutive codes, by index
        for (int i = 0; i < basic.Count;)
        {
            int j = i;
            while (j + 1 < basic.Count && basic[j + 1].Code == basic[j].Code + 1)
            {
                j++;
            }

            runs.Add((i, j));
            i = j + 1;
        }

        // Each run as delta segments (8 bytes each) or as one segment with its glyphs listed (8 bytes and 2 a code).
        var segments = new List<(int Start, int End, int Delta, int[]? Glyphs)>();
        foreach (var (first, last) in runs)
        {
            var deltas = new List<(int Start, int End, int Delta, int[]? Glyphs)>();
            for (int i = first; i <= last;)
            {
                int j = i, delta = basic[i].Glyph - basic[i].Code;
                while (j + 1 <= last && basic[j + 1].Glyph - basic[j + 1].Code == delta)
                {
                    j++;
                }

                deltas.Add((basic[i].Code, basic[j].Code, delta, null));
                i = j + 1;
            }

            if (8 * deltas.Count <= 8 + 2 * (last - first + 1))
            {
                segments.AddRange(deltas);
            }
            else
            {
                segments.Add((basic[first].Code, basic[last].Code, 0, [.. basic.Skip(first).Take(last - first + 1).Select(c => c.Glyph)]));
            }
        }

        segments.Add((0xFFFF, 0xFFFF, 1, null));
        int count = segments.Count, arrays = segments.Sum(s => s.Glyphs?.Length ?? 0);
        int length = 16 + 8 * count + 2 * arrays;
        if (count > 0x7FFF || length > 0xFFFF)
        {
            return null;
        }

        int power = 1, log = 0;
        while (power * 2 <= count)
        {
            power *= 2;
            log++;
        }

        var w = new FontBuffer(length);
        w.U16(4);
        w.U16(length);
        w.U16(0);
        w.U16(2 * count);
        w.U16(2 * power);
        w.U16(log);
        w.U16(2 * count - 2 * power);
        foreach (var s in segments)
        {
            w.U16(s.End);
        }

        w.U16(0);
        foreach (var s in segments)
        {
            w.U16(s.Start);
        }

        foreach (var s in segments)
        {
            w.U16(s.Delta & 0xFFFF);
        }

        int listed = 0;
        for (int i = 0; i < count; i++)
        {
            // From this idRangeOffset entry to the segment's first glyph in the array after the entries.
            w.U16(segments[i].Glyphs is null ? 0 : 2 * (count - i) + 2 * listed);
            listed += segments[i].Glyphs?.Length ?? 0;
        }

        foreach (var s in segments)
        {
            foreach (int glyph in s.Glyphs ?? [])
            {
                w.U16(glyph);
            }
        }

        return w.ToArray();
    }

    private static byte[] Format12(List<(int Code, int Glyph)> characters)
    {
        var groups = new List<(int Start, int End, int Glyph)>();
        foreach (var (code, glyph) in characters)
        {
            if (groups.Count > 0 && groups[^1].End == code - 1 && groups[^1].Glyph + (code - groups[^1].Start) == glyph)
            {
                groups[^1] = (groups[^1].Start, code, groups[^1].Glyph);
            }
            else
            {
                groups.Add((code, code, glyph));
            }
        }

        var w = new FontBuffer(16 + 12 * groups.Count);
        w.U16(12);
        w.U16(0);
        w.U32((uint)(16 + 12 * groups.Count));
        w.U32(0);
        w.U32((uint)groups.Count);
        foreach (var (start, end, glyph) in groups)
        {
            w.U32((uint)start);
            w.U32((uint)end);
            w.U32((uint)glyph);
        }

        return w.ToArray();
    }

    // name: the face's records (format 1's language-tag records left out), the options' strings over theirs; a face with
    // no name table gets its family and style.
    private byte[] Name()
    {
        var source = _face.Table("name").Span;
        var records = new List<(int Platform, int Encoding, int Language, int Id, byte[] Text)>();
        bool mac = false;
        if (source.Length >= 6)
        {
            int count = BinaryPrimitives.ReadUInt16BigEndian(source[2..]), storage = BinaryPrimitives.ReadUInt16BigEndian(source[4..]);
            for (int i = 0; i < count && 6 + 12 * (i + 1) <= source.Length; i++)
            {
                var r = source.Slice(6 + 12 * i, 12);
                int platform = BinaryPrimitives.ReadUInt16BigEndian(r), encoding = BinaryPrimitives.ReadUInt16BigEndian(r[2..]);
                int language = BinaryPrimitives.ReadUInt16BigEndian(r[4..]), id = BinaryPrimitives.ReadUInt16BigEndian(r[6..]);
                int length = BinaryPrimitives.ReadUInt16BigEndian(r[8..]), offset = BinaryPrimitives.ReadUInt16BigEndian(r[10..]);
                if (language >= 0x8000 || storage + offset + length > source.Length)
                {
                    continue;
                }

                records.Add((platform, encoding, language, id, source.Slice(storage + offset, length).ToArray()));
                mac |= platform == 1;
            }
        }
        else
        {
            records.Add((3, 1, 0x409, (int)FontNameKind.Family, Encoding.BigEndianUnicode.GetBytes(_face.Family)));
            records.Add((3, 1, 0x409, (int)FontNameKind.Style, Encoding.BigEndianUnicode.GetBytes(_face.Style)));
        }

        foreach (var (kind, text) in (_options.Names ?? new Dictionary<FontNameKind, string>()).OrderBy(n => n.Key))
        {
            int id = (int)kind;
            records.RemoveAll(r => r.Id == id);
            records.Add((3, 1, 0x409, id, Encoding.BigEndianUnicode.GetBytes(text)));
            if (mac && text.All(char.IsAscii))
            {
                records.Add((1, 0, 0, id, Encoding.ASCII.GetBytes(text)));
            }
        }

        var sorted = records.OrderBy(r => r.Platform).ThenBy(r => r.Encoding).ThenBy(r => r.Language).ThenBy(r => r.Id).ToList();
        var strings = new FontBuffer();
        var stored = new Dictionary<string, int>(StringComparer.Ordinal);
        var places = new List<(int Offset, int Length)>(sorted.Count);
        foreach (var r in sorted)
        {
            string key = Encoding.Latin1.GetString(r.Text);
            if (!stored.TryGetValue(key, out int offset))
            {
                stored[key] = offset = strings.Length;
                strings.Bytes(r.Text);
            }

            places.Add((offset, r.Text.Length));
        }

        if (strings.Length > 0xFFFF || places.Exists(p => p.Length > 0xFFFF))
        {
            throw new NotSupportedException($"{_name}: its name strings take {strings.Length:N0} bytes, past the 65,535 a name table holds.");
        }

        var w = new FontBuffer(6 + 12 * sorted.Count + strings.Length);
        w.U16(0);
        w.U16(sorted.Count);
        w.U16(6 + 12 * sorted.Count);
        for (int i = 0; i < sorted.Count; i++)
        {
            w.U16(sorted[i].Platform);
            w.U16(sorted[i].Encoding);
            w.U16(sorted[i].Language);
            w.U16(sorted[i].Id);
            w.U16(places[i].Length);
            w.U16(places[i].Offset);
        }

        w.Bytes(strings.Span);
        return w.ToArray();
    }

    // post: the face's angle, underline and pitch; version 2.0 with every glyph's name when the face names its glyphs
    // (a glyph past its names named by its character, else by its id, each name once), else 3.0.
    private byte[] Post(List<(int Code, int Glyph)> characters)
    {
        var source = _face.Table("post").Span;
        var names = PostNames.Each(source, _count);
        var w = new FontBuffer(64);
        w.U32(names is null ? 0x00030000u : 0x00020000u);
        if (source.Length >= 32)
        {
            w.Bytes(source[4..16]);
        }
        else
        {
            w.Zeros(12);
        }

        w.Zeros(16);                                                                       // the memory fields: unknown
        if (names is null)
        {
            return w.ToArray();
        }

        var used = new HashSet<string>(names.OfType<string>(), StringComparer.Ordinal);
        var firstCode = new Dictionary<int, int>();
        foreach (var (code, glyph) in characters)
        {
            firstCode.TryAdd(glyph, code);
        }

        for (int g = 0; g < _count; g++)
        {
            if (names[g] is not null)
            {
                continue;
            }

            string name = firstCode.TryGetValue(g, out int c) ? (c <= 0xFFFF ? $"uni{c:X4}" : $"u{c:X}") : $"glyph{g}";
            for (int k = 1; !used.Add(name); k++)
            {
                name = k == 1 ? $"glyph{g}" : $"glyph{g}.{k}";
            }

            names[g] = name;
        }

        var own = new List<byte[]>();
        var ownIndex = new Dictionary<string, int>(StringComparer.Ordinal);
        w.U16(_count);
        for (int g = 0; g < _count; g++)
        {
            string name = names[g]!;
            if (StandardNames.TryGetValue(name, out int standard))
            {
                w.U16(standard);
                continue;
            }

            if (!ownIndex.TryGetValue(name, out int index))
            {
                byte[] bytes = Encoding.Latin1.GetBytes(name);
                ownIndex[name] = index = own.Count;
                own.Add(bytes.Length <= 255 ? bytes : bytes[..255]);
            }

            w.U16(PostNames.Standard.Length + index);
        }

        foreach (byte[] bytes in own)
        {
            w.U8(bytes.Length);
            w.Bytes(bytes);
        }

        return w.ToArray();
    }

    // OS/2: the face's (version 4 when it was older, its new fields measured), else one made from the face; the
    // character and code page ranges, first and last character, average width and longest context from what is written.
    private byte[] Os2(List<(int Code, int Glyph)> characters, Written[] glyphs, int[] advances, (int XMin, int YMin, int XMax, int YMax) box, int context)
    {
        var source = _face.Table("OS/2").Span;
        int version = source.Length >= 78 ? BinaryPrimitives.ReadUInt16BigEndian(source) : -1;
        var os2 = new byte[version >= 5 && source.Length >= 100 ? 100 : 96];
        var o = os2.AsSpan();
        if (version >= 0)
        {
            source[..Math.Min(source.Length, os2.Length)].CopyTo(o);
        }
        else
        {
            var m = _face.Metrics;
            BinaryPrimitives.WriteUInt16BigEndian(o[4..], (ushort)(Bold ? 700 : 400));
            BinaryPrimitives.WriteUInt16BigEndian(o[6..], 5);
            "NONE"u8.CopyTo(o[58..]);
            BinaryPrimitives.WriteUInt16BigEndian(o[62..], (ushort)((Italic ? 1 : 0) | (Bold ? 0x20 : 0) | (!Bold && !Italic ? 0x40 : 0) | 0x80));
            BinaryPrimitives.WriteInt16BigEndian(o[68..], (short)m.Ascender);
            BinaryPrimitives.WriteInt16BigEndian(o[70..], (short)m.Descender);
            BinaryPrimitives.WriteInt16BigEndian(o[72..], (short)m.LineGap);
            BinaryPrimitives.WriteUInt16BigEndian(o[74..], (ushort)Math.Clamp(Math.Max(box.YMax, m.Ascender), 0, 0xFFFF));
            BinaryPrimitives.WriteUInt16BigEndian(o[76..], (ushort)Math.Clamp(Math.Max(-box.YMin, -m.Descender), 0, 0xFFFF));
        }

        if (version < 2)
        {
            // The fields of version 2 on: the x-height and capital height from 'x' and 'H', the default and break characters.
            BinaryPrimitives.WriteUInt16BigEndian(o, 4);
            BinaryPrimitives.WriteInt16BigEndian(o[86..], (short)Height('x', glyphs));
            BinaryPrimitives.WriteInt16BigEndian(o[88..], (short)Height('H', glyphs));
            BinaryPrimitives.WriteUInt16BigEndian(o[90..], 0);
            BinaryPrimitives.WriteUInt16BigEndian(o[92..], 32);
        }

        var widths = advances.Where(a => a > 0).ToList();
        BinaryPrimitives.WriteInt16BigEndian(o[2..], (short)(widths.Count == 0 ? 0 : (int)Math.Floor(widths.Average() + 0.5)));
        var codes = characters.Select(c => c.Code).ToHashSet();
        var (r1, r2, r3, r4) = FontRanges.Unicode(codes);
        BinaryPrimitives.WriteUInt32BigEndian(o[42..], r1);
        BinaryPrimitives.WriteUInt32BigEndian(o[46..], r2);
        BinaryPrimitives.WriteUInt32BigEndian(o[50..], r3);
        BinaryPrimitives.WriteUInt32BigEndian(o[54..], r4);
        BinaryPrimitives.WriteUInt16BigEndian(o[64..], (ushort)(codes.Count == 0 ? 0 : Math.Min(codes.Min(), 0xFFFF)));
        BinaryPrimitives.WriteUInt16BigEndian(o[66..], (ushort)(codes.Count == 0 ? 0 : Math.Min(codes.Max(), 0xFFFF)));
        var (c1, c2) = FontRanges.CodePages(codes);
        BinaryPrimitives.WriteUInt32BigEndian(o[78..], c1);
        BinaryPrimitives.WriteUInt32BigEndian(o[82..], c2);
        BinaryPrimitives.WriteUInt16BigEndian(o[94..], (ushort)Math.Min(context, 0xFFFF));
        return os2;
    }

    // The top of a character's glyph as written (0 when it has none).
    private int Height(char c, Written[] glyphs)
    {
        int g = _face.Glyph(c);
        return g > 0 && g < glyphs.Length && glyphs[g].Data is not null ? glyphs[g].YMax : 0;
    }

    // The file: the table directory by tag, the tables' data in the recommended order (each padded to 4 bytes, its
    // checksum over the padded data), then head's checkSumAdjustment so the whole file sums to 0xB1B0AFBA.
    private static byte[] Assemble(Dictionary<string, byte[]> tables)
    {
        var tags = tables.Keys.Order(StringComparer.Ordinal).ToList();
        var order = DataOrder.Where(tables.ContainsKey).Concat(tags.Where(t => Array.IndexOf(DataOrder, t) < 0)).ToList();
        int count = tags.Count, power = 1, log = 0;
        while (power * 2 <= count)
        {
            power *= 2;
            log++;
        }

        var offsets = new Dictionary<string, int>(StringComparer.Ordinal);
        int at = 12 + 16 * count;
        foreach (string tag in order)
        {
            offsets[tag] = at;
            at += (tables[tag].Length + 3) & ~3;
        }

        var font = new FontBuffer(at);
        font.U32(0x00010000);
        font.U16(count);
        font.U16(16 * power);
        font.U16(log);
        font.U16(16 * count - 16 * power);
        foreach (string tag in tags)
        {
            font.Tag(tag);
            font.U32(Checksum(tables[tag]));
            font.U32((uint)offsets[tag]);
            font.U32((uint)tables[tag].Length);
        }

        foreach (string tag in order)
        {
            font.Bytes(tables[tag]);
            font.Align(4);
        }

        font.PutU32(offsets["head"] + 8, unchecked(0xB1B0AFBA - Checksum(font.Span)));
        return font.ToArray();
    }

    /// <summary>The OpenType checksum: the sum of the big-endian 32-bit words, the last one padded with zeros.</summary>
    internal static uint Checksum(ReadOnlySpan<byte> data)
    {
        uint sum = 0;
        int whole = data.Length & ~3;
        for (int i = 0; i < whole; i += 4)
        {
            sum = unchecked(sum + BinaryPrimitives.ReadUInt32BigEndian(data[i..]));
        }

        if (whole < data.Length)
        {
            Span<byte> last = stackalloc byte[4];
            last.Clear();
            data[whole..].CopyTo(last);
            sum = unchecked(sum + BinaryPrimitives.ReadUInt32BigEndian(last));
        }

        return sum;
    }
}

/// <summary>OS/2's ranges: the Unicode blocks of the OpenType specification (version 1.7, its bit per block) and its code pages.</summary>
internal static class FontRanges
{
    // Each block's first and last code point and its bit, by first code point (the specification's table).
    private static readonly (int Start, int End, int Bit)[] Blocks =
    [
        (0x0, 0x7F, 0), (0x80, 0xFF, 1), (0x100, 0x17F, 2), (0x180, 0x24F, 3), (0x250, 0x2AF, 4), (0x2B0, 0x2FF, 5), (0x300, 0x36F, 6),
        (0x370, 0x3FF, 7), (0x400, 0x4FF, 9), (0x500, 0x52F, 9), (0x530, 0x58F, 10), (0x590, 0x5FF, 11), (0x600, 0x6FF, 13),
        (0x700, 0x74F, 71), (0x750, 0x77F, 13), (0x780, 0x7BF, 72), (0x7C0, 0x7FF, 14), (0x900, 0x97F, 15), (0x980, 0x9FF, 16),
        (0xA00, 0xA7F, 17), (0xA80, 0xAFF, 18), (0xB00, 0xB7F, 19), (0xB80, 0xBFF, 20), (0xC00, 0xC7F, 21), (0xC80, 0xCFF, 22),
        (0xD00, 0xD7F, 23), (0xD80, 0xDFF, 73), (0xE00, 0xE7F, 24), (0xE80, 0xEFF, 25), (0xF00, 0xFFF, 70), (0x1000, 0x109F, 74),
        (0x10A0, 0x10FF, 26), (0x1100, 0x11FF, 28), (0x1200, 0x137F, 75), (0x1380, 0x139F, 75), (0x13A0, 0x13FF, 76), (0x1400, 0x167F, 77),
        (0x1680, 0x169F, 78), (0x16A0, 0x16FF, 79), (0x1700, 0x171F, 84), (0x1720, 0x173F, 84), (0x1740, 0x175F, 84), (0x1760, 0x177F, 84),
        (0x1780, 0x17FF, 80), (0x1800, 0x18AF, 81), (0x1900, 0x194F, 93), (0x1950, 0x197F, 94), (0x1980, 0x19DF, 95), (0x19E0, 0x19FF, 80),
        (0x1A00, 0x1A1F, 96), (0x1B00, 0x1B7F, 27), (0x1B80, 0x1BBF, 112), (0x1C00, 0x1C4F, 113), (0x1C50, 0x1C7F, 114),
        (0x1D00, 0x1D7F, 4), (0x1D80, 0x1DBF, 4), (0x1DC0, 0x1DFF, 6), (0x1E00, 0x1EFF, 29), (0x1F00, 0x1FFF, 30), (0x2000, 0x206F, 31),
        (0x2070, 0x209F, 32), (0x20A0, 0x20CF, 33), (0x20D0, 0x20FF, 34), (0x2100, 0x214F, 35), (0x2150, 0x218F, 36), (0x2190, 0x21FF, 37),
        (0x2200, 0x22FF, 38), (0x2300, 0x23FF, 39), (0x2400, 0x243F, 40), (0x2440, 0x245F, 41), (0x2460, 0x24FF, 42), (0x2500, 0x257F, 43),
        (0x2580, 0x259F, 44), (0x25A0, 0x25FF, 45), (0x2600, 0x26FF, 46), (0x2700, 0x27BF, 47), (0x27C0, 0x27EF, 38), (0x27F0, 0x27FF, 37),
        (0x2800, 0x28FF, 82), (0x2900, 0x297F, 37), (0x2980, 0x29FF, 38), (0x2A00, 0x2AFF, 38), (0x2B00, 0x2BFF, 37), (0x2C00, 0x2C5F, 97),
        (0x2C60, 0x2C7F, 29), (0x2C80, 0x2CFF, 8), (0x2D00, 0x2D2F, 26), (0x2D30, 0x2D7F, 98), (0x2D80, 0x2DDF, 75), (0x2DE0, 0x2DFF, 9),
        (0x2E00, 0x2E7F, 31), (0x2E80, 0x2EFF, 59), (0x2F00, 0x2FDF, 59), (0x2FF0, 0x2FFF, 59), (0x3000, 0x303F, 48), (0x3040, 0x309F, 49),
        (0x30A0, 0x30FF, 50), (0x3100, 0x312F, 51), (0x3130, 0x318F, 52), (0x3190, 0x319F, 59), (0x31A0, 0x31BF, 51), (0x31C0, 0x31EF, 61),
        (0x31F0, 0x31FF, 50), (0x3200, 0x32FF, 54), (0x3300, 0x33FF, 55), (0x3400, 0x4DBF, 59), (0x4DC0, 0x4DFF, 99), (0x4E00, 0x9FFF, 59),
        (0xA000, 0xA48F, 83), (0xA490, 0xA4CF, 83), (0xA500, 0xA63F, 12), (0xA640, 0xA69F, 9), (0xA700, 0xA71F, 5), (0xA720, 0xA7FF, 29),
        (0xA800, 0xA82F, 100), (0xA840, 0xA87F, 53), (0xA880, 0xA8DF, 115), (0xA900, 0xA92F, 116), (0xA930, 0xA95F, 117),
        (0xAA00, 0xAA5F, 118), (0xAC00, 0xD7AF, 56), (0xD800, 0xDFFF, 57), (0xE000, 0xF8FF, 60), (0xF900, 0xFAFF, 61),
        (0xFB00, 0xFB4F, 62), (0xFB50, 0xFDFF, 63), (0xFE00, 0xFE0F, 91), (0xFE10, 0xFE1F, 65), (0xFE20, 0xFE2F, 64), (0xFE30, 0xFE4F, 65),
        (0xFE50, 0xFE6F, 66), (0xFE70, 0xFEFF, 67), (0xFF00, 0xFFEF, 68), (0xFFF0, 0xFFFF, 69), (0x10000, 0x1007F, 101),
        (0x10080, 0x100FF, 101), (0x10100, 0x1013F, 101), (0x10140, 0x1018F, 102), (0x10190, 0x101CF, 119), (0x101D0, 0x101FF, 120),
        (0x10280, 0x1029F, 121), (0x102A0, 0x102DF, 121), (0x10300, 0x1032F, 85), (0x10330, 0x1034F, 86), (0x10380, 0x1039F, 103),
        (0x103A0, 0x103DF, 104), (0x10400, 0x1044F, 87), (0x10450, 0x1047F, 105), (0x10480, 0x104AF, 106), (0x10800, 0x1083F, 107),
        (0x10900, 0x1091F, 58), (0x10920, 0x1093F, 121), (0x10A00, 0x10A5F, 108), (0x12000, 0x123FF, 110), (0x12400, 0x1247F, 110),
        (0x1D000, 0x1D0FF, 88), (0x1D100, 0x1D1FF, 88), (0x1D200, 0x1D24F, 88), (0x1D300, 0x1D35F, 109), (0x1D360, 0x1D37F, 111),
        (0x1D400, 0x1D7FF, 89), (0x1F000, 0x1F02F, 122), (0x1F030, 0x1F09F, 122), (0x20000, 0x2A6DF, 59), (0x2F800, 0x2FA1F, 61),
        (0xE0000, 0xE007F, 92), (0xE0100, 0xE01EF, 91), (0xF0000, 0xFFFFD, 90), (0x100000, 0x10FFFD, 90),
    ];

    /// <summary>ulUnicodeRange1 to 4: the bit of every block a code point falls in, and bit 57 for any past the basic plane.</summary>
    public static (uint, uint, uint, uint) Unicode(HashSet<int> codes)
    {
        Span<uint> bits = stackalloc uint[4];
        bits.Clear();
        foreach (int code in codes)
        {
            int lo = 0, hi = Blocks.Length - 1, found = -1;
            while (lo <= hi)
            {
                int mid = (lo + hi) >>> 1;
                if (Blocks[mid].Start <= code)
                {
                    found = mid;
                    lo = mid + 1;
                }
                else
                {
                    hi = mid - 1;
                }
            }

            if (found >= 0 && code <= Blocks[found].End)
            {
                Set(bits, Blocks[found].Bit);
            }

            if (code is >= 0x10000 and < 0x110000)
            {
                Set(bits, 57);
            }
        }

        return (bits[0], bits[1], bits[2], bits[3]);
    }

    /// <summary>
    /// ulCodePageRange1 and 2 from the characters (the rule FontForge and fontTools use: a code page's telling character,
    /// with ASCII and box drawing where it asks for them); Latin 1 when none is found.
    /// </summary>
    public static (uint, uint) CodePages(HashSet<int> codes)
    {
        Span<uint> bits = stackalloc uint[2];
        bits.Clear();
        bool ascii = true;
        for (int c = 0x20; c < 0x7E && ascii; c++)
        {
            ascii = codes.Contains(c);
        }

        bool lines = codes.Contains('┤'), root = codes.Contains('√');
        bool Has(char c) => codes.Contains(c);
        if (Has('Þ') && ascii)
        {
            Set(bits, 0);
        }

        if (Has('Ľ') && ascii)
        {
            Set(bits, 1);
            if (lines)
            {
                Set(bits, 58);
            }
        }

        if (Has('Б'))
        {
            Set(bits, 2);
            if (Has('Ѕ') && lines)
            {
                Set(bits, 57);
            }

            if (Has('╜') && lines)
            {
                Set(bits, 49);
            }
        }

        if (Has('Ά'))
        {
            Set(bits, 3);
            if (lines && Has('½'))
            {
                Set(bits, 48);
            }

            if (lines && root)
            {
                Set(bits, 60);
            }
        }

        if (Has('İ') && ascii)
        {
            Set(bits, 4);
            if (lines)
            {
                Set(bits, 56);
            }
        }

        if (Has('א'))
        {
            Set(bits, 5);
            if (lines && root)
            {
                Set(bits, 53);
            }
        }

        if (Has('ر'))
        {
            Set(bits, 6);
            if (root)
            {
                Set(bits, 51);
            }

            if (lines)
            {
                Set(bits, 61);
            }
        }

        if (Has('ŗ') && ascii)
        {
            Set(bits, 7);
            if (lines)
            {
                Set(bits, 59);
            }
        }

        foreach (var (c, bit, needsAscii) in ((char, int, bool)[])[('₫', 8, true), ('ๅ', 16, false), ('エ', 17, false), ('ㄅ', 18, false), ('ㄱ', 19, false),
                     ('央', 20, false), ('곴', 21, false), ('♥', 30, true)])
        {
            if (Has(c) && (ascii || !needsAscii))
            {
                Set(bits, bit);
            }
        }

        if (Has('þ') && ascii && lines)
        {
            Set(bits, 54);
        }

        if (Has('╚') && ascii)
        {
            Set(bits, 62);
            Set(bits, 63);
        }

        if (ascii && lines && root)
        {
            foreach (var (c, bit) in ((char, int)[])[('Å', 50), ('é', 52), ('õ', 55)])
            {
                if (Has(c))
                {
                    Set(bits, bit);
                }
            }
        }

        if (ascii && Has('‰') && Has('∑'))
        {
            Set(bits, 29);
        }

        if (bits[0] == 0 && bits[1] == 0)
        {
            Set(bits, 0);
        }

        return (bits[0], bits[1]);
    }

    private static void Set(Span<uint> bits, int bit) => bits[bit / 32] |= 1u << (bit % 32);
}
