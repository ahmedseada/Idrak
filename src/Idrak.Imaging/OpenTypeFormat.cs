// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Buffers.Binary;
using System.Text;

namespace Idrak.Imaging;

/// <summary>
/// "opentype": OpenType and TrueType fonts (.ttf, .otf, .ttc), single fonts and collections, read in C#, with TrueType
/// outlines (glyf) or CFF outlines (Type 2 charstrings, CID-keyed fonts included). CFF2 (variable fonts' outlines) is
/// refused as not supported yet. Hinting is not run: outlines are drawn as designed, at any size.
/// </summary>
public sealed class OpenTypeFormat : IFontFormat
{
    /// <summary>The format's name.</summary>
    public const string FormatName = "opentype";

    /// <inheritdoc/>
    public string Name => FormatName;

    /// <inheritdoc/>
    public IReadOnlyCollection<string> Extensions { get; } = [".ttf", ".otf", ".ttc"];

    /// <inheritdoc/>
    public bool Accepts(ReadOnlySpan<byte> header) =>
        header.Length >= 4 && (header[..4].SequenceEqual("\0\u0001\0\0"u8) || header[..4].SequenceEqual("true"u8)
                               || header[..4].SequenceEqual("OTTO"u8) || header[..4].SequenceEqual("ttcf"u8));

    /// <inheritdoc/>
    public IReadOnlyList<IFontFace> Read(ReadOnlyMemory<byte> file, string name)
    {
        var data = file.Span;
        if (data.Length < 12)
        {
            throw new InvalidDataException($"{name}: {data.Length} bytes are too few for a font.");
        }

        if (data[..4].SequenceEqual("ttcf"u8))
        {
            int count = (int)BinaryPrimitives.ReadUInt32BigEndian(data[8..]);
            if (count <= 0 || 12 + 4L * count > data.Length)
            {
                throw new InvalidDataException($"{name}: a font collection of {count} fonts does not fit its {data.Length} bytes.");
            }

            var faces = new IFontFace[count];
            for (int i = 0; i < count; i++)
            {
                faces[i] = new OpenTypeFace(file, (int)BinaryPrimitives.ReadUInt32BigEndian(data[(12 + 4 * i)..]), $"{name} #{i}");
            }

            return faces;
        }

        return [new OpenTypeFace(file, 0, name)];
    }
}

/// <summary>One font of an OpenType file (TrueType outlines): glyphs by character, advances, outlines and raw tables.</summary>
internal sealed class OpenTypeFace : IFontFace
{
    private const int MaxComponentDepth = 16;

    private readonly ReadOnlyMemory<byte> _file;
    private readonly Dictionary<string, (int Offset, int Length)> _tables = new(StringComparer.Ordinal);
    private readonly ushort[] _advances;
    private readonly int[]? _locations;                                 // TrueType outlines
    private readonly CffOutlines? _cff;                                 // or CFF outlines
    private readonly ushort[]? _basic;                                  // code point 0 to FFFF to glyph (cmap format 4)
    private readonly (uint Start, uint End, uint Glyph)[]? _groups;      // code point ranges to glyphs (cmap format 12)
    private readonly GlyphOutline?[] _outlines;                          // read once each, on first use
    private readonly string _name;

    public OpenTypeFace(ReadOnlyMemory<byte> file, int offset, string name)
    {
        _file = file;
        _name = name;
        var data = file.Span;
        if (offset < 0 || offset + 12 > data.Length)
        {
            throw new InvalidDataException($"{name}: the table directory at {offset} lies outside the file.");
        }

        int tables = BinaryPrimitives.ReadUInt16BigEndian(data[(offset + 4)..]);
        for (int i = 0; i < tables; i++)
        {
            var record = data.Slice(offset + 12 + 16 * i, 16);
            string name4 = Encoding.ASCII.GetString(record[..4]);
            int at = (int)BinaryPrimitives.ReadUInt32BigEndian(record[8..]), length = (int)BinaryPrimitives.ReadUInt32BigEndian(record[12..]);
            if (at < 0 || length < 0 || (long)at + length > data.Length)
            {
                throw new InvalidDataException($"{name}: the table '{name4}' ({at}, {length} bytes) lies outside the file.");
            }

            _tables[name4] = (at, length);
        }

        foreach (string required in (string[])["head", "hhea", "maxp", "hmtx", "cmap"])
        {
            if (!_tables.ContainsKey(required))
            {
                throw new InvalidDataException($"{name}: no '{required}' table.");
            }
        }

        bool trueType = _tables.ContainsKey("glyf") && _tables.ContainsKey("loca");
        if (!trueType && !_tables.ContainsKey("CFF "))
        {
            throw new NotSupportedException(_tables.ContainsKey("CFF2")
                ? $"{name}: its outlines are CFF2 (a variable font), which are not read yet; use a static .ttf or .otf."
                : $"{name}: no outlines the library reads (glyf, or CFF).");
        }

        var head = Span("head");
        int unitsPerEm = BinaryPrimitives.ReadUInt16BigEndian(head[18..]);
        bool longOffsets = BinaryPrimitives.ReadInt16BigEndian(head[50..]) != 0;
        var hhea = Span("hhea");
        Metrics = new FontMetrics(unitsPerEm, BinaryPrimitives.ReadInt16BigEndian(hhea[4..]), BinaryPrimitives.ReadInt16BigEndian(hhea[6..]),
            BinaryPrimitives.ReadInt16BigEndian(hhea[8..]));
        int metrics = BinaryPrimitives.ReadUInt16BigEndian(hhea[34..]);
        GlyphCount = BinaryPrimitives.ReadUInt16BigEndian(Span("maxp")[4..]);
        if (unitsPerEm == 0 || GlyphCount == 0 || metrics == 0 || metrics > GlyphCount)
        {
            throw new InvalidDataException($"{name}: {unitsPerEm} units per em, {GlyphCount} glyphs, {metrics} horizontal metrics.");
        }

        var hmtx = Span("hmtx");
        _advances = new ushort[GlyphCount];
        for (int g = 0; g < GlyphCount; g++)
        {
            _advances[g] = BinaryPrimitives.ReadUInt16BigEndian(hmtx[(4 * Math.Min(g, metrics - 1))..]);
        }

        if (trueType)
        {
            var loca = Span("loca");
            _locations = new int[GlyphCount + 1];
            for (int g = 0; g <= GlyphCount; g++)
            {
                _locations[g] = longOffsets ? (int)BinaryPrimitives.ReadUInt32BigEndian(loca[(4 * g)..]) : 2 * BinaryPrimitives.ReadUInt16BigEndian(loca[(2 * g)..]);
            }
        }
        else
        {
            _cff = new CffOutlines(Table("CFF "), unitsPerEm, GlyphCount, name);
        }

        (_basic, _groups) = ReadCmap(Span("cmap"));
        _outlines = new GlyphOutline?[GlyphCount];
        (Family, Style) = ReadNames(name);
    }

    public string Family { get; }

    public string Style { get; }

    public string Format => OpenTypeFormat.FormatName;

    public FontMetrics Metrics { get; }

    public int GlyphCount { get; }

    public int Glyph(int codePoint)
    {
        if (_groups is { } groups)
        {
            int lo = 0, hi = groups.Length - 1;
            uint c = (uint)codePoint;
            while (lo <= hi)
            {
                int mid = (lo + hi) >>> 1;
                if (c < groups[mid].Start)
                {
                    hi = mid - 1;
                }
                else if (c > groups[mid].End)
                {
                    lo = mid + 1;
                }
                else
                {
                    uint glyph = groups[mid].Glyph + (c - groups[mid].Start);
                    return glyph < GlyphCount ? (int)glyph : 0;
                }
            }

            return 0;
        }

        return (uint)codePoint <= 0xFFFF ? _basic![codePoint] : 0;
    }

    public int Advance(int glyph) => (uint)glyph < (uint)GlyphCount ? _advances[glyph] : 0;

    public GlyphOutline Outline(int glyph)
    {
        if ((uint)glyph >= (uint)GlyphCount)
        {
            return GlyphOutline.Empty;
        }

        var outline = Volatile.Read(ref _outlines[glyph]);
        if (outline is null)
        {
            if (_cff is not null)
            {
                outline = _cff.Outline(glyph);
                Volatile.Write(ref _outlines[glyph], outline);
                return outline;
            }

            var contours = new List<Contour>();
            AddGlyph(glyph, 1, 0, 0, 1, 0, 0, contours, 0);
            outline = ToOutline(contours);
            Volatile.Write(ref _outlines[glyph], outline);                // two threads may both read it: the same outline
        }

        return outline;
    }

    public ReadOnlyMemory<byte> Table(string tag) => _tables.TryGetValue(tag, out var t) ? _file.Slice(t.Offset, t.Length) : ReadOnlyMemory<byte>.Empty;

    private ReadOnlySpan<byte> Span(string tag) => Table(tag).Span;

    // A contour's points in font units, each on the curve or a quadratic control point.
    private sealed class Contour(int capacity)
    {
        public readonly List<(float X, float Y, bool On)> Points = new(capacity);
    }

    // Appends glyph's contours, transformed by [a b; c d] then moved by (dx, dy) (a component of a composite glyph).
    private void AddGlyph(int glyph, float a, float b, float c, float d, float dx, float dy, List<Contour> contours, int depth)
    {
        if (depth > MaxComponentDepth)
        {
            throw new InvalidDataException($"{_name}: glyph {glyph}'s components nest deeper than {MaxComponentDepth}.");
        }

        int start = _locations![glyph], end = _locations[glyph + 1];
        if (end <= start)
        {
            return;                                                             // no outline (a space)
        }

        var glyf = Span("glyf");
        if (end > glyf.Length)
        {
            throw new InvalidDataException($"{_name}: glyph {glyph} ({start} to {end}) lies outside the glyf table.");
        }

        var data = glyf[start..end];
        int count = BinaryPrimitives.ReadInt16BigEndian(data);
        if (count >= 0)
        {
            AddSimple(data, count, a, b, c, d, dx, dy, contours);
        }
        else
        {
            AddComposite(data, a, b, c, d, dx, dy, contours, depth);
        }
    }

    private static void AddSimple(ReadOnlySpan<byte> data, int contourCount, float a, float b, float c, float d, float dx, float dy, List<Contour> contours)
    {
        if (contourCount == 0)
        {
            return;
        }

        int at = 10;
        Span<int> ends = contourCount <= 256 ? stackalloc int[contourCount] : new int[contourCount];
        for (int i = 0; i < contourCount; i++, at += 2)
        {
            ends[i] = BinaryPrimitives.ReadUInt16BigEndian(data[at..]);
        }

        int points = ends[^1] + 1;
        at += 2 + BinaryPrimitives.ReadUInt16BigEndian(data[at..]);              // skip the instructions
        var flags = new byte[points];
        for (int i = 0; i < points;)
        {
            byte flag = data[at++];
            flags[i++] = flag;
            if ((flag & 8) != 0)
            {
                for (int repeat = data[at++]; repeat > 0 && i < points; repeat--)
                {
                    flags[i++] = flag;
                }
            }
        }

        var xs = new int[points];
        int value = 0;
        for (int i = 0; i < points; i++)
        {
            byte flag = flags[i];
            if ((flag & 2) != 0)
            {
                int delta = data[at++];
                value += (flag & 16) != 0 ? delta : -delta;
            }
            else if ((flag & 16) == 0)
            {
                value += BinaryPrimitives.ReadInt16BigEndian(data[at..]);
                at += 2;
            }

            xs[i] = value;
        }

        value = 0;
        int first = 0;
        for (int contour = 0, i = 0; contour < contourCount; contour++)
        {
            var points2 = new Contour(ends[contour] - first + 1);
            for (; i <= ends[contour]; i++)
            {
                byte flag = flags[i];
                if ((flag & 4) != 0)
                {
                    int delta = data[at++];
                    value += (flag & 32) != 0 ? delta : -delta;
                }
                else if ((flag & 32) == 0)
                {
                    value += BinaryPrimitives.ReadInt16BigEndian(data[at..]);
                    at += 2;
                }

                float x = xs[i], y = value;
                points2.Points.Add((a * x + c * y + dx, b * x + d * y + dy, (flag & 1) != 0));
            }

            first = ends[contour] + 1;
            contours.Add(points2);
        }
    }

    private void AddComposite(ReadOnlySpan<byte> data, float a, float b, float c, float d, float dx, float dy, List<Contour> contours, int depth)
    {
        const int ArgsAreWords = 1, ArgsAreXY = 2, HaveScale = 8, MoreComponents = 32, HaveXYScale = 64, HaveTwoByTwo = 128, ScaledOffset = 0x800;
        int at = 10, flags;
        do
        {
            flags = BinaryPrimitives.ReadUInt16BigEndian(data[at..]);
            int component = BinaryPrimitives.ReadUInt16BigEndian(data[(at + 2)..]);
            at += 4;
            int arg1, arg2;
            if ((flags & ArgsAreWords) != 0)
            {
                bool signed = (flags & ArgsAreXY) != 0;
                arg1 = signed ? BinaryPrimitives.ReadInt16BigEndian(data[at..]) : BinaryPrimitives.ReadUInt16BigEndian(data[at..]);
                arg2 = signed ? BinaryPrimitives.ReadInt16BigEndian(data[(at + 2)..]) : BinaryPrimitives.ReadUInt16BigEndian(data[(at + 2)..]);
                at += 4;
            }
            else
            {
                bool signed = (flags & ArgsAreXY) != 0;
                arg1 = signed ? (sbyte)data[at] : data[at];
                arg2 = signed ? (sbyte)data[at + 1] : data[at + 1];
                at += 2;
            }

            float ca = 1, cb = 0, cc = 0, cd = 1;                                // the component's own transform
            if ((flags & HaveScale) != 0)
            {
                ca = cd = F2Dot14(data[at..]);
                at += 2;
            }
            else if ((flags & HaveXYScale) != 0)
            {
                ca = F2Dot14(data[at..]);
                cd = F2Dot14(data[(at + 2)..]);
                at += 4;
            }
            else if ((flags & HaveTwoByTwo) != 0)
            {
                ca = F2Dot14(data[at..]);
                cb = F2Dot14(data[(at + 2)..]);
                cc = F2Dot14(data[(at + 4)..]);
                cd = F2Dot14(data[(at + 6)..]);
                at += 8;
            }

            if ((uint)component >= (uint)GlyphCount)
            {
                throw new InvalidDataException($"{_name}: a composite glyph names component {component} of {GlyphCount} glyphs.");
            }

            // The component in the parent's units: its transform, then its offset (scaled by its transform only when the
            // font says so; unscaled is the Microsoft rule FreeType follows by default).
            var own = new List<Contour>();
            AddGlyph(component, ca, cb, cc, cd, 0, 0, own, depth + 1);
            float ox, oy;
            if ((flags & ArgsAreXY) != 0)
            {
                (ox, oy) = (flags & ScaledOffset) != 0 ? (ca * arg1 + cc * arg2, cb * arg1 + cd * arg2) : (arg1, arg2);
            }
            else
            {
                // Point matching: the component's point arg2 lands on the parent's point arg1 (both counted across contours).
                var parent = PointAt(contours, arg1, contours.Count);
                var child = PointAt(own, arg2, own.Count);
                (ox, oy) = (parent.X - child.X, parent.Y - child.Y);
            }

            foreach (var contour in own)
            {
                var moved = new Contour(contour.Points.Count);
                foreach (var (x, y, on) in contour.Points)
                {
                    float px = x + ox, py = y + oy;
                    moved.Points.Add((a * px + c * py + dx, b * px + d * py + dy, on));
                }

                contours.Add(moved);
            }
        }
        while ((flags & MoreComponents) != 0);
    }

    private (float X, float Y) PointAt(List<Contour> contours, int index, int count)
    {
        for (int i = 0; i < count; i++)
        {
            if (index < contours[i].Points.Count)
            {
                var p = contours[i].Points[index];
                return (p.X, p.Y);
            }

            index -= contours[i].Points.Count;
        }

        throw new InvalidDataException($"{_name}: a composite glyph matches a point past its components' points.");
    }

    private static float F2Dot14(ReadOnlySpan<byte> data) => BinaryPrimitives.ReadInt16BigEndian(data) / 16384f;

    // TrueType contours to moves, lines and quadratic curves: two control points in a row have an on-curve point midway.
    private static GlyphOutline ToOutline(List<Contour> contours)
    {
        var verbs = new List<OutlineVerb>();
        var points = new List<float>();
        foreach (var contour in contours)
        {
            var p = contour.Points;
            int n = p.Count;
            if (n == 0)
            {
                continue;
            }

            // The start: the first point on the curve, else the midway point of the last and the first.
            int first = p.FindIndex(q => q.On);
            (float X, float Y) start = first >= 0 ? (p[first].X, p[first].Y) : ((p[0].X + p[^1].X) / 2, (p[0].Y + p[^1].Y) / 2);
            int begin = first >= 0 ? first : 0;
            verbs.Add(OutlineVerb.Move);
            points.Add(start.X);
            points.Add(start.Y);
            (float X, float Y)? control = null;
            for (int k = 1; k <= n; k++)
            {
                var q = p[(begin + k) % n];
                bool closing = k == n;
                if (first < 0 && closing)
                {
                    q = (start.X, start.Y, true);                               // an all-control contour ends where it began
                }

                if (q.On)
                {
                    if (control is { } ctl)
                    {
                        verbs.Add(OutlineVerb.Quadratic);
                        points.Add(ctl.X);
                        points.Add(ctl.Y);
                    }
                    else
                    {
                        verbs.Add(OutlineVerb.Line);
                    }

                    points.Add(q.X);
                    points.Add(q.Y);
                    control = null;
                }
                else
                {
                    if (control is { } ctl)
                    {
                        verbs.Add(OutlineVerb.Quadratic);
                        points.Add(ctl.X);
                        points.Add(ctl.Y);
                        points.Add((ctl.X + q.X) / 2);
                        points.Add((ctl.Y + q.Y) / 2);
                    }

                    control = (q.X, q.Y);
                }
            }

            if (control is { } last)                                            // a control point before the start
            {
                verbs.Add(OutlineVerb.Quadratic);
                points.Add(last.X);
                points.Add(last.Y);
                points.Add(start.X);
                points.Add(start.Y);
            }

            verbs.Add(OutlineVerb.Close);
        }

        return verbs.Count == 0 ? GlyphOutline.Empty : new GlyphOutline([.. verbs], [.. points]);
    }

    // The character map: format 12 (all planes) when the font has it, else format 4 (the basic plane).
    private (ushort[]? Basic, (uint, uint, uint)[]? Groups) ReadCmap(ReadOnlySpan<byte> cmap)
    {
        int tables = BinaryPrimitives.ReadUInt16BigEndian(cmap[2..]);
        int format4 = -1, format12 = -1;
        for (int i = 0; i < tables; i++)
        {
            var record = cmap.Slice(4 + 8 * i, 8);
            int platform = BinaryPrimitives.ReadUInt16BigEndian(record), encoding = BinaryPrimitives.ReadUInt16BigEndian(record[2..]);
            int offset = (int)BinaryPrimitives.ReadUInt32BigEndian(record[4..]);
            bool unicode = platform == 0 || (platform == 3 && encoding is 1 or 10);
            if (!unicode || offset < 0 || offset + 4 > cmap.Length)
            {
                continue;
            }

            int format = BinaryPrimitives.ReadUInt16BigEndian(cmap[offset..]);
            if (format == 12 && format12 < 0)
            {
                format12 = offset;
            }
            else if (format == 4 && format4 < 0)
            {
                format4 = offset;
            }
        }

        if (format12 >= 0)
        {
            var table = cmap[format12..];
            int count = (int)BinaryPrimitives.ReadUInt32BigEndian(table[12..]);
            var groups = new (uint, uint, uint)[count];
            for (int i = 0; i < count; i++)
            {
                var g = table.Slice(16 + 12 * i, 12);
                groups[i] = (BinaryPrimitives.ReadUInt32BigEndian(g), BinaryPrimitives.ReadUInt32BigEndian(g[4..]), BinaryPrimitives.ReadUInt32BigEndian(g[8..]));
            }

            return (null, groups);
        }

        if (format4 < 0)
        {
            throw new NotSupportedException($"{_name}: no Unicode character map of format 4 or 12.");
        }

        var t = cmap[format4..];
        int segments = BinaryPrimitives.ReadUInt16BigEndian(t[6..]) / 2;
        int endAt = 14, startAt = endAt + 2 * segments + 2, deltaAt = startAt + 2 * segments, rangeAt = deltaAt + 2 * segments;
        var map = new ushort[65536];
        for (int s = 0; s < segments; s++)
        {
            int last = BinaryPrimitives.ReadUInt16BigEndian(t[(endAt + 2 * s)..]), first = BinaryPrimitives.ReadUInt16BigEndian(t[(startAt + 2 * s)..]);
            int delta = BinaryPrimitives.ReadInt16BigEndian(t[(deltaAt + 2 * s)..]), range = BinaryPrimitives.ReadUInt16BigEndian(t[(rangeAt + 2 * s)..]);
            for (int code = first; code <= last && code != 0xFFFF; code++)
            {
                int glyph;
                if (range == 0)
                {
                    glyph = (code + delta) & 0xFFFF;
                }
                else
                {
                    int at = rangeAt + 2 * s + range + 2 * (code - first);
                    glyph = at + 2 <= t.Length ? BinaryPrimitives.ReadUInt16BigEndian(t[at..]) : 0;
                    if (glyph != 0)
                    {
                        glyph = (glyph + delta) & 0xFFFF;
                    }
                }

                map[code] = glyph < GlyphCount ? (ushort)glyph : (ushort)0;
            }
        }

        return (map, null);
    }

    // Family and style from the name table: the typographic names (16, 17) when present, else 1 and 2; Windows Unicode
    // (UTF-16 big-endian), else Macintosh Roman read as ASCII.
    private (string Family, string Style) ReadNames(string fallback)
    {
        var name = Span("name");
        if (name.Length < 6)
        {
            return (fallback, "Regular");
        }

        int count = BinaryPrimitives.ReadUInt16BigEndian(name[2..]), storage = BinaryPrimitives.ReadUInt16BigEndian(name[4..]);
        var found = new Dictionary<int, string>();
        for (int i = 0; i < count && 6 + 12 * (i + 1) <= name.Length; i++)
        {
            var r = name.Slice(6 + 12 * i, 12);
            int platform = BinaryPrimitives.ReadUInt16BigEndian(r), language = BinaryPrimitives.ReadUInt16BigEndian(r[4..]);
            int id = BinaryPrimitives.ReadUInt16BigEndian(r[6..]), length = BinaryPrimitives.ReadUInt16BigEndian(r[8..]), offset = BinaryPrimitives.ReadUInt16BigEndian(r[10..]);
            if (id is not (1 or 2 or 16 or 17) || storage + offset + length > name.Length)
            {
                continue;
            }

            var bytes = name.Slice(storage + offset, length);
            if (platform == 3 && (language == 0x409 || !found.ContainsKey(id)))
            {
                found[id] = Encoding.BigEndianUnicode.GetString(bytes);
            }
            else if (platform == 1 && !found.ContainsKey(id))
            {
                found[id] = Encoding.ASCII.GetString(bytes);
            }
        }

        string family = found.GetValueOrDefault(16) ?? found.GetValueOrDefault(1) ?? fallback;
        string style = found.GetValueOrDefault(17) ?? found.GetValueOrDefault(2) ?? "Regular";
        return (family, style);
    }
}
