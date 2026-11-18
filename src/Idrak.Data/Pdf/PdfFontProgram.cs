// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Buffers.Binary;

namespace Idrak.Data.Pdf;

/// <summary>
/// A font program a PDF embeds (TrueType, or OpenType), read through <see cref="FontFormats"/> at run time (this assembly
/// does not reference the readers): its face, for drawing (<see cref="DocumentTextRun.Face"/>), and its own <c>cmap</c>
/// read here both ways: each glyph's lowest code point in the best Unicode subtable (what the glyph says when the file
/// gives no other way: the third way to text), and the symbol (3,0) and Macintosh (1,0) subtables a simple TrueType
/// font's codes reach glyphs through. Read once per document, when first needed; immutable after.
/// </summary>
internal sealed class PdfFontProgram
{
    private readonly int[] _codePoints;
    private readonly Dictionary<int, int>? _symbol;
    private readonly Dictionary<int, int>? _macintosh;

    private PdfFontProgram(IFontFace face, int[] codePoints, Dictionary<int, int>? symbol, Dictionary<int, int>? macintosh) =>
        (Face, _codePoints, _symbol, _macintosh) = (face, codePoints, symbol, macintosh);

    /// <summary>The program as a face.</summary>
    public IFontFace Face { get; }

    /// <summary>About how many bytes it holds.</summary>
    public long Bytes { get; private init; }

    /// <summary>
    /// The program in <paramref name="program"/>'s bytes as the first registered font format that accepts them reads it,
    /// or null when none does or it cannot be read (a program without a <c>cmap</c>, a bare CFF, a damaged file).
    /// </summary>
    public static PdfFontProgram? Read(byte[] program, string name)
    {
        var header = program.AsSpan(0, Math.Min(program.Length, FontFormats.HeaderBytes));
        foreach (string formatName in FontFormats.Names)
        {
            IFontFormat format;
            try
            {
                format = FontFormats.Get(formatName);
                if (!format.Accepts(header))
                {
                    continue;
                }

                var faces = format.Read(program, name);
                if (faces.Count == 0)
                {
                    continue;
                }

                var face = faces[0];
                var cmap = face.Table("cmap").Span;
                var (codePoints, symbol, macintosh) = ReadCmap(cmap, face.GlyphCount);
                return new PdfFontProgram(face, codePoints, symbol, macintosh) { Bytes = program.Length + 4L * codePoints.Length + 4096 };
            }
            catch (Exception e) when (e is not (OutOfMemoryException or OperationCanceledException))
            {
                return null;   // a program the format cannot read: the file gives its text another way, or none
            }
        }

        return null;
    }

    /// <summary>The lowest code point the program's best Unicode <c>cmap</c> maps to <paramref name="glyph"/>, or -1.</summary>
    public int CodePoint(int glyph) => (uint)glyph < (uint)_codePoints.Length ? _codePoints[glyph] : -1;

    /// <summary>
    /// The glyph a simple symbolic TrueType font's <paramref name="code"/> reaches (ISO 32000-2, 9.6.5.4): the (3,0)
    /// subtable at F000, F100 or F200 plus the code, or the code itself, then the (1,0) subtable; -1 when none maps it.
    /// </summary>
    public int SymbolGlyph(int code)
    {
        if (_symbol is { } symbol)
        {
            ReadOnlySpan<int> candidates = [0xF000 + code, 0xF100 + code, 0xF200 + code, code];
            foreach (int c in candidates)
            {
                if (symbol.TryGetValue(c, out int glyph) && glyph != 0)
                {
                    return glyph;
                }
            }
        }

        return _macintosh is { } mac && mac.TryGetValue(code, out int g) && g != 0 ? g : -1;
    }

    // The cmap's subtables: the best Unicode one ((3,10), (0,6), (0,4), (3,1), (0,3), (0,2), (0,1), (0,0), as font tools
    // pick it) reversed into each glyph's lowest code point; the (3,0) and (1,0) ones forwards.
    private static (int[] CodePoints, Dictionary<int, int>? Symbol, Dictionary<int, int>? Macintosh) ReadCmap(ReadOnlySpan<byte> cmap, int glyphCount)
    {
        var codePoints = new int[Math.Max(glyphCount, 0)];
        Array.Fill(codePoints, -1);
        if (cmap.Length < 4)
        {
            return (codePoints, null, null);
        }

        int tables = BinaryPrimitives.ReadUInt16BigEndian(cmap[2..]);
        ReadOnlySpan<(int Platform, int Encoding)> preference = [(3, 10), (0, 6), (0, 4), (3, 1), (0, 3), (0, 2), (0, 1), (0, 0)];
        int best = -1, bestRank = int.MaxValue, symbolAt = -1, macAt = -1;
        for (int i = 0; i < tables && 4 + 8 * i + 8 <= cmap.Length; i++)
        {
            var record = cmap.Slice(4 + 8 * i, 8);
            int platform = BinaryPrimitives.ReadUInt16BigEndian(record), encoding = BinaryPrimitives.ReadUInt16BigEndian(record[2..]);
            long offset = BinaryPrimitives.ReadUInt32BigEndian(record[4..]);
            if (offset >= cmap.Length)
            {
                continue;
            }

            for (int rank = 0; rank < preference.Length; rank++)
            {
                if (preference[rank] == (platform, encoding) && rank < bestRank)
                {
                    (best, bestRank) = ((int)offset, rank);
                }
            }

            if (platform == 3 && encoding == 0)
            {
                symbolAt = (int)offset;
            }
            else if (platform == 1 && encoding == 0)
            {
                macAt = (int)offset;
            }
        }

        if (best >= 0)
        {
            Mappings(cmap, best, glyphCount, (code, glyph) =>
            {
                if (codePoints[glyph] < 0 || code < codePoints[glyph])
                {
                    codePoints[glyph] = code;
                }
            });
        }

        return (codePoints, Forward(cmap, symbolAt, glyphCount), Forward(cmap, macAt, glyphCount));
    }

    // A subtable's codes to glyphs, or null when there is none.
    private static Dictionary<int, int>? Forward(ReadOnlySpan<byte> cmap, int at, int glyphCount)
    {
        if (at < 0)
        {
            return null;
        }

        var map = new Dictionary<int, int>();
        Mappings(cmap, at, glyphCount, (code, glyph) => map.TryAdd(code, glyph));
        return map;
    }

    // Every (code, glyph) of the subtable at `at` (formats 0, 4, 6 and 12), glyphs within the font only; a damaged
    // subtable gives what it holds up to its damage.
    private static void Mappings(ReadOnlySpan<byte> cmap, int at, int glyphCount, Action<int, int> add)
    {
        var table = cmap[at..];
        if (table.Length < 4)
        {
            return;
        }

        int format = BinaryPrimitives.ReadUInt16BigEndian(table);
        switch (format)
        {
            case 0:
                for (int code = 0; code < 256 && 6 + code < table.Length; code++)
                {
                    Add(code, table[6 + code]);
                }

                break;
            case 4:
                if (table.Length < 14)
                {
                    return;
                }

                int segments = BinaryPrimitives.ReadUInt16BigEndian(table[6..]) / 2;
                int ends = 14, starts = 16 + 2 * segments, deltas = starts + 2 * segments, rangeOffsets = deltas + 2 * segments;
                if (rangeOffsets + 2 * segments > table.Length)
                {
                    return;
                }

                for (int s = 0; s < segments; s++)
                {
                    int end = BinaryPrimitives.ReadUInt16BigEndian(table[(ends + 2 * s)..]), start = BinaryPrimitives.ReadUInt16BigEndian(table[(starts + 2 * s)..]);
                    int delta = BinaryPrimitives.ReadInt16BigEndian(table[(deltas + 2 * s)..]), rangeOffset = BinaryPrimitives.ReadUInt16BigEndian(table[(rangeOffsets + 2 * s)..]);
                    for (int code = start; code <= end && code != 0xFFFF; code++)
                    {
                        int glyph;
                        if (rangeOffset == 0)
                        {
                            glyph = (code + delta) & 0xFFFF;
                        }
                        else
                        {
                            int address = rangeOffsets + 2 * s + rangeOffset + 2 * (code - start);
                            if (address + 2 > table.Length)
                            {
                                break;
                            }

                            glyph = BinaryPrimitives.ReadUInt16BigEndian(table[address..]);
                            if (glyph != 0)
                            {
                                glyph = (glyph + delta) & 0xFFFF;
                            }
                        }

                        Add(code, glyph);
                    }
                }

                break;
            case 6:
                if (table.Length < 10)
                {
                    return;
                }

                int first = BinaryPrimitives.ReadUInt16BigEndian(table[6..]), count = BinaryPrimitives.ReadUInt16BigEndian(table[8..]);
                for (int i = 0; i < count && 10 + 2 * i + 2 <= table.Length; i++)
                {
                    Add(first + i, BinaryPrimitives.ReadUInt16BigEndian(table[(10 + 2 * i)..]));
                }

                break;
            case 12:
                if (table.Length < 16)
                {
                    return;
                }

                long groups = BinaryPrimitives.ReadUInt32BigEndian(table[12..]);
                for (long g = 0; g < groups && 16 + 12 * g + 12 <= table.Length; g++)
                {
                    var group = table.Slice((int)(16 + 12 * g), 12);
                    long start = BinaryPrimitives.ReadUInt32BigEndian(group), end = BinaryPrimitives.ReadUInt32BigEndian(group[4..]), glyph = BinaryPrimitives.ReadUInt32BigEndian(group[8..]);
                    if (start > 0x10FFFF || end < start || glyph >= glyphCount)
                    {
                        continue;
                    }

                    long last = Math.Min(Math.Min(end, 0x10FFFF), start + (glyphCount - 1 - glyph));
                    for (long code = start; code <= last; code++)
                    {
                        Add((int)code, (int)(glyph + (code - start)));
                    }
                }

                break;
        }

        void Add(int code, int glyph)
        {
            if (glyph > 0 && glyph < glyphCount)
            {
                add(code, glyph);
            }
        }
    }
}
