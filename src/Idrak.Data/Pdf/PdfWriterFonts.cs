// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Buffers.Binary;
using System.Security.Cryptography;
using System.Text;

namespace Idrak.Data.Pdf;

/// <summary>
/// One face of a document being written: embedded once, at the end, as a composite font (<c>Type0</c> over a
/// <c>CIDFontType2</c>, <c>Identity-H</c>: each glyph shown by its id as two bytes, <c>CIDToGIDMap Identity</c>), its
/// program written through the "truetype" writer (a subset of the glyphs used, their ids kept, or the
/// whole face), <c>/W</c> from the advances, and a <c>ToUnicode</c> map from the runs' text: each glyph the characters
/// it first stood for, found by aligning a run's glyphs with its text through the face's own character map (the glyphs
/// in the text's order or reversed, whichever matches more; between two glyphs that match, the characters between them
/// go to the glyphs between them, several to one where a glyph stands for several). A run whose glyphs do not read back
/// as its text (a ligature's characters in another order, a character drawn by several glyphs, a glyph that stood for
/// other text before, text right to left) is marked so its text is written with it (<c>/ActualText</c>). No script is
/// named: the alignment uses the face's characters alone.
/// </summary>
internal sealed class PdfWriterFont
{
    /// <summary>The font writer the programs are written with.</summary>
    public const string ProgramWriter = "truetype";

    // Looking up the reading orders and the font writers by name is not on the new design yet. The ones the writer uses
    // are reached directly: the "bidi" order (Idrak.Imaging's BidiReadingOrder) and the "truetype" font writer
    // (Idrak.Imaging's TrueTypeWriter); another name stops.
    private static readonly Idrak.Imaging.Shaping.BidiReadingOrder Bidi = new();

    private static readonly Idrak.Imaging.TrueTypeWriter TrueType = new();

    private static IReadingOrder OrderNamed(string name) =>
        name.Equals(ReadingOrders.Bidi, StringComparison.OrdinalIgnoreCase) ? Bidi : throw new NotSupportedException($"The reading order '{name}' by name is not on the new design yet.");

    private static IFontWriter WriterNamed(string name) =>
        name.Equals(ProgramWriter, StringComparison.OrdinalIgnoreCase) ? TrueType : throw new NotSupportedException($"The font writer '{name}' by name is not on the new design yet.");

    // How far ahead a glyph looks for its character when aligning a run.
    private const int Window = 8;

    private readonly bool[] _used;
    private readonly double[] _widths;
    private readonly Dictionary<int, string> _text = [];
    private readonly double _scale;

    public PdfWriterFont(IFontFace face, int number, int index)
    {
        (Face, Number, Index) = (face, number, index);
        int count = Math.Min(face.GlyphCount, 65536);
        _used = new bool[count];
        _widths = new double[count];
        Array.Fill(_widths, double.NaN);
        _scale = 1000.0 / Math.Max(1, face.Metrics.UnitsPerEm);
    }

    public IFontFace Face { get; }

    /// <summary>The Type0 font's object number (an editor gives it when it saves).</summary>
    public int Number { get; set; }

    /// <summary>Its resource name's number (<c>/F1</c>, <c>/F2</c>, ... in the order faces are first used).</summary>
    public int Index { get; }

    /// <summary>How many glyph ids it can show (two bytes a glyph).</summary>
    public int GlyphLimit => _used.Length;

    /// <summary>A glyph's width in thousandths of the size as <c>/W</c> holds it (three decimals), the value a reader reads.</summary>
    public double Width(int glyph)
    {
        double width = _widths[glyph];
        if (double.IsNaN(width))
        {
            width = _widths[glyph] = PdfSyntax.Written(Face.Advance(glyph) * _scale, 3);
        }

        return width;
    }

    /// <summary>
    /// Marks the run's glyphs used and gives each glyph not yet mapped the characters it stands for in
    /// <paramref name="text"/>; whether the glyphs, in their order, read back as <paramref name="text"/> through the map.
    /// </summary>
    public bool Map(ReadOnlySpan<DocumentGlyph> glyphs, string text)
    {
        bool known = true;
        foreach (var g in glyphs)
        {
            _used[g.Glyph] = true;
            known &= _text.ContainsKey(g.Glyph);
        }

        // The run's glyphs matched to its characters, for the glyphs whose text is not known yet (a glyph keeps the first
        // text it was given): a run of known glyphs alone changes nothing, so it is not matched.
        if (!known)
        {
            var assigned = Align(glyphs, text);
            for (int i = 0; i < glyphs.Length; i++)
            {
                if (assigned[i] is { Length: > 0 } characters && !_text.ContainsKey(glyphs[i].Glyph))
                {
                    _text[glyphs[i].Glyph] = characters;
                }
            }
        }

        // What a reader makes of the glyphs: each one's mapped text in turn (a glyph with none reads as something else).
        int at = 0;
        foreach (var g in glyphs)
        {
            if (!_text.TryGetValue(g.Glyph, out string? mapped) || string.CompareOrdinal(text, at, mapped, 0, mapped.Length) != 0)
            {
                return false;
            }

            at += mapped.Length;
        }

        return at == text.Length;
    }

    /// <summary>
    /// The characters each of a run's glyphs stands for in <paramref name="text"/> (null for none), as the run's alignment
    /// gives them (<see cref="Map"/> gives each glyph not yet mapped its share).
    /// </summary>
    public string?[] Shares(ReadOnlySpan<DocumentGlyph> glyphs, string text) => Align(glyphs, text);

    /// <summary>Whether a reader reads <paramref name="glyph"/> through the map as <paramref name="characters"/> (a glyph not mapped reads as nothing it can be given).</summary>
    public bool Reads(int glyph, string? characters) => _text.TryGetValue(glyph, out string? mapped) && string.Equals(mapped, characters, StringComparison.Ordinal);

    // The characters each glyph stands for (null for none).
    private string?[] Align(ReadOnlySpan<DocumentGlyph> glyphs, string text)
    {
        var characters = new List<(int Start, int Length, int Glyph)>();
        for (int i = 0; i < text.Length;)
        {
            int length = char.IsSurrogatePair(text, i) ? 2 : 1;
            int codePoint = length == 2 ? char.ConvertToUtf32(text[i], text[i + 1]) : text[i];
            characters.Add((i, length, Face.Glyph(codePoint)));
            i += length;
        }

        int n = glyphs.Length, m = characters.Count;
        var result = new string?[n];
        if (n == 0 || m == 0)
        {
            return result;
        }

        int forward = Anchors(glyphs, characters, reversed: false, null), backward = Anchors(glyphs, characters, reversed: true, null);
        bool reversed = backward > forward
                        || (backward == forward && forward == 0 && OrderNamed(ReadingOrders.Bidi).RightToLeft(text) == true);
        var anchors = new List<(int Glyph, int Character)>();
        Anchors(glyphs, characters, reversed, anchors);

        // Each glyph's characters, as indices into `characters` in the order chosen.
        var owned = new List<int>?[n];
        void Give(int glyph, int character) => (owned[glyph] ??= []).Add(character);

        anchors.Insert(0, (-1, -1));
        anchors.Add((n, m));
        for (int k = 0; k + 1 < anchors.Count; k++)
        {
            var (g0, c0) = anchors[k];
            var (g1, c1) = anchors[k + 1];
            if (g0 >= 0)
            {
                Give(g0, c0);
            }

            int freeGlyphs = g1 - g0 - 1, freeCharacters = c1 - c0 - 1;
            if (freeCharacters <= 0)
            {
                continue;                                                      // glyphs with no character of their own (none, or drawn parts)
            }

            if (freeGlyphs == 0)
            {
                // Characters with no glyph between: the glyph before stands for them too (or, at the start, the one after).
                int owner = g0 >= 0 ? g0 : g1 < n ? g1 : -1;
                for (int c = c0 + 1; c < c1 && owner >= 0; c++)
                {
                    Give(owner, c);
                }

                continue;
            }

            int shared = Math.Min(freeGlyphs, freeCharacters);
            for (int t = 0; t < freeCharacters; t++)
            {
                Give(g0 + 1 + Math.Min(t, shared - 1), c0 + 1 + t);
            }
        }

        for (int i = 0; i < n; i++)
        {
            if (owned[i] is not { } list)
            {
                continue;
            }

            var logical = list.Select(c => reversed ? m - 1 - c : c).Order().ToList();
            var builder = new StringBuilder();
            foreach (int c in logical)
            {
                builder.Append(text, characters[c].Start, characters[c].Length);
            }

            result[i] = builder.ToString();
        }

        return result;
    }

    // Glyphs matched in order to the characters the face maps to them, each looking a few characters ahead: how many.
    private static int Anchors(ReadOnlySpan<DocumentGlyph> glyphs, List<(int Start, int Length, int Glyph)> characters, bool reversed, List<(int, int)>? found)
    {
        int m = characters.Count, next = 0, matched = 0;
        for (int g = 0; g < glyphs.Length && next < m; g++)
        {
            for (int k = next; k < Math.Min(m, next + Window); k++)
            {
                int glyph = characters[reversed ? m - 1 - k : k].Glyph;
                if (glyph != 0 && glyph == glyphs[g].Glyph)
                {
                    found?.Add((g, k));
                    matched++;
                    next = k + 1;
                    break;
                }
            }
        }

        return matched;
    }

    /// <summary>
    /// Writes the font's objects: its program (a subset of the glyphs used, or the whole face), the descriptor, the
    /// CIDFont with its widths, the ToUnicode map and the Type0 font itself.
    /// </summary>
    public void Write(PdfFileWriter file, DocumentFontEmbedding embedding)
    {
        var used = new List<int>();
        for (int g = 0; g < _used.Length; g++)
        {
            if (_used[g])
            {
                used.Add(g);
            }
        }

        string baseName = PostScriptName();
        if (embedding == DocumentFontEmbedding.Subset)
        {
            baseName = Tag(baseName, used) + "+" + baseName;
        }

        int program = file.Allocate(), descriptor = file.Allocate(), cidFont = file.Allocate(), toUnicode = file.Allocate();
        using (var bytes = new PdfSyntax(64 * 1024))
        {
            var options = embedding == DocumentFontEmbedding.Whole
                ? new FontWriteOptions { KeepLayout = true, KeepHinting = true }
                : new FontWriteOptions { Glyphs = used, KeepLayout = false };
            using (var sink = new PdfBufferStream(bytes.Buffer))
            {
                WriterNamed(ProgramWriter).Write(Face, sink, options);
            }

            file.BeginStream(program).Raw("/Length1 "u8).Integer(bytes.Length);
            file.EndStream(bytes.Span, deflate: true);
        }

        WriteDescriptor(file, descriptor, program, baseName);

        var s = file.BeginSmall(cidFont);
        s.Raw("<</Type/Font/Subtype/CIDFontType2/BaseFont"u8).Name(baseName)
            .Raw("/CIDSystemInfo<</Registry"u8).Literal("Adobe"u8).Raw("/Ordering"u8).Literal("Identity"u8)   // strings: encrypted in an encrypted file
            .Raw("/Supplement 0>>/FontDescriptor "u8).Reference(descriptor).Raw("/W["u8);
        for (int i = 0; i < used.Count;)
        {
            int start = i;
            while (i + 1 < used.Count && used[i + 1] == used[i] + 1)
            {
                i++;
            }

            s.Integer(used[start]).Byte((byte)'[');
            for (int k = start; k <= i; k++)
            {
                if (k > start)
                {
                    s.Space();
                }

                s.Number(Width(used[k]), 3);
            }

            s.Byte((byte)']');
            i++;
        }

        s.Raw("]/CIDToGIDMap/Identity>>"u8);
        file.EndSmall();

        WriteToUnicode(file, toUnicode);

        s = file.BeginSmall(Number);
        s.Raw("<</Type/Font/Subtype/Type0/BaseFont"u8).Name(baseName).Raw("/Encoding/Identity-H/DescendantFonts["u8).Reference(cidFont)
            .Raw("]/ToUnicode "u8).Reference(toUnicode).Raw(">>"u8);
        file.EndSmall();
    }

    // The descriptor: the name, the bounding box and metrics in thousandths of the em (from the face's head, post and OS/2
    // tables where it has them), and the program.
    private void WriteDescriptor(PdfFileWriter file, int number, int program, string baseName)
    {
        var metrics = Face.Metrics;
        double ascent = metrics.Ascender * _scale, descent = metrics.Descender * _scale, capHeight = ascent, italic = 0;
        double left = 0, bottom = descent, right = 1000, top = ascent;
        var head = Face.Table("head").Span;
        if (head.Length >= 44)
        {
            (left, bottom) = (BinaryPrimitives.ReadInt16BigEndian(head[36..]) * _scale, BinaryPrimitives.ReadInt16BigEndian(head[38..]) * _scale);
            (right, top) = (BinaryPrimitives.ReadInt16BigEndian(head[40..]) * _scale, BinaryPrimitives.ReadInt16BigEndian(head[42..]) * _scale);
        }

        var post = Face.Table("post").Span;
        if (post.Length >= 8)
        {
            italic = BinaryPrimitives.ReadInt32BigEndian(post[4..]) / 65536.0;
        }

        var os2 = Face.Table("OS/2").Span;
        if (os2.Length >= 90 && BinaryPrimitives.ReadUInt16BigEndian(os2) >= 2 && BinaryPrimitives.ReadInt16BigEndian(os2[88..]) is > 0 and var cap)
        {
            capHeight = cap * _scale;
        }

        var s = file.BeginSmall(number);
        s.Raw("<</Type/FontDescriptor/FontName"u8).Name(baseName).Raw("/Flags 4/FontBBox["u8)
            .Number(left, 3).Space().Number(bottom, 3).Space().Number(right, 3).Space().Number(top, 3)
            .Raw("]/ItalicAngle "u8).Number(italic, 4).Raw("/Ascent "u8).Number(ascent, 3).Raw("/Descent "u8).Number(descent, 3)
            .Raw("/CapHeight "u8).Number(capHeight, 3).Raw("/StemV 80/FontFile2 "u8).Reference(program).Raw(">>"u8);
        file.EndSmall();
    }

    // The ToUnicode CMap: each mapped glyph (two bytes) to its characters (UTF-16BE), a hundred to a block.
    private void WriteToUnicode(PdfFileWriter file, int number)
    {
        using var cmap = new PdfSyntax(4096);
        cmap.Raw("/CIDInit /ProcSet findresource begin\n12 dict begin\nbegincmap\n/CIDSystemInfo <</Registry (Adobe) /Ordering (UCS) /Supplement 0>> def\n"u8)
            .Raw("/CMapName /Adobe-Identity-UCS def\n/CMapType 2 def\n1 begincodespacerange\n<0000> <FFFF>\nendcodespacerange\n"u8);
        var glyphs = _text.Keys.Order().ToList();
        for (int start = 0; start < glyphs.Count; start += 100)
        {
            int count = Math.Min(100, glyphs.Count - start);
            cmap.Integer(count).Raw(" beginbfchar\n"u8);
            for (int k = start; k < start + count; k++)
            {
                cmap.Byte((byte)'<').HexPair(glyphs[k]).Raw("> <"u8);
                foreach (char c in _text[glyphs[k]])
                {
                    cmap.HexPair(c);
                }

                cmap.Raw(">\n"u8);
            }

            cmap.Raw("endbfchar\n"u8);
        }

        cmap.Raw("endcmap\nCMapName currentdict /CMap defineresource pop\nend\nend\n"u8);
        file.BeginStream(number);
        file.EndStream(cmap.Span, deflate: true);
    }

    // The face's PostScript name (its name table's, else its family and style), as a name holds it: printable ASCII with
    // no spaces or delimiters.
    private string PostScriptName()
    {
        string? name = NameRecord(Face.Table("name").Span, 6);
        if (string.IsNullOrWhiteSpace(name))
        {
            name = string.Equals(Face.Style, "Regular", StringComparison.OrdinalIgnoreCase) || string.IsNullOrWhiteSpace(Face.Style)
                ? Face.Family
                : Face.Family + "-" + Face.Style;
        }

        var kept = new StringBuilder(name.Length);
        foreach (char c in name)
        {
            if (c is > ' ' and <= '~' and not ('(' or ')' or '<' or '>' or '[' or ']' or '{' or '}' or '/' or '%' or '#'))
            {
                kept.Append(c);
            }
        }

        return kept.Length == 0 ? "Font" + Index : kept.Length > 63 ? kept.ToString(0, 63) : kept.ToString();
    }

    // A name record's text (Windows Unicode first, then Macintosh Roman), or null.
    private static string? NameRecord(ReadOnlySpan<byte> table, int nameId)
    {
        if (table.Length < 6)
        {
            return null;
        }

        int count = BinaryPrimitives.ReadUInt16BigEndian(table[2..]), storage = BinaryPrimitives.ReadUInt16BigEndian(table[4..]);
        string? mac = null;
        for (int i = 0; i < count && 6 + 12 * (i + 1) <= table.Length; i++)
        {
            var record = table.Slice(6 + 12 * i, 12);
            int platform = BinaryPrimitives.ReadUInt16BigEndian(record), id = BinaryPrimitives.ReadUInt16BigEndian(record[6..]);
            int length = BinaryPrimitives.ReadUInt16BigEndian(record[8..]), offset = BinaryPrimitives.ReadUInt16BigEndian(record[10..]);
            if (id != nameId || storage + offset + length > table.Length)
            {
                continue;
            }

            var bytes = table.Slice(storage + offset, length);
            if (platform is 0 or 3)
            {
                return Encoding.BigEndianUnicode.GetString(bytes);
            }

            if (platform == 1)
            {
                mac ??= Encoding.Latin1.GetString(bytes);
            }
        }

        return mac;
    }

    // A subset's tag: six capital letters from a hash of the name and the glyphs kept (the same subset, the same tag).
    private static string Tag(string name, List<int> glyphs)
    {
        using var hash = IncrementalHash.CreateHash(HashAlgorithmName.SHA256);
        hash.AppendData(Encoding.UTF8.GetBytes(name));
        Span<byte> id = stackalloc byte[4];
        foreach (int g in glyphs)
        {
            BinaryPrimitives.WriteInt32BigEndian(id, g);
            hash.AppendData(id);
        }

        Span<byte> digest = stackalloc byte[32];
        hash.GetHashAndReset(digest);
        Span<char> tag = stackalloc char[6];
        for (int i = 0; i < 6; i++)
        {
            tag[i] = (char)('A' + digest[i] % 26);
        }

        return new string(tag);
    }
}
