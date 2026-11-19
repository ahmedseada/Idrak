// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

namespace Idrak.Data.Pdf;

/// <summary>
/// A font as a page's text uses it (ISO 32000-2, 9.5-9.10): how a string's bytes split into codes, each code's width, its
/// text and its glyph. A simple font (Type 1, TrueType, Type 3) has one byte a code, each code's text and width worked out
/// once when read; a composite font (Type0 over a CIDFont) has its encoding CMap (Identity-H, Identity-V or embedded)
/// split strings into codes and give CIDs, widths from <c>/W</c> and <c>/DW</c> (vertical: <c>/W2</c> and <c>/DW2</c>),
/// glyphs through <c>/CIDToGIDMap</c>. Text comes, in this order: the font's ToUnicode CMap; a simple font's encoding (its
/// base encoding and <c>/Differences</c>) with glyph names read by the Adobe Glyph List's rules; the embedded program's
/// own <c>cmap</c> (through <see cref="FontFormats"/>), a glyph's lowest code point; otherwise none (the reader puts U+FFFD
/// and counts it). Nothing here is for one language or script. Immutable once read, so one font serves every page and
/// thread.
/// </summary>
internal sealed class PdfFont
{
    private readonly Lazy<PdfFontProgram?> _program;

    // A simple font's codes: a text of one character in _single (0: none), a longer one in _several.
    private readonly char[]? _single;
    private readonly string?[]? _several;
    private readonly double[]? _widths;
    private readonly int[]? _nameCodePoints;
    private readonly bool _programGlyphs;
    private readonly bool _symbolic;
    private readonly string?[]? _glyphNames;   // a font drawn by its glyphs' names (Type 1, a bare CFF, Type 3): each code's name
    private readonly bool _builtIn;            // a bare CFF with no /Encoding: its program's own encoding first

    // A composite font's codes.
    private readonly PdfCMap? _encoding;
    private readonly PdfCMap? _toUnicode;
    private readonly PdfCidMetrics? _cidWidths;
    private readonly PdfCidMetrics? _verticalMetrics;
    private readonly double _defaultVertical;
    private readonly byte[]? _cidToGid;

    /// <summary>A simple font (<paramref name="programGlyphs"/>: whether codes reach glyphs through the embedded program's cmap, as TrueType and OpenType programs' do).</summary>
    public PdfFont(string? name, char[] single, string?[]? several, double[] widths, int[] nameCodePoints, bool programGlyphs, bool symbolic, double ascent, double descent,
        Func<PdfFontProgram?>? program, string?[]? glyphNames = null, bool builtIn = false)
    {
        Name = name;
        _program = new Lazy<PdfFontProgram?>(program ?? (() => null), LazyThreadSafetyMode.ExecutionAndPublication);
        (_single, _several, _widths, _nameCodePoints, _programGlyphs, _symbolic) = (single, several, widths, nameCodePoints, programGlyphs, symbolic);
        (_glyphNames, _builtIn) = (glyphNames, builtIn);
        (Ascent, Descent) = (ascent, descent);
        Bytes = 4096 + 32L * (several?.Count(t => t is not null) ?? 0);
    }

    /// <summary>A composite font.</summary>
    public PdfFont(string? name, PdfCMap encoding, PdfCMap? toUnicode, PdfCidMetrics widths, PdfCidMetrics? vertical, double defaultVertical, byte[]? cidToGid,
        double ascent, double descent, Func<PdfFontProgram?>? program, bool cidKeyed = false)
    {
        Name = name;
        _cidKeyed = cidKeyed;
        _program = new Lazy<PdfFontProgram?>(program ?? (() => null), LazyThreadSafetyMode.ExecutionAndPublication);
        (_encoding, _toUnicode, _cidWidths, _verticalMetrics, _defaultVertical, _cidToGid) = (encoding, toUnicode, widths, vertical, defaultVertical, cidToGid);
        (Ascent, Descent, Vertical) = (ascent, descent, encoding.Vertical);
        Bytes = 4096 + encoding.Bytes + (toUnicode?.Bytes ?? 0) + widths.Bytes + (cidToGid?.Length ?? 0);
    }

    /// <summary>The font's name as the file gives it (its <c>/BaseFont</c>, the descendant's for a composite font), or null.</summary>
    public string? Name { get; }

    /// <summary>Whether its text is written top to bottom.</summary>
    public bool Vertical { get; }

    /// <summary>A vertical font's origin above a glyph's horizontal one, in text space units at size 1 (/DW2's first number; 0.88).</summary>
    public double VerticalOrigin { get; init; } = 0.88;

    /// <summary>How far its glyphs reach above the baseline, in text space units at size 1.</summary>
    public double Ascent { get; }

    /// <summary>How far they reach below (negative).</summary>
    public double Descent { get; }

    /// <summary>About how many bytes it holds (for the document's cache).</summary>
    public long Bytes { get; }

    /// <summary>What of the font is not read (named for the page's issues), or null.</summary>
    public string? Note { get; init; }

    /// <summary>The embedded program as a face to draw with, or null.</summary>
    public IFontFace? Face => _program.Value?.Face;

    /// <summary>Whether it is a composite font (codes through a CMap), not a simple font of one byte a code.</summary>
    public bool Composite => _encoding is not null;

    /// <summary>The code at <paramref name="at"/> of <paramref name="text"/> and its length in bytes.</summary>
    public int NextCode(ReadOnlySpan<byte> text, int at, out uint code)
    {
        if (_encoding is { } encoding)
        {
            return encoding.NextCode(text, at, out code);
        }

        code = text[at];
        return 1;
    }

    /// <summary>The code's horizontal advance, in text space units at size 1.</summary>
    public double Width(uint code, int length) => _encoding is { } encoding ? _cidWidths!.Get(encoding.Cid(code, length)) / 1000.0 : _widths![code & 0xFF];

    /// <summary>The code's vertical advance (negative: down), in text space units at size 1; only for a vertical font.</summary>
    public double VerticalAdvance(uint code, int length)
    {
        double advance = _encoding is { } encoding ? _verticalMetrics?.GetOrNaN(encoding.Cid(code, length)) ?? double.NaN : double.NaN;
        return (double.IsNaN(advance) ? _defaultVertical : advance) / 1000.0;
    }

    /// <summary>Whether word spacing applies to the code (the single-byte code 32).</summary>
    public bool IsWordSpace(uint code, int length) => code == 32 && (length == 1 || _encoding is null);

    /// <summary>The code's text written into <paramref name="destination"/>: the count written, 0 when it has none.</summary>
    public int Text(uint code, int length, Span<char> destination)
    {
        if (_encoding is { } encoding)
        {
            if (_toUnicode is { } map && map.Unicode(code, length, destination) is > 0 and var count)
            {
                return count;
            }

            return ProgramText(CidGlyph(encoding.Cid(code, length)), destination);
        }

        char single = _single![code & 0xFF];
        if (single != 0)
        {
            destination[0] = single;
            return 1;
        }

        if (_several?[code & 0xFF] is { } text)
        {
            int count = Math.Min(text.Length, destination.Length);
            text.AsSpan(0, count).CopyTo(destination);
            return count;
        }

        // The third way: the code's glyph in the embedded program, and that glyph's code point by the program's cmap.
        return _programGlyphs && _program.Value is { } program ? ProgramText(SimpleGlyph(program, (int)(code & 0xFF)), destination) : 0;
    }

    /// <summary>
    /// The code's glyph in <see cref="Face"/>, or the code itself for a simple font with no program read. A font drawn by
    /// its glyphs' names takes the code's name in the program (a bare CFF with no /Encoding its own encoding first, and
    /// any such font its own encoding when the name is not there).
    /// </summary>
    public int GlyphId(uint code, int length)
    {
        if (_encoding is { } encoding)
        {
            return CidGlyph(encoding.Cid(code, length));
        }

        if (_glyphNames is { } names && _program.Value is { } named)
        {
            int c = (int)(code & 0xFF);
            if (_builtIn && named.SymbolGlyph(c) is > 0 and var own)
            {
                return own;
            }

            if (names[c] is { } glyphName && named.GlyphByName(glyphName) is > 0 and var byName)
            {
                return byName;
            }

            return Math.Max(0, named.SymbolGlyph(c));
        }

        return _programGlyphs && _program.Value is { } program ? Math.Max(0, SimpleGlyph(program, (int)(code & 0xFF))) : (int)code;
    }

    // The text a program's glyph stands for, by its own cmap (the third way).
    private int ProgramText(int glyph, Span<char> destination)
    {
        int codePoint = glyph > 0 && _program.Value is { } program ? program.CodePoint(glyph) : -1;
        if (codePoint < 0 || codePoint is >= 0xD800 and <= 0xDFFF || destination.Length < 2)
        {
            return 0;
        }

        return new System.Text.Rune(codePoint).EncodeToUtf16(destination);
    }

    // A simple TrueType font's code to its glyph (9.6.5.4): by the glyph name's character through the Unicode cmap when
    // the font is not symbolic, else (or failing that) through the (3,0) and (1,0) subtables.
    private int SimpleGlyph(PdfFontProgram program, int code)
    {
        if (!_symbolic && _nameCodePoints![code] >= 0 && program.Face.Glyph(_nameCodePoints[code]) is > 0 and var glyph)
        {
            return glyph;
        }

        return program.SymbolGlyph(code);
    }

    // A CID's glyph: through the CIDToGIDMap stream (two bytes a CID), a CID-keyed CFF program's charset (its glyphs
    // named by their CIDs), or the CID itself.
    private int CidGlyph(int cid)
    {
        if (_cidToGid is { } map)
        {
            long at = 2L * cid;
            return at >= 0 && at + 1 < map.Length ? map[at] << 8 | map[at + 1] : 0;
        }

        if (_cidKeyed && _program.Value is { } program && program.CidGlyph(cid) is >= 0 and var glyph)
        {
            return glyph;
        }

        return cid;
    }

    // A composite font whose program may be a CID-keyed CFF (FontFile3).
    private readonly bool _cidKeyed;
}

/// <summary>A CIDFont's per-CID metrics (<c>/W</c>: one value a CID; <c>/W2</c>: the vertical advance), as sorted ranges with a default.</summary>
internal sealed class PdfCidMetrics
{
    private readonly int[] _starts;
    private readonly int[] _ends;
    private readonly double[] _values;
    private readonly double _default;

    public PdfCidMetrics(List<(int Start, int End, double Value)> ranges, double defaultValue)
    {
        ranges.Sort((a, b) => a.Start.CompareTo(b.Start));
        _starts = [.. ranges.Select(r => r.Start)];
        _ends = [.. ranges.Select(r => r.End)];
        _values = [.. ranges.Select(r => r.Value)];
        _default = defaultValue;
    }

    public long Bytes => 16L * _starts.Length;

    public double Get(int cid)
    {
        double value = GetOrNaN(cid);
        return double.IsNaN(value) ? _default : value;
    }

    public double GetOrNaN(int cid)
    {
        int lo = 0, hi = _starts.Length - 1, found = -1;
        while (lo <= hi)
        {
            int mid = (lo + hi) >>> 1;
            if (_starts[mid] <= cid)
            {
                found = mid;
                lo = mid + 1;
            }
            else
            {
                hi = mid - 1;
            }
        }

        // The range starting last at or before the CID, or one a few before it that reaches over it.
        for (int i = found; i >= 0 && i >= found - 8; i--)
        {
            if (cid <= _ends[i])
            {
                return _values[i];
            }
        }

        return double.NaN;
    }
}

/// <summary>Reads a font dictionary into a <see cref="PdfFont"/>.</summary>
internal static class PdfFonts
{
    /// <summary>The font used where a page names none it has, or one that cannot be read: Helvetica, the standard encoding.</summary>
    public static PdfFont Fallback { get; } = Standard("Helvetica");

    /// <summary>A standard font by its name, with its own encoding.</summary>
    public static PdfFont Standard(string name)
    {
        int standard = PdfFontTables.StandardFontIndex(name);
        int encoding = standard == PdfFontTables.Symbol ? PdfFontTables.SymbolEncoding : standard == PdfFontTables.ZapfDingbats ? PdfFontTables.ZapfDingbatsEncoding : PdfFontTables.StandardEncoding;
        var (ascent, descent) = PdfFontTables.StandardExtents(standard);
        return Simple(name, new byte[]?[256], encoding, null, null, null, standard, standard == PdfFontTables.ZapfDingbats, false, false, 1 / 1000.0, ascent / 1000.0, descent / 1000.0, null);
    }

    /// <summary>
    /// The font <paramref name="dictionary"/> describes. A part that cannot be read (a damaged ToUnicode, a width
    /// table) is left out; a limit passed fails the page.
    /// </summary>
    public static PdfFont Read(PdfFile file, PdfDictionary dictionary)
    {
        var subtype = file.Resolve(dictionary["Subtype"u8]).AsName;
        if (subtype is not null && subtype.Is("Type0"u8))
        {
            return Composite(file, dictionary);
        }

        bool type3 = subtype is not null && subtype.Is("Type3"u8);
        bool trueType = subtype is not null && subtype.Is("TrueType"u8);
        string? baseFont = file.Resolve(dictionary["BaseFont"u8]).AsName?.ToString();
        var descriptor = file.Dictionary(dictionary["FontDescriptor"u8]);
        var fontFile = descriptor is null ? null : file.Resolve(descriptor["FontFile"u8]).AsStream;
        var fontFile2 = descriptor is null ? null : file.Resolve(descriptor["FontFile2"u8]).AsStream;
        var fontFile3 = descriptor is null ? null : file.Resolve(descriptor["FontFile3"u8]).AsStream;
        bool embedded = fontFile is not null || fontFile2 is not null || fontFile3 is not null;
        int standard = type3 ? -1 : PdfFontTables.StandardFontIndex(baseFont);
        long flags = descriptor is not null && file.Resolve(descriptor["Flags"u8]).TryInteger(out long f) ? f : 0;
        bool symbolic = (flags & 4) != 0 && (flags & 32) == 0;
        bool zapf = standard == PdfFontTables.ZapfDingbats;

        // The base encoding and the differences.
        var encoding = file.Resolve(dictionary["Encoding"u8]);
        int baseEncoding = -1;
        PdfArray? differences = null;
        if (encoding.AsName is { } named)
        {
            baseEncoding = PdfFontTables.EncodingIndex(named.Bytes);
        }
        else if (encoding.AsDictionary is { } encodingDictionary)
        {
            if (file.Resolve(encodingDictionary["BaseEncoding"u8]).AsName is { } baseName)
            {
                baseEncoding = PdfFontTables.EncodingIndex(baseName.Bytes);
            }

            differences = file.Resolve(encodingDictionary["Differences"u8]).AsArray;
        }

        var names = new byte[]?[256];
        if (baseEncoding < 0 && !type3)
        {
            byte[]?[]? builtIn = null;
            if (standard == PdfFontTables.Symbol && !embedded)
            {
                baseEncoding = PdfFontTables.SymbolEncoding;
            }
            else if (zapf && !embedded)
            {
                baseEncoding = PdfFontTables.ZapfDingbatsEncoding;
            }
            else if (fontFile is not null)
            {
                builtIn = Type1Encoding(file, fontFile);   // a Type 1 program's own encoding
            }

            if (builtIn is not null)
            {
                builtIn.CopyTo(names, 0);
            }
            else if (baseEncoding < 0 && !(trueType && symbolic))
            {
                baseEncoding = PdfFontTables.StandardEncoding;
            }
        }

        if (differences is not null)
        {
            int code = -1;
            foreach (var item in differences.Items)
            {
                var value = file.Resolve(item);
                if (value.TryInteger(out long at))
                {
                    code = (int)Math.Clamp(at, -1, 256);
                }
                else if (value.AsName is { } glyph && code is >= 0 and < 256)
                {
                    names[code++] = glyph.Bytes.ToArray();
                }
            }
        }

        // Widths: /Widths from /FirstChar; past it, the standard font's own, else /MissingWidth. Type 3 through its matrix.
        double scale = 1 / 1000.0, ascentScale = 1 / 1000.0;
        double ascent = double.NaN, descent = double.NaN;
        if (type3)
        {
            var matrix = file.Resolve(dictionary["FontMatrix"u8]).AsArray;
            scale = Number(file, matrix, 0, 0.001);
            ascentScale = Number(file, matrix, 3, 0.001);
            if (file.Resolve(dictionary["FontBBox"u8]).AsArray is { Count: >= 4 } box)
            {
                (ascent, descent) = (Number(file, box, 3, 800) * ascentScale, Number(file, box, 1, -200) * ascentScale);
            }
        }

        if (double.IsNaN(ascent))
        {
            (ascent, descent) = Extents(file, descriptor, standard);
        }

        var toUnicode = ToUnicode(file, dictionary);
        bool programGlyphs = (trueType && fontFile2 is not null) || fontFile3 is not null;
        var program = programGlyphs ? ProgramReader(file, fontFile2 ?? fontFile3, baseFont) : null;
        // Fonts drawn by their glyphs' names: a Type 1 program (read for drawing only: its text comes from its encoding),
        // a bare CFF (FontFile3 that is not OpenType), Type 3's own glyphs.
        var subtype3 = fontFile3 is null ? null : file.Resolve(fontFile3.Dictionary["Subtype"u8]).AsName;
        bool bareCff = fontFile3 is not null && (subtype3 is null || !subtype3.Is("OpenType"u8));
        bool byName = !trueType && (fontFile is not null || (fontFile3 is not null && bareCff) || type3);
        if (type3)
        {
            program = () => PdfType3Face.Read(file, dictionary, baseFont) is { } face ? PdfFontProgram.Of(face, face.Names) : null;
        }
        else if (fontFile is not null && program is null)
        {
            program = ProgramReader(file, fontFile, baseFont);
        }

        var widths = file.Resolve(dictionary["Widths"u8]).AsArray;
        long firstChar = file.Resolve(dictionary["FirstChar"u8]).TryInteger(out long first) ? first : 0;
        double missing = descriptor is not null && file.Resolve(descriptor["MissingWidth"u8]).TryNumber(out double m) ? m : 0;
        return Simple(baseFont, names, baseEncoding, toUnicode, widths?.Items.Select(file.Resolve).ToArray(), (int)Math.Clamp(firstChar, -256, 256), standard, zapf, programGlyphs, symbolic, scale,
            ascent, descent, program, missing, byName, builtIn: byName && fontFile3 is not null && encoding.IsNull);
    }

    // A simple font's codes: each one's glyph name (an override of `names`, a Differences entry or the program's own
    // encoding, else the base encoding's), its text (the ToUnicode map's, else the name's) and its width.
    private static PdfFont Simple(string? name, byte[]?[] names, int baseEncoding, PdfCMap? toUnicode, PdfValue[]? widths, int? firstChar, int standard, bool zapf, bool programGlyphs,
        bool symbolic, double scale, double ascent, double descent, Func<PdfFontProgram?>? program, double missing = 0, bool byName = false, bool builtIn = false)
    {
        string?[]? glyphNames = byName ? new string?[256] : null;
        var single = new char[256];
        string?[]? several = null;
        var advance = new double[256];
        var nameCodePoints = new int[256];
        Span<char> chars = stackalloc char[PdfFontTables.MaxGlyphText];
        Span<char> mappedChars = stackalloc char[PdfFontTables.MaxGlyphText];
        for (int code = 0; code < 256; code++)
        {
            ReadOnlySpan<byte> glyph = names[code] is { } own ? own : baseEncoding >= 0 ? PdfFontTables.EncodingName(baseEncoding, code) : default;
            if (glyphNames is not null && !glyph.IsEmpty)
            {
                glyphNames[code] = System.Text.Encoding.Latin1.GetString(glyph);
            }

            int named = glyph.IsEmpty ? 0 : PdfFontTables.GlyphText(glyph, zapf, chars);
            nameCodePoints[code] = named switch
            {
                1 when !char.IsSurrogate(chars[0]) => chars[0],
                2 when char.IsSurrogatePair(chars[0], chars[1]) => char.ConvertToUtf32(chars[0], chars[1]),
                _ => -1,
            };

            ReadOnlySpan<char> text = chars[..named];
            if (toUnicode is not null)
            {
                int mapped = toUnicode.Unicode((uint)code, 1, mappedChars);
                if (mapped == 0)
                {
                    mapped = toUnicode.Unicode((uint)code, 2, mappedChars);   // some writers give a simple font's codes two bytes
                }

                if (mapped > 0)
                {
                    text = mappedChars[..mapped];
                }
            }

            if (text.Length == 1 && text[0] != 0)
            {
                single[code] = text[0];
            }
            else if (text.Length > 1)
            {
                (several ??= new string?[256])[code] = new string(text);
            }

            double width = double.NaN;
            int index = code - (firstChar ?? 0);
            if (widths is not null && index >= 0 && index < widths.Length && widths[index].TryNumber(out double w))
            {
                width = w;
            }
            else if (standard >= 0 && nameCodePoints[code] >= 0)
            {
                int afm = PdfFontTables.StandardWidth(standard, nameCodePoints[code]);
                width = afm >= 0 ? afm : double.NaN;
            }

            advance[code] = (double.IsNaN(width) ? missing : width) * scale;
        }

        return new PdfFont(name, single, several, advance, nameCodePoints, programGlyphs, symbolic, ascent, descent, program, glyphNames, builtIn);
    }

    // The font's ascent and descent (thousandths): its descriptor's, its bounding box's, the standard font's, or a default.
    private static (double Ascent, double Descent) Extents(PdfFile file, PdfDictionary? descriptor, int standard)
    {
        if (descriptor is not null)
        {
            bool hasAscent = file.Resolve(descriptor["Ascent"u8]).TryNumber(out double ascent), hasDescent = file.Resolve(descriptor["Descent"u8]).TryNumber(out double descent);
            if (hasAscent && hasDescent && (ascent != 0 || descent != 0) && double.IsFinite(ascent) && double.IsFinite(descent))
            {
                return (ascent / 1000, descent / 1000);
            }

            if (file.Resolve(descriptor["FontBBox"u8]).AsArray is { Count: >= 4 } box && Number(file, box, 3, 0) > Number(file, box, 1, 0))
            {
                return (Number(file, box, 3, 0) / 1000, Number(file, box, 1, 0) / 1000);
            }
        }

        if (standard >= 0)
        {
            var (a, d) = PdfFontTables.StandardExtents(standard);
            return (a / 1000.0, d / 1000.0);
        }

        return (0.75, -0.25);
    }

    private static double Number(PdfFile file, PdfArray? array, int index, double otherwise) =>
        array is not null && index < array.Count && file.Resolve(array[index]).TryNumber(out double value) && double.IsFinite(value) ? value : otherwise;

    private static PdfFont Composite(PdfFile file, PdfDictionary dictionary)
    {
        var descendant = file.Resolve(dictionary["DescendantFonts"u8]).AsArray is { Count: > 0 } descendants ? file.Dictionary(descendants[0]) : null;
        descendant ??= PdfDictionary.Empty;
        string? name = file.Resolve(descendant["BaseFont"u8]).AsName?.ToString() ?? file.Resolve(dictionary["BaseFont"u8]).AsName?.ToString();
        var toUnicode = ToUnicode(file, dictionary);

        // The encoding: Identity-H or -V, or an embedded CMap (with the one it uses); any other predefined CMap is not built in.
        PdfCMap encoding;
        string? note = null;
        var value = file.Resolve(dictionary["Encoding"u8]);
        if (value.AsName is { } predefined)
        {
            encoding = predefined.Is("Identity-V"u8) ? PdfCMap.Identity(vertical: true) : PdfCMap.Identity(vertical: false);
            if (!predefined.Is("Identity-H"u8) && !predefined.Is("Identity-V"u8))
            {
                note = $"the predefined CMap {predefined} is not built in (each is tied to one script): codes are read as two bytes, their text from the ToUnicode map alone";
            }
        }
        else if (value.AsStream is { } stream)
        {
            encoding = EmbeddedCMap(file, stream, 0);
        }
        else
        {
            encoding = PdfCMap.Identity(vertical: false);
        }

        var descriptor = file.Dictionary(descendant["FontDescriptor"u8]);
        double defaultWidth = file.Resolve(descendant["DW"u8]).TryNumber(out double dw) ? dw : 1000;
        var widths = new PdfCidMetrics(CidRanges(file, file.Resolve(descendant["W"u8]).AsArray, 1), defaultWidth);
        PdfCidMetrics? vertical = null;
        double defaultVertical = -1000, verticalOrigin = 880;
        if (encoding.Vertical)
        {
            if (file.Resolve(descendant["DW2"u8]).AsArray is { Count: >= 2 } dw2)
            {
                defaultVertical = Number(file, dw2, 1, -1000);
                verticalOrigin = Number(file, dw2, 0, 880);
            }

            vertical = new PdfCidMetrics(CidRanges(file, file.Resolve(descendant["W2"u8]).AsArray, 3), defaultVertical);
        }

        byte[]? cidToGid = null;
        if (file.Resolve(descendant["CIDToGIDMap"u8]).AsStream is { } map)
        {
            using var decoded = file.Decode(map);
            cidToGid = decoded.Span.ToArray();
        }

        var fontFile = descriptor is null ? null : file.Resolve(descriptor["FontFile2"u8]).AsStream ?? file.Resolve(descriptor["FontFile3"u8]).AsStream;
        var (ascent, descent) = Extents(file, descriptor, -1);
        bool cidKeyed = file.Resolve(descendant["Subtype"u8]).AsName is { } cidType && cidType.Is("CIDFontType0"u8);
        return new PdfFont(name, encoding, toUnicode, widths, vertical, defaultVertical, cidToGid, ascent, descent,
            fontFile is null ? null : ProgramReader(file, fontFile, name), cidKeyed) { Note = note, VerticalOrigin = verticalOrigin / 1000 };
    }

    // /W ([c [w1 w2 ...]] or [cfirst clast w]) or /W2 ([c [w1y vx vy ...]] or [cfirst clast w1y vx vy]): the first of each
    // group of `values` numbers, by CID.
    private static List<(int Start, int End, double Value)> CidRanges(PdfFile file, PdfArray? array, int values)
    {
        var ranges = new List<(int, int, double)>();
        if (array is null)
        {
            return ranges;
        }

        var items = array.Items;
        int i = 0;
        while (i < items.Length)
        {
            if (!file.Resolve(items[i]).TryInteger(out long start) || i + 1 >= items.Length)
            {
                break;
            }

            var next = file.Resolve(items[i + 1]);
            if (next.AsArray is { } list)
            {
                for (int k = 0; k + values <= list.Count; k += values)
                {
                    if (file.Resolve(list[k]).TryNumber(out double w))
                    {
                        int cid = (int)Math.Clamp(start + k / values, int.MinValue, int.MaxValue);
                        ranges.Add((cid, cid, w));
                    }
                }

                i += 2;
            }
            else if (next.TryInteger(out long end) && i + 2 < items.Length && file.Resolve(items[i + 2]).TryNumber(out double w))
            {
                ranges.Add(((int)Math.Clamp(start, int.MinValue, int.MaxValue), (int)Math.Clamp(end, int.MinValue, int.MaxValue), w));
                i += 2 + values;
            }
            else
            {
                break;
            }
        }

        return ranges;
    }

    private static PdfCMap EmbeddedCMap(PdfFile file, PdfStream stream, int depth)
    {
        if (depth > file.MaxDepth)
        {
            throw PdfLimit.Error($"CMaps that use CMaps deeper than {file.MaxDepth} levels, the limit {nameof(DocumentOptions.MaxDepth)} (DocumentOptions.MaxDepth)");
        }

        PdfCMap? parent = null;
        var used = file.Resolve(stream.Dictionary["UseCMap"u8]);
        if (used.AsStream is { } usedStream)
        {
            parent = EmbeddedCMap(file, usedStream, depth + 1);
        }
        else if (used.AsName is { } usedName && (usedName.Is("Identity-H"u8) || usedName.Is("Identity-V"u8)))
        {
            parent = PdfCMap.Identity(usedName.Is("Identity-V"u8));
        }

        bool vertical = file.Resolve(stream.Dictionary["WMode"u8]).TryInteger(out long mode) && mode == 1;
        using var decoded = file.Decode(stream);
        return PdfCMap.Parse(decoded.Span, parent, vertical);
    }

    private static PdfCMap? ToUnicode(PdfFile file, PdfDictionary dictionary)
    {
        var value = file.Resolve(dictionary["ToUnicode"u8]);
        if (value.AsStream is not { } stream)
        {
            return null;   // none, or a name (Identity-H and the like say nothing of text)
        }

        try
        {
            using var decoded = file.Decode(stream);
            var map = PdfCMap.Parse(decoded.Span, null, vertical: false);
            return map.HasUnicode ? map : null;
        }
        catch (InvalidDataException e) when (!PdfLimit.Is(e))
        {
            return null;   // a damaged ToUnicode: the other ways give the text
        }
    }

    // Reads the embedded TrueType or OpenType program when first needed.
    private static Func<PdfFontProgram?>? ProgramReader(PdfFile file, PdfStream? stream, string? name)
    {
        if (stream is null)
        {
            return null;
        }

        return () =>
        {
            try
            {
                using var decoded = file.Decode(stream);
                return PdfFontProgram.Read(decoded.Span.ToArray(), name ?? $"object {stream.Number}");
            }
            catch (InvalidDataException)
            {
                return null;   // a program that cannot be decoded: no third way
            }
        };
    }

    // A Type 1 program's own encoding, from its clear text (before eexec): "/Encoding StandardEncoding def" (null: the
    // standard one) or "dup CODE /NAME put" entries.
    private static byte[]?[]? Type1Encoding(PdfFile file, PdfStream stream)
    {
        try
        {
            using var decoded = file.Decode(stream);
            var program = decoded.Span;
            long clear = file.Resolve(stream.Dictionary["Length1"u8]).TryInteger(out long length1) && length1 > 0 ? Math.Min(length1, program.Length) : program.Length;
            var text = program[..(int)clear];
            int eexec = text.IndexOf("eexec"u8);
            if (eexec >= 0)
            {
                text = text[..eexec];
            }

            int at = text.IndexOf("/Encoding"u8);
            if (at < 0)
            {
                return null;
            }

            var lexer = new PdfContentLexer(text[(at + 9)..]);
            var data = lexer.Data;
            var token = lexer.Next(out int start, out int length, out _);
            if (token != PdfToken.Number)
            {
                return null;   // StandardEncoding, or something else: the standard encoding
            }

            var names = new byte[]?[256];
            int code = -1, step = 0;
            while ((token = lexer.Next(out start, out length, out double number)) != PdfToken.End)
            {
                var word = data.Slice(start, length);
                if (token == PdfToken.Keyword && (word.SequenceEqual("def"u8) || word.SequenceEqual("readonly"u8)))
                {
                    break;
                }

                (step, code) = (step, token) switch
                {
                    (0, PdfToken.Keyword) when word.SequenceEqual("dup"u8) => (1, -1),
                    (1, PdfToken.Number) => (2, number is >= 0 and < 256 ? (int)number : -1),
                    (2, PdfToken.Name) when code >= 0 => (3, code),
                    _ => (0, -1),
                };
                if (step == 3)
                {
                    names[code] = word.ToArray();
                    step = 0;
                }
            }

            return names;
        }
        catch (InvalidDataException e) when (!PdfLimit.Is(e))
        {
            return null;
        }
    }
}
