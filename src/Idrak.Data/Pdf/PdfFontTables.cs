// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Text;

namespace Idrak.Data.Pdf;

/// <summary>
/// What simple fonts' codes stand for, searched in place in the generated tables (PdfFontTables.Data.cs, made by
/// tools/documents/make_pdf_tables.py; nothing is built when the reader starts): the base encodings (Standard, MacRoman,
/// WinAnsi, PDFDoc) and the Symbol and ZapfDingbats fonts' built-in ones, codes to glyph names; glyph names to text by the
/// Adobe Glyph List's rules (a suffix after a period dropped, components joined by underscores, the list itself, then
/// <c>uniXXXX</c> and <c>uXXXX[XX]</c>, the ITC Zapf Dingbats list first for that font); the standard 14 fonts'
/// ascender, descender, bounding box and widths, by name or a common alternative name; and the 258 standard Macintosh
/// glyph names a TrueType font's <c>post</c> table names its glyphs by.
/// </summary>
internal static partial class PdfFontTables
{
    /// <summary>The rows of the encoding table.</summary>
    public const int StandardEncoding = 0, MacRomanEncoding = 1, WinAnsiEncoding = 2, PdfDocEncoding = 3, SymbolEncoding = 4, ZapfDingbatsEncoding = 5;

    /// <summary>The standard font Symbol's index in <see cref="StandardFonts"/>.</summary>
    public const int Symbol = 8;

    /// <summary>The standard font ZapfDingbats' index in <see cref="StandardFonts"/>.</summary>
    public const int ZapfDingbats = 13;

    /// <summary>The longest text one glyph name gives here (a name of many components is cut there).</summary>
    public const int MaxGlyphText = 64;

    /// <summary>How many standard Macintosh glyph names there are: a TrueType <c>post</c> table's index below it is one of them.</summary>
    public const int MacGlyphCount = 258;

    /// <summary>
    /// Standard Macintosh glyph name <paramref name="index"/> (0 to <see cref="MacGlyphCount"/> - 1, a TrueType
    /// <c>post</c> table's order: what format 1 names each glyph, and format 2 an index below 258), as bytes.
    /// </summary>
    public static ReadOnlySpan<byte> MacGlyphName(int index)
    {
        var ends = MacGlyphNameEnds;
        return MacGlyphNames[(index == 0 ? 0 : ends[index - 1])..ends[index]];
    }

    /// <summary>The base encoding a font's <c>/Encoding</c> (or <c>/BaseEncoding</c>) names, or -1 for another name.</summary>
    public static int EncodingIndex(ReadOnlySpan<byte> name) =>
        name.SequenceEqual("WinAnsiEncoding"u8) ? WinAnsiEncoding
        : name.SequenceEqual("StandardEncoding"u8) ? StandardEncoding
        : name.SequenceEqual("MacRomanEncoding"u8) ? MacRomanEncoding
        : name.SequenceEqual("PDFDocEncoding"u8) ? PdfDocEncoding
        : -1;

    /// <summary>The glyph name <paramref name="code"/> has in the encoding <paramref name="encoding"/>, or empty.</summary>
    public static ReadOnlySpan<byte> EncodingName(int encoding, int code)
    {
        int index = EncodingCodes[encoding * 256 + (code & 0xFF)];
        if (index < 0)
        {
            return default;
        }

        var ends = EncodingNameEnds;
        return EncodingNames[(index == 0 ? 0 : ends[index - 1])..ends[index]];
    }

    /// <summary>
    /// The text of the glyph <paramref name="name"/> by the Adobe Glyph List's rules, written into
    /// <paramref name="destination"/> (at least <see cref="MaxGlyphText"/> characters); the count written, 0 when the
    /// rules give none (<c>.notdef</c>, <c>g17</c>).
    /// </summary>
    public static int GlyphText(ReadOnlySpan<byte> name, bool zapfDingbats, Span<char> destination)
    {
        int dot = name.IndexOf((byte)'.');
        if (dot >= 0)
        {
            name = name[..dot];
        }

        int written = 0;
        while (true)
        {
            int underscore = name.IndexOf((byte)'_');
            var component = underscore < 0 ? name : name[..underscore];
            written += Component(component, zapfDingbats, destination[written..]);
            if (underscore < 0 || destination.Length - written < 2)
            {
                return written;
            }

            name = name[(underscore + 1)..];
        }
    }

    // One component of a glyph name: the Zapf Dingbats list (that font only), the glyph list, uniXXXX..., uXXXX[XX].
    private static int Component(ReadOnlySpan<byte> component, bool zapfDingbats, Span<char> destination)
    {
        if (component.IsEmpty || destination.Length < 2)
        {
            return 0;
        }

        if (zapfDingbats && component.Length >= 2 && component[0] == (byte)'a' && TryDecimal(component[1..], out int number)
            && number < ZapfText.Length && ZapfText[number] != 0)
        {
            destination[0] = (char)ZapfText[number];
            return 1;
        }

        int index = AglIndex(component);
        if (index >= 0)
        {
            var ends = AglTextEnds;
            var text = AglText[(index == 0 ? 0 : ends[index - 1])..ends[index]];
            int count = Math.Min(text.Length, destination.Length);
            for (int i = 0; i < count; i++)
            {
                destination[i] = (char)text[i];
            }

            return count;
        }

        if (component.Length > 3 && component.StartsWith("uni"u8) && (component.Length - 3) % 4 == 0)
        {
            var digits = component[3..];
            int count = digits.Length / 4;
            if (count > destination.Length)
            {
                return 0;
            }

            for (int i = 0; i < count; i++)
            {
                if (!TryUpperHex(digits.Slice(4 * i, 4), out int value) || value is >= 0xD800 and <= 0xDFFF)
                {
                    return 0;
                }

                destination[i] = (char)value;
            }

            return count;
        }

        if (component.Length is >= 5 and <= 7 && component[0] == (byte)'u' && TryUpperHex(component[1..], out int scalar)
            && (scalar <= 0xD7FF || scalar is >= 0xE000 and <= 0x10FFFF))
        {
            return new Rune(scalar).EncodeToUtf16(destination);
        }

        return 0;
    }

    // Binary search of the glyph list's names (sorted by their bytes).
    private static int AglIndex(ReadOnlySpan<byte> name)
    {
        var names = AglNames;
        var ends = AglNameEnds;
        int lo = 0, hi = ends.Length - 1;
        while (lo <= hi)
        {
            int mid = (lo + hi) >>> 1;
            int c = names[(mid == 0 ? 0 : ends[mid - 1])..ends[mid]].SequenceCompareTo(name);
            if (c == 0)
            {
                return mid;
            }

            if (c < 0)
            {
                lo = mid + 1;
            }
            else
            {
                hi = mid - 1;
            }
        }

        return -1;
    }

    private static bool TryUpperHex(ReadOnlySpan<byte> digits, out int value)
    {
        value = 0;
        if (digits.IsEmpty || digits.Length > 6)
        {
            return false;
        }

        foreach (byte b in digits)
        {
            int d = b is >= (byte)'0' and <= (byte)'9' ? b - '0' : b is >= (byte)'A' and <= (byte)'F' ? b - 'A' + 10 : -1;
            if (d < 0)
            {
                return false;
            }

            value = value * 16 + d;
        }

        return true;
    }

    private static bool TryDecimal(ReadOnlySpan<byte> digits, out int value)
    {
        value = 0;
        if (digits.IsEmpty || digits.Length > 4)
        {
            return false;
        }

        foreach (byte b in digits)
        {
            if (b is < (byte)'0' or > (byte)'9')
            {
                return false;
            }

            value = value * 10 + (b - '0');
        }

        return true;
    }

    /// <summary>
    /// The standard font a <c>/BaseFont</c> names (its subset tag dropped; the common alternative names of the three
    /// families and their styles, as Arial, TimesNewRoman and CourierNew with ",Bold", "-Italic", "PS-BoldMT" and the
    /// like, taken as the standard font they stand for), or -1.
    /// </summary>
    public static int StandardFontIndex(string? baseFont)
    {
        if (string.IsNullOrEmpty(baseFont))
        {
            return -1;
        }

        string name = baseFont.Length > 7 && baseFont[6] == '+' && !baseFont.AsSpan(0, 6).ContainsAnyExcept(UpperLetters) ? baseFont[7..] : baseFont;
        int exact = Array.IndexOf(StandardFonts, name);
        if (exact >= 0)
        {
            return exact;
        }

        int split = name.IndexOfAny([',', '-']);
        string family = (split < 0 ? name : name[..split]).Replace(" ", "", StringComparison.Ordinal);
        string style = split < 0 ? "" : name[(split + 1)..];
        if (family.EndsWith("MT", StringComparison.Ordinal))
        {
            family = family[..^2];
        }

        if (family.EndsWith("PS", StringComparison.Ordinal))
        {
            family = family[..^2];
        }

        bool bold = style.Contains("Bold", StringComparison.OrdinalIgnoreCase) || family.EndsWith("Bold", StringComparison.Ordinal);
        bool italic = style.Contains("Italic", StringComparison.OrdinalIgnoreCase) || style.Contains("Oblique", StringComparison.OrdinalIgnoreCase)
                      || family.EndsWith("Italic", StringComparison.Ordinal);
        string? standard = family switch
        {
            "Helvetica" or "Arial" => bold ? italic ? "Helvetica-BoldOblique" : "Helvetica-Bold" : italic ? "Helvetica-Oblique" : "Helvetica",
            "Times" or "TimesNewRoman" or "TimesRoman" => bold ? italic ? "Times-BoldItalic" : "Times-Bold" : italic ? "Times-Italic" : "Times-Roman",
            "Courier" or "CourierNew" => bold ? italic ? "Courier-BoldOblique" : "Courier-Bold" : italic ? "Courier-Oblique" : "Courier",
            "Symbol" => "Symbol",
            "ZapfDingbats" or "Dingbats" => "ZapfDingbats",
            _ => null,
        };
        return standard is null ? -1 : Array.IndexOf(StandardFonts, standard);
    }

    private static readonly System.Buffers.SearchValues<char> UpperLetters = System.Buffers.SearchValues.Create("ABCDEFGHIJKLMNOPQRSTUVWXYZ");

    /// <summary>A standard font's ascender and descender (thousandths of the size; the bounding box's top and bottom where its AFM gives none).</summary>
    public static (int Ascent, int Descent) StandardExtents(int font)
    {
        var m = StandardMetrics.Slice(6 * font, 6);
        return (m[0], m[1]);
    }

    /// <summary>A standard font's width of the character <paramref name="codePoint"/> (thousandths of the size), or -1 when it has none.</summary>
    public static int StandardWidth(int font, int codePoint)
    {
        var starts = StandardWidthStarts;
        var keys = StandardWidthKeys[starts[font]..starts[font + 1]];
        int lo = 0, hi = keys.Length - 1;
        while (lo <= hi)
        {
            int mid = (lo + hi) >>> 1;
            if (keys[mid] == codePoint)
            {
                return StandardWidths[starts[font] + mid];
            }

            if (keys[mid] < codePoint)
            {
                lo = mid + 1;
            }
            else
            {
                hi = mid - 1;
            }
        }

        return -1;
    }
}
