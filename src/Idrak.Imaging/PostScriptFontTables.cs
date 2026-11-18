// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Buffers.Binary;

namespace Idrak.Imaging;

/// <summary>
/// The tables PostScript fonts name their glyphs by: the CFF specification's 391 standard strings (Adobe Technical Note
/// #5176, appendix A) and the standard encoding (appendix B, the same as Type 1's), as fontTools 4.68 holds them (copied
/// once from its cffLib and encodings; MIT licence); and the two tables a face of such a font gives through
/// <see cref="IFontFace.Table"/> so a reader finds its glyphs as it finds a TrueType font's: "post" (version 2, every
/// glyph's name) and "cmap" (one Macintosh (1, 0) subtable of format 6: the font's own encoding, a code to a glyph).
/// </summary>
internal static class PostScriptFontTables
{
    public static readonly string[] StandardStrings =
    [
        ".notdef", "space", "exclam", "quotedbl", "numbersign", "dollar", "percent", "ampersand",
        "quoteright", "parenleft", "parenright", "asterisk", "plus", "comma", "hyphen", "period",
        "slash", "zero", "one", "two", "three", "four", "five", "six",
        "seven", "eight", "nine", "colon", "semicolon", "less", "equal", "greater",
        "question", "at", "A", "B", "C", "D", "E", "F",
        "G", "H", "I", "J", "K", "L", "M", "N",
        "O", "P", "Q", "R", "S", "T", "U", "V",
        "W", "X", "Y", "Z", "bracketleft", "backslash", "bracketright", "asciicircum",
        "underscore", "quoteleft", "a", "b", "c", "d", "e", "f",
        "g", "h", "i", "j", "k", "l", "m", "n",
        "o", "p", "q", "r", "s", "t", "u", "v",
        "w", "x", "y", "z", "braceleft", "bar", "braceright", "asciitilde",
        "exclamdown", "cent", "sterling", "fraction", "yen", "florin", "section", "currency",
        "quotesingle", "quotedblleft", "guillemotleft", "guilsinglleft", "guilsinglright", "fi", "fl", "endash",
        "dagger", "daggerdbl", "periodcentered", "paragraph", "bullet", "quotesinglbase", "quotedblbase", "quotedblright",
        "guillemotright", "ellipsis", "perthousand", "questiondown", "grave", "acute", "circumflex", "tilde",
        "macron", "breve", "dotaccent", "dieresis", "ring", "cedilla", "hungarumlaut", "ogonek",
        "caron", "emdash", "AE", "ordfeminine", "Lslash", "Oslash", "OE", "ordmasculine",
        "ae", "dotlessi", "lslash", "oslash", "oe", "germandbls", "onesuperior", "logicalnot",
        "mu", "trademark", "Eth", "onehalf", "plusminus", "Thorn", "onequarter", "divide",
        "brokenbar", "degree", "thorn", "threequarters", "twosuperior", "registered", "minus", "eth",
        "multiply", "threesuperior", "copyright", "Aacute", "Acircumflex", "Adieresis", "Agrave", "Aring",
        "Atilde", "Ccedilla", "Eacute", "Ecircumflex", "Edieresis", "Egrave", "Iacute", "Icircumflex",
        "Idieresis", "Igrave", "Ntilde", "Oacute", "Ocircumflex", "Odieresis", "Ograve", "Otilde",
        "Scaron", "Uacute", "Ucircumflex", "Udieresis", "Ugrave", "Yacute", "Ydieresis", "Zcaron",
        "aacute", "acircumflex", "adieresis", "agrave", "aring", "atilde", "ccedilla", "eacute",
        "ecircumflex", "edieresis", "egrave", "iacute", "icircumflex", "idieresis", "igrave", "ntilde",
        "oacute", "ocircumflex", "odieresis", "ograve", "otilde", "scaron", "uacute", "ucircumflex",
        "udieresis", "ugrave", "yacute", "ydieresis", "zcaron", "exclamsmall", "Hungarumlautsmall", "dollaroldstyle",
        "dollarsuperior", "ampersandsmall", "Acutesmall", "parenleftsuperior", "parenrightsuperior", "twodotenleader", "onedotenleader", "zerooldstyle",
        "oneoldstyle", "twooldstyle", "threeoldstyle", "fouroldstyle", "fiveoldstyle", "sixoldstyle", "sevenoldstyle", "eightoldstyle",
        "nineoldstyle", "commasuperior", "threequartersemdash", "periodsuperior", "questionsmall", "asuperior", "bsuperior", "centsuperior",
        "dsuperior", "esuperior", "isuperior", "lsuperior", "msuperior", "nsuperior", "osuperior", "rsuperior",
        "ssuperior", "tsuperior", "ff", "ffi", "ffl", "parenleftinferior", "parenrightinferior", "Circumflexsmall",
        "hyphensuperior", "Gravesmall", "Asmall", "Bsmall", "Csmall", "Dsmall", "Esmall", "Fsmall",
        "Gsmall", "Hsmall", "Ismall", "Jsmall", "Ksmall", "Lsmall", "Msmall", "Nsmall",
        "Osmall", "Psmall", "Qsmall", "Rsmall", "Ssmall", "Tsmall", "Usmall", "Vsmall",
        "Wsmall", "Xsmall", "Ysmall", "Zsmall", "colonmonetary", "onefitted", "rupiah", "Tildesmall",
        "exclamdownsmall", "centoldstyle", "Lslashsmall", "Scaronsmall", "Zcaronsmall", "Dieresissmall", "Brevesmall", "Caronsmall",
        "Dotaccentsmall", "Macronsmall", "figuredash", "hypheninferior", "Ogoneksmall", "Ringsmall", "Cedillasmall", "questiondownsmall",
        "oneeighth", "threeeighths", "fiveeighths", "seveneighths", "onethird", "twothirds", "zerosuperior", "foursuperior",
        "fivesuperior", "sixsuperior", "sevensuperior", "eightsuperior", "ninesuperior", "zeroinferior", "oneinferior", "twoinferior",
        "threeinferior", "fourinferior", "fiveinferior", "sixinferior", "seveninferior", "eightinferior", "nineinferior", "centinferior",
        "dollarinferior", "periodinferior", "commainferior", "Agravesmall", "Aacutesmall", "Acircumflexsmall", "Atildesmall", "Adieresissmall",
        "Aringsmall", "AEsmall", "Ccedillasmall", "Egravesmall", "Eacutesmall", "Ecircumflexsmall", "Edieresissmall", "Igravesmall",
        "Iacutesmall", "Icircumflexsmall", "Idieresissmall", "Ethsmall", "Ntildesmall", "Ogravesmall", "Oacutesmall", "Ocircumflexsmall",
        "Otildesmall", "Odieresissmall", "OEsmall", "Oslashsmall", "Ugravesmall", "Uacutesmall", "Ucircumflexsmall", "Udieresissmall",
        "Yacutesmall", "Thornsmall", "Ydieresissmall", "001.000", "001.001", "001.002", "001.003", "Black",
        "Bold", "Book", "Light", "Medium", "Regular", "Roman", "Semibold",
    ];

    public static readonly string[] StandardEncoding =
    [
        ".notdef", ".notdef", ".notdef", ".notdef", ".notdef", ".notdef", ".notdef", ".notdef",
        ".notdef", ".notdef", ".notdef", ".notdef", ".notdef", ".notdef", ".notdef", ".notdef",
        ".notdef", ".notdef", ".notdef", ".notdef", ".notdef", ".notdef", ".notdef", ".notdef",
        ".notdef", ".notdef", ".notdef", ".notdef", ".notdef", ".notdef", ".notdef", ".notdef",
        "space", "exclam", "quotedbl", "numbersign", "dollar", "percent", "ampersand", "quoteright",
        "parenleft", "parenright", "asterisk", "plus", "comma", "hyphen", "period", "slash",
        "zero", "one", "two", "three", "four", "five", "six", "seven",
        "eight", "nine", "colon", "semicolon", "less", "equal", "greater", "question",
        "at", "A", "B", "C", "D", "E", "F", "G",
        "H", "I", "J", "K", "L", "M", "N", "O",
        "P", "Q", "R", "S", "T", "U", "V", "W",
        "X", "Y", "Z", "bracketleft", "backslash", "bracketright", "asciicircum", "underscore",
        "quoteleft", "a", "b", "c", "d", "e", "f", "g",
        "h", "i", "j", "k", "l", "m", "n", "o",
        "p", "q", "r", "s", "t", "u", "v", "w",
        "x", "y", "z", "braceleft", "bar", "braceright", "asciitilde", ".notdef",
        ".notdef", ".notdef", ".notdef", ".notdef", ".notdef", ".notdef", ".notdef", ".notdef",
        ".notdef", ".notdef", ".notdef", ".notdef", ".notdef", ".notdef", ".notdef", ".notdef",
        ".notdef", ".notdef", ".notdef", ".notdef", ".notdef", ".notdef", ".notdef", ".notdef",
        ".notdef", ".notdef", ".notdef", ".notdef", ".notdef", ".notdef", ".notdef", ".notdef",
        ".notdef", "exclamdown", "cent", "sterling", "fraction", "yen", "florin", "section",
        "currency", "quotesingle", "quotedblleft", "guillemotleft", "guilsinglleft", "guilsinglright", "fi", "fl",
        ".notdef", "endash", "dagger", "daggerdbl", "periodcentered", ".notdef", "paragraph", "bullet",
        "quotesinglbase", "quotedblbase", "quotedblright", "guillemotright", "ellipsis", "perthousand", ".notdef", "questiondown",
        ".notdef", "grave", "acute", "circumflex", "tilde", "macron", "breve", "dotaccent",
        "dieresis", ".notdef", "ring", "cedilla", ".notdef", "hungarumlaut", "ogonek", "caron",
        "emdash", ".notdef", ".notdef", ".notdef", ".notdef", ".notdef", ".notdef", ".notdef",
        ".notdef", ".notdef", ".notdef", ".notdef", ".notdef", ".notdef", ".notdef", ".notdef",
        ".notdef", "AE", ".notdef", "ordfeminine", ".notdef", ".notdef", ".notdef", ".notdef",
        "Lslash", "Oslash", "OE", "ordmasculine", ".notdef", ".notdef", ".notdef", ".notdef",
        ".notdef", "ae", ".notdef", ".notdef", ".notdef", "dotlessi", ".notdef", ".notdef",
        "lslash", "oslash", "oe", "germandbls", ".notdef", ".notdef", ".notdef", ".notdef",
    ];

    /// <summary>A "post" table of version 2 naming every glyph (each name as one of its own strings).</summary>
    public static byte[] Post(IReadOnlyList<string> names, int italicAngle = 0)
    {
        int count = Math.Min(names.Count, ushort.MaxValue);
        var strings = new List<byte[]>(count);
        int size = 32 + 2 + 2 * count;
        for (int g = 0; g < count; g++)
        {
            byte[] name = System.Text.Encoding.Latin1.GetBytes(names[g].Length > 255 ? names[g][..255] : names[g]);
            strings.Add(name);
            size += 1 + name.Length;
        }

        var table = new byte[size];
        BinaryPrimitives.WriteUInt32BigEndian(table, 0x00020000);
        BinaryPrimitives.WriteInt32BigEndian(table.AsSpan(4), italicAngle << 16);
        BinaryPrimitives.WriteUInt16BigEndian(table.AsSpan(32), (ushort)count);
        int at = 34 + 2 * count;
        for (int g = 0; g < count; g++)
        {
            BinaryPrimitives.WriteUInt16BigEndian(table.AsSpan(34 + 2 * g), (ushort)Math.Min(258 + g, ushort.MaxValue));
            table[at++] = (byte)strings[g].Length;
            strings[g].CopyTo(table, at);
            at += strings[g].Length;
        }

        return table;
    }

    /// <summary>A "cmap" table holding one Macintosh (1, 0) subtable of format 6: <paramref name="glyphs"/>[code] for codes 0-255 (0: none).</summary>
    public static byte[] Cmap(ReadOnlySpan<int> glyphs)
    {
        var table = new byte[12 + 10 + 2 * 256];
        BinaryPrimitives.WriteUInt16BigEndian(table.AsSpan(2), 1);                 // one subtable
        BinaryPrimitives.WriteUInt16BigEndian(table.AsSpan(4), 1);                 // Macintosh
        BinaryPrimitives.WriteUInt16BigEndian(table.AsSpan(6), 0);                 // Roman
        BinaryPrimitives.WriteUInt32BigEndian(table.AsSpan(8), 12);
        BinaryPrimitives.WriteUInt16BigEndian(table.AsSpan(12), 6);                // format 6
        BinaryPrimitives.WriteUInt16BigEndian(table.AsSpan(14), 10 + 2 * 256);
        BinaryPrimitives.WriteUInt16BigEndian(table.AsSpan(20), 256);              // first code 0, 256 entries
        for (int code = 0; code < 256 && code < glyphs.Length; code++)
        {
            BinaryPrimitives.WriteUInt16BigEndian(table.AsSpan(22 + 2 * code), (ushort)Math.Clamp(glyphs[code], 0, ushort.MaxValue));
        }

        return table;
    }
}
