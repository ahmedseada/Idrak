// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Buffers.Binary;

namespace Idrak.Imaging;

/// <summary>
/// The tables PostScript fonts name their glyphs by: the CFF specification's 391 standard strings (Adobe Technical Note
/// #5176, appendix A) and the standard encoding (appendix B, the same as Type 1's), as fontTools 4.68 holds them (copied
/// once from its cffLib and encodings; MIT licence); the predefined Expert encoding and the Expert and Expert Subset
/// charsets (appendix C: the charsets as fontTools' cffLib names them, the encoding as the specification lists it, both
/// checked against FreeType's tables); and the two tables a face of such a font gives through
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

    /// <summary>
    /// The predefined Expert encoding (Top DICT Encoding 1; appendix C): each code's glyph name, ".notdef" for none. Code
    /// 175 is Macronsmall (SID 313), as the specification lists it (FreeType's table repeats 312 there).
    /// </summary>
    public static readonly string[] ExpertEncoding =
    [
        ".notdef", ".notdef", ".notdef", ".notdef", ".notdef", ".notdef", ".notdef", ".notdef",
        ".notdef", ".notdef", ".notdef", ".notdef", ".notdef", ".notdef", ".notdef", ".notdef",
        ".notdef", ".notdef", ".notdef", ".notdef", ".notdef", ".notdef", ".notdef", ".notdef",
        ".notdef", ".notdef", ".notdef", ".notdef", ".notdef", ".notdef", ".notdef", ".notdef",
        "space", "exclamsmall", "Hungarumlautsmall", ".notdef", "dollaroldstyle", "dollarsuperior", "ampersandsmall", "Acutesmall",
        "parenleftsuperior", "parenrightsuperior", "twodotenleader", "onedotenleader", "comma", "hyphen", "period", "fraction",
        "zerooldstyle", "oneoldstyle", "twooldstyle", "threeoldstyle", "fouroldstyle", "fiveoldstyle", "sixoldstyle", "sevenoldstyle",
        "eightoldstyle", "nineoldstyle", "colon", "semicolon", "commasuperior", "threequartersemdash", "periodsuperior", "questionsmall",
        ".notdef", "asuperior", "bsuperior", "centsuperior", "dsuperior", "esuperior", ".notdef", ".notdef",
        ".notdef", "isuperior", ".notdef", ".notdef", "lsuperior", "msuperior", "nsuperior", "osuperior",
        ".notdef", ".notdef", "rsuperior", "ssuperior", "tsuperior", ".notdef", "ff", "fi",
        "fl", "ffi", "ffl", "parenleftinferior", ".notdef", "parenrightinferior", "Circumflexsmall", "hyphensuperior",
        "Gravesmall", "Asmall", "Bsmall", "Csmall", "Dsmall", "Esmall", "Fsmall", "Gsmall",
        "Hsmall", "Ismall", "Jsmall", "Ksmall", "Lsmall", "Msmall", "Nsmall", "Osmall",
        "Psmall", "Qsmall", "Rsmall", "Ssmall", "Tsmall", "Usmall", "Vsmall", "Wsmall",
        "Xsmall", "Ysmall", "Zsmall", "colonmonetary", "onefitted", "rupiah", "Tildesmall", ".notdef",
        ".notdef", ".notdef", ".notdef", ".notdef", ".notdef", ".notdef", ".notdef", ".notdef",
        ".notdef", ".notdef", ".notdef", ".notdef", ".notdef", ".notdef", ".notdef", ".notdef",
        ".notdef", ".notdef", ".notdef", ".notdef", ".notdef", ".notdef", ".notdef", ".notdef",
        ".notdef", ".notdef", ".notdef", ".notdef", ".notdef", ".notdef", ".notdef", ".notdef",
        ".notdef", "exclamdownsmall", "centoldstyle", "Lslashsmall", ".notdef", ".notdef", "Scaronsmall", "Zcaronsmall",
        "Dieresissmall", "Brevesmall", "Caronsmall", ".notdef", "Dotaccentsmall", ".notdef", ".notdef", "Macronsmall",
        ".notdef", ".notdef", "figuredash", "hypheninferior", ".notdef", ".notdef", "Ogoneksmall", "Ringsmall",
        "Cedillasmall", ".notdef", ".notdef", ".notdef", "onequarter", "onehalf", "threequarters", "questiondownsmall",
        "oneeighth", "threeeighths", "fiveeighths", "seveneighths", "onethird", "twothirds", ".notdef", ".notdef",
        "zerosuperior", "onesuperior", "twosuperior", "threesuperior", "foursuperior", "fivesuperior", "sixsuperior", "sevensuperior",
        "eightsuperior", "ninesuperior", "zeroinferior", "oneinferior", "twoinferior", "threeinferior", "fourinferior", "fiveinferior",
        "sixinferior", "seveninferior", "eightinferior", "nineinferior", "centinferior", "dollarinferior", "periodinferior", "commainferior",
        "Agravesmall", "Aacutesmall", "Acircumflexsmall", "Atildesmall", "Adieresissmall", "Aringsmall", "AEsmall", "Ccedillasmall",
        "Egravesmall", "Eacutesmall", "Ecircumflexsmall", "Edieresissmall", "Igravesmall", "Iacutesmall", "Icircumflexsmall", "Idieresissmall",
        "Ethsmall", "Ntildesmall", "Ogravesmall", "Oacutesmall", "Ocircumflexsmall", "Otildesmall", "Odieresissmall", "OEsmall",
        "Oslashsmall", "Ugravesmall", "Uacutesmall", "Ucircumflexsmall", "Udieresissmall", "Yacutesmall", "Thornsmall", "Ydieresissmall",
    ];

    /// <summary>The predefined Expert charset (Top DICT charset 1; appendix C): each glyph's SID from glyph 0 (.notdef), 166 glyphs.</summary>
    public static readonly ushort[] ExpertCharset =
    [
        0, 1, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 13, 14, 15, 99,
        239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 27, 28, 249, 250, 251, 252,
        253, 254, 255, 256, 257, 258, 259, 260, 261, 262, 263, 264, 265, 266, 109, 110,
        267, 268, 269, 270, 271, 272, 273, 274, 275, 276, 277, 278, 279, 280, 281, 282,
        283, 284, 285, 286, 287, 288, 289, 290, 291, 292, 293, 294, 295, 296, 297, 298,
        299, 300, 301, 302, 303, 304, 305, 306, 307, 308, 309, 310, 311, 312, 313, 314,
        315, 316, 317, 318, 158, 155, 163, 319, 320, 321, 322, 323, 324, 325, 326, 150,
        164, 169, 327, 328, 329, 330, 331, 332, 333, 334, 335, 336, 337, 338, 339, 340,
        341, 342, 343, 344, 345, 346, 347, 348, 349, 350, 351, 352, 353, 354, 355, 356,
        357, 358, 359, 360, 361, 362, 363, 364, 365, 366, 367, 368, 369, 370, 371, 372,
        373, 374, 375, 376, 377, 378,
    ];

    /// <summary>The predefined Expert Subset charset (Top DICT charset 2; appendix C): each glyph's SID from glyph 0, 87 glyphs.</summary>
    public static readonly ushort[] ExpertSubsetCharset =
    [
        0, 1, 231, 232, 235, 236, 237, 238, 13, 14, 15, 99, 239, 240, 241, 242,
        243, 244, 245, 246, 247, 248, 27, 28, 249, 250, 251, 253, 254, 255, 256, 257,
        258, 259, 260, 261, 262, 263, 264, 265, 266, 109, 110, 267, 268, 269, 270, 272,
        300, 301, 302, 305, 314, 315, 158, 155, 163, 320, 321, 322, 323, 324, 325, 326,
        150, 164, 169, 327, 328, 329, 330, 331, 332, 333, 334, 335, 336, 337, 338, 339,
        340, 341, 342, 343, 344, 345, 346,
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
