"""Makes src/Idrak.Data/Pdf/PdfFontTables.Data.cs: the tables the PDF reader turns simple fonts' codes into text and
widths with (plan 28, phase D3), from the pinned packages of tools/documents/requirements.txt.

    python -I tools/documents/make_pdf_tables.py [OUT_FILE]

- The Adobe Glyph List (glyphlist.txt 2.0) and the ITC Zapf Dingbats glyph list, as fontTools.agl ships them
  (LEGACY_AGL2UV and the Zapf Dingbats list; BSD 3-clause, Adobe): glyph names to text.
- The base encodings StandardEncoding, MacRomanEncoding, WinAnsiEncoding and PDFDocEncoding, as pdfminer.six's latin_enc
  gives them (ISO 32000, Annex D.2): codes to glyph names.
- The built-in encodings of the standard Symbol and ZapfDingbats fonts (ISO 32000-2, Annex D.5 and D.6, written out
  below), checked here against pdfium's own reading of every code of both fonts (pypdfium2).
- The standard 14 fonts' metrics (ascender, descender, bounding box and every glyph's width), as pdfminer.six's
  fontmetrics gives them from Adobe's Core 14 AFM files, under the notice those files carry (copied into the output).
  Widths are keyed by the text of the glyph's name (as the AFM data is), the Zapf Dingbats' by their list's text, its
  space by the AFM's 278 (pdfminer.six keys that font's widths by the number in each aN name, so its space is lost).

Stored as the repository's other generated tables are: spans of constants the compiler keeps in the assembly's data
(nothing parsed or allocated when the reader starts), names as UTF-8 bytes sorted for a binary search. Reruns give the
same bytes.
"""

import os
import sys

from fontTools import agl
from pdfminer.fontmetrics import FONT_METRICS
from pdfminer.latin_enc import ENCODING

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(ROOT, "src", "Idrak.Data", "Pdf", "PdfFontTables.Data.cs")

# ISO 32000-2, Annex D.5: the Symbol font's built-in encoding (code -> glyph name).
SYMBOL = {
    0x20: "space", 0x21: "exclam", 0x22: "universal", 0x23: "numbersign", 0x24: "existential", 0x25: "percent",
    0x26: "ampersand", 0x27: "suchthat", 0x28: "parenleft", 0x29: "parenright", 0x2A: "asteriskmath", 0x2B: "plus",
    0x2C: "comma", 0x2D: "minus", 0x2E: "period", 0x2F: "slash", 0x30: "zero", 0x31: "one", 0x32: "two",
    0x33: "three", 0x34: "four", 0x35: "five", 0x36: "six", 0x37: "seven", 0x38: "eight", 0x39: "nine",
    0x3A: "colon", 0x3B: "semicolon", 0x3C: "less", 0x3D: "equal", 0x3E: "greater", 0x3F: "question",
    0x40: "congruent", 0x41: "Alpha", 0x42: "Beta", 0x43: "Chi", 0x44: "Delta", 0x45: "Epsilon", 0x46: "Phi",
    0x47: "Gamma", 0x48: "Eta", 0x49: "Iota", 0x4A: "theta1", 0x4B: "Kappa", 0x4C: "Lambda", 0x4D: "Mu", 0x4E: "Nu",
    0x4F: "Omicron", 0x50: "Pi", 0x51: "Theta", 0x52: "Rho", 0x53: "Sigma", 0x54: "Tau", 0x55: "Upsilon",
    0x56: "sigma1", 0x57: "Omega", 0x58: "Xi", 0x59: "Psi", 0x5A: "Zeta", 0x5B: "bracketleft", 0x5C: "therefore",
    0x5D: "bracketright", 0x5E: "perpendicular", 0x5F: "underscore", 0x60: "radicalex", 0x61: "alpha", 0x62: "beta",
    0x63: "chi", 0x64: "delta", 0x65: "epsilon", 0x66: "phi", 0x67: "gamma", 0x68: "eta", 0x69: "iota",
    0x6A: "phi1", 0x6B: "kappa", 0x6C: "lambda", 0x6D: "mu", 0x6E: "nu", 0x6F: "omicron", 0x70: "pi",
    0x71: "theta", 0x72: "rho", 0x73: "sigma", 0x74: "tau", 0x75: "upsilon", 0x76: "omega1", 0x77: "omega",
    0x78: "xi", 0x79: "psi", 0x7A: "zeta", 0x7B: "braceleft", 0x7C: "bar", 0x7D: "braceright", 0x7E: "similar",
    0xA0: "Euro", 0xA1: "Upsilon1", 0xA2: "minute", 0xA3: "lessequal", 0xA4: "fraction", 0xA5: "infinity",
    0xA6: "florin", 0xA7: "club", 0xA8: "diamond", 0xA9: "heart", 0xAA: "spade", 0xAB: "arrowboth",
    0xAC: "arrowleft", 0xAD: "arrowup", 0xAE: "arrowright", 0xAF: "arrowdown", 0xB0: "degree", 0xB1: "plusminus",
    0xB2: "second", 0xB3: "greaterequal", 0xB4: "multiply", 0xB5: "proportional", 0xB6: "partialdiff",
    0xB7: "bullet", 0xB8: "divide", 0xB9: "notequal", 0xBA: "equivalence", 0xBB: "approxequal", 0xBC: "ellipsis",
    0xBD: "arrowvertex", 0xBE: "arrowhorizex", 0xBF: "carriagereturn", 0xC0: "aleph", 0xC1: "Ifraktur",
    0xC2: "Rfraktur", 0xC3: "weierstrass", 0xC4: "circlemultiply", 0xC5: "circleplus", 0xC6: "emptyset",
    0xC7: "intersection", 0xC8: "union", 0xC9: "propersuperset", 0xCA: "reflexsuperset", 0xCB: "notsubset",
    0xCC: "propersubset", 0xCD: "reflexsubset", 0xCE: "element", 0xCF: "notelement", 0xD0: "angle",
    0xD1: "gradient", 0xD2: "registerserif", 0xD3: "copyrightserif", 0xD4: "trademarkserif", 0xD5: "product",
    0xD6: "radical", 0xD7: "dotmath", 0xD8: "logicalnot", 0xD9: "logicaland", 0xDA: "logicalor",
    0xDB: "arrowdblboth", 0xDC: "arrowdblleft", 0xDD: "arrowdblup", 0xDE: "arrowdblright", 0xDF: "arrowdbldown",
    0xE0: "lozenge", 0xE1: "angleleft", 0xE2: "registersans", 0xE3: "copyrightsans", 0xE4: "trademarksans",
    0xE5: "summation", 0xE6: "parenlefttp", 0xE7: "parenleftex", 0xE8: "parenleftbt", 0xE9: "bracketlefttp",
    0xEA: "bracketleftex", 0xEB: "bracketleftbt", 0xEC: "bracelefttp", 0xED: "braceleftmid", 0xEE: "braceleftbt",
    0xEF: "braceex", 0xF1: "angleright", 0xF2: "integral", 0xF3: "integraltp", 0xF4: "integralex",
    0xF5: "integralbt", 0xF6: "parenrighttp", 0xF7: "parenrightex", 0xF8: "parenrightbt", 0xF9: "bracketrighttp",
    0xFA: "bracketrightex", 0xFB: "bracketrightbt", 0xFC: "bracerighttp", 0xFD: "bracerightmid",
    0xFE: "bracerightbt",
}

# ISO 32000-2, Annex D.6: the ZapfDingbats font's built-in encoding (code -> glyph name).
_ZAPF_ROWS = [
    (0x20, "space a1 a2 a202 a3 a4 a5 a119 a118 a117 a11 a12 a13 a14 a15 a16 a105 a17 a18 a19 a20 a21 a22 a23 a24 a25 "
           "a26 a27 a28 a6 a7 a8 a9 a10 a29 a30 a31 a32 a33 a34 a35 a36 a37 a38 a39 a40 a41 a42 a43 a44 a45 a46 a47 "
           "a48 a49 a50 a51 a52 a53 a54 a55 a56 a57 a58 a59 a60 a61 a62 a63 a64 a65 a66 a67 a68 a69 a70 a71 a72 a73 "
           "a74 a203 a75 a204 a76 a77 a78 a79 a81 a82 a83 a84 a97 a98 a99 a100"),
    (0x80, "a89 a90 a93 a94 a91 a92 a205 a85 a206 a86 a87 a88 a95 a96"),
    (0xA1, "a101 a102 a103 a104 a106 a107 a108 a112 a111 a110 a109 a120 a121 a122 a123 a124 a125 a126 a127 a128 a129 "
           "a130 a131 a132 a133 a134 a135 a136 a137 a138 a139 a140 a141 a142 a143 a144 a145 a146 a147 a148 a149 a150 "
           "a151 a152 a153 a154 a155 a156 a157 a158 a159 a160 a161 a163 a164 a196 a165 a192 a166 a167 a168 a169 a170 "
           "a171 a172 a173 a162 a174 a175 a176 a177 a178 a179 a193 a180 a199 a181 a200 a182"),
    (0xF1, "a201 a183 a184 a197 a185 a194 a198 a186 a195 a187 a188 a189 a190 a191"),
]
ZAPF = {}
for _start, _names in _ZAPF_ROWS:
    for _i, _name in enumerate(_names.split()):
        ZAPF[_start + _i] = _name

LATIN = {"StandardEncoding": 1, "MacRomanEncoding": 2, "WinAnsiEncoding": 3, "PDFDocEncoding": 4}
ENCODINGS = ["StandardEncoding", "MacRomanEncoding", "WinAnsiEncoding", "PDFDocEncoding", "Symbol", "ZapfDingbats"]
STANDARD14 = ["Courier", "Courier-Bold", "Courier-BoldOblique", "Courier-Oblique", "Helvetica", "Helvetica-Bold",
              "Helvetica-BoldOblique", "Helvetica-Oblique", "Symbol", "Times-Bold", "Times-BoldItalic", "Times-Italic",
              "Times-Roman", "ZapfDingbats"]


def encoding_table(name):
    if name == "Symbol":
        return dict(SYMBOL)
    if name == "ZapfDingbats":
        return dict(ZAPF)
    col = LATIN[name]
    return {row[col]: row[0] for row in ENCODING if row[col] is not None}


def glyph_text(name, zapf=False):
    return agl.toUnicode(name, isZapfDingbats=zapf)


def check_against_pdfium():
    """Each code of Symbol and ZapfDingbats shown in one string: pdfium's text for it must be the name's text."""
    try:
        import pypdfium2 as pdfium
        import pypdfium2.raw as raw
    except ImportError:
        print("pypdfium2 is not installed: the built-in encodings are not checked against pdfium")
        return
    differences = []
    for font, table in (("Symbol", SYMBOL), ("ZapfDingbats", ZAPF)):
        codes = sorted(table)
        content = b"BT /F1 10 Tf 10 100 Td <" + bytes(codes).hex().upper().encode("ascii") + b"> Tj ET"
        objects = [b"<< /Type /Catalog /Pages 2 0 R >>", b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
                   b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 3000 200] /Resources << /Font << /F1 4 0 R >> >> "
                   b"/Contents 5 0 R >>",
                   b"<< /Type /Font /Subtype /Type1 /BaseFont /" + font.encode("ascii") + b" >>",
                   b"<< /Length %d >>\nstream\n" % len(content) + content + b"\nendstream"]
        out = bytearray(b"%PDF-1.4\n")
        offsets = []
        for i, body in enumerate(objects):
            offsets.append(len(out))
            out += b"%d 0 obj\n" % (i + 1) + body + b"\nendobj\n"
        xref = len(out)
        out += b"xref\n0 %d\n0000000000 65535 f \n" % (len(objects) + 1)
        for offset in offsets:
            out += b"%010d 00000 n \n" % offset
        out += b"trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n" % (len(objects) + 1, xref)
        pdf = pdfium.PdfDocument(bytes(out))
        text = pdf[0].get_textpage()
        count = raw.FPDFText_CountChars(text.raw)
        got = [chr(raw.FPDFText_GetUnicode(text.raw, i)) for i in range(count)]
        if len(got) != len(codes):
            differences.append(f"{font}: pdfium reads {len(got)} characters for {len(codes)} codes")
            continue
        for code, ch in zip(codes, got):
            want = glyph_text(table[code], zapf=font == "ZapfDingbats")
            if ch != want:
                differences.append(f"{font} {code:#04x} {table[code]}: pdfium U+{ord(ch):04X}, the glyph list "
                                   f"{' '.join(f'U+{ord(c):04X}' for c in want)}")
    print(f"pdfium {pdfium.PDFIUM_INFO.version if hasattr(pdfium, 'PDFIUM_INFO') else ''}: Symbol and ZapfDingbats "
          f"checked, {len(differences)} differences")
    for difference in differences:
        print("  " + difference)


def utf8_blob(names):
    """Names sorted by their bytes, as one UTF-8 literal (in lines) and each name's end."""
    data = b"".join(n.encode("ascii") for n in names)
    ends, at = [], 0
    for n in names:
        at += len(n)
        ends.append(at)
    if at > 0xFFFF:
        raise ValueError("the names do not fit 16-bit ends")
    lines, line = [], ""
    for n in names:
        if len(line) + len(n) > 110:
            lines.append(line)
            line = ""
        line += n
    lines.append(line)
    return lines, ends


def numbers(values, per_line=24, fmt=str):
    rows = []
    for i in range(0, len(values), per_line):
        rows.append("        " + ", ".join(fmt(v) for v in values[i:i + per_line]) + ",")
    return "\n".join(rows)


def u8_property(name, lines, doc):
    body = "\n        + ".join(f'"{line}"u8' for line in lines)
    return f"    /// <summary>{doc}</summary>\n    private static ReadOnlySpan<byte> {name} =>\n        {body};\n"


def span_property(name, kind, values, doc, fmt=str, per_line=24):
    return f"    /// <summary>{doc}</summary>\n    private static ReadOnlySpan<{kind}> {name} =>\n    [\n{numbers(values, per_line, fmt)}\n    ];\n"


def afm_notice():
    import pdfminer.fontmetrics as fm
    lines = open(fm.__file__, encoding="utf-8").read().splitlines()
    start = lines.index("###  BEGIN Verbatim copy of the license part")
    end = lines.index("###  END Verbatim copy of the license part")
    return [line.lstrip("#").strip() for line in lines[start + 1:end] if line.lstrip("#").strip()]


def agl_notice():
    lines = agl._aglText.splitlines()
    out = []
    for line in lines[1:]:
        if line.startswith("# ---"):
            break
        out.append(line.lstrip("#").strip())
    return [line for line in out if line]


def main():
    out_path = sys.argv[1] if len(sys.argv) > 1 else OUT

    # The glyph list: names unique, each one or more BMP code points.
    names = sorted(agl.LEGACY_AGL2UV, key=lambda n: n.encode("ascii"))
    texts = []
    for n in names:
        uvs = agl.LEGACY_AGL2UV[n]
        if any(u > 0xFFFF or 0xD800 <= u <= 0xDFFF for u in uvs):
            raise ValueError(f"{n}: a value outside the BMP")
        texts.append("".join(chr(u) for u in uvs))
    agl_lines, agl_ends = utf8_blob(names)
    text_units, text_ends = [], []
    for t in texts:
        text_units += [ord(c) for c in t]
        text_ends.append(len(text_units))

    zapf = [0] * len(agl._AGL_ZAPF_DINGBATS)
    for i, ch in enumerate(agl._AGL_ZAPF_DINGBATS):
        zapf[i] = 0 if ch == " " else ord(ch)

    # The encodings: one list of every name they use, sorted; each code an index into it (-1: none).
    tables = {e: encoding_table(e) for e in ENCODINGS}
    encoding_names = sorted({n for t in tables.values() for n in t.values()}, key=lambda n: n.encode("ascii"))
    index = {n: i for i, n in enumerate(encoding_names)}
    codes = []
    for e in ENCODINGS:
        row = [-1] * 256
        for code, name in tables[e].items():
            row[code] = index[name]
        codes += row
    enc_lines, enc_ends = utf8_blob(encoding_names)
    for e in ENCODINGS:
        for code, name in tables[e].items():
            if not glyph_text(name, zapf=e == "ZapfDingbats"):
                raise ValueError(f"{e} {code:#04x} {name}: no text by the glyph list")

    # The standard 14: metrics and widths keyed by code point.
    metrics, starts, keys, widths = [], [], [], []
    for font in STANDARD14:
        descriptor, afm = FONT_METRICS[font]
        bbox = [int(v) for v in descriptor["FontBBox"]]
        ascent = int(descriptor.get("Ascent", bbox[3]))
        descent = int(descriptor.get("Descent", bbox[1]))
        metrics += [ascent, descent] + bbox
        by_point = {}
        if font == "ZapfDingbats":
            for key, width in afm.items():
                by_point[ord(agl._AGL_ZAPF_DINGBATS[ord(key)])] = width
            by_point[0x20] = 278   # the AFM's space (C 32 ; WX 278 ; N space)
        else:
            for key, width in afm.items():
                if len(key) != 1:
                    raise ValueError(f"{font}: a key of {len(key)} characters")
                by_point[ord(key)] = width
        starts.append(len(keys))
        for point in sorted(by_point):
            keys.append(point)
            widths.append(by_point[point])
        # Every encoded name of the font's own encoding has a width.
        own = "Symbol" if font == "Symbol" else "ZapfDingbats" if font == "ZapfDingbats" else "StandardEncoding"
        for code, name in tables[own].items():
            text = glyph_text(name, zapf=font == "ZapfDingbats")
            if len(text) != 1 or ord(text) not in by_point:
                raise ValueError(f"{font}: {name} ({code:#04x}) has no width")
    starts.append(len(keys))

    check_against_pdfium()

    afm = afm_notice()
    aglnote = agl_notice()
    header = [
        "// Copyright (c) 2026 Ahmed Seada",
        "// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.",
        "",
        "// <auto-generated>",
        "// Made by tools/documents/make_pdf_tables.py from the pinned packages of tools/documents/requirements.txt; do not",
        "// edit. The glyph list and the ITC Zapf Dingbats glyph list are fontTools.agl's copies of Adobe's (glyphlist.txt",
        "// 2.0, zapfdingbats.txt, https://github.com/adobe-type-tools/agl-aglfn), under this notice:",
        "//",
    ] + [f"//   {line}" for line in aglnote] + [
        "//",
        "// The standard 14 fonts' metrics are from Adobe's Core 14 AFM files (as pdfminer.six's fontmetrics holds them),",
        "// under the notice those files carry; this table holds widths and metrics taken from them, no AFM file:",
        "//",
    ] + [f"//   {line}" for line in afm] + [
        "//",
        "// The base encodings are ISO 32000's Annex D.2 (pdfminer.six's latin_enc); the Symbol and ZapfDingbats built-in",
        "// encodings Annex D.5 and D.6, checked against pdfium.",
        "// </auto-generated>",
        "",
        "namespace Idrak.Data.Pdf;",
        "",
        "internal static partial class PdfFontTables",
        "{",
        "    /// <summary>The standard 14 fonts, in the order of the metrics below.</summary>",
        "    internal static readonly string[] StandardFonts = [" + ", ".join(f'"{f}"' for f in STANDARD14) + "];",
        "",
        "    /// <summary>The encodings, in the order of <see cref=\"EncodingCodes\"/>' rows.</summary>",
        "    internal static readonly string[] EncodingOrder = [" + ", ".join(f'"{e}"' for e in ENCODINGS) + "];",
        "",
    ]
    parts = [
        u8_property("AglNames", agl_lines, f"The glyph list's {len(names)} names, sorted by their bytes, one after another."),
        span_property("AglNameEnds", "ushort", agl_ends, "Where each name of <see cref=\"AglNames\"/> ends."),
        span_property("AglText", "ushort", text_units, "Each name's text (UTF-16), one after another.", fmt=lambda v: f"0x{v:04X}", per_line=16),
        span_property("AglTextEnds", "ushort", text_ends, "Where each name's text in <see cref=\"AglText\"/> ends."),
        span_property("ZapfText", "ushort", zapf, "The ITC Zapf Dingbats glyph list: the character of the name aN at index N (0: none).", fmt=lambda v: f"0x{v:04X}", per_line=16),
        u8_property("EncodingNames", enc_lines, f"The {len(encoding_names)} glyph names the encodings use, sorted by their bytes, one after another."),
        span_property("EncodingNameEnds", "ushort", enc_ends, "Where each name of <see cref=\"EncodingNames\"/> ends."),
        span_property("EncodingCodes", "short", codes, "Each encoding's 256 codes (rows in <see cref=\"EncodingOrder\"/>' order): an index into <see cref=\"EncodingNames\"/>, or -1.", per_line=32),
        span_property("StandardMetrics", "short", metrics, "Each standard font's ascender, descender and bounding box (left, bottom, right, top), in thousandths of the size.", per_line=6),
        span_property("StandardWidthStarts", "int", starts, "Where each standard font's widths start in <see cref=\"StandardWidthKeys\"/> (and one past the last)."),
        span_property("StandardWidthKeys", "ushort", keys, "The code points each standard font has a width for, ascending within a font.", fmt=lambda v: f"0x{v:04X}", per_line=16),
        span_property("StandardWidths", "short", widths, "The widths of <see cref=\"StandardWidthKeys\"/>, in thousandths of the size."),
    ]
    text = "\n".join(header) + "\n" + "\n".join(parts) + "}\n"
    with open(out_path, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)
    print(f"wrote {out_path}: {len(names)} glyph names, {len(encoding_names)} encoding names, "
          f"{len(keys)} standard widths ({len(text):,} characters)")


if __name__ == "__main__":
    main()
