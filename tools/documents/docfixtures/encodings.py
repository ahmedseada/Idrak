"""The simple fonts' base encodings and PDFDocEncoding, from the reference reader's own tables (pdfminer.six's
latin_enc: Standard, MacRoman, WinAnsi, PDFDoc), and glyph names to text by the Adobe Glyph List rules (fontTools.agl:
the list itself, uniXXXX, uXXXX[XX], ligatures joined by '_', suffixes after '.')."""
from fontTools import agl
from pdfminer.latin_enc import ENCODING

_COLUMNS = {"StandardEncoding": 1, "MacRomanEncoding": 2, "WinAnsiEncoding": 3, "PDFDocEncoding": 4}


def encoding_table(name):
    """Code -> glyph name for a base encoding."""
    col = _COLUMNS[name]
    return {row[col]: row[0] for row in ENCODING if row[col] is not None}


def glyph_text(name):
    """The text a glyph name stands for, or '' when the rules give none (a name such as g17)."""
    return agl.toUnicode(name)


_PDFDOC = {}
for _code, _glyph in encoding_table("PDFDocEncoding").items():
    _text = glyph_text(_glyph)
    if len(_text) == 1 and _text not in _PDFDOC:
        _PDFDOC[_text] = _code
for _c in "\t\n\r":
    _PDFDOC[_c] = ord(_c)


def pdfdoc_encode(text):
    """The PDFDocEncoding bytes of `text`, or None when a character is not in it."""
    out = bytearray()
    for ch in text:
        code = _PDFDOC.get(ch)
        if code is None:
            return None
        out.append(code)
    return bytes(out)


def simple_encoder(base, differences=None):
    """Text -> codes for a simple font: `base` an encoding name (or None for none), `differences` code -> glyph name.
    Returns (encode(text) -> bytes, code -> (glyph name, text))."""
    codes = dict(encoding_table(base)) if base else {}
    for code, glyph in (differences or {}).items():
        codes[code] = glyph
    by_code = {code: (glyph, glyph_text(glyph)) for code, glyph in codes.items()}
    reverse = {}
    for code in sorted(by_code):
        glyph, text = by_code[code]
        if text and text not in reverse:
            reverse[text] = code

    def encode(text):
        out = bytearray()
        for ch in text:
            if ch not in reverse:
                raise KeyError(f"{ch!r} (U+{ord(ch):04X}) is not in the encoding")
            out.append(reverse[ch])
        return bytes(out)

    return encode, by_code
