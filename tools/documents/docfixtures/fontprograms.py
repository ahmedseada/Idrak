"""Font programs the D7 samples embed beyond D3's: a **Type 1** font (FontFile, eexec-encrypted) made here from a
repository font's outlines (DejaVu Sans, whose licence allows it: LICENSE-DejaVu.txt), and **CFF** fonts (FontFile3
/Type1C as a simple font, FontFile3 /OpenType under a CIDFontType0) from the repository's DejaVu Sans CFF.

The Type 1 program: each glyph's TrueType outline scaled to 1000 units a em, its quadratic curves raised to cubics
exactly and every point rounded to whole units (Type 1 charstrings hold integers), as hsbw, rmoveto, rlineto, rrcurveto,
closepath and endchar; no hints, the five standard subroutines unused; the font dictionary and the eexec encryption written by fontTools'
t1Lib (eexec key 55665, charstring key 4330, lenIV 4). Its glyphs read back through fontTools' own Type 1 parser are
what a reader must get (`glyphs`: advance and outline commands, in 1000 units)."""
import io

from fontTools.encodings.StandardEncoding import StandardEncoding
from fontTools.misc.psCharStrings import T1CharString
from fontTools.pens.basePen import BasePen
from fontTools.pens.recordingPen import RecordingPen
from fontTools.pens.transformPen import TransformPen
from fontTools.t1Lib import ND_values, PD_values, RD_value, T1Font, std_subrs
from fontTools.ttLib import TTFont

from .encodings import glyph_text, simple_encoder
from .filters import flate
from .fonts import Font, Glyph, Type0, _descriptor, _font_program, _units, subset_tag, to_unicode_stream, w_array
from .pdf import N, Stream, String


class _T1Pen(BasePen):
    """Absolute integer points to a Type 1 program (relative operators)."""

    def __init__(self, glyphset):
        super().__init__(glyphset)
        self.prog = []
        self.cur = (0, 0)

    def _rel(self, pt):
        x, y = round(pt[0]), round(pt[1])
        dx, dy = x - self.cur[0], y - self.cur[1]
        self.cur = (x, y)
        return [dx, dy]

    def _moveTo(self, pt):
        self.prog += self._rel(pt) + ["rmoveto"]

    def _lineTo(self, pt):
        self.prog += self._rel(pt) + ["rlineto"]

    def _curveToOne(self, p1, p2, p3):
        self.prog += self._rel(p1) + self._rel(p2) + self._rel(p3) + ["rrcurveto"]

    def _closePath(self):
        self.prog.append("closepath")

    def _endPath(self):
        self.prog.append("closepath")


def build_type1(src, chars, font_name="IdrakSampleType1", full_name="Idrak Sample Type 1"):
    """A Type 1 font with the glyphs of `chars` (and .notdef) from the TrueType font `src`. Returns (data, lengths
    (Length1, Length2, Length3), glyphs: name -> {"width", "outline"} as fontTools reads the program back)."""
    tt = TTFont(src, recalcTimestamp=False)
    upem = tt["head"].unitsPerEm
    scale = 1000 / upem
    cmap = tt.getBestCmap()
    gs = tt.getGlyphSet()
    names = [".notdef"] + sorted({cmap[ord(ch)] for ch in chars}, key=lambda n: tt.getGlyphID(n))
    std = set(StandardEncoding)
    for n in names[1:]:
        if n not in std:
            raise ValueError(f"glyph {n} is not in StandardEncoding")
    charstrings = {}
    bbox = [0, 0, 0, 0]
    for n in names:
        pen = _T1Pen(None)
        sb = 0
        gs[n].draw(TransformPen(pen, (scale, 0, 0, scale, 0, 0)))
        width = round(tt["hmtx"][n][0] * scale)
        prog = [sb, width, "hsbw"] + pen.prog + ["endchar"]
        charstrings[n] = T1CharString(program=prog)
    for n in names:
        rec = RecordingPen()
        gs[n].draw(TransformPen(rec, (scale, 0, 0, scale, 0, 0)))
        for _, pts in rec.value:
            for x, y in pts:
                bbox = [min(bbox[0], round(x)), min(bbox[1], round(y)), max(bbox[2], round(x)), max(bbox[3], round(y))]
    hhea = tt["hhea"]
    font = {
        "FontInfo": {"version": "001.000", "Notice": "Made by Idrak's sample generator from DejaVu Sans outlines",
                     "FullName": full_name, "FamilyName": "Idrak Sample", "Weight": "Book", "ItalicAngle": 0,
                     "isFixedPitch": False, "UnderlinePosition": -100, "UnderlineThickness": 50},
        "FontName": font_name,
        "Encoding": StandardEncoding,
        "PaintType": 0,
        "FontType": 1,
        "FontMatrix": [0.001, 0, 0, 0.001, 0, 0],
        "FontBBox": tuple(bbox),
        "Private": {"RD": RD_value, "ND": ND_values[1], "NP": PD_values[1], "MinFeature": (16, 16),
                    "password": 5839, "lenIV": 4, "BlueValues": [round(hhea.descent * scale), 0, 729, 729],
                    "StdHW": [80], "StdVW": [90], "OtherSubrs": True,
                    # The standard five (flex and hint replacement); fontTools puts the Private dictionary after them.
                    "Subrs": [T1CharString(bytecode=b) for b in std_subrs]},
        "CharStrings": charstrings,
    }
    t1 = T1Font.__new__(T1Font)
    t1.encoding = "ascii"
    t1.font = font
    data = t1.createData()
    marker = b"currentfile eexec "
    len1 = data.index(marker) + len(marker)
    tail = data.index(b"\n" + b"0" * 64)
    length2 = tail - len1
    lengths = (len1, length2, len(data) - tail)
    back = T1Font.__new__(T1Font)
    back.encoding = "ascii"
    back.data = data
    back.parse()
    glyphs = {}
    bgs = back.getGlyphSet()
    for n in names:
        rec = RecordingPen()
        bgs[n].draw(rec)
        glyphs[n] = {"width": bgs[n].width, "outline": [[op] + [list(p) for p in pts] for op, pts in rec.value]}
    ascent, descent = round(hhea.ascent * scale), round(hhea.descent * scale)
    return data, lengths, glyphs, {"bbox": bbox, "ascent": ascent, "descent": descent}


class Type1Embedded(Font):
    """An embedded Type 1 font (FontFile with Length1-3) as a simple font, its built-in encoding (StandardEncoding,
    no /Encoding in the font dictionary), /Widths from its hsbw advances."""

    kind = "type1"

    def __init__(self, doc, resname, src, chars, font_name="IdrakSampleType1"):
        super().__init__(doc, resname)
        self.program, self.lengths, self.outlines, meta = build_type1(src, chars, font_name)
        self.base_font = font_name
        self.ascent, self.descent = float(meta["ascent"]), float(meta["descent"])
        self.bbox = meta["bbox"]
        self.codes = {i: n for i, n in enumerate(StandardEncoding) if n != ".notdef"}
        self.reverse = {}
        for code in sorted(self.codes):
            t = glyph_text(self.codes[code])
            if t and t not in self.reverse:
                self.reverse[t] = code
        self.used = set()

    def width(self, code):
        return self.outlines[self.codes[code]]["width"]

    def glyphs(self, text):
        out = []
        for ch in text:
            code = self.reverse[ch]
            self.used.add(code)
            out.append(Glyph(bytes([code]), self.width(code), ch, code == 32))
        return out

    def finish(self):
        l1, l2, l3 = self.lengths
        program = self.doc.pdf.add(Stream({"Length1": l1, "Length2": l2, "Length3": l3, "Filter": N("FlateDecode")},
                                          flate(self.program), self.program))
        self.program_ref = program
        descriptor = self.doc.pdf.add({"Type": N("FontDescriptor"), "FontName": N(self.base_font), "Flags": 32,
                                       "FontBBox": list(self.bbox), "ItalicAngle": 0, "Ascent": self.ascent,
                                       "Descent": self.descent, "CapHeight": 729, "StemV": 90, "FontFile": program})
        first, last = min(self.used), max(self.used)
        d = {"Type": N("Font"), "Subtype": N("Type1"), "BaseFont": N(self.base_font), "FirstChar": first,
             "LastChar": last, "Widths": [self.width(c) if c in self.used else 0 for c in range(first, last + 1)],
             "FontDescriptor": descriptor}
        self.doc.pdf.put(self.ref, d)

    def describe_program(self):
        return {"kind": "type1", "object": [self.program_ref.num, 0], "lengths": list(self.lengths),
                "fontName": self.base_font, "encoding": "StandardEncoding (built in)", "unitsPerEm": 1000,
                "glyphs": self.outlines}


class CffSimple(Font):
    """A simple font (/Subtype /Type1) whose program is a bare CFF (FontFile3 /Subtype /Type1C), subset from an OTF,
    /Encoding /WinAnsiEncoding (codes to glyph names to the CFF charset)."""

    kind = "type1c"

    def __init__(self, doc, resname, path):
        super().__init__(doc, resname)
        self.path = path
        self.full = TTFont(path, recalcTimestamp=False)
        self.upem = self.full["head"].unitsPerEm
        self.cmap = self.full.getBestCmap()
        hhea = self.full["hhea"]
        self.ascent = float(_units(hhea.ascent, self.upem))
        self.descent = float(_units(hhea.descent, self.upem))
        _, self.codes = simple_encoder("WinAnsiEncoding")
        self.reverse = {}
        for code in sorted(self.codes):
            g, t = self.codes[code]
            if t and t not in self.reverse:
                self.reverse[t] = code
        self.used = {}
        self.ps_name = self.full["name"].getDebugName(6) or "Font"

    def width(self, ch):
        return _units(self.full["hmtx"][self.cmap[ord(ch)]][0], self.upem)

    def glyphs(self, text):
        out = []
        for ch in text:
            code = self.reverse[ch]
            if self.codes[code][0] != self.cmap[ord(ch)]:
                raise ValueError(f"{ch!r}: WinAnsi names {self.codes[code][0]}, the font {self.cmap[ord(ch)]}")
            self.used[code] = ch
            out.append(Glyph(bytes([code]), self.width(ch), ch, code == 32))
        return out

    def finish(self):
        unicodes = sorted({ord(c) for c in self.used.values()})
        data, sub = _font_program(self.path, unicodes=unicodes)
        cff = sub.getTableData("CFF ")
        tag = subset_tag(cff)
        self.base_font = f"{tag}+{self.ps_name}"
        program = self.doc.pdf.add(Stream({"Subtype": N("Type1C"), "Filter": N("FlateDecode")}, flate(cff), cff))
        self.program_ref = program
        descriptor = self.doc.pdf.add(_descriptor(self.base_font, sub, self.upem, program, symbolic=False,
                                                  key="FontFile3"))
        first, last = min(self.used), max(self.used)
        widths = [self.width(self.used[c]) if c in self.used else 0 for c in range(first, last + 1)]
        d = {"Type": N("Font"), "Subtype": N("Type1"), "BaseFont": N(self.base_font), "FirstChar": first,
             "LastChar": last, "Widths": widths, "Encoding": N("WinAnsiEncoding"), "FontDescriptor": descriptor}
        self.doc.pdf.put(self.ref, d)


class CffType0(Type0):
    """A composite font over CIDFontType0 whose program is an OpenType font with CFF outlines (FontFile3 /Subtype
    /OpenType), Identity-H, CIDs the glyph ids (the CFF is not CID-keyed, so a CID is its glyph index), ToUnicode
    written; as Idrak's writer embeds CFF faces (D6)."""

    kind = "type0-cff"

    def finish(self):
        gids = sorted(set(self.cid_gid.values()) | {0})
        unicodes = sorted({ord(c) for t in self.cid_text.values() for c in t if ord(c) in self.cmap})
        data, sub = _font_program(self.path, unicodes=unicodes, gids=gids, retain_gids=True)
        self.sub_font = sub
        tag = subset_tag(data)
        self.base_font = f"{tag}+{self.ps_name}"
        program = self.doc.pdf.add(Stream({"Subtype": N("OpenType"), "Filter": N("FlateDecode")}, flate(data), data))
        self.program_ref = program
        descriptor = self.doc.pdf.add(_descriptor(self.base_font, sub, self.upem, program, symbolic=True,
                                                  key="FontFile3"))
        cids = sorted(self.cid_gid)
        cid_font = {"Type": N("Font"), "Subtype": N("CIDFontType0"), "BaseFont": N(self.base_font),
                    "CIDSystemInfo": {"Registry": String(b"Adobe"), "Ordering": String(b"Identity"), "Supplement": 0},
                    "FontDescriptor": descriptor, "DW": 1000,
                    "W": w_array({c: self.gid_width(self.cid_gid[c]) for c in cids}, self.w_style)}
        d = {"Type": N("Font"), "Subtype": N("Type0"), "BaseFont": N(self.base_font + "-" + self.encoding),
             "Encoding": N(self.encoding), "DescendantFonts": [self.doc.pdf.add(cid_font)]}
        pairs = {c.to_bytes(2, "big"): t for c, t in self.cid_text.items()}
        d["ToUnicode"] = self.doc.pdf.add(to_unicode_stream(pairs, 2, style=self.tu_style))
        self.doc.pdf.put(self.ref, d)


def cff_outlines(font_bytes, names):
    """Glyph outlines (RecordingPen commands, font units) of `names` in an OpenType font's bytes."""
    tt = TTFont(io.BytesIO(font_bytes))
    gs = tt.getGlyphSet()
    out = {}
    for n in names:
        rec = RecordingPen()
        gs[n].draw(rec)
        out[n] = {"width": tt["hmtx"][n][0], "outline": [[op] + [list(p) for p in pts] for op, pts in rec.value]}
    return out

