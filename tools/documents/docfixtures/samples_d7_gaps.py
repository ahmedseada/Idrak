"""The drawing gaps D7 left (closed 2026-11-19), a sample each, drawn by pdfium beside it as the D7 samples are: groups
that are not isolated with blend modes inside them and an opacity of their own (one a knockout group, one isolated to
compare); strokes under matrices that are not uniform (a stretched polyline, a squashed circle, a skewed square, dashes,
a turned stretch, and a uniform scale to compare); Type 3 glyphs that are more than filled shapes (a d1 glyph stroked, a
d1 glyph that is an image mask, d0 glyphs in their own colours and an image of their own, and a plain d1 glyph); TrueType
fonts found by their glyphs' standard Macintosh names (a `post` table of format 1, and one of format 2 naming its glyphs
by the standard indices), their cmaps holding no letter; bare CFF fonts with the predefined Expert encoding and the
Expert and Expert Subset charsets (no /Encoding: the program's own); tiling cells larger than their /BBox (upright,
turned, reaching past their steps, uncoloured, and one holding an image); and vertical text whose /W2 gives position
vectors of its own. Every text is generic. Each page's paths, clips and groups are listed as the D7 samples' are."""
import io
import math

from fontTools.fontBuilder import FontBuilder
from fontTools.pens.recordingPen import DecomposingRecordingPen
from fontTools.pens.t2CharStringPen import T2CharStringPen
from fontTools.pens.transformPen import TransformPen
from fontTools.pens.ttGlyphPen import TTGlyphPen
from fontTools.cffLib import cffExpertSubsetStrings, cffIExpertStrings
from fontTools.ttLib import TTFont
from fontTools.ttLib.standardGlyphOrder import standardGlyphOrder

from .common import DEJAVU
from .encodings import glyph_text, simple_encoder
from .filters import flate
from .fonts import Font, Glyph, Type0, Type3, _units, subset_tag
from .pdf import N, Stream
from .samples_d7 import H, W, _circle, _doc, _finish, _sample
from .vector import RGB, ExtGState, PatternSpace, TilingPattern

# -- groups that are not isolated ---------------------------------------------------------------------------------------

BARS = [(0.95, 0.85, 0.15), (0.25, 0.65, 0.9), (0.9, 0.35, 0.4), (0.4, 0.75, 0.45)]


def _rgb255(c):
    return [round(min(1.0, max(0.0, v)) * 255) for v in c]


def groups(seed):
    doc = _doc("vector-groups-non-isolated", "1.7", seed)
    p = doc.page(W, H)
    c = p.content
    for k, col in enumerate(BARS):                          # the backdrop: four upright bars
        c.rg(*col)
        c.re(100 * k, 0, 100, H)
        c.f()
    gstates = {}

    def state(name, entries):
        if name not in gstates:
            gstates[name] = ExtGState(doc, {"Type": N("ExtGState"), **entries})
        return gstates[name]

    groups_json = []

    def group(index, iso, ko, opacity, body):
        groups_json.append({"index": index, "isolated": iso, "knockout": ko, "form": f"Fm{index + 1}",
                            "opacity": opacity, "probes": []})
        c.q()
        c.gs(f"GA{index}", state(f"GA{index}", {"ca": opacity, "CA": opacity}))
        with c.form(f"Fm{index + 1}", [0, 0, W, H], extra={"Group": {"Type": N("Group"), "S": N("Transparency"),
                                                                     "I": iso, "K": ko, "CS": N("DeviceRGB")}}) as g:
            g.vs["group"] = index
            # A transparency group's content starts at opacity 1, Normal, no soft mask (ISO 32000-2, 11.6.6): the
            # opacity in force is the group's own, applied when it is composed.
            g.vs["ca"] = g.vs["CA"] = 1.0
            g.vs["blend"] = "Normal"
            g.vs["smask"] = None
            body(g)
        groups_json[index]["object"] = [g.ref.num, 0]
        c.Q()

    orange, blue, teal = (0.95, 0.55, 0.15), (0.15, 0.3, 0.85), (0.1, 0.6, 0.55)

    # Group 1: not isolated, not knockout, at 0.6: a Multiply strip and a Screen strip blend with the bars under it.
    def first(g):
        g.gs("BMul", state("BMul", {"BM": N("Multiply")}))
        g.rg(*orange)
        g.re(20, 225, 360, 45)
        g.f()
        g.gs("BScr", state("BScr", {"BM": N("Screen")}))
        g.rg(*blue)
        g.re(20, 175, 360, 35)
        g.f()

    group(0, False, False, 0.6, first)
    # Group 2: not isolated, knockout, at 0.8: a Normal circle at 0.5, then a Multiply circle at 0.7 knocking it out
    # (composed with the bar under the group, not with the first circle); a Difference circle over the third bar.
    def second(g):
        g.gs("Half", state("Half", {"ca": 0.5}))
        g.rg(*blue)
        _circle(g, 60, 110, 30)
        g.f()
        g.gs("MulA", state("MulA", {"BM": N("Multiply"), "ca": 0.7}))
        g.rg(*orange)
        _circle(g, 95, 110, 30)
        g.f()
        g.gs("Diff", state("Diff", {"BM": N("Difference"), "ca": 1}))
        g.rg(*teal)
        _circle(g, 250, 110, 30)
        g.f()

    group(1, False, True, 0.8, second)
    # Group 3: isolated, at 0.6, a Multiply strip: blends with nothing (drawn as it always was).
    def third(g):
        g.gs("BMul", state("BMul", {"BM": N("Multiply")}))
        g.rg(*orange)
        g.re(20, 25, 360, 35)
        g.f()

    group(2, True, False, 0.6, third)

    # The colours the transparency rules give (ISO 32000-2, 11.4.7-11.4.8): a group that is not isolated composes its
    # parts with the bars, takes the bars back out (C = Cn + (Cn - C0)(a0 / ag - a0)) and is composed at its opacity.
    def mul(a, b):
        return tuple(x * y for x, y in zip(a, b))

    def screen(a, b):
        return tuple(x + y - x * y for x, y in zip(a, b))

    def diff(a, b):
        return tuple(abs(x - y) for x, y in zip(a, b))

    def over(back, colour, alpha):
        return tuple((1 - alpha) * x + alpha * y for x, y in zip(back, colour))

    def probe(index, x, y, rgb, note):
        groups_json[index]["probes"].append({"point": [x, H - y], "rgb": _rgb255(rgb), "tolerance": 3, "note": note})

    for k, bar in enumerate(BARS):
        x = 100 * k + 50
        probe(0, x, 247, over(bar, mul(bar, orange), 0.6), "the Multiply strip over a bar (group at 0.6)")
        probe(0, x, 192, over(bar, screen(bar, blue), 0.6), "the Screen strip over a bar (group at 0.6)")
        probe(2, x, 42, over(bar, orange, 0.6), "an isolated group's Multiply strip (with nothing under it)")
    bar0, bar2 = BARS[0], BARS[2]
    probe(1, 45, 110, over(bar0, blue, 0.4), "the first circle alone (0.5 inside a group at 0.8)")
    probe(1, 78, 110, over(bar0, mul(bar0, orange), 0.56), "the overlap: the Multiply circle knocks out the first")
    probe(1, 115, 110, over(BARS[1], mul(BARS[1], orange), 0.56), "the Multiply circle over the second bar")
    probe(1, 250, 110, over(bar2, diff(bar2, teal), 0.8), "the Difference circle (group at 0.8)")
    data, w = _finish(doc)
    desc = ("Transparency groups that are not isolated (ISO 32000-2, 11.4.7): over four coloured bars, a group at "
            "opacity 0.6 holding a Multiply strip and a Screen strip, and a knockout group at 0.8 holding a circle at 0.5 "
            "and a Multiply circle at 0.7 that knocks it out, and a Difference circle; their parts blend with the bars "
            "under them. An isolated group with a Multiply strip, below, blends with nothing. Each group lists points "
            "whose colour the rules fix (`probes`).")
    sample = _sample(doc, data, w, "vector-groups-non-isolated", desc)
    sample.expected["pages"][0]["groups"] = groups_json
    return sample


# -- strokes under matrices that are not uniform ------------------------------------------------------------------------

def strokes_pen(seed):
    doc = _doc("vector-strokes-pen", "1.4", seed)
    p = doc.page(W, H)
    c = p.content
    c.RG(0.15, 0.3, 0.7)
    c.q()                                                   # a polyline stretched 3 across: round caps and joins
    c.cm(3, 0, 0, 1, 0, 0)
    c.w(4)
    c.J(1)
    c.j(1)
    c.m(8, 235); c.l(25, 280); c.l(42, 235); c.l(59, 280); c.S()
    c.Q()
    c.q()                                                   # a circle squashed to a third: the ellipse's ends thicker
    c.cm(1, 0, 0, 0.35, 0, 0)
    c.w(12)
    c.RG(0.75, 0.2, 0.25)
    _circle(c, 300, 700, 60); c.S()
    c.Q()
    c.q()                                                   # a square skewed: miter joins
    c.cm(1, 0, 0.8, 1, 0, 0)
    c.w(6)
    c.RG(0.1, 0.55, 0.3)
    c.re(-95, 135, 60, 50); c.S()
    c.Q()
    c.q()                                                   # dashes under 0.5 by 2: their lengths in user space
    c.cm(0.5, 0, 0, 2, 0, 0)
    c.w(3)
    c.d([10, 6], 0)
    c.RG(0.45, 0.25, 0.6)
    c.m(400, 20); c.l(760, 70); c.S()
    c.Q()
    a = math.radians(25)
    c.q()                                                   # turned and stretched: bevel joins, square caps
    c.cm(*[round(v, 4) for v in (2.2 * math.cos(a), 2.2 * math.sin(a), -0.6 * math.sin(a), 0.6 * math.cos(a))], 180, 20)   # as written
    c.w(7)
    c.j(2)
    c.J(2)
    c.RG(0.85, 0.5, 0.1)
    c.m(0, 0); c.l(30, 60); c.l(60, 0); c.S()
    c.Q()
    c.q()                                                   # a uniform scale to compare: its pen stays round
    c.cm(2, 0, 0, 2, 0, 0)
    c.w(2)
    c.J(1)
    c.RG(0.2, 0.2, 0.2)
    c.m(15, 15); c.l(45, 40); c.S()
    c.Q()
    data, w = _finish(doc)
    desc = ("Strokes under matrices that are not uniform (ISO 32000-2, 8.5.3.1: the line width is in user space, so the "
            "pen is stretched and sheared with it): a polyline stretched 3 across with round caps and joins, a circle "
            "squashed to a third, a skewed square with miter joins, dashes under 0.5 by 2, a stroke turned and "
            "stretched with bevel joins and square caps, and one under a uniform scale of 2 to compare.")
    return _sample(doc, data, w, "vector-strokes-pen", desc)


# -- Type 3 glyphs that are more than filled shapes ---------------------------------------------------------------------

def _mask_bytes(rows):
    """An image mask's rows ('#' paints: sample 0 under the default Decode), each padded to whole bytes."""
    out = bytearray()
    for row in rows:
        line = bytearray((len(row) + 7) // 8)
        for i, ch in enumerate(row):
            if ch != "#":
                line[i >> 3] |= 0x80 >> (i & 7)
        out += line
    return bytes(out)


GLYPH_MASK = ["##########", "#........#", "#.######.#", "#.#....#.#", "#.#.##.#.#", "#.#.##.#.#",
              "#.#.##.#.#", "#.#.##.#.#", "#.#....#.#", "#.######.#", "#........#", "##########"]
GLYPH_IMAGE = bytes([220, 60, 50, 240, 200, 40, 60, 160, 80, 40, 90, 200,
                     250, 170, 40, 90, 200, 90, 220, 60, 160, 40, 40, 40,
                     60, 160, 80, 220, 60, 50, 240, 240, 240, 250, 170, 40,
                     40, 90, 200, 40, 40, 40, 90, 200, 90, 220, 60, 160])

TYPE3_DRAWN = {
    # d1, a stroked outline and a filled bar: both in the text's colour.
    "A": (600, b"600 0 0 0 600 700 d1\n70 w 1 j 110 80 m 300 620 l 490 80 l h S\n150 230 300 70 re f\n"),
    # d1, an image mask: the glyph is a bitmap painted in the text's colour.
    "B": (600, b"600 0 0 0 600 700 d1\nq 500 0 0 600 50 50 cm\nBI /W 10 /H 12 /IM true /BPC 1 ID "
               + _mask_bytes(GLYPH_MASK) + b"\nEI Q\n"),
    # d0, its own colours: a red bar and a blue stroked frame (the text's colour unused).
    "C": (700, b"700 0 d0\n0.85 0.15 0.15 rg 80 80 220 540 re f\n0.1 0.3 0.85 RG 40 w 380 120 m 620 120 l 620 580 l "
               b"380 580 l h S\n"),
    # d0, an image of its own: four by four colours.
    "D": (700, b"700 0 d0\nq 560 0 0 560 70 70 cm\nBI /W 4 /H 4 /CS /RGB /BPC 8 ID " + GLYPH_IMAGE + b"\nEI Q\n"),
    # d1, a filled shape alone: drawn as it always was.
    "E": (600, b"600 0 0 0 600 700 d1\n60 0 m 540 0 l 540 100 l 180 100 l 180 300 l 460 300 l 460 400 l 180 400 l "
               b"180 600 l 540 600 l 540 700 l 60 700 l h f\n"),
    "space": (300, b"300 0 d0\n"),
}


def type3_drawn(seed):
    doc = _doc("vector-type3-drawn", "1.4", seed)
    f = doc.font(Type3, TYPE3_DRAWN)
    p = doc.page(W, H)
    c = p.content
    c.BT()
    c.Tf(f, 44)
    c.rg(0.1, 0.5, 0.25)
    c.Tm(1, 0, 0, 1, 20, 220)
    c.show("ABCDE EDCBA")
    p.line("ABCDE EDCBA")
    c.rg(0.6, 0.15, 0.55)
    c.Tz(160)                                               # stretched across: the glyphs' strokes stretched too
    c.Tm(1, 0, 0, 1, 20, 130)
    c.show("ABCDE")
    p.line("ABCDE")
    c.Tz(100)
    c.rg(0.85, 0.45, 0.1)
    a = math.radians(12)
    c.Tm(math.cos(a), math.sin(a), -math.sin(a), math.cos(a), 30, 20)
    c.Tf(f, 36)
    c.show("BADCE")
    p.line("BADCE")
    c.ET()
    data, w = _finish(doc)
    desc = ("A Type 3 font (ISO 32000-2, 9.6.4) whose glyphs are more than filled shapes: a d1 glyph stroked and filled "
            "(in the text's colour), a d1 glyph that is an image mask (a bitmap in the text's colour), a d0 glyph in its "
            "own colours (a red bar, a blue frame), a d0 glyph holding an image (four by four colours), and a d1 glyph "
            "filled alone; shown in three colours, once stretched across (Tz 160) and once turned.")
    return _sample(doc, data, w, "vector-type3-drawn", desc)


# -- TrueType glyphs found by their standard Macintosh names -----------------------------------------------------------

def _tt_glyphs(order):
    """Each name of `order` as DejaVu Sans draws the glyph of that name (an empty glyph where it has none)."""
    src = TTFont(DEJAVU, recalcTimestamp=False)
    gs = src.getGlyphSet()
    hmtx = src["hmtx"]
    glyphs, metrics = {}, {}
    for name in order:
        pen = TTGlyphPen(None)
        if name in gs:
            rec = DecomposingRecordingPen(gs)
            gs[name].draw(rec)
            rec.replay(pen)
            metrics[name] = hmtx[name]
        else:
            metrics[name] = (src["head"].unitsPerEm // 2, 0)
        glyphs[name] = pen.glyph()
    return src, glyphs, metrics


def _mac_font(family, order, post_format):
    src, glyphs, metrics = _tt_glyphs(order)
    upem = src["head"].unitsPerEm
    fb = FontBuilder(upem, isTTF=True)
    fb.setupGlyphOrder(order)
    fb.setupCharacterMap({0x20: "space"})                   # no letter in the cmap: a reader finds them by name
    fb.setupGlyf(glyphs)
    fb.setupHorizontalMetrics(metrics)
    hhea = src["hhea"]
    fb.setupHorizontalHeader(ascent=hhea.ascent, descent=hhea.descent)
    fb.setupNameTable({"familyName": family, "styleName": "Regular", "psName": family})
    fb.setupOS2(sTypoAscender=hhea.ascent, sTypoDescender=hhea.descent, usWinAscent=hhea.ascent,
                usWinDescent=-hhea.descent)
    fb.setupPost(keepGlyphNames=post_format == 2)
    fb.font["post"].formatType = float(post_format)
    fb.font["head"].created = fb.font["head"].modified = 3786998400   # fixed, so the bytes repeat
    buf = io.BytesIO()
    fb.font.save(buf)
    data = buf.getvalue()
    check = TTFont(io.BytesIO(data))
    if check["post"].formatType != float(post_format) or check.getGlyphOrder() != list(order):
        raise ValueError(f"{family}: post format {check['post'].formatType}, its glyph order not kept")
    return data, check


class MacNamesTrueType(Font):
    """A simple TrueType font (/WinAnsiEncoding, not symbolic) whose cmap maps no letter, so each code's glyph is found by
    its name in the program's `post` table: format 1 (the 258 standard Macintosh names, in their order) or format 2 (the
    standard names by their indices, in an order of its own)."""

    kind = "truetype"

    def __init__(self, doc, resname, family, order, post_format):
        super().__init__(doc, resname)
        self.family = family
        self.data, self.program = _mac_font(family, order, post_format)
        self.upem = self.program["head"].unitsPerEm
        hhea = self.program["hhea"]
        self.ascent = float(_units(hhea.ascent, self.upem))
        self.descent = float(_units(hhea.descent, self.upem))
        self.encode, self.codes = simple_encoder("WinAnsiEncoding")
        self.used = {}

    def glyphs(self, text):
        out = []
        for ch in text:
            code = self.encode(ch)[0]
            name = self.codes[code][0]
            width = _units(self.program["hmtx"][name][0], self.upem)
            self.used[code] = width
            out.append(Glyph(bytes([code]), width, ch, code == 32, gid=self.program.getGlyphID(name)))
        return out

    def finish(self):
        self.base_font = f"{subset_tag(self.data)}+{self.family}"
        program = self.doc.pdf.add(Stream({"Length1": len(self.data), "Filter": N("FlateDecode")}, flate(self.data),
                                          self.data))
        head, hhea = self.program["head"], self.program["hhea"]
        descriptor = self.doc.pdf.add({
            "Type": N("FontDescriptor"), "FontName": N(self.base_font), "Flags": 32,
            "FontBBox": [_units(head.xMin, self.upem), _units(head.yMin, self.upem), _units(head.xMax, self.upem),
                         _units(head.yMax, self.upem)],
            "ItalicAngle": 0, "Ascent": _units(hhea.ascent, self.upem), "Descent": _units(hhea.descent, self.upem),
            "CapHeight": _units(hhea.ascent, self.upem), "StemV": 80, "FontFile2": program})
        first, last = min(self.used), max(self.used)
        d = {"Type": N("Font"), "Subtype": N("TrueType"), "BaseFont": N(self.base_font), "FirstChar": first,
             "LastChar": last, "Widths": [self.used.get(k, 0) for k in range(first, last + 1)],
             "Encoding": N("WinAnsiEncoding"), "FontDescriptor": descriptor}
        self.doc.pdf.put(self.ref, d)


def mac_names(seed):
    doc = _doc("vector-truetype-mac-names", "1.4", seed)
    standard = list(standardGlyphOrder[:94])                # .notdef to z, in the standard order
    letters = [n for n in standard[3:] if n not in ("exclam",)]
    own = [".notdef"] + list(reversed(letters))             # the same names, in an order of the font's own
    one = doc.font(MacNamesTrueType, "MacNamesOne", standard, 1)
    two = doc.font(MacNamesTrueType, "MacNamesTwo", own, 2)
    p = doc.page(W, H)
    c = p.content
    c.BT()
    lines = [(one, "Post format 1: delivery on time, 2026.", 22, 240), (one, "Quarterly report, page 7 of 12.", 18, 200),
             (two, "Post format 2: the budget is approved.", 22, 130), (two, "New office in May (room 4B).", 18, 90)]
    for font, text, size, y in lines:
        c.Tf(font, size)
        c.Tm(1, 0, 0, 1, 20, y)
        c.rg(0.1, 0.2, 0.45)
        c.show(text)
        p.line(text)
    c.ET()
    data, w = _finish(doc)
    desc = ("Two simple TrueType fonts (/WinAnsiEncoding, not symbolic) whose cmaps map the space alone, so each code's "
            "glyph is found by its name in the program's `post` table (ISO 32000-2, 9.6.5.4): one of format 1 (the "
            "standard Macintosh order, .notdef to z) and one of format 2 naming the same glyphs, in reverse, by their "
            "standard indices. The text is the encoding's (its glyph names).")
    sample = _sample(doc, data, w, "vector-truetype-mac-names", desc)
    sample.expected["pages"][0]["glyphIds"] = [{"text": t, "font": f.resname,
                                                "glyphs": [g.gid for g in f.glyphs(t)]} for f, t, _, _ in lines]
    return sample


# -- bare CFF fonts with the Expert encoding and charsets -------------------------------------------------------------

def _expert_charstring(gs, name, upem):
    """A glyph for an expert name: DejaVu Sans's glyph of the name it varies (A for Asmall, zero for zerooldstyle, f
    twice for ff), made smaller, raised or lowered; a box where DejaVu has none. In 1,000 units an em."""
    k = 1000 / upem
    base, scale, rise = name, 1.0, 0
    for suffix, (s, r) in (("oldstyle", (0.82, 0)), ("small", (0.74, 0)), ("superior", (0.55, 380)),
                           ("inferior", (0.55, -120))):
        if name.endswith(suffix):
            base, scale, rise = name[:-len(suffix)], s, r
            break
    parts = {"ff": ["f", "f"], "fi": ["f", "i"], "fl": ["f", "l"], "ffi": ["f", "f", "i"], "ffl": ["f", "f", "l"]}.get(
        base, [base])
    if base.endswith("small"):
        parts = [base]
    pieces = [b for b in parts if b in gs]
    if len(pieces) != len(parts):
        width = 520
        pen = T2CharStringPen(width, None)
        pen.moveTo((60, 0)); pen.lineTo((460, 0)); pen.lineTo((460, 600)); pen.lineTo((60, 600)); pen.closePath()
        pen.moveTo((140, 80)); pen.lineTo((140, 520)); pen.lineTo((380, 520)); pen.lineTo((380, 80)); pen.closePath()
        return pen.getCharString(), width
    advance = sum(gs[b].width for b in pieces) * k * scale
    pen = T2CharStringPen(round(advance), None)
    x = 0.0
    for b in pieces:
        rec = DecomposingRecordingPen(gs)
        gs[b].draw(rec)
        rec.replay(TransformPen(pen, (k * scale, 0, 0, k * scale, x, rise)))
        x += gs[b].width * k * scale
    return pen.getCharString(), round(advance)


def _expert_cff(family, charset):
    """A CFF program (its table's bytes) whose glyphs are `charset` (a predefined charset's prefix, written as that
    charset's number) and whose built-in encoding is the predefined Expert encoding (1)."""
    src = TTFont(DEJAVU, recalcTimestamp=False)
    gs = src.getGlyphSet()
    upem = src["head"].unitsPerEm
    strings, widths = {}, {}
    for name in charset:
        if name == ".notdef":
            pen = T2CharStringPen(500, None)
            strings[name], widths[name] = pen.getCharString(), 500
        elif name == "space":
            pen = T2CharStringPen(320, None)
            strings[name], widths[name] = pen.getCharString(), 320
        else:
            strings[name], widths[name] = _expert_charstring(gs, name, upem)
    fb = FontBuilder(1000, isTTF=False)
    fb.setupGlyphOrder(list(charset))
    fb.setupCFF(family, {"FullName": family, "FamilyName": family}, strings, {})
    top = fb.font["CFF "].cff.topDictIndex[0]
    top.Encoding = "ExpertEncoding"
    fb.setupCharacterMap({})                                # the PDF's codes reach glyphs through the CFF's own encoding
    fb.setupHorizontalMetrics({n: (widths[n], 0) for n in charset})
    fb.setupHorizontalHeader(ascent=800, descent=-200)
    fb.setupNameTable({"familyName": family, "styleName": "Regular"})
    fb.setupOS2()
    fb.setupPost()
    buf = io.BytesIO()
    fb.font.save(buf)
    otf = TTFont(io.BytesIO(buf.getvalue()))
    cff = otf.getTableData("CFF ")
    return cff, widths


class ExpertCff(Font):
    """A simple font (/Type1) whose program is a bare CFF (FontFile3 /Type1C) with the predefined Expert encoding and
    an Expert or Expert Subset charset, and no /Encoding: its codes reach glyphs through the program's own encoding."""

    kind = "type1c"

    def __init__(self, doc, resname, family, charset):
        super().__init__(doc, resname)
        self.family = family
        self.charset = charset
        self.data, self.widths = _expert_cff(family, charset)
        self.ascent, self.descent = 800.0, -200.0
        self.used = {}

    def glyphs(self, names):
        """Glyphs by their expert names (the Expert encoding gives each its code)."""
        out = []
        for name in names:
            code = EXPERT_CODES[name]
            self.used[code] = self.widths[name]
            out.append(Glyph(bytes([code]), self.widths[name], glyph_text(name), code == 32,
                             gid=self.charset.index(name)))
        return out

    def finish(self):
        self.base_font = f"{subset_tag(self.data)}+{self.family}"
        program = self.doc.pdf.add(Stream({"Subtype": N("Type1C"), "Filter": N("FlateDecode")}, flate(self.data),
                                          self.data))
        descriptor = self.doc.pdf.add({
            "Type": N("FontDescriptor"), "FontName": N(self.base_font), "Flags": 4, "FontBBox": [-50, -250, 1300, 950],
            "ItalicAngle": 0, "Ascent": 800, "Descent": -200, "CapHeight": 700, "StemV": 80, "FontFile3": program})
        first, last = min(self.used), max(self.used)
        d = {"Type": N("Font"), "Subtype": N("Type1"), "BaseFont": N(self.base_font), "FirstChar": first,
             "LastChar": last, "Widths": [self.used.get(k, 0) for k in range(first, last + 1)],
             "FontDescriptor": descriptor}
        self.doc.pdf.put(self.ref, d)


# The predefined Expert encoding (Adobe Technical Note #5176, appendix C; pdf.js's ExpertEncoding): a code a name.
EXPERT_ENCODING = {
    0x20: "space", 0x21: "exclamsmall", 0x22: "Hungarumlautsmall", 0x24: "dollaroldstyle", 0x25: "dollarsuperior",
    0x26: "ampersandsmall", 0x27: "Acutesmall", 0x28: "parenleftsuperior", 0x29: "parenrightsuperior",
    0x2A: "twodotenleader", 0x2B: "onedotenleader", 0x2C: "comma", 0x2D: "hyphen", 0x2E: "period", 0x2F: "fraction",
    0x30: "zerooldstyle", 0x31: "oneoldstyle", 0x32: "twooldstyle", 0x33: "threeoldstyle", 0x34: "fouroldstyle",
    0x35: "fiveoldstyle", 0x36: "sixoldstyle", 0x37: "sevenoldstyle", 0x38: "eightoldstyle", 0x39: "nineoldstyle",
    0x3A: "colon", 0x3B: "semicolon", 0x3C: "commasuperior", 0x3D: "threequartersemdash", 0x3E: "periodsuperior",
    0x3F: "questionsmall", 0x41: "asuperior", 0x42: "bsuperior", 0x43: "centsuperior", 0x44: "dsuperior",
    0x45: "esuperior", 0x49: "isuperior", 0x4C: "lsuperior", 0x4D: "msuperior", 0x4E: "nsuperior", 0x4F: "osuperior",
    0x52: "rsuperior", 0x53: "ssuperior", 0x54: "tsuperior", 0x56: "ff", 0x57: "fi", 0x58: "fl", 0x59: "ffi",
    0x5A: "ffl", 0x5B: "parenleftinferior", 0x5D: "parenrightinferior", 0x5E: "Circumflexsmall",
    0x5F: "hyphensuperior", 0x60: "Gravesmall", **{0x61 + i: f"{chr(65 + i)}small" for i in range(26)},
    0x7B: "colonmonetary", 0x7C: "onefitted", 0x7D: "rupiah", 0x7E: "Tildesmall", 0xA1: "exclamdownsmall",
    0xA2: "centoldstyle", 0xA3: "Lslashsmall", 0xA6: "Scaronsmall", 0xA7: "Zcaronsmall", 0xA8: "Dieresissmall",
    0xA9: "Brevesmall", 0xAA: "Caronsmall", 0xAC: "Dotaccentsmall", 0xAF: "Macronsmall", 0xB2: "figuredash",
    0xB3: "hypheninferior", 0xB6: "Ogoneksmall", 0xB7: "Ringsmall", 0xB8: "Cedillasmall", 0xBC: "onequarter",
    0xBD: "onehalf", 0xBE: "threequarters", 0xBF: "questiondownsmall", 0xC0: "oneeighth", 0xC1: "threeeighths",
    0xC2: "fiveeighths", 0xC3: "seveneighths", 0xC4: "onethird", 0xC5: "twothirds", 0xC8: "zerosuperior",
    0xC9: "onesuperior", 0xCA: "twosuperior", 0xCB: "threesuperior", 0xCC: "foursuperior", 0xCD: "fivesuperior",
    0xCE: "sixsuperior", 0xCF: "sevensuperior", 0xD0: "eightsuperior", 0xD1: "ninesuperior", 0xD2: "zeroinferior",
    0xD3: "oneinferior", 0xD4: "twoinferior", 0xD5: "threeinferior", 0xD6: "fourinferior", 0xD7: "fiveinferior",
    0xD8: "sixinferior", 0xD9: "seveninferior", 0xDA: "eightinferior", 0xDB: "nineinferior", 0xDC: "centinferior",
    0xDD: "dollarinferior", 0xDE: "periodinferior", 0xDF: "commainferior", 0xE0: "Agravesmall", 0xE1: "Aacutesmall",
    0xE2: "Acircumflexsmall", 0xE3: "Atildesmall", 0xE4: "Adieresissmall", 0xE5: "Aringsmall", 0xE6: "AEsmall",
    0xE7: "Ccedillasmall", 0xE8: "Egravesmall", 0xE9: "Eacutesmall", 0xEA: "Ecircumflexsmall", 0xEB: "Edieresissmall",
    0xEC: "Igravesmall", 0xED: "Iacutesmall", 0xEE: "Icircumflexsmall", 0xEF: "Idieresissmall", 0xF0: "Ethsmall",
    0xF1: "Ntildesmall", 0xF2: "Ogravesmall", 0xF3: "Oacutesmall", 0xF4: "Ocircumflexsmall", 0xF5: "Otildesmall",
    0xF6: "Odieresissmall", 0xF7: "OEsmall", 0xF8: "Oslashsmall", 0xF9: "Ugravesmall", 0xFA: "Uacutesmall",
    0xFB: "Ucircumflexsmall", 0xFC: "Udieresissmall", 0xFD: "Yacutesmall", 0xFE: "Thornsmall", 0xFF: "Ydieresissmall",
}
EXPERT_CODES = {name: code for code, name in EXPERT_ENCODING.items()}
if len(EXPERT_CODES) != len(EXPERT_ENCODING) or any(n not in EXPERT_CODES for n in cffIExpertStrings[1:]):
    raise ValueError("the Expert encoding table does not name every glyph of the Expert charset once")


def cff_expert(seed):
    doc = _doc("vector-cff-expert", "1.4", seed)
    expert = doc.font(ExpertCff, "ExpertOldstyle", list(cffIExpertStrings))
    subset = doc.font(ExpertCff, "ExpertFigures", list(cffExpertSubsetStrings))
    figures = [f"{n}oldstyle" for n in ("one", "nine", "eight", "four")]
    smallcaps = [f"{ch}small" for ch in "REPORT"]
    lines = [
        (expert, smallcaps + ["space"] + figures + ["comma", "space", "ff", "fi", "fl", "ffi", "ffl"], 26, 235),
        (expert, ["onehalf", "space", "onequarter", "space", "threequarters", "space", "dollaroldstyle"]
         + [f"{n}oldstyle" for n in ("two", "zero", "two", "six")] + ["period", "space", "asuperior", "esuperior"],
         24, 180),
        (subset, ["dollaroldstyle"] + [f"{n}oldstyle" for n in ("one", "two", "five", "zero")]
         + ["space", "onesuperior", "twosuperior", "threesuperior", "space", "oneeighth", "space", "ff", "fi"], 26, 110),
        (subset, ["zeroinferior", "oneinferior", "space", "parenleftsuperior", "nsuperior", "parenrightsuperior", "space",
                  "colon", "semicolon", "space", "figuredash", "space", "twothirds"], 24, 55),
    ]
    p = doc.page(W, H)
    c = p.content
    c.BT()
    for font, names, size, y in lines:
        c.Tf(font, size)
        c.Tm(1, 0, 0, 1, 20, y)
        c.rg(0.15, 0.15, 0.35)
        glyphs = font.glyphs(names)
        rec = c.show(glyphs=glyphs)
        p.line(rec["text"])
    c.ET()
    data, w = _finish(doc)
    desc = ("Two bare CFF fonts (FontFile3 /Type1C, no /Encoding) whose built-in encoding is the predefined Expert "
            "encoding (Top DICT Encoding 1), one with the Expert charset (166 glyphs: small capitals, old-style figures, "
            "fractions, ligatures) and one with the Expert Subset charset (87 glyphs), each charset written as its "
            "number (1, 2). The text is the glyph names' by the Adobe Glyph List (small capitals and old-style figures "
            "in its private-use code points).")
    sample = _sample(doc, data, w, "vector-cff-expert", desc)
    sample.expected["pages"][0]["glyphIds"] = [{"font": f.resname, "glyphs": [f.charset.index(n) for n in names]}
                                               for f, names, _, _ in lines]
    return sample


# -- tiling cells larger than their /BBox -------------------------------------------------------------------------------

TILE_IMAGE = bytes([230, 80, 60, 60, 120, 220, 250, 210, 70, 40, 160, 90])


def tiling_bbox(seed):
    # Every place on the reference's pixels (100 dpi: 0.72 point a pixel, the page 400 pixels high), each box a little
    # under whole pixels: the comparison then measures the cells, not where a renderer rounds a half pixel.
    px = 0.72
    doc = _doc("vector-tiling-bbox", "1.6", seed)
    p = doc.page(W, 400 * px)
    c = p.content

    def fill(x0, top, x1, bottom):                          # a rectangle in pixels from the top left, filled
        c.re(x0 * px, (400 - bottom) * px, (x1 - x0) * px, (bottom - top) * px)
        c.f()

    cell = b"0.85 0.2 0.2 rg 0 0 36 36 re f 0.15 0.35 0.8 rg 8 8 m 33 8 l 20.5 33 l h f\n"   # reaches past its box
    upright = TilingPattern(doc, 1, [0, 0, 28.7, 28.7], 43.2, 43.2, cell)
    turned = TilingPattern(doc, 1, [0, 0, 28.7, 28.7], 43.2, 43.2, cell, matrix=[0.866, 0.5, -0.5, 0.866, 0, 0])
    spill_cell = (b"0.2 0.6 0.35 rg 23.8 10.8 m 23.8 17.98 17.98 23.8 10.8 23.8 c 3.62 23.8 -2.2 17.98 -2.2 10.8 c -2.2 3.62 "
                  b"3.62 -2.2 10.8 -2.2 c 17.98 -2.2 23.8 3.62 23.8 10.8 c f\n")
    spill = TilingPattern(doc, 1, [-1.4, -1.4, 23, 23], 21.6, 21.6, spill_cell, matrix=[0.9659, 0.2588, -0.2588, 0.9659, 0, 0])
    halves = TilingPattern(doc, 2, [0, 0, 21.5, 30.1], 30.24, 30.24, b"15.12 0 m 30.24 15.12 l 15.12 30.24 l 0 15.12 l h f\n")
    imaged = TilingPattern(doc, 1, [0, 0, 25.85, 25.85], 36, 36,
                           b"q 25.85 0 0 25.85 0 0 cm BI /W 2 /H 2 /CS /RGB /BPC 8 ID " + TILE_IMAGE + b"\nEI Q\n")
    c.cs(None, PatternSpace())
    c.scn_pattern("P1", upright)
    fill(20, 20, 264, 190)
    c.scn_pattern("P2", turned)
    fill(290, 20, 534, 190)
    c.scn_pattern("P3", spill)
    fill(20, 210, 188, 380)
    c.cs("CSp", PatternSpace(RGB))
    c.scn_pattern("P4", halves, comps=(0.55, 0.2, 0.65))
    fill(205, 210, 360, 380)
    c.cs(None, PatternSpace())
    c.scn_pattern("P5", imaged)
    fill(380, 210, 534, 380)
    data, w = _finish(doc)
    desc = ("Tiling patterns whose cells draw past their /BBox, which clips them (ISO 32000-2, 8.7.3.1): an upright cell "
            "of a box of 28.7 points every 43.2 (a 36-point square cut to the box, gaps between), the same turned 30 "
            "degrees, a turned cell whose box reaches past its steps (a circle cut to the box, overlapping its "
            "neighbours), an uncoloured cell narrower than its diamond (cut in half), and a cell holding an image. The "
            "places fall on the reference's pixels (100 dpi) and the boxes a little under whole pixels, so a comparison "
            "measures the cells, not a half pixel's rounding.")
    return _sample(doc, data, w, "vector-tiling-bbox", desc)


# -- vertical text with position vectors of its own ---------------------------------------------------------------------

class VerticalW2(Type0):
    """Type0 with Identity-V whose /W2 gives some glyphs their own vertical advance and position vector (w1y, vx, vy,
    thousandths): their outlines drawn from the vertical origin moved by (vx, vy), their boxes across the column from
    -vx to the width less vx."""

    def __init__(self, doc, resname, path, w2):
        super().__init__(doc, resname, path, encoding="Identity-V", tu_style="bfchar", dw2=(880, -1000))
        self.w2_chars = w2  # character -> (w1y, vx, vy)
        self.cid_w2 = {}

    def glyphs(self, text):
        out = super().glyphs(text)
        for ch, g in zip(text, out):
            if ch in self.w2_chars:
                self.cid_w2[int.from_bytes(g.code, "big")] = self.w2_chars[ch]
        return out

    def w2_of(self, g):
        return self.cid_w2.get(int.from_bytes(g.code, "big"))

    def finish(self):
        super().finish()
        font = self.doc.pdf.get(self.ref)
        cid_font = self.doc.pdf.get(font["DescendantFonts"][0])
        w2 = []
        for cid in sorted(self.cid_w2):
            w2 += [cid, list(self.cid_w2[cid])]
        cid_font["W2"] = w2


def vertical_w2(seed):
    doc = _doc("vector-vertical-w2", "1.5", seed)
    f = doc.font(VerticalW2, DEJAVU, {"R": (-1100, 200, 900), "e": (-900, 420, 760), "p": (-1000, 150, 880),
                                       "o": (-1000, 310, 820), "t": (-1200, 250, 950), "2": (-950, 380, 840)})
    p = doc.page(W, H)
    c = p.content
    c.BT()
    for k, (text, size) in enumerate((("Report", 22), ("Notes 2026", 18), ("Draft", 20))):
        c.Tf(f, size)
        c.Tm(1, 0, 0, 1, 280 - 110 * k, 270)                    # columns from the right, as vertical text is read
        c.rg(0.2, 0.25 + 0.2 * k, 0.6)
        c.show(text)
        p.line(text, direction="ttb")
    c.ET()
    data, w = _finish(doc)
    desc = ("Vertical text (a composite font, Identity-V) whose /W2 gives some glyphs a vertical advance and a position "
            "vector of their own (ISO 32000-2, 9.7.4.3: [cid [w1y vx vy]]), the others /DW2 [880 -1000] and half their "
            "width across: each glyph is drawn from its vertical origin moved by its vector, and its box spans -vx to "
            "its width less vx across the column.")
    return _sample(doc, data, w, "vector-vertical-w2", desc)


def build(seed):
    return [groups(seed), strokes_pen(seed), type3_drawn(seed), mac_names(seed), cff_expert(seed), tiling_bbox(seed),
            vertical_w2(seed)]
