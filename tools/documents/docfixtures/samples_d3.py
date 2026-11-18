"""D3, text: fonts to Unicode by every route (ToUnicode, base encodings and /Differences with glyph names, the embedded
program's cmap, nothing), widths and boxes (standard 14, TrueType, composite, Type 3, vertical), the text state (TJ, Tc,
Tw, Tz, TL, Ts, Tr), /ActualText, right-to-left and mixed-direction lines, rotated pages, CropBox and text matrices.

Right-to-left and mixed lines are written as PDF writers write them: each directional run shaped (HarfBuzz) into its
glyphs in visual order, the runs placed left to right; a run's text is its glyphs' text in content order, a line's text
is logical (Unicode's bidirectional algorithm; the visual order of every line is checked against python-bidi)."""
from bidi import get_display

from .common import AMIRI, DEJAVU, Sample, base_expected, helvetica
from .document import Doc
from .fonts import Standard14, TrueTypeSimple, Type0, Type3

D3 = ["D3"]


def _sample(doc, data, w, name, kind, desc, phases=D3, files=None):
    exp = base_expected(doc, data, name, kind, phases, desc, w, objects=False)
    return Sample(name, "d3", kind, phases, desc, data, exp, files)


def _finish(doc, kind="table"):
    doc.info = None
    doc.finish()
    return doc.write(kind)


STD14 = [("Helvetica", "Helvetica: the meeting starts at 10:30."),
         ("Helvetica-Bold", "Helvetica Bold: total sales 1,250 units."),
         ("Helvetica-Oblique", "Helvetica Oblique: draft for review."),
         ("Helvetica-BoldOblique", "Helvetica Bold Oblique: urgent."),
         ("Times-Roman", "Times Roman: the annual report is ready."),
         ("Times-Bold", "Times Bold: budget approved."),
         ("Times-Italic", "Times Italic: see the appendix."),
         ("Times-BoldItalic", "Times Bold Italic: final version."),
         ("Courier", "Courier: order 4711 shipped."),
         ("Courier-Bold", "Courier Bold: status = OK"),
         ("Courier-Oblique", "Courier Oblique: path /var/log"),
         ("Courier-BoldOblique", "Courier Bold Oblique: done")]


def standard14(seed):
    doc = Doc("standard14", "1.4", seed)
    p = doc.page(420, 340)
    c = p.content
    fonts = {name: doc.font(Standard14, name) for name, _ in STD14}
    c.BT()
    c.Td(24, 310)
    for i, (name, text) in enumerate(STD14):
        if i:
            c.Td(0, -20)
        c.Tf(fonts[name], 11)
        c.show(text)
        p.line(text)
    helv = fonts["Helvetica"]
    sym = doc.font(Standard14, "Symbol", encoding=None)
    zapf = doc.font(Standard14, "ZapfDingbats", encoding=None)
    c.Td(0, -20)
    c.Tf(helv, 11)
    c.show("Symbol: ")
    c.Tf(sym, 11)
    c.show("αβγ δπσ = 1 + 2")
    p.line("Symbol: αβγ δπσ = 1 + 2")
    c.Td(0, -20)
    c.Tf(helv, 11)
    c.show("ZapfDingbats: ")
    c.Tf(zapf, 11)
    c.show("✔ ✘ ★ ● ■ ◆")
    p.line("ZapfDingbats: ✔ ✘ ★ ● ■ ◆")
    c.ET()
    data, w = _finish(doc)
    desc = ("The standard 14 fonts, not embedded: the twelve Latin ones with WinAnsiEncoding and no /Widths (the AFM "
            "widths apply), Symbol and ZapfDingbats with their built-in encodings.")
    return _sample(doc, data, w, "standard14", "standard14", desc)


def encodings(seed):
    doc = Doc("encodings", "1.4", seed)
    p = doc.page(420, 300)
    c = p.content
    mac = doc.font(Standard14, "Helvetica", encoding="MacRomanEncoding")
    std = doc.font(Standard14, "Helvetica", encoding="StandardEncoding")
    win = doc.font(Standard14, "Helvetica", encoding="WinAnsiEncoding")
    diffs = {0x80: "uni2192", 0x81: "f_f", 0x82: "a.sc", 0x83: "Euro", 0x84: "g17", 0x85: "uni00410042",
             0x86: "u1F4C8"}
    dif = doc.font(Standard14, "Helvetica", base_encoding="WinAnsiEncoding", differences=diffs,
                   widths={0x80: 584, 0x81: 556, 0x82: 556, 0x83: 556, 0x84: 500, 0x85: 1334, 0x86: 800})
    c.BT()
    c.Td(24, 260)
    c.Tf(mac, 12)
    line = "MacRoman: Café crème brûlée – 12 items"
    c.show(line)
    p.line(line)
    c.Td(0, -24)
    c.Tf(std, 12)
    line = "Standard: It’s ‘ready’ — ship it"
    c.show(line)
    p.line(line)
    c.Td(0, -24)
    c.Tf(win, 12)
    line = "WinAnsi: “Smart” quotes – 25 £ • done"
    c.show(line)
    p.line(line)
    c.Td(0, -24)
    c.Tf(dif, 12)
    codes = list(b"Differences: Next ") + [0x80, 0x20] + list(b"o") + [0x81] + list(b"ers ") + [0x83] + list(b"5 ") \
        + [0x84, 0x20, 0x82, 0x20, 0x85, 0x20, 0x86]
    c.show(glyphs=dif.code_glyphs(codes))
    text = "Differences: Next → offers €5 � a AB \U0001F4C8"
    p.line(text)
    c.ET()
    data, w = _finish(doc)
    desc = ("Simple fonts' encodings: MacRoman, Standard and WinAnsi base encodings, and /Differences over WinAnsi with "
            "glyph names by the Adobe Glyph List rules (uni2192, f_f, a.sc, Euro, uni00410042, u1F4C8) and one that "
            "names nothing (g17: U+FFFD and one unmapped glyph).")
    return _sample(doc, data, w, "encodings", "encodings", desc)


def truetype_simple(seed):
    doc = Doc("truetype-simple", "1.4", seed)
    p = doc.page(420, 300)
    c = p.content
    plain = doc.font(TrueTypeSimple, DEJAVU, to_unicode=False)
    mapped = doc.font(TrueTypeSimple, DEJAVU, to_unicode=True)
    c.BT()
    c.Td(30, 250)
    c.Tf(plain, 14)
    line = "Résumé of the project plan"
    c.show(line)
    p.line(line)
    c.Td(0, -28)
    c.Tf(mapped, 14)
    line = "Invoice total: 1,250.00 EUR (paid)"
    c.show(line)
    p.line(line)
    c.ET()
    data, w = _finish(doc)
    desc = ("Embedded TrueType (a DejaVu Sans subset) as simple fonts with WinAnsiEncoding: one without ToUnicode (codes "
            "to glyph names to Unicode), one with it.")
    return _sample(doc, data, w, "truetype-simple", "truetype", desc)


def _line(c, p, font, size, x, y, segments, base="ltr", right=None):
    """One line of directional segments (logical order) shaped and placed in visual order; `right` aligns its end."""
    order = list(reversed(segments)) if base == "rtl" else segments
    shaped = [font.shape(text, d) for text, d in order]
    width = sum(g.width for gl in shaped for g in gl) / 1000 * size
    if right is not None:
        x = right - width
    c.Tm(1, 0, 0, 1, x, y)
    for gl in shaped:
        c.show(glyphs=gl)
    logical = "".join(t for t, d in segments)
    visual = "".join(t[::-1] if d == "rtl" else t for t, d in order)
    ref = get_display(logical, base_dir="R" if base == "rtl" else "L")
    if ref != visual:
        raise ValueError(f"visual order differs from python-bidi: {visual!r} vs {ref!r}")
    p.line(logical, base)


ENGLISH = ["Project status: on track", "The office opens at 8:45 every weekday.", "Fifty files were checked."]


def type0_identity(seed):
    doc = Doc("type0-identity", "1.5", seed)
    p = doc.page(420, 300)
    c = p.content
    f = doc.font(Type0, DEJAVU, tu_style="mixed", w_style="mixed")
    c.BT()
    c.Tf(f, 14)
    for i, line in enumerate(ENGLISH):
        _line(c, p, f, 14, 30, 250 - 28 * i, [(line, "ltr")])
    c.ET()
    data, w = _finish(doc)
    desc = ("A composite font (Type0, Identity-H, CIDFontType2, CIDToGIDMap Identity) embedding a DejaVu Sans subset, "
            "with a ToUnicode CMap (bfchar and bfrange, ranges with arrays; a ligature glyph maps to two characters).")
    return _sample(doc, data, w, "type0-identity", "type0", desc)


ARABIC = ["تقرير المبيعات الشهري",
          "اجتماع الفريق يوم الثلاثاء",
          "تم شحن الطلبات إلى العملاء"]


def _reverse_cmap_texts(doc, font, page_index):
    """Without ToUnicode a reader maps each glyph back through the embedded cmap (lowest code point) or gives U+FFFD;
    the expected runs and lines are recomputed from the subset program's cmap."""
    rev = font.reverse_cmap()
    p = doc.pages[page_index]
    # These pages hold one run a line.
    for i, rec in enumerate(sorted(p.runs, key=lambda r: r["order"])):
        if rec["font"] is not font:
            continue
        for g in rec["glyphs"]:
            g.text = rev.get(g.gid, "�")
        rec["text"] = "".join(g.text for g in rec["glyphs"])
        line = p.lines[i]
        line["text"] = rec["text"][::-1] if line["direction"] == "rtl" else rec["text"]


def type0_no_tounicode(seed):
    doc = Doc("type0-no-tounicode", "1.5", seed)
    f = doc.font(Type0, DEJAVU, to_unicode=False)
    p = doc.page(420, 300)
    c = p.content
    c.BT()
    c.Tf(f, 14)
    for i, line in enumerate(["Quarterly results", "Warehouse stock: 340 pallets"]):
        glyphs = f.shape(line, "ltr")
        c.Tm(1, 0, 0, 1, 30, 250 - 28 * i)
        c.show(glyphs=glyphs)
        p.line(line)
    c.ET()
    p2 = doc.page(420, 300)
    c = p2.content
    c.BT()
    c.Tf(f, 16)
    for i, line in enumerate(ARABIC[:2]):
        _line(c, p2, f, 16, 0, 250 - 30 * i, [(line, "rtl")], "rtl", right=390)
    c.ET()
    doc.info = None
    doc.finish()
    _reverse_cmap_texts(doc, f, 0)
    _reverse_cmap_texts(doc, f, 1)
    data, w = doc.write("table")
    desc = ("A composite font without ToUnicode: each glyph's text comes from the embedded program's own cmap reversed "
            "(the lowest code point mapping to the glyph, so an isolated letter gives its base character). Page 1: "
            "Latin. Page 2: Arabic shaped to contextual forms, which the subset's cmap maps from their presentation "
            "forms (U+FExx): that is the text a reader gets. type0-no-cmap covers glyphs with no code point.")
    s = _sample(doc, data, w, "type0-no-tounicode", "type0", desc)
    s.expected["unicodeRule"] = "embedded cmap reversed, lowest code point; else U+FFFD"
    return s


def type0_no_cmap(seed):
    doc = Doc("type0-no-cmap", "1.5", seed)
    f = doc.font(Type0, DEJAVU, to_unicode=False, keep_cmap=False)
    p = doc.page(420, 300)
    c = p.content
    c.BT()
    c.Tf(f, 14)
    line = "Unreadable without OCR"
    glyphs = f.shape(line, "ltr")
    c.Tm(1, 0, 0, 1, 30, 250)
    c.show(glyphs=glyphs)
    p.line(line)
    c.ET()
    doc.info = None
    doc.finish()
    _reverse_cmap_texts(doc, f, 0)
    data, w = doc.write("table")
    desc = ("A composite font with neither ToUnicode nor a cmap in its embedded program: every glyph is U+FFFD and "
            "counted in the page's issues (the page to send to OCR).")
    return _sample(doc, data, w, "type0-no-cmap", "type0", desc)


def cid_map_stream(seed):
    doc = Doc("cidtogidmap-stream", "1.5", seed)
    mapped = doc.font(Type0, DEJAVU, cid_map="stream", to_unicode=True, tu_style="bfrange")
    bare = doc.font(Type0, DEJAVU, cid_map="stream", to_unicode=False)
    p = doc.page(420, 300)
    c = p.content
    c.BT()
    for i, (font, line) in enumerate([(mapped, "Shipping label printed"), (bare, "Returns processed today")]):
        c.Tf(font, 14)
        c.Tm(1, 0, 0, 1, 30, 250 - 28 * i)
        c.show(glyphs=font.glyphs(line))
        p.line(line)
    c.ET()
    doc.info = None
    doc.finish()
    _reverse_cmap_texts(doc, bare, 0)
    data, w = doc.write("table")
    desc = ("Composite fonts whose CIDs are not glyph ids (CID = the character's code point, a CIDToGIDMap stream "
            "gives the glyph): one with a ToUnicode CMap (bfrange), one without (CID to glyph through the map, then the "
            "embedded cmap reversed).")
    return _sample(doc, data, w, "cidtogidmap-stream", "type0", desc)


TYPE3_PROCS = {
    "A": (600, b"600 0 0 0 600 700 d1\n50 0 m 300 700 l 550 0 l 430 0 l 300 380 l 170 0 l h f\n"),
    "B": (600, b"600 0 0 0 600 700 d1\n60 0 480 700 re f\n"),
    "C": (600, b"600 0 0 0 600 700 d1\n60 0 m 540 0 l 540 120 l 180 120 l 180 580 l 540 580 l 540 700 l 60 700 l h f\n"),
    "D": (600, b"600 0 0 0 600 700 d1\n60 0 m 400 0 l 540 350 l 400 700 l 60 700 l h f\n"),
    "E": (600, b"600 0 0 0 600 700 d1\n60 0 m 540 0 l 540 100 l 180 100 l 180 300 l 460 300 l 460 400 l 180 400 l "
               b"180 600 l 540 600 l 540 700 l 60 700 l h f\n"),
    "checkmark": (700, b"700 0 0 0 700 700 d1\n40 300 m 220 80 l 660 640 l 600 700 l 220 220 l 100 360 l h f\n"),
    "space": (300, b"300 0 d0\n"),
}


def type3(seed):
    doc = Doc("type3", "1.4", seed)
    f = doc.font(Type3, TYPE3_PROCS)
    p = doc.page(420, 300)
    c = p.content
    c.BT()
    c.Tf(f, 24)
    c.Td(30, 230)
    for i, line in enumerate(["ACE BED CAB ✓", "BEAD DECADE ✓"]):
        if i:
            c.Td(0, -40)
        c.show(line)
        p.line(line)
    c.ET()
    data, w = _finish(doc)
    desc = ("A Type 3 font: glyphs drawn by their own content streams (d1, filled shapes), FontMatrix 0.001, named by "
            "/Differences (A-E, checkmark, space) so the glyph names give the text; boxes from its FontBBox.")
    return _sample(doc, data, w, "type3", "type3", desc)


def tj_spacing(seed):
    doc = Doc("text-state", "1.4", seed)
    f = helvetica(doc)
    p = doc.page(420, 340)
    c = p.content
    c.BT()
    c.Tf(f, 14)
    c.Td(24, 300)
    c.TJ(["W", 60, "ord", -20, "s ", 40, "with kerning"])
    p.line("Words with kerning")
    c.Td(0, -26)
    c.TJ(["Annual", -320, "budget", -320, "review"])
    p.line("Annual budget review")
    c.Td(0, -26)
    c.Tc(1.5)
    c.show("Spaced letters")
    p.line("Spaced letters")
    c.Tc(0)
    c.Td(0, -26)
    c.Tw(6)
    c.show("Wide word gaps here")
    p.line("Wide word gaps here")
    c.Tw(0)
    c.Td(0, -26)
    c.Tz(70)
    c.show("Condensed to seventy percent")
    p.line("Condensed to seventy percent")
    c.Tz(100)
    c.Td(0, -26)
    c.TL(26)
    c.show("First line, then T*")
    p.line("First line, then T*")
    c.Tstar()
    c.show("Second line after T*")
    p.line("Second line after T*")
    c.show("Third line by the quote operator", op="'")
    p.line("Third line by the quote operator")
    c.show("Fourth line by double quote", op='"', aw=2, ac=0.5)
    p.line("Fourth line by double quote")
    c.Tw(0)
    c.Tc(0)
    c.Td(0, -26)
    c.show("Floor area 120 m")
    c.Tf(f, 9)
    c.Ts(5)
    c.show("2")
    c.Ts(0)
    c.Tf(f, 14)
    c.show(" in total")
    p.line("Floor area 120 m2 in total")
    c.ET()
    data, w = _finish(doc)
    desc = ("The text state: TJ with kerning (small offsets, no space) and with word gaps (offsets of 0.32 em, no space "
            "characters: a reader inserts the spaces), Tc, Tw, Tz, TL with T*, the ' and \" operators, and a "
            "superscript by Ts. Small offsets are at most 0.06 em, word gaps at least 0.3 em.")
    return _sample(doc, data, w, "text-state", "text-state", desc)


def render_modes(seed):
    doc = Doc("render-modes", "1.4", seed)
    f = helvetica(doc)
    p = doc.page(420, 300)
    c = p.content
    words = ["fill", "stroke", "fill and stroke", "invisible", "fill, clip", "stroke, clip", "both, clip",
             "clip only"]
    c.emit("0.2 0.3 0.6 rg 0.6 0.1 0.1 RG 0.5 w")
    for mode, word in enumerate(words):
        c.BT()
        c.Tf(f, 16)
        c.Tr(mode)
        c.Td(30, 260 - 30 * mode)
        text = f"Mode {mode}: {word}"
        c.show(text)
        p.line(text)
        c.ET()
    data, w = _finish(doc)
    desc = "One line in each text render mode 0-7: modes 3 (invisible) and 7 (clip only) are hidden runs."
    return _sample(doc, data, w, "render-modes", "render-modes", desc)


def actual_text(seed):
    doc = Doc("actual-text", "1.4", seed)
    f = helvetica(doc)
    lig = doc.font(Standard14, "Helvetica", base_encoding="WinAnsiEncoding", differences={0x80: "fi"},
                   widths={0x80: 556})
    p = doc.page(420, 300)
    c = p.content
    c.BT()
    c.Tf(f, 16)
    c.Td(30, 240)
    c.show("Of")
    with c.actual_text("fi"):
        c.Tf(lig, 16)
        c.show(glyphs=lig.code_glyphs([0x80]))
    c.Tf(f, 16)
    c.show("ce supplies")
    p.line("Office supplies")
    c.Td(0, -36)
    c.show("Total: ")
    with c.actual_text("€12", hex_=True):
        c.show("EUR 12")
    p.line("Total: €12")
    c.ET()
    data, w = _finish(doc)
    desc = ("Marked content with /ActualText: a ligature glyph (fi) whose span says \"fi\", and the glyphs \"EUR 12\" "
            "whose span (UTF-16, hex) says \"€12\"; each span is one run with the actual text.")
    return _sample(doc, data, w, "actual-text", "actual-text", desc)


def rtl(seed):
    doc = Doc("rtl", "1.5", seed)
    f = doc.font(Type0, AMIRI, tu_style="mixed")
    p = doc.page(420, 300)
    c = p.content
    c.BT()
    c.Tf(f, 20)
    for i, line in enumerate(ARABIC):
        _line(c, p, f, 20, 0, 240 - 40 * i, [(line, "rtl")], "rtl", right=390)
    c.ET()
    data, w = _finish(doc)
    desc = ("Right-to-left lines (generic Arabic business text) in Amiri, shaped by HarfBuzz to contextual forms (Amiri "
            "draws lam-alef as two glyphs; mixed-direction's DejaVu Sans makes it one), written in visual order, right "
            "aligned; the line text is logical.")
    return _sample(doc, data, w, "rtl", "rtl", desc)


MIXED = [
    ([("تقرير المبيعات للربع الثالث ", "rtl"), ("2024", "ltr")], "rtl"),
    ([("The team meeting moves to ", "ltr"), ("الثلاثاء", "rtl"), (" at 10:30", "ltr")], "ltr"),
    ([("يستخدم الفريق ", "rtl"), ("SQL", "ltr"), (" في التحليل", "rtl")], "rtl"),
    ([("سعر الوحدة ", "rtl"), ("25", "ltr"), (" دولارا", "rtl")], "rtl"),
]


def mixed(seed):
    doc = Doc("mixed-direction", "1.5", seed)
    f = doc.font(Type0, DEJAVU, tu_style="mixed")
    p = doc.page(420, 300)
    c = p.content
    c.BT()
    c.Tf(f, 14)
    for i, (segments, base) in enumerate(MIXED):
        y = 250 - 34 * i
        if base == "rtl":
            _line(c, p, f, 14, 0, y, segments, "rtl", right=395)
        else:
            _line(c, p, f, 14, 25, y, segments, "ltr")
    c.ET()
    data, w = _finish(doc)
    desc = ("Mixed-direction lines in one font (DejaVu Sans): right-to-left lines holding a number and a Latin word, "
            "a left-to-right line holding an Arabic word; each directional run is one Tj, placed in visual order "
            "(checked against python-bidi), the line text logical. Lam-alef is one glyph for two characters (its "
            "ToUnicode gives them in logical order).")
    return _sample(doc, data, w, "mixed-direction", "mixed", desc)


def vertical(seed):
    doc = Doc("vertical", "1.5", seed)
    f = doc.font(Type0, DEJAVU, encoding="Identity-V", tu_style="bfchar")
    for line in ["ANNUAL REPORT 2024", "PLANNING"]:
        p = doc.page(300, 420)
        c = p.content
        c.BT()
        c.Tf(f, 18)
        c.Tm(1, 0, 0, 1, 150, 380)
        c.show(glyphs=f.glyphs(line))
        p.line(line, "ttb")
        c.ET()
    data, w = _finish(doc)
    desc = ("Vertical text: a composite font with Identity-V (/DW2 [880 -1000]), one column a page, each column one "
            "run going down.")
    return _sample(doc, data, w, "vertical", "vertical", desc)


def rotated_pages(seed):
    doc = Doc("rotated-pages", "1.4", seed)
    f = helvetica(doc)
    for rot in (0, 90, 180, 270):
        p = doc.page(420, 300, rotate=rot)
        c = p.content
        c.BT()
        c.Tf(f, 14)
        c.Td(30, 250)
        text = f"Page turned {rot} degrees"
        c.show(text)
        p.line(text)
        c.Td(0, -30)
        c.show("Second line of the page")
        p.line("Second line of the page")
        c.ET()
    p = doc.page(420, 300)
    c = p.content
    c.BT()
    c.Tf(f, 12)
    c.Tm(0.866, 0.5, -0.5, 0.866, 60, 60)
    c.show("Text at thirty degrees")
    p.line("Text at thirty degrees")
    c.ET()
    p = doc.page(420, 300)
    c = p.content
    c.q()
    c.cm(0, 1, -1, 0, 400, 40)
    c.BT()
    c.Tf(f, 12)
    c.Td(0, 0)
    c.show("Up the right edge")
    p.line("Up the right edge")
    c.ET()
    c.Q()
    data, w = _finish(doc)
    desc = ("Four pages with /Rotate 0, 90, 180 and 270 (boxes given as the page is shown, top-left origin), then a "
            "line turned 30 degrees by Tm and one turned 90 degrees by cm, each on its own page.")
    return _sample(doc, data, w, "rotated-pages", "rotation", desc)


def crop_box(seed):
    doc = Doc("cropbox", "1.4", seed)
    f = helvetica(doc)
    p = doc.page(612, 792, crop=[50, 60, 470, 360])
    c = p.content
    c.BT()
    c.Tf(f, 14)
    c.Td(80, 320)
    for i, line in enumerate(["Inside the crop box", "The page shows 420 by 300 points."]):
        if i:
            c.Td(0, -24)
        c.show(line)
        p.line(line)
    c.ET()
    p2 = doc.page(420, 300, media=[100, 100, 520, 400])
    c = p2.content
    c.BT()
    c.Tf(f, 14)
    c.Td(130, 350)
    c.show("A media box away from the origin")
    p2.line("A media box away from the origin")
    c.ET()
    data, w = _finish(doc)
    desc = ("A CropBox [50 60 470 360] inside a letter-size MediaBox (the page is the crop box, boxes from its top "
            "left), and a MediaBox [100 100 520 400] away from the origin.")
    return _sample(doc, data, w, "cropbox", "cropbox", desc)


def text_matrix(seed):
    doc = Doc("text-matrix", "1.4", seed)
    f = helvetica(doc)
    p = doc.page(420, 300)
    c = p.content
    c.BT()
    c.Tf(f, 1)
    c.Tm(14, 0, 0, 14, 30, 260)
    c.show("Size from the text matrix")
    p.line("Size from the text matrix")
    c.Tf(f, 12)
    c.Tm(1, 0, 0.25, 1, 30, 225)
    c.show("Slanted by the text matrix")
    p.line("Slanted by the text matrix")
    c.ET()
    c.q()
    c.cm(2, 0, 0, 2, 0, 0)
    c.BT()
    c.Tf(f, 7)
    c.Td(15, 95)
    c.show("Scaled by the CTM")
    p.line("Scaled by the CTM")
    c.ET()
    c.Q()
    with c.form("Fm1", [0, 0, 300, 60], (1, 0, 0, 1, 30, 100)) as fm:
        fm.BT()
        fm.Tf(f, 12)
        fm.Td(0, 20)
        fm.show("Inside a form XObject")
        fm.ET()
    p.line("Inside a form XObject")
    data, w = _finish(doc)
    desc = ("Sizes and placements from matrices: Tf 1 with Tm scaling 14, a slanted Tm, Tf 7 under a CTM scaling 2, "
            "and text in a form XObject with its /Matrix.")
    return _sample(doc, data, w, "text-matrix", "text-matrix", desc)


def build(seed):
    return [standard14(seed), encodings(seed), truetype_simple(seed), type0_identity(seed), type0_no_tounicode(seed),
            type0_no_cmap(seed), cid_map_stream(seed), type3(seed), tj_spacing(seed), render_modes(seed),
            actual_text(seed), rtl(seed), mixed(seed), vertical(seed), rotated_pages(seed), crop_box(seed),
            text_matrix(seed)]
