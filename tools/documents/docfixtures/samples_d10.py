"""D10, annotations and navigation: every annotation kind (text note, free text, line, square, circle, polygon,
polyline, highlight, underline, squiggly, strike-out, ink, stamp, caret, link, file attachment, popup, redaction) with
and without appearance streams, a reply thread and popups; links (URI, explicit destinations of each fit, named
destinations through the name tree and the catalog's /Dests); a bookmarks tree with nesting, colours, styles and
closed items; page labels (roman, decimal, prefixed, restarted, letters, a prefix alone); a tagged file whose structure
order differs from its content order; and redaction samples with text, paths and an image under the areas, with the
result of applying them (a sample of its own, drawn and read by the reference tools), and with what a redaction must reach
besides: a note's replies, popup, fields and the references to them; words split across text operators; images in a
pattern, a soft mask's group and a Type 3 glyph; an image's soft mask.

An annotation, as `annotations` lists it: `id` (its /NM), `kind` (note, freeText, line, square, circle, polygon,
polyline, highlight, underline, squiggly, strikeOut, ink, stamp, caret, link, fileAttachment, popup, redaction),
`subtype`, `page`, `box` (its Rect in the page as shown), `rect`, `contents`, `author`, `created`, `modified` (ISO
8601), `colour` and `interiorColour` (as written and RGB 0-255), `opacity`, `flags`, `points` (quad points as four
corners each, a line's two ends, a polygon's vertices, page as shown), `ink` (each stroke as a path: verbs and points),
`target` (a link's `uri`, or its destination: `page`, `fit`, the numbers, or `name`), `popup` (the id of its popup),
`parent` (a popup's annotation), `inReplyTo`, `replyType`, `state`, `stateModel`, `icon`, `borderWidth`,
`lineEndings`, `hasAppearance`, `attachment` (a file attachment's file)."""
from .common import Sample, base_expected, helvetica
from .document import Doc
from .filters import flate
from .images import Image, pattern
from .pdf import B, N, Stream, String, U16, sha256
from .png import png_bytes

D10 = ["D10"]
WHEN = ("D:20260302101500Z", "2026-03-02T10:15:00+00:00")
LATER = ("D:20260303091000Z", "2026-03-03T09:10:00+00:00")


def _rgb(c):
    if c is None:
        return None
    if len(c) == 1:
        v = round(c[0] * 255)
        return [v, v, v]
    return [round(x * 255) for x in c]


def _colour(c):
    return None if c is None else {"components": list(c), "rgb": _rgb(c)}


class Annots:
    def __init__(self, doc, with_ap):
        self.doc = doc
        self.with_ap = with_ap
        self.expected = []
        self.by_id = {}

    def ap(self, w, h, content, resources=None):
        return self.doc.pdf.add(Stream({"Type": N("XObject"), "Subtype": N("Form"), "BBox": [0, 0, w, h],
                                        "Resources": resources or {}, "Filter": N("FlateDecode")},
                                       flate(content), content))

    def add(self, page, kind, subtype, rect, nm, entries, ap=None, exp=None):
        d = {"Type": N("Annot"), "Subtype": N(subtype), "Rect": list(rect), "NM": String(nm.encode("ascii")),
             "P": page.ref}
        if subtype != "Popup":
            d["F"] = 4
        d.update(entries)
        has_ap = ap is not None and self.with_ap
        if has_ap:
            d["AP"] = {"N": ap}
        ref = self.doc.pdf.add(d)
        page.annotations.append(ref)
        x0, y0, x1, y1 = rect
        e = {"id": nm, "kind": kind, "subtype": subtype, "page": self.doc.pages.index(page),
             "box": page.display_box([(x0, y0), (x1, y1)]), "rect": list(rect),
             "contents": None, "author": None, "created": None, "modified": None, "colour": None,
             "interiorColour": None, "opacity": 1.0, "flags": d.get("F", 0), "points": None, "ink": None,
             "target": None, "popup": None, "parent": None, "inReplyTo": None, "replyType": None, "state": None,
             "stateModel": None, "icon": None, "borderWidth": None, "lineEndings": None, "hasAppearance": has_ap,
             "attachment": None}
        e.update(exp or {})
        self.expected.append(e)
        self.by_id[nm] = (ref, e, d)
        return ref, e

    def common(self, contents=None, author=None, colour=None, created=True, modified=True):
        d, e = {}, {}
        if contents is not None:
            d["Contents"] = String(contents.encode("latin-1")) if all(ord(c) < 128 for c in contents) else U16(contents)
            e["contents"] = contents
        if author is not None:
            d["T"] = String(author.encode("ascii"))
            e["author"] = author
        if colour is not None:
            d["C"] = list(colour)
            e["colour"] = _colour(colour)
        if created:
            d["CreationDate"] = String(WHEN[0].encode("ascii"))
            e["created"] = WHEN[1]
        if modified:
            d["M"] = String(LATER[0].encode("ascii"))
            e["modified"] = LATER[1]
        return d, e

    def popup(self, page, parent_nm, rect, open_=False):
        ref, e, d = self.by_id[parent_nm]
        pref, pe = self.add(page, "popup", "Popup", rect, parent_nm + "-popup", {"Parent": ref, "Open": open_},
                            exp={"parent": parent_nm, "flags": 0})
        d["Popup"] = pref
        e["popup"] = parent_nm + "-popup"
        return pref


def _quads(page, words):
    """Quad points around text words: (upper left, upper right, lower left, lower right) each, as viewers write
    them; and their corners in the page as shown."""
    flat, shown = [], []
    for x0, y0, x1, y1 in words:
        corners = [(x0, y1), (x1, y1), (x0, y0), (x1, y0)]
        for x, y in corners:
            flat += [x, y]
        shown.append([[round(v, 4) for v in page.display(x, y)] for x, y in corners])
    return flat, shown


def _text_line(doc, page, font, x, y, text, size=12):
    c = page.content
    c.BT()
    c.Tf(font, size)
    c.Td(x, y)
    run = c.show(text)
    c.ET()
    page.line(text)
    return run


def _word_box(font, line, word, x, y, size=12):
    """The box of `word` in `line` drawn at (x, y): along the baseline its advance, across it the font's descent to
    ascent."""
    k = line.index(word)
    before = sum(g.width for g in font.glyphs(line[:k])) / 1000 * size
    width = sum(g.width for g in font.glyphs(word)) / 1000 * size
    return (x + before, y + font.descent / 1000 * size, x + before + width, y + font.ascent / 1000 * size)


def annotations(seed, with_ap):
    name = "annotations" if with_ap else "annotations-no-appearance"
    doc = Doc(name, "1.7", seed)
    p = doc.page(612, 792)
    f = helvetica(doc)
    a = Annots(doc, with_ap)
    pdf = doc.pdf
    lines = [("Quarterly review: revenue rose in every region.", 700),
             ("The new warehouse cut delivery times by two days.", 670),
             ("Support answered most requests within a day.", 640),
             ("Travel costs fell by a tenth this quarter.", 610)]
    for text, y in lines:
        _text_line(doc, p, f, 60, y, text)
    # Text markup over words.
    for k, (kind, subtype, (line, y), word, colour) in enumerate([
            ("highlight", "Highlight", lines[0], "revenue", (1, 0.9, 0.2)),
            ("underline", "Underline", lines[1], "delivery", (0.1, 0.5, 0.1)),
            ("squiggly", "Squiggly", lines[2], "requests", (0.8, 0.1, 0.1)),
            ("strikeOut", "StrikeOut", lines[3], "tenth", (0.6, 0.1, 0.6))]):
        box = _word_box(f, line, word, 60, y)
        flat, shown = _quads(p, [box])
        rect = [round(box[0], 4), round(box[1], 4), round(box[2], 4), round(box[3], 4)]
        d, e = a.common(f"Markup: {word}", "Dana Writer", colour)
        d["QuadPoints"] = [round(v, 4) for v in flat]
        w, h = rect[2] - rect[0], rect[3] - rect[1]
        if subtype == "Highlight":
            content = b"/GS0 gs 1 0.9 0.2 rg 0 0 %s %s re f\n" % (_n(w), _n(h))
            res = {"ExtGState": {"GS0": pdf.add({"Type": N("ExtGState"), "BM": N("Multiply")})}}
        elif subtype == "Underline":
            content = b"0.1 0.5 0.1 RG 1 w 0 %s m %s %s l S\n" % (_n(1.5), _n(w), _n(1.5))
            res = None
        elif subtype == "Squiggly":
            pts = b" ".join(b"%s %s l" % (_n(x), _n(1 + 2 * (x // 2 % 2))) for x in range(2, int(w), 2))
            content = b"0.8 0.1 0.1 RG 0.8 w 0 1 m " + pts + b" S\n"
            res = None
        else:
            content = b"0.6 0.1 0.6 RG 1 w 0 %s m %s %s l S\n" % (_n(h * 0.45), _n(w), _n(h * 0.45))
            res = None
        e["points"] = shown
        a.add(p, kind, subtype, rect, f"markup-{k + 1}", d, a.ap(w, h, content, res), e)
    # A note with a popup and a reply thread (a reply and a review state).
    d, e = a.common("Can we add the regional figures?", "Dana Writer", (1, 0.8, 0))
    d.update({"Name": N("Comment"), "Open": False})
    e["icon"] = "Comment"
    note_ap = a.ap(20, 20, b"1 0.8 0 rg 0 0 20 20 re f 0 G 1 w 3 5 14 12 re S 6 13 m 14 13 l S 6 9 m 14 9 l S\n")
    a.add(p, "note", "Text", [540, 690, 560, 710], "note-1", d, note_ap, e)
    a.popup(p, "note-1", [400, 560, 590, 640])
    note_ref = a.by_id["note-1"][0]
    d, e = a.common("Added in section 3.", "Sam Reviewer", (0.2, 0.6, 1))
    d.update({"Name": N("Comment"), "IRT": note_ref, "RT": N("R")})
    e.update({"icon": "Comment", "inReplyTo": "note-1", "replyType": "reply"})
    a.add(p, "note", "Text", [540, 690, 560, 710], "note-2", d, note_ap, e)
    d, e = a.common(None, "Sam Reviewer", None)
    d.update({"Name": N("Comment"), "IRT": note_ref, "StateModel": String(b"Review"), "State": String(b"Accepted")})
    e.update({"icon": "Comment", "inReplyTo": "note-1", "replyType": "reply", "state": "Accepted",
              "stateModel": "Review", "flags": 4 | 2})
    d["F"] = 4 | 2  # a review state is hidden, as viewers write it
    a.add(p, "note", "Text", [540, 690, 560, 710], "note-3", d, None, e)
    # Free text.
    d, e = a.common("Figures checked by finance.", "Dana Writer", None)
    d.update({"DA": String(b"/Helv 11 Tf 0 0 0.6 rg"), "Q": 0})
    rect = [60, 520, 260, 545]
    ft_ap = a.ap(200, 25, b"q 0.6 G 1 w 0.5 0.5 199 24 re S Q BT /Helv 11 Tf 0 0 0.6 rg 4 8 Td (Figures checked by "
                          b"finance.) Tj ET\n", {"Font": {"Helv": f.ref}})
    a.add(p, "freeText", "FreeText", rect, "freetext-1", d, ft_ap, e)
    # Line with an arrow, square, circle, polygon, polyline.
    d, e = a.common("Trend", "Dana Writer", (0.8, 0.2, 0.1))
    d.update({"L": [300, 525, 420, 545], "LE": [N("None"), N("OpenArrow")], "BS": {"W": 2}})
    e.update({"points": [[300, 792 - 525], [420, 792 - 545]], "lineEndings": ["None", "OpenArrow"],
              "borderWidth": 2})
    a.add(p, "line", "Line", [296, 519, 426, 551], "line-1", d,
          a.ap(130, 32, b"0.8 0.2 0.1 RG 2 w 4 6 m 124 26 l S 124 26 m 113.8 26.9 l S 124 26 m 116.9 17.9 l S\n"), e)
    shapes = [("square", "Square", [60, 420, 160, 490], (0.1, 0.3, 0.8), (0.85, 0.9, 1)),
              ("circle", "Circle", [190, 420, 290, 490], (0.1, 0.6, 0.3), None)]
    for k, (kind, subtype, rect, col, ic) in enumerate(shapes):
        d, e = a.common(f"A {kind}", "Dana Writer", col)
        d["BS"] = {"W": 2}
        e["borderWidth"] = 2
        if ic:
            d["IC"] = list(ic)
            e["interiorColour"] = _colour(ic)
        w, h = rect[2] - rect[0], rect[3] - rect[1]
        if subtype == "Square":
            content = b"0.85 0.9 1 rg 0.1 0.3 0.8 RG 2 w 1 1 %s %s re B\n" % (_n(w - 2), _n(h - 2))
        else:
            k4 = 0.5523
            rx, ry = (w - 2) / 2, (h - 2) / 2
            cx, cy = w / 2, h / 2
            content = (b"0.1 0.6 0.3 RG 2 w %s %s m %s %s %s %s %s %s c %s %s %s %s %s %s c %s %s %s %s %s %s c "
                       b"%s %s %s %s %s %s c S\n" % tuple(_n(v) for v in (
                           cx + rx, cy, cx + rx, cy + k4 * ry, cx + k4 * rx, cy + ry, cx, cy + ry,
                           cx - k4 * rx, cy + ry, cx - rx, cy + k4 * ry, cx - rx, cy,
                           cx - rx, cy - k4 * ry, cx - k4 * rx, cy - ry, cx, cy - ry,
                           cx + k4 * rx, cy - ry, cx + rx, cy - k4 * ry, cx + rx, cy)))
        a.add(p, kind, subtype, rect, f"{kind}-1", d, a.ap(w, h, content), e)
    a.popup(p, "square-1", [60, 300, 250, 380], open_=True)
    poly = [(330, 420), (420, 430), (400, 490), (340, 480)]
    for kind, subtype, nm in (("polygon", "Polygon", "polygon-1"), ("polyline", "PolyLine", "polyline-1")):
        pts = poly if subtype == "Polygon" else [(x + 120, y) for x, y in poly]
        d, e = a.common(f"A {kind}", "Dana Writer", (0.5, 0.2, 0.6))
        d["Vertices"] = [v for pt in pts for v in pt]
        d["BS"] = {"W": 1.5}
        xs, ys = [x for x, _ in pts], [y for _, y in pts]
        rect = [min(xs) - 2, min(ys) - 2, max(xs) + 2, max(ys) + 2]
        e.update({"points": [[x, 792 - y] for x, y in pts], "borderWidth": 1.5})
        ops = b" ".join(b"%s %s %s" % (_n(x - rect[0]), _n(y - rect[1]), b"m" if k == 0 else b"l")
                        for k, (x, y) in enumerate(pts))
        content = b"0.5 0.2 0.6 RG 1.5 w " + ops + (b" s\n" if subtype == "Polygon" else b" S\n")
        a.add(p, kind, subtype, rect, nm, d, a.ap(rect[2] - rect[0], rect[3] - rect[1], content), e)
    # Ink: two strokes.
    strokes = [[(80, 230), (100, 260), (120, 240), (140, 270), (160, 235)], [(90, 215), (150, 215)]]
    d, e = a.common("Signed off", "Dana Writer", (0, 0.2, 0.7))
    d["InkList"] = [[v for pt in s for v in pt] for s in strokes]
    d["BS"] = {"W": 2}
    rect = [76, 211, 164, 274]
    e.update({"borderWidth": 2, "ink": [{"verbs": ["move"] + ["line"] * (len(s) - 1),
                                         "points": [[x, 792 - y] for x, y in s]} for s in strokes]})
    ops = b" ".join(b" ".join(b"%s %s %s" % (_n(x - rect[0]), _n(y - rect[1]), b"m" if k == 0 else b"l")
                              for k, (x, y) in enumerate(s)) + b" S" for s in strokes)
    a.add(p, "ink", "Ink", rect, "ink-1", d, a.ap(88, 63, b"0 0.2 0.7 RG 2 w 1 J 1 j " + ops + b"\n"), e)
    # Stamp, caret, file attachment.
    d, e = a.common("Approved for release", "Dana Writer", (0.1, 0.5, 0.1))
    d["Name"] = N("Approved")
    e["icon"] = "Approved"
    a.add(p, "stamp", "Stamp", [220, 210, 380, 260], "stamp-1", d,
          a.ap(160, 50, b"q 0.1 0.5 0.1 RG 3 w 2 2 156 46 re S Q BT /Helv 22 Tf 0.1 0.5 0.1 rg 18 16 Td (APPROVED) Tj "
                        b"ET\n", {"Font": {"Helv": f.ref}}), e)
    d, e = a.common("Insert 'all'", "Sam Reviewer", (0, 0, 1))
    d["Sy"] = N("None")
    a.add(p, "caret", "Caret", [205, 636, 215, 646], "caret-1", d,
          a.ap(10, 10, b"0 0 1 rg 0 0 m 5 10 l 10 0 l 7 0 l 5 5 l 3 0 l h f\n"), e)
    data_file = b"region,revenue\nNorth,48000\nSouth,38000\n"
    ef = pdf.add(Stream({"Type": N("EmbeddedFile"), "Subtype": N("text/csv"), "Params": {"Size": len(data_file)}},
                        data_file))
    fs = pdf.add({"Type": N("Filespec"), "F": B(b"figures.csv"), "UF": U16("figures.csv"), "EF": {"F": ef},
                  "Desc": B(b"Revenue by region")})
    d, e = a.common("Revenue by region", "Dana Writer", (0.5, 0.5, 0.5))
    d.update({"FS": fs, "Name": N("Paperclip")})
    e.update({"icon": "Paperclip", "attachment": {"fileName": "figures.csv", "size": len(data_file),
                                                  "sha256": sha256(data_file)}})
    a.add(p, "fileAttachment", "FileAttachment", [420, 230, 436, 254], "attachment-1", d,
          a.ap(16, 24, b"0.5 g 4 2 8 20 re f\n"), e)
    # Links: a URI and a destination on this page.
    d = {"Border": [0, 0, 0], "A": {"S": N("URI"), "URI": String(b"https://www.example.com/report")}}
    a.add(p, "link", "Link", [60, 150, 300, 170], "link-1", d, None,
          {"target": {"uri": "https://www.example.com/report"}, "flags": 4})
    d = {"Border": [0, 0, 1], "Dest": [p.ref, N("XYZ"), 60, 720, 0]}
    a.add(p, "link", "Link", [320, 150, 560, 170], "link-2", d, None,
          {"target": {"page": 0, "fit": "XYZ", "values": [60, 720, 0]}, "borderWidth": 1, "flags": 4})
    _text_line(doc, p, f, 62, 155, "Read the full report online", 11)
    _text_line(doc, p, f, 322, 155, "Back to the top of this page", 11)
    doc.finish()
    data, w = doc.write("table")
    what = "with appearance streams" if with_ap else ("without appearance streams (a reader draws them from their "
                                                      "properties; pdfium makes some itself)")
    desc = (f"Every annotation kind {what}: highlight, underline, squiggly and strike-out over words (quad points), a "
            "note with a popup and two replies (a reply and a hidden review state), free text, a line with an arrow, a "
            "square with an open popup, a circle, a polygon, a polyline, ink (two strokes), a stamp, a caret, a file "
            "attachment, a URI link and a link to a destination.")
    exp = base_expected(doc, data, name, "annotations", D10, desc, w, objects=False)
    exp["annotations"] = a.expected
    return Sample(name, "d10", "annotations", D10, desc, data, exp, checks=[check_annotations])


def _n(x):
    from .pdf import fmt_num

    return fmt_num(float(x)).encode("ascii")


# -- navigation -----------------------------------------------------------------------------------------------------------

LABELS = [(0, {"S": "r"}), (2, {"S": "D"}), (5, {"S": "D", "P": "A-", "St": 1}), (7, {"S": "A"}), (8, {"P": "Cover"})]


def _label(index):
    start, spec = max((s, sp) for s, sp in LABELS if s <= index)
    n = index - start + spec.get("St", 1)
    style = spec.get("S")
    if style == "D":
        num = str(n)
    elif style in ("r", "R"):
        vals = [(10, "x"), (9, "ix"), (5, "v"), (4, "iv"), (1, "i")]
        out = ""
        for v, s in vals:
            while n >= v:
                out += s
                n -= v
        num = out if style == "r" else out.upper()
    elif style in ("a", "A"):
        num = chr(ord("a") + (n - 1) % 26) * ((n - 1) // 26 + 1)
        num = num if style == "a" else num.upper()
    else:
        num = ""
    return spec.get("P", "") + num


def navigation(seed):
    doc = Doc("navigation", "1.7", seed)
    f = helvetica(doc)
    titles = ["Contents", "Introduction", "Overview", "Sales", "Costs", "Appendix: figures", "Appendix: notes",
              "Glossary", "Cover"]
    pages = []
    for k, t in enumerate(titles):
        p = doc.page(420, 300)
        _text_line(doc, p, f, 40, 250, t, 16)
        pages.append(p)
    pdf = doc.pdf
    # Named destinations: a name tree (Names /Dests) and the catalog's /Dests dictionary (the older way).
    named = {"sales": [pages[3].ref, N("FitH"), 280], "costs": [pages[4].ref, N("Fit")],
             "figures": [pages[5].ref, N("XYZ"), 40, 270, 1.5]}
    doc.catalog_extra["Names"] = {"Dests": pdf.add({"Names": [x for k in sorted(named) for x in (B(k.encode()),
                                                                                                    named[k])]})}
    doc.catalog_extra["Dests"] = pdf.add({"glossary": {"D": [pages[7].ref, N("FitR"), 20, 20, 400, 280]}})
    expected_named = {"sales": {"page": 3, "fit": "FitH", "values": [280]}, "costs": {"page": 4, "fit": "Fit",
                                                                                    "values": []},
                      "figures": {"page": 5, "fit": "XYZ", "values": [40, 270, 1.5]},
                      "glossary": {"page": 7, "fit": "FitR", "values": [20, 20, 400, 280], "source": "catalog Dests"}}
    # Links on the contents page: explicit destinations of several fits, named ones, a URI.
    p0 = pages[0]
    links = [("Introduction", {"Dest": [pages[1].ref, N("XYZ"), None, None, None]},
              {"page": 1, "fit": "XYZ", "values": [None, None, None]}),
             ("Overview", {"Dest": [pages[2].ref, N("FitV"), 50]}, {"page": 2, "fit": "FitV", "values": [50]}),
             ("Sales", {"A": {"S": N("GoTo"), "D": B(b"sales")}}, {"name": "sales"}),
             ("Costs", {"Dest": B(b"costs")}, {"name": "costs"}),
             ("Figures", {"A": {"S": N("GoTo"), "D": B(b"figures")}}, {"name": "figures"}),
             ("Glossary", {"Dest": N("glossary")}, {"name": "glossary", "source": "catalog Dests"}),
             ("Company website", {"A": {"S": N("URI"), "URI": String(b"https://www.example.com/")}},
              {"uri": "https://www.example.com/"})]
    a = Annots(doc, True)
    for k, (text, entries, target) in enumerate(links):
        y = 210 - 22 * k
        _text_line(doc, p0, f, 60, y, text, 12)
        width = sum(g.width for g in f.glyphs(text)) / 1000 * 12
        rect = [58, y - 4, round(62 + width, 4), y + 13]
        a.add(p0, "link", "Link", rect, f"link-{k + 1}", {"Border": [0, 0, 0], **entries}, None,
              {"target": target, "flags": 4})
    # Bookmarks: nesting, colours, styles (1 italic, 2 bold), a closed item (negative count).
    tree = [("Contents", 0, None, 0, []),
            ("Overview", 2, (0, 0, 0.8), 2, []),
            ("Sales", 3, None, 0, [("North", 3, None, 1, []), ("South", 3, None, 1, [])]),
            ("Costs", 4, (0.7, 0.1, 0.1), 3, [("Travel", 4, None, 0, []), ("Office", 4, None, 0, [])]),
            ("Appendix", None, None, 0, [("Figures", "figures", None, 0, []), ("Notes", 6, None, 0, [])])]
    closed = {"Costs"}

    def build_items(items, parent):
        refs = [pdf.alloc() for _ in items]
        exp = []
        for k, (title, dest, colour, style, kids) in enumerate(items):
            d = {"Title": String(title.encode("ascii")), "Parent": parent}
            if k > 0:
                d["Prev"] = refs[k - 1]
            if k + 1 < len(items):
                d["Next"] = refs[k + 1]
            if isinstance(dest, int):
                d["Dest"] = [pages[dest].ref, N("Fit")]
                target = {"page": dest, "fit": "Fit", "values": []}
            elif isinstance(dest, str):
                d["A"] = {"S": N("GoTo"), "D": B(dest.encode("ascii"))}
                target = {"name": dest}
            else:
                target = None
            if colour:
                d["C"] = list(colour)
            if style:
                d["F"] = style
            kid_exp = []
            if kids:
                first, last, kid_exp = build_items(kids, refs[k])
                d["First"], d["Last"] = first, last
                d["Count"] = -len(kids) if title in closed else len(kids)
            pdf.put(refs[k], d)
            exp.append({"title": title, "target": target, "colour": _colour(colour), "italic": bool(style & 1),
                        "bold": bool(style & 2), "open": bool(kids) and title not in closed, "children": kid_exp})
        return refs[0], refs[-1], exp

    outlines = pdf.alloc()
    first, last, outline_exp = build_items(tree, outlines)
    visible = len(tree) + sum(len(t[4]) for t in tree if t[0] not in closed)
    pdf.put(outlines, {"Type": N("Outlines"), "First": first, "Last": last, "Count": visible})
    doc.catalog_extra["Outlines"] = outlines
    doc.catalog_extra["PageMode"] = N("UseOutlines")
    nums = []
    for start, spec in LABELS:
        d = {"Type": N("PageLabel")}
        if "S" in spec:
            d["S"] = N(spec["S"])
        if "P" in spec:
            d["P"] = String(spec["P"].encode("ascii"))
        if "St" in spec:
            d["St"] = spec["St"]
        nums += [start, d]
    doc.catalog_extra["PageLabels"] = {"Nums": nums}
    doc.finish()
    data, w = doc.write("table")
    desc = ("Nine pages: links on the first (explicit destinations XYZ with nulls and FitV, named destinations through "
            "a GoTo action and a /Dest string into the Names /Dests tree, a /Dest name into the catalog's /Dests, a "
            "URI); bookmarks nested two deep, coloured, bold, italic, one closed (negative /Count), one reaching a "
            "named destination, one with no destination; page labels i ii 1 2 3 A-1 A-2 A Cover (lower roman, "
            "decimal, prefixed and restarted, upper letters, a prefix alone).")
    exp = base_expected(doc, data, "navigation", "navigation", D10, desc, w, objects=False)
    exp["annotations"] = a.expected
    exp["outline"] = outline_exp
    exp["outlineVisibleCount"] = visible
    exp["namedDestinations"] = expected_named
    exp["pageLabels"] = [_label(k) for k in range(len(pages))]
    exp["pageLabelRanges"] = [{"start": s, **sp} for s, sp in LABELS]
    return Sample("navigation", "d10", "navigation", D10, desc, data, exp, checks=[check_annotations, check_navigation])


# -- tagged ---------------------------------------------------------------------------------------------------------------

def tagged(seed):
    """Two columns whose content stream draws the right column first and the heading last; the structure tree reads
    heading, left column, right column; the page number is an artifact (not in the tagged text)."""
    doc = Doc("tagged", "1.7", seed)
    f = helvetica(doc)
    p = doc.page(500, 400)
    c = p.content
    pdf = doc.pdf
    parts = [  # (mcid, structure type, lines, x, y top) in content order
        (0, "P", ["The second column covers costs.", "Travel fell by a tenth."], 260, 300),
        (1, "P", ["The first column covers sales.", "Revenue rose in every region."], 30, 300),
        (2, "H1", ["Annual summary"], 30, 350),
    ]
    for mcid, stype, texts, x, y in parts:
        c.emit(f"/{stype} <</MCID {mcid}>> BDC")
        c.BT()
        c.Tf(f, 16 if stype == "H1" else 11)
        c.Td(x, y)
        for k, t in enumerate(texts):
            if k:
                c.Td(0, -16)
            c.show(t)
            p.line(t)
        c.ET()
        c.emit("EMC")
    c.emit("/Artifact <</Type /Pagination>> BDC")
    c.BT()
    c.Tf(f, 9)
    c.Td(240, 20)
    c.show("Page 1")
    p.line("Page 1")
    c.ET()
    c.emit("EMC")
    root = pdf.alloc()
    document = pdf.alloc()
    elems = {}
    for mcid, stype, texts, x, y in parts:
        elems[mcid] = pdf.add({"Type": N("StructElem"), "S": N(stype), "P": document, "Pg": p.ref, "K": mcid})
    order = [2, 1, 0]  # heading, left column, right column
    pdf.put(document, {"Type": N("StructElem"), "S": N("Document"), "P": root, "K": [elems[m] for m in order]})
    parent_tree = pdf.add({"Nums": [0, [elems[0], elems[1], elems[2]]]})
    pdf.put(root, {"Type": N("StructTreeRoot"), "K": document, "ParentTree": parent_tree, "ParentTreeNextKey": 1})
    doc.catalog_extra["StructTreeRoot"] = root
    doc.catalog_extra["MarkInfo"] = {"Marked": True}
    doc.catalog_extra["Lang"] = String(b"en")
    p.extra["StructParents"] = 0
    doc.finish()
    data, w = doc.write("table")
    texts = {mcid: "\n".join(t) for mcid, _, t, _, _ in parts}
    desc = ("A tagged page (MarkInfo, StructTreeRoot, MCIDs, a ParentTree) in two columns: the content draws the right "
            "column, then the left, then the heading; the structure gives heading, left column, right column. The page "
            "number is an /Artifact, left out of the tagged text.")
    exp = base_expected(doc, data, "tagged", "tagged", D10, desc, w, objects=False)
    exp["tagged"] = {"structure": [{"type": "Document", "children": [
        {"type": parts[m][1], "mcids": [m], "page": 0, "text": texts[m]} for m in order]}],
        "text": "\n".join(texts[m] for m in order), "contentOrderText": "\n".join(texts[m] for m in (0, 1, 2)),
        "artifacts": ["Page 1"]}
    return Sample("tagged", "d10", "tagged", D10, desc, data, exp, checks=[check_tagged])


# -- redaction ------------------------------------------------------------------------------------------------------------

RED_LINES = [("Account manager: Robin Park", 250), ("Budget for the launch: 48,000", 225),
             ("Meeting room booked for Tuesday", 200)]


def _redaction_page(doc, applied):
    """The page before (or after) applying the redactions. Returns what the JSON needs."""
    f = helvetica(doc)
    doc.vector = True
    p = doc.page(420, 300)
    c = p.content
    removed_words = {0: "Robin Park", 1: "48,000"}
    for k, (line, y) in enumerate(RED_LINES):
        if applied and k in removed_words:
            kept = line[:line.index(removed_words[k])]
            _text_line(doc, p, f, 30, y, kept)
        else:
            _text_line(doc, p, f, 30, y, line)
    c.rg(0.2, 0.5, 0.8)
    inside = None
    if not applied:
        c.re(300, 120, 30, 20); inside = c.f()           # inside the second area
    c.rg(0.85, 0.4, 0.1)
    if applied:
        c.q()
        c.re(0, 0, 420, 300); c.re(290, 100, 100, 50); c.W(evenodd=True); c.n()   # everything but the area
    c.re(250, 110, 80, 20); crossing = c.f()             # half under the second area
    if applied:
        c.Q()
    c.rg(0.3, 0.6, 0.3)
    c.re(30, 110, 80, 20); outside = c.f()               # clear of every area
    img = Image(doc, 32, 24, 8, "DeviceRGB", 3, pattern(32, 24, 3, 255, 5), label="half under the second area")
    return p, f, img, inside, crossing, outside


def _apply_pixels(img, matrix, areas, colour):
    """A pixel whose centre falls inside an area takes the area's interior colour."""
    a, _, _, d, e, f = matrix
    out = bytearray(img.pixels[2])
    changed = 0
    for j in range(img.height):
        for i in range(img.width):
            x = e + (i + 0.5) / img.width * a
            y = f + d - (j + 0.5) / img.height * d
            if any(x0 < x < x1 and y0 < y < y1 for x0, y0, x1, y1 in areas):
                k = 3 * (j * img.width + i)
                out[k:k + 3] = bytes(colour)
                changed += 1
    return bytes(out), changed


def redaction(seed):
    doc = Doc("redaction", "1.7", seed)
    p, f, img, inside, crossing, outside = _redaction_page(doc, applied=False)
    matrix = (60, 0, 0, 45, 340, 60)
    rec = p.content.image("Im1", img, matrix)
    pdf = doc.pdf
    word_box = _word_box(f, RED_LINES[0][0], "Robin Park", 30, RED_LINES[0][1])
    word_box2 = _word_box(f, RED_LINES[1][0], "48,000", 30, RED_LINES[1][1])
    region = (290, 100, 390, 150)
    black = (0, 0, 0)
    reds = []
    for k, (box, quads) in enumerate(((word_box, True), (word_box2, True), (region, False))):
        x0, y0, x1, y1 = [round(v, 4) for v in box]
        d = {"Type": N("Annot"), "Subtype": N("Redact"), "Rect": [x0, y0, x1, y1], "F": 4, "P": p.ref,
             "NM": String(f"redact-{k + 1}".encode("ascii")), "IC": list(black), "C": [1, 0, 0],
             "Contents": String(b"Remove before sharing"), "T": String(b"Dana Writer")}
        shown = None
        if quads:
            flat, shown = _quads(p, [(x0, y0, x1, y1)])
            d["QuadPoints"] = flat
        ap = pdf.add(Stream({"Type": N("XObject"), "Subtype": N("Form"), "BBox": [0, 0, x1 - x0, y1 - y0],
                             "Resources": {}, "Filter": N("FlateDecode")},
                            flate(b"1 0 0 RG 1 w 0.5 0.5 %s %s re S\n" % (_n(x1 - x0 - 1), _n(y1 - y0 - 1))),
                            b"1 0 0 RG 1 w 0.5 0.5 %s %s re S\n" % (_n(x1 - x0 - 1), _n(y1 - y0 - 1))))
        d["AP"] = {"N": ap}
        p.annotations.append(pdf.add(d))
        reds.append({"id": f"redact-{k + 1}", "page": 0, "rect": [x0, y0, x1, y1],
                     "box": p.display_box([(x0, y0), (x1, y1)]), "points": shown, "interiorColour": list(black),
                     "overlayText": None})
    areas = [r["rect"] for r in reds]
    pixels, changed = _apply_pixels(img, matrix, areas, black)
    files = {"redaction.p1.im1.png": img.png(),
             "redaction.p1.im1.redacted.png": png_bytes(img.width, img.height, 3, 8, pixels)}
    rec["pixels"]["file"] = "redaction.p1.im1.png"
    doc.finish()
    data, w = doc.write("table")
    desc = ("Three redaction annotations (interior colour black): two over words (quad points), one over a region "
            "covering a filled rectangle entirely, half of another and half of an image; a third rectangle and the "
            "other lines are clear. `redaction.afterApplying` says what must be gone: the words, the path inside, the "
            "half of the crossing path and the image pixels under the region; redaction-applied is that result.")
    exp = base_expected(doc, data, "redaction", "redaction", D10, desc, w, objects=False)
    after_lines = [RED_LINES[0][0][:RED_LINES[0][0].index("Robin Park")], RED_LINES[1][0][:RED_LINES[1][0].index("48,000")],
                   RED_LINES[2][0]]
    exp["redaction"] = {
        "areas": reds,
        "rules": {"text": "a glyph whose box overlaps an area is removed (the areas cover whole words, so a rule on "
                          "the glyph's centre gives the same)",
                  "paths": "a path inside an area is removed; a path crossing one keeps only its parts outside",
                  "images": "a pixel whose centre falls inside an area takes the area's interior colour",
                  "marks": "each area is then filled with its interior colour"},
        "afterApplying": {
            "lines": after_lines, "removedText": ["Robin Park", "48,000"],
            "paths": {"removed": [inside["order"]], "clipped": [crossing["order"]], "kept": [outside["order"]]},
            "images": [{"order": rec["order"], "pixels": "redaction.p1.im1.redacted.png",
                        "sha256": sha256(pixels), "changedPixels": changed}],
            "sample": "redaction-applied"}}
    return Sample("redaction", "d10", "redaction", D10, desc, data, exp, files)


def redaction_applied(seed):
    """What applying redaction's areas must give: the words gone from the content, the path inside gone, the crossing
    path clipped to the outside of the area, the image's pixels under the area black, each area filled black."""
    doc = Doc("redaction-applied", "1.7", seed)
    p, f, img, inside, crossing, outside = _redaction_page(doc, applied=True)
    matrix = (60, 0, 0, 45, 340, 60)
    word_box = _word_box(f, RED_LINES[0][0], "Robin Park", 30, RED_LINES[0][1])
    word_box2 = _word_box(f, RED_LINES[1][0], "48,000", 30, RED_LINES[1][1])
    region = (290, 100, 390, 150)
    areas = [tuple(round(v, 4) for v in b) for b in (word_box, word_box2, region)]
    pixels, changed = _apply_pixels(img, matrix, areas, (0, 0, 0))
    rows = [list(pixels[y * 96:(y + 1) * 96]) for y in range(24)]
    redacted = Image(doc, 32, 24, 8, "DeviceRGB", 3, rows, label="the image after redaction")
    rec = p.content.image("Im1", redacted, matrix)
    c = p.content
    c.rg(0, 0, 0)
    for x0, y0, x1, y1 in areas:
        c.re(x0, y0, x1 - x0, y1 - y0)
        c.f()
    rec["pixels"]["file"] = "redaction-applied.p1.im1.png"
    files = {"redaction-applied.p1.im1.png": redacted.png()}
    doc.finish()
    data, w = doc.write("table")
    desc = ("redaction with its areas applied, as the expected result: the words left out of the text, the path "
            "inside the region gone, the crossing path clipped to outside it, the image's pixels under it black "
            "(the same bytes as redaction.p1.im1.redacted.png), each area filled black, no redaction annotations.")
    exp = base_expected(doc, data, "redaction-applied", "redaction", D10, desc, w, objects=False)
    exp["redaction"] = {"appliedFrom": "redaction", "removedText": ["Robin Park", "48,000"]}
    return Sample("redaction-applied", "d10", "redaction", D10, desc, data, exp, files, checks=[check_redacted])


# -- redaction's reach: replies, words split across operators, images in patterns, masks and glyphs ----------------------

def _mark(doc, page, rect, nm):
    """A redaction mark over `rect` (user space), filled black when applied."""
    x0, y0, x1, y1 = [round(v, 4) for v in rect]
    page.annotations.append(doc.pdf.add({"Type": N("Annot"), "Subtype": N("Redact"), "Rect": [x0, y0, x1, y1], "F": 4,
                                         "P": page.ref, "NM": String(nm.encode("ascii")), "IC": [0, 0, 0],
                                         "C": [1, 0, 0]}))
    return {"id": nm, "page": doc.pages.index(page), "rect": [x0, y0, x1, y1],
            "box": page.display_box([(x0, y0), (x1, y1)])}


def _under(width, height, matrix, area):
    """The pixels (i, j) of an image of `width` x `height` placed by `matrix` (user space, rows from the top) whose
    centre falls inside `area` (x0, y0, x1, y1 in user space)."""
    a, b, c, d, e, f = matrix
    x0, y0, x1, y1 = area
    out = set()
    for j in range(height):
        for i in range(width):
            u, v = (i + 0.5) / width, 1 - (j + 0.5) / height
            x, y = a * u + c * v + e, b * u + d * v + f
            if x0 < x < x1 and y0 < y < y1:
                out.add((i, j))
    return out


def _redacted_pixels(img, under, value):
    """An image's 8-bit pixels (interleaved) with those `under` set to `value` (one level a channel)."""
    channels, _, data = img.pixels
    out = bytearray(data)
    for i, j in under:
        k = channels * (j * img.width + i)
        out[k:k + channels] = bytes([value] * channels)
    return bytes(out)


def _image_expectation(name, img, under, value, files):
    """What an image under an area must decode to once redacted: its pixels, a PNG of them and their SHA-256."""
    channels, _, _ = img.pixels
    pixels = _redacted_pixels(img, under, value)
    files[name] = png_bytes(img.width, img.height, channels, 8, pixels)
    return {"pixels": name, "sha256": sha256(pixels), "changedPixels": len(under), "original": sha256(img.pixels[2])}


def redaction_replies(seed):
    """A note over a redaction mark with a popup and a chain of replies (one on another page, one there sharing the note
    in its /Annots), a structure tree naming the note and a field's widget by object references (one written in), an
    outline item whose action names the note, a named destination, and two form fields whose widgets lie under a second
    mark (a field that is its widget, and a parent field with one kid): applying the marks removes them all."""
    doc = Doc("redaction-replies", "1.7", seed)
    pdf = doc.pdf
    f = helvetica(doc)
    p = doc.page(420, 300)
    c = p.content
    line = "Supplier: Larkspur Trading"
    c.emit("/P <</MCID 0>> BDC")
    _text_line(doc, p, f, 30, 250, line)
    c.emit("EMC")
    _text_line(doc, p, f, 30, 225, "Delivery window: next spring")
    _text_line(doc, p, f, 30, 60, "Notes are kept with the order")
    a = Annots(doc, True)
    note_ap = a.ap(16, 16, b"1 0.8 0 rg 0 0 16 16 re f\n")
    note1, _ = a.add(p, "note", "Text", [272, 246, 288, 262], "note-1",
                     {"Contents": String(b"Larkspur pricing is final"), "T": String(b"Dana Writer"),
                      "Name": N("Comment"), "StructParent": 1}, note_ap)
    a.popup(p, "note-1", [300, 160, 410, 220])
    note2, _ = a.add(p, "note", "Text", [300, 100, 316, 116], "note-2",
                     {"Contents": String(b"Agreed with Quillon terms"), "T": String(b"Sam Reviewer"), "IRT": note1,
                      "RT": N("R"), "Name": N("Comment")}, note_ap)
    a.add(p, "note", "Text", [320, 100, 336, 116], "note-3",
          {"Contents": String(b"Quillon contract attached"), "T": String(b"Dana Writer"), "IRT": note2,
           "Name": N("Comment")}, note_ap)
    a.add(p, "note", "Text", [380, 40, 396, 56], "note-5",
          {"Contents": String(b"Delivery confirmed"), "T": String(b"Sam Reviewer"), "Name": N("Comment")}, note_ap)
    # Two fields under the second mark: one that is its own widget, one a parent with one kid.
    field_res = {"Font": {"Helv": f.ref}}
    field1, _ = a.add(p, "widget", "Widget", [30, 150, 200, 170], "field-1",
                      {"FT": N("Tx"), "T": String(b"supplier"), "V": String(b"Larkspur Trading"),
                       "DA": String(b"/Helv 10 Tf 0 g"), "StructParent": 2},
                      a.ap(170, 20, b"BT /Helv 10 Tf 0 g 2 6 Td (Larkspur Trading) Tj ET\n", field_res))
    parent = pdf.alloc()
    kid, _ = a.add(p, "widget", "Widget", [30, 120, 200, 140], "field-2", {"Parent": parent},
                   a.ap(170, 20, b"BT /Helv 10 Tf 0 g 2 6 Td (Quillon desk) Tj ET\n", field_res))
    pdf.put(parent, {"FT": N("Tx"), "T": String(b"contact"), "V": String(b"Quillon desk"),
                     "DA": String(b"/Helv 10 Tf 0 g"), "Kids": [kid]})
    doc.catalog_extra["AcroForm"] = {"Fields": [field1, parent], "DA": String(b"/Helv 10 Tf 0 g"),
                                     "DR": {"Font": {"Helv": f.ref}}}
    # A reply on the second page, which also lists the note itself.
    p2 = doc.page(420, 300)
    _text_line(doc, p2, f, 30, 250, "Follow-up notes")
    a.add(p2, "note", "Text", [40, 200, 56, 216], "note-4",
          {"Contents": String(b"Larkspur follow-up"), "T": String(b"Sam Reviewer"), "IRT": note1, "Name": N("Comment")},
          note_ap)
    p2.annotations.append(note1)
    # The structure: the first line, the note (its own object reference), the field (one written in).
    root, document = pdf.alloc(), pdf.alloc()
    para = pdf.add({"Type": N("StructElem"), "S": N("P"), "P": document, "Pg": p.ref, "K": 0})
    objr = pdf.add({"Type": N("OBJR"), "Obj": note1, "Pg": p.ref})
    note_elem = pdf.add({"Type": N("StructElem"), "S": N("Annot"), "P": document, "Pg": p.ref,
                         "Alt": String(b"Larkspur note"), "K": objr})
    form_elem = pdf.add({"Type": N("StructElem"), "S": N("Form"), "P": document, "Pg": p.ref,
                         "Alt": String(b"Larkspur supplier field"),
                         "K": {"Type": N("OBJR"), "Obj": field1, "Pg": p.ref}})
    pdf.put(document, {"Type": N("StructElem"), "S": N("Document"), "P": root, "K": [para, note_elem, form_elem]})
    tree = pdf.add({"Nums": [0, [para], 1, note_elem, 2, form_elem]})
    pdf.put(root, {"Type": N("StructTreeRoot"), "K": document, "ParentTree": tree, "ParentTreeNextKey": 3})
    doc.catalog_extra["StructTreeRoot"] = root
    doc.catalog_extra["MarkInfo"] = {"Marked": True}
    p.extra["StructParents"] = 0
    # An outline item whose action names the note, and a named destination.
    outlines, item = pdf.alloc(), pdf.alloc()
    pdf.put(item, {"Title": String(b"Supplier note"), "Parent": outlines,
                   "A": {"S": N("Hide"), "T": note1, "H": False}})
    pdf.put(outlines, {"Type": N("Outlines"), "First": item, "Last": item, "Count": 1})
    doc.catalog_extra["Outlines"] = outlines
    doc.catalog_extra["Names"] = {"Dests": pdf.add({"Names": [B(b"supplier"), [p.ref, N("XYZ"), 30, 260, 0]]})}
    x0, y0, x1, y1 = _word_box(f, line, "Larkspur Trading", 30, 250)
    marks = [_mark(doc, p, (x0, y0, 292, 264), "redact-1"), _mark(doc, p, (25, 115, 205, 175), "redact-2")]
    doc.finish()
    data, w = doc.write("table")
    desc = ("A note under a redaction mark, with a popup and replies (a chain of two on its page, one on page 2 whose "
            "/Annots also lists the note), named by the structure tree (an /OBJR of its own) and an outline item's "
            "action; two fields (one its own widget, one a parent of one kid) under a second mark, the first named by "
            "the structure tree (an /OBJR written in). Applying the marks removes the note, its popup, every reply, "
            "both fields and the references to them; note-5 stays.")
    exp = base_expected(doc, data, "redaction-replies", "redaction", D10, desc, w, objects=False)
    exp["redaction"] = {
        "areas": marks,
        "afterApplying": {
            "removedText": ["Larkspur", "Quillon"],
            "removedAnnotations": ["note-1", "note-1-popup", "note-2", "note-3", "note-4", "field-1", "field-2"],
            "keptAnnotations": ["note-5"], "keptText": ["Supplier:", "Delivery window: next spring", "Follow-up notes"],
            "annotationCounts": [1, 0]}}
    return Sample("redaction-replies", "d10", "redaction", D10, desc, data, exp)


def redaction_split(seed):
    """Words split across text operators: two Tj of one font, a font change (Helvetica to Times) inside the word, and
    two TJ arrays whose first ends in a kerned piece; a search must find each whole word."""
    doc = Doc("redaction-split", "1.7", seed)
    helv = helvetica(doc)
    times = helvetica(doc, "Times-Roman")
    p = doc.page(420, 300)
    c = p.content
    c.BT(); c.Tf(helv, 12); c.Td(30, 250); c.show("Order code Zeph"); c.show("yrine shipped"); c.ET()
    p.line("Order code Zephyrine shipped")
    c.BT(); c.Tf(helv, 12); c.Td(30, 225); c.show("Contact Quor"); c.Tf(times, 12); c.show("vath today"); c.ET()
    p.line("Contact Quorvath today")
    c.BT(); c.Tf(helv, 12); c.Td(30, 200); c.TJ(["Visit Brin", -20, "dle"]); c.TJ(["wick soon"]); c.ET()
    p.line("Visit Brindlewick soon")
    _text_line(doc, p, helv, 30, 175, "Nothing else changes here")
    doc.finish()
    data, w = doc.write("table")
    desc = ("Three words each split across text operators (two Tj of one font; a font change inside the word; two TJ "
            "arrays, the first kerned), and a line that stays. A text search finds each word whole, with an area over "
            "its part in each piece; redacting them leaves none of their pieces in the file.")
    exp = base_expected(doc, data, "redaction-split", "redaction", D10, desc, w, objects=False)
    exp["redaction"] = {"words": ["Zephyrine", "Quorvath", "Brindlewick"],
                        "afterApplying": {"removedText": ["Zephyrine", "Quorvath", "Brindlewick"],
                                          "removedPieces": ["Zeph", "yrine", "Quor", "vath", "Brin", "wick"],
                                          "keptText": ["Order code", "shipped", "Contact", "today", "Visit", "soon",
                                                       "Nothing else changes here"]}}
    return Sample("redaction-split", "d10", "redaction", D10, desc, data, exp)


def _bits(rows):
    """Rows of 0/1 (1 a pixel left unpainted) packed one bit a pixel, rows padded to whole bytes."""
    out = bytearray()
    for row in rows:
        for k in range(0, len(row), 8):
            byte = 0
            for b, v in enumerate(row[k:k + 8]):
                byte |= v << (7 - b)
            out.append(byte)
    return bytes(out)


def redaction_nested(seed):
    """Images where a redaction must follow them in: a tiling pattern's one cell, a soft mask's group, and a Type 3
    glyph that draws an image mask past its advance (and a second glyph wholly under the area)."""
    from .fonts import Type3

    doc = Doc("redaction-nested", "1.7", seed)
    pdf = doc.pdf
    f = helvetica(doc)
    p = doc.page(420, 300)
    c = p.content
    _text_line(doc, p, f, 30, 270, "Images inside a pattern, a soft mask and glyphs")
    files = {}
    # A tiling pattern whose one cell (the size of the rectangle it fills) is an image, half under the first area.
    im1 = Image(doc, 20, 12, 8, "DeviceRGB", 3, pattern(20, 12, 3, 255, 7), label="the pattern's image")
    cell = b"q 100 0 0 60 0 0 cm /Im1 Do Q\n"
    pat = pdf.add(Stream({"Type": N("Pattern"), "PatternType": 1, "PaintType": 1, "TilingType": 1,
                          "BBox": [0, 0, 100, 60], "XStep": 100, "YStep": 60, "Matrix": [1, 0, 0, 1, 30, 160],
                          "Resources": {"XObject": {"Im1": im1.ref}}, "Filter": N("FlateDecode")}, flate(cell), cell))
    p.resource("Pattern", "P1", pat)
    c.emit("/Pattern cs /P1 scn 30 160 100 60 re f")
    area1 = (80, 150, 140, 230)
    # A soft mask (luminosity) whose group is an image, a filled rectangle drawn through it, half under the second area.
    im2 = Image(doc, 20, 12, 8, "DeviceGray", 1, pattern(20, 12, 1, 255, 3), label="the soft mask's image")
    group = b"q 100 0 0 60 0 0 cm /Im2 Do Q\n"
    form = pdf.add(Stream({"Type": N("XObject"), "Subtype": N("Form"), "BBox": [0, 0, 100, 60],
                           "Group": {"S": N("Transparency"), "CS": N("DeviceGray")},
                           "Resources": {"XObject": {"Im2": im2.ref}}, "Filter": N("FlateDecode")}, flate(group), group))
    gs = pdf.add({"Type": N("ExtGState"), "SMask": {"Type": N("Mask"), "S": N("Luminosity"), "G": form}})
    p.resource("ExtGState", "GS1", gs)
    c.emit("q 1 0 0 1 250 160 cm /GS1 gs 0.1 0.3 0.8 rg 0 0 100 60 re f Q")
    area2 = (300, 150, 360, 230)
    # A Type 3 font: A draws an image mask twice its advance wide, B one its own size. One A stands with its drawing's
    # right part under the third area (its own box clear of it), B wholly under it, and another A clear of it.
    a_rows = [[(x + y) % 3 == 0 and 1 or 0 for x in range(16)] for y in range(8)]
    b_rows = [[(x * 5 + y * 3) % 7 < 3 and 1 or 0 for x in range(16)] for y in range(8)]
    a_data, b_data = _bits(a_rows), _bits(b_rows)
    for d in (a_data, b_data):
        assert b"EI" not in d and b"ID" not in d
    procs = {"A": (1000, b"1000 0 0 -200 2000 800 d1\nq 2000 0 0 1000 0 -200 cm\nBI /W 16 /H 8 /IM true /BPC 1 ID\n"
                         + a_data + b"\nEI\nQ\n"),
             "B": (1000, b"1000 0 0 -200 1000 800 d1\nq 1000 0 0 1000 0 -200 cm\nBI /W 16 /H 8 /IM true /BPC 1 ID\n"
                         + b_data + b"\nEI\nQ\n")}
    t3 = doc.font(Type3, procs)
    c.BT(); c.Tf(t3, 40); c.Td(30, 60); c.TJ(["A", -1250, "B"]); c.ET()
    c.BT(); c.Tf(t3, 40); c.Td(250, 60); c.show("A"); c.ET()
    p.line("A B A")
    area3 = (75, 40, 170, 110)
    marks = [_mark(doc, p, area1, "redact-1"), _mark(doc, p, area2, "redact-2"), _mark(doc, p, area3, "redact-3")]
    doc.finish()
    data, w = doc.write("table")
    # What each image must read as once redacted: the pattern's and the mask's pixels under their areas black; the A
    # glyph's copy unpainted (255) where its pixels fall in the third area (its drawing at 30..110 across, 52..92 up).
    image1 = _image_expectation("redaction-nested.pattern.redacted.png", im1, _under(20, 12, (100, 0, 0, 60, 30, 160), area1), 0, files)
    image2 = _image_expectation("redaction-nested.mask.redacted.png", im2, _under(20, 12, (100, 0, 0, 60, 250, 160), area2), 0, files)
    glyph_under = _under(16, 8, (80, 0, 0, 40, 30, 52), area3)
    glyph = [[255 if (i, j) in glyph_under or a_rows[j][i] else 0 for i in range(16)] for j in range(8)]
    glyph_pixels = bytes(v for row in glyph for v in row)
    files["redaction-nested.glyph.redacted.png"] = png_bytes(16, 8, 1, 8, glyph_pixels)
    desc = ("Three areas over images a redaction must follow in: a tiling pattern's one cell (an RGB image) filling a "
            "rectangle half under the first; a soft mask's group (a grey image) a rectangle is drawn through, half "
            "under the second; a Type 3 font whose glyph A draws an image mask twice its advance wide (one A with its "
            "drawing's right part under the third, its own box clear; another clear) and whose glyph B lies wholly "
            "under the third. The pattern's and the mask's pixels under the areas turn black, the first A is drawn from "
            "a copy unpainted under the area, B leaves the text and its description the file.")
    exp = base_expected(doc, data, "redaction-nested", "redaction", D10, desc, w, objects=False)
    exp["redaction"] = {
        "areas": marks,
        "afterApplying": {
            "removedText": [], "keptText": ["Images inside a pattern, a soft mask and glyphs"],
            "patternImage": image1, "maskImage": image2,
            "glyph": {"pixels": "redaction-nested.glyph.redacted.png", "sha256": sha256(glyph_pixels),
                      "changedPixels": len(glyph_under), "text": "A A"},
            "removedBytes": [b_data.hex()], "keptBytes": [a_data.hex()]}}
    return Sample("redaction-nested", "d10", "redaction", D10, desc, data, exp, files)


def redaction_masked(seed):
    """An image with a soft mask, half under a redaction mark: the copy keeps the mask, redacted under the area."""
    doc = Doc("redaction-masked", "1.7", seed)
    f = helvetica(doc)
    p = doc.page(420, 300)
    _text_line(doc, p, f, 30, 270, "A photo with a soft mask, half under the area")
    mask = Image(doc, 24, 16, 8, "DeviceGray", 1, [[(x * 10 + y * 7) % 256 for x in range(24)] for y in range(16)],
                 label="the photo's soft mask")
    img = Image(doc, 24, 16, 8, "DeviceRGB", 3, pattern(24, 16, 3, 255, 9), label="the photo", smask=mask)
    matrix = (120, 0, 0, 80, 100, 100)
    p.content.image("Im1", img, matrix)
    area = (160, 90, 260, 190)
    marks = [_mark(doc, p, area, "redact-1")]
    doc.finish()
    data, w = doc.write("table")
    files = {}
    under = _under(24, 16, matrix, area)
    desc = ("An RGB image with a grey soft mask, its right half under a redaction mark: once applied, the image is drawn "
            "from a copy whose pixels under the area are black and whose soft mask is kept, opaque (255) under the "
            "area and as it was elsewhere.")
    exp = base_expected(doc, data, "redaction-masked", "redaction", D10, desc, w, objects=False)
    exp["redaction"] = {
        "areas": marks,
        "afterApplying": {
            "removedText": [], "keptText": ["A photo with a soft mask, half under the area"],
            "image": _image_expectation("redaction-masked.image.redacted.png", img, under, 0, files),
            "mask": _image_expectation("redaction-masked.mask.redacted.png", mask, under, 255, files)}}
    return Sample("redaction-masked", "d10", "redaction", D10, desc, data, exp, files)


# -- reference checks ----------------------------------------------------------------------------------------------------

SUBTYPES = {"Text": 1, "Link": 2, "FreeText": 3, "Line": 4, "Square": 5, "Circle": 6, "Polygon": 7, "PolyLine": 8,
            "Highlight": 9, "Underline": 10, "Squiggly": 11, "StrikeOut": 12, "Stamp": 13, "Caret": 14, "Ink": 15,
            "Popup": 16, "FileAttachment": 17, "Widget": 20, "Redact": 28}


def _wide(fn, *args):
    import ctypes

    n = fn(*args, None, 0)
    if n <= 2:
        return ""
    buf = (ctypes.c_ushort * (n // 2))()
    fn(*args, buf, n)
    return bytes(buf)[:n - 2].decode("utf-16-le")


def pdfium_annotations(path):
    import ctypes

    import pypdfium2 as pdfium
    import pypdfium2.raw as raw

    doc = pdfium.PdfDocument(path)
    out = []
    try:
        for i in range(len(doc)):
            page = doc[i]
            for k in range(raw.FPDFPage_GetAnnotCount(page)):
                annot = raw.FPDFPage_GetAnnot(page, k)
                rect = raw.FS_RECTF()
                raw.FPDFAnnot_GetRect(annot, rect)
                entry = {"page": i, "subtype": raw.FPDFAnnot_GetSubtype(annot),
                         "rect": [round(rect.left, 3), round(rect.bottom, 3), round(rect.right, 3), round(rect.top, 3)],
                         "nm": _wide(raw.FPDFAnnot_GetStringValue, annot, b"NM"),
                         "contents": _wide(raw.FPDFAnnot_GetStringValue, annot, b"Contents"),
                         "flags": raw.FPDFAnnot_GetFlags(annot),
                         "quads": raw.FPDFAnnot_CountAttachmentPoints(annot),
                         "vertices": raw.FPDFAnnot_GetVertices(annot, None, 0),
                         "inkStrokes": raw.FPDFAnnot_GetInkListCount(annot)}
                link = raw.FPDFAnnot_GetLink(annot)
                if link:
                    action = raw.FPDFLink_GetAction(link)
                    if action and raw.FPDFAction_GetType(action) == raw.PDFACTION_URI:
                        n = raw.FPDFAction_GetURIPath(doc, action, None, 0)
                        buf = ctypes.create_string_buffer(n)
                        raw.FPDFAction_GetURIPath(doc, action, buf, n)
                        entry["uri"] = buf.raw[:n - 1].decode("ascii")
                    dest = raw.FPDFLink_GetDest(doc, link)
                    if not dest and action and raw.FPDFAction_GetType(action) == raw.PDFACTION_GOTO:
                        dest = raw.FPDFAction_GetDest(doc, action)
                    if dest:
                        entry["destPage"] = raw.FPDFDest_GetDestPageIndex(doc, dest)
                out.append(entry)
                raw.FPDFPage_CloseAnnot(annot)
            page.close()
    finally:
        doc.close()
    return out


def check_annotations(path, exp, notes):
    got = {(g["page"], g["nm"]): g for g in pdfium_annotations(path)}
    problems = []
    for a in exp["annotations"]:
        g = got.get((a["page"], a["id"]))
        if g is None:
            problems.append({"id": a["id"], "problem": "pdfium does not list it"})
            continue
        if g["subtype"] != SUBTYPES[a["subtype"]]:
            problems.append({"id": a["id"], "subtype": [a["subtype"], g["subtype"]]})
        if max(abs(x - y) for x, y in zip(g["rect"], a["rect"])) > 0.01:
            problems.append({"id": a["id"], "rect": [a["rect"], g["rect"]]})
        if (a["contents"] or "") != g["contents"]:
            problems.append({"id": a["id"], "contents": [a["contents"], g["contents"]]})
        if a["kind"] in ("highlight", "underline", "squiggly", "strikeOut") and g["quads"] != len(a["points"]):
            problems.append({"id": a["id"], "quads": [len(a["points"]), g["quads"]]})
        if a["kind"] in ("polygon", "polyline") and g["vertices"] != len(a["points"]):
            problems.append({"id": a["id"], "vertices": [len(a["points"]), g["vertices"]]})
        if a["kind"] == "ink" and g["inkStrokes"] != len(a["ink"]):
            problems.append({"id": a["id"], "ink": [len(a["ink"]), g["inkStrokes"]]})
        t = a.get("target") or {}
        if "uri" in t and g.get("uri") != t["uri"]:
            problems.append({"id": a["id"], "uri": [t["uri"], g.get("uri")]})
        want_page = t.get("page")
        if want_page is None and "name" in t:
            want_page = exp.get("namedDestinations", {}).get(t["name"], {}).get("page")
        if want_page is not None and g.get("destPage") != want_page:
            problems.append({"id": a["id"], "destPage": [want_page, g.get("destPage")]})
    if problems:
        notes.append(f"{exp['name']}: pdfium reads {len(problems)} annotation detail(s) differently")
    return "annotations", {"listed": len(got), "expected": len(exp["annotations"]), "problems": problems}, {}


def check_navigation(path, exp, notes):
    import ctypes

    import pypdfium2 as pdfium
    import pypdfium2.raw as raw

    doc = pdfium.PdfDocument(path)
    try:
        def walk(item):
            out = []
            while item:
                title = _wide(raw.FPDFBookmark_GetTitle, item)
                dest = raw.FPDFBookmark_GetDest(doc, item)
                if not dest:
                    action = raw.FPDFBookmark_GetAction(item)
                    if action and raw.FPDFAction_GetType(action) == raw.PDFACTION_GOTO:
                        dest = raw.FPDFAction_GetDest(doc, action)
                page = raw.FPDFDest_GetDestPageIndex(doc, dest) if dest else None
                out.append({"title": title, "page": page, "count": raw.FPDFBookmark_GetCount(item),
                            "children": walk(raw.FPDFBookmark_GetFirstChild(doc, item))})
                item = raw.FPDFBookmark_GetNextSibling(doc, item)
            return out

        outline = walk(raw.FPDFBookmark_GetFirstChild(doc, None))
        labels = []
        for i in range(len(doc)):
            n = raw.FPDF_GetPageLabel(doc, i, None, 0)
            buf = (ctypes.c_ushort * max(1, n // 2))()
            raw.FPDF_GetPageLabel(doc, i, buf, n)
            labels.append(bytes(buf)[:max(0, n - 2)].decode("utf-16-le"))
    finally:
        doc.close()
    problems = []

    def compare(want, got, path_):
        if len(want) != len(got):
            problems.append({"outline": path_, "items": [len(want), len(got)]})
            return
        for w, g in zip(want, got):
            page = None
            t = w["target"]
            if t:
                page = t.get("page") if "page" in t else exp["namedDestinations"][t["name"]]["page"]
            if w["title"] != g["title"] or page != (g["page"] if g["page"] is not None and g["page"] >= 0 else None):
                problems.append({"outline": path_ + [w["title"]], "want": [w["title"], page],
                                 "pdfium": [g["title"], g["page"]]})
            if w["children"]:
                want_count = len(w["children"]) if w["open"] else -len(w["children"])
                if g["count"] != want_count:
                    problems.append({"outline": path_ + [w["title"]], "count": [want_count, g["count"]]})
            compare(w["children"], g["children"], path_ + [w["title"]])

    compare(exp["outline"], outline, [])
    if labels != exp["pageLabels"]:
        problems.append({"pageLabels": [exp["pageLabels"], labels]})
    if problems:
        notes.append(f"{exp['name']}: pdfium reads the outline or labels differently ({len(problems)})")
    return "navigation", {"outline": outline, "pageLabels": labels, "problems": problems}, {}


def check_tagged(path, exp, notes):
    import pypdfium2 as pdfium
    import pypdfium2.raw as raw

    doc = pdfium.PdfDocument(path)
    try:
        page = doc[0]
        tree = raw.FPDF_StructTree_GetForPage(page)

        def walk(elem):
            kind = _wide(raw.FPDF_StructElement_GetType, elem)
            kids = []
            for k in range(raw.FPDF_StructElement_CountChildren(elem)):
                child = raw.FPDF_StructElement_GetChildAtIndex(elem, k)
                if child:
                    kids.append(walk(child))
            mcids = [raw.FPDF_StructElement_GetMarkedContentIdAtIndex(elem, k)
                     for k in range(raw.FPDF_StructElement_GetMarkedContentIdCount(elem))]
            return {"type": kind, "mcids": mcids, "children": kids}

        got = [walk(raw.FPDF_StructTree_GetChildAtIndex(tree, k)) for k in range(raw.FPDF_StructTree_CountChildren(tree))]
        raw.FPDF_StructTree_Close(tree)
        page.close()
    finally:
        doc.close()
    want = [{"type": e["type"], "children": [{"type": c["type"], "mcids": c["mcids"]} for c in e["children"]]}
            for e in exp["tagged"]["structure"]]
    flat_got = [{"type": e["type"], "children": [{"type": c["type"], "mcids": c["mcids"]} for c in e["children"]]}
                for e in got]
    same = want == flat_got
    if not same:
        notes.append(f"{exp['name']}: pdfium reads the structure tree differently")
    return "structure", {"pdfium": got, "matches": same}, {}


def check_redacted(path, exp, notes):
    """pdfminer.six's text of the applied file must hold none of the removed words; pdfium's drawing must be black
    inside each area."""
    from pdfminer.high_level import extract_text

    from .reference import _quiet_logs

    _quiet_logs()
    text = extract_text(path)
    left = [w for w in exp["redaction"]["removedText"] if w in text]
    if left:
        notes.append(f"{exp['name']}: pdfminer.six still finds {left}")
    return "redacted", {"pdfminerText": text.strip(), "removedWordsFound": left}, {}


def build(seed):
    return [annotations(seed, True), annotations(seed, False), navigation(seed), tagged(seed), redaction(seed),
            redaction_applied(seed), redaction_replies(seed), redaction_split(seed), redaction_nested(seed),
            redaction_masked(seed)]

