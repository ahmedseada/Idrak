"""Documents, pages and content streams. The content writer keeps the graphics and text state as a reader's interpreter
does (q/Q, cm, BT/ET, Tf, Td, TD, Tm, T*, TL, Tc, Tw, Tz, Ts, Tr, Tj, TJ, ', ", Do, BDC/EMC with /ActualText), so each
run's box, font size, hidden flag and order come from the very operators it writes.

Geometry, as every sample's JSON gives it (points, the page as shown: after its CropBox and its /Rotate, origin at the
top left, y down):
- a glyph's box: along the baseline its advance (its width, 1/1000 of the size), across it from the font's descent to
  its ascent (the FontDescriptor's /Ascent and /Descent; the standard 14: the AFM's Ascender and Descender, or the
  FontBBox's y range where the AFM has none; Type 3: its FontBBox), raised by the text rise. Vertical fonts: across the
  column the glyph's width centred on the origin, along it the vertical advance (/DW2 or /W2).
- a run (one text-showing operator: Tj, ', ", or a whole TJ; or one marked-content span with /ActualText): the
  bounding box of its glyphs' boxes, mapped through the text matrix, the CTM and the page's crop and rotation.
- its font size: the size times the length the text matrix and CTM give a unit vertical vector.
"""
import contextlib
import hashlib
import math

from .filters import flate
from .pdf import N, PdfFile, Stream, String, U16, Writer, escape_literal, fmt_num, ser

IDENT = (1.0, 0.0, 0.0, 1.0, 0.0, 0.0)


def mul(m, n):
    a, b, c, d, e, f = m
    A, B, C, D, E, F = n
    return (a * A + b * C, a * B + b * D, c * A + d * C, c * B + d * D, e * A + f * C + E, e * B + f * D + F)


def apply(m, x, y):
    return (x * m[0] + y * m[2] + m[4], x * m[1] + y * m[3] + m[5])


def r4(x):
    v = round(x, 4)
    return 0.0 if v == 0 else v


def fmt(*xs):
    return " ".join(fmt_num(float(x)) if isinstance(x, float) else str(x) for x in xs)


class Page:
    def __init__(self, doc, width, height, rotate=0, media=None, crop=None, inherit=None):
        self.doc = doc
        self.media = list(media or [0, 0, width, height])
        self.crop = list(crop) if crop else None
        self.rotate = rotate
        self.inherit = inherit or {}  # entries the page takes from its parent (rotate, media box, resources)
        self.resources = {}
        self.runs = []
        self.images = []
        self.lines = []
        self.ref = doc.pdf.alloc()
        self.content = Content(self, IDENT)
        self.content_stream = None  # set by finish: the Stream written
        self.content_encoder = None  # bytes -> (encoded, extra dict entries); default Flate
        self.extra = {}
        self.order = 0
        self.annotations = []
        self.issues = {}

    def next_order(self):
        self.order += 1
        return self.order - 1

    def box(self):
        return self.crop or self.media

    def size(self):
        x0, y0, x1, y1 = self.box()
        w, h = x1 - x0, y1 - y0
        return (h, w) if self.rotate % 180 else (w, h)

    def display(self, x, y):
        x0, y0, x1, y1 = self.box()
        w, h = x1 - x0, y1 - y0
        u, v = x - x0, y - y0
        r = self.rotate % 360
        if r == 0:
            return (u, h - v)
        if r == 90:
            return (v, u)
        if r == 180:
            return (w - u, v)
        return (h - v, w - u)

    def display_box(self, points):
        pts = [self.display(x, y) for x, y in points]
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        return [r4(min(xs)), r4(min(ys)), r4(max(xs) - min(xs)), r4(max(ys) - min(ys))]

    def resource(self, kind, name, ref):
        self.resources.setdefault(kind, {})[name] = ref

    def line(self, text, direction="ltr"):
        self.lines.append({"text": text, "direction": direction})


class Content:
    def __init__(self, page, ctm, resources=None):
        self.page = page
        self.resources = resources  # a form's own resources (None: the page's)
        self.ops = []
        self.ctm = ctm
        self.stack = []
        self.font = None
        self.size = 0.0
        self.tc = 0.0
        self.tw = 0.0
        self.th = 1.0
        self.tl = 0.0
        self.rise = 0.0
        self.mode = 0
        self.tm = IDENT
        self.tlm = IDENT
        self.span = None

    # -- output ----------------------------------------------------------------------------------------------------------

    def emit(self, text):
        self.ops.append(text.encode("latin-1") if isinstance(text, str) else bytes(text))

    def data(self):
        return b"\n".join(self.ops) + b"\n"

    def _use(self, kind, name, ref):
        if self.resources is not None:
            self.resources.setdefault(kind, {})[name] = ref
        else:
            self.page.resource(kind, name, ref)

    # -- graphics state ----------------------------------------------------------------------------------------------

    def q(self):
        self.stack.append((self.ctm, self.font, self.size, self.tc, self.tw, self.th, self.tl, self.rise, self.mode))
        self.emit("q")

    def Q(self):
        (self.ctm, self.font, self.size, self.tc, self.tw, self.th, self.tl, self.rise, self.mode) = self.stack.pop()
        self.emit("Q")

    def cm(self, a, b, c, d, e, f):
        self.ctm = mul((a, b, c, d, e, f), self.ctm)
        self.emit(fmt(a, b, c, d, e, f) + " cm")

    # -- text --------------------------------------------------------------------------------------------------------

    def BT(self):
        self.tm = self.tlm = IDENT
        self.emit("BT")

    def ET(self):
        self.emit("ET")

    def Tf(self, font, size):
        self.font = font
        self.size = float(size)
        self._use("Font", font.resname, font.ref)
        self.emit(f"/{font.resname} {fmt(float(size))} Tf")

    def Td(self, x, y):
        self.tlm = self.tm = mul((1, 0, 0, 1, x, y), self.tlm)
        self.emit(fmt(float(x), float(y)) + " Td")

    def TD(self, x, y):
        self.tl = -float(y)
        self.tlm = self.tm = mul((1, 0, 0, 1, x, y), self.tlm)
        self.emit(fmt(float(x), float(y)) + " TD")

    def Tm(self, a, b, c, d, e, f):
        self.tlm = self.tm = (a, b, c, d, e, f)
        self.emit(fmt(*(float(v) for v in (a, b, c, d, e, f))) + " Tm")

    def Tstar(self):
        self.tlm = self.tm = mul((1, 0, 0, 1, 0, -self.tl), self.tlm)
        self.emit("T*")

    def TL(self, v):
        self.tl = float(v)
        self.emit(fmt(float(v)) + " TL")

    def Tc(self, v):
        self.tc = float(v)
        self.emit(fmt(float(v)) + " Tc")

    def Tw(self, v):
        self.tw = float(v)
        self.emit(fmt(float(v)) + " Tw")

    def Tz(self, v):
        self.th = float(v) / 100
        self.emit(fmt(float(v)) + " Tz")

    def Ts(self, v):
        self.rise = float(v)
        self.emit(fmt(float(v)) + " Ts")

    def Tr(self, v):
        self.mode = int(v)
        self.emit(f"{int(v)} Tr")

    def _string(self, glyphs, hex_=None):
        data = b"".join(g.code for g in glyphs)
        if hex_ or (hex_ is None and self.font.two_byte):
            return b"<" + data.hex().upper().encode("ascii") + b">"
        return escape_literal(data)

    def _glyph_quad(self, g):
        f = self.font
        trm = mul((self.size * self.th, 0, 0, self.size, 0, self.rise), mul(self.tm, self.ctm))
        w = g.width / 1000
        if f.vertical:
            w1 = f.dw2[1] / 1000
            corners = [(-w / 2, w1), (w / 2, w1), (w / 2, 0), (-w / 2, 0)]
        else:
            a, d = f.ascent / 1000, f.descent / 1000
            corners = [(0, d), (w, d), (w, a), (0, a)]
        return [apply(trm, x, y) for x, y in corners], apply(trm, 0, 0)

    def _advance(self, g=None, adjust=0.0):
        if self.font.vertical:
            ty = (self.font.dw2[1] / 1000 * self.size + self.tc + (self.tw if g.space else 0)) if g else 0.0
            ty -= adjust / 1000 * self.size
            self.tm = mul((1, 0, 0, 1, 0, ty), self.tm)
        else:
            tx = ((g.width / 1000 * self.size + self.tc + (self.tw if g.space else 0)) * self.th) if g else 0.0
            tx -= adjust / 1000 * self.size * self.th
            self.tm = mul((1, 0, 0, 1, tx, 0), self.tm)

    def _run(self, items):
        quads, texts, origin = [], [], None
        glyphs = []
        for it in items:
            if isinstance(it, (int, float)):
                self._advance(None, it)
                continue
            quad, o = self._glyph_quad(it)
            if origin is None:
                origin = o
            quads += quad
            texts.append(it.text)
            glyphs.append(it)
            self._advance(it)
        trm_end = mul((self.size * self.th, 0, 0, self.size, 0, self.rise), mul(self.tm, self.ctm))
        end = apply(trm_end, 0, 0)
        m = mul(self.tm, self.ctm)
        size = self.size * math.hypot(m[2], m[3])
        rec = {"text": "".join(texts), "quads": quads, "origin": origin, "end": end, "font": self.font,
               "size": size, "hidden": self.mode in (3, 7), "glyphs": glyphs, "mode": self.mode}
        if self.span is not None:
            self.span["parts"].append(rec)
        else:
            rec["order"] = self.page.next_order()
            self.page.runs.append(rec)
        return rec

    def show(self, text=None, glyphs=None, op="Tj", hex_=None, aw=None, ac=None):
        glyphs = glyphs if glyphs is not None else self.font.glyphs(text)
        s = self._string(glyphs, hex_)
        if op == "'":
            self.tlm = self.tm = mul((1, 0, 0, 1, 0, -self.tl), self.tlm)
            self.emit(s + b" '")
        elif op == '"':
            self.tw, self.tc = float(aw), float(ac)
            self.tlm = self.tm = mul((1, 0, 0, 1, 0, -self.tl), self.tlm)
            self.emit(fmt(float(aw), float(ac)).encode("ascii") + b" " + s + b' "')
        else:
            self.emit(s + b" Tj")
        return self._run(glyphs)

    def TJ(self, items, hex_=None):
        """items: text (shown in the current font), glyph lists, or numbers (thousandths of the size, subtracted)."""
        parts, flat = [], []
        for it in items:
            if isinstance(it, (int, float)):
                parts.append(fmt(it).encode("ascii"))
                flat.append(it)
            else:
                gl = self.font.glyphs(it) if isinstance(it, str) else it
                parts.append(self._string(gl, hex_))
                flat += gl
        self.emit(b"[" + b" ".join(parts) + b"] TJ")
        return self._run(flat)

    @contextlib.contextmanager
    def actual_text(self, text, hex_=False):
        """A /Span with /ActualText: the span is one run whose text is `text`, over its glyphs' boxes."""
        value = U16(text, hex=True) if hex_ else (String(text.encode("latin-1")) if all(ord(c) < 128 for c in text)
                                                  else U16(text))
        self.emit(b"/Span <</ActualText " + ser(value) + b">> BDC")
        self.span = {"parts": []}
        yield
        span, self.span = self.span, None
        self.emit("EMC")
        parts = span["parts"]
        rec = {"text": text, "quads": [q for p in parts for q in p["quads"]], "origin": parts[0]["origin"],
               "end": parts[-1]["end"], "font": parts[0]["font"], "size": parts[0]["size"],
               "hidden": parts[0]["hidden"], "glyphs": [g for p in parts for g in p["glyphs"]], "actualText": True,
               "glyphText": "".join(p["text"] for p in parts), "order": self.page.next_order(), "mode": parts[0]["mode"]}
        self.page.runs.append(rec)

    # -- images and forms --------------------------------------------------------------------------------------------

    def _image_record(self, info):
        quad = [apply(self.ctm, x, y) for x, y in ((0, 0), (1, 0), (1, 1), (0, 1))]
        rec = dict(info)
        rec["quad"] = quad
        rec["ctm"] = [r4(v) for v in self.ctm]
        rec["order"] = self.page.next_order()
        self.page.images.append(rec)
        return rec

    def image(self, name, image, matrix):
        """Draws image XObject `image` (an images.Image) as /name with `matrix` (q cm Do Q)."""
        self.q()
        self.cm(*matrix)
        self._use("XObject", name, image.ref)
        self.emit(f"/{name} Do")
        rec = self._image_record({"resource": name, **image.describe()})
        self.Q()
        return rec

    def inline_image(self, image, matrix, dict_bytes, data):
        self.q()
        self.cm(*matrix)
        self.emit(b"BI " + dict_bytes + b" ID " + data + b" EI")
        rec = self._image_record({"resource": None, **image.describe()})
        self.Q()
        return rec

    @contextlib.contextmanager
    def form(self, name, bbox, matrix=IDENT, resources=None):
        """A form XObject drawn here (/name Do); what the block writes into the yielded content is the form's."""
        child = Content(self.page, mul(matrix, self.ctm), resources if resources is not None else {})
        yield child
        data = child.data()
        d = {"Type": N("XObject"), "Subtype": N("Form"), "BBox": list(bbox), "Resources": child.resources,
             "Filter": N("FlateDecode")}
        if matrix != IDENT:
            d["Matrix"] = list(matrix)
        ref = self.page.doc.pdf.add(Stream(d, flate(data), data))
        self._use("XObject", name, ref)
        self.emit(f"/{name} Do")


def run_json(page, rec):
    font = rec["font"]
    quads = rec["quads"]
    box = page.display_box(quads) if quads else None
    o = page.display(*rec["origin"]) if rec["origin"] else None
    e = page.display(*rec["end"])
    angle = math.degrees(math.atan2(e[1] - o[1], e[0] - o[0])) if o and (e[0] - o[0] or e[1] - o[1]) else 0.0
    if font.vertical:
        angle = math.degrees(math.atan2(e[1] - o[1], e[0] - o[0])) - 90 if o else 0.0
    out = {"order": rec["order"], "text": rec["text"], "box": box,
           "origin": [r4(o[0]), r4(o[1])] if o else None, "end": [r4(e[0]), r4(e[1])],
           "fontSize": r4(rec["size"]), "font": font.base_font, "fontResource": font.resname, "fontKind": font.kind,
           "ascent": font.ascent, "descent": font.descent, "vertical": font.vertical, "hidden": rec["hidden"],
           "renderMode": rec["mode"], "angle": r4(((angle + 180) % 360) - 180)}
    if rec.get("actualText"):
        out["actualText"] = True
        out["glyphText"] = rec["glyphText"]
    unmapped = sum(1 for g in rec["glyphs"] if g.text == "�")
    if unmapped:
        out["unmappedGlyphs"] = unmapped
    return out


class Doc:
    """One sample document: its pages, fonts and objects, written by `write`."""

    def __init__(self, name, version="1.7", seed="2026"):
        self.name = name
        self.seed = seed
        self.pdf = PdfFile(version)
        self.catalog = self.pdf.alloc()
        self.pages_ref = self.pdf.alloc()
        self.pages = []
        self.fonts = []
        self.info = None  # dict of PDF entries
        self.info_expected = None
        self.catalog_extra = {}
        self.attachments = []
        self.page_tree = None  # a function (doc) -> writes the tree itself
        self.file_id = hashlib.md5(f"{seed}|{name}".encode("utf-8")).digest()
        self.notes = {}

    def font(self, cls, *args, **kw):
        f = cls(self, f"F{len(self.fonts) + 1}", *args, **kw)
        self.fonts.append(f)
        return f

    def page(self, width=420, height=300, **kw):
        p = Page(self, width, height, **kw)
        self.pages.append(p)
        return p

    def finish(self):
        """Fonts, content streams, pages, the page tree, the catalog and the information dictionary as objects."""
        for f in self.fonts:
            f.finish()
        for p in self.pages:
            data = p.content.data()
            if p.content_encoder is not None:
                result = p.content_encoder(data)
                encoded, extra = result[0], result[1]
                if len(result) > 2:
                    data = result[2]
            else:
                encoded, extra = flate(data), {"Filter": N("FlateDecode")}
            p.content_stream = Stream(dict(extra), encoded, data, length=p.extra.pop("_length", None))
            contents = self.pdf.add(p.content_stream)
            d = {"Type": N("Page"), "Parent": self.pages_ref}
            if "MediaBox" not in p.inherit:
                d["MediaBox"] = p.media
            if p.crop:
                d["CropBox"] = p.crop
            if p.rotate and "Rotate" not in p.inherit:
                d["Rotate"] = p.rotate
            if "Resources" not in p.inherit:
                d["Resources"] = p.resources
            d["Contents"] = contents
            if p.annotations:
                d["Annots"] = p.annotations
            d.update(p.extra)
            self.pdf.put(p.ref, d)
        if self.page_tree is not None:
            self.page_tree(self)
        else:
            self.pdf.put(self.pages_ref, {"Type": N("Pages"), "Kids": [p.ref for p in self.pages],
                                          "Count": len(self.pages)})
        cat = {"Type": N("Catalog"), "Pages": self.pages_ref}
        cat.update(self.catalog_extra)
        self.pdf.put(self.catalog, cat)
        if self.info is not None:
            self.info_ref = self.pdf.add(self.info)
        else:
            self.info_ref = None

    def trailer(self):
        t = {"Root": self.catalog}
        if self.info_ref is not None:
            t["Info"] = self.info_ref
        t["ID"] = [String(self.file_id, hex=True), String(self.file_id, hex=True)]
        return t

    def write(self, kind="table", security=None, objstm_size=50, predictor=True, extends=False):
        """The whole file as one revision of `kind`; returns (bytes, writer)."""
        w = Writer(self.pdf, security)
        trailer = self.trailer()
        nums = sorted(self.pdf.objects)
        if security is not None:
            enc = self.pdf.add(security.dictionary())
            self.encrypt_ref = enc
            w.no_crypt.add(enc.num)
            trailer["Encrypt"] = enc
            nums = sorted(self.pdf.objects)
        w.revision(nums, kind, trailer, objstm={"extends": extends}, objstm_size=objstm_size,
                   predictor=predictor, first=True)
        return bytes(w.buf), w

    def expected_pages(self):
        out = []
        for i, p in enumerate(self.pages):
            w, h = p.size()
            runs = [run_json(p, r) for r in sorted(p.runs, key=lambda r: r["order"])]
            images = []
            for rec in sorted(p.images, key=lambda r: r["order"]):
                img = {k: v for k, v in rec.items() if k not in ("quad",)}
                img["box"] = p.display_box(rec["quad"])
                images.append(img)
            unmapped = sum(r.get("unmappedGlyphs", 0) for r in runs)
            page = {"index": i, "width": r4(w), "height": r4(h), "rotation": p.rotate % 360,
                    "mediaBox": p.media, "cropBox": p.crop or p.media,
                    "text": "\n".join(line["text"] for line in p.lines), "lines": p.lines,
                    "runs": runs, "images": images,
                    "issues": {"unmappedGlyphs": unmapped, **p.issues}}
            out.append(page)
        return out
