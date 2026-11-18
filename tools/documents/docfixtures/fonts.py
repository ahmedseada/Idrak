"""Fonts as the samples use them, each knowing what its codes show: the standard 14 (not embedded, the reference
reader's AFM widths), embedded TrueType as a simple font, composite fonts (Type0, Identity-H / Identity-V, CIDFontType2,
CIDToGIDMap identity or a stream), Type 3 fonts, and the ToUnicode CMaps.

Each font gives glyphs for text: the code bytes, the width (1/1000 of the size), the text the code stands for (what a
reader must give), and whether it is the single-byte code 32 (word spacing applies). Embedded programs are subset by
fontTools at the end, from the glyphs the pages used (timestamps kept, so the bytes repeat)."""
import hashlib
import io

import uharfbuzz as hb
from fontTools import subset
from fontTools.ttLib import TTFont
from pdfminer.fontmetrics import FONT_METRICS

from .encodings import glyph_text, simple_encoder
from .filters import flate
from .pdf import N, Stream, String

# The built-in encodings of Symbol and ZapfDingbats, for the codes the samples show (the published AFM files).
SYMBOL_CODES = {0x20: "space", 0x61: "alpha", 0x62: "beta", 0x67: "gamma", 0x64: "delta", 0x70: "pi", 0x73: "sigma",
                0x30: "zero", 0x31: "one", 0x32: "two", 0x33: "three", 0x34: "four", 0x2B: "plus", 0x3D: "equal"}
ZAPF_CODES = {0x20: "space", 0x34: "a20", 0x38: "a24", 0x48: "a35", 0x6C: "a71", 0x6E: "a73", 0x75: "a78"}
# The ITC Zapf Dingbats glyph list (zapfdingbats.txt, published with the Adobe Glyph List) for those names.
ZAPF_TEXT = {"space": " ", "a20": "✔", "a24": "✘", "a35": "★", "a71": "●", "a73": "■",
             "a78": "◆"}

# GPOS features left off when shaping, so every glyph advances by its own width (no offsets a Tj cannot carry).
NO_POSITIONING = {"kern": False, "mark": False, "mkmk": False, "curs": False, "dist": False, "abvm": False,
                  "blwm": False}


class Glyph:
    __slots__ = ("code", "width", "text", "space", "gid", "cluster")

    def __init__(self, code, width, text, space=False, gid=None, cluster=None):
        self.code = code
        self.width = width
        self.text = text
        self.space = space
        self.gid = gid
        self.cluster = cluster


def differences_array(codes):
    """A /Differences array for code -> glyph name: each run of consecutive codes after its first code."""
    out, prev = [], None
    for c in sorted(codes):
        if prev is None or c != prev + 1:
            out.append(c)
        out.append(N(codes[c]))
        prev = c
    return out


def subset_tag(data):
    digest = hashlib.sha256(data).digest()
    return "".join(chr(65 + b % 26) for b in digest[:6])


class Font:
    kind = "font"
    vertical = False
    two_byte = False

    def __init__(self, doc, resname):
        self.doc = doc
        self.resname = resname
        self.ref = doc.pdf.alloc()
        self.base_font = None
        self.ascent = 0.0
        self.descent = 0.0

    def glyphs(self, text):
        raise NotImplementedError

    def finish(self):
        raise NotImplementedError

    def describe(self):
        return {"resource": self.resname, "kind": self.kind, "baseFont": self.base_font, "ascent": self.ascent,
                "descent": self.descent}


class Standard14(Font):
    """A standard 14 font, not embedded: /Encoding a base encoding (or a dictionary with /Differences), or none
    (Symbol and ZapfDingbats keep their built-in encodings). Widths from the AFM unless /Widths is written."""

    kind = "standard14"

    def __init__(self, doc, resname, name, encoding="WinAnsiEncoding", differences=None, base_encoding=None,
                 widths=None, write_widths=False):
        super().__init__(doc, resname)
        self.base_font = name
        descriptor, afm = FONT_METRICS[name]
        self.afm = afm
        bbox = descriptor["FontBBox"]
        self.ascent = float(descriptor.get("Ascent", bbox[3]))
        self.descent = float(descriptor.get("Descent", bbox[1]))
        self.encoding = encoding
        self.differences = differences or {}
        self.base_encoding = base_encoding
        self.extra_widths = widths or {}
        self.write_widths = write_widths or bool(differences)
        if name == "Symbol":
            self.codes = {c: (g, glyph_text(g)) for c, g in SYMBOL_CODES.items()}
        elif name == "ZapfDingbats":
            self.codes = {c: (g, ZAPF_TEXT[g]) for c, g in ZAPF_CODES.items()}
        else:
            base = base_encoding if differences else encoding
            _, self.codes = simple_encoder(base, self.differences)
        self.reverse = {}
        for code in sorted(self.codes):
            g, t = self.codes[code]
            if t and t not in self.reverse:
                self.reverse[t] = code
        self.used = set()

    def afm_width(self, code):
        glyph, text = self.codes[code]
        if code in self.extra_widths:
            return self.extra_widths[code]
        if self.base_font == "ZapfDingbats":
            # pdfminer keys this table by the number in the aN names (chr(N)), so chr(32) is a32, not the space: the
            # AFM gives the space 278.
            return 278 if glyph == "space" else self.afm[chr(int(glyph[1:]))]
        return self.afm[text]

    def glyphs(self, text):
        out = []
        for ch in text:
            code = self.reverse[ch]
            self.used.add(code)
            out.append(Glyph(bytes([code]), self.afm_width(code), self.codes[code][1], code == 32))
        return out

    def code_glyphs(self, codes):
        """Glyphs for codes given directly (a code whose glyph name stands for several characters, or none)."""
        out = []
        for code in codes:
            self.used.add(code)
            out.append(Glyph(bytes([code]), self.afm_width(code), self.codes[code][1] or "�", code == 32))
        return out

    def finish(self):
        d = {"Type": N("Font"), "Subtype": N("Type1"), "BaseFont": N(self.base_font)}
        if self.differences:
            enc = {"Type": N("Encoding")}
            diffs = differences_array(self.differences)
            if self.base_encoding:
                enc["BaseEncoding"] = N(self.base_encoding)
            enc["Differences"] = diffs
            d["Encoding"] = enc
        elif self.encoding:
            d["Encoding"] = N(self.encoding)
        if self.write_widths:
            first, last_code = min(self.used), max(self.used)
            d["FirstChar"] = first
            d["LastChar"] = last_code
            d["Widths"] = [self.afm_width(c) if c in self.codes else 0 for c in range(first, last_code + 1)]
        self.doc.pdf.put(self.ref, d)


def _font_program(path, unicodes=None, gids=None, retain_gids=False, keep_cmap=True):
    """A subset of the font at `path` (no hinting, no layout tables), its bytes and the TTFont read back from them."""
    font = TTFont(path, recalcTimestamp=False, recalcBBoxes=False)
    opts = subset.Options()
    opts.layout_features = []
    opts.hinting = False
    opts.notdef_outline = True
    opts.retain_gids = retain_gids
    opts.name_IDs = [0, 1, 2, 3, 4, 5, 6, 13, 14]
    opts.name_languages = ["*"]
    opts.drop_tables += ["GSUB", "GPOS", "GDEF", "gasp", "DSIG", "fvar", "gvar", "avar", "HVAR", "MVAR", "STAT"]
    opts.recalc_timestamp = False
    s = subset.Subsetter(opts)
    s.populate(unicodes=unicodes or [], gids=gids or [])
    s.subset(font)
    if not keep_cmap:
        del font["cmap"]
    buf = io.BytesIO()
    font.save(buf, reorderTables=True)
    data = buf.getvalue()
    return data, TTFont(io.BytesIO(data))


def _units(value, upem):
    return round(value * 1000 / upem)


class TrueTypeSimple(Font):
    """An embedded TrueType font as a simple font (/Subtype /TrueType, one byte a code): nonsymbolic with a base
    encoding (codes to glyph names to Unicode to the font's (3,1) cmap), optionally with a ToUnicode CMap."""

    kind = "truetype"

    def __init__(self, doc, resname, path, encoding="WinAnsiEncoding", to_unicode=False):
        super().__init__(doc, resname)
        self.path = path
        self.full = TTFont(path, recalcTimestamp=False)
        self.upem = self.full["head"].unitsPerEm
        self.cmap = self.full.getBestCmap()
        self.hmtx = self.full["hmtx"]
        hhea = self.full["hhea"]
        self.ascent = float(_units(hhea.ascent, self.upem))
        self.descent = float(_units(hhea.descent, self.upem))
        self.encoding = encoding
        self.to_unicode = to_unicode
        _, self.codes = simple_encoder(encoding)
        self.reverse = {}
        for code in sorted(self.codes):
            g, t = self.codes[code]
            if t and t not in self.reverse:
                self.reverse[t] = code
        self.used = {}
        self.ps_name = self.full["name"].getDebugName(6) or "Font"

    def width(self, ch):
        return _units(self.hmtx[self.full.getBestCmap()[ord(ch)]][0], self.upem)

    def glyphs(self, text):
        out = []
        for ch in text:
            code = self.reverse[ch]
            if ord(ch) not in self.cmap:
                raise KeyError(f"{ch!r} is not in {self.path}")
            self.used[code] = ch
            out.append(Glyph(bytes([code]), self.width(ch), ch, code == 32))
        return out

    def finish(self):
        unicodes = sorted({ord(c) for c in self.used.values()})
        data, sub = _font_program(self.path, unicodes=unicodes)
        tag = subset_tag(data)
        self.base_font = f"{tag}+{self.ps_name}"
        first, last = min(self.used), max(self.used)
        widths = [self.width(self.used[c]) if c in self.used else 0 for c in range(first, last + 1)]
        program = self.doc.pdf.add(Stream({"Length1": len(data), "Filter": N("FlateDecode")}, flate(data), data))
        descriptor = self.doc.pdf.add(_descriptor(self.base_font, sub, self.upem, program, symbolic=False))
        d = {"Type": N("Font"), "Subtype": N("TrueType"), "BaseFont": N(self.base_font), "FirstChar": first,
             "LastChar": last, "Widths": widths, "Encoding": N(self.encoding), "FontDescriptor": descriptor}
        if self.to_unicode:
            pairs = {bytes([c]): t for c, t in self.used.items()}
            d["ToUnicode"] = self.doc.pdf.add(to_unicode_stream(pairs, 1, style="bfchar"))
        self.doc.pdf.put(self.ref, d)


def _descriptor(name, font, upem, program, symbolic, key="FontFile2"):
    head, hhea = font["head"], font["hhea"]
    os2 = font["OS/2"] if "OS/2" in font else None
    cap = getattr(os2, "sCapHeight", 0) if os2 is not None else 0
    return {"Type": N("FontDescriptor"), "FontName": N(name), "Flags": 4 if symbolic else 32,
            "FontBBox": [_units(head.xMin, upem), _units(head.yMin, upem), _units(head.xMax, upem),
                         _units(head.yMax, upem)],
            "ItalicAngle": 0, "Ascent": _units(hhea.ascent, upem), "Descent": _units(hhea.descent, upem),
            "CapHeight": _units(cap or hhea.ascent, upem), "StemV": 80, key: program}


class Type0(Font):
    """A composite font: Type0 over CIDFontType2 with the font program embedded. `encoding` Identity-H or Identity-V;
    `cid_map` "identity" (CID = glyph id) or "stream" (CID = the character's code point, a CIDToGIDMap stream);
    `to_unicode` writes the ToUnicode CMap (its `tu_style` "bfchar", "bfrange" or "mixed"); without it a reader goes
    through the embedded font's own cmap (reversed: the lowest code point of the glyph), else U+FFFD."""

    kind = "type0"
    two_byte = True

    def __init__(self, doc, resname, path, encoding="Identity-H", cid_map="identity", to_unicode=True,
                 tu_style="mixed", w_style="mixed", keep_cmap=True, dw2=(880, -1000)):
        super().__init__(doc, resname)
        self.path = path
        with open(path, "rb") as f:
            self.font_bytes = f.read()
        self.full = TTFont(path, recalcTimestamp=False)
        self.upem = self.full["head"].unitsPerEm
        self.cmap = self.full.getBestCmap()
        self.hmtx = self.full["hmtx"]
        self.order = self.full.getGlyphOrder()
        hhea = self.full["hhea"]
        self.ascent = float(_units(hhea.ascent, self.upem))
        self.descent = float(_units(hhea.descent, self.upem))
        self.encoding = encoding
        self.vertical = encoding.endswith("-V")
        self.cid_map = cid_map
        self.to_unicode = to_unicode
        self.tu_style = tu_style
        self.w_style = w_style
        self.keep_cmap = keep_cmap
        self.dw2 = dw2
        self.cid_text = {}  # cid -> text (for ToUnicode)
        self.cid_gid = {}
        self.ps_name = self.full["name"].getDebugName(6) or "Font"
        self.hb_font = hb.Font(hb.Face(hb.Blob(self.font_bytes)))
        self.sub_font = None

    def gid_width(self, gid):
        return _units(self.hmtx[self.order[gid]][0], self.upem)

    def _glyph(self, gid, text, cid=None):
        cid = gid if cid is None else cid
        if text and self.cid_text.get(cid, text) != text:
            # A ToUnicode CMap gives a code one text: a sample that needs two would expect what no reader can give.
            raise ValueError(f"CID {cid} stands for {self.cid_text[cid]!r} and {text!r}")
        if text:
            self.cid_text[cid] = text
        self.cid_gid[cid] = gid
        return Glyph(cid.to_bytes(2, "big"), self.gid_width(gid), text, False, gid)

    def shape(self, text, direction="ltr"):
        """Glyphs of `text` shaped by HarfBuzz (GSUB only) in visual order; each glyph's text is its cluster's."""
        buf = hb.Buffer()
        buf.add_codepoints([ord(c) for c in text])
        buf.direction = direction
        buf.guess_segment_properties()
        hb.shape(self.hb_font, buf, NO_POSITIONING)
        infos, positions = buf.glyph_infos, buf.glyph_positions
        starts = sorted({i.cluster for i in infos})
        out = []
        for info, pos in zip(infos, positions):
            if pos.x_offset or pos.y_offset:
                raise ValueError(f"{text!r}: glyph {info.codepoint} has an offset")
            if pos.x_advance != self.hmtx[self.order[info.codepoint]][0]:
                raise ValueError(f"{text!r}: glyph {info.codepoint} advances {pos.x_advance}, not its width")
            k = starts.index(info.cluster)
            end = starts[k + 1] if k + 1 < len(starts) else len(text)
            if sum(1 for i in infos if i.cluster == info.cluster) > 1:
                raise ValueError(f"{text!r}: cluster {info.cluster} has several glyphs")
            g = self._glyph(info.codepoint, text[info.cluster:end])
            g.cluster = info.cluster
            out.append(g)
        return out

    def glyphs(self, text):
        """Unshaped glyphs, one per character through the font's cmap (vertical text, CID = code point maps)."""
        out = []
        for ch in text:
            gid = self.order.index(self.cmap[ord(ch)])
            cid = ord(ch) if self.cid_map == "stream" else gid
            out.append(self._glyph(gid, ch, cid))
        return out

    def finish(self):
        gids = sorted(set(self.cid_gid.values()) | {0})
        unicodes = sorted({ord(c) for t in self.cid_text.values() for c in t if ord(c) in self.cmap})
        data, sub = _font_program(self.path, unicodes=unicodes, gids=gids, retain_gids=True, keep_cmap=self.keep_cmap)
        self.sub_font = sub
        tag = subset_tag(data)
        self.base_font = f"{tag}+{self.ps_name}"
        program = self.doc.pdf.add(Stream({"Length1": len(data), "Filter": N("FlateDecode")}, flate(data), data))
        descriptor = self.doc.pdf.add(_descriptor(self.base_font, sub, self.upem, program, symbolic=True))
        cids = sorted(self.cid_gid)
        cid_font = {"Type": N("Font"), "Subtype": N("CIDFontType2"), "BaseFont": N(self.base_font),
                    "CIDSystemInfo": {"Registry": String(b"Adobe"), "Ordering": String(b"Identity"), "Supplement": 0},
                    "FontDescriptor": descriptor, "DW": 1000, "W": w_array(
                        {c: self.gid_width(self.cid_gid[c]) for c in cids}, self.w_style)}
        if self.vertical:
            cid_font["DW2"] = list(self.dw2)
        if self.cid_map == "stream":
            top = max(cids)
            table = bytearray(2 * (top + 1))
            for c in cids:
                table[2 * c:2 * c + 2] = self.cid_gid[c].to_bytes(2, "big")
            cid_font["CIDToGIDMap"] = self.doc.pdf.add(Stream({"Filter": N("FlateDecode")}, flate(bytes(table)),
                                                              bytes(table)))
        else:
            cid_font["CIDToGIDMap"] = N("Identity")
        d = {"Type": N("Font"), "Subtype": N("Type0"), "BaseFont": N(self.base_font + "-" + self.encoding),
             "Encoding": N(self.encoding), "DescendantFonts": [self.doc.pdf.add(cid_font)]}
        if self.to_unicode:
            pairs = {c.to_bytes(2, "big"): t for c, t in self.cid_text.items()}
            d["ToUnicode"] = self.doc.pdf.add(to_unicode_stream(pairs, 2, style=self.tu_style))
        self.doc.pdf.put(self.ref, d)

    def reverse_cmap(self):
        """Glyph id -> the lowest code point the embedded program's cmap maps to it (the reader's third way)."""
        if self.sub_font is None or "cmap" not in self.sub_font:
            return {}
        order = self.sub_font.getGlyphOrder()
        out = {}
        for cp, name in sorted(self.sub_font.getBestCmap().items()):
            gid = order.index(name)
            out.setdefault(gid, chr(cp))
        return out


def w_array(widths, style):
    """A /W array: "list" (c [w1 w2 ...]), "range" (cfirst clast w) or "mixed" (ranges where widths repeat)."""
    cids = sorted(widths)
    out = []
    i = 0
    while i < len(cids):
        j = i
        if style in ("range", "mixed"):
            while j + 1 < len(cids) and cids[j + 1] == cids[j] + 1 and widths[cids[j + 1]] == widths[cids[i]]:
                j += 1
            if j > i or style == "range":
                out += [cids[i], cids[j], widths[cids[i]]]
                i = j + 1
                continue
        j = i
        group = [widths[cids[i]]]
        while j + 1 < len(cids) and cids[j + 1] == cids[j] + 1 and (style == "list" or widths[cids[j + 1]] != widths[cids[j]]):
            j += 1
            group.append(widths[cids[j]])
        out += [cids[i], group]
        i = j + 1
    return out


def utf16_hex(text):
    return text.encode("utf-16-be").hex().upper()


def to_unicode_stream(pairs, nbytes, style="bfchar"):
    """A ToUnicode CMap: code (bytes) -> text. style "bfchar" (one line a code), "bfrange" (ranges of consecutive codes
    and consecutive single characters, else ranges with arrays) or "mixed" (ranges where they fit, bfchar for the rest)."""
    codes = sorted(pairs)
    lines = ["/CIDInit /ProcSet findresource begin", "12 dict begin", "begincmap",
             "/CIDSystemInfo << /Registry (Adobe) /Ordering (UCS) /Supplement 0 >> def",
             "/CMapName /Adobe-Identity-UCS def", "/CMapType 2 def", "1 begincodespacerange",
             "<%s> <%s>" % ("00" * nbytes, "FF" * nbytes), "endcodespacerange"]
    chars, ranges = [], []
    i = 0
    while i < len(codes):
        j = i
        if style in ("bfrange", "mixed"):
            while (j + 1 < len(codes) and int.from_bytes(codes[j + 1], "big") == int.from_bytes(codes[j], "big") + 1
                   and codes[j + 1][:-1] == codes[i][:-1]):
                j += 1
        if j > i:
            texts = [pairs[c] for c in codes[i:j + 1]]
            seq = all(len(t) == 1 for t in texts) and all(ord(texts[k + 1]) == ord(texts[k]) + 1 for k in range(len(texts) - 1))
            if seq:
                ranges.append("<%s> <%s> <%s>" % (codes[i].hex().upper(), codes[j].hex().upper(), utf16_hex(texts[0])))
            else:
                ranges.append("<%s> <%s> [%s]" % (codes[i].hex().upper(), codes[j].hex().upper(),
                                                  " ".join("<%s>" % utf16_hex(t) for t in texts)))
        elif style == "bfrange":
            ranges.append("<%s> <%s> <%s>" % (codes[i].hex().upper(), codes[i].hex().upper(), utf16_hex(pairs[codes[i]])))
        else:
            chars.append("<%s> <%s>" % (codes[i].hex().upper(), utf16_hex(pairs[codes[i]])))
        i = j + 1
    for k in range(0, len(chars), 100):
        block = chars[k:k + 100]
        lines += ["%d beginbfchar" % len(block)] + block + ["endbfchar"]
    for k in range(0, len(ranges), 100):
        block = ranges[k:k + 100]
        lines += ["%d beginbfrange" % len(block)] + block + ["endbfrange"]
    lines += ["endcmap", "CMapName currentdict /CMap defineresource pop", "end", "end"]
    data = ("\n".join(lines) + "\n").encode("ascii")
    return Stream({"Filter": N("FlateDecode")}, flate(data), data)


class Type3(Font):
    """A Type 3 font whose glyphs are drawn by their own content streams (filled shapes, a letter's rough outline),
    named by /Differences so the glyph names give the text. FontMatrix 0.001: widths in thousandths like the others."""

    kind = "type3"

    def __init__(self, doc, resname, procs):
        super().__init__(doc, resname)
        self.procs = procs  # glyph name -> (width, content bytes)
        self.names = sorted(procs)
        self.codes = {}
        code = 65
        for name in self.names:
            if name == "space":
                self.codes[32] = name
            else:
                self.codes[code] = name
                code += 1
        self.text = {c: glyph_text(n) for c, n in self.codes.items()}
        self.reverse = {t: c for c, t in sorted(self.text.items(), reverse=True)}
        self.ascent, self.descent = 800.0, -200.0
        self.base_font = None

    def glyphs(self, text):
        return [Glyph(bytes([self.reverse[ch]]), self.procs[self.codes[self.reverse[ch]]][0], ch,
                      self.reverse[ch] == 32) for ch in text]

    def finish(self):
        procs = {}
        for name in self.names:
            w, content = self.procs[name]
            procs[name] = self.doc.pdf.add(Stream({}, content))
        first, last = min(self.codes), max(self.codes)
        diffs = differences_array(self.codes)
        d = {"Type": N("Font"), "Subtype": N("Type3"), "FontBBox": [0, -200, 1000, 800],
             "FontMatrix": [0.001, 0, 0, 0.001, 0, 0], "CharProcs": procs,
             "Encoding": {"Type": N("Encoding"), "Differences": diffs}, "FirstChar": first, "LastChar": last,
             "Widths": [self.procs[self.codes[c]][0] if c in self.codes else 0 for c in range(first, last + 1)],
             "Resources": {}}
        self.doc.pdf.put(self.ref, d)
