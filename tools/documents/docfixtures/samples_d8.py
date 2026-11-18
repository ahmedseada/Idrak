"""D8, fonts in the common formats, from the repository's test fonts only (their licences beside them: DejaVu Sans and
the fonts made from it under LICENSE-DejaVu.txt, Amiri under the SIL Open Font License, OFL-amiri.txt): each font as
WOFF (zlib) and WOFF2 (Brotli; TrueType with the glyf/loca transform, and once with the hmtx transform too), TrueType
outlines written as CFF (OpenType .otf, quadratics raised to cubics exactly), the CFF font written as TrueType (cubics
to quadratics within a stated tolerance, fontTools' cu2qu), and subsets of a fixed glyph list, with the layout tables
dropped (what a PDF embedding needs) and kept with their closure. Beside each: its dump (fontdump.py) and HarfBuzz's
shaping of generic sample lines; the source fonts' dumps and shaping are samples of their own (no font file: the
font is the repository's)."""
import io
import os

from fontTools import subset
from fontTools.fontBuilder import FontBuilder
from fontTools.pens.t2CharStringPen import T2CharStringPen
from fontTools.pens.ttGlyphPen import TTGlyphPen
from fontTools.pens.cu2quPen import Cu2QuPen
from fontTools.ttLib import TTFont, newTable

from .common import FONTS, ROOT, Sample
from .fontdump import dump, dump_bytes, shape
from .pdf import sha256

D8 = ["D8"]

LATIN = ["Quarterly report: the office moves in May.", "fi fl ffi office café 2024", "AVATAR To Ta Wa"]
ARABIC = ["تقرير المبيعات الشهري", "اجتماع الفريق يوم الثلاثاء", "لا ١٢٣ سعر"]
FEATURE_LINES = ["ba ca da", "quiet qu", "basket pitch", "cat dog catdog", "mem nen menem", "AVATAR To xx",
                 "oowwo vvzv"]

SOURCES = [
    # name, file under tests/Idrak.Tests/data/fonts, licence file, sample lines, subset characters
    ("dejavu-sans-subset", "dejavu-sans-subset.ttf", "LICENSE-DejaVu.txt", LATIN + ARABIC[:2],
     "Report 2024 office fi سلام"),
    ("dejavu-sans-cff", "dejavu-sans-cff.otf", "LICENSE-DejaVu.txt", LATIN + ARABIC[:2], "Report 2024 office fi سلام"),
    ("features", "features.ttf", "LICENSE-DejaVu.txt", FEATURE_LINES, "bacdquiet"),
    ("amiri", "arabic/amiri.ttf", "arabic/OFL-amiri.txt", ARABIC, "سلام عليكم ١٢"),
]


def _rel(path):
    return os.path.relpath(path, ROOT).replace(os.sep, "/")


def _save(font, flavor=None, transform=None):
    font.flavor = flavor
    if flavor == "woff2" and transform is not None:
        from fontTools.ttLib import woff2

        buf = io.BytesIO()
        tmp = io.BytesIO()
        font.flavor = None
        font.save(tmp, reorderTables=True)
        woff2.compress(io.BytesIO(tmp.getvalue()), buf, transform_tables=transform)
        return buf.getvalue()
    buf = io.BytesIO()
    font.save(buf, reorderTables=True)
    return buf.getvalue()


def _load(path):
    return TTFont(path, recalcTimestamp=False, recalcBBoxes=False)


def to_cff(src_path):
    """TrueType outlines as CFF: every glyph drawn through T2CharStringPen (quadratics raised to cubics exactly, the
    points kept as fixed-point numbers, not rounded), the other tables kept, maxp 0.5, post 3."""
    font = _load(src_path)
    gs = font.getGlyphSet()
    order = font.getGlyphOrder()
    charstrings = {}
    for n in order:
        pen = T2CharStringPen(font["hmtx"][n][0], gs, roundTolerance=0)
        gs[n].draw(pen)
        charstrings[n] = pen.getCharString()
    name = font["name"].getDebugName(6) or "Font"
    for tag in ("glyf", "loca", "cvt ", "fpgm", "prep", "gasp", "hdmx", "LTSH", "VDMX"):
        if tag in font:
            del font[tag]
    fb = FontBuilder(font=font)  # no glyf any more: a CFF font
    fb.setupCFF(name, {"FullName": font["name"].getDebugName(4) or name,
                       "FamilyName": font["name"].getDebugName(1) or name,
                       "Weight": "Book", "version": "1.0", "Notice": font["name"].getDebugName(0) or ""},
                charstrings, {})
    font.sfntVersion = "OTTO"
    maxp = newTable("maxp")
    maxp.tableVersion = 0x5000
    maxp.numGlyphs = len(order)
    font["maxp"] = maxp
    font["post"].formatType = 3.0
    font["head"].indexToLocFormat = 0
    return _save(font)


def to_truetype(src_path, max_err=1.0):
    """CFF outlines as TrueType: each cubic turned into quadratics within `max_err` font units (cu2qu), glyf and loca
    built, the other tables kept, maxp 1.0, post 2 with the CFF's glyph names."""
    font = _load(src_path)
    gs = font.getGlyphSet()
    order = font.getGlyphOrder()
    glyphs = {}
    for n in order:
        pen = TTGlyphPen(gs)
        gs[n].draw(Cu2QuPen(pen, max_err, reverse_direction=True))
        glyphs[n] = pen.glyph()
    fb = FontBuilder(font=font)
    fb.isTTF = True  # it reads the kind from the tables, and glyf is not there yet
    del font["CFF "]
    font.sfntVersion = "\x00\x01\x00\x00"
    fb.setupGlyf(glyphs)
    font.recalcBBoxes = True  # glyph bounds, head's box and maxp's counts from the new outlines
    maxp = newTable("maxp")
    maxp.tableVersion = 0x10000
    for k in ("maxZones", "maxTwilightPoints", "maxStorage", "maxFunctionDefs", "maxInstructionDefs",
              "maxStackElements", "maxSizeOfInstructions", "maxComponentElements", "maxPoints", "maxContours",
              "maxCompositePoints", "maxCompositeContours", "maxComponentDepth"):
        setattr(maxp, k, 0)
    maxp.maxZones = 1
    maxp.numGlyphs = len(order)
    font["maxp"] = maxp
    font["post"].formatType = 2.0
    font["post"].extraNames = []
    font["post"].mapping = {}
    font["post"].glyphOrder = order
    font["head"].indexToLocFormat = 0
    buf = io.BytesIO()
    font.save(buf, reorderTables=True)
    return TTFont(io.BytesIO(buf.getvalue()), recalcTimestamp=False)  # recalculated maxp and bounds on save


def make_subset(src_path, gids, layout):
    font = _load(src_path)
    opts = subset.Options()
    opts.glyph_names = True
    opts.notdef_outline = True
    opts.name_IDs = ["*"]
    opts.name_languages = ["*"]
    opts.name_legacy = True
    opts.recalc_timestamp = False
    opts.hinting = False
    opts.drop_tables += ["DSIG"]
    if layout:
        opts.layout_features = ["*"]
        opts.layout_closure = True
    else:
        opts.layout_features = []
        opts.drop_tables += ["GSUB", "GPOS", "GDEF"]
    src_order = font.getGlyphOrder()
    sub = subset.Subsetter(opts)
    sub.populate(gids=gids)
    sub.subset(font)
    # The kept glyphs by the source's numbering (names a font without post names gets are made up afresh when the
    # subset is read back, so the mapping is taken here, before saving).
    kept = [src_order.index(n) for n in font.getGlyphOrder()]
    return _save(font), kept


def _font_sample(stem, data, ext, desc, info, lines, files_extra=None):
    """A written font `stem.ext`: the sample is named by its file (its JSON is `stem.ext.json`), so a font's WOFF and
    WOFF2 sit side by side."""
    name = f"{stem}.{ext}"
    font_dump = dump(data)
    files = {f"{name}.dump.json": dump_bytes(font_dump),
             f"{name}.shaping.json": dump_bytes({"lines": shape(data, lines)})}
    files.update(files_extra or {})
    exp = {"schema": 1, "name": name, "file": name, "kind": "font", "phases": D8, "description": desc,
           "size": len(data), "sha256": sha256(data), "hostile": None, "font": info,
           "dump": f"{name}.dump.json", "shaping": f"{name}.shaping.json"}
    return Sample(name, "d8", "font", D8, desc, data, exp, files, ext=ext, reference="font", filename=name)


def build(seed):
    out = []
    for base, rel, licence, lines, chars in SOURCES:
        path = os.path.join(FONTS, rel)
        with open(path, "rb") as fh:
            source = fh.read()
        src = _load(path)
        is_cff = "CFF " in src
        src_ext = "otf" if is_cff else "ttf"
        order = src.getGlyphOrder()
        cmap = src.getBestCmap()
        wanted = {order.index(cmap[ord(ch)]) for ch in chars if ord(ch) in cmap}
        gids = sorted(wanted)
        common = {"source": _rel(path), "licence": _rel(os.path.join(FONTS, licence)), "lines": lines}
        # The source itself: its dump and shaping, no font file.
        src_dump = dump(source)
        files = {f"{base}.source.dump.json": dump_bytes(src_dump),
                 f"{base}.source.shaping.json": dump_bytes({"lines": shape(source, lines)})}
        desc = f"The repository's {rel} as the reference font tools read it (the font itself is not copied)."
        exp = {"schema": 1, "name": f"{base}.source", "file": None, "kind": "font-source", "phases": D8,
               "description": desc, "size": len(source), "sha256": sha256(source), "hostile": None,
               "font": {**common, "format": src_dump["format"], "derivation": "source"},
               "dump": f"{base}.source.dump.json", "shaping": f"{base}.source.shaping.json"}
        out.append(Sample(f"{base}.source", "d8", "font-source", D8, desc, None, exp, files, reference=None))
        # WOFF and WOFF2 of the whole font.
        woff = _save(_load(path), "woff")
        out.append(_font_sample(base, woff, "woff", f"{rel} as WOFF 1.0 (each table zlib-compressed).",
                                {**common, "sourcePath": _rel(path), "format": "woff", "derivation": "whole",
                                 "outlines": "cff" if is_cff else "truetype"}, lines))
        transform = None if is_cff else ["glyf", "loca"]
        woff2_data = _save(_load(path), "woff2", transform=transform or [])
        out.append(_font_sample(base, woff2_data, "woff2",
                                f"{rel} as WOFF2 (Brotli" + (", the glyf and loca tables transformed" if transform
                                                             else ", no table transformed (CFF)") + ").",
                                {**common, "sourcePath": _rel(path), "format": "woff2", "derivation": "whole",
                                 "outlines": "cff" if is_cff else "truetype", "transforms": transform or []}, lines))
        if base == "dejavu-sans-subset":
            data = _save(_load(path), "woff2", transform=["glyf", "loca", "hmtx"])
            out.append(_font_sample(f"{base}.hmtx", data, "woff2",
                                    f"{rel} as WOFF2 with the glyf, loca and hmtx transforms.",
                                    {**common, "sourcePath": _rel(path), "format": "woff2", "derivation": "whole",
                                     "outlines": "truetype", "transforms": ["glyf", "loca", "hmtx"]}, lines))
        # Across outline kinds.
        if not is_cff and base in ("dejavu-sans-subset", "features"):
            data = to_cff(path)
            out.append(_font_sample(f"{base}.cff", data, "otf",
                                    f"{rel}'s TrueType outlines written as CFF (OpenType, Type 2 charstrings): each "
                                    "quadratic raised to a cubic exactly, points as fixed-point numbers.",
                                    {**common, "sourcePath": _rel(path), "format": "opentype-cff",
                                     "derivation": "converted", "outlines": "cff", "tolerance": 0.01}, lines))
        if is_cff:
            tt = to_truetype(path, 1.0)
            data = _save(tt)
            out.append(_font_sample(f"{base}.truetype", data, "ttf",
                                    f"{rel}'s CFF outlines written as TrueType: each cubic turned into quadratics "
                                    "within 1 font unit (fontTools' cu2qu), points rounded to whole units, contours "
                                    "turned clockwise as TrueType draws them, glyf and loca built.",
                                    {**common, "sourcePath": _rel(path), "format": "truetype",
                                     "derivation": "converted", "outlines": "truetype", "tolerance": 1.5,
                                     "toleranceWhy": "cu2qu's 1 unit and half a unit of rounding"}, lines))
        # Subsets of a fixed glyph list.
        for layout in (False, True):
            data, kept = make_subset(path, gids, layout)
            sub = TTFont(io.BytesIO(data))
            suffix = "subset-layout" if layout else "subset"
            what = ("the layout tables kept, with every glyph they can reach from these (closure), rewritten to the "
                    "kept glyphs" if layout else "the layout tables dropped (what a PDF embedding needs)")
            out.append(_font_sample(
                f"{base}.{suffix}", data, src_ext,
                f"{rel} cut to a fixed glyph list ({len(gids)} glyphs, the characters {chars!r}) and their composite "
                f"parts, {what}; glyphs renumbered, hinting dropped, names kept.",
                {**common, "sourcePath": _rel(path), "format": "opentype-cff" if is_cff else "truetype",
                 "derivation": "subset", "outlines": "cff" if is_cff else "truetype",
                 "glyphs": {"requested": gids, "requestedNames": [order[g] for g in gids], "characters": chars,
                            "kept": kept, "keptNames": sub.getGlyphOrder()},
                 "layout": "closure" if layout else "dropped"}, lines))
    return out
