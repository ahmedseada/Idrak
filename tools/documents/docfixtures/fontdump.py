"""Fonts as the reference font tools read them (D8): a JSON dump (tables, metrics, the character map, advances, every
glyph's outline, names, the layout tables' lookups with the glyphs each reaches), HarfBuzz's shaping of sample lines,
the OpenType Sanitizer's verdict, and a comparison of a font with the one it was made from (outlines, advances,
characters and shaping kept). Used by the D8 samples and by tools/documents/check_font.py.

Outlines: a TrueType glyph's `contours` are its points as stored ([x, y, on-curve 1 or 0], composites resolved, as
fontTools' getCoordinates gives them) with its `components` (glyph id and the 2x2 transform and offset); a CFF glyph's
`outline` is its Type 2 charstring drawn (moveTo, lineTo, curveTo, closePath with their points, font units)."""
import io
import json
import math
import os
import subprocess
import tempfile

import uharfbuzz as hb
from fontTools.pens.boundsPen import BoundsPen
from fontTools.pens.recordingPen import DecomposingRecordingPen
from fontTools.pens.statisticsPen import StatisticsPen
from fontTools.ttLib import TTFont

from .pdf import sha256


def _num(v):
    if isinstance(v, float):
        if v == int(v):
            return int(v)
        return round(v, 4)
    return v


def open_font(data):
    return TTFont(io.BytesIO(data), recalcTimestamp=False, recalcBBoxes=False, lazy=False)


def sfnt_bytes(data):
    """The font as a plain sfnt (WOFF and WOFF2 unpacked), for HarfBuzz."""
    font = open_font(data)
    if font.flavor is None:
        return data
    font.flavor = None
    buf = io.BytesIO()
    font.save(buf, reorderTables=False)
    return buf.getvalue()


def font_format(font):
    if font.flavor in ("woff", "woff2"):
        return font.flavor
    return "opentype-cff" if "CFF " in font else "truetype"


def _lookup_glyphs(obj, names, seen=None):
    """Every glyph name a lookup's subtables refer to (coverage, classes, substitutes, rules), as a set."""
    if seen is None:
        seen = set()
    out = set()
    if id(obj) in seen:
        return out
    seen.add(id(obj))
    if isinstance(obj, str):
        if obj in names:
            out.add(obj)
        return out
    if isinstance(obj, dict):
        for k, v in obj.items():
            out |= _lookup_glyphs(k, names, seen) | _lookup_glyphs(v, names, seen)
        return out
    if isinstance(obj, (list, tuple, set)):
        for v in obj:
            out |= _lookup_glyphs(v, names, seen)
        return out
    if hasattr(obj, "__dict__"):
        for k, v in vars(obj).items():
            if k in ("reader", "font"):
                continue
            out |= _lookup_glyphs(v, names, seen)
    return out


def _layout(font, tag, ids):
    if tag not in font:
        return None
    table = font[tag].table
    names = set(ids)
    lookups = []
    for i, lk in enumerate(table.LookupList.Lookup if table.LookupList else []):
        kind = lk.LookupType
        subtables = lk.SubTable
        if kind in (7, 9):  # extension: the real type inside
            kind = subtables[0].ExtensionLookupType if subtables else kind
        glyphs = _lookup_glyphs(subtables, names)
        lookups.append({"index": i, "type": kind, "flag": lk.LookupFlag,
                        "markFilteringSet": getattr(lk, "MarkFilteringSet", None),
                        "subtables": len(subtables), "glyphs": sorted(ids[g] for g in glyphs)})
    features = []
    for rec in table.FeatureList.FeatureRecord if table.FeatureList else []:
        features.append({"tag": rec.FeatureTag, "lookups": list(rec.Feature.LookupListIndex)})
    scripts = []
    for rec in table.ScriptList.ScriptRecord if table.ScriptList else []:
        langs = [("dflt", rec.Script.DefaultLangSys)] + [(l.LangSysTag, l.LangSys) for l in rec.Script.LangSysRecord]
        scripts.append({"tag": rec.ScriptTag, "languages": [
            {"tag": t, "features": list(ls.FeatureIndex) if ls else [],
             "required": ls.ReqFeatureIndex if ls and ls.ReqFeatureIndex != 0xFFFF else None} for t, ls in langs]})
    return {"scripts": scripts, "features": features, "lookups": lookups}


def dump(data):
    """The font's JSON dump (a dict)."""
    font = open_font(data)
    order = font.getGlyphOrder()
    ids = {n: i for i, n in enumerate(order)}
    out = {"format": font_format(font), "sha256": sha256(data), "size": len(data),
           "sfntVersion": "OTTO" if font.sfntVersion == "OTTO" else "0x00010000", "flavor": font.flavor,
           "tables": [{"tag": t, "length": len(font.getTableData(t))} for t in sorted(font.keys()) if t != "GlyphOrder"],
           "unitsPerEm": font["head"].unitsPerEm, "glyphCount": len(order), "glyphOrder": order}
    head = font["head"]
    out["head"] = {k: getattr(head, k) for k in ("xMin", "yMin", "xMax", "yMax", "macStyle", "flags",
                                                  "indexToLocFormat", "fontRevision")}
    out["head"]["fontRevision"] = _num(out["head"]["fontRevision"])
    hhea = font["hhea"]
    out["hhea"] = {k: getattr(hhea, k) for k in ("ascent", "descent", "lineGap", "advanceWidthMax", "numberOfHMetrics")}
    if "OS/2" in font:
        os2 = font["OS/2"]
        out["os2"] = {k: getattr(os2, k, None) for k in ("version", "usWeightClass", "usWidthClass", "fsType",
                                                         "sTypoAscender", "sTypoDescender", "sTypoLineGap",
                                                         "usWinAscent", "usWinDescent", "sxHeight", "sCapHeight",
                                                         "fsSelection", "usFirstCharIndex", "usLastCharIndex")}
    if "post" in font:
        out["post"] = {"formatType": _num(font["post"].formatType), "italicAngle": _num(font["post"].italicAngle),
                       "underlinePosition": font["post"].underlinePosition}
    cmap = font["cmap"]
    out["cmap"] = {"subtables": [[t.platformID, t.platEncID, t.format] for t in cmap.tables],
                   "best": {str(cp): ids[n] for cp, n in sorted(font.getBestCmap().items())}}
    hmtx = font["hmtx"]
    out["hmtx"] = [list(hmtx[n]) for n in order]
    out["names"] = [[r.nameID, r.platformID, r.platEncID, r.langID, r.toUnicode()] for r in font["name"].names]
    glyphs = []
    if "glyf" in font:
        glyf = font["glyf"]
        for n in order:
            g = glyf[n]
            entry = {"name": n}
            if g.numberOfContours == 0:
                entry["kind"] = "empty"
            else:
                coords, ends, flags = g.getCoordinates(glyf)
                contours, start = [], 0
                for e in ends:
                    contours.append([[_num(coords[k][0]), _num(coords[k][1]), flags[k] & 1] for k in range(start, e + 1)])
                    start = e + 1
                entry["kind"] = "composite" if g.isComposite() else "simple"
                entry["contours"] = contours
                if g.isComposite():
                    comps = []
                    for comp in g.components:
                        t = getattr(comp, "transform", [[1, 0], [0, 1]])
                        comps.append({"glyph": ids[comp.glyphName], "transform": [_num(t[0][0]), _num(t[0][1]),
                                                                                  _num(t[1][0]), _num(t[1][1]),
                                                                                  comp.x, comp.y]})
                    entry["components"] = comps
            entry["bounds"] = [getattr(g, k, 0) for k in ("xMin", "yMin", "xMax", "yMax")] if g.numberOfContours else None
            glyphs.append(entry)
    else:
        gs = font.getGlyphSet()
        for n in order:
            pen = DecomposingRecordingPen(gs)
            gs[n].draw(pen)
            glyphs.append({"name": n, "kind": "cff", "outline": [[op] + [[_num(x), _num(y)] for x, y in pts]
                                                                  for op, pts in pen.value]})
    out["glyphs"] = glyphs
    out["layout"] = {tag: _layout(font, tag, ids) for tag in ("GSUB", "GPOS")}
    if "GDEF" in font and font["GDEF"].table.GlyphClassDef:
        out["layout"]["GDEF"] = {"glyphClasses": {str(ids[n]): c for n, c in
                                                  sorted(font["GDEF"].table.GlyphClassDef.classDefs.items(),
                                                         key=lambda kv: ids[kv[0]])}}
    return out


def dump_bytes(obj):
    """JSON with the top level keys one a line and each glyph one a line (dumps run to megabytes)."""
    lines = ["{"]
    keys = list(obj)
    for i, k in enumerate(keys):
        v = obj[k]
        tail = "," if i + 1 < len(keys) else ""
        if k == "glyphs":
            lines.append(json.dumps(k) + ": [")
            for j, g in enumerate(v):
                lines.append(" " + json.dumps(g, ensure_ascii=False, separators=(",", ":"))
                             + ("," if j + 1 < len(v) else ""))
            lines.append("]" + tail)
        else:
            lines.append(json.dumps(k) + ": " + json.dumps(v, ensure_ascii=False, separators=(",", ":")) + tail)
    lines.append("}")
    return ("\n".join(lines) + "\n").encode("utf-8")


def shape(data, lines):
    """HarfBuzz's shaping of each line (default features, direction and script guessed from the text)."""
    sfnt = sfnt_bytes(data)
    font = hb.Font(hb.Face(hb.Blob(sfnt)))
    order = open_font(sfnt).getGlyphOrder()
    out = []
    for text in lines:
        buf = hb.Buffer()
        buf.add_str(text)
        buf.guess_segment_properties()
        hb.shape(font, buf, {})
        out.append({"text": text, "direction": buf.direction, "script": buf.script,
                    "glyphs": [{"gid": i.codepoint, "name": order[i.codepoint], "cluster": i.cluster,
                                "xAdvance": p.x_advance, "yAdvance": p.y_advance, "xOffset": p.x_offset,
                                "yOffset": p.y_offset} for i, p in zip(buf.glyph_infos, buf.glyph_positions)]})
    return out


def sanitize(path):
    """The OpenType Sanitizer (the `ots` package's ots-sanitize, the checker browsers use) on a font file."""
    import ots

    with tempfile.TemporaryDirectory() as tmp:
        out = os.path.join(tmp, "sanitized")
        try:
            proc = ots.sanitize(path, out, capture_output=True)
            code, err = proc.returncode, proc.stderr
        except subprocess.CalledProcessError as e:  # older wrappers raise on failure
            code, err = e.returncode, e.stderr
    msg = (err or b"").decode("utf-8", "replace").replace(os.path.dirname(os.path.abspath(path)) + os.sep, "")
    return {"passed": code == 0, "returnCode": code, "messages": [l for l in msg.splitlines() if l.strip()]}


def opens(data):
    """fontTools opens the font and decompiles every table."""
    try:
        font = open_font(data)
        for tag in font.keys():
            font[tag]
            if tag == "glyf":
                for n in font.getGlyphOrder():
                    font["glyf"][n].expand(font["glyf"])
        return {"opened": True, "tables": sorted(t for t in font.keys() if t != "GlyphOrder")}
    except Exception as e:  # noqa: BLE001 - recorded
        return {"opened": False, "error": f"{type(e).__name__}: {e}"}


def _outlines(font):
    """Each glyph's outline as recorded (composites decomposed; an all-off-curve contour ends with None)."""
    gs = font.getGlyphSet()
    out = []
    for n in font.getGlyphOrder():
        pen = DecomposingRecordingPen(gs)
        gs[n].draw(pen)
        out.append([[op] + [tuple(p) if p is not None else None for p in pts] for op, pts in pen.value])
    return out


def _shape_stats(font):
    """Each glyph's exact bounds, area (its size: TrueType and CFF contours turn opposite ways) and centroid (font
    units): invariant to how a contour is drawn (lines or flat curves, quadratics or cubics, where it starts), so
    outlines of different kinds compare through them."""
    gs = font.getGlyphSet()
    out = []
    for n in font.getGlyphOrder():
        rec = DecomposingRecordingPen(gs)
        gs[n].draw(rec)
        # A contour of one point draws nothing (CFF cannot even hold one): left out.
        value, contour = [], []
        for op, pts in rec.value:
            contour.append((op, pts))
            if op in ("closePath", "endPath"):
                if any(o not in ("moveTo", "closePath", "endPath") for o, _ in contour):
                    value += contour
                contour = []
        bp = BoundsPen(gs)
        sp = StatisticsPen(gs)
        for op, pts in value:
            getattr(bp, op)(*pts)
            getattr(sp, op)(*pts)
        out.append({"bounds": bp.bounds, "area": abs(sp.area), "centre": (sp.meanX, sp.meanY) if sp.area else (0, 0)})
    return out


def compare(data, source, lines, tolerance=None, mapping=None):
    """Whether `data` keeps `source`'s outlines (exact for the same outline kind; for another kind, every sampled
    point within `tolerance` font units), advances, characters and the shaping of `lines`. `mapping` gives each glyph's
    id in the source (a subset's kept glyphs); without it the ids are the same."""
    f, s = open_font(data), open_font(source)
    same_kind = ("CFF " in f) == ("CFF " in s)
    tol = 0.0 if tolerance is None and same_kind else (1.0 if tolerance is None else tolerance)
    fo, so = (_outlines(f), _outlines(s)) if same_kind else (_shape_stats(f), _shape_stats(s))
    mapping = mapping or list(range(len(fo)))
    if len(mapping) != len(fo):
        return {"kept": False, "problem": f"{len(fo)} glyphs, {len(mapping)} mapped"}
    worst = 0.0
    outline_diffs = []
    for gid, ol in enumerate(fo):
        src = so[mapping[gid]]
        if same_kind:
            if ol != src:
                outline_diffs.append({"glyph": gid, "source": mapping[gid], "problem": "outline differs"})
            continue
        if (ol["bounds"] is None) != (src["bounds"] is None):
            outline_diffs.append({"glyph": gid, "source": mapping[gid], "problem": "empty in one font only"})
            continue
        if ol["bounds"] is None:
            continue
        d = max(max(abs(a - b) for a, b in zip(ol["bounds"], src["bounds"])),
                math.hypot(ol["centre"][0] - src["centre"][0], ol["centre"][1] - src["centre"][1]))
        area = abs(ol["area"] - src["area"]) / max(abs(src["area"]), 1.0)
        worst = max(worst, d)
        if d > tol or area > 0.005:
            outline_diffs.append({"glyph": gid, "source": mapping[gid], "boundsOrCentre": round(d, 3),
                                  "areaRatio": round(area, 5)})
    fh, sh = f["hmtx"], s["hmtx"]
    fn, sn = f.getGlyphOrder(), s.getGlyphOrder()
    advance_diffs = [gid for gid in range(len(fn)) if fh[fn[gid]][0] != sh[sn[mapping[gid]]][0]]
    fc, sc = f.getBestCmap(), s.getBestCmap()
    fid, sid = {n: i for i, n in enumerate(fn)}, {n: i for i, n in enumerate(sn)}
    cmap_diffs = [cp for cp, n in fc.items() if cp not in sc or mapping[fid[n]] != sid[sc[cp]]]
    shaping_diffs = []
    skipped = []
    if lines and len(fn) != len(sn):
        # A subset shapes only the lines whose characters it holds.
        skipped = [t for t in lines if any(ord(ch) not in fc for ch in t if not ch.isspace())]
        lines = [t for t in lines if t not in skipped]
    if lines:
        a, b = shape(data, lines), shape(source, lines)
        for la, lb in zip(a, b):
            ga = [(mapping[g["gid"]], g["cluster"], g["xAdvance"], g["yAdvance"], g["xOffset"], g["yOffset"])
                  for g in la["glyphs"]]
            gb = [(g["gid"], g["cluster"], g["xAdvance"], g["yAdvance"], g["xOffset"], g["yOffset"]) for g in lb["glyphs"]]
            if ga != gb:
                shaping_diffs.append(la["text"])
    return {"sameOutlineKind": same_kind, "tolerance": tol, "glyphsCompared": len(fo),
            "comparedBy": "outline commands, exactly" if same_kind else "exact bounds, area and centroid",
            "largestOutlineDistance": round(worst, 4), "outlineDifferences": outline_diffs[:20],
            "outlineDifferenceCount": len(outline_diffs), "advanceDifferences": advance_diffs[:20],
            "cmapDifferences": cmap_diffs[:20], "charactersKept": len(fc), "shapingDifferences": shaping_diffs,
            "linesShaped": len(lines or []), "linesSkipped": skipped,
            "kept": not outline_diffs and not advance_diffs and not cmap_diffs and not shaping_diffs}


def font_reference(path, sample, notes):
    """The generator's reference run on a written font sample: fontTools opens it, the Sanitizer checks it, and when
    it was made from a whole font (a conversion), its outlines, advances, characters and shaping against the source;
    a subset is compared on the glyphs it keeps."""
    with open(path, "rb") as fh:
        data = fh.read()
    exp = sample.expected
    result = {"fontTools": opens(data), "ots": sanitize(path)}
    src = exp["font"].get("sourcePath")
    if src:
        from .common import ROOT

        with open(os.path.join(ROOT, src), "rb") as fh:
            source = fh.read()
        subset_ = exp["font"]["derivation"] == "subset"
        lines = exp["font"].get("lines") if not subset_ else None
        mapping = exp["font"]["glyphs"]["kept"] if subset_ else None
        result["kept"] = compare(data, source, lines, exp["font"].get("tolerance"), mapping)
        if not result["kept"]["kept"] and exp["font"]["derivation"] in ("whole", "converted"):
            notes.append(f"{sample.folder}/{sample.name}: not every outline, advance, character or shaping kept")
    if not result["fontTools"]["opened"]:
        notes.append(f"{sample.folder}/{sample.name}: fontTools could not open it")
    if not result["ots"]["passed"]:
        notes.append(f"{sample.folder}/{sample.name}: the OpenType Sanitizer refused it: {result['ots']['messages'][:2]}")
    return result, {}
