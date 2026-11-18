"""Checks font files Idrak wrote (plan 28, D8) with the reference font tools: fontTools opens each and decompiles every
table, the OpenType Sanitizer (the `ots` package's ots-sanitize, the checker browsers use) accepts it or says why not,
and HarfBuzz shapes sample lines with it. With --source, the font is compared with the one it was made from: every
outline (exact for the same outline kind; across TrueType and CFF, each glyph's exact bounds, area and centroid within
--tolerance font units), every advance, every character and the shaping of the lines kept.

    python -I tools/documents/check_font.py FONT [FONT ...] -o REPORT.json
    python -I tools/documents/check_font.py out.ttf --source tests/Idrak.Tests/data/fonts/dejavu-sans-cff.otf --tolerance 1.5 -o r.json
    python -I tools/documents/check_font.py sub.ttf --source src.ttf --mapping sub.map.json -o r.json

A subset is compared on the glyphs it keeps: --mapping gives, for each of its glyph ids, the source's glyph id (a JSON
list, or a D8 sample's JSON, whose font.glyphs.kept is that list); without it, fonts of the same glyph count are compared id by id and others by glyph name (the shaping is then
not compared); a subset's shaping is compared on the lines whose characters it holds. The exit code is 0 when every font passes, 1 otherwise. Reads .ttf, .otf, .woff and .woff2.
"""
import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.dont_write_bytecode = True
sys.path.insert(0, HERE)

LINES = ["Quarterly report: the office moves in May.", "fi fl ffi office café 2024", "AVATAR To Ta Wa",
         "Report 2024", "تقرير المبيعات الشهري", "اجتماع الفريق يوم الثلاثاء", "سلام"]


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("fonts", nargs="+", help="font files (.ttf .otf .woff .woff2)")
    ap.add_argument("-o", "--out", required=True, help="the report (JSON)")
    ap.add_argument("--source", default=None, help="the font each was made from, to compare with")
    ap.add_argument("--tolerance", type=float, default=None,
                    help="font units allowed across outline kinds (default 1; the same kind must be exact)")
    ap.add_argument("--mapping", default=None, help="a JSON list: each glyph id's id in the source (a subset)")
    ap.add_argument("--line", action="append", default=None, help="a line to shape (repeat; default: generic lines)")
    ap.add_argument("--dump", default=None, help="a folder to write each font's JSON dump into")
    return ap.parse_args()


def main():
    args = parse_args()
    from docfixtures.fontdump import compare, dump, dump_bytes, opens, open_font, sanitize, shape

    lines = args.line or LINES
    source = None
    if args.source:
        with open(args.source, "rb") as fh:
            source = fh.read()
    mapping = None
    if args.mapping:
        with open(args.mapping, encoding="utf-8") as fh:
            mapping = json.load(fh)
        if isinstance(mapping, dict):  # a D8 sample's JSON: its kept glyphs
            mapping = mapping["font"]["glyphs"]["kept"]
    start = time.monotonic()
    results = []
    for i, path in enumerate(args.fonts):
        with open(path, "rb") as fh:
            data = fh.read()
        entry = {"file": path, "fontTools": opens(data), "ots": sanitize(path)}
        problems = []
        if not entry["fontTools"]["opened"]:
            problems.append("fontTools could not open it")
        else:
            entry["shaping"] = shape(data, lines)
            if source is not None:
                m = mapping
                compare_lines = lines
                if m is None:
                    f, s = open_font(data), open_font(source)
                    if len(f.getGlyphOrder()) != len(s.getGlyphOrder()):
                        names = {n: k for k, n in enumerate(s.getGlyphOrder())}
                        m = [names.get(n, 0) for n in f.getGlyphOrder()]
                        compare_lines = None
                        entry["mappedBy"] = "glyph name"
                        if "post" in s and s["post"].formatType == 3.0:
                            problems.append("the source stores no glyph names (post 3): give --mapping")
                entry["kept"] = compare(data, source, compare_lines, args.tolerance, m)
                if not entry["kept"]["kept"]:
                    problems.append("not every outline, advance, character or shaping kept")
            if args.dump:
                os.makedirs(args.dump, exist_ok=True)
                name = os.path.basename(path) + ".dump.json"
                with open(os.path.join(args.dump, name), "wb") as fh:
                    fh.write(dump_bytes(dump(data)))
                entry["dump"] = name
        if not entry["ots"]["passed"]:
            problems.append("the OpenType Sanitizer refused it")
        entry["ok"] = not problems
        entry["problems"] = problems
        results.append(entry)
        elapsed = time.monotonic() - start
        print(f"[{i + 1}/{len(args.fonts)}] {'ok  ' if not problems else 'FAIL'} {path} ({elapsed:.1f}s)"
              + ("" if not problems else f": {'; '.join(problems)}"), flush=True)
    failed = [r["file"] for r in results if not r["ok"]]
    report = {"tool": "tools/documents/check_font.py", "source": args.source, "lines": lines, "fonts": results,
              "summary": {"fonts": len(results), "ok": len(results) - len(failed), "failed": failed}}
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=1)
        fh.write("\n")
    print(f"{len(results) - len(failed)} of {len(results)} fonts passed; report written to {args.out}", flush=True)
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
