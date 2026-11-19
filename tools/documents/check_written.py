"""Checks PDF files Idrak wrote (plan 28, D6 onwards) with the reference tools: qpdf (through pikepdf) opens each file,
checks its syntax (qpdf --check) and decodes every stream; pdfminer.six reads every page's text; pdfium counts and
draws the pages and reads their text. Every warning any of them gives is listed. With an expected JSON (the sample
generator's format, tools/documents/README.md), the page count, page sizes and each page's text are compared too; a
redacted file's JSON may add `removedText` (words no tool may find: not in pdfminer's or pdfium's text, nor in any
stream qpdf decodes or any string of any object, as text or hex; pdfminer's words are then compared in any order, as the
gaps cut lines keep make its layout order their pieces by column) and `annotationCounts` (the annotations pdfium lists
on each page).

    python -I tools/documents/check_written.py FILE.pdf [FILE.pdf | DIR ...] -o REPORT.json
    python -I tools/documents/check_written.py test-output/documents/written -o test-output/documents/written.json --expected auto
    python -I tools/documents/check_written.py OUT.pdf --expected OUT.expected.json --render test-output/documents/pages

A folder is searched for *.pdf (sorted). `--expected auto` takes NAME.json beside NAME.pdf when there is one. The exit
code is 0 when every file opens in all three tools with no warning, no problem and no difference, 1 otherwise.
"""
import argparse
import json
import logging
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.dont_write_bytecode = True
sys.path.insert(0, HERE)


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("inputs", nargs="+", help="PDF files or folders (searched for *.pdf)")
    ap.add_argument("-o", "--out", required=True, help="the report (JSON)")
    ap.add_argument("--expected", default=None,
                    help="an expected JSON for a single input, or 'auto' (NAME.json beside each NAME.pdf)")
    ap.add_argument("--password", default=None, help="the password to open the files with")
    ap.add_argument("--render", default=None, help="a folder to write pdfium's drawing of each page into (PNG)")
    ap.add_argument("--dpi", type=float, default=100, help="the DPI pages are drawn at (default 100)")
    return ap.parse_args()


def files_of(inputs):
    out = []
    for p in inputs:
        if os.path.isdir(p):
            for dirpath, _, names in os.walk(p):
                out += [os.path.join(dirpath, n) for n in names if n.lower().endswith(".pdf")]
        else:
            out.append(p)
    return sorted(out)


class Collector(logging.Handler):
    def __init__(self):
        super().__init__(logging.WARNING)
        self.messages = []

    def emit(self, record):
        self.messages.append(f"{record.levelname}: {record.getMessage()}"[:300])


# The filters qpdf does not decode (an image codec's data, read_bytes refuses it): such a stream is decoded as an image.
IMAGE_CODECS = {"/DCTDecode", "/DCT", "/JPXDecode"}


def _strings(obj, out, depth=0):
    """Every string of a pikepdf object, depth first (references are objects of their own: not followed; numbers and
    booleans come back as Python's own values, with nothing in them)."""
    import pikepdf

    if depth > 32:
        return
    if isinstance(obj, pikepdf.String):
        out.append(bytes(obj))
        return
    if isinstance(obj, pikepdf.Array):
        items = list(obj)
    elif isinstance(obj, (pikepdf.Dictionary, pikepdf.Stream)):
        items = [obj.get(key) for key in obj.keys()]
    else:
        return
    for item in items:
        if isinstance(item, pikepdf.Object) and not item.is_indirect:
            _strings(item, out, depth + 1)


def qpdf(path, password, removed=()):
    import pikepdf

    out = {"opened": False}
    try:
        with pikepdf.open(path, password=password or "") as pdf:
            out.update({"opened": True, "pages": len(pdf.pages), "version": pdf.pdf_version,
                        "encrypted": bool(pdf.is_encrypted)})
            out["problems"] = [str(p) for p in pdf.check_pdf_syntax()]
            errors, streams, found = [], 0, []
            needles = [(w, [w.encode("latin-1"), w.encode("latin-1").hex().encode("ascii"),
                            w.encode("latin-1").hex().upper().encode("ascii")]) for w in removed]
            for obj in pdf.objects:
                data = []
                if isinstance(obj, pikepdf.Stream):
                    streams += 1
                    try:
                        filters = obj.get("/Filter")
                        names = [] if filters is None else [str(f) for f in filters] if isinstance(filters, pikepdf.Array) else [str(filters)]
                        if any(n in IMAGE_CODECS for n in names):
                            # qpdf hands an image codec's data over encoded: decoded as the image it is (Pillow).
                            pikepdf.PdfImage(obj).as_pil_image().load()
                        else:
                            data.append(obj.read_bytes())
                    except Exception as e:  # noqa: BLE001 - recorded
                        errors.append(f"{obj.objgen}: {type(e).__name__}: {e}"[:300])
                if needles and isinstance(obj, (pikepdf.Dictionary, pikepdf.Stream, pikepdf.Array, pikepdf.String)):
                    _strings(obj, data)
                    for word, forms in needles:
                        if any(form in chunk for chunk in data for form in forms):
                            found.append(f"'{word}' in {obj.objgen}")
            out["streams"] = streams
            out["streamErrors"] = errors
            out["removedFound"] = found
            out["warnings"] = [str(w) for w in pdf.get_warnings()]
    except Exception as e:  # noqa: BLE001 - recorded
        out["error"] = f"{type(e).__name__}: {e}"[:300]
    return out


def pdfminer(path, password):
    from pdfminer.high_level import extract_text

    log = logging.getLogger("pdfminer")
    collector = Collector()
    log.addHandler(collector)
    log.setLevel(logging.WARNING)
    try:
        text = extract_text(path, password=password or "")
        pages = text.split("\x0c")[:-1]
        return {"opened": True, "pages": pages, "warnings": collector.messages}
    except Exception as e:  # noqa: BLE001 - recorded
        return {"opened": False, "error": f"{type(e).__name__}: {e}"[:300], "warnings": collector.messages}
    finally:
        log.removeHandler(collector)


def pdfium(path, password, render, dpi):
    import pypdfium2 as pdfium

    out = {"opened": False}
    try:
        doc = pdfium.PdfDocument(path, password=password or None)
        try:
            pages = []
            for i in range(len(doc)):
                page = doc[i]
                w, h = page.get_size()
                tp = page.get_textpage()
                import pypdfium2.raw as raw

                entry = {"width": round(w, 4), "height": round(h, 4), "text": tp.get_text_range(),
                         "annotations": raw.FPDFPage_GetAnnotCount(page)}
                tp.close()
                if render:
                    os.makedirs(render, exist_ok=True)
                    name = os.path.splitext(os.path.basename(path))[0] + f".p{i + 1}.png"
                    page.render(scale=dpi / 72).to_pil().convert("RGB").save(os.path.join(render, name))
                    entry["drawn"] = name
                pages.append(entry)
                page.close()
            out = {"opened": True, "pageCount": len(doc), "pages": pages}
        finally:
            doc.close()
    except Exception as e:  # noqa: BLE001 - recorded
        out["error"] = f"{type(e).__name__}: {e}"[:300]
    return out


def norm(t):
    return " ".join((t or "").split())


def compare(exp, q, m, p):
    diffs = []
    want_pages = exp.get("open", {}).get("pageCount") if exp.get("open") else None
    pages = exp.get("pages") or []
    if want_pages is None:
        want_pages = len(pages)
    for tool, got in (("qpdf", q.get("pages")), ("pdfium", p.get("pageCount"))):
        if got != want_pages:
            diffs.append({"pageCount": want_pages, tool: got})
    for k, page in enumerate(pages):
        if p.get("opened") and k < len(p["pages"]):
            g = p["pages"][k]
            if abs(g["width"] - page["width"]) > 0.01 or abs(g["height"] - page["height"]) > 0.01:
                diffs.append({"page": k, "size": [page["width"], page["height"]], "pdfium": [g["width"], g["height"]]})
            if norm(g["text"]) != norm(page.get("text")):
                diffs.append({"page": k, "text": page.get("text"), "pdfium": g["text"]})
        if m.get("opened") and k < len(m["pages"]):
            # A redacted file's lines keep gaps where words were cut, which pdfminer.six's layout splits into boxes it
            # orders by column: its words are compared in any order there.
            got, want = norm(m["pages"][k]), norm(page.get("text"))
            if exp.get("removedText"):
                got, want = sorted(got.split()), sorted(want.split())
            if got != want:
                diffs.append({"page": k, "text": page.get("text"), "pdfminer": m["pages"][k]})
    # A redacted file: no tool finds a removed word; pdfium lists the annotations each page keeps.
    for word in exp.get("removedText") or []:
        for k, text in enumerate(m.get("pages") or []):
            if word in text:
                diffs.append({"page": k, "removed": word, "pdfminer": "finds it"})
        for k, page in enumerate(p.get("pages") or []):
            if word in page["text"]:
                diffs.append({"page": k, "removed": word, "pdfium": "finds it"})
    for found in q.get("removedFound") or []:
        diffs.append({"removed": found, "qpdf": "finds it"})
    for k, want in enumerate(exp.get("annotationCounts") or []):
        if p.get("opened") and k < len(p["pages"]) and p["pages"][k]["annotations"] != want:
            diffs.append({"page": k, "annotations": want, "pdfium": p["pages"][k]["annotations"]})
    return diffs


def main():
    args = parse_args()
    paths = files_of(args.inputs)
    if not paths:
        sys.exit("no PDF file given or found")
    if args.expected not in (None, "auto") and len(paths) != 1:
        sys.exit("an expected JSON file goes with a single input; use --expected auto for several")
    start = time.monotonic()
    results = []
    for i, path in enumerate(paths):
        exp_path = None
        if args.expected == "auto":
            cand = os.path.splitext(path)[0] + ".json"
            exp_path = cand if os.path.isfile(cand) else None
        elif args.expected:
            exp_path = args.expected
        exp = None
        if exp_path:
            with open(exp_path, encoding="utf-8") as fh:
                exp = json.load(fh)
        q = qpdf(path, args.password, (exp or {}).get("removedText") or ())
        m = pdfminer(path, args.password)
        p = pdfium(path, args.password, args.render, args.dpi)
        diffs = compare(exp, q, m, p) if exp is not None else []
        problems = []
        if not (q.get("opened") and m.get("opened") and p.get("opened")):
            problems.append("not opened by every tool")
        if q.get("error"):
            problems.append(f"qpdf error: {q['error']}")
        for key in ("problems", "streamErrors", "warnings"):
            if q.get(key):
                problems.append(f"qpdf {key}: {len(q[key])}")
        if m.get("warnings"):
            problems.append(f"pdfminer warnings: {len(m['warnings'])}")
        if diffs:
            problems.append(f"differences from the expected JSON: {len(diffs)}")
        ok = not problems
        results.append({"file": path, "ok": ok, "problems": problems, "expected": exp_path, "differences": diffs,
                        "qpdf": q, "pdfminer": m, "pdfium": p})
        elapsed = time.monotonic() - start
        rate = (i + 1) / elapsed if elapsed > 0 else 0
        print(f"[{i + 1}/{len(paths)}] {'ok  ' if ok else 'FAIL'} {path} ({elapsed:.1f}s, {rate:.1f} files/s, "
              f"~{(len(paths) - i - 1) / rate if rate else 0:.0f}s left)" + ("" if ok else f": {'; '.join(problems)}"),
              flush=True)
    failed = [r["file"] for r in results if not r["ok"]]
    report = {"tool": "tools/documents/check_written.py", "files": results,
              "summary": {"files": len(results), "ok": len(results) - len(failed), "failed": failed}}
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=1)
        fh.write("\n")
    print(f"{len(results) - len(failed)} of {len(results)} files passed; report written to {args.out}", flush=True)
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())

