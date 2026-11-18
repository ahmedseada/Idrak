"""The reference tools run on every sample:
- qpdf through pikepdf: opens the file (with its password), reports its problems (qpdf --check), and decodes every
  stream the JSON lists, whose bytes must hash as the generator wrote them (filters, predictors, encryption checked);
- pdfminer.six: the reference reader's text, page by page;
- pdfium through pypdfium2: its text and each character's loose box (the generator's run boxes are checked against
  them), and the reference renderer's pages as PNG at a fixed DPI.
Hostile samples run each tool in a child process with a time limit (and a memory limit on Linux), and the outcome is
recorded: the error, a crash, or a timeout."""
import json
import os
import subprocess
import sys

DPI = 100
TIMEOUT = 30
MEMORY_LIMIT = 1 << 30
PASSED_THROUGH = {"DCTDecode", "JPXDecode", "CCITTFaxDecode", "JBIG2Decode"}  # image encodings a reader hands over


def _quiet_logs():
    import logging

    logging.getLogger("pdfminer").setLevel(logging.ERROR)


def pikepdf_check(path, password, streams):
    import pikepdf

    out = {"opened": False}
    try:
        with pikepdf.open(path, password=password or "") as pdf:
            out["opened"] = True
            out["pages"] = len(pdf.pages)
            out["encrypted"] = bool(pdf.is_encrypted)
            problems = [str(p) for p in pdf.check_pdf_syntax()] if hasattr(pdf, "check_pdf_syntax") else \
                [str(p) for p in pdf.check()]
            out["problems"] = [_strip_path(p, path) for p in problems]
            mismatches = []
            checked = 0
            for rec in streams:
                num, gen = rec["object"]
                if rec.get("type") in ("XRef", "ObjStm"):
                    continue
                try:
                    obj = pdf.get_object((num, gen))
                    if rec["filters"] and set(rec["filters"]) <= PASSED_THROUGH:
                        data = obj.read_raw_bytes()
                    else:
                        data = obj.read_bytes(pikepdf.StreamDecodeLevel.generalized)
                except Exception as e:  # noqa: BLE001 - recorded
                    mismatches.append({"object": rec["object"], "error": _msg(e, path)})
                    continue
                checked += 1
                import hashlib

                if hashlib.sha256(data).hexdigest() != rec["decodedSha256"]:
                    mismatches.append({"object": rec["object"], "decodedLength": len(data),
                                       "expectedLength": rec["decodedLength"]})
            out["streamsChecked"] = checked
            out["streamMismatches"] = mismatches
    except Exception as e:  # noqa: BLE001 - recorded
        out["error"] = _msg(e, path)
    return out


def pdfminer_text(path, password):
    _quiet_logs()
    from pdfminer.high_level import extract_pages
    from pdfminer.layout import LTTextContainer

    pages = []
    for layout in extract_pages(path, password=password or ""):
        parts = [el.get_text() for el in layout if isinstance(el, LTTextContainer)]
        pages.append("".join(parts))
    return pages


def pdfium_run(path, password, dpi=DPI, forms=False):
    """Text, character boxes and renders of every page pdfium loads; the pages it cannot load are listed. `forms`
    starts pdfium's form environment first, so form fields are drawn as a viewer draws them (appearances made for
    fields without one when the form asks with /NeedAppearances)."""
    import pypdfium2 as pdfium
    import pypdfium2.raw as raw

    doc = pdfium.PdfDocument(path, password=password or None)
    if forms:
        doc.init_forms()
    pages, errors = [], []
    try:
        count = len(doc)
        for i in range(count):
            try:
                page = doc[i]
            except Exception as e:  # noqa: BLE001 - recorded
                errors.append({"page": i, "error": _msg(e, path)})
                continue
            tp = page.get_textpage()
            chars = []
            for k in range(tp.count_chars()):
                rect = raw.FS_RECTF()
                ok = raw.FPDFText_GetLooseCharBox(tp, k, rect)
                chars.append((raw.FPDFText_GetUnicode(tp, k), raw.FPDFText_IsGenerated(tp, k) == 1,
                              (rect.left, rect.bottom, rect.right, rect.top) if ok else None,
                              raw.FPDFText_GetFontSize(tp, k)))
            text = tp.get_text_range()
            img = page.render(scale=dpi / 72).to_pil().convert("RGB")
            pages.append({"index": i, "text": text, "chars": chars, "image": img})
            tp.close()
            page.close()
    finally:
        doc.close()
    return {"pageCount": count, "pages": pages, "errors": errors}


def light_check(path, exp, notes, label):
    """The large set's reference check (no renders kept, nothing slow): qpdf opens the file and counts its pages,
    pdfium counts them and draws the first page at a low DPI, pdfminer.six reads the text of the pages the JSON lists
    (`checks.textPages`), which must equal the expected text."""
    import pikepdf
    import pypdfium2 as pdfium

    out = {"qpdf": {"opened": False}, "pdfium": {"opened": False}, "pdfminer": {"opened": False}}
    try:
        with pikepdf.open(path) as pdf:
            out["qpdf"] = {"opened": True, "pages": len(pdf.pages)}
    except Exception as e:  # noqa: BLE001 - recorded
        out["qpdf"]["error"] = _msg(e, path)
    try:
        doc = pdfium.PdfDocument(path)
        try:
            n = len(doc)
            page = doc[0]
            bitmap = page.render(scale=24 / 72)
            out["pdfium"] = {"opened": True, "pageCount": n, "firstPageDrawn": [bitmap.width, bitmap.height]}
            page.close()
        finally:
            doc.close()
    except Exception as e:  # noqa: BLE001 - recorded
        out["pdfium"]["error"] = _msg(e, path)
    pages = (exp.get("checks") or {}).get("textPages") or []
    if pages:
        _quiet_logs()
        from pdfminer.high_level import extract_text

        mismatches = []
        for index, want in pages:
            got = extract_text(path, page_numbers=[index])
            if " ".join(got.split()) != " ".join(want.split()):
                mismatches.append({"page": index, "pdfminer": got.strip()[:200]})
        out["pdfminer"] = {"opened": True, "pagesRead": len(pages), "mismatches": mismatches}
    else:
        out["pdfminer"] = {"run": False, "why": "no page of text listed (checks.textPages is empty)"}
    for tool in ("qpdf", "pdfium"):
        if not out[tool].get("opened"):
            notes.append(f"{label}: {tool} could not open it: {out[tool].get('error')}")
    if out["pdfminer"].get("mismatches"):
        notes.append(f"{label}: pdfminer.six's text differs on {len(out['pdfminer']['mismatches'])} page(s)")
    return out


def _strip_path(text, path):
    """The message without the folder the sample was written to, so it reads the same from any output folder."""
    return text.replace(os.path.abspath(path), os.path.basename(path)).replace(str(path), os.path.basename(path))


def _msg(e, path):
    return _strip_path(f"{type(e).__name__}: {e}", path).strip()


# -- run boxes against pdfium's characters -------------------------------------------------------------------------------

def check_boxes(exp_page, chars, page_geom):
    """Finds each expected run's characters among pdfium's (spaces and generated characters skipped; a right-to-left
    run also reversed, as pdfium may give logical order) and measures the run's box against the union of their loose
    boxes: the largest difference of the left and right edges, and of the top and bottom edges (points)."""
    real = [(chr(cp), box) for cp, gen, box, size in chars if not gen and box is not None and not chr(cp).isspace()]
    letters = "".join(ch for ch, _ in real)
    worst_x = worst_y = 0.0
    matched = 0
    pos = 0
    for run in exp_page["runs"]:
        text = "".join(ch for ch in run.get("glyphText", run["text"]) if not ch.isspace())
        if not text or "\ufffd" in text:
            continue
        k = -1
        for candidate in (text, text[::-1]):
            k = letters.find(candidate, pos)
            if k < 0:
                k = letters.find(candidate)
            if k >= 0:
                break
        if k < 0:
            continue
        pos = k + len(text)
        pts = []
        for l, b, r, t in (box for _, box in real[k:k + len(text)]):
            pts += [page_geom(l, b), page_geom(r, t), page_geom(l, t), page_geom(r, b)]
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        box = run["box"]
        full = run.get("glyphText", run["text"])
        if full == full.strip():  # a leading or trailing space widens the run's box, not the union of its letters
            worst_x = max(worst_x, abs(min(xs) - box[0]), abs(max(xs) - box[0] - box[2]))
            worst_y = max(worst_y, abs(min(ys) - box[1]), abs(max(ys) - box[1] - box[3]))
        matched += 1
    return round(worst_x, 3), round(worst_y, 3), matched


def geometry(exp_page):
    """The page's crop and rotation as a function: PDF user space -> the page as shown (top-left origin)."""
    x0, y0, x1, y1 = exp_page["cropBox"]
    w, h = x1 - x0, y1 - y0
    r = exp_page["rotation"]

    def f(x, y):
        u, v = x - x0, y - y0
        if r == 0:
            return (u, h - v)
        if r == 90:
            return (v, u)
        if r == 180:
            return (w - u, v)
        return (h - v, w - u)

    return f


# -- hostile samples: each tool in a child process ------------------------------------------------------------------------

def probe_child(tool, path, password):
    """Runs one tool on a file and prints a JSON outcome (called in the child)."""
    if sys.platform != "win32":
        import resource

        resource.setrlimit(resource.RLIMIT_AS, (MEMORY_LIMIT * 2, MEMORY_LIMIT * 2))
    out = {}
    try:
        if tool == "pikepdf":
            import pikepdf

            with pikepdf.open(path, password=password or "") as pdf:
                n = len(pdf.pages)
                total = 0
                for page in pdf.pages:
                    contents = page.obj.get("/Contents")
                    if contents is not None:
                        total += len(page.obj.Contents.read_bytes()) if isinstance(contents, pikepdf.Stream) else 0
                out = {"result": "opened", "pages": n, "contentBytes": total}
        elif tool == "pdfminer":
            pages = pdfminer_text(path, password)
            out = {"result": "opened", "pages": len(pages)}
        elif tool == "pdfium":
            import pypdfium2 as pdfium

            doc = pdfium.PdfDocument(path, password=password or None)
            n = len(doc)
            for i in range(n):
                doc[i].render(scale=72 / 72)
            out = {"result": "opened", "pages": n}
    except MemoryError:
        out = {"result": "error", "message": "MemoryError"}
    except RecursionError:
        out = {"result": "error", "message": "RecursionError"}
    except Exception as e:  # noqa: BLE001 - recorded
        out = {"result": "error", "message": _msg(e, path)[:300]}
    print(json.dumps(out))


def probe(script, path, password=None):
    results = {}
    for tool in ("pikepdf", "pdfminer", "pdfium"):
        cmd = [sys.executable, "-I", script, "--probe", tool, path] + (["--password", password] if password else [])
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=TIMEOUT)
            line = proc.stdout.strip().splitlines()[-1] if proc.stdout.strip() else ""
            if proc.returncode == 0 and line.startswith("{"):
                results[tool] = json.loads(line)
            else:
                results[tool] = {"result": "crashed", "returnCode": proc.returncode}
        except subprocess.TimeoutExpired:
            results[tool] = {"result": "timeout", "seconds": TIMEOUT}
    return results
