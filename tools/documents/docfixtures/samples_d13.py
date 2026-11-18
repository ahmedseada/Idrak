"""D13, any workload (only with --large, into its own folder): a 3,000-page file of text and images (shared and
per-page images, object streams), 1,000 small one-page files, one page holding a very large image (an A0 poster at
300 dpi, RGB JPEG, 418 MB decoded), and deeply nested form XObjects within a reader's limits (depths 8, 16 and 24,
and one form drawn 1,000 times). Each with timing-free checks: page counts, every page's text hash, the text of some
pages (pdfminer.six reads them as the reference check), image sizes, nesting depths. The files are written from bytes
here (not through the per-glyph recorder the small samples use), so thousands of pages stay quick."""
import hashlib
import io
import random

from PIL import Image as PILImage, ImageDraw, ImageFont

from .common import DEJAVU, Sample
from .filters import flate
from .images import pattern
from .pdf import B, N, PdfFile, Stream, Writer, escape_literal, sha256

D13 = ["D13"]
WORDS = ("report quarter revenue team office project budget delivery customer meeting schedule update review plan "
         "market service order supplier training release figures growth region support product launch design "
         "warehouse dashboard payment feedback staff summary target forecast campaign survey").split()


def _lines(rnd, n):
    out = []
    for _ in range(n):
        words = [rnd.choice(WORDS) for _ in range(rnd.randint(6, 9))]
        words[0] = words[0].capitalize()
        out.append(" ".join(words) + rnd.choice([".", ".", ",", ":"]))
    return out


def _helvetica(pdf):
    return pdf.add({"Type": N("Font"), "Subtype": N("Type1"), "BaseFont": N("Helvetica"),
                    "Encoding": N("WinAnsiEncoding")})


def _image(pdf, w, h, seed):
    rows = pattern(w, h, 3, 255, seed)
    data = bytes(v for row in rows for v in row)
    return pdf.add(Stream({"Type": N("XObject"), "Subtype": N("Image"), "Width": w, "Height": h,
                           "ColorSpace": N("DeviceRGB"), "BitsPerComponent": 8, "Filter": N("FlateDecode")},
                          flate(data), data)), sha256(data)


def _finish(pdf, pages, catalog, pages_ref, kind="objstm", info=None):
    pdf.put(pages_ref, {"Type": N("Pages"), "Kids": pages, "Count": len(pages)})
    pdf.put(catalog, {"Type": N("Catalog"), "Pages": pages_ref})
    trailer = {"Root": catalog}
    if info is not None:
        trailer["Info"] = pdf.add(info)
    w = Writer(pdf)
    w.revision(sorted(pdf.objects), kind, trailer, objstm_size=100, first=True)
    return bytes(w.buf)


def _exp(name, data, desc, checks, kind):
    return {"schema": 1, "name": name, "file": f"{name}.pdf", "kind": kind, "phases": D13, "description": desc,
            "size": len(data), "sha256": sha256(data), "hostile": None, "checks": checks}


def many_pages(seed, n=3000):
    pdf = PdfFile("1.7")
    catalog, pages_ref = pdf.alloc(), pdf.alloc()
    font = _helvetica(pdf)
    shared, shared_sha = _image(pdf, 32, 24, 1)
    rnd = random.Random(f"{seed}|large-pages")
    pages, hashes, samples = [], [], []
    unique = {}
    for k in range(n):
        lines = _lines(rnd, 12)
        ops = [b"BT", b"/F1 11 Tf", b"14 TL", b"50 740 Td", b"(Page %d) Tj" % (k + 1), b"T*", b"T*"]
        for line in lines:
            ops += [escape_literal(line.encode("latin-1")) + b" Tj", b"T*"]
        ops.append(b"ET")
        xobjects = {"Im1": shared}
        ops += [b"q 120 0 0 90 430 640 cm /Im1 Do Q"]
        if k % 10 == 0:
            img, digest = _image(pdf, 64, 48, k)
            xobjects["Im2"] = img
            unique[k] = digest
            ops += [b"q 160 0 0 120 50 400 cm /Im2 Do Q"]
        content = b"\n".join(ops) + b"\n"
        cref = pdf.add(Stream({"Filter": N("FlateDecode")}, flate(content, level=6), content))
        pages.append(pdf.add({"Type": N("Page"), "Parent": pages_ref, "MediaBox": [0, 0, 612, 792],
                              "Resources": {"Font": {"F1": font}, "XObject": xobjects}, "Contents": cref}))
        text = "\n".join([f"Page {k + 1}"] + lines)
        hashes.append(hashlib.sha256(text.encode("utf-8")).hexdigest())
        if k % 300 == 0 or k == n - 1:
            samples.append([k, text])
    data = _finish(pdf, pages, catalog, pages_ref, info={"Title": B(b"Three thousand pages"),
                                                         "Producer": B(b"Idrak document samples")})
    desc = (f"{n} Letter pages, each with a page number and 12 lines of generic business text (Helvetica), the same "
            "small image on every page and a page's own image on every tenth; object streams and a cross-reference "
            "stream. `checks.pageTextSha256` is each page's text (its lines joined by newlines, UTF-8) hashed.")
    checks = {"pageCount": n, "pageSize": [612, 792], "textPages": samples, "pageTextSha256": hashes,
              "images": {"shared": {"pixelWidth": 32, "pixelHeight": 24, "pixelsSha256": shared_sha, "pages": "all"},
                         "perPage": {str(k): {"pixelWidth": 64, "pixelHeight": 48, "pixelsSha256": v}
                                     for k, v in unique.items()}},
              "imageCount": {"distinct": 1 + len(unique), "placements": n + len(unique)}}
    return Sample("pages-3000", "d13", "large", D13, desc, data, _exp("pages-3000", data, desc, checks, "large"),
                  reference="pdf-light")


def many_small(seed, n=1000):
    files = {}
    entries = []
    rnd = random.Random(f"{seed}|large-small")
    for k in range(n):
        pdf = PdfFile("1.7")
        catalog, pages_ref = pdf.alloc(), pdf.alloc()
        font = _helvetica(pdf)
        lines = [f"Note {k + 1}"] + _lines(rnd, 3)
        ops = [b"BT", b"/F1 12 Tf", b"16 TL", b"40 250 Td"]
        for line in lines:
            ops += [escape_literal(line.encode("latin-1")) + b" Tj", b"T*"]
        ops.append(b"ET")
        content = b"\n".join(ops) + b"\n"
        cref = pdf.add(Stream({"Filter": N("FlateDecode")}, flate(content), content))
        page = pdf.add({"Type": N("Page"), "Parent": pages_ref, "MediaBox": [0, 0, 420, 300],
                        "Resources": {"Font": {"F1": font}}, "Contents": cref})
        data = _finish(pdf, [page], catalog, pages_ref, kind="table")
        name = f"small-{k + 1:04d}.pdf"
        files[name] = data
        entries.append({"file": name, "pageCount": 1, "text": "\n".join(lines), "size": len(data),
                        "sha256": sha256(data)})
    desc = (f"{n} small one-page PDFs (a title and three lines each, classic tables), for reading many files at once "
            "within the machine's resources; `files` lists each one's text.")
    exp = {"schema": 1, "name": "many-small", "file": None, "kind": "large-set", "phases": D13, "description": desc,
           "size": None, "sha256": None, "hostile": None, "checks": {"files": entries, "count": n}}
    return Sample("many-small", "d13/many-small", "large-set", D13, desc, None, exp, files, reference=None,
                  checks=[check_many])


def huge_image(seed):
    w, h = 9933, 14043  # A0 (841 by 1189 mm) at 300 dpi
    pt_w, pt_h = 2384, 3370
    base = PILImage.linear_gradient("L").resize((w, h))
    im = PILImage.merge("RGB", (base, base.rotate(180), PILImage.radial_gradient("L").resize((w, h))))
    draw = ImageDraw.Draw(im)
    font = ImageFont.truetype(DEJAVU, 320)
    draw.text((600, 800), "Annual company poster", fill=(20, 30, 60), font=font)
    rnd = random.Random(f"{seed}|poster")
    for _ in range(40):
        x, y = rnd.randrange(w), rnd.randrange(h)
        r = rnd.randrange(200, 1200)
        draw.ellipse((x - r, y - r, x + r, y + r), outline=(rnd.randrange(256), 90, 160), width=24)
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=80, optimize=False, subsampling=2)
    jpeg = buf.getvalue()
    del im, base
    pdf = PdfFile("1.7")
    catalog, pages_ref = pdf.alloc(), pdf.alloc()
    img = pdf.add(Stream({"Type": N("XObject"), "Subtype": N("Image"), "Width": w, "Height": h,
                          "ColorSpace": N("DeviceRGB"), "BitsPerComponent": 8, "Filter": N("DCTDecode")}, jpeg))
    content = b"q %d 0 0 %d 0 0 cm /Im1 Do Q\n" % (pt_w, pt_h)
    cref = pdf.add(Stream({}, content))
    page = pdf.add({"Type": N("Page"), "Parent": pages_ref, "MediaBox": [0, 0, pt_w, pt_h],
                    "Resources": {"XObject": {"Im1": img}}, "Contents": cref})
    data = _finish(pdf, [page], catalog, pages_ref, kind="table")
    desc = (f"One A0 page ({pt_w} by {pt_h} points) covered by one RGB JPEG of {w} by {h} pixels (300 dpi, {w * h * 3:,} "
            "bytes decoded): read, drawn and written with one band of buffers, never the whole image twice.")
    checks = {"pageCount": 1, "pageSize": [pt_w, pt_h],
              "image": {"pixelWidth": w, "pixelHeight": h, "components": 3, "encoding": "jpeg",
                        "decodedBytes": w * h * 3, "jpegSha256": sha256(jpeg), "jpegBytes": len(jpeg)},
              "textPages": []}
    return Sample("huge-image", "d13", "large", D13, desc, data, _exp("huge-image", data, desc, checks, "large"),
                  reference="pdf-light")


def deep_forms(seed):
    pdf = PdfFile("1.7")
    catalog, pages_ref = pdf.alloc(), pdf.alloc()
    font = _helvetica(pdf)
    pages = []
    texts = []
    for depth in (8, 16, 24):
        text = f"Nested {depth} forms deep"
        inner = b"BT /F1 12 Tf 10 10 Td " + escape_literal(text.encode("ascii")) + b" Tj ET 0 0 1 RG 2 2 196 26 re S\n"
        ref = pdf.add(Stream({"Type": N("XObject"), "Subtype": N("Form"), "BBox": [0, 0, 200, 30],
                              "Resources": {"Font": {"F1": font}}}, inner))
        for level in range(depth - 1):
            body = b"q 1 0 0 1 2 2 cm /Fx Do Q\n"
            ref = pdf.add(Stream({"Type": N("XObject"), "Subtype": N("Form"), "BBox": [0, 0, 300, 100],
                                  "Resources": {"XObject": {"Fx": ref}}}, body))
        content = b"q 1 0 0 1 60 200 cm /Fx Do Q\n"
        cref = pdf.add(Stream({}, content))
        pages.append(pdf.add({"Type": N("Page"), "Parent": pages_ref, "MediaBox": [0, 0, 420, 300],
                              "Resources": {"XObject": {"Fx": ref}}, "Contents": cref}))
        texts.append({"depth": depth, "text": text})
    tile = pdf.add(Stream({"Type": N("XObject"), "Subtype": N("Form"), "BBox": [0, 0, 10, 10],
                           "Resources": {}}, b"0.2 0.4 0.8 rg 1 1 8 8 re f\n"))
    ops = [b"q 1 0 0 1 %d %d cm /Ft Do Q" % (10 + (k % 40) * 10, 10 + (k // 40) * 11) for k in range(1000)]
    content = b"\n".join(ops) + b"\n"
    cref = pdf.add(Stream({"Filter": N("FlateDecode")}, flate(content), content))
    pages.append(pdf.add({"Type": N("Page"), "Parent": pages_ref, "MediaBox": [0, 0, 420, 300],
                          "Resources": {"XObject": {"Ft": tile}}, "Contents": cref}))
    data = _finish(pdf, pages, catalog, pages_ref, kind="table")
    desc = ("Form XObjects nested 8, 16 and 24 deep (one page each; the innermost draws a line of text and a "
            "rectangle, each level shifted by 2 points), and a page drawing one small form 1,000 times. A reader "
            "whose MaxDepth admits the depth reads the text; the depth counts the forms from the page's content.")
    checks = {"pageCount": 4, "pageSize": [420, 300], "depths": [t["depth"] for t in texts] + [1],
              "textPages": [[k, t["text"]] for k, t in enumerate(texts)] + [[3, ""]],
              "formDraws": {"3": 1000}}
    return Sample("deep-forms", "d13", "large", D13, desc, data, _exp("deep-forms", data, desc, checks, "large"),
                  reference="pdf-light")


def check_many(path, exp, notes):
    """Every small file opens in qpdf with one page, and pdfminer.six reads its text."""
    import os

    import pikepdf
    from pdfminer.high_level import extract_text

    from .reference import _quiet_logs

    _quiet_logs()
    folder = path if os.path.isdir(path) else os.path.dirname(path)
    opened = 0
    mismatches = []
    for e in exp["checks"]["files"]:
        p = os.path.join(folder, e["file"])
        with pikepdf.open(p) as pdf:
            if len(pdf.pages) == 1:
                opened += 1
        text = extract_text(p)
        if " ".join(text.split()) != " ".join(e["text"].split()):
            mismatches.append(e["file"])
    if mismatches or opened != len(exp["checks"]["files"]):
        notes.append(f"many-small: {opened} opened with one page, {len(mismatches)} texts differ")
    return "reference", {"qpdfOpenedWithOnePage": opened, "pdfminerMismatches": mismatches}, {}


def build(seed, out=None):
    return [many_pages(seed), many_small(seed), huge_image(seed), deep_forms(seed)]

