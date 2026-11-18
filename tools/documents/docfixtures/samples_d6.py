"""D6, writing: the inputs D6's tests write from (images of each kind as PNG, each with what it measures and the
`Auto` encoding it must get, photographic or line art, decision 11), and image PDFs as scanners and phones write them
(one JPEG a page at several resolutions and paper sizes, a Flate page with a PNG predictor, a phone photo on a turned
page, a searchable scan with a hidden text layer in a composite font). Files Idrak writes are checked by
tools/documents/check_written.py (qpdf, pdfminer.six, pdfium)."""
import io
import random

from PIL import Image as PILImage, ImageChops, ImageDraw, ImageFilter, ImageFont

from .common import DEJAVU, Sample, base_expected
from .document import Doc
from .fonts import Type0
from .images import Image
from .pdf import sha256
from .png import png_bytes

D6 = ["D6"]


# -- the inputs ----------------------------------------------------------------------------------------------------------

def _noise(seed, label, size, amplitude):
    """A grey noise image from seeded bytes (Pillow's own noise effect is not seeded), centred on 128."""
    rnd = random.Random(f"{seed}|{label}")
    raw = rnd.randbytes(size[0] * size[1])
    im = PILImage.frombytes("L", size, raw)
    return im.point(lambda v: 128 + (v - 128) * amplitude // 128)


def _photo(seed, size, grey=False):
    """A synthetic photograph: smooth light falloff, soft shapes, sensor noise."""
    w, h = size
    r = PILImage.linear_gradient("L").resize(size).rotate(25, resample=PILImage.BICUBIC, expand=False, fillcolor=90)
    g = PILImage.radial_gradient("L").resize(size)
    b = PILImage.linear_gradient("L").transpose(PILImage.Transpose.ROTATE_90).resize(size)
    im = PILImage.merge("RGB", (r, g.point(lambda v: 255 - v), b))
    draw = ImageDraw.Draw(im)
    rnd = random.Random(f"{seed}|photo-shapes")
    for _ in range(14):
        x, y = rnd.randrange(w), rnd.randrange(h)
        rad = rnd.randrange(20, max(21, w // 5))
        col = tuple(rnd.randrange(40, 230) for _ in range(3))
        draw.ellipse((x - rad, y - rad, x + rad, y + rad), fill=col)
    im = im.filter(ImageFilter.GaussianBlur(radius=max(3, w // 90)))
    noise = _noise(seed, "photo-noise", size, 18)
    im = PILImage.merge("RGB", [PILImage.blend(c, noise, 0.12) for c in im.split()])
    return im.convert("L") if grey else im


def _scan(seed, size, lines):
    """A scanned document page: off-white paper with grain, printed lines (anti-aliased), a slight blur."""
    w, h = size
    paper = PILImage.new("L", size, 236)
    noise = _noise(seed, "scan-grain", size, 40)
    im = PILImage.blend(paper, noise, 0.18)
    draw = ImageDraw.Draw(im)
    font = ImageFont.truetype(DEJAVU, max(12, h // 55))
    for i, line in enumerate(lines):
        draw.text((w // 10, h // 10 + i * (h // 28)), line, fill=30, font=font)
    return im.filter(ImageFilter.GaussianBlur(radius=0.6))


def _line_art(size, grey=False):
    """A diagram: flat fills and thin lines in a few colours on white, no anti-aliasing."""
    w, h = size
    im = PILImage.new("RGB", size, (255, 255, 255))
    draw = ImageDraw.Draw(im)
    colours = [(31, 119, 180), (255, 127, 14), (44, 160, 44), (214, 39, 40), (0, 0, 0)]
    for i in range(6):
        x0 = 40 + i * (w - 80) // 6
        top = h - 60 - (i * 37 % 5 + 1) * (h - 120) // 6
        draw.rectangle((x0 + 6, top, x0 + (w - 80) // 6 - 6, h - 60), fill=colours[i % 4])
    draw.line((30, h - 60, w - 30, h - 60), fill=(0, 0, 0), width=2)
    draw.line((40, 30, 40, h - 50), fill=(0, 0, 0), width=2)
    draw.polygon([(w - 150, 40), (w - 60, 40), (w - 105, 110)], outline=(0, 0, 0), fill=(255, 221, 87))
    for k in range(0, w - 80, 12):
        draw.point((40 + k, 60 + (k * 7) % 40), fill=(80, 80, 80))
    return im.convert("L") if grey else im


def _screenshot(size):
    """An application window: flat panels, a toolbar, text without anti-aliasing (a few colours)."""
    w, h = size
    im = PILImage.new("RGB", size, (246, 247, 249))
    draw = ImageDraw.Draw(im)
    draw.fontmode = "1"
    font = ImageFont.truetype(DEJAVU, 14)
    draw.rectangle((0, 0, w, 36), fill=(45, 62, 80))
    draw.text((12, 10), "Project dashboard", fill=(255, 255, 255), font=font)
    draw.rectangle((0, 36, 160, h), fill=(232, 236, 241))
    for i, item in enumerate(["Overview", "Tasks", "Reports", "Settings"]):
        draw.text((16, 56 + i * 28), item, fill=(40, 40, 40), font=font)
    for i, row in enumerate(["Open tasks: 14", "Closed this week: 9", "Next release: 2.3"]):
        draw.rectangle((180, 56 + i * 52, w - 20, 96 + i * 52), fill=(255, 255, 255), outline=(200, 205, 212))
        draw.text((192, 68 + i * 52), row, fill=(20, 20, 20), font=font)
    return im


def _bilevel(size):
    w, h = size
    im = PILImage.new("L", size, 255)
    draw = ImageDraw.Draw(im)
    draw.fontmode = "1"
    font = ImageFont.truetype(DEJAVU, 18)
    for i, line in enumerate(["Shipping note", "Order 4412: two boxes of paper, one printer.",
                              "Delivered to the front desk."]):
        draw.text((24, 24 + i * 30), line, fill=0, font=font)
    draw.rectangle((20, h - 60, w - 20, h - 20), outline=0, width=2)
    return im


def measure(im):
    """What the `Auto` rule measures: distinct colours, and how neighbouring pixels differ (each pixel against its right
    and its lower neighbour, the largest channel difference): the share that differ a little (1-16 levels: smooth
    photographic shading and noise) and a lot (over 64: sharp edges)."""
    w, h = im.size
    colours = len(im.getcolors(maxcolors=w * h))
    hist = [0] * 256
    for a, b in ((im.crop((0, 0, w - 1, h)), im.crop((1, 0, w, h))), (im.crop((0, 0, w, h - 1)), im.crop((0, 1, w, h)))):
        diff = ImageChops.difference(a, b)
        if len(diff.getbands()) > 1:
            bands = diff.split()
            diff = bands[0]
            for band in bands[1:]:
                diff = ImageChops.lighter(diff, band)
        for level, count in enumerate(diff.histogram()):
            hist[level] += count
    total = sum(hist)
    small = sum(hist[1:17])
    large = sum(hist[65:])
    return {"distinctColours": colours, "smallSteps": round(small / total, 4), "largeSteps": round(large / total, 4)}


INPUTS = [
    # name, maker, expected Auto, kind
    ("photo-rgb", lambda s: _photo(s, (480, 360)), "Jpeg", "photograph"),
    ("photo-grey", lambda s: _photo(s, (480, 360), grey=True), "Jpeg", "photograph"),
    ("scan-grey", lambda s: _scan(s, (620, 877), ["Monthly report", "Sales rose in every region this quarter.",
                                                 "The new warehouse opens in May.", "Questions go to the office team."]),
     "Jpeg", "scan"),
    ("line-art-rgb", lambda s: _line_art((480, 360)), "Flate", "line art"),
    ("line-art-grey", lambda s: _line_art((480, 360), grey=True), "Flate", "line art"),
    ("screenshot-rgb", lambda s: _screenshot((480, 320)), "Flate", "screenshot"),
    ("bilevel-grey", lambda s: _bilevel((480, 200)), "Flate", "two levels"),
]


def inputs(seed):
    files = {}
    entries = []
    for name, make, auto, kind in INPUTS:
        im = make(seed)
        ch = len(im.getbands())
        stats = measure(im)
        if auto == "Jpeg":
            why = (f"{kind}: {stats['distinctColours']} distinct colours and {stats['smallSteps']:.0%} of neighbouring "
                   f"pixels differ by 1-16 levels (smooth shading and noise, which Flate cannot pack and JPEG keeps "
                   f"well)")
        else:
            why = (f"{kind}: {stats['distinctColours']} distinct colours, {stats['smallSteps']:.1%} of neighbouring "
                   f"pixels differ a little and {stats['largeSteps']:.1%} by over 64 levels (flat areas and sharp "
                   f"edges, which Flate keeps exactly and small and JPEG would blur)")
        data = im.tobytes()
        files[f"inputs.{name}.png"] = png_bytes(im.width, im.height, ch, 8, data)
        entries.append({"name": name, "file": f"inputs.{name}.png", "kind": kind, "width": im.width,
                        "height": im.height, "channels": ch, "pixelsSha256": sha256(data), **stats,
                        "expectedAuto": auto, "why": why})
    desc = ("Images D6's writer is given, as PNG (8-bit grey or RGB): photographs, a scanned page, line art, a "
            "screenshot and a two-level image, each with what the Auto rule measures (distinct colours, the share "
            "of neighbouring pixels differing by 1-16 and by over 64 levels) and the encoding Auto must choose "
            "(decision 11: JPEG for photographs and scans, Flate for line art and few levels).")
    exp = {"schema": 1, "name": "inputs", "file": None, "kind": "writer-inputs", "phases": D6, "description": desc,
           "size": None, "sha256": None, "hostile": None, "images": entries,
           "text": {"lines": ["Quarterly report", "Revenue grew by 12 percent.", "The team met every target.",
                              "Office move planned for June."],
                    "rtl": ["تقرير الربع الأول", "زادت المبيعات في كل المناطق"]}}
    return Sample("inputs", "d6", "writer-inputs", D6, desc, None, exp, files, reference=None)


# -- image PDFs as scanners and phones write them ------------------------------------------------------------------------

PAGES = [  # paper, size in points, dpi, mode
    ("A4", (595, 842), 150, "L"),
    ("Letter", (612, 792), 200, "RGB"),
    ("A5", (420, 595), 100, "L"),
]

LINES = [["Meeting notes", "The design review moved to Thursday.", "Budget figures are due next week."],
         ["Purchase summary", "Ten laptops and four monitors were ordered.", "Delivery is expected in two weeks."],
         ["Newsletter", "Our support team welcomes two new members.", "The office kitchen reopens on Monday."]]


def _attach(rec, img, base, files):
    rec["pixels"]["file"] = base + ".png"
    files[base + ".png"] = img.png()


def scanner_pages(seed):
    doc = Doc("scanner-jpeg-pages", "1.4", seed)
    files = {}
    for k, ((paper, (pw, ph), dpi, mode), lines) in enumerate(zip(PAGES, LINES)):
        px, py = round(pw * dpi / 72), round(ph * dpi / 72)
        im = _scan(f"{seed}|{k}", (px, py), lines)
        if mode == "RGB":
            im = PILImage.merge("RGB", (im, im.point(lambda v: min(255, v + 4)), im.point(lambda v: max(0, v - 6))))
        buf = io.BytesIO()
        im.save(buf, "JPEG", quality=80, optimize=False)
        img = Image(doc, px, py, 8, "DeviceGray" if mode == "L" else "DeviceRGB", 1 if mode == "L" else 3, None,
                    jpeg=buf.getvalue(), label=f"{paper} at {dpi} dpi, {'grey' if mode == 'L' else 'RGB'} JPEG")
        p = doc.page(pw, ph)
        rec = p.content.image("Im1", img, (pw, 0, 0, ph, 0, 0))
        _attach(rec, img, f"scanner-jpeg-pages.p{k + 1}.im1", files)
    doc.finish()
    data, w = doc.write("table")
    desc = ("Three scanned pages as a scanner writes them: one JPEG covering each page (A4 grey at 150 dpi, Letter RGB "
            "at 200 dpi, A5 grey at 100 dpi), no text layer.")
    exp = base_expected(doc, data, "scanner-jpeg-pages", "image-pdf", D6, desc, w, objects=False)
    return Sample("scanner-jpeg-pages", "d6", "image-pdf", D6, desc, data, exp, files)


def flate_page(seed):
    doc = Doc("flate-page", "1.5", seed)
    files = {}
    im = _bilevel((960, 400))
    rows = [list(im.tobytes()[y * 960:(y + 1) * 960]) for y in range(400)]
    img = Image(doc, 960, 400, 8, "DeviceGray", 1, rows, predictor=("png", (0, 1, 2, 3, 4)),
                label="line art, Flate with PNG predictors (per-row filter types 0-4)")
    p = doc.page(576, 240)
    rec = p.content.image("Im1", img, (576, 0, 0, 240, 0, 0))
    _attach(rec, img, "flate-page.p1.im1", files)
    im2 = _line_art((480, 360))
    rows2 = [list(im2.tobytes()[y * 1440:(y + 1) * 1440]) for y in range(360)]
    img2 = Image(doc, 480, 360, 8, "DeviceRGB", 3, rows2, predictor=("png", (4,)), label="diagram, Flate, Paeth")
    p2 = doc.page(480, 360)
    rec = p2.content.image("Im1", img2, (480, 0, 0, 360, 0, 0))
    _attach(rec, img2, "flate-page.p2.im1", files)
    doc.finish()
    data, w = doc.write("stream")
    desc = ("Image pages written exactly: a two-level grey page and an RGB diagram, Flate with PNG predictors, as the "
            "writer's Flate encoding writes line art (a cross-reference stream).")
    exp = base_expected(doc, data, "flate-page", "image-pdf", D6, desc, w, objects=False)
    return Sample("flate-page", "d6", "image-pdf", D6, desc, data, exp, files)


def phone_photo(seed):
    doc = Doc("phone-photo", "1.4", seed)
    files = {}
    im = _photo(seed, (640, 480))
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=88, optimize=False, subsampling=2)
    img = Image(doc, 640, 480, 8, "DeviceRGB", 3, None, jpeg=buf.getvalue(), label="phone photo, RGB JPEG 4:2:0")
    p = doc.page(640, 480)
    rec = p.content.image("Im1", img, (640, 0, 0, 480, 0, 0))
    _attach(rec, img, "phone-photo.p1.im1", files)
    p2 = doc.page(640, 480, rotate=90)
    img2 = Image(doc, 640, 480, 8, "DeviceRGB", 3, None, jpeg=buf.getvalue(),
                 label="the same photo on a page turned by /Rotate 90 (a portrait shot)")
    rec = p2.content.image("Im1", img2, (640, 0, 0, 480, 0, 0))
    _attach(rec, img2, "phone-photo.p2.im1", files)
    doc.finish()
    data, w = doc.write("table")
    desc = ("Phone photos as phone apps write them: the page the photo's size at 72 dpi, one RGB JPEG; the second page "
            "turned by /Rotate 90 (a portrait shot), so the page as shown is 480 by 640.")
    exp = base_expected(doc, data, "phone-photo", "image-pdf", D6, desc, w, objects=False)
    return Sample("phone-photo", "d6", "image-pdf", D6, desc, data, exp, files)


def searchable_scan(seed):
    """Two scanned pages with a hidden text layer in a composite font (Type0, Identity-H, DejaVu Sans subset with
    ToUnicode), each line scaled by Tz to its width in the image, as a searchable PDF made from an OCR run."""
    doc = Doc("searchable-scan", "1.7", seed)
    files = {}
    f = doc.font(Type0, DEJAVU, tu_style="mixed")
    dpi = 100
    pw, ph = 595, 842
    px, py = round(pw * dpi / 72), round(ph * dpi / 72)
    for k, lines in enumerate(LINES[:2]):
        im = PILImage.new("L", (px, py), 240)
        im = PILImage.blend(im, _noise(seed, f"searchable-{k}", (px, py), 30), 0.15)
        draw = ImageDraw.Draw(im)
        font = ImageFont.truetype(DEJAVU, 22)
        placed = []
        for i, line in enumerate(lines):
            x0, y0 = 90, 140 + i * 50
            draw.text((x0, y0), line, fill=25, font=font)
            bb = draw.textbbox((x0, y0), line, font=font)
            asc, _ = font.getmetrics()
            placed.append((line, x0, y0 + asc, bb[2] - bb[0]))
        buf = io.BytesIO()
        im.save(buf, "JPEG", quality=75, optimize=False)
        img = Image(doc, px, py, 8, "DeviceGray", 1, None, jpeg=buf.getvalue(), label="scanned page, grey JPEG")
        p = doc.page(pw, ph)
        c = p.content
        rec = c.image("Im1", img, (pw, 0, 0, ph, 0, 0))
        _attach(rec, img, f"searchable-scan.p{k + 1}.im1", files)
        scale = 72 / dpi
        for line, x, base, width in placed:
            size = round(22 * scale, 2)
            glyphs = f.glyphs(line)
            natural = sum(g.width for g in glyphs) / 1000 * size
            c.BT()
            c.Tr(3)
            c.Tf(f, size)
            c.Tz(round(100 * width * scale / natural, 2))
            c.Tm(1, 0, 0, 1, round(x * scale, 2), round(ph - base * scale, 2))
            c.show(glyphs=glyphs)
            c.ET()
            p.line(line)
    doc.finish()
    data, w = doc.write("objstm")
    desc = ("A searchable scan as an OCR run writes it: two pages, each one grey JPEG and a hidden text layer (render "
            "mode 3) in a composite font (Type0, Identity-H, a DejaVu Sans subset with ToUnicode), each line scaled "
            "by Tz to its width in the image; object streams and a cross-reference stream.")
    exp = base_expected(doc, data, "searchable-scan", "scan", D6, desc, w, objects=False)
    return Sample("searchable-scan", "d6", "scan", ["D3", "D4", "D6"], desc, data, exp, files)


def build(seed):
    return [inputs(seed), scanner_pages(seed), flate_page(seed), phone_photo(seed), searchable_scan(seed)]
