"""D4, images and embedded files: image XObjects and inline images in every colour space and bit depth, /Decode
arrays, stencil and soft masks, JPEGs, placement by the CTM, a scan-like page (one JPEG and a hidden OCR text layer),
and embedded files (the EmbeddedFiles name tree and a file-attachment annotation). Each image's expected pixels are a
PNG beside the sample (see images.py for the rules)."""
import hashlib
import io
import random

from PIL import Image as PILImage, ImageDraw, ImageFont

from .common import DEJAVU, Sample, base_expected, helvetica
from .document import Doc
from .filters import ahx, flate, png_predict
from .images import Image, gray_icc, jpeg_bytes, pattern, srgb_icc
from .pdf import B, N, Stream, String, U16, sha256
from .png import png_bytes

D4 = ["D4"]


def _title(doc, p, text, f=None, y=None):
    f = f or helvetica(doc)
    c = p.content
    c.BT()
    c.Tf(f, 11)
    c.Td(20, y if y is not None else p.size()[1] - 20)
    c.show(text)
    c.ET()
    p.line(text)
    return f


def _attach(rec, img, base, files):
    rec["pixels"]["file"] = base + ".png"
    files[base + ".png"] = img.png()
    if img.smask is not None:
        rec["softMask"]["pixels"]["file"] = base + ".smask.png"
        files[base + ".smask.png"] = img.smask.png()
    if isinstance(img.mask, Image):
        rec["mask"]["pixels"]["file"] = base + ".mask.png"
        files[base + ".mask.png"] = img.mask.png()


def _grid(doc, name, title, images, files, cols=4, size=(80, 50), page=(420, 300), pre=None):
    p = doc.page(*page)
    _title(doc, p, title)
    c = p.content
    for k, img in enumerate(images):
        x = 20 + (k % cols) * 100
        y = page[1] - 40 - size[1] - (k // cols) * (size[1] + 25)
        if pre is not None:
            pre(c, k, img)
        rec = c.image(f"Im{k + 1}", img, (size[0], 0, 0, size[1], x, y))
        _attach(rec, img, f"{name}.p{len(doc.pages)}.im{k + 1}", files)
    return p


def _sample(doc, data, w, name, desc, files, phases=D4, kind="images"):
    exp = base_expected(doc, data, name, kind, phases, desc, w, objects=False)
    return Sample(name, "d4", kind, phases, desc, data, exp, files)


def _lookup_rgb(n):
    return bytes(v for i in range(n) for v in ((i * 53) % 256, (i * 101 + 40) % 256, (255 - i * 29) % 256))


def colour_spaces(seed):
    doc = Doc("colour-spaces", "1.5", seed)
    files = {}
    W, H = 24, 16
    icc3 = doc.pdf.add(Stream({"N": 3, "Alternate": N("DeviceRGB"), "Filter": N("FlateDecode")}, flate(srgb_icc()),
                              srgb_icc()))
    icc1 = doc.pdf.add(Stream({"N": 1, "Alternate": N("DeviceGray"), "Filter": N("FlateDecode")}, flate(gray_icc()),
                              gray_icc()))
    white = [0.9505, 1, 1.089]
    lab = {"WhitePoint": white, "Range": [-100, 100, -100, 100]}
    lut16 = _lookup_rgb(16)
    lut_gray = bytes([0, 128, 255])
    lut_cmyk = bytes(v for i in range(10) for v in ((i * 25) % 256, (i * 60) % 256, (200 - i * 20) % 256, (i * 9) % 256))
    imgs = [
        Image(doc, W, H, 8, "DeviceGray", 1, pattern(W, H, 1, 255, 1), label="DeviceGray 8"),
        Image(doc, W, H, 8, "DeviceRGB", 3, pattern(W, H, 3, 255, 2), label="DeviceRGB 8"),
        Image(doc, W, H, 8, "DeviceCMYK", 4, pattern(W, H, 4, 255, 3), label="DeviceCMYK 8"),
        Image(doc, W, H, 4, "Indexed", 1, pattern(W, H, 1, 15, 4), label="Indexed RGB 4-bit",
              cs_object=[N("Indexed"), N("DeviceRGB"), 15, String(lut16, hex=True)], lookup=lut16, base="DeviceRGB",
              hival=15),
        Image(doc, W, H, 2, "Indexed", 1, pattern(W, H, 1, 3, 5), label="Indexed grey 2-bit, hival 2 (index 3 clamps)",
              cs_object=[N("Indexed"), N("DeviceGray"), 2, String(lut_gray)], lookup=lut_gray, base="DeviceGray",
              hival=2),
        Image(doc, W, H, 8, "Indexed", 1, pattern(W, H, 1, 9, 6), label="Indexed CMYK 8-bit",
              cs_object=[N("Indexed"), N("DeviceCMYK"), 9, String(lut_cmyk, hex=True)], lookup=lut_cmyk,
              base="DeviceCMYK", hival=9),
        Image(doc, W, H, 8, "ICCBased", 3, pattern(W, H, 3, 255, 7), label="ICCBased N 3 (sRGB)",
              cs_object=[N("ICCBased"), icc3]),
        Image(doc, W, H, 8, "ICCBased", 1, pattern(W, H, 1, 255, 8), label="ICCBased N 1 (grey)",
              cs_object=[N("ICCBased"), icc1]),
        Image(doc, W, H, 8, "CalGray", 1, pattern(W, H, 1, 255, 9), label="CalGray",
              cs_object=[N("CalGray"), {"WhitePoint": white, "Gamma": 2.2}]),
        Image(doc, W, H, 8, "CalRGB", 3, pattern(W, H, 3, 255, 10), label="CalRGB",
              cs_object=[N("CalRGB"), {"WhitePoint": white, "Gamma": [2.2, 2.2, 2.2],
                                       "Matrix": [0.4124, 0.2126, 0.0193, 0.3576, 0.7152, 0.1192, 0.1805, 0.0722,
                                                  0.9505]}]),
        Image(doc, W, H, 8, "Lab", 3, pattern(W, H, 3, 255, 11), label="Lab", cs_object=[N("Lab"), lab], lab=lab),
    ]
    _grid(doc, "colour-spaces", "Colour spaces", imgs, files)
    doc.finish()
    data, w = doc.write("table")
    desc = ("Image XObjects in DeviceGray, DeviceRGB, DeviceCMYK, Indexed (RGB, grey with an index past hival, CMYK "
            "bases), ICCBased (N 3 and 1), CalGray, CalRGB and Lab.")
    return _sample(doc, data, w, "colour-spaces", desc, files)


def bit_depths(seed):
    doc = Doc("bit-depths", "1.5", seed)
    files = {}
    W, H = 13, 9
    imgs = []
    for comps, cs in ((1, "DeviceGray"), (3, "DeviceRGB")):
        for bpc in (1, 2, 4, 8, 16):
            imgs.append(Image(doc, W, H, bpc, cs, comps, pattern(W, H, comps, (1 << bpc) - 1, bpc + comps),
                              label=f"{cs} {bpc}-bit, width 13 (rows padded)"))
    _grid(doc, "bit-depths", "Bits per component 1, 2, 4, 8, 16", imgs, files, cols=5, size=(60, 40))
    doc.finish()
    data, w = doc.write("table")
    desc = "Grey and RGB images at 1, 2, 4, 8 and 16 bits a component, 13 pixels wide so rows end mid-byte."
    return _sample(doc, data, w, "bit-depths", desc, files)


def decode_arrays(seed):
    doc = Doc("decode-arrays", "1.5", seed)
    files = {}
    W, H = 20, 12
    lut = _lookup_rgb(16)
    imgs = [
        Image(doc, W, H, 8, "DeviceGray", 1, pattern(W, H, 1, 255, 1), decode=[1, 0], label="grey [1 0]"),
        Image(doc, W, H, 8, "DeviceRGB", 3, pattern(W, H, 3, 255, 2), decode=[1, 0, 0, 1, 1, 0],
              label="RGB [1 0 0 1 1 0]"),
        Image(doc, W, H, 4, "Indexed", 1, pattern(W, H, 1, 15, 3), decode=[15, 0], label="Indexed 4-bit [15 0]",
              cs_object=[N("Indexed"), N("DeviceRGB"), 15, String(lut, hex=True)], lookup=lut, base="DeviceRGB",
              hival=15),
        Image(doc, W, H, 1, "DeviceGray", 1, pattern(W, H, 1, 1, 4), decode=[1, 0], label="grey 1-bit [1 0]"),
        Image(doc, W, H, 8, "DeviceCMYK", 4, pattern(W, H, 4, 255, 5), decode=[1, 0, 1, 0, 1, 0, 1, 0],
              label="CMYK [1 0 1 0 1 0 1 0]"),
        Image(doc, W, H, 8, "DeviceGray", 1, pattern(W, H, 1, 255, 6), decode=[0.2, 0.8], label="grey [0.2 0.8]"),
    ]
    _grid(doc, "decode-arrays", "Decode arrays", imgs, files)
    doc.finish()
    data, w = doc.write("table")
    desc = "/Decode arrays: inverted grey, RGB and CMYK, an inverted index range, an inverted bit, a narrowed range."
    return _sample(doc, data, w, "decode-arrays", desc, files)


def masks(seed):
    doc = Doc("masks", "1.5", seed)
    files = {}
    W, H = 24, 16
    stencil = Image(doc, W, H, 1, None, 1, pattern(W, H, 1, 1, 1), image_mask=True, label="stencil")
    stencil_inv = Image(doc, W, H, 1, None, 1, pattern(W, H, 1, 1, 2), image_mask=True, decode=[1, 0],
                        label="stencil, /Decode [1 0]")
    soft = Image(doc, W, H, 8, "DeviceGray", 1, pattern(W, H, 1, 255, 3), label="soft mask")
    with_soft = Image(doc, W, H, 8, "DeviceRGB", 3, pattern(W, H, 3, 255, 4), smask=soft, label="RGB with /SMask")
    soft_small = Image(doc, 12, 8, 8, "DeviceGray", 1, pattern(12, 8, 1, 255, 5), label="smaller soft mask")
    with_small = Image(doc, W, H, 8, "DeviceRGB", 3, pattern(W, H, 3, 255, 6), smask=soft_small,
                       label="RGB with a smaller /SMask")
    mask = Image(doc, 48, 32, 1, None, 1, pattern(48, 32, 1, 1, 7), image_mask=True, label="explicit mask")
    with_mask = Image(doc, W, H, 8, "DeviceRGB", 3, pattern(W, H, 3, 255, 8), mask=mask,
                      label="RGB with an explicit /Mask (twice the size)")
    keyed = Image(doc, W, H, 8, "DeviceRGB", 3, pattern(W, H, 3, 255, 9), mask=[0, 60, 0, 255, 0, 255],
                  label="RGB with a colour key /Mask [0 60 0 255 0 255]")
    imgs = [stencil, stencil_inv, with_soft, with_small, with_mask, keyed]

    def fill(c, k, img):
        if img.image_mask:
            c.emit("0.1 0.4 0.8 rg" if k == 0 else "0.8 0.3 0.1 rg")

    _grid(doc, "masks", "Masks", imgs, files, pre=fill)
    doc.finish()
    data, w = doc.write("table")
    desc = ("Stencil masks (ImageMask, painted with the fill colour; one with /Decode [1 0]), soft masks (/SMask, the "
            "same size and smaller), an explicit /Mask image and a colour key /Mask.")
    return _sample(doc, data, w, "masks", desc, files)


def _photo(w, h, comps, seed):
    rows = []
    for y in range(h):
        row = []
        for x in range(w):
            cx, cy = x - w * 0.6, y - h * 0.4
            ring = int(((cx * cx + cy * cy) ** 0.5) * 6) % 256
            for c in range(comps):
                row.append((ring + x * (2 + c) + y * (3 - c % 2) + seed * 17 + c * 50) % 256)
        rows.append(row)
    return rows


def dct(seed):
    doc = Doc("dct", "1.5", seed)
    files = {}
    imgs = [
        Image(doc, 64, 48, 8, "DeviceGray", 1, None, label="grey JPEG",
              jpeg=jpeg_bytes(_photo(64, 48, 1, 1), 64, 48, "L")),
        Image(doc, 64, 48, 8, "DeviceRGB", 3, None, label="RGB JPEG 4:2:0",
              jpeg=jpeg_bytes(_photo(64, 48, 3, 2), 64, 48, "RGB", subsampling=2)),
        Image(doc, 64, 48, 8, "DeviceRGB", 3, None, label="RGB JPEG 4:4:4 progressive",
              jpeg=jpeg_bytes(_photo(64, 48, 3, 3), 64, 48, "RGB", subsampling=0, progressive=True)),
        Image(doc, 48, 32, 8, "DeviceCMYK", 4, None, label="CMYK JPEG (Adobe, inverted), /Decode [1 0 1 0 1 0 1 0]",
              jpeg=jpeg_bytes(_photo(48, 32, 4, 4), 48, 32, "CMYK"), decode=[1, 0, 1, 0, 1, 0, 1, 0]),
    ]
    _grid(doc, "dct", "JPEG images", imgs, files, size=(80, 60))
    doc.finish()
    data, w = doc.write("table")
    desc = ("DCTDecode images (Pillow, libjpeg-turbo): grey, RGB 4:2:0, RGB 4:4:4 progressive, and an Adobe CMYK JPEG "
            "(stored inverted) with /Decode [1 0 1 0 1 0 1 0]. Expected pixels: the JPEG codec's own decode of the "
            "stream's bytes (the codec already reads Adobe CMYK as inverted, so the /Decode is not applied again).")
    return _sample(doc, data, w, "dct", desc, files)


def inline_images(seed):
    doc = Doc("inline-images", "1.5", seed)
    files = {}
    p = doc.page(420, 300)
    _title(doc, p, "Inline images")
    c = p.content
    lut4 = bytes([255, 0, 0, 0, 160, 0, 0, 0, 255, 250, 200, 0])
    p.resource("ColorSpace", "CS0", [N("Indexed"), N("DeviceRGB"), 3, String(lut4, hex=True)])
    entries = []
    # 1: grey, ASCIIHex, abbreviated keys.
    img = Image(doc, 16, 8, 8, "DeviceGray", 1, pattern(16, 8, 1, 255, 1), filters=("AHx",), inline=True,
                label="grey 8-bit, /F /AHx")
    entries.append((img, b"/W 16 /H 8 /CS /G /BPC 8 /F /AHx", ahx(img.samples)))
    # 2: RGB, Flate with a PNG predictor (/DP).
    img = Image(doc, 12, 6, 8, "DeviceRGB", 3, pattern(12, 6, 3, 255, 2), inline=True, predictor=("png", (4,)),
                label="RGB 8-bit, /F /Fl with /DP Predictor 14")
    entries.append((img, b"/W 12 /H 6 /CS /RGB /BPC 8 /F /Fl /DP <</Predictor 14 /Colors 3 /BitsPerComponent 8 "
                         b"/Columns 12>>", flate(png_predict(img.samples, 3, 8, 12, pattern=(4,)))))
    # 3: a stencil (/IM true), raw bits.
    img = Image(doc, 16, 16, 1, None, 1, pattern(16, 16, 1, 1, 3), image_mask=True, inline=True, filters=(),
                label="stencil, /IM true, /D [1 0]", decode=[1, 0])
    entries.append((img, b"/IM true /W 16 /H 16 /BPC 1 /D [1 0]", img.samples))
    # 4: Indexed through a named colour space in the page's resources, 2 bits.
    img = Image(doc, 10, 10, 2, "Indexed", 1, pattern(10, 10, 1, 3, 4), inline=True, filters=(), lookup=lut4,
                base="DeviceRGB", hival=3, label="Indexed 2-bit, /CS /CS0 from the resources")
    entries.append((img, b"/W 10 /H 10 /CS /CS0 /BPC 2", img.samples))
    # 5: Indexed in line ([/I /RGB 1 <...>]), 1 bit.
    lut2 = bytes([20, 60, 200, 240, 220, 40])
    img = Image(doc, 9, 7, 1, "Indexed", 1, pattern(9, 7, 1, 1, 5), inline=True, filters=(), lookup=lut2,
                base="DeviceRGB", hival=1, label="Indexed 1-bit, /CS [/I /RGB 1 <...>]")
    entries.append((img, b"/W 9 /H 7 /CS [/I /RGB 1 <" + lut2.hex().upper().encode("ascii") + b">] /BPC 1",
                    img.samples))
    # 6: grey raw data that holds " EIQ" (EI preceded by a space, not followed by one).
    rows = pattern(20, 4, 1, 255, 6)
    rows[1][3:7] = [0x20, 0x45, 0x49, 0x51]
    img = Image(doc, 20, 4, 8, "DeviceGray", 1, rows, inline=True, filters=(), label="grey raw data holding ' EIQ'")
    entries.append((img, b"/W 20 /H 4 /CS /DeviceGray /BPC 8", img.samples))
    for k, (img, d, data) in enumerate(entries):
        x = 20 + (k % 3) * 130
        y = 180 - (k // 3) * 110
        if img.image_mask:
            c.emit("0.2 0.5 0.2 rg")
        rec = c.inline_image(img, (90, 0, 0, 70, x, y), d, data)
        rec["inlineDictionary"] = d.decode("latin-1")
        _attach(rec, img, f"inline-images.p1.im{k + 1}", files)
    doc.finish()
    data, w = doc.write("table")
    desc = ("Inline images (BI ID EI) with abbreviated keys and names: ASCIIHex, Flate with a predictor, a stencil, "
            "Indexed by a named resource and in line, and raw data that holds ' EIQ'.")
    return _sample(doc, data, w, "inline-images", desc, files)


def placement(seed):
    doc = Doc("image-placement", "1.5", seed)
    files = {}
    W, H = 16, 12
    p = doc.page(420, 300)
    _title(doc, p, "Images placed by the CTM")
    c = p.content
    imgs = [(Image(doc, W, H, 8, "DeviceRGB", 3, pattern(W, H, 3, 255, k), label=lab), m) for k, (lab, m) in enumerate([
        ("turned 90 degrees", (0, 60, -80, 0, 120, 60)),
        ("slanted", (80, 0, 20, 50, 160, 150)),
        ("flipped upside down", (80, 0, 0, -50, 300, 250)),
        ("mirrored", (-80, 0, 0, 50, 380, 40)),
    ])]
    for k, (img, m) in enumerate(imgs):
        rec = c.image(f"Im{k + 1}", img, m)
        _attach(rec, img, f"image-placement.p1.im{k + 1}", files)
    p2 = doc.page(420, 300, rotate=90)
    _title(doc, p2, "A turned page", y=280)
    img = Image(doc, W, H, 8, "DeviceRGB", 3, pattern(W, H, 3, 255, 9), label="on a page with /Rotate 90")
    rec = p2.content.image("Im1", img, (80, 0, 0, 50, 40, 40))
    _attach(rec, img, "image-placement.p2.im1", files)
    doc.finish()
    data, w = doc.write("table")
    desc = ("Images placed by the CTM: turned, slanted, flipped, mirrored (box: the bounding box of the placed unit "
            "square, top-left origin; ctm: the matrix as written), and one on a page with /Rotate 90.")
    return _sample(doc, data, w, "image-placement", desc, files)


SCAN_LINES = ["Company newsletter, March", "Our new office in the city centre opens next month.",
              "The support team answered 1,200 requests in February.", "Please send your project updates by Friday.",
              "Thank you all for a strong first quarter."]


def scan_ocr(seed):
    """A4 at 100 dpi: a grey JPEG of printed lines on paper, and a hidden text layer (render mode 3) with each line's
    text over it, scaled by Tz to the width the line has in the image, as OCR engines write searchable scans."""
    doc = Doc("scan-ocr", "1.5", seed)
    files = {}
    dpi = 100
    pw, ph = 595, 842
    px, py = round(pw * dpi / 72), round(ph * dpi / 72)
    rnd = random.Random(f"{seed}|scan")
    im = PILImage.new("L", (px, py), 238)
    pix = im.load()
    for _ in range(px * py // 40):
        x, y = rnd.randrange(px), rnd.randrange(py)
        pix[x, y] = 238 - rnd.randrange(30)
    draw = ImageDraw.Draw(im)
    boxes = []
    for i, line in enumerate(SCAN_LINES):
        size = 30 if i == 0 else 20
        font = ImageFont.truetype(DEJAVU, size)
        x0, y0 = 90, 120 + i * 55 + (0 if i == 0 else 20)
        draw.text((x0, y0), line, fill=25, font=font)
        bb = draw.textbbox((x0, y0), line, font=font)
        asc, desc = font.getmetrics()
        boxes.append((line, x0, y0 + asc, bb[2] - bb[0], size))
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=75, optimize=False)
    jpeg = buf.getvalue()
    img = Image(doc, px, py, 8, "DeviceGray", 1, None, jpeg=jpeg, label="scanned page, grey JPEG, 100 dpi")
    p = doc.page(pw, ph)
    c = p.content
    rec = c.image("Im1", img, (pw, 0, 0, ph, 0, 0))
    _attach(rec, img, "scan-ocr.p1.im1", files)
    f = helvetica(doc)
    scale = 72 / dpi
    for line, x, base, width, size in boxes:
        pt_size = size * scale
        natural = sum(g.width for g in f.glyphs(line)) / 1000 * pt_size
        c.BT()
        c.Tr(3)
        c.Tf(f, round(pt_size, 2))
        c.Tz(round(100 * width * scale / natural, 2))
        c.Tm(1, 0, 0, 1, round(x * scale, 2), round(ph - base * scale, 2))
        c.show(line)
        c.ET()
        p.line(line)
    doc.finish()
    data, w = doc.write("table")
    desc = ("A scan-like page: one grey JPEG covering an A4 page (100 dpi, printed lines on noisy paper) and a hidden "
            "OCR text layer (render mode 3, Helvetica, each line scaled by Tz to its width in the image).")
    return _sample(doc, data, w, "scan-ocr", desc, files, phases=["D3", "D4"], kind="scan")


CSV = (b"region,quarter,units,revenue\r\nNorth,Q1,1200,48000\r\nSouth,Q1,950,38000\r\nEast,Q1,1430,57200\r\n"
       b"West,Q1,780,31200\r\n")
NOTES = "Meeting notes\n\nAttendees: product, design, support.\nDecisions: ship the café menu update on Monday.\n".encode("utf-8")
INVENTORY = b'{"items": [{"sku": "A-100", "count": 40}, {"sku": "B-220", "count": 12}]}\n'


def attachments(seed):
    doc = Doc("attachments", "1.7", seed)
    p = doc.page(420, 300)
    _title(doc, p, "Attachments: three files in the document, one on this page")
    logo_px = bytes(v for y in range(8) for x in range(8) for v in ((x * 32) % 256, (y * 32) % 256, 128))
    logo = png_bytes(8, 8, 3, 8, logo_px)
    files_in = [("logo.png", "image/png", logo, "Team logo, 8 by 8 pixels"),
                ("meeting-notes.txt", "text/plain", NOTES, "Notes from the planning meeting"),
                ("quarterly-sales.csv", "text/csv", CSV, "Units and revenue by region")]
    specs = {}
    expected = []
    for name, mt, data, desc in files_in:
        ef = doc.pdf.add(Stream({"Type": N("EmbeddedFile"), "Subtype": N(mt),
                                 "Params": {"Size": len(data), "CheckSum": String(hashlib.md5(data).digest(), hex=True)},
                                 "Filter": N("FlateDecode")}, flate(data), data))
        specs[name] = doc.pdf.add({"Type": N("Filespec"), "F": B(name.encode("ascii")), "UF": U16(name),
                                   "Desc": B(desc.encode("ascii")), "EF": {"F": ef, "UF": ef},
                                   "AFRelationship": N("Data")})
        expected.append({"name": name, "fileName": name, "mediaType": mt, "size": len(data), "sha256": sha256(data),
                         "description": desc, "source": "EmbeddedFiles", "page": None})
    leaf1 = doc.pdf.add({"Limits": [B(b"logo.png"), B(b"meeting-notes.txt")],
                         "Names": [B(b"logo.png"), specs["logo.png"], B(b"meeting-notes.txt"),
                                   specs["meeting-notes.txt"]]})
    leaf2 = doc.pdf.add({"Limits": [B(b"quarterly-sales.csv"), B(b"quarterly-sales.csv")],
                         "Names": [B(b"quarterly-sales.csv"), specs["quarterly-sales.csv"]]})
    doc.catalog_extra["Names"] = {"EmbeddedFiles": doc.pdf.add({"Kids": [leaf1, leaf2]})}
    ef = doc.pdf.add(Stream({"Type": N("EmbeddedFile"), "Subtype": N("application/json"),
                             "Params": {"Size": len(INVENTORY)}}, INVENTORY))
    fs = doc.pdf.add({"Type": N("Filespec"), "F": B(b"inventory.json"), "UF": U16("inventory.json"),
                      "EF": {"F": ef}, "Desc": B(b"Stock counts")})
    annot = doc.pdf.add({"Type": N("Annot"), "Subtype": N("FileAttachment"), "Rect": [360, 230, 380, 255],
                         "Contents": B(b"Stock counts attached"), "Name": N("PushPin"), "FS": fs, "F": 4})
    p.annotations.append(annot)
    expected.append({"name": "inventory.json", "fileName": "inventory.json", "mediaType": "application/json",
                     "size": len(INVENTORY), "sha256": sha256(INVENTORY), "description": "Stock counts",
                     "source": "annotation", "page": 0})
    doc.finish()
    data, w = doc.write("table")
    desc = ("Embedded files: an EmbeddedFiles name tree with two leaves (Kids, Limits) holding a PNG, a UTF-8 text and "
            "a CSV (Flate, /Params with /Size and /CheckSum, no dates), and a file-attachment annotation on page 1 "
            "holding a JSON file (unfiltered).")
    files = {f"attachments.{e['name']}": d for e, (_, _, d, _) in zip(expected, files_in)}
    files["attachments.inventory.json"] = INVENTORY
    for e in expected:
        e["file"] = f"attachments.{e['name']}"
    exp = base_expected(doc, data, "attachments", "attachments", D4, desc, w, objects=False)
    exp["attachments"] = expected
    return Sample("attachments", "d4", "attachments", D4, desc, data, exp, files)


def build(seed):
    return [colour_spaces(seed), bit_depths(seed), decode_arrays(seed), masks(seed), dct(seed), inline_images(seed),
            placement(seed), scan_ocr(seed), attachments(seed)]
