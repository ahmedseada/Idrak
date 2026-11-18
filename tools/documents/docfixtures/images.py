"""Image XObjects and inline images, and the pixels a reader must decode them to.

Expected pixels (as `ImageData` holds them: grey or RGB, 8 bits a sample, or 16 where the image has 16):
- samples unpacked at 1, 2, 4, 8 or 16 bits (rows padded to whole bytes), mapped by /Decode (default [0 1] a component,
  [0 2^bpc-1] for Indexed): v -> Dmin + v * (Dmax - Dmin) / (2^bpc - 1);
- DeviceGray, CalGray, ICCBased with N 1: grey round(c * 255) (round(c * 65535) at 16 bits); DeviceRGB, CalRGB,
  ICCBased with N 3: the same a channel (calibration and profiles not applied: read by the component count);
- DeviceCMYK: each component to 8 bits, then (255 - K) - X * (255 - K) / 255 with Pillow's rounding (the library's JPEG
  rule for CMYK);
- Indexed: the index rounded and clamped to hival, its base colour from the lookup, converted as the base;
- Lab: L*a*b* under the image's white point to XYZ, scaled to D65 per component, to sRGB (IEC 61966-2-1 matrix and
  curve), rounded (tolerance 2);
- an image mask (stencil) as grey: 0 where it paints, 255 where it does not (after /Decode);
- DCTDecode: the JPEG codec's own decode of the stream's bytes (Pillow, libjpeg-turbo: the library's JPEG reference).
"""
import io

from PIL import Image as PILImage

from .filters import a85, ahx, flate, lzw, png_predict, rle, tiff_predict
from .pdf import N, Stream, sha256
from .png import png_bytes

SRGB = ((3.2406, -1.5372, -0.4986), (-0.9689, 1.8758, 0.0415), (0.0557, -0.2040, 1.0570))
D65 = (0.9505, 1.0, 1.089)


def pattern(width, height, comps, maxv, seed=0):
    """Rows of component values: gradients across, bands down, a block and a diagonal, so every sample value occurs."""
    rows = []
    for y in range(height):
        row = []
        for x in range(width):
            for c in range(comps):
                if (x // 4 + y // 4 + seed) % 7 == 0:
                    v = maxv if c % 2 == 0 else 0
                elif x == y or x == width - 1 - y:
                    v = 0
                else:
                    t = (x * (c + 1) * 3 + y * (comps - c) * 5 + seed * 11 + c * 37) % (2 * maxv + 2)
                    v = t if t <= maxv else 2 * maxv + 1 - t
                row.append(v)
        rows.append(row)
    return rows


def pack(rows, bpc):
    out = bytearray()
    for row in rows:
        if bpc == 8:
            out += bytes(row)
        elif bpc == 16:
            for v in row:
                out += v.to_bytes(2, "big")
        else:
            line = bytearray((len(row) * bpc + 7) // 8)
            for i, v in enumerate(row):
                bit = i * bpc
                line[bit >> 3] |= v << (8 - bpc - (bit & 7))
            out += line
    return bytes(out)


def muldiv255(a, b):
    t = a * b + 128
    return ((t >> 8) + t) >> 8


def cmyk_rgb(c, m, y, k):
    nk = 255 - k
    return (max(0, min(255, nk - muldiv255(c, nk))), max(0, min(255, nk - muldiv255(m, nk))),
            max(0, min(255, nk - muldiv255(y, nk))))


def lab_rgb(L, a, b, white):
    fy = (L + 16) / 116
    fx = fy + a / 500
    fz = fy - b / 200

    def finv(t):
        return t ** 3 if t > 6 / 29 else 3 * (6 / 29) ** 2 * (t - 4 / 29)

    X = white[0] * finv(fx) * D65[0] / white[0]
    Y = white[1] * finv(fy) * D65[1] / white[1]
    Z = white[2] * finv(fz) * D65[2] / white[2]
    out = []
    for row in SRGB:
        v = row[0] * X + row[1] * Y + row[2] * Z
        v = 12.92 * v if v <= 0.0031308 else 1.055 * max(v, 0) ** (1 / 2.4) - 0.055
        out.append(max(0, min(255, round(v * 255))))
    return tuple(out)


def gray_icc():
    """A minimal ICC v2 grey display profile (gamma 2.2, D50 white), written here (the profile is not applied)."""
    import struct

    def tag_desc(text):
        t = text.encode("ascii") + b"\0"
        return b"desc" + bytes(4) + struct.pack(">I", len(t)) + t + bytes(4) + bytes(4) + bytes(2) + bytes(1) + bytes(67)

    def s15(v):
        return struct.pack(">i", round(v * 65536))

    tags = [(b"desc", tag_desc("Grey gamma 2.2")),
            (b"wtpt", b"XYZ " + bytes(4) + s15(0.9642) + s15(1.0) + s15(0.8249)),
            (b"kTRC", b"curv" + bytes(4) + struct.pack(">I", 1) + struct.pack(">H", 0x0233) + bytes(2)),
            (b"cprt", b"text" + bytes(4) + b"No copyright, generated test data\0")]
    table = struct.pack(">I", len(tags))
    offset = 128 + 4 + 12 * len(tags)
    body = b""
    for sig, data in tags:
        while (offset + len(body)) % 4:
            body += b"\0"
        table += sig + struct.pack(">II", offset + len(body), len(data))
        body += data
    size = 128 + len(table) + len(body)
    header = struct.pack(">I", size) + b"none" + bytes([2, 0x10, 0, 0]) + b"mntr" + b"GRAY" + b"XYZ " + bytes(12) \
        + b"acsp" + bytes(4) + bytes(4) + bytes(4) + bytes(4) + bytes(8) + bytes(4) + s15(0.9642) + s15(1.0) \
        + s15(0.8249) + bytes(4) + bytes(16) + bytes(28)
    assert len(header) == 128, len(header)
    return header + table + body


def srgb_icc():
    """Pillow's (LittleCMS's) sRGB profile, its creation date and profile ID cleared so the bytes repeat."""
    from PIL import ImageCms

    data = bytearray(ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes())
    data[24:36] = bytes(12)
    data[84:100] = bytes(16)
    return bytes(data)


class Image:
    """An image: its samples or JPEG bytes, colour space, filters; the XObject stream and the expected pixels."""

    def __init__(self, doc, width, height, bpc=8, cs="DeviceRGB", comps=3, rows=None, decode=None, filters=("Fl",),
                 predictor=None, image_mask=False, jpeg=None, cs_object=None, lookup=None, base=None, hival=None,
                 lab=None, label="", smask=None, mask=None, inline=False, extra=None):
        self.doc = doc
        self.width, self.height, self.bpc = width, height, bpc
        self.cs, self.comps = cs, comps
        self.rows = rows
        self.decode = decode
        self.filters = list(filters)
        self.predictor = predictor
        self.image_mask = image_mask
        self.jpeg = jpeg
        self.cs_object = cs_object
        self.lookup, self.base, self.hival = lookup, base, hival
        self.lab = lab
        self.label = label
        self.smask, self.mask = smask, mask
        self.inline = inline
        self.extra = extra or {}
        self.samples = pack(rows, bpc) if rows is not None else None
        self.ref = None
        self.stream = None
        self.pixels = self._pixels()
        if not inline:
            self.ref = doc.pdf.add(self._stream())

    # -- encoding ------------------------------------------------------------------------------------------------------

    def encoded(self):
        """The stored bytes and the filter / parameter names (long names; inline images abbreviate them)."""
        if self.jpeg is not None:
            return self.jpeg, ["DCTDecode"], [None]
        data = self.samples
        if self.predictor:
            kind, extra = self.predictor
            if kind == "png":
                data = png_predict(data, self.comps, self.bpc, self.width, pattern=extra)
                parms = {"Predictor": 15 if len(extra) > 1 else 10 + extra[0], "Colors": self.comps,
                         "BitsPerComponent": self.bpc, "Columns": self.width}
            else:
                data = tiff_predict(data, self.comps, self.bpc, self.width)
                parms = {"Predictor": 2, "Colors": self.comps, "BitsPerComponent": self.bpc, "Columns": self.width}
        else:
            parms = None
        names, all_parms = [], []
        for i, f in enumerate(reversed(self.filters)):
            name = {"Fl": "FlateDecode", "LZW": "LZWDecode", "AHx": "ASCIIHexDecode", "A85": "ASCII85Decode",
                    "RL": "RunLengthDecode"}[f]
            first = i == 0
            if f == "Fl":
                data = flate(data)
            elif f == "LZW":
                data = lzw(data)
            elif f == "AHx":
                data = ahx(data)
            elif f == "A85":
                data = a85(data)
            elif f == "RL":
                data = rle(data)
            names.insert(0, name)
            all_parms.insert(0, parms if first and f in ("Fl", "LZW") else None)
        return data, names, all_parms

    def cs_value(self):
        if self.cs_object is not None:
            return self.cs_object
        return N(self.cs)

    def _stream(self):
        data, names, parms = self.encoded()
        d = {"Type": N("XObject"), "Subtype": N("Image"), "Width": self.width, "Height": self.height}
        if self.image_mask:
            d["ImageMask"] = True
        else:
            d["ColorSpace"] = self.cs_value()
        d["BitsPerComponent"] = self.bpc
        if self.decode is not None:
            d["Decode"] = list(self.decode)
        if names:
            d["Filter"] = N(names[0]) if len(names) == 1 else [N(n) for n in names]
            if any(p is not None for p in parms):
                d["DecodeParms"] = parms[0] if len(parms) == 1 else [p if p is not None else None for p in parms]
        if self.smask is not None:
            d["SMask"] = self.smask.ref
        if isinstance(self.mask, Image):
            d["Mask"] = self.mask.ref
        elif self.mask is not None:
            d["Mask"] = list(self.mask)
        d.update(self.extra)
        self.stream = Stream(d, data, self.jpeg if self.jpeg is not None else self.samples)
        return self.stream

    # -- expected pixels -----------------------------------------------------------------------------------------------

    def _pixels(self):
        if self.jpeg is not None:
            im = PILImage.open(io.BytesIO(self.jpeg))
            im.load()
            if im.mode not in ("L", "RGB"):
                im = im.convert("RGB")
            return (1 if im.mode == "L" else 3, 8, im.tobytes())
        maxv = (1 << self.bpc) - 1
        if self.image_mask:
            d0, d1 = self.decode or (0, 1)
            out = bytearray()
            for row in self.rows:
                for v in row:
                    out.append(0 if d0 + v * (d1 - d0) / maxv == 0 else 255)
            return (1, 8, bytes(out))
        default = [0, maxv] if self.cs == "Indexed" else [0, 1] * self.comps
        if self.cs == "Lab":
            default = [0, 100] + list(self.lab.get("Range", [-100, 100, -100, 100]))
        dec = list(self.decode) if self.decode is not None else default
        kind = {"DeviceGray": "gray", "CalGray": "gray", "DeviceRGB": "rgb", "CalRGB": "rgb", "DeviceCMYK": "cmyk",
                "Indexed": "indexed", "Lab": "lab"}.get(self.cs)
        if self.cs == "ICCBased":
            kind = {1: "gray", 3: "rgb", 4: "cmyk"}[self.comps]
        sixteen = self.bpc == 16 and kind in ("gray", "rgb")
        out = bytearray()
        for row in self.rows:
            for x in range(self.width):
                vals = row[x * self.comps:(x + 1) * self.comps]
                cv = [dec[2 * i] + v * (dec[2 * i + 1] - dec[2 * i]) / maxv for i, v in enumerate(vals)]
                if kind in ("gray", "rgb"):
                    for c in cv:
                        c = min(1.0, max(0.0, c))
                        if sixteen:
                            out += round(c * 65535).to_bytes(2, "big")
                        else:
                            out.append(round(c * 255))
                elif kind == "cmyk":
                    out += bytes(cmyk_rgb(*(round(min(1.0, max(0.0, c)) * 255) for c in cv)))
                elif kind == "indexed":
                    idx = max(0, min(self.hival, round(cv[0])))
                    n = {"DeviceGray": 1, "DeviceRGB": 3, "DeviceCMYK": 4}[self.base]
                    entry = self.lookup[idx * n:(idx + 1) * n]
                    out += entry if n in (1, 3) else bytes(cmyk_rgb(*entry))
                elif kind == "lab":
                    out += bytes(lab_rgb(cv[0], cv[1], cv[2], self.lab["WhitePoint"]))
        channels = 1 if kind == "gray" or (kind == "indexed" and self.base == "DeviceGray") else 3
        return (channels, 16 if sixteen else 8, bytes(out))

    def png(self):
        ch, depth, data = self.pixels
        return png_bytes(self.width, self.height, ch, depth, data)

    def describe(self):
        data, names, parms = self.encoded() if self.stream is None else (self.stream.data, None, None)
        if names is None:
            from .filters import filter_names

            names = filter_names(self.stream.dict.get("Filter"))
        ch, depth, px = self.pixels
        cs = self.cs if not self.image_mask else None
        if self.cs == "ICCBased":
            cs = ["ICCBased", self.comps]
        elif self.cs == "Indexed":
            cs = ["Indexed", self.base, self.hival]
        out = {
            "label": self.label, "object": [self.ref.num, self.ref.gen] if self.ref else None, "inline": self.inline,
            "pixelWidth": self.width, "pixelHeight": self.height, "bitsPerComponent": self.bpc, "colourSpace": cs,
            "components": self.comps if not self.image_mask else 1, "decode": self.decode,
            "imageMask": self.image_mask, "encoding": "jpeg" if self.jpeg is not None else "raw", "filters": names,
            "predictor": None if not self.predictor else self.predictor[0],
            "samplesLength": len(self.samples) if self.samples is not None else None,
            "samplesSha256": sha256(self.samples) if self.samples is not None else None,
            "jpegSha256": sha256(self.jpeg) if self.jpeg is not None else None,
            "pixels": {"channels": ch, "depth": depth, "sha256": sha256(px),
                       "tolerance": 2 if self.cs == "Lab" else 0},
        }
        if self.smask is not None:
            out["softMask"] = self.smask.describe()
        if isinstance(self.mask, Image):
            out["mask"] = self.mask.describe()
        elif self.mask is not None:
            out["colourKeyMask"] = list(self.mask)
        return out


def jpeg_bytes(rows, width, height, mode, quality=85, subsampling=2, progressive=False):
    """A JPEG written by Pillow (libjpeg-turbo) from 8-bit rows; mode "L", "RGB" or "CMYK" (stored inverted, Adobe)."""
    raw = bytes(v for row in rows for v in row)
    im = PILImage.frombytes(mode, (width, height), raw)
    buf = io.BytesIO()
    opts = {"quality": quality, "optimize": False, "progressive": progressive}
    if mode != "L":
        opts["subsampling"] = subsampling
    im.save(buf, "JPEG", **opts)
    return buf.getvalue()
