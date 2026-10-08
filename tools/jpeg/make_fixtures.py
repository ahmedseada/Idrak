"""JPEG and image preprocessing fixtures for Idrak (needs: pip install pillow numpy opencv-python-headless transformers).

    python tools/jpeg/make_fixtures.py tests/Idrak.Tests/data/jpeg

Every NAME.jpg comes with NAME.ppm (or NAME.pgm for grey) holding the pixels Pillow decodes it to (libjpeg-turbo with
its defaults: the islow integer IDCT, fancy upsampling), which Idrak's JPEG codec is compared with. The files are made
by Pillow (baseline 4:4:4, 4:2:2 and 4:2:0, progressive, grey, restart intervals, RGB with Adobe's transform flag 0,
EXIF and comment segments, CMYK under Adobe's convention, the same patched to YCCK and stripped of its Adobe segment)
and OpenCV (4:4:0 and 4:1:1, which Pillow does not write). Four-component files are compared after convert("RGB").

The preprocessing cases (preprocess.json) resize, rescale and normalize source.png and small.png as transformers'
Gemma3ImageProcessorPil does (Pillow's resize, then float32 arithmetic); each case's pixel_values ([3, H, W] float32,
little-endian) is in NAME.f32. A grayscale case first converts with Pillow's "L" and back to "RGB". The colour cases
run every colour format of the codecs (colour/: PNG with alpha, palette and tRNS, grey and alpha, 4- and 16-bit,
interlaced; BMP palette, 24, 32 and 16-bit fields; PPM and PGM at 8 and 16 bits and a maxval of 100; the CMYK and YCCK
JPEGs) through the processor with do_convert_rgb, as it gets them from Image.open: once at their size, once resized.
Rerunning the script writes the same bytes.
"""
import io
import json
import os
import struct
import sys
import zlib

import numpy as np
from PIL import Image

import cv2
from transformers.models.gemma3.image_processing_pil_gemma3 import Gemma3ImageProcessorPil


def scene(width, height, seed):
    """A small test picture: gradients, hard-edged blocks, a dark stroke (text-like) and a little noise."""
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:height, 0:width].astype(np.float64)
    r = 40 + 180 * x / max(1, width - 1)
    g = 60 + 150 * y / max(1, height - 1)
    b = 128 + 100 * np.sin(x / 3.0) * np.cos(y / 4.0)
    img = np.stack([r, g, b], axis=-1)
    img[height // 4: height // 2, width // 5: width // 2] = [250, 20, 30]           # a red block
    img[height // 2: height * 3 // 4, width // 2: width * 4 // 5] = [10, 200, 240]  # a cyan block
    for t in range(min(width, height)):                                             # a dark diagonal stroke
        img[t, min(width - 1, t + width // 3)] = [15, 15, 15]
    img += rng.normal(0, 6, img.shape)
    return np.clip(np.rint(img), 0, 255).astype(np.uint8)


def jpeg_bytes(image, **options):
    out = io.BytesIO()
    image.save(out, "JPEG", **options)
    return out.getvalue()


def write_case(folder, name, data):
    with open(os.path.join(folder, name + ".jpg"), "wb") as f:
        f.write(data)
    decoded = Image.open(io.BytesIO(data))
    decoded.load()
    if decoded.mode == "CMYK":
        decoded = decoded.convert("RGB")                                             # as transformers' convert_to_rgb
    extension = ".pgm" if decoded.mode == "L" else ".ppm"
    decoded.save(os.path.join(folder, name + extension))                             # binary P5 / P6


def jpegs(folder):
    big = Image.fromarray(scene(45, 37, 1))
    odd = Image.fromarray(scene(17, 13, 2))
    grey = big.convert("L")
    cases = {
        "baseline-420": jpeg_bytes(big, quality=75, subsampling=2),
        "baseline-422": jpeg_bytes(big, quality=85, subsampling=1),
        "baseline-444": jpeg_bytes(big, quality=95, subsampling=0),
        "baseline-420-odd": jpeg_bytes(odd, quality=80, subsampling=2),
        "baseline-422-odd": jpeg_bytes(odd, quality=80, subsampling=1),
        "baseline-420-q10": jpeg_bytes(big, quality=10, subsampling=2),             # heavy quantization: clamping
        "optimized-420": jpeg_bytes(big, quality=85, subsampling=2, optimize=True),
        "progressive-420": jpeg_bytes(big, quality=85, subsampling=2, progressive=True),
        "progressive-444-odd": jpeg_bytes(odd, quality=90, subsampling=0, progressive=True),
        "grey": jpeg_bytes(grey, quality=85),
        "grey-progressive-odd": jpeg_bytes(odd.convert("L"), quality=85, progressive=True),
        "restart-420": jpeg_bytes(big, quality=85, subsampling=2, restart_marker_blocks=3),
        "restart-progressive": jpeg_bytes(big, quality=85, subsampling=2, progressive=True, restart_marker_rows=1),
        "rgb-adobe": jpeg_bytes(big, quality=90, subsampling=0, keep_rgb=True),
        "exif-comment": jpeg_bytes(big, quality=85, exif=Image.Exif().tobytes() + b"\x00" * 64, comment=b"scanned"),
        "cmyk": jpeg_bytes(Image.fromarray(scene(8, 8, 3)).convert("CMYK"), quality=85),
        "cmyk-colour": jpeg_bytes(colourful(13, 9, 6, alpha=False).convert("CMYK"), quality=90),
    }
    cases["ycck"] = adobe_transform(cases["cmyk-colour"], 2)                         # the same planes read as YCCK
    cases["cmyk-plain"] = without_adobe(cases["cmyk-colour"])                        # no Adobe segment: not inverted
    bgr = np.ascontiguousarray(scene(45, 37, 1)[:, :, ::-1])
    for name, factor in (("opencv-440", cv2.IMWRITE_JPEG_SAMPLING_FACTOR_440), ("opencv-411", cv2.IMWRITE_JPEG_SAMPLING_FACTOR_411)):
        ok, encoded = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 85, cv2.IMWRITE_JPEG_SAMPLING_FACTOR, factor])
        assert ok
        cases[name] = encoded.tobytes()
    for name, data in cases.items():
        write_case(folder, name, data)



# ------------------------------------------------------------------ colour images

def colourful(width, height, seed, alpha=True):
    """Saturated hue sweeps across, brightness down, a few pure primaries; with alpha, a gradient from clear to opaque
    (colours left under clear pixels, as transformers keeps them)."""
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:height, 0:width].astype(np.float64)
    hue = 6.0 * x / width
    channels = []
    for k in range(3):                                                              # hue to RGB, per channel
        phase = (hue + 2 * k) % 6
        channels.append(np.clip(np.abs(phase - 3) - 1, 0, 1))
    rgb = np.stack(channels, -1) * (0.35 + 0.65 * (1 - y / max(1, height - 1)))[..., None] * 255
    rgb[0, 0], rgb[0, 1], rgb[0, 2], rgb[1, 0] = [255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 255]
    rgb = np.clip(np.rint(rgb + rng.normal(0, 3, rgb.shape)), 0, 255).astype(np.uint8)
    if not alpha:
        return Image.fromarray(rgb)
    a = np.clip(np.rint(255 * (x + y) / (width + height - 2)), 0, 255).astype(np.uint8)
    return Image.fromarray(np.dstack([rgb, a]), "RGBA")


def segments(data):
    """The (marker, start, end) of each segment before the first scan."""
    at, out = 2, []
    while data[at] == 0xFF and data[at + 1] != 0xDA:
        length = data[at + 2] << 8 | data[at + 3]
        out.append((data[at + 1], at, at + 2 + length))
        at += 2 + length
    return out


def adobe_transform(data, transform):
    data = bytearray(data)
    for marker, start, end in segments(data):
        if marker == 0xEE:
            data[start + 4 + 11] = transform
    return bytes(data)


def without_adobe(data):
    for marker, start, end in segments(data):
        if marker == 0xEE:
            return data[:start] + data[end:]
    return data


def png_bytes(width, height, colour, depth, rows, interlace=False, chunks=b""):
    """A PNG written here (filter 0), for the depths and colour types Pillow does not save; rows[y] is the packed row."""
    def chunk(kind, body):
        return struct.pack(">I", len(body)) + kind + body + struct.pack(">I", zlib.crc32(kind + body) & 0xFFFFFFFF)
    samples = {0: 1, 2: 3, 3: 1, 4: 2, 6: 4}[colour]
    bits = samples * depth
    if not interlace:
        raw = b"".join(b"\x00" + bytes(r) for r in rows)
    else:
        pixels = [unpack(r, width, bits) for r in rows]                             # Adam7: each pass's own rows
        raw = b""
        for x0, y0, dx, dy in ((0, 0, 8, 8), (4, 0, 8, 8), (0, 4, 4, 8), (2, 0, 4, 4), (0, 2, 2, 4), (1, 0, 2, 2), (0, 1, 1, 2)):
            for y in range(y0, height, dy):
                picked = [pixels[y][x] for x in range(x0, width, dx)]
                if picked:
                    raw += b"\x00" + pack(picked, bits)
    header = struct.pack(">IIBBBBB", width, height, depth, colour, 0, 0, 1 if interlace else 0)
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", header) + chunks + chunk(b"IDAT", zlib.compress(raw, 9)) + chunk(b"IEND", b"")


def unpack(row, width, bits):
    """A packed row as per-pixel integers of `bits` bits (whole bytes, or sub-byte from the high bits down)."""
    if bits >= 8:
        size = bits // 8
        return [bytes(row[i * size:(i + 1) * size]) for i in range(width)]
    value = int.from_bytes(bytes(row), "big")
    total = len(row) * 8
    return [(value >> (total - bits * (i + 1))) & ((1 << bits) - 1) for i in range(width)]


def pack(pixels, bits):
    if bits >= 8:
        return b"".join(pixels)
    value, count = 0, 0
    for p in pixels:
        value, count = value << bits | int(p), count + bits
    pad = (-count) % 8
    return (value << pad).to_bytes((count + pad) // 8, "big")


def bmp_bytes(width, height, bits, rows, compression=0, masks=b""):
    """A bottom-up BMP with a 40-byte header (and masks after it for bit fields); rows[y] is the packed row, top first."""
    stride = (width * bits + 31) // 32 * 4
    body = b"".join(bytes(r).ljust(stride, b"\x00") for r in reversed(rows))
    offset = 54 + len(masks)
    return (b"BM" + struct.pack("<IHHI", offset + len(body), 0, 0, offset)
            + struct.pack("<IiiHHIIiiII", 40, width, height, 1, bits, compression, len(body), 2835, 2835, 0, 0) + masks + body)


def colours(folder):
    """The colour files of the codecs; returns their names, relative to `folder`."""
    out = os.path.join(folder, "colour")
    os.makedirs(out, exist_ok=True)
    w, h = 13, 9
    rgba = colourful(w, h, 7)
    rgb = rgba.convert("RGB")
    a = np.asarray(rgba)
    files = {}

    def save(name, image, **options):
        image.save(os.path.join(out, name), **options)
        files[name] = None

    def raw(name, data):
        with open(os.path.join(out, name), "wb") as f:
            f.write(data)
        files[name] = None

    save("rgba.png", rgba)
    save("rgb.png", rgb)
    save("la.png", rgba.convert("LA"))
    save("grey.png", rgb.convert("L"))
    palette = rgb.quantize(colors=12, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE)
    save("palette-trns.png", palette, transparency=bytes([0, 64, 128, 255] * 3))      # tRNS: transformers ignores it
    wide = (a.astype(np.uint32) * 257 + np.arange(w)[None, :, None] * 37) % 65536  # 16 bits with low bytes that matter
    raw("rgb16.png", png_bytes(w, h, 2, 16, [b"".join(struct.pack(">HHH", *p[:3]) for p in r) for r in wide]))
    raw("rgba16-interlaced.png", png_bytes(w, h, 6, 16, [b"".join(struct.pack(">HHHH", *p) for p in r) for r in wide], interlace=True))
    raw("la16.png", png_bytes(w, h, 4, 16, [b"".join(struct.pack(">HH", p[1], p[3]) for p in r) for r in wide]))
    grey4 = (np.asarray(rgb.convert("L")) >> 4).astype(int)
    raw("grey4-interlaced.png", png_bytes(w, h, 0, 4, [pack(list(r), 4) for r in grey4], interlace=True))
    raw("rgba-interlaced.png", png_bytes(w, h, 6, 8, [r.tobytes() for r in a], interlace=True))
    save("palette.bmp", palette)
    save("rgb.bmp", rgb)
    save("rgba.bmp", rgba)                                                             # 32 bits with bit fields
    p = a.astype(np.uint32)
    p565 = (p[..., 0] >> 3) << 11 | (p[..., 1] >> 2) << 5 | p[..., 2] >> 3
    raw("rgb565.bmp", bmp_bytes(w, h, 16, [b"".join(struct.pack("<H", v) for v in r) for r in p565], 3, struct.pack("<III", 0xF800, 0x07E0, 0x001F)))
    p555 = (p[..., 0] >> 3) << 10 | (p[..., 1] >> 3) << 5 | p[..., 2] >> 3
    raw("rgb555.bmp", bmp_bytes(w, h, 16, [b"".join(struct.pack("<H", v) for v in r) for r in p555]))
    save("rgb.ppm", rgb)
    raw("rgb16.ppm", f"P6\n{w} {h}\n65535\n".encode() + b"".join(struct.pack(">HHH", *q[:3]) for r in wide for q in r))
    hundred = (np.asarray(rgb.convert("L")).astype(int) * 100 // 255)
    raw("grey100.pgm", f"P5\n{w} {h}\n100\n".encode() + bytes(hundred.reshape(-1).tolist()))
    return ["colour/" + name for name in files] + ["cmyk-colour.jpg", "ycck.jpg", "cmyk-plain.jpg", "grey.jpg", "rgb-adobe.jpg"]


# ------------------------------------------------------------------ preprocessing

def config_for(height, width, resample, resize=True):
    return {
        "image_processor_type": "Gemma3ImageProcessor",
        "do_convert_rgb": None, "do_resize": resize, "size": {"height": height, "width": width}, "resample": resample,
        "do_rescale": True, "rescale_factor": 0.00392156862745098, "do_normalize": True,
        "image_mean": [0.5, 0.5, 0.5], "image_std": [0.5, 0.5, 0.5], "do_pan_and_scan": None,
    }


def preprocessing(folder, colour_files):
    Image.fromarray(scene(83, 61, 4)).save(os.path.join(folder, "source.png"))      # 83 wide, 61 high
    Image.fromarray(scene(13, 11, 5)).save(os.path.join(folder, "small.png"))
    cases = [
        ("down-bilinear", "source.png", config_for(24, 32, 2), False),
        ("down-bicubic", "source.png", config_for(24, 32, 3), False),
        ("down-bilinear-odd", "source.png", config_for(17, 13, 2), False),
        ("down-lanczos", "source.png", config_for(20, 20, 1), False),
        ("up-bilinear", "small.png", config_for(29, 31, 2), False),
        ("up-bicubic", "small.png", config_for(29, 31, 3), False),
        ("mixed-bicubic", "source.png", config_for(64, 20, 3), False),             # taller, narrower
        ("grey-bilinear", "source.png", config_for(24, 32, 2), True),
    ]
    cases[1][2]["image_mean"] = [0.48145466, 0.4578275, 0.40821073]               # CLIP's mean and std, per channel
    cases[1][2]["image_std"] = [0.26862954, 0.26130258, 0.27577711]
    for file in colour_files:
        stem = os.path.splitext(file.replace("colour/", "colour-"))[0] + "-" + os.path.splitext(file)[1][1:]
        cases.append((stem + "-as-is", file, config_for(4, 4, 2, resize=False), False))
        cases.append((stem + "-resized", file, config_for(7, 10, 2), False))
    listed = []
    for name, image, config, grey in cases:
        # null in Gemma 3's file; loading it gives the class default, True (passing None here would turn it off)
        processor = Gemma3ImageProcessorPil(**{k: (True if k == "do_convert_rgb" else v) for k, v in config.items() if k != "image_processor_type"})
        picture = Image.open(os.path.join(folder, image))                           # as transformers gets it
        if grey:
            picture = picture.convert("L").convert("RGB")
        values = processor(picture, return_tensors="np")["pixel_values"][0].astype("<f4")
        size = config["size"]
        assert not config["do_resize"] or values.shape == (3, size["height"], size["width"]), values.shape
        assert values.shape[0] == 3, values.shape
        values.tofile(os.path.join(folder, name + ".f32"))
        listed.append({"name": name, "image": image, "grayscale": grey, "config": config})
    with open(os.path.join(folder, "preprocess.json"), "w", newline="\n") as f:
        json.dump(listed, f, indent=1)
        f.write("\n")


def main():
    folder = sys.argv[1] if len(sys.argv) > 1 else "tests/Idrak.Tests/data/jpeg"
    os.makedirs(folder, exist_ok=True)
    jpegs(folder)
    preprocessing(folder, colours(folder))


if __name__ == "__main__":
    main()
