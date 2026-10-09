"""Writes tests/Idrak.Tests/data/image-transforms: Pillow's results for Idrak's image transforms and JPEG encoder.

Run: python tools/vlm/make_image_transforms.py   (Pillow 12.3.0 with libjpeg-turbo 3.1; numpy). Reruns write the same bytes.

Inputs: the fixtures of phase 0 (tests/Idrak.Tests/data/vlm) and of the JPEG decoder (tests/Idrak.Tests/data/jpeg),
plus three made here: `scan.jpg`, a scan-like page (1,654 x 2,339, an A4 at 200 dpi: off-white paper with grain, a
grey cast, lines of dark "text" strokes, a stamp, quality 80), `odd.png` (37 x 23, saturated colours: odd sizes at
every MCU edge) and `tall.png` (61 x 203). Every input is first brought to what Idrak decodes (`image_transforms.
as_decoded`: EXIF upright, grey or RGB).

For each (input, pipeline) case the manifest names the output's mode and size and the SHA-256 of its bytes in Idrak's
planar order (channels, rows, columns); outputs of 64 K pixels or fewer are also written as PNG (lossless), so a test
can report the largest difference. For the JPEG encoder each (input, quality, subsampling) case keeps Pillow's file
(`save(quality=Q, optimize=True)`) when small, else its SHA-256, and the SHA-256 of its decoded bytes.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import random
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, __version__ as pillow_version, features

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import image_transforms  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA = os.path.join(ROOT, "tests", "Idrak.Tests", "data")
OUT = os.path.join(DATA, "image-transforms")


def scan() -> Image.Image:
    """A scan-like page: paper grain, a slight colour cast, uneven light, text-like strokes, a coloured stamp."""
    rng = np.random.default_rng(20261009)
    w, h = 1654, 2339
    y, x = np.mgrid[0:h, 0:w].astype(np.float32)
    light = 236 - 18 * ((x - w * 0.6) ** 2 + (y - h * 0.4) ** 2) / (w * w + h * h)
    grain = rng.normal(0, 4.0, (h, w)).astype(np.float32)
    page = np.clip(light + grain, 0, 255)
    rgb = np.stack([page * 1.0, page * 0.985, page * 0.94], axis=-1).clip(0, 255).astype(np.uint8)
    image = Image.fromarray(rgb, "RGB")
    draw = ImageDraw.Draw(image)
    r = random.Random(7)
    top = 160
    while top < h - 160:
        left = 140 + r.randint(0, 40)
        while left < w - 160:
            word = r.randint(30, 160)
            ink = (r.randint(20, 60),) * 3
            for _ in range(word // 9):
                x0 = left + r.randint(0, word)
                draw.line([(x0, top + r.randint(0, 8)), (x0 + r.randint(-6, 10), top + 30 + r.randint(-6, 4))], fill=ink, width=r.randint(2, 4))
            draw.line([(left, top + 30), (left + word, top + 30 + r.randint(-2, 2))], fill=ink, width=3)
            left += word + r.randint(20, 45)
        top += 58 + r.randint(-4, 6)
    draw.ellipse([w - 520, h - 560, w - 180, h - 220], outline=(40, 70, 160), width=12)
    draw.text((w - 420, h - 410), "STAMP 2026", fill=(40, 70, 160))
    image = image.filter(ImageFilter.GaussianBlur(0.7))
    return image


def odd() -> Image.Image:
    rng = np.random.default_rng(37)
    a = rng.integers(0, 256, (23, 37, 3), dtype=np.uint8)
    a[:, :12] = [255, 0, 0]
    a[5:9] = [0, 0, 255]
    return Image.fromarray(a, "RGB")


def tall() -> Image.Image:
    y, x = np.mgrid[0:203, 0:61]
    a = np.stack([(x * 4) % 256, (y * 3) % 256, ((x + y) * 5) % 256], axis=-1).astype(np.uint8)
    return Image.fromarray(a, "RGB").filter(ImageFilter.GaussianBlur(1.2))


def planar(image: Image.Image) -> bytes:
    a = np.asarray(image)
    if a.ndim == 3:
        a = a.transpose(2, 0, 1)
    return np.ascontiguousarray(a).tobytes()


def main() -> None:
    os.makedirs(OUT, exist_ok=True)
    made = {"scan.jpg": scan(), "odd.png": odd(), "tall.png": tall()}
    for name, image in made.items():
        path = os.path.join(OUT, name)
        if name.endswith(".jpg"):
            image.save(path, quality=80)
        else:
            image.save(path, optimize=False, compress_level=9)

    inputs = {
        "vlm/image.png": os.path.join(DATA, "vlm", "image.png"),
        "vlm/image-gray.jpg": os.path.join(DATA, "vlm", "image-gray.jpg"),
        "vlm/image-rgba.png": os.path.join(DATA, "vlm", "image-rgba.png"),
        "jpeg/baseline-420-odd.jpg": os.path.join(DATA, "jpeg", "baseline-420-odd.jpg"),
        "jpeg/grey-progressive-odd.jpg": os.path.join(DATA, "jpeg", "grey-progressive-odd.jpg"),
        "jpeg/exif-comment.jpg": os.path.join(DATA, "jpeg", "exif-comment.jpg"),
        "image-transforms/scan.jpg": os.path.join(OUT, "scan.jpg"),
        "image-transforms/odd.png": os.path.join(OUT, "odd.png"),
        "image-transforms/tall.png": os.path.join(OUT, "tall.png"),
    }
    small = ["vlm/image.png", "vlm/image-gray.jpg", "vlm/image-rgba.png", "jpeg/baseline-420-odd.jpg", "jpeg/grey-progressive-odd.jpg",
             "jpeg/exif-comment.jpg", "image-transforms/odd.png", "image-transforms/tall.png"]
    pipelines_small = [
        "grayscale", "contrast=1.5", "contrast=0.5", "contrast=0", "contrast=1", "contrast=3.25", "brightness=1.3", "brightness=0.7",
        "sharpness=2", "sharpness=0.4", "autocontrast", "autocontrast=5", "autocontrast=2,ignore=255", "autocontrast=1,preserve_tone=true",
        "max_width=40", "max_width=40,resample=bicubic", "max_width=33,resample=bilinear", "max_height=20,resample=box",
        "max_height=29,resample=hamming", "max_width=4000", "grayscale,max_width=50,contrast=1.5",
        "contrast=1.5,grayscale", "jpeg=95", "jpeg=75", "jpeg=30", "jpeg=95,subsampling=4:4:4", "jpeg=90,subsampling=4:2:2",
        "grayscale,jpeg=95", "grayscale,max_width=64,contrast=1.5,jpeg=95",
    ]
    pipelines_scan = [
        "grayscale,max_width=1024,contrast=1.5",
        "grayscale,max_width=1024,contrast=1.5,jpeg=95",
        "max_width=1024,resample=bicubic,contrast=1.5",
        "grayscale,max_height=896,resample=bilinear,sharpness=1.5,autocontrast=1",
        "jpeg=95",
        "grayscale,jpeg=95",
    ]
    cases = [(i, p) for i in small for p in pipelines_small] + [("image-transforms/scan.jpg", p) for p in pipelines_scan]

    manifest = {"pillow": pillow_version, "libjpeg_turbo": features.version("libjpeg_turbo"), "transforms": [], "jpeg": []}
    for index, (name, pipeline) in enumerate(cases):
        image = image_transforms.as_decoded(Image.open(inputs[name]))
        out = image_transforms.apply(image, pipeline)
        assert out.mode in ("L", "RGB"), (name, pipeline, out.mode)
        entry = {"input": name, "pipeline": pipeline, "mode": out.mode, "width": out.width, "height": out.height,
                 "sha256": hashlib.sha256(planar(out)).hexdigest()}
        if out.width * out.height <= 65536:
            file = f"out-{index:03d}.png"
            out.save(os.path.join(OUT, file), optimize=False, compress_level=9)
            entry["png"] = file
        manifest["transforms"].append(entry)

    jpeg_cases = [(i, q, s) for i in small for q, s in [(95, "4:2:0"), (75, "4:2:0"), (10, "4:2:0"), (100, "4:2:0"), (90, "4:4:4"), (85, "4:2:2")]]
    jpeg_cases += [("image-transforms/scan.jpg", 95, "4:2:0"), ("image-transforms/scan.jpg", 80, "4:4:4")]
    jpeg_cases += [("image-transforms/scan.jpg#L", 95, "4:2:0")]
    sampling = {"4:2:0": 2, "4:2:2": 1, "4:4:4": 0}
    for index, (name, quality, sub) in enumerate(jpeg_cases):
        grey = name.endswith("#L")
        image = image_transforms.as_decoded(Image.open(inputs[name.removesuffix("#L")]))
        if grey:
            image = image.convert("L")
        buffer = io.BytesIO()
        extra = {} if image.mode == "L" else {"subsampling": sampling[sub]}   # grey: Pillow's default (one 1 x 1 component)
        image.save(buffer, format="JPEG", quality=quality, optimize=True, **extra)
        data = buffer.getvalue()
        decoded = Image.open(io.BytesIO(data))
        decoded.load()
        entry = {"input": name, "quality": quality, "subsampling": sub, "mode": image.mode, "bytes": len(data),
                 "sha256": hashlib.sha256(data).hexdigest(), "decoded_sha256": hashlib.sha256(planar(decoded)).hexdigest()}
        if len(data) <= 8192:
            file = f"pillow-{index:03d}.jpg"
            with open(os.path.join(OUT, file), "wb") as f:
                f.write(data)
            entry["file"] = file
        manifest["jpeg"].append(entry)

    with open(os.path.join(OUT, "manifest.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump(manifest, f, indent=1)
        f.write("\n")
    print(f"{len(manifest['transforms'])} transform cases, {len(manifest['jpeg'])} JPEG cases in {OUT}")


if __name__ == "__main__":
    main()
