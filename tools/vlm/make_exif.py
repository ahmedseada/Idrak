# Copyright (c) 2026 Ahmed Seada
# Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.
"""EXIF orientation fixtures for plan 11, phase 7 (the command line turns images upright as transformers' load_image does).

Writes into the given folder (tests/Idrak.Tests/data/vlm/exif):
  exif-N.jpg       a 7 x 5 colour image as a JPEG whose EXIF orientation (tag 0x0112) is N, for N = 1 to 8
  exif-N.png       the same image as a PNG with an eXIf chunk, for N = 3, 6 and 8 (lossless)
  upright-N.jpg.png / upright-N.png.png
                   what Pillow's ImageOps.exif_transpose gives for each file (then convert("RGB")), saved as PNG
                   without EXIF, so a decoder that matches Pillow's bytes must give exactly these pixels

Needs Pillow only (pip install pillow). Reruns write the same bytes for the same Pillow and libjpeg-turbo versions.

    python tools/vlm/make_exif.py tests/Idrak.Tests/data/vlm/exif
"""

import os
import sys

from PIL import Image, ImageOps


def base():
    image = Image.new("RGB", (7, 5))
    for y in range(5):
        for x in range(7):
            image.putpixel((x, y), ((x * 37 + y * 11) % 256, (y * 53 + 20) % 256, (x * y * 29 + 90) % 256))
    image.putpixel((0, 0), (255, 0, 0))     # a red top-left corner, a blue top-right one
    image.putpixel((6, 0), (0, 0, 255))
    return image


def main(folder):
    os.makedirs(folder, exist_ok=True)
    image = base()
    for n in range(1, 9):
        exif = Image.Exif()
        exif[0x0112] = n
        jpg = os.path.join(folder, f"exif-{n}.jpg")
        image.save(jpg, quality=95, subsampling=0, exif=exif.tobytes())
        ImageOps.exif_transpose(Image.open(jpg)).convert("RGB").save(os.path.join(folder, f"upright-{n}.jpg.png"))
        if n in (3, 6, 8):
            png = os.path.join(folder, f"exif-{n}.png")
            image.save(png, exif=exif.tobytes())
            ImageOps.exif_transpose(Image.open(png)).convert("RGB").save(os.path.join(folder, f"upright-{n}.png.png"))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "tests/Idrak.Tests/data/vlm/exif")
