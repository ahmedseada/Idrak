"""JPEG and image preprocessing fixtures for Idrak (needs: pip install pillow numpy opencv-python-headless transformers).

    python tools/jpeg/make_fixtures.py tests/Idrak.Tests/data/jpeg

Every NAME.jpg comes with NAME.ppm (or NAME.pgm for grey) holding the pixels Pillow decodes it to (libjpeg-turbo with
its defaults: the islow integer IDCT, fancy upsampling), which Idrak's JPEG codec is compared with. The files are made
by Pillow (baseline 4:4:4, 4:2:2 and 4:2:0, progressive, grey, restart intervals, RGB with Adobe's transform flag 0,
EXIF and comment segments, CMYK for the rejection test) and OpenCV (4:4:0 and 4:1:1, which Pillow does not write).

The preprocessing cases (preprocess.json) resize, rescale and normalize source.png and small.png as transformers'
Gemma3ImageProcessorPil does (Pillow's resize, then float32 arithmetic); each case's pixel_values ([3, H, W] float32,
little-endian) is in NAME.f32. A grayscale case first converts with Pillow's "L" and back to "RGB".
Rerunning the script writes the same bytes.
"""
import io
import json
import os
import sys

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
        return                                                                       # only for the rejection test
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
    }
    bgr = np.ascontiguousarray(scene(45, 37, 1)[:, :, ::-1])
    for name, factor in (("opencv-440", cv2.IMWRITE_JPEG_SAMPLING_FACTOR_440), ("opencv-411", cv2.IMWRITE_JPEG_SAMPLING_FACTOR_411)):
        ok, encoded = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 85, cv2.IMWRITE_JPEG_SAMPLING_FACTOR, factor])
        assert ok
        cases[name] = encoded.tobytes()
    for name, data in cases.items():
        write_case(folder, name, data)


def preprocessing(folder):
    source = Image.fromarray(scene(83, 61, 4))                                       # 83 wide, 61 high
    small = Image.fromarray(scene(13, 11, 5))
    source.save(os.path.join(folder, "source.png"))
    small.save(os.path.join(folder, "small.png"))
    images = {"source.png": source, "small.png": small}
    cases = [
        ("down-bilinear", "source.png", 24, 32, 2, False),
        ("down-bicubic", "source.png", 24, 32, 3, False),
        ("down-bilinear-odd", "source.png", 17, 13, 2, False),
        ("down-lanczos", "source.png", 20, 20, 1, False),
        ("up-bilinear", "small.png", 29, 31, 2, False),
        ("up-bicubic", "small.png", 29, 31, 3, False),
        ("mixed-bicubic", "source.png", 64, 20, 3, False),                         # taller, narrower
        ("grey-bilinear", "source.png", 24, 32, 2, True),
    ]
    listed = []
    for name, image, height, width, resample, grey in cases:
        config = {
            "image_processor_type": "Gemma3ImageProcessor",
            "do_convert_rgb": True, "do_resize": True, "size": {"height": height, "width": width}, "resample": resample,
            "do_rescale": True, "rescale_factor": 0.00392156862745098, "do_normalize": True,
            "image_mean": [0.5, 0.5, 0.5], "image_std": [0.5, 0.5, 0.5], "do_pan_and_scan": None,
        }
        if name == "down-bicubic":                                                  # CLIP's mean and std, to check per channel
            config["image_mean"] = [0.48145466, 0.4578275, 0.40821073]
            config["image_std"] = [0.26862954, 0.26130258, 0.27577711]
        processor = Gemma3ImageProcessorPil(**{k: v for k, v in config.items() if k != "image_processor_type"})
        picture = images[image].convert("L").convert("RGB") if grey else images[image]
        values = processor(picture, return_tensors="np")["pixel_values"][0].astype("<f4")
        assert values.shape == (3, height, width), values.shape
        values.tofile(os.path.join(folder, name + ".f32"))
        listed.append({"name": name, "image": image, "grayscale": grey, "config": config})
    with open(os.path.join(folder, "preprocess.json"), "w", newline="\n") as f:
        json.dump(listed, f, indent=1)
        f.write("\n")


def main():
    folder = sys.argv[1] if len(sys.argv) > 1 else "tests/Idrak.Tests/data/jpeg"
    os.makedirs(folder, exist_ok=True)
    jpegs(folder)
    preprocessing(folder)


if __name__ == "__main__":
    main()
