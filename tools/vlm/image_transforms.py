"""Idrak's image transforms, done in Pillow: the reference the library's `ImageTransforms` match byte for byte.

A pipeline is written as Idrak writes it: comma-separated steps, `grayscale,max_width=1024,contrast=1.5`; an item
naming an option of the step before it belongs to that step (`max_width=1024,resample=bicubic`). The transforms:

  grayscale                  image.convert("L")
  max_width=N [resample=R]   wider than N: resize to (N, int(h * (N / float(w)))) with R (lanczos unless given)
  max_height=N [resample=R]  taller than N: resize to (int(w * (N / float(h))), N)
  contrast=F                 ImageEnhance.Contrast(image).enhance(F)
  brightness=F               ImageEnhance.Brightness(image).enhance(F)
  sharpness=F                ImageEnhance.Sharpness(image).enhance(F)
  autocontrast[=C] [ignore=V] [preserve_tone=true]   ImageOps.autocontrast(image, C, V, preserve_tone)
  jpeg=Q [subsampling=S]     save as JPEG (quality Q, optimize=True), open again

Used by compare_real.py (`--image-transform`), ocr_transformers.py and make_image_transforms.py. Needs Pillow only.
"""

from __future__ import annotations

import io

from PIL import Image, ImageEnhance, ImageOps

RESAMPLE = {
    "lanczos": Image.Resampling.LANCZOS, "antialias": Image.Resampling.LANCZOS, "1": Image.Resampling.LANCZOS,
    "bilinear": Image.Resampling.BILINEAR, "linear": Image.Resampling.BILINEAR, "2": Image.Resampling.BILINEAR,
    "bicubic": Image.Resampling.BICUBIC, "cubic": Image.Resampling.BICUBIC, "3": Image.Resampling.BICUBIC,
    "box": Image.Resampling.BOX, "4": Image.Resampling.BOX,
    "hamming": Image.Resampling.HAMMING, "5": Image.Resampling.HAMMING,
}

# Each transform's option keys (besides its value).
KEYS = {
    "grayscale": [], "max_width": ["resample"], "max_height": ["resample"], "contrast": [], "brightness": [],
    "sharpness": [], "autocontrast": ["ignore", "preserve_tone"], "jpeg": ["subsampling"], "invert": [],
}


def parse(text: str | None) -> list[tuple[str, str | None, dict[str, str]]]:
    """The steps of a pipeline's text: (name, value or None, options)."""
    steps: list[tuple[str, str | None, dict[str, str]]] = []
    if not text or text.strip().lower() == "none":
        return steps
    for raw in text.split(","):
        item = raw.strip()
        if not item:
            continue
        name, _, value = item.partition("=")
        name = name.strip().lower()
        value = value.strip() if "=" in item else None
        if name in KEYS:
            steps.append((name, value, {}))
        elif steps and value is not None and name in KEYS[steps[-1][0]]:
            steps[-1][2][name] = value
        else:
            raise ValueError(f"unknown image transform '{name}' (known: {', '.join(KEYS)})")
    return steps


def as_decoded(image: Image.Image) -> Image.Image:
    """The image as Idrak decodes it: upright by its EXIF orientation, grey ('L') or colour ('RGB'), alpha dropped."""
    image = ImageOps.exif_transpose(image)
    image = image.convert("L") if image.mode in ("L", "LA", "1") else image.convert("RGB")
    image.info = {}                                     # no comment or profile carried into a saved JPEG
    return image


def apply(image: Image.Image, text: str | None) -> Image.Image:
    """The image after each step of the pipeline, in order."""
    for name, value, options in parse(text):
        if name == "grayscale":
            image = image.convert("L")
        elif name in ("max_width", "max_height"):
            n = int(value)
            resample = RESAMPLE[options.get("resample", "lanczos").lower()]
            w, h = image.size
            if name == "max_width" and w > n:
                ratio = n / float(w)
                image = image.resize((n, max(1, int(h * ratio))), resample)
            elif name == "max_height" and h > n:
                ratio = n / float(h)
                image = image.resize((max(1, int(w * ratio)), n), resample)
        elif name == "contrast":
            image = ImageEnhance.Contrast(image).enhance(float(value))
        elif name == "brightness":
            image = ImageEnhance.Brightness(image).enhance(float(value))
        elif name == "sharpness":
            image = ImageEnhance.Sharpness(image).enhance(float(value))
        elif name == "autocontrast":
            cutoff = float(value) if value else 0
            ignore = int(options["ignore"]) if "ignore" in options else None
            tone = options.get("preserve_tone", "false").lower() in ("true", "1", "yes", "on")
            image = ImageOps.autocontrast(image, cutoff=cutoff, ignore=ignore, preserve_tone=tone)
        elif name == "invert":
            image = ImageOps.invert(image)
        elif name == "jpeg":
            buffer = io.BytesIO()
            sampling = {"4:2:0": 2, "420": 2, "2": 2, "4:2:2": 1, "422": 1, "1": 1, "4:4:4": 0, "444": 0, "0": 0}
            extra = {"subsampling": sampling[options["subsampling"]]} if "subsampling" in options and image.mode != "L" else {}
            image.save(buffer, format="JPEG", quality=int(value), optimize=True, **extra)
            image = Image.open(io.BytesIO(buffer.getvalue()))
            image.load()
    return image


def describe(text: str | None) -> str:
    """The pipeline written back in Idrak's form (lower-case names, options after their step)."""
    parts = []
    for name, value, options in parse(text):
        parts.append(name if value is None else f"{name}={value}")
        parts.extend(f"{k}={v}" for k, v in sorted(options.items()))
    return ",".join(parts)
