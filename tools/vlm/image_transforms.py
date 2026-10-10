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

from PIL import Image, ImageEnhance, ImageFilter, ImageOps

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
    "scale": ["resample"], "pad": ["fill"], "equalize": [], "gamma": [], "blur": [], "unsharp": ["percent", "threshold"],
    "median": [], "min_filter": [], "max_filter": [], "binarize": [],
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


def otsu(grey: Image.Image) -> int:
    """Otsu's level: the t maximizing the between-class variance of [0, t] and (t, 255] (the first on a tie)."""
    hist = grey.histogram()
    total = sum(hist)
    sum_all = sum(i * h for i, h in enumerate(hist))
    weight_back = sum_back = 0
    best, level = -1.0, 0
    for t in range(256):
        weight_back += hist[t]
        if weight_back == 0:
            continue
        weight_fore = total - weight_back
        if weight_fore == 0:
            break
        sum_back += t * hist[t]
        mean_back = sum_back / weight_back
        mean_fore = (sum_all - sum_back) / weight_fore
        difference = mean_back - mean_fore
        between = float(weight_back) * weight_fore * (difference * difference)
        if between > best:
            best, level = between, t
    return level


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
        elif name == "scale":
            f = float(value)
            resample = RESAMPLE[options.get("resample", "lanczos").lower()]
            w, h = image.size
            image = image.resize((max(1, int(w * f + 0.5)), max(1, int(h * f + 0.5))), resample)
        elif name == "pad":
            fill = int(options.get("fill", "255"))
            image = ImageOps.expand(image, border=int(value), fill=fill if image.mode == "L" else (fill,) * len(image.getbands()))
        elif name == "equalize":
            image = ImageOps.equalize(image)
        elif name == "gamma":
            g = float(value)
            table = [int(((i / 255.0) ** (1.0 / g)) * 255 + 0.5) for i in range(256)]
            image = image.point(table * len(image.getbands()))
        elif name == "blur":
            image = image.filter(ImageFilter.GaussianBlur(float(value)))
        elif name == "unsharp":
            radius = float(value) if value else 2.0
            image = image.filter(ImageFilter.UnsharpMask(radius, int(options.get("percent", "150")), int(options.get("threshold", "3"))))
        elif name == "median":
            image = image.filter(ImageFilter.MedianFilter(int(value) if value else 3))
        elif name == "min_filter":
            image = image.filter(ImageFilter.MinFilter(int(value) if value else 3))
        elif name == "max_filter":
            image = image.filter(ImageFilter.MaxFilter(int(value) if value else 3))
        elif name == "binarize":
            grey = image.convert("L")
            t = int(value) if value else otsu(grey) + 1
            image = grey.point([255 if i >= t else 0 for i in range(256)])
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
