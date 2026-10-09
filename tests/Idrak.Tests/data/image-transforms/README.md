# Image transform fixtures

Written by `tools/vlm/make_image_transforms.py` (Pillow 12.3.0, libjpeg-turbo 3.1.4.1, numpy 2.5); reruns write the
same bytes. Checked by `IDRAK_FILTER="image transforms"`.

| File | What |
|---|---|
| `scan.jpg` | a scan-like page made by the script: 1,654 x 2,339 (A4 at 200 dpi), off-white paper with grain and uneven light, a slight colour cast, lines of dark strokes, a blue stamp, JPEG quality 80 |
| `odd.png`, `tall.png` | 37 x 23 (saturated colours, odd sizes at every JPEG MCU edge) and 61 x 203 |
| `manifest.json` | `transforms`: 238 cases (an input, a pipeline in Idrak's syntax, the output's mode and size, the SHA-256 of its bytes in planar order, and `png` for outputs of 64 K pixels or fewer); `jpeg`: 51 encoder cases (input, quality, subsampling, the SHA-256 of Pillow's file and of its decoded bytes, and `file` for small ones) |
| `out-NNN.png` | Pillow's result of transform case NNN (lossless), for reporting the largest difference |
| `pillow-NNN.jpg` | Pillow's `save(quality=Q, optimize=True)` file of JPEG case NNN |

The inputs are also the fixtures of phase 0 (`../vlm/image.png`, `image-gray.jpg`, `image-rgba.png`) and of the JPEG
decoder (`../jpeg/baseline-420-odd.jpg`, `grey-progressive-odd.jpg`, `exif-comment.jpg`), each first brought to what
Idrak decodes (EXIF upright, grey or RGB, alpha dropped; `image_transforms.as_decoded`).
