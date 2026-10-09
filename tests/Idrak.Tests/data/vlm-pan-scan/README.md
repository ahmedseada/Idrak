# Gemma 3's pan and scan on the tiny model (plan 11, "Pan and scan")

Written by `tools/vlm/make_pan_scan.py` from the tiny Gemma 3 of `../vlm` (its `tiny-gemma3-v5` folder: weights,
tokenizer, chat template, processor config; nothing downloaded), with transformers 5.19.0's `Gemma3Processor` and
`Gemma3ImageProcessorPil`, `do_pan_and_scan=True` and `pan_and_scan_min_crop_size=32` (the tiny model reads 56-pixel
images; the other settings are Gemma3Processor's defaults: at most 4 crops, an aspect ratio of 1.2 at least).
`--check` with transformers 4.57.6 (and the 4.x layout of the same weights) gives the same crops, ids, pixels and logits
(differences 0).

| File | What it is |
|---|---|
| `image-tall.png` | 60 x 170 RGB (ratio 2.83): 3 crops of 60 x 57, the last 60 x 56 |
| `image-wide.png` | 300 x 60 RGB (ratio 5): 5 crops wanted, 4 at most, 75 x 60 each |
| `image-square.png` | 90 x 80 RGB (ratio 1.125, below 1.2): no crops; its ids, pixels and logits equal pan and scan off |
| `reference/pixel_values-NAME.npy` | float32 [1 + crops, 3, 56, 56]: the whole image, then each crop (cut from the original, then resized) |
| `reference/image_features-NAME.npy` | float32 [1 + crops, 4, 24]: the projector's output per block |
| `reference/prompt-NAME-input_ids.npy` | int64 [1, L]: "Read the scan." (system), the image, "What is in this image?", as the processor expands it ("Here is the original image ... and here are some crops to help you see better ...") |
| `reference/prompt-NAME-logits.npy` | float32 [1, L, 366]: the prompt's logits (token_type_ids given: each crop's block bidirectional) |
| `reference/prompt-NAME-generate-logits.npy` | float32 [1, 20, 366]: the logits of 20 greedy steps (`generate`, KV cache) |
| `compare/` | `tools/vlm/compare_real.py` with `--pan-and-scan --pan-and-scan-min-crop-size 32` on `image-tall.png` (float32, CPU, 20 steps), the manifest's image path made relative (`../image-tall.png`): `idrak vlm check`'s reference format |
| `manifest.json` | versions, the model files' SHA-256, every file's size and SHA-256, and per image: the crops' rectangles, the rendered and expanded prompt, ids, tokens, token_type_ids, each block's first position, the 20 greedy tokens |

The tiny model's greedy answer settles on one token after the longer pan-and-scan prompts (tall and wide: token 91
twenty times); the step logits are what the tests compare closely. Reruns write the same bytes.

```
vlm/bin/python tools/vlm/make_pan_scan.py tests/Idrak.Tests/data/vlm-pan-scan
vlm-4.57.6/bin/python tools/vlm/make_pan_scan.py --check tests/Idrak.Tests/data/vlm-pan-scan
```
