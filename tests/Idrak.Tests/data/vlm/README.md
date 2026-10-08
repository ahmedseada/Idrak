# A tiny Gemma 3 vision-language reference (plan 11, phase 0)

Written by `tools/vlm/make_tiny.py`. These files are the reference that plan 11's phases are checked against. The model
is a random `Gemma3ForConditionalGeneration` built from configs (nothing comes from huggingface.co), stored in float32.

- Vision: SigLIP, a 56 x 56 image, 14 x 14 patches (4 x 4 = 16 patches), width 8, 2 heads, MLP 16, 2 layers, no head.
- Projector: 2 x 2 average pooling to 4 soft tokens, then Gemma's RMSNorm (1 + w) and x · W, with W stored as [8, 24].
- Text: width 24, 4 query heads and 2 key/value heads of 8 (so heads x head_dim = 32, not 24), MLP 48, 2 layers
  (layer 0 sliding with a window of 8, layer 1 global), `query_pre_attn_scalar` 12 (not head_dim), RoPE theta 10,000
  local and 1,000,000 global with linear scaling 8 on the global layer only (as in the 4B), tied embeddings, vocabulary 366.
- Token ids: `<pad>` 0, `<eos>` 1, `<bos>` 2, `<unk>` 3, `<mask>` 4, `<start_of_turn>` 5, `<end_of_turn>` 6,
  `<start_of_image>` 7 (`boi_token_index`), `<end_of_image>` 8 (`eoi_token_index`), `<image_soft_token>` 365
  (`image_token_index`, the last id, as in the real vocabulary).

## Files

| File | What it is |
|---|---|
| `image.png` | The test image: 100 x 75 RGB (not square, not a multiple of 8 or 16), colorful: a saturated hue sweep across, saturation and brightness falling down, black, white and colored strokes, a magenta circle, a pure red and a pure blue box |
| `image-rgba.png` | The same with alpha: 255 on the left falling to 0 on the right, and a fully transparent box (rows 20-39, columns 30-59) whose hidden color is pure green |
| `image-palette.png` | The same as a palette PNG (mode P, 16 colors by median cut, no dithering) |
| `image.jpg` | The same image as a baseline JPEG, YCbCr 4:2:0, quality 85 (libjpeg-turbo through Pillow) |
| `image-progressive.jpg` | Progressive JPEG, 4:2:0, quality 85 (PIL decodes it to the same pixels as `image.jpg`) |
| `image-restart.jpg` | Baseline 4:2:0 with a restart marker every 5 MCUs (same pixels as `image.jpg`) |
| `image-444.jpg` | Baseline YCbCr 4:4:4, quality 90 |
| `image-gray.jpg` | Baseline, one grayscale component |
| `tiny-gemma3-pre452/` | The model as transformers 4.51.3 `save_pretrained` writes it; 4.52.4 and 4.57.6 write the same tensor bytes (they reverse their key mapping on save): `language_model.model.*`, `vision_tower.vision_model.*`, `multi_modal_projector.*`, no `lm_head` |
| `tiny-gemma3/` | The in-memory layout of transformers 4.52 to 4.57 (`model.language_model.*`, `model.vision_tower.vision_model.*`, `model.multi_modal_projector.*`), as tools that save a raw state dict write it. No transformers version's `save_pretrained` writes it: the script renames the tensors; its `config.json` is in the real gemma-3-4b-it's format (`rope_theta`, `rope_local_base_freq`, `rope_scaling`) |
| `tiny-gemma3-v5/` | The model as transformers 5.19.0 `save_pretrained` writes it: `language_model.model.*`, `vision_tower.*` **without** `vision_model.`, `multi_modal_projector.*`; its `config.json` has `rope_parameters` per layer type and `dtype` |
| each model folder | `config.json`, `model.safetensors`, `tokenizer.json` (BPE over `▁`-joined text with byte fallback, Gemma-style), `tokenizer_config.json` (with Gemma 3's chat template, image parts included), `special_tokens_map.json`, `preprocessor_config.json` (resize to 56 x 56, bilinear, rescale 1/255, mean and std 0.5), `processor_config.json` (`image_seq_length` 4) |
| `reference/decoded-image.npy` | uint8 [75, 100, 3]: `image.jpg` (and the progressive and restart files) as Pillow decodes it |
| `reference/decoded-image-444.npy`, `decoded-image-gray.npy` | The same for the 4:4:4 file and the grayscale file ([75, 100]) |
| `reference/pixel_values-png.npy` | float32 [1, 3, 56, 56]: the image processor's output for `image.png` |
| `reference/pixel_values-jpg.npy` | The same for `image.jpg` as Pillow decodes it |
| `reference/pixel_values-png-gray.npy` | The same for `image.png` through the fine-tune's grayscale step first (Pillow's `convert("L")`, L = (19595 R + 38470 G + 7471 B + 32768) >> 16; the processor's `convert_to_rgb` then copies it to three equal channels) |
| `reference/pixel_values-rgba.npy` | The same for `image-rgba.png` opened as RGBA: transformers' `convert_to_rgb` is PIL's `convert("RGB")`, which **drops alpha** and keeps the hidden colors (the clear box stays green); it does not composite onto white (that would differ by up to 2.0 here) |
| `reference/pixel_values-palette.npy` | The same for `image-palette.png` opened in mode P (the palette looked up, then as RGB) |
| `reference/vision_embeddings.npy` | float32 [1, 16, 8]: patch embedding plus position embedding (before the encoder layers) |
| `reference/vision_last_hidden_state.npy` | float32 [1, 16, 8]: the vision tower's output (after `post_layernorm`) |
| `reference/image_features.npy` | float32 [1, 4, 24]: the projector's output (the soft tokens' embeddings) |
| `reference/prompt-image-input_ids.npy` | int64 [1, 38]: the processor's token ids for a system line, one image and a question, with the generation prompt |
| `reference/prompt-image-logits.npy` | float32 [1, 38, 366]: the logits of that prompt (with `token_type_ids`, so the image block is bidirectional) |
| `reference/prompt-image-generate-logits.npy` | float32 [1, 20, 366]: the logits of each of the 20 greedy steps (`generate`, KV cache) |
| `reference/prompt-text-input_ids.npy`, `prompt-text-logits.npy` | A text-only chat prompt (39 tokens, longer than the window) and its logits |
| `exif/` | EXIF orientation fixtures (plan 11, phase 7; `tools/vlm/make_exif.py`): a 7 x 5 image as `exif-N.jpg` with orientation N = 1 to 8 and `exif-N.png` (eXIf chunk) for N = 3, 6, 8, and `upright-N.<ext>.png`, what Pillow's `ImageOps.exif_transpose` gives for each |
| `compare/color`, `compare/gray` | `tools/vlm/compare_real.py`'s folders for the tiny model (`idrak vlm check`'s reference format): `image.png` with the system line and question of `prompt-image-*` (float32, 20 steps), and `image.jpg` with `--grayscale` and no system line; written from tiny-gemma3-v5 by transformers 5.19.0, then each manifest's absolute `image` path replaced by `../../image.png` (`../../image.jpg`), which `vlm check` reads relative to the folder |
| `manifest.json` | Versions, every file's size and SHA-256, each layout's tensor names, shapes and writer with its checks, array shapes, and the facts below |

`manifest.json` → `facts` holds: the prompts (messages, the chat template's text, the processor's expanded text, ids,
tokens, `token_type_ids`), the 20 greedy tokens (`generate.new_tokens`, checked against one pass without a cache), the
attention mask transformers builds for the image prompt as one `[first, last]` key range per query row for each layer
kind (`image_prompt.mask_rows_first_last_key`), how much the logits change without `token_type_ids`, the RoPE inverse
frequencies per layer kind, the embedding scale, the attention scale, the projector's pooling sizes, and the
preprocessing checks (`pixels_check`: each pixel file recomputed by hand, bit for bit).

## Regenerate

```
python -m venv vlm && vlm/bin/pip install torch transformers tokenizers safetensors pillow numpy jinja2
for v in 4.51.3 4.52.4 4.57.6; do
    python -m venv vlm-$v && vlm-$v/bin/pip install "transformers==$v" safetensors pillow numpy jinja2
    echo "$PWD/vlm/lib/python3.13/site-packages" > vlm-$v/lib/python3.13/site-packages/zz_torch.pth
done
vlm/bin/python tools/vlm/make_tiny.py tests/Idrak.Tests/data/vlm \
    --legacy-python vlm-4.51.3/bin/python --legacy-python vlm-4.52.4/bin/python --legacy-python vlm-4.57.6/bin/python
```

Built with Python 3.13.16, torch 2.14.1 (CPU), transformers 5.19.0 (and 4.51.3, 4.52.4, 4.57.6 for the older layout
and its checks), tokenizers 0.23.2, safetensors 0.8.0, Pillow 12.3.0, numpy 2.5.3 (the exact list, with libjpeg-turbo's
version, is in `manifest.json`). Two runs write the same bytes; another machine or other library versions may differ in
the last bits of the floats, so tests compare with a tolerance (1e-5 for pixels, about 1e-4 for logits) and the token
ids and greedy tokens exactly.
