# Plan 16: Qwen-VL (Qwen2.5-VL and Qwen2-VL) as a base model

The owner's request (October 2026): use Qwen as a base model in the legal OCR app, for reading and for fine-tuning.
Qwen's text models (Qwen2, Qwen3) already load; a base for OCR must read images, so this plan adds the **Qwen-VL
vision-language families**: Qwen2.5-VL first (3B and 7B Instruct), Qwen2-VL on the same work (its 2B is the base of
several Arabic OCR fine-tunes, to check in step 0). Qwen3-VL (DeepStack features injected into several decoder layers,
interleaved M-RoPE, another tower) is a later plan.

## What is there already

Plan 11 left the slots for this family, and plans 12 and 14 built most of the machinery it needs:

- `ImageFeatures.Positions` ([axes, tokens]: an image's position ids on each axis of a multi-axis rotary embedding) is
  in the contract; `ImagePrefill` carries it to the decoder and refuses it (`CheckPositions`), and `TuningVision` refuses
  a family that gives it. Those two refusals are what this plan replaces.
- `ImageTokenLayout.Grid` holds a dynamic-resolution image's token grid ([frames, rows, columns]); a count per image is
  in the contract, and the tuner, the feature cache and the chat path already take a different count per image (pan
  and scan made them do it).
- `Tensor.AttentionSpans` (and its backward, plan 12 phase 3) runs attention over arbitrary key spans, tiled over the
  keys with the scores never stored, on every device: Qwen2.5-VL's window attention and its full-attention layers are
  `KeySpans` tables (one span per window, one per image), and the tower's head size (1280 / 16 = 80) has kernels.
- The decoder takes rotary positions apart from the cache index (`DecodingContext.Positions` against `Position`), so a
  sequence whose positions run ahead of its cache index is a matter of what goes into `Positions`.
- `PillowImageOps.Resize` (Pillow's resampling, both directions) for `smart_resize`; the Qwen2 BPE tokenizer and the Qwen2
  decoder (`QkvBias`) load today.

## What Qwen-VL does (transformers, to be read exactly in step 0)

- **Processor.** `smart_resize`: height and width rounded to multiples of 28 (patch 14 x merge 2), the pixel count kept
  between `min_pixels` (56 x 56) and `max_pixels`, bicubic resize; rescale by 1/255, normalize with CLIP's mean and std;
  the image repeated to 2 frames (temporal patch 2) and cut into 14 x 14 patches, flattened to 3 x 2 x 14 x 14 = 1,176
  values each, ordered so each 2 x 2 group of patches is contiguous (the merger's order). `image_grid_thw` = [1, h / 14,
  w / 14]. A 1,024 x 1,448 page becomes about 7,700 patches and 1,900 image tokens: the image's own resolution, not
  Gemma 3's fixed 256 tokens, which is why it suits dense legal pages.
- **Tower (Qwen2.5-VL).** Patch embedding (a Conv3d whose kernel is its stride: a matrix product over the 1,176 values),
  2-D rotary positions (half the rotary dims by row, half by column), 32 blocks of RMSNorm, attention and a SwiGLU MLP
  with biases; attention within windows of 112 pixels (8 x 8 patches) except in the full-attention blocks
  (`fullatt_block_indexes`, e.g. 7, 15, 23, 31), the patches reordered by window first (`window_index`) and back after the
  merger. **Merger**: RMSNorm, the 2 x 2 neighbours concatenated (1280 x 4), Linear, GELU, Linear to the decoder's width.
  **Qwen2-VL** differs: LayerNorm, a quick-GELU MLP (fc1, fc2), full attention in every block, a LayerNorm merger.
- **Prompt.** The chat template writes `<|vision_start|><|image_pad|><|vision_end|>` for an image; the processor repeats
  `<|image_pad|>` once per merged token (grid_h x grid_w / 4). Image tokens attend causally (no bidirectional block, unlike
  Gemma 3).
- **M-RoPE in the decoder.** Each token has three positions (time, height, width); the rotary half-dimension is split in
  sections (`rope_scaling.mrope_section`, [16, 24, 24] for a head of 128), each section rotated by its own axis. Text
  tokens have (p, p, p); an image starting at position s has (s, s + row, s + column) over its merged grid; the text after
  it continues from the largest position so far plus one. So a sequence's position is no longer its cache index:
  `rope_deltas` (the largest position + 1 minus the length) is added to every later token, cached prefill and decode
  steps alike.

## Where each part goes (decision 10, no family code in the library)

| Part | Where | Why |
|---|---|---|
| Multi-axis rotary positions: `RopeSettings.Sections`, the rotary rows of a sequence from its [axes, tokens] positions, the per-sequence offset in `DecodingContext`, packed training sequences with axes | **Library** (`Idrak.Abstraction`, `Idrak`) | Generic: any model with sectioned rotary positions (Qwen2-VL, Qwen2.5-VL, others) |
| `ImagePositionRules` (contract and registry): how image tokens are placed: `sequential` (default: the next positions, Gemma 3, LLaVA) and `grid` (M-RoPE from `ImageTokenLayout.Grid`: (s, s + row, s + column), then max + 1) | **Library** (`Idrak.Abstraction.Generation`, beside `ImageAttentionRules`) | It depends only on the layout's grid, not on a family; a family names the rule it uses |
| `mrope` in `RopeScalings` (Qwen writes `"type": "mrope"` or `"default"` with `mrope_section`) | **Library** | The config key is read as sections, the frequencies unscaled |
| The text decoder of `Qwen2_5_VLForConditionalGeneration` and `Qwen2VLForConditionalGeneration` (Qwen2 style, sections from `mrope_section`, its tensor names old and new) | **Library** `Architectures` table | As `Gemma3ForConditionalGeneration`'s text decoder is: a text family; the vision part stays unregistered without the plug-in |
| Processor (`smart_resize`, patches, merge order), tower, merger, prompt format, attention rule name, position rule name, vision options (`min_pixels`, `max_pixels`), tuning parts (`projector` = merger, `tower`) | **Plug-in** `samples/QwenVision/Idrak.QwenVision` | The family, as `Idrak.Gemma3Vision` is |
| Qwen models in the switch, a "max pixels" setting, the plug-in registered | **App** `samples/Idrak.Samples.LegalOcr` | The app's choice of models |

## Steps

| # | Step | Where | Done when |
|---|---|---|---|
| 0 | **Reference fixtures.** `tools/vlm/make_tiny_qwen_vl.py`: tiny random Qwen2.5-VL and Qwen2-VL checkpoints (a few blocks, a window and a full-attention block, `mrope_section` scaled down, the real tokenizer's ids for the special tokens); for a colour image, a grey one, a tall page and a small one (upscaled by `smart_resize`): the processor's `pixel_values`, `image_grid_thw`, the expanded `input_ids`, `get_rope_index`'s position ids and `rope_deltas`, the tower's output after the patch embedding, a window block, a full block and the merger, the prompt's logits in one pass and cached, 20 greedy tokens; a two-image prompt; a tuning fixture (loss and the adapters' gradients on two conversations, as `make_tiny_tuning.py`). Read transformers' processor and model code for the exact rules (and check which Arabic OCR fine-tunes use which family) | `tools/vlm`, `tests/Idrak.Tests/data/vlm/tiny-qwen2_5-vl`, `tiny-qwen2-vl` | Fixtures written; reruns give the same bytes |
| 1 | **M-RoPE in the library.** `RopeSettings.Sections`; one op that turns a sequence's [axes, tokens] positions into its rotary rows (cos and sin per token, each section from its axis's table), made once per forward pass and shared by every layer, then the existing rotary kernels with those rows (no new attention kernel); the CPU kernel and the GPU ones (CUDA, Vulkan, HIP) under rules 80-87; `DecodingContext` keeps each sequence's offset (decode steps take cache index + offset, one position for all axes); `ImagePrefill` places text and images through the family's position rule (the `CheckPositions` refusal goes); `ImagePositionRules` with `sequential` and `grid`; `mrope` in `RopeScalings` | `Idrak.Abstraction`, `Idrak`, `Idrak.Gpu` | A text model with sections and equal axes gives the plain model's logits exactly; the tiny Qwen2.5-VL decoder with the fixture's position ids gives transformers' logits (one pass, cached, 20 steps) |
| 2 | **Training with M-RoPE.** Packed sequences carry [axes, tokens] positions (restarting per packed sequence); the rotary backward uses the same rows; `TuningVision` takes a family with a position rule (its refusal goes); the feature cache keys already hold the layout | `Idrak.Nlp`, `Idrak` | The tuning fixture's loss and adapter gradients match; a packed batch equals its sequences one by one |
| 3 | **The plug-in: Qwen2.5-VL.** Processor (exact `smart_resize`, Pillow's bicubic, patches in the merge order, `min_pixels` and `max_pixels` as vision options); the tower from `AttentionSpans` (window spans and full spans per image, several images of different sizes in one pass by their spans), 2-D rotary rows from step 1's op, `window_index` and its inverse; the merger; `IVisionEncoderStages` (tower and merger apart, for the feature cache and the trained projector); the prompt format; the registrations (family, prompt format, attention rule `causal`, position rule `grid`, tuning parts) | `samples/QwenVision` | Pixels exact against the processor; each tower stage and the features within float32 noise; the testing kit's `CheckVisionEncoder` on CPU, CUDA and Vulkan; prompt ids equal the processor's; the chat API answers the fixture's 20 tokens |
| 4 | **Qwen2-VL.** The same plug-in, the other tower (LayerNorm, quick-GELU, full attention throughout, its merger) | `samples/QwenVision` | Its fixture as step 3's |
| 5 | **Speed and memory on the device.** Measure a 1,024-wide page: the tower (window and full blocks on `AttentionSpans`), the prefill of about 2,000 image tokens, decode tokens a second, peak memory; profile and fix what is slow in the library (no app workaround); the rotary-rows op fused where the profile says so | library | Numbers recorded in the plan; the owner's CUDA runs |
| 6 | **The legal OCR app.** The plug-in registered beside Gemma 3's; Qwen2.5-VL 3B and 7B Instruct in the switch (no Hugging Face gate on Qwen's own repositories), their prompt, a "max pixels" field in the Read settings (the page's token count and time follow it); the Fine-tune tab tunes on them unchanged (`ConversationTuning`), with the projector and max pixels as options; the "As the model sees it" tab shows the resized page and its token grid | `samples/Idrak.Samples.LegalOcr` | App tests with the tiny Qwen2.5-VL: read, evaluate, tune, read with the adapter |
| 7 | **Owner's commands.** Build, the CUDA filters for steps 1-4, the app with Qwen2.5-VL-3B: CER on 20 evaluation pages against bakrianoo's model, a short tuning run (speed, memory, CER before and after) | — | Given |

## Order and size

1 and 2 are the library's part and the bulk of the risk (every rotary path: one pass, cached prefill, decode steps,
packed training, the fused norm-and-rotary kernels must take the rows or step aside); 3 depends on 1; 4 is small after 3;
5 runs beside 3 on the owner's card; 6 last. Step 0 needs Python with `transformers` and `torch` (CPU only) where the
fixtures are made: this container has neither installed; installing them here is the first action of step 0.

## Decisions for the owner (each with a default)

- **Which first:** Qwen2.5-VL-3B-Instruct (default: fits a LoRA run in bfloat16 on more cards than the 7B; the 7B with
  int4 or int8 weights after).
- **Qwen2-VL too:** yes (default), as step 4, for the 2B and the OCR fine-tunes built on it.
- **Default max pixels in the app:** the processor's own default (step 0 reads it), with the field to lower it when a
  page's ~2,000 tokens are slower than wanted.

## Rules that apply

Speed and memory first (the tower's full blocks never store a score matrix; the rotary rows are made once per pass;
measured on the device, cold and warm apart); card-agnostic (no model size tied to a memory size in code or docs);
contract plus registry for the position rule and the rotary sections, the API dump and inventory kept current; no
Qwen-VL vision code in the library (the plug-in), and an unregistered `Qwen2_5_VLForConditionalGeneration` image request
fails with "not registered"; results proven unchanged for every existing model (a text model's logits and the Gemma 3
fixtures bit for bit after step 1).
