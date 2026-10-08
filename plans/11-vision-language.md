# Plan 11: images into language models (Gemma 3 first)

**Status:** planned 2026-10-07, revised the same day against the code; phase 0 done 2026-10-08. Starts after plan 10's wave 4 (in
progress). Asked for to run `bakrianoo/arabic-legal-documents-ocr-1.0`, a fine-tune of Gemma-3-4B-IT that reads scanned
Arabic legal documents (low quality scans included) and returns their contents as structured data. Its card asks for
images resized and turned to grayscale first, and shows it running through transformers and vLLM.

**First version (decided 2026-10-07): the command line only.** `idrak chat <model> --image scan.png`, `/image` inside a
chat, and the one-shot `idrak run <model> --image scan.png "prompt"`. **Then, once the command line is proven on the
real model, images go into chat everywhere** (decided the same day): the engine's chat model, `MapChatApi` and the
OpenAI-style `/v1` API (phase 8). MCP and other model families come after that (phase 9).

## What Idrak has, and what is missing

| Part | Today |
|---|---|
| Gemma 3's text decoder (`Gemma3ForCausalLM` in `src/Idrak/Models/Architectures.cs`: sliding-window layers from `layer_types` or `sliding_window_pattern`, RoPE theta per layer kind, the 4B's linear RoPE scaling on the global layers, q/k norms, tied embeddings, safetensors, int8 and int4 weights) | supported |
| `Conv2d`, `LayerNorm`, `GELU`, `MultiHeadAttention`, image codecs (PNG, BMP, Netpbm in core's `ImageCodecs`, a slot table) | supported, as layers and in core |
| Attention that is not causal, tiled: every tiled and KV-cache kernel (`AttentionTiled`, `AttentionSegmented`, `AttentionRows`, `AttentionDecode`) masks causally, by window and by packed segment; a bidirectional pass composes the scores in full | missing (see phase 3a) |
| The SigLIP vision encoder: 896 x 896 RGB → 64 x 64 = 4,096 patches of 14 x 14, learned position embeddings, 27 bidirectional layers of width 1,152 (LayerNorm eps 1e-6, attention with biases, 16 heads, a GELU-tanh MLP of 4,304), a final LayerNorm | missing |
| The multimodal projector: 4 x 4 average pooling to 16 x 16 = 256 tokens, Gemma's RMSNorm (1 + w), then x · W with W stored as [1,152, 2,560] (the transpose of a `Linear` weight) | missing |
| Loading `Gemma3ForConditionalGeneration`: `text_config` and `vision_config` nested in `config.json` (only the CLI's `BaseModelInfo` reads `text_config` today); the image token ids (`boi_token_index`, `image_token_index`, `eoi_token_index`, `mm_tokens_per_image`) read from the config, not fixed | missing |
| Image tokens in the prompt: each image becomes `<start_of_image>`, 256 soft tokens whose embeddings are the projected image features (not scaled by √hidden like text embeddings), `<end_of_image>`; the soft tokens of one image attend to each other in both directions, the rest of the prompt stays causal | missing |
| Preprocessing as `preprocessor_config.json` says (resize to 896 x 896, bilinear, no crop; scale by 1/255; normalize with mean 0.5 and std 0.5; RGB), plus grayscale for this fine-tune | missing (codecs only) |
| Chat messages with images (`ChatMessage.Content` is a string), and Gemma 3's Jinja chat template walking content parts (`{'type': 'image'}`) | missing |
| JPEG decoding (scans are often JPEG) | missing |

### Tensor names: three layouts

Checkpoints of this model exist in three layouts; the loader takes all three (the real fine-tune's layout is checked on
the author's machine, phase 7). Corrected by phase 0: transformers 4.52 to 4.57 renamed the modules (`model.*`) but
their `save_pretrained` reverses the mapping (4.52.4 `modeling_utils.py` 3528-3540 with Gemma 3's
`_checkpoint_conversion_mapping`, `modeling_gemma3.py` 1236-1240), so 4.51.3, 4.52.4 and 4.57.6 all write the first
layout, byte for byte. The second is their in-memory naming, which tools that save a raw state dict write. Transformers 5
flattens `SiglipVisionModel` (no `vision_model` level; `conversion_mapping.py` 784, `PrefixChange` on load) and its
`save_pretrained` does not put the level back:

| Part | `save_pretrained` of 4.51 to 4.57 | raw state dict of 4.52 to 4.57 | `save_pretrained` of 5.x (5.19.0) |
|---|---|---|---|
| text decoder | `language_model.model.*` | `model.language_model.*` | `language_model.model.*` |
| vision encoder | `vision_tower.vision_model.*` | `model.vision_tower.vision_model.*` | `vision_tower.*` |
| projector | `multi_modal_projector.{mm_input_projection_weight, mm_soft_emb_norm.weight}` | the same under `model.` | as in 4.51 |
| `lm_head` | not saved (tied) | not saved (tied) | not saved (tied) |

Inside the encoder: `embeddings.patch_embedding.{weight,bias}` (a [1,152, 3, 14, 14] convolution with stride 14),
`embeddings.position_embedding.weight` ([4,096, 1,152]), `encoder.layers.N.{layer_norm1, self_attn.{q,k,v,out}_proj,
layer_norm2, mlp.fc1, mlp.fc2}`, `post_layernorm`.

## Where each part lives (decisions 8 and 10 of plan 10)

| Part | Package | Why |
|---|---|---|
| `ChatMessage` with images (content parts or an image list, open question 2) | `Idrak.Abstraction` (`Generation`) | used by Nlp now, by AspNetCore (`/v1` `image_url`) in phase 8 and by Mcp in phase 9 |
| The span-masked attention operation (phase 3a) | `Idrak.Abstraction` (an operation in `Backend.cs`, with the CPU kernel) and `Idrak.Gpu` | every device implements operations; plan 9's table |
| SigLIP layers, the projector, `Gemma3ForConditionalGeneration` loading | `Idrak` (core: layers and `Idrak.Models`) | model loading is core's; core is their only user |
| Image preprocessing from `preprocessor_config.json` | `Idrak` (core) | read when a model loads |
| Generation and chat with images, the chat template's image parts | `Idrak.Nlp` | generation is Nlp's |
| JPEG codec | `Idrak` (core), a library default in `ImageCodecs` | core owns the image codecs since phase 8c |
| `--image`, `/image` | `Idrak.Cli` | the first version's surface |

A new contract goes where `ContractsWithTheirUsers` (plan 10, phase 8c) puts it: with its one user's `.Abstractions`
namespace until a second library package uses it.

## Phases

| # | Phase | Done when |
|---|---|---|
| 0 | **A reference to check against.** A script (`tools/vlm/make_tiny.py`) writes a tiny random `Gemma3ForConditionalGeneration` (2 vision layers, 2 text layers, small widths, a 56 x 56 image, so 16 patches pooled to 4 soft tokens) as safetensors with its configs, a fixed test image (PNG and JPEG), and transformers' outputs for it: the preprocessed pixels, the vision features, the projected tokens, the logits of a prompt with the image, and 20 greedy tokens. It builds its own small tokenizer (the `tokenizers` library, with the three image tokens and Gemma's chat tokens), so it needs nothing from huggingface.co and runs in this container (transformers installs from PyPI). Every tensor layout is written (one model, saved three times). All checked into `tests/Idrak.Tests/data/vlm` (870 KB) | **done 2026-10-08** (as built below): the files exist and the script reruns to the same bytes |
| 1 | **Contracts.** Images on `ChatMessage` in `Idrak.Abstraction.Generation`, shaped so `/v1`'s content parts map onto it later; what a chat model says it accepts (text only, or images too) | the contract tests pass; text-only models and code are unchanged |
| 2 | **Preprocessing and JPEG.** The image pipeline from `preprocessor_config.json` (resize, rescale, normalize, channel order) and the grayscale option; a JPEG decoder (baseline and progressive, 4:2:0 and 4:4:4, restart markers, grayscale and YCbCr; not CMYK) registered as a library default in `ImageCodecs` | pixels match the reference to 1e-5; JPEG files from common encoders (libjpeg, phones, scanners) decode |
| 3a | **Span-masked attention.** One operation for every mask the library needs: per query row, the range of keys it sees ([first, last], read from a small device buffer or computed from a rule). Bidirectional (the encoder: every row sees all), causal, sliding window, packed segments and image blocks are rules of it. CPU kernel, Vulkan and CUDA kernels tiled like `AttentionTiled`, HIP through the host fallback; a reference in the conformance kit; the existing causal paths unchanged (they keep their kernels; this one is used where a mask is not causal) | the kit's cases pass on CPU and Vulkan; a 4,096-token bidirectional pass never builds the full score matrix |
| 3b | **The SigLIP encoder and the projector** as core layers, built from a `vision_config`, on 3a's attention | the tiny model's vision features and projected tokens match the reference (CPU and Vulkan) |
| 4 | **Loading the whole model.** `Gemma3ForConditionalGeneration` in the pretrained families: nested configs, both tensor layouts, tied embeddings, the image token ids; the text part reuses Gemma 3's decoder spec (its `text_config` read as a `Gemma3ForCausalLM` config) | the tiny model loads in each of the three layouts; a text-only prompt gives the same logits as transformers |
| 5 | **Image tokens in the decoder.** At prefill, the soft tokens' embeddings are replaced by the image features; each image's soft tokens see each other (3a's image-block rule, on top of the sliding window of the local layers); decoding afterwards uses the KV cache as today (the image costs nothing more per generated token). Several images in one prompt | the tiny model's logits with an image and its 20 greedy tokens match the reference (CPU and Vulkan) |
| 6 | **Chat with images in Nlp.** The chat template's image parts expand to the image tokens (the Jinja engine walks content parts); generation takes images; an image is encoded once per conversation (kept with the conversation by content hash) | a two-turn chat about one image encodes it once and matches the reference's first turn |
| 7 | **The command line.** `idrak chat <model> --image FILE` (repeatable; also `/image FILE` inside a chat, like `/file`); the one-shot `idrak run <model> --image FILE "prompt"` (with `--schema` for the structured answer this model gives, `-j` for JSON); `--grayscale` (or the setting stored with a pulled model) | CLI tests pass with the tiny model; on the author's RTX 5070 Ti the real model reads a sample scan and its greedy output matches transformers' for the first 100 tokens |
| 8 | **Images in chat, after the command line is proven** (phase 7 done on the real model). The engine hosts the model as a chat model that accepts images (`IChatModel`, saying which inputs it takes; the image is encoded once per conversation, as in phase 6); `MapChatApi` takes images in its messages; `MapCompletionsApi` (`/v1/chat/completions`) accepts OpenAI's `image_url` content parts (data URLs and, when allowed, http URLs); `idrak serve` serves it. The same preprocessing (and grayscale setting) as the command line | an OpenAI client sends a scan as an `image_url` part and gets the same answer as `idrak run --image`; streamed and not; the aspnetcore tests cover a tiny model with an image |
| 9 | **After that.** An MCP `chat` tool with images (and an `ocr` tool if useful); pan-and-scan for tall pages; other families (Qwen2.5-VL); fine-tuning with images (LoRA on the text decoder, encoder frozen) | decided per item after phase 8 |

**Order.** 0 first (everything is checked against it). 1, 2 and 3a in parallel (disjoint files), then 3b. 4 and 5
together (one family in the decoder). Then 6 and 7. Phase 8 starts only when phase 7 is proven on the real model on the
author's machine.

**Every phase, from plan 10's wave 4:** a public API change updates `api/` (`IDRAK_UPDATE_API=1`) with a changelog
line in the same commit; a new operation gets its conformance-kit case; a new registry entry is a library default in its
slot table; the suite passes on CPU and Vulkan (lavapipe) before the merge; PTX changes are assembled with ptxas
(`nvidia-cuda-nvcc` from PyPI) even where no NVIDIA GPU runs them.

## Phase 0, as built (2026-10-08)

`tools/vlm/make_tiny.py` writes `tests/Idrak.Tests/data/vlm` (next to the GGUF fixtures of `tools/gguf/make_fixtures.py`;
its README lists every file). Run with transformers 5.19.0 and torch 2.14.1 (CPU), tokenizers 0.23.2, Pillow 12.3.0, and
transformers 4.51.3, 4.52.4 and 4.57.6 for the older layout (each loads the same weights, gives the reference logits and
greedy tokens exactly, and writes it). Two runs write the same bytes. 870 KB: three model folders (115 KB of weights
each), the images (34 KB), the reference arrays (390 KB), `manifest.json` (47 KB).

The tiny model: SigLIP width 8, 2 heads, 2 layers, 56 x 56 image, 14 x 14 patches; text width 24, 4 query heads and 2
KV heads of 8 (heads x head_dim = 32 ≠ width), MLP 48, layer 0 sliding (window 8), layer 1 global,
`query_pre_attn_scalar` 12 (≠ head_dim), RoPE theta 1e4 local and 1e6 with linear scaling 8 global, vocabulary 366.
Weights are drawn from one seeded generator (transformers' own init leaves the norms at 1 and the projector at 0, which
would hide mistakes); the 20 greedy tokens take 11 different values. The test image is colorful (a saturated hue sweep,
strokes, pure red, blue and magenta), 100 x 75, with an RGBA and a palette copy beside it.

**Facts for phases 2 to 5, read in transformers 5.19.0** (`models/gemma3/modeling_gemma3.py` unless named; line numbers
of that version) and confirmed by the reference:

- *Preprocessing* (phase 2). The PIL image processor (`Gemma3ImageProcessorPil`; the default `Gemma3ImageProcessor` is
  torchvision's and needs torchvision, which is not installed here; its resize, on tensors, is not checked and may differ in the last step) resizes the **uint8** image with PIL's bilinear
  filter (`image_transforms.py` 362-382: antialiased when shrinking, the result rounded to uint8), then rescales as
  `uint8.astype(float64) * (1/255)` cast to float32 (`image_transforms.py` 119-123), then `(x - 0.5) / 0.5` in float32
  (409-440). Verified bit for bit (`facts.pixels_check`). `do_convert_rgb` is null in `preprocessor_config.json` (as in
  the real model's); the processor sets it to true (`processing_gemma3.py` 30), so a grayscale image becomes three equal
  channels. `convert_to_rgb` is PIL's `convert("RGB")` in every version checked (4.51.3 to 5.19.0,
  `image_transforms.py` 758-775 in 5.19.0): an RGBA image **loses its alpha** and keeps the colors under it (no
  compositing onto white; the fixture's clear box stays green, 2.0 away from white), a palette image is looked up. The
  fine-tune's grayscale step, done with Pillow's `convert("L")`, is L = (19595 R + 38470 G + 7471 B + 32768) >> 16
  (checked). Pillow decodes the progressive and the restart-marker JPEG to exactly the baseline's pixels.
- *SigLIP* (phase 3b, `models/siglip/modeling_siglip.py`). Patch embedding is a valid convolution with stride = patch
  plus a learned position embedding, no class token (116-187); pre-LayerNorm layers, attention with biases and scale
  head_dim^-0.5, not causal (250-299, 325-357); GELU-tanh MLP; `post_layernorm` at the end; no pooling head when
  `vision_use_head` is false (553-620). The output (after `post_layernorm`) is what the projector takes.
- *Projector* (651-684). The N x N patch outputs (64 x 64 real, 4 x 4 here), in row-major patch order, are transposed to
  [B, C, N, N] and average-pooled with kernel = stride = (image_size / patch_size) / sqrt(mm_tokens_per_image) (4 for the
  real model, 2 here), flattened row-major back to tokens; then Gemma's RMSNorm `x * rsqrt(mean(x²) + eps) * (1 + w)` in
  float32 with eps = the **vision** `layer_norm_eps` (1e-6); then `x @ W` with W [vision width, text width]. The image
  features are **not** multiplied by √width; text embeddings are (`Gemma3TextScaledWordEmbedding`, 105-116; the scale is
  stored in the weights' dtype, so √2560 rounds in bfloat16, 483).
- *Merging* (858-879). The processor turns each `<start_of_image>` from the chat template into
  `"\n\n<start_of_image>" + <image_soft_token> x mm_tokens_per_image + "<end_of_image>\n\n"` (`processing_gemma3.py`
  52-56); the model embeds every id as text (soft tokens included), then `masked_scatter`s the projected features over the
  soft tokens' rows in order. `<start_of_image>` and `<end_of_image>` stay ordinary text tokens.
- *Masks* (phase 3a and 5; 687-755, `masking_utils.py` 93-132). `token_type_ids` is 1 on the soft tokens only (not on
  boi/eoi). Each run of soft tokens is a block. Global layers: key k is visible to query q when `k <= q` or both are in
  the same block. Sliding layers: that **and** `k > q - window` (so a row sees `window` keys including itself, and the
  window cuts only the past: an image row sees its whole block ahead). As one range per row, `[first, last]` =
  `[0, max(q, block end)]` global and `[max(0, q - window + 1), max(q, block end)]` sliding; the reference stores both for
  the image prompt (`facts.image_prompt.mask_rows_first_last_key`). Without `token_type_ids` the logits change from the
  first soft token on (by up to 3.6 here) and not before.
- *Decoding* (1034-1042). After the first step `token_type_ids` is dropped, so generated tokens are plain causal (and
  windowed on sliding layers) over the KV cache; the 20 greedy tokens equal one pass without a cache over the whole
  sequence with `token_type_ids` 0 for the new tokens (logits within 2.2e-6).
- *Decoder* (unchanged from the text-only Gemma 3 Idrak has): q and k RMSNorm per head before RoPE (345-349), attention
  scale `query_pre_attn_scalar^-0.5` (307), sandwich norms around attention and MLP (397-416), RoPE per layer kind
  (155-214): `rope_scaling` applies to the global layers only (`configuration_gemma3.py` 137-158: linear divides the
  inverse frequencies by the factor), the local layers use `rope_local_base_freq` unscaled. `Gemma3ForConditionalGeneration`
  never applies `final_logit_softcapping` (1014-1017), unlike `Gemma3ForCausalLM` (633-636); Gemma 3 configs set it null.
  4.51.3 gives the same logits bit for bit, so the mask rules have not changed since.
- *Configs* (phase 4). transformers 5 writes `text_config.rope_parameters` per layer kind
  (`{"full_attention": {"rope_type": "linear", "factor": 8, "rope_theta": 1e6}, "sliding_attention": {...}}`) instead
  of `rope_theta`, `rope_local_base_freq` and `rope_scaling`, plus `dtype` for `torch_dtype`; 4.x writes the old keys.
  Both forms are in the fixtures and the loader reads both. `sliding_window_pattern` is kept beside `layer_types`.
- *Tokenizer* (phase 6). The fixture's tokenizer is Gemma-like (BPE over `▁`-joined text with byte fallback, `<bos>`
  added by the post-processor), its own vocabulary (ids in the README); the chat template is Gemma 3's, rendering
  `{"type": "image"}` as `<start_of_image>` and a system message as a prefix of the first user turn.

## Performance targets (author's RTX 5070 Ti, the real 4B model)

- The vision encoder (4,096 tokens, 27 layers) runs once per image: under half a second in bfloat16. This needs 3a's
  tiled bidirectional attention: composing the scores in full would hold 4,096² x 16 heads (1 GB in float32) per layer.
- Generated tokens per second within 5% of the text-only Gemma 3 4B with the same weight format (int8).
- Memory: int8 decoder plus bfloat16 encoder within 8 GB.
- On a CPU it works but is slow (the encoder's attention over 4,096 tokens); the first version does not optimize it.

## Risks

- **The mask.** The prefill paths (segmented and windowed attention on CPU, CUDA, Vulkan, HIP) are causal; 3a adds one
  operation for other masks rather than a flag on each, and the conformance kit checks it on every device. A device
  without the kernel composes it (slow, but correct).
- **Numbers that must match exactly:** GELU with the tanh approximation, LayerNorm eps 1e-6, the projector's RMSNorm
  (1 + w) and its transposed weight, embeddings scaled by √hidden for text only, the bilinear resize (transformers
  resizes with PIL, which antialiases when shrinking: a 2,000-pixel scan shrunk to 896 needs the same filter). Phase 0's
  reference catches each one; a real scan's pixels are compared in phase 7.
- **The fine-tune's own steps:** its card makes grayscale mandatory; any other step it relies on (crop, contrast) only
  shows up by comparing with transformers on real scans.
- **Downloading:** this cloud container cannot reach huggingface.co (its network policy blocks it), so the real model is
  only run on the author's machines; the tiny reference (phase 0) is what CI checks.
- **The license:** Gemma models come under Google's Gemma terms; Idrak loads the weights and does not ship them.

## Open questions

1. Grayscale: a command-line flag (proposed `--grayscale`), a setting stored with a pulled model, or both.
2. Content parts on `ChatMessage` (like `/v1`) or a separate image list (simpler now, a mapping later).
3. Pan-and-scan for tall pages (Gemma 3's crops of a long document): first version or phase 9.
4. Where an `ocr` tool or pipeline lives (phase 9): plan 10 puts OCR in `Idrak.Vision`, but one built on a language model
   needs Nlp, which Vision does not reference; in Nlp (or Mcp and the CLI) unless Vision's OCR is a model of its own.
