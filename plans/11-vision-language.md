# Plan 11: images into language models (Gemma 3 first)

**Status:** planned 2026-10-07, revised the same day against the code; not started. Starts after plan 10's wave 4 (in
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

### Tensor names: two layouts

Checkpoints of this model exist in two layouts, depending on the transformers version that saved them; the loader takes
both (the real fine-tune's layout is checked on the author's machine, phase 7):

| Part | transformers before 4.52 | 4.52 and later |
|---|---|---|
| text decoder | `language_model.model.*` (`lm_head` tied) | `model.language_model.*`, `lm_head.weight` |
| vision encoder | `vision_tower.vision_model.*` | `model.vision_tower.vision_model.*` |
| projector | `multi_modal_projector.mm_input_projection_weight`, `.mm_soft_emb_norm.weight` | the same under `model.multi_modal_projector.*` |

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
| 0 | **A reference to check against.** A script (`tools/vlm/make_tiny.py`) writes a tiny random `Gemma3ForConditionalGeneration` (2 vision layers, 2 text layers, small widths, a 56 x 56 image, so 16 patches pooled to 4 soft tokens) as safetensors with its configs, a fixed test image (PNG and JPEG), and transformers' outputs for it: the preprocessed pixels, the vision features, the projected tokens, the logits of a prompt with the image, and 20 greedy tokens. It builds its own small tokenizer (the `tokenizers` library, with the three image tokens and Gemma's chat tokens), so it needs nothing from huggingface.co and runs in this container (transformers installs from PyPI). Both tensor layouts are written (one model, saved twice). All checked into `tests/data/vlm` (well under 1 MB) | the files exist and the script reruns to the same bytes |
| 1 | **Contracts.** Images on `ChatMessage` in `Idrak.Abstraction.Generation`, shaped so `/v1`'s content parts map onto it later; what a chat model says it accepts (text only, or images too) | the contract tests pass; text-only models and code are unchanged |
| 2 | **Preprocessing and JPEG.** The image pipeline from `preprocessor_config.json` (resize, rescale, normalize, channel order) and the grayscale option; a JPEG decoder (baseline and progressive, 4:2:0 and 4:4:4, restart markers, grayscale and YCbCr; not CMYK) registered as a library default in `ImageCodecs` | pixels match the reference to 1e-5; JPEG files from common encoders (libjpeg, phones, scanners) decode |
| 3a | **Span-masked attention.** One operation for every mask the library needs: per query row, the range of keys it sees ([first, last], read from a small device buffer or computed from a rule). Bidirectional (the encoder: every row sees all), causal, sliding window, packed segments and image blocks are rules of it. CPU kernel, Vulkan and CUDA kernels tiled like `AttentionTiled`, HIP through the host fallback; a reference in the conformance kit; the existing causal paths unchanged (they keep their kernels; this one is used where a mask is not causal) | the kit's cases pass on CPU and Vulkan; a 4,096-token bidirectional pass never builds the full score matrix |
| 3b | **The SigLIP encoder and the projector** as core layers, built from a `vision_config`, on 3a's attention | the tiny model's vision features and projected tokens match the reference (CPU and Vulkan) |
| 4 | **Loading the whole model.** `Gemma3ForConditionalGeneration` in the pretrained families: nested configs, both tensor layouts, tied embeddings, the image token ids; the text part reuses Gemma 3's decoder spec (its `text_config` read as a `Gemma3ForCausalLM` config) | the tiny model loads in either layout; a text-only prompt gives the same logits as transformers |
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
