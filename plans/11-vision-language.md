# Plan 11: images into language models (Gemma 3 first)

**Status:** planned 2026-10-07, not started. Starts after plan 10's wave 4. Asked for to run
`bakrianoo/arabic-legal-documents-ocr-1.0`, a fine-tune of Gemma-3-4B-IT that reads scanned Arabic legal documents (low
quality scans included) and returns their contents as structured data. Its card asks for images resized and turned
to grayscale first, and shows it running through transformers and vLLM.

**First version (decided 2026-10-07): the command line only.** `idrak chat <model> --image scan.png` and a one-shot
form. **Then, once the command line is proven on the real model, images go into chat everywhere** (decided the same
day): the engine's chat model, `MapChatApi` and the OpenAI-style `/v1` API (phase 8). MCP and other model families
come after that (phase 9).

## What Idrak has, and what is missing

| Part | Today |
|---|---|
| Gemma 3's text decoder (`Gemma3ForCausalLM`: sliding-window layers, RoPE theta per layer kind, safetensors, int8 and int4 weights) | supported |
| Conv2d, LayerNorm, GELU, bidirectional multi-head attention, image codecs (PNG, BMP, Netpbm) | supported, as layers and in core |
| The SigLIP vision encoder: 896 x 896 RGB → 64 x 64 = 4,096 patches of 14 x 14, learned position embeddings, 27 bidirectional layers of width 1,152 (LayerNorm, attention with biases, a GELU-tanh MLP of 4,304), a final LayerNorm | missing |
| The multimodal projector: 4 x 4 average pooling to 16 x 16 = 256 tokens, Gemma's RMSNorm (1 + w), a linear map 1,152 → 2,560 | missing |
| Loading `Gemma3ForConditionalGeneration`: `text_config` and `vision_config` nested in `config.json`; tensors under `language_model.model.*`, `vision_tower.vision_model.*`, `multi_modal_projector.*`; the image token ids (`boi_token_index`, `image_token_index`, `eoi_token_index`, `mm_tokens_per_image`) | missing |
| Image tokens in the prompt: each image becomes `<start_of_image>`, 256 soft tokens whose embeddings are the projected image features (not scaled by √hidden like text embeddings), `<end_of_image>`; the soft tokens of one image attend to each other in both directions, the rest of the prompt stays causal | missing (the decoder masks causally only) |
| Preprocessing as `preprocessor_config.json` says (resize to 896 x 896, bilinear, no crop; scale by 1/255; normalize with mean 0.5 and std 0.5; RGB), plus grayscale for this fine-tune | missing (codecs only) |
| Chat messages with images, and Gemma 3's chat template with image parts | missing |
| JPEG decoding (scans are often JPEG) | missing: the codecs read PNG, BMP and Netpbm |

## Where each part lives (decisions 8 and 10 of plan 10)

| Part | Package | Why |
|---|---|---|
| `ChatMessage` with images (an image list on the message, or content parts) | `Idrak.Abstraction` (`Generation`) | used by Nlp now, by AspNetCore (`/v1` `image_url`) in phase 8 and by Mcp in phase 9 |
| SigLIP layers, the projector, `Gemma3ForConditionalGeneration` loading, the image-token mask in the decoder | `Idrak` (core: layers and `Idrak.Models`) | model loading is core's; core is their only user |
| Image preprocessing from `preprocessor_config.json` | `Idrak` (core) | read when a model loads |
| Generation and chat with images, the chat template's image parts | `Idrak.Nlp` | generation is Nlp's |
| JPEG codec | `Idrak` (core), registered in `ImageCodecs` | core owns the image codecs since phase 8c |
| `--image` | `Idrak.Cli` | the first version's surface |

## Phases

| # | Phase | Done when |
|---|---|---|
| 0 | **A reference to check against.** A script (`tools/vlm/make_tiny.py`, run where transformers is installed) writes a tiny random `Gemma3ForConditionalGeneration` (2 vision layers, 2 text layers, small widths, a 56 x 56 image) as safetensors with its configs and tokenizer, a fixed test image, and transformers' outputs for it: the preprocessed pixels, the vision features, the projected tokens, the logits of a prompt with the image, and 20 greedy tokens. All checked into `tests/data/vlm` (well under 1 MB) | the files exist and the script reruns to the same bytes |
| 1 | **Contracts.** Images on `ChatMessage` in `Idrak.Abstraction.Generation`, shaped so `/v1`'s content parts map onto it later; what a chat model says it accepts (text only, or images too) | the contract tests pass; text-only models and code are unchanged |
| 2 | **Preprocessing and JPEG.** The image pipeline from `preprocessor_config.json` (resize, rescale, normalize, channel order) and the grayscale option; a JPEG decoder (baseline and progressive) in `ImageCodecs` | pixels match the reference to 1e-5; JPEG files from common encoders decode |
| 3 | **The SigLIP encoder and the projector** as core layers, built from a `vision_config` | the tiny model's vision features and projected tokens match the reference (CPU and Vulkan) |
| 4 | **Loading the whole model.** `Gemma3ForConditionalGeneration` in the pretrained families: nested configs, tensor names, tied embeddings, the image token ids; the text part reuses Gemma 3's decoder | the tiny model loads; a text-only prompt gives the same logits as transformers |
| 5 | **Image tokens in the decoder.** At prefill, the soft tokens' embeddings are replaced by the image features; the mask lets each image's soft tokens see each other; decoding afterwards uses the KV cache as today (the image costs nothing more per generated token). Several images in one prompt | the tiny model's logits with an image and its 20 greedy tokens match the reference (CPU and Vulkan) |
| 6 | **Chat with images in Nlp.** The chat template's image parts expand to the image tokens; generation takes images; an image is encoded once per conversation (kept with the conversation by content hash) | a two-turn chat about one image encodes it once and matches the reference's first turn |
| 7 | **The command line.** `idrak chat <model> --image FILE` (repeatable; also `/image FILE` inside an interactive chat), a one-shot `idrak chat <model> --image FILE -p "..."` printing only the answer (`-j` for JSON), and `--grayscale` (or the setting stored with a pulled model) | CLI tests pass with the tiny model; on the author's RTX 5070 Ti the real model reads a sample scan and its greedy output matches transformers' for the first 100 tokens |
| 8 | **Images in chat, after the command line is proven** (phase 7 done on the real model). The engine hosts the model as a chat model that accepts images (`IChatModel`, saying which inputs it takes; the image is encoded once per conversation, as in phase 6); `MapChatApi` takes images in its messages; `MapCompletionsApi` (`/v1/chat/completions`) accepts OpenAI's `image_url` content parts (data URLs and, when allowed, http URLs); `idrak serve` serves it. The same preprocessing (and grayscale setting) as the command line | an OpenAI client sends a scan as an `image_url` part and gets the same answer as `idrak chat --image`; streamed and not; the aspnetcore tests cover a tiny model with an image |
| 9 | **After that.** An MCP `chat` tool with images (and an `ocr` tool if useful); pan-and-scan for tall pages; other families (Qwen2.5-VL); fine-tuning with images | decided per item after phase 8 |

**Order.** 0 first (everything is checked against it). 1 and 2 in parallel, then 3. 4 and 5 together (one family
in the decoder). Then 6 and 7. Phase 8 starts only when phase 7 is proven on the real model on the author's machine.

## Performance targets (author's RTX 5070 Ti, the real 4B model)

- The vision encoder (4,096 tokens, 27 layers) runs once per image: under half a second in bfloat16.
- Generated tokens per second within 5% of the text-only Gemma 3 4B with the same weight format (int8).
- Memory: int8 decoder plus bfloat16 encoder within 8 GB.
- On a CPU it works but is slow (the encoder's attention over 4,096 tokens); the first version does not optimize it.

## Risks

- **The mask.** The decoder's attention kernels mask causally (and by window). Image tokens need a block that attends both ways; the KV-cache prefill paths (segmented and windowed attention on CPU, CUDA, Vulkan, HIP) must take it, or prefill falls back to the composed path for prompts with images.
- **Numbers that must match exactly:** GELU with the tanh approximation, LayerNorm eps 1e-6, the projector's RMSNorm (1 + w), embeddings scaled by √hidden for text only, the resize filter. Phase 0's reference catches each one.
- **The fine-tune's own steps:** its card makes grayscale mandatory; any other step it relies on (crop, contrast) only shows up by comparing with transformers on real scans.
- **Downloading:** this cloud container cannot reach huggingface.co (its network policy blocks it), so the real model is only run on the author's machines; the tiny reference is what CI checks.
- **The license:** Gemma models come under Google's Gemma terms; Idrak loads the weights and does not ship them.

## Open questions

1. Grayscale: a command-line flag (proposed `--grayscale`), a setting stored with a pulled model, or both.
2. Content parts on `ChatMessage` (like `/v1`) or a separate image list (simpler now, a mapping later).
3. Pan-and-scan for tall pages (Gemma 3's crops of a long document): first version or phase 9.
