# Plan 11: images into language models (Gemma 3 first)

**Status:** planned 2026-10-07, revised the same day against the code; phase 0 (the reference) done 2026-10-08, phase 4 (loading, text side), phase 3b (the SigLIP encoder and the projector) and phase 5 (image tokens in the decoder) done 2026-10-08 on CPU and Vulkan (phase 5 on CUDA too), phase 7 (the command line, before phase 6 by the author's choice) built 2026-10-08 and checked on the real model on the author's RTX 5070 Ti 2026-10-09, phase 8 (images in `idrak serve`: `/v1` image_url data URLs, a multipart upload, `/api/chat`) built 2026-10-09 without phase 6 (stopped by the author) and passing with the tiny model on CPU, phase 1 (contracts) done 2026-10-07 on the CPU, phase 10 (vision contracts: the library registers no family, Gemma 3 vision a plug-in in samples/Gemma3Vision, a tiny LLaVA from outside) done 2026-10-09 on CPU and Vulkan; pan and scan (one image as several blocks; Gemma 3's crops in its plug-in, per-request vision options) done 2026-10-09 on CPU and Vulkan; image transforms (a contract and registry; Pillow-exact grayscale, resize, contrast, brightness, sharpness, autocontrast and a JPEG round trip with a libjpeg-turbo-exact encoder; a model card's preprocessing as a string) done 2026-10-09 on CPU; plan 10's wave 4 is done. Asked for to run `bakrianoo/arabic-legal-documents-ocr-1.0`, a fine-tune of Gemma-3-4B-IT that reads scanned
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
| Attention that is not causal, tiled: every tiled and KV-cache kernel (`AttentionTiled`, `AttentionSegmented`, `AttentionRows`, `AttentionDecode`) masks causally, by window and by packed segment; a bidirectional pass composed the scores in full | phase 3a: `AttentionSpans` (as built below) |
| The SigLIP vision encoder: 896 x 896 RGB → 64 x 64 = 4,096 patches of 14 x 14, learned position embeddings, 27 bidirectional layers of width 1,152 (LayerNorm eps 1e-6, attention with biases, 16 heads, a GELU-tanh MLP of 4,304), a final LayerNorm | phase 3b: `SiglipVisionEncoder` (as built below) |
| The multimodal projector: 4 x 4 average pooling to 16 x 16 = 256 tokens, Gemma's RMSNorm (1 + w), then x · W with W stored as [1,152, 2,560] (the transpose of a `Linear` weight) | phase 3b: `ImageProjector`, `PretrainedVision.CreateEncoder` (as built below) |
| Loading `Gemma3ForConditionalGeneration`: `text_config` and `vision_config` nested in `config.json` (only the CLI's `BaseModelInfo` reads `text_config` today); the image token ids (`boi_token_index`, `image_token_index`, `eoi_token_index`, `mm_tokens_per_image`) read from the config, not fixed | missing |
| Image tokens in the prompt: each image becomes `<start_of_image>`, 256 soft tokens whose embeddings are the projected image features (not scaled by √hidden like text embeddings), `<end_of_image>`; the soft tokens of one image attend to each other in both directions, the rest of the prompt stays causal | phase 5: `ImagePrefill` (as built below) |
| Preprocessing as `preprocessor_config.json` says (resize to 896 x 896, bilinear, no crop; scale by 1/255; normalize with mean 0.5 and std 0.5; RGB), plus grayscale for this fine-tune | missing (codecs only) |
| Chat messages with images: content as parts (`ChatPart`, `ChatImage`, phase 1); Gemma 3's Jinja chat template walking content parts (`{'type': 'image'}`) | contracts done (phase 1); templates and generation missing (phase 6) |
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
| `ChatMessage` with content parts (`ChatPart`, the `ChatParts` registry, `IChatModel.PartKinds`; open question 2, decided) | `Idrak.Abstraction` (`Generation`) | Abstraction itself uses them (`ChatMessage`, `IChatModel`, `ChatMLTemplate`); Nlp, AspNetCore and Mcp use `ChatMessage` |
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
| 2 (as built) | 2026-10-08. **JPEG**: `JpegCodec` in core (`src/Idrak/Data/JpegCodec.cs`), a library default of `ImageCodecs` (order png, jpeg, bmp, netpbm; `.jpg .jpeg .jpe .jfif`): SOF0, SOF1 and SOF2 at 8 bits, grey, three components (YCbCr; RGB when Adobe APP14 says transform 0, or ids 'R','G','B', as libjpeg decides) and four (CMYK, or YCCK under Adobe transform 2: libjpeg's ycck_cmyk_convert, then Pillow's inverted "CMYK;I", which Pillow assumes for every CMYK JPEG, and its cmyk2rgb with MULDIV255 rounding), any whole sampling ratio, restart intervals, interleaved or not; APPn and comments skipped: no ICC profile is applied (neither Pillow's decode nor transformers' processors apply one) and no EXIF orientation (the processors do not; transformers' `load_image`, given a path or URL, does with `exif_transpose`, so a rotated phone photo differs until the CLI does the same). 12/16-bit, arithmetic, lossless and hierarchical frames are a `NotSupportedException` naming the variant (`ImageCodecs.Decode` adds the path); damaged data decodes as zeros, as libjpeg does. Pixels follow libjpeg-turbo's defaults: the islow integer IDCT (chosen because it is the reference's: bit-exact at no cost in speed), fancy upsampling for h2v1, h1v2 and h2v2 (others repeat samples, as `int_upsample`), jdcolor.c's fixed-point tables. Not done: libjpeg's interblock smoothing (only used for progressive files missing their last refinement scans). `ImageCodecs.ReadInfo(path)` reads the whole file when a JPEG's frame header lies past the first 64 KiB. **Preprocessing**: `Idrak.Data.ImagePreprocessor` (a concrete class: core is its only user), `FromConfig(path or folder)` / `Parse(json)` reading `do_resize`, `size` {height, width}, `resample` (Pillow's 1 to 5: Lanczos, bilinear, bicubic, box, Hamming), `do_rescale`, `rescale_factor`, `do_normalize`, `image_mean`, `image_std`, `do_convert_rgb`; refused: `do_pan_and_scan` true (open question 3), a size by edge, `do_center_crop`, nearest. `Grayscale` (Pillow's L: (19595 R + 38470 G + 7471 B + 32768) >> 16, then three channels). `Pixels` gives [C, H, W] floats, `Process` a tensor. **Colour** is the default path (2026-10-08, the author): every format of the built-in codecs reaches the processor as the RGB bytes transformers' `convert_to_rgb` gives, which in 4.51 to 5.19 is Pillow's `convert("RGB")`: alpha **dropped**, not composited onto white (hidden colours stay), a palette's tRNS ignored, grey (and grey with alpha) repeated to three channels. To keep `ImageData` floats while matching Pillow's bytes, the codecs bring to 8 bits what Pillow brings to 8 bits (BMP bit fields up to 8 bits: v * 255 / max rounded down; Netpbm maxvals other than 255: rounded, halves to even) and the preprocessor reads any value that is not a whole number of 255ths (16-bit PNG and PPM channels) by its high byte, as Pillow does. Known difference: a 16-bit grey PNG (and a grey PGM deeper than 8 bits) is read by its high byte here, where Pillow opens it as 32-bit integers and `convert("RGB")` clips them at 255 (an almost white image); copying that was judged no use. The resize is Pillow's `ImagingResample` (precompute_coeffs, 22-bit weights, horizontal pass then vertical, 8-bit rounding between). **Measured**: all 21 JPEG fixtures (`tools/jpeg/make_fixtures.py`: baseline 4:2:0/4:2:2/4:4:4, odd 17 x 13, quality 10, optimized tables, progressive, grey, restarts in baseline and progressive, RGB Adobe, EXIF, OpenCV's 4:4:0 and 4:1:1, CMYK with and without the Adobe segment, YCCK) decode to Pillow 12.3's RGB bytes exactly (max difference 0, asserted); the 54 preprocessing cases (8 resize and normalize cases; 23 colour files, each at its size and resized: PNG RGB, RGBA, grey, grey and alpha, palette with tRNS, 16-bit RGB, interlaced 16-bit RGBA, 16-bit grey and alpha, interlaced 4-bit grey and 8-bit RGBA; BMP palette, 24, 32 with bit fields, 5-6-5, 5-5-5; PPM 8 and 16-bit, PGM maxval 100; the CMYK, YCCK, grey and RGB JPEGs) match transformers 5.19's `Gemma3ImageProcessorPil` within 1e-5 (asserted); phase 0's reference (`tests/Idrak.Tests/data/vlm`) passes too: its five JPEGs decode to Pillow's bytes exactly and the tiny model's `preprocessor_config.json` gives its pixel_values (PNG, JPEG, RGBA, palette, grey) within 1e-5; a 2,000 x 3,000 4:2:0 scan-like JPEG decodes exactly as Pillow, and shrunk to 896 x 896 (colour and grey) matches transformers exactly. Speed on this container (4 threads): that JPEG decodes in 150 ms baseline, 157 ms progressive (into floats in [0, 1]; Pillow: 41 and 121 ms), the preprocessing to 896 x 896 takes 77 to 127 ms (Pillow's resize: 40 to 45 ms). Note: transformers 5.x's default `Gemma3ImageProcessor` uses torchvision when installed (`Gemma3ImageProcessorPil` is the PIL one), whose antialiased uint8 resize is close to Pillow but not promised identical; phase 0's reference should say which it used | the JPEG and image preprocessing tests pass on CPU |
| 3a | **Span-masked attention.** One operation for every mask the library needs: per query row, the range of keys it sees ([first, last], read from a small device buffer or computed from a rule). Bidirectional (the encoder: every row sees all), causal, sliding window, packed segments and image blocks are rules of it. CPU kernel, Vulkan and CUDA kernels tiled like `AttentionTiled`, HIP through the host fallback; a reference in the conformance kit; the existing causal paths unchanged (they keep their kernels; this one is used where a mask is not causal) | the kit's cases pass on CPU and Vulkan; a 4,096-token bidirectional pass never builds the full score matrix |
| 3b | **The SigLIP encoder and the projector** as core layers, built from a `vision_config`, on 3a's attention | the tiny model's vision features and projected tokens match the reference (CPU and Vulkan); **done 2026-10-08** (as built below) |
| 4 | **Loading the whole model.** `Gemma3ForConditionalGeneration` in the pretrained families: nested configs, both tensor layouts, tied embeddings, the image token ids; the text part reuses Gemma 3's decoder spec (its `text_config` read as a `Gemma3ForCausalLM` config) | the tiny model loads in each of the three layouts; a text-only prompt gives the same logits as transformers; **done 2026-10-08** (as built below) |
| 5 | **Image tokens in the decoder.** At prefill, the soft tokens' embeddings are replaced by the image features; each image's soft tokens see each other (3a's image-block rule, on top of the sliding window of the local layers); decoding afterwards uses the KV cache as today (the image costs nothing more per generated token). Several images in one prompt | the tiny model's logits with an image and its 20 greedy tokens match the reference (CPU and Vulkan); **done 2026-10-08** (as built below) |
| 6 | **Chat with images in Nlp.** The chat template's image parts expand to the image tokens (the Jinja engine walks content parts); generation takes images; an image is encoded once per conversation (kept with the conversation by content hash) | a two-turn chat about one image encodes it once and matches the reference's first turn |
| 7 | **The command line.** `idrak chat <model> --image FILE` (repeatable; also `/image FILE` inside a chat, like `/file`); the one-shot `idrak run <model> --image FILE "prompt"` (with `--schema` for the structured answer this model gives, `-j` for JSON); `--grayscale` (or the setting stored with a pulled model) | CLI tests pass with the tiny model; on the author's RTX 5070 Ti the real model reads a sample scan and its greedy output matches transformers' for the first 100 tokens; **built 2026-10-08 before phase 6** (as built below): the tiny model passes on CPU and Vulkan; **the real model checked 2026-10-09** (below): done |
| 8 | **Images in chat, after the command line is proven** (phase 7 done on the real model). The engine hosts the model as a chat model that accepts images (`IChatModel`, saying which inputs it takes; the image is encoded once per conversation, as in phase 6); `MapChatApi` takes images in its messages; `MapCompletionsApi` (`/v1/chat/completions`) accepts OpenAI's `image_url` content parts (data URLs and, when allowed, http URLs); `idrak serve` serves it. The same preprocessing (and grayscale setting) as the command line | an OpenAI client sends a scan as an `image_url` part and gets the same answer as `idrak run --image`; streamed and not; the aspnetcore tests cover a tiny model with an image; **built 2026-10-09 without phase 6** (as built below: `/v1` data URLs, the multipart `/v1/chat/upload`, `/api/chat` images, `serve --grayscale`, limits; each request encodes its images): the tiny model passes on CPU; the real model through the server is the author's to run |
| 9 | **After that.** An MCP `chat` tool with images (and an `ocr` tool if useful); pan-and-scan for tall pages; other families (Qwen2.5-VL); fine-tuning with images (LoRA on the text decoder, encoder frozen) | decided per item after phase 8 |
| 10 | **Vision contracts (families are applications).** The library defines the contracts and the generic machinery of vision-language models and registers no family; Gemma 3's vision part moves out to a plug-in; a second family (a tiny LLaVA) registered from outside proves the contracts general | **done 2026-10-09** (as built below): the Gemma 3 tests give identical numbers with the plug-in registered and fail loudly without it; the tiny LLaVA matches transformers on CPU and Vulkan |

### Phase 1 as built (2026-10-07)

Decided by the author during the phase: no "default style" (no Ollama-like image list, no flags enum of inputs);
message content is a contract open to new kinds of input, built the way every other contract and registry of the
library is (a contract, the library's implementations as defaults beside it, a `SlotTable` registry, a testing-kit
suite, an outside plug-in test). All in `Idrak.Abstraction.Generation` (`ChatParts.cs`), since Abstraction itself uses
it (decision 10; the inventory test agrees).

- **`abstract record ChatPart`** with `abstract string Kind`. Library parts: **`ChatText(string Text)`** (kind "text")
  and **`ChatImage`** (kind "image"): the encoded bytes as received (`ReadOnlyMemory<byte> Data`, copied in), `string?
  MediaType` (as given, else sniffed from the signature: PNG, JPEG, GIF, BMP, WebP, TIFF), `string Hash` (SHA-256, 64
  lowercase hex digits; equality and `GetHashCode` by it, so phase 6 can encode an image once per conversation),
  `FromFile(path, mediaType?)`, `FromBytes(ReadOnlySpan<byte>, mediaType?)`, `FromDataUrl(url)` (base64 data URLs of an
  image type only; `FormatException` saying what is wrong otherwise), `ToDataUrl()`. No decoding in Abstraction (core's
  `ImageCodecs` decodes, phase 2). A plug-in makes its own part type (a record deriving from `ChatPart`).
- **`ChatMessage(string Role, IReadOnlyList<ChatPart> Parts, string? Thinking, IReadOnlyList<ToolCall>? ToolCalls,
  string? ToolName)`**: the parts in the order given (images and text interleaved; OpenAI's content parts and Ollama's
  `images` both map onto it). The text constructor stays (`new ChatMessage("user", "hi")`, one `ChatText`; "" gives no
  part), and **`Content`** is the joined text of the text parts (nothing between them, as templates render consecutive
  text parts); `with { Content = … }` replaces the text parts with one, after the others. Equality compares the parts in
  order (the list is not compared by reference), so text-only code and tests are unchanged.
- **`IChatPartKind`** (`Name`, `Write(ChatPart, JsonObject)`, `Read(JsonObject)`): how a kind is saved in the chat JSON,
  `{"type": name, …}`. Registry **`ChatParts`** on a `SlotTable<string, IChatPartKind>`: "text" (`{"type": "text",
  "text"}`) and "image" (`{"type": "image", "media_type", "data": base64}`) registered as library defaults (they live in
  Abstraction, so no `LibraryRegistrations`); `Register`, `Unregister` (restores the default), `Get`, `Find`, `Names`,
  `Default(name)`, `Default(name, version)`, `DefaultVersion`, `Origin`, `SetPolicy` (guarded: writing and reading a
  part are single calls, so an app's kind falls back or is shadowed per call); it shows in `Overrides.Report()` and
  `idrak overrides`. `ToJson(part)`, `FromJson(node)` (a bare string is text; an unknown type throws naming
  `ChatParts.Register`), `ContentToJson(parts)` (a string when the content is text alone, so text-only files are written
  as before; else a list of parts) and `ContentFromJson(node)`.
- **What a model takes:** `IChatModel.PartKinds` (`IReadOnlySet<string>`, a default interface member returning
  `ChatParts.TextOnly`, so no implementation broke); `ChatTemplate.PartKinds` (virtual, text only). **The one check:**
  `ChatParts.ThrowIfUnsupported(model, request, name?)` and `ChatParts.ThrowIfUnsupported(messages, kinds, who)` throw
  `NotSupportedException` naming the model or template, the message (number and role) and the part's kind. It runs in
  `ChatGenerator.RenderPrompt` (so `Chat`, `Stream`, `ChatBatch` and `StreamBatch` too), the engine's chat model (before
  a copy is taken, with the model's name), `FakeChatModel` (whose `PartKinds` is settable), `ChatMLTemplate.Render` and
  `JinjaChatTemplate.Render` (a template never drops a part); `ChatTools.WithTools` forwards the inner model's kinds.
- **JSON:** `ChatJson` (fine-tuning data, the CLI's `--history`, `/save`) writes content through `ContentToJson`;
  `ChatTranscript.FromJson` and the CLI's history loader read it through `ContentFromJson`, so images round-trip and an
  unknown part type is refused with a clear message (fine-tuning used to drop non-text parts silently). `/api/chat` and
  `/v1` keep their own request types and are unchanged for text; `/v1` now refuses non-text content parts (`image_url`)
  with a 400 naming the type instead of dropping them, until phase 8 takes them. MCP takes string content only, as before.
- **Testing kit:** `ChatPartKindSuite` (`Conformance.CheckChatPartKind(name, kind?, samples?)`): reads parts of its
  kind, the JSON it writes keeps the type and reads back equal, the same every time, and agrees with the library's kind.
  `tests/Idrak.PluginTests` registers an audio part kind of its own from outside the library (kit, chat JSON, refused
  by a text-only model, accepted by one that takes it, unregistered).
- Tests: `IDRAK_FILTER="chat parts"` (6 tests) and the outside plug-in test.

**Order.** 0 first (everything is checked against it). 1, 2 and 3a in parallel (disjoint files), then 3b. 4 and 5
together (one family in the decoder). Then 6 and 7. Phase 8 starts only when phase 7 is proven on the real model on the
author's machine.

**3a as built (2026-10-08).** One operation, `Backend.AttentionSpans(q, keys, values, starts, ends, y, logSumExp?, heads,
kvHeads, headsPerTable, rows, keyRows, dim, scale, variant)`, and `AttentionSpansBackward`: query row i of head h sees the
keys starts[t·rows + i] ≤ c < ends[t·rows + i] (half-open, clamped to [0, keyRows]; ints as floats), t = h / headsPerTable
(one table for all heads, or one per sequence), key/value head h / (heads / kvHeads); only the variant's soft-cap is
read; an empty range gives zeros and a log-sum-exp of -∞. The rules are `KeySpans` (Abstraction): `Bidirectional`,
`Causal(rows, window)`, `Segments(lengths, window)`, `ImageBlocks(rows, blocks, window)` (Gemma 3's masks exactly: a row
in an image block sees [max(0, i - W + 1) or 0, max(i + 1, block end)); a test compares every row and key with the
mask's own rule), `Concat`, `ToTensors`. `Tensor.AttentionSpans(q, keys, values, starts, ends, scale, variant)` records
the gradient (weights recomputed from the log-sum-exp). `MultiHeadAttention` without `causal` runs on it (dropout on the
weights while training still composes); every causal path keeps its kernel. **Kernels:** CPU (blocks of 64 rows, tiles
of 128 keys as small products, online softmax; the gradient in two passes, by key/value head and key block, then by
query head and row block), Vulkan (`attention_spans[_lse]`, tiled as `attention_tiled`), CUDA PTX (`attention_spans_f32`,
tiled as `attention_flash_f32`, head sizes up to 128; ptxas 12.9 for sm_50, sm_75, sm_120, no spills; **not run on a
GPU**); HIP and every gradient on a GPU take the host fallback. The kit's case ("attention over key ranges") covers
every rule, random ranges (empty, past the keys), odd sizes, grouped heads, two tables and a soft-cap against plain
loops; it passes on CPU and lavapipe. **Measured** (`--bench-spans`, 4,096 tokens, 16 heads, dim 72, inference, a
4-thread shared CPU): CPU 2.24 s a pass through `AttentionSpans`, peak 215 MB, against 2.30 s and 2,254 MB composed;
lavapipe 44.9 s and 277 MB against 6.0 s and 6,516 MB composed (lavapipe runs the products well and this kernel's
barriers poorly; at 1,024 tokens 1.8 s against 0.7 s, every workgroup width within 10%, so the width is left to the
device's own). The 1 GB score matrix of the composed pass is gone on every device; the Vulkan kernel's speed is to be
measured on a real GPU (and tuned there by measurement if it lags, as `attention_tiled` is).

**3a on the author's GPU, and the faster kernels (2026-10-08).** Measured by the author (CUDA, 70 SMs, compute 12.0,
16 GB; `--bench-spans`, 4,096 tokens, 16 heads, dim 72, inference, float32): `attention_spans_f32` 41 ms a pass, the
composed path (the products, softmax) 10 ms; "attention spans" 3/3 and "conformance kit" 8/8 passed there; the
benchmark printed "peak memory -1 MB" (it read the process's resident memory, which says nothing of device memory).
On lavapipe the Vulkan kernel was also ~7x slower than composed. Changed:
- **CUDA kernels**, one module per padded head size, built and loaded the first time a size runs, each judged first by
  the shared memory the device reports (`PtxKernels.SpansTiled.cs`, `CudaBackend.Spans.cs`):
  `attention_spans_f32_d{D}` (head sizes that are multiples of 4, padded to 8; PTX 6.0 for sm_50): 64 query rows per
  block of 128 threads, keys in tiles of 32, the products tiled in registers (each thread 4 rows × 4 keys of the scores,
  then 4 rows × D/8 columns of the output: two vector reads of shared memory per 16 multiply-adds, where the first
  kernel read shared memory once per multiply-add), dynamic shared memory 384·D + 9 KB (past 48 KB from D = 104: a
  device that cannot opt in to that keeps the first kernel); and `attention_spans_tc_d{D}_w{W}` (padded to 16; PTX 7.0,
  sm_80) when `MixedPrecision` asks for tensor cores and there is no soft-cap: FlashAttention-2 as `flash_tc_fwd` (bf16
  `mma.m16n8k16`, float32 sums and softmax), W = 4 or 8 warps of 16 query rows, **measured per device and shape**
  (`TuneOp.SpanWarps`, kept in the tuning cache), 4 before measuring. `attention_spans_f32` stays for other head sizes.
  ptxas 12.9 assembles every module for every target it lists at or above the module's own (sm_50 … sm_121f; sm_80 …
  for the tensor-core ones): no spills (the d128 tensor-core kernel stages its tiles in a loop for that). **Not run on a
  GPU here**: the speed against the composed 10 ms is the author's to measure.
- **Measured choice, card-agnostic**: `Tensor.AttentionFastest(q, keys, values, starts, ends, scale, everyKey, mask)`
  runs AttentionSpans or the composed scores, whichever the device measured faster for the shape and precision
  (`Backend.PrefersComposedAttention`: CUDA through its tuning cache, `TuneOp.AttentionPath`; Vulkan through its tuning
  file, `VulkanTuneOp.AttentionPath`; per device and driver; the CPU measures nothing and keeps AttentionSpans), and
  only where the composed path can: every key or a dense mask, as many key/value heads as query heads, one table, no
  soft-cap, and the **memory guard**: the scores and weights (four score matrices when the gradient is recorded) within
  half of `Backend.AvailableMemory()` (CUDA: the driver's free memory plus the pool's blocks, less the reserve; Vulkan:
  the storage heap less what is in use; the CPU: the runtime's available memory; none reported: never composed).
  AttentionSpans until measured. `IDRAK_ATTENTION_PATH=spans|composed` (`AttentionPaths.Forced`) forces either (the
  guard still applies). `MultiHeadAttention`'s bidirectional path takes it; a trace shows which ran (AttentionSpans or
  ScaleMaskSoftmax), and `idrak tuning show` / `IDRAK_TUNE_LOG=1` the measurement.
- **Vulkan**: the measured choice above (a path twice as fast as the other in one timed run each is kept at once, since
  the kernel tuning's budget keeps the default without measuring where one run is slow, a software driver at thousands
  of rows: where a clear winner matters most), and `attention_spans`' workgroup width (its rows and key tile) measured
  among the candidate widths per shape as `attention_tiled`'s (`VulkanTuneOp.SpanAttention`), the device's own until
  then. The kernel itself is unchanged (lavapipe, the only Vulkan device here, cannot show what a GPU's would gain).
- **Here** (4-thread shared CPU, lavapipe; 1,024 tokens, timings on a shared machine vary by 2x between runs): CPU
  spans 121 ms, 18 MB of device memory at peak, composed 64 ms and 146 MB, the CPU's measured choice AttentionSpans
  (it measures nothing); lavapipe spans 1.5-4.0 s and 22 MB, composed 0.43-1.26 s and 205 MB, the measured choice
  composed. Peak device memory now prints on every device.
- **For the author's GPU**: `IDRAK_DEVICES=cuda:0` with the filters "attention spans" (the float32 and bf16 kernels
  against the CPU at head sizes 4 to 128), "conformance kit", "attention", "transformer", "window kernels"; then
  `--bench-spans` in float32 and with `IDRAK_MATMUL=bf16` (`IDRAK_TUNE_LOG=1` prints the measured warps and path).
- **Benchmark**: `--bench-spans` runs the three paths (spans, composed, fastest, with the one it chose) in one process
  and prints each path's peak device memory from the device's own accounting (`MemoryUsage.Peak`, new, reset per path by
  `ComputeResources.ResetPeakMemoryUsage`), on CUDA and Vulkan as on the CPU, and the process's peak beside it.

**3b as built (2026-10-08).** All in core; no new contract (concrete layers only; the inventory test agrees).

- **`Idrak.Layers.SiglipVisionEncoder`** (`src/Idrak/Layers/Siglip.cs`): `FromTensors(VisionEncoderConfig, ITensorStore,
  prefix = "vision.", device)` reads SigLIP's tensors one at a time in float32 (shapes checked against the config,
  naming the tensor); `Embed(pixels)` gives the patch embeddings, the forward pass the outputs after `post_layernorm`,
  [images, patches, width] from [images, channels, size, size] (or one image [channels, size, size]). Built from the
  library's layers: the patch embedding is a `Conv2d` (stride = kernel = patch, bias) plus the learned position embedding
  (no class token); each layer is a `TransformerEncoderLayer` (pre-LayerNorm, eps from the config, `MultiHeadAttention`
  with q, k, v stacked into its one [width, 3·width] projection with their biases, scale head_dim^-0.5, bidirectional
  through `AttentionSpans` as 3a made it, GELU-tanh MLP fused with its bias at inference). The body runs in a `Sequential`,
  so without autograd each layer's intermediate results are freed as it goes (a first version holding them to the end
  reached 10 GB and ran out of memory at the real sizes). Refused: an activation other than GELU-tanh
  (`gelu_pytorch_tanh`, `gelu_new`, `gelu_fast`) and `vision_use_head`.
- **`Idrak.Layers.ImageProjector`** (Gemma 3's projector): `FromTensors(store, visionDim, textDim, poolSize, normEpsilon,
  prefix = "projector.", device)`; the row-major patch grid average-pooled with kernel = stride = `PoolSize` (two
  means: the k rows of each band, then the k columns of each cell), `RMSNorm` with offset 1 (1 + w) and
  `ProjectorNormEpsilon`, then `Linear` with W read as stored [vision, text] (x · W).
- **The entry: `PretrainedVision.CreateEncoder(device?, preprocessor?)`** builds both from `OpenTensors()` and returns an
  **`Idrak.Models.ImageEncoder`** (a `Module`: its forward takes pixel values; `Encoder`, `Projector`, `Preprocessor`,
  `Vision`). `Encode(Tensor pixelValues)`, `Encode(string path)`, `Encode(ImageData)`, `Encode(IReadOnlyList<ImageData>)`,
  `Encode(ChatImage)` and `Encode(IReadOnlyList<ChatImage>)` give [images, tokens per image, text width] (no gradients;
  the caller disposes it); files and chat images are decoded by the registered codecs (new
  `ImageCodecs.Decode(ReadOnlySpan<byte>)`) and preprocessed. `PretrainedVision.Preprocessor(grayscale = false)` reads the
  model folder's `preprocessor_config.json` (PretrainedModel.Load now records the folder) or, without one, uses Gemma 3's
  processor defaults at the encoder's size. New factories the encoder needed, public: `Conv2d.FromWeights(weight
  [out, in, k, k] or [out, in·k·k], bias, kernelSize, stride, padding)`, `MultiHeadAttention.FromWeights(qkv, output,
  heads, causal)`, `TransformerEncoderLayer.FromLayers(norm1, attention, norm2, ff1, ff2)`. Weights are float32 only;
  packed encoder weights (bf16) are not done.
- **Tests** (`IDRAK_FILTER=SigLIP`, `VisionEncoderTests.cs`): for the three folders, from phase 0's
  `pixel_values-png.npy`: the patch embeddings, the encoder outputs and the projected features within 1e-5 of
  transformers. Largest differences, the same in every folder: CPU 1.4e-6 (embeddings), 4.7e-6 (encoder), 2.6e-6
  (projected); Vulkan lavapipe 1.9e-6, 4.0e-6, 2.2e-6. End to end from `image.png` (file, pixel values from the
  preprocessor, chat images in a batch of three with a grey JPEG between): 2.6e-6 on the CPU, 2.2e-6 on lavapipe; an
  image encodes the same alone and in a batch; a preprocessor of the wrong size is refused.
- **Measured** (`--bench-vision`: random weights at gemma-3-4b-it's sizes, 27 layers of 1,152, MLP 4,304, 16 heads,
  4,096 patches, projector to 256 x 2,560; 417 M parameters, float32): one 896 x 896 image takes 71.8 s on this
  container's 4-thread shared CPU (84 s the first time), peak memory 3.3 GB (1.7 GB of weights). Most of it is the
  attention (3a measured 2.2 s a layer for this size). The GPU target (under 0.5 s in bfloat16) is for the author's
  machine.

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

## Phase 4, as built (2026-10-08): loading `Gemma3ForConditionalGeneration`

The text side is done; images in the decoder stay phase 5 (and the encoder phase 3b). All in core (`Idrak.Models`).

- **The family** `Gemma3ForConditionalGeneration` is a library default in `PretrainedArchitectures` (registered by
  `PretrainedFamilies.BuiltIns`, like the others). Its spec is Gemma 3's decoder read from `text_config`, with the
  top-level `tie_word_embeddings`, `dtype` and `torch_dtype` merged in when `text_config` lacks them, and
  `Gemma3TextConfig`'s defaults for absent keys (checked against transformers 4.57.6: the original gemma-3-4b-it's
  `text_config` names only width, layers, MLP, window and RoPE scaling; vocabulary 262,208, 8 heads, 4 KV heads, head
  size 256, `query_pre_attn_scalar` 256, five windowed layers in six). `LogitSoftcap` is always null for this family
  (the test gives it `final_logit_softcapping: 30` and checks). The embedding scale √width is rounded to the
  config's dtype (bfloat16: √2560 → 50.5; float16; float32 unchanged), as `Gemma3TextScaledWordEmbedding` does;
  `Gemma3ForCausalLM` keeps the unrounded scale (unchanged).
- **Both config formats.** transformers 5's `rope_parameters` (per layer kind, or one for all) is read into
  `rope_theta`, `rope_scaling` and `rope_local_base_freq` (it wins over the old keys when both are present; RoPE scaling
  on the sliding layers is refused). `Gemma3ForCausalLM` reads it too now (a v5 text config gives the same spec as the 4.x one).
- **Three layouts, detected from the names.** New optional contract member `PretrainedArchitecture.ForCheckpoint`
  (`Func<IReadOnlySet<string>, PretrainedArchitecture>`, in core's `Idrak.Models.Abstractions` beside the
  contract): `PretrainedModel.Load` opens the checkpoint first and keeps the architecture fitted to its names, so
  adapters and exports use the checkpoint's own naming. The text decoder is found by `<prefix>embed_tokens.weight`
  (`language_model.model.` or `model.language_model.`), the vision model by `<prefix>embeddings.patch_embedding.weight`
  (`vision_tower.vision_model.`, `model.vision_tower.vision_model.`, `vision_tower.`); the head maps to
  `language_model.lm_head.*` / `lm_head.*` (never saved: tied).
- **The vision part as data** (for 3b and 5): second optional member `PretrainedArchitecture.Vision`; the loaded
  model's `PretrainedModel.Vision` (`PretrainedVision`, null for text models): `Encoder` (`VisionEncoderConfig`:
  width, MLP, layers, heads, image and patch size, channels, LayerNorm eps, activation, `UseHead`; derived
  `HeadDim`, `PatchesPerSide`, `Patches`; `FromJson` with SigLIP's defaults), `ImageTokens` (`ImageTokenIds`:
  `BeginImage` = boi, `EndImage` = eoi, `ImageToken` = image soft token, `TokensPerImage`; Gemma3Config's defaults
  255999, 256000, 262144, 256 when absent), `TextDim`, `PoolSize` (patches per side / √tokens per image),
  `ProjectorNormEpsilon` (the vision `layer_norm_eps`), `Layout`, and `Tensors`: every encoder and projector tensor
  under one naming, `vision.` + SigLIP's names inside its vision model and `projector.` + the projector's
  (`projector.mm_input_projection_weight` [vision, text], used as x · W, read as stored), each with its stored name
  and shape, checked against the configs at load (a missing tensor or wrong shape is refused naming it).
  `OpenTensors()` reopens the checkpoint as an `ITensorStore` under those names (as stored, no transposition).
  These tensors do not count as unused; the network is the text decoder alone.
- **CLI:** `idrak show` prints a `Vision` line (and `vision` in `-j`) from `vision_config` and `mm_tokens_per_image`.
- **Test** (`IDRAK_FILTER=Gemma3ForConditionalGeneration`, in the model family group): each of the three folders
  loads with no notes, the same spec, the same vision data and projector values, and gives transformers' text-prompt
  logits within 2e-5 (largest difference 5.4e-6 on the CPU, 4.9e-6 on Vulkan lavapipe, the same in every folder); the
  original 4B's sparse config reads with transformers' defaults; a wrong vision shape is refused.

## Phase 5, as built (2026-10-08): image tokens in the decoder

All in core (`Idrak.Layers`, `src/Idrak/Layers/ImagePrefill.cs`); no new contract (core is the only user; phase 6's Nlp
calls the public API below), no Abstraction change.

- **API.** `PromptImage(int Position, Tensor Features) { Sequence }`: an image's features ([tokens, dim] or [1, tokens,
  dim], on the decoder's device, read and never disposed) and the column of its first image token in the ids given
  (`Sequence`: the batch row, 0 by default). `ImagePrefill.ForwardCached(this Sequential decoder, ids, images, context,
  layers?)` (the cached prefill), `ImagePrefill.Forward(this Sequential decoder, ids, images)` (one pass without a cache;
  gradients reach the features), `ImagePrefill.Locate(ids, imageToken, features, sequence)` (each run of
  `PretrainedVision.ImageTokens.ImageToken` paired in order with the features; counts and lengths checked). An empty
  image list is the plain `ForwardCached` / `Forward`.
- **Substitution.** Before the decoder's first `DecoderBlock` (after the embedding and its √width scale), the embeddings
  [n, t, dim] and every image's features are joined (`Concat`) and one `EmbeddingLookup` picks each row: an image token's
  row reads its feature row, every other row its own embedding. So the features are not scaled (as transformers'
  `masked_scatter` after `Gemma3TextScaledWordEmbedding`), and autograd flows to both. `Sequential` got an internal
  hook (`Run` / `ForwardCached` with `before(i, x)`) for it; its public behaviour is unchanged.
- **Masks.** While the step runs, an internal thread-static `ImageBlocks` (like `PackedSequences.Current`) holds each
  sequence's blocks. `CausalSelfAttention` checks it first (one thread-static read; a prompt without images takes
  today's kernels untouched): in `ForwardCore` it projects as usual and attends with `Tensor.AttentionSpans`; in
  `HeadsCached` it projects and **writes the KV cache exactly as the plain prefill does** (fused write, layout `Write`, or
  `WriteKeyValues` for rows of different lengths), then attends with `AttentionSpans` over the cache as the layout
  expands it (`KeyValueLayout.Expand`: float32 read in place, bfloat16 and int8 expanded to float32 for the step, every
  slot; the ranges stop at each row's last key). The ranges are `KeySpans.ImageBlocks` over the cached length plus the
  step (blocks shifted by the cached length), the layer's window on sliding layers, each row's start raised to
  `RowStarts[b]`; one table per sequence (query heads stacked as [n·kv·group, t, d], kv heads [n·kv, ...], so
  `headsPerTable` = kv·group). The soft-cap goes through the variant. After the prefill nothing changes: decoding is the
  usual one-token step (causal, windowed), as transformers drops `token_type_ids` after the first step.
- **Limits.** Each image block must lie inside one prefill step (a refusal says so): a prefill continuing a cached prefix
  (shared-prefix reuse, a later chat turn) works when the image is in the new part; the caller must not split a block
  (phase 6: truncate before a block). Not recordable as a compute graph (ranges uploaded from the host; refused while
  capturing). Packed sequences with images are refused. Batches of equal length and rows of different lengths
  (`SetRowStarts`, float32 cache) both work.
- **Measured** (`IDRAK_FILTER="image prefill"`, CPU, Vulkan lavapipe and CUDA; on an RTX 5070 Ti, compute 12.0, CUDA gives
  3.34e-6 one pass and cached, 3.81e-6 over the 20 greedy steps and the same tokens, both tests passing): phase 0's `image_features.npy` [1, 4, 24] with
  the reference ids [1, 38] give transformers' logits [1, 38, 366] within 3.58e-6 on the CPU and 5.62e-6 on Vulkan (one
  pass and cached prefill alike), the 20 greedy tokens exactly (274 361 122 122 122 122 122 122 162 128 128 116 142 131
  340 344 360 360 360 344) with their step logits within 3.81e-6 (CPU) and 3.34e-6 (Vulkan), the same in all three
  layouts; one pass without a cache over prompt and answer gives the step logits too. With each image token as a block
  of its own (no image-block mask) the logits equal the reference before the first image token and change from it on,
  by 3.596 at most, transformers' own figure without `token_type_ids`. Consistency (no reference): a two-image prompt
  (47 tokens) cached equals uncached; split into a cached prefix and a continued prefill holding the second image it
  equals the one-step prefill; rows before the second image equal the one-image prompt's and do not read it; a batch of
  two prompts with the images swapped equals the single runs (cached and not); rows of different lengths (the one-image
  prompt left-padded by 9 beside the two-image one) equal the single runs; a one-token "image" holding a text token's own
  embedding (the image path with a plainly causal mask) equals the plain prefill within 5e-6 with float32, bfloat16 and
  int8 caches. With an int8 cache the image prompt's logits are 0.255 from float32's (the tiny model's heads of 8 values
  round coarsely; a text prompt: 0.065), bfloat16 0.036.

## Phase 7, as built (2026-10-08): the command line, before phase 6

**The real model (2026-10-09, author's RTX 5070 Ti, `idrak vlm check`, transformers 5.17 Pillow processor, grayscale, a
700 x 1000 JPEG scan).** Idrak with bf16 decoder weights and its float32 encoder against transformers with the vision
tower in float32 (`compare_real.py --vision-dtype float32`): pixels identical (max |Δ| 0), encoder output cosine 1.000000
(max |Δ| 0.0019), image features relative 3.8e-5, 287 prompt ids identical, teacher-forced top-1 63 of 64 and the first
45 greedy tokens identical; the one disagreement is a tie in transformers itself (`reference`/`references`, margin 0 in
bf16 logits). Against transformers all in bfloat16, the answers part at step 17 with a margin of 7 in Idrak: the cause
is transformers' bfloat16 vision tower (features cosine 0.9968 against the float32 ones); given those features Idrak's
decoder agrees except at near-ties, and transformers with a float32 vision tower writes Idrak's answer. So the float32
encoder stays the default. Speed: prompt with the image 1.0 s, 80 tokens/s, a 1,032-token answer in 14 s.

Built before phase 6 (the author: "work on cli first, then chat after settling"). The pieces phase 6 reuses are in Nlp;
what is the command line's alone is in the CLI.

- **Usage.** `idrak run MODEL --image FILE [--image FILE...] [--grayscale] "prompt"` (with `-s`, `--schema`, `-j` and
  every generation option as for text: the user message is the images, then the prompt's text, as transformers' examples
  write it); `idrak chat MODEL --image FILE` (with the first message) and `/image FILE` inside a chat (with the next one,
  like `/file`); `idrak alias set NAME MODEL --grayscale` keeps the setting with the model (the config's alias, the CLI's
  per-model settings: `"grayscale": true`), `--grayscale` gives it for one command. `run -j` adds `images` and
  `grayscale`. `pull` (and the library's `HuggingFaceModels`) now also downloads `preprocessor_config.json` and
  `processor_config.json`.
- **Chat template (Nlp).** `JinjaChatTemplate` gives a message holding a part other than text its content as a list of
  parts, as Hugging Face does (`{"type": "image"}`, `{"type": "text", "text"}`, `{"type": kind}` for other kinds); text-only
  messages keep a string (unchanged prompts). `JinjaChatTemplate.PartKinds` is found by a probe, as the tool-call format
  is: a user turn of one text part must render as the same text given as a string, and an image part before it must change
  it; then "image" is in its kinds (Gemma 3's template: yes; ChatML templates: text only).
- **Image prompt format (Nlp, new contract and registry).** `IImagePromptFormat` (`Name`, `Expand(prompt, images,
  ImageTokenIds, ITokenizer)`) in `Idrak.Generation.Abstractions`, registry `ImagePromptFormats` on a `SlotTable` keyed by
  config.json's `model_type`; library default "gemma3": `ImageMarkerFormat` (each `<start_of_image>` the template wrote
  becomes "\n\n" + boi + soft token x `mm_tokens_per_image` + eoi + "\n\n", transformers' `full_image_sequence`; one
  marker per image or a clear error). The tiny model's prompt renders, expands and tokenizes to the reference's text and
  38 ids exactly.
- **Generation (Nlp).** `TextGenerator.Stream(prompt, imageToken, images, options)`: the tokenized prompt's runs of the
  image token are paired with the features (`ImagePrefill.Locate`) and the prefill runs `ImagePrefill.ForwardCached` (or
  `Forward` without the cache). A block is never split: the prefill is one step (no chunking), a prompt with images that
  does not fit the window throws (no truncation), the kept cache is reused only up to the first image (image tokens have
  the same id whatever the image), and a full window re-read starts after an image it would cut. `ChatGenerator.Images`
  (`ChatImages`: the image token ids, the format, an encoder `Func<IReadOnlyList<ChatImage>, Tensor>` giving
  [images, tokens, width]) makes `PartKinds` text and image, expands the prompt in `RenderPrompt`, and encodes the
  request's images in `Stream` (disposed after). **Each request re-encodes the conversation's images** (a later chat turn
  reads them again, and the prefill restarts at the first image): acceptable for the command line; encoding once per
  conversation (by `ChatImage.Hash`) is phase 6. Chat batches refuse images (text-only models still refuse them first,
  with the usual message).
- **CLI.** `Shared/ImageInputs.cs`: files as `ChatImage`s, decoded by `ImageCodecs` and turned upright by the EXIF
  orientation (tag 0x0112 of the first IFD, from a JPEG's APP1 "Exif" segment or a PNG's eXIf chunk), as transformers'
  `load_image` does with `ImageOps.exif_transpose`; only here, the codecs and processors are unchanged. `ModelImages`
  builds the encoder on first use (`PretrainedVision.CreateEncoder(model.Device, vision.Preprocessor(grayscale))`, phase
  3b) and disposes it with the model; an internal `ImageInputs.EncoderFactory` lets tests feed phase 0's features. A
  model whose chat does not take "image" (a text model, a family without a registered format, or a template that writes
  nothing for an image) refuses `--image` (exit 2, "reads text only (its chat model takes 'text' parts)") and `/image`
  (a reply in the chat).
- **Tests** (`IDRAK_FILTER="cli images"`, CPU and Vulkan lavapipe): EXIF orientations 1 to 8 in JPEG and 3, 6, 8 in PNG
  eXIf give Pillow 12.3's `exif_transpose` bytes exactly (`tools/vlm/make_exif.py` writes the 92 KB of fixtures in
  `tests/Idrak.Tests/data/vlm/exif`); the template, format and ids against phase 0's facts; `run --image` with the tiny
  Gemma 3 and the real encoder gives transformers' 20 greedy tokens (read by a recording sampler registered over the
  default) and 38 prompt tokens, the same answer with phase 0's `image_features.npy` fed in, the pixels read equal
  `pixel_values-png.npy` and with `--grayscale` (flag or alias) `pixel_values-png-gray.npy` within 1e-5; two images;
  `--schema -j`; a window too small for the image refused; `chat --image` then `/image` (two turns, two encodings, the
  images saved as parts); a text model refuses `--image` and `/image`; a missing file.
- **Not done here.** The real model on the author's RTX 5070 Ti (phase 7's last check: 100 greedy tokens against
  transformers on a sample scan); the model card's exact prompt and preprocessing (huggingface.co is blocked here).
- **First real run, and the comparison tool (2026-10-08).** On the author's RTX 5070 Ti, `idrak run MODEL -d cuda:0 -w bf16
  --grayscale --image scan.jpg --temperature 0 -j` on the real fine-tune gave a faithful structured reading (287 prompt
  tokens as in transformers, 1,032 generated at 80 tokens/s, the prompt in 1.0 s), but its greedy answer left
  transformers' (bfloat16 throughout) at about the 17th token. To tell a near-tie flipped by precision from preprocessing
  or a bug at real size: `tools/vlm/compare_real.py` (transformers' side: the prompt's ids, pixels, the vision output, the
  projected features, N greedy tokens with each step's top 5 logits, the top 5 of one teacher-forced pass, a few whole
  logit rows, a manifest with versions, dtypes, grayscale and processor class; bfloat16, decoder bfloat16 with the vision
  side float32, or float32 on the CPU or with `--offload`) and `idrak vlm check MODEL --reference DIR` (a developer
  command beside `onnx check`: pixels, encoder and features from the reference's pixels, the prompt's ids from Idrak's
  template, transformers' tokens fed back with Idrak's features and with the reference's (to separate the image side from
  the decoder), each disagreement's two top-5 lists and margins (near-tie below `--tie`, 0.5 by default), Idrak's own
  greedy tokens, a verdict; exit 1 on a real difference). On the tiny model both reference folders checked in
  (`tests/Idrak.Tests/data/vlm/compare`, colour and grey from a JPEG) agree exactly (`IDRAK_FILTER="cli images"`). Also
  `run -o FILE` writes the answer (or the `-j` document) as UTF-8 without a BOM: PowerShell decodes a native command's
  output with `[Console]::OutputEncoding` (cp437 there) even though the tool writes UTF-8; the CLI README's Windows
  section has the PowerShell setting. The real model's numbers are to come from the author's machine.

## Phase 8, as built (2026-10-09): images in `idrak serve`

Built after phase 7 was checked on the real model, without phase 6 (stopped by the author): each request encodes its
images once, when it is answered; nothing is kept between requests (no content-hash cache).

- **Engine (Nlp).** `GenerativeModelBuilder.Images(Func<TextGenerator, ChatImages>)`: called for each copy once its
  text generator loads, giving how that copy reads images; the engine disposes the new `ChatImages.Owner` (the
  encoder's holder) when it unloads the copy. The engine's chat model's `PartKinds` is text and image, known before
  anything loads, when it has `Images` and its template renders images (`ChatTemplate.PartKinds`); text only otherwise,
  so a text model refuses an image before it loads (`ChatParts.ThrowIfUnsupported`, as phase 1). Generation is as
  before: one request at a time per copy (the engine's lease), the encoder built on the copy's first image.
- **Core.** `Idrak.Data.ChatImageDecoder` (moved from the CLI's `ImageInputs`, not copied): `Decode(ChatImage)` (the
  codecs, then the EXIF orientation as `exif_transpose`), `Orientation`, `Orient`, `FormatOf(image)` (the registered
  codec whose `ReadInfo` accepts the header, or null: the check before decoding) and `Grayscale(image)`: the image
  upright and grey (Pillow's L, the preprocessor's own byte rounding) as an 8-bit binary PGM, which any preprocessor
  reads exactly as it reads the original with `Grayscale` on (asserted against phase 0's `pixel_values-png-gray.npy`),
  so one request can be grey without another preprocessor. The CLI's `ImageInputs.Decode` calls it.
- **AspNetCore.** `ImageInputOptions` (`MaxImages` 8, `AllowUrls` false, `MaxUrlBytes` 20 MB, `UrlTimeout` 30 s) on
  `CompletionsApiOptions.Images` and `ChatApiOptions.Images`. Every chat endpoint checks a request's images before
  answering: at most `MaxImages` (400), each in a format a registered codec reads (400 naming PNG, JPEG, BMP,
  PPM/PGM), grey when asked; an image for a text-only model is a 400 (`unsupported_content`), a file that does not
  decode a 400 (`invalid_image`), a body over the host's limit a 413.
  - `/v1/chat/completions`: content parts `{"type": "text"}` and `{"type": "image_url", "image_url": {"url":
    "data:image/...;base64,..."}}` (or `"image_url": "..."`); http(s) URLs are downloaded first only with `AllowUrls`
    (size and time limits; a failed download is a 400 naming the address), otherwise refused with a 400; other part
    types are refused. `"grayscale": true` reads the request's images grey. `stream` as before (chunks, `[DONE]`).
    `CompletionsTranslation.Options` now defaults `repeat_penalty` to 1 (off): the wire format has no repetition
    penalty, and with the library's 1.1 a greedy `/v1` answer left `idrak run`'s at the 11th token of the reference.
  - `POST /v1/chat/upload` (new): `multipart/form-data` with `image` (one or more files; `images`, `image[]` too),
    `prompt`, and optionally `system`, `stream`, `include_usage`, `max_tokens`, `temperature`, `top_p`, `top_k`,
    `min_p`, `seed`, `stop`, the penalties, `reasoning_effort`, `grayscale`, `model`; the user message is the images
    then the prompt (as `run --image`). Answers exactly as `/v1/chat/completions` (`chat.completion`, or
    `chat.completion.chunk` events and `data: [DONE]`); not a form: 415. `CompletionsTranslation.Upload(form)` is public.
  - `MapChatApi` (`/api/chat`): `ChatApiMessage.Content` is a `JsonNode` (a string as before, or parts in phase 1's
    chat JSON: `{"type": "image", "media_type", "data"}`), plus `images` (base64 strings or data URLs, before the
    content, as common local-model clients send them); `ChatApiRequest.Grayscale`.
- **`idrak serve`.** A served model reads images when its config has a `vision_config`, its `model_type` has an
  image prompt format and its template renders images; its copies then read them through phase 7's `ModelImages`
  (the same preprocessing, EXIF orientation and grayscale as `run --image`). New options: `--grayscale` (or the alias's
  setting), `--max-request-mb` (32: Kestrel's body limit and the form limit; 413 with a JSON error), `--max-images`
  (8), `--allow-image-urls` (downloads held to the request size and 30 s). The announcement and `--json` name the
  vision models and the upload route; uploads are logged with their fields and file sizes, not their bytes.
- **Tests** (CPU, in-process on a loopback port): `IDRAK_FILTER="aspnetcore: images"`: a vision chat model of the
  library's own (the engine with `Images`, no CLI) answers `/v1` with a data URL with transformers' 20 greedy tokens
  (recorded by a sampler) and 38 prompt tokens, streamed text equal to the non-streamed, three requests at once each
  as alone, an allowed http URL served by the same app, a URL answering 404 refused, the upload both ways, `/api/chat`
  with parts and with `images`, grayscale pixels equal to the reference; refused: a text model (400 before loading),
  bad base64, a GIF, three images over a limit of two, a damaged PNG, an unknown part type; the owner disposed on
  unload. `IDRAK_FILTER="cli serve: images"`: `idrak serve` with the tiny Gemma 3 and a text GGUF answers `/v1`
  (both modes), the upload (both modes) and `/api/chat` exactly as `idrak run --image` (and the reference's tokens);
  `grayscale=true` and `serve --grayscale` give `run --grayscale`'s answer; refused: images to the text model (JSON and
  upload), an http URL without `--allow-image-urls`, three images over `--max-images 2`, bodies over
  `--max-request-mb 1` (413, JSON and form, judged from the announced length), bad base64, a GIF upload, an upload
  with neither prompt nor image, `grayscale=maybe`. Also run: `aspnetcore`, `cli serve`, `chat parts`, `cli images`,
  `public API`, `abstraction inventory`.
- **Not done.** Phase 6 (encoding once per conversation); a content-hash cache of features; MCP images (phase 9);
  `--schema` on the server (the CLI's schema is an instruction plus a check, not a server feature yet); the real model
  through the server on the author's machine (the commands are in the hand-back).

## Phase 10, as built (2026-10-09): vision contracts (families are applications)

**Why.** Until now the vision side was Gemma 3 dressed as the library: `PretrainedVision`, `VisionEncoderConfig` and
`ImageEncoder` were sealed SigLIP-plus-Gemma-projector types in core, carrying Gemma 3's token ids (`BeginImage`,
`EndImage`), a fixed 256 `TokensPerImage`, the bidirectional image-block mask in `ImagePrefill`, and Gemma 3's
processor as the fallback when a model folder had no `preprocessor_config.json`; `ImagePromptFormats` shipped "gemma3".
Qwen2.5-VL, LLaVA or InternVL could not plug in without changing core. The author: the library abstracts any family and
exposes it; a model is an application of the library, not the other way round.

**Decision: families are registrations, never defaults.** A default is right only where implementations are
interchangeable (a GPU kernel falling back to the CPU's: the same contract, the same results). Families are not: a
checkpoint is read by its own family or not at all. So a checkpoint is matched to its family by exact key (the
architecture name in `config.json`), and when nothing matches the error names the registry ("Vision family
'Gemma3ForConditionalGeneration' is not registered (registered: none); register it with VisionFamilies.Register ...").
Nothing falls back to another family, and the library registers none, not even as a library default: Idrak.Abstraction,
Idrak, Idrak.Nlp, Idrak.AspNetCore and the CLI hold contracts, registries and generic machinery only. (Text families
are unchanged and out of scope; `Gemma3ForConditionalGeneration`'s text decoder stays a library text family, its vision
part is not.)

**Contracts** (decision 10: where their users are; the inventory and the public API files are updated):

- `Idrak.Abstraction.Generation` (core and Nlp use them):
  - `IVisionEncoder` (`Width`, `Device`, `Layout(ImageData)`, `Encode(IReadOnlyList<ImageData>)`): decoded images in,
    one `ImageFeatures` per image out. The family's own preprocessing is part of the encoder (it takes decoded pixels, so
    resizing, cropping, tiling or dynamic resolution are its business; core's `ImagePreprocessor` is a building block).
  - `ImageTokenLayout` (`Tokens`, optional `Grid` such as [rows, columns] or [frames, rows, columns], checked to hold the
    tokens): a variable number of tokens per image.
  - `ImageFeatures` (`Features` [tokens, width], `Layout`, optional `Positions` [axes, tokens]): the M-RoPE slot.
  - `IImagePromptFormat` (`Name`, `ImageToken`, `Expand(prompt, layouts, tokenizer)`): each image's marker expanded by
    its own layout; the format carries the family's token ids. `ImageTokenFormat(name, marker, imageToken, begin?,
    end?, before, after)` is a building block covering Gemma 3's (`<start_of_image>` → "\n\n" boi + soft tokens + eoi
    "\n\n") and LLaVA's (`<image>` → that many `<image>` tokens).
  - `IImageAttentionRule` (`Name`, `Causal`, `Spans(rows, blocks, window)`) and the registry `ImageAttentionRules`. The
    library has one rule, "causal" (image rows attend as text rows: the decoder's own behaviour, not a family's
    choice). Gemma 3's plug-in registers "image-blocks" (`KeySpans.ImageBlocks`); an app can take a rule's name.
  - `IVisionEncoderStages` (`PixelValues`, `Tower`, `Features`): optional, for `idrak vlm check`.
- `Idrak.Models.Abstractions` (core alone uses them): `PretrainedVision` (abstract: `Family`, `Width`, `PromptFormat`,
  `Attention`, `StoredTensors`, `CreateEncoder(VisionEncoderOptions)`), `VisionEncoderOptions` (`Device`, `Grayscale`),
  `VisionCheckpoint` (architecture, config, tensors, folder, reopen, notes), `IVisionFamily` (`Name`,
  `Read(VisionCheckpoint)`) and the registry `VisionFamilies` (`Register`, `Unregister`, `Find`, `Get`, `For(name,
  config)`, `HasVision`; keyed by architecture name; a checkpoint "has vision" when its config has a `vision_config`).
  `PretrainedArchitecture.Vision` is gone: `PretrainedModel.Load` asks `VisionFamilies.For` before reading anything.
- Testing kit: `VisionEncoderSuite` and `Conformance.CheckVisionEncoder(encoder, cpu?, samples?, tolerance)`:
  deterministic; token counts match the layout (rows, width, grid, position ids, device); batch equals single; device
  equals CPU.

**Generic machinery changed.** `ImagePrefill.Forward/ForwardCached` take the family's `IImageAttentionRule` (a causal
rule only substitutes the embeddings and keeps the decoder's causal kernels; another rule gives each row its key range);
`Locate` pairs images with image tokens by each image's count, so LLaVA's adjacent images (`<image><image>`) are told
apart, and an overload takes `ImageFeatures` (with their position ids); position ids are refused by the decoder with a
clear `NotSupportedException` (see Qwen2.5-VL below). `TextGenerator.Stream(prompt, imageToken, ImageFeatures[],
rule, options)`. `ChatImages(encoder, format, rule)` with `Decode` (default `ChatImageDecoder.Decode`): `RenderPrompt`
decodes the images for their layouts; `Stream` decodes each image once, expands by the layouts, encodes, checks the
features against the layouts. `ImagePromptFormats` (Nlp) is removed. `ImagePreprocessor`: resize by `ShortestEdge`
(transformers' `get_resize_output_image_size`), `CenterCrop` (transformers' `center_crop`, zero padding when smaller),
and `FromConfig/Parse(..., defaults)` so absent keys take the family's processor defaults. New generic layers:
`ExactGELU` (erf from element-wise operations, Abramowitz and Stegun 7.1.26, within 1.5e-7; every device, no kernel),
`QuickGELU`, registered as layer types `gelu_exact`, `quick_gelu`; `TransformerEncoderLayer.FromLayers(..., Module
activation)`.

**Where Gemma 3 lives now: `samples/Gemma3Vision/`.** `Idrak.Gemma3Vision` (a library referencing Idrak as packages,
from the clone through IdrakFromSource) holds everything Gemma-3-vision: `Gemma3VisionFamily` (config reading with
Gemma3Config's defaults, the three tensor layouts of the vision part, shape checks), `Gemma3Vision : PretrainedVision`
(token ids as `Gemma3ImageTokens`, `PoolSize`, its prompt format, its rule, its preprocessing: `preprocessor_config.json`
over Gemma3ImageProcessor's defaults, or that processor at the encoder's size without the file: the family's decision),
`Gemma3ImageEncoder` (`IVisionEncoder`, `IVisionEncoderStages`), `SiglipVisionEncoder`, `SiglipVisionConfig`,
`Gemma3Projector`, and the "image-blocks" rule. Entry points: `Gemma3VisionPlugin.Register()` and
`RegisterIdrakPlugin()` (the CLI's `-P` convention). `Idrak.Samples.Gemma3Ocr` was an app reading a scan through the
public API (removed from the branch 2026-10-09 with the web sample `Idrak.Samples.Gemma3Web`; both are in history up to
06f5808) (register, load, encode, stream, `-o` as UTF-8 without a BOM, `--grayscale`, `-d`, `-w`). The CLI's `run`,
`chat`, `serve`, `vlm check` and `show` use the contracts only: without `-P` the tiny Gemma 3 is refused with the
registry's message (`serve` refuses it at startup, exit 2), `show` says "vision family ... not registered (load its
plug-in with -P)". The Gemma 3 tests reference the sample and register it first; the numbers are identical to phases
3b to 8 (CPU: SigLIP 1.43e-6 / 4.68e-6 / 2.62e-6, prompt logits 3.58e-6, 20 greedy steps 3.81e-6, without the mask
3.596, text prompt 5.36e-6; Vulkan lavapipe: 1.91e-6 / 3.99e-6 / 2.15e-6, 5.62e-6, 3.34e-6, 4.89e-6); a core test loads the tiny Gemma 3 vision folder without the registration and checks the
error.

**The LLaVA proof** (`tools/vlm/make_tiny_llava.py` → `tests/Idrak.Tests/data/vlm-llava`, 682 KB, two runs write the
same bytes; transformers 5.19.0, torch 2.14.1, Pillow 12.3.0). A tiny `LlavaForConditionalGeneration`: CLIP tower
(width 16, 3 layers, 2 heads, quick-GELU, LayerNorm 1e-5, 28 x 28 images in 7 x 7 patches plus a class token), a
two-layer MLP projector with exact GELU, a Llama decoder (width 32, 2 layers, 4 heads, 2 KV heads), its own tokenizer,
phase 0's 100 x 75 image resized by its shortest edge (bicubic, to 37 x 28) and center-cropped. Two configurations:
`tiny-llava` (vision_feature_layer -2, "default": 16 tokens, a 4 x 4 grid) and `tiny-llava-full` (layers [-3, -1]
concatenated, "full": 17 tokens, no grid). The registration lives outside the library, in `tests/Idrak.PluginTests`
(`LlavaPlugin`: the text decoder read by the library's text family of `text_config.model_type`, with LlamaConfig's
defaults and transformers 5's `rope_parameters`; `LlavaVisionFamily`, `ClipVisionTower`, `LlavaProjector`,
`LlavaImageEncoder`; the causal rule). Largest differences against transformers (`IDRAK_FILTER="vision contracts"`):

| | CPU | Vulkan (lavapipe) |
|---|---|---|
| pixels | within 1e-5 | within 1e-5 |
| tiny-llava: selected hidden states, projector alone, features | 5.25e-6, 3.81e-6, 9.54e-6 | 5.01e-6, 5.72e-6, 1.0e-5 |
| tiny-llava: prompt logits one pass, cached, 20 greedy steps | 1.08e-5, 1.08e-5, 8.11e-6 | 8.82e-6, 1.07e-5, 1.04e-5 |
| tiny-llava-full: hidden, projector, features | 1.1e-5, 2.86e-6, 1.38e-5 | 1.03e-5, 2.86e-6, 1.82e-5 |
| tiny-llava-full: logits one pass, cached, 20 steps | 1.17e-5, 1.17e-5, 1.16e-5 | 9.42e-6, 8.94e-6, 7.15e-6 |
| 20 greedy tokens | identical (both) | identical (both) |

(The tiny LLaVA's weights are drawn 1.5x larger than phase 0's to keep its greedy answer varied, so its float32 noise is
about twice Gemma's; features are checked within 3e-5, logits within 2e-5.) The prompt renders and expands to the
processor's ids, and the public chat API (`ChatGenerator` with the family's encoder, format and rule) answers with the
20 tokens. The kit's suite passes for the Gemma 3 and both LLaVA encoders on CPU and Vulkan. The outside plug-in test
(`tests/Idrak.PluginTests`, `VisionFamilyPluginTests`) registers a family of its own (a trivial encoder giving one token
per 16 rows: a count per image; its own format; its own attention rule), passes the kit and answers a prompt with two
images of different sizes through the public API; unregistered, its checkpoint is refused naming the registry.

**What Qwen2.5-VL would still need.** (1) M-RoPE in the decoder spec: `DecoderSpec` gains the rotary sections
(`mrope_section`, e.g. [16, 24, 24] of the head's half-dimension for time, height, width); `CausalSelfAttention` takes
positions [3, n, t] instead of one per token (text tokens (p, p, p), image tokens (t0 + ti, t0 + hi, t0 + wi) from
`ImageFeatures.Positions`), each section rotated by its axis; and positions after an image continue from the image's
largest position plus one, so a sequence's position is no longer its cache index: the decoding context keeps a
per-sequence offset ("rope delta") for every later token, cached prefill and one-token steps alike. `ImagePrefill`
already carries the ids to the decoder and refuses them until then. (2) Its ViT: window attention in most layers (rows
see their window of patches, full attention every few layers: `KeySpans` per window, which `AttentionSpans` already
runs), 2-D rotary positions in the vision tower, and the 2 x 2 patch merger; its processor's dynamic resolution
(`smart_resize`) is the encoder's own business, which the contract allows. **Encoder-decoder OCR models** (TrOCR, Donut:
an image encoder and a text decoder with cross-attention) are a separate item: the decoder has no cross-attention.

**Remaining.** A tokenizer gap seen while building the LLaVA fixture: llava-hf's chat template leaves `<s>` to the
tokenizer's post-processor (`add_special_tokens`), which Idrak's chat path does not apply (the fixture's template writes
`{{ bos_token }}` itself); transformers 5 also drops `add_bos_token` from `tokenizer_config.json`. The real LLaVA
checkpoints need that, and llava-hf's template's `selectattr` filters, checked. The real Gemma 3 model through the plug-in
is the author's to run (commands in the hand-back).

## Pan and scan, as built (2026-10-09): one image as several blocks (Gemma 3's crops, in its plug-in)

**Why.** `bakrianoo/arabic-legal-documents-ocr-1.0` read a 700 x 1000 scan squeezed into one 896 x 896 image of 256
tokens, and misread and invented text. Gemma 3's processor has an answer, pan and scan: the whole page and crops of it
at full resolution, each its own block of 256 soft tokens. It is the family's, so it lives in the Gemma 3 plug-in; the
library only had to learn, generically, that one image can be several blocks with text between them, and to carry a
family's own options per request.

**transformers, read exactly** (5.19.0 `Gemma3ImageProcessorPil` / `Gemma3ImageProcessor` and `Gemma3Processor`; the
same rule and text in 4.57.6, checked: `make_pan_scan.py --check` gives identical crops, ids, pixels and logits):

- *The grid.* For a landscape or square image (width >= height): nothing when width / height < `min_ratio_to_activate`;
  else `n = floor(width / height + 0.5)` (halves round up), `n = min(floor(width / min_crop_size), n)`, then
  `n = min(max_num_crops, max(2, n))`, one row. A portrait image the same down the height. Each crop is
  `ceil(side / n)` long; when the shorter crop side is below `min_crop_size` nothing is cropped. Crops start at
  multiples of the crop size, row by row, left to right; the last one is cut at the image's edge (numpy slicing), so it
  can be a pixel or two shorter. `max_num_crops` 1 makes one crop: the whole image again (kept, as transformers does).
- *Crops come from the original* (after `convert_rgb`, before any resize); then the whole image and each crop go through
  the same resize, rescale and normalize on their own. `pixel_values` is [images x (1 + crops), 3, S, S], each image
  followed by its crops; `num_crops` per image is popped by the processor.
- *The prompt.* Each `<start_of_image>` the chat template wrote becomes, for an image with crops,
  `"Here is the original image " + full + " and here are some crops to help you see better " + " ".join([full] * n)`
  where `full = "\n\n<start_of_image>" + 256 x "<image_soft_token>" + "<end_of_image>\n\n"`; without crops just `full`.
  So the whole image's block is followed by " and here are...", the crops' blocks are separated by one space between
  their "\n\n"s ("\n\n \n\n").
- *The mask.* `token_type_ids` marks every soft token; `get_block_sequence_ids_for_mask` starts a new block at every
  soft token whose predecessor is not one, so each crop is its own bidirectional block (causal between blocks), exactly
  the plug-in's existing "image-blocks" rule once each crop is its own run.
- *Defaults and where they come from.* `Gemma3ProcessorKwargs._defaults`: `do_pan_and_scan` False,
  `pan_and_scan_min_crop_size` 256, `pan_and_scan_max_num_crops` 4, `pan_and_scan_min_ratio_to_activate` 1.2. The
  image processor's own class defaults are None; real checkpoints' `preprocessor_config.json` writes the four keys as
  null. Through `Gemma3Processor` (and `AutoProcessor`) only call kwargs turn it on: the processor passes its defaults to
  the image processor on every call, overriding `preprocessor_config.json` (a config with `do_pan_and_scan: true` gives
  one image in 4.57.6 and 5.19.0, measured), and `processor_config.json` keys (top level or `images_kwargs`) are not read
  for it either. `Gemma3ImageProcessor` called alone uses the config's values (and fails when any of the four is null).

**Contracts (generic; nothing names Gemma or pan and scan).** In `Idrak.Abstraction.Generation`:

- `IVisionEncoder.Blocks(ImageData, VisionOptions?)` replaces `Layout(ImageData)`: the blocks of image tokens an image
  becomes, in prompt order (one for most families). `Encode(images, VisionOptions?)` returns one `ImageFeatures` per
  block, images in order and each image's blocks in order. `IVisionEncoderStages.PixelValues(image, options)` gives
  [blocks, C, H, W].
- `IImagePromptFormat.Expand(prompt, IReadOnlyList<IReadOnlyList<ImageTokenLayout>>, tokenizer)`: each image's blocks;
  two blocks never touch, so each is a run of its own and `ImagePrefill.Locate` and the attention rules need no change.
  `ImageTokenFormat` gains `Block(layout, tokenizer)` (one block's text) and `Join` (an image of several blocks written
  from its blocks' texts; without it several blocks are a `NotSupportedException` naming the format).
- `VisionOptions`: a family's own options, string values by ordinal key (`Parse("KEY=VALUE")`, `FromJson`, `With`,
  `Flag`/`Integer`/`Number`, `ThrowIfUnknown(family, accepted)` naming the accepted keys, `ToJson`).
  `ChatRequest.VisionOptions` carries them per request; `ChatGenerator` passes them to `Blocks` and `Encode`.
- In `Idrak.Models.Abstractions`: `VisionEncoderOptions.VisionOptions` (every image, at creation) and
  `PretrainedVision.VisionOptionKeys` (the keys a family takes, none by default; callers check early).
- Testing kit: `VisionEncoderSuite(..., options)` and `Conformance.CheckVisionEncoder(..., visionOptions)`: features per
  block, batch equals single per block, device equals CPU per block; a new case "a tall image and a wide one".
- `ImagePreprocessor.Parse` no longer refuses `do_pan_and_scan`: keys that are not its steps are the family's.

Decision 10 holds (`VisionOptions` sits with `ChatRequest` in Abstraction, which uses it); `abstraction inventory` and
`public API` pass (api/*.txt regenerated: Abstraction, Abstraction.Testing, Idrak, AspNetCore).

**The plug-in** (`samples/Gemma3Vision/Idrak.Gemma3Vision`): `Gemma3PanAndScan` (the rule above, `Crops(height,
width)` as rectangles, `FromConfig(preprocessor_config.json)`, `With(VisionOptions)`), `Gemma3Vision.PanAndScan` (the
folder's config over Gemma3Processor's defaults: off unless `do_pan_and_scan` is true there; *the family's decision* is
to honour a true value, as Gemma3ImageProcessor alone does, although transformers' processor ignores it),
`VisionOptionKeys` = `do_pan_and_scan` (also `pan_and_scan`), `pan_and_scan_min_crop_size`,
`pan_and_scan_max_num_crops`, `pan_and_scan_min_ratio_to_activate` (transformers' names; anything else is an
`ArgumentException` naming them; values checked: crop size >= 1, crops >= 1, ratio >= 0, the two switch names must
agree). The prompt format's `Join` writes the processor's text. The encoder's `Views(image, options)` gives the whole
image and its crops (cut from the decoded pixels); it encodes one image's views per batch (at most 1 + max crops
images), so memory grows with the crops of one image, not with a request's images.

**Options, everywhere family-neutral.** CLI: `--vision-option KEY=VALUE` (repeatable) on `run`, `chat`, `serve`/`ui`
(every image of the server), `vlm check` (over the reference's), and `alias set` (kept as `"vision_options"`, the
command line's going over the alias's per key); keys are checked against the family when the model loads (exit 2
naming them). Serve and `MapChatApi`/`MapCompletionsApi`: a request's `"vision_options": {"KEY": VALUE}` on
`/v1/chat/completions` and `/api/chat`, and a `vision_options` form field (JSON text) on `/v1/chat/upload`; an unknown
key or a bad value is a 400 naming the family's keys. `tools/vlm/compare_real.py --pan-and-scan
[--pan-and-scan-min-crop-size N --pan-and-scan-max-crops N --pan-and-scan-min-ratio X]` records them as the
manifest's `"vision_options"`, which `idrak vlm check` gives the encoder (pixels, features and the prompt per block).
The OCR sample takes `--pan-and-scan`. The web sample: a "Pan and scan" switch in the Settings card (off unless the
model's config turns it on; remembered with the other settings), its max crops, min crop size and min ratio shown while
it is on, `pan_and_scan=true/false` sent on every `/api/read`, and the figures report pan and scan on or off, the crops
and the image tokens (with the blocks).

**Reference and results.** `tools/vlm/make_pan_scan.py` → `tests/Idrak.Tests/data/vlm-pan-scan` (920 KB; reruns
write the same bytes): the tiny model of phase 0 with `pan_and_scan_min_crop_size` 32, a 60 x 170 image (3 crops, the
last a row shorter), a 300 x 60 image (5 wanted, 4 made) and a 90 x 80 one (ratio 1.125: none), each with its pixels,
features, the processor's ids, prompt logits and 20 greedy steps; plus a `compare/` folder from `compare_real.py
--pan-and-scan`. Largest differences (`IDRAK_FILTER="pan and scan"`):

| | CPU | Vulkan (lavapipe) |
|---|---|---|
| pixels | within 1e-5 | within 1e-5 |
| tall (3 crops, 120 tokens): features, prompt logits, 20 steps | 2.68e-6, 9.30e-6, 2.03e-6 | 2.34e-6, 1.01e-5, 1.43e-6 |
| wide (4 crops, 129 tokens) | 4.53e-6, 7.63e-6, 1.79e-6 | 3.81e-6, 1.04e-5, 1.91e-6 |
| square (no crops, 38 tokens) | 1.67e-6, 7.39e-6, 3.58e-6 | 9.54e-7, 7.87e-6, 3.34e-6 |
| prompt ids, greedy tokens | identical | identical |

The square image with pan and scan on gives bit for bit the pixels, features, ids and logits of pan and scan off. The
tiny model's greedy answer after the longer prompts settles on one token (91, twenty times; the square's is varied), so
the step logits are the closer check. The same answers come through `ChatGenerator`, `idrak run --vision-option`, an
alias, `/v1`, the upload and `/api/chat`; `idrak vlm check` on the `compare/` folder agrees at all 20 steps. The outside
plug-in's toy family gained a `halves=true` option (each image as two blocks with " and " between) to show the contracts
from outside the library.

**Real-model sizing** (Gemma 3 4B: 256 tokens per block, SigLIP at 896 x 896). Each crop is 256 more prompt tokens and
one more SigLIP pass: measured on the author's RTX 5070 Ti, one image encodes in 0.11 s in bfloat16 and 0.43 s in
float32, so with 4 crops about 0.55 s and 2.2 s (one batch of 5). The default rule gives a 700 x 1000 scan 2 crops (3
blocks, 768 image tokens), an A4 page at any resolution 2, a page twice as tall as wide 2, a strip three times as long
3, and at most 4 (1,280 image tokens). Memory: the KV cache is allocated for the context window, not the prompt, so
crops do not grow it (float32: 34 layers x 2 x 4 KV heads x 256 = 272 KB per token, 2.2 GB for 8,192; bfloat16 half,
int8 a quarter); the encoder's transient activations for one image's batch are about 5 x (19 MB hidden + 70 MB MLP) in
float32 per layer, under half a GB, plus 72 MB per image of span attention. The web sample's context of 8,192 fits the
worst case: 1,280 image tokens + about 60 of the processor's text + the prompt + its default 4,096 answer tokens is
about 5,500. Note that Gemma 3's local layers see 1,024 tokens back, so with 4 crops the last crop's tokens no longer
see the whole page's block on those layers (as in transformers).

## Image transforms, as built (2026-10-09): a model card's preprocessing, Pillow's bytes exactly

**Why.** The card of `bakrianoo/arabic-legal-documents-ocr-1.0` prepares every scan in Pillow before the processor:
`convert('L')`; when wider than 1,024, `resize((1024, int(h * (1024 / float(w)))), Image.LANCZOS)`;
`ImageEnhance.Contrast(...).enhance(1.5)`; on its OpenAI/vLLM path also `save(format='JPEG', quality=95,
optimize=True)` and base64. Its prompt is "Extract details to JSON." with no system message; its transformers example
calls `generate(max_new_tokens=2048)` without `do_sample`, so the fine-tune's `generation_config.json` samples
(`do_sample`, top-k 64, top-p 0.95, temperature 1), and it parses the answer with `json_repair.loads`. These steps are
the fine-tune's, not Gemma 3's and not the library's: the library offers generic transforms, and the card's pipeline is
a string an app passes (`grayscale,max_width=1024,contrast=1.5`, plus `,jpeg=95` for the vLLM path).

**Contract and registry** (decision 10: `ChatRequest` in Abstraction carries a pipeline, Nlp runs it, AspNetCore
parses it, core implements the library's; so all in `Idrak.Abstraction.Data`, beside `ImageData`; the inventory and the
public API files are updated):

- `IImageTransform` (`Name`, `Summary`, `Keys`: the option keys besides its value, `Check(step)`, `Apply(ImageData,
  step)`): decoded pixels in ([C, H, W] in [0, 1]), decoded pixels out.
- `ImageTransformStep` (`Name`, `Value`, `Options`; `Number`, `Integer`, `Option`, `ThrowIfValue`,
  `ThrowIfUnknownOptions`) and `ImageTransformPipeline` (`Steps` in the user's order, `Parse(text)`, `FromJson`,
  `Then`, `Contains`, `Apply`, `ToJson`, `Empty`; equal by their text). Text: comma-separated `name` or `name=value`;
  an item naming an option of the step before it belongs to that step (`max_width=1024,resample=bicubic`); `none` or
  empty: no steps. JSON: that text, or an array of texts and objects `{"name", "value", ...options}`. An unknown
  name, an option of another step or before any step, and a bad value are an `ArgumentException` naming the
  registered transforms (and the step's keys).
- Registry `ImageTransforms` on a `SlotTable` (`Register`, `Unregister`, `Find`, `Get`, `Names`, `Default`, `Origin`,
  `SetPolicy`, `Describe`; guarded: an app's transform over a library name falls back or is shadowed per call). The
  library's are library defaults, registered by core's `LibraryRegistrations` on first use (`LibraryImageTransforms`).
  A test registers its own (`invert`) from outside the library, runs it in a pipeline and unregisters it.
- `ChatRequest.ImageTransforms` (null: the model's; empty: none) beside `VisionOptions`, and `ChatImages.Transforms`
  (every image's, replaced by a request's); `ChatImages.Read(image, request)` decodes then transforms, and
  `ChatGenerator` reads every image through it (`RenderPrompt` and `Stream`), so blocks, prompt and features all see
  the transformed image. Not a family option: nothing in it knows Gemma 3.

**The library's transforms** (`Idrak.Data.PillowImageOps`, public, and the registry's names), each copying Pillow 12's
C: 8-bit planes read as `ImagePreprocessor` reads them (Pillow's bytes), results given back as byte / 255.

| Name | Pillow | How it is exact |
|---|---|---|
| `grayscale` | `convert("L")` | `ChatImageDecoder.Grayscale(ImageData)` (new overload; the `ChatImage` one and the preprocessor share its arithmetic): (19595 R + 38470 G + 7471 B + 32768) >> 16 |
| `max_width=N`, `max_height=N` (`resample=lanczos` default, `bicubic`, `bilinear`, `box`, `hamming`) | the card's `ratio = N / float(w)`, `int(h * ratio)`, `resize(..., resample)` | the ratio in double and truncated as Python does (at least 1); `ImagePreprocessor`'s `ImagingResample` (22-bit weights, two passes, 8-bit rounding) reused; an image within the limit is untouched |
| `contrast=F` | `ImageEnhance.Contrast` | the mean of the L image's histogram (`sum / count` in double, `int(mean + 0.5)`), a flat image of it (grey to RGB for colour), then `Blend.c`: `in1 + alpha * (in2 - in1)` in float32 with alpha cast to float, truncated to a byte, clipped when alpha is outside [0, 1]; alpha 0 and 1 copy |
| `brightness=F` | `ImageEnhance.Brightness` | the blend with black |
| `sharpness=F` | `ImageEnhance.Sharpness` | `Filter.c`'s 3 x 3 `SMOOTH` (weights 1/13 and 5/13 in float32, offset 0.5, sums in Pillow's order, clamp then truncate, border rows and columns copied, images under 3 pixels copied), then the blend |
| `autocontrast[=CUTOFF]` (`ignore=V`, `preserve_tone=true`) | `ImageOps.autocontrast` | per-channel histograms (or the L one with `preserve_tone`), the cutoff with CPython's float floor division, `int(i * scale + offset)` clipped, a lookup table |
| `jpeg=Q` (`subsampling=4:2:0` default for colour, `4:2:2`, `4:4:4`) | `save("JPEG", quality=Q, optimize=True)`, then open | the new `JpegEncoder` (below), then the library's JPEG decoder (Pillow's pixels since phase 2) |

**The JPEG encoder** (`Idrak.Data.JpegEncoder.Encode(ImageData or planes, quality, optimize, JpegSubsampling)`,
public; not an `IImageCodec`, which decodes only): libjpeg-turbo 3.1 as Pillow drives it. JFIF 1.01 header; the Annex
K tables scaled by `jpeg_quality_scaling` with `force_baseline` (1 to 255); grey as one component (1 x 1, Pillow's
default for L), colour as YCbCr with `jccolor.c`'s 16-bit tables; `jcsample.c`'s h2v1 (bias 0, 1) and h2v2 (bias 1,
2) averaging over edges replicated as `jcprepct.c`/`expand_right_edge` pad them; `jfdctint.c`'s islow forward DCT;
`jcdctmgr.c`'s quantization by reciprocals for a 16-bit DCTELEM (libjpeg-turbo built with SIMD, as Pillow's wheels;
`compute_reciprocal`, its correction and shift); `jccoefct.c`'s dummy blocks at the MCU edges (zero AC, DC of the
block before, or of the row above's last block in the MCU); standard Huffman tables, or with `optimize` the image's
own from `jpeg_gen_optimal_table` (with the reserved symbol and the over-16-bit folding); markers in libjpeg's order
(DQT per table, SOF0, DHT DC/AC per table, SOS), byte stuffing and one-bits padding.

**Measured against Pillow 12.3.0 / libjpeg-turbo 3.1.4.1** (`tools/vlm/make_image_transforms.py` →
`tests/Idrak.Tests/data/image-transforms`, 2.0 MB; reruns write the same bytes; `IDRAK_FILTER="image transforms"`):

- **All 238 transform cases give Pillow's bytes exactly (max difference 0, by SHA-256 of every output, with the
  largest difference reported from the PNG when one differs):** 29 pipelines (every transform, every resample, factors
  0, 0.5, 1, 1.5, 3.25, cutoffs and `ignore`, `preserve_tone`, a limit larger than the image, the card's pipeline and
  the reversed order `contrast=1.5,grayscale`, `jpeg` at 30, 75, 95 and with 4:4:4 and 4:2:2) on 8 inputs (the colour,
  grey, RGBA phase-0 images, a 17 x 13 4:2:0 JPEG, a 45 x 37 grey progressive JPEG, an EXIF JPEG, 37 x 23 and 61 x
  203), plus 6 pipelines on the 1,654 x 2,339 scan (the card's, with `jpeg=95`, bicubic, bilinear `max_height` with
  sharpness and autocontrast, `jpeg=95` in colour and grey).
- **The JPEG encoder: all 51 files are byte for byte Pillow's** (so their pixels are too): the 8 small inputs at
  quality 95, 75, 10 and 100 (4:2:0), 90 (4:4:4) and 85 (4:2:2), the scan at 95 (4:2:0) and 80 (4:4:4) in colour and at
  95 in grey (772 KB). Optimized tables are smaller than the standard ones and leave the pixels unchanged (asserted).
- The one caveat, outside these files: the enhancers' blend and the smoothing filter are float32 in Pillow's C. Pillow's
  x86-64 wheels do not fuse multiply-adds there and neither does .NET; a Pillow compiled to contract them (some ARM
  builds) could differ by 1 at a few pixels. Integer steps (L, resize, JPEG) have no such dependence.
- Speed (this container, 4 threads): the card's pipeline on the 1,654 x 2,339 scan, about 90 ms (the web sample's
  figure; Pillow's resize alone is about as fast).

**Through the stack** (all family-neutral; the card's pipeline is only ever a string):

- CLI: `--image-transform P` (repeatable, joined in order) on `run`, `chat`, `serve`/`ui` and `vlm check`; `alias set
  --image-transform P` keeps it as `"image_transforms"` (the command line's replaces the alias's; `none` for none).
  `--grayscale` (and an alias's `"grayscale": true`) is now sugar for the `grayscale` step first (when the pipeline has
  none): the encoder is built without its own grey, and every grey path gives the reference's pixels as before. `run -j`
  reports `image_transforms`; `serve`'s announcement and `--json` name each model's transforms.
- Serve and `MapCompletionsApi`/`MapChatApi`: `"image_transforms"` (text or array) on `/v1/chat/completions` and
  `/api/chat`, and an `image_transforms` form field on `/v1/chat/upload` (`ImageRequests.WithImageTransforms`); a
  request's replace the server's; an unknown name or bad value is a 400 naming the registered transforms.
- `vlm check`: the manifest's `"image_transforms"` (written by `compare_real.py --image-transform`) run on the image
  before the family's preprocessing, or `--image-transform` instead.
- Web sample: a "Model-card preprocessing" switch in the Settings row (remembered), showing Max width (1024), Contrast
  (1.5), Resample (Lanczos) and an optional JPEG round trip (the vLLM path); a "Card prompt" button ("Extract details
  to JSON.", no system message); "Use the model's sampling settings" (temperature 1, top-k 64, top-p 0.95) and
  "Greedy" buttons (greedy stays the default). The page sends `image_transforms` (and `model_card`) to `/api/read`;
  the preview is the server's own transformed pixels (`POST /api/preview`, an exact BMP; on failure the upload with
  the browser's grey filter, labelled approximate); the figures give the transforms, the size after them and their
  time. Pretty JSON tries a lenient repair when strict parsing fails (trailing commas, a string, array or object left
  open, a dangling key, smart quotes as delimiters) and says "JSON repaired"; the Raw tab keeps the model's text. No
  JSON-repair dependency anywhere. The OCR sample takes `--image-transform`.
- Tools: `tools/vlm/image_transforms.py` (the Pillow reference of the syntax); `compare_real.py --image-transform`
  (after the EXIF orientation and `--grayscale`; recorded as `"image_transforms"`); `tools/vlm/ocr_transformers.py`,
  standalone: transformers' answer for a scan with the same transforms, the card's prompt by default, `--prompt`/
  `--prompt-file`, `--system`, `--grayscale`, `--pan-and-scan`, `--vision-float32`, `--max-tokens` (2048), greedy
  unless `--sample` (the model's generation config; `--seed`), the answer as UTF-8 at `--out` and a `.json` of figures
  beside it (sizes before and after, tokens, blocks, timings, versions, and whether the answer is JSON, repaired by
  `json_repair` when that package is installed).

**Tests** (CPU): `image transforms` (5 + the tiny model): the Pillow cases above; pipeline parsing (text and JSON,
order, case, round trip, 15 refusals with their messages, the card's truncation arithmetic); the registry (library
defaults, an outside transform); and the tiny Gemma 3 with `compare_real.py --image-transform
grayscale,max_width=64,contrast=1.5` on `image.jpg` (`tests/Idrak.Tests/data/vlm/compare/transformed`): a request's
transforms give transformers' pixels and features within 1e-5 and its 20 greedy tokens, as do `ChatImages.Transforms`,
and an empty pipeline on the request turns them off. `cli images` (vlm check on the transformed folder agrees at all
20 steps; without the transforms its pixels do not), `aspnetcore: images` (`/v1` with text and array forms, the
upload, `/api/chat`: the reference's 20 tokens; refusals), `cli serve: images` (`run --image-transform`, `--grayscale`
plus two `--image-transform`, an alias, `none` over it, an unknown name; `serve --grayscale --image-transform` answers
an upload as `run`, and a request's own transforms replace the server's). Also passing: `image preprocessing`, `jpeg`,
`cli serve`, `aspnetcore`, `vision contracts`, `pan and scan`, `outside plug-in`, `cli run`, `cli models`,
`cli design`, `cli polish` (it needed two old fixes: `vlm check`'s summary was over 120 characters and its help did not
describe `--context` and `--adapter`), `abstraction inventory`, `public API`.

**Not done.** The real model with the card's pipeline is the author's to run (commands in the hand-back). The
transforms run once more in the web sample than they need to (once for its figures and preview, once in the chat
generator), which costs about 90 ms a request.

## Performance targets (author's RTX 5070 Ti, the real 4B model)

- The vision encoder (4,096 tokens, 27 layers) runs once per image: under half a second in bfloat16. This needs 3a's
  tiled bidirectional attention: composing the scores in full would hold 4,096² x 16 heads (1 GB in float32) per layer.
- **Measured 2026-10-08 on the author's card** (random weights at the real sizes, `--bench-vision`): one 896 x 896 image
  to [1, 256, 2560] in 0.43 s in float32 and **0.11 s with bfloat16 tensor cores** (target met). `--bench-spans` (4,096
  tokens, 16 heads, dim 72): span kernel 8.2 ms against composed 9.9 ms in float32, 1.6 ms against 5.8 ms in bfloat16,
  72 MB against 2,120 MB of device memory; the measured choice took the span kernel both times. All of "attention",
  "attention spans", "conformance kit", SigLIP and "image prefill" pass on CUDA.
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
2. ~~Content parts on `ChatMessage` (like `/v1`) or a separate image list.~~ **Decided 2026-10-07: content parts as a
   contract** (`ChatPart`, registered kinds in `ChatParts`, `IChatModel.PartKinds`), open to new kinds (audio, video,
   documents) without changing the contract; see "Phase 1 as built".
3. ~~Pan-and-scan for tall pages (Gemma 3's crops of a long document): first version or phase 9.~~ **Done 2026-10-09** in the Gemma 3 plug-in, off by default (see "Pan and scan, as built").
4. Where an `ocr` tool or pipeline lives (phase 9): plan 10 puts OCR in `Idrak.Vision`, but one built on a language model
   needs Nlp, which Vision does not reference; in Nlp (or Mcp and the CLI) unless Vision's OCR is a model of its own.
