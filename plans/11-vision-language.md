# Plan 11: images into language models (Gemma 3 first)

**Status:** planned 2026-10-07, revised the same day against the code; phase 0 (the reference) done 2026-10-08, phase 4 (loading, text side) done 2026-10-08 on CPU and Vulkan, phase 3b (the SigLIP encoder and the projector) done 2026-10-08 on CPU and Vulkan, phase 1 (contracts) done 2026-10-07 on the CPU; plan 10's wave 4 is done. Asked for to run `bakrianoo/arabic-legal-documents-ocr-1.0`, a fine-tune of Gemma-3-4B-IT that reads scanned
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
| Image tokens in the prompt: each image becomes `<start_of_image>`, 256 soft tokens whose embeddings are the projected image features (not scaled by √hidden like text embeddings), `<end_of_image>`; the soft tokens of one image attend to each other in both directions, the rest of the prompt stays causal | missing |
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
| 5 | **Image tokens in the decoder.** At prefill, the soft tokens' embeddings are replaced by the image features; each image's soft tokens see each other (3a's image-block rule, on top of the sliding window of the local layers); decoding afterwards uses the KV cache as today (the image costs nothing more per generated token). Several images in one prompt | the tiny model's logits with an image and its 20 greedy tokens match the reference (CPU and Vulkan) |
| 6 | **Chat with images in Nlp.** The chat template's image parts expand to the image tokens (the Jinja engine walks content parts); generation takes images; an image is encoded once per conversation (kept with the conversation by content hash) | a two-turn chat about one image encodes it once and matches the reference's first turn |
| 7 | **The command line.** `idrak chat <model> --image FILE` (repeatable; also `/image FILE` inside a chat, like `/file`); the one-shot `idrak run <model> --image FILE "prompt"` (with `--schema` for the structured answer this model gives, `-j` for JSON); `--grayscale` (or the setting stored with a pulled model) | CLI tests pass with the tiny model; on the author's RTX 5070 Ti the real model reads a sample scan and its greedy output matches transformers' for the first 100 tokens |
| 8 | **Images in chat, after the command line is proven** (phase 7 done on the real model). The engine hosts the model as a chat model that accepts images (`IChatModel`, saying which inputs it takes; the image is encoded once per conversation, as in phase 6); `MapChatApi` takes images in its messages; `MapCompletionsApi` (`/v1/chat/completions`) accepts OpenAI's `image_url` content parts (data URLs and, when allowed, http URLs); `idrak serve` serves it. The same preprocessing (and grayscale setting) as the command line | an OpenAI client sends a scan as an `image_url` part and gets the same answer as `idrak run --image`; streamed and not; the aspnetcore tests cover a tiny model with an image |
| 9 | **After that.** An MCP `chat` tool with images (and an `ocr` tool if useful); pan-and-scan for tall pages; other families (Qwen2.5-VL); fine-tuning with images (LoRA on the text decoder, encoder frozen) | decided per item after phase 8 |

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
2. ~~Content parts on `ChatMessage` (like `/v1`) or a separate image list.~~ **Decided 2026-10-07: content parts as a
   contract** (`ChatPart`, registered kinds in `ChatParts`, `IChatModel.PartKinds`), open to new kinds (audio, video,
   documents) without changing the contract; see "Phase 1 as built".
3. Pan-and-scan for tall pages (Gemma 3's crops of a long document): first version or phase 9.
4. Where an `ocr` tool or pipeline lives (phase 9): plan 10 puts OCR in `Idrak.Vision`, but one built on a language model
   needs Nlp, which Vision does not reference; in Nlp (or Mcp and the CLI) unless Vision's OCR is a model of its own.
