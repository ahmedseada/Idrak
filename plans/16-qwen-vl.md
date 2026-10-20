# Plan 16: one contract for every vision-language family; Qwen-VL as a plug-in

The owner's request (October 2026): use Qwen as a base model in the legal OCR app, for reading and fine-tuning, and
keep Idrak family-agnostic: every vision-language family does the same jobs behind one contract with generic outputs,
and a family is a plug-in. Qwen's text models (Qwen2, Qwen3) already load; an OCR base must read images, so the family
to add is **Qwen-VL**: Qwen2.5-VL first (3B and 7B Instruct), Qwen2-VL on the same plug-in.

**Status (2026-10-13):** steps 0, 1, 2, 3, 5, 6 and 8 built and checked on the CPU against transformers 5.19 (see "Built,
2026-10-13"); step 4 (Gemma 3's text decoder into its plug-in) done 2026-10-20 by plan 26's phase P6; steps 7 and 9 are
the owner's runs on the card.

## The principle

The library holds **contracts, generic decoder capabilities and generic rules**. A **family** is one plug-in that
registers everything that is its own, in one call: its vision part and its text decoder. The decoder, the chat path,
the tuner, the feature cache, the command line and the apps see only the contracts and their generic outputs; none of
them knows which family it runs. A new family adds a plug-in; it changes the library only when it needs a capability no
family needed before, and then that capability is generic (named by a rule, usable by any family), never the family's.

## The contract: the jobs every family does

A family's vision part is a `PretrainedVision` (its descriptor, read from the checkpoint by its `IVisionFamily`). Each
job is a contract with a generic output; the descriptor names the family's choice for each.

| Job | Contract | Generic output | Gemma 3 | Qwen-VL |
|---|---|---|---|---|
| Image to features | `IVisionEncoder.Encode` (the family's own preprocessing and tower inside) | `ImageFeatures`: one embedding per image token, its layout | 896 x 896, SigLIP, pooled | `smart_resize`, its ViT, the 2 x 2 merger |
| Tokens per image | `IVisionEncoder.Blocks` | `ImageTokenLayout`: a count, and the token grid when there is one | 256 (16 x 16); pan and scan adds blocks | a count per image ([1, rows, columns]) |
| Image tokens in the prompt | `IImagePromptFormat.Expand` | token ids | `<start_of_image>` + 256 | `<|vision_start|>` + N pads + `<|vision_end|>` |
| How image tokens attend | `IImageAttentionRule` (registry `ImageAttentionRules`) | `KeySpans` | bidirectional within the image | causal (the library's) |
| **Where image tokens sit (new)** | **`IImagePositionRule` (registry `ImagePositionRules`)** | **each token's rotary positions: one axis, or several** | **sequential (the library's; as today)** | **grid (the library's): (s, s + row, s + column)** |
| What trains | `IVisionTuningPart` | the trainable tensors | projector, tower | merger as `projector`, tower |
| Its own options | `VisionOptionKeys`, `VisionOptions` per request | — | pan and scan | `min_pixels`, `max_pixels` |
| **Its text decoder (moved into the family)** | **`PretrainedArchitectures.Register`, from the plug-in** | `DecoderSpec` and tensor names | Gemma 3 text in the VL checkpoint's naming | Qwen2 with rotary sections |

### The one new contract: the image position rule

Today every family places image tokens at the prompt's next positions, one each, like text; `ImageFeatures.Positions`
was left as a slot and is refused (`ImagePrefill.CheckPositions`, `TuningVision`). The slot becomes a rule, the same
pattern as attention:

- `IImagePositionRule { Name; Axes; Place(prompt length so far, layout) → positions [axes, tokens], next position }`,
  in `Idrak.Abstraction.Generation` beside `IImageAttentionRule`, with a `SlotTable` registry `ImagePositionRules`
  (register, unregister, find, get with "not registered", default, origin, policy, the guarded wrapper).
- The library's rules, both generic: **`sequential`** (one axis; the next positions: exactly today's behaviour, every
  existing family) and **`grid`** (three axes from `ImageTokenLayout.Grid`: an image starting at s has (s + frame,
  s + row, s + column); the text after it continues from the largest position so far plus one). Text tokens have the
  same position on every axis.
- `PretrainedVision.Positions` (the family's rule; `sequential` unless it says), as `Attention` is.
- `ImageFeatures.Positions` stays for a family whose placement is not a function of the layout (none yet).

### The generic decoder capability it needs: multi-axis rotary positions

A model whose rule has several axes needs a decoder that rotates by several axes. This is a capability of the decoder,
selected by the spec, not by a family:

- `RopeSettings.Sections`: how the rotary half-dimension splits among the axes (Qwen's `mrope_section`, [16, 24, 24]
  for a head of 128); `RopeScalings` reads Qwen's `"type": "mrope"` as sections with unscaled frequencies.
- One op turns a sequence's [axes, tokens] positions into its rotary rows (each section's cos and sin from its axis),
  made once per forward pass and shared by every layer; the existing rotary kernels then run on those rows (no new
  attention kernel). CPU, CUDA, Vulkan, HIP kernels under rules 80-87.
- A sequence's position is no longer its cache index: `DecodingContext` keeps each sequence's offset (the rule's next
  position minus the tokens so far), added in cached prefill and every decode step.
- Packed training carries [axes, tokens] positions, restarting per packed sequence; the rotary backward uses the same
  rows.
- With one section (every model today) nothing changes: bit for bit.

### One registration per family

A family plug-in's `Register()` registers, together: its `IVisionFamily`, its text decoder's architecture
(`PretrainedArchitectures.Register`, with the library's public helpers such as `LlamaStyle` and `CommonSpec`), and any
rule or option of its own. The library keeps plain text families (Llama, Mistral, Qwen2, Qwen3, Gemma 1 to 3, the
mixtures of experts) as its defaults; a vision-language checkpoint's architecture (`Gemma3ForConditionalGeneration`,
`Qwen2_5_VLForConditionalGeneration`) belongs to its family's plug-in. The testing kit gains a family check: a plug-in
registers both parts, its encoder passes `CheckVisionEncoder`, and its prompt, attention and position rules resolve.

### Where one contract stops (named, not built)

Some families change what the decoder computes, not only what enters the prompt: Qwen3-VL adds image features to the
hidden states of several decoder layers (DeepStack); TrOCR, Donut and Llama 3.2 Vision read the image by cross-attention.
Each is a further generic output of the contract (features with the layers to add them at; features for cross-attention
with the decoder blocks to take them), in a later plan. Qwen2.5-VL and Qwen2-VL need neither.

## What Qwen-VL does (transformers, read exactly in step 0)

- **Processor.** `smart_resize`: height and width rounded to multiples of 28 (patch 14 x merge 2), the pixel count kept
  between `min_pixels` and `max_pixels`, bicubic resize (up or down; `PillowImageOps.Resize` has both); rescale by 1/255,
  CLIP's mean and std; the image repeated to 2 frames and cut into 14 x 14 patches of 3 x 2 x 14 x 14 = 1,176 values,
  ordered so each 2 x 2 group is contiguous. A 1,024 x 1,448 page is about 7,700 patches and 1,900 image tokens: the
  page's own resolution, not a fixed 256, which suits dense legal pages.
- **Tower (Qwen2.5-VL).** Patch embedding as a matrix product, 2-D rotary positions (half by row, half by column: the
  same rotary-rows op with two axes), blocks of RMSNorm, attention and a SwiGLU MLP; attention within 8 x 8-patch
  windows except in the full-attention blocks, as `KeySpans` tables run by `AttentionSpans` (tiled over the keys, no
  score matrix stored; head size 80 has kernels), the patches reordered by window and back. **Merger**: RMSNorm, the
  2 x 2 neighbours joined, Linear, GELU, Linear to the decoder's width. **Qwen2-VL**: LayerNorm, quick-GELU MLP, full
  attention in every block, a LayerNorm merger.
- **Decoder.** Qwen2 (`QkvBias`) with `mrope_section`; image tokens attend causally; positions by the `grid` rule.

## Steps

| # | Step | Where | Done when |
|---|---|---|---|
| 0 | **Reference fixtures.** `tools/vlm/make_tiny_qwen_vl.py`: tiny random Qwen2.5-VL and Qwen2-VL checkpoints (a window and a full block, sections scaled down, the real special-token ids); for a colour image, a grey one, a tall page and a small one (upscaled): the processor's pixels, grid, expanded ids, `get_rope_index`'s positions and `rope_deltas`, each tower stage, the merger, logits in one pass and cached, 20 greedy tokens; a two-image prompt; a tuning fixture (loss and adapter gradients). Which Arabic OCR fine-tunes use which family, checked. Needs Python with `transformers` and `torch` (CPU), not in this container yet | `tools/vlm`, `tests/Idrak.Tests/data/vlm` | Fixtures written; reruns give the same bytes |
| 1 | **The position rule** (contract, registry, `sequential`, `grid`, `PretrainedVision.Positions`); `ImagePrefill` and `TuningVision` place image tokens through it (their refusals go) | `Idrak.Abstraction`, `Idrak`, `Idrak.Nlp` | `sequential` gives every existing fixture bit for bit; `grid` gives the fixture's position ids and offsets; inventory and API dump current |
| 2 | **Multi-axis rotary positions** for inference: `RopeSettings.Sections`, `mrope` in `RopeScalings`, the rotary-rows op and its kernels, the offset in `DecodingContext`; every rotary path (one pass, cached prefill, decode steps, the fused norm-and-rotary kernels taking the rows or stepping aside) | `Idrak.Abstraction`, `Idrak`, `Idrak.Gpu` | One section equals today bit for bit; the tiny Qwen2.5-VL decoder with the fixture's positions matches transformers' logits and 20 tokens |
| 3 | **Multi-axis positions in training**: packed sequences with axes, the rotary backward | `Idrak.Nlp`, `Idrak` | The tuning fixture's loss and gradients; a packed batch equals its sequences one by one |
| 4 | **One registration per family**: the family check in the testing kit; `Gemma3ForConditionalGeneration`'s text decoder (its layouts and names) moves from the library's built-ins into the Gemma 3 plug-in, registered by its `Register()`; the library's helpers a plug-in needs made public; the outside plug-in test registers a whole family (vision part, text decoder, a position rule of its own) from outside the library | library, `samples/Gemma3Vision`, `tests/Idrak.PluginTests` | Every Gemma 3 fixture and app test unchanged with the plug-in registered; the outside family loads, answers and tunes with no library change |
| 5 | **The Qwen plug-in: Qwen2.5-VL** (`samples/QwenVision/Idrak.QwenVision`): its `IVisionFamily` and descriptor (prompt format, attention `causal`, positions `grid`, options `min_pixels` and `max_pixels`, tuning parts), the processor, the tower on `AttentionSpans` (several images of different sizes in one pass by their spans), the merger, `IVisionEncoderStages`, and its text decoder's registration (Qwen2 style with sections, the old and new tensor names) | plug-in | Pixels exact; each tower stage and the features within float32 noise; the kit's family check on CPU, CUDA and Vulkan; prompt ids equal the processor's; the chat API answers the fixture's 20 tokens |
| 6 | **Qwen2-VL** in the same plug-in (its tower and merger) | plug-in | Its fixture as step 5's |
| 7 | **Speed and memory on the device**: a 1,024-wide page's tower, its ~2,000-token prefill, decode speed, peak memory, cold and warm apart; what is slow fixed in the library | library | Numbers recorded here; the owner's CUDA runs |
| 8 | **The legal OCR app**: the Qwen plug-in registered beside Gemma 3's; Qwen2.5-VL 3B and 7B in the switch (Qwen's repositories are not gated); a "max pixels" field in the Read settings; the Fine-tune tab tunes them unchanged (`ConversationTuning`); "As the model sees it" shows the resized page and its token grid | `samples/Idrak.Samples.LegalOcr` | App tests with the tiny Qwen2.5-VL: read, evaluate, tune, read with the adapter |
| 9 | **Owner's commands**: the CUDA filters for steps 1-6; the app with Qwen2.5-VL-3B: CER on 20 evaluation pages against bakrianoo's model, a short tuning run | — | Given |

**Order.** 0, then 1 and 2 (the contract and the capability; the bulk of the risk: every rotary path), 3, 4 (independent
of Qwen: it can go first or beside 1-3), 5, 6, 7 beside 5 on the owner's card, 8 last.

## Built, 2026-10-13 (`abstraction`)

| Step | State | What and where |
|---|---|---|
| 0 | Done | `tools/vlm/make_tiny_qwen_vl.py` (transformers 5.19, CPU): tiny random Qwen2.5-VL (tied, windows 0 and 2, full blocks 1 and 3) and Qwen2-VL (untied), each in transformers 5's config.json and the hub's flat 4.x one (reloaded by transformers to the same logits), three pages (shrunk 32 x 48, tall 56 x 32 with partial windows, enlarged 32 x 40 from 20 x 15), reference outputs for pixels, tower, features, positions and rope deltas, prompt and step logits, 20 greedy tokens, two pages, text only, a tuning example. Reruns give the same bytes. In `tests/Idrak.Tests/data/vlm-qwen` (3.9 MB) |
| 1 | Done | `IImagePositionRule`, `ImageRun`, `ImagePositionRules` (SlotTable) with `grid` (transformers' `get_rope_index` for images) in `Idrak.Abstraction.Generation`; recorded in `shared-contracts.txt`. "Sequential" is images without axis positions, as before. `ImageFeatures.Positions` are relative to the image's start |
| 2 | Done | `RopeSettings.Sections` (`RotarySections`, value-compared, kept in the spec's JSON). A prompt whose images carry axis positions gets per-token cos and sin rows built on the host once per pass and setting, read through an identity index by every existing rotary kernel (fused ones step aside for that pass); text tokens' rows equal the plain table's. `DecodingContext.RotaryOffsets` (per row, set by `ImagePrefill.ForwardCached`), `RotaryPositions` computed on the device from `Position` plus a device copy of a uniform offset, so recorded decode steps stay valid; per-row offsets fall back to host positions and are not recordable (`DecodingContext.Recordable`, which `TextGenerator` now checks). `Truncate` restores the offsets before a pass's images and refuses a cut inside them |
| 3 | Done | `IVisionEncoder.AxisPositions(layout)` (default null) carried by `TrainingImage.AxisPositions` into the step's `PromptImage`s; packed rows place each sequence from 0; `ImagePrefillInputs` keeps fixed rotary buffers for recorded passes. `TuningVision` no longer refuses multi-axis families (its prefetch now also covers the first step) |
| 4 | Done 2026-10-20 (plan 26, P6) | Gemma 3's text decoder (its two namings) is `Idrak.Text.Plugins.Gemma`'s, with Gemma 1 to 3 text; the vision part moved from `samples/Gemma3Vision` to `Idrak.Vision.Plugins.Gemma`, whose `Gemma3VisionPlugin.Register()` registers the text decoder when missing; the library keeps no Gemma family (plan 26's "P6 as built") |
| 5, 6 | Done | `src/Idrak.Vision.Plugins/Qwen`, namespace `Idrak.Vision.Plugins.Qwen` (moved from `samples/QwenVision/Idrak.QwenVision` by plan 26 P5, 2026-10-20; README there): both families, both config layouts, `visual.` and `model.visual.` tensors, the processor, the tower on span attention, the merger, the encoder with axis positions, tuning parts (merger, tower) with export and import under the checkpoint's names, the text decoder |
| 7 | Owner | Speed and memory of a real page on the card |
| 8 | Done | The legal OCR app registers the plug-in; Qwen2.5-VL 3B and 7B in the switch (`MaxPixels` 1,003,520 each, the page's "Most pixels" field per read); a read sends only the vision options the loaded family takes (Gemma's pan and scan is no longer sent to other families) |
| 9 | Owner | CER on the evaluation pages against bakrianoo's model; a short tuning run |

Since plan 26 P5 (2026-10-20) the plug-in is `QwenVisionPlugin` ("qwen-vl") of the package Idrak.Vision.Plugins, which
the idrak tool registers at startup (no `-P`); `QwenTextDecoder.Register(variant)` became `QwenTextDecoder.Architecture()`
(the plug-in registers it), so the package's inventory has no registry of its own.

Checked on the CPU: `qwen vl` (6 tests): every reference output within 4.2e-5, the same greedy tokens, rotary offsets
equal to transformers' rope deltas; the legal OCR app's tests (10, the new Qwen read, pixel budget and tuning run among
them); and the groups the change touched (image prefill, vision contracts and tuning, conversation tuning, decoding,
generation, pretrained, decoder, image reader, the API guard and inventory).

## Decisions for the owner (each with a default)

- **First model:** Qwen2.5-VL-3B-Instruct (default); the 7B with int8 or int4 weights after.
- **Qwen2-VL too:** yes (default), step 6.
- **Max pixels in the app:** the processor's default (read in step 0), with the field to lower it.
- **A vision-language checkpoint without its plug-in** (step 4): today `Gemma3ForConditionalGeneration` loads as a text
  model without the Gemma 3 plug-in (images then fail with "not registered"). Once the text decoder moves into the
  plug-in, the checkpoint needs the plug-in to load at all. Default: accept it (the apps and the command line's `-P`
  already register it); the alternative keeps the text decoders of known VL checkpoints in the library, which breaks
  "one registration per family". **Taken (plan 26, P6, 2026-10-20):** the default; without `Idrak.Text.Plugins` the
  checkpoint is refused as "not registered" naming the package, and with the text plug-in but not the vision one it is
  refused naming the vision family (as before); the idrak tool registers both packages at startup.

## Rules that apply

Family-agnostic (no family code in the library; an unregistered family or rule fails with "not registered"); contract
plus registry for the position rule, the API dump and inventory current; results proven unchanged (every existing model
and fixture bit for bit after steps 1, 2 and 4); speed and memory first (the rotary rows made once per pass, the tower's
full blocks never store a score matrix, measured on the device cold and warm apart); card-agnostic (no model size tied to
a memory size in code or docs).
