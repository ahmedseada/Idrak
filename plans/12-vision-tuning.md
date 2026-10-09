# Plan 12: fine-tuning vision-language models (images in `FineTuner` and `idrak tune`)

**Status:** planned 2026-10-09, against the code at `abstraction` 4ac4fa4. Phase 0 (the reference) built 2026-10-09;
phases 1 to 4 built 2026-10-09; phase 6 built 2026-10-09 (CPU; the graph paths written, not run on a GPU); 5 not yet.

**Goal.** Fine-tune a vision-language model on pages and their answers with LoRA (or QLoRA on an int8/int4 base),
the way `bakrianoo/arabic-legal-documents-ocr-1.0` was trained in LlamaFactory (LoRA on the language model, the vision
tower frozen), and run the result through `idrak run`, `chat` and `serve`. The first data: that model's own
`data/train.json` and `data/val.json` (LlamaFactory ShareGPT, an `<image>` placeholder and an `images` list).

**Rules that hold for every phase** (see `CLAUDE.md`):

- **Card-agnostic.** No phase assumes a device, a vendor or a memory size. Memory is measured on the running device and
  the tuner's existing out-of-memory ladder (`RecomputeFeedForward`, then `Checkpointing`, then smaller batches) decides;
  speed choices (span kernel or composed attention) are measured, as `AttentionFastest` does for generation. Every new
  kernel gets a reference on the CPU and the shared device cases, so CPU, Vulkan and CUDA run the same tests.
- **Abstraction.** Everything new is a contract with a `SlotTable` registry (decision 10), in the inventory, behind the
  public API guard. The tuner speaks only to `IVisionEncoder` / `IVisionEncoderStages`, `IImagePromptFormat`,
  `IImageAttentionRule` and `VisionFamilies`, never to a family.
- **Plug-ability.** The library registers no vision family. Tuning a model whose family is not registered fails loudly
  with the registry's message ("vision family ... not registered (load its plug-in with -P)"), exactly as `run` does.
  Gemma 3 comes from `samples/Gemma3Vision/Idrak.Gemma3Vision`; a second family (the tiny LLaVA in
  `tests/Idrak.PluginTests`) proves the path is generic.

## What Idrak has, and what is missing

| Part | Today |
|---|---|
| LoRA / DoRA / QLoRA fine-tuning of decoders (`FineTuner.Train`, `idrak tune`, `tune.json`), int8/int4/bf16 bases, losses sft/dpo/orpo/simpo/distillation (`FineTuningLosses`), schedules, adapters saved and loaded (`SaveAdapter`, `--adapter`) | supported (text) |
| Loss on the assistant's turns only (`ChatTranscriptEncoder` finds them from the template; `TrainingSequence.Trained`) | supported (text) |
| Device memory handled by measurement: `Checkpointing`, `RecomputeFeedForward`, released activations, an out-of-memory ladder | supported (text) |
| Transcripts with image parts: `ChatTranscript.FromJson` reads content parts through `ChatParts.FromJson` (images included) | read, then **ignored** by the tuner |
| ShareGPT transcripts (`conversations`, `from`/`value`) | supported; LlamaFactory's `images` list and `<image>` placeholder **missing** |
| A trainable decoder pass with images: `ImagePrefill.Forward` (no cache, logits of every position, gradients reach the features) | supported, **not called by the tuner** |
| Image attention backward (`AttentionSpansBackward`) | **CPU only**; other backends copy to the host (base `Backend` fallback) |
| Image features without gradients (`IVisionEncoderStages.PixelValues`, `Tower`, `Features`) | supported |
| Image transforms per request (`ImageTransformPipeline`, `ImageTransforms` registry) and vision options (`VisionOptions`) | supported in generation; **not in tuning** |
| Checkpointed blocks (`ForwardCheckpointed`) recomputing with the image attention scope (`ImageBlocks.Current`, thread-local) | **unknown**: the recompute in the backward pass must see the same image blocks |
| Packed sequences with images | **refused** (`ImagePrefill.Forward` throws) |
| Graph capture of the step (`CudaGraphs`, `Backend.SupportsGraphs`) with per-step image features | **not handled** |
| Trainable projector (the plug-in creates it with `requiresGrad: true`) through a contract | **missing** (stages are without gradients) |
| A held-out score that is not a loss (character / word error rate on generated text) | **missing** |

## Phases

### Phase 0: the reference (CPU, tiny models)

- A tiny vision fine-tune in the reference tool (`tools/vlm`): the tiny Gemma 3 and the tiny LLaVA, a few image-answer
  pairs, LoRA on q/k/v/o, frozen tower; loss per step and the adapters' gradients written as fixtures.
- Decides the exact masking (image tokens, `<start_of_image>`/`<end_of_image>` and the prompt untrained; the answer and
  its end-of-turn trained) and checks it against LlamaFactory's gemma3 template on one record of `train.json`.

**As built.** `tools/vlm/make_tiny_tuning.py` → `tests/Idrak.Tests/data/vlm-tuning` (125 KB, `README.md` and
`manifest.json` there; two runs write the same bytes). transformers 5.19.0 / torch 2.14.1 (CPU) fine-tunes the tiny
Gemma 3 and the tiny LLaVA on three records (`image.png`, `image-palette.png`; one record with two images, one with a
system line) with a hand-written LoRA (no peft): rank 2, alpha 4, q/k/v/o of the text decoder, tower frozen; run `lora`
(projector frozen) and run `projector` (projector trained too, for phase 2). One step = all records in one batch, loss =
mean token cross-entropy over the batch's trained tokens, plain SGD (rate 0.2, no momentum, clipping or schedule), 3
steps. Per model, `tuning.json` holds the records (messages, rendered and expanded text, ids, the trained mask, image
blocks; Gemma 3's also its ShareGPT form and LlamaFactory's labels), the settings and each run's losses and step-1
per-token cross-entropies; `adapter-init/` is a PEFT folder `LoadAdapter` reads (B not zero, so A has step-1 gradients);
`<run>/grads-step1.safetensors`, `<run>/adapter-step3/`, `projector/projector-step3.safetensors`; `features/` the frozen
features per image. Rerun: `pip download llamafactory==0.9.5 --no-deps -d lf`, then `vlm/bin/python
tools/vlm/make_tiny_tuning.py tests/Idrak.Tests/data/vlm-tuning --llamafactory-wheel lf/llamafactory-0.9.5-py3-none-any.whl`.

**The masking decided.** Untrained: the whole prompt (system and user text, turn headers, image tokens, `<start_of_image>`,
`<end_of_image>` and the `"\n\n"` around them). Trained: the answer and the token that ends the turn (`<end_of_turn>`);
the `"\n"` the Gemma 3 template writes after `<end_of_turn>` untrained. This is `ChatTranscriptEncoder`'s rule for text
(span from the generation prompt's header to the trimmed end-of-message text), so text and image tuning share it.
LlamaFactory 0.9.5's `gemma3` template (read from its wheel, reproduced in the script, which asserts the source lines it
follows) gives the same ids token for token on each record written in its ShareGPT format (`<image>` placeholders, an
`images` list, `system`), and trains one token more: that `"\n"` (its assistant slot is `{{content}}<end_of_turn>\n`);
generation stops at `<end_of_turn>`, so the `"\n"` is never predicted. Both masks are in the fixture. The tiny LLaVA
template writes no end of turn, so its trained span is `" {answer} "` to the end of the text.

**Checked today** (`VisionTuningReferenceTests.cs`, `IDRAK_FILTER="vision tuning reference"`, CPU): the records through
`ChatTranscriptEncoder.Render`, the family's `IImagePromptFormat.Expand` and segment-wise tokenization (ids and mask
exact; `Encode` trains the same tokens), `ImagePrefill.Locate`; then the 3 steps by hand: `ImagePrefill.Forward` with
autograd, `AddAdapters` + `LoadAdapter(adapter-init)`, a masked mean cross-entropy, `Sgd`: losses within 1e-6, step-1
gradients within 2e-7 (Gemma 3) and 6e-7 (LLaVA) of transformers', adapters after step 3 within 1e-5; Gemma 3's
projector run through `Gemma3ImageEncoder.Projector` (gradients and values). LLaVA's projector run is in the fixture but
not checked yet: the test plug-in's projector has no trainable weights (phase 1's `IVisionTuningPart`). Phase 2 runs the
same fixtures through `FineTuner.Train` (`Optimizer = ps => new Sgd(ps, 0.2f)`, a constant schedule, `MaxGradientNorm =
0`, rank 2, alpha 4, q/k/v/o, the adapters loaded from `adapter-init`).

### Phase 1: contracts and registries (Abstraction, no family)

- **`ITuningDataFormat`** + `TuningDataFormats` (`SlotTable`): reads a data file into `ChatTranscript`s with image
  parts. Ships "messages" (today's JSON Lines) and "sharegpt" (today's `conversations`, plus LlamaFactory's `images`
  list: each `<image>` in a turn's text becomes the next image part, paths relative to the data file or an
  `--images` root, a zip opened read-only without unpacking anything but images). Extra formats are registrations.
- **`TuningImages`** (options record): the image transform pipeline (`ImageTransformPipeline`, the same strings as
  `--image-transform`) and `VisionOptions` (pan and scan and any family option, `ThrowIfUnknown` as in generation),
  applied identically to training and evaluation records, and saved next to the adapters so `run` and `serve` can
  apply the same preprocessing.
- **`IVisionTuningPart`** (optional interface a `PretrainedVision` may implement): the parts of the vision side a
  family allows to train (`"projector"`, `"tower"`), their parameters, and a forward with gradients from tower output
  to features. Families without it train the language model only; asking for a part they do not offer fails loudly.
- **`IFeatureCache`** + `FeatureCaches` (`SlotTable`): stores frozen-tower outputs keyed by image bytes hash, transform
  pipeline, vision options and the encoder's identity (family, checkpoint hash, dtype). Ships "memory" and "disk"
  (safetensors files in a cache folder). Contract tests in `Idrak.Abstraction.Testing` (`FeatureCacheSuite`).
- **`ITuningMetric`** + `TuningMetrics` (`SlotTable`): a score of generated text against the reference. Ships
  "cer" and "wer" (Unicode-aware: NFC, Arabic letters, no diacritic stripping unless an option says so). Optional,
  run on the evaluation set every N steps.
- Inventory and API guard updated; `Conformance` gains `CheckTuningDataFormat` and `CheckFeatureCache`.

### Phase 2: images through the tuner (`Idrak.Nlp`)

- `ChatTranscriptEncoder` renders image parts through the family's `IImagePromptFormat` (the same expansion as
  generation: one `ImageTokenLayout` per block), and returns, per sequence, the image positions with the features'
  source (image + transforms + options). Image tokens and their markers are never trained.
- `TrainingSequence` gains its images (an immutable list of `PromptImage` sources), so batching, shuffling, evaluation
  and checkpoints carry them; text-only sequences are unchanged.
- The training step: features from the cache (frozen tower), the projector through `IVisionTuningPart` when trained,
  then `ImagePrefill.Forward` instead of the plain forward, with the family's `IImageAttentionRule`. The loss and the
  adapters are unchanged; the tower gets no gradients.
- Checkpointed blocks recompute with the same `ImageBlocks` scope (the scope travels with the block's recompute, not
  the thread), tested against the stored-activation path to the bit on CPU.
- Packing with images: first version pads (each image prompt its own row) and says so in the plan of the step; packing
  is phase 6.
- Graph capture: off for steps with images until the features enter as graph inputs of a fixed shape (phase 6).
- A family that is not registered: the tuner fails before loading any data, with the registry's message.

**Phase 2, as built (2026-10-09; CPU).**
- **Contracts moved** (decision 10: Idrak.Nlp is now their second user): `PretrainedVision`, `VisionEncoderOptions`,
  `EncoderWeights`, `IVisionTuningPart` and `VisionTuningParts` from core's `Idrak.Models.Abstractions` to
  `Idrak.Abstraction.Generation` (global usings: sources compile unchanged). `IVisionTuningPart` gained `Export` (the
  part's tensors by the checkpoint's names and layout, on the CPU) and `Import`; `VisionTuningParts.FrozenTower` (pixel
  values through `IVisionEncoderStages.Tower`, no gradients, intermediates freed) keeps `IVisionEncoderStages` in
  Abstraction. `VisionFamilies` and `IVisionFamily` stay in core: Nlp reaches the registry through
  `PretrainedModel.CreateVisionEncoder`, which checks it first.
- **Core, public and narrow**: `ImagePrefill.Begin(decoder, ids, images, rule)` → `ImagePrefillScope` (`FirstBlock`,
  `Substitute(embeddings)`, `Dispose`), for a pass run module by module; `Checkpointing` captures the image blocks at the
  forward pass and puts them back for the recompute, so the backward pass may run after the scope closed (the test closes
  it first; without the capture Gemma 3's gradients differ by up to 36). `PretrainedModel.CreateVisionEncoder`,
  `TrainedVisionTensors`, `KeepTrainedVision(encoder, parts)`; `SaveAdapter` writes trained vision tensors as PEFT's
  `modules_to_save` (`base_model.model.` + the checkpoint's name, its layout; the module they lie under listed),
  `LoadAdapter` and `MergeAdapter` read them back (a vision part's modules only; the language model's still refused).
  No new `InternalsVisibleTo` use.
- **Nlp**: `TrainingSequence.Images` (`TrainingImage(ChatImage, TuningImages Preparation, Blocks (Position, Tokens))`,
  empty for text: the text path is unchanged); `TuningVision.Create(model, images, parts, cache)` (the family's encoder on
  the model's device, float32 weights only when a part trains, the parts' parameters marked trainable, a memory cache
  unless one is given); `ChatTranscriptEncoder.Vision` (each marker expanded by the family's `IImagePromptFormat`, blocks
  from `IVisionEncoder.Blocks` of the prepared image, cached per image; image runs located token by token; an image in an
  assistant turn refused; a cut keeps whole blocks only); `FineTuningOptions.Vision` (made from the model when null and
  the sequences hold images); `FineTuner.Evaluate(..., vision)`; `FeatureCacheKey.FeaturesStage`.
- **The step**: per batch, each distinct image's features once: from the cache, else computed (no gradients, freed at
  once) and put. The cache keeps the features when the projector is frozen, the tower's output when it trains or the
  model carries a trained projector (the projector then runs each step, with gradients when trained; its parameters join
  the optimizer). The checkpoint identity in the key hashes the weight files' names, sizes and times, config.json and the
  trained vision tensors' names. Features live in the batch's tensor scope (released with the step). `NetworkLoss` opens
  the image scope and substitutes before the first decoder block, in training, evaluation and DPO's reference pass.
- **Batching**: sequences with images go into padded batches (each its own row); with packing, text-only sequences still
  pack and both kinds are shuffled together. A packed batch with images is refused; a batch with images is never
  recorded as a graph. The trace line per batch names the images and image tokens ("padded rows, … not recorded as a
  graph") and, after the step, the features taken from the cache and encoded. The out-of-memory ladder is unchanged.
- **Saving**: `FineTuner.Train` saves the trained projector with the adapters and `tuning_images.json` when images were
  used (checkpoints too). The tower is refused for training (phase 6: it trains in the step).
- **Tests** (CPU): "vision tuning trainer" (the four runs through `FineTuner.Train`: losses within 1e-4, saved adapters
  and projector within 1e-5 of transformers'; the tower runs twice over three epochs, four cache hits; the folder read back
  by `LoadAdapter` and `MergeAdapter` into a new encoder; checkpointed == stored gradients bit for bit through
  `ImagePrefill.Begin` with the scope closed and through the tuner; the refusals), "vision tuning reference" (the encoder's
  own sequences now equal the fixture's ids, mask and image blocks).
- **Left**: packing images and graphs with images (phase 6), the tower in the step (phase 6), the command line (phase 4,
  which should load images' preparation and create encoders through `PretrainedModel.CreateVisionEncoder`), and the GPU
  runs (the owner's).

### Phase 3: image attention backward on every backend

- `AttentionSpansBackward` kernels for Vulkan and CUDA (generic, tiled; no card-specific constants beyond what the
  backend reports), the CPU kernel the reference.
- Shared device cases (`DeviceCases.Spans`) for the backward pass: random spans, bidirectional blocks, sliding windows,
  bf16 and float32; every backend runs them.
- The training step picks the span kernel or the composed path by measurement (as `AttentionFastest`), so a backend
  without a fast kernel still trains on its own device instead of copying to the host.

**Phase 3 as built (2026-10-09; built and checked on the CPU, PTX and SPIR-V checked on the host, not run on a GPU).**
- **CUDA** (`PtxKernels.SpansBackward.cs`, `CudaBackend.Spans.cs`): `AttentionSpansBackward` on the device, float32
  (also under `MixedPrecision`: the gradient's sums stay float32), head sizes up to 128 (the host fallback beyond), built
  as the backward of `attention_flash_f32`: Δ = dO · O per row (`attn_bwd_d_f32`), then `attn_spans_bwd_kv_f32` (a
  block of 4 warps owns 16 keys of a key/value head and walks every query head of its group, rows in tiles of 32 in
  shared memory; a tile none of whose rows sees the block's keys is skipped by a warp vote) and `attn_spans_bwd_q_f32`
  (a block owns 32 rows of a head and walks the keys from its rows' smallest start to their largest end in tiles of 32).
  Weights recomputed from the log-sum-exp, each row's range read from the starts and ends buffers, grouped heads and
  tables as the forward, the soft-cap's slope; every gradient element has one owner (no atomics: deterministic). In the
  main module (PTX 6.0, sm_50): ptxas 12.9 assembles it for sm_50 … sm_120, no spills, 80-110 registers, 33 KB static
  shared memory (as the causal backward kernels).
- **Vulkan** (`VulkanKernels.Spans.cs`, `VulkanBackend.Spans.cs`): `attention_spans_delta`, then
  `attention_spans_backward_dq` (a workgroup per block of rows, as `attention_spans`: R × C invocations from the width,
  q·k and dO·v staged 64 dimensions at a time, dq kept in registers) and `attention_spans_backward_dkv` (a workgroup per
  block of keys through every query head of its group, a tile of rows skipped when none sees the block's keys);
  head sizes up to 256; the width measured per shape (`VulkanTuneOp.SpanAttentionBackward`), the device's own until
  then; a device that cannot bind the kernels' 9 and 10 storages takes the host fallback. Every kernel at every width
  passes spirv-val (SPIRV-Tools 2025.1) and the workgroup-memory bound.
- **The measured choice with the gradient**: `Backend.PrefersComposedAttention(..., training)` (public API change; the
  dump regenerated): while the gradient is recorded `Tensor.AttentionFastest` asks for the faster path forward and
  backward; CUDA and Vulkan time AttentionSpans + AttentionSpansBackward against the composed pass and its gradient
  (`Backend.ComposedAttentionTraining`: the products, the softmax's gradient) and keep it apart from the inference
  choice (bit 2 of the key's mask field), in their tuning caches.
- **Cases**: the kit's "attention over key ranges, the gradient" (`DeviceCases.SpansBackward.cs`): rows past several
  blocks, keys past several tiles, head sizes 1 to 256, random ranges (empty, past the keys), bidirectional, causal,
  windows, image blocks in a window, grouped heads, tables, a soft-cap, against plain loops and, through tensors,
  against the composed path's gradients; replayed call by call on a device by "conformance kit" and "attention spans
  backward". Tests: "attention spans backward" (the kit's key-range cases on the device; the gradient against the
  CPU in float32 and under bfloat16 MixedPrecision at head sizes 4 to 256, on the device's own kernels; training
  through the fastest path). `--bench-spans train` times the three paths forward and backward.

### Phase 4: the command line (`idrak tune`)

- `idrak tune -P <plugin> --base <model> --data train.json --eval val.json --images <dir|zip>
  --image-transform "<pipeline>" --vision "<options>" [--train-projector] [-w int4|int8|bf16]`, the same in
  `tune.json` (`format`, `images`, `image_transform`, `vision`, `parts`). `idrak tune init` writes a vision example
  when `-P` registers a vision family.
- `--metric cer` and `--metric-every N`: generated answers on (a sample of) the evaluation set, scored.
- The run plan (`-v`) shows the images, blocks per image, image tokens, cache hits, and the memory the tuner measured.
- `idrak run`, `chat` and `serve` load the adapters and the saved `TuningImages` (preprocessing) with `--adapter`.

**Phase 4, as built (2026-10-09; CPU).**
- **Nlp**: `TuningAnswerScorer` (`TuningAnswers.cs`): the conversations ending with an assistant answer, generated greedily
  through the model's chat template and, with images, the run's `TuningVision` (its encoder, a training projector as it
  is now, its transforms and vision options), scored by a `TuningMetrics` metric (`TuningAnswerReport`: the summed
  `TuningScore`, each answer); its own key/value cache per request, released after each scoring, training mode put back.
  `FineTuningOptions.Answers` scores every `Every` steps (each epoch when 0) and `FineTuningProgress.Answers` carries the
  report (the only edit to `FineTuning.cs`: three lines in the step loop). `TuningImages.Family` (saved as `"family"` by
  the tuner, outside equality) and `ThrowIfOtherFamily`.
- **CLI** (`TuneTool`, `TuneCommand`): `--data` (with no command: train), `--data-format` (not `--format`: that is
  idrak's common output option; a tune.json's `"format"` may name the data's), `--images`, `--image-transform`,
  `--vision`/`--vision-option`, `--grayscale`, `--train-projector`/`--parts`, `--feature-cache` (disk under `--cache`),
  `--metric cer|wer`, `--metric-every`, `--metric-samples`; the same keys in tune.json (underscores accepted). The data
  goes through `TuningDataFormats` when an image option, `--data-format` or a text metric asks, or when the model reads
  images and the data is local .json/.jsonl; otherwise the dataset path is unchanged. `--eval-fraction` holds out a seeded
  part of the conversations. The plan: an images line per set (images, distinct, blocks per image, image tokens); `-v`
  every batch (features from the cache or encoded) and, after training, the measured memory (`ModelMemory.Decoder`, the
  encoder, trained parameters, the peak's activations, the feature cache and its hits). `tune evaluate --metric cer` on
  image data: loss and score of the base model and the adapter, side by side, `-o` answers. `tune init -P` writes the
  vision example. Ctrl+C saves adapters, projector and preparation.
- **run/chat/serve**: `ModelChoices` reads the adapter's `tuning_images.json` (command line over it, it over an alias),
  checks its family against config.json before loading and against the model after; `ImageInputs` builds the encoder with
  `PretrainedModel.CreateVisionEncoder` (the trained projector); `run -j` reports `vision_options`.
- **Tests** ("cli vision tuning", CPU): tune on ShareGPT with `<image>` (tiny Gemma 3 from a folder with the projector, a
  transform and a vision option, cer before / every step / in the summary, the -v plan; from a zip, LoRA only; tiny LLaVA
  with its projector); `run --adapter` answers as the library does with `MergeAdapter` + `CreateVisionEncoder` (the trained
  projector changes the features), `--image-transform none` overrides; `tune evaluate --metric cer`; a Gemma 3 adapter on
  LLaVA refused; an unregistered family stops tune before the data; image options on a text model, bad names; tune init -P;
  a tune.json with the image keys and the disk cache.
- **Left**: generated answers re-encode their images (the feature cache keys on image bytes, the generator reads decoded
  pixels); chat batches refuse images, so answers are generated one by one; the GPU runs (the owner's).

### Phase 5: proof on the real model

- On the owner's machine (whatever device it has; commands given in PowerShell): LoRA on
  `bakrianoo/arabic-legal-documents-ocr-1.0` (or `google/gemma-3-4b-it`) with its `train.json`, a short run, loss
  falling, `val.json` CER before and after, the adapter loaded by `idrak run`.

### Phase 6: speed and packing (after the proof)

- Packed sequences with images (spans per segment in `ImageBlocks`).
- Graph capture with images (features as fixed-shape graph inputs, one graph per image-token count).
- Tower in the step when `"tower"` is trained (no cache), with checkpointing through the tower's blocks.

**Phase 6, as built (2026-10-09; CPU, the GPU graph paths compiled and checked on the CPU without recording).**
- **Packing with images.** `PackedSequences` keeps each row's sequence lengths; `ImageBlocks` takes them (from the
  packing in effect for the ids' shape) and gives each packed sequence (and the row's padding) its own key ranges by the
  family's `IImageAttentionRule` at its columns, so a row never sees another sequence's keys; an image block must lie
  inside one sequence. The decoder attends packed rows with images through the span path with the packing's positions
  (they restart per sequence); a causal rule keeps the packed causal kernels and only substitutes the features.
  `ImagePrefill.Forward` and `Begin` no longer refuse packed rows. The tuner packs sequences with images like the others
  (`MakeBatches`), each image at its sequence's offset in the row; the trace says "packed with the others".
- **Recordable image inputs** (core, public): `ImagePrefillInputs.Create(decoder, rows, steps, imageTokens, rule)` holds
  the features ([image tokens, width]), the substitution rows and one pair of key-range tables per window the decoder's
  layers use, in persistent buffers; `Load(images, packing)` copies a batch in (features on the device, the rest
  uploaded) before recording and before each replay; `Begin()` opens a scope that reads only the buffers (recordable).
  `TrainingGraph` records steps with images through it when no vision part trains (the features are values, computed
  or taken from the cache before each replay); the step runner keeps one graph per shape (rows, positions, image
  tokens; a graph serves batches with up to its image tokens), each recorded after its shape ran twice, and another
  only while the device's free memory, measured then (`Backend.AvailableMemory`), holds twice the largest graph so far
  (a device that reports none keeps one); the out-of-memory ladder drops them all as before. With a vision part
  trained, steps with images run ordinarily (said once in the trace).
- **The tower in the step** (`TuningVision.Create(..., parts: ["tower"])`): full weights through the contract
  (`IVisionTuningPart.Parameters`/`Tower`), not LoRA: the contract's parts are saved by the checkpoint's own names
  (`modules_to_save`), and LoRA on the tower would need the contract to name its linear layers in the checkpoint's terms
  (a later extension). The cache keeps the pixel values (`FeatureCacheKey.PixelsStage`); each step runs the tower and
  the projector with gradients; when the tuner checkpoints, `ActivationMemory.CheckpointBlocks()` (Abstraction, public)
  asks the family to checkpoint its blocks (Gemma 3's SigLIP layers through `ForwardCheckpointed`). A checkpointed first
  pass now computes as the recompute does (`Checkpointing.FirstPass`: `TransformerEncoderLayer` skips its inference-only
  fused bias-GELU there), so checkpointed and stored tower gradients agree bit for bit. Every encoder parameter is frozen
  but the trained parts'. Gemma 3 (sample): `Export`/`Import` of the tower (every SigLIP tensor, the fused q, k, v split
  back into `q_proj`/`k_proj`/`v_proj`, Linear weights [out, in], the patch convolution in its stored shape),
  `PartsOf`.
- **Memory fix**: `VisionEncoderOptions.TrainedParts` (Abstraction, public): the parts built with float32 weights, the
  rest as `Weights` says; `IVisionTuningPart.PartsOf(names)` (default: every part offered) tells which parts trained
  tensors belong to. `CreateVisionEncoder` asks for those parts only, and `TuningVision.Create` for the trained parts
  only: a trained projector over a bfloat16 checkpoint keeps the tower in bfloat16 (the tiny model: 22,976 bytes of
  weights against 25,024 when the whole encoder was float32; the tower's projections at half their float32 bytes).
- **Speed**: a batch's distinct images go through the vision side together: cache misses' pixel values through the
  frozen tower (or the whole encoder for the features stage) in one pass, then the trained or carried projector (and a
  trained tower) over all of them joined, split back per image (gradients through the join); one image at a time when
  that frozen pass runs out of device memory (or the encoder shows no stages). Each image's features equal its features
  alone within 1e-5 (the property `VisionEncoderSuite` checks). The trace per step: "N images in P passes, T ms".
  Measured on the CPU (4 cores, tiny models, 12 images): Gemma 3 projector 0.18 ms one at a time against 0.06 ms in one
  pass, frozen tower 11.1 ms against 4.7 ms; LLaVA projector 1.9 ms against 0.9 ms, tower 14.6 ms against 5.2 ms.
- **Tests** (CPU, `IDRAK_FILTER="vision tuning"`): packing (each sequence's logits and adapter gradients as alone,
  Gemma 3 and LLaVA, checkpointed packed == stored bit for bit, the fixed inputs bit for bit and loaded again for another
  layout, refusals), packed against padded `FineTuner.Train` on the phase-0 fixtures (both within 1e-4 of transformers';
  the phase-2 trainer tests now run packed by default and still match), graph inputs (recorded and replayed on a device
  with graphs; on the CPU the same pass from the fixed buffers), the tower (export equals the checkpoint, import round
  trip, finite differences through the decoder, checkpointed == stored; trained through `FineTuner.Train`, saved and read
  back; checkpointing on and off equal), the trained-projector memory, batched images.
- **Left**: the GPU runs (the owner's: "vision tuning graph" records and replays there; the tuner's own graph path
  with images needs packing, which CUDA's packed kernels allow only for bfloat16 head sizes 64 and 128 without a
  window, so the tiny fixtures run padded there); LoRA on the tower's linear layers; graphs for steps whose vision part
  trains (the projector would run inside the graph on fixed tower outputs).

## Tests (targeted, `IDRAK_FILTER`; never the full suite)

- Contract suites for every new registry (format, feature cache, metric), and a test that the library registers no
  vision family for tuning.
- Tiny Gemma 3 (plug-in) and tiny LLaVA (`tests/Idrak.PluginTests`): loss and adapter gradients against phase 0's
  fixtures on CPU; the same on Vulkan (lavapipe) and CUDA where present.
- Checkpointed vs stored activations with images: identical gradients.
- Masking: image tokens and prompt untrained, answer trained, per template.
- Unregistered family: `idrak tune` exits with the registry's message.

## Risks

- Backward through image attention on non-CPU backends is new code; until phase 3, image tuning on a GPU is slow.
- Memory per record is larger than text tuning (a page's answer is long, plus 256 or more image tokens per block);
  the measured out-of-memory ladder must cover it, and the run plan must say what it chose.
- The LlamaFactory template and Idrak's rendering must agree token for token, or the model trains on a different
  prompt than the one it runs with (phase 0 checks it).
- Data hygiene: Hugging Face flags pickle content in `downloaded_images.zip`; the format reader opens images only.

## Open questions

1. Train the projector by default for Gemma 3, or the language model only (LlamaFactory's run: language model only)?
2. Feature cache on disk by default, or memory first and disk when it does not fit?
