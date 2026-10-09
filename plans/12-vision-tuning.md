# Plan 12: fine-tuning vision-language models (images in `FineTuner` and `idrak tune`)

**Status:** planned 2026-10-09, against the code at `abstraction` 4ac4fa4. Nothing built yet.

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

### Phase 3: image attention backward on every backend

- `AttentionSpansBackward` kernels for Vulkan and CUDA (generic, tiled; no card-specific constants beyond what the
  backend reports), the CPU kernel the reference.
- Shared device cases (`DeviceCases.Spans`) for the backward pass: random spans, bidirectional blocks, sliding windows,
  bf16 and float32; every backend runs them.
- The training step picks the span kernel or the composed path by measurement (as `AttentionFastest`), so a backend
  without a fast kernel still trains on its own device instead of copying to the host.

### Phase 4: the command line (`idrak tune`)

- `idrak tune -P <plugin> --base <model> --data train.json --eval val.json --images <dir|zip>
  --image-transform "<pipeline>" --vision "<options>" [--train-projector] [-w int4|int8|bf16]`, the same in
  `tune.json` (`format`, `images`, `image_transform`, `vision`, `parts`). `idrak tune init` writes a vision example
  when `-P` registers a vision family.
- `--metric cer` and `--metric-every N`: generated answers on (a sample of) the evaluation set, scored.
- The run plan (`-v`) shows the images, blocks per image, image tokens, cache hits, and the memory the tuner measured.
- `idrak run`, `chat` and `serve` load the adapters and the saved `TuningImages` (preprocessing) with `--adapter`.

### Phase 5: proof on the real model

- On the owner's machine (whatever device it has; commands given in PowerShell): LoRA on
  `bakrianoo/arabic-legal-documents-ocr-1.0` (or `google/gemma-3-4b-it`) with its `train.json`, a short run, loss
  falling, `val.json` CER before and after, the adapter loaded by `idrak run`.

### Phase 6: speed and packing (after the proof)

- Packed sequences with images (spans per segment in `ImageBlocks`).
- Graph capture with images (features as fixed-shape graph inputs, one graph per image-token count).
- Tower in the step when `"tower"` is trained (no cache), with checkpointing through the tower's blocks.

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
