# Plan 19: creating vision-language models from scratch (a family made, not loaded)

**Status:** planned 2026-10-14, against the code at `abstraction` b8ef0b25. Nothing built. The owner's request
(2026-10-14): "new vision language", after the answer that Idrak creates image models (classifiers, detectors,
segmenters, CTC recognizers) from random weights but only tunes vision-language models.

**Goal.** Make a vision-language model that did not exist before: a vision tower, a projector, a text decoder and a
tokenizer, every weight random at the start, described by one spec, pretrained in stages on the owner's data (text, then
pages and their text), evaluated, and saved as a checkpoint that loads through the same family contracts every loaded
model uses, so `idrak run`, `chat`, `serve` and `tune` work on it unchanged. Arabic documents are the first target (the
owner's; plan 18's synthetic pages are the data); the machinery serves any language and any image-text task.

**Rules that hold for every phase** (`CLAUDE.md`):

- **Card-agnostic.** No phase assumes a device, a vendor or a memory size. Batches, micro-batches and recomputation are
  measured on the running device (the tuner's out-of-memory ladder, `HostOptimizer`, `AdamW8Bit`); no size, card or GB
  budget appears in code, comments, tests or docs, this plan included.
- **Abstraction.** Everything new is a contract with a `SlotTable` registry (decision 10 of plan 10), in the inventory,
  behind the public API guard. The library registers no family: the created family is a plug-in in `samples/`, as
  Gemma 3 and Qwen-VL are; an unregistered name fails with the registry's "not registered (load its plug-in with -P)".
- **Speed and memory first.** Bytes parsed, not strings (the tokenizer trainer); memory walked in order; work done once
  (features of frozen stages cached, as `TuningVision` caches them); device memory that does not grow across steps;
  the library's telemetry on every step; cold and warm measured apart.
- **Nothing fixed.** Every stage, ratio, schedule and threshold is an option (a spec field or a run option, an
  environment variable where the library reads one, carried to the CLI and the apps).
- **Tests.** Targeted CPU tests only (`IDRAK_DEVICES=cpu IDRAK_FILTER=...`); the owner runs CUDA and Vulkan from a
  command list at the end of each phase.

## What Idrak has, and what is missing

| Part | Today |
|---|---|
| A text decoder from a spec with random weights: `DecoderSpec` (Idrak.Abstraction) built by `DecoderBuilder.Build` with `DecoderBuildOptions { Seed, InitStd }`; the GPT sample trains one from a token file (`TokenFileSource`, memory-mapped windows) with `AdamW` / `AdamW8Bit`, bf16, a schedule, checkpoints | supported (text) |
| A vision tower from random weights | **missing as a contract**: `IVisionFamily` has `Read(checkpoint)` only; `SiglipVisionEncoder` lives in the Gemma 3 plug-in (plan 16 moved families out of the library); core has the layers a tower needs (`Conv2d`, learned embeddings, `TransformerEncoderLayer`, `LayerNorm`, `GELU`, pooling) |
| A projector from random weights | **missing** (the plug-ins build theirs from tensors only) |
| Training the tower and the projector in full, with gradients, beside the decoder (`VisionTuningParts.Tower` / `Projector`, `TuningVision`, cached frozen stages, checkpointed blocks, micro-batches) | supported (plan 12) |
| Training every decoder parameter (no adapters) with images in the prompt | **missing**: `FineTuner` trains adapters only ("the base weights stay as loaded and frozen"); `ImagePrefill.Forward` gives gradients to the features and the decoder, so the pass exists, the loop does not |
| Mixing text-only windows and image-text records in one run, by ratio, in stages (text first, then alignment, then everything) | **missing** |
| A tokenizer trained on the owner's text | **missing**: `BpeTokenizer` reads a tokenizer.json; `CharTokenizer`, `WordTokenizer`, the GPT sample's rune tokenizer exist; no trainer |
| Image-text data: ShareGPT / messages / plain records with images (`TuningDataReaders`, `TuningImages`), plan 18's `SyntheticPages` (pages rendered from text, every scan and paper effect, a manifest), `idrak data synth`, `idrak data text` | supported |
| Saving: `PretrainedModel.SaveHuggingFace` writes model.safetensors with the trained vision tensors and copies config.json, the tokenizer and the chat template from the original folder | partial: **a created model has no original folder**; config.json, tokenizer.json, preprocessor_config.json and the chat template must be written from the spec |
| Loading the result: `PretrainedArchitectures.Get(name).Spec(config)` + `VisionFamilies` → `idrak run`, `chat`, `serve`, `tune` | supported once the family is registered and the files exist |
| Held-out scores: CER and WER on generated text (`TuningMetrics`), `AnswerScorer`, loss | supported |
| Distillation from a teacher (`Distillation`, `TeacherLogits`, `idrak distill`) | text only; **images missing** |
| More than one device on one run | not started (plan 6); one device per run |

## Where each part lives (decisions 8 and 10 of plan 10)

| Part | Package | Why |
|---|---|---|
| `VisionLanguageSpec` (tower spec, projector spec, `DecoderSpec`, tokenizer and prompt settings, as JSON), `IVisionFamily.Create` (random weights from a spec) and the `VisionFamilies` side of it | `Idrak.Abstraction` (`Generation`) and `Idrak.Models.Abstractions` | beside `DecoderSpec` and `IVisionFamily`, their users |
| `ITokenizerTrainer` / `TokenizerTrainers` (a registry; "bpe" the library's byte-level trainer) | `Idrak.Abstraction` (contract) and `Idrak` (the trainer, beside `BpeTokenizer`) | core owns the tokenizers |
| Writing a created model's folder (config.json from the spec, model.safetensors, tokenizer.json, preprocessor_config.json, chat_template) | `Idrak` (`PretrainedModel`) | saving is core's |
| The pretraining loop (`Pretrainer`: every parameter, stages, mixed sources, checkpoints and resume) and its options | `Idrak.Nlp` (beside `FineTuner`, reusing `TuningVision`, the losses, schedules and the memory ladder) | training with images is Nlp's |
| Images in `TeacherLogits` (phase 6) | `Idrak.Nlp` | distillation is Nlp's |
| `idrak create`, `idrak pretrain`, `idrak data tokenizer` | `Idrak.Cli` | the surface |
| The created family (tower, projector, prompt format, attention rule, preprocessing, `Create` and `Read`) | `samples/ScratchVision/Idrak.ScratchVision` (name to be decided by the owner) | a family is a plug-in, never the library |
| The reference (tiny model, one step's loss and gradients, a tokenizer trained on a small corpus) | `tools/vlm`, fixtures in `tests/Idrak.Tests/data` | as plans 11 and 12 did |

## Phases

| # | Phase | What | Done when (CPU here; GPU by the owner) |
|---|---|---|---|
| 0 | **The reference** | `tools/vlm/make_tiny_scratch.py`: a tiny ViT tower, a two-layer MLP projector and a Llama-style decoder with weights written by the tool (so Idrak and PyTorch start from the same values), three image-text records and a text window; one training step with every parameter trained: the loss and every gradient as fixtures. A byte-level BPE trained by Hugging Face `tokenizers` on a small Arabic-English corpus: its vocabulary and merges as the tokenizer fixture | the fixtures exist, two runs write the same bytes, a README names the versions |
| 1 | **Contracts and the spec** | `VisionLanguageSpec` (JSON: the tower's patch size, width, layers, heads, image size, pooling; the projector; the `DecoderSpec`; the image token ids and count; the prompt format and attention rule names; the preprocessing). `IVisionFamily.Create(spec, device, seed)` returning the same `PretrainedVision` `Read` returns, so one family does both. `PretrainedModel.Create(spec, family, options)` builds the decoder (`DecoderBuilder.Build` with the seed) and the vision part. `SaveHuggingFace` writes config.json (the spec, in the family's layout), tokenizer.json, preprocessor_config.json and the chat template when there is no original folder; the saved folder loads back through `Read` | a created tiny model saves, reloads tensor for tensor, and `idrak run` answers (nonsense, as expected) through the registered family; an unregistered family name is refused by the registry; inventory and `api/*.txt` regenerated |
| 2 | **The tokenizer trainer** | `ITokenizerTrainer` / `TokenizerTrainers`: "bpe" is a byte-level BPE trainer on UTF-8 bytes (pairs counted in one pass over blocks of bytes, pooled buffers, merges by priority queue; rule 74), with special tokens (image, chat and padding tokens from the spec), writing tokenizer.json in the format `BpeTokenizer` reads; a `--vocab`, `--min-frequency` and the pre-tokenization rule as options | the trained vocabulary and merges equal phase 0's fixture; encoding and decoding round-trip the corpus; the `TokenizerSuite` passes on the result |
| 3 | **The pretraining loop** | `Pretrainer` (Idrak.Nlp): every parameter of the decoder, tower and projector in one optimizer (`AdamW`, `AdamW8Bit`, `HostOptimizer` as options), bf16 activations, the loss on text tokens only (image tokens untrained, as plan 12 masks them), packed text windows from `TokenFileSource` and image-text records from `TuningDataReaders` mixed by a ratio, **stages as settings**: `text` (decoder alone on text), `align` (tower and projector, decoder frozen), `joint` (everything), each with its steps, rate, schedule and warm-up; the memory ladder (micro-batches, `Checkpointing`, `RecomputeFeedForward`) measured; checkpoints every N steps with `resume`; telemetry and a live line (step, loss per source, tokens and images a second, device memory, time left) | one step's loss and every gradient match phase 0's fixtures; the tiny model's loss falls over a few hundred CPU steps on the fixture data; a run stopped and resumed continues bit for bit on the CPU |
| 4 | **The family plug-in** | `samples/ScratchVision/Idrak.ScratchVision`: a ViT tower from core layers (patch convolution, learned positions, bidirectional encoder layers, a final norm, average pooling to the spec's token count), an MLP projector, the decoder from `DecoderSpec` through `PretrainedArchitectures`, a prompt format (`<start_of_image>`, N soft tokens, `<end_of_image>`), an attention rule (image blocks bidirectional, the rest causal, as plan 11 built; plain causal as the option), preprocessing (resize, optional grayscale, scale and normalize from the spec), `Create` and `Read`, registered in one call (`RegisterIdrakPlugin`) | the tiny model from phase 0 trains, saves, reloads and generates through `idrak run -P`, `chat`, `serve` and tunes through `idrak tune -P` with LoRA; the family's tower and projector match the fixture's values after one step |
| 5 | **Command line** | `idrak create SPEC.json -o FOLDER -P plugin` (a random model written as a checkpoint); `idrak pretrain FOLDER --text TOKENS.bin --pages DATA [--stage text\|align\|joint] [--steps N] ...` (run.json, `idrak resume`, `runs show`); `idrak data tokenizer CORPUS --vocab N -o tokenizer.json`; every option in `tune.json`'s style, Arabic help as everywhere | CLI tests on the tiny model (CPU): create, tokenizer, pretrain a few steps, resume, run; unregistered family refused with `-P` named |
| 6 | **Distillation from a vision-language teacher** (optional, a quality lever) | images in `TeacherLogits` and `Distillation`: a registered teacher family (Qwen-VL, Gemma 3) scores the same page and text, the student trains on its logits beside the token loss, through the existing `idrak distill` | the tiny student's loss against a tiny teacher matches a reference step; `idrak distill` accepts `--pages` |
| 7 | **The proof on the owner's machine** | A small model (the spec's sizes chosen by the owner; nothing in the library prefers one) pretrained on `idrak data text` windows and plan 18's synthetic pages, its CER and WER on held-out pages against the tuned Qwen-VL adapter from plan 18's "Why"; the GPU commands from every phase | the owner's run |

**Order.** 0, then 1 and 2 together (the spec needs the tokenizer's special tokens; the trainer needs none of the spec),
then 3, 4 (3's tests need a family: the tiny one in `tests/Idrak.PluginTests` first, the sample after), 5, then 7; 6
when the owner wants it.

## What a from-scratch run is, honestly

- **The pipeline is the deliverable of phases 0 to 5.** It is proven at tiny scale on the CPU, as every plan's reference
  is. Whether a model of useful quality comes out is a matter of the owner's data and compute: a document reader that
  competes with a tuned pretrained model needs far more text and pages than a tuning run, and plan 6 (several devices
  on one run) is not started, so a run is one device.
- **The staged recipe is the hedge.** The `text` stage is cheap and reusable (one decoder serves every later image
  run); the `align` stage trains only the tower and the projector with the decoder's features cached where the ladder
  allows; the `joint` stage is where the cost is, and it is a setting, not a requirement.
- **Distillation (phase 6) is the lever** that lets a small created model learn from a large loaded one on the owner's
  own pages, which is why it is in the plan though it is optional.

## Decisions for the owner

1. The family's name (`ScratchVision` is a placeholder) and the sample's folder.
2. The tokenizer: train one on the owner's corpus (phase 2), or start from an existing tokenizer.json file (a data file
   an app loads; phase 2 is then later work). Training one keeps Arabic bytes efficient; reusing one is faster to start.
3. The attention rule for image tokens: bidirectional image blocks (Gemma 3's, built) or plain causal (simpler kernels).
4. The first text corpus for the `text` stage (`idrak data text` can make Arabic and English; a real corpus is better).
5. Whether phase 6 (distillation from Qwen-VL on the owner's pages) comes before or after phase 7.
