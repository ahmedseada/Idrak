# What is left

Every plan's status and what is left in it, grouped by plan (2026-10-22; the backend, operation, kernel and element-type plans' items removed 2026-11-14, with the backend, kernel and element-type items of the other plans' sections and moved text). A finished plan is in [done/](done/) (its
unfinished parts are listed here, so the plan itself is closed); a plan not started is in [opened/](opened/). Each item
names its kind (bug, speed, memory, feature, check, question, docs) and the section of the plan it came from. The owner's
test runs stay in the plans (each section below says where); they are not listed as items. The text that was moved out of
the plans is kept, as it was written, at the end ("The moved text").

| Plan | Status | Where it stands | Left |
|---|---|---|---|
| [Plan 6: Idrak.Network](done/6-network.md) | Done | Superseded by Plan 23 | 0 |
| [Plan 10: Idrak.Abstraction](done/10-abstraction.md) | Done | Done, with leftovers | 5 |
| [Plan 11: Images into language models](done/11-vision-language.md) | Done | Done, with leftovers | 7 |
| [Plan 12: Fine-tuning vision-language models](done/12-vision-tuning.md) | Done | Done, with leftovers | 6 |
| [Plan 13: Idrak.Vision](done/13-vision.md) | Done | Done, with leftovers | 15 |
| Plan 14: The refactor backlog | Removed | In progress | 6 |
| [Plan 15: The Arabic legal OCR app](done/15-legal-ocr-app.md) | Done | Done | 0 |
| [Plan 16: One contract for every vision-language family; Qwen-VL](done/16-qwen-vl.md) | Done | Done, with leftovers | 2 |
| [Plan 17: Speed and memory for vision and OCR](done/17-vision-ocr-speed.md) | Done | Done, with leftovers | 7 |
| [Plan 18: Synthetic document pages](done/18-synthetic-pages.md) | Done | Done, with leftovers | 2 |
| [Plan 19: language and vision-language models from scratch](done/19-vision-language-from-scratch.md) | Done | Done, with leftovers | 5 |
| [Plan 20: Idrak.Diffusion (image generation by denoising)](opened/20-diffusion.md) | Not started | Planned, nothing built | 19 |
| [Plan 21: Idrak.Audio (sound in and out)](done/21-audio.md) | Done | Done, with leftovers | 11 |
| [Plan 22: speed and memory everywhere, and the library owns what is generic](done/22-performance-and-library.md) | Done | Done, with leftovers | 5 |
| [Plan 23: Idrak.Cluster and Idrak.Network](done/23-cluster.md) | Done | Done, with leftovers | 12 |
| [Plan 24: Unigram (SentencePiece) and WordPiece tokenizers, built in](done/24-tokenizers.md) | Done | Done, with leftovers | 15 |
| Plan 25: Idrak.Extensions.AI (no file) | Dropped | Dropped | 0 |
| [Plan 26: model families as plug-in packages, and `Idrak.Nlp` renamed `Idrak.Text`](done/26-plugin-packages.md) | Done | Done, with leftovers | 35 |
| [Plan 27: reinforcement learning](done/27-reinforcement.md) | Done | Done, with leftovers | 13 |
| [Plan 28: documents](opened/28-documents.md) | Not started | Planned, nothing built | 10 |
| [Plug-in points](done/plug-in.md) | Done | Partly built | 19 |
| [Plug-in 1: Idrak.HarfBuzz](opened/plugins/1-harfbuzz.md) | Not started | Planned, nothing built | 4 |
| [Plug-in plans index](opened/plugins/README.md) | Not started | Planned, nothing built | 0 |
| [idrak-cli: one command-line tool](done/idrak-cli.md) | Done | Done, with leftovers | 22 |
| Open questions and checks | Removed | In progress | 0 |
| [GPU checks for plans 12 and 13](done/gpu-checks.md) | Done | In progress | 0 |

## Plan 6: Idrak.Network ([6-network.md](done/6-network.md))

**Status:** Done. **Where it stands:** Superseded by Plan 23 — superseded 2026-10-18; nothing was built from this file; plan 23 restates what it reuses (the fixed summation order, the wire header, the HMAC hello, a rank's share of the draw).
**Owner's runs:** none (its proof run is plan 23's).

No leftovers.

## Plan 10: Idrak.Abstraction ([10-abstraction.md](done/10-abstraction.md))

**Status:** Done. **Where it stands:** Done, with leftovers — phases 0-8 and wave 4 done on the CPU and Vulkan (2026-10-07/08); CUDA has run much of it since (plans 17, 23). The version bump (W4.7, 0.3.1 → 0.4.0) is left for the owner's release (CLAUDE.md: agents do not bump it).
**Owner's runs:** pending: "On the author's machines after W4.7" (HIP and the 2% decoding gate never run; the CUDA-only kernels against the kit's references).

| # | Leftover | Kind | From |
|---|---|---|---|
| 10.1 | MCP tools `transcribe` and `image` through the model-kind contract (today `generate`, `chat`, `embed`, `McpModelTools.cs:16-22`). | feature | Phase 8b (as built), "Later" |
| 10.2 | `Conversation`'s tool loop on top of `ChatTools.WithTools` (it runs its own loop, `Conversation.cs:122`). | feature | Phase 8b (as built), "Later" |
| 10.3 | A name-to-factory registry of engine model kinds, once a package records its kind. | feature | Phase 3c (as built) |
| 10.5 | No failure policy for KV layouts, devices, network steps, ONNX translators, data file formats and the GGUF and family tables. | feature | Phase 6b (as built) |
| 10.6 | A typed `Defaults.Get<T>()` (only named registries' `Default(name)` exist). | feature | Phase 6b (as built), "Not built" |

**Dropped as done:** phase 4's "left for review" and 8c's unchecked one-user types (W4.3); the version per default and the outdated-override note (W4.1); the Float8 and `AttentionStrided` references and the PTX/HIP plug-in hook (W4.5, 2026-10-07/08).

## Plan 11: Images into language models ([11-vision-language.md](done/11-vision-language.md))

**Status:** Done. **Where it stands:** Done, with leftovers — phases 0-5, 7, 8, 10, pan and scan and image transforms built 2026-10-07 to 10-09 (the real model checked 2026-10-09); phase 6 stopped by the author; phase 9 partly taken by plans 12 and 16.
**Owner's runs:** pending: the real model through the server and with the card's pipeline (phase 8 and "Image transforms", "Not done").

| # | Leftover | Kind | From |
|---|---|---|---|
| 11.1 | Phase 6: an image encoded once per conversation (kept by `ChatImage.Hash`), so a later turn does not re-encode and re-prefill it; today `ChatGenerator` keeps only the last prepared page's pixels (`ChatGenerator.cs:61-86`). | speed | Phase 6; phase 7 (as built) |
| 11.2 | A content-hash cache of image features in the server (each request encodes its images). | speed | Phase 8 (as built), "Not done" |
| 11.3 | An MCP `chat` tool with images (MCP flattens image blocks to text, `McpTools.cs:99`) and an `ocr` tool if useful. | feature | Phase 9; phase 8, "Not done" |
| 11.4 | `--schema` on the server (the CLI's schema is an instruction plus a check). | feature | Phase 8 (as built), "Not done" |
| 11.5 | libjpeg's interblock smoothing (progressive files missing their last refinement scans). | feature | Phase 2 (as built), "Not done" |
| 11.7 | Generated tokens per second within 5% of the text-only Gemma 3 4B with the same weight format: never measured. | check | Performance targets |
| 11.8 | Real LLaVA checkpoints never run (the tokenizer gap they need is closed). | check | Phase 10, "Remaining" |

**Dropped as done:** packed (bfloat16) encoder weights (plan 17 D8, `EncoderWeights.BFloat16`); the LLaVA BOS post-processor gap (`ChatRequest.AddSpecialTokens`, `ChatGenerator.cs:185`) and `selectattr` (`JinjaBuiltins.cs:690`); transforms run twice per read (plan 17 B1; the web sample was removed); phase 9's Qwen2.5-VL (plan 16), image fine-tuning (plan 12) and pan and scan; open questions 1 (both: `--grayscale` and the alias setting, phase 7) and 4 (OCR is an application, plan 13; the reader is `ImageReader` in Nlp, plan 15 step 0b). Encoder-decoder OCR models (TrOCR, Donut) are filed under 16.2.

## Plan 12: Fine-tuning vision-language models ([12-vision-tuning.md](done/12-vision-tuning.md))

**Status:** Done. **Where it stands:** Done, with leftovers — phases 0-4 and 6 built 2026-10-09 (CPU); phase 5 (the real model) is the owner's.
**Owner's runs:** pending: "Phase 5: proof on the real model"; the GPU runs (`gpu-checks.md`).

| # | Leftover | Kind | From |
|---|---|---|---|
| 12.1 | The CER scorer's generated answers re-encode their images (the feature cache keys on image bytes; the generator reads decoded pixels). | speed | Phase 4 (as built), "Left" |
| 12.2 | The scorer generates answers one at a time (`TuningAnswers.cs:183-187`), though `ChatGenerator.StreamBatch` takes images since plan 17 F3a. | speed | Phase 4 (as built), "Left" |
| 12.3 | LoRA on the tower's linear layers (the contract would name them in the checkpoint's terms); today the tower trains whole. | feature | Phase 6 (as built), "Left" |
| 12.4 | Recorded graphs for steps whose vision part trains (the projector inside the graph on fixed tower outputs). | speed | Phase 6 (as built), "Left"; plan 17 F6 |
| 12.6 | Train the projector by default for Gemma 3, or the language model only (as built: language model only unless `--train-projector`)? | question | Open questions 1 |
| 12.7 | Feature cache on disk by default, or memory first and disk when it does not fit (as built: memory unless `--feature-cache disk`)? | question | Open questions 2 |

**Dropped as done:** phase 0's unchecked LLaVA projector run (checked by phase 2's four `FineTuner.Train` runs and phase 4's CLI test); phase 2's "Left" items (built in phases 4 and 6).

## Plan 13: Idrak.Vision ([13-vision.md](done/13-vision.md))

**Status:** Done. **Where it stands:** Done, with leftovers — steps 1-7 and the OCR sample built 2026-10-09/10, CPU-checked; every GPU kernel written and compiled, CUDA run since in parts (plan 14's OCR profile).
**Owner's runs:** pending: the "For the owner (GPU ...)" lists of every as-built section; `gpu-checks.md`.

| # | Leftover | Kind | From |
|---|---|---|---|
| 13.1 | A measured choice between micro-batches and activation recomputation for image training. | memory | Step 7 (as built), "Left" |
| 13.2 | Letterboxed training (the loader can; the CLI stretches). | feature | Step 7 (as built), "Left" |
| 13.3 | A segmentation family checked against a PyTorch reference (the ONNX test family is checked against the library only). | check | Steps 6 and 7 (as built), "Left" |
| 13.4 | Importing PyTorch exports that read the last time step through computed gathers. | feature | Step 2 (as built), "Left" |
| 13.5 | ONNX: adaptive pooling with uneven windows on export, ConvTranspose's `output_shape` and output padding at or above the stride, Resize's other coordinate modes (`asymmetric` linear, `tf_crop_and_resize`), cubic and antialiased resizing. | feature | Step 2 (the rest), "Left" |
| 13.11 | A safetensors tensor read straight into device storage for convolutions and float32 layers (plan 17 G1 did it for bfloat16 projections only). | memory | Step 6 (as built), "Left" |
| 13.12 | An ONNX import that streams initializers instead of reading the whole file. | memory | Step 6 (as built), "Left" |
| 13.15 | OCR sample: touching lines stay one band. | feature | The OCR sample, as built; Plan 14 backlog (gaps) |
| 13.16 | The measured batching `idrak predict` uses (`MeasuredBatches`, internal in `ImageModelSupport.cs:188`) as a library contract (the sample copies it). | feature | The OCR sample, "Library gaps found"; Plan 14 backlog (gaps) |
| 13.17 | `ChatImage` carries no source path, so pages of a data file are named by record. | feature | The OCR sample, "Library gaps found"; Plan 14 backlog (gaps) |
| 13.18 | A per-layer training-step profile as a library hook (the CLI's `ProfileCommand` recorder is internal; the library's `Profiler` gives operations and kernels on any device since plan 17 H3b). | feature | Plan 14 backlog (CUDA order 10; gaps) |
| 13.20 | The ArabicOcr sample writes a non-finite loss into `recognizer.json` and crashes (`TrainCommand.cs:346`); stop with a clear message. | bug | Plan 14 backlog 1.7 |
| 13.24 | The first OCR epoch is cold (about 270 s against 150 s warm): measure allocations and tunings a step; suspect the pool's exact-size blocks. | memory | Plan 14 backlog 12c |
| 13.25 | Held-out CER moves between epochs at a learning rate near 0 (2.0-3.0%, best 1.5%): recompute BatchNorm's running statistics over a training pass before evaluating, or evaluate an averaged copy. | bug | Plan 14 backlog 12d |
| 13.27 | `Conv2d.ForwardFused` refolds the batch norm into a weight-sized temporary on every inference call: cache the folded weights (memory) or keep refolding (time)? | question | Plan 14 backlog (R5) |

**Dropped as done:** step 3's CTC device kernels and step 2's dilated windows on devices (step 1); step 2's dedicated depthwise kernel (step 1: CPU direct loops, `conv_depthwise` on Vulkan and CUDA); `NetworkAnalysis` knowing the new steps (step 2, the rest); step 6's `predict`/`train` (step 7) and the stale table (now marked as of planning); the "noise" augmentation gap (`Idrak.Vision/LibraryRegistrations.cs:68`).

## Plan 14: The refactor backlog (removed 2026-10-22: its items are filed under their plans here)

**Status:** Removed. **Where it stands:** In progress — a backlog of other plans' leftovers (plans 12 and 13, the fused recurrent work, the optimization-rules audit R1-R11). Items 10a, 10c-10i, 11a (to verify), 11e-11g are done. Every other item is filed under its plan above (Plans 12, 13, idrak-cli; the backend plans' items were removed 2026-11-14) with From "Plan 14 backlog"; the ones belonging to no plan are below.
**Owner's runs:** pending: section 4 "Checks never run" (filed as 12 phase 5, `gpu-checks.md`).

| # | Leftover | Kind | From |
|---|---|---|---|
| 14.1 | Card names in benchmark tables and notes: README.md (51), plans/*.md (~55), src/Idrak/README.md (3), docs/performance-notes.md, docs/optimization.md. Owner's decision: a results table names the machine it ran on; advice and defaults must not. | question | 1.4 |
| 14.2 | `ActivationMemory.Release/Compress` take `params Tensor?[]`: an array per layer per step, also at inference; `params ReadOnlySpan<Tensor?>` (public API change). | memory | Section 2, Layers and models (R5); CUDA order 6 |
| 14.3 | `GraphModule.Binary` and `LibraryGraphOps.Scalar` read scalar constants back every forward (a GPU sync per node); BatchNorm in evaluation recomputes its inverse deviation every call. Both need a buffer version or invalidation hook in Idrak.Abstraction. | speed | Section 2, Layers and models (R5); CUDA order 6 |
| 14.5 | `Sequential` allocates one closure per whole-model forward. | memory | Section 2, Layers and models (R5) |
| 14.6 | Audit areas not finished (files untouched / total): R4 Abstraction 59/75, R6 core 37/57, R7 Nlp, AspNetCore, Mcp, Onnx.Runtime 32/54, R8 CLI A-M 26/53, R10 Data, Vision, kit 79/84, R11 samples and tools 42/56; hottest first: R10, R7, R4, R6. | check | Section 3 |
| 14.7 | The analyzers of rule 65 (`AnalysisModePerformance` = All) were never turned on; turning them on lists what the review missed (warnings are errors). | check | Section 3 |

**Dropped as done:** 1.2 (an unregistered GGUF pre-tokenizer is refused naming the registry; the library sets no fallback: `GgufTokenizerFormats.cs:37-46`, `GgufPreTokenizers.Fallback`); `ExactGELU` as ~20 passes (one `GeluErf` operation with CPU, CUDA and Vulkan kernels, `Layers/Activations.cs:100-104`); the "noise" augmentation; CUDA order 1 and 1.5 (the fused recurrent kernels pass on CUDA, 10g); 10a, 10c, 10d-10i, 11e, 11f, 11g.

## Plan 15: The Arabic legal OCR app ([15-legal-ocr-app.md](done/15-legal-ocr-app.md))

**Status:** Done. **Where it stands:** Done — steps 0-3 and 2b built (CPU tests with the tiny Gemma 3, the page driven in headless Chromium); step 4's commands given.
**Owner's runs:** pending: step 0 ("to run on the owner's card") and step 4.

No leftovers.

## Plan 16: One contract for every vision-language family; Qwen-VL ([16-qwen-vl.md](done/16-qwen-vl.md))

**Status:** Done. **Where it stands:** Done, with leftovers — steps 0-3, 5, 6 and 8 built 2026-10-13, step 4 done 2026-10-20 by plan 26 P6; steps 7 and 9 are the owner's; every decision for the owner taken.
**Owner's runs:** pending: steps 7 and 9 ("Built, 2026-10-13" table).

| # | Leftover | Kind | From |
|---|---|---|---|
| 16.1 | DeepStack (Qwen3-VL): image features added to the hidden states of several decoder layers, as a further generic output of the contract. | feature | Where one contract stops |
| 16.2 | Vision families that read the image by cross-attention (TrOCR, Donut, Llama 3.2 Vision): features for cross-attention with the decoder blocks that take them. The library has `CrossAttention` and an encoder-decoder engine since plan 26 (`Layers/CrossAttention.cs`, `EncoderDecoderSpec`), but no vision contract output uses them. | feature | Where one contract stops; plan 11 phase 10; plan 13 "Separate list" |

## Plan 17: Speed and memory for vision and OCR ([17-vision-ocr-speed.md](done/17-vision-ocr-speed.md))

**Status:** Done. **Where it stands:** Done, with leftovers — finished for CUDA and the CPU 2026-10-12 (`abstraction` 00d3e81), confirmed on the owner's cards and by the CER gate; Vulkan and HIP get the contracts' defaults (correct, no gain) and have never run this plan's tests or benches.
**Owner's runs:** pending: E9 on the smaller card ("Left for other devices", CUDA row).

| # | Leftover | Kind | From |
|---|---|---|---|
| 17.4 | JPEG decoder (C2), not done: MCU-row pipelining (no whole-image coefficients), restart intervals in parallel, h2v1 and h1v2 upsampling in vectors. | speed | C2 |
| 17.5 | C4's rest: `Orient` and the JPEG round trip's colour conversion; `jpeg=95` takes 94-108 ms, of which the optimized encode 70-88. | speed | C4 (part 1 and part 2) |
| 17.6 | `Idrak.AspNetCore` image uploads and `ChatImageDecoder` still give `FromBytes` fresh arrays (two large-object copies a page). | memory | F5, "Seen, not changed" |
| 17.7 | A device-resident tier of the feature cache (needs a device to measure). | speed | F8 |
| 17.8 | The batched path (`StreamBatch`) publishes no prefill and chunk telemetry events. | feature | B6 |
| 17.9 | The batched path makes its own decoding context per batch (no reuse across reads). | speed | B3 |
| 17.12 | The encoder pass recorded as a graph: not worth it after D2/D3; revisit if the owner's kernel profile shows gaps between the encoder's kernels. | speed | D7 |

## Plan 18: Synthetic document pages ([18-synthetic-pages.md](done/18-synthetic-pages.md))

**Status:** Done. **Where it stands:** Done, with leftovers — phases 1-6 and the later effect rounds built 2026-10-13/14; phase 7 (the proof) is the owner's.
**Owner's runs:** pending: phase 7 (tune on real + synthetic pages; CER on the same 20 pages before and after).

| # | Leftover | Kind | From |
|---|---|---|---|
| 18.1 | More shapers as registered entries: Indic (syllables and reordering), Thai, Khmer, Myanmar; the library registers the general OpenType shaper and the Arabic one only (`Idrak.Imaging/LibraryRegistrations.cs:57-61`). | feature | Phase 2b; "Phases" |
| 18.2 | CID-keyed CFF fonts are read but have no fixture. | check | Status (2b) |

## Plan 19: language and vision-language models from scratch ([19-vision-language-from-scratch.md](done/19-vision-language-from-scratch.md))

**Status:** Done. **Where it stands:** Done, with leftovers — tracks A (A0-A4) and B (B0-B5 and the decoder start) built and CPU-checked 2026-10-14; the two proof runs (A5, B6) are the owner's.
**Owner's runs:** pending: "Track A, as built" (the GPU filters and A5), "Track B, as built" (the GPU filters and B6).

| # | Leftover | Kind | From |
|---|---|---|---|
| 19.1 | Per-source losses in the progress record are not built (`PretrainingProgress` has no loss by source; images are counted per second, not scored apart). Checked: no `LossBySource` in `src/`. | feature | Track A, as built (A3); Track B, as built (Not built) |
| 19.2 | No contract suite for tokenizer trainers: the kit could not name core's `BpeTokenizer`, so a trained tokenizer goes through `TokenizerSuite` instead. Since Plan 24 L0 the trainers return `IPretrainedTokenizer` (in `Idrak.Abstraction`), so a `TokenizerTrainerSuite` is now possible. | feature | Track A, as built (A1) |
| 19.3 | Config writers (spec → config.json) exist for Llama, Mistral, Qwen2, Qwen3 and Gemma only. Gemma 3's text side, Gemma 2, Mixtral, DeepSeek and the rest refuse `idrak create` ("the rest as a follow-up"). | feature | Phase A1 |
| 19.4 | Answer scores during pretraining (B3's optional `--metric cer\|wer --metric-samples N` at `EvaluateEvery`) are not built: `idrak pretrain` has no `--metric`, and B6 scores with `tune evaluate` after the run. `TuningAnswerScorer` scores one request at a time (a batched path was named a later speed item). | feature | Phase B3 |
| 19.5 | A distillation teacher that cuts an image into a different number of blocks than the student is refused naming both counts; it is not re-tiled. | feature | Track B, as built (Not built) |

**Notes:** dropped as done: the three CLI checks that failed in A4 (`cli train data: help`, `cli polish: help lists`, `cli arabic`): later runs pass them (Plan 23 C8a's filter with `cli polish` and `cli train data`, 189 passed, 0 failed; Plan 24 T4's run with `cli arabic:`, 103 passed, 0 failed). Not listed: A3's AdamW epsilon finding, which the plan keeps on purpose ("Not changed").

## Plan 20: Idrak.Diffusion (image generation by denoising) ([20-diffusion.md](opened/20-diffusion.md))

**Status:** Not started. **Where it stands:** Planned, nothing built — planned 2026-10-15 against 82d84937; no `src/Idrak.Diffusion` exists (checked), only the reserved name in `LibraryDefaults`.
**Owner's runs:** none.

| # | Leftover | Kind | From |
|---|---|---|---|
| 20.1 | Phase D0, not started: the reference fixtures from diffusers (schedules, samplers, a tiny UNet, VAE and CLIP, a tiny pipeline, a pixel DDPM). | feature | Phase D0 |
| 20.2 | Phase D1, not started: the `Idrak.Diffusion` package with its contracts and registries (schedules, samplers, latent codecs, conditioners, objectives, families, data sources). | feature | Phase D1 |
| 20.3 | Phase D2, not started: the three schedules and four samplers (DDPM, DDIM, Euler, DPM++ 2M), matched to diffusers within 1e-6. | feature | Phase D2 |
| 20.4 | Phase D3, not started: the denoiser blocks and `UNet2D`, two new device operations (`GroupNormActivation`, `AddPerPlane`) on CPU, CUDA and Vulkan; bfloat16 convolution weights may go to a follow-up. | feature | Phase D3 |
| 20.5 | Phase D3b, not started: DiT (`DiTSpec`), after D6. | feature | Phase D3 (Risks), decision 5 |
| 20.6 | Phase D4, not started: the latent codecs ("pixels", `AutoencoderKl`) and measured tiled decoding. | feature | Phase D4 |
| 20.7 | Phase D5, not started: the generation API and `idrak diffuse`; `MeasuredBatches` moved from the CLI into `Idrak.Abstraction.Devices`. | feature | Phase D5 |
| 20.8 | Phase D6, not started: `DiffusionTrainer` with EMA, the latent cache and resumable checkpoints through `idrak create` / `pretrain` / `resume`. | feature | Phase D6 |
| 20.9 | Phase D7, not started: the Stable Diffusion 1.x / 2.x loader plug-in (diffusers layout); the single-file layout, SDXL and inpainting come after. | feature | Phase D7 |
| 20.10 | Phase D8, the owner's proof run (from scratch and a loaded checkpoint), after D6 and D7. | check | Phase D8 |
| 20.11 | LoRA on a loaded checkpoint's `Linear` layers for diffusion fine-tuning (a follow-up). A device-side Gaussian generator (a later measured item). | feature | What a from-scratch run is; Phase D3 (kernels table) |
| 20.12 | Decision 1: package plus plug-ins (`src/Idrak.Diffusion`, families in samples)? Recommended: yes. | question | Decisions (for the owner) |
| 20.13 | Decision 2: generic blocks (`UNet2D`, `DiT`, `AutoencoderKl`) in the package or in the plug-ins? Recommended: in the package. | question | Decisions (for the owner) |
| 20.14 | Decision 3: first families? Recommended: `IdrakDiffusionPipeline` and `StableDiffusionPipeline` (1.x, 2.x). | question | Decisions (for the owner) |
| 20.15 | Decision 4: training surface: `idrak pretrain FOLDER` on a diffusion folder (recommended) or `idrak diffuse train`? | question | Decisions (for the owner) |
| 20.16 | Decision 5: DiT in D3 or after D6? Recommended: after D6 (D3b). | question | Decisions (for the owner) |
| 20.17 | Decision 6: text conditioning from scratch? Recommended: class labels in D6, the plan 19 text model's hidden states next, CLIP only through the SD plug-in. | question | Decisions (for the owner) |
| 20.18 | Decision 7: a shared `MemoryLadder` in `Idrak.Abstraction` (recommended, adopted by `FineTuner` later) or a third copy in `DiffusionTrainer`? | question | Decisions (for the owner) |
| 20.19 | Decision 8: serving images (`idrak serve` with an image endpoint) is out of this plan; recommended: plan it with the engine's next change. | question | Decisions (for the owner) |

## Plan 21: Idrak.Audio (sound in and out) ([21-audio.md](done/21-audio.md))

**Status:** Done. **Where it stands:** Done, with leftovers — S0-S9 built and CPU-checked 2026-10-15 and 2026-10-16 (S6 and S7's training parts, voices by routes A, B and C included); the owner's proof run S8 ran 2026-10-16 on CUDA. The status line's "Next: the rest of S6, then S7's training parts" is history: both were built after it.
**Owner's runs:** pending: "Phase S8: the owner's proof run" (first run 2026-10-16, "S8 as run").

| # | Leftover | Kind | From |
|---|---|---|---|
| 21.1 | Segment batches through `Transcribe`: the Whisper plug-in decodes one segment at a time, though the encoder takes a batch; `TranscriptionOptions.MaxBatch` and `MemoryFraction` are not read by it (checked). | speed | Status ("Not in S4 yet") |
| 21.3 | Images onto the general soft-part path (`ISoftPartSource`) and the `Image*` soft-token types renamed (`ImageTokenLayout`, `ImageTokenFormat`, `ImageFeatures`, `ImagePrefill`); named for plan 14's backlog. | feature | Phase S3, decision 5 |
| 21.4 | RF64 WAV (past 4 GiB) is refused when read and not written ("a later item"). | feature | Phase S1 |
| 21.5 | Ogg-Opus and AAC decoders: plug-in plans only, none written (FLAC and MP3 are in `Idrak.Audio`; `plans/plugins/` holds HarfBuzz only). | feature | Phase S1, decision 6 |
| 21.6 | A CTC beam with a token-level language model, for spelling ("a later item"). | feature | What a from-scratch audio model is |
| 21.7 | A HiFi-GAN vocoder trained from scratch ("a later plan"); a from-scratch TTS model speaks through Griffin-Lim or a loaded HiFi-GAN of the same mel layout. | feature | Phase S6 (tts task), What a from-scratch audio model is |
| 21.8 | Route A: no speaker names shipped for VCTK (a `speakers.json` is the user's), and a multi-speaker checkpoint is not fine-tuned (route C takes one speaker's checkpoint). | feature | S9 route A as built |
| 21.9 | Route C: no dropout during fine-tuning ("a later setting"). | feature | S9 route C as built |
| 21.10 | uroman languages other than Arabic are refused until a romanisation table exists. | feature | Phase S5, decision 3 |
| 21.11 | `runs show` of a speech run picks its best step by the training loss and prints no validation column (found in S8; no later note says it was fixed). | bug | S8 as run |
| 21.12 | Not covered by a test: serve's audio parts and `voice pull` (no hub in the tests). | check | S7 as built, the training parts |

**Notes:** dropped as done: S4's sampling with the temperature fallback ("S4 addition: Whisper's guards", 2026-10-16); the `ctc/` fixture (S6 as built); S7's first "Not built" list (`data audio --align`, `create` with audio, `pretrain --audio`, `tune --audio`, `/audio`, the served engine's audio parts, `voice pull`; all in "S7 as built, the training parts"); route C's rerun after the duration-flow fix and the hiss ("The fix on the owner's card", 2026-10-16); the three pre-existing CLI check failures S7 names (see Plan 19's notes). No section of the plan was moved: its open items are remarks inside as-built notes.

## Plan 22: speed and memory everywhere, and the library owns what is generic ([22-performance-and-library.md](done/22-performance-and-library.md))

**Status:** Done. **Where it stands:** Done, with leftovers — steps 1-6h and 8 done (last commits named: 836a3141, 37e5faeb, d2b36cf0), step 9 measured and dropped, step 7 (measure every step with the benches) ongoing; section 4b's lookups not started.
**Owner's runs:** none.

| # | Leftover | Kind | From |
|---|---|---|---|
| 22.1 | Fewer launches a training step: a step recorded as a graph and replayed, which for batches of varying shape needs shapes that repeat ("a design of its own"). A VITS step is about 6,500 launches; CUDA's floor is about 6.7 µs a launch (about 43 ms a step). | speed | Section 1 (the launch row, "Open") |
| 22.2 | Indexed lookups, not started: `TensorScope` still scans its list backward on every `Keep`, `Owns` and `Untrack` (`LastIndex`, checked); name lookups in the weight readers and tensor stores. Each to be measured with the benches before and after. | speed | Section 4b |
| 22.3 | Input expansion (files, folders, wildcards) is still three copies in the CLI (`TextFiles.Expand`, `AudioFiles.Expand`, `ImageModelSupport.Expand`), not a library piece. | feature | Section 3, item 6 |
| 22.4 | Packed q/k/v helpers are not in the library (the Whisper plug-in keeps `WhisperWeights.PackedQkv`). | feature | Section 3, item 2 |
| 22.5 | Step 7: measure every step with the benches and record the numbers in the plan (ongoing). | check | Section 5, step 7 |

**Notes:** dropped as done: step 2's "exp and erf left" (step 8); the bfloat16-products row "to be run" and the open `SampleRows` case (both closed in section 1's last paragraph); section 3's other items (steps 6a-6h; the image-blocks rule and the romanizer and phonemizer contracts moved too). The eight-clip tuning row is a setting (`IDRAK_TUNE_NEIGHBOUR_OCTAVES=2`), not a leftover.

## Plan 23: Idrak.Cluster and Idrak.Network ([23-cluster.md](done/23-cluster.md))

**Status:** Done. **Where it stands:** Done, with leftovers — C0-C7, C8a, C8b, C8c and C9 built and merged 2026-10-18; every phase passes on CPU, CUDA and Vulkan except C9 (CPU and CUDA). The three-machine LAN run and a rental are still to come. (plans/README.md still says "planned 2026-10-18, nothing built".)
**Owner's runs:** pending: C9 on Vulkan (the status table); the LAN and rental runs (C5's "The owner's runs", "Runs").

| # | Leftover | Kind | From |
|---|---|---|---|
| 23.1 | The prediction is still 24% under the measured step after calibration (38.65 against 50.76 ms on `card.json`); the 12 ms left is not explained (the CPU rank's exposed network is only part of it). | check | Runs |
| 23.2 | The three-machine LAN run and a rental (`cluster bench` per pair, calibration, `pcie-host` and `nvlink` between cards) have not run; copies between two physical GPUs and peer access are untested (no machine with two cards). | check | C5 (The owner's runs), C5 as built, Runs |
| 23.3 | A shared network adapter is a static share in the topology; an event-ordered virtual clock sharing it only among transfers that overlap in virtual time is left open (a rewrite of `VirtualClock` and `SimTransport`). | feature | C3, C5, C8a as built |
| 23.4 | Under `Shard = parameters`, evaluation still gathers the whole weights (`Gather()`), `End()` gives every rank the whole model, and an optimizer's `SaveState` copies a stage's or a share's whole state to the host before it is streamed. | memory | C8b as built |
| 23.5 | `idrak profile MODEL --cluster` loads the whole model on every rank's host before placing its stage; C8b's own-tensor loads are used by `pretrain` only (checked: `Deferred` only in `PretrainCommand`). | memory | C8a as built ("C8b's stage loads fix that") |
| 23.6 | Sharding with a pipeline, and top-k with sharding, stay refused until a measured need (the LAN and rental runs, the advisor). | feature | C6, C8b as built |
| 23.7 | Pipeline placement for the fine-tuner and the `Trainer` (both refuse it, naming the pretrainer); under pipeline the pretrainer refuses conversation records, a teacher, the host optimizer, training the vision part and a late join. | feature | C7 as built |
| 23.8 | An interleaved pipeline schedule (virtual stages) to shrink the bubble, and 1F1B: not built; left until a GPU run's bubble asks for it (`IPipelineSchedule` takes one as a plug-in). | speed | C7 as built |
| 23.9 | `udp` (optional) and `rdma` (a plug-in): registered names with no implementation. | feature | Transports table, decision 5 |
| 23.11 | The dashboard's server messages (errors, run details) stay English on the Arabic page; a browser check of the page is the owner's. | feature | C9 as built (Deferred) |
| 23.12 | `LocalTransport`'s device handoffs still wake through a channel (the direct wake is in `FramedTransport` only). | speed | C8b as built |
| 23.13 | Every rank reads every sequence of a step and keeps its share; a reader that tells a sequence's length without reading it would spare that, left for a source slow to read. | speed | C2 as built |

**Notes:** dropped as done, each closed in a later phase's note: the measured `ChunkBytes` and bucket size in the tuning cache, `--profile` of a step's buckets, an address on a link override, `idrak transports` (C8a); device-side staging of finished buckets (C6); the waking inbox, a stage's and a shard's own-tensor loads, the streamed save, device-to-device activations, a pipeline stop's checkpoint, and the Trainer and fine-tuner answering a stop (C8b); the dashboard steering train and tune (C8b's merge with C8a); training on the simulator calling `BeginStep` (C4); the two GPU memory items after C6 and C7 (C8b; the status table now reads CUDA and Vulkan pass). No section of the plan was moved: its open items are remarks inside as-built notes and "Runs".

## Plan 24: Unigram (SentencePiece) and WordPiece tokenizers, built in ([24-tokenizers.md](done/24-tokenizers.md))

**Status:** Done. **Where it stands:** Done, with leftovers — L0, L1, T0-T4, library-owned Unicode normalization, the T3 callers, the T4 leftovers and two speed rounds built and pushed 2026-10-19 (7b8c85f, 147 of 147 targeted CPU tests); T5 measured, not taken.
**Owner's runs:** pending: "The owner's runs".

| # | Leftover | Kind | From |
|---|---|---|---|
| 24.1 | T5 (BPE onto the shared span pipeline): same ids, about half the speed, not taken; reopened only by the owner. A span mode without origins is the candidate if it is. | question | Phase T5, What is left (2) |
| 24.2 | Detailed BPE is 1.8 times the ids-only encode (a per-token origin lookup; `TokenizerEncodings.Resolve` reads the tokenizer.json's settings on every encode). | speed | What is left (3) |
| 24.3 | The byte-level span pre-tokenizer costs 42 of 124 ns a character on the detailed byte-level path. | speed | What is left (3) |
| 24.4 | Unigram's detailed encode realigns every cache hit (WordPiece's cached ranges would apply). | speed | What is left (3) |
| 24.5 | Each tokenizer compiles its patterns on load (about 6 ms for the long ones). | speed | What is left (3) |
| 24.6 | An nfkc or nmt_nfkc tokenizer.model is about 2 MB against sentencepiece's 240 KB (`DoubleArrayTrie` does not merge shared suffixes). | memory | What is left (3) |
| 24.7 | The Unigram trainer gains little from workers on the repeated fixture corpus (7.1 → 6.1 s on 16 MB) and peaks at about 270 MB of heap. | speed | What is left (3) |
| 24.8 | An ids-only encoding still holds the contract's six arrays (about 1.1-1.8 KB a text in a batch). | memory | What is left (3) |
| 24.9 | Truncation `stride` (overflowing windows) is not implemented. | feature | What is left (3) |
| 24.10 | Unigram trainer: sentencepiece's `_cf` rules (no case-folding table) are not built; `add_dummy_prefix` and `remove_extra_whitespaces` stay fixed. | feature | Finished after 2026-10-19, 3 (Not done) |
| 24.11 | Added tokens' `normalized`, `lstrip`, `rstrip` and `single_word` flags are not honoured (the readers write them false; nothing reads them; checked). | feature | T2 as built |
| 24.12 | A tokenizer.model's control pieces become added tokens and are matched in text, where sentencepiece never matches them (the known corpus lines); candidate: a per-added-token "not matched in text" flag for the .model path. | bug | T1 as built; T4 as built (cross-check) |
| 24.13 | The sentencepiece reader reads unused pieces (type 5) as normal ones; sentencepiece skips them (no fixture has one). | bug | T1 as built (Candidates left) |
| 24.14 | BPE's `DecoderStage` Metaspace strips one leading space whatever the prepend scheme, where tokenizers drops every ▁ of the first token unless the scheme is `never`. | bug | T1 as built (Candidates left) |
| 24.15 | In a no-cache Unigram pass the normalizer and pre-tokenizer are 38-64% of the time (the `Replace` regexes, Strip, sentencepiece-mode Precompiled over the whole text); a cold first pass on one thread runs tier-0 JIT code (3.2M tokens/s on the .model path). Measured at T1; the speed rounds did not touch them. | speed | T1 as built (Candidates left) |

**Notes:** dropped as done: L1's candidate (normalization in C# from tables) and T4's finding that NFD does nothing in the CLI ("Finished after 2026-10-19", 1); T1's "not yet in `--bench-text`" (the bench rows of item 3); T4's fixed SentencePiece choices (now settings, item 3). The owner's runs (item 1 of "What is left") stay in "The owner's runs".

## Plan 25: Idrak.Extensions.AI (no file)

**Status:** Dropped. **Where it stands:** Dropped — the owner dropped it 2026-10-19 (commit 57f2d6b); it was a draft the owner uploaded in the first session and never entered the repository.
**Owner's runs:** none.

No leftovers.

## Plan 26: model families as plug-in packages, and `Idrak.Nlp` renamed `Idrak.Text` ([26-plugin-packages.md](done/26-plugin-packages.md))

**Status:** Done. **Where it stands:** Done, with leftovers — P0-P10 built, merged and pushed 2026-10-20; leftover round L1 (rows 9, 10, 11, 13, 15, 17, 18) done 2026-10-20 and row 32 fixed 2026-10-22; the other rows wait for the owner ("no agent starts them before").
**Owner's runs:** pending: "The owner's runs" (each phase's note: the CUDA filters, the full suite for P0 and P7, the real BERT and T5 models).

| # | Leftover | Kind | From |
|---|---|---|---|
| 26.1 | To confirm: ArabicOcr's left-to-right reading order (the recognizer passes its direction to `runs`; right-to-left results identical, a left-to-right model's order changes). Keep it or restore the old order (one argument). | question | P3 |
| 26.2 | To confirm: renamed and removed public names (`WhisperSpeechPlugin` → `WhisperPlugin`, `VitsSpeechPlugin` → `VitsPlugin`, Gemma tensor names, `LlamaStyle` → `CommonArchitecture`, the tool-call formats, …). | question | P4, P6, P7 |
| 26.3 | To confirm: the plug-in packages let `Idrak.Tests` see their internals (the other choice: make them public). | question | P4 |
| 26.4 | To confirm: the tests register every text plug-in once at startup (`TextPlugins.RegisterAll()`), not per test. | question | P7 |
| 26.5 | To confirm: ArabicOcr's model folder is `config.json` + `model.safetensors`; LLaVA ships; a plug-in's `Register()` runs once until `Unregister()`. | question | P5 |
| 26.6 | The CRNN's backward LSTM reads a batch's right padding, so a line's reading depends on the batch it is in (still so in `CrnnRecognizer.cs`); a mask would allow bigger batches. | bug | P5 |
| 26.7 | ArabicOcr training is not reproducible with one seed (epoch-1 loss 2.78466, then 2.837; likely a CPU reduction order), so training cannot be proven unchanged (rules 71, 86). | bug | P5 |
| 26.8 | `idrak tune init` writes the text example for a Gemma 3 vision checkpoint unless `-P` is given (`TuneCommand.cs`: families listed only when plug-ins were passed). | bug | P6 |
| 26.12 | The CRNN prepares lines one after another (`CrnnRecognizer.Read`, rule 79), and `CrnnFamily.Read` builds random weights before overwriting them all. | speed | P5 |
| 26.14 | The public `SafeTensorsWriter.Write` holds every tensor's values at once; the streaming overload is internal. | memory | P5 |
| 26.16 | Batch measuring twice: `MeasuredLineBatches` (Vision plug-in) and the CLI's internal `MeasuredBatches` in `ImageModelSupport`; a candidate for one library contract. | feature | P5 |
| 26.19 | `PretrainedArchitectures`' refusal says "No architecture 'X' is registered", unlike the other registries' "is not registered" (still so). | bug | P6 |
| 26.20 | No test of `TextPhonemizers.First()` refusing when nothing is registered. | check | P4 |
| 26.21 | The coding agent's shell tool still starts processes in a library package (`CodingTools.cs`, `CodingAgent.cs` in `Idrak.Text`; for the owner if rule 70 is read widely). | question | P4 |
| 26.22 | The CRNN's tensor names are positional (`layers.{i}.parameters.{j}`): checkpoints from other tools cannot map to it. | feature | P5 |
| 26.23 | `DecoderSpec` still reads Idrak's earlier llama3 spec JSON (a file format, not family logic). | feature | P7 |
| 26.24 | Model names kept in the moved Gemma code's comments and test strings (the 4B config test, a "0.83 GB less" comment). | docs | P6 |
| 26.25 | The owner's machine: old sample `bin`/`obj` folders to delete (`samples/IdrakAudio`, `VitsSpeech`, `WhisperSpeech`, `OpenVoice`, `QwenVision`, `IdrakVision`, `Gemma3Vision`). | check | P4, P5, P6 |
| 26.26 | To confirm: ChatML stays a library GGUF fallback template; without `Idrak.Text.Plugins` a Llama 3 GGUF file without a template gets none; Gemma's GGUF query scalar follows llama.cpp's layer-count rule (a setting); Gemma 1 GGUF files stop at token 107. | question | L1-GGUF |
| 26.27 | Prepared GGUF folders are cached by file only (path, size, time): a folder prepared before a plug-in registered keeps its missing template or pre-tokenizer until deleted. | bug | L1-GGUF |
| 26.28 | `idrak plugins list` (`Registries.cs`) does not list `GgufChatTemplates`, `GgufPreTokenizers`, `GgufTokenizers` (only the GGUF architectures; checked). | feature | L1-GGUF |
| 26.29 | No fixture for a Gemma 3 GGUF converted without `tokenizer.model` (llama.cpp then writes a `gpt2` tokenizer). | check | L1-GGUF |
| 26.30 | `gemma3n` and the Gemma embedding architecture are not mapped for GGUF. | feature | L1-GGUF |
| 26.31 | `tools/gguf/make_fixtures.py` uses the unpinned PyPI `gguf` package; on Linux the torch pin installs the CUDA wheel (works on CPU). | bug | L1-GGUF |
| 26.33 | To confirm (L1-T5): new public API (`DistancePositionBias`, `Tensor.LogSoftmaxTopK`, `ITensorStore.SameValues`, `OnInvalidated`, the serve overloads); `Idrak.AspNetCore`'s `/chat/completions` naming a text-only model answers 400 instead of 404; the serve table and `server ps` gain a Kind column. | question | L1-T5 |
| 26.37 | The bucketed bias with a sliding-window cache uses window-relative positions (existing). | bug | L1-T5 |
| 26.38 | Agents of two lanes share one scratchpad folder; give each lane its own. | question | L1 |
| 26.39 | The padding scope is not kept open through a checkpointed backward recompute (as `EncoderMemory` today). | feature | P8 as built (Left) |
| 26.40 | A shared position bias that trains is recomputed by every layer; with a bias, a pass over several queries (encoder, training) still builds the full `[batch·heads, t, t]` scores (row 10 fixed decoding steps only). | speed | P8 as built (Measured, not taken) |
| 26.41 | Padded encoder attention passes no mask, so on a device without a per-range kernel for the head size it runs on the host; the training path with dropout builds its padding mask on the host. | speed | P8 as built (Measured, not taken) |
| 26.42 | `Seq2SeqGenerator` reads one token back each step and has no recorded-graph replay. | speed | P8 as built (Measured, not taken) |
| 26.43 | `TextEncoder` pads each batch to `MaxLength`, not its longest text (still so, checked). | speed | P9 as built (Measured, not taken) |
| 26.44 | `PretrainedEncoderModel.Hidden`/`Pooled`/`Logits` leave gradients on when the caller forgets `NoGrad`. | memory | P9 as built (Measured, not taken) |
| 26.45 | The masked-LM head scores every position. | speed | P9 as built (Measured, not taken) |
| 26.46 | The pooler weights exist twice (pooler and classification heads). | memory | P9 as built (Measured, not taken) |

**Notes:** rows 1-38 keep their numbers; rows 9, 10, 11, 13, 15, 17 and 18 are done (round L1, "Leftover rounds") and row 32 is fixed (2026-10-22, its row), so they are left out. Rows 26.39-26.46 are new: the "Measured, not taken" and "Left" lists of P8 and P9 that the table did not hold (P8's host-built bias index is row 10, done). Row 21 checked: `CodingTools.cs` and `CodingAgent.cs` are still in `src/Idrak.Text/Generation/` (Plan 27 R4 moved the coding task layer, not the agent's shell tool).

## Plan 27: reinforcement learning ([27-reinforcement.md](done/27-reinforcement.md))

**Status:** Done. **Where it stands:** Done, with leftovers — R0-R11 (with R3b, R3c and R4b) built, merged and pushed 2026-10-22; leftovers 1-5 done (R7-R11); leftover 6, a real multi-machine run, is the owner's.
**Owner's runs:** pending: "The owner's runs" (collected: "I will test all later"; the localhost paths of R3c, R7, R10 and R11 among them).

| # | Leftover | Kind | From |
|---|---|---|---|
| 27.1 | Leftover 6: a real multi-machine run (two machines, one GPU each, one LAN; commands in "The owner's runs"). No multi-process cluster over TCP has run `reinforce`, `tune` or `distill`. | check | Status; R3b, R3c, R7, R10 as built |
| 27.2 | R11's finding: a language-model run resumed after a stop ends within about 1e-7 of the uninterrupted run, not bit for bit: `FineTuningSession` draws its packing order from a `Random` no checkpoint keeps. A packing draw from (seed, step), or the generator's state saved, would make it exact but changes every run's packing. The owner decides. | question | Status; R11 as built |
| 27.3 | The CUDA measurement that the device's peak holds the generation cache or the activations, not both (the CPU test only checks that held memory does not grow over 20 iterations). | check | R3 as built |
| 27.4 | The `process` sandbox's memory limit counts the started process only, not its children. | feature | R4 as built |
| 27.5 | `CodingVerifier` (`agent-run`, `agent-check`) applies no language defaults: a task run there still names its commands. | feature | R4b as built |
| 27.6 | A double-DQN setting of its own for CartPole: measured, not taken. | feature | R6 as built |
| 27.7 | One process's `idrak train` Ctrl+C leaves `Fit` with a cancellation, so its "Stopped" line after `Fit` is never reached (found, not changed). | bug | R7 as built |
| 27.8 | A numeric run (classic environments) on `cluster sim` is refused. | feature | R9 as built (Not done) |
| 27.9 | Loading `idrak-reinforce.json` back into a policy to play it: the settings are written, but no command or builder reads them. | feature | R9 as built (Not done) |
| 27.10 | `--env-option` values are checked only by the kind's own checks. | feature | R9 as built (Not done) |
| 27.11 | The dashboard does not steer `distill`. | feature | R10 as built |
| 27.12 | Live faults on an in-process `reinforce` run are drawn at its gradient steps, while the page's "at step N" for a queued fault counts iterations (display only). | bug | R11 as built |
| 27.13 | The cost of steering a run was not measured (unsteered: one null check an iteration; with a control: a shared hold, a broadcast of one value, or a file read at most every poll). | check | R11 as built |

**Notes:** dropped as done: R2's missing "pass rate" counter (`src/Idrak.Reinforcement/ReinforcementTrainer.cs` and `ReinforcementOptions.cs` now report one; the dashboard shows it); R3's suggestion of `IPolicyLoss.NeedsReference` (`src/Idrak.Abstraction/Reinforcement/PolicyLosses.cs:164`); rollouts on a cluster (R3b), `reinforce` under `cluster sim` (R3c); ranks agreeing on a stop (R7); exact-size replay arrays (R8); `reinforce` for the classic environments and DQN (R9); on-policy distillation on a cluster (R10); the dashboard steering `reinforce` (R11); R4b's `tune` and `distill` skipping the languages' build folders (f3864ba, named in "The owner's runs"); R5's demo suite through `--env csharp` (wired; its run is in R4b's owner's block); the missing Arabic summary R11's run found (532d447). The status line's leftover list was replaced by a pointer; the "Left" remarks in the as-built notes stay as history.

## Plan 28: documents ([28-documents.md](opened/28-documents.md))

**Status:** Not started. **Where it stands:** Planned, nothing built — planned 2026-10-22 against 99872b8, widened 2026-11-18 against da19bfc (owner: PDF read, written, edited and signed, vector and image pages, forms and annotations, fonts written as TTF/OTF/WOFF/WOFF2, samples in the ignored `test-output/`, finished for any workload before plan 38; plan 38's PDF needs moved here; worked on the branch `documents` from `migration`); no phase started ("No phase starts before the owner names it"); no `IDocumentFormat` and no PDF code in `src/` (checked).
**Owner's runs:** none.

| # | Leftover | Kind | From |
|---|---|---|---|
| 28.1 | Phase D0, not started: the contract and its registries (`IDocumentFormat`, `DocumentFormats`, `IDocumentTextLayout`, `DocumentTextLayouts`, `IDocumentWriter`, `DocumentWriters`, `IDocumentPageDrawer`, `DocumentPageDrawers`, `IFontWriter`, `FontWriters`; C1-C25), the `lines` and `content` layouts, `DocumentFormatSuite`, `DocumentWriterSuite`, `DocumentPageDrawerSuite`, `FontWriterSuite`, a toy format and writer from `Idrak.PluginTests`. | feature | Phase D0 |
| 28.2 | Phase D1, not started: PDF the file (lexer, cross-references and streams, object streams, incremental updates, repair, filters, limits, document information). | feature | Phase D1 |
| 28.3 | Phase D2, not started: PDF encryption (standard handler revisions 2-6: RC4, AES-128, AES-256; empty or given password). | feature | Phase D2 |
| 28.4 | Phase D3, not started: PDF text (pages, the content interpreter, fonts to Unicode, widths, runs with boxes, the `lines` layout). | feature | Phase D3 |
| 28.5 | Phase D4, not started: PDF images and embedded files. | feature | Phase D4 |
| 28.6 | Phase D5, not started: the sample generator (writing into the ignored `test-output/documents/samples/`, PyPI versions pinned), docs, and the owner's `documents real:` group. | feature | Phase D5 |
| 28.7 | ~~Decision 5: text and images first, no rendering.~~ Answered 2026-11-18 by decisions 7-9: read and write, vector and image pages, pages drawn to pixels (D6, D7). | question (answered) | Questions for the owner (1) |
| 28.8 | Decision 6: the contract, the reader and the writer; users later except plan 38's. Confirm, or name the first user to add. | question | Questions for the owner (2) |
| 28.9 | ~~Forms, annotations, signatures and editing in place: in or later?~~ Answered 2026-11-18: in (decision 10; D9-D12). | question (answered) | Questions for the owner (3) |
| 28.10 | The Office plug-in's package name (`Idrak.Data.Plugins` proposed) and the variable names in C8, C15, C16 and C18. | question | Questions for the owner (4) |
| 28.11 | Phase D6, not started: PDF writing (image, vector and text pages, fonts embedded and subset, searchable hidden text, the same bytes from the same input, a `jpeg` encoder registered in `ImageEncoders`). | feature | Phase D6 |
| 28.12 | Phase D7, not started: PDF vector content read, pages drawn to pixels (the `coverage` drawer in `Idrak.Imaging`, a Type 1 font reader, Type 3 glyphs, shadings, transparency, a fill rule on the rasterizer, `Homography` / `Warp` shared). | feature | Phase D7 |
| 28.13 | ~~Reference reader and renderer from PyPI: agreed?~~ Answered 2026-11-18: agreed, their samples under the ignored `test-output/` (decision 13). | question (answered) | Open questions (2) |
| 28.14 | ~~The writer's default image encoding~~ Answered 2026-11-18: `Auto` (decision 11). | question (answered) | Open questions (3) |
| 28.15 | Phase D8, not started: fonts written (`truetype` built in, the one Idrak works from, with subsetting and legacy fonts converted once; `woff` read and written in a font plug-in project, the one proof of the plug-in path; no JSON font file, no OTF writer, no WOFF2); D6 embeds and plan 38 exports through it. | feature | Phase D8 |
| 28.16 | Phase D9, not started: forms (every field kind read, filled, created, flattened; appearances generated; XFA data read). | feature | Phase D9 |
| 28.17 | Phase D10, not started: annotations (every kind read, drawn, created, removed; redaction applied), links, bookmarks, destinations, page labels, the `tagged` layout. | feature | Phase D10 |
| 28.18 | Phase D11, not started: editing in place (incremental updates, rewrite, page operations, merge and split, saved with AES-256). | feature | Phase D11 |
| 28.19 | Phase D12, not started: signatures read and checked (CMS, chain, timestamps, revocation from the file) and signing (PAdES B-B, B-T, B-LT). | feature | Phase D12 |
| 28.20 | Phase D13, not started: any workload (`idrak bench documents`, large and many files, bounded memory, the owner's real files written back and drawn). | feature | Phase D13 |
| 28.21 | ~~Signing's trust by default~~ Answered 2026-11-18: only the app's certificates (decision 16). | question (answered) | Open questions (4) |
| 28.22 | ~~Redaction as a command~~ Answered 2026-11-18: `idrak data redact` in D10 (decision 17). | question (answered) | Open questions (5) |
| 28.23 | The font plug-in project's name: `Idrak.Imaging.Plugins` (proposed) or an existing plug-in project? (open question 6) | question | Open questions (6) |

## Plug-in points ([plug-in.md](done/plug-in.md))

**Status:** Done. **Where it stands:** Partly built — the outside plug-in path, items 1, 2 (most), 3, 4 (first slices, windowed kernels, mixture of experts), 5, 6, 8, 9, 10, 12, 14, 16, 26, the dataset loaders, provider-neutral serving and distillation are built; the rest of the ranked gaps is open.
**Owner's runs:** none in this file.

| # | Leftover | Kind | From |
|---|---|---|---|
| PI.1 | Item 2, still missing: tensor-by-tensor division, comparisons, a general `Gather` (only `GatherRows`). | feature | Ranked gaps, item 2 |
| PI.2 | Item 3: LongRoPE (Phi-3/4) as a built-in; GGUF's `yarn_log_multiplier` and `attn_factor` keys. | feature | Done since the survey, item 3 |
| PI.3 | Item 4: an activation registry (the closed `FeedForwardActivation`) and `DecoderNorm`; ALiBi; fused qkv (Phi-3). | feature | Remaining for item 4 |
| PI.8 | Item 5: decoder layers have no ONNX export translator; graphs and exports have one input and one output. | feature | Ranked gaps, item 5; Gaps inside the existing registries |
| PI.9 | Item 6: ONNX import of Erf (outside GELU), Floor/Ceil, Reciprocal and tensor-by-tensor Div (need tensor operations, PI.1); graph structural operations cannot be replaced. | feature | Ranked gaps, item 6; Gaps inside the existing registries |
| PI.10 | Item 7: model packages store only builder JSON, `DecoderSpec` or `GraphModule`; scalers only Standard and MinMax; tokenizers only character and word (`ModelPackage.cs:112-125`). | feature | Ranked gaps, item 7 |
| PI.11 | Item 8: tool-call formats not built in (GLM 4.5, Kimi K2, MiniMax, Command R, Granite, FunctionGemma); the pythonic template's `str()` round trip; no partial-argument streaming; Llama 3.1's built-in tools rendered back only with `builtin_tools`. | feature | Item 8: what remains |
| PI.12 | Items 9 and 10: KTO and IPO; a loss over hidden states; packing for preference batches; per-module ranks; IA3 and VeRA with a PEFT type registry. | feature | Items 9 and 10, "Still open" |
| PI.13 | Item 13: a stopping hook on token ids, logits processors and logit bias over the normal sampler, JSON-schema constrained sampling (the CLI's `run --schema` and the server's wait for it), speculative decoding. | feature | Ranked gaps, item 13 |
| PI.14 | Item 15: metric accumulators (F1, AUC, precision, recall; `Trainer`'s `Metric` is a batch mean); the `Checkpoint` callback saves weights only (stop checkpoints keep the optimizer since plan 23 C8b). | feature | Ranked gaps, item 15 |
| PI.15 | Item 17: an embedding model kind for the engine (so `idrak serve` can answer `/v1/embeddings`); `GenerationOptions.Extra` for unknown options. | feature | Ranked gaps, item 17 |
| PI.16 | Item 18: a pooling choice in `TextEncoder` (CLS, last token), an `IChunker`, vector metrics beyond dot and cosine. | feature | Ranked gaps, item 18 |
| PI.17 | Item 19: registered file formats skipped when picking hub files; a compression registry (`.zst`, `.xz`). | feature | Ranked gaps, item 19 |
| PI.19 | Item 21: checkpoint export: a `Write` on `ICheckpointFormat`, a GGUF writer. | feature | Ranked gaps, item 21 |
| PI.21 | Item 23: `Functions` and `Filters` dictionaries on `JinjaChatTemplate`. | feature | Ranked gaps, item 23 |
| PI.22 | Item 24: initializer delegates (Kaiming), activations by name in `Dense`. | feature | Ranked gaps, item 24 |
| PI.23 | Item 25: tool results as content blocks (`ToolResult.Content` is a string). | feature | Ranked gaps, item 25 |
| PI.25 | Dataset loaders: no loader for audio or video; `ISampleSource` reads one sample at a time (a batched read for memory-mapped sources). | feature | Dataset loaders, "Remaining" |
| PI.26 | Distillation: a vocabulary mapping (if asked for; `distill` needs one tokenizer for logits); speed and memory with real models on a GPU not measured. | feature | Teacher pattern, status |

**Dropped as done:** item 11 (plan 24); item 12 (`GgufArchitecture.Configure` and `TensorName` delegates, `GgufArchitectures.cs`); item 16 (evaluation scores by any `TuningMetrics` name, `Evaluation.cs:39-52`); item 2's Erf (`Activations.Erf`, `GeluErf`) and public checkpointing (`Checkpointing`); item 4's ring-buffer cache (plan 17 E1's sliding-window caches) and encoder-only models (plan 26); ONNX ConvTranspose (plan 13 step 2) and outside translators' weight loading (item 1); the keyword index registry (`KeywordIndexes`); the JPEG codec (`JpegCodec`, plan 11 phase 2); "What needs the public backend" (plan 10 phase 4); the provider-neutral forwarders (removed, not kept); "Not confirmed: DoRA as plain LoRA" (item 10's PEFT checks).

## Plug-in 1: Idrak.HarfBuzz ([plugins/1-harfbuzz.md](opened/plugins/1-harfbuzz.md))

**Status:** Not started. **Where it stands:** Planned, nothing built — planned 2026-10-14; the owner's decision: build it when a new font or script needs checking, or an app wants HarfBuzz to shape.
**Owner's runs:** none yet (phase 4's seven-font run comes with the build).

| # | Leftover | Kind | From |
|---|---|---|---|
| H.1 | Phase 1: `HarfBuzzShaper : ITextShaper` ("harfbuzz") in `samples/HarfBuzz/Idrak.HarfBuzz`: faces from `IFontFace.Table`, made once per face and shared between threads, registered by name only. | feature | Phase 1 |
| H.2 | Phase 2: named shapers in `TextShapers` (`RegisterNamed`, `Named`, `Shape(..., shaper: name)`; the library's as "general" and "arabic"). | feature | Phase 2 |
| H.3 | Phase 3: `idrak data fonts --against NAME`, `TextStyle.Shaper`, `idrak data synth --shaper`, the Legal OCR app's `Synthetic.Shaper`. | feature | Phase 3 |
| H.4 | Phase 4: the plug-in's own test runner; the owner's run on the seven Google Fonts families ("200 of 200 lines identical"). | check | Phase 4 |

## Plug-in plans index ([plugins/README.md](opened/plugins/README.md))

**Status:** Not started. **Where it stands:** Planned, nothing built — the index of optional plug-in packages; its one plan (HarfBuzz) is not started.
**Owner's runs:** none.

No leftovers.

## idrak-cli: one command-line tool ([idrak-cli.md](done/idrak-cli.md))

**Status:** Done. **Where it stands:** Done, with leftovers — every command built on `idrak-cli`, merged and polished; the "Gaps" paragraphs name what the library lacks.
**Owner's runs:** pending: "Arabic messages" (the terminal checks, "Not verifiable here").

| # | Leftover | Kind | From |
|---|---|---|---|
| CLI.1 | `report --bench` times float32 products only; model speeds from `idrak bench` are not in the report. | feature | Gaps (Health) |
| CLI.3 | `doctor` checks the Vulkan driver files on Linux only (Windows keeps them in the registry). | feature | Gaps (Health) |
| CLI.4 | The test runner's `--list-devices` can move onto `DeviceListing`. | feature | Setup and health, "Built with the group" |
| CLI.5 | `embed` has no embedding heads for decoder models trained as embedders (last-token pooling, instruction prefixes). | feature | Gaps (Run) |
| CLI.6 | `agent`: `CodingAgent` has no approval hook, so the command runs its own tool loop. | feature | Gaps (Run) |
| CLI.7 | `idrak serve` answers `/v1/embeddings` with 404 (no embedder over a pretrained language model; see PI.15). | feature | Gaps (Serve) |
| CLI.8 | The chat API's `/api/generate`, `/api/show`, `/api/embed` and `/api/pull` are not provided. | feature | Gaps (Serve) |
| CLI.9 | Server-side tools in the chat APIs: `serve --tools` works with `--mcp` only; chat requests get client-side tool calls. | feature | Gaps (Serve) |
| CLI.10 | `ps` reports a model's memory as taken at load (the engine does not expose a model's size). | feature | Gaps (Serve) |
| CLI.11 | `mcp serve` speaks over standard input/output only (HTTP needs the MCP ASP.NET Core package). | feature | Gaps (Serve) |
| CLI.12 | `memory` (and `suggest`/`explain` memory checks) show "?" for GPUs unless a limit or `--memory` is given (`MemoryCommand.cs:45, 205`), though the library now reports the device's memory (`BackendHardware`, `Backend.FreeMemory`). | feature | Gaps (Models); Gaps (Design) |
| CLI.14 | No GGUF writer and no language model from an `.ikm`, so `convert` writes Hugging Face folders only (see PI.19). | feature | Gaps (Models) |
| CLI.15 | `idrak resume` continues from the weights with a fresh optimizer and schedule (the `Checkpoint` callback keeps weights only). | feature | Gaps (library), Train |
| CLI.16 | Writing Parquet is not in the library (`data convert` writes JSON Lines, JSON, CSV, TSV). | feature | Gaps (library), Train |
| CLI.17 | `tune init` writes the library's defaults until `idrak suggest` can size them. | feature | Gaps (library), Train |
| CLI.18 | No embedding-model loader (a bi-encoder from a Hugging Face embedding model): `rag index -m` embeds with a chat model's averaged hidden states; `RetrievalIndex.Save` keeps no vectors of a non-`TextEncoder` embedder. | feature | Retrieval and measure, "Gaps" |
| CLI.19 | prep.json's "augment" list is not applied by `train` and `suggest --search`. | feature | Gaps (Design); plug-in.md, dataset loaders "Remaining" |
| CLI.21 | `shell` completes command names and options only; model names and file paths are to come from the shared completion helper. | feature | Gaps (Developers) |
| CLI.22 | Later, not in this build: a plug-in marketplace listing; remote devices (a command run on another machine's `idrak serve`). | feature | More from a second pass, "Later" |
| CLI.23 | Arabic: the bodies of commands' help and other commands' messages are English; a streamed Arabic answer appears line by line; the user's typing is echoed unreordered where the terminal has no bidirectional support. | feature | Arabic messages, "Remains" |
| CLI.24 | Arabic in real terminals (Windows Terminal, console host, Git Bash, a Linux terminal): `help`, `doctor`, `devices --lang ar` and `--lang-render logical`. | check | open-questions, hardware check 11; Arabic messages |
| CLI.25 | R9 leftovers: `TrainSession.cs` CSV relabelling (strings and `Split` per row); `ImageTraining.cs:906` uploads two tiny per-channel tensors per batch; `ServeHost` auth allocates two small byte arrays per request; the bidi algorithm allocates per right-to-left line; `DataProfile` keeps each row's JSON string. | speed | Plan 14 backlog (R9) |

**Dropped as done:** `/stats` memory (the device's usage, `RunSupport.cs:167`); eval metrics waiting on plug-in gap 16 (`TuningMetrics` names); JPEG training only with a registered codec (`JpegCodec` is a library default).

## Open questions and checks (removed 2026-10-22: its items are filed under their plans here)

**Status:** Removed. **Where it stands:** In progress — of eleven hardware checks, one is open, filed under idrak-cli (check 11); the open backend decision (4) and checks (1, 4, 5, 6, 8, 10) were removed with the backend plans on 2026-11-14.
**Owner's runs:** the hardware check above (filed as CLI.24).

No leftovers left here. Answered and dropped: decision 1 (`architecture` merged into `main` 2026-10-04), 2 (the public backend API built by plan 10 phase 4), 3 and 8 (done), 5 (the README tagline reads "Deep learning in pure .NET. Every GPU. No native dependencies."), 6 (LinkedIn series dropped), 7 (native Linux drivers from a live USB when at hand; the run is 1.15), 9 (port 7317, then `-p`, `IDRAK_PORT`, `serve.port`). Checks done: 2 and 9 (the "fine-tuning" and "window kernels" filters in plan 17's round-3 CUDA run), 3 (superseded by the later CUDA and Vulkan runs of plans 17 and 23), 7 (the tool run on the owner's card, plan 11 phase 7).

Decided 2026-10-03 (kept here, the file's history; idrak-cli.md's "Arabic messages" cites the Arabic rule): merging into `main` not yet, then `architecture` merged into `main` 2026-10-04 with the version set to 0.2.0; the public backend API parked (built later by plan 10); the LinkedIn series dropped for now; native Linux GPU drivers tested from a live USB when one is at hand; `idrak serve` on a port of its own (7317, then `-p`, `IDRAK_PORT`, `serve.port`); Arabic messages only if the console shows them right to left with joined letters, otherwise dropped (built with the tool's own shaping and reordering, `--lang ar`); `idrak-cli` fast-forwarded into `architecture` and retired; round 3 (mixture of experts, sliding-window and soft-cap kernels, dataset loaders, `idrak distill`, provider-neutral serving names, Arabic output) built and merged.

## GPU checks for plans 12 and 13 ([gpu-checks.md](done/gpu-checks.md))

**Status:** Done. **Where it stands:** In progress — the owner's GPU runs of plans 12 and 13; plan 17's CUDA runs covered several filters (conformance kit, attention spans, vision tuning, image prefill, vision kernels, fine-tuning, packed); the rest on CUDA and every Vulkan run are not recorded as run.
**Owner's runs:** pending: the whole file ("Correctness, every device", "Speed", "The real model").

No leftovers.

## The moved text

Each section or paragraph moved out of a plan on 2026-10-22, as it was written there.

### done/12-vision-tuning.md

> 1. Train the projector by default for Gemma 3, or the language model only (LlamaFactory's run: language model only)?
> 2. Feature cache on disk by default, or memory first and disk when it does not fit?

### 14-refactor-backlog.md

> Everything left unfinished by plans 12 and 13, the fused recurrent work and the optimization-rules audit
> (`docs/optimization.md`, eleven areas R1-R11), in one place, so the refactor can be planned from it. Each item says
> where it is, why it was left, and what a fix needs. Nothing here is guessed: every entry comes from an agent's report,
> a measurement on the owner's card, or a count over the branch.
>
> ## State at the time of writing
>
> | Work | State |
> |---|---|
> | Audit R5 Layers/Models, R9 CLI N-Z + Shared | finished and merged |
> | Audit R4, R6, R7, R8, R10, R11 | stopped part-way; their committed fixes merged; the rest of their areas not covered (below) |
> | Whole solution after the merges | builds, 0 errors, 0 warnings; tests after the merge are the owner's |
>
> ## 1. Correctness and rule breaks (fix first)
>
> | # | Where | What | Fix needs |
> |---|---|---|---|
> | 1.2 | `src/Idrak/Models/GgufModel.cs:297` | an unregistered `tokenizer.ggml.pre` silently gets Llama 3's pattern: a family fallback CLAUDE.md forbids | throw "pre-tokenizer '{pre}' is not registered; register its pattern with GgufPreTokenizers.Register", and change `CustomGgufPreTokenizer` in tests/Idrak.Tests/ModelPluginTests.cs, which asserts the fallback |
> | 1.4 | README.md (51), plans/*.md (~55), src/Idrak/README.md (3), docs/performance-notes.md, docs/optimization.md | card names in benchmark tables and notes | owner's decision: a results table names the machine it ran on; advice or defaults must not |
> | 1.7 | ArabicOcr sample `TrainCommand` | a non-finite loss crashes writing `recognizer.json` (JSON cannot hold ∞) | stop with a clear message on a non-finite loss (sample) |
>
> ## 2. Speed and memory left in the library
>
> ### Layers and models (R5, not fixed)
>
> - `ActivationMemory.Release/Compress` take `params Tensor?[]`: an array per layer per step (also at inference, before
>   the early return). `params ReadOnlySpan<Tensor?>` fixes every caller; public API change (regenerate the dump).
> - `GraphModule.Binary` and `LibraryGraphOps.Scalar` read scalar constants back from the device every forward (a GPU
>   sync per node); BatchNorm in eval mode recomputes its inverse standard deviation every call. Both need a buffer
>   version or invalidation hook in Idrak.Abstraction (`Module.Load` can overwrite buffers in place).
> - `Conv2d.ForwardFused` refolds the batch norm into a weight-sized temporary on every inference call (deliberate: no
>   second copy of the weights). Owner's trade-off: cache the folded weights (memory) or keep refolding (time).
> - `Sequential` allocates one closure per whole-model forward.
>
> ### CLI (R9, not fixed)
>
> - `TrainSession.cs:378-412` CSV relabelling for string classes: strings and `Split` per row (rule 74); move to bytes
>   with the library's CSV reader.
> - `ImageTraining.cs:906` uploads two tiny per-channel tensors per batch.
> - `ServeHost` auth: two small byte arrays per request with an API key.
> - The bidi algorithm allocates per right-to-left line.
> - `DataProfile` duplicate detection keeps each row's JSON string (hashing could change results).
>
> ### Gaps the OCR sample showed (the library should offer them)
>
> - The `Augmentations` registry has no `noise` (core's `GaussianNoise` is only a sample transform): the sample
>   registers its own.
> - The measured batching `idrak predict` uses (`MeasuredBatches`) is internal to the CLI: the sample copies it. Make it
>   a library contract.
> - `ChatImage` carries no source path: pages from a data file are named by record number.
> - No training-step profile outside CUDA: the CLI's operation/layer `Recorder` (ProfileCommand) is internal; a library
>   hook would give every app the per-operation table on any device (rule 78).
> - The OCR sample's segmentation keeps touching lines as one line (sample).
>
> ## 3. Audit areas not finished
>
> The agents of these areas were stopped part-way. "Untouched" = no fix committed: either not reviewed or reviewed and
> found clean; the stopped agents cannot say which, so each needs a review pass. Largest first.
>
> | Area | Files | Untouched | Lines | Largest untouched |
> |---|---|---|---|---|
> | R4 Abstraction (not Devices) | 75 | 59 | 11,093 | Tensor.Decoder.cs, ChatParts.cs, VisionLanguage.cs, Operations/Kernels.cs, Decoding.cs, RopeScalings.cs, KeyValueLayouts.cs, Data/ImageTransforms.cs, Telemetry.cs, DecoderSpec.cs, Tensor.Quantized.cs, SlotTable.cs, Augmentations.cs |
> | R6 core (not Layers/Models) | 57 | 37 | 6,099 | JpegEncoder.cs, OnnxBuiltIns.cs, InferenceEngine.cs, ModelPackage.cs, Offloading.cs, FileSources.cs, ChatImageDecoder.cs, TrainerCallbacks.cs, OnnxExport*.cs, CtcDecoders.cs, AdamW8Bit.cs |
> | R7 Nlp, AspNetCore, Mcp, Onnx.Runtime | 54 | 32 | 8,644 | Jinja.cs, JinjaBuiltins.cs, TextGenerator.cs, FineTuningLosses.cs, CodingAgent.cs, TuningVision.cs, AnswerScorer.cs, ChatGenerator.cs, FeatureCacheImplementations.cs, Conversation.cs, ChatApi.cs, TokenSamplers.cs |
> | R8 CLI commands A-M | 53 | 26 | 3,767 | DoctorCommand.cs, ConfigCommands.cs, InspectCommand.cs, TuningShowCommand.cs, CompletionCommand.cs, AliasCommands.cs, NewCommand.cs, LoginCommands.cs (mostly cold commands) |
> | R10 Data, Vision, Testing kit | 84 | 79 | 17,078 | ParquetFile.cs, DataFiles.cs, VisionAugmentations.cs, AnnotationFiles.cs, DatasetRows.cs, Sources.cs, DetectionMetrics.cs, ChatRows.cs, AugmentedImageLoader.cs, RegionClassifier.cs, DatasetRecipe.cs, Downloader.cs, ImageWarp.cs |
> | R11 samples and tools | 56 | 42 | 7,349 | Gemma3Vision.cs, Samples.Chat, Shared/Gpt/CharGpt.cs, Samples.Rag, Siglip.cs, GptService.cs, CodingAgent, Summarizer, ArabicOcr/Readers.cs |
>
> The hot ones, by the rules: R10's data readers and image code (rules 72-74 came from exactly this code), R7's
> tokenizer-adjacent paths (Jinja runs per chat turn, TextGenerator/ChatGenerator per token), R4's tensor and decoding
> code, R6's JPEG encoder and inference engine. R8's remaining files are cold commands.
>
> Also never audited: the analyzers of rule 65 (`AnalysisModePerformance` = All) are not on; turning them on for every
> project will list what the review missed (warnings are errors, so fix in the same pass).
>
> ## 4. Checks never run
>
> - `plans/gpu-checks.md`: every CUDA and Vulkan run of plans 12 and 13 (conformance kit, attention spans, vision
>   tuning, image prefill, vision kernels, conv, vulkan cnn, vision layers/sequence/detection/augment, image families,
>   fine-tuning, packed, checkpoint) and the benchmarks (`--bench-spans train`, `--bench-conv`).
> - Plan 12 phase 5: the real Gemma 3 model (`tune evaluate`, `tune`, `run --adapter`).
>
> ## 5. Suggested order
>
> 1. Correctness: 1.2 (GGUF fallback), 1.7.
> 3. Library contracts the samples need (section 2, gaps): `MeasuredBatches`, a profiling hook on any device, `noise`
>    in `Augmentations`, a source path on `ChatImage`.
> 4. Invalidation hook for buffers (GraphModule constants, BatchNorm eval, folded convolution) and `ActivationMemory`
>    spans: one public-API change, several speedups.
> 6. Finish the audit of the six unfinished areas (section 3), hottest first (R10, R7, R4, R6), then the analyzers.

### done/16-qwen-vl.md

> Some families change what the decoder computes, not only what enters the prompt: Qwen3-VL adds image features to the
> hidden states of several decoder layers (DeepStack); TrOCR, Donut and Llama 3.2 Vision read the image by cross-attention.
> Each is a further generic output of the contract (features with the layers to add them at; features for cross-attention
> with the decoder blocks to take them), in a later plan. Qwen2.5-VL and Qwen2-VL need neither.

### opened/20-diffusion.md

> 1. **Package plus plug-ins, as above?** Recommended: yes. `src/Idrak.Diffusion` for the machinery; families only in
>    `samples/IdrakDiffusion` and `samples/StableDiffusion`. It is the split plans 13 and 19 proved and the rules require.
> 2. **Generic blocks (`UNet2D`, `DiT`, `AutoencoderKl`) in the package, or in the plug-ins?** Recommended: in the
>    package, every shape a spec field and no checkpoint name in them (plan 10's table already lists them there); the
>    plug-ins own names, configs, tokenizers and composition. If the owner prefers plan 19's placement (the tower in the
>    plug-in), the blocks move to `samples/IdrakDiffusion` and the SD plug-in references that sample.
> 3. **First families:** recommended: `IdrakDiffusionPipeline` (pixel and latent UNet, class or decoder-hidden-state
>    conditioning) and `StableDiffusionPipeline` (1.x and 2.x, diffusers layout). SDXL and the single-file layout follow
>    when a checkpoint of the owner's needs them.
> 4. **The training surface:** `idrak pretrain FOLDER` on a diffusion folder (recommended: one command for every
>    from-scratch run, `resume` and `runs show` unchanged) or a separate `idrak diffuse train`.
> 5. **DiT in D3 or after D6?** Recommended: after D6 (D3b): the contracts do not change for it, and the UNet proves the
>    kernels, graphs and the loop first.
> 6. **Text conditioning from scratch:** recommended: class labels in D6; the plan 19 text model's hidden states as a
>    conditioner in the IdrakDiffusion plug-in right after; CLIP only through the SD plug-in.
> 7. **A shared `MemoryLadder` in `Idrak.Abstraction`** (the rung order fine-tuning, image training and now diffusion
>    each implement) or a third copy inside `DiffusionTrainer`? Recommended: the shared one, built in D6 and adopted by
>    `FineTuner` in a later refactor (plan 14's list).
> 8. **Serving** (`idrak serve` with an image endpoint through the engine's model kinds) is out of this plan; recommended:
>    plan it with the engine's next change, not here.

### done/22-performance-and-library.md

> Where a lookup walks a list or compares names in a loop, give it an index: candidates are `TensorScope`'s backward scan
> of its list on every `Keep`, `Owns` and `Untrack` (long scopes in a training step), name lookups in the weight readers and
> tensor stores; each is measured with the benches before and after.

### done/24-tokenizers.md

> 1. **The owner's runs** ("The owner's runs"): the targeted Windows CPU list, CUDA `pretrained:|gguf:|tokenizers:
>    gguf|retrieval: text encoder`, the CLI on real models, `make_fixtures.py --real` and `tokenizers real:`, and
>    `--bench-text tokenizers trainers`. Their results go in the Runs table.
> 2. **T5** (BPE onto the shared span pipeline): measured 2026-10-19 (its section): the same ids, about half the speed;
>    not taken. Reopened only by the owner.
> 3. **Measured, not taken** (each a candidate for the owner to schedule; none is started without the owner):
>    - detailed BPE is still 1.8 times the ids-only encode (a per-token origin lookup; `TokenizerEncodings.Resolve`
>      reads the tokenizer.json's settings on every encode);
>    - the byte-level span pre-tokenizer costs 42 of 124 ns a character on the detailed byte-level path;
>    - Unigram's detailed encode realigns every cache hit (WordPiece's cached ranges would apply);
>    - each tokenizer compiles its patterns on load (about 6 ms for the long ones);
>    - an nfkc or nmt_nfkc tokenizer.model is about 2 MB against sentencepiece's 240 KB (`DoubleArrayTrie` does not
>      merge shared suffixes);
>    - the Unigram trainer gains little from workers on the repeated fixture corpus (7.1 → 6.1 s on 16 MB) and peaks at
>      about 270 MB of heap;
>    - an ids-only encoding still holds the contract's six arrays (about 1.1-1.8 KB a text in a batch);
>    - truncation `stride` (overflowing windows) is not implemented.

### done/26-plugin-packages.md

> The owner takes these after this plan is finished; no agent starts them before.
>
> **By priority, for the discussion (2026-10-20, after P7).** The owner's decisions first (answers, not code; the
> renames settle the public API before anything builds on it), then wrong or unprovable results, bugs, speed and memory
> (the library's main goal), missing contracts and pieces, and the small ones. Severity: 🔴 wrong or unprovable results,
> 🟡 a bug, a missing piece, speed or memory, or a decision to confirm, 🟢 small. The details of each row are in the phase
> notes below.
>
> | # | Kind | Sev | Leftover | From |
> |---|---|---|---|---|
> | 1 | decide | 🟡 | To confirm: ArabicOcr's left-to-right reading order | P3 |
> | 2 | decide | 🟡 | To confirm: renamed and removed public names (Whisper and VITS plug-ins, Gemma tensor names, `LlamaStyle` → `CommonArchitecture`, the tool-call formats, …) | P4, P6, P7 |
> | 3 | decide | 🟡 | To confirm: plug-in packages let `Idrak.Tests` see their internals | P4 |
> | 4 | decide | 🟡 | To confirm: the tests register every text plug-in once at startup | P7 |
> | 5 | decide | 🟡 | To confirm: ArabicOcr's model folder is `config.json` + `model.safetensors`; LLaVA ships; `Register()` runs once | P5 |
> | 6 | results | 🔴 | The CRNN's backward LSTM reads a batch's right padding: a line's reading depends on the batch it is in | P5 |
> | 7 | results | 🔴 | ArabicOcr training is not reproducible with one seed: training cannot be proven unchanged (rules 71, 86) | P5 |
> | 8 | bug | 🟡 | `idrak tune init` writes the text example for a Gemma 3 vision checkpoint unless `-P` is given | P6 |
> | 9 | bug | ✅ | **Done (L1-T5).** ONNX export of a `TransformerEncoderLayer` with a custom activation throws an invalid cast (existing) | P10 |
> | 10 | speed | ✅ | **Done (L1-T5).** Speed: the bucketed bias index rebuilt on the host each cached step; attention with a bias scores every cache slot | P8, P10 |
> | 11 | speed | ✅ | **Done (L1-T5).** Speed: beam search reads all beams x vocabulary logits back every step | P10 |
> | 12 | speed | 🟡 | Speed: the CRNN prepares lines one after another, and builds random weights before overwriting them all | P5 |
> | 13 | memory | ✅ | **Done (L1-T5).** Memory: a tied checkpoint that also holds `lm_head.weight` is compared value by value on the host at load | P10 |
> | 14 | memory | 🟡 | Memory: the public `SafeTensorsWriter.Write` holds every tensor's values at once | P5 |
> | 15 | contract | ✅ | **Done (L1-GGUF).** GGUF fallback chat templates (Llama 3's, ChatML) still chosen in the library; needs a template-fallback contract | P7 |
> | 16 | contract | 🟡 | Batch measuring twice: `MeasuredLineBatches` (plug-in) and the CLI's in `ImageModelSupport` | P5 |
> | 17 | feature | ✅ | **Done (L1-T5).** `idrak serve` cannot serve an encoder-decoder (`idrak run` can) | P10 |
> | 18 | feature | ✅ | **Done (L1-GGUF).** No GGUF mapping for Gemma (`GgufArchitecture.Configure` is now the hook) | P6, P7 |
> | 19 | small | 🟢 | `PretrainedArchitectures`' refusal says "No architecture 'X' is registered", unlike the other registries | P6 |
> | 20 | small | 🟢 | No test of `TextPhonemizers.First()` refusing when empty | P4 |
> | 21 | small | 🟢 | The coding agent's shell tool still starts processes in a library package | P4 |
> | 22 | small | 🟢 | The CRNN's tensor names are positional: checkpoints from other tools cannot map to it | P5 |
> | 23 | small | 🟢 | `DecoderSpec` still reads Idrak's earlier llama3 spec JSON | P7 |
> | 24 | small | 🟢 | Model names kept in the moved Gemma code's comments and test strings | P6 |
> | 25 | small | 🟢 | The owner's machine: old sample `bin`/`obj` folders to delete | P4, P5, P6 |
> | 26 | decide | 🟡 | To confirm: ChatML stays a library GGUF fallback template; without `Idrak.Text.Plugins` a Llama 3 GGUF file without a template gets none; Gemma's GGUF query scalar follows llama.cpp's layer-count rule (a setting); Gemma 1 GGUF files stop at token 107 (llama.cpp's end-of-turn rule) | L1-GGUF |
> | 27 | bug | 🟡 | Prepared GGUF folders are cached by file only (path, size, time): a folder prepared before a plug-in registered keeps its missing template or pre-tokenizer until deleted | L1-GGUF |
> | 28 | small | 🟢 | `idrak plugins list` (`Registries.cs`) does not list `GgufChatTemplates`, `GgufPreTokenizers`, `GgufTokenizers` | L1-GGUF |
> | 29 | small | 🟢 | No fixture for a Gemma 3 GGUF converted without `tokenizer.model` (llama.cpp then writes a `gpt2` tokenizer) | L1-GGUF |
> | 30 | feature | 🟢 | `gemma3n` and the Gemma embedding architecture are not mapped for GGUF | L1-GGUF |
> | 31 | small | 🟢 | `tools/gguf/make_fixtures.py` uses the unpinned PyPI `gguf` package; on Linux the torch pin installs the CUDA wheel (works on CPU) | L1-GGUF |
> | 32 | bug | 🔴 | `operations: every compute method of Backend is an operation` fails on the branch: `CopyTo`, `CopiesDirectlyTo` (Plan 23 C5, f483339), `UploadRange`, `LiveStorage` (Plan 23 C8b, 8e79901) are public virtual `Backend` methods that are neither operations nor listed as device plumbing (`tools/operations/generate.py`). **Fixed 2026-10-22:** all four are copy and memory plumbing (`LiveStorages` is the method's name); added to the test's `DevicePlumbing` list (`tests/Idrak.Tests/OperationTests.cs`), the only place it is kept | Plan 23, found in L1-T5 |
> | 33 | decide | 🟡 | To confirm (L1-T5): new public API (`DistancePositionBias`, `Tensor.LogSoftmaxTopK`, `ITensorStore.SameValues`, `OnInvalidated`, the serve overloads); `Idrak.AspNetCore`'s `/chat/completions` naming a text-only model answers 400 instead of 404; the serve table and `server ps` gain a Kind column | L1-T5 |
> | 37 | bug | 🟢 | The bucketed bias with a sliding-window cache uses window-relative positions (existing) | L1-T5 |
> | 38 | small | 🟢 | Agents of two lanes share one scratchpad folder; give each lane its own | L1 |
>
> In the phases' words:
> - **ArabicOcr's left-to-right reading order (P3).** The recognizer now passes its direction to `runs`: right-to-left
>   results are identical; a left-to-right model's order changes (the old code reversed Latin runs, which did not match
>   its labels). Keep it, or restore the old order (one argument).
> - **`idrak serve` for an encoder-decoder (P10).** `ServeHost.AddModels` registers every model as a chat model; a
>   seq2seq model needs finding through `PretrainedSeq2Seq.Find`, registering as `engine.Seq2SeqModel`, the chat routes
>   (`MapChatApi`, `/api/chat`, the chat UI, `/v1/chat/completions`) skipping it, the announcement and status lines, and a
>   `cli serve:` test. `/v1/completions` and MCP generate already work through `ITextModel` once it is registered.
> - **Slowness found, not fixed (P8, P10):** a cached decoding step rebuilds the bucketed bias index on the host and
>   uploads it; attention with a bias builds the full score matrix over every cache slot; beam search reads all beams x
>   vocabulary logits back every step; a tied checkpoint that also holds `lm_head.weight` is compared value by value on
>   the host at load.
> - **ONNX bug, existing (found in P10):** exporting a `TransformerEncoderLayer` built with a custom activation casts the
>   activation to `Linear` and throws an invalid cast.
> - **P4, to confirm:** `Idrak.Audio.Plugins` lets `Idrak.Tests` see its internals (as every library package does; P1's
>   guard relaxed to the test runner only), keeping VITS's training passes, IdrakAudio's weights and OpenVoice's name map
>   internal; the other choice is making them public. Public names changed: `WhisperSpeechPlugin` → `WhisperPlugin`,
>   `VitsSpeechPlugin` → `VitsPlugin`, `TextRomanizers.UromanArabic` → `VitsPlugin.UromanArabic`,
>   `TextPhonemizers.EspeakNg` → `EspeakNgPlugin.Phonemizer`, the per-family `RegisterIdrakPlugin` methods gone (the
>   package root keeps the one `-P` entry). The CLI registers the audio families at startup, so `-P` is no longer needed.
> - **P4, process starts left in the library:** `src/Idrak.Text/Generation/CodingTools.cs` and `CodingAgent.cs` (the
>   coding agent's shell tool, not text processing) are the only process starts in library packages beyond the CLI and
>   the cluster; for the owner if rule 70 is read widely.
> - **P4, an untested refusal:** `TextPhonemizers.First()` with nothing registered now refuses naming the registry and
>   the package; no test covers it (other plug-ins may have registered a phonemizer in the shared test process).
> - **P4, local folders:** `samples/IdrakAudio`, `samples/VitsSpeech`, `samples/WhisperSpeech/Idrak.WhisperSpeech` and
>   `samples/OpenVoice/Idrak.OpenVoice` hold only untracked `obj`/`bin` output on a machine that built them; the owner
>   may delete them locally.
> - **P5, to confirm:** ArabicOcr's model folder is now `config.json` + `model.safetensors` (positional tensor names
>   `layers.{i}.parameters.{j}`) so `idrak predict` and `idrak train` read it; old `recognizer.ikm` folders still load.
>   The tool registers Qwen-VL, LLaVA, IdrakVision and CRNN at startup (`Idrak.QwenVision.dll`, `Idrak.IdrakVision.dll`
>   gone; LLaVA, test-only before, now ships). A plug-in's `Register()` runs once until `Unregister()`.
> - **P5, ArabicOcr training not reproducible with one seed** (epoch-1 loss 2.78466, then 2.837; likely a CPU reduction
>   order): blocks proving training unchanged (rules 71, 86) and makes the CER thresholds fragile.
> - **P5, the CRNN's backward LSTM reads a batch's right padding** (`ModelLineReader.Batch`, the LSTM layer): readings
>   change with batch makeup; a mask would allow bigger batches and the image family suite's batch check.
> - **P5, two copies of batch measuring:** `MeasuredLineBatches` (plug-in) and the CLI's internal one in
>   `ImageModelSupport`; a candidate for one library contract (decision 9).
> - **P5, slowness:** `CrnnRecognizer.Read` prepares lines one after another (up to 256 pages' lines at once; rule 79);
>   `CrnnFamily.Read` builds the network with random weights, then overwrites them all.
> - **P5, memory:** the public `SafeTensorsWriter.Write` holds every tensor's values at once; the streaming overload is
>   internal.
> - **P5, format question:** the CRNN's tensor names are positional, so checkpoints from other tools cannot map to it.
> - **P5, local folders:** `samples/QwenVision/Idrak.QwenVision` and `samples/IdrakVision/Idrak.IdrakVision` hold only
>   ignored `bin`/`obj` on clones that built them.
> - **P6, to confirm:** a library user needs `Idrak.Text.Plugins` for any Gemma model (and `Idrak.Vision.Plugins` for
>   Gemma 3 with images); `PretrainedFamilies.GemmaTensorName` → `GemmaArchitectures.TensorName`; the `Idrak.Gemma3Vision`
>   assembly → `Idrak.Vision.Plugins.Gemma`; `Gemma3VisionPlugin.Register()` registers once and also the text family when
>   missing; the tool registers Gemma 3 vision at startup; an app that registers over a Gemma name and unregisters gets
>   nothing back (no library default).
> - **P6, no GGUF mapping for Gemma** (`src/Idrak/Models/GgufModel.cs`): `gemma`, `gemma2`, `gemma3` GGUF files were and
>   are refused; a mapping goes through `GgufArchitectures.Register` from the plug-in (llama.cpp's names for the post-norms,
>   whether its norm weights already include the +1, a fixture).
> - **P6, `idrak tune init` picks the vision example only with `-P`** (`src/Idrak.Cli.Commands/Commands/Train/TuneCommand.cs`, near
>   line 340): with Gemma 3 vision built in, `tune init -b <a Gemma 3 vision checkpoint>` without `-P` writes the text one.
> - **P6, message wording:** `PretrainedArchitectures`' refusal says "No architecture 'X' is registered", not "not
>   registered" like the other registries (`src/Idrak/Models/Abstractions/PretrainedArchitectures.cs`).
> - **P6, model names in moved code:** the moved Gemma code keeps the model names already in its comments and test strings
>   (the 4B config test in `VisionLanguageModelTests`, a "0.83 GB less" comment in `Gemma3Vision.cs`).
> - **P6, local folders:** `samples/Gemma3Vision/Idrak.Gemma3Vision/{obj,bin}` and an old `Idrak.Gemma3Vision.dll` in the
>   test output on machines that built them.
> - **P7, to confirm:** `tests/Idrak.Tests/Program.cs` calls `TextPlugins.RegisterAll()` once at startup, as an app does
>   (about 100 tests in 40 files build Llama or Qwen models, and the cluster tests' child processes need them); tests of
>   "not registered" take a family away and put it back. The other way is a register call in each test and in the cluster
>   child entry.
> - **P7, renamed or removed public names:** `PretrainedFamilies.LlamaStyle` → `CommonArchitecture` and `LlamaTensorName`
>   → `CommonTensorName` (old names kept one release, obsolete); removed with no forwarder: `MixtralTensorName` →
>   `MistralArchitectures.MixtralTensorName`, `QwenMoeTensorName` → `QwenArchitectures.MoeTensorName`,
>   `RopeScaling.Llama3` → `LlamaRopeScaling.Create`, `ToolCallFormats.Qwen3Coder`/`Mistral`/`Harmony`/`DeepSeek` → each
>   plug-in's `ToolCallFormat`; `RopeScalingSuite` has no built-in llama3 parameters; LLaVA registers Llama, Mistral and
>   Qwen when missing; a GGUF pre-tokenizer name nobody registered is refused unless a plug-in sets
>   `GgufPreTokenizers.Fallback` (the Llama plug-in sets `llama3`, as before).
> - **P7, fallback chat templates for a GGUF file without one** (`src/Idrak/Models/GgufModel.cs`, `WriteTokenizer`): Llama
>   3's template and ChatML are still chosen in the library by their tokens; moving them needs a template-fallback
>   contract.
> - **P7, `DecoderSpec` still reads Idrak's earlier llama3 spec JSON** (`src/Idrak.Abstraction/Generation/DecoderSpec.cs`):
>   a file format, not family logic.
> - **P7 on P6's GGUF leftover:** `GgufArchitecture.Configure` now gives a plug-in the hook a Gemma GGUF mapping needs.

### done/27-reinforcement.md

> (CPU passes; each merge's run in its as-built note); the owner's runs are collected in "The owner's runs". Left, not
> started (none asked for): on-policy distillation on a cluster (the distiller on R3b's shards), ranks agreeing on a stop
> (`tune` has the same gap), the dashboard steering `reinforce`, a real multi-machine run, `idrak reinforce` for the classic
> environments and DQN; R6's measured candidate (exact-size replay arrays) is done in R8. **Leftovers 1-5 done 2026-10-22 (R7-R11)**; left: leftover 6
> (a real multi-machine run, the owner's: commands in "The owner's runs"), and R11's finding for the owner: a language-model
> run resumed after a stop ends within about 1e-7 of an uninterrupted run, not bit for bit, because `FineTuningSession`
> draws its packing order from a `Random` no checkpoint keeps (a packing draw from (seed, step), or the generator's state
> saved, would make it exact but changes every run's packing).

### opened/28-documents.md

> 1. Decision 5: text and images first, no rendering. Confirm, or add rendering.
> 2. Decision 6: only the contract and the reader, users of it later. Confirm, or name the first user to add.
> 3. Forms and annotations: later (as planned), or with D3?
> 4. The Office plug-in's package name (`Idrak.Data.Plugins` proposed) and the variable names in C8.

### done/idrak-cli.md

> Gaps (Health): `report --bench` times the float32 products on each device (GFLOP/s); model speeds come from `idrak
> bench` once the Measure group lands. `report --tests` runs the test runner of a source checkout as a child process (no
> test list ships with the tool). `doctor` checks the Vulkan driver files on Linux only (Windows keeps
> them in the registry).

> Gaps (in the commands' help too):
>
> - `run --schema`: the library has no JSON-schema constrained sampling (plug-in gap 13). The schema is given to the
>   model as an instruction and the answer is checked afterwards (types, required properties, items, enums), exit 1 when
>   it does not match.
> - `embed`: no embedding heads for decoder models trained as embedders (last-token pooling, instruction prefixes); the
>   command mean-pools the last hidden states (through the library's `TextEncoder`) and scales to unit length.
> - `/stats` memory: the library reports no device memory in use; the process and managed memory are shown.
> - `agent`: `CodingAgent` has no approval hook, so the command runs the coding tools through its own tool loop with
>   `ToolRegistry.RequireApproval` on `run_command` (no task verification, which only suite runs need).

> Gaps (Serve):
>
> - `/v1/embeddings` answers 404 from `idrak serve`: the library has no embedder over a pretrained language model (only
>   `TextEncoder` over a trained package), so nothing can be registered yet; the endpoint works when an `IEmbedder` is
>   given (`CompletionsApiOptions.Embeddings`).
> - The chat API's `/api/generate`, `/api/show`, `/api/embed` and `/api/pull` are not provided by Idrak.AspNetCore.
> - Tool calls go to the client (`ToolExecution.Client`); serving with server-side tools waits for `--tools FILE.dll`
>   loading shared with chat.
> - `ps` reports a model's memory as the device memory it took when it loaded (the engine does not expose a model's size).
> - `mcp serve` speaks MCP over standard input/output only; HTTP would need the MCP ASP.NET Core package.
> - `--timeout` and `--offline` (common options of the second pass): done in the polish; `ping` waits `--timeout` (else
>   10 s) per call, `api` and `server ...` give up when `--timeout` runs out.

> Gaps (Models): a GPU's total memory is not reported by the library (only a configured `GpuMemoryLimit`), so `memory`
> shows "?" for GPUs unless `--memory SIZE` is given; the library has no GGUF writer and loads no language model from an `.ikm` package, so `convert`
> writes Hugging Face folders only (from GGUF or another folder, in bf16, f16 or f32).

> Gaps (library): the optimizer state and learning-rate schedule are not checkpointed, so `resume` continues from the
> weights with a fresh optimizer; image files are read and written through the library's `ImageCodecs` and `ImageEncoders` (the tool's own decoder is gone);
> writing Parquet is not in the library (`data convert` reads it, writes JSON Lines, JSON, CSV or TSV); `distill` has no
> vocabulary mapping (teacher and student must share a tokenizer for logits; `--generate` works across them); `tune init` writes the
> library's defaults until `idrak suggest` can size them.

> - Gaps: the library has no embedding-model loader (a bi-encoder from a Hugging Face embedding model), so `rag index
>   -m` embeds with a chat model's averaged hidden states, a rough complement to BM25; `RetrievalIndex.Save` does not
>   keep the vectors of a non-`TextEncoder` embedder (the tool stores them itself); eval metrics beyond
>   number/exact/contains/F1 wait for plug-in gap 16. `--format csv|md` and `--output` (second pass, now applied) apply once the
>   Health agent's common options land.

> Gaps (Design): images are decoded by the library (`ImageCodecs`, `ImageFolderSource`: PNG, BMP, PGM/PPM, and any
> registered codec); the tool still reads JPEG headers itself, so JPEG folders are profiled, but trained by `--search` and
> `train` only when a JPEG codec is registered. The "augment" list of prep.json maps onto the library's transforms
> (`RandomFlip`, `RandomRotation`, `RandomShift`) but is not applied yet, to keep the search and training results as they
> were; applying it is a later step. The library does not
> report a GPU's memory size (only a configured `ComputeResources.GpuMemoryLimit`), so memory checks on a GPU say the
> size is unknown unless a limit is set.

> - `shell` completes command names and options; completing model names and file paths waits for the shared
>   completion helper (`idrak completion`).

> Later, not in this build: a plug-in marketplace listing; remote devices (run a command on another machine's `idrak
> serve`). Messages in Arabic were built after the render check (branch `feature-arabic`; "Arabic messages" below).

> Remains: the bodies of commands' help (options, examples) and the other commands' messages are English; an answer a
> model streams in Arabic is rendered line by line (a line appears when it ends) and the user's own typing is echoed
> by the terminal, unreordered, in terminals without bidirectional support; tables keep their columns left to right.

### open-questions.md

> ## Decided (2026-10-03)
> - Merging into `main`: not yet.
> - Public backend API (12c): stays in the plan, not started.
> - LinkedIn series: dropped for now.
> - Native Linux GPU drivers: the maintainer tests them from a live USB when one is at hand.
> - `idrak serve` gets a default port of its own (not 11434, which another local server uses); `-p/--port`, the config
>   and an environment variable choose another. Done: 7317, then `-p`, `IDRAK_PORT`, `serve.port` (plans/idrak-cli.md).
> - Arabic messages: only if the console can show them correctly (right to left, joined letters); otherwise dropped.
> - Branches: idrak-cli was fast-forwarded into `architecture` and retired; on 2026-10-04 `architecture` was merged into
>   `main` (the maintainer's go-ahead), with the version set to 0.2.0 for the next release.
> - Round 3, all built and merged (now on architecture): mixture of experts, fast sliding-window and soft-cap kernels
>   (CPU and Vulkan run; CUDA written, hardware check 9), dataset loaders, the teacher pattern (`idrak distill`),
>   provider-neutral serving names, Arabic output. Arabic: the tool shapes and reorders it itself for terminals that
>   do not (plans/idrak-cli.md, "Arabic messages"), checked against the Unicode conformance suite and an independent
>   implementation; `--lang ar` built. The look in real Windows terminals is hardware check 11 below.
>
>
> Decisions and hardware checks that wait for the maintainer, collected while work continued. Each item says what is
> needed, the options, and the recommended one. Nothing here blocks the work in progress.
>
> ## Decisions
>
> | # | Question | Options | Recommended |
> |---|---|---|---|
> | 1 | When to merge `architecture` (which now holds the CLI) into `main` | a) after the CLI lands and real-GPU checks pass · b) now · c) per feature | a |
> | 2 | Public backend API (item 12c) | a) start now, using the HIP backend's list of awkward contract points (plans/8-hip.md) · b) keep parked until HIP runs on real AMD hardware | b |
> | 3 | Provider-neutral names for the serving API (plans/plug-in.md, "Noted for later") | a) rename with obsolete forwarders in the next release · b) keep the old names longer | a, done |
> | 5 | README tagline | a) "Deep learning in pure .NET. Every GPU. Zero dependencies." · b) another | a |
> | 6 | LinkedIn series | a) post 2 next (GPU code without the CUDA toolkit) · b) post 4 next (an LLM on a phone) ; English only or with Arabic | a, English only |
> | 7 | Native Linux GPU drivers (NVIDIA's Linux driver, RADV, ANV, NVK) | a) live USB on a laptop · b) a cloud GPU instance (a T4 also covers compute 7.5) · c) keep "WSL2 only" | c for now; a when convenient |
> | 8 | Mixture of experts (Mixtral, Qwen-MoE): built (branch feature-moe, plans/plug-in.md); grouped expert kernels remain | a) next after the CLI · b) later | done |
> | 9 | `idrak serve`'s default port: 11434, the one common local-model clients connect to by default (so they work without settings), or a port of Idrak's own | a) keep 11434 for drop-in use · b) an Idrak port, with `-p 11434` documented for those clients | a |
>
> ## Hardware checks to run
>
> | # | Machine | What | Command |
> |---|---|---|---|
> | 2 | Any CUDA machine | The recorded fine-tuning step with a custom optimizer (could not run without CUDA) | `$env:IDRAK_FILTER="fine-tuning options"; dotnet run -c Release --project tests/Idrak.Tests` |
> | 3 | RTX 5070 Ti, laptops | The full lists after the plug-in work (315+ tests), CUDA and Vulkan | the per-device blocks in installation/windows.md |
> | 7 | Any machine | The `idrak` tool once the CLI lands: `idrak doctor`, `idrak devices`, `idrak c MODEL`, `idrak s MODEL`, `idrak b MODEL` | plans/idrak-cli.md, "Examples" |
> | 9 | RTX 5070 Ti, a laptop with CUDA | The windowed and soft-capped CUDA kernels (decoding, `attention_flash_*`, the float32 gradients; never run): correctness, then the windowed decoder's speed against the composed path | `$env:IDRAK_DEVICES="cuda:0"; $env:IDRAK_FILTER="window kernels"; dotnet run -c Release --project tests/Idrak.Tests`, then `$env:IDRAK_FILTER="sliding windows"` and `"decoding"`; a failure there: `$env:IDRAK_CUDA_DEBUG="1"` names the kernel, `$env:IDRAK_WINDOW_KERNELS="0"` confirms the composed path still passes |
> | 11 | Windows (Windows Terminal, a PowerShell or cmd window), Git Bash, a Linux terminal | Arabic messages read right to left with joined letters; the device table aligned; `logical` reversed where `auto` is right (or the other way round in terminals that reorder text themselves) | `idrak help --lang ar`, `idrak doctor --lang ar`, `idrak devices --lang ar`, then the same with `--lang-render logical` (plans/idrak-cli.md, "Arabic messages") |

### done/plug-in.md

> - Formats not built in (no verified template at hand, or a different shape): Hermes-style tags with other JSON
>   shapes (GLM 4.5 `<arg_key>`/`<arg_value>`, Kimi K2 `<|tool_call_begin|>functions.name:0`, MiniMax `<invoke>`,
>   Command R, Granite, FunctionGemma). Each can be registered with `ToolCallFormats.Register`.
> - The pythonic community template writes string values with Python's `str()` (unquoted), so calls with string
>   arguments do not round-trip through that template exactly; numbers, booleans and lists do.
> - Calls are emitted when complete (no partial-argument streaming), as before.
> - Llama 3.1's built-in tools (`<|python_tag|>name.call(...)`) are parsed, but the template renders them back only
>   when `builtin_tools` is passed as a template variable.
