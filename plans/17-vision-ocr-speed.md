# Plan 17: speed and memory for vision and OCR (the library first; the legal OCR app as the measuring ground)

**Status:** planned 2026-10-11 against `abstraction` a61c979, from a read-only audit of the read path, the image pipeline,
the vision encoder, the tuning path and every open item of plans 11 to 16 and `docs/performance-notes.md`. Nothing here
is built. The owner's demo of `samples/Idrak.Samples.LegalOcr` is the measuring ground: a scanned Arabic legal page in,
Gemma 3 4B (bakrianoo's fine-tune) reading it, JSON out, and the Fine-tune tab. Every change below is in the library
(`Idrak.Abstraction`, `Idrak`, `Idrak.Data`, `Idrak.Nlp`, `Idrak.Gpu`) or in the family's plug-in; the app only stops
doing work the library now does once.

Numbers quoted "measured" come from the plans and notes (with their line); numbers marked "est." are estimates from
tensor shapes and need measuring first (section 3 says how). Baseline on the owner's card, real model, bfloat16 decoder,
float32 encoder (`plans/11-vision-language.md:412,466`): prompt with the image 1.0 s, 80 tokens/s, a 1,032-token answer
in 14 s; one 896 x 896 image through the encoder 0.43 s in float32, 0.11 s with bfloat16 tensor cores (`:769-770,906`).

## Progress (updated as lanes land)

| Item | State | Proof |
|---|---|---|
| **A3** image attention composed on the device where the device has no span kernel for the head size | **Done** (2026-10-11): `BackendCapabilities.SpanAttentionHeadDim` (CUDA 128, Vulkan 256, CPU every size; default every size, so outside backends are unchanged); `Tensor.AttentionFastest` composes grouped heads and several tables (a table at a time) and, where the device has no kernel, whenever the scores fit, without measuring; the decoder's image path passes its ranges as a mask (`ImageBlocks.Mask`, built once per window and pass from the cached host tables). Gemma 3 4B's head size 256 on CUDA no longer copies q, k, v to the host per layer for a read's image prefill. HIP keeps the CPU's limits (its own rule) | CPU and the minimal device (no span kernel of its own): `attention spans` (new: grouped heads and two tables, forward and gradients, composed forced and unforced), `image prefill` (transformers' logits and 20 tokens through the composed path on the minimal device), `image reader`, `chat parts`, `cli images`, `minimum backend`; CUDA: the owner's commands below |
| **A1, A2** span kernels for head size 256 on CUDA | Not started: needs a GPU to verify (at 256 the float32 tiled kernel needs about 105 KB of shared memory and the tensor-core kernel declares its tiles statically, so lifting the cap alone would be refused on most cards); A3 covers reads until then | — |
| **A4** span tables uploaded once per pass | **Done** (library, generic): a cached prefill with images and without autograd (every read and the CER scorer) keeps each window's key ranges and the composed path's mask on the device for the next layer, one upload per table and pass instead of two (three with the mask) per attention layer, and frees them when the pass ends; with autograd (training) each layer gets tensors of its scope as before, so nothing recorded outlives them | `image prefill` (new: the next layer reads the same copies, another offset another table, freed with the pass; transformers' logits and 20 tokens unchanged on the CPU and the minimal device), `image reader`, `attention spans` |
| **A5** host fallbacks visible | **Done** (library, generic): a `HostFallback` event (device, operation) at `TelemetryLevel.Devices`, the first time each operation runs through the host fallback on a device while someone listens; `ConsoleLogger` prints it by default ("Device cuda AttentionSpans runs on the host: …"), JSON Lines writes `host_fallback`, `ChannelTelemetry` forwards it; `Kernels.HostCalls` and `KernelTrace` count every call as before. A family whose head size a device has no kernel for now shows in the log instead of only in the time | `minimum backend` (new: three fallbacks, one event, the console line, the JSON line, three counted), `operations: a failing`, `public API` |
| **B1** a page decoded and transformed once per read | **Done**: `ChatImages.Read` keeps the last prepared image (content hash + transforms) until the same read takes it (at most one image held); `ImageReader.Prepared` goes through it; `ImageCodecs.ReadInfo(bytes)` and `ChatImageDecoder.Size(bytes)` (EXIF-upright) give the original size from the header; the app decodes a page once (was three decodes and two transform runs) | `image reader`, `chat parts`, `cli images`, the app's `legal` tests |
| **B2** the device pool kept between reads | **Done** in the app: freed blocks go back to the driver only after a failed or stopped read, on unload and when tuning takes the device; the status shows `Cached` | `legal` tests |
| **B4** stop at the end-of-turn id at once | **Dropped**: reading each sampled token as it is made is a host sync per step, which costs more over a long answer than the up to 7 steps after the end it saves | — |
| **B5** warm-up at load | **Done** in the app: one short read of a blank page with the model's preprocessing and prompt after loading (encoder built, kernels loaded, shapes measured); "loaded in X s, warmed up in Y s" | `legal start` |
| **H1** `IDRAK_FILTER` with `a\|b\|c` | **Done** in `tests/Idrak.Tests`, `Idrak.Samples.LegalOcr.Tests`, `Idrak.Samples.ArabicOcr.Tests` | the runs above used it |
| **C9** `--bench-image` | **Done**: a seeded 1,654 x 2,339 colour scan through every CPU stage of a read, cold and warm, bytes allocated and output SHA-256 (taken after each timed call: the first version timed the hash too, which inflated every figure). Baseline, best of two on a quiet 4-thread container with the build servers stopped: JPEG decode 73 ms (72 MB), PNG decode 32 ms, card pipeline 58 ms, preprocess to 896 13 ms (grey 1024) / 41 ms (colour page), preview PNG 43 ms; colour page: contrast 125, brightness 106, sharpness 194, autocontrast 50, max_width 66; grey 1024: median/min/max 66-75, blur 39, jpeg=95 102 | the hashes are the lane's proof (rule 71) |
| **C4** (part) table-driven blends, parallel point maps | **Done**: `Blend` (contrast, brightness, sharpness) looks each byte up in a table of Pillow's own float expression (256 entries over a flat image, 65,536 over the smoothed one) in parallel chunks; contrast's and autocontrast's grey from the byte planes already made; `Smooth`, the point maps (equalize, gamma, binarize, autocontrast) and `FromBytes` parallel. Before → after (ms, best of two, quiet machine, timing the stage only): colour page contrast 125 → 33, brightness 106 → 27, sharpness 194 → 69, autocontrast 50 → 35, grayscale 20 → 15; the card pipeline of every read 58 → 47. What is left in the colour steps is the float → byte → float round trip (C1). Left of C4: rank filters (median, min, max), the blur's column pass, `Orient`, the JPEG round trip's colour conversion | every `--bench-image` hash unchanged; `image transforms` (238 Pillow SHA-256 cases), `image preprocessing`, `jpeg`, `cli images`, `image reader` |
| **C4** (part 2) rank filters, the blur's columns | **Done** (library, exact): a 3 x 3 rank filter (median, min, max, any rank) sorts each window with Knuth's 25 compare-exchanges (checked on all 512 0-1 inputs), a vector of pixels at a time, edges and tail one at a time, rows in parallel (larger windows keep the histogram); the box blur's vertical pass runs over a tiled transpose of the plane (columns read in order, rule 72) and unsharp's per-pixel step in parallel. Grey 1024 x 1448, warm, best runs of three on the 4-thread container: median=3 94 → 13-15 ms, min_filter=3 84 → 10-14, max_filter=3 78 → 8-11, blur=0.5 24 → 18-20, unsharp 33 → 20-24; every hash unchanged. Measured for next: `jpeg=95` is 94-108 ms, of which the optimized encode 70-88 and the decode 16-17 | `image transforms` (Pillow's SHA-256 for median 3 and 5, min 3, max 5, blur, unsharp on 8 inputs), `data loaders`, `image famil`, `cli images`, `aspnetcore: images`, `cli serve: images` |
| **C8** the PNG encoder (the preview of every read) | **Done**: rows written in parallel straight from the pixels, deflate at the fastest level, one buffer instead of two copies, a slicing-by-8 CRC; rows stay unfiltered (measured: on a scan-like page "Up" makes paper grain larger, 19 ms and 1,138 KiB against 13 ms and 910 KiB unfiltered). Preview encode 43 → 14 ms (best of two, the stage alone); files about 1.6x larger (a localhost preview); our PNGs decode as fast as before (72 ms). Measuring note: shut the build servers down before `--bench-image` (`dotnet build-server shutdown`): a resident compiler server doubled the timings | `cli vision: ImageEncoders` (new: multi-row grey and colour round trips, CRC-32's check value), `cli images`, `image reader`, `image famil`, the app's `legal` tests |
| **C5** the PNG decoder | **Done** (library): the rows unfiltered in order, then turned into floats in parallel (each row and each Adam7 pass writes pixels of its own) through a table of each depth's `k / max` (the same division, made once), the Up filter as a vector add (modulo 256, as the filter), Paeth without branches (the same predictor; scans' noise made its choice unpredictable) and Avg and Paeth with the first pixel apart, the pixel array uninitialised (every pixel written). Grey page 1,654 x 2,339, best of two on a quiet 4-thread container, warm: Paeth rows 82-97 → 41-44 ms, Up rows 36 → 20-22 ms, unfiltered 31-35 → 28-30 ms (inflate-bound); the same pixel hash for all three files before and after. Byte planes (C1) not part of it | `data loaders` (new: 101-wide grey and RGB rows, interlaced and not, each filter: the vector part and its tail), `image transforms`, `image famil`, `cli images`, `image reader`, `chat parts`, `pan and scan`; `--bench-image` gains Up and Paeth files |
| **E2** the cache sized to the reply, not the window | **Done** (library, generic): a generation that keeps no cache for a next turn allocates the prompt plus its most tokens (capped by the window), as the batched path already did; only a reply that can outgrow the window re-reads its last half, at the window's size. Gemma 3 4B in float32 (272 KB a token): a read of about 287 prompt tokens and 2,048 answer tokens holds 0.63 GB instead of 2.2 GB for the 8,192 window (0.92 GB with four pan-and-scan crops), allocated and cleared on every read. On CUDA the decoding splits are measured per capacity, so the summation order may differ from before: the owner checks the same pages' answers | `generation:`, `decoder:`, `pretrained:` (transformers' tokens), `cli run:`, `chat:`, `engine:`, `image reader`, `image prefill`, `aspnetcore`; the app's `legal` tests |
| **D1** the encoder's precision as an option | **Done** (library, generic, off by default): `ChatImages.EncoderPrecision` and `ImageReaderOptions.EncoderPrecision` run the vision encoder alone under `MixedPrecision.Use` (bfloat16 tensor cores where the device has them; the language model unaffected); the app's `EncoderPrecision` setting ("f32" by default, "bf16"). Measured on the owner's card for one 896 x 896 image: 0.43 s in float32, 0.11 s in bfloat16 (plan 11). Features change within bfloat16's rounding: the owner compares CER on the same evaluation pages before turning it on | `image reader` (new: a probe encoder sees the scope, the scope ends after it, the CPU's answer unchanged), `chat parts`, `cli images`, `aspnetcore: images`, `public API`, the app's `legal` tests |
| **D6** the attention choice before the memory query | **Not worth it**: the memory query must precede the measurement it guards, and it is a few microseconds per layer | — |
| **C2** the JPEG entropy decode with its reader in locals | **Tried, not merged**: byte-exact, 11% faster alone (64 → 57 ms) but 9% slower inside the benchmark (73 → 80): no proven gain. Next attempt: a SIMD islow IDCT and fused upsample + colour + output (about half the decode), measured the same two ways | `jpeg` passed (24 tests) |
| **C3** the resize's horizontal pass as vector dot products | **Tried, not merged**: byte-exact; the card pipeline 47 → 39-44 ms but the colour page's preprocess 43 → 51 ms; mixed, so reverted. Next attempt: widen rows once per plane only when the taps are many, or vectorize across output pixels | `image preprocessing`, `vision kernels` passed |
| **F2** (part) the CER scorer's repetition stop | **Done** (library, opt-in): `TuningAnswerScorer.RepeatStop`, `ConversationTuningOptions.MetricRepeatStop`; the app passes its Read tab's stop, so tuning's CER and the Read tab judge a looping answer alike (was: up to `AnswerTokens` 1,536 per looping page). The scorer's cache is sized by E2 (prompt + answer: about 0.5 GB instead of 0.98 GB) | `conversation tuning` (new: the stop reaches the scorer), `vision tuning`, the app's `legal` tests |
| **F4** the evaluation loss over the first N pages | **Done** (library, opt-in): `ConversationTuningOptions.EvaluationSamples`; the app's `Tuning:LossPages` (0: all, as before) | `conversation tuning` (new: one sequence for the loss of two held out, the scorer's two apart) |
| **F1** (part) tuning reads each page once in parallel | **Done in part**: `ConversationTuning` works out every distinct image's blocks before tokenizing, decoding and transforming the pages on worker threads (the family's `Blocks` called one at a time: the contract does not promise it is thread-safe), with a "reading … images" stage and a log line; it was one page after another inside tokenizing. Estimate: a page's decode and the card's transforms take about 0.1 s on this container, so 300 pages go from about 30 s to about 8 s on 4 threads (the owner's run gives the number). Left: the second decode when a page's features are first computed in training (the batched frozen pass during preparation) | `conversation tuning`, `vision tuning`, `cli vision tuning`, `fine-tuning`, the app's `legal` tests |
| **H3** `--bench-read`: a read profiled | **Done** (tests): the app's flow (prepare for the preview, then the read) on each device of `IDRAK_DEVICES`, the tiny Gemma 3 and a seeded scan by default, the owner's model and pages through `IDRAK_BENCH_MODEL`, `IDRAK_BENCH_PAGE`, `IDRAK_BENCH_WEIGHTS`, `_CACHE`, `_ENCODER`, `_TRANSFORMS`, `_TOKENS`, `_READS`, `_CONTEXT`. Per read, cold apart from warm: prepare, encode, prefill, decode, total and first text; prompt and answer tokens a second, ms a token; device memory in use, cached and peak; managed bytes and collections; CPU cores busy. Then estimated GFLOP/s (2 x parameters a token) and weights GB/s while decoding, one traced read (operations by count, per token, every host fallback by operation) and, where the device times kernels (CUDA), one profiled read (`GpuProfiler`: each kernel's calls, ms, share, TFLOPS). `GpuProfiler.Format` now formats with the invariant culture, straight into its builder. On the 4-thread container the CPU-side prepare (JPEG decode + the card's pipeline of a 1,654 x 2,339 page) is the largest part of a tiny-model read (about 210-280 ms warm), and the decode's time there varies 53-178 ms run to run, so it is judged on the owner's machine | `cli measure: profile`, `public API`; the bench runs |
| **H3b** the profiler contract | **Done** (library, `Idrak.Abstraction.Diagnostics`): `Profiler.Run(name, device, workload, ProfileOptions)` runs any workload cold once and warm N times (the device's queued work finished around each), the workload marking stages (`ProfileRun.Stage`) and recording its own values (`Record`, `Add` a table); `IProfileProbe` in the `ProfileProbes` registry measures around each run (library: `cpu`, `heap`, `memory`, and in runs of their own because they slow a run, `operations` and `kernels`, the latter where the device times kernels); `IProfileWriter` in `ProfileWriters` writes the `ProfileReport` (library: `table`, `markdown`, `csv`, `json` as JSON Lines), numbers in the invariant culture; `ProfileOptions.FromEnvironment` takes the criteria from `IDRAK_PROFILE_COLD`, `_WARM`, `_PROBES`, `_STATISTIC`, `_FORMAT`, `_OUT`. An app registers its own probes and writers or shadows the library's (unguarded registries: a probe holds state through a run, a writer writes as it goes). `--bench-read` is now a workload on it (the page alone before any model, then the read) | `profiler:` (2 new: runs, statistic, probes, intrusive passes, host fallbacks on the minimal device, kernels skipped where untimed, unknown probe refused; the four writers, an app's probe and writer, the environment's criteria), `public API`, `abstraction inventory` (regenerated) |
| Stale checks | Fixed: three tests used `blur` as an unknown transform, registered since d2ad9fc; they use `emboss` now | `aspnetcore: images`, `cli serve: images`, `vision tuning contracts` |
| Everything else in sections 2 and 6 | Not started | — |

**The owner's CUDA run (2026-10-11, RTX 5070 Ti, compute 12.0, driver CUDA 13.3), after wave 1 to F1:** `attention spans|image prefill|image reader|generation:|pretrained:|conversation tuning`, 35 of 35 pass: transformers' logits within 6.2e-6 and its 20 greedy tokens through the image prefill (float32 cache; bfloat16 0.0358 and int8 0.255 from float32, as before), the grouped-heads composed test, the encoder-precision probe, the cache sized to the reply, the scorer's stop and the loss cap. Not covered by these tests: the composed path at head size 256 (the tiny models' heads are small); the real model's read in the app covers it.

The cloud environment builds now: the .NET 10 SDK came from conda-forge (`dotnet-sdk` 10.0.401, `dotnet-runtime` and
`dotnet-aspnetcore` 10.0.12, unpacked into `~/.dotnet`), since the SDK's own hosts are refused by the network policy.

## 0. Rules every lane keeps

- **Abstraction and plug-ability, at no cost in speed.** The library provides the contract and, where one exists, a
  generic implementation; a family (Gemma 3, LLaVA, Qwen-VL) stays in its plug-in. No library code names a family, a
  model, a card or a memory size. A new capability is generic (a kernel for a head size, a cache layout for windowed
  layers, a prepared-image input) and selected by what the device and the `DecoderSpec` report, never by the family.
  Where an item below touches `samples/Gemma3Vision`, it is the plug-in calling a generic library path, not the library
  learning about Gemma.
- **The app side registers, the library never changes for it.** An app or plug-in brings its own family, model,
  operation, transform, cache layout or override of a default general implementation through the library's registries
  (`SlotTable`: `VisionFamilies`, `ImageTransforms`, `KeyValueLayouts`, `ImageAttentionRules`, `PretrainedArchitectures`,
  `DeviceProviders`, ...), with the library default as its fallback and the shadow and conformance kit to check it
  (plan 10's override loop). A lane that needs a new behaviour adds the **contract and its generic default** to the
  library; the family-specific or app-specific part is registered from `samples/Gemma3Vision` or the app. Nothing in a
  lane may make a registry lookup, a guarded wrapper or an override slower on the hot path than the direct call it
  replaces (one static read and a bitwise AND, as the telemetry sites).
- **`docs/optimization.md` rules 70-79 and 80-87**, and CLAUDE.md: speed and memory first; card-agnostic; the API dump
  (`api/*.txt`, `IDRAK_UPDATE_API=1`) and the inventory test current; no CHANGELOG or version changes.
- **The same result, proven (rule 71).** Every lane names the fixtures that pin its output. A change that alters bits
  (bfloat16 encoder products, bfloat16 KV cache, int8 weights, a different summation order) is **opt-in** and gated by
  the Evaluate tab's CER on the same pages before and after, never a new default on its own.
- **Cold and warm apart (rule 76).** Every number is reported twice: the first call and the steady state.
- **CUDA first** (plan 14's scope). Vulkan and HIP get the same contract and the host fallback until measured.

## 1. What the audit found (the state today)

| # | Finding | Where | Cost today |
|---|---|---|---|
| F1 | **Image attention runs on the CPU on CUDA for head size 256.** `CudaBackend.AttentionSpansKernel` and its backward hand any head size above `PtxKernels.FlashMaxDim` (128) to the base `Backend`, whose implementation is a `HostCall` to `CpuBackend` (copies q, k, v, y to the host and back, a device sync each time). Gemma 3 4B's text decoder has head size 256 (4 KV heads, `plans/11:771`). So every **image prefill of a read** (34 layers, `Decoder.AttendImages` → `AttendSpans` → `Tensor.AttentionSpans`), every **training step with images** (forward, backward, and the recompute under checkpointing) and the **CER scorer's prefill** cross PCIe and run in C#. The text prefill at 256 takes the composed scores path on the device (`Decoder.cs:343-351`); decoding uses `attention_decode` (limit 256). This is consistent with the measured 1.0 s prompt for 287 tokens. Vulkan's kernels go to 256 and are not affected. | `src/Idrak.Gpu/Cuda/CudaBackend.Spans.cs:40-44,98-103`; `PtxKernels.Rows.cs:617`; `src/Idrak.Abstraction/Devices/Backend.cs:1873-1896`; `src/Idrak/Layers/Decoder.cs:428-470` | read: est. most of the 1.0 s prompt after the encoder; tuning: est. 5-20x per step (about 3 GB forward and 8 GB backward over PCIe per padded step, no overlap), and no graph capture (a HostCall during capture fails) |
| F2 | **A read decodes the page three times and transforms it twice.** The app decodes for the original size, `ImageReader.Prepared` decodes and transforms for the preview, `ImageReader.Read` → `ChatImages.Read` decodes and transforms again. | `samples/Idrak.Samples.LegalOcr/ReaderService.cs:546-563`; `src/Idrak.Nlp/Generation/ImageReader.cs:164-165`; `ChatGenerator.cs:44-49,254` | JPEG 2,000 x 3,000: 150 ms a decode; the card's pipeline about 90 ms (`plans/11:70,851,897-899`); about 300 MB of float garbage per read (est.) |
| F3 | **The device pool is emptied after every read** (`ComputeResources.ReleaseCachedMemory` after each page, also inside Evaluate), so the next read re-allocates and re-zeroes the KV cache and the encoder's activations through the driver. | `ReaderService.cs:565-570,299`; `CudaBackend.cs:646-675,559-594`; `Decoding.cs:72-73` | est. tens of ms and a driver sync per read; defeats rule 77 |
| F4 | **The KV cache is sized to the context window in every layer**, float32 by default: 272 KB a token, 2.2 GB at 8,192 (`plans/11:771`); 29 of the 34 layers are 1,024-token sliding-window layers whose cache never needs more than the window. | `Decoding.cs:378-386`; `Architectures.cs:193-195`; `appsettings.json:9-10` | 2.2 GB where about 0.56 GB would do (est.), allocated and zeroed per read (F3) |
| F5 | **Pixels are floats everywhere.** `ImageData` holds `float[]` [C,H,W]; every codec writes floats, every Pillow-exact transform converts float → byte → float, the preprocessor converts again, then uploads float32 [3,896,896] (9.6 MB a view) after resizing on the host though a bit-exact device resize exists. | `src/Idrak.Abstraction/Data/ImageData.cs:12-33`; `PillowImageOps.cs:184-196`; `ImagePreprocessor.cs:353-396,538-610,257-320`; `Gemma3ImageEncoder.cs:147-171` | about 9 passes and 90 MB for the card's pipeline (90 ms measured); resize to 896: 77-127 ms against Pillow's 40-45 |
| F6 | **The JPEG decoder is 3.7x slower than Pillow** on a baseline file: byte-at-a-time Huffman fill through fields, scalar islow IDCT, separate upsample, colour and float passes, about 6 passes and 76 MB of large-object allocations per page. | `src/Idrak/Data/JpegCodec.cs:155-158,342,445-583,673-718,805-905,909-996,1047-1155` | 150 ms against 41 (`plans/11:70`) |
| F7 | **The encoder runs float32 products and about 19 launches a layer** (packed bf16 weights through fp32 tiles, bias as a second kernel, 3 narrow + 4 permute copies around attention, GELU and residual as passes); `AttentionFastest` queries the driver's free memory on all 27 layers before looking up the known choice; nothing is graph-captured. | `MixedPrecision.cs:49-55`; `Attention.cs:83-111,307-330`; `Linear.cs:158-178`; `CudaBackend.Quantized.cs:390,398-405`; `Tensor.AttentionPaths.cs:119-135` | 0.43 s a view against 0.11 s with tensor cores (measured); about 780 MB of element-wise traffic a layer (est.) |
| F8 | **Tuning decodes each page twice, on one thread, with the GPU idle**, scores CER one page at a time (8 pages x up to 1,536 tokens, before training and at every evaluation, with no repetition stop, a float32 cache and a new graph per page), evaluates the loss over every evaluation page twice, reads and hashes every image of the data file before `MaxPages` cuts the list, and never packs or records steps with images on CUDA. | `TuningVision.cs:180-185,342`; `ConversationTuning.cs:308-312,375-384`; `TuningAnswers.cs:144-166`; `FineTuning.cs:1046-1051,1280,1815-1827`; `TuningDataReaders.cs:418-422,567-571` | decoding est. 3 min of single-threaded CPU for 300 pages; a CER scoring est. 100-300 s; see lane F |
| F9 | **Loading widens bf16 weights to float32 on the host and rounds them back**, with the embedding as one 2.7 GB float transient; the vision encoder is built on the first read (the checkpoint reopened), not at load. | `SafeTensors.cs:107-231`; `StoredWeights.cs:54-56`; `DecoderBuilder.cs:105-123`; `Gemma3Vision.cs:284-305`; `ModelImages.cs:98-103` | load time and host peak; the first page pays the encoder build, PTX load and tuning measurements |
| F10 | **Telemetry gaps (rule 78).** `TextGenerator` emits no events for encode, prefill, decode, graph capture or allocation; the app times with stopwatches; there is no image-pipeline benchmark and `--bench-vision` runs dense float32 weights only. | `TextGenerator.cs`; `BenchVision.cs:28-30` | nothing to measure a read's stages with |

## 2. The lanes (independent; one agent each; at most two building at a time, CLAUDE.md)

Each item: **id**, where, the change (contract and generic implementation), the gain, exactness and the fixtures that
prove it, and its size (S under a day, M a day or two, L several days). Lane A is the largest win and is a prerequisite
of F's biggest items; B and C are the demo's CPU-side wins; everything else runs beside them.

### Lane A: attention over key ranges at head sizes up to 256 on CUDA (F1)

| Id | Change | Gain | Proof | Size |
|---|---|---|---|---|
| A1 | `attention_spans_f32` and its tensor-core variant for head sizes up to 256: the head split across two 128-wide accumulators (or across warps), shared memory and the tile shape from what the device reports (`CudaDeviceLimits`, rule 84), rules 80, 83, 86, 87; `BackendCapabilities` gains a `SpanAttentionHeadDim` the decoder and `AttentionFastest` read (today they read none for spans). Generic: keyed by head size, not by family | the read's image prefill leaves the CPU; est. the 1.0 s prompt falls to the encoder's time plus tens of ms | the conformance kit's span cases (head sizes 1-256 against plain loops, `DeviceCases.SpansBackward.cs`) run on CUDA through the new kernels instead of the host; `IDRAK_FILTER="attention spans"`, `"image prefill"`, `"conformance kit"`; `Kernels.HostCalls` shows 0 `AttentionSpans` per pass | L |
| A2 | `attn_spans_bwd_q` and `attn_spans_bwd_kv` for head sizes up to 256 (the same split), so training and the recompute under checkpointing stay on the device and steps can be recorded | tuning est. 5-20x a step; graph capture becomes possible (F8) | `"attention spans backward"`, `"vision tuning"` (losses within 1e-4 and adapters within 1e-5 of `tests/Idrak.Tests/data/vlm-tuning`); checkpointed == stored bit for bit | L |
| A3 | Until A1 lands: `Decoder.AttendSpans` may take the composed path for grouped heads when `ComposedFits` and the measured `PrefersComposedAttention` say so (repeat the KV heads, the mask from the spans), instead of a host fallback | a device path at once, memory-heavy (the [t, t] scores) | the same filters; the choice is measured, never assumed | S |
| A4 | Span tables uploaded once per window per pass, not twice per layer (`ImageBlocks.Spans` does `Tensor.From` twice per call; 68 small uploads a pass, again in the recompute) | small; fewer launches | `"image prefill"` tokens unchanged | S |
| A5 | `HostCall` visible: a trace line (and `idrak profile`'s summary) naming every operation that fell back to the host in a pass, so a 256-head family on any backend fails loud or shows the cost (rule 78) | finds the next F1 before a demo does | `"kernels"` tests | S |

### Lane B: a read does each thing once (F2, F3, F10)

| Id | Change | Gain | Proof | Size |
|---|---|---|---|---|
| B1 | **A prepared image as a contract input.** `ImageReadRequest` (and `ChatImage` parts) may carry decoded pixels (an `ImageData` after the transforms) beside the encoded bytes; `ChatImages.Read` uses them and skips decode and transforms; `ImageReader.Read` returns the prepared image it used (so `Prepared` is not a second pass). The size of the original comes from `ImageCodecs.ReadInfo` on the header plus `ChatImageDecoder.Orientation` (5-8 swap W and H). The app then decodes once | est. 300 ms and 200 MB per request on the measured container (`plans/11:897-899` names 90 of it) | the preview's pixel hash and the read's tokens equal today's; `"image reader"`, `"chat parts"`, `"cli images"`, the app's `legal reader` tests | M |
| B2 | **The pool stays between reads.** The app's per-read `ReleaseCachedMemory` goes; the library trims on unload, when tuning takes the device, and (new, generic) after an idle time or when `MemoryUsage.Cached` passes a share of the device's reported memory; `MemoryInfo` shows `Cached` so the pool is visible during the demo (rule 77) | no re-allocation and re-zeroing of the cache and the encoder's activations per read (est. tens of ms and a sync) | `MemoryUsage.Cached` flat across 5 reads of different pages; `"pool"`, `"memory"` tests | S |
| B3 | **The decoding context kept, not the cache's content.** A generic `ImageReaderOptions`/`TextGenerator` option keeps the `DecodingContext` allocation and the sampler across reads and always `Reset()`s (not `KeepCache`, which reuses the prefix before the first image and splits the prefill, changing rounding); later the recorded decode graph is replayed across reads | allocation, memset and graph capture off every read (est. several to tens of ms) | tokens bit for bit equal to a fresh reader's (the test from commit 23b58a8's pattern) | M |
| B4 | **Stop on the end-of-turn id as soon as it is read**, not at the next 8-token flush (`ChunkSize` 8): up to 7 decode steps run after the answer ends, and `GeneratedTokens` is overcounted by the same | est. 90 ms a read; honest tokens/s | the text is already cut at the stop, so output unchanged; `"decoding"`, `"image reader"` | S |
| B5 | **Warm-up at load** (rule 76): after a model loads, one short read of a small synthetic page (a few tokens) builds the encoder, loads the PTX modules and runs the tuning measurements; `ReaderStatus` reports cold and warm apart; the tuning cache (`~/.cache/idrak/tuning/cuda`) stays on | the first page in the demo is a warm page | `legal start` test: the status after load says "warm" | S |
| B6 | **Telemetry events for a read** (rule 78): `Inference`-level events for image decode and transforms, the encoder, the prefill, each decode chunk, graph capture and the cache allocation, with sizes; `idrak trace` and the app's "done" event read them instead of stopwatches | measurable stages; the before/after numbers of every lane | `"telemetry"` tests; events free when nobody listens | M |
| B7 | **Preview encode off the critical path**: the PNG of the prepared page (today encoded before the read starts, `CompressionLevel.Optimal`, byte-at-a-time CRC) runs on a worker beside the read; the encoder gets a fast level, a Sub/Up filter, a slicing-by-8 CRC (lane C8) | est. 20-40 ms off the first token | decoded pixels identical (no fixture pins the PNG bytes) | S |
| B8 | Greedy decoding without the top-k path: an argmax kernel when `TopK` is 1 and no penalties (today `topk_candidates` over 262 k logits, 3 pooled buffers and top-5 statistics per token); penalties, when on, in a multi-block kernel (today one block copies 262 k logits and scans the history in O(n²), `RepeatLastN` up to 4,096) | est. 10-20 µs a token; penalties est. 50-150 µs a token at n = 64 | the same token per step; `"sampling"`, `"decoding"` | S |

### Lane C: the image pipeline on bytes (F5, F6)

| Id | Change | Gain | Proof | Size |
|---|---|---|---|---|
| C1 | **8-bit planes inside `ImageData`** (public API change, `api/` updated): the codecs produce byte planes, the Pillow-exact transforms and the preprocessor work on them, `Pixels` (floats) is made on first access with each codec's exact conversion (PNG's `k/255f` and JPEG's `k*(1f/255f)` differ in the last bit for 126 of 256 values, so each keeps its own) and the byte view is dropped once floats are handed out (the array is mutable). Option without an API change: a `ConditionalWeakTable<ImageData, byte[][]>` side cache in core that `ToBytes` reads first | the card's pipeline est. 90 → 45-55 ms; a decode loses its 72 MB float write; 4x less memory per image | the 238 SHA-256 transform cases + 51 JPEG encoder files (`"image transforms"`), `"jpeg"` (21 fixtures), `"image preprocessing"` (54 cases + `data/vlm`), `"image famil"`, `"data loaders"` | L |
| C2 | **JPEG decoder**: the bit reader in locals of one scan loop, a combined AC table (run, size, value in one 9-10-bit entry), an 8-byte fill with `IndexOf((byte)0xFF)` (SIMD, rule 31), a `Vector128/256<int>` islow IDCT with the scalar fallback (the zero-AC shortcuts equal the full formula, so branch-free is bit-identical), upsample + colour + output fused per band of rows (rules 40, 73), `GC.AllocateUninitializedArray` for buffers fully overwritten, MCU-row pipelining for baseline files (no whole-image coefficient arrays), restart intervals decoded in parallel when the file has them. Never DCT-domain scaling (Pillow does not) | 150 → est. 60-80 ms for 2,000 x 3,000 | `"jpeg"` exact bytes; `DOTNET_EnableHWIntrinsic=0` too (rule 42) | L |
| C3 | **Resize**: the horizontal pass in SIMD (bytes widened to `Vector256<int>`, the weight vector padded to 8 or 16 taps, a scalar tail), the vertical pass with the accumulator in registers (vector chunks of x outside, taps inside) and `Clip8` fused into the store, `Coefficients` cached by (in, out, filter) (today `Math.Sin` per tap on every call, every crop) | 77-127 → est. 45-60 ms to 896; every `max_width`/`scale` step faster | integer sums are order-free: `"image transforms"`, `"image preprocessing"` exact | M |
| C4 | **Transforms**: `Blend` (contrast, brightness, sharpness) with `Vector<float>` and `HostParallel`; contrast's grey from the byte planes already made (today 3 planes converted twice); `Smooth`, `Map`, equalize, gamma, binarize parallel over chunks; `RankFilter` as a sorting network (3x3, 5x5) or a sliding histogram; the blur's vertical pass across a strip of columns with per-column accumulators (today one column at a time over row-major memory, rule 72); `Orient` as a tiled parallel transpose on bytes; the JPEG round trip's `ToYCbCr` parallel and SIMD with one pooled coefficient array; `ImageData.Resize` with flattened taps and parallel rows | each step 2-4x (est.) | `"image transforms"` exact (the per-pixel float order unchanged; .NET does not contract to FMA) | M |
| C5 | **PNG decode**: a 256-entry float table per depth instead of a division per sample, vectorised Up/Sub/Avg filters, byte planes for 8-bit files (C1) | est. 2-4x on conversion | `"image famil"`, `"image.png"` exact | S |
| C6 | **Encoder input resized on the device** (generic path exists: `ImagePreprocessor.ProcessOnDevice`, bit-exact with the host, `ImagePreprocessor.cs:117-122`): a batch path that uploads each view's byte planes once (`MemoryMarshal.Cast`), runs `resize_normalize` per view into a slice of [N,3,896,896], broadcasts grey to three channels in the kernel; `ProcessOnDevice`'s serial packing loop and `List<float>.Add` coefficients fixed; the Gemma 3 plug-in's `Encode` calls it (the plug-in's change, through the library's contract) | est. 10-20 ms of CPU a view → about 1 ms on the device; upload 9.6 → 0.8 MB (grey) a view | `"vision kernels"` (device resize against host), `"image preprocessing"`, `"SigLIP"` | M |
| C7 | Pan and scan crops cut from byte planes, not floats, and each crop's `Pixels` not re-converted | small | `"pan and scan"` | S |
| C8 | **PNG encode** (the preview): bytes used directly (a grey plane needs no interleaving), Sub or Up filter, `CompressionLevel.Fastest` as an encoder option, one output stream (today two `ToArray` copies), slicing-by-8 CRC (rule 0: no `System.IO.Hashing`) | est. 20-40 → 5-10 ms a preview | decoded pixels identical (round-trip test) | S |
| C9 | **`--bench-image`** in the test runner: decode, each transform and the card's pipeline, `Pixels`/`Process` to 896, PNG encode, `ChatImages.Read`, cold and warm apart, SHA-256 of every output and bytes allocated (`GC.GetTotalAllocatedBytes(true)`), on `data/image-transforms`' 1,654 x 2,339 scan and a 2,000 x 3,000 4:2:0 file | the lane's before/after numbers | — | S |

### Lane D: the vision encoder on the device (F7)

| Id | Change | Gain | Proof | Size |
|---|---|---|---|---|
| D1 | **Encoder precision as a generic option**: `VisionEncoderOptions.Precision` (and a `VisionOptionKeys` key per request) selecting bfloat16 tensor-core products for the encoder alone; the app exposes it; off by default | 0.43 → 0.11 s a view (measured); with 4 crops 2.2 → 0.55 s | **changes results** (transformers' bf16 tower: cosine 0.9968 to fp32, `plans/11:409-410`): CER on the Evaluate tab before and after decides | S |
| D2 | **Epilogues**: the packed tensor-core kernel's `p_bias`/`p_aux`/`beta` slots (`PtxKernels.TensorCore.cs:182-183,765-772`) wired for a packed `MatMulBias` (q/k/v, output, fc2), GELU in fc1's epilogue, the residual through aux or beta (under split-K only block z = 0 adds the bias; order kept as residual + (acc + bias)) | est. 370 MB a layer of element-wise traffic gone | `"SigLIP"` (1e-5), `"packed"`, `"mixed precision"` | M |
| D3 | **No head copies**: `AttentionSpans` reads q/k/v strided from the [N,T,3D] product and writes [N,T,D], as `Tensor.CausalAttentionPacked` already does for the decoder (3 `Narrow` + 4 `Permute` copies a layer gone) | est. 264 MB a layer | `"attention spans"`, `"SigLIP"` exact (same values read) | M |
| D4 | **Fused residual + LayerNorm** returning (sum, normalised), modelled on `Tensor.AddRmsNormAffine`; one pass per residual | one pass a residual | `"normalization"` cases, `"SigLIP"` | S |
| D5 | Patch embedding as one GEMM with bias and the positions in the epilogue (stride = kernel, no padding: exactly a reshape to [4096, 588]) instead of a `Conv2d` + `Permute` + add | about 1% of FLOPs, 3 launches | `"SigLIP"` within tolerance | S |
| D6 | The known attention choice looked up before any driver memory query (`ComposedChosen` runs `GetMemoryUsage`/`AvailableMemory` on every call: 27 queries a pass) | small, exact | `"attention spans"` | S |
| D7 | The encoder pass recorded as a `ComputeGraph` per batch size after a warm-up (ranges are cached device buffers already) | est. 2-5% (about 500 launches a pass; more on Windows WDDM) | features bit for bit equal to the eager pass | M |
| D8 | **`--bench-vision bf16`**: packed bfloat16 weights (as the real checkpoint stores them: today the bench runs dense float32 only) under both precisions, batches of 1 and 5 views; the tensor-core span kernel's dim-72 padding to 80 (about 11% waste) measured and, if worth it, 9 n8 tiles for P·V | the lane's numbers; the real path has never been benchmarked | — | S |
| D9 | CPU span attention: Kᵀ and V packed once per head, an every-key path without per-row masking, or composed per 64-row block (1 MB of scores, exact softmax, two GEMMs: at 1,024 tokens composed ran 64 ms against spans' 121 on the CPU, `plans/11:190`) | CPU encoder 71.8 s → est. 25-30 s on 4 threads (a laptop without a GPU) | sum order changes: `"SigLIP"` (1e-5), `"attention spans"`; set aside under CUDA-first unless the demo machine needs it | M |

### Lane E: memory that fits the page, not the window (F4)

| Id | Change | Gain | Proof | Size |
|---|---|---|---|---|
| E1 | **A key/value layout for windowed layers** (generic, `KeyValueLayouts`): a layer whose `DecoderSpec` window is W keeps a ring of W positions (modulo addressing in the write, `ExpandPrefix` and the decode and span kernels' key ranges), global layers keep the context; chosen from the spec, never from the family | 34 x 8,192 x 8 KB = 2.2 GB → 5 x 8,192 x 8 KB + 29 x 1,024 x 8 KB ≈ 0.56 GB in float32 (est.); less to zero per read | the same tokens on every fixture (`"decoding"`, `"image prefill"`, `"Gemma 3 style"`, `"decoder"`); the layout's conformance case | L |
| E2 | **The cache sized to prompt + answer**, not the configured window: `ImageReader` allocates `ContextLength` or (image tokens + prompt + `MaxTokens`), whichever is smaller; the decode-split tuning keyed by capacity may change the float order, so tokens are checked | half the cache at the app's defaults (est.), allocated faster | tokens equal; `"image reader"` | S |
| E3 | **bfloat16 KV cache** as the app's documented option with the CER gate (not bit-identical: 0.036 logit difference on the tiny model, `plans/11:756-757`); `KeyValueCache: "bf16"` already exists | half the cache | CER before and after | S (gate only) |
| E4 | **Weights**: int8 projections with the bfloat16 head (`HeadWeights` exists) as the documented option for a smaller device, gated by CER | about 2.3x less weight memory (est.) | CER before and after | S (gate only) |
| E5 | The `--bench-offload`/`idrak memory` report for the read path: weights, cache, encoder activations and pool, cold and warm, so a device's fit is measured (rule: never a GB budget in docs or code) | the owner sees what fits before the demo | — | S |

### Lane F: tuning does each thing once (F8; A2 first for the big wins)

| Id | Change | Gain | Proof | Size |
|---|---|---|---|---|
| F1 | **One "encoding images" stage in `Prepare`**: each image decoded and transformed exactly once (rule 75), on worker threads (PLINQ `AsOrdered`, as `EncodeRows`), blocks and the family's `PixelValues` from the same pixels, the feature cache filled through the batched frozen pass overlapping the next decodes; the blocks stored with the disk cache so a second run decodes nothing | est. 3 min → 20-40 s for 300 pages; no GPU-idle decode inside steps | features hash equal with and without the stage (plan 12 accepts 1e-5 for batching); `"vision tuning"`, `"conversation tuning"` (the "tower runs twice, four hits" counts move to the stage) | M |
| F2 | **CER scoring**: `TuningAnswerScorer.RepeatStop` (opt-in; the app passes its `Generation.RepeatStop`: today a looping page runs all 1,536 tokens while the Read tab stops at 4 repeats), a KV format option, one `DecodingContext` and one graph across the scoring's pages, a memoising `ChatImages.Decode` for the scored pages (today decoded again at every scoring), the sequences cut at `MaxLength` counted and logged (they never see an end-of-turn and loop when scored) | up to 1,536 tokens per looping page; about half the scorer's cache; est. 0.1-0.5 s of capture per scoring | opt-in; greedy tokens equal on `"cli vision tuning"` when off | M |
| F3 | **Batched generation with images**: images in `StreamBatch`, per-row-start windowed attention at head 256 (`AttentionRows`) on CUDA (needs lane A's kernels) | est. 4-6x on scoring (decode is memory-bound) | `"image reader"`, `"chat parts"` batch cases equal one at a time | L |
| F4 | `ConversationTuningOptions.EvaluationSamples` (the loss over N evaluation pages; today every page, twice per 1-epoch run) | proportional to val.json | default unchanged | S |
| F5 | **Data reading**: records read with images pending (`PendingImage` exists), the training and held-out indices chosen, then only the chosen images inflated and hashed; `ChatImage` reads straight into its array (today `new byte[]` then `ToArray()`: two large-object copies per page) | in proportion to the pages left out; the dataset's size in garbage | same bytes and hashes; `"tuning data"` | S |
| F6 | **Packing and graphs with images on CUDA** (after A2): packing allowed when every sequence holds images under a non-causal rule (the spans handle packed rows), the first graph recorded with `imageTokens` = the most of any batch (known up front), graphs keyed by (rows, length) only | est. 3-10% a step from launches; no graph recorded that does not fit | `"vision tuning graph"`, `"vision tuning packing"`, packed against padded within 1e-4 | M |
| F7 | The out-of-memory ladder's first rung from measured bytes per position after the first step; `Checkpointing`/`RecomputeFeedForward` exposed to the app; cached memory trimmed after each scoring | a few wasted partial steps | trace lines unchanged | S |
| F8 | Feature cache: the dtype in `FeatureCacheKey` from the tower's actual output type (today "float32" hard-coded while the frozen tower runs bf16); a bfloat16 tower stage only when the output is bf16-exact (the contract says a hit is exactly what was put); an optional device-resident tier for small sets | host RAM ÷2 for a trained projector | `"vision tuning memory"` | S |
| F9 | `ChatTranscriptEncoder`: a transcript over `MaxLength` rendered and tokenized three times before the cut; give up at once when the answer alone exceeds the budget; `Encode` parallel | seconds | `"vision tuning reference"` ids and blocks equal | S |
| F10 | CER metric: common prefix and suffix trimmed, then a bit-parallel edit distance (exact) | negligible next to generation (lowest priority) | unit test against the current loop | S |

### Lane G: load time (F9)

| Id | Change | Gain | Proof | Size |
|---|---|---|---|---|
| G1 | **bf16 straight through**: a load path that reads bf16 bytes, transposes them as 16-bit values and uploads packed, instead of widening to float32 on the host and rounding back; the embedding not held as one float32 array | load time and host peak (the 2.7 GB transient gone) | identical values: `"packed"`, `"decoder"`, the `vlm` reference tokens | M |
| G2 | The vision encoder built at load (or by B5's warm-up), not at the first read; the checkpoint opened once | the first page pays nothing but the pass | `legal start` | S |
| G3 | Load time measured and reported by stage (tokenizer, safetensors, conversion, upload, adapter merge) through B6's events | numbers for G1 | — | S |

### Lane H: measuring (F10) and the test runner

| Id | Change | Gain | Proof | Size |
|---|---|---|---|---|
| H1 | **`IDRAK_FILTER` with alternatives** (`a|b|c`, case-insensitive substrings) in `tests/Idrak.Tests`, `Idrak.Samples.LegalOcr.Tests` and the other sample runners, so a lane's whole group set runs in one invocation (today one substring: `t.Name.Contains(f)`, `Program.cs:160`) | one test run per lane, never a chain | the runner's own test | S |
| H2 | `--bench-image` (C9), `--bench-vision bf16` (D8), read-stage telemetry (B6), `HostCall` trace (A5) | the before/after numbers every lane must report | — | S each |
| H3 | A `--bench-read` in the LegalOcr tests (tiny Gemma 3 fixture, CPU): a read's stages timed and allocations counted, cold and warm | the app's regression number | — | S |

## 3. The test and build process every agent follows

From CLAUDE.md and the runner (`tests/Idrak.Tests/Program.cs:4-20,126-200`):

1. **Read `docs/optimization.md`** before touching library code; read this plan's lane and its proof column.
2. **Fix everything in the lane first; never run tests between fixes.** Update the tests the change touches (expectations,
   and failure messages that name the cause: which operation took a host fallback, which fixture differs).
3. **Build once**, only the projects that changed, Release: for a library lane `dotnet build -c Release tests/Idrak.Tests`
   (it references every package and the Gemma 3 plug-in); for an app lane also
   `dotnet build -c Release samples/Idrak.Samples.LegalOcr.Tests`. One build per lane, not per fix.
4. **One test run per lane, `--no-build`, CPU only, every related group at once**: `IDRAK_DEVICES=cpu` and one
   `IDRAK_FILTER` that covers the lane's groups. Until H1 lands the filter is one substring, so each lane names a shared
   one and lists the exact test names it covers; after H1 it is the lane's groups joined with `|`. Never a sequence of
   runs with different filters. Never the full suite. `IDRAK_TRACE=1` for stacks, `IDRAK_TIMEOUT` for long groups.

   | Lane | Filter until H1 (one substring) | Groups after H1 |
   |---|---|---|
   | A | `attention spans` (covers "attention spans" and "attention spans backward"); image prefill runs with `image` | `attention spans\|image prefill\|conformance kit\|kernels` |
   | B | `image` (image reader, image prefill, image transforms, image preprocessing, image families, cli images) | `image reader\|chat parts\|cli images\|decoding\|telemetry\|pool\|memory` |
   | C | `image` | `image transforms\|jpeg\|image preprocessing\|image famil\|image.png\|pan and scan\|data loaders\|vision kernels` |
   | D | `SigLIP` | `SigLIP\|attention spans\|packed\|mixed precision\|normalization` |
   | E | `decod` (decoding, decoder) | `decoding\|decoder\|image prefill\|image reader\|Gemma 3 style\|conformance kit` |
   | F | `tuning` (vision tuning, conversation tuning, cli vision tuning, cpu tuning, fine-tuning) | `vision tuning\|conversation tuning\|cli vision tuning\|fine-tuning` |
   | G | `decoder` | `decoder\|packed\|image reader` |
   | app | `legal` (the LegalOcr test project, all 9 tests) | same |

5. **Prove the result unchanged (rule 71)** inside that run: the fixtures of the proof column are the gate; an opt-in
   change that alters bits keeps its default off and adds the CER gate to the owner's command list.
6. **Numbers**: the lane's benchmark before and after, cold and warm apart, pasted into this plan's lane table when the
   lane merges (the owner's CUDA numbers in the owner's hand-back).
7. **Merge and push** to `abstraction`; a clean merge gets no build and no tests. **No CHANGELOG, no version bump.**
8. **Hand the owner one targeted command list** (PowerShell, `D:\Projects\Idrak`): one `dotnet build -c Release
   tests\Idrak.Tests`, then the lane's CUDA filter with `--no-build`, then the lane's benchmark. Section 4 has them.
9. **At most two agents building at a time** (CLAUDE.md), so lanes pair up: A with C, B with D, E with F, G with H.

**This cloud environment cannot build.** It has no .NET SDK and its network policy denies the SDK's download host
(`builds.dotnet.microsoft.com`, 403 from the gateway) and the apt feed. For agents to run step 3 and 4 in cloud sessions,
the owner adds the SDK install to the environment's setup script (`installation/linux.md`, section 2, the
`dotnet-install.sh --channel 10.0` line) and allows that host in the network policy; otherwise the coding session runs on
the owner's machine, where steps 3 and 4 are the same commands in PowerShell.

## 4. The owner's commands (PowerShell, `D:\Projects\Idrak`, CUDA)

Before any lane merges, the baseline once (keep the output):

```powershell
cd D:\Projects\Idrak
git pull origin abstraction
dotnet build -c Release tests\Idrak.Tests
$env:IDRAK_DEVICES = "cuda"
$env:IDRAK_FILTER = "attention spans"; dotnet run -c Release --no-build --project tests\Idrak.Tests
dotnet run -c Release --no-build --project tests\Idrak.Tests -- --bench-spans
dotnet run -c Release --no-build --project tests\Idrak.Tests -- --bench-vision
$env:IDRAK_MATMUL = "bf16"; dotnet run -c Release --no-build --project tests\Idrak.Tests -- --bench-vision; Remove-Item Env:IDRAK_MATMUL
# the app: one cold read, one warm read of the same evaluation page, then "Evaluate the model" on 20 pages (CER, seconds, tokens/s)
dotnet run -c Release --project samples\Idrak.Samples.LegalOcr
Remove-Item Env:IDRAK_DEVICES, Env:IDRAK_FILTER
```

After each lane (the lane's filter and benchmark; the app's Evaluate on the same 20 pages for any gated option):

```powershell
cd D:\Projects\Idrak
git pull origin abstraction
dotnet build -c Release tests\Idrak.Tests
$env:IDRAK_DEVICES = "cuda"
$env:IDRAK_FILTER = "<the lane's filter>"; dotnet run -c Release --no-build --project tests\Idrak.Tests
dotnet run -c Release --no-build --project tests\Idrak.Tests -- <the lane's bench: --bench-spans | --bench-vision | --bench-image | --bench-gemv>
Remove-Item Env:IDRAK_DEVICES, Env:IDRAK_FILTER
```

For lane F, a profiled short run (host fallbacks and GPU time per step, before and after A2):

```powershell
$m = "google/gemma-3-4b-it"
idrak tune -P samples\Gemma3Vision\Idrak.Gemma3Vision\bin\Release\net10.0\Idrak.Gemma3Vision.dll -b $m --data D:\Data\arabic-legal-ocr\data\train.json --images D:\Data\arabic-legal-ocr\data\downloaded_images.zip --image-transform "grayscale,max_width=1024,contrast=1.5" --max-rows 32 --profile -v
```

## 5. Demo day without code (settings and order; each gated choice measured first)

1. `LoadAtStart` on; one full read before the demo so the tuning cache and the PTX modules are warm (B5 does this
   automatically once built).
2. Keep the default float32 cache and bfloat16 weights unless the Evaluate tab shows the same CER with `KeyValueCache:
   "bf16"` (E3) or int8 projections with the bfloat16 head (E4) on the same 20 pages.
3. `Context` no larger than the longest page needs (image tokens + prompt + `MaxTokens`, with pan and scan at most
   1,280 image tokens): E2 does this per read once built; until then the setting.
4. Pan and scan off unless a page needs it (each crop is one more encoder pass and 256 more prompt tokens).
5. The Generation section's `RepeatStop` 4 on (it is), so a looping answer ends.

## 6. Order and pairing for the coding session

| Wave | Lanes in parallel (two building at a time) | Why |
|---|---|---|
| 1 | **A** (A1, A3, A5) and **B** (B1, B2, B4, B5) | the largest win on the device and the demo's CPU-side wins; both prove exact |
| 2 | **C** (C1 first, then C2, C3, C6) and **H** (H1, H2) | bytes through the pipeline; one test run per lane from here |
| 3 | **E** (E1, E2) and **D** (D2, D3, D6, D1 gated) | memory that fits the page; the encoder's launches and traffic |
| 4 | **F** (F1, F2, F4, F5; A2 then F6) and **G** (G1, G2) | tuning once-per-image; load time |
| 5 | the rest (B3, B6-B8, C4, C5, C7, C8, D4, D5, D7-D9, F3, F7-F10, G3, H3) | each small and independent |

## 7. Decisions for the owner (each with a default)

- **Encoder precision (D1)**: bfloat16 tensor cores for the encoder change features (cosine 0.9968 in transformers' own
  bf16 tower). Default: an option, off; on only if the Evaluate CER on 20 pages holds.
- **`ImageData` bytes (C1)**: the public API change (byte planes with lazy floats) or the side cache without one.
  Default: the API change, since every pipeline stage then stays in bytes; `api/` updated.
- **Windowed cache layout (E1)**: a new layout in `KeyValueLayouts` chosen from the `DecoderSpec` window. Default: yes,
  behind the registry, with the full layout kept as it is.
- **CPU encoder work (D9)** is set aside under CUDA-first unless the demo machine has no GPU.
- **The cloud environment's SDK**: add the install to the setup script and allow the download host, or run the coding
  session on the owner's machine.
