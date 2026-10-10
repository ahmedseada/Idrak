# Plan 14: the refactor backlog

## Scope from now: CUDA first

The owner's decision: **CUDA is the most important target from now. Anything not related to CUDA is set aside**
(Vulkan, HIP, CPU-only kernel work, the CLI, samples other than what CUDA training needs, the unfinished audit areas
outside the CUDA path). The sections below keep those items for later; work only from this list until the owner says
otherwise.

| Order | CUDA item | Where (section below) |
|---|---|---|
| 1 | Verify the fused LSTM/GRU cell kernels on CUDA (the `recurrent` filter) and measure the OCR training step with `train --profile 10` against the 411 ms step | 1.5, 4 |
| 2 | Regression check of the CUDA path against the library's starting numbers (`--bench-gemm`, `--bench-gemv`, the Chat sample's `profile ... --cuda --int8 --kv16`) | 4 |
| 3 | Size classes for the CUDA keys still keyed by exact data sizes: PackedSplits/PackedTile/PackedMulti/Gated (`m`), DecodeSplits/DecodeMinChunk (`capacity`) | 2, Tuning |
| 4 | Convolution choice per call: candidacy that does not depend on free memory at the moment, so a known choice is a lookup (no candidate list and `cuMemGetInfo` per convolution) | 2, Tuning |
| 5 | Cheaper measuring on CUDA (shorter warm-up once the GPU is busy, fewer rounds for far-apart candidates) | 2, Tuning |
| 6 | Library paths that cost CUDA time on every step: graph constants and BatchNorm eval read back or recomputed per forward (a GPU sync per node; needs a buffer invalidation hook), `ActivationMemory` arrays per layer per step, `Conv2d.ForwardFused` refolding per inference call, `ExactGELU` as ~20 launches (a fused CUDA kernel) | 2, Layers and models |
| 7 | R1's CUDA leftovers: `CtcMeta` per CTC call, closures and copies on prompt-sized packed products and tensor-core `Run` delegates, invariant formatting in the PTX generators | 2, CUDA |
| 8 | Card names in CUDA code comments (`CudaBackend.Quantized.cs`, `CudaDeviceLimits.cs`, `TuningCache.cs`) | 1.3 |
| 9 | The CUDA runs of `plans/gpu-checks.md`, plan 12 phase 5 on the real Gemma 3 (CUDA), and docs/performance-notes.md next steps 1-9 (all CUDA) | 4 |
| 10a | **Done** (α and β rescaled every step, offsets in double; passes on CUDA). CUDA CTC precision on long sequences: α and β in float log space reach ~-1,500 over 810 steps, so the 400-label gradient is ~0.1% off the CPU's double (test "vision kernels: ... CTC, 400 labels, batch first gradient"). Fix: rescale α and β every step (subtract the row's maximum), offsets accumulated in double; the per-state shares are unchanged and only the offsets enter the final term | 2, CUDA |
| 10c | **Done**: CUDA kernels for sqrt, sin, cos, SiLU, sign, maximum, minimum, pow, clamp and where and their gradients (they took the host fallback; "tensor ops" now requires no host fallback on CUDA); all pass on CUDA | - |
| 10d | **Done, to verify on CUDA** (the `vision kernels` filter, then `train --profile 10`). The OCR profile (1908 ms a step, 107 ms of GPU time): `ctc_loss_bwd_global_f32` 55 ms a call (30% of GPU time) and `matmul16_nt` 16x128x512 at 0.1 TFLOPS (22%). Fixed: CTC rows in dynamic shared memory whenever they fit what the device lets a block opt in to (was: only when the states fit the block width, so OCR lines took the scratch variant); the gradient groups the labels once a call instead of rescanning them for every label at every step (O(L²) loads a step); the per-step maximum by shuffles. Few-row products (gemv) take up to 16 rows in one launch (row groups along grid y), so the recurrent step's h·U and dz·Uᵀ (16 rows: the batch) no longer run on 8 blocks of the 16x16 kernel. `--profile` splits a step into data and compute | 2, CUDA |
| 10e | Measured on the owner's card after 10d: 70.1 ms a step with measuring off (was 411), data 3 ms; 913.9 ms with measuring on, so the gap is measuring new size classes during the first steps (item 5). The CTC gradient is ~0.5-0.85 ms a call (was 55). Next on the GPU: `gemv_nt_f32` at ~36 µs a call (40% of GPU time) because its 8 row loads shared one predicated register and ran one after another; fixed (a register per row, two k a pass), to verify. Then fusing h·U and dz·Uᵀ into the cell kernels (one launch a step) | 2, Tuning |
| 10f | **Done** (passes on CUDA): `gemv_nt_f32` 36 → 11.4 µs a call (a register per row). OCR step 67.2 ms with measuring off | - |
| 10g | **In progress.** The recurrent loop is launch-bound: 4 launches a time step (h·U, cell; cell gradient, dz·Uᵀ), each near the launch floor, ~1,600 a training step, ~60% of GPU time. Fix: one persistent kernel per layer and direction for the forward loop and one for the backward loop (each block keeps its units' slice of U in shared memory; blocks co-resident by a cooperative launch, a grid barrier between steps; block count from the device's reported limits) | 2, CUDA |
| 10h | **In progress.** `group_reduce_f32` 0.56 ms a call and `norm_stats_f32` 0.7 ms (BatchNorm statistics, conv bias gradients): one block per channel walks hundreds of thousands of elements while most SMs idle. Fix: split each group into fixed-size chunks across blocks (partial sums, then a short final pass in a fixed order) | 2, CUDA |
| 10i | `conv_bwd_weight_f32` 2.8 ms a call (20 calls, 57 ms in the profile): check its parallelism the same way (a reduction over N·OH·OW per weight) | 2, CUDA |
| 10j | Measuring during the first training steps: 914 ms a step with measuring on vs 67 ms off (new size classes measured one after another). Cheaper measuring (item 5) or measuring off the step's critical path | 2, Tuning |
| 10b | CUDA does not pack image-model sequences (Gemma 3, LLaVA: "cuda:0 does not pack this model's sequences (FineTuner pads them)"; LLaVA: no packed causal attention): padded rows waste work in vision fine-tuning | 4 |
| 10 | A training-step profile on the CUDA path beyond kernel times (per operation and layer: the CLI's internal `Recorder` as a library hook) | 2, gaps |
| 11a | **Done, to verify on CUDA** (PTX rules 80 and 87, results unchanged: the same instructions, each accumulator's terms in the same order). Rule 80, a register per loaded value so a pass's loads are in flight together (the gemv_nt pattern): the 18 int8/int4/bfloat16 `*_gemv_*` kernels (`PtxKernels.Quantized.cs` `Int8Gemv` `Step`: 8 rows × 4 k a pass through one predicated register, 32 loads in a chain), `attention_decode_f32/int8/bf16` (`PtxKernels.Rows.cs` `AttentionDecode`: the 8 key and 8 value loads of each cached position), `attention_flash_*` PV loop, `attn_bwd_q_f32` PQ loop and their span copies `attention_spans_f32` (`PtxKernels.Spans.cs`) and `attn_spans_bwd_q_f32` (`PtxKernels.SpansBackward.cs`) (4 shared-memory loads per key), `gemv_nn_f32`/`gemv_multi_f32` K1 tail. Rule 87, `.maxntid` on the kernels launched with blocks of 512 threads or more: `gemv_nn_f32`, `gemv_multi_f32` (1024), the `*_gemv_*` family (512), `softmax_ce_rows_f32` (`SoftmaxCrossEntropyThreads`, now also the launch's), `sample_rows_f32` (the shapes' SamplerThreads). Expected: decode-time products and attention closer to memory speed (gemv_nt went 36 → 11.4 µs with the same change); measure with `--bench-gemv` and the Chat sample's `profile ... --cuda --int8` | 2, CUDA |
| 11b | `gemv_nn_f32` / `gemv_multi_f32` (`PtxKernels.Rows.cs:2269`), rule 81: k is split only over the 32 slices of one block, so a narrow product runs on ⌈n / 32⌉ blocks per row group (n = 128: 4 blocks of 1024 threads) while the other SMs idle, however long k is. Fix: k splits across blocks (a grid dimension, `chunk` rows each) with partial sums and the last block of a column range adding them in split order, as `int8_gemv_f32` does; the split count from the device's multiprocessors and the shape. Expected: narrow products with long k fill the device (time divided by about the blocks added, up to the memory bound) | 2, CUDA |
| 11c | `gemv_nt_f32` (`PtxKernels.Rows.cs:2548`), rule 81: one warp per column walks all of k, grid x = ⌈n / 8⌉, so a few columns (the recurrent step's dz·Uᵀ with few units, attention's q·kᵀ over a short cache) launch a handful of blocks. Fix: k chunks across blocks with fixed-order partial sums (as 11b). Expected: the small-n calls stop being latency-bound on a handful of SMs | 2, CUDA |
| 11d | `matmul_f32` (`PtxKernels.cs:719`, tile loads at `:802`), rules 83 and 81: with transA (transB) the tile loads read A[(t + tx)·m + row] (B[col·k + t + ty]), so consecutive threads read addresses m (k) floats apart: uncoalesced; and 16 × 16 tiles give few blocks on small products (10d's 16×128×512 at 0.1 TFLOPS). Fix: load a transposed operand with the thread roles swapped (tx along its contiguous dimension) and write it transposed into shared memory; route small-m products to the k-split kernels. Expected: full-width reads for the transposed layouts | 2, CUDA |
| 11e | `sum_f32` (`PtxKernels.cs:642`, `atom.global.add.f32` at `:706`), rule 86: each block's partial is added atomically, so the sum's bits depend on which block finishes first (run to run, card to card). Fix: per-block partials to scratch and the last block (counted, as `int8_gemv_f32`) or a second one-block pass adds them in block order. Expected: the same bits every run, at the cost of one counter or one tiny launch | 2, CUDA |
| 11f | `sumsq_f32`, `multi_sumsq_f32` (`PtxKernels.Rows.cs:430`, `:466`; the gradient norm of clipping), `sum_cols_strided_f32` (`:539`), `layernorm_bwd_params_f32` (`PtxKernels.Decoding.cs:110`, dgamma/dbeta), rule 86: per-block or per-chunk sums added with `red.global.add.f32`, so the order follows the scheduling. Fix: partials per block or chunk, added in a fixed order (11e's pattern). Expected: deterministic clipping factors and LayerNorm parameter gradients | 2, CUDA |
| 11g | `gemm_tc_*` split k (`PtxKernels.TensorCore.cs:975`, `EmitTensorEpilogue`), rule 86: each k split adds its partial tile into C with `red.global.add.f32`, so a split product's bits change with block order. Fix: partial tiles to a workspace and the last split of a tile adding them in split order (a counter per tile), or a fixed-order reduction pass. Expected: split-k products reproducible bit for bit | 2, CUDA |
| 11h | `sum_rows_f32` (`PtxKernels.cs:590`), rule 81: one thread per column walks every row, so a narrow layer's bias gradient over a long batch runs ⌈cols / BlockSize⌉ blocks with a serial loop over all rows. Fix: chunks of rows along grid y with partials added in chunk order (11e). Expected: such gradients use the whole device | 2, CUDA |
| 11i | Rule 80 left in epilogues (one pass per tile, low value): `gemm128/64_*` beta reads (`PtxKernels.Rows.cs:1867`, 8 or 4 predicated loads through `%f2`), the dK/dV write-back of `attn_bwd_kv_f32` (`:1368`) and `attn_spans_bwd_kv_f32` (`PtxKernels.SpansBackward.cs:345`, `%f23`/`%f24`, 4 a key); the span kernels' range prologue (`%f2`/`%f3` per row). Fix when the file is next touched: a register per column / slot, the same order | 2, CUDA |
| 11j | Rule 49 left: `PtxKernels.Advanced.cs`, `PtxKernels.Convolution.cs`, `PtxKernels.Recurrent.cs` still interpolate numbers with the current culture (they were being edited elsewhere); the other generators format theirs invariantly (StringBuilder appends with `CultureInfo.InvariantCulture`, `string.Create`). The reasons in `CudaDeviceLimits.cs` (`Derive`, `Fits`) are messages for people and keep the current culture | 2, CUDA |

Set aside (not CUDA): 1.1 Vulkan CTC, 1.2 GGUF fallback, 1.4 card names in docs, 1.6 Vulkan recurrent, 1.7 the
sample's non-finite loss message, the Vulkan/HIP and CPU backend items, the CLI items, the sample gaps that are not on
the CUDA training path, and the unfinished audit areas (section 3) except where a file is on the CUDA path.

Everything left unfinished by plans 12 and 13, the fused recurrent work and the optimization-rules audit
(`docs/optimization.md`, eleven areas R1-R11), in one place, so the refactor can be planned from it. Each item says
where it is, why it was left, and what a fix needs. Nothing here is guessed: every entry comes from an agent's report,
a measurement on the owner's card, or a count over the branch.

## State at the time of writing

| Work | State |
|---|---|
| Fused LSTM/GRU time loop (CPU and CUDA cell kernels, one autograd op per layer and direction) | merged; CPU tests pass; CUDA kernels never run on a GPU |
| Measured kernel choices keyed by size class (`TuneSizes`) | merged; CUDA keys and the shared convolution key only |
| Audit R1 CUDA, R2 Vulkan/HIP, R3 CPU devices, R5 Layers/Models, R9 CLI N-Z + Shared | finished and merged |
| Audit R4, R6, R7, R8, R10, R11 | stopped part-way; their committed fixes merged; the rest of their areas not covered (below) |
| Whole solution after the merges | builds, 0 errors, 0 warnings; tests after the merge are the owner's |

## 1. Correctness and rule breaks (fix first)

| # | Where | What | Fix needs |
|---|---|---|---|
| 1.1 | `src/Idrak.Gpu/Vulkan/VulkanKernels.Ctc.cs` | CTC loss is -∞ on the first step on Vulkan (the ArabicOcr sample's tests: 4 of 13 fail on vulkan; CPU and CUDA pass) | find the kernel bug (α rows, log-sum, meta reads); the conformance kit's CTC case on Vulkan should catch it |
| 1.2 | `src/Idrak/Models/GgufModel.cs:297` | an unregistered `tokenizer.ggml.pre` silently gets Llama 3's pattern: a family fallback CLAUDE.md forbids | throw "pre-tokenizer '{pre}' is not registered; register its pattern with GgufPreTokenizers.Register", and change `CustomGgufPreTokenizer` in tests/Idrak.Tests/ModelPluginTests.cs, which asserts the fallback |
| 1.3 | `src/Idrak.Gpu/Cuda/CudaBackend.Quantized.cs` (4), `CudaDeviceLimits.cs` (4), `TuningCache.cs` (1), `Vulkan/VulkanBackend.Dispatch.cs` (1) | card names in comments (card-agnostic rule) | reword as what was measured, without the card |
| 1.4 | README.md (51), plans/*.md (~55), src/Idrak/README.md (3), docs/performance-notes.md, docs/optimization.md | card names in benchmark tables and notes | owner's decision: a results table names the machine it ran on; advice or defaults must not |
| 1.5 | fused recurrent CUDA kernels (`PtxKernels.Recurrent.cs`, `CudaBackend.Recurrent.cs`) | never run on a GPU (no GPU here) | the owner's `recurrent` filter on cuda; then the OCR `train --profile` against the 411 ms step |
| 1.6 | Vulkan recurrent | no fused cell kernels: Vulkan still runs ~13 launches a step | `LstmCell`/`GruCell` kernels for Vulkan when Vulkan is back in scope |
| 1.7 | ArabicOcr sample `TrainCommand` | a non-finite loss crashes writing `recognizer.json` (JSON cannot hold ∞) | stop with a clear message on a non-finite loss (sample); 1.1 is the cause seen |

## 2. Speed and memory left in the library

### Tuning (measured choices)

- **Per-class measuring cost:** each measurement is ~0.2-0.5 s (warm-up ≥ 25 ms of GPU time, 8 paired rounds,
  a confirmation). With size classes a varying workload measures ~15 keys per class once; cheaper measuring (shorter
  warm-up once the GPU is busy, fewer rounds for far-apart candidates) would shorten the first minute of a run.
- **Exact keys still keyed by data sizes:** `TuneSizes` is applied to CUDA's FloatTile, TensorSplits, SpanWarps,
  AttentionPath and the convolution key. Still exact: CUDA PackedSplits/PackedTile/PackedMulti/Gated (`m`),
  DecodeSplits/DecodeMinChunk (`capacity`), and every Vulkan key with rows (Splits, Rows, MatMul, MixedMatMul,
  TiledAttention, Attention, SpanAttention, PackedPrompt). Decide per key which sizes follow the data.
- **Convolution choice per call** (`CudaBackend.Convolution.cs:84-150`): the candidate list and `cuMemGetInfo` run on
  every convolution, because the composed path's candidacy depends on free memory at that moment. Make candidacy not
  depend on the moment (a measured budget, or retry on allocation failure) so a known choice is a lookup.

### CUDA (R1, not fixed)

- `CudaBackend.Ctc.cs` `CtcMeta`: a `new float[3·batch]` per CTC call (also Vulkan's); pool it.
- Prompt-sized packed products (`PackedMatMulLargeKernel`, `PackedManyLarge`, LowRank): a closure, delegate and small
  copy per call; tensor-core callers' `Run` delegates (MatMulLowRank, BFloat16Transposed, GemmStrided,
  `TensorCoreRows`): two small allocations per large product.
- PTX generators interpolate numbers under the current culture; safe today (non-negative ints, floats through `F()`),
  but rule 49 wants invariant formatting.

### Vulkan / HIP (R2, not fixed)

- HIP `Launch`: a lock and two dictionary lookups per launch.
- `SupportsGraphs` reads an environment variable per recorded graph.
- `ReductionFlags`, `MatCandidates`, `PowersOfTwo`, `GemvCandidates` allocate (only while measuring).

### CPU backend (R3, not fixed)

- `LayerNormTrainKernel` computes mean and variance twice (it calls the `LayerNormFused` wrapper so registered
  overrides still apply); a fused training layer norm needs an override-aware operation.
- `GroupMoments` merges per-chunk sums under a lock in completion order: run-to-run bit differences with several
  threads. Deterministic order changes outputs once.
- Vectorized loops without an `IsHardwareAccelerated` check (rule 34): `Moments`, `MultiplyAddInPlace`,
  `GroupAffineCore`, `AddInPlace` (Advanced.cs) and loops in CpuBackend.cs. Correct in software mode; a scalar
  fallback changes bits (FMA, lane order).
- Depthwise convolution (Vision.cs) is scalar and runs its activation as a separate pass; vectorize across outputs and
  fuse the activation (bits change: `Vector.Exp` vs `MathF.Exp`).
- `MathF` loops in Pointwise (sin, cos, pow, sqrt, silu) and the gated-activation backward are scalar; the composed
  convolution's bias and activation passes are not fused.
- bfloat16 tiled kernel's leftover columns read weights column by column (< 2 vectors wide).
- Small per-call arrays: norm scale/shift copies, `Enumerable.Repeat` in `GroupScaleShift`, CTC per-sequence tables,
  `Permute`'s shape copies.

### Layers and models (R5, not fixed)

- `ActivationMemory.Release/Compress` take `params Tensor?[]`: an array per layer per step (also at inference, before
  the early return). `params ReadOnlySpan<Tensor?>` fixes every caller; public API change (regenerate the dump).
- `GraphModule.Binary` and `LibraryGraphOps.Scalar` read scalar constants back from the device every forward (a GPU
  sync per node); BatchNorm in eval mode recomputes its inverse standard deviation every call. Both need a buffer
  version or invalidation hook in Idrak.Abstraction (`Module.Load` can overwrite buffers in place).
- `Conv2d.ForwardFused` refolds the batch norm into a weight-sized temporary on every inference call (deliberate: no
  second copy of the weights). Owner's trade-off: cache the folded weights (memory) or keep refolding (time).
- `ExactGELU` is ~20 element-wise passes: a fused erf kernel in the device contract.
- GGUF F16 decode is scalar (the BCL's vectorized Half conversion is a NuGet package, which rule 0 forbids): write a
  vectorized converter.
- `Sequential` allocates one closure per whole-model forward.

### CLI (R9, not fixed)

- `TrainSession.cs:378-412` CSV relabelling for string classes: strings and `Split` per row (rule 74); move to bytes
  with the library's CSV reader.
- `ImageTraining.cs:906` uploads two tiny per-channel tensors per batch.
- `ServeHost` auth: two small byte arrays per request with an API key.
- The bidi algorithm allocates per right-to-left line.
- `DataProfile` duplicate detection keeps each row's JSON string (hashing could change results).

### Gaps the OCR sample showed (the library should offer them)

- The `Augmentations` registry has no `noise` (core's `GaussianNoise` is only a sample transform): the sample
  registers its own.
- The measured batching `idrak predict` uses (`MeasuredBatches`) is internal to the CLI: the sample copies it. Make it
  a library contract.
- `ChatImage` carries no source path: pages from a data file are named by record number.
- No training-step profile outside CUDA: the CLI's operation/layer `Recorder` (ProfileCommand) is internal; a library
  hook would give every app the per-operation table on any device (rule 78).
- The OCR sample's segmentation keeps touching lines as one line (sample).

## 3. Audit areas not finished

The agents of these areas were stopped part-way. "Untouched" = no fix committed: either not reviewed or reviewed and
found clean; the stopped agents cannot say which, so each needs a review pass. Largest first.

| Area | Files | Untouched | Lines | Largest untouched |
|---|---|---|---|---|
| R4 Abstraction (not Devices) | 75 | 59 | 11,093 | Tensor.Decoder.cs, ChatParts.cs, VisionLanguage.cs, Operations/Kernels.cs, Decoding.cs, RopeScalings.cs, KeyValueLayouts.cs, Data/ImageTransforms.cs, Telemetry.cs, DecoderSpec.cs, Tensor.Quantized.cs, SlotTable.cs, Augmentations.cs |
| R6 core (not Layers/Models) | 57 | 37 | 6,099 | JpegEncoder.cs, OnnxBuiltIns.cs, InferenceEngine.cs, ModelPackage.cs, Offloading.cs, FileSources.cs, ChatImageDecoder.cs, TrainerCallbacks.cs, OnnxExport*.cs, CtcDecoders.cs, AdamW8Bit.cs |
| R7 Nlp, AspNetCore, Mcp, Onnx.Runtime | 54 | 32 | 8,644 | Jinja.cs, JinjaBuiltins.cs, TextGenerator.cs, FineTuningLosses.cs, CodingAgent.cs, TuningVision.cs, AnswerScorer.cs, ChatGenerator.cs, FeatureCacheImplementations.cs, Conversation.cs, ChatApi.cs, TokenSamplers.cs |
| R8 CLI commands A-M | 53 | 26 | 3,767 | DoctorCommand.cs, ConfigCommands.cs, InspectCommand.cs, TuningShowCommand.cs, CompletionCommand.cs, AliasCommands.cs, NewCommand.cs, LoginCommands.cs (mostly cold commands) |
| R10 Data, Vision, Testing kit | 84 | 79 | 17,078 | ParquetFile.cs, DataFiles.cs, VisionAugmentations.cs, AnnotationFiles.cs, DatasetRows.cs, Sources.cs, DetectionMetrics.cs, ChatRows.cs, AugmentedImageLoader.cs, RegionClassifier.cs, DatasetRecipe.cs, Downloader.cs, ImageWarp.cs |
| R11 samples and tools | 56 | 42 | 7,349 | Gemma3Vision.cs, Samples.Chat, Shared/Gpt/CharGpt.cs, Samples.Rag, Siglip.cs, GptService.cs, CodingAgent, Summarizer, ArabicOcr/Readers.cs |

The hot ones, by the rules: R10's data readers and image code (rules 72-74 came from exactly this code), R7's
tokenizer-adjacent paths (Jinja runs per chat turn, TextGenerator/ChatGenerator per token), R4's tensor and decoding
code, R6's JPEG encoder and inference engine. R8's remaining files are cold commands.

Also never audited: the analyzers of rule 65 (`AnalysisModePerformance` = All) are not on; turning them on for every
project will list what the review missed (warnings are errors, so fix in the same pass).

## 4. Checks never run

- `plans/gpu-checks.md`: every CUDA and Vulkan run of plans 12 and 13 (conformance kit, attention spans, vision
  tuning, image prefill, vision kernels, conv, vulkan cnn, vision layers/sequence/detection/augment, image families,
  fine-tuning, packed, checkpoint) and the benchmarks (`--bench-spans train`, `--bench-conv`).
- Plan 12 phase 5: the real Gemma 3 model (`tune evaluate`, `tune`, `run --adapter`).
- The regression check against the library's starting numbers (docs/performance-notes.md: `--bench-gemm`,
  `--bench-gemv`, the Chat sample's `profile Qwen/Qwen3-0.6B --cuda --int8 --kv16`): 78 TFLOPS, 474-493 tok/s,
  177 launches per token, a 7.7 ms 180-token prompt pass.
- The OCR training step after the fused recurrent loop: `train --profile 10` against 411 ms (measuring off) and
  1.9-3 s (measuring per shape, before size classes).
- docs/performance-notes.md "Next steps" 1-9 (delayed FP8 scaling by call order, FP8 fused into producers, persistent
  decoding GEMVs, int8 prefill on int8 tensor cores, prompt-pass fusions, training step outside the products, decoding
  attention at short contexts, cached CUDA modules, prompt-pass graphs) are all still open.

## 5. Suggested order (before the CUDA-first decision; see the top for the order now)

1. Correctness: 1.1 (Vulkan CTC), 1.2 (GGUF fallback), 1.3 (card names in code), 1.7.
2. Measure before refactoring: the owner's GPU runs of section 4 (recurrent on CUDA, the OCR profile, the
   regression benchmarks). Their numbers decide what is worth refactoring first.
3. Library contracts the samples need (section 2, gaps): `MeasuredBatches`, a profiling hook on any device, `noise`
   in `Augmentations`, a source path on `ChatImage`.
4. Invalidation hook for buffers (GraphModule constants, BatchNorm eval, folded convolution) and `ActivationMemory`
   spans: one public-API change, several speedups.
5. Tuning: size classes for the remaining data-sized keys (CUDA packed/decode, Vulkan rows); convolution candidacy
   independent of free memory; cheaper measuring.
6. Finish the audit of the six unfinished areas (section 3), hottest first (R10, R7, R4, R6), then the analyzers.
7. CPU kernel items of R3 that change bits (deterministic `GroupMoments`, scalar fallbacks, fused depthwise and
   activations), each with before/after numbers and the owner's agreement that the new bits are the reference.
8. Vulkan: fused recurrent kernels (1.6) when Vulkan is back in scope.
