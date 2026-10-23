# What is left

Every plan's status and what is left in it, grouped by plan (2026-10-22). A finished plan is in [done/](done/) (its
unfinished parts are listed here, so the plan itself is closed); a plan not started is in [opened/](opened/). Each item
names its kind (bug, speed, memory, feature, check, question, docs) and the section of the plan it came from. The owner's
test runs stay in the plans (each section below says where); they are not listed as items. The text that was moved out of
the plans is kept, as it was written, at the end ("The moved text").

| Plan | Status | Where it stands | Left |
|---|---|---|---|
| [Plan 1: NVIDIA](done/1-nvidia.md) | Done | In progress | 31 |
| [Plan 2: CPU](done/2-cpu.md) | Done | In progress | 17 |
| [Plan 3: AMD](done/3-amd.md) | Done | Partly built | 2 |
| [Plan 4: Intel](done/4-intel.md) | Done | Partly built | 7 |
| [Plan 5: Apple](opened/5-apple.md) | Not started | Planned, nothing built | 6 |
| [Plan 6: Idrak.Network](done/6-network.md) | Done | Superseded by Plan 23 | 0 |
| [Plan 7: Backends](done/7-backends.md) | Done | Done, with leftovers | 11 |
| [Plan 8: HIP](done/8-hip.md) | Done | Partly built | 9 |
| [Plan 9: Operations as data](done/9-operations.md) | Done | Done, with leftovers | 4 |
| [Plan 10: Idrak.Abstraction](done/10-abstraction.md) | Done | Done, with leftovers | 6 |
| [Plan 11: Images into language models](done/11-vision-language.md) | Done | Done, with leftovers | 8 |
| [Plan 12: Fine-tuning vision-language models](done/12-vision-tuning.md) | Done | Done, with leftovers | 7 |
| [Plan 13: Idrak.Vision](done/13-vision.md) | Done | Done, with leftovers | 27 |
| Plan 14: The refactor backlog | Removed | In progress | 7 |
| [Plan 15: The Arabic legal OCR app](done/15-legal-ocr-app.md) | Done | Done | 0 |
| [Plan 16: One contract for every vision-language family; Qwen-VL](done/16-qwen-vl.md) | Done | Done, with leftovers | 2 |
| [Plan 17: Speed and memory for vision and OCR](done/17-vision-ocr-speed.md) | Done | Done, with leftovers | 12 |
| [Plan 18: Synthetic document pages](done/18-synthetic-pages.md) | Done | Done, with leftovers | 2 |
| [Plan 19: language and vision-language models from scratch](done/19-vision-language-from-scratch.md) | Done | Done, with leftovers | 5 |
| [Plan 20: Idrak.Diffusion (image generation by denoising)](opened/20-diffusion.md) | Not started | Planned, nothing built | 19 |
| [Plan 21: Idrak.Audio (sound in and out)](done/21-audio.md) | Done | Done, with leftovers | 12 |
| [Plan 22: speed and memory everywhere, and the library owns what is generic](done/22-performance-and-library.md) | Done | Done, with leftovers | 5 |
| [Plan 23: Idrak.Cluster and Idrak.Network](done/23-cluster.md) | Done | Done, with leftovers | 13 |
| [Plan 24: Unigram (SentencePiece) and WordPiece tokenizers, built in](done/24-tokenizers.md) | Done | Done, with leftovers | 15 |
| Plan 25: Idrak.Extensions.AI (no file) | Dropped | Dropped | 0 |
| [Plan 26: model families as plug-in packages, and `Idrak.Nlp` renamed `Idrak.Text`](done/26-plugin-packages.md) | Done | Done, with leftovers | 38 |
| [Plan 27: reinforcement learning](done/27-reinforcement.md) | Done | Done, with leftovers | 13 |
| [Plan 28: documents](opened/28-documents.md) | Not started | Planned, nothing built | 10 |
| [Plug-in points](done/plug-in.md) | Done | Partly built | 26 |
| [Plug-in 1: Idrak.HarfBuzz](opened/plugins/1-harfbuzz.md) | Not started | Planned, nothing built | 4 |
| [Plug-in plans index](opened/plugins/README.md) | Not started | Planned, nothing built | 0 |
| [idrak-cli: one command-line tool](done/idrak-cli.md) | Done | Done, with leftovers | 25 |
| Open questions and checks | Removed | In progress | 0 |
| [GPU checks for plans 12 and 13](done/gpu-checks.md) | Done | In progress | 0 |

## Plan 1: NVIDIA ([1-nvidia.md](done/1-nvidia.md))

**Status:** Done. **Where it stands:** In progress — supported on every GPU from compute 5.0; the few-row rule and every card-dependent choice are measured per shape since 0.1.6 (step 1 done); the performance table of section 2 and the coverage gaps are open.
**Owner's runs:** pending: section 1's 3060 and Ada `--bench-gemv` runs (also README, "Open"); the regression check below.

| # | Leftover | Kind | From |
|---|---|---|---|
| 1.1 | `--bench-gemv` on an RTX 3060 and an Ada (compute 8.9) card: "auto" within 5% of the best forced column at every row (decoding, prompt-sized and attention rows), including the 9-63-row packed path, the k splits on 30 SMs (gate/up 84 µs auto against 65 µs with 2 splits) and the decoding-attention split. | check | Section 1; README, "Status of each rule" and "Open" |
| 1.2 | Delayed FP8 scaling keyed by call order, not address (~18 ms per 1.25B training step; performance-notes step 1). | speed | Section 2, item 1 |
| 1.3 | FP8 quantization fused into its producers (layer norm, GELU, product epilogues): up to ~100 ms per 1.25B step; needs 1.2. | speed | Section 2, item 2 |
| 1.4 | Persistent decoding kernel (several layers' GEMVs in one launch); decoding is at ~31% of the bandwidth ceiling. | speed | Section 2, item 3 |
| 1.5 | Int8 prompt products on int8 tensor cores; needs a second, k-major copy of the int8 weights. | speed | Section 2, item 4 |
| 1.6 | Cached compiled kernels (a cubin per driver and GPU through `cuModuleLoadData`): ~70 ms off the first prompt pass. Today every module is JIT-compiled from PTX (`CudaBackend.cs:461`). | speed | Section 2, item 5 |
| 1.7 | Memory pool size classes and partial trimming (fewer full flushes near a limit); the pool keeps exact-size blocks with a 25% best fit. Same suspect as 13.23 (the OCR first epoch). | memory | Section 2, item 6; Plan 14 backlog 12c |
| 1.8 | Fine-tuning loss read once per optimizer step: `FineTuning.cs:1939` reads every micro-batch's loss (a host wait each). | speed | Section 2, item 7 |
| 1.9 | QLoRA backward without expanding int4/int8 weights (4-8x less weight traffic per step); needs packed transposed kernels. | speed | Section 2, item 8 |
| 1.10 | Prompt-pass fusions: the gated activation and the residual + norm in the products' epilogues, the attention output written in the o product's layout (180-row pass 7.7 ms). | speed | Plan 14 backlog, section 4 (docs/performance-notes.md step 5) |
| 1.11 | Training step outside the products (1.25B, FP8): adam8 34 ms, flash backward 32 ms, sumsq 9 ms, zeroing 8 ms, dropout 10 ms; first gradient write with beta 0, clipping's norm fused into the backward. | speed | Plan 14 backlog, section 4 (performance-notes step 6) |
| 1.12 | Decoding attention at short contexts: a length-aware split (~1 µs at 200 positions) that keeps recorded steps valid. | speed | Plan 14 backlog, section 4 (performance-notes step 7) |
| 1.13 | Prompt pass recorded as a graph per prompt length, or fewer launches (7.7 ms wall against 7.3 ms GPU, 283 launches). | speed | Plan 14 backlog, section 4 (performance-notes step 9) |
| 1.14 | Compute 5.x and 6.x (Maxwell, Pascal) never run; Turing (7.5) passed on a Quadro RTX 3000. | check | Section 3; README, "Not tested yet" |
| 1.15 | A GPU run on native Linux drivers (WSL2 only so far; a live USB or a cloud GPU). | check | Section 3; README, "Not tested yet"; open-questions decision 7 |
| 1.16 | CI does not run on a GPU: a self-hosted runner or a manual "GPU check" step in the release checklist. | check | Section 3 |
| 1.17 | An FP16 `MatMulPrecision` for compute 7.0-7.5 (no bfloat16 there); `MatMulPrecision` has Float32, BFloat16, Float8 only. | feature | Section 3 |
| 1.18 | Kernel-shape candidates, each to be measured on a GPU that has the feature: 8-bit products with 4 cp.async stages, bfloat16 cp.async pipelines, `wgmma`/TMA on 9.0, `tcgen05` on 10.x, block-scaled FP8/FP6/FP4 on 12.x, a smaller tensor tile for few-SM GPUs, a per-kernel register check at load, an L2 persistence window for the KV cache. | speed | README, "Kernel shapes" (candidates) |
| 1.19 | Async upload size (4 MiB) and the 16 MiB staging ring: confirm on PCIe 3/4/5; and the 64/128 tiles on a GPU without tensor cores. | check | README, "Status of each rule" |
| 1.20 | Regression check against the library's starting numbers: `--bench-gemm` (78 TFLOPS), `--bench-gemv`, the Chat sample's `profile Qwen/Qwen3-0.6B --cuda --int8 --kv16` (474-493 tok/s, 177 launches a token, 7.7 ms 180-token prompt). | check | Plan 14 backlog (CUDA order 2; section 4) |
| 1.21 | Rule-80 change to the int8/int4/bf16 GEMVs, decoding attention and flash PV loops (11a) is done; verify on CUDA with `--bench-gemv` and the Chat sample's `profile ... --cuda --int8`. | check | Plan 14 backlog 11a |
| 1.22 | Size classes for the CUDA keys still keyed by exact data sizes: PackedSplits, PackedTile, PackedMultiSplits, GatedActivation (`m`), DecodeSplits and DecodeMinChunk (`capacity`) (`CudaBackend.Quantized.cs:400-963`). | speed | Plan 14 backlog (CUDA order 3; section 2, Tuning) |
| 1.23 | Cheaper measuring on CUDA (shorter warm-up once busy, fewer rounds for far-apart candidates) or measuring off the step's critical path: the first OCR steps took 914 ms with measuring against 67 ms without. | speed | Plan 14 backlog (CUDA order 5, 10j) |
| 1.24 | `gemv_nn_f32` / `gemv_multi_f32` split k only inside one block: a narrow product runs on ⌈n/32⌉ blocks however long k is; split k across blocks with partials added in split order (rule 81). | speed | Plan 14 backlog 11b |
| 1.25 | `gemv_nt_f32`: one warp per column walks all of k (grid x = ⌈n/8⌉); k chunks across blocks with fixed-order partials (rule 81). | speed | Plan 14 backlog 11c |
| 1.26 | `matmul_f32`: transposed operands are read uncoalesced and 16 × 16 tiles give few blocks on small products (16×128×512 at 0.1 TFLOPS); swap thread roles for transposed loads, route small-m products to k-split kernels (rules 81, 83). | speed | Plan 14 backlog 11d |
| 1.27 | `sum_rows_f32`: one thread per column walks every row; chunks of rows along grid y with partials in chunk order (rule 81). | speed | Plan 14 backlog 11h |
| 1.28 | Rule 80 left in epilogues: `gemm128/64_*` beta reads, the dK/dV write-back of `attn_bwd_kv_f32` and `attn_spans_bwd_kv_f32`, the span kernels' range prologue (fix when the file is next touched). | speed | Plan 14 backlog 11i |
| 1.29 | `PtxKernels.Advanced.cs`, `.Convolution.cs`, `.Recurrent.cs` still interpolate numbers with the current culture (rule 49). | bug | Plan 14 backlog 11j |
| 1.30 | R1 leftovers: `CtcMeta` allocates `new float[3·batch]` per CTC call (`CudaBackend.Ctc.cs:123`, Vulkan's too); prompt-sized packed products make a closure, delegate and small copy per call; tensor-core callers' `Run` delegates allocate twice per large product. | memory | Plan 14 backlog (CUDA order 7; section 2, CUDA) |

**Dropped as done:** 1.31 (card names reworded as what was measured in the CUDA and Vulkan code comments, MixedPrecision's docs and the tuning and mixed-precision tests, 2026-10-23; the device-name guards in the Vulkan and HIP tests keep their lists); plan 14's 11e (`sum_f32` adds per-block partials in block order, `PtxKernels.cs:665-669`), 11f (`sumsq`, `multi_sumsq`, `sum_cols_strided`, `layernorm_bwd_params` store partials folded in a fixed order, `PtxKernels.Rows.cs:599-600`, `PtxKernels.Decoding.cs:110-112`), 11g (deterministic tensor-core split k, plan 17), 10a, 10c, 10d-10i (marked done and passing on CUDA in plan 14), 1.5 and CUDA order 1 (the fused recurrent kernels ran on CUDA, plan 14 10g).

## Plan 2: CPU ([2-cpu.md](done/2-cpu.md))

**Status:** Done. **Where it stands:** In progress — supported everywhere .NET 10 runs; the machine-agnostic audit (section 4) is done; step 1 (decided: option a) is not built and most of section 2 is open.
**Owner's runs:** none.

| # | Leftover | Kind | From |
|---|---|---|---|
| 2.1 | Plain sampling: entropy from the scores already in hand (one `Log2` per row) instead of a `Log2` per token; target 12 → ~6 ms on the 8-thread machine over 151,936 tokens. Not built: `CpuBackend.Decoding.cs:494` still calls `MathF.Log2` per token. | speed | Section 1 |
| 2.2 | Products with a transposed B of 17-63 rows transpose all of B per call (only ≤ 16 rows read it in place): 3x at 64 rows (measured 0.1.4). | speed | Section 2, item 2 |
| 2.3 | RMS norm, rotary embedding, gated activation and add + norm are scalar per row (`CpuBackend.Quantized.cs`). | speed | Section 2, item 4 |
| 2.4 | Language-model loss (`SoftmaxCrossEntropyRows`): double-precision `Exp`, scalar, rows with weight 0 still computed. | speed | Section 2, item 5 |
| 2.5 | Memory pool: exact-size buckets, never trimmed without a limit, one lock (`CpuBackend.cs:30-150`). | memory | Section 2, item 6 |
| 2.6 | `--bench-cpu` on an ARM64 machine for the NEON 8×8 kernel's speed (the tests passed on an Apple M4 Max and an Adreno phone's CPU). | check | Section 3; section 4 "Open"; README, rule table |
| 2.7 | A run on a desktop or Xeon where the runtime accelerates AVX-512 (the 16-wide paths and 512-bit panels; only the test container's second host ran them). | check | Section 3; section 4 "Open" |
| 2.8 | The bfloat16 k chunk is a formula (8 fastest of 2-16 here): measure it at first use if other machines disagree. | speed | Section 4 "Open"; README, rule table |
| 2.9 | Register-tile the int8, int4 and float few-row products as the bfloat16 one (identical results). | speed | Section 4, "ISA paths not added" |
| 2.10 | Opt-in ISA paths (each changes a summation order or rounding): 512-bit transposed-B dots, AVX-512 BF16 / ARM BF16 dots, VNNI / SDOT int8 dots, SVE, AMX, 512-bit element-wise kernels where `Vector<T>` is 256. | speed | Section 4, "ISA paths not added" |
| 2.11 | Lifting the CPU's attention head-size limits (256 / 128): changes results for larger heads, so opt-in only. | feature | README, rule table |
| 2.12 | `LayerNormTrainKernel` computes mean and variance twice (it calls `LayerNormFused`); a fused training layer norm needs an override-aware operation. | speed | Plan 14 backlog (R3) |
| 2.13 | `GroupMoments` merges chunk sums under a lock in completion order: run-to-run bit differences with several threads (`CpuBackend.Advanced.cs:331-376`). | bug | Plan 14 backlog (R3) |
| 2.14 | Vector loops without an `IsHardwareAccelerated` check (rule 34): `Moments`, `MultiplyAddInPlace`, `GroupAffineCore`, `AddInPlace` and loops in `CpuBackend.cs` (a scalar fallback changes bits). | bug | Plan 14 backlog (R3) |
| 2.15 | Depthwise convolution is scalar with its activation as a separate pass; the composed convolution's bias and activation passes are not fused (bits change). | speed | Plan 14 backlog (R3) |
| 2.16 | `Pow` and the gated-activation backward are scalar `MathF` loops (sin, cos, sqrt, SiLU are vector loops since plan 22 step 8); the bfloat16 tiled kernel reads leftover columns one by one. | speed | Plan 14 backlog (R3) |
| 2.17 | Small per-call arrays: norm scale/shift copies, `Enumerable.Repeat` in `GroupScaleShift`, CTC per-sequence tables, `Permute`'s shape copies. | memory | Plan 14 backlog (R3) |

**Dropped as done:** section 2 item 3 (`FusedAdamW` runs SIMD on all cores, `CpuBackend.cs:312-333`); section 3's macOS row (an Apple M4 Max passed the whole CPU list, README "Not tested yet").

## Plan 3: AMD ([3-amd.md](done/3-amd.md))

**Status:** Done. **Where it stands:** Partly built — the Vulkan backend this plan chose (option A) exists for every GPU through plan 7 (device and memory, SPIR-V kernels, language models, cooperative matrices, `--bench-vulkan`, recorded steps, training kernels); none of it has run on an AMD GPU, so no phase's "Done when" is met. Option B became plan 8 (HIP).
**Owner's runs:** none written; the runs below need AMD hardware.

| # | Leftover | Kind | From |
|---|---|---|---|
| 3.1 | Every phase's check on AMD hardware: memory tests on an AMD GPU; every CPU test passing on it; Qwen3-0.6B chatting with the CPU's greedy tokens and `check qwen3.json`; the LoRA tests and a short Qwen2.5-0.5B LoRA run on RDNA 3. On RX 7600/7800 XT, RX 6600/6700, a Ryzen 780M and an RX 9070, under Adrenalin and RADV. | check | Phases 1-5; "Hardware to test on"; README, "Not tested yet" |
| 3.2 | To verify: which Adrenalin and RADV versions expose `VK_KHR_cooperative_matrix` with bfloat16 or FP16 inputs on RDNA 3/4, and whether RDNA 2 (no matrix units) is fast enough with FP32 products to enable. | question | "The choice" (to verify early) |

**Dropped as done:** option B as an `Idrak.Rocm` package (superseded by plan 8's HIP backend in `Idrak.Gpu`).

## Plan 4: Intel ([4-intel.md](done/4-intel.md))

**Status:** Done. **Where it stands:** Partly built — the shared Vulkan backend exists (plan 7, mapped memory where the device memory is host-visible) and ran `--bench-vulkan` on an Intel UHD Graphics laptop GPU; no Iris Xe or Arc run, and the NPU part (Part 2) is not started.
**Owner's runs:** none written.

| # | Leftover | Kind | From |
|---|---|---|---|
| 4.1 | Phase 1: the full test list on an Iris Xe laptop and an Arc card, with the subgroup size set per kernel (8, 16 or 32 lanes). | check | Part 1, phase 1 |
| 4.2 | Phase 2: integrated-GPU memory: the memory limit counting the shared RAM, and an upload of a model's weights with no second copy in RAM (not confirmed). | memory | Part 1, phase 2 |
| 4.3 | Phase 3: few-row products and decoding attention tuned for Intel's execution units, XMX prompt products where available; Qwen3-0.6B chatting on Arc and Iris Xe with `--bench-vulkan` numbers in the README. | check | Part 1, phase 3 |
| 4.4 | Phase 4: LoRA fine-tuning on a discrete Arc card. | check | Part 1, phase 4 |
| 4.5 | To verify: which Intel drivers expose cooperative matrices (XMX) in Vulkan on Arc and Xe-LPG/Xe2, with which formats (FP16, bfloat16, int8). | question | Part 1, "The choice" (to verify early) |
| 4.6 | Level Zero (option B), later if needed: unified shared memory with zero copy on integrated GPUs. | feature | Part 1, "The choice" |
| 4.7 | NPU, not started: a one-week spike (an exported ONNX classifier through OpenVINO, latency and watts against the CPU), then `Idrak.OpenVino` (`OpenVinoModule`), embeddings on the NPU, small language models only if OpenVINO's NPU decoders hold up. | feature | Part 2, phases 1-4; README, "Not supported yet" (NPUs) |

## Plan 5: Apple ([5-apple.md](opened/5-apple.md))

**Status:** Not started. **Where it stands:** Planned, nothing built — no Metal backend (`Device.Get("metal")` does not exist); the CPU list passed on an Apple M4 Max (README).
**Owner's runs:** none.

| # | Leftover | Kind | From |
|---|---|---|---|
| 5.1 | Phase 1: Metal device and memory through the Objective-C runtime (`libobjc`, `objc_msgSend`), shared-storage buffers (unified memory), `DeviceType.Metal`, `Device.Default` Metal then CPU on a Mac. | feature | Phase 1 |
| 5.2 | Phase 2: a C# Metal Shading Language writer and the minimum kernel set (Metal Performance Shaders for products first), many operations per command buffer. | feature | Phase 2 |
| 5.3 | Phase 3: language models (few-row products, KV caches, sampler, attention, `simdgroup_matrix` products); Qwen3-0.6B with the CPU's greedy tokens. | feature | Phase 3 |
| 5.4 | Phase 4: `--bench-metal`, backward kernels, AdamW and 8-bit AdamW, LoRA; a short Qwen2.5-0.5B LoRA run. | feature | Phase 4 |
| 5.5 | Phase 5: `Idrak.CoreML` add-on for the Neural Engine (a one-week trial first). | feature | Phase 5; README, "Not supported yet" (NPUs) |
| 5.6 | To verify: Metal from .NET through the Objective-C runtime without a binding library, and which bfloat16 support each Apple GPU family has (M1/M2 FP16 only). | question | "The choice" (to verify early) |

**Dropped as done:** the Mac test run that fills the CPU plan's macOS gap (README: an M4 Max passed the whole list on the CPU).

## Plan 6: Idrak.Network ([6-network.md](done/6-network.md))

**Status:** Done. **Where it stands:** Superseded by Plan 23 — superseded 2026-10-18; nothing was built from this file; plan 23 restates what it reuses (the fixed summation order, the wire header, the HMAC hello, a rank's share of the draw).
**Owner's runs:** none (its proof run is plan 23's).

No leftovers.

## Plan 7: Backends ([7-backends.md](done/7-backends.md))

**Status:** Done. **Where it stands:** Done, with leftovers — all six phases built: the minimum backend and providers, public KV formats, the Vulkan runtime and kernels (fused decoding kernels and subgroup size control included), the public backend API (plan 10). The file's status line ("in progress on `architecture`") is stale.
**Owner's runs:** none in this file.

| # | Leftover | Kind | From |
|---|---|---|---|
| 7.1 | Vulkan recorded steps used wherever one can be recorded, even where slower (Quadro RTX 3000: 639 tok/s with graphs, 779 without): time replays against direct steps when a graph is first recorded and keep the faster per device, driver and power source; today `SupportsGraphs` reads `IDRAK_VULKAN_GRAPHS` only (`VulkanBackend.Graphs.cs:68`, also read per recorded graph). | speed | README, "Future improvements"; Plan 14 backlog (R2) |
| 7.2 | SPIR-V memory model for the cooperative-matrix kernels: SPIRV-Tools 2025.2 rejects cooperative matrices without the Vulkan memory model; move the matrix kernels to it (recommended, after the RTX checks) or wait for a driver to refuse them. | question | open-questions, decision 4 |
| 7.3 | Operations still on the host fallback on Vulkan: the class-match accuracy count, the gated activations on bfloat16-packed activations (forward and backward), `MaxPoolBackward(count)` and max pooling's gradient when the padding below or right is at least the window. | speed | README, "Vulkan (works, with limits)"; plan 13 step 1 |
| 7.4 | No int8 or float8 cooperative-matrix types, and no attention or fused training kernels on matrix units. | speed | README, "Vulkan (works, with limits)" |
| 7.5 | Tensors larger than the device's `maxStorageBufferRange` take the host fallback (a 151,936 × 4096 float weight on a 1 GiB-range GPU). | memory | README, "Vulkan (works, with limits)" |
| 7.6 | Every Vulkan tuning key with rows is still exact (Splits, Rows, MatMul, MixedMatMul, TiledAttention, Attention, SpanAttention, PackedPrompt): decide per key which sizes follow the data. | speed | Plan 14 backlog (section 2, Tuning) |
| 7.7 | `ReductionFlags`, `MatCandidates`, `PowersOfTwo`, `GemvCandidates` allocate while measuring. | memory | Plan 14 backlog (R2) |
| 7.8 | A shared kernel description: simple operations (element-wise, reductions, gathers) generated as PTX and SPIR-V from one description, so only products and attention are written twice. | feature | README, "Shared work before any new backend", step 3 |
| 7.9 | Single-pass bfloat16 matrix units under `MixedPrecision` on a real Vulkan GPU: correctness and speed against float32. | check | open-questions, hardware check 1 |
| 7.10 | The windowed and soft-capped kernels on a real Vulkan GPU, and their speed (`--bench-vulkan window`). | check | open-questions, hardware check 10 |
| 7.11 | Android: the width probe and the chat after the merged kernels on the phone; the Honor tablet (CPU, and Vulkan if a driver is reachable from Termux); Qualcomm's own driver, Mali GPUs and a .NET Android app. | check | open-questions, hardware checks 5 and 8; README, "Not tested yet" |
| 7.12 | Packed-sequence adapter gradients wrong (relative error 0.438, "fine-tuning: packed sequences … rotary positions") on an NVIDIA card when kernel choices are measured fresh in a warm process: `IDRAK_VULKAN_TUNING_CACHE=0`, filter `sampler:\|packed projections\|packed epilogues\|pretrained: packed\|fine-tuning: packed sequences` fails; the test alone, with stored choices, or with the run's captured choices and `IDRAK_AUTOTUNE=0` passes, so no stored choice is wrong: measuring during the rotary loop (the learned loop reuses its keys and passes) corrupts its gradients. Every operation measured alone (`IDRAK_VULKAN_TUNE_OPS=<op>`) passes, so it takes two or more; next: all but one (`IDRAK_VULKAN_TUNE_OPS` with one left out; the 2026-10-23 script skipped this step because PowerShell's `-match 'FAIL'` matched "0 failed"), then read the pair's measurement (scratch bindings, what it writes outside the scratch, queue order around `Synchronize`). Lavapipe never reproduces it, even with `IDRAK_TUNE_BUDGET_MS` large enough to measure everything. | bug | Plan 7 Vulkan, 2026-10-23 |
| 7.13 | Fresh choices per process differ across a cluster: with no tuning file shared yet, the one-process reference and each rank measure separately and can settle differently on the order-different choices (reduced-precision products, attention widths), so "cli reinforce: cluster sim … against one process" misses 1e-4 on a fresh card run (stored choices pass). Option offered, not decided: ranks take only same-bits choices of their own, or rank 0's choices before the first step. | question | Plan 7 Vulkan, 2026-10-23 |
| 7.14 | Depthwise convolution in the one-order convolution (TuningVersion 4) runs 1.1-1.3× slower than before; proposed: its own FMA-chain order for depthwise shapes, the composed path dropped there. Not decided. | speed | Plan 7 Vulkan, 2026-10-23 |

**Dropped as done:** phase 4's "fewer dispatches per layer" (fused decoding kernels, `VulkanBackend.Fused.cs`, `FusedKernels = true`; the README's "No fused decoder kernels" row is stale); phase 5's required subgroup sizes (`VulkanBackend.SubgroupSize.cs`) and cooperative matrices; the sampler's subgroup reductions (taken out on purpose: workgroup memory on every device, plan 22).

## Plan 8: HIP ([8-hip.md](done/8-hip.md))

**Status:** Done. **Where it stands:** Partly built — the first slice (runtime, devices, minimum backend, hipRTC kernels for element-wise ops, RMS norm and the int8 product, a code-object cache) is in `Idrak.Gpu/Hip`; never run on an AMD GPU.
**Owner's runs:** pending: "What remains" item 1 (needs an AMD GPU).

| # | Leftover | Kind | From |
|---|---|---|---|
| 8.1 | A run on real hardware (ROCm on Linux, the HIP SDK on Windows; an RDNA and a CDNA GPU if possible): `--list-devices`, the HIP tests, the whole list on `hip:0`; fix what it finds first. | check | What remains 1; open-questions, hardware check 4; README, "Not supported yet" |
| 8.2 | More kernels, most used first: softmax and log-softmax, float32 products, the sampler and penalties, rotary positions, gathers, decoding attention, cache writes, int4 and bfloat16 products, the fused decoding kernels. | speed | What remains 2 |
| 8.3 | Tuning by measurement (block sizes, k splits, attention splits, kept per device, driver and build); wavefront reductions measured against shared memory. | speed | What remains 3 |
| 8.4 | Graphs through hipGraph (stream capture) for `GenerationOptions.UseGraph`. | speed | What remains 4 |
| 8.5 | Matrix cores: WMMA on RDNA 3/4, MFMA on CDNA, chosen by a test compile, never by target name. | speed | What remains 5 |
| 8.6 | Training kernels: backward passes, AdamW (fused and 8-bit), convolution, norms' gradients. | feature | What remains 6 |
| 8.7 | Memory: asynchronous uploads from pinned staging (`hipHostMalloc`), offloading to system memory, `IMemoryOffload`. | memory | What remains 7 |
| 8.8 | Backend API points a third family found, still open: `TensorCoresUnavailable()` and the capture members (`BeginCapture`/`EndCapture` returning `IntPtr`s) are CUDA-shaped; no per-thread device-context hook; `Upload` must finish reading its span (no asynchronous upload from backend-owned memory beyond the optional `IHostStaging`); no shared-memory figure in the capabilities; host fallbacks download operands whole; integers in float storages unnamed; no shutdown hook; `Return` on the finalizer thread undocumented. | feature | Input for the public backend API |
| 8.9 | `HipBackend` launch: a lock and two dictionary lookups per launch. | speed | Plan 14 backlog (R2) |

**Dropped as done:** `Detach`/`Attach` (now `protected abstract`, `Backend.cs:748-754`); compute units and lanes (`BackendHardware.ComputeUnits` and its lanes field).

## Plan 9: Operations as data ([9-operations.md](done/9-operations.md))

**Status:** Done. **Where it stands:** Done, with leftovers — phases 1, 2, 4 and 5 built (2026-10-06/07, with plan 10); phase 3 not started. The file's status line ("phases 0 and 2 to 5 not started") is stale.
**Owner's runs:** pending: the decoding gate below; CUDA PTX and HIP C++ plug-in kernels never run on a GPU (plan 10, W4.5).

| # | Leftover | Kind | From |
|---|---|---|---|
| 9.1 | Phase 3: backends register kernels directly, one at a time (minimal, HIP, Vulkan, CUDA, CPU); `Backend` shrinks to about 15 members. Today `Backend.cs` still declares 152 `NameKernel` virtual operations. | feature | Phases, row 3 |
| 9.2 | Profiling still lives in each backend (`StartProfile`/`StopProfile`), not in the dispatcher. | feature | Phase 2 (as built) |
| 9.3 | The decoding gate (within 2% of the previous phase) measured on a quiet machine and on the GPUs (a 4-core shared CPU's A/A spread was 7%). | check | Phase 1 as built; plan 10 phases 1b and 4 |
| 9.4 | After phase 5, a separate plan: a lazy graph built on the descriptors (trace a step, fuse element-wise chains, plan memory, replay on any device), unifying CUDA graphs, Vulkan's recorded steps and ONNX export. | feature | Phase 1 as built (closing paragraph) |

**Dropped as done:** phase 0 (the operation catalogue is generated and test-guarded in `10-abstraction-inventory.md`, its operations table by device); open questions 1-3 (answered by phase 1 as built: kernels take the arguments directly, composed forms are the `Backend` defaults written once; the names decided by plan 10).

## Plan 10: Idrak.Abstraction ([10-abstraction.md](done/10-abstraction.md))

**Status:** Done. **Where it stands:** Done, with leftovers — phases 0-8 and wave 4 done on the CPU and Vulkan (2026-10-07/08); CUDA has run much of it since (plans 17, 23). The version bump (W4.7, 0.3.1 → 0.4.0) is left for the owner's release (CLAUDE.md: agents do not bump it).
**Owner's runs:** pending: "On the author's machines after W4.7" (HIP and the 2% decoding gate never run; the CUDA-only kernels against the kit's references).

| # | Leftover | Kind | From |
|---|---|---|---|
| 10.1 | MCP tools `transcribe` and `image` through the model-kind contract (today `generate`, `chat`, `embed`, `McpModelTools.cs:16-22`). | feature | Phase 8b (as built), "Later" |
| 10.2 | `Conversation`'s tool loop on top of `ChatTools.WithTools` (it runs its own loop, `Conversation.cs:122`). | feature | Phase 8b (as built), "Later" |
| 10.3 | A name-to-factory registry of engine model kinds, once a package records its kind. | feature | Phase 3c (as built) |
| 10.4 | Host retry (`IDRAK_RETRY_ON_HOST`) for plug-in kernels. | feature | Phase 6 (as built), "Not done" |
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
| 11.6 | The Vulkan span kernel's speed on a real GPU (on lavapipe about 7x slower than composed), tuned there by measurement if it lags. | speed | 3a as built |
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
| 12.5 | CUDA packs image sequences only under a non-causal image rule whose head size the span kernels take (Gemma 3, plan 17 F6); causal-rule families (LLaVA) and head sizes the flash kernels lack still pad. | speed | Plan 14 backlog 10b |
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
| 13.6 | Implicit convolution products on matrix units (cooperative matrices, `mma.sync`). | speed | Step 1 (as built), "Left" |
| 13.7 | A CPU split of the convolution weight gradient's long sum ([filters, positions] × [positions, patch]). | speed | Step 1 (as built), "Left" |
| 13.8 | A fused group norm (statistics and affine in one pass). | speed | Steps 1 and 2 (as built), "Left" |
| 13.9 | CTC labels are clamped on the device, not checked (the CPU checks them). | bug | Step 1 (as built), "Left" |
| 13.10 | bfloat16 convolution weights (a convolution product reading bfloat16); convolutions and norms hold float32 copies. | memory | Step 6 (as built), "Left" |
| 13.11 | A safetensors tensor read straight into device storage for convolutions and float32 layers (plan 17 G1 did it for bfloat16 projections only). | memory | Step 6 (as built), "Left" |
| 13.12 | An ONNX import that streams initializers instead of reading the whole file. | memory | Step 6 (as built), "Left" |
| 13.13 | Device kernels for the box and focal losses (host fallback on CUDA, Vulkan and HIP; none in `src/Idrak.Gpu`). | speed | Steps 4 and 5 (as built), "Left" |
| 13.14 | Augmenting on the device. | speed | Step 5 (as built), "Left" |
| 13.15 | OCR sample: touching lines stay one band. | feature | The OCR sample, as built; Plan 14 backlog (gaps) |
| 13.16 | The measured batching `idrak predict` uses (`MeasuredBatches`, internal in `ImageModelSupport.cs:188`) as a library contract (the sample copies it). | feature | The OCR sample, "Library gaps found"; Plan 14 backlog (gaps) |
| 13.17 | `ChatImage` carries no source path, so pages of a data file are named by record. | feature | The OCR sample, "Library gaps found"; Plan 14 backlog (gaps) |
| 13.18 | A per-layer training-step profile as a library hook (the CLI's `ProfileCommand` recorder is internal; the library's `Profiler` gives operations and kernels on any device since plan 17 H3b). | feature | Plan 14 backlog (CUDA order 10; gaps) |
| 13.19 | Vulkan CTC loss is -∞ on the first step: 4 of the ArabicOcr sample's 13 tests fail on Vulkan (CPU and CUDA pass); not confirmed fixed. | bug | Plan 14 backlog 1.1 |
| 13.20 | The ArabicOcr sample writes a non-finite loss into `recognizer.json` and crashes (`TrainCommand.cs:346`); stop with a clear message. | bug | Plan 14 backlog 1.7 |
| 13.21 | Vulkan has no fused LSTM/GRU cell kernels (~13 launches a step). | speed | Plan 14 backlog 1.6 |
| 13.22 | The recurrent loop is 33% of the warm OCR step's GPU time: keep each block's units of U (and Uᵀ) in shared memory where the device's opt-in room allows (128 KB at H = 256 per 32 units). | speed | Plan 14 backlog 12a (and 10g, "Not done") |
| 13.23 | Convolutions are ~22% of the warm OCR step's GPU time (`conv_bwd_input` 1.9, `conv_fwd` 1.6, `conv_bwd_weight` 1.5 ms): profile each against rules 80, 81, 83. | speed | Plan 14 backlog 12b |
| 13.24 | The first OCR epoch is cold (about 270 s against 150 s warm): measure allocations and tunings a step; suspect the pool's exact-size blocks (see 1.7). | memory | Plan 14 backlog 12c |
| 13.25 | Held-out CER moves between epochs at a learning rate near 0 (2.0-3.0%, best 1.5%): recompute BatchNorm's running statistics over a training pass before evaluating, or evaluate an averaged copy. | bug | Plan 14 backlog 12d |
| 13.26 | Convolution choice per call: the candidate list and the free-memory query run on every convolution (`CudaBackend.Convolution.cs:139`); make candidacy independent of the moment so a known choice is a lookup. | speed | Plan 14 backlog (CUDA order 4; Tuning) |
| 13.27 | `Conv2d.ForwardFused` refolds the batch norm into a weight-sized temporary on every inference call: cache the folded weights (memory) or keep refolding (time)? | question | Plan 14 backlog (R5) |

**Dropped as done:** step 3's CTC device kernels and step 2's dilated windows on devices (step 1); step 2's dedicated depthwise kernel (step 1: CPU direct loops, `conv_depthwise` on Vulkan and CUDA); `NetworkAnalysis` knowing the new steps (step 2, the rest); step 6's `predict`/`train` (step 7) and the stale table (now marked as of planning); the "noise" augmentation gap (`Idrak.Vision/LibraryRegistrations.cs:68`).

## Plan 14: The refactor backlog (removed 2026-10-22: its items are filed under their plans here)

**Status:** Removed. **Where it stands:** In progress — a backlog of other plans' leftovers (plans 12 and 13, the fused recurrent work, the optimization-rules audit R1-R11). The owner's scope decision stands: **CUDA first; anything not related to CUDA is set aside** until the owner says otherwise. Items 10a, 10c-10i, 11a (to verify), 11e-11g are done. Every other item is filed under its plan above (Plans 1, 2, 7, 8, 12, 13, idrak-cli) with From "Plan 14 backlog"; the ones belonging to no plan are below.
**Owner's runs:** pending: section 4 "Checks never run" (filed as 1.20, 1.21, 12 phase 5, `gpu-checks.md`).

| # | Leftover | Kind | From |
|---|---|---|---|
| 14.1 | Card names in benchmark tables and notes: README.md (51), plans/*.md (~55), src/Idrak/README.md (3), docs/performance-notes.md, docs/optimization.md. Owner's decision: a results table names the machine it ran on; advice and defaults must not. | question | 1.4 |
| 14.2 | `ActivationMemory.Release/Compress` take `params Tensor?[]`: an array per layer per step, also at inference; `params ReadOnlySpan<Tensor?>` (public API change). | memory | Section 2, Layers and models (R5); CUDA order 6 |
| 14.3 | `GraphModule.Binary` and `LibraryGraphOps.Scalar` read scalar constants back every forward (a GPU sync per node); BatchNorm in evaluation recomputes its inverse deviation every call. Both need a buffer version or invalidation hook in Idrak.Abstraction. | speed | Section 2, Layers and models (R5); CUDA order 6 |
| 14.4 | GGUF F16 decode is scalar (`Gguf.cs:224`): a vectorized converter (the BCL's vectorized Half conversion is a NuGet package). | speed | Section 2, Layers and models (R5) |
| 14.5 | `Sequential` allocates one closure per whole-model forward. | memory | Section 2, Layers and models (R5) |
| 14.6 | Audit areas not finished (files untouched / total): R4 Abstraction 59/75, R6 core 37/57, R7 Nlp, AspNetCore, Mcp, Onnx.Runtime 32/54, R8 CLI A-M 26/53, R10 Data, Vision, kit 79/84, R11 samples and tools 42/56; hottest first: R10, R7, R4, R6. | check | Section 3 |
| 14.7 | The analyzers of rule 65 (`AnalysisModePerformance` = All) were never turned on; turning them on lists what the review missed (warnings are errors). | check | Section 3 |

**Dropped as done:** 1.2 (an unregistered GGUF pre-tokenizer is refused naming the registry; the library sets no fallback: `GgufTokenizerFormats.cs:37-46`, `GgufPreTokenizers.Fallback`); `ExactGELU` as ~20 passes (one `GeluErf` operation with CPU, CUDA and Vulkan kernels, `Layers/Activations.cs:100-104`); the "noise" augmentation; CUDA order 1 and 1.5 (the fused recurrent kernels pass on CUDA, 10g); 10a, 10c, 10d-10i, 11e, 11f, 11g (see Plan 1).

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
| 17.1 | Vulkan: no `AttentionSpansStrided` (D3, copies the heads), `PackedMatMulEpilogue` (D2), `PackedMatMulNormAddRmsNorm` (D11), `AttentionRows` for pages of different lengths (F3), greedy sampling without top-k (B8), `IMemoryOffload` (E6), chunks and long-lived blocks (E9), a driver free-memory figure (E7). | speed | Left for other devices |
| 17.2 | Vulkan: the read, the CER gate and `--bench-read` never run on a Vulkan device. | check | Left for other devices |
| 17.3 | HIP: everything Vulkan lacks, plus span kernels (it keeps the CPU's limits, A3) and `ResizeNormalize` on the device (C6); never run on a HIP device. | speed | Left for other devices |
| 17.4 | JPEG decoder (C2), not done: MCU-row pipelining (no whole-image coefficients), restart intervals in parallel, h2v1 and h1v2 upsampling in vectors. | speed | C2 |
| 17.5 | C4's rest: `Orient` and the JPEG round trip's colour conversion; `jpeg=95` takes 94-108 ms, of which the optimized encode 70-88. | speed | C4 (part 1 and part 2) |
| 17.6 | `Idrak.AspNetCore` image uploads and `ChatImageDecoder` still give `FromBytes` fresh arrays (two large-object copies a page). | memory | F5, "Seen, not changed" |
| 17.7 | A device-resident tier of the feature cache (needs a device to measure). | speed | F8 |
| 17.8 | The batched path (`StreamBatch`) publishes no prefill and chunk telemetry events. | feature | B6 |
| 17.9 | The batched path makes its own decoding context per batch (no reuse across reads). | speed | B3 |
| 17.10 | The library's own pool trim (after an idle time, or when `Cached` passes a share of the device's memory); B2 was done in the app only. | memory | B2 |
| 17.11 | CPU span attention through the composed path per 64-row block: changes the sums, so a CER-gated option at most. | speed | D9; "Left for other devices" (CPU row) |
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
| 21.2 | The fused STFT kernel (`AudioOperations.Stft` through `PluginOperations`, PTX and SPIR-V, measured per shape against the composed path) is not built; the composed and host paths run. | speed | Phase S2 (the kernel path), decision 2 |
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
| 22.2 | Indexed lookups, not started: `TensorScope` still scans its list backward on every `Keep`, `Owns` and `Untrack` (`LastIndex`, checked); name lookups in the weight readers and tensor stores; kernel and registry lookups per launch. Each to be measured with the benches before and after. | speed | Section 4b |
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
| 23.10 | HIP carving (a fake device on a HIP GPU) is compiled, never run. | check | C1 as built |
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
| 26.34 | Attention with a bias copies the filled cache slots (two extra launches a layer a step, 126 → 131 a token on 6 layers); a product reading the cache prefix in place removes them. | speed | L1-T5 |
| 26.35 | `log_softmax_topk_f32` runs one block per beam, leaving most of a large device idle (rule 81); split the vocabulary across blocks. | speed | L1-T5 |
| 26.36 | HIP and Vulkan have no `LogSoftmaxTopK` kernel (they read the logits back; CUDA only, checked); a beam step does two small readbacks (values, tokens). | speed | L1-T5 |
| 26.37 | The bucketed bias with a sliding-window cache uses window-relative positions (existing). | bug | L1-T5 |
| 26.38 | Agents of two lanes share one scratchpad folder; give each lane its own. | question | L1 |
| 26.39 | The padding scope is not kept open through a checkpointed backward recompute (as `EncoderMemory` today). | feature | P8 as built (Left) |
| 26.40 | A shared position bias that trains is recomputed by every layer; with a bias, a pass over several queries (encoder, training) still builds the full `[batch·heads, t, t]` scores (row 10 fixed decoding steps only). | speed | P8 as built (Measured, not taken) |
| 26.41 | Padded encoder attention passes no mask, so on a device without a per-range kernel for the head size it runs on the host; the training path with dropout builds its padding mask on the host. | speed | P8 as built (Measured, not taken) |
| 26.42 | `Seq2SeqGenerator` reads one token back each step and has no recorded-graph replay. | speed | P8 as built (Measured, not taken) |
| 26.43 | `TextEncoder` pads each batch to `MaxLength`, not its longest text (still so, checked). | speed | P9 as built (Measured, not taken) |
| 26.44 | `PretrainedEncoderModel.Hidden`/`Pooled`/`Logits` leave gradients on when the caller forgets `NoGrad`. | memory | P9 as built (Measured, not taken) |
| 26.45 | Exact-erf GELU is not fused into the first projection; the masked-LM head scores every position. | speed | P9 as built (Measured, not taken) |
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

**Status:** Not started. **Where it stands:** Planned, nothing built — planned 2026-10-22 against 99872b8; no phase started ("No phase starts before the owner names it"); no `IDocumentFormat` in `src/` (checked).
**Owner's runs:** none.

| # | Leftover | Kind | From |
|---|---|---|---|
| 28.1 | Phase D0, not started: the contract and its registries (`IDocumentFormat`, `DocumentFormats`, `IDocumentTextLayout`, `DocumentTextLayouts`), the `lines` and `content` layouts, `DocumentFormatSuite`, a toy format from `Idrak.PluginTests`. | feature | Phase D0 |
| 28.2 | Phase D1, not started: PDF the file (lexer, cross-references and streams, object streams, incremental updates, repair, filters, limits, document information). | feature | Phase D1 |
| 28.3 | Phase D2, not started: PDF encryption (standard handler revisions 2-6: RC4, AES-128, AES-256; empty or given password). | feature | Phase D2 |
| 28.4 | Phase D3, not started: PDF text (pages, the content interpreter, fonts to Unicode, widths, runs with boxes, the `lines` layout). | feature | Phase D3 |
| 28.5 | Phase D4, not started: PDF images and embedded files. | feature | Phase D4 |
| 28.6 | Phase D5, not started: the fixture generator, docs, and the owner's `documents real:` group. | feature | Phase D5 |
| 28.7 | Decision 5: text and images first, no rendering. Confirm, or add rendering. | question | Questions for the owner (1) |
| 28.8 | Decision 6: only the contract and the reader, users of it later. Confirm, or name the first user to add. | question | Questions for the owner (2) |
| 28.9 | Forms and annotations: later (as planned), or with D3? | question | Questions for the owner (3) |
| 28.10 | The Office plug-in's package name (`Idrak.Data.Plugins` proposed) and the variable names in C8. | question | Questions for the owner (4) |

## Plug-in points ([plug-in.md](done/plug-in.md))

**Status:** Done. **Where it stands:** Partly built — the outside plug-in path, items 1, 2 (most), 3, 4 (first slices, windowed kernels, mixture of experts), 5, 6, 8, 9, 10, 12, 14, 16, 26, the dataset loaders, provider-neutral serving and distillation are built; the rest of the ranked gaps is open.
**Owner's runs:** none in this file.

| # | Leftover | Kind | From |
|---|---|---|---|
| PI.1 | Item 2, still missing: tensor-by-tensor division, comparisons, a general `Gather` (only `GatherRows`). | feature | Ranked gaps, item 2 |
| PI.2 | Item 3: LongRoPE (Phi-3/4) as a built-in; GGUF's `yarn_log_multiplier` and `attn_factor` keys. | feature | Done since the survey, item 3 |
| PI.3 | Item 4: an activation registry (the closed `FeedForwardActivation`) and `DecoderNorm`; ALiBi; fused qkv (Phi-3). | feature | Remaining for item 4 |
| PI.4 | Vulkan kernels for packed sequences and per-row starts (both run through the host fallback). | speed | Item 4, windowed kernels; Remaining for item 4 |
| PI.5 | A window (and cap) in CUDA's tensor-core flash kernels, so windowed layers pack on CUDA. | speed | Item 4, windowed kernels; Remaining for item 4 |
| PI.6 | Mixture of experts: device-side routing and grouped expert kernels (one launch for all experts' products, reading only the chosen experts' packed weights; recordable decoding). | speed | Plan: mixture of experts, "Remaining" |
| PI.7 | Mixture of experts on CUDA (composed operations never run there) and a real model's speed (`idrak c Qwen/Qwen1.5-MoE-A2.7B-Chat -w int4`, then `idrak b`). | check | open-questions, hardware check 6 |
| PI.8 | Item 5: decoder layers have no ONNX export translator; graphs and exports have one input and one output. | feature | Ranked gaps, item 5; Gaps inside the existing registries |
| PI.9 | Item 6: ONNX import of Erf (outside GELU), Floor/Ceil, Reciprocal and tensor-by-tensor Div (need tensor operations, PI.1); graph structural operations cannot be replaced. | feature | Ranked gaps, item 6; Gaps inside the existing registries |
| PI.10 | Item 7: model packages store only builder JSON, `DecoderSpec` or `GraphModule`; scalers only Standard and MinMax; tokenizers only character and word (`ModelPackage.cs:112-125`). | feature | Ranked gaps, item 7 |
| PI.11 | Item 8: tool-call formats not built in (GLM 4.5, Kimi K2, MiniMax, Command R, Granite, FunctionGemma); the pythonic template's `str()` round trip; no partial-argument streaming; Llama 3.1's built-in tools rendered back only with `builtin_tools`. | feature | Item 8: what remains |
| PI.12 | Items 9 and 10: KTO and IPO; a loss over hidden states; packing for preference batches; per-module ranks; IA3 and VeRA with a PEFT type registry; DoRA through the fused LoRA kernels. | feature | Items 9 and 10, "Still open" |
| PI.13 | Item 13: a stopping hook on token ids, logits processors and logit bias over the normal sampler, JSON-schema constrained sampling (the CLI's `run --schema` and the server's wait for it), speculative decoding. | feature | Ranked gaps, item 13 |
| PI.14 | Item 15: metric accumulators (F1, AUC, precision, recall; `Trainer`'s `Metric` is a batch mean); the `Checkpoint` callback saves weights only (stop checkpoints keep the optimizer since plan 23 C8b). | feature | Ranked gaps, item 15 |
| PI.15 | Item 17: an embedding model kind for the engine (so `idrak serve` can answer `/v1/embeddings`); `GenerationOptions.Extra` for unknown options. | feature | Ranked gaps, item 17 |
| PI.16 | Item 18: a pooling choice in `TextEncoder` (CLS, last token), an `IChunker`, vector metrics beyond dot and cosine. | feature | Ranked gaps, item 18 |
| PI.17 | Item 19: registered file formats skipped when picking hub files; a compression registry (`.zst`, `.xz`). | feature | Ranked gaps, item 19 |
| PI.18 | Item 20: a safetensors data-type registry (FP8, I8/U8). | feature | Ranked gaps, item 20 |
| PI.19 | Item 21: checkpoint export: a `Write` on `ICheckpointFormat`, a GGUF writer. | feature | Ranked gaps, item 21 |
| PI.20 | Item 22: weights files record built-in packed layers by index; no `Module.Pack(format)`; `idrak tune` loads bases as int8, int4 or bf16 only; the built-in formats' fused paths read built-in formats only. | feature | Ranked gaps, item 22; plan 10 phase 6 |
| PI.21 | Item 23: `Functions` and `Filters` dictionaries on `JinjaChatTemplate`. | feature | Ranked gaps, item 23 |
| PI.22 | Item 24: initializer delegates (Kaiming), activations by name in `Dense`, an FP16 precision (see 1.17). | feature | Ranked gaps, item 24 |
| PI.23 | Item 25: tool results as content blocks (`ToolResult.Content` is a string). | feature | Ranked gaps, item 25 |
| PI.24 | Registry hygiene not rechecked since the survey: the `PackedFormat` and `KeyValueFormat` enum overloads bypassing the registry, internal fast paths of `KeyValueLayout` and `PackedWeight`, name matching rules (`SlotTable` since plan 10 6b gives every registry `Unregister`, a default and an origin). | check | Gaps inside the existing registries |
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
| CLI.2 | `devices` shows no hardware kind for CUDA devices (`BackendHardware.HardwareKind` is set by Vulkan and HIP only). | feature | Gaps (Health) |
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
| CLI.13 | No loader for pre-packed language-model weights (`quantize --out` writes `Module.Save`; chat, run and serve pack at load). | feature | Gaps (Models) |
| CLI.14 | No GGUF writer and no language model from an `.ikm`, so `convert` writes Hugging Face folders only (see PI.19). | feature | Gaps (Models) |
| CLI.15 | `idrak resume` continues from the weights with a fresh optimizer and schedule (the `Checkpoint` callback keeps weights only). | feature | Gaps (library), Train |
| CLI.16 | Writing Parquet is not in the library (`data convert` writes JSON Lines, JSON, CSV, TSV). | feature | Gaps (library), Train |
| CLI.17 | `tune init` writes the library's defaults until `idrak suggest` can size them. | feature | Gaps (library), Train |
| CLI.18 | No embedding-model loader (a bi-encoder from a Hugging Face embedding model): `rag index -m` embeds with a chat model's averaged hidden states; `RetrievalIndex.Save` keeps no vectors of a non-`TextEncoder` embedder. | feature | Retrieval and measure, "Gaps" |
| CLI.19 | prep.json's "augment" list is not applied by `train` and `suggest --search`. | feature | Gaps (Design); plug-in.md, dataset loaders "Remaining" |
| CLI.20 | `kernels dump` writes default shapes; per-device dumps wait for a public way to ask a device which variants it built. | feature | Gaps (Developers) |
| CLI.21 | `shell` completes command names and options only; model names and file paths are to come from the shared completion helper. | feature | Gaps (Developers) |
| CLI.22 | Later, not in this build: a plug-in marketplace listing; remote devices (a command run on another machine's `idrak serve`). | feature | More from a second pass, "Later" |
| CLI.23 | Arabic: the bodies of commands' help and other commands' messages are English; a streamed Arabic answer appears line by line; the user's typing is echoed unreordered where the terminal has no bidirectional support. | feature | Arabic messages, "Remains" |
| CLI.24 | Arabic in real terminals (Windows Terminal, console host, Git Bash, a Linux terminal): `help`, `doctor`, `devices --lang ar` and `--lang-render logical`. | check | open-questions, hardware check 11; Arabic messages |
| CLI.25 | R9 leftovers: `TrainSession.cs` CSV relabelling (strings and `Split` per row); `ImageTraining.cs:906` uploads two tiny per-channel tensors per batch; `ServeHost` auth allocates two small byte arrays per request; the bidi algorithm allocates per right-to-left line; `DataProfile` keeps each row's JSON string. | speed | Plan 14 backlog (R9) |

**Dropped as done:** `/stats` memory (the device's usage, `RunSupport.cs:167`); eval metrics waiting on plug-in gap 16 (`TuningMetrics` names); JPEG training only with a registered codec (`JpegCodec` is a library default).

## Open questions and checks (removed 2026-10-22: its items are filed under their plans here)

**Status:** Removed. **Where it stands:** In progress — of nine decisions, decision 4 is open; of eleven hardware checks, seven are open. Every open item is filed under its plan: Plan 7 (decision 4; checks 1, 5, 8, 10), Plan 8 (check 4), plug-in points (check 6), idrak-cli (check 11).
**Owner's runs:** the hardware checks above (filed as 7.9-7.11, 8.1, PI.7, CLI.24).

No leftovers left here. Answered and dropped: decision 1 (`architecture` merged into `main` 2026-10-04), 2 (the public backend API built by plan 10 phase 4), 3 and 8 (done), 5 (the README tagline reads "Deep learning in pure .NET. Every GPU. No native dependencies."), 6 (LinkedIn series dropped), 7 (native Linux drivers from a live USB when at hand; the run is 1.15), 9 (port 7317, then `-p`, `IDRAK_PORT`, `serve.port`). Checks done: 2 and 9 (the "fine-tuning" and "window kernels" filters in plan 17's round-3 CUDA run), 3 (superseded by the later CUDA and Vulkan runs of plans 17 and 23), 7 (the tool run on the owner's card, plan 11 phase 7).

Decided 2026-10-03 (kept here, the file's history; idrak-cli.md's "Arabic messages" cites the Arabic rule): merging into `main` not yet, then `architecture` merged into `main` 2026-10-04 with the version set to 0.2.0; the public backend API parked (built later by plan 10); the LinkedIn series dropped for now; native Linux GPU drivers tested from a live USB when one is at hand; `idrak serve` on a port of its own (7317, then `-p`, `IDRAK_PORT`, `serve.port`); Arabic messages only if the console shows them right to left with joined letters, otherwise dropped (built with the tool's own shaping and reordering, `--lang ar`); `idrak-cli` fast-forwarded into `architecture` and retired; round 3 (mixture of experts, sliding-window and soft-cap kernels, dataset loaders, `idrak distill`, provider-neutral serving names, Arabic output) built and merged.

## GPU checks for plans 12 and 13 ([gpu-checks.md](done/gpu-checks.md))

**Status:** Done. **Where it stands:** In progress — the owner's GPU runs of plans 12 and 13; plan 17's CUDA runs covered several filters (conformance kit, attention spans, vision tuning, image prefill, vision kernels, fine-tuning, packed); the rest on CUDA and every Vulkan run are not recorded as run.
**Owner's runs:** pending: the whole file ("Correctness, every device", "Speed", "The real model").

No leftovers.

## The moved text

Each section or paragraph moved out of a plan on 2026-10-22, as it was written there.

### done/1-nvidia.md

> | # | Item | Expected gain | Notes |
> |---|---|---|---|
> | 1 | Delayed FP8 scaling keyed by call order, not address | ~18 ms per 1.25B training step | performance-notes step 1 |
> | 2 | FP8 quantization fused into producers (layer norm, GELU, product epilogues) | up to ~100 ms per 1.25B step | needs item 1 |
> | 3 | Persistent decoding kernel (several layers' GEMVs in one launch) | decoding is at ~31% of the bandwidth ceiling | performance-notes step 3 |
> | 4 | Int8 prompt products on int8 tensor cores | prompt pass | needs a second, k-major copy of int8 weights |
> | 5 | Cached compiled kernels (cubin per driver and GPU) | ~70 ms off the first prompt pass | performance-notes step 8 |
> | 6 | Memory pool size classes and partial trimming | fewer full flushes near a memory limit | audit, CUDA #4–5 |
> | 7 | Fine-tuning loss read once per optimizer step | no host stall per micro-batch | audit, CUDA #6 |
> | 8 | QLoRA backward without expanding int4/int8 weights | 4–8× less weight traffic per step | needs packed transposed kernels |
> | Gap | Plan |
> |---|---|
> | Compute 5.0–7.5 (Maxwell, Pascal, Turing) never run | the kernels assemble for sm_50/61/75; borrow or rent a GTX 10 / RTX 20 card for one test and benchmark run |
> | Linux with a GPU never run | one run on a Linux machine (`libcuda.so.1`); a cloud GPU is enough |
> | CI does not run on a GPU | a self-hosted runner, or a manual "GPU check" step in the release checklist |
> | FP16 tensor cores (Volta/Turing have no bfloat16) | an FP16 `MatMulPrecision` for compute 7.0–7.5 |

### done/12-vision-tuning.md

> 1. Train the projector by default for Gemma 3, or the language model only (LlamaFactory's run: language model only)?
> 2. Feature cache on disk by default, or memory first and disk when it does not fit?

### 14-refactor-backlog.md

> ## Scope from now: CUDA first
>
> The owner's decision: **CUDA is the most important target from now. Anything not related to CUDA is set aside**
> (Vulkan, HIP, CPU-only kernel work, the CLI, samples other than what CUDA training needs, the unfinished audit areas
> outside the CUDA path). The sections below keep those items for later; work only from this list until the owner says
> | Order | CUDA item | Where (section below) |
> |---|---|---|
> | 1 | Verify the fused LSTM/GRU cell kernels on CUDA (the `recurrent` filter) and measure the OCR training step with `train --profile 10` against the 411 ms step | 1.5, 4 |
> | 2 | Regression check of the CUDA path against the library's starting numbers (`--bench-gemm`, `--bench-gemv`, the Chat sample's `profile ... --cuda --int8 --kv16`) | 4 |
> | 3 | Size classes for the CUDA keys still keyed by exact data sizes: PackedSplits/PackedTile/PackedMulti/Gated (`m`), DecodeSplits/DecodeMinChunk (`capacity`) | 2, Tuning |
> | 4 | Convolution choice per call: candidacy that does not depend on free memory at the moment, so a known choice is a lookup (no candidate list and `cuMemGetInfo` per convolution) | 2, Tuning |
> | 5 | Cheaper measuring on CUDA (shorter warm-up once the GPU is busy, fewer rounds for far-apart candidates) | 2, Tuning |
> | 6 | Library paths that cost CUDA time on every step: graph constants and BatchNorm eval read back or recomputed per forward (a GPU sync per node; needs a buffer invalidation hook), `ActivationMemory` arrays per layer per step, `Conv2d.ForwardFused` refolding per inference call, `ExactGELU` as ~20 launches (a fused CUDA kernel) | 2, Layers and models |
> | 7 | R1's CUDA leftovers: `CtcMeta` per CTC call, closures and copies on prompt-sized packed products and tensor-core `Run` delegates, invariant formatting in the PTX generators | 2, CUDA |
> | 8 | Card names in CUDA code comments (`CudaBackend.Quantized.cs`, `CudaDeviceLimits.cs`, `TuningCache.cs`) | 1.3 |
> | 9 | The CUDA runs of `plans/gpu-checks.md`, plan 12 phase 5 on the real Gemma 3 (CUDA), and docs/performance-notes.md next steps 1-9 (all CUDA) | 4 |
> | 10a | **Done** (α and β rescaled every step, offsets in double; passes on CUDA). CUDA CTC precision on long sequences: α and β in float log space reach ~-1,500 over 810 steps, so the 400-label gradient is ~0.1% off the CPU's double (test "vision kernels: ... CTC, 400 labels, batch first gradient"). Fix: rescale α and β every step (subtract the row's maximum), offsets accumulated in double; the per-state shares are unchanged and only the offsets enter the final term | 2, CUDA |
> | 10c | **Done**: CUDA kernels for sqrt, sin, cos, SiLU, sign, maximum, minimum, pow, clamp and where and their gradients (they took the host fallback; "tensor ops" now requires no host fallback on CUDA); all pass on CUDA | - |
> | 10d | **Done, to verify on CUDA** (the `vision kernels` filter, then `train --profile 10`). The OCR profile (1908 ms a step, 107 ms of GPU time): `ctc_loss_bwd_global_f32` 55 ms a call (30% of GPU time) and `matmul16_nt` 16x128x512 at 0.1 TFLOPS (22%). Fixed: CTC rows in dynamic shared memory whenever they fit what the device lets a block opt in to (was: only when the states fit the block width, so OCR lines took the scratch variant); the gradient groups the labels once a call instead of rescanning them for every label at every step (O(L²) loads a step); the per-step maximum by shuffles. Few-row products (gemv) take up to 16 rows in one launch (row groups along grid y), so the recurrent step's h·U and dz·Uᵀ (16 rows: the batch) no longer run on 8 blocks of the 16x16 kernel. `--profile` splits a step into data and compute | 2, CUDA |
> | 10e | Measured on the owner's card after 10d: 70.1 ms a step with measuring off (was 411), data 3 ms; 913.9 ms with measuring on, so the gap is measuring new size classes during the first steps (item 5). The CTC gradient is ~0.5-0.85 ms a call (was 55). Next on the GPU: `gemv_nt_f32` at ~36 µs a call (40% of GPU time) because its 8 row loads shared one predicated register and ran one after another; fixed (a register per row, two k a pass), to verify. Then fusing h·U and dz·Uᵀ into the cell kernels (one launch a step) | 2, Tuning |
> | 10f | **Done** (passes on CUDA): `gemv_nt_f32` 36 → 11.4 µs a call (a register per row). OCR step 67.2 ms with measuring off | - |
> | 10g | **Step 1 done, passes on CUDA** (52.3 ms a step, from 67.2) (`recurrent` filter, then `train --profile 10`): the step kernels (`Backend.LstmStep`, `GruStep` and their gradients) take the recurrent product inside, one launch a time step instead of two, forward and backward; the block's slices follow the device's reported block size. **Step 2 done, passes on CUDA**: a warm OCR step 36 → 26.3 ms (408 → 609 lines/s, `train --profile 10 --profile-after 3000`), 1,859 → 243 launches a step, the loop's GPU time about 20 → 7.8 ms a step; the step is now GPU-bound (23.9 ms of GPU time). the sequence kernels (`Backend.LstmSequence`, `GruSequence` and their gradients) run a layer direction's whole loop in one cooperative launch, a grid barrier between steps; each step is the step kernel's work and sum order (bit for bit the same values), the blocks resident at once as the device's occupancy reports, each taking the step's (row, 32 units) items in turn; while a graph is captured the step kernels run. Not done: U kept in shared memory (a block's 32 units of U are 128 KB at H = 256, past what most devices let a block hold). Was: The recurrent loop is launch-bound: 4 launches a time step (h·U, cell; cell gradient, dz·Uᵀ), each near the launch floor, ~1,600 a training step, ~60% of GPU time. Fix: one persistent kernel per layer and direction for the forward loop and one for the backward loop (each block keeps its units' slice of U in shared memory; blocks co-resident by a cooperative launch, a grid barrier between steps; block count from the device's reported limits) | 2, CUDA |
> | 10h | **Done, passes on CUDA**: per 10 OCR steps `group_reduce` 52.9 → 10.8 ms, `norm_stats` 21.6 → 1.8 ms; the step 52.3 → 38.1 ms with 10i. `group_reduce_f32` 0.56 ms a call and `norm_stats_f32` 0.7 ms (BatchNorm statistics, conv bias gradients): one block per channel walks hundreds of thousands of elements while most SMs idle. Fix: split each group into fixed-size chunks across blocks (partial sums, then a short final pass in a fixed order) | 2, CUDA |
> | 10i | **Done, passes on CUDA**: per 10 OCR steps `conv_bwd_weight` 59.0 → 23.7 ms, `conv_bwd_input` 36.9 → 21.8 ms. Was: `conv_bwd_weight_f32` 2.8 ms a call (20 calls, 57 ms in the profile): check its parallelism the same way (a reduction over N·OH·OW per weight) | 2, CUDA |
> | 10j | Measuring during the first training steps: 914 ms a step with measuring on vs 67 ms off (new size classes measured one after another). Cheaper measuring (item 5) or measuring off the step's critical path | 2, Tuning |
> | 10b | CUDA does not pack image-model sequences (Gemma 3, LLaVA: "cuda:0 does not pack this model's sequences (FineTuner pads them)"; LLaVA: no packed causal attention): padded rows waste work in vision fine-tuning | 4 |
> | 10 | A training-step profile on the CUDA path beyond kernel times (per operation and layer: the CLI's internal `Recorder` as a library hook) | 2, gaps |
> | 11a | **Done, to verify on CUDA** (PTX rules 80 and 87, results unchanged: the same instructions, each accumulator's terms in the same order). Rule 80, a register per loaded value so a pass's loads are in flight together (the gemv_nt pattern): the 18 int8/int4/bfloat16 `*_gemv_*` kernels (`PtxKernels.Quantized.cs` `Int8Gemv` `Step`: 8 rows × 4 k a pass through one predicated register, 32 loads in a chain), `attention_decode_f32/int8/bf16` (`PtxKernels.Rows.cs` `AttentionDecode`: the 8 key and 8 value loads of each cached position), `attention_flash_*` PV loop, `attn_bwd_q_f32` PQ loop and their span copies `attention_spans_f32` (`PtxKernels.Spans.cs`) and `attn_spans_bwd_q_f32` (`PtxKernels.SpansBackward.cs`) (4 shared-memory loads per key), `gemv_nn_f32`/`gemv_multi_f32` K1 tail. Rule 87, `.maxntid` on the kernels launched with blocks of 512 threads or more: `gemv_nn_f32`, `gemv_multi_f32` (1024), the `*_gemv_*` family (512), `softmax_ce_rows_f32` (`SoftmaxCrossEntropyThreads`, now also the launch's), `sample_rows_f32` (the shapes' SamplerThreads). Expected: decode-time products and attention closer to memory speed (gemv_nt went 36 → 11.4 µs with the same change); measure with `--bench-gemv` and the Chat sample's `profile ... --cuda --int8` | 2, CUDA |
> | 11b | `gemv_nn_f32` / `gemv_multi_f32` (`PtxKernels.Rows.cs:2269`), rule 81: k is split only over the 32 slices of one block, so a narrow product runs on ⌈n / 32⌉ blocks per row group (n = 128: 4 blocks of 1024 threads) while the other SMs idle, however long k is. Fix: k splits across blocks (a grid dimension, `chunk` rows each) with partial sums and the last block of a column range adding them in split order, as `int8_gemv_f32` does; the split count from the device's multiprocessors and the shape. Expected: narrow products with long k fill the device (time divided by about the blocks added, up to the memory bound) | 2, CUDA |
> | 11c | `gemv_nt_f32` (`PtxKernels.Rows.cs:2548`), rule 81: one warp per column walks all of k, grid x = ⌈n / 8⌉, so a few columns (the recurrent step's dz·Uᵀ with few units, attention's q·kᵀ over a short cache) launch a handful of blocks. Fix: k chunks across blocks with fixed-order partial sums (as 11b). Expected: the small-n calls stop being latency-bound on a handful of SMs | 2, CUDA |
> | 11d | `matmul_f32` (`PtxKernels.cs:719`, tile loads at `:802`), rules 83 and 81: with transA (transB) the tile loads read A[(t + tx)·m + row] (B[col·k + t + ty]), so consecutive threads read addresses m (k) floats apart: uncoalesced; and 16 × 16 tiles give few blocks on small products (10d's 16×128×512 at 0.1 TFLOPS). Fix: load a transposed operand with the thread roles swapped (tx along its contiguous dimension) and write it transposed into shared memory; route small-m products to the k-split kernels. Expected: full-width reads for the transposed layouts | 2, CUDA |
> | 11e | `sum_f32` (`PtxKernels.cs:642`, `atom.global.add.f32` at `:706`), rule 86: each block's partial is added atomically, so the sum's bits depend on which block finishes first (run to run, card to card). Fix: per-block partials to scratch and the last block (counted, as `int8_gemv_f32`) or a second one-block pass adds them in block order. Expected: the same bits every run, at the cost of one counter or one tiny launch | 2, CUDA |
> | 11f | `sumsq_f32`, `multi_sumsq_f32` (`PtxKernels.Rows.cs:430`, `:466`; the gradient norm of clipping), `sum_cols_strided_f32` (`:539`), `layernorm_bwd_params_f32` (`PtxKernels.Decoding.cs:110`, dgamma/dbeta), rule 86: per-block or per-chunk sums added with `red.global.add.f32`, so the order follows the scheduling. Fix: partials per block or chunk, added in a fixed order (11e's pattern). Expected: deterministic clipping factors and LayerNorm parameter gradients | 2, CUDA |
> | 11g | `gemm_tc_*` split k (`PtxKernels.TensorCore.cs:975`, `EmitTensorEpilogue`), rule 86: each k split adds its partial tile into C with `red.global.add.f32`, so a split product's bits change with block order. Fix: partial tiles to a workspace and the last split of a tile adding them in split order (a counter per tile), or a fixed-order reduction pass. Expected: split-k products reproducible bit for bit | 2, CUDA |
> | 11h | `sum_rows_f32` (`PtxKernels.cs:590`), rule 81: one thread per column walks every row, so a narrow layer's bias gradient over a long batch runs ⌈cols / BlockSize⌉ blocks with a serial loop over all rows. Fix: chunks of rows along grid y with partials added in chunk order (11e). Expected: such gradients use the whole device | 2, CUDA |
> | 11i | Rule 80 left in epilogues (one pass per tile, low value): `gemm128/64_*` beta reads (`PtxKernels.Rows.cs:1867`, 8 or 4 predicated loads through `%f2`), the dK/dV write-back of `attn_bwd_kv_f32` (`:1368`) and `attn_spans_bwd_kv_f32` (`PtxKernels.SpansBackward.cs:345`, `%f23`/`%f24`, 4 a key); the span kernels' range prologue (`%f2`/`%f3` per row). Fix when the file is next touched: a register per column / slot, the same order | 2, CUDA |
> | 11j | Rule 49 left: `PtxKernels.Advanced.cs`, `PtxKernels.Convolution.cs`, `PtxKernels.Recurrent.cs` still interpolate numbers with the current culture (they were being edited elsewhere); the other generators format theirs invariantly (StringBuilder appends with `CultureInfo.InvariantCulture`, `string.Create`). The reasons in `CudaDeviceLimits.cs` (`Derive`, `Fits`) are messages for people and keep the current culture | 2, CUDA |
> | 12a | **Not started.** The recurrent loop is 33% of the warm OCR step's GPU time (lstm_seq_bwd 4.3 ms, lstm_seq 3.4 ms a step). Keep each block's units of U (and Uᵀ in the gradient) in shared memory when the device's opt-in room allows, chosen by its reported limits (rule 84); at H = 256 one block's 32 units are 128 KB, so split the units finer or take the slice that fits | 2, CUDA |
> | 12b | **Not started.** Convolutions are about 22% of the warm OCR step's GPU time (conv_bwd_input 1.9, conv_fwd 1.6, conv_bwd_weight 1.5 ms a step): profile each against rules 80, 81 and 83 | 2, CUDA |
> | 12c | **Not started.** The first epoch is cold: lines of new widths keep it at about 270 s against 150 s warm (on the OCR cut lines; steps 21-30 took 780 ms against 26 ms warm). Measure first (device allocations and tuning measurements a step); the suspect is the pool's exact-size blocks (best fit within 25%), so growing widths allocate new blocks: size classes (an eighth of an octave) would bound the distinct sizes | 2, CUDA |
> | 12d | **Not started.** The held-out CER moves between epochs even at a learning rate near 0 (OCR cut lines, epochs 26-30: 2.0-3.0%, best 1.5%). Padding is ruled out (batches of similar widths read the same: 1.53% against 1.53%). Suspect: BatchNorm's running statistics from the last small augmented batches; recompute them over a pass of training data before evaluating (or evaluate with an averaged copy) | 2 |
>
> Set aside (not CUDA): 1.1 Vulkan CTC, 1.2 GGUF fallback, 1.4 card names in docs, 1.6 Vulkan recurrent, 1.7 the
> sample's non-finite loss message, the Vulkan/HIP and CPU backend items, the CLI items, the sample gaps that are not on
> the CUDA training path, and the unfinished audit areas (section 3) except where a file is on the CUDA path.
>
> Everything left unfinished by plans 12 and 13, the fused recurrent work and the optimization-rules audit
> (`docs/optimization.md`, eleven areas R1-R11), in one place, so the refactor can be planned from it. Each item says
> where it is, why it was left, and what a fix needs. Nothing here is guessed: every entry comes from an agent's report,
> a measurement on the owner's card, or a count over the branch.
>
> ## State at the time of writing
>
> | Work | State |
> |---|---|
> | Fused LSTM/GRU time loop (CPU and CUDA cell kernels, one autograd op per layer and direction) | merged; CPU tests pass; CUDA kernels never run on a GPU |
> | Measured kernel choices keyed by size class (`TuneSizes`) | merged; CUDA keys and the shared convolution key only |
> | Audit R1 CUDA, R2 Vulkan/HIP, R3 CPU devices, R5 Layers/Models, R9 CLI N-Z + Shared | finished and merged |
> | Audit R4, R6, R7, R8, R10, R11 | stopped part-way; their committed fixes merged; the rest of their areas not covered (below) |
> | Whole solution after the merges | builds, 0 errors, 0 warnings; tests after the merge are the owner's |
>
> ## 1. Correctness and rule breaks (fix first)
>
> | # | Where | What | Fix needs |
> |---|---|---|---|
> | 1.1 | `src/Idrak.Gpu/Vulkan/VulkanKernels.Ctc.cs` | CTC loss is -∞ on the first step on Vulkan (the ArabicOcr sample's tests: 4 of 13 fail on vulkan; CPU and CUDA pass) | find the kernel bug (α rows, log-sum, meta reads); the conformance kit's CTC case on Vulkan should catch it |
> | 1.2 | `src/Idrak/Models/GgufModel.cs:297` | an unregistered `tokenizer.ggml.pre` silently gets Llama 3's pattern: a family fallback CLAUDE.md forbids | throw "pre-tokenizer '{pre}' is not registered; register its pattern with GgufPreTokenizers.Register", and change `CustomGgufPreTokenizer` in tests/Idrak.Tests/ModelPluginTests.cs, which asserts the fallback |
> | 1.3 | `src/Idrak.Gpu/Cuda/CudaBackend.Quantized.cs` (4), `CudaDeviceLimits.cs` (4), `TuningCache.cs` (1), `Vulkan/VulkanBackend.Dispatch.cs` (1) | card names in comments (card-agnostic rule) | reword as what was measured, without the card |
> | 1.4 | README.md (51), plans/*.md (~55), src/Idrak/README.md (3), docs/performance-notes.md, docs/optimization.md | card names in benchmark tables and notes | owner's decision: a results table names the machine it ran on; advice or defaults must not |
> | 1.5 | fused recurrent CUDA kernels (`PtxKernels.Recurrent.cs`, `CudaBackend.Recurrent.cs`) | never run on a GPU (no GPU here) | the owner's `recurrent` filter on cuda; then the OCR `train --profile` against the 411 ms step |
> | 1.6 | Vulkan recurrent | no fused cell kernels: Vulkan still runs ~13 launches a step | `LstmCell`/`GruCell` kernels for Vulkan when Vulkan is back in scope |
> | 1.7 | ArabicOcr sample `TrainCommand` | a non-finite loss crashes writing `recognizer.json` (JSON cannot hold ∞) | stop with a clear message on a non-finite loss (sample); 1.1 is the cause seen |
>
> ## 2. Speed and memory left in the library
>
> ### Tuning (measured choices)
>
> - **Per-class measuring cost:** each measurement is ~0.2-0.5 s (warm-up ≥ 25 ms of GPU time, 8 paired rounds,
>   a confirmation). With size classes a varying workload measures ~15 keys per class once; cheaper measuring (shorter
>   warm-up once the GPU is busy, fewer rounds for far-apart candidates) would shorten the first minute of a run.
> - **Exact keys still keyed by data sizes:** `TuneSizes` is applied to CUDA's FloatTile, TensorSplits, SpanWarps,
>   AttentionPath and the convolution key. Still exact: CUDA PackedSplits/PackedTile/PackedMulti/Gated (`m`),
>   DecodeSplits/DecodeMinChunk (`capacity`), and every Vulkan key with rows (Splits, Rows, MatMul, MixedMatMul,
>   TiledAttention, Attention, SpanAttention, PackedPrompt). Decide per key which sizes follow the data.
> - **Convolution choice per call** (`CudaBackend.Convolution.cs:84-150`): the candidate list and `cuMemGetInfo` run on
>   every convolution, because the composed path's candidacy depends on free memory at that moment. Make candidacy not
>   depend on the moment (a measured budget, or retry on allocation failure) so a known choice is a lookup.
>
> ### CUDA (R1, not fixed)
>
> - `CudaBackend.Ctc.cs` `CtcMeta`: a `new float[3·batch]` per CTC call (also Vulkan's); pool it.
> - Prompt-sized packed products (`PackedMatMulLargeKernel`, `PackedManyLarge`, LowRank): a closure, delegate and small
>   copy per call; tensor-core callers' `Run` delegates (MatMulLowRank, BFloat16Transposed, GemmStrided,
>   `TensorCoreRows`): two small allocations per large product.
> - PTX generators interpolate numbers under the current culture; safe today (non-negative ints, floats through `F()`),
>   but rule 49 wants invariant formatting.
>
> ### Vulkan / HIP (R2, not fixed)
>
> - HIP `Launch`: a lock and two dictionary lookups per launch.
> - `SupportsGraphs` reads an environment variable per recorded graph.
> - `ReductionFlags`, `MatCandidates`, `PowersOfTwo`, `GemvCandidates` allocate (only while measuring).
>
> ### CPU backend (R3, not fixed)
>
> - `LayerNormTrainKernel` computes mean and variance twice (it calls the `LayerNormFused` wrapper so registered
>   overrides still apply); a fused training layer norm needs an override-aware operation.
> - `GroupMoments` merges per-chunk sums under a lock in completion order: run-to-run bit differences with several
>   threads. Deterministic order changes outputs once.
> - Vectorized loops without an `IsHardwareAccelerated` check (rule 34): `Moments`, `MultiplyAddInPlace`,
>   `GroupAffineCore`, `AddInPlace` (Advanced.cs) and loops in CpuBackend.cs. Correct in software mode; a scalar
>   fallback changes bits (FMA, lane order).
> - Depthwise convolution (Vision.cs) is scalar and runs its activation as a separate pass; vectorize across outputs and
>   fuse the activation (bits change: `Vector.Exp` vs `MathF.Exp`).
> - `MathF` loops in Pointwise (sin, cos, pow, sqrt, silu) and the gated-activation backward are scalar; the composed
>   convolution's bias and activation passes are not fused.
> - bfloat16 tiled kernel's leftover columns read weights column by column (< 2 vectors wide).
> - Small per-call arrays: norm scale/shift copies, `Enumerable.Repeat` in `GroupScaleShift`, CTC per-sequence tables,
>   `Permute`'s shape copies.
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
> - `ExactGELU` is ~20 element-wise passes: a fused erf kernel in the device contract.
> - GGUF F16 decode is scalar (the BCL's vectorized Half conversion is a NuGet package, which rule 0 forbids): write a
>   vectorized converter.
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
> - The regression check against the library's starting numbers (docs/performance-notes.md: `--bench-gemm`,
>   `--bench-gemv`, the Chat sample's `profile Qwen/Qwen3-0.6B --cuda --int8 --kv16`): 78 TFLOPS, 474-493 tok/s,
>   177 launches per token, a 7.7 ms 180-token prompt pass.
> - The OCR training step after the fused recurrent loop: `train --profile 10` against 411 ms (measuring off) and
>   1.9-3 s (measuring per shape, before size classes).
> - docs/performance-notes.md "Next steps" 1-9 (delayed FP8 scaling by call order, FP8 fused into producers, persistent
>   decoding GEMVs, int8 prefill on int8 tensor cores, prompt-pass fusions, training step outside the products, decoding
>   attention at short contexts, cached CUDA modules, prompt-pass graphs) are all still open.
>
> ## 5. Suggested order (before the CUDA-first decision; see the top for the order now)
>
> 1. Correctness: 1.1 (Vulkan CTC), 1.2 (GGUF fallback), 1.3 (card names in code), 1.7.
> 2. Measure before refactoring: the owner's GPU runs of section 4 (recurrent on CUDA, the OCR profile, the
>    regression benchmarks). Their numbers decide what is worth refactoring first.
> 3. Library contracts the samples need (section 2, gaps): `MeasuredBatches`, a profiling hook on any device, `noise`
>    in `Augmentations`, a source path on `ChatImage`.
> 4. Invalidation hook for buffers (GraphModule constants, BatchNorm eval, folded convolution) and `ActivationMemory`
>    spans: one public-API change, several speedups.
> 5. Tuning: size classes for the remaining data-sized keys (CUDA packed/decode, Vulkan rows); convolution candidacy
>    independent of free memory; cheaper measuring.
> 6. Finish the audit of the six unfinished areas (section 3), hottest first (R10, R7, R4, R6), then the analyzers.
> 7. CPU kernel items of R3 that change bits (deterministic `GroupMoments`, scalar fallbacks, fused depthwise and
>    activations), each with before/after numbers and the owner's agreement that the new bits are the reference.
> 8. Vulkan: fused recurrent kernels (1.6) when Vulkan is back in scope.

### done/16-qwen-vl.md

> Some families change what the decoder computes, not only what enters the prompt: Qwen3-VL adds image features to the
> hidden states of several decoder layers (DeepStack); TrOCR, Donut and Llama 3.2 Vision read the image by cross-attention.
> Each is a further generic output of the contract (features with the layers to add them at; features for cross-attention
> with the decoder blocks to take them), in a later plan. Qwen2.5-VL and Qwen2-VL need neither.

### done/17-vision-ocr-speed.md

> Plan 17 is finished for **CUDA** and the **CPU** only. Every change went through a contract with a default, so Vulkan,
> HIP and outside backends give the same results as before (the CPU and the minimal device run the defaults in the tests),
> but they do not get the speed or memory gains, and **no Vulkan or HIP device has run this plan's tests or benches**.
>
> | Device | Has | Left (runs the contract's default today: correct, no gain) |
> |---|---|---|
> | **CUDA** | everything | Only a confirmation: E9 (stack trim, chunks, the measurement's scratch) ran on the larger card; the smaller card has not run it yet |
> | **CPU** | every item that applies, its tests passing on the CPU | Nothing required. D9's rest (the composed path per 64-row block) is not taken: it changes the sums, so it could only be a CER-gated option. The device-memory items (E6, E9) do not apply to host memory |
> | **Vulkan** | span kernels up to head size 256 (A3), `ResizeNormalize` on the device (C6), `AvailableMemory` from its heap; the generic items (B, E1-E3, E7, E8, F, G, H) | No `AttentionSpansStrided` (D3: copies the heads), `PackedMatMulEpilogue` (D2), `PackedMatMulNormAddRmsNorm` (D11), `AttentionRows` for pages of different lengths (F3), greedy sampling without top-k (B8), `IMemoryOffload` (E6: lookup tables stay on the device), chunks and long-lived blocks (E9), a driver free-memory figure (E7's "device other"); the read, the CER gate and `--bench-read` never run on a Vulkan device |
> | **HIP** | the generic items only | Everything Vulkan lacks, plus span kernels (it keeps the CPU's limits, A3) and `ResizeNormalize` (C6); never run on a HIP device |

### done/2-cpu.md

> | # | Item | Where | Expected gain |
> |---|---|---|---|
> | 1 | ✅ (section 4) bfloat16 products of 4–8 rows keep up to 8 rows' sums in registers over k chunks of half a cache line (`CpuBackend.BFloat16.cs`); identical results | `CpuBackend.BFloat16.cs` | measured ~1.7–2× for 4 and 8 rows on the container |
> | 2 | Products with a transposed B of 17–63 rows still transpose all of B per call (only ≤ 16 rows read it in place) | `CpuMatMul.cs`, `NtRows` | 3× at 64 rows (measured 0.1.4) |
> | 3 | `FusedAdamW` is scalar and single-threaded | `CpuBackend.cs` | training steps on the CPU |
> | 4 | RMSNorm, rotary embedding, gated activation, add+norm are scalar | `CpuBackend.Quantized.cs` | per-layer ops of CPU inference |
> | 5 | Language-model loss: double-precision `Exp`, scalar, rows with weight 0 still computed | `CpuBackend.Decoding.cs`, `SoftmaxCrossEntropyRows` | fine-tuning on the CPU |
> | 6 | Memory pool: exact-size buckets, never trimmed without a limit, one lock | `CpuBackend.cs` | memory with varying shapes |
> | Gap | Plan |
> |---|---|
> | ARM64 (NEON): Apple silicon, Snapdragon X, Ampere servers | one test and `--bench-cpu` run on an ARM64 machine |
> | AVX-512 machines | one run on a recent Intel Xeon or AMD Zen 4/5 desktop, to check the 16-wide paths |
> | macOS | test run (CPU only) |

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
> tensor stores, and the kernel and registry lookups per launch; each is measured with the benches before and after.

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
> | 34 | speed | 🟡 | Attention with a bias copies the filled cache slots (two extra launches per layer per step); a product reading the cache prefix in place removes them | L1-T5 |
> | 35 | speed | 🟡 | `log_softmax_topk_f32` runs one block per beam, leaving most of a large device idle; split the vocabulary across blocks | L1-T5 |
> | 36 | speed | 🟡 | HIP and Vulkan have no `LogSoftmaxTopK` kernel (they read the logits back); a beam step does two small readbacks (values, tokens) | L1-T5 |
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
> - **P6, `idrak tune init` picks the vision example only with `-P`** (`src/Idrak.Cli/Commands/Train/TuneCommand.cs`, near
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

### done/8-hip.md

> 1. **A run on real hardware** (Linux with ROCm and Windows with the HIP SDK, an RDNA and a CDNA GPU if possible):
>    `--list-devices`, the HIP tests, then the whole list on `hip:0`. Fix what it finds first.
> 2. **More kernels, most used first**: softmax and log-softmax, float32 products (tiled; then `GemmStrided`,
>    `BatchedMatMul`), the sampler and penalties, rotary positions, gathers, decoding attention (float32, int8 and bfloat16
>    caches), cache writes, int4 and bfloat16 products, then the fused decoding kernels (`Capabilities.FusedKernels`).
>    The Vulkan kernels' structure (row kernels, split-k products, flash-decoding) carries over.
> 3. **Tuning by measurement**, as CUDA and Vulkan do: block sizes, k splits and attention splits measured at first use
>    and kept per device, driver and library build; wavefront-level reductions (`__shfl_xor` with `warpSize`) measured
>    against shared memory.
> 4. **Graphs** through hipGraph (stream capture), so `GenerationOptions.UseGraph` replays decoding steps.
> 5. **Matrix cores**: WMMA on RDNA 3/4 and MFMA on CDNA through rocWMMA-style intrinsics in the hipRTC source, chosen by
>    what the compiler accepts for the target (a test compile), never by target name.
> 6. **Training kernels**: backward passes, AdamW (fused and 8-bit), convolution, norms' gradients.
> 7. **Memory**: asynchronous uploads from pinned staging (`hipHostMalloc`), offloading to system memory, `IMemoryOffload`.
> What a third family found awkward in the internal `Backend` contract:
>
> - `Detach` and `Attach` are `private protected`: a backend outside the assembly cannot implement them, so `Evict` and
>   `Restore` cannot be supported from a separate package as they stand.
> - `Return` may run on the finalizer thread, so a backend may only touch its pool there; a public contract should say
>   so (or queue releases) rather than leave every backend to discover it.
> - No device context in the contract: HIP and CUDA bind a current device per thread, so every method must bind it
>   first (`MakeCurrent`); a per-call or per-thread hook would remove that repetition.
> - `Upload` takes a span and must finish reading it before it returns, so every upload waits for the device; an
>   asynchronous upload from memory the backend owns (`IHostStaging` exists but is optional and CUDA-shaped) would help.
> - `Capabilities` has no wavefront width, shared memory or compute-unit count; a backend whose fallbacks run on the
>   CPU must copy the CPU's limits so the layers agree with the fallbacks.
> - `TensorCoresUnavailable()` and the graph members (`BeginCapture` / `EndCapture` returning two `IntPtr`s, `ReplayGraph`)
>   are named and shaped after CUDA.
> - Host fallbacks download every operand whole, even for operations that read a small range.
> - Integers (token ids, int8 weights) live in float storages and are reinterpreted by each kernel; workable, but a
>   public API should name it.
> - No shutdown hook: a backend cannot release its stream and memory when the process is done with it.

### done/idrak-cli.md

> Gaps (Health): `report --bench` times the float32 products on each device (GFLOP/s); model speeds come from `idrak
> bench` once the Measure group lands. `report --tests` runs the test runner of a source checkout as a child process (no
> test list ships with the tool). `devices` shows no compute units for Vulkan devices (core Vulkan reports none) and no
> hardware kind for CUDA devices (not read yet). `doctor` checks the Vulkan driver files on Linux only (Windows keeps
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
> shows "?" for GPUs unless `--memory SIZE` is given; there is no loader for pre-packed language-model weights, so
> `quantize --out` writes the packed module (`Module.Save`) with the config and tokenizer, and chat/run/serve pack at
> load time with `-w`; the library has no GGUF writer and loads no language model from an `.ikm` package, so `convert`
> writes Hugging Face folders only (from GGUF or another folder, in bf16, f16 or f32).

> Gaps (library): the optimizer state and learning-rate schedule are not checkpointed, so `resume` continues from the
> weights with a fresh optimizer; image files are read and written through the library's `ImageCodecs` and `ImageEncoders` (the tool's own decoder is gone);
> writing Parquet is not in the library (`data convert` reads it, writes JSON Lines, JSON, CSV or TSV); `distill` has no
> vocabulary mapping (teacher and student must share a tokenizer for logits; `--generate` works across them); `tune init` writes the
> library's defaults until `idrak suggest` can size them; idrak tune loads base weights as int8, int4 or bf16 only (not
> other registered packed formats).

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

> - `kernels dump` writes the kernels in their default shapes (Vulkan at its widest width, cooperative-matrix products
>   in 16 x 16 x 16, PTX for the default kernel shapes); the library has no public way to ask a device which variants
>   it built, so per-device dumps wait for that. The tool reads the generators through `InternalsVisibleTo` (Idrak.Gpu
>   grants it to Idrak.Cli for this command only).
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
> | 4 | SPIR-V memory model for the cooperative-matrix kernels: SPIRV-Tools 2025.2 and later reject cooperative matrices without the Vulkan memory model; the drivers tested accept the kernels as they are | a) move the matrix kernels to the Vulkan memory model as its own task, tested on real GPUs · b) leave as is until a driver refuses them | a, after the RTX checks below |
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
> | 1 | RTX 5070 Ti | Single-pass bfloat16 matrix units under MixedPrecision: correctness and speed against float32 | `$env:IDRAK_DEVICES="vulkan:0"; dotnet run -c Release --project tests/Idrak.Tests -- --bench-vulkan matmul` and `$env:IDRAK_FILTER="matrix units"; dotnet run -c Release --project tests/Idrak.Tests` |
> | 2 | Any CUDA machine | The recorded fine-tuning step with a custom optimizer (could not run without CUDA) | `$env:IDRAK_FILTER="fine-tuning options"; dotnet run -c Release --project tests/Idrak.Tests` |
> | 3 | RTX 5070 Ti, laptops | The full lists after the plug-in work (315+ tests), CUDA and Vulkan | the per-device blocks in installation/windows.md |
> | 4 | An AMD discrete GPU with ROCm, or Windows with the HIP SDK | The HIP backend's first run | plans/8-hip.md, "Commands to validate" |
> | 5 | Phone | The width probe and the chat after the merged kernels | installation/android-termux.md, steps 8 and 9 |
> | 6 | RTX 5070 Ti, laptops | Mixture of experts on CUDA (composed operations, never run on CUDA here) and a real model's speed | `$env:IDRAK_DEVICES="cuda:0"; $env:IDRAK_FILTER="experts"; dotnet run -c Release --project tests/Idrak.Tests`, then `idrak c Qwen/Qwen1.5-MoE-A2.7B-Chat -w int4`, then `idrak b` on it |
> | 7 | Any machine | The `idrak` tool once the CLI lands: `idrak doctor`, `idrak devices`, `idrak c MODEL`, `idrak s MODEL`, `idrak b MODEL` | plans/idrak-cli.md, "Examples" |
> | 8 | Honor tablet (later) | CPU tests and, if its GPU has a Vulkan driver reachable from Termux (Mali through Mesa's Panfrost/PanVK, or Adreno through Turnip), the Vulkan list and a chat; first check the SoC and GPU with the phone guide's step 6 | installation/android-termux.md, steps 1-9 |
> | 9 | RTX 5070 Ti, a laptop with CUDA | The windowed and soft-capped CUDA kernels (decoding, `attention_flash_*`, the float32 gradients; never run): correctness, then the windowed decoder's speed against the composed path | `$env:IDRAK_DEVICES="cuda:0"; $env:IDRAK_FILTER="window kernels"; dotnet run -c Release --project tests/Idrak.Tests`, then `$env:IDRAK_FILTER="sliding windows"` and `"decoding"`; a failure there: `$env:IDRAK_CUDA_DEBUG="1"` names the kernel, `$env:IDRAK_WINDOW_KERNELS="0"` confirms the composed path still passes |
> | 10 | RTX 5070 Ti | The windowed kernels on Vulkan, and their speed | `$env:IDRAK_DEVICES="vulkan:0"; $env:IDRAK_FILTER="window kernels"; dotnet run -c Release --project tests/Idrak.Tests`, then `dotnet run -c Release --project tests/Idrak.Tests -- --bench-vulkan window` |
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
