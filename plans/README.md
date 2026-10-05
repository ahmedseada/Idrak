# Hardware plans

Plans for running Idrak on more hardware, and on more than one device at a time. Nothing here is built yet; each plan says what to build, in which order, how
to know a step is done, and what could stop it.

Ordered by priority: fix what shipped first, then the fallback every machine uses, then new hardware, largest
audience first.

| Priority | Plan | Hardware | Status | Why this priority |
|---|---|---|---|---|
| 1 | [NVIDIA](1-nvidia.md) | GeForce, RTX, data-center GPUs (CUDA driver API, PTX) | supported; tuning and gaps | the few-row rule is measured per card since 0.1.6; confirm on Ampere and Ada, then the remaining performance work |
| 2 | [CPU](2-cpu.md) | x64 (AVX2, AVX-512) and ARM64 (NEON); the fallback on every machine | supported; sampling and kernel speed | every AMD, Intel and Apple user runs here today; step 1 is a small fix (plain sampling 12 ms → ~6 ms on slower CPUs) |
| 3 | [AMD](3-amd.md) | Radeon RX 6000/7000/9000, Ryzen APU graphics, Instinct | not supported (runs on the CPU) | the largest group of GPUs without support; builds the Vulkan backend the Intel plan reuses |
| 4 | [Intel](4-intel.md) | Iris Xe and Arc integrated GPUs, Arc discrete GPUs, AI Boost NPUs | not supported (runs on the CPU) | GPUs come almost free once plan 3's backend exists; the NPU is inference-only, an add-on, and last |
| 5 | [Apple](5-apple.md) | Apple silicon Macs (M1–M4 GPU, Neural Engine) | not supported on the GPU; macOS untested | a smaller audience for this library, and a third kernel language (Metal); a Mac test run fills CPU gaps early |
| 6 | [Idrak.Network](6-network.md) | several GPUs or machines on one job; several machines serving one API | not started (one device per model, one machine per server) | a new package rather than new hardware; useful to NVIDIA users now, so it can run alongside plans 3–5 |
| 7 | [Backends](7-backends.md) | a minimum backend (12 core operations, host fallbacks), devices by provider, public KV cache formats, the Vulkan backend (Intel GPUs first) | in progress on `architecture` | the shared work plans 3–5 wait on, then the first new backend |
| 8 | [HIP](8-hip.md) | AMD GPUs through ROCm (Linux) and the HIP SDK (Windows): HIP runtime, hipRTC kernels | first slice on `backend/hip`, untested on real hardware | the third backend family (after CUDA and Vulkan): memory and copies on the device, a first kernel set, host fallbacks for the rest; input for the public backend API |

Plug-in points that are still closed, and the dataset loader abstraction: [plug-in.md](plug-in.md).
Decisions and hardware checks waiting for the maintainer: [open-questions.md](open-questions.md).
One command-line tool for everything (`idrak doctor`, `chat`, `serve`, `bench`, ...): [idrak-cli.md](idrak-cli.md).

## Not supported yet

What the `architecture` branch cannot do today, with an example of each. "High" marks what comes next.

### Devices and backends

| Priority | Not supported | Example |
|---|---|---|
| High | **AMD ROCm / HIP** (first slice, untested) | `hip:0` exists on the `backend/hip` branch (plan 8) but has never run on an AMD GPU; only memory, copies and a first kernel set run on the device, the rest through host fallbacks; no matrix cores (rocWMMA / MFMA), no hipGraph |
| High | **NPUs** (Intel AI Boost, Apple Neural Engine) | no device kind for them; a model cannot run there (add-on packages `Idrak.OpenVino`, `Idrak.CoreML`: plans 4 and 5) |
| | A backend added from outside the library | `DeviceProviders.Register(new MyProvider())` does not compile in an application: `Backend` and `DeviceProviders` are internal (plan 7, phase 6) |
| | Apple GPUs (Metal) | on a Mac `Device.Default` is the CPU; there is no `Device.Get("metal")` (plan 5) |
| | One model across several devices | `model.To(Device.Cuda(0))` places the whole model on one device; it cannot be split across `cuda:0` and `vulkan:1` (plan 6) |

### Vulkan (works, with limits)

| Limit | Example |
|---|---|
| Not the default device | on a machine whose only GPU is integrated, `Device.Default` is the CPU unless `IDRAK_VULKAN_DEFAULT=1` |
| Recorded steps are always used where they can be, even where replaying them is slower (see "Future improvements") | on a Quadro RTX 3000 the bench decoder ran at 639 tokens/s with graphs and 779 without |
| Matrix units only through `VK_KHR_cooperative_matrix` float16 shapes (float32 and prompt products, measured per shape) and, with `MixedPrecision`, bfloat16 shapes (single-pass reduced precision); no int8 or float8 matrix types, no attention or fused training kernels on them | a device without the extension (or with only integer shapes) runs the float32 kernels |
| Training operations on the host fallback (convolutions, group norms, 8-bit AdamW, training attention's backward) | training a CNN on `vulkan:1` works but each convolution round-trips to the CPU |
| Tensors larger than the device's storage range | on a GPU reporting a 1 GiB `maxStorageBufferRange`, a 151,936 × 4096 float weight runs on the host fallback |
| No fused decoder kernels (packed q/k/v, gate/up, add-and-normalize in one dispatch) | about 195 dispatches per decoded token on an 8-layer int8 model, fewer on CUDA |

### Plug-ins (work, with limits)

| Limit | Example |
|---|---|
| Custom ONNX operators apply to chain models only | `OnnxImportOps.Register("MyOp", ...)` is not used for a model with skip connections (imported as a graph) |
| Outside packed-weight formats take the general path | `PackedWeight.Register("nf4", ...)` works on every device, without fused kernels and without CUDA graphs unless its `MatMul` stays on the device |
| Outside KV cache formats attend through the composed fallback | `KeyValueLayouts.Register("fp8", ...)` works, slower than the built-in int8 cache |

### Not tested yet (expected to work)

| Platform | Status |
|---|---|
| Linux with a GPU (CUDA or Vulkan) | partly: CUDA under WSL2 (RTX 5070 Ti, all pass) and Mesa's software Vulkan driver; native Linux GPU drivers (NVIDIA, RADV, ANV, NVK) never run: WSL2 only for now (a live USB or a cloud GPU instance would cover them; guide in installation/linux.md) |
| macOS (CPU) | done: an Apple M4 Max passed the whole list on the CPU backend (README, "Tested on architectures") |
| NVIDIA before compute 8.6 | partly: a Quadro RTX 3000 (Turing, compute 7.5) passed the whole list on CUDA and Vulkan, also with tuning off; Maxwell and Pascal (5.x, 6.x) never run |
| Intel Arc (XMX), AMD RDNA | no such GPU tested |
| Android GPUs (Adreno and others, through Vulkan) | partly: an Adreno 730 passed the whole list through Mesa's Turnip (KGSL build) in Termux + proot Ubuntu, and the CPU (ARM64, NEON) too; Qualcomm's own driver, Mali GPUs and a .NET Android app (the loader now names libvulkan.so there) not yet |

## Next abstraction

[10-abstractions.md](10-abstractions.md): `Idrak.Abstractions`, one package with every contract and its default
implementation, the public device API (item 12c) and plan 9's dispatcher; CUDA, Vulkan and HIP become packages built
only on that public API. [9-operations.md](9-operations.md) (operations as data) is part of it. Planned, not started.

## Future improvements (measured)

Found on real hardware; each one is a measurement to act on, not a guess.

### Vulkan: recorded graphs slower than direct steps on some GPUs

On a Quadro RTX 3000 (Turing, compute 7.5, NVIDIA driver 595.95, Windows 10, release 0.3.1, 2026-10-05) the
`--bench-vulkan` medium decoder (dim 1024, 8 layers, int8 weights) ran at **639 tokens/s with recorded graphs** (9
dispatches per token) and **779 tokens/s without** (83 dispatches per token), 22% faster without. On the same laptop's
Intel UHD Graphics the two were within a few percent (117.5 and 124.5), while on Mesa's software driver graphs cut the
host time per step from about 300 µs to 5 µs. So whether replaying a recorded step pays off depends on the device and driver, while today a
graph is used wherever one can be recorded.

- **What to do:** measure it, like the other runtime choices. When a decoding graph is first recorded, time a few
  replays against the same number of direct steps and keep the faster, stored per device, driver and power source in
  the tuning cache (as the kernel width, batch size and in-flight count are), with `IDRAK_VULKAN_GRAPHS=0/1` to force
  either. The fused kernels stay in both paths; only the replay is in question.
- **What to look at first:** whether the slowdown comes from `vkCmdExecuteCommands` of secondary command buffers on
  this driver (a replay as primary command buffers, or one secondary buffer per graph, may avoid it), or from the
  barriers recorded into the graph, which cannot be dropped at replay even when the next step does not need them.
- **How to check:** `idrak test -d vulkan:0 -- --bench-vulkan` prints both numbers ("again; without graphs") on every
  device; the choice is right when the kept path is never the slower one on any GPU of the README's table.

## The rule every plan keeps

The core library has no native dependencies: the CUDA backend calls the NVIDIA **display driver** and generates its
kernels (PTX) in C#. A new backend should do the same where it can: talk to what the vendor's **driver** installs, and
generate its kernels in C# (PTX for NVIDIA, SPIR-V for AMD and Intel, Metal source for Apple). Where that is impossible (NPUs), the work goes into an optional add-on package, and the core
stays dependency-free.

## Rule: no card-specific tuning (mostly done in 0.1.6)

The library must run well on **every** card, not on the one it was measured on. Before 0.1.6, several rules were
tuned with `--bench-gemm` / `--bench-gemv` on one RTX 5070 Ti (compute 12.0, 70 SMs); the few-row rule in 0.1.5 was
1.2–1.4× slower on an RTX 3060 (compute 8.6). The rule:

- **No card names in decisions.** A heuristic reads only what the device reports: compute capability, SM count, shared
  memory per block and per SM, L2 size, memory size, bus width and clocks (`cuDeviceGetAttribute`), and the matrix
  formats it supports. A card name may appear in a comment as *where it was measured*, never as the reason.
- **Relative, not absolute, sizes.** Limits scale with the device: blocks per SM, a fraction of total memory, a
  fraction of L2. Fixed byte counts are a smell.
- **When no rule wins everywhere, measure on the user's card.** Since 0.1.6 (`CudaBackend.Tuning.cs`): the first time
  a shape needs a choice, every candidate runs on the real inputs, back to back and in turns (forwards, then backwards)
  over seven rounds, and the candidate with the fastest **median** is kept (`TuneTiming`; until 0.1.7 the best of five,
  which one lucky timing could decide). The device-count formula is the default while measuring is impossible (a graph
  is being recorded, the profiler runs, `IDRAK_AUTOTUNE=0`) and wins unless a candidate's median is 3% faster. Measured
  choices are **kept per GPU** (name, compute capability, SM count, memory), driver (CUDA version and release) and library
  build (version and a hash of the kernels) in `IDRAK_CACHE` (default `~/.cache/idrak`) under `tuning/cuda/`, one text
  file per GPU with a format version; a file of another identity is ignored and rewritten, and a kept value is used only
  while it is still one of the shape's candidates (`TuningCache`; `IDRAK_TUNING_CACHE=0` turns it off,
  `IDRAK_TUNING_CACHE=<folder>` moves it). Every tuned value keeps a benchmark override (`GemvSplits`,
  `TensorSplitsOverride`, `DecodeSplits`, ...) for tests.
- **Checked on several cards across generations.** The README's "Tested on" table records the runs. A choice may not
  be more than 5% slower than the best fixed one on any tested card.
- **Other backends follow the same rule.** AMD, Intel and Apple (plans 3–5) read their own device limits through
  `Backend.Capabilities` (shared work step 1) and use the same measure-on-first-use approach, so no plan copies NVIDIA
  numbers. The CPU does too: its choices come from the cores, vector width, registers and caches the machine reports, or
  are measured on it (`CpuTuning`; the audit is in [plan 2](2-cpu.md#4-machine-agnostic-audit)).

### Status of each rule

Every constant, threshold and heuristic in `src/Idrak/Backends/Cuda` that decides a kernel, split, tile, row cut-over,
block size or buffer size, classified: **(a)** a hardware/ISA limit or fixed by the kernel's own design (fine, reason
given), **(b)** measured on the device or relative to what it reports (fine), **(c)** violates the rule (fixed here, or
left open with the reason).

| Rule (file) | Before 0.1.6 | Now | Class / left to do |
|---|---|---|---|
| Few-row int8 rows → GEMV or tensor cores (`PrefersPackedMatMul`) | 4–8 rows to tensor cores on every tensor-core GPU (tuned on 5070 Ti) | ✅ measured per shape (median), kept per GPU | (b); an RTX 3060 and an Ada (8.9) run |
| Few-row k splits (`GemvSplitCount`, `GemvMultiSplits`) | ~2 blocks per SM, power of two | ✅ measured per shape (powers of two up to k/64, at most 64: the kernels' 64-row unrolled step) | (b) |
| Tensor-core k splits (`TensorSplits`) | ~16 blocks per SM, ≤ 8 splits | ✅ measured per shape (1, 2, 3, 4, 6, 8, 12, ... up to k/128) | (b) |
| Prompt k splits (`PromptSplits`, multi and low-rank products) | none at 85–100% of a wave | ✅ measured per shape below 4 waves of tiles (relative to the SM count) | (b) |
| Fused activation (`PackedMatMulGated`) | int4 only | ✅ measured per shape, every format | (b) |
| 64 / 128 tiles without tensor cores (float and packed) | by SM count | ✅ measured per shape | (b); a run on a GPU without tensor cores |
| Decoding-attention splits (`DecodeSplit`) | ~5 blocks per SM | ✅ measured per shape (kernel, rows, capacity, head size) **at a full cache** (a scratch position at the last row; the formula is the default until then) | (b) was (c) until 0.1.7 |
| Measured choices per process | measured at every start | ✅ kept per GPU + driver + library build (`TuningCache`) | (b) |
| Timing noise | best of 5 rounds | ✅ median of 7 rounds, alternating order, 3% margin over the formula | (b) |
| GPU memory reserve (`GpuMemoryReserve`) | 512 MiB fixed | ✅ 1/16 of the GPU's memory, at least 256 MiB (the driver's own context, not a card property) | (b) |
| Offload headroom (`OffloadReturnHeadroom`) | 1 GiB fixed | ✅ 1/8 of the GPU's memory, at least 256 MiB | (b) |
| Reduction grid (`Sum`) | – | at most 8 blocks per SM (grid-stride loop) | (b) relative to the SM count |
| Prompt row tile 64 / 128 (`PromptTileRows`) | by tile fill | the last 128-row tile at most half full → 64 | (a) geometry of the prompt, not of the card |
| Row cut-over few rows → prompt products (`GemvRows` = 8) | – | the GEMV kernels keep 8 rows in registers | (a) kernel design; 4–8 rows measured against tensor cores (above) |
| Float GEMV for ≤ 8 rows when n·k ≥ 65536 (`BatchedMatMul`) | – | smaller products keep the 16 × 16 kernel so a small model's results do not depend on the batch | (a) a numerical-consistency rule, the same on every card |
| Skinny products to tensor cores (side 8–63, `BatchedMatMul`) | – | by size (≥ 2²⁴ multiply-adds or k ≥ 2048) | (a) which products MixedPrecision moves to bfloat16: a precision rule that must not depend on timing |
| Tensor cores / FP8 / INT8 available | – | compute capability ≥ 8.0 / 8.9 / 8.0 as reported; a module the driver rejects is skipped | (a) ISA |
| Head-size limits (`DecodeMaxDim` 256, `FlashMaxDim` 128, flash tensor-core 64/128) | – | 8 values per lane × 32 lanes; tiles in registers and shared memory | (a) kernel design |
| Kernel shapes (block sizes, tiles, stages, rows in registers) | written into the kernels | ✅ a `KernelShapes` record derived from `CudaDeviceLimits` (cuDeviceGetAttribute); PTX generated per distinct shapes; geometry checked against the reported limits | (a)/(b), see "Kernel shapes" below |
| Grid z limit (`BatchedMatMul`) | 65535 fixed | ✅ as reported (`MAX_GRID_DIM_Z`) | (a) |
| Top-k sampling (`CandidateSlice` 2048, `CandidateSlots` 64; one block per row when the vocabulary ≤ 4 slices) | – | kernel design (slots per block, top-k ≤ 64 takes the candidate pass) | (a) |
| Split counters (≥ 4096), JIT log (16 KiB), grid z (65535) | – | buffer sizes / the CUDA grid limit | (a) |
| Pool best fit (a cached block ≤ 25% larger, from 1024 floats) | – | relative to the request | (b) host-side allocator policy |
| Async upload size (`AsyncUploadBytes` 4 MiB, 16 MiB staging ring) | fixed | host-side pinned memory: no GPU attribute applies | ⚠️ confirm on PCIe 3 / 4 / 5; measure if a run shows a difference |
| CPU: parallel cut-overs, element-wise and products (`CpuTuning.ParallelElements` / `ParallelFlops`) | 65,536 / 131,072 fixed | ✅ measured at first use (median of 7 rounds, parallel at least 10% faster; element-wise over windows larger than the L2, the product on one row against a square B), the median of three passes after an untimed one, kept per machine, runtime and thread count in ~/.cache/idrak/cpu/tuning.tsv (`IDRAK_CACHE`; `IDRAK_TUNING_CACHE=0` off); fallback L2 / 16 and L2 / 8 | an ARM64 and an AVX-512 run |
| CPU: tiled product kernel (`CpuTuning.Kernel`) | AVX-512 8×32 or AVX2 6×16 by vector width; none on ARM64 | ✅ by reported instruction sets: AVX-512 8×32, AVX2+FMA 6×16, NEON 8×8 (new) | NEON speed on a real ARM64 machine |
| CPU: product tiles (`TileBytes`, `TileGrid`, `MaxRowBlocks`) | A tile 256 KB, at most 16 row blocks, ~2 tiles per thread | ✅ A tile = L2 / 4, at most L1 / 2 KiB row blocks, ≥ 2 tiles per thread in whole rounds (a multiple of the threads) | – |
| CPU: column blocks, transpose tile, k chunk, register rows | 64 columns, 32 × 32, –, – | ✅ 8 vectors; L1-sized tiles; half a cache line; rows that fit the vector registers | the k chunk is a formula; a first-use measurement if machines disagree |
| CPU: summation-order switches (`ReductionElements`, `TransposedInPlace`, `FewRows`, column-split rule with beta ≠ 0) | 65,536, 16, 8 (CUDA's `GemvRows`), 4 / 128 / 131,072 | fine: a numerical contract (the CPU is the reference), the same on every machine, not tuning | – |
| CPU: attention head-size limits, few-row split (`Capabilities`) | CUDA's `DecodeMaxDim` 256 / `FlashMaxDim` 128 / `GemvRows` 8 | ✅ the CPU's own constants (`CpuTuning.DecodeAttentionHeads`, `TiledAttentionHeads`, `FewRows`), same values: they choose the attention / product algorithm, so a numerical contract | lifting the head limits (the CPU kernels take any size) changes results for larger heads: opt-in only |

### Kernel shapes

Every CUDA device's limits are read once (`CudaDeviceLimits.Read`: compute capability, SMs, warp size, threads per block
and per SM, shared memory per block / with opt-in / per SM, registers per block and per SM, L2, bus width, clocks, grid
limits, memory) and a `KernelShapes` record is derived from them; the main PTX module is generated for that record
(`PtxKernels.SourceFor`, one text per distinct record) and the C# launches read the same record. Two kinds of shape:

- **Shapes the kernels take at any value**, relative to what the device reports. On every CUDA GPU so far (1024 threads
  per block, 32-lane warps) they come out at the sizes the kernels always had, so the PTX on the RTX 5070 Ti (12.0,
  70 SMs), RTX 5050 Laptop (12.0, 20 SMs) and RTX 3060 Laptop (8.6, 30 SMs) is byte for byte what it was (a test checks
  the derivation for 5.0–12.x as documented, and on the GPU it runs on).
- **Geometry a kernel is written for** (tiles, warps per tile, pipeline stages, rows kept in registers): fixed by the
  kernel's code; the device's limits judge whether it can run. A tensor-core module whose static or opt-in shared memory
  exceeds what a block may use is not loaded (`PtxKernels.Fits`, read from the PTX's `.shared` arrays plus the dynamic
  shared memory each kernel asks for) and its operations take the other path; a GPU that cannot run the main module
  (warps other than 32 lanes, blocks under 1024 threads, too little shared memory) gets a clear reason instead of a
  failed launch. Where two geometries exist, the faster is measured per shape (64 / 128 tiles, 64 / 128-row tiles).

| Shape | Kind | Formula / reason | 5070 Ti · 5050 · 3060 |
|---|---|---|---|
| `BlockSize` (1-D kernels, sum, group statistics, column sums) | relative | a quarter of the largest block (≥ 4 resident blocks per SM on every generation), power of two | 256 · 256 · 256 |
| `SamplerThreads` | relative | the largest block | 1024 · 1024 · 1024 |
| Per-warp scratch of the row kernels (`_rv`, `_ri`) | relative | warps in the largest block | 32 · 32 · 32 |
| Grid z limit | reported | `MAX_GRID_DIM_Z` | 65535 · 65535 · 65535 |
| Warp of 32 lanes | ISA | PTX shuffles and lane masks are written for 32 lanes; checked | – |
| `GemvThreads` 1024 (gemv_nn: 32 columns × 32 slices), `softmax_ce_rows` 1024 | geometry | needs 1024 threads per block; checked | fits all three |
| `RowThreads` 256 (8 warps: gemv_nt's column per warp, the decoding attention's partial sums, 8-bit Adam's 256-value blocks) | geometry / data format | the 8-bit Adam block is 256 values | fits |
| `Int8GemvThreads` 512 (32 column words × 16 k slices, unrolled) | geometry | – | fits |
| `GemmThreads` 256, 64 / 128 tiles (8 × 8 / 4 × 4 per thread) | geometry | tile measured per shape | fits |
| `Tile` 16 (16 × 16 threads) | geometry | – | fits |
| `TensorTile` 128, `TensorThreads` 256, `TensorK` 32, two register-staged stages (40 KiB static) | geometry | `mma.sync.m16n8k16` (ISA), 8 warps of 64 × 32 | fits (≤ 48 KiB) |
| 8-bit products: k 64 × 3 cp.async stages (60 KiB dynamic) | geometry | needs opt-in shared memory; checked | fits (99 KiB) |
| Flash tensor-core: 64 query rows, 32 backward rows; d128 backward 68 KiB dynamic | geometry | checked | fits (99 KiB) |
| `FlashTile` 32, `FlashMaxDim` 128, `DecodeMaxDim` 256 (8 values per lane × 32 lanes) | geometry | head-size limits follow from registers per lane | – |
| `GemvRows` 8 | geometry | rows kept in registers; more rows go to the prompt kernels | – |
| `CandidateSlice` 2048, `CandidateSlots` 64 | geometry | top-k ≤ 64 takes the candidate pass | – |
| `Int4Group` 32, `MaxPermuteRank` 6 | data format / API | the stored int4 format; the permute API | – |
| Staging ring 16 MiB, async uploads ≤ 4 MiB, pool best fit 25% / 1024 floats | host side | no GPU attribute describes host memory or the PCIe link | ⚠️ measure if a PCIe 3 / 4 / 5 run shows a difference |

Architecture-specific paths are chosen by the reported compute capability, never a card name: bfloat16 tensor cores
(`mma.sync`, `ldmatrix`, `cp.async`) and INT8 products from 8.0, FP8 (e4m3) products from 8.9. Candidates for later,
each to be measured against today's path on a GPU that has it (not added here: they cannot be validated without one):
- 8-bit products with 4 cp.async stages where a block may opt in to ≥ 80 KiB (8.0, 9.0, 10.x), measured against 3.
- bfloat16 products through cp.async multi-stage pipelines (8.0+) instead of register staging.
- `wgmma` with TMA on 9.0 (`sm_90a`) for products and flash attention; `tcgen05` / tensor memory on 10.x (`sm_100a`);
  block-scaled FP8 / FP6 / FP4 `mma` on 12.x.
- A smaller tensor tile for GPUs with few SMs, measured per shape like the 64-row tile.
- A per-kernel register check at load (`cuFuncGetAttribute` max threads per block) against each kernel's launch size.
- An L2 persistence window sized from the reported L2 for the KV cache during decoding (8.0+).

**Open:**
- Re-run `--bench-gemv` on the RTX 3060 to confirm "auto" is within 5% of the best column there, as it is on the
  RTX 5050 and the RTX 5070 Ti (0.1.7: every decoding and prompt-sized row within about 5%), now including the
  attention rows (auto measured at a full cache).

## Shared work before any new backend

The backend design is device-neutral in one place (`src/Idrak/Backends/Backend.cs`) and NVIDIA-specific in others. These steps come first, because both the AMD
and the Intel plans need them.

1. **A device-neutral capability query.** `MixedPrecision.TensorCoresUnavailable` casts to `CudaBackend`, and several
   callers check `PtxKernels` constants (`GemvRows`, `DecodeMaxDim`, `FlashMaxDim`). Replace them with a
   `Backend.Capabilities` record (matrix units and their number formats, subgroup size, shared memory per block,
   compute units, the few-rows limit, attention head-size limits). Done when no code outside `Backends/Cuda` names
   `CudaBackend` or `PtxKernels`.
2. **`DeviceType` beyond `Cpu` and `Cuda`.** Add the new kinds, `Device.Default` choosing the best device present, and
   an `IDRAK_DEVICES` / `IDRAK_DISABLE_*` switch per backend (the test runner already runs every test on every device).
3. **A kernel set that can be shared.** The 178 PTX kernels are written once per operation. A second kernel language
   (SPIR-V, see the AMD and Intel plans) should be generated from the same description where the operation is simple
   (element-wise, reductions, gathers), so only the hard kernels (matrix products, attention) are written twice.
4. **A minimum backend.** `Backend` has 87 abstract and 34 virtual members. Split them into the minimum every device
   must provide (memory, copies, element-wise, matrix product, reductions, softmax, gather/scatter) and the fused or
   quantized ones that have a composed fallback, so a new backend runs every model early and gets faster step by step.
5. **Tests per device, and benchmarks per device.** Every backend is checked by the same test list it runs on the CPU
   and by `--bench-gemv` / `--bench-cpu`-style timings; each new device row goes into the README's "Tested on" table.

## Order of work

1. **NVIDIA, step 1:** the few-row rule keyed on compute capability (a regression in a shipped release).
2. **CPU, step 1:** entropy without a logarithm per token (a small fix, measured on four machines).
3. **The shared work above:** device-neutral capabilities, more device types, a shared kernel description, a minimum
   backend. Nothing user-visible, but plans 3 and 4 cannot start without it.
4. **AMD, phases 1–5:** the Vulkan backend, built once.
5. **Intel GPUs:** tuning the same backend for Iris Xe and Arc, then integrated-GPU memory.
6. **NVIDIA and CPU, the rest:** the remaining speed work in plans 1 and 2, done between the larger steps above as
   time allows.
7. **Apple:** a test and `--bench-cpu` run on a Mac as soon as one is available (fills the CPU plan's macOS and ARM64
   gaps); the Metal backend after the Intel GPUs.
8. **Idrak.Network:** once steps 1–2 are done, alongside steps 3–5: training over a local network first, for every kind
   of training (machines over TCP, bfloat16 gradients), then several GPUs in one machine, models split across GPUs, and cluster serving.
9. **High priority (see "Not supported yet"): AMD ROCm / HIP and NPUs.** A HIP backend for AMD's own compute stack
   (first slice: [plan 8](8-hip.md)), and the NPU add-ons, Intel (`Idrak.OpenVino`) and Apple's Neural Engine (`Idrak.CoreML`), each a
   one-week trial first.
