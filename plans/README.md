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
  a shape needs a choice, every candidate runs on the real inputs, back to back and in turns over five rounds, and the
  fastest is kept for the process. The device-count formula is the default while measuring is impossible (a graph is
  being recorded, the profiler runs, `IDRAK_AUTOTUNE=0`) and wins unless a candidate is 3% faster. Every tuned value
  keeps a benchmark override (`GemvSplits`, `TensorSplitsOverride`, ...) for tests.
- **Checked on several cards across generations.** The README's "Tested on" table records the runs. A choice may not
  be more than 5% slower than the best fixed one on any tested card.
- **Other backends follow the same rule.** AMD, Intel and Apple (plans 3–5) read their own device limits through
  `Backend.Capabilities` (shared work step 1) and use the same measure-on-first-use approach, so no plan copies NVIDIA
  numbers.

### Status of each rule

| Rule (file) | Before 0.1.6 | Since 0.1.6 | Left to do |
|---|---|---|---|
| Few-row int8 rows → GEMV or tensor cores (`PrefersPackedMatMul`) | 4–8 rows to tensor cores on every tensor-core GPU (tuned on 5070 Ti) | ✅ measured per shape | an RTX 3060 and an Ada (8.9) run |
| Few-row k splits (`GemvSplitCount`) | ~2 blocks per SM, power of two | ✅ measured per shape (powers of two) | – |
| Tensor-core k splits (`TensorSplits`) | ~16 blocks per SM, ≤ 8 splits | ✅ measured per shape (1, 2, 3, 4, 6, 8, 12, ...) | – |
| Prompt k splits (`PromptSplits`) | none at 85–100% of a wave | ✅ measured per shape | – |
| Fused activation (`PackedMatMulGated`) | int4 only | ✅ measured per shape, every format | – |
| 64 / 128 tiles without tensor cores | by SM count | ✅ measured per shape | a run on a GPU without tensor cores |
| GPU memory reserve (`GpuMemoryReserve`) | 512 MiB fixed | ✅ 1/16 of the GPU's memory, at least 256 MiB | – |
| Offload headroom (`OffloadReturnHeadroom`) | 1 GiB fixed | ✅ 1/8 of the GPU's memory, at least 256 MiB | – |
| Decoding-attention splits (`DecodeSplit`) | ~5 blocks per SM | ⚠️ still the formula: its time depends on how full the cache is, so a first-call measurement would mislead | measure at a full cache (e.g. an `--autotune` pass) |
| Prompt row tile 64 / 128 (`PromptTileRows`) | by tile fill | fine (geometry, not hardware) | – |
| Async upload size (`AsyncUploadBytes`, 16 MB staging ring) | fixed | fine as a host-side size | confirm on PCIe 3 / 4 / 5 |

**Open:**
- Measured choices last for one process; caching them per GPU, compute capability and driver in the user's cache
  folder would save the few milliseconds of measuring at each start.
- Re-run `--bench-gemv` on the RTX 3060 to confirm "auto" is within 5% of the best column there, as it is on the
  RTX 5050 and the RTX 5070 Ti (0.1.7: every decoding and prompt-sized row within about 5%).

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
9. **NPUs last:** Intel (`Idrak.OpenVino`) and Apple's Neural Engine (`Idrak.CoreML`), each a one-week trial first.
