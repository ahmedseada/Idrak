# Apple plan

**Status:** not supported on the GPU; Macs run everything on the CPU, and macOS has not been tested yet. Goal: Apple
silicon Macs (M1 to M4 and later) running the same models and tests on their GPU, without making users install
anything beyond macOS.

## The hardware

- **GPU:** Apple's own, on the same chip as the CPU (7–10 cores on a base M-series chip, up to 40 or more on Max and
  Ultra). It shares the system memory with the CPU (unified memory), so a 32 GB Mac can hold models a 16 GB graphics
  card cannot.
- **Neural Engine:** an NPU on every Apple silicon chip, reachable only through Core ML.
- **Intel Macs** (AMD Radeon or Intel graphics) are out of scope: they are a shrinking group, and macOS on them still
  needs Metal, so they would get the same backend at lower priority.

## The choice: how to reach the GPU

macOS has no CUDA and no native Vulkan; its GPU interface is Metal.

| Option | What the user installs | Kernels | Fit with the no-dependency rule | Notes |
|---|---|---|---|---|
| **A. Metal, kernels as source** (recommended) | nothing (Metal ships with macOS) | **Metal Shading Language source generated in C#**, compiled at runtime by `newLibraryWithSource` | full: the same model as PTX for NVIDIA | Metal is an Objective-C API, called from C# through the Objective-C runtime (`objc_msgSend`), with no wrapper library |
| B. Vulkan through MoltenVK | the MoltenVK library | the SPIR-V of the AMD/Intel plans | partial: a translation library to ship | reuses plan 3's kernels, but adds a translation layer and its limits |
| C. Metal Performance Shaders | nothing | Apple's own matrix kernels | full | fast matrix products without writing them; the other operations are still ours (combine with A) |
| D. Core ML add-on | an add-on package | none | as an optional package only | the way to the Neural Engine; inference only |

**Recommendation: A, with C for matrix products where they help.** Generating Metal source in C# follows the PTX
approach, and macOS compiles it on the user's machine, with nothing to install. Metal Performance Shaders can supply
matrix products early, so the backend runs language models before its own matrix kernels are tuned.

🤔 To verify early: calling Metal from .NET through the Objective-C runtime without a binding library (selectors,
autorelease pools, reference counting), and which bfloat16 support each Apple GPU family has (M1 and M2 have FP16
but no bfloat16 arithmetic; bfloat16 arrived with later families).

## Phases (after the shared work in plans/README.md)

### 1. Metal device and memory
- Bind the Objective-C runtime (`libobjc`) and Metal with `LibraryImport`; device, command queue, buffers.
- **Unified memory:** buffers in shared storage mode, so uploads and downloads are plain memory copies or no copy at
  all; the memory limit counts system RAM.
- `DeviceType.Metal`, `Device.Metal()`, and `Device.Default` order on a Mac: Metal, then CPU.
- **Done when** tensors upload, download and pass the memory tests on an M-series Mac.

### 2. Kernel generator and the minimum kernel set
- A C# Metal Shading Language writer: threadgroups, threadgroup memory, SIMD-group operations (32 lanes on Apple GPUs).
- The minimum backend from plans/README.md: element-wise ops, reductions, softmax, gather/scatter, layer/RMS norm,
  attention; matrix products through Metal Performance Shaders first, our own tiled kernel next.
- Many operations per command buffer, committed once (Metal's answer to launch overhead).
- **Done when** every test that runs on the CPU also passes on the Mac's GPU (CPU fallbacks allowed for the rest).

### 3. Language models
- Few-row products through int8/int4/bf16 weights, KV caches, the sampler, decoding attention; prompt attention.
- SIMD-group matrix operations (`simdgroup_matrix`, 8×8 tiles) for the products.
- **Done when** Qwen3-0.6B chats on a Mac with the same greedy tokens as the CPU and the transformers check
  (`check qwen3.json`) passes.

### 4. Tuning and training
- A `--bench-metal` table like `--bench-gemv`; tile sizes from the GPU's core count.
- Backward kernels, AdamW and 8-bit AdamW, LoRA fine-tuning. Unified memory makes fine-tuning larger models practical
  on Macs with more RAM.
- **Done when** the LoRA fine-tuning tests and a short Qwen2.5-0.5B LoRA run pass on an M-series Mac.

### 5. Neural Engine (add-on, last)
- An optional `Idrak.CoreML` package: convert an exported model to Core ML, run it on the Neural Engine behind
  `IPredictor` / `IEmbedder`, as the Intel plan does for its NPU with OpenVINO.
- Start with a one-week trial on a classifier or text encoder; continue only if latency and battery use clearly beat
  the GPU.

## Hardware to test on

| Device | Why |
|---|---|
| MacBook Air M1/M2 (8–16 GB) | the most common Macs; no bfloat16 arithmetic on these families |
| A Mac with M3 or M4 | newer GPU families (bfloat16, faster SIMD-group matrices) |
| Mac with Max or Ultra chip, 32 GB or more | many GPU cores, large models in unified memory |

Even before phase 1, one run of the tests and `--bench-cpu` on any Apple silicon Mac fills the macOS and ARM64 (NEON)
gaps in the CPU plan.

## Risks

- **No GPU in CI and no Mac in the team's hands yet:** every phase needs runs on real hardware.
- **Objective-C from C# without bindings** is unusual and error-prone (memory management); a small, well-tested
  wrapper layer inside the backend contains it.
- **Metal Performance Shaders' coverage** may not include every shape or number format needed; our own kernels must
  cover the gaps.
- **A third kernel language** (PTX, SPIR-V, Metal): the shared kernel description in plans/README.md matters even more.

## Effort (rough)

Phases 1–2: 3–5 weeks; phase 3: 3–4 weeks; phase 4: 3–4 weeks; Neural Engine: 1-week trial, 2–3 weeks for the add-on.
