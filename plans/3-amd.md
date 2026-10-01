# AMD plan

**Status:** not supported; AMD GPUs run everything on the CPU. Goal: Radeon RX 6000/7000/9000 (RDNA 2–4), Ryzen
APU graphics (RDNA 2/3) and, later, Instinct accelerators (CDNA) running the same models and tests as NVIDIA GPUs,
without making users install a toolkit.

## The choice: how to reach the GPU

NVIDIA's driver accepts PTX (a portable assembly) and compiles it for the installed GPU. AMD's drivers have no
equivalent for HIP: HIP loads **finished machine code per GPU family** (gfx1030, gfx1100, gfx1201, …) or compiles HIP
C++ at runtime with hipRTC, which is part of the HIP SDK / ROCm rather than the display driver.

| Option | What the user installs | Kernels | Fit with the no-dependency rule | Speed ceiling |
|---|---|---|---|---|
| **A. Vulkan compute** (recommended) | the display driver only (Windows: Adrenalin; Linux: Mesa RADV, in every distribution) | **SPIR-V generated in C#**, compiled by the driver for the installed GPU | full: the same model as PTX | high; matrix units through `VK_KHR_cooperative_matrix` (RDNA 3 and newer) |
| B. HIP driver API + hipRTC | the HIP SDK (Windows) or ROCm (Linux) | HIP C++ source strings compiled at runtime | partial: a vendor toolkit, like requiring the CUDA Toolkit | highest (rocBLAS-class kernels possible) |
| C. HIP + our own AMDGPU code objects | the driver's HIP runtime | machine code per GPU family, assembled by us | full, but needs an assembler and ELF writer per family | highest, at a very high cost |
| D. DirectML / ONNX Runtime add-on | an add-on package | none (vendor operators) | as an optional package only | good for inference, no training |

**Recommendation: A.** One Vulkan backend also serves Intel GPUs (see the Intel plan) and could run on NVIDIA where CUDA
is unavailable. SPIR-V is a binary format with a fixed instruction list; generating it in C# is the same kind of work
as the PTX generator. B can follow later as an optional `Idrak.Rocm` package for Instinct servers.

🤔 To verify early: which Adrenalin and RADV versions expose `VK_KHR_cooperative_matrix` with bfloat16 or FP16 inputs on
RDNA 3/4; whether RDNA 2 (no matrix units) is fast enough with plain FP32 products to be worth enabling.

## Phases

### 1. Vulkan device and memory (after the shared work in plans/README.md)
- Bind `vulkan-1.dll` / `libvulkan.so.1` with `LibraryImport`, as `CudaDriver` binds `nvcuda`.
- Instance, physical-device choice (discrete before integrated), one compute queue, a buffer pool like the CUDA one
  (exact-size reuse, release under a memory limit), host-visible staging for uploads and downloads.
- `DeviceType.Vulkan`, `Device.Vulkan(n)`, and `Device.Default` order: CUDA, then Vulkan, then CPU.
- **Done when** tensors upload, download and survive the memory tests on an AMD GPU.

### 2. SPIR-V generator and the minimum kernel set
- A C# SPIR-V writer (types, buffers, workgroup memory, subgroup operations, `GLSL.std.450` math).
- The minimum backend from README.md: element-wise ops, reductions, softmax, gather/scatter, a tiled FP32 matrix
  product, layer/RMS norm, attention without fused tricks.
- Command buffers recorded per operation first; batching many dispatches per submit next (Vulkan's equivalent of
  launch overhead).
- **Done when** every test that runs on the CPU also passes on the AMD GPU (CPU fallbacks allowed for the rest).

### 3. Language models
- Decoding: few-row products (GEMV) through int8/int4/bf16 weights, KV caches, the sampler, decoding attention.
- Prompt: tiled products, flash-style attention.
- Wave size: RDNA runs 32 or 64 lanes per wave; kernels take the subgroup size from the device.
- **Done when** Qwen3-0.6B chats on a Radeon with the same tokens as the CPU (greedy) and the transformers check
  (`check qwen3.json`) passes.

### 4. Matrix units and tuning
- Cooperative-matrix products (FP16/bfloat16 in, FP32 out) on RDNA 3/4; int8 where exposed.
- A `--bench-vulkan` table like `--bench-gemv`; tile sizes and splits from the device's compute-unit count.
- Recorded command buffers replayed per decoding step (Vulkan's equivalent of CUDA graphs).

### 5. Training
- Backward kernels, the fused AdamW / 8-bit AdamW, LoRA fine-tuning; checkpointing and memory fallbacks as on CUDA.
- **Done when** the LoRA fine-tuning tests and a short Qwen2.5-0.5B LoRA run pass on an RDNA 3 GPU.

## Hardware to test on

| Device | Why |
|---|---|
| Radeon RX 7600 / 7800 XT (RDNA 3) | matrix units, the main target |
| Radeon RX 6600 / 6700 (RDNA 2) | no matrix units: the plain FP32 path |
| Ryzen 7040/8040 laptop (Radeon 780M, RDNA 3 iGPU) | shared memory with the CPU; the most common AMD GPU |
| Radeon RX 9070 (RDNA 4) | newest matrix formats |

Both Windows (Adrenalin) and Linux (Mesa RADV) for each, since the two drivers differ.

## Risks

- **Largest piece of work in these plans:** a second kernel language and a second backend; most of the 178 kernels
  need a SPIR-V version (the shared kernel description in README.md reduces this).
- **Driver differences:** RADV and AMD's Windows driver expose different extensions and bugs; both need test runs.
- **Speed:** Vulkan compute without vendor libraries may trail ROCm/rocBLAS on matrix products; the target is "much
  faster than the CPU" first, "close to ROCm" later.
- **AMD NPUs (Ryzen AI / XDNA):** out of scope here; like Intel's NPU they only run networks compiled by the vendor's
  toolchain, so they would need an add-on package.

## Effort (rough)

Phases 1–2: 3–5 weeks; phase 3: 3–4 weeks; phase 4: 2–3 weeks; phase 5: 3–4 weeks. Shared with the Intel GPU plan:
phases 1–2 and most of 3.
