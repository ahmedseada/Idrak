# HIP plan: a ROCm / HIP backend for AMD GPUs

**Done (2026-10-22).**

**Status:** first slice on the `backend/hip` branch, **untested on real hardware**: it builds, it is inert where HIP is
absent (no runtime, no device: no HIP device, no exception, a reason in `--list-devices`), and every code path except
the GPU itself has run against a host stand-in of the runtime (see "How it was checked"). Plan 3 chose Vulkan for AMD
GPUs (option A) and kept HIP (option B) for later; this is option B's start. HIP is the third backend family after CUDA
and Vulkan, which makes it the input the public backend API (plan 7, phase 6) was waiting for.

## What the first slice does

| Part | What | Where |
|---|---|---|
| Runtime | libamdhip64.so (ROCm, Linux) or amdhip64_N.dll (HIP SDK or graphics driver, Windows), loaded at run time; every entry point resolved by name into a function pointer (required ones missing: a reason, no devices; optional ones: null). Also looked for under `HIP_PATH`, `ROCM_PATH` and `/opt/rocm/lib` | `Idrak.Gpu/Hip/HipRuntime.cs` |
| Devices | `DeviceType.Hip`, devices `hip:0`, `hip:1`, … from `hipGetDeviceCount`; name, memory, compute units, wavefront width, threads per block, shared memory (LDS) per block, grid limits, L2, PCI address and UUID as reported; the compiler target (gcnArchName, with its feature flags) read from the device properties | `HipDeviceLimits.cs` |
| Policy | Listed (a plain test run includes HIP devices), never `Device.Default` unless `IDRAK_HIP_DEFAULT=1` (then after CUDA, before Vulkan; discrete before integrated, as reported); `IDRAK_DISABLE_HIP=1` hides them | `HipProvider` in `HipBackend.cs` |
| Backend | The minimum backend: a caching allocator (exact sizes, a reserve of 1/16 of memory as on CUDA), uploads and downloads, device-to-device copies, fills (`hipMemsetD32Async`), strided copies (`hipMemcpy2DAsync`), all on one stream; every other operation through the host fallbacks | `HipBackend.cs` |
| Kernels | HIP C++ compiled at first use by hipRTC (libhiprtc.so / hiprtcMMmm.dll) for the device's own target: add, sub, mul, axpy, affine, multiply-add, row-vector add, ReLU, sigmoid, tanh, square, abs, exp, log, RMS norm (with and without gain) and the int8 product (any row count). No hipRTC, a compiler error, implausible limits or `IDRAK_HIP_KERNELS=0`: those operations take the host fallback too | `HipKernels.cs`, `HipRtc.cs`, `HipBackend.Kernels.cs` |
| Cache | Code objects kept in `IDRAK_CACHE` (default `~/.cache/idrak`) under `hip/kernels/`, keyed by a hash of the source, the options (block size, target), the target, the device's identity, the runtime, driver and compiler versions and the library version; a file the runtime refuses is deleted and compiled again; `IDRAK_HIP_KERNEL_CACHE=0` off, `=folder` elsewhere | `HipKernelCache.cs` |

Which operations run on the device: memory, copies, fills, strided copies and the kernels above. Everything else
(matrix products, softmax, attention, sampling, embeddings, the fused and packed decoding paths, training steps) runs
through the host fallback, correct and slow: each call copies its operands to the CPU and back.

### Card-agnostic

- The block size is a quarter of the largest block (the CUDA rule), a power of two, at least one wavefront, within the
  block's x limit and with the reductions' scratch within the reported shared memory (`HipKernels.BlockSizeFor`).
- No kernel assumes a wavefront width (64 or 32 lanes, depending on the GPU): reductions go through shared memory.
- The grid follows the reported limits (element-wise kernels stride over a grid capped at `maxGridDimX`; row kernels
  and the int8 product fall back when rows exceed the grid).
- Reported limits are checked for plausibility (a power-of-two wavefront, threads per block at least one wavefront,
  shared memory for the scratch, non-zero grids): a runtime that numbers its attributes differently turns the kernels
  off instead of launching them with wrong sizes.
- The target string is passed to the compiler and used in cache keys, never in a decision; a test forbids vendor ids,
  card names and GPU target literals in the code of `src/Idrak.Gpu/Hip`, as for Vulkan.

### How it was checked (no AMD GPU here)

- The kernel source compiles to AMDGPU code objects with clang 18 (`-x hip --offload-arch=...` with a stand-in for what
  hipRTC pre-includes) for gfx906, gfx90a, gfx942, gfx1030, gfx1100 and gfx1201 at block sizes 64, 256 and 1024; each
  kernel's metadata names it, with the declared block bound.
- The same source, compiled for the host with each block's threads as real threads meeting at a barrier, matches a
  reference for every kernel (block sizes 64 and 256; rows shorter and longer than a block).
- A host stand-in for libamdhip64 / libhiprtc (host memory, memcpy, kernels run from the HIP source) ran the HIP tests
  and the whole test list on `hip:0` (280 of 280): entry points, argument marshalling, grid and block sizes, cache hits.
- None of this is a GPU: timings, the runtime's real behaviour and the compiler's real output are unknown until a run on
  AMD hardware.

## Risks

- **Untested**: the attribute numbering (`hipDeviceAttribute_t`, the CUDA-compatible range as since ROCm 5.0) is checked
  for plausibility only; the architecture string is found by its content in the properties struct (its offset moved in
  6.0); `hipModuleLaunchKernel` gets each argument in a 64-bit slot (the runtime copies each parameter's size from it).
- **Toolkit**: hipRTC ships with ROCm and the HIP SDK, not with the display driver alone (Windows' Adrenalin driver
  installs the runtime but not the compiler: memory and copies on the GPU, every kernel operation on the host).
- **Supported GPUs**: ROCm supports a list of GPUs per release; others report no devices or fail at `hipInit`, which
  shows as "no devices" with the runtime's reason.
