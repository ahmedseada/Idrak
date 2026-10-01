# Hardware plans

Plans for running Idrak on more hardware. Nothing here is built yet; each plan says what to build, in which order, how
to know a step is done, and what could stop it.

| Plan | Hardware | Status |
|---|---|---|
| [CPU](cpu.md) | x64 (AVX2, AVX-512) and ARM64 (NEON); the fallback on every machine | supported; plan covers sampling and kernel speed |
| [NVIDIA](nvidia.md) | GeForce, RTX, data-center GPUs (CUDA driver API, PTX) | supported; plan covers tuning and gaps |
| [AMD](amd.md) | Radeon RX 6000/7000/9000, Ryzen APU graphics, Instinct | not supported (runs on the CPU) |
| [Intel](intel.md) | Iris Xe and Arc integrated GPUs, Arc discrete GPUs, AI Boost NPUs | not supported (runs on the CPU) |

## The rule every plan keeps

The core library has no native dependencies: the CUDA backend calls the NVIDIA **display driver** and generates its
kernels (PTX) in C#. A new backend should do the same where it can: talk to what the vendor's **driver** installs, and
generate its kernels in C#. Where that is impossible (NPUs), the work goes into an optional add-on package, and the core
stays dependency-free.

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

## Order

1. NVIDIA plan, step 1 (a measured regression on a shipped release), and CPU plan, step 1 (plain sampling).
2. The shared work above.
3. The AMD and Intel GPU plans share one Vulkan backend: build it once, tune it per vendor.
4. Intel NPU last, as an add-on package.
