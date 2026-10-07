# Idrak.Gpu

The GPU devices of [Idrak](https://www.nuget.org/packages/Idrak): CUDA, Vulkan and HIP, with all their kernels. No CUDA
Toolkit, no shader compiler, no native files in the package: the kernels are generated in C# (PTX for NVIDIA, SPIR-V
for Vulkan, HIP C for AMD) and compiled by the GPU's own driver on the user's machine.

## Install

```bash
dotnet add package Idrak       # brings Idrak.Gpu along: every device, as before
dotnet add package Idrak.Gpu   # the GPU devices alone, on Idrak.Abstraction
```

The package depends on `Idrak.Abstraction` only and is built on its public device API (`Backend`, `Storage`,
`DeviceProvider`, `Kernels`), the same surface an outside device package would use. Once an application ships it,
`Device.Available` lists its devices and `Device.Parse("vulkan:0")` reaches them; no call is needed to turn them on.

## What is in the package

| Device | How it works | Needs |
|--------|--------------|-------|
| CUDA (`cuda:N`) | The NVIDIA driver API directly; PTX kernels generated in C#; bfloat16 and FP8 tensor cores, flash attention, CUDA graphs, memory offloading | An NVIDIA display driver (`nvcuda.dll`, `libcuda.so.1`) |
| Vulkan (`vulkan:N`) | SPIR-V kernels generated in C# for NVIDIA, AMD, Intel and phone GPUs; fused decoding, recorded graphs, cooperative matrices; widths and splits measured on the device | A Vulkan 1.1+ driver |
| HIP (`hip:N`, first slice) | HIP C kernels compiled on the device with hipRTC and cached; the rest through host fallbacks | ROCm or the HIP SDK |

Every operation a device has no kernel for falls back to the CPU device of `Idrak.Abstraction`, so a model runs on any
of them unchanged. Kernel choices come from what each device reports or from measurements on it, stored per device and
driver, never from a card's name. `Device.Default` picks the first CUDA GPU when there is one; a Vulkan or HIP GPU
becomes the default only with `IDRAK_VULKAN_DEFAULT=1` or `IDRAK_HIP_DEFAULT=1`. `IDRAK_DISABLE_CUDA`,
`IDRAK_DISABLE_VULKAN` and `IDRAK_DISABLE_HIP` turn a device kind off.

Memory offloading of layer weights (`--offload`) and the layers that use it are in `Idrak`; with `Idrak.Gpu` alone, the
devices compute and the tensors stay in device memory.

`idrak devices` lists the devices found, `idrak doctor` what is missing, `idrak kernels -d DEVICE` which kernel each
operation runs, and `idrak kernels dump` writes the generated kernel sources.

A plug-in can ship Vulkan kernels of its own: `VulkanKernel` (`Idrak.Gpu.Vulkan`) takes SPIR-V 1.3 words (compiled
from GLSL with glslc, say; storage buffers at set 0, binding i the i-th storage, scalars in one push-constant block) and
`VulkanKernel.Dispatch(backend, groupsX, groupsY, groupsZ, storages, pushConstants)` queues it on a Vulkan device, from
a kernel the plug-in registers for the "vulkan" kind with `Kernels.Register`.

## Documentation

Guides, the tested devices and the results on each: https://github.com/ahmedseada/Idrak

Licensed under the Apache License 2.0.
