# Idrak.Gpu

The GPU devices of [Idrak](https://www.nuget.org/packages/Idrak): CUDA, Vulkan and HIP, with all their kernels. No CUDA
Toolkit, no HIP SDK, no shader compiler, no native files in the package: the kernels are generated in C# (PTX for
NVIDIA, SPIR-V for Vulkan, GCN for AMD) and finished on the user's machine (PTX and SPIR-V by the GPU's own driver, GCN
by Idrak's own assembler).

## Install

```bash
dotnet add package Idrak       # brings Idrak.Gpu along: every device, as before
dotnet add package Idrak.Gpu   # the GPU devices alone, on Idrak.Abstraction
```

The package depends on `Idrak.Abstraction` only and is built on its public device API (`Backend`, `Storage`,
`Kernels`), the same surface an outside device package would use. `Device` is CPU-only on the new design
(`Device.Available` lists the CPU and `Device.Parse` reads "cpu" alone): reaching these devices through it is not on the
new design yet.

## What is in the package

| Device | How it works | Needs |
|--------|--------------|-------|
| CUDA (`cuda:N`) | The NVIDIA driver API directly; PTX kernels generated in C#; flash attention, CUDA graphs, memory offloading | An NVIDIA display driver (`nvcuda.dll`, `libcuda.so.1`) |
| Vulkan (`vulkan:N`) | SPIR-V kernels generated in C# for NVIDIA, AMD, Intel and phone GPUs; fused decoding, recorded graphs, cooperative matrices; widths and splits measured on the device | A Vulkan 1.1+ driver |
| HIP (`hip:N`) | The HIP runtime's memory and copies; Idrak's own GCN kernels as they land (plan 35), the rest through host fallbacks | The AMD graphics driver (Windows) or ROCm's runtime (Linux); no SDK |

Every operation a device has no kernel for falls back to the CPU device of `Idrak.Abstraction`, so a model runs on any
of them unchanged. Kernel choices come from what each device reports or from measurements on it, stored per device and
driver, never from a card's name. `Device.Default` is the CPU unless a host sets it.

Memory offloading of layer weights (`--offload`) and the layers that use it are in `Idrak`; with `Idrak.Gpu` alone, the
devices compute and the tensors stay in device memory.

`idrak devices` lists the devices found, `idrak doctor` what is missing, `idrak kernels -d DEVICE` which kernel each
operation runs, and `idrak kernels dump` writes the generated kernel sources.

A plug-in can ship Vulkan kernels of its own: `VulkanKernel` (`Idrak.Gpu.Vulkan`) takes SPIR-V 1.3 words (compiled
from GLSL with glslc, say; storage buffers at set 0, binding i the i-th storage, scalars in one push-constant block) and
`VulkanKernel.Dispatch(backend, groupsX, groupsY, groupsZ, storages, pushConstants)` queues it on a Vulkan device;
making it the kernel an operation runs is not on the new design yet. The same for CUDA: `CudaKernel`
(`Idrak.Gpu.Cuda`) takes PTX text and an `.entry` name; `Launch(backend, gridX, gridY, gridZ, blockX, blockY, blockZ,
arguments)` queues it on the device's stream, the arguments being storages, `int`, `float` and `long` values
(`KernelArgument`) checked against the parameters the PTX declares. Each device compiles a text once (the PTX by the
driver) and keeps the module; driver errors raise `CudaException`. The same for HIP: `GcnKernel` (`Idrak.Gpu.Hip`)
takes GCN text, AMD's assembly written with virtual registers as PTX is, whose `.kernel` line names it and whose `.arg`
lines declare its arguments; each device assembles it once with Idrak's own assembler for the target it reports (no
SDK), keeps the code object in the HIP kernel cache and loads it once, and `Launch` checks the arguments and the block
against what the text declares. `GcnKernel.UnavailableReason(b)` says why GCN kernels cannot run on a device (one whose
target no GCN family of `GcnTargets` handles); errors raise `HipException` (text that does not assemble names its line
and column). `installation/hip.md` shows a whole example.

## Documentation

Guides, the tested devices and the results on each: https://github.com/ahmedseada/Idrak

Licensed under the Apache License 2.0.
