# Idrak.Abstraction

The contracts [Idrak](https://www.nuget.org/packages/Idrak) is built on, each with its default implementation.

```bash
dotnet add package Idrak              # brings Idrak.Abstraction along
dotnet add package Idrak.Abstraction  # the contracts alone, for a plug-in
```

## What is in the package

| Area | What is there |
|------|---------------|
| Devices | `Device`, `DeviceType`, `ComputeResources`; the CPU device (SIMD, multi-threaded), which is also every other device's host fallback |

The GPU devices (CUDA, Vulkan, HIP) ship in the `Idrak` package; when an application includes it, `Device.Available`
lists them, even before any other type of Idrak is used.

Projects with implicit usings get `using Idrak.Abstraction;` from this package, so code written for Idrak 0.3 compiles
unchanged.

## Status

Preview, while the version is 0.y.z: the contracts move here one area at a time (tensors and modules, then training,
generation, data and formats), each with its default implementation. Plan:
[plans/10-abstraction.md](https://github.com/ahmedseada/Idrak/blob/main/plans/10-abstraction.md).

Licensed under the Apache License 2.0.
