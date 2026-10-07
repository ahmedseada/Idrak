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
| Device API (`Idrak.Abstraction.Devices`) | `Backend` (memory, copies and a `NameKernel` per operation; `RetryOnHost`, off by default, runs an operation whose kernel fails on the CPU instead), `Storage`, `BackendCapabilities`, `DeviceProvider` and `DeviceProviders` (where a new kind of device registers), `DeviceException` (the base of every GPU error, reported to telemetry when raised), offloading and staging |
| Operations (`Idrak.Abstraction.Operations`) | `Ops` (a descriptor per operation), `Kernels.Register` (a kernel for an operation on a kind of device), `Kernels.Chain` (which kernel each operation runs on a device), `Kernels.Trace` (calls and host fallbacks) |
| Telemetry (`Idrak.Abstraction.Diagnostics`) | `Telemetry` (subscribe hooks), `ITelemetryHook`, `TelemetryLevel`, the events (`DeviceFailed` among them: a GPU error, or an operation retried on the CPU, with a hint naming `IDRAK_RETRY_ON_HOST` while the retry is off), `ConsoleLogger` and `JsonLinesLogger` (both always include device failures) |
| Generation: tools | `Tool`, `[Tool]`, `ToolResult`, `IToolRegistry` and its default `ToolRegistry` (validation, allow rules, approvals, a timeout); `IToolChatModel` (a chat model that carries tools) and `ChatTools.WithTools` (runs a chat model's tool calls on the server) |
| Serving | the engine's model kinds (`EngineModel<TCopy>`), `IPredictor<TIn, TOut>`, `IModelCatalog` (named models reached through their contracts; the inference engine implements it), `KeepAlive` (parses "30m", "1h30m", 0, -1) |

The GPU devices (CUDA, Vulkan, HIP) ship in the `Idrak.Gpu` package (which `Idrak` brings) and are built on this
public device API alone; when an application includes it, `Device.Available` lists them, even before any other type of
Idrak is used. `idrak kernels -d DEVICE` lists which kernel each operation runs there.

A kernel of one's own for an operation, on one kind of device, wherever a requirement on the device holds:

```csharp
using Idrak.Abstraction.Devices;
using Idrak.Abstraction.Operations;

OperationKernels.Softmax mine = (backend, x, y, rows, cols, log) =>
{
    if (log || cols > 4096)
    {
        backend.SoftmaxKernel(x, y, rows, cols, log);   // the device's own kernel for the other cases
        return;
    }

    // ... my kernel
};
using var registration = Kernels.Register(Ops.Softmax, "vulkan", mine, requirement: b => b.Capabilities.FusedKernels);
```

Projects with implicit usings get `using Idrak.Abstraction;` from this package, so code written for Idrak 0.3 compiles
unchanged.

## Status

Preview, while the version is 0.y.z: the contracts move here one area at a time (tensors and modules, then training,
generation, data and formats), each with its default implementation. Plan:
[plans/10-abstraction.md](https://github.com/ahmedseada/Idrak/blob/main/plans/10-abstraction.md).

Licensed under the Apache License 2.0.
