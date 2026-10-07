# Idrak.Abstraction

The contracts [Idrak](https://www.nuget.org/packages/Idrak) is built on, each with its default implementation.

```bash
dotnet add package Idrak              # brings Idrak.Abstraction along
dotnet add package Idrak.Abstraction  # the contracts alone, for a plug-in
```

## What is in the package

The contracts more than one package uses (or Abstraction itself does), each with its default. A contract only one
package uses lives in that package, under its `.Abstractions` namespace (for example `Idrak.Vision.Abstractions`), and
its README lists it.

| Area | What is there |
|------|---------------|
| Root (`Idrak.Abstraction`) | `Device`, `DeviceType`, `ComputeResources`, `Tensor`, `TensorScope`, autograd, `Module`; the CPU device (SIMD, multi-threaded), which is also every other device's host fallback |
| Device API (`Idrak.Abstraction.Devices`) | `Backend` (memory, copies and a `NameKernel` per operation; `RetryOnHost`, off by default, runs an operation whose kernel fails on the CPU instead), `Storage`, `BackendCapabilities`, `DeviceProvider` and `DeviceProviders` (where a new kind of device registers), `DeviceException` (the base of every GPU error, reported to telemetry when raised), `IMemoryOffload`, `IHostStaging`, `IBackwardStaging` |
| Operations (`Idrak.Abstraction.Operations`) | `Ops` (a descriptor per operation), `Kernels.Register` (a kernel for an operation on a kind of device), `Kernels.Chain` (which kernel each operation runs on a device), `Kernels.Trace` (calls and host fallbacks) |
| Modules (`Idrak.Abstraction.Modules`) | `ILinearAdapter` with LoRA and DoRA, `ILinearLayer` |
| Training (`Idrak.Abstraction.Training`) | `Optimizer` with SGD, Adam and AdamW; `LearningRateScheduler` with the step, exponential, cosine and lambda schedules |
| Data (`Idrak.Abstraction.Data`) | `ISampleSource`, `ISampleStream`, `ISampleReader`, `ISampleTransform`, `ImageData`, `IDownloader` |
| Formats (`Idrak.Abstraction.Formats`) | `IModelSource` and `ModelSources` (core registers the folder, store and .gguf sources, Idrak.Data the Hugging Face one) |
| Generation (`Idrak.Abstraction.Generation`) | `ITokenizer` (char and word tokenizers), `ChatTemplate` and `ChatTemplates`, `IChatModel`, `ITextModel`, `IToolCallParser` and `ToolCallFormats` (every built-in parser), `PackedWeight` (int8, int4, bfloat16), `KeyValueLayout` and `KeyValueLayouts`, `RopeScalings`, `DecoderSpec`, `GenerationOptions` |
| Generation: tools | `Tool`, `[Tool]`, `ToolResult`, `IToolRegistry` and its default `ToolRegistry` (validation, allow rules, approvals, a timeout); `IToolChatModel` (a chat model that carries tools) and `ChatTools.WithTools` (runs a chat model's tool calls on the server) |
| Retrieval (`Idrak.Abstraction.Retrieval`) | `IEmbedder` |
| Serving (`Idrak.Abstraction.Serving`) | the engine's model kinds (`EngineModel<TCopy>`, `IEngineHost`, `IEngineLease`, `IEngineBatcher`), `IPredictor<TIn, TOut>`, `IModelCatalog` (named models reached through their contracts; the inference engine implements it), `KeepAlive` (parses "30m", "1h30m", 0, -1) |
| Diagnostics (`Idrak.Abstraction.Diagnostics`) | `Telemetry`, `ITelemetryHook`, `TelemetryLevel` and the telemetry events (`DeviceFailed` among them: a GPU error, or an operation retried on the CPU, with a hint naming `IDRAK_RETRY_ON_HOST` while the retry is off); `ConsoleLogger` and `JsonLinesLogger` always include device failures |

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

Preview, while the version is 0.y.z. A test counts the library packages that use each contract and keeps it here only
while Abstraction or more than one package uses it (plan 10, decision 10). Plan:
[plans/10-abstraction.md](https://github.com/ahmedseada/Idrak/blob/main/plans/10-abstraction.md).

Licensed under the Apache License 2.0.
