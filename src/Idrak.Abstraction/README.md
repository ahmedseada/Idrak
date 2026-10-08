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
| Operations (`Idrak.Abstraction.Operations`) | `Ops` (a descriptor per operation), `PluginOperations` and `PluginOperation<TKernel>` (operations a plug-in declares, with a default kernel), `Kernels.Register` (a kernel for an operation on a kind of device), `Kernels.Chain` (which kernel each operation runs on a device), `Kernels.Trace` (calls and host fallbacks) |
| Modules (`Idrak.Abstraction.Modules`) | `ILinearAdapter` with LoRA and DoRA, `ILinearLayer` |
| Training (`Idrak.Abstraction.Training`) | `Optimizer` with SGD, Adam and AdamW; `LearningRateScheduler` with the step, exponential, cosine and lambda schedules |
| Data (`Idrak.Abstraction.Data`) | `ISampleSource`, `ISampleStream`, `ISampleReader`, `ISampleTransform`, `ImageData`, `IDownloader` |
| Formats (`Idrak.Abstraction.Formats`) | `IModelSource` and `ModelSources` (core registers the folder, store and .gguf sources, Idrak.Data the Hugging Face one) |
| Generation (`Idrak.Abstraction.Generation`) | `ITokenizer` (char and word tokenizers), `ChatTemplate` and `ChatTemplates`, `IChatModel` (with the part kinds it takes), `ChatPart` and `ChatParts` (text and image parts), `ITextModel`, `IToolCallParser` and `ToolCallFormats` (every built-in parser), `PackedWeight` (int8, int4, bfloat16), `KeyValueLayout` and `KeyValueLayouts`, `RopeScalings`, `DecoderSpec`, `GenerationOptions` |
| Generation: tools | `Tool`, `[Tool]`, `ToolResult`, `IToolRegistry` and its default `ToolRegistry` (validation, allow rules, approvals, a timeout); `IToolChatModel` (a chat model that carries tools) and `ChatTools.WithTools` (runs a chat model's tool calls on the server) |
| Retrieval (`Idrak.Abstraction.Retrieval`) | `IEmbedder` |
| Serving (`Idrak.Abstraction.Serving`) | the engine's model kinds (`EngineModel<TCopy>`, `IEngineHost`, `IEngineLease`, `IEngineBatcher`), `IPredictor<TIn, TOut>`, `IModelCatalog` (named models reached through their contracts; the inference engine implements it), `KeepAlive` (parses "30m", "1h30m", 0, -1) |
| Diagnostics (`Idrak.Abstraction.Diagnostics`) | `Telemetry`, `ITelemetryHook`, `TelemetryLevel` and the telemetry events (`DeviceFailed`, `OverrideFailed` and `OverrideCompared` among them; `DeviceFailed`: a GPU error, or an operation retried on the CPU, with a hint naming `IDRAK_RETRY_ON_HOST` while the retry is off); `ConsoleLogger` and `JsonLinesLogger` always include device failures and overrides failing or compared |
| The override loop (root) | `SlotTable<TKey, TValue>` (every registry's entries, each with its library default), `Slot`, `SlotPolicy` (`FallBack`, `Throw`, `Shadow`), `SlotGuard<TValue>`, `Overrides` (`Report()`, `AsLibraryDefaults`), `Comparisons` (shared with the testing kit) |

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

A plug-in (a packed weight format, a key/value cache layout, a graph operation, an `Autograd.Function`) declares the
operations it needs with a default kernel that runs on every device, and gets a fast path on the devices it registers
kernels for. Its operations show in `Kernels.Chain` and `idrak kernels` ("plug-in" rows), and `Kernels.Trace` counts
their calls:

```csharp
public delegate void ScaleRows(Backend backend, Storage rows, Storage scales, Storage output, int n, int width);

// Declared once (a static field): its default composes the library's operations.
static readonly PluginOperation<ScaleRows> Scale = PluginOperations.Register<ScaleRows>("MyLayout.ScaleRows",
    (b, rows, scales, output, n, width) => b.GroupScaleShift(rows, scales, null, output, n, n / width, width, false),
    KernelSource.Composed);

// A faster kernel for one kind of device (Idrak.Gpu's VulkanKernel, CudaKernel and HipKernel run SPIR-V, PTX and HIP C++ of one's own).
Kernels.Register(Scale, "cpu", (b, rows, scales, output, n, width) => { /* loops over rows.HostMemory, ... */ });

// Where the plug-in runs it (KeyValueLayout.Expand, say): the kernel for this device, cached per device.
Scale.KernelFor(backend)(backend, rows.Storage, scales.Storage, output.Storage, rows.Size, width);
```

A default declared as `KernelSource.Host` reads its operands with `Download` and writes them back with `Upload`; off the
CPU each call counts as a host fallback (`Kernels.HostCalls`, `KernelTrace.HostCallsByOperation`).

To check a device or an override of a contract (a token sampler, a tokenizer, a RoPE scaling) against the default
implementation, stress it and keep its failing cases, use the testing kit,
[Idrak.Abstraction.Testing](https://www.nuget.org/packages/Idrak.Abstraction.Testing), from your test project:
`Conformance.Check(Device.Get("mydevice")).ThrowIfFailed();`.

## The override loop

Every registry keeps its entries in a `SlotTable<TKey, TValue>`: each entry is a slot with the library default and,
over it, what an app registered. An app's registration shadows the default without removing it, `Unregister` brings it
back, and every registry has `Default(name)` and `Origin(name)`. While an app's entry shadows a default, the table hands
it out guarded by the registry's `SlotGuard<TValue>` under the slot's `SlotPolicy`: `Throw` (the default: the failure
reaches the caller and is reported with a hint naming the exact call, or `IDRAK_OVERRIDE_POLICY` setting, that switches
the slot to the others), `FallBack` (a call that throws is retried on the default), or `Shadow` (the default answers,
the app's runs on a share of the calls, 1% unless set, and `Comparisons` says how they differ). A slot with only its default hands it out as it
is. The library's packages register their built-ins inside `Overrides.AsLibraryDefaults`, so they are defaults, not
overrides. A default the library improves gets a higher version (`RegisterDefault(key, value, version, since)`); the
older ones stay reachable with `Default(key, version)`. `Overrides.Report()` lists every override with its origin and
policy, and says when one was built against a release older than the default it shadows (`SlotOverride.Outdated`). The
events are `OverrideFailed` and
`OverrideCompared` (`TelemetryLevel.Overrides`, always on in `ConsoleLogger` and `JsonLinesLogger`).

A registry of one's own gets the same behavior from a table:

```csharp
public static class Greeters
{
    private static readonly SlotTable<string, Func<string, string>> Table = new("Greeters",
        guard: (slot, app, library) => name => slot.Call(() => app(name), () => library(name), Comparisons.Exact));

    static Greeters() => Overrides.AsLibraryDefaults(() => Table.Register("plain", name => $"Hello, {name}"));

    [MethodImpl(MethodImplOptions.NoInlining)]   // its caller is the registering assembly, the entry's Origin
    public static void Register(string name, Func<string, string> greet) => Table.Register(name, greet, Assembly.GetCallingAssembly());
    public static bool Unregister(string name) => Table.Unregister(name);
    public static Func<string, string> Get(string name) => Table.Find(name) ?? throw new NotSupportedException(name);
    public static void SetPolicy(string name, SlotPolicy policy, double shadowRate = Slot.DefaultShadowRate) => Table.SetPolicy(name, policy, shadowRate);
}
```

Projects with implicit usings get `using Idrak.Abstraction;` from this package, so code written for Idrak 0.3 compiles
unchanged.

## Status

Preview, while the version is 0.y.z. A test counts the library packages that use each contract and keeps it here only
while Abstraction or more than one package uses it (plan 10, decision 10). Plan:
[plans/10-abstraction.md](https://github.com/ahmedseada/Idrak/blob/main/plans/10-abstraction.md).

Licensed under the Apache License 2.0.
