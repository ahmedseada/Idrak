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
| Device API (`Idrak.Abstraction.Devices`) | `Backend` (memory, copies and a `NameKernel` per operation; `RetryOnHost`, off by default, runs an operation whose kernel fails on the CPU instead) and `IDeviceBackend` (what every device provides), the capability interfaces a device adds (`IGraphCaptureBackend`, `IKernelProfiler`, `IRowAttentionBackend`, `IMaskedCacheAttentionBackend`, `ISegmentedAttentionBackend`, `IFusedPackedKernels`, `IMatrixUnitBackend`, `IRecurrentKernels`, `IOffloadingBackend`, `IPeerCopyBackend`, `IMemoryReport`, `ILongLivedAllocations`), `Storage` and `IHostStorage`, `BackendCapabilities`, `IDeviceProvider`, `ISeparableDeviceProvider` and `DeviceProviders` (where a new kind of device registers), `DeviceException` (the base of every GPU error, reported to telemetry when raised), `IMemoryOffload`, `IHostStaging` and `ISlotStaging`, `ITensorOffloadingHooks`, `IBackwardStaging` |
| Operations (`Idrak.Abstraction.Operations`) | `Ops` (a descriptor per operation), `Kernels.Chain` (which kernel each operation runs on a device), `Kernels.Trace` (calls and host fallbacks) |
| Modules (`Idrak.Abstraction.Modules`) | `ILinearAdapter` with LoRA and DoRA, `ILinearLayer` |
| Training (`Idrak.Abstraction.Training`) | `IOptimizer` and `IStatefulOptimizer` (saved state), the `Optimizer` base with SGD, Adam and AdamW; `ILearningRateScheduler` with the step, exponential, cosine and lambda schedules |
| Data (`Idrak.Abstraction.Data`) | `ISampleSource`, `ISampleStream`, `ISampleReader`, `ISampleTransform`, `ImageData`, `IDownloader` and `IImageCodec`; for detection and segmentation `BoundingBox`, `AnnotatedImage`, `ObjectAnnotation`, `AnnotatedDataset`, and `IAugmentation` |
| Formats (`Idrak.Abstraction.Formats`) | `IModelSource` and `ModelSources` (core registers the folder, store and .gguf sources, Idrak.Data the Hugging Face one); `IOnnxRuntimeLoader` and `OnnxRuntimeLoaders` (a runtime that runs an .onnx model outside Idrak's import; Idrak.Onnx.Runtime registers ONNX Runtime); |
| Generation (`Idrak.Abstraction.Generation`) | `ITokenizer` (char and word tokenizers), `ChatTemplate` and `ChatTemplates`, `IChatModel` (with the part kinds it takes), `ChatPart` and `ChatParts` (text and image parts), `ITextModel`, `IToolCallParser` and `ToolCallFormats` (every built-in parser), `IFillInMiddleFormat` and `FillInMiddleFormats` (the prefix-suffix-middle and suffix-prefix-middle `FillInMiddleLayout`s; no family: Idrak.Text.Plugins registers them), `KeyValueLayout` and `KeyValueLayouts`, `RopeScalings`, `DecoderSpec`, `GenerationOptions`; the image side of vision-language models: `IVisionEncoder` (with `IVisionEncoderStages`), `IImagePromptFormat`, `IImageAttentionRule` and `ImageAttentionRules`, `PretrainedVision` (with `VisionEncoderOptions`), `IVisionTuningPart` (with `VisionTuningParts`: the projector or tower a family lets a fine-tune train, saved and read back by `Export` and `Import`); the feature caches the image and audio packages share: `IFeatureCache`, `FeatureCaches` and `FeatureCacheKey` (Idrak.Text registers "memory" and "disk") |
| Generation: tools | `Tool`, `[Tool]`, `ToolResult`, `IToolRegistry` and its default `ToolRegistry` (validation, allow rules, approvals, a timeout); `IToolChatModel` (a chat model that carries tools) and `ChatTools.WithTools` (runs a chat model's tool calls on the server); `IToolProvider` (a tool assembly's tools as `ITool`, declared with `[assembly: ToolProvider(typeof(X))]`) and `ToolProviders` (`Of` reads an assembly's, `FromMethods` turns `[Tool]` methods into tools) |
| Retrieval (`Idrak.Abstraction.Retrieval`) | `IEmbedder` |
| Serving (`Idrak.Abstraction.Serving`) | the engine's model kinds (`IEngineModel<TCopy>` and its base `EngineModel<TCopy>`, `IEngineHost`, `IEngineLease`, `IEngineBatcher`), `IPredictor<TIn, TOut>`, `IModelCatalog` (named models reached through their contracts; the inference engine implements it), `KeepAlive` (parses "30m", "1h30m", 0, -1) |
| Diagnostics (`Idrak.Abstraction.Diagnostics`) | `Telemetry`, `ITelemetryHook`, `TelemetryLevel` and the telemetry events (`DeviceFailed`, `OverrideFailed` and `OverrideCompared` among them; `DeviceFailed`: a GPU error, or an operation retried on the CPU, with a hint naming `IDRAK_RETRY_ON_HOST` while the retry is off); `ConsoleLogger` and `JsonLinesLogger` always include device failures and overrides failing or compared |
| Reinforcement (`Idrak.Abstraction.Reinforcement`) | `IEnvironment<TObservation, TAction>` and `Space` (episodes reset from a seed and stepped; discrete, box, text and conversation spaces), `IEnvironmentKind`, `IPolicy<TObservation, TAction>`, `IRewardFunction`, `IAdvantageEstimator` (a fixed summation order), `IPolicyLoss`, `IPreferencePairSource` (pairs from episodes: "group" and "verdict" built in, reading `PreferenceInfoKeys`) and `IPreferencePolicy`; `Trajectory`, `ActionRecord` and `RolloutBatch` (pooled arrays). The estimators and losses are Idrak.Reinforcement's, the environments and rewards Idrak.Text's and the plug-ins' |
| Command sandboxes (`Idrak.Abstraction`) | `ICommandSandbox`: where an outside command a model asked for runs and under which `CommandLimits` (time, output, memory, processes, CPUs, network, the programs allowed), giving a `CommandResult`; a sandbox that cannot keep a limit refuses, naming one that can. `StartAsync` starts a command interactively (`CheckStart`; a `CommandStart` holding an `ICommandSession`: its standard input, output and error as streams, `Completion`, `Stop` ending its tree); a sandbox that cannot refuses with the reason. Idrak.Text has `process` (both ways), the coding plug-in of Idrak.Reinforcement.Plugins `container` (runs only) |

The GPU devices (CUDA, Vulkan, HIP) ship in the `Idrak.Gpu` package (which `Idrak` brings) and are built on this
public device API alone; when an application includes it, `Device.Available` lists them, even before any other type of
Idrak is used. `idrak kernels -d DEVICE` lists which kernel each operation runs there.

To check a device against the default implementation, stress it and keep its failing cases, use the testing kit,
[Idrak.Abstraction.Testing](https://www.nuget.org/packages/Idrak.Abstraction.Testing), from your test project:
`Conformance.Check(Device.Get("mydevice")).ThrowIfFailed();`.

Projects with implicit usings get `using Idrak.Abstraction;` from this package, so code written for Idrak 0.3 compiles
unchanged.

## Status

Preview, while the version is 0.y.z. A test counts the library packages that use each contract and keeps it here only
while Abstraction or more than one package uses it (plan 10, decision 10). Plan:
[plans/done/10-abstraction.md](https://github.com/ahmedseada/Idrak/blob/main/plans/10-abstraction.md).

Licensed under the Apache License 2.0.
