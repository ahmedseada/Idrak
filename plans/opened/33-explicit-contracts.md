# Plan 33: every contract an interface

**Status: opened 2026-10-25; C0 done (`IKind`, `IConfigurableKind`, `IKind<TRequest, T>`, `KindTable<TKind>`, `OptionKeys`,
`SlotTable`'s span lookups; the six existing kind interfaces adopt the base); C1 done (`IDeviceBackend` and the capability interfaces, read once per
backend; `OwnKernels`; typed `Operation<TKernel>`; `IDeviceProvider : IKind`; `IDeviceKernel`; the string-named kernel
lookups left as is: keyed by name across hundreds of call sites); C2-C7 in progress.** The list below comes from an audit of `src/` (the owner asked: where, no
code yet). Each row was found by reading the code; a sample of the rows was checked against the source.

## Why

In many places what a new implementation must provide exists only as a set of members on a base class, a flag that has
to agree with other members, a naming convention read by reflection, a string key known from docs, or a type check on
a library class. Someone writing a device, a cache format, a family, a teacher, a reward or a plug-in has no single
place that says what to implement, and the compiler cannot tell them they missed a part. `IRowStartsLayout`
(d6ef2b3f, `src/Idrak.Abstraction/Generation/KeyValueLayouts.cs`) is the shape every row below takes: an interface
names the required surface, implementations implement it, and a capability flag, where one is kept, is
`this is IThatInterface` instead of a separate member that must agree.

## The rule

- A behaviour an outside implementation may provide is an interface (in `Idrak.Abstraction` when any package can use
  it, else in the package's `{package}.Abstractions` namespace), registered through a `SlotTable` whose element is
  that interface (CLAUDE.md, "Abstraction always").
- An optional capability is its own interface; the base class or caller tests `is ICapability`, never a `bool` that
  must agree with members that throw or return null by default.
- No contract by reflection over names, by catching `NotSupportedException`, by matching an error message, or by
  `is ConcreteLibraryType` where a third party could need to opt in.
- String option keys are declared by the kind (`Keys`, or an option schema with types, ranges and defaults).
- Each phase keeps the public API guard (`api/*.txt`) and the inventory test passing; old members that become the
  interface's stay as forwarding members only where removing them would break callers without gain.

## Phases (in order of what outside implementers meet first)

| Phase | Covers | Tables below |
|---|---|---|
| C0 | The shared kind contract (`IKind`, `IConfigurableKind`, `IKind<TRequest, T>`), `KindTable<TKind>`, and `SlotTable`'s span lookup: the base the other phases' registries and kinds use. | "C0: the shared kind contract" below |
| C1 | The device contract: `Backend` split into `IDeviceBackend` and capability interfaces (graphs, profiling, FP8, row and segmented attention, fused packed products, matrix units, recurrent kernels, offloading and peer copies, host storage); `DeviceProvider`; own kernels without reflection over names. CPU, CUDA, Vulkan, HIP and the test devices implement them. | Idrak.Abstraction: devices and kernels; Idrak.Gpu |
| C2 | Generation contracts: `KeyValueLayout`'s remaining flags, `PackedWeight`, `ChatTemplate`, chat template readers, tool-call formats, `Tool`, `PretrainedVision`, `PretrainedArchitecture`, GGUF, and every `Func`-element registry in Idrak.Abstraction and Idrak. | Idrak.Abstraction: generation, training, cluster (generation rows); Idrak (core) (models rows) |
| C3 | Training and serving: `Optimizer` and its state, `Adam`'s fused path, schedulers, `EngineModel`, the `Module` leaf, `Trainer` and data sources, metrics and losses, cluster contracts with stub defaults. | Idrak.Abstraction: generation, training, cluster (training and cluster rows); Idrak (core) (training rows) |
| C4 | Plug-in entry points: `LibraryRegistrations` by attribute and interface, `IIdrakPluginPackage` for the CLI's `-P` and every `.Plugins` package's shape, tool assemblies, the ONNX runtime loader, `NotRegisteredException`. | Idrak.Cli and Idrak.AspNetCore; the plug-in row of Idrak.Text |
| C5 | Domain packages: Text, Vision, Audio, Reinforcement, Data, Cluster, Network. | Their tables |
| C6 | Type checks that decide behaviour (`is Linear`, `is DecoderBlock`, `is DataLoader`, `is CausalSelfAttention`, …) replaced by the interfaces above or small ones of their own. | The type-check rows of every table |
| C7 | The testing kit: suites take the real interfaces instead of `…UnderTest` delegate bags; annotation format traits on the format. | Idrak.Abstraction.Testing rows |

Rows marked "(low)" go last in their phase; rows marked "leave as is" are listed for completeness and need no change.

## C0: the shared kind contract

Most registries below hold the same shape under different names: a name, a summary, the option keys it reads, and a
factory (`IDetectionDecoderKind { Name; Keys; Create; CreateHead }`, `IVisionMetricKind { Name; Task; Create }`,
`IVocoderKind { Accepts; Create }`, `IKeywordIndexKind { Name; Summary; Create }`, `IEncoderHeadKind { Name; Keys;
Create }`, `ISampleSourceKind { Name; Keys; Create }`, …), and the interfaces that exist today repeat it too
(`IEnvironmentKind`, `IPolicyLoss`, `IAdvantageEstimator`: `Name`, `Summary`, `Keys`; `IOptimizerKind`,
`ILearningRateSchedule`: `Name`, `Summary`, a `Create`; `IChatPartKind`: `Name`, a `Read`). One base, in
`Idrak.Abstraction`:

```csharp
// What every registered kind is: listed, described, found by name.
public interface IKind
{
    string Name { get; }        // stored, a key, printed: string (optimization rule 3)
    string Summary { get; }
}

// A kind that reads options: the keys it takes (later the option schema: types, ranges, defaults).
public interface IConfigurableKind : IKind
{
    IReadOnlySet<string> Keys { get; }   // a FrozenSet: a key read from text is checked by span (rule 15)
}

// A kind that makes something from one request record (a new field never breaks an implementation).
public interface IKind<in TRequest, out T> : IConfigurableKind where TRequest : allows ref struct   // rule 20
{
    T Create(TRequest request);
}

// The registry of kinds: over SlotTable (two layers, policies, versions, the overrides report, as every registry);
// registers by kind.Name, so a name never drifts from what it registers; lists names and summaries alike for every
// --list, help page and "not registered" error.
public sealed class KindTable<TKind> where TKind : class, IKind   // class: no box on a null or type test (rule 89)
{
    public void Register(TKind kind);
    public void Register(params ReadOnlySpan<TKind> kinds);              // rule 19
    public TKind? Find(ReadOnlySpan<char> name);                         // alternate lookup, no string made (rules 1, 15)
    public bool TryGet(ReadOnlySpan<char> name, [MaybeNullWhen(false)] out TKind kind);
}
```

Each specific kind keeps only what is its own:

| Kind | Becomes |
|---|---|
| `IDetectionDecoderKind` | `IKind<DetectionDecoderRequest, DetectionDecoder>` + `CreateHead` |
| `IVisionMetricKind` | `IKind<VisionMetricOptions, IVisionMetric>` + `Task` |
| `IVocoderKind` | `IKind<AudioFeatureOptions, IVocoder>` + `Accepts(options)` |
| `IKeywordIndexKind` | `IKind<KeywordIndexRequest, IKeywordIndex>` |
| `IEncoderHeadKind` | `IKind<EncoderHeadRequest, EncoderHead>` |
| `ISampleSourceKind` | `IKind<SampleSourceRequest, ISampleSource>` |
| `ITokenSamplerKind`, `IPositionBiasKind`, `IFeatureCacheKind`, `IProgressReporterKind`, `IAudioFeatureKind`, `ITokenizerModelKind`, `ITokenizerComponentKind<T>`, `IChatTemplateReader`, `IPackedFormat`, `IToolCallFormat`, `IVisionLoss`, `IBoxMatcher`, `ILayerType<T>`, `IGraphOp`, `INetworkOp`, `IOnnxImportOp`, `IOnnxExportOp`, `IRopeScaling`, the GGUF kinds | `IKind<TheirRequest, TheirResult>` + their own member where they have one |
| `IEnvironmentKind`, `IPolicyLoss`, `IAdvantageEstimator`, `IOptimizerKind`, `ILearningRateSchedule`, `IChatPartKind`, `IDeviceProvider` (C1) | Adopt `IKind` / `IConfigurableKind`; their factories take a request record (`IOptimizerKind.Create(parameters, settings)` → `Create(OptimizerRequest)`) |

**`SlotTable`'s span lookup.** `SlotTable<TKey, TValue>` (`src/Idrak.Abstraction/Overrides/SlotTable.cs`) is the store
behind all 46 registries (a name to the library default and an app's override, with the Throw/FallBack/Shadow
policies, default versions and the overrides report). Its `Find`, `TryGet` and `Contains` take only `TKey`, which is
`string` for every named registry, so every name read from a command line, a JSON config or an options dictionary is
made into a string before it is looked up (against rules 1 and 15). Fix it in `SlotTable` itself, so every registry
gains it, not only `KindTable`: for string keys, `TryGet(ReadOnlySpan<char>, …)`, `Find(ReadOnlySpan<char>)` and
`Contains(ReadOnlySpan<char>)` through `GetAlternateLookup<ReadOnlySpan<char>>()` on its snapshot's map, the lookup
made once per snapshot when the snapshot is built (rule 75), still lock-free.

## The optimization rules every new contract follows

The contracts below are not only renamed members: each new interface is shaped by `docs/optimization.md`. Per phase:

| Phase | What the rules require of its contracts |
|---|---|
| All | Stored names, keys and printed text `string` (rule 3); anything that takes a name, key or option read from input `ReadOnlySpan<char>` or `ReadOnlySpan<byte>` (rule 1), found through alternate lookups (rule 15) built once (rule 75); option text split with `span.Split` and `static readonly SearchValues` (rules 16, 17) and values parsed with `T.TryParse` on spans, `InvariantCulture` (rule 29); several registrations as `params ReadOnlySpan<T>` (rule 19); generic tables and helpers constrained `class` where `T` is a contract (rule 89); new interfaces' hot members free of telemetry cost when nobody listens (rule 78); every change proven to give the same output (rule 71). |
| C1 (devices) | Kernel contracts take `Storage` and sizes, never arrays or lists per call (rules 22, 75); capability interfaces tested once per backend and kept (`is IGraphCaptureBackend` at construction, not per kernel call: rule 75); `IHostStorage.HostMemory` a `Span<float>`; the device's limits read from the device (rule 84), and no member of a capability interface assumes a card. |
| C2 (generation) | `IChatTemplate.Render` writes into a caller's buffer (`IBufferWriter<char>` or a pooled builder returned in `finally`) instead of returning a new string per turn where the caller only tokenizes it; tool-call parsing and detection on `ReadOnlySpan<char>` (rule 1, section 2's sync core); `IGgufType.Dequantize(ReadOnlySpan<byte>, Span<float>)` with SIMD and a scalar tail (section 5); `ITool.InvokeAsync` keeps the async shell and its argument parsing in a sync span core (section 2). |
| C3 (training, serving) | `IStatefulOptimizer.StateParameter(ReadOnlySpan<char>)` parses the index from the span (rule 29), not with `Split`/`Substring`; `IMetric` and `ILoss` take tensors, no per-batch allocations on the host; `IShardableBatchSource.Shard` hands out indices without copying the source. |
| C4 (plug-ins) | The plug-in package and library registrations found through an assembly attribute read once per assembly, not by scanning every exported type with reflection on each load (rule 75). |
| C5 (domain packages) | `EpisodeInfoKey<T>` stores typed values (no `"true"`/number-as-text round trips: rules 4 and 74); audio, vision and text kinds take spans of samples and pixels (rule 72: memory walked in order); `IOptionSchema` parses each declared option from a span once per run. |
| C6 (type checks) | The interface test (`is IAttentionCacheInfo`) made once when a model is built or a generator made, kept, not per step or per token (rule 75). |
| C7 (testing kit) | Suites call the real interfaces, so they measure the real allocations (rule 90's diagnosis applies to them as to the library). |

## Where (Current File, Convention Followed, Candidate or Proposed Fix)

Paths are under `src/`.

### Idrak.Abstraction: devices and kernels

| Current File | Convention Followed | Candidate or Proposed Fix |
|---|---|---|
| `Idrak.Abstraction/Devices/Backend.cs:300` | `Backend` abstract class: 13 abstract members (`Kind`, `Name`, `Capabilities`, `Allocate`, `Return`, `Detach`, `Attach`, `Upload`, `Download`, `DownloadRange`, `GetMemoryUsage`, `ReleaseCachedMemory`, `Synchronize`) plus about 190 optional `…Kernel` virtuals | `IDeviceBackend` with the 13 core members, plus the capability interfaces below |
| `Backend.cs:2537-2552` | `SupportsGraphs => false` with `BeginCapture`/`EndCapture`/`ReplayGraph` (throw), `AbortCapture`, `DestroyGraph` | `IGraphCaptureBackend`; `SupportsGraphs => this is IGraphCaptureBackend` |
| `Backend.cs:538,543` + `BackendCapabilities.cs:50` | `Capabilities.Profiling` flag + `StartProfile()` (empty) + `StopProfile() => []` | `IKernelProfiler { StartProfile; StopProfile }` |
| `Backend.cs:1165,1177,1186` | `Float8PaddedK(k) => 0` means "no FP8", must agree with `Float8QuantizeWeightKernel`/`Float8MatMulKernel => false` | `IFloat8Backend { PaddedK; QuantizeWeight; MatMul }` |
| `Backend.cs:2238-2282,2421` | `SupportsSegmentedAttention`/`SupportsAttentionRows` must agree with `AttentionSegmented[Backward]Kernel`, `AttentionRowsKernel`, `AttentionRowsBFloat16Kernel`, `PrefersComposedAttention`; defaults ask the CPU, so a new backend says "supported" and runs on the host | `IRowAttentionBackend { Supports(dim, variant); AttentionRows; AttentionRowsBFloat16 }`, `ISegmentedAttentionBackend` |
| `Backend.cs:983-1127,1193` + `BackendCapabilities.cs:47` | `Capabilities.FusedKernels`, `PrefersPackedMatMul` + 7 `PackedMatMul…Kernel => false` | `IFusedPackedKernels`; `FusedKernels => this is IFusedPackedKernels` |
| `Backend.cs:535,1143-1230` + `BackendCapabilities.cs:36-40` | `MatrixUnits`, `MatrixUnitAttentionHeadDim` (`Func<int,bool>`), `TensorCoresUnavailable()` gate `MatMulLowRank`, `BFloat16TransposedMatMul`, `GemmStrided`, `AttentionStrided[Backward]` | `IMatrixUnitBackend` |
| `Backend.cs:1616-1736` | 12 LSTM/GRU cell, step and sequence kernels `=> false`, no flag | `IRecurrentKernels` |
| `Backend.cs:546,557-590,608-643,2462-2472` | `Offload => null` plus "call `GuardMoves()` in the constructor, then `EnterOperation`/`BeginMove`/`EndMove`"; `CopyTo` must agree with `CopiesDirectlyTo`; `AvailableMemory`/`FreeMemory => null`, `LongLivedAllocations` | `IOffloadingBackend { Offload }` (the base guards moves itself), `IPeerCopyBackend { CopiesDirectlyTo; CopyTo }`, `IMemoryReport` |
| `Backend.cs:198,282` + `HostCall.cs:39` | `Storage.HostMemory` throws by default; "already on host" decided by `storage.Backend is CpuBackend` | `IHostStorage { Span<float> HostMemory }` |
| `Operations/Kernels.cs:266-305`, `HostCall.cs:28-31` | A device's own kernels found by reflection on methods named `…Kernel`, signatures matched as strings | Per-operation kernel interfaces, or `IReadOnlySet<Operation> OwnKernels` on the backend |
| `Operations/Kernels.cs:45` | `Kernels.Register(…, Delegate, …)` untyped overload, checked only at run time | Typed overloads only |
| `Devices/DeviceProviders.cs:13,34,47-65,76` | `DeviceProvider` abstract class in `SlotTable<string, DeviceProvider>`; `CreateSeparate => null`, 6 optional metadata members | `IDeviceProvider`, `ISeparableDeviceProvider { CreateSeparate }` |
| `Devices/TensorOffloading.cs:28-53` | Public settable static hooks `EnterForward`, `TrainableChanged`, `ForBackward`, `MarkFrozen` | `ITensorOffloadingHooks`, set by one registration call |
| `Devices/IHostStaging.cs:27` | `PlacesCopies` flag + `CopyToSlot` throws when the offset is not 0 | `ISlotStaging { CopyToSlot }` |
| `Idrak.Abstraction/LibraryDefaults.cs:27-90` (+ every package's `LibraryRegistrations.cs`) | Each package must have `<root>.LibraryRegistrations.RegisterFor(Type)`, found by name with `Type.GetType` from a hard-coded assembly list | `ILibraryRegistrations { RegisterFor(Type) }` + an assembly attribute |

### Idrak.Abstraction: generation, training, cluster

| Current File | Convention Followed | Candidate or Proposed Fix |
|---|---|---|
| `Generation/KeyValueLayouts.cs:24,193` | `KeyValueLayout` abstract class in `SlotTable<string, KeyValueLayout>`: `Name`, `RowWidth`, `Write`, `Expand` + optional `ExpandPrefix`, `Attend` | `IKeyValueLayout` |
| `KeyValueLayouts.cs:42,49,127,130,144,151` | Pairs left after `IRowStartsLayout`: `SlidingWindows` with `AttendVariant`; `FusedWrite` with `HalfWords`; `HasScales` (fill `KeyScales`); `Recordable` (`Write`/`Attend` queue device work only) | `ISlidingWindowLayout`, `IFusedWriteLayout`, `IScaledLayout`, `IRecordableLayout` |
| `Generation/Quantization.cs:35,99-148,192` | `PackedWeight` abstract class: 8 abstract + optional `MatMul`/`DequantizeInto`; built-in-only `Format`, `PackedValues` (throws), `ScaleValues`, `LowRankProducts`, `Float8Copy`; overriding `MatMul` means "recordable" with no flag; the registry holds a `PackedWeightFactory` delegate | `IPackedWeight`, `IKernelPackedWeight`, `IRecordable`, `IPackedFormat { Name; Pack(...) }` |
| `Generation/Chat.cs:104-140` | `ChatTemplate` abstract class: `ToolCalls`, `ToolCallFormatName`, `CreateToolCallParser` must agree; `PartKinds` must match `Render`; `ThinkTags`, `AddedPrefix` | `IChatTemplate` |
| `Generation/ChatTemplates.cs:16` | `SlotTable<string, Func<string, ITokenizer?, ChatTemplate?>>` | `IChatTemplateReader { Name; Read(folder, tokenizer) }` |
| `Generation/ToolCallFormats.cs:100,131` | Registry entry of two Funcs (Detect, Create) | `IToolCallFormat { Name; Detect; Create }` |
| `Generation/Tool.cs:32-180`, `ToolRegistry.cs:210` | `Tool` record with an `Invoke` Func; `[Tool]` methods by reflection, results awaited through `GetMethod("AsTask")` | `ITool { Definition; InvokeAsync(JsonObject, CancellationToken) }` |
| `Generation/VisionParts.cs:15` | `PretrainedVision` abstract class: `Family`, `Width`, `PromptFormat`, `Attention`, `StoredTensors`, `CreateEncoder` + `VisionOptionKeys => []` | `IPretrainedVision` |
| `Generation/DecoderSpec.cs:231-240` | `DecoderSpec.Describe(model)` puts the model in a weak side table the engine reads ("a family that builds its own network may too") | `IDescribedModel { Architecture; Spec }` |
| `Generation/Decoding.cs:611,694,707` | `BeginStep`/`EndStep` must wrap a step (outermost module only); `CacheFor(object owner)` keyed by identity | An `IDecodingStep` scope object, documented on `ICachedModule` |
| `Generation/TokenSamplers.cs:23` | `Func<SamplerRequest, ITokenSampler>` | `ITokenSamplerKind { Name; Create }` |
| `Generation/PositionBiases.cs:441,454` | `Func<PositionBiasRequest, …>`; `Settings` keys and `Weights("table")` only in docs | `IPositionBiasKind { Name; Keys; Weights; Create }` |
| `Generation/RopeScalings.cs:88,102-125` | `RopeScalingMethod` delegate; `Parameters` keys only in docs; `FrequenciesAt` Func | `IRopeScaling { Name; Keys; Frequencies }` |
| `Generation/FeatureCache.cs:123` | `Func<FeatureCacheOptions, IFeatureCache>` | `IFeatureCacheKind` |
| `Generation/Tokenizer.cs:655-676` | `Save` switches on `CharTokenizer`/`WordTokenizer`; `Load` on a "type" string | `ISavableTokenizer { Type; Save }` + a loader registry |
| `Generation/VisionLanguage.cs:251` | `ImagePromptFormat.Join` Func | A member of `IPretrainedVision`'s prompt format |
| `Training/Optimizer.cs:9,71-89,213` | `Optimizer` abstract class: `SaveState`/`LoadState` throw by default, overridden together; "keeps no state" detected by catching the exception (`Idrak.Network/Training/Sharding.cs:461`, `SyncCore.cs:328,351`); `StateParameter` parses "x.&lt;index&gt;"; `ScalesGradientsInStep` + `GradientScale` | `IOptimizer`, `IStatefulOptimizer { SaveState; LoadState; StateParameter }` |
| `Training/Adam.cs:23,80` | Fused path only when `GetType() == typeof(Adam)` or `typeof(AdamW)`; a subclass loses it | `protected virtual bool Fusable`, or `IFusedAdam` |
| `Training/Schedulers.cs:10` | `LearningRateScheduler` abstract class; "call `Initialize()` from the derived constructor once fields are set" | `ILearningRateScheduler { Step; Epoch }` |
| `Training/LearningRateSchedules.cs` | `ILearningRateSchedule.Create` returns a `Func<Optimizer, int, …>` | Return `ILearningRateScheduler` |
| `Serving/EngineModel.cs:77` | `EngineModel<TCopy>` abstract class: `LoadCopy`, `UnloadCopy`, `Describe`, `OnAttached` | `IEngineModel<TCopy>` |
| `Modules/Module.cs:135-221` | A leaf holding tensors must override `Parameters()`, `Buffers()`, `MoveTo()`, `Dispose` consistently; only `ForwardCore` is abstract | `ITensorOwner { Parameters; Buffers; MoveTo }` |
| `Cluster/GradientSync.cs:127-182` | `IGradientSync`: `Updates => false` with `Update` (throws), `Gather`/`SaveOptimizerState`/`Owner`/`Local`/`LoadOptimizerState` stubs | Split out `IShardedGradientSync` |
| `Cluster/StepGroup.cs:48,72,79` | `IHostValueCollectives`: `SupportsHostValues` + `Try*`, `TryReduceScatter`/`TryAllGather => false` | Split the optional half into its own interface |
| `Cluster/StopAgreement.cs:50-59` | `IStopAgreement`: `Carries` + `Contribution`/`Carried` | `ICarriedStopAgreement` |
| `Formats/ITensorStore.cs:32,66` | `ReadBFloat16 => false`, `FormatOf => null` | `IBFloat16TensorStore` |
| `Data/SampleSource.cs:75`, `Data/SampleSources.cs:25` | `SampleSourceFactory` delegate; options keys only in docs | `ISampleSourceKind { Name; Keys; Create }` |
| `Diagnostics/Progress.cs:145,165` | `Func<ProgressRequest, IProgressReporter>` | `IProgressReporterKind` |
| `Idrak.Abstraction.Testing/Contracts/*.cs` (AudioCodec, SpeechRecognizer, Voice, KeywordIndex, AnnotationFormat, ImageModelFamily, FeatureCache, Stft, Sampler, NetworkRecipe, Transport/Collectives) + `Conformance.cs:163,243` | Suites take `…UnderTest` bags of delegates that mirror the real interfaces and can drift | Suites take the real interfaces (moved into `Idrak.Abstraction` where the dependency needs it) |
| `Idrak.Abstraction.Testing/Contracts/AnnotationFormatSuite.cs:12` | What an annotation format keeps (`AnnotationFormatTraits`) is declared only in the test kit | `IAnnotationFormat.Keeps : AnnotationFormatTraits` |
| `Idrak.Abstraction.Testing/ContractSuite.cs:38` (low) | `ContractSuite<T>` abstract class | Optional `IContractSuite<T>` |

### Idrak (core)

| Current File | Convention Followed | Candidate or Proposed Fix |
|---|---|---|
| `Idrak/Models/Abstractions/PretrainedArchitectures.cs:16,86` | `PretrainedArchitecture` sealed bag of Funcs (`Spec`, `Build`, `TensorName`, `Config`, `ForCheckpoint`, `Transposed`) in a SlotTable; 16 plug-in registrations | `IPretrainedArchitecture` |
| `Idrak/Models/Abstractions/Gguf*.cs` (`GgufArchitectures:16-64,115`, `GgufChatTemplates:19,36`, `GgufTypes:32,75`, `GgufPreTokenizers:17`, `GgufTokenizers:17`) | Classes holding Funcs (`Configure`, `TensorName`, `StoredOffset`, `Matches`, `Dequantize`); registries of strings and Funcs | `IGgufArchitecture`, `IGgufChatTemplate`, `IGgufType`, `IGgufTokenizer` |
| `Idrak/Models/Abstractions/PretrainedEncoders.cs:166,205` | `Func<EncoderHeadRequest, EncoderHead>` (sealed class); `Settings` keys and weight names ("dense.weight") only in docs | `IEncoderHeadKind { Name; Keys; Create }` |
| `Idrak/Models/Abstractions/TokenizerComponents.cs:32,99-114` | `TokenizerModels` + four `Func<JsonObject, T>` tables | `ITokenizerModelKind`, `ITokenizerComponentKind<T>` |
| `Idrak/Models/PretrainedModel.cs:868-891,951-960,1062,1085-1093` | Stored tensors and PEFT saving switch on `Linear`/`Embedding`/`RMSNorm`/`LayerNorm` and `LoraAdapter`/`DoraAdapter`, although `ILinearAdapter` exists | `IPeftAdapter { PeftType; Rank; Tensors }`, `INamedWeights` |
| `Idrak/Layers/Abstractions/RecurrentModule.cs:46,133,153,276` | `RecurrentModule` abstract class: `States`, `Cell` + optional `Sequence => null`; `ScalesLastHiddenBias` must agree with the constructor | `IRecurrentCell`, `IFusedRecurrentCell { Sequence }` |
| `Idrak/Layers/Abstractions/LayerTypes.cs:18,40` | Registry entry of a `Type` + Describe/Create Funcs | `ILayerType<T> { Type; Describe; Create }` |
| `Idrak/Layers/Abstractions/GraphOps.cs:14,41,76`, `NetworkOps` (`:13,66`), `NetworkBuilder.cs:42,104` | `GraphOp`/`NetworkOp` delegates; values typed `object` (Tensor or HostValue); `JsonObject` arguments; architectures in a side table | `IGraphOp`, `INetworkOp` with `Keys`; `IGraphValue` |
| `Idrak/Onnx/Abstractions/OnnxImportOps.cs:16,109`, `OnnxExportOps.cs:87-89,123` | Translator delegates; lambda modules matched by their `ToString()` text | `IOnnxImportOp`, `IOnnxExportOp`, `INamedLambda { Name }` |
| `Idrak/Training/Trainer.cs:19,204,213,717,741`, `TrainingRun.cs:42-57` | `is DataLoader` decides seeding, sharing and sharding although `IBatchSource` exists; `Metric` record of Funcs; `Loss`/`Optimizer`/`Scheduler` Funcs | `IShardableBatchSource { Shard; Seeded; Device }`, `IMetric`, `ILoss` |
| `Idrak/Layers/PackedSequences.cs:163-186`, `Idrak/Generation/TextGenerator.cs:112,134-140,411-414` | Packing, batching and KV sizing by `OfType<CausalSelfAttention>`, `is MultiHeadAttention or PositionalEncoding`, `MixtureOfExperts` | `IAttentionCacheInfo { KvHeads; HeadDim; SlidingWindow; SupportsRows; SupportsPacking }`, `IPositionAware` |
| `Idrak/Offloading.cs:196-242` | `is LayerGroup`/`Embedding`/`Linear` decides what is staged | `IOffloadableLayer` |
| `Idrak/Inference/ModelPackage.cs:112-127`, `Predictor.cs:425-435`, `Onnx/OnnxImport.cs:39` | Scaler and tokenizer switched by concrete type; `is GraphModule` | `ISavableScaler`, `IDescribedModel` |
| `Idrak/Layers/ImagePrefill.cs:279`, `Training/PipelineStage.cs:242,677` | `is DecoderBlock` for the first image block and checkpointing | `ICheckpointable` / `IDecoderBlock` |
| `Idrak/Layers/SequenceEncoder.cs:52`, `Diagnostics/ModelMemory.cs:75-130`, `Models/EncoderModels.cs:217,223` | `as PositionEmbedding`, `as Linear`, `OfType<Embedding>`, `is not Linear head` | `IPositionLimit`; the existing `ILinearLayer` |
| `Idrak/Layers/ModuleExtensions.cs:178-321`, `ModuleFiles.cs:45-172` | LoRA, quantization and packed saving use `OfType<Linear>` (sealed), although `ILinearLayer` exists | `ILinearLayer` |
| `Idrak/Layers/Decoder.cs:317,777`, `Attention.cs:412,426` (low) | `_bias is Module` decides whether a position bias is owned and moved | `IOwnedPositionBias` |
| `Idrak/Datasets/Dataset.cs:174`, `DataLoader.cs:91` (low) | `source is/as Dataset` | `IIndexedSource` |
| `Decoder.cs:1069-1147`, `DecoderBuilder.cs:239`, `Convolution.Fused.cs:36`, `Sequential.cs:77` | Fused paths checked by concrete type (`is RMSNorm`, Conv2d + BatchNorm) | Leave as is: library-kernel optimizations, not opt-in points |

### Idrak.Gpu (internal)

| Current File | Convention Followed | Candidate or Proposed Fix |
|---|---|---|
| `Cuda/CudaKernel.cs:77`, `HipKernel`, `VulkanKernel` | Three public sealed kernel classes, each checking `backend is not XBackend` | `IDeviceKernel { Launch(Backend, …) }` |
| `Vulkan/VulkanKernels.cs:140-146`, `VulkanBackend.LargeStorage.cs:218`, `HipBackend.Kernels.cs:293`, CUDA kernel dictionaries (low) | Kernels looked up by built string names (`"unary_" + op`) | Typed kernel handles |

### Idrak.Text and Idrak.Text.Plugins

| Current File | Convention Followed | Candidate or Proposed Fix |
|---|---|---|
| `Idrak.Text/Abstractions/Distillation.cs:143,155,158,171,281` | `DistillationTeacher` + `TeacherDistributions` abstract classes ("derive from this class"); `Tokenizer => null` and nullable `Fingerprint`, `Check()` passes silently when both are null | `IDistillationTeacher`, `ITeacherDistributions`, a required vocabulary identity |
| `Idrak.Text/Distillation.cs:74` | `teacher is StoredTeacher` decides the on-policy check | `ILiveTeacher`, or `bool Live` on the teacher |
| `Idrak.Text/FineTuningLosses.cs:20`, `FineTuning.cs:935` | `FineTuningLoss` delegate; cannot say it needs pairs, a reference or a teacher | `IFineTuningLoss { NeedsPairs; NeedsReference; NeedsTeacher; Compute }` |
| `Idrak.Text/Pretraining.cs:881`, `PipelinePretraining.cs:52,272,278`, `FineTuning.cs:783,1813,2727,2754` | Module names `embed`/`pos`/`head` and LoRA targets `q,k,v,o,gate,up,down` by name; `is DecoderBlock`, `Head is not Linear`, `TiedTo`, `is not HostOptimizer`, `is AdamW` | `INetworkParts { PartOf; AdapterTargets }`, `ITiedWeights`, `IInPlaceOptimizer` |
| `Idrak.Text/Reinforcement/TextRewards.cs:66-95,228`, `PromptsEnvironment.cs:191` | Episode `info` keys (`"tokens"`) written and read as strings; reward options parsed privately | Typed `EpisodeInfoKey<T>`; `IOptionSchema` |
| `Idrak.Text/TuningDataReaders.cs:363,606,750,900` | Binary parts hard-coded to `"audio"` and the record keys `"images"`/`"audios"` | `IBinaryPartKind.RecordKeys` |
| `Idrak.Text/Abstractions/KeywordIndexes.cs:60,75` | `Register(name, Func<…>, summary)` with a private Entry | `IKeywordIndexKind { Name; Summary; Create }` |
| `Idrak.Text/Generation/ImageReaderAdvisor.cs:28` | Advisor switches on `object Subject` (`is not ImageReader`) | `ISettingAdvisor<TSubject>` |
| `Idrak.Text/Retrieval/Rag.cs:180` (low) | `Retriever as RetrievalIndex` | `ISearchableRetriever` |
| `Idrak.Text/Retrieval/CrossEncoder.cs:21,38` (low) | `EncodePair` Func + a separate shape | `IPairEncoder { Shape; Encode }` |
| `Idrak.Text/Abstractions/SyntheticPageData.cs:22`, `SyntheticPages.cs:1159-1259` (low) | `SyntheticRecord.Write` Func | `ISyntheticRecord` |
| `Idrak.Text.Plugins/TextPlugins.cs:8-12` (same in the Vision, Audio and Reinforcement `.Plugins` packages) | Plug-in class shape by convention: static `XxxPlugin { Plugin; IsRegistered; Register; Unregister }`, per package `{ Package; All; RegisterAll; UnregisterAll; RegisterIdrakPlugin }` | `IIdrakPluginPackage { All }`, exposed as one static `Package` |
| `Idrak.Text.Plugins/Bert/BertFamily.cs:134` | `encoder as SequenceEncoder` | An interface on the encoder |

### Idrak.Vision and Idrak.Vision.Plugins

| Current File | Convention Followed | Candidate or Proposed Fix |
|---|---|---|
| `Idrak.Vision/Abstractions/DetectionDecoders.cs:14-53`, `DetectionDecoder.cs:14`, `DetectionHeads.cs:43-56` | Decoder and head factories are delegates; a head must be registered under its decoder's name; settings are a free-form `JsonObject` | `IDetectionDecoderKind { Name; Keys; Create; CreateHead }` |
| `Idrak.Vision/Abstractions/VisionLosses.cs:41,51` | `SlotTable<string, VisionLoss>` delegate; box, class or mask is a string elsewhere | `IVisionLoss { Name; Summary; Target; Compute }` |
| `Idrak.Vision/Abstractions/BoxMatchers.cs:33,56` | `SlotTable<string, BoxMatcher>` delegate | `IBoxMatcher` |
| `Idrak.Vision/Abstractions/VisionMetrics.cs:61,73` | Factory delegate; a metric must be created to learn its task | `IVisionMetricKind { Name; Task; Create }` |
| `Idrak.Vision.Plugins/Qwen/QwenVision.cs:421`, `Llava/Llava.cs:466`, `IdrakVision/IdrakVision.cs:300`, `Gemma/Gemma3Vision.cs:495` | `IVisionTuningPart` takes any encoder but each requires its own concrete one | Training members on the encoder (`ITrainableVisionEncoder`) |

### Idrak.Audio and Idrak.Audio.Plugins

| Current File | Convention Followed | Candidate or Proposed Fix |
|---|---|---|
| `Idrak.Audio/Abstractions/AudioProcessors.cs:32,44,73,82` | `IAudioProcessor.Live` decides which of `Open`/`Apply` must be overridden; the other throws | `ILiveAudioProcessor { Open }`, `IRecordingAudioProcessor { Apply }`, `Live => this is ILiveAudioProcessor` |
| `Idrak.Audio/Abstractions/AudioAugmentations.cs:106-118` | `Stage` says which of two no-op `Augment` overloads to override | `ISoundAugmentation`, `IFeatureAugmentation` |
| `Idrak.Audio/Abstractions/AdversarialSynthesisTraining.cs:187,244,252` | `Parts => []` paired with `Freeze` that throws; `Generate()` returns a dictionary keyed by `"mel"`, `"kl"`, … | `IPartFreezingSynthesizer`, a typed `AdversarialTerms` record |
| `Idrak.Audio/Abstractions/SpeechRecognition.cs:217`, `SpeechSynthesis.cs:134`, `AudioFamilies.cs:161` + `Idrak.Cli/Commands/Train/CreateCommand.cs:222` | Family `Create(...)` throws by default; callers detect support by catching the exception | `ICreatableSpeechFamily`, `ICreatableSynthesisFamily`, `ICreatableAudioFamily` |
| `Idrak.Audio/Abstractions/SpeechSynthesis.cs:113` + `SpeechSynthesisTrainer.cs:69`, `AdversarialSynthesisTrainer.cs:75`, `Idrak.Cli/…/VoiceTrainCommand.cs:344` | With `Training` set the family must return an `IAdversarialSpeechSynthesizer`; callers cast and throw otherwise | `ITrainableSpeechSynthesisFamily { ReadForTraining }` |
| `Idrak.Audio/Abstractions/Voices.cs:347` | `IVoiceStore.AddCheckpoint` throws by default | `ICheckpointVoiceStore` |
| `Idrak.Audio/Abstractions/AudioFamilies.cs:39-58,94` + `Idrak.Audio.Plugins/IdrakAudio/IdrakAudioLanguage.cs:366` | `PretrainedAudio` abstract class; `IAudioTuningPart` takes any encoder but requires its own | `IPretrainedAudio`, `ITrainableAudioEncoder` |
| `Idrak.Audio/Abstractions/AudioCleaners.cs:43,103` | `Settings` `JsonObject` with undeclared keys; no `Summary`/`Keys` | `Summary` and `Parameters` on `IAudioCleaner` |
| `Idrak.Audio/Abstractions/AudioFeatures.cs:146,158` | `SlotTable<string, Func<AudioFeatureOptions, IAudioFeatureExtractor>>` | `IAudioFeatureKind` |
| `Idrak.Audio/Abstractions/Vocoders.cs:36,51`, `Layers/HifiGanVocoder.cs:245` | Func registry; whether a vocoder takes a mel layout is implicit | `IVocoderKind { Accepts(options); Create }` |
| `Idrak.Audio/Mp3/Mp3EncodeOptions.cs:101` (low) | Encoder options through the constructor or env vars, outside `IAudioEncoder` | `IAudioEncoder.Parameters` |

### Idrak.Reinforcement and Idrak.Reinforcement.Plugins

| Current File | Convention Followed | Candidate or Proposed Fix |
|---|---|---|
| `Idrak.Reinforcement.Plugins/Coding/Abstractions/CodingEnvironmentVariant.cs:16` | Abstract class: `Name`, `Summary`, `Language` + 7 no-op virtuals (`Keys` and `Check` must agree) | `ICodingEnvironmentVariant`, `IVariantOptions`, `IVerificationHooks` |
| `Idrak.Cli/Commands/Train/ReinforceCommand.cs:214,507-566` | `kind is PromptsEnvironmentKind`; `--suite`/`--reward`/`--rounds`/`--sandbox`/`--think`/`--data` map onto option keys by name only | `IPromptDataEnvironmentKind`, `ISuiteEnvironmentKind`, `IRewardedEnvironmentKind` |
| `Idrak.Reinforcement.Plugins/Coding/CodingEnvironment.cs:668-681`, `CSharp/CSharpEnvironmentVariant.cs:169-182`, `CodingRewards.cs`, `CSharpRewards.cs` | Episode info as string keys and string values (`"true"`, numbers as text) | `EpisodeInfoKey<T>` |
| `Idrak.Reinforcement/ReinforcementNames.cs:83`, `Idrak.Reinforcement.Plugins/Classic/ClassicOptions.cs:10` | Option values' types, ranges and defaults only in docs (parsed privately) | `IOptionSchema { Options }` |

### Idrak.Data

| Current File | Convention Followed | Candidate or Proposed Fix |
|---|---|---|
| `Idrak.Data/Abstractions/DataFileFormats.cs:76-81`, `DataFiles.cs:45,84-88` | Code, Parquet and table formats told apart by matching built-in names | `IDataFileFormat.Kind { Table, Binary, Code }` |

### Idrak.Cluster and Idrak.Network

| Current File | Convention Followed | Candidate or Proposed Fix |
|---|---|---|
| `Idrak.Cluster/Profiles.cs:192`, `LinkModel.cs:110`, `TopologyInspection.cs:85`, `TopologyBuilder.cs:602` | `link is MeasuredLink` | `IDeviceMeasuredLink { On(Device) }` |
| `Idrak.Cluster/TopologyInspection.cs:100`, `Idrak.Cli/…/Dashboard/DashboardRuns.cs:616`, `Idrak.Abstraction.Testing/Contracts/ClusterSuites.cs:754` | `topology is ClusterTopology` to reach `Overrides`/`Simulation` | `ISimulatedTopology`, `IOverridingTopology` |
| `Idrak.Cluster/Faults.cs:73` | `_model is LiveFailures` | `ILiveFailureModel` |
| `Idrak.Cluster/InProcessCluster.cs:181` (low) | `transport is SimTransport` | `ISteppableTransport` |
| `Idrak.Network/Training/ClusterSettingAdvisor.cs:47` | `is not ClusterStepMeasure` on `object Subject` | `ISettingAdvisor<TSubject>` |

### Idrak.Cli and Idrak.AspNetCore

| Current File | Convention Followed | Candidate or Proposed Fix |
|---|---|---|
| `Idrak.Cli/CommandContext.cs:326-330` | `-P` plug-ins found by reflection on any `public static void RegisterIdrakPlugin()`, bypassing `IIdrakPlugin` | `[assembly: IdrakPluginPackage(typeof(X))]`, `X : IIdrakPluginPackage` |
| `Idrak.Cli/Shared/ToolAssemblies.cs:34-90`, `Commands/Serve/McpServeCommand.cs:119-160` | Two tool loaders with different reflection rules (`[Tool]` methods + a parameterless constructor) | `IToolProvider { Tools }` declared by an assembly attribute |
| `Idrak.Cli/Commands/Developer/OnnxCommands.cs:187,308-320` | The ONNX runtime found by type name and a static `Load` through reflection | `IOnnxRuntimeLoader` in a registry |
| `Idrak.Cli/Commands/Train/ImageModelSupport.cs:125`, `Run/AgentCommand.cs:81` | "Not registered" recognised by matching the error message's text | `NotRegisteredException(registry, name)` |
| `Idrak.AspNetCore/CompletionsApi.cs:859` | `input_audio` hard-coded like the tuning readers | `IBinaryPartKind.RecordKeys` |
