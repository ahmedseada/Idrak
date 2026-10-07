# Plan 10: Idrak.Abstraction (every contract, its default implementation, and the public device API)

**Status:** phases 0 to 2 done (CPU and an RTX 5070 Ti: 880 of 880; CUDA decoding unchanged), phases 3a and 3b (training,
data and generation contracts) done on the CPU (branch `abstraction`, 2026-10-07). Phase 3c (the rest of Idrak's
contracts) done on the CPU. Open: `EngineModel` and `NetworkOps`; then the contracts of LanguageModels, Datasets and
Onnx. It contains plan 9
(operations as data) and plan 7's item 12c (devices from outside the library).

## Goal

One package, `Idrak.Abstraction`, holds every contract the library is built on, with a default implementation of
each, so that:

- **a plug-in author references one small package** (no GPU code, no language models) and writes a sampler, a KV
  layout, a packed format, a tokenizer part, a dataset source or a whole device against stable types;
- **the library's own GPU backends are built the way an outsider would build one**: CUDA, Vulkan and HIP stay in the
  `Idrak` assembly but see only the public surface of `Idrak.Abstraction` (it grants `Idrak` no internals). If they
  can be built that way, item 12c is done for real, not just declared;
- **every abstraction has one address**: every interface, abstract base, plug-in point and registry, in every library
  package, is declared in the `Idrak.Abstraction` namespace (or one under it) and nowhere else;
- **apps can fix the library without waiting for a release**: an app overrides any contract with its own
  implementation, the library default stays as the fallback, and a proven app implementation moves into the library
  (see "The override loop");
- **the cleanest design, not compatibility** (decision 7): `dotnet add package Idrak` still brings everything, and the
  package adds global usings for its areas, but nothing is kept for code written against older versions.

## Where things are today (measured on main, release 0.3.1)

| What | Count | Where |
|---|---|---|
| Public interfaces | 24 in `Idrak`, 6 in `Idrak.LanguageModels`, 4 in `Idrak.Datasets` | `ITokenSampler`, `ILinearAdapter`, `ISampleSource`, `IImageCodec`, `IToolCallParser`, `ITokenizer`, ... |
| Public abstract bases | 7 in `Idrak`, 1 in `Idrak.LanguageModels` | `Module`, `Optimizer`, `LearningRateScheduler`, `PackedWeight`, `KeyValueLayout`, `ChatTemplate`, ... |
| Registries (`Register`/`Unregister`) | 22 files across `Idrak`, `LanguageModels`, `Datasets`, `Onnx` | `GraphOps`, `RopeScalings`, `KeyValueLayouts`, `ToolCallFormats`, `SampleSources`, `ImageCodecs`, `CheckpointFormats`, `ModelSources`, ... |
| Device contract | `Backend`: 10 abstract and 126 virtual operation methods, **internal** | overridden 130 times by CUDA, 139 by Vulkan, 34 by HIP |
| Internal access shared between assemblies | `Idrak` lets `Idrak.Onnx`, `Idrak.LanguageModels`, `Idrak.Cli` and the tests see its internals; 90 internal members in the `Tensor` files alone | `src/Idrak/Idrak.csproj` |

The obstacle is plain: almost every contract mentions `Tensor`, `Module` or `Device`, so the contracts cannot move
without those types, and the types lean on internals that other assemblies reach through `InternalsVisibleTo`.

## Namespaces

The assembly and the namespace are the same thing: whatever is an abstraction lives in the `Idrak.Abstraction`
assembly **and** under the `Idrak.Abstraction` namespace.

- **What counts as an abstraction**: every interface (public or internal), every abstract class, every registry
  (a type with `Register` / `Unregister`), every plug-in base (`Autograd.Function`, `PackedWeight`, `KeyValueLayout`,
  `ChatTemplate`, ...), and the device contract. Delegates that are extension points (a `Func` a registry stores)
  count too, by their registry.
- **Public types every program names** (`Device`, `DeviceType`, `ComputeResources`, `PackedFormat`, `Tensor`,
  `TensorScope`, `Autograd`, `Module`, ...) go in the root `Idrak.Abstraction` namespace, which the package adds as a global using; the device
  contract (internal until phase 4) is in `Idrak.Abstraction.Devices` and the CPU device in
  `Idrak.Abstraction.Devices.Cpu` (decided in phase 1a).
- **Where it goes**: one namespace per area, mirroring the target layout; the default implementation sits in the
  same namespace as its contract, so `using Idrak.Abstraction.Generation;` gives both `ITokenSampler` and the default
  sampler.

| Namespace | Holds (examples) |
|---|---|
| `Idrak.Abstraction` | `Tensor`, `Device`, `DeviceType`, `TensorScope`: the types nearly every contract mentions |
| `Idrak.Abstraction.Devices` | device providers, storage, capabilities, the CPU device, `IMemoryOffload`, `IHostStaging` |
| `Idrak.Abstraction.Operations` | operation descriptors, kernel keys and requirements, the kernel table, the dispatcher, `IRangeKernel` |
| `Idrak.Abstraction.Autograd` | `Autograd.Function`, `GraphOps` |
| `Idrak.Abstraction.Modules` | `Module`, `RecurrentModule`, `ICachedModule`, `ILinearAdapter`, LoRA, DoRA |
| `Idrak.Abstraction.Training` | `Optimizer`, `LearningRateScheduler`, `ITrainerCallback`, `IScaler`, SGD / Adam / AdamW, schedules |
| `Idrak.Abstraction.Data` | `ISampleSource`, `ISampleStream`, `ISampleTransform`, `IBatchSource`, `IImageCodec`, the PNG / BMP / Netpbm codecs, dataset sources and file formats |
| `Idrak.Abstraction.Generation` | `ITokenizer`, tokenizer components, `ITokenSampler`, `KeyValueLayout`, `PackedWeight`, `IChatModel`, `ChatTemplate`, `IToolCallParser`, `RopeScalings`, the defaults |
| `Idrak.Abstraction.Formats` | `IWeightSource`, checkpoint formats, model sources, ONNX import and export operators |
| `Idrak.Abstraction.Vision` | `IObjectDetector`, `ISegmenter`, `IRegionProposer` (added by the phase 0 inventory) |
| `Idrak.Abstraction.Retrieval` | `IEmbedder`, `IReranker`, `IRetriever`, `IVectorStore` (added by the phase 0 inventory) |
| `Idrak.Abstraction.Serving` | `IPredictor`, engine models (added by the phase 0 inventory) |
| `Idrak.Abstraction.Diagnostics` | `ITelemetryHook` (added by the phase 0 inventory) |
| `Idrak.Abstraction.Testing` (own package) | the conformance kit |

The exact split is settled by the phase 0 inventory; the rule is not. The inventory
([10-abstraction-inventory.md](10-abstraction-inventory.md)) proposes a target for each type and adds four areas the
table first lacked, now in it: `Vision`, `Retrieval`, `Serving` and `Diagnostics` (accepted 2026-10-06).

**Enforced by a test** (phase 0 adds it with an allow list of today's violations; each phase shrinks the list; it
must be empty at the end): it loads every library assembly (`Idrak`, `Idrak.LanguageModels`, `Idrak.Datasets`,
`Idrak.Onnx`, ...) and fails on any interface, abstract class or registry declared outside `Idrak.Abstraction.*`.
Implementations may live anywhere; contracts may not. The CLI is an application, not a library, so its own private
helpers (for example `IToolHost`) are outside the rule.

**What it costs users** (the one break this plan accepts, deliberately): a namespace change is a source and binary
change, so `[TypeForwardedTo]` alone cannot hide it.

- App code that only builds networks, trains and generates mostly names concrete types (`Sequential`, `Trainer`,
  `ChatSession`) and does not change.
- Code that names a contract (`Tensor`, `Module`, `ITokenSampler`, ...) adds `using Idrak.Abstraction;` or the area's
  namespace. The `Idrak` package ships a `buildTransitive` props file adding `Idrak.Abstraction` to the global usings
  of projects that have implicit usings on, so most projects compile unchanged (🤔 to be confirmed in phase 1 on a
  fresh `dotnet new console`).
- Plug-ins compiled against 0.3.x must be rebuilt. The namespace change and the new assembly land together, in one
  minor version (0.4.0), with a migration table in the changelog, while the version is still 0.y.z. Users break
  once, not once per phase (see rule 5).

## Target layout

```
Idrak.Abstraction        no dependencies, no native code; namespace Idrak.Abstraction.*
  Devices                 Device, DeviceType, device providers, storage, capabilities (item 12c)
  Operations              operation descriptors, kernel table, dispatcher (plan 9)
  Tensors                 Tensor, autograd (Autograd.Function), TensorScope
  Modules                 Module, RecurrentModule, ICachedModule, ILinearAdapter
  Training contracts      Optimizer, LearningRateScheduler, ITrainerCallback, ISampleSource / ISampleStream /
                          ISampleTransform / IBatchSource, IScaler
  Generation contracts    ITokenizer, ITokenSampler, KeyValueLayout, PackedWeight, IToolCallParser, IChatModel,
                          ChatTemplate
  Data and format         IImageCodec, IWeightSource, dataset file formats and sources, checkpoint formats,
  contracts               model sources, tokenizer components, ONNX import and export operators
  Registries              every Register / Unregister above, with the built-in names
  Defaults                the CPU device (reference kernels for every operation, the host fallback of every
                          other device), float32 and int8 KV layouts, int8 / int4 / bfloat16 packed weights,
                          the default sampler, LoRA and DoRA adapters, SGD / Adam / AdamW, the step / cosine
                          schedules, the JSON tool-call format, PNG / BMP / Netpbm codecs
  Testing (separate       a conformance kit: the operation tests runnable against any registered device, and a
  package)                contract test per interface (tokenizer round trip, KV layout, packed format, ...)

Idrak                     layers, network builder, trainer, generation, chat, retrieval, inference engine,
                          telemetry, and the GPU devices in Backends/Cuda, Backends/Vulkan, Backends/Hip, each
                          built only on the public surface of Idrak.Abstraction; references Abstraction, so
                          installing Idrak still gives everything

Idrak.LanguageModels, Idrak.Datasets, Idrak.Onnx, ...
                          keep their implementations (GGUF, safetensors, Parquet, ONNX); their contracts move
```

"Default implementation" means the one that makes the contract work out of the box and is small and dependency-free.
Heavy implementations (GGUF reading, Parquet, the GPU kernels) stay in their packages.

## Rules the move keeps

1. **`Idrak.Abstraction` references nothing** (a test reads its metadata and fails on any reference beyond the base
   library).
2. **The GPU devices see no internals of Abstraction.** At the end, `Idrak.Abstraction` grants `Idrak` no
   `InternalsVisibleTo` at all (a test fails if one appears), so the CUDA, Vulkan and HIP code in `Idrak` compiles
   against the public surface alone. Whatever they need becomes public, which is exactly the list item 12c must
   publish. Splitting them into their own packages later is then a move of folders, not a redesign.
3. **One address for every abstraction.** Every interface, abstract base, plug-in point and registry is declared
   under `Idrak.Abstraction.*` and nowhere else, checked by the namespace test above. Concrete layers, trainers and
   engines keep their namespaces (`Idrak.Layers`, `Idrak.Generation`, ...).
4. **Card-agnostic, as every plan.** Kernel requirements read reported capabilities and measurements; no contract
   names a vendor or a card.
5. **Every phase is a separate merge into the `abstraction` branch** that passes the whole list on every device we
   have (CPU, CUDA, Vulkan on lavapipe, then the RTX 5070 Ti and one other real GPU) with the decoding benchmarks
   within 2%. `main` receives the work once, after phase 4, and ships it as 0.4.0; fixes to `main` in between are
   merged into `abstraction` so it never falls behind.

## The override loop (apps fix the library without waiting for a release)

**The problem.** When an app hits a weak spot in the library today, the fix goes: change Idrak, publish to NuGet,
update the app, check. That is slow, and the app cannot ship until the release does. The goal is the opposite order:
the app ships its own implementation of the contract now, the library keeps running its default everywhere else, and
the app's version moves into the library only once it has proved itself.

```
 library default ──► app overrides one contract ──► guarded / shadow run in the app ──► conformance + stress kit green
        ▲                                                                                        │
        └──── app removes its override ◄── NuGet release (bumped) ◄── implementation moves into the library
```

### What exists already (measured on main, release 0.3.1)

- Most registries already let an app replace a built-in: `RopeScalings.Register` says "registers (or replaces)", and
  `PackedWeight.Register`, `NetworkOps.Register`, `ToolCallFormats.Register` work the same way.
- Some plug points are delegates on the object instead: `TextGenerator.CreateSampler` (default `TokenSampler.Create`).
- `tests/Idrak.PluginTests` proves an outside assembly can write an optimizer, an operation, a packed format and a KV
  layout on the public API, and the runner in `tests/Idrak.Tests` runs them on every device.

### What is missing

| # | Gap | Today | After |
|---|---|---|---|
| 1 | **Replacing loses the default** | `Register` overwrites the built-in; `Unregister` of a built-in name leaves nothing | every slot keeps its library default; the app's registration *shadows* it; `Unregister` brings the default back |
| 2 | **No single way to override** | registries by name, delegates on objects, internal virtual hooks, closed switches | one shape for every contract (the slot below); phase 3 moves each registry onto it |
| 3 | **No fallback** | an app implementation that throws breaks the call | per slot, a failure policy: throw, or fall back to the default and report it |
| 4 | **No side-by-side check** | the app cannot see whether its version agrees with the library's on real traffic | a shadow mode runs both on a sample of calls and records differences and timings |
| 5 | **No stress kit for apps** | the contract tests live inside `tests/Idrak.Tests` | `Idrak.Abstraction.Testing` runs the contract tests and stress runs against *any* implementation, from the app's own test project |
| 6 | **No way to know an override is obsolete** | after a NuGet update the app cannot tell the library now ships the same fix | each implementation carries an id and a version; a startup report lists overrides and newer library versions |

### The slot: one shape for every contract

Every contract gets a slot (a registry entry or a named property) holding three layers, resolved in this order:

1. **Per call**: an implementation passed explicitly (an option, a constructor argument), as today.
2. **App**: what the app registered (`Slots.Use<ITokenSampler>(...)`, or `RopeScalings.Register("yarn", ...)` for named
   registries). Registered at startup, before the first use.
3. **Library default**: the implementation the library ships. Never removed; always reachable by name
   (`Defaults.Get<ITokenSampler>()`) so an app's implementation can *wrap* it (fix one case, delegate the rest).

Each implementation declares `Id` (for example `idrak.sampler.top-p`), `Version` and `Origin` (library or app). The
exact API is settled in phase 1 with the device providers; named registries keep their current methods and gain
`Default(name)`, `Origin(name)` and an `Unregister` that restores the default.

### Failure policy (per slot)

| Policy | What happens when the app's implementation throws | When to use |
|---|---|---|
| `Throw` (default) | the error reaches the caller, as today | tests, development: failures must be loud |
| `FallBack` | the call is retried on the library default; the failure goes to telemetry with the slot, id and exception | production, once the override is trusted enough to ship but not trusted alone |
| `Shadow` | the library default answers; the app's version runs on a sample of calls (rate set per slot) and its output, time and memory are compared and recorded | before switching: real traffic as a stress test, with no risk to answers |

⚠️ Limits, stated up front:

- Fallback works at **call boundaries only**. A stateful implementation (a KV layout half-way through a sequence, a
  sampler holding penalty history, a device kernel inside a captured CUDA graph) cannot switch mid-way; for those the
  fallback happens when the object is **created** (creation fails → default object), and `Shadow` compares whole runs.
- Shadow mode on device operations doubles the work on sampled calls; it is for staging and benchmarks, not for every
  production request.
- "Agrees with the default" needs a comparison per contract (exact for tokenizers and parsers, a tolerance for
  floating-point outputs, the same top-1 text for samplers). The testing kit owns these comparisons, so shadow mode
  and the conformance tests use the same ones.

### The testing kit as the app's stress harness

`Idrak.Abstraction.Testing` (phase 5) is what the app references in its own test project:

- **Conformance**: `Conformance.Check<ITokenizer>(myTokenizer)` runs every contract test the library runs on its own
  default (round trips, shapes, gradients against finite differences, thread safety where the contract promises it).
- **Stress**: `Stress.Run(slot, budget)` drives the implementation with generated inputs (sizes, empty and huge inputs,
  odd Unicode, many threads, long sessions) on every device present, comparing against the library default with the
  kit's comparison, and writes a report (failures, slowest cases, memory).
- **Regression capture**: a failing shadow sample or stress case is saved as a small reproducible case file; the kit
  replays saved cases, so the case that made the app write its override becomes a test the library inherits.

### Promotion: the app's solution, generalized, becomes the library default

The app's implementation is **not** copied into the library. It is the proof that a better solution exists; what moves
is that solution made general. An app's override usually leans on things only that app knows (its vocabulary, its one
model family, its device, its input sizes, its constants); the library default cannot.

**1. The app proves the idea.** Its override passes the conformance kit and runs in `Shadow` or `FallBack` on real
traffic with no unexplained differences (the telemetry shows it). Its failing cases are saved by the kit.

**2. The idea is generalized in the library.** A new implementation is written in the library from the app's one:

- every app-specific constant becomes a parameter, a measurement or a capability read from the device (rule 4: no
  card, model or app is named);
- it runs on every device (CPU reference first, then the device kernels, or the host fallback where none exists yet);
- it handles every input the contract allows, not only the inputs the app sends;
- when the better solution needs something the contract does not give (more context, another hook, a different
  shape), **the contract changes**: a new member with a default body when possible, otherwise a deliberate public API
  change under phase 7's rules, in the same release.

**3. The general version is checked against the app's.** Before any release, on the library's source
(`IdrakFromSource`, below):

- the kit's conformance and stress runs pass on CPU, CUDA and Vulkan (lavapipe at least);
- the app's saved cases pass, and they move into `tests/` with it;
- the app itself drops its override, runs on the general version and stays as good: the same comparisons in `Shadow`,
  against its own old override this time;
- the decoding and training benchmarks stay within 2% where it is on a hot path.

**4. It replaces the default.** A minor release ships it as the slot's default with a new `Version` and a changelog
line saying what changed and which app case motivated it. The previous default stays selectable by name for one more
minor release (`Defaults.Get<ITokenSampler>("idrak.sampler.top-p", version: 1)`), so an app that sees a regression
after updating has a one-line way back while it reports it.

**5. The app updates and deletes its override.** The startup report says "your override of `idrak.sampler.top-p` is
older than the library default v2". Every other app gets the better default on its next update; a new app starts from
it.

### Faster than NuGet while working on the library itself

For fixes that belong in the library from the start: a `IdrakFromSource` property in the app's
`Directory.Build.props` switches the `PackageReference` to `ProjectReference`s on a local clone, so a library change
is tested in the app on the next build, with no publish. A local NuGet feed (`dotnet pack` into a folder) checks the
packaged form before the real release.

## Phases

| # | Phase | Done when |
|---|---|---|
| 0 | **Inventory.** Every interface (public and internal), abstract base and registry in every library package, and every operation of `Backend`, with: what it references, its default implementation, which internals other assemblies use, and its target namespace under `Idrak.Abstraction`. Generated by a test so it cannot drift. The namespace test lands here too, with today's violations as its allow list | done: `plans/10-abstraction-inventory.md` (generated by `tests/Idrak.Tests/InventoryTests.cs`, which fails when it drifts) and the namespace test with its allow list (`tests/Idrak.Tests/data/abstraction-allow-list.txt`, 74 types) |
| 1a | **The project, with devices.** Create `Idrak.Abstraction`; move `Device`, `DeviceType`, `ComputeResources`, `PackedFormat`, the device contract (`Backend`, `Storage`, capabilities, providers) and the CPU device in as the default implementation and host fallback. The GPU providers stay in `Idrak`, which registers them; `Idrak.Abstraction` finds them by name when the application ships `Idrak`, so `Device.Available` lists them before any other type of Idrak is used | done: `Idrak` references Abstraction; nothing else changes (436 of 436 on the CPU; GPUs to run); a test checks the GPU providers come first and Abstraction references nothing of Idrak |
| 1b | **Operations.** Plan 9's operation descriptors, kernel table and dispatcher in `Idrak.Abstraction.Operations`; the old `Backend` methods registered through an adapter; every call site goes through the dispatcher | done on the CPU (see plan 9, "Phase 1 as built"): every call site goes through `Backend.Name(...)`; registered kernels tested on every device; about 1 ns per call; GPUs and the decoding gate to run |
| 2 | **Tensors and modules.** Move `Tensor`, autograd, `TensorScope`, `Module` and the module contracts; turn the internals other assemblies use into public or protected members where they belong to the contract, and keep the rest internal behind a narrow, documented bridge during the move | the internals list from phase 0 is empty or justified line by line |
| 2 (as built) | **2a**: `Tensor` (operations, autograd, decoding and distillation parts), `TensorScope`, `Autograd`, `DifferentiableFunction`, `ActivationMemory`, `MixedPrecision`, `ComputeGraph` move to the root namespace; the 24 internal members that take layer types stay in Idrak as `TensorLayerPaths`; telemetry and offloading reach Idrak through hooks (`OperationTelemetry`, `TensorOffloading`). **2b**: `Module` moves (root namespace); its weights files stay in Idrak as `ModuleFiles` extension methods (`model.Save(path)` unchanged); `Forward`/`Predict` report and stage weights through `ModuleHooks`. `RecurrentModule`, `ICachedModule` and `ILinearAdapter` move in phase 3 with the decoding and adapter contracts | done on the CPU: every internal another library assembly uses is listed with why in `tests/Idrak.Tests/data/internals-justified.txt`, which a test keeps exact (36 lines; Idrak's own use of Abstraction is phase 4's) |
| 3 | **The other contracts and registries.** Training, generation, data and format contracts of `Idrak`, then those of `LanguageModels`, `Datasets` and `Onnx`, each with its registry and default | every registry lives in Abstraction; the satellites keep only implementations |
| 3a (as built) | **Training and data.** `Optimizer`, `Sgd`, `Adam`, `AdamW` and the schedules (`Idrak.Abstraction.Training`), `IScaler`; the sample-source contracts, `Batch`, `SampleSources`, and `IImageCodec`/`ImageCodecs` with the PNG, BMP and Netpbm codecs (`Idrak.Abstraction.Data`). Built-ins that live in Idrak (its GPU devices, the csv/images/tokens/npy sources) are registered by `LibraryRegistrations` the first time any registry is used (`LibraryDefaults.Ensure`). Stay in Idrak: `AdamW8Bit`, `GroupedOptimizer`, `HostOptimizer`, the scalers (they take `Dataset`). `ITrainerCallback` moved after decision 7: `TrainerContext` no longer exposes the concrete `Trainer` (it carries the model, optimizer, epochs, step, history and `Stop`); with it `TrainingHistory` and the telemetry event records (`Idrak.Abstraction.Diagnostics`) | done on the CPU; allow list 66 to 54 |
| 3b (as built) | **Generation.** Into `Idrak.Abstraction.Generation`: tokenizers, chat (templates, messages, `IChatModel`), tool-call parsing (`IToolCallParser`, `ToolCallFormats` and every built-in parser), the token sampler with `GenerationOptions`, packed weights (`PackedWeight` with int8, int4 and bfloat16), key/value caches (`KeyValueLayout`, `KeyValueLayouts`, `KeyValueCache`, `DecodingContext`, `ICachedModule`) and `RopeScalings`. `TensorLayerPaths` is gone: packed products and KV-cache attention are `Tensor` members again; the fused paths over several layers belong to `Linear`, `RMSNorm` and `PackedSequences`; gradient checkpointing is `Checkpointing`. **Not moved: `ILinearAdapter`** (it takes the `Linear` layer; next step: pass it the layer's sizes, weights and base product instead) | done on the CPU; allow list 54 to 42 |
| 3c (as built) | **The rest of Idrak's contracts.** `ILinearAdapter` redesigned: adapters receive an `ILinearLayer` (sizes, weight values, base product) instead of `Linear`, and move with LoRA and DoRA (`Modules`). `RecurrentModule` (`Modules`; LSTM and GRU stay with the layers), `IWeightSource` and the weight codecs (`Formats`), telemetry whole (`Diagnostics`: with it in Abstraction, tensors and modules report directly and the operation and layer hooks of phase 2 are gone), retrieval contracts (`Retrieval`), vision contracts with boxes, foreground, components, segmentation and `ModelDetector` (`Vision`), `IPredictor` (`Serving`). `LayerTypes` and `GraphOps` (with `GraphOpContext`, `GraphNode`) move as registries whose built-ins Idrak registers (`LibraryLayerTypes`, `LibraryGraphOps`); `GraphOps.IsKnown` became the public `Contains`. `NetworkOps` moves after decision 9: steps see an `INetworkBuilder` (`CurrentShape`, `Lambda`, `Add` of a layer factory, `Op` to compose registered steps), which `NetworkBuilder` implements; built-in steps are registered by Idrak (`LibraryNetworkOps`). **Not moved yet**: `EngineModel` (wave 1, the model-kind contract) | done on the CPU; allow list 42 to 26 |
| 3d (as built) | **Datasets and Onnx contracts** (wave 1). Into `Idrak.Abstraction.Data`: file formats, Parquet codecs, text normalizers, dataset sources with `DatasetSpec`, `ReadOptions`; new contracts `IDatasetRows` (what a source returns; `Dataset` implements it) and `IDownloader` (`Downloader` implements it). Into `Idrak.Abstraction.Formats`: the ONNX import and export registries with their translators and contexts; `OnnxGraph` and `OnnxImportContext` became abstract, and import translators add network steps by name instead of taking the concrete builder. `OnnxBuiltIns` folded into the built-in registration. Built-ins registered by `Idrak.Datasets.` / `Idrak.Onnx.LibraryRegistrations` | done on the CPU (440 of 440); allow list 26 to 16 |
| 4 | **Devices on the public API (item 12c).** Remove the last `InternalsVisibleTo` from `Idrak.Abstraction` to `Idrak`; the CUDA, Vulkan and HIP code in `Idrak` builds against the public surface alone | `Idrak.Abstraction` grants no internals; every device passes; `dotnet add package Idrak` behaves as before |
| 5 | **A device from outside.** A plain-loop reference device in `tests/Idrak.PluginTests`, written against the public API only, passes the conformance kit | the kit (`Idrak.Abstraction.Testing`) runs it green |
| 6 | **Kernels for plug-ins.** `Autograd.Function`, `GraphOps`, `PackedWeight` and `KeyValueLayout` implementations register device kernels | a plug-in packed format and KV layout run their own device kernel |
| 6b | **Slots and failure policies.** Every slot keeps its default under an app's registration; `Unregister` restores it; `Throw`, `FallBack` and `Shadow` per slot, with fallback at creation for stateful contracts; the startup report of overrides | a test app overrides a sampler, a tokenizer part and a RoPE scaling: each one throws once and falls back under `FallBack`, and is compared on sampled calls under `Shadow` |
| 6c | **The kit as the app's stress harness.** `Conformance.Check`, `Stress.Run` and saved regression cases, usable from an app's test project on the NuGet package alone | a sample app (in `samples/`) overrides one contract and runs the kit green from its own tests |
| 6d | **One promotion end to end.** A sample app's override is generalized in the library (steps 2 to 5 of "Promotion"), becomes the default, the old default stays selectable by version, and the app deletes its override | the sample app passes on the new default with no override; the previous default is still reachable by version |
| 7 | **The surface is guarded.** A checked-in dump of the public API of **every library package** (Abstraction, core, Gpu, Data, Nlp, Vision, Diffusion and Audio when they exist, and the bridges), compared by a test on every build; an intended change updates the dump in the same commit (decided 2026-10-07); a changelog section per change | an accidental public change fails the build; the namespace test's allow list is empty |

## What changes for whom

| Who | Before | After |
|---|---|---|
| App developer | `dotnet add package Idrak` | the same; code naming a contract adds `using Idrak.Abstraction...` (automatic with implicit usings) |
| Plug-in author | references all of `Idrak` (GPU code included); contracts spread over 15 namespaces | references `Idrak.Abstraction` (and the testing kit); every contract under `Idrak.Abstraction.*`; rebuilds once for 0.4.0 |
| Device author | not possible (internal `Backend`) | a device on the public API (the same one CUDA, Vulkan and HIP use), checked by the conformance kit |
| Library maintainers | one large `Idrak` assembly with shared internals | clear layers; the GPU devices prove the public API is enough |

## Risks

| Risk | Answer |
|---|---|
| Hot paths slow down across assemblies | .NET inlines across assemblies; the dispatcher caches per (operation, device); phase 1 and 4 are gated on the benchmarks |
| Internals that do not belong in a public contract | phase 2 lists each one; the fix is a narrower public member or a contract method, never `InternalsVisibleTo` from Abstraction to `Idrak` |
| Graph capture (CUDA, Vulkan) depends on call order | the same kernels run in the same order; the graph tests run every phase |
| The public API freezes too early | marked preview through 0.y.z; phase 7 makes every change deliberate |
| The namespace change breaks users' code | one minor version (0.4.0), a migration table, the transitive global using; done while the version is 0.y.z |
| The move is large (400+ files) | eight separate merges; phase 1 lands without touching any backend's kernels |

## Target package layout (decided 2026-10-07)

Large packages by domain, plus bridges only where a third-party dependency would otherwise reach every app. Every
domain talks to the others through contracts in `Idrak.Abstraction` (decision 9), so the inference engine, the CLI and
apps combine domains without the domains knowing each other.

| Package | Holds | Depends on |
|---|---|---|
| `Idrak.Abstraction` | every contract (including the model-kind contract of the engine), `Tensor`, `Module`, autograd, the CPU device, small dependency-free defaults | — |
| `Idrak.Gpu` | the CUDA, Vulkan and HIP devices and all GPU kernels (reverses decision 3; needs phase 4 first) | Abstraction |
| `Idrak` (core) | layers, network builder, trainer, data loaders, optimizers, ONNX import and export, the **inference engine** and model packages, **model loading** (safetensors, GGUF, hub sources) and **tokenizers** (both from LanguageModels), generic LoRA attach and merge | Abstraction |
| `Idrak.Data` | today's Datasets: file formats, Parquet, hub downloads, chat rows | Idrak |
| `Idrak.Nlp` | generation and chat, Jinja templates, LLM fine-tuning (QLoRA, DoRA, PEFT, distillation, evaluation), retrieval and RAG, the coding agent and tools; registers the text and chat model kinds | Idrak, Data |
| `Idrak.Vision` | region classification, content framing, model-based detection and segmentation, later OCR and detectors (decided 2026-10-07) | Idrak |
| `Idrak.Diffusion` (planned) | UNet, DiT, VAE, schedulers, text encoders, text-to-image pipelines; registers an image model kind | Idrak |
| `Idrak.Audio` (planned) | audio decoding, spectrograms, speech recognition, text-to-speech, vocoders; registers transcription and speech model kinds | Idrak |
| bridge `Idrak.Mcp` | MCP client (tools of any MCP server, for agents and chat) and server (an app's tools over MCP, `idrak serve --mcp`, and the engine's models as MCP tools: `generate`, `chat`, `embed`, later `transcribe` and `image`, through the model-kind contract), over the tool contracts (`Tool`, the tool registry, `ToolResult`, moving to Idrak.Abstraction) | Abstraction, ModelContextProtocol |
| bridge `Idrak.Onnx.Runtime` | ONNX Runtime models as modules | Idrak, Microsoft.ML.OnnxRuntime |
| bridge `Idrak.AspNetCore` | thin HTTP endpoints over the inference engine (`MapChatApi`, `MapCompletionsApi`, `MapGenerate`, `MapPredictor`), speaking only to the engine and the contracts | Idrak, ASP.NET (the shared framework) |
| app `Idrak.Cli` | the `idrak` tool (`idrak serve` uses the bridge) | everything |

Retired: `Idrak.LanguageModels`, `Idrak.Datasets` (renamed `Idrak.Data`), `Idrak.Onnx` (into core). Dependencies run one way: Abstraction ← Idrak ← Data ← Nlp; Abstraction ← Gpu; Idrak ← Vision,
Diffusion, Audio; the bridges depend on contracts (and their third-party package) only where they can.

## Decided (2026-10-05)

1. Package name: **`Idrak.Abstraction`**.
2. The CPU device is **inside `Idrak.Abstraction`**, as the default implementation and every device's host fallback.
3. The GPU devices **stay in the `Idrak` assembly** (Backends/Cuda, Backends/Vulkan, Backends/Hip), built only on
   the public API of `Idrak.Abstraction`; no separate device packages for now.
4. The contracts of `Idrak.LanguageModels`, `Idrak.Datasets` and `Idrak.Onnx` **move into `Idrak.Abstraction`** too:
   all contracts in one place; the satellites keep their implementations.
5. **Every abstraction lives under the `Idrak.Abstraction` namespace**: every interface, abstract base, plug-in point
   and registry of every library package, with its default implementation beside it; a test enforces it. This
   replaces the earlier "namespaces stay" rule.
6. **A separate package from the start**, not a namespace inside `Idrak` first: the contracts move into the
   `Idrak.Abstraction` assembly and namespace together, so the compiler (not only a test) keeps the GPU devices on the
   public API. The cost is the up-front work on `Tensor`'s internals (phase 2).
7. **No backward compatibility** (2026-10-07): the library has one user, its author, so every phase takes the cleanest
   design and breaks whatever it needs to; no shims, forwarders or compatibility members are kept for older code. The
   changelog still records what changed. This replaces the earlier "users change little" goal and the one-release
   migration rules (rule 5's single break, the migration table).
8. **The package layout above** is the end state (2026-10-07): domains as large packages, the inference engine in core
   with model kinds as a public contract, the ASP.NET endpoints kept as a thin bridge (`Idrak.AspNetCore`) over the engine and
   the contracts, the GPU devices in one package.
9. **Everything is a contract** (2026-10-07): every point a domain, an app or a plug-in can supply or replace is a
   public contract in `Idrak.Abstraction`, including the engine's model kinds (`EngineModel` today) and the network
   builder's steps (`NetworkOps` today), so the override loop works everywhere.

## Open (the override loop)

1. Default failure policy in production: `Throw` everywhere (proposed: loud by default, `FallBack` opted into per
   slot) or `FallBack` everywhere.
2. Shadow sampling: a rate per slot (proposed, default 1%) or a global rate.
3. Where shadow and fallback reports go: the existing telemetry only (proposed), or also a file the kit can replay.
