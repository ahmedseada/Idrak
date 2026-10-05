# Plan 10: Idrak.Abstractions (every contract, its default implementation, and the public device API)

**Status:** planned (branch `abstraction`, 2026-10-05). Planning only: nothing is built yet. It contains plan 9
(operations as data) and plan 7's item 12c (devices from outside the library).

## Goal

One package, `Idrak.Abstractions`, holds every contract the library is built on, with a default implementation of
each, so that:

- **a plug-in author references one small package** (no GPU code, no language models) and writes a sampler, a KV
  layout, a packed format, a tokenizer part, a dataset source or a whole device against stable types;
- **the library's own GPU backends are built the way an outsider would build one**: CUDA, Vulkan and HIP become
  packages that use only the public surface of `Idrak.Abstractions`. If they can, item 12c is done for real, not
  just declared;
- **users see no change**: `dotnet add package Idrak` still brings everything, namespaces stay as they are, and moved
  types are forwarded.

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

## Target layout

```
Idrak.Abstractions        no dependencies, no native code
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

Idrak.Cuda, Idrak.Vulkan, Idrak.Hip
                          each a device built only on the public surface of Idrak.Abstractions

Idrak                     layers, network builder, trainer, generation, chat, retrieval, inference engine,
                          telemetry; references Abstractions and the three device packages, so installing it
                          still gives everything

Idrak.LanguageModels, Idrak.Datasets, Idrak.Onnx, ...
                          keep their implementations (GGUF, safetensors, Parquet, ONNX); their contracts move
```

"Default implementation" means the one that makes the contract work out of the box and is small and dependency-free.
Heavy implementations (GGUF reading, Parquet, the GPU kernels) stay in their packages.

## Rules the move keeps

1. **`Idrak.Abstractions` references nothing** (a test reads its metadata and fails on any reference beyond the base
   library).
2. **Device packages see no internals.** `Idrak.Cuda`, `Idrak.Vulkan` and `Idrak.Hip` get no `InternalsVisibleTo`
   from Abstractions; a test fails if one appears. Whatever they need becomes public, which is exactly the list item
   12c must publish.
3. **No source break for users.** Namespaces stay (`Idrak`, `Idrak.Layers`, `Idrak.Generation`, ...); moved types
   carry `[TypeForwardedTo]` from their old assembly.
4. **Card-agnostic, as every plan.** Kernel requirements read reported capabilities and measurements; no contract
   names a vendor or a card.
5. **Every phase is a separate merge** that passes the whole list on every device we have (CPU, CUDA, Vulkan on
   lavapipe, then the RTX 5070 Ti and one other real GPU) with the decoding benchmarks within 2%.

## Phases

| # | Phase | Done when |
|---|---|---|
| 0 | **Inventory.** Every public interface, abstract base and registry in every package, and every operation of `Backend`, with: what it references, its default implementation, which internals other assemblies use, and its target (Abstractions or stays). Generated by a test so it cannot drift | `plans/10-abstractions-inventory.md` exists and its test passes |
| 1 | **The project, with devices and operations.** Create `Idrak.Abstractions`; move `Device`, providers, storage and capabilities; add plan 9's operation descriptors, kernel table and dispatcher (the old `Backend` methods registered through an adapter); the CPU device moves in as the default implementation and host fallback | `Idrak` references Abstractions; nothing else changes; every device passes |
| 2 | **Tensors and modules.** Move `Tensor`, autograd, `TensorScope`, `Module` and the module contracts; turn the internals other assemblies use into public or protected members where they belong to the contract, and keep the rest internal behind a narrow, documented bridge during the move | the internals list from phase 0 is empty or justified line by line |
| 3 | **The other contracts and registries.** Training, generation, data and format contracts of `Idrak`, then those of `LanguageModels`, `Datasets` and `Onnx`, each with its registry and default | every registry lives in Abstractions; the satellites keep only implementations |
| 4 | **Devices as packages (item 12c).** `Idrak.Cuda`, `Idrak.Vulkan`, `Idrak.Hip` move out of `Idrak`, built only on the public API; `Idrak` references them | the device packages build with no `InternalsVisibleTo`; every device passes; `dotnet add package Idrak` behaves as before |
| 5 | **A device from outside.** A plain-loop reference device in `tests/Idrak.PluginTests`, written against the public API only, passes the conformance kit | the kit (`Idrak.Abstractions.Testing`) runs it green |
| 6 | **Kernels for plug-ins.** `Autograd.Function`, `GraphOps`, `PackedWeight` and `KeyValueLayout` implementations register device kernels | a plug-in packed format and KV layout run their own device kernel |
| 7 | **The surface is guarded.** A checked-in dump of the public API of Abstractions, compared by a test on every build; a changelog section per change; versioned separately from the rest if needed | an accidental public change fails the build |

## What changes for whom

| Who | Before | After |
|---|---|---|
| App developer | `dotnet add package Idrak` | the same; nothing to change |
| Plug-in author | references all of `Idrak` (GPU code included) | references `Idrak.Abstractions` (and the testing kit) |
| Device author | not possible (internal `Backend`) | a device package on the public API, checked by the conformance kit |
| Library maintainers | one large `Idrak` assembly with shared internals | clear layers; the device packages prove the API is enough |

## Risks

| Risk | Answer |
|---|---|
| Hot paths slow down across assemblies | .NET inlines across assemblies; the dispatcher caches per (operation, device); phase 1 and 4 are gated on the benchmarks |
| Internals that do not belong in a public contract | phase 2 lists each one; the fix is a narrower public member or a contract method, never `InternalsVisibleTo` to device packages |
| Graph capture (CUDA, Vulkan) depends on call order | the same kernels run in the same order; the graph tests run every phase |
| The public API freezes too early | marked preview through 0.y.z; phase 7 makes every change deliberate |
| The move is large (400+ files) | eight separate merges; phase 1 lands without touching any backend's kernels |

## Open questions (for the maintainer)

1. Package name: a) `Idrak.Abstractions`, the .NET convention for contract packages (proposed) · b) `Idrak.Abstraction`
2. The CPU device as the default implementation: a) inside Abstractions, so every device has a host fallback out of
   the box (proposed) · b) its own package `Idrak.Cpu`
3. Device packages: a) `Idrak.Cuda`, `Idrak.Vulkan`, `Idrak.Hip` as separate NuGet packages that `Idrak` references
   (proposed) · b) folders in one assembly that only promise to use the public API
4. Contracts of `LanguageModels` and `Datasets`: a) into Abstractions too (proposed, "all contracts in one place")
   · b) a second contracts package per area
