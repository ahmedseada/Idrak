# Plan 39: Idrak.Adad (the proof's Adad design as its own devices)

> **Status: under review. Do not follow this plan until the owner marks it ready** (this banner removed by the owner's
> word). Earlier versions of this plan (commits 2be7627d, 7e7248fc, 1e504c50, 59aa4611) are superseded and must not be
> followed.

The first version of the Idrak.Adad package: the Adad design the proof built and measured on branch `migration`
(commits `5a056e27` and `cf77ceeb`), moved out of Idrak's core into a package of its own that registers two devices,
`adad-cpu` and `adad-cuda`. The plain `cpu` and `cuda` devices keep the behaviour of the proof's `name` route at
`cf77ceeb` (today's kernels, chosen by the format's name).

## Rules for the session that follows this plan

1. **Read [39-proof-record.md](39-proof-record.md) first, all of it**, including its glossary ("Words used here"). It is
   what the proof followed, how it was done, the tools, the choices and why, what it proved and what it did not prove.
   This plan does not repeat it and does not override it.
2. **Build only what the record says the proof did, under the owner's decisions below.** Move the proof's code; do not
   rewrite it, redesign it, change what a kept lane computes, or add a lane, a kernel, a setting or a design of your own.
   Where a decision needs code the proof did not have, this plan specifies it (below); nothing beyond that.
3. **Nothing from the record's "Not proved" list is part of this plan** unless a decision below requires it.
4. **Sources: only the proof**, commits `5a056e27` and `cf77ceeb`, and commit `cf7b8c48` only to restore core's pre-proof
   bodies that step 3 and step 4 name. Never `main`, an older Idrak, the memory notes as a source of code, or the old Adad
   in the Qiyas clone (`D:\Projects\Qiyas`), which is not read for this plan. The proof's code already holds the pieces it
   ported from Adad (the record names them); they are taken from the proof as they are.
5. **Stop and ask the owner** in one line at every stop point listed under "Stop points", and whenever the code differs
   from what this plan says (a name, a line, a member, a file), before going on. Do not decide it.
6. **CLAUDE.md applies, except where the owner settles otherwise at stop S0** (this plan and CLAUDE.md conflict on the
   branch, fixing without asking, settings, packaging and build cadence). Until S0 is settled, nothing is done.
7. **Every step is checked**: its tests pass, and step 11's bench reproduces the acceptance table within its tolerances.
   A number out of line is traced with the record's tools and reported to the owner with its cause before the step closes.

## Decisions (the owner's)

1. **Shape: its own devices.** Idrak.Adad registers device providers `adad-cpu` and `adad-cuda` (`-d adad-cuda`,
   `IDRAK_DEVICES=adad-cpu,adad-cuda`). `cpu` and `cuda` are unchanged from the proof's `name` route.
2. **Split: the route code moves; the general fixes and the descriptions stay in core.**
   - Moves to Idrak.Adad: the proof's route code (the choice log, the measurement port `Faster`, the CPU and CUDA lanes kept
     under decision 4, the tensor rows, the grouped and fused stored products, and the kernel generators only they use).
   - Stays in core (every device uses it): the bfloat16 cache's tensor-core attention, token-major attention output
     (`IMergedHeadsLayout`, `AttentionVariant.Merged`), key/value writes and queries read as views, the CPU pool fix, the
     dispatcher fix (`TYPED_AS_STORED`), `norm_rope_heads`' measured block size, loading without a float32 copy, the bf16
     head streamed to the device (`UploadTransposed16`), the element-type descriptions (`ScaledBlock`, `SuperBlock`,
     `GridBlock`, the GGUF descriptions and encoders), `idrak trace --stacks`, `idrak profile --pass prompt`.
3. **No `name`, in any scenario, case or stage of the package**: the Adad devices never run, measure or fall back to a
   kernel chosen by a format's name (Idrak's int8, int4 and bfloat16 packed formats and their kernels).
4. **Only what proved fast and efficient, with good bench results**: the package keeps the lanes the benches chose and
   leaves out the ones they never chose (table below).
5. **Every measured choice keyed by the description**, so each type is measured on its own (the record: the proof's keys
   were shared across types).
6. **Decision 3 covers the embedding table and the key/value-cache formats too**: on the Adad devices the embedding table
   and the cache are handled from descriptions, not by Idrak's packed `BFloat16Weight` table or the cache formats' named
   kernels. This is new design beyond the proof (Phase B below).
7. **The tied bfloat16 head on the Adad devices is a stored copy** of the table's rows (+297 MiB on Qwen3-0.6B), read by the
   Adad lanes, as the proof's rounds before round 7 ran it.

### Decision 4's table

Evidence (the table below is filled from it): the re-bench (`plans/opened/39-rebench/`, a copy of `test-output\rebench\`, each type in its own folder: the
acceptance table; the record's "Which lanes each type chose in the re-bench" reads each tuning entry), the record's
"Which lanes the final benches chose", the profiles and the tuning files. A lane counts as chosen only where a log line,
a profile's kernel name or a tuning entry of the re-bench shows it chosen for that type.

| Device | Kept: chosen | Kept: needed though not shown chosen | Left out: never chosen |
|---|---|---|---|
| CUDA prompts | lane 6 (the tensor rows, every mode as built, both tiles); the group launch (over every type's tensor-rows source, the super-blocks and grids decoded as the tiles load included); lane 2 (the native kind); for super-blocks and grids `StoredSuper` lanes 1 (the native kind) and 4 (widened rows) | `Tune`'s fallback, lane 0 (the f32 tile; float32 runs have no tensor rows) | lanes 1 (`TypedMma`), 3 (the CUDA block route), 4 (direct), 5 (raw int8); `StoredSuper` lane 2 (direct) and lane 3 as a product's own candidate (its decoding kernels stay: the group launch runs them) |
| CUDA decode | lane 0's few-row kernels from the description, grouped (`_many`); the fused gated pass (chosen for 7 of the 10 types) and the fused residual + norm pass (chosen for all) | the composition, the epilogues' fallback | |
| CPU | the row lane, the tiles lane, "today's path" (the expanded lane), Adad's block route (each chosen in the re-bench) | | the panel lane (never chosen in the re-bench) |

The native kind is kept both as a lane and as infrastructure. As a lane: a product's own measurement chose it for q in 8
of the 10 types (k and v in q4_k and q6_k, gate/up in iq4_nl too), and it ran for iq3_xxs's q, where the group launch
lost; elsewhere the group launch won and ran instead. As infrastructure: the tensor rows' gates and widened source depend
on its table, pack kernels and loading (the record, "What the tensor rows depend on in the native kind's code"). Changing
those gates is not part of this plan.

## What the decisions require that the proof did not have

These are built here, specified exactly, and proved only by step 11 (and Phase B's own proof).

1. **No `RoutedNamed`, no `NamedLane`/`MeasureLane` on the Adad devices.** The proof's routes had the named kernels of
   Idrak's packed int8 and bfloat16 formats as candidates; the Adad devices have only the kept lanes. Removing candidates
   changes the CUDA keys' candidate count and the CPU tournament's order; different choices from the proof's are expected
   there and are measured, not argued (step 11).
2. **Every weight kept as stored rows on the Adad devices**, the library's int8 and bfloat16 included (the proof's routes
   did this through `ProductRoutes.Current` in `Quantization.Pack`, `PackRows`, `DecoderBuilder`, `Tensor.Quantized`).
3. **The named entry points refuse on the Adad devices**, each throwing `NotSupportedException` naming the format and the
   device ("adad-cuda multiplies weights kept as stored rows; '{format}' is a packed format of its own name"):
   - CUDA (`CudaBackend.Quantized.cs` at `cf77ceeb`): `Int8MatMulKernel`, `Int4MatMulKernel`, `BFloat16MatMulKernel`,
     `PackedMatMulEpilogueKernel`, `PackedMatMulNormAddRmsNormKernel`, and the explicit `IFusedPackedKernels` members
     `PrefersPackedMatMul` (returns false instead of throwing), `PackedMatMulLarge`, `PackedMatMulLowRank`,
     `PackedMatMulGated`, `PackedMatMulAddRmsNorm`, `PackedMatMulMany`, `PackedMatMulGatedPair` (`AdadCudaBackend`
     re-implements `IFusedPackedKernels`).
   - CPU: `Int8MatMulKernel`, `Int4MatMulKernel`, `BFloat16MatMulKernel`.
   - The `Int8/Int4/BFloat16DequantizeKernel` overrides stay (the cache layouts and loaders use them) until Phase B decides.
4. **Keys by description** (decision 5): every key built as `X + Ids.GetOrAdd(..., _ => Ids.Count)` in the moved code
   (`StoredMeasured`'s `TypedIds`, `StoredSuper`'s `SuperIds`, `EpilogueKey`'s `EpilogueIds`, `TensorRowsGroup`'s
   `GroupIds`) keeps its base variant (2000, 3000, 4000/5000, 9700; `StoredMeasured`'s 1000/2000 becomes 2000 alone) and
   puts the description's stable id in its first free field: `E` for `StoredMeasured`, `StoredSuper` and `EpilogueKey`,
   `F` for `TensorRowsGroup`. The stable id is FNV-1a 32-bit over the description code's characters (`TypedDecode.Code`,
   `SuperCode`, `GridCode`), masked to `0x7FFFFFFF`. The other keys of the moved code (`PackedSplits` and
   `PackedMultiSplits` variants 40-43 and 100+, `PackedTile` 200+/1200+, the `GroupIds`-based 100+/300+ variants) get the
   same id the same way. Stop S6 if a key has no free field.
5. **The capability the loader reads** (replacing `ProductRoutes.Current`): `BackendCapabilities.KeepsPackedAsStored`
   (`bool`, `init`, default `false`; doc: "the library's int8 and bfloat16 formats are kept as stored rows for this device
   and its products take every row count"); the Adad devices set it, with `StoredFormats` as the proof's
   `DescribedStoredFormats` (the described types, bfloat16 and float16).
6. **The tied head as a stored copy** (decision 7): `DecoderBuilder`'s branch that keeps today's bfloat16 head under the
   routes (`routed == "bfloat16" && tiedTable is not null && ... headBFloat16`) is not taken on an Adad device; the head
   is packed from the table's rows (`TableRows`), the proof's path before round 7.
7. **The devices**: `AdadCpuProvider` `Name` "adad-cpu", `Type` `DeviceType.Cpu`, `Count` 1, `Listed` false;
   `AdadCudaProvider` `Name` "adad-cuda", `Type` `DeviceType.Cuda`, `Count`, `UnavailableReason`, `DeviceUuid` from CUDA's
   provider, `Listed` false, `DefaultRank` null. `AdadCpuBackend : CpuBackend`, `AdadCudaBackend : CudaBackend` (through
   the constructors `CpuBackend()`/`CudaBackend(int ordinal)` made internal), each overriding `Kind` to "adad-cpu" /
   "adad-cuda" (so the choice log and `CudaMeasuredChoices.Handles` never mix them with `cpu`/`cuda`; stop S7 if
   `Handles` needs more than the kind). Separate backends (`CreateSeparate`) are not offered by the Adad providers.
   `cuda:0` and `adad-cuda:0` share the GPU's primary context and the CUDA tuning file of that GPU in a folder (the keys
   of the kept lanes differ from `cuda`'s).

## The package

```
src/Idrak.Adad/
  Idrak.Adad.csproj                 as Idrak.Cluster's (no version bump); references Idrak.Abstraction, Idrak.Gpu
  LibraryRegistrations.cs           [assembly: LibraryRegistrations(...)]: DeviceProviders.Register(adad-cpu, adad-cuda)
  AdadChoices.cs                    the proof's ProductRoutes.Chose / Choices / ForgetChoices, as they are
  Faster.cs                         from src/Idrak.Abstraction/Devices/Faster.cs, as it is
  StableIds.cs                      the stable id of a description's code (requirement 4)
  Cpu/AdadCpuProvider.cs, AdadCpuBackend.cs, AdadCpuBackend.Lanes.cs, .Blocks.cs, .Typed.cs
  Cuda/AdadCudaProvider.cs, AdadCudaBackend.cs, AdadCudaBackend.Typed.cs, .Many.cs, .Super.cs, .TensorRows.cs,
       AdadKernels.Typed.cs, .Super.cs, .Epilogues.cs, .TensorRows.cs
  README.md
```

## What moves, file by file (every file the proof changed)

| Proof file | Goes to | What is done |
|---|---|---|
| `src/Idrak.Abstraction/Devices/ProductRoutes.cs` | `Idrak.Adad/AdadChoices.cs` | The choice log moves; the route switch (`ProductRoute`, `Current`, `Parse`, `IDRAK_ROUTES`) is deleted |
| `src/Idrak.Abstraction/Devices/Faster.cs` | `Idrak.Adad/Faster.cs` | Moved as it is |
| `.../Cpu/CpuBackend.Lanes.cs`, `.Blocks.cs` | `Idrak.Adad/Cpu/AdadCpuBackend.*.cs` | The panel lane's candidate removed (never chosen); route checks take the `adad` branch |
| `.../Cpu/CpuBackend.Typed.cs` | `Idrak.Adad/Cpu/AdadCpuBackend.Typed.cs` | `NamedLane`, `MeasureLane`, `MeasureInt8`, `MeasureBFloat16`, `Int8Named` and the two packed overrides not carried (requirement 3) |
| `.../Cpu/CpuBackend.Quantized.cs` | stays | `cf7b8c48`'s public `Int8MatMulKernel` and `BFloat16MatMulKernel` restored (the proof made them private `Int8Direct`/`BFloat16Direct`); the in-place `RmsNormRopeKernel` and the merged-output writes stay |
| `.../Cpu/CpuBackend.Stored.cs` | stays | Its 6 routed lines move to `AdadCpuBackend.StoredMatMulKernel` (the proof's call, then the base); core's body as at `cf7b8c48` |
| `.../Cpu/CpuBackend.cs`, `.CacheAttention.cs`, `.Layout.cs`, `CpuMatMul.cs` | stay | Unsealed (`CpuBackend`); members the moved code calls made internal |
| `src/Idrak.Gpu/Cuda/CudaBackend.Typed.cs`, `.Many.cs`, `.Super.cs`, `.TensorRows.cs` | `Idrak.Adad/Cuda/AdadCudaBackend.*.cs` | Lanes 1, 3, 4, 5, `RoutedNamed`, `MeasureRoute`, `Int8Named` not carried (lane 2, the native kind, carried); `StoredSuper` lane 2 and lane 3's candidate not carried (lane 3's decoding source kept for the group launch); route checks take the `adad` branch; keys by description |
| `TypedUnavailableReason`, `_typedLock` (declared in `CudaBackend.Typed.cs`) | stay in core's `CudaBackend` | Core's `CudaBackend.Transpose16.cs` sets them |
| `src/Idrak.Gpu/Cuda/CudaBackend.NativeRaw.cs`, `.NativeDirect.cs` | deleted from the tree | Lanes 5 and 4, never chosen (they stay in the proof's commits) |
| `src/Idrak.Gpu/Cuda/PtxKernels.Typed.cs`, `.Super.cs`, `.Epilogues.cs` | `Idrak.Adad/Cuda/AdadKernels.*.cs` | `TypedMma`'s generator not carried; `TableWords` (in `.Typed.cs`) and `BlockDecode` (in `.Super.cs`) stay in core |
| `src/Idrak.Gpu/Cuda/PtxKernels.NativeMma.cs` | stays in core | `PtxBuilder`, the `NativeKind` table, `NativePadded`, `NativePackGrid`, the pack modules and the native kernel's generator stay (the kept lanes' gates and widened source use them; requirement: the record) |
| `src/Idrak.Gpu/Cuda/PtxKernels.Blocks.cs` | deleted from the tree | The CUDA block route, never chosen; `TableLookup` moves to core first if a kept file calls it (stop S6 if so) |
| `src/Idrak.Gpu/Cuda/PtxKernels.TensorCore.cs` | stays | The calls at lines 122-133 (`cf77ceeb`) move to `AdadKernels.TensorRows.cs`, with `TensorRowsBlockModule` and `TensorRowsDecodedModule`; the generator's modes stay; core's own `gemm_tc_nt_*` kernels (`gemm_tc_nt_f32`, `_gelugrad_`, `_lr_`, `gemm_tc_nt_bf16w_lr_f32`) stay |
| `src/Idrak.Gpu/Cuda/CudaBackend.Quantized.cs` | stays | Its route switches (`Int8MatMulKernel`, `BFloat16MatMulKernel`) and `RoutedNamed` calls removed (`name` only); attention, merged strides, row threads stay |
| `src/Idrak.Gpu/Cuda/CudaBackend.cs` | stays | `StoredFormats = DescribedStoredFormats.Instance` removed (`cf7b8c48`'s value); `LaunchPack` stays |
| `src/Idrak.Gpu/Cuda/CudaBackend.Tuning.cs` | stays | `TuneOp.ProductRoute` keeps its number and name; its doc reworded (no reference to `ProductRoute`); `RowThreads` stays |
| `src/Idrak.Gpu/Cuda/CudaBackend.Transpose16.cs`, `PtxKernels.FlashTensorCore.cs` | stay | |
| `src/Idrak.Abstraction/Generation/Quantization.cs`, `StoredWeight.cs`, `StoredPacking.cs`, `Decoding.cs`, `KeyValueLayouts.cs`, `ITensorStore.cs` | stay | The route checks read `KeepsPackedAsStored` (requirement 5) |
| `src/Idrak.Abstraction/Tensors/*.cs` (`ElementType.cs`, `Tensor.Decoding.cs`, `Tensor.Indices.cs`, `Tensor.Quantized.cs`) | stay | `Tensor.Quantized.cs`'s route check reads `KeepsPackedAsStored` |
| `src/Idrak.Abstraction/Devices/Backend.cs`, `Backend.Views.cs`, `Backend.Routes.cs`, `Backend.Stored.cs`, `BackendCapabilities.cs` | stay | `KeepsPackedAsStored` added |
| `src/Idrak/Layers/*.cs`, `src/Idrak/Models/*.cs` | stay | `DecoderBuilder.cs`: requirement 5 and 6 |
| Generated: `Backend.Operations.g.cs`, `Operations.g.cs`, `OperationCalls.g.cs` | stay | Regenerated only if an operation changes (it should not: 168) |
| `src/Idrak.Cli.Commands/...` `BenchCommand.cs`, `KernelBench.cs`, `ProfileCommand.cs`, `EnvironmentVariables.cs`, `Messages.Arabic.cs`, `ModelChoices.cs`, `ConvertCommand.cs`, `TraceCommand.cs`, `Stacks/*` | stay | Step 8 |
| `tests/Idrak.Tests/ProductRouteTests.cs` | `tests/Idrak.Tests/AdadTests.cs` | Step 9 lists each test |
| `tests/Idrak.Tests/BackendDisposeTests.cs`, `ElementTypeTests.cs`, `CliTraceStacksTests.cs`, `Program.cs`, `data/shared-contracts.txt`, `tests/Idrak.Tests.Library/OperationTests.cs` | stay | `Program.cs` gains the "adad" groups; `OperationTests.DevicePlumbing` unchanged |
| `tools/operations/generate.py`, `api/*.txt`, `plans/10-abstraction-inventory.md` | stay | Regenerated in step 10 |

## Steps

Phase A moves the proof (steps 0-11). Phase B (the embedding table and the cache by description, decision 6) starts only
after the owner approves its design note (stop S9). Each step ends with: one build of the changed projects
(`-nodeReuse:false`), the step's tests, the CLI reinstalled from the local build (the record's commands), and a commit
pushed to the branch S0 names.

### Step 0: settle the conflicts with CLAUDE.md (stop S0)

Ask the owner, in one message, to settle each: the branch (`migration`, against "main only"); a slow path found during
the move is reported and asked about, not fixed (against "fixed without asking"); no setting for the lanes (against
"every behaviour has a setting"); `dotnet pack` to `test-output\packages` for the local CLI (against "do not package");
a build per step (against "build once after all the work"); the CLI printing `AdadChoices` directly (against
"abstraction always"); agents or not. Write the answers into this plan before step 1.

### Step 1: open core to a subclass (no behaviour change)

1. `<InternalsVisibleTo Include="Idrak.Adad" />` in `Idrak.Abstraction.csproj` and `Idrak.Gpu.csproj`.
2. `CpuBackend` and `CudaBackend`: unsealed (still internal); their constructors internal.
3. `TensorCoreGemm` and `Module` internal.
4. `LibraryDefaults.Assemblies` and its `DynamicDependency` attributes gain `Idrak.Adad`.
5. Build; `operations:|public API|inventory|product routes` on `cpu`, `product routes|decoder` on `cuda`: all pass, API
   dump unchanged. (The members the moved code calls are made internal in steps 3 and 4, as the compiler names them.)

### Step 2: the package and its two devices, empty

1. `src/Idrak.Adad/` with the project, `LibraryRegistrations`, the two providers and backends (requirement 7), no
   overrides; in `Idrak.slnx`; referenced by `tests/Idrak.Tests`, `tests/Idrak.Tests.Library` and
   `src/Idrak.Cli.Commands` (so `idrak` ships the devices).
2. Test group "adad devices": `Device.Parse("adad-cpu")`, `Device.Parse("adad-cuda:0")` resolve to the Adad backends,
   their `Kind` is their name, `Device.Available` does not list them, `IDRAK_DEVICES=adad-cpu,adad-cuda` runs `tensor ops`
   on both (stop S8 if `IDRAK_DEVICES` does not take an unlisted device).

### Step 3: move the CPU lanes

As the file table says for `CpuBackend.Lanes.cs`, `.Blocks.cs`, `.Typed.cs`, `.Quantized.cs`, `.Stored.cs`, `Faster.cs`;
`AdadCpuBackend` gets requirement 3's CPU overrides and `KeepsPackedAsStored`. Build; the CPU tests of the kept lanes on
`adad-cpu` pass; `cpu` runs the pre-proof packed products.

### Step 4: move the CUDA lanes and their kernels

As the file table says for the CUDA files; `AdadCudaBackend` gets the stored products' overrides, requirement 3's CUDA
overrides, `KeepsPackedAsStored` and `StoredFormats`; the tensor rows' modules load once per Adad backend. Build; on
`adad-cuda` the tests of the kept lanes pass (step 9's list); on `cuda` the packed products and decoder tests pass as
before the proof.

### Step 5: keys by description

Requirement 4. Test: two types with the same shapes measured in one tuning folder in one process get different keys and
both are measured (a hook counts the measurements).

### Step 6: the loader asks the device

Requirements 2, 5 and 6. Tests: the proof's checkpoint test ("a bfloat16 checkpoint ... packs each weight from the file's
rows") on `adad-cpu`, bit for bit as the values path, with the tied head a stored copy; on `cpu` the same checkpoint loads
as today.

### Step 7: remove the routes from core

Delete `ProductRoutes.cs` and every reference; `TuneOp.ProductRoute`'s doc reworded; `python tools/operations/generate.py`
gives 168 operations unchanged.

### Step 8: the CLI

1. `BenchCommand`: the header's route line (`cf77ceeb` line 512) becomes the device's kind; the saved JSON's `"routes"`
   field (line 561) is no longer written, and `--compare` ignores it in files saved before; the "measured lanes" block
   (condition at line 329) prints `AdadChoices` when the device's backend is an Adad backend.
2. `KernelBench`: `ProductRoutes.ForgetChoices()` becomes `AdadChoices.ForgetChoices()`; the `prompt` set stays.
3. `ProfileCommand`: no new output.
4. `EnvironmentVariables.cs` and `Messages.Arabic.cs`: `IDRAK_ROUTES` removed; nothing added.
5. Check `idrak devices` lists the two devices with their note.

### Step 9: tests

1. `ProductRouteTests.cs` → `AdadTests.cs`, group "adad". Carried, on the Adad devices: the matrix-unit test's f32
   tile, native kind (`StoredNative`) and tensor rows (its `StoredMatrixUnits`, `StoredNativeDirect`, `StoredNativeRaw`
   checks not carried), the tensor rows alone and as a group (the super-blocks and grids included), every block type
   (`ProductRoutesEveryBlockType`), the super-block lanes 0, 1, 4,
   the grouped few-row kernels, the fused epilogues through the dispatchers, the stored layout, q8_0 packing, the CPU
   panels and tiles, the block route on the CPU, the checkpoint test (step 6), the merged attention test. Not carried: the
   tests of `IDRAK_ROUTES` (`ProductRoutesSetting`) and of the generic and measured routes as routes. Stop S10 for any test
   whose place is unclear.
2. Run, with `IDRAK_DEVICES=cpu,adad-cpu`: `adad|operations:|public API|inventory|decoder|key/value|window|attention|tensor views|backend dispose`;
   with `IDRAK_DEVICES=cuda,adad-cuda`: `adad|decoder|key/value|window|attention|tuning|backend dispose|quant types|gguf`
   (the record's groups).

### Step 10: API dump, inventory, docs

`IDRAK_UPDATE_API=1` (core loses `ProductRoute`/`ProductRoutes`, gains `KeepsPackedAsStored`; new `api/Idrak.Adad.txt`),
`IDRAK_UPDATE_INVENTORY=1`; `src/Idrak.Adad/README.md`; this plan's state in `plans/README.md`.

### Step 11: acceptance bench

1. The acceptance table is the re-bench's (`plans/opened/39-rebench/results.txt`, the record's "What it proved"): each
   type in its own fresh tuning folder, warm runs.
2. CUDA: the re-bench's script with `-d adad-cuda` in place of `IDRAK_ROUTES=adad` and `-d cuda` in place of
   `IDRAK_ROUTES=name` (same order: adad-cuda fill, cuda fill, adad-cuda warm, cuda warm; a fresh folder per type shared by
   both devices; types in the order bf16, f16, int8, q8_0, q4_0, iq4_nl, q4_k, q6_k, iq2_xxs, iq3_xxs). Tolerances per
   column against the re-bench's warm runs: prompt and generation within 5%; weights memory equal; peak memory equal
   within 4 MiB, except bf16 (+297 MiB expected: the stored head, decision 7) and what Phase B changes.
3. CPU: the re-bench's CPU part with `-d adad-cpu` and `-d cpu` (a fresh `IDRAK_CPU_TUNING_FILE` per type, two runs each);
   for each type compare both runs and the lanes they chose with the re-bench's; a run whose lanes differ from both of the
   re-bench's runs is reported with its lanes.
4. A traced run of each Adad device (`idrak trace --stacks -- bench ...`): no host fallback (the bench's host-fallback
   count, `-v`), no element conversion counted (`ElementsConverted` telemetry, as the epilogue test reads it).
5. Results in `test-output\adad\`. Stop S11 for any column outside its tolerance and for any ordering against `name`
   that differs from the re-bench's (for example a type that beat `name` and no longer does).

### Phase B: the embedding table and the cache by description (decision 6)

Not started until the owner approves a design note (stop S9). The note says, for the Adad devices: how the embedding
table is kept (stored rows of its description) and read (a lookup generated from the description, as the lanes are), how
the key/value cache's rows are kept (a description) and how attention reads them, what stays of the current formats, how
each is proved (tests bit for bit against the values where the arithmetic allows, else within float32's bound; the bench
against step 11's numbers), and what it costs or saves in memory. Nothing of Phase B is part of Phase A.

## Stop points

- **S0** before step 1: the CLAUDE.md conflicts (step 0).
- **S1** the code differs from a name, line, member or file this plan gives.
- **S2** a kept lane would compute something different from the proof (any change beyond moving, renaming the class,
  taking the `adad` branch, removing a left-out candidate, keys by description).
- **S3** a test of a kept lane fails after a move.
- **S4** a gate of a kept lane depends on a left-out lane in a way the record does not list.
- **S5** a lane this plan leaves out is named by a kept file, test or log after the move (the re-bench chose none of them).
- **S6** a key has no free field for the stable id, or a kept file calls a function in a deleted file.
- **S7** `CudaMeasuredChoices.Handles` or anything else mixes the Adad devices with `cpu`/`cuda`.
- **S8** `IDRAK_DEVICES` or a command does not take an unlisted device.
- **S9** Phase B's design note, before any Phase B code.
- **S10** a test whose place (carried, not carried, core) is unclear.
- **S11** an acceptance column outside its tolerance, or an ordering against `name` that changed.

## Done when (Phase A)

- `adad-cpu` and `adad-cuda` exist only through the Idrak.Adad package, run the kept lanes with keys by description, and
  meet step 11's tolerances; `cpu` and `cuda` meet them against the re-bench's `name` columns.
- Core names no route; every test group above passes on its devices; the API dump and inventory are current.
- Everything is pushed, a commit per step.

## Not in this plan

- Everything in the record's "Not proved" list that no decision requires (the tensor rows generated per description, a
  lanes setting, Vulkan, HIP, descriptions for more types, other machines and models).
- Changing the gates the kept lanes inherit from the native kind.
- Making the Adad devices the default.
- Replacing Idrak.Gpu. The owner may decide it later, only if Idrak.Adad has proved itself and has every backend; until
  the owner says so, Idrak.Adad depends on Idrak.Gpu (its CUDA device is a subclass of `CudaBackend`).
