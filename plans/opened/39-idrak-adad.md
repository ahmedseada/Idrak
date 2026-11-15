# Plan 39: Idrak.Adad (the proof's Adad design as its own devices)

> **Status: under review. Do not follow this plan until the owner marks it ready.** A blind review (2026-11-15) found
> that the record's CUDA lane choices were partly shared between types through colliding tuning keys, that the CPU's
> lane choices are measured anew in every run, that the final CUDA table came from an uncommitted build, and gaps in the
> steps below. The record and this plan are being corrected; earlier versions of this plan (commits 2be7627d, 7e7248fc,
> 1e504c50) are superseded and must not be followed.

The first version of the Idrak.Adad package: the Adad design the proof built and measured on branch `migration`
(commits `5a056e27` and `cf77ceeb`), moved out of Idrak's core into a package of its own that registers two devices,
`adad-cpu` and `adad-cuda`. The plain `cpu` and `cuda` devices keep today's behaviour (the proof's `name` route).

## Rules for the session that follows this plan

1. **Read [39-proof-record.md](39-proof-record.md) first, all of it.** It is what the proof followed, how it was done, the
   choices made and why, what it proved, and what it did not prove. This plan does not repeat it and does not override it.
2. **Build only what the record says the proof did, under the owner's decisions below.** Move the proof's code; do not
   rewrite it, redesign it, rename its lanes' behaviour, or add a lane, a kernel, a setting or a design of your own.
3. **Nothing from the record's "Not proved" list is part of this plan** unless it is listed below as a consequence of an
   owner's decision; proposals made in the proof's session (for example the tensor rows generated per description) are
   not steps here.
4. **Sources: only the proof.** The code on branch `migration` at `cf77ceeb` and later commits of this plan. Never `main`,
   an older Idrak, or the old Adad in the Qiyas clone (`D:\Projects\Qiyas`), which is not read for this plan. The proof's
   code already holds the pieces it ported from Adad's code (the record names them); they are taken from the proof as they
   are, never fetched or re-ported from the clone.
5. **When something is unclear or seems to need a change the record does not have, stop and ask the owner** in one line;
   do not decide it.
6. **Every step is checked against the record**: its tests pass, and step 11's bench reproduces the record's numbers
   within the record's noise. A number out of line is traced before it counts (the record's process, step 6).

## Decisions (the owner's)

1. **Shape: its own devices.** Idrak.Adad registers device providers `adad-cpu` and `adad-cuda` (`-d adad-cuda`,
   `IDRAK_DEVICES=adad-cpu,adad-cuda`). Their backends run the proof's lanes on weights kept as stored rows. `cpu` and
   `cuda` are unchanged from today.
2. **Split: the route code moves; the general fixes stay in core.**
   - Moves to Idrak.Adad: the proof's route code (`ProductRoutes`' choice log, the measurement port `Faster`, the CPU and
     CUDA lanes kept under decision 4, the tensor rows, the grouped and fused stored products, and the kernel generators
     only they use).
   - Stays in core (every device uses it): the bfloat16 cache's tensor-core attention, token-major attention output
     (`IMergedHeadsLayout`, `AttentionVariant.Merged`), key/value writes and queries read as views, the CPU pool fix, the
     dispatcher fix (`TYPED_AS_STORED`), `norm_rope_heads`' measured block size, loading without a float32 copy, the bf16
     head streamed to the device (`UploadTransposed16`), the element-type descriptions (`ScaledBlock`, `SuperBlock`,
     `GridBlock`, the GGUF descriptions and encoders), `idrak trace --stacks`, `idrak profile --pass prompt`.
3. **No `name`, in any scenario, case or stage.** The package never runs, measures or falls back to a kernel chosen by a
   format's name (the proof's `name` route: Idrak's int8, int4 and bfloat16 packed formats and their kernels).
4. **Only what proved fast and efficient, with good bench results.** The package keeps the lanes the proof's final benches
   chose and leaves out the ones they never chose (the record, "Which lanes the final benches chose"):

   | Device | Kept (chosen in the final benches) | Left out (never chosen in the final benches) |
   |---|---|---|
   | CUDA prompts | lane 6, the tensor rows, as built: `bf16B`, `f16B`, `int8B`, `blockB`, `decodeB` and the widened rows, 128- and 64-row tiles, the group launch (`TensorRowsGroup`); for super-blocks and grids lanes 3 and 4 of `StoredSuper` | lane 1 `TypedMma`, lane 2 the native kind, lane 3 the CUDA block route, lane 4 the direct lane, lane 5 the raw int8 lane; `StoredSuper`'s lanes 1 and 2 |
   | CUDA decode | lane 0's few-row kernels from the description, grouped (`_many`), the fused gated and residual + norm passes as measured | |
   | CUDA without bfloat16 products | lane 0's f32 tile from the description | |
   | CPU | the row lane, the tiles lane, the "today's path" lane (widened once into panels), the expanded lane above 8 rows, Adad's block route | the panel lane |

   A measurement still picks among the kept lanes per shape at run time, as in the proof. The proof's `generic` route has
   no place in the package (it is not a lane).

### What decisions 3 and 4 require that the proof did not have (built here, proved only by step 11)

- **The candidates without `RoutedNamed`.** The proof's `measured` and `adad` routes included the named kernels of
  Idrak's packed int8 and bfloat16 formats as candidates; the package's measurement has only the kept lanes.
- **Every weight kept as stored rows on the Adad devices**, the library's int8 and bfloat16 included (the proof's routes
  already did this for the formats they packed: `Quantization.Pack`, `PackRows`, `DecoderBuilder`). A packed weight of a
  named format handed to an Adad device fails loudly naming the format; Idrak's own int4 has no description and fails
  until it is described.
- **The tied bfloat16 head as stored rows on the Adad devices.** Under the proof's routes it was today's bfloat16 head (the
  record, "Where the proof chose to use `name`"). It is built here as the embedding table's stored rows (`[vocab, dim]`
  is the head's `[outputs, inputs]`), shared with the embedding where the embedding can read them; if sharing cannot be
  done without new design, the head is a stored copy (+297 MiB on Qwen3-0.6B, as the proof's first rounds had) and the
  owner is told.

## The package

```
src/Idrak.Adad/
  Idrak.Adad.csproj                 net10.0, references Idrak.Abstraction and Idrak.Gpu; packable as Idrak.Adad
  LibraryRegistrations.cs           [assembly: LibraryRegistrations(...)]: DeviceProviders.Register(adad-cpu, adad-cuda)
  AdadChoices.cs                    the measured choices said once per shape (the proof's ProductRoutes.Chose / Choices /
                                    ForgetChoices)
  Faster.cs                         the CPU's measurement (moved as it is)
  Cpu/
    AdadCpuProvider.cs              IDeviceProvider "adad-cpu": Type Cpu, Count 1, Listed false (asked for by name)
    AdadCpuBackend.cs               : CpuBackend; the override of the stored products (the proof's routed body)
    AdadCpuBackend.Lanes.cs         was CpuBackend.Lanes.cs (without the panel lane)
    AdadCpuBackend.Blocks.cs        was CpuBackend.Blocks.cs
    AdadCpuBackend.Typed.cs         was CpuBackend.Typed.cs
  Cuda/
    AdadCudaProvider.cs             IDeviceProvider "adad-cuda": Type Gpu, Count = CUDA's, the same UUIDs, Listed false
    AdadCudaBackend.cs              : CudaBackend; its capabilities and the overrides of the stored products
    AdadCudaBackend.Typed.cs        was CudaBackend.Typed.cs (lane 0, lane 6, the measurement; without lanes 1-5 and RoutedNamed)
    AdadCudaBackend.Many.cs         was CudaBackend.Many.cs (the few-row groups, the fused epilogues, SharedInput's group
                                    choice; the native lanes' x packs left out)
    AdadCudaBackend.Super.cs        was CudaBackend.Super.cs (lanes 0, 3, 4)
    AdadCudaBackend.TensorRows.cs   was CudaBackend.TensorRows.cs
    AdadKernels.Typed.cs            was PtxKernels.Typed.cs (the f32 tile and the few-row kernels; TypedMma left out)
    AdadKernels.Super.cs            was PtxKernels.Super.cs (widenings, few-row products, decoders)
    AdadKernels.Epilogues.cs        was PtxKernels.Epilogues.cs
    AdadKernels.TensorRows.cs       the modules of the tensor rows' kernels (the gemm_tc_nt_* entries that were in core's
                                    module list, and the block and decoded modules), built with core's TensorCoreGemm
  README.md                         what the devices are, how to pick them, what they run
```

Not carried into the package (decision 4): `CudaBackend.NativeRaw.cs`, `CudaBackend.NativeDirect.cs`,
`PtxKernels.NativeMma.cs` (except `PtxBuilder`, which the core generator uses and stays in core with `BlockDecode` and
`TableWords`), `PtxKernels.Blocks.cs` (the CUDA block route). They stay in the proof's commits.

Both backends subclass core's, so every operation the proof did not touch runs exactly as on `cpu` and `cuda`
(attention, norms, the tensor-core attention fix, the pool, the views), and only the stored-weight products differ.
`cuda:0` and `adad-cuda:0` take the same GPU's primary context (`cuDevicePrimaryCtxRetain`), each with its own memory
pool and streams.

## What moves, file by file

| Proof file | Goes to | Notes |
|---|---|---|
| `Idrak.Abstraction/Devices/ProductRoutes.cs` | `Idrak.Adad/AdadChoices.cs` (the choice log) | The route switch has no place in the package (decisions 3, 4); the public `ProductRoute`/`ProductRoutes` leave core (API change) |
| `Idrak.Abstraction/Devices/Faster.cs` | `Idrak.Adad/Faster.cs` | Unchanged |
| `Idrak.Abstraction/Devices/Cpu/CpuBackend.Lanes.cs`, `.Blocks.cs`, `.Typed.cs` | `Idrak.Adad/Cpu/AdadCpuBackend.*.cs` | The panel lane left out; route checks removed (the device is always the proof's `adad`) |
| The routed part of `CpuBackend.Stored.cs` (6 lines in `StoredMatMulKernel`) | `AdadCpuBackend.cs` (override, then the base) | Core's goes back to the pre-proof body |
| `Idrak.Gpu/Cuda/CudaBackend.Typed.cs`, `.Many.cs`, `.Super.cs`, `.TensorRows.cs` | `Idrak.Adad/Cuda/AdadCudaBackend.*.cs` | The kept lanes only; the overrides of `StoredMatMulKernel`, `StoredMatMulManyKernel`, `StoredMatMulGatedPairKernel`, `StoredMatMulAddRmsNormKernel`, `StoredEpilogueFused`, `StoredMatMulManyAvailable` |
| The route switches in `CudaBackend.Quantized.cs` (`Int8MatMulKernel`, `BFloat16MatMulKernel`) and `RoutedNamed`'s calls | removed from core | Core's packed products as before the proof (`name` only) |
| `StoredFormats = DescribedStoredFormats.Instance` in `CudaBackend.cs` | `AdadCudaBackend`'s capabilities | Core's `cuda` keeps its pre-proof `StoredFormats` |
| `Idrak.Gpu/Cuda/PtxKernels.Typed.cs`, `.Super.cs`, `.Epilogues.cs` | `Idrak.Adad/Cuda/AdadKernels.*.cs` | `partial class PtxKernels` → `AdadKernels`; core helpers through internals |
| The `gemm_tc_nt_*` entries of core's `TensorCoreModules`, `TensorRowsBlockModule`, `TensorRowsDecodedModule` | `AdadKernels.TensorRows.cs` | The generator's modes stay in core's `TensorCoreGemm` |
| `tests/Idrak.Tests/ProductRouteTests.cs` | `tests/Idrak.Tests/AdadTests.cs` | The tests of the kept lanes, on `adad-cpu`, `adad-cuda` by name; general tests stay (step 9) |

Stays in core, as committed: `Backend.cs` (AttentionVariant.Merged, MergesAttention, UploadTransposed16),
`Backend.Views.cs`, `Backend.Routes.cs` (the stored products' operation contracts; their defaults stay false, so plain
devices keep today's path), `CpuBackend.CacheAttention.cs`, `.Layout.cs`, `.Quantized.cs`, `.cs` (pool, views),
`CpuMatMul.cs` (MultiplyPanels, MultiplyPanelsFrom, reached by Adad through internals), `StoredWeight.cs`,
`StoredPacking.cs`, `Quantization.cs`, the loading files (`ITensorStore.cs`, `IWeightSource.cs`, `Architectures.cs`,
`SafeTensors.cs`, `RenamedTensorStore.cs`, `Tensor.Indices.cs`, `DecoderBuilder.cs`), `Decoding.cs`, `KeyValueLayouts.cs`,
`Tensor.Decoding.cs`, `Tensor.Quantized.cs`, `Decoder.cs`, `Linear.cs`, `Linear.Fused.cs`, `ElementType.cs`, `Gguf.cs`,
`GgufElementTypes.cs`, `CudaBackend.Quantized.cs` (attention, merged strides, row threads), `CudaBackend.Tuning.cs` (its
`TuneOp` values keep their numbers), `CudaBackend.Transpose16.cs`, `PtxKernels.FlashTensorCore.cs`,
`PtxKernels.TensorCore.cs` (the generator), the trace tool, `generate.py`.

## Steps

Each step ends with: one build of the changed projects (`-nodeReuse:false`), the targeted tests named in the step, the
CLI reinstalled from the local build (the record, "The CLI reinstall"), and a commit pushed to `origin/migration`.

### Step 1: open core to a subclass (no behaviour change)

1. `Idrak.Abstraction.csproj` and `Idrak.Gpu.csproj`: `<InternalsVisibleTo Include="Idrak.Adad" />`.
2. `CpuBackend` and `CudaBackend`: unsealed (`internal partial class`, still internal); their constructors and the members
   the moved code calls made `internal` (the compiler names them in steps 3 and 4); nothing made public.
3. `PtxBuilder`, `BlockDecode`, `TableWords` moved within core into a generator-helpers file; `TensorCoreGemm` and `Module`
   made `internal`.
4. `LibraryDefaults.Assemblies` and its `DynamicDependency` attributes gain `Idrak.Adad`.
5. Build; `operations:|public API|inventory|product routes` on `cpu`, `product routes|decoder` on `cuda`: all pass, API
   dump unchanged.

### Step 2: the package and its two devices, empty

1. `src/Idrak.Adad/Idrak.Adad.csproj` (as `Idrak.Cluster`'s: no version bump), in `Idrak.slnx`, referenced by the test
   projects and by `Idrak.Cli.Commands` (so `idrak` ships the devices).
2. `AdadCpuProvider` ("adad-cpu", Type Cpu, Count 1, Listed false) and `AdadCudaProvider` ("adad-cuda", Type Gpu, Count,
   `UnavailableReason` and `DeviceUuid` from CUDA's provider, Listed false); `AdadCpuBackend : CpuBackend`,
   `AdadCudaBackend : CudaBackend`, no overrides yet; `LibraryRegistrations` registers both providers.
3. Test group "adad devices": `Device.Parse("adad-cpu")` and `("adad-cuda:0")` resolve to the Adad backends;
   `Device.Available` does not list them; `IDRAK_DEVICES=adad-cpu,adad-cuda` runs `tensor ops` on both.

### Step 3: move the CPU lanes

1. `git mv` `CpuBackend.Lanes.cs`, `.Blocks.cs`, `.Typed.cs` to `Idrak.Adad/Cpu/AdadCpuBackend.*.cs` (class
   `AdadCpuBackend`, namespace `Idrak.Adad.Cpu`); the panel lane removed from the candidates; `Faster.cs` moved.
2. The 6 routed lines of `CpuBackend.StoredMatMulKernel` become `AdadCpuBackend.StoredMatMulKernel` (the proof's routed
   call, then the base); core's goes back to its pre-proof body (`git show cf7b8c48:src/Idrak.Abstraction/Devices/Cpu/CpuBackend.Stored.cs`).
3. In the moved code every route check takes the proof's `adad` branch (the device is always that route); nothing else
   in the lanes changes.
4. Build; the route tests of the CPU lanes run on `adad-cpu` and pass; `cpu` runs today's path.

### Step 4: move the CUDA lanes and their kernels

1. `git mv` `CudaBackend.Typed.cs`, `.Many.cs`, `.Super.cs`, `.TensorRows.cs` to `Idrak.Adad/Cuda/AdadCudaBackend.*.cs` and
   `PtxKernels.Typed.cs`, `.Super.cs`, `.Epilogues.cs` to `AdadKernels.*.cs`; in them, the lanes left out by decision 4
   and `RoutedNamed` removed, every route check takes the proof's `adad` branch, and nothing else in the kept lanes
   changes. `NativeRaw.cs`, `NativeDirect.cs`, `NativeMma.cs` (but `PtxBuilder`) and `Blocks.cs` deleted from the tree
   (they remain in the proof's commits).
2. The overrides of the stored products and `DescribedStoredFormats` leave core's `CudaBackend` for `AdadCudaBackend`; core
   keeps the pre-proof defaults; `CudaBackend.Quantized.cs`'s route switches and `RoutedNamed` calls removed (core = `name`).
3. The `gemm_tc_nt_*` entries leave core's `TensorCoreModules` for `AdadKernels.TensorRows.cs`, loaded once per Adad backend
   (as the proof's block and decoded modules already are).
4. Build; on `adad-cuda` the tests of the kept lanes pass (lane 0, the tensor rows of every mode and both tiles, every
   block type, groups, epilogues with no conversion); on `cuda` the packed products and decoder tests pass as before the
   proof.

### Step 5: the choices log

1. `AdadChoices`: the proof's `ProductRoutes.Chose`, `Choices`, `ForgetChoices`, as they are.
2. The proof's `IDRAK_ROUTES` leaves the environment table (English and Arabic help); no setting takes its place.

### Step 6: the loader asks the device, not the route

1. The proof's route checks in `Quantization.Pack`, `PackRows`, `DecoderBuilder` and `Tensor.Quantized` (the library's int8
   and bfloat16 formats kept as stored rows; an int8 product of any row count sent to the device's product) read the
   device's capabilities instead: `StoredFormats` (it exists: the types kept as stored) and one capability flag in core
   saying the library's int8 and bfloat16 formats are kept as stored on that device; the Adad devices set both.
2. The tied bfloat16 head on the Adad devices (decision 3's requirement above).
3. Tests: the proof's checkpoint test ("a bfloat16 checkpoint ... packs each weight from the file's rows") on `adad-cpu`,
   bit for bit as the values path; on `cpu` the same checkpoint loads as today.

### Step 7: remove the routes from core

1. Delete `ProductRoutes.cs`; nothing in core names a route.
2. `BenchCommand`, `ProfileCommand`, `KernelBench` print `AdadChoices` (the CLI references the package) instead of the
   route's choices; the kernel bench's `prompt` set stays.
3. `python tools/operations/generate.py`: unchanged sources; check the operation count (168).

### Step 8: CLI

1. `-d adad-cpu` and `-d adad-cuda` work on every command that takes a device (device names come from the providers);
   `idrak devices` shows them with their note.

### Step 9: tests

1. `ProductRouteTests.cs` → `AdadTests.cs` (group "adad"): the tests of the kept lanes on the Adad devices (CPU tests on
   `adad-cpu`, CUDA tests on `adad-cuda`), the hooks (`StoredTensorRows`, `StoredTensorRowsMany`, `RowsTileOverride`, …)
   on the Adad backends; the tests of the lanes left out are not carried.
2. Stay in core's test files: merged attention (all devices), the CPU pool test, the trace tool, the element-type decode
   tests.
3. Run `adad|operations:|public API|inventory|decoder|key/value|tensor views|backend dispose` with
   `IDRAK_DEVICES=cpu,adad-cpu`, and `adad|decoder|key/value|attention|tuning` with `IDRAK_DEVICES=cuda,adad-cuda`.

### Step 10: API dump, inventory, docs

1. `IDRAK_UPDATE_API=1` (core loses `ProductRoute`/`ProductRoutes`, gains the capability flag; new `api/Idrak.Adad.txt`);
   `IDRAK_UPDATE_INVENTORY=1`.
2. `src/Idrak.Adad/README.md`; this plan's state in `plans/README.md`.

### Step 11: acceptance bench (the record's bench, exactly)

1. CUDA, Qwen3-0.6B, `--prompt-tokens 512 --tokens 128 -n 3`, one fresh tuning folder per device, a fill run then a warm
   run, for bf16, f16, int8, q8_0, q4_0, iq4_nl, q4_k, q6_k, iq2_xxs, iq3_xxs: `-d cuda` against the record's `name`
   column, `-d adad-cuda` against its `adad` column, within the record's noise (about 3–5% between fresh folders); peak
   memory as the record's.
2. CPU, `--prompt-tokens 128 --tokens 32 -n 2`, bf16, int8, q8_0, q4_0, q4_k: `-d cpu` and `-d adad-cpu` against the
   record's CPU table (noise about 10%).
3. A traced run of each Adad device (`idrak trace --stacks -- bench ... -d adad-cuda`): no host fallback, no conversion
   counted, allocations as the record's.
4. Results in `test-output\adad\`. A line outside the noise is traced (the record's tools) and reported to the owner with
   its cause before the step closes; the bf16 rows show the tied head's change (decision 3) and are reported as such.

## Done when

- `adad-cpu` and `adad-cuda` exist only through the Idrak.Adad package, run the proof's kept lanes, and bench as the
  record's `adad` column; `cpu` and `cuda` bench as its `name` column.
- Core names no route; every test group above passes on its devices; the API dump and inventory are current.
- Everything is on `origin/migration`, a commit per step.

## Not in this plan

- Everything in the record's "Not proved" list that is not a requirement above (the tensor rows generated per
  description, a lanes setting, Vulkan, HIP, descriptions for more types, other machines and models).
- Making the Adad devices the default.
