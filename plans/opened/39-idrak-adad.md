# Plan 39: Idrak.Adad (the proof's Adad design as its own devices)

The first version of the Idrak.Adad package: the Adad design built and tested in the proof on branch `migration`
(commits `5a056e27` and `cf77ceeb`), moved out of Idrak's core into a package of its own that registers two devices,
`adad-cpu` and `adad-cuda`. The plain `cpu` and `cuda` devices keep today's behaviour (the proof's `name` route).

## Source of truth

- **Only the proof**: the code on branch `migration` at `cf77ceeb` and everything this plan builds on it. Not `main`,
  not an older Idrak, not the old Adad in the Qiyas clone (`D:\Projects\Qiyas`), which is never read for this plan.
- **What "the proof" means**: every lane, kernel, measured choice, test and fix of the routes experiment
  (`IDRAK_ROUTES=generic|measured|adad`), as committed. Nothing is rewritten from scratch: files move, their classes are
  renamed where an assembly boundary needs it, and the route checks become the devices' own settings.
- **The proof's numbers are the acceptance bar** (step 11): `adad-cuda` must give the proof's `adad` route numbers,
  `cuda` the proof's `name` numbers, within run-to-run noise. The references are in `test-output\apply\`
  (`results-dec-warm.txt`, `results-gather.txt`, `results-cpu-warm.txt`, `results-cpu-fill.txt`).

## Decisions (the owner's)

1. **Shape: its own devices.** Idrak.Adad registers device providers `adad-cpu` and `adad-cuda` (`-d adad-cuda`,
   `IDRAK_DEVICES=adad-cpu,adad-cuda`). Their backends run the proof's description-driven lanes on weights kept in the
   file's layout. `cpu` and `cuda` are unchanged from today.
2. **Split: the route code moves; the general fixes stay in core.**
   - Moves to Idrak.Adad: ProductRoutes (becoming the devices' settings), the measurement port (Faster), the
     description-driven CPU and CUDA lanes, the tensor rows over stored rows, the raw int8 lane, Adad's block route, the
     native matrix-unit kind table, the grouped and fused stored products, and the kernel generators only they use.
   - Stays in core (every device uses it): the bfloat16 cache's tensor-core attention, token-major attention output
     (`IMergedHeadsLayout`, `AttentionVariant.Merged`), key/value writes and queries read as views, the CPU pool fix, the
     dispatcher fix (`TYPED_AS_STORED`), `norm_rope_heads`' measured block size, loading without a float32 copy, the bf16
     head streamed to the device (`UploadTransposed16`), the element-type descriptions (`ScaledBlock`, `SuperBlock`,
     `GridBlock`, the GGUF descriptions and encoders), `idrak trace --stacks`.

3. **No `name`, in any scenario, case or stage.** The package never runs, measures or falls back to a kernel chosen by
   a format's name (the proof's `name` route: Idrak's int8, int4 and bfloat16 packed formats and their kernels). So:
   - its lanes setting has no `name` value, and its measured lanes never include a named kernel as a candidate (the
     proof's `RoutedNamed`, which timed a format's own kernel against the generic lanes, is not carried over);
   - on the Adad devices every weight is kept as stored rows (`StoredWeight`): int8 and bfloat16 too, never the named
     packed formats; a packed weight of a named format handed to an Adad device fails loudly naming the format (Idrak's
     own int4 has no description: it fails until it is described);
   - the tied bfloat16 head is the embedding table's own stored rows (`[vocab, dim]` is the head's `[outputs, inputs]`),
     shared with the embedding, read by the Adad lanes; never today's bfloat16 head (under the proof's routes it was, and
     its decode step took the named kernel: CPU 4.6 ms against the generic lane's 34 ms over the named layout).
4. **Only what proved fast and efficient.** The package keeps the lanes the proof's final benches chose (the "measured
   lane" lines of `test-output\apply\dec-*-cuda-*-adad.txt` and `cpu-*-cpu-*-adad.txt`, and the profiles of the last round)
   and drops the experiments that never won:

   | Device | Kept (chosen in the final benches) | Dropped (never chosen) |
   |---|---|---|
   | CUDA prompts | the tensor rows: bfloat16, float16 and int8 rows as stored, 32-value blocks decoded as the tiles load, super-blocks and grids decoded as the tiles load or widened rows, 128- and 64-row tiles, one launch for a group | the native kind (Adad's GemmMma port, its x packs and shape table), the raw int8 lane, the direct lane, the simple matrix-unit kernel, Adad's block route on CUDA (dp4a few rows, s8 tiles) |
   | CUDA decode | the few-row lanes from the description (warp a column), grouped (`_many`), and the fused gated and residual + norm passes where they measure faster | Adad's thread-a-row block gemv |
   | CUDA without bfloat16 products | the f32 tile from the description (the generic product a float32 run needs) | |
   | CPU | the row lane, the tiles lane (panels widened in the product), the widened-once lane, Adad's block route (s8 dots, the k-quants' fused unpack, the grids) | the panel lane |

   A measurement still picks among the kept lanes per shape at run time; the generic route (`IDRAK_ROUTES=generic`, no
   measurement) is not carried over, nor a lanes setting: the devices always measure among their lanes.

## The package

```
src/Idrak.Adad/
  Idrak.Adad.csproj                 net10.0, references Idrak.Abstraction and Idrak.Gpu; packable as Idrak.Adad
  LibraryRegistrations.cs           [assembly: LibraryRegistrations(...)]: DeviceProviders.Register(adad-cpu, adad-cuda)
  AdadChoices.cs                    the measured choices said once per shape (was ProductRoutes.Chose / Choices)
  Faster.cs                         the CPU's measurement (moved as it is)
  Cpu/
    AdadCpuProvider.cs              IDeviceProvider "adad-cpu": Type Cpu, Count 1, Listed false (asked for by name)
    AdadCpuBackend.cs               : CpuBackend; the overrides of the stored products
    AdadCpuBackend.Lanes.cs         was CpuBackend.Lanes.cs
    AdadCpuBackend.Blocks.cs        was CpuBackend.Blocks.cs
    AdadCpuBackend.Typed.cs         was CpuBackend.Typed.cs
  Cuda/
    AdadCudaProvider.cs             IDeviceProvider "adad-cuda": Type Gpu, Count = CUDA's, the same UUIDs, Listed false
    AdadCudaBackend.cs              : CudaBackend; the overrides of the stored products and their capabilities
    AdadCudaBackend.Typed.cs        was CudaBackend.Typed.cs
    AdadCudaBackend.Many.cs         was CudaBackend.Many.cs
    AdadCudaBackend.Super.cs        was CudaBackend.Super.cs (the native and direct lanes left out)
    AdadCudaBackend.TensorRows.cs   was CudaBackend.TensorRows.cs
    AdadKernels.Typed.cs            was PtxKernels.Typed.cs (the f32 tile and the few-row kernels; TypedMma left out)
    AdadKernels.Super.cs            was PtxKernels.Super.cs (widenings, few-row products, decoders)
    AdadKernels.Epilogues.cs        was PtxKernels.Epilogues.cs
    (not carried: CudaBackend.NativeRaw.cs, .NativeDirect.cs, PtxKernels.NativeMma.cs but its PtxBuilder, which stays in
     core with BlockDecode and TableWords as generator helpers; PtxKernels.Blocks.cs, the CUDA block route)
    AdadKernels.TensorRows.cs       the modules of the transposed tensor-core kernels (bf16w, f16w, int8w, the block
                                    and decoded modules), built with core's generator (PtxKernels.TensorCoreGemm)
  README.md                         what the devices are, how to pick them, the setting, what they run
```

Both backends subclass core's, so every operation the proof did not touch runs exactly as on `cpu` and `cuda`
(attention, norms, the tensor-core attention fix, the pool, the views), and only the stored-weight products differ.
`cuda:0` and `adad-cuda:0` take the same GPU's primary context (`cuDevicePrimaryCtxRetain`), each with its own memory
pool and streams; a process normally uses one of them.

## What moves, file by file

| Proof file | Goes to | Notes |
|---|---|---|
| `Idrak.Abstraction/Devices/ProductRoutes.cs` | `Idrak.Adad/AdadOptions.cs`, `AdadChoices.cs` | The public `ProductRoute`/`ProductRoutes` leave core (API change) |
| `Idrak.Abstraction/Devices/Faster.cs` | `Idrak.Adad/Faster.cs` | Unchanged |
| `Idrak.Abstraction/Devices/Cpu/CpuBackend.Lanes.cs`, `.Blocks.cs`, `.Typed.cs` | `Idrak.Adad/Cpu/AdadCpuBackend.*.cs` | Partials of `AdadCpuBackend`; the route checks read `AdadOptions` |
| The routed part of `CpuBackend.Stored.cs` (`StoredMatMulKernel`'s lane choice) | `AdadCpuBackend.cs` (override) | Core's override goes back to the pre-proof body |
| `Idrak.Gpu/Cuda/CudaBackend.Typed.cs`, `.Many.cs`, `.Super.cs`, `.NativeRaw.cs`, `.NativeDirect.cs`, `.TensorRows.cs` | `Idrak.Adad/Cuda/AdadCudaBackend.*.cs` | Its overrides of `StoredMatMulKernel`, `StoredMatMulManyKernel`, `StoredMatMulGatedPairKernel`, `StoredMatMulAddRmsNormKernel`, `StoredEpilogueFused`, `StoredMatMulManyAvailable` |
| The route checks in `CudaBackend.Quantized.cs` (2) | `AdadCudaBackend` | Core's packed products as before the proof |
| `Idrak.Gpu/Cuda/PtxKernels.Typed.cs`, `.Blocks.cs`, `.Super.cs`, `.NativeMma.cs`, `.Epilogues.cs` | `Idrak.Adad/Cuda/AdadKernels.*.cs` | `partial class PtxKernels` → `AdadKernels`; core helpers through internals |
| The module entries of the `gemm_tc_nt_*` kernels in `PtxKernels.TensorCore.cs` | `AdadKernels.TensorRows.cs` | Core stops compiling kernels only Adad loads; the generator's modes (bf16B without LoRA, f16B, int8B, blockB, decodeB) stay in core's generator |
| `tests/Idrak.Tests/ProductRouteTests.cs` | `tests/Idrak.Tests/AdadTests.cs` | Run on `adad-cpu`, `adad-cuda` by name; general tests stay (below) |

Stays in core, as committed: `Backend.cs` (AttentionVariant.Merged, MergesAttention, UploadTransposed16),
`Backend.Views.cs`, `Backend.Routes.cs` (the stored products' operation contracts; their defaults stay false, so plain
devices keep today's path), `CpuBackend.CacheAttention.cs`, `.Layout.cs`, `.Quantized.cs`, `.cs` (pool, views),
`CpuMatMul.cs` (MultiplyPanels, MultiplyPanelsFrom: kernel building blocks, reached by Adad through internals),
`StoredWeight.cs`, `StoredPacking.cs` (the file-layout packed format, chosen per device: step 6), `Quantization.cs`,
the loading files (`ITensorStore.cs`, `IWeightSource.cs`, `Architectures.cs`, `SafeTensors.cs`, `RenamedTensorStore.cs`,
`Tensor.Indices.cs`, `DecoderBuilder.cs`), `Decoding.cs`, `KeyValueLayouts.cs`, `Tensor.Decoding.cs`,
`Tensor.Quantized.cs`, `Decoder.cs`, `Linear.cs`, `Linear.Fused.cs`, `ElementType.cs`, `Gguf.cs`, `GgufElementTypes.cs`,
`CudaBackend.Quantized.cs` (attention, merged strides, row threads), `CudaBackend.Tuning.cs` (its `TuneOp` values keep
their numbers), `CudaBackend.Transpose16.cs`, `PtxKernels.FlashTensorCore.cs`, `PtxKernels.TensorCore.cs` (the generator),
the trace tool, `generate.py`.

## Steps

Each step ends with: one build of the changed projects (`-nodeReuse:false`), the targeted tests named in the step on
`cpu` (and `cuda`, `adad-cpu`, `adad-cuda` where it says so), the CLI reinstalled from the local build (pack to
`test-output\packages`, uninstall, install that version; never NuGet), and a commit pushed to `origin/migration`.

### Step 1: open core to a subclass (no behaviour change)

1. `Idrak.Abstraction.csproj` and `Idrak.Gpu.csproj`: `<InternalsVisibleTo Include="Idrak.Adad" />`.
2. `CpuBackend`: `internal sealed partial class` → `internal partial class`; the constructor `Adad` needs becomes
   `private protected`; the members the moved lanes call become `private protected` (or `internal` where a static helper):
   `D(...)`, `For(...)`, the pool, `CpuTuning`, `CpuMatMul` entry points. The exact list is what the compiler names in
   step 3; nothing is made public.
3. `CudaBackend`: `internal sealed unsafe partial class` → `internal unsafe partial class`; the members the lanes call
   become `private protected` or `internal`: `Launch`, `Launch1D`, `LaunchRows`, `LaunchPack`, `LaunchTensor`,
   `Allocate`, `P`/`U`/`F`, `K`, `TensorKernel`, `TensorCoreKernels`, `LoadDynamicModule`, `Tune`, `TryTuned`,
   `PromptSplitsTuned`, `ForcedSplits`, `PackedSplits`, `SplitCounters`, `_splitPartials`, `_typedLock`,
   `TypedUnavailableReason`, `BFloat16Products`, `MatMul`, `_multiprocessors`, `_profile`/`_profileLabel`/`_profileFlops`.
   Its constructor takes the ordinal as today; `AdadCudaBackend(ordinal)` calls it.
4. `PtxKernels`: `TensorCoreGemm`, `Module`, `TableWords`, `NativeHeader`, `TensorTile`, `TypedGemvRows` and the other
   helpers the moved generators use become `internal` (most already are).
5. Build; run `operations:|public API|inventory|product routes` on `cpu` and `product routes|decoder` on `cuda`: all pass,
   API dump unchanged (only internals changed).

### Step 2: the package and its two devices, empty

1. `src/Idrak.Adad/Idrak.Adad.csproj` (as the other library packages: version from `Directory.Build.props`, no version
   bump), added to the solution and to the test projects' references.
2. `AdadCpuProvider` (`Name` "adad-cpu", `Type` Cpu, `Count` 1, `Create` → one `AdadCpuBackend`, `Listed` false,
   `Summary` "the CPU computing stored weights in their own types (the Adad design)") and `AdadCudaProvider` (`Name`
   "adad-cuda", `Type` Gpu, `Count` and `UnavailableReason` from CUDA's provider, `DeviceUuid` the same as `cuda`'s,
   `Listed` false).
3. `AdadCpuBackend : CpuBackend`, `AdadCudaBackend : CudaBackend`, no overrides yet; their `Name` says "Adad".
4. `LibraryRegistrations`: `[typeof(DeviceProviders)] = () => { Register(adad-cpu); Register(adad-cuda); }`.
5. Tests (new `AdadTests.cs` group "adad devices"): `Device.Parse("adad-cpu")` and `("adad-cuda:0")` resolve, their
   backends are the Adad types, `Device.Available` does not list them (by name only), `idrak devices` lists them with
   their note. Run `adad devices` on `cpu`; `IDRAK_DEVICES=adad-cpu,adad-cuda` runs a basic group (`tensor ops`) on both.

### Step 3: move the CPU lanes

1. `git mv` `CpuBackend.Lanes.cs`, `CpuBackend.Blocks.cs`, `CpuBackend.Typed.cs` to `Idrak.Adad/Cpu/AdadCpuBackend.*.cs`;
   `partial class CpuBackend` → `partial class AdadCpuBackend`, namespace `Idrak.Adad.Cpu`.
2. The routed body of `CpuBackend.StoredMatMulKernel` moves to `AdadCpuBackend.StoredMatMulKernel`; core's goes back to
   its pre-proof body (`git show cf7b8c48:src/Idrak.Abstraction/Devices/Cpu/CpuBackend.Stored.cs`).
3. Every `ProductRoutes.Current` in the moved code reads `AdadOptions.Current` (Generic | Measured | All).
4. `Faster.cs` moves to `Idrak.Adad/Faster.cs`.
5. Build; `product routes` tests (moved in step 9, run here from their old file with the device switched) on `adad-cpu`
   pass; `cpu` runs today's path (the routes' CPU tests skip there).

### Step 4: move the CUDA lanes and their kernels

1. `git mv` the six `CudaBackend.*.cs` lane files to `Idrak.Adad/Cuda/AdadCudaBackend.*.cs` (`partial class
   AdadCudaBackend`); the five `PtxKernels.*.cs` generators to `AdadKernels.*.cs` (`internal static partial class
   AdadKernels`), their calls to core helpers qualified `PtxKernels.…`.
2. The overrides of the stored products leave `CudaBackend` for `AdadCudaBackend`; core's `CudaBackend` keeps the
   defaults (false: the caller's composition, as before the proof).
3. The `gemm_tc_nt_*` module entries leave core's `TensorCoreModules` list for `AdadKernels.TensorRows.cs`, loaded once
   per Adad backend with `LoadDynamicModule` (as the block modules already are); `BuiltInRows` reads them from there.
4. The two route checks in `CudaBackend.Quantized.cs` go; `AdadCudaBackend` overrides where they decided.
5. Build; on `adad-cuda` the CUDA route tests pass (matrix units, tensor rows, every block type, groups, epilogues);
   on `cuda` the packed products and decoder tests pass as on `main`.

### Step 5: the lanes setting and the choices

1. `AdadOptions`: `Lanes` (Generic: the generic lanes only; Measured: the fastest lane per shape; All: Measured plus the
   block route, the proof's "adad"; default All), read once from `IDRAK_ADAD_LANES` (any case; blank is All; anything else
   throws naming the variable and its values), a thread-scoped `AdadOptions.Use(lanes)` for tests and benches, and the
   option on `ComputeResources`-style run options where the library already carries device settings.
2. `AdadChoices`: the proof's `ProductRoutes.Chose`/`Choices`/`ForgetChoices`, per backend.
3. The proof's `IDRAK_ROUTES` goes from the environment table; `IDRAK_ADAD_LANES` takes its row (English and Arabic help).
4. Test: the setting parses, refuses a wrong value naming `IDRAK_ADAD_LANES`, and `Generic` keeps a product on the generic
   lanes (as the proof's `ProductRoutesSetting`).

### Step 6: the loader asks the device, not the route

1. A capability in core, `BackendCapabilities.StoredWeights`: the element types a device multiplies as stored
   (empty on every core device; Adad's: bfloat16, float16, int8 and every described type).
2. `Quantization.cs` (2 checks), `DecoderBuilder.cs` (1), `Tensor.Quantized.cs` (1): `ProductRoutes.Current != Name`
   becomes "the target device lists the type in `StoredWeights`".
3. `StoredPackedFormat` stays in core as the file-layout packed format the loader picks for such a device.
4. Tests: the proof's checkpoint test ("a bfloat16 checkpoint ... packs each weight from the file's rows") runs on
   `adad-cpu` (bits as the values path); on `cpu` the same checkpoint loads as today.

### Step 7: remove the routes from core

1. Delete `ProductRoutes.cs`; nothing in core names a route.
2. `BenchCommand`, `ProfileCommand`, `KernelBench`: print the device's choices (`AdadChoices` through a small core contract,
   `IChoiceReport` on the backend: the choices said so far, and forget them) instead of the route; the kernel bench's
   `prompt` set stays.
3. `generate.py`: unchanged sources (`Backend.Routes.cs` stays); regenerate and check the operation count (168).

### Step 8: CLI and settings carried through

1. `-d adad-cpu` / `-d adad-cuda` on every command that takes a device (nothing to add: device names come from the
   providers); `idrak devices` and `idrak help env` show them and `IDRAK_ADAD_LANES`.
2. The bench's header names the device's lanes setting where the device is Adad's.

### Step 9: tests

1. `ProductRouteTests.cs` → `AdadTests.cs` (group "adad"), each test on the Adad devices (CPU tests on `adad-cpu`, CUDA
   tests on `adad-cuda`), the hooks (`StoredNative`, `StoredTensorRows`, `RowsTileOverride`, …) on the Adad backends.
2. Stay in core's test files: merged attention (all devices), the CPU pool test, the epilogue dispatcher check (moved with
   the epilogues but asserting the core dispatcher converts nothing), the trace tool, the element-type decode tests.
3. `OperationTests.DevicePlumbing`: unchanged names; `data/shared-contracts.txt`: `IChoiceReport` if added.
4. Run: `adad|operations:|public API|inventory|decoder|key/value|tensor views|backend dispose` with
   `IDRAK_DEVICES=cpu,adad-cpu`, and `adad|decoder|key/value|attention|tuning` with `IDRAK_DEVICES=cuda,adad-cuda`.

### Step 10: API dump, inventory, docs

1. `IDRAK_UPDATE_API=1` for the public API (core loses `ProductRoute`/`ProductRoutes`, gains `StoredWeights`; new
   `api/Idrak.Adad.txt`); `IDRAK_UPDATE_INVENTORY=1` for `plans/10-abstraction-inventory.md`.
2. `src/Idrak.Adad/README.md`; this plan's state in `plans/README.md`.

### Step 11: acceptance bench

1. CUDA, Qwen3-0.6B, `--prompt-tokens 512 --tokens 128 -n 3`, one fresh tuning folder per device shared by nothing else,
   a fill run then a warm run, for bf16, f16, int8, q8_0, q4_0, iq4_nl, q4_k, q6_k, iq2_xxs, iq3_xxs:
   `-d cuda` against the proof's `name` lines, `-d adad-cuda` against the proof's `adad` lines (`results-dec-warm.txt`;
   q6_k and iq3_xxs against `results-gather.txt`), within run-to-run noise (about 3–5% on a fresh folder). Peak memory
   equal.
2. CPU, `--prompt-tokens 128 --tokens 32 -n 2`, bf16, int8, q8_0, q4_0, q4_k: `-d cpu` and `-d adad-cpu` against
   `results-cpu-fill.txt` / `results-cpu-warm.txt` (CPU noise about 10%).
3. A traced run of each Adad device (`idrak trace --stacks -- bench ... -d adad-cuda`): no host fallback, no conversion
   counted, allocations as the proof's.
4. Results in `test-output\adad\`; any line outside the noise is traced and fixed before the step closes.

## Done when

- `adad-cpu` and `adad-cuda` exist only through the Idrak.Adad package, run the proof's lanes, and bench as the proof's
  `adad` route; `cpu` and `cuda` bench as the proof's `name` route.
- Core names no route; every test group above passes on its devices; the API dump and inventory are current.
- Everything is on `origin/migration`, a commit per step.

## Left for later (not this version)

- Vulkan (`adad-vulkan`) and HIP.
- Types not described yet (q4_1, q5_0, q5_1, IQ4_XS, IQ1_S, IQ1_M, IQ2_XS, IQ2_S, IQ3_S): they load as today on every
  device until described.
- Making the Adad devices the default.
