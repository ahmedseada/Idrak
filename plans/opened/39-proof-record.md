# The proof: what it followed, how it was done, what it proved

> **Status: corrected after a blind review (2026-11-15); the owner marks it final.** The review found: the CUDA tuning
> keys of the stored-weight lanes were shared across types (a later type in a shared folder reused an earlier type's
> choices); the CPU's lane choices are measured anew in every process; round 13's CUDA table came from a build before
> `cf77ceeb`; the CUDA bf16 head chose the named kernel at decode too. Each is recorded below where it applies. The
> acceptance table is now the re-bench of `cf77ceeb`, each type measured on its own ("What it proved"); the earlier
> tables are kept only as superseded history.

A record of the routes experiment on branch `migration`, for the session that builds Idrak.Adad (plan 39). It states
only what the proof built, measured and proved, with where each fact comes from. What the proof did not prove is listed
at the end as not proved; nothing there may be taken as proved.

## Where it is

- Branch `migration`, on top of `cf7b8c48`. Commits: `5a056e27` (the proof: 79 files), `cf77ceeb` (tensor rows for every
  described type, 16-byte loads, 64-row tiles, float16 rows, super-blocks and grids decoded as the tiles load, cold timing
  for the gathering decoders).
- The switch: `IDRAK_ROUTES` = `name` (default) | `generic` | `measured` | `adad` (`src/Idrak.Abstraction/Devices/ProductRoutes.cs`).
- Results: `test-output\apply\` (this session's rounds: `results-before*.txt`, `results-after*.txt`, `results-all-warm.txt`,
  `results-dec-warm.txt`, `results-cold.txt`, `results-gather.txt`, `results-cpu-fill.txt`, `results-cpu-warm.txt`, and the
  profiles `*.txt` beside them); earlier rounds `test-output\routes*\`, `test-output\final\results.txt`.
- The running notes of every round: the memory note `routes-experiment` (the session's own record, written as each round
  ended).
- Plan 39's earlier versions (commits `2be7627d`, `7e7248fc`, `1e504c50`) are superseded and must not be followed; the
  current one is the one beside this record.
- The re-bench of the committed proof (`cf77ceeb`), each type measured on its own: `test-output\rebench\`, pushed as
  `plans/opened/39-rebench/` (the script `rebench.sh`, every run's log with its "measured lane:" lines, each type's CUDA
  tuning file `tuning-cuda-{type}/` and CPU tuning file `tuning-cpu-{type}.tsv`, `results.txt`).

## Words used here

- **`name`** (`IDRAK_ROUTES=name`, the default at `cf77ceeb`): Idrak as it was before the proof, the baseline every
  bench compared against. A weight's product runs the kernel written for its format, picked by the format's name:
  Idrak's int8 format runs `Int8MatMulKernel`, bfloat16 `BFloat16MatMulKernel`, int4 `Int4MatMulKernel`. A type with no
  kernel of its own (float16, q8_0, q4_0, iq4_nl, the k-quants, the grids) is held as float32 and multiplied by the float32
  product: weights memory 2,590 MiB on Qwen3-0.6B in the re-bench, against 464–1,454 MiB under `adad`. Every bench ran it
  on the same machine and tuning folder right after `adad`, so a number that moves under `name` too (its code unchanged)
  is the machine's run-to-run spread, not a change.
- **Named kernel**: a kernel written for one format and chosen by that format's name: the three above and the fused
  passes over Idrak's packed formats (`PackedMatMulEpilogueKernel`, `PackedMatMulNormAddRmsNormKernel`, the
  `IFusedPackedKernels` members).
- **`adad`** (`IDRAK_ROUTES=adad`): the proof's design. Each weight is kept as stored (the file's rows) and read through its
  type's description (`FloatBits`, integer bits and sign, `ScaledBlock`, `SuperBlock`, `GridBlock`); each product's lane
  is chosen by measuring the candidates on the device at that shape.
- **Route**: the switch `IDRAK_ROUTES` the proof added (`src/Idrak.Abstraction/Devices/ProductRoutes.cs`) so one build
  could run `name` and `adad` side by side; it also has `generic` and `measured` ("The routes, exactly").
- **Lane**: one way a device multiplies x by a stored weight (a kernel or a short pipeline of kernels). CUDA's lanes are
  numbered as `StoredMeasured`'s candidates (below); the CPU's are named (row, panel, today's path, tiles, block route).
- **Few-row lane**: the product for a decoding step (at most 8 rows of x), one warp a column (CUDA) or a row of the weight
  a step (CPU).
- **Kind, native kind**: a type the GPU's matrix unit takes as it is (bfloat16, float16: the `NativeKind` table); the
  "native kind" lane is Adad's GemmMma kernel ported (CUDA lane 2).
- **A row an output**: the stored weight's layout, `[outputs, inputs]`, one row of k values for each output (the file's
  order); the tensor rows' "nt" kernels read it.
- **Fill run, warm run**: on a fresh tuning folder the first run measures the choices while it is timed (fill); the second
  run reads them (warm). The CPU keeps no lane choices between runs (below), so on the CPU both are fresh measurements.
- **Cold timing**: each timed run after the L2 cache is flushed (`Tune`'s `cold: true`); warm timing: back-to-back runs.
- **Clearly faster**: the CPU's named-against-generic choice keeps the named kernel unless the generic product is 3%
  faster (`CpuBackend.Typed.cs`); `Tune` keeps its fallback unless a candidate measures faster (its threshold is in
  `CudaBackend.Tuning.cs`).
- **"today's path"** (CPU): the lane that widens the weight once into panels and runs the float32 product; in the code it
  is the expanded lane (`ExpandedLaneId`, label "today's path (widened once, float32 product)"); it is one lane.

## What it followed (the design)

The design followed Adad's findings (recorded from the Qiyas repository's `CLAUDE.md` on branch `q8-generic`). During
the proof some pieces were also ported from Adad's code in the Qiyas clone, and the code's comments name them ("Adad's
..."): `Measure.Faster` (`Faster.cs`), GemmMma and its shape table (the native kind, CUDA lane 2), the kind table
(`NativeKind`), the block route (the CPU's `BlockProducts`, `MatMulBlocks`, `GemvBlocks`, `S8Dot`, the q8_0 encoder; CUDA's
dp4a few rows and s8 tiles), and `PtxBuilder`. The findings:

1. An operation is defined once.
2. A type is added by declaring a description (float bits, integer bits and sign, a scaled block's layout), never by a
   kernel of the type's own or a route keyed by the type's name.
3. A fast path is chosen by the device's capability, then the description, then a measurement at the shape.
4. CPU and CUDA first; Vulkan, HIP later.

In Idrak this became:

- **Descriptions as data** (`src/Idrak.Abstraction/Tensors/ElementType.cs`): `FloatBits` (exponent, mantissa), integer
  bits and sign, `ScaledBlock` (values, bytes, code bits, signedness, bias, scale bits and offset, code offset, ggml's split
  nibble order, a code table), `SuperBlock` (code planes, sub-scale and minimum pieces), `GridBlock` (grid entries, signs,
  sub-scales). The GGUF types described (`src/Idrak/Models/GgufElementTypes.cs`): Q8_0, Q4_0, IQ4_NL (scaled blocks);
  Q2_K, Q3_K, Q4_K, Q5_K, Q6_K (super-blocks); IQ2_XXS, IQ3_XXS (grids). Each description decodes to the same bits as
  ggml's decoder (tests: "product routes: the k-quants' super-block descriptions ... decode random blocks to the same bits
  as ggml's decoders"). Encoders exist for Q4_0, IQ4_NL, Q4_K, Q6_K, IQ2_XXS, IQ3_XXS (and q8_0, int8, bfloat16, float16):
  Idrak's own simple encoders ("a simple encoder, not ggml's searching quantize_row_*_ref"); Q2_K, Q3_K, Q5_K have none.
  Every benched weight was quantized by these encoders from the bfloat16 checkpoint; no GGUF file made by ggml was benched.
- **Weights kept as the file stores them** under the generic, measured and adad routes: `StoredWeight`, a row of an
  output's values (int8: codes and a scale an output; bfloat16, float16, every described type as stored), packed from the
  file's rows without a float32 copy of the matrix (`StoredPacking.cs`, `IWeightSource.ReadOutputRows`, `WeightRows`).
- **The product from the description, the lane by measurement**: each device generates its kernels from the description;
  the lane for each product shape is chosen by measuring the candidates on the real operands (CUDA: `Tune`; the CPU: a
  port of Adad's `Measure.Faster`, `src/Idrak.Abstraction/Devices/Faster.cs`), once per shape, kept in the tuning cache.

## The routes, exactly

- `name`: today's kernels, chosen by the format's name (Idrak's int8, int4 and bfloat16 packed formats); a type with
  none of these (float16, the GGUF types) is held as float32 and multiplied by the float32 product.
- `generic`: one product per device from the description; no named kernel; no measurement.
- `measured`: per shape, the faster of the lanes. For weights still in Idrak's int8 or bfloat16 packed formats, the
  format's own (named) kernel is one of the candidates: on CUDA `RoutedNamed` (`CudaBackend.Typed.cs`), on the CPU
  `NamedLane` and `MeasureLane` (`CpuBackend.Typed.cs`: the named kernel kept unless the generic product is clearly
  faster, by 3%).
- `adad`: `measured` plus Adad's block route among the lanes.

## What was built

### CPU (`src/Idrak.Abstraction/Devices/Cpu/CpuBackend.Lanes.cs`, `.Blocks.cs`, `.Typed.cs`; `src/Idrak.Abstraction/Devices/Faster.cs`)

- Row lane (a row of stored values a step), panel lane (transposed B), "today's path" lane (the expanded lane: the weight
  widened once from its description into the tiled product's panels, then the CPU's float32 product:
  `CpuMatMul.MultiplyPanels`, bit for bit the same sums as over B [k, n], tested; measured only above 8 rows; its buffer
  rented), tiles lane (panels widened inside the product: `MultiplyPanelsFrom`, tested bit for bit).
- The CPU's packed int8 and bfloat16 products (`Int8MatMulKernel`, `BFloat16MatMulKernel` in `CpuBackend.Typed.cs`): under
  the routes, the named kernel against the generic product (`NamedLane`, `MeasureLane`); the proof turned the pre-proof
  overrides in `CpuBackend.Quantized.cs` into private `Int8Direct`/`BFloat16Direct`, which these call.
- Adad's block route (`adad` only): x quantized into the weights' 8-bit blocks, integer dots with AVX-VNNI-INT8; for the
  k-quants a fused SIMD unpack and dot (`Codes32`); for the grids their block route (`GemvGrid`, `UnpackGridRow`).
- The lane per shape measured with `Faster.Time`, kept only in the process (`StoredLanes`, `s_lanes`: static dictionaries):
  every run measures again. The CPU's own cut-overs for its other operations are kept in `IDRAK_CPU_TUNING_FILE` or
  `<cache>/cpu/tuning.tsv` (`CpuTuning.cs`), not in `IDRAK_TUNING_CACHE`.

### CUDA (`src/Idrak.Gpu/Cuda/CudaBackend.Typed.cs`, `.Many.cs`, `.Super.cs`, `.NativeRaw.cs`, `.NativeDirect.cs`, `.TensorRows.cs`; generators `PtxKernels.Typed.cs`, `.Blocks.cs`, `.Super.cs`, `.NativeMma.cs`, `.Epilogues.cs`, and modes added to `PtxKernels.TensorCore.cs`)

The candidates of `StoredMeasured`, by number:

- 0: the f32 tile (more rows) or the few-row lane (warp a column, 16-byte loads), generated from the description.
- 1: the simple matrix-unit kernel (`TypedMma`).
- 2: the native kind: Adad's GemmMma ported, its shape measured on the device (9 shapes), x and W packed into the kind's
  rows; the kinds as a table (`NativeKind`: bfloat16, float16).
- 3: Adad's block route (`adad` only): dp4a few rows, s8 `mma` tiles.
- 4: the direct native lane (x staged from float32, W decoded per chunk in the kernel).
- 5: the raw int8 lane (int8 codes read as stored, widened in registers).
- 6: the tensor rows (`CudaBackend.TensorRows.cs`): Idrak's tensor-core GEMM generator (`TensorCoreGemm`) reading the
  stored rows. The generator already had a `bf16B` mode before the proof (LoRA's `dx = g · Wᵀ`, kernel
  `gemm_tc_nt_bf16w_lr_f32`, at `cf7b8c48`); the proof added, as generator flags → kernel names: `bf16B` without the
  low-rank stage → `gemm_tc_nt_bf16w_f32`; `f16B` → `gemm_tc_nt_f16w_f32`; `int8B` → `gemm_tc_nt_int8w_f32` (codes widened,
  a scale an output in the epilogue); `blockB` → `gemm_tc_nt_{code}_f32` (32-value scaled blocks decoded as each tile
  loads, a module per description); `decodeB` → `gemm_tc_nt_{code}_f32` (super-blocks and grids decoded by the
  description's decoder); each with `_multi` (a group) and `_m64` (64-row tiles); 16-byte loads for `bf16B`, `f16B` and
  `int8B`. The added module entries are at `PtxKernels.TensorCore.cs` lines 122-133 (`cf77ceeb`); the pre-proof
  `gemm_tc_nt_f32`, `gemm_tc_nt_gelugrad_f32` and the `_lr` kernels are core's own. Super-block and grid rows may instead be
  widened to bfloat16 first (the widened source). The tile and the k splits are measured per shape; k is split in the
  kernel with an ordered sum.
- What the tensor rows depend on in the native kind's code: lane 6 picks its 16-bit modes with `NativeKind.BF16/F16.Holds`,
  widens other typed descriptions with the native kind's pack kernel (`NativePack(d, BF16).W`, `LaunchPack`,
  `NativePadded`, `NativePackGrid`); `SharedInput` (the only way into `TensorRowsGroup`) returns false unless
  `NativeKernel(kind)` and `NativePack(...)` load; `StoredSuper` measures its lanes 3 and 4 only where the native kernel and
  pack load. So the native kind's table, pack kernels and loading gate the tensor rows whichever lane is chosen (every
  adad log shows the native kind measured: "9 of 9 shapes accepted").

Super-block and grid weights take their own lanes (`StoredSuper`): 0 the few-row kernel or the f32 product, 1 native,
2 direct, 3 tensor rows decoding, 4 tensor rows over widened rows.

The keys the CUDA lane choices are kept under: `(adad ? 2000 : 1000) + TypedIds` (`StoredMeasured`), `3000 + SuperIds`
(`StoredSuper`), `4000/5000 + EpilogueIds` (the fused epilogues), `9700 + GroupIds` (the group launch), each id given in
first-use order inside one process: in a bench of one type every type gets id 0, so in a tuning folder shared by several
types a later type reads an earlier type's stored choice wherever the shape and the candidate count match, without
measuring its own (the final benches' folders hold only ids 0).

Groups: products sharing x (q/k/v, gate/up) through `StoredMatMulManyKernel`: on few rows one launch of the `_many`
few-row kernels; on prompts one launch of the tensor rows' `_multi` kernel measured against each product's own lane
(`TensorRowsGroup`). Fused epilogues (`PtxKernels.Epilogues.cs`): the gate/up pair (`_many_gated`) and residual + norm
(`_addnorm`) on the few-row kernels, each measured per shape against the composition.

### Fixed in core along the way (every route uses these)

- The bfloat16 cache's tensor-core attention (`flash_tc_fwd_bf16_d64/128`): attention per 512-token pass 17.3 → 1.1 ms.
- Attention written token-major for the output projection (`IMergedHeadsLayout`, `AttentionVariant.Merged`,
  `Backend.MergesAttention`): the head-merging copy gone.
- Key/value writes and attention queries read as views (`Backend.Views.cs`, `KeyValueWriteStrided`, `AttentionTiledStrided`).
- `norm_rope_heads_f32`: its block size measured (`TuneOp.RowThreads`): 1.55 → 0.63 ms per pass.
- The CPU pool: a miss frees only the bytes it needs, least recently used sizes first: a bf16 prompt pass's allocations
  1,037 → 441 MiB.
- The dispatcher converted `StoredMatMulAddRmsNorm`'s stored weight to float32 on every call (6 ms a call):
  `tools/operations/generate.py` `TYPED_AS_STORED`.
- Loading without a float32 copy (safetensors bfloat16 read directly, rows packed as read, the tied bfloat16 head streamed
  to the device: `UploadTransposed16`); allocations per bench 6–11 GB → 0.5–2.4 GB.
- `idrak trace --stacks` (the runtime's sampling read by Idrak's own `.nettrace` reader) and `idrak profile --pass prompt`.

## The process every round followed

1. **Trace first (from round 6 on).** Rounds 1–5 built lanes from the design and judged them by the bench. From round 6
   on, each round started from a trace: `idrak profile` (one decoding step, or `--pass prompt`: per layer, per
   operation, per kernel), `idrak trace --stacks` (hot spots, waits, collections, allocations), once `dotnet-trace`; the
   cause was found in the trace before the change.
2. **A fix is a measured candidate.** A new lane is added beside the others; the measurement keeps it only where it is
   faster. A lane is not removed because another exists.
3. **Results proved.** Bit for bit where the claim is bits (panels, merged attention, the bfloat16 cache's tensor-core pass
   against the float32 cache's, a checkpoint packed from rows against the values path); otherwise within float32's bound
   for k terms (`k · 2^-23 · Σ|terms|`) against a double reference with x and the weights rounded as the lane rounds them.
   Targeted test groups on `cpu` and `cuda`.
4. **One build, the CLI reinstalled** from the local build (`dotnet pack` to `test-output\packages`, uninstall, install
   that version; never NuGet), once the owner asked for it; before that the benches ran `dotnet run --no-build`.
5. **Bench.** `idrak bench Qwen/Qwen3-0.6B -w {type} -d {cuda|cpu}`; CUDA `--prompt-tokens 512 --tokens 128 -n 3`, CPU
   `--prompt-tokens 128 --tokens 32 -n 2`; both routes (`IDRAK_ROUTES=adad` and `name`) share one tuning folder per
   device (`IDRAK_TUNING_CACHE`, which holds CUDA's choices; the CPU's lane choices are not kept, see above). From round
   11: on a fresh folder a fill run, then a warm run, and the warm run is the result (earlier rounds read one run on a
   shared folder; round 5 used one folder per route, later found wrong). The folders of rounds 11-13 were shared by all
   types (see the keys above).
6. **A number out of line is traced before it counts.** Examples: the fused residual-norm conversion (int8 adad
   generation had fallen from 643 in an earlier round to 343; after the fix 626); q6_k and iq3_xxs slower after decoding
   in the kernel (attributed to warm timing hiding the decoders' scattered reads; see the caveat under "Measurement facts").
7. **Recorded and reported**: the round's numbers in `test-output`, the round in the memory note, the report to the owner.

## The tools used, and exactly how

All runs on Windows 11 from Git Bash or PowerShell at `D:\Projects\idrak`; `idrak` is the CLI installed from the local
build (below). The route is chosen with `IDRAK_ROUTES`, the tuning folder with `IDRAK_TUNING_CACHE`.

### idrak bench (the model bench: the results)

```
IDRAK_ROUTES=adad IDRAK_TUNING_CACHE=$PWD/test-output/apply/tuning-{label}-cuda \
  idrak bench Qwen/Qwen3-0.6B -w {bf16|f16|int8|q8_0|q4_0|iq4_nl|q4_k|q6_k|iq2_xxs|iq3_xxs} -d cuda --prompt-tokens 512 --tokens 128 -n 3
IDRAK_ROUTES=name ... (the same, the same folder)
IDRAK_ROUTES=adad IDRAK_TUNING_CACHE=$PWD/test-output/apply/tuning-{label}-cpu \
  idrak bench Qwen/Qwen3-0.6B -w {bf16|int8|q8_0|q4_0|q4_k} -d cpu --prompt-tokens 128 --tokens 32 -n 2
```

- Read: `prompt ... tokens/s`, `generation ... tokens/s`, `memory peak ... MiB`, and the `measured lane:` lines (each
  product shape's chosen lane, once a run; the CPU's with every candidate's time).
- Scripted (scratchpad `apply_bench.sh`, `apply_bench2.sh`, earlier `final_bench.sh`): for each type, adad then name, one
  log a run (`test-output\apply\{label}-{device}-{type}-{route}.txt`), one summary line a run into
  `results-{label}.txt`; run twice on a fresh folder (fill, warm).
- `idrak bench --kernels matmul,gemv,prompt,attention` (the kernel bench, `KernelBench.cs`; `prompt` was added in the
  proof): products of packed weights by shape, with the chosen lane; used in earlier rounds to check lanes at the kernel
  level.

### idrak profile (where the time goes, by layer, operation and kernel)

```
IDRAK_ROUTES=adad IDRAK_TUNING_CACHE=... idrak profile Qwen/Qwen3-0.6B -w {type} -d cuda --pass prompt --prompt-tokens 512 --top 30
IDRAK_ROUTES=adad IDRAK_TUNING_CACHE=... idrak profile Qwen/Qwen3-0.6B -w {type} -d cuda --top 25        (one decoding step)
IDRAK_ROUTES=adad ... idrak profile Qwen/Qwen3-0.6B -w {type} -d cpu --pass prompt --prompt-tokens 128 --top 20
```

- Read: the pass or step time (each layer drained), MiB allocated and collections, the layers table, the operations
  table (by name and shape), the kernels table (CUDA: calls, ms, share).
- Every layer is drained, so times are higher than a bench's: compare shares, not absolute times.
- `Tune` does not measure while the profiler records (`CanMeasure` needs `_profile` null). In this session profiles were
  run on a folder a bench had filled, or run twice on a folder and the second read; which parts of the profile command
  measure outside the recording was not checked.
- Run from the build under test with `dotnet run -c Release --no-build --project src/Idrak.Cli -- profile ...` when the
  installed CLI was older.
- `--pass prompt` and the per-layer allocation and collection columns were added in the proof's round 7 step 1.

### idrak trace --stacks (hot spots, waits, collections, allocations)

```
IDRAK_ROUTES=adad IDRAK_TUNING_CACHE=... idrak trace --stacks -o test-output/apply/cpu-bf16-alloc.nettrace --top 12 \
  -- bench Qwen/Qwen3-0.6B -w bf16 -d cpu --prompt-tokens 128 --tokens 8 -n 1
idrak trace --read test-output/apply/cpu-bf16-alloc.nettrace --focus "Backend.Allocate" --top 15
```

- Built in the proof (round 8): the runtime's own sampling (`DOTNET_EnableEventPipe`, the sample profiler and the
  runtime's allocation events) of a child process, read by Idrak's `.nettrace` reader; reports the time running .NET
  code, outside it, spinning, stopped by the runtime and waiting, methods by own and total time, allocations by type and
  the code that made them; writes `.speedscope.json` beside the trace.
- Options used: `-o FILE.nettrace`, `--read FILE`, `--focus TEXT` (a method's callers, callees and time inside), `--top N`;
  available: `--from/--to`, `--timeline`, `--slices`, `--by-thread`, `--compare FILE`, `--waits`, `--all-frames`.
- What it found: the load's float32 copies (allocations per bench 6–11 GB → 0.5–2.4 GB), the CPU pool emptying itself
  (1,037 MiB of activations a pass).

### idrak trace --levels operations (earlier rounds)

```
IDRAK_ROUTES=measured idrak trace --levels operations -- bench ...
```

- The library's telemetry of every operation; used in round 6: under it the measured route was faster (577 against 451)
  in every operation, norms included, which pointed the search outside the operations.

### dotnet-trace (round 6, once)

- `dotnet-trace` 10.0.750501 (installed as a global tool), `--profile dotnet-sampled-thread-time`, launching the built
  `Idrak.Cli.exe` by its full Windows path with the bench's arguments; the trace converted to speedscope; a scratchpad
  script (`speedscope_prefill.py`) split the CPU time to the timed prefill windows.
- What it found: the CPU "today's path" lane building a full float32 copy and transposes (about 170 ms a pass on the main
  thread) → `MultiplyPanels`.
- `trace --stacks` was built afterwards so the same question is answered without it; both were used as references for
  each other while it was built.

### The tests (the targeted groups)

```
dotnet build -c Release tests/Idrak.Tests -nodeReuse:false
IDRAK_DEVICES=cpu  IDRAK_FILTER="product routes|decoder|key/value|window|attention|operations:|backend dispose" \
  dotnet run -c Release --no-build --project tests/Idrak.Tests -- --failures-only --failures-file test-output/fails-{label}.txt
IDRAK_DEVICES=cuda IDRAK_FILTER="product routes|decoder|key/value|window|attention|tuning|backend dispose|quant types|gguf" \
  dotnet run -c Release --no-build --project tests/Idrak.Tests -- --failures-only --failures-file test-output/fails-{label}-cuda.txt
IDRAK_UPDATE_API=1 IDRAK_DEVICES=cpu IDRAK_FILTER="public API" dotnet run -c Release --no-build --project tests/Idrak.Tests
IDRAK_UPDATE_INVENTORY=1 IDRAK_DEVICES=cpu IDRAK_FILTER="inventory" dotnet run -c Release --no-build --project tests/Idrak.Tests
```

- `python tools/operations/generate.py` after an operation's signature or its stored operands changed (168 operations).
- The route tests: `tests/Idrak.Tests/ProductRouteTests.cs`, group "product routes" (the CUDA lanes through internal hooks:
  `StoredNative`, `StoredNativeRaw`, `StoredTensorRows`, `StoredTensorRowsMany`, `CudaBackend.RowsTileOverride`, ...).

### The CLI reinstall (after every build that a bench or profile used)

```
dotnet build -c Release src/Idrak.Cli -nodeReuse:false
dotnet pack -c Release --no-build src/Idrak.Cli -o test-output/packages
dotnet tool uninstall -g Idrak.Cli
dotnet tool install -g Idrak.Cli --version {the packed version} --add-source D:/Projects/idrak/test-output/packages --ignore-failed-sources
```

## The rounds, in order

1. Routes name/generic/measured; int8, bfloat16, q8_0 stored; CPU row and panel lanes measured; CUDA few-row lane and
   `TypedMma`. Generation ahead everywhere; prompts behind.
2. The native kind (GemmMma port); route `adad` with the block route (CPU AVX-VNNI-INT8, CUDA dp4a and s8 tiles).
3. Fresh tuning folders per route: an earlier CPU "win" came from a stale cache; A/B must share a folder.
4. Q4_0 described; the block route's layout from the description; the CPU "today's path" lane.
5. The k-quants described (SuperBlock); CPU and CUDA super lanes.
6. The CPU bf16 prompt gap traced with dotnet-trace (full float32 copies and transposes) → `MultiplyPanels`.
7. Fused epilogues, IQ4_NL (code table), the CPU's k-quant block route, grouped launches; then the owner's seven steps:
   `profile --pass prompt`, the tied bf16 head sharing the embedding (today's head), the k-quants' fused SIMD, the fused
   epilogues measured, per-tile widening (tiles lane, direct lane), x packed once per group, IQ2_XXS and IQ3_XXS (grids,
   tables in `.global`: `.const` serialized divergent lanes, 94 → 504 tok/s).
8. `idrak trace --stacks`.
9. Loading without float32 copies; per-call copies removed.
10. "Trace and fix": the native kind table (float16 in place), the raw int8 lane, the tied head streamed, views.
11. "Trace and bench and apply all": the bfloat16 cache's tensor-core attention; token-major attention output; the group's
    x packs made lazy and the raw int8 lane given a shared pack; the residual-norm conversion regression fixed.
12. "Same treatment on all left" and "same approach for all": the tensor rows (lane 6: `bf16B`, `int8B`, then `blockB` and
    the widened rows) and their groups; `norm_rope_heads`' block size; the CPU pool.
13. "Fix for all types": 16-byte loads, 64-row tiles, float16 rows, super-blocks and grids decoded as the tiles load, cold
    timing for the gathering decoders.

## The choices made in the proof, and why

Each choice as it was made, the reason given at the time, and what measured it. Where a choice had no measurement behind
it, it says so.

### How the experiment was set up

- **The routes behind a switch, `name` by default** (`IDRAK_ROUTES`). Effect: every bench ran both routes on the same
  build, and today's behaviour is unchanged unless a run asks for a route.
- **Weights kept as the file stores them under the routes** (`StoredWeight`, a row an output). Why: Adad keeps every weight
  as stored; it also keeps a type at its file size on the device. Measured (the bench's "weights memory" line): q8_0
  2,590 → 921 MiB, q4_0 2,590 → 637 MiB (`test-output\routes*\`).
- **int8 kept with today's codes and a scale an output.** The reason was not written down at the time. Effect: the
  routes multiply the same quantized values as `name`; only the layout and the arithmetic differ.
- **Results judged within float32's bound, not bit for bit against `name`.** Why: the lanes sum in another order than
  today's kernels; the bound used is `k · 2^-23 · Σ|terms|` against a double reference, with x (and the weights, where a
  lane rounds them) rounded as the lane rounds them. Bit for bit only where the arithmetic is the same by construction
  (panels against the tiled product, merged attention against merging after it, the bfloat16 cache's tensor-core pass
  against the float32 cache's over the same values, a checkpoint packed from rows against the values path).

### How a lane is chosen

- **Measured at run time per product shape, on the real operands**, once: on CUDA with `Tune`, kept in the tuning folder
  (but keyed without the type: see "The keys" above, so a later type in a shared folder reused an earlier type's choice);
  on the CPU with `Faster.Time` (Adad's `Measure.Faster` ported), kept only in the process. `Tune`'s fallback is lane 0
  (the f32 tile or the few-row lane): another lane is taken only where it measures faster.
- **A new lane is added as a candidate, never replacing one.** Why: a lane that loses on one shape or card may win on
  another; the measurement decides. So the proof kept lanes that lost (the direct lane, which lost every shape) as
  candidates; the native kind, after the tensor rows existed, still won some products' own measurements (the re-bench).
- **Under `measured`, the named kernels of Idrak's packed int8 and bfloat16 formats were candidates** (CUDA `RoutedNamed`,
  CPU `NamedLane`/`MeasureLane`), as the route's definition says ("the faster of the generic product and the format's own
  kernel"). This is one of the places the proof used `name`; see below.
- **Fused epilogues measured against the composition, the composition the fallback.** Measured in round 7: on this card
  the fused passes were slower (a barrier, an atomic and a serial tail against cheap launches), so the composition ran.
  In the final round's folder the stored choices for some decode shapes are fused (`4000 1 3072 1024 0 = 1`,
  `5000 1 1024 2048 * = 1`), measured once by the first type and read by the others (the keys above).
- **Cold timing only for the super-block and grid decoders.** Measured: cold timing everywhere gave bf16, int8, q4_0 about
  6% less; cold for the gathering decoders only gave q6_k and iq3_xxs back their earlier numbers. The reason given was that
  warm repeats serve the decoders' scattered byte reads from the cache. Caveat (found later): each of these was a
  different fresh folder (3–5% apart), and with the shared keys q6_k and iq3_xxs read q4_k's and iq2_xxs's choices in the
  warm-timed run (they ran after them) but measured their own in the cold-timed run (they ran first): the change is not
  separable from the run order.
- **64-row tiles as a candidate beside 128.** Why: the down projection (1,024 outputs) gave 32 tiles of 128 × 128 for 70
  SMs, so the 128-row kernel had to split k; with 64-row tiles the measurement moved it onto the tensor rows.
- **`norm_rope_heads`' block size measured** (32, 64, 128, 256) instead of fixed at 256. Why: a head of 128 values left most
  of 256 threads idle and paid 8 barriers for nothing (traced: 55 µs a call on a prompt).

### How the CUDA products were built

- **The tensor rows built on Idrak's own tensor-core GEMM generator**, not on Adad's GemmMma port. Why: the trace showed
  `name`'s kernel faster than the port (about 50 against 44–46 TFLOPS), with no separate x pack, an ordered k split inside
  the kernel, and one launch for a group, where the port paid an x pack, a partial-sum pass and a slower kernel. The
  generator was given modes that read the stored rows; which mode a weight takes is chosen from its description.
- **16-byte loads of B in the transposed modes, with 64-row tiles, in one round.** The profiles showed the transposed
  kernels about 5% slower than `name`'s; the 4-byte loads from rows far apart were the suspected cause (not measured
  apart). After both changes bf16 and int8 prompts passed `name`.
- **32-value blocks decoded as each tile loads** (one block per row of a k tile), driven by the description's fields; the
  super-blocks and grids decoded by their descriptions' decoders (8 values at a time) or widened first, measured.
- **float16 weights on float16 products** (x rounded to float16, the weights exact), not widened to bfloat16. Why: exact
  weights, the native float16 kind's arithmetic (the earlier lanes' rule). f16 prompts went 42,209 → 47,756 tok/s in the
  round that added them (with the 16-byte loads and 64-row tiles).
- **The raw int8 lane widened codes with full-rate instructions** (`prmt` into float bits, a subtraction) instead of the
  conversion unit. Measured: the first version with `cvt` lost its saving to the slow conversion unit.
- **Grid tables in `.global` memory, not `.const`.** Measured: `.const` serialized the divergent lanes' reads (94 tok/s);
  `.global` with read-only loads gave 504.
- **The bfloat16 cache's tensor-core attention stages the cache's halves as they are.** Why: the float32 cache's pass rounds
  its values to the same halves, so the two give the same bits over the same values (tested).
- **Token-major attention output as an optional capability** (`IMergedHeadsLayout`), not a change to `Attend`. Why: every
  existing cache format keeps working unchanged; a device opts in (`MergesAttention`).

### How the CPU products were built

- **The weight widened into the tiled product's panels**, not into a float32 matrix then transposed. Traced with
  dotnet-trace: the full float32 copy, the transposes and the extra parallel regions cost the main thread about 170 ms a
  pass; `MultiplyPanels` keeps the same tiles, kernels and sums (bit for bit) over panels widened from the stored rows.
- **The tiles lane** (panels widened inside the product) **added beside it** as a candidate; it won most prompt shapes in
  the final benches.
- **The pool frees only what a miss needs** (least recently used sizes first) instead of emptying itself. Why: rule 77
  (memory must not grow when shapes change) only needs in use plus cached not to grow; emptying the pool on every miss
  dropped the sizes still in use, so every layer allocated its activations anew (traced: about 1 GB a pass).

### Where the proof chose to use `name`

- The named candidates of the packed int8 and bfloat16 formats (above: CUDA `RoutedNamed`, CPU `NamedLane`).
- **The tied bfloat16 head kept today's bfloat16 head** (with `Embedding.FromHead`). Why: the stored head held a second copy
  of the table (+297 MiB); today's head shares the embedding's bits. Measured: bf16 weights 1,454 → 1,157 MiB, speed the
  same. Seen later: its decode step chose the named kernel on the CPU (4.6 ms; the generic lane over the named layout took
  34 ms) and on CUDA too ("cuda: bf16 1x1024->151936 mixed bfloat16: named", `dec-warm-cuda-bf16-adad.txt`); its prompt
  product ran `gemm_tc_nn_bf16w`. So every bf16 `adad` number includes named kernels for the head. (float16, whose head is
  a stored copy, peak 1,578 MiB, took the few-row lane and reached 433 tok/s.)

### How the benches were run

- **One model, one machine:** Qwen/Qwen3-0.6B; an RTX 5070 Ti (70 SMs, compute 12.0) and its CPU (24 threads, AVX-VNNI-INT8).
- **CUDA** `--prompt-tokens 512 --tokens 128 -n 3`; **CPU** `--prompt-tokens 128 --tokens 32 -n 2`.
- **Both routes share one tuning folder per device.** Why (as recorded at the time): an earlier CPU "win" came from a
  stale cache holding `name` back; per-route folders also changed the CPU's cut-overs for every operation (attention 2×
  between folders). Caveat: the code does not move the CPU's cut-overs with `IDRAK_TUNING_CACHE` (`CpuTuning.cs`), so for
  the CPU the recorded cause cannot be the folder; it was not re-checked.
- **A fresh folder: a fill run, then a warm run; the warm run is the result.** Why: the fill run measures while it is
  timed; fresh folders differ by about 3–5% on CUDA prompts.
- **A number out of line is traced or re-run before it counts** (the residual-norm regression; q6_k and iq3_xxs; q8_0's
  CPU generation re-run twice).

## What it proved

### The acceptance table: the committed proof re-benched, each type measured on its own (`test-output\rebench\`)

Code: commit `cf77ceeb` (no code change after it), CLI 0.4.1-dev.2611151417 built from it. CUDA: a fresh tuning folder
per type (`tuning-cuda-{type}`), shared by both routes, runs in the order adad fill, name fill, adad warm, name warm; the
warm runs below. CPU: a fresh CPU tuning file per type (`IDRAK_CPU_TUNING_FILE`), two runs per route (adad, name, adad,
name); the CPU's lane choices are measured anew in each run. Script `test-output\rebench\rebench.sh`; every run's log
beside it (`cuda-{type}-{route}-{fill|warm}.txt`, `cpu-{type}-{route}-run{1|2}.txt`); summary `results.txt`.

**CUDA, Qwen3-0.6B, 512-token prompt and 128 generated tokens, tok/s, warm runs** (weights: the bench's "weights memory"
line, MiB; peak: "memory peak", MiB)

| Type | adad prompt | name prompt | adad gen | name gen | adad weights | name weights | adad peak | name peak |
|---|---|---|---|---|---|---|---|---|
| bf16 | 47,854 | 43,825 | 430 | 395 | 1,157 | 1,157 | 1,282 | 1,282 |
| f16 | 47,214 | 35,649 | 426 | 252 | 1,454 | 2,590 | 1,578 | 2,715 |
| int8 | 48,735 | 46,818 | 614 | 437 | 887 | 887 | 1,012 | 1,012 |
| q8_0 | 42,953 | 35,131 | 520 | 253 | 921 | 2,590 | 1,046 | 2,715 |
| q4_0 | 43,727 | 35,915 | 569 | 252 | 637 | 2,590 | 761 | 2,715 |
| iq4_nl | 43,810 | 35,023 | 536 | 253 | 637 | 2,590 | 760 | 2,715 |
| q4_k | 42,612 | 35,908 | 486 | 253 | 637 | 2,590 | 767 | 2,715 |
| q6_k | 40,618 | 35,557 | 499 | 252 | 783 | 2,590 | 914 | 2,715 |
| iq2_xxs | 40,268 | 35,648 | 470 | 252 | 464 | 2,590 | 594 | 2,715 |
| iq3_xxs | 39,004 | 35,664 | 463 | 252 | 535 | 2,590 | 665 | 2,715 |

The fill runs (same files) were within 3.3% of the warm runs on CUDA, two of them (iq2_xxs, iq3_xxs) faster than the
warm run (`results.txt`). The bf16 rows include named
kernels (the tied head: below).

**CPU, 128-token prompt and 32 generated tokens, tok/s, both runs** (weights and peak equal between the routes but for
bf16 and int8's peaks, `results.txt`)

| Type | adad prompt (run 1 / 2) | name prompt (run 1 / 2) | adad gen (run 1 / 2) | name gen (run 1 / 2) |
|---|---|---|---|---|
| bf16 | 475 / 377 | 269 / 272 | 44.4 / 44.4 | 31.7 / 30.9 |
| int8 | 465 / 507 | 348 / 319 | 70.1 / 69.3 | 46.7 / 42.5 |
| q8_0 | 483 / 534 | 139 / 151 | 52.1 / 62.7 | 8.73 / 8.84 |
| q4_0 | 496 / 467 | 148 / 145 | 37.3 / 37.8 | 8.21 / 8.20 |
| q4_k | 484 / 560 | 155 / 158 | 29.5 / 29.0 | 8.26 / 8.29 |

### Which lanes each type chose in the re-bench (CUDA: each type's own tuning file and the warm logs' "measured lane:" lines; CPU: each run's log)

How to read a CUDA tuning entry (`TuneKey(Op, Variant, A, B, C, D, ...)`, `CudaBackend.Tuning.cs`): a product's entry
(variant 2000, or 3000 for super-blocks and grids) is `m n k candidates = lane`, n the outputs and k the inputs (so
`2000 512 1024 2048 5 = 6` is the 2048→1024 product); a group's (9700) is `m k n0 n1 n2 = 0|1`; a fused pass's (4000
gated gate/up, 5000 residual + norm) is `m n k detail = 0|1`. Qwen3-0.6B's products: q 1024→2048, k and v 1024→1024,
o 2048→1024, gate and up 1024→3072, down 3072→1024, the head 1024→151936; q/k/v and gate/up share their input (groups).

CUDA lane numbers (`StoredMeasured`, the stored value is the lane's number): 0 the few-row lane or the f32 tile,
1 `TypedMma`, 2 the native kind, 3 the block route, 4 direct, 5 raw int8, 6 the tensor rows. `StoredSuper` (super-blocks,
grids): 0 few-row or f32, 1 the native kind (rows widened), 2 direct, 3 the tensor rows decoding as the tiles load, 4 the
tensor rows over rows widened for the call. A group (`TensorRowsGroup`): 1 = one tensor-rows launch for the whole group,
over the type's tensor-rows source as it loads (`TensorRowsSource(type, scales, m, k)`: the rows as stored, the blocks
decoded, the super-blocks and grids decoded as the tiles load; never widened), 0 = each product by its own lane. A fused
pass: 1 = the fused pass, 0 = the composition.

Prompt (m = 512), each product's own choice, then the groups:

| Type | q | k, v | o | gate, up | down | group q/k/v | group gate/up |
|---|---|---|---|---|---|---|---|
| bf16 | 2 | 6 | 6 | 6 | 6 | 1 | 1 |
| f16 | 2 | 6 | 6 | 6 | 6 | 1 | 1 |
| int8 | 6 | 6 | 6 | 6 | 6 | 1 | 1 |
| q8_0 | 2 | 6 | 6 | 6 | 6 | 1 | 1 |
| q4_0 | 2 | 6 | 6 | 6 | 6 | 1 | 1 |
| iq4_nl | 2 | 6 | 6 | 2 | 6 | 1 | 1 |
| q4_k | 1 | 1 | 4 | 4 | 4 | 1 | 0 |
| q6_k | 1 | 1 | 4 | 4 | 4 | 1 | 0 |
| iq2_xxs | 1 | 4 | 4 | 4 | 4 | 1 | 0 |
| iq3_xxs | 1 | 4 | 4 | 4 | 4 | 0 | 0 |

So what ran in the prompt pass (a product in a group whose group launch was chosen ran in the group launch, not by its
own choice):

- bf16, f16, int8, q8_0, q4_0, iq4_nl: the tensor rows everywhere (lane 6 for o and down, the group launch for q/k/v and
  for gate/up). Their own choice of the native kind (q; iq4_nl's gate/up too) never ran.
- q4_k, q6_k, iq2_xxs: q/k/v in the group launch (the super-blocks or grids decoded as the tiles load); gate/up, o and
  down the tensor rows over widened rows (`StoredSuper` 4). Their own choice of the native kind for q (and k, v for the
  k-quants) never ran.
- iq3_xxs: q by the native kind (`StoredSuper` 1; log: "cuda: iq3_xxs 512x1024->2048: native bfloat16 kind"), k, v, gate,
  up, o, down the tensor rows over widened rows.

Decode (m = 1), every type: the few-row lane (lane 0; the logs' "few-row lane" for the o and down products and the head)
but for the bf16 head (named, below); the fused passes: gated gate/up (4000 `1 3072 1024`) 1 for bf16, f16, int8, q8_0,
q4_k, q6_k, iq2_xxs and 0 for q4_0, iq4_nl, iq3_xxs; residual + norm after o (5000 `1 1024 2048`) 1 for every type.

The CPU (each run's own measurement):

| Type | Prompt (128 rows), run 1 / run 2 | Decode (1 row), run 1 / run 2 |
|---|---|---|
| bf16 | tiles for the k/v, q, gate/up shapes; o and down "today's path" / tiles | row lane / row lane; the head named in both |
| int8 | tiles, but gate/up, o, down "today's path" / tiles but o, down "today's path" | row lane, but the down product the block route / row lane |
| q8_0 | tiles, but down "today's path" / tiles, but o and down the block route | the block route / the block route |
| q4_0 | tiles, but o and down "today's path" / tiles | row lane / row lane |
| q4_k | tiles, but o and down "today's path" / the same | the block route / the block route |

Summary of the evidence:

- **Ran on CUDA**: the tensor rows (lane 6; `StoredSuper` 4; the group launch over every type's source, including the
  super-blocks and grids decoded as the tiles load), the native kind once (iq3_xxs's q), the few-row lanes and both fused
  decode passes (each where its entry is 1).
- **Chosen by a product's own measurement but never run** (its group launch won): the native kind for q in bf16, f16,
  q8_0, q4_0, iq4_nl, q4_k, q6_k, iq2_xxs, for k and v in q4_k and q6_k, for gate/up in iq4_nl.
- **Never chosen on CUDA**: lanes 1 (`TypedMma`), 3 (the block route), 4 (direct), 5 (raw int8); `StoredSuper` 2
  (direct) and 3 (as a product's own lane; its kernels run in the group launch).
- **Chosen on the CPU**: the tiles lane, "today's path", the row lane, Adad's block route (q8_0 and q4_k on every decode
  shape in both runs, q8_0's o and down prompts in run 2, int8's down decode in run 1), the named kernel for bf16's head.
  **Never chosen**: the panel lane.
- The earlier rounds' lane lists were made with keys shared across types and are superseded by this one.

### Qwen3-1.7B baseline (M0, plan 39 decision 10)

Today's Idrak (`IDRAK_ROUTES=name`, set on every command) on `Qwen/Qwen3-1.7B` (revision `70d244cc86cc`, pulled once
before the first run: `pull.txt`; 1,720.6M parameters, hidden 2,048, 28 layers, a tied head), on the same machine as the
re-bench, the reference the Adad devices are measured against once the plan is finished. Measuring only: no
code was changed. Files: `test-output\adad\m0-qwen3-1.7b\`, copied (without the `.nettrace` and `.speedscope.json`
files) to [39-m0-qwen3-1.7b/](39-m0-qwen3-1.7b/): the script `m0.sh`, `results.txt`, every bench log, every profile
log, every trace report, the CUDA tuning folders and CPU tuning files.

- **Code and CLI**: commit `4802653c`, the CLI built from it and installed as the record's "The CLI reinstall" says:
  `idrak 0.4.1-dev.2611191613` (package `0.4.1-dev.2611191614`; the assemblies' informational version
  `0.4.1-dev.2611191613+4802653c40f71a7ad9280ff77a362f9b686c26be`; the CUDA tuning files' `library:` line
  `0.4.1+4802653c...; kernels 29d753baeebe4de8`). Between `cf77ceeb` and `4802653c` only document code, help text and
  member visibility changed (`git diff --stat cf77ceeb 4802653c`: nothing under `src/Idrak.Gpu`, no model or device code),
  so this is the same `name` code as the re-bench's `name` columns.
- **Commands** (`m0.sh`, one model run a call; `IDRAK_ROUTES=name` exported; paths absolute):
  - CUDA, types in the order bf16, f16, int8, q8_0, q4_0, iq4_nl, q4_k, q6_k, iq2_xxs, iq3_xxs, each in a fresh folder
    `tuning-cuda-{type}`, a fill run then a warm run: `IDRAK_TUNING_CACHE=.../tuning-cuda-{type} idrak bench
    Qwen/Qwen3-1.7B -w {type} -d cuda --prompt-tokens 512 --tokens 128 -n 3` → `cuda-{type}-{fill|warm}.txt`.
  - CPU, bf16, int8, q8_0, q4_0, q4_k, each with a fresh `tuning-cpu-{type}.tsv`, two runs:
    `IDRAK_CPU_TUNING_FILE=.../tuning-cpu-{type}.tsv idrak bench Qwen/Qwen3-1.7B -w {type} -d cpu --prompt-tokens 128
    --tokens 32 -n 2` → `cpu-{type}-run{1|2}.txt`.
  - Profiles, on the type's filled folder or file: CUDA `idrak profile Qwen/Qwen3-1.7B -w {type} -d cuda --pass prompt
    --prompt-tokens 512 --top 30` and `... -d cuda --top 25` (one decoding step); CPU `... -d cpu --pass prompt
    --prompt-tokens 128 --top 20`. Each ran twice, as the record's profiles did ("run twice on a folder and the second
    read"): the first (`profile-{device}-{type}-{prompt|decode}-first.txt`) stored the shapes the bench never ran, inside
    its timed pass (finding M0-3); the second (`profile-{device}-{type}-{prompt|decode}.txt`) is the one read. The
    tuning lines each run added (the script diffed the tuning file around every run and printed the diff) are collected
    in `tuning-added.txt`. **The layer tables and pass totals of these profiles are not trustworthy** (on CUDA float32
    products and the head over every row; on both devices every layer's intermediate results held to the pass's end):
    read their operations and kernels tables only, and for layer times the profiles made after the fix
    (`after-profile-fix/`, "After the profile fix" below).
  - Traces: `idrak trace --stacks -o trace-{device}-{type}.nettrace --top 20 -- bench Qwen/Qwen3-1.7B -w {type} -d cuda
    --prompt-tokens 512 --tokens 128 -n 1` (CUDA bf16, int8, q4_0, q4_k, iq2_xxs) and `... -d cpu --prompt-tokens 128
    --tokens 8 -n 1` (CPU bf16, int8, q4_0) → `trace-{device}-{type}.txt`; further reads of the same traces with
    `idrak trace --read ... --focus` or `--timeline` → `trace-{device}-{type}-{focus-...|timeline}.txt`.
- **Every type loaded and ran on both devices** (every run's exit code 0, `results.txt`). No warm run, second CPU run,
  profile read or trace wrote a tuning file (`tuning-added.txt`: each file's last write is its first decode profile on
  CUDA, its run 1 on the CPU).

**CUDA, Qwen3-1.7B, 512-token prompt and 128 generated tokens, tok/s, warm runs** (weights: "weights memory", MiB; peak:
"memory peak", MiB). The fill runs are within 0.9% of the warm runs on prompts and equal on generation (`results.txt`).

| Type | prompt | generation | weights | peak | prompt kernels (profile) | decode kernels (profile) |
|---|---|---|---|---|---|---|
| bf16 | 19,330 | 183 | 3,302 | 3,465 | `gemm_tc_nn_bf16w` (`_multi` for q/k/v and gate/up) | `bf16_gemv_*` |
| f16 | 17,812 | 106 | 7,177 | 7,348 | `gemm_tc_nn` over float32 rows | `gemv_nn_f32` |
| int8 | 20,809 | 215 | 2,257 | 2,412 | `gemm_tc_nn_int8w` (`_multi`) | `int8_gemv_*` |
| q8_0 | 17,402 | 106 | 7,177 | 7,348 | `gemm_tc_nn` | `gemv_nn_f32` |
| q4_0 | 17,902 | 106 | 7,177 | 7,348 | `gemm_tc_nn` | `gemv_nn_f32` |
| iq4_nl | 17,451 | 106 | 7,177 | 7,348 | `gemm_tc_nn` | `gemv_nn_f32` |
| q4_k | 17,752 | 106 | 7,177 | 7,348 | `gemm_tc_nn` | `gemv_nn_f32` |
| q6_k | 17,316 | 106 | 7,177 | 7,348 | `gemm_tc_nn` | `gemv_nn_f32` |
| iq2_xxs | 17,688 | 106 | 7,177 | 7,348 | `gemm_tc_nn` | `gemv_nn_f32` |
| iq3_xxs | 17,448 | 106 | 7,177 | 7,348 | `gemm_tc_nn` | `gemv_nn_f32` |

Under `name` on CUDA every type without a kernel of its own (f16 and the GGUF types) is held as float32: 7,177 MiB of
weights against bf16's 3,302 and int8's 2,257; their decode is `gemv_nn_f32` over the float32 rows (about 84% of a step's
kernel time, `profile-cuda-{type}-decode.txt`), and their prompt products run as separate launches (343 a pass, where
bf16 and int8 group q/k/v and gate/up in 259).

**CPU, Qwen3-1.7B, 128-token prompt and 32 generated tokens, tok/s, both runs** (weights and peak, MiB, equal in both runs)

| Type | prompt (run 1 / 2) | generation (run 1 / 2) | weights | peak | product (profile) |
|---|---|---|---|---|---|
| bf16 | 118 / 115 | 15.0 / 15.0 | 3,302 | 3,405 | `matmul_bf16` |
| int8 | 123 / 115 | 26.1 / 25.3 | 2,257 | 2,361 | `matmul_int8` |
| q8_0 | 104 / 98.4 | 3.33 / 3.36 | 2,357 | 2,414 | `matmul_stored` |
| q4_0 | 98.4 / 107 | 3.34 / 3.34 | 1,537 | 1,594 | `matmul_stored` |
| q4_k | 105 / 106 | 3.35 / 3.39 | 1,537 | 1,594 | `matmul_stored` |

**Findings** (what the bench, profiles and traces show slow, wrong or wasteful; none was fixed in M0; each names its
files and the fix piece proposed for it):

- **M0-1. The profile's layer table times float32 products on CUDA.** `ProfileCommand`'s per-layer pass calls each
  module's `ForwardCached` itself, outside `Sequential.ForwardCached`'s `MixedPrecision.Prefer(Precision)`, so
  `MixedPrecision.Current` is the process default (float32) there; the operations and kernels tables run inside
  `network.ForwardCached` and are right. Shown by the first prompt profiles storing float32-tile choices the bench never
  stores: `FloatTile 0 512 {2048|6144} ...` for f16 and the GGUF types (the bench stored `TensorSplits` for the same
  shapes), `PackedTile 0 512 ...` for int8 (`tuning-added.txt`, `profile-cuda-{type}-prompt-first.txt`); and by the head:
  17.7–18.4 ms in every type's layer table against 3.85–4.00 ms for its kernel (`profile-cuda-{type}-prompt.txt`).
  Proposed piece: the profile's per-layer pass in the network's precision scope (a tool fix; no library result changes).
  **Fixed in piece 2** ("After the profile fix" below).
- **M0-2. `idrak profile --pass prompt` runs the head over every prompt row**, which generation never does
  (`TextGenerator` sets `DecodingContext.LastPositionOnly`; the profile's contexts do not). CUDA bf16:
  `gemm_tc_nn_bf16w 512x151936x2048` 3.965 ms, 13.2% of the pass's kernel time (`profile-cuda-bf16-prompt.txt`); CPU
  bf16: `matmul_bf16 [1x128x151936]` 126.5 ms (11.2%) and 1,261 MiB of the pass's 5,650 MiB
  (`profile-cpu-bf16-prompt.txt`). Proposed piece: with M0-1, the profile's prompt pass with `LastPositionOnly`, so it
  times what the bench runs. **Fixed in piece 2.**
- **M0-3. A profile's first run on a bench-filled folder measures inside its timed pass, and one decoding step catches
  one-off pauses.** First prompt profiles: head 829–907 ms and `layers.0` 160–207 ms (`profile-cuda-*-prompt-first.txt`;
  second runs 17.7–18.4 and 4.5–4.8 ms; the lines the first runs stored: `tuning-added.txt`). The read bf16 decode
  step took 22.10 ms against 6.88 ms in its first run: one layer held 15.4 ms and 2 collections
  (`profile-cuda-bf16-decode.txt`); in the bench the collections fall in loading
  (1.5–1.8 s), none in generation (`trace-cuda-bf16-timeline.txt`). An iq3_xxs decode step read `gemv_nn_f32` 18.4 ms
  against 10.3–10.5 ms in every other float32-held type (`profile-cuda-iq3_xxs-decode-outlier.txt`); re-run: 10.43 ms
  (`profile-cuda-iq3_xxs-decode.txt`). Proposed piece: with M0-1 and M0-2, the decode profile timed over several steps
  (median) after a collection; until then profiles run twice. **Fixed in piece 2**: a first profile on a bench-filled
  folder now agrees with the second; no extra untimed pass was needed.
- **M0-4. Loading a GGUF type for a device without its product (`cuda` under `name`) expands it on about two threads
  and copies the matrix in float32 several times.** q4_0: 11.3 s of a 13.4 s run in `ModelChoices.Load`, 16,086 MiB
  allocated (`SafeTensorsReader.ReadTransposed` 3,792 + 1,584, `StoredPackedFormat.Create` 3,794 + 2,772,
  `SafeTensorsReader.Read` 1,187, `HostParallel.Transpose` 1,187), `StoredPackedFormat.Create` 9.7 s with 7.4 s of its own
  code, from 2.0 to 11.9 s at about 2 of 24 threads (`trace-cuda-q4_0.txt`, `-focus-create.txt`, `-timeline.txt`); q4_k
  the same (16,088 MiB, `Create` 10.9 s, `trace-cuda-q4_k.txt`); bf16 for comparison: `ModelChoices.Load` 1.3 s, 243 MiB
  (`trace-cuda-bf16.txt`). Cause: `StoredPackedFormat.Create`'s expansion (`StoredPacking.cs` lines 58-71) decodes each
  output's row on one thread and writes it down a column of the float32 matrix (`expanded[r * n + j]`, n apart). Proposed
  piece: the expansion in parallel and written along the matrix's rows, and the float32 transposed copies removed (the
  same decoded values at the same places: bit for bit).
- **M0-5. The grid encoder rebuilds the grid's levels on every row** (`GgufElementTypes.EncodeGrid`:
  `g.Grid.Distinct().Order().ToArray()`, while the nearest-entry table beside it is kept per description). iq2_xxs: 90.3 s
  of CPU in `EncodeGrid` (67% of the run's busy time), 72.9 s of it in `Enumerable.ToArray` over hash sets, about
  31,500 MiB of hash-set entries, int arrays and enumerators allocated in it, 1,780 collections and the pool trimming
  they start (`SharedArrayPool.Trim` 18.1 s, `GetMemoryPressure` 15.5 s); the load takes the first 20 s of a 22.2 s run
  (`trace-cuda-iq2_xxs.txt`, `-focus-encodegrid.txt`, `-timeline.txt`). The other encoders are slow too: `EncodeQ4_K`
  21.1 s of CPU (`PutCode` 9.0 s, `trace-cuda-q4_k.txt`), `EncodeQ4_0` 6.5 s (`trace-cuda-q4_0.txt`). The encoders stay
  in core (decision 2), so every route and device pays this. Proposed piece: the levels kept with the nearest table (the
  same bytes), then the q4_k and q4_0 encoders traced.
- **M0-6. The CPU's named bf16 and int8 products widen the whole weight to float32 on every prompt call, and the tiled
  product copies it again.** `BFloat16Direct` (`CpuBackend.Quantized.cs`, more rows than the few-row limit) and int8's
  `Tensor.Dequantized` path: 5,650 MiB allocated per 128-token pass, 156.8 MiB a layer (`profile-cpu-bf16-prompt.txt`,
  `profile-cpu-int8-prompt.txt`; the stored types' pass: 678 MiB, `profile-cpu-q4_0-prompt.txt`); a short bench run
  allocates 12,217 MiB (bf16: 5,709 in the product, 4,064 of activations through `Tensor.Empty`, 1,992 in
  `BFloat16Direct`, `trace-cpu-bf16.txt`) and 11,159 MiB (int8: 6,318 through `Tensor.Empty`, 4,400 in
  `Tensor.Dequantized`, `trace-cpu-int8.txt`). `CpuMatMul.Tiled.RunTile` then copies B's panels for every row tile:
  copies are 8.75 s of its 23.8 s (`trace-cpu-int8-focus-runtile.txt`; bf16 8.8 s, `trace-cpu-bf16-focus-memmove.txt`).
  Proposed piece (a change to the plain `cpu` device, which decision 1 keeps as `name`: the owner's decision): widen into
  the product's panels per tile, as the proof's tiles lane does, and pack each column panel once for all row tiles.
- **M0-7. On the CPU the products with 2,048 outputs run at about half the rate of those with 6,144.** bf16:
  `matmul_bf16 [1x128x2048]` (q, o, down) 542.9 ms for 150 GFLOP, 277 GFLOP/s, against `[1x128x6144]` (gate, up)
  347.8 ms for 180 GFLOP, 519 GFLOP/s (`profile-cpu-bf16-prompt.txt`; int8 590.2 against 376.4 ms); the stored types' k/v
  (`matmul_stored [1x128x1024]`) 221.9–227.6 ms against bf16's 73.4 ms (`profile-cpu-q4_0-prompt.txt`,
  `profile-cpu-q8_0-prompt.txt`, `profile-cpu-q4_k-prompt.txt`). Not yet traced apart (q and down share the shape's
  label). Proposed piece: `idrak bench --kernels` and a trace at these shapes to find the cause (the tile grid for n =
  2,048 is the first suspect).
- **M0-8. The CPU's decode over stored types runs at 3.3 tok/s.** q8_0, q4_0, q4_k: 3.33–3.39 tok/s against int8's
  25–26 and bf16's 15.0 (`results.txt`). Each step widens every tile's stored rows to float32
  (`GgufFile.DequantizeBuiltInParallel`, a `Parallel.For` inside the tiles' `Parallel.For`), writes them down the columns of a
  [k, columns] panel and multiplies one row through `CpuMatMul.Multiply` (`CpuBackend.Stored.cs`,
  `StoredTileProduct`): `StoredMatMulKernel` 103.9 s of the run's 136.0 s busy time: 44.3 s its own code (the tile's
  body inlined: the panel writes and the result copy), 26.4 s `CpuMatMul.Multiply`, 15.3 s the nested widening, 7.9 s
  returning buffers to the shared array pool (nearly all of it `Thread.GetCurrentProcessorNumber`,
  `-focus-processornumber.txt`) and 4.9 s `Task.SpinWait` (`trace-cpu-q4_0.txt`, `-focus-storedmatmul.txt`,
  `-timeline.txt`). Proposed piece (the plain `cpu` device again: the owner's decision): a few-row path over the stored
  rows read in place, with no panel and no nested parallel loop (the proof's row lane and block route do this for `adad`).
- **M0-9. CUDA's int8 decode kernels read at about half the bandwidth of bf16's.** `int8_gemv_multi_act_f32` 2.453 ms
  for 705 MB (287 GB/s) against `bf16_gemv_multi_act_f32` 2.613 ms for 1,409 MB (539 GB/s); `int8_gemv_addnorm_f32`
  1.545 ms for 470 MB (304 GB/s) against bf16's 1.802 ms for 940 MB (521 GB/s); the head's single kernels both reach
  about 810–830 GB/s (`int8_gemv_r1_f32` 0.384 ms, `bf16_gemv_r1_f32` 0.753 ms) (`profile-cuda-int8-decode.txt`,
  `profile-cuda-bf16-decode.txt`). So int8 generates 215 tok/s against bf16's 183 while reading half the bytes. Proposed
  piece (the plain `cuda` device, decision 1: the owner's decision): trace the int8 few-row kernels; the proof's few-row
  lane is the `adad` answer (int8 614 against 437 on Qwen3-0.6B).

#### After the profile fix (plan 39 phase 1, piece 2: M0-1, M0-2, M0-3)

`idrak profile` now times the network's own cached pass (`Sequential.ForwardCached`, through its hook before each layer;
`Idrak` makes its internals visible to `Idrak.Cli.Commands` for it): the network's precision, each layer's intermediate
results freed as generation frees them, its decoding contexts with `LastPositionOnly` (a prompt's head over the last
row), each timed pass after a full collection, and a decode profile the median of `--steps` steps (default 5, printed).
Measured with CLI `0.4.1-dev.2611191709` (built from `45a32809` with the change), `IDRAK_ROUTES=name`, the M0 profile
commands, each type's folder filled by a bench of this build, a first and a second profile each. Files:
`test-output\adad\p2-profile\`, copied to [39-m0-qwen3-1.7b/after-profile-fix/](39-m0-qwen3-1.7b/after-profile-fix/)
(`p2.sh`, `before-after.txt`, `results.txt`, `tuning-added.txt`, `profile-{device}-{type}-{prompt|decode}-{first|read}.txt`,
the bench logs, the tuning files, `test-*.txt`).

CUDA, from `before-after.txt` (ms; the layer table's sum against the kernel table's sum; the head's layer time against
its own kernel):

| Profile | M0 first / read pass | after first / read pass | M0 read: layers / kernels | after read: layers / kernels | M0 read head: layer / kernel | after read head: layer / kernel |
|---|---|---|---|---|---|---|
| bf16 prompt | 1,181 / 164 | 27.2 / 27.1 | 147.4 / 30.0 | 26.0 / 27.0 | 17.7 / 3.97 | 0.79 / 0.78 |
| int8 prompt | 1,220 / 147 | 25.4 / 25.4 | 145.5 / 28.3 | 24.3 / 24.9 | 18.4 / 3.85 | 0.42 / 0.40 |
| q4_0 prompt | 1,118 / 146 | 29.3 / 29.3 | 145.1 / 32.1 | 28.2 / 29.7 | 18.0 / 4.00 | 1.54 / 1.59 |
| q4_k prompt | 1,122 / 148 | 29.3 / 29.5 | 146.2 / 32.1 | 28.3 / 29.3 | 18.3 / 4.00 | 1.55 / 1.54 |
| iq2_xxs prompt | 1,121 / 153 | 30.5 / 30.2 | 152.1 / 32.1 | 29.1 / 30.3 | 18.2 / 3.99 | 1.54 / 1.54 |
| bf16 decode | 6.88 / 22.10 | 6.44 / 6.34 | 21.5 / 6.85 | 6.28 / 7.29 | 0.76 / 0.75 | 0.76 / 0.76 |
| int8 decode | 6.55 / 6.37 | 5.51 / 5.39 | 5.84 / 6.08 | 5.29 / 6.15 | 0.41 / 0.38 | 0.39 / 0.39 |
| q4_0 decode | 11.46 / 11.55 | 10.36 / 10.38 | 10.9 / 12.4 | 10.3 / 12.2 | 1.52 / – | 1.53 / – |
| q4_k decode | 11.41 / 12.13 | 10.40 / 10.43 | 11.5 / 12.3 | 10.3 / 12.5 | 1.52 / – | 1.53 / – |
| iq2_xxs decode | 11.44 / 11.59 | 10.40 / 10.35 | 10.9 / 12.5 | 10.3 / 12.3 | 1.53 / – | 1.53 / – |

(–: the float32-held types' decode head is one of the 197 `gemv_nn_f32` calls.) The bench's own prompt pass, for
comparison: 512 / 19,339 tok/s = 26.5 ms for bf16 (`bench-check/cuda-bf16-warm.txt`).

- **The layer tables now sum to about the kernel tables** (prompt 24.3–29.1 ms against 24.9–30.3 ms; decode within the
  kernel table's own timing overhead) and the head's layer time is its kernel's (and a prompt's head is one row:
  `bf16_gemv_r1_f32`, `int8_gemv_r1_f32`, `gemv_nn_f32 1`).
- **First and second runs agree**: prompts within 1%, decode within 2.2% (M0: a first prompt pass 7–8× the second; a bf16
  decode step 6.88 then 22.10 ms).
- **No float32-path entry is stored by a profile**: the first prompt profiles added no tuning line; the first decode
  profiles added only the decode profile's own 128-token warm-up and window-160 shapes (`PackedSplits 4|0 128 ...`,
  `TensorSplits 0 128 ...`, `DecodeSplits 2 16 160 ...`, `RowThreads 0 128 4096`), measured in its untimed warm-up; no
  `FloatTile 0` or `PackedTile 0` line (`after-profile-fix/tuning-added.txt`). So no extra untimed pass was added.
- **CPU** (`before-after.txt`): bf16's prompt pass 1,064 / 1,061 ms with 4,362 MiB allocated (M0 1,251 / 1,294 ms,
  5,650 MiB): its head now one row, 8.9–10.6 ms and 0.6 MiB (M0 119–128 ms, 1,261 MiB); q4_0 1,171 / 1,262 ms, 314–326 MiB
  (M0 1,287 / 1,308 ms, 677–679 MiB).
- **Nothing else moved**: the bf16 bench, warm, 19,339 tok/s prompt, 183 generation, weights 3,302 MiB, peak 3,465 MiB
  against M0's 19,330 / 183 / 3,302 / 3,465 (`bench-check/cuda-bf16-warm.txt`); each type's fill with this build within
  2.2% of M0's fill on prompts (`results.txt`).
- Tests: "cli measure: profile ..." (extended: the median of `--steps`, the precision the layers ran in, a bfloat16 run's
  layers in bfloat16 on CUDA, a prompt's head over the last row only, `--steps 0` refused), "cli measure: every command
  has help ...", "cluster: cli profile --cluster ..." on `cpu` (3 passed) and `cuda` (2 passed, the help test checked on
  the CPU only); "public API|inventory" on `cpu` (6 passed) (`test-*.txt`).

New findings from this piece:

- **M0-10. The CUDA tuning file names the commit it was measured by**, so a build of another commit measures every
  choice again: the first bench on a copy of M0's bf16 folder rewrote it (its `library:` line `4802653c` →
  `45a32809`; `after-profile-fix/tuning-added.txt`), and M0's folders cannot be read by any later build. A "warm" run is
  warm only on a folder the same commit's build filled. (The CPU tuning file names no commit: M0's were read as filled.)
  Every later piece fills its folders with its own build.
- **M0-11. Holding every layer's intermediates doubled the CUDA layer times**: the interim build (the profile's own
  per-layer loop in the network's precision, one scope for the pass) gave bf16 layers of 1.6–2.0 ms (but the first,
  0.93 ms) and a 47.1 ms layer sum against 27 ms of kernels (`after-profile-fix/interim-2611191703/`); the network's
  own pass, which frees each layer's intermediates, gives 0.89–0.94 ms. Fixed by timing the network's pass.
- **M0-12. The CPU's one-row product over stored rows is slow**: q4_0's head, one row of 151,936 outputs, takes 49.8–50.6
  ms (`matmul_stored [1x1x151936]`, `after-profile-fix/profile-cpu-q4_0-prompt-*.txt`) against bf16's 8.9–10.6 ms: the
  stored decode path of M0-8, seen on one product.

### The proof on Qwen3-1.7B (M6)

Plan 40's M6: the re-bench's protocol on `Qwen/Qwen3-1.7B` (revision `70d244cc86cc`, M0's pull), `adad` against `name`
in the same runs, to see whether the winning design holds at products about 4× larger. Measuring only: no code was
changed. Files: `test-output\adad\m6-proof-1.7b\`, copied (without the `.nettrace` and `.speedscope.json` files) to
[39-m6-proof-1.7b/](39-m6-proof-1.7b/): the script `m6.sh`, `results.txt`, every bench log, every profile log, every
trace report, the CUDA tuning folders and CPU tuning files, `tuning-added.txt`, the 0.6B check (`check-0.6b/`) and the
derived tables with the scripts that made them (`ratios.sh` → `ratios.txt`, `spread.sh` → `spread.txt`,
`kernel-groups.sh` → `prompt-kernel-groups.txt`, `product-rates.sh` → `product-rates.txt`).

- **Code and CLI**: commit `de1f8a96`, the CLI built from it and installed as "The CLI reinstall" says (at 21:11, before
  the first bench at 21:12): `idrak 0.4.1-dev.2611191810` (package `0.4.1-dev.2611191811`; the installed assemblies'
  informational version `0.4.1-dev.2611191810+de1f8a96e66d9b2d98860553311a470cf69243ed`; every CUDA tuning file's
  `library:` line `0.4.1+de1f8a96...; kernels 29d753baeebe4de8`, the re-bench's kernels hash). Between `cf77ceeb` (the
  re-bench) and `de1f8a96` nothing under `src/Idrak.Gpu`, the devices or the model changed (`git diff --stat cf77ceeb
  de1f8a96 -- src`: documents, imaging, help text, member visibility and the profile command), so `adad` and `name` are
  the re-bench's code. Every run's exit code 0; every bench log ends with its summary lines.
- **The machine**: no idrak, dotnet-trace or build process ran beside the runs; one idle .NET service of another project
  (started before the runs) ran throughout. Runs one at a time, in this order (the files' times): CUDA benches 21:12–21:21,
  CPU benches 21:21–21:27, CUDA `adad` prompt profiles 21:27–21:28, the other profiles 21:33–21:42, traces 21:42–21:43, the
  0.6B check 21:44–21:45.
- **Commands** (`m6.sh`, one model run a call; the route set on every run; `out` = `D:/Projects/idrak/test-output/adad/m6-proof-1.7b`):
  - CUDA, types in the order bf16, f16, int8, q8_0, q4_0, iq4_nl, q4_k, q6_k, iq2_xxs, iq3_xxs, a fresh folder
    `tuning-cuda-{type}` per type shared by both routes, runs adad fill, name fill, adad warm, name warm:
    `IDRAK_ROUTES={adad|name} IDRAK_TUNING_CACHE=$out/tuning-cuda-{type} idrak bench Qwen/Qwen3-1.7B -w {type} -d cuda
    --prompt-tokens 512 --tokens 128 -n 3` → `cuda-{type}-{route}-{fill|warm}.txt`.
  - CPU, bf16, int8, q8_0, q4_0, q4_k, a fresh `tuning-cpu-{type}.tsv` per type, runs adad, name, adad, name:
    `IDRAK_ROUTES={adad|name} IDRAK_CPU_TUNING_FILE=$out/tuning-cpu-{type}.tsv idrak bench Qwen/Qwen3-1.7B -w {type} -d cpu
    --prompt-tokens 128 --tokens 32 -n 2` → `cpu-{type}-{route}-run{1|2}.txt`.
  - Profiles, on the type's filled folder or file, both routes: CUDA prompt (every type) `idrak profile Qwen/Qwen3-1.7B
    -w {type} -d cuda --pass prompt --prompt-tokens 512 --top 30`; CUDA decode step (bf16, int8, q4_0, q4_k) `... -d
    cuda --top 25`; CPU prompt (bf16, q4_0) `... -d cpu --pass prompt --prompt-tokens 128 --top 20` →
    `profile-{device}-{type}-{route}-{prompt|decode}.txt`; a repeat is `...-rerun.txt` (int8 both routes, q4_0 name,
    iq3_xxs adad).
  - Traces (adad: int8, q8_0, q4_k): `IDRAK_ROUTES=adad IDRAK_TUNING_CACHE=$out/tuning-cuda-{type} idrak trace --stacks
    -o $out/trace-cuda-{type}-adad.nettrace --top 20 -- bench Qwen/Qwen3-1.7B -w {type} -d cuda --prompt-tokens 512
    --tokens 128 -n 1` → `trace-cuda-{type}-adad.txt`; read again with `idrak trace --read ... --timeline --slices 40
    --top 5` (q4_k 48) → `-timeline.txt`, `--from S --to S+0.4 --timeline --slices 40 --top 5` → `-timeline-fine.txt`
    (int8 also 3.55–3.95 s → `-timeline-end.txt`), and `--from S --waits --by-thread --top 15` from the generator's start
    (int8 3.31 s, q8_0 3.29 s, q4_k 4.07 s) → `-run-window.txt`.
  - The 0.6B check (the re-bench has no profiles): `Qwen/Qwen3-0.6B`, int8 and q4_0, a fresh folder
    `check-0.6b/tuning-cuda-{type}` filled by an adad bench then a name bench (`--prompt-tokens 512 --tokens 128 -n 3`),
    then each route's prompt profile (`--pass prompt --prompt-tokens 512 --top 30`) → `check-0.6b/`. Both types chose the
    re-bench's prompt lanes (every 512-row entry and both groups equal to `39-rebench/`'s).
  - `tuning-added.txt`: the lines each profile and trace stored in its tuning file (a copy before, a diff after). No
    prompt profile, CPU profile or trace stored a line; the decode profiles stored their untimed warm-up's 128-row and
    window-160 shapes, as in M0. The CUDA adad prompt profiles were made before the diff was added: their folders' last
    writes are the benches' (21:12–21:20), the profiles ran at 21:27–21:28.

**CUDA, Qwen3-1.7B, 512-token prompt and 128 generated tokens, tok/s, warm runs** (ratio adad/name, beside the 0.6B
re-bench's; weights and peak in MiB; `results.txt`, `ratios.txt`)

| Type | adad prompt | name prompt | ratio | 0.6B | adad gen | name gen | ratio | 0.6B | adad weights | name weights | adad peak | name peak |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| bf16 | 21,204 | 19,184 | 1.105 | 1.092 | 197 | 182 | 1.082 | 1.089 | 3,302 | 3,302 | 3,457 | 3,465 |
| f16 | 20,963 | 17,710 | 1.184 | 1.324 | 196 | 106 | 1.849 | 1.690 | 3,895 | 7,177 | 4,050 | 7,348 |
| int8 | 20,410 | 20,749 | 0.984 | 1.041 | 333 | 215 | 1.549 | 1.405 | 2,257 | 2,257 | 2,420 | 2,412 |
| q8_0 | 18,294 | 17,524 | 1.044 | 1.223 | 290 | 105 | 2.762 | 2.055 | 2,357 | 7,177 | 2,534 | 7,340 |
| q4_0 | 18,840 | 17,697 | 1.065 | 1.218 | 332 | 105 | 3.162 | 2.258 | 1,537 | 7,177 | 1,714 | 7,348 |
| iq4_nl | 18,632 | 17,344 | 1.074 | 1.251 | 301 | 105 | 2.867 | 2.119 | 1,537 | 7,177 | 1,714 | 7,348 |
| q4_k | 17,981 | 17,711 | 1.015 | 1.187 | 263 | 106 | 2.481 | 1.921 | 1,537 | 7,177 | 1,718 | 7,348 |
| q6_k | 18,092 | 17,593 | 1.028 | 1.142 | 275 | 106 | 2.594 | 1.980 | 1,960 | 7,177 | 2,141 | 7,348 |
| iq2_xxs | 18,582 | 17,417 | 1.067 | 1.130 | 240 | 105 | 2.286 | 1.865 | 1,037 | 7,177 | 1,214 | 7,348 |
| iq3_xxs | 17,899 | 17,506 | 1.022 | 1.094 | 218 | 106 | 2.057 | 1.837 | 1,242 | 7,177 | 1,423 | 7,348 |

Fill against warm (`spread.txt`): prompts within 0.8% (adad) and 2.7% (name), generation within 1.0%; adad's fill runs
peak 192–270 MiB above its warm runs (the measurement's candidates), name's 0–18 MiB. The `name` columns agree with M0
(prompts within 1.6%, generation within 1 tok/s). f16's adad weights are bf16's plus 593 MiB (151,936 × 2,048 × 2
bytes): the tied head's stored copy (plan 40's decision 9; +297 MiB at 0.6B); bf16's head stays the named one and holds
no copy.

**CPU, Qwen3-1.7B, 128-token prompt and 32 generated tokens, tok/s, both runs** (ratio of the two runs' means, beside
the 0.6B re-bench's; weights equal between the routes; `results.txt`, `ratios.txt`)

| Type | adad prompt (1 / 2) | name prompt (1 / 2) | ratio | 0.6B | adad gen (1 / 2) | name gen (1 / 2) | ratio | 0.6B | weights | adad peak | name peak |
|---|---|---|---|---|---|---|---|---|---|---|---|
| bf16 | 233 / 248 | 115 / 112 | 2.119 | 1.575 | 19.8 / 19.7 | 15.1 / 15.0 | 1.312 | 1.419 | 3,302 | 3,359 | 3,405 |
| int8 | 236 / 209 | 117 / 112 | 1.943 | 1.457 | 32.0 / 31.2 | 25.8 / 25.0 | 1.244 | 1.563 | 2,257 | 2,315 | 2,361 |
| q8_0 | 233 / 274 | 106 / 104 | 2.414 | 3.507 | 26.8 / 27.1 | 3.37 / 3.37 | 7.997 | 6.534 | 2,357 | 2,414 | 2,414 |
| q4_0 | 233 / 204 | 104 / 98.4 | 2.159 | 3.287 | 20.6 / 21.2 | 3.29 / 3.33 | 6.314 | 4.576 | 1,537 | 1,594 | 1,594 |
| q4_k | 196 / 219 | 97.4 / 104 | 2.061 | 3.335 | 14.5 / 14.7 | 3.35 / 3.30 | 4.391 | 3.535 | 1,537 | 1,594 | 1,594 |

**Which lanes each type chose** (CUDA: each type's tuning file and the warm logs' "measured lane:" lines; read as in
"Which lanes each type chose in the re-bench"). Qwen3-1.7B's products (its `config.json`: hidden 2,048, 16 query and 8
key/value heads of 128, intermediate 6,144, 28 layers, a tied head of 151,936): q 2048→2048, k and v 2048→1024, o
2048→2048, gate and up 2048→6144, down 6144→2048, the head 2048→151936. q and o now have the same shape, so one entry
(`2000 512 2048 2048`) serves both: q's runs in its group, o's alone.

Prompt (m = 512), each product's own choice (`StoredMeasured` 2000 for bf16 to iq4_nl, `StoredSuper` 3000 for the
super-block and grid types), then the groups (9700):

| Type | q, o | k, v | gate, up | down | group q/k/v | group gate/up |
|---|---|---|---|---|---|---|
| bf16 | 2 | 2 | 2 | 2 | 1 | 0 |
| f16 | 2 | 2 | 2 | 2 | 1 | 0 |
| int8 | 2 | 6 | 6 | 5 | 1 | 1 |
| q8_0 | 2 | 2 | 2 | 2 | 1 | 0 |
| q4_0 | 2 | 2 | 2 | 2 | 1 | 0 |
| iq4_nl | 2 | 2 | 2 | 2 | 1 | 0 |
| q4_k | 1 | 1 | 1 | 1 | 0 | 0 |
| q6_k | 1 | 1 | 1 | 1 | 0 | 0 |
| iq2_xxs | 1 | 1 | 1 | 1 | 1 | 0 |
| iq3_xxs | 1 | 1 | 1 | 1 | 0 | 0 |

So what ran in the prompt pass (the prompt profiles' kernels tables):

- bf16, f16: q/k/v in the group launch (tensor rows), gate, up, o and down by the native kind (lane 2;
  `native_mma_{bf16|f16}_r64_s6_n128_x`, 112 calls), x packed for it (`native_pack_x_*`).
- q8_0, q4_0, iq4_nl, iq2_xxs: the same, and each weight of gate, up, o and down widened to bfloat16 on every call
  (`native_pack_w_*` or `grid_widen_*`, 112 calls).
- q4_k, q6_k, iq3_xxs: all seven products by the native kind over rows widened on every call (`StoredSuper` 1:
  `super_widen_*` or `grid_widen_*`, 196 calls), and `typed_sum_parts` (56 calls).
- int8: q/k/v and gate/up in group launches (tensor rows over the int8 rows), o by the native kind over its weight packed
  on every call (`native_pack_w_s8_bf16`), down by the native kind reading the int8 codes raw (lane 5,
  `native_mma_bf16_r64_s3_n256_x_ri8`), and `typed_sum_parts` (28 calls).
- At 0.6B (the re-bench; this build's check for int8 and q4_0) bf16, f16, int8, q8_0, q4_0 and iq4_nl ran the tensor
  rows for every product (the groups included), with no widening; the super-block and grid types ran q/k/v in the group
  launch (but iq3_xxs: q by the native kind) and the rest by the tensor rows over rows widened for the call
  (`StoredSuper` 4).

Decode (m = 1): the few-row lane for every product and the head, but bf16's head (named: `5 1 151936 2048 0 = 0`, the warm
log's "bf16 1x2048->151936: named"); the fused gated pass (`4000 1 6144 2048`) 0 for every type; residual + norm after o
(`5000 1 2048 2048`) 1 for every type. The native kernel's tile setting (`9000 2048 2048 2048 9`): 4 (3 stages, swizzled)
in seven folders, 5 (4 stages, swizzled) in bf16 and iq4_nl, 3 (3 stages, not swizzled) in iq2_xxs.

The CPU (each run's own measurement, `cpu-{type}-adad-run{1|2}.txt`):

| Type | Prompt (128 rows), run 1 / run 2 | Decode (1 row), run 1 / run 2 |
|---|---|---|
| bf16 | tiles for k/v, q/o, gate/up; down "today's path" / the same | row lane, the head named (8.8 against the generic 30.4 ms) / the same |
| int8 | "today's path" for k/v, gate/up, down; tiles for q/o / "today's path" for k/v and down; tiles for q/o and gate/up | row lane / row lane, but the head the block route |
| q8_0 | "today's path" for k/v, gate/up, down; tiles for q/o / tiles for k/v and q/o; the block route for gate/up and down | the block route / the block route |
| q4_0 | "today's path" for k/v, q/o, down; tiles for gate/up / tiles, but down "today's path" | row lane for k/v and q/o; the block route for gate/up, down and the head / the same |
| q4_k | "today's path", but gate/up tiles / the same | the block route / the block route |

**CUDA prompt profiles by group** (`prompt-kernel-groups.txt`, ms; "products": every product of the layers with its
widening, packing and summing; pass: each layer drained; q4_0 name and iq3_xxs adad from their repeats, below)

| Type | adad pass | name pass | adad products | name products | adad widening, x packing, sums | adad head | name head |
|---|---|---|---|---|---|---|---|
| bf16 | 25.17 | 27.78 | 20.57 | 21.91 | 0.88 | 0.78 | 0.79 |
| f16 | 24.88 | 29.46 | 19.97 | 23.99 | 0.78 | 0.77 | 1.54 |
| int8 (two runs) | 25.71 / 25.74 | 25.35 / 25.40 | 21.59 / 21.77 | 20.99 / 20.77 | 1.34 / 1.46 | 0.39 | 0.40 |
| q8_0 | 28.39 | 29.36 | 24.52 | 23.89 | 4.49 | 0.43 | 1.54 |
| q4_0 | 28.56 | 29.39 | 23.97 | 23.78 | 4.00 | 0.34 | 1.54 |
| iq4_nl | 28.38 | 29.34 | 24.39 | 23.91 | 3.81 | 0.39 | 1.54 |
| q4_k | 30.00 | 29.30 | 25.87 | 23.85 | 6.17 | 0.46 | 1.54 |
| q6_k | 29.36 | 29.47 | 26.14 | 23.87 | 6.69 | 0.44 | 1.61 |
| iq2_xxs | 28.38 | 29.70 | 24.10 | 24.15 | 4.34 | 0.53 | 1.54 |
| iq3_xxs | 29.23 | 29.44 | 26.33 | 24.12 | 6.66 | 0.65 | 1.54 |

**Findings** (measuring only; none fixed):

- **M6-1. Why adad's prompt lead mostly disappears on Qwen3-1.7B** (`prompt-kernel-groups.txt`, `product-rates.txt`,
  the prompt profiles, the tuning files, `check-0.6b/`):
  - At 0.6B the lead on a stored type came from the group launches and the head. q4_0 (this build's check): adad's
    products 8.60 ms (the gate/up group at 58.2 TFLOP/s, the q/k/v group 53.4, down and o 48.5 and 43.4) against name's
    10.77 ms of separate float32 launches (k and v at 24.4 TFLOP/s, gate/up 45.2, q 48.8, o 45.4, down 51.2); the head
    0.19 against 0.79 ms; pass 12.36 against 15.24 ms. int8's two routes ran nearly the same kernels over the int8 rows
    (adad's tensor rows `gemm_tc_nt_int8w` against name's `gemm_tc_nn_int8w`): pass 11.50 against 11.40 ms.
  - At 1.7B name's separate launches are fast: its float32 products run at 43–66 TFLOP/s in every float32-held type
    (q4_0: k/v 44.0, q/o 55.5, down 62.2, gate/up 66.1), each product 2–4× larger than at 0.6B. So the groups gain less:
    adad's q/k/v group runs at 55–61 TFLOP/s (3.96–4.40 ms) against name's q, k and v in about 4.9 ms (q4_0: half of the
    q/o kernel's 4.33 ms, the two having one shape, plus k and v's 2.74); and the gate/up group is no longer chosen for
    any type but int8 (`9700 512 2048 6144 6144 0 0 = 0`).
  - At 1.7B the measurement chose the native kind for gate, up, o and down of every stored type (`2000 512 ... = 2`,
    `3000 512 ... = 1`), and for q/k/v too where the group lost (q4_k, q6_k, iq3_xxs). For q8_0, q4_0 and iq4_nl this
    replaces 0.6B's tensor rows, which decoded the blocks as the tiles load; the super-block and grid types widened their
    rows for the call at 0.6B too (`StoredSuper` 4). For a stored type the native kind widens the weight to bfloat16 on
    every call and packs x: 3.8–6.7 ms a pass, 13–23% of it (q4_0: `native_pack_w` 3.19 + `native_pack_x` 0.81 ms; q4_k:
    `super_widen` 4.46 + `native_pack_x` 1.19 + `typed_sum_parts` 0.53). The native kernel itself runs at 73–77 TFLOP/s,
    faster than name's float32 kernels, but with the widening adad's products cost as much as name's or more (q4_0 23.97
    against 23.78 ms; q4_k 25.87 against 23.85; iq2_xxs 24.10 against 24.15). What is left of adad's lead is the head
    (0.34–0.65 ms against name's float32 head, 1.54–1.61 ms).
  - Where no weight is widened the lead holds: bf16 (the native kind over its own rows, x packed only: 0.88 ms) 25.17
    against 27.78 ms a pass (bench 1.105; 0.6B 1.092); f16 24.88 against 29.46 (1.184).
  - int8: the measurement chose the native kind for o (its weight packed on every call) and for down (the int8 codes read
    raw). Their kernels beat name's named int8 kernels (o 1.79–1.85 against 2.09–2.11 ms; down 5.01–5.05 against
    5.29–5.31), but bring 1.34–1.46 ms of packing and summing (`native_pack_w_s8_bf16` 0.37–0.41, `native_pack_x_bf16_p16`
    0.37–0.39, `native_pack_x_bf16` 0.26, `typed_sum_parts` 0.33–0.40). So adad's int8 pass is 25.71 / 25.74 against
    25.35 / 25.40 ms in two runs each (bench 0.984).
  - The host is not where it goes: in each trace's run phase (the generator's start to the end) one thread is busy, 69–73%
    of it in native code, mostly `CudaBackend.DownloadRange` (waiting for the device); its collections fall in the
    warm-up's first kernel builds (`PtxKernels`, `CudaBackend.MeasureNative`; `trace-cuda-{int8,q8_0,q4_k}-adad-run-window.txt`,
    `-timeline-fine.txt`). The profiles' passes match the benches' prompt times (512 tokens over the warm bench's tok/s:
    int8 25.7 against 25.1 ms; q4_0 28.6 against 27.2). A 2 ms
    sampling round cannot separate a 25 ms prompt pass from the device waits around it: the traces show no host work
    there, not the pass itself.
  - **Not settled**: whether these choices (the native kind with the weight widened on every call, the gate/up group off)
    are the fastest adad has in the pass: the profiles show only what was chosen. A tuning-entry swap like M2's (for
    example q4_0's 512-row entries set to 6 and its gate/up group to 1), profiled against the type's own choice, would
    settle it (done in M7, below: the choices are the fastest or tied in the pass). Nor why name's separate launches
    gained with the size (the tiles a launch fills is the first suspect): no kernel's tile count or occupancy was read.
- **M6-2. Generation: the few-row lanes read the stored rows** (`profile-cuda-{bf16,int8,q4_0,q4_k}-{adad,name}-decode.txt`,
  a step's median of 5): bf16 5.90 against 6.21 ms; int8 3.87 against 5.37; q4_0 3.88 against 10.41; q4_k 4.57 against
  10.35. name's float32-held types spend 10.49–10.57 ms of a step's kernels in `gemv_nn_f32` over 7,177 MiB of float32
  rows; adad's q4_0 step runs `typed_gemv_*` over its 1,537 MiB. int8's q/k/v and gate/up weights take 2.10 ms in adad's
  `typed_gemv_s8_n4_r1_many` against 3.77 ms in name's `int8_gemv_multi_f32` and `int8_gemv_multi_act_f32` (the gated
  activation fused in; adad's `gated_act_f32` 0.19 ms beside it). Why the
  lead grows from 0.6B was not profiled (no decode profile at 0.6B).
- **M6-3. Two single prompt profiles were outliers**: q4_0 name 36.51 ms (every kernel slower, its head `gemv_nn_f32`
  3.03 against 1.54 ms; `profile-cuda-q4_0-name-prompt.txt`) and iq3_xxs adad 32.77 ms (norms 1.82 against 1.27 ms,
  activation 1.55 against 0.99; `profile-cuda-iq3_xxs-adad-prompt.txt`). Their repeats gave 29.39 and 29.23 ms, in line
  with the other types and the benches (`-rerun.txt`); the int8 repeats agreed within 0.2% (25.71 / 25.74, 25.35 /
  25.40). The cause was not found (no clock or temperature was read): one profile a type can mislead; the comparisons
  above use the repeats.
- **M6-4. The fused gated decode pass is never chosen on 1.7B**: `4000 1 6144 2048` is 0 in all ten folders (the re-bench:
  1 for bf16, f16, int8, q8_0, q4_k, q6_k, iq2_xxs at 0.6B; this build's 0.6B check stored 0 for int8, the re-bench 1).
- **M6-5. The native kernel's tile setting is measured once a folder, on one shape, and came out three ways in ten
  folders** (`9000 2048 2048 2048 9` = 3, 4 or 5; the fill and warm logs of each type print the same setting). Which
  setting is fastest at each product was not checked.
- **M6-6. The CPU** (`results.txt`, `profile-cpu-{bf16,q4_0}-{adad,name}-prompt.txt`): adad's prompt lead grew for bf16
  and int8 (2.12 and 1.94 against 1.58 and 1.46 at 0.6B) and shrank for the stored types (2.06–2.41 against 3.29–3.51);
  its generation lead grew for the stored types (4.4–8.0 against 3.5–6.5) and shrank for bf16 and int8 (1.31 and 1.24
  against 1.42 and 1.56). Prompt profiles: bf16 adad 585 ms a pass with 289 MiB allocated against name's 1,068 ms and
  4,362 MiB (name widens each weight on every call, M0-6); q4_0 654 against 1,259 ms (name's k/v `matmul_stored
  [1x128x1024]` 250.8 ms, M0-7). The lane choices differ between the two runs of a type (above): M1's subject. The causes
  of the CPU's moves from 0.6B were not traced.

### The lane swap on Qwen3-1.7B (M7)

Plan 40's M7, the question M6-1 left open. At 1.7B the measurement chose the native kind for the stored types' prompt
products, a lane that widens each weight on every call (13–23% of the pass). Is that choice the fastest in the full pass,
or would the tensor rows (0.6B's choice, which read the stored rows with no per-call widening) win there, although they
lost when each product was timed alone? Measuring only: no code was changed; the choices were swapped by editing copies
of M6's tuning folders. Files: `test-output\adad\m7-swap\`, copied to [40-m7-swap-1.7b/](40-m7-swap-1.7b/) (no trace was
made).

- **Code and CLI**: the M6 build (commit `de1f8a96`, `idrak 0.4.1-dev.2611191810`, informational version
  `0.4.1-dev.2611191810+de1f8a96e66d9b2d98860553311a470cf69243ed`), checked before the first run. The M6 tuning folders
  name its commit, so it reads them, and their copies, as filled. No other run beside these (the same idle service of
  another project ran throughout).
- **The variants** (`m7.sh setup`; each edit written as `swap-{type}[-A|-B].diff`, M6's file against the edited copy):
  - q4_0, q8_0: the four `2000 512 …` entries 2 → 6 (o's and down's lanes, and the grouped products' own entries); the
    gate/up group `9700 512 2048 6144 6144 0 0` 0 → 1.
  - int8: q/o `2000 512 2048 2048 7` 2 → 6, down `2000 512 2048 6144 7` 5 → 6 (its groups were already 1).
  - q4_k, iq2_xxs, A: the four `3000 512 …` entries 1 → 4 (the tensor rows over rows widened for the call) and the
    512-row groups → 1 (q4_k both, iq2_xxs gate/up); B: the same with 3 (the tensor rows decoding as each tile loads, no
    widening).
  - Each key's candidate-count field kept, and each lane set among that key's candidates. `StoredMeasured` builds its
    candidates in lane order 0–6 (`CudaBackend.Typed.cs`), so q4_0's and q8_0's six and int8's seven include 6 (0.6B
    chose 6 with the same counts). `StoredSuper`'s are {0, 1, [2], [3], [4]} (`CudaBackend.Super.cs`): q4_k's five are all
    of them, iq2_xxs's four are {0, 1, 3, 4} (its q/k/v group already runs the decoding kernel `gemm_tc_nt_iq2_xxs`). A
    group's are {0, 1}. A stored value that is not among the candidates is measured again (`TryPersistedLocked`), which
    the diffs would show.
  - Controls: `tuning-control-{type}`, untouched copies (byte-equal to M6's folders at the start; the q4_0, q4_k and int8
    folders also hold the 128-row entries M6's decode profiles stored, which a 512-token prompt does not read).
- **Runs**, one at a time (23:03–23:11), `m7.sh`, the route set on every run, `out` = `D:/Projects/idrak/test-output/adad/m7-swap`:
  - Each type's control: `IDRAK_ROUTES={adad|name} IDRAK_TUNING_CACHE=$out/tuning-control-{type} idrak bench
    Qwen/Qwen3-1.7B -w {type} -d cuda --prompt-tokens 512 --tokens 128 -n 3` (a warm run of each route); the adad prompt
    profile `idrak profile Qwen/Qwen3-1.7B -w {type} -d cuda --pass prompt --prompt-tokens 512 --top 30`; the greedy
    continuation `idrak complete Qwen/Qwen3-1.7B -w {type} -d cuda -i $out/prompt.txt --temperature 0 --max-tokens 48`.
  - Each variant on its `tuning-swap-…` folder: an adad fill bench, the greedy continuation (compared with the control's
    before going on), an adad warm bench, the adad prompt profile.
  - `prompt.txt`: a generic newsletter text of 475 tokens (`idrak tokenize … --count`). Its prompt pass falls in the
    bench's 512 size class (`TuneSizes.Class`), so it runs the same product lanes.
  - Every run diffed its folder (`tuning-added.txt`). No bench or profile stored or changed a line; no run stored a
    `ProductRoute` line; every swapped entry still held its edited value at the end. The 12 continuations stored
    exact-size (m = 475) tile and split entries and window-512 decode entries, none of them a product's lane.
  - Derived tables: `compare.sh` → `compare.txt` (the benches), `lanes.sh` → `lanes.txt` (the profiles' kernels by
    lane), M6's `product-rates.sh` → `product-rates.txt`.
- **The swapped lanes ran**: the warm logs' "measured lane:" lines name the tensor rows for o and down (q4_0, q8_0, int8:
  "tensor rows (today's tensor-core kernel over the rows as stored …)"; A: "tensor rows (widened to bfloat16 rows …)";
  B: "tensor rows (the description's values decoded as each tile loads …)"), and no gate/up or q/k/v product runs on its
  own lane (the groups). The swapped profiles run only `gemm_tc_nt_*` product kernels (A also `super_widen`/`grid_widen`,
  56 calls: o's and down's rows widened for the call): no `native_mma_*`, `native_pack_*` or `typed_sum_parts`.
- **Same output**: every variant's 48 greedy tokens are byte-equal to its control's (`complete-*-adad.txt`). For iq2_xxs
  the text is degenerate ("ony" repeated) under the control, both variants and `name` alike
  (`complete-control-iq2_xxs-name.txt`): the 2-bit type's quality on this text, not a lane's; for that type the check
  proves less.

**The benches and passes** (prompt tok/s; the controls in the same session; pass: the prompt profile, each layer
drained; `compare.txt`, `lanes.txt`)

| Type, variant | What ran in the prompt pass | swapped fill / warm | control adad | control name | warm / control adad | pass ms swapped / control |
|---|---|---|---|---|---|---|
| q4_0 | o, down: tensor rows; gate/up, q/k/v: groups | 18,312 / 18,504 | 18,665 | 17,922 | 0.991 | 28.61 / 27.94 |
| q8_0 | the same | 17,492 / 17,714 | 18,330 | 17,679 | 0.966 | 29.66 / 28.67 |
| int8 | the same | 20,274 / 20,169 | 20,214 | 20,585 | 0.998 | 26.21 / 25.72 |
| q4_k A | o, down: tensor rows over rows widened for the call; groups decoding as tiles load | 16,923 / 17,077 | 18,247 | 17,775 | 0.936 | 30.81 / 29.67 |
| q4_k B | every product decoding as tiles load | 16,228 / 16,284 | 18,247 | 17,775 | 0.892 | 32.17 / 29.67 |
| iq2_xxs A | as q4_k A | 18,541 / 18,775 | 18,946 | 17,832 | 0.991 | 27.88 / 27.80 |
| iq2_xxs B | as q4_k B | 18,576 / 18,479 | 18,946 | 17,832 | 0.975 | 28.50 / 27.80 |

The spread: a variant's fill and warm runs differ by up to 1.3%; the controls against M6's warm runs of the same folders
by up to 2.0% (adad) and 2.4% (name) (`compare.txt`).

**The products by lane** (ms, the prompt profiles' kernels: `lanes.txt`; "products" the sum of the lane columns)

| Profile | native kind | widening | x packing | sums | tensor rows alone | tensor-row groups | products | pass |
|---|---|---|---|---|---|---|---|---|
| q4_0 control | 15.81 | 3.09 | 0.87 | – | – | 4.25 | 24.02 | 27.94 |
| q4_0 swapped | – | – | – | – | 8.56 | 15.54 | 24.10 | 28.61 |
| q8_0 control | 15.74 | 3.62 | 0.88 | – | – | 4.25 | 24.49 | 28.67 |
| q8_0 swapped | – | – | – | – | 8.92 | 15.81 | 24.74 | 29.66 |
| int8 control | 6.79 | 0.39 | 0.64 | 0.36 | – | 13.46 | 21.65 | 25.72 |
| int8 swapped | – | – | – | – | 7.80 | 13.51 | 21.32 | 26.21 |
| q4_k control | 19.55 | 4.61 | 1.43 | 0.65 | – | – | 26.24 | 29.67 |
| q4_k A | – | 1.48 | – | – | 7.44 | 17.81 | 26.73 | 30.81 |
| q4_k B | – | – | – | – | 9.94 | 17.64 | 27.57 | 32.17 |
| iq2_xxs control | 15.77 | 3.41 | 0.92 | – | – | 3.91 | 24.00 | 27.80 |
| iq2_xxs A | – | 1.42 | – | – | 7.42 | 15.03 | 23.86 | 27.88 |
| iq2_xxs B | – | – | – | – | 8.87 | 14.97 | 23.84 | 28.50 |

Each product in the swapped passes (`product-rates.txt`; ms, TFLOP/s: gate/up group, down, q/k/v group, o):

- q4_0: 11.11 (64.9), 6.28 (57.5), 4.42 (54.4), 2.28 (52.7).
- q8_0: 11.50 (62.7), 6.56 (55.0), 4.31 (55.8), 2.37 (50.8).
- int8: 9.91 (72.8), 5.76 (62.6), 3.61 (66.7), 2.04 (58.9); the control's native kind: down 5.05 (71.4), o 1.74 (69.0).
- q4_k A: 13.16 (54.8), 5.44 (66.3, rows widened for the call), 4.65 (51.7), 2.00 (60.3, widened); B: 12.98 (55.6),
  7.32 (49.3), 4.65 (51.7), 2.62 (46.0).
- iq2_xxs A: 10.98 (65.7), 5.41 (66.7, widened), 4.05 (59.4), 2.01 (59.8, widened); B: 10.99 (65.6), 6.45 (55.9), 3.97
  (60.5), 2.42 (49.7).
- The controls' native kind (q4_0, q8_0, q4_k, iq2_xxs): 73.8–76.4 TFLOP/s.

A kernel table's sum carries each launch's own timing, and the controls' passes launch 373–709 kernels against the
swaps' 261–317, so a kernel sum overstates a many-launch pass. For example, q4_0's control sums 28.55 ms of kernels
against a 27.94 ms pass, the swap 28.21 against 28.61 (products + head + other in `lanes.txt`). So the passes and the
benches decide, not the kernel sums.

**Conclusion**:

- **In the full pass the measured choice is the fastest or tied, for every type; the tensor rows never win.** Within the
  spread (−0.2% to −0.9%): q4_0, int8, iq2_xxs A. Slower beyond it: q8_0 −3.4%, iq2_xxs B −2.5%, q4_k A −6.4%, q4_k B
  −10.8% (`compare.txt`). Every swapped pass is the longer one (`lanes.txt`).
- **Timing each product alone did not mislead here**: warm for the 32-value blocks and int8 (`MeasureStoredRoute`, whose
  comment records that cold timing cost bfloat16, int8 and q4_0 prompts 6%), cold for the super-blocks and grids
  (`MeasureSuper`), and the groups both ways. No lane it rejected beats its choice in the pass by more than the spread.
- **The per-call widening is paid back.** The lanes that avoid it run slower per product. The decoding tensor rows run
  q4_0's products at 52.7–64.9 TFLOP/s against the native kind's 76.1, so its products take 24.10 ms against the native
  kind's 24.02 ms with the widening. q4_k's decoding tensor rows run at 46.0–55.6 TFLOP/s: 27.57 against 26.24 ms. For
  the block, super-block and grid types no tensor-rows lane measured here reaches the native kind's rate: 46.0–65.7
  TFLOP/s decoding as the tiles load, 59.8–66.7 over widened rows, against 73.8–76.4. So M6-1's open question is
  settled: at 1.7B, adad's stored-type prompt is at the best of its own lanes.
- **int8**:
  - The swap removes 1.39 ms of packing and sums, and its o is quicker (2.04 ms against the native kind's 1.74 plus
    packing).
  - But its down by the tensor rows (5.76 ms) is slower than the native kind reading the int8 codes raw (5.05, plus its
    share of the packing and sums).
  - The benches are equal (0.998), the passes 26.21 against 25.72 ms.
  - Neither adad variant reaches name's int8 (20,585 tok/s: 0.980 and 0.982).
- **An observation from the profiles**: with the gate/up group launch on (every swap), the gated activation after it
  takes 0.56–0.75 ms, against 0.98–1.00 ms after the native kind's separate gate and up products (the profiles'
  `gated_act_f32`; int8 0.74 in both, its group on in both). Its cause was not checked.

### The winner everywhere on Qwen3-1.7B (M8)

Plan 40's M8, on the owner's word ("test with one type of each type family on the winner"; "use the same method as M7
on the same model"). The winner is the native kind: Adad's GemmMma, `StoredMeasured` lane 2 or `StoredSuper` lane 1. M8
forces it on every prompt product (the 512-row entries) of one type per family, with every 512-row group launch off, so
each product runs the native kind on its own. Decode entries stay as they were. The question: does that beat the
measured choices (the group launches, int8's raw and tensor-row lanes) in the full pass? Measuring only, M7's method:
copies of M6's tuning folders edited, no code changed. Files: `test-output\adad\m8-winner\`, copied to
[40-m8-winner-1.7b/](40-m8-winner-1.7b/) (no trace was made).

- **Code and CLI**: the M6 build (`idrak 0.4.1-dev.2611191810`, informational version
  `0.4.1-dev.2611191810+de1f8a96e66d9b2d98860553311a470cf69243ed`), checked before the first run. Worktree at `78dd095f`.
  No other run beside these (the same idle service of another project ran throughout).
- **The edits** (`m8.sh setup`; each written as `force-{type}.diff`, M6's file against the edited copy): every
  `ProductRoute 2000 512 …` entry set to 2, or every `3000 512 …` entry to 1, and every `9700 512 …` group to 0.
  - Each key's candidate-count field is kept. Lane 2 is among each 2000 key's candidates: bf16's five and q4_0's six
    already hold 2, and int8's seven are every lane 0–6 (`CudaBackend.Typed.cs`). `StoredSuper` always lists 0 and 1
    (`CudaBackend.Super.cs`).
  - What each type needed, checked against its M6 folder:
    - bf16 (16-bit float): the q/k/v group 1 → 0 (its products were already 2, gate/up's group already 0).
    - int8 (plain integer): k/v and gate/up 6 → 2, down 5 → 2 (q/o was already 2), both groups 1 → 0.
    - q4_0 (32-value blocks): the q/k/v group 1 → 0.
    - iq2_xxs (grids): the q/k/v group 1 → 0.
    - q4_k (super-blocks): nothing. Every product was already `StoredSuper` 1 and both groups 0 (`force-q4_k.diff` is
      empty), so in M6 q4_k already ran the winner everywhere. It runs here only as a control, with no forced runs.
  - Controls: `tuning-control-{type}`, untouched copies (byte-equal to M6's folders at the start).
- **Runs**, one at a time (23:27–23:32), `m8.sh`, the route set on every run, with M7's commands:
  - each type's control: adad warm and name warm benches, the adad prompt profile, the greedy continuation (q4_k: no
    continuation, nothing to compare it with);
  - each forced type: an adad fill bench, the greedy continuation (compared with the control's before going on), an adad
    warm bench, the adad prompt profile.
  - The continuation is M7's `prompt.txt` (475 tokens, the bench's 512 size class), 48 greedy tokens.
  - One profile was repeated (`profile-force-iq2_xxs-prompt-rerun.txt`). The first (31.64 ms) had its norm and
    activation kernels slower than every other profile's (`add_rms_norm_affine_f32` 1.46 ms against 1.24–1.28,
    `gated_act_f32` 1.21 against 0.95–1.00 where gate and up run as separate products), as in M6-3; the repeat is used.
  - Every run diffed its folder (`tuning-added.txt`, 36 runs). No bench or profile stored or changed a line; no run
    stored a `ProductRoute` line; every forced entry still held its value at the end. The 8 continuations stored
    exact-size (m = 475) tile and split entries and window-512 decode entries, none of them a product's lane.
  - Derived tables: `compare.sh` → `compare.txt`, M7's `lanes.sh` → `lanes.txt`, `shares.sh` → `shares.txt`, M6's
    `product-rates.sh` → `product-rates.txt`.
- **The winner ran everywhere it was forced**: each forced warm log's "measured lane:" lines name the native kind for all
  four 512-row shapes (k/v, q/o, gate/up, down; "native bfloat16 kind (packed)", iq2_xxs "native bfloat16 kind (rows
  widened from the description)"). The forced profiles run `native_mma_bf16_r64_s6_n128_x` 196 times (seven products
  × 28 layers), and no `gemm_tc_nt_*` product kernel.
- **Same output**: every forced type's 48 greedy tokens are byte-equal to its control's (`complete-*-adad.txt`).
  iq2_xxs's text is the degenerate one M7 found under every route.

**The benches and passes** (prompt tok/s, the controls in the same session; pass: the prompt profile, each layer
drained; `compare.txt`, `lanes.txt`)

| Type (family) | forced fill / warm | control adad | control name | forced warm / control adad | pass ms forced / control |
|---|---|---|---|---|---|
| bf16 (16-bit float) | 20,682 / 20,785 | 20,965 | 19,294 | 0.991 | 26.70 / 25.30 |
| int8 (plain integer) | 18,377 / 18,358 | 20,252 | 20,569 | 0.906 | 29.09 / 25.73 |
| q4_0 (32-value blocks) | 18,658 / 18,596 | 18,835 | 17,809 | 0.987 | 28.71 / 28.00 |
| q4_k (super-blocks) | nothing to force | 18,286 | 17,796 | – | – / 29.28 |
| iq2_xxs (grids) | 18,389 / 18,320 | 18,781 | 17,917 | 0.975 | 29.35 / 27.86 |

The spread: a forced type's fill and warm runs differ by at most 0.5%; the controls against M6's warm runs of the same
folders by up to 1.7% (adad) and 2.9% (name) (`compare.txt`). Forced against name: bf16 1.077, int8 0.893, q4_0 1.044,
iq2_xxs 1.022 (q4_k's control 1.028).

**The products by lane and by step** (ms, `lanes.txt`: the native kind's kernel, the weight's conversion to bfloat16 on
every call (`native_pack_w_*`, `super_widen_*`, `grid_widen_*`), x packing (`native_pack_x_*`), the parts' sums
(`typed_sum_parts`), the tensor-row group launches; "products" their sum)

| Profile | native kernel | weight conversion | x packing | sums | tensor-row groups | products | pass |
|---|---|---|---|---|---|---|---|
| bf16 control | 15.81 | – | 0.85 | – | 3.50 | 20.16 | 25.30 |
| bf16 forced | 19.46 | – | 1.10 | 0.51 | – | 21.07 | 26.70 |
| int8 control | 6.73 | 0.39 | 0.59 | 0.34 | 13.46 | 21.51 | 25.73 |
| int8 forced | 19.45 | 4.49 | 1.13 | 0.51 | – | 25.59 | 29.09 |
| q4_0 control | 15.71 | 2.96 | 0.80 | – | 4.26 | 23.73 | 28.00 |
| q4_0 forced | 19.72 | 4.19 | 1.11 | 0.58 | – | 25.59 | 28.71 |
| q4_k control (the winner everywhere) | 19.50 | 4.54 | 1.21 | 0.66 | – | 25.91 | 29.28 |
| iq2_xxs control | 15.90 | 3.79 | 0.87 | – | 3.96 | 24.51 | 27.86 |
| iq2_xxs forced (repeat) | 19.62 | 4.68 | 1.25 | 0.56 | – | 26.10 | 29.35 |

The native kernel runs at 73.2–74.2 TFLOP/s over all seven products in the forced passes (and q4_k's), against
75.6–76.5 over the four large ones (gate, up, o, down) in the controls; the controls' q/k/v groups run at 68.8 (bf16),
56.5 (q4_0) and 60.8 TFLOP/s (iq2_xxs) (`product-rates.txt`).

**How much of the winner's time is conversion** (`shares.txt`; the winner's products: the native kernel, the weight's
conversion, x packing and the sums)

| Family (type, profile) | weight conversion ms | share of the winner's products | share of the pass | conversion, x packing and sums: share of products |
|---|---|---|---|---|
| 16-bit float (bf16, forced) | 0 | 0% | 0% | 7.6% |
| plain integer (int8, forced) | 4.49 | 17.6% | 15.4% | 24.0% |
| 32-value blocks (q4_0, forced) | 4.19 | 16.4% | 14.6% | 23.0% |
| super-blocks (q4_k, control: already the winner everywhere) | 4.54 | 17.5% | 15.5% | 24.7% |
| grids (iq2_xxs, forced, repeat) | 4.68 | 17.9% | 15.9% | 24.8% |

**Conclusion**:

- **Forcing the winner everywhere does not beat the measured choices in any family; they hold.** Against its same-session
  control, each forced type ties or loses: q4_0 −1.3% and bf16 −0.9% (within the session-to-session spread, but their
  passes are 0.71 and 1.40 ms longer), iq2_xxs −2.5% (pass +1.49 ms), int8 −9.4% (pass +3.36 ms) (`compare.txt`,
  `lanes.txt`).
- **Where the q/k/v group launch is turned off** (bf16, q4_0, iq2_xxs), q, k and v run as three native-kind products. The
  kernel then costs about what the group did, but the separate products add x packing, `typed_sum_parts` (56 calls) and,
  for a stored type, the conversion of the q/k/v weights on every call:
  - bf16: native kernel 19.46 ms against 15.81 + the group's 3.50, and +0.24 ms x packing and +0.51 ms sums;
  - q4_0: conversion 4.19 against 2.96 ms;
  - iq2_xxs: conversion 4.68 against 3.79 ms.
- **int8**: the native kind converts every int8 weight to bfloat16 on every call (4.49 ms), where the measured choices
  read the int8 rows as they are (the tensor-row groups, 13.46 ms for q/k/v and gate/up) or the int8 codes raw (down, no
  packing pass). That conversion is why the forced int8 is the clearest loss.
- **The conversion share**: wherever the winner runs a stored type, converting the weight to bfloat16 on every call takes
  16.4–17.9% of its products' time (14.6–15.9% of the pass), and 23.0–24.8% with x packing and the sums. bf16, which converts
  nothing, spends 7.6% of its products on x packing and the sums. So the measured choices, which keep the group launches
  and int8's lanes over the stored rows, are the faster ones in the full pass on Qwen3-1.7B.

### Another family: Llama 3.2 1B and 3B (M9)

Plan 40's M9, on the owner's word ("i need a different family model with 2 sizes to bench with othere than qwen", "to
confirm result"). M6's protocol on `meta-llama/Llama-3.2-1B-Instruct` and `meta-llama/Llama-3.2-3B-Instruct`, `adad`
against `name` in the same runs. Measuring only: no code changed. Files: `test-output\adad\m9-llama3.2\`, copied
(without the `.nettrace` and `.speedscope.json` files) to [40-m9-llama3.2/](40-m9-llama3.2/).

- **Code and CLI**: the M6 build (`idrak 0.4.1-dev.2611191810`, informational version
  `0.4.1-dev.2611191810+de1f8a96e66d9b2d98860553311a470cf69243ed`), checked before the first run. No source changed
  since `de1f8a96` (`git diff --stat de1f8a96 5783b047 -- src`: nothing). Worktree at `5783b047`. No other run beside
  these (the same idle service of another project ran throughout).
- **The models** (`idrak pull`, `{1b|3b}/pull.txt`: 1B revision `9213176726f5`, 2.31 GB; 3B `0cb88a4f764b`, 5.99 GB). Both
  are `LlamaForCausalLM` in bfloat16, with tied embeddings and a vocabulary of 128,256 (`{1b|3b}/config.txt`, from each
  `config.json`):

| Model | hidden | layers | heads × size | KV heads | FFN | q | k, v | o | gate, up | down | head |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Llama 3.2 1B | 2,048 | 16 | 32 × 64 | 8 | 8,192 | 2048→2048 | 2048→512 | 2048→2048 | 2048→8192 | 8192→2048 | 2048→128256 |
| Llama 3.2 3B | 3,072 | 28 | 24 × 128 | 8 | 8,192 | 3072→3072 | 3072→1024 | 3072→3072 | 3072→8192 | 8192→3072 | 3072→128256 |

  The 1B's hidden size equals Qwen3-1.7B's (2,048), with a wider FFN (8,192 against 6,144), narrower k and v (512) and
  half the layers. The 3B's products are the largest measured so far. The bench accepted the family (no "not
  registered"); the 1B's head size of 64 runs `flash_tc_fwd_bf16_d64`.
- **Commands** (`m9.sh SIZE …`, one model run a call, the route set on every run, M6's commands with the model changed):
  - CUDA: the ten types each in a fresh `tuning-cuda-{type}`, runs adad fill, name fill, adad warm, name warm.
  - CPU: bf16, int8, q8_0, q4_0, q4_k each with a fresh `tuning-cpu-{type}.tsv`, runs adad, name, adad, name.
  - Profiles on the filled folders and files, both routes:
    - CUDA prompt: bf16, int8, q4_0, q4_k, iq2_xxs;
    - CUDA decode: bf16, int8, q4_0, q4_k;
    - CPU prompt: bf16, q4_0.
  - One trace (`trace --stacks`) of the one ratio out of line with Qwen3's pattern (CPU q4_k at 3B, below).
  - The greedy continuation of M7's `prompt.txt` (475 tokens under the Llama tokenizer too) on CUDA bf16, adad and name.
  - Derived tables: `ratios.sh` → `ratios.txt`, M6's `spread.sh` → `spread.txt`, M6's `kernel-groups.sh` →
    `kernel-groups.txt`, M7's `lanes.sh` → `lanes.txt`, `product-rates.sh` (M6's with the layer count and shapes as
    arguments) → `product-rates.txt`, `cpu-lanes.txt` (the CPU logs' lane lines).
- **Every run loaded and ran** (all 120 bench runs exit 0, `{1b|3b}/results.txt`; every profile and continuation exit 0).
  Under `name` the 3B's float32-held types fit on the card (weights 13,071 MiB, peak 13,298 MiB). No profile or trace
  stored a product's lane (`{1b|3b}/tuning-added.txt`: the decode profiles stored their warm-up's 128-row and window
  entries, the continuations their exact-size entries).
- **Fill against warm** (`spread.txt`): prompts within 0.7% (adad) and 0.9% (name); adad's fill runs peak 267–406 MiB
  above its warm runs.

**CUDA, 512-token prompt and 128 generated tokens, tok/s, warm runs** (`ratios.txt`; the ratio is adad/name, beside
Qwen3-0.6B's re-bench and Qwen3-1.7B's M6)

| Type | 1B adad prompt | 1B name prompt | 3B adad prompt | 3B name prompt | prompt ratio Q0.6B / Q1.7B / L1B / L3B | 1B gen adad / name | 3B gen adad / name | gen ratio Q0.6B / Q1.7B / L1B / L3B |
|---|---|---|---|---|---|---|---|---|
| bf16 | 30,953 | 28,033 | 11,519 | 10,638 | 1.092 / 1.105 / 1.104 / 1.083 | 275 / 242 | 112 / 98.4 | 1.089 / 1.082 / 1.136 / 1.138 |
| f16 | 30,925 | 25,998 | 11,425 | 9,974 | 1.324 / 1.184 / 1.190 / 1.145 | 277 / 150 | 111 / 58.7 | 1.690 / 1.849 / 1.847 / 1.891 |
| int8 | 29,084 | 30,343 | 10,999 | 11,200 | 1.041 / 0.984 / 0.959 / 0.982 | 469 / 293 | 193 / 118 | 1.405 / 1.549 / 1.601 / 1.636 |
| q8_0 | 26,281 | 26,194 | 9,764 | 9,855 | 1.223 / 1.044 / 1.003 / 0.991 | 416 / 150 | 174 / 58.7 | 2.055 / 2.762 / 2.773 / 2.964 |
| q4_0 | 27,457 | 26,011 | 10,169 | 9,952 | 1.218 / 1.065 / 1.056 / 1.022 | 480 / 150 | 209 / 58.8 | 2.258 / 3.162 / 3.200 / 3.554 |
| iq4_nl | 27,349 | 26,065 | 10,094 | 9,900 | 1.251 / 1.074 / 1.049 / 1.020 | 426 / 149 | 176 / 58.7 | 2.119 / 2.867 / 2.859 / 2.998 |
| q4_k | 26,607 | 26,262 | 9,787 | 9,922 | 1.187 / 1.015 / 1.013 / 0.986 | 369 / 150 | 153 / 58.7 | 1.921 / 2.481 / 2.460 / 2.606 |
| q6_k | 26,169 | 26,251 | 9,674 | 9,938 | 1.142 / 1.028 / 0.997 / 0.973 | 385 / 149 | 160 / 58.8 | 1.980 / 2.594 / 2.584 / 2.721 |
| iq2_xxs | 27,036 | 26,116 | 9,953 | 9,834 | 1.130 / 1.067 / 1.035 / 1.012 | 337 / 150 | 139 / 58.7 | 1.865 / 2.286 / 2.247 / 2.368 |
| iq3_xxs | 26,397 | 25,991 | 9,778 | 9,809 | 1.094 / 1.022 / 1.016 / 0.997 | 305 / 150 | 126 / 58.8 | 1.837 / 2.057 / 2.033 / 2.143 |

Weights (MiB, `{1b|3b}/results.txt`): adad 1B bf16 2,389, f16 2,890, int8 1,714, q8_0 1,785, q4_0/iq4_nl/q4_k 1,196,
q6_k 1,500, iq2_xxs 837, iq3_xxs 984; 3B 6,192, 6,944, 3,883, 4,071, 2,540, 3,329, 1,606, 1,989. Under name, bf16 and
int8 hold the same weights, and f16 and the GGUF types are held as float32: 5,247 MiB (1B) and 13,071 MiB (3B).

**CPU, 128-token prompt and 32 generated tokens, tok/s** (the mean of two runs; `ratios.txt`, the runs in
`{1b|3b}/results.txt`)

| Type | 1B prompt adad / name | 3B prompt adad / name | prompt ratio Q0.6B / Q1.7B / L1B / L3B | 1B gen adad / name | 3B gen adad / name | gen ratio Q0.6B / Q1.7B / L1B / L3B |
|---|---|---|---|---|---|---|
| bf16 | 334 / 134 | 121 / 49.4 | 1.575 / 2.119 / 2.493 / 2.449 | 28.3 / 22.1 | 11.4 / 8.59 | 1.419 / 1.312 / 1.281 / 1.327 |
| int8 | 391 / 137 | 130 / 48.0 | 1.457 / 1.943 / 2.850 / 2.695 | 46.9 / 38.0 | 19.2 / 15.3 | 1.563 / 1.244 / 1.233 / 1.259 |
| q8_0 | 346 / 152 | 123 / 68.0 | 3.507 / 2.414 / 2.281 / 1.807 | 40.6 / 5.62 | 16.7 / 1.77 | 6.534 / 7.997 / 7.222 / 9.380 |
| q4_0 | 399 / 154 | 130 / 69.2 | 3.287 / 2.159 / 2.591 / 1.879 | 32.8 / 5.55 | 12.8 / 1.75 | 4.576 / 6.314 / 5.896 / 7.314 |
| q4_k | 314 / 156 | 102 / 65.0 | 3.335 / 2.061 / 2.013 / 1.569 | 21.9 / 5.85 | 6.71 / 1.79 | 3.535 / 4.391 / 3.744 / 3.735 |

**Which lanes each type chose** (CUDA: each type's tuning file and its adad warm log; read as in the re-bench's section).
Prompt (m = 512), each product's own choice, then the groups (`9700`):

| Type | 1B q, o | 1B k, v | 1B gate, up | 1B down | 1B groups q/k/v, gate/up | 3B q, o | 3B k, v | 3B gate, up | 3B down | 3B groups q/k/v, gate/up |
|---|---|---|---|---|---|---|---|---|---|---|
| bf16 | 2 | 6 | 2 | 2 | 1, 0 | 2 | 2 | 2 | 2 | 0, 0 |
| f16 | 2 | 6 | 2 | 2 | 1, 0 | 2 | 2 | 2 | 6 | 0, 0 |
| int8 | 2 | 6 | 6 | 6 | 1, 0 | 6 | 2 | 5 | 6 | 1, 0 |
| q8_0 | 2 | 6 | 2 | 2 | 1, 0 | 2 | 2 | 2 | 2 | 1, 0 |
| q4_0 | 2 | 6 | 2 | 2 | 1, 0 | 2 | 2 | 2 | 6 | 1, 0 |
| iq4_nl | 2 | 6 | 2 | 2 | 1, 0 | 2 | 2 | 2 | 2 | 1, 0 |
| q4_k | 1 | 1 | 1 | 1 | 0, 0 | 1 | 1 | 1 | 4 | 0, 0 |
| q6_k | 1 | 1 | 1 | 4 | 0, 0 | 1 | 1 | 1 | 4 | 0, 0 |
| iq2_xxs | 1 | 1 | 1 | 1 | 0, 0 | 1 | 1 | 1 | 4 | 0, 0 |
| iq3_xxs | 1 | 1 | 1 | 1 | 0, 0 | 1 | 1 | 1 | 4 | 0, 0 |

- **Against Qwen3-1.7B** (M6; the same hidden size as the 1B):
  - The 1B chose as Qwen3-1.7B did for most types: the native kind for the large products (all but int8's, and q6_k's
    down: the tensor rows over widened rows), the q/k/v group on for bf16, f16, q8_0, q4_0 and iq4_nl and off for the
    super-block and grid types, the gate/up group off. Its k and v (2048→512) chose the tensor rows, but where the q/k/v
    group is on they run in it either way.
  - int8 at 1B: down by the tensor rows (Qwen3-1.7B: raw int8, lane 5); gate/up by the tensor rows on their own (the
    group off; Qwen3-1.7B: the group on).
  - iq2_xxs at 1B: no q/k/v group (Qwen3-1.7B: on).
- **At 3B** (the largest products measured): the native kind still wins most products, but:
  - the down product (8192→3072, the longest k) moved to the tensor rows for f16, int8 and q4_0, and to the tensor rows
    over rows widened for the call (`StoredSuper` 4) for the super-block and grid types;
  - int8's gate/up chose raw int8 (lane 5), its q/o the tensor rows;
  - bf16's and f16's q/k/v group is off.
- **Decode** (m = 1): the few-row lane for every product and head but bf16's head (named: "bf16 1x3072->128256 …:
  named"). The fused gated pass (`4000`) is 0 in all twenty folders. Residual + norm (`5000`) is 1 but for iq4_nl (both
  sizes) and iq3_xxs (3B), which chose 0; on Qwen3-1.7B every type chose 1.
- **CPU** (`cpu-lanes.txt`): as on Qwen3, every decode shape takes the row lane or the block route (bf16's head the named
  kernel). The prompt shapes take tiles, "today's path" or the block route, and the choice differs between the two runs
  of a type for int8, q8_0, q4_0 and q4_k (M1's subject).

**The prompt profiles** (CUDA, ms; `kernel-groups.txt`: "products" every product of the layers with its widening, x
packing and sums; 3B q4_k's adad profile is its repeat, below)

| Profile | 1B adad pass / products / widening, x packing, sums / head | 1B name pass / products / head | 3B adad pass / products / widening, x packing, sums / head | 3B name pass / products / head |
|---|---|---|---|---|
| bf16 | 17.17 / 13.74 / 0.56 / 0.66 | 18.77 / 14.83 / 0.66 | 45.22 / 40.23 / 3.43 / 0.99 | 48.48 / 41.87 / 0.98 |
| int8 | 18.18 / 14.62 / 0.34 / 0.33 | 17.83 / 14.07 / 0.34 | 47.03 / 41.14 / 0.32 / 0.49 | 45.98 / 39.95 / 0.50 |
| q4_0 | 19.49 / 16.23 / 2.99 / 0.29 | 20.10 / 16.01 / 1.30 | 50.63 / 45.56 / 5.69 / 0.42 | 51.64 / 44.72 / 1.94 |
| q4_k | 20.19 / 17.15 / 4.24 / 0.39 | 20.14 / 15.88 / 1.30 | 52.76 / 48.97 / 11.25 / 0.58 | 51.98 / 44.64 / 1.94 |
| iq2_xxs | 19.74 / 17.61 / 4.34 / 0.45 | 20.08 / 16.07 / 1.30 | 52.27 / 47.78 / 10.18 / 0.67 | 52.98 / 45.67 / 1.93 |

The native kind runs its products at 70–81 TFLOP/s on both models; name's float32 products at 30.9–68.4 TFLOP/s (1B,
the 512-wide k/v lowest) and 49.6–67.5 (3B) (`product-rates.txt`).

**Decode steps** (CUDA, the median of 5; `{1b|3b}/profile-cuda-{type}-{route}-decode.txt`), ms adad / name: 1B bf16
4.06 / 4.51, int8 2.57 / 3.86, q4_0 2.52 / 7.21, q4_k 3.13 / 7.19; 3B bf16 9.46 / 10.64, int8 5.74 / 8.94, q4_0
5.40 / 17.83, q4_k 7.10 / 17.90. name's float32-held types spend 85–90% of a step's kernels in `gemv_nn_f32`.

**CPU prompt profiles** (`{1b|3b}/profile-cpu-{bf16,q4_0}-{adad,name}-prompt.txt`), pass ms and MiB allocated, adad /
name:

- bf16: 1B 387 ms, 226 MiB / 906 ms, 2,912 MiB; 3B 1,049 ms, 465 MiB / 2,652 ms, 8,444 MiB (name widens each weight on
  every call, M0-6).
- q4_0: 1B 412 / 795 ms; 3B 1,261 / 1,872 ms (name's k/v `matmul_stored [1x128x1024]` 342 ms, M0-7).

**Findings** (measuring only; none fixed):

- **M9-1. Qwen3's CUDA pattern holds on Llama, ordered by size.**
  - Llama 1B lands next to Qwen3-1.7B (the same hidden size) on every ratio (`ratios.txt`): prompt within 0.041 of it
    (the furthest q8_0 1.003 against 1.044, iq2_xxs 1.035 against 1.067, q6_k 0.997 against 1.028, int8 0.959 against
    0.984), generation within 0.06.
  - At 3B the stored types' prompt lead is gone (0.973–1.022; q8_0, q4_k, q6_k and iq3_xxs slightly below name), as the
    trend from Qwen3-0.6B through Qwen3-1.7B and Llama 1B runs.
  - bf16 (1.083) and f16 (1.145) keep a prompt lead; int8 stays just below name (0.959, 0.982), as on Qwen3-1.7B.
  - The generation lead grows with size: q4_0 2.26 → 3.16 → 3.20 → 3.55; every stored type 2.14–3.55 at 3B; int8 1.64;
    f16 1.89; bf16 1.14.
- **M9-2. The cause is M6-1's, seen again in the profiles** (`kernel-groups.txt`, `product-rates.txt`):
  - For a stored type, adad's products with the per-call widening cost as much as name's float32 products or more: 1B
    q4_0 16.23 against 16.01 ms; 3B q4_0 45.56 against 44.72, q4_k 48.97 against 44.64, iq2_xxs 47.78 against 45.67.
  - What is left of adad's lead is the head: 0.29–0.67 ms against name's float32 head, 1.30 ms (1B) and 1.94 ms (3B).
  - At 3B the widening, x packing and sums are 10.2–11.3 ms of the super-block and grid types' passes (19–21%).
  - bf16, which widens nothing, keeps its lead in the products: 13.74 against 14.83 ms (1B), 40.23 against 41.87 (3B).
- **M9-3. Memory holds**: adad holds every type as stored (1B 837–2,890 MiB, 3B 1,606–6,944), name the float32 copies
  (5,247 and 13,071 MiB).
- **M9-4. The CPU**:
  - adad leads every type on both models: prompt 1.57–2.85, generation 1.23–9.38.
  - The prompt lead for bf16 and int8 is larger on Llama (2.45–2.85, against 1.46–2.12 on Qwen3). For the stored types it
    shrinks with size, as on Qwen3 (3B: 1.57–1.88).
  - The generation lead grows with size for q8_0 and q4_0 (7.22 → 9.38, 5.90 → 7.31).
- **M9-5. One ratio out of line: CPU q4_k's generation at 3B is flat** (3.74 at 1B and 3B, where q8_0 and q4_0 grow).
  - Its two adad runs give 7.43 and 5.98 tok/s with the same lanes (the block route on every decode shape,
    `cpu-q4_k-adad-run{1,2}.txt`).
  - The trace's own run gave 10.0 tok/s over its 8 tokens (`trace-cpu-q4_k-adad.txt`), again with the block route
    everywhere.
  - The same block-route product (1x3072->8192) was measured at 1.289, 1.155 and 0.635 ms in the three processes.
  - So the CPU's q4_k decode at 3B moves by up to 1.7× between processes, and the flat ratio sits inside that move. The
    trace's decode window (`trace-cpu-q4_k-adad-decode-window.txt`, 6.2 threads busy on average) holds mostly the first
    decode step's lane measurements, not the timed steps; what makes a process fast or slow is not settled (no thread
    placement or clock was read).
- **M9-6. One profile was an outlier**: 3B q4_k adad's first prompt profile, 64.02 ms with the same kernels as its repeat
  (52.76 ms, in line with the bench's 52.3 ms); the repeat is used (`profile-cuda-q4_k-adad-prompt{,-rerun}.txt`).
- **M9-7. The same text**: on both models the 48 greedy tokens of bf16 under adad and under name are byte-equal
  (`{1b|3b}/complete-bf16-{adad,name}.txt`).

**Conclusion**:

- **Qwen3's results hold on Llama 3.2 at both sizes**:
  - the generation lead, growing with size;
  - the memory (stored rows against float32 copies);
  - the CPU's lead on every type;
  - the prompt pattern: bf16 and f16 ahead, int8 just behind, the stored types' lead shrinking with size.
- **At 3B, a larger shape than any measured before, the stored types' prompt is even with name (0.973–1.022)**.
  M6-1's mechanism shows directly in the profiles: the per-call widening costs what name's float32 copy saves, and only
  the head is left.
- **What does not carry over exactly**: some lane choices (int8's down and gate/up, the 3B's down product on the tensor
  rows, bf16's q/k/v group off at 3B, iq4_nl's and iq3_xxs's residual + norm). These are choices measured per shape, as
  designed, not failures.
- **The one out-of-line ratio** (M9-5) is the CPU's run-to-run variance at 3B. It was not explained.

### B's quality: lane 3 forced on the prompt (M10)

Plan 40's M10. Candidate B for the stored types' prompt product keeps each weight's codes exact as int8, rounds x to int8
per block of 32 values, and scales the int32 sums into float32. The existing lane 3 (`StoredMeasured` 3, Adad's block
route: `blocks_quantize_q8_0` then `matmul_blocks_*`, `PtxKernels.Blocks.cs`) does that rounding. Today's native lane
instead rounds x and the widened weight to bfloat16. So lane 3's perplexity against the native lane's is B's quality cost.
Lane 3's speed is a side measurement only: its kernel is the simple one (64 × 64 tiles). Measuring only: no code changed.
The choices were forced by editing tuning folders, as in M7.

Models, on the owner's word: Qwen3-1.7B and Llama 3.2 3B Instruct; Llama 3.2 1B was dropped, and no 1B run was made.
Files: `test-output\adad\m10-lane3-quality\`, copied (no `.nettrace`, `.speedscope.json` or lock files) to
[40-m10-lane3-quality/](40-m10-lane3-quality/) (`qwen3-1.7b/`, `llama-3b/`, the text and the scripts).

- **Code and CLI**: the M6 build (`idrak 0.4.1-dev.2611191810`, informational version
  `0.4.1-dev.2611191810+de1f8a96e66d9b2d98860553311a470cf69243ed`), checked before the first run. No source changed
  since `de1f8a96` (`git diff --stat de1f8a96 e6dc787b -- src`: nothing). Worktree at `e6dc787b`.
- **The perplexity path** (`PerplexityCommand.cs`, read before any run):
  - The text is tokenized and read in windows. Each window runs `network.ForwardFirst` (inside the network's precision
    scope, `Sequential.Run`: bfloat16 products, as generation runs them) over all its rows, with no key/value cache.
  - The head then runs on every scored row in one chunk (`Losses.TokenLogProbabilities`, 1,024 rows a chunk). It runs
    outside that scope.
  - `IDRAK_ROUTES` (`ProductRoutes.Current`) and `IDRAK_TUNING_CACHE` (`TuningCache`) are read by the library for the
    whole process, so `perplexity` honours them as the bench does.
  - An operations trace of one lane-3 run per model (`{model}/ops-q4_0-lane3.jsonl`, `idrak trace --levels operations`)
    shows every product at 512 rows. Per layer and window: q `matmul_stored_many [1,512,hidden]`; k/v and gate/up
    `matmul_stored_many`; o and down `matmul_stored [1,512,hidden]`; and the head `matmul_stored [511,vocabulary]`, 13 calls.
- **The text and settings, the same for every run** (`text-info.txt`):
  - `text.txt` is the repository's `docs/optimization.md` at `e6dc787b`: technology English with a few C# examples,
    27,225 bytes, sha256 `72873d2f…3f27f86`.
  - It is 7,341 Qwen3 tokens and 6,988 Llama tokens.
  - Every run: `-d cuda --window 512 --max-tokens 6656`. That is 13 windows of exactly 512 tokens and 6,643 scored
    tokens on both models, so every window's layers run at m = 512 (the bench's 512-row keys).
- **Runs** (`m10.sh MODEL …`, one model run a call, the route set on every run):
  - For q4_0 and int8: perplexity under adad on a fresh folder (fill, then warm) — the native lane (the measured choice);
    a copy with lane 3 forced (two runs); and under name on its own fresh folder (two runs).
  - bf16 under adad (fill, warm) and under name (two runs): the unquantized reference.
  - The side speed measurement per type:
    - a copy of the native folder (`TYPE-ctl`) and the lane-3 folder, each a bench fill (the perplexity folders hold no
      decode keys) then a warm bench at the record's CUDA sizes;
    - then the adad prompt profile of each.
  - Command: `IDRAK_ROUTES={adad|name} IDRAK_TUNING_CACHE=… idrak perplexity MODEL text.txt -w TYPE -d cuda --window 512
    --max-tokens 6656`.
- **The edit** (`m10.sh MODEL force TYPE`, written as `{model}/force-{type}.diff`):
  - Every prompt-shaped `StoredMeasured` entry (`ProductRoute 2000 512 N K C 0 0`) but the head's set to 3, each key's
    candidate count kept.
  - Every 512-row group (`9700 512 …`) set to 0. Left on, a group would run q/k/v (and int8's gate/up on Qwen3) as one
    tensor-row launch instead of each product's lane 3.
  - Lane 3 is among every forced key's candidates: no forced entry was measured again (`tuning-added.txt`: no run removed
    or changed a line).
- **What the fills chose** (the native lane):
  - Qwen3-1.7B: q4_0 as in M6 (every product the native kind; the q/k/v group on). int8: q/o the native kind; k/v,
    gate/up and down the tensor rows; both groups on (M6's bench had down on lane 5).
  - Llama 3B: as in M9 (q4_0's down the tensor rows; int8's gate/up raw int8, q/o and down the tensor rows, the q/k/v
    group on).
  - In every q4_0 and int8 adad folder the 511-row head (outside the precision scope its key has two candidates, `… 2 0 0`)
    chose lane 3. So the head takes B's rounding in both the native and the lane-3 runs, and the comparison isolates the
    layers' products. bf16's head has its own entry (`4 512 …`).
- **Lane 3 ran**:
  - The lane-3 bench logs name "block route (x as q8_0, s8 tensor cores)" for all four 512-row shapes.
  - The lane-3 prompt profiles run `blocks_quantize_q8_0` and `matmul_blocks_*` 196 times each (7 products × 28 layers),
    and no `native_mma_*`, `native_pack_w_*` or `gemm_tc_nt_*` (`{model}/profile-{type}-lane3-prompt.txt`).
- **Every run of a kind repeated exactly**: the two runs of each kind give the same perplexity and mean negative
  log-likelihood to the printed digits (3 and 4 decimals) (`perplexity.txt`).

**Qwen3-1.7B, perplexity** (`perplexity.txt`, `qwen3-1.7b/results.txt`; bf16: adad 24.142, name 24.148)

| Type | native lane | lane 3 | name | lane 3 − native | lane 3 − name | native − name | native, lane 3, name against bf16 (adad) |
|---|---|---|---|---|---|---|---|
| q4_0 | 27.690 | 27.737 | 27.683 | +0.047 (+0.17%) | +0.054 (+0.20%) | +0.007 (+0.03%) | +14.70%, +14.89%, +14.67% |
| int8 | 23.969 | 23.943 | 23.961 | −0.026 (−0.11%) | −0.018 (−0.08%) | +0.008 (+0.03%) | −0.72%, −0.82%, −0.75% |

**Llama 3.2 3B, perplexity** (`perplexity.txt`, `llama-3b/results.txt`; bf16: adad 26.871, name 26.874)

| Type | native lane | lane 3 | name | lane 3 − native | lane 3 − name | native − name | native, lane 3, name against bf16 (adad) |
|---|---|---|---|---|---|---|---|
| q4_0 | 28.763 | 28.776 | 28.772 | +0.013 (+0.05%) | +0.004 (+0.01%) | −0.009 (−0.03%) | +7.04%, +7.09%, +7.07% |
| int8 | 26.796 | 26.831 | 26.791 | +0.035 (+0.13%) | +0.040 (+0.15%) | +0.005 (+0.02%) | −0.28%, −0.15%, −0.30% |

**Across the two models**:

- Lane 3 against the native lane: q4_0 +0.17% (Qwen3-1.7B) and +0.05% (Llama 3B); int8 −0.11% and +0.13%.
- Lane 3 against name: −0.08% to +0.20%. The native lane against name: −0.03% to +0.03%.
- Against the bf16 reference, the weight types themselves: q4_0 +14.7% (Qwen3-1.7B) and +7.0% (Llama 3B); int8 −0.7%
  and −0.3% (int8 is below bf16 on this text on both models).
- bf16 adad against bf16 name: −0.02% and −0.01%.

**The side speed measurement** (`speed.txt`; warm benches at the record's CUDA sizes, tok/s; the prompt profiles' kernels,
ms; the block products' rate: the layers' 512-row products' work, 1,443 GFLOP (Qwen3-1.7B) and 2,886 GFLOP (Llama 3B),
over `matmul_blocks_*`'s ms)

| Model, type | prompt, native control / lane 3 | lane 3 / control | pass ms, control / lane 3 | lane 3: quantize x / block products | block products' rate | generation, control / lane 3 |
|---|---|---|---|---|---|---|
| Qwen3-1.7B q4_0 | 19,079 / 7,734 | 0.405 | 27.87 / 67.03 | 2.69 / 61.07 | 23.6 TFLOP/s | 334 / 334 |
| Qwen3-1.7B int8 | 20,536 / 8,162 | 0.397 | 25.79 / 62.94 | 2.86 / 56.89 | 25.4 TFLOP/s | 333 / 333 |
| Llama 3B q4_0 | 10,089 / 4,826 | 0.478 | 51.16 / 106.47 | 3.38 / 98.82 | 29.2 TFLOP/s | 209 / 209 |
| Llama 3B int8 | 11,022 / 5,097 | 0.462 | 47.12 / 101.67 | 3.53 / 92.90 | 31.1 TFLOP/s | 195 / 195 |

The controls' products (ms; `speed.txt`): Qwen3-1.7B q4_0 the native kind 15.71 + conversion 3.01 + x packing 0.83 +
the q/k/v group 4.08; int8 the tensor rows 18.86 + the native kind 1.73 + 0.70 of packing; Llama 3B q4_0 21.42 + 4.95 +
0.62 + sums 0.44 + tensor rows 18.72; int8 the native kind (raw int8) 19.84 + x packing 0.32 + tensor rows 21.13. The
decode keys were not forced, so generation is unchanged.

**Findings** (measuring only; none fixed):

- **M10-1. Lane 3's rounding moves perplexity by −0.11% to +0.17% against the native lane, on both models and both
  types**, and by −0.08% to +0.20% against name (`perplexity.txt`). Repeated runs agree to the printed digits.
- **M10-2. The weight type moves it more**: q4_0 is +7.0–14.9% above bf16 on these models and text, under every lane;
  int8 is 0.15–0.85% below bf16.
- **M10-3. The perplexity path's head is already on lane 3** in every q4_0 and int8 adad run: it runs outside the
  network's precision scope, where its key has two candidates and the fill chose 3. Generation's head is not measured
  here.
- **M10-4. Lane 3's current kernel runs the prompt at 0.40–0.48 of the native control** (block products 23.6–31.1
  TFLOP/s, x quantization 2.7–3.5 ms a pass) (`speed.txt`, the profiles).
- **Not measured**: any kernel for B other than lane 3's, perplexity on other texts, windows or types, and the CPU.

### q4_k traced before B (M11)

Plan 40's M11. q4_k shows the per-call widening most clearly; the prompt profiles give only the GPU's kernel split. M11
adds what they cannot show: the host's work and waits in the prompt, the kernel sequence, the widening's memory speed, and
the device memory. Models: Qwen3-1.7B and Llama 3.2 3B Instruct, q4_k under adad and under name, on CUDA. Measuring only:
no code changed. Files: `test-output\adad\m11-q4k-trace\`, copied (no `.nettrace`, `.speedscope.json` or lock files) to
[40-m11-q4k-trace/](40-m11-q4k-trace/) (`qwen3-1.7b/`, `llama-3b/`, `m11.sh`, `derived.sh`, `derived.txt`,
`kernels-gemv.txt`).

- **Code and CLI**: the M6 build (`idrak 0.4.1-dev.2611191810`, informational version
  `0.4.1-dev.2611191810+de1f8a96e66d9b2d98860553311a470cf69243ed`), checked before the first run. No source changed
  since `de1f8a96` (`git diff --stat de1f8a96 bd4ece61 -- src`: nothing). Worktree at `bd4ece61`.
- **Tuning folders**:
  - Copies of the warm q4_k folders, one per route: Qwen3-1.7B from M6 (`m6-proof-1.7b/tuning-cuda-q4_k`), Llama 3B from
    M9 (`m9-llama3.2/3b/tuning-cuda-q4_k`). Each was filled by its adad and name runs.
  - No run stored or changed a line (`{model}/tuning-added.txt`: every one of the 6 runs a model "(none)").
- **Runs** (`m11.sh MODEL …`, one model run a call, the route set on every run):
  - `idrak trace --stacks -o … --top 25 -- bench MODEL -w q4_k -d cuda --prompt-tokens 512 --tokens 1 -n 10`: a warm-up and
    10 timed prompt passes, one token each, so the prompt dominates.
  - Reads of each trace: `--timeline` (coarse, then 10 ms slices: `trace-{route}-timeline*.txt`), the prompt window
    `--from S --to E --by-thread --all-frames` (`trace-{route}-prompt-window.txt`), and `--focus` on the dispatcher's
    methods (`trace-{route}-focus-*.txt`).
  - `idrak profile MODEL -w q4_k -d cuda --pass prompt --prompt-tokens 512 --top 40` (`profile-{route}-prompt.txt`).
  - A warm bench at the record's CUDA sizes (`bench-{route}.txt`).
  - `idrak bench --kernels gemv -d cuda` once (`kernels-gemv.txt`).
  - Derived figures: `derived.sh` → `derived.txt`.
- **The prompt window** (the timelines):
  - The bench's generator starts, then the warm-up pass enqueues and builds its kernels on first use. From the end of that
    build the main thread waits for the device in `CudaBackend.DownloadRange` until the run ends. The 10 timed passes'
    launches all fall in that window, about 1,960 product calls (10 × 28 layers × 7).
  - Windows: Qwen3-1.7B adad 4.34–4.65 s, name 13.21–13.76 s; Llama 3B adad 6.93–7.63 s, name 29.54–30.28 s.

**The host in the prompt window** (`trace-{route}-prompt-window.txt`, `trace-adad-focus-*.txt`,
`trace-name-focus-matmul.txt`, `derived.txt`; sampled every 2 ms)

| Model, route | busy threads | main thread in `DownloadRange` (waiting for the device) | host's prompt work (`TextGenerator.Stream.Prefill`) | collections | managed allocations |
|---|---|---|---|---|---|
| Qwen3-1.7B adad | 1.0 | 225 of 309 ms (73%) | 73.0 ms | 1 (0.4 ms) | 26 MiB |
| Qwen3-1.7B name | 1.0 | 502 of 550 ms (91%) | 36.6 ms | 0 | 10 MiB |
| Llama 3B adad | 1.0 | 605 of 693 ms (87%) | 70.3 ms | 1 (4.9 ms) | 23 MiB |
| Llama 3B name | 1.0 | 688 of 738 ms (93%) | 29.6 ms | 0 | 8 MiB |

- **Host time a product call** (the sampled ms over the 1,960 calls):
  - adad's dispatcher (`CudaBackend.StoredSuper`): 26.6 µs a call (Qwen3-1.7B) and 29.3 µs (Llama 3B). Of that:
    - building the candidate list's lane-2 entry (`NativeDirectLane` → `NativeDirect`, which formats a string on every
      call: `StringBuilder` work and `PtxKernels.SuperCode`): 12.0 and 13.1 µs;
    - the lane itself (`RunSuperLane` → `SuperNative`): 11.1 and 11.5 µs. Within it: renting the widened buffer
      (`TryAllocate`), the k-split list (`NativeSplits`: LINQ `Where`, `List` growth), the launches (`LaunchStaged`) and
      `RuntimeFieldInfoStub.FromPtr`;
    - `TensorRowsSource` 1.8 and 1.9 µs; `TryTuned` (the stored choice's lookup) 0.8 and 2.1 µs.
  - adad's group decision (`SharedInput` → `TensorRowsGroup`, its string concatenation included) adds 4.9 ms over the
    window (Qwen3-1.7B: 39.1 ms total, of which 34.2 ms are the products' `StoredSuper`).
  - name (`Tensor.MatMul` over float32 rows): 10.4 and 12.0 µs a call.
- **Synchronisation**: the only wait on the main thread is `DownloadRange` (each pass's logits). The host's enqueue (adad
  about 7 ms a pass against passes of 29 ms and 53 ms) finishes before the device does, so the host's time is hidden
  behind the device's in every window. The other threads are idle (thread pools).
- **The widened weight's buffer is rented per call from the device pool**:
  - `SuperNative` calls `Allocate(n × kp / 2)` and releases it after the product (`CudaBackend.Super.cs`).
  - `Allocate` takes a free block of that size from the size-keyed pool (`_pool`, `CudaBackend.cs`) and calls
    `cuMemAlloc` only when none is free. So the block is reused across calls of the same size; a call pays only the
    rental (`TryAllocate`; a `CudaStorage` wrapper allocated each time, 1.9 MiB of the window's allocations).
  - The k-split partials come from a cached buffer (`_splitPartials.Get`, `CudaBackend.Typed.cs`).
- **The managed allocations**: per call, the type code rebuilt as a string (`PtxKernels.SuperCode` via
  `NativeSource.get_Code`: Int32 boxes, char arrays, strings), `NativeSplits`' arrays, LINQ iterators and delegates, and
  the `CudaStorage` wrappers.

**The prompt profiles' kernels** (`profile-{route}-prompt.txt`, `derived.txt`)

- **Launch order**: the profiler keeps only each kernel's calls and time, not the launch order. The order below is the
  code's (`SuperNative`, `RunNativeShape`); the counts a layer are the profiles'. For one product:
  1. x packed into bfloat16, once for a group sharing x: `native_pack_x_bf16`;
  2. the weight's q4_k rows widened into bfloat16 rows: `super_widen_k256x144_*`;
  3. the product: `native_mma_bf16_*`, into partials when k is split;
  4. the partials summed: `typed_sum_parts`.
- **Qwen3-1.7B, a layer**: 7 `super_widen`, 7 `native_mma` (every product the native kind, no group), 4
  `native_pack_x` (by the code: q/k/v sharing one, o, gate/up sharing one, down), 2 `typed_sum_parts`.
- **Llama 3B, a layer**:
  - 7 `super_widen`;
  - 6 `native_mma` (4 of `r64_s6_n128`, 2 of `r64_s4_n256`) and 1 `gemm_tc_nt_bf16w widened` (down: the tensor rows over
    its widened rows, `StoredSuper` 4);
  - 3 `native_pack_x`, 4 `typed_sum_parts`.
- **Name, a layer**: 7 `gemm_tc_nn` over the float32 rows, no conversion kernels.

| Profile | pass (each layer drained) | sum of the kernels | launches | pass − kernels |
|---|---|---|---|---|
| Qwen3-1.7B adad | 29.20 ms | 30.29 ms | 709 | −1.09 ms |
| Qwen3-1.7B name | 29.31 ms | 29.52 ms | 345 | −0.21 ms |
| Llama 3B adad | 53.07 ms | 54.73 ms | 737 | −1.66 ms |
| Llama 3B name | 51.74 ms | 51.94 ms | 345 | −0.20 ms |

In every profile the kernels' timed sum is larger than the pass. Each launch's timing adds its own time, so the excess
grows with the launches. So no idle gap between kernels shows at this tool's resolution. The layers table gives each
layer's drained time (Qwen3-1.7B adad 0.968–1.054 ms, name 0.938–0.991; Llama 3B adad 1.808–1.924, name 1.722–1.792),
not its gaps.

**The widening's memory speed** (`derived.txt`)

| Model | values widened a pass | q4_k rows read | bfloat16 rows written (k needs no padding) | `super_widen` | rate |
|---|---|---|---|---|---|
| Qwen3-1.7B | 1,409,286,144 | 756 MiB | 2,688 MiB | 4.407 ms | 819 GB/s |
| Llama 3B | 2,818,572,288 | 1,512 MiB | 5,376 MiB | 8.782 ms | 822 GB/s |

- The card's measured read rates, from Idrak's tools:
  - the bf16 head's one-row product in the decode profiles reads the head's rows once: 593.5 MiB in 0.755–0.756 ms,
    823–824 GB/s (M6); 751.5 MiB in 0.953–0.959 ms, 822–827 GB/s (M9);
  - the kernel bench's largest bf16 gemv (62.5 MiB, warm repeats): 736.8 GB/s.
- No Idrak tool measures a device-to-device copy (`bench --kernels` offers matmul, gemv and attention).
- So the widening moves its bytes at the same rate as the device's measured single reads. Its writes are 3.6× its reads.

**Device memory** (`bench-{route}.txt`, `trace-{route}.txt`, `derived.txt`; MiB)

| Model, route | weights | in use after the runs | peak | peak − weights |
|---|---|---|---|---|
| Qwen3-1.7B adad | 1,537 | 1,541 | 1,718 (trace run 1,704) | 181 (167) |
| Qwen3-1.7B name | 7,177 | 7,177 | 7,348 (7,334) | 171 (157) |
| Llama 3B adad | 2,540 | 2,552 | 2,790 (2,776) | 250 (236) |
| Llama 3B name | 13,071 | 13,071 | 13,296 (13,282) | 225 (211) |

- The widened buffer at the largest product (n × k × 2 bytes): 24 MiB (Qwen3-1.7B: gate, up, down) and 48 MiB (Llama 3B).
- adad's peak above its weights is 10 MiB (Qwen3-1.7B) and 25 MiB (Llama 3B) more than name's, less than one widened
  buffer. The bench's figures do not split the peak by buffer, so whether the widened buffer sits on top of the rest is
  not settled.
- After the runs adad holds 4 and 12 MiB more than its weights; name holds none.

**Findings** (measuring only; none fixed):

- **M11-1. The host does not limit the prompt.** In every window one thread is busy, and the main thread spends 73–93% of
  it waiting for the device. adad's host work is 2.0 and 2.4 times name's (73.0 against 36.6 ms over 10 passes; 70.3
  against 29.6), all of it hidden behind the device.
- **M11-2. adad's dispatcher costs 26.6–29.3 µs a product call, against name's 10.4–12.0.** About 45% of adad's is the
  candidate list's lane-2 check (`NativeDirectLane`), which formats a string on every call. Managed allocations are 23–26
  MiB over 10 passes, against 8–10 under name.
- **M11-3. The widened buffer is rented per call from the device pool** (no driver allocation after the first call of a
  size); the split partials come from a cached buffer.
- **M11-4. No idle gap between kernels shows**: the kernels' timed sum exceeds the pass in all four profiles.
- **M11-5. The widening runs at the device's measured memory rate**: 819–822 GB/s against single reads of 822–827 GB/s.
  It writes 2,688 and 5,376 MiB of bfloat16 rows a pass.
- **M11-6. Device memory**: adad's peak above its weights is 10–25 MiB above name's, its weights 19% to 21% of name's.

### Earlier measurements (superseded: lane choices shared across types, mixed builds)

- Round 13's CUDA table (`results-dec-warm.txt`, made by a build before `cf77ceeb`; q6_k and iq3_xxs from
  `results-gather.txt`, the `cf77ceeb` build): bf16 adad 48,016 / name 46,191 prompt; f16 47,756 / 35,503; int8 49,136 /
  48,530; q8_0 43,826 / 36,438; q4_0 45,660 / 35,904; iq4_nl 43,986 / 36,386; q4_k 42,871 / 36,295; q6_k 41,753 / 36,383;
  iq2_xxs 42,184 / 35,877; iq3_xxs 40,064 / 36,247. In the `results-gather.txt` run (adad only, `cf77ceeb`): bf16 46,462,
  int8 46,770, q4_0 43,158, q4_k 41,971.
- At the start of this session (`results-before-warm.txt`): bf16 adad 17,368 / name 18,056 prompt; int8 16,713 / 18,335;
  f16 17,309 / 16,200; q4_0 16,742 / 16,198; adad generation bf16 365, int8 343 (the residual-norm regression).
- The CPU rounds before the re-bench (`results-cpu-warm.txt`, `results-cpu-fill.txt`, made before the `5a056e27` commit):
  two independent measurements each, their lanes differing between runs.
- The lane list those rounds gave ("never chosen: the native kind ...") was wrong for the native kind: with the shared
  keys, later types read the first type's choices.

### Where the proof used `name`

- Under `measured` and `adad`, a packed int8 or bfloat16 weight's own named kernel was a candidate against the generic
  lanes (CUDA `RoutedNamed`, CPU `NamedLane`/`MeasureLane`).
- The tied bfloat16 head stayed today's bfloat16 head under the routes (`DecoderBuilder.cs`, with `Embedding.FromHead`):
  its prompt product ran `gemm_tc_nn_bf16w`, and on the CPU its decode step chose the named kernel (4.6 ms against the
  generic lane's 34 ms over the named layout).
- The tensor rows are built with Idrak's tensor-core generator (`TensorCoreGemm`); their sources are chosen from the
  description (float bits, integer bits, block layout, decoder), not from a format's name.

## Measurement facts learned

- A fresh tuning folder's first run under-reports; the warm run of a filled folder is the result. Fresh folders differ by
  about 3–5% on CUDA prompts: the final build gave bf16 46,462 on a fresh folder and 47,390 on the previous round's folder
  (read on the console, in no file), whose choices had given 48,016 with the previous build (the two builds measure bf16
  the same way; they differ only in timing the gathering decoders cold).
- The CUDA lane choices were keyed without the type (above): in the shared folders of rounds 11-13, types after the first
  reused earlier types' choices. The CPU's lane choices are measured anew in every run, and one run's measurement can pick
  the slower lane (q8_0 above).
- `Tune` does not measure while profiling or during graph capture (`CanMeasure`); a profile on an unfilled folder shows
  fallback choices.
- Warm timings fit lanes that read contiguous rows; the super-block and grid decoders, which gather scattered bytes, need
  cold timings (measured both ways: cold everywhere cost bf16, int8, q4_0 about 6%; cold for the gathering decoders only
  fixed q6_k 38,952 → 41,753 and iq3_xxs 38,317 → 40,064).
- CPU benches move run to run (the unchanged `name` route moved about 10%; `adad`'s prompts between the fill and warm runs
  17–20% for int8 and q4_k, where the two runs chose different lanes).

## Not proved (must not be taken as proved)

Nothing below was built, measured or checked by the proof. A session must not present any of it as a result of the
proof; to count, each must be built, tested and benched against the numbers above.

### Not built

- **Idrak.Adad itself** (plan 39): the package, its devices `adad-cpu` and `adad-cuda`, and moving the code out of core.
- **The tied bfloat16 head as stored rows shared with the embedding.** Under the proof's routes the head was today's
  bfloat16 head (see "Where the proof chose to use `name`").
- **The tensor rows generated per description** in place of the `bf16B`, `f16B`, `int8B` modes: a proposal made after the
  owner asked why the lanes name specific types; not built, not measured. The proof's tensor rows are the modes as built.
- **A lanes setting for the package** (`IDRAK_ADAD_LANES`, named in plan 39's first draft): not built; the proof's setting
  is `IDRAK_ROUTES`.
- **Vulkan and HIP lanes.**
- **Descriptions for q4_1, q5_0, q5_1, IQ4_XS, IQ1_S, IQ1_M, IQ2_XS, IQ2_S, IQ3_S**: not described, so the routes have no
  lane of their own for them.

### Decided by the owner for the package, not proved by the proof

- No `name` in any scenario, case or stage of the package.
- Only the lanes that proved fast in the benches.
- Its own devices; the route code moves, the general fixes and the descriptions stay in core.

These are decisions; whether the package keeps the proof's numbers under them is what plan 39's acceptance bench (step 11)
has to show. In particular: without `RoutedNamed` and today's tied bfloat16 head, the proof has no measurement of the
packed-format weights or the tied head on the Adad lanes alone.

### Not measured

- **Each type's own lane choices in rounds 11-13 on CUDA**: with the shared keys, a type that ran after another with the
  same shapes and candidate counts did not measure its own choices (the re-bench in `test-output\rebench\` measures each
  type in its own folder).
- **A GGUF file made by ggml**: every benched weight was quantized by Idrak's own simple encoders.

- **Any other GPU or CPU.** Every number is from one machine (an RTX 5070 Ti, 70 SMs, compute 12.0, and its 24-thread CPU
  with AVX-VNNI-INT8). Which lane wins, and by how much, may differ elsewhere; a CPU without AVX-VNNI-INT8 has no block
  route measured.
- **Any other model or size.** Only Qwen/Qwen3-0.6B (hidden 1,024, 28 layers, a tied head).
- **Other prompt lengths and batches.** Prompts of 512 tokens (CUDA) and 128 (CPU), batch 1; generation of 128 and 32
  tokens. Long contexts, batches above 1 and rows of different lengths were not benched under the routes.
- **Quality on real text.** Correctness was checked by the bounds and bit-for-bit tests above, not by perplexity or task
  results.
- **float32 runs on CUDA under the routes** (no bfloat16 products, so no tensor rows): not benched.
- **The `generic` and `measured` routes in the last rounds**: the final benches ran `adad` and `name` only.
- **Q2_K, Q3_K, Q5_K in a model**: they have no encoder, so they were tested from random blocks only, never benched.
- **float16 products' range**: x is rounded to float16 (largest 65,504); activations beyond it were not tested.
- **Generation numbers between fresh folders**: the decode lanes' choices are noisy on a fresh folder (an earlier round
  measured about 4%).

### Not checked

- Whether the routes' decode steps are recorded and replayed as CUDA graphs as today's are.
- Training, or a backward pass, through stored weights.
- Several threads or requests measuring or running the lanes at once (serving).
- Linux or macOS: every run was on Windows 11.
