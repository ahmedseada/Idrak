# The proof: what it followed, how it was done, what it proved

> **Status: under correction.** A blind review (2026-11-15) found errors this record still has to fix: the CUDA tuning
> keys of the stored-weight lanes (`2000`, `3000`, `4000`, `5000`, `9700` + an id given in first-use order) were the
> same for every type in a one-type bench process, so in a shared tuning folder a later type reused an earlier type's
> lane choices; the CPU's lane choices are in-process only (`StoredLanes`, `s_lanes`) and are not kept by
> `IDRAK_TUNING_CACHE`, so the CPU's fill and warm runs are two independent measurements (q8_0's 39.3 tok/s run chose
> the row lane on every decode shape: a measurement that picked a slower lane, not noise); `results-dec-warm.txt` was
> made by a build before commit `cf77ceeb`; the CUDA bf16 head chose the named kernel at decode too. Until this banner
> is removed, the tables below are not an acceptance bar.

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
  as ggml's decoders"). Encoders exist for Q4_0, IQ4_NL, Q4_K, Q6_K, IQ2_XXS, IQ3_XXS (and q8_0, int8, bfloat16, float16);
  Q2_K, Q3_K, Q5_K have none.
- **Weights kept as the file stores them** under the generic, measured and adad routes: `StoredWeight`, a row of an
  output's values (int8: codes and a scale an output; bfloat16, float16, every described type as stored), packed from the
  file's rows without a float32 copy of the matrix (`StoredPacking.cs`, `IWeightSource.ReadOutputRows`, `WeightRows`).
- **The product from the description, the lane by measurement**: each device generates its kernels from the description;
  the lane for each product shape is chosen by measuring the candidates on the real operands (CUDA: `Tune`; the CPU: a
  port of Adad's `Measure.Faster`, `src/Idrak.Abstraction/Devices/Faster.cs`), once per shape, kept in the tuning cache.

## The routes, exactly

- `name`: today's kernels, chosen by the format's name (Idrak's int8, int4 and bfloat16 packed formats).
- `generic`: one product per device from the description; no named kernel; no measurement.
- `measured`: per shape, the faster of the lanes. For weights still in Idrak's int8 or bfloat16 packed formats, the
  format's own (named) kernel is one of the candidates: on CUDA `RoutedNamed` (`CudaBackend.Typed.cs`), on the CPU
  `NamedLane` and `MeasureLane` (`CpuBackend.Typed.cs`: the named kernel kept unless the generic product is clearly
  faster, by 3%).
- `adad`: `measured` plus Adad's block route among the lanes.

## What was built

### CPU (`src/Idrak.Abstraction/Devices/Cpu/CpuBackend.Lanes.cs`, `.Blocks.cs`, `.Typed.cs`, `Faster.cs`)

- Row lane (a row of stored values a step), panel lane (transposed B), "today's path" lane (the weight widened once from
  its description into the tiled product's panels, then the CPU's float32 product: `CpuMatMul.MultiplyPanels`, bit for bit
  the same sums as over B [k, n], tested), tiles lane (panels widened inside the product: `MultiplyPanelsFrom`, tested bit
  for bit), expanded lane (measured only above 8 rows; its buffer rented).
- Adad's block route (`adad` only): x quantized into the weights' 8-bit blocks, integer dots with AVX-VNNI-INT8; for the
  k-quants a fused SIMD unpack and dot (`Codes32`); for the grids their block route (`GemvGrid`, `UnpackGridRow`).
- The lane per shape measured with `Faster.Time`.

### CUDA (`src/Idrak.Gpu/Cuda/CudaBackend.Typed.cs`, `.Many.cs`, `.Super.cs`, `.NativeRaw.cs`, `.NativeDirect.cs`, `.TensorRows.cs`; generators `PtxKernels.Typed.cs`, `.Blocks.cs`, `.Super.cs`, `.NativeMma.cs`, `.Epilogues.cs`, and modes added to `PtxKernels.TensorCore.cs`)

The candidates of `StoredMeasured`, by number:

- 0: the f32 tile (more rows) or the few-row lane (warp a column, 16-byte loads), generated from the description.
- 1: the simple matrix-unit kernel (`TypedMma`).
- 2: the native kind: Adad's GemmMma ported, its shape measured on the device (9 shapes), x and W packed into the kind's
  rows; the kinds as a table (`NativeKind`: bfloat16, float16).
- 3: Adad's block route (`adad` only): dp4a few rows, s8 `mma` tiles.
- 4: the direct native lane (x staged from float32, W decoded per chunk in the kernel).
- 5: the raw int8 lane (int8 codes read as stored, widened in registers).
- 6: the tensor rows (`CudaBackend.TensorRows.cs`): Idrak's tensor-core GEMM generator (`TensorCoreGemm`) given modes that
  read the stored rows (`gemm_tc_nt_*`): bfloat16 and float16 rows copied as stored (`bf16B`, `f16B`: 16 bytes a thread),
  int8 codes widened with a scale an output (`int8B`), 32-value scaled blocks decoded as each tile loads (`blockB`, a
  module per description), super-blocks and grids decoded as each tile loads (`decodeB`, with the description's decoder)
  or their rows widened to bfloat16 first; 128- and 64-row tiles, the tile and the k splits measured per shape; k split in
  the kernel with an ordered sum.

Super-block and grid weights take their own lanes (`StoredSuper`): 0 the few-row kernel or the f32 product, 1 native,
2 direct, 3 tensor rows decoding, 4 tensor rows over widened rows.

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
   device (`IDRAK_TUNING_CACHE`). From round 11: on a fresh folder a fill run, then a warm run, and the warm run is the
   result (earlier rounds read one run on a shared folder; round 5 used one folder per route, later found wrong).
6. **A number out of line is traced before it counts.** Examples: the fused residual-norm conversion (generation 643 →
   343); q6_k and iq3_xxs slower after decoding in the kernel (warm timing hid the decoders' scattered reads).
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
  as stored; it also keeps a type at its file size on the device. Measured: device memory for q8_0 2,590 → 921 MiB, q4_0
  2,590 → 637 MiB (`test-output\routes*\`).
- **int8 kept with today's codes and a scale an output.** The reason was not written down at the time. Effect: the
  routes multiply the same quantized values as `name`; only the layout and the arithmetic differ.
- **Results judged within float32's bound, not bit for bit against `name`.** Why: the lanes sum in another order than
  today's kernels; the bound used is `k · 2^-23 · Σ|terms|` against a double reference, with x (and the weights, where a
  lane rounds them) rounded as the lane rounds them. Bit for bit only where the arithmetic is the same by construction
  (panels against the tiled product, merged attention against merging after it, the bfloat16 cache's tensor-core pass
  against the float32 cache's over the same values, a checkpoint packed from rows against the values path).

### How a lane is chosen

- **Measured at run time per product shape, on the real operands**, once, kept in the tuning cache (CUDA `Tune`; the CPU
  `Faster.Time`, Adad's `Measure.Faster` ported). `Tune`'s fallback is lane 0 (the f32 tile or the few-row lane): another
  lane is taken only where it measures faster.
- **A new lane is added as a candidate, never replacing one.** Why: a lane that loses on one shape or card may win on
  another; the measurement decides. So the proof kept lanes that lost (the direct lane, the native kind once the tensor
  rows existed) as candidates.
- **Under `measured`, the named kernels of Idrak's packed int8 and bfloat16 formats were candidates** (CUDA `RoutedNamed`,
  CPU `NamedLane`/`MeasureLane`), as the route's definition says ("the faster of the generic product and the format's own
  kernel"). This is one of the places the proof used `name`; see below.
- **Fused epilogues measured against the composition, the composition the fallback.** Measured: on this card the fused
  passes were slower (a barrier, an atomic and a serial tail against cheap launches), so the composition ran.
- **Cold timing only for the super-block and grid decoders.** Measured both ways: cold timing everywhere cost bf16, int8,
  q4_0 about 6%; cold for the gathering decoders only restored q6_k and iq3_xxs (warm repeats served their scattered byte
  reads from the cache, while a pass reads them from device memory).
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
  same. The cost seen later: its CPU decode step chose the named kernel (4.6 ms; the generic lane over the named layout
  took 34 ms).

### How the benches were run

- **One model, one machine:** Qwen/Qwen3-0.6B; an RTX 5070 Ti (70 SMs, compute 12.0) and its CPU (24 threads, AVX-VNNI-INT8).
- **CUDA** `--prompt-tokens 512 --tokens 128 -n 3`; **CPU** `--prompt-tokens 128 --tokens 32 -n 2`.
- **Both routes share one tuning folder per device.** Why: an earlier CPU "win" came from a stale cache holding `name`
  back; per-route folders also changed the CPU's cut-overs for every operation (attention 2× between folders).
- **A fresh folder: a fill run, then a warm run; the warm run is the result.** Why: the fill run measures while it is
  timed; fresh folders differ by about 3–5% on CUDA prompts.
- **A number out of line is traced or re-run before it counts** (the residual-norm regression; q6_k and iq3_xxs; q8_0's
  CPU generation re-run twice).

## What it proved

### CUDA, Qwen3-0.6B, 512-token prompt and 128 generated tokens, tok/s (`results-dec-warm.txt`; q6_k and iq3_xxs after cold timing: `results-gather.txt`)

| Type | adad prompt | name prompt | adad gen | name gen | adad peak MiB | name peak MiB |
|---|---|---|---|---|---|---|
| bf16 | 48,016 | 46,191 | 438 | 403 | 1,282 | 1,282 |
| f16 | 47,756 | 35,503 | 433 | 258 | 1,578 | 2,715 |
| int8 | 49,136 | 48,530 | 626 | 444 | 1,012 | 1,012 |
| q8_0 | 43,826 | 36,438 | 526 | 258 | 1,046 | 2,715 |
| q4_0 | 45,660 | 35,904 | 553 | 258 | 761 | 2,715 |
| iq4_nl | 43,986 | 36,386 | 524 | 258 | 761 | 2,715 |
| q4_k | 42,871 | 36,295 | 494 | 258 | 761 | 2,715 |
| q6_k | 41,753 | 36,383 | 509 | 258 | 908 | 2,715 |
| iq2_xxs | 42,184 | 35,877 | 478 | 258 | 594 | 2,715 |
| iq3_xxs | 40,064 | 36,247 | 448 | 258 | 665 | 2,715 |

At the start of this session (`results-before-warm.txt`): bf16 adad 17,368 / name 18,056 prompt; int8 16,713 / 18,335;
f16 17,309 / 16,200; q4_0 16,742 / 16,198; adad generation bf16 365, int8 343 (the residual-norm regression).

### CPU, 128-token prompt and 32 generated tokens, tok/s (`results-cpu-warm.txt`, `results-cpu-fill.txt`; noise about 10%)

| Type | adad prompt (warm / fill) | name prompt (warm / fill) | adad gen | name gen |
|---|---|---|---|---|
| bf16 | 476 / 393 | 305 / 282 | 45.5 | 33.2 |
| int8 | 437 / 525 | 357 / 332 | 71.8 | 49.6 |
| q8_0 | 544 / 525 | 155 / 169 | 62.8–64.9 | 8.97 |
| q4_0 | 492 / 547 | 160 / 130 | 38.4 | 8.19 |
| q4_k | 445 / 536 | 159 / 157 | 28.8 | 8.54 |

(q8_0's warm generation read 39.3 once; two re-runs gave 63.5 and 64.9: noise.)

### Which lanes the final benches chose (the "measured lane" lines of `dec-*-cuda-*-adad.txt`, `cpu-*-cpu-*-adad.txt`, and the last profiles)

- CUDA prompts: the tensor rows for every type (bf16, f16, int8 rows as stored; q8_0, q4_0, iq4_nl blocks decoded; the
  k-quants and grids decoded or widened, as measured), the group launches, the down projection included.
- CUDA decode: the few-row lanes, grouped (`_many`), with the fused gated and residual-norm passes where measured faster
  ("few-row lane (block route slower)" on every block type).
- CPU: the tiles lane, Adad's block route, the row lane, the "today's path" lane.
- Never chosen in the final benches: the native kind (lane 2), the raw int8 lane (5), the direct lane (4), `TypedMma` (1),
  the CUDA block route (3), the CPU panel lane. Earlier: the native kind, `TypedMma` and the raw int8 lane were chosen for
  some shapes before the tensor rows existed; the direct lane lost every shape it was measured on (round 7); the CUDA block
  route reached 31–32 TFLOPS without bfloat16 products in round 2, while the native kind won under them; the panel lane's
  earlier choices were not recorded.

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
  about 3–5% on CUDA prompts: the final build gave bf16 46,462 on a fresh folder and 47,390 on the previous round's folder,
  whose choices had given 48,016 with the previous build (the two builds measure bf16 the same way; they differ only in
  timing the gathering decoders cold).
- `Tune` does not measure while profiling or during graph capture (`CanMeasure`); a profile on an unfilled folder shows
  fallback choices.
- Warm timings fit lanes that read contiguous rows; the super-block and grid decoders, which gather scattered bytes, need
  cold timings (measured both ways: cold everywhere cost bf16, int8, q4_0 about 6%; cold for the gathering decoders only
  fixed q6_k 38,952 → 41,753 and iq3_xxs 38,317 → 40,064).
- CPU benches move about 10% run to run (the unchanged `name` route moved as much).

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
