# CPU plan

**Status:** supported everywhere .NET 10 runs: `Vector<T>` SIMD kernels (AVX2, AVX-512, NEON), a register-blocked
matrix product, multi-threading for large tensors. Tested on Windows 11 with 4, 8, 16, 20 and 24 threads and on Ubuntu
(4 threads); every machine passes the full CPU test list. The CPU is also the fallback for every device without a GPU
backend (AMD, Intel, Apple), so its speed matters to most users. This plan covers what the benchmarks on those machines
showed.

## 1. Plain sampling: entropy without a logarithm per token (first)

**Problem.** Sampling without top-k or top-p keeps all 151,936 tokens of a Qwen-sized vocabulary in play, and the
sampler (`SampleRows` in `src/Idrak/Backends/Cpu/CpuBackend.Decoding.cs`) then computes an `Exp` and a `Log2` per token:
the `Exp` for the weights (needed), the `Log2` only for the entropy statistic. On the 8-thread CPU-only Windows machine
this is the slowest sampling mode by far, and it repeats across runs:

| `--bench-cpu`, 151,936 tokens | 8-thread machine, run 1 / run 2 | other machines |
|---|---|---|
| plain | **13.54 / 12.44 ms** | 1.9–2.4 ms (20 and 24 threads), 2.1 ms (4-core container) |
| top-k 20 | 0.90 / 0.93 ms | 0.65–1.6 ms |
| top-p 0.95 | 6.40 / 5.56 ms | 3.3–8.8 ms |

It is not a regression: on the 4-core container plain sampling went 3.2 → 2.1 ms with the 0.1.4 rewrite. That CPU is
simply slower at the per-token math calls.

**Plan (decided: option a).** On a new branch, with before-and-after `--bench-cpu` numbers:

1. Compute the entropy from the scores already in hand instead of a `Log2` per token. With `w = exp(s − max)` and
   `p = w / sum`: `log2 p = (s − max) · log2(e) − log2(sum)`, so
   `H = log2(sum) − log2(e) · Σ p · (s − max)`, one multiply-add per token and one `Log2` per row.
2. The chosen tokens and their probabilities stay bit-identical. The entropy changes in its last bits, so the
   sampler test that compares statistics with the reference exactly keeps exact equality for tokens and
   probabilities and allows a relative tolerance (about 1e-5) for the entropy only.
3. Vectorize the remaining weight pass (`Exp` over all tokens) with `CpuMath.ExpShifted`, as the attention kernels do,
   if it keeps the chosen tokens identical; otherwise leave it.
4. Run `--bench-cpu` before and after on the 8-thread machine, the 20- and 24-thread machines and the container, and
   put the numbers in the CHANGELOG.

**Done when** plain sampling on the 8-thread machine drops to about half (estimate: ~12 → ~6 ms) with every test
passing and the same tokens as before.

## 2. Remaining CPU speed work

| # | Item | Where | Expected gain |
|---|---|---|---|
| 1 | ✅ (section 4) bfloat16 products of 4–8 rows keep up to 8 rows' sums in registers over k chunks of half a cache line (`CpuBackend.BFloat16.cs`); identical results | `CpuBackend.BFloat16.cs` | measured ~1.7–2× for 4 and 8 rows on the container |
| 2 | Products with a transposed B of 17–63 rows still transpose all of B per call (only ≤ 16 rows read it in place) | `CpuMatMul.cs`, `NtRows` | 3× at 64 rows (measured 0.1.4) |
| 3 | `FusedAdamW` is scalar and single-threaded | `CpuBackend.cs` | training steps on the CPU |
| 4 | RMSNorm, rotary embedding, gated activation, add+norm are scalar | `CpuBackend.Quantized.cs` | per-layer ops of CPU inference |
| 5 | Language-model loss: double-precision `Exp`, scalar, rows with weight 0 still computed | `CpuBackend.Decoding.cs`, `SoftmaxCrossEntropyRows` | fine-tuning on the CPU |
| 6 | Memory pool: exact-size buckets, never trimmed without a limit, one lock | `CpuBackend.cs` | memory with varying shapes |

## 3. Coverage

| Gap | Plan |
|---|---|
| ARM64 (NEON): Apple silicon, Snapdragon X, Ampere servers | one test and `--bench-cpu` run on an ARM64 machine |
| AVX-512 machines | one run on a recent Intel Xeon or AMD Zen 4/5 desktop, to check the 16-wide paths |
| macOS | test run (CPU only) |

## 4. Machine-agnostic audit

The rule of [the plans' README](README.md#rule-no-card-specific-tuning-mostly-done-in-016) holds for the CPU too: no
choice is tuned to one processor, vendor or machine. Every tiling, blocking and threading value comes from what the
machine reports (`CpuInfo`: logical and physical cores, `Vector<float>.Count`, the instruction sets the runtime
accelerates and the vector registers they imply, L1d / L2 / L3 and line sizes from `/sys/devices/system/cpu` on Linux,
`GetLogicalProcessorInformationEx` on Windows, `sysctl hw.*` / `hw.perflevel0.*` on macOS, with 32 KiB / 1 MiB / 64 B
fallbacks), as fractions of the caches or multiples of the vector width and thread count, or is measured on the machine
(`CpuTuning`), kept per machine (the reported model, instruction sets, cores and cache sizes), .NET version and thread
count in `IDRAK_CACHE`/cpu/tuning.tsv (default ~/.cache/idrak/cpu/tuning.tsv; `IDRAK_TUNING_CACHE=0` keeps nothing,
`IDRAK_AUTOTUNE=0` uses the formulas). Physical cores are reported but decide nothing: the measured cut-overs already include what SMT does.

The values that decide a summation order are not tuning: they define what the CPU computes, and the CPU is the reference
the other backends are checked against, so they are the same on every machine (the "numerical contract" in
`CpuTuning`). Old and new builds give bit-identical results on a dump of 187 results (products of 1–512 rows in every
layout with beta 0 and 0.75, element-wise kernels and sums of 1,000–2,000,003 values, softmax, bfloat16 / int8 / int4
products of 1–20 rows, decoding attention), with the default settings, with the cut-overs forced to 1 and to "never",
and with the runtime limited to AVX2 (`DOTNET_EnableAVX512F=0`), to 4-wide vectors (`DOTNET_EnableAVX2=0`), to no
SIMD (`DOTNET_EnableHWIntrinsic=0`) and with 512-bit vectors accelerated (`DOTNET_PreferredVectorBitWidth=512`, which
runs the AVX-512 kernel and the 512-bit bfloat16 panels). The tests (`CpuTuningTests.cs`) check every kernel, register
row count, k chunk, cut-over extreme and the streaming loop against the fallback formulas bit for bit.

Columns "16 thr" and "24 thr" are what the formulas give on a 16-thread AVX2 machine with 32 KiB L1d and 512 KiB L2
per core, and on a 24-thread hybrid AVX2 machine whose performance cores have 48 KiB L1d and 2 MiB L2 (the largest
per-core values are used). "Here" is the 4-thread test container. It moved between hosts during the work, which
showed the values following the machine: first AVX-512F reported but 512-bit vectors not accelerated by the runtime
(AVX2 kernel, 32 KiB L1d, 1 MiB L2, 33 MiB L3: A tile 256 KiB, 16 row blocks, 8 register rows, no 512-bit panels), then
512-bit vectors accelerated (AVX-512 8 × 32 kernel, 48 KiB L1d, 2 MiB L2, 260 MiB L3: A tile 512 KiB, at most 24 row
blocks, 8 register rows, 512-bit bfloat16 panels, formula cut-overs 131,072 / 262,144). Where the two differ, the
"Here" column shows the second host.

| Choice (where) | Before | Now | Kind | Here | 16 thr | 24 thr |
|---|---|---|---|---|---|---|
| Element-wise parallel cut-over (`Run`, `For`, `HostParallel` grains, `CpuTuning.ParallelElements`) | 65,536 | measured at first use: an axpy over a window that moves through buffers several times the L2 (the same window again and again sits in one core's L2 and favoured the sequential run: 262,144 measured where real element-wise work splits well from 65,536–131,072), after an untimed pass (thread pool start-up and unoptimized code had set it 4× too high), parallel at least 10% faster, the median of three passes (single passes ranged 32,768–524,288 on the shared container; a too-early cut-over, 32,768, made the work around it 2× slower, a too-late one, 262,144 and up, left 131,072-element work 1.4–1.6× slower); passes more than 16× apart give the formula, not kept; fallback L2 / 16 | measured | 65,536–131,072 on the first host (formula 65,536); 262,144–524,288 on the second with the earlier method | fallback 32,768 | fallback 131,072 |
| Product parallel cut-over in m·n·k (`CpuMatMul`, `ParallelFlops`) | 131,072 | measured at first use on one row against a square B (the column split spreads it evenly over the threads, so the timing shows the cost of starting them; cubes were tried first and kept 1 × 4096 × 1024 on one thread, 6× slower, because their tiled grid has one or two tiles at these sizes); fallback 2 × the element-wise | measured | 524,176–1,048,576 (first host) | fallback 65,536 | fallback 262,144 |
| Chunks of `Run` / `For` | ≤ 2 × threads, ≥ cut-over / 4 / 4 × threads | unchanged | relative | 8 / 16 | 32 / 64 | 48 / 96 |
| Tiled product kernel (`CpuTuning.Kernel`) | AVX-512 8×32 or AVX2 6×16 (by vector width); none on ARM64 | by reported instruction sets: AVX-512 8×32, AVX2+FMA 6×16, NEON 8×8 (new) | ISA | 8×32 | 6×16 | 6×16 |
| Tiled from rows | 6 | the kernel's rows (6 / 8 / 8) | ISA | 8 | 6 | 6 |
| A tile (`TileBytes`) | 256 KB | L2 / 4 | relative | 512 KiB | 128 KiB | 512 KiB |
| Tile grid (`TileGrid`, `MaxRowBlocks`) | rows: A tile, at most 16 row blocks; columns: ~2 tiles per thread | row tiles: A tile, at most L1 / 2 KiB row blocks (C rows of a panel stay in L1); columns: ≥ 2 tiles per thread, split further (up to 2×) so the tile count is a multiple of the threads (whole rounds: 1024³ ~9% faster here, timed in one process against the former grid). One tile per thread as large as the caches allow was 15–30% slower at 512³–1024³ on the shared container; sharing the rows evenly among the row tiles gained nothing (2–4% slower at 128³–512³) and was dropped | relative | 512³: 3 × 4 tiles of 192 × 128 | 512³: 9 × 4 tiles of 60 × 128 | 512³: 4 × 11 tiles of 144 × 48 |
| Column block: few-row chunk floor, int8 / int4 / bfloat16 block floor, transposed-B chunk (`ColumnBlock`) | 64 | 8 vectors | relative | 64 | 64 | 64 (NEON: 32) |
| Transpose tile (`TransposeSide`; products and `HostParallel.Transpose`) | 32 and 64 | 2 · side² floats ≤ L1 / 2, a power of two | relative | 32 | 32 | 32 |
| bfloat16 few-row rows in registers (`RegisterRows`) | – (sums in memory) | 8 with 32 vector registers, 4 with 16 | ISA | 8 | 4 | 4 |
| bfloat16 few-row k chunk (`KChunk`) | – | half a cache line of floats | relative | 8 | 8 | 8 |
| bfloat16 512-bit panels (`WidePanels`) | – | where 512-bit vectors are accelerated and `Vector<float>` is narrower | ISA | yes | no | no |
| Portable register block `Mr` | 4 | 4: 2 × 4 accumulators + 3 fit the 16 registers every SIMD set has | fixed: ISA fact | 4 | 4 | 4 |
| `HostParallel` threads | `ProcessorCount` × 4, ignoring `MaxCpuThreads` | `MaxCpuThreads` × 4 | relative (fix) | 16 | 64 | 96 |
| Load-time quantization grains (`Quantization.cs`) | 65,536 / 4,096 × 32 elements | the element-wise cut-over (× 2 for int4 groups) | measured | 65,536 | – | – |
| Attention head sizes (`Capabilities`, `DecodeAttentionHeads` / `TiledAttentionHeads`) | CUDA's `DecodeMaxDim` 256 / `FlashMaxDim` 128 | the CPU's own constants, same values (larger heads take the [t, t] weights, a different summation order) | fixed: numerical contract | 256 / 128 | 256 / 128 | 256 / 128 |
| Reduction chunks (Σx, Σx², chunk-summed `For` bodies) | sequential below 65,536, chunks ≥ 16,384 | unchanged; the chunks run on threads only past the measured cut-over | fixed: numerical contract | 65,536 | 65,536 | 65,536 |
| Transposed B read in place up to (`TransposedInPlace`) | 16 rows | unchanged (dot products in vector lanes vs one FMA chain) | fixed: numerical contract | 16 | 16 | 16 |
| Few-row split of packed products, `Capabilities.FewRows` | 8 (CUDA's `GemvRows`) | 8, the CPU's own (int4 sums without FMA below, FMA above) | fixed: numerical contract | 8 | 8 | 8 |
| Column-split product when beta ≠ 0 | ≤ 4 rows, ≥ 128 columns, ≥ 131,072, > 1 thread | unchanged (beta · C added without FMA there); beta = 0 follows the measured cut-over | fixed: numerical contract | – | – | – |

Fixed, with the reason: the 8-bit AdamW block (a stored format), the int4 group of 32 rows (format), the sampler's 24
bisection steps (float's 24-bit mantissa), its `max − 40` floor (e⁻⁴⁰ is below float resolution next to the largest
weight) and 5 statistics (output format), the `stackalloc` limit of 4,096 floats in `SumAxis` (thread stack safety, not
speed), the GELU and hash constants (mathematics), the 16-lane test in `CpuMath` (the lanes of a 512-bit vector), the
per-element cost weights passed to `For` (operation counts, relative), and the measurement method (7 rounds in turns,
medians, 10% margin, the median of three passes after an untimed one, sizes from 1/32 to 32 times the formula's;
about two seconds once per machine, runtime and thread count).

AVX-512 BF16 and AVX-512 VNNI have no `IsSupported` flag in .NET 10 (only `AvxVnni`, `AvxVnniInt8`, `Avx10v1/2`, which
are reported and part of the cache key) and SVE is experimental, so no decision reads them.

ISA paths not added (each would change a summation order or rounding, so it would have to be opt-in, or needs a machine
to test on): transposed-B dot products in 512-bit lanes; AVX-512 BF16 / ARM BF16 dot products (`VDPBF16PS`, `BFDOT`,
pairs summed before accumulating); AVX-VNNI / AVX-512 VNNI / ARM `SDOT` int8 dot products (need int8 activations); SVE
(experimental in .NET 10); AMX (no .NET API); element-wise kernels at 512 bits where `Vector<T>` is 256 (the
transcendental ones may round differently per width; the linear ones would be identical); register-tiling the int8,
int4 and float few-row products like the bfloat16 one (identical results, not done yet).

**Open:** the bfloat16 k chunk is a formula (8 was fastest of 2–16 here; 16 was twice as slow): measure it at first
use if other machines disagree. Runs on an ARM64 machine (the NEON kernel's speed) and on a machine where the runtime
accelerates AVX-512 (the 512-bit panels).

## Risks

- **Bit-exact statistics:** callers that compared sampling statistics across versions see entropy change in the last
  bits (tokens unchanged).
- **Benchmark noise:** laptops and shared machines vary by 20–60% between runs; compare medians of three runs.
