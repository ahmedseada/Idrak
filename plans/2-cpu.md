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
| 1 | bfloat16 products of 4–8 rows are compute-bound (8 rows: 42 ms on 4 cores, 5.1 ms on 24 threads); register-tile the rows (several rows' sums kept in registers per weight vector) | `CpuBackend.Quantized.cs`, `BFloat16MatMul` | ~2× for 4–8 rows |
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

## Risks

- **Bit-exact statistics:** callers that compared sampling statistics across versions see entropy change in the last
  bits (tokens unchanged).
- **Benchmark noise:** laptops and shared machines vary by 20–60% between runs; compare medians of three runs.
