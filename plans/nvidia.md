# NVIDIA plan

**Status:** supported on every GPU from compute 5.0 (Maxwell) through the CUDA driver API, with kernels generated as
PTX in C#. Tested on an RTX 5070 Ti and RTX 5050 Laptop GPU (compute 12.0) and an RTX 3060 Laptop GPU (compute 8.6).
This plan covers what the tests on more than one GPU showed, and the gaps that remain.

## 1. Few-row rule tied to the GPU (first: a measured regression in 0.1.5)

**Problem.** 0.1.5 sends 4–8 rows through int8 weights of 32 M values or more (a vocabulary head) to the packed
tensor-core product instead of the GEMV, when `MixedPrecision` uses tensor cores
(`CudaBackend.PrefersPackedMatMul`, `src/Idrak/Backends/Cuda/CudaBackend.Quantized.cs`). The rule was measured on the
RTX 5070 Ti only. On the RTX 3060 Laptop GPU it makes those rows slower:

| Rows, head 1024 → 151,936 | RTX 5070 Ti: GEMV / packed | RTX 3060 Laptop: GEMV / packed / chosen |
|---|---|---|
| 1 | 217 / 280 µs | 1,804 / 2,998 / 963 µs |
| 3 | 296 / 265 µs | 2,190 / 2,987 / 1,471 µs |
| 4 | 323 / **265** µs | **1,677** / 2,996 / 2,378 µs |
| 6 | 408 / **266** µs | **1,881** / 3,493 / 2,444 µs |
| 8 | 514 / **268** µs | **2,304** / 2,965 / 2,700 µs |

(`--bench-gemv`, "rows: GEMV / packed / chosen"; bold is the faster path. The 3060's "chosen" column also shows the
eager timings are noisy there: one run picked the GEMV at 1–3 rows and still timed below the GEMV column.)

**Plan.** Tie the switch to the GPU instead of a fixed 4 rows:

1. Read the deciding numbers the backend already has: SM count (`_multiprocessors`), compute capability, and the
   memory bandwidth (from the driver's memory clock and bus width attributes).
2. Candidate rule: prefer the packed product for few rows only when the GPU's tensor-core throughput per byte of
   memory bandwidth is high, i.e. compute ≥ 8.9 (Ada and newer) or SM count ≥ 40; otherwise keep the GEMV up to 8 rows.
3. Better long-term: decide once per process by timing both paths on the first large few-row product (a few
   microseconds of warm-up, as the fine-tuning step already does for checkpointing settings), and cache the choice per
   shape class.
4. **Check the new rule against both benchmark runs** (RTX 5070 Ti and RTX 3060 Laptop): for every row of the
   "rows: GEMV / packed / chosen" table, "chosen" must be within 5% of the faster of the two.
5. Run the full tests on both GPUs; update the CHANGELOG with both GPUs' numbers.

**Done when** the table above shows "chosen" at the faster path on both GPUs, and the tests pass on both.

**Also check with the same method:** the 9–63-row packed path on the 3060 (2,496 µs at 9 rows against 3,266 µs for the
8-row GEMV, so it still helps there), the k-split heuristics (`PromptSplits`, `GemvSplits`) on 30 SMs (on the 3060
"auto" is up to 30% off the best forced split: the gate/up decoding product takes 84 µs on "auto" against 65 µs with 2
splits), and the decoding-attention split rule.

## 2. Remaining performance work (from docs/performance-notes.md and the 0.1.4 audit)

| # | Item | Expected gain | Notes |
|---|---|---|---|
| 1 | Delayed FP8 scaling keyed by call order, not address | ~18 ms per 1.25B training step | performance-notes step 1 |
| 2 | FP8 quantization fused into producers (layer norm, GELU, product epilogues) | up to ~100 ms per 1.25B step | needs item 1 |
| 3 | Persistent decoding kernel (several layers' GEMVs in one launch) | decoding is at ~31% of the bandwidth ceiling | performance-notes step 3 |
| 4 | Int8 prompt products on int8 tensor cores | prompt pass | needs a second, k-major copy of int8 weights |
| 5 | Cached compiled kernels (cubin per driver and GPU) | ~70 ms off the first prompt pass | performance-notes step 8 |
| 6 | Memory pool size classes and partial trimming | fewer full flushes near a memory limit | audit, CUDA #4–5 |
| 7 | Fine-tuning loss read once per optimizer step | no host stall per micro-batch | audit, CUDA #6 |
| 8 | QLoRA backward without expanding int4/int8 weights | 4–8× less weight traffic per step | needs packed transposed kernels |

## 3. Coverage gaps

| Gap | Plan |
|---|---|
| Compute 5.0–7.5 (Maxwell, Pascal, Turing) never run | the kernels assemble for sm_50/61/75; borrow or rent a GTX 10 / RTX 20 card for one test and benchmark run |
| Linux with a GPU never run | one run on a Linux machine (`libcuda.so.1`); a cloud GPU is enough |
| CI does not run on a GPU | a self-hosted runner, or a manual "GPU check" step in the release checklist |
| FP16 tensor cores (Volta/Turing have no bfloat16) | an FP16 `MatMulPrecision` for compute 7.0–7.5 |

## Risks

- **A rule tuned on two GPUs can still be wrong on a third**; the timed choice (step 1.3) avoids guessing, at the cost
  of a short warm-up.
- **Laptop GPUs change clocks with power and temperature**; benchmark on mains power, after a warm-up run.
