# NVIDIA plan

**Done (2026-10-22).**

**Status:** supported on every GPU from compute 5.0 (Maxwell) through the CUDA driver API, with kernels generated as
PTX in C#. Tested on an RTX 5070 Ti and RTX 5050 Laptop GPU (compute 12.0) and an RTX 3060 Laptop GPU (compute 8.6).
This plan covers what the tests on more than one GPU showed, and the gaps that remain.

## 1. Few-row rule tied to the GPU (first: a measured regression in 0.1.5)

**Done in 0.1.6** (point 2 below, for every card-dependent choice, not only this one): the first time a shape needs
a choice, the device times the candidates back to back on the real inputs and keeps the fastest; on the RTX 5050,
"auto" now matches or beats the best forced column. Still to do: an RTX 3060 and an Ada card run of `--bench-gemv`.

**Problem.** 0.1.5 sends 4–8 rows through int8 weights of 32 M values or more (a vocabulary head) to the packed
tensor-core product instead of the GEMV, when `MixedPrecision` uses tensor cores
(`CudaBackend.PrefersPackedMatMul`, `src/Idrak.Gpu/Cuda/CudaBackend.Quantized.cs`). The rule was measured on the
RTX 5070 Ti, and it holds on the RTX 5050 Laptop GPU too; on the RTX 3060 Laptop GPU it makes those rows slower:

| Rows, head 1024 → 151,936 | RTX 5070 Ti (70 SMs, 12.0): GEMV / packed | RTX 5050 Laptop (20 SMs, 12.0): GEMV / packed / chosen | RTX 3060 Laptop (30 SMs, 8.6): GEMV / packed / chosen |
|---|---|---|---|
| 1 | **217** / 280 µs | **680** / 910 / 681 µs | 1,804 / 2,998 / 963 µs |
| 3 | 296 / **265** µs | **858** / 914 / 858 µs | 2,190 / 2,987 / 1,471 µs |
| 4 | 323 / **265** µs | 1,017 / **934** / 961 µs | **1,677** / 2,996 / 2,378 µs |
| 6 | 408 / **266** µs | 1,316 / **928** / 954 µs | **1,881** / 3,493 / 2,444 µs |
| 8 | 514 / **268** µs | 1,656 / **919** / 957 µs | **2,304** / 2,965 / 2,700 µs |

(`--bench-gemv`, "rows: GEMV / packed / chosen"; bold is the faster path; the RTX 5050 column is from a run on an idle
machine. The 3060's "chosen" column also shows the
eager timings are noisy there: one run picked the GEMV at 1–3 rows and still timed below the GEMV column.)

**What decides it: the GPU generation, not its size.** The RTX 5050 has fewer SMs than the RTX 3060 (20 against 30)
and still gains from the packed product at 4–8 rows; the 3060 loses. Both Blackwell cards (compute 12.0) gain, the
Ampere card (8.6) does not: newer tensor cores do more work per byte of weights read, so the packed product pays off
at fewer rows. SM count would choose wrong (the 5050 would lose a 1.7× gain at 8 rows).

**Plan.** Tie the switch to the compute capability instead of applying it on every GPU:

1. Rule: prefer the packed product for 4–8 rows only when **compute ≥ 8.9** (Ada, Hopper, Blackwell); on older GPUs
   keep the GEMV up to 8 rows. The weight-size condition (k·n ≥ 32 M) and the tensor-core condition stay as they are.
2. Better long-term: decide once per process by timing both paths on the first large few-row product (a few
   microseconds of warm-up, as the fine-tuning step already does for checkpointing settings), and cache the choice per
   shape class. This also covers GPUs no one has measured (Ada laptops, Hopper).
3. **Check the new rule against all three benchmark runs** (RTX 5070 Ti, RTX 5050 Laptop, RTX 3060 Laptop): for every
   row of the "rows: GEMV / packed / chosen" table, "chosen" must be within 5% of the faster of the two.
4. Run the full tests on the three GPUs; update the CHANGELOG with their numbers.

**Done when** the table above shows "chosen" at the faster path on all three GPUs, and the tests pass on all three.
An Ada (compute 8.9) card is the open question: the rule includes it on the strength of its newer tensor cores, not a
measurement; one `--bench-gemv` run on an RTX 40 card confirms or moves the line.

**Also check with the same method:** the 9–63-row packed path on the 3060 (2,496 µs at 9 rows against 3,266 µs for the
8-row GEMV, so it still helps there), the k-split heuristics (`PromptSplits`, `GemvSplits`) on 30 SMs (on the 3060
"auto" is up to 30% off the best forced split: the gate/up decoding product takes 84 µs on "auto" against 65 µs with 2
splits), and the decoding-attention split rule.

## Risks

- **A rule tuned on two GPUs can still be wrong on a third**; the timed choice (step 1.3) avoids guessing, at the cost
  of a short warm-up.
- **Laptop GPUs change clocks with power and temperature**; benchmark on mains power, after a warm-up run.

## Device round, 2026-10-26 (closed)

The owner's runs on an RTX 5050 Laptop GPU (20 SMs, 12.0) and an RTX 3060 Laptop GPU (30 SMs, 8.6). What it built;
what is left is in [leftovers.md](../leftovers.md#plan-1-nvidia-1-nvidiamd).

| Commit | What | Measured |
|---|---|---|
| c95321a | int8, int4 and bfloat16 few-row products compiled for 1, 2 and 4 rows (`{int8,int4,bf16}_gemv_r{1,2,4}_f32`); m ≤ 4 plain products take the smallest; the 217 existing PTX kernels byte-identical; bit-for-bit test against the eight-row kernel | RTX 5050, 1×1024→32000: int8 186 → 62 µs, int4 166 → 61-66 µs, bfloat16 315 → 251-259 µs; small shapes 26-54 → 14-19 µs; CUDA now ahead of Vulkan on the same card on every shape (1.6-4×). RTX 3060: 17-19 µs small shapes, int4 at 32000 99.9 µs |
| 7264ad1 | Split partial sums in one kept block (`KeptScratch`), the pool's size index without nodes (`PoolSizes`): a warm few-row product allocates nothing | 0 B a call on cuda:0 (bench Detail column) |
| 076ce7d, 42aedca | The bench's settled warm-up and median, Cold apart, and the compile time inside each Cold | Cold first call 14-19 ms, 10-13 ms of it compiling C#; later rows 0.04-0.9 ms once the split choices are stored |

Cold rows of 70-310 ms are the split choices measured again: the tuning file's identity carries the library's commit and
the power source, so every pull (and a switch between battery and mains) measures them once more. On the RTX 5050 the
32000-column rows read 62 or 88 µs from one run to the next, either format; the weights stay partly in L2 between the
bench's repeats (int8 at 32000 reads 450-529 GB/s against a measured 257 GB/s).
