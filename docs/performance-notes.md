# Performance notes: where the optimization work stopped

Measured on an RTX 5070 Ti (70 SMs, 16 GB). Resume from here when returning to optimization.

## Checks done (2026-09-28)

1. **Pretrained check against transformers** (Qwen3-0.6B, `qwen3.json` from `tools/pytorch/pretrained_reference.py`):
   - float32 (`check qwen3.json --cuda`): everything matches; max |Δlogit| 4e-5 to 1.4e-4, greedy output identical.
   - int8 weights, bfloat16 KV cache (`--int8 --kv16`): top-1 the same everywhere, greedy identical on 3 of 4 prompts
     (one C# continuation diverges after 12 characters into another plausible one); max |Δlogit| 0.5 to 1.4, as int8
     weights give. Decoding 486–575 tok/s in the check.
   - int8 weights, bfloat16 KV cache, bfloat16 tensor-core prompts (`--int8 --kv16 --matmul bf16`, the path with split-k
     int8 tensor-core products): the same differences as with float32 products (0.53 / 0.53 / 1.42 / 1.05 against
     0.52 / 0.52 / 1.43 / 1.01), same top-1 and greedy output; the 291-token forward pass 147 ms instead of 211.
2. **FP8 against bfloat16 loss curves**, quick version (100.9M-parameter char model: d_model 768, 12 layers, batch 32,
   block 384; 600 steps, same seed and schedule; `--loss-log` and `compare`):

   | | bf16 | fp8 | fp8 + delayed scaling |
   |---|---|---|---|
   | validation loss, step 600 | 2.7425 | 2.7388 (-0.13%) | 2.7250 (-0.64%) |
   | training loss, last fifth | 2.6703 | 2.6693 (-0.03%) | 2.6589 (-0.43%) |
   | largest gap at any validation step | | 0.43% (step 100) | 0.64% |
   | steps/s | 5.64 | 5.75 | 5.76 |

   Both FP8 variants stay within 1% of bfloat16 throughout. Not yet run: the long comparison on the 1.25B model (3000
   steps, about 36 + 28 minutes), advisable once before a long `--fp8` run of that size:

   ```
   $common = "--bin D:\TF.NET\POC\TinyCharTransformer\python\data\final_corpus.bin --vocab-file D:\TF.NET\POC\TinyCharTransformer\python\data\final_corpus.vocab --block 512 --batch 8 --dmodel 2048 --heads 16 --layers 24 --steps 3000 --warmup 300 --eval-every 250 --eval-steps 20 --grad-checkpoint --optim8bit --no-save --log-every 250".Split(' ')
   dotnet run -c Release --project samples/Idrak.Samples.GptTraining -- @common --out checkpoints/cmp-bf16 --loss-log bf16.csv
   dotnet run -c Release --project samples/Idrak.Samples.GptTraining -- @common --out checkpoints/cmp-fp8 --loss-log fp8.csv --fp8
   dotnet run -c Release --project samples/Idrak.Samples.GptTraining -- compare bf16.csv fp8.csv --window 250
   ```

   At this model size FP8 is barely faster than bfloat16 (2%): the quantization passes cost about what the 8-bit
   products save. The gain grows with the model (1.25B: 560 against 712 ms per step).

## Results so far

| | Start | Now |
|---|---|---|
| Qwen3-0.6B int8, bf16 KV cache: greedy decoding | 391 tok/s | 474–493 tok/s |
| Chat sampling (temperature 0.6, top-k 20, top-p 0.95) | 227 tok/s | 249–255 tok/s |
| Kernel launches per decoded token | 345 | 177 |
| 180-token prompt pass | 13.6 ms | 7.7 ms (GPU 7.3 ms; products 5.0 ms at 42.8 TFLOPS) |
| Decoding attention, 4000 cached positions | 76 µs | 38 µs |
| 1.25B char model training step, FP8 | 592 ms | 560 ms (548 with delayed scaling) |
| 1.25B char model training step, bfloat16 | 736 ms | 712 ms |
| Weight-gradient products, small output and long k (768×768×12288 tn) | 29 TFLOPS | 59 TFLOPS |

What changed:

- Decoding: q/k RMS norm, rotary embedding, head layout and KV-cache writes in one kernel (`norm_rope_heads_f32`);
  split decoding attention merged by its last block; packed GEMVs with the residual addition and next RMSNorm
  (`*_gemv_addnorm_f32`) and with the gated activation (`*_gemv_multi_act_f32`); GEMV k splits ≈ 2 blocks per SM
  (power of two); decoding attention ≈ 5 blocks per SM over the cache.
- Prompts: packed tensor-core products split k up to ≈ 4 blocks per SM (none when the tiles fill one wave); layers
  sharing an input (q/k/v, gate/up) in one launch (`gemm_tc_nn_*w_multi_f32`): q/k/v 1.84 → 1.20 ms and gate/up
  2.24 → 1.66 ms per 180-token pass, 367 → 283 launches; 64-row tiles when the last 128-row tile would be at most
  half full (`*_m64_f32`; 180 rows: 192 computed instead of 256): products 5.6 → 5.0 ms, the vocabulary head 0.99 →
  0.79 ms (70.6 TFLOPS).
- Training: plain tensor-core products with fewer than 4 output tiles per SM split k (≈ 16 blocks per SM, ≤ 8 splits,
  ≥ 1024 k each). Delayed FP8 column scaling (kept per-column maxima with 2× headroom, recorded in the same pass,
  correction pass for columns that more than doubled) behind `IDRAK_FP8_DELAYED=1`.
- Tools: `--bench-gemm` (with a split sweep for long-k shapes), `--bench-gemv` (decoding products, decoding attention
  and 180-row products with forced splits), `--bench-fp8` (column quantizers and delayed-scaling error), the
  per-kernel GPU-time tables of the Pretrained `profile` command and CharGpt `--profile` (CUDA events; small kernels
  read about 5 µs high, so compare ratios).

## Against PyTorch (transformers, bfloat16), Qwen3-0.6B

| | PyTorch | Idrak bf16 | Idrak int8 | bf16 speedup | int8 speedup |
|---|---|---|---|---|---|
| Prompt pass, 180 tokens | 21.0 ms (passes 2 and 3) | 9.2 ms (before the prompt work) | 7.7 ms (now) | 2.3× | 2.7× |
| Greedy decoding, 32 tokens | 48.6 tok/s | 372.8 tok/s | 492.6 tok/s (now) | 7.7× | 10.1× |
| Chat sampling, 128 tokens | 49.1 tok/s | 221.6 tok/s | 254.8 tok/s | 4.5× | 5.2× |
| Interactive chat | ~49 tok/s (estimated) | 151–169 tok/s | 161–181 tok/s | ~3.1–3.4× | ~3.3–3.7× |
| First prompt pass (one-time setup) | 304.7 ms | 79.1 ms | 76.6 ms | 3.9× | 4.0× |

The prompt-pass lead is the smallest because PyTorch's prompt products are cuBLAS already; Idrak's 180-row
products run at about 30 TFLOPS against the 78 its GEMM reaches on large shapes (next steps 4 and 5).

## Next steps, in order of expected gain

1. **Delayed FP8 scaling: identify operands by call order, not address.** In training it saved only 12 ms of the
   expected ~30 (`quant_cols` 73 → 61 ms against ~41 predicted by `--bench-fp8`). The likely cause: with
   `--grad-checkpoint` the same buffer address holds different tensors in the forward pass, the recompute and the
   backward pass, so kept maxima belong to another tensor and the correction fires. Key the kept maxima by the
   position of the quantization within the step (reset at each optimizer step) plus the shape, and count corrections
   to confirm. The quick loss-curve test already passes with it (within 0.64% of bfloat16); after the fix, rerun it
   and the 1.25B comparison, then consider turning it on by default.
2. **FP8 quantization fused into its producers** (the larger payoff of delayed scaling): with scales known before a
   tensor exists, layer norm, GELU and the product epilogues can write FP8 directly, removing most of the remaining
   ~100 ms of `quant_rows` / `quant_cols` per 1.25B step.
3. **Decoding GEMVs:** 400–580 GB/s on 4–6 MB of weights, about 2–3 µs of fixed cost per launch; the next step is fewer,
   larger kernels (several layers' products in one persistent kernel) rather than more split tuning. Decoding is at
   ~31% of the memory-bandwidth ceiling (~1.5k tok/s for this model).
4. **INT8 prefill on int8 tensor cores:** prompt products currently expand int8 weights to bfloat16 tiles; the int8
   tensor-core path needs a second, k-major copy of the weights (memory cost: the int8 weights once more).
5. **Prompt pass (180 rows, now 7.7 ms, 2.7× PyTorch):** products run at 37–41 TFLOPS with 64-row tiles and one launch
   per shared input; the o product (180×1024×2048) is still at 30. The rest: norm/rope 0.69 ms, residual + norm
   0.53 ms, attention 0.48 ms, head permute 0.29 ms, gated activation 0.24 ms; candidates are the gated activation and
   the residual + norm in the products' epilogues, and the attention output written in the o product's layout. The
   split rule is 1–1.3 µs off the best for a few 64-row shapes (--bench-gemv), about 0.1 ms per pass.
6. **Training step outside the products** (1.25B, FP8): `adam8` 34 ms, flash-attention backward 32 ms, `sumsq` for
   gradient clipping 9 ms, `fill` (zeroing gradients) 8 ms, dropout 10 ms. Candidates: the first gradient write with
   beta 0 instead of zeroing, clipping's norm fused into the backward pass.
7. **Decoding attention at short contexts:** the ≈5 blocks per SM rule is ~1 µs slower than 16 splits at 200 positions;
   a length-aware split would need the graph-replay constraint handled (splits must not depend on the cache length).
8. **First prompt pass (one-time setup): ~78 ms** against 7.7 ms afterwards (PyTorch 305 ms). Mostly loading the CUDA
   modules (the driver compiles the PTX on first use) and first allocations; caching the compiled modules
   (cuModuleLoadData on a saved cubin, keyed by driver version and GPU) would cut the start-up of every run.
9. **Host overhead of the prompt pass:** 7.7 ms wall against 7.3 ms of GPU time (283 launches); recording the prompt
   pass as a graph per prompt length, or fewer launches (the fusions in step 5), closes that gap.

Not worth pursuing (measured): one-launch column quantization through L2 (removed), more k splits for decoding GEMVs
beyond ≈2 blocks per SM, splitting prompt products whose tiles already fill one wave.

## How to resume

```
git pull
dotnet run -c Release --project tests/Idrak.Tests                          # 272 tests on CPU and GPU
dotnet run -c Release --project tests/Idrak.Tests -- --bench-gemm          # large products, long-k split sweep, 8-bit
dotnet run -c Release --project tests/Idrak.Tests -- --bench-gemv          # decoding products and attention, 180-row products
dotnet run -c Release --project tests/Idrak.Tests -- --bench-fp8           # FP8 column quantizers, delayed-scaling error
dotnet run -c Release --project samples/Idrak.Samples.Chat -- profile Qwen/Qwen3-0.6B --cuda --int8 --kv16
dotnet run -c Release --project samples/Idrak.Samples.Chat -- check qwen3.json --cuda --int8 --kv16 --matmul bf16
dotnet run -c Release --project samples/Idrak.Samples.GptTraining -- <1.25B arguments> --profile [--fp8]
```

Before claiming a speed-up: the tests, the check against transformers, and for training changes the loss comparison
(`--loss-log` + `compare`, the quick 600-step version above).

## Measured facts worth keeping

- One-launch column quantization (maxima and conversion in one kernel through L2) was built and measured: only 13% faster
  on 67 MB tensors and no gain elsewhere, so it was removed.
- FP8 relative precision does not suffer from a 2× larger scale (e4m3 is floating point); the drift test in
  `--bench-fp8` gives the same error with and without the headroom.
- Split-K weight gradients make bfloat16 training sums order-dependent (atomic additions): runs are no longer
  bit-for-bit reproducible.
- The event-timed profile tables overstate kernels of a few microseconds; `--bench-gemv` replays graphs and gives the
  real per-call times.
