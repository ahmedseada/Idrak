# The reference for data-parallel training (plan 23, phase C0)

Written by `tools/cluster/make_tiny_ddp.py` (nothing downloaded) from plan 19's fixture, which it only reads:
`../pretrain/tiny-llama/` (the model) and `../pretrain/tokens.bin` (the tokens; both SHA-256 in `manifest.json`). The
versions are in `manifest.json`. Two runs write the same bytes on the same machine and library versions; other machines
or versions may differ in the last bits. It checks results only; it does not shape Idrak's design.

Every run trains on the first eight 32-token windows of `tokens.bin` (33 ids each, window *i* at 32·*i*:
`TokenFileSource(path, 32)`'s windows 0 to 7), every parameter trained, the loss the mean token cross entropy over every
position. Tensors carry transformers' names (linear weights `[out, in]`; the head is tied, so it has no tensor of its own).

| File | What |
|---|---|
| `ddp/ddp.json` | PyTorch `DistributedDataParallel`, `gloo` backend, two CPU processes on 127.0.0.1; rank *r* takes windows *r*, *r*+2, *r*+4, *r*+6 (`rank_windows`); the window ids, the settings, and per run `rank_losses[r]` (rank *r*'s own loss before each of three updates) and `losses` (the two ranks' mean, computed in double: the global mean, since every rank has as many positions); `vs_single`: the largest differences against `single/` |
| `ddp/grads-step1.safetensors` | the averaged gradient every rank holds after step 1's backward, before the update (DDP's average: each rank's gradient of its own mean loss, summed, divided by 2) |
| `ddp/weights-step3.safetensors` | the weights after three plain SGD steps (rate 0.2, no momentum, decay or clipping) |
| `ddp/adamw/` | the same with AdamW (rate 1e-3, betas 0.9 and 0.95, epsilon 1e-8, weight decay 0.1, no clipping): the weights and the moments (`NAME.exp_avg`, `NAME.exp_avg_sq`) after three steps |
| `ddp/adamw-folded/` | the same AdamW with both bias corrections folded into the step size, as Idrak's AdamW computes it (plan 19's `adamw-folded`, the class reused from `make_tiny_pretrain.py`): the weights after three steps |
| `single/` | the same three steps in one process on all eight windows at once (batch 8): `single.json` (losses), `grads-step1`, `weights-step3`, `adamw/`, `adamw-folded/`, laid out as `ddp/` |
| `bf16/` | the SGD run of `ddp/` again with each rank's gradients rounded to bfloat16 (round to nearest even) before the all-reduce, the sum done in float32 and then divided by 2 (a DDP communication hook): `bf16.json` (losses per rank and their mean; `vs_ddp`: the largest differences against `ddp/`), `grads-step1`, `weights-step3` |
| `manifest.json` | the versions, the sources' SHA-256, each tensor file's names, dtypes and shapes, and every file's size and SHA-256 |

The script asserts that both ranks hold the same gradients, weights and moments bit for bit after every run.

## Tolerances measured

`ddp/` against `single/` (the script fails above 1e-5):

| Run | Losses | Gradients at step 1 | Weights after step 3 | Moments after step 3 |
|---|---|---|---|---|
| SGD | 4.8e-7 | 1.3e-7 (largest gradient 0.44) | 1.2e-7 | |
| AdamW | 9.5e-7 | | 2.3e-6 | 6.7e-8 |
| AdamW folded | 9.5e-7 | | 6.7e-6 | 1.8e-7 |

AdamW turns the last-bit differences of a batch of 8 against two of 4 into larger weight differences where a second
moment is small (`model.layers.1.self_attn.o_proj.weight`); they stay inside 1e-5.

`bf16/` against `ddp/` (measured, not asserted): **losses 1.5e-4, gradients at step 1 5.9e-4 (1.3e-3 of the largest
gradient, 0.44; in `model.norm.weight`), weights after three SGD steps 1.6e-4**. A bf16 wire keeps 8 bits of mantissa, so
its gradients are only good to about 2⁻⁹ of their size; these numbers are the tolerance a run over a bf16 wire is held to.

## How the C# tests read it

- `"cluster: ddp fixture agrees with single"`: reads both folders' files and re-checks what the script asserts: the
  `ddp.json` mean losses against `single.json`'s within 1e-5, and every gradient, weight and moment within 1e-5.
- `"cluster: data parallel 2 ranks on sim matches the ddp fixture"` (later, C1 onward): Idrak's `Trainer` with two
  ranks on the `sim` transport, the same windows per rank, matches `ddp/`: every gradient at step 1 within 1e-4 of the
  largest, the losses (per rank and their mean) and the weights after three steps within 1e-5; AdamW is held to
  `ddp/adamw-folded/` (the formula Idrak's AdamW uses), the losses also to `ddp/adamw/`. A run with a bf16 wire is held to
  `bf16/` within the tolerances above.

## Regenerate

```
python -m venv ddp && ddp/bin/pip install torch transformers tokenizers safetensors numpy jinja2
ddp/bin/python tools/cluster/make_tiny_ddp.py tests/Idrak.Tests/data/cluster
```

On Windows (PowerShell, from the repository root):

```
python -m venv ddp; ddp\Scripts\pip install torch transformers tokenizers safetensors numpy jinja2
ddp\Scripts\python tools\cluster\make_tiny_ddp.py tests\Idrak.Tests\data\cluster
```

The script keeps this README, rewrites everything else, and prints the largest differences it measured.
