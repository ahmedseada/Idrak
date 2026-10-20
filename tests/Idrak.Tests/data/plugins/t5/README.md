# T5 fixtures (plan 26, P10)

The two tiny random T5 models the `plugins: t5` tests read through `Idrak.Text.Plugins.T5`, and transformers' outputs
for them. Made by [tools/t5/make_tiny_t5.py](../../../../../tools/t5/make_tiny_t5.py) with fixed seeds; reruns write the
same bytes on the same machine and library versions (`manifest.json` records the versions and each file's SHA-256).

| File | What |
|---|---|
| `tiny-t5/config.json`, `tiny-t5/model.safetensors` | T5 v1.0 (285 KB, float32): the committed Unigram tokenizer's 1,002 ids, width 32, 4 heads of 6 (inner width 24), feed-forward 48 with ReLU, 3 encoder and 2 decoder layers, 32 relative-position buckets with max distance 20, RMS norm epsilon 1e-6; the head tied to `shared.weight` (no `lm_head.weight`), the decoder's output scaled by 32^-0.5 |
| `tiny-t5-gated/config.json`, `tiny-t5-gated/model.safetensors` | T5 v1.1 / mT5 (444 KB): the same sizes with the gated GELU feed-forward (`gated-gelu`), its own `lm_head.weight` and `tie_word_embeddings` false (nothing scaled); its head's row of `</s>` points at the decoder's state after four greedy tokens of the first text, so some generations end before their limit |
| `reference.safetensors`, `reference.json` | transformers' outputs (eager attention, float32), each model loaded back from its folder: for three single texts and a padded batch of three, the source ids and lengths, the encoder's last hidden states and the decoder's logits for a fixed target prefix; for each single text, the greedy generation (16 new tokens at most) and beam search of 3 (10 new tokens, `early_stopping` true and false) with transformers' sequence scores, as ids and text; and `_relative_position_bucket` of -700..700 for (32, 20) and (32, 128), both ways |
| `manifest.json` | the versions (Python, torch, transformers 5.18.0, tokenizers, safetensors) and each file's size and SHA-256 |
| `real/` | written only by `--real` (the owner's run, below): the real models' references |

The tests copy a model and the tokenizer (`tests/Idrak.Tests/data/tokenizers/unigram`: `tokenizer.json`, or its
`tokenizer.model` as `spiece.model`) into a folder of their own. The tolerance against transformers is 1e-4 (absolute up
to 1, relative above), see `T5Tests.cs`; generations are compared token for token.

## Rebuilding

```powershell
python -m venv $env:USERPROFILE\ref
& $env:USERPROFILE\ref\Scripts\pip.exe install -r tools\t5\requirements.txt
& $env:USERPROFILE\ref\Scripts\python.exe -I tools\t5\make_tiny_t5.py
```

Everything in this folder but this README and `real/` is removed first.

## The real models (`--real`, the owner's)

```powershell
& $env:USERPROFILE\ref\Scripts\python.exe -I tools\t5\make_tiny_t5.py --real
```

Downloads google-t5/t5-small and google/mt5-small (config, tokenizer files, and `model.safetensors`, else
`pytorch_model.bin`) into `<cache>\models\<org>--<name>\` (the cache is `IDRAK_CACHE`, else `~/.cache/idrak`; `--cache`
overrides it) and writes only references: `real/<name>.json` (the files' sizes and SHA-256, the versions, the texts,
their ids and the greedy generations of 20 tokens at most) and `real/<name>.safetensors` (the encoder's states and the
decoder's logits for the greedy target). The `t5 real:` tests then read the models from the same cache.
