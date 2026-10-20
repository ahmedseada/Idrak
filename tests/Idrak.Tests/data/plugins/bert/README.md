# BERT fixtures (plan 26, P9)

The tiny random BERT the `plugins: bert` tests read through `Idrak.Text.Plugins.Bert`, and transformers' outputs for it.
Made by [tools/bert/make_tiny_bert.py](../../../../../tools/bert/make_tiny_bert.py) with fixed seeds; reruns write the
same bytes on the same machine and library versions (`manifest.json` records the versions and each file's SHA-256).

| File | What |
|---|---|
| `tiny-bert/config.json`, `tiny-bert/model.safetensors` | the model (337 KB, float32): the committed WordPiece tokenizer's 1,861 tokens, 2 layers of width 32, 4 heads, feed-forward 64 (exact GELU), 64 positions, 2 token types, LayerNorm epsilon 1e-12; one checkpoint with the tensors of `BertModel` (under `bert.`, the pooler included), `BertForMaskedLM` (`cls.predictions.`, the decoder tied to the word embeddings) and `BertForSequenceClassification` (`classifier.`, 3 labels); config.json names `BertForMaskedLM` |
| `reference.safetensors`, `reference.json` | transformers' outputs (eager attention, float32): for three single texts and a padded batch of three text pairs with token types, the ids, type ids and lengths, the last hidden states, the pooled output, the masked-LM logits at two positions per sequence, the classification logits, and the mean and first-token sentence vectors scaled to length 1 |
| `manifest.json` | the versions (Python, torch, transformers 5.18.0, tokenizers, safetensors) and each file's size and SHA-256 |
| `real/` | written only by `--real` (the owner's run, below): the real models' references |

The tests copy the model and the tokenizer (`tests/Idrak.Tests/data/tokenizers/wordpiece-uncased`) into a folder of
their own. The tolerance against transformers is 1e-4 (absolute up to 1, relative above), see `BertTests.cs`.

## Rebuilding

```powershell
python -m venv $env:USERPROFILE\ref
& $env:USERPROFILE\ref\Scripts\pip.exe install -r tools\bert\requirements.txt
& $env:USERPROFILE\ref\Scripts\python.exe -I tools\bert\make_tiny_bert.py
```

Everything in this folder but this README and `real/` is removed first.

## The real models (`--real`, the owner's)

```powershell
& $env:USERPROFILE\ref\Scripts\python.exe -I tools\bert\make_tiny_bert.py --real
```

Downloads google-bert/bert-base-uncased and aubmindlab/bert-base-arabertv02 (config, tokenizer files, and
`model.safetensors`, else `pytorch_model.bin`) into `<cache>\models\<org>--<name>\` (the cache is `IDRAK_CACHE`, else
`~/.cache/idrak`; `--cache` overrides it) and writes only references: `real/<name>.json` (the files' sizes and SHA-256,
the versions, the texts and their ids) and `real/<name>.safetensors` (hidden states, the pooled output when the
checkpoint has a pooler, and the 32 largest masked-LM logits at each `[MASK]` with their ids). The `bert real:` tests
then read the models from the same cache.
