# The reference for models made from scratch (plan 19, phase A0)

Written by `tools/vlm/make_tiny_pretrain.py` (nothing downloaded); the versions are in `manifest.json`. Two runs write
the same bytes on the same machine and library versions; other machines or versions may differ in the last bits.

| File | What |
|---|---|
| `corpus.txt` | 96 documents, Arabic and English by turns, separated by lines of `---` (the layout `idrak data text` writes): lines of 6 to 14 words with numbers, Hijri-style dates and punctuation, from fixed word lists and a seeded generator |
| `tokenizer/tokenizer.json`, `train.json` | a byte-level BPE trained by Hugging Face tokenizers from `corpus.txt` (`ByteLevel` without a prefix space, vocabulary 512, minimum frequency 2, eight special tokens first); `train.json` holds the trainer's settings, the vocabulary by id, the merges in rank order and the ids of 21 sample texts |
| `tokenizer-full/` | the same merging until no pair is left (minimum frequency 1, vocabulary 4096): 1,628 merges through many count ties |
| `metaspace/` | the same with the `Metaspace` pre-tokenizer (`▁`, prepend always, split) |
| `tokens.bin` | every document encoded with `tokenizer/` and followed by `</s>`, uint16 little-endian (25,515 tokens) |
| `tiny-llama/` | a `LlamaForCausalLM` (hidden 32, 2 layers, 4 heads, 2 key-value heads, head size 8, feed-forward 64, vocabulary 512, 64 positions, tied embeddings), every weight from one seeded generator; the tokenizer, and a ChatML template that walks content parts (an image part is `<image>`), the template created models get |
| `step/step.json` | training on the first four 32-token windows of `tokens.bin` (33 ids each, at 0, 32, 64, 96: `TokenFileSource(path, 32)`'s windows 0 to 3), every parameter trained, the loss the mean cross entropy over all 128 positions; the window ids and the losses before each of three updates |
| `step/grads-step1.safetensors` | every parameter's gradient at step 1 (transformers' names; linear weights `[out, in]`) |
| `step/weights-step3.safetensors` | the weights after three plain SGD steps (rate 0.2, no momentum, decay or clipping) |
| `step/adamw/` | the same with AdamW (rate 1e-3, betas 0.9 and 0.95, epsilon 1e-8, weight decay 0.1, no clipping): the weights and the moments (`NAME.exp_avg`, `NAME.exp_avg_sq`) after three steps |
| `step/adamw-folded/` | the same AdamW with both bias corrections folded into the step size (`p -= lr·√(1-β2^t)/(1-β1^t) · m/(√v + ε)`), as Idrak's AdamW computes it: epsilon is then ε/√(1-β2^t) of torch's, which moves weights whose gradients are near ε (embedding rows reached only through the tied head) |

## How the C# tests read it

- `"tokenizer training:"`: the library's "bpe" trainer on `corpus.txt` with `train.json`'s settings gives the same
  vocabulary (ids included) and the same merges in the same order, exactly, and encodes the sample texts to the same ids.
- `"pretrain:"`: `tiny-llama/` loads; the pretraining loop's first step on the four windows matches the loss within
  1e-4 and every gradient within 1e-4 of the largest; three SGD steps' losses within 1e-4 and the weights within 5e-5;
  three AdamW steps' losses within 1e-4 of torch's AdamW and the weights within 1e-5 of `adamw-folded` (the formula
  Idrak's AdamW uses); a run resumed after step 2 matches the uninterrupted one bit for bit.

## Regenerate

```
python -m venv vlm && vlm/bin/pip install torch transformers tokenizers safetensors numpy jinja2
vlm/bin/python tools/vlm/make_tiny_pretrain.py tests/Idrak.Tests/data/pretrain
```
