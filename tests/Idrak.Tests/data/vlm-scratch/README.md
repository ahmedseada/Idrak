# The reference for vision-language models made from scratch (plan 19, phase B0)

Written by `tools/vlm/make_tiny_scratch_vlm.py` (nothing downloaded); the versions are in `manifest.json`. Two runs
write the same bytes on the same machine and library versions; this README (by hand) is kept across them.

The family is IdrakVision (`samples/IdrakVision`): a SigLIP-style tower (a 7 x 7 patch convolution with a bias over a
28 x 28 image, 16 learned positions, 3 pre-LayerNorm encoder layers of width 16 with 2 heads, biased attention and a
GELU-tanh MLP of 32, a final LayerNorm), the 4 x 4 patches average-pooled 2 x 2 into 4 tokens (row-major), a projector
`linear_1` (16 to 32), GELU-tanh, `linear_2` (32 to 32), and the from-scratch text reference's Llama decoder (width 32, 2
layers, its tokenizer and ChatML template) with 128 positions. An image is `<start_of_image>`, 4 `<image>` tokens whose
embeddings are the features (unscaled) and `<end_of_image>`; an image's 4 tokens see each other, the rest is causal.

| File | What |
|---|---|
| `tiny-idrak-vision/` | the model folder: `config.json` (`IdrakVisionForConditionalGeneration`, `text_config`, `vision_config`, `projector_config`, the image token ids, `mm_tokens_per_image`, `image_attention`), `model.safetensors` (`vision_tower.*`, `multi_modal_projector.*`, `language_model.model.*`, the head tied), the tokenizer, template and `preprocessor_config.json` (transformers' SiglipImageProcessor: 28 x 28 bilinear, 1/255, mean and std 0.5) |
| `records.jsonl`, `images/` | three conversations (messages format, image parts by path): an image then a question; a system line and an image after the text; two images |
| `tuning.json` | each record's rendered and expanded text, ids, trained mask (the answer and `<|im_end|>`, not the newline after) and image blocks |
| `reference/` | per image its pixel values, tower output and features; the first record's logits |
| `steps.json` | the training runs' settings and losses (before each update) |
| `joint/` | SGD at 0.05 (no momentum, decay or clipping), every parameter trained, one step = the three records, the mean cross entropy over their 57 trained tokens: every gradient at step 1 (checkpoint names, linear weights `[out, in]`), the weights after three steps |
| `align/` | the same with only the tower and the projector trained |

## How the C# tests read it

- `"vlm scratch:"`: IdrakVision reads the folder; the pixel values, tower outputs and features within 1e-5, the logits
  within 1e-4; Idrak's chat encoder gives each record's ids, mask and image blocks exactly; a model created from the
  same configuration has the same tensor names and shapes.
- `"pretrain vision:"`: the pretraining loop on `records.jsonl` gives the joint and align losses within 1e-4, every
  gradient within 1e-4 of the largest (as the weights moved after one step) and the weights after three steps within
  5e-5; in the align run the decoder does not move.

## Regenerate

```
vlm/bin/python tools/vlm/make_tiny_scratch_vlm.py tests/Idrak.Tests/data/vlm-scratch
```
