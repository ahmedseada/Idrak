# IdrakVision: Idrak's own vision-language family (plan 19, track B)

A family made to be created, not only loaded: a model of it is made from a description (`idrak create` with
`"vision"`), pretrained (`idrak pretrain`), tuned (`idrak tune`) and read back like any checkpoint. Every shape is a
setting, so anyone can make the model they need; nothing in it prefers a size or a card.

- **Tower**: a patch convolution with a bias, learned positions, pre-LayerNorm encoder layers (biased attention, every
  patch seeing every other; an MLP with the configured activation), a final LayerNorm.
- **Projector**: the patch grid average-pooled `pool_size x pool_size`, then `depth` linear layers to the text width.
- **Decoder**: a Llama decoder read from `text_config` (the library's Llama-style family), tensors under `language_model.`.
- **Prompt**: the chat template writes `<image>`; it becomes `<start_of_image>`, one `<image>` per image token and
  `<end_of_image>`. The image tokens attend to each other in both directions (`"image_attention": "image-blocks"`, the
  default) or causally (`"causal"`).

## config.json

| Key | What |
|---|---|
| `architectures` | `["IdrakVisionForConditionalGeneration"]` |
| `text_config` | the decoder (Llama keys) |
| `vision_config` | `hidden_size`, `intermediate_size`, `num_hidden_layers`, `num_attention_heads`, `num_channels`, `image_size`, `patch_size`, `layer_norm_eps`, `hidden_act` (gelu_pytorch_tanh, gelu, quick_gelu, relu), `pool_size` |
| `projector_config` | `depth`, `hidden_act`, `bias` |
| `image_token_index`, `boi_token_index`, `eoi_token_index`, `mm_tokens_per_image` | the image tokens |
| `image_attention` | `image-blocks` or `causal` |

## A description to create from (`CreateOptions.Vision`, or `"vision"` in idrak create's spec)

```json
{
  "vision_config": { "hidden_size": 384, "intermediate_size": 1536, "num_hidden_layers": 6, "num_attention_heads": 6,
                     "image_size": 448, "patch_size": 14, "pool_size": 2 },
  "projector_config": { "depth": 2, "hidden_act": "gelu_pytorch_tanh" },
  "image_attention": "image-blocks",
  "image_token": "<image>", "begin_image": "<start_of_image>", "end_image": "<end_of_image>",
  "init_std": 0.02
}
```

Every key is optional; `"preprocessing"` (a preprocessor_config.json) is too (the image size, bilinear, mean and deviation
0.5 otherwise). The tokenizer needs the three image tokens (`idrak data tokenizer` adds them by default).

## Use

```
idrak create spec.json -o model -P Idrak.IdrakVision.dll
idrak pretrain model -C pretrain.json -o run -P Idrak.IdrakVision.dll
idrak run run/checkpoint-2000 --image page.png "Read the page." -P Idrak.IdrakVision.dll
```

In an app: reference the project and call `IdrakVisionPlugin.Register()`.

## From nothing to a reader, in one script

`scripts/b6-curriculum.ps1` (from the repository root) makes everything a missing `work` folder needs and teaches the
model to read in three stages: a corpus, tokenizer and token file; the text model; the vision model on it; clean pages
cut to their ink (`data synth --page crop=1`); then single words, two or three words and full lines, each stage from
the previous one's checkpoint and evaluated by its character error rate on held-out images. What exists is kept and a
stopped run resumes, so it can be run again; its output also goes to `work\b6.log`. Every size and step count is a
parameter (`-Device`, `-Fonts`, `-ImageSize`, `-Steps1`, `-Rate1`, ...).

```
powershell -ExecutionPolicy Bypass -File samples\IdrakVision\scripts\b6-curriculum.ps1 -Device cuda:0
```
