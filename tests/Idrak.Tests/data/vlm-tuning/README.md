# A tiny vision fine-tune reference (plan 12, phase 0)

Written by `tools/vlm/make_tiny_tuning.py`. The reference that plan 12's tuner (images in `FineTuner` and `idrak tune`)
is checked against: transformers fine-tunes plan 11's tiny models, `../vlm/tiny-gemma3` and `../vlm-llava/tiny-llava`,
on three image-answer records with a hand-written LoRA (no peft), and these files hold what it did. Nothing is
downloaded; the images are plan 11's `../vlm/image.png` and `../vlm/image-palette.png`.

## The setup (the same for both models)

- **LoRA** on `q_proj`, `k_proj`, `v_proj`, `o_proj` of every text decoder layer: rank 2, alpha 4 (scale alpha / rank = 2),
  no dropout, `base(x) + 2 · (x Aᵀ) Bᵀ` with A [rank, in] and B [out, rank] as peft stores them. The initial values come
  from `torch.Generator().manual_seed(2026)`, per layer and target in order: A uniform in ±1/√in, then B uniform in
  ±0.2. **B is not zero** (unlike the usual initialization), so step 1 has gradients for A as well.
- **Frozen:** the vision tower, the embeddings, every base weight. Run `lora`: the projector frozen too. Run `projector`:
  the projector trained as well (Gemma 3: `mm_input_projection_weight`, `mm_soft_emb_norm.weight`; LLaVA: `linear_1`
  and `linear_2`, weights and biases), from the same initial adapters.
- **A step** is all three records in one batch, in order. The loss is the mean token cross-entropy over the batch's
  trained tokens: Σ over records and positions t of `trained[t + 1] · CE(logits[t], input_ids[t + 1])`, divided by the
  sum of the records' `trained_tokens` (39 + 12 + 21 = 72 for Gemma 3, 36 + 13 + 20 = 69 for LLaVA).
  Each record is its own row: no padding, no packing; Gemma 3's image blocks bidirectional (`token_type_ids`), LLaVA's
  image tokens causal.
- **Optimizer:** plain SGD, learning rate 0.2, no momentum, no weight decay, no clipping, no schedule; 3 steps. The loss
  of step k is computed before step k's update.

## The masking (plan 12's decision)

Every prompt token is untrained: the system and user text, the turn headers, the image tokens, `<start_of_image>` and
`<end_of_image>` and the `"\n\n"` around them. The answer is trained, with the token that ends the assistant's turn
(Gemma 3: `<end_of_turn>`). The `"\n"` the Gemma 3 template writes after `<end_of_turn>` is **not** trained.

That is the rule of Idrak's `ChatTranscriptEncoder` (the assistant's span runs from the text the template's generation
prompt adds, `<start_of_turn>model\n`, to the text that ends an assistant message in a probe conversation, trimmed:
`<end_of_turn>`), which the script reproduces from the template alone. The tiny LLaVA template writes no end of turn
(`ASSISTANT: {answer} `), so its trained span runs from after `ASSISTANT:` to the end of the text: `" {answer} "`.

**Against LlamaFactory** (0.9.5, its `gemma3` template and gemma3 multimodal plug-in, read from the wheel and reproduced
in the script, which asserts the source lines it follows): each record written in LlamaFactory's ShareGPT format (an
`<image>` placeholder per image in the human turn, an `images` list, `system` when there is one) gives **the same token
ids**, token for token; its labels train one token more, the `"\n"` after `<end_of_turn>` (its assistant slot is
`{{content}}<end_of_turn>\n`). The tuner keeps Idrak's rule (the text tuner's, and generation stops at `<end_of_turn>`,
so that `"\n"` is never predicted); both masks are written.

## Files

| File | What it is |
|---|---|
| `manifest.json` | Versions (`versions`), each file's size and SHA-256 (`files`), every safetensors file's tensor names, dtypes and shapes (`tensors`), the losses (`summary`) |
| `<model>/tuning.json` | `settings`, `records`, `runs` (below); Gemma 3's also `llamafactory_gemma3` (version, wheel name and SHA-256, the template's source) |
| `<model>/adapter-init/` | The initial adapters as a PEFT folder (`adapter_config.json`: r 2, lora_alpha 4, LORA; `adapter_model.safetensors`) that Idrak's `PretrainedModel.LoadAdapter` reads: `base_model.model.<checkpoint tensor name>.lora_A.weight` [2, in] and `.lora_B.weight` [out, 2], the checkpoint's own names (Gemma 3: `model.language_model.layers.N.self_attn.q_proj`; LLaVA: `language_model.model.layers.N.self_attn.q_proj`) |
| `<model>/features/image.npy`, `image-palette.npy` | float32 [1, tokens, text width]: the image's features with the checkpoint's projector (Gemma 3 [1, 4, 24], LLaVA [1, 16, 32]) |
| `<model>/<run>/grads-step1.safetensors` | float32 gradients of step 1 under the adapters' names (peft orientation, as above); in the `projector` run also the projector's under the checkpoint's names (Gemma 3's `mm_input_projection_weight` [8, 24] as stored, x · W; LLaVA's `linear_*.weight` [out, in] as torch keeps them) |
| `<model>/<run>/adapter-step3/` | The adapters after the 3 steps, a PEFT folder as `adapter-init` |
| `<model>/projector/projector-step3.safetensors` | The projector's values after the 3 steps (checkpoint names) |

`tuning.json`:

- `settings`: `model` (the folder, relative to this one), `architecture`, `checkpoint_prefix`, `lora` (`rank`, `alpha`,
  `scale`, `dropout`, `targets`, `init`), `optimizer` (`kind` "sgd", `learning_rate`, `momentum`, `weight_decay`,
  `max_gradient_norm` 0, `schedule` "constant"), `steps`, `batch`, `loss`, `tower`, `image_token`,
  `assistant_header` and `assistant_end` (what `ChatTranscriptEncoder` finds), `projector_parameters`.
- `records[]`: `messages` (the system's and the assistant's content as strings, the user's as parts: `{"type":
  "image"}` and `{"type": "text", "text"}`), `images` (file names in `../vlm`, in prompt order), `rendered` (the chat
  template's text, one marker per image), `assistant_span` ([start, end) character offsets of the trained text in
  `rendered`), `expanded` (the markers expanded as the processor expands them), `input_ids`, `tokens`, `trained` (one
  flag per token: token i is the target of position i - 1; `trained[0]` is false), `trained_tokens` (trained targets),
  `image_blocks` (`[{position, tokens}]`: each image's first image token and count, what `ImagePrefill.Locate` finds);
  Gemma 3's also `token_type_ids`, `sharegpt` (the record in LlamaFactory's format) and `llamafactory` (`input_ids`,
  `labels` with -100 untrained, `trained`, `same_ids`, `differs_from_trained_at`: the last index, the `"\n"`).
- `runs.lora`, `runs.projector`: `trained` (the parameter names), `losses` (one per step, before its update),
  `step1_grad_norm` (the L2 norm of all step-1 gradients), `step1_per_record` (`cross_entropy_sum` and every trained
  token's cross-entropy at step 1, in order), `files` (`grads_step1`, `adapter_after`, `projector_after`).

The ids are checked three ways in the script: transformers' processor on the whole rendered conversation, the encoder's
segment-by-segment tokenization (prompt, answer, the rest) and LlamaFactory's encoding. `VisionTuningReferenceTests.cs`
checks the records through `ChatTranscriptEncoder`, the family's `IImagePromptFormat` and `ImagePrefill.Locate`, and runs
the 3 steps by hand on the CPU (`ImagePrefill.Forward` with autograd, the adapters loaded from `adapter-init`, a masked
mean cross-entropy, `Sgd`): losses, step-1 gradients and the final adapters (and Gemma 3's projector) agree within
about 1e-6.

## Regenerate

```
python -m venv vlm && vlm/bin/pip install torch transformers tokenizers safetensors pillow numpy jinja2
pip download llamafactory==0.9.5 --no-deps -d lf            # read from the wheel, never installed
vlm/bin/python tools/vlm/make_tiny_tuning.py tests/Idrak.Tests/data/vlm-tuning \
    --llamafactory-wheel lf/llamafactory-0.9.5-py3-none-any.whl
```

Built with Python 3.13.16, torch 2.14.1 (CPU), transformers 5.19.0, tokenizers 0.23.3, safetensors 0.8.0, Pillow 12.3.0,
numpy 2.5.3, and LlamaFactory 0.9.5's source. Two runs write the same bytes; other machines or versions may differ in
the last bits, so tests compare losses and cross-entropies within 1e-4, gradients within 1e-4 of the largest, values
within 1e-5, and ids and masks exactly.
