# Tiny Qwen-VL reference (plan 16, step 0)

Written by `tools/vlm/make_tiny_qwen_vl.py` (transformers 5.19, CPU; the versions and every file's SHA-256 are in
`manifest.json`). Tiny random Qwen2.5-VL and Qwen2-VL checkpoints, each as transformers 5 saves it and with the hub
checkpoints' config.json (`-flat`), three pages cut from `../vlm/image.png`, and transformers' outputs in `reference/`:
each page's pixel values, patch embedding, tower output (in patch order) and merged features; the image prompt's ids,
position ids [3, tokens], logits and 20 greedy tokens with each step's logits; the same for two pages and for text
only; a tuning example's logits and loss. Rerun the script to regenerate; reruns give the same bytes.
