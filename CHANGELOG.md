# Changelog

## 0.1.0 (unreleased)

First release under the name Idrak.

- `Idrak`: tensors with automatic differentiation, layers, optimizers, training, a SIMD CPU backend and a CUDA backend
  that uses the NVIDIA driver directly (hand-written PTX kernels, any CUDA GPU).
- `Idrak.Pretrained`: Hugging Face and GGUF language models, tokenizers, chat templates, LoRA / QLoRA fine-tuning.
- `Idrak.Datasets`: datasets from files, Hugging Face, GitHub, Kaggle, Zenodo and URLs.
- `Idrak.AspNetCore`, `Idrak.Mcp`, `Idrak.Onnx`, `Idrak.Onnx.Runtime`: serving, MCP tools, ONNX export and import.
- Tools: `idrak-tune` (fine-tuning) and `idrak-data` (datasets).
