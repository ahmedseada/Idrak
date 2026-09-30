# Changelog

## Unreleased

- Chat: a reply that is only a tool call's JSON (`{"name", "arguments"}`, bare or in a ``` block) counts as that call
  when the request offers the tool. Small models such as Qwen2.5-Coder-1.5B write calls this way instead of in their
  template's tags, and the call was taken as the final answer. `ChatOutputParser.CallsInAnswer` does the check.
- CodingAgent sample: a typed task (no verification commands) ends with "Done (the task has no checks)", not "Passed".

## 0.1.1 (2026-09-30)

- `Device.Name` of a GPU adds its compute capability and the CUDA version of its driver; the test runner prints the
  operating system and .NET version first, so a pasted result says what it ran on.
- `idrak-tune`, `idrak-data`: output is UTF-8, so "…" and emoji show on Windows consoles; the help's second lines line up.
- The `Idrak` package page shows its example twice: in the original API and in the simplified API.
- The `Idrak.Datasets` and `Idrak.LanguageModels` package pages name their command-line tools (`idrak-data`,
  `idrak-tune`) and how to install them.

## 0.1.0 (2026-09-29)

First release under the name Idrak.

- `Idrak`: tensors with automatic differentiation, layers, optimizers, training, a SIMD CPU backend and a CUDA backend
  that uses the NVIDIA driver directly (hand-written PTX kernels, any CUDA GPU).
- `Idrak.LanguageModels`: Hugging Face and GGUF language models, tokenizers, chat templates, LoRA / QLoRA fine-tuning.
- `Idrak.Datasets`: datasets from files, Hugging Face, GitHub, Kaggle, Zenodo and URLs.
- `Idrak.AspNetCore`, `Idrak.Mcp`, `Idrak.Onnx`, `Idrak.Onnx.Runtime`: serving, MCP tools, ONNX export and import.
- Tools: `idrak-tune` (fine-tuning) and `idrak-data` (datasets).
