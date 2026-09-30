# Changelog

## 0.1.2 (2026-09-30)

- `Dataset.Split` removes duplicate rows before splitting (the default), so no row is in both parts;
  `removeDuplicates: false` keeps every row, identical copies included, and `duplicates:` sets the rules. Splits of data
  with repeated rows now have fewer rows than before.
- `Dataset.Deduplicate(options)` / `DeduplicateWithReport(options)`: exact duplicates on the features or on features and
  targets; rows whose features repeat with other targets: keep the first, keep the most frequent targets, or drop them;
  near duplicates from an embedding the caller gives (cosine threshold). The report has the kept row indices and the
  count removed for each reason. One hash per row in parallel, no allocation per row, no copy when nothing is removed;
  near duplicates compare blocks of rows as matrix products.
- `Losses.CrossEntropy(logits, targets)` and `Losses.SparseCrossEntropy(logits, classIndices)`: two-argument overloads
  (no label smoothing), so `Loss = Losses.CrossEntropy` compiles as the README shows; the optional `labelSmoothing`
  parameter kept the method group from converting to `Func<Tensor, Tensor, Tensor>`. The smoothing overloads now take
  it as a required third argument (calls such as `CrossEntropy(a, b, labelSmoothing: 0.1f)` are unchanged).
- Chat: a tool call written as JSON in the reply (`{"name", "arguments"}`: the whole reply, bare or in a ``` block, or a
  ``` block that ends the reply after a sentence) counts as that call when the request offers the tool; the sentence
  stays the reply's text. Small models such as Qwen2.5-Coder-1.5B write calls this way instead of in their template's
  tags, and the call was taken as the final answer. `ChatOutputParser.CallsInAnswer` does the check.
- `CodingAgent`: calls identical to the round just before are not run again; the model gets "Not run again: … use
  that result" in their place (counted as tool errors). A small model ran `dotnet --version` three times instead of
  answering. The same call after a different one still runs.
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
