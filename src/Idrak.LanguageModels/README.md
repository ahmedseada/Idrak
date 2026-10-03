# Idrak.LanguageModels

Language models for Idrak (no dependencies): load Llama, Qwen, Mistral and Gemma (1, 2 and 3) models by Hugging Face id, folder or GGUF file (safetensors and quantized GGUF weights, an extensible architecture registry), with their own BPE tokenizers and chat templates; chat with them, fine-tune them (LoRA / QLoRA) and score their answers.

## Install

```bash
dotnet add package Idrak.LanguageModels
```

This is a library (for your code). The command-line tool `idrak-tune` (fine-tune, evaluate, chat with and export models, no code needed) is a separate package:

```bash
dotnet tool install -g Idrak.FineTuning.Cli
```

Part of [Idrak](https://www.nuget.org/packages/Idrak), a self-contained deep-learning library for .NET.

## Documentation

Guides, samples and the full API overview: https://github.com/ahmedseada/Idrak

License: Apache 2.0.
