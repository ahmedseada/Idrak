# Idrak.Nlp

Natural language processing for Idrak (no dependencies beyond Idrak and Idrak.Data): generate text and chat with tools, using each model's own Jinja chat template; host text and chat models in the inference engine; fine-tune language models (LoRA, DoRA, QLoRA, DPO / ORPO / SimPO, distillation from a teacher model), evaluate them and score their answers; search documents (BM25, vectors, rerankers) and answer from them with citations (RAG); run a coding agent with file and shell tools.

Loading the models themselves (Llama, Qwen, Mistral, Gemma and mixture-of-experts models by Hugging Face id, folder or GGUF file, with their BPE tokenizers) is in the core package, `Idrak` (`Idrak.Models`); this package adds `CreateGenerator` and `CreateChat` to a loaded `PretrainedModel`.

## Install

```bash
dotnet add package Idrak.Nlp
```

This is a library (for your code). The command-line tool `idrak` chats with, serves, fine-tunes, evaluates and exports models with no code needed (`idrak chat`, `idrak serve`, `idrak tune`); it is a separate package:

```bash
dotnet tool install -g Idrak.Cli
idrak help tune
```

Part of [Idrak](https://www.nuget.org/packages/Idrak), a self-contained deep-learning library for .NET.

## Documentation

Guides, samples and the full API overview: https://github.com/ahmedseada/Idrak

License: Apache 2.0.
