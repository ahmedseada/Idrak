# Idrak.Text

Natural language processing for Idrak (no dependencies beyond Idrak and Idrak.Data): generate text and chat with tools, using each model's own Jinja chat template; host text and chat models in the inference engine; fine-tune language models (LoRA, DoRA, QLoRA, DPO / ORPO / SimPO, distillation from a teacher model), evaluate them and score their answers; search documents (BM25, vectors, rerankers) and answer from them with citations (RAG); run a coding agent with file and shell tools.

Loading the models themselves (Llama, Qwen, Mistral and mixture-of-experts models by Hugging Face id, folder or GGUF file, with their BPE tokenizers; Gemma and the other families from `Idrak.Text.Plugins`) is in the core package, `Idrak` (`Idrak.Models`); this package adds `CreateGenerator` and `CreateChat` to a loaded `PretrainedModel`.

## Contracts

The plug-in points only this package uses live beside their implementations (add the `using` line to name them):

| Namespace | Contracts |
|-----------|-----------|
| `Idrak.Abstraction.Generation` (Idrak.Abstraction, for every package that samples tokens) | `ITokenSampler` (with `SamplerRequest`, `SampledToken`) and `TokenSamplers`, whose "default" is `TokenSampler`: register a sampler under "default" to override it for every generation; under `SlotPolicy.FallBack` (opted into with `TokenSamplers.SetPolicy`) it falls back to the built-in when it cannot be made, and under `SlotPolicy.Shadow` the built-in chooses while the app's is compared over whole generations. The built-in is version 2 (since 0.4.0: NaN logits never sampled, +∞ wins, a row with nothing finite reported by `Read`); `TokenSamplers.Default("default", 1)` is 0.3's, which refuses non-finite logits |
| `Idrak.Text.Retrieval.Abstractions` | `IRetriever`, `IReranker` (with `Chunk`, `RetrievedChunk`); the vector store (`IVectorStore`, `VectorRecord`, `VectorMatch`, with `InMemoryVectorStore`, `VectorIndex`, `TopHits`) is in `Idrak.Abstraction.Retrieval`, for any package's vectors |
| `Idrak.Text.Abstractions` | `DistillationTeacher`, `TeacherDistributions`, `TrainingSequence`; for fine-tuning vision-language models: `ITuningDataFormat` and `TuningDataFormats` ("messages": chat JSON Lines with image parts; "sharegpt": LlamaFactory's `conversations` with `<image>` placeholders and an `images` list, JSON Lines or a JSON array, images from a folder or a zip read in place; `ReadPending` reads and checks every record and a conversation's images only when it is taken, `PendingConversations`), `FeatureCacheKey.For` (an image's key in the feature caches: `IFeatureCache` and `FeatureCaches` live in `Idrak.Abstraction.Generation`, this package registering "memory", within a budget measured from the machine's free memory, and "disk", safetensors files under `IDRAK_CACHE`), `ITuningMetric` and `TuningMetrics` ("cer", "wer": Unicode-aware error rates); for tuning data drawn from text (`SyntheticPages`, plan 18): `ISyntheticRecordFormat` and `SyntheticRecordFormats` ("sharegpt", "messages", "text": the records pages are drawn from and written back in their shape), `IPagePartFiller` and `PagePartFillers` (the slots of header, subject and footer parts: "digits", "number", "pick", "field", "line", "date", "hijri"), `IRandomTextSource` and `RandomTextSources` (random text to draw pages from when there is none: "ar" and "en" built in, any language from a pools file or a word list with `PooledTextSource`) |

Shared contracts (tokenizers, chat templates, chat and text models, tools, `IEmbedder`) are in Idrak.Abstraction. `TuningImages` (in `Idrak.Text`) is how a vision fine-tune prepares its images (transforms, vision options, grayscale), saved next to the adapters as `tuning_images.json`.

## Install

```bash
dotnet add package Idrak.Text
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
