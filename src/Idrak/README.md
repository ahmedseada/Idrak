# Idrak

**Deep learning in pure .NET. Every GPU. Zero dependencies.**

A self-contained deep-learning library for .NET 10, written in C#, with its own GPU kernels for every backend. No
TensorFlow, no PyTorch, no CUDA Toolkit, no cuBLAS or cuDNN, no shader compiler, no native libraries: kernels are
generated in C# (PTX for NVIDIA, SPIR-V for Vulkan, HIP C for AMD) and compiled by the GPU's own driver on the user's
machine.

## Install

```bash
dotnet add package Idrak
```

Runs on any machine with .NET 10 (x64 or ARM64: Windows, Linux, macOS, Android) and uses the GPU through its driver
alone.

## What is in the package

| Area | What is there |
|------|---------------|
| Tensors | N-D tensors with automatic differentiation, float32, bfloat16 and FP8 precision, int8 / int4 / bfloat16 weights |
| Layers | Linear, Conv2d, pooling, BatchNorm, LayerNorm, Embedding, LSTM, GRU, multi-head attention, transformer layers, mixture of experts; graph modules with skip connections and branches |
| Training | `Trainer` and `TrainingRun`, losses, metrics, early stopping, SGD, Adam, AdamW (fused on the GPU), 8-bit Adam, schedules, gradient clipping, memory offloading |
| Data | In-memory datasets and lazy sources (streamed CSV, image folders, memory-mapped token files and .npy arrays), views, image augmentation, PNG / BMP / PGM / PPM decoding |
| Pretrained models | Llama, Qwen, Mistral, Gemma and mixture-of-experts models by Hugging Face id, folder or GGUF file: safetensors and quantized GGUF weights, BPE tokenizers, LoRA and DoRA adapters in the PEFT layout (`Idrak.Models`) |
| Inference | Predictors, model packages (`.ikm`), an inference engine with batching and keep-alive (text and chat models are added by `Idrak.Nlp`) |
| Vision | Channel normalization layer; per-pixel loss for segmentation (region classification and content framing are in `Idrak.Vision`) |
| ONNX | Export to `.onnx` (opset 17) that ONNX Runtime, TensorRT, OpenVINO, DirectML and other runtimes can run; import of `.onnx` files into layers or graph modules; no dependencies (the protobuf is read and written here) |
| Telemetry | Hooks that cost nothing when unused: console, CSV metrics, JSON Lines |
| Extending | More than twenty registries: samplers, KV cache layouts, packed weight formats, builder steps, graph operations, RoPE scalings, tool-call formats, data sources, image codecs, optimizers, differentiable operations |

## Contracts of this package

The plug-in points only this package uses live here, each under the `.Abstractions` namespace beside its
implementations (the shared ones are in [Idrak.Abstraction](https://www.nuget.org/packages/Idrak.Abstraction)). Code
that names them adds the `using` line. The registries register their built-ins themselves, on first use, as the library
defaults of their slots: an app's registration under a built-in name overrides it, the default stays behind it, and
`Unregister` brings it back (`Default(name)`, `Origin(name)`). Under the default policy, `Throw`, a failure of the
app's version reaches the caller and is reported with a hint. `SetPolicy(name, ...)` (or `IDRAK_OVERRIDE_POLICY`) opts
a slot into `FallBack`, where the library's answers instead: per call for graph operations, layer descriptions, image
codecs, checkpoint formats and GGUF dequantizers; when the object is made for tokenizer parts, layers, sample sources and
checkpoint stores; or into `Shadow`. Network steps, ONNX
translators and the GGUF and family descriptions are used as registered (they change a builder or graph as they run, or
are data).

| Namespace | Contracts |
|-----------|-----------|
| `Idrak.Layers.Abstractions` | `LayerTypes`, `GraphOps` (with `GraphOp`, `GraphOpContext`, `GraphNode`), `NetworkOps` and `INetworkBuilder` (network-builder steps), `ICachedModule` (layers that decode with a KV cache), `RecurrentModule` |
| `Idrak.Models.Abstractions` | `CheckpointFormats` and `ICheckpointFormat`, `ITensorStore`, `IWeightSource`, `WeightCodec` and `WeightFormat`, `GgufTypes`, `GgufArchitectures`, `GgufPreTokenizers`, `PretrainedArchitectures` (with `PretrainedBuildContext`, `DecoderBuildOptions`), `TokenizerComponents` (`ITokenizerNormalizer`, `IPreTokenizer`, `ITokenizerDecoder`); the vision side of vision-language models: `IVisionFamily` and `VisionFamilies` (the library registers no family; `PretrainedVision` and `IVisionTuningPart` are in `Idrak.Abstraction.Generation`, since Idrak.Nlp's tuner uses them too) |
| `Idrak.Onnx.Abstractions` | `OnnxImportOps` and `OnnxImportContext`, `OnnxExportOps` and `OnnxGraph` (with `OnnxValue`, `OnnxAttribute`) |
| `Idrak.Data.Abstractions` | `SampleSources`, `IBatchSource` and `Batch`, `ImageCodecs` and `IImageCodec` (PNG, JPEG, BMP and Netpbm built in), `ImagePreprocessor` (in `Idrak.Data`: a `preprocessor_config.json`'s resize, rescale and normalize, as transformers does them), `IScaler` |
| `Idrak.Training.Abstractions` | `ITrainerCallback`, `TrainerContext`, `TrainingHistory` |

## Backends

The CPU device is in `Idrak.Abstraction`; the CUDA, Vulkan and HIP devices are in `Idrak.Gpu`. This package brings
both, so `dotnet add package Idrak` gives every device.

| Backend | How it works | Needs |
|---------|--------------|-------|
| CPU | SIMD kernels (AVX2, AVX-512, NEON), register-tiled products sized from the CPU's caches, multi-threading measured per machine | .NET 10 |
| CUDA | The NVIDIA driver API directly; PTX kernels generated in C#; bfloat16 and FP8 tensor cores, flash attention, CUDA graphs | An NVIDIA display driver |
| Vulkan | SPIR-V kernels generated in C# for NVIDIA, AMD, Intel and phone GPUs; fused decoding, recorded graphs, cooperative matrices | A Vulkan 1.1+ driver |
| HIP (first slice) | hipRTC kernels for AMD GPUs; the rest through host fallbacks | ROCm or the HIP SDK |

`Device.Default` picks a CUDA GPU when there is one, else the CPU; any device is available by name
(`Device.Parse("vulkan:0")`). Kernel choices come from what each device reports or from measurements on it, stored
per device and driver, never from a card's name.

Tested, every test passing on each: GeForce RTX 5070 Ti, 5050 and 3060 (CUDA and Vulkan), Radeon Vega and Intel UHD
Graphics (Vulkan), an Adreno 730 phone GPU (Vulkan through Mesa Turnip), Snapdragon and Apple M4 Max CPUs (ARM64) and
x64 CPUs.

## Example: predict house prices, written two ways

The library has two ways to write the same code. The **original API** spells out every step; the **simplified API**
makes the same calls with less code (a network builder, data extensions, one training record, a predictor). Both
train the same kind of network with the same settings on the same data.

### Original API

```csharp
using Idrak;
using Idrak.Data;
using Idrak.Layers;
using Idrak.Optimizers;
using Idrak.Training;
using Idrak.Training.Abstractions;

var data = Dataset.LoadCsv("houses.csv", new CsvOptions { TargetColumns = ["price"], IgnoreColumns = ["id"] });
var (train, test) = data.Split(0.8, seed: 1);

var x = StandardScaler.FitFeatures(train);
var y = StandardScaler.FitTargets(train);
var trainLoader = new DataLoader(train.Scale(x, y), batchSize: 64, shuffle: true);
var testLoader  = new DataLoader(test.Scale(x, y), batchSize: 512);

using var model = new Sequential
{
    new Linear(data.FeatureCount, 64), new ReLU(), new Dropout(0.05f),
    new Linear(64, 32), new ReLU(),
    new Linear(32, 1),
};

var trainer = new Trainer(model, new Adam(model.Parameters(), 2e-3f), Losses.MeanSquaredError)
{
    Metrics = { Metric.MeanAbsoluteError },
    EarlyStoppingPatience = 30,             // restores the best weights when it stops
};
TrainingHistory history = trainer.Fit(trainLoader, epochs: 400, validation: testLoader);
float[,] predictions = trainer.Predict(test.Scale(x, y));
```

### Simplified API

```csharp
using Idrak;
using Idrak.Data;
using Idrak.Inference;
using Idrak.Layers;
using Idrak.Optimizers;
using Idrak.Training;
using Idrak.Training.Abstractions;

var data = Dataset.LoadCsv("houses.csv", new CsvOptions { TargetColumns = ["price"], IgnoreColumns = ["id"] });
var split = data.Split(0.8, seed: 1).StandardizeFeatures().StandardizeTargets();   // scalers fitted on the training part

using var model = Network.Input(data.FeatureCount)
    .Linear(64).ReLU().Dropout(0.05f)
    .Linear(32).ReLU()
    .Linear(1)
    .Build();

TrainingHistory history = new TrainingRun
{
    Model = model, Loss = Losses.MeanSquaredError, Optimizer = p => new Adam(p, 2e-3f),
    Train = split.Train.Batches(64, shuffle: true), Validation = split.Test.Batches(512), Epochs = 400,
    Metrics = [Metric.MeanAbsoluteError], EarlyStoppingPatience = 30,   // restores the best weights when it stops
}.Fit();

using var predictor = Predictor.For(model)
    .ScaleInputs(split.FeatureScaler!).UnscaleOutputs(split.TargetScaler!)   // raw features in, dollars out
    .Output(v => v[0])
    .Build();
float price = predictor.Predict([2291, 5, 3, 4, 35.7f, 8, 1, 0, 11952]);
```

`houses.csv` is a table of listings with numeric columns and a `price` column (the repository's HousePrices sample
has one with 2,500 rows).

## Language models

This package loads Llama, Qwen, Mistral, Gemma and mixture-of-experts models from Hugging Face folders or GGUF files,
with their own tokenizers (`PretrainedModel`). The companion package `Idrak.Nlp` generates and chats with them (using
each model's own chat template), fine-tunes them (LoRA, DoRA, QLoRA, preference losses, distillation) and answers from
documents (RAG); Hugging Face ids are downloaded by `Idrak.Data`, which it brings along:

```csharp
using Idrak;
using Idrak.Models;
using Idrak.Nlp;

using var model = PretrainedModel.Load(ModelSource.Resolve("Qwen/Qwen3-0.6B"), new PretrainedOptions { Device = Device.Default, Int8 = true });
var chat = model.CreateChat();
var reply = chat.Chat(new ChatRequest([new ChatMessage("user", "Explain gradient descent in one sentence.")], Think: false));
Console.WriteLine(reply.Message!.Content);
```

## Related packages

| Package | What it adds |
|---------|--------------|
| `Idrak.Gpu` | The CUDA, Vulkan and HIP devices (brought along by this package; usable alone on `Idrak.Abstraction`) |
| `Idrak.Nlp` | Text generation and chat with tools, Jinja chat templates, the engine's text and chat models, LLM fine-tuning and evaluation, retrieval and RAG, a coding agent |
| `Idrak.Data` | JSON Lines, JSON, CSV, text and Parquet files; Hugging Face, GitHub, Kaggle, Zenodo and URL sources |
| `Idrak.Vision` | Region classification in batches, content framing for classifiers of single objects, image statistics |
| `Idrak.AspNetCore` | Prediction, generation, a local chat API and an OpenAI-style `/v1` API in ASP.NET Core |
| `Idrak.Mcp` | Model Context Protocol tools |
| `Idrak.Onnx.Runtime` | Running `.onnx` models with ONNX Runtime as Idrak modules |
| `Idrak.Cli` | The `idrak` command-line tool: checks, chat, serving, benchmarks, fine-tuning, data (`dotnet tool install -g Idrak.Cli`) |

## Documentation

Guides, samples, the full API overview and the results on each tested device: https://github.com/ahmedseada/Idrak

License: Apache 2.0.
