<p align="center"><img src="assets/icon.svg" width="128" alt="Idrak"></p>

# Idrak

> **Idrak** (Arabic: **إدراك**, pronounced *id-RAAK*) means **perception, comprehension, awareness**: the act of
> taking in the world and coming to understand it. The name fits a library whose models learn to perceive patterns in
> data and to understand language.
>
> <p dir="rtl" lang="ar"><b>إدراك</b>: الفهم والوعي، والقدرة على استيعاب الأشياء ومعرفة حقيقتها. اخترنا الاسم لأن المكتبة تُعلِّم النماذج أن تُدرِك الأنماط في البيانات وأن تفهم اللغة.</p>

A self-contained deep-learning library for **.NET 10**, written in C#, with its own GPU kernels. The core has
**no NuGet dependencies and no native libraries**: no TensorFlow, no PyTorch, no CUDA Toolkit, no cuBLAS or cuDNN.

- **Build and train networks**: N-D tensors with automatic differentiation; dense, convolutional, recurrent,
  attention and transformer layers; losses, optimizers (including 8-bit AdamW) and schedules; a trainer, data
  loading, predictors, model packages, telemetry and resource limits.
- **Run and fine-tune language models**: Llama, Qwen, Mistral and Gemma from Hugging Face folders or GGUF files,
  with their own tokenizers and chat templates; streaming chat with reasoning and tool calls; LoRA / QLoRA
  fine-tuning; answer scoring and evaluation.
- **Fast on NVIDIA GPUs**: bfloat16 and FP8 tensor cores, flash attention, int8 / int4 / bfloat16 weights, int8
  KV caches, CUDA graphs; on any CUDA GPU from Maxwell to Blackwell, with CPU fallbacks everywhere.
- **Use it in applications**: an inference engine with batching, ASP.NET Core endpoints (including an
  Ollama-compatible API), retrieval and RAG, MCP tools, ONNX export and import, datasets from files and hubs, and
  two command-line tools (`idrak-tune`, `idrak-data`).

| Backend | How it works | Requirements |
|---------|--------------|--------------|
| **CPU** | `Vector<T>` SIMD kernels (AVX2/AVX-512/NEON), a register-blocked matrix multiply, multi-threading for large tensors, pooled buffers | Any machine running .NET 10 |
| **CUDA** | P/Invoke straight into the NVIDIA **driver** API (`nvcuda.dll` / `libcuda.so.1`). The GPU kernels are PTX assembly generated in C# (`PtxKernels.cs`), and the driver JIT-compiles them for the installed GPU | An NVIDIA GPU and display driver. No CUDA Toolkit, cuBLAS, cuDNN or NVRTC |

`Device.Default` picks the first GPU when a driver is present and falls back to the CPU otherwise.
Set `IDRAK_DISABLE_CUDA=1` to force the CPU.

## Install

**Libraries**: add them to a project (`dotnet add package`, run in the project's folder):

```bash
dotnet add package Idrak                      # tensors, layers, training, CPU and CUDA backends
dotnet add package Idrak.LanguageModels       # Hugging Face and GGUF language models, fine-tuning
dotnet add package Idrak.Datasets             # datasets from files, Hugging Face, GitHub, Kaggle, Zenodo, URLs
dotnet add package Idrak.AspNetCore           # serve models from ASP.NET Core
dotnet add package Idrak.Mcp                  # Model Context Protocol tools
dotnet add package Idrak.Onnx                 # ONNX export
dotnet add package Idrak.Onnx.Runtime         # run ONNX models with ONNX Runtime
```

**Command-line tools**: install once, then run them from any folder (no project needed; the `.Cli` packages are
the tools, the others are libraries):

```bash
dotnet tool install -g Idrak.FineTuning.Cli   # the idrak-tune command: fine-tune, evaluate, chat with and export language models
dotnet tool install -g Idrak.Datasets.Cli     # the idrak-data command: inspect, download and build datasets
idrak-tune --help
idrak-data --help
dotnet tool update -g Idrak.FineTuning.Cli    # later: update to the newest version (same for Idrak.Datasets.Cli)
```

Licensed under the [Apache License 2.0](LICENSE) (see [NOTICE](NOTICE)) from 0.1.7 on; versions up to 0.1.6 were released under the MIT license. Contributions: see [CONTRIBUTING.md](CONTRIBUTING.md) and the [CLA](CLA.md). Releases are listed in the [changelog](CHANGELOG.md).

## Layout

```
src/Idrak/
  Tensor*.cs                        N-D tensors, operators, autograd, shape ops, number-type interop
  Device.cs, ComputeResources.cs    devices; thread and memory budgets
  Autograd.cs, TensorScope.cs       NoGrad(); deterministic disposal
  Losses.cs                         MSE, MAE, CrossEntropy, BinaryCrossEntropy(WithLogits), token log-probabilities
  MixedPrecision.cs, ComputeGraph.cs  tensor-core precision (bfloat16, FP8); CUDA graph capture and replay
  Sampling.cs                       the token sampler (temperature, top-k/p, min-p, penalties)
  Layers/                           Linear, Conv2d, MaxPool2d, GlobalAveragePool2d, Flatten,
                                    BatchNorm, LayerNorm, Embedding, LSTM, GRU,
                                    MultiHeadAttention, TransformerEncoderLayer, PositionalEncoding,
                                    ReLU, Tanh, Sigmoid, GELU, Softmax, Dropout, Lambda, Sequential;
                                    Network builder, Blocks, Architectures; GraphModule (layers in a graph: skip
                                    connections, branches); freezing, LoRA (ModuleExtensions)
  Optimizers/                       Sgd, Adam, AdamW, GroupedOptimizer, schedulers (step, exponential, cosine + warm-up)
  Data/                             Dataset (CSV, class labels, feature shapes), scalers, DataLoader, DataExtensions
  Training/                         Trainer, TrainingRun, Metric (MAE, RMSE, Accuracy), RegressionReport
  Generation/                       tokenizers, TextGenerator (streaming, batches), chat, Conversation, tools
                                    (ToolRegistry), ModelHost, CodingAgent and CodingTools
  Inference/                        Predictor, ModelPackage (.ikm), InferenceEngine
  Retrieval/                        chunking, BM25, TextEncoder (bi-encoder), VectorIndex, RetrievalIndex (hybrid
                                    search with rank fusion), CrossEncoder (re-ranking), Rag pipeline, search tool
  Diagnostics/                      Telemetry hub, events, ConsoleLogger, MetricsRecorder,
                                    ChannelTelemetry, JsonLinesLogger
  Backends/Cpu, Backends/Cuda       device implementations (CPU SIMD kernels, PTX kernels)
src/Idrak.LanguageModels/           optional package, no dependencies: language models and fine-tuning
  PretrainedModel, Architectures    Hugging Face folders; the architecture registry (Llama, Mistral, Qwen2/3, Gemma)
  SafeTensors, Gguf, GgufModel      safetensors and GGUF weights (F32/F16/BF16, Q4_0–Q8_0, K-quants, IQ4)
  BpeTokenizer, Jinja, ChatTemplates  tokenizer.json; the model's own Jinja chat template; tool-call formats
  FineTuning, TuningManifest        LoRA / QLoRA fine-tuning (packing, CUDA graphs, memory fallbacks), adapters
  AnswerScorer, Evaluation          log-probabilities of given answers, answer metrics
  ModelSource                       Hugging Face ids: found in a cache or downloaded once
src/Idrak.Datasets/                 optional package, no dependencies: JSON Lines, JSON, CSV, text, code and Parquet
                                    files (also compressed and archived); Hugging Face, GitHub, Kaggle, Zenodo and URL
                                    sources with a download cache; rows into conversations; recipes
src/Idrak.AspNetCore/               optional package: AddIdrak(), MapPredictor, MapGenerate, MapOllamaApi, MapIdrakStatus
src/Idrak.Mcp/                      optional package: tools of Model Context Protocol servers, and serving tools over MCP
src/Idrak.Onnx/                     optional package, no dependencies: export networks to .onnx (opset 17), import .onnx into layers
src/Idrak.Onnx.Runtime/             optional package: run .onnx models with ONNX Runtime as Idrak modules
src/Idrak.FineTuning.Cli/           idrak-tune: fine-tune any language model the library loads (LoRA / QLoRA), evaluate, chat, export
src/Idrak.Datasets.Cli/             idrak-data: inspect, download and assemble training datasets
samples/
  Idrak.Samples.Xor                 the classic XOR problem
  Idrak.Samples.HousePrices         regression: predict house prices from a CSV file
  Idrak.Samples.HouseApi            house-price Web API in a few lines (Idrak.AspNetCore + the HousePrices package)
  Idrak.Samples.Spirals             multi-class: 3 spirals, softmax + cross-entropy, BatchNorm
  Idrak.Samples.ShapeRecognition    CNN: classify drawn shapes (Conv2d, MaxPool2d, BatchNorm)
  Idrak.Samples.Ocr                 OCR: CNN character recognizer + line segmentation, reads PGM images
  Idrak.Samples.Sentiment           sentiment with negation: bag-of-words vs LSTM, GRU, Transformer
  Idrak.Samples.TextGeneration      small GPT: character-level causal transformer that generates text
  Idrak.Samples.GptApi              ASP.NET Core Web API + browser UI serving the GPT (Scalar docs, streaming, Ollama-style /api/chat)
  Idrak.Samples.GptTraining         trains a full-size character-level GPT (bfloat16 / FP8 tensor cores, checkpoints, resume)
  Idrak.Samples.Summarizer          summarization: extractive baselines vs a word-level transformer (WordTokenizer + TextGenerator)
  Idrak.Samples.ReRanker            search re-ranking: BM25 first stage + transformer cross-encoder, listwise training
  Idrak.Samples.Rag                 retrieval-augmented generation: hybrid search, re-ranking, a chat model that cites passages
  Idrak.Samples.Quantization        int8 weights and Float16/BFloat16 files: accuracy, size and decoding speed
  Idrak.Samples.OnnxImport          imports another framework's .onnx model, runs it on Idrak (CPU/CUDA), checks its outputs
  Idrak.Samples.Chat                chat with a Hugging Face or GGUF language model: info, chat, profile, and a check against transformers
  Idrak.Samples.CodingAgent         a coding agent (read, search, edit, run commands) on a language model, and a task-suite runner
  Shared/SampleOptions.cs           command-line options shared by the samples (train / predict modes)
  Shared/Gpt/                       GPT model, generation with metrics, training (console + Web API)
tests/Idrak.Tests                   self-contained test runner: every test on the CPU and on every CUDA GPU present
  data/                             small GGUF, safetensors and Parquet fixtures (made by the scripts in tools/)
tools/
  pytorch/xor_to_onnx.py            trains XOR in PyTorch and exports it to ONNX with PyTorch's outputs, for OnnxImport
  pytorch/export_models.py          exports a PyTorch CNN or ResNet (skip connections) to ONNX with PyTorch's outputs
  pytorch/pretrained_reference.py   records transformers' ids, templates, logits and greedy output, for `Chat check`
  gguf/make_fixtures.py             the GGUF test fixtures (every quantization type, tiny Llama and Qwen3 models)
  datasets/make_parquet_fixtures.py the Parquet test fixtures, with the rows pyarrow reads
  publish.ps1                       publishes a release tag to nuget.org from a machine (when CI cannot)
docs/                               performance notes (where optimization stopped) and items to review
assets/                             the icon (icon.svg source, icon.png for the packages)
```

## Samples

| Sample | Shows | Typical CPU result |
|--------|-------|--------------------|
| `Xor` | smallest possible network | 4/4 correct in 0.1 s |
| `HousePrices` | CSV loading, scaling, regression, early stopping | R² 0.975, 5.3% mean error, 1 s |
| `Spirals` | softmax + cross-entropy, BatchNorm, AdamW, cosine schedule, confusion matrix | 97.8% accuracy, 2.4 s |
| `ShapeRecognition` | Conv2d, MaxPool2d, BatchNorm, Flatten on 16×16 images | 100% accuracy, about 11 s |
| `Sentiment` | Embedding, LSTM, GRU, Transformer vs an order-blind baseline | LSTM/GRU/Transformer 99.5–100%, bag of words 70% |
| `Ocr` | CNN over 36 characters, projection-profile segmentation, PGM input | 100% per character, 99.9% of characters across whole lines |
| `TextGeneration` | decoder-only GPT: causal attention, sparse cross-entropy, sampling | 89% next-character accuracy, 100% real words generated |
| `GptApi` | serving a model: REST + server-sent events, Scalar, browser UI, Ollama-compatible `/api/chat` | about 600 characters/s on 4 CPU cores |
| `ReRanker` | two-stage search: BM25 + cross-encoder, hard negatives, listwise loss, placeholder tokens for unseen names | Hit@1 on unseen towns 27.1% (BM25) → 86.6% re-ranked, 6.9 ms per question, 180 s training |
| `HouseApi` | `AddIdrak().AddPredictor(...)` + `MapPredictor`: the HousePrices package served over HTTP with micro-batching | same prices as `HousePrices --predict` |
| `Summarizer` | word-level decoder-only transformer, loss masking, greedy generation with a stop token, ROUGE | ROUGE-1 0.999 vs 0.503 (first sentence), 90% exact, 7.7 ms per summary |
| `Quantization` | `QuantizeInt8`, int8 KV cache, Float16/BFloat16 files: a trained summarizer compared with float32, and decoding speed and memory of a 98M-parameter GPT | int8 weights + int8 KV cache: same summaries as float32 on all 300 test reports, ⅓ of the file; on 4 CPU threads, weights 373 → 112 MB, KV cache 18 → 4.8 MB, decoding 19.8 → 45.7 tokens/s |
| `OnnxImport` | `OnnxImport.Load(path, device)` on a PyTorch-exported model (`tools/pytorch/xor_to_onnx.py`, `export_models.py` for a CNN or ResNet), compared with PyTorch's own outputs, then saved as .ikm and reloaded | XOR: same outputs as PyTorch on the RTX 5050 (1e-11), both PyTorch exporters |
| `Rag` | `RetrievalIndex` (BM25 + trained bi-encoder + rank fusion), `CrossEncoder` re-ranking, `Rag.For(chat)` with a word-level ChatML model that cites passages; hashing and placeholder tokens for unseen names | unseen towns: Hit@1 27.1% (BM25), 85.8% (hybrid), 99.9% (re-ranked); answers 91.1% correct (0% closed book), 99.9% cite the right passage; 14 min training |
| `Chat` | `PretrainedModel.Load(folder)` on a Hugging Face model folder; `chat` in the model's own template (reasoning, tool calls); `check` against a reference from `tools/pytorch/pretrained_reference.py` (token ids, chat templates, logits, greedy output) | on small Qwen3- and Llama-layout models built here (trained `tokenizers` tokenizers, transformers' `apply_chat_template`, a NumPy port of the Hugging Face forward pass): identical ids, templates and greedy text, logits within 1e-5 (F32 and BF16 files, sharded, SentencePiece and byte-level) |
| `CodingAgent` | `CodingAgent` + `CodingTools` on any model `PretrainedModel` loads: `agent` works on a task in a folder (streams reasoning and tool calls); `agent-run` scores a model on a suite of tasks, each verified by its own commands; `agent-check` checks a suite without a model | runs every task in a fresh copy of its workspace |

### Train and predict modes

Every console sample trains, saves its model under `models/` (next to the executable), and can then run
**inference only** from the saved model:

```bash
dotnet run -c Release --project samples/Idrak.Samples.HousePrices                       # train + save
dotnet run -c Release --project samples/Idrak.Samples.HousePrices -- --predict --input "2100,4,2,15,9.5,7,2,0,6500"
dotnet run -c Release --project samples/Idrak.Samples.Sentiment -- --predict --input "the movie was not good;not bad at all"
dotnet run -c Release --project samples/Idrak.Samples.Ocr -- --predict --input "HELLO WORLD 2026"
dotnet run -c Release --project samples/Idrak.Samples.Ocr -- --predict --image scan.pgm
dotnet run -c Release --project samples/Idrak.Samples.ReRanker -- --predict --input "how many people live in armor"
dotnet run -c Release --project samples/Idrak.Samples.Summarizer -- --predict --input "the lions played the owls in kelso on friday . the owls scored 2 goals . the lions scored 4 goals ."
dotnet run -c Release --project samples/Idrak.Samples.TextGeneration -- --predict --input "the old wizard " --temperature 0.8
```

| Sample | `--input` in predict mode |
|--------|---------------------------|
| `Xor` | `"a,b;a,b"` bit pairs |
| `HousePrices` | 9 features per house, `;` between houses (or `--data file.csv` to price every row) |
| `Spirals` | `"x,y;x,y"` points in [-1, 1] |
| `ShapeRecognition` | shape names to draw and classify, e.g. `"circle,cross"` |
| `Sentiment` | sentences separated by `;`, judged by all four saved models side by side |
| `Ocr` | text to render and read (or `--image file.pgm`) |
| `TextGeneration` | prompt to continue (`--length`, `--temperature`, `--top-k`) |

Everything a model needs at inference time is saved next to its weights. That includes the scalers for
house prices, the BatchNorm running statistics, and the GPT's vocabulary and architecture (a `.json`
file). `--model <path>` chooses another location.

### GPT Web API

```bash
dotnet run -c Release --project samples/Idrak.Samples.GptApi
# http://localhost:5080         inference UI
# http://localhost:5080/scalar  API reference (Scalar)
```

| Endpoint | Purpose |
|----------|---------|
| `GET /api/status` | Loading / Training (with live progress) / Ready / Failed |
| `GET /api/model` | parameters, blocks, heads, width, context, vocabulary, validation results, layer summary |
| `GET /api/devices` | CPU and CUDA GPUs with memory usage and the active one |
| `POST /api/generate` | one or more samples, each character's probability, entropy, top-5 alternatives and latency, plus aggregate metrics (including decoding mode) |
| `POST /api/generate/stream` | the same as server-sent events: `token` events (with their sample index) every `chunkSize` characters, then `metrics` |

The request body is `{ "prompt", "length", "temperature", "topK", "seed", "device": "cpu" | "cuda",
"samples", "useCache", "useGraph", "chunkSize" }`. The UI has matching controls: a samples slider
(shown as tabs), and KV cache and CUDA graph switches for comparing modes.
The device can change per request, and the model moves between CPU and GPU as needed.

The service loads `Gpt:ModelPath` from `appsettings.json`. Point it at a model saved by the
`TextGeneration` sample (or pass `--Gpt:ModelPath path`). When no model exists, the service trains one in
the background and reports progress through Idrak telemetry.

The browser UI (`wwwroot/index.html`, dark and light themes, responsive) is laid out in two columns:
- **Left:** the streamed output, with each character coloured by the model's confidence (hover a
  character for its alternatives, click it for the token inspector); metric cards for throughput,
  total time, first-token latency, time per token, confidence, perplexity, entropy and device
  memory; and live confidence and latency charts.
- **Right:** prompt, length, temperature, top-k, seed and streaming controls; the CPU/GPU switch; and
  the model's parameters and layer summary.

All samples take the same options:

```bash
dotnet run -c Release --project samples/Idrak.Samples.HousePrices               # GPU if available, else CPU
dotnet run -c Release --project samples/Idrak.Samples.HousePrices -- --cpu      # force CPU
dotnet run -c Release --project samples/Idrak.Samples.HousePrices -- --cuda     # force GPU
dotnet run -c Release --project samples/Idrak.Samples.Xor -- --help
```

| Option | Meaning |
|--------|---------|
| `--device auto\|cpu\|cuda\|cuda:N`, `--cpu`, `--cuda` | where to run |
| `--threads N` | CPU threads to use (default: all cores) |
| `--gpu-memory MiB`, `--cpu-memory MiB` | cap tensor memory (default: unlimited) |
| `--epochs N`, `--batch-size N` | training settings |
| `--log training,batches,gradients,layers,operations,inference,all` | what the console logger prints |
| `--log-file run.jsonl` | also write telemetry as JSON Lines |
| `--data path.csv` | dataset file (house prices) |

The house-price sample loads `data/houses.csv`: 2,500 synthetic but realistic listings with 9 features
(area, bedrooms, bathrooms, age, distance to the city centre, quality, garage, pool, lot size).
It splits them 80/20, standardizes features and prices, and trains a 9→64→32→1 network with
dropout, Adam and early stopping. It then reports test error in dollars, prices three new listings,
and saves the weights, scalers and a training-history CSV. A typical CPU run reaches **R² ≈ 0.97 and
about 5% mean error** in about a second, close to the ±6% noise built into the data.

## Using the library

### Training with the `Trainer`

```csharp
using Idrak;
using Idrak.Data;
using Idrak.Layers;
using Idrak.Optimizers;
using Idrak.Training;

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

#### Callbacks

`Callbacks` run inside `Fit` at fixed points: `OnTrainBegin`, `OnBatchEnd`, `OnEpochEnd` and `OnTrainEnd` (all
optional). Each receives a `TrainerContext` (model, optimizer, epoch, step, history so far, the cancellation token) and
can call `context.Stop()` to end training cleanly after the current batch or epoch. Batch losses are only read from the
device when a callback sets `NeedsBatchLoss`, so callbacks add no synchronization by default. Three are built in:

```csharp
var trainer = new Trainer(model, new Adam(model.Parameters(), 2e-3f), Losses.MeanSquaredError)
{
    Metrics = { Metric.MeanAbsoluteError },
    Callbacks =
    {
        new EarlyStopping(patience: 30),           // = EarlyStoppingPatience = 30; also monitor: "val_mae", maximize: ...
        new Checkpoint("checkpoints"),             // checkpoints/last.ikw every epoch, best.ikw for the best one (Module.Load)
        new CsvLog("training.csv"),                // epoch, loss, mae, val_loss, val_mae, learning_rate
        new StopAtLoss(0.01),                      // your own: an ITrainerCallback
    },
};
TrainingHistory history = trainer.Fit(trainLoader, epochs: 400, validation: testLoader);

sealed class StopAtLoss(double target) : ITrainerCallback
{
    public void OnEpochEnd(TrainerContext context, EpochCompleted epoch)
    {
        if (epoch.ValidationLoss < target) context.Stop();
    }
}
```

### Low-level: tensors and a hand-written loop

```csharp
var a = Tensor.From(new float[,] { { 1, 2 }, { 3, 4 } }, requiresGrad: true);
var loss = (a.MatMul(a).Relu() * 2f + 1f).Mean();
loss.Backward();                      // a.Grad holds d(loss)/da

for (int epoch = 0; epoch < 1000; epoch++)
{
    using var scope = new TensorScope();   // releases this iteration's tensors
    var l = Losses.MeanSquaredError(model.Forward(inputs), targets);
    optimizer.ZeroGrad();
    l.Backward();
    optimizer.Step();
}

using var prediction = model.Predict(inputs);   // evaluation mode, no gradients
```

### Classification

Classifiers output raw scores (logits). `CrossEntropy` applies log-softmax itself, which is more
stable than a separate softmax layer. Add `Softmax` only when you want probabilities at inference time.

```csharp
var data = Dataset.FromClassLabels(features, labels, classes: 3);      // one-hot targets
// or: Dataset.LoadCsv(path, new CsvOptions { TargetColumns = ["label"] }).ToOneHot(3)

var trainer = new Trainer(model, new AdamW(model.Parameters(), 1e-3f), (p, t) => Losses.CrossEntropy(p, t))
{
    Metrics = { Metric.Accuracy },
    Scheduler = new CosineAnnealing(optimizer, totalEpochs: 100, warmupEpochs: 5),
};
using var probabilities = model.Predict(x).Softmax();
using var classes = probabilities.ArgMax();
```

For two classes with one output column, use `Losses.BinaryCrossEntropyWithLogits` and
`Metric.BinaryAccuracy(threshold: 0)`.

### Images (CNN)

```csharp
var images = Dataset.FromClassLabels(pixels, labels, 10).WithFeatureShape(1, 28, 28);   // batches are [N, 1, 28, 28]
var cnn = new Sequential
{
    new Conv2d(1, 32, kernelSize: 3, padding: 1), new BatchNorm(32), new ReLU(), new MaxPool2d(2),
    new Conv2d(32, 64, kernelSize: 3, padding: 1), new BatchNorm(64), new ReLU(), new MaxPool2d(2),
    new Flatten(), new Linear(64 * 7 * 7, 128), new ReLU(), new Dropout(0.3f), new Linear(128, 10),
};
```

`Conv2d` unfolds image patches (im2col) and runs one large matrix product on the same optimized GEMM
as `Linear`, on both CPU and GPU.

### Sequences (RNN and transformer)

Token ids go in as floats, shape [batch, time]:

```csharp
var lstm = new Sequential
{
    new Embedding(vocabulary, 64), new LSTM(64, 128), new Linear(128, classes),   // LSTM returns the last state
};

var transformer = new Sequential
{
    new Embedding(vocabulary, 64), new PositionalEncoding(maxLength, 64),
    new TransformerEncoderLayer(64, heads: 4), new TransformerEncoderLayer(64, heads: 4),
    new LayerNorm(64), new Lambda(x => x.Mean(1), "MeanOverTime"), new Linear(64, classes),
};
```

Use `LSTM(..., returnSequences: true)` or `GRU` for per-step outputs. `MultiHeadAttention(dim, heads, causal: true)`
masks future positions for autoregressive models. Setting `Trainer.MaxGradientNorm` clips gradients,
which recurrent networks usually need.

### Tensor operations

N-D tensors support batched `MatMul` (with transpose flags), `Permute`, `Transpose`, `Narrow`,
`Tensor.Concat`, `Tensor.Stack`, `Flatten`, `Sum(dim)`, `Mean(dim)`, `Softmax`, `LogSoftmax`,
`ArgMax`, `Exp`, `Log` and `Gelu`. Adding a tensor whose shape matches the trailing dimensions
broadcasts it, as with a bias [F] over [N, F] or a mask [T, T] over [B, T, T]. All of these are
differentiated automatically.

### Other number types

Every kernel computes in float32, the standard for neural networks and the only type with fast
SIMD and GPU paths everywhere. Data of any numeric type converts at the edges:

```csharp
using var t = Tensor.From<double>(doubles, [rows, cols]);   // also int, long, byte, Half, decimal, ...
using var m = Tensor.From(new int[,] { { 1, 2 }, { 3, 4 } });
int[] ints = t.ToArray<int>();                               // saturating, truncating conversion
```

### Fast autoregressive generation

Generating text one token at a time is dominated by overhead rather than math, so Idrak provides
three tools that work together (`CharGpt.Generate` in `samples/Shared/Gpt` shows them in use):

* **KV cache.** Pass a `DecodingContext` to `Sequential.ForwardCached(ids, context)`. The prompt is
  processed once (prefill), then every step processes only the newest token, while each attention
  layer's keys and values accumulate in a `KeyValueCache`. `MultiHeadAttention`,
  `TransformerEncoderLayer` and `PositionalEncoding` implement `ICachedModule`.
* **On-device sampling.** `TokenSampler` draws the next token (temperature, top-k, top-p, min-p,
  repeat/presence/frequency penalties over a device-side history of recent tokens) on the device
  that holds the logits and writes each token's probability, entropy and top-5 alternatives to a
  device buffer. The next step therefore needs no host round trip; `Read` fetches the statistics in
  chunks. Its randomness is counter-based (seed, step, row), so results are reproducible and match
  between CPU and GPU.
* **Compute graphs.** `DecodingContext.CaptureStep(step)` records a whole decoding step, and
  `ReplayStep` runs it again with one launch; on CUDA this is a CUDA Graph. The position counter,
  causal mask and cache offsets are computed on the device, so one recording is valid at every
  position. On the CPU, or if the driver refuses, replay simply re-runs the step.
* **Batching and fused kernels.** Many samples decode together as one batch. During inference,
  scale+mask+softmax, LayerNorm and bias+GELU each run as a single kernel; training uses the
  differentiable path.

Measured on the sample GPT (341K parameters) on a 4-core CPU container
(`dotnet run --project samples/Idrak.Samples.TextGeneration -- --predict --benchmark true`):

| Mode | chars/s | Speedup |
|------|---------|---------|
| Full recompute, 1 sample | 283 | 1× |
| KV cache, 1 sample | 4,119 | 14.5× |
| KV cache, 8 samples | 6,595 | 23× |
| KV cache, 32 samples | 8,957 | 32× |

On a GPU, CUDA graphs remove the per-step launch cost, and batching keeps the GPU busy.

### Text generation, chat and tools (`Idrak.Generation`)

A higher-level layer on top of the cache, sampler and graphs, with the options and conventions of common
local LLM servers:

| Type | Purpose |
|------|---------|
| `ITokenizer`, `CharTokenizer`, `WordTokenizer` | text ↔ token ids (characters; words, numbers, punctuation and `<special>` tokens) |
| `GenerationOptions` | `Temperature`, `TopK`, `TopP`, `MinP`, `RepeatPenalty`, `RepeatLastN`, `PresencePenalty`, `FrequencyPenalty`, `Seed`, `NumCtx`, `NumPredict`, `Stop`, plus `UseCache`, `UseGraph`, `ChunkSize` |
| `TextGenerator` | streams a continuation: prompt truncated to `NumCtx`, sliding context window, stop sequences (never partially emitted), done reason `stop` / `length`, prompt and generation timings |
| `ChatMessage`, `ToolDefinition`, `ToolCall` | conversations with `system`, `user`, `assistant` and `tool` roles, and function tools |
| `ChatTemplate`, `ChatMLTemplate` | renders a conversation and its tools as the prompt (Qwen-style ChatML: `<think>`, `<tool_call>`, `<tool_response>`); `think: false` closes an empty reasoning block |
| `ChatOutputParser` | splits streamed output into reasoning, answer and tool calls (JSON), holding back partial tags |
| `ChatGenerator` | chat = template + generator + parser; streams `ChatChunk`s and ends with the full assistant message and statistics |
| `ModelHost<T>`, `KeepAlive` | keeps models loaded and unloads each one when its keep-alive (`"30m"`, `"1h30m"`, `300`, `0`, `-1`) expires |

```csharp
var chat = new ChatGenerator(new TextGenerator(model, new CharTokenizer(vocabulary), contextLength: 256));
var request = new ChatRequest(
    [new ChatMessage("system", "You are a helpful assistant."), new ChatMessage("user", "What is the latest Ollama version?")],
    Tools: [new ToolDefinition("web_fetch", "Fetch a page.", JsonNode.Parse("""{"type":"object","properties":{"url":{"type":"string"}}}"""))],
    Think: true,
    Options: new GenerationOptions { Temperature = 1f, TopK = 20, TopP = 0.95f, NumCtx = 4096, NumPredict = 2048 });
foreach (var chunk in chat.Stream(request))
    Console.Write(chunk.Delta.Thinking + chunk.Delta.Content);   // chunk.Delta.ToolCalls: completed tool calls
```

### Ollama-compatible chat API

The GPT Web API also serves `POST /api/chat` with the Ollama request body: `model` (any name selects the
served model), `messages`, `stream` (NDJSON, default true), `think` (true/false or low/medium/high),
`keep_alive`, `options` (the keys above in snake_case; other keys are accepted and ignored) and `tools`.
Replies have Ollama's shape: `message.content`, `message.thinking`, `message.tool_calls`, `done`,
`done_reason` and nanosecond `total_duration`, `load_duration`, `prompt_eval_count`,
`prompt_eval_duration`, `eval_count`, `eval_duration`. `GET /api/tags`, `GET /api/ps` (loaded models with
`expires_at`) and `GET /api/version` are there too.

```bash
curl http://localhost:5080/api/chat -d '{"model":"any","stream":true,"think":true,"keep_alive":"30m",
  "options":{"temperature":1,"top_k":20,"top_p":0.95,"num_ctx":4096,"num_predict":2048},
  "messages":[{"role":"user","content":"What is the latest Ollama version?"}],
  "tools":[{"type":"function","function":{"name":"web_fetch","parameters":{"type":"object","properties":{"url":{"type":"string"}}}}}]}'
```

The GptApi sample maps these endpoints with `app.MapOllamaApi(...)` from `Idrak.AspNetCore` (see
below); the request body is read as JSON whatever its Content-Type, as Ollama does. The endpoint serves
`Gpt:ChatModelPath` if that file exists, otherwise `Gpt:ModelPath`. A model trained on
plain text only continues text. To see reasoning and tool calls, train the small chat model on synthetic
ChatML transcripts (reasoning, `web_fetch` calls, answers citing tool results):

```bash
dotnet run -c Release --project samples/Idrak.Samples.TextGeneration -- --chat true   # saves models/chat.weights and runs a two-turn demo
```

### Optimizers and schedules

`Sgd(momentum, weightDecay)`, `Adam(weightDecay)` (L2) and `AdamW` (decoupled weight decay) are
available, together with the `StepDecay`, `ExponentialDecay`, `CosineAnnealing(warmupEpochs)` and
`LambdaSchedule` schedulers. `optimizer.ClipGradientNorm(max)` and `optimizer.GradientNorm()` work
in hand-written loops too.

## Two ways to write it: the original API and the simplified API

Everything above keeps working exactly as shown. The simplified API is a second way to write the same
things with less code: each builder step or extension method makes the same calls you would write by
hand, with the same parameters and the same defaults. Nothing is chosen for you, and whatever the
original API requires is still required. The tests check that both ways give identical weights,
outputs and training histories.

### Networks: a fluent builder (the input size of each layer comes from the previous layer)

```csharp
// original
var model = new Sequential
{
    new Linear(9, 64, random: r), new ReLU(), new Dropout(0.05f, r),
    new Linear(64, 32, random: r), new ReLU(),
    new Linear(32, 1, random: r),
};

// simplified: same layers, same weights with the same seed
var model = Network.Input(9).Seed(1).Linear(64).ReLU().Dropout(0.05f).Linear(32).ReLU().Linear(1).Build();

var cnn = Network.Image(1, 16, 16)
    .Conv2d(16, kernelSize: 3, padding: 1).BatchNorm().ReLU().MaxPool2d(2)
    .Conv2d(32, kernelSize: 3, padding: 1).BatchNorm().ReLU().MaxPool2d(2)
    .Flatten().Linear(4)                                   // 32·4·4 inputs worked out for you
    .Build();

var gpt = Architectures.Gpt(vocabulary: 76, context: 64, dim: 96, heads: 4, layers: 3, ffDim: 384, dropout: 0.1f).Build();
var layers = Blocks.Repeat(3, () => new TransformerEncoderLayer(96, 4, 384, 0.1f, causal: true));   // in a new Sequential { ... }
```

`Network.Input / Image / Tokens / Sequence` start a builder; every layer has a method (`Linear`, `Conv2d`,
`LSTM`, `GRU`, `TransformerEncoderLayer`, `Embedding`, `PositionalEncoding`, `LayerNorm`, …) plus
`MeanOverTime`, `LastStep`, `FirstStep`, `Reshape`, `Lambda(fn, name, outputShape)` and `Add(module, outputShape)`.
Shape mistakes fail while building, with a clear message. `ToJson()` / `Network.FromJson(...)` describe and
replay a network; packages use this to store the architecture.

### Data: extension methods

```csharp
// original
var (train, test) = data.Split(0.8, seed: 1);
var fs = StandardScaler.FitFeatures(train);
var ts = StandardScaler.FitTargets(train);
var trainLoader = new DataLoader(train.Scale(fs, ts), 64, shuffle: true);

// simplified: fitted on the training part, applied to both parts
var split = data.Split(0.8, seed: 1).StandardizeFeatures().StandardizeTargets();   // or NormalizeFeatures()
var trainLoader = split.Train.Batches(64, shuffle: true);                          // = new DataLoader(...)
// split.Train, split.Test, split.FeatureScaler, split.TargetScaler
```

### Duplicates: removed before a split unless you keep them

`Split` deduplicates the rows first, so no row lands in both parts and the test score is on rows the model has not
seen. `removeDuplicates: false` turns that off and keeps every row, identical copies included (augmentation, or when
how often a row repeats matters). The rules, and `Deduplicate` on its own:

```csharp
var (train, test) = data.Split(0.8, seed: 1);                                   // deduplicated: same features count once
var (all, held)   = data.Split(0.8, seed: 1, removeDuplicates: false);          // every row, copies included

var split = data.Split(0.8, seed: 1, duplicates: new DeduplicationOptions
{
    Match = DuplicateMatch.FeaturesAndTargets,         // or Features (the default): same features whatever the targets
    Conflicts = DuplicateConflicts.KeepMostFrequent,   // same features, other targets: KeepFirst (default), KeepMostFrequent, DropAll
    Embedding = row => embeddings[row],                // optional near duplicates: your vectors, cosine >= SimilarityThreshold
    SimilarityThreshold = 0.95f,
});

DeduplicationResult report = data.DeduplicateWithReport();   // .Data, .Kept (row indices, to filter texts kept alongside),
                                                              // .ExactDuplicates, .Conflicting, .NearDuplicates
```

Exact duplicates cost one hash per row (in parallel) and no allocation per row: 300,000 rows of 256 features take
about 0.2 s on 4 cores, and a dataset without duplicates is returned as is (no copy). Near duplicates compare blocks of
rows with the kept rows as matrix products on the CPU kernels: 20,000 embeddings of 384 values take about 2 s on 4
cores, growing with rows × kept rows.

### Training: factories for the wiring, and one settings object

```csharp
// the trainer creates the optimizer and schedule from factories, and disposes them
using var trainer = new Trainer(model, Losses.CrossEntropy,
    optimizer: p => new AdamW(p, 0.003f, weightDecay: 1e-4f),
    scheduler: o => new CosineAnnealing(o, 40, warmupEpochs: 1));

// or everything as one record; the required members are what Trainer and Fit require
var run = new TrainingRun
{
    Model = model, Loss = Losses.CrossEntropy, Optimizer = p => new AdamW(p, 0.003f),
    Train = split.Train.Batches(64, shuffle: true), Validation = split.Test.Batches(500), Epochs = 40,
    Metrics = [Metric.Accuracy], MaxGradientNorm = 1f,
    Callbacks = [new EarlyStopping(5), new Checkpoint("checkpoints")],   // = trainer.Callbacks
};
TrainingHistory history = run.Fit();
var clipped = run with { MaxGradientNorm = 0.5f };                    // a variant (give it a fresh Model)

await run.FitAsync(new Progress<EpochCompleted>(e => label.Text = $"epoch {e.Epoch}: {e.Loss:F4}"), token);
await foreach (var e in run.TrainAsync()) { if (e.ValidationLoss < 0.01) break; }   // leaving the loop stops training
```

### Telemetry: one builder, one disposable

```csharp
await using var telemetry = Telemetry.Configure()
    .Console(TelemetryLevel.Training, epochInterval: 10)                 // = new ConsoleLogger(...)
    .JsonLines("run.jsonl", TelemetryLevel.All & ~TelemetryLevel.Operations)
    .Record(out var recorder)                                            // = new MetricsRecorder()
    .Start();
```

### Predicting: a predictor instead of scale → tensor → predict → unscale

```csharp
var predictor = Predictor.For(model)
    .Input<House>(h => [h.Area, h.Beds, h.Baths, h.Age, h.Distance, h.Quality, h.Garage, h.Pool, h.Lot])
    .ScaleInputs(featureScaler)
    .UnscaleOutputs(priceScaler)
    .Output(v => v[0])
    .Build();
float price = predictor.Predict(house);                      // or Predict(listOfHouses): one batched call

var classifier = Predictor.For(model).Input<string>(Encode).Softmax().Classes(["negative", "positive"]).Build();
ClassPrediction answer = classifier.Predict("not bad at all");   // .Class, .Probability, .Scores

predictor.Save("house-price.ikm");                            // weights, architecture (if built with Network), scalers, settings
```

### Model packages: one file instead of several

```csharp
ModelPackage.Create("model.ikm")
    .Architecture(network).Weights(model)                    // the existing formats, zipped together
    .Scaler("features", featureScaler).Tokenizer("words", tokenizer).Json("settings", settings)
    .Save();

using var package = ModelPackage.Open("model.ikm");
using var model = package.BuildNetwork();                   // no layer code needed
var generator = package.TextGenerator("words");              // model + tokenizer + context from one file
```

`MinMaxScaler`, `CharTokenizer` and `WordTokenizer` now have `Save` / `Load` too.

### The inference engine: one reusable place for loading, batching and serving

```csharp
await using var engine = await InferenceEngine.Create()
    .Predictor<House, float>("house-price", "models/house-price.ikm", p => p
        .Input<House>(h => [h.Area, h.Beds, h.Baths, h.Age, h.Distance, h.Quality, h.Garage, h.Pool, h.Lot])
        .Output(v => v[0])
        .WarmUp(sampleHouse)                                           // each feature is off unless set
        .Batching(maxBatch: 256, maxWait: TimeSpan.FromMilliseconds(5)))
    .ChatModel("my-gpt", "models/chat.ikm", "chars", c => c
        .Instances(2)                                                  // two generations at once
        .KeepAlive(TimeSpan.FromMinutes(5)).QueueLimit(32).Timeout(TimeSpan.FromSeconds(30)))
    .Telemetry()
    .BuildAsync();                                                     // or .LoadOnFirstUse()

float p = await engine.PredictAsync<House, float>("house-price", house);
await foreach (var chunk in engine.StreamAsync("my-gpt", "Once upon a time", options)) Console.Write(chunk.Text);
await foreach (var result in engine.PredictManyAsync<House, float>("house-price", millionsOfHouses, batchSize: 1024)) { … }
var models = engine.Models;                                            // loaded state, running and queued requests, expiry
var stats = engine.Stats("house-price");                               // requests, latency (average, p95), rows/s, batch size
```

Predictors run concurrently (they are thread-safe); a text or chat copy runs one generation at a time
(it owns its KV cache). Models come from a package, a factory, a model object or a network builder
plus a weights file.

### Chat: conversations and tools

```csharp
public sealed class ReleaseTools(HttpClient http)
{
    [Tool("latest_release", "Fetch the release notes page of a product.")]
    public Task<string> LatestAsync([Description("Product name, e.g. ollama.")] string product) =>
        http.GetStringAsync($"https://{product}.com/releases");
}

var tools = ToolRegistry.Create()
    .Add(new ReleaseTools(http))                                        // every [Tool] method; schema from the parameters
    .Add(WebTools.Fetch(http, allow: u => u.Host == "ollama.com", maxCharacters: 4000))   // built-in, allowlist required
    .Allow("web_fetch", args => ((string)args["url"]!).StartsWith("https://ollama.com"))
    .RequireApproval("delete_file", (call, token) => AskUserAsync(call, token))
    .Timeout(TimeSpan.FromSeconds(10)).Parallel()
    .Build();

var conversation = Conversation.For(chatGenerator)       // or engine.Conversation("my-gpt", c => ...)
    .System("You are a helpful assistant. Cite sources as [1], [2].")
    .Think(true).Tools(tools).MaxToolRounds(5)             // MaxToolRounds is required when tools are added
    .Build();
var reply = await conversation.SendAsync("What is the latest Ollama version?");   // runs the tool loop
```

Arguments are validated against each tool's schema before the tool runs; mistakes go back to the model as
an error it can correct. `FakeChatModel.Script(...)` replays scripted replies for testing tool code, and
`TextGenerator.StreamAsync` / `ChatGenerator.StreamAsync` stream with `await foreach`.

### ASP.NET Core: the optional `Idrak.AspNetCore` package

```csharp
builder.Services.AddIdrak()
    .AddPredictor<House, float>("house-price", "models/house-price.ikm", p => p.Input<House>(h => [...]).Output(v => v[0]))
    .AddChatModel("my-gpt", "models/chat.ikm", "chars", c => c.KeepAlive(TimeSpan.FromMinutes(5)));

app.MapPredictor<House, float>("/predict/house-price", "house-price");      // POST a House (or /batch an array)
app.MapGenerate("/api/generate", "my-gpt");                                 // JSON, or server-sent events with "stream": true
app.MapOllamaApi("/api", "my-gpt", o => o.Tools(ToolExecution.Client));    // or ToolExecution.Server with maxRounds
app.MapIdrakStatus("/status");
```

The endpoints are ordinary ASP.NET Core endpoints (`.RequireAuthorization()`, rate limiting and OpenAPI work
as usual), and `IPredictor<TIn, TOut>` can be injected (keyed by model name). The GptApi sample serves its
Ollama-compatible API this way, and the HouseApi sample is a complete prediction API in about ten lines.

### Retrieval and RAG (`Idrak.Retrieval`)

These are new building blocks; the original way is writing the search, the scoring loop and the prompt by hand.

```csharp
var encoder = new TextEncoder(model, tokenizer, maxLength: 20, padId: 0);     // token ids → unit vectors (masked mean)
encoder.Train(pairs, epochs: 20, batchSize: 64, p => new AdamW(p, 2e-3f), temperature: 0.05f, seed: 3);   // in-batch negatives

var index = RetrievalIndex.Create()
    .Documents(documents, ChunkUnit.Sentences, size: 1, overlap: 0)          // or .Add(chunks)
    .Bm25(k1: 1.2, b: 0.75)                                                  // keyword search
    .Embeddings(encoder)                                                     // vector search
    .Fusion(k: 60, depth: 20)                                                // required when both are on
    .Build();
index.Save("towns.index");                                                   // RetrievalIndex.Load(path, encoder)

var rag = Rag.For(chatModel)                                                 // any IChatModel: ChatGenerator, engine model, fake
    .Retrieve(index, top: 10)
    .Rerank(new CrossEncoder(scorer, encodePair, pairShape: [36]), keep: 3)   // optional
    .Prompt((question, passages) => ...)                                     // optional (Rag.DefaultPrompt)
    .Build();
var answer = await rag.AskAsync("who is the mayor of armorden");           // answer.Text, answer.Cited, answer.Passages

var tools = ToolRegistry.Create().Add(RetrievalTools.Search(index, 3, "search_towns", "Searches facts about towns.")).Build();
```

`Bm25Index`, `VectorIndex` (exact SIMD search, dot or cosine, save/load) and `Chunker.Split` can also be used on
their own.

Each stage is an interface, so another model, database or service can take its place: `IEmbedder` (`TextEncoder` is
one), `IVectorStore` (`InMemoryVectorStore`: replace and delete by id, metadata filters), `IRetriever` (`RetrievalIndex`
is one) and `IReranker` (`CrossEncoder` is one).

```csharp
public sealed class MyEmbedder(HttpClient http) : IEmbedder            // a hosted embedding model
{
    public async ValueTask<float[][]> EmbedAsync(IReadOnlyList<string> texts, CancellationToken ct = default) => await CallApiAsync(http, texts, ct);
}

var index = await RetrievalIndex.Create()
    .Documents(documents, ChunkUnit.Sentences, size: 3, overlap: 1)
    .Bm25()
    .Embeddings(new MyEmbedder(http))                                    // or a TextEncoder
    .VectorStore(myQdrantStore)                                          // optional: an IVectorStore (in memory unless set)
    .Fusion(k: 60, depth: 20)
    .BuildAsync();

var rag = Rag.For(chatModel)
    .Retrieve(index, top: 10)                                            // or any IRetriever: a database's full-text search, a web search
    .Rerank(myHostedReranker, keep: 3)                                   // or a CrossEncoder
    .Build();
var passages = await rag.RetrieveAsync("who is the mayor of armorden");
```

An index over an `IVectorStore` stores each chunk's vector under its id ("0", "1", …) with its document and position as
metadata; `Save` writes the chunks and settings, and `RetrievalIndex.Load(path, embedder, store)` attaches them to the
store again.

### MCP: the optional `Idrak.Mcp` package

```csharp
await using var files = await McpTools.ConnectStdioAsync("npx", ["-y", "@modelcontextprotocol/server-filesystem", "/data"]);
var tools = ToolRegistry.Create()
    .Add(await files.ListToolsAsync(prefix: "fs_"))                         // the server's tools as Idrak tools
    .RequireApproval("fs_write_file", (call, ct) => AskUserAsync(call, ct))
    .Build();                                                                // use in a Conversation, the engine or MapOllamaApi

options.ToolCollection = [.. McpTools.ServerTools(registry)];                // or serve a registry to MCP clients
```

`ConnectHttpAsync(uri)` and `ConnectAsync(transport)` connect to other servers. The package depends on
`ModelContextProtocol.Core`; the core library stays dependency-free.

### Quantization: int8 weights, half-precision files

```csharp
model.QuantizeInt8();                          // every Linear: 1 byte per weight + 1 scale per output column (¼ of the memory)
model.QuantizeInt8(l => l.OutFeatures > 64);   // or only some layers
model.Save("model.ikw");                       // int8 stays int8; a float model built the same way loads it (and becomes int8)
model.Save("model.f16.ikw", WeightFormat.Float16);   // or BFloat16: half-size files, float32 again after loading
package.Weights("model", model, WeightFormat.BFloat16);

model.QuantizeInt8();                          // QLoRA-style fine-tuning: frozen int8 weights,
model.AddLora(rank: 8, alpha: 16, targets: _ => true, freezeBase: true);   // trainable float adapters on top
model.DequantizeInt8();                        // float weights again (the rounding stays)

var generator = new TextGenerator(model, tokenizer, 256) { CacheFormat = KeyValueFormat.Int8 };   // int8 KV cache
using var context = new DecodingContext(device, batch: 4, capacity: 2048, KeyValueFormat.Int8);   // or directly
```

Int8 layers read their bytes directly when few rows go through them (token-by-token generation, where reading
weights is the bottleneck), and dequantize once per call for larger batches. Gradients flow through them, so
layers before them and LoRA adapters on them still train. ONNX export writes the dequantized weights. Biases,
normalization, embeddings and convolutions stay float32. The int8 KV cache stores each head's keys and values
at every position as bytes plus one scale (17/64 of the memory at head size 64); attention reads the bytes
directly, and recorded CUDA graphs work with it. Cache formats of one's own derive from `KeyValueLayout` (write the
rows, expand them to float32) and attend through masked float32 products on any device; register them with
`KeyValueLayouts.Register` and choose them with `TextGenerator.CacheLayout` or `new DecodingContext(..., layout)`.

### ONNX: the optional `Idrak.Onnx` and `Idrak.Onnx.Runtime` packages

```csharp
model.ExportOnnx("house-price.onnx", 9);                                  // one sample is [9]; the batch stays dynamic

OnnxExport.For(model).Input(28)                                          // or configure the export
    .Names("ids", "logits").Metadata("tokenizer", "words")
    .Lambda("ClsToken", OnnxExport.FirstStep)                             // your own lambdas need a translator
    .Save("reranker.onnx");

using var imported = OnnxImport.Load("model.onnx", Device.Cuda());       // .onnx → Idrak layers, on Idrak's CUDA kernels
imported.Model.Predict(input);                                          // a Sequential: predict, fine-tune (LoRA), move devices
imported.SavePackage("model.ikm");                                      // architecture + weights; Predictor.Load("model.ikm", device)

using var onnx = OnnxModule.Load("model.onnx");                          // or run the file with ONNX Runtime (CPU package)
```

`Idrak.Onnx` has no dependencies: it reads and writes the protobuf itself. **Export** covers Linear (LoRA
adapters merged), activations, Softmax, BatchNorm, LayerNorm, Conv2d, pooling, Flatten, Embedding,
PositionalEncoding, attention, transformer layers, LSTM, GRU and the builder's shape helpers. **Import** turns a
chain of those layers into a `Sequential` (with its `Network` builder), and anything else into a `GraphModule`:
layers where the importer recognizes them (Conv, BatchNorm, MatMul/Gemm → Linear, attention, LSTM/GRU, …) and graph
operations for the rest (skip-connection Add, Concat, Mul, Reshape, Transpose, Squeeze/Unsqueeze, Slice, Gather,
ReduceMean, and shape arithmetic such as PyTorch's flatten, computed for any batch size). So ResNet-style models
with skip connections and branches import too. Exact GELU becomes the tanh approximation and is listed in
`Notes`; an operator with no Idrak equivalent is reported by name. Imported models are ordinary Idrak
models: they run on Idrak's own CPU and CUDA backends, can be fine-tuned, and save as `.ikm` (graphs included).

`Idrak.Onnx.Runtime` references `Microsoft.ML.OnnxRuntime`, which is ONNX Runtime's **CPU** build; running
ONNX Runtime on a GPU needs its `.Gpu` (CUDA) or `.DirectML` package instead. `OnnxModule` works with predictors,
the inference engine and the ASP.NET Core endpoints. The tests run every exported layer in ONNX Runtime and
import it back, and both must match Idrak within 1e-4.

### Pretrained models: the optional `Idrak.LanguageModels` package

```csharp
using var model = PretrainedModel.Load("Qwen3-0.6B", new PretrainedOptions { Device = Device.Cuda(), Int8 = true, MaxPositions = 8192 });
model.Spec;                      // the DecoderSpec read from config.json (layers, heads, GQA, RoPE, q/k norm, …)
model.Notes;                     // anything approximated (for example sliding-window attention beyond the window)

var chat = model.CreateChat(KeyValueFormat.Int8);                          // the model's own chat template and stop tokens
var reply = chat.Chat(new ChatRequest(messages, tools, Think: false));     // reasoning and tool calls come back parsed

string text = model.ChatTemplate!.Render(messages, tools, think: null, addGenerationPrompt: false);   // training text

PretrainedArchitectures.Register("MyForCausalLM", PretrainedArchitectures.LlamaStyle((config, spec, notes) => spec with { QkNorm = true }));
```

A model folder in the Hugging Face layout becomes an ordinary Idrak `Sequential`: `config.json` is read into a
`DecoderSpec` by an architecture registry (Llama, Mistral, Qwen2, Qwen3 and Gemma are registered; others are one
`Register` call, usually `LlamaStyle` with a few spec changes), and the weights are read from safetensors (F32, F16,
BF16; single files or sharded with an index) one tensor at a time, transposed to Idrak's layout and, with
`Int8`, quantized as they are read. Nothing is built for one family: the decoder is assembled from generic blocks
(RMSNorm, RoPE with linear/llama3 scaling, grouped-query attention, gated feed-forward, optional q/k norm, biases,
post-norms, tied embeddings), so other models use the same code. The model runs on Idrak's CPU and CUDA
backends, and trains, takes LoRA adapters, quantizes and saves like any other.

`BpeTokenizer` reads `tokenizer.json` (byte-level BPE as in Qwen, Llama 3 and GPT-2; SentencePiece-style BPE with
byte fallback as in Llama 2 and Mistral; the normalizers, pre-tokenizers and decoders those use), and
`JinjaChatTemplate` renders the model's own chat template with a Jinja interpreter (the subset templates use,
with Hugging Face's settings), so prompts, tool definitions, tool calls and reasoning are laid out exactly as the
model was trained, for any family. The tests compare the interpreter with Python's jinja2 on real templates.

To check a model against transformers on your machine (it needs PyTorch and access to the Hugging Face Hub):

```
pip install torch transformers huggingface_hub
python tools/pytorch/pretrained_reference.py --model Qwen/Qwen3-0.6B --out qwen3.json
dotnet run -c Release --project samples/Idrak.Samples.Chat -- check qwen3.json --cuda [--int8] [--kv8]
dotnet run -c Release --project samples/Idrak.Samples.Chat -- chat <model folder> --cuda
```

### Fine-tuning: freezing, a learning rate per group, saving only what changed, LoRA

```csharp
model.Freeze(0..^1);                                           // every layer but the last (= RequiresGrad = false)
using var trainer = new Trainer(model, Losses.MeanSquaredError, p => new Adam(p, 1e-3f));   // p = trainable parameters only
model.SaveTrainable("head.nsp");                               // only the head; LoadTrainable restores it

var grouped = new GroupedOptimizer(new AdamW(body.Parameters(), 1e-4f), new AdamW(head.Parameters(), 1e-3f));

gpt.AddLora(rank: 8, alpha: 16, targets: layer => true, freezeBase: true);   // adapters on every Linear, outputs unchanged at start
// ... train: only the adapters change; SaveTrainable stores just them ...
gpt.MergeLora();                                                // fold them into the weights
```

## Telemetry: logging and tracking

Every step of training and inference is published to a hub, and you choose what to listen to.
Hooks are **zero-cost when nothing is subscribed**. Each publishing site checks one static field
with a bitwise AND, and no event is built or allocated unless some hook asked for that level.

```csharp
using Idrak.Diagnostics;

using var a = Telemetry.Subscribe(new ConsoleLogger(TelemetryLevel.Training, epochInterval: 10));
using var b = Telemetry.Subscribe(new MetricsRecorder());                        // in memory, SaveCsv()
await using var file = new JsonLinesLogger("run.jsonl", TelemetryLevel.All);     // background writer
using var c = Telemetry.Subscribe(file);
```

| Level | Event | Data |
|-------|-------|------|
| `Training` | `TrainingStarted`, `EpochCompleted`, `TrainingCompleted` | model summary, device, sizes; per-epoch loss, metrics, validation loss/metrics, learning rate, duration, samples/s, memory, best flag; early-stop info |
| `Batches` | `BatchCompleted` | epoch, batch, global step, batch loss, learning rate, data-wait time, compute time |
| `Gradients` | (adds to `BatchCompleted`) | global gradient L2 norm |
| `Layers` | `LayerForward` | layer name/type, nesting depth, input and output shapes, time, train/eval mode |
| `Operations` | `OperationCompleted` | every tensor op, forward and backward (∇), shape, device, time |
| `Inference` | `InferenceCompleted` | model, samples, shapes, device, latency, samples/s |

**Custom hooks.** Implement `ITelemetryHook`, set `Levels`, and override only the methods you need.
The other methods default to no-ops. Events are `readonly record struct`s passed by `in`, so they
are not copied or boxed.

```csharp
sealed class SlackAlert : ITelemetryHook
{
    public TelemetryLevel Levels => TelemetryLevel.Training;
    public void OnEpochCompleted(in EpochCompleted e) { if (double.IsNaN(e.Loss)) Alert("loss diverged"); }
}
```

**Why synchronous hooks plus an optional `Channel`.** A direct interface call is the fastest possible
dispatch, with no queueing, allocation or thread hop. Hooks that do slow work, such as files,
databases or HTTP, should use `ChannelTelemetry`. It copies each event into a bounded `Channel`
that never blocks training (the oldest events are dropped when it is full), and your consumer reads
it with `await foreach`. `JsonLinesLogger` is built this way.

GPU work is asynchronous, so layer and operation timings measure launch time by default. Set
`Telemetry.SynchronizeForTiming = true` when profiling to get true execution times.

## Data loading

* `Dataset.LoadCsv(path, options)` reads numeric CSV files. Lines are parsed in parallel, and errors
  name the line and column. `Dataset.FromArrays` and `Dataset.FromFlat` build a dataset from memory.
* `Split`, `Subset` and `Scale` return new, immutable datasets.
* `StandardScaler` and `MinMaxScaler` are fitted on training data. They provide `Transform` and
  `InverseTransform` (to turn predictions back into real units), plus `Save`/`Load` for deployment.
* `DataLoader` batches and shuffles data and moves each batch to the device. For large batches, the
  next batch is gathered on a worker thread while the current one trains, using double-buffered
  prefetch.

## Resources and parallelism

```csharp
ComputeResources.MaxCpuThreads  = 4;                   // default: every core
ComputeResources.GpuMemoryLimit = 2L << 30;            // bytes per GPU; default: unlimited
ComputeResources.CpuMemoryLimit = 1L << 30;            // default: unlimited
MemoryUsage usage = ComputeResources.GetMemoryUsage(Device.Default);   // in use / cached / limit
ComputeResources.ReleaseCachedMemory();
```

By default nothing is reserved up front, and memory is allocated as the network needs it. Freed blocks
are cached and reused, so a training loop stops allocating after its first iteration. Going over a
limit first releases the cache, then throws `ResourceLimitExceededException` with a clear message.

Parallelism is used where it pays off:

* SIMD in every CPU kernel.
* Multi-threaded kernels above 65,536 elements, and a multi-threaded matrix multiply for large
  products. On 4 AVX2 cores, a 1024×1024 multiply reaches about 86 GFLOP/s.
* Parallel CSV parsing.
* Background batch prefetch.
* On the GPU, thousands of threads per operation. Loss and metric sums stay on the device, so each
  epoch needs only one synchronization.

Everything on the CPU respects `MaxCpuThreads`, and small tensors stay single-threaded because
scheduling would cost more than it saves.

## NPU support

NPUs (Intel AI Boost, AMD Ryzen AI / XDNA, Qualcomm Hexagon) are **not supported**, and they cannot be
supported under this library's no-dependency rule:

* Unlike NVIDIA GPUs, which accept PTX through the driver, NPUs expose no general-purpose instruction
  set. They only run networks compiled by the vendor's own toolchain: OpenVINO for Intel, Ryzen AI /
  Vitis AI for AMD, QNN for Qualcomm, or DirectML / Windows ML on Windows.
* Those toolchains are the "prebuilt libraries" the project excludes.
* Current NPUs are mostly inference engines (INT8/FP16) and do not support training.

The backend design (`Backends/Backend.cs`) leaves room for an optional add-on package, for example
`Idrak.OpenVino`, that runs inference on an NPU while the core stays dependency-free.

## GPU support

The GPU kernels are PTX (NVIDIA's GPU assembly) generated in C# and compiled by the display driver when the
library starts, so any NVIDIA GPU the driver supports runs them. Faster kernels load where the GPU has the hardware
for them; every feature has a fallback that runs everywhere.

| Compute capability | GPUs (examples) | What runs |
|---|---|---|
| 5.0 and newer | GTX 900 (Maxwell) and later | every core kernel (float32 products, convolution, attention, LSTM/GRU, int8 / int4 weights, KV caches, sampling, CUDA graphs) |
| 8.0 and newer | RTX 30 (Ampere), RTX 40 (Ada), RTX 50 (Blackwell) | adds bfloat16 tensor-core products, flash attention, sequence packing, int8 tensor-core products |
| 8.9 and newer | RTX 40 (Ada), H100 (Hopper), RTX 50 (Blackwell) | adds FP8 tensor-core products |

Sizes (split-k, grids) come from the device's own counts, and memory adapts to the card: when a fine-tuning step
does not fit, it retries with lighter settings (released or recomputed activations, bfloat16 activations,
checkpointing); `--offload` keeps tensors in system memory when the GPU is full: cold data first (optimizer state,
then frozen weights), each layer's offloaded weights copied to the GPU while it computes (the next layer's in the
background), and tensors brought back when memory frees up. `--cpu-optimizer` runs AdamW's update on the CPU with its
state in system memory (only gradients and weights cross PCIe).

The 178 kernels (134 core, 44 tensor-core) assemble without errors for sm_50, sm_61, sm_75, sm_86, sm_89, sm_90
and sm_120 (Maxwell to Blackwell), and the FP8 kernels for sm_89, sm_90 and sm_120 (`ptxas` 12.9, 2026-09-30).
Every kernel launch is also checked against the kernel's declared parameter count.

Requirements: an NVIDIA display driver (Windows `nvcuda.dll`, Linux `libcuda.so.1`); nothing else. Without a
driver, or with `IDRAK_DISABLE_CUDA=1`, everything runs on the CPU (SIMD: AVX2, AVX-512 or NEON).

## Tests

```bash
dotnet run -c Release --project tests/Idrak.Tests
```

The runner prints the system it runs on, then runs **every test on every device**: the CPU and each CUDA GPU
present (a machine with one GPU runs 170 tests twice, 340 in all). Each GPU's line shows its memory, SM count,
compute capability and the CUDA version of its driver.

| Setting | Meaning |
|---|---|
| `IDRAK_DEVICES=cpu` / `cuda` / `cpu,cuda:1` | only these devices |
| `IDRAK_FILTER=text` | only the tests whose name contains the text |
| `IDRAK_TIMEOUT=600` | seconds a test may run before it counts as hung (default 300) |
| `IDRAK_TRACE=1` | full stack traces for failures |
| `IDRAK_CUDA_DEBUG=1` | synchronize after every kernel, to name the one that faults |

The 170 tests, by area:

| Area | Tests | What they check |
|---|---|---|
| Tensors and training basics | 24 | matrix products and element-wise ops against references; gradients of every basic op (finite differences); dropout masks equal on CPU and GPU; save/load; CSV, scalers, data loader; trainer; telemetry; memory limits; thread counts; no leaks |
| Layers and learning | 27 | softmax, cross-entropy, batched products, shape ops, BatchNorm, LayerNorm, embeddings, convolution, pooling, LSTM/GRU, attention: outputs against references and gradients; spirals, a CNN, an LSTM and a transformer learn; schedulers; number types |
| Decoding | 7 | fused kernels equal their unfused forms; KV-cache and batched decoding equal the full pass; CUDA graph replay; the sampler (top-k/p, min-p, penalties); every kernel's parameter count |
| Generation | 8 | streamed output parsed into thinking, content and tool calls; tokenizers; templates; stop sequences; cache/graph/recompute agree; prompt-cache reuse; emoji and other characters stay whole however tokens split them |
| Simplified API, engine and tools | 18 | builder, predictors, packages and trainer equal the manual steps; LoRA; tools and conversations; the inference engine (batching, queues, timeouts, keep-alive) |
| ASP.NET Core | 1 | predict, generate (JSON and server-sent events), Ollama API, status, over a real Kestrel server |
| Retrieval and MCP | 6 | chunking; BM25 against the formula; encoders; hybrid index with rank fusion; re-ranking and RAG citations; tools served over MCP |
| ONNX | 8 | MLP, CNN, LSTM/GRU, transformer and GPT exported and run in ONNX Runtime, and imported back (PyTorch-style graphs, ResNet blocks) |
| Quantization | 11 | int8, int4 and bfloat16 weights (products, gradients, save/load, QLoRA); packed projections; half-precision files; int8 KV cache |
| Decoder | 8 | RMSNorm, rotary embeddings and tiled attention (with gradients); every DecoderSpec variant against a plain reference; cached decoding through quantized weights; LoRA by layer name |
| Language models | 7 | safetensors; Hugging Face checkpoints through the registry; BPE tokenizers (byte-level and SentencePiece); text with half a character; Jinja templates against jinja2; a Qwen3 template as transformers renders it; tool-call formats |
| Fine-tuning and scoring | 19 | chunked cross-entropy and LoRA terms in tensor-core products; sequence packing; CUDA-graph steps; out-of-memory fallbacks; checkpointing; answer balancing; answer scoring; model download; PEFT adapters and merged export |
| Mixed precision | 11 | bfloat16 and 8-bit (FP8, int8) tensor-core products; flash attention (forward and backward); fused LayerNorm and decoder blocks; 8-bit AdamW |
| Coding agent | 4 | workspace-bound file tools; allowlisted commands; agent transcripts; a task run and verified |
| Datasets | 7 | Parquet (as pyarrow reads it); JSON Lines, JSON, CSV, text, archives; lazy transforms; download cache; conversation layouts; recipes; hub sources against a fake server |
| GGUF and evaluation | 3 | every GGUF quantization type dequantized as llama.cpp does; GGUF models equal their Hugging Face copies; answer metrics |
| Naming | 1 | no file uses the library's former name |

The ASP.NET Core, MCP, dataset and naming tests do not depend on the device, and run on each device like the others.

## Tested on

| Date | Library | GPU | Compute | Driver | System | Result |
|---|---|---|---|---|---|---|
| 2026-10-01 | 0.1.7 | NVIDIA GeForce RTX 5070 Ti, 16 GB, 70 SMs (Blackwell) | 12.0 | CUDA 13.3 | Windows (x64), 24-thread CPU, .NET 10.0.12 | 382 of 382 (CPU and GPU) |
| 2026-10-01 | 0.1.6 | NVIDIA GeForce RTX 5050 Laptop GPU, 8 GB, 20 SMs (Blackwell) | 12.0 | CUDA 13.3 | Windows 11 (build 26200, x64), 16-thread CPU with 8-wide SIMD, .NET 10.0.12 | 382 of 382 (CPU and GPU) |
| 2026-10-01 | 0.1.5 | NVIDIA GeForce RTX 3060 Laptop GPU, 6 GB, 30 SMs (Ampere) | 8.6 | CUDA 13.0 | Windows 11 (build 26200, x64), 20-thread CPU with 8-wide SIMD, .NET 10.0.12 | 364 of 364 (CPU and GPU; FP8 products skipped, no FP8 tensor cores) |
| 2026-10-01 | 0.1.5 | NVIDIA GeForce RTX 5050 Laptop GPU, 8 GB, 20 SMs (Blackwell) | 12.0 | CUDA 13.3 | Windows 11 (build 26200, x64), 16-thread CPU with 8-wide SIMD, .NET 10.0.12 | 364 of 364 (CPU and GPU) |
| 2026-10-01 | 0.1.5 | none (CPU only) | – | – | Windows 11 (build 22631, x64), 8-thread CPU with 8-wide SIMD, .NET 10.0.8 | 182 of 182 |
| 2026-09-30 | main (after 0.1.0) | NVIDIA GeForce RTX 5050 Laptop GPU, 8 GB, 20 SMs (Blackwell) | 12.0 | 610.88 (CUDA 13.3) | Windows 11 (build 26200, x64), 16-thread CPU with 8-wide SIMD, .NET 10.0.12 | 340 of 340 (CPU and GPU) |
| 2026-09-30 | main (after 0.1.0) | none (CPU only) | – | – | Ubuntu 24.04 (x64), 4-thread CPU with 8-wide SIMD, .NET 10.0.12 | 170 of 170 |
| 2026-09-29 | 0.1.0 | NVIDIA GeForce RTX 5070 Ti, 16 GB, 70 SMs (Blackwell) | 12.0 | not recorded | Windows (x64), .NET 10 | 340 of 340 (CPU and GPU) |

Fine-tuning speed on the RTX 5070 Ti (Qwen2.5-0.5B, LoRA, 4k-token steps): 17.4k tokens/s on chat data, 25.5k on
short classification rows; chat with Qwen3-0.6B generates about 140 tokens/s. Not tested yet: GPUs before
compute 8.6 (the kernels assemble for them, see GPU support), Linux with a GPU, and macOS (CPU only).

Benchmarks on the RTX 3060 Laptop GPU (`--bench-gemv`, Qwen3-0.6B shapes, int8 weights): decoding products of one row
run at 75–125 GB/s (q/k/v 34 µs, gate/up 84 µs, the 151,936-column head 1.9 ms); decoding attention over 4,000 cached
positions 360 µs; a 1 KB upload behind queued work holds the host 5 µs through the staging ring against 521 µs
synchronously. On its 20-thread CPU (`--bench-cpu`): a bfloat16 product of one row, 1536 × 32,000, 7.8 ms; attention
over 4,000 cached positions 6.7 ms (float32), 5.2 ms (int8), 10.1 ms (bfloat16); sampling 151,936 tokens with top-k 20
1.6 ms. On the RTX 5050 Laptop GPU (idle machine): decoding products of one row at 180–245 GB/s
(q/k/v 18–22 µs, gate/up 35 µs, the head 0.64 ms), decoding attention over 4,000 positions 94 µs, a 1 KB upload behind
queued work 5.1 µs (299 µs synchronously); its 16-thread CPU: the bfloat16 product 4.3 ms (1 row), attention over 4,000
positions 2.6 ms (float32), 1.6 ms (int8), 2.8 ms (bfloat16), sampling with top-k 20 0.7 ms. On the 8-thread CPU-only Windows machine: the bfloat16 product 14.2 ms (1 row) and 32.2 ms (8 rows); attention
over 4,000 cached positions 9.5 ms (float32), 3.7 ms (int8), 15.0 ms (bfloat16); sampling with top-k 20 0.9 ms. On the RTX 5070 Ti (0.1.7): decoding products of one row at 500–716 GB/s
(q/k/v 7.2 µs, gate/up 12.5 µs, the head 0.22 ms), decoding attention over 4,000 positions 38 µs, a 1 KB upload behind
queued work 8.9 µs (83 µs synchronously); with the choices measured on the card, "auto" is within a few percent of the
best fixed split in every row. A training step of 8 layers of 2048 × 2048 (`--bench-offload`): 5.9 ms on the GPU,
20.7 ms with the weights in system memory (staged, next layer prefetched; 87 ms read over PCIe), 41.7 ms with AdamW on
the CPU. Its 24-thread CPU (`--bench-text`): BM25 indexing of the README 1.8 ms, tokenizer encoding 45 MB/s.

Small models such as the samples are dominated by kernel-launch overhead on the GPU; recurrent models are hit
hardest, since they launch kernels for every time step. The GPU pays off with wide layers, large batches,
convolutions and language models. Tensors hold up to 2³¹ elements, and embedding ids must be below 2²⁴.
