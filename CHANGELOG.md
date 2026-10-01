# Changelog

## Unreleased

- Core code no longer branches on the device or a data format; each goes through an abstraction its implementations
  fill in (same results, same kernels, nothing added on the per-token path beyond one interface call per layer):
  - **Device capabilities** (`Backend.Capabilities`): few-row limit, decoding and tiled attention head sizes, matrix
    units, fused kernels, profiling. Layers read them instead of `DeviceType.Cuda` checks and CUDA kernel constants;
    `Device.Name`, `MixedPrecision.TensorCoresUnavailable` and `GpuProfiler` no longer cast to the CUDA backend. New
    `Device.IsGpu`.
  - **Token sampling** (`ITokenSampler`): `TokenSampler` is the built-in one; `TextGenerator.CreateSampler` plugs in
    another (constrained or grammar-guided decoding, custom rules). A sampler that decides on the host reports
    `Recordable = false` and the decoding step then runs without a CUDA graph.
  - **KV cache formats**: one layout per `KeyValueFormat` writes and attends; the attention layers no longer branch on
    the format. `DecodingContext[layer]` returns a layer's cache.
  - **Weight formats** (`PackedWeight`, `PackedFormat`): `Int8Weight`, `Int4Weight` and `BFloat16Weight` share a base
    class that multiplies, expands and moves each format and says which fused paths it takes; `Linear.PackedWeight`
    and `Linear.FromPacked` are new (`Int8` / `Int4` / `BFloat16` stay). The backends' packed products take
    `PackedFormat` instead of an untyped number.
  - **Model families**: GGUF families are a table (Llama's interleaved query/key rows are a property of its entry).
- Plugins for ONNX import and network steps: registries instead of closed switches, with the built-ins registered the
  same way (same results). `OnnxImportOps.Register("MyOp", context => context.Add(b => ...))` imports an operator of
  your own in a chain of layers (`OnnxImportContext` gives its attributes, constant inputs, the current shape and notes).
  `NetworkOps.Register("scale", (builder, arguments) => ...)` adds a network step; `NetworkBuilder.Op(name, arguments)`
  uses it, writes it to JSON even when it adds a lambda or a custom layer, and `Network.FromJson` (model packages too)
  replays it. New `NetworkBuilder.Add((device, random) => layer, shape)` creates a custom layer anew on each build.
  Unknown steps and operators are reported with the registered names and how to register one.

## 0.1.7 (2026-10-01)

- License: Apache License 2.0 instead of MIT, from this version on (`LICENSE`, `NOTICE`, the packages' license
  expression `Apache-2.0`). Apache 2.0 is as permissive as MIT and adds an explicit patent grant. Versions 0.1.0
  through 0.1.6 stay under the MIT license.
- Every C# file starts with a copyright and license header (the new `.editorconfig` adds it to new files).
- Contributions are accepted under a Contributor License Agreement (`CLA.md`): contributors keep their copyright and
  give the project the rights to ship their work, including under other license terms later. `CONTRIBUTING.md` and a
  pull request template describe the checks and the CLA box.

## 0.1.6 (2026-10-01)

- Text paths allocate less and copy less, with the same results (checked against the old code on random inputs):
  spans, `SearchValues`, UTF-8 and pooled buffers instead of per-word, per-chunk and per-character strings. Measured
  with the new `--bench-text` (README as input, CPU): BM25 indexing 8.1-9.3 ms → 2.8 ms and 6.0 → 1.5 MiB allocated,
  100 BM25 searches 0.8-1.2 ms → 0.31 ms, chunking 0.8 → 0.47 ms and 646 → 189 KiB, parsing a streamed reply 99-145 µs →
  79 µs and 137 → 33 KiB. Also: stop sequences found with one scan of the new text; the chat output parser keeps no
  growing copies (tool calls were O(n²)); ChatML and `tojson` write JSON directly; SentencePiece decoding through a
  per-token table (10x less allocated) and byte fallback without falling back to string merges; CSV read in blocks
  (44-68 ms instead of 71-114 ms for 100k rows), JSON Lines parsed as UTF-8 (37-50 ms instead of 70-111 ms),
  `WriteJsonLines` with one writer, Parquet strings decoded from the page; SSE and NDJSON events written as UTF-8;
  CUDA kernel names from tables.
- No card-specific tuning: choices that depend on the GPU are measured on the GPU in use the first time a shape needs
  them (each candidate timed on the real inputs, the fastest kept for the process), instead of rules set from one
  card's benchmarks. Covered: the k splits of few-row packed products, prompt-sized packed products (one, several and
  low-rank) and long-k tensor-core products; whether 2-8 int8 rows take the GEMV or the packed tensor-core product (the
  old rule was 1.2-1.4× slower on compute 8.6 cards); whether the gated activation is fused into the down projection
  (now for int8 and bfloat16 too, where it wins); and the 64 / 128 tile of the float and packed products without tensor
  cores. Before measuring, and while a graph is recorded, the formulas from the device's own counts apply; a measured
  choice replaces them only when at least 3% faster. `IDRAK_AUTOTUNE=0` keeps the formulas. Tuning keys are numbers
  and kernel names come from prebuilt tables, so the per-token path builds no strings.
- `ComputeResources.GpuMemoryReserve` and `OffloadReturnHeadroom` are now `long?`: null (the default) scales with each
  GPU (a sixteenth and an eighth of its memory, at least 256 MiB) instead of fixed 512 MiB and 1 GiB.
- Offloading (`ComputeResources.OffloadToHostMemory`, `IDRAK_OFFLOAD=1`, `idrak-tune --offload`) is now an internal
  interface, `IMemoryOffload`, that each GPU backend implements (CUDA today); the library uses whatever the current
  device's backend provides, and a device without it (the CPU) skips every step. With offloading on:
  - **Cold data first:** when a step finds the GPU full, its extra data still goes to system memory, but at the next
    step boundary (`Optimizer.ZeroGrad`) as many bytes of cold data move out instead: optimizer state first, then frozen
    weights (a LoRA base, `Freeze()`d layers, int8 / int4 / bfloat16 weights), largest first. The following steps keep
    their activations on the GPU. `ComputeResources.OffloadColdFirst` (on by default).
  - **Staging and prefetch:** before a layer computes, forward or backward, its offloaded weights are copied to the GPU
    in one transfer (instead of being read over PCIe by every kernel), and the next layer's weights are copied on a
    separate stream while this one computes; the copies go back to the cache afterwards, so after the first pass
    staging allocates nothing. Embedding tables are not staged (a lookup reads only its rows).
    `ComputeResources.PrefetchOffloadedWeights` (on by default).
  - **Coming back:** offloaded tensors return to the GPU, hottest first, when it has room beyond
    `ComputeResources.OffloadReturnHeadroom` (an eighth of the GPU's memory; cold data also leaves room for the largest spill seen), checked at
    each step boundary; `ComputeResources.ReturnOffloaded()` brings everything back that fits (e.g. after training).
    Pinned system memory is freed as tensors come home. `ComputeResources.ReturnOffloadedTensors` (on by default).
  - The decoder's fused products (q/k/v together, gate/up together, the down projection with its activation, a
    projection with its residual and normalization) stage all their layers' weights as one unit, forward and backward,
    and each text generation request starts with a rebalance (cold data out, or tensors back).
  - Nothing moves while a CUDA graph exists (graphs hold raw addresses), and the fine-tuner records no graphs when
    offloading.
- `HostOptimizer` runs any optimizer's update on the CPU with its state in system memory: each step downloads the
  gradients, updates CPU copies of the parameters with the CPU kernels and uploads the new values (only gradients and
  weights cross PCIe; Adam's state, twice the parameters, never touches the GPU). `FineTuningOptions.HostOptimizer` /
  `idrak-tune --cpu-optimizer` uses it for AdamW. On a GPU the copies go through 96 MiB of pinned buffers in the
  background (a parameter downloads while the previous one is updated and uploaded), the global norm is clipped on the
  GPU, and each parameter has its own CPU optimizer so it updates as soon as it arrives. `--bench-offload` in the tests
  times a training step each way.
- The CPU's fused AdamW (the path `ClipAndStep` takes) and its sum of squares run as SIMD on all cores instead of one
  scalar loop; each element is computed with the same operations in the same order as before.
- Text generation runs the first decoding step as it is and records the CUDA graph from the second, so the device
  measures its choices for the decoding shapes before the graph keeps them.

## 0.1.5 (2026-09-30)

- Retrieval: each stage is an interface. `IEmbedder` (`TextEncoder` implements it), `IVectorStore` (new
  `InMemoryVectorStore`: replace and delete by id, metadata filters, exact SIMD search), `IRetriever` (`RetrievalIndex`
  implements it) and `IReranker` (`CrossEncoder` implements it). `RetrievalIndexBuilder.Embeddings` takes any embedder,
  `VectorStore(store)` keeps the vectors in a vector database, `BuildAsync` and `RetrievalIndex.SearchAsync` await
  hosted services, and `Load(path, embedder, store)` re-attaches a saved index to its store. `Rag.For(...).Retrieve` and
  `Rerank`, and `RetrievalTools.Search`, take any retriever and re-ranker; `RagPipeline.RetrieveAsync` is new and
  `RagPipeline.Index` is null for a retriever that is not a `RetrievalIndex`. Existing code compiles unchanged, and an
  index built with a `TextEncoder` and no store is searched and saved as before.
- Training callbacks: `Trainer.Callbacks` (and `TrainingRun.Callbacks`) take `ITrainerCallback`s, whose optional
  `OnTrainBegin`, `OnBatchEnd`, `OnEpochEnd` and `OnTrainEnd` run inside `Fit` with a `TrainerContext` (model, optimizer,
  epoch, step, history, cancellation token); `context.Stop()` ends training after the current batch or epoch and returns
  the history as usual. Batch losses are read from the device only when a callback sets `NeedsBatchLoss`. Built in:
  `EarlyStopping` (the same history and weights as `EarlyStoppingPatience`, plus a choice of monitored value),
  `Checkpoint` (`last.ikw` and `best.ikw`, loaded with `Module.Load`) and `CsvLog`. A cancelled `Fit` now stops after the
  current batch instead of the current epoch.

## 0.1.4 (2026-09-30)

- Loading and exporting weights makes fewer full copies (the same values, bit for bit). Measured on the CPU (4 cores)
  with a 78 M-parameter Llama-style model (vocabulary 32,768, dim 1024, 4 layers, tied head):
  - `Module.Load` of an int8 / int4 / bfloat16 weights file allocates room for the packed weights instead of first
    quantizing the model's current (random) weights that the file replaces: int8 3.1 s → 0.15 s and 1.4 GB → 0.36 GB peak
    working set, int4 9.2 s → 0.1 s and 1.26 → 0.26 GB, bfloat16 3.9 s → 0.27 s and 1.6 → 0.69 GB. Float32 tensors are
    read and written as their bytes (no second float array).
  - `Int8Weight.Quantize`, `Int4Weight.Quantize` and `BFloat16Weight.FromValues` read span inputs in place and upload the
    packed values without copying them first; zero-filled buffers (dequantized weights, FP8 copies) are allocated on the
    device instead of uploaded from a host array.
  - Checkpoint matrices are transposed while they are read (safetensors and GGUF), so a tensor no longer exists in the
    stored order and transposed at once; a packed tied head is made with the embedding, so the table's float values are
    not kept while every layer is read. Safetensors int8 load: 1.09 → 0.76 GB allocated, peak working set 1.05 → 0.65 GB;
    bfloat16: 1.9 → 0.73 s, 1.32 → 0.91 GB allocated.
  - GGUF: tensors dequantize straight into the result on all cores (F16 and BF16 too), and Llama's q / k rows are put back
    in order in place or while transposing. Load (Q8_0 file): float32 1.4 s → 0.7 s and 1.47 → 1.01 GB peak working set,
    int8 1.8 → 1.0 s and 1.61 → 0.99 GB; `GgufFile.Dequantize` of 64 MB of Q8_0 plus 32 MB each of F16 and BF16,
    three times: 1.2 s and 961 MB allocated → 0.38 s and nothing allocated.
  - `PretrainedModel.SaveHuggingFace` writes the header from the shapes, then copies, transposes and encodes one tensor at
    a time (in parallel chunks) instead of holding every tensor on the host first (an 8B model held about 32 GB): peak
    working set 532 → 354 MB, 0.87 → 0.66 s.

- CPU decoding kernels (`--bench-cpu`, 4 cores, median of three runs): bfloat16 weights of up to 8 rows are multiplied
  directly (each vector widened once) instead of expanded to float32 per call (1536 × 32,000: 1 row 69 → 8.6 ms, 8 rows
  83 → 42 ms); attention over a bfloat16 cache reads it in place instead of copying the filled cache per step (4,000
  positions, 16 heads of 128: 22.7 → 10.6 ms); the attention kernels' dot products and weighted sums are vectorized
  (float32 cache 9.6 → 6.6 ms, int8 6.0 → 3.8 ms); `MultiHeadAttention` with an int8 cache reads only the filled
  positions; sampling finds the top-k cut-off and the top-5 statistics in one pass and computes the top-p weights once
  (151,936 tokens: top-k 20 4.5 → 1.5 ms, top-k 20 + top-p 0.95 14.4 → 1.6 ms, top-p alone 20 → 6 ms, its bisection
  dropping the scores below its lower bound as it rises), with the same tokens and statistics as before bit for bit.
- CPU products with a transposed B (`x.MatMul(w, transposeB: true)`, the tied head, backward products) of at most 16
  rows read B in place, one dot product per output, instead of transposing all of B on every call (k = 1536,
  n = 32,000 on 4 cores: 1 row 188 → 9 ms, 8 rows 196 → 19 ms).
- `Tensor.OneHot` writes the rows directly on the device (a new `OneHot` backend operation and `one_hot_f32` kernel)
  instead of reading them from a classes × classes identity table kept for the life of the process (16,000 classes:
  1 GB held before, nothing now; `Losses.SparseCrossEntropy`, `Metric.SparseAccuracy` and the trainer use it).
- `Idrak.Datasets`: a JSON file holding an array of rows is read in linear time; the rows were detached one by one from
  the front of the array (100,000 rows: 951 → 64 ms, the time of the same rows as JSON Lines).
- Rotary tables (cos/sin over every position) are built once per model, in parallel, and shared by its attention
  layers instead of once per layer (28 layers × 40,960 positions on the CPU: 6.6 s and 631 MB → 0.57 s and 65 MB
  including the layers' weights; on a GPU the same ~570 MB of device memory for Qwen3-0.6B).
- Retrieval: `Bm25Index` keeps an inverted index (each word's texts and counts) and scores only the texts holding a query
  word (50,000 texts: 54 → 0.22 ms per query, the same scores); `VectorIndex.Search` and BM25 keep the best hits in a heap
  instead of sorting every score (the same hits and order, ties to the lower id).
- `Idrak.Datasets` Parquet: top-level columns go straight into the row (no tree and merge per cell), dictionary strings are
  decoded once, binary values are checked as UTF-8 instead of throwing per value, and a row group finds its column chunks
  by path once (300,000 rows × 5 columns: 2.5 s and 1.4 GB allocated → 0.49 s and 0.47 GB).
- CUDA: prompts of 9 to 63 rows through int8, int4 and bfloat16 weights use the packed products (tensor cores with
  `MixedPrecision`) instead of expanding the whole weight to float32 on every call; 8 rows and fewer keep the GEMV
  kernels. Before, on an RTX 5070 Ti: 1024 → 151,936 int8 took 2.3 ms for 9–63 rows against 0.36 ms for 64, and
  1024 → 3072 took 70 µs against 23 µs.
- CUDA with `MixedPrecision` tensor cores: 4 to 8 rows through int8 weights of 32 M values or more (a vocabulary head)
  take the packed product too, whose time stays flat while the GEMV's grows with the rows (RTX 5070 Ti,
  1024 → 151,936: 4 rows 328 → 276 µs, 8 rows 521 → 268 µs). Smaller layers and 1–3 rows keep the GEMV.
- CUDA: uploads of up to 4 MB (`Tensor.From`, `Load`) copy the values into a 16 MB pinned staging ring and queue an
  asynchronous copy on the work stream (an event per copy guards its part of the ring) instead of a synchronous copy
  from pageable memory, which waited for all queued GPU work first (1 KB measured at 8–51 µs). Order with the kernels
  around it is unchanged; larger uploads and uploads while recording a graph keep the synchronous copy. RTX 5070 Ti:
  a 1 KB upload behind queued work holds the host 81.7 → 5.2 µs; idle, 1 MB 78 → 66 µs and 4 MB 192 → 151 µs.
- `--bench-gemv` also times short prompts (1–96 rows) through int8 weights and host-to-device uploads, for the next
  CUDA changes.

## 0.1.3 (never published: these changes shipped in 0.1.4)

- `Idrak.Datasets`: `ITextNormalizer`, the consumer's text rules (the library has none of its own), and
  `Dataset.Normalize(normalizer, columns)` (in place) / `Normalize(normalizer, column, into: key)` (a key column, the
  text kept as written) as their own step. `Dataset.Deduplicate(columns)` compares values exactly and no longer takes
  `normalize` (it lower-cased and collapsed spaces, which its name did not say): normalize first, then deduplicate:
  `data.Normalize(rules, "text", into: "key").Deduplicate(["key", "label"])`.
- `Idrak.Datasets` `Dataset.Deduplicate` is 2.7 times faster and allocates almost nothing beyond reading: a 128-bit
  hash straight from the values' characters replaces SHA-256 over a joined UTF-8 copy of each row (400,000 CSV rows:
  731 ms and 134 MB of garbage before, 275 ms and 21 MB now; reading alone takes 168 ms). Normalize and Deduplicate
  are streamed steps of one pass over the rows, not two loops.
- `Idrak.Datasets` `Dataset.Split` uses the same hash (no SHA-256, no UTF-8 copy; a whole row is hashed from its
  properties, not its JSON text): 400,000 rows, both sides read, 1,070 ms → 379 ms with a key and 1,485 ms → 399 ms
  on whole rows. The sides are still reproducible for a seed, but rows fall on different sides than in 0.1.2.
- Tests: the activation-memory check lets pending finalizers run before its first reading (other tests' tensors freed
  between its two readings made its count negative, now and then).

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
