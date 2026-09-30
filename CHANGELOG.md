# Changelog

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

## 0.1.3 (2026-09-30)

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
