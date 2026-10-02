# Plug-in points: what is still closed

What a user or an outside package cannot plug into Idrak today without changing the library, from a read-only survey
of the `architecture` branch (2026-10-02), and the dataset loader abstraction that follows from it. Nothing here is
built yet. File and line references are from that survey; check them again before starting an item.

"Closed" means a switch, enum, sealed type or fixed list that a user cannot extend without editing Idrak.

The registries that exist today are listed in the README ("What Idrak gives you" and "Extending Idrak: plug-in
points"). Devices are not among them yet: the device registry and `Backend` stay internal until the public backend
API (plan 7, item 12c).

## The first thing to fix: plug-ins written outside the library

`src/Idrak/Idrak.csproj:15-17` grants `InternalsVisibleTo` to Idrak.Tests, Idrak.Onnx and Idrak.LanguageModels, so the
first-party packages and every plug-in test use internals that an outside package cannot reach:

- `Tensor.Persistent` (src/Idrak/Tensor.cs:179) and `Tensor.Load` (Tensor.cs:188) are internal; the plug-in tests use
  them (tests/Idrak.Tests/PackedPluginTests.cs:55 and :173, KeyValuePluginTests.cs:70).
- Idrak.Onnx calls `Tensor.Load` 13 times (e.g. OnnxImport.cs:542, :851); Idrak.LanguageModels uses
  `Tensor.Persistent`, `Backend` and `Storage` (e.g. FineTuning.cs:1402-1497, PretrainedModel.cs:281).

No test proves that an outside assembly can write a plug-in. Done when: a test project without `InternalsVisibleTo`
registers a packed weight format, a KV cache format, a network builder step and an ONNX import operator, and runs
them.

## Ranked gaps

| # | Gap | Evidence | Who needs it | Suggested shape | Size | Priority |
|---|---|---|---|---|---|---|
| 1 | No public way to write values into an existing tensor | `Tensor.Load` internal (Tensor.cs:188); `AddInPlace`/`FillInPlace` internal (Tensor.Decoding.cs:274, 281); the protected helpers of `Optimizer` are only weight decay, state and gradient scale (Optimizer.cs:143-177) | Lion, Adafactor or Muon optimizers; an ONNX import translator's `load:` callback (OnnxImportOps.cs:77); custom initializers; an outside packed weight format | public `Tensor.CopyFrom(ReadOnlySpan<float>)`, in-place `AddScaled`/`Assign`, or a protected `Optimizer.Update(Tensor parameter, Tensor value)` | small | high |
| 2 | No custom differentiable operation (own forward and backward); a small public operation set | `Record` private (Tensor.cs:685); `UnaryOp`/`BinaryOp` internal (Backends/Backend.cs:8-20); `Tensor.Checkpoint` internal (Tensor.Decoder.cs:71). No public Sqrt, Pow, Clamp, Max/Min, Where, Sin/Cos, Erf, SiLU, Gather; TextEncoder computes 1/norm through Log, scale and Exp (Retrieval/TextEncoder.cs:164) | RoPE variants, ALiBi, Mish, focal loss | `Autograd.Function(name, forward, backward)` composed from public operations (no backend access needed); more element-wise operations | medium | high |
| 3 | RoPE scaling is closed: linear and llama3 only | switch at Layers/Decoder.cs:113-143; config parsing at LanguageModels/Architectures.cs:110-121; GGUF drops YaRN with a note (GgufModel.cs:292-301); `RopeScaling` is a sealed record with llama3's fields (Decoder.cs:93) | YaRN (Qwen2.5 and Qwen3 long context, DeepSeek), dynamic NTK, LongRoPE (Phi-3, Phi-4) | `RopeScalings.Register(type, Func<double[] frequencies, JsonObject parameters, double[]>)` plus an optional attention scale; `RopeScaling` holds the type and the JSON parameters | small (frequencies), medium (YaRN's attention factor) | high |
| 4 | Model families must fit the fixed `DecoderSpec` | `PretrainedArchitecture.Spec` returns a sealed `DecoderSpec` of fixed flags (Architectures.cs:17; DecoderSpec.cs:97-175); mixture of experts rejected (Architectures.cs:102-105); sliding window noted, not implemented (Architectures.cs:123-126); `FeedForwardActivation` (Decoder.cs:524-534) and `DecoderNorm` (DecoderSpec.cs:9-16) closed; no ALiBi, logit soft-capping or encoder-only models | Gemma 2/3 (soft-capping, alternating sliding window), Mixtral and Qwen-MoE, Phi-3 (fused qkv), BERT and BGE embedding models | an optional `Build` delegate on `PretrainedArchitecture` so a family builds its own module; later per-layer attention options and an activation registry | large | high |
| 5 | ONNX export has no global registry and misses layer kinds | per-exporter `Module<T>`/`Lambda` only (OnnxExport.cs:90-101); built-ins in a closed switch (OnnxExport.cs:169-202); no `GraphModule` (an imported graph cannot be exported again), no decoder layers; one input only (OnnxExport.cs:70) | a package shipping a layer with its ONNX translator; round trips of imported graphs | `OnnxExportOps.Register<T>(OnnxTranslator<T>)` and `Unregister`, built-ins registered the same way; the per-exporter call keeps precedence | small to medium | high |
| 6 | ONNX import of graph models ignores registered operators | fixed `Primitives` map (OnnxImport.cs:336-344) and closed `MatchLayer` (402-413), documented at OnnxImportOps.cs:90-92; closed `GraphModule` operation switch (GraphModule.cs:159-233) and layer list (`LayerDescriptions`, GraphModule.cs:464-522); one input and output (OnnxImport.cs:125, 350) | a ResNet or U-Net with Erf, Pow, Clip or ConvTranspose; a custom layer inside a graph | `GraphOps.Register(op, ...)`; `LayerDescriptions` becomes a layer type registry (describe, create), shared with item 7 | medium | high |
| 7 | Model packages (.ikm) cannot store custom architectures, scalers or tokenizers | `BuildModel` handles DecoderSpec, GraphModule or builder JSON only (Inference/ModelPackage.cs:293-325); scalers in a closed switch (ModelPackage.cs:111-116) with the `PackageEntryKind` enum (:50); tokenizers limited to character and word (:118-126; Generation/Tokenizer.cs:282-298) | shipping a custom module, a robust scaler or a BPE model in one file | an architecture registry keyed by the description's format; `IScaler` gains a save method and a format name with a scaler registry; the same for tokenizers | medium | medium |
| 8 | Tool-call parsing handles only JSON between tags | sealed `ToolCallFormat(Open, Close, List, ArgumentsKey)` (Generation/Chat.cs:33); sealed, JSON-only `ChatOutputParser` (Chat.cs:211, 492); the detector probes that shape only (ChatTemplates.cs:67-120) | Llama 3 pythonic calls, Qwen3-Coder XML, Mistral `[TOOL_CALLS]`, GPT-OSS channels, DeepSeek special tokens | an `IToolCallParser` (feed text, finish, calls) created by the chat template; `ToolCallFormat` stays the default | medium | high |
| 9 | Fine-tuning fixes the optimizer, schedule, loss and adapter type | AdamW (LanguageModels/FineTuning.cs:984-987), cosine schedule (:729-730), token cross-entropy (:1324, :1443); LoRA only, through a sealed `LoraAdapter` record with an internal setter (Layers/Linear.cs:121, 497) | other optimizers, a warm-up/stable/decay schedule, DPO, ORPO or KTO losses, DoRA | `FineTuningOptions.Optimizer`, `Scheduler` and loss delegates; adapters need an `IAdapter` on `Linear` | medium (adapters: large) | high |
| 10 | PEFT adapters: only plain LoRA, and the configuration is not checked | `LoadAdapter` and `AdapterMerge` read only `r`, `lora_alpha`, `use_rslora` (PretrainedModel.cs:250-283; Architectures.cs:240-246); `peft_type` and `use_dora` are never checked, so a DoRA adapter would likely load as plain LoRA (inferred from the code, not run) | loading DoRA, IA3 or VeRA adapters | first reject an unknown `peft_type` and `use_dora`; then an adapter type registry | small (checks), large (variants) | medium |
| 11 | Only BPE tokenizer models | other tokenizer.json models throw (BpeTokenizer.cs:87-90); GGUF accepts only the "gpt2" tokenizer model (GgufModel.cs:339-342); `BpeTokenizer` is always created (PretrainedModel.cs:138); `TokenizerComponents` has no model or post-processor entry (TokenizerComponents.cs:57-63) | WordPiece (BERT, BGE), Unigram/SentencePiece (T5, Gemma, Llama-2-style GGUF) | `TokenizerModels.Register(type, Func<JsonObject, ITokenizerModel>)` | medium to large | medium |
| 12 | The GGUF architecture registry cannot describe a family unlike Llama | `GgufArchitecture` holds only the Hugging Face name and query/key interleaving (GgufModel.cs:17-23); tensor names in a closed switch (:563-595); fixed config translation with `hidden_act` set to silu (:250-270) | Phi-3 (`attn_qkv`), Gemma 2 (post norms, GELU), Command R | `TensorName` and `Config` delegates on `GgufArchitecture` | small | medium |
| 13 | Generation: no stopping hook, logits processors, speculative decoding or structured output | string stops only (TextGenerator.cs:409-411, 470-488); the only plug point replaces the whole sampler (:89) and loses temperature, top-k and top-p; no logit bias (Sampling.cs); the Ollama request has no `format` field (AspNetCore/Ollama.cs:14-21) | stopping on token ids or a pattern, logit bias, a JSON schema mask over the normal sampler, a draft model | `GenerationOptions.StopWhen`; an `ILogitsProcessor` run before the sampler; a speculative mode with a draft model | small (stop, bias), medium (processors), large (speculative) | medium |
| 14 | Training data is in memory only | `Trainer.Fit` takes a sealed `DataLoader` (Training/Trainer.cs:162), which takes a sealed `Dataset` of `float[]` (Data/DataLoader.cs:52-67; Data/Dataset.cs:32-35); `Batch` has an internal constructor (DataLoader.cs:11) | images too large for memory, augmentation each epoch, custom batching | the dataset loader abstraction below | medium | high |
| 15 | Metrics are batch means only; no optimizer state in checkpoints | `Metric(Name, BatchMean, Finalize)` (Trainer.cs:16) cannot express F1, AUC, precision or recall; `Tensor.MatchRate` internal (Tensor.Advanced.cs:197); the checkpoint callback saves weights only (TrainerCallbacks.cs:217-250) | classification users; resuming a run exactly | an `IMetric` accumulator (reset, update, compute); `Optimizer.State()` and `LoadState()` as named tensors | small to medium | medium |
| 16 | Evaluation metrics are closed and need a concrete `ChatGenerator` | `AnswerMetric` enum and switch (LanguageModels/Evaluation.cs:13-29, 126-133); `Run(ChatGenerator ...)` (:68); the CLI parses the enum (FineTuning.Cli/Program.cs:141) | ROUGE, pass@k for code, a model as judge, multiple-choice letters; evaluating a remote model | a scoring delegate or an answer metric registry; take `IChatModel` | small | medium |
| 17 | Serving: closed model kinds and options | `EngineModelKind` is predictor, text or chat (InferenceEngine.cs:16-26); `ChatModel(...)` accepts only a `TextGenerator` (:339-348); no embedding kind, no `/api/embed` or OpenAI `/v1` routes; unknown Ollama options are dropped (Ollama.cs:128-143) | serving an embedder, a proxy model or a custom chat model; passing extra sampling options through | `ChatModel(name, Func<IChatModel>)`; an embedding model kind; `GenerationOptions.Extra` for unknown options | medium | medium |
| 18 | Retrieval: fixed pooling, chunking, keyword index and fusion | sealed `TextEncoder` with masked mean pooling only (TextEncoder.cs:18, 138-165); `ChunkUnit` is words or sentences (Chunking.cs:21-28); sealed `Bm25Index` and fixed fusion (RetrievalIndex.cs:30-31, 119-140); `VectorMetric` is dot or cosine (VectorIndex.cs:12-19) | CLS or last-token pooling (BGE, Qwen3-Embedding), token- or markdown-aware chunking, SPLADE | workarounds exist (`IEmbedder`, passing your own chunks); add a pooling choice, an `IChunker` and an `IKeywordIndex` | small each | low to medium |
| 19 | Datasets: registered file formats are skipped when picking hub files; compression and archives are closed | `FormatOf` maps only built-in formats (DataFiles.cs:121-122), so `RemoteFiles.IsDataFile`/`BestFormat` (Sources.cs:87-115) ignore registered ones; `.gz` only (DataFileFormats.cs:104; DataFiles.cs:225, 279); zip and tar only (DataFiles.cs:125-127) | Arrow or Feather files, `.jsonl.zst` corpora, `.tar.xz` archives | rank by the registered format (with a priority); `DataCompressions.Register(".zst", Func<Stream, Stream>)` | small | medium |
| 20 | Safetensors data types are closed | `SafeTensorType` is F32, F16 or BF16 (SafeTensors.cs:12-21, 283-286) | FP8 checkpoints (Qwen3-FP8, DeepSeek), I8/U8 (GPTQ, AWQ) | `SafeTensorTypes.Register(dtype, decoder)`, like `GgufTypes` | small | medium |
| 21 | Checkpoint export is closed | `SaveCheckpoint` is a closed switch over modules and throws for LayerNorm (PretrainedModel.cs:319-341); safetensors only; `ICheckpointFormat` reads only (CheckpointFormats.cs:12-33) | exporting a GPT-2-style model; writing GGUF | an optional `Write` on `ICheckpointFormat`; a reverse mapping per architecture | medium | low to medium |
| 22 | Packed weight plug-ins are second class | plug-ins get only the generic dequantize path; the fast hooks are internal virtual (Layers/Quantization.cs:129-152); weights files record built-in packed layers by index only (Layers/Module.cs:232-262), so a plug-in model must be rebuilt in the same format before loading (PackedPluginTests.cs:177-181); no `Pack(format)` extension (ModuleExtensions.cs:195-262); no `--weights` option in the CLI (FineTuning.Cli/Program.cs:93-95) | NF4, GPTQ or FP8 formats | record the format name in the weights file header; `Module.Pack(format, filter)`; `--weights NAME` | small to medium | medium |
| 23 | Jinja filters, tests and functions are closed | filter switch ends with "Unknown filter" (JinjaBuiltins.cs:628-845), tests at :1006; the callable delegate is internal (Jinja.cs:73), so `JinjaChatTemplate.Variables` (ChatTemplates.cs:178) cannot add functions | templates using a filter Idrak lacks; a fixed `strftime_now` for reproducible tests | `Functions` and `Filters` dictionaries on `JinjaChatTemplate` | small | low to medium |
| 24 | Fixed initializers; closed activation and precision enums | Xavier-uniform in `Linear` (Layers/Linear.cs:28-29) with no hook; builder `Activation` enum (NetworkBuilder.cs:78-87, 195-201); sealed activation layers (Activations.cs); `MatMulPrecision` is float32, bfloat16 or FP8 (MixedPrecision.cs:7-55) | Kaiming initialization; SiLU or LeakyReLU in `Dense`; FP16 on GPUs without bfloat16 | initializer delegates (needs item 1); activations by `NetworkOps` name; new precisions need the public backend (12c) | small / medium | low |
| 25 | Tool results are text only | `Tool.Invoke` returns `Task<string>` (Generation/Tools.cs:36); MCP results are flattened to text (Mcp/McpTools.cs:93-101) | image or structured tool output for multimodal models | `ToolResult` content blocks | medium | low |
| 26 | The command-line tools cannot load plug-ins | no assembly loading or `--plugin` option in src or samples | using any registry from `idrak-tune` or `idrak-data` | `--plugin path.dll`, calling a module initializer or an `IIdrakPlugin.Register()` | small | medium |

## Gaps inside the existing registries

- No public unregister: `PretrainedArchitectures` has register, names and get only, and no null checks
  (Architectures.cs:54-81); `PackedWeight` has register and format names only (its tests restore "int8" by registering
  it again, PackedPluginTests.cs:353, 362); `KeyValueLayouts.Unregister` is internal (KeyValueLayouts.cs:183-189).
- Silent replacement: every registry "registers or replaces", with no way to detect two packages claiming one name
  (e.g. NetworkOps.cs:84, OnnxImportOps.cs:98, DeviceProviders.cs:73-88, DataFileFormats.cs:45-60).
- Name matching differs: ordinal for NetworkOps, OnnxImportOps, PretrainedArchitectures and GgufArchitectures; ignoring
  case for PackedWeight, KeyValueLayouts, DataFileFormats and DeviceProviders.
- Built-ins not registered like plug-ins: `TokenizerComponents` stores empty entries that `BpeTokenizer`'s own switch
  handles (TokenizerComponents.cs:44-52); the `PackedFormat` and `KeyValueFormat` enum overloads bypass the registry
  (Quantization.cs:155; KeyValueLayouts.cs:136-139).
- Hidden fast paths: `KeyValueLayout` and `PackedWeight` keep theirs internal (KeyValueLayouts.cs:106-118;
  Quantization.cs:129-152); `KeyValueLayout` does give plug-ins a protected `WriteRows` (:91).
- Tests: every registry has a custom plug-in test except `PretrainedArchitectures` (exercised only at
  PretrainedTests.cs:209), and all of them run with internal access (see the first section).

## What needs the public backend (item 12c)

Custom kernels (item 2, when composed operations are not enough); fused optimizer updates (Optimizer calls
`Backend.SumSquares`, `ClipFactor` and `Axpy`, Optimizer.cs:78, 125-136, 160); the fast hooks of packed weights
(`DequantizeInto(Storage)`, Quantization.cs:134); new `MatMulPrecision` values (MixedPrecision.cs:72); graph capture
(ComputeGraph.cs:57-126) and fine-tuning's graph recording (FineTuning.cs:1421-1497); devices added from outside.
Everything else in this plan can be done without it.

## Dataset loaders: an abstraction and common implementations

Today `Trainer.Fit` takes a sealed `DataLoader`, which takes a sealed `Dataset` holding every sample in `float[]`;
`Batch` cannot be constructed outside the library; and `Idrak.Datasets` reads files and hubs only into chat rows, not
into training batches (item 14).

### The abstraction

| Layer | Type | What it does |
|---|---|---|
| Samples | `ISampleSource` | `Count`, `FeatureShape`, `TargetShape`, `Read(index, Span<float> features, Span<float> targets)`: random access, one sample at a time |
| Streams | `ISampleStream` | samples in order when the count is unknown or the data is too large to index |
| Transforms | `ISampleTransform` | per-sample changes (flips, crops, noise, masking), seeded per epoch so runs repeat |
| Batches | `IBatchSource : IEnumerable<Batch>` | what `Trainer.Fit` accepts; `DataLoader` is one, and anyone can write another (`Batch` gets a public constructor) |
| Registry | `SampleSources.Register(name, factory)` | loaders by name, like the other registries, built-ins registered the same way |

`Dataset` becomes one `ISampleSource`, so existing code keeps working. `DataLoader` keeps shuffling, dropping the last
batch, background prefetch and device placement, for any source.

### Common implementations

| Loader | Data | Notes |
|---|---|---|
| In memory | arrays and today's `Dataset` | unchanged behaviour |
| CSV, streamed | large CSV files read lazily | no full load into memory |
| Image folder | `folder/<class>/*` with class labels from folder names | PGM, PPM, BMP and PNG decoded without dependencies (PNG through the zlib in .NET); JPEG through a registered codec, keeping the no-dependency rule |
| Token files | packed token ids for language model pretraining | memory-mapped; fixed-length windows |
| Binary arrays | `.npy` files | memory-mapped, so large arrays are not loaded at once |
| Tables from `Idrak.Datasets` | JSON Lines, Parquet and CSV columns as features and targets | an adapter in the Datasets package |
| Views | concatenation, subsets, train/test splits, shuffles | no copying |

### Open choices

1. Where the loaders live: the interfaces and the in-memory, CSV, image, token and `.npy` loaders in the core, the
   JSON Lines and Parquet adapter in `Idrak.Datasets` (proposed); or everything in `Idrak.Datasets`.
2. Images: PGM, PPM, BMP and PNG built in with JPEG as a plug-in (proposed); or PGM and BMP only built in.

Done when: `Trainer.Fit` trains from each built-in loader with the same results as from an in-memory `Dataset` of
the same samples; a custom `ISampleSource` and a custom `IBatchSource` written in a test project without internal
access train a model; the OCR and shape-recognition samples use the image folder loader with an augmentation
transform.

## Proposed order

1. Plug-ins from outside the library: public tensor writes (item 1) and a test project without internal access.
2. Quick, high-value registries: RoPE scaling (item 3), ONNX export (item 5), `--plugin` for the command-line tools
   (item 26), registry hygiene (unregister, collisions, one name-matching rule).
3. Dataset loaders (item 14 and the section above).
4. Tool-call parsers (item 8) and fine-tuning options (item 9).
5. Custom differentiable operations and more element-wise operations (item 2).
6. Model families beyond `DecoderSpec` (item 4), then tokenizer models (item 11) and GGUF families (item 12).
7. The rest by priority; anything that needs the public backend waits for item 12c.

## Not confirmed by the survey

- That a DoRA adapter loads silently as plain LoRA (item 10): inferred from the code, not run.
- Whether an outside package can keep long-lived tensors with the public `Tensor.From` outside a `TensorScope`, as a
  substitute for the internal `Tensor.Persistent`.
- The exact line where `PretrainedModel.Load` calls `spec.Build` (around PretrainedModel.cs:110-125).
