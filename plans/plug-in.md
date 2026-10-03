# Plug-in points: what is still closed

What a user or an outside package cannot plug into Idrak today without changing the library, from a read-only survey
of the `architecture` branch (2026-10-02), and the dataset loader abstraction that follows from it. Items marked "Done" or
"Partly done" below have been built since; the rest is not built yet. File and line references are from that survey; check them again before starting an item.

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

Status (plugin/tensor-ops): done. `Tensor.Persistent`, `Tensor.PersistentZeros` and in-place writes (`CopyFrom`,
`Fill`, `Scale`, `AddScaled`) are public; `tests/Idrak.PluginTests` (not a friend assembly of Idrak or Idrak.Onnx)
writes a Lion optimizer, a softplus operation through `Autograd.Function` registered as a network step and an ONNX
import operator, a packed weight format and a KV cache format, and the runner in tests/Idrak.Tests runs them on every
device ("outside plug-in"). Still internal-only from outside: `KeyValueLayouts.Unregister` and an unregister for
`PackedWeight` (so the test registers names of its own and leaves them), the fast hooks of packed weights and KV
layouts, and loading a weights file into a plug-in packed format without rebuilding (item 22). `Tensor.Load` stays
internal (the first-party packages keep calling it); outside code uses `CopyFrom`.

## Ranked gaps

| # | Gap | Evidence | Who needs it | Suggested shape | Size | Priority |
|---|---|---|---|---|---|---|
| 1 | Done (plugin/tensor-ops): `CopyFrom`, `Fill`, `Scale`, `AddScaled`, public `Persistent`/`PersistentZeros`, `Optimizer.ApplyDecoupledWeightDecay`. Was: no public way to write values into an existing tensor | `Tensor.Load` internal (Tensor.cs:188); `AddInPlace`/`FillInPlace` internal (Tensor.Decoding.cs:274, 281); the protected helpers of `Optimizer` are only weight decay, state and gradient scale (Optimizer.cs:143-177) | Lion, Adafactor or Muon optimizers; an ONNX import translator's `load:` callback (OnnxImportOps.cs:77); custom initializers; an outside packed weight format | public `Tensor.CopyFrom(ReadOnlySpan<float>)`, in-place `AddScaled`/`Assign`, or a protected `Optimizer.Update(Tensor parameter, Tensor value)` | small | high |
| 2 | Done (plugin/tensor-ops): `Autograd.Function`; `Sqrt`, `Sin`, `Cos`, `Silu`, `Sign`, `Pow`, `Clamp`, `Maximum`/`Minimum`, `Where` (CPU and Vulkan kernels, CUDA through the host fallback). Still missing: Erf, Gather, tensor-by-tensor division, comparisons, a public `Checkpoint`. Was: no custom differentiable operation (own forward and backward); a small public operation set | `Record` private (Tensor.cs:685); `UnaryOp`/`BinaryOp` internal (Backends/Backend.cs:8-20); `Tensor.Checkpoint` internal (Tensor.Decoder.cs:71). No public Sqrt, Pow, Clamp, Max/Min, Where, Sin/Cos, Erf, SiLU, Gather; TextEncoder computes 1/norm through Log, scale and Exp (Retrieval/TextEncoder.cs:164) | RoPE variants, ALiBi, Mish, focal loss | `Autograd.Function(name, forward, backward)` composed from public operations (no backend access needed); more element-wise operations | medium | high |
| 3 | Done: `RopeScalings` (see "Done since the survey"). RoPE scaling is closed: linear and llama3 only | switch at Layers/Decoder.cs:113-143; config parsing at LanguageModels/Architectures.cs:110-121; GGUF drops YaRN with a note (GgufModel.cs:292-301); `RopeScaling` is a sealed record with llama3's fields (Decoder.cs:93) | YaRN (Qwen2.5 and Qwen3 long context, DeepSeek), dynamic NTK, LongRoPE (Phi-3, Phi-4) | `RopeScalings.Register(type, Func<double[] frequencies, JsonObject parameters, double[]>)` plus an optional attention scale; `RopeScaling` holds the type and the JSON parameters | small (frequencies), medium (YaRN's attention factor) | high |
| 4 | Partly done (see "Done since the survey"). Model families must fit the fixed `DecoderSpec` | `PretrainedArchitecture.Spec` returns a sealed `DecoderSpec` of fixed flags (Architectures.cs:17; DecoderSpec.cs:97-175); mixture of experts rejected (Architectures.cs:102-105); sliding window noted, not implemented (Architectures.cs:123-126); `FeedForwardActivation` (Decoder.cs:524-534) and `DecoderNorm` (DecoderSpec.cs:9-16) closed; no ALiBi, logit soft-capping or encoder-only models | Gemma 2/3 (soft-capping, alternating sliding window), Mixtral and Qwen-MoE, Phi-3 (fused qkv), BERT and BGE embedding models | an optional `Build` delegate on `PretrainedArchitecture` so a family builds its own module; later per-layer attention options and an activation registry | large | high |
| 5 | (done: `OnnxExportOps`, built-ins registered, `GraphModule` export; decoder layers and several inputs remain) ONNX export has no global registry and misses layer kinds | per-exporter `Module<T>`/`Lambda` only (OnnxExport.cs:90-101); built-ins in a closed switch (OnnxExport.cs:169-202); no `GraphModule` (an imported graph cannot be exported again), no decoder layers; one input only (OnnxExport.cs:70) | a package shipping a layer with its ONNX translator; round trips of imported graphs | `OnnxExportOps.Register<T>(OnnxTranslator<T>)` and `Unregister`, built-ins registered the same way; the per-exporter call keeps precedence | small to medium | high |
| 6 | (done: translators in graphs, `GraphOps`, `LayerTypes`, Clip/Pow/Sqrt/...; Erf, ConvTranspose and several inputs/outputs remain) ONNX import of graph models ignores registered operators | fixed `Primitives` map (OnnxImport.cs:336-344) and closed `MatchLayer` (402-413), documented at OnnxImportOps.cs:90-92; closed `GraphModule` operation switch (GraphModule.cs:159-233) and layer list (`LayerDescriptions`, GraphModule.cs:464-522); one input and output (OnnxImport.cs:125, 350) | a ResNet or U-Net with Erf, Pow, Clip or ConvTranspose; a custom layer inside a graph | `GraphOps.Register(op, ...)`; `LayerDescriptions` becomes a layer type registry (describe, create), shared with item 7 | medium | high |
| 7 | Model packages (.ikm) cannot store custom architectures, scalers or tokenizers | `BuildModel` handles DecoderSpec, GraphModule or builder JSON only (Inference/ModelPackage.cs:293-325); scalers in a closed switch (ModelPackage.cs:111-116) with the `PackageEntryKind` enum (:50); tokenizers limited to character and word (:118-126; Generation/Tokenizer.cs:282-298) | shipping a custom module, a robust scaler or a BPE model in one file | an architecture registry keyed by the description's format; `IScaler` gains a save method and a format name with a scaler registry; the same for tokenizers | medium | medium |
| 8 | Done (plugin/tool-calls): `IToolCallParser` and `ToolCallFormats` with json, pythonic, qwen3-coder, mistral, harmony and deepseek built in. Was: tool-call parsing handles only JSON between tags | sealed `ToolCallFormat(Open, Close, List, ArgumentsKey)` (Generation/Chat.cs:33); sealed, JSON-only `ChatOutputParser` (Chat.cs:211, 492); the detector probes that shape only (ChatTemplates.cs:67-120) | Llama 3 pythonic calls, Qwen3-Coder XML, Mistral `[TOOL_CALLS]`, GPT-OSS channels, DeepSeek special tokens | an `IToolCallParser` (feed text, finish, calls) created by the chat template; `ToolCallFormat` stays the default | medium | high |
| 9 (done) | Fine-tuning fixes the optimizer, schedule, loss and adapter type | AdamW (LanguageModels/FineTuning.cs:984-987), cosine schedule (:729-730), token cross-entropy (:1324, :1443); LoRA only, through a sealed `LoraAdapter` record with an internal setter (Layers/Linear.cs:121, 497) | other optimizers, a warm-up/stable/decay schedule, DPO, ORPO or KTO losses, DoRA | `FineTuningOptions.Optimizer`, `Scheduler` and loss delegates; adapters need an `IAdapter` on `Linear` | medium (adapters: large) | high |
| 10 (checks and DoRA done) | PEFT adapters: only plain LoRA, and the configuration is not checked | `LoadAdapter` and `AdapterMerge` read only `r`, `lora_alpha`, `use_rslora` (PretrainedModel.cs:250-283; Architectures.cs:240-246); `peft_type` and `use_dora` are never checked, so a DoRA adapter would likely load as plain LoRA (inferred from the code, not run) | loading DoRA, IA3 or VeRA adapters | first reject an unknown `peft_type` and `use_dora`; then an adapter type registry | small (checks), large (variants) | medium |
| 11 | Only BPE tokenizer models | other tokenizer.json models throw (BpeTokenizer.cs:87-90); GGUF accepts only the "gpt2" tokenizer model (GgufModel.cs:339-342); `BpeTokenizer` is always created (PretrainedModel.cs:138); `TokenizerComponents` has no model or post-processor entry (TokenizerComponents.cs:57-63) | WordPiece (BERT, BGE), Unigram/SentencePiece (T5, Gemma, Llama-2-style GGUF) | `TokenizerModels.Register(type, Func<JsonObject, ITokenizerModel>)` | medium to large | medium |
| 12 | The GGUF architecture registry cannot describe a family unlike Llama | `GgufArchitecture` holds only the Hugging Face name and query/key interleaving (GgufModel.cs:17-23); tensor names in a closed switch (:563-595); fixed config translation with `hidden_act` set to silu (:250-270) | Phi-3 (`attn_qkv`), Gemma 2 (post norms, GELU), Command R | `TensorName` and `Config` delegates on `GgufArchitecture` | small | medium |
| 13 | Generation: no stopping hook, logits processors, speculative decoding or structured output | string stops only (TextGenerator.cs:409-411, 470-488); the only plug point replaces the whole sampler (:89) and loses temperature, top-k and top-p; no logit bias (Sampling.cs); the chat API request has no `format` field (`ChatApiRequest`, AspNetCore/ChatApi.cs) | stopping on token ids or a pattern, logit bias, a JSON schema mask over the normal sampler, a draft model | `GenerationOptions.StopWhen`; an `ILogitsProcessor` run before the sampler; a speculative mode with a draft model | small (stop, bias), medium (processors), large (speculative) | medium |
| 14 | Training data is in memory only | `Trainer.Fit` takes a sealed `DataLoader` (Training/Trainer.cs:162), which takes a sealed `Dataset` of `float[]` (Data/DataLoader.cs:52-67; Data/Dataset.cs:32-35); `Batch` has an internal constructor (DataLoader.cs:11) | images too large for memory, augmentation each epoch, custom batching | the dataset loader abstraction below | medium | high |
| 13 | Generation: no stopping hook, logits processors, speculative decoding or structured output | string stops only (TextGenerator.cs:409-411, 470-488); the only plug point replaces the whole sampler (:89) and loses temperature, top-k and top-p; no logit bias (Sampling.cs); the Ollama request has no `format` field (AspNetCore/Ollama.cs:14-21) | stopping on token ids or a pattern, logit bias, a JSON schema mask over the normal sampler, a draft model | `GenerationOptions.StopWhen`; an `ILogitsProcessor` run before the sampler; a speculative mode with a draft model | small (stop, bias), medium (processors), large (speculative) | medium |
| 14 | Done (feature-data-loaders): sample sources, streams, transforms, batch sources and the loaders below. Was: training data is in memory only | `Trainer.Fit` takes a sealed `DataLoader` (Training/Trainer.cs:162), which takes a sealed `Dataset` of `float[]` (Data/DataLoader.cs:52-67; Data/Dataset.cs:32-35); `Batch` has an internal constructor (DataLoader.cs:11) | images too large for memory, augmentation each epoch, custom batching | the dataset loader abstraction below | medium | high |
| 15 | Metrics are batch means only; no optimizer state in checkpoints | `Metric(Name, BatchMean, Finalize)` (Trainer.cs:16) cannot express F1, AUC, precision or recall; `Tensor.MatchRate` internal (Tensor.Advanced.cs:197); the checkpoint callback saves weights only (TrainerCallbacks.cs:217-250) | classification users; resuming a run exactly | an `IMetric` accumulator (reset, update, compute); `Optimizer.State()` and `LoadState()` as named tensors | small to medium | medium |
| 16 | Evaluation metrics are closed and need a concrete `ChatGenerator` | `AnswerMetric` enum and switch (LanguageModels/Evaluation.cs:13-29, 126-133); `Run(ChatGenerator ...)` (:68); the CLI parses the enum (Idrak.Cli/Commands/Train/TuneTool.cs, `--metric`; formerly FineTuning.Cli/Program.cs:141) | ROUGE, pass@k for code, a model as judge, multiple-choice letters; evaluating a remote model | a scoring delegate or an answer metric registry; take `IChatModel` | small | medium |
| 17 | Serving: closed model kinds and options | `EngineModelKind` is predictor, text or chat (InferenceEngine.cs:16-26); `ChatModel(...)` accepts only a `TextGenerator` (:339-348); no embedding kind, no `/api/embed` or OpenAI `/v1` routes; unknown chat API options are dropped (`ChatApiTranslation.Options`, ChatApi.cs) | serving an embedder, a proxy model or a custom chat model; passing extra sampling options through | `ChatModel(name, Func<IChatModel>)`; an embedding model kind; `GenerationOptions.Extra` for unknown options | medium | medium |
| 18 | Retrieval: fixed pooling, chunking, keyword index and fusion | sealed `TextEncoder` with masked mean pooling only (TextEncoder.cs:18, 138-165); `ChunkUnit` is words or sentences (Chunking.cs:21-28); sealed `Bm25Index` and fixed fusion (RetrievalIndex.cs:30-31, 119-140); `VectorMetric` is dot or cosine (VectorIndex.cs:12-19) | CLS or last-token pooling (BGE, Qwen3-Embedding), token- or markdown-aware chunking, SPLADE | workarounds exist (`IEmbedder`, passing your own chunks); add a pooling choice, an `IChunker` and an `IKeywordIndex` | small each | low to medium |
| 19 | Datasets: registered file formats are skipped when picking hub files; compression and archives are closed | `FormatOf` maps only built-in formats (DataFiles.cs:121-122), so `RemoteFiles.IsDataFile`/`BestFormat` (Sources.cs:87-115) ignore registered ones; `.gz` only (DataFileFormats.cs:104; DataFiles.cs:225, 279); zip and tar only (DataFiles.cs:125-127) | Arrow or Feather files, `.jsonl.zst` corpora, `.tar.xz` archives | rank by the registered format (with a priority); `DataCompressions.Register(".zst", Func<Stream, Stream>)` | small | medium |
| 20 | Safetensors data types are closed | `SafeTensorType` is F32, F16 or BF16 (SafeTensors.cs:12-21, 283-286) | FP8 checkpoints (Qwen3-FP8, DeepSeek), I8/U8 (GPTQ, AWQ) | `SafeTensorTypes.Register(dtype, decoder)`, like `GgufTypes` | small | medium |
| 21 | Checkpoint export is closed | `SaveCheckpoint` is a closed switch over modules and throws for LayerNorm (PretrainedModel.cs:319-341); safetensors only; `ICheckpointFormat` reads only (CheckpointFormats.cs:12-33) | exporting a GPT-2-style model; writing GGUF | an optional `Write` on `ICheckpointFormat`; a reverse mapping per architecture | medium | low to medium |
| 22 | Packed weight plug-ins are second class | plug-ins get only the generic dequantize path; the fast hooks are internal virtual (Layers/Quantization.cs:129-152); weights files record built-in packed layers by index only (Layers/Module.cs:232-262), so a plug-in model must be rebuilt in the same format before loading (PackedPluginTests.cs:177-181); no `Pack(format)` extension (ModuleExtensions.cs:195-262); `idrak tune -w` takes only int8, int4 and bf16 (Idrak.Cli/Commands/Train/TuneCommand.cs; the removed idrak-tune had no `--weights` option) | NF4, GPTQ or FP8 formats | record the format name in the weights file header; `Module.Pack(format, filter)`; `--weights NAME` | small to medium | medium |
| 23 | Jinja filters, tests and functions are closed | filter switch ends with "Unknown filter" (JinjaBuiltins.cs:628-845), tests at :1006; the callable delegate is internal (Jinja.cs:73), so `JinjaChatTemplate.Variables` (ChatTemplates.cs:178) cannot add functions | templates using a filter Idrak lacks; a fixed `strftime_now` for reproducible tests | `Functions` and `Filters` dictionaries on `JinjaChatTemplate` | small | low to medium |
| 24 | Fixed initializers; closed activation and precision enums | Xavier-uniform in `Linear` (Layers/Linear.cs:28-29) with no hook; builder `Activation` enum (NetworkBuilder.cs:78-87, 195-201); sealed activation layers (Activations.cs); `MatMulPrecision` is float32, bfloat16 or FP8 (MixedPrecision.cs:7-55) | Kaiming initialization; SiLU or LeakyReLU in `Dense`; FP16 on GPUs without bfloat16 | initializer delegates (needs item 1); activations by `NetworkOps` name; new precisions need the public backend (12c) | small / medium | low |
| 25 | Tool results are text only | `Tool.Invoke` returns `Task<string>` (Generation/Tools.cs:36); MCP results are flattened to text (Mcp/McpTools.cs:93-101) | image or structured tool output for multimodal models | `ToolResult` content blocks | medium | low |
| 26 | Done for `idrak` (`-P, --plugin PATH` and the config's `plugins`, loaded before every command, `idrak tune` and `idrak data` included; the separate `idrak-tune` and `idrak-data` tools were removed). The command-line tools cannot load plug-ins | no assembly loading or `--plugin` option in src or samples | using any registry from the fine-tuning or dataset commands | `--plugin path.dll`, calling a module initializer or an `IIdrakPlugin.Register()` | small | medium |

### Item 8: what remains

- Formats not built in (no verified template at hand, or a different shape): Hermes-style tags with other JSON
  shapes (GLM 4.5 `<arg_key>`/`<arg_value>`, Kimi K2 `<|tool_call_begin|>functions.name:0`, MiniMax `<invoke>`,
  Command R, Granite, FunctionGemma). Each can be registered with `ToolCallFormats.Register`.
- The pythonic community template writes string values with Python's `str()` (unquoted), so calls with string
  arguments do not round-trip through that template exactly; numbers, booleans and lists do.
- Calls are emitted when complete (no partial-argument streaming), as before.
- Llama 3.1's built-in tools (`<|python_tag|>name.call(...)`) are parsed, but the template renders them back only
  when `builtin_tools` is passed as a template variable.

## Done since the survey

- Item 3, RoPE scaling: `RopeScalings.Register(type, RopeScalingMethod)` with `Unregister`, `Names`, `Contains` and
  `Get` (names ignore case; unknown names list the registered ones). A method receives the unscaled frequencies, θ,
  the rotary size and the JSON parameters, and returns the frequencies, an attention factor on the cos/sin tables
  (YaRN's mscale) and optionally per-position frequencies. `RopeScaling` holds the type and the raw parameters;
  config.json and GGUF (rope.scaling.type, factor, original_context_length, yarn_beta_fast/slow) forward them. Built
  in: linear, llama3 (results unchanged), yarn and dynamic, following transformers' modeling_rope_utils.py. Limitation:
  dynamic NTK rotates each token with the frequencies of its own position, as transformers does token by token; a
  prompt longer than max_position_embeddings given in one pass differs from transformers for its earlier tokens.
  Not done: LongRoPE (Phi-3/4) as a built-in; GGUF's yarn_log_multiplier and attn_factor keys.
- Item 4, first slice:
  - `PretrainedArchitecture.Build` (a delegate receiving `PretrainedBuildContext`: config, spec, weights by Idrak
    name, the raw checkpoint, build options, notes) builds a family's own `Sequential`; `Spec` still gives the sizes.
    `PretrainedArchitectures.Unregister` and null checks added.
  - Sliding-window attention per layer: `DecoderSpec.SlidingWindow`, `SlidingWindowLayers`, `SlidingWindowRope`
    (Gemma 3's local base); `CausalSelfAttention.SlidingWindow`. Read from config.json for Mistral (every layer),
    Qwen2/Qwen3 (`use_sliding_window`, `max_window_layers`), `layer_types`, and Gemma 2/3 (alternating defaults).
    At first the device kernels masked causally only, so such layers attended through basic operations; the windowed
    kernels below replaced that.
  - Soft-capping and score scale: `DecoderSpec.AttentionSoftcap`, `LogitSoftcap`, `AttentionScale`
    (`CausalSelfAttention.ScoreSoftcap`, `ScoreScale`; `Linear.OutputSoftcap` on the head, so fine-tuning losses and
    answer scoring see capped logits).
- Item 4, windowed and soft-capped kernels (done):
  - `AttentionVariant` (internal: window, cap; the default is plain causal attention) goes to every attention method
    of `Backend`. A query whose causal limit is `end` (exclusive) starts at max(end - window, 0), as transformers
    masks (keys above the query's position minus the window); the scaled score s becomes cap · tanh(s / cap) before
    the softmax, and the gradients take the cap's slope 1 - (s / cap)².
  - CPU: the reference (tiled, decoding over float32, int8 and bfloat16 caches, per-row, packed, both gradients).
  - Vulkan: `attention_decode`/`_bf16`/`_int8` (the splits share the window's positions; a window shorter than the
    capacity is measured under its own tuning key, over the window), `attention_tiled`/`_lse` (tiles from the block's
    smallest window start; a row with no position yet in a tile keeps alpha 1) and `attention_backward_dq`/`_dkv`.
    Two push constants each (window, cap), selected off for plain attention: the same results as before.
  - CUDA (not run here): `attention_decode_f32`/`_int8`/`_bf16` (window start before the splits, cap after the
    scale), `attention_flash_f32`/`_int8`/`_bf16` (a start per row, the block's tiles from its smallest) and
    `attn_bwd_kv_f32`/`attn_bwd_q_f32` (mask and slope); window and cap are predicated, so plain launches execute the
    old instructions. Windowed or soft-capped layers do not use the tensor-core flash kernels (prompt and training go
    to `attention_flash_f32` and its float32 gradient; packed sequences and per-row starts are refused there, so
    `PackedSequences.Supports` and `TextGenerator.SupportsBatches` say no on CUDA for these layers).
  - `KeyValueLayout.AttendVariant` (internal) gives the built-in formats' kernels a variant; a format of one's own
    returns null and the layer takes the composed path. Packed sequences and rows of different lengths pass the
    variant (`CausalSelfAttention.SupportsSegmented`). Recorded steps keep the kernels past the window.
  - `IDRAK_WINDOW_KERNELS=0` (`CausalSelfAttention.WindowKernels`) forces the composed path for windowed and
    soft-capped layers, refusing packing and per-row starts as before.
  - Tests: "window kernels" (every kernel against a direct reference written in the test; a window beyond the keys is
    the plain result bit for bit; gradients by finite differences and against the CPU; the Vulkan kernels against the
    CPU at every width and split count with no host fallback; Mistral, Gemma 2 and Gemma 3 style decoders through the
    kernels against the composed path: prompt, training and every gradient, cached decoding in every format, recorded
    steps, packed sequences, rows of different lengths).
  - Measured on lavapipe (`--bench-vulkan window`): decoding attention over a 4096-position cache, 62 ms without a
    window, 7.1 ms with a window of 512; a windowed, soft-capped decoder (dim 256, 4 layers, window 128, a context of
    1024) 42 → 67 tokens/s. Its 630-token prompt took 3.2 s through the tiled kernel against 2.5 s composed: on a CPU
    driver one large matrix product beats the tiled kernel for prefill, as for plain layers (the tiled kernel's choice
    is measured against the decoding kernel only). To measure on a GPU: `IDRAK_DEVICES=vulkan:0 dotnet run -c
    Release --project tests/Idrak.Tests -- --bench-vulkan window`.
  - Not done, the ring-buffer cache of the window's size: it does not fit the current design. `DecodingContext.Truncate`
    and the generator's prompt reuse rewind to earlier positions, which a ring has overwritten; a prompt longer than
    the ring in one step would overwrite keys its own earlier queries read (prefill would have to go in window-sized
    chunks); `KeyValueLayout` is public and addresses slots by position (`Write` at a position, `Expand` to
    [rows, capacity]), as do the composed path's masks; every write and read kernel (the fused projection's cache
    write on every backend included) would take positions modulo the ring. A plan: per-layer capacity window +
    largest step, slot = position mod capacity in the layouts' kernels, prefill chunked to the window, `Truncate`
    allowed only within the positions still held; a layout property so custom formats opt in.
  - Not done: Vulkan kernels for packed sequences and per-row starts (Vulkan runs both through the host fallback,
    the CPU reference, for plain layers too); the tensor-core flash kernels with a window (CUDA packing for these layers).
  - Gemma 2 and Gemma 3 (text, `Gemma3ForCausalLM`) registered, with their four norms per layer
    (`PretrainedArchitectures.GemmaTensorName`). gelu_pytorch_tanh was already read; gelu_fast added.
- Remaining for item 4:
  - A ring-buffer KV cache of the window's size (see "windowed and soft-capped kernels" above for why and a plan);
    Vulkan packed and per-row kernels; a window in the CUDA tensor-core flash kernels.

  - An activation registry (the closed `FeedForwardActivation`) and `DecoderNorm`; ALiBi; encoder-only models (BERT,
    BGE) and fused qkv (Phi-3), which the `Build` delegate already allows as a family's own network.
  - GGUF families unlike Llama (item 12: Gemma 2/3 tensor names, sliding_window key).
  - Mixture of experts: done (below); grouped expert kernels remain.

### Plan: mixture of experts (Mixtral, Qwen2-MoE, Qwen3-MoE)

Status (feature-moe): done as planned, with these choices and what remains:

- `MixtureOfExperts` (src/Idrak/Layers/MixtureOfExperts.cs): router ("router", float32 in every build, also when the
  experts are packed: its rounding could change the routing), experts ("experts.j"), optional shared expert ("shared")
  and its gate ("shared_gate", [dim, 1]). The router's softmax is read back to the host and the top-k chosen there
  (`Route`, ties to the lowest index); each expert gathers its rows (the embedding lookup's gather/scatter, so the
  gradients flow), the outputs are stacked, gathered back per token and combined with one batched product [tokens, 1, k]
  × [tokens, k, dim]. With gradients, the weights are gathered from the softmax (and divided by their sum), so the
  router trains. Composed from existing operations, so it runs and trains on every backend with the CPU's results.
  `LoadBalancingLoss` (transformers' definition, so router_aux_loss_coef applies) and
  `FineTuningOptions.LoadBalancingWeight`.
- `DecoderBlock.FeedForward` is a `Module` (breaking for code reading `.FeedForward.Up`; cast to `FeedForward`); the
  fused down projection + residual + norm path stays for `FeedForward`.
- `DecoderSpec.Experts`, `ExpertsPerToken`, `ExpertFfDim`, `SharedExpertFfDim`, `NormalizeTopK`, `ExpertLayers` (Qwen's
  decoder_sparse_step and mlp_only_layers), `IsExpertLayer`, `ActiveParameterCount`; written to JSON only for models
  with experts. Weight names `layers.i.mlp.router`, `layers.i.mlp.experts.j.gate/up/down`, `layers.i.mlp.shared.*`,
  `layers.i.mlp.shared_gate`.
- Families: `MixtralForCausalLM`, `Qwen2MoeForCausalLM`, `Qwen3MoeForCausalLM` with `PretrainedArchitectures.ExpertSpec`,
  `MixtralTensorName`, `QwenMoeTensorName`; `CommonSpec` still refuses expert keys so dense families never misread
  them. GGUF: qwen2moe, qwen3moe, and llama files with experts as Mixtral (`GgufArchitecture.WithExperts`,
  `NormalizeTopK`); stacked expert tensors (`ffn_*_exps`) are read one expert at a time, the older per-expert names
  (`ffn_gate.j`) too, and layers without `ffn_gate_inp` become mlp_only_layers.
- Recorded graphs: a decoding step through experts cannot be recorded (the routing is read back): `TextGenerator`
  does not try, `ComputeGraph.Capture` fails cleanly with a reason (replays run the step), and fine-tuning does not
  record steps of models with experts.
- `idrak families` has an Experts column (the probe with expert keys added).
- Tests (MixtureOfExpertsTests.cs): the routing and its ties, a known routing on the device, three family-shaped
  decoders against the loop reference (prompt, training path, cached decoding, two-row batches), packed experts
  (device against CPU, near float32, cached against prompt), recording falls back, gradients against finite
  differences, the load-balancing loss, LoRA fine-tuning, configs, and checkpoints in each family's names as
  safetensors and GGUF; CPU and Vulkan (lavapipe).
- Remaining: device-side routing and grouped expert kernels (one launch for all experts' gate/up and down products,
  reading only the chosen experts' packed weights), which would remove the per-layer host round trip, make decoding
  recordable and batch the experts' products; until then a GPU decodes a mixture-of-experts model slower than a dense
  model of its active size. Checked only on CPU and lavapipe here; CUDA and HIP run the same composed operations
  (gather, scatter-add, batched products), not run here. Expert parallelism across devices is not planned.

- A `MixtureOfExperts : Module` beside `FeedForward`: a router `Linear(dim, experts)` ("router"), `experts` ×
  `FeedForward` ("experts.i"), and for Qwen2-MoE a shared expert with a sigmoid gate. Forward on the CPU first:
  router logits → softmax over experts → top-k per token (Mixtral k = 2; Qwen `num_experts_per_tok`) → renormalize the
  k weights when `norm_topk_prob` (Mixtral always does; Qwen3-MoE per config) → per expert, gather its tokens, run it,
  scatter-add weight · output. Training needs gather/scatter with gradients (item 2's public operations) and, for
  fine-tuning, a load-balancing auxiliary loss (optional).
- `DecoderBlock` takes a `Module` for its feed-forward part (today `FeedForward`); the fused inference paths stay for
  `FeedForward` only.
- `DecoderSpec.Experts`, `ExpertsPerToken`, `ExpertFfDim`, `SharedExpertFfDim`, `NormalizeTopK`; names
  `layers.i.mlp.router`, `layers.i.mlp.experts.j.gate/up/down`; Mixtral's checkpoint names
  (`block_sparse_moe.gate`, `experts.j.w1/w3/w2`) and Qwen's (`mlp.gate`, `mlp.experts.j.*_proj`,
  `shared_expert`, `shared_expert_gate`) in their `TensorName`.
- Tests: a tiny random model against a loop reference with a known routing; ties broken as torch.topk does (lowest
  index first); then GPU kernels (grouped products per expert) and packed expert weights.

### Items 9 and 10: what was done

- `FineTuningOptions.Optimizer` (factory from the trainable parameters), `Scheduler` (from the optimizer and the step
  count) and `Loss` (a `FineTuningLoss` over the trained tokens' log-probabilities); null keeps AdamW, cosine and the
  fused token cross-entropy with identical results. Built-ins: `FineTuningOptimizers`, `FineTuningSchedules` (cosine,
  linear, constant, warm-up/stable/decay), `FineTuningLosses` (token cross-entropy, DPO, ORPO, SimPO).
- Recorded steps record only the forward and backward pass, so any optimizer that keeps its gradient buffers is
  recorded; the CPU update (and any optimizer that releases or replaces the buffers, checked before recording and before
  every replay) runs ordinary steps. A custom loss runs ordinary steps (its gradient becomes the weights of a token
  cross-entropy pass). Not run on CUDA here: on Mesa's software Vulkan the recording itself falls back (a read during
  capture), so the recorded path with a custom optimizer is covered by construction and by CUDA runs to come.
- Preference data: `RowKind.Preference`, `ChatRows.Preference`, `ChatTranscriptEncoder.EncodePreference`,
  `PreferencePair` and `FineTuner.Train(model, pairs, ...)`. DPO's reference: adapters disabled
  (`ModuleExtensions.DisableAdapters`), computed once per sequence.
- `ILinearAdapter` on `Linear.Adapter` (LoRA one implementation, `Linear.Lora` for the fused paths) and `DoraAdapter`
  with PEFT save/load (`use_dora`, `lora_magnitude_vector`), merging (in memory, while loading, ONNX export).
- PEFT checks: another `peft_type`, `rank_pattern`, `alpha_pattern`, trained biases, `lora_bias`, `modules_to_save`,
  `layer_replication`, `trainable_token_indices`, `use_qalora`, and tensors no layer takes are refused; `use_rslora`
  now applies to `LoadAdapter` too.

Still open: KTO (unpaired data and a KL reference point) and IPO; a loss over hidden states rather than log-probabilities;
packing for preference batches; per-module ranks; IA3 and VeRA as `ILinearAdapter`s with a PEFT type registry; DoRA
through the fused LoRA kernels (its products run unfused today).

## Gaps inside the existing registries

Items 5 and 6 (done): `OnnxExportOps` (`Register<T>`/`Unregister<T>`, `RegisterLambda`, `RegisterGraphOp`), `GraphOps`
and `LayerTypes` follow `OnnxImportOps`: ordinal names (types for export), register-or-replace, public unregister,
built-ins registered the same way, errors listing the registered names. What remains: decoder layers have no export
translator; graphs and exports have one input and one output (`Module.Forward` takes one tensor); Erf (outside
GELU), Floor/Ceil, Reciprocal, tensor-by-tensor Div and ConvTranspose need tensor operations (item 2); the graph
structural operations (shape arithmetic) cannot be replaced; outside translators that load weights still need public
tensor writes (item 1) because `OnnxImportContext.Add`'s `load:` callback has only internal `Tensor.Load`.

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
   JSON Lines and Parquet adapter in `Idrak.Datasets` (proposed); or everything in `Idrak.Datasets`. Chosen: the
   proposal (`Idrak.Datasets` now references `Idrak`).
2. Images: PGM, PPM, BMP and PNG built in with JPEG as a plug-in (proposed); or PGM and BMP only built in. Chosen: the
   proposal.

Done when: `Trainer.Fit` trains from each built-in loader with the same results as from an in-memory `Dataset` of
the same samples; a custom `ISampleSource` and a custom `IBatchSource` written in a test project without internal
access train a model; the OCR and shape-recognition samples use the image folder loader with an augmentation
transform.

### Status (feature-data-loaders): done

Built as planned, in `src/Idrak/Data` and `src/Idrak.Datasets/TableSamples.cs`:

- `ISampleSource`, `ISampleStream` (opened once per epoch as an `ISampleReader`), `ISampleTransform` (given a `Random`
  seeded from the loader's seed, the epoch and the sample, so prefetching on a worker thread changes nothing) and
  `IBatchSource` (optional `BatchCount`, `SampleCount`, `BatchSize` as default interface members). `Trainer.Fit`,
  `FitAsync`, `TrainAsync`, `Evaluate` and `TrainingRun` take `IBatchSource`; `Trainer.Predict` takes `ISampleSource`;
  telemetry reports 0 for counts a batch source does not give. `Batch` has a public constructor.
- `Dataset : ISampleSource` and `Dataset.FromSource`; `DataLoader(ISampleSource ...)` keeps the shuffle order of the
  old loader exactly (existing results unchanged) and adds `Transforms`; `DataLoader(ISampleStream ..., shuffleBuffer)`
  batches streams (counts known after the first epoch).
- `CsvSource` (one pass for row offsets, then positional reads; `AsStream()`), `ImageFolderSource` (class folders, a
  file list, or unlabelled files), `TokenFileSource` and `NpySource` (memory-mapped), `TableSamples` (Datasets: `Load`,
  `FromRows`, `Stream`, `Factory`), views (`Subset`, `Shuffle`, `Split`, `Concat`, `ToDataset`), transforms
  (`RandomFlip`, `RandomShift`, `RandomRotation`, `GaussianNoise`), `SampleSources` (csv, images, tokens, npy) and
  `ImageCodecs` (png, bmp, netpbm; `IImageCodec` for JPEG or others).
- Beyond the plan: `ImageCodecs` is a registry of its own (the folder source picks up any registered codec's
  extensions), the PNG decoder reads every bit depth and interlacing (the tool's old one read 8-bit, non-interlaced
  only), and BMP reads palettes and bit fields.
- The command-line tool's `train`, `predict` and `suggest --search` read images through `ImageFolderSource`; its own
  decoder is gone (it still reads JPEG headers for profiling).
- Tests: the "data loaders" group (each source trains to the weights of the same samples in memory; codecs against
  encoders written in the test; transforms against direct computation) and two "outside plug-in" tests (a sample
  source and a batch source in tests/Idrak.PluginTests).

Remaining: a JPEG codec stays a plug-in (none ships); the tool does not apply prep.json's "augment" list yet; no
loader for audio or video; `ISampleSource` reads one sample at a time (a batched read for sources that can do better,
such as a memory-mapped array, could come later).

## Proposed order

1. Plug-ins from outside the library: public tensor writes (item 1) and a test project without internal access.
2. Quick, high-value registries: RoPE scaling (item 3), ONNX export (item 5), `--plugin` for the command-line tools
   (item 26), registry hygiene (unregister, collisions, one name-matching rule).
3. Dataset loaders (item 14 and the section above). Done.
4. Tool-call parsers (item 8, done) and fine-tuning options (item 9).
5. Custom differentiable operations and more element-wise operations (item 2).
6. Model families beyond `DecoderSpec` (item 4), then tokenizer models (item 11) and GGUF families (item 12).
7. The rest by priority; anything that needs the public backend waits for item 12c.

## Not confirmed by the survey

- That a DoRA adapter loads silently as plain LoRA (item 10): inferred from the code, not run.
- (Resolved: `Tensor.Persistent` is public now.) Whether an outside package can keep long-lived tensors with the public
  `Tensor.From` outside a `TensorScope`.
- The exact line where `PretrainedModel.Load` calls `spec.Build` (around PretrainedModel.cs:110-125).

## Noted for later

### Provider-neutral names instead of "Ollama"

The serving API and its types are named after one product: `MapOllamaApi`, `OllamaApiOptions`, `OllamaChatRequest`,
`OllamaChatResponse`, `OllamaMessage`, `OllamaToolCall`, `OllamaModel` and others in `src/Idrak.AspNetCore/Ollama.cs`
and `IdrakEndpoints.cs`, and "Ollama-compatible" in the READMEs, samples and tests. The library should name no
provider: the endpoints are a local chat API that common clients understand, and further wire formats (an OpenAI-style
`/v1` API, item 17) should sit next to it under the same neutral scheme.

- Rename the public types and methods to neutral names (for example `MapChatApi`, `ChatApiOptions`, `ChatApiRequest`,
  `ChatApiMessage`), with the wire format named as a format option rather than in every type name.
- Keep the old names for one release as `[Obsolete]` forwarders, so applications move without breaking.
- The routes stay the same (`/api/chat`, `/api/tags`, ...): clients depend on them, not on the C# names.
- Check the other mentions before renaming them: `GgufModel.cs`, `Gguf.cs` and `ModelSource.cs` mention Ollama's model
  store as a source of GGUF files, which is a different concern (a model source), and documentation and samples
  (README, Idrak.AspNetCore and Idrak.Cli READMEs, the Chat, TextGeneration and GptApi samples, tests).
- Done when: no public type, method or option in the library is named after a provider; the old names compile with an
  obsolete warning; README and samples describe the API without a product name.

Done (branch `feature-neutral-serving`):

- `MapChatApi`, `ChatApiOptions`, and the wire types `ChatApiRequest`, `ChatApiResponse`, `ChatApiMessage`,
  `ChatApiTool`, `ChatApiFunction`, `ChatApiToolCall`, `ChatApiCalledFunction`, `ChatApiModelTag`,
  `ChatApiModelDetails`, `ChatApiRunningModel` and `ChatApiTranslation` (src/Idrak.AspNetCore/ChatApi.cs). The scheme
  is one `Map...Api` and one options type per wire format, as `MapCompletionsApi` and `CompletionsApiOptions` already
  were for `/v1`, rather than a format option on one method: the two formats differ in routes, bodies and streaming,
  and a further format (item 17) is added the same way. Endpoint names are `Chat-NAME`, `ChatTags-NAME`, ... .
- The former names are removed rather than kept as obsolete forwarders: nothing was released under them (versions
  are 0.y.z), so there is no one to stay compatible with.
- The model source: `ModelSource.LocalStoreModel` (the former name is removed), the source is
  named "store" and reads `store:NAME`. It still reads that store's folder, its
  `OLLAMA_MODELS` variable and its manifest layout, which belong to the store, not to Idrak's names.
- A test scans the public types, members and parameters of every Idrak assembly for provider names (obsolete members
  allowed). Documentation (README, the AspNetCore and FineTuning.Cli READMEs, the samples) names no product; test data
  that happens to name a product (version-question strings) was left as data.
- Remains: remove the forwarders in the release after next.

### Teacher pattern (knowledge distillation)

A larger "teacher" model guides a smaller "student" model, to be added later:

- Distillation loss: the student trains on the teacher's output distribution (soft targets with a temperature,
  Kullback-Leibler divergence), alone or mixed with the usual cross-entropy on the labels; per token for language
  models, per sample for classifiers. It plugs into the fine-tuning loss hook (item 9) and the `Trainer`.
- Teacher-generated data: the teacher writes answers (or reasoning) for prompts, and the student fine-tunes on them;
  a recipe in `Idrak.Datasets` and the fine-tuning CLI.
- Memory: teacher logits computed on the fly (both models loaded, the teacher in inference mode, int8 or int4 weights)
  or precomputed and stored (top-k logits per token, to keep files small).
- Requirements to check: teacher and student share a tokenizer (or a vocabulary mapping is needed); the teacher can
  run on another device than the student.
- Done when: a student fine-tuned with distillation scores closer to its teacher than the same student fine-tuned on
  labels alone, in a test with tiny models, and the CLI offers it.

Status (feature-distillation): done.

- Loss, language models: `FineTuningLossInput.TeacherDivergence(T)` gives KL(q_T ‖ p_T) per trained token, recorded,
  so any `FineTuningLoss` can use it; `FineTuningLosses.Distillation(temperature = 1, alpha = 0.5)` is α · T² · KL +
  (1 − α) · the token cross-entropy, averaged over the step's trained tokens (the default loss when
  `FineTuningOptions.Teacher` is set and `Loss` is not). The trainer handles it as it handles the log-probabilities:
  values first without gradients, then `Tensor.TokenDivergenceRows` over the same rows weighted by their gradient (row
  t's logits get g_t · (p − q) / T, back-propagated through the head chunk by chunk), so no [tokens, vocabulary] logits
  are stored. Each temperature costs one more pass of both heads over the trained rows.
- Loss, classifiers: `Losses.Distillation(student, teacher, T)` (T² · mean KL) and `Losses.Distillation(T, alpha)` for
  the `Trainer`, whose targets are the teacher's logits followed by the labels (one-hot or index), built by
  `Distillation.WithTeacher(data, teacher)` (the teacher's logits computed once, in inference mode on its own device).
  Classifier teachers are stored with the targets rather than run on the fly: the loss receives (prediction, target),
  not the input, and the logits of a dataset are small.
- Teacher on the fly: `DistillationTeacher.FromModel(model, topK = 0)`: padded batches of the step's sequences, run
  without the student's packing (`PackedSequences.Suspend`), on the teacher's own device and weight format; only its
  hidden states at the trained positions are kept between the two passes of a batch, the head runs per chunk and the
  probabilities move to the student's device. A head wider than the student's is cut (renormalized), a narrower one
  padded with zeros, so a family's padded vocabularies work.
- Stored: `TeacherLogitsWriter` (and `.Write(path, teacher, sequences, topK)`) writes the top k logits per trained
  token as 32-bit ids and 16-bit floats relative to the position's largest, keyed by the sequence's content, with the
  teacher's vocabulary fingerprint and an index; `TeacherLogitsFile` reads them; `DistillationTeacher.FromFile` trains on
  them as the softmax over the k stored tokens, renormalized (documented as an approximation that drops the tail).
- Vocabulary: `DistillationTeacher.CheckVocabulary` (same token text for every id; padding ids count as empty) and
  `VocabularyFingerprint`; training checks the teacher first and refuses a mismatch with the first id that differs and
  the advice to use teacher-written data. A vocabulary mapping is not built (left for later, if asked for).
- Teacher-generated data: `TeacherData.Generate(chat, rows, TeacherDataOptions)` (in Idrak.LanguageModels, next to
  `ChatTranscriptEncoder`, since Idrak.Datasets does not reference the generators): prompts from any chat layout or a
  prompt column, greedy answers in batches, optionally with the teacher's reasoning (`reasoning_content`), answers cut
  off by the length limit left out unless kept.
- CLI: `idrak distill` (src/Idrak.Cli/Commands/Train/DistillCommand.cs) runs idrak tune's training with a
  `TuneTool.BeforeTraining` hook: on the fly, `--precompute FILE` then `--teacher FILE`, or `--generate`; `--temperature`,
  `--alpha`, `--top-k`, `--teacher-weights`, `--teacher-device`, and tune's options (`--lora-alpha` for tune's
  `--alpha`).
- Tests (tests/Idrak.Tests/DistillationTests.cs and the CLI group): the divergence and its gradient against a dense
  computation; the classifier loss and gradient against the formula; the fine-tuning loss and the adapters' gradients
  (one SGD step) against both models' full logits, with the teacher on the CPU for a GPU student too; top-k storage
  round trip and the stored teacher's loss against the renormalized formula; mismatch refusal; teacher-written data;
  and the done-when test (held-out KL to a tiny teacher after 16 epochs: 5.91 before, 5.50 from labels sampled from the
  teacher, 5.26 distilled).
- Not measured here: speed and memory with real models on a GPU (the teacher's head runs twice per batch).
