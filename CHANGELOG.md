# Changelog

## Unreleased

- Idrak.AspNetCore: provider-neutral names for the chat API (`/api/chat`, `/api/tags`, `/api/ps`, `/api/version`,
  routes unchanged): `MapChatApi`, `ChatApiOptions` and the `ChatApi*` wire types (`ChatApiRequest`,
  `ChatApiResponse`, `ChatApiMessage`, `ChatApiToolCall`, `ChatApiModelTag`, ... , `ChatApiTranslation`). The former
  names are removed (nothing was released under them). Endpoint names are now `Chat-NAME`, `ChatTags-NAME`, ... .
- Idrak.LanguageModels: the local model store source is `ModelSource.LocalStoreModel` and the "store" source
  (`store:NAME`); `ModelSource.OllamaModel` is an obsolete forwarder and the former prefix is still read.
- `idrak serve` and `ui` listen on port 7317 by default (it was 11434, another local server's default); `-p/--port`,
  then `IDRAK_PORT`, then the config's `serve.port` choose another, and `-p 11434` serves clients that expect that
  port. The client commands (`api`, `ping`, `server ...`) take `--port` or `IDRAK_PORT`, then the server started
  last, then `serve.port` and 7317.
- Mixture-of-experts language models: Mixtral, Qwen2-MoE and Qwen3-MoE from Hugging Face folders and GGUF files
  (qwen2moe, qwen3moe, llama with experts). `MixtureOfExperts` (router, top-k routing with ties to the lowest index,
  renormalization per family, experts run only on their tokens, a gated shared expert), on every backend and in
  training (gradients to the router; `LoadBalancingLoss`, `FineTuningOptions.LoadBalancingWeight`); `DecoderSpec`
  `Experts`, `ExpertsPerToken`, `ExpertFfDim`, `SharedExpertFfDim`, `NormalizeTopK`, `ExpertLayers`,
  `ActiveParameterCount`; `PretrainedArchitectures.ExpertSpec`, `MixtralTensorName`, `QwenMoeTensorName`;
  `GgufArchitecture.WithExperts` and `NormalizeTopK`. Experts can be packed (int8, int4, bfloat16; the router stays
  float32). Decoding steps of these models are not recorded as graphs (the routing is read back per layer).
  `idrak families` shows which families read experts. Breaking: `DecoderBlock.FeedForward` and its constructor take
  any `Module` (cast to `FeedForward` where its projections are read).
- Data loaders: `ISampleSource` (random access), `ISampleStream` (in order), `ISampleTransform` (per sample, seeded by
  the loader's seed, the epoch and the sample) and `IBatchSource` (what `Trainer.Fit`, `Evaluate`, `FitAsync`,
  `TrainAsync` and `TrainingRun` now take; `Batch` has a public constructor). `Dataset` is a sample source and
  `Dataset.FromSource` reads any source into memory; `DataLoader` batches any source or stream (shuffle buffer for
  streams) with the same shuffling, dropLast, prefetch and device placement, and `Transforms`. Built in:
  `CsvSource` (large CSV files read lazily, the samples of `Dataset.LoadCsv`), `ImageFolderSource` (class folders),
  `TokenFileSource` (memory-mapped token ids in windows for pretraining), `NpySource` (memory-mapped .npy arrays),
  views (`Subset`, `Shuffle`, `Split`, `Concat`), image transforms (`RandomFlip`, `RandomShift`, `RandomRotation`,
  `GaussianNoise`), and the `SampleSources` registry (csv, images, tokens, npy). Image codecs without dependencies in
  `ImageCodecs`: PNG at every depth and colour type, interlaced or not (zlib from .NET), BMP 1 to 32 bits, PGM/PPM;
  JPEG and other formats register an `IImageCodec`. Idrak.Datasets: `TableSamples` turns JSON Lines, JSON, CSV and
  Parquet columns (numbers, booleans, arrays, class names) into a dataset or a stream; the package now references
  Idrak. `Trainer.Predict` takes any source. The OCR and shape-recognition samples train from image folders with
  augmentation; the OCR sample reads PNG and BMP pages too.
- idrak: `train`, `predict` and `suggest --search` read images through the library's `ImageFolderSource` and codecs
  (the tool's own decoder is gone, JPEG headers are still profiled); PNG files of 16 bits, packed bits or interlacing
  and BMP files of 1 to 16 bits are now read too, and a registered codec's formats are picked up.
- Sliding windows and soft-capped scores in the attention kernels: the CPU reference, Vulkan (decoding attention with
  split positions over float32, int8 and bfloat16 caches, the tiled prompt and training kernel, the gradient kernels)
  and CUDA (the decoding kernels, `attention_flash_f32`/`_int8`/`_bf16` and the float32 gradient kernels; windowed
  and soft-capped layers leave the tensor-core flash kernels for these) start each query at its window and cap its
  scores, so windowed layers (Mistral, Qwen2/3, Gemma 2/3) read only their window of the cache instead of scoring
  every cached slot, and soft-capped layers (Gemma 2) stop taking the composed path. Plain layers run the same
  arithmetic as before (window and cap are uniform and predicated off). Recorded decoding steps stay on the kernels
  past the window (the start is computed on the device). Packed fine-tuning batches and batched generation now work
  for windowed and soft-capped layers where the device's packed attention takes a window (CPU, Vulkan; not the CUDA
  tensor-core packed kernels). `IDRAK_WINDOW_KERNELS=0` forces the earlier composed path. On lavapipe a windowed,
  soft-capped decoder (dim 256, 4 layers, window 128, a 1024-position context) generates 67 tokens/s instead of 42;
  decoding attention over a 4096-position cache with a window of 512 takes 7.1 ms instead of 62 ms without one.
  `--bench-vulkan window` measures both. The window-sized ring-buffer cache is not done (plans/plug-in.md says why).

- Knowledge distillation (the teacher pattern): `FineTuningOptions.Teacher` with `DistillationTeacher.FromModel` (the
  teacher run on the fly, any weight format, on the student's device or another) or `DistillationTeacher.FromFile`
  (top-k logits per token written by `TeacherLogitsWriter`, read by `TeacherLogitsFile`; the stored top k
  renormalized); `FineTuningLossInput.TeacherDivergence(T)` and `FineTuningLosses.Distillation(temperature, alpha)`
  (α · T² · KL plus (1 − α) · the token cross-entropy, without storing the full logits); a vocabulary check that refuses
  a teacher with another tokenizer (`DistillationTeacher.CheckVocabulary`, `VocabularyFingerprint`); teacher-written
  data for sequence-level distillation (`TeacherData.Generate`, optionally with the teacher's reasoning); for
  classifiers, `Losses.Distillation` and `Distillation.WithTeacher` for the `Trainer`. `idrak distill` replaces its
  stub: on the fly, `--precompute FILE` then `--teacher FILE`, or `--generate`, with `--temperature`, `--alpha`,
  `--top-k`, `--teacher-weights`, `--teacher-device` and idrak tune's options.
- idrak: messages in Arabic with `--lang ar` (the config's `lang`, `IDRAK_LANG`): `idrak help` (titles, groups, every
  command's summary, the common options), usage errors, questions, `doctor`, `devices` and the chat's own lines; JSON,
  CSV and Markdown output are never translated. Because most terminals (Windows' console host, Windows Terminal, VS
  Code's, xterm, kitty, ...) show Arabic reversed and unjoined, the tool shapes it (presentation forms, lam-alef
  ligatures) and reorders each line by the Unicode Bidirectional Algorithm (complete UAX #9, passing Unicode 18.0's
  conformance suite; tables generated from the Unicode Character Database by tools/unicode), keeping English words,
  numbers, paths and table columns in place; terminals that do it themselves (VTE, Konsole, mlterm, mintty, macOS
  Terminal) and files get the text as stored. `--lang-render auto|visual|visual-right|logical` (`IDRAK_LANG_RENDER`)
  overrides; `idrak help arabic` explains.

- idrak, across the groups: every command's help has one layout (usage and aliases, "Arguments:", "Options:" with
  short forms, examples, limits and gaps, the common options, and the environment variables that affect it, generated
  from the one variable table instead of hand-written lines); `idrak help` lists the commands by group with their
  aliases. Model resolution looks in the cache first (no network; last use shown by `idrak list`), downloads into
  `--cache`, and honours `--offline` (a refused download names the missing file); downloads, model loading, training
  and benchmarks draw one kind of progress line (none with `--plain` or without a terminal); `--timeout` bounds
  downloads, server calls (`ping`, `api`, `server ...`) and long runs (`run`, `train`, `mcp serve` stop as at
  Ctrl+C); `--format csv|md` covers `show`, `version`, `plugins list`, `doctor`, `eval` and `rag eval`; `--seed`
  reaches `suggest`, `demo`, `rag ask`, `profile` and `onnx check`. Duplicate helpers merged (sizes and times, image
  decoding, confirmation, the base-model reader); a test group checks the help layout, `--cache`/`--offline`,
  `--timeout`, `--format` and progress.

- idrak: the run commands: `chat` (streamed, with `/help`, `/system`, `/reset`, `/save`, `/load`, `/file`, `/stats`,
  `/think`, `/tools`, `/set`, `/copy`, `/retry`, `/exit`; `--history`, `--file`, `--mcp`), `run` (argument, `--input` or piped
  prompt; `--json`; `--schema` checked afterwards), `batch` (resumable), `compare`, `complete`, `embed` (JSON or `.npy`),
  `tokenize`, `template`, `agent` and `tools list`/`test`.
- idrak developer commands: `new console|webapi|rag|plugin NAME` (embedded project templates; the plug-in starter
  registers a packed weight format and a network step, with tests on the public API only), `test` (the test runner of
  a source checkout), `onnx import/export/check` (check against a round trip, ONNX Runtime when Idrak.Onnx.Runtime is
  loaded, and reference outputs), `kernels dump ptx|spirv|hip`, `trace` (telemetry live or as JSON Lines), `demo
  xor|spirals|shapes|gpt` and `shell` (history and completion).

- `idrak serve`, `ui`, `server ps/stop/load/unload/keys`, `api`, `ping` and `mcp serve`: several models on one port
  over the chat API and the OpenAI-style API, loaded on first use and unloaded when idle, with API keys, CORS, a
  concurrency limit, metrics and a request log.
- Idrak.AspNetCore: the OpenAI-style API, `MapCompletionsApi` (`/models`, `/chat/completions` with streaming and tool
  calls, `/completions`, `/embeddings`) over the engine's text and chat models and any `IChatModel` or `IEmbedder`
  given in `CompletionsApiOptions`; `CompletionsTranslation` converts requests and messages.
- `idrak` model commands: `pull` (Hugging Face models and GGUF files, with progress, resume, `--dry-run`), `list`
  (`ls`), `rm`, `show`, `search`, `alias set/list/rm`, `memory`, `quantize`, `merge`, `inspect` (`i`), `verify`,
  `convert`, `diff` and `families`, over the library's download cache.
- `idrak suggest` (`sg`), `idrak explain` (`x`) and `idrak viz`: design a network for a table, an image folder, text,
  chat or preference rows from documented rules (network.json in the builder's JSON, train.json, prep.json, or a LoRA
  tune.json with `--base`), measure candidates with `--search N`, and show a network's shapes, parameters, FLOPs and
  memory, or draw it as text, Mermaid or SVG.
- idrak CLI, Train and data group: `idrak tune` (the commands and options of the former idrak-tune, plus `-b`/`--base`, `-w`/`--weights`,
  aliases and a tune.json given with `--config`) and `tune init` (writes a commented tune.json); `idrak train SPEC.json
  --data FILE|FOLDER` (a builder network on a CSV or a folder of class folders of PNG/PGM/PPM images; run folders with
  checkpoints and a JSON Lines log; a `.ikm` package), `resume`, `runs list/show/compare` (text loss plots), `predict`,
  `package` and `distill` (explains that it waits for the teacher pattern); `idrak data` (the former idrak-data's commands) with
  `package` and `distill` (see the distillation entry above); `idrak data` (idrak-data's commands) with
  `preview`, `validate --as chat|preference|table`, `stats -m MODEL`, `convert`, `dedupe`, `split`, `sample`, `mix`.
- The separate tools `idrak-tune` (package `Idrak.FineTuning.Cli`) and `idrak-data` (package `Idrak.Datasets.Cli`)
  are removed, folded into `idrak`: `idrak tune` and `idrak data` take the same commands and options (see `idrak help
  tune`, `idrak help data` and src/Idrak.Cli/README.md, which now holds their documentation). Differences from the old
  tools: idrak's exit codes (a usage error exits with 2, a failed data command with 1, where idrak-data used 2), and
  idrak's options for the model, device, cache and output. A tune.json keeps its format `idrak-tune/1` (it names the
  kind of file, like `idrak-train/1`); `idrak tune init` now writes it, and `idrak tune --config` accepts it (it
  rejected the key, so a tune.json from `idrak suggest --base` did not run) and refuses another file's format.
- `idrak` tool, Health group: `doctor` (`--android`, `--network`, `--explain`, `--fix`), `devices`, `version` with the
  drivers, `report` (`--tests`, `--bench`, `--readme`, `--zip`), `env` and `help env` (every environment variable the
  libraries read, from one table a test keeps complete), `init`, `cache info/clear`, `config get/set/unset/list` with
  profiles, `plugins list`, `formats`, `completion`, `update`, `login`/`logout`, `setup android` and `help topics`;
  with the shared `@file`, `--log`, `--output`, `--format`, `--color`, `--plain`, `--offline`, `--threads`, `--seed`
  and `--timeout` options, progress lines, questions with `--yes` and Ctrl+C handling. `Idrak.Diagnostics.
  DeviceListing` lists every device with what it reports (memory, compute units, lanes, kernel width, driver).

- Tool calls are parsed in the model's own format, not only as JSON between tags. `ChatOutputParser` takes the
  reasoning out and feeds the answer to an `IToolCallParser` (`Feed`, `Finish`, emitting text, reasoning and calls as
  before) that the template makes per reply (`ChatTemplate.CreateToolCallParser`, by `ToolCallFormatName`).
  `ToolCallFormats.Register(name, detect, create)`, `Unregister`, `Names`, `Create` and `Detect` add formats; a
  `JinjaChatTemplate` detects its format from its source and a rendered probe call (`ProbeToolCalls`), or is told
  with `CallFormatName`. Built in, each checked against the family's published template and round-tripped through it:
  `json` (the previous behaviour, plus Llama 3.1's `<|python_tag|>`, calls separated by `;` and built-in
  `name.call(...)`), `pythonic` (`[f(a=1, b="x"), g()]`), `qwen3-coder` (XML parameters, typed by the tool's schema),
  `mistral` (`[TOOL_CALLS]` with a JSON list or `name[ARGS]{...}`), `harmony` (GPT-OSS channels; analysis becomes
  reasoning) and `deepseek` (V3/R1 fenced JSON, V3.1 bare JSON). `ChatOutputParser` gains a constructor taking the
  request's tools, which `ChatGenerator` uses. Messages given to Jinja templates carry `thinking` next to
  `reasoning_content`, the name GPT-OSS's template reads.
- ONNX export registry: `OnnxExportOps.Register<T>` / `Unregister<T>` (modules by type, the nearest registered base
  type applies), `RegisterLambda` and `RegisterGraphOp`, with the built-in layers registered the same way; translators
  given to one exporter still take precedence. `OnnxGraph.Module(module, input)` translates a child module inside a
  translator. A package can now ship a layer with its ONNX translator.
- ONNX export of `GraphModule` (skip connections, branches, shape arithmetic): an imported graph exports again, and
  import, export and import give the same outputs (tested against ONNX Runtime on the ResNet blocks).
- ONNX graph import uses registered `OnnxImportOps` translators: builder steps become layer nodes holding the built
  layers, and `OnnxImportContext.AddGraphOp` adds an operation node. A translator that needs `CurrentShape` in a
  graph gets it by running the graph so far on a zero sample.
- `GraphOps.Register(name, GraphOp)`: graph nodes run registered operations; the element-wise and tensor ones are
  registered the same way, the structural ones that compute host integers stay in `GraphModule`. New built-ins
  (imported from ONNX and exported back): clip, pow, sqrt, neg, leaky_relu, elu, hard_sigmoid, hard_swish, max, min,
  composed from existing tensor operations. Erf and ConvTranspose wait for tensor operations.
- `LayerTypes.Register<T>(type, describe, create)`: the layer types of `GraphModule` JSON are a registry (the closed
  list before), so a custom layer inside a graph survives JSON and model packages; blocks built by the network
  builder are described by their builder JSON, so registered builder steps survive too. A graph with an unknown
  operation is rejected when it is built, with the registered names.
- Public ways to write into an existing tensor: `Tensor.CopyFrom` (values or a tensor), `Fill`, `Scale` and
  `AddScaled`, not recorded by autograd; refused on the result of a recorded operation, and a backward step whose
  operation read the values before such a write throws instead of computing from the new values (a write count per
  storage). `Tensor.Persistent` and `Tensor.PersistentZeros` are public (tensors no `TensorScope` disposes), and
  `Optimizer.ApplyDecoupledWeightDecay` is a protected helper next to `ApplyCoupledWeightDecay`, so an optimizer such
  as Lion can be written outside the library.
- `Autograd.Function(name, forward, backward)`: an operation with its own backward step written with public tensor
  operations, recorded as one operation (`DifferentiableFunction.Apply`).
- Element-wise operations with gradients: `Sqrt`, `Sin`, `Cos`, `Silu`, `Sign`, `Pow(exponent)`, `Clamp(min, max)`,
  `Maximum`/`Minimum` (with a tensor or a number) and `Tensor.Where(condition, a, b)`; CPU and Vulkan kernels (sin and
  cos reduce their argument first), the host fallback on CUDA for now.
- tests/Idrak.PluginTests: plug-ins written from an assembly without internal access (Lion, a custom operation as a
  network step and an ONNX import operator, a packed weight format, a KV cache format), run by the test runner as the
  "outside plug-in" group.

- RoPE scaling is pluggable: `RopeScalings.Register(type, method)` (with `Unregister`, `Names`, `Get`); a method
  returns the scaled frequencies, an attention factor on the rotary tables and optionally per-position frequencies.
  `RopeScaling` now holds the type and the raw JSON parameters (`RopeScaling.Linear`, `Llama3`, `Yarn`, `Dynamic`
  build them); its earlier constructor with llama3's fields is gone. Built in: linear and llama3 (unchanged results),
  YaRN (beta_fast/slow, original context, attention factor or mscale/mscale_all_dim, truncate) and dynamic NTK, as
  transformers computes them. config.json forwards any registered `rope_type`; GGUF files with YaRN keys use it
  instead of falling back to the original context. Decoder specs written before keep reading.
- Sliding-window attention: `DecoderSpec.SlidingWindow`, `SlidingWindowLayers` and `SlidingWindowRope`, read from
  config.json for Mistral, Qwen2/Qwen3 (use_sliding_window, max_window_layers), layer_types and Gemma 2/3; outputs
  beyond the window are now right instead of noted as approximate. Windowed layers attend through basic operations on
  every device once the window falls inside the keys (correct, slower than the attention kernels); they refuse packed
  sequences and rows of different lengths.
- Soft-capping and score scale: `DecoderSpec.AttentionSoftcap`, `LogitSoftcap` and `AttentionScale`
  (`CausalSelfAttention.ScoreSoftcap`/`ScoreScale`, `Linear.OutputSoftcap`).
- Gemma 2 and Gemma 3 (text) are registered model families.
- `PretrainedArchitecture.Build`: a family can build its own network from the config, the weights and the build
  options instead of a `DecoderSpec`. `PretrainedArchitectures.Unregister`.
- Fine-tuning options: `FineTuningOptions.Optimizer`, `Scheduler` and `Loss` replace the fixed AdamW, cosine schedule
  and token cross-entropy (null keeps them, with the same results and recorded steps). Built-ins in
  `FineTuningOptimizers`, `FineTuningSchedules` (cosine, linear, constant, warm-up/stable/decay) and `FineTuningLosses`.
  Recorded steps work with any optimizer that keeps its gradient buffers; others run ordinary steps.
- Preference training: DPO (with label smoothing), ORPO and SimPO over `PreferencePair`s, read from TRL's preference
  layouts (`RowKind.Preference`, `ChatRows.Preference`, `ChatTranscriptEncoder.EncodePreference`); DPO's reference is the
  model with its adapters disabled (`ModuleExtensions.DisableAdapters`).
- Adapters: `Linear.Adapter` is an `ILinearAdapter` (LoRA one implementation; `Linear.Lora` for the LoRA case) and
  `DoraAdapter` adds DoRA (`AddDora`, `FineTuningOptions.Dora`), saved and loaded in the PEFT format with `use_dora` and
  merged like LoRA. Code reading `linear.Adapter.A` reads `linear.Lora.A` now.
- PEFT adapters are checked: another `peft_type`, per-module ranks or alphas, trained biases, modules to save and tensors
  no layer takes are refused with an error instead of loading as plain LoRA; `LoadAdapter` applies `use_rslora`.
- idrak-tune: `--optimizer`, `--schedule`, `--warmup`, `--min-lr`, `--weight-decay`, `--loss dpo|orpo|simpo`,
  `--beta`, `--margin`, `--adapter-type lora|dora`.

- Vulkan: reduced-precision matrix products when the user opts in with `MixedPrecision` (`BFloat16`, or `Float8`,
  which takes the same path as on CUDA GPUs without FP8). Before, Vulkan ignored `MixedPrecision` and computed every
  product in float32. On a device with cooperative matrices, the float32 product and the int8, int4 and bfloat16 prompt
  products now gain a single-pass kernel in mixed mode: each operand rounded once (to nearest, ties to even), one
  matrix product per step instead of three, sums in float32. Operands are bfloat16 where the device reports
  `VK_KHR_shader_bfloat16` (shaderBFloat16Type, shaderBFloat16CooperativeMatrix) and a bfloat16 × bfloat16 → float32
  shape, exactly the rounding `MatMulPrecision.BFloat16` states; otherwise 16-bit floats after the per-row and
  per-column power-of-two scaling (11 significant bits, within bfloat16's error bound; a bfloat16 kernel the driver
  rejects falls back to it). The kernel is measured against the float32 choice for each shape under a tuning key of its
  own and used only where faster; float32 mode runs exactly what it ran before. Tests check each output against the
  product of the rounded operands and against bfloat16's error bound of float32 (the device's kernel, or emulated on
  lavapipe for both operand types), and that float32 mode never runs it; spirv-val checks every new kernel (the
  bfloat16 ones with a validator that knows SPV_KHR_bfloat16, named by `IDRAK_SPIRV_VAL`; newer validators require the
  Vulkan memory model with cooperative matrices, so the matrix kernels are validated with it declared, as they are
  otherwise unchanged). `--bench-vulkan matmul` prints the float32 and the mixed-mode product, the path each took and
  every kernel's time.
- Direct decoding steps (not recorded as a graph) keep the residual addition fused with the next block's RMS norm for
  that norm: `Sequential` freed it with the layer's intermediate results, so the norm ran again, one kernel more per
  layer and token on every device (74 dispatches per token instead of 82 for the medium decoder of `--bench-vulkan`).
- Fine-tuning's trace and the CLI's README say "graph" rather than "CUDA graph": the training step is recorded on
  Vulkan too.
- Vulkan: matrix units through cooperative matrices (`VK_KHR_cooperative_matrix`). Where a device reports the
  extension, shaderFloat16 and a float16 × float16 → float32 shape with subgroup scope, the float32 product and the
  int8, int4 and bfloat16 prompt products gain a cooperative-matrix kernel as one more measured candidate (the shape
  with the most work per operation among those reported; staging depth from the device's workgroup memory; a kernel
  the driver rejects is dropped). Precision stays float32's: operands are split into float16 high and low parts after
  an exact power-of-two scaling per row and column (high·high + high·low + low·high; bfloat16 and int8 weights are
  exact in one half), checked element by element against the CPU. Tuning choices are stored per cooperative-matrix
  shape; devices without the extension, and `IDRAK_VULKAN_MATRIX=0`, run exactly what they ran before. Without a GPU
  that reports it, the kernels are validated by spirv-val and checked against the CPU through an emulation of the
  matrix operations (lavapipe); `--bench-vulkan` shows the path a 1024³ product took.
- HIP backend, first slice (untested on real hardware): AMD GPUs through ROCm (Linux) or the HIP SDK (Windows) as
  `hip:0`, `hip:1`, … (`DeviceType.Hip`). The runtime (libamdhip64 / amdhip64_N.dll) and hipRTC are loaded at run time
  through function pointers, so a machine without them sees no HIP device, a reason in `--list-devices` and no
  exception. Memory, uploads, downloads, copies, fills and strided copies run on the device; element-wise arithmetic,
  seven unary operations, RMS norms and the int8 product run as HIP C++ kernels compiled by hipRTC for the device's own
  target, with a block size from its reported limits, and kept in `~/.cache/idrak/hip/kernels` per target, driver and
  library build; everything else takes the host fallback. Listed in plain test runs; the default device only with
  `IDRAK_HIP_DEFAULT=1`. Plan in plans/8-hip.md, setup in installation/hip.md.

- installation/: how Idrak was installed, built, tested and chatted with on Windows, Linux and WSL2, and an
  Android phone (Termux, Ubuntu in proot, a Turnip driver built for KGSL, the .NET heap limit).
- Vulkan: the kernels' width is measured when the device opens (a 256³ float32 product and one-row decoding work,
  int8 products, an RMS norm and a softmax, at each power-of-two width from the subgroup size up to the formula's,
  ranked by the geometric mean of each part's time against its fastest width), kept only when clearly faster (10%), and stored per device, driver and
  power state; IDRAK_VULKAN_WIDTH still overrides it and IDRAK_VULKAN_WIDTH_PROBE=0 turns it off. On an Adreno 730 the
  formula's width (1024) ran large products 30 times slower than 256, and a long dispatch reset the device.
- Vulkan width probe: times the operations as models call them (a kernel that does not fit a width is no longer
  timed as taking no time), adds an int8 product of 64 rows, and rebuilds kernels looked up by name when the width
  changes (they kept the width they were first built at, so dispatches sized for another width ran them).
- Vulkan: the embedding gradient's scatter rereads its table on every load (Volatile, Coherent): with repeated ids
  one driver kept the first load and lost the second add.
- `--bench-vulkan`: each decoder is timed again and without recorded graphs, so a slow first use or replay shows.
- Vulkan: prompts (tiled attention with or without the log-sum-exp; int8, int4 and bfloat16 products for many rows,
  each measured against the few-rows kernels and expanding the weights), training (layer-norm and batch-norm
  gradients and statistics, group reductions, column sums, products with a bias, fused AdamW with clipping, 8-bit Adam
  with the CPU's codes, the tiled attention gradient), convolution and pooling (im2col, col2im and max-pool gradients as
  gathers, bit for bit the CPU's) as generated kernels; fused decoding kernels (several products sharing an input,
  gate/up with the activation, the gated down projection, projection + residual + RMS norm, head norms with rotation
  and cache writes); graph capture and replay of decoding and training steps; storages larger than the device's
  binding range bound in windows (packed products, gathers, dequantization); required subgroup sizes measured where
  the device offers several; libvulkan.so on Android. Every choice comes from reported limits or is measured and stored
  per device. `CpuBackend` now implements `SumColumns` (its fallback called itself).
- GGUF pre-tokenizers are a registry (`GgufPreTokenizers.Register(name, pattern)`): the llama3, qwen2, tekken and gpt2
  families are registered with llama.cpp's patterns, unchanged; an unregistered name still uses Llama 3's rule and the
  note now says how to register it. `PackedWeight.FromValues(PackedFormat, ...)` makes the built-in weights from the same
  table the name registry starts from (no switch). Weight files encode and decode through one codec per `WeightFormat`,
  a call per tensor instead of a branch per value; the files are unchanged.
- CUDA decoding attention measures its split count over filled lengths 64, 128, ... up to the capacity (geometric mean
  of their times) instead of at a full cache alone. At a full cache 8 and 64 splits were within 2% on an RTX 3060
  Laptop; 64 was kept and ran 200 positions in 36.8 µs against 12.0 with 8. Cache format 5 re-measures saved choices.
  --bench-gemv warms the GPU until its speed settles before each timing.
- CUDA decoding attention reads at least a measured number of cached positions per block, so a short cache runs on
  fewer blocks than the split count measured for a full one (on an RTX 5070 Ti, 200 positions took 9.0 µs with the 24
  splits a full 4096-position cache wants, 7.6 µs with 16). The split count stays fixed per shape (recorded graphs stay
  valid); the least chunk is measured after it over filled lengths 64, 128, ... up to the capacity, by the geometric
  mean of their times, against plain chunks. On the RTX 5070 Ti the best least chunk was 2-4% faster and did not pass
  the margin twice, so plain chunks stay there; the choice is per card.
- CUDA tuning times a candidate that beats the formula's choice a second time, in new pairs, and keeps it only when
  it wins again. Short prompt products timed warm (about 13 µs on an RTX 5070 Ti) gave medians 10% apart from one
  process to the next: two runs kept 2 and 3 splits for the same shape, and an earlier one kept 1 split (22 µs
  against 12.7 µs for 6) in the tuning cache. With timings off by up to 20%, a slower candidate was kept in 57 of 400
  simulated measurements before and 6 now. The cache format is 4, so choices saved before are measured again.
- CUDA tuning times each candidate in pairs with the formula's choice (back to back, the order alternating by round)
  and compares the medians of the pair ratios. Before, every candidate ran once per round against one timing of the
  reference, four of seven rounds with the reference first: a clock still drifting during the measurement moved each
  median by the candidate's distance from the reference in the round, which favoured the most splits when the clock
  rose, so a candidate a few percent slower could be kept (the 151936-column head on an RTX 5050 Laptop ran 7-9%
  slower than 1 split; 2.5% on an RTX 3060 Laptop). Decoding-sized products (GEMV splits, fused add-and-normalize,
  q/k/v and gate/up in one launch, the few-row kernel against the packed product, the gated activation) are timed
  with a cold L2 cache: each timed run follows a read of twice the L2 size the device reports, as decoding meets the
  weights. `IDRAK_TUNE_LOG=1` prints each measurement (median ratios and the choice). The cache format is now 3, so
  earlier choices are measured again. `--bench-gemv` rotates copies of each weight adding up to twice the L2 size, so
  its columns are timed under the same conditions.

- Measured choices are kept per power source on every backend (CUDA, Vulkan, CPU): a laptop on battery clocks down,
  so what wins there can lose on mains power. The source is what the operating system reports (Windows
  GetSystemPowerStatus, Linux /sys/class/power_supply); a machine without a battery counts as mains;
  `IDRAK_POWER_SOURCE=ac|battery` overrides it.

- Vulkan measures where storages live instead of assuming it. Mapping system memory on integrated GPUs (an entry
  below) made a Radeon Vega iGPU upload twice as fast but compute much slower (element-wise 39 → 27 GB/s, int8 decoder
  343 → 222 tokens/s): it reads cached (snooped) system memory slower than its device-local carve-out. The candidates
  the device reports (staging into device-local memory; device-local memory the host maps when about all of it is
  mappable; on shared-memory devices, host-cached or write-combined system memory on the largest host-visible heap,
  one per memory type) are each timed at start: an element-wise kernel over 4 MiB storages, uploads and downloads,
  medians of five rounds after a warm-up. The fastest device reads win, unless another candidate reads within 3% and
  uploads faster. The choice and the numbers are cached with the other runtime choices (per device, driver and power
  source); pages and the heap shown follow the chosen type. `IDRAK_VULKAN_STORAGE=staging|mapped-device|mapped-cached|mapped-uncached`
  overrides it, `IDRAK_VULKAN_STAGING=1` still forces staging, and the `--bench-vulkan` header shows the choice and
  every candidate's numbers. Discrete GPUs have staging, plus mapped device memory with resizable BAR.
- Vulkan devices keep their number across runs: `vulkan:N` follows what each device reports (discrete GPUs, then
  integrated, virtual, CPU and other; each kind by deviceUUID bytes, then driverUUID; a device without a UUID after
  those, by PCI address and name), not the loader's order, which changed between two processes on a laptop with an
  RTX 3060 and a Radeon iGPU (so `IDRAK_DEVICES=vulkan:1` named either GPU). A device without a UUID gets a tuning-cache
  key of its own (PCI address and name) instead of sharing the all-zero one.
- Vulkan on integrated GPUs and CPU drivers (reported device type) maps storages from host-visible memory on the largest
  host-visible heap (device-local and host-cached first, then host-cached, then coherent): it is the same system memory
  as their "device-local" one, which on APUs is a small carve-out, so they no longer copy through staging (the Radeon
  iGPU did). Pages are sized from that heap. Discrete GPUs keep the rule (mapped only with resizable BAR);
  `IDRAK_VULKAN_STAGING=1` still copies through staging everywhere. The benchmark header shows the storage heap's size.

- CUDA autotuning is steadier and remembered: candidates are timed over seven rounds in alternating order and compared by
  their median (one lucky timing picked 8 k splits at 7.7 µs over 16 at 6.6 µs on "o + add + norm"), and measured
  choices are kept per GPU, driver and library build in `IDRAK_CACHE` (default `~/.cache/idrak`) under `tuning/cuda/`,
  so later starts read them instead of measuring (`IDRAK_TUNING_CACHE=0` turns that off, `=<folder>` moves it; a file
  of another GPU, driver, build or format version is ignored and rewritten). Decoding-attention splits are now measured
  per shape at a full cache instead of set by "~5 blocks per SM" (the formula stays the default until measured and with
  `IDRAK_AUTOTUNE=0`). Every CUDA constant and heuristic is classified in plans/README.md ("Status of each rule").
- CUDA kernel shapes come from what the GPU reports: its limits are read once (threads per block and per SM, warp size,
  shared memory per block and with opt-in, registers, L2, grid limits, ...), the block sizes the kernels take at any
  value are derived from them and generated into the PTX per GPU, and the geometry a kernel is written for is checked
  against them (a tensor-core module that needs more shared memory than a block may use is skipped with the reason, as a
  GPU the main kernels cannot run on is). Every CUDA GPU so far derives the sizes the kernels always had, so the PTX is
  unchanged byte for byte (plans/README.md, "Kernel shapes", lists each shape and its value on the tested cards).
- CPU: no choice is tuned to one machine any more (same results, bit for bit). Tiling, blocking and threading come from
  what the machine reports (`CpuTuning`, `CpuInfo`: threads, vector width and registers, instruction sets, L1/L2/L3 and
  line sizes from /sys on Linux, `GetLogicalProcessorInformationEx` on Windows, sysctl on macOS, with fallbacks) or are
  measured: the element-wise and product parallel cut-overs are timed at first use (medians of 7 rounds, parallel at least
  10% faster, the median of three passes after an untimed warm-up pass, element-wise over a rotating window larger than
  the L2, the product on one row against a square B, about two seconds once per machine) and kept per machine, runtime
  and thread count in `IDRAK_CACHE`/cpu/tuning.tsv (default ~/.cache/idrak/cpu; `IDRAK_TUNING_CACHE=0` keeps nothing;
  overrides `IDRAK_CPU_TUNING_FILE`, `IDRAK_CPU_PARALLEL_ELEMENTS` / `IDRAK_CPU_PARALLEL_FLOPS`; `IDRAK_AUTOTUNE=0`: the
  cache-size formulas). The product's A tile is a quarter of the L2, at most L1 / 2 KiB row blocks, and the tile count
  a multiple of the threads where splitting the columns up to twice as finely allows it (1024³ ~9% faster on 4 threads);
  column blocks are eight vectors; transpose tiles fit half the L1; the tiled kernel is chosen by reported instruction
  sets (AVX-512, AVX2+FMA, and a new NEON 8×8 kernel for ARM64). Few-row bfloat16 products keep up to 8 rows' sums in
  registers (4 with 16 vector registers; 512-bit panels where AVX-512 is accelerated): 4 and 8 rows about 2× faster.
  `Backend.Capabilities` of the CPU no longer borrow CUDA's constants: the few-row split (8) and the attention head
  sizes (256 decoding, 128 tiled) are the CPU's own numerical contract, unchanged, like the other values that decide a
  summation order (reduction chunks, transposed-B rows, the column-split rule with beta ≠ 0): the CPU is the reference.
  `HostParallel` honours `ComputeResources.MaxCpuThreads`.
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
- Packed-weight formats of one's own (NF4, FP8, …): derive from `PackedWeight` (now with a protected constructor) and
  implement `Name`, `Rows`, `Columns`, `Bytes`, `Dequantize`, `Buffers`, `MoveTo` and `Dispose`; `MatMul` is public and
  virtual, by default expanding the weights for each product (not recorded into CUDA graphs; override it with a
  device-only product to be). `PackedWeight.Format` is now `PackedFormat?`, null for such formats, which never reach the
  built-in formats' kernels (fused projections, gate/up pairs, LoRA products): each layer runs its own product. Formats
  register by name (`PackedWeight.Register`, `FormatNames`, `FromValues(string, …)`; "int8", "int4", "bfloat16" built in)
  and are chosen with `DecoderBuildOptions.PackedFormatName` / `PretrainedOptions.PackedFormatName`. `Save` stores their
  buffers exactly, `ToFloat32` expands them, `To` moves them. Nothing changes for the built-in formats.
- Language models take plug-ins through public registries, with the built-ins registered the same way (same outputs,
  tokenizer fast paths unchanged): **checkpoint formats** (`ICheckpointFormat`, `CheckpointFormats`; safetensors and
  GGUF; `ITensorStore` is public), **GGUF tensor types** (`GgufType`, `GgufTypes`, `GgufType.Blockwise`) and **GGUF
  architectures** (`GgufArchitecture`, `GgufArchitectures`), **model sources** (`IModelSource`, `ModelSources`:
  folder, `ollama:`, .gguf, Hugging Face id; `ModelSource.Resolve` asks them) and **tokenizer components**
  (`TokenizerComponents` with `ITokenizerNormalizer`, `IPreTokenizer`, `ITokenizerDecoder`, by tokenizer.json
  "type"). A model's notes now list where its weights come from before the others.
- Datasets are pluggable: file formats (`IDataFileFormat`, `DataFileFormats.Register`, chosen by extension, by
  `ReadOptions.FileFormat` or by `format=` in recipes), recipe sources (`IDatasetSource`, `DatasetSources.Register`, for
  prefixes beyond `hf:`, `github:`, `kaggle:`, `zenodo:` and URLs, with their own options) and Parquet compression codecs
  (`IParquetCodec`, `ParquetCodecs.Register`, by Parquet codec id, e.g. to add Zstandard). The built-ins are registered
  the same way and read as before; `DataFormat` and `DataFiles.FormatOf` still name them.
- Vulkan devices (`vulkan:0`, `DeviceType.Vulkan`): Intel and AMD GPUs, and any Vulkan 1.1 device, through the
  graphics driver's Vulkan loader (no SDK, nothing to install). Memory, copies (mapped memory on integrated GPUs,
  a staging buffer elsewhere) and in-order SPIR-V dispatches; every operation runs through the host fallback until
  its kernel lands, so every model already runs. Listed (so tests run on it), but picked by `Device.Default` only with
  `IDRAK_VULKAN_DEFAULT=1` until its kernels are tuned (after CUDA, discrete before integrated); software drivers (lavapipe) and NVIDIA GPUs that CUDA drives are reached by name only.
  `IDRAK_DISABLE_VULKAN=1` turns it off.

- KV cache formats of one's own (FP8, 4-bit, …): derive from the public `KeyValueLayout` (`RowWidth`, `Write`,
  `Expand`) and attend through its default `Attend`, expanded to float32 and masked products, on any backend.
  `KeyValueLayouts.Register` / `Get` / `Names` choose them by name ("float32", "int8", "bfloat16" built in); new
  `DecodingContext(..., KeyValueLayout)`, `KeyValueCache.Layout`, `TextGenerator.CacheLayout`,
  `PretrainedModel.CreateGenerator` / `CreateChat` overloads, `idrak-tune --kv NAME`, and `KeyValueFormat.Custom`
  (what caches of such formats report). The built-in formats keep their kernels; a bfloat16 cache now also works in
  `MultiHeadAttention` (expanded to float32) instead of throwing.
- Vulkan groundwork (internal, not used by any device yet): a SPIR-V 1.3 generator in C# (`SpirvModule`, and
  `KernelBuilder`, which writes kernels as C# code with structured ifs, loops, workgroup memory and barriers) and the
  first 74 Vulkan kernels (element-wise and their gradients, reductions, softmax, norms, rotary positions, gathers, the
  key/value caches, float32 tiled and packed int8 / int4 / bfloat16 products, decoding attention over float, bfloat16
  and int8 caches). Every kernel passes `spirv-val --target-env vulkan1.1` in the tests when it is installed;
  `--dump-spirv <folder>` writes them with their bindings and push constants.
- Vulkan devices run 58 operations as these kernels instead of the host fallback: element-wise operations and their
  gradients, optimizer steps, dropout, sums, softmax (masked, log, backward, cross-entropy), RMS and layer norms,
  arg max, strided copies, permutations, axis sums, gathers and scatters, products (float32, small and tiled; int8,
  int4 and bfloat16 weights), rotary positions, the key/value caches (float, bfloat16, int8) and decoding attention
  over them. Short rows (up to 64 columns) take one invocation per row, head sizes up to 64 a 64-wide attention, and
  small products a kernel without workgroup memory. Cases a kernel does not cover (storages over the device's binding
  limit, head sizes over 256, permutations of more than six dimensions) still take the host fallback;
  `IDRAK_VULKAN_KERNELS=0` sends everything there. Embedding indices out of range are clamped, as on CUDA.
- Vulkan dispatches cost less: descriptors are pushed into the command buffer (`VK_KHR_push_descriptor`, on Intel,
  AMD and NVIDIA drivers; pooled descriptor sets without it and on CPU drivers, where pushing runs slower;
  `IDRAK_VULKAN_PUSH_DESCRIPTORS=1`/`0` decides instead), a barrier comes only before a dispatch that reads or writes a
  storage written since the last one or writes one read since (kernels declare what they write, and their read-only
  storages are `NonWritable`), the bound pipeline is kept, and recording a dispatch allocates nothing. Recording a
  dispatch on the host: 9.3 → 0.7 µs (lavapipe, pushed descriptors).
- Vulkan decoding steps take no host fallback, so the host no longer waits for the device on every token: kernels for
  the sampler (temperature, top-k, top-p, min-p, the counter-based random stream and the 13 statistics per token,
  with the CPU's tokens for the same seed; one invocation per row for small vocabularies, a workgroup per row, and for
  top-k over large vocabularies a first pass keeping each 2,048-token slice's candidates), repetition penalties, the
  token history, and RMS-normalized queries and keys with rotary positions (one dispatch for both). Measured on a
  small decoder: 3 host fallbacks per token before (7 with query/key norms, 19 for an 8-layer one), 0 now, for
  float32, int8, int4 and bfloat16 weights over float32, int8 and bfloat16 caches, and multi-head attention models.
- `--bench-vulkan [sections]` in the test runner times every Vulkan device (or those `IDRAK_DEVICES` names): dispatch
  overhead, copies, element-wise bandwidth, float32 products, decoding-sized int8 / int4 / bfloat16 products, decoding
  attention over 200 to 4,000 positions, sampling, and decoders' tokens per second with their kernels per token.
- Vulkan storages share device memory: each keeps its own buffer, carved from 64 MiB pages (one memory allocation each,
  mapped once on integrated GPUs); storages over a quarter page keep an allocation of their own. Drivers that cap memory
  allocations (4,096 on AMD's and others' Windows drivers) no longer run out of them with many tensors (LSTM, GRU and
  transformer gradients failed on an AMD Radeon iGPU). Empty pages go back to the driver with the cached memory.
  `IDRAK_VULKAN_MAX_ALLOCATIONS=<n>` lowers the cap the runtime assumes, to reproduce such drivers anywhere.
- Vulkan sampling with top-k over large vocabularies (the first stage keeping each slice's candidates) gave other tokens
  than the CPU on an NVIDIA GPU for rows with ties or long runs of equal scores; the stage now finds each slice's
  candidates with workgroup reductions (as the sampler's own top-k) instead of a sort and one invocation's walk, and one
  workgroup per (row, slice) along one dispatch dimension. New test: the sampler run 1,000 times on a GPU gives the same
  tokens and statistics bit for bit.
- The Vulkan runtime decides only from what the device reports or what is measured on it (no vendor ids, no card
  names, no fixed sizes): a Vulkan device another provider already drives is recognized by its UUID (Vulkan's
  deviceUUID against CUDA's `cuDeviceGetUuid`, read without a CUDA context) and listed by name only, whatever its
  vendor ("driven by CUDA as cuda:0"); CPU-type devices stay by name only. Storages are mapped where a device-local,
  host-visible memory type lives on a heap at least half the largest device-local heap (shared memory, CPU drivers,
  discrete GPUs whose whole memory the host maps), with reads through a host-cached staging buffer when that memory is
  not host-cached; a small host-visible window still means staging copies (`IDRAK_VULKAN_STAGING=1` everywhere). The
  staging buffer (1/512 of its heap) and pages (1/128 of the heap, raised until half the driver's allocation cap covers
  it) scale with the reported heap and `maxMemoryAllocationSize` (`IDRAK_VULKAN_STAGING_BYTES`,
  `IDRAK_VULKAN_PAGE_BYTES` override). Pushed descriptors or sets, commands per batch and batches in flight are measured
  when a device starts (medians of a few rounds of empty dispatches) and kept per device and driver in
  `~/.cache/idrak/vulkan/tuning.tsv` (`IDRAK_CACHE`; `IDRAK_VULKAN_TUNING_CACHE` names another file or `0` none;
  `IDRAK_VULKAN_PUSH_DESCRIPTORS`, `IDRAK_VULKAN_BATCH_COMMANDS`, `IDRAK_VULKAN_IN_FLIGHT` override); descriptor pools
  hold a batch's sets. The device's subgroup sizes and operations, workgroup, shared-memory and allocation limits are
  read once (`VulkanDeviceFacts`, internal) and shown in the `--bench-vulkan` header. A test keeps vendor ids and card
  names out of the Vulkan code.
- Vulkan kernels are shaped by the device they run on, never by its vendor or name: the workgroup width comes from the
  reported limits (8 subgroups, within the invocations, workgroup size and workgroup memory the device takes; kernels
  are generated per width, from 64 to 1024), reductions use subgroup arithmetic where the device reports it for
  compute shaders (two barriers instead of one per halving), and the choices no limit decides are measured on the device the first time a shape
  needs them (the median of five rounds; the formula's choice kept unless 3% slower) and stored per device and driver
  beside the runtime's choices (`~/.cache/idrak/vulkan/tuning.tsv`; `IDRAK_AUTOTUNE=0` uses the formulas,
  `IDRAK_VULKAN_WIDTH` forces a width, `IDRAK_VULKAN_SUBGROUPS=0` plain reductions). Decoding products and
  attention are split across workgroups: packed int8 / int4 / bfloat16 products with few rows read each weight word
  once per workgroup of 32 or 64 words × slices of k, 1 to 8 rows of x at a time, and split k over workgroups (a second
  pass adds the splits in order); decoding attention scores a workgroup's width of positions at once with an online
  softmax, spreads small heads' values over parts of the workgroup, and splits the cached positions over workgroups
  (a second pass merges them, flash-decoding style); float32 products have a register-blocked kernel (4 × 4 outputs
  per invocation) beside the tiled and small ones. Measured choices: widths, words per row and splits of the packed
  products, splits of attention, the float32 product kernel, row kernels' narrow or wide form and their reductions.
  Same results as before within rounding (sums in a different order), bit-identical from run to run. On lavapipe
  (a CPU driver: indicative only; width 64 from its subgroups of 8): int8 product 1 × 1024 → 3072 5.1 → 1.3 ms, int4
  1 × 3072 → 1024 6.7 → 1.2 ms, int4 1 × 1024 → 151,936 213 → 39 ms, float32 1024³ 1.6 → 6.7 GFLOP/s, attention over
  4,000 positions 28.8 / 24.9 / 29.6 → 27.8 / 18.5 / 16.7 ms (float32 / int8 / bfloat16 caches; 200 positions slower
  there, 4.1 → 9.1 ms, from the splits sized for a full cache).

- `idrak` Measure and Retrieval commands: `bench` (alias `b`; a model's prompt and generation speed, GFLOP/s and
  memory, or the kernel benchmarks; `--devices`, `--matrix`, `--save`, `--compare`), `eval`, `perplexity`, `profile`,
  `check`, `tuning show`, and `rag index`, `rag search`, `rag ask`, `rag eval`.

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
