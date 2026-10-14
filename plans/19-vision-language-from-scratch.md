# Plan 19: language and vision-language models from scratch (made, not loaded)

**Status:** planned 2026-10-14, detailed the same day against the code at `abstraction` e0b8d701. Track A built
2026-10-14 (A0 to A4, CPU-checked; A5 is the owner's run, commands in "Track A, as built"); track B in progress. Decided by the owner 2026-10-14: the family is **IdrakVision**; the BPE trainer is built (A2);
image tokens attend in bidirectional blocks by default; distillation (B5) is built after B0 to B4, before the proof run. The
owner's request (2026-10-14): "new vision language", then "complete both text and vision from scratch", the text
model first because the vision model needs it.

**Goal.** Make models that did not exist before, from one spec and random weights, pretrain them on the owner's data,
and save them as checkpoints that load through the same contracts every loaded model uses, so `idrak run`, `chat`,
`serve`, `tune` and `distill` work on them unchanged.

- **Track A, a language model from scratch:** a `DecoderSpec` built with random weights, a tokenizer trained on the
  owner's text, a pretraining loop that trains every parameter on token windows, checkpoints with resume, and the
  command line (`idrak create`, `idrak pretrain`, `idrak data tokenizer`). Proven alone, on the CPU, before track B.
- **Track B, a vision-language model on top of it:** a vision tower and a projector with random weights, made by a
  family plug-in through a new `Create` contract beside `Read`, the same loop extended with images and stages (align
  the vision side, then train everything), and the family saved and reloaded through `VisionFamilies`. Arabic
  documents are the first target (plan 18's synthetic pages are the data); the machinery serves any language and any
  image-text task.

**Rules that hold for every phase** (`CLAUDE.md`, `docs/optimization.md`):

- **Card-agnostic.** No phase assumes a device, a vendor or a memory size. Batches, micro-batches and recomputation are
  measured on the running device (the tuner's out-of-memory ladder, `HostOptimizer`, `AdamW8Bit`); no size, card or GB
  budget appears in code, comments, tests or docs, this plan included. The tiny sizes below are fixture sizes.
- **Abstraction (decision 10).** Everything new is a contract with a `SlotTable` registry: a contract one library
  package uses lives in that package's `.Abstractions` namespace (`Idrak.Models.Abstractions`, `Idrak.Nlp.Abstractions`);
  one that Abstraction itself or two packages use lives in `Idrak.Abstraction`. The inventory test
  (`plans/10-abstraction-inventory.md`) and the public API guard (`api/*.txt`) are regenerated when they change (the
  commands are in "Conventions" below). The library registers no family: the created vision family is a plug-in in
  `samples/`; an unregistered name fails with the registry's "is not registered" message.
- **Speed and memory first.** Bytes parsed, not strings (rule 74: the tokenizer trainer works on UTF-8 bytes in blocks);
  one pass (73); memory walked in order (72); work done once (75: frozen stages cached, as `TuningVision` caches them);
  device memory that does not grow across steps (77); the library's telemetry on every step, free when nobody listens
  (78); independent work spread and measured (79); cold and warm measured apart (76); results proven unchanged (71).
- **Nothing fixed.** Every stage, ratio, schedule, threshold and path is an option (a spec field or a run option,
  carried to `idrak` and to `pretrain.json`), with an environment variable where the library already reads one.
- **Tests.** Targeted CPU runs only (`IDRAK_DEVICES=cpu IDRAK_FILTER=...`); the owner runs CUDA and Vulkan from the
  command list at the end of each phase. Never the full suite.
- **The fix workflow.** Every phase: fix all errors first, update the tests the fixes touch, build once (only the
  projects that changed), run the phase's filter once with `--no-build`, commit and push on `abstraction`, hand the
  owner the GPU command list. At most two agents at a time.

## What Idrak has, and what is missing

Every row names the code the phases build on (file: line at e0b8d701).

| Part | Today |
|---|---|
| A decoder from a spec with random weights | supported: `DecoderSpec` (`src/Idrak.Abstraction/Generation/DecoderSpec.cs:56`), `DecoderBuilder.Build(this DecoderSpec, IWeightSource? = null, DecoderBuildOptions? = null)` (`src/Idrak/Layers/DecoderBuilder.cs:22`): with no weights a seeded init (`DecoderBuildOptions.Seed`, `InitStd`, `src/Idrak/Models/Abstractions/PretrainedArchitectures.cs:109-168`), every parameter `requiresGrad: true`; the GPT sample (`samples/Idrak.Samples.GptTraining/Program.cs:168`) trains one on Idrak's CUDA kernels |
| A `PretrainedModel` around a built network | **missing**: the only constructor is private (`src/Idrak/Models/PretrainedModel.cs:105`), the only factory `Load(folder, options)` (:166); `FineTuner`, `ConversationTuning`, `idrak run` and `serve` all take a `PretrainedModel` with a `Folder` |
| Writing a model's folder | partial: `SaveHuggingFace(folder, type)` (:623) merges LoRA, writes `model.safetensors` from `Linear`, `Embedding` and `RMSNorm` only (throws on `LayerNorm`, skips `PositionEmbedding` and vision tensors) and copies config.json, tokenizer and template files from the **original folder**; a created model has none |
| A config.json from a spec | **missing**: `PretrainedArchitecture.Spec(config, notes)` maps config to spec (`PretrainedArchitectures.cs:16`); nothing maps a spec back |
| Training every decoder parameter | **missing**: `FineTuner.TrainCore` adds adapters whenever none exist and `AddAdapters` freezes the base (`src/Idrak.Nlp/FineTuning.cs:1043-1046`, `src/Idrak/Layers/ModuleExtensions.cs:174`); the trainable list is `network.TrainableParameters()` + `vision.Parameters` (:1048). The step itself (`StepRunner` :1179, `NetworkLoss` :2227, `BatchLoss` :2147, `StepImages` :2165, the out-of-memory ladder :1228-1294, graphs :2337) is general and reusable |
| Resume | **missing**: no optimizer state save or load (`src/Idrak.Abstraction/Training/Optimizer.cs`), no step offset; `SaveEvery` writes adapter folders only (:1126); the GPT sample reloads weights and reseeds, not the optimizer |
| Token windows for pretraining | supported: `TokenFileSource(path, length, stride?, type)` (`src/Idrak/Datasets/FileSources.cs:44`), memory-mapped, uint16/int32/uint32 or `.npy` |
| A tokenizer trained on the owner's text | **missing**: `BpeTokenizer` (`src/Idrak/Models/BpeTokenizer.cs:19`) reads tokenizer.json (BPE model, ByteLevel / Metaspace / Split / Digits / Whitespace pre-tokenizers, added tokens; no `post_processor`) and has no `Save`; `make_tiny.py` trains merges in Python (`train_merges`, line 195) |
| Chat template for a created model | partial: `JinjaChatTemplate.Load` reads `chat_template.jinja`, `chat_template.json` or `tokenizer_config.json` (`src/Idrak.Nlp/ChatTemplates.cs:309`); `ChatMLTemplate` is text-only (`Chat.cs:140`) and `ChatTranscriptEncoder` needs a `JinjaChatTemplate` |
| A vision tower and projector from random weights | **missing as a contract**: `IVisionFamily` has `Name` and `Read(VisionCheckpoint)` only (`src/Idrak/Models/Abstractions/VisionFamilies.cs:23-34`); the Gemma 3 and LLaVA encoders exist only through `FromTensors` (`samples/Gemma3Vision/Idrak.Gemma3Vision/Siglip.cs:87`, `tests/Idrak.PluginTests/Llava.cs:479`); random weights appear only in a test's fake store (`tests/Idrak.Tests/BenchVision.cs:136`) |
| Training the tower and projector in full, with gradients | supported: `TuningVision.Create(model, images, parts, cache)` (`src/Idrak.Nlp/TuningVision.cs:86`), `IVisionTuningPart.Parameters/Tower/Features/Export/Import` (`src/Idrak.Abstraction/Generation/VisionParts.cs:122-182`), cached frozen stages (`FeatureCacheKey.PixelsStage/TowerStage/FeaturesStage`), checkpointed tower blocks, images in packed rows (`ImagePrefill.cs:137`, `:173`) |
| Image-text data | supported: `TuningDataFormats` "messages" and "sharegpt" (`src/Idrak.Nlp/Abstractions/TuningData.cs:136`), images beside the file or in a zip, plan 18's `SyntheticPages.Write` (`src/Idrak.Nlp/SyntheticPages.cs:409`) writing `<stem>.jsonl` + `<stem>/` or `<stem>.zip` + `<stem>.manifest.jsonl`; `RandomText.Write` (`RandomText.cs:112`) |
| Held-out scores | supported: `TuningMetrics` "cer", "wer" (`src/Idrak.Nlp/Abstractions/TuningMetrics.cs:85`), `TuningAnswerScorer.Score(model, vision)` (`TuningAnswers.cs:143`, one request at a time), `FineTuner.Evaluate(model, sequences, batchTokens, chunkRows, trace, vision)` (:1707) |
| Distillation from a vision-language teacher | partial: the student trains with images (`NetworkLoss`), but `ModelTeacher.Hidden` runs the teacher on token ids alone (`src/Idrak.Nlp/Distillation.cs:148-238`) and `idrak distill` passes no image option (`DistillCommand.cs:28-34`) |
| Telemetry from training | partial: `Trainer.Fit` publishes `TrainingStarted/BatchCompleted/EpochCompleted/TrainingCompleted` (`src/Idrak/Training/Trainer.cs:166-318`); `FineTuner` publishes nothing (progress and trace only) |
| More than one device on one run | not started (plan 6); one device per run |

## Where each part lives (decisions 8 and 10 of plan 10)

| Part | Package and namespace | Why |
|---|---|---|
| `ITokenizerTrainer`, `TokenizerTrainers` (registry; "bpe" the library default), `TokenizerTrainingOptions` | `Idrak.Models.Abstractions` (core) | core owns the tokenizers; one library user |
| The byte-level BPE trainer, `BpeTokenizer.ToJson` / `Save` | `Idrak.Models` (core) | beside `BpeTokenizer` |
| `PretrainedArchitecture.Config` (spec → config.json), `PretrainedModel.Create(...)`, `SaveHuggingFace` completed (LayerNorm, positions, vision parts, files written from the spec) | `Idrak.Models` / `Idrak.Models.Abstractions` (core) | model creation and saving are core's |
| `Optimizer.SaveState` / `LoadState` | `Idrak.Abstraction.Training` (the base), implementations in `Idrak.Abstraction` (Adam, AdamW, SGD) and `Idrak.Optimizers` (AdamW8Bit, HostOptimizer, GroupedOptimizer) | the optimizers' own |
| `VisionLanguageSpec`, `IVisionFamily.Create` | `Idrak.Abstraction.Generation` (beside `DecoderSpec`, `PretrainedVision`) and `Idrak.Models.Abstractions` (beside `IVisionFamily`) | the spec is read by core and Nlp; the contract is core's |
| `Pretrainer`, `PretrainingOptions`, `PretrainingStage`, `PretrainingSource` kinds and their registry `PretrainingSources` ("tokens", "records"), `PretrainingProgress`, `PretrainingCheckpoint` | `Idrak.Nlp` (loop and sources) and `Idrak.Nlp.Abstractions` (the source contract and registry) | training with images is Nlp's; reuses `FineTuner`'s step |
| Images in `ModelTeacher` and `idrak distill` (phase B5) | `Idrak.Nlp`, `Idrak.Cli` | distillation is Nlp's |
| `idrak create`, `idrak pretrain` (+ `resume`, `runs show`), `idrak data tokenizer` | `Idrak.Cli` (`Commands/Train`, `Commands/Data`) | the surface |
| The created vision family (`Create` and `Read`, tower, projector, prompt format, attention rule, preprocessing) | `samples/IdrakVision/Idrak.IdrakVision` (decided: IdrakVision) | a family is a plug-in, never the library |
| References and fixtures | `tools/vlm/make_tiny_pretrain.py`, `tools/vlm/make_tiny_scratch_vlm.py`; `tests/Idrak.Tests/data/pretrain/`, `tests/Idrak.Tests/data/vlm-scratch/` | as plans 11 and 12 did |

## Conventions every phase follows (read before coding)

- **A registry** copies `RandomTextSources` (`src/Idrak.Nlp/Abstractions/RandomTextData.cs:31`): a `SlotTable<string, T>`
  named after the class, `comparer: StringComparer.OrdinalIgnoreCase`, library defaults through `RegisterDefault` (inside
  `Overrides.AsLibraryDefaults`, or `LibraryDefaults.Ensure(typeof(X))` when the defaults live in another assembly),
  `[MethodImpl(NoInlining)] Register(T)` with `Assembly.GetCallingAssembly()`, `Unregister`, `Names`, `Find`, `Get`,
  `Default`, `Origin`, `Describe`. `Get`'s message: `"{Kind} '{name}' is not registered (registered: {list}); add it
  with {Registry}.Register."` (`NotSupportedException`; the CLI recognises " is not registered" and appends "Load its
  plug-in with -P (--plugin).", `ImageModelSupport.cs:120-127`). Guard it (`SlotGuard`, Throw / FallBack / Shadow) when
  two implementations can answer the same call; mark it `unguarded:` with the reason when they cannot be mixed.
- **A contract suite** for each new registry: `src/Idrak.Abstraction.Testing/Contracts/XSuite.cs` (`ContractSuite<T>`:
  `Name`, `FixedCases`, `RandomCase`, `Run`) plus a `Conformance.CheckX(...)` wrapper, called from the tests.
- **Tests** are entries of a `(string Name, Action<Device> Run)[] XGroup` in a `Tests` partial file, added to
  `Tests.All` (`tests/Idrak.Tests/Program.cs:234`); names start with the group prefix; `TestData("pretrain/...")`
  finds fixtures; CLI tests run in-process through `CommandLine.Run(args, out, err)` (`TrainCli`, `TrainCliJson` in
  `CliTrainDataTests.cs:36-49`); registries are left as found (`Unregister` in `finally`).
- **Fixtures** come from a `tools/vlm/make_*.py` whose reruns write the same bytes (`torch.manual_seed`, one thread,
  deterministic algorithms, sorted JSON through `make_tiny.dump_json`, `shutil.rmtree` first); a `manifest.json` with
  versions and each file's size and SHA-256; a hand-written README; tolerances as `vlm-tuning/README.md`: losses within
  1e-4, gradients within 1e-4 of the largest, values within 1e-5, ids and masks exact.
- **A command** is an `internal sealed class XCommand : Command` added to `TrainCommands.All` or `DataCommands.All`
  (`src/Idrak.Cli/Commands/TrainCommands.cs:11`, `DataCommands.cs:11`), with `Name`, `Summary` (an Arabic entry in
  `Messages.Arabic.cs`, the summaries section from line 254), `Usage` (sections Arguments, Options, Examples in that
  order, a blank line before each heading, every option named, `-x, --long` for short forms, no line wider than 120),
  `ValueOptions`, `Flags`, `ShortForms` (one letter, not a common one, the same meaning across commands), required
  options as `UsageException`, library exceptions turned into `UsageException`, progress through `ProgressLine`
  (`src/Idrak.Cli/Shared/Progress.cs:22`), `context.Write` + `context.WriteJson`, `ExitCodes`. Add the command to the
  variable lists in `src/Idrak.Cli/Shared/EnvironmentVariables.cs` (`OnDevice`, `Models`, `Data`). The tests "cli
  polish", "cli health", "cli arabic" and "cli train data" check all of this.
- **Spec and run files** are read with `CliConfig.ReadOptions` (comments and trailing commas allowed), carry a
  `"format"` string (`"idrak-pretrain/1"`), map underscore keys to dashed options, and refuse unknown keys
  (`TuneCommand.ReadSettings`, `TuneCommand.cs:192-242`); the command line wins over the file.
- **Regenerate** after a public change: the inventory with
  `IDRAK_UPDATE_INVENTORY=1 IDRAK_DEVICES=cpu IDRAK_FILTER=inventory dotnet run -c Release --project tests/Idrak.Tests`
  and the API dump with
  `IDRAK_UPDATE_API=1 IDRAK_DEVICES=cpu IDRAK_FILTER="public API" dotnet run -c Release --project tests/Idrak.Tests`
  (no CHANGELOG line, whatever the test's message says).
- **Progress and telemetry.** Library loops report through `IProgress<T>` and publish `Telemetry.TrainingStarted`,
  `BatchCompleted` (one per optimizer step), `TrainingCompleted` (`src/Idrak.Abstraction/Diagnostics/Telemetry.cs:239-275`;
  `IsEnabled` is one static read and an AND). The CLI draws `ProgressLine` and writes the run folder's `log.jsonl`
  through `RunLogWriter` / `TelemetryJson.Write` (`src/Idrak.Cli/Shared/RunLogs.cs:185`), so `runs show` reads it.

---

# Track A: a language model from scratch

## Phase A0: the reference

**What.** `tools/vlm/make_tiny_pretrain.py` (imports `dump_json`, `save_npy`, `sha256`, `safetensors_index` from
`make_tiny.py`) writes `tests/Idrak.Tests/data/pretrain/`:

- `corpus.txt`: a small deterministic Arabic and English corpus (about 2,000 lines, generated in the script from fixed
  word lists and a seeded generator; no download), the same shape `idrak data text` writes (UTF-8, `\n`, documents
  separated by `---` lines).
- `tokenizer/tokenizer.json`: a byte-level BPE trained by Hugging Face `tokenizers` (`ByteLevel` pre-tokenizer with
  `add_prefix_space=false`, `ByteLevel` decoder, `BpeTrainer(vocab_size=512, min_frequency=2,
  special_tokens=["<pad>","<s>","</s>","<start_of_image>","<image>","<end_of_image>"], initial_alphabet=ByteLevel.alphabet())`),
  plus `tokenizer/train.json`: the vocabulary and merges in training order, the trainer's settings, and `encodings`:
  twenty corpus lines with their ids (so Idrak's trainer and encoder are checked separately).
- `tiny-llama/`: a Llama-style decoder (`LlamaForCausalLM`, hidden 32, 2 layers, 4 heads, 2 KV heads, head_dim 8,
  intermediate 64, vocab 512, max positions 64, RMSNorm, SwiGLU, RoPE 10000, tied embeddings) initialised with
  `torch.Generator().manual_seed(2026)` as `make_tiny.randomize` does (normal, std 0.02 for matrices and the embedding;
  norms 1), saved with `save_pretrained(safe_serialization=True)` + the tokenizer files + `tokenizer_config.json` with
  a ChatML-style Jinja template that walks content parts (text, and `<image>` for an image item) and `add_bos_token`,
  `special_tokens_map.json`; `finish_folder` as `make_tiny.py` does.
- `tokens.bin`: the corpus encoded with the tokenizer as uint16, documents joined with `</s>`.
- `step/`: one full-parameter step on a batch of 4 windows of 33 tokens (window i starts at i·stride, fixed in the
  JSON) with the mean token cross-entropy (every position trained), plain SGD at 0.2 (as plan 12's fixture, so no
  optimizer arithmetic is in the check): `loss-step1`, `grads-step1.safetensors` (every parameter's gradient by its
  stored name), `losses` for 3 steps, `weights-step3.safetensors`. A second run with AdamW (lr 1e-3, betas 0.9/0.95,
  eps 1e-8, weight decay 0.1, no clipping) for 3 steps: `adamw/losses`, `adamw/weights-step3.safetensors`, and the
  optimizer's moments after step 3 (`adamw/state-step3.safetensors`), for the resume check.
- `manifest.json` (versions, files with sizes and SHA-256), `README.md` (what the script did, tolerances, regenerate).

**Done when** the folder exists, two runs write the same bytes, the README names the versions and the commands.

## Phase A1: a model from a spec, saved as a folder

**Contracts and code.**

- `PretrainedArchitecture` (`src/Idrak/Models/Abstractions/PretrainedArchitectures.cs:16`) gains
  `Func<DecoderSpec, JsonObject>? Config { get; init; }`: the config.json of a spec in the family's layout (the inverse
  of `Spec`). The library fills it for its own families (`LibraryModelFormats.RegisterPretrainedArchitectures`): Llama
  first (every `DecoderSpec` field the Llama reader understands: `hidden_size`, `num_hidden_layers`,
  `num_attention_heads`, `num_key_value_heads`, `head_dim`, `intermediate_size`, `vocab_size`,
  `max_position_embeddings`, `rms_norm_eps`, `rope_theta` and `rope_scaling`, `tie_word_embeddings`,
  `attention_bias`, `mlp_bias`, `hidden_act`), then Qwen2, Gemma 3's text side and the rest as a follow-up. A family
  without `Config` refuses creation: `NotSupportedException("Architecture 'X' cannot be written from a spec (it reads
  checkpoints only); give a config.json or choose an architecture with a config writer.")`. A round trip test:
  `Spec(Config(spec)) == spec` for every field, on every family that has a writer.
- `PretrainedModel.Create` (core, `PretrainedModel.cs`, public, beside `Load`):
  ```csharp
  public static PretrainedModel Create(string folder, DecoderSpec spec, string architecture, ModelFiles files, CreateOptions? options = null)
  ```
  `ModelFiles` (a record): `TokenizerJson` (a path to copy, or a `BpeTokenizer` to write), `ChatTemplate` (Jinja
  source; null writes the library's ChatML-with-parts template), `BosToken`, `EosToken`, `PadToken`, `AddBosToken`,
  `Extra` (more files to copy, e.g. `preprocessor_config.json`). `CreateOptions`: `Device`, `Seed`, `InitStd`,
  `SafeTensorType` (default F32 for a model that will train), `MaxPositions`. It builds with
  `spec.Build(options: new DecoderBuildOptions { Device, Seed, InitStd })`, writes the folder (config.json from
  `Config(spec)` with `"architectures": [architecture]` and `"torch_dtype"`, `model.safetensors`, `tokenizer.json`,
  `tokenizer_config.json` with the template, bos/eos/pad and `add_bos_token`, `special_tokens_map.json`,
  `generation_config.json` with the eos id), then returns `Load(folder, new PretrainedOptions { Device })` so every
  invariant of a loaded model holds (`Folder`, `Tokenizer`, `ChatTemplate`, the checkpoint identity `TuningVision`
  hashes). The spec is also kept as `"idrak_spec"` in config.json for `explain`.
- `SaveHuggingFace` (`:623`) completed: writes `LayerNorm` (`weight`, `bias`), `PositionEmbedding` (the spec's learned
  positions), skips a `Linear.Tied` head as today, and when `Folder` is null or the files are missing writes
  config.json and the tokenizer files from the model's own `Config`, `Tokenizer` and `ChatTemplate` instead of copying.
- `BpeTokenizer.ToJson()` and `Save(path)` (core): the loaded model's `vocab`, `merges`, `added_tokens`, normalizer,
  pre-tokenizer and decoder written back in the tokenizer.json layout it reads (round trip: `Load(Save(t))` encodes
  every fixture text to the same ids).
- The library's ChatML-with-parts Jinja template (a resource string in `Idrak.Nlp`, `Resources/`): ChatML turns, a
  content list walked item by item, text written as is, an image item written as the family's marker text taken from a
  template variable (`image_marker`, default `<image>`), tools as `ChatMLTemplate` writes them, `add_generation_prompt`.

**Tests** (`tests/Idrak.Tests/PretrainTests.cs`, group `"pretrain:"`): config round trip per family; `Create` writes
the folder and `Load` gives the same tensors (compare `model.safetensors` to a second `SaveHuggingFace`); the created
tiny Llama with the fixture's seed and `InitStd` produces tensors with the fixture's shapes and names
(`safetensors_index` of `tiny-llama`), and loading the fixture's own weights into a created model (`Load` of
`tiny-llama/`) then `SaveHuggingFace` reproduces the bytes of its tensors; `idrak run tiny-llama "..."` answers;
`BpeTokenizer` save round trip. Inventory and API regenerated.

**Done when** the tests pass on the CPU and `IDRAK_FILTER="pretrain:"` is the only run.

## Phase A2: the tokenizer trainer

**Contracts** (`src/Idrak/Models/Abstractions/TokenizerTraining.cs`):

```csharp
public sealed record TokenizerTrainingOptions
{
    public int VocabularySize { get; init; } = 32000;
    public int MinFrequency { get; init; } = 2;
    public IReadOnlyList<string> SpecialTokens { get; init; } = [];   // added first, in order, as added_tokens with special=true
    public bool AddPrefixSpace { get; init; }                           // ByteLevel's add_prefix_space
    public string PreTokenizer { get; init; } = "bytelevel";            // the pre-tokenizer written to tokenizer.json ("bytelevel" | "metaspace" | "whitespace")
    public int? LimitAlphabet { get; init; }
    public int Workers { get; init; }                                   // 0: the CPU threads (rule 79)
}
public interface ITokenizerTrainer
{
    string Name { get; }
    string Summary { get; }
    BpeTokenizer Train(IReadOnlyList<string> files, TokenizerTrainingOptions options, IProgress<TokenizerTrainingProgress>? progress = null, CancellationToken cancellationToken = default);
}
public static class TokenizerTrainers   // the registry: "bpe" the library default
```

**The "bpe" trainer** (`src/Idrak/Models/BpeTrainer.cs`), Hugging Face `BpeTrainer` semantics so the fixture is the
check: read each file as UTF-8 bytes in blocks (rule 74; pooled buffers returned in `finally`; a block ends on a
newline so a word is never cut), pre-tokenize with the GPT-2 regex on bytes (the ByteLevel pre-tokenizer
`BpeTokenizer` already has, reused), count words in one pass per block with a dictionary keyed by `ReadOnlySpan<byte>`
(alternate lookup, rule 15) merged across workers (rule 79, measured); map bytes to the byte-level alphabet; then the
merge loop: pair counts over distinct words weighted by their counts, a max-heap on (count, then the pair ordered as
`tokenizers` orders it, so ties resolve the same way; the fixture's merge order is the test), words updated in place
(rule 73), until `VocabularySize` or no pair reaches `MinFrequency`. Output: a `BpeTokenizer` whose `ToJson` writes
`model.type "BPE"`, `vocab`, `merges`, `byte_fallback false`, the `ByteLevel` pre-tokenizer and decoder, and the
special tokens as `added_tokens`. `TokenizerTrainingProgress(Stage, Done, Total)` for the two stages (counting words,
merging). A `TokenizerTrainerSuite` in the kit (fixed cases: tiny corpora with known merges; random cases: round trip
of the trained tokenizer through `TokenizerSuite`).

**Tests** (`"tokenizer training:"`): the trained vocabulary and merges equal `tokenizer/train.json` (exact, in order);
the twenty `encodings` match; `TokenizerSuite` on the result; `Describe` names "bpe"; an unregistered name is refused;
the counting stage's time with one worker and with all workers is reported in `-v` (rule 79, "check it paid").

**Done when** the fixture matches exactly on the CPU.

## Phase A3: the pretraining loop

**Design.** `Pretrainer` (`src/Idrak.Nlp/Pretraining.cs`) is a new loop over `FineTuner`'s step: the pieces
`StepRunner`, `NetworkLoss`, `BatchLoss`, `StepImages`, `Prepare`, `CreateOptimizer`, `Update` and the packing of
`MakeBatches` become `internal` members shared by both (same assembly; no behaviour change for `FineTuner`, whose
tests stay green). What is new is the trained set, the data, the stages, the checkpoints and the telemetry.

**Options** (`src/Idrak.Nlp/Pretraining.cs`):

```csharp
public sealed record PretrainingSource(string Kind, IReadOnlyList<string> Paths)   // Kind: a PretrainingSources name
{
    public double Weight { get; init; } = 1;        // share of each step's rows, normalised over the stage's sources
    public string? Format { get; init; }            // records: a TuningDataFormats name, else detected
    public string? Images { get; init; }            // records: the images folder or zip
    public TokenType TokenType { get; init; } = TokenType.UInt16;   // tokens: the file's element type (or .npy)
}
public sealed record PretrainingStage(string Name)
{
    public int Steps { get; init; }                                        // optimizer steps in this stage
    public IReadOnlyList<string> Trained { get; init; } = ["decoder"];    // "decoder", "embeddings", "head", VisionTuningParts.Tower, VisionTuningParts.Projector
    public IReadOnlyList<PretrainingSource> Sources { get; init; } = [];
    public float LearningRate { get; init; } = 3e-4f;
    public float MinLearningRate { get; init; }
    public float WarmupFraction { get; init; } = 0.03f;
    public string Schedule { get; init; } = "cosine";                     // FineTuningSchedules names
}
public sealed record PretrainingOptions
{
    public IReadOnlyList<PretrainingStage> Stages { get; init; } = [];
    public int MaxLength { get; init; } = 2048;        // window length for token sources; longest record otherwise
    public int BatchTokens { get; init; } = 4096;       // rows × positions per micro-batch, as FineTuningOptions.BatchTokens
    public int GradientAccumulation { get; init; } = 1;
    public float MaxGradientNorm { get; init; } = 1f;
    public float WeightDecay { get; init; } = 0.1f;
    public string Optimizer { get; init; } = "adamw";   // FineTuningOptimizers names; HostOptimizer as the flag below
    public bool HostOptimizer { get; init; }
    public bool? Checkpointing { get; init; }           // the ladder's three settings, as FineTuningOptions
    public bool? RecomputeFeedForward { get; init; }
    public bool? BFloat16Activations { get; init; }
    public bool CudaGraphs { get; init; } = true;
    public bool Packing { get; init; } = true;
    public int Seed { get; init; }
    public int SaveEvery { get; init; }                 // steps between checkpoints (0: stage ends and the end)
    public int EvaluateEvery { get; init; }
    public IReadOnlyList<PretrainingSource> Evaluation { get; init; } = [];
    public int NonFiniteSteps { get; init; } = 8;
    public int LossChunkRows { get; init; } = 512;
    public TuningImages Images { get; init; } = TuningImages.None;   // transforms and vision options for record images
    public string FeatureCache { get; init; } = FeatureCaches.Memory;
}
public static class Pretrainer
{
    public static PretrainingResult Train(PretrainedModel model, PretrainingOptions options, string outputFolder,
        IProgress<PretrainingProgress>? progress = null, CancellationToken cancellationToken = default, Action<string>? trace = null);
    public static PretrainingResult Resume(string checkpointFolder, PretrainingOptions? overrides = null, ...);   // same signature after the folder
}
public sealed record PretrainingProgress(string Stage, int Step, int StageSteps, int TotalSteps, float Loss, IReadOnlyDictionary<string, float> LossBySource,
    float LearningRate, double TokensPerSecond, double ImagesPerSecond, long DeviceBytes, TimeSpan Remaining, float? EvaluationLoss);
```

**Sources** (`src/Idrak.Nlp/Abstractions/PretrainingData.cs`): `IPretrainingSource { Name; Summary; IPretrainingReader
Open(PretrainingSource source, PretrainingContext context) }` and the registry `PretrainingSources` with two library
defaults. `"tokens"`: `TokenFileSource(path, MaxLength)` per file; a window is a `TrainingSequence(ids[0..L], Trained
= all true)`; order from a seeded permutation over all files' windows, an epoch counter per source, no overlap
(`stride = length`). `"records"`: `TuningDataFormats.Get/Detect` → `PendingConversations` → `ChatTranscriptEncoder.Encode`
(with `Vision` when the model has one; the masking rule of plan 12 phase 0: prompt and image tokens untrained, the
answer and its end of turn trained) in a worker pool ahead of the step (as `ConversationTuning.Load` encodes), images
read through `TuningVision`. A reader gives `TrainingSequence`s in a deterministic order from (`Seed`, source index,
epoch) and reports `Position` for the checkpoint. A `PretrainingSourceSuite` in the kit (a source gives the same
sequence of records for the same seed; a position round-trips).

**The trained set.** No adapters: for each stage, every parameter of the network gets `RequiresGrad = Trained.Contains
(its group)` ("decoder" = blocks and the final norm, "embeddings" = `embed` (+ `pos`), "head" = the head unless tied);
vision parts through `TuningVision.Create(model, options.Images, parts, cache)` with `parts` = the stage's vision
names (`VisionTuningParts.Tower`, `Projector`), `Parameters` joined. The optimizer is rebuilt at each stage boundary
over the stage's parameters (`FineTuningOptimizers.Create(name, lr, weightDecay)`, `HostOptimizer` when asked), its
schedule from `FineTuningSchedules.Create(stage.Schedule, warmup, minLr)` over the stage's steps. A tied head's
embedding appears once in the list (dedupe by reference).

**The step.** Each optimizer step draws rows from the stage's sources by `Weight` (a deterministic pattern from the
seed: the source with the largest deficit next), packs them as `MakeBatches` does (packing when `PackedSequences.Supports`;
rows with images laid out as `StepImages` does), and runs `StepRunner.Run` with the group of `GradientAccumulation`
micro-batches; the ladder (feed-forward recompute, bf16 activations, checkpointing, micro-batch halving) applies
unchanged; graphs record only when no vision part trains, as today. The loss is the fused
`Losses.TokenCrossEntropyRows` over trained rows (no custom loss in A3). Loss per source is accumulated from each row's
share (`Prepare`'s weights by source) for the progress record.

**Checkpoints and resume.** `outputFolder/checkpoint-{step}/`: `model.safetensors` + config and tokenizer files through
`SaveHuggingFace(folder, F32)` (vision parts included after B1), `optimizer.safetensors` through the new
`Optimizer.SaveState(Stream)` (Adam/AdamW: `m`, `v` and the step per parameter in parameter order; `AdamW8Bit`: the
8-bit blocks and their scales; `HostOptimizer`: its CPU optimizers; `GroupedOptimizer`: its groups; each `LoadState`
checks shapes and refuses a mismatch), and `pretrain-state.json` (`format "idrak-pretrain-state/1"`, the stage index,
step, every source's epoch and position, the RNG step, the options as given, the ladder's settled settings, the loss
history). `Resume` reloads the model from the checkpoint, the optimizer state, the positions and the RNG, and continues;
on the CPU a resumed run's next losses equal the uninterrupted run's bit for bit (the test). `last/` is updated at
every save; `best/` on the lowest evaluation loss.

**Telemetry and progress.** `Telemetry.TrainingStarted` once (model, optimizer, device, parameter count, learning rate),
`BatchCompleted` per optimizer step (loss, rate, tokens), `TrainingCompleted`; `IProgress<PretrainingProgress>` every
step with the device's peak bytes (the backend's peak counter, as `MeasuredBatches` reads it) and the time left from the
measured steps per second; the trace tells each ladder rung once.

**Tests** (`"pretrain:"`): step 1 on the fixture's 4 windows with plain SGD 0.2 (`Optimizer = "sgd"`, momentum 0, no
clipping, no weight decay): loss within 1e-4 and every gradient within 1e-4 of the fixture's `grads-step1`; 3 steps'
losses and `weights-step3` within 1e-5; the AdamW run's 3 losses and weights; a stage with `Trained = ["head"]` moves
the head only; two sources mixed 3:1 give rows in that ratio over 40 steps; packing on and off give the same loss on
the CPU; a run stopped after step 2 and resumed from `checkpoint-2` gives the uninterrupted run's losses for steps 3
to 6 (bit for bit; AdamW state included); `SaveEvery` writes and `last/` loads through `PretrainedModel.Load`;
`Telemetry` receives one `BatchCompleted` per step; the loss on `tokens.bin` falls over 200 CPU steps of the tiny model
(a smoke test, no threshold but "lower than step 1 by half"). `FineTuner`'s filters `"vlm"` and `"cli vision tuning"`
still pass (the refactor changed no behaviour).

**Done when** every test above passes on the CPU with `--no-build` after one build of Idrak, Idrak.Nlp and the tests.

## Phase A4: the command line

- **`idrak create SPEC.json -o FOLDER`** (`src/Idrak.Cli/Commands/Train/CreateCommand.cs`, in `TrainCommands.All`):
  SPEC is `{"format": "idrak-create/1", "architecture": "LlamaForCausalLM", "spec": {DecoderSpec JSON}, "tokenizer":
  "path/tokenizer.json", "chat_template": "path.jinja" | null, "bos": "<s>", "eos": "</s>", "pad": "<pad>",
  "seed": 0, "init_std": 0.02, "dtype": "f32"}`; every key also an option (`--architecture`, `--tokenizer`,
  `--chat-template`, `--seed`, `--init-std`, `--dtype`; `-o/--out`, `-t/--tokenizer`); with B1, `"vision": {...}` and
  `-P`. Prints the parameter count, the files written and a "next: idrak pretrain FOLDER ..." hint; `-j` gives them as
  JSON. An unknown architecture or one without a config writer exits 1 with the library's message.
- **`idrak pretrain FOLDER [--config pretrain.json] -o RUN`** (`PretrainCommand.cs`): FOLDER is a created or any
  float32 model folder; `pretrain.json` (`"format": "idrak-pretrain/1"`, the `PretrainingOptions` with stages and
  sources as JSON; `idrak pretrain init` writes a commented one, as `tune init` does) or the one-stage options
  `--tokens FILE... --data FILE... --images DIR|ZIP --weight NAME=W... --steps N --lr X --min-lr X --warmup F
  --schedule NAME --trained decoder,embeddings,head --max-length N --batch-tokens N --accumulate N --optimizer NAME
  --weight-decay X --clip X --save-every N --eval FILE... --eval-every N --image-transform ... --vision K=V
  --feature-cache NAME --checkpointing/--no-checkpointing --recompute --bf16-activations --host-optimizer --no-packing
  --no-graphs --seed N`. The run folder (`CACHE/runs/yyyyMMdd-HHmmss-NAME` as `NewRunFolder`) holds `run.json`
  (`"format": "idrak-pretrain-run/1"`, the model folder and the options), `log.jsonl` (telemetry records), the
  checkpoints, `training.json`. Ctrl+C and `--timeout` save a checkpoint and exit (`Interrupt`). Progress:
  `ProgressLine` with "stage s step i/n  loss  lr  tok/s  img/s  mem  left", each ladder rung and each checkpoint as a
  line (`-v` for the per-source losses).
- **`idrak resume RUN`** (`ResumeCommand`, `TrainingCommands.cs:237`) dispatches on `run.json`'s format to
  `Pretrainer.Resume(last checkpoint)`; `--steps N` adds steps to the last stage. **`runs show`** reads the
  pretraining log's `batch` records (steps instead of epochs; `TextPlot` over the loss).
- **`idrak data tokenizer FILE... --vocab N -o tokenizer.json [--min-frequency N] [--special TOK]... [--pre-tokenizer
  bytelevel|metaspace|whitespace] [--prefix-space] [--trainer NAME]`** (`Commands/Data/DataTokenizerCommand.cs`, in
  `DataCommands.All`): `TokenizerTrainers.Get(name).Train(files, options, progress)`, then `Save`; prints the vocabulary
  size, merges, the time per stage; `-j`. And **`idrak data tokens FILE... --tokenizer tokenizer.json -o tokens.bin
  [--type u16|u32] [--separator </s>]`**: encodes text files to a token file for `"tokens"` sources (bytes parsed in
  blocks, `EncodeRange`, pooled buffers), with a `.json` beside it (count, type, tokenizer hash).
- Arabic summaries for the four commands; `EnvironmentVariables` lists updated; `idrak explain FOLDER` shows a created
  model's spec from `idrak_spec`.

**Tests** (`"cli pretrain:"`): `create` from a spec file and from options; `data tokenizer` on `corpus.txt` reproduces
the fixture's tokenizer.json (vocab and merges); `data tokens` reproduces `tokens.bin`; `pretrain` 4 steps on
`tokens.bin` writes `run.json`, `log.jsonl` and `checkpoint-4`; `resume` continues 2 steps; `runs show` prints the
loss; `run` and `chat` on `last/` answer; an unregistered architecture, trainer or source is refused with the `-P`
hint; `help` for each passes the polish tests.

**Done when** the CLI tests pass on the CPU and the four `cli` health filters still pass.

## Phase A5: the owner's proof (text)

The owner's run, from `D:\Projects\Idrak`, with the owner's corpus (or `idrak data text --unit pages --count N` to
start): `data tokenizer`, `data tokens`, `create` (a spec of the owner's sizes; nothing in the library prefers one),
`pretrain` one `text` stage, `run`. The GPU filters: `"pretrain:"`, `"cli pretrain:"` on `cuda` and `vulkan`.

---

## Track A, as built (2026-10-14)

Built on `abstraction` in four commits (A0 and A2, A1, A3, A4), every phase checked on the CPU here.

- **A0.** `tools/vlm/make_tiny_pretrain.py` → `tests/Idrak.Tests/data/pretrain` (about 1.5 MB; README and manifest there;
  reruns write the same bytes, the hand-written README kept). transformers 5.19.0, tokenizers 0.23.3, torch 2.14.1.
  Beside the plan: `tokenizer-full/` (merging until no pair is left, 1,628 merges through many ties) and
  `step/adamw-folded/` (AdamW as Idrak computes it, see A3).
- **A2.** `ITokenizerTrainer`, `TokenizerTrainingOptions`, `TokenizerTrainingProgress` and the `TokenizerTrainers`
  registry ("bpe", unguarded) in `Idrak.Models.Abstractions`; `BpeTrainer` (`src/Idrak/Models/BpeTrainer.cs`) follows
  tokenizers 0.23's `trainer.rs` and `word.rs` line by line (the queue's order, stale counts, a pair queued twice merged
  twice with its later rank written), words counted from UTF-8 blocks on every processor; pre-tokenizers "bytelevel"
  and "metaspace". `BpeTokenizer.Save`, `ToJson`, `WriteTo` (the model's vocabulary as read, added tokens outside it
  kept outside). Tests `"tokenizer training:"` 5: the vocabulary, merges and sample ids exactly Hugging Face's for both
  pre-tokenizers and the full run; save round trip, the tokenizer contract suite, one worker against four.
- **A1.** `PretrainedArchitecture.Config`; `PretrainedFamilies.CommonConfig` and `LlamaStyle(adjust, modelType)` give
  Llama, Mistral, Qwen2, Qwen3 and Gemma a writer, read back and refused with each differing value; numbers written as
  a file holds them. `PretrainedModel.Create(folder, spec, architecture, ModelFiles, CreateOptions)`, `ModelFiles`
  (`ChatMLWithParts`, the template created models get), `CreateOptions`. `SaveHuggingFace` writes `LayerNorm` and
  `PositionEmbedding` and replaces model.safetensors through a temporary file (it can save over its own folder). Tests
  `"pretrain:"` 4 of 8. Not built: a contract suite for trainers (the kit cannot name core's `BpeTokenizer`; the
  trained tokenizer goes through `TokenizerSuite` instead).
- **A3.** `Pretrainer.Train(model, options, output, resume, progress, cancel, trace)`, `PretrainingOptions` (JSON,
  "idrak-pretrain/1"), `PretrainingStage`, `PretrainingProgress`, `PretrainingResult`, `PretrainingCheckpoint`,
  `ReadCheckpoint`, `LatestCheckpoint`; `PretrainingSource`, `PretrainingContext`, `IPretrainingReader`,
  `IPretrainingSource` and `PretrainingSources` ("tokens", "records") in `Idrak.Nlp.Abstractions`. The fine-tuner's
  `StepRunner`, `Batch`, `MakeBatches`, `CustomLoss`, `BatchData` and `StepImages` are internal now; the runner takes
  each step's sequences (`Run(group, sequences, …)`), so data streams. `Optimizer.SaveState` / `LoadState` and
  `OptimizerStateValue` (Abstraction) on Adam, AdamW, Sgd, AdamW8Bit, HostOptimizer and GroupedOptimizer. A checkpoint
  is `checkpoint-STEP/`: a model folder, `optimizer.safetensors`, `pretrain-state.json` (stage, its steps, every
  source's pass, next item and draws, losses, options); `KeepCheckpoints` removes the oldest. Telemetry
  `TrainingStarted`, `BatchCompleted` per step, `TrainingCompleted`. **Found:** Idrak's AdamW folds both bias
  corrections into the step size, so its epsilon is ε/√(1−β2^t) of torch's; losses agree with torch's AdamW within
  1e-4, weights within 1e-5 of the same formula (`adamw-folded`), and differ from torch's in the fifth digit where a
  second moment is near ε². Not changed (it would change every fine-tune's numbers). Not built: per-source losses in
  the progress record. Tests `"pretrain:"` 8: transformers' step-1 loss and every gradient, SGD's three steps, AdamW's,
  a run stopped after step 2 and resumed giving the uninterrupted losses bit for bit, trained parts, 3:1 source
  weights, records, telemetry, the tiny model's loss falling by half in 200 steps, refusals. Fine-tuning, vision
  tuning, conversation tuning and distillation (56 tests) pass unchanged.
- **A4.** `idrak data tokenizer`, `idrak data tokens` (core `TokenFileWriter`), `idrak create`, `idrak create init`,
  `idrak pretrain` (common `-C, --config` for pretrain.json, as tune does), `idrak pretrain init`; `idrak resume`
  continues a pretraining run (`--steps N` more in its last stage); `runs show` reads a log of steps. Shared
  `TextFiles.Expand` (files, folders, wildcards, so PowerShell patterns work). Tests `"cli pretrain:"` 3; `"cli
  health"` passes. Three CLI checks fail exactly as they do on the base before this session (`"cli train data: help"`:
  `-n` means three things; `"cli polish: help lists"`: `--head-weights` undescribed, `data` and `data fonts` help lines
  too wide; `"cli arabic"`: the `data fonts` summary's tatweel), none of their lines naming these commands.

**For the owner (GPU and the proof run, from `D:\Projects\Idrak`):**

```powershell
cd D:\Projects\Idrak
git pull
dotnet build -c Release tests\Idrak.Tests
$env:IDRAK_DEVICES="cuda"; $env:IDRAK_FILTER="pretrain:|cli pretrain:"; dotnet run -c Release --no-build --project tests\Idrak.Tests
$env:IDRAK_DEVICES="vulkan"; $env:IDRAK_FILTER="pretrain:|cli pretrain:"; dotnet run -c Release --no-build --project tests\Idrak.Tests
```

A5, a first model of your own (sizes in `spec.json` are yours to set; the corpus is yours, or random text to start):

```powershell
dotnet run -c Release --project src\Idrak.Cli -- data text -n 20000 --unit pages -o work\corpus.txt
dotnet run -c Release --project src\Idrak.Cli -- data tokenizer work\corpus.txt --vocab 32000 -o work\tokenizer.json
dotnet run -c Release --project src\Idrak.Cli -- data tokens work\corpus.txt --tokenizer work\tokenizer.json -o work\tokens.bin
dotnet run -c Release --project src\Idrak.Cli -- create init -o work\spec.json
dotnet run -c Release --project src\Idrak.Cli -- create work\spec.json -o work\model
dotnet run -c Release --project src\Idrak.Cli -- pretrain work\model --tokens work\tokens.bin --steps 2000 --lr 6e-4 -o work\run -d cuda:0
dotnet run -c Release --project src\Idrak.Cli -- resume work\run
dotnet run -c Release --project src\Idrak.Cli -- runs show work\run
```

---

# Track B: a vision-language model on the created language model

## Phase B0: the reference

`tools/vlm/make_tiny_scratch_vlm.py` writes `tests/Idrak.Tests/data/vlm-scratch/`:

- `tiny-scratch/`: a model folder in the new family's layout (architecture name
  `IdrakVisionForConditionalGeneration`, decided): config.json with `architectures`, `text_config` (the
  tiny Llama of A0, same tokenizer with the three image special tokens), `vision_config` (`hidden_size 16, image_size
  28, patch_size 7, num_hidden_layers 3, num_attention_heads 2, intermediate_size 32, layer_norm_eps 1e-6, hidden_act
  gelu_pytorch_tanh, pool_size 2` → 16 patches pooled to 4 image tokens), `projector` (`hidden_act gelu, bias true`),
  `image_token_index`, `boi_token_index`, `eoi_token_index`, `mm_tokens_per_image 4`; `model.safetensors` under the
  names `vision_tower.*` (`embeddings.patch_embedding.{weight,bias}`, `embeddings.position_embedding.weight`,
  `encoder.layers.N.{layer_norm1,self_attn.{q,k,v,out}_proj,layer_norm2,mlp.fc1,mlp.fc2}`, `post_layernorm`),
  `multi_modal_projector.{linear_1,linear_2}`, `language_model.model.*`, initialised with one seeded generator
  (seed 2027); `tokenizer.json`, `tokenizer_config.json` with the chat template from A0 (`<image>` marker written
  as `<start_of_image><image><end_of_image>` is the prompt format's job, not the template's: the template writes the
  marker only), `preprocessor_config.json` (resize 28×28 bilinear, rescale 1/255, mean 0.5, std 0.5, RGB).
- The reference model in the script: patch conv (stride = patch), learned positions added, pre-norm encoder layers
  with GELU-tanh, final LayerNorm, average pooling over `pool_size × pool_size` patch squares, projector
  Linear–GELU–Linear to the text width; the decoder from A0's weights; image tokens' embeddings replaced by the
  features (not scaled), attention causal with each image's tokens bidirectional among themselves (plan 11's rule).
- Three records (the images of `tests/Idrak.Tests/data/vlm/`: `image.png`, `image-palette.png`, one record with two
  images), one with a system line, in ShareGPT with `images` lists (`records.json`), their rendered and expanded text,
  ids, trained mask and image blocks (plan 12's masking) in `tuning.json`.
- `joint/`: one step with every parameter trained (SGD 0.2): loss, `grads-step1.safetensors` (tower, projector and
  decoder), 3 steps' losses, `weights-step3.safetensors`. `align/`: the same with the decoder frozen (tower and projector
  only). `mixed/`: a step whose batch is two text windows from `pretrain/tokens.bin` and one record (the text rows'
  loss and the record's loss, and the combined loss).
- `reference/`: `pixel_values`, `tower`, `features` per image (`.npy`), logits of one prompt.
- `manifest.json`, `README.md`.

**Done when** the folder exists and reruns match.

## Phase B1: the spec and the `Create` contract

- **`VisionLanguageSpec`** (`src/Idrak.Abstraction/Generation/VisionLanguageSpec.cs`):
  ```csharp
  public sealed record VisionLanguageSpec(string Architecture, DecoderSpec Text, JsonObject Vision)
  {
      public int ImageToken { get; init; }            // the soft token repeated per image token
      public int? BeginImage { get; init; }           // <start_of_image>, or null
      public int? EndImage { get; init; }
      public int TokensPerImage { get; init; }
      public JsonObject? Preprocessing { get; init; } // preprocessor_config.json's content
      public JsonObject ToJson(); public static VisionLanguageSpec FromJson(JsonObject);
  }
  ```
  `Vision` is the family's own config (the library does not read it: card-agnostic and family-agnostic); the text
  architecture's `Config` writes `text_config`; the token ids go to config.json's top level under the family's key
  names (the family says which: `IVisionFamily.ConfigKeys`, below).
- **`IVisionFamily`** (`VisionFamilies.cs:23`) gains two default members:
  ```csharp
  PretrainedVision Create(VisionLanguageSpec spec, int textDim, Device device, int seed)
      => throw new NotSupportedException($"Vision family '{Name}' cannot be created from a spec (it reads checkpoints only).");
  JsonObject Config(VisionLanguageSpec spec) => throw ...;   // config.json's vision keys for Create's result
  ```
  A created `PretrainedVision` has `StoredTensors` empty, implements `IVisionTuningPart` with every part trainable, and
  its `Export(encoder, part)` gives the tensors under the family's checkpoint names, which is how they are saved.
- **`PretrainedModel.Create`** overload: `Create(string folder, VisionLanguageSpec spec, ModelFiles files, CreateOptions?)`:
  builds the decoder from `spec.Text` through the architecture registered under `spec.Architecture`
  (`PretrainedArchitectures.Get`), the vision part through `VisionFamilies.Get(spec.Architecture).Create(spec, dim,
  device, seed)`, an encoder with `TrainedParts = all`, writes config.json (`architectures`, `text_config`,
  `vision_config` and keys from `Config`), `model.safetensors` with the decoder's tensors and every part's `Export`,
  `preprocessor_config.json` from `Preprocessing`, the tokenizer and template files, then `Load`s the folder, so
  `Read` is exercised at once.
- **`SaveHuggingFace`** writes vision parts: when `Vision is IVisionTuningPart parts` and the model holds a vision
  encoder created for training (`TrainedVisionTensors` or a created model), every part's `Export` goes into
  `model.safetensors`; otherwise the checkpoint's vision tensors are copied from the source store (so a merged export
  of a loaded Gemma 3 is complete too, closing a gap plan 12 left).

**Tests** (`"vlm scratch:"`, `tests/Idrak.Tests/VlmScratchTests.cs`): `Create` with the fixture's spec writes a folder
whose tensor names and shapes equal `tiny-scratch`'s index; loading `tiny-scratch` through the family and
`SaveHuggingFace` reproduces its tensors; a family without `Create` refuses with its name; `VisionLanguageSpec` JSON
round trip. Inventory and API regenerated (Abstraction +1 record, core +0 registries, the interface's members).

**Done when** the tests pass with the family of B2 (B1 and B2 are built together; B1's tests need B2's family).

## Phase B2: the family plug-in

`samples/IdrakVision/Idrak.IdrakVision/` (a Library, `PackageReference Idrak` and `Idrak.Imaging` as the Gemma 3
sample; referenced by `tests/Idrak.Tests` as the Gemma 3 and Qwen samples are), files:

- `IdrakVisionPlugin.cs`: `Register()` → `ImageAttentionRules.Register(new ImageBlockRule())` under
  `"scratch-image-blocks"` (`KeySpans.ImageBlocks`, as Gemma 3's; the plug-in owns its copy) and
  `VisionFamilies.Register(new IdrakVisionFamily())`; `RegisterIdrakPlugin()`; `Unregister()`. The text side is the
  library's Llama architecture read from `text_config` (registered once under the family's architecture name through
  `PretrainedArchitectures.Register(Architecture, Llama with text_config)` as `LlavaText` does, `Llava.cs:42-113`).
- `IdrakVisionConfig.cs`: `Dim, FfDim, Layers, Heads, ImageSize, PatchSize, Channels, LayerNormEpsilon, Activation,
  PoolSize, ProjectorActivation, ProjectorBias`, `FromJson`, `ToJson`; `IdrakVisionImageTokens(BeginImage?, EndImage?,
  ImageToken, TokensPerImage)`.
- `IdrakVisionTower.cs`: `Module` from core layers: `Conv2d(channels, dim, patch, stride: patch)`, a position
  parameter `[patches, dim]`, `TransformerEncoderLayer(dim, heads, ffDim, dropout: 0f)` × layers (its `LayerNorm`s
  replaced by `LayerNorm(dim, epsilon)` through `FromLayers`), a final `LayerNorm`, pooling by reshape and mean
  (as `Gemma3Projector` pools). Two constructors: `Random(config, device, Random)` (uniform ±sqrt(6/fanIn) for
  convolutions, Xavier for linears, positions normal 0.02, norms 1 and 0) and `FromTensors(config, store, prefix,
  device, weights)`; `ForwardTraining(x, checkpointed)` with `ForwardCheckpointed` per block; `Export(prefix)`.
- `IdrakVisionProjector.cs`: Linear–activation–Linear; `Random` and `FromTensors`; `Export`.
- `IdrakVisionFamily.cs`: `IVisionFamily` with `Read` (the one layout `vision_tower.` / `multi_modal_projector.` /
  `language_model.model.`; shapes checked as `LlavaVisionFamily.Read` does), `Create`, `Config`.
- `IdrakVision.cs`: `PretrainedVision, IVisionTuningPart`: `PromptFormat = new ImageTokenFormat("scratch", marker:
  ImageToken, ImageToken, BeginImage, EndImage)` (the template writes `<image>` once per image; `Expand` replaces it by
  begin + N soft tokens + end), `Attention = ImageAttentionRules.Get("scratch-image-blocks")`, `Preprocessor` from
  `preprocessor_config.json` (`ImagePreprocessor.FromConfig`), `CreateEncoder(options)` with `Trains(part)` → float32,
  `TrainableParts = [Tower, Projector]`, `Parameters`, `Tower`, `Features`, `Export`, `Import`, `PartsOf`.
- `IdrakVisionImageEncoder.cs`: `Module, IVisionEncoder, IVisionEncoderStages` (one block per image, `Blocks`, `Encode`,
  `PixelValues`, `Tower`, `Features`), as `Gemma3ImageEncoder` without pan and scan.
- `README.md`: the family's layout, how to create one, how to register it.

**Tests** (`"vlm scratch:"`): `Conformance.CheckVisionEncoder` on the created encoder; `pixel_values`, `tower` and
`features` of the fixture's images within 1e-5 of `reference/`; the logits of the fixture prompt; `idrak run -P
Idrak.IdrakVision.dll tiny-scratch --image image.png "..."` answers through the CLI; `idrak tune -P ... --parts
projector` runs one LoRA step (the family is tunable like any other); an outside family test as
`VisionFamilyPluginTests.OutsideFamily` does (`Origin` is the sample's assembly, `Default` null).

**Done when** the family reads, creates, encodes and tunes on the CPU.

## Phase B3: the loop with images and stages

- **Sources with images**: `"records"` sources carry `Images` and the run's `TuningImages`; `Pretrainer` makes one
  `TuningVision.Create(model, options.Images, parts of the stage, cache)` per stage (the cache keeps the frozen stage:
  features when nothing of the vision trains, the tower output when the projector trains, pixels when the tower does).
- **Stages**: `Trained` accepts `tower` and `projector`; the `align` recipe is `Trained = [tower, projector]`
  (decoder, embeddings and head frozen; the decoder still runs forward and backward to the features), `joint` is
  everything. Rows with images are packed with text rows in one micro-batch (`PackedSequences.SupportsImages`),
  images attend by the family's rule; graphs stay off while a vision part trains (the existing rule); the ladder's
  micro-batch halving covers an image batch the device has no room for.
- **Checkpoints** include the vision parts (B1's `SaveHuggingFace`) and the feature cache's folder when it is on disk.
- **Evaluation**: the evaluation sources' loss (`FineTuner.Evaluate` with the vision), and optionally
  `--metric cer|wer --metric-samples N` through `TuningAnswerScorer.Score(model, vision)` on N records at
  `EvaluateEvery` (one request at a time, as it is today; a batched path is a later speed item).

**Tests** (`"pretrain vision:"`): the `joint` step's loss and every gradient (tower, projector, decoder) against
`vlm-scratch/joint`; `align`'s (decoder gradients absent, its weights unchanged after the step); `mixed`'s combined loss
and per-source losses; packing with and without images gives the same loss; a 2-stage run (`align` 2 steps then `joint`
2 steps) resumed after the first stage matches bit for bit; `last/` loads and `idrak run --image` answers.

**Done when** the fixture matches on the CPU.

## Phase B4: the command line with vision

- `idrak create SPEC.json -o FOLDER -P plugin.dll` where SPEC has `"vision": {family config}`, `"image_tokens": {...}`,
  `"preprocessing": {...}`; the family's `Create`; the refusal names `VisionFamilies.Register` and `-P`.
- `idrak pretrain FOLDER -P plugin.dll --config pretrain.json` with record sources (`--data pages.jsonl --images
  pages.zip`, plan 18's output as it is), `--image-transform`, `--vision K=V`, `--feature-cache`, `--parts` per stage in
  the JSON; `resume` with `-P` again (as `idrak resume` for image runs does).
- `pretrain init --vision` writes a two-stage (`align`, `joint`) commented `pretrain.json`.

**Tests** (`"cli pretrain vision:"`): create with the plug-in, pretrain 2 + 2 steps on the fixture's records and
`tokens.bin`, resume, run with an image, tune the result with LoRA, an unregistered family refused with the `-P` hint.

## Phase B5: distillation from a loaded vision-language teacher (optional, the quality lever)

- `ModelTeacher.Hidden` (`Distillation.cs:148`) runs the teacher with its own `TuningVision` (created from the teacher
  model with the same `TuningImages`, no parts trained, a cache) through `ImagePrefill.Begin` / `Substitute` on the
  student's `TrainingSequence.Images` re-expanded for the teacher's prompt format (the teacher's image-token count may
  differ; the trained positions are aligned by the answer's tokens, which both tokenizers must share:
  `DistillationTeacher.CheckVocabulary` already requires one vocabulary, so a created student uses the teacher's
  tokenizer file in `idrak create --tokenizer`).
- `TeacherLogitsWriter` keys records by content including each image's hash (`ChatImage.Hash`).
- `idrak distill` passes `--images`, `--image-transform`, `--vision`, `--feature-cache`, `--data-format` through to the
  tool, and `--student` may be a created folder; `Pretrainer` accepts a `Teacher` (the `CustomLoss` path with
  `FineTuningLosses.Distillation(T, alpha)`).

**Tests** (`"distill vision:"`): the tiny LLaVA as teacher and the tiny scratch model as student on the fixture's
records: the divergence term equals a reference computed in the script (added to B0 as `distill/`), and `idrak distill`
runs one step.

## Phase B6: the owner's proof (vision)

The owner's run: `idrak data synth` pages from the owner's texts (plan 18), the text model of A5 as the decoder (its
folder given as `"text"` in the create spec, weights copied, so the `text` stage is not repeated), `create` with the
family, `pretrain` with `align` then `joint`, `tune evaluate --metric cer` against the tuned Qwen-VL adapter of plan 18's
"Why". GPU filters: `"vlm scratch:"`, `"pretrain vision:"`, `"cli pretrain vision:"` on `cuda` and `vulkan`.

---

## Order and parallelism

A0 → A1 and A2 (two agents; A2 needs nothing of A1) → A3 → A4 → A5 (the owner). Then B0 → B1 + B2 (one agent: they are
tested together) → B3 → B4 → B6 (the owner); B5 when the owner wants it. Each phase is one commit series on
`abstraction` with its "as built" section appended to this plan (what was built, filter names and counts, what is
left, the owner's GPU commands), as plans 11 to 13 did.

## What a from-scratch run is, honestly

- **The pipeline is the deliverable of A0 to A4 and B0 to B4.** It is proven at tiny scale on the CPU, as every plan's
  reference is. Whether a model of useful quality comes out is the owner's data and compute: a document reader that
  competes with a tuned pretrained model needs far more text and pages than a tuning run, and a run is one device
  (plan 6).
- **The staged recipe is the hedge.** The `text` stage is reusable (one decoder serves every later image run); `align`
  trains only the tower and the projector with the decoder's features cached where the ladder allows; `joint` is where
  the cost is, and it is a setting, not a requirement.
- **Distillation (B5) is the lever** that lets a small created model learn from a large loaded one on the owner's own
  pages, which is why it is in the plan though it is optional.

## Decisions (taken by the owner, 2026-10-14)

1. **The family is IdrakVision**: `samples/IdrakVision/Idrak.IdrakVision`, architecture
   `IdrakVisionForConditionalGeneration`, the plug-in `Idrak.IdrakVision.dll`. The owner's condition: anyone can create
   any model. So `IVisionFamily.Create` is an open contract every family plug-in may implement (the library holds the
   contract, never a family), and IdrakVision's own config exposes every shape choice as a setting: image size, patch
   size, channels, tower width, layers, heads, feed-forward size, activation, norm epsilon, pooling size, projector
   depth, width and activation, the image tokens' ids, the preprocessing, and the image attention rule.
2. **The tokenizer trainer is built** (A2). A tokenizer.json file can still be given to `idrak create --tokenizer`.
3. **Image tokens attend in bidirectional blocks** by default (Gemma 3's rule); plain causal is IdrakVision's
   `attention: "causal"` setting.
4. The first text corpus is the owner's (`idrak data text` until one is chosen).
5. **Distillation (B5) is built after B0 to B4**, before the proof run (B6).
