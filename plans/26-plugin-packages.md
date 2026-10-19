# Plan 26: model families as plug-in packages, and `Idrak.Nlp` renamed `Idrak.Text`

**Status (2026-10-19, `abstraction`):** planned 2026-10-19 from the owner's decisions in the Plan 24 session, against the
code at 63f8893 (a read of the code, not of the plans: every "Comes from" below was found in the source). Nothing is
built. No phase starts until the owner names it.

| Phase | What | State |
|---|---|---|
| P0 | `Idrak.Nlp` renamed `Idrak.Text` (project, assembly, package, namespaces, every reference) | Not started |
| P1 | The three plug-in packages: `Idrak.Text.Plugins`, `Idrak.Vision.Plugins`, `Idrak.Audio.Plugins` (skeletons, registration, the CLI) | Not started |
| P2 | The plug-ins that already live outside the library move in: Qwen-VL, LLaVA, IdrakVision, Whisper, VITS, OpenVoice, IdrakAudio | Not started |
| P3 | Gemma out of the library: Gemma 1/2/3 text and the Gemma 3 vision-language text decoder, with the Gemma 3 vision plug-in | Not started |
| P4 | The other text families out of the library: Llama, Mistral, Qwen, DeepSeek, Harmony (architectures, GGUF mappings, pre-tokenizer patterns, tool-call formats) | Not started |
| P5 | The espeak-ng phonemizer out of the library | Not started |
| P6 | The engine for encoders and encoder-decoders (contracts, attention options), with no family in it | Not started |
| P7 | BERT (`Idrak.Text.Plugins.Bert`) | Not started |
| P8 | T5 (`Idrak.Text.Plugins.T5`) | Not started |

## Goal

The library is the engine every plug-in runs on; model families are plug-ins. Each family leaves the library for its
parent's plug-in package, in a namespace of its own, so a consumer references the package of the kind of model it uses
(text, vision, audio) and registers the family it wants. What stays in the library is generic: the defaults and fallbacks
meant to work with any family. BERT and T5, which Plan 24 made loadable as tokenizers but which have no network, are
the first new families, once the engine can run encoders and encoder-decoders.

## Decisions (taken by the owner, 2026-10-19)

1. **One plug-in package per parent package, one namespace per family:** `Idrak.Text.Plugins`, `Idrak.Vision.Plugins`,
   `Idrak.Audio.Plugins`; for example `Idrak.Vision.Plugins.Qwen`, `Idrak.Audio.Plugins.Whisper`. Not one package per
   family (too many packages to maintain), not one package for every plug-in.
2. **The rule holds for every family, the text language models included:** the library keeps only default
   implementations and fallbacks.
3. **`Idrak.Nlp` is renamed `Idrak.Text`** (the parent, not only the plug-in package), in line with `Idrak.Audio`,
   `Idrak.Imaging`, `Idrak.Vision` and `Idrak.Network`.
4. **BERT and T5 head the candidate list.** They are built last (P7, P8) because they need the engine of P6.
5. **Plan first:** this file is written before any code.
6. **The plug-in tests live in `tests/Idrak.Tests`,** one filter group per family.

## Rules that hold for every phase

CLAUDE.md and [docs/optimization.md](../docs/optimization.md) apply in full. The ones that shape this plan:

- **Results unchanged, proven (rule 71).** A move changes where code lives, not what it computes: every moved family's
  existing tests pass unchanged after it, and Plan 24's BPE baseline hashes (`tokenizers/bpe-baseline.json`) hold.
- **Plug-ins use the public API only.** No plug-in package gets `InternalsVisibleTo`. Where a family in the library uses
  an internal today, the move makes a generic member public (with a reason in the API dump) or rewrites the family on
  public parts; it never grants internals.
- **Contract plus registry (decision 10).** New engine behaviour (P6) is a contract and a `SlotTable` registry in
  `Idrak.Abstraction` (or a package's `Abstractions` namespace when only that package can use it); the inventory and the
  API dump are regenerated after each public change. No CHANGELOG line, no version bump.
- **A family that is not registered fails loudly** with "not registered" and the hint to reference the plug-in package
  and call its registration; no family falls back to another.
- **Tests only through filters,** CPU here; GPU tests, apps and the full suite are the owner's. Each phase lists the
  owner's commands.
- **The fix workflow:** fix everything, update the tests, build once, run the phase's filters once with `--no-build`;
  after a merge, build and run before pushing (the owner, 2026-10-19).

## What exists (2026-10-19, 63f8893)

**Plug-ins that already live outside the library** (each references Idrak only as packages and exposes a
`…Plugin.Register()`; the CLI loads them with `-P`):

| Family | Lives | References | Registers into | Tests (groups) |
|---|---|---|---|---|
| Qwen-VL (2, 2.5) | `samples/QwenVision/Idrak.QwenVision` (1,724 lines) | Idrak, Idrak.Imaging | `VisionFamilies`, `PretrainedArchitectures` | `qwen vl:` (6); plan 16 steps 0-3, 5, 6, 8 built (its .csproj comment "skeleton" is stale) |
| Gemma 3 vision (SigLIP, projector) | `samples/Gemma3Vision/Idrak.Gemma3Vision` (1,205 lines) | Idrak, Idrak.Imaging, Idrak.Vision | `VisionFamilies` | `pretrained:` and others (18 test files); no README |
| LLaVA (CLIP tower, projector) | `tests/Idrak.PluginTests/Llava.cs` | (test project) | `VisionFamilies`, `PretrainedArchitectures` | `vision contracts`, `vision tuning contracts`, `vision tuning reference` |
| IdrakVision | `samples/IdrakVision/Idrak.IdrakVision` (819 lines) | Idrak, Idrak.Imaging, Idrak.Vision | `VisionFamilies`, `PretrainedArchitectures` | plan 19 tracks A and B (4 test files) |
| Whisper | `samples/WhisperSpeech/Idrak.WhisperSpeech` (1,379 lines) | Idrak, Idrak.Audio | `SpeechFamilies` | `audio asr`, `audio cli` |
| VITS / MMS-TTS | `samples/VitsSpeech/Idrak.VitsSpeech` (2,060 lines) | Idrak, Idrak.Audio | `SpeechSynthesisFamilies` | `audio tts`, `audio cli`, `audio voices`, `audio voice train` |
| OpenVoice | `samples/OpenVoice/Idrak.OpenVoice` (837 lines) | Idrak, Idrak.Audio | `VoiceConverters` | `audio voices` (its demo also uses VITS) |
| IdrakAudio | `samples/IdrakAudio/Idrak.IdrakAudio` (2,844 lines) | Idrak, Idrak.Audio | `AudioFamilies`, `SpeechFamilies`, `SpeechSynthesisFamilies`, `PretrainedArchitectures` | `audio language`, `audio seq2seq`, `audio train`, `audio tts train` |

**Family code inside the library:**
- `src/Idrak/Models/Architectures.cs` (`PretrainedFamilies.BuiltIns`): `LlamaForCausalLM`, `MistralForCausalLM`,
  `Qwen2ForCausalLM`, `Qwen3ForCausalLM`, `GemmaForCausalLM`, `Gemma2ForCausalLM`, `Gemma3ForCausalLM`,
  `Gemma3ForConditionalGeneration` (with its three checkpoint layouts; plan 16 step 4, moving it to its plug-in, was
  never done), `MixtralForCausalLM`, `Qwen2MoeForCausalLM`, `Qwen3MoeForCausalLM`.
- `src/Idrak/Models/GgufModel.cs` (`GgufBuiltIns`): the GGUF architecture mappings `llama`, `qwen2`, `qwen3`,
  `qwen2moe`, `qwen3moe` and the per-family pre-tokenizer patterns (Qwen, DeepSeek and others).
- `src/Idrak.Abstraction/Generation/ToolCallParsers.cs`: tool-call formats `qwen3-coder`, `mistral`, `deepseek`,
  `harmony` (the generic `json` and `pythonic` stay); the shared base `MarkedToolCallParser` is internal.
- `src/Idrak.Abstraction/Generation/RopeScalings.cs`: `llama3` (`linear`, `dynamic`, `yarn` stay).
- `src/Idrak.Audio/SpeechText/EspeakNgPhonemizer.cs`: a phonemizer that runs the outside `espeak-ng` program (also
  against rule 70).

**Not candidates:** the test families of `tests/Idrak.PluginTests` (`TinyResNetFamily`, `GridDetectorFamily`,
`OnnxSegmenterFamily`, `OnnxBackboneFamily`, `ToyFamily`, `OutsideRawCodec`, `AudioClipKind`) stay there: they prove that
an outside assembly can plug into each contract. `YoloFormat` (`src/Idrak.Data`) is an annotation format, not a model.

**The engine today runs decoder-only text models only:** `PretrainedArchitecture.Build` returns ids → logits
`[batch, time, vocabulary]`; generation is causal. There is no encoder contract (hidden states, pooling, heads), no text
encoder-decoder contract (seq2seq generation exists only in `Idrak.Audio`), no key-padding mask in `MultiHeadAttention`
(only causal, `src/Idrak/Layers/Attention.cs:135`), no position-bias hook (T5's bucketed bias; the
`RelativePositionAttention` there is VITS's windowed kind), a fixed 1/√d attention scale (T5 uses none), and no
token-type input. The building blocks exist: `LayerNorm`, `RMSNorm`, GELU, `Embedding`, learned positions,
`CrossAttention` with `EncoderMemory` and `DecodingContext.SetMemory`, `PaddedSequences`, `TextEncoder` / `IEmbedder`.

## The candidates, their package and namespace

| # | Family | Package | Namespace | Comes from | Phase |
|---|---|---|---|---|---|
| 1 | BERT | `Idrak.Text.Plugins` | `Idrak.Text.Plugins.Bert` | new | P7 |
| 2 | T5 | `Idrak.Text.Plugins` | `Idrak.Text.Plugins.T5` | new | P8 |
| 3 | Llama (GGUF `llama`, its pre-tokenizer pattern, RoPE `llama3`) | `Idrak.Text.Plugins` | `Idrak.Text.Plugins.Llama` | `src/Idrak`, `src/Idrak.Abstraction` | P4 |
| 4 | Mistral, Mixtral (tool-call `mistral`) | `Idrak.Text.Plugins` | `Idrak.Text.Plugins.Mistral` | `src/Idrak`, `src/Idrak.Abstraction` | P4 |
| 5 | Qwen2, Qwen3 and their MoE (GGUF mappings, pre-tokenizer patterns, tool-call `qwen3-coder`) | `Idrak.Text.Plugins` | `Idrak.Text.Plugins.Qwen` | `src/Idrak`, `src/Idrak.Abstraction` | P4 |
| 6 | Gemma 1, 2, 3 and the Gemma 3 vision-language text decoder | `Idrak.Text.Plugins` | `Idrak.Text.Plugins.Gemma` | `src/Idrak` | P3 |
| 7 | DeepSeek (tool-call `deepseek`, GGUF pre-tokenizer patterns) | `Idrak.Text.Plugins` | `Idrak.Text.Plugins.DeepSeek` | `src/Idrak`, `src/Idrak.Abstraction` | P4 |
| 8 | Harmony (the gpt-oss tool-call format) | `Idrak.Text.Plugins` | `Idrak.Text.Plugins.Harmony` | `src/Idrak.Abstraction` | P4 |
| 9 | Qwen-VL | `Idrak.Vision.Plugins` | `Idrak.Vision.Plugins.Qwen` | `samples/QwenVision` | P2 |
| 10 | Gemma 3 vision (SigLIP, projector) | `Idrak.Vision.Plugins` | `Idrak.Vision.Plugins.Gemma` | `samples/Gemma3Vision` | P3 |
| 11 | LLaVA (CLIP, projector) | `Idrak.Vision.Plugins` | `Idrak.Vision.Plugins.Llava` | `tests/Idrak.PluginTests/Llava.cs` | P2 |
| 12 | IdrakVision | `Idrak.Vision.Plugins` | `Idrak.Vision.Plugins.IdrakVision` | `samples/IdrakVision` | P2 |
| 13 | Whisper | `Idrak.Audio.Plugins` | `Idrak.Audio.Plugins.Whisper` | `samples/WhisperSpeech` | P2 |
| 14 | VITS / MMS-TTS | `Idrak.Audio.Plugins` | `Idrak.Audio.Plugins.Vits` | `samples/VitsSpeech` | P2 |
| 15 | OpenVoice | `Idrak.Audio.Plugins` | `Idrak.Audio.Plugins.OpenVoice` | `samples/OpenVoice` | P2 |
| 16 | IdrakAudio | `Idrak.Audio.Plugins` | `Idrak.Audio.Plugins.IdrakAudio` | `samples/IdrakAudio` | P2 |
| 17 | espeak-ng phonemizer | `Idrak.Audio.Plugins` | `Idrak.Audio.Plugins.EspeakNg` | `src/Idrak.Audio` | P5 |

Text and vision both have a `.Gemma` and a `.Qwen`: one per package, so the names need no version number.

## What stays in the library (defaults and fallbacks)

- **Text models:** the generic decoder (`DecoderSpec`, its builder and knobs: norm offset, q/k norm, QKV bias, sliding
  window, softcap, experts), the registries (`PretrainedArchitectures`, `GgufArchitectures`, `GgufPreTokenizers`), the
  safetensors and GGUF readers, generation, the KV cache, fine-tuning.
- **Tokenizers:** BPE, Unigram, WordPiece, their readers (`tokenizer.json`, `tokenizer.model`, `vocab.txt`) and the GGUF
  tokenizer formats (`gpt2`, `llama`, `t5`, `bert` name formats there, not model families).
- **Chat:** Jinja chat templates; the `json` and `pythonic` tool-call formats as the fallback.
- **Generic algorithms:** RoPE `linear`, `dynamic`, `yarn`; the HiFi-GAN layers; log-mel features; CTC decoding; any layer
  two plug-ins share (a CLIP or SigLIP tower moves to the library only when a second family needs it).
- **The CLI** is an app: it references the three plug-in packages and registers them, so `idrak chat`, `run`,
  `transcribe` and `speak` keep working with nothing added; `-P` stays for outside plug-ins.

---

## Phase P0: `Idrak.Nlp` renamed `Idrak.Text`

**Why first:** every later phase touches files that name `Idrak.Nlp`; the rename is mechanical and is done once, alone.

**What changes:**
- `src/Idrak.Nlp/` → `src/Idrak.Text/`: project file, assembly name, package id, root namespace; the namespaces
  `Idrak.Nlp`, `Idrak.Nlp.Retrieval`, `Idrak.Nlp.Abstractions` and the rest → `Idrak.Text…`.
- Every reference (measured: 59 files in `tests/Idrak.Tests`, 55 in the project itself, 32 in `src/Idrak.Cli`, 26 in
  `samples/`, 6 in `src/Idrak.Abstraction`, 5 each in `src/Idrak.Audio` and `src/Idrak`, 4 in
  `src/Idrak.Abstraction.Testing`, 3 in `tests/Idrak.PluginTests`), the solution file, docs and READMEs.
- The places a search-and-replace misses, each checked by hand:
  - `src/Idrak.Abstraction/LibraryDefaults.cs`: the package list (`("Idrak.Nlp", "Idrak.Nlp")`) and the
    `DynamicDependency` on `Idrak.Nlp.LibraryRegistrations` (missed, the package's built-ins stop registering, silently);
  - `src/Idrak/Idrak.csproj`: `InternalsVisibleTo Include="Idrak.Nlp"`;
  - `tests/Idrak.Tests/data/internals-justified.txt` (users named `Idrak.Nlp`), `shared-contracts.txt`,
    `abstraction-allow-list.txt`;
  - `api/Idrak.Nlp.txt` → `api/Idrak.Text.txt` (regenerated), the abstraction inventory (regenerated);
  - `tests/Idrak.Tests/NamingTests.cs` (`naming: no file in the repository uses the library's former name …`): add
    `Idrak.Nlp` to its former names, so nothing brings it back;
  - the README's package table (`README.md:29`) and its history note (`README.md:278`, kept as history).
- `Idrak.Text` holds model-level text work; low-level text stays where it is (`Idrak.Abstraction.Text`: Unicode
  normalization; core `Idrak`: tokenizers). Its types avoid names .NET already uses in `System.Text`.
- `CLAUDE.md` names packages in its examples; it is the owner's file: the new names are proposed to the owner, not
  edited.

**Done when:** no file outside history notes names `Idrak.Nlp` (the naming test), the inventory, internals, API and
naming guards pass, and every group that touched `Idrak.Nlp` passes.

**Tests (CPU here):** one build; then `naming:|abstraction inventory|public API|fine-tuning:|retrieval:|answer texts:|
json repair:|conversation tuning:|cli health|cli arabic:|cli polish|cli run:|cli train data:|audio language|distill` and
every group whose test file names `Idrak.Nlp` (listed in the phase's as-built note). **Owner:** the full suite once, as
the rename reaches every package.

## Phase P1: the three plug-in packages

- `src/Idrak.Text.Plugins` (references `Idrak.Text`, `Idrak`), `src/Idrak.Vision.Plugins` (references `Idrak.Vision`,
  `Idrak.Imaging`, `Idrak`), `src/Idrak.Audio.Plugins` (references `Idrak.Audio`, `Idrak`). None gets internals.
- Each family namespace has a public static `…Plugin` class with `Register()` (idempotent: calling it twice registers
  once) and `Unregister()`; each package has one root class (`TextPlugins`, `VisionPlugins`, `AudioPlugins`) with
  `RegisterAll()`, and a public static `RegisterIdrakPlugin()` so `-P` loads the package as it loads a sample today
  (`src/Idrak.Cli/CommandContext.cs:314`).
- A family that needs another package's family (Gemma 3 vision needs the Gemma text decoder) references that plug-in
  package and calls its `Register()`; the dependency is one way, vision and audio on text, never back.
- The CLI references the three packages and calls `RegisterAll()` at startup, before `-P` plug-ins (so an outside plug-in
  can still override a family through the registries' policies).
- The "not registered" messages of `PretrainedArchitectures`, `VisionFamilies`, `SpeechFamilies`,
  `SpeechSynthesisFamilies`, `VoiceConverters`, `GgufArchitectures` and the tool-call formats name the plug-in package to
  reference.
- **The plug-in tests live in `tests/Idrak.Tests`** (the owner, 2026-10-19): one filter group per family
  (`plugins: qwen vl`, `plugins: whisper`, …), in the one runner, as the samples' tests are today; `Idrak.PluginTests`
  keeps only its test families that prove the contracts from outside.

**Done when:** the three packages build empty, the registration contract suite (`Register` twice, `Unregister`,
`RegisterAll`, `-P` on the package) passes, and the inventory and API dump list them.

## Phase P2: the plug-ins that already live outside the library

- **Audio (one agent):** Whisper, VITS, OpenVoice, IdrakAudio from `samples/` into `Idrak.Audio.Plugins`, code moved as it
  is, namespaces changed. The scripts that use them (`samples/WhisperSpeech/transcribe.cs`, `samples/OpenVoice/voice.cs`)
  stay in `samples/` as apps referencing the package.
- **Vision (one agent):** Qwen-VL and IdrakVision from `samples/`, LLaVA from `tests/Idrak.PluginTests/Llava.cs`, into
  `Idrak.Vision.Plugins`. `Idrak.PluginTests` keeps its test families; LLaVA's place there (proving the vision contracts
  from outside) is taken by the package itself, which has no internals either.
- The tests reference the packages instead of the sample projects; the `-P Idrak.VitsSpeech.dll`-style examples in the CLI
  help and READMEs become the built-in registrations (or `-P Idrak.Audio.Plugins.dll` where an app opts out).
- Plans 16, 19 and 21's paths are updated in their "as built" notes.

**Done when:** every listed group of "What exists" passes unchanged (same counts), and `samples/` holds only apps.

## Phase P3: Gemma

- Gemma 1, 2, 3 (`GemmaForCausalLM`, `Gemma2ForCausalLM`, `Gemma3ForCausalLM`) and the text decoder of
  `Gemma3ForConditionalGeneration` with its three checkpoint layouts move from `src/Idrak/Models/Architectures.cs` to
  `Idrak.Text.Plugins.Gemma` (plan 16 step 4, done here).
- Gemma 3 vision (SigLIP, projector) from `samples/Gemma3Vision` to `Idrak.Vision.Plugins.Gemma`, which registers the text
  decoder through `Idrak.Text.Plugins.Gemma`.
- The internal helpers they use (`GemmaSpec`, `Gemma3Spec`, `CommonSpec`, the tensor-name maps) either move with them or,
  where generic, become public helpers of the decoder (named for what they do, not for a family).

**Done when:** `pretrained:`, the Gemma 3 vision groups and the GGUF groups pass unchanged; a library user without the
package gets "not registered" naming `Idrak.Text.Plugins`.

## Phase P4: the other text families

- Llama, Mistral and Mixtral, Qwen2/Qwen3 and their MoE, from `PretrainedFamilies.BuiltIns`; their GGUF mappings and
  pre-tokenizer patterns from `GgufBuiltIns` (registered through the public `GgufArchitectures.Register` and
  `GgufPreTokenizers.Register`); DeepSeek's patterns.
- The tool-call formats `qwen3-coder`, `mistral`, `deepseek`, `harmony` move to their families; `MarkedToolCallParser`
  (the shared base, internal today) becomes a public generic base, since four plug-ins need it.
- RoPE `llama3` moves to `Idrak.Text.Plugins.Llama`.
- `LlamaStyle` (a public helper that reads the common decoder keys, named after a family) is renamed for what it does;
  the old name is kept one release as obsolete, as the naming test allows.
- Core keeps an empty family table; `PretrainedModel.Load` of a model whose architecture nobody registered fails with the
  hint.

**Done when:** `pretrained:`, `gguf:`, `decoder:`, `rope scaling:`, `chat templates:`, `tool calls`, `fine-tuning:`,
`cli run:`, `cli serve:` and Plan 24's BPE baseline pass unchanged through the plug-ins, and `idrak chat` loads a model
with nothing added.

## Phase P5: espeak-ng

- `EspeakNgPhonemizer` moves to `Idrak.Audio.Plugins.EspeakNg`, with its `IDRAK_*` variable. The library's
  `TextPhonemizers` keeps no default that runs an outside program (rule 70); a family that needs espeak-ng registers it.

**Done when:** the speech groups pass unchanged and no library file starts an outside program for text.

## Phase P6: the engine for encoders and encoder-decoders

Generic, no family in it; each part is a contract and a registry or a setting:
- **An encoder contract and registry** (`PretrainedEncoders`): a configuration and checkpoint → a network from ids,
  attention mask and token types to hidden states `[batch, time, width]`; poolers (first token, mean, max) and heads
  (classification, masked language model) as registered parts. `TextEncoder` (`Idrak.Text`) takes such an encoder.
- **A text encoder-decoder contract** (`PretrainedSeq2Seq`): encode once, then decode with cross-attention to the
  encoder's output, through `CrossAttention`, `EncoderMemory` and `DecodingContext.SetMemory` (as `Idrak.Audio`'s
  seq2seq does); generation, the CLI's `run` and the server take it.
- **Attention options:** a key-padding mask in `MultiHeadAttention` (through `PaddedSequences`), a position-bias provider
  (a contract: per layer or shared, computed once per length and cached), the score scale as a setting (default 1/√d).
- **Token types:** an optional type-id input and embedding.
- Each with its contract suite and a toy family registered from `Idrak.PluginTests` (an encoder, a seq2seq model).

**Done when:** the suites pass on the CPU, the toy families run batched with padding equal to one by one, and existing
decoder results are unchanged (`pretrained:`, `decoder:`, `sampler:`).

## Phase P7: BERT (`Idrak.Text.Plugins.Bert`)

- `BertModel` on P6's encoder contract: word, position and token-type embeddings, post-norm blocks with GELU, the
  pooler; `BertForMaskedLM`, `BertForSequenceClassification`, and sentence embeddings (mean or first-token pooling) through
  `TextEncoder`. The tokenizer is Plan 24's WordPiece (`tokenizer.json` or `vocab.txt`).
- Fixtures: a tiny random BERT and its outputs from transformers (pinned, a script under `tools/`), committed small;
  real models (`google-bert/bert-base-uncased`, `aubmindlab/bert-base-arabertv02`) are the owner's run.

**Done when:** hidden states, pooled vectors and logits equal transformers' within float tolerance, batched with padding
equal to one by one.

## Phase P8: T5 (`Idrak.Text.Plugins.T5`)

- `T5ForConditionalGeneration` on P6's encoder-decoder contract: RMS norms, the bucketed relative position bias of the
  first layer shared by the rest, no score scaling, ReLU (T5 v1.0) or gated GELU (v1.1, mT5) feed-forward, tied or untied
  head. The tokenizer is Plan 24's Unigram (`tokenizer.json` or `spiece.model`).
- Fixtures as P7's; real models (`google-t5/t5-small`, `google/mt5-small`) are the owner's run.

**Done when:** encoder states and greedy generations equal transformers' on the fixtures, and the KV-cached decode equals
the uncached one.

## Order and parallelism

At most two agents at a time (CLAUDE.md), each in its own worktree; a merge is built and its filters run before the push.
The longest chain is P0 → P6 → P7 / P8, so the engine (lane B) starts right after the rename, and the moves (lane A) run
beside it (the owner, 2026-10-19):

| Step | Lane A: moves | Lane B: engine and new families |
|---|---|---|
| 1 | P0, the rename (alone: it touches every package) | none |
| 2 | P1, the three package skeletons | P6, the engine, starts |
| 3 | P2, audio (Whisper, VITS, OpenVoice, IdrakAudio) | P6, continued |
| 4 | P2, vision (Qwen-VL, LLaVA, IdrakVision) | P6, continued |
| 5 | P3, Gemma | P7, BERT (after P6) |
| 6 | P4, Llama, Mistral, Qwen, DeepSeek, Harmony | P8, T5 |
| 7 | P5, espeak-ng | none |

Why the lanes do not collide:
- P1 and P2 add packages and move sample code; P6 works in layers, contracts and generation and touches no family.
- P3 and P4 both edit `src/Idrak/Models/Architectures.cs`, so they share lane A, one after the other.
- P7 and P8 add to `Idrak.Text.Plugins`; P3 and P4 remove from core and add other namespaces there: separate files.
- P5 can take any free slot after P1.
- Files both lanes touch (`api/*.txt`, the abstraction inventory, the test list in `tests/Idrak.Tests/Program.cs`,
  `src/Idrak.Cli/Shared/EnvironmentVariables.cs`) are merged one lane at a time; the generated dumps are regenerated,
  never merged by hand.

A step is a dependency order, not a barrier: when a lane finishes, it takes its next step without waiting for the other.
No step starts until the owner has asked for this plan to go on.

## The owner's runs

Each phase's as-built note gives the exact PowerShell list (from `D:\Projects\Idrak`): one
`dotnet build -c Release tests\Idrak.Tests`, the phase's CPU filters, then its CUDA filters with `--no-build`. P0 and P4
also ask for the full suite, because they reach every package. P7 and P8 add the real models' references.

## Risks

- **Internals in the moved text families.** The families in `src/Idrak` use internal helpers; each must become a public,
  generic member or be rewritten on public parts. The API dump shows every member opened.
- **The API breaks for consumers:** namespaces change (`Idrak.Nlp` → `Idrak.Text`, families → `…Plugins.*`), and a
  library user who loaded a Llama or Qwen model with nothing referenced must now reference `Idrak.Text.Plugins`. The
  "not registered" message names it. No version bump (CLAUDE.md); the owner decides when to release.
- **Silent loss of registrations** in the rename (the library defaults list, `DynamicDependency`): checked by the
  inventory test and by `pretrained:` in P0.
- **Plug-ins that depend on each other** (vision on text): one way only; a cycle is refused at review.
- **Encoder and seq2seq engine scope (P6)** is the largest new work; BERT and T5 wait for it.

## Runs

| Date | Phase | Machine | What | Result |
|---|---|---|---|---|
