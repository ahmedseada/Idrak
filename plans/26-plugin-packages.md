# Plan 26: model families as plug-in packages, and `Idrak.Nlp` renamed `Idrak.Text`

**Status (2026-10-20, `abstraction`):** planned 2026-10-19 from the owner's decisions in the Plan 24 session, against the
code at 63f8893, and widened 2026-10-20 to every reusable piece outside the library (a read of the code of every
package, sample and test project, not of the plans: every "Comes from" below was found in the source). Nothing is
built. No phase starts until the owner names it.

| Phase | What | Lane | State |
|---|---|---|---|
| P0 | `Idrak.Nlp` renamed `Idrak.Text` (project, assembly, package, namespaces, every reference) | A, alone | Done 2026-10-20; CPU pass |
| P1 | The three plug-in packages: `Idrak.Text.Plugins`, `Idrak.Vision.Plugins`, `Idrak.Audio.Plugins` (skeletons, registration, the CLI) | A | Done 2026-10-20; CPU pass (2 leftovers, see its note) |
| P2 | Generic components into the library, core part (contracts first): tensor store, timing probe, character tokenizer, progress, keyword index, network recipes, JSON numbers | A | Done 2026-10-20; CPU pass |
| P3 | Generic components into the library, vision part (contracts first): image-block rule, noise augmentation, PGM, line segmentation, text matching, reading order, the line-recognition task | A | Done 2026-10-20; CPU pass |
| P4 | Audio plug-ins move in: Whisper, VITS (with uroman), OpenVoice, IdrakAudio, espeak-ng | A | Done 2026-10-20; CPU pass |
| P5 | Vision plug-ins move in: Qwen-VL, LLaVA, IdrakVision, the CRNN line recognizer | A | Done 2026-10-20; CPU pass |
| P6 | Gemma out of the library (text, the vision-language text decoder, the Gemma 3 vision plug-in) | A | Done 2026-10-20; CPU pass |
| P7 | The other text families out of the library: Llama, Mistral, Qwen, DeepSeek, Harmony | A | Done 2026-10-20; CPU pass |
| P8 | The engine for encoders and encoder-decoders (contracts, attention options), with no family in it | B | Done 2026-10-20 (core); CLI `run`/server for seq2seq and a beam setting go with P10 |
| P9 | BERT (`Idrak.Text.Plugins.Bert`) | B | Done 2026-10-20; CPU pass (real models: the owner) |
| P10 | T5 (`Idrak.Text.Plugins.T5`) | B | Done 2026-10-20; CPU pass (`idrak serve` not yet) |

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
4. **BERT and T5 head the candidate list.** They wait for the engine (P8), which starts right after the rename.
5. **Plan first:** this file is written before any code.
6. **The plug-in tests live in `tests/Idrak.Tests`,** one filter group per family.
7. **Not only model families (2026-10-20):** every plug-in or registration that lives outside the library and another
   project could use is a candidate; moving it cuts the sample or test that held it.
8. **Contract first (2026-10-20):** a candidate with no contract gets a contract and a registry first, then the
   library's default implementation registered into it.
9. **No own implementations (2026-10-20):** no plug-in or sample writes its own implementation of something the library
   does, unless it exists to test the plug-in and override abilities.
10. **At most two agents in parallel** (CLAUDE.md), in two lanes (see "Order and parallelism").
11. **Plug-in packages consume the library's contracts, they do not own them (2026-10-20):** the inventory counts
    `Idrak.Text.Plugins`, `Idrak.Vision.Plugins` and `Idrak.Audio.Plugins` as apps, not as users, when it decides where a
    contract lives.

## Rules that hold for every phase

CLAUDE.md and [docs/optimization.md](../docs/optimization.md) apply in full. The ones that shape this plan:

- **Results unchanged, proven (rule 71).** A move changes where code lives, not what it computes: every moved family's
  existing tests pass unchanged after it, and Plan 24's BPE baseline hashes (`tokenizers/bpe-baseline.json`) hold.
- **Plug-ins use the public API only.** No plug-in package gets `InternalsVisibleTo`. Where a family in the library uses
  an internal today, the move makes a generic member public (with a reason in the API dump) or rewrites the family on
  public parts; it never grants internals.
- **Contract plus registry (decision 10).** New behaviour (P2, P3, P8) is a contract and a `SlotTable` registry in
  `Idrak.Abstraction` (or a package's `Abstractions` namespace when only that package can use it); the inventory and the
  API dump are regenerated after each public change. No CHANGELOG line, no version bump.
- **Contract first, then the default (decision 8);** a sample or plug-in then uses the library's, never its own copy
  (decision 9). The proof families of `tests/Idrak.PluginTests`, the fakes of `tests/Idrak.Tests` and the Override
  sample are the only own implementations: they exist to test overriding.
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

**Generic components outside the library (the 2026-10-20 survey).** Copies and missing defaults found in samples and the
command line; each is in "Generic components" below with its contract and phase:
- `RenamedTensorStore` twice (`samples/QwenVision`, `samples/Gemma3Vision`) and `OpenVoiceTensors` (renames, folds
  weight-norm) — one `ITensorStore` behaviour, three copies.
- `ImageBlockRule` twice (`samples/IdrakVision`, `samples/Gemma3Vision`) — the library registers only `causal` and `grid`.
- `samples/Idrak.Samples.ArabicOcr/LineNoise.cs` (its doc: the library's augmentations have no noise),
  `LineSegmenter.cs` (page → lines by projection, on the library's `Foreground`), `Text.cs` (`TextAlignment`: Sellers'
  semi-global edit distance; `ReadingOrder`: visual column order), `Recognizer.cs` (a conv + BiLSTM + CTC line
  recognizer; the image-family contract has no line-recognition task), `LiveLine.cs` (a live progress line).
- `src/Idrak.Cli/Shared/Progress.cs` (`ProgressLine`, internal to the tool) and
  `samples/Idrak.Samples.LegalOcr/StartupLoad.cs` (its own progress line); `FiniteNumbers.cs` (NaN → null in JSON).
- `samples/Idrak.Samples.Chat` `ProfileRecorder` (time per operation and shape, per layer type: the library's
  `operations` probe counts calls only).
- `samples/Idrak.Samples.ReRanker/Collection.cs` `Bm25` (the library has `Bm25Index`, a concrete class with no
  registry); `samples/Idrak.Samples.GptTraining` `RuneTokenizer` (the library's `CharTokenizer` works per UTF-16 unit);
  `samples/Idrak.Samples.Ocr/Pgm.cs` (the library has the Netpbm codec and encoder); `samples/Shared/Gpt/CharGpt.cs`
  (builds its own causal transformer; the library's `Architectures.Gpt` is a static recipe, not a registry).

**App-specific, stays in its sample:** LegalOcr's services (reader, tuning, synthetic data, evaluation pages, the
preprocessing editor on the library's `IImageTransform.Parameters`, the adapter catalogue on `TuningManifest`), GptApi's
service, the Ocr sample's toy font, renderer and glyph segmenter, Rag's `TownTokenizer` (it shows an app's own
tokenizer), the synthetic data of the house, report, ReRanker, chat-corpus and Rag samples. VITS's `vits-frames`
features reproduce VITS's own spectrogram and stay in its plug-in.

**Not candidates (built to test overriding):** the test families of `tests/Idrak.PluginTests` (`TinyResNetFamily`, `GridDetectorFamily`,
`OnnxSegmenterFamily`, `OnnxBackboneFamily`, `ToyFamily`, `OutsideRawCodec`, `AudioClipKind`) stay there: they prove that
an outside assembly can plug into each contract. `YoloFormat` (`src/Idrak.Data`) is an annotation format, not a model. The registrations
of `tests/Idrak.Tests` are fakes for override and failure tests (`SmallFamily`, `TinyCodec`, `FakeAudioFamily`,
`FailingEncoder`, `BrokenImageKind`, "imaginary", "broken", "testllama"). `Idrak.Network`, `Idrak.Cluster`,
`Idrak.Gpu`, `Idrak.Imaging`, `Idrak.Data`, `Idrak.Mcp`, `Idrak.AspNetCore` and `Idrak.Onnx.Runtime` hold no
plug-in candidates: their named built-ins are formats (safetensors, GGUF, COCO, VOC, YOLO, ShareGPT, JPEG, PNG, WAV,
FLAC, MP3), protocols (OpenAI-compatible endpoints, MCP), services (the Hugging Face Hub) and device or link kinds
(CUDA, HIP, Vulkan, `nvlink` — a vendor-neutral peer copy despite its name — InfiniBand, RoCE), which stay as defaults.

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
| 1 | BERT | `Idrak.Text.Plugins` | `Idrak.Text.Plugins.Bert` | new | P9 |
| 2 | T5 | `Idrak.Text.Plugins` | `Idrak.Text.Plugins.T5` | new | P10 |
| 3 | Llama (GGUF `llama`, its pre-tokenizer names `llama3`, `llama-bpe`, `llama-v3`, `smaug-bpe`, `falcon3`, `pixtral`, RoPE `llama3`) | `Idrak.Text.Plugins` | `Idrak.Text.Plugins.Llama` | `src/Idrak`, `src/Idrak.Abstraction` | P7 |
| 4 | Mistral, Mixtral (tool-call `mistral`, pre-tokenizer `tekken`) | `Idrak.Text.Plugins` | `Idrak.Text.Plugins.Mistral` | `src/Idrak`, `src/Idrak.Abstraction` | P7 |
| 5 | Qwen2, Qwen3 and their MoE (GGUF mappings, pre-tokenizer names `qwen2`, `qwen35`, `megrez`, `hunyuan`, tool-call `qwen3-coder`) | `Idrak.Text.Plugins` | `Idrak.Text.Plugins.Qwen` | `src/Idrak`, `src/Idrak.Abstraction` | P7 |
| 6 | Gemma 1, 2, 3 and the Gemma 3 vision-language text decoder | `Idrak.Text.Plugins` | `Idrak.Text.Plugins.Gemma` | `src/Idrak` | P6 |
| 7 | DeepSeek (tool-call `deepseek`, pre-tokenizer `deepseek-r1-qwen`) | `Idrak.Text.Plugins` | `Idrak.Text.Plugins.DeepSeek` | `src/Idrak`, `src/Idrak.Abstraction` | P7 |
| 8 | Harmony (the gpt-oss tool-call format) | `Idrak.Text.Plugins` | `Idrak.Text.Plugins.Harmony` | `src/Idrak.Abstraction` | P7 |
| 9 | Qwen-VL | `Idrak.Vision.Plugins` | `Idrak.Vision.Plugins.Qwen` | `samples/QwenVision` | P5 |
| 10 | Gemma 3 vision (SigLIP, projector) | `Idrak.Vision.Plugins` | `Idrak.Vision.Plugins.Gemma` | `samples/Gemma3Vision` | P6 |
| 11 | LLaVA (CLIP, projector) | `Idrak.Vision.Plugins` | `Idrak.Vision.Plugins.Llava` | `tests/Idrak.PluginTests/Llava.cs` | P5 |
| 12 | IdrakVision | `Idrak.Vision.Plugins` | `Idrak.Vision.Plugins.IdrakVision` | `samples/IdrakVision` | P5 |
| 13 | CRNN line recognizer (conv + BiLSTM + CTC) | `Idrak.Vision.Plugins` | `Idrak.Vision.Plugins.Crnn` | `samples/Idrak.Samples.ArabicOcr/Recognizer.cs` | P5 |
| 14 | Whisper | `Idrak.Audio.Plugins` | `Idrak.Audio.Plugins.Whisper` | `samples/WhisperSpeech` | P4 |
| 15 | VITS / MMS-TTS, with the uroman Arabic romanizer | `Idrak.Audio.Plugins` | `Idrak.Audio.Plugins.Vits` | `samples/VitsSpeech`, `src/Idrak.Audio/SpeechText/UromanArabicRomanizer.cs` | P4 |
| 16 | OpenVoice | `Idrak.Audio.Plugins` | `Idrak.Audio.Plugins.OpenVoice` | `samples/OpenVoice` | P4 |
| 17 | IdrakAudio | `Idrak.Audio.Plugins` | `Idrak.Audio.Plugins.IdrakAudio` | `samples/IdrakAudio` | P4 |
| 18 | espeak-ng phonemizer (runs an outside program) | `Idrak.Audio.Plugins` | `Idrak.Audio.Plugins.EspeakNg` | `src/Idrak.Audio/SpeechText` | P4 |

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

## Generic components into the library (contracts first)

Each moves into the library behind a contract (new ones marked), its default registered there, and the samples and
plug-ins that held a copy use it (decision 9):

| # | Component | Copies or need today | Contract | Library default | Cut from | Phase |
|---|---|---|---|---|---|---|
| G1 | A tensor store that renames tensors and folds weight-norm | Qwen-VL, Gemma 3 (twice), OpenVoice | `ITensorStore` (exists) | a public renaming store | the three plug-ins | P2 |
| G2 | Time per operation (by shape) and per layer type | Chat's `ProfileRecorder` | `ProfileProbes` (exists) | probe `timings` | Chat | P2 |
| G3 | A tokenizer of whole code points | GptTraining's `RuneTokenizer` | `ITokenizer` (exists) | `CharTokenizer` reads code points (surrogate pairs whole), an unknown-character setting | GptTraining | P2 |
| G4 | A live progress line | `ProgressLine` (the CLI, internal), ArabicOcr's `LiveLine`, LegalOcr's `StartupLoad` | **new**: `IProgressReporter`, registry `ProgressReporters` (`Idrak.Abstraction.Diagnostics`) | `console` (redrawn in place: what, how far, speed, time left), `plain` (a line a minute when redirected), `json`, `quiet`; `IDRAK_PROGRESS` | the CLI's copy, ArabicOcr, LegalOcr | P2 |
| G5 | A keyword index | ReRanker's `Bm25`; `Bm25Index` is a concrete class | **new**: `IKeywordIndex`, registry `KeywordIndexes` (`Idrak.Text.Abstractions`) | `bm25`; retrieval and `idrak rag` use the registry | ReRanker | P2 |
| G6 | Ready-made network recipes | Shared `CharGpt` builds its own causal transformer; `Architectures` is a static class | **new**: registry `NetworkRecipes` | the existing recipes (`gpt`, …) registered by name | `samples/Shared/Gpt` (GptApi, GptTraining, TextGeneration) | P2 |
| G7 | Non-finite numbers in JSON (NaN, infinity → null) | LegalOcr's `FiniteNumbers`; the profile writers privately | a public helper (a `JsonConverter`), not a registry: one fixed meaning | the helper, used by the profile writers too | LegalOcr | P2 |
| G8 | Image-block attention | IdrakVision, Gemma 3 (`ImageBlockRule` twice) | `ImageAttentionRules` (exists) | rule `image-blocks` beside `causal` and `grid` | the two plug-ins | P3 |
| G9 | Pixel noise | ArabicOcr's `LineNoise` | `Augmentations` (exists) | augmentation `noise` (`std`, `p`) | ArabicOcr | P3 |
| G10 | PGM read and write | the Ocr sample's `Pgm` | `ImageCodecs` (exists, Netpbm) | the library's codec and encoder | Ocr | P3 |
| G11 | Page → text lines | ArabicOcr's `LineSegmenter` | **new**: `ITextLineSegmenter`, registry `TextLineSegmenters` (`Idrak.Vision`) | `projection` (binarize, deskew, row bands, marks joined to their line) | ArabicOcr | P3 |
| G12 | Approximate text matching | ArabicOcr's `TextAlignment` | **new**: `ITextMatcher`, registry `TextMatchers` (`Idrak.Text.Abstractions`) | `semi-global` (Sellers' edit distance, word-boundary ends) | ArabicOcr | P3 |
| G13 | A line's visual (column) order | ArabicOcr's `ReadingOrder` | **new**: registry `ReadingOrders` over the library's `Bidi` (`Idrak.Imaging`) | `bidi` (UAX #9 levels) | ArabicOcr | P3 |
| G14 | Line recognition as an image task | ArabicOcr's `Recognizer` has no family contract to plug into | **extended**: `IImageModelFamily` gains a line-recognition task (image → label sequence, CTC decoding through `CtcDecoders`) | the task; the CRNN family itself is plug-in #13 | ArabicOcr | P3 |

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

**P0 as built (2026-10-20).** As planned, with these specifics:
- `git mv` of `src/Idrak.Nlp` to `src/Idrak.Text`, its project file and `api/Idrak.Nlp.txt` (regenerated); `Idrak.Nlp`
  replaced by `Idrak.Text` in 213 files (code, projects, the solution, `build/IdrakFromSource.targets`' package list,
  `LibraryDefaults`' package list and `DynamicDependency`, `InternalsVisibleTo`, `internals-justified.txt`, samples,
  tools' comments, READMEs); the plans and `CHANGELOG.md` keep their history. One short reference
  (`typeof(Nlp.ChatTranscriptEncoder)` in `src/Idrak.Cli/Commands/Health/Machine.cs`) and two comments by hand.
- The README's history note says `Idrak.Nlp` (renamed `Idrak.Text` in 2026-10).
- **The naming check:** the existing former-names test scans every file, plans and changelog included, so a renamed
  package (whose history the plans must keep) gets its own test: `naming: no code, project, build file, API dump or test
  list names a renamed package` (code and data kinds; Markdown and `plans/` skipped).
- No `Idrak.Text` namespace existed; `Idrak.Abstraction.Text` (Unicode normalization) stays as it is.
- **Run (CPU):** one build, 15 changed sample projects built; the API dump and inventory regenerated; `naming:`,
  `public API`, `abstraction inventory`, `override loop:`, `plugins:`, `conformance kit:`, `retrieval:`,
  `cli retrieval:`, `fine-tuning:`, `chat templates:`, `conversation tuning:`, `answer texts:`, `json repair:`,
  `random text:`, `synthetic data:`, `coding agent:`, `tool contracts:`, `distillation:`, `pretrain:`,
  `audio language:`, `cli serve:`, `cli health`, `cli polish`, `cli arabic:`, `evaluation:`, `model host:`: 158 passed,
  1 failed (the new naming test matched its own description); after one rebuild `naming:` 3 of 3. Sample tests:
  Override 4, ArabicOcr 14, LegalOcr 12, all passed.
- **Owner:** the full suite once (the rename reaches every package), commands in "The owner's runs".

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

**P1 as built (2026-10-20).** As planned, with these specifics:
- **The plug-in contract** (`Idrak.Abstraction.Plugins`, since the three packages use it): `IIdrakPlugin` (`Name`,
  `IsRegistered`, `Register()` once however often called, `Unregister()` restoring the registries); `IdrakPlugin`, the
  base a family derives from (`RegisterParts()` with `Part(register, unregister)` per part; registered once under a lock;
  a failing part takes back the ones before it; `Requires(plugin)` registers a needed plug-in only when missing and
  removes it only if it added it, for vision on text); `IdrakPlugins`, the registry (`SlotTable`, unguarded: a second
  `Register` does nothing, one under a taken name replaces the earlier one, restored if the new one fails; `RegisterAll`,
  `UnregisterAll`, `Names`, `Find`, `Get` whose "not registered" names the three packages).
- **The packages** `src/Idrak.Text.Plugins`, `src/Idrak.Vision.Plugins`, `src/Idrak.Audio.Plugins`, each with a README
  and a root class (`TextPlugins`, `VisionPlugins`, `AudioPlugins`: `Package`, `All` — empty until P4 —, `RegisterAll()`,
  `UnregisterAll()`, `RegisterIdrakPlugin()`); none has internals access either way (a test). Listed in `Idrak.slnx`,
  `build/IdrakFromSource.targets`, the READMEs, the API guard and the inventory; not in `LibraryDefaults` (a test).
- **The CLI** references them and registers text, vision, audio before every command (`CommandLine.PluginPackages`),
  before `plugins list`'s snapshot (so they count as the tool's own) and before `-P` plug-ins (which can still override).
- **Messages:** `VisionFamilies`, `SpeechFamilies`, `SpeechSynthesisFamilies`, `VoiceConverters`, `GgufArchitectures`,
  `ToolCallFormats` and `GgufModel`'s architecture message name the package to reference.
- **The inventory's registry rule:** a type counts as a registry only when its static `Register` takes arguments, so a
  family's parameterless `…Plugin.Register()` is not one; no existing registry dropped out.
- **Tests:** `plugins: packages` (5: load and register twice, the contract with threads, a failing part, `Requires` and
  replacement, the CLI's order and `-P` on each package, no internals, every message); the message checks of five
  existing test files extended.
- **Leftovers:** `PretrainedArchitectures`' message (`src/Idrak/Models/Abstractions/PretrainedArchitectures.cs:107`)
  and its check in `QwenVisionTests.cs:321` wait for P8's merge (P8 owns that file), then name `Idrak.Text.Plugins` /
  `Idrak.Vision.Plugins`; `ImageModelFamilies`' message goes with P3. **A setting for the CLI's startup registration**
  (CLAUDE.md: nothing fixed) goes with P2: `IDRAK_PLUGIN_PACKAGES` (all by default; `none` or a list).
  `PretrainedArchitectures`' message: done in P8's merge (it names `Idrak.Text.Plugins` and `Idrak.Vision.Plugins`;
  `qwen vl:` checks it).
- **Run (CPU, the agent):** 126 passed, 1 failed (a GGUF message in `GgufModel.cs`), then `plugins:` 23 of 23.
  **Merged (2026-10-20, on c023893):** one build; `plugins:`, `public API`, `abstraction inventory`, `naming:`,
  `override loop:`, `cli health`, `cli polish`, `cli arabic:`, `cli run:`, `pretrained:`, `gguf:`, `vision contracts`,
  `audio asr`, `audio tts`, `audio voices`, `tool calls`, `chat templates:`, `qwen vl`: 133 passed, 0 failed.

## Phase P2: generic components, core part (G1-G7)

- New contracts first (G4 progress, G5 keyword index, G6 network recipes), each with its `SlotTable` registry, contract
  suite, "not registered" message and inventory line; then the defaults; then the existing code moved onto them (the
  CLI's `ProgressLine` becomes the `console` reporter; retrieval reads `KeywordIndexes`).
- G1, G2, G3, G7 as the table says.
- The samples drop their copies in the same phase: Chat, GptTraining, ReRanker, `samples/Shared/Gpt`, ArabicOcr's
  `LiveLine`, LegalOcr's `StartupLoad` progress and `FiniteNumbers`.
- Settings (CLAUDE.md): the progress reporter by option and `IDRAK_PROGRESS`; the keyword index's `k1` and `b` stay
  options.

**Done when:** each new contract's suite passes; the moved code gives the same results (retrieval rankings, CLI output
byte for byte on the CLI's own tests, the tokenizer's ids for BMP text unchanged); the samples' tests pass with the
library's versions.

**P2 as built (2026-10-20).** G1-G7 and P1's setting:
- **G1** `RenamedTensorStore` (`src/Idrak/Models/RenamedTensorStore.cs`): a name map or a rename function, weight-norm
  folding (`weight_g`/`weight_v` and the `parametrizations` form), owning the inner store or not; passes the inner
  store's format, bfloat16 copies, transposed reads and read statistics. The Qwen-VL and Gemma 3 copies are gone;
  `OpenVoiceTensors` keeps only its name map.
- **G2** probe `timings` (intrusive): each operation by name, direction and shape, each layer by type and depth (tables
  "operation timings", "layer timings"); timing synchronization only during its run, a key stored once per new shape.
  The Chat sample's `ProfileRecorder` is gone; its profile command runs `Profiler.Run` with the probe.
- **G3** `CharTokenizerOptions { CodePoints, UnknownCharacter }`: default unchanged (BMP ids and the saved file).
  GptTraining's `RuneTokenizer` is gone (same ids on random text). Code-point mode refuses a vocabulary with a repeated
  or empty entry, where the sample's tolerated it.
- **G4** `IProgressReporter` + `ProgressReporters` (`Idrak.Abstraction.Diagnostics`): `console` (the CLI's line, output
  byte for byte), `plain`, `json`, `quiet`; chosen by option, then `IDRAK_PROGRESS`, then the output; a report between
  showings allocates nothing (the suite checks it). The CLI's `ProgressLine` is a thin adapter; ArabicOcr's `LiveLine` is
  gone; LegalOcr's `StartupLoad` uses the library's (its test expects the library's line).
- **G5** `IKeywordIndex` + `KeywordIndexes` (`Idrak.Text.Abstractions`), `bm25` with `k1`/`b`; `RetrievalIndex` builds
  through it (rankings and the saved default index unchanged); `idrak rag index --keywords NAME`. ReRanker's `Bm25` is
  gone (its old one also returned zero-score passages after the matches; the library's leaves them out).
- **G6** `INetworkRecipe` + `NetworkRecipes` (`src/Idrak/Layers/Abstractions/NetworkRecipes.cs`, guarded): `mlp`, `cnn`,
  `rnn`, `transformer-classifier`, `gpt` over the static `Architectures`; `CharGpt.Create` builds through `gpt` (the
  same network weight for weight, tested).
- **G7** `FiniteJsonNumbers` (single and double converters); the profile and telemetry JSON writers use it (output
  unchanged); LegalOcr's `FiniteNumbers.cs` is gone.
- **P1's setting:** `IDRAK_PLUGIN_PACKAGES` (`all`, `none`, or `text,vision,audio`), refused naming the variable; it
  and `IDRAK_PROGRESS` are in the tool's environment table.
- **Left (goes with P3):** GptTraining's own tqdm-style progress line (decision 9) moves onto the library's reporter; the
  new registries join `plugins list`. **Found, listed for the owner:** `Downloader.Size` formats with the current
  culture (rule 49; the CLI hides it with invariant globalization).
- **Run (CPU, the agent):** 219 passed, 1 failed (`profiler:`'s list of probes lacked `timings`), then 2 of 2; samples
  ArabicOcr 14, LegalOcr 12. **Merged on P8 (2026-10-20, b9540df):** one build, the 7 changed samples built; the P2
  filters with `engine:`, `decoder:`, `sampler:`: 249 passed, 0 failed; ArabicOcr 14, LegalOcr 12.

## Phase P3: generic components, vision part (G8-G14)

- New contracts first (G11 line segmentation, G12 text matching, G13 reading order, and G14's extension of
  `IImageModelFamily`), then the defaults, then ArabicOcr and the Ocr sample on them.
- G8 lands before P5 and P6 (IdrakVision and Gemma 3 use it); G14 lands before P5 (the CRNN family needs it).

**Done when:** the suites pass; ArabicOcr's tests (`samples/Idrak.Samples.ArabicOcr.Tests`) give the same lines, matches
and readings; the vision tuning and Gemma groups pass unchanged.

**P3 as built (2026-10-20).** G8-G14 and the leftovers:
- **G8** `ImageAttentionRules.ImageBlocks` (`image-blocks`), a library default: the two copies did not differ (both pass
  the window to `KeySpans.ImageBlocks`); IdrakVision's rule name `idrak-image-blocks` became `image-blocks` (its config
  value maps the same). Both `ImageBlockRule` classes are gone.
- **G9** augmentation `noise` (`PixelNoise`: `std` 0.05, `p` 1, clamped, objects kept); `LineNoise.cs` gone.
- **G10** the Ocr sample reads and writes through `ImageCodecs` and `ImageEncoders`; `Pgm.cs` gone.
- **G11** `ITextLineSegmenter` + `TextLineSegmenters` (guarded, `Idrak.Vision.Abstractions`; `PageLine`, `PageLines`,
  `LineSegmentationOptions` with the sample's hidden constants as options); default `projection`. ArabicOcr's cut
  command and line reader use it; `LineSegmenter.cs` gone.
- **G12** `ITextMatcher` + `TextMatchers` (guarded, `Idrak.Text.Abstractions`; `TextMatch`, `AlignedText`,
  `TextMatchOptions`); default `semi-global`. ArabicOcr's `TextAlignment` gone; an uninitialized pooled-slot read on a
  span past the last character fixed on the way.
- **G13** `IReadingOrder` + `ReadingOrders` in `Idrak.Abstraction.Data` (beside `TextShapers`: Imaging implements it and
  Vision uses it, so decision 10 puts it there); defaults `bidi` (UAX #9) and `runs` (the app's order). **`bidi`
  differs from ArabicOcr's order** (in a right-to-left line "abc 123" and "123 abc" give the same columns; "abc ١٢";
  punctuation inside Latin text): 9,518 of 20,000 random mixed lines, 0 of 5,000 of ArabicOcr's own; ArabicOcr keeps
  `runs`. ArabicOcr's recognizer now passes its direction to `runs`: right-to-left results identical; a left-to-right
  model's order changes (the old code reversed Latin runs there, which did not match its labels) — **for the owner to
  confirm**.
- **G14** `ImageTask.LineRecognition`, `LineRecognitionSettings` (height, stride, blank, direction, reading order, ink,
  log-probabilities), `ImageModel.Lines`; `ModelLineReader` (ArabicOcr's preparation, batches, CTC decoding through
  `CtcDecoders`) and `model.LineReader()`; `idrak predict` reads lines (`--decoder greedy|beam`), `idrak train --task
  line-recognition` trains or fine-tunes on line images with `NAME.txt` (CTC loss; cer, wer; `--direction`,
  `--reading-order`). ArabicOcr's line preparation calls the library's. The CRNN family is P5.
- **Leftovers done:** GptTraining's progress line on `ProgressReporters` (in place, or plain every N steps with
  `--log-every`); `plugins list` shows progress, keyword indexes, recipes, line segmenters, text matchers and reading
  orders; `ImageModelFamilies`' message names `Idrak.Vision.Plugins`; `CtcDecoders` left `shared-contracts.txt` (Audio
  and Vision use it).
- **Proven unchanged** (a scratch program, the removed code against the library's): segmentation on 480 pages, 23,000
  matches, `runs` on 20,000 lines, noise on 300 images, line preparation on 400, PGM training files on 300, all identical
  (the Ocr demo's `--save-image` dark-ink PGM: 1 of 300 differs by one level, the inversion now before rounding).
  Segmentation speed equal in Release (2.67 against 2.64 ms a page).
- **Run (CPU, the agent):** 172 passed, 1 failed (`CtcDecoders` in `shared-contracts.txt`), then 4 of 4; ArabicOcr 14.
  **Merged on P9 (2026-10-20, 135716e):** one build, 3 changed samples built; the P3 filters with `engine:`: 190 passed,
  0 failed; ArabicOcr 14 passed.

## Phase P4: the audio plug-ins

- Whisper, VITS (with uroman, moved from `src/Idrak.Audio/SpeechText`), OpenVoice and IdrakAudio from `samples/` into
  `Idrak.Audio.Plugins`, code moved as it is, namespaces changed, their own tensor stores replaced by G1.
- espeak-ng moves to `Idrak.Audio.Plugins.EspeakNg` with its `IDRAK_*` variable; the library's `TextPhonemizers` keeps
  no default that runs an outside program (rule 70); a family that needs it registers it.
- The scripts that use them (`samples/WhisperSpeech/transcribe.cs`, `samples/OpenVoice/voice.cs`) stay in `samples/` as
  apps referencing the package; the CLI help's `-P Idrak.VitsSpeech.dll`-style examples become the built-in
  registrations.

**Done when:** `audio asr`, `audio cli`, `audio tts`, `audio voices`, `audio voice train`, `audio language`,
`audio seq2seq`, `audio train`, `audio tts train` pass unchanged (same counts), and no library file starts an outside
program for text.

**P4 as built (2026-10-20).** The five plug-ins in `src/Idrak.Audio.Plugins`, one namespace each:
- **Moved as they were** (`git mv`; no computation changed: the `namespace` lines, two names' constants, one internal
  interface's namespace, usings and comments, so the outputs are the same code's, rule 71, and the nine groups below give
  the same counts): `samples/WhisperSpeech` → `Idrak.Audio.Plugins.Whisper`,
  `samples/VitsSpeech` and `src/Idrak.Audio/SpeechText/UromanArabic*.cs` → `Idrak.Audio.Plugins.Vits`, `samples/OpenVoice`
  → `Idrak.Audio.Plugins.OpenVoice`, `samples/IdrakAudio` → `Idrak.Audio.Plugins.IdrakAudio`,
  `src/Idrak.Audio/SpeechText/EspeakNgPhonemizer.cs` → `Idrak.Audio.Plugins.EspeakNg` (its `IDRAK_ESPEAK` and
  `IDRAK_ESPEAK_VOICE` with it, the tool's table updated). The four sample projects are gone from `Idrak.slnx`, the test
  project and the benches (which reference the package). Each family's README moved into its folder.
- **G1:** no own tensor store was left: OpenVoice's names already went through the library's `RenamedTensorStore` (P2);
  `WhisperWeights` and `IdrakAudioWeights` are weight readers (shape checks, seeded draws, the saved layout), not stores.
- **Registration** in P1's pattern: `WhisperPlugin` ("whisper"), `EspeakNgPlugin` ("espeak-ng"), `VitsPlugin` ("vits":
  the family, uroman's romanizer as `VitsPlugin.UromanArabic`, and `Requires(EspeakNgPlugin.Plugin)`), `OpenVoicePlugin`
  ("openvoice"), `IdrakAudioPlugin` ("idrak-audio": its four registrations as parts); `AudioPlugins.All` lists them in that
  order. The entry classes `WhisperSpeechPlugin` and `VitsSpeechPlugin` are now `WhisperPlugin` and `VitsPlugin`; the
  families' own `RegisterIdrakPlugin` are gone (the package's root has the one `-P` entry).
- **Rule 70:** `TextPhonemizers` and `TextRomanizers` register nothing in the library (their constants `EspeakNg` and
  `UromanArabic` moved to the plug-ins); `TextPhonemizers.First()` with none registered refuses naming the registry and
  the package. No library file starts an outside program for text (the espeak-ng plug-in is the only one).
- **The CLI:** the tool registers the audio package at startup, so `transcribe`, `speak`, `voice add|train`, `create`,
  `pretrain`, `resume` and `data audio --align` need no `-P`: the help's `-P Idrak.*.dll` examples, the hints and
  `idrak help audio` say so. `samples/WhisperSpeech/transcribe.cs` and `samples/OpenVoice/voice.cs` stay as apps on the
  package (`#:project` on it; both build).
- **Tests:** the moved tests keep their names. Their three CLI refusals "without the plug-in" run with
  `IDRAK_PLUGIN_PACKAGES=none` (the tool registers the package otherwise); the VITS tokenizer test registers the family
  for uroman and checks the library registers no phonemizer or romanizer; `plugins: packages`' internals check allows
  the test runner only. VITS's internal `IVitsDurationPredictor` moved to `Idrak.Audio.Plugins.Vits.Abstractions` (the
  inventory's rule; its two predictors unchanged). API dumps and the inventory regenerated.
- **For the owner to confirm:** `Idrak.Audio.Plugins` grants its internals to `Idrak.Tests` (as every library package
  does), so the moved code stays internal where it was (VITS's training passes and draws, IdrakAudio's weights and
  exports, OpenVoice's name map); P1's guard said "grant none". The other way is to make those members public.
- **Run (CPU, the agent):** before: the nine groups 64 passed (asr 8, cli 10, language 7, seq2seq 3, train 10, tts 5,
  tts train 6, voice train 9, voices 6); guards and touched groups 95 passed. After one build: 157 passed, 2 failed (the
  inventory's allow list named `IVitsDurationPredictor`; the `create` refusal ran after `create init` had registered the
  package again); after one rebuild `abstraction inventory|audio cli|public API` 16 of 16. The nine groups: the same 64.

## Phase P5: the vision plug-ins

- Qwen-VL and IdrakVision from `samples/`, LLaVA from `tests/Idrak.PluginTests/Llava.cs`, into `Idrak.Vision.Plugins`,
  on G1 and G8. `Idrak.PluginTests` keeps its proof families; LLaVA's place there is taken by the package itself, which
  has no internals either.
- The CRNN line recognizer from ArabicOcr's `Recognizer` into `Idrak.Vision.Plugins.Crnn`, on G14; ArabicOcr becomes an
  app that references it.
- Plans 16 and 19's paths are updated in their "as built" notes.

**Done when:** `qwen vl:`, `vision contracts`, `vision tuning contracts`, `vision tuning reference`, IdrakVision's groups
and ArabicOcr's tests pass unchanged, and `samples/` holds only apps.

**P5 as built (2026-10-20).** The four families in `Idrak.Vision.Plugins`, each an `IdrakPlugin` in `VisionPlugins.All`
(in this order), public API only:
- **Qwen-VL** → `Idrak.Vision.Plugins.Qwen` (`src/Idrak.Vision.Plugins/Qwen`, its README moved), plug-in "qwen-vl": both
  variants' vision part and text decoder, one part each. Already on G1 (P2's `RenamedTensorStore`).
  `QwenTextDecoder.Register(variant)` became `QwenTextDecoder.Architecture()`: a public static `Register` with an argument
  counts as a registry in the inventory.
- **LLaVA** → `Idrak.Vision.Plugins.Llava` (`tests/Idrak.PluginTests/Llava.cs` moved), plug-in "llava".
  `Idrak.PluginTests` keeps its proof families and references the package for the outside family's text decoder (no
  internals either way).
- **IdrakVision** → `Idrak.Vision.Plugins.IdrakVision` (README moved), plug-in "idrak-vision", on G8 since P3 (the
  library's `image-blocks`). `samples/IdrakVision/scripts/b6-curriculum.ps1` stays an app: it builds the CLI only and
  passes no `-P`.
- **CRNN** → `Idrak.Vision.Plugins.Crnn` (from ArabicOcr's `Recognizer.cs`), plug-in "crnn", on G14: `CrnnFamily`
  (`IImageModelFamily`, task `LineRecognition`, architecture `CrnnForLineRecognition`) reads a model folder's config.json
  (`CrnnSettings`: alphabet, height, direction, reading order, now a setting with `runs` as the default, sizes) and its
  safetensors (tensors named by layer: `layers.{i}.parameters.{j}`, `layers.{i}.buffers.{j}`), each shape checked.
  `CrnnRecognizer` makes, loads (through `ImageModels.Load`), saves, encodes and reads through the library's
  `ModelLineReader` and `CtcDecoders`, with an optional text normalizer (the app passes its NFC and white-space form);
  `MeasuredLineBatches` is ArabicOcr's measured batching, moved with it. So `idrak predict MODEL LINES` and `idrak train
  MODEL --data LINES` read and fine-tune the app's recognizer.
- **ArabicOcr** is an app on the package: it registers `CrnnPlugin` beside Gemma 3; its model folder is now config.json
  and model.safetensors (it was `recognizer.ikm` and `recognizer.json`; such a folder still loads) — **for the owner to
  confirm**. Its `runs` order and direction stay as P3 left them. LegalOcr references the package for Qwen-VL.
- **The tool:** it registers the four with the package at startup, so the `-P Idrak.IdrakVision.dll` of `create`'s and
  `pretrain`'s help is gone (and `create init --vision`'s comment). `samples/` keeps no vision family but Gemma 3 (P6).
- **Tests:** the moved groups keep their names and counts. Two changed to the plug-in contract: `qwen vl:`'s
  not-registered test unregisters through `QwenVisionPlugin.Unregister()` (a second `Register` does nothing now), and
  `cli pretrain vision`'s refusal runs with `IDRAK_PLUGIN_PACKAGES=none` (the tool registers the package at startup).
  New: `plugins: vision` (the four are `VisionPlugins.All`; Register twice, Unregister as found, origin the package) and
  `plugins: crnn` (save, load through the family, the same readings as the recognizer and through `model.LineReader()`,
  the image model family suite read a line at a time, the earlier layout, refusals, not registered). ArabicOcr's test
  reads `CrnnRecognizer.ConfigFile`.
- **Proven unchanged** (a scratch comparison, not committed): ArabicOcr's `eval` (the 32 held-out and the 160 training
  lines) and `read` (8 pages), greedy and beam, on a model trained before the move, give the same JSON (texts, labels,
  scores; timings aside) from the old app and the new one, and from the new one on that model saved in the new layout.
  Training speed equal (0.37 and 0.38 s an epoch on the test set).
- **Run (CPU, the agent):** 192 passed, 2 failed (`cli pretrain vision`'s refusal came after `create init` had
  registered the package again; the family suite's batch check on the CRNN, whose backward LSTM reads a batch's
  padding), then 2 of 2; final 194 passed, 0 failed; ArabicOcr 14, LegalOcr 12.

## Phase P6: Gemma

- Gemma 1, 2, 3 (`GemmaForCausalLM`, `Gemma2ForCausalLM`, `Gemma3ForCausalLM`) and the text decoder of
  `Gemma3ForConditionalGeneration` with its three checkpoint layouts move from `src/Idrak/Models/Architectures.cs` to
  `Idrak.Text.Plugins.Gemma` (plan 16 step 4, done here).
- Gemma 3 vision (SigLIP, projector) from `samples/Gemma3Vision` to `Idrak.Vision.Plugins.Gemma`, on G1 and G8, which
  registers the text decoder through `Idrak.Text.Plugins.Gemma`.
- The internal helpers they use (`GemmaSpec`, `Gemma3Spec`, `CommonSpec`, the tensor-name maps) either move with them or,
  where generic, become public helpers of the decoder (named for what they do, not for a family).

**Done when:** `pretrained:`, the Gemma 3 vision groups and `gguf:` pass unchanged; a library user without the package
gets "not registered" naming `Idrak.Text.Plugins`.

**P6 as built (2026-10-20).** Gemma leaves the library; plan 16's step 4 is done:
- **Text** (`src/Idrak.Text.Plugins/Gemma/`): `GemmaPlugin` ("gemma", in `TextPlugins.All` after T5) registers
  `GemmaForCausalLM`, `Gemma2ForCausalLM`, `Gemma3ForCausalLM` and the text decoder of `Gemma3ForConditionalGeneration`
  in `PretrainedArchitectures`, one part each; `GemmaArchitectures` holds them (`Gemma()`, `Gemma2()`, `Gemma3()`,
  `Gemma3VisionLanguage()`, `TensorName`). The code moved from `src/Idrak/Models/Architectures.cs` as it was
  (`GemmaSpec`, `Gemma3Spec`, `Gemma3Keys`, the two text layouts, `Gemma3TextConfig` with its defaults, `InType`); only
  calls to the library's public helpers are qualified (`PretrainedFamilies.CommonSpec`, `LlamaStyle`,
  `LlamaTensorName`, all public already) and `PretrainedFamilies.GemmaTensorName` became `GemmaArchitectures.TensorName`.
  No internal was opened and the package grants none. `Architectures.cs` lost only the Gemma lines (P7 edits it next).
- **Vision** (`src/Idrak.Vision.Plugins/Gemma/`): `samples/Gemma3Vision`'s four files by `git mv`, only the namespace
  changed (`Idrak.Gemma3Vision` → `Idrak.Vision.Plugins.Gemma`); already on G1 (`RenamedTensorStore`) and G8
  (`image-blocks`). `Gemma3VisionPlugin` ("gemma3-vision", `VisionPlugins.All`) is now an `IdrakPlugin`:
  `Requires(GemmaPlugin.Plugin)`, then `VisionFamilies`; `Register()` registers once, `Unregister()` added, its own
  `RegisterIdrakPlugin` gone (the package root keeps the `-P` entry). `Idrak.Vision.Plugins` references
  `Idrak.Text.Plugins` (one way). The sample project is gone from `Idrak.slnx` and the test project; ArabicOcr and
  LegalOcr reference the package; ArabicOcr's `tune-vlm` prints `idrak tune` without `-P` (the tool registers the
  family), its test expects that; the CLI help's `-P Idrak.Gemma3Vision.dll` examples dropped the option.
- **Contracts:** none new. The registries the move needed exist (`PretrainedArchitectures`, `VisionFamilies`,
  `IdrakPlugins`). **GGUF:** `GgufBuiltIns` never mapped a Gemma architecture (`gemma`, `gemma2`, `gemma3` files are
  refused as not registered, before and after), so nothing was routed; `GgufArchitectures.Register` is the contract a
  mapping would use (a leftover).
- **Not registered:** without the package, `PretrainedArchitectures.Get` and `PretrainedModel.Load` refuse with "No
  architecture 'Gemma3ForConditionalGeneration' is registered (…); … or reference Idrak.Text.Plugins (text models) or
  Idrak.Vision.Plugins (vision-language models) and call its RegisterAll()" (P8's text, unchanged); with the text
  plug-in and not the vision one, the vision family's message, as before. The library keeps no Gemma default
  (`PretrainedArchitectures.Default` is null for all four, tested).
- **Tests:** new `plugins: gemma` (1). The tests that read Gemma text configurations register `GemmaPlugin`
  (`pretrained:`'s family configs, `pretrain:`'s round trip and refusals); the tests that take the vision family away
  use `Gemma3VisionPlugin.Unregister()` and keep the text family (`UnregisterGemma3Vision`), and their in-process tool
  runs use `IDRAK_PLUGIN_PACKAGES=none` (the tool registers the family again otherwise), as P4's did.
- **Proven unchanged (rule 71):** the moved code diffed against the removed one (above); the groups below give the same
  counts, among them the tiny Gemma 3's logits against transformers' in its three layouts, a Gemma 2 checkpoint against
  the reference decoder, and every vision, tuning, CLI and serve group of the tiny Gemma 3.
- **Run (CPU, the agent):** before, 315 passed over `pretrained:`, `gguf:`, `plugins:`, `chat templates`, `cli run:`,
  `cli models:`, the guards, `engine:`, `generic components`, the vision, tuning, image, CLI image and serve groups,
  `decoder:`, `pretrain:`, tokenizers and the rest that touch Gemma; ArabicOcr 14, LegalOcr 12. After one build (one
  cref fix, then the build again): 316 passed, 0 failed (every group the same count, `plugins:` 22 → 23); ArabicOcr 14,
  LegalOcr 12.
- **For the owner to confirm:** a library user needs `Idrak.Text.Plugins` for any Gemma model (and
  `Idrak.Vision.Plugins` for Gemma 3 with images); `PretrainedFamilies.GemmaTensorName` → `GemmaArchitectures.TensorName`;
  `Idrak.Gemma3Vision` (assembly and namespace) → `Idrak.Vision.Plugins.Gemma`; `Gemma3VisionPlugin.Register()`
  registers once (it re-registered before) and brings the text family; an app that registers over a Gemma name and
  unregisters it gets nothing back (there is no library default to return to).

## Phase P7: the other text families

- Llama, Mistral and Mixtral, Qwen2/Qwen3 and their MoE, from `PretrainedFamilies.BuiltIns`; their GGUF mappings and
  every pre-tokenizer name of the candidates table from `GgufBuiltIns` (registered through the public
  `GgufArchitectures.Register` and `GgufPreTokenizers.Register`).
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

**P7 as built (2026-10-20).** Llama, Mistral and Mixtral, Qwen2/Qwen3 and their MoE, and the DeepSeek and harmony
tool-call formats leave the library for `Idrak.Text.Plugins`, one namespace each, public API only:
- **The plug-ins** (each an `IdrakPlugin`; `TextPlugins.All` is now BERT, T5, Gemma, Llama, Qwen, Mistral, harmony,
  DeepSeek, an order that keeps the tool-call formats asked as before: deepseek, harmony, mistral, qwen3-coder, then the
  library's pythonic and json):
  - `Idrak.Text.Plugins.Llama` ("llama"): `LlamaForCausalLM` (`LlamaArchitectures.Llama()`), the GGUF mapping `llama`
    (`LlamaArchitectures.Gguf()`: interleaved q/k rows, experts as `MixtralForCausalLM`, Llama 3's `rope_freqs.weight`
    read back as the llama3 scaling), the pre-tokenizer names `llama3`, `llama-bpe`, `llama-v3`, `smaug-bpe`, `falcon3`,
    `pixtral` (`LlamaArchitectures.SplitPattern`), RoPE `llama3` (`LlamaRopeScaling`: `Name`, `Create`, `Scale`), and the
    pre-tokenizer fallback `llama3` when none is set (taken back only if it set it).
  - `Idrak.Text.Plugins.Qwen` ("qwen"): `Qwen2ForCausalLM`, `Qwen3ForCausalLM`, `Qwen2MoeForCausalLM`,
    `Qwen3MoeForCausalLM` (`QwenArchitectures`, `MoeTensorName`), the GGUF mappings `qwen2`, `qwen3`, `qwen2moe`,
    `qwen3moe`, the pre-tokenizer names `qwen2`, `qwen35`, `megrez`, `hunyuan`, the tool-call format `qwen3-coder`.
  - `Idrak.Text.Plugins.Mistral` ("mistral"): `MistralForCausalLM`, `MixtralForCausalLM` (`MistralArchitectures`,
    `MixtralTensorName`), the pre-tokenizer `tekken`, the tool-call format `mistral`; `Requires(LlamaPlugin)` (its GGUF
    files are llama.cpp's `llama`).
  - `Idrak.Text.Plugins.DeepSeek` ("deepseek"): the tool-call format `deepseek` and the pre-tokenizer
    `deepseek-r1-qwen` (Qwen2's pattern, `QwenArchitectures.SplitPattern`); `Idrak.Text.Plugins.Harmony` ("harmony"): the
    tool-call format `harmony` (the namespaces the candidates table gives).
- **Moved as they were** (rule 71, a scratch diff of each block against the removed one): the four parsers byte for byte;
  `MixtralTensorName`, `QwenMoeTensorName` (now `MoeTensorName`), Llama 3's GGUF scaling and the llama3 method with
  only their names and the call to `PretrainedFamilies.CommonTensorName` changed (the method gained an argument check);
  the three split patterns and the GGUF mappings' values identical; the detectors' lambdas as they were.
- **The library keeps no family:** `PretrainedFamilies.BuiltIns` and the static registrations of
  `PretrainedArchitectures` and `GgufArchitectures` are gone (empty tables); `GgufPreTokenizers` keeps GPT-2's rule
  (`gpt2`, `gpt-2`, `default`); `RopeScalings` keeps `linear`, `yarn`, `dynamic`; `ToolCallFormats` keeps `json`,
  `pythonic`. `PretrainedModel.Load` of an architecture nobody registered is refused with
  "No architecture 'X' is registered (…); … reference Idrak.Text.Plugins …" (P8's text); `GgufModel.Prepare` of a
  family nobody registered with "GGUF architecture 'x' is not registered … Idrak.Text.Plugins"; an unknown RoPE scaling
  now names the package too.
- **Contracts:** no new registry. Where the GGUF converter held family code it now asks the mapping:
  `GgufArchitecture.ExpertCountKey` (the expert count's config key: `num_experts` unless set; Llama's mapping sets
  Mixtral's `num_local_experts`) and `GgufArchitecture.Configure` (the family's own keys, run before the generic RoPE
  keys, which are skipped when it sets `rope_scaling`: Llama 3's `rope_freqs.weight`). The pre-tokenizer of a name
  nobody registered was Llama 3's, fixed in the library: now the setting `GgufPreTokenizers.Fallback` (null in the
  library: such a file is refused naming the registry and the package; the Llama plug-in sets `llama3` when unset, so
  behaviour with the package is as before).
- **Public members:** `MarkedToolCallParser` is a public abstract base (protected constructor, `Context`, `Open`,
  `Close`, `Parse`, `Decide`); `ToolCallText` (its JSON and marker helpers) and `CharBuffer` (harmony's held text) are
  public; `RopeScaling.Number` is public (the llama3 method reads its parameters with it).
- **Renamed or removed (for the owner to confirm):** `PretrainedFamilies.LlamaStyle` → `CommonArchitecture` and
  `LlamaTensorName` → `CommonTensorName` (both old names kept one release, `[Obsolete]`; the plug-ins and the vision and
  audio plug-ins use the new ones); `PretrainedFamilies.MixtralTensorName` → `MistralArchitectures.MixtralTensorName`,
  `QwenMoeTensorName` → `QwenArchitectures.MoeTensorName`, `RopeScaling.Llama3` → `LlamaRopeScaling.Create`,
  `ToolCallFormats.Qwen3Coder`, `Mistral`, `Harmony`, `DeepSeek` → each plug-in's `ToolCallFormat` (removed from the
  library, no obsolete copy: a forwarder would keep family code there). `DecoderSpec`'s reader of Idrak's earlier
  JSON form of a llama3 scaling builds the same scaling data itself. The conformance kit's `RopeScalingSuite` has no
  built-in llama3 parameters (a plug-in's scaling is checked with its own).
- **Users moved on:** LLaVA (`Idrak.Vision.Plugins.Llava`) `Requires` Llama, Mistral and Qwen (its text_config's
  families); Gemma, IdrakAudio, IdrakVision and Qwen-VL's text decoder use the renamed helpers. The samples Chat and
  CodingAgent (they loaded any family the library had) reference `Idrak.Text.Plugins` and call
  `TextPlugins.RegisterAll()`; no bench loads these families. The tool registers the package at startup (P1), so
  `idrak chat` and `run` need no `-P`: checked in the tool's own process (`plugins: llama`).
- **Tests:** the runner registers `TextPlugins.RegisterAll()` at startup, as an app and the tool do (some hundred tests
  in forty files, the cluster ranks' child processes among them, build Llama or Qwen models; P6's per-test
  `GemmaPlugin.Register()` calls stay and do nothing) — **for the owner to confirm** (the other way is a registration
  in each test and in the cluster child entry). New: `plugins: llama`, `plugins: qwen`, `plugins: mistral`,
  `plugins: deepseek`, `plugins: harmony` (the library's empty tables, not registered naming the package, Register
  twice, Unregister as found, Mistral's `Requires`, the fallback, the tool's own process). Changed expectations, each
  naming the cause: the GGUF architectures and pre-tokenizers come from `Idrak.Text.Plugins`, none from the library
  (`plugins:`), an unregistered pre-tokenizer without a fallback is refused, the tool-call formats' order (the families'
  then the library's), the override report's RoPE rows (a plug-in's added scaling has no default), the conformance
  kit's llama3 check with its parameters, `LlamaRopeScaling.Create` and the moved name maps in `rope scaling:`,
  `decoder:` and `mixture of experts:`. The inventory's registered-names column lists the library's own names (by each
  registry's `Origin`), since the runner now registers the package before it runs.
- **Left as they were (listed for the owner):** `GgufModel.WriteTokenizer`'s fallback chat templates for a GGUF file
  without one (ChatML by `<|im_start|>`, Llama 3's by `<|start_header_id|>`) stay in the library: a template fallback
  needs a contract of its own. `DecoderSpec` reads Idrak's own earlier llama3 JSON form (a file format, not a family).
- **Run (CPU, the agent):** before, the phase's groups 178 passed (`pretrained:` 9, `gguf:` 3, `decoder:` 12,
  `rope scaling:` 2, `chat templates:` 5 (the tool-call formats), `fine-tuning:` 16, `cli run:` 9, `cli serve:` 7,
  `tokenizers:` 53 (Plan 24's BPE baseline among them), `plugins:` 25 with the op, packed and kv plug-ins 4, 5, 3,
  `mixture of experts:` 3, `generation:` 13, `coding tools:` 3, `cli models:` 6). After one build (the two changed
  samples built too): those groups the same counts, `plugins:` 30 (the five new), and with the guards (`public API`,
  `abstraction inventory`, `naming:`, `conformance kit:`, `override loop:`, `engine:`, `generic components`) and every
  group whose tests load a moved family (pretrain, vlm scratch, audio chat, language, asr, cli and train, distillation,
  fine-tuning options, cli measure, design, polish, health, pretrain, vision, images, arabic, tool contracts, mcp, the
  vision contracts and tuning groups, qwen vl, image prefill, pan and scan, image families and transforms, the image
  reader, conversation tuning, tokenizer training, pytorch checkpoint, four cluster tests through rank processes):
  467 passed, 0 failed. The inventory's names column fixed after (one rebuild): `abstraction inventory|public API` 6
  of 6.
- **Owner:** the full suite once (P7 reaches every package), the CUDA filters, and the real models: `idrak chat` and
  `idrak run` on Llama, Mistral, Mixtral, Qwen2, Qwen3 folders and a GGUF file with no `-P`.

## Phase P8: the engine for encoders and encoder-decoders

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

**P8 as built (2026-10-20).** The core of the phase; families register only:
- **Contracts in `Idrak.Abstraction.Generation`:** `IAttentionPositionBias` (`Heads`, `For(queries, keys, queryStart,
  device)` → `[heads, q, k]`) with the base `AttentionPositionBias` (computed once per shape while nothing trains) and the
  registry `AttentionPositionBiases` (default `relative`: a learned value per head and distance, `max_distance`,
  `bidirectional`); `ISequenceEncoder` (`Width`, `Vocabulary`, `TypeVocabulary`, `MaxPositions`, `Device`, `Network`,
  `Encode(ids, lengths, tokenTypes)` → `[batch, time, width]`); `ISequencePooler` with `SequencePoolers` (`first`,
  `mean`, `max`); `ISeq2SeqModel` (`Encoder`, `Vocabulary`, `DecoderStartToken`, `EndToken`, `MaxTargetPositions`,
  `Decode`, `SetMemory(memory, context, lengths)`, `DecodeCached`); `DecodingContext.SetMemory(…, lengths)` and
  `MemoryLengthsOf`. `IAttentionPositionBias` and `ISeq2SeqModel` stay in `Idrak.Abstraction` with reasons in
  `tests/Idrak.Tests/data/shared-contracts.txt` (only `Idrak` uses them today).
- **Contracts in `Idrak.Models.Abstractions`:** `PretrainedCheckpoint`; `IPretrainedEncoderFamily` (`Build`, `Heads`,
  `Pooler`) with `PretrainedEncoders`; `IPretrainedSeq2SeqFamily` with `PretrainedSeq2Seq` (no library family in either;
  "not registered" names `Idrak.Text.Plugins`); `EncoderHeads` (`sequence-classification`, `masked-lm`).
- **Implementations:** `MultiHeadAttention`, `CausalSelfAttention` and `CrossAttention` take the score scale, a padding
  mask and a position bias as settings (nothing set: bit for bit as before, tested); `SequenceEncoder`,
  `Seq2SeqNetwork`, `EncoderHead`, `PretrainedEncoderModel.Load`, `PretrainedSeq2SeqModel.Load`, `Seq2SeqGenerator`
  (an `ITextModel`); `TextEncoder` takes an encoder and a `Pooler` setting (existing vectors bit for bit). Suites:
  `PositionBiasSuite`, `SequencePoolerSuite`, `SequenceEncoderSuite`, `Seq2SeqModelSuite`; toy families from
  `tests/Idrak.PluginTests/EncoderFamilyPlugin.cs` (public API only). Groups `engine: attention options` (4),
  `engine: encoders` (3), `engine: seq2seq` (2). No new kernels: padding takes the per-range attention path, the bias
  basic operations every backend has.
- **Left, and where it goes:** the CLI's `run` and the server with an encoder-decoder (a new inference-engine kind beside
  the text one) and a beam setting (`GenerationOptions` has none; `BeamSearch.Decode` already works through
  `DecodeCached`) go with P10; the post-norm (BERT) and RMS-norm (T5) encoder blocks are generic blocks P9 and P10 add
  to the library; the padding scope is not kept open through a checkpointed backward recompute (as `EncoderMemory`
  today).
- **Measured, not taken** (listed for the owner, not chased): with a bias, attention builds the full `[batch·heads, t, t]`
  scores, and cached decoding does so over every cache slot each step; `relative` builds its index on the host and
  uploads it each decoding step, and a shared bias that trains is recomputed by every layer; padded encoder attention
  passes no mask, so on a device without a per-range kernel for the head size it runs on the host; the training path
  with dropout builds its padding mask on the host; `Seq2SeqGenerator` reads one token back each step and has no
  recorded-graph replay.
- **Run (CPU, the agent):** 205 passed, 6 failed (dumps, inventory placement, two bugs in its own suites), then
  `engine:|abstraction inventory|public API` 18 of 18. **Merged with P1 (2026-10-20, on 064be4c):** one build; the P1 and
  P8 filters together: 287 passed, 0 failed.

## Phase P9: BERT (`Idrak.Text.Plugins.Bert`)

- `BertModel` on P8's encoder contract: word, position and token-type embeddings, post-norm blocks with GELU, the
  pooler; `BertForMaskedLM`, `BertForSequenceClassification`, and sentence embeddings (mean or first-token pooling) through
  `TextEncoder`. The tokenizer is Plan 24's WordPiece (`tokenizer.json` or `vocab.txt`).
- Fixtures: a tiny random BERT and its outputs from transformers (pinned, a script under `tools/`), committed small;
  real models (`google-bert/bert-base-uncased`, `aubmindlab/bert-base-arabertv02`) are the owner's run.

**Done when:** hidden states, pooled vectors and logits equal transformers' within float tolerance, batched with padding
equal to one by one.

**P9 as built (2026-10-20).**
- **Library (generic):** `TransformerEncoderLayer` takes `NormPlacement` (`Before`, the default, unchanged; `After`), by
  an init property or a `FromLayers(…, activation, placement)` overload, with fused inference, gradient and cached paths,
  carried by the layer description, the network builder (one optional parameter), the `transformer` network step and
  ONNX export. Encoder heads gain `pooler` (dense + tanh over the first token) and `EncoderHeadRequest.TiedTo` (a
  masked-LM decoder reading the word embeddings in place). The safetensors reader skips integer and boolean tensors
  (listed in `LeftOut`) instead of refusing the file (real checkpoints carry an int64 `position_ids`).
  `TextEncoder.SpecialTokens` (off by default) adds `[CLS] … [SEP]`.
- **The plug-in** (`src/Idrak.Text.Plugins/Bert/`, `BertPlugin` in `TextPlugins.All`): `BertModel`, `BertForMaskedLM`,
  `BertForSequenceClassification`; names with or without `bert.`, LayerNorm `weight`/`bias` or `gamma`/`beta`; tied
  decoder unless `tie_word_embeddings` is false; refuses, naming the setting, other activations than gelu, gelu_new,
  gelu_pytorch_tanh and relu, relative position embeddings and decoder checkpoints.
- **Fixtures:** `tools/bert/make_tiny_bert.py` (torch 2.14.1, transformers 5.18.0, tokenizers 0.23.3, safetensors 0.8.0;
  reruns write the same bytes) → `tests/Idrak.Tests/data/plugins/bert/` (a 337 KB model on the committed WordPiece
  vocabulary, 2 layers, width 32; 110 KB of references: three texts and a padded batch of pairs with token types).
  `--real` records the real models' references (the owner).
- **Tolerance 1e-4** (absolute to 1, relative above): float32 on both sides, measured worst 6.7e-6 (masked-LM logits);
  each real mistake it guards against (LayerNorm epsilon, tanh GELU, pre-norm, a transposed weight) moves outputs by
  1e-3 or more. Real models 2e-3.
- **Tests:** `plugins: bert` (5), `engine: post-norm block` (1), `bert real:` (the owner's).
- **Confirmed by the owner (2026-10-20):** `InventoryTests` does not count the three plug-in packages as users when it
  decides where a contract lives (they count as apps, like the CLI); otherwise the first plug-in demanded the library's
  `EncoderHeads` move into a plug-in package. This is decision 11.
- **Measured, not taken** (from the code): `TextEncoder` pads each batch to `MaxLength`, not its longest text;
  `PretrainedEncoderModel.Hidden`/`Pooled`/`Logits` leave gradients on when the caller forgets `NoGrad`; exact-erf GELU
  is not fused into the first projection; the masked-LM head scores every position; the pooler weights exist twice
  (pooler and classification heads).
- **Run (CPU, the agent):** 73 and 59 passed. **Merged on P2 (2026-10-20, 207bf1d):** one build; `plugins:`, `engine:`,
  `retrieval:`, `tokenizers: wordpiece`, `onnx`, `pretrained:`, `layers:`, `layer norm:`, `pytorch checkpoint`,
  `safetensors`, `generic components`, `decoder:`, `builder:`, the guards: 150 passed, 0 failed.

## Phase P10: T5 (`Idrak.Text.Plugins.T5`)

- `T5ForConditionalGeneration` on P8's encoder-decoder contract: RMS norms, the bucketed relative position bias of the
  first layer shared by the rest, no score scaling, ReLU (T5 v1.0) or gated GELU (v1.1, mT5) feed-forward, tied or untied
  head. The tokenizer is Plan 24's Unigram (`tokenizer.json` or `spiece.model`).
- Fixtures as P9's; real models (`google-t5/t5-small`, `google/mt5-small`) are the owner's run.

**Done when:** encoder states and greedy generations equal transformers' on the fixtures, and the KV-cached decode equals
the uncached one.

**P10 as built (2026-10-20).**
- **Plug-in** `Idrak.Text.Plugins.T5` (`T5Family`, `T5Plugin`, in `TextPlugins.All`; public API only):
  `T5ForConditionalGeneration` and `MT5ForConditionalGeneration` as encoder-decoders, `T5EncoderModel` and
  `MT5EncoderModel` as encoders (mean pooler). One embedding table shared in place by the decoder and a tied head;
  pre-norm RMS blocks; score scale 1; each stack's first-layer bias shared by the rest; ReLU, gated GELU or gated SiLU;
  tied head with the `d_model^-0.5` rescale (or `scale_decoder_outputs`), else `lm_head.weight`; start and end tokens
  from config.json; longest target `n_positions` (512 when absent). Other `feed_forward_proj` values, bucket settings
  that do not fit, untied checkpoints without a head and wrong shapes are refused naming the setting.
- **Library (no family name):** `BucketedPositionBias` (kind `bucketed`, beside `relative`; buckets equal
  transformers' over -700..700, computed once per shape and shared); `TransformerEncoderLayer` takes RMS norms and its
  own `FeedForward` (`BlockNormalization`, a `FromLayers` overload, `FeedForwardBlock`; the description's `"norm":
  "rms"`; ONNX export refuses it; LayerNorm stays the default on the old paths); `GenerationOptions.NumBeams`,
  `LengthPenalty`, `EarlyStopping` (refused when invalid); `Seq2SeqGenerator` beams through `BeamSearch.Decode` with one
  encoder memory, `GenerateBeams`, `AddSpecialTokens`, `SkipSpecialTokens`, `EndSource`;
  `IPretrainedSeq2SeqFamily.EndsSource` (`</s>` appended for a bare `spiece.model`); the tokenizer padded to the model's
  vocabulary; the engine kind `seq2seq` (`engine.Seq2SeqModel`, an `ITextModel`). `TextGenerator` ignores `NumBeams`.
- **CLI:** `idrak run` answers a registered encoder-decoder directly (`--beams`, `--length-penalty`); options that do
  not apply are refused naming the option; `--beams` on a decoder model is refused.
- **Not done: `idrak serve`.** `ServeHost.AddModels` registers every model as a chat model; serving a seq2seq model
  needs it found through `PretrainedSeq2Seq.Find` and registered as `engine.Seq2SeqModel`, the chat routes skipping it,
  the announcement and status lines, and a `cli serve:` test. For the owner to decide.
- **Fixtures** (`tools/t5/make_tiny_t5.py`, pinned `requirements.txt`; a rerun writes the same bytes):
  `tests/Idrak.Tests/data/plugins/t5/` `tiny-t5` (v1.0) and `tiny-t5-gated` (v1.1, its `</s>` row aimed so some
  generations end early), 3 encoder and 2 decoder layers, with `reference.safetensors` and `reference.json`. Tolerance
  1e-4; largest measured: encoder states 6.4e-6, logits 8.4e-6; greedy and beam outputs equal transformers' token for
  token; cached decode equals uncached.
- **Tests:** `engine: bucketed position bias`, `engine: rms-norm block`, `engine: beam setting`, `plugins: t5` (5),
  `cli run: encoder-decoder`, `t5 real:` (the owner's).
- **Slowness noticed, not fixed:** each cached step rebuilds the bucket index on the host and uploads it, and attention
  with a bias builds the full score matrix over every cache slot (as P8); beam search reads all beams x vocabulary logits
  back each step; a tied checkpoint that also holds `lm_head.weight` is compared value by value on the host at load. An
  existing ONNX bug: exporting a `TransformerEncoderLayer` with a custom activation casts it to `Linear` and throws.
- **Run (CPU, the agent):** 150 passed, 1 failed (its own expectation named `wi` where the family reads `wo` first),
  then the inventory guard's `ISeq2SeqModel` line; final 6 of 6. **Merged on P3 (2026-10-20, 4b2ea6a):** one build; the
  P10 filters with the generic, override and guard groups: 178 passed, 0 failed.

## Leftovers for the owner (noted 2026-10-20, not worked on)

The owner takes these after this plan is finished; no agent starts them before.
- **ArabicOcr's left-to-right reading order (P3).** The recognizer now passes its direction to `runs`: right-to-left
  results are identical; a left-to-right model's order changes (the old code reversed Latin runs, which did not match
  its labels). Keep it, or restore the old order (one argument).
- **`idrak serve` for an encoder-decoder (P10).** `ServeHost.AddModels` registers every model as a chat model; a
  seq2seq model needs finding through `PretrainedSeq2Seq.Find`, registering as `engine.Seq2SeqModel`, the chat routes
  (`MapChatApi`, `/api/chat`, the chat UI, `/v1/chat/completions`) skipping it, the announcement and status lines, and a
  `cli serve:` test. `/v1/completions` and MCP generate already work through `ITextModel` once it is registered.
- **Slowness found, not fixed (P8, P10):** a cached decoding step rebuilds the bucketed bias index on the host and
  uploads it; attention with a bias builds the full score matrix over every cache slot; beam search reads all beams x
  vocabulary logits back every step; a tied checkpoint that also holds `lm_head.weight` is compared value by value on
  the host at load.
- **ONNX bug, existing (found in P10):** exporting a `TransformerEncoderLayer` built with a custom activation casts the
  activation to `Linear` and throws an invalid cast.
- **P4, to confirm:** `Idrak.Audio.Plugins` lets `Idrak.Tests` see its internals (as every library package does; P1's
  guard relaxed to the test runner only), keeping VITS's training passes, IdrakAudio's weights and OpenVoice's name map
  internal; the other choice is making them public. Public names changed: `WhisperSpeechPlugin` → `WhisperPlugin`,
  `VitsSpeechPlugin` → `VitsPlugin`, `TextRomanizers.UromanArabic` → `VitsPlugin.UromanArabic`,
  `TextPhonemizers.EspeakNg` → `EspeakNgPlugin.Phonemizer`, the per-family `RegisterIdrakPlugin` methods gone (the
  package root keeps the one `-P` entry). The CLI registers the audio families at startup, so `-P` is no longer needed.
- **P4, process starts left in the library:** `src/Idrak.Text/Generation/CodingTools.cs` and `CodingAgent.cs` (the
  coding agent's shell tool, not text processing) are the only process starts in library packages beyond the CLI and
  the cluster; for the owner if rule 70 is read widely.
- **P4, an untested refusal:** `TextPhonemizers.First()` with nothing registered now refuses naming the registry and
  the package; no test covers it (other plug-ins may have registered a phonemizer in the shared test process).
- **P4, local folders:** `samples/IdrakAudio`, `samples/VitsSpeech`, `samples/WhisperSpeech/Idrak.WhisperSpeech` and
  `samples/OpenVoice/Idrak.OpenVoice` hold only untracked `obj`/`bin` output on a machine that built them; the owner
  may delete them locally.
- **P5, to confirm:** ArabicOcr's model folder is now `config.json` + `model.safetensors` (positional tensor names
  `layers.{i}.parameters.{j}`) so `idrak predict` and `idrak train` read it; old `recognizer.ikm` folders still load.
  The tool registers Qwen-VL, LLaVA, IdrakVision and CRNN at startup (`Idrak.QwenVision.dll`, `Idrak.IdrakVision.dll`
  gone; LLaVA, test-only before, now ships). A plug-in's `Register()` runs once until `Unregister()`.
- **P5, ArabicOcr training not reproducible with one seed** (epoch-1 loss 2.78466, then 2.837; likely a CPU reduction
  order): blocks proving training unchanged (rules 71, 86) and makes the CER thresholds fragile.
- **P5, the CRNN's backward LSTM reads a batch's right padding** (`ModelLineReader.Batch`, the LSTM layer): readings
  change with batch makeup; a mask would allow bigger batches and the image family suite's batch check.
- **P5, two copies of batch measuring:** `MeasuredLineBatches` (plug-in) and the CLI's internal one in
  `ImageModelSupport`; a candidate for one library contract (decision 9).
- **P5, slowness:** `CrnnRecognizer.Read` prepares lines one after another (up to 256 pages' lines at once; rule 79);
  `CrnnFamily.Read` builds the network with random weights, then overwrites them all.
- **P5, memory:** the public `SafeTensorsWriter.Write` holds every tensor's values at once; the streaming overload is
  internal.
- **P5, format question:** the CRNN's tensor names are positional, so checkpoints from other tools cannot map to it.
- **P5, local folders:** `samples/QwenVision/Idrak.QwenVision` and `samples/IdrakVision/Idrak.IdrakVision` hold only
  ignored `bin`/`obj` on clones that built them.
- **P6, to confirm:** a library user needs `Idrak.Text.Plugins` for any Gemma model (and `Idrak.Vision.Plugins` for
  Gemma 3 with images); `PretrainedFamilies.GemmaTensorName` → `GemmaArchitectures.TensorName`; the `Idrak.Gemma3Vision`
  assembly → `Idrak.Vision.Plugins.Gemma`; `Gemma3VisionPlugin.Register()` registers once and also the text family when
  missing; the tool registers Gemma 3 vision at startup; an app that registers over a Gemma name and unregisters gets
  nothing back (no library default).
- **P6, no GGUF mapping for Gemma** (`src/Idrak/Models/GgufModel.cs`): `gemma`, `gemma2`, `gemma3` GGUF files were and
  are refused; a mapping goes through `GgufArchitectures.Register` from the plug-in (llama.cpp's names for the post-norms,
  whether its norm weights already include the +1, a fixture).
- **P6, `idrak tune init` picks the vision example only with `-P`** (`src/Idrak.Cli/Commands/Train/TuneCommand.cs`, near
  line 340): with Gemma 3 vision built in, `tune init -b <a Gemma 3 vision checkpoint>` without `-P` writes the text one.
- **P6, message wording:** `PretrainedArchitectures`' refusal says "No architecture 'X' is registered", not "not
  registered" like the other registries (`src/Idrak/Models/Abstractions/PretrainedArchitectures.cs`).
- **P6, model names in moved code:** the moved Gemma code keeps the model names already in its comments and test strings
  (the 4B config test in `VisionLanguageModelTests`, a "0.83 GB less" comment in `Gemma3Vision.cs`).
- **P6, local folders:** `samples/Gemma3Vision/Idrak.Gemma3Vision/{obj,bin}` and an old `Idrak.Gemma3Vision.dll` in the
  test output on machines that built them.
- Later phases add theirs here.

## Order and parallelism

At most two agents at a time (decision 10), each in its own worktree; a merge is built and its filters run before the
push. The longest chain is P0 → P8 → P9 → P10, so the engine (lane B) starts right after the rename, and the moves
(lane A) run beside it:

| Step | Lane A: contracts, defaults and moves | Lane B: engine and new families |
|---|---|---|
| 1 | P0, the rename (alone: it touches every package) | none |
| 2 | P1, the three package skeletons | P8, the engine, starts |
| 3 | P2, generic components, core | P8, continued |
| 4 | P3, generic components, vision | P8, continued |
| 5 | P4, audio plug-ins | P9, BERT (after P8) |
| 6 | P5, vision plug-ins (after P3) | P10, T5 |
| 7 | P6, Gemma (after P3) | the next ready step of lane A, if any |
| 8 | P7, the other text families (after P6: both edit `Architectures.cs`) | none |

Why the lanes do not collide:
- P1-P5 add packages, contracts and defaults and move sample code; P8 works in the attention layers, the encoder and
  seq2seq contracts and generation, and touches no family. Where both touch one file (G6's recipes and P8's attention
  options in `src/Idrak/Layers/NetworkBuilder.cs`), the later merge takes the earlier one's change first.
- P6 and P7 both edit `src/Idrak/Models/Architectures.cs`, so they stay in lane A, one after the other.
- P9 and P10 add to `Idrak.Text.Plugins` in their own namespaces; P6 and P7 add other namespaces there.
- Files both lanes touch (`api/*.txt`, the abstraction inventory, the test list in `tests/Idrak.Tests/Program.cs`,
  `src/Idrak.Cli/Shared/EnvironmentVariables.cs`) are merged one lane at a time; the generated dumps are regenerated,
  never merged by hand.

A step is a dependency order, not a barrier: when a lane finishes, it takes its next ready step without waiting for the
other; when lane B is done early, it takes lane A's next step whose inputs are merged. No step starts until the owner
has asked for this plan to go on.

## The owner's runs

Each phase's as-built note gives the exact PowerShell list (from `D:\Projects\Idrak`): one
`dotnet build -c Release tests\Idrak.Tests`, the phase's CPU filters, then its CUDA filters with `--no-build`. P0 and P7
also ask for the full suite, because they reach every package. P9 and P10 add the real models' references.

## Risks

- **Internals in the moved text families.** The families in `src/Idrak` use internal helpers; each must become a public,
  generic member or be rewritten on public parts. The API dump shows every member opened.
- **The API breaks for consumers:** namespaces change (`Idrak.Nlp` → `Idrak.Text`, families → `…Plugins.*`), and a
  library user who loaded a Llama or Qwen model with nothing referenced must now reference `Idrak.Text.Plugins`. The
  "not registered" message names it. No version bump (CLAUDE.md); the owner decides when to release.
- **Silent loss of registrations** in the rename (the library defaults list, `DynamicDependency`): checked by the
  inventory test and by `pretrained:` in P0.
- **Plug-ins that depend on each other** (vision on text): one way only; a cycle is refused at review.
- **Encoder and seq2seq engine scope (P8)** is the largest new work; BERT and T5 wait for it.
- **New contracts grow the public API** (G4, G5, G6, G11, G12, G13, G14's extension): each with its suite and inventory
  line; the samples' copies are removed in the same phase, so no copy outlives its contract.

## Runs

| Date | Phase | Machine | What | Result |
|---|---|---|---|---|
| 2026-10-20 | P0 | CPU (cloud, 4 threads) | the P0 filters above, the three sample test projects | 158 passed, 1 failed (naming test matched itself), then 3 of 3; samples 4, 14, 12 passed |
| 2026-10-20 | P1 merged | CPU (cloud, 4 threads) | the P1 filters (merge on c023893) | 133 passed, 0 failed |
| 2026-10-20 | P8 merged (with P1's leftover message) | CPU (cloud, 4 threads) | the P1 and P8 filters together | 287 passed, 0 failed |
| 2026-10-20 | P2 merged (on P8) | CPU (cloud, 4 threads) | the P2 filters with engine, decoder, sampler; ArabicOcr and LegalOcr tests | 249 passed, 0 failed; 14 and 12 passed |
| 2026-10-20 | P9 merged (on P2) | CPU (cloud, 4 threads) | the P9 filters with the guards | 150 passed, 0 failed |
| 2026-10-20 | P3 merged (on P9) | CPU (cloud, 4 threads) | the P3 filters with engine; ArabicOcr tests | 190 passed, 0 failed; 14 passed |
| 2026-10-20 | P10 merged (on P3) | CPU (cloud, 4 threads) | the P10 filters with generic components, override loop and the guards | 178 passed, 0 failed |
| 2026-10-20 | P4 | CPU (cloud, 4 threads) | the nine audio groups, the guards, engine, cli health, generic components, pytorch checkpoint, audio cleaning | 157 passed, 2 failed, then 16 of 16; the nine audio groups 64 before and after |
| 2026-10-20 | P4 merged (on P10) | CPU (cloud, 4 threads) | the nine audio groups, audio cleaning, the guards, engine, cli health, cli run, generic components, pytorch checkpoint | 176 passed, 0 failed |
| 2026-10-20 | P5 | CPU (cloud, 4 threads) | the P5 filters (qwen vl, vision contracts and tuning, image families, cli vision and images, plugins, vlm scratch, pretrain and distill vision, generic components, image prefill, outside plug-in, the guards, cli health, models, run, pretrained); ArabicOcr and LegalOcr tests | 192 passed, 2 failed, then 2 of 2; final 194 passed, 0 failed; 14 and 12 passed |
| 2026-10-20 | P5 merged (on P4) | CPU (cloud, 4 threads) | the P5 and P4 filters with the guards, engine, cli run, cli models, pretrained; ArabicOcr and LegalOcr tests | 278 passed, 0 failed; 14 and 12 passed |
| 2026-10-20 | P6 | CPU (cloud, 4 threads) | the Gemma groups: pretrained, gguf, plugins, chat templates, cli run, cli models, the guards, engine, generic components, vision, tuning, image, CLI image and serve, decoder, pretrain, tokenizers; ArabicOcr and LegalOcr tests | 316 passed, 0 failed (315 before, + `plugins: gemma`); 14 and 12 passed, as before |
| 2026-10-20 | P6 merged (on P5) | CPU (cloud, 4 threads) | the P6 and P5 filters with the guards; ArabicOcr and LegalOcr tests | 282 passed, 1 failed (P5's `plugins: vision` expected four plug-ins; Gemma 3 vision is the fifth), fixed: `plugins:` and the guards 43 passed, 0 failed; 14 and 12 passed |
| 2026-10-20 | P7 | CPU (cloud, 4 threads) | the P7 groups (pretrained, gguf, decoder, rope scaling, chat templates, fine-tuning, cli run, cli serve, tokenizers with the BPE baseline, plugins, mixture of experts, generation, coding tools, cli models), the guards and every group loading a moved family | 178 before; after one build 467 passed, 0 failed (the P7 groups the same counts, plugins 25 → 30 with the five new); inventory column fix, then 6 of 6 |
