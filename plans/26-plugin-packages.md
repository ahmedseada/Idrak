# Plan 26: model families as plug-in packages, and `Idrak.Nlp` renamed `Idrak.Text`

**Status (2026-10-20, `abstraction`):** planned 2026-10-19 from the owner's decisions in the Plan 24 session, against the
code at 63f8893, and widened 2026-10-20 to every reusable piece outside the library (a read of the code of every
package, sample and test project, not of the plans: every "Comes from" below was found in the source). Nothing is
built. No phase starts until the owner names it.

| Phase | What | Lane | State |
|---|---|---|---|
| P0 | `Idrak.Nlp` renamed `Idrak.Text` (project, assembly, package, namespaces, every reference) | A, alone | Done 2026-10-20; CPU pass |
| P1 | The three plug-in packages: `Idrak.Text.Plugins`, `Idrak.Vision.Plugins`, `Idrak.Audio.Plugins` (skeletons, registration, the CLI) | A | Done 2026-10-20; CPU pass (2 leftovers, see its note) |
| P2 | Generic components into the library, core part (contracts first): tensor store, timing probe, character tokenizer, progress, keyword index, network recipes, JSON numbers | A | Not started |
| P3 | Generic components into the library, vision part (contracts first): image-block rule, noise augmentation, PGM, line segmentation, text matching, reading order, the line-recognition task | A | Not started |
| P4 | Audio plug-ins move in: Whisper, VITS (with uroman), OpenVoice, IdrakAudio, espeak-ng | A | Not started |
| P5 | Vision plug-ins move in: Qwen-VL, LLaVA, IdrakVision, the CRNN line recognizer | A | Not started |
| P6 | Gemma out of the library (text, the vision-language text decoder, the Gemma 3 vision plug-in) | A | Not started |
| P7 | The other text families out of the library: Llama, Mistral, Qwen, DeepSeek, Harmony | A | Not started |
| P8 | The engine for encoders and encoder-decoders (contracts, attention options), with no family in it | B | Not started |
| P9 | BERT (`Idrak.Text.Plugins.Bert`) | B | Not started |
| P10 | T5 (`Idrak.Text.Plugins.T5`) | B | Not started |

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

## Phase P3: generic components, vision part (G8-G14)

- New contracts first (G11 line segmentation, G12 text matching, G13 reading order, and G14's extension of
  `IImageModelFamily`), then the defaults, then ArabicOcr and the Ocr sample on them.
- G8 lands before P5 and P6 (IdrakVision and Gemma 3 use it); G14 lands before P5 (the CRNN family needs it).

**Done when:** the suites pass; ArabicOcr's tests (`samples/Idrak.Samples.ArabicOcr.Tests`) give the same lines, matches
and readings; the vision tuning and Gemma groups pass unchanged.

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

## Phase P5: the vision plug-ins

- Qwen-VL and IdrakVision from `samples/`, LLaVA from `tests/Idrak.PluginTests/Llava.cs`, into `Idrak.Vision.Plugins`,
  on G1 and G8. `Idrak.PluginTests` keeps its proof families; LLaVA's place there is taken by the package itself, which
  has no internals either.
- The CRNN line recognizer from ArabicOcr's `Recognizer` into `Idrak.Vision.Plugins.Crnn`, on G14; ArabicOcr becomes an
  app that references it.
- Plans 16 and 19's paths are updated in their "as built" notes.

**Done when:** `qwen vl:`, `vision contracts`, `vision tuning contracts`, `vision tuning reference`, IdrakVision's groups
and ArabicOcr's tests pass unchanged, and `samples/` holds only apps.

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

## Phase P9: BERT (`Idrak.Text.Plugins.Bert`)

- `BertModel` on P8's encoder contract: word, position and token-type embeddings, post-norm blocks with GELU, the
  pooler; `BertForMaskedLM`, `BertForSequenceClassification`, and sentence embeddings (mean or first-token pooling) through
  `TextEncoder`. The tokenizer is Plan 24's WordPiece (`tokenizer.json` or `vocab.txt`).
- Fixtures: a tiny random BERT and its outputs from transformers (pinned, a script under `tools/`), committed small;
  real models (`google-bert/bert-base-uncased`, `aubmindlab/bert-base-arabertv02`) are the owner's run.

**Done when:** hidden states, pooled vectors and logits equal transformers' within float tolerance, batched with padding
equal to one by one.

## Phase P10: T5 (`Idrak.Text.Plugins.T5`)

- `T5ForConditionalGeneration` on P8's encoder-decoder contract: RMS norms, the bucketed relative position bias of the
  first layer shared by the rest, no score scaling, ReLU (T5 v1.0) or gated GELU (v1.1, mT5) feed-forward, tied or untied
  head. The tokenizer is Plan 24's Unigram (`tokenizer.json` or `spiece.model`).
- Fixtures as P9's; real models (`google-t5/t5-small`, `google/mt5-small`) are the owner's run.

**Done when:** encoder states and greedy generations equal transformers' on the fixtures, and the KV-cached decode equals
the uncached one.

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
