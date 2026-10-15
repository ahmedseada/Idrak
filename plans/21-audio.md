# Plan 21: Idrak.Audio (sound in and out: codecs, features, speech recognition and synthesis)

**Status:** planned 2026-10-15 against the code at `abstraction` 82d84937; S0 (the reference fixture) and S1 (AudioData, the codec and encoder registries, the WAV codec and encoder) built the same day, CPU-checked; S1's FLAC decoder (decision 6) built after, its fixture in flac/ from libFLAC through soundfile, "audio codecs: flac:" passing on the CPU (ten minutes of 16-bit FLAC in about 130 ms). **Built and CPU-checked 2026-10-15 (after the 09:25 UTC snapshot):** S2 (Resampler, Stft host FFT and composed path, MelFilterbank, LogMelFeatures, AudioFeatureOptions/Extractors, AudioFeatureSuite, "audio features:" tests); S3 (ChatAudio part and AudioPartKind, SoftParts contract, the tuning encoder and readers with "audios" lists, "audio chat:" tests); S5 (SpeechSynthesis and Vocoders contracts, Stft.Inverse, GriffinLim, samples/VitsSpeech, "audio tts:" tests). IDRAK_FILTER="audio" passes 19 of 19 on the CPU. S4 is written but has no tests yet (CrossAttention, the DecoderBlock cross block, EncoderDecoderSpec, BeamSearch, the SpeechRecognition contracts, samples/WhisperSpeech): next is its "audio asr:" tests against tiny-whisper/ and ctc/, then S6 and S7. The package name
`Idrak.Audio` is reserved in the library (`src/Idrak.Abstraction/LibraryDefaults.cs:29-30`, with a
`DynamicDependency` on `Idrak.Audio.LibraryRegistrations` at line 43), so the day the assembly exists its
registrations are found without a change to Idrak.Abstraction.

**Goal.** Audio files in and out of the library without a platform codec or an outside program (rule 70), the
features speech models read (resampling, STFT, mel) as operations that run on every device with host fallbacks,
speech recognition and speech synthesis through contracts and registries with every family a plug-in, an "audio" part
in chat messages for audio-language models, and a from-scratch audio family (IdrakAudio) made by `idrak create`,
trained on the owner's recordings with CTC or an encoder-decoder, and reloaded through the same contracts a loaded
model uses, so `idrak transcribe`, `speak`, `run`, `chat`, `serve`, `tune` and `pretrain` treat the two alike.

**Rules that hold for every phase** (`CLAUDE.md`, `docs/optimization.md`):

- **Speed and memory first.** Bytes parsed, not strings (rule 74: a WAV header is read with `BinaryPrimitives`, samples
  converted in one pass from a pooled block); memory walked in order (72: frames are rows, mels are columns, every
  pass goes row by row); one pass (73: framing, windowing and the power spectrum do not revisit samples); work done
  once (75: the window, the DFT basis and the mel filterbank are built once per shape and device and cached; a
  utterance's encoder output and the cross-attention keys and values are computed once per segment); cold and warm
  measured apart (76); device memory that does not grow across utterances (77: feature and encoder buffers are reused
  per batch shape); the library's telemetry on every stage (78); independent work spread and checked (79).
- **Fixed summation order** (rule 86): the DFT and the mel projection are matrix products, whose order is fixed by the
  product's tiling; the host STFT sums each bin in frame order. A later fused kernel is checked against the composed
  path (rule 71).
- **Card-agnostic.** No sample rate, segment length, batch or buffer size is tied to a card or a GB budget: segment
  batches are measured on the running device and halved on `ResourceLimitExceededException`, as `idrak predict` and the
  ArabicOcr sample do. The numbers below (16 kHz, 400, 160, 80 mels, 30 s) are the fixture's and a family's
  configuration values, read from its files, never library constants.
- **Abstraction (decision 10 of plan 10).** Every new behaviour is a contract with a `SlotTable` registry. A contract
  only `Idrak.Audio` uses lives in `Idrak.Audio.Abstractions`; one that Abstraction itself or two packages use lives in
  `Idrak.Abstraction`. The inventory test and the API guard are regenerated when they change.
- **No family in the library.** The package registers no family: an unregistered architecture fails with the
  registry's "is not registered" message and the CLI's "-P" hint. Families are plug-ins under `samples/`.
- **Nothing fixed.** Every window, hop, rate, chunk, beam, threshold and path is an option on the library's option
  records, carried to the CLI and its JSON files; an environment variable only where the library already reads one for
  the same kind of choice (`IDRAK_STFT_PATH` beside `IDRAK_ATTENTION_PATH`, see S2).
- **Tests.** Targeted CPU runs only (`IDRAK_DEVICES=cpu IDRAK_FILTER=...`); CUDA and Vulkan are the owner's, with the
  exact commands at the end of each phase. The fix workflow of CLAUDE.md applies to each phase.

## Separate package or plug-in sample?

**Decision: `src/Idrak.Audio` is a package; every model family is a plug-in sample.** The repository's own rules decide
it:

- CLAUDE.md splits the library into general machinery behind contracts and `SlotTable` registries, and families that
  "live in plug-ins or apps, never as library defaults". Codecs, resampling, STFT and mel, an audio part kind, the
  recognition and synthesis contracts and a CTC trainer are family-agnostic machinery; a Whisper-style loader or a VITS
  synthesizer is a family.
- `plans/plugins/README.md` reserves plug-in packages for "a dependency the library must not have"; audio needs none
  (WAV, FLAC, the DFT and the mel filterbank are plain C#), so the dependency rule does not push audio out of the
  library.
- Plan 13 put the image building blocks in `Idrak.Vision` and the families in plug-ins; plan 19 put `IVisionFamily.Create`
  and the pretraining contracts in the library and the IdrakVision family in `samples/IdrakVision`. Audio follows the
  same split: `IAudioFamily.Create`, `ISpeechFamily`, `ISpeechSynthesisFamily` and their registries in the package,
  `samples/WhisperSpeech`, `samples/VitsSpeech` and `samples/IdrakAudio` as plug-ins.
- `LibraryDefaults.cs:29-30` already names `Idrak.Audio` among the first-party assemblies that register built-ins
  ("planned packages are listed so they work the day they appear"), beside `Idrak.Imaging`, `Idrak.Vision` and
  `Idrak.Nlp`. The owner reserved a package, not a sample.
- Decision 8 of plan 10 (the package layout: "domains as large packages") names audio as a domain of its own; a sample
  cannot register into `ChatParts`, `SampleSources` or `PretrainingSources` as a library default, which the audio part
  and the audio data sources must be so that `idrak` and the apps see them without `-P`.

What goes where is in "Where each part lives" below. The package references `Idrak` (core) and `Idrak.Nlp` (the chat
transcript encoder, the tuning data formats and the pretraining sources it registers into); `Idrak.Cli` references it
as it references `Idrak.Vision` (`src/Idrak.Cli/Idrak.Cli.csproj:27`).

## What exists today

Every row names the code at 82d84937. A grep of `src/` and `samples/` for audio, wav, mel, spectrogram, stft, fft,
whisper and resample finds image resampling only (`src/Idrak.Imaging/ImagePreprocessor.cs`, the `Interpolate2d` and
`ResizeNormalize` operations), the `AudioContentBlock` of the MCP bridge, and CTC in the ArabicOcr sample.

| Part | Today |
|---|---|
| The package | **missing**: no `src/Idrak.Audio`; the assembly name and its `LibraryRegistrations` hook are reserved (`src/Idrak.Abstraction/LibraryDefaults.cs:29-30`, `:43`) |
| Audio files | **missing**: no codec, no `AudioData`; the MCP bridge flattens an audio block to the text `[audio {mime}]` (`src/Idrak.Mcp/McpTools.cs:100`) |
| The pattern to mirror | `IImageCodec` and `ImageCodecs` (`src/Idrak.Abstraction/Data/ImageCodecs.cs:20`, `:50`: a `SlotTable` asked newest first, `ReadInfo` from the first 64 KiB, `Decode` of pooled bytes, built-ins registered through `LibraryDefaults.Ensure` at `:60`), `ImageData` (`ImageData.cs:16`: float planes made from 8-bit planes on first read, `Resize` with precomputed taps), `IImageEncoder` / `ImageEncoders` (`ImageEncoders.cs:12`, `:33`) |
| FFT, STFT, mel | **missing**: no Fourier operation in `src/Idrak.Abstraction/Operations/Operations.g.cs`. Present and enough to compose them: `Im2Col` (framing: `Backend.Operations.g.cs:2392`), `BatchedMatMul`, `MatMulBias`, `GemmStrided` (the DFT and the mel projection as products), `Unary` (log, exp), `Binary`, `Pow`, `Clamp`, `AddRowVector`, `SumRows`, `Convolution` (`:2646`, for the polyphase resampler) |
| Convolution layers | `Conv2d` with rectangular kernels, strides, padding, groups and dilation (`src/Idrak/Layers/Convolution.cs:44`), so a Conv1d of kernel k is a Conv2d of (1, k) over [N, C, 1, T]; `ConvTranspose2d` (`:238`) covers the ConvTranspose1d of a HiFi-GAN vocoder; device kernels on CUDA and Vulkan (plan 13, step 1) |
| Encoder layers | `MultiHeadAttention` (`src/Idrak/Layers/Attention.cs:13`), `TransformerEncoderLayer` (`:242`, pre-norm, `FromLayers` to replace its norms and activation), `PositionalEncoding` (`:376`, sinusoidal), `LayerNorm` (`Normalization.cs:264`), `GELU` (`Activations.cs:74`), `LSTM` / `GRU` bidirectional (`Recurrent.cs:13`, `:79`) |
| Decoder layers | `CausalSelfAttention` (`src/Idrak/Layers/Decoder.cs:104`) and `DecoderBlock` (`:835`, constructor at `:845`: attention norm, attention, feed-forward norm, feed-forward). **No cross-attention**: plan 13 (`plans/13-vision.md:59-60`) lists "cross-attention in the decoder (TrOCR, Donut)" as later core work. An encoder-decoder recognizer needs it (S4) |
| Decoding | `TextGenerator` (`src/Idrak/Generation/TextGenerator.cs`; greedy and sampling through `TokenSamplers`, `CreateSampler` at `:72`), `DecodingContext` with per-layer caches (`src/Idrak.Abstraction/Generation/Decoding.cs:164`, `CacheFor(owner, rows, headDim)` at `:661`), recorded steps. **No beam search** for text decoders (a grep for "beam" finds the CTC decoders only) |
| CTC | in **core**, not Idrak.Vision (plan 13 step 3 placed it by decision 10): `Losses.Ctc` (`src/Idrak/Losses.cs:197` on a padded or flat target tensor, `:227` on label lists; PyTorch's `ctc_loss` semantics, mean, sum or none, `zeroInfinity`, `batchFirst`), `Tensor.CtcLoss` (`src/Idrak.Abstraction/Tensors/Tensor.Advanced.cs:754`), the operations `CtcLoss` / `CtcLossBackward` (`Backend.Operations.g.cs:3245`, `:3303`, each with a host fallback), kernels on the CPU (`Devices/Cpu/CpuBackend.Ctc.cs`: log space, double accumulators, sequences in parallel), CUDA (`src/Idrak.Gpu/Cuda/CudaBackend.Ctc.cs`, `PtxKernels.Ctc.cs`: a block per sequence, rows in shared memory when the device's opt-in limit allows, rule 84) and Vulkan (`Vulkan/VulkanBackend.Ctc.cs`, `VulkanKernels.Ctc.cs`); decoding through `CtcDecoders` (`src/Idrak/Inference/Abstractions/CtcDecoders.cs:52`: "greedy" and "beam", a registry with `Register`, `Decode` over a tensor with lengths at `:115`, `CtcDecodeOptions` at `:15`, `CtcHypothesis` at `:12`). The ArabicOcr sample uses all of it (`samples/Idrak.Samples.ArabicOcr/TrainCommand.cs:257`, `Recognizer.cs:312`) |
| Chat parts | `ChatPart` (`src/Idrak.Abstraction/Generation/ChatParts.cs:19`), `ChatImage` (`:42`: bytes as received, a sniffed media type, SHA-256 identity), `IChatPartKind` (`:179`: "a plug-in adds its own (audio, video, documents) with `ChatParts.Register` and makes its own `ChatPart` type"), `ChatParts` (`:203`; `BuiltIn` at `:216` registers text and image directly and does **not** call `LibraryDefaults.Ensure`, unlike `ImageCodecs.cs:60` and `SampleSources.cs:30`), the kit's `ChatPartKindSuite` and `Conformance.CheckChatPartKind` (`src/Idrak.Abstraction.Testing/Conformance.cs:141`) |
| Images into a prompt (the path audio mirrors) | `IImagePromptFormat.Expand` (`src/Idrak.Abstraction/Generation/VisionLanguage.cs:184`, `:204`), `ImageTokenFormat` (`:222`: a marker token becomes begin + N soft tokens + end), `ImageTokenLayout(Tokens)` (`:17`), `ImagePrefill.Locate(ids, imageToken, features)` and `ForwardCached` (`src/Idrak/Layers/ImagePrefill.cs:199`, `:65`: the soft tokens' embeddings replaced by features), `ChatGenerator.Images` and `PartKinds` (`src/Idrak.Nlp/Generation/ChatGenerator.cs:120`, `:128`), `ChatTranscriptEncoder.Tokens` (`src/Idrak.Nlp/FineTuning.cs:418-432`: `vision.Expand` at `:430`, `TrainingImage` blocks), `PretrainedModel.Vision` read through `VisionFamilies.For` (`src/Idrak/Models/PretrainedModel.cs:251`, `:271`). None of it knows the part is an image except by type: the layout, format and prefill work for any soft-token part |
| A family from a spec | `IVisionFamily.Create(JsonObject vision, int textDim, ITokenizer tokenizer, int seed)` and `CreatedVision` (`src/Idrak/Models/Abstractions/VisionFamilies.cs:48`, `:59`), `PretrainedModel.Create` with `CreateOptions.Vision` (`PretrainedModel.cs:94`), `PretrainedArchitecture.Build` and `Config` (plan 19 A1); the IdrakVision plug-in (`samples/IdrakVision/Idrak.IdrakVision`: `Register`, `RegisterIdrakPlugin`, `Unregister` in `IdrakVisionPlugin.cs`) |
| Pretraining | `Pretrainer.Train` (`src/Idrak.Nlp/Pretraining.cs:311`, `:330`), `PretrainingStage` (`:21`), `PretrainingOptions` (`:77`), `PretrainingSources` with "tokens" and "records" (`src/Idrak.Nlp/Abstractions/PretrainingData.cs`; a source's `Images` at `:26`), `TrainingSequence` and `TrainingImage` (`Abstractions/Distillation.cs:14`, `:65`), checkpoints with `Optimizer.SaveState` / `LoadState`, distillation with images (plan 19 B5) |
| Tokenizer trainer | `TokenizerTrainers` "bpe" (`src/Idrak/Models/Abstractions/TokenizerTraining.cs`, `BpeTrainer.cs`), `idrak data tokenizer` |
| Data | `ISampleSource` / `ISampleStream` (`src/Idrak.Abstraction/Data/SampleSource.cs:19`, `:42`), `SampleSources` (`SampleSources.cs:22`), `TuningDataFormats` / `ITuningDataFormat` (`src/Idrak.Nlp/Abstractions/TuningData.cs:136`, `:31`; `TuningDataOptions.Images` at `:17`), `DataFileFormats` in `Idrak.Data` |
| Telemetry | `Telemetry.InferenceStage(stage, device, start, count, bytes)` (`src/Idrak.Abstraction/Diagnostics/Telemetry.cs:304`) with `InferenceStageKind` ImageDecode, ImageTransforms, Encoder, Prefill, Decode (`Events.cs:183`); `Backend.GetMemoryUsage()` and `ResetPeakMemoryUsage()` (`Devices/Backend.cs:623`, `:2202`), `ComputeResources.ResetPeakMemoryUsage(device)` (`ComputeResources.cs:181`); training events `TrainingStarted` / `BatchCompleted` / `TrainingCompleted` |
| Precision | `MixedPrecision.BFloat16()` (`src/Idrak.Abstraction/Tensors/MixedPrecision.cs:36`), `PretrainingOptions.BFloat16Activations`, bfloat16 attention kernels (`AttentionBFloat16`), `EncoderWeights.AsStored` for bfloat16 checkpoints |
| Plug-in kernels | `PluginOperations.Register<TKernel>(name, defaultKernel, fallback)` and `Kernels.Register(operation, deviceKind, kernel)` (`src/Idrak.Abstraction/Operations/PluginOperations.cs:91`, `Kernels.cs:45`): a package declares an operation with a host or composed default and adds a kernel per device kind later; Vulkan takes SPIR-V (`VulkanKernel.Dispatch`), CUDA PTX |
| The minimum backend (plan 7) | 12 core operations; every other operation runs through `HostCall` (copy to the CPU, run its kernel, copy back) until a backend adds a kernel (`plans/7-backends.md:10`), so anything composed from existing operations runs on HIP and any future backend on day one |
| CLI | commands in `TrainCommands.All` / `DataCommands.All` / `RunCommands` (`src/Idrak.Cli/Commands/*.cs`), `--image` on `run` (`Commands/Run/RunCommand.cs:61`) through `ImageInputs.Read` (`Shared/ImageInputs.cs:23`), `MeasuredBatches` (`Commands/Train/ImageModelSupport.cs:183`, internal to the CLI), the `EnvironmentVariables` lists (`Shared/EnvironmentVariables.cs:29-43`), `-P` plug-in loading |

## Where each part lives (decisions 8 and 10 of plan 10)

| Part | Package and namespace | Why |
|---|---|---|
| `AudioData`, `AudioInfo`, `IAudioCodec`, `AudioCodecs`, `IAudioEncoder`, `AudioEncoders`, the WAV codec and encoder, FLAC (decision 6) | `Idrak.Audio.Abstractions` (contracts, registries), `Idrak.Audio` (codecs) | one library user today; moved to `Idrak.Abstraction` the day a second package (Mcp, AspNetCore) needs the type, as the inventory test will say |
| `Resampler`, `Stft`, `MelFilterbank`, `LogMelFeatures`, `AudioFeatureOptions`, `IAudioFeatureExtractor`, `AudioFeatureExtractors` ("log-mel") | `Idrak.Audio`, contracts in `Idrak.Audio.Abstractions` | the feature path families read; composed from core operations |
| A fused `Stft` operation and its kernels (after measuring, S2) | declared by `Idrak.Audio` through `PluginOperations.Register`; kernels in `Idrak.Gpu` (PTX, SPIR-V) registered through `Kernels.Register` under `Idrak.Gpu.LibraryRegistrations` | the GPU package owns kernels; the operation's composed default is the fallback on every device |
| `ChatAudio` (the part), the "audio" `IChatPartKind` | `Idrak.Audio.Abstractions` (the part type), registered into `Idrak.Abstraction`'s `ChatParts` by `Idrak.Audio.LibraryRegistrations` as a library default | ChatParts.cs says a plug-in adds its own kinds; a first-party package does the same through `LibraryDefaults` (one line added to `ChatParts`: `static ChatParts() => LibraryDefaults.Ensure(typeof(ChatParts));`) |
| Soft-token parts in generation and tuning: `ISoftPartSource` (kind, format, attention rule, blocks, features) on `ChatGenerator.Parts` and `ChatTranscriptEncoder.Parts` | `Idrak.Abstraction.Generation` (the contract: Nlp and Audio use it), `Idrak.Nlp` (the wiring beside `Images`) | the image path stays as it is; audio uses the general one; folding images onto it is a backlog item |
| `IAudioFamily` (Read, Create), `AudioCheckpoint`, `PretrainedAudio`, `CreatedAudio`, `AudioFamilies` | `Idrak.Audio.Abstractions` | the audio side of an audio-language checkpoint, read from the model's folder by the package (core is not touched: `PretrainedModel` gets no audio property) |
| `ISpeechRecognizer`, `Transcription`, `TranscriptionOptions`, `ISpeechFamily`, `SpeechFamilies`, `SpeechCheckpoint` | `Idrak.Audio.Abstractions` | recognition as a model kind of its own, beside chat |
| `CrossAttention` layer, `DecoderBlock` with an optional cross block, `EncoderDecoderSpec`, `BeamSearch` (decision 4) | core: `Idrak.Layers`, `Idrak.Abstraction.Generation` (the spec), `Idrak.Generation` | general decoder machinery plan 13 already assigned to core |
| `ISpeechSynthesizer`, `SpeechSynthesisOptions`, `ISpeechSynthesisFamily`, `SpeechSynthesisFamilies`, `IVocoder`, `Vocoders` ("griffin-lim" the library default) | `Idrak.Audio.Abstractions`, the Griffin-Lim vocoder in `Idrak.Audio` | the vocoder is a slot so a mel-producing model speaks without a neural vocoder |
| `ISpeechSource`, `SpeechSources` ("transcripts", "audio-folder"), the `TuningDataFormats` "transcripts" format, the `PretrainingSources` "audio-records" source, the `SampleSources` "audio" source, `CtcTrainer`, `CtcAlignment` | `Idrak.Audio.Abstractions` (own contracts), `Idrak.Audio` (registrations into Nlp's and Abstraction's registries) | training data and the CTC loop are the package's; the token loop is `Pretrainer`'s |
| `idrak transcribe`, `idrak speak`, `idrak data audio`, `--audio` on `run` and `chat`, `create` with `"audio"`, `pretrain --audio` | `Idrak.Cli` (`Commands/Run`, `Commands/Data`, `Commands/Train`) | the surface |
| The Whisper-style family, the VITS-style family, the IdrakAudio family | `samples/WhisperSpeech/Idrak.WhisperSpeech`, `samples/VitsSpeech/Idrak.VitsSpeech`, `samples/IdrakAudio/Idrak.IdrakAudio` | a family is a plug-in, never the library |
| References and fixtures | `tools/audio/make_tiny_audio.py`; `tests/Idrak.Tests/data/audio/` | as `tools/vlm/make_tiny_*.py` and `tests/Idrak.Tests/data/vlm-scratch/` |

## Conventions every phase follows

- **A registry** copies `RandomTextSources` and `ImageCodecs`: a `SlotTable<string, T>` named after the class,
  `StringComparer.OrdinalIgnoreCase` for codecs, features and vocoders, `Ordinal` for architecture names (as
  `VisionFamilies`), `RegisterDefault` inside `Overrides.AsLibraryDefaults` (or `LibraryDefaults.Ensure(typeof(X))` when
  the defaults live in another assembly), `[MethodImpl(NoInlining)] Register` with `Assembly.GetCallingAssembly()`,
  `Unregister`, `Names`, `Find`, `Get`, `Default`, `Origin`, `SetPolicy` where two implementations can answer one call
  (codecs, feature extractors, vocoders: guarded), `unguarded:` with the reason where they cannot (families).
  `Get`'s message: `"{Kind} '{name}' is not registered (registered: {list}); add it with {Registry}.Register."`.
- **A contract suite** per registry in `src/Idrak.Abstraction.Testing/Contracts/` (`AudioCodecSuite`,
  `AudioFeatureSuite`, `SpeechRecognizerSuite`, `VocoderSuite`) with a `Conformance.CheckX` wrapper.
- **Tests** are entries of a `(string Name, Action<Device> Run)[]` group in a `Tests` partial file added to `Tests.All`
  (`tests/Idrak.Tests/Program.cs:234`); names start with the group prefix (`"audio codecs:"`, `"audio features:"`,
  `"audio chat:"`, `"audio asr:"`, `"audio tts:"`, `"audio train:"`, `"audio cli:"`), so `IDRAK_FILTER=audio` runs them
  all; fixtures through `TestData("audio/...")`; registries left as found (`Unregister` in `finally`).
- **Fixtures** come from `tools/audio/make_tiny_audio.py` whose reruns write the same bytes (`torch.manual_seed`, one
  thread, deterministic algorithms, sorted JSON through `make_tiny.dump_json`), with a `manifest.json` (versions, sizes,
  SHA-256) and a hand-written README. Tolerances: samples exact for PCM, within 1e-6 for float; resampled samples
  within 1e-5; STFT magnitudes and log-mel within 1e-4 of the largest value (float32 products against torch's FFT);
  logits within 1e-4; CTC losses within 1e-4; ids exact.
- **A command** follows plan 19's conventions (`Command` with `Name`, `Summary` and its Arabic entry, `Usage`,
  `ValueOptions`, `Flags`, `ShortForms`, `UsageException`, `ProgressLine`, `ExitCodes`); added to
  `EnvironmentVariables` (`OnDevice`, `Models`, `Data`); the "cli polish", "cli health", "cli arabic" and "cli train
  data" tests check it.
- **Progress and telemetry.** Inference stages through `Telemetry.InferenceStage` with three new
  `InferenceStageKind` values (`AudioDecode`: count samples, bytes the file; `Resample`: count samples out; `Features`:
  count frames, bytes the features) and the existing `Encoder`, `Prefill`, `Decode`; training through
  `TrainingStarted` / `BatchCompleted` / `TrainingCompleted`. Device peak bytes from `GetMemoryUsage()` after
  `ResetPeakMemoryUsage()` in every `-v` report. Free when nobody listens (one static read and an AND).
- **Regenerate** after a public change: the inventory
  (`IDRAK_UPDATE_INVENTORY=1 IDRAK_DEVICES=cpu IDRAK_FILTER=inventory dotnet run -c Release --project tests/Idrak.Tests`)
  and the API dumps (`IDRAK_UPDATE_API=1 IDRAK_DEVICES=cpu IDRAK_FILTER="public API" ...`), a new `api/Idrak.Audio.txt`
  among them.

## Memory and speed, where the bytes go

Sizes below are formulas over a family's configuration (rate R, window N, hop H, mels M, encoder width D, layers L,
heads A), never numbers tied to a card. With the fixture's Whisper-style values (R 16000, N 400, H 160, M 80, a 30 s
segment: T = 3000 frames, T/2 = 1500 encoder rows) the formulas are written out so the owner can check a run.

| Where | Bytes | How it is kept small |
|---|---|---|
| Decoded samples | 4 × R × seconds per channel (1.9 MB for 30 s) | one pooled block per file; a 16-bit view (`AudioData.TryGetSamples16`) stays as bytes until a float read, as `ImageData`'s 8-bit planes do; channels mixed to one on the host in one pass |
| Framing (`Im2Col`) | 4 × T × N (4.8 MB) | per segment, reused across segments of one batch shape (rule 77); a fused STFT kernel removes it (S2, after measuring) |
| STFT basis, window, mel filterbank | 4 × N × (N + 2) (640 KB), 4 × N, 4 × (N/2 + 1) × M | built once per (N, M, R, device) and cached per device (rule 75); double-precision construction, float32 storage |
| Spectrum and log-mel | 4 × T × (N + 2), 4 × T × M (3.9 MB, 960 KB) | the power spectrum is written over the product's output in place; the features are the only tensor kept per segment |
| Encoder activations | 4 × (T/2) × D per layer step (3 MB at D 512), bfloat16 halves it where `MixedPrecision.BFloat16` applies | attention over T/2 rows goes through the tiled kernels (`AttentionTiled`), never a [A, T/2, T/2] score matrix (that would be 4 × A × (T/2)² = 72 MB at A 8); checkpointed blocks in training (`ForwardCheckpointed`) |
| Cross-attention keys and values | 2 × 4 × L × (T/2) × D per segment (12 MB at L 6, D 512) | computed once per segment (rule 75), held in the decoder's `DecodingContext` through `CacheFor(owner, rows, headDim)`, released with the segment; bfloat16 through the cache layouts that have it |
| Decoder KV cache | 2 × 4 × L × P × D for P positions (the family's maximum, 448 in the fixture) | the existing cache formats (float32, int8, bfloat16) and recorded steps apply unchanged; beam search keeps B rows of the cache, B a setting |
| Batches of segments | all of the above × B | B measured on the running device (first one segment, then doubled while the peak stays under the fraction of free memory the options give, halved on `ResourceLimitExceededException`), kept per shape in the run; `TranscriptionOptions.MaxBatch` caps it |
| Streaming | one chunk of `ChunkSeconds` with `OverlapSeconds` at a time | the encoder runs per chunk; the decoder's prompt carries the previous chunk's text (a setting, `CarryText`); memory does not grow with the file's length |

Measured, not assumed: every phase reports cold and warm times per stage (rule 76) and the peak device bytes, and S8
records them in the plan's "as built" section as plans 11 to 19 did.

---

## Phase S0: the reference fixture

**What.** `tools/audio/make_tiny_audio.py` (imports `dump_json`, `save_npy`, `sha256`, `safetensors_index` from
`tools/vlm/make_tiny.py`; needs torch, torchaudio, transformers; no download) writes `tests/Idrak.Tests/data/audio/`:

- `wav/`: generated signals (a 440 Hz tone with a sweep and a seeded noise burst, 1.5 s) as `pcm8.wav`, `pcm16.wav`,
  `pcm24.wav`, `pcm32.wav`, `float32.wav`, `float64.wav`, `stereo16.wav`, `extensible16.wav` (WAVE_FORMAT_EXTENSIBLE
  with a channel mask), `list-chunk16.wav` (a LIST/INFO chunk before `data`, an odd-sized chunk with its pad byte),
  `rate8k.wav`, `rate44k.wav`; the samples of each as `.npy` (float32, [channels, samples]) so decoding is checked value
  for value; `bad-truncated.wav`, `bad-no-fmt.wav` for the refusals.
- `resample/`: `rate44k.wav` and `rate8k.wav` resampled to 16 kHz with `torchaudio.functional.resample(waveform, orig,
  16000, lowpass_filter_width=6, rolloff=0.99, resampling_method="sinc_interp_hann")`, as `.npy`, plus the parameters in
  `resample.json`.
- `features/`: for `pcm16.wav` resampled to 16 kHz: `stft-magnitude.npy` (`torch.stft(n_fft=400, hop_length=160,
  window=torch.hann_window(400, periodic=True), center=True, pad_mode="reflect", return_complex=True).abs()`), the
  mel filterbank (`torchaudio.functional.melscale_fbanks(201, 0, 8000, 80, 16000, norm="slaney", mel_scale="slaney")`,
  the Whisper filterbank), `log-mel.npy` computed as transformers' `WhisperFeatureExtractor` does (power, mel, log10
  clamped at 1e-10, max minus 8 floor, (x + 4) / 4) on a 30 s padded segment, and `features.json` with every
  parameter (n_fft, hop, window, mels, fmin, fmax, padding seconds, the normalisation).
- `tiny-whisper/`: `WhisperForConditionalGeneration` with `d_model 32, encoder_layers 2, decoder_layers 2,
  encoder_attention_heads 4, decoder_attention_heads 4, encoder_ffn_dim 64, decoder_ffn_dim 64, num_mel_bins 80,
  max_source_positions 1500, max_target_positions 448, vocab_size 51865` cut to a 600-token vocabulary with the
  special tokens kept (`<|startoftranscript|>`, `<|en|>`, `<|ar|>`, `<|transcribe|>`, `<|translate|>`,
  `<|notimestamps|>`, the timestamp tokens), weights from `torch.Generator().manual_seed(2028)` (normal, std 0.02;
  norms 1), `save_pretrained(safe_serialization=True)` with the tokenizer, `preprocessor_config.json` and
  `generation_config.json`; `reference/`: `encoder-output.npy` for `log-mel.npy`, `logits-step1.npy` for the prompt
  `<|startoftranscript|><|ar|><|transcribe|><|notimestamps|>`, `greedy-ids.json` for 8 greedy steps, `beam4-ids.json`
  for a beam of 4 (transformers' `generate(num_beams=4, length_penalty=1.0, early_stopping=True)`), `timestamps-ids.json`
  for a run with timestamps on.
- `ctc/`: a tiny encoder (two Conv2d (1, 3) stride (1, 2) + GELU, 2 `TransformerEncoderLayer`s, width 32, a linear head
  to 40 classes with blank 0) initialised with seed 2029, 4 utterances of different lengths with labels, `loss.json`
  (`torch.nn.functional.ctc_loss` mean, zero_infinity), `grads-step1.safetensors`, 3 SGD steps' losses, and the
  Viterbi alignment of each utterance (`ctc/alignment.json`) from the same forward.
- `tiny-vits/`: a `VitsModel` with `hidden_size 32, num_hidden_layers 2, num_attention_heads 2, ffn_dim 64, flow_size
  32, prior_encoder_num_flows 1, posterior_encoder_num_wavenet_layers 2, upsample_rates [4, 4], upsample_kernel_sizes
  [8, 8], resblock_kernel_sizes [3], resblock_dilation_sizes [[1, 3]], spectrogram_bins 65, speaker_embedding_size 0`,
  seed 2030, a 20-symbol `vocab.json`, `reference/waveform.npy` for a fixed input id sequence with
  `noise_scale 0, noise_scale_duration 0` (so the output is deterministic) and `durations.npy`.
- `transcripts/`: `train.jsonl` of `{"audio": "wav/pcm16.wav", "text": "..."}` rows (the fixture's files repeated with
  different texts), the same as a ShareGPT-with-audio file, and `tuning.json` with the expanded prompt, ids, trained mask
  and audio blocks for the audio-language check of S3.
- `manifest.json`, `README.md` (versions, what the script did, tolerances, how to regenerate).

**Done when** the folder exists, two runs write the same bytes, the README names the versions and the commands.

## Phase S1: AudioData and codecs

**Contracts** (`src/Idrak.Audio/Abstractions/AudioCodecs.cs`, mirroring `ImageCodecs.cs` member for member):

```csharp
public readonly record struct AudioInfo(int SampleRate, int Channels, long Samples, int BitsPerSample, bool IsFloat, string Format);

public interface IAudioCodec
{
    string Name { get; }                                   // "wav" (names ignore case)
    IReadOnlyCollection<string> Extensions { get; }        // ".wav", ".wave"
    AudioInfo? ReadInfo(ReadOnlySpan<byte> header);        // from the first AudioCodecs.HeaderBytes, or null when not this format
    AudioData Decode(ReadOnlySpan<byte> file);             // InvalidDataException for a damaged file, NotSupportedException for a variant not read
}

public interface IAudioEncoder
{
    string Name { get; }                                   // "wav"
    string Extension { get; }
    void Encode(AudioData audio, Stream output, AudioEncodeOptions? options = null);   // options: BitsPerSample (16), Float (false)
}

public static class AudioCodecs      // HeaderBytes 64 KiB; Register, Unregister, Names, Extensions, Get, Default, Origin, SetPolicy,
                                      // CanDecode(path), ReadInfo(path | span), Decode(path | span), Load(path, sampleRate, channels)
public static class AudioEncoders    // Register, Unregister, Names, Get, Default, Origin, SetPolicy, Encode(audio, path, options)
```

`AudioData` (`src/Idrak.Audio/Abstractions/AudioData.cs`): planar float32 samples in [-1, 1], `[Channels, Samples]`,
`SampleRate`, `Channels`, `Samples`, `Duration`; a constructor over 16-bit interleaved bytes with the byte order and the
scale, whose float samples are made on the first read of `Samples` (as `ImageData.Pixels` from planes, in one pass per
channel, `HostParallel.For` over blocks), and `TryGetSamples16(out ReadOnlyMemory<byte>)` for a path that wants the
integers (a 16-bit WAV writer copies them without a float round trip; rule 75). `ToMono(Span<float>)` (the mean of the
channels, one pass), `Slice(fromSample, count)` (a view, no copy), `Pad(samples)` into a caller's span.

**The WAV codec** (`src/Idrak.Audio/WavCodec.cs`): RIFF/WAVE chunks walked in order with `BinaryPrimitives` (rule 74,
no strings: chunk ids compared as `"fmt "u8`, `"data"u8`, odd chunk sizes padded), `WAVE_FORMAT_PCM` 8 (unsigned), 16,
24 and 32 bits, `WAVE_FORMAT_IEEE_FLOAT` 32 and 64, `WAVE_FORMAT_EXTENSIBLE` with the PCM and float sub-formats and the
channel mask, `RF64` refused with a clear message (a later item), a `data` chunk larger than the file cut to the file
with a note. Samples converted in one vectorised pass per channel from a pooled block (`Vector<int>` widening for 16-bit,
3-byte reads for 24-bit; a scalar tail; results identical to the scalar loop, tested), the integer scale 1/2^(bits-1).
`WavEncoder`: 8, 16, 24, 32-bit PCM and float32, extensible when channels > 2 or bits > 16.

**FLAC** (decision 6, when accepted): `FlacCodec` in C# (the stream info, the fixed and LPC subframes, Rice residuals,
the CRC checks), decode only; its fixture rows come from `soundfile` in S0. Without it, `.flac` is "not an audio format
of the registered codecs" with the `AudioCodecs.Register` hint, and MP3 / Ogg-Opus / AAC are plug-in plans under
`plans/plugins/` (a dependency, or a large C# decoder, the library must not carry by default).

**Registrations.** `Idrak.Audio.LibraryRegistrations` (`src/Idrak.Audio/LibraryRegistrations.cs`, the shape of
`Idrak.Imaging/LibraryRegistrations.cs`): `[typeof(AudioCodecs)]` registers wav (and flac), `[typeof(AudioEncoders)]`
wav, `[typeof(SampleSources)]` "audio" (S6), `[typeof(ChatParts)]` "audio" (S3), `[typeof(TuningDataFormats)]`
"transcripts" (S6), `[typeof(PretrainingSources)]` "audio-records" (S6). `AudioCodecs`' static constructor calls
`LibraryDefaults.Ensure(typeof(AudioCodecs))` as `ImageCodecs.cs:60` does, so an app that references only
`Idrak.Audio.Abstractions`' types gets the codecs the day it touches the registry.

**Tests** (`"audio codecs:"`, `tests/Idrak.Tests/AudioCodecTests.cs`): every `wav/` file decodes to its `.npy` (PCM
exact, float within 1e-6); `ReadInfo` from the first bytes and the LIST-chunk file; the two bad files are refused with
messages naming the file; the encoder round-trips every bit depth (decode(encode(x)) equals x to the depth's step), the
16-bit view goes through unchanged; a 10-minute synthetic file decodes with the pooled block and no large-object
allocation per call (`GC.GetTotalAllocatedBytes` around the decode, reported in `-v`); the `AudioCodecSuite` on wav; an
outside codec registered from `tests/Idrak.PluginTests` is asked first and `Unregister` restores the order; the
vectorised conversion equals the scalar one (`DOTNET_EnableHWIntrinsic=0` noted in the README's command list).

**Done when** the tests pass on the CPU; inventory and API regenerated (a new `api/Idrak.Audio.txt`).

**Risks.** WAV files from phones with wrong `data` sizes (handled: cut to the file); 24-bit alignment on big-endian
hosts (`BinaryPrimitives` fixes the order; rule 48).

## Phase S2: resampling and features

**Resampling** (`src/Idrak.Audio/Resampler.cs`): torchaudio's windowed-sinc resampler (`sinc_interp_hann`,
`lowpass_filter_width` 6, `rolloff` 0.99, the rate pair reduced by their gcd) as a polyphase filter whose kernel
[phases, taps] is built once per (from, to, width, rolloff) and cached (rule 75). On the host: each output phase is one
dot product over the taps, rows in order (rule 72), `Vector<float>` with a scalar tail, channels and blocks spread with
`HostParallel.For` (rule 79, measured on 1 and all cores). On a device: the same kernel as a `Convolution` of [1, 1, 1,
samples] by [phases, 1, 1, taps] with stride (1, from/gcd) and the phases as output channels, then a permute to
interleave the phases; one launch, the existing convolution kernels, the product's fixed order (rule 86). The host and
device paths give the same values within 1e-5 (tested). `ResampleOptions { Width = 6, Rolloff = 0.99, Method = "sinc-hann" }`.

**Features** (`src/Idrak.Audio/Features/`):

```csharp
public sealed record AudioFeatureOptions
{
    public int SampleRate { get; init; } = 16000;          // the family's; a file at another rate is resampled first
    public int WindowSize { get; init; } = 400;            // n_fft
    public int HopSize { get; init; } = 160;
    public string Window { get; init; } = "hann";          // "hann" (periodic), "hamming", "rectangular"; a registry later if a family needs more
    public bool Center { get; init; } = true;              // reflect padding of WindowSize / 2 at both ends
    public int Mels { get; init; } = 80;
    public float MinFrequency { get; init; }
    public float? MaxFrequency { get; init; }              // SampleRate / 2 when null
    public string MelScale { get; init; } = "slaney";      // "slaney" | "htk"
    public string MelNorm { get; init; } = "slaney";       // "slaney" | "none"
    public float LogFloor { get; init; } = 1e-10f;
    public float? DynamicRange { get; init; } = 8f;         // values below max - range are raised to it; null for none
    public (float Offset, float Scale)? Normalize { get; init; } = (4f, 0.25f);   // (x + offset) * scale; null for none
    public double? PadSeconds { get; init; } = 30;          // pad or cut a segment to this length; null for the signal's length
    public string Path { get; init; } = "auto";            // "auto" | "composed" | "kernel": IDRAK_STFT_PATH overrides
}

public interface IAudioFeatureExtractor
{
    string Name { get; }                                    // "log-mel"
    AudioFeatureOptions Options { get; }
    int FramesFor(long samples);
    Tensor Extract(AudioData audio, Device device);         // [frames, mels] (or [batch, frames, mels] for Extract(IReadOnlyList<AudioData>, device))
    static abstract IAudioFeatureExtractor FromConfig(JsonObject preprocessorConfig, AudioFeatureOptions? defaults = null);   // transformers' preprocessor_config.json keys
}

public static class AudioFeatureExtractors   // "log-mel" the library default; Register, Get, ...
public static class Stft                     // Magnitude(samples, options, device), Power(...), Complex(...) as [frames, bins] tensors; Inverse(...) for Griffin-Lim (S5)
public static class MelFilterbank            // Build(bins, mels, rate, fmin, fmax, scale, norm) as a cached [bins, mels] tensor per device
```

**The composed path** (the default until a kernel is measured faster): the signal as a [1, 1, 1, samples] tensor, reflect
padding through a gather (`Gather` with precomputed indices, one per shape), framing by `Im2Col` with kernel (1, N) and
stride (1, H) to [frames, N], the window as `AddRowVector`-style scaling (`Binary` with a broadcast row), the DFT as one
product [frames, N] × [N, N + 2] (the cosine and sine basis of the N/2 + 1 bins, built in double, stored float32), the
power as `Pow` (square) and `Axpy` of the real and imaginary halves, the mel projection as [frames, N/2 + 1] × [N/2 + 1,
M], then `Clamp` (floor), `Unary` log10, the dynamic-range floor (a `SumRows`/max read of one value per segment and a
`Clamp`) and the affine normalisation. Every step is an existing operation with a kernel on CUDA and Vulkan and a host
fallback on HIP (plan 7's minimum-backend rule), so the path runs on every device from day one and takes no new
kernel. Summation order: a product's fixed tiling (rule 86); the host reference sums bins in frame order. Buffers per
(frames, N, M) shape are reused across segments (rule 77).

**The host path** (`CpuBackend` is not touched: this is the package's own host code for small inputs and for the
reference): an iterative radix-2 real FFT for power-of-two N and a mixed-radix (2, 3, 5) one for the others (400 = 2^4
× 5^2), butterflies in a fixed order, frames in parallel over blocks of frames (rule 79), the window applied while the
frame is copied into the FFT buffer (one pass, rule 73). Chosen over the composed path when the input is short
(`frames × N` under the CPU's measured parallel cut-over, `CpuTuning.ParallelElements`) or the device is the CPU.

**The kernel path** (after measuring): `Idrak.Audio` declares `AudioOperations.Stft` through
`PluginOperations.Register<StftKernel>("Audio.Stft", composed default, KernelSource.Composed)`; `Idrak.Gpu` registers a
PTX and a SPIR-V kernel for it (a block per frame, the window and the twiddles in shared memory sized from the reported
limits, rule 84; one launch per segment batch, rule 82; coalesced frame loads, rule 83; a fixed butterfly order, rule
86; `.maxntid` declared, rule 87), measured against the composed path per shape on first use as the matrix-product
tuning is (median of 7, kept in the tuning cache) and used only where faster. `Path = "kernel"` or
`IDRAK_STFT_PATH=kernel|composed` forces either; `idrak kernels` lists the operation.

**Tests** (`"audio features:"`): the resampled fixture files within 1e-5 on the host and on the device; the STFT
magnitude and log-mel of `pcm16.wav` within 1e-4 of the largest on the composed path, the host path and (when
registered) the kernel path, on every device; the three paths agree with one another within 1e-5; `FramesFor` matches
the fixture's frame count with and without `Center`; `FromConfig` reads `tiny-whisper/preprocessor_config.json` to the
fixture's options; the mel filterbank equals `features/mel-filterbank.npy` within 1e-6; a 10-minute signal extracted
in `PadSeconds` segments allocates the same device bytes for each segment after the first (`GetMemoryUsage` before
and after, rule 77); the `AudioFeatureSuite` (random signals: Parseval's identity between the signal's energy and the
spectrum's, a pure tone's bin, zero input gives the floor); cold and warm times per path in `-v`.

**Done when** the fixture matches on the CPU for the composed and host paths; the owner runs the device filter.

**Risks.** The reflect padding's gather on Vulkan for very long signals exceeding `maxStorageBufferRange` (segments
bound the length; a note); float32 products against torch's float32 FFT differ in the last bits (the 1e-4 tolerance
is the fixture's statement, as `vlm-tuning/README.md`'s are).

## Phase S3: the audio chat part and the audio-language contracts

**The part** (`src/Idrak.Audio/Abstractions/ChatAudio.cs`): `ChatAudio : ChatPart` with the shape of `ChatImage`:
bytes as received (never decoded here), `MediaType` given or sniffed (`RIFF....WAVE` → audio/wav, `fLaC` → audio/flac,
`ID3` or an MPEG frame sync → audio/mpeg, `OggS` → audio/ogg, `ftyp` with `M4A` → audio/mp4), `Hash` (SHA-256, the
identity: a model encodes an audio once per conversation), `FromFile`, `FromBytes`, `FromStream(stream, length)`,
`FromDataUrl` (`data:audio/wav;base64,...`), `ToDataUrl`, value equality by hash, `Kind => ChatParts.Audio` (a new
constant "audio" in `ChatParts`, beside Text and Image, so the name is one string in the library). The "audio"
`IChatPartKind` writes `{"type": "audio", "media_type": "audio/wav", "data": "<base64>"}` and reads it back; registered as a
library default by `LibraryRegistrations` through `LibraryDefaults.Ensure(typeof(ChatParts))` (the one-line change in
`ChatParts.cs:216`). The MCP bridge (`McpTools.cs:100`) gets a hook: `ChatParts.Find("audio")` and a new
`IBinaryPartKind : IChatPartKind { ChatPart FromBytes(ReadOnlySpan<byte> data, string? mediaType); }` in Abstraction,
so an `AudioContentBlock` becomes a `ChatAudio` part when the kind is registered and the text placeholder otherwise.

**The contracts** (`src/Idrak.Audio/Abstractions/AudioFamilies.cs`), the audio twin of `VisionFamilies.cs`:

```csharp
public sealed record AudioCheckpoint(string Architecture, JsonObject Config, ITensorStore Tensors, string? Folder, Func<ITensorStore> Open, List<string> Notes);

public abstract class PretrainedAudio                       // what a family's Read returns
{
    public abstract IReadOnlySet<string> StoredTensors { get; }
    public abstract IAudioFeatureExtractor Features { get; }                       // from preprocessor_config.json
    public abstract IImagePromptFormat PromptFormat { get; }                       // the marker → begin + N audio tokens + end (ImageTokenFormat reused, decision 5)
    public abstract IImageAttentionRule Attention { get; }                         // "causal" or a family rule
    public abstract IAudioEncoder CreateEncoder(AudioEncoderOptions options);      // features → [tokens, textDim] per audio
}

public interface IAudioEncoder : IDisposable                 // mirrors IVisionEncoder: Device, Blocks(AudioData) → ImageTokenLayout list, Encode(IReadOnlyList<AudioData>) → ImageFeatures
public interface IAudioFamily
{
    string Name { get; }
    PretrainedAudio? Read(AudioCheckpoint checkpoint);
    CreatedAudio Create(JsonObject audio, int textDim, ITokenizer tokenizer, int seed) => throw new NotSupportedException(...);   // as IVisionFamily.Create
}
public sealed record CreatedAudio(JsonObject Config, IReadOnlyList<(string Name, int[] Shape, float[] Values)> Tensors, IReadOnlyDictionary<string, string> Files);
public static class AudioFamilies   // Register, Unregister, Names, Find, Get, Default, Origin, HasAudio(config) ("audio_config"), For(architecture, config)

public static class AudioLanguage   // Idrak.Audio: Attach(PretrainedModel model) → PretrainedAudio? (AudioFamilies.For(model.Architecture, model.Config), the checkpoint opened
                                    // again from model.Folder through CheckpointFormats, as VisionCheckpoint.Open does); null for a model without audio_config
```

**Soft parts in generation and tuning.** A general contract in `Idrak.Abstraction.Generation/SoftParts.cs`:

```csharp
public interface ISoftPartSource
{
    string Kind { get; }                                                            // the ChatPart kind it reads ("audio")
    IImagePromptFormat Format { get; }
    IImageAttentionRule Attention { get; }
    IReadOnlyList<ImageTokenLayout> Blocks(ChatPart part);                         // how many soft tokens the part takes
    ImageFeatures Encode(IReadOnlyList<ChatPart> parts);                            // [tokens, dim] per part, on the model's device
}
```

`ChatGenerator` gains `Parts { get; init; } = []` beside `Images` (`ChatGenerator.cs:120`); `PartKinds` is text, image
when `Images` is set, and every `Parts[i].Kind`; the prompt is expanded by each source's `Format.Expand` for its own
parts in message order, and the prefill goes through `ImagePrefill.Locate` / `ForwardCached` with the features of all
soft parts concatenated in prompt order (the prefill replaces runs of a soft token id by features and knows nothing
of images). `ChatTranscriptEncoder` gains `Parts` the same way (`FineTuning.cs:418-432`), writing `TrainingImage`-shaped
blocks (`TrainingPart(ChatPart, Blocks)`) so `StepImages` lays them out unchanged. `ChatAudioParts` (`Idrak.Audio`)
implements `ISoftPartSource` over a `PretrainedAudio` with a feature cache (`FeatureCaches`, keyed by the audio's hash
and the extractor's options, as `TuningVision` caches image stages). Images keep their path exactly (their tests stay
green); moving them onto `Parts` is listed in plan 14's backlog, not done here.

**Tests** (`"audio chat:"`): `Conformance.CheckChatPartKind("audio")` with the fixture's files; JSON round trip,
`FromDataUrl`, sniffing of each signature, equality by bytes; `ContentToJson` of a message with text and audio;
`ThrowIfUnsupported` names "audio" for a text-only model; the transcript encoder with a fake `ISoftPartSource` (two
tokens per audio) gives `transcripts/tuning.json`'s ids, trained mask and blocks; `ChatGenerator` with the same source on
the tiny Llama of `pretrain/` answers and the features land at the soft tokens' positions (the prefill's inputs
checked); `AudioLanguage.Attach` on a folder without `audio_config` is null and on one with an unregistered family
names `AudioFamilies.Register`; the MCP bridge turns an audio block into a part when the kind is registered.

**Done when** the tests pass on the CPU with no vision test changed. Inventory and API regenerated (Abstraction: one
constant, one interface, one record; Nlp: two properties).

**Risks.** `ImageTokenLayout`, `ImageTokenFormat`, `ImageFeatures` and `ImagePrefill` carry "Image" in their names while
serving audio (decision 5: reuse now, rename later; a rename touches the Gemma 3, LLaVA, Qwen and IdrakVision plug-ins).

## Phase S4: speech recognition contracts and a Whisper-style loader plug-in

**Core additions** (plan 13 assigned them to core; one user today, but they are decoder machinery beside
`CausalSelfAttention`):

- `CrossAttention : Module, ICachedModule` (`src/Idrak/Layers/CrossAttention.cs`): queries from the decoder rows, keys and
  values from a memory `[batch, memory, dim]` given once per sequence through `DecodingContext.SetMemory(owner, keys,
  values)` (the projections computed once and held in a cache from `CacheFor(owner, rows, headDim)`; released by
  `Reset`); attention through `AttentionTiled` (rows attend to every memory position: no mask) and, for one query row, the
  decoding kernels over the memory cache; gradients through the existing tiled backward for training. Rotary positions
  are not applied (encoder-decoder families use learned or sinusoidal positions). `DecoderBlock` takes an optional
  `(Module norm, CrossAttention cross)` pair run between the self-attention and the feed-forward (`Decoder.cs:845`); a
  block without one is unchanged (every decoder test stays as it is).
- `EncoderDecoderSpec(DecoderSpec Decoder, JsonObject Encoder)` in `Idrak.Abstraction.Generation`, written to and read
  from config.json by the family (the library reads nothing of `Encoder`).
- `BeamSearch` (`src/Idrak/Generation/BeamSearch.cs`, decision 4): `Decode(step: Func<int[] ids, DecodingContext, Tensor
  logits>, ...)` over `TextGenerator`'s decoder with `BeamOptions { Beams = 4, LengthPenalty = 1, MaxTokens, EarlyStopping
  = true, NoRepeatNgram = 0 }`, the cache rows reordered per step with a gather over the layouts' `Expand` (the
  int8 and bfloat16 layouts included), the scores kept on the device and read back once per step (the top-k over
  beams × vocabulary through `SampleRows`' candidate pass), ties broken as transformers' `BeamSearchScorer` breaks them.
  Greedy stays `TextGenerator`'s.

**The contracts** (`src/Idrak.Audio/Abstractions/SpeechRecognition.cs`):

```csharp
public sealed record TranscriptionOptions
{
    public string? Language { get; init; }                 // null: detected by the family when it can
    public string Task { get; init; } = "transcribe";      // "transcribe" | "translate" (a family may refuse one)
    public bool Timestamps { get; init; }
    public int Beams { get; init; } = 1;                   // 1: greedy
    public float LengthPenalty { get; init; } = 1f;
    public double ChunkSeconds { get; init; }              // 0: the family's segment length (its PadSeconds)
    public double OverlapSeconds { get; init; } = 1;
    public bool CarryText { get; init; } = true;           // the previous chunk's text as the prompt of the next
    public int MaxBatch { get; init; }                     // 0: measured on the device
    public double MemoryFraction { get; init; } = 0.5;     // of the device's free memory the segment batch may take
    public float Temperature { get; init; }                // 0: greedy; the fallback ladder [0, 0.2, 0.4, ...] is TemperatureFallback
    public IReadOnlyList<float> TemperatureFallback { get; init; } = [];
    public float CompressionRatioThreshold { get; init; } = 2.4f;
    public float LogProbThreshold { get; init; } = -1f;
    public string? Prompt { get; init; }
    public bool BFloat16 { get; init; } = true;             // where the device has it; float32 otherwise
}
public sealed record TranscriptSegment(double Start, double End, string Text, float AverageLogProb, IReadOnlyList<int> Tokens);
public sealed record Transcription(string Text, string? Language, IReadOnlyList<TranscriptSegment> Segments, TranscriptionStats Stats);
public sealed record TranscriptionStats(TimeSpan AudioDuration, TimeSpan Decode, TimeSpan Features, TimeSpan Encoder, TimeSpan Decoder, long PeakDeviceBytes, int Segments, int Batch);

public interface ISpeechRecognizer : IDisposable
{
    Device Device { get; }
    IReadOnlySet<string> Languages { get; }
    Transcription Transcribe(AudioData audio, TranscriptionOptions? options = null, IProgress<TranscriptionProgress>? progress = null, CancellationToken cancellationToken = default);
    IAsyncEnumerable<TranscriptSegment> TranscribeStream(IAsyncEnumerable<AudioData> chunks, TranscriptionOptions? options = null, CancellationToken cancellationToken = default);
}
public sealed record SpeechCheckpoint(string Architecture, JsonObject Config, string Path, string? Folder, ITensorStore Tensors, SpeechModelOptions Options, List<string> Notes);
public sealed record SpeechModelOptions { Device? Device; EncoderWeights Weights = AsStored; KeyValueFormat Cache = Float32; string? Architecture; }
public interface ISpeechFamily
{
    string Name { get; }
    ISpeechRecognizer Read(SpeechCheckpoint checkpoint);
    CreatedSpeechModel Create(JsonObject spec, ITokenizer tokenizer, int seed) => throw new NotSupportedException(...);   // S6
}
public static class SpeechFamilies     // Register, ..., For(architecture, config); unguarded
public static class SpeechModels       // Load(path, SpeechModelOptions?) → ISpeechRecognizer: config.json's "architectures", CheckpointFormats, the family's Read
```

`SpeechModels.Load` reads a folder as `ImageModels.Load` does (`src/Idrak.Vision/ImageModels.cs`): the format
registry opens the tensors one at a time, the family builds its layers, bfloat16 projection weights kept as stored.

**The plug-in** `samples/WhisperSpeech/Idrak.WhisperSpeech` (a Library with `PackageReference Idrak`, `Idrak.Nlp`,
`Idrak.Audio`; referenced by `tests/Idrak.Tests` as the IdrakVision sample is):

- `WhisperSpeechPlugin.cs`: `Register()` → `SpeechFamilies.Register(new WhisperFamily())` under
  `"WhisperForConditionalGeneration"` and `AudioFamilies.Register(...)` under the same name for use as an audio-language
  encoder (its encoder output projected by a `Linear` the plug-in adds for S6's audio-language IdrakAudio);
  `RegisterIdrakPlugin()`, `Unregister()`.
- `WhisperConfig.cs`: `d_model`, layers, heads, ffn sizes, `num_mel_bins`, `max_source_positions`,
  `max_target_positions`, `vocab_size`, the special token ids from `generation_config.json` (`decoder_start_token_id`,
  `lang_to_id`, `task_to_id`, `no_timestamps_token_id`, `suppress_tokens`, `begin_suppress_tokens`), every one a
  setting read from the files.
- `WhisperEncoder.cs`: `Conv2d(mels, D, (1, 3), padding (0, 1))` + GELU, `Conv2d(D, D, (1, 3), stride (1, 2), padding
  (0, 1))` + GELU (the Conv1d weights `[D, mels, 3]` loaded as `[D, mels, 1, 3]`), sinusoidal positions
  (`PositionalEncoding`), `TransformerEncoderLayer` × L through `FromLayers` with `LayerNorm(D, 1e-5)` and `GELU`, a final
  `LayerNorm`; `ForwardCheckpointed` for training; bfloat16 activations under `MixedPrecision.BFloat16` when asked.
- `WhisperDecoder.cs`: `Embedding` + learned positions, `DecoderBlock`s with the new cross block, a tied head
  (`Linear.Tied`), through `DecoderBuilder` where the spec allows and `FromLayers` elsewhere; tensor names mapped from
  `model.encoder.*` / `model.decoder.*` / `proj_out`.
- `WhisperRecognizer.cs`: `Transcribe`: decode (`AudioCodecs`), mono, resample to the config's rate, segments of
  `PadSeconds` (chunks with overlap when `ChunkSeconds` is set), features per segment batch (`AudioFeatureExtractors`
  "log-mel" from `preprocessor_config.json`), the encoder once per segment, the prompt
  `<|startoftranscript|><|lang|><|task|>[<|notimestamps|>]` (language detection: one decoder step over the language
  tokens' logits when `Language` is null), greedy through `TextGenerator` or `BeamSearch`, the suppress lists as a
  logits mask, timestamps read from the timestamp tokens (0.02 s each, as the config's `time_precision`) into
  segments, the temperature fallback ladder on the compression-ratio and log-prob thresholds, the batch measured as
  "Memory and speed" says, stages through `Telemetry.InferenceStage`. `TranscribeStream`: chunks accumulated to
  `ChunkSeconds`, each decoded as above with `CarryText`.
- `README.md`: the files a checkpoint needs, the options, the owner's commands with a public tiny or base checkpoint.

**Tests** (`"audio asr:"`): `tiny-whisper`'s encoder output within 1e-4 of `reference/encoder-output.npy`; the first
step's logits within 1e-4; greedy ids equal `greedy-ids.json`; beam-4 ids equal `beam4-ids.json`; timestamps give the
fixture's segments; a two-segment input through chunking equals two single calls joined; a batch of 3 segments equals
3 single segments within 1e-5 (padding rows attend nowhere); `CrossAttention` against a loop reference and its
gradients by finite differences on the CPU; `BeamSearch` on the tiny Llama of `pretrain/` against a reference beam
written in the test; an unregistered architecture refused with `SpeechFamilies.Register`; `Conformance.Check` of the
`SpeechRecognizerSuite` (silence gives an empty or short text, the same output twice, several threads); peak device bytes
after the second segment equal to after the first (rule 77); cold and warm stage times in `-v`.

**Done when** the fixture matches on the CPU; the owner runs the device filter and a real checkpoint.

**Risks.** Real checkpoints are bfloat16 or float16 with `proj_out` tied; v3 uses 128 mels (a config value); the
temperature fallback changes outputs run to run only when `Temperature > 0` (seeded samplers keep it repeatable).

## Phase S5: text-to-speech contracts and a family plug-in

**Which open architecture (decision 3).** Recommended: **VITS as shipped by MMS-TTS** (`facebook/mms-tts-<lang>`,
transformers' `VitsModel`, safetensors, one checkpoint per language, Arabic among them). One model holds the text
encoder, the stochastic duration predictor, the flow and the HiFi-GAN decoder, so one plug-in speaks with no second
checkpoint; inference is forward only (no mel, no vocoder search); every layer it needs exists or is a composition:
`Conv2d` with (1, k) kernels and dilation (the WaveNet-style posterior and the residual blocks), `ConvTranspose2d` for
the upsampling, `LayerNorm`, `MultiHeadAttention` with relative position bias (the plug-in's own attention module from
public tensor operations: the bias is an added term before the softmax), `Tanh`, `Sigmoid`, a LeakyReLU as
`Lambda(x => Maximum(x, αx))` from public operations (or a `LeakyReLU` layer in core if measured slower), weight norm
folded at load (`weight_g` × normalised `weight_v`, once; rule 75), flows run in reverse at inference. SpeechT5 is the
alternative (encoder-decoder to mel plus a separate HiFi-GAN; needs speaker vectors; English checkpoints); it fits the
contracts the same way and is not planned here.

**The contracts** (`src/Idrak.Audio/Abstractions/SpeechSynthesis.cs`):

```csharp
public sealed record SpeechSynthesisOptions
{
    public string? Voice { get; init; }                    // a speaker id or name when the family has several
    public float Rate { get; init; } = 1f;                 // 1 / length scale
    public float NoiseScale { get; init; } = 0.667f;       // family defaults read from config; 0 for a deterministic output
    public float DurationNoiseScale { get; init; } = 0.8f;
    public int Seed { get; init; }
    public string? Vocoder { get; init; }                  // a Vocoders name for a mel-producing family; null: the family's own or the library default
    public int? SampleRate { get; init; }                  // resample the output when set
    public int MaxCharacters { get; init; } = 0;           // 0: no split; otherwise sentences are synthesised in chunks of at most this many characters and joined
}
public sealed record SynthesisStats(TimeSpan Text, TimeSpan Model, TimeSpan Vocoder, long PeakDeviceBytes, double RealTimeFactor);
public interface ISpeechSynthesizer : IDisposable
{
    Device Device { get; }
    int SampleRate { get; }
    IReadOnlyList<string> Voices { get; }
    (AudioData Audio, SynthesisStats Stats) Speak(string text, SpeechSynthesisOptions? options = null, CancellationToken cancellationToken = default);
    IAsyncEnumerable<AudioData> SpeakStream(string text, SpeechSynthesisOptions? options = null, CancellationToken cancellationToken = default);   // sentence by sentence
}
public interface IVocoder { string Name { get; } int SampleRate { get; } int Mels { get; } AudioData Synthesize(Tensor logMel, Device device); }
public static class Vocoders                 // "griffin-lim" the library default (Idrak.Audio): Stft.Inverse iterated (Iterations a setting, 32), the mel pseudo-inverse built once
public interface ISpeechSynthesisFamily { string Name { get; } ISpeechSynthesizer Read(SpeechCheckpoint checkpoint); CreatedSpeechModel Create(...) => throw ...; }
public static class SpeechSynthesisFamilies  // Register, ..., For
public static class SpeechSynthesisModels    // Load(path, options)
```

**Text in.** A family's tokenizer: MMS-TTS uses a character vocabulary (`vocab.json`) with a phonemizer or
romanizer for some languages (`is_uroman` in the tokenizer config): the plug-in reads `vocab.json`, normalises as
`VitsTokenizer` does (lower case, the `phonemize` flag refused with a clear message, since the library carries no
phonemizer), and for `is_uroman` languages applies a romanisation table the plug-in ships for Arabic (decision 3's
sub-question); other uroman languages are refused naming the flag until a table exists.

**The plug-in** `samples/VitsSpeech/Idrak.VitsSpeech`: `VitsSpeechPlugin` (`Register` under `"VitsModel"`),
`VitsConfig`, `VitsTextEncoder`, `VitsDurationPredictor` (the stochastic flows run in reverse with the seed's noise),
`VitsFlow`, `HifiGanDecoder` (registered too as a `Vocoders` "hifigan" entry so a mel-producing model of S6 can use it
when its mel layout matches), `VitsSynthesizer` (`Speak`: tokens, encoder, durations, length regulation by a gather of
repeated frames with precomputed indices, the flow, the decoder, in one device pass per sentence; the sentence split a
setting; stats and telemetry as S4). The `Rate`, noise scales and seed from `SpeechSynthesisOptions`.

**Tests** (`"audio tts:"`): `tiny-vits` with noise 0 gives `reference/waveform.npy` within 1e-4 of the largest and the
fixture's durations exactly; the same text twice gives the same samples (seeded noise); `Rate 2` halves the length
within one frame; the Griffin-Lim vocoder on the fixture's log-mel gives a signal whose log-mel is within a stated
tolerance of the input (the test prints the error; the README records it); the `VocoderSuite` on both vocoders
(finite output, the length from the frames and hop, the same twice); `SpeakStream` joined equals `Speak` with the
same split; an unregistered family refused; weight-norm folding equals the unfolded product within 1e-6.

**Done when** the fixture matches on the CPU; the owner runs a real MMS-TTS checkpoint.

**Risks.** The relative-position attention and the flows are long (the plug-in is the largest of the three); MMS
checkpoints are float32 (no memory issue) but a sentence's length regulation can make tens of thousands of frames
(the sentence split keeps it bounded; a setting).

## Phase S6: training

**CTC reused.** The from-scratch recognizer trains with `Losses.Ctc` exactly as the ArabicOcr sample does
(`TrainCommand.cs:257`): log-softmax over the head's classes, `[batch, steps, classes]` with `batchFirst`, each
utterance's frame count after the subsampling as its input length, `zeroInfinity` on; the device kernels of plan 13
step 1 run it on CUDA and Vulkan with the CPU as the reference; decoding through `CtcDecoders` "greedy" or "beam"
(`CtcDecodeOptions.BeamWidth`, a setting). Nothing of CTC is written again. New beside it: `CtcAlignment.Viterbi
(logProbs, labels, blank)` in `Idrak.Audio` (the best path through the same 2L + 1 states, on the host, one pass per
utterance; the durations S5's from-scratch synthesizer trains on) checked against `ctc/alignment.json`.

**The trainer** (`src/Idrak.Audio/CtcTrainer.cs`):

```csharp
public sealed record CtcTrainingOptions
{
    public int Steps { get; init; }
    public int BatchSeconds { get; init; } = 160;          // audio seconds per micro-batch (lengths bucketed, sorted within a bucket); halved on ResourceLimitExceededException
    public int GradientAccumulation { get; init; } = 1;
    public float LearningRate { get; init; } = 1e-3f;
    public float WarmupFraction { get; init; } = 0.05f;
    public string Schedule { get; init; } = "cosine";      // FineTuningSchedules
    public string Optimizer { get; init; } = "adamw";      // FineTuningOptimizers; HostOptimizer flag as Pretraining
    public float WeightDecay { get; init; } = 0.01f;
    public float MaxGradientNorm { get; init; } = 1f;
    public bool? BFloat16Activations { get; init; }
    public bool Checkpointing { get; init; } = true;
    public int SaveEvery { get; init; }
    public int EvaluateEvery { get; init; }
    public string Metric { get; init; } = "cer";           // TuningMetrics names; "wer" too
    public IReadOnlyList<string> Augmentations { get; init; } = [];   // AudioAugmentations names: "speed", "noise", "spec-mask" (SpecAugment: time and frequency masks on the features)
    public int Seed { get; init; }
}
public static class CtcTrainer
{
    public static CtcTrainingResult Train(Module encoder, int blank, ITokenizer labels, IAudioFeatureExtractor features, ISpeechSource train, ISpeechSource? evaluation,
        CtcTrainingOptions options, string outputFolder, IProgress<CtcTrainingProgress>? progress = null, CancellationToken cancellationToken = default, Action<string>? trace = null);
    public static CtcTrainingResult Resume(string checkpointFolder, CtcTrainingOptions? overrides = null, ...);
}
```

Reuses `FineTuningOptimizers`, `FineTuningSchedules`, `Optimizer.SaveState` / `LoadState`, `Telemetry`'s training
events, the checkpoint layout of plan 19 (`checkpoint-STEP/` with `model.safetensors`, `optimizer.safetensors`,
`train-state.json`, resumed bit for bit on the CPU). Features extracted on worker threads ahead of the step (as
`ConversationTuning.Load` encodes) with a disk cache (`FeatureCaches` "disk") keyed by the file's hash and the options,
so a second epoch extracts nothing (rule 75); augmentations seeded per utterance and epoch (`Augmentations`' rule).
`ISpeechSource` / `SpeechSources` (`"transcripts"`: a JSONL of `{"audio", "text"}` rows, paths relative to the file or a
zip; `"audio-folder"`: `name.wav` + `name.txt` pairs) give `(AudioData, string)` in a seeded order with a position for
the checkpoint.

**`IAudioFamily.Create` and `idrak create`.** The create spec gains `"audio": {family config}` (as `"vision"` did in
plan 19 B4): `PretrainedModel.Create` asks `AudioFamilies.Get(architecture).Create(audio, textDim, tokenizer, seed)`
through a `CreateOptions.Audio` (one nullable `JsonObject` on core's record, as `Vision` is; core names no family), and
writes `audio_config`, the tensors and `preprocessor_config.json` beside the decoder's files. A recognizer-only model
(`"task": "ctc"` or `"seq2seq"`) is created through `SpeechFamilies.Get(name).Create(spec, tokenizer, seed)` into a
folder `SpeechModels.Load` reads (`CreatedSpeechModel(Config, Tensors, Files)`).

**The IdrakAudio family** (`samples/IdrakAudio/Idrak.IdrakAudio`, decision 1): one plug-in, three tasks chosen by the
spec, every shape a setting in its config (`README.md` lists them):

- `"task": "ctc"`: a subsampler (two `Conv2d` (1, k) stride (1, 2) + GELU, k and D settings) over log-mel, a
  `TransformerEncoderLayer` stack (optionally a depthwise-convolution module between attention and feed-forward,
  Conformer-style, `"convolution_module": true`), a linear head to the tokenizer's vocabulary + blank. Its tokenizer: a
  character or byte-level BPE trained by `idrak data tokenizer` on the transcripts (`--vocab` small; a setting). Trained
  by `CtcTrainer`; read back as an `ISpeechRecognizer` decoding through `CtcDecoders`.
- `"task": "seq2seq"`: the same encoder with S4's decoder (`DecoderSpec` for the text side, cross blocks on), trained
  by `Pretrainer` through a `PretrainingSources` `"audio-records"` source (each row a 2-turn conversation: the audio part
  with the family's marker, then the transcript as the trained answer; `Stages.Trained` accepts `"audio-encoder"` and
  `"audio-projector"` as plan 19 B3's `tower` and `projector`); `KeepTrainedVision`'s rule carried for the audio part.
  Recommended stages: `align` (encoder and projector on the transcripts with the decoder frozen; or the CTC encoder of
  the first task given as `--encoder`, its weights copied), then `joint`.
- `"task": "audio-language"`: the chat model: a text decoder (A5's model through `--decoder`) with the audio encoder as
  an `IAudioFamily` part; the same two stages; then `tune` with audio conversations (the "transcripts" tuning format
  turns pairs into conversations with a prompt template of the owner's) through `ChatTranscriptEncoder.Parts`.
- `"task": "tts"`: a text encoder (`TransformerEncoderLayer`s over characters), a duration predictor (two Conv2d (1, 3)
  + LayerNorm + a linear to log durations), length regulation, a mel decoder (`TransformerEncoderLayer`s), a linear to
  M mels, trained with L1 on log-mel plus MSE on log durations from `CtcAlignment` of the owner's CTC model
  (`SpeechSynthesisTrainer` in `Idrak.Audio`, the CTC trainer's loop with a different loss, its options the same record
  plus `Durations` = a path to alignments `idrak data audio --align MODEL` writes); spoken through `Vocoders`:
  "griffin-lim" by default, "hifigan" from the VITS plug-in when its mel layout matches (the plan does not train a
  HiFi-GAN: adversarial training is a later plan, named in "What a from-scratch audio model is").

**Distillation** (as plan 19 B5): `Pretrainer`'s `Teacher` accepts a loaded audio-language model (the Whisper-style
plug-in registered as an `IAudioFamily` gives one through `AudioLanguage.Attach`); the teacher re-expands each audio
to its own token count and maps the trained positions by the answer's tokens (`DistillationTeacher.CheckVocabulary`
holds: the student is created with the teacher's tokenizer). For CTC students, `CtcTrainingOptions.Teacher` adds a
frame-level KL term against the teacher's CTC posteriors when the teacher is a CTC model of the same vocabulary
(a setting, `DistillationAlpha`); a sequence-level teacher (Whisper-style text) goes through pseudo-labels written by
`idrak transcribe -o train.jsonl`, the simplest and most robust path.

**Tests** (`"audio train:"`): the CTC fixture's step-1 loss within 1e-4 and every gradient within 1e-4 of the largest;
3 SGD steps' losses; the Viterbi alignment exact; a run stopped after step 2 and resumed gives steps 3 to 6 bit for bit;
`BatchSeconds` halved on a forced `ResourceLimitExceededException` (the test backend) and the step repeated; the feature
cache hit on the second epoch (no extraction stage in the telemetry); `create` with `"audio"` through the plug-in writes
a folder `SpeechModels.Load` and `PretrainedModel.Load` read; a `seq2seq` step's loss equals a transformers-style
reference added to S0 (`tiny-whisper` as the created shapes with the fixture's weights copied in); the `tts` task's
loss on the fixture's durations; a 200-step CPU run of the tiny CTC model halves its loss (a smoke test, "lower than
step 1 by half"); the `transcripts` tuning format through `Conformance.CheckTuningDataFormat`.

**Done when** every test above passes on the CPU; `"pretrain:"`, `"pretrain vision:"` and `"vlm"` unchanged.

**Risks.** CTC needs more frames than labels (the trainer counts the rows it drops and says so, as the OCR sample
does); the TTS task without a neural vocoder sounds like Griffin-Lim (stated plainly in the README).

## Phase S7: the command line

- **`idrak transcribe MODEL AUDIO... [-P plugin.dll] [--language ar] [--task transcribe|translate] [--timestamps]
  [--beam N] [--chunk-seconds S] [--overlap-seconds S] [--no-carry] [--temperature T] [--batch N] [--memory-fraction F]
  [--bf16/--no-bf16] [--cache float32|int8|bf16] [-o FILE|DIR] [--format text|json|srt|vtt] [-j] [-d DEVICE]`**
  (`src/Idrak.Cli/Commands/Run/TranscribeCommand.cs`, in `RunCommands`): MODEL a folder, an alias or a hub id (through
  `ModelChoices`); AUDIO files, folders, globs or `-` for standard input (a WAV stream); `SpeechModels.Load`, one
  `Transcribe` per file in the order given, progress per file ("segment i/n  batch b  x.x× real time  mem"), `-o` a
  file for one input or a folder of `<stem>.txt|json|srt|vtt`; `-j` the `Transcription` as JSON; `-v` the stage times
  cold and warm and the peak bytes.
- **`idrak speak MODEL TEXT|-|--input FILE -o out.wav [-P plugin.dll] [--voice NAME] [--rate R] [--noise X] [--seed N]
  [--vocoder NAME] [--sample-rate R] [--split N] [--bits 16|24|32|float] [-d DEVICE]`** (`SpeakCommand.cs`):
  `SpeechSynthesisModels.Load`, `Speak`, `AudioEncoders.Encode` to the `-o` extension's encoder; `--play` is not offered
  (no platform audio in the library; the README says how to play the file).
- **`idrak run MODEL --audio clip.wav "..."`, `idrak chat --audio clip.wav`, `serve`**: `--audio` (repeatable) through
  an `AudioInputs.Read` beside `ImageInputs.Read`, parts in message order; the chat generator built with
  `ChatAudioParts` when `AudioLanguage.Attach(model)` finds an audio part; the chat API's parts carry `"type": "audio"` as
  the JSON of S3 (an OpenAI-style `input_audio` part mapped to it in `ChatApiTranslation`).
- **`idrak data audio FILE|DIR... -o DIR [--rate 16000] [--mono] [--bits 16] [--split silence|seconds] [--min-seconds
  S] [--max-seconds S] [--silence-db D] [--transcripts FILE] [--align MODEL -P plugin] [--stats] [-j]`**
  (`Commands/Data/DataAudioCommand.cs`, in `DataCommands.All`): converts and resamples into a folder of WAV files with a
  `manifest.jsonl` (name, seconds, rate, source, text when `--transcripts` gives one), splits by silence (an energy
  threshold over `HopSize` frames, a setting) or fixed seconds, `--stats` prints durations, rates, channels and a
  histogram (`TextPlot`), `--align` writes `alignments.jsonl` from a CTC model (S6's `CtcAlignment`). **`idrak data mel
  FILE... -o DIR [--config preprocessor_config.json | the feature options]`** writes `.npy` log-mel features and a JSON of
  the options, for inspection and for `data audio --features` caches; `idrak explain` of a speech model folder shows its
  tasks, shapes and feature options.
- **`idrak create`** accepts `"audio": {...}` in the spec and `create init --audio` writes a commented example (the three
  tasks); **`idrak pretrain`** gains `--audio DIR|ZIP` for records sources and the `"audio-records"` source in
  `pretrain.json`; **`idrak train speech`**? No: the CTC loop is **`idrak pretrain`** with `"task": "ctc"` in the model's
  config dispatching to `CtcTrainer` (one command for training from scratch, as the owner has it today), `resume` and
  `runs show` reading its log; **`idrak tune`** with `--data transcripts.jsonl --audio DIR` for audio-language models.
- Arabic summaries, `EnvironmentVariables` lists (`transcribe`, `speak`, `data audio`, `data mel` in `OnDevice`,
  `Models`, `Data`), `IDRAK_STFT_PATH` in the table, `idrak help` topics.

**Tests** (`"audio cli:"`): `transcribe` on `wav/pcm16.wav` with the Whisper plug-in writes text, JSON, SRT and VTT
(the SRT's timestamps from the fixture); `speak` with the VITS plug-in writes a WAV that decodes to the fixture's
samples; `run --audio` on a created audio-language model answers; `data audio` converts the fixture's 44 kHz stereo file
to 16 kHz mono and splits a two-tone file at its silence; `data mel` writes the fixture's features; `create` with
`"audio"` and `pretrain` 2 steps of each task through the plug-in; an unregistered family exits 1 with the `-P` hint;
`help` for each command passes the polish tests.

**Done when** the CLI tests pass on the CPU and the four `cli` health filters still pass.

## Phase S8: the owner's proof run

From `D:\Projects\Idrak`, after `git pull`. The GPU filters first:

```powershell
cd D:\Projects\Idrak
git pull
dotnet build -c Release tests\Idrak.Tests
$env:IDRAK_DEVICES="cuda"; $env:IDRAK_FILTER="audio"; dotnet run -c Release --no-build --project tests\Idrak.Tests
$env:IDRAK_DEVICES="vulkan"; $env:IDRAK_FILTER="audio"; dotnet run -c Release --no-build --project tests\Idrak.Tests
```

Loaded models (a public Whisper-style checkpoint and an MMS-TTS checkpoint of the owner's language; `$whisper` and
`$tts` are the plug-in paths):

```powershell
dotnet build -c Release samples\WhisperSpeech\Idrak.WhisperSpeech
dotnet build -c Release samples\VitsSpeech\Idrak.VitsSpeech
$whisper = "samples\WhisperSpeech\Idrak.WhisperSpeech\bin\Release\net10.0\Idrak.WhisperSpeech.dll"
$tts = "samples\VitsSpeech\Idrak.VitsSpeech\bin\Release\net10.0\Idrak.VitsSpeech.dll"
dotnet run -c Release --project src\Idrak.Cli -- pull openai/whisper-small
dotnet run -c Release --project src\Idrak.Cli -- transcribe openai/whisper-small work\clip.wav --language ar --timestamps -P $whisper -d cuda:0 -v
dotnet run -c Release --project src\Idrak.Cli -- transcribe openai/whisper-small work\clip.wav --beam 4 --format srt -o work\clip.srt -P $whisper -d cuda:0
dotnet run -c Release --project src\Idrak.Cli -- pull facebook/mms-tts-ara
dotnet run -c Release --project src\Idrak.Cli -- speak facebook/mms-tts-ara "مرحبا بكم" -o work\hello.wav -P $tts -d cuda:0 -v
```

The from-scratch recognizer on the owner's recordings (`work\recordings` holds the owner's files; `work\transcripts.jsonl`
their texts, or `idrak transcribe -o` writes a first draft to correct; `$audio` is the IdrakAudio plug-in):

```powershell
dotnet build -c Release samples\IdrakAudio\Idrak.IdrakAudio
$audio = "samples\IdrakAudio\Idrak.IdrakAudio\bin\Release\net10.0\Idrak.IdrakAudio.dll"
dotnet run -c Release --project src\Idrak.Cli -- data audio work\recordings -o work\clips --rate 16000 --mono --split silence --max-seconds 20 --transcripts work\transcripts.jsonl --stats
dotnet run -c Release --project src\Idrak.Cli -- data tokenizer work\clips\manifest.jsonl --vocab 256 -o work\speech-tokenizer.json
dotnet run -c Release --project src\Idrak.Cli -- create init --audio -o work\ctc-spec.json
dotnet run -c Release --project src\Idrak.Cli -- create work\ctc-spec.json --tokenizer work\speech-tokenizer.json -o work\ctc -P $audio
dotnet run -c Release --project src\Idrak.Cli -- pretrain work\ctc --data work\clips\manifest.jsonl --steps 5000 --lr 1e-3 --eval work\val\manifest.jsonl --metric cer -o work\ctc-run -d cuda:0 -P $audio
$ctc = (Get-ChildItem work\ctc-run -Directory -Filter checkpoint-* | Sort-Object { [int]($_.Name -replace 'checkpoint-', '') } | Select-Object -Last 1).FullName
dotnet run -c Release --project src\Idrak.Cli -- transcribe $ctc work\val\clip-001.wav -P $audio -d cuda:0 -v
```

The audio-language model on A5's text model and the synthesizer (`$text` is plan 19's checkpoint):

```powershell
dotnet run -c Release --project src\Idrak.Cli -- create init --audio --task audio-language -o work\al-spec.json
dotnet run -c Release --project src\Idrak.Cli -- create work\al-spec.json --decoder $text --encoder $ctc -o work\al -P $audio
dotnet run -c Release --project src\Idrak.Cli -- pretrain work\al --data work\clips\manifest.jsonl --trained audio-encoder,audio-projector --steps 1000 --lr 1e-3 -o work\al-align -d cuda:0 -P $audio
$aligned = (Get-ChildItem work\al-align -Directory -Filter checkpoint-* | Sort-Object { [int]($_.Name -replace 'checkpoint-', '') } | Select-Object -Last 1).FullName
dotnet run -c Release --project src\Idrak.Cli -- pretrain $aligned --data work\clips\manifest.jsonl --tokens work\tokens.bin --weight data=3 --trained all --steps 2000 --lr 1e-4 -o work\al-joint -d cuda:0 -P $audio
$joint = (Get-ChildItem work\al-joint -Directory -Filter checkpoint-* | Sort-Object { [int]($_.Name -replace 'checkpoint-', '') } | Select-Object -Last 1).FullName
dotnet run -c Release --project src\Idrak.Cli -- run $joint --audio work\val\clip-001.wav "اكتب ما قيل" -P $audio -d cuda:0
dotnet run -c Release --project src\Idrak.Cli -- data audio work\clips -o work\clips --align $ctc -P $audio
dotnet run -c Release --project src\Idrak.Cli -- create init --audio --task tts -o work\tts-spec.json
dotnet run -c Release --project src\Idrak.Cli -- create work\tts-spec.json --tokenizer work\speech-tokenizer.json -o work\tts -P $audio
dotnet run -c Release --project src\Idrak.Cli -- pretrain work\tts --data work\clips\manifest.jsonl --durations work\clips\alignments.jsonl --steps 5000 -o work\tts-run -d cuda:0 -P $audio
$ttsRun = (Get-ChildItem work\tts-run -Directory -Filter checkpoint-* | Sort-Object { [int]($_.Name -replace 'checkpoint-', '') } | Select-Object -Last 1).FullName
dotnet run -c Release --project src\Idrak.Cli -- speak $ttsRun "مرحبا بكم" -o work\hello-scratch.wav --vocoder griffin-lim -P $audio -d cuda:0
```

The comparison is `tune evaluate` style: `idrak transcribe` of the loaded model and of `$ctc` on `work\val`, scored
with `--metric cer` (`transcribe --truth work\val\manifest.jsonl --metric cer` prints both), recorded in the "as
built" section with the stage times and peak bytes.

## Order and parallelism

S0 → S1 → S2 (one agent: S2 needs S1's `AudioData`) and S3 (a second agent: it needs S1's part type only) → S4 (the core
layers first, then the plug-in) and S5 (a second agent; it needs S2's `Stft.Inverse` for Griffin-Lim) → S6 → S7 → S8
(the owner). At most two agents at a time. Each phase is one commit series on `abstraction` with its "as built"
section appended to this plan (what was built, filter names and counts, what is left, the owner's GPU commands), as
plans 11 to 19 did.

## What a from-scratch audio model is, honestly

- **The pipeline is the deliverable of S0 to S7.** It is proven at tiny scale on the CPU against PyTorch, as every
  plan's reference is. A recognizer of useful quality needs hours of transcribed speech per speaker and language
  (tens to hundreds for a general one); the owner's recordings give a speaker-specific model first, which is a
  reasonable target for dictation of one voice. Pseudo-labels from the loaded Whisper-style model (`transcribe -o`)
  stretch the owner's data; distillation (S6) is the lever when the teacher's vocabulary is shared.
- **CTC is the cheap, robust first model**: no decoder, no cross-attention, greedy decoding at a few milliseconds per
  utterance; its weakness is spelling without a language model (a CTC beam with a token-level language model is a later
  item). The encoder-decoder task reuses the CTC encoder and adds the decoder's quality at the decoder's cost.
- **Synthesis from scratch without a neural vocoder sounds like Griffin-Lim.** The plan makes the mel model and the
  durations (from the owner's own CTC model) and leaves the vocoder as a slot: the VITS plug-in's HiFi-GAN when its
  layout matches, or a HiFi-GAN trained in a later plan (adversarial training with multi-period and multi-scale
  discriminators is its own piece of work, not hidden here). Loading MMS-TTS gives a voice on day one.
- **One device per run** (plan 6), as every training plan.

## Decisions (for the owner)

1. **The from-scratch family's name and shape.** One plug-in `Idrak.IdrakAudio` with `"task": ctc | seq2seq |
   audio-language | tts` in its spec, or one plug-in per task. **Recommended: one plug-in** (`samples/IdrakAudio`), one
   encoder shared by the three recognition tasks, every shape a setting, as IdrakVision is one.
2. **The STFT path.** Composed from existing operations first and a fused kernel per backend only after it is measured
   faster per shape (as the plan writes it), or a kernel from day one. **Recommended: composed first**; the kernel
   is a measured follow-up inside S2 and never the only path.
3. **The synthesizer to load.** VITS as MMS-TTS ships it (one checkpoint per language, Arabic available, the vocoder
   inside) or SpeechT5 (English, a separate HiFi-GAN, speaker vectors). **Recommended: VITS**, with the plug-in shipping a
   romanisation table for Arabic (MMS's `is_uroman`); other uroman languages refused until a table exists.
4. **Where beam search lives.** Core beside `TextGenerator` (usable by `idrak run --beam` later) or in `Idrak.Audio`
   (decision 10's one user today). **Recommended: core** (`Idrak.Generation.BeamSearch`): it is decoder machinery
   on `DecodingContext`, not an audio contract.
5. **Soft-token types.** Reuse `ImageTokenLayout`, `ImageTokenFormat`, `ImageFeatures` and `ImagePrefill` for audio as
   they are, or rename them to `SoftToken*` first (touching four plug-ins and the vision tests). **Recommended: reuse
   now**, the rename in plan 14's backlog once images also go through `ISoftPartSource`.
6. **Compressed formats.** FLAC decoded in C# in the package (lossless, the format of most speech corpora), MP3 and
   Ogg-Opus as plug-in plans under `plans/plugins/`; or every compressed format a plug-in. **Recommended: FLAC in the
   package** (S1), the rest plug-in plans.
7. **The first data.** The owner's own recordings segmented by `data audio` and transcribed by the loaded model, then
   corrected; or a public corpus. **Recommended: the owner's recordings** (one voice, 1 to 3 hours to start), the proof
   run's CER against the loaded model on held-out clips as the measure.
