# Plan 20: Idrak.Diffusion (image generation by denoising)

**Not started.**

**Status:** planned 2026-10-15 against the code at `abstraction` 82d84937. Nothing is built: the package name is
reserved (`src/Idrak.Abstraction/LibraryDefaults.cs:30` lists `Idrak.Diffusion` among the first-party assemblies and
`:42` keeps `Idrak.Diffusion.LibraryRegistrations` for the trimmer; plan 10's package table,
`plans/10-abstraction.md:432`, assigns it "UNet, DiT, VAE, schedulers, text encoders, text-to-image pipelines; registers
an image model kind"), and no project, type, kernel, test, fixture or sample of it exists. The owner's two questions
(is it ready; a package or a plug-in) are answered in the first two sections.

**Goal.** Generate images by denoising: a denoiser (a UNet, later a DiT) predicts the noise in a noisy latent at a
timestep, a sampler walks the schedule from pure noise to an image, a latent codec (a VAE, or the pixels themselves)
turns latents into pixels, and a conditioner (class labels, or a text encoder) steers the denoiser. The same machinery
trains a small diffusion model from scratch on the owner's images (`idrak create`, `idrak pretrain`) and loads
existing checkpoints in the diffusers layout (Stable Diffusion 1.x and 2.x first) through a family plug-in, so
`idrak diffuse` generates from either.

**Rules that hold for every phase** (`CLAUDE.md`, `docs/optimization.md`):

- **Card-agnostic.** No phase assumes a device, a vendor or a memory size. Batch, resolution, tiling, precision and
  graphs are settings whose automatic values are measured on the running device (`Backend.FreeMemory`,
  `GetMemoryUsage().Peak`, `ResourceLimitExceededException` caught by a ladder as `FineTuner`'s is); no size, card or GB
  budget appears in code, comments, tests or docs, this plan included. The sizes below are fixture sizes or model facts.
- **Abstraction (decision 10).** Everything new is a contract with a `SlotTable` registry. Today no second library
  package uses any diffusion contract (Vision does not denoise, Nlp does not schedule), so every contract starts in
  `Idrak.Diffusion.Abstractions`; the inventory test moves one to `Idrak.Abstraction` the day a second package uses it.
  The library registers no family: an unregistered name fails with the registry's "is not registered" message and the
  CLI adds "Load its plug-in with -P (--plugin)."
- **Speed and memory first.** Scores of image attention are never stored where a span kernel serves (rule 77); every
  per-step coefficient is computed once on the host in double and applied on the device in a fixed order (rules 85, 86);
  the denoiser pass is one recorded graph replayed per step where the device records graphs (rule 82); the latent of
  each training image is encoded once and cached (rule 75); memory is measured with the library's telemetry and the
  backend's peak counter, never an ad hoc stopwatch (rule 78); cold and warm steps reported apart (rule 76); results
  proven unchanged against the fixture (rule 71).
- **Nothing fixed.** Every schedule, sampler, step count, guidance scale, resolution, batch, precision, tiling choice,
  EMA decay, objective, weighting and path is a library option carried to `idrak` and to the run file, with an
  environment variable where the library already reads one (`IDRAK_MATMUL`, `IDRAK_ATTENTION_PATH`, `IDRAK_AUTOTUNE`,
  `IDRAK_OFFLOAD`, `IDRAK_GPU_CHUNK_MB`, `IDRAK_VULKAN_GRAPHS`, `IDRAK_RETRY_ON_HOST`).
- **Tests.** Targeted CPU runs only (`IDRAK_DEVICES=cpu IDRAK_FILTER=...`); the owner runs CUDA and Vulkan from the
  command list at the end of each phase. Never the full suite.
- **The fix workflow.** Every phase: fix all errors first, update the tests the fixes touch, build once (only the
  projects that changed), run the phase's filter once with `--no-build`, commit and push on `abstraction`, hand the
  owner the GPU command list. At most two agents at a time.

## Is it ready? What exists today

**No.** Nothing of diffusion is built. Evidence at 82d84937:

- `src/Idrak.Abstraction/LibraryDefaults.cs:30,42` names `Idrak.Diffusion` so a package of that name registers its
  built-ins the day it appears; there is no `src/Idrak.Diffusion`, no `api/Idrak.Diffusion.txt`, no entry in
  `Idrak.slnx`, no `tools/diffusion`, no `tests/Idrak.Tests/data/diffusion`.
- A word-bounded, case-insensitive search of `src/`, `samples/`, `tests/`, `tools/`, `docs/` and `plans/` for diffus,
  unet, u-net, vae, ddpm, ddim, timestep, noise schedule, denois and latent finds no code: the only hits are the
  `unet.json` example of `idrak train` (`src/Idrak.Cli.Commands/Commands/Train/TrainingCommands.cs:79`, a plain convolutional
  segmenter) and the README's segmentation example (`README.md:1036`), plan 10's package table, and the test that names
  a segmenter file `unet` (`tests/Idrak.Tests/CliVisionTests.cs:357`). There is no noise schedule, sampler, timestep
  embedding, VAE, cross-attention block, denoising loss or image generation command anywhere.

What exists and diffusion reuses (file: line at 82d84937):

| Part | Today |
|---|---|
| Convolution layers | `Conv2d` (`src/Idrak/Layers/Convolution.cs:16`: rectangular, strided, dilated, grouped; the fused inference epilogue `Convolution.Fused.cs`), `ConvTranspose2d`, `Upsample` (`Layers/Resampling.cs:13`, nearest and bilinear), `AvgPool2d`, `AdaptiveAvgPool2d` |
| Normalization | `GroupNorm` (`Layers/Normalization.cs:114`: the batch-norm kernels `NormStats` / `NormApply` / `NormBackward` / `GroupReduce` / `GroupScaleShift` over the [1, N·groups, C/groups·positions] view, then `GroupAffine`; no fused activation), `LayerNorm` (`:264`, `LayerNormFused` / `LayerNormTrain` / `LayerNormBackward` kernels) |
| Attention | `MultiHeadAttention` (`Layers/Attention.cs:13`): a bidirectional inference pass takes `Tensor.AttentionFastestPacked` / `AttentionFastest` (`src/Idrak.Abstraction/Tensors/Tensor.AttentionPaths.cs:91,121`: the span kernels, or the composed scores where measured faster and the memory allows); training takes the composed scores `[N·H, T, T]` (`Attention.cs:131-160`). Head-size limits are reported, not assumed: `BackendCapabilities.SpanAttentionHeadDim` (`Devices/BackendCapabilities.cs:31`; CUDA 128 or 256 from the device's shared memory, Vulkan 256, CPU any: plan 17 A1-A3), `TiledAttentionHeadDim` (`:22`). Cross-attention (queries of one length, keys of another) is `AttentionFastest(q, keys, values, starts, ends, scale, everyKey: true)`; its gradient `AttentionSpansBackward` |
| Transformer blocks | `TransformerEncoderLayer` (`Attention.cs:242`), `Linear` (bfloat16 weights kept as stored, `KeepsBFloat16`), `Embedding`, `PositionalEncoding`, `Gelu`, `Silu`, `Dropout`, `Sequential`, `NetworkBuilder` / `NetworkOps` / `LayerTypes` |
| Tensor operations | `Sin`, `Cos`, `Exp`, `Log`, `Sqrt`, `Pow`, `Clamp`, `Where`, `Concat`, `Stack`, `Narrow`, `Permute`, `Reshape`, `Interpolate`, `Convolution`, `ConvolutionTranspose`, `Normalize`, `GroupAffine` (`src/Idrak.Abstraction/Tensors/Tensor.*.cs`); `Tensor.Normal` (Box-Muller on the host, `Tensor.cs:158`) and `Uniform` (`:145`); `Dropout` with a hash mask equal on every device (`DropoutMask.Keep`, `Devices/Backend.cs:2310`). No device-side Gaussian generator |
| Device operations | `Backend.Operations.g.cs` (6,906 lines, generated by `tools/operations/generate.py` from `Backend.cs`; every `NameKernel` defaults to a host fallback, plan 7's minimum-backend rule): `Convolution` / `ConvolutionBackwardInput` / `ConvolutionBackwardWeight` with a `ConvActivation` epilogue, `Im2Col` / `Col2Im`, `Interpolate2d(+Backward)`, `AvgPool`, `AdaptiveAvgPool`, the norm kernels above, `AttentionSpans(+Backward)`, `AttentionSpansStrided`, `AttentionTiled(+Backward)`, `AttentionBFloat16`, `BatchedMatMul`, `BFloat16MatMul`, `MatMulBias`, `Unary` / `Binary`, `Axpy`, `Affine`, `MulAdd`, `Pow`, `Clamp`, `Where`, `Sum*`, `ClipFactor`, `AdamStep`, `AdamStep8Bit`, `FusedAdamW`, `SgdStep`, `Dropout`, `Permute`, `Copy2D`, `Gather`, `ScatterAdd`, `BroadcastAxis`, `AddRowVector`, `PackBFloat16`. CUDA and Vulkan kernels for convolution (implicit products measured per shape, `TuneOp.Convolution`), resampling and group norm exist (plan 13 step 1); HIP takes the host fallbacks |
| Memory controls | `ActivationMemory.Recompute` / `CompressToBFloat16` / `CheckpointBlocks` (`Tensors/ActivationMemory.cs:41-72`), `Checkpointing.Checkpoint` (`src/Idrak/Layers/Checkpointing.cs:30`, internal), `MixedPrecision.BFloat16()` (`Tensors/MixedPrecision.cs:78`), `ComputeResources.GpuMemoryLimit` / `GpuMemoryReserve` / `OffloadToHostMemory` (`ComputeResources.cs:69-85`), `Backend.GetMemoryUsage()` with `Peak` and `ResetPeakMemoryUsage` (`Backend.cs:623, 2202`), `FreeMemory` / `AvailableMemory` (`:2192-2199`) |
| Graphs | `ComputeGraph.Capture(device, step)` / `Replay` (`Tensors/ComputeGraph.cs:85, 136`), `Backend.SupportsGraphs` (`:2267`); the rule in its remarks: scalars frozen at recording, no host copies while recording, buffers alive and in place |
| Training | `Optimizer.SaveState` / `LoadState` and `OptimizerStateValue` (`src/Idrak.Abstraction/Training/Optimizer.cs:71, 80, 260`), `Adam` / `AdamW` / `Sgd` (`Training/Adam.cs:16, 163`, `Sgd.cs:7`), `AdamW8Bit`, `HostOptimizer`, `GroupedOptimizer` (`src/Idrak/Optimizers/`), the learning-rate schedulers (`Training/Schedulers.cs`; `FineTuningSchedules` cosine / linear / constant / wsd and `FineTuningOptimizers` live in `src/Idrak.Nlp/FineTuningLosses.cs:394, 464`), `Trainer.Fit` over an `IBatchSource` with telemetry (`src/Idrak/Training/Trainer.cs:166-318`), `Pretrainer` (`src/Idrak.Nlp/Pretraining.cs:311`: `checkpoint-STEP/` with `optimizer.safetensors` and a state file, `KeepCheckpoints`, `ReadCheckpoint` `:869`, `LatestCheckpoint` `:864`, resume bit for bit), `FineTuner.StepRunner` (`FineTuning.cs:1179`; internal and token-shaped) with its out-of-memory ladder (`:1257-1302`: recompute feed-forward, bfloat16 activations, checkpointing, micro-batch halving) and recorded forward-and-backward graphs (`:2337-2428`). **No EMA anywhere** |
| Telemetry | `Telemetry.TrainingStarted` / `BatchCompleted` / `EpochCompleted` / `TrainingCompleted` (`Diagnostics/Telemetry.cs:239-275`; `BatchCompleted` fields `Events.cs:42-52`), `InferenceStage` (`:219`), `HostFallback` (`:253`), `OperationCompleted` (`:122`); `Profiler.Run` (`Diagnostics/Profiling.cs:406`) and the GPU profiler behind `--profile` |
| Images | `ImageData` (`src/Idrak.Abstraction/Data/ImageData.cs:16`: float pixels or byte planes, `Resize`), `ImageCodecs` / `ImageEncoders` (`Data/ImageEncoders.cs:33`; png and netpbm in core, jpeg in Imaging), `ImageFolderSource` (`src/Idrak.Imaging/ImageFolderSource.cs:31-73`; unlabelled files at a fixed size), `ImagePreprocessor` (`Imaging/ImagePreprocessor.cs:67`; resize, crop, rescale, normalize, on the device through `ResizeNormalize`), the `Augmentations` registry (flip, resized-crop, affine, color-jitter, ...), `AugmentedImageLoader` (`src/Idrak.Vision/AugmentedImageLoader.cs:82`; worker threads, batches ahead sized from free memory) |
| Checkpoints | `SafeTensorsReader : ITensorStore` (`src/Idrak/Models/SafeTensors.cs:38`; F32, F16, BF16 only: `SafeTensorType` `:14`), `SafeTensorsWriter`, `CheckpointFormats`, `StoredWeights.Conv2d` / `BatchNorm` / `Linear` (`Models/StoredWeights.cs:68, 94, 129`: one tensor read straight into a layer; no `GroupNorm` or `LayerNorm` helper yet), `ImageModels.Load` and `IImageModelFamily` (core, `Idrak.Models.Abstractions`), `PretrainedModel.Create` / `Load` / `SaveHuggingFace`, `BpeTokenizer` (`Models/BpeTokenizer.cs`; normalizers `:45-50, :153`), `ModelPackage` (.ikm) |
| Text | `TextEncoder : IEmbedder` (`src/Idrak.Nlp/Retrieval/TextEncoder.cs:19`) gives pooled embeddings only, not per-token hidden states; a CLIP text model is core's `Embedding` + `PositionalEncoding` + `TransformerEncoderLayer(causal: true)` + `LayerNorm`, which is a family's job (plug-in) |
| Command line | `Command` (`src/Idrak.Cli.Commands/Command.cs:7-41`: `Name`, `Aliases`, `Summary`, `Usage`, `ValueOptions`, `Flags`, `ShortForms`, `BeforePlugins`, `Run`), `CommandTable.cs:21` (`("Train", TrainCommands.All)`), `TrainCommands.All` (`Commands/TrainCommands.cs:11-27`: tune, train, resume, runs, predict, package, distill, create, pretrain), `-P/--plugin`, `RunLogs` (`Shared/RunLogs.cs:55 RunsFolder, :194 RunLogWriter`), `ProgressLine`, `EnvironmentVariables.cs:29-74` (the `OnDevice`, `Models`, `Data` command lists), `MeasuredBatches` (`Commands/Train/ImageModelSupport.cs:183`, internal to the CLI: plan 13 noted the ArabicOcr app had to copy it), `DeviceMemory` (`Shared/DeviceMemory.cs:12`) |
| Registries and kit | `SlotTable<TKey, TValue>` (`src/Idrak.Abstraction/Overrides/SlotTable.cs:32`: `Register` `:128`, `RegisterDefault` `:161`, `Find` `:210`, `Unregister`, `SetPolicy`, `Origin`, `Default`), the pattern every registry copies (`RandomTextSources`, `src/Idrak.Nlp/Abstractions/RandomTextData.cs:31`), one `LibraryRegistrations` per assembly (six exist), `Idrak.Abstraction.Testing` (`ContractSuite<T>`, `Conformance.CheckX`, `DeviceCases.*` replayed per device, `OperationCalls.g.cs`) |
| References | `tools/vlm/make_tiny_*.py`, `tools/pytorch/*_reference.py`; fixtures under `tests/Idrak.Tests/data/<name>/` with `manifest.json` and a README (tolerances as `vlm-tuning/README.md`) |

What is missing, in one line each: noise schedules and samplers; timestep embedding; residual blocks with a timestep
injection; cross-attention blocks over image positions; a UNet and a DiT from a spec; a KL autoencoder and its loss;
class and text conditioning as a contract; classifier-free guidance; a generation API and command; a denoising
objective and trainer with EMA; a diffusers-layout loader; a fused group-norm activation kernel; a per-plane add
kernel; bfloat16 convolution weights (left in plan 13 step 6); measured batching inside the library.

## Separate package or plug-in sample?

**Decided by this plan: both, split exactly as plans 13 and 19 split vision.** The general machinery is the package
`src/Idrak.Diffusion` (`Idrak.Diffusion`, namespace `Idrak.Diffusion`, contracts in `Idrak.Diffusion.Abstractions`);
every model family is a plug-in sample under `samples/`. From the repository's own rules:

1. **No family in the library** (`CLAUDE.md`). Stable Diffusion's tensor names (`down_blocks.N.attentions.M.*`), its
   VAE's scaling factor, its CLIP tokenizer and 77-token limit, SDXL's two text encoders and added conditioning, and the
   from-scratch family's block layout are family facts. They are what `samples/StableDiffusion/Idrak.StableDiffusion`
   and `samples/IdrakDiffusion/Idrak.IdrakDiffusion` register, as `samples/IdrakVision` registers its family (plan 19,
   decision 1: "anyone can create any model", the library holds the contract, never a family).
2. **Abstraction through contracts and `SlotTable` registries** (`CLAUDE.md`; plan 10 decision 10,
   `plans/10-abstraction.md:466`): schedules, samplers, latent codecs, conditioners, objectives and families are
   registries. A contract one package uses lives in that package's `.Abstractions` namespace (`:81-85`); none of these
   has a second library user today, so they start in `Idrak.Diffusion.Abstractions`, and the inventory test
   (`plans/10-abstraction-inventory.md`) moves one to `Idrak.Abstraction` when a second package uses it.
3. **Why a package and not a sample:** `plans/plugins/README.md` defines plug-in packages as "a dependency the library
   must not have"; diffusion has none (pure C#, kernels generated in C# as rule 70 and plans/README's "rule every plan
   keeps" require), so it is library work. `CLAUDE.md` "Speed and memory first ... fixed in the library, an app or
   sample never works around it": the denoising pass's kernels, graphs, attention at latent resolution and memory
   ladders are library work that every family needs; a sample cannot add a kernel. Plan 10's layout already names the
   package (`:432`) and `LibraryDefaults` already looks for it.
4. **Why generic blocks (UNet, DiT, VAE) are in the package and not in the plug-in:** plan 13 put residual blocks in
   its test plug-in and plan 19 put the IdrakVision tower in its plug-in, but both were one family's composition of
   core layers. A `UNet2D` whose every shape is a spec field (channels per level, blocks per level, attention levels,
   heads, context width, groups) is the diffusion analogue of core's `TransformerEncoderLayer` and `DecoderSpec`: no
   checkpoint name, no family default, every value a setting. Plan 10's table lists "UNet, DiT, VAE" under the package,
   and the fused kernels of D3 apply inside these blocks. A family plug-in owns the names, the config files, the
   tokenizer and the composition (decision 2 below asks the owner to confirm).
5. **References.** `Idrak.Diffusion` references `Idrak` and `Idrak.Imaging` only (as `Idrak.Vision` does); not
   `Idrak.Nlp` (its trainer is token-shaped and internal, its `TextEncoder` pools). The CLI references `Idrak.Diffusion`
   as it references `Idrak.Vision`. The plug-ins reference `Idrak`, `Idrak.Imaging` and `Idrak.Diffusion` as packages
   (`IdrakFromSource` in this clone, `samples/Directory.Build.props:14`).

## Where each part lives (decisions 8 and 10 of plan 10)

| Part | Package and namespace | Why |
|---|---|---|
| `INoiseSchedule` + `NoiseSchedules`, `ISampler` + `Samplers`, `ILatentCodec` + `LatentCodecs`, `IConditioner` + `Conditioners`, `IDiffusionObjective` + `DiffusionObjectives`, `IDiffusionFamily` + `DiffusionFamilies`, `IDiffusionDataSource` + `DiffusionDataSources`, the records they speak in | `Idrak.Diffusion.Abstractions` | one library user each; moved to `Idrak.Abstraction` by the inventory test when a second appears |
| Built-in schedules ("linear", "scaled-linear", "cosine"), samplers ("ddpm", "ddim", "euler", "dpm++2m"), codec "pixels", conditioners "none" and "class", objectives ("epsilon", "v", "sample"), data sources ("images", "captions") | `Idrak.Diffusion` | the library defaults, registered by their own registries (as Vision's are) |
| `TimestepEmbedding`, `ResidualBlock2d`, `SpatialTransformer`, `Downsample2d`, `Upsample2d`, `UNet2D` + `UNetSpec`, `DiT` + `DiTSpec`, `AutoencoderKl` + `AutoencoderSpec` | `Idrak.Diffusion` | generic blocks, every shape a spec field (see "Separate package" 4) |
| New device operations (`GroupNormActivation`, `AddPerPlane`), their CPU kernels, CUDA PTX and Vulkan SPIR-V | `Idrak.Abstraction` (`Backend.cs`, generated files), `Idrak.Gpu` | operations are the device contract's (plan 9); host fallback by generation |
| `MeasuredBatches` (moved from the CLI and generalised) | `Idrak.Abstraction.Devices` | Vision's predictors, Diffusion's generator and the CLI use it: two library packages |
| `ImageGenerator`, `ImageGenerationRequest` / `Options` / `Progress` / `Result`, `DiffusionModel`, `DiffusionModels.Load` / `Create`, `SamplerPlan`, `Guidance` | `Idrak.Diffusion` | the generation API |
| `DiffusionTrainer`, `DiffusionTrainingOptions` / `Progress` / `Result` / `Checkpoint`, `EmaWeights`, `LatentCache` | `Idrak.Diffusion` | the loop; reuses the optimizers, schedulers, state save and load, memory scopes, graphs and telemetry of core and Abstraction |
| `idrak diffuse`, `idrak create` (a `"diffusion"` spec), `idrak pretrain` / `resume` / `runs show` on a diffusion folder, `idrak explain` of one | `Idrak.Cli` (`Commands/Run/DiffuseCommand.cs`, `Commands/Train/`) | the surface |
| The from-scratch family (`IdrakDiffusionPipeline`: pixel or latent UNet, class or decoder-hidden-state conditioning, `Create` / `Read` / `Config`) | `samples/IdrakDiffusion/Idrak.IdrakDiffusion` | a family is a plug-in |
| The Stable Diffusion loader family (`StableDiffusionPipeline`: UNet2DConditionModel, AutoencoderKL, CLIPTextModel, CLIP tokenizer, scheduler config) | `samples/StableDiffusion/Idrak.StableDiffusion` | a family is a plug-in |
| References and fixtures | `tools/diffusion/make_tiny_diffusion.py`; `tests/Idrak.Tests/data/diffusion/` | as plans 11 to 19 did |

## Conventions every phase follows (read before coding)

- **A registry** copies `RandomTextSources` (`src/Idrak.Nlp/Abstractions/RandomTextData.cs:31`): a `SlotTable<string, T>`
  named after the class, `comparer: StringComparer.OrdinalIgnoreCase`, library defaults through `RegisterDefault`,
  `[MethodImpl(NoInlining)] Register(T)` with `Assembly.GetCallingAssembly()`, `Unregister`, `Names`, `Find`, `Get`,
  `Default`, `Origin`, `Describe`. `Get`'s message: `"{Kind} '{name}' is not registered (registered: {list}); add it
  with {Registry}.Register."`. Guarded (`SlotGuard`: Throw / FallBack / Shadow) where two implementations can answer
  the same call (samplers, schedules, objectives: Shadow compares the tensors); `unguarded:` with the reason where they
  cannot (families, codecs, conditioners: their outputs are their own).
- **A contract suite** per registry in `src/Idrak.Abstraction.Testing/Contracts/` would need the kit to name
  `Idrak.Diffusion`, which it cannot (it depends on `Idrak.Abstraction` alone, as plan 13 step 4 found); the suites take
  delegates (as `VisionLossSuite` does) and live in the kit: `SamplerSuite`, `NoiseScheduleSuite`, `LatentCodecSuite`.
- **Tests** are entries of a `(string Name, Action<Device> Run)[] DiffusionGroup` in a `Tests` partial file added to
  `Tests.All` (`tests/Idrak.Tests/Program.cs:234`); names start with the group prefix (`"diffusion: ..."`,
  `"diffusion training: ..."`, `"cli diffuse: ..."`, `"stable diffusion: ..."`); `TestData("diffusion/...")` finds
  fixtures; CLI tests run in-process through `CommandLine.Run`; registries are left as found (`Unregister` in `finally`).
- **Fixtures** come from `tools/diffusion/make_tiny_diffusion.py`, whose reruns write the same bytes (`torch.manual_seed`,
  one thread, deterministic algorithms, sorted JSON, `shutil.rmtree` first), with `manifest.json` (versions, sizes,
  SHA-256) and a hand-written README; tolerances: sampler arithmetic within 1e-6, losses within 1e-4, gradients within
  1e-4 of the largest, values within 1e-5, a generated latent within 1e-4 after five steps, ids exact.
- **A command** is an `internal sealed class XCommand : Command` added to a command list, with an Arabic `Summary` entry
  (`Messages.Arabic.cs`), `Usage` sections Arguments, Options, Examples, `-x, --long` short forms, required options as
  `UsageException`, library exceptions turned into `UsageException`, progress through `ProgressLine`, `-j` JSON,
  `ExitCodes`; the command added to the lists in `src/Idrak.Cli.Commands/Shared/EnvironmentVariables.cs`. The tests "cli
  polish", "cli health", "cli arabic" and "cli train data" check all of this.
- **A new device operation** is a `NameKernel` on `Backend` with a `<summary>` and a host-fallback body, then
  `python3 tools/operations/generate.py` (it writes `Operations.g.cs`, `Backend.Operations.g.cs` and the kit's
  `OperationCalls.g.cs`); a CPU kernel in `Devices/Cpu/`, PTX in `src/Idrak.Gpu/Cuda/PtxKernels.*.cs` and SPIR-V in
  `src/Idrak.Gpu/Vulkan/VulkanKernels.*.cs`, a `DeviceCases` entry replayed per device, `idrak kernels` showing it.
- **Regenerate** after a public change: the inventory with
  `IDRAK_UPDATE_INVENTORY=1 IDRAK_DEVICES=cpu IDRAK_FILTER=inventory dotnet run -c Release --project tests/Idrak.Tests`
  and the API dump (a new `api/Idrak.Diffusion.txt`) with
  `IDRAK_UPDATE_API=1 IDRAK_DEVICES=cpu IDRAK_FILTER="public API" dotnet run -c Release --project tests/Idrak.Tests`.
- **Progress and telemetry.** Loops report through `IProgress<T>` and publish `Telemetry.TrainingStarted`,
  `BatchCompleted` (one per optimizer step), `TrainingCompleted`; generation publishes `InferenceStage` per stage
  (conditioning, denoising, decoding) with the device bytes; `IsEnabled` is one static read and an AND.

## Memory and speed: where the bytes go

Stated as formulas, measured by the library, never as a card's budget. B is the batch (times 2 under classifier-free
guidance in one pass), h × w the latent size (the image size over the codec's factor; the image itself for "pixels"),
T = h·w the attention positions, C the channels of a level, d the head size, L the conditioning length.

- **Latents:** B·C_latent·h·w floats; the sampler keeps one (DPM++ 2M: two) latent-sized tensors beside the current one.
- **The denoiser pass:** the largest activations are at the first level: a handful of [B, C_0, h, w] tensors alive at
  once through a residual block, plus the composed convolution's unfolded patches [B·h·w, C·9] where the measured choice
  takes the composed path (chunked by the existing rule: a storage's range or half the reported free memory; the
  implicit kernels never unfold). Self-attention at that level: the span kernels (`AttentionSpans`) never store the
  [B·heads, T, T] scores; the composed path stores them only where `Tensor.AttentionFastest` measured it faster and the
  device has the room (`AvailableMemory`), and `IDRAK_ATTENTION_PATH=spans` forces the kernels. Head sizes of the known
  families (Stable Diffusion 1.x: 40, 80, 160 at 8 heads; 2.x and SDXL: 64; DiT: 64 or 72) are within the reported
  `SpanAttentionHeadDim` on CUDA and Vulkan; the CPU takes any. The self-attention K and V at image resolution are
  [B·heads, T, d] each, read in place from the packed projection where the device has `AttentionSpansStrided`; the
  cross-attention K and V are [B·heads, L, d], small.
- **Weights:** the denoiser's parameters once (float32, or bfloat16 in `Linear` as stored; convolution weights float32
  until the bfloat16 convolution item of D3 lands); the codec's and the conditioner's beside them, or offloaded
  (`IDRAK_OFFLOAD`, `IMemoryOffload`) when a component is idle: the conditioner runs once per request, the codec once at
  the end, so both may leave the device between uses (a setting, `ComponentPlacement`).
- **Decoding:** the codec's decoder works at pixel resolution with C_dec channels: [B, C_dec, H, W] tensors are the
  peak of a generation at large sizes. `DecodeTiles` (auto by default) decodes in overlapping tiles when a whole decode
  throws `ResourceLimitExceededException`, halving the tile until it fits (the ladder is said once in the trace).
- **Training:** the above plus the backward pass; `ActivationMemory.CompressToBFloat16` where plan 19's loop already
  takes it, `CheckpointBlocks` per level, micro-batches halved on `ResourceLimitExceededException`; the EMA copy of
  every trained parameter (on the device, or on the host with `EmaOnHost`); the optimizer's moments (8-bit with
  `adamw8bit`, on the host with `HostOptimizer`); the latent cache (B_total·C_latent·h·w floats on disk or in memory).
- **Measured, not assumed:** `Backend.ResetPeakMemoryUsage()` before a pass, `GetMemoryUsage().Peak` after it, reported
  in `ImageGenerationProgress.DeviceBytes` and `DiffusionTrainingProgress.DeviceBytes`, in `-v` and `-j` of the commands,
  in `log.jsonl`, and through `InferenceStage` / `BatchCompleted` to any listener; `--profile` runs the pass under
  `Profiler.Run` for the kernel breakdown. The batch of a request is a setting with an automatic value from
  `MeasuredBatches` (the first request one image, its peak against half the free memory, halved on failure), as
  `idrak predict` sizes its batches today.
- **Graphs:** on a device that records graphs, the denoiser pass (conditioning concatenated, the denoiser, the guidance
  combine) is recorded once per (B, h, w, L, precision) and replayed per step; the timestep is a device tensor the
  sampler updates with `CopyFrom` between replays (no host scalar is frozen), and the sinusoidal embedding is a device
  table [train steps, dim] built once and gathered by that index (rule 85); the sampler's own step (three to five
  element-wise launches with that step's host coefficients) runs outside the graph, so no coefficient is frozen. On
  Vulkan the measured graph choice of plans/README ("Future improvements") applies, `IDRAK_VULKAN_GRAPHS` with it;
  `ImageGenerationOptions.Graphs = false` and `--no-graphs` turn it off anywhere.

---

## Phase D0: the reference

**What.** `tools/diffusion/make_tiny_diffusion.py` (torch, diffusers, transformers, tokenizers, Pillow; imports
`dump_json`, `save_npy`, `sha256`, `safetensors_index` from `tools/vlm/make_tiny.py`; nothing downloaded) writes
`tests/Idrak.Tests/data/diffusion/`:

- `schedules/`: for each built-in schedule and option set (`linear` 1000 steps 1e-4..0.02; `scaled-linear` 1000 steps
  0.00085..0.012; `cosine` 1000 steps; `scaled-linear` with zero-terminal-SNR rescaling) the `betas` and
  `alphas_cumprod` as float64 `.npy`, computed as diffusers' `DDPMScheduler` computes them.
- `samplers/`: for each sampler (DDPM, DDIM with eta 0 and eta 1, Euler, DPM-Solver++ 2M) and step count (5, 10) with
  spacing `leading` and `trailing`: the timesteps, a seeded initial latent [2, 4, 8, 8], a seeded sequence of fake model
  outputs (one per step, saved), the seeded noise DDPM and DDIM eta 1 draw, and the sample after every step, from
  diffusers' scheduler classes (`DDPMScheduler`, `DDIMScheduler`, `EulerDiscreteScheduler`,
  `DPMSolverMultistepScheduler` with `algorithm_type "dpmsolver++"`, `solver_order 2`). The sampler arithmetic is
  then checked without any network, to 1e-6.
- `tiny-unet/`: a `UNet2DConditionModel` (sample 16, 4 in and out channels, `block_out_channels (32, 64)`,
  `down_block_types ("DownBlock2D", "CrossAttnDownBlock2D")`, `up_block_types ("CrossAttnUpBlock2D", "UpBlock2D")`,
  `layers_per_block 1`, `attention_head_dim 8`, `cross_attention_dim 32`, `norm_num_groups 8`) seeded with
  `torch.Generator().manual_seed(2028)`, saved with `save_pretrained` (safetensors); `reference/`: `latents` [2, 4, 16,
  16], `timesteps` [2], `encoder_hidden_states` [2, 7, 32], the output, and the output of each down block (the blocks
  are checked one by one before the whole); `step/`: one training step (seeded noise and timesteps, the epsilon MSE
  loss, every parameter's gradient, plain SGD 0.2) and three steps' losses and weights, `v`-prediction's loss too.
- `tiny-vae/`: an `AutoencoderKL` (`block_out_channels (32,)`, `latent_channels 4`, `layers_per_block 1`,
  `norm_num_groups 8`, `sample_size 32`, `scaling_factor 0.18215`): the posterior mean and log-variance of the fixture
  image, a seeded sample, its decode, and one reconstruction-plus-KL training step.
- `tiny-clip/`: a `CLIPTextModel` (hidden 32, 2 layers, 4 heads, intermediate 37, positions 16) and a tiny CLIP-style
  tokenizer trained by `tokenizers` with `end_of_word_suffix "</w>"`, the CLIP normalizer (NFC, lowercase) and
  pre-tokenizer, `<|startoftext|>` / `<|endoftext|>`, written as `tokenizer.json` and as `vocab.json` + `merges.txt`;
  ten prompts with their ids, the last hidden state, the penultimate hidden state (`clip_skip`) and the pooled output.
- `tiny-pipeline/`: `model_index.json` (`StableDiffusionPipeline`) tying `unet/`, `vae/`, `text_encoder/`,
  `tokenizer/`, `scheduler/` (a `DDIMScheduler` config, `scaled_linear`, `prediction_type epsilon`); `generated/`: two
  prompts, one negative prompt, 5 steps, guidance 7.5, seed 0, on the CPU, with the latent after every step
  (`callback_on_step_end`) and the decoded image as `.npy` and PNG; a second run with `euler` and `dpm++2m`.
- `pixel-ddpm/`: a `UNet2DModel` (16 × 16, 3 channels, `block_out_channels (32, 64)`, `num_class_embeds 4`) for the
  from-scratch family: 24 seeded synthetic training images (shapes drawn by the script, four classes, PNG) with
  `labels.csv`, one training step (seeded noise and timesteps, loss, gradients) and three steps' losses and weights with
  SGD; an EMA of decay 0.9 over the three steps (so the EMA arithmetic is checked); 5 DDPM sampling steps from a seeded
  latent with the untrained weights.
- `manifest.json`, `README.md` (what the script did, versions, tolerances, how to regenerate).

**Done when** the folder exists, two runs write the same bytes, the README names the versions and the commands.

## Phase D1: contracts and registries

**Files.** `src/Idrak.Diffusion/Idrak.Diffusion.csproj` (as `Idrak.Vision.csproj`: `IsAotCompatible`,
`GenerateDocumentationFile`, references `Idrak.Imaging`), `README.md`, `LibraryRegistrations.cs` (empty table at first),
`Abstractions/NoiseSchedules.cs`, `Abstractions/Samplers.cs`, `Abstractions/LatentCodecs.cs`,
`Abstractions/Conditioners.cs`, `Abstractions/DiffusionObjectives.cs`, `Abstractions/DiffusionFamilies.cs`,
`Abstractions/DiffusionDataSources.cs`; `Idrak.slnx` (+1 under `/src/`), `tests/Idrak.Tests/Idrak.Tests.csproj` (+1
reference), `src/Idrak.Cli/Idrak.Cli.csproj` (+1 reference), `api/Idrak.Diffusion.txt` (new).

**Contracts** (`Idrak.Diffusion.Abstractions`):

```csharp
public sealed record NoiseScheduleOptions
{
    public int TrainSteps { get; init; } = 1000;
    public double BetaStart { get; init; } = 1e-4;
    public double BetaEnd { get; init; } = 0.02;
    public bool ZeroTerminalSnr { get; init; }
}
public sealed class NoiseSchedule(string name, ReadOnlyMemory<double> betas)   // alphas, alphas_cumprod, sqrt(...) derived once, in double
{
    public string Name { get; } public int TrainSteps { get; }
    public ReadOnlySpan<double> Betas { get; } public ReadOnlySpan<double> AlphasCumprod { get; }
    public double SignalToNoise(int t);
    public Tensor Table(Device device);            // [TrainSteps, 4] float: sqrt(ā), sqrt(1 - ā), ... built once per device, gathered by a device timestep
}
public interface INoiseSchedule { string Name { get; } string Summary { get; } NoiseSchedule Create(NoiseScheduleOptions options); }
public static class NoiseSchedules   // "linear", "scaled-linear", "cosine"; guarded (Shadow compares AlphasCumprod)

public sealed record SamplerOptions
{
    public int Steps { get; init; } = 30;
    public string Spacing { get; init; } = "leading";    // leading | trailing | linspace
    public float Eta { get; init; }                      // DDIM's stochasticity, 0 deterministic
    public bool KarrasSigmas { get; init; }
    public int Seed { get; init; }
}
public abstract class SamplerPlan     // built once per request on the host, in double (rule 85)
{
    public ReadOnlySpan<int> Timesteps { get; }
    public abstract int History { get; }                 // latent-sized tensors kept (DPM++ 2M: 1)
    public abstract Tensor Step(int index, Tensor sample, Tensor modelOutput, Func<Tensor> noise);   // one fused MulAdd / Axpy sequence in a fixed order
    public abstract Tensor ScaleInput(int index, Tensor sample);                                      // Euler's 1 / sqrt(σ² + 1); identity elsewhere
    public abstract float InitialNoiseScale { get; }
}
public interface ISampler { string Name { get; } string Summary { get; } SamplerPlan Plan(NoiseSchedule schedule, string prediction, SamplerOptions options); }
public static class Samplers          // "ddpm", "ddim", "euler", "dpm++2m"; guarded (Shadow compares the sample after a step)

public sealed record LatentInfo(int Channels, int Factor, float Scale, float Shift = 0f);
public interface ILatentCodec : IDisposable
{
    string Name { get; } LatentInfo Info { get; }
    Tensor Encode(Tensor pixels, Random? posteriorSample = null);   // [B, 3, H, W] in [-1, 1] → [B, C, H / Factor, W / Factor], scaled
    Tensor Decode(Tensor latents, DecodeOptions options);           // → [B, 3, H, W]; options: Tiles (null = measured), Overlap
    IEnumerable<Tensor> Parameters();
}
public static class LatentCodecs      // the factory registry: "pixels" (identity, Factor 1); unguarded (a codec's latent space is its own)

public sealed record Conditioning(Tensor? Tokens, Tensor? Pooled, Tensor? Mask, IReadOnlyDictionary<string, Tensor>? Extra);
public interface IConditioner : IDisposable
{
    string Name { get; } int ContextDim { get; } int? MaxLength { get; }
    Conditioning Encode(IReadOnlyList<ConditioningInput> inputs, Device device);   // prompts, class ids, or both; the unconditional input for guidance
    IEnumerable<Tensor> Parameters();
}
public static class Conditioners      // "none", "class"; unguarded

public interface IDiffusionObjective
{
    string Name { get; }
    Tensor Target(Tensor x0, Tensor noise, Tensor scheduleRows);   // epsilon: noise; v: sqrt(ā)·noise - sqrt(1-ā)·x0; sample: x0
    Tensor Predicted(Tensor modelOutput, Tensor xt, Tensor scheduleRows, out Tensor epsilon);   // the sampler's view of the output as ε and x0
    Tensor Weight(Tensor snr, float? gamma);                         // 1, or min-SNR
}
public static class DiffusionObjectives   // "epsilon", "v", "sample"; guarded (Shadow compares the target)

public interface IDenoiser : IDisposable
{
    Tensor Predict(Tensor latents, Tensor timesteps, Conditioning conditioning);   // [B, C, h, w], [B] on the device → [B, C, h, w]
    DenoiserInfo Info { get; }    // latent channels, sample size, context dim, prediction type, parameter count
    IEnumerable<Tensor> Parameters(); Module Network { get; }
}
public sealed class DiffusionModel : IDisposable   // the pipeline a family returns
{
    public required string Architecture { get; init; } public required IDenoiser Denoiser { get; init; }
    public required ILatentCodec Codec { get; init; } public required IConditioner Conditioner { get; init; }
    public required NoiseSchedule Schedule { get; init; } public required string Objective { get; init; }
    public string? Folder { get; init; } public JsonObject? Spec { get; init; } public IReadOnlyList<string> Notes { get; init; }
}
public interface IDiffusionFamily
{
    string Name { get; } string Summary { get; }
    DiffusionModel Read(DiffusionCheckpoint checkpoint, DiffusionModelOptions options);
    DiffusionModel Create(DiffusionSpec spec, DiffusionModelOptions options) => throw new NotSupportedException($"Diffusion family '{Name}' cannot be created from a spec (it reads checkpoints only).");
    void Save(DiffusionModel model, string folder, SafeTensorType type);   // the family's own layout (model_index.json and its parts)
}
public static class DiffusionFamilies     // empty in the library; unguarded
public sealed record DiffusionCheckpoint(string Folder, JsonObject Index, Func<string, ITensorStore> Open, Func<string, JsonObject?> Config);   // model_index.json, a store per part folder
public sealed record DiffusionSpec(string Architecture, JsonObject Denoiser, JsonObject? Codec, JsonObject? Conditioner, NoiseScheduleOptions Schedule, string ScheduleName = "scaled-linear", string Objective = "epsilon");
public sealed record DiffusionModelOptions { Device? Device; EncoderWeights Weights = AsStored; string? Precision; ComponentPlacement Placement = Device; int Seed; }

public interface IDiffusionDataSource { string Name; string Summary; IDiffusionReader Open(DiffusionDataSpec spec, DiffusionDataContext context); }
public static class DiffusionDataSources   // "images" (a folder or class folders), "captions" (JSON Lines with image and text); unguarded
```

`DiffusionModels` (`Idrak.Diffusion`): `Load(path, options)` reads `model_index.json` (its `_class_name`, or
`config.json`'s `architectures` for a single-part folder) and asks `DiffusionFamilies.Get(name).Read(...)`;
`Create(spec, options)` asks `Create`; an unregistered architecture is refused before any tensor is opened, naming the
registry and `-P`.

**Tests** (`"diffusion: ..."`, `tests/Idrak.Tests/DiffusionTests.cs`): every registry refuses an unknown name with its
message and lists the built-ins; `Describe` of each; `Register` / `Unregister` round trip with `Origin` the test
assembly; `NoiseScheduleOptions` and `SamplerOptions` JSON round trip; `DiffusionModels.Load` on a folder whose
`_class_name` is unregistered fails naming `DiffusionFamilies.Register`. Inventory and API regenerated (a new
assembly in the public API guard and the namespace test).

**Done when** the package builds, the tests pass on the CPU, `IDRAK_FILTER="diffusion:"` is the only run.

**Risks.** The namespace test's rule (contracts under `Idrak.Diffusion.Abstractions`, implementations anywhere) must
hold from the first commit, or the inventory group fails: run `"abstraction inventory"` with the phase's filter.

## Phase D2: schedules and samplers

**Files.** `src/Idrak.Diffusion/Schedules.cs` (the three schedules; betas in double; the zero-terminal-SNR rescale as
diffusers' `rescale_zero_terminal_snr`), `Samplers/Ddpm.cs`, `Samplers/Ddim.cs`, `Samplers/Euler.cs`,
`Samplers/DpmSolverPlusPlus.cs` (2M, with the first step's first-order fallback and `lower_order_final`),
`Samplers/Timesteps.cs` (leading, trailing, linspace spacings; Karras sigmas), `Samplers/SamplerPlan.cs`.

**Design.** A plan holds, per index, the coefficients of its update as doubles computed once; `Step` applies them as
`sample = a·sample + b·modelOutput (+ c·noise)` through `MulAdd` / `Axpy` in the order (a, b, c) on every device
(rule 86: no summation order depends on the device), so a step is at most five launches and allocates one latent-sized
tensor (the result; `History` tensors kept by the plan). `noise` is drawn by the caller from a seeded host generator
(`Tensor.Normal` with `Random(seed)`, uploaded; the same bits on every device, as the dropout mask is) and asked only
by samplers that need it. `ScaleInput` is a scalar `Affine`. Prediction types go through the objective's `Predicted`
(epsilon, v, sample), so every sampler works with every objective.

**Tests** (`"diffusion: schedules ..."`, `"diffusion: samplers ..."`): each schedule's `Betas` and `AlphasCumprod` equal
`schedules/` to 1e-12 in double; the zero-terminal-SNR rescale; the timesteps of every (sampler, steps, spacing) equal
the fixture's exactly; the sample after every step equals `samplers/` to 1e-6 (fake model outputs fed in, the fixture's
noise fed to DDPM and DDIM eta 1), on the CPU and replayed on the device (the same bits: `SequenceEqual` against the CPU
run); the kit's `SamplerSuite` (finite, deterministic for a seed, a step with zero model output under DDIM eta 0 is the
rescaled sample, history length honoured) passes for every built-in and fails a sampler whose `Step` reorders its sum;
a plan with `Steps` above `TrainSteps` is refused.

**Done when** every sampler matches diffusers on the CPU bit for bit within 1e-6 and the device run equals the CPU run.

**Risks.** diffusers' Euler uses sigmas from `alphas_cumprod` with `interpolation_type linear` and its own timestep
rounding; the script records diffusers' exact timesteps, and the test checks them before any sample, so a rounding
difference fails by name. DPM++ 2M's `final_sigmas_type` and `lower_order_final` defaults are recorded in the fixture's
config and mirrored as settings.

## Phase D3: the denoiser building blocks and their kernels

**Files.** `src/Idrak.Diffusion/Blocks/TimestepEmbedding.cs`, `ResidualBlock2d.cs`, `SpatialTransformer.cs`,
`Downsample2d.cs`, `Upsample2d.cs`, `UNet2D.cs` + `UNetSpec.cs`, later `DiT.cs` + `DiTSpec.cs`;
`src/Idrak.Abstraction/Devices/Backend.cs` (+2 `NameKernel`), the generated files, `Devices/Cpu/CpuBackend.Diffusion.cs`,
`src/Idrak.Gpu/Cuda/PtxKernels.Diffusion.cs` + `CudaBackend.Diffusion.cs`, `src/Idrak.Gpu/Vulkan/VulkanKernels.Diffusion.cs`
+ `VulkanBackend.Diffusion.cs`; `src/Idrak.Abstraction.Testing/DeviceCases.Diffusion.cs`.

**Blocks** (every shape a constructor argument; parameters named for the family's `Export`):

```csharp
public sealed class TimestepEmbedding(int channels, int timeDim, int trainSteps, bool flipSinCos = true, float downscaleShift = 0f, Device? device = null) : Module
    // a device table [trainSteps, channels] of sin and cos built once, gathered by a device timestep index (no host copy in a graph), then Linear, SiLU, Linear
public sealed class ResidualBlock2d(int inChannels, int outChannels, int timeDim, int groups = 32, float epsilon = 1e-5f, float dropout = 0f, Device? device = null, Random? random = null) : Module
    // GroupNorm → SiLU → Conv3x3 → + time projection per (sample, channel) → GroupNorm → SiLU → Dropout → Conv3x3 → + skip (1x1 when channels differ)
    public Tensor Forward(Tensor x, Tensor timeEmbedding);
public sealed class SpatialTransformer(int channels, int heads, int headDim, int contextDim, int depth = 1, int groups = 32, bool linearProjection = false, ...) : Module
    // GroupNorm → 1x1 (or Linear) → [B, T, C] → depth × (LayerNorm → self-attention → LayerNorm → cross-attention(context) → LayerNorm → GEGLU feed-forward) → 1x1 → + x
    public Tensor Forward(Tensor x, Tensor? context);
public sealed class Downsample2d(int channels, bool conv = true) : Module;   // stride-2 3x3, or average pooling
public sealed class Upsample2d(int channels, bool conv = true) : Module;     // nearest ×2 (Interpolate2d) then 3x3
public sealed record UNetSpec
{
    public int InChannels { get; init; } = 4; public int OutChannels { get; init; } = 4;
    public IReadOnlyList<int> Channels { get; init; } = [320, 640, 1280, 1280];
    public IReadOnlyList<bool> Attention { get; init; } = [true, true, true, false];
    public int BlocksPerLevel { get; init; } = 2; public int Heads { get; init; } = 8; public int? HeadDim { get; init; }
    public int ContextDim { get; init; } = 768; public int TransformerDepth { get; init; } = 1;
    public int Groups { get; init; } = 32; public float Epsilon { get; init; } = 1e-5f; public float Dropout { get; init; }
    public int? ClassCount { get; init; }              // a class embedding added to the time embedding
    public int TrainSteps { get; init; } = 1000; public bool FlipSinCos { get; init; } = true; public float FreqShift { get; init; }
    public JsonObject ToJson(); public static UNetSpec FromJson(JsonObject);
}
public sealed class UNet2D(UNetSpec spec, Device? device = null, Random? random = null) : Module, IDenoiser
    // conv in → levels (res, attn?) with skips → middle (res, attn, res) → levels up (concat skip, res, attn?, upsample) → GroupNorm, SiLU, conv out
    public Tensor Predict(Tensor latents, Tensor timesteps, Conditioning conditioning);
    public Tensor ForwardTraining(Tensor latents, Tensor timesteps, Conditioning conditioning, bool checkpointLevels);
```

**Kernels: what exists, what is added.**

| Need | Today | D3 |
|---|---|---|
| GroupNorm → SiLU | two passes: the norm kernels then `Unary` SiLU (one extra read and write of every activation at every block) | `GroupNormActivation(x, gamma, beta, y, groups, channels, positions, epsilon, activation)`: statistics and the affine with the activation in one apply pass (plan 13 left "a fused group norm"); CPU kernel, PTX, SPIR-V; its backward composes the existing `NormBackward` with the activation's slope (`UnaryBackward`), exact |
| x[n, c, :, :] += e[n, c] (the time projection) | `AddRowVector` adds one vector along a row; a per-row scalar over `[N·C, H·W]` has no single launch (a `BroadcastAxis` copy then `Binary`) | `AddPerPlane(x, scalars, y, planes, positions)`: one launch, in place; backward is `SumRows` (exists) |
| Timestep embedding | `Sin`, `Cos` exist; a host sin/cos upload would break graph recording | a device table gathered by index (`Gather` exists); no kernel |
| Self-attention at T = h·w, head sizes 40 to 160 | `AttentionFastest` / span kernels within `SpanAttentionHeadDim`; composed where the scores fit and measure faster | no kernel; the layer takes `AttentionFastestPacked` as `MultiHeadAttention` does; head sizes above the reported limit compose (plan 17 A3), never fall to the host |
| Cross-attention over L context keys | `AttentionFastest(q, keys, values, ...)` with every-key ranges | no kernel |
| Nearest ×2 upsampling, strided 3x3 downsampling | `Interpolate2d` kernels on CUDA and Vulkan; strided `Convolution` | no kernel |
| GEGLU | `GatedActivation` (gate × GELU) exists | a GELU variant flag if the kernel has SiLU only (checked first) |
| Gaussian noise on the device | none | not added: host Box-Muller, seeded, uploaded (the same bits on every device); a device generator is a later measured item |
| bfloat16 convolution weights | `Linear` keeps bfloat16; `Conv2d` holds float32 | `Conv2d.KeepsBFloat16` for inference through the composed path's `BFloat16MatMul` (the implicit kernels stay float32 and the measured choice weighs both); left to a follow-up if the phase runs long |

Every added operation has the host fallback by generation (plan 7's minimum-backend rule), a `DeviceCases` entry
("diffusion kernels: fused group norm with each activation and its gradient; per-plane add and its gradient") against
plain loops, and shows in `idrak kernels`. Activation checkpointing per level through `ActivationMemory.CheckpointBlocks`
and `Checkpointing.Checkpoint` (made `internal` visible or wrapped by a public `Module.Checkpointed`, decision 10 keeps
it in core).

**Tests** (`"diffusion: blocks ..."`): `TimestepEmbedding`'s table equals diffusers' `get_timestep_embedding` for
every t to 1e-6 (flip and shift variants); `ResidualBlock2d`, `SpatialTransformer`, `Downsample2d`, `Upsample2d` loaded
from `tiny-unet` tensors through the test's own name map match the recorded block outputs to 1e-5; `UNet2D` from the
spec of `tiny-unet/config.json` with its weights matches `reference/` output to 1e-5 in float32 (the test maps names;
the family of D7 does it for real); finite differences on every block; the fused group norm equals the two-pass path
bit for bit on the CPU and to 1e-6 on the device, forward and gradient; the per-plane add; a `UNet2D` of random
weights gives the same output on the CPU and the device to 1e-5 with no host fallback for the kernels the device has
(`Kernels.HostCalls` counted); checkpointed levels give the uncheckpointed gradients bit for bit; `"kernel shapes"`,
`"spirv"`, `"operations"`, `"abstraction inventory"`, `"public API"` pass.

**Done when** the tiny UNet matches diffusers on the CPU and the device test reports no host fallback for group norm,
convolution, interpolation and attention on Vulkan (lavapipe here; CUDA the owner's).

**Risks.** The composed convolution path's patches at C = 1280 and small h·w are fine, at C = 320 and h·w = 4096 they
are 47 MB a sample a layer: within the existing chunking rule, and the implicit kernels avoid them; measured per shape,
nothing to tune by hand. DiT (`DiTSpec`: patch size, width, depth, heads, adaLN-Zero, class or text conditioning,
unpatchify) is D3b after D6 proves the loop (decision 5); the contracts do not change for it.

## Phase D4: the latent codec

**Files.** `src/Idrak.Diffusion/Codecs/PixelCodec.cs` ("pixels": identity, `Factor 1`, `Scale 1`),
`Codecs/AutoencoderKl.cs` + `AutoencoderSpec.cs` (encoder: conv in, levels of `ResidualBlock2d` without a time
projection and `Downsample2d`, a middle block with one self-attention, GroupNorm, SiLU, conv to 2·latent channels;
`DiagonalGaussian` (mean, log-variance clamped, a seeded sample); decoder mirrored with `Upsample2d`), `Codecs/DecodeTiling.cs`
(overlapping tiles blended linearly, as diffusers' `enable_vae_tiling`; the tile size measured: a whole decode first,
halved on `ResourceLimitExceededException`, kept per (h, w, device) for the process), `Codecs/AutoencoderLoss.cs`
(reconstruction L1 or L2 + `KlWeight`·KL; no perceptual or adversarial term: both need another network, out of this plan).

```csharp
public sealed record AutoencoderSpec { int Channels = 3; IReadOnlyList<int> Levels = [128, 256, 512, 512]; int BlocksPerLevel = 2; int LatentChannels = 4; int Groups = 32; bool MiddleAttention = true; float Scale = 0.18215f; float Shift; JsonObject ToJson(); static AutoencoderSpec FromJson(JsonObject); }
public sealed class AutoencoderKl(AutoencoderSpec spec, Device? device = null, Random? random = null) : Module, ILatentCodec
{
    public (Tensor Mean, Tensor LogVariance) Posterior(Tensor pixels);
    public Tensor Encode(Tensor pixels, Random? posteriorSample = null);   // mean (null) or mean + exp(logvar / 2)·noise, times Scale, plus Shift
    public Tensor Decode(Tensor latents, DecodeOptions options);
}
public sealed record DecodeOptions { int? Tiles { get; init; }  int Overlap { get; init; } = 8; }   // null: measured
```

**Tests** (`"diffusion: codec ..."`): "pixels" is the identity both ways and reports `Factor 1`; `AutoencoderKl` from
`tiny-vae`'s weights (the test's name map) gives the fixture's posterior mean and log-variance to 1e-5, the decode of
the fixture's sample to 1e-4, and the training step's loss and gradients to 1e-4; tiled decoding equals the whole decode
to 1e-4 inside tiles and blends at the seams (a tolerance setting); the measured tile ladder halves on a device whose
free memory the test lowers through `ComputeResources.GpuMemoryLimit`; the kit's `LatentCodecSuite` (round trip within
a stated tolerance for the identity codec, shapes by `Factor`, deterministic for a seed, the same alone as in a batch).

**Done when** the tiny VAE matches diffusers on the CPU; the SD VAE's weights are the plug-in's (D7).

**Risks.** A VAE trained from scratch at useful quality needs a perceptual loss and long runs (see "What a from-scratch
run is"); D4 proves the block and its loss, not that quality.

## Phase D5: the generation API and `idrak diffuse`

**Files.** `src/Idrak.Diffusion/ImageGenerator.cs`, `Guidance.cs`, `GenerationGraph.cs`;
`src/Idrak.Abstraction/Devices/MeasuredBatches.cs` (moved from `src/Idrak.Cli.Commands/Commands/Train/ImageModelSupport.cs:183`,
generalised: `MeasuredBatches(Device, int? ceiling, Func<long> free)`, `Run(items, batch => ...)`; the CLI and Vision's
`ModelDetector` / `ImageModelPredictors` call it); `src/Idrak.Cli/Commands/Run/DiffuseCommand.cs`, `RunCommands.cs`
(+1), `Shared/EnvironmentVariables.cs` ("diffuse" in `OnDevice` and `Models`), `Messages.Arabic.cs`.

```csharp
public sealed record ImageGenerationRequest
{
    public IReadOnlyList<string> Prompts { get; init; } = [];     // or ClassIds for a class conditioner
    public IReadOnlyList<int>? ClassIds { get; init; }
    public string? NegativePrompt { get; init; }
    public int Width { get; init; } = 512; public int Height { get; init; } = 512;   // defaults from the model's sample size, never fixed here
    public int Steps { get; init; } = 30; public float Guidance { get; init; } = 7.5f;
    public string Sampler { get; init; } = "ddim"; public SamplerOptions? SamplerOptions { get; init; }
    public int Seed { get; init; } public int Count { get; init; } = 1;
    public ImageData? InitImage { get; init; } public float Strength { get; init; } = 0.75f;   // image to image: start at Strength of the schedule
}
public sealed record ImageGenerationOptions
{
    public Device? Device { get; init; }
    public int? Batch { get; init; }                     // null: measured (MeasuredBatches)
    public bool GuidanceInOnePass { get; init; } = true; // cond and uncond as one batch of 2N; false: two passes of N (less memory, measured by the ladder on failure)
    public string? Precision { get; init; }              // null: IDRAK_MATMUL; "bf16": MixedPrecision.BFloat16 around the denoiser
    public bool Graphs { get; init; } = true;
    public DecodeOptions Decode { get; init; } = new();
    public ComponentPlacement Placement { get; init; } = ComponentPlacement.Device;   // or Offload: conditioner and codec leave the device between uses
}
public sealed record ImageGenerationProgress(int Step, int Steps, double SecondsPerStep, long DeviceBytes, TimeSpan Remaining, string Stage);
public sealed record ImageGenerationResult(IReadOnlyList<ImageData> Images, IReadOnlyList<int> Seeds, TimeSpan ConditioningTime, TimeSpan DenoisingTime, TimeSpan DecodingTime, long PeakDeviceBytes, string Settings);
public sealed class ImageGenerator(DiffusionModel model, ImageGenerationOptions? options = null) : IDisposable
{
    public ImageGenerationResult Generate(ImageGenerationRequest request, IProgress<ImageGenerationProgress>? progress = null, CancellationToken cancellationToken = default, Action<string>? trace = null);
    public Tensor Denoise(Tensor latents, Conditioning conditioning, SamplerPlan plan, ...);   // the loop alone, for tests and training evaluation
}
```

**The loop.** Conditioning once (positive and the negative or empty prompt); the initial latent `Tensor.Normal` from
`Random(seed + i)` per image, times `InitialNoiseScale`; per step: the timestep index copied into the persistent device
tensor, the recorded pass replayed (or run: concat of the two halves where `GuidanceInOnePass`, `ScaleInput`,
`Denoiser.Predict`, `Guidance.Combine` as one `MulAdd`), the sampler's `Step`; then `Codec.Decode` with the tiling
ladder; pixels to `ImageData` ([-1, 1] to bytes). Telemetry: `InferenceStage` per stage with `GetMemoryUsage().Peak`;
`BatchCompleted` is not used (not training). A device that records no graphs runs the same code.

**`idrak diffuse MODEL "PROMPT" [-o out.png|DIR]`** (`RunCommands.All`): `--negative TEXT`, `--class N`, `--steps N`,
`--guidance X`, `--sampler NAME`, `--schedule NAME` (over the model's), `--eta X`, `--spacing NAME`, `--width W`,
`--height H`, `--seed N`, `--count N`, `--batch N`, `--init IMAGE --strength F`, `--precision f32|bf16`, `--no-graphs`,
`--two-pass` (guidance in two passes), `--tiles N|auto|off`, `--offload-parts`, `--profile`, `-d`, `-P`, `-j`, `-v`.
MODEL is a folder (diffusers layout or a created one), or a hub id the model cache resolves as `idrak run` does (D7
teaches the hub download the folder layout). Progress: `ProgressLine` "step i/n  ms/step  mem  left"; `-v` prints the
settled batch, the attention path, the graph state and the decode tiles; `-j` the result record. Output names
`NAME-SEED.png` through `ImageEncoders.For(path)`.

**Tests** (`"diffusion: generation ..."`, `"cli diffuse: ..."`): the loop on `tiny-pipeline` through the test's own
family (the D0 name maps) reproduces the latent after each of the 5 steps to 1e-4 and the decoded image to 1e-3 for
every sampler recorded; guidance in one pass and in two give the same latents bit for bit on the CPU; the recorded pass
replayed equals the direct run (on lavapipe the recording's fallback runs the step; on CUDA the owner's run); the
measured batch halves when the test lowers `GpuMemoryLimit`; `--seed` repeats an image byte for byte; `-j` carries the
peak bytes; an unregistered family exits 1 with `-P`; `help` passes the polish tests; `MeasuredBatches` moved: `"cli
vision:"` unchanged.

**Done when** the tests pass on the CPU and the four `cli` health filters still pass.

**Risks.** Image-to-image needs `Encode` of the init image with the codec; "pixels" makes it trivial, the VAE path is
D7's. `GuidanceInOnePass` doubles the pass's activations: the ladder falls to two passes on
`ResourceLimitExceededException`, said once in the trace.

## Phase D6: training from scratch

**Why a `DiffusionTrainer` and not the `Pretrainer`.** `Pretrainer` and `FineTuner.StepRunner`
(`src/Idrak.Nlp/FineTuning.cs:1179`) are token-shaped and internal to `Idrak.Nlp`: a step is `TrainingSequence`s packed
into rows of `BatchTokens`, the loss is the token cross-entropy over trained positions, the ladder measures bytes per
position, graphs record `(rows, positions)`. A denoising step is images, noise and timesteps with an MSE on the
prediction; nothing of the packing or the loss applies, and `Idrak.Diffusion` must not reference `Idrak.Nlp`. `Trainer.Fit`
(core) takes an `IBatchSource` and a loss over (prediction, target) but calls `Module.Forward(Tensor)` with one input,
not (latents, t, conditioning). So D6 writes `DiffusionTrainer` in `Idrak.Diffusion` and **reuses**: the optimizers and
`FineTuningOptimizers`' names (AdamW, Adam, AdamW8Bit, SGD; `HostOptimizer`), `Optimizer.SaveState` / `LoadState`, the
learning-rate schedulers (cosine, linear, constant, wsd: `FineTuningSchedules` moves to `Idrak.Abstraction.Training`
as `LearningRateSchedules` since a second package now uses it, decision 10), `ActivationMemory` scopes,
`ComputeGraph.Capture` of the forward-and-backward step (as `FineTuner` records it), `Pretrainer`'s checkpoint layout
and state-file conventions, the telemetry events, and the CLI's run folder, `RunLogWriter`, `resume` and `runs show`.
The out-of-memory ladder's rungs are copied in the same order (bfloat16 activations, checkpointed levels, micro-batch
halving); a shared `MemoryLadder` in `Idrak.Abstraction` is proposed to the owner (decision 7) rather than a third copy.

**Files.** `src/Idrak.Diffusion/Training/DiffusionTrainer.cs`, `DiffusionTrainingOptions.cs`, `EmaWeights.cs`,
`LatentCache.cs`, `Data/ImageDiffusionSource.cs` ("images": `ImageFolderSource.Open` or class folders, resized and
centre-cropped or letterboxed to the resolution by `ImagePreprocessor`, `RandomFlip` and any `Augmentations` pipeline on
worker threads through `AugmentedImageLoader`'s producer, pixels scaled to [-1, 1]), `Data/CaptionSource.cs`
("captions": JSON Lines `{"image": "...", "text": "..."}` beside the images or in a zip, read as plan 19's record sources
read theirs); `src/Idrak.Cli.Commands/Commands/Train/CreateCommand.cs` (a `"diffusion"` spec: `{"format": "idrak-create/1",
"architecture": "IdrakDiffusionPipeline", "diffusion": {...}}` → `DiffusionModels.Create`), `PretrainCommand.cs`
(dispatch on `model_index.json`: diffusion options below), `ResumeCommand`, `RunsShowCommand` (steps, loss, EMA loss).

```csharp
public sealed record DiffusionTrainingOptions
{
    public IReadOnlyList<DiffusionDataSpec> Sources { get; init; } = [];   // kind, paths, weight, images folder or zip
    public int Resolution { get; init; } = 64;            // the training image size (square; width and height separately allowed)
    public int Steps { get; init; } = 10000;
    public int Batch { get; init; } = 16;                 // images per optimizer step; micro-batches measured
    public int GradientAccumulation { get; init; } = 1;
    public float LearningRate { get; init; } = 1e-4f; public float MinLearningRate { get; init; }
    public float WarmupFraction { get; init; } = 0.02f; public string Schedule { get; init; } = "cosine";
    public string Optimizer { get; init; } = "adamw"; public float WeightDecay { get; init; } = 0.01f;
    public float Beta1 { get; init; } = 0.9f; public float Beta2 { get; init; } = 0.999f;
    public float MaxGradientNorm { get; init; } = 1f; public bool HostOptimizer { get; init; }
    public string Objective { get; init; } = "epsilon"; public float? SnrGamma { get; init; }   // min-SNR weighting, null off
    public string TimestepSampling { get; init; } = "uniform";   // or "logit-normal" (a registry entry later)
    public float EmaDecay { get; init; } = 0.9999f; public bool Ema { get; init; } = true; public bool EmaOnHost { get; init; }
    public float ConditioningDropout { get; init; } = 0.1f;      // the unconditional share for classifier-free guidance
    public IReadOnlyList<string> Trained { get; init; } = ["denoiser"];   // "denoiser", "codec", "conditioner"
    public bool? BFloat16Activations { get; init; } public bool? CheckpointLevels { get; init; }
    public bool CudaGraphs { get; init; } = true; public string LatentCache { get; init; } = "memory";   // memory | disk | none
    public int Seed { get; init; } public int SaveEvery { get; init; } public int KeepCheckpoints { get; init; } = 3;
    public int EvaluateEvery { get; init; } public IReadOnlyList<DiffusionDataSpec> Evaluation { get; init; } = [];
    public int SampleEvery { get; init; } public int SampleCount { get; init; } = 4; public int SampleSteps { get; init; } = 20;   // images written to the run folder with the EMA weights
    public int NonFiniteSteps { get; init; } = 8;
}
public static class DiffusionTrainer
{
    public static DiffusionTrainingResult Train(DiffusionModel model, DiffusionTrainingOptions options, string outputFolder, DiffusionTrainingCheckpoint? resume = null,
        IProgress<DiffusionTrainingProgress>? progress = null, CancellationToken cancellationToken = default, Action<string>? trace = null);
    public static DiffusionTrainingCheckpoint ReadCheckpoint(string folder); public static string? LatestCheckpoint(string outputFolder);
}
public sealed record DiffusionTrainingProgress(int Step, int Steps, float Loss, float LearningRate, double ImagesPerSecond, long DeviceBytes, TimeSpan Remaining, float? EvaluationLoss);
public sealed class EmaWeights(IEnumerable<Tensor> parameters, float decay, bool onHost)   // one Axpy per parameter per step; Swap() in and out for sampling; SaveState / LoadState
```

**The step.** Draw the micro-batch's images (latent-cached: the codec's `Encode` with the posterior mean once per image
when the codec is frozen, kept by `LatentCache`; else encoded per step), timesteps from `Random(seed, step)` on the
host, noise from `Tensor.Normal` with the same generator (uploaded), `x_t = sqrt(ā)·x0 + sqrt(1-ā)·noise` as one
`MulAdd` with the schedule's gathered rows, conditioning (class ids or captions; a `ConditioningDropout` share replaced
by the unconditional input), the denoiser's `ForwardTraining`, the objective's `Target` and `Weight`, MSE per image
weighted and averaged, backward, clip, the optimizer's step, the EMA update, `BatchCompleted`. Graphs record the
forward and backward (as `FineTuner` does) when no part of the codec or conditioner trains.

**Checkpoints.** `outputFolder/checkpoint-STEP/`: the family's `Save` (a model folder the generator loads), plus
`ema.safetensors`, `optimizer.safetensors`, `diffusion-state.json` (`"format": "idrak-diffusion-state/1"`: step, sources'
positions, RNG state, options, the ladder's settled settings, loss history); `last/`, `best/` on the evaluation loss;
`samples/step-N-i.png` when `SampleEvery` is set. Resume reloads all of it; the next losses equal the uninterrupted
run's bit for bit on the CPU.

**Command line.** `idrak create init --diffusion -o spec.json` writes a commented spec (pixel UNet at the resolution,
class conditioning, DDPM schedule); `idrak create spec.json -o FOLDER -P plugin.dll`; `idrak pretrain FOLDER --data DIR
[--captions FILE] --resolution N --steps N --batch N --lr X --objective NAME --snr-gamma X --ema-decay X --no-ema
--cond-dropout F --trained denoiser,codec --latent-cache memory|disk|none --sample-every N --eval DIR -o RUN -d DEVICE -P
plugin.dll` (every option also in `pretrain.json` under `"diffusion"`, `"format": "idrak-pretrain/1"`); `idrak resume
RUN`; `runs show RUN` plots the loss and the EMA evaluation loss over steps.

**Tests** (`"diffusion training: ..."`, `"cli diffuse: ..."` for the commands): step 1 on `pixel-ddpm` (the 24 images,
SGD 0.2, the fixture's noise and timesteps fed through a test seam) gives the loss within 1e-4 and every gradient within
1e-4; three steps' losses and weights; the EMA after three steps equals the fixture's; `v` and `sample` objectives'
losses; min-SNR weights against the formula; conditioning dropout replaces the stated share (counted over 100 steps); a
run stopped after step 2 and resumed from `checkpoint-2` gives steps 3 to 6 bit for bit (EMA and optimizer state
included); the latent cache gives the same loss as encoding per step; `Telemetry` receives one `BatchCompleted` per
step; the loss on the 24 images falls by half in 300 CPU steps (a smoke test); `create` from a spec, `pretrain` 4 steps
writing `run.json`, `log.jsonl`, `checkpoint-4` and two sample PNGs, `resume` 2 steps, `runs show`, `diffuse` from
`last/`; an unregistered family or objective refused with `-P`.

**Done when** the fixture matches on the CPU and the CLI filters pass.

**Risks.** `FineTuningSchedules` moving to Abstraction touches `Idrak.Nlp`'s public API (the dump regenerated; the Nlp
names kept as thin forwarders is against decision 7, so they move: one commit, both dumps). The EMA doubles the
parameter memory; `EmaOnHost` is the setting for a device with no room (the trace says so).

## Phase D7: a loader family plug-in for an existing checkpoint format

**Which.** Stable Diffusion 1.x and 2.x in the **diffusers folder layout** (`model_index.json` naming
`StableDiffusionPipeline`; `unet/config.json` + `diffusion_pytorch_model.safetensors`, `vae/`, `text_encoder/`
(`CLIPTextModel`), `tokenizer/` (`vocab.json`, `merges.txt`, or `tokenizer.json`), `scheduler/scheduler_config.json`).
Chosen because it is the layout the fixture is made in, the one most checkpoints ship in, and one folder gives every
component with its config. The single-file `.safetensors` layout (`model.diffusion_model.*`, the original names) is a
name map on top of the same family, a follow-up; SDXL (two text encoders, pooled and time-id conditioning) and
inpainting variants are later families of the same plug-in. The sample stays a sample: the library registers none of it.

**Files.** `samples/StableDiffusion/Idrak.StableDiffusion/` (a Library; `PackageReference` Idrak, Idrak.Imaging,
Idrak.Diffusion; in `Idrak.slnx` under `/samples/` and referenced by `tests/Idrak.Tests` as the three vision plug-ins
are): `StableDiffusionPlugin.cs` (`Register()` → `DiffusionFamilies.Register(new StableDiffusionFamily())`,
`RegisterIdrakPlugin()`, `Unregister()`), `StableDiffusionFamily.cs` (`Read`: `UNetSpec` from `unet/config.json`
(`block_out_channels`, `down_block_types` with or without `CrossAttn`, `layers_per_block`, `attention_head_dim`,
`cross_attention_dim`, `norm_num_groups`, `use_linear_projection`, `transformer_layers_per_block`), the tensors read one
at a time into the blocks through `StoredWeights` and a name map (`down_blocks.i.resnets.j.*`, `attentions.j.transformer_blocks.k.attn1/attn2/ff/norm*`,
`proj_in/out`, `time_embedding.linear_1/2`, `conv_in/out`, `conv_norm_out`); `AutoencoderSpec` from `vae/config.json`
and its map; the CLIP text conditioner; the schedule from `scheduler_config.json` (`beta_schedule`, `beta_start`,
`beta_end`, `num_train_timesteps`, `prediction_type`, `timestep_spacing`, `rescale_betas_zero_snr`, `steps_offset`);
`Save` writes the same layout back), `ClipTextConditioner.cs` (`Embedding` + learned positions +
`TransformerEncoderLayer(causal: true)` with quick-GELU + final `LayerNorm`; `ClipSkip` setting (SD 2.x's penultimate
layer); the pooled output at the end-of-text token; `MaxLength` 77 from the config, prompts truncated with a note),
`ClipTokenizer.cs` (reads `tokenizer.json` through `BpeTokenizer` when its normalizer and `end_of_word_suffix` are
supported, else `vocab.json` + `merges.txt` with the plug-in's own CLIP BPE: lowercase, NFC, the CLIP regex, `</w>`
merges, start and end tokens, padding to 77), `README.md`.

**Library changes the phase may need** (each a general item, none family-specific): `StoredWeights.GroupNorm` and
`StoredWeights.LayerNorm` (core, beside `BatchNorm`); `BpeTokenizer` support for `end_of_word_suffix` and the
`Lowercase` / `NFC` normalizers if missing (core; a `TokenizerComponents` registration from the plug-in is the fallback);
the hub download of a diffusers folder (`Shared/Hub.cs`: every file of the listed subfolders, `.safetensors` preferred
over `.bin`, `fp16` variants by `--variant`); `SafeTensorType` F16 is already read.

**Tests** (`"stable diffusion: ..."`): `tiny-pipeline` loads through the family (every component's shapes); the UNet's
output on `reference/` to 1e-5; the VAE's posterior and decode; the CLIP ids of the ten prompts exact and the hidden
states to 1e-5 (last and penultimate); the five-step generation for each recorded sampler to 1e-4 in latent space and
1e-3 in pixels; `Save` then `Load` reproduces the tensors; `idrak diffuse -P Idrak.StableDiffusion.dll tiny-pipeline "..."
-o out.png` runs; without the plug-in the architecture is "not registered" with the `-P` hint; `Origin` is the sample's
assembly. The real checkpoint is the owner's run (D8).

**Done when** the tiny pipeline matches diffusers through the plug-in on the CPU.

**Risks.** Real checkpoints carry `attention_head_dim` as a list in SD 2.x and `use_linear_projection true`; both are
spec fields. `fp16` safetensors load as float32 (`AsStored` keeps `Linear` bfloat16 only; F16 is widened: a bfloat16
or float16 convolution weight is D3's left item). Quality parity with diffusers under bfloat16 products is a tolerance
the test states, not bit equality.

## Phase D8: the owner's proof run

From `D:\Projects\Idrak`, after D6 (from scratch) and D7 (the loader):

```powershell
cd D:\Projects\Idrak
git pull
dotnet build -c Release tests\Idrak.Tests
$env:IDRAK_DEVICES="cuda"; $env:IDRAK_FILTER="diffusion|cli diffuse|stable diffusion"; dotnet run -c Release --no-build --project tests\Idrak.Tests
$env:IDRAK_DEVICES="vulkan"; $env:IDRAK_FILTER="diffusion|cli diffuse|stable diffusion"; dotnet run -c Release --no-build --project tests\Idrak.Tests
$env:IDRAK_DEVICES="cuda"; dotnet run -c Release --no-build --project tests\Idrak.Tests -- --bench-diffusion
```

A model of your own, from scratch, on your images (`work\images` is a folder of images, or class folders; sizes and
steps in `spec.json` and on the command line are yours; nothing in the library prefers one):

```powershell
dotnet build -c Release samples\IdrakDiffusion\Idrak.IdrakDiffusion
$plugin = "samples\IdrakDiffusion\Idrak.IdrakDiffusion\bin\Release\net10.0\Idrak.IdrakDiffusion.dll"
dotnet run -c Release --project src\Idrak.Cli -- create init --diffusion -o work\diffusion-spec.json
dotnet run -c Release --project src\Idrak.Cli -- create work\diffusion-spec.json -o work\ddpm -P $plugin
dotnet run -c Release --project src\Idrak.Cli -- pretrain work\ddpm --data work\images --resolution 64 --steps 20000 --batch 32 --lr 1e-4 --sample-every 1000 -o work\ddpm-run -d cuda:0 -P $plugin
dotnet run -c Release --project src\Idrak.Cli -- resume work\ddpm-run
dotnet run -c Release --project src\Idrak.Cli -- runs show work\ddpm-run
$last = (Get-ChildItem work\ddpm-run -Directory -Filter checkpoint-* | Sort-Object { [int]($_.Name -replace 'checkpoint-', '') } | Select-Object -Last 1).FullName
dotnet run -c Release --project src\Idrak.Cli -- diffuse $last --class 0 --count 8 --steps 50 --sampler ddim -o work\samples -d cuda:0 -P $plugin -v
```

An existing checkpoint (a Stable Diffusion 1.x or 2.x folder in the diffusers layout, downloaded to `work\sd` by you
or by `idrak pull` once D7 teaches it the layout; the id is yours to choose):

```powershell
dotnet build -c Release samples\StableDiffusion\Idrak.StableDiffusion
$sd = "samples\StableDiffusion\Idrak.StableDiffusion\bin\Release\net10.0\Idrak.StableDiffusion.dll"
dotnet run -c Release --project src\Idrak.Cli -- diffuse work\sd "a photograph of a lighthouse at dusk" --negative "blurry" --steps 30 --guidance 7.5 --sampler dpm++2m --seed 1 -o work\sd-out -d cuda:0 -P $sd -v
dotnet run -c Release --project src\Idrak.Cli -- diffuse work\sd "a photograph of a lighthouse at dusk" --steps 30 --precision bf16 --seed 1 -o work\sd-bf16 -d cuda:0 -P $sd -j
dotnet run -c Release --project src\Idrak.Cli -- diffuse work\sd "a photograph of a lighthouse at dusk" --steps 30 --seed 1 -o work\sd-profile -d cuda:0 -P $sd --profile
dotnet run -c Release --project src\Idrak.Cli -- diffuse work\sd "a watercolour of the same lighthouse" --init work\sd-out\a-photograph-1.png --strength 0.6 --seed 2 -o work\sd-img2img -d cuda:0 -P $sd
```

What to read from them: `-v` prints the measured batch, the attention path per level, whether the pass was recorded,
the decode tiles and the peak device bytes; `-j` the seconds per stage; `--profile` the kernel breakdown (expect the
implicit or composed convolution products, `attention_spans_*` at the levels with attention, the fused group norm, no
host fallback). The bf16 run is a comparison of time and peak against float32 on the same seed; the images differ
within the tolerance the test states.

## Order and parallelism

D0 → D1 → D2 and D3 (two agents; D2 needs nothing of D3) → D4 → D5 → D6 → D7 (D7 may start after D4 in parallel with
D5 and D6: it needs D0's tiny pipeline and the blocks) → D8 (the owner). D3b (DiT) after D6. Each phase is one commit
series on `abstraction` with its "as built" section appended to this plan (what was built, filter names and counts, what
is left, the owner's GPU commands), as plans 13 and 19 did.

## What a from-scratch run is, honestly

- **The pipeline is the deliverable of D0 to D7.** It is proven at tiny scale on the CPU against diffusers, as every
  plan's reference is. Whether a model of useful quality comes out is the owner's data and compute: a pixel-space
  class-conditional model at a small resolution on a few thousand images is a run of hours on one GPU; a latent
  text-to-image model that competes with a loaded checkpoint needs far more images, captions and steps than that, and a
  run is one device (plan 6).
- **The hedges.** Pixel space first (no VAE to train); then latent space with a loaded VAE as a frozen component (as
  plan 19's `--decoder` start reused a trained decoder: the SD plug-in's VAE through `LatentCodecs`, the denoiser from
  scratch); a VAE from scratch last, since without a perceptual loss its reconstructions stay soft. Class conditioning
  first; text conditioning from the plan 19 text model's hidden states (`Conditioners` entry "decoder-hidden" in the
  IdrakDiffusion plug-in, the model folder a setting) second; CLIP through the SD plug-in whenever a loaded checkpoint
  is the base.
- **Fine-tuning a loaded checkpoint** on the owner's images is the same trainer with `Trained = ["denoiser"]` (or a
  LoRA on its `Linear` layers through the existing `ILinearAdapter`, a follow-up) and a lower rate: the quality lever
  that costs least, in the plan though not a phase of its own.

