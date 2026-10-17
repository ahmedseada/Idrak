# Plan 22: speed and memory everywhere, and the library owns what is generic

The owner's mandate (October 2026): find the memory the library wastes and free it; use the room, and every core and
device, for faster fine-tuning, training and inference, with no default resource limit (only a limit the user sets
bounds the library); fuse operation chains; move everything generic out of the plug-ins and the CLI into the library or
a companion package; and give the repository a benchmark and stress-test project so bottlenecks and gains are measured,
not guessed. This plan records what the survey found, the order of the work, and what each step changed. Each step keeps
the rules of `docs/optimization.md` (rules 70-87) and the fix workflow of `CLAUDE.md`.

## 1. The benchmark project (`benches/Idrak.Benches`)

Public API only (what an app sees; a measurement it cannot take is a telemetry gap to fill, rule 78), every device
`IDRAK_DEVICES` names, cold and warm apart (rule 76), launches counted through the operations telemetry, peak memory
read from `ComputeResources`, rows written as JSON and compared run to run (`-- compare before.json after.json`).
Sections: `ops`, `matmul`, `train-decoder`, `infer-decoder`, `train-speech`, `memory` (stress). See `benches/README.md`.
The kernel-level benches that reach into the backends' internals stay in the test runner (`--bench-gemm`, ...), since
`Idrak.Abstraction` grants its internals to the tests only (the inventory test enforces it).

Baseline on the 4-thread CPU container (small size), the slow spots the first run exposed:

| Operation (262,144 elements) | GB/s | Why |
|---|---|---|
| add, mul, leaky relu | 23-31 | SIMD loops: the ceiling here |
| tanh, gelu (tanh form) | 13-15 | vectorized rational |
| exp | 2.8 | a scalar `MathF.Exp` loop |
| gelu (erf) | 1.1 | scalar erf and exp |
| sum over rows (inner = 1) | 2.0 | a span per element in the short-run branch |
| softmax, layer norm | 1.7-1.9 | scalar mean/variance/exp loops |

## 2. Resource caps found (src/, by the survey)

Measured or settable already: `MaxCpuThreads` (all cores), the memory limits (null = unlimited), the GPU reserve (a
sixteenth, `GpuMemoryReserve`), the CPU cut-overs (`IDRAK_CPU_*`), Vulkan's pages and staging, the out-of-memory
halving of speech micro-batches. Fixed with no override, to make settings or measurements:

- `ComputeResources`: no environment variables for `MaxCpuThreads`, `CpuMemoryLimit`, `GpuMemoryLimit`,
  `GpuMemoryReserve`, `OffloadReturnHeadroom` (the CLI's `--threads` is the only path).
- Three "half of the free memory" fractions: the composed attention's score share (`Tensor.AttentionPaths.cs`,
  `ScoreShare = 0.5`), the unfolded convolution patches (`Backend.Convolution.cs`, `free / 2`), the Vulkan composed
  convolution (`VulkanBackend.Convolution.cs`); the CLI's image batch (`ImageModelSupport.cs`, `free / 2`).
- Vulkan applies no memory reserve at all (`VulkanBackend.Spans.cs`, `AvailableMemory`), unlike CUDA and HIP.
- `TextGenerator.MaxTokens = 4096`: generation stops at 4,096 tokens whatever `NumPredict` asks.
- `BpeTrainer`, `TokenFileWriter`, `SyntheticPages` use `Environment.ProcessorCount` and ignore `MaxCpuThreads`.
- Speech training: `BatchSeconds` 160 is a fixed starting point, halved after running out of memory; nothing grows it to
  what the device has.
- Fixed policy constants left as they are (documented, deterministic or small): the pinned staging ring (16 MiB), the host
  optimizer's transfer slots, the pool's best-fit slack (25%), chunk thresholds, the BPE cache (100k), prefetch depth
  (one step ahead), CPU attention tiles, `SumBlocks`, `ReductionElements`.

## 3. Generic code in the plug-ins and the CLI (the survey's list, by dependency)

The strongest signal: `Idrak.OpenVoice` references `Idrak.VitsSpeech` and reads its internals (`VitsWeights`,
`FoldWeightNorm`, `VitsFlow`, `VitsPosteriorEncoder`, `HifiGanDecoder`). One plug-in depending on another's internals
means that code is generic.

1. Tensor operations: `Pad` (zeros, reflect, edge, differentiable), `Flip`, index tensors; Gaussian draws and initializers.
2. Layers: `Conv1d`, `ConvTranspose1d` (today `Conv2d` with a (1, k) kernel in four plug-ins), a `WeightNorm`
   parametrization (`Tensor.WeightNorm` exists; the layer does not), `StoredWeights` additions (`LayerNorm`, vectors,
   `Conv1d`, `ConvTranspose`, weight-norm names `weight_g/weight_v` and `parametrizations.*`), a `StoredParameter` and
   `ParameterSet`, an in-memory `ITensorStore`, renamed and folding tensor-store wrappers, packed q/k/v helpers.
3. `PaddedSequences` (the VITS mask: per-channel masks, row masks, key ranges), also duplicated in IdrakAudio.
4. Audio layers: dilated depth-separable convolution, WaveNet with global conditioning, residual coupling flows, the
   rational-quadratic spline flow, relative-position attention, the posterior encoder.
5. HiFi-GAN generator (as an `IVocoder` "hifigan"), the multi-period and multi-scale discriminators, the least-squares
   GAN and feature-matching losses, the differentiable batched log-mel. After this OpenVoice drops its VitsSpeech reference.
6. In parallel: the SigLIP/ViT encoder duplicated in Gemma3Vision and IdrakVision (→ `Idrak.Vision`), patch-grid
   pooling, the image-blocks attention rule; the romanizer and phonemizer contracts (→ a text front end package);
   the CLI's Bidi/Arabic shaping (duplicated in Imaging), Npy, subtitle writers, input expansion, `SyncProgress`.
7. The MP3 codec and encoder are a file format behind the codec registries, not a model family: a candidate for
   `Idrak.Audio` or a companion `Idrak.Audio.Mp3`; the owner decides (size, licence).

## 4. Fusion candidates (the census of a VITS step)

- WaveNet's gate `tanh(a) · sigmoid(b)` with its conditioning add: 3 launches forward, 4 back, per layer (32 layers a
  step): one gated operation.
- Mask multiplies after every masked layer (`mul` 355 a step at four clips): fold into the producing kernels where a
  kernel exists (layer norm, affine), else keep.
- The CPU's exp, erf, softmax, layer norm and axis sums (section 1): vectorized loops, not fusions, but the same gain.

## 4b. Lookups (the owner's note: indices and hash tables for quick access)

Where a lookup walks a list or compares names in a loop, give it an index: candidates are `TensorScope`'s backward scan
of its list on every `Keep`, `Owns` and `Untrack` (long scopes in a training step), name lookups in the weight readers and
tensor stores, and the kernel and registry lookups per launch; each is measured with the benches before and after.

## 5. Order of work and status

| # | Step | Status |
|---|---|---|
| 1 | Benchmark and stress project, baseline rows | done |
| 2 | CPU kernels the bench exposed: exp, erf and GELU(erf), axis sums with inner 1, softmax and layer norm vectorized | |
| 3 | `ComputeResources`: environment variables for every limit; one `WorkspaceShare` setting behind the three fixed halves; Vulkan's memory reserve; `MaxTokens` cap removed; `MaxCpuThreads` honoured everywhere | |
| 4 | Speech training: `BatchSeconds` automatic (fills the device from a measured probe), checkpointing honoured by the VITS step through a public checkpoint API | |
| 5 | WaveNet gate as one operation (CPU, CUDA, Vulkan, conformance kit) | |
| 6 | Library moves in dependency order (section 3, items 1-5), each with its tests and the API dumps | |
| 7 | Measure every step with the benches; record the numbers here | |
