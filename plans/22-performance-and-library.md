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
| exp | 2.8 | `Vector.Exp`, a runtime method the JIT does not inline and compiles unoptimized at first (step 8: the library's own, 16 GB/s) |
| gelu (erf) | 1.1 | scalar erf and exp (step 8: vector loops, 9.6 GB/s) |
| sum over rows (inner = 1) | 2.0 | a span per element in the short-run branch |
| softmax, layer norm | 1.7-1.9 | scalar mean/variance/exp loops (steps 2 and 8: 7.5 and 11 GB/s) |

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
- Mask multiplies after every masked layer (`mul` 355 a step at four clips): measured on the MMS-sized model (step 9),
  the multiplies and their backward take 156 ms of a 17.9 s step on the CPU, under one percent, while the convolutions take
  over ninety; no fused kernel pays for its code on three backends, so they stay as they are.
- The CPU's exp, erf, softmax, layer norm and axis sums (section 1): vectorized loops, not fusions, but the same gain.

## 4b. Lookups (the owner's note: indices and hash tables for quick access)

Where a lookup walks a list or compares names in a loop, give it an index: candidates are `TensorScope`'s backward scan
of its list on every `Keep`, `Owns` and `Untrack` (long scopes in a training step), name lookups in the weight readers and
tensor stores, and the kernel and registry lookups per launch; each is measured with the benches before and after.

## 5. Order of work and status

| # | Step | Status |
|---|---|---|
| 1 | Benchmark and stress project, baseline rows | done (836a3141) |
| 2 | CPU kernels the bench exposed: axis sums with inner 1 (521 → 31 µs over 256 × 1,024), layer norm (1,200 → 181 µs forward); exp and erf left (the built-in vector exp is latency-bound, a modest win) | done (836a3141) |
| 3 | `ComputeResources`: environment variables for every limit; `WorkspaceShare` behind the three fixed halves; Vulkan's memory reserve; `MaxTokens` cap removed; `MaxCpuThreads` honoured everywhere | done (37e5faeb) |
| 4 | Speech training: `BatchSeconds` 0 (the default) fills the device from a probe of the longest utterance; checkpointing reaches the VITS step (MMS size, four clips: peak 3,486 → 2,491 MiB, the step about a fifth longer on the CPU) | done (37e5faeb) |
| 5 | WaveNet gate as one operation (`FeedForwardActivation.TanhSigmoid`; CPU, CUDA, Vulkan, conformance kit): 7,124 → 6,932 launches a step, peak 2,491 → 2,460 MiB | done (d2b36cf0) |
| 6a | Library moves, batch 1: `Tensor.Pad` (zeros, reflect, replicate) and `Tensor.Flip`, `TensorDictionary`, `Conv1d` and `ConvTranspose1d` with the `WeightNorm` parametrization, `PaddedSequences`, `StoredWeights.FoldWeightNorm` and `WeightNormNames`; the VITS plug-in and OpenVoice use them | done |
| 6b | Library moves, batch 2 (section 3, items 2, 4 and 5): `StoredLayers`, `StoredParameter` and `ParameterSet` (Idrak.Models), `Random.NextGaussian`, `DilatedDepthSeparableConv`, `RationalQuadraticSpline`, `ConvolutionalSplineFlow`, `RelativePositionAttention`, `Checkpointing.RunBlock` (Idrak.Layers); `WaveNet`, `ResidualCouplingBlock`, `PosteriorEncoder`, `HifiGanGenerator`, `HifiGanDiscriminators` (Idrak.Audio.Layers), `AdversarialLosses`, `MonotonicAlignment`, `DifferentiableLogMel` (Idrak.Audio). The VITS plug-in keeps its tokenizer, text encoder, duration predictor's flow order and training step (five files gone); OpenVoice drops its VitsSpeech reference and builds the library layers from its own config. The torch fixture's step-1 losses and gradients, the three SGD steps and the conversion fixture match as before (within 1e-4); 16 layer tests, three new | done |
| 6c | Library moves, batch 3 (section 3, item 6, vision): `VisionTransformerEncoder`, `VisionTransformerConfig`, `VisionTransformerLayerWeights` and `PatchGrid.AveragePool` in Idrak.Vision; the Gemma 3 plug-in's SigLIP tower and IdrakVision's tower (two copies of 360 and 500 lines, with their export, import and pooling) are gone, both plug-ins build the library encoder from their own configs; transformers' and PyTorch's references still match (SigLIP encoder, vlm scratch, pretrain, distill and tuning groups) | done |
| 6d | Library moves, batch 4 (section 3, item 6, text and tools): the speech text front end into Idrak.Audio (`ITextRomanizer`, `TextRomanizers` with the uroman Arabic table, `ITextPhonemizer`, `TextPhonemizers` with the espeak-ng runner; the VITS plug-in keeps only its tokenizer), `Subtitles` (SubRip, WebVTT) in Idrak.Audio, `NpyFile` in Idrak.Abstraction.Formats and `SyncProgress<T>` in Idrak.Abstraction out of the CLI | done |
| 6e | `HifiGanVocoder` (Idrak.Audio.Layers): HiFi-GAN as an `IVocoder` read from a checkpoint folder under transformers' root-level names (SpeechT5's and FastSpeech 2's vocoders), its config from their `HifiGanConfig` keys, the log-mel standardized with the checkpoint's mean and scale as one affine, long spectrograms in pieces with context, `RegisterFolder` for the `Vocoders` registry; `StoredLayers.Join` lets the `FromStored` factories read root-level names | done |
| 6f | The CLI's Bidi and Arabic shaping unified with Imaging's: one UAX #9 implementation (the CLI's, complete with explicit embeddings, isolates, the paired-brackets table and the -1 removal levels) as the public `Idrak.Imaging.Shaping.Bidi`, with `Resolve` and `Reorder` for text layout built on its `Levels`; `ArabicPresentationForms` (the CLI's shaping, a zero width non-joiner now breaking the join as the OpenType shaper already read it) shared by the OpenType Arabic shaper's joining; the CLI's `Bidi.cs`, `ArabicShaping.cs` and `UnicodeTables.cs` (1,147 lines) and tools/unicode/make_tables.py gone. One table set, UCD 18.0.0, made by tools/unicode/make_text_tables.py from the UCD files (no Python package; the old unicodedataplus route stopped at Unicode 16 and gave 218 of 428 mirror pairs, L for unassigned code points the UCD defaults to BN or ET, and joining types from a newer Unicode than it claimed). Checked against BidiTest.txt and BidiCharacterTest.txt of UCD 18 | done |
| 6g | `ConvolutionalFeedForward` (two convolutions over time with the activation by name, even kernels padded as transformers does), `ElementwiseAffineFlow` (forward with its log-determinant, the host inverse) and `Tensor.Indices` in the library; the VITS plug-in's text encoder and duration predictor use them, its own helpers are down to a column repeat | done |
| 6h | Item 7: MP3 into Idrak.Audio or a companion `Idrak.Audio.Mp3` package (size, licence): the owner's call | waiting on the owner |
| 8 | CPU math kernels: the library's own vector exp (`CpuMath.Exp`, 256- and 512-bit; the runtime's `Vector.Exp` is not inlined and runs unoptimized for its first hundreds of calls), every range kernel compiled optimized at its first call (`AggressiveOptimization`), erf-GELU, SiLU, sqrt, sin, cos and sign as vector loops forward and backward, the scale-mask softmax's pass and the softmax backward row vectorized; the bench warms up 30 calls before timing (`IDRAK_BENCH_WARMUP`). Warm µs on the 4-thread CPU, 16k / 262k elements: exp 72 → 14 / 672 → 131, erf-GELU 577 → 34 / 1,587 → 219, softmax 103 → 18 / 1,108 → 281, exp forward and backward 1,349 → 610 | done |
| 9 | Mask folding: measured and dropped (section 4). The speech bench on the MMS-sized model, four clips a step: `mul` 375 launches, 105 ms; `bwd mul` 316, 51 ms; `conv2d` 545, 9,448 ms; `bwd conv2d` 352, 11,628 ms; the step 17.9 s warm | dropped |
| 7 | Measure every step with the benches; record the numbers here | ongoing |
