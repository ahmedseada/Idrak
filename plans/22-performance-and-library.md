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
Sections: `ops`, `matmul`, `train-decoder`, `infer-decoder`, `train-speech`, `memory` (stress); the settings are in the
header of `benches/Idrak.Benches/Program.cs`.
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

First GPU run (the owner's CUDA card, 16 GiB and 70 SMs, medium size, 2026-10-17; the device is the measurement's context,
nothing is tuned to it):

| Section | Measured | What it says |
|---|---|---|
| ops, 8M elements | exp, tanh, gelu, leaky relu, add, mul 780-820 GB/s; softmax and layer norm 710-730 GB/s | at the card's bandwidth: memory-bound, as they should be |
| ops, `sum over rows [rows, 1024]` | 40 GB/s at 1M elements; 105 µs at 1M and at 8M alike | `sum_axis_f32` ran one thread per output row, each walking its row serially with its neighbours a row apart. Fixed: `sum_axis_rows_f32`, a warp a row reading coalesced and summing in a fixed butterfly order (inner 1, rows of a warp's lanes or more). After: 7.0 µs at 65k, 9.2 µs at 1M (455 GB/s), 15.1 µs at 8M (the 32 MiB input stays in the card's cache between calls, so the 2.2 TB/s read there is the cache's); the backward step 137 → 82 µs at 1M |
| ops, small sizes | tanh, gelu, leaky relu 30-35 µs at 65k and at 1M where exp and add took 8-13; mul 35 µs at 65k and 12 at 1M | not the kernels (their PTX is as plain as exp's): the driver's submission latency on one call a round, charged to a 10 µs kernel. The ops section now queues `IDRAK_BENCH_CALLS` calls a round (default 8) with one sync; the second run read 6.0-6.6 µs for all of them at 65k and 7-9 µs at 1M (1.0-1.4 TB/s, cache-fed), with the 8M rows unchanged at 830-870 GB/s |
| matmul | 1024³ 14.3 TFLOPS, 4096 × 1024 × 4096 17.7; GEMV 1 × 2048 × 8192 0.088 ms (760 GB/s, bandwidth-bound); conv 3×3 64→64 9.1 TFLOPS, 7×7 s2 8.8, 1-D k11 9.5; the first product cold 67 ms | float32 products at a fair share of the card's rate; the decoder's bfloat16 paths run on the tensor cores |
| train-decoder, large | out of memory: 1.64 B parameters and a batch of 8 × 1024 do not fit a 16 GiB device | the bench crashed. Now it halves the batch until the device holds a step and names the batch in the row, or skips with the reason when the model itself does not fit; every section skips on `ResourceLimitExceededException`. The second run halved 8 → 4 → 2 → 1 and skipped: the float32 weights and the optimizer's state alone pass the device's memory, so the large training row needs a larger device or `IDRAK_OFFLOAD=1`; medium is the row for a 16 GiB card |
| train-decoder, medium (254 M parameters, batch 8 × 512) | 490.6 ms a step warm (1,727 cold), 761 launches, 4,357 MiB peak above the weights, 8,349 tok/s; with the blocks checkpointed 615.7 ms, 943 launches, 800 MiB peak, 6,653 tok/s | 6 × 254 M × 4,096 tokens = 6.2 TFLOP a step at 12.7 TFLOPS: the float32 products' rate (the matmul row gave 14-18), so the step is product-bound and the float32 path is the ceiling here. Checkpointing trades a quarter of the step for a fifth of the activation memory. The bench now adds a bfloat16-products row (the tensor cores), to be run. `bwd reshape` 241 a step, "5,008 MiB out": a view's elements counted as output, nothing written (the reshape backward adopts the gradient) |
| infer-decoder, large (1.64 B parameters, float32) | 96.9 tok/s with a float32 cache, 93.1 int8, 97.9 bfloat16; 291 launches a token; the prompt of 256 in 74 ms (505 ms the first, cold); 6.3 GiB in use | token by token the step reads the whole model, 6.5 GB a token at 97 tok/s = 630 GB/s: bandwidth-bound, as decoding is; the cache's format does not matter at this size |
| train-speech, MMS-sized VITS, the owner's clips (12.1 s average) | one clip a step 3.45 s warm, 20.1 cold, 6,465 launches, 1,794 MiB; four clips 3.07 s warm, 25.5 cold, 3,086 MiB; eight clips 28.3 s "warm", 4.2 cold, 5,151 MiB | the eight-clip row was the wrong way round because the bench ran three steps and a step's batch is cut by count × longest clip: with clips of 6 to 26 s each batch has another count and padded length, so every step landed in a new tuning size class (counts up to 32 are their own class) and re-tuned every convolution shape (`bwd conv2d` 22 s of host time a step; once per class, kept in the tuning cache file). The bench now runs one cold step then `IDRAK_BENCH_CALLS` timed ones, lists each step's time with the tuning in it (a new `TuningCompleted` telemetry event at `TelemetryLevel.Devices`), takes the fastest timed step with the audio it trained (`SpeechTrainingProgress.BatchAudioSeconds`) for the row and its real-time factor, prints the trainer's notes (a micro-batch halved after running out of memory) and the top operations by host time. The rerun with nine steps a row showed it was no three-step artefact: 819, 1,849 and 2,102 tunings a row (47, 150 and 268 s), and the tuned steps kept recurring (one clip: 20.9, 4.0, 3.7, 0.7, 3.8, 0.7 ... s, 4-5 s of tuning in every other step), since counts up to 32 are each their own class and widths a quarter octave apart are too. Fixed in the library: a convolution at a size class none measured takes the choice of the nearest measured class of its shape within `ComputeResources.TuningNeighbourOctaves` (`IDRAK_TUNE_NEIGHBOUR_OCTAVES`, 1 by default, 0 measures every class; CUDA and Vulkan), so a run with varied lengths measures each convolution a handful of times instead of once per batch shape. Also found: the bench timed the host without draining the device, so a step's queued backward leaked into the next (0.17 s for 34 s of audio); it synchronizes before timing now. The fastest untuned steps of that run: one clip 0.66 s for 15 s of audio, four clips 1.78 s for 36 s (20 × real time). The third run, with the lending: tuning 47 → 21 s at one clip and 150 → 32 s at four, the four-clip step 1.78 → 0.72 s for 34 s of audio (48 × real time), one clip 0.66 s as before. What it exposes next: the step is about 0.7 s whatever the batch, 6,500 launches at about 100 µs each on the host, so the card idles most of a step; and the steps after the first still carry 0.5-0.8 s of tuning each (products and other data-sized shapes are not lent). The bench now keeps a census of the first step apart from the warm ones (host µs a call by operation without the cold tuning in them) and lists the tunings by what was tuned; the fourth run said: not launches. The warm steps' host time is tuning still: `FloatTile` (the float32 product tiles, keyed by the rows class, and every clip's text length is another class) 146-335 tunings in the eight steps after the first (5.7-11.8 s), and at eight clips 191 convolution tunings (47 s) for batches more than an octave from the measured ones; the steps without any ran 34 s of audio in 0.18 s (190 × real time). The lending now covers every key with data-sized fields: convolutions, float32 and tensor-core products, and attention (rows and key rows), one `DataSized` map per key kind. The fifth run: convolutions lent (18 measured in the eight steps after the first at four clips, 191 at eight clips for batches more than an octave from the first), the product tiles not (112-209 still): the text encoder's attention runs as plain products, so its scores and context products carry the text length in n or k, which the product key holds exactly, and no neighbour matched. The lending now compares every data-sized field of a key (rows × batch, n, k, the leading dimension) and takes the farthest in octaves, so a weight product (n and k the model's) lends as before and an attention product within the reach lends too; the sixth run, with that: tunings in the eight steps after the first 5 at one clip and 32 at four (from 112 and 196), the step 0.62 → 0.14 s at one clip (7.7 s of audio, 57 × real time), 0.73 → 0.19 s at four clips (31 s of audio, 162 ×), 0.17 s at eight (34 s, 198 ×); the first step of each row 14-19 s, the measuring of about 210 convolution shapes once. The step is now about 0.15 s whatever the batch: 6,500 launches at about 22 µs of host time each (143 ms in calls at one clip), so a batch that fills the device is free and a small one is host-bound; the launch cost is the next thing to measure. The eight-clip row still measures 191 convolutions after its first step because the bench's first batch there is a small remainder and the full batches lie more than an octave from it; `IDRAK_TUNE_NEIGHBOUR_OCTAVES=2` covers that, the default stays 1 |
| memory, varied shapes | 20,000 allocations, 159,709 MiB asked, 206 MiB peak in use, 662 MiB cached after, 15.2 µs an allocation | the pool keeps about three times the peak as cached blocks of the sizes it saw; reclaimable (`ReleaseCachedMemory`, and released before a failed allocation retries), so not waste, but the first thing to measure if a step ever runs out |
| memory, conv net step | batch 8: 2.1 ms, 3,782 images/s; batch 256: 41.9 ms, 2,112 MiB, 6,104 images/s; 256 (the largest tried) fit | throughput rises with the batch to the end of the sweep |
| memory, causal attention 8 heads × 64 | 4,096 steps 2.3 ms, 8,192 8.2 ms, 16,384 30.3 ms; peak 8, 16, 32 MiB | time quadratic, memory linear: the tiled kernel keeps no score matrix |

The Vulkan axis sum over the last axis (inner 1) now runs as a row kernel (`sum_axis_rows`: a workgroup a row with the reduction in a fixed order, or an invocation a row for short rows, whichever the device measures faster, as every row kernel), the per-output loop staying for inner > 1 where it is coalesced. On lavapipe (a software device) the measurement picks the invocation-a-row form, which times 1.8 × slower than the old kernel at 256 × 1,024 (170 → 320 µs, the same build, the old path switched in for the comparison) and the same within noise at 16 rows, for the same SPIR-V instructions: the software renderer's own behaviour, not a GPU's. On the owner's card through Vulkan the measurement picks the workgroup form and the row sum went 42.9 → 11.6 µs at 65k, 104.7 → 11.4 µs at 1M (367 GB/s) and 104.7 → 46.8 µs at 8M (717 GB/s, the card's bandwidth); the backward step 139 → 39 µs at 1M. The other rows moved within their noise (±20 % at 65k; `mul` at 8M read 122 → 208 µs forward with its forward-and-backward unchanged: a one-off). The same run's conformance kit on Vulkan failed one operation, `SampleRows` (a random case: ids 0 where 65 was expected, a statistic -1), which has its own Vulkan kernel and reads no axis sum: not this change's; open, to replay from the kit's saved case.

`train-decoder`, medium, with the bfloat16-products row: 126.9 ms a step, 32,286 tok/s (841 ms cold), 761 launches, 4,405 MiB peak, against 488.4 ms and 8,386 tok/s in float32: 6.2 TFLOP a step at 49 TFLOPS on the tensor cores, 3.85 × the float32 step. The float32 and checkpointed rows repeated within 1 %. Open: the `SampleRows` case on Vulkan.

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
| 7 | Measure every step with the benches; record the numbers here (section 1: the CPU baseline and the first GPU run, with what each exposed and what was fixed) | ongoing |
