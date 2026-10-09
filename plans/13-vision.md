# Plan 13: Idrak.Vision (general image building blocks, fast on every device)

**Status:** planned 2026-10-09. Built 2026-10-09: step 3 (CTC loss and decoding) and the part of step 2 the line
recognizer uses (rectangular, strided, dilated and grouped convolutions, rectangular max and average pooling,
bidirectional and stacked LSTM/GRU); then the rest of step 2 (transposed convolution, upsampling, adaptive pooling,
`GroupNorm`, ceil-mode pooling and padding below and right); see "as built" below. The rest starts after plan 12's
phases 4 and 6 are merged, or earlier where the OCR sample (below) needs a step first.

**Goal.** `Idrak.Vision` and the core layers under it are general building blocks for any image application
(classification, detection, segmentation, document reading), fast and lean on every device. Applications are not the
library's target: OCR is an application, built as a sample on the public API (as `samples/Gemma3Vision` is), and the
library holds nothing OCR-specific. Every step keeps the rules of `CLAUDE.md`:

- **Card-agnostic.** Tiles, layouts and paths are read from what the device reports or measured on it (per shape,
  cached like the matrix-product tuning); no card, vendor or memory size in code, comments, tests or docs.
- **Abstraction.** New behaviour is a contract with a `SlotTable` registry, placed by decision 10 (plan 10).
- **No family in the library.** Image model families (ResNet, ViT, YOLO-style, ...) are plug-ins or samples; an
  unregistered family fails with the registry's "not registered" message.
- **Pure C#.** No package references; kernels generated in C# (PTX, SPIR-V) as today.
- **Tests.** Targeted CPU tests only (`IDRAK_DEVICES=cpu IDRAK_FILTER=...`); the owner runs CUDA and Vulkan from a
  command list given at the end of each step.

## What Idrak has, and what is missing

| Part | Today |
|---|---|
| `Idrak.Vision` (about 1,400 lines) | `RegionClassifier` + `ComponentProposer`, `ContentFrame`, foreground and connected components, non-maximum suppression, `ModelDetector` (the app writes its `DetectionDecoder`), `ModelSegmenter`, segmentation metrics, `ChannelStatistics` |
| Convolution layers (core) | `Conv2d` (square kernels; no groups, no dilation), `MaxPool2d`, `GlobalAveragePool2d`, `BatchNorm`, `Flatten` |
| Other layers (core) | `LayerNorm`, `MultiHeadAttention`, `TransformerEncoderLayer`, `LSTM`, `GRU` (one direction) |
| Convolution on the GPU | CUDA kernels; on Vulkan, convolution and group-norm training run on the host fallback (a round trip to the CPU per convolution) |
| Image input | PNG, BMP, Netpbm, JPEG codecs; Pillow-exact transforms (`ImageTransforms`: grayscale, max_width/height, contrast, brightness, sharpness, autocontrast, jpeg); `ImagePreprocessor`; resize and normalize on the host |
| Augmentation | `RandomFlip`, `RandomShift` (image only: no boxes or masks) |
| Losses for sequences | none for unsegmented text lines (no CTC) |
| Detection and segmentation training | `PixelCrossEntropy`; no IoU/GIoU, focal or dice loss, no matching, no mAP |
| Dataset formats | image folders, CSV, Parquet, chat rows; no COCO, YOLO text or Pascal VOC |
| Image model families | none as a registry (vision-language families have `VisionFamilies`) |
| Vision-language machinery | done (plans 11 and 12): encoders, image prompts, span attention, tuning with images, families as plug-ins |

## Steps

| # | Step | What | Where | Done when (CPU here; GPU by the owner) |
|---|---|---|---|---|
| 1 | **Convolution speed and memory on every device** | Device kernels for the convolution forward and both backward passes (implicit GEMM) on Vulkan and CUDA; tiles and layout (NCHW or NHWC) measured per shape and cached; bfloat16 and matrix units where the device reports them; fused convolution + batch norm + activation for inference; image resize and normalize on the device | `Idrak.Gpu`, core, operations through the dispatcher | the conformance kit's convolution cases pass on the CPU and replay on the device; a CNN trains on Vulkan without host round trips (the owner's run); measured choice never slower than the host fallback |
| 2 | **Layers modern backbones need** | Grouped and depthwise convolution, dilation, rectangular kernels and strides; `ConvTranspose2d`; upsampling (nearest, bilinear); `AvgPool2d`, adaptive pooling; `GroupNorm` as a layer; bidirectional `LSTM`/`GRU` | core `Idrak.Layers`, with ONNX import and export translators | CPU reference, gradient checks against finite differences, conformance-kit cases, an ONNX import test per layer |
| 3 | **Sequence losses** | CTC loss (forward and gradient, blank, log space) and greedy and beam CTC decoding, as registrations | core (loss), contracts by decision 10 | matches a plain-loop reference and finite differences; the kit's suite |
| 4 | **Training detection and segmentation** | Losses: IoU / GIoU, focal, dice. Matching: IoU threshold and Hungarian assignment. Metrics: mAP, mIoU | `Idrak.Vision`, each a contract with a registry and defaults | values on known cases; kit suites |
| 5 | **Augmentation and dataset formats** | Augmentations that move boxes and masks with the image (random resized crop, rotation, affine, color jitter, cutout, mosaic, mixup), measured to run off the training thread; COCO, YOLO text and Pascal VOC readers as registrations | core / `Idrak.Vision`; formats in `Idrak.Data` | boxes and masks stay aligned; each format round-trips; kit suites |
| 6 | **Image model families as plug-ins** | A registry of image model families (classifiers, detectors, segmenters) by architecture name, read from safetensors or ONNX; a `DetectionDecoders` registry; the library registers none | contracts by decision 10; families in samples / plug-ins | a tiny family from a plug-in loads and matches its reference; unregistered names the registry |
| 7 | **Command line** | `idrak predict` with detectors and segmenters from any registered family; `idrak train` for image models with step 5's augmentations and step 4's metrics | `Idrak.Cli` | CLI tests on tiny models (CPU) |

Separate list (core, for later vision-language plug-ins, not `Idrak.Vision`): M-RoPE and window attention in a vision
tower (Qwen2.5-VL); cross-attention in the decoder (TrOCR, Donut); the BOS post-processor gap real LLaVA checkpoints hit.

**Order.** 1, then 2 (they share the convolution kernels and their tests); 3 with 2 when the OCR sample needs it; 4 and 5
together; 6 once the layers exist; 7 last.

## The OCR sample (an application on the library)

An app in `samples/`, built only on the public API, that reads document images into text (Arabic first). The library
gains nothing OCR-specific; what the app needs from this plan comes from the steps above.

| Approach | What it needs from the library | Plan 13 steps first |
|---|---|---|
| **A vision-language model** (a Gemma 3 fine-tune such as `bakrianoo/arabic-legal-documents-ocr-1.0`, through the `samples/Gemma3Vision` plug-in) | everything exists: loading, image transforms, pan and scan, generation, tuning with images (plan 12) | none |
| **A trained line recognizer** (convolutional features, a bidirectional recurrent layer, CTC; page → lines → text) | rectangular kernels and strides, bidirectional LSTM/GRU, CTC loss and decoding, text-line augmentation, the convolution speed on the GPU | 3, the parts of 2 it uses, then 1 |

**Decided 2026-10-09: both**, in one app, chosen per run. So plan 13 starts with step 3 and the parts of step 2 the
recognizer uses (rectangular kernels and strides, bidirectional recurrent layers), then step 1; the sample follows.

## Step 3, as built (2026-10-09)

- **Loss.** `Losses.Ctc(logProbs, targets, inputLengths, targetLengths, blank = 0, reduction = Mean, zeroInfinity = false,
  batchFirst = false)` and an overload taking each sequence's labels as an `int[]`; `LossReduction` (`Mean`, `Sum`, `None`)
  in core. The same semantics as PyTorch's `F.ctc_loss`: log-probabilities [T, N, C] (or [N, T, C] with `batchFirst`),
  padded [N, S] or concatenated targets, `Mean` divides each loss by its target length (at least 1) and averages,
  `zeroInfinity` turns an impossible sequence's loss and gradient into 0. Underneath, `Tensor.CtcLoss(...)` gives the
  [N] losses with autograd.
- **Operations.** `Ops.CtcLoss` and `Ops.CtcLossBackward` through the dispatcher (`Backend.CtcLossKernel`,
  `CtcLossBackwardKernel`; lengths and target offsets as spans, targets as a storage of ids). CPU kernel
  (`CpuBackend.Ctc.cs`): α and β in log space and double precision over the 2L + 1 extended states, only the states an
  alignment can be in at each step, parallel over the batch (each sequence writes only its own gradient rows). The loss
  keeps two rows of α (O(2L + 1) a sequence); the gradient recomputes α into an `ArrayPool` buffer (O(T·(2L + 1))) and
  runs β a row at a time, so the forward pass stores nothing for the backward one and no [T, N, C] gradient is held
  beyond the one autograd accumulates into (only the label classes are written). CUDA, Vulkan and HIP use the host
  fallback (no device kernel yet: a log-space dynamic program is sequential in time; worth a kernel once a GPU recognizer
  is measured).
- **Gradient.** The derivative with respect to the log-probabilities themselves (checked against finite differences);
  behind a log-softmax it equals PyTorch's, which returns its gradient with respect to the scores directly.
- **Decoding.** `CtcDecoders` (`Idrak.Inference.Abstractions`, a `SlotTable` with Throw / FallBack / Shadow; Shadow
  compares the labels): the contract is the delegate `CtcDecoder(ReadOnlyMemory<float> logProbs, int steps, int
  classes, CtcDecodeOptions options)` → `IReadOnlyList<CtcHypothesis>` (labels and log-probability). Built in: "greedy"
  (best path, repeats collapsed, blanks dropped) and "beam" (prefix beam search: `BeamWidth`, `Results`, per-step pruning
  `TopClasses` and `MinLogProbability`; candidates keyed by (prefix, label), only kept prefixes become trie nodes, top-k
  by a heap). `CtcDecoders.Decode(Tensor logProbs, lengths, name, options, batchFirst)` decodes a batch. Placement by
  decision 10: core is its only library user.
- **Conformance kit.** "sequence losses: CTC loss and its gradient, against every alignment listed" (both layouts, empty
  and repeated labels, an impossible sequence, random sizes for device comparison).
- **Reference.** `tools/pytorch/vision_sequence_reference.py` (torch 2.14.1 on the CPU, onnx 1.23.2 for two exports)
  writes `tests/Idrak.Tests/data/vision-sequence` (reruns write the same bytes); `IDRAK_FILTER="vision sequence"`
  (10 tests) checks the losses (all reductions, zero infinity, batch first, blank 4) and the gradient on the logits to
  1e-5, the beam search against every path listed, and finite differences.
- **Measured** (this container's CPU, 4 threads): loss and gradient of 32 lines of 100 steps, 80 classes, 25 labels in
  27 ms; 16 lines of 400 steps, 200 classes, 80 labels in 109 ms (log-softmax included; about 1.7x faster than the first
  version after a one-logarithm three-way sum and linear-space gradient sums). Beam width 10 over 200 steps of 100
  classes: 71 ms with every class, 4 ms with `TopClasses = 8`; greedy 0.08 ms.

## Step 2 (the recognizer's part), as built (2026-10-09)

- **Dilation in the window geometry.** `ConvGeometry` has `DH` / `DW` (init-only, default 1, stored so `default` and
  every existing geometry keep dilation 1) and `Dilated`; `OH` / `OW` account for it. The CPU im2col and col2im read
  dilated windows; CUDA and Vulkan send a dilated window to the host fallback until step 1's kernels (non-dilated
  windows are unchanged on every device). Max pooling refuses a dilated geometry. The conformance kit's convolution case
  adds rectangular, strided, padded and dilated geometries, and rectangular pooling windows.
- **`Conv2d`.** A second constructor `Conv2d(in, out, (kh, kw), stride?, padding?, dilation?, groups = 1, bias, device,
  random)`; the square constructor is the same layer (same weights from the same seed). Groups run as one batched
  product ([G, N·OH·OW, C/G·kh·kw] × [G, OC/G, C/G·kh·kw]ᵀ), depthwise included; the weight is PyTorch's [out, in/groups,
  kh, kw] flattened. `FromWeights` takes the pairs and groups; properties `KernelHeight/Width`, `StrideHeight/Width`,
  `PaddingHeight/Width`, `DilationHeight/Width`, `Groups`, `IsSquare`, `Geometry(...)` (`KernelSize`, `Stride`, `Padding`
  stay: the height values).
- **Pooling.** `MaxPool2d((kh, kw), stride?, padding?)`; new `AvgPool2d` (square and pair constructors,
  `countIncludePad` as PyTorch's, default true): each channel's windows unfolded by im2col and averaged, so it runs on
  every device's im2col and its gradient is col2im's.
- **Recurrent layers.** `LSTM(in, hidden, returnSequences, bidirectional, layers = 1)` and `GRU(..., bidirectional,
  layers = 1, candidateBias = false)`: the backward direction reads the sequence reversed and the two hidden states are
  concatenated (forward first), as PyTorch's `bidirectional=True` with `batch_first`; without `returnSequences` the
  output is the last state of each direction ([N, 2H], PyTorch's `h_n[-2:]`). Stacked layers read the previous one's
  every step. `RecurrentModule` now owns the time loop, directions and layers; a cell type implements `Cell(projected,
  state, weights)` and `States` (the old `Run`/`Step` pair is gone, decision 7). `RecurrentWeights` per layer and
  direction (PyTorch's order), `LoadGateWeights(layer, reverse, weight_ih, weight_hh, bias_ih, bias_hh)` loads PyTorch's
  layout. The GRU's `candidateBias` keeps the candidate gate's recurrent bias apart inside the reset product (PyTorch's
  `b_hn`), so PyTorch's GRU weights load exactly; without it a non-zero `b_hn` is refused with that remedy. One-direction,
  one-layer layers keep their parameters and saved files.
- **Builder, layer types, saving.** `NetworkBuilder.Conv2d(out, (kh, kw), ...)`, `MaxPool2d((kh, kw), ...)`,
  `AvgPool2d(...)`, `LSTM/GRU(hidden, returnSequences, bidirectional, layers[, candidateBias])` and `ColumnsToSequence()`
  ([C, H, W] → [W, C·H], a feature map's columns as steps); `NetworkOps` replays them ("avgpool2d",
  "columnsToSequence"; pairs written as [h, w] only when they differ) and `LayerTypes` describes them ("avgpool2d";
  square and one-direction layers are described exactly as before).
- **ONNX.** Export: `Conv` with `kernel_shape`, `strides`, `pads`, `dilations`, `group`; `MaxPool` and `AveragePool`
  (`count_include_pad`) with rectangular windows; LSTM/GRU with `direction = "bidirectional"` and one node per stacked
  layer, Y read as PyTorch writes it (Transpose [0, 2, 1, 3], Reshape [0, 0, -1]); the GRU candidate bias in the
  recurrent bias. ONNX Runtime matches. Import: the same, plus PyTorch's own exports (zero initial states expanded to a
  computed shape are recognized; `tests/.../vision-sequence/conv.onnx` and `recurrent.onnx` import and match PyTorch).
- **Checked.** PyTorch's outputs for rectangular / dilated-grouped / depthwise convolutions (1e-4), pools (exact /
  1e-5), a bidirectional two-layer LSTM and GRU with random biases (1e-5, every step and the last states); finite
  differences for every new path; a convolutional recurrent network replayed from JSON, saved and loaded, exported
  (ONNX Runtime) and imported again. Existing convolution, recurrent, ONNX, gradient, builder, conformance-kit,
  inventory and public-API tests pass on the CPU.
- **Left of step 2:** `ConvTranspose2d`, upsampling, adaptive pooling, `GroupNorm` as a layer; `ceil_mode` pooling and
  asymmetric padding (all built since: "Step 2 (the rest)" below); a dedicated depthwise kernel (the batched product is general, not the fastest for depthwise) and
  device kernels for dilated windows and CTC (step 1); importing PyTorch exports that read the last time step through
  computed gathers. The CLI's network description (`NetworkAnalysis`) does not know the new steps or [h, w] pairs yet.

**For the owner (GPU, from `D:\Projects\Idrak`):**

```powershell
$env:IDRAK_DEVICES="cuda"; $env:IDRAK_FILTER="vision sequence"; dotnet run -c Release --project tests/Idrak.Tests
$env:IDRAK_DEVICES="cuda"; $env:IDRAK_FILTER="conformance kit"; dotnet run -c Release --project tests/Idrak.Tests
$env:IDRAK_DEVICES="vulkan"; $env:IDRAK_FILTER="vision sequence"; dotnet run -c Release --project tests/Idrak.Tests
$env:IDRAK_DEVICES="vulkan"; $env:IDRAK_FILTER="conformance kit"; dotnet run -c Release --project tests/Idrak.Tests
$env:IDRAK_DEVICES="vulkan"; $env:IDRAK_FILTER="vulkan cnn"; dotnet run -c Release --project tests/Idrak.Tests
```

## Step 2 (the rest), as built (2026-10-09)

- **Window geometry.** `ConvGeometry` has `PadBottom` / `PadRight` (init-only; PH / PW unless set, stored so `default`,
  every existing geometry and one set to PH compare equal) and `Asymmetric`; `OH` / `OW` use them. They change only how
  many windows there are, never where they start, so every existing kernel (CPU, CUDA, Vulkan: they read OH and OW from
  the host and skip positions outside the input) runs them unchanged. The CPU's col2im now runs in parallel over (image,
  channel) planes instead of images, so a batch of one (a transposed convolution, a decoder) uses every core.
- **New operations** (dispatcher, CPU kernels in `CpuBackend.Resampling.cs`, host fallback elsewhere):
  `Interpolate2d` / `Interpolate2dBackward` (`InterpolationMode` Nearest or Bilinear, align corners, the coordinate
  scales passed in as PyTorch computes them in float; index and weight tables per axis built once a call; forward in
  parallel over output rows, backward over planes, each writing only its own part of dx), `AdaptiveAvgPool` /
  `AdaptiveAvgPoolBackward` and `AdaptiveMaxPool` (PyTorch's windows [floor(o·H/OH), ceil((o+1)·H/OH)); the max keeps
  arg-max indices, so `MaxPoolBackward` is its gradient). `Tensor.Col2Im(geometry)` (the fold with autograd; its
  gradient is im2col, written in place when it is the first), `Tensor.Interpolate(size, mode, alignCorners, scaleFactor)`,
  `Tensor.AdaptiveAvgPool(size)`, `Tensor.AdaptiveMaxPool(size)`. Conformance kit: "image resampling" (every mode, up,
  down, odd ratios, one row or column, against plain loops, gradients included) and padding below and right in the
  convolution and pooling case.
- **`ConvTranspose2d`** (square and pair constructors, `FromWeights`; PyTorch's semantics and weight layout [in,
  out/groups, kh, kw]; stride, padding, output padding below the stride, dilation, groups, bias): one product
  [N·H·W, in] × [in, out·kh·kw] (batched per group) and a fold, so it runs on the devices' product and col2im; its
  backward is im2col and two products. `OutputSize(h, w)`, `Geometry(...)`.
- **`Upsample`** (`new Upsample(factor or (h, w) factors, mode, alignCorners)`, `Upsample.ToSize((h, w), ...)`):
  F.interpolate's semantics, the output floor(size · factor) and, as PyTorch, a given factor (as written: 1.7f is read as
  1.7) sets the coordinate scale. **`AdaptiveAvgPool2d` / `AdaptiveMaxPool2d`** (int or (h, w)): when the output divides
  the input the windows are ordinary ones, and a device without its own adaptive kernel (`Kernels.Chain`) pools them with
  its max-pool / im2col kernels instead of the host; the CPU runs the adaptive kernel. **`GroupNorm(groups, channels,
  epsilon, affine)`** over [N, C, ...]: the batch-norm kernels on the [1, N·groups, C/groups · positions] view (each
  sample's group is one contiguous run; the statistics are freed at once), then the per-channel affine; the same in
  training and evaluation.
- **Pooling.** `MaxPool2d` and `AvgPool2d` pair constructors take `ceilMode` and `paddingEnd` (padding below and right);
  windows counted as PyTorch and ONNX do (a last ceil-mode window must start inside the input or the padding before it),
  the ceil mode's extra windows given to the geometry as padding below and right; `AvgPool2d` divides as PyTorch does
  (counted padding up to the padded end, a ceil-mode window cut there; without `countIncludePad` the input positions
  covered). `MaxPool2d.Geometry(...)`. Padding of at most half the window on each side.
- **Builder, layer types, saving.** `ConvTranspose2d(...)`, `Upsample(factor)`, `UpsampleToSize(size)`,
  `AdaptiveAvgPool2d(...)`, `AdaptiveMaxPool2d(...)`, `GroupNorm(groups, epsilon, affine)`, and the pooling options;
  `NetworkOps` replays them ("convtranspose2d", "upsample", "adaptiveavgpool2d", "adaptivemaxpool2d", "groupnorm";
  "ceilMode" and "paddingEnd" written only when set) and `LayerTypes` describes them (pooling layers without the new
  options are described as before). `idrak explain` / `viz` (`NetworkAnalysis`) know every new step (description,
  parameters, FLOPs); the CLI test builds a decoder with all of them and checks the parameter count and that no step is
  unknown.
- **ONNX.** Export: `ConvTranspose` (output_padding, dilations, group), `Resize` (nearest: asymmetric + floor; bilinear:
  half_pixel or align_corners; scales [1, 1, h, w], or sizes computed from the input's shape for a fixed size, as PyTorch
  writes them), GroupNorm as opset 17 has it (Reshape, InstanceNormalization, Reshape, Mul, Add, as PyTorch), adaptive
  pools as GlobalAveragePool / GlobalMaxPool (1x1) or AveragePool / MaxPool when the output divides the input (else a
  clear error, as PyTorch's exporter), pools with four pads and `ceil_mode`. ONNX Runtime matches. Import: the same,
  plus `GroupNormalization` (opset 18's per-group and 21's per-channel scale), `InstanceNormalization` (group norm with
  one channel a group), PyTorch's group norm with the shape read through `Shape`, PyTorch's resize to a size (Shape,
  Slice, Concat), `GlobalAveragePool` without a flatten and `GlobalMaxPool` (adaptive pooling to 1x1), Resize-10's
  (X, scales), in chains and in graphs.
- **Checked** (`IDRAK_FILTER="vision layers"`, 9 tests): `tools/pytorch/vision_layers_reference.py` (torch 2.14.1 CPU,
  onnx 1.23.2; reruns write the same bytes) writes `tests/Idrak.Tests/data/vision-layers`: PyTorch's outputs and the
  gradients of sum(output · w) on input and parameters for two transposed convolutions (1e-4), eight interpolations
  (1e-5), four adaptive pools (1e-5), two group norms (1e-4) and three ceil-mode pools (exact / 1e-5); finite
  differences for every layer; a decoder of every new step replayed from JSON, saved and loaded, exported (ONNX
  Runtime) and imported again; PyTorch's `decoder.onnx` and the hand-written `windows.onnx` (padding below and right,
  GroupNormalization-21, a Resize to constant sizes; output by onnx's reference evaluator) import and match (1e-4).
- **Measured** (this container's CPU, 4 threads shared with other work, so a range): `ConvTranspose2d` 64 → 32, kernel
  4, stride 2 on one 64x64 image: forward 12–20 ms with the planes' fold in parallel, 22–24 ms with it on one core (the
  old per-image split); interpolation kernels over 512 planes 64x64 → 128x128: nearest 8 ms forward, 7–10 ms backward,
  bilinear about 25 ms each way; adaptive average pooling of [8, 256, 30, 30] to 7x7 (uneven windows) 2–3 ms forward;
  `GroupNorm` (32 groups) over [8, 64, 64, 64] costs about what `BatchNorm` does (forward 9–11 ms each). Memory: the
  transposed convolution holds its input's rows [N·H·W, in] and the columns [N·H·W, out·kh·kw] until the backward
  pass (as `Conv2d` holds its unfolded patches); interpolation and adaptive average pooling keep nothing but their
  input, adaptive max pooling its indices (one int per output), group norm its output and one inverse deviation per
  (sample, group).
- **On GPUs** (no device kernels written here; step 1's): `Interpolate2d`, `Interpolate2dBackward`, `AdaptiveAvgPool`,
  `AdaptiveAvgPoolBackward` and `AdaptiveMaxPool` run on the host fallback on CUDA, Vulkan and HIP (adaptive pools whose
  output divides the input take the devices' window kernels instead); adaptive max pooling's gradient uses
  `MaxPoolBackward(count)`, which Vulkan runs on the host (its own kernel is the geometry overload). `ConvTranspose2d`
  uses the products and col2im/im2col every device has (a dilated one falls back, as `Conv2d`'s does); `GroupNorm` the
  batch-norm kernels (on the host on Vulkan in training, as before); padding below and right and ceil mode need nothing
  new.
- **Left:** a fused group-norm kernel (statistics and affine in one pass), a dedicated depthwise kernel, adaptive pooling
  with uneven windows on ONNX export (ONNX has none), ConvTranspose's `output_shape` and output padding at or above the
  stride, Resize's other coordinate modes (`asymmetric` linear, `tf_crop_and_resize`), cubic and antialiased resizing.

**For the owner (GPU, from `D:\Projects\Idrak`):**

```powershell
$env:IDRAK_DEVICES="cuda"; $env:IDRAK_FILTER="vision layers"; dotnet run -c Release --project tests/Idrak.Tests
$env:IDRAK_DEVICES="cuda"; $env:IDRAK_FILTER="conformance kit"; dotnet run -c Release --project tests/Idrak.Tests
$env:IDRAK_DEVICES="vulkan"; $env:IDRAK_FILTER="vision layers"; dotnet run -c Release --project tests/Idrak.Tests
$env:IDRAK_DEVICES="vulkan"; $env:IDRAK_FILTER="conformance kit"; dotnet run -c Release --project tests/Idrak.Tests
$env:IDRAK_DEVICES="vulkan"; $env:IDRAK_FILTER="vulkan cnn"; dotnet run -c Release --project tests/Idrak.Tests
```
