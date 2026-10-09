# Plan 13: Idrak.Vision (general image building blocks, fast on every device)

**Status:** planned 2026-10-09, nothing built. Starts after plan 12's phases 4 and 6 are merged, or earlier where the OCR
sample (below) needs a step first.

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
