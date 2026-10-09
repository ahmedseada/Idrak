# Idrak.Vision

Computer vision for Idrak (no dependencies beyond Idrak): region classification in batches (`RegionClassifier`, with
`ComponentProposer` for the regions), content framing for classifiers of single objects (`ContentFrame`, also as a
loader transform), per-channel image statistics for normalization (`ChannelStatistics`), and detection and segmentation
over any network (`ModelDetector` with the application's `DetectionDecoder`, `ModelSegmenter`), with the small defaults
they build on (foreground extraction, connected components, non-maximum suppression, segmentation metrics).

Training detectors and segmenters: box losses (IoU, GIoU, DIoU, CIoU as one fused operation each way; L1 and smooth L1),
the sigmoid focal loss and soft dice (`DetectionLosses`, `SegmentationLosses`, by name in `VisionLosses`); matching
predictions to objects by IoU thresholds or one to one by the Hungarian algorithm (`BoxMatching`, `BoxMatchers`); COCO and
Pascal VOC mean average precision and mean IoU (`CocoAveragePrecision`, `VocAveragePrecision`, `SegmentationMetrics`, by
name in `VisionMetrics`); augmentations that move boxes and masks with the pixels (resized crop, rotation, affine, colour
jitter, cutout, mosaic, mixup, resize; core's flip and shift; by name in `Augmentations`), and `AugmentedImageLoader`,
which batches augmented samples on worker threads while the model trains.

## Contracts

In `Idrak.Vision.Abstractions` (add the `using` line to name them): `IObjectDetector`, `ISegmenter`, `IRegionProposer`,
`DetectionDecoder`, `VisionLoss` and `VisionLosses`, `BoxMatcher` and `BoxMatchers`, `IVisionMetric` (`IDetectionMetric`,
`ISegmentationMetric`) and `VisionMetrics`, and the records they speak in (`Detection`, `SegmentationMask`,
`ForegroundImage`). Boxes (`BoundingBox`, `PixelBox`), training samples (`AnnotatedImage`), dataset annotations
(`AnnotatedDataset`, `ObjectAnnotation`, `ObjectMask`) and augmentations (`IAugmentation`, `Augmentations`) are in
`Idrak.Abstraction.Data`, which core and Idrak.Data use too.

`DetectionDecoders` (also in `Idrak.Vision.Abstractions`) holds the detection decoders by name: `ModelDetector` takes one
by name, and an image model family names its own. The library registers only the generic "boxes-scores" (a network that
already outputs boxes and class scores); a family's decoder (anchors, grids, box encodings) is registered by the plug-in
or app that brings the family.

## Image models from their families

Image model families (classifiers, detectors, segmenters, backbones) are plug-ins: core's `ImageModelFamilies` registry,
filled by whoever brings a family (the library registers none). `ImageModels.Load` (core) reads a checkpoint (a
safetensors folder with config.json, or an ONNX file) through its family; this package makes it ready for images with
the family's preprocessing:

```csharp
using Idrak.Models;   // ImageModels
using Idrak.Vision;

ImageModelFamilies.Register(new MyDetectorFamily());          // a plug-in's family
DetectionDecoders.Register("my-grid", MyGridDecoder.Create);  // and its decoder
using var model = ImageModels.Load("models/my-detector");
var detections = model.Detector(new DetectorOptions { MinScore = 0.4f }).Detect(ImageCodecs.Decode("street.jpg"));
// model.Classifier().Build(), model.Segmenter(), model.Outputs().Build(), RegionClassifier.For(model)
```

## Install

```bash
dotnet add package Idrak.Vision
```

## Example

```csharp
using Idrak.Vision;

// Single objects on a plain background (symbols, parts, cells, characters): regions found, framed, classified in batches.
using var classifier = RegionClassifier.Load("shapes.ikm").Build();   // a package from Predictor.Save
var found = classifier.Classify(ImageCodecs.Decode("board.png"), new ComponentProposer());
for (int i = 0; i < found.Count; i++) Console.WriteLine($"{found.Label(i)} at {found.Boxes[i]} ({found.Confidence(i):P0})");
```

Part of [Idrak](https://www.nuget.org/packages/Idrak), a self-contained deep-learning library for .NET.

## Documentation

Guides, samples and the full API overview: https://github.com/ahmedseada/Idrak

License: Apache 2.0.
