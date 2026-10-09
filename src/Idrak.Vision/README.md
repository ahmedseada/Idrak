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
