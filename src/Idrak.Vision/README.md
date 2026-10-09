# Idrak.Vision

Computer vision for Idrak (no dependencies beyond Idrak): region classification in batches (`RegionClassifier`, with
`ComponentProposer` for the regions), content framing for classifiers of single objects (`ContentFrame`, also as a
loader transform), per-channel image statistics for normalization (`ChannelStatistics`), and detection and segmentation
over any network (`ModelDetector` with the application's `DetectionDecoder`, `ModelSegmenter`), with the small defaults
they build on (foreground extraction, connected components, non-maximum suppression, segmentation metrics).

## Contracts

In `Idrak.Vision.Abstractions` (add the `using` line to name them): `IObjectDetector`, `ISegmenter`, `IRegionProposer`,
`DetectionDecoder`, and the records they speak in (`PixelBox`, `BoundingBox`, `Detection`, `SegmentationMask`,
`ForegroundImage`).

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
using Idrak.Data.Abstractions;   // ImageCodecs
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
