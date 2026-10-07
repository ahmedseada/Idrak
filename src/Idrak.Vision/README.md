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
