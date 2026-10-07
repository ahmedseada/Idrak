# Idrak.Vision

Computer vision for Idrak (no dependencies beyond Idrak): region classification in batches (`RegionClassifier`, with
`ComponentProposer` for the regions), content framing for classifiers of single objects (`ContentFrame`, also as a
loader transform) and per-channel image statistics for normalization (`ChannelStatistics`). The contracts and small
defaults it builds on (foreground extraction, connected components, boxes, non-maximum suppression, detectors and
segmenters over any network) are in Idrak.Abstraction.

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
