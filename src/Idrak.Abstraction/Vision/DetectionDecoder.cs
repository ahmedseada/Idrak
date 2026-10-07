// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.


namespace Idrak.Abstraction.Vision;

/// <summary>
/// Turns one image's network outputs into detections, in the network's input pixels (width x height of
/// Idrak.Vision's <c>ModelDetector</c>). This is the part that depends on the network (anchors, grid cells, box encoding,
/// how scores are stored); the detector does the rest.
/// </summary>
/// <param name="outputs">The outputs of one image (the model's output for a batch of one, flattened).</param>
/// <param name="outputShape">Their shape, without the batch dimension.</param>
public delegate IEnumerable<Detection> DetectionDecoder(ReadOnlySpan<float> outputs, IReadOnlyList<int> outputShape);
