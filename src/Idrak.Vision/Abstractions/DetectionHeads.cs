// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

namespace Idrak.Vision.Abstractions;

/// <summary>
/// A detection network's outputs read as candidate boxes for training (<see cref="DetectionHead"/>): differentiable tensors
/// of the network's outputs, so a loss on them trains the network. Dispose it after the step (a training step's tensor
/// scope does that too).
/// </summary>
/// <param name="Boxes">Each candidate's box, [N, P, 4] corners (x1, y1, x2, y2) in the network input's pixels.</param>
/// <param name="ClassLogits">Each candidate's class scores as logits (a sigmoid gives the probabilities), [N, P, classes].</param>
/// <param name="Objectness">Each candidate's objectness logit, [N, P], or null when the head has none (the class scores alone say whether a candidate is an object).</param>
public sealed record DetectionCandidates(Tensor Boxes, Tensor ClassLogits, Tensor? Objectness) : IDisposable
{
    /// <summary>The images.</summary>
    public int Images => Boxes.Shape[0];

    /// <summary>The candidates per image.</summary>
    public int Candidates => Boxes.Shape[1];

    /// <summary>The classes.</summary>
    public int Classes => ClassLogits.Shape[^1];

    /// <inheritdoc />
    public void Dispose()
    {
        Boxes.Dispose();
        ClassLogits.Dispose();
        Objectness?.Dispose();
    }
}

/// <summary>
/// The training side of a detection decoder (<see cref="IDetectionHeadKind.CreateHead"/>): reads the network's outputs
/// for a batch, [N, ...], as the <see cref="DetectionCandidates"/> the decoder would turn into detections, but as tensors
/// that carry gradients.
/// </summary>
public delegate DetectionCandidates DetectionHead(Tensor outputs);
