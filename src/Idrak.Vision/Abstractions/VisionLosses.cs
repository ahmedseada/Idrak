// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

namespace Idrak.Vision.Abstractions;

/// <summary>
/// The settings of a detection or segmentation loss; each loss reads those it has and
/// ignores the rest.
/// </summary>
public sealed record VisionLossOptions
{
    /// <summary>How the losses become the value returned (default the mean).</summary>
    public LossReduction Reduction { get; init; } = LossReduction.Mean;

    /// <summary>Focal loss: the positives' weight α (default 0.25; negative for no weighting).</summary>
    public float Alpha { get; init; } = 0.25f;

    /// <summary>Focal loss: the focusing exponent γ (default 2).</summary>
    public float Gamma { get; init; } = 2f;

    /// <summary>Smooth L1: where the loss turns from squared to linear (default 1; 0 is the L1 loss).</summary>
    public float Beta { get; init; } = 1f;

    /// <summary>Dice: added to the numerator and the denominator (default 1), so empty classes score 1, not 0/0.</summary>
    public float Smooth { get; init; } = 1f;

    /// <summary>Dice: sum over the whole batch per class (true), or per sample and class (false, the default).</summary>
    public bool Batch { get; init; }

    /// <summary>Box overlap losses: keeps the divisions finite (default torchvision's 1e-7).</summary>
    public float Epsilon { get; init; } = 1e-7f;
}

/// <summary>What a vision loss scores (<see cref="IVisionLoss.Target"/>): what its predictions and targets hold.</summary>
public enum VisionLossTarget
{
    /// <summary>Corner boxes [..., 4] against corner boxes: a detector's boxes.</summary>
    Boxes,

    /// <summary>Logits against 0/1 targets of one shape: classes (a detector's, or each pixel's), scored through a sigmoid.</summary>
    Classes,

    /// <summary>Probabilities against one-hot targets [N, classes, ...]: masks (a segmenter's pixels), the logits softmaxed first.</summary>
    Masks,
}

/// <summary>
/// A detection or segmentation loss:
/// predictions against targets, with gradients to the predictions. What the tensors hold is its <see cref="Target"/>:
/// corner boxes [..., 4] for the box losses, logits and 0/1 targets of one shape for the focal loss, probabilities and
/// one-hot targets [N, classes, ...] for dice.
/// </summary>
public interface IVisionLoss
{
    /// <summary>What it scores, so a trainer gives it the predictions it reads (a segmenter's logits softmaxed for <see cref="VisionLossTarget.Masks"/>).</summary>
    VisionLossTarget Target { get; }

    /// <summary>The loss of <paramref name="predicted"/> against <paramref name="target"/>.</summary>
    Tensor Compute(Tensor predicted, Tensor target, VisionLossOptions options);
}

