// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

namespace Idrak.Vision;

/// <summary>
/// The losses detection trains with, on tensors with gradients to the predictions; registered by name in
/// <see cref="Abstractions.VisionLosses"/>. Boxes are corners (x1, y1, x2, y2), [..., 4], as torchvision's losses take them.
/// </summary>
public static class DetectionLosses
{
    /// <summary>
    /// The overlap loss of each predicted box against its target (<see cref="Tensor.BoxIouLoss"/>: torchvision's IoU-family
    /// formulas in one fused operation), reduced by <paramref name="reduction"/> (<see cref="LossReduction.None"/>: [...]).
    /// </summary>
    public static Tensor BoxIou(Tensor predicted, Tensor target, BoxOverlap overlap = BoxOverlap.GIoU, LossReduction reduction = LossReduction.Mean, float eps = 1e-7f)
    {
        ArgumentNullException.ThrowIfNull(predicted);
        return Reduce(predicted.BoxIouLoss(target, overlap, eps), reduction);
    }

    /// <summary>
    /// PyTorch's <c>smooth_l1_loss</c> per coordinate: 0.5·d²/β where |d| &lt; β, else |d| - 0.5·β; with β = 0 the L1 loss
    /// |d|. Composed as 0.5·m²/β + |d| - m with m = min(|d|, β), so its gradient is the tensors' own.
    /// </summary>
    public static Tensor SmoothL1(Tensor predicted, Tensor target, float beta = 1f, LossReduction reduction = LossReduction.Mean)
    {
        ArgumentNullException.ThrowIfNull(predicted);
        ArgumentNullException.ThrowIfNull(target);
        ArgumentOutOfRangeException.ThrowIfNegative(beta);
        if (!predicted.Shape.SequenceEqual(target.Shape))
        {
            throw new ArgumentException($"Smooth L1 takes predictions and targets of one shape, not {Tensor.FormatShape(predicted.Shape)} and {Tensor.FormatShape(target.Shape)}.");
        }

        var distance = (predicted - target).Abs();
        if (beta == 0)
        {
            return Reduce(distance, reduction);
        }

        var near = distance.Minimum(beta);
        return Reduce(near.Square() * (0.5f / beta) + (distance - near), reduction);
    }

    /// <summary>
    /// torchvision's <c>sigmoid_focal_loss</c> on logits against 0/1 (or soft) targets of the same shape
    /// (<see cref="Tensor.SigmoidFocalLoss"/>, fused), reduced by <paramref name="reduction"/>.
    /// </summary>
    public static Tensor Focal(Tensor logits, Tensor targets, float alpha = 0.25f, float gamma = 2f, LossReduction reduction = LossReduction.Mean)
    {
        ArgumentNullException.ThrowIfNull(logits);
        return Reduce(logits.SigmoidFocalLoss(targets, alpha, gamma), reduction);
    }

    internal static Tensor Reduce(Tensor losses, LossReduction reduction) => reduction switch
    {
        LossReduction.Mean => losses.Mean(),
        LossReduction.Sum => losses.Sum(),
        LossReduction.None => losses,
        _ => throw new ArgumentOutOfRangeException(nameof(reduction)),
    };
}

/// <summary>The losses segmentation trains with besides <c>Losses.PixelCrossEntropy</c>; registered by name in <see cref="Abstractions.VisionLosses"/>.</summary>
public static class SegmentationLosses
{
    /// <summary>
    /// Soft dice loss per class: 1 - (2·Σ p·t + s) / (Σ p + Σ t + s) over each sample's pixels (or the whole batch's with
    /// <paramref name="batch"/>, as MONAI's <c>DiceLoss(batch=True)</c>), on <paramref name="probabilities"/> [N, classes,
    /// ...] (a softmax or sigmoid of the logits) against one-hot or soft <paramref name="targets"/> of the same shape. The
    /// losses are [N, classes] ([classes] with <paramref name="batch"/>) before <paramref name="reduction"/>. Composed from
    /// tensor operations: one product the size of the input, the rest per class.
    /// </summary>
    public static Tensor Dice(Tensor probabilities, Tensor targets, float smooth = 1f, bool batch = false, LossReduction reduction = LossReduction.Mean)
    {
        ArgumentNullException.ThrowIfNull(probabilities);
        ArgumentNullException.ThrowIfNull(targets);
        ArgumentOutOfRangeException.ThrowIfNegative(smooth);
        if (probabilities.Rank < 2 || !probabilities.Shape.SequenceEqual(targets.Shape))
        {
            throw new ArgumentException($"Dice takes probabilities and targets of one shape [N, classes, ...], not {Tensor.FormatShape(probabilities.Shape)} and {Tensor.FormatShape(targets.Shape)}.");
        }

        int n = probabilities.Shape[0], classes = probabilities.Shape[1], pixels = n * classes == 0 ? 0 : probabilities.Size / (n * classes);
        var p = probabilities.Reshape(n, classes, pixels);
        var t = targets.Reshape(n, classes, pixels);
        Tensor intersection = (p * t).Sum(2), total = p.Sum(2) + t.Sum(2);
        if (batch)
        {
            (intersection, total) = (intersection.Sum(0), total.Sum(0));
        }

        var dice = (intersection * 2f + smooth) * (total + smooth).Pow(-1f);
        return DetectionLosses.Reduce(1f - dice, reduction);
    }
}
