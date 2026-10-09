// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Idrak.Abstraction.Devices;
using Idrak.Abstraction.Diagnostics;

namespace Idrak.Abstraction;

/// <summary>How two boxes' overlap becomes a loss (<see cref="Tensor.BoxIouLoss"/>); torchvision's formulas.</summary>
public enum BoxOverlap
{
    /// <summary>1 - intersection over union.</summary>
    IoU,

    /// <summary>Generalized IoU (Rezatofighi et al. 2019): also the share of the enclosing box neither box covers, so boxes apart still pull together.</summary>
    GIoU,

    /// <summary>Distance IoU (Zheng et al. 2020): also the centres' squared distance over the enclosing box's squared diagonal.</summary>
    DIoU,

    /// <summary>Complete IoU (Zheng et al. 2020): DIoU plus an aspect-ratio term.</summary>
    CIoU,
}

public sealed partial class Tensor
{
    /// <summary>
    /// The overlap loss of each predicted box (this tensor, [..., 4] corners x1, y1, x2, y2) against its target box
    /// (the same shape): [...] losses, by <paramref name="overlap"/>'s formula (torchvision's
    /// <c>generalized_box_iou_loss</c>, <c>distance_box_iou_loss</c>, <c>complete_box_iou_loss</c>, or 1 - IoU), in one pass.
    /// The gradient flows to the predicted boxes; the targets are constants.
    /// </summary>
    /// <param name="target">The target boxes, the same shape.</param>
    /// <param name="overlap">The loss's formula.</param>
    /// <param name="eps">Keeps the divisions finite (torchvision's default 1e-7).</param>
    public Tensor BoxIouLoss(Tensor target, BoxOverlap overlap = BoxOverlap.GIoU, float eps = 1e-7f)
    {
        ThrowIfDisposed();
        ArgumentNullException.ThrowIfNull(target);
        target.ThrowIfDisposed();
        CheckSameDevice(this, target);
        if (Rank == 0 || _shape[^1] != 4 || !_shape.AsSpan().SequenceEqual(target._shape))
        {
            throw new ArgumentException($"Box losses take predicted and target corners of one shape [..., 4], not {FormatShape(_shape)} and {FormatShape(target._shape)}.");
        }

        if (!Enum.IsDefined(overlap))
        {
            throw new ArgumentOutOfRangeException(nameof(overlap));
        }

        int count = Size / 4;
        long start = Telemetry.Start(TelemetryLevel.Operations);
        var y = Empty(_shape.AsSpan(0, Rank - 1), Device);
        Backend.BoxIouLoss(Storage, target.Storage, y.Storage, count, overlap, eps);
        if (WillRecord(this))
        {
            var x = this;
            y.Record("box_iou_loss", g => x.Backend.BoxIouLossBackward(x.Storage, target.Storage, g.Storage, x.GradStorage(), count, overlap, eps), x);
        }

        return Traced("box_iou_loss", y, start);
    }

    /// <summary>
    /// The sigmoid focal loss of each logit (this tensor) against its target in [0, 1] (the same shape), torchvision's
    /// <c>sigmoid_focal_loss</c> with reduction none: α_t · (1 - p_t)^γ · BCE, in one pass. The gradient flows to the
    /// logits; the targets are constants.
    /// </summary>
    /// <param name="targets">The targets (0 or 1, or soft), the same shape.</param>
    /// <param name="alpha">The weight of the positives (1 - α of the negatives); negative for no weighting.</param>
    /// <param name="gamma">How much easy examples are discounted (0: plain binary cross-entropy).</param>
    public Tensor SigmoidFocalLoss(Tensor targets, float alpha = 0.25f, float gamma = 2f)
    {
        ThrowIfDisposed();
        ArgumentNullException.ThrowIfNull(targets);
        targets.ThrowIfDisposed();
        CheckSameDevice(this, targets);
        if (!_shape.AsSpan().SequenceEqual(targets._shape))
        {
            throw new ArgumentException($"The focal loss takes logits and targets of one shape, not {FormatShape(_shape)} and {FormatShape(targets._shape)}.");
        }

        if (!float.IsFinite(gamma) || gamma < 0 || alpha > 1 || float.IsNaN(alpha))
        {
            throw new ArgumentOutOfRangeException(gamma < 0 || !float.IsFinite(gamma) ? nameof(gamma) : nameof(alpha), "γ ≥ 0 and α ≤ 1 (negative: no weighting).");
        }

        int count = Size;
        long start = Telemetry.Start(TelemetryLevel.Operations);
        var y = Empty(_shape, Device);
        Backend.SigmoidFocalLoss(Storage, targets.Storage, y.Storage, count, alpha, gamma);
        if (WillRecord(this))
        {
            var x = this;
            y.Record("sigmoid_focal_loss", g => x.Backend.SigmoidFocalLossBackward(x.Storage, targets.Storage, g.Storage, x.GradStorage(), count, alpha, gamma), x);
        }

        return Traced("sigmoid_focal_loss", y, start);
    }
}
