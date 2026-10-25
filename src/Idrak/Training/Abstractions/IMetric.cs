// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

namespace Idrak.Training.Abstractions;

/// <summary>
/// A quantity tracked during training and evaluation (<c>Trainer.Metrics</c>), a mean over samples: the trainer adds each
/// batch's mean, weighted by the batch's samples, on the device (without gradients), and reads the sums once per epoch,
/// so a metric reads nothing back to the host per batch. <c>Metric</c> makes one from functions.
/// </summary>
public interface IMetric
{
    /// <summary>Short name in logs and results ("mae").</summary>
    string Name { get; }

    /// <summary>The batch's mean as a scalar tensor on the predictions' device, from (predictions, targets).</summary>
    Tensor BatchMean(Tensor predictions, Tensor targets);

    /// <summary>The reported value from the epoch's mean over samples (a square root for RMSE); the mean itself by default.</summary>
    double Finish(double mean) => mean;
}
