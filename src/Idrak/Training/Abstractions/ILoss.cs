// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

namespace Idrak.Training.Abstractions;

/// <summary>
/// The loss a trainer minimizes (<c>Trainer.Loss</c>): a scalar tensor from a batch's predictions and targets, which the
/// trainer runs <see cref="Tensor.Backward()"/> from. It stays on the device: the trainer adds it to the epoch's sums there
/// and reads them once per epoch. Its intermediate tensors belong to the autograd graph and are left to the caller's
/// <see cref="TensorScope"/>. <c>Loss</c> makes one from a function (<c>Losses.CrossEntropy</c>, a lambda).
/// </summary>
public interface ILoss
{
    /// <summary>The batch's loss, a scalar tensor, from (predictions, targets).</summary>
    Tensor Compute(Tensor predictions, Tensor targets);
}
