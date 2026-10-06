// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

namespace Idrak.Abstraction.Data;

/// <summary>
/// One mini-batch: [Size, ..FeatureShape] features and [Size, ..TargetShape] targets on one device. Dispose it (the
/// trainer does) to recycle its memory. <c>DataLoader</c> makes them; a custom <see cref="IBatchSource"/> makes
/// its own with the public constructor.
/// </summary>
public sealed class Batch : IDisposable
{
    /// <summary>Creates a batch that owns <paramref name="features"/> and <paramref name="targets"/> (disposing it disposes them).</summary>
    /// <param name="features">The inputs, with the samples along the first dimension.</param>
    /// <param name="targets">The expected outputs, with as many samples along the first dimension.</param>
    /// <param name="index">Zero-based batch number within the epoch.</param>
    public Batch(Tensor features, Tensor targets, int index = 0)
    {
        ArgumentNullException.ThrowIfNull(features);
        ArgumentNullException.ThrowIfNull(targets);
        if (features.Shape.Length == 0 || targets.Shape.Length == 0 || features.Shape[0] != targets.Shape[0])
        {
            throw new ArgumentException($"Features [{string.Join(", ", features.Shape.ToArray())}] and targets [{string.Join(", ", targets.Shape.ToArray())}] need the same number of samples along the first dimension.");
        }

        ArgumentOutOfRangeException.ThrowIfNegative(index);
        Features = features;
        Targets = targets;
        Index = index;
    }

    /// <summary>[Size, ..FeatureShape] inputs.</summary>
    public Tensor Features { get; }

    /// <summary>[Size, ..TargetShape] expected outputs.</summary>
    public Tensor Targets { get; }

    /// <summary>Zero-based batch number within the epoch.</summary>
    public int Index { get; }

    /// <summary>Samples in this batch.</summary>
    public int Size => Features.Shape[0];

    /// <inheritdoc />
    public void Dispose()
    {
        Features.Dispose();
        Targets.Dispose();
    }
}
