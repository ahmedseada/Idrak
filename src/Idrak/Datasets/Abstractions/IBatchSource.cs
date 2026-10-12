// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

namespace Idrak.Datasets.Abstractions;

/// <summary>
/// Mini-batches for <c>Trainer.Fit</c>: each enumeration is one epoch. <c>DataLoader</c> is
/// one; write your own to batch in another way (variable lengths, sampling by class, batches made on the device). The
/// trainer disposes every batch after its step. The counts are for telemetry and progress; leave them null when unknown.
/// </summary>
public interface IBatchSource : IEnumerable<Batch>
{
    /// <summary>Batches per epoch, or null when unknown.</summary>
    int? BatchCount => null;

    /// <summary>Samples per epoch, or null when unknown.</summary>
    int? SampleCount => null;

    /// <summary>Samples per batch (the last may be smaller), or null when it varies.</summary>
    int? BatchSize => null;
}
