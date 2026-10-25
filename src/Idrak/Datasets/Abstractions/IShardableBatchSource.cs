// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

namespace Idrak.Datasets.Abstractions;

/// <summary>
/// A batch source that a data-parallel run (<c>Trainer.Cluster</c>, plan 23) splits by sample among its ranks instead of
/// by batch: every rank draws an epoch's order as one process draws it and takes its own positions, so batch k of every
/// rank together is batch k of one process with batches W times as large. <c>DataLoader</c> is one. Any other
/// <see cref="IBatchSource"/> is split by batch (rank r takes batches k·W + r).
/// </summary>
public interface IShardableBatchSource : IBatchSource
{
    /// <summary>
    /// Whether every process draws the same order of the samples: the source does not shuffle, or it was given a seed. A
    /// data-parallel run refuses a source that is not, since its ranks would take overlapping shares.
    /// </summary>
    bool Seeded { get; }

    /// <summary>Where the batches' tensors are made.</summary>
    Device Device { get; }

    /// <summary>
    /// Rank <paramref name="rank"/>'s share of each epoch among <paramref name="world"/> ranks: the epoch's order drawn as one
    /// process draws it, and the samples at positions k·world + rank, in batches of <see cref="IBatchSource.BatchSize"/>.
    /// Each enumeration is one epoch; the samples are read by index from the source, which is not copied.
    /// </summary>
    IEnumerable<Batch> Shard(int rank, int world);

    /// <summary>
    /// The source's epochs drawn one step at a time, for a group whose ranks can change from step to step (a late joiner,
    /// a replay's end); null when the source cannot (a stream, whose length is not known ahead), and a run then shards it
    /// by epoch with <see cref="Shard"/>.
    /// </summary>
    IStepShares? StepShares { get; }
}

/// <summary>
/// An epoch drawn one step at a time for a data-parallel group whose ranks can change from step to step
/// (<see cref="IShardableBatchSource.StepShares"/>): each step's samples, drawn as one process with batches world times as
/// large draws them, are shared among the ranks that take part in it.
/// </summary>
public interface IStepShares
{
    /// <summary>
    /// One epoch: <paramref name="place"/> is asked as each step is drawn (this rank's place among the ranks taking part and
    /// their count, or null when it sits the step out), and sample i of a step goes to place i mod count, so with every rank
    /// taking part rank r has the samples <see cref="IShardableBatchSource.Shard"/> gives it, in the same order. Every rank
    /// yields one item a step, the same number on every rank: its batch, or null when it has none.
    /// </summary>
    /// <param name="world">The ranks of the whole group (a step's samples are world · batch size).</param>
    /// <param name="place">This rank's place and the count of the ranks taking part in the step, or null to sit it out.</param>
    IEnumerable<Batch?> Draw(int world, Func<(int Place, int Count)?> place);
}
