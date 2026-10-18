// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

namespace Idrak.Abstraction.Cluster;

/// <summary>
/// Collectives whose group of ranks can change from one training step to the next (plan 23, phase C4): a simulated
/// cluster's late joiner sits out its first steps while the others run them as a group of their own, and every rank sits
/// out the steps after the one a replay stops at. Idrak.Cluster's in-process ranks implement it; the gradient
/// synchronisations of Idrak.Network take part through <see cref="IGradientSync.BeginStep"/>, so a step's gradients are
/// summed over the ranks that train it, weighted by their rows, and a rank that joins takes the trained state first.
/// </summary>
public interface IStepGroup
{
    /// <summary>The step this rank last began, or -1 before the first.</summary>
    long Step { get; }

    /// <summary>
    /// Begins <paramref name="step"/> on this rank (steps go forward): false when this rank sits it out (it runs none of the
    /// step's collectives); while it does, the collectives of the others run over the ranks that take part.
    /// </summary>
    /// <exception cref="ArgumentOutOfRangeException">A step not after the last one begun.</exception>
    bool BeginStep(long step);

    /// <summary>Whether this rank takes part in the step it last began (true before the first).</summary>
    bool Present { get; }

    /// <summary>Whether the ranks taking part in the step last begun differ from those of the step before.</summary>
    bool Changed { get; }

    /// <summary>
    /// When <see cref="Changed"/>: the rank, in the current group's numbering, that took part in the step before as well and
    /// so holds the trained state (the weights, the optimizer's state) every rank of the group takes; else 0.
    /// </summary>
    int Holder { get; }
}

/// <summary>
/// Collectives that also reduce values the caller holds on the host (plan 23, phase C4): a gradient synchronisation's
/// buckets travel straight from its host buffer, from the thread that reduces them while the backward pass continues,
/// with no device copy. Each method returns false when it cannot (collectives of a plug-in behind a simulated rank), and
/// the caller goes through the device instead. The sums are those of <see cref="ICollectives.AllReduce"/>: the same
/// algorithm, chunks and order (rule 86 of docs/optimization.md).
/// </summary>
public interface IHostValueCollectives
{
    /// <summary>Whether the methods below work here (they return false otherwise); the same for the whole run.</summary>
    bool SupportsHostValues { get; }

    /// <summary>This rank's network time on a simulated transport's virtual clock; zero on a real transport.</summary>
    TimeSpan SimulatedTime { get; }

    /// <summary>Whether the transport under these collectives is a simulated one (<see cref="ITransport.Simulated"/>).</summary>
    bool Simulated => false;

    /// <summary>Reduces <paramref name="values"/> across the ranks in place, as <see cref="ICollectives.AllReduce"/> does.</summary>
    bool TryAllReduce(Span<float> values, ReduceOp op, WireType wire);

    /// <summary><paramref name="all"/> holds this rank's <paramref name="count"/> values at Rank · count; every rank's afterwards, in rank order.</summary>
    bool TryAllGather(Span<float> all, int count);

    /// <summary>Copies rank <paramref name="root"/>'s <paramref name="values"/> to every rank's.</summary>
    bool TryBroadcast(Span<float> values, int root);

    /// <summary>
    /// Reduces <paramref name="values"/> across the ranks so that this rank's piece holds the reduction (plan 23, phase C6:
    /// a sharded run's gradient bucket, each rank's piece the values of the parameters it owns). The pieces are
    /// <paramref name="counts"/> long, one per rank in rank order, contiguous from the start; the values outside this
    /// rank's piece are left in an unspecified state. Each sum is made in an order fixed by the algorithm and the sizes, as
    /// <see cref="ICollectives.ReduceScatter"/>'s. False (the default) where the collectives cannot.
    /// </summary>
    bool TryReduceScatter(Span<float> values, ReadOnlySpan<int> counts, ReduceOp op, WireType wire) => false;

    /// <summary>
    /// <paramref name="values"/> holds this rank's piece (pieces of <paramref name="counts"/> values, one per rank in rank
    /// order, contiguous from the start); every rank's afterwards (plan 23, phase C6: a sharded run's weights). False (the
    /// default) where the collectives cannot.
    /// </summary>
    bool TryAllGather(Span<float> values, ReadOnlySpan<int> counts) => false;
}
