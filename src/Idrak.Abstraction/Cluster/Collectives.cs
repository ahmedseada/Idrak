// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Runtime.CompilerServices;
using Idrak.Abstraction.Devices;

namespace Idrak.Abstraction.Cluster;

/// <summary>How a reduction combines the ranks' values.</summary>
public enum ReduceOp
{
    /// <summary>The sum, made in float32 in an order the algorithm fixes by the sizes alone.</summary>
    Sum,

    /// <summary>The largest value.</summary>
    Max,

    /// <summary>The smallest value.</summary>
    Min,
}

/// <summary>
/// The collective operations of one rank over its device's buffers (plan 23). Every rank of the group calls the same
/// operation with the same sizes. Each sum is made in an order fixed by the algorithm and the sizes alone (rule 86 of
/// docs/optimization.md), never by timing, so every rank ends with identical bytes and a rerun gives identical bytes;
/// two algorithms may order a sum differently and so agree within float rounding.
/// </summary>
public interface ICollectives : IDisposable
{
    /// <summary>The algorithm's name in <c>Collectives</c> ("naive", "ring", "hierarchical").</summary>
    string Name { get; }

    /// <summary>This rank.</summary>
    int Rank { get; }

    /// <summary>The number of ranks.</summary>
    int WorldSize { get; }

    /// <summary>This rank's device, where its buffers live.</summary>
    Device Device { get; }

    /// <summary>Reduces the first <paramref name="count"/> values of <paramref name="buffer"/> across the ranks, in place.</summary>
    void AllReduce(Storage buffer, int count, ReduceOp op);

    /// <summary>Gathers every rank's first <paramref name="count"/> values of <paramref name="input"/> into <paramref name="output"/>, in rank order (world · count values).</summary>
    void AllGather(Storage input, int count, Storage output);

    /// <summary>
    /// Reduces <paramref name="input"/> (world · count values) across the ranks and leaves this rank's slice, values
    /// [rank · count, rank · count + count), in the first <paramref name="count"/> values of <paramref name="output"/>.
    /// </summary>
    void ReduceScatter(Storage input, int count, Storage output, ReduceOp op);

    /// <summary>Copies rank <paramref name="root"/>'s first <paramref name="count"/> values of <paramref name="buffer"/> to every rank's.</summary>
    void Broadcast(Storage buffer, int count, int root);

    /// <summary>Returns once every rank has called it.</summary>
    void Barrier();

    /// <summary>
    /// The transport the collectives run over, for point-to-point messages of one's own (a pipeline's activations, plan 23
    /// phase C7) on tags 0 and up (collectives use negative ones), or null when the collectives do not offer it.
    /// </summary>
    ITransport? Transport => null;

    /// <summary>The topology, rank and options the collectives were made for, or null when they do not say.</summary>
    ClusterContext? Context => null;
}

/// <summary>One collective algorithm: it builds a rank's collectives over a connected transport. Picking one by its name
/// is not on the new design yet.</summary>
public interface ICollectivesProvider
{
    /// <summary>The algorithm's name ("ring").</summary>
    string Name { get; }

    /// <summary>One line for listings.</summary>
    string Summary { get; }

    /// <summary>The collectives of <paramref name="transport"/>'s rank, over buffers on <paramref name="device"/>.</summary>
    ICollectives Create(ITransport transport, Device device, ClusterContext context);
}

