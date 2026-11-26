// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Runtime.CompilerServices;

namespace Idrak.Abstraction.Cluster;

/// <summary>
/// Picks the transport for each pair of ranks from those available between them (plan 23). The choice is published as
/// <c>TransportChosen</c> once per pair; <see cref="ClusterOptions.Transport"/> forces one.
/// </summary>
public interface ITransportSelector
{
    /// <summary></summary>
    string Name { get; }

    /// <summary>One line for listings.</summary>
    string Summary { get; }

    /// <summary>
    /// The transport (its name) for ranks <paramref name="rankA"/> and <paramref name="rankB"/>,
    /// from <paramref name="available"/> (each transport's answer for the pair), with the reason logged.
    /// </summary>
    /// <exception cref="NotSupportedException">None of them can join the pair; the message lists each one's
    /// reason.</exception>
    (string Transport, string Reason) Choose(int rankA, int rankB, IClusterTopology topology, IReadOnlyList<TransportAvailability> available);
}

/// <summary>
/// The transport joining each pair of ranks of a group (plan 23, phase C3), and why, so different pairs may use different
/// pipes (peer copies inside a machine, TCP between machines). Picking them (each transport saying whether it can join
/// the pair here, <see cref="ITransportProvider.Availability"/>, and the selector <see cref="ClusterOptions.Selector"/>
/// names picking among them) is not on the new design yet.
/// </summary>
public sealed class TransportPlan
{
    private readonly (string Transport, string Reason)[] _pairs;

    private TransportPlan(int worldSize, (string, string)[] pairs, IReadOnlyList<string> used)
    {
        WorldSize = worldSize;
        _pairs = pairs;
        Used = used;
    }

    /// <summary>The number of ranks.</summary>
    public int WorldSize { get; }

    /// <summary>The transports the plan uses, in the order of their first pair.</summary>
    public IReadOnlyList<string> Used { get; }

    /// <summary>The one transport every pair uses, or null when pairs use different ones (or there is no pair).</summary>
    public string? Single => Used.Count == 1 ? Used[0] : null;

    /// <summary>The transport joining <paramref name="rankA"/> and <paramref name="rankB"/> (either order), and why.</summary>
    /// <exception cref="ArgumentOutOfRangeException">Not two different ranks of the group.</exception>
    public (string Transport, string Reason) this[int rankA, int rankB] => _pairs[Index(rankA, rankB)];

    private int Index(int rankA, int rankB)
    {
        if (rankA == rankB || (uint)rankA >= (uint)WorldSize || (uint)rankB >= (uint)WorldSize)
        {
            throw new ArgumentOutOfRangeException(nameof(rankB), $"Ranks {rankA} and {rankB} are not two different ranks of 0 to {WorldSize - 1}.");
        }

        return Index(WorldSize, Math.Min(rankA, rankB), Math.Max(rankA, rankB));
    }

    // The pair (a, b), a < b, in row order of the upper triangle.
    private static int Index(int world, int a, int b) => (a * ((2 * world) - a - 1) / 2) + (b - a - 1);
}
