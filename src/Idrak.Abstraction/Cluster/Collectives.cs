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

/// <summary>How values travel between ranks; sums are always made in float32.</summary>
public enum WireType
{
    /// <summary>Four bytes a value, exact.</summary>
    Float32,

    /// <summary>Two bytes a value (bfloat16, rounded to nearest even), half the traffic.</summary>
    BFloat16,
}

/// <summary>
/// The collective operations of one rank over its device's buffers (plan 23). Every rank of the group calls the same
/// operation with the same sizes. Each sum is made in an order fixed by the algorithm and the sizes alone (rule 86 of
/// docs/optimization.md), never by timing, so every rank ends with identical bytes and a rerun gives identical bytes;
/// two algorithms may order a sum differently and so agree within float rounding.
/// </summary>
public interface ICollectives : IDisposable
{
    /// <summary>The algorithm's name in <see cref="Collectives"/> ("naive", "ring", "hierarchical").</summary>
    string Name { get; }

    /// <summary>This rank.</summary>
    int Rank { get; }

    /// <summary>The number of ranks.</summary>
    int WorldSize { get; }

    /// <summary>This rank's device, where its buffers live.</summary>
    Device Device { get; }

    /// <summary>Reduces the first <paramref name="count"/> values of <paramref name="buffer"/> across the ranks, in place.</summary>
    void AllReduce(Storage buffer, int count, ReduceOp op, WireType wire = WireType.Float32);

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

/// <summary>One collective algorithm, registered by name in <see cref="Collectives"/>: it builds a rank's collectives over a connected transport.</summary>
public interface ICollectivesProvider
{
    /// <summary>The algorithm's name ("ring").</summary>
    string Name { get; }

    /// <summary>One line for listings.</summary>
    string Summary { get; }

    /// <summary>The collectives of <paramref name="transport"/>'s rank, over buffers on <paramref name="device"/>.</summary>
    ICollectives Create(ITransport transport, Device device, ClusterContext context);
}

/// <summary>
/// The collective algorithms, by name (plan 23). Idrak.Network registers "naive" (every rank to rank 0 and back, the
/// reference), "ring" and "hierarchical" (inside each node first, then across nodes); none is built into Idrak.Abstraction.
/// </summary>
public static class Collectives
{
    private static readonly SlotTable<string, ICollectivesProvider> Table = new(nameof(Collectives), comparer: StringComparer.OrdinalIgnoreCase,
        unguarded: "a rank's collectives hold staging buffers and their place in the group's message order, so another algorithm cannot take over a call");

    static Collectives() => LibraryDefaults.Ensure(typeof(Collectives));

    /// <summary>The registered algorithms, in registration order.</summary>
    public static IReadOnlyList<ICollectivesProvider> All => Table.Values;

    /// <summary>The registered names.</summary>
    public static IReadOnlyCollection<string> Names => Table.Keys;

    /// <summary>Registers <paramref name="provider"/> under its name (over the library's of that name, which stays behind it).</summary>
    [MethodImpl(MethodImplOptions.NoInlining)]   // the caller is the registering assembly (its Origin)
    public static void Register(ICollectivesProvider provider)
    {
        ArgumentNullException.ThrowIfNull(provider);
        ArgumentException.ThrowIfNullOrWhiteSpace(provider.Name);
        Table.Register(provider.Name, provider, System.Reflection.Assembly.GetCallingAssembly());
    }

    /// <summary>Removes the app's algorithm of that name (the library's, if any, answers again).</summary>
    public static bool Unregister(string name) => Table.Unregister(name);

    /// <summary>The algorithm of that name, or null.</summary>
    public static ICollectivesProvider? Find(string name) => Table.Find(name);

    /// <summary>The algorithm of that name.</summary>
    /// <exception cref="NotSupportedException">None is registered under that name.</exception>
    public static ICollectivesProvider Get(string name) => Table.Find(name)
        ?? throw new NotRegisteredException(nameof(Collectives), name, Table.Keys,
            $"Collective algorithm '{name}' is not registered (registered: {ClusterNames.List(Table.Keys)}); add one with Collectives.Register.");

    /// <summary>The library's algorithm of that name, or null.</summary>
    public static ICollectivesProvider? Default(string name) => Table.Default(name);

    /// <summary>Who registered the algorithm of that name.</summary>
    public static string? Origin(string name) => Table.Origin(name);

    /// <summary>Each algorithm as "name: summary", one a line.</summary>
    public static string Describe() => string.Join("\n", Table.Values.Select(c => $"{c.Name}: {c.Summary}"));
}
