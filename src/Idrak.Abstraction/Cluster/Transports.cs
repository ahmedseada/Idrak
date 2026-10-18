// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Runtime.CompilerServices;

namespace Idrak.Abstraction.Cluster;

/// <summary>
/// What a transport says about itself, so the selector and the collectives can plan around it (plan 23). The figures are
/// the transport's own estimate until a bench or a calibration measured the link; nothing here is a card's figure.
/// </summary>
/// <param name="BandwidthBytesPerSecond">Bytes a second one direction carries, as estimated or measured; 0 when unknown.</param>
/// <param name="Latency">The time a message of no bytes takes.</param>
/// <param name="ZeroCopy">Whether the bytes go from the caller's buffer without a copy of their own.</param>
/// <param name="GpuDirect">Whether device memory is sent without a round trip through the host (peer copies).</param>
/// <param name="Reliable">Whether delivery is guaranteed and in order (false for a datagram transport).</param>
/// <param name="CrossProcess">Whether its ranks may be different processes.</param>
/// <param name="CrossMachine">Whether its ranks may be different machines.</param>
public readonly record struct TransportCapabilities(double BandwidthBytesPerSecond, TimeSpan Latency, bool ZeroCopy, bool GpuDirect,
    bool Reliable, bool CrossProcess, bool CrossMachine);

/// <summary>
/// Bytes between the ranks of a cluster, point to point (plan 23): one rank's end of a connected group. Messages from one
/// rank with one tag arrive in the order they were sent. The core is synchronous over spans (rules 7-8 of
/// docs/optimization.md); the async methods only await the carrier.
/// </summary>
public interface ITransport : IDisposable
{
    /// <summary>The transport's name in <see cref="Transports"/> ("sim", "tcp", …).</summary>
    string Name { get; }

    /// <summary>This end's rank, from 0.</summary>
    int Rank { get; }

    /// <summary>The number of ranks in the group.</summary>
    int WorldSize { get; }

    /// <summary>What the transport says about itself.</summary>
    TransportCapabilities Capabilities { get; }

    /// <summary>
    /// Sends <paramref name="bytes"/> to rank <paramref name="to"/> under <paramref name="tag"/>. The span is not held after
    /// the call returns (it is copied or already on its way).
    /// </summary>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="to"/> is not another rank of the group.</exception>
    void Send(int to, ReadOnlySpan<byte> bytes, int tag);

    /// <summary>
    /// Receives the next message rank <paramref name="from"/> sent under <paramref name="tag"/> into
    /// <paramref name="bytes"/> and returns its length, waiting up to <see cref="ClusterOptions.Timeout"/>.
    /// </summary>
    /// <exception cref="TimeoutException">Nothing arrived in time; the message names the rank and the tag.</exception>
    /// <exception cref="InvalidDataException">The message is longer than <paramref name="bytes"/>.</exception>
    int Receive(int from, Span<byte> bytes, int tag);

    /// <summary><see cref="Send"/> for a caller that awaits.</summary>
    ValueTask SendAsync(int to, ReadOnlyMemory<byte> bytes, int tag, CancellationToken cancellationToken = default);

    /// <summary><see cref="Receive"/> for a caller that awaits.</summary>
    ValueTask<int> ReceiveAsync(int from, Memory<byte> bytes, int tag, CancellationToken cancellationToken = default);

    /// <summary>Returns once every rank of the group has called it.</summary>
    void Barrier();

    /// <summary>
    /// This rank's time on the virtual clock of a simulated transport (the network time the simulated cluster would have
    /// spent so far); zero on a real transport. A collective's simulated time is the difference across it.
    /// </summary>
    TimeSpan SimulatedTime => TimeSpan.Zero;

    /// <summary>
    /// On a simulated transport, the rank (in this group's numbering) whose delay this rank's simulated clock last waited
    /// for: the origin of the critical path that reached it (a slow worker, say). Null on a real transport, or when the
    /// rank that held this one back is not in the group.
    /// </summary>
    int? SimulatedCriticalRank => null;
}

/// <summary>Whether a transport can join two ranks here, and why not when it cannot.</summary>
/// <param name="Transport">The transport's name.</param>
/// <param name="Available">Whether it can join them.</param>
/// <param name="Reason">Why (or why not), for the selector's log and <c>idrak transports</c>.</param>
public readonly record struct TransportAvailability(string Transport, bool Available, string Reason);

/// <summary>Everything one rank is started with: the topology it belongs to, its rank, and the options.</summary>
/// <param name="Topology">The cluster.</param>
/// <param name="Rank">This rank.</param>
/// <param name="Options">The cluster options.</param>
public sealed record ClusterContext(IClusterTopology Topology, int Rank, ClusterOptions Options)
{
    /// <summary>The node this rank is on.</summary>
    public ClusterNodeSpec Node => Topology.Nodes[Topology.Place(Rank).Node];
}

/// <summary>
/// One kind of transport, registered by name in <see cref="Transports"/>: it says whether it can join two ranks here and
/// connects a rank's end. Ranks rendezvous at <see cref="ClusterOptions.Coordinator"/>, an address for a network
/// transport, a name for one inside a process.
/// </summary>
public interface ITransportProvider
{
    /// <summary>The transport's name ("tcp").</summary>
    string Name { get; }

    /// <summary>One line for listings.</summary>
    string Summary { get; }

    /// <summary>The transport's own estimate of itself, before any measurement.</summary>
    TransportCapabilities Capabilities { get; }

    /// <summary>Whether it can join <paramref name="context"/>'s rank and <paramref name="peer"/> on this machine.</summary>
    TransportAvailability Availability(ClusterContext context, int peer);

    /// <summary>Connects <paramref name="context"/>'s rank to the group, waiting for the others up to the timeout.</summary>
    ITransport Connect(ClusterContext context, CancellationToken cancellationToken = default);

    /// <summary>
    /// The connection this transport's pairs ride on: transports of one carrier are connected once for a group that mixes
    /// them (<see cref="TransportPlan.Connect"/>), so "direct-link" pairs and "tcp" pairs share one mesh of sockets. Its
    /// own name unless it says otherwise.
    /// </summary>
    string Carrier => Name;
}

/// <summary>
/// The transports, by name (plan 23). None is built into Idrak.Abstraction: Idrak.Cluster registers "sim" (ranks inside
/// one process, charged on a virtual clock) and Idrak.Network the real ones ("shared-memory", "pcie-host", "nvlink",
/// "direct-link", "tcp"); "udp" and "rdma" are names a plug-in may fill.
/// </summary>
public static class Transports
{
    private static readonly SlotTable<string, ITransportProvider> Table = new(nameof(Transports), comparer: StringComparer.OrdinalIgnoreCase,
        unguarded: "a transport holds connections and each rank's state, and a group never falls back to another transport mid-run");

    static Transports() => LibraryDefaults.Ensure(typeof(Transports));

    /// <summary>The registered providers, in registration order.</summary>
    public static IReadOnlyList<ITransportProvider> All => Table.Values;

    /// <summary>The registered names.</summary>
    public static IReadOnlyCollection<string> Names => Table.Keys;

    /// <summary>Registers <paramref name="provider"/> under its name (over the library's of that name, which stays behind it).</summary>
    [MethodImpl(MethodImplOptions.NoInlining)]   // the caller is the registering assembly (its Origin)
    public static void Register(ITransportProvider provider)
    {
        ArgumentNullException.ThrowIfNull(provider);
        ArgumentException.ThrowIfNullOrWhiteSpace(provider.Name);
        Table.Register(provider.Name, provider, System.Reflection.Assembly.GetCallingAssembly());
    }

    /// <summary>Removes the app's transport of that name (the library's, if any, answers again).</summary>
    public static bool Unregister(string name) => Table.Unregister(name);

    /// <summary>The transport of that name, or null.</summary>
    public static ITransportProvider? Find(string name) => Table.Find(name);

    /// <summary>The transport of that name.</summary>
    /// <exception cref="NotSupportedException">None is registered under that name.</exception>
    public static ITransportProvider Get(string name) => Table.Find(name)
        ?? throw new NotSupportedException($"Transport '{name}' is not registered (registered: {ClusterNames.List(Table.Keys)}); add one with Transports.Register.");

    /// <summary>The library's transport of that name, or null.</summary>
    public static ITransportProvider? Default(string name) => Table.Default(name);

    /// <summary>Who registered the transport of that name.</summary>
    public static string? Origin(string name) => Table.Origin(name);

    /// <summary>Each transport as "name: summary", one a line.</summary>
    public static string Describe() => string.Join("\n", Table.Values.Select(t => $"{t.Name}: {t.Summary}"));
}

// The registered names for "not registered" messages.
internal static class ClusterNames
{
    public static string List(IReadOnlyCollection<string> names) => names.Count == 0 ? "none" : string.Join(", ", names);
}
