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
    /// <summary>The transport's name in <c>Transports</c> ("sim", "tcp", …).</summary>
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
    /// Whether this transport's time is a simulated cluster's virtual clock (the <c>sim</c> transport), whether or not the
    /// clock has moved yet: a link cheaper than the clock's smallest tick leaves <see cref="SimulatedTime"/> at zero on a
    /// simulated run, so callers ask this rather than read the clock. False on a real transport.
    /// </summary>
    bool Simulated => false;

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

    /// <summary>
    /// Each rank's free memory as the ranks joined, in rank order (<see cref="ClusterMemory.Measure"/>: what its device reports
    /// new tensors could take, within its cap), the same on every rank; an entry is null where a device reports none, and
    /// the whole is null when the ranks did not report it (a group made by hand). A pipeline cuts its stages by it
    /// (<see cref="PipelinePlan.For"/>).
    /// </summary>
    public IReadOnlyList<long?>? RankMemory { get; init; }
}

/// <summary>
/// One kind of transport: it says whether it can join two ranks here and connects a rank's end. Picking one by its name
/// is not on the new design yet. Ranks rendezvous at <see cref="ClusterOptions.Coordinator"/>, an address for a network
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
    /// The connection this transport's pairs ride on: transports of one carrier are connected once for a group that
    /// mixes
    /// them (<c>Connect</c>), so "direct-link" pairs and "tcp" pairs share one mesh of sockets.
    /// Its
    /// own name unless it says otherwise.
    /// </summary>
    string Carrier => Name;
}

// A list of names for messages.
internal static class ClusterNames
{
    public static string List(IReadOnlyCollection<string> names) => names.Count == 0 ? "none" : string.Join(", ", names);
}
