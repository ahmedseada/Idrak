// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Buffers;
using System.Diagnostics;
using System.Net.Sockets;
using Idrak.Networking.Abstractions;

namespace Idrak.Networking.Transports;

/// <summary>
/// "tcp" (plan 23): one socket per pair of ranks, made at the rendezvous (<see cref="TcpRendezvous"/>: every rank meets rank
/// 0 at the coordinator and proves the cluster key, then the ranks connect to each other), Nagle off, the socket buffers
/// as <see cref="ClusterOptions.SocketBufferBytes"/> sets them. A reader thread per socket reads whole frames into pooled
/// buffers (rules 7-14: the header parsed from the bytes with <see cref="WireFormat.TryReadHeader"/>), so every rank keeps
/// draining while it sends. "direct-link" pairs are the same sockets, dialled at the address the topology file gives.
/// </summary>
internal sealed class TcpTransport : FramedTransport
{
    // A frame whose payload is at most this is written in one call (the header and the payload copied together): fewer
    // calls for the many small messages of a collective; a larger payload goes straight from the caller's span.
    private const int Gathered = 64 << 10;
    private readonly Socket?[] _sockets;
    private readonly Lock[] _writing;
    private readonly Thread[] _readers;

    public TcpTransport(string name, int rank, int worldSize, Socket?[] sockets, ClusterOptions options) : base(name, rank, worldSize, options)
    {
        _sockets = sockets;
        _writing = [.. Enumerable.Range(0, worldSize).Select(_ => new Lock())];
        _readers = new Thread[worldSize];
        for (int r = 0; r < worldSize; r++)
        {
            if (r == rank || sockets[r] is not { } socket)
            {
                continue;
            }

            int peer = r;
            socket.SendTimeout = (int)Math.Min(int.MaxValue, options.Timeout.TotalMilliseconds);
            socket.ReceiveTimeout = 0;                                    // a reader waits as long as the group lives
            _readers[r] = new Thread(() => ReadLoop(peer, socket)) { IsBackground = true, Name = $"{name} rank {rank} from {peer}" };
            _readers[r].Start();
        }
    }

    public override TransportCapabilities Capabilities => TcpProvider.Estimate;

    protected override void WriteFrame(int to, in WireHeader header, ReadOnlySpan<byte> payload)
    {
        var socket = _sockets[to] ?? throw new InvalidOperationException($"rank {Rank} has no connection to rank {to}");
        lock (_writing[to])
        {
            try
            {
                if (payload.Length <= Gathered)
                {
                    var frame = ArrayPool<byte>.Shared.Rent(WireFormat.HeaderBytes + payload.Length);
                    try
                    {
                        WireFormat.WriteHeader(frame, header);
                        payload.CopyTo(frame.AsSpan(WireFormat.HeaderBytes));
                        SendAll(socket, frame.AsSpan(0, WireFormat.HeaderBytes + payload.Length));
                    }
                    finally
                    {
                        ArrayPool<byte>.Shared.Return(frame);
                    }
                }
                else
                {
                    Span<byte> head = stackalloc byte[WireFormat.HeaderBytes];
                    WireFormat.WriteHeader(head, header);
                    SendAll(socket, head);
                    SendAll(socket, payload);
                }
            }
            catch (SocketException e) when (e.SocketErrorCode == SocketError.TimedOut || e.SocketErrorCode == SocketError.WouldBlock)
            {
                throw new TimeoutException($"rank {Rank} waited {Timeout.TotalSeconds:F0} s to send to rank {to}: it does not take its messages");
            }
            catch (Exception e) when (e is SocketException or ObjectDisposedException)
            {
                throw new InvalidOperationException($"rank {Rank} cannot send to rank {to}: {GoneReason(to) ?? e.Message}", e);
            }
        }
    }

    protected override void CloseCarriers()
    {
        foreach (var socket in _sockets)
        {
            if (socket is null)
            {
                continue;
            }

            try
            {
                socket.Shutdown(SocketShutdown.Both);
            }
            catch (Exception e) when (e is SocketException or ObjectDisposedException)
            {
                // closed already
            }

            socket.Dispose();
        }

        foreach (var reader in _readers)
        {
            reader?.Join(TimeSpan.FromSeconds(5));
        }
    }

    internal static void SendAll(Socket socket, ReadOnlySpan<byte> bytes)
    {
        while (!bytes.IsEmpty)
        {
            int sent = socket.Send(bytes, SocketFlags.None);
            bytes = bytes[sent..];
        }
    }

    // Fills `into`; false when the peer closed before its first byte (a clean end between frames).
    internal static bool ReceiveAll(Socket socket, Span<byte> into)
    {
        int at = 0;
        while (at < into.Length)
        {
            int read = socket.Receive(into[at..], SocketFlags.None);
            if (read == 0)
            {
                if (at == 0)
                {
                    return false;
                }

                throw new IOException($"the connection closed {at} bytes into a {into.Length}-byte piece");
            }

            at += read;
        }

        return true;
    }

    // The reader of one peer's socket: whole frames, each into a pooled buffer the queue owns from then on.
    private void ReadLoop(int peer, Socket socket)
    {
        var head = new byte[WireFormat.HeaderBytes];
        try
        {
            while (!Disposed)
            {
                if (!ReceiveAll(socket, head))
                {
                    PeerGone(peer, $"rank {peer}'s connection closed without a goodbye (its process ended?)");
                    return;
                }

                if (!WireFormat.TryReadHeader(head, out var header))
                {
                    continue;
                }

                if (header.Rank != peer)
                {
                    throw new InvalidDataException($"a frame on rank {peer}'s connection says it is from rank {header.Rank}");
                }

                long start = Telemetry.IsEnabled(TelemetryLevel.Network) ? Stopwatch.GetTimestamp() : 0;
                int length = checked((int)header.PayloadBytes);
                var buffer = ArrayPool<byte>.Shared.Rent(Math.Max(1, length));
                try
                {
                    if (length > 0 && !ReceiveAll(socket, buffer.AsSpan(0, length)))
                    {
                        throw new IOException("the connection closed before a frame's payload");
                    }
                }
                catch
                {
                    ArrayPool<byte>.Shared.Return(buffer);
                    throw;
                }

                Arrived(peer, header, buffer, length, Since(start));
                if (header.Kind == WireMessageKind.Bye)
                {
                    return;
                }
            }
        }
        catch (Exception e) when (e is SocketException or ObjectDisposedException or IOException or InvalidDataException or OverflowException)
        {
            if (!Disposed)
            {
                PeerGone(peer, $"rank {peer}'s connection failed: {e.Message}");
            }
        }
    }
}

/// <summary>"tcp" (plan 23): any machines; rendezvous at the coordinator, the keyed hello, a socket per pair.</summary>
internal sealed class TcpProvider : ITransportProvider
{
    public static readonly TransportCapabilities Estimate = new(0, TimeSpan.Zero, ZeroCopy: false, GpuDirect: false, Reliable: true, CrossProcess: true, CrossMachine: true);

    public string Name => "tcp";

    public string Summary => "any machines: rendezvous at the coordinator, an HMAC hello under the cluster key, a socket per pair of ranks";

    public TransportCapabilities Capabilities => Estimate;

    public string Carrier => "tcp";

    public TransportAvailability Availability(ClusterContext context, int peer)
    {
        var topology = context.Topology;
        bool local = TcpRendezvous.AllOnThisMachine(context) || topology.Place(context.Rank).Node == topology.Place(peer).Node;
        if (!local && context.Options.Key is null)
        {
            return new(Name, false, "no cluster key (IDRAK_CLUSTER_KEY): nothing listens beyond this machine without one");
        }

        return new(Name, true, local ? "a socket over loopback or this machine's network" : "a socket over the network, the hello proven with the cluster key");
    }

    public ITransport Connect(ClusterContext context, CancellationToken cancellationToken = default) => TcpRendezvous.Connect(Name, context, cancellationToken);
}

/// <summary>
/// "direct-link" (plan 23): two machines joined by a cable (crossover Ethernet, Thunderbolt or USB4 networking), the pair's
/// link an override in the topology file; the socket goes straight to the peer's address on the cable (the override's
/// "address" for that end when the cable has a subnet of its own, else the peer node's address in the file). It rides on
/// the "tcp" carrier: a group mixing the two connects one mesh of sockets.
/// </summary>
internal sealed class DirectLinkProvider : ITransportProvider
{
    private static readonly string[] Cables = ["direct-link", "thunderbolt", "usb4"];

    public string Name => "direct-link";

    public string Summary => "two machines on one cable (a topology override of direct-link, thunderbolt or usb4): a socket straight to the peer's address";

    public TransportCapabilities Capabilities => TcpProvider.Estimate;

    public string Carrier => "tcp";

    /// <summary>Whether ranks <paramref name="a"/> and <paramref name="b"/> are joined by a cable in <paramref name="topology"/>.</summary>
    public static bool IsCable(IClusterTopology topology, int a, int b)
    {
        int nodeA = topology.Place(a).Node, nodeB = topology.Place(b).Node;
        if (nodeA == nodeB)
        {
            return false;
        }

        string link = topology.Link(a, b).Name;
        return Cables.Any(c => link.StartsWith(c, StringComparison.OrdinalIgnoreCase));
    }

    public TransportAvailability Availability(ClusterContext context, int peer)
    {
        var topology = context.Topology;
        int me = context.Rank;
        if (!IsCable(topology, me, peer))
        {
            return new(Name, false, $"ranks {me} and {peer} are not joined by a cable in the topology (an override of direct-link, thunderbolt or usb4)");
        }

        if (TcpRendezvous.AllOnThisMachine(context))
        {
            return new(Name, false, "every rank is on this machine (the coordinator is on loopback): no cable to take; tcp carries the pair, paced to the link's profile when throttled");
        }

        var node = topology.Nodes[topology.Place(peer).Node];
        if (topology.AddressOf(peer, me) is not { } address)
        {
            return new(Name, false, $"node {node.Name} has no address in the topology to reach it over the cable (its own, or the override's \"address\")");
        }

        return context.Options.Key is null
            ? new(Name, false, "no cluster key (IDRAK_CLUSTER_KEY): nothing listens beyond this machine without one")
            : new(Name, true, $"a cable to {node.Name} at {TcpRendezvous.HostOf(address)}");
    }

    public ITransport Connect(ClusterContext context, CancellationToken cancellationToken = default) => TcpRendezvous.Connect(Name, context, cancellationToken);
}
