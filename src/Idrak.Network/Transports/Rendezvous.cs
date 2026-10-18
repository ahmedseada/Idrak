// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Buffers;
using System.Buffers.Binary;
using System.Collections.Concurrent;
using System.Diagnostics;
using System.Globalization;
using System.Net;
using System.Net.Sockets;
using System.Text;
using Idrak.Networking.Abstractions;

namespace Idrak.Networking.Transports;

/// <summary>
/// How TCP ranks find each other (plan 23, phase C5). Rank 0 listens at the coordinator (<see cref="ClusterOptions.Coordinator"/>,
/// else the topology's first node's address); every other rank listens on a port of its own
/// (<see cref="ClusterOptions.ListenPort"/>, 0: the system picks), connects to rank 0 (retrying
/// <see cref="ClusterOptions.Retries"/> times, the wait doubling from <see cref="ClusterOptions.RetryDelay"/>) and proves
/// it holds the cluster key; rank 0 answers every rank with every rank's address (a welcome), and each rank then connects
/// to the ranks below it and takes the connections of the ranks above, proving the key again on each. The proof is a
/// challenge each way: the side that accepts sends a fresh nonce, the side that connects answers with the HMAC of it
/// (<see cref="WireFormat.SignHello"/>) and a nonce of its own, which the acceptor answers the same way, so a recorded
/// hello is worth nothing. A hello that fails is refused with an error frame naming why (never the key) and nothing from
/// that peer is accepted. Without a key nothing listens beyond loopback.
/// </summary>
internal static class TcpRendezvous
{
    private const int ConnectorHello = WireFormat.NonceBytes + WireFormat.MacBytes + sizeof(int);

    // The port rank 0 got from the system, for ranks of the same process that were given port 0 (by run).
    private static readonly ConcurrentDictionary<string, TaskCompletionSource<int>> Ports = new(StringComparer.Ordinal);

    public static TcpTransport Connect(string name, ClusterContext context, CancellationToken cancellationToken)
    {
        ArgumentNullException.ThrowIfNull(context);
        var options = context.Options;
        var topology = context.Topology;
        int me = context.Rank, world = topology.WorldSize;
        if (world == 1)
        {
            return new TcpTransport(name, 0, 1, [null], options);
        }

        string coordinator = options.Coordinator ?? topology.Nodes[0].Address
            ?? throw new InvalidOperationException("A TCP group needs a coordinator: set IDRAK_COORDINATOR (host:port) or give the topology's first node an address.");
        var (host, port) = Endpoint(coordinator);
        bool loopback = IsLoopback(host);
        if (options.Key is null && !loopback)
        {
            throw new InvalidOperationException($"Without a cluster key (IDRAK_CLUSTER_KEY) nothing listens beyond this machine, and the coordinator {coordinator} is not a loopback address: set the same key of at least {ClusterKey.MinimumLength} characters on every rank.");
        }

        var address = Resolve(host);
        var listenOn = address.AddressFamily == AddressFamily.InterNetworkV6
            ? (loopback ? IPAddress.IPv6Loopback : IPAddress.IPv6Any)
            : (loopback ? IPAddress.Loopback : IPAddress.Any);
        string run = options.RunId ?? coordinator;
        var sockets = new Socket?[world];
        try
        {
            if (me == 0)
            {
                Coordinate(sockets, new IPEndPoint(listenOn, port), coordinator, run, context, cancellationToken);
            }
            else
            {
                Join(sockets, address, port, listenOn, coordinator, run, context, cancellationToken);
            }
        }
        catch
        {
            foreach (var socket in sockets)
            {
                socket?.Dispose();
            }

            throw;
        }

        return new TcpTransport(name, me, world, sockets, options);
    }

    // Rank 0: takes every rank's hello at the coordinator, then sends each the table of addresses.
    private static void Coordinate(Socket?[] sockets, IPEndPoint endpoint, string coordinator, string run, ClusterContext context, CancellationToken cancellationToken)
    {
        var options = context.Options;
        int world = sockets.Length;
        using var listener = Listen(endpoint, world);
        int actual = ((IPEndPoint)listener.LocalEndPoint!).Port;
        if (endpoint.Port == 0)
        {
            Ports.GetOrAdd(run, _ => new TaskCompletionSource<int>(TaskCreationOptions.RunContinuationsAsynchronously)).TrySetResult(actual);
        }

        try
        {
            var ports = new int[world];
            var addresses = new string[world];
            long deadline = Deadline(options.Timeout);
            string? refused = null;
            int joined = 0;
            while (joined < world - 1)
            {
                var socket = Accept(listener, deadline, cancellationToken, () =>
                    $"rank 0 waited {options.Timeout.TotalSeconds:F0} s at {coordinator} for rank(s) {string.Join(", ", Enumerable.Range(1, world - 1).Where(r => sockets[r] is null))} to join"
                    + (refused is null ? "" : $" (refused: {refused})"));
                try
                {
                    Configure(socket, options);
                    var (peer, peerPort) = Greet(socket, 0, world, options.Key, r => r > 0 && r < world && sockets[r] is null);
                    sockets[peer] = socket;
                    ports[peer] = peerPort;
                    addresses[peer] = ((IPEndPoint)socket.RemoteEndPoint!).Address is { IsIPv4MappedToIPv6: true } mapped ? mapped.MapToIPv4().ToString() : ((IPEndPoint)socket.RemoteEndPoint!).Address.ToString();
                    joined++;
                }
                catch (Exception e) when (e is IOException or SocketException or InvalidDataException or HelloRefusedException)
                {
                    refused = e.Message;                             // a stranger, a wrong key or a rank twice: wait for the ranks
                    socket.Dispose();
                }
            }

            // The welcome: each rank's port and address, in rank order.
            int size = world * (sizeof(int) + sizeof(ushort));
            foreach (string a in addresses)
            {
                size += a is null ? 0 : Encoding.ASCII.GetByteCount(a);
            }

            var table = ArrayPool<byte>.Shared.Rent(size);
            try
            {
                int at = 0;
                for (int r = 0; r < world; r++)
                {
                    BinaryPrimitives.WriteInt32LittleEndian(table.AsSpan(at), ports[r]);
                    int length = addresses[r] is { } text ? Encoding.ASCII.GetBytes(text, table.AsSpan(at + 6)) : 0;
                    BinaryPrimitives.WriteUInt16LittleEndian(table.AsSpan(at + 4), (ushort)length);
                    at += 6 + length;
                }

                for (int r = 1; r < world; r++)
                {
                    Write(sockets[r]!, new WireHeader(WireMessageKind.Welcome, 0, 0, 0, 0, (uint)world, WireElement.Bytes, ReduceOp.Sum, 0, (uint)at), table.AsSpan(0, at));
                }
            }
            finally
            {
                ArrayPool<byte>.Shared.Return(table);
            }
        }
        finally
        {
            if (endpoint.Port == 0)
            {
                Ports.TryRemove(run, out _);
            }
        }
    }

    // Rank r > 0: listens for the ranks above, meets rank 0, then connects to the ranks below and takes the ranks above.
    private static void Join(Socket?[] sockets, IPAddress coordinatorAddress, int port, IPAddress listenOn, string coordinator, string run, ClusterContext context,
        CancellationToken cancellationToken)
    {
        var options = context.Options;
        int me = context.Rank, world = sockets.Length;
        if (port == 0)
        {
            var picked = Ports.GetOrAdd(run, _ => new TaskCompletionSource<int>(TaskCreationOptions.RunContinuationsAsynchronously)).Task;
            port = picked.Wait(options.Timeout, cancellationToken)
                ? picked.Result
                : throw new TimeoutException($"rank {me} waited {options.Timeout.TotalSeconds:F0} s for rank 0 of this process to pick the coordinator's port ({coordinator}: port 0 is for ranks of one process; ranks in processes of their own are given a port)");
        }

        using var listener = Listen(new IPEndPoint(listenOn, options.ListenPort), world);
        int mine = ((IPEndPoint)listener.LocalEndPoint!).Port;
        string where = $"the coordinator {Display(coordinatorAddress, port)}";
        var first = Dial(coordinatorAddress, port, me, where, options, cancellationToken);
        sockets[0] = first;
        int zero = Answer(first, me, world, options.Key, mine, where);
        if (zero != 0)
        {
            throw new InvalidDataException($"{where} answered as rank {zero}, not rank 0");
        }

        // The welcome: every rank's port and address.
        var ports = new int[world];
        var addresses = new string[world];
        var buffer = ArrayPool<byte>.Shared.Rent(Math.Max(512, world * 64));
        try
        {
            var (header, length) = ReadFrame(first, buffer, where);
            if (header.Kind != WireMessageKind.Welcome)
            {
                throw new InvalidDataException($"{where} sent a {header.Kind} where its welcome was due");
            }

            int at = 0;
            for (int r = 0; r < world; r++)
            {
                ports[r] = BinaryPrimitives.ReadInt32LittleEndian(buffer.AsSpan(at));
                int size = BinaryPrimitives.ReadUInt16LittleEndian(buffer.AsSpan(at + 4));
                addresses[r] = Encoding.ASCII.GetString(buffer, at + 6, size);
                at += 6 + size;
                if (at > length)
                {
                    throw new InvalidDataException($"{where} sent a welcome shorter than its table");
                }
            }
        }
        finally
        {
            ArrayPool<byte>.Shared.Return(buffer);
        }

        // Down: the ranks below (each listening by now, or soon: the dial retries); a pair on a cable dials the file's address.
        var topology = context.Topology;
        bool cables = !IsLoopback(Endpoint(coordinator).Host);       // every rank on this machine: no cable to take
        for (int j = 1; j < me; j++)
        {
            var host = cables && DirectLinkProvider.IsCable(topology, me, j) && topology.AddressOf(j, me) is { } file
                ? Resolve(HostOf(file))
                : IPAddress.Parse(addresses[j]);
            string peer = $"rank {j} at {Display(host, ports[j])}";
            var socket = Dial(host, ports[j], me, peer, options, cancellationToken);
            sockets[j] = socket;
            int answered = Answer(socket, me, world, options.Key, mine, peer);
            if (answered != j)
            {
                throw new InvalidDataException($"{peer} answered as rank {answered}");
            }
        }

        // Up: the ranks above connect here.
        long deadline = Deadline(options.Timeout);
        string? refused = null;
        for (int count = world - 1 - me; count > 0;)
        {
            var socket = Accept(listener, deadline, cancellationToken, () =>
                $"rank {me} waited {options.Timeout.TotalSeconds:F0} s for rank(s) {string.Join(", ", Enumerable.Range(me + 1, world - me - 1).Where(r => sockets[r] is null))} to connect"
                + (refused is null ? "" : $" (refused: {refused})"));
            try
            {
                Configure(socket, options);
                var (peer, _) = Greet(socket, me, world, options.Key, r => r > me && r < world && sockets[r] is null);
                sockets[peer] = socket;
                count--;
            }
            catch (Exception e) when (e is IOException or SocketException or InvalidDataException or HelloRefusedException)
            {
                refused = e.Message;
                socket.Dispose();
            }
        }
    }

    /// <summary>
    /// The accepting side's half of the hello: a fresh nonce out, the peer's hello in (its rank, the group's size, the HMAC
    /// of the nonce and a nonce of its own), checked; then the HMAC of the peer's nonce back. Returns the peer's rank and
    /// the port it listens on.
    /// </summary>
    internal static (int Rank, int Port) Greet(Socket socket, int me, int world, ClusterKey? key, Func<int, bool> expected)
    {
        Span<byte> nonce = stackalloc byte[WireFormat.NonceBytes];
        WireFormat.NewNonce(nonce);
        Write(socket, Hello(WireMessageKind.Hello, me, world, WireFormat.NonceBytes), nonce);
        var buffer = ArrayPool<byte>.Shared.Rent(256);
        try
        {
            var (hello, length) = ReadFrame(socket, buffer, "a peer");
            if (hello.Kind != WireMessageKind.Hello || length != ConnectorHello)
            {
                throw new InvalidDataException($"a peer sent a {hello.Kind} of {length} bytes where a hello was due");
            }

            int peer = hello.Rank;
            string? refusal = hello.ChunkCount != (uint)world ? $"rank {me} is in a group of {world} ranks; rank {peer} says {hello.ChunkCount}"
                : !expected(peer) ? $"rank {me} does not expect rank {peer} (it joined already, or is outside 0 to {world - 1})"
                : !Proven(key, nonce, peer, world, buffer.AsSpan(WireFormat.NonceBytes, WireFormat.MacBytes))
                    ? $"rank {me} refused the hello of rank {peer}: it does not prove the cluster key (IDRAK_CLUSTER_KEY differs between the two)"
                : null;
            if (refusal is not null)
            {
                Refuse(socket, me, refusal);
                throw new HelloRefusedException(refusal);
            }

            int port = BinaryPrimitives.ReadInt32LittleEndian(buffer.AsSpan(WireFormat.NonceBytes + WireFormat.MacBytes));
            Span<byte> mac = stackalloc byte[WireFormat.MacBytes];
            Sign(key, buffer.AsSpan(0, WireFormat.NonceBytes), me, world, mac);
            Write(socket, Hello(WireMessageKind.Ack, me, world, WireFormat.MacBytes), mac);
            return (peer, port);
        }
        finally
        {
            ArrayPool<byte>.Shared.Return(buffer, clearArray: true);
        }
    }

    /// <summary>The connecting side's half of the hello: answers the acceptor's nonce, checks its answer to ours; returns its rank.</summary>
    internal static int Answer(Socket socket, int me, int world, ClusterKey? key, int listenPort, string where)
    {
        var buffer = ArrayPool<byte>.Shared.Rent(256);
        try
        {
            var (hello, length) = ReadFrame(socket, buffer, where);
            if (hello.Kind != WireMessageKind.Hello || length != WireFormat.NonceBytes)
            {
                throw new InvalidDataException($"{where} sent a {hello.Kind} of {length} bytes where a hello was due");
            }

            int peer = hello.Rank;
            Span<byte> answer = stackalloc byte[ConnectorHello];
            Sign(key, buffer.AsSpan(0, WireFormat.NonceBytes), me, world, answer.Slice(WireFormat.NonceBytes, WireFormat.MacBytes));
            var ours = answer[..WireFormat.NonceBytes];
            WireFormat.NewNonce(ours);
            BinaryPrimitives.WriteInt32LittleEndian(answer[(WireFormat.NonceBytes + WireFormat.MacBytes)..], listenPort);
            Write(socket, Hello(WireMessageKind.Hello, me, world, ConnectorHello), answer);
            var (reply, replyLength) = ReadFrame(socket, buffer, where);
            if (reply.Kind == WireMessageKind.Error)
            {
                throw new HelloRefusedException($"{where} refused rank {me}: {Encoding.UTF8.GetString(buffer, 0, replyLength)}");
            }

            if (reply.Kind != WireMessageKind.Ack || replyLength != WireFormat.MacBytes || !Proven(key, ours, peer, world, buffer.AsSpan(0, WireFormat.MacBytes)))
            {
                throw new HelloRefusedException($"rank {me} refused {where}: its answer does not prove the cluster key (IDRAK_CLUSTER_KEY differs between the two)");
            }

            return peer;
        }
        finally
        {
            ArrayPool<byte>.Shared.Return(buffer, clearArray: true);
        }
    }

    /// <summary>"host:port" (an IPv6 address in brackets) as its parts.</summary>
    /// <exception cref="FormatException">Not host:port with a port from 0 to 65535.</exception>
    public static (string Host, int Port) Endpoint(string text)
    {
        int colon = text.LastIndexOf(':');
        if (colon <= 0 || !int.TryParse(text.AsSpan(colon + 1), NumberStyles.None, CultureInfo.InvariantCulture, out int port) || port > 65535)
        {
            throw new FormatException($"'{text}' is not host:port (IDRAK_COORDINATOR, or a node's address in the topology).");
        }

        string host = text[..colon];
        return (host.StartsWith('[') && host.EndsWith(']') ? host[1..^1] : host, port);
    }

    /// <summary>The host of an address in the topology: "host:port" (an IPv6 address in brackets), or a bare host (a cable's).</summary>
    internal static string HostOf(string address)
    {
        string text = address.Trim();
        if (text.StartsWith('['))
        {
            int close = text.IndexOf(']');
            return close > 1 ? text[1..close] : text;
        }

        int colon = text.LastIndexOf(':');
        return colon > 0 && text.IndexOf(':') == colon ? text[..colon] : text;   // one colon: host:port; more: a bare IPv6 address
    }

    /// <summary>Whether every rank of <paramref name="context"/> is on this machine: its coordinator is a loopback address.</summary>
    public static bool AllOnThisMachine(ClusterContext context)
    {
        string? coordinator = context.Options.Coordinator ?? context.Topology.Nodes[0].Address;
        int colon = coordinator?.LastIndexOf(':') ?? -1;
        return coordinator is not null && colon > 0 && IsLoopback(coordinator[..colon].Trim('[', ']'));
    }

    public static bool IsLoopback(string host) =>
        string.Equals(host, "localhost", StringComparison.OrdinalIgnoreCase) || (IPAddress.TryParse(host, out var ip) && IPAddress.IsLoopback(ip));

    private static IPAddress Resolve(string host)
    {
        if (IPAddress.TryParse(host, out var ip))
        {
            return ip;
        }

        if (string.Equals(host, "localhost", StringComparison.OrdinalIgnoreCase))
        {
            return IPAddress.Loopback;
        }

        var all = Dns.GetHostAddresses(host);
        return all.FirstOrDefault(a => a.AddressFamily == AddressFamily.InterNetwork) ?? all.FirstOrDefault()
            ?? throw new InvalidOperationException($"'{host}' has no address.");
    }

    private static string Display(IPAddress address, int port) => new IPEndPoint(address, port).ToString();

    private static Socket Listen(IPEndPoint endpoint, int world)
    {
        var socket = new Socket(endpoint.AddressFamily, SocketType.Stream, ProtocolType.Tcp);
        try
        {
            if (endpoint.Address.Equals(IPAddress.IPv6Any))
            {
                socket.DualMode = true;
            }

            if (!OperatingSystem.IsWindows())
            {
                socket.SetSocketOption(SocketOptionLevel.Socket, SocketOptionName.ReuseAddress, true);   // a fixed port again soon after a run
            }

            socket.Bind(endpoint);
            socket.Listen(Math.Max(16, world));
            return socket;
        }
        catch (SocketException e)
        {
            socket.Dispose();
            throw new InvalidOperationException($"Cannot listen on {endpoint}: {e.Message} (IDRAK_CLUSTER_PORT, IDRAK_COORDINATOR).", e);
        }
    }

    private static Socket Accept(Socket listener, long deadline, CancellationToken cancellationToken, Func<string> late)
    {
        while (true)
        {
            cancellationToken.ThrowIfCancellationRequested();
            long left = deadline - Stopwatch.GetTimestamp();
            if (left <= 0)
            {
                throw new TimeoutException(late());
            }

            int wait = (int)Math.Clamp(left * 1_000_000 / Stopwatch.Frequency, 1, 100_000);
            if (listener.Poll(wait, SelectMode.SelectRead))
            {
                var socket = listener.Accept();
                socket.ReceiveTimeout = (int)Math.Clamp((deadline - Stopwatch.GetTimestamp()) * 1000 / Stopwatch.Frequency, 1000, int.MaxValue);
                return socket;
            }
        }
    }

    private static Socket Dial(IPAddress address, int port, int me, string where, ClusterOptions options, CancellationToken cancellationToken)
    {
        int attempts = Math.Max(1, options.Retries);
        var delay = options.RetryDelay;
        Exception? last = null;
        for (int attempt = 1; attempt <= attempts; attempt++)
        {
            var socket = new Socket(address.AddressFamily, SocketType.Stream, ProtocolType.Tcp);
            try
            {
                using var timeout = CancellationTokenSource.CreateLinkedTokenSource(cancellationToken);
                timeout.CancelAfter(options.Timeout);
                socket.ConnectAsync(new IPEndPoint(address, port), timeout.Token).AsTask().GetAwaiter().GetResult();
                Configure(socket, options);
                socket.ReceiveTimeout = (int)Math.Min(int.MaxValue, options.Timeout.TotalMilliseconds);
                return socket;
            }
            catch (Exception e) when (e is SocketException or OperationCanceledException && !cancellationToken.IsCancellationRequested)
            {
                socket.Dispose();
                last = e;
                if (attempt < attempts)
                {
                    cancellationToken.WaitHandle.WaitOne(delay);
                    delay += delay;
                }
            }
            catch
            {
                socket.Dispose();
                throw;
            }
        }

        throw new InvalidOperationException($"rank {me} could not reach {where} after {attempts} attempt(s) (IDRAK_CLUSTER_RETRIES, IDRAK_CLUSTER_RETRY_DELAY): {last?.Message}", last);
    }

    private static void Configure(Socket socket, ClusterOptions options)
    {
        socket.NoDelay = true;
        if (options.SocketBufferBytes is { } bytes)
        {
            int size = (int)Math.Min(int.MaxValue, bytes);
            socket.SendBufferSize = size;
            socket.ReceiveBufferSize = size;
        }
    }

    private static long Deadline(TimeSpan timeout) => Stopwatch.GetTimestamp() + (long)(timeout.TotalSeconds * Stopwatch.Frequency);

    private static WireHeader Hello(WireMessageKind kind, int rank, int world, int payload) =>
        new(kind, rank, 0, 0, 0, (uint)world, WireElement.Bytes, ReduceOp.Sum, 0, (uint)payload);

    private static void Write(Socket socket, in WireHeader header, ReadOnlySpan<byte> payload)
    {
        var frame = ArrayPool<byte>.Shared.Rent(WireFormat.HeaderBytes + payload.Length);
        try
        {
            WireFormat.WriteHeader(frame, header);
            payload.CopyTo(frame.AsSpan(WireFormat.HeaderBytes));
            TcpTransport.SendAll(socket, frame.AsSpan(0, WireFormat.HeaderBytes + payload.Length));
        }
        finally
        {
            ArrayPool<byte>.Shared.Return(frame);
        }
    }

    // One frame of the handshake into `buffer`; a timeout or a close names `where`.
    private static (WireHeader Header, int Length) ReadFrame(Socket socket, byte[] buffer, string where)
    {
        Span<byte> head = stackalloc byte[WireFormat.HeaderBytes];
        try
        {
            if (!TcpTransport.ReceiveAll(socket, head) || !WireFormat.TryReadHeader(head, out var header))
            {
                throw new IOException($"{where} closed the connection during the hello");
            }

            if (header.PayloadBytes > buffer.Length)
            {
                throw new InvalidDataException($"{where} sent a {header.PayloadBytes}-byte {header.Kind} during the hello");
            }

            int length = (int)header.PayloadBytes;
            if (length > 0 && !TcpTransport.ReceiveAll(socket, buffer.AsSpan(0, length)))
            {
                throw new IOException($"{where} closed the connection during the hello");
            }

            return (header, length);
        }
        catch (SocketException e) when (e.SocketErrorCode == SocketError.TimedOut)
        {
            throw new IOException($"{where} did not finish its hello in time", e);
        }
    }

    private static void Refuse(Socket socket, int me, string reason)
    {
        try
        {
            var text = Encoding.UTF8.GetBytes(reason);
            Write(socket, new WireHeader(WireMessageKind.Error, me, 0, 0, 0, 1, WireElement.Bytes, ReduceOp.Sum, 0, (uint)text.Length), text);
        }
        catch (Exception e) when (e is SocketException or ObjectDisposedException)
        {
            // the peer is gone already
        }
    }

    // The HMAC under the key; without a key, zeros (and a keyless rank only listens on loopback).
    private static void Sign(ClusterKey? key, ReadOnlySpan<byte> nonce, int rank, int world, Span<byte> mac)
    {
        if (key is null)
        {
            mac.Clear();
            return;
        }

        WireFormat.SignHello(key, nonce, rank, world, mac);
    }

    private static bool Proven(ClusterKey? key, ReadOnlySpan<byte> nonce, int rank, int world, ReadOnlySpan<byte> mac) =>
        key is null ? mac.IndexOfAnyExcept((byte)0) < 0 : WireFormat.VerifyHello(key, nonce, rank, world, mac);
}

/// <summary>A hello that did not prove the cluster key, or came from a rank that was not expected; the message says which, never the key.</summary>
internal sealed class HelloRefusedException(string message) : IOException(message);
