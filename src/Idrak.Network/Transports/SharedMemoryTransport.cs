// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Buffers;
using System.Diagnostics;
using System.IO.MemoryMappedFiles;
using System.Runtime.CompilerServices;
using System.Security.Cryptography;
using System.Text;
using Idrak.Networking.Abstractions;

namespace Idrak.Networking.Transports;

/// <summary>
/// One direction between two ranks of one machine: a file mapped into both processes (a ring of
/// <see cref="ClusterOptions.SharedMemoryBytes"/> after a 128-byte header). The writer copies a frame in and moves its
/// position; the reader copies out and moves its own; each waits for the other only when the ring is full or empty. The
/// two positions only grow and sit on cache lines of their own; a position is written after the bytes it covers
/// (<see cref="Volatile.Write(ref long, long)"/>), so the other side never reads bytes not yet copied. Files, not named
/// maps or semaphores, so it works alike on every system .NET runs on (rule 70).
/// </summary>
internal sealed unsafe class SharedRing : IDisposable
{
    public const int HeaderBytes = 128;
    private const int WriteAt = 0, ReadAt = 64, WriterPidAt = 72, ReaderPidAt = 76, StateAt = 80, CapacityAt = 88;
    private const int Ready = 1, WriterClosed = 2;
    private readonly MemoryMappedFile _file;
    private readonly MemoryMappedViewAccessor _view;
    private readonly byte* _base;
    private bool _disposed;

    private SharedRing(string path, MemoryMappedFile file, MemoryMappedViewAccessor view, long capacity)
    {
        Path = path;
        _file = file;
        _view = view;
        byte* pointer = null;
        view.SafeMemoryMappedViewHandle.AcquirePointer(ref pointer);
        _base = pointer + view.PointerOffset;
        Capacity = capacity;
    }

    public string Path { get; }

    public long Capacity { get; }

    private byte* Data => _base + HeaderBytes;

    private ref long WritePosition => ref Unsafe.AsRef<long>(_base + WriteAt);

    private ref long ReadPosition => ref Unsafe.AsRef<long>(_base + ReadAt);

    private ref int State => ref Unsafe.AsRef<int>(_base + StateAt);

    public int WriterProcess => Volatile.Read(ref Unsafe.AsRef<int>(_base + WriterPidAt));

    public int ReaderProcess => Volatile.Read(ref Unsafe.AsRef<int>(_base + ReaderPidAt));

    public bool WriterHasClosed => (Volatile.Read(ref State) & WriterClosed) != 0;

    /// <summary>The receiver makes its ring afresh (an old file of that name is removed first) and marks it ready last.</summary>
    public static SharedRing Create(string path, long capacity)
    {
        try
        {
            File.Delete(path);
        }
        catch (IOException)
        {
            // still mapped by a process of an earlier run: the new file below fails and says so
        }

        var stream = new FileStream(path, FileMode.CreateNew, FileAccess.ReadWrite, FileShare.ReadWrite | FileShare.Delete);
        try
        {
            stream.SetLength(HeaderBytes + capacity);
            var file = MemoryMappedFile.CreateFromFile(stream, null, HeaderBytes + capacity, MemoryMappedFileAccess.ReadWrite, HandleInheritability.None, leaveOpen: false);
            var ring = new SharedRing(path, file, file.CreateViewAccessor(0, HeaderBytes + capacity), capacity);
            Unsafe.AsRef<long>(ring._base + CapacityAt) = capacity;
            Unsafe.AsRef<int>(ring._base + ReaderPidAt) = Environment.ProcessId;
            Volatile.Write(ref ring.State, Ready);
            return ring;
        }
        catch
        {
            stream.Dispose();
            throw;
        }
    }

    /// <summary>The sender's end, or null while the receiver has not made the ring yet.</summary>
    public static SharedRing? TryOpen(string path)
    {
        FileStream stream;
        try
        {
            stream = new FileStream(path, FileMode.Open, FileAccess.ReadWrite, FileShare.ReadWrite | FileShare.Delete);
        }
        catch (Exception e) when (e is FileNotFoundException or DirectoryNotFoundException or IOException or UnauthorizedAccessException)
        {
            return null;
        }

        try
        {
            if (stream.Length < HeaderBytes)
            {
                stream.Dispose();
                return null;
            }

            var file = MemoryMappedFile.CreateFromFile(stream, null, stream.Length, MemoryMappedFileAccess.ReadWrite, HandleInheritability.None, leaveOpen: false);
            var view = file.CreateViewAccessor(0, stream.Length);
            var ring = new SharedRing(path, file, view, stream.Length - HeaderBytes);
            if ((Volatile.Read(ref ring.State) & Ready) == 0 || Unsafe.AsRef<long>(ring._base + CapacityAt) != ring.Capacity)
            {
                ring.Dispose();
                return null;
            }

            Unsafe.AsRef<int>(ring._base + WriterPidAt) = Environment.ProcessId;
            return ring;
        }
        catch
        {
            stream.Dispose();
            throw;
        }
    }

    /// <summary>
    /// Copies <paramref name="bytes"/> in as room frees, publishing each piece; <paramref name="wait"/> is called while the
    /// ring is full and throws to give up.
    /// </summary>
    public void Write(ReadOnlySpan<byte> bytes, ref long written, Action wait)
    {
        long capacity = Capacity;
        while (!bytes.IsEmpty)
        {
            long free = capacity - (written - Volatile.Read(ref ReadPosition));
            if (free <= 0)
            {
                wait();
                continue;
            }

            int at = (int)(written % capacity);
            int n = (int)Math.Min(Math.Min(free, bytes.Length), capacity - at);
            bytes[..n].CopyTo(new Span<byte>(Data + at, n));
            bytes = bytes[n..];
            written += n;
            Volatile.Write(ref WritePosition, written);
        }
    }

    /// <summary>Copies what has arrived (at most <paramref name="into"/>.Length bytes) and frees it; returns the bytes copied.</summary>
    public int Read(Span<byte> into, ref long read)
    {
        long available = Volatile.Read(ref WritePosition) - read;
        if (available <= 0 || into.IsEmpty)
        {
            return 0;
        }

        long capacity = Capacity;
        int total = 0;
        while (available > 0 && !into.IsEmpty)
        {
            int at = (int)(read % capacity);
            int n = (int)Math.Min(Math.Min(available, into.Length), capacity - at);
            new ReadOnlySpan<byte>(Data + at, n).CopyTo(into);
            into = into[n..];
            read += n;
            available -= n;
            total += n;
        }

        Volatile.Write(ref ReadPosition, read);
        return total;
    }

    /// <summary>Whether bytes are waiting past <paramref name="read"/>.</summary>
    public bool HasData(long read) => Volatile.Read(ref WritePosition) > read;

    public void MarkWriterClosed() => Interlocked.Or(ref State, WriterClosed);

    public void Dispose()
    {
        if (_disposed)
        {
            return;
        }

        _disposed = true;
        _view.SafeMemoryMappedViewHandle.ReleasePointer();
        _view.Dispose();
        _file.Dispose();
    }
}

/// <summary>
/// "shared-memory" (plan 23): the ranks of one machine, each pair joined by two <see cref="SharedRing"/>s under a folder of
/// the run's own (named after <see cref="ClusterOptions.RunId"/>, else the coordinator, in
/// <see cref="ClusterOptions.SharedMemoryFolder"/>). Each rank makes the rings it reads, opens the ones it writes, and
/// sends a hello first: the HMAC of a nonce under the cluster key, refused when it does not prove it. One reader thread per
/// rank drains every ring into the queues, spinning briefly, then yielding, then sleeping a millisecond while nothing
/// arrives; while it sleeps it checks now and then that each writer's process is still there.
/// </summary>
internal sealed class SharedMemoryTransport : FramedTransport
{
    private static readonly TimeSpan Napping = TimeSpan.FromMilliseconds(5);       // idle this long, then sleep between looks
    private static readonly TimeSpan Checking = TimeSpan.FromSeconds(1);           // while idle, look for vanished writers this often
    private readonly SharedRing?[] _incoming;
    private readonly SharedRing?[] _outgoing;
    private readonly long[] _written;
    private readonly Lock[] _writing;
    private readonly string _folder;
    private readonly ClusterKey? _key;
    private readonly CountdownEvent _hellos;
    private readonly bool[] _greeted;
    private readonly Thread _reader;
    private volatile bool _stopping;

    private SharedMemoryTransport(int rank, int world, string folder, ClusterOptions options) : base("shared-memory", rank, world, options.Timeout)
    {
        _folder = folder;
        _key = options.Key;
        _incoming = new SharedRing?[world];
        _outgoing = new SharedRing?[world];
        _written = new long[world];
        _writing = [.. Enumerable.Range(0, world).Select(_ => new Lock())];
        _greeted = new bool[world];
        _hellos = new CountdownEvent(world - 1);
        _reader = new Thread(ReadLoop) { IsBackground = true, Name = $"shared-memory rank {rank}" };
    }

    public override TransportCapabilities Capabilities => SharedMemoryProvider.Estimate;

    public static SharedMemoryTransport Connect(ClusterContext context, CancellationToken cancellationToken)
    {
        var options = context.Options;
        int me = context.Rank, world = context.Topology.WorldSize;
        string run = options.RunId ?? options.Coordinator ?? "default";
        string folder = Path.Combine(options.SharedMemoryFolder ?? Path.GetTempPath(), "idrak-cluster", Convert.ToHexString(SHA256.HashData(Encoding.UTF8.GetBytes(run)))[..16]);
        Directory.CreateDirectory(folder);
        long capacity = Math.Max(WireFormat.HeaderBytes * 2, options.SharedMemoryBytes);
        var transport = new SharedMemoryTransport(me, world, folder, options);
        try
        {
            for (int p = 0; p < world; p++)
            {
                if (p != me)
                {
                    transport._incoming[p] = SharedRing.Create(RingPath(folder, p, me), capacity);
                }
            }

            long deadline = Stopwatch.GetTimestamp() + (long)(options.Timeout.TotalSeconds * Stopwatch.Frequency);
            for (int p = 0; p < world; p++)
            {
                while (p != me && transport._outgoing[p] is null)
                {
                    cancellationToken.ThrowIfCancellationRequested();
                    transport._outgoing[p] = SharedRing.TryOpen(RingPath(folder, me, p));
                    if (transport._outgoing[p] is null)
                    {
                        if (Stopwatch.GetTimestamp() > deadline)
                        {
                            throw new TimeoutException($"rank {me} waited {options.Timeout.TotalSeconds:F0} s for rank {p} to join on this machine (its ring in {folder} did not appear)");
                        }

                        Thread.Sleep(5);
                    }
                }
            }

            transport._reader.Start();
            Span<byte> hello = stackalloc byte[WireFormat.NonceBytes + WireFormat.MacBytes];
            for (int p = 0; p < world; p++)
            {
                if (p == me)
                {
                    continue;
                }

                WireFormat.NewNonce(hello[..WireFormat.NonceBytes]);
                if (options.Key is { } key)
                {
                    WireFormat.SignHello(key, hello[..WireFormat.NonceBytes], me, world, hello[WireFormat.NonceBytes..]);
                }
                else
                {
                    hello[WireFormat.NonceBytes..].Clear();
                }

                transport.WriteFrame(p, new WireHeader(WireMessageKind.Hello, me, 0, 0, 0, (uint)world, WireElement.Bytes, ReduceOp.Sum, 0, (uint)hello.Length), hello);
            }

            if (!transport._hellos.Wait(options.Timeout, cancellationToken))
            {
                var missing = Enumerable.Range(0, world).Where(p => p != me && !Volatile.Read(ref transport._greeted[p]));
                throw new TimeoutException($"rank {me} waited {options.Timeout.TotalSeconds:F0} s for the hello of rank(s) {string.Join(", ", missing)}");
            }

            for (int p = 0; p < world; p++)
            {
                if (p != me && transport.GoneReason(p) is { } refused)
                {
                    throw new InvalidOperationException(refused);
                }
            }

            return transport;
        }
        catch
        {
            transport.Dispose();
            throw;
        }
    }

    private static string RingPath(string folder, int from, int to) => Path.Combine(folder, $"{from}-{to}.ring");

    protected override void WriteFrame(int to, in WireHeader header, ReadOnlySpan<byte> payload)
    {
        var ring = _outgoing[to] ?? throw new InvalidOperationException($"rank {Rank} has no ring to rank {to}");
        lock (_writing[to])
        {
            Span<byte> head = stackalloc byte[WireFormat.HeaderBytes];
            WireFormat.WriteHeader(head, header);
            long since = Stopwatch.GetTimestamp();
            var idle = new SpinWait();
            void Wait()
            {
                if (GoneReason(to) is { } reason)
                {
                    throw new InvalidOperationException($"rank {Rank} cannot send to rank {to}: {reason}");
                }

                if (Stopwatch.GetElapsedTime(since) > (Disposed ? Napping * 40 : Timeout))   // a goodbye to a full ring waits briefly
                {
                    throw new TimeoutException($"rank {Rank} waited {Timeout.TotalSeconds:F0} s to send to rank {to}: its ring stayed full (it does not take its messages)");
                }

                if (Stopwatch.GetElapsedTime(since) > Napping)
                {
                    Thread.Sleep(1);
                }
                else
                {
                    idle.SpinOnce(sleep1Threshold: -1);
                }
            }

            ring.Write(head, ref _written[to], Wait);
            ring.Write(payload, ref _written[to], Wait);
        }
    }

    protected override void CloseCarriers()
    {
        _stopping = true;
        foreach (var ring in _outgoing)
        {
            ring?.MarkWriterClosed();
        }

        if (_reader.IsAlive)
        {
            _reader.Join(TimeSpan.FromSeconds(5));
        }

        // Every ring this rank used goes: its own, and those it wrote to (a peer that died cannot remove them). A file still
        // mapped by a live peer stays readable to it until it closes it.
        foreach (var ring in _incoming.Concat(_outgoing))
        {
            if (ring is null)
            {
                continue;
            }

            ring.Dispose();
            try
            {
                File.Delete(ring.Path);
            }
            catch (Exception e) when (e is IOException or UnauthorizedAccessException)
            {
                // removed already, or held where removing an open file is refused: the peer removes it as it leaves
            }
        }

        try
        {
            Directory.Delete(_folder);                            // the last rank out removes the run's folder
        }
        catch (Exception e) when (e is IOException or UnauthorizedAccessException)
        {
            // other ranks' rings are still there
        }

        _hellos.Dispose();
    }

    // Every incoming ring in turn: a frame's header, then its payload into a pooled buffer, handed to the queues whole.
    private void ReadLoop()
    {
        int world = WorldSize;
        var head = new byte[world][];
        var filled = new int[world];
        var headers = new WireHeader?[world];
        var payloads = new byte[]?[world];
        var read = new long[world];
        var started = new long[world];
        for (int p = 0; p < world; p++)
        {
            head[p] = new byte[WireFormat.HeaderBytes];
        }

        long idleSince = Stopwatch.GetTimestamp(), lastCheck = idleSince;
        var spin = new SpinWait();
        try
        {
            while (!_stopping)
            {
                bool moved = false;
                for (int p = 0; p < world; p++)
                {
                    if (_incoming[p] is not { } ring || GoneReason(p) is not null)
                    {
                        continue;
                    }

                    while (ring.HasData(read[p]))
                    {
                        moved = true;
                        if (headers[p] is null)
                        {
                            filled[p] += ring.Read(head[p].AsSpan(filled[p]), ref read[p]);
                            if (filled[p] < WireFormat.HeaderBytes)
                            {
                                continue;
                            }

                            WireFormat.TryReadHeader(head[p], out var header);
                            if (header.Rank != p)
                            {
                                throw new InvalidDataException($"a frame in rank {p}'s ring says it is from rank {header.Rank}");
                            }

                            headers[p] = header;
                            filled[p] = 0;
                            started[p] = Telemetry.IsEnabled(TelemetryLevel.Network) ? Stopwatch.GetTimestamp() : 0;
                            payloads[p] = ArrayPool<byte>.Shared.Rent(Math.Max(1, checked((int)header.PayloadBytes)));
                        }

                        var current = headers[p]!.Value;
                        int length = (int)current.PayloadBytes;
                        filled[p] += ring.Read(payloads[p].AsSpan(filled[p], length - filled[p]), ref read[p]);
                        if (filled[p] < length)
                        {
                            continue;
                        }

                        var buffer = payloads[p]!;
                        (headers[p], payloads[p], filled[p]) = (null, null, 0);
                        Complete(p, current, buffer, length, Since(started[p]));
                    }

                    if (!ring.HasData(read[p]) && ring.WriterHasClosed && headers[p] is null)
                    {
                        PeerGone(p, $"rank {p} left the group");
                    }
                }

                long now = Stopwatch.GetTimestamp();
                if (moved)
                {
                    idleSince = now;
                    spin.Reset();
                    continue;
                }

                if (Stopwatch.GetElapsedTime(idleSince, now) < Napping)
                {
                    spin.SpinOnce(sleep1Threshold: -1);
                    continue;
                }

                if (Stopwatch.GetElapsedTime(lastCheck, now) > Checking)
                {
                    lastCheck = now;
                    CheckWriters();
                }

                Thread.Sleep(1);
            }
        }
        catch (Exception e) when (e is InvalidDataException or OverflowException or ObjectDisposedException)
        {
            for (int p = 0; p < world; p++)
            {
                if (p != Rank)
                {
                    PeerGone(p, $"rank {Rank}'s shared-memory reader stopped: {e.Message}");
                }
            }
        }
        finally
        {
            foreach (var payload in payloads)
            {
                if (payload is not null)
                {
                    ArrayPool<byte>.Shared.Return(payload);
                }
            }
        }
    }

    // A whole frame from `from`: the hello is checked here; nothing else is taken from a rank before its hello passed.
    private void Complete(int from, in WireHeader header, byte[] buffer, int length, TimeSpan wall)
    {
        if (header.Kind == WireMessageKind.Hello)
        {
            bool proven = length == WireFormat.NonceBytes + WireFormat.MacBytes && header.ChunkCount == (uint)WorldSize
                && (_key is null
                    ? buffer.AsSpan(WireFormat.NonceBytes, WireFormat.MacBytes).IndexOfAnyExcept((byte)0) < 0
                    : WireFormat.VerifyHello(_key, buffer.AsSpan(0, WireFormat.NonceBytes), from, WorldSize, buffer.AsSpan(WireFormat.NonceBytes, WireFormat.MacBytes)));
            ArrayPool<byte>.Shared.Return(buffer);
            if (!proven)
            {
                PeerGone(from, $"rank {Rank} refused the hello of rank {from}: it does not prove the cluster key (IDRAK_CLUSTER_KEY differs between the two), or its group size differs");
            }

            if (!_greeted[from])
            {
                Volatile.Write(ref _greeted[from], true);
                _hellos.Signal();
            }

            return;
        }

        if (!_greeted[from])
        {
            ArrayPool<byte>.Shared.Return(buffer);
            PeerGone(from, $"rank {from} sent a {header.Kind} before its hello");
            return;
        }

        Arrived(from, header, buffer, length, wall);
    }

    // A writer whose process is gone left without its goodbye.
    private void CheckWriters()
    {
        for (int p = 0; p < WorldSize; p++)
        {
            if (_incoming[p] is not { } ring || GoneReason(p) is not null)
            {
                continue;
            }

            int pid = ring.WriterProcess;
            if (pid == 0 || pid == Environment.ProcessId)
            {
                continue;
            }

            bool alive;
            try
            {
                using var process = Process.GetProcessById(pid);
                alive = !process.HasExited;
            }
            catch (Exception e) when (e is ArgumentException or InvalidOperationException)
            {
                alive = false;
            }

            if (!alive)
            {
                PeerGone(p, $"rank {p}'s process ({pid}) ended without a goodbye");
            }
        }
    }
}

/// <summary>"shared-memory" (plan 23): the ranks of one machine, as separate processes or not, through mapped rings.</summary>
internal sealed class SharedMemoryProvider : ITransportProvider
{
    public static readonly TransportCapabilities Estimate = new(0, TimeSpan.Zero, ZeroCopy: false, GpuDirect: false, Reliable: true, CrossProcess: true, CrossMachine: false);

    public string Name => "shared-memory";

    public string Summary => "the ranks of one machine: a memory-mapped ring per direction, a keyed hello first";

    public TransportCapabilities Capabilities => Estimate;

    public TransportAvailability Availability(ClusterContext context, int peer)
    {
        var topology = context.Topology;
        var a = topology.Place(context.Rank).Node;
        var b = topology.Place(peer).Node;
        return a == b
            ? new(Name, true, $"ranks {context.Rank} and {peer} are on one machine ({topology.Nodes[a].Name})")
            : new(Name, false, $"ranks {context.Rank} and {peer} are on different machines ({topology.Nodes[a].Name}, {topology.Nodes[b].Name})");
    }

    public ITransport Connect(ClusterContext context, CancellationToken cancellationToken = default) => SharedMemoryTransport.Connect(context, cancellationToken);
}
