// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Buffers;
using System.Diagnostics;
using System.IO.MemoryMappedFiles;
using System.Net;
using System.Net.Sockets;
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
/// <para>
/// Phase C6: a side with nothing to do after a short spin raises its flag in the header (the reader: "waiting for bytes";
/// the writer: "waiting for room") and blocks on its <see cref="Doorbell"/>, whose port the header holds; the other side,
/// after publishing a position, takes the flag and rings that doorbell. Each side writes its position (or flag), then a
/// full fence, then reads the other's flag (or position), so a wake is never lost.
/// </para>
/// </summary>
internal sealed unsafe class SharedRing : IDisposable
{
    public const int HeaderBytes = 128;
    private const int WriteAt = 0, ReadAt = 64, WriterPidAt = 72, ReaderPidAt = 76, StateAt = 80, CapacityAt = 88;
    private const int ReaderBellAt = 96, ReaderWaitingAt = 100, WriterBellAt = 104, WriterWaitingAt = 108;
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

    private ref int ReaderWaiting => ref Unsafe.AsRef<int>(_base + ReaderWaitingAt);

    private ref int WriterWaiting => ref Unsafe.AsRef<int>(_base + WriterWaitingAt);

    /// <summary>The port of the reader's doorbell (set by the reader as it makes the ring).</summary>
    public int ReaderBell => Volatile.Read(ref Unsafe.AsRef<int>(_base + ReaderBellAt));

    /// <summary>The port of the writer's doorbell (set by the writer as it opens the ring).</summary>
    public int WriterBell => Volatile.Read(ref Unsafe.AsRef<int>(_base + WriterBellAt));

    /// <summary>The reader is about to block: the writer rings it after its next published piece.</summary>
    public void ReaderWaits()
    {
        Volatile.Write(ref ReaderWaiting, 1);
        Interlocked.MemoryBarrier();                                 // the flag before the reader's last look at the position
    }

    public void ReaderAwake() => Volatile.Write(ref ReaderWaiting, 0);

    /// <summary>After a published piece: whether the reader waits for it (taking the flag, so it is rung once).</summary>
    public bool TakeReaderWaiting()
    {
        Interlocked.MemoryBarrier();                                 // the position before the look at the flag
        return Volatile.Read(ref ReaderWaiting) != 0 && Interlocked.Exchange(ref ReaderWaiting, 0) != 0;
    }

    /// <summary>The writer is about to block for room: the reader rings it after it next frees some.</summary>
    public void WriterWaits()
    {
        Volatile.Write(ref WriterWaiting, 1);
        Interlocked.MemoryBarrier();
    }

    public void WriterAwake() => Volatile.Write(ref WriterWaiting, 0);

    /// <summary>After freeing room: whether the writer waits for it (taking the flag).</summary>
    public bool TakeWriterWaiting()
    {
        Interlocked.MemoryBarrier();
        return Volatile.Read(ref WriterWaiting) != 0 && Interlocked.Exchange(ref WriterWaiting, 0) != 0;
    }

    /// <summary>The room the writer at <paramref name="written"/> has.</summary>
    public long Free(long written) => Capacity - (written - Volatile.Read(ref ReadPosition));

    /// <summary>The receiver makes its ring afresh (an old file of that name is removed first) and marks it ready last.</summary>
    public static SharedRing Create(string path, long capacity, int readerBell)
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
            Unsafe.AsRef<int>(ring._base + ReaderBellAt) = readerBell;
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
    public static SharedRing? TryOpen(string path, int writerBell)
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
            Volatile.Write(ref Unsafe.AsRef<int>(ring._base + WriterBellAt), writerBell);
            return ring;
        }
        catch
        {
            stream.Dispose();
            throw;
        }
    }

    /// <summary>
    /// Copies <paramref name="bytes"/> in as room frees, publishing each piece (<paramref name="published"/> after each);
    /// <paramref name="wait"/> is called while the ring is full and throws to give up.
    /// </summary>
    public void Write(ReadOnlySpan<byte> bytes, ref long written, Action wait, Action published)
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
            published();
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
/// A wake-up between processes of one machine (plan 23, phase C6): a datagram socket on loopback, which every system .NET
/// runs on has (rule 70). A side that has nothing to do blocks on its own doorbell (<see cref="Wait"/>, with a timeout so
/// a peer that vanished is still noticed); the other side rings it by port (<see cref="Ring"/>) only when the waiting
/// side's flag says it waits. A wake costs a datagram on loopback, microseconds, where sleeping waited a timer tick.
/// </summary>
internal sealed class Doorbell : IDisposable
{
    private static readonly byte[] Knock = [1];
    private readonly Socket _socket;
    private readonly byte[] _drain = new byte[64];

    public Doorbell()
    {
        _socket = new Socket(AddressFamily.InterNetwork, SocketType.Dgram, ProtocolType.Udp);
        try
        {
            _socket.Bind(new IPEndPoint(IPAddress.Loopback, 0));
        }
        catch
        {
            _socket.Dispose();
            throw;
        }

        Port = ((IPEndPoint)_socket.LocalEndPoint!).Port;
    }

    /// <summary>The loopback port others ring.</summary>
    public int Port { get; }

    /// <summary>Rings the doorbell at <paramref name="port"/> (nothing when it is gone).</summary>
    public void Ring(int port)
    {
        if (port <= 0)
        {
            return;
        }

        try
        {
            lock (_socket)
            {
                _socket.SendTo(Knock, new IPEndPoint(IPAddress.Loopback, port));
            }
        }
        catch (Exception e) when (e is SocketException or ObjectDisposedException)
        {
            // The other side is gone: its waits end with its own timeout.
        }
    }

    /// <summary>Blocks until rung or <paramref name="timeout"/> passed; every ring that arrived meanwhile is taken. True when rung.</summary>
    public bool Wait(TimeSpan timeout)
    {
        try
        {
            bool rung = _socket.Poll((int)Math.Clamp(timeout.TotalMicroseconds, 0, int.MaxValue), SelectMode.SelectRead);
            while (_socket.Available > 0)
            {
                _socket.Receive(_drain);
            }

            return rung;
        }
        catch (Exception e) when (e is SocketException or ObjectDisposedException)
        {
            return false;
        }
    }

    public void Dispose() => _socket.Dispose();

    /// <summary>
    /// What a wake costs on this machine: a thread blocked on a doorbell, rung from another thread, until it runs again (the
    /// median of a few, measured once per process). A side that spins this long before it blocks pays at most twice what an
    /// oracle would (the classic spin-then-block bound), so it is the default spin (<see cref="ClusterOptions.SharedMemorySpin"/>).
    /// </summary>
    public static TimeSpan WakeCost => LazyWakeCost.Value;

    private static readonly Lazy<TimeSpan> LazyWakeCost = new(() =>
    {
        using var bell = new Doorbell();
        var times = new double[7];
        for (int i = 0; i < times.Length; i++)
        {
            long rung = 0, woke = 0;
            var waiter = new Thread(() =>
            {
                bell.Wait(TimeSpan.FromSeconds(1));
                woke = Stopwatch.GetTimestamp();
            }) { IsBackground = true, Name = "doorbell wake measurement" };
            waiter.Start();
            Thread.Sleep(2);                                         // the waiter is blocked by now
            rung = Stopwatch.GetTimestamp();
            bell.Ring(bell.Port);
            waiter.Join();
            times[i] = Math.Max(0, woke - rung) / (double)Stopwatch.Frequency;
        }

        Array.Sort(times);
        return TimeSpan.FromSeconds(times[times.Length / 2]);
    });
}

/// <summary>
/// "shared-memory" (plan 23): the ranks of one machine, each pair joined by two <see cref="SharedRing"/>s under a folder of
/// the run's own (named after <see cref="ClusterOptions.RunId"/>, else the coordinator, in
/// <see cref="ClusterOptions.SharedMemoryFolder"/>). Each rank makes the rings it reads, opens the ones it writes, and
/// sends a hello first: the HMAC of a nonce under the cluster key, refused when it does not prove it. One reader thread per
/// rank drains every ring into the queues, spinning for <see cref="ClusterOptions.SharedMemorySpin"/> while nothing arrives,
/// then blocking on its <see cref="Doorbell"/> (phase C6: it slept a millisecond between looks before, a whole timer tick
/// on some systems, which held every message back by that much); while it waits it checks every
/// <see cref="ClusterOptions.SharedMemoryCheck"/> that each writer's process is still there. A writer facing a full ring
/// spins the same, then blocks on a doorbell of its own the reader rings as it frees room.
/// </summary>
internal sealed class SharedMemoryTransport : FramedTransport
{
    private readonly TimeSpan _spin;                          // idle this long, then block on the doorbell
    private readonly TimeSpan _check;                         // while idle, look for vanished writers this often
    private readonly Doorbell _bell;                          // the reader thread's
    private readonly Doorbell?[] _roomBells;                  // each outgoing ring's writer waits on its own
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

    private SharedMemoryTransport(int rank, int world, string folder, ClusterOptions options) : base("shared-memory", rank, world, options)
    {
        _folder = folder;
        _key = options.Key;
        _incoming = new SharedRing?[world];
        _outgoing = new SharedRing?[world];
        _written = new long[world];
        _writing = [.. Enumerable.Range(0, world).Select(_ => new Lock())];
        _greeted = new bool[world];
        _hellos = new CountdownEvent(world - 1);
        _spin = options.SharedMemorySpin ?? Doorbell.WakeCost;
        _check = options.SharedMemoryCheck;
        _bell = new Doorbell();
        _roomBells = new Doorbell?[world];
        for (int p = 0; p < world; p++)
        {
            _roomBells[p] = p == rank ? null : new Doorbell();
        }

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
                    transport._incoming[p] = SharedRing.Create(RingPath(folder, p, me), capacity, transport._bell.Port);
                }
            }

            long deadline = Stopwatch.GetTimestamp() + (long)(options.Timeout.TotalSeconds * Stopwatch.Frequency);
            for (int p = 0; p < world; p++)
            {
                while (p != me && transport._outgoing[p] is null)
                {
                    cancellationToken.ThrowIfCancellationRequested();
                    transport._outgoing[p] = SharedRing.TryOpen(RingPath(folder, me, p), transport._roomBells[p]!.Port);
                    if (transport._outgoing[p] is null)
                    {
                        if (Stopwatch.GetTimestamp() > deadline)
                        {
                            throw new TimeoutException($"rank {me} waited {options.Timeout.TotalSeconds:F0} s for rank {p} to join on this machine (its ring in {folder} did not appear)");
                        }

                        Thread.Sleep(5);                          // once, while the peer process starts: not on any message's path
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
            var bell = _roomBells[to]!;
            void Wait()
            {
                if (GoneReason(to) is { } reason)
                {
                    throw new InvalidOperationException($"rank {Rank} cannot send to rank {to}: {reason}");
                }

                var waited = Stopwatch.GetElapsedTime(since);
                var limit = Disposed ? _check : Timeout;              // a goodbye to a full ring waits briefly
                if (waited > limit)
                {
                    throw new TimeoutException($"rank {Rank} waited {Timeout.TotalSeconds:F0} s to send to rank {to}: its ring stayed full (it does not take its messages)");
                }

                if (waited < _spin)
                {
                    idle.SpinOnce(sleep1Threshold: -1);
                    return;
                }

                ring.WriterWaits();                                 // the reader rings as it frees room
                if (ring.Free(_written[to]) <= 0)
                {
                    bell.Wait(Shortest(_check, limit - waited));
                }

                ring.WriterAwake();
            }

            void Published()
            {
                if (ring.TakeReaderWaiting())
                {
                    _bell.Ring(ring.ReaderBell);                    // the reader blocked on its doorbell: wake it
                }
            }

            ring.Write(head, ref _written[to], Wait, Published);
            ring.Write(payload, ref _written[to], Wait, Published);
        }
    }

    private static TimeSpan Shortest(TimeSpan a, TimeSpan b)
    {
        var shortest = a < b ? a : b;
        return shortest < TimeSpan.Zero ? TimeSpan.Zero : shortest;
    }

    protected override void CloseCarriers()
    {
        _stopping = true;
        foreach (var ring in _outgoing)
        {
            if (ring is not null)
            {
                ring.MarkWriterClosed();
                _bell.Ring(ring.ReaderBell);                        // the peer's reader learns it now, not at its next check
            }
        }

        _bell.Ring(_bell.Port);                                     // this rank's reader stops now
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
        _bell.Dispose();
        foreach (var bell in _roomBells)
        {
            bell?.Dispose();
        }
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
                            filled[p] += Take(ring, head[p].AsSpan(filled[p]), ref read[p]);
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
                        filled[p] += Take(ring, payloads[p].AsSpan(filled[p], length - filled[p]), ref read[p]);
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

                if (Stopwatch.GetElapsedTime(idleSince, now) < _spin)
                {
                    spin.SpinOnce(sleep1Threshold: -1);
                    continue;
                }

                if (Stopwatch.GetElapsedTime(lastCheck, now) > _check)
                {
                    lastCheck = now;
                    CheckWriters();
                }

                // Nothing for a while: raise the flag on every ring, look once more (a piece published before the flag was
                // seen), then block on the doorbell until a writer rings it or the check is due.
                for (int p = 0; p < world; p++)
                {
                    _incoming[p]?.ReaderWaits();
                }

                bool any = false;
                for (int p = 0; p < world && !any; p++)
                {
                    any = _incoming[p] is { } ring && (ring.HasData(read[p]) || ring.WriterHasClosed && GoneReason(p) is null);
                }

                if (!any && !_stopping)
                {
                    _bell.Wait(_check - Stopwatch.GetElapsedTime(lastCheck, Stopwatch.GetTimestamp()));
                }

                for (int p = 0; p < world; p++)
                {
                    _incoming[p]?.ReaderAwake();
                }

                idleSince = Stopwatch.GetTimestamp();
                spin.Reset();
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

    // Copies what arrived out of a ring and, when its writer waits for room, rings it.
    private int Take(SharedRing ring, Span<byte> into, ref long read)
    {
        int n = ring.Read(into, ref read);
        if (n > 0 && ring.TakeWriterWaiting())
        {
            _bell.Ring(ring.WriterBell);
        }

        return n;
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
