// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Buffers;
using System.Collections.Concurrent;
using System.Diagnostics;
using System.Threading.Channels;

namespace Idrak.Networking.Abstractions;

/// <summary>A message that arrived: its bytes in a pooled buffer the receiver returns.</summary>
internal readonly record struct Inbound(byte[] Buffer, int Length);

/// <summary>
/// What every real transport here shares (plan 23, phase C5): messages travel as <see cref="WireFormat"/> frames (a
/// 36-byte header, then the payload) over a carrier of the transport's own (a socket, a shared-memory ring, a call inside
/// the process); a reader per carrier drains what arrives into one queue per sending rank and tag, so a send never waits
/// on a peer that is itself sending (every rank's reader keeps taking), and a receive takes the oldest message of its
/// rank and tag, waiting up to <see cref="ClusterOptions.Timeout"/>. A barrier is a frame to rank 0 and one back. A rank
/// that leaves says so (a bye frame); a carrier that closes without one ends every wait on that rank at once, naming it.
/// <para>
/// Flow control (phase C6): for each peer and tag a sender has at most <see cref="ClusterOptions.InFlight"/> messages, and
/// <see cref="ClusterOptions.ReceiveBytes"/> bytes, that the peer has not taken yet; the receiver acknowledges each message
/// as its receive takes it (an ack frame), and a sender beyond its credit waits for one (an awaited send awaits). A message
/// is always let go when none of its peer and tag is outstanding, so a message larger than the budget travels alone and
/// ranks that all send before they receive never wait on each other in a circle, while a fast sender can no longer fill a
/// slow receiver's memory: the queues hold at most the credit, however fast the sender.
/// </para>
/// </summary>
internal abstract class FramedTransport : ITransport
{
    /// <summary>The tag barrier frames are queued under: below every tag a program or an algorithm uses.</summary>
    protected const int BarrierTag = int.MinValue;

    private readonly ConcurrentDictionary<(int From, int Tag), Channel<Inbound>> _inbox = new();
    private readonly ConcurrentDictionary<(int To, int Tag), StrongBox> _sequence = new();
    private readonly ConcurrentDictionary<(int To, int Tag), Credit> _credits = new();
    private readonly string?[] _gone;
    private readonly int _inFlight;
    private readonly long? _receiveBytes;
    private volatile bool _disposed;
    private long _queued;
    private long _peakQueued;

    protected FramedTransport(string name, int rank, int worldSize, ClusterOptions options)
    {
        ArgumentNullException.ThrowIfNull(options);
        Name = name;
        Rank = rank;
        WorldSize = worldSize;
        Timeout = options.Timeout;
        _inFlight = Math.Max(1, options.InFlight);
        _receiveBytes = options.ReceiveBytes;
        _gone = new string?[worldSize];
    }

    /// <summary>The most bytes this rank held received and not yet taken, over every peer and tag at once (for tests and probes).</summary>
    internal long PeakQueuedBytes => Interlocked.Read(ref _peakQueued);

    /// <summary>The bytes this rank holds received and not yet taken.</summary>
    internal long QueuedBytes => Interlocked.Read(ref _queued);

    public string Name { get; }

    public int Rank { get; }

    public int WorldSize { get; }

    public abstract TransportCapabilities Capabilities { get; }

    protected TimeSpan Timeout { get; }

    protected bool Disposed => _disposed;

    public void Send(int to, ReadOnlySpan<byte> bytes, int tag)
    {
        Check(to, nameof(to));
        if (_gone[to] is { } reason)
        {
            throw new InvalidOperationException($"rank {Rank} cannot send to rank {to} (tag {tag}): {reason}");
        }

        Acquire(to, tag, bytes.Length);
        WriteChunk(to, bytes, tag);
    }

    private void WriteChunk(int to, ReadOnlySpan<byte> bytes, int tag)
    {
        var header = new WireHeader(WireMessageKind.Chunk, Rank, tag, Next(to, tag), 0, 1, WireElement.Bytes, ReduceOp.Sum, 0, (uint)bytes.Length);
        try
        {
            WriteFrame(to, header, bytes);
        }
        catch
        {
            Credits(to, tag).Release(bytes.Length);            // never sent: its credit comes back
            throw;
        }
    }

    public int Receive(int from, Span<byte> bytes, int tag) => Deliver(from, tag, Take(from, tag), bytes);

    public ValueTask SendAsync(int to, ReadOnlyMemory<byte> bytes, int tag, CancellationToken cancellationToken = default)
    {
        try
        {
            cancellationToken.ThrowIfCancellationRequested();
            Check(to, nameof(to));
            if (_gone[to] is { } reason)
            {
                throw new InvalidOperationException($"rank {Rank} cannot send to rank {to} (tag {tag}): {reason}");
            }

            if (!Credits(to, tag).TryTake(bytes.Length, _inFlight, _receiveBytes))
            {
                return SendWhenFree(to, bytes, tag, cancellationToken);   // beyond the credit: awaited, never a blocked thread
            }

            WriteChunk(to, bytes.Span, tag);                       // the frame is written (or queued by the carrier) when it returns
            return ValueTask.CompletedTask;
        }
        catch (Exception e)
        {
            return ValueTask.FromException(e);
        }
    }

    private async ValueTask SendWhenFree(int to, ReadOnlyMemory<byte> bytes, int tag, CancellationToken cancellationToken)
    {
        var credit = Credits(to, tag);
        long deadline = Stopwatch.GetTimestamp() + Ticks(Timeout);
        while (!credit.TryTake(bytes.Length, _inFlight, _receiveBytes))
        {
            var freed = credit.Freed();
            if (credit.TryTake(bytes.Length, _inFlight, _receiveBytes))
            {
                break;
            }

            CheckWaiting(to, tag, deadline);
            try
            {
                await freed.WaitAsync(Remaining(deadline), cancellationToken).ConfigureAwait(false);
            }
            catch (TimeoutException)
            {
                CheckWaiting(to, tag, deadline);
            }
        }

        WriteChunk(to, bytes.Span, tag);
    }

    // Waits until rank `to` has room for `bytes` more bytes under `tag` (always when nothing of that peer and tag is outstanding).
    private void Acquire(int to, int tag, int bytes)
    {
        var credit = Credits(to, tag);
        if (credit.TryTake(bytes, _inFlight, _receiveBytes))
        {
            return;
        }

        long deadline = Stopwatch.GetTimestamp() + Ticks(Timeout);
        lock (credit.Gate)
        {
            while (!credit.TryTakeLocked(bytes, _inFlight, _receiveBytes))
            {
                CheckWaiting(to, tag, deadline);
                Monitor.Wait(credit.Gate, Clamp(Remaining(deadline)));
            }
        }
    }

    // A sender waiting for credit gives up when the peer is gone, the transport closed, or the wait reached the timeout.
    private void CheckWaiting(int to, int tag, long deadline)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        if (_gone[to] is { } reason)
        {
            throw new InvalidOperationException($"rank {Rank} cannot send to rank {to} (tag {tag}): {reason}");
        }

        if (Stopwatch.GetTimestamp() >= deadline)
        {
            throw new TimeoutException($"rank {Rank} waited {Timeout.TotalSeconds:F0} s to send to rank {to} (tag {tag}): it holds {_inFlight} message(s)"
                                       + (_receiveBytes is { } budget ? $" or {budget:N0} bytes" : "") + " of this rank's it has not taken (IDRAK_CLUSTER_IN_FLIGHT, IDRAK_CLUSTER_RECEIVE_BYTES)");
        }
    }

    private static long Ticks(TimeSpan span) => (long)(span.TotalSeconds * Stopwatch.Frequency);

    private static TimeSpan Remaining(long deadline) => TimeSpan.FromSeconds(Math.Max(0, deadline - Stopwatch.GetTimestamp()) / (double)Stopwatch.Frequency);

    // A wait is cut into pieces of at most a second, so a peer that left is noticed even when nothing signals.
    private static TimeSpan Clamp(TimeSpan wait) => wait > TimeSpan.FromSeconds(1) ? TimeSpan.FromSeconds(1) : wait;

    private Credit Credits(int to, int tag) => _credits.GetOrAdd((to, tag), static _ => new Credit());

    public async ValueTask<int> ReceiveAsync(int from, Memory<byte> bytes, int tag, CancellationToken cancellationToken = default)
    {
        Check(from, nameof(from));
        var reader = Inbox(from, tag).Reader;
        if (!reader.TryRead(out var message))
        {
            using var timeout = CancellationTokenSource.CreateLinkedTokenSource(cancellationToken);
            timeout.CancelAfter(Timeout);
            try
            {
                message = await reader.ReadAsync(timeout.Token).ConfigureAwait(false);
            }
            catch (OperationCanceledException) when (!cancellationToken.IsCancellationRequested)
            {
                throw TimedOut(from, tag);
            }
            catch (ChannelClosedException)
            {
                throw Gone(from, tag);
            }
        }

        return Deliver(from, tag, message, bytes.Span);
    }

    public void Barrier()
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        if (WorldSize == 1)
        {
            return;
        }

        var frame = new WireHeader(WireMessageKind.Barrier, Rank, 0, 0, 0, 1, WireElement.Bytes, ReduceOp.Sum, 0, 0);
        if (Rank != 0)
        {
            WriteFrame(0, frame, []);
            Return(Take(0, BarrierTag));
            return;
        }

        for (int r = 1; r < WorldSize; r++)
        {
            Return(Take(r, BarrierTag));
        }

        for (int r = 1; r < WorldSize; r++)
        {
            WriteFrame(r, frame, []);
        }
    }

    public void Dispose()
    {
        if (_disposed)
        {
            return;
        }

        _disposed = true;
        var bye = new WireHeader(WireMessageKind.Bye, Rank, 0, 0, 0, 1, WireElement.Bytes, ReduceOp.Sum, 0, 0);
        for (int r = 0; r < WorldSize; r++)
        {
            if (r != Rank && _gone[r] is null)
            {
                try
                {
                    WriteFrame(r, bye, []);
                }
                catch (Exception e) when (e is IOException or InvalidOperationException or TimeoutException or ObjectDisposedException or System.Net.Sockets.SocketException)
                {
                    // The peer is gone already: nothing to say goodbye to.
                }
            }
        }

        CloseCarriers();
        foreach (var (_, credit) in _credits)
        {
            credit.Wake();                                         // senders waiting for credit see the transport closed
        }

        foreach (var (_, channel) in _inbox)
        {
            channel.Writer.TryComplete();
            while (channel.Reader.TryRead(out var left))
            {
                Return(left);
            }
        }
    }

    /// <summary>Writes one frame to <paramref name="to"/> (thread-safe per peer); the payload is not held after it returns.</summary>
    protected abstract void WriteFrame(int to, in WireHeader header, ReadOnlySpan<byte> payload);

    /// <summary>Closes every carrier and stops the readers (after the bye frames went out).</summary>
    protected abstract void CloseCarriers();

    /// <summary>
    /// A reader has a whole frame from <paramref name="from"/>: a chunk or a barrier goes to its queue (the buffer is the
    /// queue's from now on), a bye or an error ends that rank. <paramref name="wall"/> is the time its bytes took to arrive.
    /// </summary>
    protected void Arrived(int from, in WireHeader header, byte[] buffer, int length, TimeSpan wall)
    {
        switch (header.Kind)
        {
            case WireMessageKind.Chunk:
            case WireMessageKind.Barrier:
                int tag = header.Kind == WireMessageKind.Barrier ? BarrierTag : header.Tag;
                if (header.Kind == WireMessageKind.Chunk)
                {
                    Queued(length);
                }

                if (!Inbox(from, tag).Writer.TryWrite(new Inbound(buffer, length)))
                {
                    if (header.Kind == WireMessageKind.Chunk)
                    {
                        Queued(-length);
                    }

                    Return(new Inbound(buffer, length));          // the transport is closing
                }
                else if (header.Kind == WireMessageKind.Chunk && Telemetry.IsEnabled(TelemetryLevel.Network))
                {
                    Telemetry.Transfer(new TransferCompleted(Name, from, Rank, header.Tag, length, wall, TimeSpan.Zero));
                }

                return;
            case WireMessageKind.Ack:
                Return(new Inbound(buffer, length));
                Credits(from, header.Tag).Release((long)header.Sequence);   // the peer took a message: its bytes ride in the sequence field
                return;
            case WireMessageKind.Bye:
                Return(new Inbound(buffer, length));
                PeerGone(from, $"rank {from} left the group");
                return;
            case WireMessageKind.Error:
                string text = System.Text.Encoding.UTF8.GetString(buffer, 0, length);
                Return(new Inbound(buffer, length));
                PeerGone(from, $"rank {from} failed: {text}");
                return;
            default:
                Return(new Inbound(buffer, length));               // a heartbeat, or a hello's late copy: nothing to queue
                return;
        }
    }

    /// <summary>Rank <paramref name="from"/> is gone: every wait on it ends with <paramref name="reason"/>; its queued messages stay readable.</summary>
    protected void PeerGone(int from, string reason)
    {
        if (Interlocked.CompareExchange(ref _gone[from], reason, null) is not null)
        {
            return;
        }

        foreach (var ((to, _), credit) in _credits)
        {
            if (to == from)
            {
                credit.Wake();                                     // a sender waiting on it learns it is gone
            }
        }

        for (int pass = 0; pass < 2; pass++)                       // twice: a queue made while the first pass ran
        {
            foreach (var ((source, _), channel) in _inbox)
            {
                if (source == from)
                {
                    channel.Writer.TryComplete();
                }
            }
        }
    }

    /// <summary>Why rank <paramref name="peer"/> is gone, or null while it is connected.</summary>
    protected string? GoneReason(int peer) => Volatile.Read(ref _gone[peer]);

    protected static void Return(Inbound message) => ArrayPool<byte>.Shared.Return(message.Buffer);

    private Channel<Inbound> Inbox(int from, int tag) => _inbox.GetOrAdd((from, tag), static (key, self) =>
    {
        var channel = Channel.CreateUnbounded<Inbound>(new UnboundedChannelOptions { SingleReader = false, SingleWriter = true });
        if (self.GoneReason(key.From) is not null || self._disposed)
        {
            channel.Writer.TryComplete();
        }

        return channel;
    }, this);

    private Inbound Take(int from, int tag)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        Check(from, nameof(from));
        var reader = Inbox(from, tag).Reader;
        if (reader.TryRead(out var message))
        {
            return message;
        }

        using var timeout = new CancellationTokenSource(Timeout);
        try
        {
            return reader.ReadAsync(timeout.Token).AsTask().GetAwaiter().GetResult();
        }
        catch (OperationCanceledException)
        {
            throw TimedOut(from, tag);
        }
        catch (ChannelClosedException)
        {
            throw Gone(from, tag);
        }
    }

    private int Deliver(int from, int tag, Inbound message, Span<byte> bytes)
    {
        try
        {
            if (message.Length > bytes.Length)
            {
                throw new InvalidDataException($"a {message.Length}-byte message from rank {from} (tag {tag}) does not fit rank {Rank}'s {bytes.Length}-byte buffer");
            }

            message.Buffer.AsSpan(0, message.Length).CopyTo(bytes);
            return message.Length;
        }
        finally
        {
            Queued(-message.Length);
            Return(message);
            Acknowledge(from, tag, message.Length);
        }
    }

    // Tells `from` this rank took one of its messages under `tag`, so its credit comes back (nothing when it is gone).
    private void Acknowledge(int from, int tag, int bytes)
    {
        if (_gone[from] is not null || _disposed)
        {
            return;
        }

        var ack = new WireHeader(WireMessageKind.Ack, Rank, tag, (uint)bytes, 0, 1, WireElement.Bytes, ReduceOp.Sum, 0, 0);
        try
        {
            WriteFrame(from, ack, []);
        }
        catch (Exception e) when (e is IOException or InvalidOperationException or TimeoutException or ObjectDisposedException or System.Net.Sockets.SocketException)
        {
            // The peer is going: it waits for no credit any more.
        }
    }

    private void Queued(long bytes)
    {
        long now = Interlocked.Add(ref _queued, bytes);
        long peak = Interlocked.Read(ref _peakQueued);
        while (now > peak)
        {
            long seen = Interlocked.CompareExchange(ref _peakQueued, now, peak);
            if (seen == peak)
            {
                break;
            }

            peak = seen;
        }
    }

    private uint Next(int to, int tag)
    {
        var box = _sequence.GetOrAdd((to, tag), static _ => new StrongBox());
        return (uint)Interlocked.Increment(ref box.Value) - 1;
    }

    private TimeoutException TimedOut(int from, int tag) => new(tag == BarrierTag
        ? $"rank {Rank} waited {Timeout.TotalSeconds:F0} s at a barrier for rank {from}"
        : $"rank {Rank} waited {Timeout.TotalSeconds:F0} s for rank {from} (tag {tag})");

    private Exception Gone(int from, int tag) => _disposed
        ? new ObjectDisposedException(GetType().Name)
        : new InvalidOperationException($"{GoneReason(from) ?? $"rank {from} is gone"} before sending rank {Rank} its message" + (tag == BarrierTag ? " (a barrier)" : $" (tag {tag})"));

    private void Check(int peer, string name)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        if (peer == Rank || (uint)peer >= (uint)WorldSize)
        {
            throw new ArgumentOutOfRangeException(name, peer, $"rank {Rank} reaches the other ranks of 0 to {WorldSize - 1}");
        }
    }

    private sealed class StrongBox
    {
        public long Value;
    }

    /// <summary>What a sender has outstanding with one peer under one tag: messages and bytes the peer has not taken yet.</summary>
    private sealed class Credit
    {
        private int _messages;
        private long _bytes;
        private TaskCompletionSource? _freed;

        public object Gate { get; } = new();

        public bool TryTake(long bytes, int messages, long? budget)
        {
            lock (Gate)
            {
                return TryTakeLocked(bytes, messages, budget);
            }
        }

        // Let go when nothing is outstanding (a message larger than the budget travels alone), else within both bounds.
        public bool TryTakeLocked(long bytes, int messages, long? budget)
        {
            if (_messages > 0 && (_messages >= messages || budget is { } most && _bytes + bytes > most))
            {
                return false;
            }

            _messages++;
            _bytes += bytes;
            return true;
        }

        public void Release(long bytes)
        {
            lock (Gate)
            {
                _messages = Math.Max(0, _messages - 1);
                _bytes = Math.Max(0, _bytes - bytes);
            }

            Wake();
        }

        // A task completed at the next release (or wake): an awaited send waits on it.
        public Task Freed()
        {
            lock (Gate)
            {
                return (_freed ??= new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously)).Task;
            }
        }

        public void Wake()
        {
            TaskCompletionSource? freed;
            lock (Gate)
            {
                Monitor.PulseAll(Gate);
                (freed, _freed) = (_freed, null);
            }

            freed?.TrySetResult();
        }
    }

    /// <summary>The time since <paramref name="start"/> (a <see cref="Stopwatch"/> timestamp), or zero when none was taken.</summary>
    protected static TimeSpan Since(long start) => start == 0 ? TimeSpan.Zero : Stopwatch.GetElapsedTime(start);
}
