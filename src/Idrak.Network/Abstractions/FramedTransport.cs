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
/// </summary>
internal abstract class FramedTransport : ITransport
{
    /// <summary>The tag barrier frames are queued under: below every tag a program or an algorithm uses.</summary>
    protected const int BarrierTag = int.MinValue;

    private readonly ConcurrentDictionary<(int From, int Tag), Channel<Inbound>> _inbox = new();
    private readonly ConcurrentDictionary<(int To, int Tag), StrongBox> _sequence = new();
    private readonly string?[] _gone;
    private volatile bool _disposed;

    protected FramedTransport(string name, int rank, int worldSize, TimeSpan timeout)
    {
        Name = name;
        Rank = rank;
        WorldSize = worldSize;
        Timeout = timeout;
        _gone = new string?[worldSize];
    }

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

        var header = new WireHeader(WireMessageKind.Chunk, Rank, tag, Next(to, tag), 0, 1, WireElement.Bytes, ReduceOp.Sum, 0, (uint)bytes.Length);
        WriteFrame(to, header, bytes);
    }

    public int Receive(int from, Span<byte> bytes, int tag) => Deliver(from, tag, Take(from, tag), bytes);

    public ValueTask SendAsync(int to, ReadOnlyMemory<byte> bytes, int tag, CancellationToken cancellationToken = default)
    {
        try
        {
            cancellationToken.ThrowIfCancellationRequested();
            Send(to, bytes.Span, tag);                             // the frame is written (or queued by the carrier) when Send returns
            return ValueTask.CompletedTask;
        }
        catch (Exception e)
        {
            return ValueTask.FromException(e);
        }
    }

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
                if (!Inbox(from, tag).Writer.TryWrite(new Inbound(buffer, length)))
                {
                    Return(new Inbound(buffer, length));          // the transport is closing
                }
                else if (header.Kind == WireMessageKind.Chunk && Telemetry.IsEnabled(TelemetryLevel.Network))
                {
                    Telemetry.Transfer(new TransferCompleted(Name, from, Rank, header.Tag, length, wall, TimeSpan.Zero));
                }

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
            Return(message);
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

    /// <summary>The time since <paramref name="start"/> (a <see cref="Stopwatch"/> timestamp), or zero when none was taken.</summary>
    protected static TimeSpan Since(long start) => start == 0 ? TimeSpan.Zero : Stopwatch.GetElapsedTime(start);
}
