// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Buffers;

namespace Idrak.Audio;

/// <summary>
/// What a speaker playback shares (the mirror of <see cref="CaptureQueue"/>): the writes convert each block once into the
/// device's samples (<see cref="PlaybackConverter"/>) and put them into a ring of <see cref="AudioPlaybackOptions.BufferSeconds"/>,
/// waiting while it is full; a playing thread of its own runs the driver's loop, which takes the samples out a device
/// block at a time (<see cref="Take"/>) and, when a drain is asked and the ring is empty, lets the device play out what
/// it holds (<see cref="Drained"/>). Nothing is allocated per block on the playing thread.
/// </summary>
internal sealed class PlaybackQueue : IAudioOutput
{
    private readonly object _gate = new();
    private readonly Thread _thread;
    private readonly PlaybackConverter _converter;
    private readonly int _capacity;
    private byte[]? _ring;
    private int _head, _count;
    private volatile bool _stopping;
    private volatile bool _drainPending;
    private Exception? _failure;
    private TaskCompletionSource? _space;
    private int _spaceWanted;
    private TaskCompletionSource? _drain;
    private int _writing;
    private int _disposed;

    /// <summary>
    /// A queue whose thread runs <paramref name="play"/> (until <see cref="Stopping"/>, the device closed before it returns)
    /// once <see cref="Start"/> is called; the device plays <paramref name="format"/> samples interleaved by frame.
    /// </summary>
    public PlaybackQueue(string description, int sampleRate, int channels, PcmSampleFormat format, double bufferSeconds, ResampleOptions? resample,
        Action<PlaybackQueue> play)
    {
        Description = description;
        SampleRate = sampleRate;
        Channels = channels;
        Format = format;
        FrameBytes = channels * (format == PcmSampleFormat.F32LE ? sizeof(float) : sizeof(short));
        _capacity = Math.Max(1, (int)Math.Round(bufferSeconds * sampleRate)) * FrameBytes;
        _ring = ArrayPool<byte>.Shared.Rent(_capacity);
        _converter = new PlaybackConverter(sampleRate, channels, format, resample ?? new ResampleOptions());
        _thread = new Thread(() =>
        {
            try
            {
                play(this);
                Release(new ObjectDisposedException(Description, "The playback was stopped."));
            }
            catch (Exception e)
            {
                Volatile.Write(ref _failure, e);
                Release(new InvalidOperationException($"The playback on {Description} failed: {e.Message}", e));
            }
        }) { IsBackground = true, Name = $"Idrak audio playback ({description})", Priority = ThreadPriority.AboveNormal };
    }

    /// <inheritdoc />
    public string Description { get; }

    /// <inheritdoc />
    public int SampleRate { get; }

    /// <inheritdoc />
    public int Channels { get; }

    /// <summary>The device's samples.</summary>
    public PcmSampleFormat Format { get; }

    /// <summary>The bytes of one frame (every channel's sample).</summary>
    public int FrameBytes { get; }

    /// <summary>The bytes the ring holds: what is written ahead of the device at most.</summary>
    public int Capacity => _capacity;

    /// <summary>Whether the playback was stopped: the driver's loop returns soon after.</summary>
    public bool Stopping => _stopping;

    /// <summary>Whether a drain waits: once <see cref="Take"/> gives nothing, the driver plays out what it holds and calls <see cref="Drained"/>.</summary>
    public bool DrainPending => _drainPending;

    /// <summary>Starts the playing thread (the device already opened).</summary>
    public void Start() => _thread.Start();

    /// <summary>
    /// The playing thread's read: up to <paramref name="destination"/>'s length of whole frames, waiting at most
    /// <paramref name="millisecondsTimeout"/> for any; 0 when none came, at once when stopping or a drain waits.
    /// </summary>
    public int Take(Span<byte> destination, int millisecondsTimeout)
    {
        TaskCompletionSource? wake = null;
        int n;
        lock (_gate)
        {
            if (_count == 0 && !_stopping && !_drainPending)
            {
                Monitor.Wait(_gate, millisecondsTimeout);
            }

            n = Math.Min(_count, destination.Length / FrameBytes * FrameBytes);
            if (n > 0)
            {
                var ring = _ring!;
                int first = Math.Min(n, _capacity - _head);
                ring.AsSpan(_head, first).CopyTo(destination);
                ring.AsSpan(0, n - first).CopyTo(destination[first..]);
                _head = (_head + n) % _capacity;
                _count -= n;
                if (_space is not null && _capacity - _count >= _spaceWanted)
                {
                    (wake, _space) = (_space, null);
                }
            }
        }

        wake?.TrySetResult();                                                // the writer's continuation runs on the pool, not here
        return n;
    }

    /// <summary>The playing thread's word that the device has played everything (after a drain was asked and the ring emptied).</summary>
    public void Drained()
    {
        TaskCompletionSource? done = null;
        lock (_gate)
        {
            if (_count == 0 && _drain is not null)
            {
                (done, _drain) = (_drain, null);
                _drainPending = false;
            }
        }

        done?.TrySetResult();
    }

    /// <inheritdoc />
    public async ValueTask WriteAsync(AudioData block, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(block);
        Enter("write");
        try
        {
            await _converter.WriteAsync(block, this, cancellationToken).ConfigureAwait(false);
        }
        finally
        {
            Interlocked.Exchange(ref _writing, 0);
        }
    }

    /// <inheritdoc />
    public async Task DrainAsync(CancellationToken cancellationToken = default)
    {
        Enter("drain");
        try
        {
            await _converter.FlushAsync(this, cancellationToken).ConfigureAwait(false);
            Task played;
            lock (_gate)
            {
                ThrowIfStopped();
                _drain ??= new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously);
                _drainPending = true;
                played = _drain.Task;
                Monitor.Pulse(_gate);
            }

            await played.WaitAsync(cancellationToken).ConfigureAwait(false);
        }
        finally
        {
            Interlocked.Exchange(ref _writing, 0);
        }
    }

    /// <summary>Queues device bytes (whole frames), waiting while the ring is full.</summary>
    internal async ValueTask EnqueueAsync(ReadOnlyMemory<byte> bytes, CancellationToken cancellationToken)
    {
        while (!bytes.IsEmpty)
        {
            var (put, wait) = Put(bytes.Span);
            bytes = bytes[put..];
            if (wait is not null)
            {
                await wait.WaitAsync(cancellationToken).ConfigureAwait(false);
            }
        }
    }

    // As much of `bytes` as fits; when not all, the task that completes once half the ring (or the rest) is free.
    private (int Put, Task? Wait) Put(ReadOnlySpan<byte> bytes)
    {
        lock (_gate)
        {
            ThrowIfStopped();
            int n = Math.Min(_capacity - _count, bytes.Length);
            if (n > 0)
            {
                var ring = _ring!;
                int tail = (_head + _count) % _capacity;
                int first = Math.Min(n, _capacity - tail);
                bytes[..first].CopyTo(ring.AsSpan(tail));
                bytes[first..n].CopyTo(ring);
                _count += n;
                Monitor.Pulse(_gate);
            }

            if (n == bytes.Length)
            {
                return (n, null);
            }

            _spaceWanted = Math.Min(bytes.Length - n, Math.Max(FrameBytes, _capacity / 2 / FrameBytes * FrameBytes));
            _space = new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously);
            return (n, _space.Task);
        }
    }

    // One write or drain at a time, on an output that still plays.
    private void Enter(string what)
    {
        if (Interlocked.Exchange(ref _writing, 1) != 0)
        {
            throw new InvalidOperationException($"A {what} on {Description} started while another write or drain runs: await each before the next.");
        }

        try
        {
            lock (_gate)
            {
                ThrowIfStopped();
            }
        }
        catch
        {
            Interlocked.Exchange(ref _writing, 0);
            throw;
        }
    }

    private void ThrowIfStopped()
    {
        if (Volatile.Read(ref _failure) is { } failure)
        {
            throw new InvalidOperationException($"The playback on {Description} failed: {failure.Message}", failure);
        }

        if (_stopping)
        {
            throw new ObjectDisposedException(Description, "The playback was stopped.");
        }
    }

    // Ends every wait with `error` (the thread ended, or the output was disposed).
    private void Release(Exception error)
    {
        TaskCompletionSource? space, drain;
        lock (_gate)
        {
            _stopping = true;
            (space, _space) = (_space, null);
            (drain, _drain) = (_drain, null);
        }

        space?.TrySetException(error);
        drain?.TrySetException(error);
    }

    /// <inheritdoc />
    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0)
        {
            return;
        }

        lock (_gate)
        {
            _stopping = true;
            Monitor.PulseAll(_gate);
        }

        Release(new ObjectDisposedException(Description, "The playback was stopped."));
        if (_thread.ThreadState != ThreadState.Unstarted)
        {
            _thread.Join();
        }

        byte[]? ring;
        lock (_gate)
        {
            (ring, _ring) = (_ring, null);                                   // no Put copies once stopping; the thread has ended
        }

        if (ring is not null)
        {
            ArrayPool<byte>.Shared.Return(ring);
        }
    }
}
