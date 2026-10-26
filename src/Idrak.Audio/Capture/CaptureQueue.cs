// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Runtime.CompilerServices;
using System.Threading.Channels;

namespace Idrak.Audio;

/// <summary>
/// What a microphone capture shares: a recording thread of its own (running the driver's recording loop) that puts each
/// block into a queue the moment the system gives it, and <see cref="ReadAsync"/>, which hands the blocks out in order;
/// cancelling the reading stops the thread and gives what it recorded until then.
/// </summary>
internal sealed class CaptureQueue : IAudioInput
{
    private readonly Channel<AudioData> _blocks = Channel.CreateUnbounded<AudioData>(new UnboundedChannelOptions { SingleReader = true, SingleWriter = true });
    private readonly Thread _thread;
    private volatile bool _stopping;
    private bool _read;
    private int _disposed;

    /// <summary>A queue whose thread runs <paramref name="record"/> (until <see cref="Stopping"/>, the device closed before it returns) once <see cref="Start"/> is called.</summary>
    public CaptureQueue(string description, int sampleRate, int channels, Action<CaptureQueue> record)
    {
        Description = description;
        SampleRate = sampleRate;
        Channels = channels;
        _thread = new Thread(() =>
        {
            try
            {
                record(this);
                _blocks.Writer.TryComplete();
            }
            catch (Exception e)
            {
                _blocks.Writer.TryComplete(e);
            }
        }) { IsBackground = true, Name = $"Idrak audio capture ({description})", Priority = ThreadPriority.AboveNormal };
    }

    /// <inheritdoc />
    public string Description { get; }

    /// <inheritdoc />
    public int SampleRate { get; }

    /// <inheritdoc />
    public int Channels { get; }

    /// <summary>Whether the reading was stopped: the recording loop returns soon after.</summary>
    public bool Stopping => _stopping;

    /// <summary>Starts the recording thread (the device already opened).</summary>
    public void Start() => _thread.Start();

    /// <summary>Queues one block of 16-bit interleaved samples (copied).</summary>
    public void Deliver(ReadOnlySpan<byte> pcm16)
    {
        if (pcm16.Length >= 2 * Channels)
        {
            _blocks.Writer.TryWrite(new AudioData(pcm16[..(pcm16.Length / (2 * Channels) * 2 * Channels)].ToArray(), Channels, SampleRate));
        }
    }

    /// <summary>Queues one block of float samples interleaved by frame, as planar floats (one pass over the frames, in order).</summary>
    public void DeliverFloat(ReadOnlySpan<float> interleaved)
    {
        int frames = interleaved.Length / Channels;
        if (frames == 0)
        {
            return;
        }

        var planar = GC.AllocateUninitializedArray<float>(frames * Channels);   // every value written below
        if (Channels == 1)
        {
            interleaved[..frames].CopyTo(planar);
        }
        else
        {
            for (int f = 0, at = 0; f < frames; f++)
            {
                for (int c = 0; c < Channels; c++, at++)
                {
                    planar[c * frames + f] = interleaved[at];
                }
            }
        }

        _blocks.Writer.TryWrite(new AudioData(planar, Channels, SampleRate));
    }

    /// <inheritdoc />
    /// <exception cref="InvalidOperationException">Read a second time, or the device failed while recording (the system's message given).</exception>
    public async IAsyncEnumerable<AudioData> ReadAsync([EnumeratorCancellation] CancellationToken cancellationToken = default)
    {
        if (_read)
        {
            throw new InvalidOperationException("A capture is read once.");
        }

        _read = true;
        using var stopping = cancellationToken.Register(() => _stopping = true);
        var reader = _blocks.Reader;
        while (await reader.WaitToReadAsync(CancellationToken.None).ConfigureAwait(false))   // ends when the thread does
        {
            while (reader.TryRead(out var block))
            {
                yield return block;
            }
        }
    }

    /// <inheritdoc />
    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0)
        {
            return;
        }

        _stopping = true;
        if (_thread.ThreadState != ThreadState.Unstarted)
        {
            _thread.Join();
        }
    }
}
