// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Runtime.CompilerServices;
using Idrak.Audio.Abstractions;
using Idrak.Maui.Abstractions;
using Idrak.Maui.Controls;

namespace Idrak.Maui;

/// <summary>
/// The app's speaker (<see cref="ISoundPlayer"/>) over the library's <see cref="AudioPlaybacks"/>: the playback the request
/// names, else the first that can run here (an app's before the library's; aaudio on Android). One sound at a time. The
/// levels of what it plays (<see cref="Levels"/>) are made as each block is handed to the output and shown when it plays.
/// </summary>
public sealed class SoundPlayer : ISoundPlayer, IDisposable, ILiveLevelSource
{
    private readonly Lock _gate = new();
    private readonly TimeProvider _time;
    private readonly IdrakMauiOptions? _options;
    private Run? _current;

    /// <summary>A player timing its plays on <paramref name="time"/> (the system's clock when null); <paramref name="options"/> give its levels' window.</summary>
    public SoundPlayer(TimeProvider? time = null, IdrakMauiOptions? options = null)
    {
        _time = time ?? TimeProvider.System;
        _options = options;
    }

    /// <summary>The levels of the sound playing, each shown when its sound plays (paced from the first block written), for a meter to follow.</summary>
    public LiveLevels Levels { get; } = new();

    /// <inheritdoc />
    public bool IsPlaying
    {
        get
        {
            lock (_gate)
            {
                return _current is not null;
            }
        }
    }

    /// <inheritdoc />
    public string? Playing
    {
        get
        {
            lock (_gate)
            {
                return _current?.Label;
            }
        }
    }

    /// <inheritdoc />
    public Task<SoundPlayback> PlayAsync(AudioData audio, SoundPlayRequest? request = null, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(audio);
        return PlayAsync(One(audio), request, cancellationToken);
    }

    /// <inheritdoc />
    public async Task<SoundPlayback> PlayAsync(IAsyncEnumerable<AudioData> blocks, SoundPlayRequest? request = null, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(blocks);
        request ??= new SoundPlayRequest();
        request.Output.Check();
        using var stop = CancellationTokenSource.CreateLinkedTokenSource(cancellationToken);
        var done = new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously);
        var run = new Run(stop, done.Task, request.Label);
        Run? previous;
        lock (_gate)
        {
            (previous, _current) = (_current, run);
        }

        try
        {
            if (previous is not null)
            {
                previous.Cancel();
                await previous.Done.ConfigureAwait(false);                       // one sound at a time: the one before has let the device go
            }

            return await PlayCoreAsync(blocks, request, stop.Token).ConfigureAwait(false);
        }
        finally
        {
            lock (_gate)
            {
                if (ReferenceEquals(_current, run))
                {
                    _current = null;
                }
            }

            run.Ended();
            done.TrySetResult();
        }
    }

    /// <inheritdoc />
    public bool Stop()
    {
        Run? current;
        lock (_gate)
        {
            current = _current;
        }

        current?.Cancel();
        return current is not null;
    }

    /// <summary>Stops what plays.</summary>
    public void Dispose() => Stop();

    private async Task<SoundPlayback> PlayCoreAsync(IAsyncEnumerable<AudioData> blocks, SoundPlayRequest request, CancellationToken token)
    {
        long started = _time.GetTimestamp();
        var playback = request.Playback is { Length: > 0 } name ? AudioPlaybacks.Get(name) : AudioPlaybacks.ForThisMachine();
        if (!playback.IsAvailable(out string reason))
        {
            throw new NotSupportedException($"The {playback.Name} playback cannot play here: {reason}.");
        }

        var enumerator = blocks.GetAsyncEnumerator(token);
        IAudioOutput? output = null;
        ValueTask<bool> next = default;
        bool pending = false, completed = false;
        int written = 0;
        double seconds = 0;
        TimeSpan? first = null;
        string description = playback.Name;
        int rate = 0, channels = 0;
        try
        {
            if (token.IsCancellationRequested)
            {
                return new SoundPlayback(request.Label, description, 0, 0, 0, 0, null, _time.GetElapsedTime(started), false);
            }

            next = enumerator.MoveNextAsync();                                   // the first block is made while the output opens
            pending = true;
            output = playback.Open(request.Output);
            (description, rate, channels) = (output.Description, output.SampleRate, output.Channels);
            Levels.Reset(_options?.LiveLevelWindow.TotalSeconds is > 0 and var window ? window : null);
            Levels.Pace(System.Diagnostics.Stopwatch.GetTimestamp());
            while (true)
            {
                pending = false;
                if (!await next.ConfigureAwait(false))
                {
                    break;
                }

                var block = enumerator.Current;
                next = enumerator.MoveNextAsync();                               // the next block is made while this one is written
                pending = true;
                var writing = output.WriteAsync(block, token);                   // the write starts first: the levels never delay the sound
                Levels.Add(block.Samples, block.SampleRate, block.Channels);     // made while it queues, shown as it plays
                await writing.ConfigureAwait(false);
                first ??= _time.GetElapsedTime(started);
                written++;
                seconds += block.Duration.TotalSeconds;
            }

            await output.DrainAsync(token).ConfigureAwait(false);
            completed = true;
        }
        catch (OperationCanceledException) when (token.IsCancellationRequested)
        {
            // Stopped: the sound is cut below.
        }
        catch (ObjectDisposedException) when (token.IsCancellationRequested)
        {
            // Stopped while the output closed.
        }
        finally
        {
            output?.Dispose();                                                   // stops the sound at once when not drained
            if (pending)
            {
                try
                {
                    await next.ConfigureAwait(false);                            // the block asked for before the end, let finish before the source is closed
                }
                catch (Exception e) when (e is OperationCanceledException || token.IsCancellationRequested)
                {
                    // Cancelled with the play.
                }
            }

            await enumerator.DisposeAsync().ConfigureAwait(false);
        }

        return new SoundPlayback(request.Label, description, rate, channels, written, seconds, first, _time.GetElapsedTime(started), completed);
    }

    private static async IAsyncEnumerable<AudioData> One(AudioData audio, [EnumeratorCancellation] CancellationToken cancellationToken = default)
    {
        cancellationToken.ThrowIfCancellationRequested();
        await Task.CompletedTask.ConfigureAwait(false);
        yield return audio;
    }

    // One play: how to stop it, and when it has let the device go.
    private sealed class Run(CancellationTokenSource stop, Task done, string label)
    {
        private int _ended;

        public Task Done { get; } = done;

        public string Label { get; } = label;

        public void Cancel()
        {
            if (Volatile.Read(ref _ended) != 0)
            {
                return;
            }

            try
            {
                stop.Cancel();
            }
            catch (ObjectDisposedException)
            {
                // It ended meanwhile.
            }
        }

        public void Ended() => Volatile.Write(ref _ended, 1);
    }
}
