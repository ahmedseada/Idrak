// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Diagnostics;
using System.Numerics;
using Idrak.Maui.Abstractions;

namespace Idrak.Maui.Controls;

/// <summary>
/// The levels of live sound for drawing, handed from the thread that hears (or plays) it to the UI thread without a lock:
/// the producer adds its samples as they come (<see cref="Add"/>), and a level (0 to 1, from <see cref="FloorDecibels"/> up
/// to full scale) is made for each <see cref="WindowSeconds"/> of sound, however the blocks are cut; the views read the new
/// levels once a display frame (<see cref="Read"/>). One producer at a time, any number of readers. A paced feed
/// (<see cref="Pace"/>, for sound that is made ahead of the speaker) shows each level when its sound plays, not when it was
/// made. Nothing is allocated after construction.
/// </summary>
public sealed class LiveLevels
{
    private readonly float[] _levels;
    private readonly long[] _due;
    private readonly int _mask;
    private long _written;
    private int _generation;
    private double _sum;
    private long _taken;
    private long _windowSamples;
    private int _rate;
    private int _channels;
    private long _paceStart;
    private long _paceIndex;
    private bool _paced;

    /// <summary>A feed keeping the latest <paramref name="capacity"/> levels (rounded up to a power of two), one each <paramref name="windowSeconds"/>.</summary>
    public LiveLevels(int capacity = 512, double windowSeconds = 0.05, double floorDecibels = -60)
    {
        ArgumentOutOfRangeException.ThrowIfLessThan(capacity, 8);
        int size = (int)BitOperations.RoundUpToPowerOf2((uint)capacity);
        _levels = new float[size];
        _due = new long[size];
        _mask = size - 1;
        WindowSeconds = windowSeconds;
        FloorDecibels = floorDecibels;
    }

    /// <summary>The sound one level stands for, in seconds (taken at the next <see cref="Reset"/> or the next sample rate).</summary>
    public double WindowSeconds
    {
        get;
        set => field = double.IsFinite(value) && value > 0 ? value : throw new ArgumentOutOfRangeException(nameof(value), value, "A level's window is a positive time.");
    }

    /// <summary>The level drawn as empty, in decibels of full scale (negative).</summary>
    public double FloorDecibels
    {
        get;
        set => field = double.IsFinite(value) && value < 0 ? value : throw new ArgumentOutOfRangeException(nameof(value), value, "The floor is a negative level in decibels.");
    }

    /// <summary>How many levels were made since the last <see cref="Reset"/>.</summary>
    public long Written => Volatile.Read(ref _written);

    /// <summary>Raised by one each <see cref="Reset"/>: a reader that sees it change starts again from nothing.</summary>
    public int Generation => Volatile.Read(ref _generation);

    /// <summary>The latest level made (0 before any).</summary>
    public float Latest
    {
        get
        {
            long written = Volatile.Read(ref _written);
            return written == 0 ? 0 : _levels[(written - 1) & _mask];
        }
    }

    /// <summary>
    /// Starts afresh (a new recording, a new sound): the levels made are forgotten, the readers start again
    /// (<see cref="Generation"/>), and the next level is the first. Called by the producer before it adds, or while none adds.
    /// </summary>
    public void Reset(double? windowSeconds = null)
    {
        if (windowSeconds is { } window)
        {
            WindowSeconds = window;
        }

        _sum = 0;
        _taken = 0;
        _rate = 0;
        _paced = false;
        Volatile.Write(ref _written, 0);
        Interlocked.Increment(ref _generation);
    }

    /// <summary>
    /// Shows each level from now on when its sound plays: the first sound added after this plays at
    /// <paramref name="startTimestamp"/> (a <see cref="Stopwatch"/> timestamp), the next ones one after another.
    /// </summary>
    public void Pace(long startTimestamp)
    {
        _paceStart = startTimestamp;
        _paceIndex = _written;
        _paced = true;
    }

    /// <summary>Adds sound heard or played: interleaved <paramref name="samples"/> of <paramref name="channels"/> channels at <paramref name="sampleRate"/>.</summary>
    public void Add(ReadOnlySpan<float> samples, int sampleRate, int channels = 1)
    {
        if (samples.IsEmpty || sampleRate <= 0 || channels <= 0)
        {
            return;
        }

        if (sampleRate != _rate || channels != _channels)
        {
            _rate = sampleRate;
            _channels = channels;
            _windowSamples = Math.Max(channels, (long)Math.Round(WindowSeconds * sampleRate) * channels);
            _sum = 0;
            _taken = 0;
        }

        while (!samples.IsEmpty)
        {
            int take = (int)Math.Min(samples.Length, _windowSamples - _taken);
            _sum += SumOfSquares(samples[..take]);
            _taken += take;
            samples = samples[take..];
            if (_taken >= _windowSamples)
            {
                double decibels = 10 * Math.Log10(Math.Max(_sum / _taken, 1e-12));
                Emit((float)Math.Clamp(1 - decibels / FloorDecibels, 0, 1));
                _sum = 0;
                _taken = 0;
            }
        }
    }

    /// <summary>Adds one level (0 to 1) as it is, shown at once.</summary>
    public void AddLevel(double level) => Emit(double.IsFinite(level) ? (float)Math.Clamp(level, 0, 1) : 0f);

    /// <summary>
    /// Copies the levels made since <paramref name="cursor"/> that are due by <paramref name="now"/> (a <see cref="Stopwatch"/>
    /// timestamp; every one for a feed that is not paced) into <paramref name="into"/>, the newest when there are more than it
    /// holds, and moves the cursor past them; returns how many it copied.
    /// </summary>
    public int Read(ref long cursor, Span<float> into, long now)
    {
        long written = Volatile.Read(ref _written);
        if (cursor > written)
        {
            cursor = 0;                                                         // reset meanwhile: the reader's generation check clears its own
        }

        long end = written;
        if (_paced)
        {
            while (end > cursor && _due[(end - 1) & _mask] > now)
            {
                end--;                                                          // not played yet
            }
        }

        long start = Math.Max(cursor, Math.Max(end - _levels.Length, end - into.Length));
        int count = (int)Math.Max(0, end - start);
        for (int i = 0; i < count; i++)
        {
            into[i] = _levels[(start + i) & _mask];
        }

        cursor = Math.Max(cursor, end);
        return count;
    }

    private void Emit(float level)
    {
        long index = _written;                                                  // the producer's own counter
        _levels[index & _mask] = level;
        _due[index & _mask] = _paced ? _paceStart + (long)((index - _paceIndex) * WindowSeconds * Stopwatch.Frequency) : 0;
        Volatile.Write(ref _written, index + 1);                                // the level, then its count: a reader sees both
    }

    // The sum of the squares in one pass over the samples where they are, a vector main loop and a scalar tail.
    private static double SumOfSquares(ReadOnlySpan<float> samples)
    {
        double sum = 0;
        int i = 0;
        if (Vector.IsHardwareAccelerated && samples.Length >= Vector<float>.Count)
        {
            var acc = Vector<float>.Zero;
            for (; i <= samples.Length - Vector<float>.Count; i += Vector<float>.Count)
            {
                var v = new Vector<float>(samples[i..]);
                acc += v * v;
            }

            sum = Vector.Sum(acc);
        }

        for (; i < samples.Length; i++)
        {
            sum += (double)samples[i] * samples[i];
        }

        return sum;
    }
}

/// <summary>Something that hears or plays live sound and hands its levels to the views (<see cref="LiveLevels"/>).</summary>
public interface ILiveLevelSource
{
    /// <summary>The levels of what it hears or plays now.</summary>
    LiveLevels Levels { get; }
}

/// <summary>
/// The "timer" frame clock: a dispatcher timer on the UI thread at the display's refresh rate (MAUI's display information;
/// the rate asked for when it is lower). It ticks only while a view is subscribed.
/// </summary>
public sealed class TimerFrameClock : IFrameClock
{
    /// <summary>The rate used where the platform does not say its display's.</summary>
    public static double FallbackRefreshRate { get; set; } = 60;

    /// <inheritdoc />
    public string Name => FrameClocks.Timer;

    /// <inheritdoc />
    public string Summary => "a dispatcher timer on the UI thread at the display's refresh rate";

    /// <inheritdoc />
    public bool IsAvailable(out string reason)
    {
        reason = "";
        return true;
    }

    /// <inheritdoc />
    public IDisposable Subscribe(Action frame, double framesPerSecond = 0)
    {
        ArgumentNullException.ThrowIfNull(frame);
        var dispatcher = Dispatcher.GetForCurrentThread() ?? Application.Current?.Dispatcher;
        if (dispatcher is null)
        {
            return Stopped.Instance;                                            // no UI thread (the tests): nothing ticks
        }

        IDispatcherTimer timer;
        try
        {
            timer = dispatcher.CreateTimer();
        }
        catch (NotSupportedException)
        {
            return Stopped.Instance;
        }

        double rate = DisplayRate();
        if (framesPerSecond > 0)
        {
            rate = Math.Min(rate, framesPerSecond);
        }

        timer.Interval = TimeSpan.FromSeconds(1 / rate);
        timer.IsRepeating = true;
        timer.Tick += (_, _) => frame();
        timer.Start();
        return new Ticking(timer);
    }

    private static double DisplayRate()
    {
        try
        {
            float rate = DeviceDisplay.Current.MainDisplayInfo.RefreshRate;
            return rate > 1 ? rate : FallbackRefreshRate;
        }
        catch (Exception e) when (e is NotImplementedException or NotSupportedException or InvalidOperationException)
        {
            return FallbackRefreshRate;                                         // no platform: the fallback
        }
    }

    private sealed class Ticking(IDispatcherTimer timer) : IDisposable
    {
        public void Dispose() => timer.Stop();
    }

    private sealed class Stopped : IDisposable
    {
        public static readonly Stopped Instance = new();

        public void Dispose()
        {
        }
    }
}
