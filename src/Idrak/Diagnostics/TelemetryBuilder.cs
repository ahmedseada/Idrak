// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

namespace Idrak.Diagnostics;

/// <summary>
/// Collects telemetry hooks and subscribes them together (<see cref="Telemetry.Configure"/>). Each method creates one
/// of the existing hooks with the same parameters and defaults as its constructor; <see cref="Start"/> subscribes them
/// all and returns one session that unsubscribes and flushes everything when disposed.
/// </summary>
/// <example>
/// <code>
/// await using var telemetry = Telemetry.Configure()
///     .Console(TelemetryLevel.Training, epochInterval: 10)
///     .JsonLines("run.jsonl", TelemetryLevel.All &amp; ~TelemetryLevel.Operations)
///     .Record(out var recorder)
///     .Start();
/// </code>
/// </example>
public sealed class TelemetryBuilder
{
    private readonly List<ITelemetryHook> _hooks = [];
    private readonly List<IAsyncDisposable> _owned = [];
    private bool? _synchronizeForTiming;
    private bool _started;

    internal TelemetryBuilder()
    {
    }

    /// <summary><c>new ConsoleLogger(levels, epochInterval, batchInterval, output)</c>.</summary>
    public TelemetryBuilder Console(TelemetryLevel levels = TelemetryLevel.Training, int epochInterval = 1, int batchInterval = 1, TextWriter? output = null) =>
        Hook(new ConsoleLogger(levels, epochInterval, batchInterval, output));

    /// <summary><c>new JsonLinesLogger(path, levels)</c>; the session flushes and closes the file when disposed.</summary>
    public TelemetryBuilder JsonLines(string path, TelemetryLevel levels = TelemetryLevel.Training | TelemetryLevel.Batches | TelemetryLevel.Inference)
    {
        var logger = new JsonLinesLogger(path, levels);
        _owned.Add(logger);
        return Hook(logger);
    }

    /// <summary><c>new MetricsRecorder(levels)</c>, returned through <paramref name="recorder"/>.</summary>
    public TelemetryBuilder Record(out MetricsRecorder recorder, TelemetryLevel levels = TelemetryLevel.Training)
    {
        recorder = new MetricsRecorder(levels);
        return Hook(recorder);
    }

    /// <summary><c>new ChannelTelemetry(levels, capacity)</c>, returned through <paramref name="channel"/> (read <see cref="ChannelTelemetry.Reader"/>).</summary>
    public TelemetryBuilder Channel(out ChannelTelemetry channel, TelemetryLevel levels, int capacity = 100_000)
    {
        channel = new ChannelTelemetry(levels, capacity);
        return Hook(channel);
    }

    /// <summary>Any hook you created yourself (it is subscribed, but not disposed, by the session).</summary>
    public TelemetryBuilder Hook(ITelemetryHook hook)
    {
        ArgumentNullException.ThrowIfNull(hook);
        _hooks.Add(hook);
        return this;
    }

    /// <summary>Sets <see cref="Telemetry.SynchronizeForTiming"/> while the session runs (restored when it ends).</summary>
    public TelemetryBuilder SynchronizeForTiming(bool synchronize = true)
    {
        _synchronizeForTiming = synchronize;
        return this;
    }

    /// <summary>Subscribes every hook (<see cref="Telemetry.Subscribe"/>) and returns the session that ends them.</summary>
    public TelemetrySession Start()
    {
        if (_started)
        {
            throw new InvalidOperationException("This telemetry configuration was already started.");
        }

        _started = true;
        bool previous = Telemetry.SynchronizeForTiming;
        if (_synchronizeForTiming is { } sync)
        {
            Telemetry.SynchronizeForTiming = sync;
        }

        return new TelemetrySession([.. _hooks.Select(Telemetry.Subscribe)], [.. _owned], _synchronizeForTiming is null ? null : previous);
    }
}

/// <summary>Subscribed telemetry hooks; disposing unsubscribes them, then flushes and closes the files the builder opened.</summary>
public sealed class TelemetrySession : IDisposable, IAsyncDisposable
{
    private readonly IDisposable[] _subscriptions;
    private readonly IAsyncDisposable[] _owned;
    private readonly bool? _restoreSynchronize;
    private int _disposed;

    internal TelemetrySession(IDisposable[] subscriptions, IAsyncDisposable[] owned, bool? restoreSynchronize)
    {
        _subscriptions = subscriptions;
        _owned = owned;
        _restoreSynchronize = restoreSynchronize;
    }

    /// <summary>Number of subscribed hooks.</summary>
    public int HookCount => _subscriptions.Length;

    /// <inheritdoc />
    public void Dispose() => DisposeAsync().AsTask().GetAwaiter().GetResult();

    /// <inheritdoc />
    public async ValueTask DisposeAsync()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0)
        {
            return;
        }

        foreach (var s in _subscriptions)
        {
            s.Dispose();
        }

        foreach (var o in _owned)
        {
            await o.DisposeAsync().ConfigureAwait(false);
        }

        if (_restoreSynchronize is { } previous)
        {
            Telemetry.SynchronizeForTiming = previous;
        }
    }
}
