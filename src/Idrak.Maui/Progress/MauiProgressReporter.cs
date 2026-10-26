// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

namespace Idrak.Maui;

// "maui": one ProgressItem of a ProgressCenter. A report stores the amount and compares the clock with the last showing;
// at most once an interval it takes a snapshot (the detail asked for then, on the reporting thread) and hands the item one
// update to the UI thread. While an update waits there, a later snapshot replaces the earlier one instead of queueing
// another, so a fast reporter never floods the UI thread.
internal sealed class MauiProgressReporter : IProgressReporter
{
    private const long Never = long.MinValue;
    private readonly ProgressCenter _center;
    private readonly TimeProvider _time;
    private readonly TimeSpan? _interval;
    private readonly ProgressItem _item = new();
    private readonly Action _publish;
    private readonly Lock _gate = new();
    private readonly ITimer? _clock;
    private long _started;
    private long _last = Never;
    private long _done;
    private int _scheduled;
    private volatile bool _finished;
    private ProgressSnapshot _snapshot;
    private string? _previousStage;

    public MauiProgressReporter(ProgressCenter center, ProgressRequest request, TimeProvider time)
    {
        _center = center;
        _time = time;
        _interval = request.Interval;
        Label = request.Label;
        Total = request.Total;
        Unit = request.Unit;
        _started = time.GetTimestamp();
        _publish = Publish;
        center.Dispatch(() => center.Add(_item));
        if (request.Unit == ProgressUnit.Elapsed)
        {
            // Work without amounts (loading a model): the time so far, shown once an interval.
            _clock = time.CreateTimer(_ => Touch(force: true), null, TimeSpan.Zero, Interval);
        }
    }

    /// <summary>The item this reporter shows on (tests and views that hold one piece of work).</summary>
    public ProgressItem Item => _item;

    /// <summary>The center this reporter shows on.</summary>
    public ProgressCenter Center => _center;

    public string Label { get; set; }

    public long? Total { get; set; }

    public ProgressUnit Unit { get; set; }

    public long Done => Interlocked.Read(ref _done);

    public bool Enabled => true;

    public Func<string>? Detail { get; set; }

    private TimeSpan Interval => _interval ?? _center.Interval;

    public void Report(long value)
    {
        Interlocked.Exchange(ref _done, value);
        Touch(force: false);
    }

    public void Advance(long amount = 1)
    {
        Interlocked.Add(ref _done, amount);
        Touch(force: false);
    }

    public void Show() => Touch(force: true);

    public void NextStage(string label, long? total, ProgressUnit unit, string? message = null, bool keepLine = true)
    {
        ArgumentNullException.ThrowIfNull(label);
        if (_finished)
        {
            return;
        }

        lock (_gate)
        {
            if (keepLine)
            {
                _previousStage = message ?? ProgressText.Format(Label, Done, Total, _time.GetElapsedTime(_started), Unit);
            }

            Label = label;
            Total = total;
            Unit = unit;
            Interlocked.Exchange(ref _done, 0);
            _started = _time.GetTimestamp();
        }

        Touch(force: true);
    }

    public void Erase()
    {
        // Nothing is drawn among other output: the item is a view of its own.
    }

    public void Finish(string? message = null) => End(message, remove: false);

    public void Clear() => End(null, remove: true);

    public void Dispose() => Finish();

    private void End(string? message, bool remove)
    {
        if (_finished)
        {
            return;
        }

        _clock?.Dispose();
        lock (_gate)
        {
            _snapshot = Take(_time.GetTimestamp(), finished: true, message);
            _finished = true;
        }

        _center.Dispatch(() =>
        {
            Publish();
            if (remove)
            {
                _center.Remove(_item);
            }
        });
    }

    private void Touch(bool force)
    {
        if (_finished)
        {
            return;
        }

        long now = _time.GetTimestamp();
        long last = Interlocked.Read(ref _last);
        if (!force && last != Never && _time.GetElapsedTime(last, now) < Interval)
        {
            return;                                                             // between showings: the amount is stored, nothing else
        }

        if (Interlocked.CompareExchange(ref _last, now, last) != last && !force)
        {
            return;                                                             // another thread took this showing
        }

        lock (_gate)
        {
            if (_finished)
            {
                return;
            }

            _snapshot = Take(now, finished: false, null);
        }

        if (Interlocked.Exchange(ref _scheduled, 1) == 0)
        {
            _center.Dispatch(_publish);                                         // one update waits on the UI thread at a time
        }
    }

    private ProgressSnapshot Take(long now, bool finished, string? message) =>
        new(Label, Done, Total, Unit, _time.GetElapsedTime(_started, now), Detail?.Invoke(), _previousStage, finished, message);

    // On the UI thread: the latest snapshot onto the item.
    private void Publish()
    {
        Volatile.Write(ref _scheduled, 0);
        ProgressSnapshot snapshot;
        lock (_gate)
        {
            snapshot = _snapshot;
        }

        _item.Apply(snapshot);
    }
}
