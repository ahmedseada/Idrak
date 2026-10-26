// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Collections.ObjectModel;
using Microsoft.Maui.Dispatching;

namespace Idrak.Maui;

/// <summary>
/// What is in progress in the app, for views to bind: one <see cref="ProgressItem"/> a piece of work, made by the "maui"
/// progress reporter (<see cref="MauiProgressReporterKind"/>, registered in <see cref="ProgressReporters"/> by
/// <c>UseIdrak</c>). Every change reaches <see cref="Items"/> and the items on the UI thread, at most once an
/// <see cref="Interval"/> a piece of work: a report between showings stores its amount and nothing else.
/// </summary>
public sealed class ProgressCenter
{
    private readonly TimeProvider _time;

    /// <summary>A center showing at most once an <paramref name="interval"/>.</summary>
    /// <param name="interval">The least time between two showings of one piece of work.</param>
    /// <param name="dispatcher">The UI thread's dispatcher; null: the app's (<c>Application.Current.Dispatcher</c>) when there is one, else the reporting thread itself.</param>
    /// <param name="time">The clock (tests give their own).</param>
    public ProgressCenter(TimeSpan interval, IDispatcher? dispatcher = null, TimeProvider? time = null)
    {
        Interval = interval;
        Dispatcher = dispatcher;
        _time = time ?? TimeProvider.System;
    }

    /// <summary>The least time between two showings of one piece of work (a reporter asked with its own interval keeps that).</summary>
    public TimeSpan Interval
    {
        get;
        set => field = value > TimeSpan.Zero ? value : throw new ArgumentOutOfRangeException(nameof(value), value, "The progress interval is a positive time.");
    }

    /// <summary>The clock its reporters read.</summary>
    internal TimeProvider Time => _time;

    /// <summary>The UI thread's dispatcher, or null for the app's.</summary>
    public IDispatcher? Dispatcher { get; set; }

    /// <summary>The finished items kept in <see cref="Items"/> (the latest ones; 3 by default).</summary>
    public int KeepFinished { get; set; } = 3;

    /// <summary>The work in progress and the latest finished, oldest first; changed on the UI thread only.</summary>
    public ObservableCollection<ProgressItem> Items { get; } = [];

    /// <summary>A reporter showing on this center (what the "maui" kind makes; <see cref="ProgressReporters.Start"/> names it).</summary>
    public IProgressReporter Start(string label, long? total = null, ProgressUnit unit = ProgressUnit.Items, TimeSpan? interval = null) =>
        new MauiProgressReporter(this, new ProgressRequest(label, total, unit, TextWriter.Null, interval), _time);

    /// <summary>
    /// The item <paramref name="reporter"/> shows on, when it is one of this center's (<see cref="Start"/> or the "maui"
    /// kind made it); else null. A page binds it to show the one piece of work it started.
    /// </summary>
    public ProgressItem? ItemOf(IProgressReporter reporter) => reporter is MauiProgressReporter mine && ReferenceEquals(mine.Center, this) ? mine.Item : null;

    internal void Dispatch(Action action)
    {
        var dispatcher = Dispatcher ?? Application.Current?.Dispatcher;
        if (dispatcher is null || !dispatcher.IsDispatchRequired)
        {
            action();
        }
        else
        {
            dispatcher.Dispatch(action);
        }
    }

    // On the UI thread: a new item, the oldest finished ones beyond KeepFinished dropped first.
    internal void Add(ProgressItem item)
    {
        int finished = Items.Count(i => i.IsFinished);
        for (int i = 0; i < Items.Count && finished > Math.Max(0, KeepFinished);)
        {
            if (Items[i].IsFinished)
            {
                Items.RemoveAt(i);
                finished--;
            }
            else
            {
                i++;
            }
        }

        Items.Add(item);
    }

    internal void Remove(ProgressItem item) => Items.Remove(item);
}

/// <summary>The "maui" progress reporter kind: each piece of work an item of the app's <see cref="ProgressCenter"/>.</summary>
public sealed class MauiProgressReporterKind(ProgressCenter center) : IProgressReporterKind
{
    /// <summary>Its name in <see cref="ProgressReporters"/> (and <c>IDRAK_PROGRESS</c>).</summary>
    public const string ReporterName = "maui";

    /// <summary>The center its reporters show on.</summary>
    public ProgressCenter Center { get; } = center ?? throw new ArgumentNullException(nameof(center));

    /// <inheritdoc />
    public string Name => ReporterName;

    /// <inheritdoc />
    public string Summary => "an app's bindable progress (what, how far, speed, time left), updated on the UI thread at most once a progress interval";

    /// <inheritdoc />
    public IReadOnlySet<string> Keys => OptionKeys.None;

    /// <inheritdoc />
    public IProgressReporter Create(ProgressRequest request) => new MauiProgressReporter(Center, request ?? throw new ArgumentNullException(nameof(request)), Center.Time);
}
