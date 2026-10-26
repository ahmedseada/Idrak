// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.ComponentModel;
using System.Runtime.CompilerServices;

namespace Idrak.Maui;

/// <summary>
/// The state of one piece of long work as a view binds it (what, how far, how fast, the time left), as the CLI's progress
/// line shows it. Changed on the UI thread only, by its "maui" reporter, at most once a progress interval.
/// </summary>
public sealed class ProgressItem : INotifyPropertyChanged
{
    /// <inheritdoc />
    public event PropertyChangedEventHandler? PropertyChanged;

    /// <summary>What is in progress ("pull Qwen/Qwen3-0.6B").</summary>
    public string Label { get; private set; } = "";

    /// <summary>The amount done.</summary>
    public long Done { get; private set; }

    /// <summary>The amount when known.</summary>
    public long? Total { get; private set; }

    /// <summary>What the amounts are.</summary>
    public ProgressUnit Unit { get; private set; }

    /// <summary>The share done, from 0 to 1 (0 while the total is unknown: see <see cref="IsIndeterminate"/>).</summary>
    public double Fraction { get; private set; }

    /// <summary>Whether the total is unknown (a bar that only moves).</summary>
    public bool IsIndeterminate { get; private set; } = true;

    /// <summary>The time since the stage started.</summary>
    public TimeSpan Elapsed { get; private set; }

    /// <summary>The amount a second, or 0 before there is one.</summary>
    public double Rate { get; private set; }

    /// <summary>The rate as text ("3.1 MB/s", "2.6/s", "3.6 s each"), or "" before there is one.</summary>
    public string RateText { get; private set; } = "";

    /// <summary>The time left at this rate, when the total is known.</summary>
    public TimeSpan? TimeLeft { get; private set; }

    /// <summary>The time left as text ("4s left"), or "".</summary>
    public string TimeLeftText { get; private set; } = "";

    /// <summary>The amounts as text ("12.3 MB / 27.0 MB", "1,234").</summary>
    public string AmountText { get; private set; } = "";

    /// <summary>The reporter's detail (a loss, the memory in use), or "".</summary>
    public string Detail { get; private set; } = "";

    /// <summary>The whole state as one line, as the CLI shows it ("label  45%  12.3 MB / 27.0 MB  3.1 MB/s  4s left").</summary>
    public string Text { get; private set; } = "";

    /// <summary>The last stage's line, kept when a stage ends (<see cref="IProgressReporter.NextStage"/>), or "".</summary>
    public string PreviousStage { get; private set; } = "";

    /// <summary>Whether the work has ended.</summary>
    public bool IsFinished { get; private set; }

    /// <summary>The message the work ended with, or "".</summary>
    public string Message { get; private set; } = "";

    /// <summary>How many times the state was shown (each a UI update).</summary>
    public int Updates { get; private set; }

    internal void Apply(in ProgressSnapshot s)
    {
        Set(s.Label, Label, v => Label = v);
        Set(s.Done, Done, v => Done = v);
        Set(s.Total, Total, v => Total = v);
        Set(s.Unit, Unit, v => Unit = v);
        double fraction = s.Total is > 0 and var t ? Math.Clamp((double)s.Done / t, 0, 1) : 0;
        Set(fraction, Fraction, v => Fraction = v);
        Set(s.Total is not > 0, IsIndeterminate, v => IsIndeterminate = v);
        Set(s.Elapsed, Elapsed, v => Elapsed = v);
        bool measured = s.Unit != ProgressUnit.Elapsed && s.Elapsed.TotalSeconds >= 0.2 && s.Done > 0;
        double rate = measured ? s.Done / s.Elapsed.TotalSeconds : 0;
        Set(rate, Rate, v => Rate = v);
        Set(rate > 0 ? ProgressText.Rate(rate, s.Unit) : "", RateText, v => RateText = v);
        TimeSpan? left = rate > 0 && s.Total is { } all && all > s.Done ? TimeSpan.FromSeconds((all - s.Done) / rate) : null;
        Set(left, TimeLeft, v => TimeLeft = v);
        Set(left is { } l ? ProgressText.Duration(l) + " left" : "", TimeLeftText, v => TimeLeftText = v);
        Set(s.Unit == ProgressUnit.Elapsed ? ProgressText.Duration(s.Elapsed)
            : s.Total is { } total ? $"{ProgressText.Amount(s.Done, s.Unit)} / {ProgressText.Amount(total, s.Unit)}" : ProgressText.Amount(s.Done, s.Unit), AmountText, v => AmountText = v);
        Set(s.Detail ?? "", Detail, v => Detail = v);
        string text = ProgressText.Format(s.Label, s.Done, s.Total, s.Elapsed, s.Unit);
        Set(s.Detail is { Length: > 0 } d ? text + "  " + d : text, Text, v => Text = v);
        Set(s.PreviousStage ?? PreviousStage, PreviousStage, v => PreviousStage = v);
        Set(s.Finished, IsFinished, v => IsFinished = v);
        Set(s.Message ?? "", Message, v => Message = v);
        Updates++;
        PropertyChanged?.Invoke(this, new PropertyChangedEventArgs(nameof(Updates)));
    }

    private void Set<T>(T value, T current, Action<T> store, [CallerArgumentExpression(nameof(current))] string name = "")
    {
        if (!EqualityComparer<T>.Default.Equals(value, current))
        {
            store(value);
            PropertyChanged?.Invoke(this, new PropertyChangedEventArgs(name));
        }
    }
}

// What one showing reads, taken on the reporting thread (the detail asked for there, once a showing).
internal readonly record struct ProgressSnapshot(string Label, long Done, long? Total, ProgressUnit Unit, TimeSpan Elapsed, string? Detail, string? PreviousStage,
    bool Finished, string? Message);
