// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Globalization;

namespace Idrak.Maui.ViewModels;

/// <summary>One lab setting on a page (<see cref="ILabSetting"/>): its title, help and value; an empty value is the default (its placeholder).</summary>
public sealed class LabSettingEntry : ViewModel
{
    private string _text;
    private string _error = "";

    /// <summary>The entry of <paramref name="setting"/> showing <paramref name="text"/>.</summary>
    public LabSettingEntry(ILabSetting setting, string text)
    {
        Setting = setting ?? throw new ArgumentNullException(nameof(setting));
        _text = text ?? "";
        Choices = setting.Choices;
    }

    /// <summary>The setting.</summary>
    public ILabSetting Setting { get; }

    /// <summary>Its key.</summary>
    public string Key => Setting.Key;

    /// <summary>Its title.</summary>
    public string Title => Setting.Title;

    /// <summary>Its help.</summary>
    public string Summary => Setting.Summary;

    /// <summary>The default, shown in an empty field.</summary>
    public string Placeholder => Setting.Placeholder;

    /// <summary>Edited as text (a field).</summary>
    public bool IsEntry => Setting.Kind != LabSettingKind.Choice;

    /// <summary>Edited as a choice (a picker).</summary>
    public bool IsChoice => Setting.Kind == LabSettingKind.Choice;

    /// <summary>The keyboard for the field.</summary>
    public Keyboard Keyboard => Setting.Kind == LabSettingKind.Number ? Keyboard.Numeric : Keyboard.Plain;

    /// <summary>The values to choose from, the first standing for the default.</summary>
    public IReadOnlyList<string> Choices { get; }

    /// <summary>The value as text ("" for the default).</summary>
    public string Text
    {
        get => _text;
        set
        {
            if (SetProperty(ref _text, value ?? ""))
            {
                OnPropertyChanged(nameof(SelectedChoice));
            }
        }
    }

    /// <summary>The value as a choice (the first choice standing for "").</summary>
    public string? SelectedChoice
    {
        get => Text.Length == 0 && Choices.Count > 0 ? Choices[0] : Text;
        set => Text = value is null || Choices.Count > 0 && value == Choices[0] ? "" : value;
    }

    /// <summary>Why the value was refused, or "".</summary>
    public string Error
    {
        get => _error;
        set
        {
            if (SetProperty(ref _error, value ?? ""))
            {
                OnPropertyChanged(nameof(HasError));
            }
        }
    }

    /// <summary>Whether <see cref="Error"/> has something to show.</summary>
    public bool HasError => Error.Length > 0;

    /// <summary>
    /// Sets each of <paramref name="entries"/> (settings of <typeparamref name="TOptions"/>) into <paramref name="options"/>
    /// (a copy of the lab's); the copy, or null when one is refused (each refused entry says why).
    /// </summary>
    internal static TOptions? Apply<TOptions>(IEnumerable<LabSettingEntry> entries, TOptions options) where TOptions : class
    {
        bool valid = true;
        foreach (var entry in entries)
        {
            try
            {
                ((LabSetting<TOptions>)entry.Setting).Set(options, entry.Text);
                entry.Error = "";
            }
            catch (FormatException e)
            {
                entry.Error = e.Message;
                valid = false;
            }
        }

        return valid ? options : null;
    }
}

/// <summary>
/// A page's showings while work runs on worker threads: <see cref="Touch"/> from any thread hands <c>publish</c> to the UI
/// thread at most once a progress interval (<see cref="ProgressCenter.Interval"/>; a forced one at once), one waiting there
/// at a time; a touch skipped as too early is shown by a timer once due. Between showings a touch costs a clock read.
/// </summary>
internal sealed class UiBatcher : IDisposable
{
    private const long Never = long.MinValue;
    private readonly ProgressCenter _progress;
    private readonly Action _publish;
    private readonly Action _dispatched;
    private readonly Lock _gate = new();
    private ITimer? _trailing;
    private long _last = Never;
    private int _scheduled;
    private volatile bool _running;

    public UiBatcher(ProgressCenter progress, Action publish)
    {
        _progress = progress;
        _publish = publish;
        _dispatched = Dispatched;
    }

    /// <summary>The showings handed to the UI thread so far.</summary>
    public int Dispatches { get; private set; }

    /// <summary>Showings may be made from now on.</summary>
    public void Start()
    {
        Interlocked.Exchange(ref _last, Never);
        _running = true;
    }

    /// <summary>No more showings (the caller shows the last values itself).</summary>
    public void End()
    {
        _running = false;
        lock (_gate)
        {
            _trailing?.Dispose();
            _trailing = null;
        }
    }

    /// <summary>A showing is wanted: made now when one is due (or <paramref name="force"/>), else by a timer once due.</summary>
    public void Touch(bool force = false)
    {
        if (!_running)
        {
            return;
        }

        var time = _progress.Time;
        long now = time.GetTimestamp();
        long last = Interlocked.Read(ref _last);
        var interval = _progress.Interval;
        if (!force && last != Never && time.GetElapsedTime(last, now) is var since && since < interval)
        {
            lock (_gate)
            {
                _trailing ??= time.CreateTimer(_ => Trail(), null, interval - since, Timeout.InfiniteTimeSpan);
            }

            return;
        }

        if (Interlocked.CompareExchange(ref _last, now, last) != last && !force)
        {
            return;                                                             // another thread took this showing
        }

        if (Interlocked.Exchange(ref _scheduled, 1) == 0)
        {
            Dispatches++;
            _progress.Dispatch(_dispatched);
        }
    }

    public void Dispose() => End();

    private void Dispatched()
    {
        Volatile.Write(ref _scheduled, 0);
        _publish();
    }

    private void Trail()
    {
        lock (_gate)
        {
            _trailing?.Dispose();
            _trailing = null;
        }

        Touch(force: true);
    }
}

/// <summary>The text the audio labs show: seconds, times, levels, rates.</summary>
internal static class AudioText
{
    public static string Seconds(double seconds) => string.Create(CultureInfo.InvariantCulture, $"{seconds:0.0} s");

    public static string Milliseconds(double milliseconds) => string.Create(CultureInfo.InvariantCulture, $"{milliseconds:0.#} ms");

    public static string Time(TimeSpan time) => time.TotalSeconds >= 10
        ? string.Create(CultureInfo.InvariantCulture, $"{time.TotalSeconds:0.0} s")
        : string.Create(CultureInfo.InvariantCulture, $"{time.TotalMilliseconds:0} ms");

    public static string Decibels(double decibels) => double.IsFinite(decibels) ? string.Create(CultureInfo.InvariantCulture, $"{decibels:0.0} dB") : "silent";

    // The real-time factor and what it means ("0.25, 4.0x faster than real time").
    public static string Factor(double factor) => !double.IsFinite(factor) || factor <= 0 ? "-"
        : factor < 1 ? string.Create(CultureInfo.InvariantCulture, $"{factor:0.###} ({1 / factor:0.0}x faster than real time)")
        : string.Create(CultureInfo.InvariantCulture, $"{factor:0.###} ({factor:0.0}x slower than real time)");

    public static string Clock(TimeSpan time) => time.ToString(time.TotalHours >= 1 ? @"h\:mm\:ss" : @"mm\:ss", CultureInfo.InvariantCulture);
}
