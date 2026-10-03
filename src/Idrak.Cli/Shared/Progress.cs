// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Diagnostics;
using System.Globalization;

namespace Idrak.Cli.Shared;

/// <summary>
/// A progress line for downloads, loading, training and long benchmarks (plans/idrak-cli.md, "Helpers every command
/// shares"): the label, the share done, the amount, the rate and the time left, redrawn in place on the error output
/// (so the standard output stays clean for pipes). Nothing is drawn when the error output is not a terminal, or with
/// <c>--quiet</c>, <c>--json</c> or <c>--plain</c>.
/// </summary>
/// <example><code>
/// using var progress = new ProgressLine(context, "pull model.gguf", total: length, unit: ProgressUnit.Bytes);
/// progress.Report(done);   // as often as convenient: redrawn at most 10 times a second
/// </code></example>
internal sealed class ProgressLine : IProgress<long>, IDisposable
{
    private static readonly TimeSpan Interval = TimeSpan.FromMilliseconds(100);
    private readonly TextWriter _writer;
    private readonly Stopwatch _watch = Stopwatch.StartNew();
    private TimeSpan _lastDraw = TimeSpan.MinValue;
    private int _lastWidth;
    private long _done;
    private bool _finished;
    private Timer? _clock;

    /// <param name="context">The command (its <c>--quiet</c>, <c>--json</c> and error output).</param>
    /// <param name="label">What is in progress ("pull Qwen/Qwen3-0.6B").</param>
    /// <param name="total">The amount when known; null shows the amount and rate without a share or time left.</param>
    /// <param name="unit">How amounts are shown.</param>
    /// <param name="enabled">Tests: draw (true) or not (false) whatever the terminal; null decides as described above.</param>
    public ProgressLine(CommandContext context, string label, long? total = null, ProgressUnit unit = ProgressUnit.Items, bool? enabled = null)
    {
        Label = label;
        Total = total;
        Unit = unit;
        _writer = context.ErrorOutput;
        Enabled = enabled ?? (!context.Quiet && !context.Json && !context.Plain && Terminal.IsTerminal(_writer));
        if (Enabled && unit == ProgressUnit.Elapsed)
        {
            // Work without amounts (loading a model): the time so far, redrawn twice a second.
            _clock = new Timer(_ => Draw(force: true), null, TimeSpan.Zero, TimeSpan.FromMilliseconds(500));
        }
    }

    public string Label { get; set; }

    public long? Total { get; set; }

    public ProgressUnit Unit { get; }

    /// <summary>Whether the line is drawn.</summary>
    public bool Enabled { get; }

    /// <summary>The amount done so far.</summary>
    public long Done => Interlocked.Read(ref _done);

    /// <summary>Sets the amount done and redraws when the last drawing is more than a tenth of a second old.</summary>
    public void Report(long value)
    {
        Interlocked.Exchange(ref _done, value);
        Draw(force: false);
    }

    /// <summary>Adds <paramref name="amount"/> to the amount done.</summary>
    public void Advance(long amount = 1)
    {
        Interlocked.Add(ref _done, amount);
        Draw(force: false);
    }

    /// <summary>Draws the final state and ends the line (with <paramref name="message"/> in place of the progress, when given).</summary>
    public void Finish(string? message = null)
    {
        if (_finished)
        {
            return;
        }

        _finished = true;
        _clock?.Dispose();
        if (!Enabled)
        {
            return;
        }

        lock (_writer)
        {
            string text = message ?? Format(Label, Done, Total, _watch.Elapsed, Unit);
            _writer.Write("\r" + text.PadRight(_lastWidth) + "\n");
            _writer.Flush();
        }
    }

    /// <summary>Removes the line (nothing stays on the screen), for work whose result is reported elsewhere.</summary>
    public void Clear()
    {
        if (_finished)
        {
            return;
        }

        _finished = true;
        _clock?.Dispose();
        if (!Enabled)
        {
            return;
        }

        lock (_writer)
        {
            _writer.Write("\r" + new string(' ', _lastWidth) + "\r");
            _writer.Flush();
        }
    }

    public void Dispose() => Finish();

    /// <summary>
    /// The line's text: "label  45%  12.3 MB / 27.0 MB  3.1 MB/s  4s left" (without a total: "label  12.3 MB  3.1 MB/s").
    /// </summary>
    public static string Format(string label, long done, long? total, TimeSpan elapsed, ProgressUnit unit)
    {
        if (unit == ProgressUnit.Elapsed)
        {
            return $"{label}  {Units.Duration(elapsed)}";
        }

        double seconds = elapsed.TotalSeconds;
        double rate = seconds > 0 ? done / seconds : 0;
        var parts = new List<string> { label };
        if (total is long t && t > 0)
        {
            parts.Add($"{Math.Min(100, done * 100 / t),3}%");
            parts.Add($"{Amount(done, unit)} / {Amount(t, unit)}");
        }
        else
        {
            parts.Add(Amount(done, unit));
        }

        if (seconds >= 0.2 && rate > 0)
        {
            parts.Add($"{Amount((long)rate, unit)}/s");
            if (total is long all && all > done)
            {
                parts.Add($"{Units.Duration(TimeSpan.FromSeconds((all - done) / rate))} left");
            }
        }

        return string.Join("  ", parts);
    }

    /// <summary>An amount as the unit shows it (bytes as "12.3 MB", items as "1,234").</summary>
    public static string Amount(long value, ProgressUnit unit) => unit switch
    {
        ProgressUnit.Bytes => Units.Bytes(value),
        _ => value.ToString("N0", CultureInfo.InvariantCulture),
    };

    private void Draw(bool force)
    {
        if (!Enabled || _finished)
        {
            return;
        }

        var now = _watch.Elapsed;
        if (!force && now - _lastDraw < Interval)
        {
            return;
        }

        lock (_writer)
        {
            if (_finished)
            {
                return;                                                          // finished while the clock was drawing
            }

            _lastDraw = now;
            string text = Format(Label, Done, Total, now, Unit);
            _writer.Write("\r" + text.PadRight(_lastWidth));
            _lastWidth = text.Length;
            _writer.Flush();
        }
    }
}

/// <summary>How a <see cref="ProgressLine"/> shows amounts.</summary>
internal enum ProgressUnit
{
    /// <summary>Counted items ("1,234").</summary>
    Items,

    /// <summary>Bytes ("12.3 MB").</summary>
    Bytes,

    /// <summary>No amounts: the time so far, redrawn while the work runs (loading a model).</summary>
    Elapsed,
}
