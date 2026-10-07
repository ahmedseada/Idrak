// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Diagnostics;
using System.Text.Json.Nodes;

namespace Idrak.Data;

/// <summary>
/// Console output for dataset work: log lines, and one status line redrawn in place as a progress bar (downloads with
/// size, speed and time left; rows with count and rate). When the output goes to a file, the bar becomes a plain line
/// every few seconds. <see cref="CreateDownloader"/> and <see cref="Track"/> connect it to downloads and rows.
/// </summary>
public sealed class ConsoleStatus
{
    private readonly object _lock = new();
    private readonly bool _interactive = !Console.IsOutputRedirected;
    private readonly Stopwatch _sinceDraw = Stopwatch.StartNew();
    private int _shown;

    /// <summary>Prints a line above the status line.</summary>
    public void Log(string line)
    {
        lock (_lock)
        {
            Erase();
            Console.WriteLine(line);
        }
    }

    /// <summary>
    /// Redraws the status line for counted work: the bar and percentage, how many <paramref name="unit"/> are done, still
    /// to do and in total, what is happening now (<paramref name="current"/>), the time elapsed and the estimated time left.
    /// </summary>
    /// <example><c>status.Progress("training", step, totalSteps, clock.Elapsed, $"epoch 1/3, loss {loss:F4}");</c></example>
    public void Progress(string label, long done, long total, TimeSpan elapsed, string? current = null, string unit = "steps") =>
        Bar(label, done, total, elapsed,
            $"{unit}: done {done:N0}, still {Math.Max(0, total - done):N0}, total {total:N0}" + (string.IsNullOrEmpty(current) ? "" : $" | {current}"));

    /// <summary>
    /// Redraws the status line: a bar with the time elapsed and left when <paramref name="total"/> is known, else a count
    /// (at most ten times a second, and once more when the work is done).
    /// </summary>
    public void Bar(string label, long done, long? total, TimeSpan elapsed, string detail)
    {
        lock (_lock)
        {
            bool finished = total is > 0 and var whole && done >= whole;
            if (!finished && _sinceDraw.ElapsedMilliseconds < (_interactive ? 100 : 5000))
            {
                return;
            }

            _sinceDraw.Restart();
            string text;
            if (total is > 0 and var t)
            {
                double fraction = Math.Clamp(done / (double)t, 0, 1);
                int filled = (int)(fraction * 24);
                string eta = finished ? "done" : fraction > 0 ? $"ETA {Time(TimeSpan.FromSeconds(elapsed.TotalSeconds / fraction * (1 - fraction)))}" : "ETA --:--";
                text = $"  {label} [{new string('█', filled)}{new string('░', 24 - filled)}] {fraction,6:P1}  {detail} | elapsed {Time(elapsed)} | {eta}";
            }
            else
            {
                text = $"  {label}  {detail} | elapsed {Time(elapsed)}";
            }

            if (!_interactive)
            {
                Console.WriteLine(text);
                return;
            }

            int width = Math.Max(20, SafeWidth() - 1);
            text = text.Length > width ? text[..width] : text;
            Console.Write("\r" + text.PadRight(Math.Max(_shown, text.Length)));
            _shown = text.Length;
        }
    }

    /// <summary>Keeps the status line as it is (the finished bar) and moves below it; the next status starts a new line.</summary>
    public void Finish()
    {
        lock (_lock)
        {
            if (_shown > 0)
            {
                Console.WriteLine();
                _shown = 0;
            }
        }
    }

    /// <summary>Removes the status line.</summary>
    public void Clear()
    {
        lock (_lock)
        {
            Erase();
        }
    }

    private void Erase()
    {
        if (_shown > 0)
        {
            Console.Write("\r" + new string(' ', _shown) + "\r");
            _shown = 0;
        }
    }

    private static string Time(TimeSpan time) =>
        time.TotalHours >= 1 ? $"{(int)time.TotalHours}:{time.Minutes:D2}:{time.Seconds:D2}" : $"{time.Minutes:D2}:{time.Seconds:D2}";

    private static int SafeWidth()
    {
        try
        {
            return Console.WindowWidth;
        }
        catch (IOException)
        {
            return 120;
        }
    }

    /// <summary>A downloader that logs its steps here and shows each download as a progress bar.</summary>
    public Downloader CreateDownloader(HttpClient? http = null, string? cacheFolder = null, bool refresh = false) => new(http, cacheFolder)
    {
        Refresh = refresh,
        Log = line => Log("  " + line),
        Progress = new Reporter<DownloadProgress>(p =>
        {
            if (p.Completed)
            {
                Clear();
                Log($"  downloaded {p.File} ({Downloader.Size(p.Received)} in {p.Elapsed.TotalSeconds:F1} s, {p.BytesPerSecond / (1 << 20):F1} MB/s)");
            }
            else
            {
                Bar(p.File, p.Received, p.Total, p.Elapsed,
                    Downloader.Size(p.Received) + (p.Total is { } t ? " / " + Downloader.Size(t) : "") + $"  {p.BytesPerSecond / (1 << 20):F1} MB/s");
            }
        }),
    };

    /// <summary>The rows, passed through while a live count and rate (and a bar, when <paramref name="total"/> is known) are shown.</summary>
    public IEnumerable<JsonObject> Track(IEnumerable<JsonObject> rows, string label, long? total = null, Func<string>? extra = null)
    {
        var clock = Stopwatch.StartNew();
        long count = 0;
        foreach (var row in rows)
        {
            count++;
            if ((count & 63) == 0)
            {
                Bar(label, count, total, clock.Elapsed,
                    $"{count:N0}{(total is { } t ? $" / {t:N0}" : "")} rows  {count / Math.Max(clock.Elapsed.TotalSeconds, 1e-3):N0} rows/s{extra?.Invoke()}");
            }

            yield return row;
        }

        Clear();
    }

    private sealed class Reporter<T>(Action<T> report) : IProgress<T>
    {
        public void Report(T value) => report(value);
    }
}
