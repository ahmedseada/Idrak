// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Text.Json.Nodes;
using Idrak.Datasets;

namespace Idrak.Cli.Shared;

/// <summary>
/// Where the fine-tuning and dataset tools (<c>idrak tune</c>, <c>idrak data</c> and the <c>idrak-tune</c> and
/// <c>idrak-data</c> forwarders, which compile the same code) write: their output and error streams, the input for
/// interactive chat, and the status line. <see cref="Standard"/> is the console with its live progress bar, as the old
/// tools had it; <c>idrak</c> passes its own writers (captured in tests, silent with <c>--quiet</c>) and an
/// <see cref="IToolHost"/> that draws progress and downloads the way every idrak command does.
/// </summary>
internal sealed class ToolConsole(TextWriter output, TextWriter error, TextReader input, bool live, IToolHost? host = null, bool colour = false)
{
    private readonly ConsoleStatus? _status = live && host is null ? new ConsoleStatus() : null;

    /// <summary>The console, with the live status line and colours.</summary>
    public static ToolConsole Standard() => new(Console.Out, Console.Error, Console.In, live: true);

    public TextWriter Out { get; } = output;

    public TextWriter Error { get; } = error;

    public TextReader In { get; } = input;

    /// <summary>Whether this is the terminal itself, where colours are used.</summary>
    public bool Live => _status is not null || colour;

    /// <summary>A line above the status line (the output itself when there is no status line).</summary>
    public void Log(string line)
    {
        if (_status is not null)
        {
            _status.Log(line);
        }
        else
        {
            host?.Clear();
            Out.WriteLine(line);
        }
    }

    /// <summary>Counted work on the status line (or the host's progress line; nothing without either).</summary>
    public void Progress(string label, long done, long total, TimeSpan elapsed, string? current = null, string unit = "steps")
    {
        if (_status is not null)
        {
            _status.Progress(label, done, total, elapsed, current, unit);
        }
        else
        {
            host?.Progress(current is null ? label : $"{label}  {current}", done, total);
        }
    }

    /// <summary>Keeps the finished status line.</summary>
    public void Finish()
    {
        _status?.Finish();
        host?.Finish();
    }

    /// <summary>Removes the status line.</summary>
    public void Clear()
    {
        _status?.Clear();
        host?.Clear();
    }

    /// <summary>The rows, counted on the status line when there is one.</summary>
    public IEnumerable<JsonObject> Track(IEnumerable<JsonObject> rows, string label, long? total = null, Func<string>? extra = null) =>
        _status is null ? rows : _status.Track(rows, label, total, extra);

    /// <summary>A downloader that logs its steps here (with a progress bar per download on the status line).</summary>
    public Downloader CreateDownloader(string? cacheFolder = null, bool refresh = false) =>
        host is not null ? host.CreateDownloader(cacheFolder, refresh)
        : _status is not null ? _status.CreateDownloader(cacheFolder: cacheFolder, refresh: refresh)
        : new Downloader(null, cacheFolder) { Refresh = refresh, Log = line => Log("  " + line) };
}

/// <summary>
/// What <c>idrak</c> gives a <see cref="ToolConsole"/>: its progress line (on a terminal only, none with <c>--plain</c>)
/// and its downloader (the cache folder, <c>--offline</c> and <c>--timeout</c>).
/// </summary>
internal interface IToolHost
{
    /// <summary>Counted work: redraws the progress line (a new line when the label changes).</summary>
    void Progress(string label, long done, long total);

    /// <summary>Ends the progress line, keeping it.</summary>
    void Finish();

    /// <summary>Removes the progress line (before a line of output).</summary>
    void Clear();

    /// <summary>A downloader into <paramref name="cacheFolder"/> (null: the command's cache folder's downloads).</summary>
    Downloader CreateDownloader(string? cacheFolder, bool refresh);
}
