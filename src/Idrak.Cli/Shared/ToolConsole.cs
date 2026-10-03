// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Text.Json.Nodes;
using Idrak.Datasets;

namespace Idrak.Cli.Shared;

/// <summary>
/// Where the fine-tuning and dataset tools (<c>idrak tune</c>, <c>idrak data</c>) write: their output and error
/// streams, the input for interactive chat, and progress. <c>idrak</c> passes its own writers (captured in tests,
/// silent with <c>--quiet</c>) and an <see cref="IToolHost"/> that draws progress and downloads the way every idrak
/// command does.
/// </summary>
internal sealed class ToolConsole(TextWriter output, TextWriter error, TextReader input, IToolHost host, bool colour = false)
{
    public TextWriter Out { get; } = output;

    public TextWriter Error { get; } = error;

    public TextReader In { get; } = input;

    /// <summary>Whether this is the terminal itself, where colours are used.</summary>
    public bool Live => colour;

    /// <summary>A line of output (after removing the progress line).</summary>
    public void Log(string line)
    {
        host.Clear();
        Out.WriteLine(line);
    }

    /// <summary>Counted work on the host's progress line.</summary>
    public void Progress(string label, long done, long total, TimeSpan elapsed, string? current = null, string unit = "steps") =>
        host.Progress(current is null ? label : $"{label}  {current}", done, total);

    /// <summary>Keeps the finished progress line.</summary>
    public void Finish() => host.Finish();

    /// <summary>Removes the progress line.</summary>
    public void Clear() => host.Clear();

    /// <summary>The rows as they are read (the label, total and extra text name the work; idrak shows no row counter).</summary>
    public IEnumerable<JsonObject> Track(IEnumerable<JsonObject> rows, string label, long? total = null, Func<string>? extra = null) => rows;

    /// <summary>The host's downloader (its cache folder, <c>--offline</c>, <c>--timeout</c> and progress).</summary>
    public Downloader CreateDownloader(string? cacheFolder = null, bool refresh = false) => host.CreateDownloader(cacheFolder, refresh);
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
