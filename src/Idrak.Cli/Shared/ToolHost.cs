// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Idrak.Datasets;

namespace Idrak.Cli.Shared;

/// <summary>
/// The <see cref="IToolHost"/> of <c>idrak tune</c> and <c>idrak data</c>: progress through <see cref="ProgressLine"/>
/// and downloads through <see cref="Http.Downloader"/>, so they behave as every other command (<c>--plain</c>,
/// <c>--quiet</c>, <c>--offline</c>, <c>--timeout</c>, <c>--cache</c>).
/// </summary>
internal sealed class ToolHost(CommandContext context) : IToolHost
{
    private readonly Lock _lock = new();
    private ProgressLine? _line;

    /// <summary>A console for the tools writing to <paramref name="output"/>, with this host.</summary>
    public static ToolConsole Console(CommandContext context, TextWriter output) =>
        new(output, context.ErrorOutput, System.Console.In, live: false, new ToolHost(context),
            colour: ReferenceEquals(Terminal.Unwrap(output), System.Console.Out) && Terminal.UseColour(context, output));

    public void Progress(string label, long done, long total)
    {
        lock (_lock)
        {
            if (_line is null || _line.Label != label && !SameWork(_line.Label, label))
            {
                _line?.Finish();
                _line = new ProgressLine(context, label, total);
            }

            _line.Label = label;
            _line.Total = total;
            _line.Report(done);
        }
    }

    public void Finish()
    {
        lock (_lock)
        {
            _line?.Finish();
            _line = null;
        }
    }

    public void Clear()
    {
        lock (_lock)
        {
            _line?.Clear();
            _line = null;
        }
    }

    public Downloader CreateDownloader(string? cacheFolder, bool refresh) => Http.Downloader(context, cacheFolder, refresh);

    // "training  loss 1.23" and "training  loss 1.20" are the same work with a new detail: the line is redrawn, not ended.
    private static bool SameWork(string a, string b) => a.Split("  ")[0] == b.Split("  ")[0];
}
