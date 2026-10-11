// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Diagnostics;
using System.Globalization;

namespace Idrak.Samples.LegalOcr;

/// <summary>
/// Loads the model of <see cref="LegalOcrSettings.LoadAtStart"/> before the server starts, with a live progress line: what
/// it is doing, the download's size, speed and time left, the time so far and the device memory in use.
/// </summary>
public static class StartupLoad
{
    /// <summary>
    /// Loads <paramref name="id"/>, writing its progress to <paramref name="output"/> (one line rewritten in place when
    /// <paramref name="live"/>, else a line every ten seconds); true when the model is ready. A model that cannot be loaded
    /// is reported, and the page shows why too.
    /// </summary>
    public static async Task<bool> RunAsync(ReaderService service, string id, TextWriter output, bool live, CancellationToken cancellationToken = default)
    {
        var clock = Stopwatch.StartNew();
        Task<bool> loading;
        try
        {
            loading = service.LoadAsync(id, cancellationToken);
        }
        catch (Exception ex) when (ex is ArgumentException or InvalidOperationException)
        {
            output.WriteLine($"Not loaded at start ({id}): {ex.Message}");
            return false;
        }

        int width = 0;
        var nextLine = TimeSpan.Zero;
        (string File, long Received, TimeSpan At) first = ("", 0, TimeSpan.Zero);          // where the current file's speed is measured from
        while (!loading.IsCompleted)
        {
            await Task.WhenAny(loading, Task.Delay(500, CancellationToken.None));
            if (loading.IsCompleted)
            {
                break;
            }

            var status = service.Status;
            if (status.Download is { } file && file.File != first.File)
            {
                first = (file.File, file.Received, clock.Elapsed);
            }

            double? rate = status.Download is { } now && clock.Elapsed - first.At is { TotalSeconds: >= 1 } span ? (now.Received - first.Received) / span.TotalSeconds : null;
            string line = Line(status, clock.Elapsed, rate);
            if (live)
            {
                output.Write('\r' + line.PadRight(width));
                width = line.Length;
            }
            else if (clock.Elapsed >= nextLine)
            {
                output.WriteLine(line);
                nextLine = clock.Elapsed + TimeSpan.FromSeconds(10);
            }
        }

        if (live && width > 0)
        {
            output.Write('\r' + new string(' ', width) + '\r');
        }

        bool ready = await loading;
        var end = service.Status;
        output.WriteLine(cancellationToken.IsCancellationRequested ? $"Loading {id} stopped."
            : ready ? $"{end.Message} {Memory(end)}" : end.Message);
        return ready;
    }

    // The status as one line: the message (the file being downloaded with its size, speed (bytes a second, when measured)
    // and time left), the time so far, the device memory.
    internal static string Line(ReaderStatus status, TimeSpan elapsed, double? rate)
    {
        var parts = new List<string> { status.Message.TrimEnd('…', '.') };
        if (status.Download is { } download)
        {
            string size = download.Total is { } total ? $"{Size(download.Received)} of {Size(total)}" : Size(download.Received);
            string speed = rate is > 0 ? $" at {Size((long)rate.Value)}/s" : "";
            string left = download.Total is { } all && rate is > 0 ? $", {Duration(TimeSpan.FromSeconds((all - download.Received) / rate.Value))} left" : "";
            parts.Add(size + speed + left);
        }

        parts.Add(Duration(elapsed));
        parts.Add(Memory(status));
        return string.Join(" · ", parts.Where(p => p.Length > 0));
    }

    private static string Memory(ReaderStatus status) =>
        status.Memory is { } memory ? $"{status.Device}: {Size(memory.InUse)} in use, peak {Size(memory.Peak)}" : "";

    private static string Size(long bytes) => bytes >= 1L << 30 ? (bytes / (double)(1L << 30)).ToString("F2", CultureInfo.InvariantCulture) + " GB"
        : (bytes / (double)(1 << 20)).ToString("F0", CultureInfo.InvariantCulture) + " MB";

    private static string Duration(TimeSpan time) => time.TotalMinutes >= 1 ? $"{(int)time.TotalMinutes} min {time.Seconds} s" : $"{time.TotalSeconds:F0} s";
}
