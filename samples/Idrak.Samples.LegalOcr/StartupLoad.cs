// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Globalization;

namespace Idrak.Samples.LegalOcr;

/// <summary>
/// Loads the model of <see cref="LegalOcrSettings.LoadAtStart"/> before the server starts, with a live progress line from
/// the library's progress reporters (<see cref="ProgressReporters"/>): what it is doing, the download's size, speed and time
/// left, the time so far and the device memory in use.
/// </summary>
public static class StartupLoad
{
    /// <summary>
    /// Loads <paramref name="id"/>, writing its progress to <paramref name="output"/> (the "console" reporter's line
    /// rewritten in place when <paramref name="live"/>, else the "plain" reporter's line every ten seconds;
    /// <c>IDRAK_PROGRESS</c> chooses another); true when the model is ready. A model that cannot be loaded is reported, and
    /// the page shows why too.
    /// </summary>
    public static async Task<bool> RunAsync(ReaderService service, string id, TextWriter output, bool live, CancellationToken cancellationToken = default)
    {
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

        var options = new ProgressOptions
        {
            Reporter = ProgressReporters.FromEnvironment() ?? (live ? ProgressReporters.ConsoleReporter : ProgressReporters.PlainReporter),
            Output = output,
            Interval = live ? null : TimeSpan.FromSeconds(10),
        };
        using (var progress = ProgressReporters.Start($"loading {id}", null, ProgressUnit.Elapsed, options))
        {
            progress.Detail = () => Memory(service.Status);
            string? stage = null;                                                   // the file downloading (its speed is measured from its start)
            while (!loading.IsCompleted)
            {
                await Task.WhenAny(loading, Task.Delay(500, CancellationToken.None));
                if (loading.IsCompleted)
                {
                    break;
                }

                var status = service.Status;
                string current = status.Download?.File ?? "";
                if (current != stage)
                {
                    progress.NextStage(Label(status), status.Download?.Total, status.Download is null ? ProgressUnit.Elapsed : ProgressUnit.Bytes, keepLine: false);
                    stage = current;
                }

                progress.Label = Label(status);
                progress.Report(status.Download?.Received ?? 0);
            }

            progress.Clear();
        }

        bool ready = await loading;
        var end = service.Status;
        output.WriteLine(cancellationToken.IsCancellationRequested ? $"Loading {id} stopped."
            : ready ? $"{end.Message} {Memory(end)}" : end.Message);
        return ready;
    }

    /// <summary>What the service is doing, as the progress line's label (the status message without its trailing dots).</summary>
    internal static string Label(ReaderStatus status) => status.Message.TrimEnd('\u2026', '.');

    /// <summary>
    /// The status as the progress line shows it <paramref name="elapsed"/> into its stage: the label, the download's share,
    /// size, speed and time left (when downloading), and the device memory.
    /// </summary>
    internal static string Line(ReaderStatus status, TimeSpan elapsed)
    {
        string text = status.Download is { } download
            ? ProgressText.Format(Label(status), download.Received, download.Total, elapsed, ProgressUnit.Bytes)
            : ProgressText.Format(Label(status), 0, null, elapsed, ProgressUnit.Elapsed);
        return Memory(status) is { Length: > 0 } memory ? text + "  " + memory : text;
    }

    private static string Memory(ReaderStatus status) =>
        status.Memory is { } memory ? $"{status.Device}: {Size(memory.InUse)} in use, peak {Size(memory.Peak)}" : "";

    private static string Size(long bytes) => bytes >= 1L << 30 ? (bytes / (double)(1L << 30)).ToString("F2", CultureInfo.InvariantCulture) + " GB"
        : (bytes / (double)(1 << 20)).ToString("F0", CultureInfo.InvariantCulture) + " MB";
}
