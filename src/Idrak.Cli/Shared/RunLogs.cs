// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Globalization;
using System.Text;
using System.Text.Json;
using System.Text.Json.Nodes;
using Idrak.Diagnostics;
using Idrak.Training;
using Idrak.Training.Abstractions;

namespace Idrak.Cli.Shared;

/// <summary>One epoch of a training log.</summary>
internal sealed record LoggedEpoch(int Epoch, double? Loss, double? ValidationLoss, IReadOnlyDictionary<string, double> Metrics,
    IReadOnlyDictionary<string, double> ValidationMetrics, double LearningRate, double Milliseconds, bool Best);

/// <summary>
/// A training run as its JSON Lines telemetry log tells it (the library's format, <see cref="TelemetryJson"/>: the
/// <c>training_started</c>, <c>epoch</c> and <c>training_completed</c> events), with the run folder's <c>run.json</c>
/// when there is one. <c>idrak train</c> writes runs as folders under <c>CACHE/runs</c>; any log written by
/// <see cref="JsonLinesLogger"/> is read the same way.
/// </summary>
internal sealed class RunLog
{
    public required string Name { get; init; }

    public required string Path { get; init; }

    public string? Folder { get; init; }

    public JsonObject? Started { get; init; }

    public JsonObject? Completed { get; init; }

    public JsonObject? Settings { get; init; }

    public required List<LoggedEpoch> Epochs { get; init; }

    public DateTimeOffset? Time { get; init; }

    /// <summary>The epoch with the lowest validation loss (the training loss without validation), or null.</summary>
    public LoggedEpoch? Best => Epochs.Where(e => (e.ValidationLoss ?? e.Loss) is not null).MinBy(e => e.ValidationLoss ?? e.Loss);

    public double TotalSeconds => Epochs.Sum(e => e.Milliseconds) / 1000;

    /// <summary>"completed", "stopped early", "cancelled" or "running / interrupted".</summary>
    public string Status => Completed is null ? "running or interrupted"
        : (bool?)Completed["cancelled"] == true ? "cancelled" : (bool?)Completed["stopped_early"] == true ? "stopped early" : "completed";

    /// <summary>The runs folder: CACHE/runs.</summary>
    public static string RunsFolder(CommandContext context) => System.IO.Path.Combine(context.CacheFolder, "runs");

    /// <summary>The log of RUN: a log file, a run folder, or the name of a folder under the runs folder.</summary>
    public static RunLog Find(CommandContext context, string run)
    {
        string? path = File.Exists(run) ? run
            : Directory.Exists(run) ? LogIn(run)
            : Directory.Exists(System.IO.Path.Combine(RunsFolder(context), run)) ? LogIn(System.IO.Path.Combine(RunsFolder(context), run))
            : null;
        return path is null
            ? throw new UsageException($"No run '{run}': give a log file (.jsonl), a run folder, or a name from 'idrak runs list' (runs are under {RunsFolder(context)}).")
            : Read(path);
    }

    /// <summary>Every run under <paramref name="folder"/>: run folders with a log, and log files.</summary>
    public static List<RunLog> All(string folder)
    {
        if (!Directory.Exists(folder))
        {
            return [];
        }

        var logs = Directory.GetDirectories(folder).Select(d => LogIn(d)).OfType<string>()
            .Concat(Directory.GetFiles(folder, "*.jsonl"));
        return [.. logs.Select(Read).OrderBy(r => r.Time ?? DateTimeOffset.MinValue)];
    }

    private static string? LogIn(string folder)
    {
        string log = System.IO.Path.Combine(folder, "log.jsonl");
        return File.Exists(log) ? log : Directory.GetFiles(folder, "*.jsonl").Order(StringComparer.Ordinal).FirstOrDefault();
    }

    /// <summary>Reads a log file (and run.json next to it, when the file is a run folder's log).</summary>
    public static RunLog Read(string path)
    {
        JsonObject? started = null, completed = null;
        var epochs = new List<LoggedEpoch>();
        DateTimeOffset? time = null;
        foreach (string line in File.ReadLines(path))
        {
            if (string.IsNullOrWhiteSpace(line))
            {
                continue;
            }

            JsonObject record;
            try
            {
                record = JsonNode.Parse(line) as JsonObject ?? throw new JsonException();
            }
            catch (JsonException)
            {
                continue;                                                           // a line cut by an interruption
            }

            time ??= DateTimeOffset.TryParse((string?)record["time"], CultureInfo.InvariantCulture, DateTimeStyles.None, out var t) ? t : null;
            switch ((string?)record["event"])
            {
                case "training_started":
                    started ??= record;
                    break;
                case "training_completed":
                    completed = record;
                    break;
                case "epoch":
                    epochs.Add(new LoggedEpoch((int?)record["epoch"] ?? epochs.Count + 1, (double?)record["loss"], (double?)record["val_loss"],
                        Values(record["metrics"]), Values(record["val_metrics"]), (double?)record["learning_rate"] ?? 0,
                        (double?)record["duration_ms"] ?? 0, (bool?)record["best"] ?? false));
                    break;
            }
        }

        string? folder = System.IO.Path.GetFileName(path) == "log.jsonl" ? System.IO.Path.GetDirectoryName(System.IO.Path.GetFullPath(path)) : null;
        string runFile = folder is null ? "" : System.IO.Path.Combine(folder, "run.json");
        return new RunLog
        {
            Name = folder is not null ? System.IO.Path.GetFileName(folder) : System.IO.Path.GetFileNameWithoutExtension(path),
            Path = path,
            Folder = folder,
            Started = started,
            Completed = completed,
            Settings = File.Exists(runFile) ? JsonNode.Parse(File.ReadAllText(runFile)) as JsonObject : null,
            Epochs = epochs,
            Time = time,
        };
    }

    private static Dictionary<string, double> Values(JsonNode? node) =>
        node is JsonObject o ? o.Where(p => p.Value is JsonValue).ToDictionary(p => p.Key, p => (double)p.Value!) : [];

    /// <summary>A summary as JSON (settings, best epoch, losses per epoch).</summary>
    public JsonObject ToJson(bool epochs)
    {
        var best = Best;
        var json = new JsonObject
        {
            ["name"] = Name,
            ["log"] = Path,
            ["folder"] = Folder,
            ["time"] = Time?.ToString("O", CultureInfo.InvariantCulture),
            ["status"] = Status,
            ["epochs"] = Epochs.Count,
            ["bestEpoch"] = best?.Epoch,
            ["bestLoss"] = best is null ? null : Finite(best.ValidationLoss ?? best.Loss),
            ["finalLoss"] = Epochs.Count == 0 ? null : Finite(Epochs[^1].Loss),
            ["seconds"] = Math.Round(TotalSeconds, 3),
            ["settings"] = Settings?.DeepClone() ?? Started?.DeepClone(),
        };
        if (epochs)
        {
            json["history"] = new JsonArray([.. Epochs.Select(e => (JsonNode)new JsonObject
            {
                ["epoch"] = e.Epoch, ["loss"] = Finite(e.Loss), ["val_loss"] = Finite(e.ValidationLoss), ["learning_rate"] = e.LearningRate,
                ["metrics"] = new JsonObject([.. e.Metrics.Select(p => KeyValuePair.Create(p.Key, (JsonNode?)Finite(p.Value)))]),
                ["val_metrics"] = new JsonObject([.. e.ValidationMetrics.Select(p => KeyValuePair.Create(p.Key, (JsonNode?)Finite(p.Value)))]),
            })]);
        }

        return json;
    }

    private static double? Finite(double? value) => value is { } v && double.IsFinite(v) ? Math.Round(v, 6) : null;

    /// <summary>A loss as text (or "-").</summary>
    public static string Format(double? value) => value is { } v && double.IsFinite(v) ? v.ToString("G5", CultureInfo.InvariantCulture) : "-";
}

/// <summary>
/// A trainer callback that appends the run to a JSON Lines log in the library's telemetry format, written as it goes
/// (so <c>idrak runs show</c> reads a run while it trains); epochs are numbered from <paramref name="epochOffset"/> + 1
/// so a resumed run continues the same log.
/// </summary>
internal sealed class RunLogWriter(string path, TrainingStarted started, int epochOffset) : ITrainerCallback
{
    private readonly DateTimeOffset _start = DateTimeOffset.UtcNow;

    public void OnTrainBegin(TrainerContext context) => Append(started);

    public void OnEpochEnd(TrainerContext context, EpochCompleted epoch) => Append(epoch with { Epoch = epoch.Epoch + epochOffset, Epochs = epoch.Epochs + epochOffset });

    public void OnTrainEnd(TrainerContext context, TrainingHistory history) => Append(new TrainingCompleted(history.Epochs.Count, DateTimeOffset.UtcNow - _start,
        history.Epochs.Count > 0 ? history.Epochs[^1].Loss : double.NaN, history.BestEpoch + epochOffset, history.BestLoss, history.StoppedEarly,
        context.CancellationToken.IsCancellationRequested));

    internal void Append(object record)
    {
        using var stream = new FileStream(path, FileMode.Append, FileAccess.Write, FileShare.Read);
        using (var writer = new Utf8JsonWriter(stream))
        {
            TelemetryJson.Write(writer, new TelemetryRecord(DateTimeOffset.UtcNow, record));
        }

        stream.Write("\n"u8);
    }
}

/// <summary>Curves as text: a few series of values against the epoch, on a character grid.</summary>
internal static class TextPlot
{
    private static readonly char[] Marks = ['*', 'o', 'x', '+', '#', '@'];

    public static string Plot(IReadOnlyList<(string Name, IReadOnlyList<double?> Values)> series, int width = 60, int height = 12)
    {
        var all = series.SelectMany(s => s.Values).OfType<double>().Where(double.IsFinite).ToList();
        int length = series.Count == 0 ? 0 : series.Max(s => s.Values.Count);
        if (all.Count == 0 || length == 0)
        {
            return "";
        }

        double low = all.Min(), high = all.Max();
        if (high - low < 1e-12)
        {
            (low, high) = (low - 0.5, high + 0.5);
        }

        int columns = Math.Min(width, length);
        var grid = Enumerable.Range(0, height).Select(_ => Enumerable.Repeat(' ', columns).ToArray()).ToArray();
        for (int s = 0; s < series.Count; s++)
        {
            for (int c = 0; c < columns; c++)
            {
                int index = length == 1 ? 0 : (int)Math.Round(c * (length - 1) / (double)Math.Max(1, columns - 1));
                if (index < series[s].Values.Count && series[s].Values[index] is { } v && double.IsFinite(v))
                {
                    int row = height - 1 - (int)Math.Round((v - low) / (high - low) * (height - 1));
                    grid[row][c] = grid[row][c] == ' ' ? Marks[s % Marks.Length] : '%';
                }
            }
        }

        var text = new StringBuilder();
        for (int r = 0; r < height; r++)
        {
            string label = r == 0 ? RunLog.Format(high) : r == height - 1 ? RunLog.Format(low) : "";
            text.Append(label.PadLeft(10)).Append(" |").Append(new string(grid[r]).TrimEnd()).Append('\n');
        }

        text.Append(new string(' ', 11)).Append('+').Append(new string('-', columns)).Append('\n');
        text.Append(new string(' ', 12)).Append("epoch 1").Append($"epoch {length}".PadLeft(Math.Max(0, columns - 7))).Append('\n');
        text.Append(new string(' ', 12)).Append(string.Join("   ", series.Select((s, i) => $"{Marks[i % Marks.Length]} {s.Name}"))).Append('\n');
        return text.ToString();
    }
}
