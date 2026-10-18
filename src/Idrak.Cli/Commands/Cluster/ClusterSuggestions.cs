// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Text.Json;
using System.Text.Json.Nodes;
using Idrak.Cli.Shared;
using Idrak.Diagnostics;

namespace Idrak.Cli.Commands.Cluster;

/// <summary>
/// Shows each setting the library suggests while a command runs (plan 23, decision 4: the "cluster" advisor's, and any
/// other advisor's), once, as a line above the progress line: the progress lines drawn on <paramref name="progress"/> are
/// taken off the screen first and drawn again by their next report. Subscribing it is also what turns the cluster
/// advisor's measurement on (unset <c>IDRAK_CLUSTER_SUGGEST</c> measures while someone listens).
/// </summary>
internal sealed class SuggestionLine(TextWriter output, TextWriter progress) : ITelemetryHook
{
    // What each output has shown, so a command that trains more than once (a profile's cold and warm runs) shows each once.
    private static readonly System.Runtime.CompilerServices.ConditionalWeakTable<TextWriter, HashSet<(string Source, string Setting, string Value)>> Shown = new();
    private readonly HashSet<(string Source, string Setting, string Value)> _shown = Shown.GetValue(output, _ => []);

    public TelemetryLevel Levels => TelemetryLevel.Suggestions;

    /// <summary>Shows the suggestions published until disposed.</summary>
    public static IDisposable Show(TextWriter output, TextWriter progress) => Telemetry.Subscribe(new SuggestionLine(output, progress));

    /// <summary>A suggestion as one line: "suggested: SETTING = VALUE (SOURCE on DEVICE): REASON".</summary>
    public static string Format(string setting, string value, string source, string device, string reason) =>
        $"suggested: {setting} = {value} ({source} on {device}): {reason}";

    public void OnSuggestion(in SettingSuggestion e)
    {
        lock (_shown)
        {
            if (!_shown.Add((e.Source, e.Setting, e.Value)))
            {
                return;
            }
        }

        ProgressLine.EraseOn(progress);
        string line = Format(e.Setting, e.Value, e.Source, e.Device, e.Reason);
        lock (output)
        {
            output.WriteLine(line);
            output.Flush();
        }
    }
}

/// <summary>
/// Appends each setting suggested while a run trains to its log (<c>RUN/log.jsonl</c>, as <see cref="TelemetryJson"/> writes
/// it: a <c>setting_suggestion</c> record), so <c>idrak suggest RUN</c> lists them afterwards. A command whose own log
/// writer keeps the file open (pretrain's) records them through that writer instead.
/// </summary>
internal sealed class SuggestionLog(string path) : ITelemetryHook
{
    public TelemetryLevel Levels => TelemetryLevel.Suggestions;

    public void OnSuggestion(in SettingSuggestion e)
    {
        try
        {
            lock (this)
            {
                using var stream = new FileStream(path, FileMode.Append, FileAccess.Write, FileShare.Read);
                using (var writer = new Utf8JsonWriter(stream))
                {
                    TelemetryJson.Write(writer, new TelemetryRecord(DateTimeOffset.UtcNow, e));
                }

                stream.Write("\n"u8);
            }
        }
        catch (IOException)
        {
            // A log another writer holds this instant: the suggestion was shown; the log goes without it.
        }
    }
}

/// <summary>A suggestion recorded in a run's log.</summary>
internal sealed record LoggedSuggestion(string Setting, string Value, string Reason, string Device, string Source);

/// <summary>The suggestions a run's log holds (<c>setting_suggestion</c> records), each setting and value once, in order.</summary>
internal static class RunSuggestions
{
    /// <summary>Whether <paramref name="path"/> is a run (a folder with a log.jsonl, or a log.jsonl) rather than data.</summary>
    public static string? LogOf(CommandContext context, string path)
    {
        if (File.Exists(path))
        {
            return Path.GetFileName(path) == "log.jsonl" ? path : null;
        }

        foreach (string folder in new[] { path, Path.Combine(RunLog.RunsFolder(context), path) })
        {
            if (Directory.Exists(folder) && File.Exists(Path.Combine(folder, "log.jsonl")) && File.Exists(Path.Combine(folder, "run.json")))
            {
                return Path.Combine(folder, "log.jsonl");
            }
        }

        return null;
    }

    public static List<LoggedSuggestion> Read(string log)
    {
        var found = new List<LoggedSuggestion>();
        var seen = new HashSet<(string, string)>();
        foreach (string line in File.ReadLines(log))
        {
            if (!line.Contains("\"setting_suggestion\"", StringComparison.Ordinal))
            {
                continue;
            }

            try
            {
                if (JsonNode.Parse(line) is JsonObject record && (string?)record["setting"] is { } setting && (string?)record["value"] is { } value && seen.Add((setting, value)))
                {
                    found.Add(new LoggedSuggestion(setting, value, (string?)record["reason"] ?? "", (string?)record["device"] ?? "", (string?)record["source"] ?? ""));
                }
            }
            catch (JsonException)
            {
                // A line cut by an interruption.
            }
        }

        return found;
    }
}
