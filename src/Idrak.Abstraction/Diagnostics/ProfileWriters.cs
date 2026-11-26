// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Globalization;
using System.Runtime.CompilerServices;
using System.Text;
using System.Text.Json;

namespace Idrak.Abstraction.Diagnostics;

/// <summary>
/// The writers of a <see cref="ProfileReport"/>, by name (ignoring case). The library's:
/// <list type="bullet">
/// <item>"table" (<see cref="Table"/>): aligned text for people: the title and the report's notes, the summary (cold, the
/// statistic, least and most of the warm runs; for a watched run its values), then each run's tables (the kernels, the
/// operations), at most 30 rows each.</item>
/// <item>"markdown" (<see cref="Markdown"/>): the same as Markdown tables, every row.</item>
/// <item>"csv" (<see cref="Csv"/>): one row per value of every run (report, device, run, kind, probe, source, name,
/// value, unit), with a header.</item>
/// <item>"json" (<see cref="Json"/>): the whole report as one JSON object on one line (so a file of several reports is
/// JSON Lines): the options, the notes, every run's values and tables, and the summary.</item>
/// </list>
/// Numbers are written the same on every machine (the invariant culture). The profiler holds the writers directly, in a
/// fixed list, and picks one by its own name (<see cref="Named"/>).
/// </summary>
public static class ProfileWriters
{
    /// <summary>The aligned text table's name: "table".</summary>
    public const string Table = "table";

    /// <summary>The Markdown writer's name: "markdown".</summary>
    public const string Markdown = "markdown";

    /// <summary>The CSV writer's name: "csv".</summary>
    public const string Csv = "csv";

    /// <summary>The JSON writer's name: "json".</summary>
    public const string Json = "json";

    // The writers, each under its own name: a fixed list, built once.
    private static readonly IProfileWriter[] Writers = [new TableWriter(markdown: false), new TableWriter(markdown: true), new CsvWriter(), new JsonWriter()];

    /// <summary>The writers' names, in order.</summary>
    public static IReadOnlyCollection<string> Names { get; } = [.. Writers.Select(w => w.Name)];

    /// <summary>The writer named <paramref name="name"/> (any case).</summary>
    /// <exception cref="NotSupportedException">No writer has that name: the message names the writers.</exception>
    public static IProfileWriter Named(string name)
    {
        ArgumentNullException.ThrowIfNull(name);
        foreach (var writer in Writers)
        {
            if (writer.Name.Equals(name, StringComparison.OrdinalIgnoreCase))
            {
                return writer;
            }
        }

        throw new NotSupportedException($"No profile writer '{name}' (writers: {string.Join(", ", Names)}).");
    }

    /// <summary>Each writer as "name: summary", one per line.</summary>
    public static string Describe() => string.Join("\n", Writers.Select(w => $"{w.Name}: {w.Summary}"));

    internal static string Number(double? value) => value is { } v ? ProfileProbes.Text(v) : "";

    internal static string Title(ProfileReport report)
    {
        var o = report.Options;
        string runs = report.Watched ? "one run watched" : $"{(o.Cold ? "a cold run, " : "")}{o.WarmRuns} warm runs ({o.Statistic.ToString().ToLowerInvariant()})";
        return string.Create(CultureInfo.InvariantCulture,
            $"Profile \"{report.Name}\" on {report.Device} ({report.Device.Backend.Name}): {runs}; "
            + $"probes {(report.Probes.Count == 0 ? "none" : string.Join(", ", report.Probes))}"
            + $"{(report.Skipped.Count > 0 ? "; not on this device: " + string.Join(", ", report.Skipped) : "")}");
    }
}

internal sealed class TableWriter(bool markdown) : IProfileWriter
{
    public string Name => markdown ? ProfileWriters.Markdown : ProfileWriters.Table;

    public string Summary => markdown ? "Markdown tables: the summary and every run's tables" : "aligned text: the summary, then each run's tables (30 rows each)";

    public void Write(ProfileReport report, TextWriter output)
    {
        output.WriteLine(markdown ? "## " + ProfileWriters.Title(report) : ProfileWriters.Title(report));
        foreach (string note in report.Notes)
        {
            output.WriteLine(markdown ? "- " + note : "  " + note);
        }

        output.WriteLine();
        string statistic = report.Options.Statistic.ToString().ToLowerInvariant();
        if (report.Watched)
        {
            Grid(output, ["value", "unit", "measured", "from"],
                [.. report.Summary.Select(s => (IReadOnlyList<string>)[s.Name, s.Unit, ProfileWriters.Number(s.Value), s.Source ?? "workload"])], rightFrom: 2);
        }
        else
        {
            Grid(output, ["value", "unit", "cold", statistic, "min", "max", "from"],
                [.. report.Summary.Select(s => (IReadOnlyList<string>)[s.Name, s.Unit, ProfileWriters.Number(s.Cold), ProfileWriters.Number(s.Value), ProfileWriters.Number(s.Min), ProfileWriters.Number(s.Max), s.Source ?? "workload"])],
                rightFrom: 2);
        }

        foreach (var run in report.Runs)
        {
            foreach (var table in run.Tables)
            {
                output.WriteLine();
                string title = report.Watched ? $"{table.Name} ({table.Rows.Count} rows)"
                    : $"{table.Name} ({run.Kind.ToString().ToLowerInvariant()} run {run.Index}{(run.Probe is { } p ? ", " + p : "")}, {table.Rows.Count} rows)";
                output.WriteLine(markdown ? "### " + title : title);
                Grid(output, table.Columns, markdown ? table.Rows : [.. table.Rows.Take(30)], rightFrom: 1);
            }
        }
    }

    // Columns padded to their widest cell (numbers right-aligned from `rightFrom` on), or a Markdown table.
    private void Grid(TextWriter output, IReadOnlyList<string> columns, IReadOnlyList<IReadOnlyList<string>> rows, int rightFrom)
    {
        if (markdown)
        {
            output.WriteLine("| " + string.Join(" | ", columns) + " |");
            output.WriteLine("|" + string.Concat(columns.Select((_, i) => i >= rightFrom ? "---:|" : "---|")));
            foreach (var row in rows)
            {
                output.WriteLine("| " + string.Join(" | ", row.Select(c => c.Replace("|", "\\|", StringComparison.Ordinal))) + " |");
            }

            return;
        }

        var widths = columns.Select((c, i) => Math.Max(c.Length, rows.Count == 0 ? 0 : rows.Max(r => i < r.Count ? r[i].Length : 0))).ToArray();
        var line = new StringBuilder();
        void Line(IReadOnlyList<string> cells)
        {
            line.Clear();
            for (int i = 0; i < widths.Length; i++)
            {
                string cell = i < cells.Count ? cells[i] : "";
                line.Append(i >= rightFrom ? cell.PadLeft(widths[i]) : cell.PadRight(widths[i])).Append("  ");
            }

            output.WriteLine(line.ToString().TrimEnd());
        }

        Line(columns);
        foreach (var row in rows)
        {
            Line(row);
        }
    }
}

internal sealed class CsvWriter : IProfileWriter
{
    public string Name => ProfileWriters.Csv;

    public string Summary => "one row per value of every run: report, device, run, kind, probe, source, name, value, unit";

    public void Write(ProfileReport report, TextWriter output)
    {
        output.WriteLine("report,device,run,kind,probe,source,name,value,unit");
        foreach (var run in report.Runs)
        {
            foreach (var value in run.Values)
            {
                output.WriteLine(string.Join(',', Cell(report.Name), Cell(report.Device.ToString()), run.Index.ToString(CultureInfo.InvariantCulture),
                    run.Kind.ToString().ToLowerInvariant(), Cell(run.Probe ?? ""), Cell(value.Source ?? ""), Cell(value.Name),
                    value.Value.ToString("R", CultureInfo.InvariantCulture), Cell(value.Unit)));
            }
        }
    }

    // RFC 4180: quoted when it holds a comma, a quote or a line break, its quotes doubled.
    private static string Cell(string text) => text.AsSpan().IndexOfAny(",\"\r\n") >= 0 ? "\"" + text.Replace("\"", "\"\"", StringComparison.Ordinal) + "\"" : text;
}

internal sealed class JsonWriter : IProfileWriter
{
    public string Name => ProfileWriters.Json;

    public string Summary => "the whole report as one JSON object on one line (JSON Lines across reports)";

    public void Write(ProfileReport report, TextWriter output)
    {
        var buffer = new System.Buffers.ArrayBufferWriter<byte>(4096);
        using (var w = new Utf8JsonWriter(buffer))
        {
            var o = report.Options;
            w.WriteStartObject();
            w.WriteString("report", report.Name);
            w.WriteString("device", report.Device.ToString());
            w.WriteString("backend", report.Device.Backend.Name);
            w.WriteString("time", DateTimeOffset.UtcNow);
            w.WriteStartObject("options");
            w.WriteBoolean("cold", o.Cold);
            w.WriteNumber("warm_runs", o.WarmRuns);
            w.WriteString("statistic", o.Statistic.ToString().ToLowerInvariant());
            w.WriteBoolean("watched", report.Watched);
            Strings(w, "probes", report.Probes);
            Strings(w, "skipped", report.Skipped);
            w.WriteEndObject();
            Strings(w, "notes", report.Notes);
            w.WriteStartArray("runs");
            foreach (var run in report.Runs)
            {
                w.WriteStartObject();
                w.WriteNumber("index", run.Index);
                w.WriteString("kind", run.Kind.ToString().ToLowerInvariant());
                if (run.Probe is { } probe)
                {
                    w.WriteString("probe", probe);
                }

                w.WriteStartArray("values");
                foreach (var value in run.Values)
                {
                    w.WriteStartObject();
                    w.WriteString("name", value.Name);
                    Number(w, "value", value.Value);
                    w.WriteString("unit", value.Unit);
                    if (value.Source is { } source)
                    {
                        w.WriteString("source", source);
                    }

                    w.WriteEndObject();
                }

                w.WriteEndArray();
                w.WriteStartArray("tables");
                foreach (var table in run.Tables)
                {
                    w.WriteStartObject();
                    w.WriteString("name", table.Name);
                    Strings(w, "columns", table.Columns);
                    w.WriteStartArray("rows");
                    foreach (var row in table.Rows)
                    {
                        w.WriteStartArray();
                        foreach (string cell in row)
                        {
                            w.WriteStringValue(cell);
                        }

                        w.WriteEndArray();
                    }

                    w.WriteEndArray();
                    w.WriteEndObject();
                }

                w.WriteEndArray();
                w.WriteEndObject();
            }

            w.WriteEndArray();
            w.WriteStartArray("summary");
            foreach (var s in report.Summary)
            {
                w.WriteStartObject();
                w.WriteString("name", s.Name);
                w.WriteString("unit", s.Unit);
                if (s.Source is { } source)
                {
                    w.WriteString("source", source);
                }

                Optional(w, "cold", s.Cold);
                Optional(w, "value", s.Value);
                Optional(w, "min", s.Min);
                Optional(w, "max", s.Max);
                w.WriteNumber("count", s.Count);
                w.WriteEndObject();
            }

            w.WriteEndArray();
            w.WriteEndObject();
        }

        output.WriteLine(Encoding.UTF8.GetString(buffer.WrittenSpan));
    }

    private static void Strings(Utf8JsonWriter w, string name, IEnumerable<string> values)
    {
        w.WriteStartArray(name);
        foreach (string value in values)
        {
            w.WriteStringValue(value);
        }

        w.WriteEndArray();
    }

    // JSON has no NaN or infinity: those are written as null (FiniteJsonNumbers).
    private static void Number(Utf8JsonWriter w, string name, double value) => FiniteJsonNumbers.WriteNumber(w, name, value);

    private static void Optional(Utf8JsonWriter w, string name, double? value)
    {
        if (value is { } v)
        {
            Number(w, name, v);
        }
    }
}
