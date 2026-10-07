// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Globalization;
using System.Text;
using System.Text.Json;
using System.Text.Json.Nodes;
using Idrak.Data;

namespace Idrak.Cli.Shared;

/// <summary>
/// Rows of data files for the data and train commands: read through the dataset library (CSV, TSV, JSON Lines, JSON,
/// Parquet, also compressed), written as JSON Lines, JSON or CSV by the output's extension, and shown as short cells.
/// </summary>
internal static class RowFiles
{
    /// <summary>The rows of a file (its format by extension).</summary>
    public static DatasetRows Read(string path)
    {
        if (Directory.Exists(path))
        {
            return DatasetRows.FromFolder(path);
        }

        if (!File.Exists(path))
        {
            throw new FileNotFoundException($"{path} not found.", path);
        }

        return DatasetRows.FromFile(path);
    }

    /// <summary>The format's name for messages: csv, tsv, jsonl, json, parquet or the extension.</summary>
    public static string FormatName(string path) => DataFiles.FormatOf(path) switch
    {
        DataFormat.JsonLines => "jsonl",
        DataFormat.Json => "json",
        DataFormat.Csv => "csv",
        DataFormat.Tsv => "tsv",
        DataFormat.Parquet => "parquet",
        DataFormat.Text => "text",
        _ => Path.GetExtension(path).TrimStart('.').ToLowerInvariant(),
    };

    /// <summary>Writes rows by the extension of <paramref name="path"/> (.jsonl, .json, .csv, .tsv); returns how many.</summary>
    public static long Write(IEnumerable<JsonObject> rows, string path)
    {
        string format = FormatName(path);
        Directory.CreateDirectory(Path.GetDirectoryName(Path.GetFullPath(path))!);
        switch (format)
        {
            case "jsonl":
                return new DatasetRows(() => rows).WriteJsonLines(path);
            case "json":
            {
                var array = new JsonArray([.. rows.Select(r => (JsonNode)r.DeepClone())]);
                File.WriteAllText(path, array.ToJsonString(CommandContext.JsonOutput) + "\n");
                return array.Count;
            }

            case "csv" or "tsv":
            {
                char delimiter = format == "csv" ? ',' : '\t';
                var all = rows.ToList();
                var columns = new List<string>();
                foreach (var row in all)
                {
                    columns.AddRange(row.Select(p => p.Key).Where(k => !columns.Contains(k)));
                }

                using var writer = new StreamWriter(path, false, new UTF8Encoding(false));
                writer.WriteLine(string.Join(delimiter, columns.Select(c => Quote(c, delimiter))));
                foreach (var row in all)
                {
                    writer.WriteLine(string.Join(delimiter, columns.Select(c => Quote(Raw(row[c]), delimiter))));
                }

                return all.Count;
            }

            case "parquet":
                throw new NotSupportedException($"{path}: writing Parquet is not in the library yet (it reads Parquet); write .jsonl, .json or .csv.");
            default:
                throw new NotSupportedException($"{path}: unknown output format; use .jsonl, .json, .csv or .tsv.");
        }
    }

    /// <summary>A value as plain text: strings as they are, numbers invariant, lists and objects as JSON, null as empty.</summary>
    public static string Raw(JsonNode? node) => node switch
    {
        null => "",
        JsonValue v when v.TryGetValue(out string? s) => s,
        _ => node.ToJsonString(),
    };

    /// <summary>A value cut to <paramref name="width"/> characters on one line, for tables.</summary>
    public static string Cell(JsonNode? node, int width = 24)
    {
        string text = Raw(node).Replace("\r", "", StringComparison.Ordinal).Replace('\n', ' ');
        return text.Length > width ? text[..(width - 3)] + "..." : text;
    }

    /// <summary>
    /// A table cell that keeps the end of a long text (a file path: its name, where <see cref="Cell"/> would keep only
    /// the folder every row shares).
    /// </summary>
    public static string CellEnd(string text, int width = 40) =>
        text.Length > width ? "..." + text[^(width - 3)..] : text;

    /// <summary>The kind of a value: integer, number, text, bool, list, object or null.</summary>
    public static string TypeOf(JsonNode? node) => node switch
    {
        null => "null",
        JsonArray => "list",
        JsonObject => "object",
        JsonValue v => v.GetValueKind() switch
        {
            JsonValueKind.Number => v.TryGetValue(out long _) || (v.TryGetValue(out double d) && d == Math.Floor(d) && Math.Abs(d) < 1e15) ? "integer" : "number",
            JsonValueKind.True or JsonValueKind.False => "bool",
            JsonValueKind.Null => "null",
            _ => "text",
        },
        _ => "text",
    };

    /// <summary>A number from a value (a number, or text that parses as one), or null.</summary>
    public static double? Number(JsonNode? node) => node switch
    {
        JsonValue v when v.GetValueKind() == JsonValueKind.Number => double.Parse(v.ToJsonString(), CultureInfo.InvariantCulture),
        JsonValue v when v.TryGetValue(out string? s) && double.TryParse(s, NumberStyles.Float, CultureInfo.InvariantCulture, out double d) => d,
        JsonValue v when v.GetValueKind() is JsonValueKind.True or JsonValueKind.False => v.GetValue<bool>() ? 1 : 0,
        _ => null,
    };

    /// <summary>The column kinds over rows: a column's kind is the most general one seen (integer, number, then text).</summary>
    public static List<(string Name, string Type, long Missing)> Columns(IReadOnlyList<JsonObject> rows)
    {
        var order = new List<string>();
        var kinds = new Dictionary<string, HashSet<string>>(StringComparer.Ordinal);
        foreach (var row in rows)
        {
            foreach (var (key, value) in row)
            {
                if (!kinds.TryGetValue(key, out var set))
                {
                    order.Add(key);
                    kinds[key] = set = [];
                }

                set.Add(TypeOf(value));
            }
        }

        return [.. order.Select(name =>
        {
            var set = kinds[name];
            set.Remove("null");
            string type = set.Count == 0 ? "null" : set.Count == 1 ? set.First()
                : set.SetEquals(["integer", "number"]) ? "number" : "mixed (" + string.Join(", ", set.Order(StringComparer.Ordinal)) + ")";
            long missing = rows.LongCount(r => r[name] is null);
            return (name, type, missing);
        })];
    }

    private static string Quote(string text, char delimiter) =>
        text.IndexOfAny([delimiter, '"', '\n', '\r']) >= 0 ? "\"" + text.Replace("\"", "\"\"", StringComparison.Ordinal) + "\"" : text;
}
