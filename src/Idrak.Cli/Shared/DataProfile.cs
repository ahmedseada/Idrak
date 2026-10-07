// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Globalization;
using System.Text.Json.Nodes;
using System.Text.Json;
using System.Text;
using Idrak.Data.Abstractions;
using Idrak.Data;

namespace Idrak.Cli.Shared;

/// <summary>What a data file or folder holds, as <c>idrak suggest</c> reads it.</summary>
internal enum DataKind
{
    /// <summary>Rows with columns (CSV, TSV, JSON Lines, JSON, Parquet): a table, possibly with text columns.</summary>
    Table,

    /// <summary>A folder of class folders holding images.</summary>
    Images,

    /// <summary>Conversations or plain text rows (the layouts <see cref="ChatRows"/> reads).</summary>
    Chat,

    /// <summary>Preference pairs (chosen and rejected answers).</summary>
    Preference,
}

/// <summary>The values of one column: types, missing cells, distinct values, numeric moments and text lengths.</summary>
internal sealed class ColumnProfile(string name)
{
    // Distinct values are counted up to this many; beyond it the column is "many distinct values".
    private const int DistinctCap = 10_000;

    private readonly List<double> _numbers = [];
    private readonly List<int> _words = [];

    public string Name { get; } = name;

    public int Missing { get; private set; }

    public int Numbers { get; private set; }

    public int Strings { get; private set; }

    public int Booleans { get; private set; }

    public int Others { get; private set; }

    /// <summary>Distinct values (as text) and their counts, up to 10,000 values.</summary>
    public Dictionary<string, int> Distinct { get; } = new(StringComparer.Ordinal);

    /// <summary>Whether there were more distinct values than are counted.</summary>
    public bool ManyDistinct { get; private set; }

    public int Present => Numbers + Strings + Booleans + Others;

    /// <summary>
    /// The column's type: "number" (numbers, or booleans as 0/1), "text" (strings of five words or more on average),
    /// "category" (other strings), or "other" (lists and objects). Mixed columns take the type of most of their values;
    /// numbers stored as strings count as numbers.
    /// </summary>
    public string Type => Present == 0 ? "other"
        : Others * 2 > Present ? "other"
        : (Numbers + Booleans) * 10 >= Present * 9 ? "number"
        : MeanWords >= 5 ? "text"
        : "category";

    public double Mean => _numbers.Count == 0 ? 0 : _numbers.Average();

    public double Std
    {
        get
        {
            if (_numbers.Count < 2)
            {
                return 0;
            }

            double mean = Mean;
            return Math.Sqrt(_numbers.Sum(v => (v - mean) * (v - mean)) / (_numbers.Count - 1));
        }
    }

    /// <summary>The sample skewness of the numbers (third standardized moment); 0 for fewer than three values.</summary>
    public double Skew
    {
        get
        {
            double std = Std, mean = Mean;
            return _numbers.Count < 3 || std == 0 ? 0 : _numbers.Sum(v => Math.Pow((v - mean) / std, 3)) / _numbers.Count;
        }
    }

    public double Min => _numbers.Count == 0 ? 0 : _numbers.Min();

    public double Max => _numbers.Count == 0 ? 0 : _numbers.Max();

    public double Median => Quantile([.. _numbers], 0.5);

    /// <summary>Whether every number is a whole number.</summary>
    public bool Integers => _numbers.All(v => v == Math.Floor(v));

    public double MeanWords => _words.Count == 0 ? 0 : _words.Average();

    public int MaxWords => _words.Count == 0 ? 0 : _words.Max();

    /// <summary>The word count below which <paramref name="q"/> of the strings fall.</summary>
    public int WordsQuantile(double q) => (int)Math.Ceiling(Quantile([.. _words.Select(v => (double)v)], q));

    public void Add(JsonNode? node)
    {
        switch (node)
        {
            case null:
                Missing++;
                return;
            case JsonValue v when v.GetValueKind() == JsonValueKind.Number:
                Numbers++;
                _numbers.Add(DataPreparation.ParseNumber(v));
                Count(DataPreparation.Key(v)!);
                return;
            case JsonValue v when v.GetValueKind() is JsonValueKind.True or JsonValueKind.False:
                Booleans++;
                bool truth = v.GetValueKind() == JsonValueKind.True;
                _numbers.Add(truth ? 1 : 0);
                Count(truth ? "true" : "false");
                return;
            case JsonValue v when v.TryGetValue(out string? text):
                if (string.IsNullOrWhiteSpace(text))
                {
                    Missing++;
                    return;
                }

                if (double.TryParse(text, NumberStyles.Float, CultureInfo.InvariantCulture, out double parsed) && double.IsFinite(parsed))
                {
                    Numbers++;
                    _numbers.Add(parsed);
                }
                else
                {
                    Strings++;
                }

                _words.Add(Words(text));
                Count(text);
                return;
            default:
                Others++;
                return;
        }
    }

    private void Count(string value)
    {
        if (Distinct.TryGetValue(value, out int count))
        {
            Distinct[value] = count + 1;
        }
        else if (Distinct.Count < DistinctCap)
        {
            Distinct[value] = 1;
        }
        else
        {
            ManyDistinct = true;
        }
    }

    /// <summary>Whitespace-separated words in <paramref name="text"/>.</summary>
    public static int Words(string text)
    {
        int words = 0;
        bool inWord = false;
        foreach (char c in text)
        {
            bool space = char.IsWhiteSpace(c);
            words += !space && !inWord ? 1 : 0;
            inWord = !space;
        }

        return words;
    }

    /// <summary>The <paramref name="q"/> quantile of <paramref name="values"/> (nearest rank), or 0 for none.</summary>
    public static double Quantile(List<double> values, double q)
    {
        if (values.Count == 0)
        {
            return 0;
        }

        values.Sort();
        return values[Math.Clamp((int)Math.Ceiling(q * values.Count) - 1, 0, values.Count - 1)];
    }
}

/// <summary>One image of an image folder: its file, class and header.</summary>
internal readonly record struct ImageItem(string Path, int Class, ImageFiles.ImageInfo Info);

/// <summary>
/// Step 1 of <c>idrak suggest</c> (plans/idrak-cli.md, "Design a model"): reads the data and describes it. Tables come
/// from the readers of Idrak.Data (CSV, TSV, JSON Lines, JSON, Parquet, also compressed or in a folder); a folder
/// whose sub-folders hold images is an image set (one class per folder); rows the chat readers recognize are
/// conversations, or preference pairs when they have chosen and rejected answers.
/// </summary>
internal sealed class DataProfile
{
    /// <summary>At most this many rows are read and profiled; the rest are counted.</summary>
    public const int MaxRows = 200_000;

    private DataProfile(string path, DataKind kind)
    {
        Path = path;
        Kind = kind;
    }

    public string Path { get; }

    public DataKind Kind { get; }

    /// <summary>The file format ("csv", "jsonl", "parquet", ..., "images").</summary>
    public string Format { get; private set; } = "";

    /// <summary>Rows (or images) in the data.</summary>
    public long TotalRows { get; private set; }

    /// <summary>The rows read (at most <see cref="MaxRows"/>).</summary>
    public List<JsonObject> Rows { get; } = [];

    /// <summary>Rows that repeat an earlier row exactly.</summary>
    public int Duplicates { get; private set; }

    public List<ColumnProfile> Columns { get; } = [];

    /// <summary>Images, with the class folder names in <see cref="ClassNames"/>.</summary>
    public List<ImageItem> Images { get; } = [];

    public List<string> ClassNames { get; } = [];

    /// <summary>Normalized conversations (<c>{"messages"}</c> or <c>{"text"}</c>) or preference rows.</summary>
    public List<JsonObject> Conversations { get; } = [];

    /// <summary>The layout the chat readers saw in the first rows ("messages", "sharegpt", "alpaca", ...).</summary>
    public string? Layout { get; private set; }

    public bool Truncated => TotalRows > Rows.Count && Kind != DataKind.Images;

    /// <summary>The column named <paramref name="name"/> (ignoring case), or null.</summary>
    public ColumnProfile? Column(string name) =>
        Columns.FirstOrDefault(c => c.Name == name) ?? Columns.FirstOrDefault(c => string.Equals(c.Name, name, StringComparison.OrdinalIgnoreCase));

    /// <summary>The share of missing cells over every column.</summary>
    public double MissingShare => Columns.Count == 0 || Rows.Count == 0 ? 0 : Columns.Sum(c => (double)c.Missing) / (Columns.Count * (double)Rows.Count);

    /// <summary>
    /// Reads <paramref name="path"/>. With <paramref name="target"/> set, rows are a table (the target is a column);
    /// with <paramref name="task"/> "chat" or "preference" they are read as conversations or pairs.
    /// </summary>
    public static DataProfile Read(string path, string? target, string? task)
    {
        string full = System.IO.Path.GetFullPath(path);
        if (Directory.Exists(full) && ImageClasses(full) is { Count: > 0 } classes)
        {
            return ReadImages(full, classes);
        }

        DatasetRows data;
        string format;
        if (Directory.Exists(full))
        {
            data = DatasetRows.FromFolder(full);
            format = "folder";
        }
        else if (File.Exists(full))
        {
            data = DatasetRows.FromFile(full);
            format = DataFileFormats.Find(full)?.Name.ToLowerInvariant() ?? System.IO.Path.GetExtension(full).TrimStart('.');
        }
        else
        {
            throw new UsageException($"No data at {full}: give a CSV, JSON Lines, JSON or Parquet file, or a folder of class folders of images.");
        }

        var rows = new List<JsonObject>();
        long total = 0;
        foreach (var row in data)
        {
            if (rows.Count < MaxRows)
            {
                rows.Add(row);
            }

            total++;
        }

        if (rows.Count == 0)
        {
            throw new InvalidDataException($"{full} has no rows.");
        }

        var kind = task switch
        {
            "chat" => DataKind.Chat,
            "preference" => DataKind.Preference,
            _ when target is not null || task is "regression" or "classify" or "sequence" => DataKind.Table,
            _ => Detect(rows),
        };
        var profile = new DataProfile(full, kind) { Format = format, TotalRows = total };
        profile.Rows.AddRange(rows);
        profile.Layout = ChatRows.Describe(rows[0]);
        var seen = new HashSet<string>(StringComparer.Ordinal);
        foreach (var row in rows)
        {
            profile.Duplicates += seen.Add(row.ToJsonString()) ? 0 : 1;
        }

        if (kind == DataKind.Table)
        {
            var names = new List<string>();
            foreach (var row in rows.Take(1000))
            {
                names.AddRange(row.Select(p => p.Key).Where(k => !names.Contains(k)));
            }

            foreach (string name in names)
            {
                var column = new ColumnProfile(name);
                foreach (var row in rows)
                {
                    column.Add(row[name]);
                }

                profile.Columns.Add(column);
            }
        }
        else
        {
            foreach (var row in rows)
            {
                var normalized = kind == DataKind.Preference ? ChatRows.Preference(row) : ChatRows.Normalize(row);
                if (normalized is not null)
                {
                    profile.Conversations.Add(normalized);
                }
            }

            if (profile.Conversations.Count == 0)
            {
                throw new InvalidDataException(kind == DataKind.Preference
                    ? $"No preference pairs in {full}: rows need chosen and rejected answers (and a prompt)."
                    : $"No conversations or text in {full}: see 'idrak data validate {path} --as chat'.");
            }
        }

        return profile;
    }

    /// <summary>
    /// The text of a conversation as the language-model rules count it: one "role: content" line per message, or the
    /// text of a text row; preference rows give the prompt and the chosen answer.
    /// </summary>
    public static string Render(JsonObject row)
    {
        if ((string?)row["text"] is { } text)
        {
            return text;
        }

        var sb = new StringBuilder();
        foreach (string list in new[] { "messages", "prompt", "chosen" })
        {
            if (row[list] is JsonArray messages)
            {
                foreach (var m in messages.OfType<JsonObject>())
                {
                    sb.Append((string?)m["role"] ?? "user").Append(": ").Append(Text(m["content"])).Append('\n');
                }
            }
        }

        return sb.ToString();
    }

    private static string Text(JsonNode? node) => node switch
    {
        JsonValue v when v.TryGetValue(out string? s) => s,
        JsonArray parts => string.Join(" ", parts.Select(p => p is JsonObject o ? (string?)o["text"] ?? "" : Text(p))),
        null => "",
        _ => node.ToJsonString(),
    };

    // Most of the first rows are conversations (or pairs) when the chat readers recognize them, or rows of nothing but
    // text; otherwise a table (a text column next to other columns is a table with text).
    private static DataKind Detect(List<JsonObject> rows)
    {
        var sample = rows.Take(50).ToList();
        int pairs = sample.Count(r => ChatRows.Preference(r) is not null);
        if (pairs * 5 >= sample.Count * 4)
        {
            return DataKind.Preference;
        }

        int chats = sample.Count(r => ChatRows.Describe(r) is { } layout && (layout != "text" || r.Count == 1));
        return chats * 5 >= sample.Count * 4 ? DataKind.Chat : DataKind.Table;
    }

    // The sub-folders that hold image files, in name order (the class order).
    private static List<string> ImageClasses(string folder) =>
        [.. Directory.EnumerateDirectories(folder).Where(d => Directory.EnumerateFiles(d).Any(ImageFiles.IsImage)).Order(StringComparer.Ordinal)];

    private static DataProfile ReadImages(string folder, List<string> classes)
    {
        var profile = new DataProfile(folder, DataKind.Images) { Format = "images" };
        for (int c = 0; c < classes.Count; c++)
        {
            profile.ClassNames.Add(System.IO.Path.GetFileName(classes[c]));
            foreach (string file in Directory.EnumerateFiles(classes[c], "*", SearchOption.AllDirectories).Where(ImageFiles.IsImage).Order(StringComparer.Ordinal))
            {
                if (ImageFiles.ReadHeader(file) is { } info)
                {
                    profile.Images.Add(new ImageItem(file, c, info));
                }
            }
        }

        profile.TotalRows = profile.Images.Count;
        if (profile.Images.Count == 0)
        {
            throw new InvalidDataException($"No readable images in the class folders of {folder} (PNG, JPEG, BMP, PGM or PPM).");
        }

        return profile;
    }
}
