// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Globalization;
using System.Text.Json.Nodes;

namespace Idrak.Abstraction.Data;

/// <summary>
/// One source of a dataset recipe (<c>DatasetRecipe</c> in Idrak.Data), written as a string with options after '?' or as a JSON object:
/// <list type="bullet">
/// <item><c>hf:HuggingFaceH4/ultrachat_200k?split=train_sft</c> (options config, split, files, max_files, revision)</item>
/// <item><c>github:owner/repo[@ref]</c> (a repository's files as documents, files=src/**/*.cs to narrow them; files=data/*.jsonl reads
/// data files instead; release=latest|tag with asset=*.csv reads release assets)</item>
/// <item><c>kaggle:owner/dataset</c> (files), <c>zenodo:123456</c> (files)</item>
/// <item><c>https://host/data.jsonl.gz</c>, or a local file or folder (files)</item>
/// <item>any source added with <see cref="DatasetSources.Register"/></item>
/// </list>
/// Options for any source: take, skip, weight, text (lines | paragraphs | document), documents (every file one row),
/// columns (a,b,…), format (a <see cref="DataFileFormats"/> name), json_property, and the chat
/// mapping system, user, assistant (templates over columns such as <c>user={question}</c>). Idrak.Data opens it
/// (<c>spec.Open()</c>, through the registered <see cref="DatasetSources"/>).
/// </summary>
public sealed class DatasetSpec
{
    private DatasetSpec(string source, Dictionary<string, string> options)
    {
        Source = source;
        Options = options;
    }

    /// <summary>The source, without options.</summary>
    public string Source { get; }

    /// <summary>The options (lower-case names).</summary>
    public IReadOnlyDictionary<string, string> Options { get; }

    /// <summary>The source's weight in a mix (default 1).</summary>
    public double Weight => GetDouble("weight") ?? 1;

    /// <summary>Reads <c>source?name=value&amp;name=value</c>.</summary>
    public static DatasetSpec Parse(string text)
    {
        var options = new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase);
        int question = text.IndexOf('?', StringComparison.Ordinal);
        bool isUrl = text.StartsWith("http://", StringComparison.OrdinalIgnoreCase) || text.StartsWith("https://", StringComparison.OrdinalIgnoreCase);
        string source = text;
        if (question > 0)
        {
            // A URL keeps its own query unless the part after '?' holds only options this class knows.
            var pairs = text[(question + 1)..].Split('&', StringSplitOptions.RemoveEmptyEntries).Select(p => p.Split('=', 2)).ToList();
            if (!isUrl || pairs.All(p => Known.Contains(p[0])))
            {
                source = text[..question];
                foreach (var pair in pairs)
                {
                    options[pair[0]] = pair.Length > 1 ? Uri.UnescapeDataString(pair[1].Replace('+', ' ')) : "true";
                }
            }
        }

        return new DatasetSpec(source.Trim(), options);
    }

    /// <summary>Reads a JSON object: {"source": "...", "split": "...", "weight": 2, ...}.</summary>
    public static DatasetSpec FromJson(JsonObject json)
    {
        var options = new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase);
        foreach (var (key, value) in json)
        {
            if (key != "source" && value is not null)
            {
                options[key] = value is JsonValue v && v.TryGetValue<string>(out var s) ? s : value.ToJsonString();
            }
        }

        return new DatasetSpec((string?)json["source"] ?? throw new InvalidDataException("A recipe source needs \"source\"."), options);
    }

    private static readonly HashSet<string> Known = new(["config", "split", "files", "max_files", "revision", "ref", "release", "asset", "take", "skip", "weight",
        "text", "columns", "system", "user", "assistant", "format", "json_property", "documents"], StringComparer.OrdinalIgnoreCase);

    /// <summary>The options every source reads (or that the built-in sources read); a source adds its own with <see cref="IDatasetSource.Options"/>.</summary>
    public static IReadOnlyCollection<string> CommonOptions => Known;

    /// <inheritdoc />
    public override string ToString() => Options.Count == 0 ? Source : $"{Source}?{string.Join('&', Options.Select(p => $"{p.Key}={p.Value}"))}";

    /// <summary>The option <paramref name="name"/> (ignoring case), or null when the spec does not set it.</summary>
    public string? Get(string name) => Options.TryGetValue(name, out var v) ? v : null;

    /// <summary>Whether the option <paramref name="name"/> is set to true, 1 or yes.</summary>
    public bool Flag(string name) => Get(name) is { } v && v is "true" or "1" or "yes";

    /// <summary>The integer option <paramref name="name"/>, or null when the spec does not set it.</summary>
    public long? GetInt64(string name) => Get(name) is { } v ? long.Parse(v, CultureInfo.InvariantCulture) : null;

    /// <summary>The integer option <paramref name="name"/>, or null when the spec does not set it.</summary>
    public int? GetInt32(string name) => Get(name) is { } v ? int.Parse(v, CultureInfo.InvariantCulture) : null;

    /// <summary>The number option <paramref name="name"/>, or null when the spec does not set it.</summary>
    public double? GetDouble(string name) => Get(name) is { } v ? double.Parse(v, CultureInfo.InvariantCulture) : null;
}
