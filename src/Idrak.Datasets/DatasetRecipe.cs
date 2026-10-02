// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Globalization;
using System.Text.Json.Nodes;

namespace Idrak.Datasets;

/// <summary>
/// One source of a <see cref="DatasetRecipe"/>, written as a string with options after '?' or as a JSON object:
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
/// mapping system, user, assistant (templates over columns such as <c>user={question}</c>).
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
    public double Weight => Double("weight") ?? 1;

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

    /// <summary>The rows of this source, with take / skip / columns applied (not yet normalized to chat or text).</summary>
    public Dataset Open(Downloader? downloader = null)
    {
        var data = OpenSource(downloader);
        if (Long("skip") is { } skip)
        {
            data = data.Skip(skip);
        }

        if (Long("take") is { } take)
        {
            data = data.Take(take);
        }

        if (String("columns") is { } columns)
        {
            data = data.SelectColumns([.. columns.Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries)]);
        }

        return data;
    }

    /// <summary>Downloads the source's files into the cache (without reading rows) and returns their local paths.</summary>
    public IReadOnlyList<string> Download(Downloader? downloader = null) => OpenSource(downloader).Download();

    private Dataset OpenSource(Downloader? downloader)
    {
        var opener = DatasetSources.Find(Source);
        foreach (var key in Options.Keys.Where(k => !Known.Contains(k) && opener?.Options.Contains(k, StringComparer.OrdinalIgnoreCase) != true))
        {
            throw new ArgumentException($"{Source}: unknown option '{key}'. Known: {string.Join(", ", Known.Concat(opener?.Options ?? []).Order())}.");
        }

        var read = new ReadOptions
        {
            Text = String("text")?.ToLowerInvariant() switch
            {
                null or "lines" => TextRows.Lines,
                "paragraphs" => TextRows.Paragraphs,
                "document" or "documents" => TextRows.Document,
                var other => throw new ArgumentException($"{Source}: text={other}: use lines, paragraphs or document."),
            },

            // A built-in format by its DataFormat name, or any registered one by its name.
            Format = String("format") is { } format && Enum.TryParse<DataFormat>(format, ignoreCase: true, out var builtIn) ? builtIn : null,
            FileFormat = String("format") is { } named && !Enum.TryParse<DataFormat>(named, ignoreCase: true, out _) ? DataFileFormats.Get(named) : null,
            JsonProperty = String("json_property"),
            Documents = Flag("documents"),
        };

        // The registered sources, in order (see DatasetSources): hf:, github:, kaggle:, zenodo:, http(s)://, folders, files.
        return opener?.Open(this, read, downloader)
            ?? throw new FileNotFoundException($"'{Source}' is not a file, a folder or a known source (hf:, github:, kaggle:, zenodo:, http(s)://).");
    }

    /// <summary>The chat mapping given by the user / assistant / system options, or null to detect the layout.</summary>
    public ChatMapping? Mapping => String("user") is { } user
        ? new ChatMapping { User = user, Assistant = String("assistant") ?? throw new ArgumentException($"{Source}: user= needs assistant= too."), System = String("system") }
        : null;

    /// <inheritdoc />
    public override string ToString() => Options.Count == 0 ? Source : $"{Source}?{string.Join('&', Options.Select(p => $"{p.Key}={p.Value}"))}";

    internal string? String(string name) => Options.TryGetValue(name, out var v) ? v : null;

    internal bool Flag(string name) => String(name) is { } v && v is "true" or "1" or "yes";

    internal long? Long(string name) => String(name) is { } v ? long.Parse(v, CultureInfo.InvariantCulture) : null;

    internal int? Int(string name) => String(name) is { } v ? int.Parse(v, CultureInfo.InvariantCulture) : null;

    internal double? Double(string name) => String(name) is { } v ? double.Parse(v, CultureInfo.InvariantCulture) : null;
}

/// <summary>
/// A training set assembled from several sources: each is read, turned into conversations or text, then the sources are
/// concatenated (or mixed by weight), filtered, deduplicated, shuffled and split. As JSON:
/// <code>
/// {
///   "sources": [
///     "hf:HuggingFaceH4/ultrachat_200k?split=train_sft&amp;take=20000",
///     {"source": "hf:openai/gsm8k", "config": "main", "user": "{question}", "assistant": "{answer}", "weight": 0.5},
///     {"source": "data/my-examples.jsonl", "weight": 2}
///   ],
///   "kind": "chat", "system": "You are a helpful assistant.",
///   "mix": "weights", "seed": 1, "shuffle": true, "deduplicate": true,
///   "min_chars": 20, "max_chars": 40000, "max_rows": 100000, "eval_fraction": 0.02
/// }
/// </code>
/// </summary>
public sealed record DatasetRecipe
{
    /// <summary>The sources.</summary>
    public required IReadOnlyList<DatasetSpec> Sources { get; init; }

    /// <summary>Conversations, text, or whichever each row is.</summary>
    public RowKind Kind { get; init; } = RowKind.Auto;

    /// <summary>A system message added to conversations that have none.</summary>
    public string? System { get; init; }

    /// <summary>Mix the sources by their weights (default: when any weight is given) instead of one after another.</summary>
    public bool? MixByWeight { get; init; }

    /// <summary>With mixing: stop when the first source runs out (proportions hold) or when all have.</summary>
    public MixStop Stop { get; init; } = MixStop.AllExhausted;

    /// <summary>Seed of shuffling, mixing and splitting.</summary>
    public int Seed { get; init; }

    /// <summary>Shuffle the rows (buffered, see <see cref="Dataset.Shuffle"/>).</summary>
    public bool Shuffle { get; init; } = true;

    /// <summary>Drop repeated rows.</summary>
    public bool Deduplicate { get; init; } = true;

    /// <summary>Drop rows with fewer characters of text.</summary>
    public int MinCharacters { get; init; }

    /// <summary>Drop rows with more characters of text (0: no limit).</summary>
    public int MaxCharacters { get; init; }

    /// <summary>At most this many rows (0: all).</summary>
    public long MaxRows { get; init; }

    /// <summary>
    /// Share of rows held out for evaluation, by a hash of each row's prompt (see <see cref="Dataset.Split"/>): everything
    /// but the assistant's turns, lower-cased with whitespace collapsed, or a text row's text. The same question with
    /// different answers, or differently spaced or cased, lands on one side, so the evaluation holds only unseen prompts.
    /// </summary>
    public double EvaluationFraction { get; init; }

    /// <summary>A recipe of the given sources with the defaults.</summary>
    public static DatasetRecipe Of(params string[] sources) => new() { Sources = [.. sources.Select(DatasetSpec.Parse)] };

    /// <summary>Reads a recipe file (see the class summary).</summary>
    public static DatasetRecipe Load(string path) => FromJson(JsonNode.Parse(File.ReadAllText(path)) as JsonObject
                                                               ?? throw new InvalidDataException($"{path} is not a JSON object."), Path.GetDirectoryName(Path.GetFullPath(path)));

    /// <summary>Reads a recipe; local sources are relative to <paramref name="folder"/>.</summary>
    public static DatasetRecipe FromJson(JsonObject json, string? folder = null)
    {
        var sources = new List<DatasetSpec>();
        foreach (var node in json["sources"] as JsonArray ?? throw new InvalidDataException("A recipe needs \"sources\": [...]."))
        {
            var spec = node switch
            {
                JsonValue v => DatasetSpec.Parse((string)v!),
                JsonObject o => DatasetSpec.FromJson(o),
                _ => throw new InvalidDataException("Each recipe source is a string or an object."),
            };
            if (folder is not null && !spec.Source.Contains(':', StringComparison.Ordinal) && !Path.IsPathRooted(spec.Source))
            {
                var o = new JsonObject { ["source"] = Path.Combine(folder, spec.Source) };
                foreach (var (k, v) in spec.Options)
                {
                    o[k] = v;
                }

                spec = DatasetSpec.FromJson(o);
            }

            sources.Add(spec);
        }

        return new DatasetRecipe
        {
            Sources = sources,
            Kind = (string?)json["kind"] is { } kind ? Enum.Parse<RowKind>(kind, ignoreCase: true) : RowKind.Auto,
            System = (string?)json["system"],
            MixByWeight = (string?)json["mix"] is { } mix ? mix.Equals("weights", StringComparison.OrdinalIgnoreCase) : null,
            Stop = (string?)json["stop"] is "first" ? MixStop.FirstExhausted : MixStop.AllExhausted,
            Seed = (int?)json["seed"] ?? 0,
            Shuffle = (bool?)json["shuffle"] ?? true,
            Deduplicate = (bool?)json["deduplicate"] ?? true,
            MinCharacters = (int?)json["min_chars"] ?? 0,
            MaxCharacters = (int?)json["max_chars"] ?? 0,
            MaxRows = (long?)json["max_rows"] ?? 0,
            EvaluationFraction = (double?)json["eval_fraction"] ?? 0,
        };
    }

    /// <summary>
    /// The training rows and, with <see cref="EvaluationFraction"/> &gt; 0, the evaluation rows: conversations
    /// ({"messages", "tools"}) and / or texts ({"text"}), as <see cref="ChatRows"/> normalizes them.
    /// </summary>
    public (Dataset Train, Dataset? Evaluation) Build(Downloader? downloader = null) => Build(downloader, null);

    /// <summary>
    /// <see cref="Build(Downloader?)"/>, with <paramref name="counts"/> filled as the rows are read: how many the
    /// sources gave, and how many the length limits and the deduplication dropped (for the last pass over the rows).
    /// </summary>
    public (Dataset Train, Dataset? Evaluation) Build(Downloader? downloader, RecipeCounts? counts)
    {
        if (Sources.Count == 0)
        {
            throw new InvalidOperationException("The recipe has no sources.");
        }

        var parts = Sources.Select(s => (Data: ChatRows.Normalize(s.Open(downloader), Kind, s.Mapping, System), s.Weight)).ToList();
        bool mix = MixByWeight ?? Sources.Any(s => s.Options.ContainsKey("weight"));
        var data = parts.Count == 1 ? parts[0].Data : mix ? Dataset.Mix(parts, Seed, Stop) : Dataset.Concat([.. parts.Select(p => p.Data)]);
        if (counts is not null)
        {
            var sources = data;
            data = new Dataset(() => Count(sources, counts), sources.Name);
        }

        if (MinCharacters > 0 || MaxCharacters > 0)
        {
            data = data.Where(row =>
            {
                long length = Length(row);
                bool kept = length >= MinCharacters && (MaxCharacters <= 0 || length <= MaxCharacters);
                if (!kept && counts is not null)
                {
                    counts.OutsideLengths++;
                }

                return kept;
            });
        }

        if (Deduplicate)
        {
            data = data.Deduplicate();
            if (counts is not null)
            {
                var distinct = data;
                data = distinct.Select(row =>
                {
                    counts.Kept++;
                    return row;
                });
            }
        }

        if (Shuffle)
        {
            data = data.Shuffle(Seed);
        }

        if (EvaluationFraction <= 0)
        {
            return (MaxRows > 0 ? data.Take(MaxRows) : data, null);
        }

        var (train, evaluation) = data.Split(EvaluationFraction, Seed, Prompt);
        return (MaxRows > 0 ? train.Take(MaxRows) : train, evaluation);
    }

    // Each pass over the rows starts the counts again.
    private static IEnumerable<JsonObject> Count(Dataset rows, RecipeCounts counts)
    {
        counts.Reset();
        foreach (var row in rows)
        {
            counts.Rows++;
            yield return row;
        }
    }

    // What a row asks: every message but the assistant's (a preference row's prompt), or the text, lower-cased, whitespace
    // collapsed.
    private static string Prompt(JsonObject row)
    {
        string text = (row["messages"] ?? row["prompt"]) is JsonArray messages
            ? string.Join("\u0001", messages.Where(m => (string?)m?["role"] != "assistant").Select(m => $"{m?["role"]}:{m?["content"]}"))
            : ((string?)row["text"]) ?? row.ToJsonString();
        return string.Join(' ', text.Split((char[]?)null, StringSplitOptions.RemoveEmptyEntries)).ToLowerInvariant();
    }

    private static long Length(JsonObject row) => row["messages"] is JsonArray messages ? Length(messages)
        : row["prompt"] is JsonArray prompt ? Length(prompt) + Length(row["chosen"] as JsonArray) + Length(row["rejected"] as JsonArray)
        : ((string?)row["text"])?.Length ?? 0;

    private static long Length(JsonArray? messages) => messages?.Sum(m => (long)(((string?)m?["content"])?.Length ?? 0)) ?? 0;
}

/// <summary>What one pass over a <see cref="DatasetRecipe"/>'s rows read and dropped (see <see cref="DatasetRecipe.Build(Downloader?, RecipeCounts?)"/>).</summary>
public sealed class RecipeCounts
{
    /// <summary>Rows the sources gave (conversations and texts).</summary>
    public long Rows { get; internal set; }

    /// <summary>Rows dropped by the length limits.</summary>
    public long OutsideLengths { get; internal set; }

    /// <summary>Rows kept by the deduplication (when on).</summary>
    public long Kept { get; internal set; }

    /// <summary>Rows dropped as repeats of earlier ones (when deduplicating).</summary>
    public long Duplicates => Kept == 0 ? 0 : Rows - OutsideLengths - Kept;

    internal void Reset() => (Rows, OutsideLengths, Kept) = (0, 0, 0);
}
