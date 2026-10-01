// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Diagnostics;
using System.Globalization;
using System.Text.Json;
using System.Text.Json.Nodes;
using Idrak.Datasets;

// idrak-data: datasets from the command line, written as JSON Lines that any tool (Idrak's finetune, Python, …) reads.
const string Usage = """
    idrak-data: inspect, download and assemble datasets

      idrak-data show <spec…>                 columns, detected layout and the first rows, as read and as conversations
      idrak-data count <spec…>                rows per source
      idrak-data download <spec…>             fetch every file of the sources into the cache and list them (--refresh: again)
      idrak-data build <spec…|recipe.json> --out <file.jsonl>
                                              assemble a training set (conversations {"messages"} and / or texts {"text"})
      idrak-data cache [--clear]              where downloads are kept (and remove them)

    A spec is a source with options after '?':
      hf:openai/gsm8k?config=main                     Hugging Face (config, split, files, max_files, revision)
      hf:HuggingFaceH4/ultrachat_200k?split=train_sft  HF_TOKEN or huggingface-cli login for gated / private data
      github:owner/repo[@ref][?files=src/**/*.cs]      a repository's files as documents (GITHUB_TOKEN)
      github:owner/repo?files=data/*.jsonl             data files in a repository
      github:owner/repo?release=latest&asset=*.csv     release assets
      kaggle:owner/dataset, zenodo:123456              (KAGGLE_USERNAME + KAGGLE_KEY or ~/.kaggle/kaggle.json)
      https://host/file.jsonl.gz, a local file or folder
    Options for any source: take, skip, weight, columns=a,b, text=lines|paragraphs|document, documents=true,
    and conversations from columns: user=…&assistant=…&system=… (e.g. user={question}&assistant={answer}).

    Options:
      --take N            show: rows to show (default 3)
      --out F             build: the output file
      --eval F            build: evaluation output (default <out>.eval.jsonl) with --eval-fraction 0.02
      --kind K            auto | chat | text (default auto: conversations when a row is one, else text)
      --system S          a system message for conversations without one
      --seed N, --max-rows N, --min-chars N, --max-chars N, --no-shuffle, --no-dedup, --mix (mix sources by weight)
      --cache DIR         download cache (default IDRAK_CACHE or ~/.cache/idrak)
      --refresh           download again even when cached
    """;

Console.OutputEncoding = System.Text.Encoding.UTF8;                       // "…" and emoji on Windows consoles
var positional = new List<string>();
string? output = null, evalOutput = null, system = null, cacheFolder = null;
double evalFraction = 0;
long take = 3, maxRows = 0;
int seed = 0, minChars = 0, maxChars = 0;
bool shuffle = true, dedup = true, mix = false, clear = false, refresh = false;
var kind = RowKind.Auto;
try
{
    for (int i = 0; i < args.Length; i++)
    {
        string Next() => i + 1 < args.Length ? args[++i] : throw new ArgumentException($"{args[i]} needs a value.");
        switch (args[i])
        {
            case "--out": output = Next(); break;
            case "--eval": evalOutput = Next(); break;
            case "--eval-fraction": evalFraction = double.Parse(Next(), CultureInfo.InvariantCulture); break;
            case "--system": system = Next(); break;
            case "--take": take = long.Parse(Next(), CultureInfo.InvariantCulture); break;
            case "--max-rows": maxRows = long.Parse(Next(), CultureInfo.InvariantCulture); break;
            case "--seed": seed = int.Parse(Next(), CultureInfo.InvariantCulture); break;
            case "--min-chars": minChars = int.Parse(Next(), CultureInfo.InvariantCulture); break;
            case "--max-chars": maxChars = int.Parse(Next(), CultureInfo.InvariantCulture); break;
            case "--kind": kind = Enum.Parse<RowKind>(Next(), ignoreCase: true); break;
            case "--cache": cacheFolder = Next(); break;
            case "--no-shuffle": shuffle = false; break;
            case "--no-dedup": dedup = false; break;
            case "--mix": mix = true; break;
            case "--clear": clear = true; break;
            case "--refresh": refresh = true; break;
            case "-h" or "--help" or "help": Console.WriteLine(Usage); return 0;
            case ['-', '-', ..]: throw new ArgumentException($"Unknown option {args[i]}.");
            default: positional.Add(args[i]); break;
        }
    }
}
catch (Exception ex) when (ex is ArgumentException or FormatException)
{
    Console.Error.WriteLine(ex.Message);
    return 1;
}

if (positional.Count == 0 || positional[0] is not ("show" or "count" or "download" or "build" or "cache")
    || positional[0] is not "cache" && positional.Count < 2 || positional[0] == "build" && output is null)
{
    Console.WriteLine(Usage);
    return 1;
}

var status = new ConsoleStatus();
var downloads = status.CreateDownloader(cacheFolder: cacheFolder is null ? null : Path.Combine(cacheFolder, "downloads"), refresh: refresh);

var specs = positional.Skip(1).ToList();
try
{
    switch (positional[0])
    {
        case "cache":
        {
            string folder = downloads.CacheFolder;
            var files = Directory.Exists(folder) ? new DirectoryInfo(folder).EnumerateFiles("*", SearchOption.AllDirectories).ToList() : [];
            Console.WriteLine($"{folder}: {Downloader.Size(files.Sum(f => f.Length))} in {files.Count} files");

            // Per source: huggingface/<kind>/<owner>/<name>, github/<owner>/<repo>, kaggle/<owner>/<dataset>, zenodo/<record>, urls/<host>.
            static string SourceOf(string relative)
            {
                var parts = relative.Split(Path.DirectorySeparatorChar, Path.AltDirectorySeparatorChar);
                int depth = parts[0] switch { "huggingface" => 4, "github" or "kaggle" => 3, "zenodo" or "urls" => 2, _ => 1 };
                return string.Join('/', parts.Take(Math.Min(depth, parts.Length - 1)));
            }

            foreach (var group in files.GroupBy(f => SourceOf(Path.GetRelativePath(folder, f.FullName))).OrderBy(g => g.Key, StringComparer.Ordinal))
            {
                Console.WriteLine($"  {Downloader.Size(group.Sum(f => f.Length)),10}  {group.Key}  ({group.Count()} file{(group.Count() == 1 ? "" : "s")})");
            }

            if (clear && Directory.Exists(folder))
            {
                Directory.Delete(folder, true);
                Console.WriteLine("cleared");
            }

            return 0;
        }

        case "show":
            foreach (var text in specs)
            {
                var spec = DatasetSpec.Parse(text);
                Console.WriteLine(spec.ToString());
                var rows = status.Track(spec.Open(downloads), "reading").Take((int)Math.Min(take, int.MaxValue)).ToList();
                Console.WriteLine($"  columns: {string.Join(", ", rows.SelectMany(r => r.Select(p => p.Key)).Distinct())}");
                Console.WriteLine($"  layout:  {(rows.Count == 0 ? "no rows" : spec.Mapping is not null ? "mapped with user= / assistant=" : ChatRows.Describe(rows[0]) ?? "not recognized (map columns with user=…&assistant=…)")}");
                if (rows.Count == 0)
                {
                    Console.WriteLine(spec.Source.StartsWith("github:", StringComparison.OrdinalIgnoreCase)
                        ? "  hint: nothing matched. Check the files= pattern, and the branch: without @branch the default branch is read (github:owner/repo@branch)."
                        : "  hint: nothing matched. Check the split, config and files= options.");
                }
                foreach (var row in rows)
                {
                    Console.WriteLine($"  row:        {Short(row)}");
                    var normalized = ChatRows.Normalize((JsonObject)row.DeepClone(), kind, spec.Mapping, system);
                    Console.WriteLine($"  normalized: {(normalized is null ? "(dropped: neither a conversation nor text)" : Short(normalized))}");
                }
            }

            return 0;

        case "count":
            foreach (var text in specs)
            {
                var watch = Stopwatch.StartNew();
                var spec = DatasetSpec.Parse(text);
                long rows = status.Track(spec.Open(downloads), "counting").LongCount();
                Console.WriteLine($"{spec}: {rows:N0} rows ({watch.Elapsed.TotalSeconds:F1} s)");
            }

            return 0;

        case "download":
            foreach (var text in specs)
            {
                var watch = Stopwatch.StartNew();
                var spec = DatasetSpec.Parse(text);
                var files = spec.Download(downloads);
                Console.WriteLine($"{spec}: {files.Count} file{(files.Count == 1 ? "" : "s")}, {Downloader.Size(files.Sum(f => new FileInfo(f).Length))} ({watch.Elapsed.TotalSeconds:F1} s)");
                foreach (var file in files)
                {
                    Console.WriteLine($"  {Downloader.Size(new FileInfo(file).Length),10}  {file}");
                }
            }

            return 0;

        default:
        {
            var recipe = specs is [var single] && single.EndsWith(".json", StringComparison.OrdinalIgnoreCase) && File.Exists(single)
                                                && JsonNode.Parse(File.ReadAllText(single)) is JsonObject json && json.ContainsKey("sources")
                ? DatasetRecipe.Load(single) is var loaded && evalFraction > 0 ? loaded with { EvaluationFraction = evalFraction } : DatasetRecipe.Load(single)
                : new DatasetRecipe
                {
                    Sources = [.. specs.Select(DatasetSpec.Parse)],
                    Kind = kind,
                    System = system,
                    MixByWeight = mix ? true : null,
                    Seed = seed,
                    Shuffle = shuffle,
                    Deduplicate = dedup,
                    MinCharacters = minChars,
                    MaxCharacters = maxChars,
                    MaxRows = maxRows,
                    EvaluationFraction = evalFraction,
                };
            Console.WriteLine($"building from {recipe.Sources.Count} source{(recipe.Sources.Count == 1 ? "" : "s")}:");
            foreach (var source in recipe.Sources)
            {
                Console.WriteLine($"  {source}");
            }

            Console.WriteLine($"  {recipe.Kind.ToString().ToLowerInvariant()} rows, "
                              + $"{(recipe.MixByWeight ?? recipe.Sources.Any(x => x.Options.ContainsKey("weight")) ? "mixed by weight" : "one source after another")}, "
                              + $"{(recipe.Deduplicate ? "duplicates removed" : "duplicates kept")}, {(recipe.Shuffle ? $"shuffled (seed {recipe.Seed})" : "in order")}"
                              + (recipe.EvaluationFraction > 0 ? $", {recipe.EvaluationFraction:P1} held out for evaluation" : ""));
            var (train, evaluation) = recipe.Build(downloads);
            var watch = Stopwatch.StartNew();
            long written = new Dataset(() => status.Track(train, "writing")).WriteJsonLines(output!);
            Console.WriteLine($"{written:N0} rows written to {output} ({watch.Elapsed.TotalSeconds:F1} s)");
            if (evaluation is not null)
            {
                string evalFile = evalOutput ?? Path.ChangeExtension(output!, null) + ".eval.jsonl";
                long held = new Dataset(() => status.Track(evaluation, "evaluation")).WriteJsonLines(evalFile);
                Console.WriteLine($"{held:N0} evaluation rows written to {evalFile}");
            }

            return 0;
        }
    }
}
catch (Exception ex) when (ex is HttpRequestException or IOException or InvalidDataException or NotSupportedException or ArgumentException
                               or InvalidOperationException or JsonException or FormatException)
{
    status.Clear();
    Console.Error.WriteLine($"error: {ex.Message}");
    return 2;
}

// A row on one line: long strings cut, long lists shortened.
static string Short(JsonNode? node)
{
    switch (node)
    {
        case JsonObject o:
            return "{ " + string.Join(", ", o.Select(p => $"\"{p.Key}\": {Short(p.Value)}")) + " }";
        case JsonArray a:
            return "[" + string.Join(", ", a.Take(6).Select(n => Short(n))) + (a.Count > 6 ? $", … {a.Count - 6} more" : "") + "]";
        case JsonValue v when v.TryGetValue<string>(out var text):
            string flat = text.Replace("\n", "⏎", StringComparison.Ordinal);
            return "\"" + (flat.Length > 160 ? flat[..160] + $"… ({text.Length} chars)" : flat).Replace("\"", "\\\"", StringComparison.Ordinal) + "\"";
        default:
            return node?.ToJsonString() ?? "null";
    }
}
