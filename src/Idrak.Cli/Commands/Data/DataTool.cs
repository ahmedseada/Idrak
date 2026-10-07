// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Diagnostics;
using System.Globalization;
using System.Text.Json;
using System.Text.Json.Nodes;
using Idrak.Cli.Shared;
using Idrak.Data;

namespace Idrak.Cli.Commands.Data;

/// <summary>
/// The dataset tool: show, count, download and assemble datasets, written as JSON Lines that any tool reads.
/// Run by <c>idrak data</c> (<see cref="DataCommand"/>, which turns idrak's options into these). <see cref="Parse"/>
/// reads the arguments and <see cref="Execute"/> runs the command (its errors are exceptions, which idrak prints and
/// exits with 1).
/// </summary>
internal sealed class DataTool(ToolConsole console)
{
    public const string Usage = """
        Inspect, download and assemble datasets.

          idrak data show <spec…>                 columns, detected layout and the first rows, as read and as conversations
          idrak data count <spec…>                rows per source
          idrak data download <spec…>             fetch every file of the sources into the cache and list them (--refresh: again)
          idrak data build <spec…|recipe.json> --out <file.jsonl>
                                                  assemble a training set (conversations {"messages"} and / or texts {"text"})
          idrak data cache [--clear]              where downloads are kept (and remove them)

        A spec is a source with options after '?':
          hf:owner/qa-set?config=main                     Hugging Face (config, split, files, max_files, revision)
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

    /// <summary>The commands, the first argument.</summary>
    public static readonly string[] Commands = ["show", "count", "download", "build", "cache"];

    /// <summary>The options that take a value.</summary>
    public static readonly string[] ValueOptions =
        ["--out", "--eval", "--eval-fraction", "--system", "--take", "--max-rows", "--seed", "--min-chars", "--max-chars", "--kind", "--cache"];

    /// <summary>The flags.</summary>
    public static readonly string[] Flags = ["--no-shuffle", "--no-dedup", "--mix", "--clear", "--refresh"];

    private readonly TextWriter _out = console.Out;
    private readonly List<string> positional = [];
    private string? output, evalOutput, system, cacheFolder;
    private double evalFraction;
    private long take = 3, maxRows;
    private int seed, minChars, maxChars;
    private bool shuffle = true, dedup = true, mix, clear, refresh;
    private RowKind kind = RowKind.Auto;

    /// <summary>The arguments that are not options: the command and the specs.</summary>
    public IReadOnlyList<string> Positional => positional;

    /// <summary>The failures the tool reports as a message (others are bugs and keep their stack).</summary>
    public static bool IsDataError(Exception ex) =>
        ex is HttpRequestException or IOException or InvalidDataException or NotSupportedException or ArgumentException
            or InvalidOperationException or JsonException or FormatException;

    /// <summary>Reads the arguments; false when they ask for the help.</summary>
    public bool Parse(IReadOnlyList<string> args)
    {
        for (int i = 0; i < args.Count; i++)
        {
            string Next() => i + 1 < args.Count ? args[++i] : throw new ArgumentException($"{args[i]} needs a value.");
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
                case "-h" or "--help" or "help": return false;
                case ['-', '-', ..]: throw new ArgumentException($"Unknown option {args[i]}.");
                default: positional.Add(args[i]); break;
            }
        }

        return true;
    }

    /// <summary>What is missing from the arguments for the command to run (null when nothing is).</summary>
    public string? Problem() =>
        positional.Count == 0 || !Commands.Contains(positional[0]) ? $"Give a command: {string.Join(", ", Commands)}."
        : positional[0] is not "cache" && positional.Count < 2 ? $"{positional[0]} needs one or more specs (files, folders, hf:, github:, kaggle:, zenodo: or URLs)."
        : positional[0] == "build" && output is null ? "build needs --out FILE."
        : null;

    /// <summary>Runs the parsed command; returns the exit code. Failures are exceptions (<see cref="IsDataError"/>).</summary>
    public int Execute()
    {
        var downloads = console.CreateDownloader(cacheFolder: cacheFolder is null ? null : Path.Combine(cacheFolder, "downloads"), refresh: refresh);
        var specs = positional.Skip(1).ToList();
        switch (positional[0])
        {
            case "cache":
            {
                string folder = downloads.CacheFolder;
                var files = Directory.Exists(folder) ? new DirectoryInfo(folder).EnumerateFiles("*", SearchOption.AllDirectories).ToList() : [];
                _out.WriteLine($"{folder}: {Downloader.Size(files.Sum(f => f.Length))} in {files.Count} files");

                // Per source: huggingface/<kind>/<owner>/<name>, github/<owner>/<repo>, kaggle/<owner>/<dataset>, zenodo/<record>, urls/<host>.
                static string SourceOf(string relative)
                {
                    var parts = relative.Split(Path.DirectorySeparatorChar, Path.AltDirectorySeparatorChar);
                    int depth = parts[0] switch { "huggingface" => 4, "github" or "kaggle" => 3, "zenodo" or "urls" => 2, _ => 1 };
                    return string.Join('/', parts.Take(Math.Min(depth, parts.Length - 1)));
                }

                foreach (var group in files.GroupBy(f => SourceOf(Path.GetRelativePath(folder, f.FullName))).OrderBy(g => g.Key, StringComparer.Ordinal))
                {
                    _out.WriteLine($"  {Downloader.Size(group.Sum(f => f.Length)),10}  {group.Key}  ({group.Count()} file{(group.Count() == 1 ? "" : "s")})");
                }

                if (clear && Directory.Exists(folder))
                {
                    Directory.Delete(folder, true);
                    _out.WriteLine("cleared");
                }

                return 0;
            }

            case "show":
                foreach (var text in specs)
                {
                    var spec = DatasetSpec.Parse(text);
                    _out.WriteLine(spec.ToString());
                    var rows = console.Track(spec.Open(downloads), "reading").Take((int)Math.Min(take, int.MaxValue)).ToList();
                    _out.WriteLine($"  columns: {string.Join(", ", rows.SelectMany(r => r.Select(p => p.Key)).Distinct())}");
                    _out.WriteLine($"  layout:  {(rows.Count == 0 ? "no rows" : spec.Mapping is not null ? "mapped with user= / assistant=" : ChatRows.Describe(rows[0]) ?? "not recognized (map columns with user=…&assistant=…)")}");
                    if (rows.Count == 0)
                    {
                        _out.WriteLine(spec.Source.StartsWith("github:", StringComparison.OrdinalIgnoreCase)
                            ? "  hint: nothing matched. Check the files= pattern, and the branch: without @branch the default branch is read (github:owner/repo@branch)."
                            : "  hint: nothing matched. Check the split, config and files= options.");
                    }

                    foreach (var row in rows)
                    {
                        _out.WriteLine($"  row:        {Short(row)}");
                        var normalized = ChatRows.Normalize((JsonObject)row.DeepClone(), kind, spec.Mapping, system);
                        _out.WriteLine($"  normalized: {(normalized is null ? "(dropped: neither a conversation nor text)" : Short(normalized))}");
                    }
                }

                return 0;

            case "count":
                foreach (var text in specs)
                {
                    var watch = Stopwatch.StartNew();
                    var spec = DatasetSpec.Parse(text);
                    long rows = console.Track(spec.Open(downloads), "counting").LongCount();
                    _out.WriteLine($"{spec}: {rows:N0} rows ({watch.Elapsed.TotalSeconds:F1} s)");
                }

                return 0;

            case "download":
                foreach (var text in specs)
                {
                    var watch = Stopwatch.StartNew();
                    var spec = DatasetSpec.Parse(text);
                    var files = spec.Download(downloads);
                    _out.WriteLine($"{spec}: {files.Count} file{(files.Count == 1 ? "" : "s")}, {Downloader.Size(files.Sum(f => new FileInfo(f).Length))} ({watch.Elapsed.TotalSeconds:F1} s)");
                    foreach (var file in files)
                    {
                        _out.WriteLine($"  {Downloader.Size(new FileInfo(file).Length),10}  {file}");
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
                _out.WriteLine($"building from {recipe.Sources.Count} source{(recipe.Sources.Count == 1 ? "" : "s")}:");
                foreach (var source in recipe.Sources)
                {
                    _out.WriteLine($"  {source}");
                }

                _out.WriteLine($"  {recipe.Kind.ToString().ToLowerInvariant()} rows, "
                               + $"{(recipe.MixByWeight ?? recipe.Sources.Any(x => x.Options.ContainsKey("weight")) ? "mixed by weight" : "one source after another")}, "
                               + $"{(recipe.Deduplicate ? "duplicates removed" : "duplicates kept")}, {(recipe.Shuffle ? $"shuffled (seed {recipe.Seed})" : "in order")}"
                               + (recipe.EvaluationFraction > 0 ? $", {recipe.EvaluationFraction:P1} held out for evaluation" : ""));
                var (train, evaluation) = recipe.Build(downloads);
                var buildWatch = Stopwatch.StartNew();
                long written = new DatasetRows(() => console.Track(train, "writing")).WriteJsonLines(output!);
                _out.WriteLine($"{written:N0} rows written to {output} ({buildWatch.Elapsed.TotalSeconds:F1} s)");
                if (evaluation is not null)
                {
                    string evalFile = evalOutput ?? Path.ChangeExtension(output!, null) + ".eval.jsonl";
                    long held = new DatasetRows(() => console.Track(evaluation, "evaluation")).WriteJsonLines(evalFile);
                    _out.WriteLine($"{held:N0} evaluation rows written to {evalFile}");
                }

                return 0;
            }
        }
    }

    /// <summary>A row on one line: long strings cut, long lists shortened.</summary>
    public static string Short(JsonNode? node)
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
}
