// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Globalization;
using Idrak.Cli.Shared;
using Idrak.Data;
using Idrak.Data.Abstractions;
using Idrak.Data.Code;
using Idrak.Text.Abstractions;

namespace Idrak.Cli.Commands.Data;

/// <summary>
/// <c>idrak data code REPO -o OUT</c> (plan 36, G1): a repository or project turned into pretraining rows and proven task
/// folders (<see cref="CodeExtractor"/>) by the registered task families, with a manifest of what was kept and rejected. No
/// model is used. Every <see cref="CodeExtractorOptions"/> setting is a flag over its <c>IDRAK_CODE_*</c> variable.
/// </summary>
internal sealed class DataCodeCommand : Command
{
    public override string Name => "data code";

    public override string Summary => "Turn a code repository into pretraining rows and proven coding tasks (task families, a manifest)";

    public override string Usage => """
        REPO -o OUT [options]

        Walks REPO (a repository or a project folder) once and runs the task families on it: each writes rows (pretraining
        text) and drafts (coding tasks: a workspace, a check, a solution). A draft becomes a task folder only when the
        verifier proves it: its check fails on the workspace and passes with the solution. With no verifier (none
        registered, or --verifier none) the drafts are written under drafts/, marked unproven. OUT gets rows/, tasks/,
        drafts/, rejected/ (each rejected draft's proof) and manifest.json (the families, the counts asked, kept and
        rejected with the reasons, the time). Each option is over its IDRAK_CODE_* variable.

        Options:
          -o, --out DIR             the output folder (under test-output, for example)
              --families A,B        the task families to run, by name, or all (IDRAK_CODE_FAMILIES; all registered)
              --skip A,B            folders never read, by name (IDRAK_CODE_SKIPPED_FOLDERS; .git, .vs, .idea, the coding
                                    languages' build folders and the folders the repository's .gitignore names)
              --max-file-bytes N    the largest file read (IDRAK_CODE_MAX_FILE_BYTES; 1048576)
              --dedupe X            the near-duplicate similarity the row filters drop above (IDRAK_CODE_DEDUPE; 0.85)
              --seed N              the split's seed (IDRAK_CODE_SEED; 0)
              --split-by KIND       what the split groups by (IDRAK_CODE_SPLIT_BY; project)
              --eval-fraction F     the share of groups held out (IDRAK_CODE_EVAL_FRACTION; 0.1)
              --eval-groups A,B     the groups held out, by name (IDRAK_CODE_EVAL_GROUPS)
              --max-tasks N         the most tasks (and unproven drafts) kept; 0 for no limit (IDRAK_CODE_MAX_TASKS)
              --max-time SECONDS    the longest the run takes before it writes what it has; 0 for none (IDRAK_CODE_MAX_TIME)
              --formats A,B         the row formats (IDRAK_CODE_FORMATS; JsonLines, the one Idrak.Data writes)
              --concurrency N       drafts proven at once; 0 for the CPU threads (IDRAK_CODE_CONCURRENCY)
              --verifier NAME       the verifier that proves drafts, or none (IDRAK_CODE_VERIFIER; the first registered)
              --list                the registered task families and verifiers (nothing is run)

        The split settings are recorded in the manifest; the group splits and the leakage check come with plan 36's G2.

        Examples:
          idrak data code . -o test-output/code-data --max-tasks 50
          idrak data code ../my-app -o test-output/my-app-data --families file-text,method-body
          idrak data code . -o test-output/code-drafts --families test-subject --verifier none
        """;

    public override IReadOnlyCollection<string> ValueOptions { get; } =
    [
        "--out", "--families", "--skip", "--max-file-bytes", "--dedupe", "--split-by", "--eval-fraction", "--eval-groups", "--max-tasks", "--max-time",
        "--formats", "--concurrency", "--verifier",
    ];

    public override IReadOnlyCollection<string> Flags { get; } = ["--list"];

    public override IReadOnlyDictionary<string, string> ShortForms { get; } = new Dictionary<string, string> { ["-o"] = "--out" };

    public override int Run(CommandContext context)
    {
        if (context.Flag("--list"))
        {
            context.Write("task families:");
            foreach (string line in TaskFamilies.Describe().Split('\n', StringSplitOptions.RemoveEmptyEntries))
            {
                context.Write("  " + line);
            }

            context.Write(TaskVerifiers.Names.Count == 0 ? "verifiers: none registered (drafts are written unproven)" : "verifiers:");
            foreach (string line in TaskVerifiers.Describe().Split('\n', StringSplitOptions.RemoveEmptyEntries))
            {
                context.Write("  " + line);
            }

            return ExitCodes.Ok;
        }

        if (context.Positional is not [var repository])
        {
            throw new UsageException("Give one repository or project folder: idrak data code REPO -o OUT.");
        }

        string output = context.Option("--out") ?? throw new UsageException("Give -o, --out DIR for the rows, tasks and manifest (under test-output, for example).");
        if (!Directory.Exists(repository))
        {
            throw new UsageException($"'{repository}' is not a folder.");
        }

        string root = Path.GetFullPath(repository);
        CodeExtractorOptions options;
        try
        {
            options = Read(context, CodeExtractorOptions.FromEnvironment());
        }
        catch (ArgumentException e)
        {
            throw new UsageException(e.Message);
        }

        var skipped = options.SkippedFolders ?? DefaultSkipped(root);
        var source = new CodeSource(root) { ReadOptions = ReadOptions.Default with { SkippedFolders = [.. skipped], MaxDocumentBytes = options.MaxFileBytes } };
        options = options with { SkippedFolders = skipped };
        CodeExtraction result;
        using (var line = new ProgressLine(context, $"data code {root}"))
        {
            var extractor = new CodeExtractor(options) { Progress = line };
            result = extractor.RunAsync(source, output, context.TimeoutToken).GetAwaiter().GetResult();
            line.Finish();
        }

        context.Write(string.Create(CultureInfo.InvariantCulture,
            $"{result.Rows:N0} rows, {result.Drafts:N0} drafts: {result.Tasks:N0} proven tasks, {result.Unproven:N0} unproven, {result.Rejected:N0} rejected{(result.Stopped is { } why ? $" (stopped: {why})" : "")}"));
        context.Write($"manifest: {result.ManifestPath}");
        if (result.Unproven > 0)
        {
            context.Write($"note: {result.Unproven:N0} drafts are unproven (no verifier): they are under drafts/, never tasks/");
        }

        context.WriteJson(result.Manifest);
        return ExitCodes.Ok;
    }

    // The flags over the variables' settings.
    private static CodeExtractorOptions Read(CommandContext context, CodeExtractorOptions options)
    {
        static IReadOnlyList<string> List(string text) => text.Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries);
        if (context.Option("--families") is { } families)
        {
            options = options with { Families = families.Equals("all", StringComparison.OrdinalIgnoreCase) ? null : List(families) };
        }

        if (context.Option("--skip") is { } skip)
        {
            options = options with { SkippedFolders = List(skip) };
        }

        if (context.Option("--max-file-bytes") is { } bytes)
        {
            options = options with { MaxFileBytes = long.TryParse(bytes, NumberStyles.Integer, CultureInfo.InvariantCulture, out long b) && b > 0 ? b : throw new UsageException($"--max-file-bytes is a count of bytes above 0, not '{bytes}'.") };
        }

        if (context.Option("--dedupe") is { } dedupe)
        {
            options = options with { DedupeThreshold = double.TryParse(dedupe, NumberStyles.Float, CultureInfo.InvariantCulture, out double d) && d > 0 && d <= 1 ? d : throw new UsageException($"--dedupe is a similarity above 0 and at most 1, not '{dedupe}'.") };
        }

        if (context.Seed is { } seed)
        {
            options = options with { Seed = seed };
        }

        if (context.Option("--split-by") is { } splitBy)
        {
            options = options with { SplitBy = splitBy };
        }

        if (context.Option("--eval-fraction") is { } fraction)
        {
            options = options with { EvalFraction = double.TryParse(fraction, NumberStyles.Float, CultureInfo.InvariantCulture, out double f) && f >= 0 && f < 1 ? f : throw new UsageException($"--eval-fraction is a share of at least 0 and below 1, not '{fraction}'.") };
        }

        if (context.Option("--eval-groups") is { } groups)
        {
            options = options with { EvalGroups = List(groups) };
        }

        if (context.Option("--max-tasks") is { } tasks)
        {
            options = options with { MaxTasks = int.TryParse(tasks, NumberStyles.Integer, CultureInfo.InvariantCulture, out int t) && t >= 0 ? (t == 0 ? null : t) : throw new UsageException($"--max-tasks is a count above 0, or 0 for no limit, not '{tasks}'.") };
        }

        if (context.Option("--max-time") is { } time)
        {
            options = options with
            {
                MaxTime = double.TryParse(time, NumberStyles.Float, CultureInfo.InvariantCulture, out double s) && s >= 0 ? (s == 0 ? null : TimeSpan.FromSeconds(s)) : throw new UsageException($"--max-time is seconds above 0, or 0 for no limit, not '{time}'."),
            };
        }

        if (context.Option("--formats") is { } formats)
        {
            options = options with { Formats = List(formats) is { Count: > 0 } list ? list : throw new UsageException($"--formats takes data format names, such as JsonLines, not '{formats}'.") };
        }

        if (context.Option("--concurrency") is { } concurrency)
        {
            options = options with { Concurrency = int.TryParse(concurrency, NumberStyles.Integer, CultureInfo.InvariantCulture, out int c) && c >= 0 ? c : throw new UsageException($"--concurrency is a count above 0, or 0 for the CPU threads, not '{concurrency}'.") };
        }

        if (context.Option("--verifier") is { } verifier)
        {
            options = options with { Verifier = verifier };
        }

        return options;
    }

    // The folders skipped unless named: the reader's own, the registered coding languages' build and dependency folders, and
    // the plain folder names the repository's .gitignore lists (test-output/, bin/ …: what the repository keeps out of itself).
    internal static IReadOnlyList<string> DefaultSkipped(string root)
    {
        var folders = new List<string>(ReadOptions.DefaultSkippedFolders);
        foreach (var language in CodingLanguages.Registered)
        {
            folders.AddRange(language.IgnoredDirectories);
            folders.AddRange(language.LinkedDirectories);
        }

        string ignore = Path.Combine(root, ".gitignore");
        if (File.Exists(ignore))
        {
            foreach (string raw in File.ReadLines(ignore))
            {
                string entry = raw.Trim().TrimStart('/').TrimEnd('/');
                if (entry.Length > 0 && !entry.StartsWith('#') && !entry.StartsWith('!') && entry.IndexOfAny(['*', '?', '[', '/', '\\', '.']) < 0)
                {
                    folders.Add(entry);
                }
            }
        }

        return [.. folders.Distinct(StringComparer.OrdinalIgnoreCase)];
    }
}
