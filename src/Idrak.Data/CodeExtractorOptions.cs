// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Globalization;
using Idrak.Data.Abstractions;

namespace Idrak.Data;

/// <summary>
/// The settings of a code extractor run (plan 36: <c>idrak data code</c>, which turns a repository into pretraining rows
/// and proven task folders). Every one has an <c>IDRAK_CODE_*</c> variable (<see cref="FromEnvironment"/>), and every
/// default is the documented rule named beside it, never the only way. Nothing here names a language, a model family or a
/// provider: the languages come from the registered coding languages, the families from <see cref="TaskFamilies"/>.
/// </summary>
public sealed record CodeExtractorOptions
{
    /// <summary>The variable of <see cref="Families"/>: "IDRAK_CODE_FAMILIES" (names separated by commas, or "all").</summary>
    public const string FamiliesVariable = "IDRAK_CODE_FAMILIES";

    /// <summary>The variable of <see cref="SkippedFolders"/>: "IDRAK_CODE_SKIPPED_FOLDERS" (folder names separated by commas).</summary>
    public const string SkippedFoldersVariable = "IDRAK_CODE_SKIPPED_FOLDERS";

    /// <summary>The variable of <see cref="MaxFileBytes"/>: "IDRAK_CODE_MAX_FILE_BYTES" (bytes, above 0).</summary>
    public const string MaxFileBytesVariable = "IDRAK_CODE_MAX_FILE_BYTES";

    /// <summary>The variable of <see cref="DedupeThreshold"/>: "IDRAK_CODE_DEDUPE" (a similarity in (0, 1]).</summary>
    public const string DedupeVariable = "IDRAK_CODE_DEDUPE";

    /// <summary>The variable of <see cref="Seed"/>: "IDRAK_CODE_SEED" (a whole number).</summary>
    public const string SeedVariable = "IDRAK_CODE_SEED";

    /// <summary>The variable of <see cref="SplitBy"/>: "IDRAK_CODE_SPLIT_BY" (a group kind's name).</summary>
    public const string SplitByVariable = "IDRAK_CODE_SPLIT_BY";

    /// <summary>The variable of <see cref="EvalFraction"/>: "IDRAK_CODE_EVAL_FRACTION" (in [0, 1)).</summary>
    public const string EvalFractionVariable = "IDRAK_CODE_EVAL_FRACTION";

    /// <summary>The variable of <see cref="EvalGroups"/>: "IDRAK_CODE_EVAL_GROUPS" (group names separated by commas).</summary>
    public const string EvalGroupsVariable = "IDRAK_CODE_EVAL_GROUPS";

    /// <summary>The variable of <see cref="MaxTasks"/>: "IDRAK_CODE_MAX_TASKS" (a count above 0, or 0 for no limit).</summary>
    public const string MaxTasksVariable = "IDRAK_CODE_MAX_TASKS";

    /// <summary>The variable of <see cref="MaxTime"/>: "IDRAK_CODE_MAX_TIME" (seconds above 0, or 0 for no limit).</summary>
    public const string MaxTimeVariable = "IDRAK_CODE_MAX_TIME";

    /// <summary>The variable of <see cref="Formats"/>: "IDRAK_CODE_FORMATS" (data format names separated by commas).</summary>
    public const string FormatsVariable = "IDRAK_CODE_FORMATS";

    /// <summary>The variable of <see cref="Concurrency"/>: "IDRAK_CODE_CONCURRENCY" (a count above 0, or 0 for the CPU threads).</summary>
    public const string ConcurrencyVariable = "IDRAK_CODE_CONCURRENCY";

    /// <summary>The variable of <see cref="Verifier"/>: "IDRAK_CODE_VERIFIER" (a registered verifier's name, or "none").</summary>
    public const string VerifierVariable = "IDRAK_CODE_VERIFIER";

    /// <summary>
    /// The near-duplicate threshold unless set: 0.85, the similarity (Jaccard over shingles) above which code
    /// deduplication commonly counts two files as one; 1 keeps all but exact copies.
    /// </summary>
    public const double DefaultDedupeThreshold = 0.85;

    /// <summary>The share of groups held out for evaluation unless set: 0.1.</summary>
    public const double DefaultEvalFraction = 0.1;

    /// <summary>What a split's groups are unless set: "project" (the project a file or task belongs to).</summary>
    public const string DefaultSplitBy = "project";

    /// <summary>The task families to run, by name (<see cref="TaskFamilies"/>); null (the default): every registered one.</summary>
    public IReadOnlyList<string>? Families { get; init; }

    /// <summary>
    /// The folders never read, by name (ignoring case); null (the default): <see cref="ReadOptions.DefaultSkippedFolders"/>
    /// and the build and dependency folders of the registered coding languages, which the caller adds.
    /// </summary>
    public IReadOnlyList<string>? SkippedFolders { get; init; }

    /// <summary>The largest file read, in bytes (a larger one is often generated): the data reader's own rule by default (<see cref="ReadOptions.MaxDocumentBytes"/>, 1 MiB).</summary>
    public long MaxFileBytes { get; init; } = ReadOptions.Default.MaxDocumentBytes;

    /// <summary>The similarity in (0, 1] above which two rows are near duplicates and one is dropped (<see cref="DefaultDedupeThreshold"/>).</summary>
    public double DedupeThreshold { get; init; } = DefaultDedupeThreshold;

    /// <summary>The seed of the split's choice of groups (0 by default): the same seed gives the same split.</summary>
    public int Seed { get; init; }

    /// <summary>What the split's groups are (<see cref="DefaultSplitBy"/>): splits are by group, never by random row.</summary>
    public string SplitBy { get; init; } = DefaultSplitBy;

    /// <summary>The share of groups held out for evaluation, in [0, 1) (<see cref="DefaultEvalFraction"/>); ignored when <see cref="EvalGroups"/> names some.</summary>
    public double EvalFraction { get; init; } = DefaultEvalFraction;

    /// <summary>The groups held out for evaluation, by name; empty (the default): <see cref="EvalFraction"/> of them, chosen by <see cref="Seed"/>.</summary>
    public IReadOnlyList<string> EvalGroups { get; init; } = [];

    /// <summary>The most tasks a run keeps; null (the default): no limit.</summary>
    public int? MaxTasks { get; init; }

    /// <summary>The longest a run takes before it stops and writes what it has; null (the default): no limit.</summary>
    public TimeSpan? MaxTime { get; init; }

    /// <summary>The formats the rows are written in, by data format name (<see cref="DataFileFormats"/>): JSON Lines by default.</summary>
    public IReadOnlyList<string> Formats { get; init; } = [nameof(DataFormat.JsonLines)];

    /// <summary>How many files or tasks are worked on at once; 0 (the default): <see cref="ComputeResources.MaxCpuThreads"/> when the run starts.</summary>
    public int Concurrency { get; init; }

    /// <summary>The concurrency a run uses now: <see cref="Concurrency"/>, or the CPU threads <see cref="ComputeResources"/> allows.</summary>
    public int EffectiveConcurrency => Concurrency > 0 ? Concurrency : ComputeResources.MaxCpuThreads;

    /// <summary>
    /// The verifier that proves drafts (<see cref="TaskVerifiers"/>), by name; "none" writes every draft as unproven; null
    /// (the default): the first registered, or none when nobody registered one. A draft becomes a task only when proven.
    /// </summary>
    public string? Verifier { get; init; }

    /// <summary>
    /// <paramref name="defaults"/> (or the defaults) with what the <c>IDRAK_CODE_*</c> variables set; a variable that is
    /// unset or empty leaves its setting as it is.
    /// </summary>
    /// <exception cref="ArgumentException">A value that does not parse or is out of range: the message names the variable and what it takes.</exception>
    public static CodeExtractorOptions FromEnvironment(CodeExtractorOptions? defaults = null) => From(Environment.GetEnvironmentVariable, defaults);

    internal static CodeExtractorOptions From(Func<string, string?> variable, CodeExtractorOptions? defaults)
    {
        var options = defaults ?? new CodeExtractorOptions();
        if (Set(variable, FamiliesVariable) is { } families)
        {
            options = options with { Families = families.Equals("all", StringComparison.OrdinalIgnoreCase) ? null : List(families) };
        }

        if (Set(variable, SkippedFoldersVariable) is { } folders)
        {
            options = options with { SkippedFolders = List(folders) };
        }

        if (Set(variable, MaxFileBytesVariable) is { } bytes)
        {
            options = options with
            {
                MaxFileBytes = long.TryParse(bytes, NumberStyles.Integer, CultureInfo.InvariantCulture, out long b) && b > 0 ? b : throw Bad(MaxFileBytesVariable, bytes, "a count of bytes above 0"),
            };
        }

        if (Set(variable, DedupeVariable) is { } dedupe)
        {
            options = options with
            {
                DedupeThreshold = double.TryParse(dedupe, NumberStyles.Float, CultureInfo.InvariantCulture, out double d) && d > 0 && d <= 1 ? d : throw Bad(DedupeVariable, dedupe, "a similarity above 0 and at most 1"),
            };
        }

        if (Set(variable, SeedVariable) is { } seed)
        {
            options = options with { Seed = int.TryParse(seed, NumberStyles.Integer, CultureInfo.InvariantCulture, out int s) ? s : throw Bad(SeedVariable, seed, "a whole number") };
        }

        if (Set(variable, SplitByVariable) is { } splitBy)
        {
            options = options with { SplitBy = splitBy };
        }

        if (Set(variable, EvalFractionVariable) is { } fraction)
        {
            options = options with
            {
                EvalFraction = double.TryParse(fraction, NumberStyles.Float, CultureInfo.InvariantCulture, out double f) && f >= 0 && f < 1 ? f : throw Bad(EvalFractionVariable, fraction, "a share of at least 0 and below 1"),
            };
        }

        if (Set(variable, EvalGroupsVariable) is { } groups)
        {
            options = options with { EvalGroups = List(groups) };
        }

        if (Set(variable, MaxTasksVariable) is { } tasks)
        {
            options = options with
            {
                MaxTasks = int.TryParse(tasks, NumberStyles.Integer, CultureInfo.InvariantCulture, out int t) && t >= 0 ? (t == 0 ? null : t) : throw Bad(MaxTasksVariable, tasks, "a count above 0, or 0 for no limit"),
            };
        }

        if (Set(variable, MaxTimeVariable) is { } time)
        {
            options = options with
            {
                MaxTime = double.TryParse(time, NumberStyles.Float, CultureInfo.InvariantCulture, out double seconds) && seconds >= 0 && seconds <= TimeSpan.MaxValue.TotalSeconds
                    ? (seconds == 0 ? null : TimeSpan.FromSeconds(seconds))
                    : throw Bad(MaxTimeVariable, time, "seconds above 0, or 0 for no limit"),
            };
        }

        if (Set(variable, FormatsVariable) is { } formats)
        {
            var list = List(formats);
            options = options with { Formats = list.Count > 0 ? list : throw Bad(FormatsVariable, formats, "data format names separated by commas, such as JsonLines,Parquet") };
        }

        if (Set(variable, ConcurrencyVariable) is { } concurrency)
        {
            options = options with
            {
                Concurrency = int.TryParse(concurrency, NumberStyles.Integer, CultureInfo.InvariantCulture, out int c) && c >= 0 ? c : throw Bad(ConcurrencyVariable, concurrency, "a count above 0, or 0 for the CPU threads"),
            };
        }

        if (Set(variable, VerifierVariable) is { } verifier)
        {
            options = options with { Verifier = verifier };
        }

        return options;
    }

    // The variable's value, trimmed, or null when it is unset or empty.
    private static string? Set(Func<string, string?> variable, string name) => variable(name) is { } value && value.Trim() is { Length: > 0 } trimmed ? trimmed : null;

    private static IReadOnlyList<string> List(string text) => text.Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries);

    private static ArgumentException Bad(string variable, string value, string takes) => new($"{variable}='{value}': it takes {takes}.");
}
