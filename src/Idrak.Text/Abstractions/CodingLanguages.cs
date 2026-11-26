// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Idrak.Data;

namespace Idrak.Text.Abstractions;

/// <summary>
/// What a coding language adds to the coding tools' descriptions (<c>CodingTools</c> composes them from every language it
/// works with, in order): the folders <c>list_files</c> names as skipped, example globs and patterns, example commands.
/// </summary>
public sealed record CodingToolExamples
{
    /// <summary>No examples.</summary>
    public static CodingToolExamples None { get; } = new();

    /// <summary>Folders named in <c>list_files</c>' description as skipped (build output, dependencies), e.g. <c>bin</c>.</summary>
    public IReadOnlyList<string> Folders { get; init; } = [];

    /// <summary>File-name globs in <c>list_files</c>' <c>pattern</c>, e.g. <c>*.cs</c>.</summary>
    public IReadOnlyList<string> FileGlobs { get; init; } = [];

    /// <summary>Regular expressions in <c>search</c>'s <c>pattern</c>.</summary>
    public IReadOnlyList<string> SearchPatterns { get; init; } = [];

    /// <summary>File-name globs in <c>search</c>'s <c>glob</c>.</summary>
    public IReadOnlyList<string> SearchGlobs { get; init; } = [];

    /// <summary>Commands in <c>run_command</c>'s description (how to build, test, lint).</summary>
    public IReadOnlyList<string> Commands { get; init; } = [];

    /// <summary>Command lines in <c>run_command</c>'s <c>command</c>.</summary>
    public IReadOnlyList<string> CommandLines { get; init; } = [];
}

/// <summary>
/// A programming language the coding agent and its tools know how to work in (<see cref="CodingLanguages"/>): how to
/// recognize a project of it, the programs its builds and tests start, what the agent is told about checking its work,
/// the folders that are build output or dependencies, and the commands a coding task of it runs when it names none. The
/// library registers no language; Idrak.Text.Plugins has the library's own profiles, and an
/// app registers its own.
/// </summary>
public interface ICodingLanguage
{
    /// <summary>Its name in <see cref="CodingLanguages"/> (ignoring case), e.g. "csharp".</summary>
    string Name { get; }

    /// <summary>Whether the files within <paramref name="depth"/> folders of <paramref name="workspace"/>'s top (0: the top only) show a project of the language.</summary>
    bool Detect(string workspace, int depth);

    /// <summary>
    /// The programs <c>run_command</c> may start for it by default (file names without extension, in order): its build and
    /// test runners only, never a program that runs any code it is given (<c>node</c>, <c>npx</c>), which the coding
    /// tools' options name when an app wants it.
    /// </summary>
    IReadOnlyList<string> Commands { get; }

    /// <summary>How to build, check and test it, as "Name: commands" (the agent's prompt lists them in its line on checking the work).</summary>
    IReadOnlyList<string> PromptLines { get; }

    /// <summary>Its examples in the tools' descriptions.</summary>
    CodingToolExamples ToolExamples { get; }

    /// <summary>Folders never listed, searched or copied into a coding task's workspace (build output, dependencies, tool state).</summary>
    IReadOnlyList<string> IgnoredDirectories { get; }

    /// <summary>Dependency folders linked into a coding task's workspace instead of copied.</summary>
    IReadOnlyList<string> LinkedDirectories { get; }

    /// <summary>Lines of a coding workspace's git excludes (so <c>git diff</c> shows the agent's changes, not build output).</summary>
    IReadOnlyList<string> GitExcludes { get; }

    /// <summary>Variables its programs get so they run without questions, colours or telemetry prompts.</summary>
    IReadOnlyDictionary<string, string> Environment { get; }

    /// <summary>Setup commands of a coding task of the language that names none (a restore, an install).</summary>
    IReadOnlyList<string> DefaultSetup { get; }

    /// <summary>Verification commands of a coding task of the language that names none (a build, then the tests).</summary>
    IReadOnlyList<string> DefaultVerify { get; }
}

/// <summary>
/// A coding language as data (<see cref="ICodingLanguage"/>): its lists as given, and <see cref="Detect"/> finding one of
/// its <see cref="Markers"/> within the depth asked.
/// </summary>
/// <param name="Name">Its name in <see cref="CodingLanguages"/>.</param>
/// <param name="Summary">One line on what it covers.</param>
public sealed record CodingLanguage(string Name, string Summary) : ICodingLanguage
{
    /// <summary>File names (or <c>*.extension</c>, ignoring case) whose presence shows a project of the language.</summary>
    public IReadOnlyList<string> Markers { get; init; } = [];

    /// <inheritdoc />
    public IReadOnlyList<string> Commands { get; init; } = [];

    /// <inheritdoc />
    public IReadOnlyList<string> PromptLines { get; init; } = [];

    /// <inheritdoc />
    public CodingToolExamples ToolExamples { get; init; } = CodingToolExamples.None;

    /// <inheritdoc />
    public IReadOnlyList<string> IgnoredDirectories { get; init; } = [];

    /// <inheritdoc />
    public IReadOnlyList<string> LinkedDirectories { get; init; } = [];

    /// <inheritdoc />
    public IReadOnlyList<string> GitExcludes { get; init; } = [];

    /// <inheritdoc />
    public IReadOnlyDictionary<string, string> Environment { get; init; } = new Dictionary<string, string>();

    /// <inheritdoc />
    public IReadOnlyList<string> DefaultSetup { get; init; } = [];

    /// <inheritdoc />
    public IReadOnlyList<string> DefaultVerify { get; init; } = [];

    /// <inheritdoc />
    public bool Detect(string workspace, int depth) =>
        Markers.Count > 0 && CodingLanguages.FindFile(workspace, depth, name => Markers.Any(m => CodingLanguages.NameMatches(name, m)), [.. IgnoredDirectories, .. LinkedDirectories]) is not null;
}

/// <summary>
/// The coding languages' choices and the helpers that compose them. <b>The library holds no language</b>: what stays in
/// it is language-free (the agent's loop, the tools, git's read-only subcommands, <c>.git</c> as an ignored folder, a
/// prompt with no build or test commands). The coding agent and its tools compose their defaults from the languages a run
/// works with (every one, <see cref="All"/>, the default; those that detect the workspace, <see cref="Detected"/>; or
/// names), in their order: the prompt's line on checking the work, the programs allowed (git last), the folders skipped,
/// the tools' examples, the commands' variables. The languages live in Idrak.Text.Plugins.
/// </summary>
public static class CodingLanguages
{
    /// <summary>Every language (the default choice).</summary>
    public const string All = "all";

    /// <summary>The languages that detect the workspace.</summary>
    public const string Detected = "detected";

    /// <summary>No language: the language-free defaults alone.</summary>
    public const string None = "none";

    /// <summary>The variable that chooses the languages when an option does not: "IDRAK_CODING_LANGUAGES" (all, detected, none, or names separated by commas).</summary>
    public const string EnvironmentVariable = "IDRAK_CODING_LANGUAGES";

    /// <summary>How many folders below the workspace's top detection looks (0: the top only), unless an option says otherwise: 2.</summary>
    public const int DefaultDepth = 2;

    /// <summary>The package whose plug-ins register the library's own languages, as the "not registered" messages name it.</summary>
    public const string PluginPackage = "Idrak.Text.Plugins";

    /// <summary>The choice <see cref="EnvironmentVariable"/> names, or <see cref="All"/> when it is not set.</summary>
    public static string DefaultChoice => System.Environment.GetEnvironmentVariable(EnvironmentVariable) is { } choice && choice.Trim().Length > 0 ? choice.Trim() : All;

    /// <summary>
    /// The lists joined in order, each item once (by <paramref name="comparer"/>, ordinal by default), every list's own
    /// order kept: an item new to the result goes right after the one before it in its own list, or at the end.
    /// </summary>
    public static IReadOnlyList<string> Merge(IEnumerable<IReadOnlyList<string>> lists, IEqualityComparer<string>? comparer = null)
    {
        ArgumentNullException.ThrowIfNull(lists);
        comparer ??= StringComparer.Ordinal;
        var merged = new List<string>();
        foreach (var list in lists)
        {
            int after = -1;
            foreach (var item in list)
            {
                int at = merged.FindIndex(m => comparer.Equals(m, item));
                if (at >= 0)
                {
                    after = at;
                    continue;
                }

                after = after < 0 ? merged.Count : after + 1;
                merged.Insert(after, item);
            }
        }

        return merged;
    }

    /// <summary>Whether a file's name matches <paramref name="pattern"/>: the same name, or <c>*.extension</c> (its end), ignoring case.</summary>
    public static bool NameMatches(string name, string pattern) =>
        pattern.StartsWith('*') ? name.EndsWith(pattern.AsSpan(1), StringComparison.OrdinalIgnoreCase) : name.Equals(pattern, StringComparison.OrdinalIgnoreCase);

    /// <summary>
    /// The first file within <paramref name="depth"/> folders of <paramref name="workspace"/>'s top (breadth first, names
    /// in order) whose name <paramref name="match"/> accepts, skipping <c>.git</c> and <paramref name="skipped"/>; null for none.
    /// </summary>
    public static string? FindFile(string workspace, int depth, Func<string, bool> match, IReadOnlyCollection<string>? skipped = null)
    {
        ArgumentNullException.ThrowIfNull(workspace);
        ArgumentNullException.ThrowIfNull(match);
        var skip = new HashSet<string>(skipped ?? [], StringComparer.OrdinalIgnoreCase) { ".git" };
        var level = new List<string> { workspace };
        for (int d = 0; d <= depth && level.Count > 0; d++)
        {
            var next = new List<string>();
            foreach (var folder in level)
            {
                string[] files, folders;
                try
                {
                    files = Directory.GetFiles(folder);
                    folders = d < depth ? Directory.GetDirectories(folder) : [];
                }
                catch (Exception e) when (e is IOException or UnauthorizedAccessException)
                {
                    continue;
                }

                Array.Sort(files, StringComparer.Ordinal);
                foreach (var file in files)
                {
                    if (match(Path.GetFileName(file)))
                    {
                        return file;
                    }
                }

                Array.Sort(folders, StringComparer.Ordinal);
                next.AddRange(folders.Where(f => !skip.Contains(Path.GetFileName(f))));
            }

            level = next;
        }

        return null;
    }
}
