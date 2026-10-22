// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Runtime.CompilerServices;
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

    /// <summary>One line on what it covers.</summary>
    string Summary { get; }

    /// <summary>Whether the files within <paramref name="depth"/> folders of <paramref name="workspace"/>'s top (0: the top only) show a project of the language.</summary>
    bool Detect(string workspace, int depth);

    /// <summary>The programs <c>run_command</c> may start for it (file names without extension, in order).</summary>
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
/// The coding languages by name (ignoring case). <b>The library registers none</b>: what stays in it is language-free (the
/// agent's loop, the tools, git's read-only subcommands, <c>.git</c> as an ignored folder, a prompt with no build or test
/// commands). The coding agent and its tools compose their defaults from the languages a run works with
/// (<see cref="Resolve"/>: every registered one, <see cref="All"/>, the default; those that detect the workspace,
/// <see cref="Detected"/>; or names), in their order: the prompt's line on checking the work, the programs allowed (git
/// last), the folders skipped, the tools' examples, the commands' variables. Idrak.Text.Plugins registers the library's
/// own (<c>TextPlugins.RegisterAll()</c>); a language that is not registered is refused with "not registered".
/// </summary>
public static class CodingLanguages
{
    /// <summary>Every registered language (the default choice).</summary>
    public const string All = "all";

    /// <summary>The registered languages that detect the workspace.</summary>
    public const string Detected = "detected";

    /// <summary>No language: the language-free defaults alone.</summary>
    public const string None = "none";

    /// <summary>The variable that chooses the languages when an option does not: "IDRAK_CODING_LANGUAGES" (all, detected, none, or names separated by commas).</summary>
    public const string EnvironmentVariable = "IDRAK_CODING_LANGUAGES";

    /// <summary>How many folders below the workspace's top detection looks (0: the top only), unless an option says otherwise: 2.</summary>
    public const int DefaultDepth = 2;

    /// <summary>The package whose plug-ins register the library's own languages, as the "not registered" messages name it.</summary>
    public const string PluginPackage = "Idrak.Text.Plugins";

    private static readonly SlotTable<string, ICodingLanguage> Table = new(nameof(CodingLanguages), comparer: StringComparer.OrdinalIgnoreCase,
        unguarded: "a language's lists are read once, when the tools are made, and composed into their defaults: there is no call a guard could fall back on");

    /// <summary>The choice <see cref="EnvironmentVariable"/> names, or <see cref="All"/> when it is not set.</summary>
    public static string DefaultChoice => System.Environment.GetEnvironmentVariable(EnvironmentVariable) is { } choice && choice.Trim().Length > 0 ? choice.Trim() : All;

    /// <summary>Registers <paramref name="language"/> under its <see cref="ICodingLanguage.Name"/> (one of the same name takes its place).</summary>
    [MethodImpl(MethodImplOptions.NoInlining)]   // the caller is the registering assembly (its Origin)
    public static void Register(ICodingLanguage language)
    {
        ArgumentNullException.ThrowIfNull(language);
        ArgumentException.ThrowIfNullOrWhiteSpace(language.Name);
        if (language.Name.Contains(',', StringComparison.Ordinal) || language.Name.Equals(All, StringComparison.OrdinalIgnoreCase)
            || language.Name.Equals(Detected, StringComparison.OrdinalIgnoreCase) || language.Name.Equals(None, StringComparison.OrdinalIgnoreCase))
        {
            throw new ArgumentException($"'{language.Name}' cannot name a coding language: it holds a comma or is one of {All}, {Detected}, {None}.", nameof(language));
        }

        Table.Register(language.Name, language, System.Reflection.Assembly.GetCallingAssembly());
    }

    /// <summary>Removes the language <paramref name="name"/>; false when none is registered under it.</summary>
    public static bool Unregister(string name) => Table.Unregister(name);

    /// <summary>The registered names, in the order they were registered.</summary>
    public static IReadOnlyCollection<string> Names => Table.Keys;

    /// <summary>The registered languages, in the order they were registered.</summary>
    public static IReadOnlyList<ICodingLanguage> Registered => [.. Table.Values];

    /// <summary>The language <paramref name="name"/>, or null.</summary>
    public static ICodingLanguage? Find(string name) => Table.Find(name);

    /// <summary>The language <paramref name="name"/>.</summary>
    /// <exception cref="NotSupportedException">None is: the message names the registered ones and the package with the library's own.</exception>
    public static ICodingLanguage Get(string name) => Table.Find(name)
        ?? throw new NotSupportedException($"Coding language '{name}' is not registered (registered: {(Table.Keys.Count == 0 ? "none" : string.Join(", ", Table.Keys))}); "
                                           + $"register its plug-in ({PluginPackage}: TextPlugins.RegisterAll() registers the library's own, each as '<name>-language') "
                                           + "or your own with CodingLanguages.Register.");

    /// <summary>Who registered the language <paramref name="name"/>: the app's assembly; null when none is.</summary>
    public static string? Origin(string name) => Table.Origin(name);

    /// <summary>Each registered language as "name: summary", one per line.</summary>
    public static string Describe() => string.Join("\n", Table.Values.Select(l => $"{l.Name}: {l.Summary}"));

    /// <summary>
    /// The languages <paramref name="choice"/> names: <see cref="All"/> (every registered one, in their order),
    /// <see cref="Detected"/> (those that detect <paramref name="workspace"/>), <see cref="None"/>, or names separated by
    /// commas (in that order); null or empty for <see cref="DefaultChoice"/>.
    /// </summary>
    /// <exception cref="NotSupportedException">A name is not registered.</exception>
    /// <exception cref="ArgumentException"><see cref="Detected"/> without a workspace.</exception>
    public static IReadOnlyList<ICodingLanguage> Resolve(string? choice, string? workspace = null, int depth = DefaultDepth)
    {
        string text = string.IsNullOrWhiteSpace(choice) ? DefaultChoice : choice.Trim();
        if (text.Equals(All, StringComparison.OrdinalIgnoreCase))
        {
            return Registered;
        }

        if (text.Equals(None, StringComparison.OrdinalIgnoreCase))
        {
            return [];
        }

        if (text.Equals(Detected, StringComparison.OrdinalIgnoreCase))
        {
            return workspace is null ? throw new ArgumentException($"The coding languages '{Detected}' are found in a workspace; none was given.", nameof(workspace))
                : Detect(workspace, depth);
        }

        var languages = new List<ICodingLanguage>();
        foreach (var name in text.Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries))
        {
            var language = Get(name);
            if (!languages.Contains(language))
            {
                languages.Add(language);
            }
        }

        return languages;
    }

    /// <summary>The registered languages that detect <paramref name="workspace"/> (<see cref="ICodingLanguage.Detect"/>), in their order.</summary>
    public static IReadOnlyList<ICodingLanguage> Detect(string workspace, int depth = DefaultDepth)
    {
        ArgumentNullException.ThrowIfNull(workspace);
        return [.. Table.Values.Where(l => l.Detect(workspace, depth))];
    }

    /// <summary>
    /// A notice for a workspace none of <paramref name="languages"/> detects although it holds source files (named by
    /// Idrak.Data's table of file extensions): which languages its files are in, and that a coding-language plug-in that
    /// provides one is to be registered (the agent gets no build or test commands for them; a task's own commands still
    /// run). Null when one of them detects it, or it holds no source file within <paramref name="depth"/>.
    /// </summary>
    public static string? Notice(string workspace, IReadOnlyList<ICodingLanguage> languages, int depth = DefaultDepth)
    {
        ArgumentNullException.ThrowIfNull(workspace);
        ArgumentNullException.ThrowIfNull(languages);
        if (languages.Any(l => l.Detect(workspace, depth)))
        {
            return null;
        }

        var counts = new Dictionary<string, int>(StringComparer.Ordinal);
        var skipped = Table.Values.SelectMany(l => l.IgnoredDirectories.Concat(l.LinkedDirectories)).ToList();
        FindFile(workspace, depth, name =>
        {
            if (DataFiles.LanguageOf(name) is { } language)
            {
                counts[language] = counts.GetValueOrDefault(language) + 1;
            }

            return false;
        }, skipped);
        if (counts.Count == 0)
        {
            return null;
        }

        var found = counts.OrderByDescending(p => p.Value).ThenBy(p => p.Key, StringComparer.Ordinal).Take(3).Select(p => p.Key).ToList();
        string chosen = languages.Count == 0 ? "no coding language" : $"none of the coding languages {string.Join(", ", languages.Select(l => l.Name))}";
        return $"Note: {chosen} detects this workspace, whose files are {string.Join(", ", found)}: the agent gets no build or test commands for them. "
               + $"Register a coding-language plug-in that provides '{found[0]}' ({PluginPackage}: TextPlugins.RegisterAll() registers the library's own, each as "
               + $"'<name>-language'), or choose the languages ({EnvironmentVariable}: {All}, {Detected} or names).";
    }

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
