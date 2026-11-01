// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Text;
using System.Text.Json.Nodes;
using System.Text.RegularExpressions;
using Idrak.Text.Abstractions;

namespace Idrak.Text.Generation;

/// <summary>Limits and permissions of <see cref="CodingTools"/>.</summary>
public sealed record CodingToolOptions
{
    /// <summary>
    /// The folders skipped whatever the languages (no language's: <c>.git</c>, the JetBrains editors' <c>.idea</c>, and the
    /// usual names of coverage reports and build output, <c>coverage</c> and <c>out</c>).
    /// </summary>
    public static IReadOnlyList<string> LanguageFreeIgnoredDirectories { get; } = [".git", ".idea", "coverage", "out"];

    /// <summary>
    /// The coding languages (<see cref="CodingLanguages"/>) the tools work with, whose programs, folders, examples and
    /// variables compose the defaults: <see cref="CodingLanguages.All"/> (every registered one), <see cref="CodingLanguages.Detected"/>
    /// (those that detect the workspace), <see cref="CodingLanguages.None"/>, or names separated by commas; by default
    /// <see cref="CodingLanguages.EnvironmentVariable"/> (IDRAK_CODING_LANGUAGES), else all.
    /// </summary>
    public string Languages { get; init; } = CodingLanguages.DefaultChoice;

    /// <summary>How many folders below the workspace's top the languages are detected in (0: the top only; 2 by default).</summary>
    public int DetectionDepth { get; init; } = CodingLanguages.DefaultDepth;

    /// <summary>
    /// Programs <c>run_command</c> may start (file names without extension; resolved on the PATH); null (the default) for
    /// the languages' programs, then <c>git</c>.
    /// </summary>
    public IReadOnlyList<string>? Commands { get; init; }

    /// <summary>The only <c>git</c> subcommands allowed (read-only by default: the agent edits files, not history).</summary>
    public IReadOnlyList<string> GitSubcommands { get; init; } = ["status", "diff", "log", "show", "ls-files", "grep", "blame"];

    /// <summary>Longest a command may run (the model may ask for less).</summary>
    public TimeSpan CommandTimeout { get; init; } = TimeSpan.FromMinutes(5);

    /// <summary>Characters of command output returned (the start and the end are kept, the middle is cut).</summary>
    public int MaxOutputCharacters { get; init; } = 12_000;

    /// <summary>Lines <c>read_file</c> returns per call.</summary>
    public int MaxReadLines { get; init; } = 400;

    /// <summary>
    /// When edit_file's old_text is not in the file exactly, it is looked for again, in order: with read_file's line
    /// numbers ("12| ") taken off every line (only when every non-empty line has one, numbered in sequence; new_text's
    /// taken off too when all its lines have them), then line by line with trailing white space and a missing or extra
    /// final line break ignored. Only a single match is taken, and the result says which way it matched; the exact match
    /// is always tried first. True by default; false: the exact text only.
    /// </summary>
    public bool EditTolerance { get; init; } = true;

    /// <summary>
    /// Lines of the file an edit_file error quotes from its start when old_text is not found (beside old_text itself, cut
    /// at 200 characters), so the model sees the difference and can copy the text: 10 by default; 0 quotes none.
    /// </summary>
    public int EditQuotedLines { get; init; } = 10;

    /// <summary>Matches <c>search</c> returns.</summary>
    public int MaxSearchResults { get; init; } = 100;

    /// <summary>Paths <c>list_files</c> returns.</summary>
    public int MaxListedFiles { get; init; } = 400;

    /// <summary>
    /// Directories never listed or searched (build output, dependencies, tool state); null (the default) for
    /// <see cref="LanguageFreeIgnoredDirectories"/> and the languages' ignored and linked folders.
    /// </summary>
    public IReadOnlyList<string>? IgnoredDirectories { get; init; }

    /// <summary>Only the reading tools (no edits, writes or commands).</summary>
    public bool ReadOnly { get; init; }

    /// <summary>
    /// edit_file refuses a file the task has not read (read_file) or written (write_file) since the tools were made or
    /// <see cref="CodingTools.ForgetReads"/> was called (the agent calls it as each task starts), saying to read it first:
    /// an edit replaces exact text, and a model that guesses a file's text writes edits that do not match. True by default.
    /// </summary>
    public bool ReadBeforeEdit { get; init; } = true;

    /// <summary>
    /// The <see cref="CommandSandboxes"/> sandbox every command runs in: <see cref="CommandSandboxes.Process"/> (a plain
    /// process) unless <see cref="CommandSandboxes.EnvironmentVariable"/> (IDRAK_SANDBOX) names another, such as
    /// <see cref="CommandSandboxes.Container"/> (Idrak.Reinforcement.Plugins).
    /// </summary>
    public string Sandbox { get; init; } = CommandSandboxes.DefaultName;

    /// <summary>The most memory a command may use, in bytes; 0 (the default) for no limit.</summary>
    public long MaxMemoryBytes { get; init; }

    /// <summary>The most processes a command may start; 0 (the default) for no limit (a sandbox that cannot limit them refuses one).</summary>
    public int MaxProcesses { get; init; }

    /// <summary>The most CPUs a command may use; 0 (the default) for no limit (a sandbox that cannot limit them refuses one).</summary>
    public double MaxCpus { get; init; }

    /// <summary>Whether commands may reach the network (true by default; false needs a sandbox that can turn it off, such as container).</summary>
    public bool Network { get; init; } = true;

    /// <summary>Variables set for every command, over the tools' own non-interactive ones; null for none.</summary>
    public IReadOnlyDictionary<string, string>? Environment { get; init; }
}

/// <summary>
/// Tools for a coding agent working in one folder (the workspace): <c>list_files</c>, <c>read_file</c>, <c>search</c>,
/// <c>edit_file</c> (exact search and replace), <c>write_file</c> and <c>run_command</c> (allowlisted programs, started
/// without a shell, in the command sandbox the options name: <see cref="CodingToolOptions.Sandbox"/>). The programs
/// allowed, the folders skipped, the commands' variables and the examples in the tools' descriptions come from the coding
/// languages the options choose (<see cref="CodingToolOptions.Languages"/>, <see cref="CodingLanguages"/>); with none,
/// only the language-free ones (git, <c>.git</c>). Paths are relative to the workspace and cannot leave it; results are
/// plain text sized for a model's context; mistakes come back as messages the model can act on.
/// </summary>
public sealed partial class CodingTools
{
    // Non-interactive and uncoloured, whatever the language (each language adds its programs' own).
    private static readonly Dictionary<string, string> Quiet = new(StringComparer.Ordinal) { ["CI"] = "1", ["NO_COLOR"] = "1", ["FORCE_COLOR"] = "0" };

    private readonly string _root;
    private readonly CodingToolOptions _options;
    private readonly CommandLimits _limits;
    private readonly HashSet<string> _ignored;
    private readonly Lazy<string?> _notice;

    /// <summary>Tools over <paramref name="workspace"/>.</summary>
    /// <param name="workspace">The folder the tools work in.</param>
    /// <param name="options">What they may do; null for the defaults.</param>
    /// <param name="sandbox">The sandbox commands run in; null for the one <see cref="CodingToolOptions.Sandbox"/> names in <see cref="CommandSandboxes"/>.</param>
    /// <exception cref="DirectoryNotFoundException">The workspace does not exist.</exception>
    /// <exception cref="NotSupportedException">The sandbox is not registered, or cannot enforce the options' limits (the message names one that can).</exception>
    public CodingTools(string workspace, CodingToolOptions? options = null, ICommandSandbox? sandbox = null)
    {
        _root = Path.GetFullPath(workspace).TrimEnd(Path.DirectorySeparatorChar, Path.AltDirectorySeparatorChar);
        if (!Directory.Exists(_root))
        {
            throw new DirectoryNotFoundException($"Workspace '{_root}' does not exist.");
        }

        _options = options ?? new CodingToolOptions();
        Languages = CodingLanguages.Resolve(_options.Languages, _root, _options.DetectionDepth);
        var languages = Languages;
        Commands = _options.Commands ?? CodingLanguages.Merge([.. languages.Select(l => l.Commands), ["git"]], StringComparer.OrdinalIgnoreCase);
        IgnoredDirectories = _options.IgnoredDirectories ?? CodingLanguages.Merge(
            [CodingToolOptions.LanguageFreeIgnoredDirectories.Take(1).ToList(), .. languages.Select(l => l.IgnoredDirectories), .. languages.Select(l => l.LinkedDirectories),
             CodingToolOptions.LanguageFreeIgnoredDirectories.Skip(1).ToList()], StringComparer.OrdinalIgnoreCase);
        _ignored = new HashSet<string>(IgnoredDirectories, StringComparer.OrdinalIgnoreCase);
        string choice = string.IsNullOrWhiteSpace(_options.Languages) ? CodingLanguages.DefaultChoice : _options.Languages.Trim();
        _notice = new Lazy<string?>(() => choice.Equals(CodingLanguages.None, StringComparison.OrdinalIgnoreCase) ? null : CodingLanguages.Notice(_root, languages, _options.DetectionDepth));
        Sandbox = sandbox ?? CommandSandboxes.Get(_options.Sandbox);
        var environment = new Dictionary<string, string>(Quiet, StringComparer.Ordinal);
        foreach (var (name, value) in languages.SelectMany(l => l.Environment).Concat(_options.Environment ?? new Dictionary<string, string>()))
        {
            environment[name] = value;
        }

        _limits = new CommandLimits
        {
            Time = _options.CommandTimeout, MaxOutputCharacters = _options.MaxOutputCharacters, MaxMemoryBytes = _options.MaxMemoryBytes, MaxProcesses = _options.MaxProcesses,
            MaxCpus = _options.MaxCpus, Network = _options.Network, Programs = Commands, Environment = environment, Root = _root,
        };
        if (Sandbox.Check(_limits) is { } refusal)
        {
            throw new NotSupportedException(refusal);
        }

        // The examples in the descriptions: every language's, in their order (none without a language).
        IReadOnlyList<string> Examples(Func<CodingToolExamples, IReadOnlyList<string>> pick) => CodingLanguages.Merge([.. languages.Select(l => pick(l.ToolExamples))]);
        static string Example(string text, IReadOnlyList<string> examples, string separator) => examples.Count == 0 ? text + "." : $"{text}, e.g. {string.Join(separator, examples)}.";
        var tools = new List<Tool>
        {
            Tool.Create("list_files", "List the files of the workspace, or of a folder in it, as paths relative to the workspace. " +
                $"Build output and dependency folders ({string.Join(", ", Examples(e => e.Folders).Append(".git"))} …) are skipped.",
                Schema(("path", "string", "Folder to list, relative to the workspace (default: the whole workspace).", false),
                    ("pattern", "string", Example("Only file names matching this glob", Examples(e => e.FileGlobs), " or "), false)),
                (args, _) => Task.FromResult(ListFiles(Text(args, "path") ?? ".", Text(args, "pattern")))),
            Tool.Create("read_file", "Read a text file, with line numbers. Long files are returned in parts: pass start_line and end_line for the rest.",
                Schema(("path", "string", "File path, relative to the workspace.", true),
                    ("start_line", "integer", "First line to return (1-based, default 1).", false),
                    ("end_line", "integer", "Last line to return (default: as many as fit).", false)),
                (args, _) => Task.FromResult(Seen(Text(args, "path")!, ReadFile(Text(args, "path")!, Number(args, "start_line"), Number(args, "end_line"))))),
            Tool.Create("search", "Search the files of the workspace for a regular expression; returns path:line: text for each match.",
                Schema(("pattern", "string", Example("Regular expression (.NET syntax)", Examples(e => e.SearchPatterns), " or "), true),
                    ("path", "string", "Folder or file to search, relative to the workspace (default: everything).", false),
                    ("glob", "string", Example("Only files whose names match this glob", Examples(e => e.SearchGlobs), " or "), false),
                    ("ignore_case", "boolean", "Match case-insensitively.", false)),
                (args, _) => Task.FromResult(Search(Text(args, "pattern")!, Text(args, "path") ?? ".", Text(args, "glob"), args["ignore_case"]?.GetValue<bool>() ?? false))),
        };
        if (!_options.ReadOnly)
        {
            tools.Add(Tool.Create("edit_file", "Replace an exact piece of a file with new text. old_text must match the file exactly " +
                "(whitespace and indentation included) and occur once, unless replace_all is set; include enough surrounding lines to make it unique.",
                Schema(("path", "string", "File path, relative to the workspace.", true),
                    ("old_text", "string", "The exact text to replace.", true),
                    ("new_text", "string", "The replacement.", true),
                    ("replace_all", "boolean", "Replace every occurrence.", false)),
                (args, _) => Task.FromResult(Unread(Text(args, "path")!) ?? EditFile(Text(args, "path")!, Text(args, "old_text")!, Text(args, "new_text")!, args["replace_all"]?.GetValue<bool>() ?? false))));
            tools.Add(Tool.Create("write_file", "Create a file, or replace a file's whole content. Prefer edit_file for changes to existing files.",
                Schema(("path", "string", "File path, relative to the workspace; missing folders are created.", true),
                    ("content", "string", "The complete file content.", true)),
                (args, _) => Task.FromResult(Seen(Text(args, "path")!, WriteFile(Text(args, "path")!, Text(args, "content")!)))));
            tools.Add(Tool.Create("run_command", $"Run a command in the workspace and return its exit code and output. Allowed programs: " +
                $"{string.Join(", ", Commands)} (git only {string.Join(", ", _options.GitSubcommands)}). The command is not run by a shell: " +
                "no pipes, redirection, && or variables. " + Example("Use it to build, test, lint and inspect", Examples(e => e.Commands), ", "),
                Schema(("command", "string", Example("The command line", Examples(e => e.CommandLines), " or "), true),
                    ("directory", "string", "Folder to run in, relative to the workspace (default: the workspace).", false),
                    ("timeout_seconds", "integer", $"Stop the command after this many seconds (at most {(int)_options.CommandTimeout.TotalSeconds}).", false)),
                (args, token) => RunCommandAsync(Text(args, "command")!, Text(args, "directory") ?? ".", Number(args, "timeout_seconds"), token)));
        }

        All = [.. tools.Select(FailsOnError)];
    }

    // The files the task has read or written (CodingToolOptions.ReadBeforeEdit), by their full paths.
    private readonly System.Collections.Concurrent.ConcurrentDictionary<string, byte> _seen = new(StringComparer.Ordinal);

    /// <summary>Forgets which files were read or written, so edit_file asks for a read again (<see cref="CodingToolOptions.ReadBeforeEdit"/>); the agent calls it as each task starts.</summary>
    public void ForgetReads() => _seen.Clear();

    // `answer` (read_file's or write_file's), with the file marked as seen when it is not an error.
    private string Seen(string path, string answer)
    {
        if (!answer.StartsWith(ErrorPrefix, StringComparison.Ordinal) && Resolve(path, out string full) is null)
        {
            _seen[full] = 0;
        }

        return answer;
    }

    // The refusal of an edit of a file the task has not read or written, or null.
    private string? Unread(string path) =>
        _options.ReadBeforeEdit && Resolve(path, out string full) is null && File.Exists(full) && !_seen.ContainsKey(full)
            ? $"{ErrorPrefix}read {path} first (read_file): this task has not read it, and edit_file needs its exact text as it is now."
            : null;

    // What a tool's method answers a refusal or an error with: "Error: …" (the methods' own text, which callers of the
    // methods read as before).
    private const string ErrorPrefix = "Error: ";

    // The tool with its "Error: …" answers as failed calls (ToolFailedException: the registry's failed ToolResult with the
    // message), so the model reads that the call failed and an app marks it failed; every other answer as it is.
    private static Tool FailsOnError(Tool tool) => tool with
    {
        Invoke = async (arguments, token) =>
        {
            string text = await tool.Invoke(arguments, token).ConfigureAwait(false);
            return text.StartsWith(ErrorPrefix, StringComparison.Ordinal) ? throw new ToolFailedException(text[ErrorPrefix.Length..]) : text;
        },
    };

    /// <summary>The workspace folder (absolute).</summary>
    public string Workspace => _root;

    /// <summary>The coding languages the tools work with (<see cref="CodingToolOptions.Languages"/>, resolved), in order.</summary>
    public IReadOnlyList<ICodingLanguage> Languages { get; }

    /// <summary>The programs <c>run_command</c> may start (the options', or the languages' then git).</summary>
    public IReadOnlyList<string> Commands { get; }

    /// <summary>The folders never listed or searched (the options', or the language-free ones and the languages').</summary>
    public IReadOnlyList<string> IgnoredDirectories { get; }

    /// <summary>
    /// A notice when none of the languages detects the workspace although it holds source files
    /// (<see cref="CodingLanguages.Notice"/>: which plug-in to register); null otherwise, and with the choice
    /// <see cref="CodingLanguages.None"/>. Found on first read.
    /// </summary>
    public string? LanguageNotice => _notice.Value;

    /// <summary>The sandbox every command runs in.</summary>
    public ICommandSandbox Sandbox { get; }

    /// <summary>The limits every command runs under (the time limit capped per call by <see cref="ExecuteAsync"/>).</summary>
    public CommandLimits Limits => _limits;

    /// <summary>The tools, in the order a model sees them.</summary>
    public IReadOnlyList<Tool> All { get; }

    /// <summary>A registry with the tools (calls of one reply run one after another, as edits depend on each other).</summary>
    public ToolRegistry Registry() => ToolRegistry.Create().Add(All).Build();

    /// <summary><c>list_files</c>.</summary>
    public string ListFiles(string path = ".", string? pattern = null)
    {
        if (Resolve(path, out string full) is { } error)
        {
            return error;
        }

        if (File.Exists(full))
        {
            return Relative(full);
        }

        if (!Directory.Exists(full))
        {
            return $"Error: folder '{path}' does not exist.";
        }

        var glob = pattern is null ? null : GlobRegex(pattern);
        var files = Files(full).Where(f => glob is null || glob.IsMatch(Path.GetFileName(f))).Select(Relative).Order(StringComparer.Ordinal).ToList();
        if (files.Count == 0)
        {
            return pattern is null ? "(no files)" : $"(no files matching {pattern})";
        }

        var shown = files.Take(_options.MaxListedFiles);
        return string.Join('\n', shown) + (files.Count > _options.MaxListedFiles ? $"\n… {files.Count - _options.MaxListedFiles} more (list a folder or give a pattern)" : "");
    }

    /// <summary><c>read_file</c>.</summary>
    public string ReadFile(string path, int? startLine = null, int? endLine = null)
    {
        if (Resolve(path, out string full) is { } error)
        {
            return error;
        }

        if (!File.Exists(full))
        {
            return Directory.Exists(full) ? $"Error: '{path}' is a folder; use list_files." : $"Error: file '{path}' does not exist.";
        }

        string text = File.ReadAllText(full);
        var lines = SplitLines(text);
        if (lines.Count == 0)
        {
            return "(empty file)";
        }

        int first = Math.Max(1, startLine ?? 1);
        if (first > lines.Count)
        {
            return $"Error: the file has {lines.Count} lines.";
        }

        int last = Math.Min(lines.Count, Math.Min(endLine ?? int.MaxValue, first + _options.MaxReadLines - 1));
        int width = last.ToString(System.Globalization.CultureInfo.InvariantCulture).Length;
        var sb = new StringBuilder();
        Span<char> number = stackalloc char[11];        // the line number, right-aligned to width (as PadLeft)
        for (int i = first; i <= last; i++)
        {
            var (start, length) = lines[i - 1];
            i.TryFormat(number, out int digits, provider: System.Globalization.CultureInfo.InvariantCulture);
            sb.Append(' ', width - digits).Append(number[..digits]).Append("| ").Append(text.AsSpan(start, length)).Append('\n');
        }

        if (last < lines.Count && (endLine is null || last < endLine))
        {
            sb.Append($"… lines {last + 1}-{lines.Count} not shown (read_file with start_line {last + 1})\n");
        }

        return sb.ToString().TrimEnd('\n');
    }

    /// <summary><c>search</c>.</summary>
    public string Search(string pattern, string path = ".", string? glob = null, bool ignoreCase = false)
    {
        if (Resolve(path, out string full) is { } error)
        {
            return error;
        }

        Regex regex;
        try
        {
            regex = new Regex(pattern, RegexOptions.CultureInvariant | (ignoreCase ? RegexOptions.IgnoreCase : RegexOptions.None), TimeSpan.FromSeconds(2));
        }
        catch (ArgumentException ex)
        {
            return $"Error: invalid regular expression: {ex.Message}";
        }

        var files = File.Exists(full) ? [full] : Directory.Exists(full) ? Files(full) : null;
        if (files is null)
        {
            return $"Error: '{path}' does not exist.";
        }

        var nameFilter = glob is null ? null : GlobRegex(glob);
        var results = new List<string>();
        int total = 0;
        foreach (var file in files.Where(f => nameFilter is null || nameFilter.IsMatch(Path.GetFileName(f))).Order(StringComparer.Ordinal))
        {
            if (IsBinary(file))
            {
                continue;
            }

            string text = File.ReadAllText(file);
            var lines = SplitLines(text);
            for (int i = 0; i < lines.Count; i++)
            {
                var (start, length) = lines[i];
                try
                {
                    if (!regex.IsMatch(text.AsSpan(start, length)))
                    {
                        continue;
                    }
                }
                catch (RegexMatchTimeoutException)
                {
                    return "Error: the regular expression took too long; make it more specific.";
                }

                total++;
                if (results.Count < _options.MaxSearchResults)
                {
                    var line = text.AsSpan(start, length).Trim();
                    results.Add(line.Length > 200 ? $"{Relative(file)}:{i + 1}: {line[..200]} …" : $"{Relative(file)}:{i + 1}: {line}");
                }
            }
        }

        if (total == 0)
        {
            return "(no matches)";
        }

        return string.Join('\n', results) + (total > results.Count ? $"\n… {total - results.Count} more matches (narrow the path, glob or pattern)" : "");
    }

    /// <summary><c>edit_file</c>.</summary>
    public string EditFile(string path, string oldText, string newText, bool replaceAll = false)
    {
        if (Resolve(path, out string full) is { } error)
        {
            return error;
        }

        if (!File.Exists(full))
        {
            return $"Error: file '{path}' does not exist; use write_file to create it.";
        }

        if (oldText.Length == 0)
        {
            return "Error: old_text is empty; give the exact text to replace (or use write_file for a whole file).";
        }

        string text = File.ReadAllText(full);
        bool crlf = text.Contains("\r\n", StringComparison.Ordinal);
        string Match(string s) => crlf ? s.Replace("\r\n", "\n", StringComparison.Ordinal).Replace("\n", "\r\n", StringComparison.Ordinal) : s;
        string find = Match(oldText), replacement = Match(newText);
        int count = Count(text, find);
        if (count == 0)
        {
            if (!_options.EditTolerance || Tolerant(text, oldText, newText, Match) is not { } tolerant)
            {
                string hint = Count(Collapse(text), Collapse(find)) > 0
                    ? " It matches when whitespace is ignored: copy the exact indentation and line breaks from the file."
                    : " That text is not in the file: copy old_text from the file's text, not from memory.";
                return $"Error: old_text was not found in {path}. old_text was: {Escaped(oldText)}.{hint}{Quote(text)}";
            }

            if (tolerant.Count > 1)
            {
                return $"Error: old_text was not found exactly in {path}; {tolerant.How} it occurs {tolerant.Count} or more times. Copy the exact text, with enough surrounding lines to make it unique.";
            }

            string changed = string.Concat(text.AsSpan(0, tolerant.Start), tolerant.Replacement, text.AsSpan(tolerant.Start + tolerant.Length));
            WriteKeepingBom(full, changed);
            return Edited(path, full, text, tolerant.Start, tolerant.Replacement, $"1 replacement; old_text matched {tolerant.How}");
        }

        if (count > 1 && !replaceAll)
        {
            return $"Error: old_text occurs {count} times in {path}; include more surrounding lines to make it unique, or set replace_all.";
        }

        int at = text.IndexOf(find, StringComparison.Ordinal);
        string updated = replaceAll ? text.Replace(find, replacement, StringComparison.Ordinal) : string.Concat(text.AsSpan(0, at), replacement, text.AsSpan(at + find.Length));
        WriteKeepingBom(full, updated);
        return Edited(path, full, text, at, replacement, replaceAll ? $"{count} replacements" : "1 replacement");
    }

    // The changed lines with a little context, so the model sees the result without reading the file again.
    private string Edited(string path, string full, string before, int at, string replacement, string what)
    {
        int firstLine = before.AsSpan(0, at).Count('\n') + 1;
        int changedLines = SplitLines(replacement).Count;
        string view = ReadFile(path, Math.Max(1, firstLine - 2), firstLine + Math.Max(changedLines, 1) + 1);
        return $"Edited {Relative(full)} ({what}):\n{view}";
    }

    // Writes text as UTF-8, with the byte order mark when the file had one (File.ReadAllText drops it and
    // File.WriteAllText would not write it back: an edit would change the file's encoding).
    private static void WriteKeepingBom(string full, string text)
    {
        bool bom = false;
        if (File.Exists(full))
        {
            using var stream = File.OpenRead(full);
            Span<byte> head = stackalloc byte[3];
            bom = stream.Read(head) == 3 && head.SequenceEqual(Utf8Bom);
        }

        File.WriteAllText(full, text, new UTF8Encoding(encoderShouldEmitUTF8Identifier: bom));
    }

    private static ReadOnlySpan<byte> Utf8Bom => [0xEF, 0xBB, 0xBF];

    private const int MaxQuotedLine = 200;

    // old_text as an error shows it: in quotes, its line breaks, tabs, quotes and backslashes escaped, cut at
    // MaxQuotedLine characters.
    private static string Escaped(string text)
    {
        int shown = Math.Min(text.Length, MaxQuotedLine);
        var sb = new StringBuilder(shown + 8).Append('"');
        foreach (char c in text.AsSpan(0, shown))
        {
            _ = c switch
            {
                '\n' => sb.Append("\\n"),
                '\r' => sb.Append("\\r"),
                '\t' => sb.Append("\\t"),
                '"' => sb.Append("\\\""),
                '\\' => sb.Append("\\\\"),
                _ => sb.Append(c),
            };
        }

        return sb.Append(text.Length > MaxQuotedLine ? "\" …" : "\"").ToString();
    }

    // The file's first lines as they are (no line numbers), for an edit whose old_text was not found.
    private string Quote(string text)
    {
        if (_options.EditQuotedLines <= 0)
        {
            return "";
        }

        var lines = SplitLines(text);
        if (lines.Count == 0)
        {
            return " The file is empty.";
        }

        var invariant = System.Globalization.CultureInfo.InvariantCulture;
        bool all = lines.Count <= _options.EditQuotedLines;
        var sb = new StringBuilder(all ? " The file's text (without line numbers):\n```\n"
            : string.Create(invariant, $" The file's first {_options.EditQuotedLines} lines (without line numbers):\n```\n"));
        foreach (var (start, length) in lines.Take(_options.EditQuotedLines))
        {
            sb.Append(text.AsSpan(start, Math.Min(length, MaxQuotedLine))).Append(length > MaxQuotedLine ? " …\n" : "\n");
        }

        sb.Append("```");
        if (!all)
        {
            sb.Append(invariant, $"\n… {lines.Count - _options.EditQuotedLines} more lines (read_file with start_line {_options.EditQuotedLines + 1})");
        }

        return sb.ToString();
    }

    // read_file's decorations: a line's number before "| " (right-aligned), and the note on lines not shown.
    [GeneratedRegex(@"^[ \t]*(\d+)\| ?")]
    private static partial Regex LinePrefix();

    [GeneratedRegex(@"^… lines \d+-\d+ not shown")]
    private static partial Regex NotShown();

    // text without read_file's line numbers when every non-empty line has one, numbered in sequence (its "not shown"
    // notes dropped); null otherwise.
    private static string? WithoutLineNumbers(string text)
    {
        var sb = new StringBuilder(text.Length);
        int previous = -1, numbered = 0;
        foreach (var line in text.Replace("\r\n", "\n", StringComparison.Ordinal).Split('\n'))
        {
            if (NotShown().IsMatch(line))
            {
                continue;
            }

            var prefix = LinePrefix().Match(line);
            if (!prefix.Success)
            {
                if (line.Trim().Length > 0)
                {
                    return null;
                }

                sb.Append(line).Append('\n');
                continue;
            }

            if (!int.TryParse(prefix.Groups[1].ValueSpan, System.Globalization.NumberStyles.None, System.Globalization.CultureInfo.InvariantCulture, out int number)
                || previous >= 0 && number != previous + 1)
            {
                return null;
            }

            previous = number;
            numbered++;
            sb.Append(line.AsSpan(prefix.Length)).Append('\n');
        }

        if (numbered == 0)
        {
            return null;
        }

        sb.Length--;                                                         // the split's last piece had no line break after it
        return sb.ToString();
    }

    // Where a tolerant match of edit_file's old_text is, what replaces it, how it matched, and how many matches (2: two or more).
    private sealed record TolerantMatch(int Start, int Length, string Replacement, string How, int Count);

    // old_text found the tolerant ways (CodingToolOptions.EditTolerance): the first way that finds it; null when none does.
    private static TolerantMatch? Tolerant(string text, string oldText, string newText, Func<string, string> endings)
    {
        const string Lines = "with trailing white space and the final line break ignored";
        if (WithoutLineNumbers(oldText) is { Length: > 0 } stripped)
        {
            const string Numbers = "after removing read_file's line numbers";
            string replacement = endings(WithoutLineNumbers(newText) ?? newText), find = endings(stripped);
            int count = Count(text, find);
            if (count > 0)
            {
                return new TolerantMatch(text.IndexOf(find, StringComparison.Ordinal), find.Length, replacement, Numbers, Math.Min(count, 2));
            }

            if (ByLines(text, stripped, replacement, $"{Numbers}, {Lines}") is { } numbered)
            {
                return numbered;
            }
        }

        return ByLines(text, oldText, endings(newText), Lines);
    }

    // old_text's whole lines against the file's, each without its trailing white space; a final line break in old_text
    // ignored (one then taken off the replacement: the file keeps its own). The region is the matched lines' text
    // without the last one's line break.
    private static TolerantMatch? ByLines(string text, string oldText, string replacement, string how)
    {
        string find = oldText.Replace("\r\n", "\n", StringComparison.Ordinal);
        bool broken = find.EndsWith('\n');
        if (broken)
        {
            find = find[..^1];
        }

        string[] wanted = find.Split('\n');
        if (wanted.All(w => w.Trim().Length == 0))
        {
            return null;
        }

        var lines = SplitLines(text);
        int found = 0, first = -1;
        for (int i = 0; i + wanted.Length <= lines.Count && found < 2; i++)
        {
            bool same = true;
            for (int k = 0; k < wanted.Length && same; k++)
            {
                var (start, length) = lines[i + k];
                same = text.AsSpan(start, length).TrimEnd(" \t\r").SequenceEqual(wanted[k].AsSpan().TrimEnd(" \t\r"));
            }

            if (same)
            {
                found++;
                first = first < 0 ? i : first;
            }
        }

        if (found == 0)
        {
            return null;
        }

        int from = lines[first].Start;
        var (lastStart, lastLength) = lines[first + wanted.Length - 1];
        if (broken)
        {
            replacement = replacement.EndsWith("\r\n", StringComparison.Ordinal) ? replacement[..^2] : replacement.EndsWith('\n') ? replacement[..^1] : replacement;
        }

        return new TolerantMatch(from, lastStart + lastLength - from, replacement, how, found);
    }

    /// <summary><c>write_file</c>.</summary>
    public string WriteFile(string path, string content)
    {
        if (Resolve(path, out string full) is { } error)
        {
            return error;
        }

        if (Directory.Exists(full))
        {
            return $"Error: '{path}' is a folder.";
        }

        bool existed = File.Exists(full);
        if (existed && File.ReadAllText(full).Contains("\r\n", StringComparison.Ordinal))
        {
            content = content.Replace("\r\n", "\n", StringComparison.Ordinal).Replace("\n", "\r\n", StringComparison.Ordinal);   // keep the file's line endings
        }

        Directory.CreateDirectory(Path.GetDirectoryName(full)!);
        WriteKeepingBom(full, content);
        return $"{(existed ? "Replaced" : "Created")} {Relative(full)} ({SplitLines(content).Count} lines).";
    }

    /// <summary><c>run_command</c>: the command's outcome as the model sees it.</summary>
    public async Task<string> RunCommandAsync(string command, string directory = ".", int? timeoutSeconds = null, CancellationToken cancellationToken = default)
    {
        var result = await ExecuteAsync(command, directory, timeoutSeconds is { } s ? TimeSpan.FromSeconds(Math.Max(1, s)) : null, cancellationToken).ConfigureAwait(false);
        if (result.Refused)
        {
            return result.Output;
        }

        // The text the model reads: the same digits and decimal point on every machine, whatever its culture.
        var invariant = System.Globalization.CultureInfo.InvariantCulture;
        string status = result.TimedOut ? string.Create(invariant, $"timed out after {result.Duration.TotalSeconds:0} s (stopped)")
            : result.OutOfMemory ? string.Create(invariant, $"stopped: it used more than the memory limit ({_limits.MaxMemoryBytes / (1024.0 * 1024.0):0.#} MB)")
            : string.Create(invariant, $"exit code {result.ExitCode}");
        return string.Create(invariant, $"$ {command}\n{status}, {result.Duration.TotalSeconds:0.0} s\n{(result.Output.Length == 0 ? "(no output)" : result.Output)}");
    }

    /// <summary>
    /// Runs <paramref name="command"/> under the same rules as <c>run_command</c> (allowlist, no shell, the workspace) in
    /// <see cref="Sandbox"/>; the timeout is capped at <see cref="CodingToolOptions.CommandTimeout"/>.
    /// </summary>
    public async Task<CommandResult> ExecuteAsync(string command, string directory = ".", TimeSpan? timeout = null, CancellationToken cancellationToken = default)
    {
        static CommandResult Refuse(string why) => new(null, false, why, TimeSpan.Zero, true);
        if (Resolve(directory, out string folder) is { } error)
        {
            return Refuse(error);
        }

        if (!Directory.Exists(folder))
        {
            return Refuse($"Error: folder '{directory}' does not exist.");
        }

        if (command.IndexOfAny(['|', '&', ';', '>', '<', '`', '$', '\n']) >= 0)
        {
            return Refuse("Error: the command is not run by a shell, so pipes, redirection, &&, ; and variables are not available. Run one program per call.");
        }

        var words = SplitCommandLine(command);
        if (words.Count == 0)
        {
            return Refuse("Error: empty command.");
        }

        string program = Path.GetFileNameWithoutExtension(words[0]);
        if (!Commands.Contains(program, StringComparer.OrdinalIgnoreCase) || words[0].Contains('/') || words[0].Contains('\\'))
        {
            return Refuse($"Error: '{words[0]}' is not an allowed program. Allowed: {string.Join(", ", Commands)}.");
        }

        if (program.Equals("git", StringComparison.OrdinalIgnoreCase) && (words.Count < 2 || !_options.GitSubcommands.Contains(words[1])))
        {
            return Refuse($"Error: only git {string.Join(", ", _options.GitSubcommands)} are allowed.");
        }

        var limitTime = timeout is { } t && t < _options.CommandTimeout ? t : _options.CommandTimeout;
        return await Sandbox.RunAsync(words, folder, _limits with { Time = limitTime }, cancellationToken).ConfigureAwait(false);
    }

    // Null when `path` stays inside the workspace (then `full` is its absolute path), else the error for the model.
    private string? Resolve(string path, out string full)
    {
        full = Path.GetFullPath(Path.Combine(_root, path.Trim()));
        bool inside = full.Equals(_root, StringComparison.Ordinal)
                      || full.StartsWith(_root + Path.DirectorySeparatorChar, OperatingSystem.IsWindows() ? StringComparison.OrdinalIgnoreCase : StringComparison.Ordinal);
        return inside ? null : $"Error: '{path}' is outside the workspace; use paths relative to it.";
    }

    private string Relative(string full) => Path.GetRelativePath(_root, full).Replace('\\', '/');

    private IEnumerable<string> Files(string folder)
    {
        var pending = new Stack<string>();
        pending.Push(folder);
        while (pending.Count > 0)
        {
            string current = pending.Pop();
            IEnumerable<string> entries;
            try
            {
                entries = Directory.EnumerateFileSystemEntries(current);
            }
            catch (Exception ex) when (ex is UnauthorizedAccessException or IOException)
            {
                continue;
            }

            foreach (var entry in entries)
            {
                if (Directory.Exists(entry))
                {
                    if (!_ignored.Contains(Path.GetFileName(entry)))
                    {
                        pending.Push(entry);
                    }
                }
                else
                {
                    yield return entry;
                }
            }
        }
    }

    private static JsonObject Schema(params (string Name, string Type, string Description, bool Required)[] properties)
    {
        var props = new JsonObject();
        foreach (var (name, type, description, _) in properties)
        {
            props[name] = new JsonObject { ["type"] = type, ["description"] = description };
        }

        return new JsonObject
        {
            ["type"] = "object",
            ["properties"] = props,
            ["required"] = new JsonArray([.. properties.Where(p => p.Required).Select(p => (JsonNode?)JsonValue.Create(p.Name))]),
        };
    }

    private static string? Text(JsonObject args, string name) => args[name] is JsonValue v && v.TryGetValue<string>(out var s) ? s : null;

    private static int? Number(JsonObject args, string name) => args[name] is JsonValue v && v.TryGetValue<double>(out var d) ? (int)d : null;

    // The lines of text as places in it (no string per line): cut at '\n', a '\r' right before it left out (so "\r\n" ends
    // a line as '\n' does; a lone '\r' stays in the line).
    private static List<(int Start, int Length)> SplitLines(string text)
    {
        var lines = new List<(int Start, int Length)>();
        for (int start = 0; ;)
        {
            int end = text.IndexOf('\n', start);
            if (end < 0)
            {
                if (start < text.Length)
                {
                    lines.Add((start, text.Length - start));        // no newline after the last line (none after a final one)
                }

                return lines;
            }

            lines.Add((start, end > start && text[end - 1] == '\r' ? end - 1 - start : end - start));
            start = end + 1;
        }
    }

    private static int Count(string text, string find)
    {
        int count = 0;
        for (int at = text.IndexOf(find, StringComparison.Ordinal); at >= 0; at = text.IndexOf(find, at + find.Length, StringComparison.Ordinal))
        {
            count++;
        }

        return count;
    }

    private static string Collapse(string text) => Regex.Replace(text, @"\s+", " ").Trim();

    private static Regex GlobRegex(string glob) =>
        new("^" + Regex.Escape(glob).Replace(@"\*", ".*", StringComparison.Ordinal).Replace(@"\?", ".", StringComparison.Ordinal) + "$",
            RegexOptions.IgnoreCase | RegexOptions.CultureInvariant);

    private static bool IsBinary(string file)
    {
        try
        {
            using var stream = File.OpenRead(file);
            Span<byte> head = stackalloc byte[512];
            int read = stream.Read(head);
            return head[..read].Contains((byte)0);
        }
        catch (IOException)
        {
            return true;
        }
    }

    // Words of a command line: spaces separate, double or single quotes group (without the quotes).
    private static List<string> SplitCommandLine(string command)
    {
        var words = new List<string>();
        var current = new StringBuilder();
        char quote = '\0';
        bool any = false;
        foreach (char c in command.Trim())
        {
            if (quote != '\0')
            {
                if (c == quote)
                {
                    quote = '\0';
                }
                else
                {
                    current.Append(c);
                }
            }
            else if (c is '"' or '\'')
            {
                quote = c;
                any = true;
            }
            else if (char.IsWhiteSpace(c))
            {
                if (current.Length > 0 || any)
                {
                    words.Add(current.ToString());
                    current.Clear();
                    any = false;
                }
            }
            else
            {
                current.Append(c);
            }
        }

        if (current.Length > 0 || any)
        {
            words.Add(current.ToString());
        }

        return words;
    }
}
