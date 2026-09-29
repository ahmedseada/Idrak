using System.Diagnostics;
using System.Text;
using System.Text.Json.Nodes;
using System.Text.RegularExpressions;

namespace Idrak.Generation;

/// <summary>The outcome of <see cref="CodingTools.ExecuteAsync"/>.</summary>
/// <param name="ExitCode">The exit code, or null when the command timed out or was refused.</param>
/// <param name="TimedOut">Stopped at the time limit.</param>
/// <param name="Output">Standard output and error, interleaved and trimmed (for a refused command: why).</param>
/// <param name="Duration">How long it ran.</param>
/// <param name="Refused">Not started (not allowed, not installed, outside the workspace).</param>
public sealed record CommandResult(int? ExitCode, bool TimedOut, string Output, TimeSpan Duration, bool Refused)
{
    /// <summary>Ran and exited with 0.</summary>
    public bool Succeeded => ExitCode == 0;
}

/// <summary>Limits and permissions of <see cref="CodingTools"/>.</summary>
public sealed record CodingToolOptions
{
    /// <summary>Programs <c>run_command</c> may start (file names without extension; resolved on the PATH).</summary>
    public IReadOnlyList<string> Commands { get; init; } = ["dotnet", "npm", "npx", "ng", "node", "tsc", "git"];

    /// <summary>The only <c>git</c> subcommands allowed (read-only by default: the agent edits files, not history).</summary>
    public IReadOnlyList<string> GitSubcommands { get; init; } = ["status", "diff", "log", "show", "ls-files", "grep", "blame"];

    /// <summary>Longest a command may run (the model may ask for less).</summary>
    public TimeSpan CommandTimeout { get; init; } = TimeSpan.FromMinutes(5);

    /// <summary>Characters of command output returned (the start and the end are kept, the middle is cut).</summary>
    public int MaxOutputCharacters { get; init; } = 12_000;

    /// <summary>Lines <c>read_file</c> returns per call.</summary>
    public int MaxReadLines { get; init; } = 400;

    /// <summary>Matches <c>search</c> returns.</summary>
    public int MaxSearchResults { get; init; } = 100;

    /// <summary>Paths <c>list_files</c> returns.</summary>
    public int MaxListedFiles { get; init; } = 400;

    /// <summary>Directories never listed or searched (build output, dependencies, tool state).</summary>
    public IReadOnlyList<string> IgnoredDirectories { get; init; } = [".git", "bin", "obj", "node_modules", "dist", ".angular", ".vs", ".idea", "coverage", "out"];

    /// <summary>Only the reading tools (no edits, writes or commands).</summary>
    public bool ReadOnly { get; init; }
}

/// <summary>
/// Tools for a coding agent working in one folder (the workspace): <c>list_files</c>, <c>read_file</c>, <c>search</c>,
/// <c>edit_file</c> (exact search and replace), <c>write_file</c> and <c>run_command</c> (allowlisted programs such as
/// dotnet, npm, ng, started without a shell). Paths are relative to the workspace and cannot leave it; results are
/// plain text sized for a model's context; mistakes come back as messages the model can act on.
/// </summary>
public sealed class CodingTools
{
    private readonly string _root;
    private readonly CodingToolOptions _options;

    /// <summary>Tools over <paramref name="workspace"/>.</summary>
    public CodingTools(string workspace, CodingToolOptions? options = null)
    {
        _root = Path.GetFullPath(workspace).TrimEnd(Path.DirectorySeparatorChar, Path.AltDirectorySeparatorChar);
        if (!Directory.Exists(_root))
        {
            throw new DirectoryNotFoundException($"Workspace '{_root}' does not exist.");
        }

        _options = options ?? new CodingToolOptions();
        var tools = new List<Tool>
        {
            Tool.Create("list_files", "List the files of the workspace, or of a folder in it, as paths relative to the workspace. " +
                "Build output and dependency folders (bin, obj, node_modules, dist, .git …) are skipped.",
                Schema(("path", "string", "Folder to list, relative to the workspace (default: the whole workspace).", false),
                    ("pattern", "string", "Only file names matching this glob, e.g. *.cs or *.component.ts.", false)),
                (args, _) => Task.FromResult(ListFiles(Text(args, "path") ?? ".", Text(args, "pattern")))),
            Tool.Create("read_file", "Read a text file, with line numbers. Long files are returned in parts: pass start_line and end_line for the rest.",
                Schema(("path", "string", "File path, relative to the workspace.", true),
                    ("start_line", "integer", "First line to return (1-based, default 1).", false),
                    ("end_line", "integer", "Last line to return (default: as many as fit).", false)),
                (args, _) => Task.FromResult(ReadFile(Text(args, "path")!, Number(args, "start_line"), Number(args, "end_line")))),
            Tool.Create("search", "Search the files of the workspace for a regular expression; returns path:line: text for each match.",
                Schema(("pattern", "string", "Regular expression (.NET syntax), e.g. class \\w+Service or ngOnInit.", true),
                    ("path", "string", "Folder or file to search, relative to the workspace (default: everything).", false),
                    ("glob", "string", "Only files whose names match this glob, e.g. *.ts.", false),
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
                (args, _) => Task.FromResult(EditFile(Text(args, "path")!, Text(args, "old_text")!, Text(args, "new_text")!, args["replace_all"]?.GetValue<bool>() ?? false))));
            tools.Add(Tool.Create("write_file", "Create a file, or replace a file's whole content. Prefer edit_file for changes to existing files.",
                Schema(("path", "string", "File path, relative to the workspace; missing folders are created.", true),
                    ("content", "string", "The complete file content.", true)),
                (args, _) => Task.FromResult(WriteFile(Text(args, "path")!, Text(args, "content")!))));
            tools.Add(Tool.Create("run_command", $"Run a command in the workspace and return its exit code and output. Allowed programs: " +
                $"{string.Join(", ", _options.Commands)} (git only {string.Join(", ", _options.GitSubcommands)}). The command is not run by a shell: " +
                "no pipes, redirection, && or variables. Use it to build, test, lint and inspect, e.g. dotnet build, dotnet test, npm test, npx tsc --noEmit, ng build.",
                Schema(("command", "string", "The command line, e.g. dotnet test --no-restore.", true),
                    ("directory", "string", "Folder to run in, relative to the workspace (default: the workspace).", false),
                    ("timeout_seconds", "integer", $"Stop the command after this many seconds (at most {(int)_options.CommandTimeout.TotalSeconds}).", false)),
                (args, token) => RunCommandAsync(Text(args, "command")!, Text(args, "directory") ?? ".", Number(args, "timeout_seconds"), token)));
        }

        All = tools;
    }

    /// <summary>The workspace folder (absolute).</summary>
    public string Workspace => _root;

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

        var lines = SplitLines(File.ReadAllText(full));
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
        for (int i = first; i <= last; i++)
        {
            sb.Append(i.ToString(System.Globalization.CultureInfo.InvariantCulture).PadLeft(width)).Append("| ").Append(lines[i - 1]).Append('\n');
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

            var lines = SplitLines(File.ReadAllText(file));
            for (int i = 0; i < lines.Count; i++)
            {
                try
                {
                    if (!regex.IsMatch(lines[i]))
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
                    string line = lines[i].Trim();
                    results.Add($"{Relative(file)}:{i + 1}: {(line.Length > 200 ? line[..200] + " …" : line)}");
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
            string hint = Count(Collapse(text), Collapse(find)) > 0
                ? " It matches when whitespace is ignored: copy the exact indentation and line breaks from read_file."
                : " Read the file again (read_file) and copy the text exactly.";
            return $"Error: old_text was not found in {path}.{hint}";
        }

        if (count > 1 && !replaceAll)
        {
            return $"Error: old_text occurs {count} times in {path}; include more surrounding lines to make it unique, or set replace_all.";
        }

        int at = text.IndexOf(find, StringComparison.Ordinal);
        string updated = replaceAll ? text.Replace(find, replacement, StringComparison.Ordinal) : text[..at] + replacement + text[(at + find.Length)..];
        File.WriteAllText(full, updated);

        // The changed lines with a little context, so the model sees the result without reading the file again.
        int firstLine = text[..at].Count(c => c == '\n') + 1;
        int changedLines = SplitLines(replacement).Count;
        string view = ReadFile(path, Math.Max(1, firstLine - 2), firstLine + Math.Max(changedLines, 1) + 1);
        return $"Edited {Relative(full)} ({(replaceAll ? $"{count} replacements" : "1 replacement")}):\n{view}";
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
        File.WriteAllText(full, content);
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

        string status = result.TimedOut ? $"timed out after {result.Duration.TotalSeconds:0} s (stopped)" : $"exit code {result.ExitCode}";
        return $"$ {command}\n{status}, {result.Duration.TotalSeconds:0.0} s\n{(result.Output.Length == 0 ? "(no output)" : result.Output)}";
    }

    /// <summary>
    /// Runs <paramref name="command"/> under the same rules as <c>run_command</c> (allowlist, no shell, the workspace);
    /// the timeout is capped at <see cref="CodingToolOptions.CommandTimeout"/>.
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
        if (!_options.Commands.Contains(program, StringComparer.OrdinalIgnoreCase) || words[0].Contains('/') || words[0].Contains('\\'))
        {
            return Refuse($"Error: '{words[0]}' is not an allowed program. Allowed: {string.Join(", ", _options.Commands)}.");
        }

        if (program.Equals("git", StringComparison.OrdinalIgnoreCase) && (words.Count < 2 || !_options.GitSubcommands.Contains(words[1])))
        {
            return Refuse($"Error: only git {string.Join(", ", _options.GitSubcommands)} are allowed.");
        }

        if (FindProgram(program) is not { } executable)
        {
            return Refuse($"Error: '{program}' is not installed (not found on the PATH).");
        }

        var start = new ProcessStartInfo(executable)
        {
            WorkingDirectory = folder,
            RedirectStandardOutput = true,
            RedirectStandardError = true,
            RedirectStandardInput = true,
            UseShellExecute = false,
            CreateNoWindow = true,
        };
        foreach (var word in words.Skip(1))
        {
            start.ArgumentList.Add(word);
        }

        // Non-interactive, uncoloured, no telemetry prompts.
        start.Environment["CI"] = "1";
        start.Environment["NO_COLOR"] = "1";
        start.Environment["FORCE_COLOR"] = "0";
        start.Environment["NG_CLI_ANALYTICS"] = "false";
        start.Environment["DOTNET_CLI_TELEMETRY_OPTOUT"] = "1";
        start.Environment["DOTNET_NOLOGO"] = "1";
        start.Environment["DOTNET_SKIP_FIRST_TIME_EXPERIENCE"] = "1";
        start.Environment["npm_config_yes"] = "true";
        start.Environment["npm_config_fund"] = "false";
        start.Environment["npm_config_audit"] = "false";

        var output = new StringBuilder();
        var limitTime = timeout is { } t && t < _options.CommandTimeout ? t : _options.CommandTimeout;
        var watch = Stopwatch.StartNew();
        using var process = new Process { StartInfo = start };
        process.OutputDataReceived += (_, e) => { if (e.Data is not null) { lock (output) { output.Append(e.Data).Append('\n'); } } };
        process.ErrorDataReceived += (_, e) => { if (e.Data is not null) { lock (output) { output.Append(e.Data).Append('\n'); } } };
        try
        {
            process.Start();
        }
        catch (System.ComponentModel.Win32Exception ex)
        {
            return Refuse($"Error: could not start '{program}': {ex.Message}");
        }

        process.StandardInput.Close();
        process.BeginOutputReadLine();
        process.BeginErrorReadLine();
        using var limit = CancellationTokenSource.CreateLinkedTokenSource(cancellationToken);
        limit.CancelAfter(limitTime);
        bool timedOut = false;
        try
        {
            await process.WaitForExitAsync(limit.Token).ConfigureAwait(false);
        }
        catch (OperationCanceledException)
        {
            timedOut = !cancellationToken.IsCancellationRequested;
            try
            {
                process.Kill(entireProcessTree: true);
            }
            catch (InvalidOperationException)
            {
            }

            if (!timedOut)
            {
                throw;
            }
        }

        process.WaitForExit();                                      // flushes the output events
        string text;
        lock (output)
        {
            text = Trim(output.ToString().TrimEnd());
        }

        return new CommandResult(timedOut ? null : process.ExitCode, timedOut, text, watch.Elapsed, false);
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
                    if (!_options.IgnoredDirectories.Contains(Path.GetFileName(entry), StringComparer.OrdinalIgnoreCase))
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

    private string Trim(string text)
    {
        int max = _options.MaxOutputCharacters;
        if (text.Length <= max)
        {
            return text;
        }

        int head = max / 3, tail = max - head;
        return $"{text[..head]}\n… {text.Length - max} characters cut …\n{text[^tail..]}";
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

    private static List<string> SplitLines(string text)
    {
        var lines = text.Replace("\r\n", "\n", StringComparison.Ordinal).Split('\n').ToList();
        if (lines.Count > 0 && lines[^1].Length == 0)
        {
            lines.RemoveAt(lines.Count - 1);                        // the newline that ends the last line
        }

        return lines;
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

    // The program on the PATH (with the Windows executable extensions: npm is npm.cmd there).
    private static string? FindProgram(string name)
    {
        var extensions = OperatingSystem.IsWindows()
            ? (Environment.GetEnvironmentVariable("PATHEXT") ?? ".COM;.EXE;.BAT;.CMD").Split(';', StringSplitOptions.RemoveEmptyEntries)
            : [""];
        foreach (var folder in (Environment.GetEnvironmentVariable("PATH") ?? "").Split(Path.PathSeparator, StringSplitOptions.RemoveEmptyEntries))
        {
            foreach (var extension in extensions)
            {
                string candidate = Path.Combine(folder.Trim('"'), name + extension.ToLowerInvariant());
                if (File.Exists(candidate))
                {
                    return candidate;
                }
            }
        }

        return null;
    }
}
