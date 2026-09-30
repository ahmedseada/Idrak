using System.Diagnostics;
using System.Text.Json.Nodes;

namespace Idrak.Generation;

/// <summary>
/// A coding task: a prompt, the project it starts from, and the commands that decide whether it was done. On disk a task
/// is a folder with <c>task.json</c> (<c>{"prompt", "language", "base", "setup": [...], "verify": [...], "tags": [...]}</c>), a
/// <c>workspace/</c> folder with the starting files (laid over <c>base</c>, a shared project folder relative to the task),
/// optionally a <c>verify/</c> folder whose files (hidden tests) are copied in after the agent finishes, and a
/// <c>solution/</c> folder with a reference solution laid over the workspace (<see cref="CodingAgent.CheckTaskAsync"/>
/// checks that verification fails without it and passes with it).
/// </summary>
/// <param name="Id">Name (the folder's path within the suite).</param>
/// <param name="Prompt">What the user asks for.</param>
/// <param name="Language">"csharp", "typescript", "angular" … (for reports).</param>
/// <param name="Base">A project folder copied first, or null.</param>
/// <param name="Workspace">Starting files laid over <paramref name="Base"/>, or null.</param>
/// <param name="VerifyFiles">Files copied in before verification, or null.</param>
/// <param name="Setup">Commands run before the agent starts (for example dotnet restore), not shown to it.</param>
/// <param name="Verify">Commands that must all exit with 0 for the task to count as done.</param>
/// <param name="Tags">Free-form labels.</param>
/// <param name="Solution">A reference solution laid over the workspace, or null.</param>
public sealed record AgentTask(string Id, string Prompt, string? Language = null, string? Base = null, string? Workspace = null, string? VerifyFiles = null,
    IReadOnlyList<string>? Setup = null, IReadOnlyList<string>? Verify = null, IReadOnlyList<string>? Tags = null, string? Solution = null)
{
    /// <summary>Every task under <paramref name="folder"/> (each folder holding a task.json), ordered by id.</summary>
    public static IReadOnlyList<AgentTask> LoadSuite(string folder)
    {
        folder = Path.GetFullPath(folder);
        return [.. Directory.EnumerateFiles(folder, "task.json", SearchOption.AllDirectories)
            .Where(f => !f.Split(Path.DirectorySeparatorChar).Contains("node_modules"))
            .Select(f => Load(Path.GetDirectoryName(f)!, Path.GetRelativePath(folder, Path.GetDirectoryName(f)!).Replace('\\', '/')))
            .OrderBy(t => t.Id, StringComparer.Ordinal)];
    }

    /// <summary>The task in <paramref name="folder"/>.</summary>
    public static AgentTask Load(string folder, string? id = null)
    {
        folder = Path.GetFullPath(folder);
        var json = JsonNode.Parse(File.ReadAllText(Path.Combine(folder, "task.json"))) as JsonObject
                   ?? throw new InvalidDataException($"{folder}/task.json is not a JSON object.");
        static IReadOnlyList<string>? List(JsonNode? node) => node is JsonArray a ? [.. a.Select(n => (string?)n ?? "")] : null;
        string? Folder(string? path) => path is not null && Directory.Exists(Path.Combine(folder, path)) ? Path.GetFullPath(Path.Combine(folder, path)) : null;
        string? baseFolder = (string?)json["base"];
        if (baseFolder is not null && Folder(baseFolder) is null)
        {
            throw new DirectoryNotFoundException($"{folder}/task.json: base folder '{baseFolder}' does not exist.");
        }

        return new AgentTask(id ?? (string?)json["id"] ?? Path.GetFileName(folder),
            (string?)json["prompt"] ?? throw new InvalidDataException($"{folder}/task.json has no prompt."),
            (string?)json["language"], Folder(baseFolder), Folder("workspace"), Folder("verify"),
            List(json["setup"]), List(json["verify"]), List(json["tags"]), Folder("solution"));
    }
}

/// <summary>How <see cref="CodingAgent"/> works.</summary>
public sealed record AgentOptions
{
    /// <summary>The system prompt (default: <see cref="CodingAgent.DefaultSystemPrompt"/>).</summary>
    public string? SystemPrompt { get; init; }

    /// <summary>Reasoning mode for every request.</summary>
    public bool? Think { get; init; }

    /// <summary>Sampling and length options for every request.</summary>
    public GenerationOptions? Sampling { get; init; }

    /// <summary>Model replies allowed per task.</summary>
    public int MaxRounds { get; init; } = 40;

    /// <summary>Wall-clock limit per task (model and tools; not setup or verification).</summary>
    public TimeSpan TimeLimit { get; init; } = TimeSpan.FromMinutes(20);

    /// <summary>What the tools may do.</summary>
    public CodingToolOptions Tools { get; init; } = new();

    /// <summary>Time limit of each setup and verification command.</summary>
    public TimeSpan VerifyTimeout { get; init; } = TimeSpan.FromMinutes(10);
}

/// <summary>How a run ended.</summary>
public enum AgentOutcome
{
    /// <summary>The agent finished and every verification command succeeded (or the task has none).</summary>
    Passed,

    /// <summary>A verification command failed.</summary>
    Failed,

    /// <summary>The agent used all its rounds or time; verification still ran and failed.</summary>
    OutOfBudget,

    /// <summary>A setup command failed: the task, not the agent, is broken.</summary>
    SetupFailed,

    /// <summary>The model failed (for example the server was unreachable).</summary>
    Error,
}

/// <summary>One agent run: the conversation and its outcome.</summary>
/// <param name="Task">The task.</param>
/// <param name="Messages">System, user, assistant and tool messages.</param>
/// <param name="Tools">The tools the model was offered.</param>
/// <param name="Think">The reasoning mode used.</param>
/// <param name="Outcome">How it ended.</param>
/// <param name="VerifyOutput">The verification commands' output (or the setup's, or the error).</param>
/// <param name="Rounds">Model replies.</param>
/// <param name="ToolCalls">Tool calls made.</param>
/// <param name="ToolErrors">Tool calls that returned an error.</param>
/// <param name="GeneratedTokens">Tokens the model generated (when it reports them).</param>
/// <param name="ModelTime">Time spent waiting for the model.</param>
/// <param name="ToolTime">Time spent running tools.</param>
/// <param name="Workspace">The folder the agent worked in.</param>
public sealed record AgentRun(AgentTask Task, IReadOnlyList<ChatMessage> Messages, IReadOnlyList<ToolDefinition> Tools, bool? Think, AgentOutcome Outcome,
    string VerifyOutput, int Rounds, int ToolCalls, int ToolErrors, int GeneratedTokens, TimeSpan ModelTime, TimeSpan ToolTime, string Workspace)
{
    /// <summary>The run as one JSON line: the conversation in the chat format with the outcome and figures as extra fields.</summary>
    public JsonObject ToJson()
    {
        var json = ChatJson.Transcript(Messages, Tools, Think);
        json["task"] = Task.Id;
        if (Task.Language is { } language)
        {
            json["language"] = language;
        }

        json["outcome"] = Outcome.ToString();
        json["rounds"] = Rounds;
        json["tool_calls"] = ToolCalls;
        json["tool_errors"] = ToolErrors;
        json["generated_tokens"] = GeneratedTokens;
        json["model_seconds"] = Math.Round(ModelTime.TotalSeconds, 2);
        json["tool_seconds"] = Math.Round(ToolTime.TotalSeconds, 2);
        return json;
    }
}

/// <summary>
/// A coding agent: a chat model with <see cref="CodingTools"/> in a workspace, looping until it answers without a tool
/// call. <see cref="RunAsync(AgentTask, string, CancellationToken)"/> prepares a copy of a task's project, runs the agent and
/// verifies the result: an evaluation of the model on the task.
/// </summary>
public sealed class CodingAgent(IChatModel model, AgentOptions? options = null)
{
    /// <summary>The instructions the agent gets by default.</summary>
    public const string DefaultSystemPrompt = """
        You are a coding agent working in a software project on the user's machine. You have tools to list, read and
        search the project's files, edit and write files, and run commands such as builds and tests.

        How to work:
        - Understand before changing: find the relevant files (list_files, search) and read them before editing.
        - Make focused changes with edit_file; copy old_text exactly as the file has it (read_file shows line numbers
          before a "| " separator: they are not part of the file). Create new files with write_file.
        - Follow the project's conventions: its naming, formatting, file layout, frameworks and library versions.
        - Check your work: build and run the tests after changing code (C#: dotnet build, dotnet test; TypeScript:
          npx tsc --noEmit, npm test; Angular: npx ng build, npx ng test --watch=false). Read the errors, fix them and
          run the checks again until they pass.
        - Do not weaken or delete tests to make them pass, unless the user asks for it.
        - Do only what was asked; mention anything else worth doing instead of doing it.
        - When the task is done, answer without a tool call: say briefly what you changed and how you checked it.
        """;

    private readonly AgentOptions _options = options ?? new AgentOptions();

    /// <summary>Called with each piece of the model's replies as they stream.</summary>
    public Action<ChatDelta>? OnDelta { get; init; }

    /// <summary>Called after each tool call.</summary>
    public Action<ToolResult>? OnToolResult { get; init; }

    /// <summary>Called with progress lines (setup, verification).</summary>
    public Action<string>? OnStatus { get; init; }

    /// <summary>
    /// Runs <paramref name="task"/> in a fresh copy of its project under <paramref name="workFolder"/> (created; left in
    /// place for inspection): setup, the agent, the hidden files, verification.
    /// </summary>
    public async Task<AgentRun> RunAsync(AgentTask task, string workFolder, CancellationToken cancellationToken = default)
    {
        PrepareWorkspace(task, workFolder);
        var tools = new CodingTools(workFolder, _options.Tools);
        var checks = new CodingTools(workFolder, _options.Tools with { CommandTimeout = _options.VerifyTimeout });
        foreach (var command in task.Setup ?? [])
        {
            OnStatus?.Invoke($"setup: {command}");
            var setup = await checks.ExecuteAsync(command, ".", null, cancellationToken).ConfigureAwait(false);
            if (!setup.Succeeded)
            {
                return new AgentRun(task, [], ToolDefinitions(tools), _options.Think, AgentOutcome.SetupFailed, $"$ {command}\n{setup.Output}", 0, 0, 0, 0,
                    TimeSpan.Zero, TimeSpan.Zero, workFolder);
            }
        }

        CommitStart(workFolder);
        var run = await RunAsync(task, tools, cancellationToken).ConfigureAwait(false);
        if (run.Outcome == AgentOutcome.Error)
        {
            return run;
        }

        var (passed, output) = await VerifyAsync(task, checks, cancellationToken).ConfigureAwait(false);
        var outcome = passed ? AgentOutcome.Passed : run.Outcome == AgentOutcome.OutOfBudget ? AgentOutcome.OutOfBudget : AgentOutcome.Failed;
        return run with { Outcome = outcome, VerifyOutput = output };
    }

    /// <summary>
    /// Checks a task without a model: its verification must fail on the starting files (else the task tests nothing) and
    /// pass with its <see cref="AgentTask.Solution"/> (else it cannot be solved). Uses two folders under <paramref name="workFolder"/>.
    /// </summary>
    public async Task<(bool StartFails, bool SolutionPasses, string Output)> CheckTaskAsync(AgentTask task, string workFolder, CancellationToken cancellationToken = default)
    {
        var report = new System.Text.StringBuilder();
        var results = new bool[2];
        for (int pass = 0; pass < 2; pass++)
        {
            string folder = Path.Combine(workFolder, pass == 0 ? "start" : "solution");
            PrepareWorkspace(task, folder);
            if (pass == 1 && task.Solution is { } solution)
            {
                CopyTree(solution, folder, overwrite: true);
            }

            var checks = new CodingTools(folder, _options.Tools with { CommandTimeout = _options.VerifyTimeout });
            bool setupOk = true;
            foreach (var command in task.Setup ?? [])
            {
                OnStatus?.Invoke($"setup: {command}");
                var setup = await checks.ExecuteAsync(command, ".", null, cancellationToken).ConfigureAwait(false);
                if (!setup.Succeeded)
                {
                    report.Append($"[{(pass == 0 ? "start" : "solution")}] setup failed: $ {command}\n{setup.Output}\n");
                    setupOk = false;
                    break;
                }
            }

            var (passed, output) = setupOk ? await VerifyAsync(task, checks, cancellationToken).ConfigureAwait(false) : (false, "");
            results[pass] = pass == 0 ? setupOk && !passed : passed;
            report.Append($"[{(pass == 0 ? "start" : "solution")}] {(passed ? "verification passed" : "verification failed")}\n{output}");
        }

        return (results[0], task.Solution is not null && results[1], report.ToString());
    }

    private async Task<(bool Passed, string Output)> VerifyAsync(AgentTask task, CodingTools checks, CancellationToken cancellationToken)
    {
        if (task.VerifyFiles is { } hidden)
        {
            CopyTree(hidden, checks.Workspace, overwrite: true);
        }

        var output = new System.Text.StringBuilder();
        foreach (var command in task.Verify ?? [])
        {
            OnStatus?.Invoke($"verify: {command}");
            var result = await checks.ExecuteAsync(command, ".", null, cancellationToken).ConfigureAwait(false);
            output.Append($"$ {command}\n{(result.TimedOut ? "timed out" : $"exit code {result.ExitCode}")}\n{result.Output}\n");
            if (!result.Succeeded)
            {
                return (false, output.ToString());
            }
        }

        return (true, output.ToString());
    }

    /// <summary>Runs the agent on <paramref name="task"/>'s prompt with <paramref name="tools"/> as they are (no setup or verification).</summary>
    public async Task<AgentRun> RunAsync(AgentTask task, CodingTools tools, CancellationToken cancellationToken = default)
    {
        var registry = tools.Registry();
        var definitions = ToolDefinitions(tools);
        var messages = new List<ChatMessage> { new("system", _options.SystemPrompt ?? DefaultSystemPrompt), new("user", task.Prompt) };
        int rounds = 0, calls = 0, errors = 0, generated = 0;
        var modelTime = TimeSpan.Zero;
        var toolTime = TimeSpan.Zero;
        string? previousCalls = null;
        using var limit = CancellationTokenSource.CreateLinkedTokenSource(cancellationToken);
        limit.CancelAfter(_options.TimeLimit);
        AgentRun Result(AgentOutcome outcome, string note) =>
            new(task, messages, definitions, _options.Think, outcome, note, rounds, calls, errors, generated, modelTime, toolTime, tools.Workspace);

        try
        {
            while (true)
            {
                if (rounds >= _options.MaxRounds)
                {
                    return Result(AgentOutcome.OutOfBudget, $"stopped after {rounds} rounds");
                }

                var watch = Stopwatch.StartNew();
                ChatChunk? final = null;
                await foreach (var chunk in model.StreamAsync(new ChatRequest([.. messages], definitions, _options.Think, _options.Sampling), limit.Token).ConfigureAwait(false))
                {
                    if (!chunk.Delta.IsEmpty)
                    {
                        OnDelta?.Invoke(chunk.Delta);
                    }

                    if (chunk.Done)
                    {
                        final = chunk;
                    }
                }

                modelTime += watch.Elapsed;
                rounds++;
                var reply = final?.Message ?? throw new InvalidOperationException("The model ended without a final message.");
                generated += final.Stats?.GeneratedTokens ?? 0;
                messages.Add(reply);
                if (reply.ToolCalls is not { Count: > 0 })
                {
                    return Result(final.DoneReason == "length" ? AgentOutcome.OutOfBudget : AgentOutcome.Passed, "");
                }

                // The same calls as the round before, with nothing in between, cannot give anything new (small models
                // repeat a call instead of answering from its result): they are not run again, and the model is told so.
                string key = string.Join("\n", reply.ToolCalls.Select(c => c.Name + " " + c.Arguments.ToJsonString()));
                bool repeated = key == previousCalls;
                previousCalls = key;
                watch.Restart();
                IReadOnlyList<ToolResult> results = repeated
                    ? [.. reply.ToolCalls.Select(c => new ToolResult(c, "", false,
                        "Not run again: this is the same call as the one just before, and its result is above. Use that result to answer, or make a different call.",
                        TimeSpan.Zero))]
                    : await registry.InvokeAsync(reply.ToolCalls, limit.Token).ConfigureAwait(false);
                foreach (var result in results)
                {
                    calls++;
                    bool failed = !result.Succeeded || result.Content.StartsWith("Error:", StringComparison.Ordinal);
                    errors += failed ? 1 : 0;
                    messages.Add(result.Succeeded ? result.ToMessage() : new ChatMessage("tool", $"Error: {result.Error}", ToolName: result.Call.Name));
                    OnToolResult?.Invoke(result);
                }

                toolTime += watch.Elapsed;
            }
        }
        catch (OperationCanceledException) when (!cancellationToken.IsCancellationRequested)
        {
            return Result(AgentOutcome.OutOfBudget, $"stopped at the time limit ({_options.TimeLimit})");
        }
        catch (Exception ex) when (ex is HttpRequestException or InvalidOperationException or IOException)
        {
            return Result(AgentOutcome.Error, ex.Message);
        }
    }

    /// <summary>
    /// Fills <paramref name="workFolder"/> with the task's project: <see cref="AgentTask.Base"/>, then
    /// <see cref="AgentTask.Workspace"/> over it. Build output is not copied; a node_modules folder is linked, not copied
    /// (copied where links are not allowed).
    /// </summary>
    public static void PrepareWorkspace(AgentTask task, string workFolder)
    {
        if (Directory.Exists(workFolder) && Directory.EnumerateFileSystemEntries(workFolder).Any())
        {
            throw new IOException($"The work folder '{workFolder}' is not empty.");
        }

        Directory.CreateDirectory(workFolder);
        foreach (var source in new[] { task.Base, task.Workspace })
        {
            if (source is null)
            {
                continue;
            }

            CopyTree(source, workFolder, overwrite: true);
            string modules = Path.Combine(source, "node_modules"), target = Path.Combine(workFolder, "node_modules");
            if (Directory.Exists(modules) && !Directory.Exists(target))
            {
                try
                {
                    Directory.CreateSymbolicLink(target, modules);
                }
                catch (Exception ex) when (ex is IOException or UnauthorizedAccessException)
                {
                    CopyTree(modules, target, overwrite: false, skipBuildOutput: false);
                }
            }
        }
    }

    private static IReadOnlyList<ToolDefinition> ToolDefinitions(CodingTools tools) => [.. tools.All.Select(t => t.Definition)];

    private static readonly HashSet<string> BuildOutput = new(["bin", "obj", "node_modules", ".angular", "dist", ".vs", ".git"], StringComparer.OrdinalIgnoreCase);

    private static void CopyTree(string source, string target, bool overwrite, bool skipBuildOutput = true)
    {
        Directory.CreateDirectory(target);
        foreach (var file in Directory.EnumerateFiles(source))
        {
            File.Copy(file, Path.Combine(target, Path.GetFileName(file)), overwrite);
        }

        foreach (var folder in Directory.EnumerateDirectories(source))
        {
            if (!skipBuildOutput || !BuildOutput.Contains(Path.GetFileName(folder)))
            {
                CopyTree(folder, Path.Combine(target, Path.GetFileName(folder)), overwrite, skipBuildOutput);
            }
        }
    }

    // A git repository with the starting state committed, so the agent can review its changes with git diff (best effort:
    // skipped without git).
    private static void CommitStart(string folder)
    {
        if (Directory.Exists(Path.Combine(folder, ".git")))
        {
            return;
        }

        try
        {
            Git(folder, "init", "-q");
            File.AppendAllText(Path.Combine(folder, ".git", "info", "exclude"), "\nbin/\nobj/\nnode_modules\n.angular/\ndist/\n");
            Git(folder, "add", "-A");
            Git(folder, "-c", "user.name=agent", "-c", "user.email=agent@localhost", "-c", "commit.gpgsign=false", "commit", "-q", "-m", "start");
        }
        catch (Exception ex) when (ex is System.ComponentModel.Win32Exception or IOException or InvalidOperationException)
        {
        }
    }

    private static void Git(string folder, params string[] arguments)
    {
        var start = new ProcessStartInfo("git") { WorkingDirectory = folder, RedirectStandardOutput = true, RedirectStandardError = true, UseShellExecute = false };
        foreach (var argument in arguments)
        {
            start.ArgumentList.Add(argument);
        }

        using var process = Process.Start(start) ?? throw new InvalidOperationException("git did not start.");
        process.StandardOutput.ReadToEnd();
        process.StandardError.ReadToEnd();
        process.WaitForExit();
    }
}
