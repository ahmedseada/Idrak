// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Diagnostics;
using System.Text.Json.Nodes;

namespace Idrak.Text.Generation;

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
}

/// <summary>How the agent's loop ended (<see cref="CodingAgentRun.End"/>).</summary>
public enum CodingAgentEnd
{
    /// <summary>The model answered without a tool call: it says it is done.</summary>
    Answered,

    /// <summary>The agent used all its rounds or time, or its last reply was cut at the length limit.</summary>
    OutOfBudget,

    /// <summary>The model failed (for example the server was unreachable).</summary>
    Error,
}

/// <summary>One run of the agent's loop on a prompt: the conversation and its figures (no task, no verification).</summary>
/// <param name="Messages">System, user, assistant and tool messages.</param>
/// <param name="Tools">The tools the model was offered.</param>
/// <param name="Think">The reasoning mode used.</param>
/// <param name="End">How the loop ended.</param>
/// <param name="Note">Why it stopped, when it did not answer (the budget, the error); empty when it answered.</param>
/// <param name="Rounds">Model replies.</param>
/// <param name="ToolCalls">Tool calls made.</param>
/// <param name="ToolErrors">Tool calls that returned an error.</param>
/// <param name="GeneratedTokens">Tokens the model generated (when it reports them).</param>
/// <param name="ModelTime">Time spent waiting for the model.</param>
/// <param name="ToolTime">Time spent running tools.</param>
/// <param name="Workspace">The folder the agent worked in.</param>
public sealed record CodingAgentRun(IReadOnlyList<ChatMessage> Messages, IReadOnlyList<ToolDefinition> Tools, bool? Think, CodingAgentEnd End, string Note,
    int Rounds, int ToolCalls, int ToolErrors, int GeneratedTokens, TimeSpan ModelTime, TimeSpan ToolTime, string Workspace)
{
    /// <summary>The run as one JSON line: the conversation in the chat format with the figures as extra fields.</summary>
    public JsonObject ToJson()
    {
        var json = ChatJson.Transcript(Messages, Tools, Think);
        AddFigures(json);
        return json;
    }

    /// <summary>Adds the figures (rounds, tool calls and errors, tokens, seconds) to <paramref name="json"/>, after what it holds.</summary>
    public void AddFigures(JsonObject json)
    {
        ArgumentNullException.ThrowIfNull(json);
        json["rounds"] = Rounds;
        json["tool_calls"] = ToolCalls;
        json["tool_errors"] = ToolErrors;
        json["generated_tokens"] = GeneratedTokens;
        json["model_seconds"] = Math.Round(ModelTime.TotalSeconds, 2);
        json["tool_seconds"] = Math.Round(ToolTime.TotalSeconds, 2);
    }
}

/// <summary>The tool calls of one assistant reply, run as the agent runs them (<see cref="CodingAgent.RunToolCallsAsync"/>).</summary>
/// <param name="Results">Each call's result, in call order.</param>
/// <param name="Messages">The tool messages sent back to the model, in call order (an error as "Error: …").</param>
/// <param name="Errors">The calls that failed or returned an error.</param>
/// <param name="Key">The calls as one text, to compare with the next reply's (<c>previousCalls</c>).</param>
public sealed record CodingToolTurn(IReadOnlyList<ToolResult> Results, IReadOnlyList<ChatMessage> Messages, int Errors, string Key);

/// <summary>
/// A coding agent: a chat model with <see cref="CodingTools"/> in a workspace, looping on a prompt until it answers
/// without a tool call (<see cref="RunAsync"/>). The coding tasks (a project, setup, hidden tests, verification) are the
/// coding plug-in's (Idrak.Reinforcement.Plugins), which runs this loop to evaluate a model on a task suite.
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

    /// <summary>How the agent works.</summary>
    public AgentOptions Options => _options;

    /// <summary>Called with each piece of the model's replies as they stream.</summary>
    public Action<ChatDelta>? OnDelta { get; init; }

    /// <summary>Called after each tool call.</summary>
    public Action<ToolResult>? OnToolResult { get; init; }

    /// <summary>
    /// Runs the agent on <paramref name="prompt"/> with <paramref name="tools"/> as they are: the model replies, its tool
    /// calls run (a call repeated right after itself is not run again), until it answers without a call, uses its rounds
    /// (<see cref="AgentOptions.MaxRounds"/>) or its time (<see cref="AgentOptions.TimeLimit"/>), or fails.
    /// </summary>
    public async Task<CodingAgentRun> RunAsync(string prompt, CodingTools tools, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(prompt);
        ArgumentNullException.ThrowIfNull(tools);
        var registry = tools.Registry();
        var definitions = ToolDefinitions(tools);
        var messages = new List<ChatMessage> { new("system", _options.SystemPrompt ?? DefaultSystemPrompt), new("user", prompt) };
        int rounds = 0, calls = 0, errors = 0, generated = 0;
        var modelTime = TimeSpan.Zero;
        var toolTime = TimeSpan.Zero;
        string? previousCalls = null;
        using var limit = CancellationTokenSource.CreateLinkedTokenSource(cancellationToken);
        limit.CancelAfter(_options.TimeLimit);
        CodingAgentRun Result(CodingAgentEnd end, string note) =>
            new(messages, definitions, _options.Think, end, note, rounds, calls, errors, generated, modelTime, toolTime, tools.Workspace);

        try
        {
            while (true)
            {
                if (rounds >= _options.MaxRounds)
                {
                    return Result(CodingAgentEnd.OutOfBudget, $"stopped after {rounds} rounds");
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
                    return Result(final.DoneReason == "length" ? CodingAgentEnd.OutOfBudget : CodingAgentEnd.Answered, "");
                }

                watch.Restart();
                var turn = await RunToolCallsAsync(registry, reply, previousCalls, limit.Token).ConfigureAwait(false);
                previousCalls = turn.Key;
                calls += turn.Results.Count;
                errors += turn.Errors;
                messages.AddRange(turn.Messages);
                foreach (var result in turn.Results)
                {
                    OnToolResult?.Invoke(result);
                }

                toolTime += watch.Elapsed;
            }
        }
        catch (OperationCanceledException) when (!cancellationToken.IsCancellationRequested)
        {
            return Result(CodingAgentEnd.OutOfBudget, $"stopped at the time limit ({_options.TimeLimit})");
        }
        catch (Exception ex) when (ex is HttpRequestException or InvalidOperationException or IOException)
        {
            return Result(CodingAgentEnd.Error, ex.Message);
        }
    }

    /// <summary>
    /// Runs the tool calls of <paramref name="reply"/> through <paramref name="registry"/> as the agent's loop does: one
    /// after another, the same calls as the reply before (<paramref name="previousCalls"/>, the last turn's
    /// <see cref="CodingToolTurn.Key"/>) not run again but answered with a note to use that result (small models repeat a
    /// call instead of answering from it), a failure sent back as "Error: …". Any environment that steps a coding agent
    /// turn by turn uses this, so its conversations are the loop's.
    /// </summary>
    public static async Task<CodingToolTurn> RunToolCallsAsync(IToolRegistry registry, ChatMessage reply, string? previousCalls, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(registry);
        ArgumentNullException.ThrowIfNull(reply);
        var toolCalls = reply.ToolCalls ?? [];
        // The same calls as the round before, with nothing in between, cannot give anything new (small models repeat a
        // call instead of answering from its result): they are not run again, and the model is told so.
        string key = string.Join("\n", toolCalls.Select(c => c.Name + " " + c.Arguments.ToJsonString()));
        bool repeated = key == previousCalls;
        IReadOnlyList<ToolResult> results = repeated
            ? [.. toolCalls.Select(c => new ToolResult(c, "", false,
                "Not run again: this is the same call as the one just before, and its result is above. Use that result to answer, or make a different call.",
                TimeSpan.Zero))]
            : await registry.InvokeAsync(toolCalls, cancellationToken).ConfigureAwait(false);
        var messages = new ChatMessage[results.Count];
        int errors = 0;
        for (int i = 0; i < results.Count; i++)
        {
            var result = results[i];
            bool failed = !result.Succeeded || result.Content.StartsWith("Error:", StringComparison.Ordinal);
            errors += failed ? 1 : 0;
            messages[i] = result.Succeeded ? result.ToMessage() : new ChatMessage("tool", $"Error: {result.Error}", ToolName: result.Call.Name);
        }

        return new CodingToolTurn(results, messages, errors, key);
    }

    private static IReadOnlyList<ToolDefinition> ToolDefinitions(CodingTools tools) => [.. tools.All.Select(t => t.Definition)];
}
