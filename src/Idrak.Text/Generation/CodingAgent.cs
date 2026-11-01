// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Diagnostics;
using System.Text;
using System.Text.Json.Nodes;
using Idrak.Text.Abstractions;

namespace Idrak.Text.Generation;

/// <summary>How <see cref="CodingAgent"/> works.</summary>
public sealed record AgentOptions
{
    /// <summary>
    /// The system prompt; null (the default) for <see cref="CodingAgent.SystemPrompt"/> of the tools' coding languages
    /// (<see cref="CodingTools.Languages"/>, chosen by <see cref="CodingToolOptions.Languages"/> in <see cref="Tools"/>).
    /// </summary>
    public string? SystemPrompt { get; init; }

    /// <summary>Reasoning mode for every request.</summary>
    public bool? Think { get; init; }

    /// <summary>Sampling and length options for every request.</summary>
    public GenerationOptions? Sampling { get; init; }

    /// <summary>Model replies allowed per task.</summary>
    public int MaxRounds { get; init; } = 40;

    /// <summary>Wall-clock limit per task (model and tools; not setup or verification).</summary>
    public TimeSpan TimeLimit { get; init; } = TimeSpan.FromMinutes(20);

    /// <summary>What the tools may do, and the coding languages they and the prompt work with (<see cref="CodingToolOptions.Languages"/>).</summary>
    public CodingToolOptions Tools { get; init; } = new();

    /// <summary>
    /// Tools offered beside the coding tools (an editor's code search, a project's own lookups), after them, in order: none
    /// by default. A name a coding tool already has, or one given twice, is refused when the run starts (no tool shadows
    /// another silently).
    /// </summary>
    public IReadOnlyList<ITool> ExtraTools { get; init; } = [];

    /// <summary>
    /// Whether tool calls a reply writes as text instead of in its template's format (markdown fences, JSON in the
    /// prose) are read after the turn and run (<see cref="ToolCallText.CallsInText"/>: only values naming an offered tool):
    /// true by default; false reads calls only as the template writes them (a reply of calls as text then gets a
    /// correction, <see cref="Nudges"/>).
    /// </summary>
    public bool CallsInText { get; init; } = true;

    /// <summary>
    /// Corrections in a row the agent sends when a reply has no call it could read but looks like calls
    /// (<see cref="ToolCallText.LooksLikeCall"/>: a JSON fence, <c>"name"</c> with an offered tool's name): the model is
    /// told how its template writes a call, with an example (<see cref="CodingAgent.Correction"/>), and the run goes on.
    /// One more such reply ends the run with <see cref="CodingAgentEnd.CallsNotRead"/>; a reply whose calls are read
    /// starts the count again. 2 by default; 0 sends none (the run ends on the first such reply).
    /// </summary>
    public int Nudges { get; init; } = 2;

    /// <summary>
    /// The chat template the model's prompts are rendered with, for reading calls written as text (its arguments key)
    /// and for the corrections' example (its tool-call format); null (the default): the model's own when it is a
    /// <see cref="ChatGenerator"/>, else none (the correction's example is a generic one).
    /// </summary>
    public IChatTemplate? Template { get; init; }
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

    /// <summary>
    /// The model kept writing calls the agent could not read (in a shape its template's format does not read, naming no
    /// offered tool, or JSON that does not parse) after every correction (<see cref="AgentOptions.Nudges"/>): nothing
    /// from its last reply was run; the note says so.
    /// </summary>
    CallsNotRead,
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
    /// <summary>Corrections sent because a reply's calls could not be read (<see cref="AgentOptions.Nudges"/>); each is a user message in <see cref="Messages"/>.</summary>
    public int Nudges { get; init; }

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
        if (Nudges > 0)
        {
            json["nudges"] = Nudges;
        }
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
    // The prompt's language-free parts; the line on checking the work lists each language's way (PromptLines).
    private const string PromptHead = "You are a coding agent working in a software project on the user's machine. You have tools to list, read and\n"
                                      + "search the project's files, edit and write files, and run commands such as builds and tests.\n\n"
                                      + "How to work:\n"
                                      + "- Understand before changing: find the relevant files (list_files, search) and read them before editing.\n"
                                      + "- Read a file with read_file before you edit it; never guess its text (edit_file refuses a file not read).\n"
                                      + "- Make focused changes with edit_file; copy old_text exactly as the file has it (read_file shows line numbers\n"
                                      + "  before a \"| \" separator: they are not part of the file). Create new files with write_file.\n"
                                      + "- Follow the project's conventions: its naming, formatting, file layout, frameworks and library versions.\n";

    private const string PromptTail = "\n- Do not weaken or delete tests to make them pass, unless the user asks for it.\n"
                                      + "- Do only what was asked; mention anything else worth doing instead of doing it.\n"
                                      + "- When the task is done, answer without a tool call: say briefly what you changed and how you checked it.";

    // The prompt's lines are cut at this width, continued with two spaces.
    private const int PromptWidth = 110;

    /// <summary>
    /// The instructions the agent gets by default with every registered coding language (<see cref="CodingLanguages.All"/>):
    /// <see cref="SystemPrompt"/> of <see cref="CodingLanguages.Registered"/>.
    /// </summary>
    public static string DefaultSystemPrompt => SystemPrompt(CodingLanguages.Registered);

    /// <summary>
    /// The instructions the agent gets by default with <paramref name="languages"/>: how to work, and in the line on checking
    /// the work each language's way to build and test (<see cref="ICodingLanguage.PromptLines"/>, in order); with none, a
    /// prompt that names no build or test command. Lines end with '\n' on every platform.
    /// </summary>
    public static string SystemPrompt(IReadOnlyList<ICodingLanguage> languages)
    {
        ArgumentNullException.ThrowIfNull(languages);
        var ways = CodingLanguages.Merge([.. languages.Select(l => l.PromptLines)]);
        var line = new PromptLine(PromptWidth);
        foreach (var word in "- Check your work: build and run the tests after changing code".Split(' '))
        {
            line.Put(ways.Count == 0 && word == "code" ? "code." : word);
        }

        for (int i = 0; i < ways.Count; i++)
        {
            // A way stays whole on a line when it fits; else it breaks after its "Name:", then between words.
            string way = (i == 0 ? "(" : "") + ways[i] + (i == ways.Count - 1 ? ")." : ";");
            int label = way.IndexOf(": ", StringComparison.Ordinal);
            if (line.Fits(way) || label < 0)
            {
                line.PutWords(way);
            }
            else
            {
                line.PutWords(way[..(label + 1)]);
                line.PutWords(way[(label + 2)..]);
            }
        }

        foreach (var word in "Read the errors, fix them and run the checks again until they pass.".Split(' '))
        {
            line.Put(word);
        }

        return PromptHead + line + PromptTail;
    }

    // A bullet line cut at a width: a piece goes on the line when it fits, else it starts the next one (two spaces in).
    private sealed class PromptLine(int width)
    {
        private const string Indent = "  ";
        private readonly StringBuilder _text = new();
        private int _column;

        public bool Fits(string piece) => _column == 0 || _column + 1 + piece.Length <= width;

        public void Put(string piece)
        {
            if (_column == 0)
            {
                _text.Append(piece);
                _column = piece.Length;
            }
            else if (Fits(piece))
            {
                _text.Append(' ').Append(piece);
                _column += 1 + piece.Length;
            }
            else
            {
                _text.Append('\n').Append(Indent).Append(piece);
                _column = Indent.Length + piece.Length;
            }
        }

        // A piece kept whole when it fits on this line or the next, else word by word.
        public void PutWords(string piece)
        {
            if (Fits(piece) || Indent.Length + piece.Length <= width)
            {
                Put(piece);
                return;
            }

            foreach (var word in piece.Split(' '))
            {
                Put(word);
            }
        }

        public override string ToString() => _text.ToString();
    }

    private readonly AgentOptions _options = options ?? new AgentOptions();
    private readonly System.Collections.Concurrent.ConcurrentQueue<string> _steering = new();

    /// <summary>How the agent works.</summary>
    public AgentOptions Options => _options;

    /// <summary>Called with each piece of the model's replies as they stream.</summary>
    public Action<ChatDelta>? OnDelta { get; init; }

    /// <summary>Called after each tool call.</summary>
    public Action<ToolResult>? OnToolResult { get; init; }

    /// <summary>The steering notes given (<see cref="Steer"/>) and not yet read by the model.</summary>
    public int PendingNotes => _steering.Count;

    /// <summary>
    /// Gives the running agent a note from the user (from any thread): before its next request to the model, every note
    /// given since the last one is added to the conversation as a user message (<c>Note from the user: …</c>), so the
    /// model reads it before its next step and it stays in the conversation from then on. A note given while no run is
    /// going waits for the next run's first request.
    /// </summary>
    /// <exception cref="ArgumentException">The note is empty or white space.</exception>
    public void Steer(string note)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(note);
        _steering.Enqueue(note.Trim());
    }

    /// <summary>
    /// Runs the agent on <paramref name="prompt"/> with <paramref name="tools"/> as they are: the model replies, its tool
    /// calls run (a call repeated right after itself is not run again), until it answers without a call, uses its rounds
    /// (<see cref="AgentOptions.MaxRounds"/>) or its time (<see cref="AgentOptions.TimeLimit"/>), or fails. Calls a reply
    /// writes as text are read after the turn (<see cref="AgentOptions.CallsInText"/>); a reply that looks like calls but
    /// has none that can be read gets a correction and the run goes on, at most <see cref="AgentOptions.Nudges"/> in a row
    /// (then it ends with <see cref="CodingAgentEnd.CallsNotRead"/>).
    /// </summary>
    public Task<CodingAgentRun> RunAsync(string prompt, CodingTools tools, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(prompt);
        ArgumentNullException.ThrowIfNull(tools);
        tools.ForgetReads();                                    // a new task reads a file before it edits it (CodingToolOptions.ReadBeforeEdit)
        return LoopAsync([new("system", _options.SystemPrompt ?? SystemPrompt(tools.Languages)), new("user", prompt)], tools, cancellationToken);
    }

    /// <summary>
    /// Continues the conversation of <paramref name="earlier"/> (a run of this agent or another, on the same workspace):
    /// its messages are kept as they are (the system prompt, the tasks, the replies and the tool results), and
    /// <paramref name="prompt"/> follows as the next user message; then the loop runs as <see cref="RunAsync"/>'s, with its
    /// own rounds and time. The returned run holds the whole conversation, so it can be continued again. An earlier run
    /// whose messages do not start with a system message gets this agent's system prompt first.
    /// </summary>
    public Task<CodingAgentRun> ContinueAsync(CodingAgentRun earlier, string prompt, CodingTools tools, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(earlier);
        ArgumentNullException.ThrowIfNull(prompt);
        ArgumentNullException.ThrowIfNull(tools);
        var messages = new List<ChatMessage>(earlier.Messages.Count + 2);
        if (earlier.Messages.Count == 0 || earlier.Messages[0].Role != "system")
        {
            messages.Add(new("system", _options.SystemPrompt ?? SystemPrompt(tools.Languages)));
        }

        messages.AddRange(earlier.Messages);
        messages.Add(new("user", prompt));
        return LoopAsync(messages, tools, cancellationToken);
    }

    private async Task<CodingAgentRun> LoopAsync(List<ChatMessage> messages, CodingTools tools, CancellationToken cancellationToken)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(_options.Nudges, nameof(AgentOptions.Nudges));
        var registry = Registry(tools, _options.ExtraTools);
        var definitions = ToolDefinitions(tools, _options.ExtraTools);
        var callContext = CallContext(_options.Template ?? (model as ChatGenerator)?.Template, definitions);
        int rounds = 0, calls = 0, errors = 0, generated = 0, nudges = 0, inRow = 0;
        var modelTime = TimeSpan.Zero;
        var toolTime = TimeSpan.Zero;
        string? previousCalls = null;
        using var limit = CancellationTokenSource.CreateLinkedTokenSource(cancellationToken);
        limit.CancelAfter(_options.TimeLimit);
        CodingAgentRun Result(CodingAgentEnd end, string note) =>
            new(messages, definitions, _options.Think, end, note, rounds, calls, errors, generated, modelTime, toolTime, tools.Workspace) { Nudges = nudges };

        try
        {
            while (true)
            {
                if (rounds >= _options.MaxRounds)
                {
                    return Result(CodingAgentEnd.OutOfBudget, $"stopped after {rounds} rounds");
                }

                while (_steering.TryDequeue(out var note))
                {
                    messages.Add(new ChatMessage("user", "Note from the user: " + note));
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
                if (reply.ToolCalls is not { Count: > 0 } && _options.CallsInText)
                {
                    reply = ReadCallsInText(reply, callContext);
                }

                messages.Add(reply);
                if (reply.ToolCalls is not { Count: > 0 })
                {
                    if (final.DoneReason == "length")
                    {
                        return Result(CodingAgentEnd.OutOfBudget, "");
                    }

                    if (Correction(reply, callContext) is not { } correction)
                    {
                        return Result(CodingAgentEnd.Answered, "");
                    }

                    if (inRow >= _options.Nudges)
                    {
                        return Result(CodingAgentEnd.CallsNotRead, inRow == 0
                            ? "the model wrote tool calls that could not be read (no correction allowed); nothing from its last reply was run"
                            : $"the model wrote tool calls that could not be read, after {inRow} correction{(inRow == 1 ? "" : "s")}; nothing from its last reply was run");
                    }

                    messages.Add(new ChatMessage("user", correction));
                    nudges++;
                    inRow++;
                    continue;
                }

                inRow = 0;

                watch.Restart();
                // Each call's result is told as that call ends (a later call may wait for an approval, or the turn be
                // cancelled part way), never only when the whole turn has run.
                var turn = await RunToolCallsAsync(registry, reply, previousCalls, limit.Token, OnToolResult).ConfigureAwait(false);
                previousCalls = turn.Key;
                calls += turn.Results.Count;
                errors += turn.Errors;
                messages.AddRange(turn.Messages);

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
    /// <remarks>
    /// <paramref name="onResult"/> is told each call's result as soon as that call ends, in call order (a registry that
    /// runs a reply's calls together, <see cref="ToolRegistry.Parallel"/>, tells them when all have ended), so a call that
    /// waits (an approval) or a turn cancelled part way (the time limit, a stop) never hides the calls already done.
    /// </remarks>
    public static async Task<CodingToolTurn> RunToolCallsAsync(IToolRegistry registry, ChatMessage reply, string? previousCalls, CancellationToken cancellationToken = default,
        Action<ToolResult>? onResult = null)
    {
        ArgumentNullException.ThrowIfNull(registry);
        ArgumentNullException.ThrowIfNull(reply);
        var toolCalls = reply.ToolCalls ?? [];
        // The same calls as the round before, with nothing in between, cannot give anything new (small models repeat a
        // call instead of answering from its result): they are not run again, and the model is told so.
        string key = string.Join("\n", toolCalls.Select(c => c.Name + " " + c.Arguments.ToJsonString()));
        bool repeated = key == previousCalls;
        IReadOnlyList<ToolResult> results;
        if (repeated || onResult is null || registry is ToolRegistry { Parallel: true })
        {
            results = repeated
                ? [.. toolCalls.Select(c => new ToolResult(c, "", false,
                    "Not run again: this is the same call as the one just before, and its result is above. Use that result to answer, or make a different call.",
                    TimeSpan.Zero))]
                : await registry.InvokeAsync(toolCalls, cancellationToken).ConfigureAwait(false);
            if (onResult is not null)
            {
                foreach (var result in results)
                {
                    onResult(result);
                }
            }
        }
        else
        {
            // One call after another, as the registry runs a reply's calls, each told as it ends.
            var done = new List<ToolResult>(toolCalls.Count);
            foreach (var call in toolCalls)
            {
                var result = await registry.InvokeAsync(call, cancellationToken).ConfigureAwait(false);
                done.Add(result);
                onResult(result);
            }

            results = done;
        }
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

    /// <summary>
    /// What the agent reads calls written as text with (<see cref="ReadCallsInText"/>, <see cref="Correction"/>):
    /// <paramref name="template"/> (null when it is not known: the correction's example is then a generic one) and the
    /// request's <paramref name="tools"/>, whose names are the only ones a call may have. Made once per run.
    /// </summary>
    public static ToolCallContext CallContext(IChatTemplate? template, IReadOnlyList<ToolDefinition> tools)
    {
        ArgumentNullException.ThrowIfNull(tools);
        return new ToolCallContext(template ?? UnknownTemplate.Instance, tools, tools.Select(t => t.Name).ToHashSet(StringComparer.Ordinal));
    }

    /// <summary>
    /// <paramref name="reply"/> with the calls its text holds (<see cref="ToolCallText.CallsInText"/>: fenced or bare JSON
    /// naming an offered tool) when its template's parser found none; the reply itself otherwise. Its text is kept as the
    /// model wrote it. Used after each turn by the agent's loop and by any loop that steps it (the CLI's agent).
    /// </summary>
    public static ChatMessage ReadCallsInText(ChatMessage reply, ToolCallContext context)
    {
        ArgumentNullException.ThrowIfNull(reply);
        ArgumentNullException.ThrowIfNull(context);
        return reply.ToolCalls is not { Count: > 0 } && ToolCallText.CallsInText(reply.Content, context) is { } calls ? reply with { ToolCalls = calls } : reply;
    }

    /// <summary>
    /// The correction to send after <paramref name="reply"/> when it has no call but its text looks like calls
    /// (<see cref="ToolCallText.LooksLikeCall"/>): that nothing in it was run, how the template writes a call with an
    /// example (in its markers for the "json" format, the bare JSON for a template whose answer is the call, else a
    /// generic line), the offered tools, and how to say the task is done. Null when the reply has calls or looks like
    /// none.
    /// </summary>
    public static string? Correction(ChatMessage reply, ToolCallContext context)
    {
        ArgumentNullException.ThrowIfNull(reply);
        ArgumentNullException.ThrowIfNull(context);
        if (reply.ToolCalls is { Count: > 0 } || !ToolCallText.LooksLikeCall(reply.Content, context))
        {
            return null;
        }

        var tools = context.Tools;
        var example = tools.FirstOrDefault(t => t.Name == "write_file") ?? tools.FirstOrDefault();
        string call = new JsonObject
        {
            ["name"] = example?.Name ?? "tool_name",
            [context.Template.ToolCalls.ArgumentsKey] = ExampleArguments(example),
        }.ToJsonString();
        var format = context.Template.ToolCalls;
        string how = context.Template is UnknownTemplate || context.Template.ToolCallFormatName != ToolCallFormats.Json
            ? "To call a tool, write the call in your chat template's own tool-call format, as you were trained to, never inside a markdown code block. "
              + "As JSON, a call reads:\n" + call
            : format.Open.Length == 0
                ? "To call a tool, write the call itself as your whole reply, with nothing before or after it and not in a code block:\n"
                  + (format.List ? "[" + call + "]" : call)
                : $"To call a tool, write the call itself between {format.Open} and {format.Close}, not in a code block and not as a description, one block per call:\n"
                  + format.Open + "\n" + (format.List ? "[" + call + "]" : call) + "\n" + format.Close;
        return "Your last reply has no tool call that could be read, so nothing in it was run and no file was changed. " + how
               + "\nThe tools: " + string.Join(", ", tools.Select(t => t.Name)) + ".\nIf the task is done, answer in words, without any JSON or tool call.";
    }

    // The example call's arguments: the tool's required parameters (all of them when it lists none), each "...".
    private static JsonObject ExampleArguments(ToolDefinition? tool)
    {
        var arguments = new JsonObject();
        if (tool?.Parameters is not JsonObject schema || schema["properties"] is not JsonObject properties)
        {
            return arguments;
        }

        var required = schema["required"] is JsonArray list ? list.Select(n => n?.ToString()).ToHashSet(StringComparer.Ordinal) : null;
        foreach (var (name, _) in properties)
        {
            if (required is null || required.Count == 0 || required.Contains(name))
            {
                arguments[name] = "...";
            }
        }

        return arguments;
    }

    // A template that is not known: calls are read with the common "arguments" key, and the correction's example is generic.
    private sealed class UnknownTemplate : ChatTemplate
    {
        public static readonly UnknownTemplate Instance = new();

        public override IReadOnlyList<string> StopSequences => [];

        public override string ToolCallFormatName => "";

        public override void Render(IReadOnlyList<ChatMessage> messages, IReadOnlyList<ToolDefinition> tools, bool? think, StringBuilder prompt) =>
            throw new NotSupportedException("The agent's template is not known; it renders nothing.");
    }

    private static IReadOnlyList<ToolDefinition> ToolDefinitions(CodingTools tools, IReadOnlyList<ITool> extra) =>
        [.. tools.All.Select(t => t.Definition), .. extra.Select(t => t.Definition)];

    // The coding tools, then the extra ones; a name given twice is refused by the builder, naming the tool.
    private static IToolRegistry Registry(CodingTools tools, IReadOnlyList<ITool> extra)
    {
        if (extra.Count == 0)
        {
            return tools.Registry();
        }

        var builder = ToolRegistry.Create().Add(tools.All);
        foreach (var tool in extra)
        {
            ArgumentNullException.ThrowIfNull(tool, nameof(AgentOptions.ExtraTools));
            if (tools.All.Any(t => t.Definition.Name == tool.Definition.Name))
            {
                throw new ArgumentException($"The extra tool '{tool.Definition.Name}' has the name of a coding tool; give it another name.", nameof(AgentOptions.ExtraTools));
            }

            builder.Add(tool);
        }

        return builder.Build();
    }
}
