// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Diagnostics.CodeAnalysis;
using System.Text.Json.Nodes;

namespace Idrak.Generation;

/// <summary>The outcome of <see cref="Conversation.SendAsync"/>.</summary>
/// <param name="Message">The assistant's final message (also added to <see cref="Conversation.Messages"/>).</param>
/// <param name="DoneReason">"stop" or "length" for the final model reply.</param>
/// <param name="Stats">Generation statistics of the final model reply, when the model reports them.</param>
/// <param name="ToolResults">Every tool call run while answering, in order.</param>
/// <param name="Rounds">Model replies produced (1 without tool calls).</param>
/// <param name="ToolLimitReached">True when the last reply still asked for tools but <see cref="ConversationBuilder.MaxToolRounds"/> rounds were used.</param>
public sealed record ConversationReply(ChatMessage Message, string? DoneReason, GenerationStats? Stats, IReadOnlyList<ToolResult> ToolResults,
    int Rounds, bool ToolLimitReached);

/// <summary>
/// A chat that keeps its history. Each <see cref="SendAsync"/> adds the user message, asks the model (with the
/// conversation's think mode, options and tools) and adds the reply. When tools are registered, tool calls are run,
/// their results sent back and the model asked again, for at most <see cref="ConversationBuilder.MaxToolRounds"/> rounds.
/// Without tools, tool calls are returned in the message for your code to handle, as with <see cref="ChatGenerator"/>.
/// </summary>
public sealed class Conversation
{
    private readonly IChatModel _model;
    private readonly bool? _think;
    private readonly GenerationOptions? _options;
    private readonly int _maxToolRounds;

    internal Conversation(IChatModel model, List<ChatMessage> messages, bool? think, GenerationOptions? options, IToolRegistry? tools, int maxToolRounds)
    {
        _model = model;
        Messages = messages;
        _think = think;
        _options = options;
        Tools = tools;
        _maxToolRounds = maxToolRounds;
    }

    /// <summary>Starts a conversation with <paramref name="model"/> (a <see cref="ChatGenerator"/>, an engine model, or a fake for tests).</summary>
    public static ConversationBuilder For(IChatModel model) => new(model);

    /// <summary>The history: system, user, assistant and tool messages. You may read and edit it between calls.</summary>
    public List<ChatMessage> Messages { get; }

    /// <summary>The tools the model may call, or null.</summary>
    public IToolRegistry? Tools { get; }

    /// <summary>Adds <paramref name="user"/> as a user message and returns the assistant's answer (running tools as configured).</summary>
    public async Task<ConversationReply> SendAsync(string user, CancellationToken cancellationToken = default)
    {
        Messages.Add(new ChatMessage("user", user));
        return await ContinueAsync(cancellationToken).ConfigureAwait(false);
    }

    /// <summary>Asks the model to answer the history as it is (for example after you added messages yourself).</summary>
    public async Task<ConversationReply> ContinueAsync(CancellationToken cancellationToken = default)
    {
        ConversationReply? reply = null;
        await foreach (var _ in RunAsync(r => reply = r, cancellationToken).ConfigureAwait(false))
        {
        }

        return reply!;
    }

    /// <summary>
    /// Adds <paramref name="user"/> and streams the answer's pieces (across tool rounds; tool results go into
    /// <see cref="Messages"/>). When the stream ends, <see cref="LastReply"/> holds the outcome.
    /// </summary>
    public IAsyncEnumerable<ChatDelta> StreamAsync(string user, CancellationToken cancellationToken = default)
    {
        Messages.Add(new ChatMessage("user", user));
        return RunAsync(r => LastReply = r, cancellationToken);
    }

    /// <summary>Streams the answer to the history as it is; when the stream ends, <see cref="LastReply"/> holds the outcome.</summary>
    public IAsyncEnumerable<ChatDelta> StreamContinueAsync(CancellationToken cancellationToken = default) =>
        RunAsync(r => LastReply = r, cancellationToken);

    /// <summary>The outcome of the most recent streamed answer.</summary>
    public ConversationReply? LastReply { get; private set; }

    private async IAsyncEnumerable<ChatDelta> RunAsync(Action<ConversationReply>? done,
        [System.Runtime.CompilerServices.EnumeratorCancellation] CancellationToken cancellationToken)
    {
        var results = new List<ToolResult>();
        int rounds = 0;
        while (true)
        {
            var request = new ChatRequest([.. Messages], Tools?.Definitions, _think, _options);
            ChatChunk? final = null;
            await foreach (var chunk in _model.StreamAsync(request, cancellationToken).ConfigureAwait(false))
            {
                if (!chunk.Delta.IsEmpty)
                {
                    yield return chunk.Delta;
                }

                if (chunk.Done)
                {
                    final = chunk;
                }
            }

            if (final?.Message is not { } message)
            {
                throw new InvalidOperationException("The model ended without a final message.");
            }

            Messages.Add(message);
            rounds++;
            bool wantsTools = message.ToolCalls is { Count: > 0 };
            if (!wantsTools || Tools is null || rounds > _maxToolRounds)
            {
                done?.Invoke(new ConversationReply(message, final.DoneReason, final.Stats, results, rounds, wantsTools && Tools is not null));
                yield break;
            }

            foreach (var result in await Tools.InvokeAsync(message.ToolCalls!, cancellationToken).ConfigureAwait(false))
            {
                results.Add(result);
                Messages.Add(result.ToMessage());
            }
        }
    }
}

/// <summary>Configures a <see cref="Conversation"/>. Everything is optional except <see cref="MaxToolRounds"/> once tools are added.</summary>
public sealed class ConversationBuilder
{
    private readonly IChatModel _model;
    private readonly List<ChatMessage> _messages = [];
    private ToolRegistryBuilder? _toolBuilder;
    private IToolRegistry? _tools;
    private bool? _think;
    private GenerationOptions? _options;
    private int? _maxToolRounds;

    internal ConversationBuilder(IChatModel model) => _model = model ?? throw new ArgumentNullException(nameof(model));

    /// <summary>Adds a system message (the model's instructions).</summary>
    public ConversationBuilder System(string text)
    {
        _messages.Add(new ChatMessage("system", text));
        return this;
    }

    /// <summary>Starts from existing messages (for example a stored history).</summary>
    public ConversationBuilder Messages(IEnumerable<ChatMessage> history)
    {
        _messages.AddRange(history);
        return this;
    }

    /// <summary><see cref="ChatRequest.Think"/> for every request.</summary>
    public ConversationBuilder Think(bool? think)
    {
        _think = think;
        return this;
    }

    /// <summary><see cref="ChatRequest.Options"/> for every request.</summary>
    public ConversationBuilder Options(GenerationOptions options)
    {
        _options = options;
        return this;
    }

    /// <summary>Uses a ready-made registry (cannot be combined with <see cref="Tool(Abstraction.Generation.Tool)"/>).</summary>
    public ConversationBuilder Tools(IToolRegistry tools)
    {
        _tools = tools;
        return this;
    }

    /// <summary>Adds one tool (collected into a registry without rules).</summary>
    public ConversationBuilder Tool(Tool tool)
    {
        (_toolBuilder ??= ToolRegistry.Create()).Add(tool);
        return this;
    }

    /// <summary>Adds a tool with an explicit schema (<see cref="Abstraction.Generation.Tool.Create"/>).</summary>
    public ConversationBuilder Tool(string name, string description, JsonNode parameters, Func<JsonObject, CancellationToken, Task<string>> invoke) =>
        Tool(Abstraction.Generation.Tool.Create(name, description, parameters, invoke));

    /// <summary>Adds a tool from a delegate (<see cref="Abstraction.Generation.Tool.FromDelegate"/>).</summary>
    [RequiresUnreferencedCode("See Tool.FromDelegate.")]
    [RequiresDynamicCode("See Tool.FromDelegate.")]
    public ConversationBuilder Tool(string name, string description, Delegate function) =>
        Tool(Abstraction.Generation.Tool.FromDelegate(name, description, function));

    /// <summary>How many times tool results may be sent back to the model for one user message. Required when tools are added.</summary>
    public ConversationBuilder MaxToolRounds(int rounds)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(rounds);
        _maxToolRounds = rounds;
        return this;
    }

    /// <summary>Creates the conversation.</summary>
    public Conversation Build()
    {
        if (_tools is not null && _toolBuilder is not null)
        {
            throw new InvalidOperationException("Use either Tools(registry) or Tool(...), not both.");
        }

        var tools = _tools ?? _toolBuilder?.Build();
        if (tools is not null && _maxToolRounds is null)
        {
            throw new InvalidOperationException("MaxToolRounds must be set when tools are added.");
        }

        return new Conversation(_model, [.. _messages], _think, _options, tools, _maxToolRounds ?? 0);
    }
}

/// <summary>
/// A chat model that replays scripted replies, for testing tool code and conversation logic without a trained model.
/// Every request is recorded in <see cref="Requests"/>.
/// </summary>
public sealed class FakeChatModel : IChatModel
{
    private readonly Queue<ChatMessage> _replies;

    private FakeChatModel(IEnumerable<ChatMessage> replies) => _replies = new(replies);

    /// <summary>A model that answers with <paramref name="replies"/>, one per request, in order.</summary>
    public static FakeChatModel Script(params ChatMessage[] replies) => new(replies);

    /// <summary>A scripted reply that calls <paramref name="name"/> with <paramref name="arguments"/>.</summary>
    public static ChatMessage ToolCall(string name, JsonObject arguments) => new("assistant", ToolCalls: [new ToolCall(name, arguments)]);

    /// <summary>A scripted plain answer.</summary>
    public static ChatMessage Answer(string content, string? thinking = null) => new("assistant", content, thinking);

    /// <summary>The requests received, in order.</summary>
    public List<ChatRequest> Requests { get; } = [];

    /// <inheritdoc />
    public async IAsyncEnumerable<ChatChunk> StreamAsync(ChatRequest request,
        [System.Runtime.CompilerServices.EnumeratorCancellation] CancellationToken cancellationToken = default)
    {
        cancellationToken.ThrowIfCancellationRequested();
        Requests.Add(request);
        if (!_replies.TryDequeue(out var reply))
        {
            throw new InvalidOperationException("The fake model has no scripted reply left.");
        }

        await Task.Yield();
        yield return new ChatChunk(new ChatDelta(reply.Content, reply.Thinking ?? "", reply.ToolCalls ?? []));
        yield return new ChatChunk(new ChatDelta("", "", []), true, "stop", reply);
    }
}
