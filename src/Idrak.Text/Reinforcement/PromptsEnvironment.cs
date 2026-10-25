// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Globalization;
using Idrak.Abstraction.Reinforcement;
using Idrak.Text.Abstractions;
using Idrak.Text.Reinforcement.Abstractions;

namespace Idrak.Text.Reinforcement;

/// <summary>One task of the <c>prompts</c> environment: a chat prompt, the expected answer, the tools it offers.</summary>
/// <param name="Prompt">The conversation the agent answers (system and user messages, earlier turns).</param>
/// <param name="Reference">The expected answer (what a metric reward compares with); null when the reward needs none.</param>
/// <param name="Tools">The tools offered; null for none.</param>
/// <param name="Think">The reasoning mode the template renders with; null: the model's default.</param>
public sealed record PromptTask(IReadOnlyList<ChatMessage> Prompt, string? Reference = null, IReadOnlyList<ToolDefinition>? Tools = null, bool? Think = null)
{
    /// <summary>
    /// The task of a chat transcript (<see cref="ChatTranscript.ReadJsonLines"/>, the tuning data's format): its last
    /// assistant message is the reference and the messages before it the prompt; a transcript that does not end with an
    /// assistant message is a prompt without a reference.
    /// </summary>
    public static PromptTask FromTranscript(ChatTranscript transcript)
    {
        ArgumentNullException.ThrowIfNull(transcript);
        var messages = transcript.Messages;
        bool answered = messages.Count > 1 && messages[^1].Role == "assistant";
        return new PromptTask(answered ? [.. messages.Take(messages.Count - 1)] : messages, answered ? messages[^1].Content : null,
            transcript.Tools.Count > 0 ? transcript.Tools : null, transcript.Think);
    }
}

/// <summary>
/// The <c>prompts</c> environment kind (Plan 27, R3; <see cref="EnvironmentKinds"/>, registered by Idrak.Text): chat
/// prompts with references and optional tools, each episode one prompt drawn from its seed. The agent's reply ends the
/// episode, rewarded by a reward expression (option <c>reward</c>: a <see cref="RewardFunctions"/> or
/// <see cref="TuningMetrics"/> name, "exact" by default, or weighted terms "exact:1,length:0.1"; each term's options as
/// "term.key"); a reply that calls tools, when the kind was made with a tool registry, has them run and the results sent
/// back, for at most <c>rounds</c> replies (1 by default), the last one cut by the limit (truncated) when it still
/// called tools. The library's kind reads its tasks from <c>data</c> (a chat-transcript JSON-lines file); an app makes
/// one of its own tasks, tools and tokenizer (to give each answer's token count for the length reward).
/// </summary>
public sealed class PromptsEnvironmentKind : IPromptDataEnvironmentKind, IRewardedEnvironmentKind
{
    /// <summary>The kind's name.</summary>
    public const string KindName = "prompts";

    // The library's kind reads its tasks from "data"; a kind made with tasks refuses it.
    private static readonly OptionKeys DataKeys = new("data", "reward", "rounds"), TaskKeys = new("reward", "rounds");

    private readonly IReadOnlyList<PromptTask>? _tasks;
    private readonly IToolRegistry? _tools;

    /// <summary>The library's kind: its tasks come from the option <c>data</c>.</summary>
    public PromptsEnvironmentKind()
    {
        Name = KindName;
    }

    /// <summary>A kind of <paramref name="tasks"/> (the option <c>data</c> is then refused), with <paramref name="tools"/> to run the calls of replies.</summary>
    /// <param name="tasks">The tasks, at least one.</param>
    /// <param name="tools">The tools the agent's calls run on; null: a reply's calls end the episode as any reply does.</param>
    /// <param name="name">The kind's name (to register it beside the library's).</param>
    public PromptsEnvironmentKind(IReadOnlyList<PromptTask> tasks, IToolRegistry? tools = null, string name = KindName)
    {
        ArgumentNullException.ThrowIfNull(tasks);
        ArgumentException.ThrowIfNullOrWhiteSpace(name);
        if (tasks.Count == 0)
        {
            throw new ArgumentException("A prompts environment has at least one task.", nameof(tasks));
        }

        _tasks = [.. tasks];
        _tools = tools;
        Name = name;
    }

    /// <inheritdoc />
    public string Name { get; }

    /// <inheritdoc />
    public string Summary => "chat prompts with references: the reply is scored by a reward expression (reward=exact, or exact:1,length:0.1)";

    /// <inheritdoc />
    public IReadOnlySet<string> Keys => _tasks is null ? DataKeys : TaskKeys;

    /// <inheritdoc />
    public string? DataKey => _tasks is null ? "data" : null;

    /// <inheritdoc />
    public string RewardKey => "reward";

    /// <inheritdoc />
    public string? RoundsKey => "rounds";

    /// <inheritdoc />
    public string? ThinkKey => null;

    /// <inheritdoc />
    public Type ObservationType => typeof(ChatObservation);

    /// <inheritdoc />
    public Type ActionType => typeof(ChatMessage);

    /// <summary>A kind of this one's name over the prompts of <paramref name="conversations"/> (<see cref="PromptTask.FromTranscript"/>), counting answers' tokens with <paramref name="tokenizer"/>.</summary>
    public IEnvironmentKind WithTasks(IReadOnlyList<ChatTranscript> conversations, ITokenizer? tokenizer)
    {
        ArgumentNullException.ThrowIfNull(conversations);
        return new PromptsEnvironmentKind([.. conversations.Select(PromptTask.FromTranscript)], _tools, Name) { Tokenizer = tokenizer };
    }

    /// <summary>Counts each answer's tokens into the episode's info (<see cref="EpisodeInfoKeys.Tokens"/>, for the length reward); null: not counted.</summary>
    public ITokenizer? Tokenizer { get; init; }

    /// <inheritdoc />
    public IEnvironment Create(EnvironmentRequest request)
    {
        ArgumentNullException.ThrowIfNull(request);
        var options = request.Options;
        var rewardOptions = new Dictionary<string, string>(StringComparer.Ordinal);
        string reward = "exact", data = "";
        int rounds = 1;
        foreach (var (key, value) in options ?? new Dictionary<string, string>())
        {
            switch (key)
            {
                case "reward":
                    reward = value;
                    break;
                case "rounds":
                    rounds = int.TryParse(value, NumberStyles.Integer, CultureInfo.InvariantCulture, out int r) && r > 0 ? r
                        : throw new ArgumentException($"The {Name} environment's rounds={value} is not a positive whole number.", nameof(request));
                    break;
                case "data" when _tasks is null:
                    data = value;
                    break;
                default:
                    if (!key.Contains('.', StringComparison.Ordinal))
                    {
                        throw new ArgumentException($"The {Name} environment does not take the option '{key}'; it takes: {string.Join(", ", Keys)}, and a reward term's options as term.key.",
                            nameof(request));
                    }

                    rewardOptions[key] = value;
                    break;
            }
        }

        var expression = RewardExpression.Parse(reward, rewardOptions);
        var tasks = _tasks ?? (data.Length > 0 ? [.. ChatTranscript.ReadJsonLines(data).Select(PromptTask.FromTranscript)]
            : throw new ArgumentException($"The {Name} environment reads its tasks from the option data (a chat-transcript JSON-lines file).", nameof(request)));
        if (tasks.Count == 0)
        {
            throw new ArgumentException($"The {Name} environment's data holds no task: {data}.", nameof(request));
        }

        return new PromptsEnvironment(tasks, expression, rounds, _tools, Tokenizer);
    }
}

// An episode of a prompts kind: a task drawn from the seed, then replies until one answers (or the rounds run out).
internal sealed class PromptsEnvironment(IReadOnlyList<PromptTask> tasks, RewardExpression reward, int rounds, IToolRegistry? tools, ITokenizer? tokenizer)
    : IEnvironment<ChatObservation, ChatMessage>
{
    private PromptTask? _task;
    private List<ChatMessage> _messages = [];
    private int _turns;
    private bool _active;

    public Space ObservationSpace => Space.Conversation;

    public Space ActionSpace => Space.Conversation;

    public ChatObservation Reset(int seed)
    {
        _task = tasks[new Random(seed).Next(tasks.Count)];
        _messages = [.. _task.Prompt];
        _turns = 0;
        _active = true;
        return Observation();
    }

    public async ValueTask<EnvironmentStep<ChatObservation>> StepAsync(ChatMessage action, CancellationToken cancellationToken = default)
    {
        cancellationToken.ThrowIfCancellationRequested();
        if (!_active || _task is null)
        {
            throw new InvalidOperationException("No episode is under way: reset the prompts environment first (also after an episode ended).");
        }

        ArgumentNullException.ThrowIfNull(action);
        _turns++;
        _messages.Add(action);
        bool calls = action.ToolCalls is { Count: > 0 };
        if (calls && tools is not null && _turns < rounds)
        {
            var results = await tools.InvokeAsync(action.ToolCalls!, cancellationToken).ConfigureAwait(false);
            _messages.AddRange(results.Select(r => r.ToMessage()));
            return new EnvironmentStep<ChatObservation>(Observation(), 0f, false, false, Info(action));
        }

        _active = false;
        var info = Info(action);
        string prompt = _task.Prompt.LastOrDefault(m => m.Role == "user")?.Content ?? "";
        var scores = await reward.ScoreAsync([new RewardInput(prompt, action.Content, _task.Reference, info)], cancellationToken).ConfigureAwait(false);
        bool truncated = calls && tools is not null;
        return new EnvironmentStep<ChatObservation>(Observation(), scores[0], !truncated, truncated, info);
    }

    private ChatObservation Observation() => new([.. _messages], _task!.Tools, _task.Think);

    private EpisodeInfo Info(ChatMessage action)
    {
        var info = new EpisodeInfo().Set(EpisodeInfoKeys.Rounds, _turns);
        if (tokenizer is not null)
        {
            info.Set(EpisodeInfoKeys.Tokens, tokenizer.Encode(action.Content).Count);
        }

        return info;
    }

    public void Dispose()
    {
        _active = false;
    }
}
