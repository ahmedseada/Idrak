// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Idrak.Abstraction.Reinforcement;

namespace Idrak.Reinforcement.Plugins.Coding;

/// <summary>
/// What the <c>coding</c> environment reports in its steps' info (<see cref="EnvironmentStep{TObservation}.Info"/>), and
/// the coding rewards read: every step gives <see cref="Rounds"/>, <see cref="ToolCalls"/> and <see cref="ToolErrors"/>;
/// the last one also the task, the verification and the files the agent changed where the hidden tests go.
/// </summary>
public static class CodingInfo
{
    /// <summary>The task's id: "task".</summary>
    public static EpisodeInfoKey<string> Task { get; } = new("task");

    /// <summary>The assistant's replies so far: "rounds" (<see cref="EpisodeInfoKeys.Rounds"/>).</summary>
    public static EpisodeInfoKey<int> Rounds => EpisodeInfoKeys.Rounds;

    /// <summary>The tool calls made so far: "tool_calls".</summary>
    public static EpisodeInfoKey<int> ToolCalls { get; } = new("tool_calls");

    /// <summary>The tool calls that failed or returned an error so far: "tool_errors".</summary>
    public static EpisodeInfoKey<int> ToolErrors { get; } = new("tool_errors");

    /// <summary>Whether every verification command passed: "verify_passed".</summary>
    public static EpisodeInfoKey<bool> VerifyPassed { get; } = new("verify_passed");

    /// <summary>The task's verification commands: "verify_commands".</summary>
    public static EpisodeInfoKey<int> VerifyCommands { get; } = new("verify_commands");

    /// <summary>The verification commands that passed (in order, before the first that failed): "verify_passed_commands".</summary>
    public static EpisodeInfoKey<int> VerifyPassedCommands { get; } = new("verify_passed_commands");

    /// <summary>The files the agent changed where the hidden tests go, comma-separated ('/' paths); empty for none: "test_edits".</summary>
    public static EpisodeInfoKey<string> TestEdits { get; } = new("test_edits");

    /// <summary>"test-edits" when the episode was refused for them (the option refuse-test-edits): verification did not run and counts as failed: "refused".</summary>
    public static EpisodeInfoKey<string> Refused { get; } = new("refused");

    /// <summary>How the episode ended: a <see cref="CodingOutcome"/> name (Passed, Failed, OutOfBudget): "outcome".</summary>
    public static EpisodeInfoKey<string> Outcome { get; } = new("outcome");

    /// <summary>The tasks left out because their setup failed, comma-separated (only when there are some): "broken_tasks".</summary>
    public static EpisodeInfoKey<string> BrokenTasks { get; } = new("broken_tasks");
}

/// <summary>
/// The coding rewards (registered by <see cref="CodingPlugin"/> in <see cref="RewardFunctions"/>), read from the
/// <c>coding</c> environment's info (<see cref="CodingInfo"/>), combined with weights in its reward expression
/// ("tests-passed:1,rounds:1,tool-errors:1"):
/// <list type="bullet">
/// <item><see cref="TestsPassed"/>: 1 when every verification command passed, else 0.</item>
/// <item><see cref="VerifyFraction"/>: the share of the verification commands that passed (in order, up to the first
/// failure: a build passed and its tests failed is 0.5 of two).</item>
/// <item><see cref="RoundsPenalty"/>: −<c>penalty</c> (0.01) per reply past <c>free</c> (0).</item>
/// <item><see cref="ToolErrorsPenalty"/>: −<c>penalty</c> (0.05) per tool call that failed.</item>
/// </list>
/// </summary>
public static class CodingRewards
{
    /// <summary>1 when the tests pass.</summary>
    public const string TestsPassed = "tests-passed";

    /// <summary>The share of the verification commands that pass.</summary>
    public const string VerifyFraction = "verify-fraction";

    /// <summary>A penalty per round.</summary>
    public const string RoundsPenalty = "rounds";

    /// <summary>A penalty per failed tool call.</summary>
    public const string ToolErrorsPenalty = "tool-errors";

    /// <summary>The four rewards, as the plug-in registers them.</summary>
    public static IReadOnlyList<IRewardFunction> All { get; } = [new TestsPassedReward(), new VerifyFractionReward(), new RoundsReward(), new ToolErrorsReward()];

    // A whole number the coding environment reported; an input without it is refused, naming the environment.
    private static int Count(string reward, RewardInput input, EpisodeInfoKey<int> key) =>
        input.Info is { } info && info.TryGet(key, out int value) && value >= 0 ? value
        : throw new ArgumentException($"The reward {reward} reads '{key.Name}' from the coding environment's info (Idrak.Reinforcement.Plugins.Coding); this answer has none.");

    private static bool Flag(string reward, RewardInput input, EpisodeInfoKey<bool> key) =>
        input.Info is { } info && info.TryGet(key, out bool value) ? value
        : throw new ArgumentException($"The reward {reward} reads '{key.Name}' from the coding environment's info (Idrak.Reinforcement.Plugins.Coding); this answer has none.");

    private static ValueTask<float[]> Score(IReadOnlyList<RewardInput> inputs, CancellationToken cancellationToken, Func<RewardInput, float> score)
    {
        ArgumentNullException.ThrowIfNull(inputs);
        cancellationToken.ThrowIfCancellationRequested();
        var scores = new float[inputs.Count];
        for (int i = 0; i < scores.Length; i++)
        {
            scores[i] = score(inputs[i] ?? throw new ArgumentException("An input is null.", nameof(inputs)));
        }

        return ValueTask.FromResult(scores);
    }

    private sealed class TestsPassedReward : IRewardFunction, IOptionSchema
    {
        public string Name => TestsPassed;

        public string Summary => "1 when every verification command of the coding task passed (its hidden tests), else 0";

        public OptionSchema Options { get; } = new($"reward {TestsPassed}");

        public IReadOnlySet<string> Keys => Options.Keys;

        public ValueTask<float[]> ScoreAsync(IReadOnlyList<RewardInput> inputs, IReadOnlyDictionary<string, string>? options = null, CancellationToken cancellationToken = default)
        {
            _ = Options.Parse(options);
            return Score(inputs, cancellationToken, input => Flag(Name, input, CodingInfo.VerifyPassed) ? 1f : 0f);
        }
    }

    private sealed class VerifyFractionReward : IRewardFunction, IOptionSchema
    {
        public string Name => VerifyFraction;

        public string Summary => "the share of the coding task's verification commands that passed, in order up to the first failure";

        public OptionSchema Options { get; } = new($"reward {VerifyFraction}");

        public IReadOnlySet<string> Keys => Options.Keys;

        public ValueTask<float[]> ScoreAsync(IReadOnlyList<RewardInput> inputs, IReadOnlyDictionary<string, string>? options = null, CancellationToken cancellationToken = default)
        {
            _ = Options.Parse(options);
            return Score(inputs, cancellationToken, input =>
            {
                int commands = Count(Name, input, CodingInfo.VerifyCommands), passed = Count(Name, input, CodingInfo.VerifyPassedCommands);
                return commands == 0 ? (Flag(Name, input, CodingInfo.VerifyPassed) ? 1f : 0f) : (float)Math.Min(passed, commands) / commands;
            });
        }
    }

    private sealed class RoundsReward : IRewardFunction, IOptionSchema
    {
        private static readonly OptionSpec<float> Penalty = OptionSpec.Number("penalty", 0.01f, takes: "a number", summary: "taken per reply past free");
        private static readonly OptionSpec<float> Free = OptionSpec.Number("free", 0f, takes: "a number", summary: "replies without a penalty");

        public string Name => RoundsPenalty;

        public string Summary => "−penalty (0.01) per reply of the coding agent past free (0)";

        public OptionSchema Options { get; } = new($"reward {RoundsPenalty}", Penalty, Free);

        public IReadOnlySet<string> Keys => Options.Keys;

        public ValueTask<float[]> ScoreAsync(IReadOnlyList<RewardInput> inputs, IReadOnlyDictionary<string, string>? options = null, CancellationToken cancellationToken = default)
        {
            var own = Options.Parse(options);
            float penalty = own.Get(Penalty), free = own.Get(Free);
            return Score(inputs, cancellationToken, input => -penalty * Math.Max(0f, Count(Name, input, CodingInfo.Rounds) - free));
        }
    }

    private sealed class ToolErrorsReward : IRewardFunction, IOptionSchema
    {
        private static readonly OptionSpec<float> Penalty = OptionSpec.Number("penalty", 0.05f, takes: "a number", summary: "taken per failed tool call");

        public string Name => ToolErrorsPenalty;

        public string Summary => "−penalty (0.05) per tool call of the coding agent that failed";

        public OptionSchema Options { get; } = new($"reward {ToolErrorsPenalty}", Penalty);

        public IReadOnlySet<string> Keys => Options.Keys;

        public ValueTask<float[]> ScoreAsync(IReadOnlyList<RewardInput> inputs, IReadOnlyDictionary<string, string>? options = null, CancellationToken cancellationToken = default)
        {
            var own = Options.Parse(options);
            float penalty = own.Get(Penalty);
            return Score(inputs, cancellationToken, input => -penalty * Count(Name, input, CodingInfo.ToolErrors));
        }
    }
}
