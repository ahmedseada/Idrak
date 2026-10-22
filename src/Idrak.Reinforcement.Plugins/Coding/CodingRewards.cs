// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Globalization;
using Idrak.Abstraction.Reinforcement;

namespace Idrak.Reinforcement.Plugins.Coding;

/// <summary>
/// What the <c>coding</c> environment reports in its steps' info (<see cref="EnvironmentStep{TObservation}.Info"/>), and
/// the coding rewards read: every step gives <see cref="Rounds"/>, <see cref="ToolCalls"/> and <see cref="ToolErrors"/>;
/// the last one also the task, the verification and the files the agent changed where the hidden tests go.
/// </summary>
public static class CodingInfo
{
    /// <summary>The task's id.</summary>
    public const string Task = "task";

    /// <summary>The assistant's replies so far.</summary>
    public const string Rounds = "rounds";

    /// <summary>The tool calls made so far.</summary>
    public const string ToolCalls = "tool_calls";

    /// <summary>The tool calls that failed or returned an error so far.</summary>
    public const string ToolErrors = "tool_errors";

    /// <summary>"true" when every verification command passed, else "false".</summary>
    public const string VerifyPassed = "verify_passed";

    /// <summary>The task's verification commands.</summary>
    public const string VerifyCommands = "verify_commands";

    /// <summary>The verification commands that passed (in order, before the first that failed).</summary>
    public const string VerifyPassedCommands = "verify_passed_commands";

    /// <summary>The files the agent changed where the hidden tests go, comma-separated ('/' paths); empty for none.</summary>
    public const string TestEdits = "test_edits";

    /// <summary>"test-edits" when the episode was refused for them (the option refuse-test-edits): verification did not run and counts as failed.</summary>
    public const string Refused = "refused";

    /// <summary>How the episode ended: a <see cref="CodingOutcome"/> name (Passed, Failed, OutOfBudget).</summary>
    public const string Outcome = "outcome";

    /// <summary>The tasks left out because their setup failed, comma-separated (only when there are some).</summary>
    public const string BrokenTasks = "broken_tasks";
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

    private static Dictionary<string, string> Options(string reward, IReadOnlyDictionary<string, string>? options, IReadOnlyCollection<string> keys)
    {
        var own = new Dictionary<string, string>(StringComparer.Ordinal);
        foreach (var (key, value) in options ?? new Dictionary<string, string>())
        {
            if (!keys.Contains(key, StringComparer.Ordinal))
            {
                throw new ArgumentException($"The reward {reward} does not take the option '{key}'; it takes: {(keys.Count == 0 ? "none" : string.Join(", ", keys))}.", nameof(options));
            }

            own[key] = value;
        }

        return own;
    }

    private static float Number(string reward, Dictionary<string, string> options, string key, float fallback) =>
        !options.TryGetValue(key, out var text) ? fallback
        : float.TryParse(text, NumberStyles.Float, CultureInfo.InvariantCulture, out float value) && float.IsFinite(value) ? value
        : throw new ArgumentException($"The reward {reward}'s option {key}={text} is not a number.");

    // A whole number the coding environment reported; an input without it is refused, naming the environment.
    private static int Count(string reward, RewardInput input, string key) =>
        input.Info is { } info && info.TryGetValue(key, out var text) && int.TryParse(text, NumberStyles.Integer, CultureInfo.InvariantCulture, out int value) && value >= 0 ? value
        : throw new ArgumentException($"The reward {reward} reads '{key}' from the coding environment's info (Idrak.Reinforcement.Plugins.Coding); this answer has none.");

    private static bool Flag(string reward, RewardInput input, string key) =>
        input.Info is { } info && info.TryGetValue(key, out var text) && bool.TryParse(text, out bool value) ? value
        : throw new ArgumentException($"The reward {reward} reads '{key}' from the coding environment's info (Idrak.Reinforcement.Plugins.Coding); this answer has none.");

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

    private sealed class TestsPassedReward : IRewardFunction
    {
        public string Name => TestsPassed;

        public string Summary => "1 when every verification command of the coding task passed (its hidden tests), else 0";

        public IReadOnlyCollection<string> Keys { get; } = [];

        public ValueTask<float[]> ScoreAsync(IReadOnlyList<RewardInput> inputs, IReadOnlyDictionary<string, string>? options = null, CancellationToken cancellationToken = default)
        {
            _ = Options(Name, options, Keys);
            return Score(inputs, cancellationToken, input => Flag(Name, input, CodingInfo.VerifyPassed) ? 1f : 0f);
        }
    }

    private sealed class VerifyFractionReward : IRewardFunction
    {
        public string Name => VerifyFraction;

        public string Summary => "the share of the coding task's verification commands that passed, in order up to the first failure";

        public IReadOnlyCollection<string> Keys { get; } = [];

        public ValueTask<float[]> ScoreAsync(IReadOnlyList<RewardInput> inputs, IReadOnlyDictionary<string, string>? options = null, CancellationToken cancellationToken = default)
        {
            _ = Options(Name, options, Keys);
            return Score(inputs, cancellationToken, input =>
            {
                int commands = Count(Name, input, CodingInfo.VerifyCommands), passed = Count(Name, input, CodingInfo.VerifyPassedCommands);
                return commands == 0 ? (Flag(Name, input, CodingInfo.VerifyPassed) ? 1f : 0f) : (float)Math.Min(passed, commands) / commands;
            });
        }
    }

    private sealed class RoundsReward : IRewardFunction
    {
        public string Name => RoundsPenalty;

        public string Summary => "−penalty (0.01) per reply of the coding agent past free (0)";

        public IReadOnlyCollection<string> Keys { get; } = ["penalty", "free"];

        public ValueTask<float[]> ScoreAsync(IReadOnlyList<RewardInput> inputs, IReadOnlyDictionary<string, string>? options = null, CancellationToken cancellationToken = default)
        {
            var own = Options(Name, options, Keys);
            float penalty = Number(Name, own, "penalty", 0.01f), free = Number(Name, own, "free", 0f);
            return Score(inputs, cancellationToken, input => -penalty * Math.Max(0f, Count(Name, input, CodingInfo.Rounds) - free));
        }
    }

    private sealed class ToolErrorsReward : IRewardFunction
    {
        public string Name => ToolErrorsPenalty;

        public string Summary => "−penalty (0.05) per tool call of the coding agent that failed";

        public IReadOnlyCollection<string> Keys { get; } = ["penalty"];

        public ValueTask<float[]> ScoreAsync(IReadOnlyList<RewardInput> inputs, IReadOnlyDictionary<string, string>? options = null, CancellationToken cancellationToken = default)
        {
            var own = Options(Name, options, Keys);
            float penalty = Number(Name, own, "penalty", 0.05f);
            return Score(inputs, cancellationToken, input => -penalty * Count(Name, input, CodingInfo.ToolErrors));
        }
    }
}
