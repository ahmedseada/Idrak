// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Globalization;
using Idrak.Abstraction.Reinforcement;

namespace Idrak.Reinforcement.Plugins.CSharp;

/// <summary>What the <c>csharp</c> environment adds to its episodes' last info (<see cref="CSharpEnvironmentVariant"/>), read by the C# rewards.</summary>
public static class CSharpInfo
{
    /// <summary>The tests the TRX results count (0 when the tests did not run).</summary>
    public const string TestsTotal = "csharp_tests_total";

    /// <summary>The tests that passed.</summary>
    public const string TestsPassed = "csharp_tests_passed";

    /// <summary>"true" when verification's <c>dotnet build</c> ran and passed, else "false".</summary>
    public const string BuildPassed = "csharp_build_passed";

    /// <summary>The distinct errors in the build's error log.</summary>
    public const string BuildErrors = "csharp_build_errors";

    /// <summary>The distinct warnings in the build's warning log.</summary>
    public const string BuildWarnings = "csharp_build_warnings";
}

/// <summary>
/// The C# rewards (registered by <see cref="CSharpCodingPlugin"/> in <see cref="RewardFunctions"/>), read from the
/// <c>csharp</c> environment's info (<see cref="CSharpInfo"/>), combined with the coding ones (<c>rounds</c>,
/// <c>tool-errors</c>) and a teacher term by weights in the reward expression:
/// <list type="bullet">
/// <item><see cref="Tests"/>: the share of the tests that passed (seven of ten is 0.7; tests that did not run count as
/// failed; 0 when none ran).</item>
/// <item><see cref="Build"/>: 1 when the build passed, else 0, less <c>error</c> (0.1) per error and <c>warning</c> (0) per
/// warning in its logs, never below <c>floor</c> (−1).</item>
/// </list>
/// </summary>
public static class CSharpRewards
{
    /// <summary>The share of the tests passed.</summary>
    public const string Tests = "csharp-tests";

    /// <summary>The build, less its errors and warnings.</summary>
    public const string Build = "csharp-build";

    /// <summary>The two rewards, as the plug-in registers them.</summary>
    public static IReadOnlyList<IRewardFunction> All { get; } = [new TestsReward(), new BuildReward()];

    private static int Count(string reward, RewardInput input, string key) =>
        input.Info is { } info && info.TryGetValue(key, out var text) && int.TryParse(text, NumberStyles.Integer, CultureInfo.InvariantCulture, out int value) && value >= 0 ? value
        : throw new ArgumentException($"The reward {reward} reads '{key}' from the csharp environment's info (Idrak.Reinforcement.Plugins.CSharp); this answer has none.");

    private static bool Flag(string reward, RewardInput input, string key) =>
        input.Info is { } info && info.TryGetValue(key, out var text) && bool.TryParse(text, out bool value) ? value
        : throw new ArgumentException($"The reward {reward} reads '{key}' from the csharp environment's info (Idrak.Reinforcement.Plugins.CSharp); this answer has none.");

    private static Dictionary<string, float> Options(string reward, IReadOnlyDictionary<string, string>? options, IReadOnlyDictionary<string, float> defaults)
    {
        var values = new Dictionary<string, float>(defaults, StringComparer.Ordinal);
        foreach (var (key, text) in options ?? new Dictionary<string, string>())
        {
            if (!defaults.ContainsKey(key))
            {
                throw new ArgumentException($"The reward {reward} does not take the option '{key}'; it takes: {(defaults.Count == 0 ? "none" : string.Join(", ", defaults.Keys))}.", nameof(options));
            }

            values[key] = float.TryParse(text, NumberStyles.Float, CultureInfo.InvariantCulture, out float value) && float.IsFinite(value) ? value
                : throw new ArgumentException($"The reward {reward}'s option {key}={text} is not a number.");
        }

        return values;
    }

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

    private sealed class TestsReward : IRewardFunction
    {
        private static readonly Dictionary<string, float> None = new(StringComparer.Ordinal);

        public string Name => Tests;

        public string Summary => "the share of the C# task's tests that passed, from dotnet test's TRX results (tests that did not run count as failed)";

        public IReadOnlyCollection<string> Keys { get; } = [];

        public ValueTask<float[]> ScoreAsync(IReadOnlyList<RewardInput> inputs, IReadOnlyDictionary<string, string>? options = null, CancellationToken cancellationToken = default)
        {
            _ = Options(Name, options, None);
            return Score(inputs, cancellationToken, input =>
            {
                int total = Count(Name, input, CSharpInfo.TestsTotal), passed = Count(Name, input, CSharpInfo.TestsPassed);
                return total == 0 ? 0f : (float)Math.Min(passed, total) / total;
            });
        }
    }

    private sealed class BuildReward : IRewardFunction
    {
        private static readonly Dictionary<string, float> Defaults = new(StringComparer.Ordinal) { ["error"] = 0.1f, ["warning"] = 0f, ["floor"] = -1f };

        public string Name => Build;

        public string Summary => "1 when the C# task's build passed, else 0, less error (0.1) per error and warning (0) per warning in the build's logs, at least floor (-1)";

        public IReadOnlyCollection<string> Keys { get; } = ["error", "warning", "floor"];

        public ValueTask<float[]> ScoreAsync(IReadOnlyList<RewardInput> inputs, IReadOnlyDictionary<string, string>? options = null, CancellationToken cancellationToken = default)
        {
            var own = Options(Name, options, Defaults);
            return Score(inputs, cancellationToken, input =>
            {
                float score = (Flag(Name, input, CSharpInfo.BuildPassed) ? 1f : 0f) - own["error"] * Count(Name, input, CSharpInfo.BuildErrors)
                              - own["warning"] * Count(Name, input, CSharpInfo.BuildWarnings);
                return Math.Max(own["floor"], score);
            });
        }
    }
}
