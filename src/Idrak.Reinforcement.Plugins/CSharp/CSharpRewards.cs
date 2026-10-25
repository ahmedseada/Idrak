// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Idrak.Abstraction.Reinforcement;

namespace Idrak.Reinforcement.Plugins.CSharp;

/// <summary>What the <c>csharp</c> environment adds to its episodes' last info (<see cref="CSharpEnvironmentVariant"/>), read by the C# rewards.</summary>
public static class CSharpInfo
{
    /// <summary>The tests the TRX results count (0 when the tests did not run): "csharp_tests_total".</summary>
    public static EpisodeInfoKey<int> TestsTotal { get; } = new("csharp_tests_total");

    /// <summary>The tests that passed: "csharp_tests_passed".</summary>
    public static EpisodeInfoKey<int> TestsPassed { get; } = new("csharp_tests_passed");

    /// <summary>Whether verification's <c>dotnet build</c> ran and passed: "csharp_build_passed".</summary>
    public static EpisodeInfoKey<bool> BuildPassed { get; } = new("csharp_build_passed");

    /// <summary>The distinct errors in the build's error log: "csharp_build_errors".</summary>
    public static EpisodeInfoKey<int> BuildErrors { get; } = new("csharp_build_errors");

    /// <summary>The distinct warnings in the build's warning log: "csharp_build_warnings".</summary>
    public static EpisodeInfoKey<int> BuildWarnings { get; } = new("csharp_build_warnings");
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

    private static int Count(string reward, RewardInput input, EpisodeInfoKey<int> key) =>
        input.Info is { } info && info.TryGet(key, out int value) && value >= 0 ? value
        : throw new ArgumentException($"The reward {reward} reads '{key.Name}' from the csharp environment's info (Idrak.Reinforcement.Plugins.CSharp); this answer has none.");

    private static bool Flag(string reward, RewardInput input, EpisodeInfoKey<bool> key) =>
        input.Info is { } info && info.TryGet(key, out bool value) ? value
        : throw new ArgumentException($"The reward {reward} reads '{key.Name}' from the csharp environment's info (Idrak.Reinforcement.Plugins.CSharp); this answer has none.");

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

    private sealed class TestsReward : IRewardFunction, IOptionSchema
    {
        public string Name => Tests;

        public string Summary => "the share of the C# task's tests that passed, from dotnet test's TRX results (tests that did not run count as failed)";

        public OptionSchema Options { get; } = new($"reward {Tests}");

        public IReadOnlySet<string> Keys => Options.Keys;

        public ValueTask<float[]> ScoreAsync(IReadOnlyList<RewardInput> inputs, IReadOnlyDictionary<string, string>? options = null, CancellationToken cancellationToken = default)
        {
            _ = Options.Parse(options);
            return Score(inputs, cancellationToken, input =>
            {
                int total = Count(Name, input, CSharpInfo.TestsTotal), passed = Count(Name, input, CSharpInfo.TestsPassed);
                return total == 0 ? 0f : (float)Math.Min(passed, total) / total;
            });
        }
    }

    private sealed class BuildReward : IRewardFunction, IOptionSchema
    {
        private static readonly OptionSpec<float> Error = OptionSpec.Number("error", 0.1f, takes: "a number", summary: "taken per error");
        private static readonly OptionSpec<float> Warning = OptionSpec.Number("warning", 0f, takes: "a number", summary: "taken per warning");
        private static readonly OptionSpec<float> Floor = OptionSpec.Number("floor", -1f, takes: "a number", summary: "the least score");

        public string Name => Build;

        public string Summary => "1 when the C# task's build passed, else 0, less error (0.1) per error and warning (0) per warning in the build's logs, at least floor (-1)";

        public OptionSchema Options { get; } = new($"reward {Build}", Error, Warning, Floor);

        public IReadOnlySet<string> Keys => Options.Keys;

        public ValueTask<float[]> ScoreAsync(IReadOnlyList<RewardInput> inputs, IReadOnlyDictionary<string, string>? options = null, CancellationToken cancellationToken = default)
        {
            var own = Options.Parse(options);
            float error = own.Get(Error), warning = own.Get(Warning), floor = own.Get(Floor);
            return Score(inputs, cancellationToken, input =>
            {
                float score = (Flag(Name, input, CSharpInfo.BuildPassed) ? 1f : 0f) - error * Count(Name, input, CSharpInfo.BuildErrors)
                              - warning * Count(Name, input, CSharpInfo.BuildWarnings);
                return Math.Max(floor, score);
            });
        }
    }
}
