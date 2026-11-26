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
/// The C# rewards (registered by <see cref="CSharpCodingPlugin"/> in <c>RewardFunctions</c>), read from the
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

    // The options by key the contract still passes: a reward's options are its constructor's, so every key is one it does
    // not take.
    private static void NoOptions(string reward, IReadOnlyDictionary<string, string>? options)
    {
        if (options is { Count: > 0 })
        {
            throw new ArgumentException($"The reward {reward} takes its options as constructor parameters, not by key (given '{string.Join("', '", options.Keys)}').", nameof(options));
        }
    }

    private sealed class TestsReward : IRewardFunction
    {
        public string Name => Tests;

        public ValueTask<float[]> ScoreAsync(IReadOnlyList<RewardInput> inputs, IReadOnlyDictionary<string, string>? options = null, CancellationToken cancellationToken = default)
        {
            NoOptions(Name, options);
            return Score(inputs, cancellationToken, input =>
            {
                int total = Count(Name, input, CSharpInfo.TestsTotal), passed = Count(Name, input, CSharpInfo.TestsPassed);
                return total == 0 ? 0f : (float)Math.Min(passed, total) / total;
            });
        }
    }

    // 1 when the C# task's build passed, else 0, less error (0.1) per error and warning (0) per warning in the build's logs,
    // at least floor (-1).
    private sealed class BuildReward(float error = 0.1f, float warning = 0f, float floor = -1f) : IRewardFunction
    {
        public string Name => Build;

        public ValueTask<float[]> ScoreAsync(IReadOnlyList<RewardInput> inputs, IReadOnlyDictionary<string, string>? options = null, CancellationToken cancellationToken = default)
        {
            NoOptions(Name, options);
            return Score(inputs, cancellationToken, input =>
            {
                float score = (Flag(Name, input, CSharpInfo.BuildPassed) ? 1f : 0f) - error * Count(Name, input, CSharpInfo.BuildErrors)
                              - warning * Count(Name, input, CSharpInfo.BuildWarnings);
                return Math.Max(floor, score);
            });
        }
    }
}
