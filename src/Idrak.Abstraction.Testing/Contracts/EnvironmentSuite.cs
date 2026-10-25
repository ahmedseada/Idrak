// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Text.Json.Nodes;
using Idrak.Abstraction.Reinforcement;

namespace Idrak.Abstraction.Testing;

/// <summary>
/// The checks of an environment kind (<see cref="IEnvironmentKind"/>) and the environments it creates
/// (<see cref="IEnvironment{TObservation, TAction}"/>), on episodes the suite plays with actions from
/// <c>act</c> (the suite cannot make an action of every type: the test chooses them, from a random source the suite seeds):
/// <list type="bullet">
/// <item>the kind has a name and a summary, says the observation and action types asked for, creates an environment of
/// them, and refuses an option not in its <see cref="IConfigurableKind.Keys"/>, naming it;</item>
/// <item>a step before the first reset is refused (<see cref="InvalidOperationException"/>), and so is a step after one
/// that ended the episode: a finished episode must be reset;</item>
/// <item>observations lie in the observation space, the chosen actions in the action space, rewards are finite;</item>
/// <item>a seeded replay gives the same steps (observations, rewards, ends, info): in the same environment reset again,
/// in a new one of the kind, and in two run at once on two threads;</item>
/// <item>a cancelled step throws an <see cref="OperationCanceledException"/>; disposing twice is harmless.</item>
/// </list>
/// Each episode runs until it ends or <see cref="MaxSteps"/> steps.
/// </summary>
/// <typeparam name="TObservation">The kind's observations.</typeparam>
/// <typeparam name="TAction">Its actions.</typeparam>
/// <param name="act">An action for an observation, drawn from the random source given (the same draws give the same action).</param>
/// <param name="options">The kind's options for every environment the suite creates; null for its defaults.</param>
/// <param name="compare">How two observations differ (null: they are the same); by default float and int arrays element by element, anything else by its equality.</param>
public sealed class EnvironmentSuite<TObservation, TAction>(Func<Random, TObservation, TAction> act, IReadOnlyDictionary<string, string>? options = null,
    Func<TObservation, TObservation, string?>? compare = null) : ContractSuite<IEnvironmentKind>
{
    private const string UnknownOption = "conformance-unknown-option";

    private readonly Func<Random, TObservation, TAction> _act = act ?? throw new ArgumentNullException(nameof(act));
    private readonly Func<TObservation, TObservation, string?> _compare = compare ?? Same;

    /// <inheritdoc />
    public override string Name => "environment";

    /// <summary>The steps an episode may take before the suite stops it (an environment that never ends one).</summary>
    public int MaxSteps { get; init; } = 64;

    /// <inheritdoc />
    public override IEnumerable<ContractCase> FixedCases()
    {
        yield return Case("one episode", 1, 1);
        yield return Case("seed 0", 0, 2);
        yield return Case("a few episodes", 7, 4);
    }

    /// <inheritdoc />
    public override ContractCase RandomCase(Random random, bool large)
    {
        ArgumentNullException.ThrowIfNull(random);
        return Case("random", random.Next(), random.Next(1, large ? 12 : 4));
    }

    /// <inheritdoc />
    public override void Run(IEnvironmentKind implementation, ContractCase @case, CaseChecks checks)
    {
        ArgumentNullException.ThrowIfNull(implementation);
        ArgumentNullException.ThrowIfNull(@case);
        ArgumentNullException.ThrowIfNull(checks);
        var kind = implementation;
        int seed = (int)@case.Data["seed"]!, episodes = (int)@case.Data["episodes"]!;
        checks.Check("a name and a summary", !string.IsNullOrWhiteSpace(kind.Name) && !string.IsNullOrWhiteSpace(kind.Summary), $"name \"{kind.Name}\", summary \"{kind.Summary}\"");
        checks.Check("the observation and action types asked for", kind.ObservationType == typeof(TObservation) && kind.ActionType == typeof(TAction),
            $"the kind observes {kind.ObservationType.Name} and acts with {kind.ActionType.Name}, not {typeof(TObservation).Name} and {typeof(TAction).Name}");
        try
        {
            kind.Create(new EnvironmentRequest(new Dictionary<string, string>(StringComparer.Ordinal) { [UnknownOption] = "1" }, seed)).Dispose();
            checks.Fail("an option not in Keys is refused, naming it", $"created an environment with the option '{UnknownOption}'");
        }
        catch (ArgumentException e)
        {
            checks.Check("an option not in Keys is refused, naming it", e.Message.Contains(UnknownOption, StringComparison.Ordinal), $"\"{e.Message}\" does not name '{UnknownOption}'");
        }

        var draws = new Random(seed);
        var plan = new (int Seed, int Actions)[episodes];
        for (int e = 0; e < episodes; e++)
        {
            plan[e] = (draws.Next(), draws.Next());
        }

        using var environment = Create(kind, seed, checks);
        if (environment is null)
        {
            return;
        }

        checks.Check("observation and action spaces", environment.ObservationSpace is not null && environment.ActionSpace is not null, "a space is null");
        using (var fresh = Create(kind, seed, checks))
        {
            if (fresh is not null)
            {
                var action = _act(new Random(seed), environment.Reset(plan[0].Seed));
                checks.Check("a step before the first reset is refused", Refuses(fresh, action), "a new environment took a step before it was reset");
            }
        }

        // The episodes, played once; each checked as it goes, then replayed three ways.
        var played = new List<Episode>();
        foreach (var (episodeSeed, actionSeed) in plan)
        {
            played.Add(Play(environment, episodeSeed, actionSeed, checks));
        }

        var again = plan.Select(p => Play(environment, p.Seed, p.Actions, null)).ToList();
        checks.Compare("the same environment reset with the same seed replays the episode", Differs(played, again));
        using (var other = Create(kind, seed, checks))
        {
            if (other is not null)
            {
                checks.Compare("a new environment of the kind replays the episode from its seed", Differs(played, [.. plan.Select(p => Play(other, p.Seed, p.Actions, null))]));
            }
        }

        var parallel = new List<Episode>[2];
        Parallel.For(0, 2, i =>
        {
            using var mine = kind.Create(new EnvironmentRequest(options, seed));
            parallel[i] = [.. plan.Select(p => Play((IEnvironment<TObservation, TAction>)mine, p.Seed, p.Actions, null))];
        });
        checks.Compare("environments run at once give the steps of one alone", Differs(played, parallel[0]) ?? Differs(played, parallel[1]));

        var first = _act(new Random(plan[0].Actions), environment.Reset(plan[0].Seed));
        using (var cancelled = new CancellationTokenSource())
        {
            cancelled.Cancel();
            try
            {
                Wait(environment.StepAsync(first, cancelled.Token));
                checks.Fail("a cancelled step throws OperationCanceledException", "the step ran with its token cancelled");
            }
            catch (OperationCanceledException)
            {
                checks.Pass("a cancelled step throws OperationCanceledException");
            }
        }

        var disposed = kind.Create(new EnvironmentRequest(options, seed));
        disposed.Dispose();
        try
        {
            disposed.Dispose();
            checks.Pass("Dispose twice is harmless");
        }
        catch (Exception e) when (e is not OutOfMemoryException)
        {
            checks.Fail("Dispose twice is harmless", $"the second Dispose threw {e.GetType().Name}: {e.Message}");
        }
    }

    // One episode played: its observations, rewards, ends and info, and the actions taken.
    private sealed record Episode(List<TObservation> Observations, List<float> Rewards, List<(bool Terminated, bool Truncated)> Ends, List<string> Info);

    private Episode Play(IEnvironment<TObservation, TAction> environment, int episodeSeed, int actionSeed, CaseChecks? checks)
    {
        var random = new Random(actionSeed);
        var observation = environment.Reset(episodeSeed);
        var episode = new Episode([observation], [], [], []);
        checks?.Check("the first observation is in the observation space", environment.ObservationSpace.ContainsValue(observation),
            $"{Show(observation)} is not in {environment.ObservationSpace}");
        string? outside = null, notFinite = null, actionOutside = null;
        for (int step = 0; step < MaxSteps; step++)
        {
            var action = _act(random, observation);
            if (actionOutside is null && !environment.ActionSpace.ContainsValue(action))
            {
                actionOutside = $"step {step}: the action {Show(action)} is not in {environment.ActionSpace}";
            }

            var result = Wait(environment.StepAsync(action));
            observation = result.Observation;
            episode.Observations.Add(observation);
            episode.Rewards.Add(result.Reward);
            episode.Ends.Add((result.Terminated, result.Truncated));
            episode.Info.Add(Show(result.Info));
            if (outside is null && !environment.ObservationSpace.ContainsValue(observation))
            {
                outside = $"step {step}: {Show(observation)} is not in {environment.ObservationSpace}";
            }

            if (notFinite is null && !float.IsFinite(result.Reward))
            {
                notFinite = $"step {step}: reward {result.Reward}";
            }

            if (result.Done)
            {
                if (checks is not null)
                {
                    checks.Check("a finished episode must be reset: stepping it again is refused", Refuses(environment, _act(random, observation)),
                        $"the environment took a step after the episode ended at step {step}");
                }

                break;
            }
        }

        if (checks is not null)
        {
            checks.Check("observations are in the observation space", outside is null, outside ?? "");
            checks.Check("the chosen actions are in the action space", actionOutside is null, actionOutside ?? "");
            checks.Check("rewards are finite", notFinite is null, notFinite ?? "");
        }

        return episode;
    }

    private string? Differs(List<Episode> expected, List<Episode> actual)
    {
        for (int e = 0; e < expected.Count; e++)
        {
            var (a, b) = (expected[e], actual[e]);
            if (a.Rewards.Count != b.Rewards.Count)
            {
                return $"episode {e}: {b.Rewards.Count} steps, {a.Rewards.Count} expected";
            }

            for (int s = 0; s < a.Observations.Count; s++)
            {
                if (_compare(a.Observations[s], b.Observations[s]) is { } difference)
                {
                    return $"episode {e}, observation {s}: {difference}";
                }
            }

            if (Comparisons.Difference(a.Rewards.ToArray(), b.Rewards.ToArray(), 0f) is { } rewards)
            {
                return $"episode {e}, rewards: {rewards}";
            }

            if (Comparisons.Sequences(a.Ends, b.Ends) is { } ends)
            {
                return $"episode {e}, ends (terminated, truncated): {ends}";
            }

            if (Comparisons.Sequences(a.Info, b.Info) is { } info)
            {
                return $"episode {e}, info: {info}";
            }
        }

        return null;
    }

    private IEnvironment<TObservation, TAction>? Create(IEnvironmentKind kind, int seed, CaseChecks checks)
    {
        var made = kind.Create(new EnvironmentRequest(options, seed));
        if (made is IEnvironment<TObservation, TAction> typed)
        {
            checks.Pass("creates an environment of the types asked for");
            return typed;
        }

        checks.Fail("creates an environment of the types asked for", $"created a {made?.GetType().Name ?? "null"}, not an IEnvironment<{typeof(TObservation).Name}, {typeof(TAction).Name}>");
        made?.Dispose();
        return null;
    }

    // Whether a step is refused because no episode is under way.
    private static bool Refuses(IEnvironment<TObservation, TAction> environment, TAction action)
    {
        try
        {
            Wait(environment.StepAsync(action));
            return false;
        }
        catch (InvalidOperationException)
        {
            return true;
        }
    }

    private static T Wait<T>(ValueTask<T> task) => task.IsCompletedSuccessfully ? task.Result : task.AsTask().GetAwaiter().GetResult();

    private static string? Same(TObservation a, TObservation b) => (a, b) switch
    {
        (float[] x, float[] y) => Comparisons.Difference(x, y, 0f),
        (int[] x, int[] y) => Comparisons.Difference(x, y),
        _ => Comparisons.Exact(a, b),
    };

    private static string Show(object? value) => value switch
    {
        null => "null",
        float[] values => $"[{string.Join(", ", values.Take(8).Select(v => v.ToString("G6", System.Globalization.CultureInfo.InvariantCulture)))}{(values.Length > 8 ? ", ..." : "")}]",
        EpisodeInfo info => string.Join("; ", info.Entries.Select(p => $"{p.Key}={p.Value}")),
        _ => Convert.ToString(value, System.Globalization.CultureInfo.InvariantCulture) ?? "",
    };

    private static ContractCase Case(string name, int seed, int episodes) =>
        new($"{name} ({episodes} episodes, seed {seed})", new JsonObject { ["seed"] = seed, ["episodes"] = episodes });
}
