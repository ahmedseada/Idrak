// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Text.Json.Nodes;
using Idrak.Abstraction.Reinforcement;

namespace Idrak.Abstraction.Testing;

/// <summary>
/// The checks of a preference-pair source (<see cref="IPreferencePairSource"/>) on episodes the suite draws: groups of
/// episodes of one to several steps, returns of either sign with ties (pass or fail rewards), and some episodes whose info
/// records a verdict and a kept text (<see cref="PreferenceInfoKeys"/>, some at a step of their own):
/// <list type="bullet">
/// <item>a name and a summary, and <see cref="IConfigurableKind.Keys"/> are its declared options';</item>
/// <item>every choice names an episode and a step there are, a kept text replaces one step's answer, and its two
/// answers differ;</item>
/// <item>a source that needs groups pairs episodes of the group the choice names;</item>
/// <item>the choices are added after what the list held, which stays, and the count returned is theirs;</item>
/// <item>the same episodes give the same choices in the same order, also from several threads at once; no episode gives none;</item>
/// <item>an option not in its keys is refused, naming it;</item>
/// <item>given a reference (the library's source of that name), the same choices.</item>
/// </list>
/// </summary>
/// <param name="options">The source's options for every call; null for its defaults.</param>
/// <param name="reference">The source to agree with, or null.</param>
public sealed class PreferencePairSourceSuite(IReadOnlyDictionary<string, string>? options = null, IPreferencePairSource? reference = null)
    : ContractSuite<IPreferencePairSource>
{
    private const string UnknownOption = "conformance-unknown-option";

    private static readonly PreferenceChoice Sentinel = new(PreferenceAnswer.Sampled(-7), PreferenceAnswer.Sampled(-8), 42f, -9);

    /// <inheritdoc />
    public override string Name => "preference-pair-source";

    /// <inheritdoc />
    public override IEnumerable<ContractCase> FixedCases()
    {
        yield return Case("one episode", 1, 1, 1, 1, 0.0);
        yield return Case("a group of two with a tie", 2, 2, 1, 1, 0.0);
        yield return Case("a group of four, one step each", 3, 4, 1, 1, 0.0);
        yield return Case("three groups, kept texts", 4, 12, 3, 3, 0.5);
        yield return Case("every episode a group of its own, all kept", 5, 6, 6, 2, 1.0);
    }

    /// <inheritdoc />
    public override ContractCase RandomCase(Random random, bool large)
    {
        ArgumentNullException.ThrowIfNull(random);
        int episodes = random.Next(1, large ? 257 : 17);
        return Case("random", random.Next(), episodes, random.Next(1, episodes + 1), random.Next(1, large ? 9 : 4), random.NextDouble());
    }

    /// <inheritdoc />
    public override void Run(IPreferencePairSource implementation, ContractCase @case, CaseChecks checks)
    {
        ArgumentNullException.ThrowIfNull(implementation);
        ArgumentNullException.ThrowIfNull(@case);
        ArgumentNullException.ThrowIfNull(checks);
        var source = implementation;
        var episodes = Episodes(@case);
        checks.Check("a name and a summary", !string.IsNullOrWhiteSpace(source.Name) && !string.IsNullOrWhiteSpace(source.Summary),
            $"name \"{source.Name}\", summary \"{source.Summary}\"");
        checks.Check("its keys are its declared options'", source.Keys.SetEquals(source.Options.Keys), $"keys {string.Join(", ", source.Keys)}, options {source.Options.Keys}");

        var into = new List<PreferenceChoice> { Sentinel };
        int added = source.Choose(episodes, into, options);
        checks.Check("the choices are added after what the list held, which stays", into.Count == added + 1 && into[0] == Sentinel,
            $"returned {added}, the list holds {into.Count} (the first {(into.Count > 0 && into[0] == Sentinel ? "kept" : "changed")})");
        var choices = into.Skip(1).ToList();
        checks.Compare("every choice names an episode and a step there are", choices.Select(c => Named(c.Chosen, episodes) ?? Named(c.Rejected, episodes)).FirstOrDefault(d => d is not null));
        checks.Check("a choice's two answers differ", choices.All(c => c.Chosen != c.Rejected), "a choice prefers an answer to itself");
        if (source.NeedsGroups)
        {
            var outside = choices.FirstOrDefault(c => episodes[c.Chosen.Episode].Group != c.Group || episodes[c.Rejected.Episode].Group != c.Group);
            checks.Check("a source that needs groups pairs episodes of the choice's group", choices.All(c => episodes[c.Chosen.Episode].Group == c.Group && episodes[c.Rejected.Episode].Group == c.Group),
                $"episodes {outside.Chosen.Episode} and {outside.Rejected.Episode} for group {outside.Group}");
        }
        else
        {
            checks.Skip("a source that needs groups pairs episodes of the choice's group", "the source does not need groups");
        }

        checks.Compare("the same episodes give the same choices", Comparisons.Sequences(choices, Choose(source, episodes)));
        var parallel = new List<PreferenceChoice>[4];
        Parallel.For(0, parallel.Length, i => parallel[i] = Choose(source, episodes));
        checks.Compare("the same choices from several threads at once", parallel.Select(p => Comparisons.Sequences(choices, p)).FirstOrDefault(d => d is not null));
        checks.Check("no episode gives no choice", Choose(source, []).Count == 0, "chose from no episode");

        var unknown = new Dictionary<string, string>(StringComparer.Ordinal) { [UnknownOption] = "1" };
        foreach (var (key, value) in options ?? new Dictionary<string, string>())
        {
            unknown[key] = value;
        }

        try
        {
            source.Choose(episodes, [], unknown);
            checks.Fail("an option not in Keys is refused, naming it", $"chose with the option '{UnknownOption}'");
        }
        catch (ArgumentException e)
        {
            checks.Check("an option not in Keys is refused, naming it", e.Message.Contains(UnknownOption, StringComparison.Ordinal), $"\"{e.Message}\" does not name '{UnknownOption}'");
        }

        if (reference is not null)
        {
            checks.Compare("agrees with the reference", Comparisons.Sequences(Choose(reference, episodes), choices));
        }
        else
        {
            checks.Skip("agrees with the reference", "no reference given");
        }
    }

    private List<PreferenceChoice> Choose(IPreferencePairSource source, IReadOnlyList<Trajectory<int, int>> episodes)
    {
        var into = new List<PreferenceChoice>();
        source.Choose(episodes, into, options);
        return into;
    }

    // Null when the answer names an episode and a step of the list (a kept text one step); else what is wrong.
    private static string? Named(PreferenceAnswer answer, IReadOnlyList<Trajectory<int, int>> episodes)
    {
        if ((uint)answer.Episode >= (uint)episodes.Count)
        {
            return $"episode {answer.Episode} of {episodes.Count}";
        }

        int steps = episodes[answer.Episode].Steps;
        if (answer.Text is not null)
        {
            return answer.Step >= 0 && answer.Step < steps ? null : $"a kept text at step {answer.Step} of episode {answer.Episode}'s {steps}";
        }

        return answer.Step == PreferenceAnswer.WholeEpisode || (answer.Step >= 0 && answer.Step < steps) ? null : $"step {answer.Step} of episode {answer.Episode}'s {steps}";
    }

    // The case's episodes: in groups, each of 1 to the longest steps with an action record per step, returns of either
    // sign with ties, and a verdict with a kept text on the case's share of them.
    internal static IReadOnlyList<Trajectory<int, int>> Episodes(ContractCase @case)
    {
        var data = @case.Data;
        var random = new Random((int)data["seed"]!);
        int count = (int)data["episodes"]!, groups = (int)data["groups"]!, longest = (int)data["longest"]!;
        double kept = (double)data["kept"]!;
        string[] verdicts = [PreferenceInfoKeys.Rejected, PreferenceInfoKeys.Edited, PreferenceInfoKeys.Undone, PreferenceInfoKeys.Accepted];
        var episodes = new Trajectory<int, int>[count];
        for (int e = 0; e < count; e++)
        {
            int steps = random.Next(1, longest + 1);
            float[] rewards = [.. Enumerable.Range(0, steps).Select(_ => random.Next(4) switch { 0 => 0f, 1 => 1f, _ => (random.NextSingle() - 0.5f) * 4 })];
            EpisodeInfo? info = null;
            if (random.NextDouble() < kept)
            {
                info = new EpisodeInfo().Set(PreferenceInfoKeys.Kept, $"kept {e}").Set(PreferenceInfoKeys.Verdict, verdicts[random.Next(verdicts.Length)]);
                if (random.Next(2) == 0)
                {
                    info.Set(PreferenceInfoKeys.KeptStep, random.Next(steps));
                }
            }

            episodes[e] = new Trajectory<int, int>(e, e, [.. Enumerable.Range(0, steps + 1)], [.. Enumerable.Range(0, steps)],
                [.. Enumerable.Range(0, steps).Select(s => new ActionRecord([s], [-1f]))], rewards, terminated: true, truncated: false,
                group: (int)((long)e * groups / count), info: info);
        }

        return episodes;
    }

    private static ContractCase Case(string name, int seed, int episodes, int groups, int longest, double kept) =>
        new(FormattableString.Invariant($"{name} ({episodes} episodes in {groups} groups, up to {longest} steps, {kept:P0} kept)"),
            new JsonObject { ["seed"] = seed, ["episodes"] = episodes, ["groups"] = groups, ["longest"] = longest, ["kept"] = kept });
}
