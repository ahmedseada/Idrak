// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Diagnostics;
using System.Globalization;
using System.Text.Json.Nodes;
using Idrak.Abstraction.Reinforcement;
using Idrak.Abstraction.Testing;
using Idrak.Generation;
using Idrak.Models;
using Idrak.Reinforcement;
using Idrak.Text;
using Idrak.Text.Abstractions;
using Idrak.Text.Reinforcement;

// Craft's plan, phase R0: preference learning from episodes. Pairs from a rollout or recorded episodes (the library's
// IPreferencePairSources "group" and "verdict"), and how fast they are chosen.
internal abstract partial class DesktopTests : Tests
{
    private static readonly (string Name, Action<Device> Run)[] PreferenceGroup =
    [
        ("preference pairs: built-ins: group's pairs (the best and the worst of each group by return, the first on a tie, groups in number order, a margin to beat, ties and lone episodes skipped) and verdict's (the kept text against the sampled answer at the kept step or the last, an accepted answer skipped, the verdicts option, no verdict counted as rejected) equal hand-made ones; a kept step past the episode is refused naming it; the reference names parse", d => { CpuOnly(d); PreferenceBuiltIns(); }),
        ("preference pairs: bench: group and verdict choose 1000 rounds of 256 episodes allocating nothing after the warm-up", d => { CpuOnly(d); PreferenceBenchChecks(); }),
    ];

    // Hand-made episodes: returns as given (one step each unless a step count is given), groups as given, info as given.
    private static Trajectory<int, int> Episode(int index, float ret, int group, int steps = 1, EpisodeInfo? info = null) =>
        new(index, index, [.. Enumerable.Range(0, steps + 1)], [.. Enumerable.Range(0, steps)], [.. Enumerable.Range(0, steps).Select(s => new ActionRecord([s], [-1f]))],
            [ret, .. Enumerable.Repeat(0f, steps - 1)], terminated: true, truncated: false, group: group, info: info);

    private static List<PreferenceChoice> Chosen(IPreferencePairSource source, IReadOnlyList<Trajectory<int, int>> episodes)
    {
        var into = new List<PreferenceChoice>();
        source.Choose(episodes, into);
        return into;
    }

    private static void PreferenceBuiltIns()
    {
        var group = new GroupSource();
        Check(group.NeedsGroups && !new VerdictSource().NeedsGroups, "group needs groups, verdict does not");
        // Groups 3 (listed first), 1 and 2; a tie for the best in group 1 (episode 1 first), a tie for the worst in group 3
        // (episode 0 first), a group of one (4) and a group whose returns are all equal (2).
        var episodes = new[]
        {
            Episode(0, 0f, 3), Episode(1, 2f, 1), Episode(2, 0f, 3), Episode(3, 2f, 1), Episode(4, 5f, 4), Episode(5, -1f, 1), Episode(6, 1f, 2), Episode(7, 1f, 2),
            Episode(8, 0.75f, 3, steps: 3),
        };
        List<PreferenceChoice> expected =
        [
            new(PreferenceAnswer.Sampled(1), PreferenceAnswer.Sampled(5), 3f, 1),
            new(PreferenceAnswer.Sampled(8), PreferenceAnswer.Sampled(0), 0.75f, 3),
        ];
        Check(Comparisons.Sequences(expected, Chosen(group, episodes)) is null, $"group: {string.Join("; ", Chosen(group, episodes))}");
        Check(Comparisons.Sequences(expected[..1], Chosen(new GroupSource(0.75f), episodes)) is null,
            $"group with margin 0.75 (more than it): {string.Join("; ", Chosen(new GroupSource(0.75f), episodes))}");
        Check(Chosen(group, [Episode(0, 1f, 0)]).Count == 0 && Chosen(group, []).Count == 0, "one episode or none: no pair");

        // verdict: the kept text at the kept step (the last when not set), no verdict counted as rejected, accepted skipped.
        var verdict = new VerdictSource();
        EpisodeInfo Info(string? said, string kept, int? step = null)
        {
            var info = new EpisodeInfo().Set(PreferenceInfoKeys.Kept, kept);
            if (said is not null)
            {
                info.Set(PreferenceInfoKeys.Verdict, said);
            }

            return step is { } s ? info.Set(PreferenceInfoKeys.KeptStep, s) : info;
        }

        var recorded = new[]
        {
            Episode(0, 0f, 0, steps: 2, info: Info(PreferenceInfoKeys.Rejected, "mine")), Episode(1, 1f, 1, info: Info(PreferenceInfoKeys.Accepted, "same")),
            Episode(2, 0f, 2, steps: 3, info: Info("EDITED", "edited text", step: 1)), Episode(3, 0f, 3), Episode(4, 0f, 4, info: Info(null, "")),
            Episode(5, 0f, 5, info: Info(PreferenceInfoKeys.Undone, "before")),
        };
        List<PreferenceChoice> verdicts =
        [
            new(PreferenceAnswer.Kept(0, 1, "mine"), PreferenceAnswer.Sampled(0, 1), float.NaN, 0),
            new(PreferenceAnswer.Kept(2, 1, "edited text"), PreferenceAnswer.Sampled(2, 1), float.NaN, 2),
            new(PreferenceAnswer.Kept(4, 0, ""), PreferenceAnswer.Sampled(4, 0), float.NaN, 4),
            new(PreferenceAnswer.Kept(5, 0, "before"), PreferenceAnswer.Sampled(5, 0), float.NaN, 5),
        ];
        Check(Comparisons.Sequences(verdicts, Chosen(verdict, recorded)) is null, $"verdict: {string.Join("; ", Chosen(verdict, recorded))}");
        var onlyRejected = Chosen(new VerdictSource("rejected"), recorded);
        Check(Comparisons.Sequences([verdicts[0], verdicts[2]], onlyRejected) is null, $"verdicts=rejected (and none counted as rejected): {string.Join("; ", onlyRejected)}");
        ThrowsNaming<InvalidOperationException>(() => Chosen(verdict, [Episode(0, 0f, 0, steps: 2, info: Info(null, "x", step: 2))]), "a kept step past the episode",
            "kept_step", "step 2");
        Check(PreferenceReferences.Parse(" Base ") == "base" && PreferenceReferences.Parse("PREVIOUS") == "previous", "the reference names parse");
        ThrowsNaming<ArgumentException>(() => PreferenceReferences.Parse("teacher"), "an unknown reference", "'teacher'", "previous, base");
    }

    // A tiny chat model's policy on one prompt, warmed up on "yes" and "no" so the reward has something to choose.
    private static (LanguageModelPolicy Policy, PretrainedModel Model, IReadOnlyList<ChatMessage> Prompt) WarmPolicy(string folder, Device device, int steps = 200)
    {
        var model = PretrainedModel.Load(folder, new PretrainedOptions { Device = device });
        var session = new FineTuningSession(model, SessionOptions(1e-2f), totalSteps: steps);
        var policy = new LanguageModelPolicy(session, new LanguageModelPolicyOptions { MaxNewTokens = 6 });
        IReadOnlyList<ChatMessage> prompt = [new ChatMessage("user", "Q")];
        var warm = Enumerable.Range(0, 8).Select(i => AnswerSequence(model, prompt, i % 2 == 0 ? "yes" : "no")).ToList();
        for (int i = 0; i < 40; i++)
        {
            session.Step(warm);
        }

        return (policy, model, prompt);
    }

    /// <summary>What <see cref="PreferenceBench"/> measured.</summary>
    internal sealed record PreferenceBenchResult(string Source, int Episodes, int Rounds, int PairsPerRound, double ColdMicroseconds, double WarmMicroseconds, double PairsPerSecond,
        long WarmBytesPerRound)
    {
        public string Report() => string.Create(CultureInfo.InvariantCulture,
            $"  {Source}: {Episodes} episodes, {PairsPerRound} pairs a round; cold {ColdMicroseconds:F1} µs, warm {WarmMicroseconds:F2} µs a round, {PairsPerSecond:N0} pairs/s, {WarmBytesPerRound} bytes a warm round");
    }

    // Choosing pairs, timed: a cold round (the first), then `rounds` warm ones; the managed bytes of the warm rounds.
    internal static PreferenceBenchResult PreferenceBench(string name, IPreferencePairSource source, int episodes, int rounds, int seed)
    {
        var drawn = PreferencePairSourceSuiteCase(seed, episodes);
        var into = new List<PreferenceChoice>(episodes);
        long start = Stopwatch.GetTimestamp();
        int pairs = source.Choose(drawn, into);
        double cold = Stopwatch.GetElapsedTime(start).TotalMicroseconds;
        for (int i = 0; i < 50; i++)
        {
            into.Clear();
            source.Choose(drawn, into);
        }

        long bytes = GC.GetAllocatedBytesForCurrentThread();
        start = Stopwatch.GetTimestamp();
        for (int i = 0; i < rounds; i++)
        {
            into.Clear();
            source.Choose(drawn, into);
        }

        double warm = Stopwatch.GetElapsedTime(start).TotalMicroseconds / rounds;
        bytes = GC.GetAllocatedBytesForCurrentThread() - bytes;
        return new PreferenceBenchResult(name, episodes, rounds, pairs, cold, warm, pairs / Math.Max(1e-9, warm * 1e-6), bytes / rounds);
    }

    // Episodes in groups of 8 of one or two steps, returns of either sign with ties, about half with a verdict and a kept text.
    private static IReadOnlyList<Trajectory<int, int>> PreferencePairSourceSuiteCase(int seed, int episodes)
    {
        var random = new Random(seed);
        string[] verdicts = [PreferenceInfoKeys.Rejected, PreferenceInfoKeys.Edited, PreferenceInfoKeys.Undone, PreferenceInfoKeys.Accepted];
        return [.. Enumerable.Range(0, episodes).Select(e =>
        {
            int steps = random.Next(1, 3);
            var info = random.Next(2) == 0 ? new EpisodeInfo().Set(PreferenceInfoKeys.Kept, "kept").Set(PreferenceInfoKeys.Verdict, verdicts[random.Next(verdicts.Length)]) : null;
            return Episode(e, random.Next(3) switch { 0 => 0f, 1 => 1f, _ => random.NextSingle() }, e / 8, steps, info);
        })];
    }

    private static void PreferenceBenchChecks()
    {
        foreach (var (name, source) in new (string, IPreferencePairSource)[] { ("group", new GroupSource()), ("verdict", new VerdictSource()) })
        {
            var measured = PreferenceBench(name, source, 256, 1000, seed: 3);
            Check(measured.PairsPerRound > 0 && measured.WarmBytesPerRound == 0, $"{name}: {measured.Report()}");
        }
    }
}
