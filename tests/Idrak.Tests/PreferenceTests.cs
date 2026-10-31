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

// Craft's plan, phase R0: preference learning from episodes. Pairs from a rollout or recorded episodes
// (IPreferencePairSource in PreferencePairSources: "group" and "verdict"), the preference losses by name
// (PreferenceLosses over FineTuningLosses), iterative preference training (ReinforcementTrainer with
// ReinforcementOptions.Pairs over an IPreferencePolicy: Idrak.Text's language-model policy), idrak reinforce --pairs,
// the suite, and the bench (--bench-preference).
internal abstract partial class DesktopTests : Tests
{
    private static readonly (string Name, Action<Device> Run)[] PreferenceGroup =
    [
        ("preference pairs: contract: group and verdict are the library's in PreferencePairSources and dpo, orpo, simpo in PreferenceLosses; a name nobody registered is refused as not registered; an app's source over group is its origin, falls back to the library's under FallBack (its additions taken back) and unregisters; a loss by name is its factory's (same values), an unknown option or a bad value refused naming it", d => { if (d == Device.Cpu) PreferenceContract(); }),
        ("preference pairs: built-ins: group's pairs (the best and the worst of each group by return, the first on a tie, groups in number order, a margin to beat, ties and lone episodes skipped) and verdict's (the kept text against the sampled answer at the kept step or the last, an accepted answer skipped, the verdicts option, no verdict counted as rejected) equal hand-made ones; a kept step past the episode is refused naming it; the reference names parse", d => { if (d == Device.Cpu) PreferenceBuiltIns(); }),
        ("preference pairs: suite: PreferencePairSourceSuite passes group (with and without a margin), verdict and an app's copy of group against the library's, and fails a source that pairs across groups and one whose choices change from call to call", d => { if (d == Device.Cpu) PreferenceSuiteKit(); }),
        ("preference pairs: training: on a tiny random decoder, pairs built from each scored rollout of the prompts task (group, equal to the hand-made best and worst; trained ids the sampled ids) train with DPO against the base model, the loss falling and the chosen answer's margin over the rejected rising on the first rollout's pairs; against the previous adapter every round's first update is log 2 and its last below, with flat device memory over the rounds", PreferenceTrainingRounds),
        ("preference pairs: verdict on a language model: a rejected reply against the text kept instead gives the pair the encoder says (the prompt ids, the kept text with the end of the turn; the sampled ids), DPO on it lowers its loss from log 2 and raises the kept text's margin; PreferencePairs.Build gives the same pairs; orpo and simpo by name step without a reference pass; a policy that is not an IPreferencePolicy is refused", PreferenceVerdictPolicy),
        ("preference pairs: cli: idrak reinforce --pairs group --loss dpo trains a tiny model on the prompts environment (its JSON names the pairs and the loss, a line per iteration with its pairs, the adapter written); --list shows the pair sources and preference losses; an unregistered source or loss, --loss without --pairs, a bad --reference and --pairs with distill exit with 2 before a model loads", d => { if (d == Device.Cpu) PreferenceCli(); }),
        ("preference pairs: bench: group and verdict choose 1000 rounds of 256 episodes allocating nothing after the warm-up; 12 preference rounds on a tiny decoder hold flat device and managed memory", d => { if (d == Device.Cpu) PreferenceBenchChecks(); }),
    ];

    // Hand-made episodes: returns as given (one step each unless a step count is given), groups as given, info as given.
    private static Trajectory<int, int> Episode(int index, float ret, int group, int steps = 1, EpisodeInfo? info = null) =>
        new(index, index, [.. Enumerable.Range(0, steps + 1)], [.. Enumerable.Range(0, steps)], [.. Enumerable.Range(0, steps).Select(s => new ActionRecord([s], [-1f]))],
            [ret, .. Enumerable.Repeat(0f, steps - 1)], terminated: true, truncated: false, group: group, info: info);

    private static List<PreferenceChoice> Chosen(IPreferencePairSource source, IReadOnlyList<Trajectory<int, int>> episodes, IReadOnlyDictionary<string, string>? options = null)
    {
        var into = new List<PreferenceChoice>();
        source.Choose(episodes, into, options);
        return into;
    }

    private static void PreferenceContract()
    {
        Check(PreferencePairSources.Names.Take(2).SequenceEqual(["group", "verdict"]) && PreferencePairSources.Origin("GROUP") == Idrak.Abstraction.Overrides.Library
              && PreferencePairSources.Origin("verdict") == Idrak.Abstraction.Overrides.Library, $"the library's sources: {string.Join(", ", PreferencePairSources.Names)}");
        ThrowsNaming<NotRegisteredException>(() => PreferencePairSources.Get("no-such-source"), "an unregistered source", "not registered", "PreferencePairSources.Register");
        Check(PreferenceLosses.Names.Take(3).SequenceEqual(["dpo", "orpo", "simpo"]) && PreferenceLosses.Origin("DPO") == Idrak.Abstraction.Overrides.Library,
            $"the library's preference losses: {string.Join(", ", PreferenceLosses.Names)}");
        ThrowsNaming<NotRegisteredException>(() => FineTuningLosses.Preference("no-such-loss"), "an unregistered loss", "not registered", "PreferenceLosses.Register");

        // An app's source over group: its origin; under FallBack a failure gives the library's choices, its own taken back.
        var episodes = new[] { Episode(0, 1f, 0), Episode(1, 0f, 0) };
        var app = new FailingSource();
        PreferencePairSources.Register(app);
        try
        {
            Check(PreferencePairSources.Origin("group") == typeof(DesktopTests).Assembly.GetName().Name, $"the app's source is its origin: {PreferencePairSources.Origin("group")}");
            ThrowsNaming<InvalidOperationException>(() => Chosen(PreferencePairSources.Get("group"), episodes), "the app's failing source under Throw", "test source fails");
            PreferencePairSources.SetPolicy("group", SlotPolicy.FallBack);
            var chosen = Chosen(PreferencePairSources.Get("group"), episodes);
            Check(chosen.Count == 1 && chosen[0] == new PreferenceChoice(PreferenceAnswer.Sampled(0), PreferenceAnswer.Sampled(1), 1f, 0),
                $"FallBack: the library's choice alone ({string.Join("; ", chosen)})");
        }
        finally
        {
            PreferencePairSources.SetPolicy("group", SlotPolicy.Throw);
            PreferencePairSources.Unregister("group");
        }

        Check(PreferencePairSources.Origin("group") == Idrak.Abstraction.Overrides.Library && PreferencePairSources.Get("group") is not FailingSource, "unregistering brings the library's back");

        // A loss by name is its factory's: the same value on fixed log-probabilities.
        float[] policy = [-1.2f, -0.4f, -2.0f, -0.9f, -1.1f, -0.3f], reference = [-1.0f, -0.6f, -1.5f, -1.0f, -0.8f, -0.5f];
        float Value(IFineTuningLoss loss)
        {
            var input = new FineTuningLossInput(Tensor.From(policy, [6], Device.Cpu), [2, 1, 2, 1], Tensor.From(reference, [6], Device.Cpu), pairs: true);
            return loss.Compute(input).Item();
        }

        var options = new Dictionary<string, string> { ["beta"] = "0.3", ["label-smoothing"] = "0.1" };
        AssertClose([Value(FineTuningLosses.Dpo(0.3f, 0.1f))], [Value(FineTuningLosses.Preference("DPO", options))], 0f, "dpo by name");
        AssertClose([Value(FineTuningLosses.Orpo(0.25f))], [Value(FineTuningLosses.Preference("orpo", new Dictionary<string, string> { ["lambda"] = "0.25" }))], 0f, "orpo by name");
        AssertClose([Value(FineTuningLosses.SimPo(2.5f, 0.5f))], [Value(FineTuningLosses.Preference("simpo", new Dictionary<string, string> { ["beta"] = "2.5", ["gamma"] = "0.5" }))], 0f,
            "simpo by name");
        AssertClose([Value(FineTuningLosses.Dpo())], [Value(PreferenceLosses.Get("dpo").Create())], 0f, "dpo's defaults are the factory's");
        Check(FineTuningLosses.Preference("dpo").NeedsReference && !FineTuningLosses.Preference("orpo").NeedsReference && FineTuningLosses.Preference("simpo").NeedsPairs,
            "what each needs");
        ThrowsNaming<ArgumentException>(() => FineTuningLosses.Preference("dpo", new Dictionary<string, string> { ["gamma"] = "1" }), "an option dpo does not take", "'gamma'", "beta");
        ThrowsNaming<ArgumentException>(() => FineTuningLosses.Preference("dpo", new Dictionary<string, string> { ["label-smoothing"] = "0.5" }), "label smoothing of 0.5",
            "'label-smoothing'", "up to (not including) 0.5");
        ThrowsNaming<ArgumentException>(() => Chosen(PreferencePairSources.Get("group"), episodes, new Dictionary<string, string> { ["margin"] = "-1" }), "a negative margin",
            "'margin'", "at least 0");
    }

    // An app's "group" that adds a choice, then fails.
    private sealed class FailingSource : IPreferencePairSource
    {
        public string Name => "group";

        public string Summary => "a test source that fails after adding a choice";

        public OptionSchema Options { get; } = new("test source", OptionSpec.Number("margin", 0f, 0f));

        public IReadOnlySet<string> Keys => Options.Keys;

        public bool NeedsGroups => true;

        public int Choose<TObservation, TAction>(IReadOnlyList<Trajectory<TObservation, TAction>> episodes, List<PreferenceChoice> into, IReadOnlyDictionary<string, string>? options = null)
        {
            into.Add(new PreferenceChoice(PreferenceAnswer.Sampled(5), PreferenceAnswer.Sampled(6), 9f, 9));
            throw new InvalidOperationException("The test source fails.");
        }
    }

    private static void PreferenceBuiltIns()
    {
        var group = PreferencePairSources.Get("group");
        Check(group.NeedsGroups && !PreferencePairSources.Get("verdict").NeedsGroups, "group needs groups, verdict does not");
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
        Check(Comparisons.Sequences(expected[..1], Chosen(group, episodes, new Dictionary<string, string> { ["margin"] = "0.75" })) is null,
            $"group with margin 0.75 (more than it): {string.Join("; ", Chosen(group, episodes, new Dictionary<string, string> { ["margin"] = "0.75" }))}");
        Check(Chosen(group, [Episode(0, 1f, 0)]).Count == 0 && Chosen(group, []).Count == 0, "one episode or none: no pair");

        // verdict: the kept text at the kept step (the last when not set), no verdict counted as rejected, accepted skipped.
        var verdict = PreferencePairSources.Get("verdict");
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
        var onlyRejected = Chosen(verdict, recorded, new Dictionary<string, string> { ["verdicts"] = "rejected" });
        Check(Comparisons.Sequences([verdicts[0], verdicts[2]], onlyRejected) is null, $"verdicts=rejected (and none counted as rejected): {string.Join("; ", onlyRejected)}");
        ThrowsNaming<InvalidOperationException>(() => Chosen(verdict, [Episode(0, 0f, 0, steps: 2, info: Info(null, "x", step: 2))]), "a kept step past the episode",
            "kept_step", "step 2");
        Check(PreferenceReferences.Parse(" Base ") == "base" && PreferenceReferences.Parse("PREVIOUS") == "previous", "the reference names parse");
        ThrowsNaming<ArgumentException>(() => PreferenceReferences.Parse("teacher"), "an unknown reference", "'teacher'", "previous, base");
    }

    private static void PreferenceSuiteKit()
    {
        var options = new ContractCheckOptions { RandomCases = 12 };
        Conformance.CheckPreferencePairSource(PreferencePairSources.Get("group"), options: options).ThrowIfFailed();
        Conformance.CheckPreferencePairSource(PreferencePairSources.Get("group"), new Dictionary<string, string> { ["margin"] = "0.5" }, options: options with { Large = true, RandomCases = 4 })
            .ThrowIfFailed();
        Conformance.CheckPreferencePairSource(PreferencePairSources.Get("verdict"), options: options).ThrowIfFailed();
        Conformance.CheckPreferencePairSource(new Copied(PreferencePairSources.Get("group"), across: false, drifting: false), reference: PreferencePairSources.Get("group"), options: options)
            .ThrowIfFailed();
        var across = Conformance.CheckPreferencePairSource(new Copied(PreferencePairSources.Get("group"), across: true, drifting: false), options: options);
        Check(!across.Passed && across.ToString().Contains("pairs episodes of the choice's group", StringComparison.Ordinal), $"a source pairing across groups fails:\n{across}");
        var drifting = Conformance.CheckPreferencePairSource(new Copied(PreferencePairSources.Get("group"), across: false, drifting: true), options: options);
        Check(!drifting.Passed && drifting.ToString().Contains("the same episodes give the same choices", StringComparison.Ordinal), $"a source whose choices change fails:\n{drifting}");
    }

    // A source over another: as it is, with its choices' groups moved (across), or with a choice's margin counting calls (drifting).
    private sealed class Copied(IPreferencePairSource inner, bool across, bool drifting) : IPreferencePairSource
    {
        private int _calls;

        public string Name => "suite-copy";

        public string Summary => "a copy of a source, for the suite's tests";

        public OptionSchema Options => inner.Options;

        public IReadOnlySet<string> Keys => inner.Keys;

        public bool NeedsGroups => inner.NeedsGroups;

        public int Choose<TObservation, TAction>(IReadOnlyList<Trajectory<TObservation, TAction>> episodes, List<PreferenceChoice> into, IReadOnlyDictionary<string, string>? options = null)
        {
            int before = into.Count, added = inner.Choose(episodes, into, options);
            int call = Interlocked.Increment(ref _calls);
            for (int i = before; i < into.Count; i++)
            {
                into[i] = into[i] with { Group = across ? into[i].Group + 1000 : into[i].Group, Margin = drifting ? call : into[i].Margin };
            }

            return added;
        }
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

    private static ReinforcementOptions PreferenceRun(string reference, int iterations, int epochs, int seed) => new()
    {
        Pairs = "group", PreferenceLoss = "dpo", PreferenceReference = reference, GroupSize = 8, EpisodesPerIteration = 16, Iterations = iterations, Epochs = epochs,
        Seed = seed, Environments = 1, ParallelSteps = false,
        EnvironmentOptions = new Dictionary<string, string> { ["reward"] = "metric", ["metric.metric"] = "exact" },
    };

    // The mean log p(chosen) − log p(rejected) of `pairs` under the policy now.
    private static double Margin(FineTuningSession session, IReadOnlyList<PreferencePair> pairs)
    {
        var values = session.LogProbabilities([.. pairs.SelectMany(p => new[] { p.Chosen, p.Rejected })]);
        double sum = 0;
        for (int p = 0; p < pairs.Count; p++)
        {
            sum += values[2 * p].Sum(v => (double)v) - values[2 * p + 1].Sum(v => (double)v);
        }

        return sum / Math.Max(1, pairs.Count);
    }

    private static void PreferenceTrainingRounds(Device device)
    {
        string folder = WriteChatModelWith(TinyChatSpec(), ChatMlTemplate);
        var kind = new PromptsEnvironmentKind([new PromptTask([new ChatMessage("user", "Q")], "yes")], name: "preference-test-prompts");
        try
        {
            // Against the base model: the loss falls round after round, and the first rollout's pairs lean further.
            {
                var (policy, model, _) = WarmPolicy(folder, device);
                using var owned = model;
                using var _p = policy;
                List<PreferencePair> first = [];
                double before = double.NaN;
                var trainer = new ReinforcementTrainer<ChatObservation, ChatMessage>(kind, policy, PreferenceRun(PreferenceReferences.Base, 12, 2, seed: 5))
                {
                    OnBatch = (iteration, batch) =>
                    {
                        // The pairs equal the hand-made ones: in each group the best return (the first on a tie) over the worst.
                        var choices = new List<PreferenceChoice>();
                        PreferencePairSources.Get("group").Choose(batch.Trajectories, choices);
                        var expected = new List<PreferenceChoice>();
                        foreach (var g in Enumerable.Range(0, batch.Episodes).GroupBy(e => batch.Trajectories[e].Group).OrderBy(g => g.Key))
                        {
                            int best = g.OrderByDescending(e => batch.Trajectories[e].Return).ThenBy(e => e).First();
                            int worst = g.OrderBy(e => batch.Trajectories[e].Return).ThenBy(e => e).First();
                            float gap = batch.Trajectories[best].Return - batch.Trajectories[worst].Return;
                            if (gap > 0f)
                            {
                                expected.Add(new PreferenceChoice(PreferenceAnswer.Sampled(best), PreferenceAnswer.Sampled(worst), gap, g.Key));
                            }
                        }

                        Check(Comparisons.Sequences(expected, choices) is null, $"iteration {iteration}: group's pairs are the hand-made ones: {string.Join("; ", choices)} | {string.Join("; ", expected)}");
                        if (first.Count == 0 && choices.Count > 0)
                        {
                            PreferencePairs.Encode(batch.Trajectories, choices, null, first);
                            for (int p = 0; p < first.Count; p++)
                            {
                                var record = batch.Trajectories[choices[p].Chosen.Episode].Records[0];
                                Check(first[p].Chosen.Tokens.SequenceEqual(record.Context!.Concat(record.Tokens)) && first[p].Chosen.Trained.Count(t => t) == record.Tokens.Length,
                                    $"pair {p}: the chosen answer trains on its prompt and sampled ids, the reply alone trained");
                            }

                            before = Margin(policy.Session, first);
                        }
                    },
                };
                var result = trainer.Train();
                var losses = result.History.Where(h => h.Updates > 0).Select(h => h.Loss).ToList();
                Check(result.History.Count == 12 && losses.Count >= 6 && result.History.All(h => h.Metrics.ContainsKey("pairs")),
                    $"12 rounds, at least 6 with pairs: {string.Join(", ", result.History.Select(h => $"{h.Metrics.GetValueOrDefault("pairs")} pairs, {h.Updates} updates"))}");
                double early = losses.Take(3).Average(), late = losses.TakeLast(3).Average();
                Check(late < early && early < Math.Log(2) + 0.05, string.Create(CultureInfo.InvariantCulture,
                    $"DPO against the base model falls ({early:F4} to {late:F4}): {string.Join(", ", losses.Select(l => l.ToString("F4", CultureInfo.InvariantCulture)))}"));
                double after = Margin(policy.Session, first);
                Check(first.Count > 0 && after > before + 0.5, string.Create(CultureInfo.InvariantCulture,
                    $"the first rollout's {first.Count} pairs: the chosen answer's log-probability margin over the rejected rises ({before:F3} to {after:F3})"));
            }

            // Against the previous adapter (iterative): the first update of every round is log 2 (the reference is the
            // policy as the round began), its last below; device memory held does not grow over the rounds.
            {
                var recorded = new List<float>();
                var loss = new RecordingPreference("preference-test-dpo", PreferenceLosses.Get("dpo"), recorded);
                PreferenceLosses.Register(loss);
                try
                {
                    var (policy, model, _) = WarmPolicy(folder, device);
                    using var owned = model;
                    using var _p = policy;
                    long heldAfterSecond = 0, heldAtEnd = 0;
                    var rounds = new List<(int Updates, int Pairs)>();
                    var trainer = new ReinforcementTrainer<ChatObservation, ChatMessage>(kind, policy,
                        PreferenceRun(PreferenceReferences.Previous, 10, 3, seed: 6) with { PreferenceLoss = loss.Name })
                    {
                        OnIteration = summary =>
                        {
                            long held = model.Device.Backend.GetMemoryUsage().InUse;
                            heldAfterSecond = summary.Iteration == 2 ? held : heldAfterSecond;
                            heldAtEnd = held;
                            rounds.Add((summary.Updates, (int)summary.Metrics["pairs"]));
                            Check(summary.Updates == 0 || double.IsFinite(summary.Metrics.GetValueOrDefault("margin", double.NaN)),
                                $"iteration {summary.Iteration}: the margin at the round's start is reported against the previous adapter");
                        },
                    };
                    trainer.Train();
                    Check(recorded.Count == rounds.Sum(r => r.Updates) && rounds.Count(r => r.Updates == 3) >= 4,
                        $"one batch a step, three steps a round with pairs: {recorded.Count} losses for {string.Join(", ", rounds)}");
                    int at = 0;
                    var firsts = new List<float>();
                    var lasts = new List<float>();
                    foreach (var (updates, _) in rounds.Where(r => r.Updates > 0))
                    {
                        firsts.Add(recorded[at]);
                        lasts.Add(recorded[at + updates - 1]);
                        at += updates;
                    }

                    Check(firsts.All(f => Math.Abs(f - MathF.Log(2f)) < 2e-3f), $"every round's first update is log 2 against the previous adapter: {string.Join(", ", firsts)}");
                    Check(lasts.Average() < MathF.Log(2f) - 1e-3f, $"and falls within the round: {string.Join(", ", lasts)}");
                    Check(heldAtEnd <= heldAfterSecond + (1L << 20), $"device memory held after round 2 ({heldAfterSecond} bytes) does not grow by the last ({heldAtEnd} bytes)");
                }
                finally
                {
                    PreferenceLosses.Unregister(loss.Name);
                }
            }
        }
        finally
        {
            Directory.Delete(folder, true);
        }
    }

    // A preference loss over a library one that records the value of each batch.
    private sealed class RecordingPreference(string name, IPreferenceLoss inner, List<float> values) : IPreferenceLoss
    {
        public string Name => name;

        public string Summary => "records " + inner.Name;

        public OptionSchema Options => inner.Options;

        public IReadOnlySet<string> Keys => inner.Keys;

        public IFineTuningLoss Create(IReadOnlyDictionary<string, string>? options = null)
        {
            var loss = inner.Create(options);
            return FineTuningLosses.Of(name, input =>
            {
                var value = loss.Compute(input);
                values.Add(value.Item());
                return value;
            }, needsPairs: true, needsReference: loss.NeedsReference);
        }
    }

    private static void PreferenceVerdictPolicy(Device device)
    {
        string folder = WriteChatModelWith(TinyChatSpec(), ChatMlTemplate);
        try
        {
            var (policy, model, prompt) = WarmPolicy(folder, device);
            using var owned = model;
            using var _p = policy;
            var observation = new ChatObservation(prompt);
            var acted = policy.ActAsync([observation], new ActionSampling { Temperature = 1f, Seed = 3 }).AsTask().GetAwaiter().GetResult();
            var record = acted.Records[0];
            var info = new EpisodeInfo().Set(PreferenceInfoKeys.Verdict, PreferenceInfoKeys.Rejected).Set(PreferenceInfoKeys.Kept, "maybe so");
            var episode = new Trajectory<ChatObservation, ChatMessage>(0, 0, [observation, observation], [acted.Actions[0]], [record], [0f], terminated: true, truncated: false, info: info);
            using var batch = new RolloutBatch<ChatObservation, ChatMessage>([episode]);
            var choices = new List<PreferenceChoice>();
            PreferencePairSources.Get("verdict").Choose(batch.Trajectories, choices);
            var encoder = new ChatTranscriptEncoder(model.JinjaTemplate!, model.Tokenizer!);
            var pairs = new List<PreferencePair>();
            PreferencePairs.Encode(batch.Trajectories, choices, encoder, pairs);
            var built = PreferencePairs.Build(batch.Trajectories, "verdict", encoder);
            Check(built.Count == 1 && built[0].Chosen.Tokens.SequenceEqual(pairs[0].Chosen.Tokens) && built[0].Rejected.Tokens.SequenceEqual(pairs[0].Rejected.Tokens),
                "PreferencePairs.Build chooses and encodes as the two steps do (recorded episodes, trained offline)");
            int[] kept = [.. model.Tokenizer!.Encode("maybe so<|im_end|>")];
            Check(choices.Count == 1 && pairs.Count == 1 && encoder.AssistantEnd == "<|im_end|>"
                  && pairs[0].Chosen.Tokens.SequenceEqual(record.Context!.Concat(kept)) && pairs[0].Rejected.Tokens.SequenceEqual(record.Context!.Concat(record.Tokens))
                  && pairs[0].Chosen.Trained.Count(t => t) == kept.Length && pairs[0].Rejected.Trained.Count(t => t) == record.Tokens.Length,
                "the verdict's pair: the prompt ids then the kept text and the end of the turn (chosen), the sampled ids (rejected), the answers alone trained");

            double before = Margin(policy.Session, pairs);
            var losses = new List<float>();
            for (int i = 0; i < 12; i++)
            {
                var stats = policy.UpdatePreferences(batch, choices, "dpo", new Dictionary<string, string> { ["beta"] = "0.5" });
                losses.Add(stats.Loss);
                Check(stats.Metrics!["pairs"] == 1 && stats.Episodes == 1 && stats.Units == kept.Length + record.Tokens.Length, $"update {i}: one pair of one episode");
            }

            double after = Margin(policy.Session, pairs);
            Check(Math.Abs(losses[0] - MathF.Log(2f)) < 2e-3f && losses[^1] < losses[0] - 0.05f, $"DPO against the adapter as the round began: {string.Join(", ", losses)}");
            Check(after > before + 0.5, string.Create(CultureInfo.InvariantCulture, $"the kept text's margin over the rejected reply rises ({before:F3} to {after:F3})"));

            foreach (string name in new[] { "orpo", "simpo" })
            {
                using var other = new RolloutBatch<ChatObservation, ChatMessage>([episode]);
                var stats = policy.UpdatePreferences(other, choices, name);
                Check(float.IsFinite(stats.Loss) && !stats.Metrics!.ContainsKey("margin"), $"{name} by name steps, without a reference pass (no margin): {stats}");
            }

            ThrowsNaming<NotRegisteredException>(() => policy.CheckPreferences("nope"), "an unregistered preference loss", "not registered");
            var prompts = new PromptsEnvironmentKind([new PromptTask(prompt, "yes")], name: "preference-test-refused");
            ThrowsNaming<NotSupportedException>(() => ReinforcementTrainer.Train(prompts, new NotPreferring(), new ReinforcementOptions { Pairs = "group", Iterations = 1 }),
                "a policy that is not an IPreferencePolicy", "IPreferencePolicy", nameof(NotPreferring));
        }
        finally
        {
            Directory.Delete(folder, true);
        }
    }

    // A chat policy that learns no preferences.
    private sealed class NotPreferring : IPolicy<ChatObservation, ChatMessage>
    {
        public ValueTask<PolicyActions<ChatMessage>> ActAsync(IReadOnlyList<ChatObservation> observations, ActionSampling sampling, CancellationToken cancellationToken = default) =>
            ValueTask.FromResult(new PolicyActions<ChatMessage>([.. observations.Select(_ => new ChatMessage("assistant", ""))], [.. observations.Select(_ => new ActionRecord([0], [0f]))]));

        public UpdateStats Update(RolloutBatch<ChatObservation, ChatMessage> batch, IPolicyLoss loss, IReadOnlyList<int>? episodes = null, IReadOnlyDictionary<string, string>? lossOptions = null) =>
            new(0f, 0, 0);

        public void Save(string path)
        {
        }

        public void Load(string path)
        {
        }

        public void Dispose()
        {
        }
    }

    private static void PreferenceCli()
    {
        var listed = TrainCliJson("reinforce", "--list");
        HashSet<string> Names(string section) => [.. listed[section]!.AsArray().Select(e => (string)e!["name"]!)];
        Check(Names("pairs").SetEquals(PreferencePairSources.Names) && Names("preference-losses").SetEquals(PreferenceLosses.Names) && Names("pairs").Contains("verdict"),
            $"--list: {listed["pairs"]} {listed["preference-losses"]}");
        string folder = TempFolder();
        string model = WriteChatModelWith(TinyChatSpec(), ChatMlTemplate);
        try
        {
            string data = Path.Combine(folder, "prompts.jsonl");
            File.WriteAllLines(data, ["{\"question\": \"Q\", \"answer\": \"yes\"}", "{\"question\": \"R\", \"answer\": \"no\"}"]);
            string[] missing = ["reinforce", "no-such-model-folder", "--data", data, "-o", Path.Combine(folder, "x")];
            void Usage(string what, string expected, params string[] args)
            {
                var (code, _, error) = TrainCli(args);
                Check(code == 2 && error.Contains(expected, StringComparison.Ordinal), $"{what}: exit {code}, '{expected}' in: {error}");
            }

            Usage("an unregistered pair source", "--pairs nope", [.. missing, "--pairs", "nope"]);
            Usage("an unregistered preference loss", "--loss nope", [.. missing, "--pairs", "group", "--loss", "nope"]);
            Usage("--loss without --pairs", "give --pairs", [.. missing, "--loss", "dpo"]);
            Usage("a bad --reference", "--reference teacher", [.. missing, "--pairs", "group", "--reference", "teacher"]);
            Usage("an option the source does not take", "'beta'", [.. missing, "--pairs", "group", "--pair-option", "beta=1"]);
            Usage("--pairs with distill", "give one of them", [.. missing, "--pairs", "group", "--algorithm", "distill", "--teacher", "t"]);

            string output = Path.Combine(folder, "dpo");
            JsonObject result = null!;
            WithEnvironment(new Dictionary<string, string?> { ["IDRAK_RL_ENVIRONMENTS"] = "1", ["IDRAK_RL_PARALLEL_STEPS"] = "off" }, () =>
                result = TrainCliJson(["reinforce", model, "--env", "prompts", "--data", data, "--reward", "cer", "--pairs", "group", "--loss", "dpo", "--loss-option", "beta=0.2",
                    "--group", "4", "--episodes", "8", "--iterations", "2", "--epochs", "2", "--seed", "3", "-o", output,
                    "--rank", "4", "--alpha", "8", "--lr", "1e-2", "--batch-tokens", "256", "--max-length", "64", "--max-new", "4", "-d", "cpu", "--cache", folder]));
            var history = result["history"]!.AsArray();
            Check((int?)result["exitCode"] == 0 && (string?)result["pairs"] == "group" && (string?)result["loss"] == "dpo" && history.Count == 2,
                $"two preference rounds: {result}");
            Check(result["output"]!.AsArray().Any(l => ((string?)l)!.StartsWith("reinforce: dpo on group pairs against the previous adapter", StringComparison.Ordinal))
                  && result["output"]!.AsArray().Any(l => ((string?)l)!.StartsWith("iteration 1/2: reward", StringComparison.Ordinal) && ((string?)l)!.Contains(" pairs", StringComparison.Ordinal)),
                $"the run's lines name the pairs: {result["output"]}");
            Check(File.Exists(Path.Combine(output, "adapter_model.safetensors")), "the adapter is written");
        }
        finally
        {
            Directory.Delete(folder, true);
            Directory.Delete(model, true);
        }
    }

    /// <summary>What <see cref="PreferenceBench"/> measured.</summary>
    internal sealed record PreferenceBenchResult(string Source, int Episodes, int Rounds, int PairsPerRound, double ColdMicroseconds, double WarmMicroseconds, double PairsPerSecond,
        long WarmBytesPerRound)
    {
        public string Report() => string.Create(CultureInfo.InvariantCulture,
            $"  {Source}: {Episodes} episodes, {PairsPerRound} pairs a round; cold {ColdMicroseconds:F1} µs, warm {WarmMicroseconds:F2} µs a round, {PairsPerSecond:N0} pairs/s, {WarmBytesPerRound} bytes a warm round");
    }

    // Choosing pairs, timed: a cold round (the first), then `rounds` warm ones; the managed bytes of the warm rounds.
    internal static PreferenceBenchResult PreferenceBench(string name, int episodes, int rounds, int seed)
    {
        var source = PreferencePairSources.Get(name);
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

    /// <summary>What a run of preference rounds measured: each round's rollout and update times and the memory after it.</summary>
    internal sealed record PreferenceRoundsResult(IReadOnlyList<IterationSummary> History, long[] DeviceHeld, long[] ManagedHeld)
    {
        public string Report()
        {
            var updates = History.Select(h => h.UpdateTime.TotalMilliseconds).ToList();
            var rollouts = History.Select(h => h.RolloutTime.TotalMilliseconds).ToList();
            double Median(List<double> values) => values.Count == 0 ? double.NaN : values.Order().ElementAt(values.Count / 2);
            var warmUpdates = updates.Skip(1).ToList();
            var warmRollouts = rollouts.Skip(1).ToList();
            return string.Create(CultureInfo.InvariantCulture,
                $"  rounds: {History.Count}, pairs a round {History.Average(h => h.Metrics.GetValueOrDefault("pairs")):F1}; update cold {updates[0]:F0} ms, warm median {Median(warmUpdates):F0} ms; "
                + $"rollout cold {rollouts[0]:F0} ms, warm median {Median(warmRollouts):F0} ms\n"
                + $"  device memory held after round 2: {DeviceHeld[Math.Min(1, DeviceHeld.Length - 1)]:N0} bytes, after the last: {DeviceHeld[^1]:N0}; "
                + $"managed after round 2: {ManagedHeld[Math.Min(1, ManagedHeld.Length - 1)]:N0}, after the last: {ManagedHeld[^1]:N0}\n"
                + $"  loss by round: {string.Join(", ", History.Select(h => h.Loss.ToString("F4", CultureInfo.InvariantCulture)))}");
        }
    }

    // Preference rounds on the warmed tiny decoder (or a real model's folder), timed and with the memory after each round.
    internal static PreferenceRoundsResult PreferenceRounds(Device device, string folder, bool tiny, int rounds, int group, int episodes, int epochs, string loss, string reference, int seed,
        string prompt)
    {
        PretrainedModel model;
        LanguageModelPolicy policy;
        IReadOnlyList<ChatMessage> messages = [new ChatMessage("user", prompt)];
        if (tiny)
        {
            (policy, model, _) = WarmPolicy(folder, device, steps: rounds * epochs + 1);
            messages = [new ChatMessage("user", "Q")];
        }
        else
        {
            model = PretrainedModel.Load(folder, new PretrainedOptions { Device = device });
            policy = new LanguageModelPolicy(new FineTuningSession(model, SessionOptions(1e-4f), totalSteps: rounds * epochs + 1), new LanguageModelPolicyOptions { MaxNewTokens = 32 });
        }

        using var owned = model;
        using var _p = policy;
        var kind = new PromptsEnvironmentKind([new PromptTask(messages, tiny ? "yes" : null)], name: "preference-bench-prompts");
        var options = PreferenceRun(reference, rounds, epochs, seed) with
        {
            PreferenceLoss = loss, GroupSize = group, EpisodesPerIteration = episodes,
            EnvironmentOptions = tiny ? new Dictionary<string, string> { ["reward"] = "metric", ["metric.metric"] = "exact" } : new Dictionary<string, string> { ["reward"] = "length" },
        };
        var device1 = new List<long>();
        var managed = new List<long>();
        var trainer = new ReinforcementTrainer<ChatObservation, ChatMessage>(kind, policy, options)
        {
            OnIteration = _ =>
            {
                device1.Add(model.Device.Backend.GetMemoryUsage().InUse);
                managed.Add(GC.GetTotalMemory(forceFullCollection: true));
            },
        };
        var result = trainer.Train();
        return new PreferenceRoundsResult(result.History, [.. device1], [.. managed]);
    }

    private static void PreferenceBenchChecks()
    {
        foreach (string name in new[] { "group", "verdict" })
        {
            var measured = PreferenceBench(name, 256, 1000, seed: 3);
            Check(measured.PairsPerRound > 0 && measured.WarmBytesPerRound == 0, $"{name}: {measured.Report()}");
        }

        string folder = WriteChatModelWith(TinyChatSpec(), ChatMlTemplate);
        try
        {
            var rounds = PreferenceRounds(Device.Cpu, folder, tiny: true, rounds: 12, group: 8, episodes: 16, epochs: 2, loss: "dpo", reference: PreferenceReferences.Previous, seed: 7,
                prompt: "Q");
            Check(rounds.DeviceHeld[^1] <= rounds.DeviceHeld[1] + (1L << 20), $"device memory flat over the rounds:\n{rounds.Report()}");
            Check(rounds.ManagedHeld[^1] <= rounds.ManagedHeld[1] + (4L << 20), $"managed memory flat over the rounds:\n{rounds.Report()}");
        }
        finally
        {
            Directory.Delete(folder, true);
        }
    }

    // --bench-preference: pairs a second for each source (cold and warm, bytes a warm round) and preference rounds on a tiny
    // random decoder (or a real model: IDRAK_BENCH_PREFERENCE_MODEL, a model folder), each round's update and rollout cold
    // and warm and the memory after each. Settings: IDRAK_BENCH_PREFERENCE_SOURCES (group,verdict), _EPISODES (1024: the
    // episodes the sources choose from), _CHOICE_ROUNDS (1000), _ROUNDS (40: preference rounds), _GROUP (8), _ROLLOUT (16:
    // episodes a round), _EPOCHS (2), _LOSS (dpo), _REFERENCE (previous), _SEED (7), _PROMPT (a real model's prompt);
    // IDRAK_DEVICES (default the CPU).
    internal static int BenchPreference()
    {
        static int Setting(string name, int fallback) =>
            int.TryParse(Environment.GetEnvironmentVariable(name), NumberStyles.Integer, CultureInfo.InvariantCulture, out int value) && value > 0 ? value : fallback;
        static string Text(string name, string fallback) => Environment.GetEnvironmentVariable(name) is { Length: > 0 } value ? value.Trim() : fallback;

        int seed = Setting("IDRAK_BENCH_PREFERENCE_SEED", 7);
        Console.WriteLine("pair sources:");
        foreach (string name in Text("IDRAK_BENCH_PREFERENCE_SOURCES", "group,verdict").Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries))
        {
            Console.WriteLine(PreferenceBench(name, Setting("IDRAK_BENCH_PREFERENCE_EPISODES", 1024), Setting("IDRAK_BENCH_PREFERENCE_CHOICE_ROUNDS", 1000), seed).Report());
        }

        List<Device> devices = Environment.GetEnvironmentVariable("IDRAK_DEVICES") is { Length: > 0 } chosen
            ? [.. chosen.Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries).Select(Device.Parse)]
            : [Device.Cpu];
        string? real = Environment.GetEnvironmentVariable("IDRAK_BENCH_PREFERENCE_MODEL") is { Length: > 0 } given ? given : null;
        foreach (var device in devices)
        {
            string folder = real ?? WriteChatModelWith(TinyChatSpec(), ChatMlTemplate);
            try
            {
                Console.WriteLine($"{device}: preference rounds on {(real is null ? "a tiny random decoder (2 layers, width 32), warmed on yes and no" : real)}");
                var result = PreferenceRounds(device, folder, real is null, Setting("IDRAK_BENCH_PREFERENCE_ROUNDS", 40), Setting("IDRAK_BENCH_PREFERENCE_GROUP", 8),
                    Setting("IDRAK_BENCH_PREFERENCE_ROLLOUT", 16), Setting("IDRAK_BENCH_PREFERENCE_EPOCHS", 2), Text("IDRAK_BENCH_PREFERENCE_LOSS", "dpo"),
                    Text("IDRAK_BENCH_PREFERENCE_REFERENCE", PreferenceReferences.Previous), seed, Text("IDRAK_BENCH_PREFERENCE_PROMPT", "Answer yes or no: is the sky blue?"));
                Console.WriteLine(result.Report());
            }
            finally
            {
                if (real is null)
                {
                    Directory.Delete(folder, true);
                }
            }
        }

        return 0;
    }
}
