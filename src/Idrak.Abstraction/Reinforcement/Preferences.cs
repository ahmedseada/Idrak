// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Buffers;
using System.Globalization;
using System.Runtime.CompilerServices;

namespace Idrak.Abstraction.Reinforcement;

/// <summary>
/// One side of a preference (<see cref="PreferenceChoice"/>): an answer an episode sampled, or a text the episode records
/// in its place (what a user kept instead of the model's answer). It names the episode by its position in the list the
/// pairs were chosen from (<see cref="RolloutBatch{TObservation, TAction}.Trajectories"/>), so a policy finds the ids it
/// sampled without a copy; a policy turns it into what it trains on (Idrak.Text: a <c>PreferencePair</c> of token
/// sequences after the same prompt).
/// </summary>
/// <param name="Episode">The episode: its position in the list the pairs were chosen from.</param>
/// <param name="Step">The step whose answer it is, or <see cref="WholeEpisode"/> for every step of the episode.</param>
/// <param name="Text">The text answered in the step's place (after the step's own prompt); null for the answer the episode sampled.</param>
public readonly record struct PreferenceAnswer(int Episode, int Step, string? Text)
{
    /// <summary>The <see cref="Step"/> of an answer that is the whole episode: every step's action, in order.</summary>
    public const int WholeEpisode = -1;

    /// <summary>The answer episode <paramref name="episode"/> sampled at <paramref name="step"/> (every step by default).</summary>
    public static PreferenceAnswer Sampled(int episode, int step = WholeEpisode) => new(episode, step, null);

    /// <summary>The text <paramref name="text"/> answered in the place of episode <paramref name="episode"/>'s answer at <paramref name="step"/>.</summary>
    /// <exception cref="ArgumentNullException">No text.</exception>
    /// <exception cref="ArgumentOutOfRangeException">A step below 0: a kept text replaces one step's answer.</exception>
    public static PreferenceAnswer Kept(int episode, int step, string text)
    {
        ArgumentNullException.ThrowIfNull(text);
        ArgumentOutOfRangeException.ThrowIfNegative(step);
        return new PreferenceAnswer(episode, step, text);
    }

    /// <summary>Whether it is an answer the episode sampled (not a text in its place).</summary>
    public bool IsSampled => Text is null;
}

/// <summary>
/// A preference between two answers to the same situation (<see cref="IPreferencePairSource"/>): the chosen one, the
/// rejected one, by how much (a reward difference, NaN when the source has no number for it) and the group they came from.
/// </summary>
/// <param name="Chosen">The preferred answer.</param>
/// <param name="Rejected">The answer to move away from.</param>
/// <param name="Margin">How much better the chosen one is (the best return less the worst, for "group"); NaN when unknown.</param>
/// <param name="Group">The group of the episodes (<see cref="Trajectory{TObservation, TAction}.Group"/>).</param>
public readonly record struct PreferenceChoice(PreferenceAnswer Chosen, PreferenceAnswer Rejected, float Margin, int Group);

/// <summary>
/// The episode info keys a host writes for the "verdict" pair source (<see cref="PreferencePairSources.Verdict"/>): what a
/// user did with the model's answer and the text kept instead. An editor's episode of an inline completion, say: the
/// suggestion rejected (<see cref="Verdict"/> "rejected"), then the text the user wrote in its place (<see cref="Kept"/>),
/// at the step that answered (<see cref="KeptStep"/>, the last by default).
/// </summary>
public static class PreferenceInfoKeys
{
    /// <summary>The answer was accepted and kept as it was: no pair.</summary>
    public const string Accepted = "accepted";

    /// <summary>The answer was accepted, then edited: <see cref="Kept"/> holds the text after the edit.</summary>
    public const string Edited = "edited";

    /// <summary>The answer was rejected: <see cref="Kept"/> holds the text written or kept instead.</summary>
    public const string Rejected = "rejected";

    /// <summary>The answer was accepted, then undone: <see cref="Kept"/> holds the text as it was before.</summary>
    public const string Undone = "undone";

    /// <summary>What the user did with the answer: <see cref="Accepted"/>, <see cref="Edited"/>, <see cref="Rejected"/>, <see cref="Undone"/> or a host's own word: "verdict".</summary>
    public static EpisodeInfoKey<string> Verdict { get; } = new("verdict");

    /// <summary>The text kept in the answer's place (what the user wrote, or left, after the same prompt): "kept".</summary>
    public static EpisodeInfoKey<string> Kept { get; } = new("kept");

    /// <summary>The step whose answer the kept text replaces (0 for the first; the episode's last when not set): "kept_step".</summary>
    public static EpisodeInfoKey<int> KeptStep { get; } = new("kept_step");
}

/// <summary>
/// What turns played episodes into preferences (<see cref="PreferenceChoice"/>) for a preference loss (DPO, ORPO, SimPO):
/// the best and the worst answer of a group to one prompt ("group"), or an answer a user rejected against the text the
/// episode records as kept ("verdict"); registered in <see cref="PreferencePairSources"/> under its <see cref="IKind.Name"/>.
/// Its options are declared once (<see cref="IOptionSchema.Options"/>; <see cref="IConfigurableKind.Keys"/> are theirs).
/// <see cref="Choose{TObservation, TAction}"/> is deterministic (the same episodes give the same choices, in the same
/// order) and thread-safe: it keeps no state between calls.
/// </summary>
public interface IPreferencePairSource : IConfigurableKind, IOptionSchema
{
    /// <summary>Whether its pairs come from groups of episodes that played the same situation (a trainer then plays groups of more than one).</summary>
    bool NeedsGroups { get; }

    /// <summary>Adds the preferences of <paramref name="episodes"/> to <paramref name="into"/> (what it held stays); returns how many it added.</summary>
    /// <param name="episodes">The episodes, in the order a <see cref="PreferenceAnswer.Episode"/> counts them (a batch's <see cref="RolloutBatch{TObservation, TAction}.Trajectories"/>).</param>
    /// <param name="into">Receives the choices; a caller reuses it from round to round (cleared), so the memory stays flat.</param>
    /// <param name="options">The source's options by key, values as text; null for its defaults.</param>
    /// <exception cref="ArgumentException">An option the source does not take (the message names it and the keys) or a bad value.</exception>
    /// <exception cref="InvalidOperationException">An episode's info the source reads holds what it cannot use (a kept step past the episode's steps).</exception>
    int Choose<TObservation, TAction>(IReadOnlyList<Trajectory<TObservation, TAction>> episodes, List<PreferenceChoice> into, IReadOnlyDictionary<string, string>? options = null);
}

/// <summary>
/// The preference-pair sources by name (ignoring case): "group" and "verdict" are the library's (<see cref="Group"/>,
/// <see cref="Verdict"/>); an app or a plug-in registers its own with <see cref="Register"/>, and one under a library name
/// shadows the library's, which <see cref="Unregister"/> brings back, guarded by its <see cref="SetPolicy">policy</see>.
/// A name nobody registered fails "not registered". Idrak.Abstraction holds the contract and its built-ins (they read
/// only <see cref="Trajectory{TObservation, TAction}"/>), so any package's policy uses them; the pairs' token sequences
/// are the policy's (Idrak.Text's language-model policy encodes them).
/// </summary>
public static class PreferencePairSources
{
    /// <summary>The best and the worst episode of each group by return: "group".</summary>
    public const string Group = "group";

    /// <summary>An episode's answer the user rejected against the text it records as kept: "verdict".</summary>
    public const string Verdict = "verdict";

    private static readonly KindTable<IPreferencePairSource> Table = BuiltIn();

    private static KindTable<IPreferencePairSource> BuiltIn()
    {
        var table = new KindTable<IPreferencePairSource>(nameof(PreferencePairSources), "Preference-pair source", "add it with PreferencePairSources.Register",
            (slot, app, library) => new GuardedSource(slot, app, library), StringComparer.OrdinalIgnoreCase);
        table.RegisterDefault(new GroupSource());
        table.RegisterDefault(new VerdictSource());
        return table;
    }

    /// <summary>Registers <paramref name="source"/> under its <see cref="IKind.Name"/> (over the library's of that name, which stays behind it).</summary>
    [MethodImpl(MethodImplOptions.NoInlining)]   // the caller is the registering assembly (its Origin)
    public static void Register(IPreferencePairSource source) => Table.Register(source, System.Reflection.Assembly.GetCallingAssembly());

    /// <summary>Removes the app's source <paramref name="name"/> (a library name gets the library's back); false when the app registered none.</summary>
    public static bool Unregister(string name) => Table.Unregister(name);

    /// <summary>The registered names, in the order they were registered.</summary>
    public static IReadOnlyCollection<string> Names => Table.Names;

    /// <summary>The source registered as <paramref name="name"/>, or null.</summary>
    public static IPreferencePairSource? Find(ReadOnlySpan<char> name) => Table.Find(name);

    /// <summary>The source registered as <paramref name="name"/>; an app's, guarded by its <see cref="SetPolicy">policy</see>.</summary>
    /// <exception cref="NotRegisteredException">None is: the message names the registered ones.</exception>
    public static IPreferencePairSource Get(ReadOnlySpan<char> name) => Table.Get(name);

    /// <summary>The library's source <paramref name="name"/>, whatever an app registered over it; null when the library has none.</summary>
    public static IPreferencePairSource? Default(string name) => Table.Default(name);

    /// <summary>Who registered the source <paramref name="name"/>: <see cref="Overrides.Library"/> or the app's assembly; null when none is.</summary>
    public static string? Origin(string name) => Table.Origin(name);

    /// <summary>
    /// What happens when the app's source <paramref name="name"/> fails (<see cref="SlotPolicy.Throw"/> unless set;
    /// <see cref="SlotPolicy.FallBack"/> chooses with the library's; <see cref="SlotPolicy.Shadow"/> lets the library's
    /// choose and compares the app's choices on a sample of calls).
    /// </summary>
    public static void SetPolicy(string name, SlotPolicy policy, double shadowRate = Slot.DefaultShadowRate) => Table.SetPolicy(name, policy, shadowRate);

    /// <summary>Each registered source as "name: summary", one per line.</summary>
    public static string Describe() => Table.Describe();

    // "group": in each group of two or more episodes, the one of the highest return chosen over the one of the lowest (the
    // first in list order on a tie), when the best beats the worst by more than the margin. One pass over the episodes'
    // groups, sorted once in a rented buffer (group, then list position): groups in number order, nothing else allocated.
    private sealed class GroupSource : IPreferencePairSource
    {
        private static readonly OptionSpec<float> MarginOption = OptionSpec.Number("margin", 0f, 0f, takes: "a number of at least 0",
            summary: "the least return the best of a group must beat its worst by (more than it: 0 pairs any difference, never a tie)");

        public string Name => Group;

        public string Summary => "the best and the worst answer of each group to one prompt, by return (a margin to beat)";

        public OptionSchema Options { get; } = new("pair source group", MarginOption);

        public IReadOnlySet<string> Keys => Options.Keys;

        public bool NeedsGroups => true;

        public int Choose<TObservation, TAction>(IReadOnlyList<Trajectory<TObservation, TAction>> episodes, List<PreferenceChoice> into,
            IReadOnlyDictionary<string, string>? options = null)
        {
            ArgumentNullException.ThrowIfNull(episodes);
            ArgumentNullException.ThrowIfNull(into);
            float margin = Options.Parse(options).Get(MarginOption);
            int count = episodes.Count, added = 0;
            if (count < 2)
            {
                return 0;
            }

            long[] keys = ArrayPool<long>.Shared.Rent(count);
            float[] returns = ArrayPool<float>.Shared.Rent(count);
            try
            {
                for (int i = 0; i < count; i++)
                {
                    var episode = episodes[i] ?? throw new ArgumentException($"Episode {i} is null.", nameof(episodes));
                    keys[i] = ((long)episode.Group << 32) | (uint)i;          // the group, then the list position
                    returns[i] = episode.Return;
                }

                Array.Sort(keys, 0, count);
                for (int start = 0; start < count;)
                {
                    int group = (int)(keys[start] >> 32), end = start + 1;
                    while (end < count && (int)(keys[end] >> 32) == group)
                    {
                        end++;
                    }

                    if (end - start >= 2)
                    {
                        int best = (int)(uint)keys[start], worst = best;
                        for (int k = start + 1; k < end; k++)
                        {
                            int at = (int)(uint)keys[k];
                            if (returns[at] > returns[best])
                            {
                                best = at;
                            }

                            if (returns[at] < returns[worst])
                            {
                                worst = at;
                            }
                        }

                        float gap = returns[best] - returns[worst];
                        if (gap > margin)
                        {
                            into.Add(new PreferenceChoice(PreferenceAnswer.Sampled(best), PreferenceAnswer.Sampled(worst), gap, group));
                            added++;
                        }
                    }

                    start = end;
                }
            }
            finally
            {
                ArrayPool<long>.Shared.Return(keys);
                ArrayPool<float>.Shared.Return(returns);
            }

            return added;
        }
    }

    // "verdict": each episode whose info holds a kept text (PreferenceInfoKeys.Kept) and a verdict among the option's (none
    // set counts as rejected) gives the kept text chosen over the answer it sampled at the kept step, in list order.
    private sealed class VerdictSource : IPreferencePairSource
    {
        private const string DefaultVerdicts = PreferenceInfoKeys.Rejected + "," + PreferenceInfoKeys.Edited + "," + PreferenceInfoKeys.Undone;

        private static readonly OptionSpec<string?> VerdictsOption = OptionSpec.Text("verdicts", DefaultVerdicts, "verdicts separated by commas",
            "the verdicts (PreferenceInfoKeys.Verdict) whose kept text is preferred to the answer; an episode without a verdict counts as rejected");

        private Parsed _parsed = new(DefaultVerdicts, Split(DefaultVerdicts));   // swapped whole: a reader on another thread sees one or the other

        public string Name => Verdict;

        public string Summary => "an episode's answer the user rejected, edited or undid, against the text it records as kept (PreferenceInfoKeys)";

        public OptionSchema Options { get; } = new("pair source verdict", VerdictsOption);

        public IReadOnlySet<string> Keys => Options.Keys;

        public bool NeedsGroups => false;

        public int Choose<TObservation, TAction>(IReadOnlyList<Trajectory<TObservation, TAction>> episodes, List<PreferenceChoice> into,
            IReadOnlyDictionary<string, string>? options = null)
        {
            ArgumentNullException.ThrowIfNull(episodes);
            ArgumentNullException.ThrowIfNull(into);
            string[] verdicts = Verdicts(Options.Parse(options).Get(VerdictsOption) ?? DefaultVerdicts);
            int added = 0;
            for (int i = 0; i < episodes.Count; i++)
            {
                var episode = episodes[i] ?? throw new ArgumentException($"Episode {i} is null.", nameof(episodes));
                if (episode.Info is not { } info || !info.TryGet(PreferenceInfoKeys.Kept, out string kept))
                {
                    continue;
                }

                string verdict = info.GetOrDefault(PreferenceInfoKeys.Verdict, PreferenceInfoKeys.Rejected);
                if (!Contains(verdicts, verdict))
                {
                    continue;
                }

                int step = info.GetOrDefault(PreferenceInfoKeys.KeptStep, episode.Steps - 1);
                if ((uint)step >= (uint)episode.Steps)
                {
                    throw new InvalidOperationException(string.Create(CultureInfo.InvariantCulture,
                        $"Episode {i} keeps a text at step {step} ({PreferenceInfoKeys.KeptStep.Name}) of its {episode.Steps}: a kept step is one of the episode's."));
                }

                into.Add(new PreferenceChoice(PreferenceAnswer.Kept(i, step, kept), PreferenceAnswer.Sampled(i, step), float.NaN, episode.Group));
                added++;
            }

            return added;
        }

        // The verdicts of the option's text, split once per text (the last kept: a run asks with the same options each round).
        private string[] Verdicts(string text)
        {
            var parsed = _parsed;
            if (string.Equals(parsed.Text, text, StringComparison.Ordinal))
            {
                return parsed.Verdicts;
            }

            var verdicts = Split(text);
            _parsed = new Parsed(text, verdicts);
            return verdicts;
        }

        private sealed record Parsed(string Text, string[] Verdicts);

        private static string[] Split(string text) => text.Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries);

        private static bool Contains(string[] verdicts, string verdict)
        {
            foreach (string known in verdicts)
            {
                if (string.Equals(known, verdict, StringComparison.OrdinalIgnoreCase))
                {
                    return true;
                }
            }

            return false;
        }
    }

    // An app's source over the library's: a call that fails falls back to the library's (its own additions taken back
    // first), or the library's chooses and the app's choices are compared on a sample of calls.
    private sealed class GuardedSource(Slot slot, IPreferencePairSource app, IPreferencePairSource library) : IPreferencePairSource
    {
        public string Name => app.Name;

        public string Summary => app.Summary;

        public OptionSchema Options => app.Options;

        public IReadOnlySet<string> Keys => app.Keys;

        public bool NeedsGroups => app.NeedsGroups || library.NeedsGroups;

        public int Choose<TObservation, TAction>(IReadOnlyList<Trajectory<TObservation, TAction>> episodes, List<PreferenceChoice> into,
            IReadOnlyDictionary<string, string>? options = null)
        {
            ArgumentNullException.ThrowIfNull(into);
            int before = into.Count;
            if (slot.Policy == SlotPolicy.Shadow)
            {
                var run = slot.Shadow();
                int added = library.Choose(episodes, into, options);
                if (run is not null)
                {
                    run.Answered();
                    try
                    {
                        var other = new List<PreferenceChoice>(added);
                        app.Choose(episodes, other, options);
                        run.Done(Comparisons.Sequences(into.GetRange(before, added), other));
                    }
                    catch (Exception e) when (e is not OperationCanceledException)
                    {
                        run.Failed(e);
                    }
                }

                return added;
            }

            try
            {
                return app.Choose(episodes, into, options);
            }
            catch (Exception e) when (slot.Failed(e))
            {
                into.RemoveRange(before, into.Count - before);
                return library.Choose(episodes, into, options);
            }
        }
    }
}

/// <summary>
/// What a preference update trains against (<see cref="IPreferencePolicy{TObservation, TAction}"/>): the policy as it was
/// when the round began (<see cref="Previous"/>, iterative preference training: each round's reference is the last
/// round's result), or the base model (<see cref="Base"/>, adapters off).
/// </summary>
public static class PreferenceReferences
{
    /// <summary>The policy as it was before the round's first update on its pairs: "previous" (the default).</summary>
    public const string Previous = "previous";

    /// <summary>The base model (a language model with its adapters off): "base".</summary>
    public const string Base = "base";

    /// <summary>The references, as a setting takes them.</summary>
    public static IReadOnlyList<string> Names { get; } = [Previous, Base];

    /// <summary>The reference <paramref name="name"/> names (ignoring case).</summary>
    /// <exception cref="ArgumentException">Not one of <see cref="Names"/>.</exception>
    public static string Parse(string name)
    {
        ArgumentNullException.ThrowIfNull(name);
        foreach (string known in Names)
        {
            if (known.Equals(name.Trim(), StringComparison.OrdinalIgnoreCase))
            {
                return known;
            }
        }

        throw new ArgumentException($"The preference reference '{name}' is not one of {string.Join(", ", Names)}.", nameof(name));
    }
}

/// <summary>
/// A policy that also learns from preferences between its answers (<see cref="PreferenceChoice"/>, made by an
/// <see cref="IPreferencePairSource"/>): one optimizer step on the pairs with a preference loss chosen by name (Idrak.Text's
/// language-model policy: DPO, ORPO, SimPO from its <c>PreferenceLosses</c>), against a reference
/// (<see cref="PreferenceReferences"/>). The trainer of Idrak.Reinforcement drives it round after round: collect a rollout,
/// choose its pairs, update.
/// </summary>
/// <typeparam name="TObservation">The environment's observations.</typeparam>
/// <typeparam name="TAction">The environment's actions.</typeparam>
public interface IPreferencePolicy<TObservation, TAction> : IPolicy<TObservation, TAction>
{
    /// <summary>Checks, before any rollout, that <paramref name="loss"/> is a preference loss the policy has, that it takes <paramref name="lossOptions"/>, and the reference's name.</summary>
    /// <exception cref="NotSupportedException">No preference loss of that name is registered ("not registered").</exception>
    /// <exception cref="ArgumentException">An option the loss does not take, a bad value, or an unknown reference.</exception>
    void CheckPreferences(string loss, IReadOnlyDictionary<string, string>? lossOptions = null, string reference = PreferenceReferences.Previous);

    /// <summary>
    /// One optimizer step on <paramref name="pairs"/> of <paramref name="batch"/> with the preference loss <paramref name="loss"/>.
    /// The first update on a batch's pairs computes the reference's log-probabilities once (<paramref name="reference"/>:
    /// the policy as it is then, or the base model), before any weight moves; later epochs on the same pairs reuse them.
    /// No pair: no step (the stats say 0 pairs).
    /// </summary>
    /// <param name="batch">The episodes the pairs name.</param>
    /// <param name="pairs">The preferences (<see cref="IPreferencePairSource.Choose{TObservation, TAction}"/>).</param>
    /// <param name="loss">The preference loss's name.</param>
    /// <param name="lossOptions">Its options by key; null for its defaults.</param>
    /// <param name="reference">What the loss trains against (<see cref="PreferenceReferences"/>).</param>
    UpdateStats UpdatePreferences(RolloutBatch<TObservation, TAction> batch, IReadOnlyList<PreferenceChoice> pairs, string loss,
        IReadOnlyDictionary<string, string>? lossOptions = null, string reference = PreferenceReferences.Previous);
}
