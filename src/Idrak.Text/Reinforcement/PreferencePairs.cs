// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Idrak.Abstraction.Reinforcement;
using Idrak.Text.Abstractions;

namespace Idrak.Text.Reinforcement;

/// <summary>
/// A language model's preferences as training data: each <see cref="PreferenceChoice"/> of an
/// <see cref="IPreferencePairSource"/> becomes a <see cref="PreferencePair"/> of token sequences after the same prompt, only
/// the answers trained. An answer the episode sampled trains on exactly the ids sampled (<see cref="ActionRecord.Context"/>
/// and <see cref="ActionRecord.Tokens"/>; never a re-encoding of its text): one step's prompt and reply, or the whole
/// episode as one sequence (<see cref="EpisodeTurns.Episode"/>, every reply trained; its ids must be prefix-stable). A
/// kept text (<see cref="PreferenceAnswer.Text"/>) follows the same step's prompt ids, tokenized with the template's end of
/// an assistant turn after it (<see cref="ChatTranscriptEncoder.AssistantEnd"/>), so both answers end as a reply does.
/// </summary>
public static class PreferencePairs
{
    /// <summary>
    /// Adds the pair of each of <paramref name="choices"/> over <paramref name="episodes"/> to <paramref name="into"/>, in
    /// order; returns how many it added. A choice whose two answers are the same ids gives no pair (it has nothing to teach).
    /// </summary>
    /// <param name="episodes">The episodes the choices name (by position).</param>
    /// <param name="choices">The preferences.</param>
    /// <param name="encoder">Tokenizes kept texts; null when no choice holds one.</param>
    /// <param name="into">Receives the pairs (reused from round to round).</param>
    /// <exception cref="ArgumentException">A choice names an episode or step there is not, or a turn without its prompt or reply ids.</exception>
    /// <exception cref="InvalidOperationException">A kept text without an encoder, or a whole episode of several turns whose ids are not prefix-stable.</exception>
    public static int Encode(IReadOnlyList<Trajectory<ChatObservation, ChatMessage>> episodes, IReadOnlyList<PreferenceChoice> choices, ChatTranscriptEncoder? encoder,
        List<PreferencePair> into)
    {
        ArgumentNullException.ThrowIfNull(episodes);
        ArgumentNullException.ThrowIfNull(choices);
        ArgumentNullException.ThrowIfNull(into);
        int added = 0;
        for (int c = 0; c < choices.Count; c++)
        {
            var choice = choices[c];
            var chosen = Answer(episodes, choice.Chosen, encoder, c);
            var rejected = Answer(episodes, choice.Rejected, encoder, c);
            if (chosen.Tokens.AsSpan().SequenceEqual(rejected.Tokens) && chosen.Trained.AsSpan().SequenceEqual(rejected.Trained))
            {
                continue;
            }

            into.Add(new PreferencePair(chosen, rejected));
            added++;
        }

        return added;
    }

    /// <summary>
    /// The preference pairs of recorded or played <paramref name="episodes"/> by the source <paramref name="source"/> of
    /// <see cref="PreferencePairSources"/> ("group", "verdict", or an app's), encoded as <see cref="Encode"/> does: what
    /// <see cref="FineTuner.Train(Models.PretrainedModel, IReadOnlyList{PreferencePair}, IReadOnlyList{PreferencePair}?, FineTuningOptions, string?, IProgress{FineTuningProgress}?, CancellationToken, Action{string}?)"/>
    /// trains on, for an app that keeps its episodes (an editor's accepted and rejected suggestions) and trains offline.
    /// </summary>
    /// <param name="episodes">The episodes.</param>
    /// <param name="source">The pair source's name.</param>
    /// <param name="encoder">Tokenizes kept texts; null when no choice holds one.</param>
    /// <param name="options">The source's options; null for its defaults.</param>
    /// <exception cref="NotRegisteredException">No source of that name is registered.</exception>
    public static List<PreferencePair> Build(IReadOnlyList<Trajectory<ChatObservation, ChatMessage>> episodes, string source, ChatTranscriptEncoder? encoder = null,
        IReadOnlyDictionary<string, string>? options = null)
    {
        ArgumentNullException.ThrowIfNull(episodes);
        ArgumentException.ThrowIfNullOrWhiteSpace(source);
        var choices = new List<PreferenceChoice>();
        PreferencePairSources.Get(source).Choose(episodes, choices, options);
        var pairs = new List<PreferencePair>(choices.Count);
        Encode(episodes, choices, encoder, pairs);
        return pairs;
    }

    /// <summary>The sequence of one answer (see the type).</summary>
    public static TrainingSequence Answer(IReadOnlyList<Trajectory<ChatObservation, ChatMessage>> episodes, PreferenceAnswer answer, ChatTranscriptEncoder? encoder,
        int choice = 0)
    {
        ArgumentNullException.ThrowIfNull(episodes);
        if ((uint)answer.Episode >= (uint)episodes.Count)
        {
            throw new ArgumentException($"Choice {choice} names episode {answer.Episode} of {episodes.Count}.", nameof(answer));
        }

        var records = episodes[answer.Episode].Records;
        if (answer.Step == PreferenceAnswer.WholeEpisode)
        {
            if (answer.Text is not null)
            {
                throw new ArgumentException($"Choice {choice}: a kept text replaces one step's answer, not the whole episode.", nameof(answer));
            }

            return records.Count == 1 ? Turn(records[0], records[0].Tokens, choice) : EpisodeTurns.Lay(records, EpisodeTurns.Episode).Sequences[0];
        }

        if ((uint)answer.Step >= (uint)records.Count)
        {
            throw new ArgumentException($"Choice {choice} names step {answer.Step} of episode {answer.Episode}'s {records.Count}.", nameof(answer));
        }

        var record = records[answer.Step];
        if (answer.Text is null)
        {
            return Turn(record, record.Tokens, choice);
        }

        var tokens = encoder ?? throw new InvalidOperationException($"Choice {choice} keeps a text: encoding it needs the model's chat template and tokenizer (a ChatTranscriptEncoder).");
        return Turn(record, [.. tokens.Tokenizer.Encode(answer.Text + tokens.AssistantEnd)], choice);
    }

    // The turn's prompt ids, then the reply's, the reply trained.
    private static TrainingSequence Turn(ActionRecord record, int[] reply, int choice)
    {
        int[] prompt = record.Context ?? throw new ArgumentException($"Choice {choice}: the turn has no prompt ids (ActionRecord.Context): a language model's turn records them.");
        if (prompt.Length == 0 || reply.Length == 0)
        {
            throw new ArgumentException($"Choice {choice}: a turn trains on its prompt ids and reply ids ({prompt.Length} and {reply.Length} given).");
        }

        int[] tokens = [.. prompt, .. reply];
        var trained = new bool[tokens.Length];
        trained.AsSpan(prompt.Length).Fill(true);
        return new TrainingSequence(tokens, trained);
    }
}
