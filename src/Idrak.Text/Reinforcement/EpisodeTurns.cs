// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Idrak.Abstraction.Reinforcement;
using Idrak.Text.Abstractions;

namespace Idrak.Text.Reinforcement;

/// <summary>The training sequences of one episode (<see cref="EpisodeTurns.Lay"/>) and how they were laid out.</summary>
/// <param name="Mode"><see cref="EpisodeTurns.Episode"/> (one sequence holding every turn) or <see cref="EpisodeTurns.Turn"/> (a sequence per turn).</param>
/// <param name="Sequences">The sequences, in turn order; only the replies' ids are trained, and their trained tokens follow the turns' units in order.</param>
public sealed record EpisodeLayout(string Mode, IReadOnlyList<TrainingSequence> Sequences);

/// <summary>
/// How a multi-turn episode of a language model trains (Plan 27, R3). Each assistant turn records the prompt ids it was
/// generated from and the reply ids it sampled (<see cref="ActionRecord.Context"/>, <see cref="ActionRecord.Tokens"/>,
/// from <see cref="ChatTokens"/>). When the chat template renders turn n + 1's prompt as turn n's prompt followed by its
/// reply (checked at run time on the episode's own ids: <see cref="PrefixStable"/>), the episode trains as one sequence,
/// the last prompt and reply, with only the replies' ids trained (<see cref="Episode"/>); otherwise each turn trains on
/// its own prompt (<see cref="Turn"/>: a template that drops earlier reasoning, say). The choice is automatic
/// (<see cref="Auto"/>) and a setting (<see cref="EnvironmentVariable"/>). Prompts, tool results and the environment's
/// messages are never trained: a trajectory trains on exactly the ids sampled, never on a re-encoding of the text.
/// </summary>
public static class EpisodeTurns
{
    /// <summary>One sequence when the episode's ids are prefix-stable, a sequence per turn otherwise (the default).</summary>
    public const string Auto = "auto";

    /// <summary>One sequence for the whole episode (refused when its ids are not prefix-stable).</summary>
    public const string Episode = "episode";

    /// <summary>A sequence per turn, each on its own prompt.</summary>
    public const string Turn = "turn";

    /// <summary>The setting's variable: auto, episode or turn.</summary>
    public const string EnvironmentVariable = "IDRAK_RL_TURNS";

    /// <summary>The modes, as the setting takes them.</summary>
    public static IReadOnlyList<string> Modes { get; } = [Auto, Episode, Turn];

    /// <summary>The mode <paramref name="mode"/> names (ignoring case).</summary>
    /// <exception cref="ArgumentException">Not one of <see cref="Modes"/>.</exception>
    public static string Parse(string mode)
    {
        ArgumentNullException.ThrowIfNull(mode);
        foreach (string known in Modes)
        {
            if (known.Equals(mode.Trim(), StringComparison.OrdinalIgnoreCase))
            {
                return known;
            }
        }

        throw new ArgumentException($"The turn layout '{mode}' is not one of {string.Join(", ", Modes)} ({EnvironmentVariable}).", nameof(mode));
    }

    /// <summary>
    /// Whether every turn's prompt ids begin with the previous turn's prompt ids followed by its reply ids: the template
    /// renders the conversation as one growing text, so the last prompt holds every reply as it was sampled.
    /// </summary>
    /// <exception cref="ArgumentException">A record without its prompt ids (<see cref="ActionRecord.Context"/>).</exception>
    public static bool PrefixStable(IReadOnlyList<ActionRecord> records)
    {
        ArgumentNullException.ThrowIfNull(records);
        for (int s = 0; s + 1 < records.Count; s++)
        {
            int[] prompt = Context(records[s], s), reply = records[s].Tokens, next = Context(records[s + 1], s + 1);
            if (next.Length < prompt.Length + reply.Length || !next.AsSpan(0, prompt.Length).SequenceEqual(prompt)
                || !next.AsSpan(prompt.Length, reply.Length).SequenceEqual(reply))
            {
                return false;
            }
        }

        return true;
    }

    /// <summary>
    /// The training sequences of an episode whose turns are <paramref name="records"/> (one per assistant turn, in order):
    /// one sequence (<see cref="Episode"/>) or one per turn (<see cref="Turn"/>), as <paramref name="mode"/> says
    /// (<see cref="Auto"/>: one when <see cref="PrefixStable"/>). Only the replies' ids are trained, so the sequences'
    /// trained tokens are the turns' units, in order.
    /// </summary>
    /// <exception cref="ArgumentException">No turn, a turn without prompt ids or reply ids, or an unknown mode.</exception>
    /// <exception cref="InvalidOperationException"><see cref="Episode"/> asked for an episode whose ids are not prefix-stable.</exception>
    public static EpisodeLayout Lay(IReadOnlyList<ActionRecord> records, string mode = Auto)
    {
        ArgumentNullException.ThrowIfNull(records);
        mode = Parse(mode);
        if (records.Count == 0)
        {
            throw new ArgumentException("An episode has at least one turn.", nameof(records));
        }

        for (int s = 0; s < records.Count; s++)
        {
            if (Context(records[s], s).Length == 0 || records[s].Tokens.Length == 0)
            {
                throw new ArgumentException($"Turn {s} has {(records[s].Tokens.Length == 0 ? "no reply ids" : "no prompt ids")}: a turn trains on the ids it was "
                                            + "generated from and sampled (GenerationOptions.ReturnTokens).", nameof(records));
            }
        }

        bool stable = PrefixStable(records);
        if (mode == Episode && !stable)
        {
            throw new InvalidOperationException("The episode's turns are not prefix-stable (a turn's prompt ids do not begin with the previous prompt and reply as sampled: "
                                                + $"the template re-renders earlier turns), so it cannot train as one sequence; use {Turn} or {Auto} ({EnvironmentVariable}).");
        }

        if (mode == Turn || mode == Auto && !stable)
        {
            return new EpisodeLayout(Turn, [.. records.Select(r => Sequence(r.Context!, r.Tokens, [(r.Context!.Length, r.Tokens.Length)]))]);
        }

        var last = records[^1];
        return new EpisodeLayout(Episode, [Sequence(last.Context!, last.Tokens, [.. records.Select(r => (r.Context!.Length, r.Tokens.Length))])]);
    }

    // The prompt then the reply, the reply ranges (start, length) trained.
    private static TrainingSequence Sequence(int[] prompt, int[] reply, IReadOnlyList<(int Start, int Length)> trained)
    {
        int[] tokens = [.. prompt, .. reply];
        var flags = new bool[tokens.Length];
        foreach (var (start, length) in trained)
        {
            flags.AsSpan(start, length).Fill(true);
        }

        return new TrainingSequence(tokens, flags);
    }

    private static int[] Context(ActionRecord record, int turn) =>
        record.Context ?? throw new ArgumentException($"Turn {turn} has no prompt ids (ActionRecord.Context): a language model's turn records the ids it was generated from.");
}
