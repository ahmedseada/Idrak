// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Runtime.CompilerServices;

namespace Idrak.Abstraction.Generation;

/// <summary>
/// A model that proposes tokens for another to check (speculative decoding, <see cref="GenerationOptions.Draft"/>): usually
/// a smaller model of the target's family, sharing its vocabulary. Idrak's <c>TextGenerator</c> is one; implement this for
/// another way of drafting (an n-gram lookup, a model behind a gate of one's own).
/// </summary>
public interface IDraftModel
{
    /// <summary>The draft's tokenizer: its vocabulary must be the target's (<see cref="SpeculativeDecoding.Refusal"/>).</summary>
    ITokenizer Tokenizer { get; }

    /// <summary>The width of the draft's logits (its output head's features): the ids it can propose are below it.</summary>
    int Vocabulary { get; }

    /// <summary>
    /// Starts drafting after <paramref name="prompt"/> (the ids the target was fed, its first reply token not among them),
    /// sampling as <paramref name="options"/> say (greedy when <see cref="SpeculativeDecoding.IsGreedy"/>). The session is
    /// the caller's to dispose; one session serves one generation, on the generation's thread.
    /// </summary>
    IDraftSession StartDraft(IReadOnlyList<int> prompt, GenerationOptions options);
}

/// <summary>One generation's drafting (<see cref="IDraftModel.StartDraft"/>): proposals, then what the target kept of them.</summary>
public interface IDraftSession : IDisposable
{
    /// <summary>
    /// Proposes up to <c>ids.Length</c> tokens following every token accepted so far, one after another, into
    /// <paramref name="ids"/>. When <paramref name="probabilities"/> is not empty (sampling: <c>ids.Length</c> rows of
    /// <paramref name="width"/>), row i receives the distribution token i was drawn from, which the target's check needs
    /// (ids at or past the draft's <see cref="IDraftModel.Vocabulary"/> 0); greedy drafting passes none. Returns how many
    /// it proposed: fewer, or 0, when it can propose no more (its context is full, or <paramref name="cancellationToken"/>
    /// was cancelled between two of its steps).
    /// </summary>
    int Propose(Span<int> ids, Span<float> probabilities, int width, CancellationToken cancellationToken);

    /// <summary>
    /// The target kept the first <paramref name="accepted"/> tokens of the last proposal (0 when none was made) and then
    /// chose <paramref name="next"/>: the session goes back to that point (its cache cut after the kept tokens) and
    /// continues after <paramref name="next"/>.
    /// </summary>
    void Accept(int accepted, int next);
}

/// <summary>Speculative decoding's shared rules: whether a draft fits a target, greedy or sampled, and the adapted round size.</summary>
public static class SpeculativeDecoding
{
    // A verdict per pair of tokenizers, computed once (a vocabulary of 150,000 ids is compared entry by entry).
    private static readonly ConditionalWeakTable<ITokenizer, ConditionalWeakTable<ITokenizer, Verdict>> Verdicts = new();

    private sealed record Verdict(string? Reason);

    /// <summary>
    /// Why <paramref name="draft"/> cannot draft for a target with <paramref name="target"/>'s vocabulary, or null when it
    /// can: every id both tokenizers have must be the same vocabulary entry (<see cref="ITokenizer.TokenOf"/>), and the ids
    /// only one has must be padding (an empty entry; a family's models often pad their vocabulary to different widths).
    /// Compared once per pair of tokenizers.
    /// </summary>
    public static string? Refusal(ITokenizer target, IDraftModel draft)
    {
        ArgumentNullException.ThrowIfNull(target);
        ArgumentNullException.ThrowIfNull(draft);
        var other = draft.Tokenizer;
        if (ReferenceEquals(target, other))
        {
            return null;
        }

        var verdicts = Verdicts.GetValue(target, _ => new());
        return verdicts.GetValue(other, d => new Verdict(Compare(target, d))).Reason;
    }

    private static string? Compare(ITokenizer target, ITokenizer draft)
    {
        int shared = Math.Min(target.VocabularySize, draft.VocabularySize);
        for (int id = 0; id < shared; id++)
        {
            string mine = target.TokenOf(id), theirs = draft.TokenOf(id);
            if (!string.Equals(mine, theirs, StringComparison.Ordinal))
            {
                return $"The draft's vocabulary is not the target's: id {id} is '{Shown(theirs)}' in the draft and '{Shown(mine)}' in the target "
                       + $"({draft.VocabularySize} and {target.VocabularySize} ids). Speculative decoding needs one vocabulary: draft with a model of the "
                       + "target's family (the same tokenizer), or generate without a draft.";
            }
        }

        var (longer, which) = target.VocabularySize > draft.VocabularySize ? (target, "target") : (draft, "draft");
        for (int id = shared; id < longer.VocabularySize; id++)
        {
            if (longer.TokenOf(id) is { Length: > 0 } extra)
            {
                return $"The draft's vocabulary is not the target's: the {which} has {longer.VocabularySize} ids and the other {shared}, and its id {id} is "
                       + $"'{Shown(extra)}', not padding. Speculative decoding needs one vocabulary: draft with a model of the target's family (the same "
                       + "tokenizer), or generate without a draft.";
            }
        }

        return null;
    }

    private static string Shown(string token) => token.Length > 24 ? token[..24] + "…" : token;

    /// <summary>Whether <paramref name="options"/> decode greedily (<see cref="GenerationOptions.TopK"/> 1): a proposal is kept while it is the target's own choice.</summary>
    public static bool IsGreedy(GenerationOptions options)
    {
        ArgumentNullException.ThrowIfNull(options);
        return options.TopK == 1;
    }

    /// <summary>
    /// The tokens the next round proposes after one that proposed <paramref name="proposed"/> (<paramref name="current"/>
    /// asked for) and kept <paramref name="accepted"/>: <see cref="GenerationOptions.DraftTokens"/> when it is set, else
    /// two more when every proposal was kept, one fewer when one was refused, within <see cref="GenerationOptions.DraftMinTokens"/>
    /// and <see cref="GenerationOptions.DraftMaxTokens"/>.
    /// </summary>
    public static int NextRound(int current, int proposed, int accepted, GenerationOptions options)
    {
        ArgumentNullException.ThrowIfNull(options);
        if (options.DraftTokens > 0)
        {
            return options.DraftTokens;
        }

        var (least, most) = Bounds(options);
        if (proposed <= 0)
        {
            return Math.Clamp(current, least, most);
        }

        return Math.Clamp(accepted >= proposed ? current + 2 : current - 1, least, most);
    }

    /// <summary>The tokens the first round proposes: <see cref="GenerationOptions.DraftTokens"/> when set, else <see cref="GenerationOptions.DraftMinTokens"/> + 3 within the bounds.</summary>
    public static int FirstRound(GenerationOptions options)
    {
        ArgumentNullException.ThrowIfNull(options);
        var (least, most) = Bounds(options);
        return options.DraftTokens > 0 ? options.DraftTokens : Math.Clamp(least + 3, least, most);
    }

    private static (int Least, int Most) Bounds(GenerationOptions options) =>
        (options.DraftMinTokens, Math.Max(options.DraftMinTokens, options.DraftMaxTokens));
}
