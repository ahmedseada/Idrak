// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Text.Json.Nodes;

namespace Idrak.Abstraction.Testing;

/// <summary>
/// The checks of a pretrained tokenizer (<see cref="IPretrainedTokenizer"/>, the kind a tokenizer model builds): every check
/// of <see cref="TokenizerSuite"/>, and
/// <list type="bullet">
/// <item><see cref="IPretrainedTokenizer.EncodeRange"/> gives the ids <see cref="ITokenizer.Encode"/> gives for the whole text
/// and for a slice of it, appended after the ids already in the list;</item>
/// <item><see cref="IPretrainedTokenizer.IdOf"/> of an id's token names a token with the same entry, and
/// <see cref="IPretrainedTokenizer.IsSpecial"/> is false outside the vocabulary;</item>
/// <item><see cref="IPretrainedTokenizer.ToJson"/> names the tokenizer's <see cref="IPretrainedTokenizer.Model"/>.</item>
/// </list>
/// </summary>
/// <param name="reference">The tokenizer to agree with (the library's model an app's one replaces), or null.</param>
public sealed class PretrainedTokenizerSuite(IPretrainedTokenizer? reference = null) : ContractSuite<IPretrainedTokenizer>
{
    private TokenizerSuite? _tokenizer;

    /// <summary>The library tokenizer this one must agree with, or null.</summary>
    public IPretrainedTokenizer? Reference { get; } = reference;

    /// <summary>Whether decoding a text's ids must give the text exactly (byte-level tokenizers); else it must be stable.</summary>
    public bool ExactRoundTrip { get; init; }

    /// <summary>Texts of your own, checked after the built-in ones (the inputs your app sends).</summary>
    public IReadOnlyList<string> ExtraTexts { get; init; } = [];

    // The tokenizer checks this suite runs first, with its settings.
    private TokenizerSuite Tokenizer => _tokenizer ??= new TokenizerSuite(Reference) { ExactRoundTrip = ExactRoundTrip, ExtraTexts = ExtraTexts };

    /// <inheritdoc />
    public override string Name => "pretrained tokenizer";

    /// <inheritdoc />
    public override IEnumerable<ContractCase> FixedCases()
    {
        foreach (var @case in Tokenizer.FixedCases())
        {
            yield return @case;
        }

        yield return new ContractCase("the saved tokenizer.json", new JsonObject { ["json"] = true });
    }

    /// <inheritdoc />
    public override ContractCase RandomCase(Random random, bool large) => Tokenizer.RandomCase(random, large);

    /// <inheritdoc />
    public override void Run(IPretrainedTokenizer implementation, ContractCase @case, CaseChecks checks)
    {
        ArgumentNullException.ThrowIfNull(implementation);
        ArgumentNullException.ThrowIfNull(@case);
        ArgumentNullException.ThrowIfNull(checks);
        if (@case.Data["json"] is not null)
        {
            var model = implementation.ToJson()["model"]?["type"];
            checks.Check("tokenizer.json names the model", (string?)model == implementation.Model,
                $"model.type is '{model}', the tokenizer's Model is '{implementation.Model}'");
            return;
        }

        Tokenizer.Run(implementation, @case, checks);
        if (@case.Data["text"] is JsonValue value)
        {
            RunRanges(implementation, (string)value!, checks);
        }
        else
        {
            RunTokens(implementation, checks);
        }
    }

    private static void RunRanges(IPretrainedTokenizer implementation, string text, CaseChecks checks)
    {
        var whole = implementation.Encode(text);
        var ids = new List<int> { -7 };                                    // the range's ids go after what the list holds
        implementation.EncodeRange(text, 0, text.Length, ids);
        checks.Check("a range keeps the list's ids", ids.Count > 0 && ids[0] == -7, "the id already in the list was changed or removed");
        checks.Compare("the whole text as a range", Comparisons.Difference(whole, ids.Skip(1).ToList()));

        // The middle third, moved off the inside of a surrogate pair.
        int start = text.Length / 3, end = text.Length - text.Length / 3;
        if (start > 0 && start < text.Length && char.IsLowSurrogate(text[start]))
        {
            start++;
        }

        if (end > start && end < text.Length && char.IsLowSurrogate(text[end]))
        {
            end++;
        }

        if (end <= start)
        {
            return;
        }

        var slice = new List<int>();
        implementation.EncodeRange(text, start, end - start, slice);
        checks.Compare("a slice as a range", Comparisons.Difference(implementation.Encode(text[start..end]), slice));
    }

    private static void RunTokens(IPretrainedTokenizer implementation, CaseChecks checks)
    {
        int size = implementation.VocabularySize;
        checks.Check("no special id outside the vocabulary", !implementation.IsSpecial(-1) && !implementation.IsSpecial(size), "IsSpecial is true outside [0, VocabularySize)");
        int step = Math.Max(1, size / 997);
        for (int id = 0; id < size; id += step)
        {
            string token = implementation.TokenOf(id);
            if (token.Length == 0)
            {
                continue;                                                  // padding past the tokens, or an empty entry
            }

            // Two ids may share an entry (an added token written like a vocabulary one): IdOf names one of them.
            if (implementation.IdOf(token) is not { } found || implementation.TokenOf(found) != token)
            {
                checks.Check("IdOf finds each token", false, $"id {id} is '{token}', IdOf gives {implementation.IdOf(token)?.ToString(System.Globalization.CultureInfo.InvariantCulture) ?? "null"}");
                return;
            }
        }

        checks.Check("IdOf finds each token", true, "");
    }
}
