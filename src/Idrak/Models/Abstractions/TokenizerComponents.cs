// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Idrak.Models;

namespace Idrak.Models.Abstractions;

/// <summary>A tokenizer.json normalizer: rewrites the text between added tokens before it is split.</summary>
public interface ITokenizerNormalizer
{
    /// <summary>The normalized <paramref name="text"/>.</summary>
    string Normalize(string text);

    /// <summary>
    /// Normalizes <paramref name="text"/> in place, keeping where each character came from (the form the library's
    /// tokenizers run, with no string per call). By default the string form runs and the changed middle of the text
    /// stands for the old middle as a whole, so offsets through such a component are coarse; implement this to keep them
    /// exact.
    /// </summary>
    void Normalize(NormalizedText text)
    {
        ArgumentNullException.ThrowIfNull(text);
        string before = text.ToString();
        string after = Normalize(before);
        if (!string.Equals(before, after, StringComparison.Ordinal))
        {
            text.Set(after);
        }
    }
}

/// <summary>A tokenizer.json pre-tokenizer: splits (and may rewrite) text into the pieces BPE encodes one by one.</summary>
public interface IPreTokenizer
{
    /// <summary>
    /// The pieces made of <paramref name="pieces"/> (the previous stage's, or the normalized text as one piece);
    /// <paramref name="atStart"/> says whether the text begins the input (not after an added token). The final pieces are
    /// looked up in the vocabulary as they are.
    /// </summary>
    IReadOnlyList<string> PreTokenize(IReadOnlyList<string> pieces, bool atStart);

    /// <summary>
    /// Splits (and may rewrite) the pieces of <paramref name="text"/> in place, keeping where each character came from.
    /// By default the string form runs: a result piece found in order in the text keeps its characters' ranges, one that
    /// was rewritten stands for the piece it was found in as a whole.
    /// </summary>
    void PreTokenize(PreTokenizedText text, bool atStart)
    {
        ArgumentNullException.ThrowIfNull(text);
        TokenizerPipelineDefaults.PreTokenize(this, text, atStart);
    }
}

/// <summary>A tokenizer.json decoder stage: turns token texts (or the previous stage's pieces) into pieces of the decoded text.</summary>
public interface ITokenizerDecoder
{
    /// <summary>The pieces made of <paramref name="tokens"/>; the decoded text is the last stage's pieces joined.</summary>
    IReadOnlyList<string> Decode(IReadOnlyList<string> tokens);
}

/// <summary>
/// A tokenizer.json post-processor: adds a model's special tokens around an encoded text (or a pair of texts), with
/// their type ids, as <c>[CLS] … [SEP]</c> or <c>&lt;s&gt; … &lt;/s&gt;</c>.
/// </summary>
public interface ITokenizerPostProcessor
{
    /// <summary>How many tokens it adds to one text (<paramref name="pair"/> false) or to a pair (true), for truncation.</summary>
    int AddedTokens(bool pair);

    /// <summary>
    /// The encoding of <paramref name="encoding"/> (and <paramref name="pair"/>, when given) as the model takes it: with
    /// its special tokens when <paramref name="addSpecialTokens"/> is true, else the two joined as they are.
    /// </summary>
    TokenizerEncoding Process(TokenizerEncoding encoding, TokenizerEncoding? pair, bool addSpecialTokens);
}
