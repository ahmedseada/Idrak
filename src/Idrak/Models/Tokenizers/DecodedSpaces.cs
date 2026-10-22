// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

namespace Idrak.Models;

// The clean-up a model's tokenizer_config.json may ask for (clean_up_tokenization_spaces, TokenizerOptions.CleanUpSpaces):
// the space a decode leaves before punctuation and English contractions taken out, each replacement over the whole text
// before the next, in this order, as transformers does it ("Friday ." → "Friday.", "do n't" → "don't").
internal static class DecodedSpaces
{
    private static readonly (string From, string To)[] Replacements =
        [(" .", "."), (" ?", "?"), (" !", "!"), (" ,", ","), (" ' ", "'"), (" n't", "n't"), (" 'm", "'m"), (" 's", "'s"), (" 've", "'ve"), (" 're", "'re")];

    internal static string CleanUp(string text)
    {
        if (!text.Contains(' ', StringComparison.Ordinal))
        {
            return text;
        }

        foreach (var (from, to) in Replacements)
        {
            text = text.Replace(from, to, StringComparison.Ordinal);                // the same string back when none is there
        }

        return text;
    }
}
