// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Globalization;

namespace Idrak.Models.Abstractions;

/// <summary>
/// Settings for reading a tokenizer (<see cref="PretrainedTokenizers.Load"/>, <see cref="PretrainedOptions.Tokenizer"/>).
/// A setting left null takes its environment variable, then the default.
/// </summary>
public sealed record TokenizerOptions
{
    /// <summary>The variable <see cref="Format"/> falls back to.</summary>
    public const string FormatVariable = "IDRAK_TOKENIZER_FORMAT";

    /// <summary>The variable <see cref="PieceCacheLimit"/> falls back to.</summary>
    public const string PieceCacheVariable = "IDRAK_TOKENIZER_CACHE";

    /// <summary>The variable <see cref="MaxInputCharsPerWord"/> falls back to.</summary>
    public const string MaxWordCharsVariable = "IDRAK_TOKENIZER_MAX_WORD_CHARS";

    /// <summary>The number of encoded pieces a tokenizer keeps when nothing sets <see cref="PieceCacheLimit"/>.</summary>
    public const int DefaultPieceCacheLimit = 100_000;

    /// <summary>The variable <see cref="SamplingAlpha"/> falls back to.</summary>
    public const string SamplingAlphaVariable = "IDRAK_TOKENIZER_ALPHA";

    /// <summary>The variable <see cref="SamplingBest"/> falls back to.</summary>
    public const string SamplingBestVariable = "IDRAK_TOKENIZER_NBEST";

    /// <summary>The variable <see cref="SamplingSeed"/> falls back to.</summary>
    public const string SamplingSeedVariable = "IDRAK_TOKENIZER_SEED";

    /// <summary>The n-best size sampling uses when nothing sets <see cref="SamplingBest"/>: the whole lattice.</summary>
    public const int DefaultSamplingBest = -1;

    /// <summary>
    /// The reader (<see cref="TokenizerFiles"/>: "tokenizer.json", or a registered one) to use when a folder holds the files
    /// of several; null takes IDRAK_TOKENIZER_FORMAT, then the first reader that finds its file, in registration order.
    /// </summary>
    public string? Format { get; init; }

    /// <summary>
    /// How many encoded pieces (words) a tokenizer caches, so a repeated piece is looked up instead of encoded again (0:
    /// none); null takes IDRAK_TOKENIZER_CACHE, then 100,000.
    /// </summary>
    public int? PieceCacheLimit { get; init; }

    /// <summary>
    /// Subword regularization (Unigram): encoding samples a segmentation instead of taking the best one, each with
    /// probability proportional to exp(alpha × its score); a larger alpha keeps closer to the best. Null takes
    /// IDRAK_TOKENIZER_ALPHA; when neither is set, sampling is off (every encode gives the best segmentation). A sampled
    /// encode bypasses the piece cache.
    /// </summary>
    public double? SamplingAlpha { get; init; }

    /// <summary>
    /// With <see cref="SamplingAlpha"/>: how many of the best segmentations a sample is drawn from (as SentencePiece's
    /// nbest_size): more than 1 draws from that many; 1 takes the best (no sampling); 0 or less draws from every
    /// segmentation of the lattice (forward filtering, backward sampling). Null takes IDRAK_TOKENIZER_NBEST, then -1.
    /// </summary>
    public int? SamplingBest { get; init; }

    /// <summary>
    /// With <see cref="SamplingAlpha"/>: the seed of the tokenizer's random draws, so the same encodes in the same order
    /// sample the same segmentations; null takes IDRAK_TOKENIZER_SEED, then an unseeded generator.
    /// </summary>
    public int? SamplingSeed { get; init; }

    /// <summary>The sampling alpha, from the setting or its variable; null when sampling is off.</summary>
    /// <exception cref="ArgumentOutOfRangeException">The setting is negative or not a finite number.</exception>
    /// <exception cref="InvalidOperationException">The variable is not a finite number of 0 or more.</exception>
    public double? ResolvedSamplingAlpha()
    {
        if (SamplingAlpha is { } alpha)
        {
            if (!double.IsFinite(alpha) || alpha < 0)
            {
                throw new ArgumentOutOfRangeException(nameof(SamplingAlpha), alpha, "Give a finite number, 0 or more.");
            }

            return alpha;
        }

        if (Environment.GetEnvironmentVariable(SamplingAlphaVariable) is not { Length: > 0 } text)
        {
            return null;
        }

        return double.TryParse(text, NumberStyles.Float, CultureInfo.InvariantCulture, out double value) && double.IsFinite(value) && value >= 0
            ? value
            : throw new InvalidOperationException($"{SamplingAlphaVariable} is '{text}': give a finite number, 0 or more (0.1 is common).");
    }

    /// <summary>The sampling n-best size, from the setting, its variable or the default (-1).</summary>
    /// <exception cref="InvalidOperationException">The variable is not a whole number.</exception>
    public int ResolvedSamplingBest()
    {
        if (SamplingBest is { } best)
        {
            return best;
        }

        if (Environment.GetEnvironmentVariable(SamplingBestVariable) is not { Length: > 0 } text)
        {
            return DefaultSamplingBest;
        }

        return int.TryParse(text, NumberStyles.Integer, CultureInfo.InvariantCulture, out int value)
            ? value
            : throw new InvalidOperationException($"{SamplingBestVariable} is '{text}': give a whole number (more than 1: draw from that many best; 0 or less: from the whole lattice).");
    }

    /// <summary>The sampling seed, from the setting or its variable; null for an unseeded generator.</summary>
    /// <exception cref="InvalidOperationException">The variable is not a whole number.</exception>
    public int? ResolvedSamplingSeed()
    {
        if (SamplingSeed is { } seed)
        {
            return seed;
        }

        if (Environment.GetEnvironmentVariable(SamplingSeedVariable) is not { Length: > 0 } text)
        {
            return null;
        }

        return int.TryParse(text, NumberStyles.Integer, CultureInfo.InvariantCulture, out int value)
            ? value
            : throw new InvalidOperationException($"{SamplingSeedVariable} is '{text}': give a whole number.");
    }

    /// <summary>
    /// WordPiece: a word of more characters (code points) than this is one unknown token; null takes
    /// IDRAK_TOKENIZER_MAX_WORD_CHARS, then the tokenizer's own (<c>max_input_chars_per_word</c>, 100 when it names none).
    /// </summary>
    public int? MaxInputCharsPerWord { get; init; }

    /// <summary>The longest word, from the setting, its variable, or <paramref name="file"/> (the tokenizer's own).</summary>
    /// <exception cref="ArgumentOutOfRangeException">The setting is less than 1.</exception>
    /// <exception cref="InvalidOperationException">The variable is not a whole number of 1 or more.</exception>
    public int ResolvedMaxInputCharsPerWord(int file)
    {
        if (MaxInputCharsPerWord is { } limit)
        {
            ArgumentOutOfRangeException.ThrowIfLessThan(limit, 1, nameof(MaxInputCharsPerWord));
            return limit;
        }

        if (Environment.GetEnvironmentVariable(MaxWordCharsVariable) is not { Length: > 0 } text)
        {
            return file;
        }

        return int.TryParse(text, NumberStyles.Integer, CultureInfo.InvariantCulture, out int value) && value >= 1
            ? value
            : throw new InvalidOperationException($"{MaxWordCharsVariable} is '{text}': give a whole number of characters, 1 or more.");
    }

    /// <summary>The reader's name, from the setting or its variable; null when neither sets one.</summary>
    public string? ResolvedFormat() => Format ?? (Environment.GetEnvironmentVariable(FormatVariable) is { Length: > 0 } name ? name.Trim() : null);

    /// <summary>The piece cache's size, from the setting, its variable or the default.</summary>
    /// <exception cref="ArgumentOutOfRangeException">The setting is negative.</exception>
    /// <exception cref="InvalidOperationException">The variable is not a whole number of 0 or more.</exception>
    public int ResolvedPieceCacheLimit()
    {
        if (PieceCacheLimit is { } limit)
        {
            ArgumentOutOfRangeException.ThrowIfNegative(limit, nameof(PieceCacheLimit));
            return limit;
        }

        if (Environment.GetEnvironmentVariable(PieceCacheVariable) is not { Length: > 0 } text)
        {
            return DefaultPieceCacheLimit;
        }

        return int.TryParse(text, NumberStyles.Integer, CultureInfo.InvariantCulture, out int value) && value >= 0
            ? value
            : throw new InvalidOperationException($"{PieceCacheVariable} is '{text}': give a whole number of pieces, 0 or more.");
    }
}
