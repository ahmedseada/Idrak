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

    /// <summary>The variable <see cref="Workers"/> falls back to.</summary>
    public const string WorkersVariable = "IDRAK_TOKENIZER_WORKERS";

    /// <summary>The variable <see cref="MaxInputCharsPerWord"/> falls back to.</summary>
    public const string MaxWordCharsVariable = "IDRAK_TOKENIZER_MAX_WORD_CHARS";

    /// <summary>The number of encoded pieces a tokenizer keeps when nothing sets <see cref="PieceCacheLimit"/>.</summary>
    public const int DefaultPieceCacheLimit = 100_000;

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

    /// <summary>
    /// How many threads a batch encode (<see cref="IPretrainedTokenizer.EncodeBatch"/>) spreads over (0: every core); null
    /// takes IDRAK_TOKENIZER_WORKERS, then 0.
    /// </summary>
    public int? Workers { get; init; }

    /// <summary>The batch threads, from the setting, its variable or 0 (every core).</summary>
    /// <exception cref="ArgumentOutOfRangeException">The setting is negative.</exception>
    /// <exception cref="InvalidOperationException">The variable is not a whole number of 0 or more.</exception>
    public int ResolvedWorkers()
    {
        if (Workers is { } workers)
        {
            ArgumentOutOfRangeException.ThrowIfNegative(workers, nameof(Workers));
            return workers;
        }

        if (Environment.GetEnvironmentVariable(WorkersVariable) is not { Length: > 0 } text)
        {
            return 0;
        }

        return int.TryParse(text, NumberStyles.Integer, CultureInfo.InvariantCulture, out int value) && value >= 0
            ? value
            : throw new InvalidOperationException($"{WorkersVariable} is '{text}': give a whole number of threads, 0 (every core) or more.");
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
