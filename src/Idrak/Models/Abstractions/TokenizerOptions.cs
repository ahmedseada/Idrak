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
