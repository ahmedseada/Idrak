// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

namespace Idrak.Text.Abstractions;

/// <summary>
/// Random text in one language, a line at a time, for drawing training pages when there is no text to draw (plan 18):
/// the words, word shapes, numbers and punctuation a reader of that language meets, not meaningful sentences. Register
/// one with <c>RandomTextSources.Register</c>; <see cref="PooledTextSource"/> is one made from word pools.
/// </summary>
public interface IRandomTextSource
{
    /// <summary>The source's name, a language tag for a language's own ("ar").</summary>
    string Name { get; }

    /// <summary>A line of exactly <paramref name="words"/> words (numbers and dates count as words), from <paramref name="random"/>.</summary>
    string Line(Random random, int words, RandomTextOptions options);
}

