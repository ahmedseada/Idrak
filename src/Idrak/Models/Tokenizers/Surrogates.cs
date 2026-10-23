// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Buffers;

namespace Idrak.Models;

/// <summary>
/// The UTF-16 surrogates (U+D800 to U+DFFF) as <see cref="SearchValues{T}"/>: the tokenizers' check for half of a pair is
/// one vectorized search through the non-generic char overloads. The generic range searches
/// (<c>ContainsAnyInRange</c>, <c>IndexOfAnyInRange</c>) box their bounds in the code a method runs before it is
/// optimized (measured: 96 bytes a call while the JIT is busy elsewhere), which a warm encode must not.
/// </summary>
internal static class Surrogates
{
    /// <summary>Every surrogate code unit, high and low.</summary>
    public static readonly SearchValues<char> Any = SearchValues.Create(string.Create(0x800, 0, static (chars, _) =>
    {
        for (int i = 0; i < chars.Length; i++)
        {
            chars[i] = (char)(0xD800 + i);
        }
    }));
}
