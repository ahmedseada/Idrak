// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Text.RegularExpressions;

namespace Idrak.Models;

/// <summary>
/// A tokenizer's pattern (an added-token alternation, a pre-tokenizer's split, a normalizer's replace), compiled once and
/// matched on each thread through a copy of its own (<see cref="Local"/>). A <see cref="Regex"/> keeps one matcher: a
/// thread that finds it taken by another builds a new one for that match, so threads sharing one instance allocate on
/// every match and run slower than one thread (measured on 4 threads: 20-123 bytes a match and 1.6-2.6 times the time).
/// A copy shares the compiled pattern (its runner factory, as a source-generated <see cref="Regex"/> does), so it costs a
/// few microseconds once per thread, not a compile; it matches as the pattern does in every API.
/// </summary>
internal sealed class TokenizerRegex : Regex
{
    // This thread's copy (null on a copy itself).
    private readonly ThreadLocal<TokenizerRegex>? _copies;

    /// <summary>Compiles <paramref name="pattern"/> with the tokenizers' options (<see cref="BpeTokenizer.Fast"/>).</summary>
    public TokenizerRegex(string pattern)
        : base(pattern, BpeTokenizer.Fast) => _copies = new ThreadLocal<TokenizerRegex>(() => new TokenizerRegex(this));

    // A copy of `pattern` with a matcher of its own: the protected state a derived Regex sets (as generated ones do).
    private TokenizerRegex(TokenizerRegex pattern)
    {
        this.pattern = pattern.pattern;
        roptions = pattern.roptions;
        factory = pattern.factory;
        internalMatchTimeout = pattern.internalMatchTimeout;
        caps = pattern.caps;
        capnames = pattern.capnames;
        capslist = pattern.capslist;
        capsize = pattern.capsize;
    }

    /// <summary>The calling thread's copy: match through it, once per call that matches (it is never contended).</summary>
    public Regex Local => _copies is null ? this : _copies.Value!;
}
