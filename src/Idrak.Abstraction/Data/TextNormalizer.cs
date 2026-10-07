// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

namespace Idrak.Abstraction.Data;

/// <summary>
/// Rewrites a text by the consumer's rules (case, spacing, punctuation, a language's letter forms and digits …) so texts
/// that mean the same compare equal. The library has no rules of its own: implement this and apply it with
/// <c>Dataset.Normalize</c> (Idrak.Datasets) before the steps that compare values (<c>Dataset.Deduplicate</c>,
/// a split key). Implementations are called from one thread per enumeration; keep them free of shared mutable state.
/// </summary>
public interface ITextNormalizer
{
    /// <summary>The normalized form of <paramref name="text"/>.</summary>
    string Normalize(string text);
}
