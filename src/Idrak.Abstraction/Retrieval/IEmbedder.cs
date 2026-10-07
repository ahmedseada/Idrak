// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

namespace Idrak.Abstraction.Retrieval;

/// <summary>
/// Turns texts into vectors whose dot product (or cosine) measures how related they are. <c>TextEncoder</c> is the
/// built-in one; implement this to index with another model or a hosted embedding service.
/// </summary>
public interface IEmbedder
{
    /// <summary>The vectors of <paramref name="texts"/>, in the same order, all of the same length.</summary>
    ValueTask<float[][]> EmbedAsync(IReadOnlyList<string> texts, CancellationToken cancellationToken = default);
}
