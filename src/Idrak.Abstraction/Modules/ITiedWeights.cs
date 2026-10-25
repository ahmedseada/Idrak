// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

namespace Idrak.Abstraction;

/// <summary>
/// A layer that can read its weights from another module's (an output head tied to the token embeddings): while it is
/// tied, its weights are that module's, so training, adapters and saving count them once, there.
/// </summary>
public interface ITiedWeights
{
    /// <summary>The module whose weights this layer reads, or null when it holds its own.</summary>
    Module? WeightsOf { get; }
}
