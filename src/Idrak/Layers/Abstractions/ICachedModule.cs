// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

namespace Idrak.Layers.Abstractions;

/// <summary>A module with an incremental (cached) forward pass for autoregressive decoding.</summary>
public interface ICachedModule
{
    /// <summary>
    /// Processes only the new positions in <paramref name="input"/> ([batch, newSteps, ...]), reading and
    /// extending the state kept in <paramref name="context"/>. Inference only (no gradients).
    /// </summary>
    Tensor ForwardCached(Tensor input, DecodingContext context);
}
