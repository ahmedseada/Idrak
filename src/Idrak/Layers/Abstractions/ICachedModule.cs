// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

namespace Idrak.Layers.Abstractions;

/// <summary>
/// A module with an incremental (cached) forward pass for autoregressive decoding. A pass runs in a step of the context
/// (<see cref="DecodingContext.OpenStep"/>): a module that may be called on its own opens one and completes it when its
/// pass ends; inside a network that opened the step already, the scope joins it, so only the outermost counts the
/// positions. A module's own state lives in the context, found by the module itself as its key
/// (<see cref="DecodingContext.CacheFor(object, int, int)"/>, by identity), so one context serves every layer of a model.
/// </summary>
public interface ICachedModule
{
    /// <summary>
    /// Processes only the new positions in <paramref name="input"/> ([batch, newSteps, ...]), reading and
    /// extending the state kept in <paramref name="context"/>. Inference only (no gradients).
    /// </summary>
    Tensor ForwardCached(Tensor input, DecodingContext context);
}
