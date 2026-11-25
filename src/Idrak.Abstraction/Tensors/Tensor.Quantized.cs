// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Idrak.Abstraction.Devices;
using Idrak.Abstraction.Diagnostics;
using Idrak.Abstraction.Generation;

namespace Idrak.Abstraction;

// Attention over a float key/value cache.
public sealed partial class Tensor
{
    /// <summary>
    /// Attention of q [heads, rowsPerHead, dim] over a float cache filled up to <paramref name="position"/> (see
    /// Backend.AttentionDecode): the softmax and the weighted values in one pass, reading only filled positions.
    /// </summary>
    internal static Tensor AttentionDecode(Tensor q, KeyValueCache cache, Tensor position, int steps, float scale, AttentionVariant variant = default)
    {
        long start = Telemetry.Start(TelemetryLevel.Operations);
        int heads = q._shape[0], rowsPerHead = q._shape[1], dim = q._shape[2];
        var y = Tensor.Empty([heads, rowsPerHead, dim], q.Device);
        q.Backend.AttentionDecode(q.Storage, cache.Keys.Storage, cache.Values.Storage, position.Storage, y.Storage, heads, rowsPerHead, steps,
            cache.Keys._shape[1], dim, scale, variant);
        return Tensor.Traced("attention_decode", y, start);
    }
}
