// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Idrak.Diagnostics;
using Idrak.Layers;
using System.Numerics;

namespace Idrak.Layers;

// Normalization and rotary embedding of attention heads in one pass.
public sealed partial class RMSNorm
{
    /// <summary>
    /// An attention layer's projections q [batch, steps, heads·dim], k and v [batch, steps, kvHeads·dim] in the layouts
    /// attention reads, in one pass where the device can (inference: not recorded): each query and key head
    /// RMS-normalized with gain (when the norms are given) and rotated (when the tables are given), queries as
    /// [batch·kvHeads, group·steps, dim] (the heads sharing a key/value head stacked), keys and values as
    /// [batch·kvHeads, steps, dim], or, with a cache, written into it at <paramref name="position"/> (then K and V are
    /// null). Null when the device has no fused version.
    /// </summary>
    internal static (Tensor Q, Tensor? K, Tensor? V)? NormRopeHeads(Tensor q, Tensor k, Tensor v, int heads, int kvHeads, int dim,
        Layers.RMSNorm? queryNorm, Layers.RMSNorm? keyNorm, Tensor? cos, Tensor? sin, Tensor positions, bool interleaved,
        Abstraction.Generation.KeyValueCache? cache, Tensor? position)
    {
        q.ThrowIfDisposed();
        k.ThrowIfDisposed();
        v.ThrowIfDisposed();
        if ((queryNorm is null) != (keyNorm is null) || (cos is null) != (sin is null) || (cache is null) != (position is null)
            || cache is { Layout.FusedWrite: false } || !q.Backend.Capabilities.FusedKernels)
        {
            return null;
        }

        long start = Telemetry.Start(TelemetryLevel.Operations);
        int n = q.Shape[0], t = q.Shape[1];
        var yq = Tensor.Empty([n * kvHeads, heads / kvHeads * t, dim], q.Device);
        Tensor? yk = null, yv = null;
        int capacity = t, stride = dim;
        bool bfloat16 = cache is { Layout.HalfWords: true };
        if (cache is null)
        {
            yk = Tensor.Empty([n * kvHeads, t, dim], q.Device);
            yv = Tensor.Empty([n * kvHeads, t, dim], q.Device);
        }
        else
        {
            capacity = cache.Keys.Shape[1];
            stride = bfloat16 ? 2 * cache.Keys.Shape[2] : cache.Keys.Shape[2];
        }

        if (!q.Backend.NormRopeHeads(q.Storage, k.Storage, v.Storage, n, t, heads, kvHeads, dim, queryNorm?.Gain.Storage, queryNorm?.Epsilon ?? 0f,
            queryNorm?.Offset ?? 0f, keyNorm?.Gain.Storage, keyNorm?.Epsilon ?? 0f, keyNorm?.Offset ?? 0f, cos?.Storage, sin?.Storage, positions.Storage,
            cos?.Shape[1] ?? 0, interleaved, yq.Storage, (yk ?? cache!.Keys).Storage, (yv ?? cache!.Values).Storage, position?.Storage, capacity, stride,
            bfloat16))
        {
            yq.Dispose();
            yk?.Dispose();
            yv?.Dispose();
            return null;
        }

        Tensor.Traced("norm_rope_heads", yq, start);
        return (yq, yk, yv);
    }
}
