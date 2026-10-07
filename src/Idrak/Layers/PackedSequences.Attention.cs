// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Idrak.Backends;
using Idrak.Diagnostics;
using Idrak.Layers;
using System.Numerics;

namespace Idrak.Layers;

// Causal attention over packed sequences, each attending within its own segment.
public sealed partial class PackedSequences
{
    /// <summary>
    /// <see cref="Tensor.CausalAttention"/> over packed sequences: keys and values [heads, steps, dim], and each position sees
    /// only its own sequence (<paramref name="packing"/>'s starts; head h belongs to packed row h / <paramref name="headsPerRow"/>).
    /// Null when the device has no such pass.
    /// </summary>
    internal static Tensor? CausalAttentionSegmented(Tensor q, Tensor keys, Tensor values, Layers.PackedSequences packing, int headsPerRow, float scale,
        AttentionVariant variant = default)
    {
        long start = Telemetry.Start(TelemetryLevel.Operations);
        int heads = q.Shape[0], rowsPerHead = q.Shape[1], dim = q.Shape[2], steps = keys.Shape[1];
        var y = Tensor.Empty([heads, rowsPerHead, dim], q.Device);
        bool record = Autograd.IsEnabled && (q.RequiresGrad || keys.RequiresGrad || values.RequiresGrad);
        var lse = record ? Tensor.Empty([heads, rowsPerHead], q.Device, track: false) : null;
        if (!q.Backend.AttentionSegmented(q.Storage, keys.Storage, values.Storage, y.Storage, lse?.Storage, packing.Starts.Storage, packing.Ends.Storage,
                heads, headsPerRow, rowsPerHead, steps, dim, scale, variant))
        {
            y.Dispose();
            lse?.Dispose();
            return null;
        }

        if (record)
        {
            y.Record("attention_packed", g =>
            {
                using var dq = q.RequiresGrad ? null : Tensor.Empty(q.Shape, q.Device, zeroed: true, track: false);
                using var dk = keys.RequiresGrad ? null : Tensor.Empty(keys.Shape, q.Device, zeroed: true, track: false);
                using var dv = values.RequiresGrad ? null : Tensor.Empty(values.Shape, q.Device, zeroed: true, track: false);
                if (!q.Backend.AttentionSegmentedBackward(q.Storage, keys.Storage, values.Storage, y.Storage, lse!.Storage, g.Storage,
                        dq?.Storage ?? q.GradStorage(), dk?.Storage ?? keys.GradStorage(), dv?.Storage ?? values.GradStorage(),
                        packing.Starts.Storage, packing.Ends.Storage, heads, headsPerRow, rowsPerHead, steps, dim, scale, variant))
                {
                    throw new NotSupportedException("Packed attention's backward pass is not available on this device.");
                }

                lse.Dispose();
            }, q, keys, values);
        }

        return Tensor.Traced("attention_packed", y, start);
    }
}
