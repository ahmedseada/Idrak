// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Idrak.Abstraction.Diagnostics;

namespace Idrak.Abstraction;

// Training building blocks that run as few passes as possible on bfloat16 tensor cores (MixedPrecision.BFloat16):
// the query/key/value projections into one packed tensor, attention reading its heads in place, and a GELU
// feed-forward block with the activation inside the products. Each returns null when the device cannot run it, and
// callers then use the composed operations.
public sealed partial class Tensor
{    /// <summary>
    /// Causal self-attention over packed projections [batch, steps, (heads + 2·kvHeads)·dim] (queries, keys, values per
    /// position; query head h uses key/value head h / (heads / kvHeads)), read in place: [batch, steps, heads·dim].
    /// </summary>
    public static Tensor? CausalAttentionPacked(Tensor packed, int heads, int kvHeads, int dim, float scale)
    {
        int batch = packed._shape[0], steps = packed._shape[1], width = packed._shape[2], group = heads / kvHeads;
        long kOffset = (long)heads * dim, vOffset = (long)(heads + kvHeads) * dim;
        long start = Telemetry.Start(TelemetryLevel.Operations);
        var y = Empty([batch, steps, heads * dim], packed.Device);
        bool record = Autograd.IsEnabled && packed.RequiresGrad;
        var lse = record ? Empty([batch * kvHeads * group * steps], packed.Device, track: false) : null;
        if (!packed.Backend.AttentionStrided(packed.Storage, 0, packed.Storage, kOffset, packed.Storage, vOffset, width, width,
                y.Storage, lse?.Storage, batch, kvHeads, group, steps, dim, scale))
        {
            lse?.Dispose();
            y.Dispose();
            return null;
        }

        if (record)
        {
            y.Record("attention_packed", g =>
            {
                var grad = packed.GradStorage();
                packed.Backend.AttentionStridedBackward(packed.Storage, 0, packed.Storage, kOffset, packed.Storage, vOffset, width, width,
                    y.Storage, lse!.Storage, g.Storage, grad, 0, grad, kOffset, grad, vOffset, batch, kvHeads, group, steps, dim, scale);
                lse.Dispose();
            }, packed);
        }

        return Traced("attention_packed", y, start);
    }
}
