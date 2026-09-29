using Idrak.Diagnostics;

namespace Idrak;

// Training building blocks that run as few passes as possible on bfloat16 tensor cores (MixedPrecision.BFloat16):
// the query/key/value projections into one packed tensor, attention reading its heads in place, and a GELU
// feed-forward block with the activation inside the products. Each returns null when the device cannot run it, and
// callers then use the composed operations.
public sealed partial class Tensor
{
    /// <summary>
    /// x [..., k] through several projections, their outputs side by side: [..., Σ outᵢ] (for attention: queries, keys and
    /// values packed per position, so the attention kernels read the heads in place). Gradients as for the separate
    /// projections.
    /// </summary>
    internal static Tensor? ProjectPacked(Tensor x, IReadOnlyList<Layers.Linear> layers)
    {
        int k = x._shape[^1], m = x.Size / Math.Max(1, k), width = layers.Sum(l => l.OutFeatures);
        long start = Telemetry.Start(TelemetryLevel.Operations);
        var y = Empty([.. x._shape[..^1], width], x.Device);
        var offsets = new int[layers.Count];
        using var reuse = x.Backend.ReuseQuantizedOperands();              // x is quantized once for all projections
        for (int j = 0, offset = 0; j < layers.Count; offset += layers[j].OutFeatures, j++)
        {
            offsets[j] = offset;
            var layer = layers[j];
            if (!x.Backend.GemmStrided(x.Storage, 0, k, false, layer.Weight.Storage, 0, layer.OutFeatures, false,
                    y.Storage, offset, width, m, layer.OutFeatures, k, 0f, layer.Bias?.Storage))
            {
                y.Dispose();
                return null;
            }
        }

        var parameters = layers.SelectMany(l => l.Bias is null ? new[] { l.Weight } : [l.Weight, l.Bias]).ToList();
        if (Autograd.IsEnabled && (x.RequiresGrad || parameters.Any(p => p.RequiresGrad)))
        {
            y.Record("project_packed", g =>
            {
                var backend = x.Backend;
                using var reuseBackward = backend.ReuseQuantizedOperands();  // xᵀ once for the weight gradients
                for (int j = 0; j < layers.Count; j++)
                {
                    var layer = layers[j];
                    int n = layer.OutFeatures;
                    if (layer.Bias is { RequiresGrad: true } bias)
                    {
                        backend.SumColumns(g.Storage, offsets[j], width, bias.GradStorage(), m, n);
                    }

                    if (layer.Weight.RequiresGrad)
                    {
                        backend.GemmStrided(x.Storage, 0, k, true, g.Storage, offsets[j], width, false, layer.Weight.GradStorage(), 0, n, k, n, m, 1f);
                    }

                    if (x.RequiresGrad)
                    {
                        backend.GemmStrided(g.Storage, offsets[j], width, false, layer.Weight.Storage, 0, n, true, x.GradStorage(), 0, k, m, k, n, 1f);
                    }
                }
            }, [x, .. parameters]);
        }

        return Traced("project_packed", y, start);
    }

    /// <summary>
    /// Causal self-attention over packed projections [batch, steps, (heads + 2·kvHeads)·dim] (queries, keys, values per
    /// position; query head h uses key/value head h / (heads / kvHeads)), read in place: [batch, steps, heads·dim].
    /// </summary>
    internal static Tensor? CausalAttentionPacked(Tensor packed, int heads, int kvHeads, int dim, float scale)
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

    /// <summary>
    /// down(gelu(up(x))) (GELU tanh approximation, biases optional) with the activation applied as the up projection's
    /// products are stored, and its gradient applied as the down projection's input gradient is: two products forward,
    /// no separate activation passes. Keeps the pre-activations for the backward pass.
    /// </summary>
    internal static Tensor? FeedForwardGelu(Tensor x, Layers.Linear up, Layers.Linear down)
    {
        int d = x._shape[^1], m = x.Size / Math.Max(1, d), f = up.OutFeatures;
        Tensor wu = up.Weight, wd = down.Weight;
        Tensor? bu = up.Bias, bd = down.Bias;
        var parameters = new[] { wu, bu, wd, bd }.OfType<Tensor>().ToList();
        bool record = Autograd.IsEnabled && (x.RequiresGrad || parameters.Any(p => p.RequiresGrad));
        long start = Telemetry.Start(TelemetryLevel.Operations);
        var backend = x.Backend;
        var act = Empty([m, f], x.Device, track: false);
        var pre = record ? Empty([m, f], x.Device, track: false) : null;
        var y = Empty(x._shape, x.Device);
        if (!backend.GemmStrided(x.Storage, 0, d, false, wu.Storage, 0, f, false, act.Storage, 0, f, m, f, d, 0f, bu?.Storage,
                Backends.GemmEpilogue.Gelu, pre?.Storage)
            || !backend.GemmStrided(act.Storage, 0, f, false, wd.Storage, 0, d, false, y.Storage, 0, d, m, d, f, 0f, bd?.Storage))
        {
            act.Dispose();
            pre?.Dispose();
            y.Dispose();
            return null;
        }

        if (!record)
        {
            act.Dispose();
            return Traced("feed_forward_gelu", y, start);
        }

        y.Record("feed_forward_gelu", g =>
        {
            if (bd is { RequiresGrad: true })
            {
                backend.SumRows(g.Storage, bd.GradStorage(), m, d);
            }

            if (wd.RequiresGrad)
            {
                backend.GemmStrided(act.Storage, 0, f, true, g.Storage, 0, d, false, wd.GradStorage(), 0, d, f, d, m, 1f);
            }

            // dPre = (g · Wdᵀ) ∘ gelu'(pre), in the product's epilogue.
            using var dPre = Empty([m, f], x.Device, track: false);
            backend.GemmStrided(g.Storage, 0, d, false, wd.Storage, 0, d, true, dPre.Storage, 0, f, m, f, d, 0f, null,
                Backends.GemmEpilogue.GeluGradient, pre!.Storage);
            if (bu is { RequiresGrad: true })
            {
                backend.SumRows(dPre.Storage, bu.GradStorage(), m, f);
            }

            if (wu.RequiresGrad)
            {
                backend.GemmStrided(x.Storage, 0, d, true, dPre.Storage, 0, f, false, wu.GradStorage(), 0, f, d, f, m, 1f);
            }

            if (x.RequiresGrad)
            {
                backend.GemmStrided(dPre.Storage, 0, f, false, wu.Storage, 0, f, true, x.GradStorage(), 0, d, m, d, f, 1f);
            }

            act.Dispose();
            pre.Dispose();
        }, [x, .. parameters]);
        return Traced("feed_forward_gelu", y, start);
    }
}
