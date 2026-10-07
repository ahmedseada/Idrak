// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Idrak.Diagnostics;
using Idrak.Layers;
using System.Numerics;

namespace Idrak.Layers;

// Fused paths over several linear layers at once (packed products, LoRA, a projection or feed-forward in one pass).
public sealed partial class Linear
{
    /// <summary>
    /// x · W_j + scale_j · (x·A_j)·B_j for 1-3 layers with frozen weights (float32, bfloat16 or 4-bit) and LoRA adapters of
    /// one rank ≤ 32, with each low-rank term computed inside its base product (one more k step of the tensor-core kernel)
    /// instead of in separate passes over the full-width output. The backward pass does the same for the input's
    /// gradient: dx = g·Wᵀ + (scale · g·Bᵀ)·Aᵀ in one product. Null (nothing computed) when the device has no fused
    /// version; callers then run the base products and <c>Tensor.AddLowRank</c>. A frozen bias is added to each output in
    /// place (a bias that trains is not: null then).
    /// </summary>
    internal static Tensor[]? LoraProducts(Tensor input, IReadOnlyList<Layers.Linear> layers, bool withBias = true)
    {
        input.ThrowIfDisposed();
        int k = input.Shape[^1], m = input.Size / Math.Max(1, k);
        if (!input.Backend.Capabilities.MatrixUnits || layers.Count is < 1 or > 3 || m < 64 || k < 32 || !MixedPrecision.UsesTensorCores
            || layers[0].Lora is not { } first || first.Rank > 32)
        {
            return null;
        }

        // One weight form for all: float32 (one frozen, untied layer), or one built-in packed format the low-rank products
        // read (formats of one's own have no Format: their layers run their own products).
        var packedFormat = layers[0].PackedWeight?.Format;
        bool dense = layers[0].PackedWeight is null;
        foreach (var layer in layers)
        {
            bool fits = dense
                ? layer.PackedWeight is null && layer.TiedTo is null && layers.Count == 1 && !layer.Weight.RequiresGrad
                : layer.PackedWeight is { LowRankProducts: true, Format: { } own } && own == packedFormat;
            if (!fits || layer.InFeatures != k || layer.Lora is not { } adapter || adapter.Rank != first.Rank
                || adapter.A.Device != input.Device || withBias && layer.Bias is { RequiresGrad: true })
            {
                return null;
            }
        }

        long start = Telemetry.Start(TelemetryLevel.Operations);
        var backend = input.Backend;
        int rank = first.Rank;
        var flat = input.Rank == 2 ? input : input.Reshape(-1, k);
        var us = new Tensor[layers.Count];
        var outputs = new Tensor[layers.Count];
        for (int j = 0; j < layers.Count; j++)
        {
            var adapter = layers[j].Lora!;
            us[j] = Tensor.Empty([m, rank], input.Device);                                      // scale · x·A (kept for dB)
            backend.BatchedMatMul(flat.Storage, adapter.A.Storage, us[j].Storage, 1, m, rank, k, false, false, 0f);
            backend.Affine(us[j].Storage, us[j].Storage, m * rank, adapter.Scale, 0f);
            outputs[j] = Tensor.Empty([m, layers[j].OutFeatures], input.Device);
        }

        // FP8 copies of the frozen weights (Linear.AttachFloat8): y = u·B first, then x·W on FP8 tensor cores added to it.
        bool done = false;
        if (layers.All(l => l.Float8 is not null))
        {
            done = true;
            for (int j = 0; j < layers.Count && done; j++)
            {
                var f8 = layers[j].Float8!;
                backend.BatchedMatMul(us[j].Storage, layers[j].Lora!.B.Storage, outputs[j].Storage, 1, m, layers[j].OutFeatures, rank, false, false, 0f);
                done = backend.Float8MatMul(flat.Storage, m, k, f8.Values.Storage, f8.Scales.Storage, f8.Columns, outputs[j].Storage, 1f);
            }
        }

        if (!done && dense)
        {
            done = backend.MatMulLowRank(flat.Storage, layers[0].Weight.Storage, outputs[0].Storage, m, layers[0].OutFeatures, k, false, 0f,
                us[0].Storage, layers[0].Lora!.B.Storage, rank);
        }
        else if (!done)
        {
            done = backend.PackedMatMulLowRank(packedFormat!.Value, flat.Storage, m, k,
                [.. layers.Select((l, j) => (l.PackedWeight!.PackedValues.Storage, l.PackedWeight.ScaleValues?.Storage,
                    outputs[j].Storage, l.OutFeatures, us[j].Storage, l.Lora!.B.Storage))], rank);
        }

        if (!done)
        {
            foreach (var unused in us.Concat(outputs))
            {
                unused.Dispose();
            }

            if (!ReferenceEquals(flat, input))
            {
                flat.Dispose();
            }

            return null;
        }

        var results = new Tensor[layers.Count];
        for (int j = 0; j < layers.Count; j++)
        {
            var layer = layers[j];
            if (withBias && layer.Bias is { } bias)
            {
                backend.AddRowVector(outputs[j].Storage, bias.Storage, outputs[j].Storage, m, layer.OutFeatures);   // frozen: no gradient
            }

            var (a, b, scale) = (layer.Lora!.A, layer.Lora.B, layer.Lora.Scale);
            var u = us[j];
            int n = layer.OutFeatures;
            if (Autograd.IsEnabled && (flat.RequiresGrad || a.RequiresGrad || b.RequiresGrad))
            {
                outputs[j].Record("lora_fused", g =>
                {
                    if (b.RequiresGrad)
                    {
                        backend.BatchedMatMul(u.Storage, g.Storage, b.GradStorage(), 1, rank, n, m, true, false, 1f);         // dB += uᵀ·g
                    }

                    using var dt = Tensor.Empty([m, rank], flat.Device, track: false);
                    backend.BatchedMatMul(g.Storage, b.Storage, dt.Storage, 1, m, rank, n, false, true, 0f);                 // g·Bᵀ
                    backend.Affine(dt.Storage, dt.Storage, m * rank, scale, 0f);                                            // dt = scale · g·Bᵀ
                    if (a.RequiresGrad)
                    {
                        backend.BatchedMatMul(flat.Storage, dt.Storage, a.GradStorage(), 1, k, rank, m, true, false, 1f);   // dA += xᵀ·dt
                    }

                    float beta = 1f;
                    var dx = flat.RequiresGrad ? flat.GradientTarget(out beta) : null;         // the first gradient written, not added
                    if (dx is not null && layer.PackedWeight is { } stored && stored.TransposedProduct(g, dx, m, k, n, beta, dt.Storage, a.Storage, rank))
                    {
                        // dx (+)= g·Wᵀ + dt·Aᵀ with W read as the words it is stored in (bfloat16).
                    }
                    else if (dx is not null)
                    {
                        // dx += g·Wᵀ + dt·Aᵀ (W [k, n] as float32; packed weights expanded first).
                        Tensor? expanded = null;
                        if (layer.PackedWeight is { } weight)
                        {
                            expanded = Tensor.Empty([k, n], flat.Device, track: false);
                            weight.DequantizeInto(expanded.Storage);
                        }

                        using (expanded)
                        {
                            var w = (expanded ?? layer.Weight).Storage;
                            if (!backend.MatMulLowRank(g.Storage, w, dx, m, k, n, true, beta, dt.Storage, a.Storage, rank))
                            {
                                backend.BatchedMatMul(g.Storage, w, dx, 1, m, k, n, false, true, beta);
                                backend.BatchedMatMul(dt.Storage, a.Storage, dx, 1, m, k, rank, false, true, 1f);
                            }
                        }
                    }
                }, flat, a, b);
            }
            else
            {
                u.Dispose();
            }

            results[j] = input.Rank == 2 ? outputs[j] : outputs[j].Reshape([.. input.Shape[..^1], n]);
            Tensor.Traced("lora_fused", results[j], start);
        }

        return results;
    }

    /// <summary>
    /// The frozen packed layers' products of one input in one device pass (<see cref="MatMulPackedMany"/>), recorded for
    /// training: each output's gradient flows to the input (dx += g · Wᵀ with the weight expanded to float32). Null when
    /// the device has no such pass.
    /// </summary>
    internal static Tensor[]? MatMulPackedManyRecorded(Tensor input, PackedFormat format, IReadOnlyList<Layers.Linear> layers)
    {
        Tensor[]? outputs;
        using (Autograd.NoGrad())
        {
            outputs = MatMulPackedMany(input, format, layers);
        }

        if (outputs is null || !Tensor.WillRecord(input))
        {
            return outputs;
        }

        int k = input.Shape[^1], m = input.Size / k;
        for (int j = 0; j < outputs.Length; j++)
        {
            var layer = layers[j];
            int n = layer.OutFeatures;
            outputs[j].Record("matmul_packed", g =>
            {
                var weight = layer.PackedWeight!;
                if (weight.TransposedProduct(g, input.GradStorage(), m, k, n, 1f, null, null, 0))
                {
                    return;                                                                        // dx += g · Wᵀ, W as stored
                }

                using var w = weight.Dequantize();
                input.Backend.BatchedMatMul(g.Storage, w.Storage, input.GradStorage(), 1, m, k, n, false, true, 1f);   // dx += g · wᵀ
            }, input);
        }

        return outputs;
    }

    /// <summary>
    /// (act(gate) · up) · W for a packed layer W with few rows, the activation applied as the input is read (not recorded;
    /// no bias), or null when the layer is not packed or the device has no fused version.
    /// </summary>
    internal static Tensor? MatMulPackedGated(Tensor gate, Tensor up, int activation, Layers.Linear layer)
    {
        using var offload = Offloading.Enter(layer, gate);              // offloaded weights staged (else nothing)
        gate.ThrowIfDisposed();
        up.ThrowIfDisposed();
        int k = gate.Shape[^1], m = gate.Size / Math.Max(1, k);
        if (layer.PackedWeight is not { Format: { } format } weight || k != layer.InFeatures || up.Size != gate.Size || m > gate.Backend.Capabilities.FewRows)
        {
            return null;                                                // not packed, or a format the kernels do not read
        }

        var (packed, scales) = (weight.PackedValues, weight.ScaleValues);
        if (packed.Device != gate.Device)
        {
            return null;
        }

        long start = Telemetry.Start(TelemetryLevel.Operations);
        var y = Tensor.Empty([.. gate.Shape[..^1], layer.OutFeatures], gate.Device);
        if (!gate.Backend.PackedMatMulGated(format, activation, gate.Storage, up.Storage, packed.Storage, scales?.Storage, y.Storage, m, layer.OutFeatures, k))
        {
            y.Dispose();
            return null;
        }

        return Tensor.Traced("matmul_gated_packed", y, start);
    }

    /// <summary>
    /// residual + x · W for a packed layer W with few rows (no bias, no adapter) and the RMS normalization of that sum
    /// with gain, in one device pass (inference: not recorded): the output projection of a block, its residual addition
    /// and the next normalization. Null when the layer or the shapes do not fit or the device has no fused version.
    /// </summary>
    internal static (Tensor Sum, Tensor Normalized)? MatMulPackedAddRmsNorm(Tensor x, Layers.Linear layer, Tensor residual, Layers.RMSNorm norm)
    {
        using var offload = Offloading.EnterMany([layer, norm], x);
        x.ThrowIfDisposed();
        residual.ThrowIfDisposed();
        int k = x.Shape[^1], m = x.Size / Math.Max(1, k), n = layer.OutFeatures;
        if (layer.PackedWeight is not { Format: { } format } weight || layer.Bias is not null || layer.Adapter is not null || k != layer.InFeatures || m > x.Backend.Capabilities.FewRows
            || residual.Size != m * n || residual.Shape[^1] != n || norm.Features != n || !x.Backend.Capabilities.FusedKernels)
        {
            return null;
        }

        var (packed, scales) = (weight.PackedValues, weight.ScaleValues);
        if (packed.Device != x.Device || residual.Device != x.Device || norm.Gain.Device != x.Device)
        {
            return null;
        }

        long start = Telemetry.Start(TelemetryLevel.Operations);
        var y = Tensor.Empty(residual.Shape, x.Device, track: false);
        var sum = Tensor.Empty(residual.Shape, x.Device);
        var normalized = Tensor.Empty(residual.Shape, x.Device);
        bool done = x.Backend.PackedMatMulAddRmsNorm(format, x.Storage, packed.Storage, scales?.Storage, y.Storage, m, n, k, residual.Storage, sum.Storage,
            norm.Gain.Storage, normalized.Storage, norm.Epsilon, norm.Offset);
        y.Dispose();                                                 // stream-ordered: freed after the kernel read it
        if (!done)
        {
            sum.Dispose();
            normalized.Dispose();
            return null;
        }

        return (sum, Tensor.Traced("matmul_add_rms_norm_packed", normalized, start));
    }

    /// <summary>
    /// act(input · gate) · (input · up) for packed gate and up layers of one kind with few rows (no biases or adapters),
    /// the activation applied by the product's own last blocks (inference: not recorded), or null when the layers do
    /// not fit or the device has no fused version.
    /// </summary>
    internal static Tensor? MatMulPackedGatedPair(Tensor input, Layers.Linear gate, Layers.Linear up, int activation)
    {
        using var offload = Offloading.EnterMany([gate, up], input);
        input.ThrowIfDisposed();
        int k = input.Shape[^1], m = input.Size / Math.Max(1, k), n = gate.OutFeatures;
        if (gate.PackedWeight is not { Format: { } format } gateWeight || up.PackedWeight is not { } upWeight || upWeight.Format != format
            || m > input.Backend.Capabilities.FewRows || !input.Backend.Capabilities.FusedKernels || up.OutFeatures != n
            || gate.InFeatures != k || up.InFeatures != k || gate.Bias is not null || up.Bias is not null || gate.Adapter is not null
            || up.Adapter is not null)
        {
            return null;
        }

        var (gatePacked, gateScales) = (gateWeight.PackedValues, gateWeight.ScaleValues);
        var (upPacked, upScales) = (upWeight.PackedValues, upWeight.ScaleValues);
        if (gatePacked.Device != input.Device || upPacked.Device != input.Device)
        {
            return null;
        }

        long start = Telemetry.Start(TelemetryLevel.Operations);
        var gateOut = Tensor.Empty([m * n], input.Device, track: false);
        var upOut = Tensor.Empty([m * n], input.Device, track: false);
        var hidden = Tensor.Empty([.. input.Shape[..^1], n], input.Device);
        bool done = input.Backend.PackedMatMulGatedPair(format, activation, input.Storage, m, k,
            [(gatePacked.Storage, gateScales?.Storage, null, gateOut.Storage, n), (upPacked.Storage, upScales?.Storage, null, upOut.Storage, n)],
            hidden.Storage);
        gateOut.Dispose();                                           // stream-ordered: freed after the kernel read them
        upOut.Dispose();
        if (!done)
        {
            hidden.Dispose();
            return null;
        }

        return Tensor.Traced("matmul_gated_pair_packed", hidden, start);
    }

    /// <summary>
    /// The layers' packed products of one input in one device pass (few rows, not recorded), or null when the device has
    /// no single-pass version.
    /// </summary>
    internal static Tensor[]? MatMulPackedMany(Tensor input, PackedFormat format, IReadOnlyList<Layers.Linear> layers)
    {
        input.ThrowIfDisposed();
        long start = Telemetry.Start(TelemetryLevel.Operations);
        int k = input.Shape[^1], m = input.Size / k;
        var outputs = new Tensor[layers.Count];
        var products = new (Abstraction.Devices.Storage, Abstraction.Devices.Storage?, Abstraction.Devices.Storage?, Abstraction.Devices.Storage, int)[layers.Count];
        for (int j = 0; j < layers.Count; j++)
        {
            var layer = layers[j];
            var weight = layer.PackedWeight!;                                  // the caller checked: one format for all
            if (weight.Format != format)
            {
                return null;                                                   // never reached: a format the kernels do not read
            }

            var (packed, scales) = (weight.PackedValues, weight.ScaleValues);
            if (packed.Device != input.Device)
            {
                return null;
            }

            outputs[j] = Tensor.Empty([.. input.Shape[..^1], layer.OutFeatures], input.Device);
            products[j] = (packed.Storage, scales?.Storage, layer.Bias?.Storage, outputs[j].Storage, layer.OutFeatures);
        }

        if (!input.Backend.PackedMatMulMany(format, input.Storage, m, k, products))
        {
            foreach (var output in outputs)
            {
                output.Dispose();
            }

            return null;
        }

        foreach (var output in outputs)
        {
            Tensor.Traced("matmul_many_packed", output, start);
        }

        return outputs;
    }

    /// <summary>
    /// x [..., k] through several projections, their outputs side by side: [..., Σ outᵢ] (for attention: queries, keys and
    /// values packed per position, so the attention kernels read the heads in place). Gradients as for the separate
    /// projections.
    /// </summary>
    internal static Tensor? ProjectPacked(Tensor x, IReadOnlyList<Layers.Linear> layers)
    {
        int k = x.Shape[^1], m = x.Size / Math.Max(1, k), width = layers.Sum(l => l.OutFeatures);
        long start = Telemetry.Start(TelemetryLevel.Operations);
        var y = Tensor.Empty([.. x.Shape[..^1], width], x.Device);
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

        return Tensor.Traced("project_packed", y, start);
    }

    /// <summary>
    /// down(gelu(up(x))) (GELU tanh approximation, biases optional) with the activation applied as the up projection's
    /// products are stored, and its gradient applied as the down projection's input gradient is: two products forward,
    /// no separate activation passes. Keeps the pre-activations for the backward pass.
    /// </summary>
    internal static Tensor? FeedForwardGelu(Tensor x, Layers.Linear up, Layers.Linear down)
    {
        int d = x.Shape[^1], m = x.Size / Math.Max(1, d), f = up.OutFeatures;
        Tensor wu = up.Weight, wd = down.Weight;
        Tensor? bu = up.Bias, bd = down.Bias;
        var parameters = new[] { wu, bu, wd, bd }.OfType<Tensor>().ToList();
        bool record = Autograd.IsEnabled && (x.RequiresGrad || parameters.Any(p => p.RequiresGrad));
        long start = Telemetry.Start(TelemetryLevel.Operations);
        var backend = x.Backend;
        var act = Tensor.Empty([m, f], x.Device, track: false);
        var pre = record ? Tensor.Empty([m, f], x.Device, track: false) : null;
        var y = Tensor.Empty(x.Shape, x.Device);
        if (!backend.GemmStrided(x.Storage, 0, d, false, wu.Storage, 0, f, false, act.Storage, 0, f, m, f, d, 0f, bu?.Storage,
                Abstraction.Devices.GemmEpilogue.Gelu, pre?.Storage)
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
            return Tensor.Traced("feed_forward_gelu", y, start);
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
            using var dPre = Tensor.Empty([m, f], x.Device, track: false);
            backend.GemmStrided(g.Storage, 0, d, false, wd.Storage, 0, d, true, dPre.Storage, 0, f, m, f, d, 0f, null,
                Abstraction.Devices.GemmEpilogue.GeluGradient, pre!.Storage);
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
        return Tensor.Traced("feed_forward_gelu", y, start);
    }
}
