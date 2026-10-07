// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Idrak.Abstraction.Devices;
using Idrak.Abstraction.Diagnostics;
using Idrak.Abstraction.Generation;

namespace Idrak.Abstraction;

// Products with packed weights and attention over key/value caches (their types live here since phase 3b).
public sealed partial class Tensor
{


    // Up to this many rows the product reads the int8 weights directly (token-by-token decoding: 4× less weight traffic);
    // above it the weights are dequantized once and the float matrix product is used.
    private const int Int8DirectRows = 8;



    /// <summary>
    /// [..., k] × <paramref name="weight"/> ([k, n], int8) → [..., n]. Gradients flow to this tensor (the int8 weights are
    /// fixed), so layers before a quantized layer, and LoRA adapters on it, can still be trained.
    /// </summary>
    internal Tensor MatMulInt8(Int8Weight weight)
    {
        this.ThrowIfDisposed();
        int k = weight.Rows, n = weight.Columns;
        if (this.Rank < 2 || this._shape[^1] != k)
        {
            throw new ArgumentException($"MatMul with an int8 [{k}, {n}] weight needs [..., {k}], got {Tensor.FormatShape(this._shape)}.");
        }

        if (weight.Packed.Device != this.Device)
        {
            throw new ArgumentException($"The int8 weight is on {weight.Packed.Device}, the input on {this.Device}.");
        }

        long start = Telemetry.Start(TelemetryLevel.Operations);
        var flat = this.Rank == 2 ? this : this.Reshape(-1, k);
        int m = flat._shape[0];
        var y = Tensor.Empty([m, n], this.Device);
        if (m <= Int8DirectRows)
        {
            // Few rows: the direct kernels, unless the device's packed product is faster for this shape.
            if (!(this.Backend.PrefersPackedMatMul(PackedFormat.Int8, flat.Storage, weight.Packed.Storage, weight.Scales.Storage, y.Storage, m, n, k)
                  && this.Backend.PackedMatMulLarge(PackedFormat.Int8, flat.Storage, weight.Packed.Storage, weight.Scales.Storage, y.Storage, m, n, k)))
            {
                this.Backend.Int8MatMul(flat.Storage, weight.Packed.Storage, weight.Scales.Storage, y.Storage, m, n, k);
            }
        }
        else if (!this.Backend.PackedMatMulLarge(PackedFormat.Int8, flat.Storage, weight.Packed.Storage, weight.Scales.Storage, y.Storage, m, n, k))
        {
            using var w = Dequantized(weight);
            this.Backend.MatMul(flat.Storage, w.Storage, y.Storage, m, n, k, false, false, 0f);
        }

        if (Tensor.WillRecord(flat))
        {
            y.Record("matmul_int8", g =>
            {
                using var w = Dequantized(weight);
                flat.Backend.BatchedMatMul(g.Storage, w.Storage, flat.GradStorage(), 1, m, k, n, false, true, 1f);   // dx += dy · wᵀ
            }, flat);
        }

        var result = this.Rank == 2 ? y : y.Reshape([.. this._shape[..^1], n]);
        return Tensor.Traced("matmul_int8", result, start);
    }



    /// <summary>
    /// [..., k] × <paramref name="weight"/> ([k, n], bfloat16) → [..., n]. Few rows read the bfloat16 weights directly;
    /// more expand them to float32 once. Gradients flow to this tensor (the weights are fixed).
    /// </summary>
    internal Tensor MatMulBFloat16(BFloat16Weight weight)
    {
        this.ThrowIfDisposed();
        int k = weight.Rows, n = weight.Columns;
        if (this.Rank < 2 || this._shape[^1] != k)
        {
            throw new ArgumentException($"MatMul with a bfloat16 [{k}, {n}] weight needs [..., {k}], got {Tensor.FormatShape(this._shape)}.");
        }

        if (weight.Packed.Device != this.Device)
        {
            throw new ArgumentException($"The bfloat16 weight is on {weight.Packed.Device}, the input on {this.Device}.");
        }

        long start = Telemetry.Start(TelemetryLevel.Operations);
        var flat = this.Rank == 2 ? this : this.Reshape(-1, k);
        int m = flat._shape[0];
        var y = Tensor.Empty([m, n], this.Device);
        this.Backend.BFloat16MatMul(flat.Storage, weight.Packed.Storage, y.Storage, m, n, k);
        if (Tensor.WillRecord(flat))
        {
            y.Record("matmul_bf16", g =>
            {
                if (flat.Backend.BFloat16TransposedMatMul(g.Storage, weight.Packed.Storage, flat.GradStorage(), m, k, n, 1f, null, null, 0))
                {
                    return;                                                                        // dx += dy · wᵀ, w as stored
                }

                using var w = Tensor.Empty([k, n], weight.Packed.Device, track: false);
                weight.Packed.Backend.BFloat16Dequantize(weight.Packed.Storage, w.Storage, k, n);
                flat.Backend.BatchedMatMul(g.Storage, w.Storage, flat.GradStorage(), 1, m, k, n, false, true, 1f);   // dx += dy · wᵀ
            }, flat);
        }

        var result = this.Rank == 2 ? y : y.Reshape([.. this._shape[..^1], n]);
        return Tensor.Traced("matmul_bf16", result, start);
    }



    /// <summary>
    /// [..., k] × <paramref name="weight"/> ([k, n], 4-bit) → [..., n]. Few rows read the nibbles directly; more expand
    /// them to float32 once. Gradients flow to this tensor (the weights are fixed), so LoRA adapters on it train (QLoRA).
    /// </summary>
    internal Tensor MatMulInt4(Int4Weight weight)
    {
        this.ThrowIfDisposed();
        int k = weight.Rows, n = weight.Columns;
        if (this.Rank < 2 || this._shape[^1] != k)
        {
            throw new ArgumentException($"MatMul with an int4 [{k}, {n}] weight needs [..., {k}], got {Tensor.FormatShape(this._shape)}.");
        }

        if (weight.Packed.Device != this.Device)
        {
            throw new ArgumentException($"The int4 weight is on {weight.Packed.Device}, the input on {this.Device}.");
        }

        long start = Telemetry.Start(TelemetryLevel.Operations);
        var flat = this.Rank == 2 ? this : this.Reshape(-1, k);
        int m = flat._shape[0];
        var y = Tensor.Empty([m, n], this.Device);
        this.Backend.Int4MatMul(flat.Storage, weight.Packed.Storage, weight.Scales.Storage, y.Storage, m, n, k);
        if (Tensor.WillRecord(flat))
        {
            y.Record("matmul_int4", g =>
            {
                using var w = Tensor.Empty([k, n], weight.Packed.Device, track: false);
                weight.Packed.Backend.Int4Dequantize(weight.Packed.Storage, weight.Scales.Storage, w.Storage, k, n);
                flat.Backend.BatchedMatMul(g.Storage, w.Storage, flat.GradStorage(), 1, m, k, n, false, true, 1f);   // dx += dy · wᵀ
            }, flat);
        }

        var result = this.Rank == 2 ? y : y.Reshape([.. this._shape[..^1], n]);
        return Tensor.Traced("matmul_int4", result, start);
    }



    /// <summary>
    /// [..., k] × <paramref name="weight"/> ([k, n], a format without a product of its own) → [..., n]: the weights
    /// expanded with <see cref="PackedWeight.Dequantize"/> for the product (and again for the gradient), freed after it.
    /// Gradients flow to this tensor (the weights are fixed). Not recordable into a device graph: the expansion may read
    /// the host, which a replay would not repeat.
    /// </summary>
    internal Tensor MatMulExpanded(PackedWeight weight)
    {
        this.ThrowIfDisposed();
        int k = weight.Rows, n = weight.Columns;
        if (this.Rank < 2 || this._shape[^1] != k)
        {
            throw new ArgumentException($"MatMul with a {weight.Name} [{k}, {n}] weight needs [..., {k}], got {Tensor.FormatShape(this._shape)}.");
        }

        if (ComputeGraph.IsCapturing)
        {
            throw new NotSupportedException($"{weight.Name} weights have no product of their own (PackedWeight.MatMul), so the step is not recorded as a graph.");
        }

        long start = Telemetry.Start(TelemetryLevel.Operations);
        var flat = this.Rank == 2 ? this : this.Reshape(-1, k);
        int m = flat._shape[0];
        var y = Tensor.Empty([m, n], this.Device);
        using (var w = Expanded(weight, this.Device))
        {
            this.Backend.MatMul(flat.Storage, w.Storage, y.Storage, m, n, k, false, false, 0f);
        }

        if (Tensor.WillRecord(flat))
        {
            y.Record("matmul_packed", g =>
            {
                using var w = Expanded(weight, flat.Device);
                flat.Backend.BatchedMatMul(g.Storage, w.Storage, flat.GradStorage(), 1, m, k, n, false, true, 1f);   // dx += dy · wᵀ
            }, flat);
        }

        var result = this.Rank == 2 ? y : y.Reshape([.. this._shape[..^1], n]);
        return Tensor.Traced("matmul_packed", result, start);
    }



    /// <summary>
    /// x · Wᵀ for a frozen <paramref name="weight"/> W [outputs, inputs] also kept as <paramref name="transposed"/> Wᵀ
    /// [inputs, outputs]: the product reads Wᵀ as stored and the input's gradient (g · W) reads W as stored, so neither
    /// direction makes a transposed copy of W (the tensor-core kernels copy transposed operands into place first).
    /// W receives no gradient.
    /// </summary>
    public static Tensor MatMulFrozenTransposed(Tensor input, Tensor weight, BFloat16Weight transposed)
    {
        Tensor output;
        using (Autograd.NoGrad())
        {
            output = input.MatMulBFloat16(transposed);
        }

        if (Tensor.WillRecord(input))
        {
            int outputs = weight._shape[0], inputs = weight._shape[1];
            output.Record("matmul_frozen", g =>
            {
                // dx += g · W, accumulated by the product itself (no temporary).
                input.Backend.BatchedMatMul(g.Storage, weight.Storage, input.GradStorage(), 1, g.Size / outputs, inputs, outputs, false, false, 1f);
            }, input);
        }

        return output;
    }



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



    /// <summary>Attention of q [heads, rowsPerHead, dim] over an int8 cache filled up to <paramref name="position"/>.</summary>
    internal static Tensor AttentionInt8(Tensor q, KeyValueCache cache, Tensor position, int steps, float scale, bool tiled, AttentionVariant variant = default)
    {
        long start = Telemetry.Start(TelemetryLevel.Operations);
        int heads = q._shape[0], rowsPerHead = q._shape[1], dim = q._shape[2];
        var y = Tensor.Empty([heads, rowsPerHead, dim], q.Device);
        q.Backend.AttentionInt8(q.Storage, cache.Keys.Storage, cache.Values.Storage, cache.KeyScales!.Storage, cache.ValueScales!.Storage,
            position.Storage, y.Storage, heads, rowsPerHead, steps, cache.Keys._shape[1], dim, scale, tiled, variant);
        return Tensor.Traced("attention_int8", y, start);
    }



    /// <summary>Attention of q [heads, rows, dim] over the filled part of a bfloat16 cache.</summary>
    internal static Tensor AttentionBFloat16(Tensor q, KeyValueCache cache, Tensor position, int steps, float scale, bool tiled, AttentionVariant variant = default)
    {
        long start = Telemetry.Start(TelemetryLevel.Operations);
        int heads = q._shape[0], rowsPerHead = q._shape[1], dim = q._shape[2];
        var y = Tensor.Empty([heads, rowsPerHead, dim], q.Device);
        q.Backend.AttentionBFloat16(q.Storage, cache.Keys.Storage, cache.Values.Storage, position.Storage, y.Storage, heads, rowsPerHead, steps,
            cache.Keys._shape[1], dim, scale, tiled, variant);

        return Tensor.Traced("attention_bf16", y, start);
    }



    /// <summary>q [rows, steps, dim] · int8 keysᵀ → [rows, steps, capacity].</summary>
    internal static Tensor AttentionScoresInt8(Tensor q, KeyValueCache cache)
    {
        int rows = q._shape[0], steps = q._shape[1], capacity = cache.Keys._shape[1];
        var y = Tensor.Empty([rows, steps, capacity], q.Device);
        q.Backend.AttentionScoresInt8(q.Storage, cache.Keys.Storage, cache.KeyScales!.Storage, y.Storage, rows, steps, capacity, cache.HeadDim);
        return y;
    }



    /// <summary>weights [rows, steps, capacity] · int8 values → [rows, steps, dim].</summary>
    internal static Tensor AttentionContextInt8(Tensor weights, KeyValueCache cache)
    {
        int rows = weights._shape[0], steps = weights._shape[1], capacity = weights._shape[2];
        var y = Tensor.Empty([rows, steps, cache.HeadDim], weights.Device);
        weights.Backend.AttentionContextInt8(weights.Storage, cache.Values.Storage, cache.ValueScales!.Storage, y.Storage, rows, steps, capacity, cache.HeadDim);
        return y;
    }



    // The weights as float32 [rows, columns], checked: on `device`, the shape the format reports.
    private static Tensor Expanded(PackedWeight weight, Device device)
    {
        var w = weight.Dequantize();
        if (w.Device != device || w.Rank != 2 || w.Shape[0] != weight.Rows || w.Shape[1] != weight.Columns)
        {
            w.Dispose();
            throw new InvalidOperationException(
                $"{weight.Name}.Dequantize gave {Tensor.FormatShape(w.Shape)} on {w.Device}; a [{weight.Rows}, {weight.Columns}] product on {device} needs that shape there.");
        }

        return w;
    }



    private static Tensor Dequantized(Int8Weight weight)
    {
        var w = Tensor.Empty([weight.Rows, weight.Columns], weight.Packed.Device, track: false);
        weight.Packed.Backend.Int8Dequantize(weight.Packed.Storage, weight.Scales.Storage, w.Storage, weight.Rows, weight.Columns);
        return w;
    }



    /// <summary>Embedding lookup from a bfloat16 [vocabulary, dim] table (fixed: no gradient).</summary>
    public static Tensor EmbeddingLookup(BFloat16Weight table, Tensor indices)
    {
        indices.ThrowIfDisposed();
        if (table.Packed.Device != indices.Device)
        {
            throw new ArgumentException($"The embedding table is on {table.Packed.Device}, the ids on {indices.Device}.");
        }

        long start = Telemetry.Start(TelemetryLevel.Operations);
        var y = Tensor.Empty([.. indices._shape, table.Columns], indices.Device);
        indices.Backend.GatherBFloat16(table.Packed.Storage, indices.Storage, y.Storage, indices.Size, table.Columns, table.Rows);
        return Tensor.Traced("embedding_bf16", y, start);
    }
}
