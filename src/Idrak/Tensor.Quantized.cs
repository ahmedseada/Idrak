using Idrak.Diagnostics;
using Idrak.Layers;

namespace Idrak;

// Matrix products with int8 weights (weight-only quantization).
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
        ThrowIfDisposed();
        int k = weight.Rows, n = weight.Columns;
        if (Rank < 2 || _shape[^1] != k)
        {
            throw new ArgumentException($"MatMul with an int8 [{k}, {n}] weight needs [..., {k}], got {FormatShape(_shape)}.");
        }

        if (weight.Packed.Device != Device)
        {
            throw new ArgumentException($"The int8 weight is on {weight.Packed.Device}, the input on {Device}.");
        }

        long start = Telemetry.Start(TelemetryLevel.Operations);
        var flat = Rank == 2 ? this : Reshape(-1, k);
        int m = flat._shape[0];
        var y = Empty([m, n], Device);
        if (m <= Int8DirectRows)
        {
            Backend.Int8MatMul(flat.Storage, weight.Packed.Storage, weight.Scales.Storage, y.Storage, m, n, k);
        }
        else if (!Backend.PackedMatMulLarge(0, flat.Storage, weight.Packed.Storage, weight.Scales.Storage, y.Storage, m, n, k))
        {
            using var w = Dequantized(weight);
            Backend.MatMul(flat.Storage, w.Storage, y.Storage, m, n, k, false, false, 0f);
        }

        if (WillRecord(flat))
        {
            y.Record("matmul_int8", g =>
            {
                using var w = Dequantized(weight);
                flat.Backend.BatchedMatMul(g.Storage, w.Storage, flat.GradStorage(), 1, m, k, n, false, true, 1f);   // dx += dy · wᵀ
            }, flat);
        }

        var result = Rank == 2 ? y : y.Reshape([.. _shape[..^1], n]);
        return Traced("matmul_int8", result, start);
    }

    /// <summary>
    /// [..., k] × <paramref name="weight"/> ([k, n], bfloat16) → [..., n]. Few rows read the bfloat16 weights directly;
    /// more expand them to float32 once. Gradients flow to this tensor (the weights are fixed).
    /// </summary>
    internal Tensor MatMulBFloat16(BFloat16Weight weight)
    {
        ThrowIfDisposed();
        int k = weight.Rows, n = weight.Columns;
        if (Rank < 2 || _shape[^1] != k)
        {
            throw new ArgumentException($"MatMul with a bfloat16 [{k}, {n}] weight needs [..., {k}], got {FormatShape(_shape)}.");
        }

        if (weight.Packed.Device != Device)
        {
            throw new ArgumentException($"The bfloat16 weight is on {weight.Packed.Device}, the input on {Device}.");
        }

        long start = Telemetry.Start(TelemetryLevel.Operations);
        var flat = Rank == 2 ? this : Reshape(-1, k);
        int m = flat._shape[0];
        var y = Empty([m, n], Device);
        Backend.BFloat16MatMul(flat.Storage, weight.Packed.Storage, y.Storage, m, n, k);
        if (WillRecord(flat))
        {
            y.Record("matmul_bf16", g =>
            {
                if (flat.Backend.BFloat16TransposedMatMul(g.Storage, weight.Packed.Storage, flat.GradStorage(), m, k, n, 1f, null, null, 0))
                {
                    return;                                                                        // dx += dy · wᵀ, w as stored
                }

                using var w = Empty([k, n], weight.Packed.Device, track: false);
                weight.Packed.Backend.BFloat16Dequantize(weight.Packed.Storage, w.Storage, k, n);
                flat.Backend.BatchedMatMul(g.Storage, w.Storage, flat.GradStorage(), 1, m, k, n, false, true, 1f);   // dx += dy · wᵀ
            }, flat);
        }

        var result = Rank == 2 ? y : y.Reshape([.. _shape[..^1], n]);
        return Traced("matmul_bf16", result, start);
    }

    /// <summary>
    /// [..., k] × <paramref name="weight"/> ([k, n], 4-bit) → [..., n]. Few rows read the nibbles directly; more expand
    /// them to float32 once. Gradients flow to this tensor (the weights are fixed), so LoRA adapters on it train (QLoRA).
    /// </summary>
    internal Tensor MatMulInt4(Int4Weight weight)
    {
        ThrowIfDisposed();
        int k = weight.Rows, n = weight.Columns;
        if (Rank < 2 || _shape[^1] != k)
        {
            throw new ArgumentException($"MatMul with an int4 [{k}, {n}] weight needs [..., {k}], got {FormatShape(_shape)}.");
        }

        if (weight.Packed.Device != Device)
        {
            throw new ArgumentException($"The int4 weight is on {weight.Packed.Device}, the input on {Device}.");
        }

        long start = Telemetry.Start(TelemetryLevel.Operations);
        var flat = Rank == 2 ? this : Reshape(-1, k);
        int m = flat._shape[0];
        var y = Empty([m, n], Device);
        Backend.Int4MatMul(flat.Storage, weight.Packed.Storage, weight.Scales.Storage, y.Storage, m, n, k);
        if (WillRecord(flat))
        {
            y.Record("matmul_int4", g =>
            {
                using var w = Empty([k, n], weight.Packed.Device, track: false);
                weight.Packed.Backend.Int4Dequantize(weight.Packed.Storage, weight.Scales.Storage, w.Storage, k, n);
                flat.Backend.BatchedMatMul(g.Storage, w.Storage, flat.GradStorage(), 1, m, k, n, false, true, 1f);   // dx += dy · wᵀ
            }, flat);
        }

        var result = Rank == 2 ? y : y.Reshape([.. _shape[..^1], n]);
        return Traced("matmul_int4", result, start);
    }

    private static Tensor Dequantized(Int8Weight weight)
    {
        var w = Empty([weight.Rows, weight.Columns], weight.Packed.Device, track: false);
        weight.Packed.Backend.Int8Dequantize(weight.Packed.Storage, weight.Scales.Storage, w.Storage, weight.Rows, weight.Columns);
        return w;
    }
}
