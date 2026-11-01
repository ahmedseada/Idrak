// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Globalization;
using Idrak.Abstraction.Devices;
using Weights = Idrak.Gpu.Hip.GcnKernels.PackedWeights;

namespace Idrak.Gpu.Hip;

// HIP's bodies for the operations CUDA overrode and HIP did not (plan 35, HIP parity): the token cross-entropy rows and
// the packed gated activation on their GCN kernels (GcnKernels.Parity.cs), and the packed product's epilogue. CUDA's
// epilogue runs in its tensor-core products' tail; HIP has no matrix units yet (G9), so its prompt-sized packed products
// expand the weights and run the float32 tiled product (HipBackend.Products.cs), whose tail adds the bias: the product and
// its bias in one pass, then the residual added in the contract's order (residual + (product + bias)), the same bits as
// the composed passes. GELU (which needs the bias) keeps the composed path, whose BiasGelu already adds the bias.
internal sealed unsafe partial class HipBackend
{
    public override void SoftmaxCrossEntropyRowsKernel(Storage logits, Storage targets, Storage weights, Storage losses, int rows, int vocabulary, float scale)
    {
        if (rows <= 0)
        {
            return;
        }

        if ((long)rows * vocabulary >= GcnKernels.PointwiseMostElements || RowKernel("softmax_ce_rows_f32", vocabulary) is not { } f)
        {
            base.SoftmaxCrossEntropyRowsKernel(logits, targets, weights, losses, rows, vocabulary, scale);
            return;
        }

        LaunchRows(f, rows, vocabulary, Cost(f.Name).Flops, P(logits), P(targets), P(weights), P(losses), U(vocabulary), F(scale), U(rows));
    }

    public override void GatedActivationPackedKernel(Storage gate, Storage up, Storage packedGate, Storage packedUp, Storage y, Storage packedY, int n, int kind, int flags)
    {
        if (n <= 0)
        {
            return;
        }

        int words = (n + 1) / 2;
        if (PointwiseKernel("gated_act_bf16_f32", n) is not { } f)
        {
            base.GatedActivationPackedKernel(gate, up, packedGate, packedUp, y, packedY, n, kind, flags);
            return;
        }

        if (Profiling)
        {
            LabelNextLaunch(string.Create(CultureInfo.InvariantCulture, $"{f.Name} {n}"), Cost(f.Name).Flops * n);
        }

        LaunchKernel(f, PointwiseBlocks(words), 1, P(gate), P(up), P(packedGate), P(packedUp), P(y), P(packedY), U(kind), U(flags), U(n), U(words));
    }

    public override void PackedMatMulEpilogueKernel(PackedFormat format, Storage x, Storage packed, Storage? scales, Storage y, int m, int n, int k, Storage? bias,
        Storage? residual, bool gelu)
    {
        Weights? weights = format switch
        {
            PackedFormat.BFloat16 => Weights.BFloat16,
            PackedFormat.Int8 when scales is not null => Weights.Int8,
            PackedFormat.Int4 when scales is not null => Weights.Int4,
            _ => null,
        };
        if (weights is not { } expanded || gelu || bias is null || m <= GcnKernels.GemvRows || n <= 0 || k <= 0 || (long)k * n >= GcnKernels.PointwiseMostElements
            || (long)m * n >= GcnKernels.PointwiseMostElements)
        {
            base.PackedMatMulEpilogueKernel(format, x, packed, scales, y, m, n, k, bias, residual, gelu);       // few rows, GELU, no bias: the passes
            return;
        }

        var w = Allocate(k * n, zeroed: false);
        Storage? product = residual is null ? null : Allocate(m * n, zeroed: false);
        try
        {
            if (!Expand(expanded, packed, scales, w, k, n) || !FloatProduct(x, w, product ?? y, 1, m, n, k, false, false, 0f, bias))
            {
                base.PackedMatMulEpilogueKernel(format, x, packed, scales, y, m, n, k, bias, residual, gelu);
                return;
            }

            if (residual is not null)
            {
                Binary(BinaryOp.Add, residual, product!, y, m * n);
            }
        }
        finally
        {
            w.Release();                                                         // reused in stream order
            product?.Release();
        }
    }
}
