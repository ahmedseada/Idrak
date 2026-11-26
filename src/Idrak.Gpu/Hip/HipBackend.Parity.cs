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
}
