// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Globalization;
using Idrak.Abstraction.Devices;

namespace Idrak.Gpu.Hip;

// HIP's bodies for the operations CUDA overrode and HIP did not (plan 35, HIP parity): the token cross-entropy rows on
// their GCN kernel (GcnKernels.Parity.cs).
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
