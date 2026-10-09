// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

namespace Idrak.Gpu.Vulkan;

// Convolution and pooling as generated kernels (VulkanKernels.Conv): im2col, col2im, max pooling and its gradient, and
// the per-group scale, shift and sums of a convolution's bias, dilated windows included. A case the kernels do not cover
// (a storage larger than the device binds, more elements than a 32-bit index reaches) goes to the host fallback.
internal sealed partial class VulkanBackend
{
    public override void Im2ColKernel(Storage x, Storage cols, in ConvGeometry g)
    {
        long n = (long)g.Positions * g.PatchSize;
        if (n > int.MaxValue || !Fit(x, cols))
        {
            base.Im2ColKernel(x, cols, in g);
            return;
        }

        Span<byte> b = stackalloc byte[56];
        Grid("im2col", n, [x, cols], PushGeometry(b, g));
    }

    public override void Col2ImKernel(Storage dcols, Storage dx, in ConvGeometry g)
    {
        if ((long)g.Positions * g.PatchSize > int.MaxValue || !Fit(dcols, dx))
        {
            base.Col2ImKernel(dcols, dx, in g);
            return;
        }

        Span<byte> b = stackalloc byte[56];
        Grid("col2im", (long)g.N * g.C * g.H * g.W, [dcols, dx], PushGeometry(b, g));
    }

    public override void MaxPoolKernel(Storage x, Storage y, Storage argmax, in ConvGeometry g)
    {
        if ((long)g.N * g.C * g.H * g.W > int.MaxValue || !Fit(x, y, argmax))
        {
            base.MaxPoolKernel(x, y, argmax, in g);
            return;
        }

        Span<byte> b = stackalloc byte[56];
        Grid("max_pool", (long)g.N * g.C * g.OH * g.OW, [x, y, argmax], PushGeometry(b, g));
    }

    // The overload without the geometry stays on the host: gathering over the windows needs it, and scattering would add
    // overlapping windows' gradients in no fixed order. Tensor.MaxPool passes the geometry.
    public override void MaxPoolBackwardKernel(Storage dy, Storage argmax, Storage dx, in ConvGeometry g)
    {
        if ((long)g.N * g.C * g.H * g.W > int.MaxValue || !Fit(dy, argmax, dx))
        {
            base.MaxPoolBackwardKernel(dy, argmax, dx, in g);
            return;
        }

        Span<byte> b = stackalloc byte[56];
        Grid("max_pool_backward", (long)g.N * g.C * g.H * g.W, [dy, argmax, dx], PushGeometry(b, g));
    }

    // The fourteen geometry push constants of the window kernels: N, C, H, W, KH, KW, SH, SW, PH, PW, OH, OW, DH, DW.
    private static ReadOnlySpan<byte> PushGeometry(Span<byte> bytes, in ConvGeometry g) =>
        new Push(bytes).I(g.N).I(g.C).I(g.H).I(g.W).I(g.KH).I(g.KW).I(g.SH).I(g.SW).I(g.PH).I(g.PW).I(g.OH).I(g.OW).I(g.DH).I(g.DW).Bytes;
}
