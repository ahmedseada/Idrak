// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

namespace Idrak.Backends.Vulkan;

// Convolution and pooling as generated kernels (VulkanKernels.Conv): im2col, col2im, max pooling and its gradient, and
// the per-group scale, shift and sums of a convolution's bias. A case the kernels do not cover (a storage larger than
// the device binds, more elements than a 32-bit index reaches) goes to the host fallback.
internal sealed partial class VulkanBackend
{
    public override void Im2Col(Storage x, Storage cols, in ConvGeometry g)
    {
        long n = (long)g.Positions * g.PatchSize;
        if (n > int.MaxValue || !Fit(x, cols))
        {
            base.Im2Col(x, cols, in g);
            return;
        }

        Span<byte> b = stackalloc byte[48];
        Grid("im2col", n, [x, cols], PushGeometry(b, g));
    }

    public override void Col2Im(Storage dcols, Storage dx, in ConvGeometry g)
    {
        if ((long)g.Positions * g.PatchSize > int.MaxValue || !Fit(dcols, dx))
        {
            base.Col2Im(dcols, dx, in g);
            return;
        }

        Span<byte> b = stackalloc byte[48];
        Grid("col2im", (long)g.N * g.C * g.H * g.W, [dcols, dx], PushGeometry(b, g));
    }

    public override void MaxPool(Storage x, Storage y, Storage argmax, in ConvGeometry g)
    {
        if ((long)g.N * g.C * g.H * g.W > int.MaxValue || !Fit(x, y, argmax))
        {
            base.MaxPool(x, y, argmax, in g);
            return;
        }

        Span<byte> b = stackalloc byte[48];
        Grid("max_pool", (long)g.N * g.C * g.OH * g.OW, [x, y, argmax], PushGeometry(b, g));
    }

    // The overload without the geometry stays on the host: gathering over the windows needs it, and scattering would add
    // overlapping windows' gradients in no fixed order. Tensor.MaxPool passes the geometry.
    public override void MaxPoolBackward(Storage dy, Storage argmax, Storage dx, in ConvGeometry g)
    {
        if ((long)g.N * g.C * g.H * g.W > int.MaxValue || !Fit(dy, argmax, dx))
        {
            base.MaxPoolBackward(dy, argmax, dx, in g);
            return;
        }

        Span<byte> b = stackalloc byte[48];
        Grid("max_pool_backward", (long)g.N * g.C * g.H * g.W, [dy, argmax, dx], PushGeometry(b, g));
    }

    // The twelve geometry push constants of the window kernels: N, C, H, W, KH, KW, SH, SW, PH, PW, OH, OW.
    private static ReadOnlySpan<byte> PushGeometry(Span<byte> bytes, in ConvGeometry g) =>
        new Push(bytes).I(g.N).I(g.C).I(g.H).I(g.W).I(g.KH).I(g.KW).I(g.SH).I(g.SW).I(g.PH).I(g.PW).I(g.OH).I(g.OW).Bytes;
}
