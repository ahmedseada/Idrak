// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

namespace Idrak.Gpu.Vulkan;

// Convolution and pooling kernels over NCHW data (im2col, col2im, max pooling and its gradient), and the per-group
// scale, shift and sums that a convolution's bias and batch statistics use. Each window kernel takes the geometry as
// fourteen push constants: N, C, H, W, KH, KW, SH, SW, PH, PW, OH, OW, DH, DW (the dilation; pooling passes 1). The
// gradients gather (an invocation per input element walks the windows that cover it, in the CPU's order) rather than
// scatter, so they need no atomics and give the same bits run after run.
internal static partial class VulkanKernels
{
    private static IEnumerable<(string, Func<SpirvKernel>)> ConvKernels()
    {
        // cols[(n, oh, ow), (c, kh, kw)] = x[n, c, oh·sh - ph + kh·dh, ow·sw - pw + kw·dw], 0 outside the image.
        yield return ("im2col", () =>
        {
            var k = new KernelBuilder("im2col", Block);
            var (x, cols) = (k.Buffer("x"), k.Buffer("cols"));
            var g = Geometry(k);
            var patch = g.C * g.KH * g.KW;
            Grid(k, g.N * g.OH * g.OW * patch, idx =>
            {
                var (row, col) = (idx / patch, idx % patch);
                var ow = row % g.OW;
                var t = row / g.OW;
                var (oh, n) = (t % g.OH, t / g.OH);
                var kw = col % g.KW;
                var u = col / g.KW;
                var (kh, c) = (u % g.KH, u / g.KH);
                var ih = oh * g.SH - g.PH + kh * g.DH;
                var iw = ow * g.SW - g.PW + kw * g.DW;
                var inside = (ih >= 0) & (ih < g.H) & (iw >= 0) & (iw < g.W);
                var at = ((n * g.C + c) * g.H + k.Clamp(ih, k.Int(0), g.H - 1)) * g.W + k.Clamp(iw, k.Int(0), g.W - 1);
                cols[idx] = k.Select(inside, x[at], k.Float(0f));
            });
            return k.Build();
        });

        // dx += fold(dcols): an invocation per input element adds the column entries that read it, windows in row order
        // (a dilated window reads the element only where its offset from the window's start is a whole number of steps).
        yield return ("col2im", () =>
        {
            var k = new KernelBuilder("col2im", Block);
            var (dcols, dx) = (k.Buffer("dcols"), k.Buffer("dx"));
            var g = Geometry(k);
            var patch = g.C * g.KH * g.KW;
            Grid(k, g.N * g.C * g.H * g.W, idx =>
            {
                var iw = idx % g.W;
                var t = idx / g.W;
                var ih = t % g.H;
                t = t / g.H;
                var (c, n) = (t % g.C, t / g.C);
                var (ohFirst, ohLast) = Covering(k, ih, g.PH, (g.KH - 1) * g.DH + 1, g.SH, g.OH);
                var (owFirst, owLast) = Covering(k, iw, g.PW, (g.KW - 1) * g.DW + 1, g.SW, g.OW);
                var acc = k.Local(dx[idx]);
                k.For(ohFirst, ohLast, 1, oh =>
                {
                    var th = ih + g.PH - oh * g.SH;
                    k.If((th % g.DH).Eq(0), () =>
                    {
                        var rowBase = (n * g.OH + oh) * g.OW;
                        var colBase = (c * g.KH + th / g.DH) * g.KW;
                        k.For(owFirst, owLast, 1, ow =>
                        {
                            var tw = iw + g.PW - ow * g.SW;
                            k.If((tw % g.DW).Eq(0), () => acc.V = acc.V + dcols[(rowBase + ow) * patch + colBase + tw / g.DW]);
                        });
                    });
                });
                dx[idx] = acc.V;
            });
            return k.Build();
        });

        // y = the largest value of each window; argmax = its flat input index (int bits). The first maximum in row order
        // wins; padded positions never do; a NaN wins and stays (as the CPU's Max); a window wholly in the padding gives
        // -inf at the plane's first element.
        yield return ("max_pool", () =>
        {
            var k = new KernelBuilder("max_pool", Block);
            var (x, y, argmax) = (k.Buffer("x"), k.Buffer("y"), k.Buffer("argmax"));
            var g = Geometry(k);
            Grid(k, g.N * g.C * g.OH * g.OW, idx =>
            {
                var ow = idx % g.OW;
                var t = idx / g.OW;
                var (oh, nc) = (t % g.OH, t / g.OH);
                var plane = nc * g.H * g.W;
                var r0 = oh * g.SH - g.PH;
                var c0 = ow * g.SW - g.PW;
                var (rEnd, cEnd) = (k.Min(r0 + g.KH, g.H), k.Min(c0 + g.KW, g.W));
                var best = k.Local(float.NegativeInfinity);
                var bestIndex = k.Local(0);
                k.For(k.Max(r0, k.Int(0)), rEnd, 1, r =>
                {
                    var rowOffset = r * g.W;
                    k.For(k.Max(c0, k.Int(0)), cEnd, 1, c =>
                    {
                        var v = x[plane + rowOffset + c];
                        var greater = v > best.V;
                        bestIndex.V = k.Select(greater, rowOffset + c, bestIndex.V);
                        best.V = k.Select(greater | v.IsNan(), v, best.V);
                    });
                });
                y[idx] = best.V;
                argmax[idx] = plane + bestIndex.V;
            });
            return k.Build();
        });

        // dx[argmax[o]] += dy[o]: an invocation per input element adds dy of the windows that cover it and chose it, in
        // window order. Where a window can lie wholly in the padding (padding ≥ window), its argmax is the plane's first
        // element, so that element's invocation walks every window of the plane.
        yield return ("max_pool_backward", () =>
        {
            var k = new KernelBuilder("max_pool_backward", Block);
            var (dy, argmax, dx) = (k.Buffer("dy"), k.Buffer("argmax"), k.Buffer("dx"));
            var g = Geometry(k);
            Grid(k, g.N * g.C * g.H * g.W, idx =>
            {
                var iw = idx % g.W;
                var t = idx / g.W;
                var (ih, nc) = (t % g.H, t / g.H);
                var acc = k.Local(dx[idx]);
                var all = ((g.PH >= g.KH) | (g.PW >= g.KW)) & ih.Eq(0) & iw.Eq(0);
                var (ohFirst, ohLast) = Covering(k, ih, g.PH, g.KH, g.SH, g.OH);
                var (owFirst, owLast) = Covering(k, iw, g.PW, g.KW, g.SW, g.OW);
                ohFirst = k.Select(all, k.Int(0), ohFirst);
                ohLast = k.Select(all, g.OH, ohLast);
                owFirst = k.Select(all, k.Int(0), owFirst);
                owLast = k.Select(all, g.OW, owLast);
                k.For(ohFirst, ohLast, 1, oh =>
                {
                    var rowBase = (nc * g.OH + oh) * g.OW;
                    k.For(owFirst, owLast, 1, ow =>
                    {
                        var o = rowBase + ow;
                        acc.V = acc.V + k.Select(argmax.Int(o).Eq(idx), dy[o], k.Float(0f));
                    });
                });
                dx[idx] = acc.V;
            });
            return k.Build();
        });
    }

    // The window geometry push constants, in the order every window kernel declares them.
    private readonly record struct WindowGeometry(Val N, Val C, Val H, Val W, Val KH, Val KW, Val SH, Val SW, Val PH, Val PW, Val OH, Val OW, Val DH, Val DW);

    private static WindowGeometry Geometry(KernelBuilder k) => new(
        k.PushInt("N"), k.PushInt("C"), k.PushInt("H"), k.PushInt("W"), k.PushInt("KH"), k.PushInt("KW"),
        k.PushInt("SH"), k.PushInt("SW"), k.PushInt("PH"), k.PushInt("PW"), k.PushInt("OH"), k.PushInt("OW"), k.PushInt("DH"), k.PushInt("DW"));

    // The windows [first, last) along one axis whose span covers input coordinate i: o·stride - pad ≤ i < o·stride - pad + size,
    // within [0, outputs).
    private static (Val First, Val Last) Covering(KernelBuilder k, Val i, Val pad, Val size, Val stride, Val outputs)
    {
        var low = i + pad - size + 1;                                    // o·stride ≥ low
        var first = k.Select(low > 0, (low + stride - 1) / stride, k.Int(0));
        var last = k.Min((i + pad) / stride + 1, outputs);
        return (first, last);
    }
}
