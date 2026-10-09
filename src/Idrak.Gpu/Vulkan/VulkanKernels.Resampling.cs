// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

namespace Idrak.Gpu.Vulkan;

// Resampling planes of [height, width] (Backend.Interpolate2d, AdaptiveAvgPool, AdaptiveMaxPool and their gradients): an
// invocation per output for the forward passes, which computes the input positions it reads as the CPU does (in float:
// nearest min(floor(o · scale), size - 1) with the exact cases output = input and output = 2 · input; bilinear floor(s) and
// the next position with weights 1 - λ and λ), and an invocation per input element for the gradients, which gathers
// over the outputs that read it (in output order: no atomics, the same bits run after run). Push constants: planes,
// height, width, outHeight, outWidth (then mode, alignCorners, scaleHeight, scaleWidth for interpolation).
internal static partial class VulkanKernels
{
    private static IEnumerable<(string, Func<SpirvKernel>)> ResamplingKernels()
    {
        yield return ("interpolate", () =>
        {
            var k = new KernelBuilder("interpolate", Block);
            var (x, y) = (k.Buffer("x"), k.Buffer("y"));
            var p = ResamplePushes(k, interpolation: true);
            Grid(k, p.Planes * p.OutHeight * p.OutWidth, o =>
            {
                var j = o % p.OutWidth;
                var t = o / p.OutWidth;
                var (i, plane) = (t % p.OutHeight, t / p.OutHeight);
                var rows = AxisTaps(k, i, p.ScaleHeight, p.Height, p.OutHeight, p.Mode, p.Align);
                var cols = AxisTaps(k, j, p.ScaleWidth, p.Width, p.OutWidth, p.Mode, p.Align);
                var r0 = (plane * p.Height + rows.First) * p.Width;
                var r1 = (plane * p.Height + rows.Second) * p.Width;
                var value = k.Local(0f);
                k.If(p.Mode.Eq(0), () => value.V = x[r0 + cols.First], () =>
                    value.V = rows.FirstWeight * (cols.FirstWeight * x[r0 + cols.First] + cols.SecondWeight * x[r0 + cols.Second])
                              + rows.SecondWeight * (cols.FirstWeight * x[r1 + cols.First] + cols.SecondWeight * x[r1 + cols.Second]));
                y[o] = value.V;
            });
            return k.Build();
        });

        // dx[ih, iw] += Σ over the outputs reading it of their weight on it times dy, outputs in row order. The rows and
        // columns tried: those whose source position can be within two of it, the scale's whole range when it is 0.
        yield return ("interpolate_backward", () =>
        {
            var k = new KernelBuilder("interpolate_backward", Block);
            var (dy, dx) = (k.Buffer("dy"), k.Buffer("dx"));
            var p = ResamplePushes(k, interpolation: true);
            Grid(k, p.Planes * p.Height * p.Width, idx =>
            {
                var iw = idx % p.Width;
                var t = idx / p.Width;
                var (ih, plane) = (t % p.Height, t / p.Height);
                var (rowFirst, rowLast) = Readers(k, ih, p.ScaleHeight, p.OutHeight);
                var (colFirst, colLast) = Readers(k, iw, p.ScaleWidth, p.OutWidth);
                var acc = k.Local(dx[idx]);
                k.For(rowFirst, rowLast, 1, i =>
                {
                    var rows = AxisTaps(k, i, p.ScaleHeight, p.Height, p.OutHeight, p.Mode, p.Align);
                    var wr = WeightOn(k, rows, ih, p.Mode);
                    k.If(rows.First.Eq(ih) | rows.Second.Eq(ih), () =>
                    {
                        var rowBase = (plane * p.OutHeight + i) * p.OutWidth;
                        k.For(colFirst, colLast, 1, j =>
                        {
                            var cols = AxisTaps(k, j, p.ScaleWidth, p.Width, p.OutWidth, p.Mode, p.Align);
                            var wc = WeightOn(k, cols, iw, p.Mode);
                            k.If(cols.First.Eq(iw) | cols.Second.Eq(iw), () => acc.V = acc.V + wr * wc * dy[rowBase + j]);
                        });
                    });
                });
                dx[idx] = acc.V;
            });
            return k.Build();
        });

        // Adaptive windows (PyTorch's): output o covers input [floor(o · size / outSize), ceil((o + 1) · size / outSize)).
        yield return ("adaptive_avg_pool", () =>
        {
            var k = new KernelBuilder("adaptive_avg_pool", Block);
            var (x, y) = (k.Buffer("x"), k.Buffer("y"));
            var p = ResamplePushes(k, interpolation: false);
            Grid(k, p.Planes * p.OutHeight * p.OutWidth, o =>
            {
                var j = o % p.OutWidth;
                var t = o / p.OutWidth;
                var (i, plane) = (t % p.OutHeight, t / p.OutHeight);
                var (r0, r1) = Window(i, p.Height, p.OutHeight);
                var (c0, c1) = Window(j, p.Width, p.OutWidth);
                var sum = k.Local(0f);
                k.For(r0, r1, 1, r => k.For(c0, c1, 1, c => sum.V = sum.V + x[(plane * p.Height + r) * p.Width + c]));
                y[o] = sum.V / (r1 - r0).ToFloat() / (c1 - c0).ToFloat();
            });
            return k.Build();
        });

        yield return ("adaptive_avg_pool_backward", () =>
        {
            var k = new KernelBuilder("adaptive_avg_pool_backward", Block);
            var (dy, dx) = (k.Buffer("dy"), k.Buffer("dx"));
            var p = ResamplePushes(k, interpolation: false);
            Grid(k, p.Planes * p.Height * p.Width, idx =>
            {
                var iw = idx % p.Width;
                var t = idx / p.Width;
                var (ih, plane) = (t % p.Height, t / p.Height);
                var acc = k.Local(dx[idx]);
                var (rowFirst, rowLast) = Covers(k, ih, p.Height, p.OutHeight);
                var (colFirst, colLast) = Covers(k, iw, p.Width, p.OutWidth);
                k.For(rowFirst, rowLast, 1, i =>
                {
                    var (r0, r1) = Window(i, p.Height, p.OutHeight);
                    k.If((r0 <= ih) & (ih < r1), () => k.For(colFirst, colLast, 1, j =>
                    {
                        var (c0, c1) = Window(j, p.Width, p.OutWidth);
                        k.If((c0 <= iw) & (iw < c1), () =>
                            acc.V = acc.V + dy[(plane * p.OutHeight + i) * p.OutWidth + j] / (r1 - r0).ToFloat() / (c1 - c0).ToFloat());
                    }));
                });
                dx[idx] = acc.V;
            });
            return k.Build();
        });

        // The largest value of each window and its flat input index (int bits); the first in row order wins, a NaN wins
        // (each later NaN too), as the CPU's.
        yield return ("adaptive_max_pool", () =>
        {
            var k = new KernelBuilder("adaptive_max_pool", Block);
            var (x, y, argmax) = (k.Buffer("x"), k.Buffer("y"), k.Buffer("argmax"));
            var p = ResamplePushes(k, interpolation: false);
            Grid(k, p.Planes * p.OutHeight * p.OutWidth, o =>
            {
                var j = o % p.OutWidth;
                var t = o / p.OutWidth;
                var (i, plane) = (t % p.OutHeight, t / p.OutHeight);
                var (r0, r1) = Window(i, p.Height, p.OutHeight);
                var (c0, c1) = Window(j, p.Width, p.OutWidth);
                var start = plane * p.Height * p.Width;
                var best = k.Local(float.NegativeInfinity);
                var bestIndex = k.Local(r0 * p.Width + c0);
                k.For(r0, r1, 1, r => k.For(c0, c1, 1, c =>
                {
                    var v = x[start + r * p.Width + c];
                    var wins = (v > best.V) | v.IsNan();
                    bestIndex.V = k.Select(wins, r * p.Width + c, bestIndex.V);
                    best.V = k.Select(wins, v, best.V);
                }));
                y[o] = best.V;
                argmax[o] = (start + bestIndex.V).AsFloat();
            });
            return k.Build();
        });

        // dx[e] += dy[o] of each window covering element e whose argmax is e, windows in order.
        yield return ("adaptive_max_pool_backward", () =>
        {
            var k = new KernelBuilder("adaptive_max_pool_backward", Block);
            var (dy, argmax, dx) = (k.Buffer("dy"), k.Buffer("argmax"), k.Buffer("dx"));
            var p = ResamplePushes(k, interpolation: false);
            Grid(k, p.Planes * p.Height * p.Width, idx =>
            {
                var iw = idx % p.Width;
                var t = idx / p.Width;
                var (ih, plane) = (t % p.Height, t / p.Height);
                var acc = k.Local(dx[idx]);
                var (rowFirst, rowLast) = Covers(k, ih, p.Height, p.OutHeight);
                var (colFirst, colLast) = Covers(k, iw, p.Width, p.OutWidth);
                k.For(rowFirst, rowLast, 1, i => k.For(colFirst, colLast, 1, j =>
                {
                    var o = (plane * p.OutHeight + i) * p.OutWidth + j;
                    acc.V = acc.V + k.Select(argmax.Int(o).Eq(idx), dy[o], k.Float(0f));
                }));
                dx[idx] = acc.V;
            });
            return k.Build();
        });
    }

    private readonly record struct ResamplePush(Val Planes, Val Height, Val Width, Val OutHeight, Val OutWidth, Val Mode, Val Align, Val ScaleHeight, Val ScaleWidth);

    private static ResamplePush ResamplePushes(KernelBuilder k, bool interpolation)
    {
        var (planes, height, width, outHeight, outWidth) = (k.PushInt("planes"), k.PushInt("height"), k.PushInt("width"), k.PushInt("outHeight"), k.PushInt("outWidth"));
        if (!interpolation)
        {
            return new(planes, height, width, outHeight, outWidth, k.Int(0), k.Int(0), k.Float(0f), k.Float(0f));
        }

        return new(planes, height, width, outHeight, outWidth, k.PushInt("mode"), k.PushInt("align"), k.PushFloat("scaleHeight"), k.PushFloat("scaleWidth"));
    }

    private readonly record struct Taps(Val First, Val Second, Val FirstWeight, Val SecondWeight);

    // The input positions output position o of one axis reads and their weights (mode 0 nearest, 1 bilinear), as the CPU.
    private static Taps AxisTaps(KernelBuilder k, Val o, Val scale, Val size, Val outSize, Val mode, Val align)
    {
        var nearest = k.Select(outSize.Eq(size), o, k.Select(outSize.Eq(size * 2), o >> 1, k.Min(k.Floor(o.ToFloat() * scale).ToInt(), size - 1)));
        var source = k.Select(align.Ne(0), scale * o.ToFloat(), k.Max(scale * (o.ToFloat() + 0.5f) - 0.5f, k.Float(0f)));
        var index = k.Min(k.Floor(source).ToInt(), size - 1);
        var lambda = k.Clamp(source - index.ToFloat(), k.Float(0f), k.Float(1f));
        var next = index + k.Select(index < size - 1, k.Int(1), k.Int(0));
        var isNearest = mode.Eq(0);
        return new(k.Select(isNearest, nearest, index), k.Select(isNearest, nearest, next), k.Select(isNearest, k.Float(1f), 1f - lambda),
            k.Select(isNearest, k.Float(0f), lambda));
    }

    // The weight output taps put on input position i (nearest: 1 where it reads it).
    private static Val WeightOn(KernelBuilder k, Taps taps, Val i, Val mode) =>
        k.Select(taps.First.Eq(i), taps.FirstWeight, k.Float(0f)) + k.Select(taps.Second.Eq(i) & mode.Ne(0), taps.SecondWeight, k.Float(0f));

    // The outputs [first, last) whose source position can lie within two of input position i: all when the scale is 0.
    private static (Val First, Val Last) Readers(KernelBuilder k, Val i, Val scale, Val outSize)
    {
        var positive = scale > 0f;
        var safe = k.Select(positive, scale, k.Float(1f));
        var low = k.Floor((i - 2).ToFloat() / safe).ToInt() - 2;
        var high = k.Floor((i + 2).ToFloat() / safe).ToInt() + 4;
        return (k.Select(positive, k.Clamp(low, k.Int(0), outSize), k.Int(0)), k.Select(positive, k.Clamp(high, k.Int(0), outSize), outSize));
    }

    // Adaptive window o of an axis: [floor(o · size / outSize), ceil((o + 1) · size / outSize)).
    private static (Val Start, Val End) Window(Val o, Val size, Val outSize) => (o * size / outSize, ((o + 1) * size + outSize - 1) / outSize);

    // The adaptive windows that can cover input position i: [floor(i · outSize / size) - 1, ceil((i + 1) · outSize / size) + 1), clipped.
    private static (Val First, Val Last) Covers(KernelBuilder k, Val i, Val size, Val outSize) =>
        (k.Max(i * outSize / size - 1, k.Int(0)), k.Min(((i + 1) * outSize + size - 1) / size + 1, outSize));
}
