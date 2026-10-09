// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Idrak.Abstraction.Devices;

namespace Idrak.Abstraction.Testing;

// Image operations: the convolution and its two gradients (groups, depthwise, dilation, rectangular windows, a bias and each
// activation), average pooling and its gradient, and resampling with per-channel normalization (float and 8-bit), each
// against plain loops.
public static partial class DeviceCases
{
    private static void Convolutions(DeviceCaseContext c)
    {
        var b = c.Backend;

        // (geometry, filters, groups): plain, grouped and dilated, depthwise (one and two filters a channel), pointwise, one
        // with more filters and positions than a 64 x 64 tile and a depth past 16, and a random one.
        int G = c.Size(1, 3), perGroup = c.Size(1, 3), filtersPerGroup = c.Size(1, 3);
        var random = new ConvGeometry(c.Size(1, 3), G * perGroup, c.Size(4, 12), c.Size(4, 12), c.Size(1, 3), c.Size(1, 3), c.Size(1, 2), c.Size(1, 2),
            c.Random.Next(3), c.Random.Next(3)) { DH = c.Size(1, 2), DW = c.Size(1, 2) };
        (ConvGeometry Geometry, int Filters, int Groups)[] cases =
        [
            (new(2, 3, 7, 5, 3, 3, 1, 1, 1, 1), 4, 1),
            (new(1, 4, 9, 8, 3, 2, 2, 1, 1, 0) { DH = 2 }, 6, 2),
            (new(2, 3, 8, 7, 3, 3, 1, 1, 1, 1), 3, 3),
            (new(1, 2, 9, 6, 5, 3, 2, 1, 2, 1) { DW = 2 }, 4, 2),
            (new(2, 8, 5, 5, 1, 1, 1, 1, 0, 0), 5, 1),
            (new(2, 16, 12, 11, 3, 3, 1, 1, 1, 1), 70, 1),
            (new(1, 4, 7, 9, 3, 3, 2, 2, 0, 1) { PadBottom = 2, PadRight = 0 }, 4, 4),
            (random, G * filtersPerGroup, G),
        ];

        foreach (var (g, filters, groups) in cases)
        {
            if (g.OH <= 0 || g.OW <= 0)
            {
                continue;
            }

            int input = g.N * g.C * g.H * g.W, output = g.N * filters * g.OH * g.OW, weights = filters * (g.PatchSize / groups);
            var xs = c.Values(input);
            var ws = c.Values(weights, 0.5f);
            var bs = c.Values(filters);
            var x = c.Storage(xs);
            var w = c.Storage(ws);
            string shape = $"{g.N}x{g.C}x{g.H}x{g.W}, {g.KH}x{g.KW} stride {g.SH}x{g.SW} padding {g.PH}x{g.PW} dilation {g.DH}x{g.DW}, {filters} filters in {groups} groups";

            var y = c.Zeros(output);
            b.Convolution(x, w, c.Storage(bs), y, g, filters, groups, ConvActivation.None);
            c.ExpectClose(ConvolutionReference(xs, ws, bs, g, filters, groups, ConvActivation.None), Read(y), 2e-4f, $"convolution {shape}");
            foreach (var activation in Enum.GetValues<ConvActivation>().Where(a => a != ConvActivation.None))
            {
                var ya = c.Zeros(output);
                b.Convolution(x, w, null, ya, g, filters, groups, activation);
                c.ExpectClose(ConvolutionReference(xs, ws, null, g, filters, groups, activation), Read(ya), 2e-4f, $"convolution {shape}, {activation}");
            }

            var dys = c.Values(output);
            var dy = c.Storage(dys);
            var dx0 = c.Values(input);
            var dx = c.Storage(dx0);
            b.ConvolutionBackwardInput(dy, w, dx, g, filters, groups);
            c.ExpectClose(ConvolutionInputGradient(dys, ws, dx0, g, filters, groups), Read(dx), 2e-4f, $"convolution input gradient {shape}");

            var dw0 = c.Values(weights);
            var dw = c.Storage(dw0);
            b.ConvolutionBackwardWeight(x, dy, dw, g, filters, groups);
            c.ExpectClose(ConvolutionWeightGradient(xs, dys, dw0, g, filters, groups), Read(dw), 5e-4f, $"convolution weight gradient {shape}");
        }

        // Average pooling: square, rectangular, strided, padded (both divisors), a window past the padded border's start.
        ConvGeometry[] pools =
        [
            new(1, 1, 2, 2, 2, 2, 2, 2, 0, 0), new(2, 3, 8, 6, 2, 2, 2, 2, 0, 0), new(2, 2, 7, 10, 3, 2, 1, 2, 1, 1), new(1, 3, 9, 9, 3, 3, 2, 2, 1, 1),
            new(c.Size(1, 3), c.Size(1, 4), c.Size(3, 12), c.Size(3, 12), 3, 3, c.Size(1, 3), c.Size(1, 3), c.Random.Next(2), c.Random.Next(2)),

            // A ceil-mode geometry (more padding below and right for the last window, not counted in its divisor) and
            // asymmetric padding that is counted.
            new(2, 2, 8, 7, 3, 3, 2, 2, 1, 0) { PadBottom = 2, PadRight = 2 },
            new(1, 3, 6, 9, 2, 3, 2, 2, 0, 1) { PadBottom = 1, PadRight = 2 },
        ];
        foreach (var g in pools)
        {
            int input = g.N * g.C * g.H * g.W, count = g.N * g.C * g.OH * g.OW;
            foreach (bool countPad in new[] { true, false })
            {
                // The padding a divisor counts: the geometry's, or (the ceil mode's) less.
                int bottom = Math.Max(g.PH, g.PadBottom - 1), right = Math.Max(g.PW, g.PadRight - 1);
                var xs = c.Values(input);
                var y = c.Zeros(count);
                b.AvgPool(c.Storage(xs), y, g, countPad, bottom, right);
                c.ExpectClose(AvgPoolReference(xs, g, countPad, bottom, right), Read(y), 1e-5f, $"average pooling {g} (padding counted: {countPad})");
                var dys = c.Values(count);
                var dx0 = c.Values(input);
                var dx = c.Storage(dx0);
                b.AvgPoolBackward(c.Storage(dys), dx, g, countPad, bottom, right);
                c.ExpectClose(AvgPoolGradient(dys, dx0, g, countPad, bottom, right), Read(dx), 1e-5f, $"average pooling gradient {g} (padding counted: {countPad})");
            }
        }
    }

    private static void ResizeNormalizeCase(DeviceCaseContext c)
    {
        var b = c.Backend;

        // (channels, planes, height, width, outHeight, outWidth): shrink, enlarge, one axis kept (its pass skipped), both kept,
        // a random one.
        (int Channels, int Planes, int H, int W, int OH, int OW)[] sizes =
        [
            (3, 3, 17, 23, 8, 11), (1, 2, 5, 4, 12, 9), (3, 6, 9, 13, 9, 7), (2, 2, 6, 6, 6, 6), (3, 3, 10, 12, 21, 12),
            (c.Size(1, 3), 0, c.Size(1, 40), c.Size(1, 40), c.Size(1, 40), c.Size(1, 40)),
        ];
        foreach (var (channels, planesGiven, h, w, oh, ow) in sizes)
        {
            int planes = planesGiven > 0 ? planesGiven : channels * c.Size(1, 2);
            foreach (bool antialias in new[] { true, false })
            {
                var (across, xTaps) = w == ow ? ([], 0) : Tensor.BilinearTaps(w, ow, antialias);
                var (down, yTaps) = h == oh ? ([], 0) : Tensor.BilinearTaps(h, oh, antialias);

                // Floats: values from 0 to 1, a scale and shift per channel.
                var xs = c.Values(planes * h * w).Select(v => v * 0.5f + 0.5f).ToArray();
                float[] coefficients = [.. across, .. down, 0f], affine = c.Values(2 * channels);
                var y = c.Zeros(planes * oh * ow);
                b.ResizeNormalize(c.Storage(xs), c.Storage(coefficients), c.Storage(affine), y, planes, channels, h, w, oh, ow, xTaps, yTaps, bytes: false);
                c.ExpectClose(ResampleReference(xs, coefficients, affine, planes, channels, h, w, oh, ow, xTaps, yTaps, bytes: false), Read(y), 1e-5f,
                    $"resampling {planes} planes {h}x{w} to {oh}x{ow} (antialias {antialias})");

                // Bytes: Pillow's 8-bit passes with weights of 22 fractional bits, a table per channel.
                var bytes = Enumerable.Range(0, planes * h * w).Select(_ => (byte)c.Random.Next(256)).ToArray();
                var packed = new float[(bytes.Length + 3) / 4];
                for (int i = 0; i < bytes.Length; i++)
                {
                    packed[i / 4] = BitConverter.UInt32BitsToSingle(BitConverter.SingleToUInt32Bits(packed[i / 4]) | (uint)bytes[i] << (8 * (i % 4)));
                }

                var fixedPoint = coefficients.ToArray();
                Fixed(fixedPoint, 0, ow, xTaps);
                Fixed(fixedPoint, xTaps > 0 ? ow * (2 + xTaps) : 0, oh, yTaps);
                var table = c.Values(256 * channels);
                var yb = c.Zeros(planes * oh * ow);
                b.ResizeNormalize(c.Storage(packed), c.Storage(fixedPoint), c.Storage(table), yb, planes, channels, h, w, oh, ow, xTaps, yTaps, bytes: true);
                var expanded = bytes.Select(v => (float)v).ToArray();
                c.ExpectClose(ResampleReference(expanded, fixedPoint, table, planes, channels, h, w, oh, ow, xTaps, yTaps, bytes: true), Read(yb), 0f,
                    $"8-bit resampling {planes} planes {h}x{w} to {oh}x{ow} (antialias {antialias})");
            }
        }

        // The float weights of `count` outputs at `start` as integers with 22 fractional bits, rounded away from zero (Pillow's).
        static void Fixed(float[] coefficients, int start, int count, int taps)
        {
            for (int o = 0; o < count && taps > 0; o++)
            {
                for (int t = 0; t < taps; t++)
                {
                    int at = start + o * (2 + taps) + 2 + t;
                    double weight = coefficients[at] * (double)(1 << 22);
                    coefficients[at] = BitConverter.Int32BitsToSingle((int)(weight < 0 ? weight - 0.5 : weight + 0.5));
                }
            }
        }
    }

    // ------------------------------------------------------------------ plain-loop references

    private static float Activate(float v, ConvActivation activation) => activation switch
    {
        ConvActivation.Relu => MathF.Max(v, 0f),
        ConvActivation.Sigmoid => 1f / (1f + MathF.Exp(-v)),
        ConvActivation.Tanh => MathF.Tanh(v),
        ConvActivation.Gelu => 0.5f * v * (1f + MathF.Tanh(0.7978845608f * (v + 0.044715f * v * v * v))),
        ConvActivation.Silu => v / (1f + MathF.Exp(-v)),
        _ => v,
    };

    // Calls body(n, f, oh, ow, c, kh, kw, input index) for every term of the convolution with its input inside the image.
    private static void ConvolutionTerms(in ConvGeometry g, int filters, int groups, Action<int, int, int, int, int, int, int, int> body)
    {
        int perGroup = g.C / groups, fPerGroup = filters / groups;
        for (int n = 0; n < g.N; n++)
        {
            for (int f = 0; f < filters; f++)
            {
                int q = f / fPerGroup;
                for (int oh = 0; oh < g.OH; oh++)
                {
                    for (int ow = 0; ow < g.OW; ow++)
                    {
                        for (int ci = 0; ci < perGroup; ci++)
                        {
                            for (int kh = 0; kh < g.KH; kh++)
                            {
                                int ih = oh * g.SH - g.PH + kh * g.DH;
                                for (int kw = 0; kw < g.KW; kw++)
                                {
                                    int iw = ow * g.SW - g.PW + kw * g.DW;
                                    if (ih >= 0 && ih < g.H && iw >= 0 && iw < g.W)
                                    {
                                        body(n, f, oh, ow, ci, kh, kw, ((n * g.C + q * perGroup + ci) * g.H + ih) * g.W + iw);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    private static float[] ConvolutionReference(float[] x, float[] w, float[]? bias, ConvGeometry g, int filters, int groups, ConvActivation activation)
    {
        int patch = g.PatchSize / groups;
        var sums = new double[g.N * filters * g.OH * g.OW];
        ConvolutionTerms(g, filters, groups, (n, f, oh, ow, ci, kh, kw, at) =>
            sums[((n * filters + f) * g.OH + oh) * g.OW + ow] += (double)w[f * patch + (ci * g.KH + kh) * g.KW + kw] * x[at]);
        var y = new float[sums.Length];
        for (int i = 0; i < y.Length; i++)
        {
            int f = i / (g.OH * g.OW) % filters;
            y[i] = Activate((float)(sums[i] + (bias?[f] ?? 0f)), activation);
        }

        return y;
    }

    private static float[] ConvolutionInputGradient(float[] dy, float[] w, float[] dx0, ConvGeometry g, int filters, int groups)
    {
        int patch = g.PatchSize / groups;
        var sums = dx0.Select(v => (double)v).ToArray();
        ConvolutionTerms(g, filters, groups, (n, f, oh, ow, ci, kh, kw, at) =>
            sums[at] += (double)w[f * patch + (ci * g.KH + kh) * g.KW + kw] * dy[((n * filters + f) * g.OH + oh) * g.OW + ow]);
        return [.. sums.Select(v => (float)v)];
    }

    private static float[] ConvolutionWeightGradient(float[] x, float[] dy, float[] dw0, ConvGeometry g, int filters, int groups)
    {
        int patch = g.PatchSize / groups;
        var sums = dw0.Select(v => (double)v).ToArray();
        ConvolutionTerms(g, filters, groups, (n, f, oh, ow, ci, kh, kw, at) =>
            sums[f * patch + (ci * g.KH + kh) * g.KW + kw] += (double)dy[((n * filters + f) * g.OH + oh) * g.OW + ow] * x[at]);
        return [.. sums.Select(v => (float)v)];
    }

    // Each window's divisor, its sum and the elements it covers, by output index.
    private static IEnumerable<(int Output, float Divisor, IEnumerable<int> Inputs)> PoolWindows(ConvGeometry g, bool countPad, int bottom, int right)
    {
        for (int nc = 0; nc < g.N * g.C; nc++)
        {
            for (int oh = 0; oh < g.OH; oh++)
            {
                for (int ow = 0; ow < g.OW; ow++)
                {
                    int r0 = oh * g.SH - g.PH, c0 = ow * g.SW - g.PW;
                    var inputs = new List<int>();
                    for (int r = Math.Max(r0, 0); r < Math.Min(r0 + g.KH, g.H); r++)
                    {
                        for (int col = Math.Max(c0, 0); col < Math.Min(c0 + g.KW, g.W); col++)
                        {
                            inputs.Add((nc * g.H + r) * g.W + col);
                        }
                    }

                    int counted = (Math.Min(r0 + g.KH, g.H + bottom) - r0) * (Math.Min(c0 + g.KW, g.W + right) - c0);
                    yield return ((nc * g.OH + oh) * g.OW + ow, countPad ? counted : Math.Max(inputs.Count, 1), inputs);
                }
            }
        }
    }

    private static float[] AvgPoolReference(float[] x, ConvGeometry g, bool countPad, int bottom, int right)
    {
        var y = new float[g.N * g.C * g.OH * g.OW];
        foreach (var (o, divisor, inputs) in PoolWindows(g, countPad, bottom, right))
        {
            y[o] = (float)(inputs.Sum(i => (double)x[i]) / divisor);
        }

        return y;
    }

    private static float[] AvgPoolGradient(float[] dy, float[] dx0, ConvGeometry g, bool countPad, int bottom, int right)
    {
        var dx = dx0.Select(v => (double)v).ToArray();
        foreach (var (o, divisor, inputs) in PoolWindows(g, countPad, bottom, right))
        {
            foreach (int i in inputs)
            {
                dx[i] += dy[o] / divisor;
            }
        }

        return [.. dx.Select(v => (float)v)];
    }

    // The resampling with plain loops: x as floats (the bytes widened), coefficients and values as the operation reads them.
    private static float[] ResampleReference(float[] x, float[] coefficients, float[] values, int planes, int channels, int h, int w, int oh, int ow, int xTaps, int yTaps,
        bool bytes)
    {
        int across = xTaps > 0 ? ow : w, yBase = xTaps > 0 ? ow * (2 + xTaps) : 0;
        var y = new float[planes * oh * ow];
        var middle = new float[h * across];
        for (int p = 0; p < planes; p++)
        {
            for (int r = 0; r < h; r++)
            {
                for (int o = 0; o < across; o++)
                {
                    middle[r * across + o] = xTaps == 0 ? x[(p * h + r) * w + o] : Pass(i => x[(p * h + r) * w + i], 0 + o * (2 + xTaps), xTaps);
                }
            }

            for (int oy = 0; oy < oh; oy++)
            {
                for (int ox = 0; ox < ow; ox++)
                {
                    float v = yTaps == 0 ? middle[oy * across + ox] : Pass(i => middle[i * across + ox], yBase + oy * (2 + yTaps), yTaps);
                    int c = p % channels;
                    y[(p * oh + oy) * ow + ox] = bytes ? values[256 * c + (int)v] : v * values[2 * c] + values[2 * c + 1];
                }
            }
        }

        return y;

        float Pass(Func<int, float> read, int at, int taps)
        {
            int first = BitConverter.SingleToInt32Bits(coefficients[at]), count = BitConverter.SingleToInt32Bits(coefficients[at + 1]);
            if (bytes)
            {
                long sum = 1 << 21;
                for (int t = 0; t < count; t++)
                {
                    sum += (long)read(first + t) * BitConverter.SingleToInt32Bits(coefficients[at + 2 + t]);
                }

                return sum >= 1 << 30 ? 255 : sum <= 0 ? 0 : sum >> 22;
            }

            float total = 0f;
            for (int t = 0; t < count; t++)
            {
                total += read(first + t) * coefficients[at + 2 + t];
            }

            return total;
        }
    }
}
