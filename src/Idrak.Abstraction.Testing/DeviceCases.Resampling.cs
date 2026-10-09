// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Idrak.Abstraction.Devices;

namespace Idrak.Abstraction.Testing;

// The cases of image resampling: interpolation (nearest, bilinear) and adaptive pooling, against plain loops.
public static partial class DeviceCases
{
    private static void Resampling(DeviceCaseContext c)
    {
        var b = c.Backend;

        // Up, down, odd ratios and one row or column; scales as F.interpolate computes them.
        (int H, int W, int OH, int OW)[] sizes =
        [
            (1, 1, 1, 1), (2, 3, 4, 6), (5, 4, 7, 9), (7, 9, 3, 4), (1, 5, 3, 2), (4, 1, 4, 3),
            (c.Size(1, 12), c.Size(1, 12), c.Size(1, 20), c.Size(1, 20)),
        ];
        foreach (var (h, w, oh, ow) in sizes)
        {
            int planes = c.Size(1, 3);
            var x = c.Values(planes * h * w, 2f);
            var dy = c.Values(planes * oh * ow);
            Storage xs = c.Storage(x), dys = c.Storage(dy);
            foreach (var (mode, align) in new[] { (InterpolationMode.Nearest, false), (InterpolationMode.Bilinear, false), (InterpolationMode.Bilinear, true) })
            {
                float sh = Scale(h, oh, align), sw = Scale(w, ow, align);
                var y = c.Zeros(planes * oh * ow);
                b.Interpolate2d(xs, y, planes, h, w, oh, ow, mode, align, sh, sw);
                var expected = new float[planes * oh * ow];
                var expectedGrad = new float[planes * h * w];
                for (int p = 0; p < planes; p++)
                {
                    for (int i = 0; i < oh; i++)
                    {
                        for (int j = 0; j < ow; j++)
                        {
                            int o = (p * oh + i) * ow + j;
                            foreach (var (r, rw) in Taps(mode, align, sh, i, h, oh))
                            {
                                foreach (var (col, cw) in Taps(mode, align, sw, j, w, ow))
                                {
                                    expected[o] += rw * cw * x[(p * h + r) * w + col];
                                    expectedGrad[(p * h + r) * w + col] += rw * cw * dy[o];
                                }
                            }
                        }
                    }
                }

                c.ExpectClose(expected, Read(y), 1e-5f, $"interpolation {mode} (align corners {align}) {h}x{w} to {oh}x{ow}");
                var dx = c.Zeros(planes * h * w);
                b.Interpolate2dBackward(dys, dx, planes, h, w, oh, ow, mode, align, sh, sw);
                c.ExpectClose(expectedGrad, Read(dx), 1e-5f, $"interpolation gradient {mode} (align corners {align}) {h}x{w} to {oh}x{ow}");
            }

            // A scale factor whose product with the size is not whole reads by 1 / factor, not by the size ratio.
            b.Interpolate2d(xs, c.Zeros(planes * oh * ow), planes, h, w, oh, ow, InterpolationMode.Nearest, false, 1f / 1.7f, 1f / 2.3f);

            // Adaptive pooling: PyTorch's windows [floor(o·H/OH), ceil((o+1)·H/OH)).
            var avg = c.Zeros(planes * oh * ow);
            var max = c.Zeros(planes * oh * ow);
            var argmax = c.Zeros(planes * oh * ow);
            b.AdaptiveAvgPool(xs, avg, planes, h, w, oh, ow);
            b.AdaptiveMaxPool(xs, max, argmax, planes, h, w, oh, ow);
            var expectedAvg = new float[planes * oh * ow];
            var expectedMax = new float[planes * oh * ow];
            var expectedAvgGrad = new float[planes * h * w];
            var expectedMaxGrad = new float[planes * h * w];
            for (int p = 0; p < planes; p++)
            {
                for (int i = 0; i < oh; i++)
                {
                    int r0 = i * h / oh, r1 = ((i + 1) * h + oh - 1) / oh;
                    for (int j = 0; j < ow; j++)
                    {
                        int c0 = j * w / ow, c1 = ((j + 1) * w + ow - 1) / ow, o = (p * oh + i) * ow + j, best = -1;
                        double sum = 0;
                        for (int r = r0; r < r1; r++)
                        {
                            for (int col = c0; col < c1; col++)
                            {
                                int at = (p * h + r) * w + col;
                                sum += x[at];
                                best = best < 0 || x[at] > x[best] ? at : best;
                            }
                        }

                        int count = (r1 - r0) * (c1 - c0);
                        expectedAvg[o] = (float)(sum / count);
                        expectedMax[o] = x[best];
                        expectedMaxGrad[best] += dy[o];
                        for (int r = r0; r < r1; r++)
                        {
                            for (int col = c0; col < c1; col++)
                            {
                                expectedAvgGrad[(p * h + r) * w + col] += dy[o] / count;
                            }
                        }
                    }
                }
            }

            c.ExpectClose(expectedAvg, Read(avg), 1e-5f, $"adaptive average pooling {h}x{w} to {oh}x{ow}");
            c.ExpectClose(expectedMax, Read(max), 0f, $"adaptive max pooling {h}x{w} to {oh}x{ow}");
            var dAvg = c.Zeros(planes * h * w);
            b.AdaptiveAvgPoolBackward(dys, dAvg, planes, h, w, oh, ow);
            c.ExpectClose(expectedAvgGrad, Read(dAvg), 1e-5f, $"adaptive average pooling gradient {h}x{w} to {oh}x{ow}");
            var dMax = c.Zeros(planes * h * w);
            b.MaxPoolBackward(dys, argmax, dMax, planes * oh * ow);
            c.ExpectClose(expectedMaxGrad, Read(dMax), 1e-6f, $"adaptive max pooling gradient {h}x{w} to {oh}x{ow}");
            var dMaxGathered = c.Zeros(planes * h * w);
            b.AdaptiveMaxPoolBackward(dys, argmax, dMaxGathered, planes, h, w, oh, ow);
            c.ExpectClose(expectedMaxGrad, Read(dMaxGathered), 1e-6f, $"adaptive max pooling gradient by windows {h}x{w} to {oh}x{ow}");
        }
    }

    // F.interpolate's coordinate scale (input positions per output position), in float.
    private static float Scale(int input, int output, bool align) => align ? (output > 1 ? (input - 1) / (float)(output - 1) : 0f) : input / (float)output;

    // The input positions output position o reads along one axis and their weights.
    private static IEnumerable<(int Index, float Weight)> Taps(InterpolationMode mode, bool align, float scale, int o, int size, int outSize)
    {
        if (mode == InterpolationMode.Nearest)
        {
            yield return (outSize == size ? o : outSize == 2 * size ? o / 2 : Math.Min((int)MathF.Floor(o * scale), size - 1), 1f);
            yield break;
        }

        float source = align ? scale * o : Math.Max(scale * (o + 0.5f) - 0.5f, 0f);
        int first = Math.Min((int)MathF.Floor(source), size - 1);
        float lambda = Math.Clamp(source - first, 0f, 1f);
        yield return (first, 1f - lambda);
        yield return (Math.Min(first + 1, size - 1), lambda);
    }
}
