// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Runtime.InteropServices;

namespace Idrak.Abstraction.Devices.Cpu;

// Resampling planes of [height, width]: interpolation (nearest and bilinear, PyTorch's F.interpolate) and adaptive pooling
// (PyTorch's AdaptiveAvgPool2d / AdaptiveMaxPool2d windows). Rows and columns are read through small tables computed once
// per call (an index and a weight per output row and column), so the inner loops only gather and blend. The forward passes
// run in parallel over the output rows of every plane; the backward passes over the planes, each writing only its own
// part of dx.
internal sealed partial class CpuBackend
{
    public override void Interpolate2dKernel(Storage x, Storage y, int planes, int height, int width, int outHeight, int outWidth, InterpolationMode mode,
        bool alignCorners, float scaleHeight, float scaleWidth)
    {
        float[] xv = D(x), yv = D(y);
        var rows = Axis(mode, alignCorners, scaleHeight, height, outHeight);
        var columns = Axis(mode, alignCorners, scaleWidth, width, outWidth);
        For(planes * outHeight, (long)planes * outHeight * outWidth * (mode == InterpolationMode.Nearest ? 1 : 4), (start, end) =>
        {
            for (int pr = start; pr < end; pr++)
            {
                int p = pr / outHeight, o = pr % outHeight;
                var target = yv.AsSpan(pr * outWidth, outWidth);
                var row0 = xv.AsSpan((p * height + rows.First[o]) * width, width);
                if (mode == InterpolationMode.Nearest)
                {
                    for (int j = 0; j < outWidth; j++)
                    {
                        target[j] = row0[columns.First[j]];
                    }

                    continue;
                }

                var row1 = xv.AsSpan((p * height + rows.Second[o]) * width, width);
                float h0 = rows.FirstWeight[o], h1 = rows.SecondWeight[o];
                for (int j = 0; j < outWidth; j++)
                {
                    int c0 = columns.First[j], c1 = columns.Second[j];
                    float w0 = columns.FirstWeight[j], w1 = columns.SecondWeight[j];
                    target[j] = h0 * (w0 * row0[c0] + w1 * row0[c1]) + h1 * (w0 * row1[c0] + w1 * row1[c1]);
                }
            }
        });
    }

    public override void Interpolate2dBackwardKernel(Storage dy, Storage dx, int planes, int height, int width, int outHeight, int outWidth, InterpolationMode mode,
        bool alignCorners, float scaleHeight, float scaleWidth)
    {
        float[] gv = D(dy), dv = D(dx);
        var rows = Axis(mode, alignCorners, scaleHeight, height, outHeight);
        var columns = Axis(mode, alignCorners, scaleWidth, width, outWidth);
        For(planes, (long)planes * outHeight * outWidth * (mode == InterpolationMode.Nearest ? 1 : 4), (start, end) =>
        {
            for (int p = start; p < end; p++)
            {
                for (int o = 0; o < outHeight; o++)
                {
                    var grad = gv.AsSpan((p * outHeight + o) * outWidth, outWidth);
                    var row0 = dv.AsSpan((p * height + rows.First[o]) * width, width);
                    if (mode == InterpolationMode.Nearest)
                    {
                        for (int j = 0; j < outWidth; j++)
                        {
                            row0[columns.First[j]] += grad[j];
                        }

                        continue;
                    }

                    var row1 = dv.AsSpan((p * height + rows.Second[o]) * width, width);
                    float h0 = rows.FirstWeight[o], h1 = rows.SecondWeight[o];
                    for (int j = 0; j < outWidth; j++)
                    {
                        int c0 = columns.First[j], c1 = columns.Second[j];
                        float w0 = columns.FirstWeight[j], w1 = columns.SecondWeight[j], g = grad[j];
                        row0[c0] += h0 * w0 * g;
                        row0[c1] += h0 * w1 * g;
                        row1[c0] += h1 * w0 * g;
                        row1[c1] += h1 * w1 * g;
                    }
                }
            }
        });
    }

    // The input positions (and their weights) each output position of one axis reads, computed as PyTorch computes them in
    // float: nearest reads min(floor(o · scale), size - 1), with output = input and output = 2 · input as exact special
    // cases; bilinear reads floor(s) and the next position (the last reads itself twice) with weights 1 - λ and λ, where
    // s = o · scale (align corners) or max((o + 0.5) · scale - 0.5, 0) and λ = s - floor(s).
    private readonly record struct AxisTable(int[] First, int[] Second, float[] FirstWeight, float[] SecondWeight);

    private static AxisTable Axis(InterpolationMode mode, bool alignCorners, float scale, int size, int outSize)
    {
        var first = new int[outSize];
        if (mode == InterpolationMode.Nearest)
        {
            for (int o = 0; o < outSize; o++)
            {
                first[o] = outSize == size ? o : outSize == 2 * size ? o >> 1 : Math.Min((int)MathF.Floor(o * scale), size - 1);
            }

            return new(first, first, [], []);
        }

        var second = new int[outSize];
        float[] w0 = new float[outSize], w1 = new float[outSize];
        for (int o = 0; o < outSize; o++)
        {
            float source = alignCorners ? scale * o : MathF.Max(scale * (o + 0.5f) - 0.5f, 0f);
            int index = Math.Min((int)MathF.Floor(source), size - 1);
            float lambda = Math.Clamp(source - index, 0f, 1f);
            first[o] = index;
            second[o] = index + (index < size - 1 ? 1 : 0);
            w0[o] = 1f - lambda;
            w1[o] = lambda;
        }

        return new(first, second, w0, w1);
    }

    // PyTorch's adaptive windows: output o covers input [floor(o · size / outSize), ceil((o + 1) · size / outSize)).
    private static int WindowStart(int o, int size, int outSize) => (int)((long)o * size / outSize);

    private static int WindowEnd(int o, int size, int outSize) => (int)(((long)(o + 1) * size + outSize - 1) / outSize);

    public override void AdaptiveAvgPoolKernel(Storage x, Storage y, int planes, int height, int width, int outHeight, int outWidth)
    {
        float[] xv = D(x), yv = D(y);
        For(planes * outHeight, (long)planes * height * width + (long)planes * outHeight * outWidth, (start, end) =>
        {
            for (int pr = start; pr < end; pr++)
            {
                int p = pr / outHeight, o = pr % outHeight;
                int r0 = WindowStart(o, height, outHeight), r1 = WindowEnd(o, height, outHeight);
                for (int j = 0; j < outWidth; j++)
                {
                    int c0 = WindowStart(j, width, outWidth), c1 = WindowEnd(j, width, outWidth);
                    float sum = 0f;
                    for (int r = r0; r < r1; r++)
                    {
                        var row = xv.AsSpan((p * height + r) * width + c0, c1 - c0);
                        foreach (float v in row)
                        {
                            sum += v;
                        }
                    }

                    yv[pr * outWidth + j] = sum / (r1 - r0) / (c1 - c0);
                }
            }
        });
    }

    public override void AdaptiveAvgPoolBackwardKernel(Storage dy, Storage dx, int planes, int height, int width, int outHeight, int outWidth)
    {
        float[] gv = D(dy), dv = D(dx);
        For(planes, (long)planes * height * width + (long)planes * outHeight * outWidth, (start, end) =>
        {
            for (int p = start; p < end; p++)
            {
                for (int o = 0; o < outHeight; o++)
                {
                    int r0 = WindowStart(o, height, outHeight), r1 = WindowEnd(o, height, outHeight);
                    for (int j = 0; j < outWidth; j++)
                    {
                        int c0 = WindowStart(j, width, outWidth), c1 = WindowEnd(j, width, outWidth);
                        float share = gv[(p * outHeight + o) * outWidth + j] / (r1 - r0) / (c1 - c0);
                        for (int r = r0; r < r1; r++)
                        {
                            var row = dv.AsSpan((p * height + r) * width + c0, c1 - c0);
                            for (int c = 0; c < row.Length; c++)
                            {
                                row[c] += share;
                            }
                        }
                    }
                }
            }
        });
    }

    public override void AdaptiveMaxPoolKernel(Storage x, Storage y, Storage argmax, int planes, int height, int width, int outHeight, int outWidth)
    {
        float[] xv = D(x), yv = D(y), av = D(argmax);
        For(planes * outHeight, (long)planes * height * width + (long)planes * outHeight * outWidth, (start, end) =>
        {
            var index = MemoryMarshal.Cast<float, int>(av.AsSpan());
            for (int pr = start; pr < end; pr++)
            {
                int p = pr / outHeight, o = pr % outHeight, plane = p * height * width;
                int r0 = WindowStart(o, height, outHeight), r1 = WindowEnd(o, height, outHeight);
                for (int j = 0; j < outWidth; j++)
                {
                    int c0 = WindowStart(j, width, outWidth), c1 = WindowEnd(j, width, outWidth);
                    float best = float.NegativeInfinity;
                    int bestIndex = r0 * width + c0;
                    for (int r = r0; r < r1; r++)
                    {
                        for (int c = c0; c < c1; c++)
                        {
                            float v = xv[plane + r * width + c];
                            if (v > best || float.IsNaN(v))                // as PyTorch: a NaN wins
                            {
                                best = v;
                                bestIndex = r * width + c;
                            }
                        }
                    }

                    yv[pr * outWidth + j] = best;
                    index[pr * outWidth + j] = plane + bestIndex;
                }
            }
        });
    }
}
