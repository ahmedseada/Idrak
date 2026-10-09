// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

namespace Idrak.Vision;

// Moving pixels for the augmentations, in continuous coordinates where pixel (i, j) covers [i, i + 1) x [j, j + 1), the
// coordinates boxes are in, so a box moved by the same map stays on its pixels. Images are sampled bilinearly (by the
// area each output pixel covers when shrinking), masks and pixel classes by the nearest pixel.
internal static class ImageWarp
{
    // The source taps of each output position along one axis: the region [start, start + length) of an axis of `size`
    // pixels resized to `output` pixels. Index -1 stands for the fill value (outside the image).
    private static (int Index, float Weight)[][] Taps(int size, double start, double length, int output)
    {
        var taps = new (int, float)[output][];
        double step = length / output;
        for (int i = 0; i < output; i++)
        {
            if (step <= 1)
            {
                double centre = start + (i + 0.5) * step - 0.5;
                if (centre < -0.5 || centre > size - 0.5)
                {
                    taps[i] = [(-1, 1f)];
                    continue;
                }

                int i0 = (int)Math.Floor(centre);
                float f = (float)(centre - i0);
                taps[i] = [(Math.Clamp(i0, 0, size - 1), 1 - f), (Math.Clamp(i0 + 1, 0, size - 1), f)];
                continue;
            }

            double from = start + i * step, to = from + step;
            var covered = new List<(int, float)>();
            for (int j = (int)Math.Floor(from); j < Math.Ceiling(to); j++)
            {
                double overlap = Math.Min(to, j + 1) - Math.Max(from, j);
                if (overlap > 0)
                {
                    covered.Add(((uint)j < (uint)size ? j : -1, (float)(overlap / step)));
                }
            }

            taps[i] = [.. covered];
        }

        return taps;
    }

    // The region [x0, x0 + width) x [y0, y0 + height) of `image` (it may reach outside, which takes `fill`) resized to
    // outWidth x outHeight with `channels` channels (grey repeated, colour averaged), separably: the source rows the output
    // reads along x first, then each output row as a weighted sum of those rows (contiguous runs, no per-pixel branches).
    public static float[] Resample(ImageData image, double x0, double y0, double width, double height, int outWidth, int outHeight, int channels, float fill)
    {
        int w = image.Width, h = image.Height, c = image.Channels, plane = w * h;
        var (columnStart, columnIndex, columnWeight) = Flatten(Taps(w, x0, width, outWidth));
        var (rowStart, rowIndex, rowWeight) = Flatten(Taps(h, y0, height, outHeight));
        var used = new bool[h];                                     // only the source rows the output reads
        foreach (int index in rowIndex)
        {
            if (index >= 0)
            {
                used[index] = true;
            }
        }

        var result = GC.AllocateUninitializedArray<float>(channels * outWidth * outHeight);
        var line = new float[h * outWidth];
        float[]? grey = null;
        for (int oc = 0; oc < channels; oc++)
        {
            // The source plane this output channel reads: its own, the grey one, or the mean of the colour ones.
            ReadOnlySpan<float> source;
            if (c == channels || c == 1)
            {
                source = image.Pixels.AsSpan((c == 1 ? 0 : oc) * plane, plane);
            }
            else
            {
                if (grey is null)
                {
                    grey = new float[plane];
                    for (int k = 0; k < c; k++)
                    {
                        var channel = image.Pixels.AsSpan(k * plane, plane);
                        for (int i = 0; i < plane; i++)
                        {
                            grey[i] += channel[i] / c;
                        }
                    }
                }

                source = grey;
            }

            for (int y = 0; y < h; y++)
            {
                if (!used[y])
                {
                    continue;
                }

                var row = source.Slice(y * w, w);
                var output = line.AsSpan(y * outWidth, outWidth);
                for (int x = 0; x < outWidth; x++)
                {
                    float sum = 0;
                    for (int t = columnStart[x]; t < columnStart[x + 1]; t++)
                    {
                        int index = columnIndex[t];
                        sum += columnWeight[t] * (index < 0 ? fill : row[index]);
                    }

                    output[x] = sum;
                }
            }

            for (int y = 0; y < outHeight; y++)
            {
                var output = result.AsSpan((oc * outHeight + y) * outWidth, outWidth);
                output.Clear();
                for (int t = rowStart[y]; t < rowStart[y + 1]; t++)
                {
                    float weight = rowWeight[t];
                    int index = rowIndex[t];
                    if (index < 0)
                    {
                        for (int x = 0; x < outWidth; x++)
                        {
                            output[x] += weight * fill;
                        }

                        continue;
                    }

                    var from = line.AsSpan(index * outWidth, outWidth);
                    for (int x = 0; x < outWidth; x++)
                    {
                        output[x] += weight * from[x];
                    }
                }
            }
        }

        return result;
    }

    // Taps as flat arrays: output i reads index[start[i] .. start[i + 1]) with those weights.
    private static (int[] Start, int[] Index, float[] Weight) Flatten((int Index, float Weight)[][] taps)
    {
        var start = new int[taps.Length + 1];
        for (int i = 0; i < taps.Length; i++)
        {
            start[i + 1] = start[i] + taps[i].Length;
        }

        var index = new int[start[^1]];
        var weight = new float[start[^1]];
        for (int i = 0, k = 0; i < taps.Length; i++)
        {
            foreach (var (at, w) in taps[i])
            {
                (index[k], weight[k]) = (at, w);
                k++;
            }
        }

        return (start, index, weight);
    }

    // A mask or class map (w x h) over the same region, each output pixel the source pixel under its centre.
    public static T[] ResampleNearest<T>(T[] plane, int w, int h, double x0, double y0, double width, double height, int outWidth, int outHeight, T fill)
    {
        var result = new T[outWidth * outHeight];
        var xs = new int[outWidth];
        for (int x = 0; x < outWidth; x++)
        {
            double s = Math.Floor(x0 + (x + 0.5) * width / outWidth);
            xs[x] = s >= 0 && s < w ? (int)s : -1;
        }

        for (int y = 0; y < outHeight; y++)
        {
            double s = Math.Floor(y0 + (y + 0.5) * height / outHeight);
            int sy = s >= 0 && s < h ? (int)s : -1;
            for (int x = 0; x < outWidth; x++)
            {
                result[y * outWidth + x] = sy < 0 || xs[x] < 0 ? fill : plane[sy * w + xs[x]];
            }
        }

        return result;
    }

    // An affine map [a, b, c, d, e, f]: x' = a·x + b·y + c, y' = d·x + e·y + f.
    public static double[] Invert(double[] m)
    {
        double det = m[0] * m[4] - m[1] * m[3];
        if (Math.Abs(det) < 1e-12)
        {
            throw new ArgumentException("The affine map flattens the image (its determinant is 0).");
        }

        double a = m[4] / det, b = -m[1] / det, d = -m[3] / det, e = m[0] / det;
        return [a, b, -(a * m[2] + b * m[5]), d, e, -(d * m[2] + e * m[5])];
    }

    public static (double X, double Y) Apply(double[] m, double x, double y) => (m[0] * x + m[1] * y + m[2], m[3] * x + m[4] * y + m[5]);

    // The image under the affine map `forward` (source to output), sampled bilinearly at each output pixel's centre mapped
    // back: the source point found once per pixel (stepping along the row), then read for every channel.
    public static float[] Affine(ImageData image, double[] forward, int outWidth, int outHeight, float fill)
    {
        var inverse = Invert(forward);
        int w = image.Width, h = image.Height, c = image.Channels, plane = w * h, outPlane = outWidth * outHeight;
        var pixels = image.Pixels;
        var result = GC.AllocateUninitializedArray<float>(c * outPlane);
        for (int y = 0; y < outHeight; y++)
        {
            // The centre of pixel (0, y) mapped back, less half a pixel (sample positions are pixel indices).
            double sx = inverse[0] * 0.5 + inverse[1] * (y + 0.5) + inverse[2] - 0.5, sy = inverse[3] * 0.5 + inverse[4] * (y + 0.5) + inverse[5] - 0.5;
            for (int x = 0; x < outWidth; x++, sx += inverse[0], sy += inverse[3])
            {
                int ix = (int)Math.Floor(sx), iy = (int)Math.Floor(sy), at = y * outWidth + x;
                float fx = (float)(sx - ix), fy = (float)(sy - iy);
                bool left = (uint)ix < (uint)w, right = (uint)(ix + 1) < (uint)w, top = (uint)iy < (uint)h, bottom = (uint)(iy + 1) < (uint)h;
                if (!(left || right) || !(top || bottom))
                {
                    for (int k = 0; k < c; k++)
                    {
                        result[k * outPlane + at] = fill;
                    }

                    continue;
                }

                int i00 = iy * w + ix;
                float w00 = (1 - fx) * (1 - fy), w01 = fx * (1 - fy), w10 = (1 - fx) * fy, w11 = fx * fy;
                for (int k = 0, offset = 0; k < c; k++, offset += plane)
                {
                    float v00 = top && left ? pixels[offset + i00] : fill, v01 = top && right ? pixels[offset + i00 + 1] : fill;
                    float v10 = bottom && left ? pixels[offset + i00 + w] : fill, v11 = bottom && right ? pixels[offset + i00 + w + 1] : fill;
                    result[k * outPlane + at] = v00 * w00 + v01 * w01 + v10 * w10 + v11 * w11;
                }
            }
        }

        return result;
    }

    // A mask or class map under the affine map, each output pixel the source pixel under its centre mapped back.
    public static T[] AffineNearest<T>(T[] plane, int w, int h, double[] forward, int outWidth, int outHeight, T fill)
    {
        var inverse = Invert(forward);
        var result = new T[outWidth * outHeight];
        for (int y = 0; y < outHeight; y++)
        {
            for (int x = 0; x < outWidth; x++)
            {
                var (sx, sy) = Apply(inverse, x + 0.5, y + 0.5);
                double fx = Math.Floor(sx), fy = Math.Floor(sy);
                result[y * outWidth + x] = fx >= 0 && fx < w && fy >= 0 && fy < h ? plane[(int)fy * w + (int)fx] : fill;
            }
        }

        return result;
    }

    // The smallest box holding the box's four corners under the map.
    public static BoundingBox Transform(BoundingBox box, double[] forward)
    {
        double x1 = double.MaxValue, y1 = double.MaxValue, x2 = double.MinValue, y2 = double.MinValue;
        foreach (var (cx, cy) in new[] { (box.X, box.Y), (box.Right, box.Y), (box.X, box.Bottom), (box.Right, box.Bottom) })
        {
            var (x, y) = Apply(forward, cx, cy);
            (x1, x2, y1, y2) = (Math.Min(x1, x), Math.Max(x2, x), Math.Min(y1, y), Math.Max(y2, y));
        }

        return BoundingBox.FromCorners((float)x1, (float)y1, (float)x2, (float)y2);
    }
}
