// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

namespace Idrak.Imaging;

/// <summary>
/// Perspective and resampling, shared: the homography that takes four points to four others, and images warped through a
/// map from each output pixel back into the input (bilinear, rows walked in order and spread over the cores). The scan
/// transforms bend and turn pages with them, and the document drawer places images by their matrix with the same map.
/// </summary>
internal static class PerspectiveWarp
{
    /// <summary>The homography taking the four points `from` to `to` (x, y pairs), as 9 numbers row by row.</summary>
    internal static double[] Homography(double[] from, double[] to)
    {
        var m = new double[8, 9];
        for (int k = 0; k < 4; k++)
        {
            double x = from[2 * k], y = from[2 * k + 1], u = to[2 * k], v = to[2 * k + 1];
            double[] r1 = [x, y, 1, 0, 0, 0, -u * x, -u * y, u];
            double[] r2 = [0, 0, 0, x, y, 1, -v * x, -v * y, v];
            for (int j = 0; j < 9; j++)
            {
                m[2 * k, j] = r1[j];
                m[2 * k + 1, j] = r2[j];
            }
        }

        // Gaussian elimination with partial pivoting.
        for (int col = 0; col < 8; col++)
        {
            int pivot = col;
            for (int r = col + 1; r < 8; r++)
            {
                if (Math.Abs(m[r, col]) > Math.Abs(m[pivot, col]))
                {
                    pivot = r;
                }
            }

            for (int j = 0; j < 9; j++)
            {
                (m[col, j], m[pivot, j]) = (m[pivot, j], m[col, j]);
            }

            for (int r = 0; r < 8; r++)
            {
                if (r == col)
                {
                    continue;
                }

                double f = m[r, col] / m[col, col];
                for (int j = col; j < 9; j++)
                {
                    m[r, j] -= f * m[col, j];
                }
            }
        }

        var h = new double[9];
        for (int i = 0; i < 8; i++)
        {
            h[i] = m[i, 8] / m[i, i];
        }

        h[8] = 1;
        return h;
    }

    /// <summary>Each output pixel sampled (bilinear) where <paramref name="map"/> takes it in the input; outside the input, the fill.</summary>
    internal static ImageData Warp(ImageData image, double[] map, int? fill) => WarpRows(image, (y, sxs, sys) =>
    {
        for (int x = 0; x < sxs.Length; x++)
        {
            double d = map[6] * x + map[7] * y + map[8];
            sxs[x] = (map[0] * x + map[1] * y + map[2]) / d;
            sys[x] = (map[3] * x + map[4] * y + map[5]) / d;
        }
    }, fill);

    /// <summary>
    /// Each output pixel sampled (bilinear) where <paramref name="map"/> says, row by row (it fills a row's source x and
    /// y); outside the image, the fill (the paper's level unless given).
    /// </summary>
    internal static ImageData WarpRows(ImageData image, Action<int, double[], double[]> map, int? fill)
    {
        var planes = PillowImageOps.Bytes(image);
        int w = image.Width, h = image.Height, channels = planes.Length;
        var fills = ScanImageOps.Fills(planes, w * h, fill);
        var output = new byte[channels][];
        for (int c = 0; c < channels; c++)
        {
            output[c] = GC.AllocateUninitializedArray<byte>(w * h);
        }

        HostParallel.For(h, ScanImageOps.Grain(w), (first, last) =>
        {
            // Each row's source places once (the corner index, or -1 where a neighbour is off the image), then each
            // plane along the row.
            var index = new int[w];
            var fxs = new float[w];
            var fys = new float[w];
            var xs = new int[w];
            var ys = new int[w];
            var sxs = new double[w];
            var sys = new double[w];
            for (int y = first; y < last; y++)
            {
                map(y, sxs, sys);
                for (int x = 0; x < w; x++)
                {
                    double sx = sxs[x], sy = sys[x];
                    int x0 = (int)Math.Floor(sx), y0 = (int)Math.Floor(sy);
                    fxs[x] = (float)(sx - x0);
                    fys[x] = (float)(sy - y0);
                    xs[x] = x0;
                    ys[x] = y0;
                    index[x] = x0 >= 0 && y0 >= 0 && x0 + 1 < w && y0 + 1 < h ? y0 * w + x0 : -1;
                }

                for (int c = 0; c < channels; c++)
                {
                    var p = planes[c];
                    var row = output[c].AsSpan(y * w, w);
                    byte f = fills[c];
                    for (int x = 0; x < w; x++)
                    {
                        int i = index[x];
                        float v00, v10, v01, v11;
                        if (i >= 0)
                        {
                            (v00, v10, v01, v11) = (p[i], p[i + 1], p[i + w], p[i + w + 1]);
                        }
                        else
                        {
                            int x0 = xs[x], y0 = ys[x];
                            v00 = At(p, w, h, x0, y0, f);
                            v10 = At(p, w, h, x0 + 1, y0, f);
                            v01 = At(p, w, h, x0, y0 + 1, f);
                            v11 = At(p, w, h, x0 + 1, y0 + 1, f);
                        }

                        float fx = fxs[x];
                        float top = v00 + (v10 - v00) * fx, bottom = v01 + (v11 - v01) * fx;
                        row[x] = (byte)(top + (bottom - top) * fys[x] + 0.5f);
                    }
                }
            }
        });

        return PillowImageOps.FromBytes(output, h, w);
    }

    internal static float At(byte[] plane, int w, int h, int x, int y, byte fill) =>
        (uint)x < (uint)w && (uint)y < (uint)h ? plane[y * w + x] : fill;
}
