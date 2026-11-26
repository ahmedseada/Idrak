// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Globalization;

namespace Idrak.Imaging;

/// <summary>
/// The library's scan degradations as image transforms (<see cref="ImageTransforms"/>), for pages made to look scanned
/// (plan 18, phase 4): <c>rotate</c>, <c>perspective</c>, <c>noise</c>, <c>speckle</c>, <c>dropout</c>, <c>fade</c>, <c>tint</c>,
/// <c>stain</c>, <c>lowres</c> and <c>bleed</c>; with the Pillow ones (<c>blur</c>, <c>jpeg</c>, <c>binarize</c>, ...) they
/// make the scan profiles (<see cref="ScanProfiles"/>). The random ones take a <c>seed</c> option: the same seed, the
/// same pixels, on every machine and with any number of threads.
/// </summary>
internal static class ScanTransforms
{
    internal static readonly ImageTransformParameter SeedInput = new("seed", "Seed", ImageTransformInput.Integer) { Min = 0, Max = int.MaxValue, Step = 1, Optional = true };

    internal static readonly ImageTransformParameter FillInput = new("fill", "Fill, grey level (empty: the paper's)", ImageTransformInput.Integer) { Min = 0, Max = 255, Step = 1, Optional = true };

    internal static ImageTransformParameter Number(string label, double min, double max, double step, string help) =>
        new("", label, ImageTransformInput.Number) { Min = min, Max = max, Step = step, Help = help.Length == 0 ? null : help };

    internal static double Check(ImageTransformStep step, double min, double max, string? fill = null)
    {
        double v = step.Number();
        if (!(v >= min && v <= max))
        {
            throw new ArgumentException($"{step.Name} is from {min.ToString(CultureInfo.InvariantCulture)} to {max.ToString(CultureInfo.InvariantCulture)}, not {v.ToString(CultureInfo.InvariantCulture)}.");
        }

        if (fill is not null)
        {
            Fill(step);
        }

        Seed(step);
        return v;
    }

    internal static int Seed(ImageTransformStep step) => step.Option("seed") is not { } text ? 0
        : int.TryParse(text, NumberStyles.Integer, CultureInfo.InvariantCulture, out int seed) ? seed
        : throw new ArgumentException($"{step.Name}'s seed is a whole number, not '{text}'.");

    internal static int? Fill(ImageTransformStep step) => step.Option("fill") is not { } text ? null
        : int.TryParse(text, NumberStyles.Integer, CultureInfo.InvariantCulture, out int v) && v is >= 0 and <= 255 ? v
        : throw new ArgumentException($"{step.Name}'s fill is a grey level from 0 to 255, not '{text}'.");

    internal static double OptionNumber(ImageTransformStep step, string key, double otherwise, double min, double max) =>
        step.Option(key) is not { } text ? otherwise
        : double.TryParse(text, NumberStyles.Float, CultureInfo.InvariantCulture, out double v) && v >= min && v <= max ? v
        : throw new ArgumentException($"{step.Name}'s {key} is from {min.ToString(CultureInfo.InvariantCulture)} to {max.ToString(CultureInfo.InvariantCulture)}, not '{text}'.");

    private static double Size(ImageTransformStep step) => OptionNumber(step, "size", 1.5, 0.3, 20);

    private static double PatchSize(ImageTransformStep step) => OptionNumber(step, "size", 6, 1, 60);

    private static double Strength(ImageTransformStep step) => OptionNumber(step, "strength", 0.85, 0, 1);

    private static double Uneven(ImageTransformStep step) => OptionNumber(step, "uneven", 0, 0, 1);

    private static double Opacity(ImageTransformStep step) => OptionNumber(step, "opacity", 0.25, 0, 1);

    private static double BleedBlur(ImageTransformStep step) => OptionNumber(step, "spread", 2, 0, 50);

    private static int Count(ImageTransformStep step)
    {
        int n = step.Integer();
        Seed(step);
        return n is >= 0 and <= 50 ? n : throw new ArgumentException($"stain is 0 to 50 stains, not {n}.");
    }

    private static (float R, float G, float B) Color(ImageTransformStep step)
    {
        string text = (step.Option("color") ?? "f2e6c4").TrimStart('#');
        return text.Length == 6 && int.TryParse(text, NumberStyles.HexNumber, CultureInfo.InvariantCulture, out int rgb)
            ? (((rgb >> 16) & 255) / 255f, ((rgb >> 8) & 255) / 255f, (rgb & 255) / 255f)
            : throw new ArgumentException($"tint's color is RRGGBB in hexadecimal (f2e6c4), not '{text}'.");
    }
}

/// <summary>The scan degradations on 8-bit planes; each returns a new image and leaves its input alone.</summary>
internal static class ScanImageOps
{
    // Rows a worker takes at least: about 32 thousand pixels.
    internal static int Grain(int width) => Math.Max(1, (1 << 15) / Math.Max(1, width));

    /// <summary>Each plane's paper level: its 90th percentile (most of a page is paper), from a sample of the pixels.</summary>
    internal static byte[] PaperLevels(byte[][] planes, int size)
    {
        var paper = new byte[planes.Length];
        int stride = Math.Max(1, size / 65536);
        Span<int> histogram = stackalloc int[256];
        for (int c = 0; c < planes.Length; c++)
        {
            histogram.Clear();
            int n = 0;
            var plane = planes[c];
            for (int i = 0; i < size; i += stride)
            {
                histogram[plane[i]]++;
                n++;
            }

            int target = (int)(n * 0.9), seen = 0, v = 0;
            for (; v < 255; v++)
            {
                seen += histogram[v];
                if (seen > target)
                {
                    break;
                }
            }

            paper[c] = (byte)v;
        }

        return paper;
    }

    internal static byte[] Fills(byte[][] planes, int size, int? fill) =>
        fill is { } f ? [.. Enumerable.Repeat((byte)f, planes.Length)] : PaperLevels(planes, size);

    /// <summary>The image turned <paramref name="degrees"/> counterclockwise about its centre (bilinear, same size).</summary>
    public static ImageData Rotate(ImageData image, double degrees, int? fill)
    {
        if (degrees == 0)
        {
            return image;
        }

        double a = degrees * Math.PI / 180, cos = PortableMath.Cos(a), sin = PortableMath.Sin(a);
        double cx = (image.Width - 1) / 2.0, cy = (image.Height - 1) / 2.0;

        // From an output pixel back to where it was: turned the other way about the centre.
        double[] map =
        [
            cos, -sin, cx - cos * cx + sin * cy,
            sin, cos, cy - sin * cx - cos * cy,
            0, 0, 1,
        ];
        return PerspectiveWarp.Warp(image, map, fill);
    }

    /// <summary>The image with each corner moved up to <paramref name="strength"/> of its size, as a photographed page.</summary>
    public static ImageData Perspective(ImageData image, double strength, int seed, int? fill)
    {
        if (strength == 0)
        {
            return image;
        }

        int w = image.Width, h = image.Height;
        var random = new Random(seed);
        double[] corners = [0, 0, w - 1, 0, w - 1, h - 1, 0, h - 1];
        var moved = new double[8];
        for (int k = 0; k < 4; k++)
        {
            // Each corner moves inward or along its edges, so the page stays on the image.
            double sx = k is 0 or 3 ? 1 : -1, sy = k is 0 or 1 ? 1 : -1;
            moved[2 * k] = corners[2 * k] + sx * random.NextDouble() * strength * w;
            moved[2 * k + 1] = corners[2 * k + 1] + sy * random.NextDouble() * strength * h;
        }

        return PerspectiveWarp.Warp(image, PerspectiveWarp.Homography(moved, corners), fill);
    }

    // 65536 quantiles of the standard normal distribution (the midpoints of equal-probability slices), made once: a
    // pixel's grain is one of them, picked by its row's generator.
    private static readonly Lazy<float[]> NormalTable = new(() =>
    {
        var table = new float[1 << 16];
        for (int i = 0; i < table.Length; i++)
        {
            table[i] = (float)InverseNormal((i + 0.5) / table.Length);
        }

        return table;
    });

    // Acklam's rational approximation of the normal quantile (relative error under 1.2e-9).
    private static double InverseNormal(double p)
    {
        double[] a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02, 1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00];
        double[] b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02, 6.680131188771972e+01, -1.328068155288572e+01];
        double[] c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00, -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00];
        double[] d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00, 3.754408661907416e+00];
        const double Low = 0.02425;
        if (p < Low || p > 1 - Low)
        {
            double q = Math.Sqrt(-2 * PortableMath.Log(p < Low ? p : 1 - p));
            double x = (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1);
            return p < Low ? x : -x;
        }

        double r = p - 0.5, s = r * r;
        return (((((a[0] * s + a[1]) * s + a[2]) * s + a[3]) * s + a[4]) * s + a[5]) * r / (((((b[0] * s + b[1]) * s + b[2]) * s + b[3]) * s + b[4]) * s + 1);
    }

    // A row's own generator state (SplitMix64 of the seed and the row; never 0), so rows give the same numbers on any
    // number of threads, in every process (HashCode is seeded per process, so it is not used).
    private static ulong RowState(int seed, int row)
    {
        ulong z = ((ulong)(uint)seed << 32 | (uint)row) + 0x9E3779B97F4A7C15UL;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9UL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBUL;
        z ^= z >> 31;
        return z == 0 ? 1 : z;
    }

    /// <summary>Grain: each pixel moved by a normal amount of <paramref name="sigma"/> grey levels (the same in every channel).</summary>
    public static ImageData Noise(ImageData image, double sigma, int seed)
    {
        if (sigma == 0)
        {
            return image;
        }

        var planes = PillowImageOps.Bytes(image);
        int w = image.Width, h = image.Height;
        var table = NormalTable.Value;
        float scale = (float)sigma;
        HostParallel.For(h, Grain(w), (first, last) =>
        {
            var amounts = new float[w];
            for (int y = first; y < last; y++)
            {
                ulong state = RowState(seed, y);
                for (int x = 0; x < w; x++)
                {
                    // xorshift64*: the top 16 bits pick a quantile.
                    state ^= state >> 12;
                    state ^= state << 25;
                    state ^= state >> 27;
                    amounts[x] = table[(int)((state * 0x2545F4914F6CDD1DUL) >> 48)] * scale;
                }

                foreach (var plane in planes)
                {
                    var row = plane.AsSpan(y * w, w);
                    for (int x = 0; x < w; x++)
                    {
                        row[x] = (byte)Math.Clamp((int)MathF.Round(row[x] + amounts[x]), 0, 255);
                    }
                }
            }
        });

        return PillowImageOps.FromBytes(planes, h, w);
    }

    /// <summary>Specks: <paramref name="perMegapixel"/> soft dots a megapixel, most dark (dust, toner), some light (holes in the ink).</summary>
    public static ImageData Speckle(ImageData image, double perMegapixel, double size, int seed)
    {
        var planes = PillowImageOps.Bytes(image);
        int w = image.Width, h = image.Height;
        var paper = PaperLevels(planes, w * h);
        var random = new Random(seed);
        int count = (int)Math.Round(perMegapixel * w * h / 1e6);
        for (int k = 0; k < count; k++)
        {
            double cx = random.NextDouble() * w, cy = random.NextDouble() * h;
            double radius = 0.3 + random.NextDouble() * (size - 0.3);
            bool light = random.NextDouble() < 0.2;
            float strength = (float)(0.5 + 0.5 * random.NextDouble());
            int x0 = Math.Max(0, (int)(cx - radius - 1)), x1 = Math.Min(w - 1, (int)(cx + radius + 1));
            int y0 = Math.Max(0, (int)(cy - radius - 1)), y1 = Math.Min(h - 1, (int)(cy + radius + 1));
            for (int y = y0; y <= y1; y++)
            {
                for (int x = x0; x <= x1; x++)
                {
                    double dx = x + 0.5 - cx, dy = y + 0.5 - cy;
                    float cover = (float)Math.Clamp(radius + 0.5 - Math.Sqrt(dx * dx + dy * dy), 0, 1) * strength;
                    if (cover <= 0)
                    {
                        continue;
                    }

                    for (int c = 0; c < planes.Length; c++)
                    {
                        ref byte v = ref planes[c][y * w + x];
                        float target = light ? paper[c] : paper[c] * 0.08f;
                        v = (byte)(v + (target - v) * cover + 0.5f);
                    }
                }
            }
        }

        return PillowImageOps.FromBytes(planes, h, w);
    }

    // A smooth field in [0, 1] over the page: a coarse grid of random values, bilinear between them.
    internal static float[] SmoothGrid(int seed, out int cells)
    {
        cells = 6;
        var random = new Random(seed);
        var grid = new float[(cells + 1) * (cells + 1)];
        for (int i = 0; i < grid.Length; i++)
        {
            grid[i] = (float)random.NextDouble();
        }

        return grid;
    }

    internal static float Smooth(float[] grid, int cells, float u, float v)
    {
        float gx = u * cells, gy = v * cells;
        int ix = Math.Min(cells - 1, (int)gx), iy = Math.Min(cells - 1, (int)gy);
        float fx = gx - ix, fy = gy - iy;
        int n = cells + 1;
        float top = grid[iy * n + ix] + (grid[iy * n + ix + 1] - grid[iy * n + ix]) * fx;
        float bottom = grid[(iy + 1) * n + ix] + (grid[(iy + 1) * n + ix + 1] - grid[(iy + 1) * n + ix]) * fx;
        return top + (bottom - top) * fy;
    }

    /// <summary>
    /// Worn ink: soft elliptical patches over about <paramref name="share"/> of the page, each taking up to
    /// <paramref name="strength"/> of the ink under it toward the paper; the paper itself is left alone.
    /// </summary>
    public static ImageData Dropout(ImageData image, double share, double size, double strength, int seed)
    {
        if (share == 0 || strength == 0)
        {
            return image;
        }

        var planes = PillowImageOps.Bytes(image);
        int w = image.Width, h = image.Height;
        var paper = PaperLevels(planes, w * h);
        var random = new Random(seed);
        double meanArea = Math.PI * PortableMath.Pow(size * 0.7, 2) * 0.7;
        int count = (int)Math.Round(share * w * h / Math.Max(1, meanArea));
        for (int k = 0; k < count; k++)
        {
            double cx = random.NextDouble() * w, cy = random.NextDouble() * h;
            double rx = size * (0.4 + 0.6 * random.NextDouble()), ry = rx * (0.4 + 0.6 * random.NextDouble());
            double angle = random.NextDouble() * Math.PI, ca = PortableMath.Cos(angle), sa = PortableMath.Sin(angle);
            float take = (float)(strength * (0.5 + 0.5 * random.NextDouble()));
            int reach = (int)Math.Ceiling(Math.Max(rx, ry)) + 1;
            int x0 = Math.Max(0, (int)cx - reach), x1 = Math.Min(w - 1, (int)cx + reach);
            int y0 = Math.Max(0, (int)cy - reach), y1 = Math.Min(h - 1, (int)cy + reach);
            for (int y = y0; y <= y1; y++)
            {
                for (int x = x0; x <= x1; x++)
                {
                    double dx = x + 0.5 - cx, dy = y + 0.5 - cy;
                    double u = (dx * ca + dy * sa) / rx, v = (-dx * sa + dy * ca) / ry;
                    double d = Math.Sqrt(u * u + v * v);
                    float cover = (float)Math.Clamp((1 - d) * Math.Min(rx, ry) + 0.5, 0, 1) * take;
                    if (cover <= 0)
                    {
                        continue;
                    }

                    for (int c = 0; c < planes.Length; c++)
                    {
                        ref byte p = ref planes[c][y * w + x];
                        if (p < paper[c])
                        {
                            p = (byte)(p + (paper[c] - p) * cover + 0.5f);
                        }
                    }
                }
            }
        }

        return PillowImageOps.FromBytes(planes, h, w);
    }

    /// <summary>Faded ink: every pixel darker than the paper brought toward it by <paramref name="amount"/>, varied over the page by <paramref name="uneven"/>.</summary>
    public static ImageData Fade(ImageData image, double amount, double uneven, int seed)
    {
        if (amount == 0)
        {
            return image;
        }

        var planes = PillowImageOps.Bytes(image);
        int w = image.Width, h = image.Height;
        var paper = PaperLevels(planes, w * h);
        var grid = SmoothGrid(seed, out int cells);
        HostParallel.For(h, Grain(w), (first, last) =>
        {
            var keep = new float[w];
            for (int y = first; y < last; y++)
            {
                float v = (y + 0.5f) / h;
                for (int x = 0; x < w; x++)
                {
                    float field = uneven == 0 ? 1 : 1 - (float)uneven * Smooth(grid, cells, (x + 0.5f) / w, v);
                    keep[x] = 1 - (float)amount * field;
                }

                for (int c = 0; c < planes.Length; c++)
                {
                    var row = planes[c].AsSpan(y * w, w);
                    float p = paper[c];
                    for (int x = 0; x < w; x++)
                    {
                        if (row[x] < p)
                        {
                            row[x] = (byte)(p - (p - row[x]) * keep[x] + 0.5f);
                        }
                    }
                }
            }
        });

        return PillowImageOps.FromBytes(planes, h, w);
    }

    /// <summary>Tinted paper: each channel multiplied toward <paramref name="color"/> by <paramref name="amount"/> (grey: their mean).</summary>
    public static ImageData Tint(ImageData image, double amount, (float R, float G, float B) color)
    {
        if (amount == 0)
        {
            return image;
        }

        var planes = PillowImageOps.Bytes(image);
        float[] of = planes.Length == 3 ? [color.R, color.G, color.B] : [(color.R + color.G + color.B) / 3];
        Span<byte> lut = stackalloc byte[256];
        for (int c = 0; c < planes.Length; c++)
        {
            float factor = 1 - (float)amount * (1 - of[Math.Min(c, of.Length - 1)]);
            for (int v = 0; v < 256; v++)
            {
                lut[v] = (byte)(v * factor + 0.5f);
            }

            var plane = planes[c];
            for (int i = 0; i < plane.Length; i++)
            {
                plane[i] = lut[plane[i]];
            }
        }

        return PillowImageOps.FromBytes(planes, image.Height, image.Width);
    }

    /// <summary>Stains: <paramref name="count"/> soft blotches with darker, wavy rims, brownish, multiplied onto the page.</summary>
    public static ImageData Stain(ImageData image, int count, double opacity, int seed)
    {
        if (count == 0 || opacity == 0)
        {
            return image;
        }

        var planes = PillowImageOps.Bytes(image);
        int w = image.Width, h = image.Height, side = Math.Min(w, h);
        var random = new Random(seed);
        for (int k = 0; k < count; k++)
        {
            double cx = random.NextDouble() * w, cy = random.NextDouble() * h;
            double rx = side * (0.03 + random.NextDouble() * 0.12), ry = rx * (0.6 + random.NextDouble() * 0.8);
            double angle = random.NextDouble() * Math.PI, phase1 = random.NextDouble() * 6.3, phase2 = random.NextDouble() * 6.3;
            float alpha = (float)(opacity * (0.5 + 0.5 * random.NextDouble()));
            float[] tone = [(float)(0.7 + random.NextDouble() * 0.15), (float)(0.55 + random.NextDouble() * 0.15), (float)(0.35 + random.NextDouble() * 0.15)];
            if (planes.Length == 1)
            {
                tone = [(tone[0] + tone[1] + tone[2]) / 3];
            }

            double reach = Math.Max(rx, ry) * 1.3;
            int x0 = Math.Max(0, (int)(cx - reach)), x1 = Math.Min(w - 1, (int)(cx + reach));
            int y0 = Math.Max(0, (int)(cy - reach)), y1 = Math.Min(h - 1, (int)(cy + reach));
            double ca = PortableMath.Cos(angle), sa = PortableMath.Sin(angle);
            double c1 = PortableMath.Cos(phase1), s1 = PortableMath.Sin(phase1), c2 = PortableMath.Cos(phase2), s2 = PortableMath.Sin(phase2);
            HostParallel.For(y1 - y0 + 1, Grain(x1 - x0 + 1), (first, last) =>
            {
                for (int y = y0 + first; y < y0 + last; y++)
                {
                    for (int x = x0; x <= x1; x++)
                    {
                        double dx = x - cx, dy = y - cy;
                        double u = (dx * ca + dy * sa) / rx, v = (-dx * sa + dy * ca) / ry;
                        double r = Math.Sqrt(u * u + v * v);
                        if (r >= 1.15 * 1.23)
                        {
                            continue;   // beyond the widest the wavy edge reaches
                        }

                        // The edge's waves, sin(3θ + φ1) and sin(5θ + φ2), from cos θ and sin θ (Chebyshev), no angle.
                        double co = r > 0 ? u / r : 1, si = r > 0 ? v / r : 0;
                        double cos2 = co * co - si * si, sin2 = 2 * si * co;
                        double cos3 = cos2 * co - sin2 * si, sin3 = sin2 * co + cos2 * si;
                        double cos5 = cos3 * cos2 - sin3 * sin2, sin5 = sin3 * cos2 + cos3 * sin2;
                        double t = r / (1 + 0.15 * (sin3 * c1 + cos3 * s1) + 0.08 * (sin5 * c2 + cos5 * s2));
                        if (t >= 1.15)
                        {
                            continue;
                        }

                        // A faint body and a darker rim just inside the edge.
                        double body = t < 0.6 ? 1 : t < 1 ? 1 - (t - 0.6) / 0.4 : 0;
                        double e = (t - 0.95) / 0.06;
                        float a = alpha * (float)Math.Min(1, 0.45 * body + 0.9 * PortableMath.Exp(-e * e));
                        for (int c = 0; c < planes.Length; c++)
                        {
                            ref byte p = ref planes[c][y * w + x];
                            p = (byte)(p * (1 - a * (1 - tone[c])) + 0.5f);
                        }
                    }
                }
            });
        }

        return PillowImageOps.FromBytes(planes, h, w);
    }

    /// <summary>A scan at <paramref name="share"/> of the resolution: shrunk by area, then back to its size, bilinear.</summary>
    public static ImageData LowResolution(ImageData image, double share)
    {
        if (share >= 1)
        {
            return image;
        }

        int w = image.Width, h = image.Height;
        var small = PillowImageOps.Resize(image, Math.Max(1, (int)Math.Round(w * share)), Math.Max(1, (int)Math.Round(h * share)), ImageResampling.Box);
        return PillowImageOps.Resize(small, w, h, ImageResampling.Bilinear);
    }

    /// <summary>Bleed-through: the page's mirror image, blurred, darkening it by <paramref name="amount"/> where its ink is.</summary>
    public static ImageData BleedThrough(ImageData image, double amount, double blur)
    {
        if (amount == 0)
        {
            return image;
        }

        var planes = PillowImageOps.Bytes(image);
        int w = image.Width, h = image.Height;

        // The back's ink, in grey: the page's grey mirrored left to right, blurred, as darkness under its paper.
        var grey = planes.Length == 1 ? planes[0] : PillowImageOps.Grey(planes);
        var mirrored = GC.AllocateUninitializedArray<byte>(w * h);
        HostParallel.For(h, Grain(w), (first, last) =>
        {
            for (int y = first; y < last; y++)
            {
                grey.AsSpan(y * w, w).CopyTo(mirrored.AsSpan(y * w, w));
                mirrored.AsSpan(y * w, w).Reverse();
            }
        });

        var back = PillowImageOps.Bytes(PillowImageOps.GaussianBlur(PillowImageOps.FromBytes([mirrored], h, w), blur))[0];
        float paper = Math.Max(1, (int)PaperLevels([grey], w * h)[0]);
        Span<float> keep = stackalloc float[256];
        for (int v = 0; v < 256; v++)
        {
            keep[v] = 1 - (float)amount * Math.Clamp((paper - v) / paper, 0, 1);
        }

        var keepOf = keep.ToArray();
        HostParallel.For(h, Grain(w), (first, last) =>
        {
            foreach (var plane in planes)
            {
                for (int i = first * w; i < last * w; i++)
                {
                    plane[i] = (byte)(plane[i] * keepOf[back[i]] + 0.5f);
                }
            }
        });

        return PillowImageOps.FromBytes(planes, h, w);
    }
}
