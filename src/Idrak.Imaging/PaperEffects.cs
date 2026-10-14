// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

namespace Idrak.Imaging;

/// <summary>
/// More of what happens to a paper (plan 18): age and damage (water, burn, foxing, staple, dogear, clip, residue,
/// wormholes, sunfade, piece, whiteout, sticky, burnhole, cutoff), copiers and printers (photocopy, carbon, lowtoner,
/// banding, ghost, feather, misregister), cameras (desk, glare, vignette, motion, moire, handshadow, fingerprint) and
/// the paper itself (lined, grid, guilloche, recycled). None takes away text a page's answer holds: what removes paper
/// (a missing piece, a burnt edge, holes, a page cut off, a dog-ear, a sticky note, a staple or a clip) stays in the
/// blank margins measured round the page's ink, correction fluid goes only on blank paper, and the rest leaves the
/// text readable.
/// </summary>
internal static partial class PaperImageOps
{
    // The blank margin on each side of the page's ink (top, right, bottom, left), less `keep` pixels.
    private static double[] Rooms(byte[][] planes, int w, int h, int keep = 8)
    {
        var ink = InkBox(planes, w, h);
        return [ink.Top - keep, w - 1 - ink.Right - keep, h - 1 - ink.Bottom - keep, ink.Left - keep];
    }

    private static byte[] GreyOf(byte[][] planes) => planes.Length == 1 ? planes[0] : PillowImageOps.Grey(planes);

    // Rows of the page, spread over the processors.
    private static void Rows(int w, int h, Action<int> row) =>
        HostParallel.For(h, ScanImageOps.Grain(w), (first, last) =>
        {
            for (int y = first; y < last; y++)
            {
                row(y);
            }
        });

    // How far (x, y) is inside the page from side s (0 top, 1 right, 2 bottom, 3 left), and where along that side it is.
    private static (double Inside, int Along) FromSide(int s, int x, int y, int w, int h) => s switch
    {
        0 => (y, x),
        1 => (w - 1 - x, y),
        2 => (h - 1 - y, x),
        _ => (x, y),
    };

    private static float[] Tint(int channels, float r, float g, float b) => channels == 3 ? [r, g, b] : [0.299f * r + 0.587f * g + 0.114f * b];

    // Each pixel of a box times 1 - weight * (1 - color), the weight from `weight(x, y)` (0 leaves it).
    private static void MultiplyIn(byte[][] planes, int w, int h, int x0, int y0, int x1, int y1, float[] color, Func<int, int, float> weight)
    {
        for (int y = Math.Max(0, y0); y <= Math.Min(h - 1, y1); y++)
        {
            for (int x = Math.Max(0, x0); x <= Math.Min(w - 1, x1); x++)
            {
                float k = weight(x, y);
                if (k <= 0)
                {
                    continue;
                }

                for (int c = 0; c < planes.Length; c++)
                {
                    ref byte v = ref planes[c][y * w + x];
                    v = (byte)Math.Clamp(v * (1 - k * (1 - color[c])) + 0.5f, 0, 255);
                }
            }
        }
    }

    /// <summary>Water marks: a wavy tide line, the paper inside a little stained and its ink run (softened and paler), still readable.</summary>
    public static ImageData Water(ImageData image, int count, int seed)
    {
        if (count == 0)
        {
            return image;
        }

        var planes = PillowImageOps.Bytes(image);
        int w = image.Width, h = image.Height, side = Math.Min(w, h);
        var random = new Random(seed);
        var run = PillowImageOps.Bytes(PillowImageOps.GaussianBlur(image, 1.1));
        var paper = ScanImageOps.PaperLevels(planes, w * h);
        var tide = Tint(planes.Length, 0.84f, 0.76f, 0.6f);
        var wash = Tint(planes.Length, 0.985f, 0.96f, 0.9f);
        for (int k = 0; k < count; k++)
        {
            double cx = w * random.NextDouble(), cy = h * random.NextDouble(), r = side * (0.06 + 0.12 * random.NextDouble());
            var waves = Enumerable.Range(0, 3).Select(_ => (Amp: 0.04 + 0.08 * random.NextDouble(), Freq: 2 + random.Next(5), Phase: random.NextDouble() * 6.3)).ToArray();
            double line = 1.5 + 2.5 * random.NextDouble(), soften = 0.4 + 0.4 * random.NextDouble();
            int reach = (int)(r * 1.4 + line + 2);
            for (int y = Math.Max(0, (int)cy - reach); y <= Math.Min(h - 1, (int)cy + reach); y++)
            {
                for (int x = Math.Max(0, (int)cx - reach); x <= Math.Min(w - 1, (int)cx + reach); x++)
                {
                    double dx = x + 0.5 - cx, dy = y + 0.5 - cy, d = Math.Sqrt(dx * dx + dy * dy), angle = Math.Atan2(dy, dx);
                    double edge = r * (1 + waves.Sum(v => v.Amp * Math.Sin(v.Freq * angle + v.Phase)));
                    double off = Math.Abs(d - edge);
                    if (d > edge + line)
                    {
                        continue;
                    }

                    int i = y * w + x;
                    for (int c = 0; c < planes.Length; c++)
                    {
                        float v = planes[c][i];
                        if (d < edge)
                        {
                            v += (run[c][i] - v) * (float)soften;   // the ink ran
                            v = v < paper[c] ? v + (paper[c] - v) * 0.12f : v;
                            v *= wash[c];
                        }

                        if (off < line)
                        {
                            v *= 1 - (float)(1 - off / line) * (1 - tide[c]);
                        }

                        planes[c][i] = (byte)Math.Clamp(v + 0.5f, 0, 255);
                    }
                }
            }
        }

        return PillowImageOps.FromBytes(planes, h, w);
    }

    /// <summary>Scorched edges: one to three sides burnt away to a jagged charred line, browned behind it, inside the margins.</summary>
    public static ImageData Burn(ImageData image, double depth, int? fill, int seed)
    {
        if (depth == 0)
        {
            return image;
        }

        var planes = PillowImageOps.Bytes(image);
        int w = image.Width, h = image.Height, side = Math.Min(w, h);
        var random = new Random(seed);
        var background = Background(planes.Length, fill, random);
        var rooms = Rooms(planes, w, h);
        var cut = new double[4][];
        var scorch = new double[4];
        var sides = Enumerable.Range(0, 4).OrderBy(_ => random.Next()).Take(1 + random.Next(3)).ToHashSet();
        for (int s = 0; s < 4; s++)
        {
            double most = Math.Min(depth * side, rooms[s] * 0.4);
            scorch[s] = Math.Min(side * (0.01 + 0.02 * random.NextDouble()), rooms[s] - most - 4);
            if (!sides.Contains(s) || scorch[s] < 2)
            {
                continue;
            }

            int length = s % 2 == 0 ? w : h;
            cut[s] = new double[length];
            double at = most * 0.5;
            for (int i = 0; i < length; i++)
            {
                at = Math.Clamp(at + (random.NextDouble() - 0.5) * 2.2, 0, most);
                cut[s][i] = at;
            }
        }

        var brown = Tint(planes.Length, 0.55f, 0.38f, 0.22f);
        var charred = Tint(planes.Length, 35, 25, 18);
        Rows(w, h, y =>
        {
            for (int x = 0; x < w; x++)
            {
                for (int s = 0; s < 4; s++)
                {
                    if (cut[s] is null)
                    {
                        continue;
                    }

                    var (inside, along) = FromSide(s, x, y, w, h);
                    double e = inside - cut[s][along];
                    if (e >= 3 + scorch[s])
                    {
                        continue;
                    }

                    int i = y * w + x;
                    for (int c = 0; c < planes.Length; c++)
                    {
                        ref byte v = ref planes[c][i];
                        if (e < 0)
                        {
                            v = background[c];
                        }
                        else if (e < 3)
                        {
                            v = (byte)(v + (charred[c] - v) * 0.85f);
                        }
                        else
                        {
                            float t = (float)((e - 3) / scorch[s]);
                            v = (byte)Math.Clamp(v * (brown[c] + (1 - brown[c]) * t * t), 0, 255);
                        }
                    }

                    break;
                }
            }
        });
        return PillowImageOps.FromBytes(planes, h, w);
    }

    /// <summary>Foxing: small rust-brown spots of old paper, alone or in clusters.</summary>
    public static ImageData Foxing(ImageData image, int count, int seed)
    {
        if (count == 0)
        {
            return image;
        }

        var planes = PillowImageOps.Bytes(image);
        int w = image.Width, h = image.Height, side = Math.Min(w, h);
        var random = new Random(seed);
        var rust = Tint(planes.Length, 0.8f, 0.6f, 0.4f);
        double cx = w * random.NextDouble(), cy = h * random.NextDouble();
        for (int k = 0; k < count; k++)
        {
            if (random.NextDouble() < 0.5)
            {
                (cx, cy) = (w * random.NextDouble(), h * random.NextDouble());
            }
            else
            {
                (cx, cy) = (Math.Clamp(cx + (random.NextDouble() - 0.5) * 40, 0, w - 1), Math.Clamp(cy + (random.NextDouble() - 0.5) * 40, 0, h - 1));
            }

            double r = Math.Max(0.8, side * (0.0008 + 0.005 * random.NextDouble() * random.NextDouble()));
            float strength = (float)(0.2 + 0.35 * random.NextDouble());
            double sx = cx, sy = cy;
            MultiplyIn(planes, w, h, (int)(sx - r - 1), (int)(sy - r - 1), (int)(sx + r + 1), (int)(sy + r + 1), rust, (x, y) =>
            {
                double d = Math.Sqrt((x + 0.5 - sx) * (x + 0.5 - sx) + (y + 0.5 - sy) * (y + 0.5 - sy));
                return d >= r ? 0 : strength * (float)Math.Pow(1 - d / r, 1.5);
            });
        }

        return PillowImageOps.FromBytes(planes, h, w);
    }

    /// <summary>Staples in a top corner's margin: the wire, or the two holes and rust a removed one leaves.</summary>
    public static ImageData Staple(ImageData image, int count, int? fill, int seed)
    {
        if (count == 0)
        {
            return image;
        }

        var planes = PillowImageOps.Bytes(image);
        int w = image.Width, h = image.Height, side = Math.Min(w, h);
        var random = new Random(seed);
        var rooms = Rooms(planes, w, h);
        bool left = random.NextDouble() < 0.6;
        // The corner square misses the ink when it stays inside either of the corner's two margins.
        double room = Math.Max(left ? rooms[3] : rooms[1], rooms[0]);
        double length = Math.Min(side * (0.022 + 0.008 * random.NextDouble()), room - 26);
        if (length < 8)
        {
            return image;   // no corner margin for a staple
        }

        var background = Background(planes.Length, fill, random);
        var rust = Tint(planes.Length, 0.75f, 0.55f, 0.35f);
        double angle = random.NextDouble() < 0.6 ? (left ? -1 : 1) * (Math.PI / 4 + (random.NextDouble() - 0.5) * 0.5) : (random.NextDouble() - 0.5) * 0.28;
        double ux = Math.Cos(angle), uy = Math.Sin(angle);
        double cx0 = room * (0.44 + 0.06 * random.NextDouble()), cy0 = room * (0.44 + 0.06 * random.NextDouble());   // the staple, its rust and a second one stay within the room
        for (int k = 0; k < count; k++)
        {
            double cx = cx0 + k * (4 + 4 * random.NextDouble()) * ux, cy = cy0 + k * (4 + 4 * random.NextDouble()) * uy;
            cx = left ? cx : w - 1 - cx;
            double half = length / 2, ax = cx - ux * half, ay = cy - uy * half, bx = cx + ux * half, by = cy + uy * half;
            bool removed = random.NextDouble() < 0.35;
            if (removed || random.NextDouble() < 0.5)
            {
                double halo = 2 + 3 * random.NextDouble();
                float strength = (float)(0.25 + 0.3 * random.NextDouble());
                MultiplyIn(planes, w, h, (int)(Math.Min(ax, bx) - halo), (int)(Math.Min(ay, by) - halo), (int)(Math.Max(ax, bx) + halo), (int)(Math.Max(ay, by) + halo), rust, (x, y) =>
                {
                    double px = x + 0.5 - ax, py = y + 0.5 - ay, t = Math.Clamp(px * ux + py * uy, 0, length);
                    double ex = px - t * ux, ey = py - t * uy, d = Math.Sqrt(ex * ex + ey * ey);
                    return d >= halo ? 0 : strength * (float)(1 - d / halo);
                });
            }

            if (removed)
            {
                foreach (var (hx, hy) in new[] { (ax, ay), (bx, by) })
                {
                    double r = 1.2 + 0.6 * random.NextDouble();
                    for (int y = (int)(hy - r - 2); y <= (int)(hy + r + 2); y++)
                    {
                        for (int x = (int)(hx - r - 2); x <= (int)(hx + r + 2); x++)
                        {
                            double d = Math.Sqrt((x + 0.5 - hx) * (x + 0.5 - hx) + (y + 0.5 - hy) * (y + 0.5 - hy));
                            if (x < 0 || y < 0 || x >= w || y >= h || d > r + 1)
                            {
                                continue;
                            }

                            for (int c = 0; c < planes.Length; c++)
                            {
                                ref byte v = ref planes[c][y * w + x];
                                v = d < r ? background[c] : (byte)(v * 0.8f);
                            }
                        }
                    }
                }

                continue;
            }

            float metal = (float)(85 + 30 * random.NextDouble());
            Stroke(planes, w, h, [(ax, ay), (bx, by)], 2.2, [.. Enumerable.Repeat(metal, planes.Length)], 1f);
            Stroke(planes, w, h, [(ax - uy * 0.6, ay + ux * 0.6), (bx - uy * 0.6, by + ux * 0.6)], 0.8, [.. Enumerable.Repeat(210f, planes.Length)], 0.8f, lighten: true);
        }

        return PillowImageOps.FromBytes(planes, h, w);
    }

    /// <summary>A dog-ear: a corner folded over onto the page (the back of the paper showing, a shadow round it), the background where the corner was; inside the margins.</summary>
    public static ImageData DogEar(ImageData image, double size, int? fill, int seed)
    {
        if (size == 0)
        {
            return image;
        }

        var planes = PillowImageOps.Bytes(image);
        int w = image.Width, h = image.Height, side = Math.Min(w, h);
        var random = new Random(seed);
        var rooms = Rooms(planes, w, h);
        int corner = random.Next(4);

        // The folded square [0, a]² misses the ink when it stays inside one of the corner's two margins.
        double roomX = corner is 1 or 2 ? rooms[1] : rooms[3], roomY = corner is 2 or 3 ? rooms[2] : rooms[0];
        double a = Math.Min(size * side, Math.Max(roomX, roomY) - 6);
        if (a < 10)
        {
            return image;
        }

        var background = Background(planes.Length, fill, random);
        var paper = ScanImageOps.PaperLevels(planes, w * h);
        float back = (float)(0.88 + 0.07 * random.NextDouble());
        int reach = (int)a + 6;
        for (int v = 0; v <= reach && v < h; v++)
        {
            for (int u = 0; u <= reach && u < w; u++)
            {
                int x = corner is 1 or 2 ? w - 1 - u : u, y = corner is 2 or 3 ? h - 1 - v : v, i = y * w + x;
                double s = u + v;
                for (int c = 0; c < planes.Length; c++)
                {
                    ref byte p = ref planes[c][i];
                    if (s < a)
                    {
                        p = background[c];
                    }
                    else if (u <= a && v <= a)
                    {
                        // The flap: the back of the paper, a little darker toward the fold, the crease a dark line.
                        float shade = back * (float)(0.97 + 0.03 * Math.Min(1, (s - a) / a));
                        p = (byte)Math.Clamp(paper[c] * shade * (s - a < 1.2 ? 0.85f : 1f), 0, 255);
                    }
                    else if (s >= a)
                    {
                        double beyond = Math.Max(u - a, v - a);
                        if (beyond < 5)
                        {
                            p = (byte)(p * (1 - 0.18f * (float)(1 - beyond / 5)));
                        }
                    }
                }
            }
        }

        return PillowImageOps.FromBytes(planes, h, w);
    }

    // A loop of wire: a rounded rectangle `width` wide from `top` to `bottom` (local coordinates, x across, y down).
    private static List<(double X, double Y)> Loop(double width, double top, double bottom)
    {
        double r = width / 2;
        var points = new List<(double, double)>();
        for (int k = 0; k <= 12; k++)
        {
            double t = Math.PI + Math.PI * k / 12;
            points.Add((r * Math.Cos(t), top + r + r * Math.Sin(t)));
        }

        for (int k = 0; k <= 12; k++)
        {
            double t = Math.PI * k / 12;
            points.Add((r * Math.Cos(t), bottom - r + r * Math.Sin(t)));
        }

        points.Add(points[0]);
        return points;
    }

    /// <summary>Paper clips over the top edge, in the top margin: the wire with its shadow, or only the dents a removed one leaves.</summary>
    public static ImageData Clip(ImageData image, int count, int seed)
    {
        if (count == 0)
        {
            return image;
        }

        var planes = PillowImageOps.Bytes(image);
        int w = image.Width, h = image.Height, side = Math.Min(w, h);
        var random = new Random(seed);
        double room = Rooms(planes, w, h)[0];
        double length = Math.Min(side * (0.07 + 0.04 * random.NextDouble()), (room - 6) / 0.6);
        if (length < 20)
        {
            return image;
        }

        var paper = ScanImageOps.PaperLevels(planes, w * h);
        for (int k = 0; k < count; k++)
        {
            double width = length * 0.28, cx = w * (0.15 + 0.7 * random.NextDouble()), tilt = (random.NextDouble() - 0.5) * 0.2;
            double cos = Math.Cos(tilt), sin = Math.Sin(tilt);
            List<(double, double)> Place(List<(double X, double Y)> loop, double dx, double dy) =>
                [.. loop.Select(p => (cx + p.X * cos - p.Y * sin + dx, p.X * sin + p.Y * cos + dy))];
            var outer = Loop(width, -0.4 * length, 0.6 * length);
            var inner = Loop(width * 0.62, -0.25 * length, 0.45 * length);
            if (random.NextDouble() < 0.3)
            {
                float[] dent = [.. paper.Select(p => p * 0.88f)];
                Stroke(planes, w, h, Place(outer, 0, 0), 1.2, dent, 0.6f);
                Stroke(planes, w, h, Place(inner, 0, 0), 1.2, dent, 0.6f);
                continue;
            }

            float[] shadow = [.. paper.Select(p => p * 0.78f)], wire = Tint(planes.Length, 120, 122, 128), shine = [.. Enumerable.Repeat(228f, planes.Length)];
            foreach (var loop in new[] { outer, inner })
            {
                Stroke(planes, w, h, Place(loop, 2, 2), 2.5, shadow, 0.6f);
                Stroke(planes, w, h, Place(loop, 0, 0), 1.8, wire, 1f);
                Stroke(planes, w, h, Place(loop, -0.5, -0.5), 0.7, shine, 0.7f, lighten: true);
            }
        }

        return PillowImageOps.FromBytes(planes, h, w);
    }

    // The pixels of a strip centred on (cx, cy) along (ax, ay), `length` by `width`, its ends cut in teeth when `teeth`:
    // each with its place along (a) and across (b) the strip and whether it is on the strip's edge.
    private static void Strip(int w, int h, double cx, double cy, double ax, double ay, double length, double width, bool teeth, Action<int, int, double, double, bool> pixel)
    {
        double half = length / 2, across = width / 2, reach = Math.Sqrt((half + 3) * (half + 3) + across * across);
        for (int y = Math.Max(0, (int)(cy - reach)); y <= Math.Min(h - 1, (int)(cy + reach)); y++)
        {
            for (int x = Math.Max(0, (int)(cx - reach)); x <= Math.Min(w - 1, (int)(cx + reach)); x++)
            {
                double px = x + 0.5 - cx, py = y + 0.5 - cy;
                double a = px * ax + py * ay, b = -px * ay + py * ax;
                double end = half - (teeth ? Math.Abs((Math.Abs(b) / 2 % 2) - 1) * 2 : 0);   // a 4-pixel zigzag, 2 deep
                if (Math.Abs(b) > across || Math.Abs(a) > end)
                {
                    continue;
                }

                pixel(x, y, a / half, b / across, Math.Abs(b) > across - 1.2 || Math.Abs(a) > end - 1);
            }
        }
    }

    /// <summary>Old tape residue: yellow-brown, uneven, see-through patches where tape was pulled off, with flecks of torn paper.</summary>
    public static ImageData Residue(ImageData image, int count, int seed)
    {
        if (count == 0)
        {
            return image;
        }

        var planes = PillowImageOps.Bytes(image);
        int w = image.Width, h = image.Height, side = Math.Min(w, h);
        var random = new Random(seed);
        var glue = Tint(planes.Length, 0.95f, 0.86f, 0.66f);
        for (int k = 0; k < count; k++)
        {
            double angle = random.NextDouble() * Math.PI, length = side * (0.06 + 0.09 * random.NextDouble()), width = side * (0.02 + 0.015 * random.NextDouble());
            var field = ScanImageOps.SmoothGrid(random.Next(), out int cells);
            float strength = (float)(0.4 + 0.5 * random.NextDouble());
            var flecks = new Random(random.Next());
            Strip(w, h, w * random.NextDouble(), h * random.NextDouble(), Math.Cos(angle), Math.Sin(angle), length, width, false, (x, y, a, b, edge) =>
            {
                float k = strength * (0.5f + 0.5f * ScanImageOps.Smooth(field, cells, (float)(a + 1) / 2, (float)(b + 1) / 2)) * (edge ? 1.3f : 1f);
                bool fleck = flecks.NextDouble() < 0.02;
                for (int c = 0; c < planes.Length; c++)
                {
                    ref byte v = ref planes[c][y * w + x];
                    float value = v * (1 - Math.Min(1, k) * (1 - glue[c]));
                    v = (byte)Math.Clamp(fleck ? value + 20 : value, 0, 255);
                }
            });
        }

        return PillowImageOps.FromBytes(planes, h, w);
    }

    /// <summary>Worm holes: small ragged holes, some in short runs, with a brown rim; in the margins.</summary>
    public static ImageData Wormholes(ImageData image, int count, int? fill, int seed)
    {
        if (count == 0)
        {
            return image;
        }

        var planes = PillowImageOps.Bytes(image);
        int w = image.Width, h = image.Height, side = Math.Min(w, h);
        var random = new Random(seed);
        var background = Background(planes.Length, fill, random);
        var ink = InkBox(planes, w, h);
        var rim = Tint(planes.Length, 0.7f, 0.58f, 0.45f);
        double cx = -1, cy = -1;
        for (int k = 0; k < count; k++)
        {
            double r = Math.Max(1.5, side * (0.0015 + 0.002 * random.NextDouble()));
            bool placed = false;
            for (int attempt = 0; attempt < 40 && !placed; attempt++)
            {
                (double x, double y) = cx >= 0 && random.NextDouble() < 0.4
                    ? (cx + (random.NextDouble() - 0.5) * 6 * r, cy + (random.NextDouble() - 0.5) * 6 * r)
                    : (w * random.NextDouble(), h * random.NextDouble());
                double keep = r * 1.4 + 8;
                bool outside = x < ink.Left - keep || x > ink.Right + keep || y < ink.Top - keep || y > ink.Bottom + keep;
                if (outside && x > keep && y > keep && x < w - keep && y < h - keep)
                {
                    (cx, cy, placed) = (x, y, true);
                }
            }

            if (!placed)
            {
                continue;
            }

            double phase = random.NextDouble() * 6.3, hx = cx, hy = cy;
            int reach = (int)(r * 1.4 + 3);
            for (int y = (int)hy - reach; y <= (int)hy + reach; y++)
            {
                for (int x = (int)hx - reach; x <= (int)hx + reach; x++)
                {
                    double dx = x + 0.5 - hx, dy = y + 0.5 - hy, d = Math.Sqrt(dx * dx + dy * dy), t = Math.Atan2(dy, dx);
                    double edge = r * (1 + 0.25 * Math.Sin(3 * t + phase) + 0.15 * Math.Sin(5 * t + 2 * phase));
                    if (d > edge + 1.5)
                    {
                        continue;
                    }

                    for (int c = 0; c < planes.Length; c++)
                    {
                        ref byte v = ref planes[c][y * w + x];
                        v = d < edge ? background[c] : (byte)(v * rim[c]);
                    }
                }
            }
        }

        return PillowImageOps.FromBytes(planes, h, w);
    }

    /// <summary>Sun-faded: a band along one edge where the paper yellowed and the ink went paler, by up to <paramref name="amount"/>.</summary>
    public static ImageData Sunfade(ImageData image, double amount, int seed)
    {
        if (amount == 0)
        {
            return image;
        }

        var planes = PillowImageOps.Bytes(image);
        int w = image.Width, h = image.Height;
        var random = new Random(seed);
        var paper = ScanImageOps.PaperLevels(planes, w * h);
        var yellow = Tint(planes.Length, 1f, 0.95f, 0.8f);
        int s = random.Next(4);
        double extent = s % 2 == 0 ? h : w, band = extent * (0.15 + 0.3 * random.NextDouble()), soft = extent * (0.05 + 0.07 * random.NextDouble());
        Rows(w, h, y =>
        {
            for (int x = 0; x < w; x++)
            {
                double inside = FromSide(s, x, y, w, h).Inside;
                double t = Math.Clamp((band + soft - inside) / soft, 0, 1);
                if (t <= 0)
                {
                    continue;
                }

                float f = (float)(t * t * (3 - 2 * t) * amount);
                for (int c = 0; c < planes.Length; c++)
                {
                    ref byte v = ref planes[c][y * w + x];
                    float value = v < paper[c] ? v + (paper[c] - v) * f : v;
                    v = (byte)Math.Clamp(value * (1 - f * 0.3f * (1 - yellow[c])) + 0.5f, 0, 255);
                }
            }
        });
        return PillowImageOps.FromBytes(planes, h, w);
    }

    /// <summary>Pieces torn out of the page's edge: deep ragged bites, the background through them, inside the margins.</summary>
    public static ImageData Piece(ImageData image, int count, int? fill, int seed)
    {
        if (count == 0)
        {
            return image;
        }

        var planes = PillowImageOps.Bytes(image);
        int w = image.Width, h = image.Height;
        var random = new Random(seed);
        var background = Background(planes.Length, fill, random);
        var rooms = Rooms(planes, w, h);
        var cut = new double[4][];
        var usable = Enumerable.Range(0, 4).Where(s => rooms[s] >= 12).ToList();
        for (int k = 0; k < count && usable.Count > 0; k++)
        {
            int s = usable[random.Next(usable.Count)], length = s % 2 == 0 ? w : h;
            cut[s] ??= new double[length];
            double center = length * random.NextDouble(), width = length * (0.08 + 0.17 * random.NextDouble()), deepest = rooms[s] * (0.6 + 0.35 * random.NextDouble());
            double jag = 0;
            for (int i = Math.Max(0, (int)(center - width)); i < Math.Min(length, (int)(center + width)); i++)
            {
                double bump = Math.Max(0, 1 - Math.Abs(i - center) / (width / 2));
                jag = Math.Clamp(jag + (random.NextDouble() - 0.5) * 2, -3, 3);
                double depth = bump > 0 ? Math.Clamp(deepest * Math.Pow(bump, 0.7) + jag, 0, rooms[s]) : 0;
                cut[s][i] = Math.Max(cut[s][i], depth);
            }
        }

        Rows(w, h, y =>
        {
            for (int x = 0; x < w; x++)
            {
                double inside = double.MaxValue;
                for (int s = 0; s < 4; s++)
                {
                    if (cut[s] is not null)
                    {
                        var (from, along) = FromSide(s, x, y, w, h);
                        inside = Math.Min(inside, from - cut[s][along]);
                    }
                }

                if (inside >= 1.5)
                {
                    continue;
                }

                for (int c = 0; c < planes.Length; c++)
                {
                    ref byte v = ref planes[c][y * w + x];
                    v = inside < 0 ? background[c] : (byte)Math.Min(255, v + 14);
                }
            }
        });
        return PillowImageOps.FromBytes(planes, h, w);
    }

    /// <summary>Correction fluid: dried white patches, slightly uneven with a grey rim, only on blank paper (it hides no text).</summary>
    public static ImageData Whiteout(ImageData image, int count, int seed)
    {
        if (count == 0)
        {
            return image;
        }

        var planes = PillowImageOps.Bytes(image);
        int w = image.Width, h = image.Height, side = Math.Min(w, h);
        var random = new Random(seed);
        var paper = ScanImageOps.PaperLevels(planes, w * h);
        var grey = GreyOf(planes);
        int light = (int)(paper.Average(p => p) * 0.85);

        // Ink counted over any box at once: a summed table of the pixels darker than the paper.
        var sum = new int[(w + 1) * (h + 1)];
        for (int y = 0; y < h; y++)
        {
            int row = 0;
            for (int x = 0; x < w; x++)
            {
                row += grey[y * w + x] < light ? 1 : 0;
                sum[(y + 1) * (w + 1) + x + 1] = sum[y * (w + 1) + x + 1] + row;
            }
        }

        int Ink(int x0, int y0, int x1, int y1) => sum[y1 * (w + 1) + x1] - sum[y0 * (w + 1) + x1] - sum[y1 * (w + 1) + x0] + sum[y0 * (w + 1) + x0];
        var bands = Bands(planes, w, h, paper);
        double line = bands.Count > 0 ? bands.Average(b => b.Bottom - b.Top + 1) : side * 0.02;
        for (int k = 0; k < count; k++)
        {
            double bw = side * (0.04 + 0.08 * random.NextDouble()), bh = line * (0.8 + 0.5 * random.NextDouble());
            for (int attempt = 0; attempt < 60; attempt++)
            {
                int x0 = (int)(4 + (w - bw - 8) * random.NextDouble()), y0 = (int)(4 + (h - bh - 8) * random.NextDouble()), x1 = (int)(x0 + bw), y1 = (int)(y0 + bh);
                if (Ink(Math.Max(0, x0 - 3), Math.Max(0, y0 - 3), Math.Min(w, x1 + 4), Math.Min(h, y1 + 4)) > 0)
                {
                    continue;
                }

                double cx = (x0 + x1) / 2.0, cy = (y0 + y1) / 2.0, rx = bw / 2, ry = bh / 2, phase = random.NextDouble() * 6.3;
                for (int y = y0; y <= y1; y++)
                {
                    for (int x = x0; x <= x1; x++)
                    {
                        double u = Math.Abs(x + 0.5 - cx) / rx, v = Math.Abs(y + 0.5 - cy) / ry;
                        double shape = Math.Pow(u, 4) + Math.Pow(v, 4), edge = 1 + 0.08 * Math.Sin(x * 0.3 + phase);
                        if (shape > edge)
                        {
                            continue;
                        }

                        float bump = (float)(Math.Sin(x * 0.7 + phase) * Math.Sin(y * 0.9 - phase) * 2.5);
                        for (int c = 0; c < planes.Length; c++)
                        {
                            float white = (planes.Length == 3 && c == 2 ? 245 : 249) + bump;
                            planes[c][y * w + x] = (byte)Math.Clamp(shape > edge * 0.8 ? white * 0.9f : white, 0, 255);
                        }
                    }
                }

                break;
            }
        }

        return PillowImageOps.FromBytes(planes, h, w);
    }

    /// <summary>Sticky notes stuck on an edge, mostly off the page: the part on the page, with its shadow, inside the margin.</summary>
    public static ImageData Sticky(ImageData image, int count, int seed)
    {
        if (count == 0)
        {
            return image;
        }

        var planes = PillowImageOps.Bytes(image);
        int w = image.Width, h = image.Height, side = Math.Min(w, h);
        var random = new Random(seed);
        var rooms = Rooms(planes, w, h);
        (float, float, float)[] colors = [(255, 240, 120), (255, 190, 205), (175, 220, 255), (195, 240, 170), (255, 205, 140)];
        for (int k = 0; k < count; k++)
        {
            var usable = Enumerable.Range(0, 4).Where(s => rooms[s] >= 16).ToList();
            if (usable.Count == 0)
            {
                break;
            }

            int s = usable[random.Next(usable.Count)];
            double size = side * (0.08 + 0.05 * random.NextDouble()), turn = (random.NextDouble() - 0.5) * 0.28;
            double extent = size * (Math.Abs(Math.Cos(turn)) + Math.Abs(Math.Sin(turn))) / 2, depth = rooms[s] * (0.4 + 0.45 * random.NextDouble()) - 4;
            double along = (s % 2 == 0 ? w : h) * (0.1 + 0.8 * random.NextDouble()), from = depth - extent;   // its centre, off the page
            (double cx, double cy) = s switch
            {
                0 => (along, from),
                1 => (w - 1 - from, along),
                2 => (along, h - 1 - from),
                _ => (from, along),
            };
            var (red, green, blue) = colors[random.Next(colors.Length)];
            var color = Tint(planes.Length, red, green, blue);
            double ux = Math.Cos(turn), uy = Math.Sin(turn), half = size / 2;
            for (int y = Math.Max(0, (int)(cy - extent - 4)); y <= Math.Min(h - 1, (int)(cy + extent + 4)); y++)
            {
                for (int x = Math.Max(0, (int)(cx - extent - 4)); x <= Math.Min(w - 1, (int)(cx + extent + 4)); x++)
                {
                    double px = x + 0.5 - cx, py = y + 0.5 - cy, a = px * ux + py * uy, b = -px * uy + py * ux;
                    double outside = Math.Max(Math.Abs(a), Math.Abs(b)) - half;
                    for (int c = 0; c < planes.Length; c++)
                    {
                        ref byte v = ref planes[c][y * w + x];
                        if (outside < 0)
                        {
                            float shade = (float)(0.96 + 0.04 * (b / half + 1) / 2) * (outside > -1.2 ? 0.92f : 1f);
                            v = (byte)Math.Clamp(color[c] * shade, 0, 255);
                        }
                        else if (outside < 3)
                        {
                            v = (byte)(v * (1 - 0.12f * (float)(1 - outside / 3)));
                        }
                    }
                }
            }
        }

        return PillowImageOps.FromBytes(planes, h, w);
    }

    /// <summary>Burn holes (a cigarette, an ember): a ragged hole, a charred ring and a brown scorch round it; in the margins.</summary>
    public static ImageData BurnHole(ImageData image, int count, int? fill, int seed)
    {
        if (count == 0)
        {
            return image;
        }

        var planes = PillowImageOps.Bytes(image);
        int w = image.Width, h = image.Height, side = Math.Min(w, h);
        var random = new Random(seed);
        var background = Background(planes.Length, fill, random);
        var ink = InkBox(planes, w, h);
        var brown = Tint(planes.Length, 0.5f, 0.35f, 0.2f);
        var charred = Tint(planes.Length, 30, 22, 15);
        for (int k = 0; k < count; k++)
        {
            double r = Math.Max(2, side * (0.003 + 0.005 * random.NextDouble())), halo = r * (2.2 + random.NextDouble());
            for (int attempt = 0; attempt < 40; attempt++)
            {
                double cx = w * random.NextDouble(), cy = h * random.NextDouble(), keep = halo + 8;
                bool outside = cx < ink.Left - keep || cx > ink.Right + keep || cy < ink.Top - keep || cy > ink.Bottom + keep;
                if (!outside || cx < halo || cy < halo || cx > w - halo || cy > h - halo)
                {
                    continue;
                }

                double phase = random.NextDouble() * 6.3;
                for (int y = (int)(cy - halo); y <= (int)(cy + halo); y++)
                {
                    for (int x = (int)(cx - halo); x <= (int)(cx + halo); x++)
                    {
                        double dx = x + 0.5 - cx, dy = y + 0.5 - cy, d = Math.Sqrt(dx * dx + dy * dy);
                        double edge = r * (1 + 0.2 * Math.Sin(4 * Math.Atan2(dy, dx) + phase));
                        if (d > halo)
                        {
                            continue;
                        }

                        for (int c = 0; c < planes.Length; c++)
                        {
                            ref byte v = ref planes[c][y * w + x];
                            float t = (float)Math.Clamp((d - edge - 1.5) / (halo - edge - 1.5), 0, 1);
                            v = d < edge ? background[c]
                                : d < edge + 1.5 ? (byte)(v + (charred[c] - v) * 0.9f)
                                : (byte)Math.Clamp(v * (brown[c] + (1 - brown[c]) * t), 0, 255);
                        }
                    }
                }

                break;
            }
        }

        return PillowImageOps.FromBytes(planes, h, w);
    }

    /// <summary>The page cut off by the scanner: a band along one or two sides showing the scanner bed, slightly skewed, inside the margins.</summary>
    public static ImageData Cutoff(ImageData image, double depth, int? fill, int seed)
    {
        if (depth == 0)
        {
            return image;
        }

        var planes = PillowImageOps.Bytes(image);
        int w = image.Width, h = image.Height, side = Math.Min(w, h);
        var random = new Random(seed);
        var background = Background(planes.Length, fill, random);
        var rooms = Rooms(planes, w, h);
        int first = random.Next(4);
        var sides = random.NextDouble() < 0.25 ? new[] { first, (first + 1) % 4 } : [first];
        var ends = new (double Start, double End)?[4];
        foreach (int s in sides)
        {
            double most = Math.Min(depth * side, rooms[s] - 3);
            if (most >= 2)
            {
                ends[s] = (most * (0.3 + 0.7 * random.NextDouble()), most * (0.3 + 0.7 * random.NextDouble()));
            }
        }

        Rows(w, h, y =>
        {
            for (int x = 0; x < w; x++)
            {
                for (int s = 0; s < 4; s++)
                {
                    if (ends[s] is not { } band)
                    {
                        continue;
                    }

                    var (inside, along) = FromSide(s, x, y, w, h);
                    double e = inside - (band.Start + (band.End - band.Start) * along / (s % 2 == 0 ? w : h));
                    if (e >= 2)
                    {
                        continue;
                    }

                    for (int c = 0; c < planes.Length; c++)
                    {
                        ref byte v = ref planes[c][y * w + x];
                        v = e < 0 ? background[c] : (byte)(v * 0.8f);
                    }

                    break;
                }
            }
        });
        return PillowImageOps.FromBytes(planes, h, w);
    }

    /// <summary>A photocopy, <paramref name="generations"/> times over: grey, harder contrast, strokes grown, the paper whiter and uneven, toner dust.</summary>
    public static ImageData Photocopy(ImageData image, int generations, int seed)
    {
        if (generations == 0)
        {
            return image;
        }

        var planes = PillowImageOps.Bytes(image);
        int w = image.Width, h = image.Height;
        var random = new Random(seed);
        var grey = (byte[])GreyOf(planes).Clone();
        for (int g = 0; g < generations; g++)
        {
            byte paper = ScanImageOps.PaperLevels([grey], w * h)[0];
            var thick = PillowImageOps.Bytes(PillowImageOps.MinFilter(PillowImageOps.FromBytes([grey], h, w), 3))[0];
            float spread = (float)(0.2 + 0.2 * random.NextDouble()), contrast = (float)(1.2 + 0.15 * random.NextDouble());
            var field = ScanImageOps.SmoothGrid(random.Next(), out int cells);
            var source = grey;
            var next = new byte[w * h];
            Rows(w, h, y =>
            {
                for (int x = 0; x < w; x++)
                {
                    int i = y * w + x;
                    float v = source[i] * (1 - spread) + thick[i] * spread;
                    v = paper + (v - paper) * contrast;
                    v += v > paper - 20 ? 8 : 0;   // the copier whitens the paper
                    v += (ScanImageOps.Smooth(field, cells, (float)x / w, (float)y / h) - 0.5f) * 18;
                    next[i] = (byte)Math.Clamp(v + 0.5f, 0, 255);
                }
            });
            grey = next;

            // Toner dust.
            int dust = (int)(w * (double)h / 1e6 * (60 + 120 * random.NextDouble()));
            for (int k = 0; k < dust; k++)
            {
                int x = random.Next(w), y = random.Next(h);
                byte dark = (byte)(30 + random.Next(60));
                grey[y * w + x] = Math.Min(grey[y * w + x], dark);
                if (random.NextDouble() < 0.3 && x + 1 < w)
                {
                    grey[y * w + x + 1] = Math.Min(grey[y * w + x + 1], dark);
                }
            }
        }

        return PillowImageOps.FromBytes([grey], h, w);
    }

    /// <summary>A carbon copy: the ink turned carbon blue or violet by <paramref name="amount"/>, soft, with a faint doubled edge.</summary>
    public static ImageData Carbon(ImageData image, double amount, int seed)
    {
        if (amount == 0)
        {
            return image;
        }

        var planes = PillowImageOps.Bytes(image);
        int w = image.Width, h = image.Height;
        var random = new Random(seed);
        var paper = ScanImageOps.PaperLevels(planes, w * h);
        var soft = GreyOf(PillowImageOps.Bytes(PillowImageOps.GaussianBlur(image, 0.6 + 0.4 * random.NextDouble())));
        float paperGrey = Math.Max(1, (float)paper.Average(p => p));
        var carbon = random.NextDouble() < 0.6 ? Tint(planes.Length, 70, 60, 150) : Tint(planes.Length, 40, 45, 95);
        int gx = 1 + random.Next(2), gy = random.Next(2);
        float ghost = (float)(0.15 + 0.15 * random.NextDouble()), a = (float)amount;
        Rows(w, h, y =>
        {
            for (int x = 0; x < w; x++)
            {
                int i = y * w + x;
                float d = Math.Clamp(1 - soft[i] / paperGrey, 0, 1);
                if (x >= gx && y >= gy)
                {
                    d = Math.Max(d, ghost * Math.Clamp(1 - soft[i - gy * w - gx] / paperGrey, 0, 1));
                }

                for (int c = 0; c < planes.Length; c++)
                {
                    ref byte v = ref planes[c][i];
                    float target = paper[c] - d * (paper[c] - carbon[c]);
                    v = (byte)Math.Clamp(v + (target - v) * a + 0.5f, 0, 255);
                }
            }
        });
        return PillowImageOps.FromBytes(planes, h, w);
    }

    /// <summary>Low toner: the ink faded in streaks down the page (a worn drum), by up to <paramref name="amount"/>.</summary>
    public static ImageData LowToner(ImageData image, double amount, int seed)
    {
        if (amount == 0)
        {
            return image;
        }

        var planes = PillowImageOps.Bytes(image);
        int w = image.Width, h = image.Height;
        var random = new Random(seed);
        var paper = ScanImageOps.PaperLevels(planes, w * h);

        // How faded each column is: a slow wave and a few sharp light streaks.
        var fade = new float[w];
        double period = w * (0.2 + 0.5 * random.NextDouble()), phase = random.NextDouble() * 6.3;
        var streaks = Enumerable.Range(0, 1 + random.Next(4)).Select(_ => (At: w * random.NextDouble(), Width: 3 + 25 * random.NextDouble())).ToArray();
        for (int x = 0; x < w; x++)
        {
            double f = 0.35 + 0.35 * Math.Sin(x / period * 6.283 + phase);
            f = Math.Max(f, streaks.Max(s => Math.Max(0, 1 - Math.Abs(x - s.At) / s.Width)));
            fade[x] = (float)(Math.Clamp(f, 0, 1) * amount);
        }

        Rows(w, h, y =>
        {
            for (int x = 0; x < w; x++)
            {
                for (int c = 0; c < planes.Length; c++)
                {
                    ref byte v = ref planes[c][y * w + x];
                    if (v < paper[c])
                    {
                        v = (byte)(v + (paper[c] - v) * fade[x] + 0.5f);
                    }
                }
            }
        });
        return PillowImageOps.FromBytes(planes, h, w);
    }

    /// <summary>Printer banding: the ink lighter and darker in bands across the page, by up to <paramref name="amount"/>.</summary>
    public static ImageData Banding(ImageData image, double amount, int seed)
    {
        if (amount == 0)
        {
            return image;
        }

        var planes = PillowImageOps.Bytes(image);
        int w = image.Width, h = image.Height;
        var random = new Random(seed);
        var paper = ScanImageOps.PaperLevels(planes, w * h);
        double period = 20 + 60 * random.NextDouble(), phase = random.NextDouble() * 6.3, period2 = period * (2.3 + random.NextDouble());
        Rows(w, h, y =>
        {
            float band = (float)(amount * (0.5 + 0.35 * Math.Sin(y / period * 6.283 + phase) + 0.15 * Math.Sin(y / period2 * 6.283)));
            for (int x = 0; x < w; x++)
            {
                for (int c = 0; c < planes.Length; c++)
                {
                    ref byte v = ref planes[c][y * w + x];
                    if (v < paper[c])
                    {
                        v = (byte)Math.Clamp(v + (paper[c] - v) * band + 0.5f, 0, 255);
                    }
                }
            }
        });
        return PillowImageOps.FromBytes(planes, h, w);
    }

    /// <summary>A ghost: a faint copy of the page's ink a little way off (a printer's or copier's double image), <paramref name="amount"/> as dark.</summary>
    public static ImageData Ghost(ImageData image, double amount, int seed)
    {
        if (amount == 0)
        {
            return image;
        }

        var planes = PillowImageOps.Bytes(image);
        int w = image.Width, h = image.Height;
        var random = new Random(seed);
        var paper = ScanImageOps.PaperLevels(planes, w * h);
        int dx = (random.NextDouble() < 0.5 ? -1 : 1) * (3 + random.Next(30)), dy = random.NextDouble() < 0.5 ? random.Next(-4, 5) : (random.NextDouble() < 0.5 ? -1 : 1) * (5 + random.Next(35));
        var source = planes.Select(p => (byte[])p.Clone()).ToArray();
        float a = (float)amount;
        Rows(w, h, y =>
        {
            int sy = y - dy;
            if (sy < 0 || sy >= h)
            {
                return;
            }

            for (int x = 0; x < w; x++)
            {
                int sx = x - dx;
                if (sx < 0 || sx >= w)
                {
                    continue;
                }

                for (int c = 0; c < planes.Length; c++)
                {
                    float ghost = paper[c] - (paper[c] - source[c][sy * w + sx]) * a;
                    ref byte v = ref planes[c][y * w + x];
                    v = (byte)Math.Min(v, Math.Clamp(ghost, 0, 255));
                }
            }
        });
        return PillowImageOps.FromBytes(planes, h, w);
    }

    /// <summary>Feathering: ink spread into the paper (an inkjet on soft paper), strokes fuzzier and a little bolder, by <paramref name="amount"/>.</summary>
    public static ImageData Feather(ImageData image, double amount, int seed)
    {
        if (amount == 0)
        {
            return image;
        }

        var planes = PillowImageOps.Bytes(image);
        int w = image.Width, h = image.Height;
        var random = new Random(seed);
        var spread = PillowImageOps.Bytes(PillowImageOps.GaussianBlur(PillowImageOps.MinFilter(image, 3), 0.8 + 0.6 * random.NextDouble()));
        float a = (float)amount;
        Rows(w, h, y =>
        {
            for (int i = y * w; i < (y + 1) * w; i++)
            {
                for (int c = 0; c < planes.Length; c++)
                {
                    ref byte v = ref planes[c][i];
                    v = (byte)Math.Min(v, v + (spread[Math.Min(c, spread.Length - 1)][i] - v) * a + 0.5f);
                }
            }
        });
        return PillowImageOps.FromBytes(planes, h, w);
    }

    /// <summary>Colour misregistration: the red and blue of a colour print shifted up to <paramref name="pixels"/> apart (a grey page is left as it is).</summary>
    public static ImageData Misregister(ImageData image, double pixels, int seed)
    {
        if (pixels == 0 || image.Channels != 3)
        {
            return image;
        }

        var planes = PillowImageOps.Bytes(image);
        int w = image.Width, h = image.Height;
        var random = new Random(seed);
        double angle = random.NextDouble() * 6.283, size = pixels * (0.5 + 0.5 * random.NextDouble());
        int dx = (int)Math.Round(Math.Cos(angle) * size), dy = (int)Math.Round(Math.Sin(angle) * size);
        if (dx == 0 && dy == 0)
        {
            dx = 1;
        }

        var source = planes.Select(p => (byte[])p.Clone()).ToArray();
        Rows(w, h, y =>
        {
            for (int x = 0; x < w; x++)
            {
                int rx = Math.Clamp(x - dx, 0, w - 1), ry = Math.Clamp(y - dy, 0, h - 1), bx = Math.Clamp(x + dx, 0, w - 1), by = Math.Clamp(y + dy, 0, h - 1);
                planes[0][y * w + x] = source[0][ry * w + rx];
                planes[2][y * w + x] = source[2][by * w + bx];
            }
        });
        return PillowImageOps.FromBytes(planes, h, w);
    }

    /// <summary>A page photographed on a desk: the page <paramref name="share"/> smaller, on wood, a grey table or a dark cloth, with a soft shadow.</summary>
    public static ImageData Desk(ImageData image, double share, int seed)
    {
        if (share == 0)
        {
            return image;
        }

        var planes = PillowImageOps.Bytes(image);
        int w = image.Width, h = image.Height;
        var random = new Random(seed);
        double scale = 1 - share * (0.5 + 0.5 * random.NextDouble());
        double pw = w * scale, ph = h * scale, ox = (w - pw) * random.NextDouble(), oy = (h - ph) * random.NextDouble();
        (float, float, float)[] tables = [(140, 95, 60), (110, 110, 112), (45, 45, 50), (175, 150, 115)];
        var (tr, tg, tb) = tables[random.Next(tables.Length)];
        var table = Tint(planes.Length, tr, tg, tb);
        bool wood = table[0] > table[planes.Length - 1] * 1.3 || (planes.Length == 1 && random.NextDouble() < 0.5);
        var field = ScanImageOps.SmoothGrid(random.Next(), out int cells);
        double grainPeriod = 3 + 4 * random.NextDouble(), grainPhase = random.NextDouble() * 6.3;
        double shadow = 4 + 8 * random.NextDouble();
        var result = planes.Select(_ => new byte[w * h]).ToArray();
        Rows(w, h, y =>
        {
            for (int x = 0; x < w; x++)
            {
                int i = y * w + x;
                double u = (x + 0.5 - ox) / scale - 0.5, v = (y + 0.5 - oy) / scale - 0.5;
                if (u >= 0 && v >= 0 && u <= w - 1 && v <= h - 1)
                {
                    int x0 = Math.Min(w - 2, (int)u), y0 = Math.Min(h - 2, (int)v);
                    float fx = (float)(u - x0), fy = (float)(v - y0);
                    for (int c = 0; c < planes.Length; c++)
                    {
                        var p = planes[c];
                        float top = p[y0 * w + x0] + (p[y0 * w + x0 + 1] - p[y0 * w + x0]) * fx;
                        float bottom = p[(y0 + 1) * w + x0] + (p[(y0 + 1) * w + x0 + 1] - p[(y0 + 1) * w + x0]) * fx;
                        result[c][i] = (byte)Math.Clamp(top + (bottom - top) * fy + 0.5f, 0, 255);
                    }

                    continue;
                }

                // The table: wood with long curving grain lines, or a mottled table or cloth; darker in the page's shadow.
                double outside = Math.Max(Math.Max(ox - x, x - (ox + pw)), Math.Max(oy - y, y - (oy + ph)));
                float shade = outside < shadow ? (float)(0.55 + 0.45 * outside / shadow) : 1f;
                float mottle = ScanImageOps.Smooth(field, cells, (float)x / w, (float)y / h);
                uint hash = (uint)(x * 73856093) ^ (uint)(y * 19349663) ^ (uint)seed;
                hash = (hash ^ (hash >> 13)) * 0x5bd1e995;
                float speck = ((hash >> 8) & 0xFF) / 255f - 0.5f;
                float grain = wood
                    ? (float)(1 + 0.07 * Math.Sin(y / grainPeriod + mottle * 25 + grainPhase) * Math.Sin(y / (grainPeriod * 7.3) + mottle * 9) + 0.03 * speck)
                    : 1 + 0.08f * (mottle - 0.5f) + 0.04f * speck;
                for (int c = 0; c < planes.Length; c++)
                {
                    result[c][i] = (byte)Math.Clamp(table[c] * grain * shade, 0, 255);
                }
            }
        });
        return PillowImageOps.FromBytes(result, h, w);
    }

    /// <summary>Glare: a soft bright patch where light reflects off the page, washing it toward white by up to <paramref name="amount"/>.</summary>
    public static ImageData Glare(ImageData image, double amount, int seed)
    {
        if (amount == 0)
        {
            return image;
        }

        var planes = PillowImageOps.Bytes(image);
        int w = image.Width, h = image.Height, side = Math.Min(w, h);
        var random = new Random(seed);
        double cx = w * random.NextDouble(), cy = h * random.NextDouble(), rx = side * (0.1 + 0.25 * random.NextDouble()), ry = rx * (0.4 + 0.6 * random.NextDouble());
        double turn = random.NextDouble() * Math.PI, cos = Math.Cos(turn), sin = Math.Sin(turn);
        Rows(w, h, y =>
        {
            for (int x = 0; x < w; x++)
            {
                double dx = x - cx, dy = y - cy, a = (dx * cos + dy * sin) / rx, b = (-dx * sin + dy * cos) / ry;
                double d2 = a * a + b * b;
                if (d2 > 9)
                {
                    continue;
                }

                float k = (float)(amount * Math.Exp(-d2));
                for (int c = 0; c < planes.Length; c++)
                {
                    ref byte v = ref planes[c][y * w + x];
                    v = (byte)Math.Clamp(v + (255 - v) * k + 0.5f, 0, 255);
                }
            }
        });
        return PillowImageOps.FromBytes(planes, h, w);
    }

    /// <summary>Vignetting: the corners darker than the middle (a camera lens), by up to <paramref name="amount"/>.</summary>
    public static ImageData Vignette(ImageData image, double amount, int seed)
    {
        if (amount == 0)
        {
            return image;
        }

        var planes = PillowImageOps.Bytes(image);
        int w = image.Width, h = image.Height;
        var random = new Random(seed);
        double cx = w * (0.4 + 0.2 * random.NextDouble()), cy = h * (0.4 + 0.2 * random.NextDouble());
        Rows(w, h, y =>
        {
            for (int x = 0; x < w; x++)
            {
                double dx = (x - cx) / (w * 0.5), dy = (y - cy) / (h * 0.5), d2 = (dx * dx + dy * dy) / 2;
                float f = (float)(1 - amount * Math.Min(1, d2 * d2 * 1.5));
                for (int c = 0; c < planes.Length; c++)
                {
                    ref byte v = ref planes[c][y * w + x];
                    v = (byte)Math.Clamp(v * f + 0.5f, 0, 255);
                }
            }
        });
        return PillowImageOps.FromBytes(planes, h, w);
    }

    /// <summary>Motion blur: a camera moved <paramref name="length"/> pixels while the photo was taken, in one direction.</summary>
    public static ImageData Motion(ImageData image, double length, int seed)
    {
        if (length == 0)
        {
            return image;
        }

        var planes = PillowImageOps.Bytes(image);
        int w = image.Width, h = image.Height;
        var random = new Random(seed);
        double angle = random.NextDouble() * Math.PI, ux = Math.Cos(angle), uy = Math.Sin(angle);
        int taps = Math.Max(2, (int)Math.Ceiling(length) + 1);
        var result = planes.Select(_ => new byte[w * h]).ToArray();
        Rows(w, h, y =>
        {
            for (int x = 0; x < w; x++)
            {
                for (int c = 0; c < planes.Length; c++)
                {
                    float total = 0;
                    for (int t = 0; t < taps; t++)
                    {
                        double s = length * ((double)t / (taps - 1) - 0.5);
                        int sx = Math.Clamp((int)Math.Round(x + ux * s), 0, w - 1), sy = Math.Clamp((int)Math.Round(y + uy * s), 0, h - 1);
                        total += planes[c][sy * w + sx];
                    }

                    result[c][y * w + x] = (byte)(total / taps + 0.5f);
                }
            }
        });
        return PillowImageOps.FromBytes(result, h, w);
    }

    /// <summary>Moiré: faint interference waves (a page photographed off a screen or a printed halftone), <paramref name="amount"/> strong.</summary>
    public static ImageData Moire(ImageData image, double amount, int seed)
    {
        if (amount == 0)
        {
            return image;
        }

        var planes = PillowImageOps.Bytes(image);
        int w = image.Width, h = image.Height;
        var random = new Random(seed);
        double f1 = 0.6 + 0.4 * random.NextDouble(), f2 = f1 * (1.02 + 0.05 * random.NextDouble()), a1 = random.NextDouble() * Math.PI, a2 = a1 + 0.05 + 0.1 * random.NextDouble();
        double c1 = Math.Cos(a1), s1 = Math.Sin(a1), c2 = Math.Cos(a2), s2 = Math.Sin(a2);
        Rows(w, h, y =>
        {
            for (int x = 0; x < w; x++)
            {
                double p = Math.Sin((x * c1 + y * s1) * f1) * Math.Sin((x * c2 + y * s2) * f2);
                for (int c = 0; c < planes.Length; c++)
                {
                    // Each colour its own phase: the rainbow of a screen's moiré.
                    double wave = planes.Length == 3 ? 0.5 * p + 0.5 * Math.Sin((x * c1 + y * s1) * (f1 - f2) + c * 2.1) : p;
                    ref byte v = ref planes[c][y * w + x];
                    v = (byte)Math.Clamp(v * (1 - amount * 0.3 * (1 + wave)) + 0.5, 0, 255);
                }
            }
        });
        return PillowImageOps.FromBytes(planes, h, w);
    }

    /// <summary>A hand's or a phone's shadow reaching in from one edge: a soft dark shape, up to <paramref name="amount"/> darker.</summary>
    public static ImageData HandShadow(ImageData image, double amount, int seed)
    {
        if (amount == 0)
        {
            return image;
        }

        var planes = PillowImageOps.Bytes(image);
        int w = image.Width, h = image.Height;
        var random = new Random(seed);
        int s = random.Next(4);
        int length = s % 2 == 0 ? w : h;
        double center = length * (0.2 + 0.6 * random.NextDouble()), width = length * (0.15 + 0.25 * random.NextDouble());
        double reach = (s % 2 == 0 ? h : w) * (0.15 + 0.35 * random.NextDouble()), soft = Math.Min(w, h) * (0.05 + 0.08 * random.NextDouble());
        Rows(w, h, y =>
        {
            for (int x = 0; x < w; x++)
            {
                var (inside, along) = FromSide(s, x, y, w, h);

                // A rounded finger-like shape: widest at the edge, narrowing to its tip.
                double t = inside / reach, half = width / 2 * Math.Sqrt(Math.Max(0, 1 - t * t));
                double beyond = Math.Max(Math.Abs(along - center) - half, inside - reach);
                if (beyond > soft)
                {
                    continue;
                }

                // A soft penumbra on both sides of the shape's outline, the full shade only well inside it.
                double t2 = Math.Clamp(0.5 - beyond / (2 * soft), 0, 1);
                float k = (float)(amount * t2 * t2 * (3 - 2 * t2));
                for (int c = 0; c < planes.Length; c++)
                {
                    ref byte v = ref planes[c][y * w + x];
                    v = (byte)Math.Clamp(v * (1 - k) + 0.5f, 0, 255);
                }
            }
        });
        return PillowImageOps.FromBytes(planes, h, w);
    }

    /// <summary>Fingerprints: faint greasy ridges in a thumb-sized oval, darkening the page a little.</summary>
    public static ImageData Fingerprint(ImageData image, int count, int seed)
    {
        if (count == 0)
        {
            return image;
        }

        var planes = PillowImageOps.Bytes(image);
        int w = image.Width, h = image.Height, side = Math.Min(w, h);
        var random = new Random(seed);
        var grease = Tint(planes.Length, 0.86f, 0.84f, 0.8f);
        for (int k = 0; k < count; k++)
        {
            double cx = w * random.NextDouble(), cy = h * random.NextDouble(), rx = side * (0.025 + 0.015 * random.NextDouble()), ry = rx * 1.35;
            double turn = random.NextDouble() * Math.PI, cos = Math.Cos(turn), sin = Math.Sin(turn), pitch = 2.5 + random.NextDouble(), phase = random.NextDouble() * 6.3;
            float strength = (float)(0.3 + 0.4 * random.NextDouble());
            int reach = (int)ry + 2;
            MultiplyIn(planes, w, h, (int)cx - reach, (int)cy - reach, (int)cx + reach, (int)cy + reach, grease, (x, y) =>
            {
                double dx = x - cx, dy = y - cy, a = (dx * cos + dy * sin) / rx, b = (-dx * sin + dy * cos) / ry, d = Math.Sqrt(a * a + b * b);
                if (d >= 1)
                {
                    return 0;
                }

                // Ridges: loops round the core, wavering.
                double ridge = Math.Sin(d * rx / pitch * 6.283 + Math.Sin(a * 3 + phase) * 0.8);
                return ridge > 0.2 ? strength * (float)(1 - d * d) : 0;
            });
        }

        return PillowImageOps.FromBytes(planes, h, w);
    }

    /// <summary>Ruled paper: faint blue lines across the page (and now and then a red margin line) under the ink, <paramref name="amount"/> strong.</summary>
    public static ImageData Lined(ImageData image, double amount, int seed) => Ruled(image, amount, seed, grid: false);

    /// <summary>Graph paper: a faint grid under the ink, <paramref name="amount"/> strong.</summary>
    public static ImageData Grid(ImageData image, double amount, int seed) => Ruled(image, amount, seed, grid: true);

    private static ImageData Ruled(ImageData image, double amount, int seed, bool grid)
    {
        if (amount == 0)
        {
            return image;
        }

        var planes = PillowImageOps.Bytes(image);
        int w = image.Width, h = image.Height;
        var random = new Random(seed);
        double spacing = grid ? 14 + 16 * random.NextDouble() : 24 + 18 * random.NextDouble(), offset = spacing * random.NextDouble();
        var blue = random.NextDouble() < 0.7 ? Tint(planes.Length, 0.75f, 0.85f, 1f) : Tint(planes.Length, 0.8f, 0.92f, 0.85f);
        var red = Tint(planes.Length, 1f, 0.7f, 0.72f);
        double margin = !grid && random.NextDouble() < 0.5 ? w * (random.NextDouble() < 0.5 ? 0.1 : 0.9) : -10;
        float a = (float)amount;
        Rows(w, h, y =>
        {
            double ry = (y - offset) % spacing;
            bool row = ry >= 0 && ry < 1.2;
            for (int x = 0; x < w; x++)
            {
                double rx = (x - offset) % spacing;
                bool line = row || (grid && rx >= 0 && rx < 1.2);
                bool edge = Math.Abs(x - margin) < 1;
                if (!line && !edge)
                {
                    continue;
                }

                var color = edge ? red : blue;
                for (int c = 0; c < planes.Length; c++)
                {
                    ref byte v = ref planes[c][y * w + x];
                    v = (byte)Math.Clamp(v * (1 - a * (1 - color[c])) + 0.5f, 0, 255);
                }
            }
        });
        return PillowImageOps.FromBytes(planes, h, w);
    }

    /// <summary>A security background (guilloche): fine wavy lines in a light colour over the paper, under the ink, <paramref name="amount"/> strong.</summary>
    public static ImageData Guilloche(ImageData image, double amount, int seed)
    {
        if (amount == 0)
        {
            return image;
        }

        var planes = PillowImageOps.Bytes(image);
        int w = image.Width, h = image.Height;
        var random = new Random(seed);
        (float, float, float)[] inks = [(0.85f, 0.95f, 0.85f), (0.85f, 0.88f, 1f), (1f, 0.88f, 0.9f), (0.95f, 0.92f, 0.82f)];
        var tint = inks[random.Next(inks.Length)];
        var color = Tint(planes.Length, tint.Item1, tint.Item2, tint.Item3);
        double f1 = 0.02 + 0.03 * random.NextDouble(), f2 = 0.01 + 0.02 * random.NextDouble(), amp = 10 + 30 * random.NextDouble(), pitch = 5 + 5 * random.NextDouble();
        float a = (float)amount;
        Rows(w, h, y =>
        {
            for (int x = 0; x < w; x++)
            {
                double wave1 = Math.Sin((y + amp * Math.Sin(x * f1)) / pitch * 6.283), wave2 = Math.Sin((x + amp * Math.Sin(y * f2)) / pitch * 6.283);
                float k = (float)(Math.Max(0, Math.Max(wave1, wave2) - 0.8) / 0.2) * a;
                if (k <= 0)
                {
                    continue;
                }

                for (int c = 0; c < planes.Length; c++)
                {
                    ref byte v = ref planes[c][y * w + x];
                    v = (byte)Math.Clamp(v * (1 - k * (1 - color[c])) + 0.5f, 0, 255);
                }
            }
        });
        return PillowImageOps.FromBytes(planes, h, w);
    }

    /// <summary>Recycled paper: short dark and coloured fibres and a fine grain in the paper, <paramref name="amount"/> strong.</summary>
    public static ImageData Recycled(ImageData image, double amount, int seed)
    {
        if (amount == 0)
        {
            return image;
        }

        var planes = PillowImageOps.Bytes(image);
        int w = image.Width, h = image.Height;
        var random = new Random(seed);
        var paper = ScanImageOps.PaperLevels(planes, w * h);
        int fibres = (int)(w * (double)h / 1e6 * 400 * amount);
        for (int k = 0; k < fibres; k++)
        {
            double x0 = w * random.NextDouble(), y0 = h * random.NextDouble(), angle = random.NextDouble() * Math.PI, length = 3 + 12 * random.NextDouble();
            float[] color = random.NextDouble() < 0.7 ? [.. paper.Select(p => p * (float)(0.55 + 0.3 * random.NextDouble()))]
                : Tint(planes.Length, 120 + random.Next(80), 100 + random.Next(80), 140 + random.Next(80));
            Stroke(planes, w, h, HandLine(x0, y0, x0 + Math.Cos(angle) * length, y0 + Math.Sin(angle) * length, random, 0.8), 0.7, color, (float)(0.4 + 0.4 * random.NextDouble()));
        }

        int grain = (int)(6 * amount);
        if (grain > 0)
        {
            Rows(w, h, y =>
            {
                ulong z = ((ulong)y * 0x9E3779B97F4A7C15UL ^ (ulong)(uint)seed) | 1;
                for (int x = 0; x < w; x++)
                {
                    z ^= z << 13;
                    z ^= z >> 7;
                    z ^= z << 17;
                    int d = (int)(z % (ulong)(2 * grain + 1)) - grain;
                    for (int c = 0; c < planes.Length; c++)
                    {
                        ref byte v = ref planes[c][y * w + x];
                        v = (byte)Math.Clamp(v + d, 0, 255);
                    }
                }
            });
        }

        return PillowImageOps.FromBytes(planes, h, w);
    }
}
