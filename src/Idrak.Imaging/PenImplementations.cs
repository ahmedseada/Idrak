// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Buffers;

namespace Idrak.Imaging;

/// <summary>
/// An outline built contour by contour in pooled buffers (pixels, y down), every contour wound the same way so that
/// overlapping ones fill once (the coverage rasterizer clamps the winding count); <see cref="ToOutline"/> copies it out
/// at its exact length. Reused stroke after stroke; disposed once.
/// </summary>
internal sealed class OutlineBuilder : IDisposable
{
    private OutlineVerb[] _verbs = ArrayPool<OutlineVerb>.Shared.Rent(256);
    private float[] _points = ArrayPool<float>.Shared.Rent(1024);
    private int _verbCount, _pointCount;

    public bool IsEmpty => _verbCount == 0;

    public void Clear() => (_verbCount, _pointCount) = (0, 0);

    /// <summary>Adds the closed polygon of x, y pairs (pixels), wound the builder's way; a polygon of no area adds nothing.</summary>
    public void Polygon(ReadOnlySpan<float> xy)
    {
        int n = xy.Length / 2;
        if (n < 3)
        {
            return;
        }

        double area = 0;
        for (int i = 0; i < n; i++)
        {
            int j = i + 1 == n ? 0 : i + 1;
            area += (double)xy[2 * i] * xy[2 * j + 1] - (double)xy[2 * j] * xy[2 * i + 1];
        }

        if (area == 0)
        {
            return;
        }

        Reserve(n + 1, 2 * n);
        // Every contour negative in pixel space (the rasterizer's y flip turns them all the other way alike).
        bool reverse = area > 0;
        for (int k = 0; k < n; k++)
        {
            int i = reverse ? n - 1 - k : k;
            _verbs[_verbCount++] = k == 0 ? OutlineVerb.Move : OutlineVerb.Line;
            _points[_pointCount++] = xy[2 * i];
            _points[_pointCount++] = -xy[2 * i + 1];               // y up, as outlines are: Fill flips it back
        }

        _verbs[_verbCount++] = OutlineVerb.Close;
    }

    /// <summary>The outline so far, copied out (a <see cref="GlyphOutline"/> keeps arrays of its exact length).</summary>
    public GlyphOutline ToOutline() => _verbCount == 0 ? GlyphOutline.Empty : new GlyphOutline(_verbs.AsSpan(0, _verbCount).ToArray(), _points.AsSpan(0, _pointCount).ToArray());

    public void Dispose()
    {
        if (_verbs.Length > 0)
        {
            ArrayPool<OutlineVerb>.Shared.Return(_verbs);
            ArrayPool<float>.Shared.Return(_points);
            (_verbs, _points) = ([], []);
        }
    }

    private void Reserve(int verbs, int points)
    {
        ObjectDisposedException.ThrowIf(_verbs.Length == 0, this);
        if (_verbCount + verbs > _verbs.Length)
        {
            var bigger = ArrayPool<OutlineVerb>.Shared.Rent(Math.Max(_verbs.Length * 2, _verbCount + verbs));
            _verbs.AsSpan(0, _verbCount).CopyTo(bigger);
            ArrayPool<OutlineVerb>.Shared.Return(_verbs);
            _verbs = bigger;
        }

        if (_pointCount + points > _points.Length)
        {
            var bigger = ArrayPool<float>.Shared.Rent(Math.Max(_points.Length * 2, _pointCount + points));
            _points.AsSpan(0, _pointCount).CopyTo(bigger);
            ArrayPool<float>.Shared.Return(_points);
            _points = bigger;
        }
    }
}

/// <summary>
/// Strokes to outlines, the one way every pen's stroke is drawn: each segment a quadrilateral between its two points'
/// widths, each join filled (a wedge on a gentle turn, a disc on a sharp one), the ends of an open piece round, every
/// contour wound alike; a dash pattern cuts the stroke into pieces (an "on" of 0 is a dot).
/// </summary>
internal static class StrokeOutline
{
    // The most a disc's polygon strays inside the true circle, in pixels.
    private const float Flatness = 0.1f;

    // A join wider than this turn (its cosine) is filled with a disc; a gentler one with two wedges.
    private const float SharpTurn = 0.94f;

    // The unit circle with n points, made once per n (the same values on every machine: PortableMath).
    private static readonly float[]?[] Circles = new float[65][];

    /// <summary>Adds the stroke through <paramref name="points"/> to <paramref name="outline"/>, cut by <paramref name="dash"/> (on, off, ... in units of <paramref name="dashScale"/> pixels; empty for solid).</summary>
    public static void Add(OutlineBuilder outline, ReadOnlySpan<PenPoint> points, bool closed, ReadOnlySpan<float> dash, float dashScale)
    {
        if (points.IsEmpty)
        {
            return;
        }

        float period = 0;
        foreach (float d in dash)
        {
            period += d * dashScale;
        }

        if (dash.IsEmpty || period < 0.5f)
        {
            Piece(outline, points, closed);
            return;
        }

        // Walk the stroke (closed: back to its first point) cutting it where the pattern turns on and off.
        int n = points.Length, count = closed ? n + 1 : n;
        var piece = ArrayPool<PenPoint>.Shared.Rent(count + 2);
        try
        {
            int m = 0, k = 0;
            bool on = true;
            float left = dash[0] * dashScale;
            piece[m++] = points[0];
            for (int i = 0; i + 1 < count; i++)
            {
                var a = points[i];
                var b = points[i + 1 == n ? 0 : i + 1];
                float length = MathF.Sqrt((b.X - a.X) * (b.X - a.X) + (b.Y - a.Y) * (b.Y - a.Y));
                float at = 0;
                while (length - at > left)
                {
                    at += left;
                    var p = Lerp(a, b, length > 0 ? at / length : 0);
                    if (on)
                    {
                        piece[m++] = p;
                        Piece(outline, piece.AsSpan(0, m), closed: false);
                        m = 0;
                    }
                    else
                    {
                        if (m + 1 >= piece.Length)
                        {
                            m = 0;
                        }

                        piece[m++] = p;
                    }

                    on = !on;
                    k = (k + 1) % dash.Length;
                    left = dash[k] * dashScale;
                }

                left -= length - at;
                if (on)
                {
                    if (m + 1 >= piece.Length)
                    {
                        Piece(outline, piece.AsSpan(0, m), closed: false);
                        piece[0] = piece[m - 1];
                        m = 1;
                    }

                    piece[m++] = b;
                }
            }

            if (on && m > 0)
            {
                Piece(outline, piece.AsSpan(0, m), closed: false);
            }
        }
        finally
        {
            ArrayPool<PenPoint>.Shared.Return(piece);
        }
    }

    // One unbroken piece: its segments, joins and (open) round ends.
    private static void Piece(OutlineBuilder outline, ReadOnlySpan<PenPoint> p, bool closed)
    {
        int n = p.Length;
        if (n == 1 || Length(p, closed) < 1e-4f)
        {
            Disc(outline, p[0].X, p[0].Y, p[0].Width / 2);
            return;
        }

        Span<float> quad = stackalloc float[8];
        int segments = closed ? n : n - 1;
        for (int i = 0; i < segments; i++)
        {
            var a = p[i];
            var b = p[i + 1 == n ? 0 : i + 1];
            float dx = b.X - a.X, dy = b.Y - a.Y, length = MathF.Sqrt(dx * dx + dy * dy);
            if (length < 1e-6f)
            {
                continue;
            }

            float nx = -dy / length, ny = dx / length, ha = a.Width / 2, hb = b.Width / 2;
            quad[0] = a.X + nx * ha;
            quad[1] = a.Y + ny * ha;
            quad[2] = b.X + nx * hb;
            quad[3] = b.Y + ny * hb;
            quad[4] = b.X - nx * hb;
            quad[5] = b.Y - ny * hb;
            quad[6] = a.X - nx * ha;
            quad[7] = a.Y - ny * ha;
            outline.Polygon(quad);
        }

        // Joins: where one segment meets the next (every point of a closed piece).
        int first = closed ? 0 : 1, last = closed ? n - 1 : n - 2;
        Span<float> wedge = stackalloc float[6];
        for (int i = first; i <= last; i++)
        {
            var at = p[i];
            int before = Neighbour(p, i, -1, closed), after = Neighbour(p, i, 1, closed);
            if (before < 0 || after < 0)
            {
                continue;
            }

            float ax = at.X - p[before].X, ay = at.Y - p[before].Y, bx = p[after].X - at.X, by = p[after].Y - at.Y;
            float la = MathF.Sqrt(ax * ax + ay * ay), lb = MathF.Sqrt(bx * bx + by * by);
            float h = at.Width / 2;
            if ((ax * bx + ay * by) / (la * lb) < SharpTurn)
            {
                Disc(outline, at.X, at.Y, h);
                continue;
            }

            // A gentle turn: the gap on its outer side filled by a wedge (and its twin on the inner side, inside the
            // quadrilaterals already), so the join is a bevel there.
            float n0x = -ay / la * h, n0y = ax / la * h, n1x = -by / lb * h, n1y = bx / lb * h;
            for (int side = -1; side <= 1; side += 2)
            {
                wedge[0] = at.X;
                wedge[1] = at.Y;
                wedge[2] = at.X + side * n0x;
                wedge[3] = at.Y + side * n0y;
                wedge[4] = at.X + side * n1x;
                wedge[5] = at.Y + side * n1y;
                outline.Polygon(wedge);
            }
        }

        if (!closed)
        {
            Disc(outline, p[0].X, p[0].Y, p[0].Width / 2);
            Disc(outline, p[n - 1].X, p[n - 1].Y, p[n - 1].Width / 2);
        }
    }

    // The next point from `i` in `step`'s direction that is not on it (a closed piece wraps); -1 when none.
    private static int Neighbour(ReadOnlySpan<PenPoint> p, int i, int step, bool closed)
    {
        int n = p.Length;
        for (int k = 1; k < n; k++)
        {
            int j = i + step * k;
            if (closed)
            {
                j = ((j % n) + n) % n;
            }
            else if (j < 0 || j >= n)
            {
                return -1;
            }

            if (MathF.Abs(p[j].X - p[i].X) + MathF.Abs(p[j].Y - p[i].Y) > 1e-5f)
            {
                return j;
            }
        }

        return -1;
    }

    private static float Length(ReadOnlySpan<PenPoint> p, bool closed)
    {
        float total = 0;
        int segments = closed ? p.Length : p.Length - 1;
        for (int i = 0; i < segments; i++)
        {
            var a = p[i];
            var b = p[i + 1 == p.Length ? 0 : i + 1];
            total += MathF.Sqrt((b.X - a.X) * (b.X - a.X) + (b.Y - a.Y) * (b.Y - a.Y));
        }

        return total;
    }

    private static PenPoint Lerp(in PenPoint a, in PenPoint b, float t) =>
        new(a.X + (b.X - a.X) * t, a.Y + (b.Y - a.Y) * t, a.Width + (b.Width - a.Width) * t, a.Ink + (b.Ink - a.Ink) * t);

    // A disc of radius r at (x, y), as a polygon within the flatness of the true circle.
    private static void Disc(OutlineBuilder outline, float x, float y, float r)
    {
        if (!(r > 0))
        {
            return;
        }

        int n = Math.Clamp((int)MathF.Ceiling(MathF.PI * MathF.Sqrt(r / (2 * Flatness))), 8, Circles.Length - 1);
        var unit = Circle(n);
        Span<float> disc = stackalloc float[2 * (Circles.Length - 1)];
        for (int k = 0; k < n; k++)
        {
            disc[2 * k] = x + r * unit[2 * k];
            disc[2 * k + 1] = y + r * unit[2 * k + 1];
        }

        outline.Polygon(disc[..(2 * n)]);
    }

    private static float[] Circle(int n)
    {
        if (Circles[n] is { } made)
        {
            return made;
        }

        var unit = new float[2 * n];
        for (int k = 0; k < n; k++)
        {
            double a = 2 * Math.PI * k / n;
            unit[2 * k] = (float)PortableMath.Cos(a);
            unit[2 * k + 1] = (float)PortableMath.Sin(a);
        }

        Circles[n] = unit;                                          // made alike on any thread: a race writes the same values
        return unit;
    }
}

/// <summary>"ideal": the path exactly, each point at its width and ink.</summary>
internal sealed class IdealPen : IPen
{
    public string Name => Pens.Ideal;

    public string Summary => "the path exactly, at its width (no wobble, no pressure)";

    public IReadOnlySet<string> Keys => OptionKeys.None;

    public PenSettings Read(IReadOnlyDictionary<string, string> options)
    {
        ArgumentNullException.ThrowIfNull(options);
        foreach (string key in options.Keys)
        {
            throw new ArgumentException($"The pen ideal takes no options (given '{key}').");
        }

        return PenSettings.None;
    }

    public float Stray(PenSettings settings, float width) => 0;

    public void Stroke(ReadOnlySpan<PenPoint> path, bool closed, PenSettings settings, int seed, PenStroke stroke)
    {
        ArgumentNullException.ThrowIfNull(stroke);
        stroke.Clear();
        stroke.Closed = closed;
        foreach (var point in path)
        {
            stroke.Add(point);
        }
    }
}

/// <summary>
/// "ballpoint": a hand's ballpoint. Along the path it drifts in a slow wobble (pen-wobble, stroke widths), shakes a
/// little (pen-jitter, stroke widths), presses harder and lighter (pen-pressure, a share of the width and ink), starts
/// and stops a little past or short of its ends (pen-overshoot, stroke widths) and drifts sideways along the way
/// (pen-tilt, stroke widths at its end); each a range drawn per stroke from its seed.
/// </summary>
internal sealed class BallpointPen : IPen
{
    private static readonly OptionKeys Own = new("pen-wobble", "pen-jitter", "pen-pressure", "pen-overshoot", "pen-tilt");

    private static readonly Settings Defaults = new(PageOptions.None, (0.15f, 0.5f), (0.03f, 0.1f), (0.1f, 0.3f), (0f, 1.5f), (0f, 1f));

    public string Name => "ballpoint";

    public string Summary => "a hand's ballpoint: a slow wobble, a little jitter, pressure along the stroke, an overshoot or a short stop at its ends, a slight drift (pen-wobble, pen-jitter, pen-pressure, pen-overshoot, pen-tilt: ranges)";

    public IReadOnlySet<string> Keys => Own;

    public PenSettings Read(IReadOnlyDictionary<string, string> options)
    {
        ArgumentNullException.ThrowIfNull(options);
        var (wobble, jitter, pressure, overshoot, tilt) = (Defaults.Wobble, Defaults.Jitter, Defaults.Pressure, Defaults.Overshoot, Defaults.Tilt);
        foreach (var (raw, value) in options)
        {
            string key = raw.ToLowerInvariant();
            string what = $"Page setting {key}";
            switch (key)
            {
                case "pen-wobble":
                    wobble = PageOptions.Range(what, value, 0, 10);
                    break;
                case "pen-jitter":
                    jitter = PageOptions.Range(what, value, 0, 5);
                    break;
                case "pen-pressure":
                    pressure = PageOptions.Range(what, value, 0, 0.9f);
                    break;
                case "pen-overshoot":
                    overshoot = PageOptions.Range(what, value, 0, 20);
                    break;
                case "pen-tilt":
                    tilt = PageOptions.Range(what, value, 0, 20);
                    break;
                default:
                    throw new ArgumentException($"The pen ballpoint does not take '{raw}'; it takes: {Own}.");
            }
        }

        return new Settings(options, wobble, jitter, pressure, overshoot, tilt);
    }

    public float Stray(PenSettings settings, float width)
    {
        var s = settings as Settings ?? Defaults;
        return (s.Wobble.Max + s.Jitter.Max + s.Tilt.Max + s.Overshoot.Max) * width;
    }

    public void Stroke(ReadOnlySpan<PenPoint> path, bool closed, PenSettings settings, int seed, PenStroke stroke)
    {
        ArgumentNullException.ThrowIfNull(stroke);
        var s = settings as Settings ?? Defaults;
        stroke.Clear();
        stroke.Closed = closed;
        if (path.IsEmpty)
        {
            return;
        }

        int n = path.Length, segments = closed ? n : n - 1;
        double total = 0;
        float meanWidth = 0;
        for (int i = 0; i < n; i++)
        {
            meanWidth += path[i].Width / n;
        }

        for (int i = 0; i < segments; i++)
        {
            total += Distance(path[i], path[i + 1 == n ? 0 : i + 1]);
        }

        var random = new Random(seed);
        // The stroke's own draws, always the same count: the wobble, the pressure, the two ends and the drift.
        double wobble = Draw(s.Wobble, random) * meanWidth, wavelength = 120 + 240 * random.NextDouble(), phase = 2 * Math.PI * random.NextDouble();
        double pressure = Draw(s.Pressure, random), pressureWave = 60 + 180 * random.NextDouble(), pressurePhase = 2 * Math.PI * random.NextDouble();
        double startEnd = Draw(s.Overshoot, random) * meanWidth * (random.NextDouble() < 0.6 ? 1 : -1);
        double finish = Draw(s.Overshoot, random) * meanWidth * (random.NextDouble() < 0.6 ? 1 : -1);
        double drift = Draw(s.Tilt, random) * meanWidth * (random.NextDouble() < 0.5 ? 1 : -1);
        double jitter = Draw(s.Jitter, random) * meanWidth;
        if (total < 1e-3)
        {
            stroke.Add(path[0]);
            return;
        }

        // Samples along the path about a stroke width apart (1 to 4 pixels).
        double step = Math.Clamp(meanWidth, 1.0, 4.0);
        if (total < 6 * step)
        {
            startEnd = finish = 0;                                  // too short a stroke to land past or short of its ends
        }

        if (closed)
        {
            // Around a closed path the wobble comes back to where it began, and there is no drift and no end.
            wavelength = total / Math.Max(1, Math.Round(total / wavelength));
            pressureWave = total / Math.Max(1, Math.Round(total / pressureWave));
            drift = startEnd = finish = 0;
        }

        int samples = Math.Max(2, (int)Math.Ceiling(total / step) + (closed ? 0 : 1));
        double spacing = closed ? total / samples : total / (samples - 1);
        // An end cut short leaves at least most of the stroke.
        double skipStart = startEnd < 0 ? Math.Min(-startEnd, 0.3 * total) : 0, skipEnd = finish < 0 ? Math.Min(-finish, 0.3 * total) : 0;
        int segment = 0;
        double segmentStart = 0, segmentLength = Distance(path[0], path[1 == n ? 0 : 1]);
        double shake = 0;
        bool first = true;
        PenPoint last = default;
        for (int k = 0; k < samples; k++)
        {
            double at = Math.Min(k * spacing, total);
            while (at > segmentStart + segmentLength && segment < segments - 1)
            {
                segmentStart += segmentLength;
                segment++;
                segmentLength = Distance(path[segment], path[segment + 1 == n ? 0 : segment + 1]);
            }

            shake = 0.6 * shake + 0.4 * (2 * random.NextDouble() - 1) * jitter;      // drawn for every sample
            if (at < skipStart || at > total - skipEnd)
            {
                continue;
            }

            var a = path[segment];
            var b = path[segment + 1 == n ? 0 : segment + 1];
            double t = segmentLength > 0 ? Math.Clamp((at - segmentStart) / segmentLength, 0, 1) : 0;
            double tx = segmentLength > 0 ? (b.X - a.X) / segmentLength : 1, ty = segmentLength > 0 ? (b.Y - a.Y) / segmentLength : 0;
            double offset = wobble * PortableMath.Sin(2 * Math.PI * at / wavelength + phase) + shake + drift * at / total;
            double press = PortableMath.Sin(2 * Math.PI * at / pressureWave + pressurePhase);
            double x = a.X + (b.X - a.X) * t - ty * offset, y = a.Y + (b.Y - a.Y) * t + tx * offset;
            float width = (float)Math.Max(0.05 * (a.Width + (b.Width - a.Width) * t), (a.Width + (b.Width - a.Width) * t) * (1 + pressure * press));
            float ink = (float)Math.Clamp((a.Ink + (b.Ink - a.Ink) * t) * (1 - 0.5 * pressure + 0.5 * pressure * press), 0.3, 1);
            var point = new PenPoint((float)x, (float)y, width, ink);
            if (first && startEnd > 0)
            {
                // Started a little before the path: the pen lands back along it.
                stroke.Add(point with { X = (float)(x - tx * startEnd), Y = (float)(y - ty * startEnd) });
            }

            first = false;
            stroke.Add(point);
            last = point;
            if (k == samples - 1 && finish > 0)
            {
                stroke.Add(last with { X = (float)(x + tx * finish), Y = (float)(y + ty * finish) });
            }
        }

    }

    private static double Distance(in PenPoint a, in PenPoint b) => Math.Sqrt((double)(b.X - a.X) * (b.X - a.X) + (double)(b.Y - a.Y) * (b.Y - a.Y));

    private static double Draw((float Min, float Max) range, Random random) => range.Min + random.NextDouble() * (range.Max - range.Min);

    // The ranges read from a page's options, drawn per stroke.
    private sealed class Settings(IReadOnlyDictionary<string, string> given, (float Min, float Max) wobble, (float Min, float Max) jitter, (float Min, float Max) pressure,
        (float Min, float Max) overshoot, (float Min, float Max) tilt) : PenSettings(given)
    {
        public (float Min, float Max) Wobble { get; } = wobble;

        public (float Min, float Max) Jitter { get; } = jitter;

        public (float Min, float Max) Pressure { get; } = pressure;

        public (float Min, float Max) Overshoot { get; } = overshoot;

        public (float Min, float Max) Tilt { get; } = tilt;
    }
}
