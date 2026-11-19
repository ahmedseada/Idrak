// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Buffers;

namespace Idrak.Imaging;

/// <summary>
/// An outline built in surface pixels (y down), kept with y turned up as <see cref="GlyphOutline"/>s are, so the
/// rasterizer's fill at scale 1 puts each point back where it was; its box in pixels follows the points.
/// </summary>
internal sealed class PixelOutline
{
    private OutlineVerb[] _verbs = new OutlineVerb[16];
    private float[] _points = new float[32];
    private int _verbCount, _pointCount;
    private bool _open;

    public int VerbCount => _verbCount;

    public float MinX { get; private set; } = float.PositiveInfinity;

    public float MinY { get; private set; } = float.PositiveInfinity;

    public float MaxX { get; private set; } = float.NegativeInfinity;

    public float MaxY { get; private set; } = float.NegativeInfinity;

    public bool IsEmpty => _verbCount == 0;

    public void Move(float x, float y)
    {
        Close();
        Add(OutlineVerb.Move, 2);
        Point(x, y);
        _open = true;
    }

    public void Line(float x, float y)
    {
        Add(OutlineVerb.Line, 2);
        Point(x, y);
    }

    public void Quadratic(float x1, float y1, float x, float y)
    {
        Add(OutlineVerb.Quadratic, 4);
        Point(x1, y1);
        Point(x, y);
    }

    public void Cubic(float x1, float y1, float x2, float y2, float x, float y)
    {
        Add(OutlineVerb.Cubic, 6);
        Point(x1, y1);
        Point(x2, y2);
        Point(x, y);
    }

    public void Close()
    {
        if (_open)
        {
            Add(OutlineVerb.Close, 0);
            _open = false;
        }
    }

    /// <summary>A closed polygon of x, y pairs.</summary>
    public void Polygon(ReadOnlySpan<float> xy)
    {
        if (xy.Length < 6)
        {
            return;
        }

        Move(xy[0], xy[1]);
        for (int i = 2; i < xy.Length; i += 2)
        {
            Line(xy[i], xy[i + 1]);
        }

        Close();
    }

    /// <summary>The outline (y up), its arrays of their exact length; empty when nothing was added.</summary>
    public GlyphOutline Build()
    {
        Close();
        return _verbCount == 0 ? GlyphOutline.Empty : new GlyphOutline(_verbs.AsSpan(0, _verbCount).ToArray(), _points.AsSpan(0, _pointCount).ToArray());
    }

    private void Add(OutlineVerb verb, int points)
    {
        if (_verbCount == _verbs.Length)
        {
            Array.Resize(ref _verbs, _verbs.Length * 2);
        }

        if (_pointCount + points > _points.Length)
        {
            Array.Resize(ref _points, Math.Max(_points.Length * 2, _pointCount + points));
        }

        _verbs[_verbCount++] = verb;
    }

    private void Point(float x, float y)
    {
        _points[_pointCount++] = x;
        _points[_pointCount++] = -y;
        if (float.IsFinite(x) && float.IsFinite(y))
        {
            (MinX, MaxX, MinY, MaxY) = (MathF.Min(MinX, x), MathF.Max(MaxX, x), MathF.Min(MinY, y), MathF.Max(MaxY, y));
        }
    }
}

/// <summary>
/// A document stroke turned into an outline the way PDF readers stroke (ISO 32000-2, 8.5.3.2): a polyline's two sides
/// offset by half its width (each point's own, as a pen gives it), joined at its turns (a miter up to the miter limit, else
/// a bevel; or round; or a bevel), its open ends capped (butt, round or square); a closed piece's sides as two loops, the
/// inner one wound back; the inner side of a turn passing through the turn's point, so every turn stays inside the stroke
/// under the nonzero rule. A dash pattern cuts the polyline into open pieces first (an "on" of 0 a dot when its caps give
/// one). Curves arrive flattened; a round join's, cap's or dot's polygon strays at most the stroke's flatness inside the
/// true circle (<see cref="DocumentDrawOptions.StrokeFlatness"/>, in pixels).
/// </summary>
internal static class DocumentStroker
{
    /// <summary>Cuts <paramref name="points"/> (x, y pairs; closed when <paramref name="closed"/>) by the dash pattern (pixels), calling <paramref name="piece"/> with each piece (open), or once with the whole when it has no dash.</summary>
    public static void Dash(ReadOnlySpan<float> points, bool closed, ReadOnlySpan<float> dash, float phase, PieceHandler piece)
    {
        int n = points.Length / 2;
        if (n == 0)
        {
            return;
        }

        float period = 0;
        foreach (float d in dash)
        {
            period += MathF.Max(0f, d);
        }

        if (dash.IsEmpty || !(period > 1e-3f) || !float.IsFinite(period))
        {
            piece(points, closed, 0f, 0f);
            return;
        }

        // Where in the pattern the stroke starts.
        int k = 0;
        bool on = true;
        float left = MathF.Max(0f, dash[0]);
        float skip = phase % period;
        if (skip < 0)
        {
            skip += period;
        }

        while (skip > 0)
        {
            if (skip >= left)
            {
                skip -= left;
                k = (k + 1) % dash.Length;
                on = !on;
                left = MathF.Max(0f, dash[k]);
            }
            else
            {
                left -= skip;
                skip = 0;
            }
        }

        int segments = closed ? n : n - 1;
        var buffer = ArrayPool<float>.Shared.Rent(2 * (n + 2));
        try
        {
            int m = 0;
            float dirX = 1, dirY = 0;
            if (on)
            {
                buffer[m++] = points[0];
                buffer[m++] = points[1];
            }

            for (int i = 0; i < segments; i++)
            {
                int j = i + 1 == n ? 0 : i + 1;
                float ax = points[2 * i], ay = points[2 * i + 1], bx = points[2 * j], by = points[2 * j + 1];
                float length = MathF.Sqrt((bx - ax) * (bx - ax) + (by - ay) * (by - ay));
                if (length > 0)
                {
                    (dirX, dirY) = ((bx - ax) / length, (by - ay) / length);
                }

                float at = 0;
                int guard = 0;
                while (length - at >= left && guard++ < 1_000_000)
                {
                    at += left;
                    float t = length > 0 ? at / length : 0;
                    float px = ax + (bx - ax) * t, py = ay + (by - ay) * t;
                    if (on)
                    {
                        buffer[m++] = px;
                        buffer[m++] = py;
                        piece(buffer.AsSpan(0, m), false, dirX, dirY);
                        m = 0;
                    }
                    else
                    {
                        m = 0;
                        buffer[m++] = px;
                        buffer[m++] = py;
                    }

                    on = !on;
                    k = (k + 1) % dash.Length;
                    left = MathF.Max(0f, dash[k]);
                }

                left -= length - at;
                if (on)
                {
                    if (m + 2 > buffer.Length)
                    {
                        piece(buffer.AsSpan(0, m), false, dirX, dirY);
                        (buffer[0], buffer[1]) = (buffer[m - 2], buffer[m - 1]);
                        m = 2;
                    }

                    buffer[m++] = bx;
                    buffer[m++] = by;
                }
            }

            if (on && m >= 2)
            {
                piece(buffer.AsSpan(0, m), false, dirX, dirY);
            }
        }
        finally
        {
            ArrayPool<float>.Shared.Return(buffer);
        }
    }

    /// <summary>What a dashed stroke's pieces are given to: the piece's points, whether it is closed, and the direction it ran in (for a piece of no length).</summary>
    public delegate void PieceHandler(ReadOnlySpan<float> points, bool closed, float directionX, float directionY);

    /// <summary>
    /// Adds the stroke through <paramref name="points"/> (each at its own width) to <paramref name="outline"/>; a piece of
    /// no length is a dot by its caps (a disc for round, a square along (<paramref name="directionX"/>,
    /// <paramref name="directionY"/>) for square, nothing for butt). Round joins, caps and dots are polygons within
    /// <paramref name="flatness"/> pixels of the true circle.
    /// </summary>
    public static void Stroke(PixelOutline outline, ReadOnlySpan<PenPoint> points, bool closed, DocumentLineCap cap, DocumentLineJoin join, float miterLimit,
        float flatness, float directionX = 1f, float directionY = 0f)
    {
        if (points.IsEmpty)
        {
            return;
        }

        // Points that do not move are dropped (a closed piece's last point that is its first too).
        var kept = ArrayPool<PenPoint>.Shared.Rent(points.Length);
        try
        {
            int n = 0;
            foreach (var p in points)
            {
                if (n == 0 || MathF.Abs(p.X - kept[n - 1].X) + MathF.Abs(p.Y - kept[n - 1].Y) > 1e-4f)
                {
                    kept[n++] = p;
                }
            }

            if (closed && n > 1 && MathF.Abs(kept[0].X - kept[n - 1].X) + MathF.Abs(kept[0].Y - kept[n - 1].Y) <= 1e-4f)
            {
                n--;
            }

            var p2 = kept.AsSpan(0, n);
            if (n == 1)
            {
                Dot(outline, p2[0], cap, directionX, directionY, flatness);
                return;
            }

            if (closed && n >= 3)
            {
                Side(outline, p2, forward: true, join, miterLimit, flatness);
                Side(outline, p2, forward: false, join, miterLimit, flatness);
                return;
            }

            Open(outline, p2, cap, join, miterLimit, flatness);
        }
        finally
        {
            ArrayPool<PenPoint>.Shared.Return(kept);
        }
    }

    // A dot: a disc (round caps) or a square (square caps) of the stroke's width; butt caps draw nothing.
    private static void Dot(PixelOutline outline, PenPoint p, DocumentLineCap cap, float dx, float dy, float flatness)
    {
        float r = p.Width / 2;
        if (!(r > 0))
        {
            return;
        }

        if (cap == DocumentLineCap.Round)
        {
            Disc(outline, p.X, p.Y, r, flatness);
        }
        else if (cap == DocumentLineCap.Square)
        {
            float l = MathF.Sqrt(dx * dx + dy * dy);
            (dx, dy) = l > 0 ? (dx / l, dy / l) : (1f, 0f);
            float nx = -dy * r, ny = dx * r, ex = dx * r, ey = dy * r;
            Span<float> square = [p.X - ex + nx, p.Y - ey + ny, p.X + ex + nx, p.Y + ey + ny, p.X + ex - nx, p.Y + ey - ny, p.X - ex - nx, p.Y - ey - ny];
            outline.Polygon(square);
        }
    }

    // An open polyline: one contour, its left side forward, the end cap, its right side back, the start cap.
    private static void Open(PixelOutline outline, ReadOnlySpan<PenPoint> p, DocumentLineCap cap, DocumentLineJoin join, float miterLimit, float flatness)
    {
        int n = p.Length;
        var (dx, dy) = Direction(p[0], p[1]);
        float r0 = p[0].Width / 2;
        outline.Move(p[0].X - dy * r0, p[0].Y + dx * r0);
        Walk(outline, p, forward: true, closed: false, join, miterLimit, flatness);
        var (ex, ey) = Direction(p[n - 2], p[n - 1]);
        Cap(outline, p[n - 1], ex, ey, cap, flatness);
        Walk(outline, p, forward: false, closed: false, join, miterLimit, flatness);
        Cap(outline, p[0], -dx, -dy, cap, flatness);
        outline.Close();
    }

    // One side of a closed polyline as a loop from its first point: the left side forward, or the right side back (the
    // other way round, so the loops wind against each other and the inside of the inner one stays empty).
    private static void Side(PixelOutline outline, ReadOnlySpan<PenPoint> p, bool forward, DocumentLineJoin join, float miterLimit, float flatness)
    {
        int n = p.Length;
        var (dx, dy) = Direction(p[0], p[forward ? 1 : n - 1]);
        float r = p[0].Width / 2;
        outline.Move(p[0].X - dy * r, p[0].Y + dx * r);
        Walk(outline, p, forward, closed: true, join, miterLimit, flatness);
        outline.Close();
    }

    // Along one side: each segment's offset end, then the join at the point after it (for a closed loop, every point,
    // back to the first).
    private static void Walk(PixelOutline outline, ReadOnlySpan<PenPoint> p, bool forward, bool closed, DocumentLineJoin join, float miterLimit, float flatness)
    {
        int n = p.Length;
        int segments = closed ? n : n - 1;
        for (int s = 0; s < segments; s++)
        {
            int a = forward ? s : (closed ? (n - s) % n : n - 1 - s);
            int b = forward ? (a + 1) % n : (a - 1 + n) % n;
            var (dx, dy) = Direction(p[a], p[b]);
            float rb = p[b].Width / 2;
            float ox = p[b].X - dy * rb, oy = p[b].Y + dx * rb;            // the segment's end, offset to its left
            outline.Line(ox, oy);
            bool last = s == segments - 1;
            if (last && !closed)
            {
                break;
            }

            int c = forward ? (b + 1) % n : (b - 1 + n) % n;
            var (fx, fy) = Direction(p[b], p[c]);
            Join(outline, p[b], dx, dy, fx, fy, rb, join, miterLimit, flatness);
        }
    }

    // The join at `v` from the incoming direction (dx, dy) to the outgoing one (fx, fy), on the left side: outside the turn
    // a miter, a round or a bevel; inside it through the turn's point.
    private static void Join(PixelOutline outline, PenPoint v, float dx, float dy, float fx, float fy, float r, DocumentLineJoin join, float miterLimit, float flatness)
    {
        float ax = v.X - dy * r, ay = v.Y + dx * r;                         // the incoming side's end
        float bx = v.X - fy * r, by = v.Y + fx * r;                         // the outgoing side's start
        float cross = dx * fy - dy * fx, dot = dx * fx + dy * fy;
        if (MathF.Abs(cross) < 1e-6f && dot > 0)
        {
            outline.Line(bx, by);                                          // straight on
            return;
        }

        // The left side (-dy, dx) is inside the turn when the path turns towards it (cross > 0 in y-down pixels).
        bool outer = cross < 0 || (MathF.Abs(cross) < 1e-6f && dot < 0);
        if (!outer)
        {
            outline.Line(v.X, v.Y);
            outline.Line(bx, by);
            return;
        }

        switch (join)
        {
            case DocumentLineJoin.Round:
                Arc(outline, v.X, v.Y, r, ax, ay, bx, by, true, flatness);
                break;
            case DocumentLineJoin.Miter:
            {
                // The offset lines meet at v + r (n0 + n1) / (1 + n0 . n1); the miter's length over the width is
                // 1 / sin(phi / 2) = 1 / sqrt((1 + n0 . n1) / 2).
                float n0x = -dy, n0y = dx, n1x = -fy, n1y = fx;
                float nn = n0x * n1x + n0y * n1y;
                float ratio = 1f / MathF.Sqrt(MathF.Max(1e-12f, (1 + nn) / 2));
                if (ratio <= miterLimit && 1 + nn > 1e-6f)
                {
                    float k = r / (1 + nn);
                    outline.Line(v.X + (n0x + n1x) * k, v.Y + (n0y + n1y) * k);
                }

                outline.Line(bx, by);
                break;
            }

            default:
                outline.Line(bx, by);
                break;
        }
    }

    // A cap at `p` facing (dx, dy): from the left side's end to the right side's start.
    private static void Cap(PixelOutline outline, PenPoint p, float dx, float dy, DocumentLineCap cap, float flatness)
    {
        float r = p.Width / 2;
        float lx = p.X - dy * r, ly = p.Y + dx * r, rx = p.X + dy * r, ry = p.Y - dx * r;
        switch (cap)
        {
            case DocumentLineCap.Round:
                Arc(outline, p.X, p.Y, r, lx, ly, rx, ry, true, flatness, through: (p.X + dx * r, p.Y + dy * r));
                break;
            case DocumentLineCap.Square:
                outline.Line(lx + dx * r, ly + dy * r);
                outline.Line(rx + dx * r, ry + dy * r);
                outline.Line(rx, ry);
                break;
            default:
                outline.Line(rx, ry);
                break;
        }
    }

    // An arc of radius r about (cx, cy) from a to b: the way through `through` when given, else the short way; its polygon
    // within `flatness` pixels inside the circle.
    private static void Arc(PixelOutline outline, float cx, float cy, float r, float ax, float ay, float bx, float by, bool clockwiseShort, float flatness, (float X, float Y)? through = null)
    {
        double a0 = Math.Atan2(ay - cy, ax - cx), a1 = Math.Atan2(by - cy, bx - cx);
        double sweep = a1 - a0;
        if (through is { } t)
        {
            double am = Math.Atan2(t.Y - cy, t.X - cx);
            // The sweep that passes the middle angle.
            double s1 = Normalize(sweep), s2 = s1 > 0 ? s1 - 2 * Math.PI : s1 + 2 * Math.PI;
            double m = Normalize(am - a0);
            sweep = Within(m, s1) ? s1 : s2;
        }
        else
        {
            sweep = Normalize(sweep);
        }

        int steps = Math.Clamp((int)Math.Ceiling(Math.Abs(sweep) / (2 * Math.Acos(Math.Max(-1, 1 - flatness / Math.Max(r, flatness))))), 1, 256);
        for (int i = 1; i <= steps; i++)
        {
            double a = a0 + sweep * i / steps;
            outline.Line(i == steps ? bx : cx + r * (float)PortableMath.Cos(a), i == steps ? by : cy + r * (float)PortableMath.Sin(a));
        }
    }

    private static double Normalize(double angle)
    {
        while (angle > Math.PI)
        {
            angle -= 2 * Math.PI;
        }

        while (angle < -Math.PI)
        {
            angle += 2 * Math.PI;
        }

        return angle;
    }

    // Whether the angle m (from the start) lies within a sweep of s.
    private static bool Within(double m, double s) => s >= 0 ? (m >= 0 ? m <= s : m + 2 * Math.PI <= s) : (m <= 0 ? m >= s : m - 2 * Math.PI >= s);

    private static (float X, float Y) Direction(PenPoint a, PenPoint b)
    {
        float dx = b.X - a.X, dy = b.Y - a.Y, l = MathF.Sqrt(dx * dx + dy * dy);
        return l > 0 ? (dx / l, dy / l) : (1f, 0f);
    }

    // A disc of radius r at (x, y), within `flatness` pixels of the true circle.
    public static void Disc(PixelOutline outline, float x, float y, float r, float flatness)
    {
        int n = Math.Clamp((int)MathF.Ceiling(MathF.PI / MathF.Acos(MathF.Max(-1f, 1f - flatness / MathF.Max(r, flatness)))), 8, 512);
        outline.Move(x + r, y);
        for (int k = 1; k < n; k++)
        {
            double a = 2 * Math.PI * k / n;
            outline.Line(x + r * (float)PortableMath.Cos(a), y + r * (float)PortableMath.Sin(a));
        }

        outline.Close();
    }
}
