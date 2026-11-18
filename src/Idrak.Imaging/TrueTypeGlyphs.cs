// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Buffers.Binary;

namespace Idrak.Imaging;

/// <summary>
/// TrueType glyph records as a font stores them (glyf): how long one is without its padding, where its instructions lie,
/// a composite's parts, and a copy without the instructions. Every read is checked against the record's length.
/// </summary>
internal static class GlyphRecords
{
    private const int ArgsAreWords = 1, HaveScale = 8, MoreComponents = 32, HaveXYScale = 64, HaveTwoByTwo = 128, HaveInstructions = 0x100;

    /// <summary>
    /// A record's layout: its contours (-1 for a composite), its points (a simple glyph's), its length without padding,
    /// where the instructions' length field is and how many bytes they take, and (a composite) where its last part's flags are.
    /// </summary>
    internal readonly record struct Layout(int Contours, int Points, int Length, int InstructionsAt, int Instructions, int LastFlagsAt);

    /// <summary>The layout of <paramref name="record"/> (glyph <paramref name="glyph"/> of <paramref name="font"/>, for messages).</summary>
    /// <exception cref="InvalidDataException">The record ends before what it declares.</exception>
    public static Layout Measure(ReadOnlySpan<byte> record, string font, int glyph)
    {
        Need(record, 10, font, glyph);
        int contours = BinaryPrimitives.ReadInt16BigEndian(record);
        if (contours >= 0)
        {
            int at = 10 + 2 * contours;
            Need(record, at + 2, font, glyph);
            int points = contours == 0 ? 0 : BinaryPrimitives.ReadUInt16BigEndian(record[(at - 2)..]) + 1;
            int instructions = BinaryPrimitives.ReadUInt16BigEndian(record[at..]);
            int i = at + 2 + instructions, xBytes = 0, yBytes = 0;
            for (int p = 0; p < points;)
            {
                Need(record, i + 1, font, glyph);
                byte flag = record[i++];
                int run = 1;
                if ((flag & 8) != 0)
                {
                    Need(record, i + 1, font, glyph);
                    run += record[i++];
                }

                int n = Math.Min(run, points - p);                                 // a repeat past the last point is cut, as the reader does
                xBytes += n * ((flag & 2) != 0 ? 1 : (flag & 16) != 0 ? 0 : 2);
                yBytes += n * ((flag & 4) != 0 ? 1 : (flag & 32) != 0 ? 0 : 2);
                p += n;
            }

            int length = i + xBytes + yBytes;
            Need(record, length, font, glyph);
            return new Layout(contours, points, length, at, instructions, -1);
        }

        int c = 10, flags, last;
        do
        {
            Need(record, c + 4, font, glyph);
            flags = BinaryPrimitives.ReadUInt16BigEndian(record[c..]);
            last = c;
            c += 4 + ((flags & ArgsAreWords) != 0 ? 4 : 2)
                 + ((flags & HaveScale) != 0 ? 2 : (flags & HaveXYScale) != 0 ? 4 : (flags & HaveTwoByTwo) != 0 ? 8 : 0);
        }
        while ((flags & MoreComponents) != 0);

        Need(record, c, font, glyph);
        if ((flags & HaveInstructions) == 0)
        {
            return new Layout(-1, 0, c, c, 0, last);
        }

        Need(record, c + 2, font, glyph);
        int count = BinaryPrimitives.ReadUInt16BigEndian(record[c..]);
        Need(record, c + 2 + count, font, glyph);
        return new Layout(-1, 0, c + 2 + count, c, count, last);
    }

    /// <summary>The glyph ids a composite record names, in order (none for a simple glyph).</summary>
    public static void Components(ReadOnlySpan<byte> record, Layout layout, List<int> into)
    {
        if (layout.Contours >= 0)
        {
            return;
        }

        int c = 10, flags;
        do
        {
            flags = BinaryPrimitives.ReadUInt16BigEndian(record[c..]);
            into.Add(BinaryPrimitives.ReadUInt16BigEndian(record[(c + 2)..]));
            c += 4 + ((flags & ArgsAreWords) != 0 ? 4 : 2)
                 + ((flags & HaveScale) != 0 ? 2 : (flags & HaveXYScale) != 0 ? 4 : (flags & HaveTwoByTwo) != 0 ? 8 : 0);
        }
        while ((flags & MoreComponents) != 0);
    }

    /// <summary>The record as it is (without padding), or with its instructions left out (<paramref name="hinting"/> false).</summary>
    public static byte[] Copy(ReadOnlySpan<byte> record, Layout layout, bool hinting)
    {
        if (hinting || (layout.Instructions == 0 && (layout.Contours >= 0 || layout.InstructionsAt == layout.Length)))
        {
            return record[..layout.Length].ToArray();
        }

        if (layout.Contours >= 0)
        {
            // The header and contour ends, an empty program, then the flags and coordinates.
            int tail = layout.InstructionsAt + 2 + layout.Instructions;
            var simple = new byte[layout.InstructionsAt + 2 + (layout.Length - tail)];
            record[..layout.InstructionsAt].CopyTo(simple);
            record[tail..layout.Length].CopyTo(simple.AsSpan(layout.InstructionsAt + 2));
            return simple;
        }

        // A composite: its parts, the last one no longer saying instructions follow.
        var composite = record[..layout.InstructionsAt].ToArray();
        int flags = BinaryPrimitives.ReadUInt16BigEndian(composite.AsSpan(layout.LastFlagsAt));
        BinaryPrimitives.WriteUInt16BigEndian(composite.AsSpan(layout.LastFlagsAt), (ushort)(flags & ~HaveInstructions));
        return composite;
    }

    /// <summary>The bounding box a record's header states.</summary>
    public static (int XMin, int YMin, int XMax, int YMax) Box(ReadOnlySpan<byte> record) =>
        (BinaryPrimitives.ReadInt16BigEndian(record[2..]), BinaryPrimitives.ReadInt16BigEndian(record[4..]),
            BinaryPrimitives.ReadInt16BigEndian(record[6..]), BinaryPrimitives.ReadInt16BigEndian(record[8..]));

    private static void Need(ReadOnlySpan<byte> record, int length, string font, int glyph)
    {
        if (length > record.Length)
        {
            throw new InvalidDataException($"{font}: glyph {glyph}'s record ({record.Length} bytes) ends before the {length} bytes it declares.");
        }
    }
}

/// <summary>A glyph written as a simple TrueType record: its bytes (empty for no contour), contours, points and box.</summary>
internal readonly record struct EncodedGlyph(byte[] Data, int Contours, int Points, int XMin, int YMin, int XMax, int YMax);

/// <summary>
/// Any outline as a simple TrueType glyph (no instructions): each contour starts at its first point, on the curve;
/// lines and quadratic curves are written as on- and off-curve points (an on-curve point exactly midway between two
/// controls is left implied, as TrueType reads it); a cubic curve becomes the fewest quadratics within the tolerance
/// (<see cref="CubicSpline"/>); every point is rounded to whole font units (half up), and the segment that closes a
/// contour back on its start is the contour's own closing. An outline read from TrueType data of whole units is written
/// back to the same points, so it reads back the same.
/// </summary>
internal static class GlyphEncoder
{
    private readonly record struct Segment(OutlineVerb Verb, float X1, float Y1, float X2, float Y2, float X, float Y);

    /// <summary>The record of <paramref name="outline"/>; cubic curves within <paramref name="tolerance"/> font units.</summary>
    /// <exception cref="NotSupportedException">A point or a step between two lies outside what TrueType's 16 bits hold, or the glyph has more than 65,535 points.</exception>
    public static EncodedGlyph Encode(GlyphOutline outline, double tolerance, string font, int glyph)
    {
        if (outline.IsEmpty)
        {
            return new EncodedGlyph([], 0, 0, 0, 0, 0, 0);
        }

        var verbs = outline.Verbs.Span;
        var p = outline.Points.Span;
        var xs = new List<int>(p.Length / 2 + 8);
        var ys = new List<int>(p.Length / 2 + 8);
        var on = new List<bool>(p.Length / 2 + 8);
        var ends = new List<int>();
        var segments = new List<Segment>();
        var controls = new List<(double X, double Y)>();
        float sx = 0, sy = 0;
        int v = 0, k = 0;
        while (v < verbs.Length)
        {
            if (verbs[v] == OutlineVerb.Close)
            {
                v++;
                continue;
            }

            if (verbs[v] == OutlineVerb.Move)
            {
                (sx, sy) = (p[k], p[k + 1]);
                k += 2;
                v++;
            }

            segments.Clear();
            for (; v < verbs.Length && verbs[v] is not (OutlineVerb.Move or OutlineVerb.Close); v++)
            {
                switch (verbs[v])
                {
                    case OutlineVerb.Line:
                        segments.Add(new Segment(OutlineVerb.Line, 0, 0, 0, 0, p[k], p[k + 1]));
                        k += 2;
                        break;
                    case OutlineVerb.Quadratic:
                        segments.Add(new Segment(OutlineVerb.Quadratic, p[k], p[k + 1], 0, 0, p[k + 2], p[k + 3]));
                        k += 4;
                        break;
                    default:
                        segments.Add(new Segment(OutlineVerb.Cubic, p[k], p[k + 1], p[k + 2], p[k + 3], p[k + 4], p[k + 5]));
                        k += 6;
                        break;
                }
            }

            if (v < verbs.Length && verbs[v] == OutlineVerb.Close)
            {
                v++;
            }

            Add(sx, sy, true);
            float cx = sx, cy = sy;
            for (int j = 0; j < segments.Count; j++)
            {
                var s = segments[j];
                bool closes = j == segments.Count - 1 && s.X == sx && s.Y == sy;   // the contour's own closing: not stored
                switch (s.Verb)
                {
                    case OutlineVerb.Line:
                        if (!closes)
                        {
                            Add(s.X, s.Y, true);
                        }

                        break;
                    case OutlineVerb.Quadratic:
                        Add(s.X1, s.Y1, false);
                        bool implied = j + 1 < segments.Count && segments[j + 1] is { Verb: OutlineVerb.Quadratic } next
                                       && s.X == (s.X1 + next.X1) / 2 && s.Y == (s.Y1 + next.Y1) / 2;
                        if (!closes && !implied)
                        {
                            Add(s.X, s.Y, true);
                        }

                        break;
                    default:
                        controls.Clear();
                        CubicSpline.Approximate((cx, cy), (s.X1, s.Y1), (s.X2, s.Y2), (s.X, s.Y), tolerance, controls);
                        foreach (var (qx, qy) in controls)
                        {
                            Add(qx, qy, false);
                        }

                        if (!closes)
                        {
                            Add(s.X, s.Y, true);
                        }

                        break;
                }

                (cx, cy) = (s.X, s.Y);
            }

            ends.Add(xs.Count - 1);
        }

        if (xs.Count == 0)
        {
            return new EncodedGlyph([], 0, 0, 0, 0, 0, 0);
        }

        if (xs.Count > 0xFFFF)
        {
            throw new NotSupportedException($"{font}: glyph {glyph} has {xs.Count:N0} points; a TrueType glyph holds at most 65,535.");
        }

        int xMin = xs.Min(), yMin = ys.Min(), xMax = xs.Max(), yMax = ys.Max();
        var w = new FontBuffer(12 + 2 * ends.Count + 5 * xs.Count);
        w.I16(ends.Count);
        w.I16(xMin);
        w.I16(yMin);
        w.I16(xMax);
        w.I16(yMax);
        foreach (int end in ends)
        {
            w.U16(end);
        }

        w.U16(0);                                                                      // no instructions
        var flags = new byte[xs.Count];
        int px = 0, py = 0;
        for (int i = 0; i < xs.Count; i++)
        {
            int dx = xs[i] - px, dy = ys[i] - py;
            if (dx is < short.MinValue or > short.MaxValue || dy is < short.MinValue or > short.MaxValue)
            {
                throw new NotSupportedException($"{font}: glyph {glyph}'s point {i} lies {dx}, {dy} units from the one before; TrueType holds 16-bit steps.");
            }

            byte f = on[i] ? (byte)1 : (byte)0;
            f |= dx == 0 ? (byte)16 : dx is >= -255 and <= 255 ? (byte)(2 | (dx > 0 ? 16 : 0)) : (byte)0;
            f |= dy == 0 ? (byte)32 : dy is >= -255 and <= 255 ? (byte)(4 | (dy > 0 ? 32 : 0)) : (byte)0;
            flags[i] = f;
            (px, py) = (xs[i], ys[i]);
        }

        for (int i = 0; i < flags.Length;)
        {
            int run = 1;
            while (i + run < flags.Length && flags[i + run] == flags[i] && run < 256)
            {
                run++;
            }

            if (run >= 3)
            {
                w.U8(flags[i] | 8);
                w.U8(run - 1);
            }
            else
            {
                for (int r = 0; r < run; r++)
                {
                    w.U8(flags[i]);
                }
            }

            i += run;
        }

        px = 0;
        for (int i = 0; i < xs.Count; i++)
        {
            int dx = xs[i] - px;
            if ((flags[i] & 2) != 0)
            {
                w.U8(Math.Abs(dx));
            }
            else if ((flags[i] & 16) == 0)
            {
                w.I16(dx);
            }

            px = xs[i];
        }

        py = 0;
        for (int i = 0; i < ys.Count; i++)
        {
            int dy = ys[i] - py;
            if ((flags[i] & 4) != 0)
            {
                w.U8(Math.Abs(dy));
            }
            else if ((flags[i] & 32) == 0)
            {
                w.I16(dy);
            }

            py = ys[i];
        }

        return new EncodedGlyph(w.ToArray(), ends.Count, xs.Count, xMin, yMin, xMax, yMax);

        void Add(double x, double y, bool onCurve)
        {
            double rx = Math.Floor(x + 0.5), ry = Math.Floor(y + 0.5);
            if (!(rx >= short.MinValue && rx <= short.MaxValue && ry >= short.MinValue && ry <= short.MaxValue))
            {
                throw new NotSupportedException($"{font}: glyph {glyph} has a point at ({x}, {y}), outside the 16-bit range TrueType holds.");
            }

            xs.Add((int)rx);
            ys.Add((int)ry);
            on.Add(onCurve);
        }
    }
}

/// <summary>
/// A cubic curve as a spline of quadratic curves within a tolerance (the method of fontTools' cu2qu): for n = 1, 2, ...
/// segments, the cubic is cut into n pieces of equal parameter, each piece's quadratic control placed where its end
/// tangents meet (n = 1) or along the line through the pieces' raised controls (n &gt; 1), the on-curve points implied
/// midway between the controls; the first n whose every piece stays within the tolerance of the cubic (checked by
/// subdividing their difference) is taken.
/// </summary>
internal static class CubicSpline
{
    private const int MaxSegments = 100, MaxDepth = 48;

    /// <summary>Adds the spline's controls (off-curve points; the on-curve ones lie midway between them) to <paramref name="controls"/>.</summary>
    public static void Approximate((double X, double Y) p0, (double X, double Y) p1, (double X, double Y) p2, (double X, double Y) p3, double tolerance,
        List<(double X, double Y)> controls)
    {
        V a = new(p0.X, p0.Y), b = new(p1.X, p1.Y), c = new(p2.X, p2.Y), d = new(p3.X, p3.Y);
        var spline = new List<V>(4);
        for (int n = 1; n <= MaxSegments; n++)
        {
            spline.Clear();
            if (Spline(a, b, c, d, n, tolerance, spline))
            {
                break;
            }

            if (n == MaxSegments)
            {
                spline.Clear();
                Spline(a, b, c, d, n, double.PositiveInfinity, spline);              // none within it (a degenerate curve): the finest tried
            }
        }

        foreach (var q in spline)
        {
            controls.Add((q.X, q.Y));
        }
    }

    private static bool Spline(V p0, V p1, V p2, V p3, int n, double tolerance, List<V> spline)
    {
        if (n == 1)
        {
            var q = Intersect(p0, p1, p2, p3);
            if (double.IsNaN(q.X) || double.IsNaN(q.Y))
            {
                return false;
            }

            var c1 = p0 + (q - p0) * (2.0 / 3.0);
            var c2 = p3 + (q - p3) * (2.0 / 3.0);
            if (!FitsInside(default, c1 - p1, c2 - p2, default, tolerance, 0))
            {
                return false;
            }

            spline.Add(q);
            return true;
        }

        var pieces = Split(p0, p1, p2, p3, n);
        var nextQ1 = Control(0, pieces[0]);
        spline.Add(nextQ1);
        V q2 = p0, d1 = default;
        for (int i = 1; i <= n; i++)
        {
            var (_, c1, c2, c3) = pieces[i - 1];
            var q0 = q2;
            var q1 = nextQ1;
            if (i < n)
            {
                nextQ1 = Control(i / (double)(n - 1), pieces[i]);
                spline.Add(nextQ1);
                q2 = (q1 + nextQ1) * 0.5;
            }
            else
            {
                q2 = c3;
            }

            var d0 = d1;
            d1 = q2 - c3;
            if (d1.Length > tolerance || !FitsInside(d0, q0 + (q1 - q0) * (2.0 / 3.0) - c1, q2 + (q1 - q2) * (2.0 / 3.0) - c2, d1, tolerance, 0))
            {
                return false;
            }
        }

        return true;
    }

    // A piece's control: on the line through its two controls raised to a quadratic's, at t.
    private static V Control(double t, (V P0, V P1, V P2, V P3) piece)
    {
        var a = piece.P0 + (piece.P1 - piece.P0) * 1.5;
        var b = piece.P3 + (piece.P2 - piece.P3) * 1.5;
        return a + (b - a) * t;
    }

    // Whether the cubic of these differences stays within the tolerance: its controls, else its two halves.
    private static bool FitsInside(V p0, V p1, V p2, V p3, double tolerance, int depth)
    {
        if (p2.Length <= tolerance && p1.Length <= tolerance)
        {
            return true;
        }

        var mid = (p0 + (p1 + p2) * 3 + p3) * 0.125;
        if (mid.Length > tolerance || depth > MaxDepth)
        {
            return false;
        }

        var deriv = (p3 + p2 - p1 - p0) * 0.125;
        return FitsInside(p0, (p0 + p1) * 0.5, mid - deriv, mid, tolerance, depth + 1)
               && FitsInside(mid, mid + deriv, (p2 + p3) * 0.5, p3, tolerance, depth + 1);
    }

    // Where the line through a and b meets the line through c and d (NaN when they are parallel).
    private static V Intersect(V a, V b, V c, V d)
    {
        var ab = b - a;
        var cd = d - c;
        var perpendicular = new V(-ab.Y, ab.X);
        double denominator = perpendicular.Dot(cd);
        if (denominator == 0)
        {
            return new V(double.NaN, double.NaN);
        }

        return c + cd * (perpendicular.Dot(a - c) / denominator);
    }

    // The cubic cut into n pieces of equal parameter, from its polynomial's coefficients.
    private static (V, V, V, V)[] Split(V p0, V p1, V p2, V p3, int n)
    {
        var c = (p1 - p0) * 3.0;
        var b = (p2 - p1) * 3.0 - c;
        var d = p0;
        var a = p3 - d - c - b;
        double dt = 1.0 / n, dt2 = dt * dt, dt3 = dt * dt2;
        var pieces = new (V, V, V, V)[n];
        for (int i = 0; i < n; i++)
        {
            double t1 = i * dt, t12 = t1 * t1;
            var a1 = a * dt3;
            var b1 = (a * (3 * t1) + b) * dt2;
            var c1 = (b * (2 * t1) + c + a * (3 * t12)) * dt;
            var d1 = a * (t1 * t12) + b * t12 + c * t1 + d;
            var q1 = c1 * (1.0 / 3.0) + d1;
            pieces[i] = (d1, q1, (b1 + c1) * (1.0 / 3.0) + q1, a1 + d1 + c1 + b1);
        }

        return pieces;
    }

    private readonly record struct V(double X, double Y)
    {
        public double Length => Math.Sqrt(X * X + Y * Y);

        public double Dot(V other) => X * other.X + Y * other.Y;

        public static V operator +(V a, V b) => new(a.X + b.X, a.Y + b.Y);

        public static V operator -(V a, V b) => new(a.X - b.X, a.Y - b.Y);

        public static V operator *(V a, double s) => new(a.X * s, a.Y * s);
    }
}

/// <summary>A growing big-endian buffer for font tables.</summary>
internal sealed class FontBuffer(int capacity = 256)
{
    private byte[] _bytes = new byte[Math.Max(16, capacity)];

    public int Length { get; private set; }

    public ReadOnlySpan<byte> Span => _bytes.AsSpan(0, Length);

    public void U8(int value)
    {
        Ensure(1);
        _bytes[Length++] = (byte)value;
    }

    public void U16(int value)
    {
        Ensure(2);
        BinaryPrimitives.WriteUInt16BigEndian(_bytes.AsSpan(Length), (ushort)value);
        Length += 2;
    }

    public void I16(int value)
    {
        Ensure(2);
        BinaryPrimitives.WriteInt16BigEndian(_bytes.AsSpan(Length), (short)value);
        Length += 2;
    }

    public void U32(uint value)
    {
        Ensure(4);
        BinaryPrimitives.WriteUInt32BigEndian(_bytes.AsSpan(Length), value);
        Length += 4;
    }

    public void Tag(string tag)
    {
        Ensure(4);
        for (int i = 0; i < 4; i++)
        {
            _bytes[Length + i] = (byte)tag[i];
        }

        Length += 4;
    }

    public void Bytes(ReadOnlySpan<byte> bytes)
    {
        Ensure(bytes.Length);
        bytes.CopyTo(_bytes.AsSpan(Length));
        Length += bytes.Length;
    }

    public void Zeros(int count)
    {
        Ensure(count);
        _bytes.AsSpan(Length, count).Clear();
        Length += count;
    }

    /// <summary>Zeros up to the next multiple of <paramref name="size"/>.</summary>
    public void Align(int size) => Zeros((size - Length % size) % size);

    public void PutU16(int at, int value) => BinaryPrimitives.WriteUInt16BigEndian(_bytes.AsSpan(at), (ushort)value);

    public void PutU32(int at, uint value) => BinaryPrimitives.WriteUInt32BigEndian(_bytes.AsSpan(at), value);

    public byte[] ToArray() => _bytes.AsSpan(0, Length).ToArray();

    private void Ensure(int more)
    {
        if (Length + more > _bytes.Length)
        {
            Array.Resize(ref _bytes, Math.Max(_bytes.Length * 2, Length + more));
        }
    }
}
