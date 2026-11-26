// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Buffers;
using System.Globalization;
using System.Runtime.InteropServices;

namespace Idrak.Imaging;

/// <summary>
/// "contour" (plan 38, F1): a glyph's ink to its outline, in C# alone. The coverage is cut at an ink level
/// (<see cref="ThresholdKey"/>: Otsu's level of the coverage, <see cref="ImageThresholds.Otsu"/>, kept from 0.1 to 0.9,
/// unless set), and the edge where it crosses that level is found in one pass over the rows: marching squares between
/// pixel centres, each crossing placed between its two pixels by their coverage, a cell whose opposite corners are ink
/// joined when its middle is. Each contour is walked once with the ink on its right, so outer contours turn clockwise in
/// font units and holes the other way: one winding, holes kept. Its corners are where it turns by at least
/// <see cref="CornerKey"/> degrees within <see cref="SmoothingKey"/> pixels either side; between them the boundary is
/// simplified as Douglas-Peucker does (a piece is kept when every boundary point lies within <see cref="SimplifyKey"/>
/// pixels of it, else split at its farthest point), each piece a line when its chord holds the points, else a quadratic
/// curve whose control point is where the tangents at its ends meet, so curves meet smoothly away from corners. A piece
/// is kept only when no pixel centre beside it changes side, and the whole outline is checked against every pixel centre
/// once at the end (a piece that misplaces one is split again): filled, the outline holds exactly the pixel centres whose
/// coverage reaches the level. Every buffer is rented and returned; the tracer holds no state, so glyphs are traced from
/// several threads at once.
/// </summary>
internal sealed class ContourTracer : IOutlineTracer
{
    /// <summary>The ink level the edge is traced at: "auto" (Otsu's level of the coverage, kept from 0.1 to 0.9; the default) or a level from 0.01 to 0.99.</summary>
    internal const string ThresholdKey = "trace-threshold";

    /// <summary>The simplification tolerance: how far, in pixels, a line or curve may stray from the traced boundary (0 to 8; 0.5 unless set).</summary>
    internal const string SimplifyKey = "trace-simplify";

    /// <summary>The corner angle: a turn of at least this many degrees within the smoothing length is kept as a corner (0 to 180; 60 unless set).</summary>
    internal const string CornerKey = "trace-corner";

    /// <summary>
    /// The smoothing length: how many pixels of boundary either side of a point its direction is measured over, for its
    /// corners and its curves' tangents (0.5 to 16; 1.5 unless set).
    /// </summary>
    internal const string SmoothingKey = "trace-smoothing";

    /// <summary>The range "auto" keeps Otsu's level in: Otsu's own least, and as far from the top.</summary>
    internal const float AutoLeast = ImageThresholds.OtsuLeast, AutoMost = 1f - ImageThresholds.OtsuLeast;

    /// <summary>The defaults: half a pixel of simplification, corners from 60 degrees, directions over 1.5 pixels.</summary>
    internal const float DefaultSimplify = 0.5f, DefaultCorner = 60f, DefaultSmoothing = 1.5f;

    // The most a flattened curve strays from the true one, in pixels, where the fit and the checks walk it (as the rasterizer).
    private const float Flatness = 1f / 32;

    // The most chords a curve is flattened into.
    private const int MostSteps = 64;

    // The most times the whole outline is checked against the pixel centres (each round splits the pieces that misplace one).
    private const int MostRounds = 64;

    private static readonly string Own = string.Join(", ", ThresholdKey, SimplifyKey, CornerKey, SmoothingKey);

    private static readonly Settings Defaults = new(PageOptions.None, null, DefaultSimplify, DefaultCorner, DefaultSmoothing);

    /// <inheritdoc/>
    public string Name => OutlineTracers.Contour;

    /// <inheritdoc/>
    public string Summary => "the ink's edge at a level (trace-threshold: auto, Otsu's, or 0.01 to 0.99) traced in one pass and simplified within trace-simplify pixels "
                             + "into lines and quadratic curves, corners kept where it turns trace-corner degrees within trace-smoothing pixels";

    /// <inheritdoc/>
    /// <inheritdoc/>
    public OutlineTracerSettings Read(IReadOnlyDictionary<string, string> options)
    {
        ArgumentNullException.ThrowIfNull(options);
        if (options.Count == 0)
        {
            return Defaults;
        }

        float? threshold = null;
        float simplify = DefaultSimplify, corner = DefaultCorner, smoothing = DefaultSmoothing;
        foreach (var (raw, value) in options)
        {
            switch (raw.ToLowerInvariant())
            {
                case ThresholdKey:
                    threshold = Level(value);
                    break;
                case SimplifyKey:
                    simplify = PageOptions.Number($"The tracer contour's {SimplifyKey} (pixels)", value, 0f, 8f);
                    break;
                case CornerKey:
                    corner = PageOptions.Number($"The tracer contour's {CornerKey} (degrees)", value, 0f, 180f);
                    break;
                case SmoothingKey:
                    smoothing = PageOptions.Number($"The tracer contour's {SmoothingKey} (pixels)", value, 0.5f, 16f);
                    break;
                default:
                    throw new ArgumentException($"The outline tracer contour does not take '{raw}'; it takes: {Own}.", nameof(options));
            }
        }

        return new Settings(options, threshold, simplify, corner, smoothing);
    }

    /// <inheritdoc/>
    /// <remarks>
    /// The simplification tolerance and the most the traced level's edge strays from where a straight edge's coverage is
    /// half (<see cref="EdgeReach"/>; "auto" at its farthest level, 0.1 or 0.9). Sharp tips and strokes thinner than a pixel
    /// move further with a level away from the half.
    /// </remarks>
    public float Tolerance(OutlineTracerSettings settings)
    {
        var own = Of(settings);
        return own.Simplify + EdgeReach(own.Threshold);
    }

    /// <inheritdoc/>
    public GlyphOutline Trace(GlyphCoverage coverage, OutlinePlacement placement, OutlineTracerSettings settings)
    {
        ArgumentNullException.ThrowIfNull(coverage);
        var own = Of(settings);
        if (!placement.IsValid)
        {
            throw new ArgumentException(string.Create(CultureInfo.InvariantCulture,
                $"Not a placement (a scale above 0, every value finite): scale {placement.Scale}, origin ({placement.X}, {placement.Y})."), nameof(placement));
        }

        if (coverage.IsBlank)
        {
            return GlyphOutline.Empty;
        }

        float level = own.Threshold ?? Math.Clamp(ImageThresholds.Otsu(coverage.Values.Span), AutoLeast, AutoMost);
        using var tracing = new Tracing(coverage, level, own);
        return tracing.Outline(placement);
    }

    /// <summary>
    /// How far, in pixels, the edge traced at <paramref name="threshold"/> (null: "auto", at its farthest level) can stray
    /// from where a straight edge's coverage is half: the level's own offset and the crossing's placement between two pixel
    /// centres together, measured over every angle on exact area coverage (0.086 at 0.5, 0.24 at 0.4 and 0.6, 0.61 at 0.2 and
    /// 0.8, 0.87 at 0.1 and 0.9, 1.30 at 0.01 and 0.99) and bounded above them: 0.1 + 1.5 d + 2 d², d the level's distance
    /// from the half.
    /// </summary>
    internal static float EdgeReach(float? threshold)
    {
        float d = threshold is { } t ? MathF.Abs(t - 0.5f) : 0.5f - AutoLeast;
        return 0.1f + 1.5f * d + 2f * d * d;
    }

    // The tracer's settings, read from their options when another kind's were given.
    private Settings Of(OutlineTracerSettings settings)
    {
        ArgumentNullException.ThrowIfNull(settings);
        return settings as Settings ?? (Settings)Read(settings.Options);
    }

    private static float? Level(string value)
    {
        string v = (value ?? "").Trim();
        if (v.Equals("auto", StringComparison.OrdinalIgnoreCase))
        {
            return null;
        }

        return float.TryParse(v, NumberStyles.Float, CultureInfo.InvariantCulture, out float level) && level >= 0.01f && level <= 0.99f
            ? level
            : throw new ArgumentException($"The tracer contour's {ThresholdKey} is auto (Otsu's level of the coverage) or an ink level from 0.01 to 0.99, not '{value}'.");
    }

    // The settings read once from a tracer's options.
    private sealed class Settings(IReadOnlyDictionary<string, string> options, float? threshold, float simplify, float corner, float smoothing) : OutlineTracerSettings(options)
    {
        public float? Threshold { get; } = threshold;

        public float Simplify { get; } = simplify;

        public float Corner { get; } = corner;

        public float Smoothing { get; } = smoothing;

        // A turn is a corner when the cosine of its angle is at most this.
        public float CornerCosine { get; } = (float)PortableMath.Cos(corner * Math.PI / 180);
    }

    // A closed contour's points, in walk order: where they start in the shared buffers, how many, its length and the length its directions are measured over.
    private readonly record struct Contour(int Start, int Count, float Length, float Window);

    // A piece of a contour's outline: from point A to point B (B may pass the contour's count, wrapping round), a line or a quadratic curve with its control point.
    private readonly record struct Piece(int Contour, int A, int B, bool Curve, float Qx, float Qy);

    // A part of a piece that runs one way in y, from (X0, Y0) to (X1, Y1); of a curve, its parameters T0 to T1.
    private readonly record struct Part(float X0, float Y0, float X1, float Y1, float T0, float T1);

    // One trace's working state, in pixels (x right, y down; pixel (i, j) covers [i, i + 1] x [j, j + 1]); every buffer rented and returned on Dispose.
    private sealed class Tracing : IDisposable
    {
        private readonly float[] _values;
        private readonly int _offset, _width, _height;
        private readonly float _level, _tolerance, _cornerCosine, _smoothing;

        // The cells marched: the ink box's and one more each way. Cell (cx, cy) lies between the pixel centres of pixels
        // (cx - 1, cy - 1) and (cx, cy); outside the coverage there is no ink.
        private readonly int _left, _top, _right, _bottom;

        // The crossings in the order made: x, y; then the crossing its segment reaches and the segment's cell.
        private float[] _xy;
        private int[] _link;
        private int _crossings;

        // The contours' points in walk order: x, y; their segments' cells; arc length from the contour's first point; turn
        // cosine; corner. And each contour.
        private float[] _pos = [], _arc = [], _turn = [];
        private int[] _cell = [];
        private bool[] _corner = [];
        private Contour[] _contours = [];
        private int _contourCount;

        // The pieces, contour by contour, in order.
        private Piece[] _pieces = [];
        private int _pieceCount;

        // Scratch: a flattened curve's points, each span point's nearest chord, the fit's stack, a contour's corners, the
        // pixel centres the outline misplaces.
        private float[] _curve = [], _bad = [];
        private int[] _hint = [], _stack = [], _marks = [];
        private int _stackCount, _badCount;

        public Tracing(GlyphCoverage coverage, float level, Settings settings)
        {
            if (MemoryMarshal.TryGetArray(coverage.Values, out var segment) && segment.Array is not null)
            {
                (_values, _offset) = (segment.Array, segment.Offset);
            }
            else
            {
                (_values, _offset) = (coverage.Values.ToArray(), 0);
            }

            (_width, _height, _level) = (coverage.Width, coverage.Height, level);
            (_tolerance, _cornerCosine, _smoothing) = (settings.Simplify, settings.CornerCosine, settings.Smoothing);
            var box = coverage.InkBox;
            (_left, _top, _right, _bottom) = (box.X, box.Y, box.X + box.Width, box.Y + box.Height);
            int guess = 4 * (box.Width + box.Height) + 16;
            _xy = Rent<float>(2 * guess);
            _link = Rent<int>(3 * guess);
        }

        public void Dispose()
        {
            Return(ref _xy);
            Return(ref _link);
            Return(ref _pos);
            Return(ref _arc);
            Return(ref _turn);
            Return(ref _cell);
            Return(ref _corner);
            Return(ref _contours);
            Return(ref _pieces);
            Return(ref _curve);
            Return(ref _bad);
            Return(ref _hint);
            Return(ref _stack);
            Return(ref _marks);
        }

        // The outline in font units placed by `at`.
        public GlyphOutline Outline(OutlinePlacement at)
        {
            March();
            Collect();
            if (_contourCount == 0)
            {
                return GlyphOutline.Empty;
            }

            _pieces = Rent<Piece>(_contourCount * 4 + 16);
            _curve = Rent<float>(2 * (MostSteps + 1));
            for (int c = 0; c < _contourCount; c++)
            {
                FitContour(c);
            }

            for (int round = 0; round < MostRounds; round++)
            {
                if (Check() == 0 || !Refit())
                {
                    break;
                }
            }

            return Emit(at);
        }

        // ------------------------------------------------------------------ the boundary, in one pass

        // Every cell row in order, two rows of coverage at a time: where the level falls between two pixel centres a
        // crossing is made, and each cell joins its crossings with the ink on the right (a cell's crossings on its top and
        // left edges were made by the cells above and before it).
        private void March()
        {
            ReadOnlySpan<float> values = _values.AsSpan(_offset, _width * _height);
            int width = _width;
            float level = _level;
            int[] above = Rent<int>(_right + 2);
            try
            {
                above.AsSpan(0, _right + 2).Fill(-1);
                for (int cy = _top; cy <= _bottom; cy++)
                {
                    var upper = cy >= 1 && cy <= _height ? values.Slice((cy - 1) * width, width) : default;
                    var lower = cy < _height ? values.Slice(cy * width, width) : default;
                    float tl = At(upper, _left - 1), bl = At(lower, _left - 1);
                    int before = -1;
                    for (int cx = _left; cx <= _right; cx++)
                    {
                        float tr = At(upper, cx), br = At(lower, cx);
                        int code = (tl >= level ? 1 : 0) | (tr >= level ? 2 : 0) | (br >= level ? 4 : 0) | (bl >= level ? 8 : 0);
                        int right = -1, bottom = -1;
                        if (code != 0 && code != 15)
                        {
                            int top = ((code ^ (code >> 1)) & 1) != 0 ? above[cx] : -1;
                            int left = ((code ^ (code >> 3)) & 1) != 0 ? before : -1;
                            if ((((code >> 1) ^ (code >> 2)) & 1) != 0)
                            {
                                right = Cross(cx + 0.5f, cy - 0.5f + (level - tr) / (br - tr));
                            }

                            if ((((code >> 2) ^ (code >> 3)) & 1) != 0)
                            {
                                bottom = Cross(cx - 0.5f + (level - bl) / (br - bl), cy + 0.5f);
                            }

                            Join(code, top, right, bottom, left, cx, cy, (tl + tr + br + bl) * 0.25f >= level);
                        }

                        above[cx] = bottom;
                        before = right;
                        (tl, bl) = (tr, br);
                    }
                }
            }
            finally
            {
                ArrayPool<int>.Shared.Return(above);
            }
        }

        private static float At(ReadOnlySpan<float> row, int x) => (uint)x < (uint)row.Length ? row[x] : 0f;

        // A cell's segments, ink on the right (y down): corners top-left 1, top-right 2, bottom-right 4, bottom-left 8.
        private void Join(int code, int top, int right, int bottom, int left, int cx, int cy, bool middleInk)
        {
            switch (code)
            {
                case 1:
                    Link(top, left, cx, cy);
                    break;
                case 2:
                    Link(right, top, cx, cy);
                    break;
                case 3:
                    Link(right, left, cx, cy);
                    break;
                case 4:
                    Link(bottom, right, cx, cy);
                    break;
                case 5:
                    // Top-left and bottom-right ink: one piece through the middle when it is ink, else two.
                    if (middleInk)
                    {
                        Link(top, right, cx, cy);
                        Link(bottom, left, cx, cy);
                    }
                    else
                    {
                        Link(top, left, cx, cy);
                        Link(bottom, right, cx, cy);
                    }

                    break;
                case 6:
                    Link(bottom, top, cx, cy);
                    break;
                case 7:
                    Link(bottom, left, cx, cy);
                    break;
                case 8:
                    Link(left, bottom, cx, cy);
                    break;
                case 9:
                    Link(top, bottom, cx, cy);
                    break;
                case 10:
                    // Top-right and bottom-left ink.
                    if (middleInk)
                    {
                        Link(left, top, cx, cy);
                        Link(right, bottom, cx, cy);
                    }
                    else
                    {
                        Link(right, top, cx, cy);
                        Link(left, bottom, cx, cy);
                    }

                    break;
                case 11:
                    Link(right, bottom, cx, cy);
                    break;
                case 12:
                    Link(left, right, cx, cy);
                    break;
                case 13:
                    Link(top, right, cx, cy);
                    break;
                case 14:
                    Link(left, top, cx, cy);
                    break;
            }
        }

        private int Cross(float x, float y)
        {
            int k = _crossings++;
            if (2 * _crossings > _xy.Length)
            {
                Grow(ref _xy, 2 * k, 4 * _crossings);
            }

            if (3 * _crossings > _link.Length)
            {
                Grow(ref _link, 3 * k, 6 * _crossings);
            }

            _xy[2 * k] = x;
            _xy[2 * k + 1] = y;
            _link[3 * k] = -1;
            return k;
        }

        private void Link(int from, int to, int cx, int cy)
        {
            if (from < 0 || to < 0 || _link[3 * from] >= 0)
            {
                throw new InvalidOperationException($"The contour tracer joined cell ({cx}, {cy}) wrongly: a crossing is missing or taken twice.");
            }

            _link[3 * from] = to;
            _link[3 * from + 1] = cx;
            _link[3 * from + 2] = cy;
        }

        // Each closed boundary walked once from its first crossing; points that coincide (a pixel centre exactly at the
        // level) kept once, contours with no area left out; then each contour's lengths, turns and corners.
        private void Collect()
        {
            int n = _crossings;
            _pos = Rent<float>(2 * n);
            _cell = Rent<int>(2 * n);
            _arc = Rent<float>(n);
            _turn = Rent<float>(n);
            _corner = Rent<bool>(n);
            _contours = Rent<Contour>(16);
            int written = 0;
            for (int k = 0; k < n; k++)
            {
                if (_link[3 * k] < 0)
                {
                    continue;                                                       // walked already
                }

                int start = written, at = k;
                do
                {
                    int next = _link[3 * at];
                    if (next < 0)
                    {
                        throw new InvalidOperationException("The contour tracer found a boundary that does not close.");
                    }

                    _link[3 * at] = ~next;
                    float x = _xy[2 * at], y = _xy[2 * at + 1];
                    if (written > start && _pos[2 * written - 2] == x && _pos[2 * written - 1] == y)
                    {
                        // The same point again: its segment is the one that leaves it.
                        _cell[2 * written - 2] = _link[3 * at + 1];
                        _cell[2 * written - 1] = _link[3 * at + 2];
                    }
                    else
                    {
                        _pos[2 * written] = x;
                        _pos[2 * written + 1] = y;
                        _cell[2 * written] = _link[3 * at + 1];
                        _cell[2 * written + 1] = _link[3 * at + 2];
                        written++;
                    }

                    at = next;
                }
                while (at != k);

                int count = written - start;
                if (count > 1 && _pos[2 * start] == _pos[2 * written - 2] && _pos[2 * start + 1] == _pos[2 * written - 1])
                {
                    count--;
                    written--;
                }

                if (count < 3 || Math.Abs(Area(start, count)) < 1e-6)
                {
                    written = start;
                    continue;
                }

                AddContour(start, count);
            }
        }

        private double Area(int start, int count)
        {
            double sum = 0;
            for (int i = 0; i < count; i++)
            {
                int j = i + 1 == count ? 0 : i + 1;
                sum += (double)_pos[2 * (start + i)] * _pos[2 * (start + j) + 1] - (double)_pos[2 * (start + j)] * _pos[2 * (start + i) + 1];
            }

            return sum / 2;
        }

        private void AddContour(int start, int count)
        {
            float length = 0;
            _arc[start] = 0;
            for (int i = 1; i < count; i++)
            {
                length += Distance(start + i - 1, start + i);
                _arc[start + i] = length;
            }

            length += Distance(start + count - 1, start);
            var contour = new Contour(start, count, length, MathF.Min(_smoothing, length / 4));
            if (_contourCount == _contours.Length)
            {
                Grow(ref _contours, _contourCount, 2 * _contourCount);
            }

            _contours[_contourCount++] = contour;

            // Each point's turn: the angle between the boundary's direction over the window before it and after it.
            for (int i = 0; i < count; i++)
            {
                var (px, py) = Point(contour, i);
                var (bx, by) = Along(contour, i, -contour.Window);
                var (fx, fy) = Along(contour, i, contour.Window);
                float ix = px - bx, iy = py - by, ox = fx - px, oy = fy - py;
                float norms = MathF.Sqrt((ix * ix + iy * iy) * (ox * ox + oy * oy));
                _turn[start + i] = norms > 0 ? (ix * ox + iy * oy) / norms : 1f;
            }

            // A corner: a turn of at least the corner angle, the sharpest within the window (the first of equal ones).
            for (int i = 0; i < count; i++)
            {
                float turn = _turn[start + i];
                bool corner = turn <= _cornerCosine;
                float reach = 0;
                for (int j = i; corner;)
                {
                    int previous = j == 0 ? count - 1 : j - 1;
                    reach += Segment(contour, previous);
                    if (reach > contour.Window || previous == i)
                    {
                        break;
                    }

                    corner = _turn[start + previous] > turn;
                    j = previous;
                }

                reach = 0;
                for (int j = i; corner;)
                {
                    reach += Segment(contour, j);
                    int next = j + 1 == count ? 0 : j + 1;
                    if (reach > contour.Window || next == i)
                    {
                        break;
                    }

                    corner = _turn[start + next] >= turn;
                    j = next;
                }

                _corner[start + i] = corner;
            }
        }

        private float Distance(int i, int j)
        {
            float dx = _pos[2 * j] - _pos[2 * i], dy = _pos[2 * j + 1] - _pos[2 * i + 1];
            return MathF.Sqrt(dx * dx + dy * dy);
        }

        // Point i of a contour (i from 0 to twice its count: a span may wrap round once).
        private (float X, float Y) Point(in Contour c, int i)
        {
            int g = c.Start + (i < c.Count ? i : i - c.Count);
            return (_pos[2 * g], _pos[2 * g + 1]);
        }

        // The cell of the segment that leaves point i.
        private (int X, int Y) Cell(in Contour c, int i)
        {
            int g = c.Start + (i < c.Count ? i : i - c.Count);
            return (_cell[2 * g], _cell[2 * g + 1]);
        }

        // The length of the segment from point i (0 to count - 1) to the next.
        private float Segment(in Contour c, int i) => i + 1 < c.Count ? _arc[c.Start + i + 1] - _arc[c.Start + i] : c.Length - _arc[c.Start + i];

        // The point `distance` along the boundary from point i (backwards when negative), less than the contour's length.
        private (float X, float Y) Along(in Contour c, int i, float distance)
        {
            int n = c.Count;
            float left = MathF.Abs(distance);
            for (int step = 0; step < n; step++)
            {
                int j = distance >= 0 ? (i + 1 == n ? 0 : i + 1) : (i == 0 ? n - 1 : i - 1);
                float length = Segment(c, distance >= 0 ? i : j);
                if (left <= length)
                {
                    var (ax, ay) = Point(c, i);
                    if (!(length > 0))
                    {
                        return (ax, ay);
                    }

                    var (bx, by) = Point(c, j);
                    float u = left / length;
                    return (ax + u * (bx - ax), ay + u * (by - ay));
                }

                left -= length;
                i = j;
            }

            return Point(c, i);
        }

        // The boundary's direction at point i (a unit vector, zero when it has none): over the window either side at a
        // smooth point; leaving it or reaching it at a corner.
        private (float X, float Y) Tangent(in Contour c, int i, bool leaving)
        {
            int local = i < c.Count ? i : i - c.Count;
            bool corner = _corner[c.Start + local];
            var (fx, fy) = corner && leaving ? Point(c, local) : Along(c, local, -c.Window);
            var (tx, ty) = corner && !leaving ? Point(c, local) : Along(c, local, c.Window);
            float dx = tx - fx, dy = ty - fy, norm = MathF.Sqrt(dx * dx + dy * dy);
            return norm > 0 ? (dx / norm, dy / norm) : (0f, 0f);
        }

        // ------------------------------------------------------------------ simplification and curves

        // A contour's pieces: between its corners, or from its first point to its farthest and back when it has none.
        private void FitContour(int ci)
        {
            var c = _contours[ci];
            if (_marks.Length < c.Count)
            {
                Return(ref _marks);
                _marks = Rent<int>(c.Count);
            }

            int corners = 0;
            for (int i = 0; i < c.Count; i++)
            {
                if (_corner[c.Start + i])
                {
                    _marks[corners++] = i;
                }
            }

            if (corners == 0)
            {
                int farthest = Farthest(c, 0, c.Count);
                FitSpan(ci, 0, farthest, forced: false);
                FitSpan(ci, farthest, c.Count, forced: false);
                return;
            }

            for (int k = 0; k < corners; k++)
            {
                FitSpan(ci, _marks[k], k + 1 < corners ? _marks[k + 1] : _marks[0] + c.Count, forced: false);
            }
        }

        // The pieces of points a to b, in order: a line when the chord holds every point (and keeps every pixel centre's
        // side), else a quadratic curve between the tangents at both ends when it does, else split at the farthest point.
        // A forced span (one that misplaced a pixel centre) is split at once.
        private void FitSpan(int ci, int a, int b, bool forced)
        {
            var c = _contours[ci];
            float tolerance2 = _tolerance * _tolerance;
            Push(a, b, forced);
            while (_stackCount > 0)
            {
                _stackCount--;
                (a, b, forced) = (_stack[3 * _stackCount], _stack[3 * _stackCount + 1], _stack[3 * _stackCount + 2] != 0);
                if (b - a == 1)
                {
                    Add(new Piece(ci, a, b, false, 0f, 0f));
                    continue;
                }

                int split;
                if (b - a >= c.Count)
                {
                    split = Farthest(c, a, b);                                      // a whole contour from one corner: split it first
                }
                else
                {
                    var (ax, ay) = Point(c, a);
                    var (bx, by) = Point(c, b);
                    float worst = -1;
                    split = a + (b - a) / 2;
                    for (int k = a + 1; k < b; k++)
                    {
                        var (px, py) = Point(c, k);
                        float d = SegmentDistance2(px, py, ax, ay, bx, by);
                        if (d > worst)
                        {
                            (worst, split) = (d, k);
                        }
                    }

                    if (!forced)
                    {
                        if (worst <= tolerance2 && LineKeeps(c, a, b))
                        {
                            Add(new Piece(ci, a, b, false, 0f, 0f));
                            continue;
                        }

                        if (CurveFits(c, a, b, out float qx, out float qy))
                        {
                            Add(new Piece(ci, a, b, true, qx, qy));
                            continue;
                        }
                    }
                }

                if (split <= a || split >= b)
                {
                    split = a + (b - a) / 2;
                }

                Push(split, b, false);
                Push(a, split, false);                                              // the first half comes off the stack first
            }
        }

        private void Push(int a, int b, bool forced)
        {
            if (3 * (_stackCount + 1) > _stack.Length)
            {
                Grow(ref _stack, 3 * _stackCount, 3 * (_stackCount + 1) * 2 + 48);
            }

            _stack[3 * _stackCount] = a;
            _stack[3 * _stackCount + 1] = b;
            _stack[3 * _stackCount + 2] = forced ? 1 : 0;
            _stackCount++;
        }

        private void Add(Piece piece)
        {
            if (_pieceCount == _pieces.Length)
            {
                Grow(ref _pieces, _pieceCount, 2 * _pieceCount + 16);
            }

            _pieces[_pieceCount++] = piece;
        }

        // The point of a + 1 .. b - 1 farthest from point a.
        private int Farthest(in Contour c, int a, int b)
        {
            var (ax, ay) = Point(c, a);
            int farthest = a + (b - a) / 2;
            float best = -1;
            for (int k = a + 1; k < b; k++)
            {
                var (px, py) = Point(c, k);
                float d = (px - ax) * (px - ax) + (py - ay) * (py - ay);
                if (d > best)
                {
                    (best, farthest) = (d, k);
                }
            }

            return farthest;
        }

        // Whether the chord from point a to point b keeps every pixel centre beside the points between on its side: a
        // centre within the tolerance of the chord, across from it, must lie on the side its ink says (ink on the right).
        private bool LineKeeps(in Contour c, int a, int b)
        {
            var (ax, ay) = Point(c, a);
            var (bx, by) = Point(c, b);
            float dx = bx - ax, dy = by - ay, length2 = dx * dx + dy * dy;
            if (!(length2 > 0))
            {
                return false;
            }

            float reach = _tolerance * MathF.Sqrt(length2);
            for (int k = a; k < b; k++)
            {
                var (cx, cy) = Cell(c, k);
                for (int corner = 0; corner < 4; corner++)
                {
                    int gx = cx + (corner & 1), gy = cy + (corner >> 1);
                    int side = Side(gx, gy);
                    if (side == 0)
                    {
                        continue;
                    }

                    float sx = gx - 0.5f - ax, sy = gy - 0.5f - ay;
                    float along = sx * dx + sy * dy, cross = dx * sy - dy * sx;
                    if (along <= 0 || along >= length2 || MathF.Abs(cross) > reach)
                    {
                        continue;
                    }

                    if (side * cross <= 0)
                    {
                        return false;
                    }
                }
            }

            return true;
        }

        // The quadratic curve from point a to point b whose control point is where the tangents at both ends meet, when
        // they meet ahead of both ends and the curve holds the points between both ways (and keeps every pixel centre's side).
        private bool CurveFits(in Contour c, int a, int b, out float qx, out float qy)
        {
            (qx, qy) = (0f, 0f);
            var (ax, ay) = Point(c, a);
            var (bx, by) = Point(c, b);
            var (sx, sy) = Tangent(c, a, leaving: true);
            var (ex, ey) = Tangent(c, b, leaving: false);
            float across = sx * ey - sy * ex;
            if (MathF.Abs(across) < 1e-4f)
            {
                return false;                                                       // parallel: straight, or an S a quadratic cannot make
            }

            float dx = bx - ax, dy = by - ay;
            float s = (dx * ey - dy * ex) / across, u = (sx * dy - sy * dx) / across;
            if (!(s > 0 && u > 0))
            {
                return false;
            }

            (qx, qy) = (ax + s * sx, ay + s * sy);
            int steps = Flatten(ax, ay, qx, qy, bx, by);
            return Holds(c, a, b, steps) && CurveKeeps(c, a, b, steps, ax, ay, qx, qy, bx, by);
        }

        // The curve as `steps` chords (within the flatness), its points in _curve.
        private int Flatten(float ax, float ay, float qx, float qy, float bx, float by)
        {
            float bend = MathF.Sqrt(Square(ax - 2 * qx + bx) + Square(ay - 2 * qy + by));
            int steps = Math.Clamp((int)MathF.Ceiling(MathF.Sqrt(bend / (8 * Flatness))), 2, MostSteps);
            for (int k = 0; k <= steps; k++)
            {
                float t = (float)k / steps, u = 1 - t;
                _curve[2 * k] = u * u * ax + 2 * u * t * qx + t * t * bx;
                _curve[2 * k + 1] = u * u * ay + 2 * u * t * qy + t * t * by;
            }

            return steps;
        }

        // Whether every point between a and b lies within the tolerance of the flattened curve and every point of the
        // curve within it of the boundary, each found walking forward (so never nearer than it is); each point's chord kept.
        private bool Holds(in Contour c, int a, int b, int steps)
        {
            float tolerance2 = _tolerance * _tolerance;
            if (_hint.Length < b - a + 1)
            {
                Return(ref _hint);
                _hint = Rent<int>(b - a + 1);
            }

            _hint[0] = 0;
            int j = 0;
            for (int k = a + 1; k < b; k++)
            {
                var (px, py) = Point(c, k);
                float d = Chord2(px, py, j);
                while (j + 1 < steps)
                {
                    float further = Chord2(px, py, j + 1);
                    if (further > d)
                    {
                        break;
                    }

                    (d, j) = (further, j + 1);
                }

                if (d > tolerance2)
                {
                    return false;
                }

                _hint[k - a] = j;
            }

            int q = a;
            for (int v = 1; v < steps; v++)
            {
                float vx = _curve[2 * v], vy = _curve[2 * v + 1];
                var (px, py) = Point(c, q);
                var (nx, ny) = Point(c, q + 1);
                float d = SegmentDistance2(vx, vy, px, py, nx, ny);
                while (q + 1 < b)
                {
                    var (fx, fy) = Point(c, q + 2);
                    float further = SegmentDistance2(vx, vy, nx, ny, fx, fy);
                    if (further > d)
                    {
                        break;
                    }

                    (d, q, nx, ny) = (further, q + 1, fx, fy);
                }

                if (d > tolerance2)
                {
                    return false;
                }
            }

            return true;
        }

        private float Chord2(float px, float py, int j) => SegmentDistance2(px, py, _curve[2 * j], _curve[2 * j + 1], _curve[2 * j + 2], _curve[2 * j + 3]);

        // Whether the curve keeps every pixel centre beside the points between a and b on its side: a centre within the
        // tolerance of the curve, its nearest point inside the curve, must lie on the side its ink says.
        private bool CurveKeeps(in Contour c, int a, int b, int steps, float ax, float ay, float qx, float qy, float bx, float by)
        {
            float tolerance2 = _tolerance * _tolerance;
            float ddx = 2 * (ax - 2 * qx + bx), ddy = 2 * (ay - 2 * qy + by);
            for (int k = a; k < b; k++)
            {
                var (cx, cy) = Cell(c, k);
                int hint = _hint[k - a];
                for (int corner = 0; corner < 4; corner++)
                {
                    int gx = cx + (corner & 1), gy = cy + (corner >> 1);
                    int side = Side(gx, gy);
                    if (side == 0)
                    {
                        continue;
                    }

                    // The nearest chord, among those near the point's own.
                    float sx = gx - 0.5f, sy = gy - 0.5f, nearest = float.MaxValue, along = 0;
                    int chord = -1;
                    for (int j = Math.Max(0, hint - 4), last = Math.Min(steps - 1, hint + 4); j <= last; j++)
                    {
                        float x0 = _curve[2 * j], y0 = _curve[2 * j + 1], cdx = _curve[2 * j + 2] - x0, cdy = _curve[2 * j + 3] - y0;
                        float l2 = cdx * cdx + cdy * cdy, u = l2 > 0 ? ((sx - x0) * cdx + (sy - y0) * cdy) / l2 : 0f, w = Math.Clamp(u, 0f, 1f);
                        float d = Square(sx - x0 - w * cdx) + Square(sy - y0 - w * cdy);
                        if (d < nearest)
                        {
                            (nearest, chord, along) = (d, j, u);
                        }
                    }

                    if (nearest > tolerance2 || (chord == 0 && along <= 0) || (chord == steps - 1 && along >= 1))
                    {
                        continue;
                    }

                    // From the chord's nearest point to the curve's, by Newton's steps on (B(t) - s) . B'(t) = 0.
                    float t = (chord + Math.Clamp(along, 0f, 1f)) / steps;
                    for (int step = 0; step < 3; step++)
                    {
                        float v = 1 - t;
                        float px = v * v * ax + 2 * v * t * qx + t * t * bx - sx, py = v * v * ay + 2 * v * t * qy + t * t * by - sy;
                        float dx = 2 * (v * (qx - ax) + t * (bx - qx)), dy = 2 * (v * (qy - ay) + t * (by - qy));
                        float slope = dx * dx + dy * dy + px * ddx + py * ddy;
                        if (!(slope > 0))
                        {
                            break;
                        }

                        t = Math.Clamp(t - (px * dx + py * dy) / slope, 0f, 1f);
                    }

                    if (t <= 0 || t >= 1)
                    {
                        continue;
                    }

                    float r = 1 - t;
                    float mx = r * r * ax + 2 * r * t * qx + t * t * bx, my = r * r * ay + 2 * r * t * qy + t * t * by;
                    float tx = 2 * (r * (qx - ax) + t * (bx - qx)), ty = 2 * (r * (qy - ay) + t * (by - qy));
                    if (side * (tx * (sy - my) - ty * (sx - mx)) <= 0)
                    {
                        return false;
                    }
                }
            }

            return true;
        }

        // Pixel centre (gx, gy) of the marching grid (pixel (gx - 1, gy - 1)): 1 ink, -1 paper, 0 exactly at the level.
        private int Side(int gx, int gy)
        {
            int x = gx - 1, y = gy - 1;
            float v = (uint)x < (uint)_width && (uint)y < (uint)_height ? _values[_offset + y * _width + x] : 0f;
            return v == _level ? 0 : v > _level ? 1 : -1;
        }

        // ------------------------------------------------------------------ the check against every pixel centre

        // Every pixel centre the pieces could reach: the outline's winding there (its crossings to the left, row by row)
        // must be -1 where the coverage reaches the level and 0 where it does not. The centres it misplaces are kept.
        private int Check()
        {
            int reach = (int)MathF.Ceiling(_tolerance) + 1;
            int gx0 = _left - reach + 1, gx1 = _right + reach, gy0 = _top - reach + 1, gy1 = _bottom + reach, rows = gy1 - gy0 + 1;
            int[] starts = Rent<int>(rows + 1);
            float[] xs = [];
            int[] directions = [];
            Span<Part> parts = stackalloc Part[2];
            try
            {
                starts.AsSpan(0, rows + 1).Clear();
                for (int p = 0; p < _pieceCount; p++)
                {
                    int count = Parts(_pieces[p], parts);
                    for (int k = 0; k < count; k++)
                    {
                        var (from, to) = Rows(parts[k], gy0, gy1);
                        for (int gy = from; gy <= to; gy++)
                        {
                            if (Crosses(parts[k], gy - 0.5f))
                            {
                                starts[gy - gy0 + 1]++;
                            }
                        }
                    }
                }

                for (int r = 0; r < rows; r++)
                {
                    starts[r + 1] += starts[r];
                }

                xs = Rent<float>(starts[rows]);
                directions = Rent<int>(starts[rows]);
                int[] filled = Rent<int>(rows);
                try
                {
                    starts.AsSpan(0, rows).CopyTo(filled);
                    for (int p = 0; p < _pieceCount; p++)
                    {
                        var piece = _pieces[p];
                        int count = Parts(piece, parts);
                        for (int k = 0; k < count; k++)
                        {
                            var part = parts[k];
                            var (from, to) = Rows(part, gy0, gy1);
                            for (int gy = from; gy <= to; gy++)
                            {
                                float yc = gy - 0.5f;
                                if (Crosses(part, yc))
                                {
                                    int slot = filled[gy - gy0]++;
                                    xs[slot] = CrossingX(piece, part, yc);
                                    directions[slot] = part.Y1 > part.Y0 ? 1 : -1;
                                }
                            }
                        }
                    }
                }
                finally
                {
                    ArrayPool<int>.Shared.Return(filled);
                }

                _badCount = 0;
                for (int r = 0; r < rows; r++)
                {
                    int from = starts[r], to = starts[r + 1];
                    for (int i = from + 1; i < to; i++)                              // a row holds few crossings: insertion sort
                    {
                        float x = xs[i];
                        int direction = directions[i], j = i - 1;
                        for (; j >= from && xs[j] > x; j--)
                        {
                            xs[j + 1] = xs[j];
                            directions[j + 1] = directions[j];
                        }

                        xs[j + 1] = x;
                        directions[j + 1] = direction;
                    }

                    int gy = gy0 + r, winding = 0, next = from;
                    for (int gx = gx0; gx <= gx1; gx++)
                    {
                        float xc = gx - 0.5f;
                        while (next < to && xs[next] < xc)
                        {
                            winding += directions[next++];
                        }

                        int side = Side(gx, gy);
                        if (side != 0 && (winding is not (0 or -1) || (winding != 0) != (side > 0)))
                        {
                            Bad(xc, gy - 0.5f);
                        }
                    }
                }

                return _badCount;
            }
            finally
            {
                ArrayPool<int>.Shared.Return(starts);
                Return(ref xs);
                Return(ref directions);
            }
        }

        // The piece's parts that run one way in y: a line whole, a curve split where it turns back.
        private int Parts(in Piece piece, Span<Part> parts)
        {
            var c = _contours[piece.Contour];
            var (ax, ay) = Point(c, piece.A);
            var (bx, by) = Point(c, piece.B);
            if (piece.Curve)
            {
                float bend = ay - 2 * piece.Qy + by;
                float turn = bend != 0 ? (ay - piece.Qy) / bend : -1f;
                if (turn > 0 && turn < 1)
                {
                    float v = 1 - turn;
                    float mx = v * v * ax + 2 * v * turn * piece.Qx + turn * turn * bx, my = v * v * ay + 2 * v * turn * piece.Qy + turn * turn * by;
                    parts[0] = new Part(ax, ay, mx, my, 0f, turn);
                    parts[1] = new Part(mx, my, bx, by, turn, 1f);
                    return 2;
                }
            }

            parts[0] = new Part(ax, ay, bx, by, 0f, 1f);
            return 1;
        }

        // The rows of pixel centres a part may cross, within gy0 to gy1.
        private static (int From, int To) Rows(in Part part, int gy0, int gy1)
        {
            float lo = MathF.Min(part.Y0, part.Y1), hi = MathF.Max(part.Y0, part.Y1);
            return (Math.Max(gy0, (int)MathF.Ceiling(lo + 0.5f)), Math.Min(gy1, (int)MathF.Ceiling(hi + 0.5f) - 1));
        }

        // Whether the part crosses the row at yc (its start's row counted, its end's not: each crossing once).
        private static bool Crosses(in Part part, float yc) => (part.Y0 <= yc && yc < part.Y1) || (part.Y1 <= yc && yc < part.Y0);

        // Where the part crosses the row at yc: on a line by proportion, on a curve by solving its quadratic in y.
        private float CrossingX(in Piece piece, in Part part, float yc)
        {
            if (!piece.Curve)
            {
                return part.X0 + (yc - part.Y0) * (part.X1 - part.X0) / (part.Y1 - part.Y0);
            }

            var c = _contours[piece.Contour];
            var (ax, ay) = Point(c, piece.A);
            var (bx, by) = Point(c, piece.B);
            double qa = ay - 2.0 * piece.Qy + by, qb = 2.0 * (piece.Qy - ay), qc = ay - (double)yc, t;
            if (Math.Abs(qa) < 1e-12)
            {
                t = qb != 0 ? -qc / qb : part.T0;
            }
            else
            {
                double root = Math.Sqrt(Math.Max(0, qb * qb - 4 * qa * qc));
                double q = -0.5 * (qb + (qb >= 0 ? root : -root));
                double first = q / qa, second = q != 0 ? qc / q : first;
                t = Outside(first, part.T0, part.T1) <= Outside(second, part.T0, part.T1) ? first : second;
            }

            t = Math.Clamp(t, part.T0, part.T1);
            double v = 1 - t;
            return (float)(v * v * ax + 2 * v * t * piece.Qx + t * t * bx);
        }

        private static double Outside(double t, double from, double to) => t < from ? from - t : t > to ? t - to : 0;

        private void Bad(float x, float y)
        {
            if (2 * (_badCount + 1) > _bad.Length)
            {
                Grow(ref _bad, 2 * _badCount, 4 * (_badCount + 1) + 16);
            }

            _bad[2 * _badCount] = x;
            _bad[2 * _badCount + 1] = y;
            _badCount++;
        }

        // Each piece whose boundary points come within the tolerance and a pixel of a misplaced centre fitted again, split
        // at least once; false when none can be split further.
        private bool Refit()
        {
            float reach2 = Square(_tolerance + 1f);
            var old = _pieces;
            int count = _pieceCount;
            _pieces = Rent<Piece>(count + 16);
            _pieceCount = 0;
            bool split = false;
            try
            {
                for (int p = 0; p < count; p++)
                {
                    var piece = old[p];
                    if (piece.B - piece.A > 1 && Near(piece, reach2))
                    {
                        FitSpan(piece.Contour, piece.A, piece.B, forced: true);
                        split = true;
                    }
                    else
                    {
                        Add(piece);
                    }
                }
            }
            finally
            {
                ArrayPool<Piece>.Shared.Return(old);
            }

            return split;
        }

        private bool Near(in Piece piece, float reach2)
        {
            var c = _contours[piece.Contour];
            for (int k = piece.A; k <= piece.B; k++)
            {
                var (px, py) = Point(c, k);
                for (int i = 0; i < _badCount; i++)
                {
                    if (Square(px - _bad[2 * i]) + Square(py - _bad[2 * i + 1]) <= reach2)
                    {
                        return true;
                    }
                }
            }

            return false;
        }

        // ------------------------------------------------------------------ the outline in font units

        // Each contour: a move to its first point, its pieces (the last line left to the close), a close.
        private GlyphOutline Emit(OutlinePlacement at)
        {
            int verbs = 0, points = 0;
            for (int p = 0; p < _pieceCount;)
            {
                int end = End(p);
                int last = !_pieces[end - 1].Curve && end - p > 1 ? end - 1 : end;
                verbs += 2 + (last - p);
                points += 2;
                for (int k = p; k < last; k++)
                {
                    points += _pieces[k].Curve ? 4 : 2;
                }

                p = end;
            }

            var verbList = new OutlineVerb[verbs];
            var pointList = new float[points];
            int v = 0, w = 0;
            for (int p = 0; p < _pieceCount;)
            {
                int end = End(p);
                int last = !_pieces[end - 1].Curve && end - p > 1 ? end - 1 : end;
                var c = _contours[_pieces[p].Contour];
                verbList[v++] = OutlineVerb.Move;
                Put(c, _pieces[p].A);
                for (int k = p; k < last; k++)
                {
                    var piece = _pieces[k];
                    if (piece.Curve)
                    {
                        verbList[v++] = OutlineVerb.Quadratic;
                        (pointList[w], pointList[w + 1]) = at.ToUnits(piece.Qx, piece.Qy);
                        w += 2;
                    }
                    else
                    {
                        verbList[v++] = OutlineVerb.Line;
                    }

                    Put(c, piece.B);
                }

                verbList[v++] = OutlineVerb.Close;
                p = end;
            }

            return new GlyphOutline(verbList, pointList);

            void Put(in Contour c, int i)
            {
                var (x, y) = Point(c, i);
                (pointList[w], pointList[w + 1]) = at.ToUnits(x, y);
                w += 2;
            }
        }

        // The end of the run of pieces of the contour piece p belongs to.
        private int End(int p)
        {
            int contour = _pieces[p].Contour, end = p;
            while (end < _pieceCount && _pieces[end].Contour == contour)
            {
                end++;
            }

            return end;
        }

        // ------------------------------------------------------------------ helpers

        private static float Square(float v) => v * v;

        // The squared distance from (px, py) to the segment from (ax, ay) to (bx, by).
        private static float SegmentDistance2(float px, float py, float ax, float ay, float bx, float by)
        {
            float dx = bx - ax, dy = by - ay, l2 = dx * dx + dy * dy;
            float u = l2 > 0 ? Math.Clamp(((px - ax) * dx + (py - ay) * dy) / l2, 0f, 1f) : 0f;
            return Square(px - ax - u * dx) + Square(py - ay - u * dy);
        }

        private static T[] Rent<T>(int length) => ArrayPool<T>.Shared.Rent(Math.Max(length, 16));

        private static void Return<T>(ref T[] array)
        {
            if (array.Length > 0)
            {
                ArrayPool<T>.Shared.Return(array);
            }

            array = [];
        }

        // The array grown to at least `length`, its first `used` values kept.
        private static void Grow<T>(ref T[] array, int used, int length)
        {
            var bigger = Rent<T>(length);
            array.AsSpan(0, used).CopyTo(bigger);
            Return(ref array);
            array = bigger;
        }
    }
}
