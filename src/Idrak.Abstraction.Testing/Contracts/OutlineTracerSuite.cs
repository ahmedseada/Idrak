// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Globalization;
using System.Text.Json.Nodes;
using Idrak.Abstraction.Data;

namespace Idrak.Abstraction.Testing;

/// <summary>
/// The checks of an outline tracer (<see cref="IOutlineTracer"/>, registered in <see cref="OutlineTracers"/>) on outlines
/// drawn here (squares, rings with one and two holes, discs of cubic and of quadratic curves, a dot over a stem, a
/// slanted stroke, a thin bar, random star shapes with and without a hole) and the outlines given, each filled into
/// coverage by an <see cref="IOutlineRasterizer"/> at several sizes and sub-pixel places, with the tracer's default
/// settings and the settings given:
/// <list type="bullet">
/// <item>the promised tolerance is a finite number of pixels, 0 or more;</item>
/// <item>coverage with no ink gives an empty outline;</item>
/// <item>the traced outline, filled again with the same placement, inks the same pixels to within the promised
/// tolerance and a pixel, both ways (nothing lost, nothing added);</item>
/// <item>it has one winding: every pixel centre lies inside contours of one direction at most once;</item>
/// <item>its holes are kept: the drawn holes' points lie outside it;</item>
/// <item>the same outline again and from several threads at once;</item>
/// <item>an option it does not take is refused, naming it.</item>
/// </list>
/// </summary>
public sealed class OutlineTracerSuite : ContractSuite<IOutlineTracer>
{
    // The shapes drawn here, in font units of a 1,000-unit em, by index.
    private static readonly string[] ShapeNames =
    [
        "a square", "a ring of straight sides (one hole)", "a disc of cubic curves", "a ring of curves (one hole)", "a dot over a stem", "a slanted stroke",
        "a disc of quadratic curves", "a box with two holes", "a thin bar",
    ];

    private const int OwnUnitsPerEm = 1000, Star = -1, Given = -2, Blank = -3, Margin = 4;

    private readonly IReadOnlyList<GlyphOutline> _given;
    private readonly int _givenUnitsPerEm;
    private readonly IOutlineRasterizer? _rasterizer;

    /// <summary>The checks, on the shapes drawn here and <paramref name="outlines"/>.</summary>
    /// <param name="outlines">Outlines of one's own to trace besides the suite's (a font's glyphs), in font units; empty ones are passed over.</param>
    /// <param name="unitsPerEm">The em of <paramref name="outlines"/>, in font units.</param>
    /// <param name="settings">Option sets to read the tracer's settings from besides its defaults (its own keys only).</param>
    /// <param name="rasterizer">What fills outlines into coverage; null for <see cref="OutlineRasterizers.Get"/>'s "coverage".</param>
    public OutlineTracerSuite(IReadOnlyList<GlyphOutline>? outlines = null, int unitsPerEm = OwnUnitsPerEm, IReadOnlyList<IReadOnlyDictionary<string, string>>? settings = null,
        IOutlineRasterizer? rasterizer = null)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(unitsPerEm);
        _given = [.. (outlines ?? []).Where(o => o is not null && !o.IsEmpty)];
        _givenUnitsPerEm = unitsPerEm;
        _rasterizer = rasterizer;
        Settings = [PageOptions.None, .. settings ?? []];
    }

    /// <summary>The option sets the tracer is checked with, its defaults first.</summary>
    public IReadOnlyList<IReadOnlyDictionary<string, string>> Settings { get; }

    /// <summary>The option no tracer takes, which each must refuse naming it.</summary>
    public const string UnknownOption = "outline-tracer-suite-unknown-option";

    /// <inheritdoc />
    public override string Name => "outline-tracer";

    /// <inheritdoc />
    public override IEnumerable<ContractCase> FixedCases()
    {
        for (int s = 0; s < Settings.Count; s++)
        {
            yield return Case("coverage with no ink", s, Blank, 48, 0, 0);
            for (int shape = 0; shape < ShapeNames.Length; shape++)
            {
                yield return Case(ShapeNames[shape], s, shape, 48, 0.25f, 0.5f);
            }

            yield return Case($"{ShapeNames[0]}, small", s, 0, 20, 0.5f, 0.75f);
            yield return Case($"{ShapeNames[3]}, small", s, 3, 24, 0.1f, 0.3f);
            yield return Case($"{ShapeNames[1]}, large", s, 1, 160, 0.6f, 0.2f);
            yield return Case($"{ShapeNames[6]}, large", s, 6, 140, 0.9f, 0.4f);
            for (int g = 0; g < _given.Count; g++)
            {
                yield return Case($"outline {g + 1} given", s, Given, 48, 0.3f, 0.6f, given: g);
                yield return Case($"outline {g + 1} given, large", s, Given, 112, 0.7f, 0.1f, given: g);
            }
        }
    }

    /// <inheritdoc />
    public override ContractCase RandomCase(Random random, bool large)
    {
        ArgumentNullException.ThrowIfNull(random);
        int s = random.Next(Settings.Count);
        float em = 16 + (float)random.NextDouble() * (large ? 240 : 96), dx = (float)random.NextDouble(), dy = (float)random.NextDouble();
        int kind = random.Next(ShapeNames.Length + 2 + (_given.Count > 0 ? 1 : 0));
        if (kind < ShapeNames.Length)
        {
            return Case($"{ShapeNames[kind]}, random size and place", s, kind, em, dx, dy);
        }

        if (kind == ShapeNames.Length + 2)
        {
            return Case("an outline given, random size and place", s, Given, em, dx, dy, given: random.Next(_given.Count));
        }

        // A star of 5 to 12 points around (400, 400), each point's radius 70% to 100% of the largest; with a hole, a
        // square turned the other way around the same centre, well inside the star.
        bool hole = kind == ShapeNames.Length + 1;
        int points = random.Next(5, 13);
        float outer = 250 + (float)random.NextDouble() * 150;
        var corners = new JsonArray();
        for (int k = 0; k < points; k++)
        {
            double angle = 2 * Math.PI * k / points;
            float radius = outer * (0.7f + 0.3f * (float)random.NextDouble());
            corners.Add((JsonNode)Round(400 + radius * PortableMath.Cos(angle)));
            corners.Add((JsonNode)Round(400 + radius * PortableMath.Sin(angle)));
        }

        // A hole at least a few pixels wide, so that ink around it is not only edge.
        var data = Case(hole ? "a random star with a hole" : "a random star", s, Star, hole ? MathF.Max(em, 48) : em, dx, dy).Data;
        data["corners"] = corners;
        data["hole"] = hole ? Round(outer * 0.25f * (0.8f + 0.4f * (float)random.NextDouble())) : 0;
        return new ContractCase(hole ? "a random star with a hole" : "a random star", data);
    }

    /// <inheritdoc />
    public override void Run(IOutlineTracer implementation, ContractCase @case, CaseChecks checks)
    {
        ArgumentNullException.ThrowIfNull(implementation);
        ArgumentNullException.ThrowIfNull(@case);
        ArgumentNullException.ThrowIfNull(checks);
        var data = @case.Data;
        var settings = implementation.Read(Settings[(int)data["settings"]!]);
        float tolerance = implementation.Tolerance(settings);
        checks.Check("the promised tolerance is a finite number of pixels, 0 or more", tolerance >= 0f && float.IsFinite(tolerance), $"tolerance {tolerance}");
        CheckRefusal(implementation, checks);

        int shape = (int)data["shape"]!;
        if (shape == Blank)
        {
            var blank = GlyphCoverage.Blank(32, 32);
            var placement = new OutlinePlacement(0.048f, 4f, 28f);
            var empty = implementation.Trace(blank, placement, settings);
            checks.Check("coverage with no ink gives an empty outline", empty.IsEmpty, $"{empty.Verbs.Length} steps traced from no ink, spanning {empty.Bounds()}");
            CheckAgain(implementation, blank, placement, settings, empty, checks);
            return;
        }

        var (outline, unitsPerEm, holes) = Outline(data);
        float scale = (float)data["em"]! / unitsPerEm;
        var (minX, minY, maxX, maxY) = outline.Bounds();
        int width = (int)MathF.Ceiling((maxX - minX) * scale) + 2 * Margin + 1, height = (int)MathF.Ceiling((maxY - minY) * scale) + 2 * Margin + 1;
        var at = new OutlinePlacement(scale, Margin - minX * scale + (float)data["dx"]!, Margin + maxY * scale + (float)data["dy"]!);
        var rasterizer = _rasterizer ?? OutlineRasterizers.Get();
        var coverage = Fill(rasterizer, outline, at, width, height);
        var traced = implementation.Trace(coverage, at, settings);

        // Filled again, the same pixels to within the tolerance (and a pixel for both fills' rounding), both ways.
        var redrawn = Fill(rasterizer, traced, at, width, height);
        int reach = (int)MathF.Ceiling(MathF.Min(tolerance, 1e6f)) + 1;
        bool[] drawn = Inked(coverage), back = Inked(redrawn);
        string? lost = Outside(drawn, Dilate(back, width, height, reach), width, "is ink in the coverage, but the traced outline inks nothing within");
        string? added = Outside(back, Dilate(drawn, width, height, reach), width, "is inked by the traced outline, but the coverage has no ink within");
        checks.Check("drawn and traced back within the promised tolerance", lost is null && added is null,
            string.Create(CultureInfo.InvariantCulture, $"{lost ?? added} {reach} pixels ({width} x {height} pixels, {data["em"]} px per em, tolerance {tolerance})"));

        // One winding: at every pixel centre the outline winds 0 times or once, the same way everywhere.
        checks.Compare("one winding", Windings(traced, at, width, height));
        if (holes.Count > 0)
        {
            var filled = holes.Where(h => Winding(traced, h.X, h.Y) != 0).Select(h => string.Create(CultureInfo.InvariantCulture, $"({h.X}, {h.Y}) winds {Winding(traced, h.X, h.Y)}")).ToList();
            checks.Check("holes kept", filled.Count == 0, $"the hole points {string.Join(", ", filled)} lie inside the traced outline");
        }
        else
        {
            checks.Skip("holes kept", "the shape has no hole");
        }

        CheckAgain(implementation, coverage, at, settings, traced, checks);
    }

    /// <inheritdoc />
    protected override string Describe(IOutlineTracer implementation) => $"{implementation.Name} ({implementation.GetType().Name})";

    // The same outline again, and from several threads at once.
    private static void CheckAgain(IOutlineTracer implementation, GlyphCoverage coverage, OutlinePlacement at, OutlineTracerSettings settings, GlyphOutline first, CaseChecks checks)
    {
        string? again = Different(first, implementation.Trace(coverage, at, settings));
        var parallel = new GlyphOutline[4];
        Parallel.For(0, parallel.Length, i => parallel[i] = implementation.Trace(coverage, at, settings));
        checks.Compare("the same outline again and from several threads", again ?? parallel.Select(o => Different(first, o)).FirstOrDefault(d => d is not null));
    }

    private static void CheckRefusal(IOutlineTracer implementation, CaseChecks checks)
    {
        try
        {
            implementation.Read(new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase) { [UnknownOption] = "1" });
            checks.Fail("an option it does not take is refused", $"'{UnknownOption}' was taken");
        }
        catch (ArgumentException e)
        {
            checks.Check("an option it does not take is refused", e.Message.Contains(UnknownOption, StringComparison.Ordinal), $"the refusal does not name the option: {e.Message}");
        }
    }

    private static string? Different(GlyphOutline expected, GlyphOutline actual) =>
        !expected.Verbs.Span.SequenceEqual(actual.Verbs.Span) ? $"{actual.Verbs.Length} steps, the first time {expected.Verbs.Length}"
        : !expected.Points.Span.SequenceEqual(actual.Points.Span) ? Comparisons.Difference(expected.Points.Span, actual.Points.Span, 0f) ?? "other points"
        : null;

    // The case's outline, its em and the points inside its holes.
    private (GlyphOutline Outline, int UnitsPerEm, IReadOnlyList<(float X, float Y)> Holes) Outline(JsonObject data)
    {
        int shape = (int)data["shape"]!;
        if (shape == Given)
        {
            return (_given[(int)data["given"]!], _givenUnitsPerEm, []);
        }

        var pen = new OutlinePen();
        if (shape == Star)
        {
            float[] corners = [.. ((JsonArray)data["corners"]!).Select(c => (float)c!)];
            pen.Polygon(corners, outer: true);
            float hole = (float)data["hole"]!;
            if (hole <= 0)
            {
                return (pen.Outline(), OwnUnitsPerEm, []);
            }

            pen.Polygon([400 - hole, 400 - hole, 400 + hole, 400 - hole, 400 + hole, 400 + hole, 400 - hole, 400 + hole], outer: false);
            return (pen.Outline(), OwnUnitsPerEm, [(400, 400)]);
        }

        (float X, float Y)[] holes = [];
        switch (shape)
        {
            case 0:
                pen.Rectangle(100, 0, 600, 500, outer: true);
                break;
            case 1:
                pen.Rectangle(100, 0, 700, 600, outer: true);
                pen.Rectangle(260, 160, 540, 440, outer: false);
                holes = [(400, 300)];
                break;
            case 2:
                pen.Ellipse(400, 300, 300, 300, outer: true);
                break;
            case 3:
                pen.Ellipse(400, 350, 330, 350, outer: true);
                pen.Ellipse(400, 350, 150, 170, outer: false);
                holes = [(400, 350)];
                break;
            case 4:
                pen.Rectangle(300, 0, 440, 520, outer: true);
                pen.Rectangle(300, 620, 440, 760, outer: true);
                break;
            case 5:
                pen.Polygon([100, 0, 260, 0, 620, 700, 460, 700], outer: true);
                break;
            case 6:
                pen.QuadraticCircle(350, 350, 300);
                break;
            case 7:
                pen.Rectangle(50, 0, 750, 700, outer: true);
                pen.Rectangle(150, 100, 350, 600, outer: false);
                pen.Rectangle(450, 100, 650, 600, outer: false);
                holes = [(250, 350), (550, 350)];
                break;
            default:
                pen.Rectangle(100, 0, 700, 100, outer: true);
                break;
        }

        return (pen.Outline(), OwnUnitsPerEm, holes);
    }

    // The outline filled into coverage of width x height, values kept within 0 to 1.
    private static GlyphCoverage Fill(IOutlineRasterizer rasterizer, GlyphOutline outline, OutlinePlacement at, int width, int height)
    {
        var values = new float[width * height];
        rasterizer.Fill(outline, at.Scale, at.X, at.Y, values, width, height);
        for (int i = 0; i < values.Length; i++)
        {
            values[i] = Math.Clamp(values[i], 0f, 1f);
        }

        return new GlyphCoverage(values, width, height);
    }

    // The pixels at least half inked.
    private static bool[] Inked(GlyphCoverage coverage)
    {
        var values = coverage.Values.Span;
        var inked = new bool[values.Length];
        for (int i = 0; i < values.Length; i++)
        {
            inked[i] = values[i] >= 0.5f;
        }

        return inked;
    }

    // The mask grown by reach pixels every way (a square), rows then columns, each from running counts.
    private static bool[] Dilate(bool[] mask, int width, int height, int reach)
    {
        var rows = new bool[mask.Length];
        var counts = new int[Math.Max(width, height) + 1];
        for (int y = 0; y < height; y++)
        {
            for (int x = 0; x < width; x++)
            {
                counts[x + 1] = counts[x] + (mask[y * width + x] ? 1 : 0);
            }

            for (int x = 0; x < width; x++)
            {
                rows[y * width + x] = counts[Math.Min(width, x + reach + 1)] - counts[Math.Max(0, x - reach)] > 0;
            }
        }

        var grown = new bool[mask.Length];
        for (int x = 0; x < width; x++)
        {
            for (int y = 0; y < height; y++)
            {
                counts[y + 1] = counts[y] + (rows[y * width + x] ? 1 : 0);
            }

            for (int y = 0; y < height; y++)
            {
                grown[y * width + x] = counts[Math.Min(height, y + reach + 1)] - counts[Math.Max(0, y - reach)] > 0;
            }
        }

        return grown;
    }

    // The first pixel set in mask and not in within, as "pixel (x, y) {what}", or null.
    private static string? Outside(bool[] mask, bool[] within, int width, string what)
    {
        for (int i = 0; i < mask.Length; i++)
        {
            if (mask[i] && !within[i])
            {
                return $"pixel ({i % width}, {i / width}) {what}";
            }
        }

        return null;
    }

    // Whether the outline winds 0 times or once at every pixel centre, the same way everywhere; null when it does.
    private static string? Windings(GlyphOutline outline, OutlinePlacement at, int width, int height)
    {
        var segments = Segments(outline, (x, y) => at.ToPixels(x, y));
        var crossings = new List<(float X, int Direction)>();
        int sign = 0;
        for (int row = 0; row < height; row++)
        {
            float yc = row + 0.5f;
            crossings.Clear();
            foreach (var (x0, y0, x1, y1) in segments)
            {
                if ((y0 <= yc && yc < y1) || (y1 <= yc && yc < y0))
                {
                    crossings.Add((x0 + (yc - y0) * (x1 - x0) / (y1 - y0), y1 > y0 ? 1 : -1));
                }
            }

            crossings.Sort((a, b) => a.X.CompareTo(b.X));
            int winding = 0, next = 0;
            for (int column = 0; column < width; column++)
            {
                float xc = column + 0.5f;
                while (next < crossings.Count && crossings[next].X < xc)
                {
                    winding += crossings[next++].Direction;
                }

                if (winding == 0)
                {
                    continue;
                }

                if (Math.Abs(winding) > 1)
                {
                    return $"pixel ({column}, {row}) lies inside {Math.Abs(winding)} contours turning the same way";
                }

                if (sign != 0 && winding != sign)
                {
                    return $"pixel ({column}, {row}) winds {winding}, other ink winds {sign}: contours of both directions fill";
                }

                sign = winding;
            }
        }

        return null;
    }

    // How many times the outline winds around the point (x, y), in font units.
    private static int Winding(GlyphOutline outline, float x, float y)
    {
        int winding = 0;
        foreach (var (x0, y0, x1, y1) in Segments(outline, (px, py) => (px, py)))
        {
            if (((y0 <= y && y < y1) || (y1 <= y && y < y0)) && x0 + (y - y0) * (x1 - x0) / (y1 - y0) < x)
            {
                winding += y1 > y0 ? 1 : -1;
            }
        }

        return winding;
    }

    // The outline as line segments (curves split into 16 lines each), every contour closed, its points mapped by map.
    private static List<(float X0, float Y0, float X1, float Y1)> Segments(GlyphOutline outline, Func<float, float, (float X, float Y)> map)
    {
        const int Steps = 16;
        var segments = new List<(float, float, float, float)>();
        var verbs = outline.Verbs.Span;
        var p = outline.Points.Span;
        int i = 0;
        (float X, float Y) start = default, current = default;
        bool open = false;
        void LineTo((float X, float Y) to)
        {
            segments.Add((current.X, current.Y, to.X, to.Y));
            current = to;
        }

        foreach (var verb in verbs)
        {
            switch (verb)
            {
                case OutlineVerb.Move:
                    if (open)
                    {
                        LineTo(start);
                    }

                    start = current = map(p[i], p[i + 1]);
                    open = true;
                    i += 2;
                    break;
                case OutlineVerb.Line:
                    LineTo(map(p[i], p[i + 1]));
                    i += 2;
                    break;
                case OutlineVerb.Quadratic:
                {
                    var (c, e) = (map(p[i], p[i + 1]), map(p[i + 2], p[i + 3]));
                    var s = current;
                    for (int k = 1; k <= Steps; k++)
                    {
                        float t = k / (float)Steps, u = 1 - t;
                        LineTo((u * u * s.X + 2 * u * t * c.X + t * t * e.X, u * u * s.Y + 2 * u * t * c.Y + t * t * e.Y));
                    }

                    i += 4;
                    break;
                }

                case OutlineVerb.Cubic:
                {
                    var (c1, c2, e) = (map(p[i], p[i + 1]), map(p[i + 2], p[i + 3]), map(p[i + 4], p[i + 5]));
                    var s = current;
                    for (int k = 1; k <= Steps; k++)
                    {
                        float t = k / (float)Steps, u = 1 - t;
                        float a = u * u * u, b = 3 * u * u * t, c = 3 * u * t * t, d = t * t * t;
                        LineTo((a * s.X + b * c1.X + c * c2.X + d * e.X, a * s.Y + b * c1.Y + c * c2.Y + d * e.Y));
                    }

                    i += 6;
                    break;
                }

                default:
                    if (open)
                    {
                        LineTo(start);
                        open = false;
                    }

                    break;
            }
        }

        if (open)
        {
            LineTo(start);
        }

        return segments;
    }

    private static float Round(double value) => MathF.Round((float)value, 2);

    private static ContractCase Case(string name, int settings, int shape, float em, float dx, float dy, int given = -1) =>
        new(name, new JsonObject { ["settings"] = settings, ["shape"] = shape, ["em"] = em, ["dx"] = dx, ["dy"] = dy, ["given"] = given });

    // Builds outlines in font units (y up): outer contours clockwise, holes counter-clockwise.
    private sealed class OutlinePen
    {
        // 4/3 (sqrt 2 - 1): a quarter circle's cubic control distance, as a share of the radius.
        private const float Kappa = 0.5522848f;

        private readonly List<OutlineVerb> _verbs = [];
        private readonly List<float> _points = [];

        public GlyphOutline Outline() => new([.. _verbs], [.. _points]);

        public void Rectangle(float x0, float y0, float x1, float y1, bool outer) =>
            Polygon(outer ? [x0, y1, x1, y1, x1, y0, x0, y0] : [x0, y0, x1, y0, x1, y1, x0, y1], outer: outer);

        // A polygon of x, y pairs, turned the way it should: clockwise when outer, else counter-clockwise.
        public void Polygon(float[] xy, bool outer)
        {
            double area = 0;
            int n = xy.Length / 2;
            for (int k = 0; k < n; k++)
            {
                int j = (k + 1) % n;
                area += (double)xy[2 * k] * xy[2 * j + 1] - (double)xy[2 * j] * xy[2 * k + 1];
            }

            bool clockwise = area < 0;
            Add(OutlineVerb.Move, xy[0], xy[1]);
            for (int k = 1; k < n; k++)
            {
                int index = clockwise == outer ? k : n - k;
                Add(OutlineVerb.Line, xy[2 * index], xy[2 * index + 1]);
            }

            _verbs.Add(OutlineVerb.Close);
        }

        // An ellipse of four cubic quarters, clockwise from the top when outer, else counter-clockwise.
        public void Ellipse(float cx, float cy, float rx, float ry, bool outer)
        {
            float kx = Kappa * rx, ky = Kappa * ry;
            Add(OutlineVerb.Move, cx, cy + ry);
            if (outer)
            {
                Add(OutlineVerb.Cubic, cx + kx, cy + ry, cx + rx, cy + ky, cx + rx, cy);
                Add(OutlineVerb.Cubic, cx + rx, cy - ky, cx + kx, cy - ry, cx, cy - ry);
                Add(OutlineVerb.Cubic, cx - kx, cy - ry, cx - rx, cy - ky, cx - rx, cy);
                Add(OutlineVerb.Cubic, cx - rx, cy + ky, cx - kx, cy + ry, cx, cy + ry);
            }
            else
            {
                Add(OutlineVerb.Cubic, cx - kx, cy + ry, cx - rx, cy + ky, cx - rx, cy);
                Add(OutlineVerb.Cubic, cx - rx, cy - ky, cx - kx, cy - ry, cx, cy - ry);
                Add(OutlineVerb.Cubic, cx + kx, cy - ry, cx + rx, cy - ky, cx + rx, cy);
                Add(OutlineVerb.Cubic, cx + rx, cy + ky, cx + kx, cy + ry, cx, cy + ry);
            }

            _verbs.Add(OutlineVerb.Close);
        }

        // A circle of eight quadratic arcs, clockwise.
        public void QuadraticCircle(float cx, float cy, float r)
        {
            float reach = r / (float)PortableMath.Cos(Math.PI / 8);
            Add(OutlineVerb.Move, cx + r, cy);
            for (int k = 1; k <= 8; k++)
            {
                double end = -k * Math.PI / 4, middle = end + Math.PI / 8;
                Add(OutlineVerb.Quadratic, cx + reach * (float)PortableMath.Cos(middle), cy + reach * (float)PortableMath.Sin(middle),
                    cx + r * (float)PortableMath.Cos(end), cy + r * (float)PortableMath.Sin(end));
            }

            _verbs.Add(OutlineVerb.Close);
        }

        private void Add(OutlineVerb verb, params ReadOnlySpan<float> points)
        {
            _verbs.Add(verb);
            foreach (float v in points)
            {
                _points.Add(v);
            }
        }
    }
}
